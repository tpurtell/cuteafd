//! Plan-only attention placement and the buffers required by context split.
use super::*;
use std::{fmt, str::FromStr};

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AttentionPlacement {
    #[default]
    Heads,
    Context,
    Layers,
}

impl fmt::Display for AttentionPlacement {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self { Self::Heads => "heads", Self::Context => "context", Self::Layers => "layers" })
    }
}

impl FromStr for AttentionPlacement {
    type Err = String;
    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            "heads" => Ok(Self::Heads), "context" => Ok(Self::Context), "layers" => Ok(Self::Layers),
            _ => Err(format!("attention placement must be heads, context or layers (got {value:?})")),
        }
    }
}

/// `None` is auto; only auto may use the solver's memory lever.
pub fn parse(value: &str) -> Result<Option<AttentionPlacement>, String> {
    if value == "auto" { Ok(None) } else { value.parse().map(Some) }
}

/// Pool bytes per logical unit. A context unit is charged at half its bytes
/// on each GPU; the solver reserves an even number of units for the half pools.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct KvDemand {
    pub unit_bytes_whole: u64,
    pub unit_bytes_split: [u64; 2],
    /// None: this layer stays head-split (V4 C128/window, GLM Flash KDA).
    pub unit_bytes_context: Option<[u64; 2]>,
}

impl From<ModeBytes> for KvDemand {
    fn from(bytes: ModeBytes) -> Self {
        Self { unit_bytes_whole: bytes.whole, unit_bytes_split: bytes.split, unit_bytes_context: None }
    }
}

/// New context exchange payloads per row, each way. Existing attention/FFN
/// all-reduces are unchanged and are not counted here.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ContextBuffers {
    /// Largest context layer's gathered records + keys, per logical unit.
    pub staging_unit_bytes: u64,
    /// Source tokens per staging unit (0 is treated as 1 for empty defaults).
    pub staging_unit_rows: u64,
    pub query_row_bytes: u64,
    pub partial_row_bytes: u64,
    pub candidate_row_bytes: u64,
    pub compiled_extent: u64,
    pub decode_rows: u64,
    /// Decode lanes that exchange concurrently (the gather-route prefill
    /// never uses the context exchange).
    pub lanes: u64,
}

/// One GPU's context exchange storage, per (layer parity, lane) slot:
/// the assembled query `[rows, 2 halves, q]` and candidate lists `[rows, 2K]`
/// (own half written locally, the peer's half pushed in, both read in place
/// by the partial and merge kernels), the peer's partial of this GPU's heads,
/// and three flags of 16 B each. Shared by the solver's demand and the
/// runtime allocation (`shared/peer_split/context.rs`), so they agree to the
/// byte.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ContextExchangeLayout {
    pub slots: u64,
    pub rows: u64,
    pub query_bytes: u64,
    pub candidate_bytes: u64,
    pub partial_bytes: u64,
}

/// Exchanges per slot: q, candidates, partial.
pub const CONTEXT_EXCHANGES: u64 = 3;

impl ContextExchangeLayout {
    /// Two layer parities per lane.
    pub fn new(buffers: &ContextBuffers) -> Self {
        Self { slots: 2 * buffers.lanes, rows: buffers.decode_rows, query_bytes: buffers.query_row_bytes,
            candidate_bytes: buffers.candidate_row_bytes, partial_bytes: buffers.partial_row_bytes }
    }
    /// Assembled query rows: both GPUs' halves.
    pub fn query_row(&self) -> Option<u64> { self.query_bytes.checked_mul(2) }
    /// Assembled candidate rows: both GPUs' lists.
    pub fn candidate_row(&self) -> Option<u64> { self.candidate_bytes.checked_mul(2) }
    /// Receive bytes of one slot (query, candidates, partial), before flags.
    pub fn slot_bytes(&self) -> Option<[u64; 3]> {
        Some([self.rows.checked_mul(self.query_row()?)?, self.rows.checked_mul(self.candidate_row()?)?,
            self.rows.checked_mul(self.partial_bytes)?])
    }
    /// Flag words: four u32 per exchange per slot (sequence, send state, recv state).
    pub fn control_bytes(&self) -> Option<u64> {
        self.slots.checked_mul(CONTEXT_EXCHANGES)?.checked_mul(16)
    }
    /// Everything one GPU allocates for the exchange.
    pub fn bytes(&self) -> Option<u64> {
        let [q, c, p] = self.slot_bytes()?;
        q.checked_add(c)?.checked_add(p)?.checked_mul(self.slots)?.checked_add(self.control_bytes()?)
    }
}

impl ContextBuffers {
    /// PLAN attention placement sections 1/2: two staging parity slots and the
    /// context exchange exactly as it allocates ([`ContextExchangeLayout`]).
    pub fn demands(self) -> Result<Vec<Demand>, PlacementError> {
        let overflow = || PlacementError::Overflow("context buffers");
        let units = self.compiled_extent.div_ceil(self.staging_unit_rows.max(1));
        let staging = self.staging_unit_bytes.checked_mul(units)
            .and_then(|n| n.checked_mul(2)).ok_or_else(overflow)?;
        let exchange = ContextExchangeLayout::new(&self).bytes().ok_or_else(overflow)?;
        Ok((0..2).flat_map(|gpu| [
            Demand::new(gpu, Category::Workspace, "context staging", staging, Basis::Formula),
            Demand::new(gpu, Category::Transport, "context exchange", exchange, Basis::Formula),
        ]).collect())
    }
}

impl ExecutorModes {
    /// Recorded family default. No decision gate has enabled another mode yet.
    pub fn attention_default(&self) -> AttentionPlacement { AttentionPlacement::Heads }

    /// Prefer split FFN ownership when available, otherwise owner reduction.
    pub fn whole_mode(&self, gpu: u8) -> Option<LayerMode> {
        [FfnMode::Split, FfnMode::Owner].into_iter()
            .map(|ffn| LayerMode::Whole { gpu, ffn }).find(|&mode| self.runs(mode))
    }

    /// Unlike the legacy layer-range modes, the new selectors are strict and
    /// never silently become heads or a one-GPU layout.
    pub fn check_attention(&self, requested: Option<AttentionPlacement>, gpus: usize, peer: bool)
        -> Result<AttentionPlacement, PlacementError> {
        let mode = requested.unwrap_or_else(|| self.attention_default());
        if mode != AttentionPlacement::Heads {
            if gpus != 2 || !peer {
                return Err(PlacementError::AttentionPlacement { family: self.family, mode,
                    reason: "requires two coordinator GPUs with peer access" });
            }
            let supported = match mode {
                AttentionPlacement::Context => self.runs(LayerMode::ContextSplit),
                AttentionPlacement::Layers => (0..2).all(|gpu| self.whole_mode(gpu).is_some()),
                AttentionPlacement::Heads => unreachable!(),
            };
            if !supported {
                return Err(PlacementError::AttentionPlacement { family: self.family, mode,
                    reason: "executor not implemented; only heads attention placement is qualified" });
            }
        }
        Ok(mode)
    }
}
