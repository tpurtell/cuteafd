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
    pub lanes: u64,
}

impl ContextBuffers {
    /// PLAN attention placement sections 1/2: two staging parity slots and
    /// q/candidates/partial receive slots per parity and lane, at decode rows.
    pub fn demands(self) -> Result<Vec<Demand>, PlacementError> {
        let overflow = || PlacementError::Overflow("context buffers");
        let units = self.compiled_extent.div_ceil(self.staging_unit_rows.max(1));
        let staging = self.staging_unit_bytes.checked_mul(units)
            .and_then(|n| n.checked_mul(2)).ok_or_else(overflow)?;
        let payload = self.query_row_bytes.checked_add(self.partial_row_bytes)
            .and_then(|n| n.checked_add(self.candidate_row_bytes)).ok_or_else(overflow)?;
        let exchange = payload.checked_mul(self.decode_rows).and_then(|n| n.checked_mul(self.lanes))
            .and_then(|n| n.checked_mul(2)).ok_or_else(overflow)?;
        Ok((0..2).flat_map(|gpu| [
            Demand::new(gpu, Category::Workspace, "context staging", staging, Basis::Formula),
            Demand::new(gpu, Category::Transport, "context exchange", exchange, Basis::Formula),
        ]).collect())
    }
}

impl ExecutorModes {
    /// Recorded family default. No decision gate has enabled another mode yet.
    pub fn attention_default(&self) -> AttentionPlacement { AttentionPlacement::Heads }

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
            let needed = match mode {
                AttentionPlacement::Context => &[LayerMode::ContextSplit][..],
                AttentionPlacement::Layers => &[
                    LayerMode::Whole { gpu: 0, ffn: FfnMode::Split },
                    LayerMode::Whole { gpu: 1, ffn: FfnMode::Split },
                ][..],
                AttentionPlacement::Heads => unreachable!(),
            };
            if needed.iter().any(|m| !self.runs(*m)) {
                return Err(PlacementError::AttentionPlacement { family: self.family, mode,
                    reason: "executor not implemented; only heads attention placement is qualified" });
            }
        }
        Ok(mode)
    }
}
