//! Per-family request builders (`FamilyPlacement` in the design). Each
//! family's planner and runtime admission build their request here.
//!
//! Each family also states what its executor runs today ([`ExecutorModes`]).
//! The solver never picks anything else, and the engine checks the
//! placement it is handed against the same set. Families not yet on the
//! solver state theirs now, so their solver ports (P6, P8-P10, P12) start
//! from the executor's real limits.
pub mod deepseek_v4;
pub mod qwen4;

use super::{ExecutorModes, FfnMode, LayerMode};

const WHOLE0: LayerMode = LayerMode::Whole { gpu: 0, ffn: FfnMode::Owner };
const WHOLE1: LayerMode = LayerMode::Whole { gpu: 1, ffn: FfnMode::Owner };

/// V4 Flash/Pro: every layer head-split on two GPUs, or whole on GPU0. Its
/// only hop is the entry broadcast of the embedding rows into GPU1.
/// Per-layer ownership (`Whole{gpu}` mixes, ranges) waits for its per-layer
/// executor (P7).
pub const DEEPSEEK_V4: ExecutorModes = ExecutorModes { family: "deepseek_v4", modes: &[LayerMode::HeadSplit, WHOLE0],
    hops: false };
/// V4.1: layer ranges, layers 0-19 / 20-39 whole on each GPU with one boundary
/// hop (`BlockTransfer`) and the owner-reduce FFN.
pub const DEEPSEEK_V41: ExecutorModes = ExecutorModes { family: "deepseek_v41", modes: &[WHOLE0, WHOLE1], hops: true };
/// GLM 5.3, GLM Flash and MiMo: `attach_peer` requires every layer split.
pub const GLM5: ExecutorModes = ExecutorModes { family: "glm5", modes: &[LayerMode::HeadSplit, WHOLE0], hops: false };
pub const GLM5_FLASH: ExecutorModes = ExecutorModes { family: "glm5_flash", modes: &[LayerMode::HeadSplit, WHOLE0],
    hops: false };
pub const MIMO_V2: ExecutorModes = ExecutorModes { family: "mimo_v2", modes: &[LayerMode::HeadSplit, WHOLE0],
    hops: false };
/// Qwen: one GPU.
pub const QWEN4: ExecutorModes = ExecutorModes { family: "qwen4", modes: &[WHOLE0], hops: false };

/// Every family's executor set, for lookups by family name.
pub const EXECUTORS: [ExecutorModes; 6] = [DEEPSEEK_V4, DEEPSEEK_V41, GLM5, GLM5_FLASH, MIMO_V2, QWEN4];

pub fn executor(family: &str) -> Option<ExecutorModes> {
    EXECUTORS.into_iter().find(|e| e.family == family)
}
