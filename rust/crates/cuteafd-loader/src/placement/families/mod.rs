//! Per-family request builders (`FamilyPlacement` in the design). Each
//! family's planner and runtime admission build their request here.
//!
//! Each family also states what its executor runs today ([`ExecutorModes`]).
//! The solver never picks anything else, and the engine checks the
//! placement it is handed against the same set. Families not yet on the
//! solver state theirs now, so their solver ports (P6, P8-P10, P12) start
//! from the executor's real limits.
pub mod deepseek_v4;
pub mod deepseek_v41;
pub mod glm5;
pub mod glm5_flash;
pub mod qwen4;
pub mod qwen3_5;
pub mod mimo_v2;

use super::{AttentionPlacement, ExecutorModes, FfnMode, LayerMode};

const WHOLE0: LayerMode = LayerMode::Whole { gpu: 0, ffn: FfnMode::Owner };
const WHOLE1: LayerMode = LayerMode::Whole { gpu: 1, ffn: FfnMode::Owner };

/// V4 Flash/Pro: every layer head-split on two GPUs, or whole on GPU0; under
/// `context` the C4 layers token-split their compressed records (#58; C128,
/// window rings and carry stay replicated). Its only hop is the entry
/// broadcast of the embedding rows into GPU1. Per-layer ownership (`Whole{gpu}`
/// mixes, ranges) waits for its per-layer executor (P7). `context` is the
/// default (TJ 2026-10-18: no context-scaling KV replicated); `heads` stays an explicit option.
pub const DEEPSEEK_V4: ExecutorModes = ExecutorModes { family: "deepseek_v4",
    modes: &[LayerMode::HeadSplit, LayerMode::ContextSplit, WHOLE0], hops: false, attention: AttentionPlacement::Context };
/// V4.1: layer ranges, layers 0-19 / 20-39 whole on each GPU with one boundary
/// hop (`DirectInputHop`) and the owner-reduce FFN; direct-input storage is V4.1-only today.
pub const DEEPSEEK_V41: ExecutorModes = ExecutorModes { family: "deepseek_v41", modes: &[WHOLE0, WHOLE1], hops: true, attention: AttentionPlacement::Layers };
/// GLM 5.3: every layer split over two GPUs, by heads (latent replicated) or
/// by context (heads for the projections, KV pages owned by parity; K3).
/// `context` is its recorded default (K3, TJ: GLM 5.3 must not replicate KV; pool first
/// is not C1-gated): at max it admits 2,097,152 tokens against heads' 1,321,216, C8 1.015x,
/// 8K prefill 1.027x; C1 parity after the per-row-bucket split plans (7689c8c3: 1.043x). A build without the context programs
/// fails a chosen `context` closed (`glm5::solve_working_set`, #202); `heads` stays an
/// explicit option.
pub const GLM5: ExecutorModes = ExecutorModes { family: "glm5",
    modes: &[LayerMode::HeadSplit, LayerMode::ContextSplit, WHOLE0,
        LayerMode::Whole { gpu: 0, ffn: FfnMode::Split },
        LayerMode::Whole { gpu: 1, ffn: FfnMode::Split }], hops: true, attention: AttentionPlacement::Context };
/// GLM Flash: every layer split over two GPUs; its MLA layers by heads (latent replicated) or by
/// context (projections head-split, 256-token units owned by parity; #57), KDA always by heads.
/// `context` is the recorded default on two GPUs (TJ: no context-scaling KV replicated; pool
/// first); a build without its programs fails a chosen `context` closed
/// (`glm5_flash::resolve_attention`, #202), and `heads` stays an explicit option. Owned MLA
/// layers (`layers`) are an explicit per-layer option with split FFNs; auto stays context.
pub const GLM5_FLASH: ExecutorModes = ExecutorModes { family: "glm5_flash",
    modes: &[LayerMode::HeadSplit, LayerMode::ContextSplit, WHOLE0,
        LayerMode::Whole { gpu: 0, ffn: FfnMode::Split },
        LayerMode::Whole { gpu: 1, ffn: FfnMode::Split }], hops: true, attention: AttentionPlacement::Context };
/// MiMo: `attach_peer` requires every layer split.
pub const MIMO_V2: ExecutorModes = ExecutorModes { family: "mimo_v2", modes: &[LayerMode::HeadSplit, WHOLE0],
    hops: false, attention: AttentionPlacement::Heads };
/// Qwen: qualified whole-width contiguous layer ranges on a Spark-free pair.
pub const QWEN4: ExecutorModes = ExecutorModes { family: "qwen4", modes: &[WHOLE0, WHOLE1],
    hops: true, attention: AttentionPlacement::Layers };

/// Kimi: TP2 KDA, alternating whole MLA owners and split dense/shared FFNs.
pub const KIMI_K3: ExecutorModes = ExecutorModes { family: "kimi_k3",
    modes: &[LayerMode::HeadSplit,
        LayerMode::Whole { gpu: 0, ffn: FfnMode::Split },
        LayerMode::Whole { gpu: 1, ffn: FfnMode::Split }],
    hops: true, attention: AttentionPlacement::Layers };

/// Every family's executor set, for lookups by family name.
pub const EXECUTORS: [ExecutorModes; 7] = [DEEPSEEK_V4, DEEPSEEK_V41, GLM5, GLM5_FLASH, MIMO_V2, QWEN4, KIMI_K3];

pub fn executor(family: &str) -> Option<ExecutorModes> {
    EXECUTORS.into_iter().find(|e| e.family == family)
}
