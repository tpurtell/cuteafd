//! Hy4 (`hy_v4_vl`, `HYV4VLForConditionalGeneration`) coordinator family
//! runtime over the shared engine (#2550):
//! - 78-layer uniform gated DSA MLA (64H TP1 / 32H TP2, `q_lora_rank=2048`,
//!   `qa_kva_eps=1e-6`, `kv_lora_rank=512`, `qk=192+64`, `v=256`, F32 sinks,
//!   sigmoid output gate, adjacent-pair RoPE),
//! - 4-stream iHC (H=6144, FP32 mixing, no sandwich norms),
//! - 32-head per-token DSA IndexCache (21 full layers `0, 1, 5, ..., 77`;
//!   57 shared layers reuse the preceding full layer's per-wave top-2048),
//! - layer 0 dense FFN (`I=18432`, unclamped SiLU), layers 1..=77 direct MoE
//!   (256 experts top-8, `H=6144`, `I=2048`, `swiglu_limit=10.0`, 1 shared expert),
//! - pages-only exact prefix cache (`MarkStore::None`), rank-local CUDA graphs,
//!   1x RTX and 2x RTX TP2 (`ContextSplit` default + `heads` opt-in).
pub(crate) mod backend;
pub(crate) mod cache;
pub(crate) mod draft;
pub(crate) mod execution;
pub(crate) mod golden;
pub(crate) mod graph_sites;
pub(crate) mod graphs;
pub(crate) mod programs;
pub(crate) mod serve;
pub(crate) mod weights;
