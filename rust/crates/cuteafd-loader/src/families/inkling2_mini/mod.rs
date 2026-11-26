//! Inkling 2 Mini (`Inkling2LoopForConditionalGeneration`, `model_type=inkling2_loop`):
//! looped 24-layer transformer (prelude 0..6, core 6..18 looped 3 passes, coda 18..24 ->
//! 48 visits with 48 distinct KV + short-conv cache slots), native NVFP4/MXFP8 MoE,
//! bundled 3-layer recurrent dSpark drafter, and bundled 23-layer PE-Lang ViT (#2701).
pub mod admission;
pub mod config;
pub mod execution;
pub mod dflash;
pub mod weight_layout;

pub use crate::families::inkling2::{Inkling2MtpConfig, Inkling2VisionConfig, InklingKvCache};
pub use config::{Inkling2LoopConfig, Inkling2MiniConfig, LoopVisit};
