//! Hy4 (`hy_v4_vl`, `HYV4VLForConditionalGeneration`) checkpoint facts,
//! IndexCache schedule, packed coordinator weight arenas, and pages-only
//! execution layouts (#2550).
pub mod admission;
pub mod config;
pub mod execution;
pub mod weight_layout;

pub use super::hy4_common::{Hy4DflyConfig, Hy4ExpertStorage, Hy4KvCache, Hy4VisionConfig};
pub use config::{Hy4Config, Hy4IndexerKind, IndexCacheSchedule};
