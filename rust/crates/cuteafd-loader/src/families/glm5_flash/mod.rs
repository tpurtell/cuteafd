//! GLM 5.3 Flash (glm5_next) family: configuration for the generic engine.
pub mod config;
pub mod core;
pub mod core_admission;
pub mod resident;
pub use config::{GlmNextAttention, GlmNextConfig};

/// The same projection contract drives native loading and offline checkpoint stitching.
/// KDA and the vocabulary head retain their checkpoint BF16 representation.
pub fn core_fp8_projections(kind: &str) -> Vec<String> {
    let contract: std::collections::HashMap<String, Vec<String>> = serde_json::from_str(
        include_str!("core-fp8-projections.json")).expect("checked-in FP8 projection contract");
    contract.get(kind).expect("known FP8 projection kind").clone()
}
