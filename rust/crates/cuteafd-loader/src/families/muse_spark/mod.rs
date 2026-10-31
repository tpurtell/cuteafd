//! Muse Spark (`muse_spark` wrapping `text_config.model_type=muse_spark_text`) checkpoint readers.
pub mod config;
pub mod cache;
pub mod core;
pub mod reference;
pub mod weight_layout;

pub use config::{MuseAttention, MuseSparkConfig};

/// The checkpoint's compiled context (max_position_embeddings).
pub const COMPILED_CONTEXT: usize = 1 << 20;
