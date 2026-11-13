//! Inkling 2 Small checkpoint facts. Execution uses the shared engine contracts.
pub mod admission;
pub mod config;
pub mod experts;
pub mod weight_layout;
pub use config::{Inkling2Config, Inkling2MtpConfig, Inkling2VisionConfig};
