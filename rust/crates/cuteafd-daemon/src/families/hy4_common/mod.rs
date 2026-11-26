//! Shared Hy4 family daemon primitives (`hy4` and `hy4_flash`).
//!
//! Lifts the genuinely identical compact-core native linear owner, absorbed
//! KV-B / dense weight loading helpers, and golden scoring helpers so `hy4`
//! and `hy4_flash` share one implementation without copy-paste.

pub(crate) mod golden;
pub(crate) mod native;
pub(crate) mod weights;
