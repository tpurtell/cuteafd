//! Routed-expert pieces every family uses: layer identity and native resident
//! weights, per-wave execution and the host exchange, the EXL3 package path,
//! the expertd-native service, the exact FP8 expert packages and the TP2 RTX
//! expert halves (`rtx`).

pub(crate) mod execution;
pub(crate) mod exl3;
pub(crate) mod fp8;
pub(crate) mod layer;
pub(crate) mod paired_load;
pub(crate) mod rtx;
pub(crate) mod service;
