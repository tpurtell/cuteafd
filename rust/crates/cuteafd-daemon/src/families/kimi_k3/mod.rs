//! Kimi K3 (`kimi_k3`, text `kimi_linear`): the layer chain, its CPU oracle and
//! the coordinator backend's operation table. Serving binds the W4-W8 exports
//! into `device`; see #273 for the work items.
#![allow(dead_code)]
pub(crate) mod admission;
pub(crate) mod block;
pub(crate) mod device;
pub(crate) mod engine;
pub(crate) mod golden;
pub(crate) mod serve;
pub(crate) mod startup;
pub(crate) mod graphs;
pub(crate) mod backend;
pub(crate) mod layers;
pub(crate) mod prefix;
pub(crate) mod spark;
pub(crate) mod mla;
pub(crate) mod state;
pub(crate) mod weights;
#[cfg(test)]
pub(crate) mod reference;
