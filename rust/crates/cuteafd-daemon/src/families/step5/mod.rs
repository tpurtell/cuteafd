//! Native Step-5 storage owners; execution remains unavailable until admitted.
mod admission;
mod backend;
mod cache;
mod chain;
mod routed;
mod csa;
mod ffn;
mod execution;
pub(crate) mod golden;
pub(crate) mod serve;
mod head;
mod mtp;
#[cfg(test)]
mod norm_accuracy;
mod prefix;
mod rope;
mod swa;
mod scratch;
mod transaction;
mod weights;
