//! Vendor MiniMax target operands. Serving is not yet wired.
pub(crate) mod weights;
pub(crate) mod dense;
pub(crate) mod attention;
pub(crate) mod layer;
mod experts;
pub(crate) mod cache;
pub(crate) mod prefix;
pub(crate) mod prefill;
pub(crate) mod head;
mod graphs;
mod engine;
pub(crate) mod admission;
pub(crate) mod golden;
