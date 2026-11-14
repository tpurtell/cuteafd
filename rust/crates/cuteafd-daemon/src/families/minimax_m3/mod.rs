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
mod owners;
mod backend;
pub(crate) mod startup;
pub(crate) mod serve;
pub(crate) mod admission;
pub(crate) mod golden;

/// Select MiniMax's native API contract before allocating device storage.
pub(crate) fn encoding(snapshot: &std::path::Path) -> anyhow::Result<cuteafd_api::openai::ModelEncoding> {
    Ok(cuteafd_api::openai::ModelEncoding::MiniMax(std::sync::Arc::new(
        cuteafd_api::openai::chat::minimax::MiniMaxEncoding::from_snapshot(snapshot)?,
    )))
}
