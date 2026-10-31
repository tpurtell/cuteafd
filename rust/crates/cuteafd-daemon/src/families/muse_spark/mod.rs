//! Muse Spark (`muse_spark`, text `muse_spark_text`): the layer chain, its CPU
//! oracle (checked against the official transformers modules) and, next, the
//! coordinator backend on the shared lane/prefill/decode design.
#![allow(dead_code)]
pub(crate) mod block;
pub(crate) mod core;
pub(crate) mod programs;
pub(crate) mod serve;
pub(crate) mod startup;
pub(crate) mod step;
pub(crate) mod weights;
#[cfg(test)]
pub(crate) mod reference;
#[cfg(test)]
mod reference_tests;
#[cfg(test)]
mod drafter_tests;
