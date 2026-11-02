//! Muse Spark (`muse_spark`, text `muse_spark_text`): the layer chain, its CPU
//! oracle (checked against the official transformers modules) and, next, the
//! coordinator backend on the shared lane/prefill/decode design.
#![allow(dead_code)]
pub(crate) mod admission;
pub(crate) mod backend;
pub(crate) mod block;
pub(crate) mod core;
pub(crate) mod device;
pub(crate) mod engine;
pub(crate) mod experts;
pub(crate) mod forward;
pub(crate) mod lane;
pub(crate) mod onboard;
pub(crate) mod draft;
pub(crate) mod peer;
pub(crate) mod prefill;
pub(crate) mod prefix;
pub(crate) mod programs;
pub(crate) mod serve;
pub(crate) mod spark;
pub(crate) mod startup;
pub(crate) mod storage;
pub(crate) mod step;
pub(crate) mod weights;
#[cfg(test)]
pub(crate) mod reference;
#[cfg(test)]
mod reference_tests;
#[cfg(test)]
mod drafter_tests;
