//! Always-on graph-capture counters, owned by the shared bank.
//!
//! The graph bank is shared code (`docs/architecture/shared-engine.md` §1): every
//! graph-using family captures through it, so its counters live here beside the bank
//! rather than in one family's scheduler. A family publishes them; it does not own them.
//!
//! One relaxed add per capture and one relaxed load per publish. No allocation, no lock,
//! no clock read the capture did not already take.

use std::sync::atomic::{AtomicU64, Ordering};

/// One process's graph-capture counters. Serving uses [`GLOBAL`]; a test owns its own, so
/// its assertions never see another test's captures.
pub(crate) struct GraphStats {
    /// Time inside graph instantiation (instantiate and upload) of the captures that
    /// reached it, failed ones included.
    graph_instantiate_ns: AtomicU64,
}

pub(crate) static GLOBAL: GraphStats = GraphStats::new();

impl GraphStats {
    pub(crate) const fn new() -> Self {
        Self { graph_instantiate_ns: AtomicU64::new(0) }
    }

    /// One capture's instantiation and upload, counted whether or not it succeeds.
    pub(crate) fn record_graph_instantiate(&self, ns: u64) {
        self.graph_instantiate_ns.fetch_add(ns, Ordering::Relaxed);
    }

    /// Nanoseconds spent instantiating graphs since startup.
    pub(crate) fn graph_instantiate_ns(&self) -> u64 {
        self.graph_instantiate_ns.load(Ordering::Relaxed)
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn graph_instantiate_sums_every_capture() {
        let stats = super::GraphStats::new();
        assert_eq!(stats.graph_instantiate_ns(), 0);
        stats.record_graph_instantiate(900_000);
        stats.record_graph_instantiate(100_000);
        assert_eq!(stats.graph_instantiate_ns(), 1_000_000);
    }
}
