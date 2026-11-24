//! Graph capture-site classification (#2303, #2609). An unclassified site is
//! spanning: it stays prewarmed under `GRAPH_PREWARM=off` and sealed after
//! readiness (#1878).
use crate::shared::decode_graph::sites::{Scope, Table};

// (site, scope, capture and host-issue-order evidence).
pub(super) const SITES: &Table = &[
    ("glm5.decode_local", Scope::Local,
        "engine/local_graph.rs:95 decode_layers_local and engine/layers.rs:255 keep every PeerExchange, ContextExchange, and HopLink push/wait outside replay_on with zero unmatched waits during capture"),
];

pub(super) fn spans(name: &str, declared: bool) -> bool {
    crate::shared::decode_graph::sites::spans(SITES, name, declared)
}

#[cfg(test)]
mod tests {
    #[test]
    fn every_site_is_classified_and_unknown_sites_span() {
        crate::shared::decode_graph::sites::testing::assert_classified("glm5", super::SITES);
        assert!(super::spans("new-unreviewed-site", false));
    }

    #[test]
    fn graph_prewarm_off_warms_spanning_and_defers_local() {
        let off = |name, declared| crate::shared::decode_graph::sites::testing::assert_off_transitions(super::SITES, name, declared);
        assert_eq!(off("glm5.decode_local", false), false, "glm5 exact-row local site (no peer dependency inside capture)");
    }
}
