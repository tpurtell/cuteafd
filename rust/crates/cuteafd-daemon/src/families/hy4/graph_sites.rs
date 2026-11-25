//! Graph capture-site classification for Hy4 (#2550).
use crate::shared::decode_graph::sites::{Scope, Table};

pub(super) const SITES: &Table = &[(
    "hy4.section",
    Scope::Local,
    "execution.rs: rank-local decode sections (including ContextSplit DsaPartial and \
     DsaAttnOut combine+gate_uv+o_proj) captured on each rank's exclusive stream; \
     DSA index_topk/scored_index_topk/dsa_candidate_merge, TP2 PeerExchange & \
     ContextExchange push/wait, Spark waves send/land/reduce, and TP2 head_handoff \
     stay outside capture",
)];

pub(super) fn spans(name: &str, declared: bool) -> bool {
    crate::shared::decode_graph::sites::spans(SITES, name, declared)
}

#[cfg(test)]
mod tests {
    #[test]
    fn every_site_is_classified_and_unknown_sites_span() {
        crate::shared::decode_graph::sites::testing::assert_classified("hy4", super::SITES);
        assert!(super::spans("new-unreviewed-site", false));
    }

    #[test]
    fn graph_prewarm_off_warms_spanning_and_defers_local() {
        let off = |name, declared| {
            crate::shared::decode_graph::sites::testing::assert_off_transitions(
                super::SITES,
                name,
                declared,
            )
        };
        assert_eq!(off("hy4.section", false), false, "hy4 rank-local layout");
    }
}
