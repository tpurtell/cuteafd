//! Graph capture-site classification for Hy4 (#2550).
use crate::shared::decode_graph::sites::{Scope, Table};

pub(super) const SITES: &Table = &[(
    "hy4.section",
    Scope::Local,
    "execution.rs: rank-local decode sections (DsaFullPre, DsaSharedPre, TP1 \
     Spark-routed DsaSpark with bucketed index_topk_decode, DsaLocalPost, DsaAttnOut, \
     FfnFrontend, TP2 SparkRouter [fusing rank-1 shared_ffn], FfnFinish, Tail, and \
     Draft*) captured on each rank's exclusive stream; TP2/local index_topk, \
     ContextSplit scored_index_topk/dsa_candidate_merge and short partial/combine \
     hops, TP2 PeerExchange & ContextExchange push/wait, Spark waves send/land/reduce \
     and rank-0 shared_ffn RDMA-shadow overlap, and TP2 head_handoff stay outside capture",
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
