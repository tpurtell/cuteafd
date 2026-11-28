//! Graph capture-site classification for Hy4-flash (#2591 C2 / #2604).
//! An unclassified site is spanning: it stays prewarmed under `GRAPH_PREWARM=off`
//! and sealed after readiness (#1878).
use crate::shared::decode_graph::sites::{Scope, Table};

// (site, scope, capture and host-issue-order evidence).
pub(super) const SITES: &Table = &[(
    "hy4_flash.section",
    Scope::Local,
    "execution.rs: rank-local decode sections (including TP1 Spark-routed SwaSpark, \
     SwaExtendSpark, DsaPreScored, DsaSparkPost, Tail, TP2 SparkRouter [fusing rank-1 \
     shared_ffn], and ContextSplit DsaPartial / DsaAttnOut combine+gate_uv+o_proj) \
     captured on each rank's exclusive stream; Engram H2D/PeerExchange, DSA indexer \
     pending D2D/pool/score/topk/merge_owned, TP2 PeerExchange & ContextExchange \
     push/wait, Spark waves send/land/reduce and rank-0 shared_ffn RDMA-shadow overlap, \
     DFly2 metadata and dual recall gathers, and TP2 head_handoff stay outside capture",
)];

pub(super) fn spans(name: &str, declared: bool) -> bool {
    crate::shared::decode_graph::sites::spans(SITES, name, declared)
}

#[cfg(test)]
mod tests {
    #[test]
    fn every_site_is_classified_and_unknown_sites_span() {
        crate::shared::decode_graph::sites::testing::assert_classified("hy4_flash", super::SITES);
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
        assert_eq!(off("hy4_flash.section", false), false, "hy4_flash rank-local layout");
    }
}
