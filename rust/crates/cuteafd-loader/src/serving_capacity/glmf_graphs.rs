//! GLM Flash exact-row decode graph geometry and admission.
pub const DECODE_ROWS: usize = 64;

pub fn verify_budget(decode_rows: usize, sms: usize) -> usize {
    if decode_rows <= DECODE_ROWS { decode_rows } else { (3 * sms / 4).clamp(DECODE_ROWS, decode_rows) }
}

/// Lag-8 exact-row C1 census, short and 32K, on GLM Flash K4-MC K8 compact.
/// TP2 also matches the K4-MC original max W1 from #2645. No shape met the
/// one-half-of-band-cycles pin bar, so all captures are lazy.
pub fn working_set(gpus: usize, split_draft: bool, rank: usize) -> crate::placement::graph_reserve::WorkingSet {
    use crate::placement::graph_reserve::WorkingSet;
    let (w1_mib, step_growth) = match (gpus, split_draft, rank) {
        (1, _, _) => (74, 8_339_064),
        // Explicit TP2 DFlash A-on census (short and 32K C1).
        (2, true, 0) => (94, 9_789_077),
        (2, true, _) => (94, 9_629_699),
        (_, _, 0) => (94, 9_755_762),
        _ => (92, 9_566_036),
    };
    WorkingSet { w1: w1_mib << 20, step_growth, capture_growth: 4 << 20 }
}

pub fn bootstrap_bytes(gpus: usize, sparks: bool, rank: usize) -> u64 {
    // Initial per-executable estimates from the old traced layouts, unchanged
    // from #2645. The device controller replaces these with observed topology.
    let bootstrap: &[u64] = match (gpus, sparks) {
        (1, true) => &[175_849],
        (2, true) => &[156_746, 154_122],
        _ => &[195_387, 186_414],
    };
    bootstrap[rank.min(bootstrap.len() - 1)]
}

pub use super::decode_graph_key::{GraphGeometry, MIN_PAGE_STRIDE, decode_page_stride, decode_strides};

/// Every exchange slot owns its payload, completion word and native control storage.
pub fn peer_exchange_bytes(lanes: u64, rows: u64, hidden: u64, partial_bytes: u64, output_shard: bool) -> u64 {
    let slots = lanes * if output_shard { 6 } else { 4 };
    slots * (rows.max(64) * hidden * partial_bytes).max(256) + ((slots + 1) * 16).max(256)
}
