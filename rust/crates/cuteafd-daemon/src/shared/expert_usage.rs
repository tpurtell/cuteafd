//! Per-layer x per-expert route hit counters.
//!
//! Every MoE family already builds a [`RoundRoutes`] for its verify step: the
//! exact per-layer, per-row routed expert ids, at whatever top-k the family
//! uses (6 for DeepSeek V4/V4.1, 8 for GLM 5.3 / GLM 5.3 Flash / MiMo V2,
//! 10 for Qwen 3.8, 16 for Kimi K3). Under TP slicing every Spark sees every
//! route, so a coordinator-side count is the whole truth, including RTX-local
//! layers. This module keeps that matrix behind `EXPERT_USAGE` (off by
//! default: enabling it turns on route capture).
//!
//! Counters are monotonic; there is no reset endpoint (Prometheus diffs them).
//! The matrix is exported in `/v1/stats` as `expert_usage`. Prefill routes are
//! not captured today, and a failed round is not counted (its routes describe
//! nothing).

use crate::shared::draft::routes::RoundRoutes;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Mutex;

/// One enabled matrix: `experts` counters per routed layer, grown as layers are
/// seen. `rows` counts the verifier rows of every recorded round.
struct Table {
    experts: usize,
    rows: AtomicU64,
    layers: Vec<Box<[AtomicU64]>>,
}

static USAGE: Mutex<Option<Table>> = Mutex::new(None);
/// Mirrors `USAGE.is_some()` for the hot path (one check per lane round).
static ENABLED: AtomicBool = AtomicBool::new(false);

/// Enable the matrix for a model with `experts` routed experts per layer.
/// Re-enabling replaces the table (start-up only).
pub(crate) fn enable(experts: u32) {
    let mut slot = USAGE.lock().unwrap_or_else(|error| error.into_inner());
    *slot = Some(Table { experts: experts as usize, rows: AtomicU64::new(0), layers: Vec::new() });
    ENABLED.store(true, Ordering::Release);
}

/// Whether the matrix is enabled (route capture must be on). Lock-free: this
/// runs once per lane round on the serving hot path.
pub(crate) fn enabled() -> bool {
    ENABLED.load(Ordering::Acquire)
}

/// Count every routed expert of every real row of every routed layer.
///
/// `routes` is layer-major `[layer][row][slot]`; `topk` is whatever the family
/// uses. Dense and unfilled layers are skipped (their ids are zero and would
/// otherwise be booked to expert 0). Rows come from the round's own row count,
/// not from layer 0, so a partial layer 0 cannot undercount. Out-of-range ids
/// are ignored (a different model's geometry must never panic serving).
pub(crate) fn record(routes: &RoundRoutes) {
    let mut slot = USAGE.lock().unwrap_or_else(|error| error.into_inner());
    let Some(table) = slot.as_mut() else { return };
    let (layers, rows, topk) = (routes.layers(), routes.rows(), routes.topk());
    if rows == 0 || topk == 0 { return; }
    table.rows.fetch_add(rows as u64, Ordering::Relaxed);
    let words = rows * topk;
    let flat = routes.flat();
    for layer in 0..layers {
        if routes.is_dense(layer) || !routes.is_filled(layer) { continue; }
        let start = layer * words;
        let Some(ids) = flat.get(start..start + words) else { continue };
        while table.layers.len() <= layer {
            let counters = (0..table.experts).map(|_| AtomicU64::new(0)).collect::<Vec<_>>();
            table.layers.push(counters.into_boxed_slice());
        }
        let hits = &table.layers[layer];
        for &id in ids {
            if let Some(cell) = hits.get(id as usize) {
                cell.fetch_add(1, Ordering::Relaxed);
            }
        }
    }
}

/// Record a V4.1-style `[layer][row][slot]` capture (top-k 6) through the
/// generic [`record`]. Empty layers (not captured this round) are skipped.
pub(crate) fn record_capture(capture: &[Vec<[u32; 6]>]) {
    let rows = capture.iter().find(|rows| !rows.is_empty()).map_or(0, Vec::len);
    if rows == 0 { return; }
    let mut routes = RoundRoutes::new(capture.len(), 6);
    if routes.fill_partial(capture, rows, u32::MAX).is_ok() {
        record(&routes);
    }
}

/// The `expert_usage` object exported in `/v1/stats`. Disabled is the small
/// `{"enabled": false}` shape; enabled carries the full matrix.
pub(crate) fn snapshot() -> serde_json::Value {
    let slot = USAGE.lock().unwrap_or_else(|error| error.into_inner());
    let Some(table) = slot.as_ref() else {
        return serde_json::json!({ "enabled": false });
    };
    let hits: Vec<Vec<u64>> = table.layers.iter()
        .map(|layer| layer.iter().map(|cell| cell.load(Ordering::Relaxed)).collect())
        .collect();
    serde_json::json!({
        "enabled": true,
        "experts": table.experts,
        "layers": table.layers.len(),
        "rows": table.rows.load(Ordering::Relaxed),
        "hits": hits,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The matrix is process-wide; serialize the tests that mutate it.
    static TEST_LOCK: Mutex<()> = Mutex::new(());

    fn reset() {
        *USAGE.lock().unwrap_or_else(|error| error.into_inner()) = None;
        ENABLED.store(false, Ordering::Release);
    }

    fn routes<const K: usize>(layers: usize, rows: &[Vec<[u32; K]>]) -> RoundRoutes {
        let mut routes = RoundRoutes::new(layers, K);
        routes.fill(rows, rows.first().map_or(0, Vec::len), u32::MAX).unwrap();
        routes
    }

    #[test]
    fn record_counts_each_route_once_per_row() {
        let _guard = TEST_LOCK.lock().unwrap();
        reset();
        enable(8);
        record(&routes(1, &[vec![[0, 1, 2, 3, 4, 5], [0, 0, 1, 1, 2, 2]]]));
        let snapshot = snapshot();
        assert_eq!(snapshot["rows"], 2);
        assert_eq!(snapshot["hits"][0][0], 3);
        assert_eq!(snapshot["hits"][0][1], 3);
        assert_eq!(snapshot["hits"][0][2], 3);
        assert_eq!(snapshot["hits"][0][3], 1);
        assert_eq!(snapshot["hits"][0][4], 1);
        assert_eq!(snapshot["hits"][0][5], 1);
        assert_eq!(snapshot["hits"][0][6], 0);
    }

    /// top-k is whatever the family uses: 6, 8, 10 and 16 all count.
    #[test]
    fn arbitrary_topk_is_counted() {
        let _guard = TEST_LOCK.lock().unwrap();
        for topk in [6usize, 8, 10, 16] {
            reset();
            enable(32);
            let row: Vec<u32> = (0..topk as u32).collect();
            let mut routes = RoundRoutes::new(1, topk);
            routes.begin(1);
            routes.push(0, &row, u32::MAX).unwrap();
            record(&routes);
            let snapshot = snapshot();
            assert_eq!(snapshot["rows"], 1, "topk {topk}");
            for expert in 0..topk {
                assert_eq!(snapshot["hits"][0][expert], 1, "topk {topk} expert {expert}");
            }
            assert_eq!(snapshot["hits"][0][topk], 0, "topk {topk} next expert");
        }
    }

    /// Rows come from the round, not from layer 0: a partial layer 0 must not
    /// undercount the rows of the layers that did run.
    #[test]
    fn rows_come_from_the_round_not_layer_zero() {
        let _guard = TEST_LOCK.lock().unwrap();
        reset();
        enable(8);
        let mut routes = RoundRoutes::new(2, 6);
        routes.begin(3);
        // Layer 0 is dense (no routes); layer 1 has three real rows.
        routes.dense(0).unwrap();
        routes.push_rows(1, &[[1; 6], [2; 6], [3; 6]], u32::MAX).unwrap();
        record(&routes);
        let snapshot = snapshot();
        assert_eq!(snapshot["rows"], 3);
        assert_eq!(snapshot["hits"][0][0], 0, "dense layer 0 is not booked to expert 0");
        assert_eq!(snapshot["hits"][1][1], 6);
        assert_eq!(snapshot["hits"][1][2], 6);
        assert_eq!(snapshot["hits"][1][3], 6);
    }

    /// The V4.1 capture adapter skips empty (uncaptured) layers and counts the
    /// rows of the layers that did run.
    #[test]
    fn capture_adapter_skips_empty_layers() {
        let _guard = TEST_LOCK.lock().unwrap();
        reset();
        enable(8);
        record_capture(&[vec![], vec![[1, 2, 3, 4, 5, 6], [1, 2, 3, 4, 5, 6]]]);
        let view = snapshot();
        assert_eq!(view["rows"], 2);
        assert_eq!(view["hits"][0][0], 0);
        assert_eq!(view["hits"][1][1], 2);
        // An all-empty capture records nothing.
        reset();
        enable(8);
        record_capture(&[vec![], vec![]]);
        assert_eq!(snapshot(), serde_json::json!({ "enabled": true, "experts": 8, "layers": 0, "rows": 0, "hits": [] }));
    }

    #[test]
    fn out_of_range_ids_are_ignored() {
        let _guard = TEST_LOCK.lock().unwrap();
        reset();
        enable(4);
        record(&routes(1, &[vec![[0, 9, 100, 3, 4, 5]]]));
        let snapshot = snapshot();
        assert_eq!(snapshot["experts"], 4);
        assert_eq!(snapshot["hits"][0][0], 1);
        assert_eq!(snapshot["hits"][0][3], 1);
        assert_eq!(snapshot["hits"][0].as_array().unwrap().len(), 4);
    }

    #[test]
    fn layers_grow_to_the_routes_seen() {
        let _guard = TEST_LOCK.lock().unwrap();
        reset();
        enable(4);
        let mut routes = RoundRoutes::new(3, 6);
        routes.begin(1);
        routes.dense(0).unwrap();
        routes.push_rows(1, &[[1; 6]], u32::MAX).unwrap();
        routes.push_rows(2, &[[2; 6]], u32::MAX).unwrap();
        record(&routes);
        let snapshot = snapshot();
        assert_eq!(snapshot["layers"], 3);
        assert_eq!(snapshot["rows"], 1);
        assert_eq!(snapshot["hits"][0][0], 0);
        assert_eq!(snapshot["hits"][1][1], 6);
        assert_eq!(snapshot["hits"][2][2], 6);
    }

    #[test]
    fn snapshot_shape_is_stable() {
        let _guard = TEST_LOCK.lock().unwrap();
        reset();
        enable(384);
        record(&routes(1, &[vec![[0; 6]]]));
        let snapshot = snapshot();
        assert_eq!(snapshot["enabled"], true);
        assert_eq!(snapshot["experts"], 384);
        assert_eq!(snapshot["layers"], 1);
        assert_eq!(snapshot["rows"], 1);
        let hits = snapshot["hits"].as_array().unwrap();
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].as_array().unwrap().len(), 384);
    }

    /// `enabled()` is the per-lane-round hot-path check; it reads an atomic, so
    /// it neither takes the table lock nor reads a poisoned lock as "off" while
    /// the matrix is enabled.
    #[test]
    fn enabled_is_a_lock_free_flag() {
        let _guard = TEST_LOCK.lock().unwrap();
        reset();
        assert!(!enabled());
        enable(4);
        assert!(enabled());
        let _ = std::thread::spawn(|| {
            let _held = USAGE.lock().unwrap_or_else(|error| error.into_inner());
            panic!("poison the table lock");
        }).join();
        assert!(USAGE.is_poisoned());
        assert!(enabled());
        USAGE.clear_poison();
        reset();
        assert!(!enabled());
    }

    #[test]
    fn disabled_snapshot_is_small() {
        let _guard = TEST_LOCK.lock().unwrap();
        reset();
        let snapshot = snapshot();
        assert_eq!(snapshot, serde_json::json!({ "enabled": false }));
        assert!(!enabled());
    }
}
