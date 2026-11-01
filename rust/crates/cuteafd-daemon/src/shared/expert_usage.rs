//! Per-layer x per-expert route hit counters.
//!
//! The coordinator already holds the exact per-layer route ids of every verify
//! round (`pass.captured_routes()`, `[u32; 6]` per row). Under TP slicing every
//! Spark sees every route, so a coordinator-side count is the whole truth,
//! including RTX-local layers. This module keeps that matrix behind
//! `V41_EXPERT_USAGE` (off by default: enabling it turns on route capture).
//!
//! Counters are monotonic; there is no reset endpoint (Prometheus diffs them).
//! The matrix is exported in `/v1/stats` as `expert_usage` and, per layer, as
//! skew gauges by the metrics exporter. Prefill routes are not captured today.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Mutex;

/// One enabled matrix: `experts` counters per routed layer, grown as layers are
/// seen. `rows` counts the real (non-pad) rows of layer 0.
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

/// Count every routed expert of every real row. `routes[layer][row]` is the
/// six selected expert ids of that row; empty layers are skipped. Out-of-range
/// ids are ignored (a different model's geometry must never panic serving).
pub(crate) fn record(routes: &[Vec<[u32; 6]>]) {
    let mut slot = USAGE.lock().unwrap_or_else(|error| error.into_inner());
    let Some(table) = slot.as_mut() else { return };
    if let Some(first) = routes.first() {
        table.rows.fetch_add(first.len() as u64, Ordering::Relaxed);
    }
    for (layer, rows) in routes.iter().enumerate() {
        if rows.is_empty() { continue; }
        while table.layers.len() <= layer {
            let counters = (0..table.experts).map(|_| AtomicU64::new(0)).collect::<Vec<_>>();
            table.layers.push(counters.into_boxed_slice());
        }
        let hits = &table.layers[layer];
        for row in rows {
            for &id in row {
                if let Some(cell) = hits.get(id as usize) {
                    cell.fetch_add(1, Ordering::Relaxed);
                }
            }
        }
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

    #[test]
    fn record_counts_each_route_once_per_row() {
        let _guard = TEST_LOCK.lock().unwrap();
        reset();
        enable(8);
        record(&[vec![[0, 1, 2, 3, 4, 5], [0, 0, 1, 1, 2, 2]]]);
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

    #[test]
    fn out_of_range_ids_are_ignored() {
        let _guard = TEST_LOCK.lock().unwrap();
        reset();
        enable(4);
        record(&[vec![[0, 9, 100, 3, 4, 5]]]);
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
        record(&[vec![], vec![[1; 6]], vec![[2; 6]]]);
        let snapshot = snapshot();
        assert_eq!(snapshot["layers"], 3);
        assert_eq!(snapshot["rows"], 0);
        assert_eq!(snapshot["hits"][0][0], 0);
        assert_eq!(snapshot["hits"][1][1], 6);
        assert_eq!(snapshot["hits"][2][2], 6);
    }

    #[test]
    fn snapshot_shape_is_stable() {
        let _guard = TEST_LOCK.lock().unwrap();
        reset();
        enable(384);
        record(&[vec![[0; 6]]]);
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
