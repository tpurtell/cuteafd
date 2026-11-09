//! Kimi K3 prewarm shapes and per-layout graph reserve.
//!
//! Each coordinator layout (rank count) captures a different inventory: TP2
//! peer segments and RTX expert onboarding change which sites exist, so the
//! checked-in list carries one row per layout and admission reserves that
//! row's arena on every physical device.
//!
//! `reserve_base_bytes` holds each physical rank's measured prewarm arena;
//! `reserve_extra_bytes` is the 2 MiB allocation tail. Shared
//! GPU_GRAPH_RESERVE_EXTRA supplies runtime headroom; GPU_GRAPH_RESERVE
//! overrides the whole reserve. The census base is provisional until a
//! GRAPH_TRACE run replaces them; regenerate with
//! `scripts/dev/graph-prewarm-from-trace.py --family kimi_k3`.
//!
//! Listed rows preserve observed operation, verify variant and attention
//! sequence count within the native 64-row replay band. Prefill runs eagerly
//! inside a row window: its keys never replay and are not prewarmed (#1646).

use std::sync::OnceLock;

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum KimiPrewarmOp { Residual, Ffn, Attention, Context }

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct KimiPrewarmEntry {
    pub rows: usize,
    /// Verify-shaped attention or context capture; false for flagless sites.
    pub speculative: bool,
    pub pin: bool,
    /// Absent only in the provisional startup bootstrap set.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub op: Option<KimiPrewarmOp>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sequences: Option<usize>,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct KimiPrewarmLayout {
    /// Coordinator ranks this row was measured on.
    pub ranks: usize,
    pub layout: String,
    pub reserve_base_bytes: Vec<u64>,
    pub reserve_extra_bytes: u64,
    pub entries: Vec<KimiPrewarmEntry>,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct KimiPrewarmList {
    pub schema_version: u32,
    pub family: String,
    pub layouts: Vec<KimiPrewarmLayout>,
}

fn kimi_prewarm_list() -> &'static KimiPrewarmList {
    static LIST: OnceLock<KimiPrewarmList> = OnceLock::new();
    LIST.get_or_init(|| {
        let list: KimiPrewarmList = serde_json::from_str(include_str!("kimi_prewarm.json"))
            .expect("checked-in Kimi K3 prewarm list");
        assert_eq!(list.schema_version, 1);
        assert_eq!(list.family, "kimi_k3");
        for layout in &list.layouts {
            assert_eq!(layout.reserve_base_bytes.len(), layout.ranks, "Kimi per-rank graph bases");
        }
        list
    })
}

/// An untraced layout must not borrow another layout's geometries.
pub fn kimi_layout(ranks: usize) -> &'static KimiPrewarmLayout {
    let list = kimi_prewarm_list();
    list.layouts.iter().find(|layout| layout.ranks == ranks)
        .unwrap_or_else(|| {
            tracing::warn!(ranks, "Kimi graph prewarm trace missing for layout; using bootstrap startup set");
            kimi_bootstrap_layout()
        })
}

fn kimi_bootstrap_layout() -> &'static KimiPrewarmLayout {
    static BOOTSTRAP: OnceLock<KimiPrewarmLayout> = OnceLock::new();
    BOOTSTRAP.get_or_init(|| KimiPrewarmLayout {
        ranks: 0,
        layout: "bootstrap startup set (untraced)".into(),
        reserve_base_bytes: vec![],
        reserve_extra_bytes: 2 << 20,
        entries: [1, 4, 16, 64].into_iter().map(|rows|
            KimiPrewarmEntry { rows, speculative: false, pin: true, op: None, sequences: None })
            .chain((2..=64).map(|rows|
                KimiPrewarmEntry { rows, speculative: true, pin: true, op: None, sequences: None })).collect(),
    })
}

/// Per-device measured prewarm base and allocation tail.
pub fn kimi_reserve(ranks: usize, rank: usize) -> (u64, u64) {
    assert!(rank < ranks, "Kimi graph reserve rank");
    let layout = kimi_layout(ranks);
    let base = if layout.ranks == 0 {
        crate::families::kimi_k3::allocation::graph_census(ranks, rank).0
    } else {
        layout.reserve_base_bytes[rank]
    };
    (base, layout.reserve_extra_bytes)
}

/// Select an exact traced operation/attention geometry. Bootstrap entries have
/// no operation and retain the original startup inventory; flagless sites can
/// share a row across decode and verify but attention variants cannot.
pub fn kimi_selected(layout: &KimiPrewarmLayout, rows: usize, speculative: bool,
    op: KimiPrewarmOp, sequences: Option<usize>, pinned: bool) -> bool {
    (1..=64).contains(&rows) && layout.entries.iter().any(|entry| {
        (!pinned || entry.pin) && entry.rows == rows && match entry.op {
            Some(kind) => kind == op && entry.speculative == speculative && entry.sequences == sequences,
            None => matches!(op, KimiPrewarmOp::Residual | KimiPrewarmOp::Ffn)
                || entry.speculative == speculative,
        }
    })
}

/// Provisional per-capture estimate before the cache has a measurement: the
/// layout's startup census divided by its captured executables. That is a
/// byte basis (Kimi has no calibrated node rate for this route), so the cache
/// itself still selects by physical growth once a capture completes.
pub fn kimi_bootstrap(ranks: usize, rank: usize) -> u64 {
    let (bytes, executables) = crate::families::kimi_k3::allocation::graph_census(ranks, rank);
    bytes.div_ceil(executables.max(1)).max(1)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_layout_names_its_shapes_and_reserves_each_physical_rank() {
        let list = kimi_prewarm_list();
        assert_eq!(list.layouts.len(), 2);
        for layout in &list.layouts {
            assert!(layout.ranks >= 1 && !layout.layout.is_empty());
            assert_eq!(layout.reserve_base_bytes.len(), layout.ranks);
            assert_eq!(layout.reserve_extra_bytes, 2 << 20);
            assert!(!layout.entries.is_empty());
            assert!(layout.entries.iter().all(|entry| entry.rows > 0));
        }
        for layout in &list.layouts {
            for rank in 0..layout.ranks {
                assert_eq!(kimi_reserve(layout.ranks, rank).0, layout.reserve_base_bytes[rank]);
            }
        }
        assert_eq!(kimi_layout(9).ranks, 0, "untraced layouts use bootstrap, not another row");
        assert_eq!(kimi_reserve(1, 0).1, 2 << 20);
        assert_eq!(kimi_reserve(2, 1).1, 2 << 20);
    }

    #[test]
    fn bootstrap_only_contains_replayable_rows() {
        let layout = kimi_bootstrap_layout();
        for entry in &layout.entries {
            assert!((1..=64).contains(&entry.rows));
        }
        for rows in [0, 65, 1024, 3072, 4096] {
            assert!(!kimi_selected(layout, rows, false, KimiPrewarmOp::Ffn, None, false));
        }
    }

    #[test]
    fn selection_stays_within_the_layout_operation_and_sequence_variant() {
        let layout = KimiPrewarmLayout { ranks: 1, layout: "test".into(),
            reserve_base_bytes: vec![1], reserve_extra_bytes: 2 << 20,
            entries: vec![KimiPrewarmEntry { rows: 20, speculative: true, pin: true,
                op: Some(KimiPrewarmOp::Attention), sequences: Some(4) }] };
        let select = |rows, spec, op, seq| kimi_selected(&layout, rows, spec, op, seq, false);
        assert!(select(20, true, KimiPrewarmOp::Attention, Some(4)));
        assert!(!select(20, true, KimiPrewarmOp::Attention, Some(1)));
        assert!(!select(20, false, KimiPrewarmOp::Attention, Some(4)));
        assert!(!select(20, true, KimiPrewarmOp::Context, None));
        assert!(!select(4, true, KimiPrewarmOp::Attention, Some(4)));
    }

    #[test]
    fn context_entry_selects_only_its_exact_context_variant() {
        let layout = KimiPrewarmLayout { ranks: 2, layout: "context".into(),
            reserve_base_bytes: vec![1], reserve_extra_bytes: 2 << 20,
            entries: vec![KimiPrewarmEntry { rows: 16, speculative: false, pin: true,
                op: Some(KimiPrewarmOp::Context), sequences: None }] };
        assert!(kimi_selected(&layout, 16, false, KimiPrewarmOp::Context, None, false));
        assert!(!kimi_selected(&layout, 16, true, KimiPrewarmOp::Context, None, false));
        for op in [KimiPrewarmOp::Residual, KimiPrewarmOp::Ffn, KimiPrewarmOp::Attention] {
            assert!(!kimi_selected(&layout, 16, false, op, None, false));
        }
    }

    #[test]
    fn generic_entries_select_only_their_own_operation_class() {
        for kind in [KimiPrewarmOp::Residual, KimiPrewarmOp::Ffn] {
            let layout = KimiPrewarmLayout { ranks: 1, layout: "generic".into(),
                reserve_base_bytes: vec![1], reserve_extra_bytes: 2 << 20,
                entries: vec![KimiPrewarmEntry { rows: 20, speculative: false, pin: false,
                    op: Some(kind), sequences: None }] };
            for op in [KimiPrewarmOp::Residual, KimiPrewarmOp::Ffn,
                KimiPrewarmOp::Attention, KimiPrewarmOp::Context] {
                assert_eq!(kimi_selected(&layout, 20, false, op, None, false), op == kind);
            }
            assert!(!kimi_selected(&layout, 20, false, kind, None, true));
            assert!(!kimi_selected(&layout, 16, false, kind, None, false));
        }
    }

    #[test]
    fn bootstrap_is_the_census_share_of_one_capture() {
        let (bytes, executables) = crate::families::kimi_k3::allocation::graph_census(1, 0);
        assert_eq!(kimi_bootstrap(1, 0), bytes.div_ceil(executables));
        assert!(kimi_bootstrap(2, 1) > 0);
    }
}
