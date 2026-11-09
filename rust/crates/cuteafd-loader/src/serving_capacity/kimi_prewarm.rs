//! Kimi K3 prewarm shapes and per-layout graph reserve.
//!
//! Each coordinator layout (rank count) captures a different inventory: TP2
//! peer segments and RTX expert onboarding change which sites exist, so the
//! checked-in list carries one row per layout and admission reserves that
//! row's arena on every physical device.
//!
//! `reserve_base_bytes` is the layout's measured prewarm arena on its largest
//! rank and `reserve_extra_bytes` is the 2 MiB allocation tail. Shared
//! GPU_GRAPH_RESERVE_EXTRA supplies runtime headroom; GPU_GRAPH_RESERVE
//! overrides the whole reserve. The census base is provisional until a
//! GRAPH_TRACE run replaces them; regenerate with
//! `scripts/dev/graph-prewarm-from-trace.py --family kimi_k3`.
//!
//! Listed rows are the replayable startup geometry: the plain buckets plus
//! every verify width the drafter reaches (two through its block of eight for
//! non-context sites, and context decode buckets through the native 64-row
//! cap). The 4096 prefill bucket and wider prefill-window rows are captured
//! lazily, if at all: prefill runs the eager body inside a row window, so
//! those keys never replay and must not be prewarmed (#1646).

use std::sync::OnceLock;

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct KimiPrewarmEntry {
    pub rows: usize,
    /// Verify-shaped attention capture at this width. Flagless residual/FFN
    /// captures at the same rows are covered by any pinned row.
    pub speculative: bool,
    pub pin: bool,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct KimiPrewarmLayout {
    /// Coordinator ranks this row was measured on.
    pub ranks: usize,
    pub layout: String,
    pub reserve_base_bytes: u64,
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
        reserve_base_bytes: crate::families::kimi_k3::allocation::GRAPH_CENSUS_BYTES,
        reserve_extra_bytes: 2 << 20,
        entries: [1, 4, 16, 64].into_iter().map(|rows|
            KimiPrewarmEntry { rows, speculative: false, pin: true })
            .chain((2..=64).map(|rows|
                KimiPrewarmEntry { rows, speculative: true, pin: true })).collect(),
    })
}

/// Per-device measured prewarm base and allocation tail.
pub fn kimi_reserve(ranks: usize) -> (u64, u64) {
    let layout = kimi_layout(ranks);
    (layout.reserve_base_bytes, layout.reserve_extra_bytes)
}

/// Whether a listed layout prewarms this row width. Only replayable widths are
/// listed: prefill-window rows never appear, so they stay eager.
pub fn kimi_prewarm(ranks: usize, rows: usize, speculative: bool) -> bool {
    (1..=64).contains(&rows) && selected(kimi_layout(ranks), rows, speculative, false)
}

fn selected(layout: &KimiPrewarmLayout, rows: usize, speculative: bool, pinned: bool) -> bool {
    layout.entries.iter().any(|entry| (!pinned || entry.pin) && entry.rows == rows
        && (entry.speculative == speculative || !speculative))
}

/// A pinned shape. A pin at `(rows, speculative)` also pins the flagless
/// residual/FFN captures at those rows; it never pins a verify capture of a
/// different width.
pub fn kimi_pinned(ranks: usize, rows: usize, speculative: bool) -> bool {
    (1..=64).contains(&rows) && selected(kimi_layout(ranks), rows, speculative, true)
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
    fn every_layout_names_its_shapes_and_reserves_a_positive_arena() {
        let list = kimi_prewarm_list();
        assert_eq!(list.layouts.len(), 2);
        for layout in &list.layouts {
            assert!(layout.ranks >= 1 && !layout.layout.is_empty());
            assert!(layout.reserve_base_bytes > 0 && layout.reserve_extra_bytes > 0);
            assert!(!layout.entries.is_empty());
            assert!(layout.entries.iter().all(|entry| entry.rows > 0));
        }
        assert!(kimi_reserve(2).0 > kimi_reserve(1).0, "TP2 holds more executables");
        assert_eq!(kimi_layout(9).ranks, 0, "untraced layouts use bootstrap, not another row");
        assert_eq!(kimi_reserve(1).1, 2 << 20);
        assert_eq!(kimi_reserve(2).1, 2 << 20);
    }

    #[test]
    fn prefill_rows_never_prewarm() {
        assert!(kimi_prewarm(1, 1, false) && kimi_prewarm(1, 64, true));
        for rows in [1024, 3072, 4096] {
            assert!(!kimi_prewarm(1, rows, false), "prefill windows never replay");
        }
        assert!(!kimi_prewarm(1, 4097, true));
    }

    #[test]
    fn the_whole_replayable_startup_row_band_is_listed() {
        // Plain buckets 1/4/16/64, non-context verify widths 2..=8 (DFlash2's
        // block is eight) and context decode buckets 2..=64 all capture at
        // startup and replay, so the bootstrap prewarms every one.
        for rows in 1..=64 {
            assert!(kimi_prewarm(1, rows, false), "row {rows} is startup geometry");
        }
        assert!(!kimi_prewarm(1, 65, false), "rows above the native 64-row cap are lazy");
    }

    #[test]
    fn a_pin_covers_flagless_sites_at_its_rows_and_listed_verify_widths() {
        assert!(kimi_pinned(1, 1, false));
        assert!(!kimi_pinned(1, 1, true), "row 1 has no verify capture");
        assert!(kimi_pinned(1, 4, false));
        assert!(kimi_pinned(1, 4, true));
        assert!(kimi_pinned(1, 5, true), "verify widths through the block are listed");
        assert!(kimi_pinned(1, 16, false));
        assert!(kimi_pinned(1, 16, true), "context decode reaches 64 rows");
        assert!(kimi_pinned(1, 64, true));
        assert!(!kimi_pinned(1, 65, true), "wider verify batches stay lazy");
        assert!(!kimi_pinned(1, 4096, false));
    }

    #[test]
    fn selection_stays_within_the_layout_and_speculative_variant() {
        let layout = KimiPrewarmLayout { ranks: 1, layout: "test".into(),
            reserve_base_bytes: 1, reserve_extra_bytes: 2 << 20,
            entries: vec![KimiPrewarmEntry { rows: 3, speculative: false, pin: true }] };
        assert!(selected(&layout, 3, false, false));
        assert!(!selected(&layout, 3, true, false));
        assert!(!selected(&layout, 4, false, false));
        assert!(selected(kimi_bootstrap_layout(), 4, true, false));
    }

    #[test]
    fn bootstrap_is_the_census_share_of_one_capture() {
        let (bytes, executables) = crate::families::kimi_k3::allocation::graph_census(1, 0);
        assert_eq!(kimi_bootstrap(1, 0), bytes.div_ceil(executables));
        assert!(kimi_bootstrap(2, 1) > 0);
    }
}
