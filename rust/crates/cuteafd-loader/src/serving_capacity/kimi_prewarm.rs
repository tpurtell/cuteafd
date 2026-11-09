//! Kimi K3 prewarm shapes and per-layout graph reserve.
//!
//! Each coordinator layout (rank count) captures a different inventory: TP2
//! peer segments and RTX expert onboarding change which sites exist, so the
//! checked-in list carries one row per layout and admission reserves that
//! row's arena on every physical device.
//!
//! `reserve_base_bytes` is the layout's measured startup graph census (the
//! charged bytes of the prewarm set on its largest rank) and
//! `reserve_extra_bytes` is one largest live capture above it, which is the
//! room lazy captures and recaptures need. Both are provisional until a
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

/// The row for a coordinator rank count. Wider layouts than any measured row
/// reuse the widest one: a reserve that is too small only costs an eager miss,
/// never a refusal.
pub fn kimi_layout(ranks: usize) -> &'static KimiPrewarmLayout {
    let list = kimi_prewarm_list();
    list.layouts.iter().find(|layout| layout.ranks == ranks)
        .or_else(|| list.layouts.iter().max_by_key(|layout| layout.ranks))
        .expect("Kimi K3 prewarm list has a layout row")
}

/// Per-device graph reserve for a coordinator layout: census base plus one
/// capture of headroom.
pub fn kimi_reserve(ranks: usize) -> (u64, u64) {
    let layout = kimi_layout(ranks);
    (layout.reserve_base_bytes, layout.reserve_extra_bytes)
}

/// Whether a listed layout prewarms this row width. Only replayable widths are
/// listed: prefill-window rows never appear, so they stay eager.
pub fn kimi_prewarm(rows: usize) -> bool {
    kimi_prewarm_list().layouts.iter()
        .flat_map(|layout| &layout.entries)
        .any(|entry| entry.rows == rows)
}

/// A pinned shape. A pin at `(rows, speculative)` also pins the flagless
/// residual/FFN captures at those rows; it never pins a verify capture of a
/// different width.
pub fn kimi_pinned(ranks: usize, rows: usize, speculative: bool) -> bool {
    kimi_layout(ranks).entries.iter().any(|entry| entry.pin && entry.rows == rows
        && (entry.speculative == speculative || !speculative))
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
        assert_eq!(kimi_reserve(9), kimi_reserve(2), "wider layouts reuse the widest row");
    }

    #[test]
    fn prefill_rows_never_prewarm() {
        assert!(kimi_prewarm(1) && kimi_prewarm(64));
        for rows in [1024, 3072, 4096] {
            assert!(!kimi_prewarm(rows), "prefill windows never replay");
        }
        assert!(!kimi_prewarm(4097));
    }

    #[test]
    fn the_whole_replayable_startup_row_band_is_listed() {
        // Plain buckets 1/4/16/64, non-context verify widths 2..=8 (DFlash2's
        // block is eight) and context decode buckets 2..=64 all capture at
        // startup and replay, so the bootstrap prewarms every one.
        for rows in 1..=64 {
            assert!(kimi_prewarm(rows), "row {rows} is startup geometry");
        }
        assert!(!kimi_prewarm(65), "rows above the native 64-row cap are lazy");
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
    fn bootstrap_is_the_census_share_of_one_capture() {
        let (bytes, executables) = crate::families::kimi_k3::allocation::graph_census(1, 0);
        assert_eq!(kimi_bootstrap(1, 0), bytes.div_ceil(executables));
        assert!(kimi_bootstrap(2, 1) > 0);
    }
}
