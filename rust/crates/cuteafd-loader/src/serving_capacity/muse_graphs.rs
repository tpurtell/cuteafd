//! Muse decode prewarm: exact live rows, with a fixed admitted page-table pitch.
//! A layout without its own trace has no startup graphs or pins and charges the
//! family × ranks envelope, so its decode graphs still capture lazily.
use serde::{Deserialize, Serialize};
use std::sync::OnceLock;

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Entry {
    pub rows: usize,
    pub pin: bool,
    /// Physical capture shares, including capture-only siblings, by rank.
    pub capture_bytes: Vec<u64>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Layout {
    pub ranks: usize,
    pub onboard_layers: usize,
    pub provenance: String,
    pub reserve_base_bytes: Vec<u64>,
    pub reserve_extra_bytes: Vec<u64>,
    pub entries: Vec<Entry>,
}

#[derive(Debug, Deserialize)]
struct List {
    schema_version: u32,
    family: String,
    layouts: Vec<Layout>,
}

pub fn layout(ranks: usize, onboard_layers: usize) -> Option<&'static Layout> {
    static LIST: OnceLock<List> = OnceLock::new();
    let list = LIST.get_or_init(|| {
        let list: List = serde_json::from_str(include_str!("muse_prewarm.json")).expect("Muse prewarm JSON");
        assert_eq!(list.schema_version, 1);
        assert_eq!(list.family, "muse_spark");
        let mut keys = std::collections::HashSet::new();
        for layout in &list.layouts {
            assert!(matches!(layout.ranks, 1 | 2));
            assert!(layout.onboard_layers <= 62);
            assert!(keys.insert((layout.ranks, layout.onboard_layers)), "duplicate Muse capture layout");
            assert_eq!(layout.reserve_base_bytes.len(), layout.ranks);
            assert_eq!(layout.reserve_extra_bytes, vec![2 << 20; layout.ranks]);
            assert!(layout.entries.iter().all(|entry| (1..=64).contains(&entry.rows)
                && entry.capture_bytes.len() == layout.ranks));
        }
        list
    });
    list.layouts.iter().find(|layout| layout.ranks == ranks && layout.onboard_layers == onboard_layers)
}

/// The traced layout's arena plus the shared extra, else the Muse × ranks
/// envelope; `decode_graphs = false` charges nothing.
pub fn policy(ranks: usize, onboard_layers: usize, rank: usize, decode_graphs: bool)
    -> Result<crate::placement::graph_reserve::Resolved, crate::placement::graph_reserve::Error> {
    use crate::placement::graph_reserve::{self, GraphFamily};
    if !decode_graphs { return Ok(graph_reserve::disabled(rank)); }
    match layout(ranks, onboard_layers) {
        Some(layout) => graph_reserve::current(rank, false, layout.reserve_base_bytes[rank], layout.reserve_extra_bytes[rank]),
        None => graph_reserve::current_envelope(GraphFamily::Muse, ranks, rank),
    }
}

pub fn selected<'a>(layout: &'a Layout, bases: &[u64]) -> Vec<&'a Entry> {
    let pins = layout.entries.iter().map(|entry| entry.pin).collect::<Vec<_>>();
    crate::placement::graph_reserve::prewarm_selection(&pins, bases, |indices, bases| {
        bases.iter().enumerate().all(|(rank, &base)| {
            let sum = indices.iter().try_fold(0u64, |sum, &i|
                sum.checked_add(layout.entries[i].capture_bytes[rank]));
            sum.is_some_and(|bytes| bytes.next_multiple_of(2 << 20) <= base)
        })
    }).into_iter().map(|i| &layout.entries[i]).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn layout_without_trace_charges_the_envelope_and_off_charges_nothing() {
        cuteafd_core::memory_policy::MemoryPolicy::default().scoped(|| {
            for (ranks, onboard) in [(1, 0), (1, 5), (2, 2), (2, 62)] {
                for rank in 0..ranks {
                    let charged = policy(ranks, onboard, rank, true).unwrap();
                    assert_eq!(charged.bytes, crate::placement::graph_reserve::envelope_bytes(
                        crate::placement::graph_reserve::GraphFamily::Muse, ranks, rank));
                    assert_eq!(policy(ranks, onboard, rank, false).unwrap().bytes, 0);
                }
            }
            assert_eq!(policy(1, 2, 0, true).unwrap().bytes, 1_559_061_760);
        });
    }

    #[test]
    fn untraced_layout_borrows_no_inventory_and_pins_nothing() {
        // This family had no startup graphs before this implementation.
        assert!(layout(3, 0).is_none());
        for ranks in [1, 2] {
            for onboard_layers in [0, 2, 62] {
                if let Some(layout) = layout(ranks, onboard_layers) {
                    assert_eq!(layout.reserve_base_bytes.len(), ranks);
                } else {
                    assert!(layout(ranks, onboard_layers).is_none());
                }
            }
        }
    }

    #[test]
    fn measured_min_inventory_prices_only_its_own_rows_and_pins() {
        let list = layout(1, 2).expect("measured 1 RTX/onboard 2 inventory");
        assert_eq!(list.entries.len(), 55);
        assert_eq!(list.entries.iter().filter(|entry| entry.pin).map(|entry| entry.rows)
            .collect::<Vec<_>>(), [56]);
        assert_eq!(list.reserve_base_bytes, [1_056_964_608]);
        let sum = list.entries.iter().map(|entry| entry.capture_bytes[0]).sum::<u64>();
        assert_eq!(sum.next_multiple_of(2 << 20), list.reserve_base_bytes[0]);
        assert_eq!(selected(list, &list.reserve_base_bytes).len(), list.entries.len());
        assert!(list.entries.iter().any(|entry| entry.rows == 1));
        assert!(layout(2, 2).is_none(), "untraced TP2 cannot inherit min inventory");
    }

    #[test]
    fn measured_max_inventory_is_rank_local_and_does_not_seed_onboarded_layouts() {
        let list = layout(2, 0).expect("measured 2 RTX/onboard 0 inventory");
        assert_eq!(list.entries.len(), 53);
        assert!(list.entries.iter().all(|entry| !entry.pin));
        assert_eq!(list.reserve_base_bytes, [1_287_651_328, 1_302_331_392]);
        for rank in 0..2 {
            let sum = list.entries.iter().map(|entry| entry.capture_bytes[rank]).sum::<u64>();
            assert_eq!(sum.next_multiple_of(2 << 20), list.reserve_base_bytes[rank]);
        }
        assert_eq!(selected(list, &list.reserve_base_bytes).len(), list.entries.len());
        assert!(list.entries.iter().any(|entry| entry.rows == 1));
        assert!(layout(2, 2).is_none());
    }

    #[test]
    fn smaller_base_fits_pins_as_a_set_then_a_heat_ordered_prefix() {
        let list = Layout { ranks: 1, onboard_layers: 2, provenance: "fixture".into(),
            reserve_base_bytes: vec![8 << 20], reserve_extra_bytes: vec![2 << 20],
            entries: vec![Entry { rows: 1, pin: true, capture_bytes: vec![2 << 20] },
                Entry { rows: 4, pin: false, capture_bytes: vec![4 << 20] },
                Entry { rows: 8, pin: false, capture_bytes: vec![1 << 20] }] };
        assert!(selected(&list, &[0]).is_empty());
        assert_eq!(selected(&list, &[4 << 20]).iter().map(|e| e.rows).collect::<Vec<_>>(), [1]);
        assert_eq!(selected(&list, &[6 << 20]).iter().map(|e| e.rows).collect::<Vec<_>>(), [1, 4]);
    }
}
