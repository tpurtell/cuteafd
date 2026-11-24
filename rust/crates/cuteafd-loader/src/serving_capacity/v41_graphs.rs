//! Trace-qualified V4.1 inventories: each physical rank owns its measured base.

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct V41PrewarmEntry {
    pub rows: usize,
    pub pin: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct V41DraftPrewarmEntry {
    pub requests: usize,
    pub width: usize,
    pub pin: bool,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct V41RankPrewarm {
    pub rank: usize,
    /// Sum of selected identities' measured capture shares, rounded once.
    pub reserve_base_bytes: u64,
    /// One physical allocation tail, not runtime policy headroom.
    pub reserve_extra_bytes: u64,
    pub entries: Vec<V41PrewarmEntry>,
    pub draft_entries: Vec<V41DraftPrewarmEntry>,
    pub prefill_entries: Vec<V41PrewarmEntry>,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct V41LayoutPrewarm {
    pub layout: usize,
    pub ranks: Vec<V41RankPrewarm>,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct V41PrewarmList {
    pub schema_version: u32,
    pub family: String,
    pub layouts: Vec<V41LayoutPrewarm>,
}

pub fn v41_prewarm_list() -> &'static V41PrewarmList {
    static LIST: std::sync::OnceLock<V41PrewarmList> = std::sync::OnceLock::new();
    LIST.get_or_init(|| {
        let list: V41PrewarmList = serde_json::from_str(include_str!("v41_prewarm.json"))
            .expect("checked-in V4.1 prewarm inventories");
        assert_eq!(list.schema_version, 2);
        assert_eq!(list.family, "deepseek_v41");
        list
    })
}

/// Unknown layouts never inherit another layout's measurements or pins.
pub fn v41_prewarm(ranks: usize, rank: usize) -> Option<&'static V41RankPrewarm> {
    v41_prewarm_list().layouts.iter().find(|list| list.layout == ranks)?
        .ranks.iter().find(|list| list.rank == rank)
}

/// Physical driver graph high-water, by coordinator rank. Rounded only by
/// the shared graph reserve policy; no prewarm allocation tail is added here.
pub fn v41_soft_cap(ranks: usize, rank: usize) -> u64 {
    match (ranks, rank) {
        // TP1: larger of the measured lag-16 charged and settled levels (#2460).
        (1, 0) => 1_673_527_296,
        // TP2: lag-8 max-card calibration (#2460), short C1/C4/C16 and 32K C1/C8.
        (2, 0 | 1) => 680 << 20,
        _ => 0,
    }
}

pub fn v41_prewarm_rows(ranks: usize, rank: usize, decode_rows: usize) -> Vec<V41PrewarmEntry> {
    v41_prewarm(ranks, rank).into_iter().flat_map(|list| &list.entries)
        .copied().filter(|entry| entry.rows <= decode_rows).collect()
}

/// Only traced replay demand qualifies prefill startup geometry. Production
/// windowed full chunks remain eager; a changed chunk never borrows their pins.
pub fn v41_prefill_rows(ranks: usize, rank: usize, capacity: usize, chunk: Option<usize>) -> Vec<usize> {
    let replay = cuteafd_core::coordinator_programs::V41_DECODER_REPLAY_ROWS as usize;
    v41_prewarm(ranks, rank).into_iter().flat_map(|list| &list.prefill_entries)
        .filter_map(|entry| (entry.rows <= capacity && (entry.rows == replay || Some(entry.rows) == chunk))
            .then_some(entry.rows)).collect()
}

/// Joined startup owners capture only geometry measured on every participant.
/// Rank-local lanes still consume their own inventory and pin policy.
pub fn v41_joint_prewarm_rows(ranks: usize, decode_rows: usize) -> Vec<usize> {
    v41_prewarm_rows(ranks, 0, decode_rows).into_iter().filter_map(|entry| {
        (1..ranks).all(|rank| v41_prewarm_rows(ranks, rank, decode_rows)
            .iter().any(|other| other.rows == entry.rows)).then_some(entry.rows)
    }).collect()
}

pub fn v41_prewarm_drafts(ranks: usize, rank: usize, requests: usize, maximum_width: usize) -> Vec<V41DraftPrewarmEntry> {
    v41_prewarm(ranks, rank).into_iter().flat_map(|list| &list.draft_entries).copied()
        .filter(|entry| entry.requests <= requests && entry.width <= maximum_width).collect()
}

pub fn v41_pinned_draft(ranks: usize, rank: usize, requests: usize, width: usize) -> bool {
    v41_prewarm(ranks, rank).is_some_and(|list| list.draft_entries.iter()
        .any(|entry| entry.requests == requests && entry.width == width && entry.pin))
}

pub fn v41_pinned_rows(ranks: usize, rank: usize, decode_rows: usize) -> Vec<usize> {
    v41_prewarm_rows(ranks, rank, decode_rows).into_iter().filter(|entry| entry.pin).map(|entry| entry.rows).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn checked_in_inventories_are_rank_specific_and_roundtrip() {
        let list = v41_prewarm_list();
        let _: V41PrewarmList = serde_json::from_value(serde_json::to_value(list).unwrap()).unwrap();
        for ranks in [1, 2] {
            for rank in 0..ranks {
                let inventory = v41_prewarm(ranks, rank).unwrap();
                assert!(inventory.reserve_base_bytes > 0);
                assert_eq!(inventory.reserve_base_bytes % (2 << 20), 0);
                assert_eq!(inventory.reserve_extra_bytes, 2 << 20);
                let rows = v41_prewarm_rows(ranks, rank, cuteafd_core::V41_VERIFY_ROWS);
                assert_eq!(rows, inventory.entries);
                let mut seen = std::collections::HashSet::new();
                assert!(rows.iter().all(|entry| entry.rows > 0 && entry.rows <= 64 && seen.insert(entry.rows)));
                assert!(v41_prewarm_rows(ranks, rank, 1).iter().all(|entry| entry.rows == 1));
                assert!(v41_prewarm_drafts(ranks, rank, 0, 7).is_empty());
                assert!(v41_prefill_rows(ranks, rank, 127, Some(2752)).is_empty());
                assert_eq!(v41_prefill_rows(ranks, rank, 4096, Some(2048)), vec![128]);
            }
        }
        assert_ne!(v41_prewarm(2, 0).unwrap().reserve_base_bytes, v41_prewarm(2, 1).unwrap().reserve_base_bytes);
        let joined = v41_joint_prewarm_rows(2, 64);
        for rank in 0..2 {
            assert!(joined.iter().all(|rows| v41_prewarm_rows(2, rank, 64)
                .iter().any(|entry| entry.rows == *rows)));
        }
    }
    #[test]
    fn unknown_layouts_have_no_borrowed_inventory() {
        for (ranks, rank) in [(0, 0), (3, 0), (1, 1), (2, 2)] {
            assert!(v41_prewarm(ranks, rank).is_none());
            assert!(v41_prewarm_rows(ranks, rank, 64).is_empty());
            assert!(v41_prefill_rows(ranks, rank, 4096, Some(2752)).is_empty());
            assert!(v41_prewarm_drafts(ranks, rank, 64, 7).is_empty());
            assert!(!v41_pinned_draft(ranks, rank, 1, 5));
        }
    }
}
