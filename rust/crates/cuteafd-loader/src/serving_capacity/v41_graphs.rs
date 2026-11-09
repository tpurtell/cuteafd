//! V4.1 graph prewarm: one checked-in list drives the admission reserve and
//! the daemon's startup capture. Keys are decode rows; every bank on a device
//! shares them, so a row entry warms each site's geometry at that width.

/// One decode width the startup capture warms, and whether it is pinned.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct V41PrewarmEntry {
    pub rows: usize,
    pub pin: bool,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct V41PrewarmList {
    pub schema_version: u32,
    pub family: String,
    /// Measured arena high-water for the whole live graph set. Bootstrap from
    /// the #1635 census10 lazy growth (1,046,478,848 bytes) until the trace card
    /// measures this family's warmed arena on the qualification config.
    pub reserve_base_bytes: u64,
    /// One largest capture, the peak a capture adds before its victim is freed.
    pub reserve_extra_bytes: u64,
    pub entries: Vec<V41PrewarmEntry>,
}

pub fn v41_prewarm_list() -> V41PrewarmList {
    let list: V41PrewarmList = serde_json::from_str(include_str!("v41_prewarm.json"))
        .expect("checked-in V4.1 prewarm list");
    assert_eq!(list.schema_version, 1);
    assert_eq!(list.family, "deepseek_v41");
    list
}

/// Decode widths the startup capture warms. The full prefill chunk and the
/// 128-row decoder replay are added at runtime because the chunk is a launch
/// argument, not a checked-in constant.
pub fn v41_prewarm_rows(decode_rows: usize) -> Vec<V41PrewarmEntry> {
    v41_prewarm_list().entries.into_iter().filter(|entry| entry.rows <= decode_rows).collect()
}

/// The decode rows that must survive eviction: the C1 width the sweep shows is
/// always hot. The 128-row decoder replay and the prefill chunk are pinned by
/// their capture sites, which know the live chunk width.
pub fn v41_pinned_rows(decode_rows: usize) -> Vec<usize> {
    v41_prewarm_rows(decode_rows).into_iter().filter(|entry| entry.pin).map(|entry| entry.rows).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    #[test]
    fn checked_in_list_roundtrips_and_matches_the_startup_set() {
        let list = v41_prewarm_list();
        assert!(!list.entries.is_empty());
        let mut rows = HashSet::new();
        for entry in &list.entries {
            assert!(rows.insert(entry.rows), "duplicate prewarm width");
            assert!(entry.rows > 0 && entry.rows <= cuteafd_core::V41_VERIFY_ROWS);
        }
        assert_eq!(list.reserve_base_bytes % (2 << 20), 0);
        assert_eq!(list.reserve_extra_bytes, 2 << 20);
        // One entry per decode width the startup capture enumerates.
        assert_eq!(v41_prewarm_rows(cuteafd_core::V41_VERIFY_ROWS).len(), cuteafd_core::V41_VERIFY_ROWS);
        assert_eq!(v41_prewarm_rows(8).len(), 8);
        assert_eq!(v41_pinned_rows(cuteafd_core::V41_VERIFY_ROWS), vec![1]);
        let _: V41PrewarmList = serde_json::from_value(serde_json::to_value(&list).unwrap()).unwrap();
    }

}
