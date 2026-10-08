//! A retained snapshot on the device and what follows it.
use super::marks::MarkSlot;
use cuteafd_core::prefix::SnapshotKind;
use cuteafd_hostcache::cache::StoreTicket;
use std::sync::Arc;

pub type EntryId = u64;

/// What follows a snapshot: the first token after an exact-length restore needs no forward
/// (every generic engine needs at least one row to prefill). `greedy` is the argmax where it is
/// known; `logits` the last row, so a sampled request is served too.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct After {
    pub greedy: Option<u32>,
    pub logits: Option<Arc<[f32]>>,
}

impl After {
    /// From the row that follows the snapshot: its argmax (lowest index of the largest non-NaN
    /// value, 0 when every value is NaN) and, with `keep`, the row itself.
    pub fn from_logits(logits: &[f32], keep: bool) -> Self {
        Self { greedy: Some(greedy(logits)), logits: keep.then(|| Arc::from(logits)) }
    }
    /// Whether a request resuming exactly at the snapshot gets its first token without a forward.
    pub fn serves(&self, sampled: bool) -> bool {
        if sampled { self.logits.is_some() } else { self.greedy.is_some() || self.logits.is_some() }
    }
    pub fn host_bytes(&self) -> usize {
        self.logits.as_ref().map_or(0, |l| l.len() * 4)
    }
}

pub fn greedy(logits: &[f32]) -> u32 {
    let mut best: Option<(f32, usize)> = None;
    for (i, &v) in logits.iter().enumerate() {
        if !v.is_nan() && best.is_none_or(|(b, _)| v > b) {
            best = Some((v, i));
        }
    }
    best.map_or(0, |(_, i)| i as u32)
}

/// Where a snapshot's positional mark lives ([`super::MarkStore`]).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Mark {
    /// A slot of the family's device arena.
    Slot(MarkSlot),
    /// Pool pages the snapshot owns alone (never shared), in ascending order, which is the
    /// order the family lays the mark out in.
    Pages(Vec<u32>),
}

/// A device snapshot: `tokens.len()` rows of `pages` (every page immutable: full pages are
/// shared with their writer, the partial tail is the entry's own copy) and its mark.
pub(crate) struct Entry {
    pub tokens: Vec<u32>,
    pub media: Vec<cuteafd_core::MediaSpan>,
    pub kind: SnapshotKind,
    pub pages: Vec<u32>,
    pub mark: Option<Mark>,
    pub after: After,
    pub last_use: u64,
    /// The host tier's write-behind copy, when one was issued.
    pub ticket: Option<StoreTicket>,
}

impl Entry {
    pub fn len(&self) -> usize {
        self.tokens.len()
    }
    /// The pool pages of a pool-page mark (none for an arena slot).
    pub fn mark_pages(&self) -> &[u32] {
        match &self.mark {
            Some(Mark::Pages(pages)) => pages,
            _ => &[],
        }
    }
}

/// Which snapshot the device evicts next: least recently used; at equal use a prompt snapshot
/// before a turn snapshot (a prompt snapshot shares its pages with its conversation's turn, so
/// evicting it first frees its mark and keeps the longer state); then the oldest id.
/// hughmadden/glm53f-afd `victim`: prompts-before-turns deleted fresh prompt snapshots under page
/// pressure and made retries prefill from cold.
pub fn victim(entries: impl Iterator<Item = (EntryId, SnapshotKind, u64)>) -> Option<EntryId> {
    entries.min_by_key(|&(id, kind, last)| (last, kind == SnapshotKind::Turn, id)).map(|(id, _, _)| id)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn victim_is_least_recent_then_prompt_first() {
        use SnapshotKind::*;
        assert_eq!(victim([(1, Turn, 3), (2, Prompt, 9), (3, Turn, 10)].into_iter()), Some(1));
        assert_eq!(victim([(1, Turn, 5), (2, Prompt, 5)].into_iter()), Some(2));
        assert_eq!(victim(std::iter::empty()), None);
    }

    #[test]
    fn after_serves_greedy_and_sampled_requests() {
        let after = After::from_logits(&[0.5, f32::NAN, 2.0, 2.0], false);
        assert_eq!(after.greedy, Some(2));
        assert!(after.serves(false) && !after.serves(true));
        assert!(After::from_logits(&[1.0], true).serves(true));
        assert!(!After::default().serves(false));
        assert_eq!(greedy(&[f32::NAN, f32::NAN]), 0);
    }
}
