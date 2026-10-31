//! Host metadata of one Muse step (a prefill unit or a decode/verify batch):
//! the rows' positions, the full layers' page tables and the sliding layers'
//! ring addressing, exactly as the attention programs read them
//! (`_mimo_kernels.MimoGqaAttention._resolve`, `MimoRingCommit`):
//!
//! - a sliding row at position `p` of a sequence whose rings start at ring row
//!   `base` (`ring index * ring_rows`) has `ring_slots[row] = base + p % ring_rows`
//!   and `seq_first[row]` = the step row of its sequence's first row; keys at or
//!   past that first row's position come from this step's records, older keys
//!   from ring slot `base + k % ring_rows` (the [`MuseStateLayout`] slots, so
//!   prefix marks and kernels agree on where position `k` lives);
//! - a full row reads key `k` at page `page_table[row * stride + k / 64]`.
//!
//! Rows of one sequence are consecutive with consecutive positions.
//!
//! [`MuseStateLayout`]: cuteafd_loader::families::muse_spark::cache::MuseStateLayout
use anyhow::{ensure, Context, Result};

pub(crate) const PAGE_ROWS: usize = 64;

/// One sequence's rows in a step.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct StepSequence<'p> {
    /// Its sequence slot (ring index).
    pub slot: usize,
    /// Position of its first row in this step (its committed length).
    pub start: usize,
    pub rows: usize,
    /// Its pages in row order, covering at least `start + rows` positions.
    pub pages: &'p [u32],
}

/// Device-ready metadata of a step, one row per entry.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct StepMetadata {
    pub positions: Vec<i64>,
    pub ring_slots: Vec<i64>,
    pub seq_first: Vec<i32>,
    /// Full layers' K|V record index per row in the paged cache
    /// (`page * 64 + p % 64`), the `kv_slots` of `muse_qk_norm_nope_full`.
    pub page_slots: Vec<i64>,
    /// `[rows, stride]` page ids (prefill: one sequence, a single shared row).
    pub page_table: Vec<i32>,
    pub table_stride: usize,
}

impl StepMetadata {
    /// SWA layers write each row's record into the step's own `kv_step` row.
    pub(crate) fn step_slots(&self) -> Vec<i64> { (0..self.positions.len() as i64).collect() }

    /// `ring_rows`: rows of one sequence's ring; `slots`: rings admitted.
    /// `shared_table`: the prefill route reads one table row for its one sequence.
    pub(crate) fn build(sequences: &[StepSequence<'_>], ring_rows: usize, slots: usize, step_capacity: usize,
        shared_table: bool) -> Result<Self> {
        ensure!(!sequences.is_empty() && ring_rows > 0, "Muse step without sequences or rings");
        ensure!(!shared_table || sequences.len() == 1, "Muse prefill route runs one sequence per launch");
        let rows = sequences.iter().try_fold(0usize, |n, s| n.checked_add(s.rows)).context("Muse step rows")?;
        ensure!((1..=step_capacity).contains(&rows), "Muse step of {rows} rows outside 1..={step_capacity}");
        let stride = sequences.iter().map(|s| s.pages.len()).max().unwrap_or(0).max(1);
        let mut seen = std::collections::HashSet::new();
        let mut out = Self { positions: Vec::with_capacity(rows), ring_slots: Vec::with_capacity(rows),
            seq_first: Vec::with_capacity(rows), page_slots: Vec::with_capacity(rows), page_table: Vec::new(),
            table_stride: stride };
        for sequence in sequences {
            ensure!(sequence.rows > 0 && sequence.slot < slots && seen.insert(sequence.slot),
                "Muse step sequence slot {} invalid or repeated", sequence.slot);
            let end = sequence.start.checked_add(sequence.rows).context("Muse step positions")?;
            ensure!(end.div_ceil(PAGE_ROWS) <= sequence.pages.len(),
                "Muse sequence {} lacks pages for {end} positions", sequence.slot);
            // A step's own keys come from its records and the commit runs after attention,
            // so a prefill unit may be longer than the ring (`cache::SWA_RING_STEP`). A
            // rollback step (decode/verify, `!shared_table`) writes past the committed
            // frontier and must leave the visible window intact.
            ensure!(shared_table || sequence.rows as u64 <= cuteafd_loader::families::muse_spark::cache::SWA_RING_STEP,
                "Muse decode/verify step of {} rows exceeds the ring's rollback step", sequence.rows);
            let first = i32::try_from(out.positions.len())?;
            let base = (sequence.slot * ring_rows) as i64;
            let mut table = sequence.pages.iter().map(|&p| i32::try_from(p)).collect::<Result<Vec<_>, _>>()?;
            table.resize(stride, -1);
            for p in sequence.start..end {
                out.positions.push(i64::try_from(p)?);
                out.ring_slots.push(base + (p % ring_rows) as i64);
                out.seq_first.push(first);
                out.page_slots.push(i64::from(sequence.pages[p / PAGE_ROWS]) * PAGE_ROWS as i64 + (p % PAGE_ROWS) as i64);
                if !shared_table { out.page_table.extend_from_slice(&table); }
            }
            if shared_table { out.page_table = table; }
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cuteafd_loader::families::muse_spark::cache::{MuseCacheLayout, MuseStateLayout};

    /// The kernel's key resolution, on the host: where key `k` of `row` is read.
    fn key_source(meta: &StepMetadata, row: usize, k: i64, ring_rows: usize) -> (bool, i64) {
        let first = meta.seq_first[row] as usize;
        let seq_start = meta.positions[first];
        let p = meta.positions[row];
        let ring_base = meta.ring_slots[row] - p % ring_rows as i64;
        if k >= seq_start { (true, first as i64 + k - seq_start) } else { (false, ring_base + k % ring_rows as i64) }
    }

    #[test]
    fn ring_slots_agree_with_the_prefix_state_layout() -> Result<()> {
        let cfg = cuteafd_loader::families::muse_spark::MuseSparkConfig::from_hf(
            &cuteafd_loader::plan::testing::muse_spark_config())?;
        let cache = MuseCacheLayout::new(&cfg, 1)?;
        let state = MuseStateLayout::new(&cfg, &cache, 16)?;
        let ring = state.ring.rows as usize;
        let pages: Vec<u32> = (100..400).collect();
        for (start, rows) in [(0, 1), (0, 2752), (5000, 2752), (ring * 3 - 7, 64), (10_000, 1), (100, 4096)] {
            let meta = StepMetadata::build(&[StepSequence { slot: 5, start, rows, pages: &pages }], ring, 16, 4096, true)?;
            for (row, &p) in meta.positions.iter().enumerate() {
                // The commit writes p's record where the mark layout expects it.
                assert_eq!(meta.ring_slots[row], (5 * ring) as i64 + p % ring as i64);
                // Every key in the window that predates the step is read from that same ring slot.
                for k in (p - (cfg.window as i64 - 1)).max(0)..=p {
                    let (step, at) = key_source(&meta, row, k, ring);
                    if k < start as i64 {
                        assert!(!step && at == (5 * ring) as i64 + k % ring as i64);
                    } else {
                        assert!(step && meta.positions[at as usize] == k);
                    }
                }
            }
        }
        Ok(())
    }

    #[test]
    fn decode_batches_keep_each_sequence_on_its_own_ring_and_pages() -> Result<()> {
        let (a, b): (Vec<u32>, Vec<u32>) = ((0..3).collect(), (10..12).collect());
        let meta = StepMetadata::build(&[StepSequence { slot: 0, start: 130, rows: 4, pages: &a },
            StepSequence { slot: 3, start: 64, rows: 1, pages: &b }], 2112, 4, 64, false)?;
        assert_eq!(meta.positions, [130, 131, 132, 133, 64]);
        assert_eq!(meta.seq_first, [0, 0, 0, 0, 4]);
        assert_eq!(meta.ring_slots[4], 3 * 2112 + 64);
        assert_eq!(meta.table_stride, 3);
        assert_eq!(&meta.page_table[..3], [0, 1, 2]);
        assert_eq!(&meta.page_table[12..], [10, 11, -1]);
        Ok(())
    }

    #[test]
    fn malformed_steps_refuse_before_any_launch() {
        let pages: Vec<u32> = vec![0];
        let one = |slot, start, rows| StepSequence { slot, start, rows, pages: &pages };
        assert!(StepMetadata::build(&[], 2112, 4, 64, false).is_err());
        assert!(StepMetadata::build(&[one(4, 0, 1)], 2112, 4, 64, false).is_err(), "slot outside rings");
        assert!(StepMetadata::build(&[one(0, 0, 1), one(0, 1, 1)], 2112, 4, 64, false).is_err(), "repeated slot");
        assert!(StepMetadata::build(&[one(0, 60, 8)], 2112, 4, 64, false).is_err(), "lacks pages");
        assert!(StepMetadata::build(&[one(0, 0, 65)], 2112, 4, 64, false).is_err(), "over capacity");
        let many: Vec<u32> = (0..2).collect();
        assert!(StepMetadata::build(&[StepSequence { slot: 0, start: 0, rows: 65, pages: &many }], 2112, 4, 4096, false)
            .is_err(), "a verify step past the rollback step");
        assert!(StepMetadata::build(&[one(0, 0, 1), one(1, 0, 1)], 2112, 4, 64, true).is_err(), "prefill is one sequence");
    }

    /// Records written through `kv_slots` land where the attention reads them:
    /// a full layer's key `k` of a row is read at `page_table[k / 64] * 64 + k % 64`,
    /// which is the slot the row at position `k` wrote; a sliding row's own-step
    /// keys come from `kv_step[seq_first + k - start]`, the slot the SWA record write
    /// used. Two prefills of the same tokens on different pages write identical
    /// records at their own slots (restore byte-identity rests on this).
    #[test]
    fn records_land_where_attention_and_marks_read_them() -> Result<()> {
        let record = |p: i64| -> [u8; 4] { (p as u32).wrapping_mul(2654435761).to_le_bytes() };
        let pages_a: Vec<u32> = vec![7, 2, 9];
        let pages_b: Vec<u32> = vec![4, 11, 0];
        let mut cache = vec![[0u8; 4]; 12 * 64];
        fn write(cache: &mut [[u8; 4]], record: impl Fn(i64) -> [u8; 4], pages: &[u32], start: usize, rows: usize)
            -> Result<StepMetadata> {
            let meta = StepMetadata::build(&[StepSequence { slot: 0, start, rows, pages }], 2112, 1, 4096, true)?;
            for (row, &slot) in meta.page_slots.iter().enumerate() {
                cache[slot as usize] = record(meta.positions[row]);
            }
            Ok(meta)
        }
        let meta = write(&mut cache, record, &pages_a, 0, 130)?;
        // Every key the last row reads (all earlier positions) is its own position's record.
        for k in 0..130i64 {
            let page = meta.page_table[(k / 64) as usize] as usize;
            assert_eq!(cache[page * 64 + (k % 64) as usize], record(k));
        }
        // The same tokens prefilled onto other pages: identical records at their own slots.
        write(&mut cache, record, &pages_b, 0, 100)?;
        write(&mut cache, record, &pages_b, 100, 30)?;
        for k in 0..130usize {
            assert_eq!(cache[pages_a[k / 64] as usize * 64 + k % 64], cache[pages_b[k / 64] as usize * 64 + k % 64]);
        }
        // SWA: own-step keys at kv_step[seq_first + k - start] = the row's step slot.
        let swa = StepMetadata::build(&[StepSequence { slot: 0, start: 500, rows: 8, pages: &[0u32; 9] }], 2112, 1, 64, false)?;
        let slots = swa.step_slots();
        for (row, &p) in swa.positions.iter().enumerate() {
            let first = swa.seq_first[row] as i64;
            assert_eq!(slots[(first + p - swa.positions[first as usize]) as usize], row as i64);
        }
        Ok(())
    }
}
