//! Spark expert partial replies (TP residency): every Spark returns its
//! partial down-proj output per routed (row, expert), as BF16 or NVFP4 (E2M1
//! with an E4M3 scale per 16). The format of a request's rows is decided by
//! that request's own row count in the step, never by the wave: a request
//! with at most `bf16_max_rows` rows in the step (every decode and verify
//! step at the default 64) replies BF16; a wider segment (prefill) replies
//! NVFP4. So one request's rows are bit-identical in any wave composition,
//! and verify matches plain decode (lossless speculation).
//!
//! A wave's reply holds a BF16 region and an NVFP4 region, each with its row
//! map (wave row of every region row); no request is ever converted. The
//! coordinator combine (`muse_tp_combine`) reads both regions.
//!
//! The cutoff is the shared worker's generic key
//! (`expert_partial_bf16_max_rows`); it must cover the widest verify step.
use anyhow::{ensure, Result};

/// Default cutoff: the sliding ring's rollback step (every decode/verify step).
pub const DEFAULT_BF16_MAX_ROWS: usize = super::cache::SWA_RING_STEP as usize;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PartialFormat { Bf16, Nvfp4 }

impl PartialFormat {
    /// Bytes of one E-wide partial row (NVFP4: values, then the per-16 scales).
    pub fn row_bytes(self, width: usize) -> usize {
        match self { Self::Bf16 => width * 2, Self::Nvfp4 => width / 2 + width / 16 }
    }
}

/// The validated cutoff.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Cutoff(usize);

impl Cutoff {
    /// `max_verify_rows`: the widest decode/verify step a request appends (1
    /// without a drafter). A lower cutoff would reply a verify step in NVFP4
    /// and make speculation lossy against plain BF16 decode.
    pub fn new(bf16_max_rows: usize, max_verify_rows: usize) -> Result<Self> {
        ensure!(bf16_max_rows >= max_verify_rows,
            "expert_partial_bf16_max_rows {bf16_max_rows} is below the {max_verify_rows}-row verify step: \
             verify replies would be NVFP4 while plain decode is BF16, so speculation would not be lossless");
        Ok(Self(bf16_max_rows))
    }

    pub fn rows(self) -> usize { self.0 }

    /// The format of a request contributing `rows` rows to a step.
    pub fn format(self, rows: usize) -> PartialFormat {
        if rows <= self.0 { PartialFormat::Bf16 } else { PartialFormat::Nvfp4 }
    }
}

/// One wave's reply regions: for each region, the wave rows it carries in order.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReplyPlan {
    pub bf16_rows: Vec<u32>,
    pub nvfp4_rows: Vec<u32>,
}

impl ReplyPlan {
    /// `segments`: each request's row count, in wave order (rows consecutive).
    pub fn new(cutoff: Cutoff, segments: &[usize]) -> Self {
        let mut plan = Self { bf16_rows: Vec::new(), nvfp4_rows: Vec::new() };
        let mut row = 0u32;
        for &rows in segments {
            let region = match cutoff.format(rows) {
                PartialFormat::Bf16 => &mut plan.bf16_rows,
                PartialFormat::Nvfp4 => &mut plan.nvfp4_rows,
            };
            region.extend(row..row + rows as u32);
            row += rows as u32;
        }
        plan
    }

    /// Reply bytes one rank sends (top-`topk` partials per row).
    pub fn rank_bytes(&self, topk: usize, width: usize) -> usize {
        topk * (self.bf16_rows.len() * PartialFormat::Bf16.row_bytes(width)
            + self.nvfp4_rows.len() * PartialFormat::Nvfp4.row_bytes(width))
    }
}

/// The largest reply one rank sends for any wave of at most `unit_rows` rows
/// from at most `sequences` requests: `k` BF16 segments of at most the cutoff
/// each, and (with a slot left) the remaining rows as one NVFP4 segment; the
/// maximum over `k`. Admission charges this, never all-BF16 at the unit.
pub fn max_rank_bytes(cutoff: Cutoff, unit_rows: usize, sequences: usize, topk: usize, width: usize) -> usize {
    let (bf16, nvfp4) = (PartialFormat::Bf16.row_bytes(width), PartialFormat::Nvfp4.row_bytes(width));
    (0..=sequences.min(unit_rows)).map(|k| {
        let bf16_rows = (k * cutoff.rows()).min(unit_rows);
        let rest = unit_rows - bf16_rows;
        // The rest replies NVFP4 only as one segment wider than the cutoff, in a free slot.
        let nvfp4_rows = if k < sequences && rest > cutoff.rows() { rest } else { 0 };
        topk * (bf16_rows * bf16 + nvfp4_rows * nvfp4)
    }).max().unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_request_keeps_its_format_in_any_wave() -> Result<()> {
        let cutoff = Cutoff::new(64, 8)?;
        // Decode (1), verify (8) and a 2752-row prefill segment, alone and packed.
        for (rows, expected) in [(1, PartialFormat::Bf16), (8, PartialFormat::Bf16), (64, PartialFormat::Bf16),
            (65, PartialFormat::Nvfp4), (2752, PartialFormat::Nvfp4)] {
            for others in [vec![], vec![1; 15], vec![2752], vec![64, 2000]] {
                let mut segments = others.clone();
                segments.push(rows);
                let plan = ReplyPlan::new(cutoff, &segments);
                let first = others.iter().sum::<usize>() as u32;
                let mine: Vec<u32> = (first..first + rows as u32).collect();
                let region = if expected == PartialFormat::Bf16 { &plan.bf16_rows } else { &plan.nvfp4_rows };
                assert!(mine.iter().all(|r| region.contains(r)), "{rows} rows beside {others:?}");
            }
        }
        // All-BF16 (cutoff at the unit) and all-NVFP4 above the verify step.
        assert_eq!(Cutoff::new(4096, 8)?.format(2752), PartialFormat::Bf16);
        assert_eq!(Cutoff::new(8, 8)?.format(9), PartialFormat::Nvfp4);
        let error = Cutoff::new(4, 8).unwrap_err().to_string();
        assert!(error.contains("verify") && error.contains("lossless"), "{error}");
        Ok(())
    }

    /// The admission bound covers every wave composition (exhaustive on a small geometry).
    #[test]
    fn max_rank_bytes_bounds_every_wave() -> Result<()> {
        let (unit, sequences, topk, width) = (12usize, 3usize, 2usize, 32usize);
        for c in [1usize, 2, 4, 12, 64] {
            let cutoff = Cutoff::new(c, 1)?;
            let bound = max_rank_bytes(cutoff, unit, sequences, topk, width);
            let mut worst = 0;
            // Every composition of up to `sequences` segments totalling at most `unit` rows.
            fn walk(left: usize, slots: usize, at: &mut Vec<usize>, f: &mut dyn FnMut(&[usize])) {
                f(at);
                if slots == 0 { return; }
                for rows in 1..=left { at.push(rows); walk(left - rows, slots - 1, at, f); at.pop(); }
            }
            walk(unit, sequences, &mut Vec::new(), &mut |segments| {
                worst = worst.max(ReplyPlan::new(cutoff, segments).rank_bytes(topk, width));
            });
            assert_eq!(worst, bound, "cutoff {c}");
        }
        // Release numbers: 2752-row unit, 16 sequences, top-8, E 4096.
        let release = max_rank_bytes(Cutoff::new(64, 8)?, 2752, 16, 8, 4096);
        // 15 decode/verify segments of 64 BF16 rows beside one 1792-row NVFP4 prefill segment.
        assert_eq!(release, 8 * (960 * 8192 + 1792 * 2304));
        assert_eq!(max_rank_bytes(Cutoff::new(4096, 8)?, 2752, 16, 8, 4096), 8 * 2752 * 8192);
        Ok(())
    }
}
