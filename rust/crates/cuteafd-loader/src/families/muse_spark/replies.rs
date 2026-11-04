//! Phase-aware Muse partial reply regions and their admitted capacity.
use anyhow::{ensure, Result};

/// Default cutoff: the sliding ring's rollback step (every decode/verify step).
pub const DEFAULT_BF16_MAX_ROWS: usize = super::cache::SWA_RING_STEP as usize;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PartialFormat { Bf16, Nvfp4, Fp8 }

impl PartialFormat {
    /// Bytes of one E-wide partial record: values, then the trailing scales.
    /// BF16 `2E`; NVFP4 `E/2 + E/16` (E4M3 per 16); FP8 `E + E/32` (UE8M0 per 32).
    pub fn row_bytes(self, width: usize) -> usize {
        match self { Self::Bf16 => width * 2, Self::Nvfp4 => width / 2 + width / 16, Self::Fp8 => width + width / 32 }
    }
}

/// Prefill-wide partial encoding. Omission retains NVFP4 above the cutoff;
/// explicit phase selections also support BF16 and FP8 E4M3 per-32.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum WideFormat { Bf16, #[default] Nvfp4, Fp8 }

impl WideFormat {
    pub fn format(self) -> PartialFormat {
        match self { Self::Bf16 => PartialFormat::Bf16, Self::Nvfp4 => PartialFormat::Nvfp4, Self::Fp8 => PartialFormat::Fp8 }
    }
}

impl From<cuteafd_core::SparkReplyFormat> for WideFormat {
    fn from(format: cuteafd_core::SparkReplyFormat) -> Self {
        match format { cuteafd_core::SparkReplyFormat::Bf16 => Self::Bf16,
            cuteafd_core::SparkReplyFormat::Fp8 => Self::Fp8,
            cuteafd_core::SparkReplyFormat::Nvfp4 => Self::Nvfp4 }
    }
}

/// The validated cutoff and the format above it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Cutoff(usize, WideFormat, cuteafd_core::spark_reply::SparkReplyFormats);

impl Cutoff {
    /// `max_verify_rows`: the widest decode/verify step a request appends (1
    /// without a drafter). Admission bounds each decode segment by this cutoff,
    /// independently of an explicit phase encoding.
    pub fn new(bf16_max_rows: usize, max_verify_rows: usize) -> Result<Self> {
        Self::with_wide(bf16_max_rows, max_verify_rows, WideFormat::default())
    }

    pub fn with_wide(bf16_max_rows: usize, max_verify_rows: usize, wide: WideFormat) -> Result<Self> {
        ensure!(bf16_max_rows >= 1, "expert_partial_bf16_max_rows must admit at least one decode row");
        ensure!(bf16_max_rows >= max_verify_rows,
            "expert_partial_bf16_max_rows {bf16_max_rows} is below the {max_verify_rows}-row verify step: \
             the admitted reply capacity would not cover every verify row");
        Ok(Self(bf16_max_rows, wide, Default::default()))
    }

    pub fn with_formats(bf16_max_rows: usize, max_verify_rows: usize,
        formats: cuteafd_core::spark_reply::SparkReplyFormats) -> Result<Self> {
        let mut cutoff = Self::with_wide(bf16_max_rows, max_verify_rows, formats.prefill(true).into())?;
        cutoff.2 = formats;
        Ok(cutoff)
    }

    pub fn phase_format(self, rows: usize, prefill: bool) -> PartialFormat {
        let selected = if prefill { self.2.prefill } else { self.2.decode };
        match selected {
            Some(format) => WideFormat::from(format).format(),
            None if prefill => self.format(rows),
            None => PartialFormat::Bf16,
        }
    }

    pub fn rows(self) -> usize { self.0 }

    /// The format of a request contributing `rows` rows to a step.
    pub fn format(self, rows: usize) -> PartialFormat {
        if rows <= self.0 { PartialFormat::Bf16 } else { self.1.format() }
    }

    pub fn wide(self) -> WideFormat { self.1 }
}

/// One wave's reply regions: for each region, the wave rows it carries in
/// order. `wide_rows` are in the cutoff's wide format (NVFP4 or FP8); a wave
/// may additionally carry the other compressed format in `other_rows`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReplyPlan {
    pub bf16_rows: Vec<u32>,
    pub wide_rows: Vec<u32>,
    pub other_rows: Vec<u32>,
    pub wide: PartialFormat,
}

impl ReplyPlan {
    /// `segments`: each request's row count, in wave order (rows consecutive).
    pub fn new(cutoff: Cutoff, segments: &[usize]) -> Self {
        Self::for_phase(cutoff, segments, true)
    }

    pub fn for_phase(cutoff: Cutoff, segments: &[usize], prefill: bool) -> Self {
        let mut plan = Self { bf16_rows: Vec::new(), wide_rows: Vec::new(), other_rows: Vec::new(), wide: match cutoff.wide() { WideFormat::Bf16 => PartialFormat::Nvfp4, wide => wide.format() } };
        let mut row = 0u32;
        for &rows in segments {
            let region = match cutoff.phase_format(rows, prefill) {
                PartialFormat::Bf16 => &mut plan.bf16_rows,
                format if format == plan.wide => &mut plan.wide_rows,
                _ => &mut plan.other_rows,
            };
            region.extend(row..row + rows as u32);
            row += rows as u32;
        }
        plan
    }

    pub fn other(&self) -> PartialFormat {
        match self.wide { PartialFormat::Nvfp4 => PartialFormat::Fp8, _ => PartialFormat::Nvfp4 }
    }

    pub fn rows(&self) -> usize { self.bf16_rows.len() + self.wide_rows.len() + self.other_rows.len() }

    /// Reply bytes one rank sends (top-`topk` partials per row).
    pub fn rank_bytes(&self, topk: usize, width: usize) -> usize {
        [(self.bf16_rows.len(), PartialFormat::Bf16), (self.wide_rows.len(), self.wide),
            (self.other_rows.len(), self.other())].into_iter().fold(0usize, |bytes, (rows, format)| {
                if rows == 0 { bytes } else { bytes.next_multiple_of(256) + topk * rows * format.row_bytes(width) }
            })
    }
}

/// Largest reachable reply across the prefill and decode phase selections.
/// Explicit phases use one region; omitted prefill can mix cutoff-bounded
/// BF16 segments with one wide segment, including inter-region alignment.
pub fn max_rank_bytes(cutoff: Cutoff, unit_rows: usize, sequences: usize, topk: usize, width: usize) -> usize {
    let prefill = match cutoff.2.prefill {
        Some(format) => topk * unit_rows * WideFormat::from(format).format().row_bytes(width),
        None => {
            let (bf16, wide) = (PartialFormat::Bf16.row_bytes(width), cutoff.wide().format().row_bytes(width));
            let all_bf16 = unit_rows.min(sequences.saturating_mul(cutoff.rows())) * topk * bf16;
            if sequences == 0 { 0 } else if cutoff.wide() == WideFormat::Bf16 { unit_rows * topk * bf16 } else {
                // Alignment can make a partly filled narrow segment larger than
                // its full-cutoff neighbour. Check every reachable narrow count.
                (0..=unit_rows.min(sequences.saturating_sub(1).saturating_mul(cutoff.rows())))
                    .filter_map(|bf16_rows| {
                        let wide_rows = unit_rows - bf16_rows;
                        (wide_rows > cutoff.rows()).then(|| (topk * bf16_rows * bf16).next_multiple_of(256)
                            + topk * wide_rows * wide)
                    }).max().unwrap_or(0).max(all_bf16)
            }
        }
    };
    // Explicit units have one region, even when the two phases select different
    // compressed formats. Only omitted prefill can mix BF16 and wide regions.
    let decode_rows = unit_rows.min(sequences.saturating_mul(cutoff.rows()));
    let decode = topk * decode_rows * WideFormat::from(cutoff.2.decode()).format().row_bytes(width);
    prefill.max(decode)
}

/// The shared Spark intake's per-row plane width for partial replies: the
/// intake allocates `rows x row_bytes` per rank and slot and lands each frame
/// at offset 0 with its own length (`cuteafd_pull_frames` bounds by capacity),
/// so a row width of `ceil(max_rank_bytes / unit_rows)` (16-byte aligned)
/// covers every region mix the admission allows. Planner and runtime use this.
pub fn intake_row_bytes(cutoff: Cutoff, unit_rows: usize, sequences: usize, topk: usize, width: usize) -> usize {
    max_rank_bytes(cutoff, unit_rows, sequences, topk, width).div_ceil(unit_rows.max(1)).next_multiple_of(16)
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
                let region = if expected == PartialFormat::Bf16 { &plan.bf16_rows } else { &plan.wide_rows };
                assert!(mine.iter().all(|r| region.contains(r)), "{rows} rows beside {others:?}");
            }
        }
        // All-BF16 (cutoff at the unit) and all-NVFP4 above the verify step.
        assert_eq!(Cutoff::new(4096, 8)?.format(2752), PartialFormat::Bf16);
        assert_eq!(Cutoff::new(8, 8)?.format(9), PartialFormat::Nvfp4);
        assert!(Cutoff::new(0, 0).is_err() && Cutoff::new(0, 1).is_err());
        assert!(Cutoff::new(1, 0).is_ok());
        let error = Cutoff::new(4, 8).unwrap_err().to_string();
        assert!(error.contains("verify") && error.contains("capacity"), "{error}");
        Ok(())
    }

    /// The admission bound covers every wave composition (exhaustive on a small geometry).
    #[test]
    fn max_rank_bytes_bounds_every_wave() -> Result<()> {
        let (unit, sequences, topk, width) = (12usize, 3usize, 2usize, 32usize);
        for (c, wide) in [1usize, 2, 4, 12, 64].into_iter().flat_map(|c| [(c, WideFormat::Nvfp4), (c, WideFormat::Fp8)]) {
            let cutoff = Cutoff::with_wide(c, 1, wide)?;
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
            assert_eq!(worst, bound, "cutoff {c} {wide:?}");
        }
        // Release numbers: 2752-row unit, 16 sequences, top-8, E 4096.
        let release = max_rank_bytes(Cutoff::new(64, 8)?, 2752, 16, 8, 4096);
        // 15 decode/verify segments of 64 BF16 rows beside one 1792-row NVFP4 prefill segment.
        assert_eq!(release, 8 * (960 * 8192 + 1792 * 2304));
        assert_eq!(max_rank_bytes(Cutoff::new(4096, 8)?, 2752, 16, 8, 4096), 8 * 2752 * 8192);
        // FP8 above the cutoff: 4224-byte records (4096 E4M3 + 128 UE8M0).
        assert_eq!(PartialFormat::Fp8.row_bytes(4096), 4224);
        assert_eq!(max_rank_bytes(Cutoff::with_wide(64, 8, WideFormat::Fp8)?, 2752, 16, 8, 4096),
            8 * (960 * 8192 + 1792 * 4224));
        assert_eq!(WideFormat::from(cuteafd_core::SparkReplyFormat::Fp8), WideFormat::Fp8);
        let bf16 = ReplyPlan::new(Cutoff::with_wide(64, 8, WideFormat::Bf16)?, &[1, 65, 2752]);
        assert_eq!(bf16.bf16_rows.len(), 2818);
        assert!(bf16.wide_rows.is_empty());
        Ok(())
    }

    #[test]
    fn every_explicit_phase_pair_is_honoured_and_admitted() -> Result<()> {
        use cuteafd_core::{SparkReplyFormat::{Bf16, Fp8, Nvfp4}, spark_reply::SparkReplyFormats};
        for prefill in [Bf16, Fp8, Nvfp4] {
            for decode in [Bf16, Fp8, Nvfp4] {
                let cutoff = Cutoff::with_formats(64, 8, SparkReplyFormats {
                    prefill: Some(prefill), decode: Some(decode) })?;
                let (unit, sequences, topk, width) = (2752, 16, 8, 4096);
                let expected = (unit * prefill.row_bytes(width).unwrap())
                    .max(sequences * 64 * decode.row_bytes(width).unwrap()) * topk;
                assert_eq!(max_rank_bytes(cutoff, unit, sequences, topk, width), expected);
                for rows in [1, 8, 64, 65, unit] {
                    for (is_prefill, format) in [(true, prefill), (false, decode)] {
                        let plan = ReplyPlan::for_phase(cutoff, &[rows], is_prefill);
                        assert_eq!(plan.rank_bytes(topk, width), rows * topk * format.row_bytes(width).unwrap());
                        assert_eq!(cutoff.phase_format(rows, is_prefill), WideFormat::from(format).format());
                    }
                }
                let mixed = ReplyPlan::for_phase(cutoff, &[1, 8, 65, 2000], true);
                assert_eq!(mixed.rank_bytes(topk, width), 2074 * topk * prefill.row_bytes(width).unwrap());
            }
        }
        Ok(())
    }

    #[test]
    fn intake_rows_cover_the_worst_reply_without_all_bf16() -> Result<()> {
        let cutoff = Cutoff::new(64, 8)?;
        let row = intake_row_bytes(cutoff, 2752, 16, 8, 4096);
        assert!(row * 2752 >= max_rank_bytes(cutoff, 2752, 16, 8, 4096) && row % 16 == 0);
        assert!(row < 8 * 8192, "{row} would be the all-BF16 charge");
        // A one-row decode lane holds its single BF16 row exactly.
        assert_eq!(intake_row_bytes(cutoff, 1, 1, 8, 4096), 8 * 8192);
        Ok(())
    }
}
