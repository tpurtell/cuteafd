//! Device-sized launch policies for coordinator kernels.
use std::num::NonZeroUsize;

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub(crate) enum SmCountError {
    #[error("device reported zero multiprocessors")]
    Unavailable,
    #[error("--sms must be between 1 and the device's {actual} multiprocessors, got {requested}")]
    InvalidLimit { requested: usize, actual: usize },
}

/// Native AOT counterpart of SparkInfer dense MLA's `_wave_balanced_splits`.
/// Minimize idle SMs in the final wave (up to three waves), preferring fewer
/// splits on a tie.
pub(crate) fn mla_wave_balanced_splits(chunks: usize, tiles: usize, sms: usize, max_splits: usize) -> usize {
    let chunks = chunks.max(1);
    let sms = sms.max(1);
    let mut best = (usize::MAX, 1);
    for per_cta in 1..=chunks {
        let splits = chunks.div_ceil(per_cta);
        let active = tiles.max(1) * splits;
        if active.div_ceil(sms) <= 3 {
            best = best.min(((sms - active % sms) % sms, splits));
        }
    }
    best.1.min(max_splits.max(1))
}

/// Cached at engine creation, so captured and ordinary launches use the same
/// device geometry without querying CUDA on the request path.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Fp8QuantizeGrid(NonZeroUsize);

impl Fp8QuantizeGrid {
    pub fn new(actual_sms: usize, limit: Option<usize>) -> Result<Self, SmCountError> {
        let actual = NonZeroUsize::new(actual_sms).ok_or(SmCountError::Unavailable)?;
        let selected = match limit {
            Some(requested) => NonZeroUsize::new(requested).filter(|sms| sms.get() <= actual.get())
                .ok_or(SmCountError::InvalidLimit { requested, actual: actual.get() })?,
            None => actual,
        };
        Ok(Self(selected))
    }

    /// Eight 256-column tiles per block, at most four blocks per SM. The
    /// exported kernel strides over any work beyond the chosen grid.
    pub fn blocks(self, rows: usize, hidden: usize) -> usize {
        rows.saturating_mul(hidden.div_ceil(256)).div_ceil(8)
            .clamp(1, self.0.get().saturating_mul(4))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mla_wave_balance_follows_live_sms_and_the_export_cap() {
        assert_eq!(mla_wave_balanced_splits(1, 4, 188, 16), 1);
        assert_eq!(mla_wave_balanced_splits(32, 4, 188, 16), 16);
        assert_eq!(mla_wave_balanced_splits(32, 4, 32, 16), 8);
        assert_eq!(mla_wave_balanced_splits(9, 8, 188, 8), 8);
        assert_eq!(mla_wave_balanced_splits(32, 1024, 188, 16), 1);
    }

    #[test]
    fn workload_rounds_up_tiles_and_blocks() {
        let grid = Fp8QuantizeGrid::new(188, None).unwrap();
        for (rows, hidden, blocks) in [
            (1, 1, 1), (1, 2048, 1), (1, 2049, 2), (8, 257, 2), (9, 257, 3),
            (64, 7168, 224), (256, 4096, 512), (512, 4096, 752),
        ] {
            assert_eq!(grid.blocks(rows, hidden), blocks, "{rows} rows x {hidden} columns");
        }
    }

    #[test]
    fn grid_follows_the_device_and_covers_grid_stride_work() {
        for sms in [1, 17, 128, 170, 188, 256] {
            let grid = Fp8QuantizeGrid::new(sms, None).unwrap();
            assert_eq!(grid.blocks(1, 256), 1);
            assert_eq!(grid.blocks(8192, 7168), 4 * sms);
            assert_eq!(grid.blocks(usize::MAX, usize::MAX), 4 * sms);
        }
    }

    #[test]
    fn tuning_can_reduce_but_never_exceed_device_capacity() {
        let grid = Fp8QuantizeGrid::new(170, Some(73)).unwrap();
        assert_eq!(grid.blocks(8192, 7168), 292);
        assert_eq!(Fp8QuantizeGrid::new(170, Some(170)).unwrap().blocks(8192, 7168), 680);
        for requested in [0, 171, 188] {
            assert_eq!(Fp8QuantizeGrid::new(170, Some(requested)).unwrap_err(),
                SmCountError::InvalidLimit { requested, actual: 170 });
        }
        assert_eq!(Fp8QuantizeGrid::new(0, None).unwrap_err(), SmCountError::Unavailable);
    }
}
