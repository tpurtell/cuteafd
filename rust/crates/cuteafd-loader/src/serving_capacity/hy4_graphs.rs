//! Hy4-flash exact-row decode graph geometry and provisional reserve admission.
//!
//! Uses the shared pressure-only donor default (`donor_always = false`,
//! `donor_cold_lag = 0`): rank-local decode sections capture lazily on first
//! replay and only reuse donors at or above the soft cap.
//!
//! PROVISIONAL: the per-rank reserve constants below are sized from measured
//! physical graph high-water under the default donor rule (calibration pending
//! final trace-based working-set sizing under #2673).

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Hy4GraphLayout {
    Exl3Min,
    Exl3Max,
    Exl3MinSpark,
    Exl3MaxSpark,
}

impl Hy4GraphLayout {
    pub fn resolve(ranks: usize, sparks: usize) -> Option<Self> {
        match (ranks, sparks) {
            (1, 0) => Some(Self::Exl3Min),
            (2, 0) => Some(Self::Exl3Max),
            (1, s) if s > 0 => Some(Self::Exl3MinSpark),
            (2, s) if s > 0 => Some(Self::Exl3MaxSpark),
            _ => None,
        }
    }

    pub fn ranks(self) -> usize {
        match self {
            Self::Exl3Min | Self::Exl3MinSpark => 1,
            Self::Exl3Max | Self::Exl3MaxSpark => 2,
        }
    }
}

/// PROVISIONAL per-rank `(base_bytes, tail_bytes)` passed to the shared
/// `graph_reserve::current` resolver (which adds `min(500 MB, base + tail)`
/// bounded runtime headroom, or honors an explicit `GPU_GRAPH_RESERVE` total).
///
/// Sized from donor-default (`donor_always = false`, `donor_cold_lag = 0`)
/// physical high-water calibration on `wrldsuksgo2mars/Hy4-flash-EXL3-K4-MC`
/// (`CORE_VIEW=k6-compact`, `GPU_GRAPH_RESERVE=1536MiB`, `C1/C2/C4/C8/C16` +
/// `32K C1/C8`, zero evictions / zero eager runs):
/// - `Exl3Min` (1x RTX, rank 0): `12 -> 18 -> 32 -> 44 -> 56 MiB` across
///   `C1/C2/C4/C8/C16` (`56.0 MiB` peak, `58.1 MiB` level, 270 execs, ~11.2 MiB/width);
///   `32K C1/C8` added `0 MiB`. Full 1..=16 staggered widths ~180 MiB;
///   `(128 MiB, 2 MiB)` resolves to `260 MiB` total reserve (+44% margin over 16 widths).
/// - `Exl3Max` (2x RTX, rank 0 / rank 1):
///   rank 0: `16 -> 34 -> 48 -> 66 -> 80 MiB` (`80.0 MiB` peak, 485 execs, ~16.0 MiB/width);
///   rank 1: `14 -> 30 -> 46 -> 62 -> 78 MiB` (`78.0 MiB` peak, 480 execs, ~15.6 MiB/width);
///   `32K C1/C8` added `0 MiB`. Full 1..=16 staggered widths ~256 MiB/rank;
///   `(192 MiB, 2 MiB)` resolves to `388 MiB`/rank total reserve (+52% margin over 16 widths).
///
/// PROVISIONAL: subject to replacement once the #2673 trace-based working-set
/// sizing rule lands.
pub fn reserve(layout: Option<Hy4GraphLayout>, rank: usize) -> (u64, u64) {
    match layout {
        Some(Hy4GraphLayout::Exl3Min | Hy4GraphLayout::Exl3MinSpark) if rank == 0 => {
            (128 << 20, 2 << 20)
        }
        Some(Hy4GraphLayout::Exl3Max | Hy4GraphLayout::Exl3MaxSpark) if rank < 2 => {
            (192 << 20, 2 << 20)
        }
        _ => (0, 0),
    }
}

/// Resolve the per-rank graph reserve policy for admission, provenance, and runtime
/// using the shared `graph_reserve::current` resolver (no #2645 lag-8 working-set formula).
pub fn reserve_policy(
    layout: Option<Hy4GraphLayout>,
    rank: usize,
) -> Result<crate::placement::graph_reserve::Resolved, crate::placement::graph_reserve::Error> {
    let (base, tail) = reserve(layout, rank);
    crate::placement::graph_reserve::current(rank, false, base, tail)
}

/// Rank-local exact-row layouts capture lazily without startup pins.
pub fn selected_rows(
    _layout: Option<Hy4GraphLayout>,
    _max_rows: usize,
) -> anyhow::Result<Vec<(usize, bool)>> {
    Ok(Vec::new())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn layout_resolution_distinguishes_rank_and_spark_shapes() {
        assert_eq!(Hy4GraphLayout::resolve(1, 0), Some(Hy4GraphLayout::Exl3Min));
        assert_eq!(Hy4GraphLayout::resolve(2, 0), Some(Hy4GraphLayout::Exl3Max));
        assert_eq!(Hy4GraphLayout::resolve(1, 4), Some(Hy4GraphLayout::Exl3MinSpark));
        assert_eq!(Hy4GraphLayout::resolve(2, 6), Some(Hy4GraphLayout::Exl3MaxSpark));
        assert_eq!(Hy4GraphLayout::resolve(0, 0), None);
        assert_eq!(Hy4GraphLayout::resolve(3, 0), None);
        assert_eq!(Hy4GraphLayout::Exl3Min.ranks(), 1);
        assert_eq!(Hy4GraphLayout::Exl3Max.ranks(), 2);
        assert_eq!(Hy4GraphLayout::Exl3MinSpark.ranks(), 1);
        assert_eq!(Hy4GraphLayout::Exl3MaxSpark.ranks(), 2);
    }

    #[test]
    fn all_supported_layouts_resolve_provisional_reserves() {
        assert_eq!(reserve(None, 0), (0, 0));
        assert_eq!(reserve(Some(Hy4GraphLayout::Exl3Min), 1), (0, 0));
        assert!(reserve(Some(Hy4GraphLayout::Exl3Min), 0).0 > 0);
        assert!(reserve(Some(Hy4GraphLayout::Exl3Max), 0).0 > 0);
        assert!(reserve(Some(Hy4GraphLayout::Exl3Max), 1).0 > 0);
        assert!(reserve(Some(Hy4GraphLayout::Exl3MinSpark), 0).0 > 0);
        assert!(reserve(Some(Hy4GraphLayout::Exl3MaxSpark), 1).0 > 0);
        assert!(selected_rows(Some(Hy4GraphLayout::Exl3Max), 64).unwrap().is_empty());

        let (base, tail) = reserve(Some(Hy4GraphLayout::Exl3Max), 0);
        let max_p = crate::placement::graph_reserve::resolve(None, 0, false, base, tail).unwrap();
        assert_eq!(max_p.base_bytes, base + tail);
        assert_eq!(max_p.soft_margin_floor, 0);
        assert!(max_p.bytes >= max_p.base_bytes);
    }
}
