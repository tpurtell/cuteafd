//! Onboarded expert layers on the coordinator RTXs (release-mandatory TP2):
//! each GPU holds its intermediate half of every onboarded layer (a pair) or
//! the whole layer (one GPU), runs the shared EXL3 core on the full expert
//! input and writes every routed (row, slot)'s unweighted BF16 partial with
//! the shared partial-reply epilogue (`Exl3Tp2::enqueue_partials`: the Spark
//! worker's core and epilogue, so the bytes are the Sparks'). A pair swaps
//! the partials (`Channel::Partials`) and both combine with ranks 2
//! (`rank_regions` = this GPU's plane, the received slot, in rank order); one
//! GPU combines its own with ranks 1. The combine, its norm and the backend
//! are the Spark layers'.
//!
//! Memory: admission charged each GPU's TP2 (or whole) arena as the
//! executor workspace and the half (or whole) layers plus the own and peer
//! partial planes (`tp2_workspace` / `expert_workspace`, with the epilogue's
//! row maps and the combine tables in `local_package`'s workspace); the loader
//! gets the arena minus what this module allocates (the own plane, the
//! epilogue maps, the tables; the peer plane is the exchange's slot).
use super::experts::CombineTables;
use crate::shared::experts::{exl3::partial::PartialEpilogue, rtx::{exl3::Exl3Tp2, RtxSelectionPolicy}};
use crate::shared::memory::device::{Allocation, Device};
use anyhow::{ensure, Context, Result};
use cuteafd_ffi::{programs::Programs, CuteafdDeviceBuffer as Buffer};
use cuteafd_loader::families::muse_spark::MuseSparkConfig;
use cuteafd_loader::placement::Placement;

/// One GPU's onboarded expert state.
pub(crate) struct OnboardedExperts<'a> {
    pub executor: Exl3Tp2<'a>,
    pub epilogue: PartialEpilogue<'a>,
    /// This GPU's BF16 partials of one unit: `[unit, topk, E]`.
    pub own: Allocation<'a>,
    /// One combine image (the onboarded combine completes within its hook).
    pub tables: Allocation<'a>,
    pub layout: CombineTables,
    pub layers: Vec<usize>,
    /// The rank planes `tables` was written for (fixed: this GPU's own plane
    /// and the pair's single `Partials` receive slot).
    written: Option<Vec<u64>>,
}

/// Bytes of one GPU's own partial plane at `rows`.
pub(crate) fn own_bytes(cfg: &MuseSparkConfig, rows: usize) -> usize { rows * cfg.topk * cfg.expert_hidden * 2 }

/// Loads every GPU's onboarded layers (`layers`, ascending) from the installed
/// coordinator package (`package`: `rtx-tp2` on a pair, `rtx-tp1` on one GPU).
#[allow(clippy::too_many_arguments)]
pub(crate) fn load<'a>(devices: &[Device<'a>], programs: &[&'a Programs<'a>], cfg: &MuseSparkConfig,
    catalog: &cuteafd_loader::OfficialV41Catalog, package: &std::path::Path, placement: &Placement,
    layers: Vec<usize>, unit_rows: usize) -> Result<Vec<OnboardedExperts<'a>>> {
    ensure!(!layers.is_empty() && layers.windows(2).all(|pair| pair[0] < pair[1]), "Muse onboarded layers ascending");
    ensure!(devices.len() == programs.len() && matches!(devices.len(), 1 | 2), "Muse onboarding on one GPU or a pair");
    let own = own_bytes(cfg, unit_rows);
    let layout = CombineTables { capacity: unit_rows };
    // Allocated here, outside the loader's budget: the epilogue's row maps and the combine tables.
    let fixed = 2 * unit_rows * 4 + layout.bytes();
    let geometry = cuteafd_core::expert_geometry();
    let executors: Vec<Exl3Tp2<'a>> = if devices.len() == 2 {
        // The pair's arenas hold the executor, the halves and both partial planes.
        let tp2 = placement.tp2.context("Muse onboarded pair without a TP2 arena")?;
        let budgets = tp2.peak_bytes.map(|bytes| usize::try_from(bytes).unwrap_or(0).saturating_sub(2 * own + fixed));
        let weights = Exl3Tp2::load_pair_weights([devices[0], devices[1]], catalog, geometry, [package; 2],
            layers.clone(), unit_rows, budgets, RtxSelectionPolicy::CanonicalDecode)?;
        weights.iter().map(|weights| weights.execution()).collect::<Result<_>>()?
    } else {
        let range = placement.expert_ranges.first().context("Muse onboarded GPU without an expert arena")?;
        let budget = usize::try_from(range.peak_bytes)?.saturating_sub(own + fixed);
        vec![Exl3Tp2::load_whole_weights(devices[0], catalog, package, layers.clone(), unit_rows, budget,
            RtxSelectionPolicy::CanonicalDecode)?.execution()?]
    };
    executors.into_iter().zip(devices).zip(programs).map(|((executor, &device), programs)| {
        let _scope = cuteafd_ffi::memory_ledger::scope("experts/muse onboarded partials");
        Ok(OnboardedExperts { epilogue: device.run(|| PartialEpilogue::load(device.library, programs, unit_rows))?,
            own: Allocation::new(device, own)?, tables: Allocation::new(device, layout.bytes())?, layout,
            executor, layers: layers.clone(), written: None })
    }).collect()
}

impl OnboardedExperts<'_> {
    pub(crate) fn holds(&self, layer: usize) -> bool { self.layers.binary_search(&layer).is_ok() }

    /// Every onboarded combine reads all rows from the BF16 region in wave
    /// order, so one image at full capacity (identity row maps; a wave reads
    /// its prefix) serves every layer and unit. It is written once per plane
    /// set, keeping the per-layer path free of host waits on the upload.
    pub(crate) fn tables_for(&mut self, library: &cuteafd_ffi::NativeLibrary, planes: &[u64], stream: *mut std::ffi::c_void)
        -> Result<()> {
        if self.written.as_deref() == Some(planes) { return Ok(()); }
        let rows: Vec<u32> = (0..u32::try_from(self.layout.capacity)?).collect();
        let image = self.layout.image(planes, &super::experts::ReplyPlanView { bf16_rows: &rows, wide_rows: &[], wide_at: 0 })?;
        // SAFETY: this GPU's fixed tables; every combine that reads them follows on `stream`.
        unsafe { library.copy_h2d_ordered(self.tables.buffer, &image, stream)?; }
        self.written = Some(planes.to_vec());
        Ok(())
    }

    /// This GPU's partial plane at `rows` (the region the epilogue writes, BF16).
    pub(crate) fn own_rows(&self, cfg: &MuseSparkConfig, rows: usize) -> Buffer {
        Buffer { bytes: own_bytes(cfg, rows), ..self.own.buffer }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::families::muse_spark::experts::ReplyPlanView;

    /// The cached full-capacity image (identity BF16 map, no wide rows) holds,
    /// for every wave of `rows`, exactly the bytes the combine reads from that
    /// wave's own image: the rank tables and the first `rows` map entries.
    #[test]
    fn cached_tables_serve_every_wave() -> Result<()> {
        let layout = CombineTables { capacity: 64 };
        let planes = [0x1000u64, 0x9000];
        let all: Vec<u32> = (0..64).collect();
        let cached = layout.image(&planes, &ReplyPlanView { bf16_rows: &all, wide_rows: &[], wide_at: 0 })?;
        let [bf16_table, _, bf16_map, _] = layout.offsets();
        for rows in [1usize, 7, 64] {
            let wave = layout.image(&planes, &ReplyPlanView { bf16_rows: &all[..rows], wide_rows: &[], wide_at: 0 })?;
            assert_eq!(cached[bf16_table..bf16_table + 2 * 8], wave[bf16_table..bf16_table + 2 * 8]);
            assert_eq!(cached[bf16_map..bf16_map + rows * 4], wave[bf16_map..bf16_map + rows * 4]);
        }
        Ok(())
    }

    /// The partial epilogue's BF16 map for an all-BF16 plan is the identity
    /// prefix, so successive launches stage identical leading bytes.
    #[test]
    fn all_bf16_maps_are_identity_prefixes() {
        for rows in [1usize, 5, 64] {
            let plan = cuteafd_transport::PartialReplyPlan { wide: cuteafd_transport::PartialWire::Nvfp4,
                regions: vec![0; rows] };
            let (bf16, wide) = crate::shared::experts::exl3::partial::row_maps(&plan);
            assert_eq!(bf16, (0..rows as u32).collect::<Vec<_>>());
            assert!(wide.is_empty());
        }
    }
}
