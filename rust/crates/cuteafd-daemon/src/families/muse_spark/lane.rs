//! One coordinator rank's lane: the admitted activation planes
//! (`cuteafd_loader::families::muse_spark::lane::LanePlanes`) as one device
//! allocation carved into named views, plus checked row-extent access. Every
//! program launch takes its buffers from here, so a launch can never address
//! past the plane the planner charged.
use crate::shared::memory::device::{Allocation, Device};
use anyhow::{ensure, Context, Result};
use cuteafd_ffi::CuteafdDeviceBuffer as Buffer;
use cuteafd_loader::families::muse_spark::lane::LanePlanes;
use std::collections::BTreeMap;

/// Named views of one lane allocation. Span planes (`LanePlanes::SPAN`) hold
/// one unit's rows per in-flight slot; [`Self::slot`] views a slot's region.
#[derive(Clone)]
pub(crate) struct LaneViews {
    pub rows: usize,
    waves: usize,
    views: BTreeMap<&'static str, Buffer>,
    /// Bytes per row of each plane (planes are `rows`-major).
    row_bytes: BTreeMap<&'static str, usize>,
}

impl LaneViews {
    /// Carve `base` (the lane allocation) by the admitted plane offsets.
    pub(crate) fn carve(planes: &LanePlanes, base: Buffer) -> Result<Self> {
        ensure!(base.bytes >= planes.bytes()?, "Muse lane allocation smaller than its admitted planes");
        let mut views = BTreeMap::new();
        let mut row_bytes = BTreeMap::new();
        for (name, (offset, bytes)) in planes.offsets()? {
            let span = if LanePlanes::SPAN.contains(&name) { planes.waves } else { 1 };
            ensure!(bytes % (planes.rows * span) == 0, "Muse lane plane {name} is not rows-major");
            // A span plane's base view is slot 0's region.
            views.insert(name, Buffer { ptr: base.ptr.cast::<u8>().wrapping_add(offset).cast(), bytes: bytes / span, ..base });
            row_bytes.insert(name, bytes / span / planes.rows);
        }
        Ok(Self { rows: planes.rows, waves: planes.waves, views, row_bytes })
    }

    /// The views of in-flight slot `slot`: span planes at the slot's region (a
    /// unit's carried state: residual stream, routes, the wire), every other
    /// plane shared (each hook reads and writes them within the hook).
    pub(crate) fn slot(&self, slot: usize) -> Result<Self> {
        ensure!(slot < self.waves, "Muse lane slot {slot} of {}", self.waves);
        let mut views = self.views.clone();
        for name in LanePlanes::SPAN {
            let view = views.get_mut(name).with_context(|| format!("Muse lane has no span plane {name}"))?;
            view.ptr = view.ptr.cast::<u8>().wrapping_add(slot * view.bytes).cast();
        }
        Ok(Self { views, ..self.clone() })
    }

    /// The first `rows` rows of plane `name`.
    pub(crate) fn rows(&self, name: &str, rows: usize) -> Result<Buffer> {
        ensure!((1..=self.rows).contains(&rows), "Muse lane step of {rows} rows outside 1..={}", self.rows);
        let view = *self.views.get(name).with_context(|| format!("Muse lane has no plane {name}"))?;
        Ok(Buffer { bytes: rows * self.row_bytes[name], ..view })
    }

    /// The whole plane `name` (capacity rows).
    pub(crate) fn plane(&self, name: &str) -> Result<Buffer> {
        self.views.get(name).copied().with_context(|| format!("Muse lane has no plane {name}"))
    }
}

/// The lane's owning allocation and its views.
pub(crate) struct Lane<'a> {
    _owner: Allocation<'a>,
    pub views: LaneViews,
}

impl<'a> Lane<'a> {
    pub(crate) fn allocate(device: Device<'a>, planes: &LanePlanes) -> Result<Self> {
        let _scope = cuteafd_ffi::memory_ledger::scope("workspace/muse lane planes");
        let owner = Allocation::new(device, planes.bytes()?)?;
        let views = LaneViews::carve(planes, owner.buffer)?;
        Ok(Self { _owner: owner, views })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn views_tile_the_admitted_allocation_without_overlap() -> Result<()> {
        let cfg = cuteafd_loader::families::muse_spark::MuseSparkConfig::from_hf(
            &cuteafd_loader::plan::testing::muse_spark_config())?;
        for ranks in [1, 2] {
            let planes = LanePlanes::new(&cfg, ranks, 2752)?;
            let base = Buffer { ptr: 0x1000_0000usize as *mut _, bytes: planes.bytes()?, device_id: 0, flags: 0 };
            let views = LaneViews::carve(&planes, base)?;
            let mut spans: Vec<(usize, usize)> = planes.planes.keys().map(|name| {
                let v = views.plane(name).unwrap();
                (v.ptr as usize, v.ptr as usize + v.bytes)
            }).collect();
            spans.sort_unstable();
            for pair in spans.windows(2) { assert!(pair[0].1 <= pair[1].0, "planes overlap"); }
            assert!(spans.last().unwrap().1 <= 0x1000_0000 + base.bytes);
            // Row access: q at 7 rows is 7 x heads x 64 x 2 bytes for this rank's heads.
            assert_eq!(views.rows("q", 7)?.bytes, 7 * (128 / ranks) * 64 * 2);
            assert_eq!(views.rows("route_ids", 2752)?.bytes, 2752 * 8 * 4);
            assert!(views.rows("q", 0).is_err() && views.rows("q", 2753).is_err());
            assert!(views.rows("nope", 1).is_err());
        }
        // Slots: span planes step by one unit's region; other planes are shared.
        let span = LanePlanes::with_waves(&cfg, 2, 2752, 3)?;
        let base = Buffer { ptr: 0x1000_0000usize as *mut _, bytes: span.bytes()?, device_id: 0, flags: 0 };
        let views = LaneViews::carve(&span, base)?;
        for slot in 0..3 {
            let slot_views = views.slot(slot)?;
            for name in LanePlanes::SPAN {
                let unit = span.planes[name] / 3;
                assert_eq!(slot_views.plane(name)?.ptr as usize, views.plane(name)?.ptr as usize + slot * unit);
                assert_eq!(slot_views.plane(name)?.bytes, unit);
            }
            assert_eq!(slot_views.plane("q")?.ptr, views.plane("q")?.ptr);
        }
        assert!(views.slot(3).is_err());
        let planes = LanePlanes::new(&cfg, 1, 64)?;
        let small = Buffer { ptr: 0x1000usize as *mut _, bytes: planes.bytes()? - 1, device_id: 0, flags: 0 };
        assert!(LaneViews::carve(&planes, small).is_err());
        Ok(())
    }
}
