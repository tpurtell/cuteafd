//! Resident storage and measured recipes shared with native packaging.
use serde::Deserialize;
use std::sync::OnceLock;

#[derive(Deserialize)]
struct StorageHold { hidden: usize, width: usize, stored: usize }
#[derive(Deserialize)]
struct SliceRecipe { hidden: usize, width: usize, sm: u32, capacities: Vec<u32>, slices: Vec<usize>,
    #[serde(default)] prefill: Vec<u32> }
#[derive(Deserialize)]
struct Ownership { intermediate: usize, world: usize, unit: usize }
#[derive(Deserialize)]
struct Layouts { schema: u32, storage_holds: Vec<StorageHold>, native_slices: Vec<SliceRecipe>,
    ownership: Vec<Ownership>, generic_storage_holds: Vec<StorageHold> }

fn validate(data: &Layouts) -> Result<(), &'static str> {
    if data.schema != 1 { return Err("unsupported MXFP4 layout schema"); }
    let mut keys = std::collections::HashSet::new();
    for row in &data.native_slices {
        if !keys.insert((row.hidden, row.width, row.sm))
            || row.prefill.iter().enumerate().any(|(i, c)| ![256, 1024, 4096].contains(c) || row.prefill[..i].contains(c))
            || (row.capacities.is_empty() && row.prefill.is_empty()) {
            return Err("invalid native MXFP4 prefill recipe");
        }
        if row.capacities.len() != row.slices.len()
            || row.capacities.iter().enumerate().any(|(i, capacity)| row.capacities[..i].contains(capacity)) {
            return Err("native MXFP4 capacities must pair uniquely with slices");
        }
        let decode: Vec<_> = row.capacities.iter().zip(&row.slices)
            .filter_map(|(&capacity, &slice)| (capacity <= 80).then_some(slice)).collect();
        if (!row.capacities.is_empty() && decode.is_empty()) || !decode.windows(2).all(|pair| pair[0] == pair[1]) {
            return Err("native MXFP4 decode capacities must share one slice recipe");
        }
        if row.slices.iter().any(|slice| ![32, 64, 128, 192].contains(slice)) {
            return Err("invalid native MXFP4 slice width");
        }
    }
    for row in &data.ownership {
        if ![32, 64, 128].contains(&row.unit) || row.intermediate % row.unit != 0
            || !(1..=8).contains(&row.world) || row.intermediate / row.unit < row.world {
            return Err("invalid measured MXFP4 ownership");
        }
    }
    Ok(())
}

fn layouts() -> &'static Layouts {
    static DATA: OnceLock<Layouts> = OnceLock::new();
    DATA.get_or_init(|| {
        let data: Layouts = serde_json::from_str(include_str!(
            "../../../../scripts/build/mxfp4-layouts.json")).expect("checked MXFP4 layouts");
        validate(&data).expect("checked MXFP4 layout policy");
        data
    })
}

/// Measured adjacent ownership; K32 remains the default scale-aligned unit.
pub fn ownership_unit(intermediate: usize, world: usize) -> usize {
    layouts().ownership.iter().find(|row| row.intermediate == intermediate && row.world == world)
        .map_or(32, |row| row.unit)
}

/// Generic packages may retain a padded extent without changing source ownership.
pub fn generic_stored_width(hidden: usize, width: usize) -> usize {
    layouts().generic_storage_holds.iter().find(|row| row.hidden == hidden && row.width == width)
        .map_or(width, |row| row.stored)
}

/// The same SM121 recipe the native exporter selects. Unmeasured capacities
/// retain their matching tile; measured decode recipes are fixed through C80.
pub fn native_slice_width(hidden: usize, width: usize, sm: u32, capacity: u32,
    matching: usize) -> usize {
    layouts().native_slices.iter().find(|row| row.hidden == hidden && row.width == width && row.sm == sm)
        .and_then(|row| row.capacities.iter().position(|&rows| rows == capacity).map(|index| row.slices[index]))
        .unwrap_or(matching)
}

/// Hardware-qualified prefill backend from the same table as native decode.
pub fn native_prefill(hidden: usize, width: usize, sm: u32, capacity: u32) -> bool {
    layouts().native_slices.iter().any(|row| row.hidden == hidden && row.width == width
        && row.sm == sm && row.prefill.contains(&capacity))
}

/// The seven STREAM planes and two native ABI per-expert trailing planes.
pub fn native_prefill_scratch_bytes(hidden: usize, width: usize, experts: usize,
    topk: usize, capacity: u32) -> Option<u64> {
    stored_width(hidden, width)?;
    if hidden % 512 != 0 || experts == 0 || topk == 0 || ![256, 1024, 4096].contains(&capacity) { return None; }
    let (h, w, e) = (hidden as u64, width as u64, experts as u64);
    let pairs = u64::from(capacity).checked_mul(topk as u64)?;
    let tiles = pairs.div_ceil(128).checked_add(e.min(pairs))?;
    let row = w.checked_add(w / 32)?.checked_add(15)?.checked_div(16)?.checked_mul(16)?;
    let meta = 4u64.checked_add(e.checked_mul(3)?)?.checked_add(tiles)?.checked_mul(4)?;
    [meta, pairs.checked_mul(4)?, pairs.checked_mul(4)?,
        tiles.checked_mul(128)?.checked_mul(row)?, pairs.checked_mul(4)?, pairs.checked_mul(4)?,
        pairs.checked_mul(h)?.checked_mul(4)?, e.checked_mul(4)?, e.checked_mul(4)?]
        .into_iter().try_fold(0u64, |offset, size| offset.checked_add(15)?
            .checked_div(16)?.checked_mul(16)?.checked_add(size))
}

/// `export_b12x_slices_aot.py`'s 16-byte-aligned scratch planes, excluding
/// shared owner I/O. Qualified prefill uses original-route STREAM planes;
/// unqualified geometries retain their existing native slice recipe.
pub fn native_scratch_bytes(hidden: usize, experts: usize, topk: usize,
    width: usize, role: u32, capacity: u32, coded: bool) -> Option<u64> {
    stored_width(hidden, width)?;
    if experts == 0 || topk == 0 || ![1, 16, 80, 256, 1024, 4096].contains(&capacity) { return None; }
    if native_prefill(hidden, width, 121, capacity) {
        return native_prefill_scratch_bytes(hidden, width, experts, topk, capacity);
    }
    let matching = if coded && (crate::tp_units::native_role_width(role).is_none() || width % 192 == 0) {
        192
    } else { 128 };
    let slice = native_slice_width(hidden, width, 121, capacity, matching) as u64;
    let (hidden, experts, routes) = (hidden as u64, experts as u64,
        u64::from(capacity).checked_mul(topk as u64)?);
    let atomic = !coded && crate::tp_units::native_role_width(role).is_none() && capacity >= 256;
    // FC1 activation storage is independent of the weights' scale encoding.
    let partial_cols = slice / 4 + slice / 32;
    let partial = if atomic { 4 } else {
        (width as u64).div_ceil(slice).checked_mul(routes)?.checked_mul(partial_cols)?.checked_mul(4)?
    };
    let output = (if atomic { u64::from(capacity) } else { routes }).checked_mul(hidden)?.checked_mul(4)?;
    [4, experts.checked_mul(routes)?.checked_mul(4)?, experts.checked_mul(4)?,
        experts.checked_mul(8)?, routes.checked_mul(76)?, routes.checked_mul(4)?,
        routes.checked_mul(4)?, partial, output, experts.checked_mul(4)?, experts.checked_mul(4)?]
        .into_iter().try_fold(0u64, |offset, size| offset.checked_add(15)?
            .checked_div(16)?.checked_mul(16)?.checked_add(size))
}

/// Storage decisions depend on matrix geometry, not a family-specific switch.
/// Holds retain the measured layout pending an explicit qualification decision.
pub fn stored_width(hidden: usize, width: usize) -> Option<usize> {
    if hidden == 0 || hidden % 256 != 0 || width == 0 || width % 32 != 0 || width > 8192 {
        return None;
    }
    Some(layouts().storage_holds.iter().find(|row| row.hidden == hidden && row.width == width)
        .map_or(width, |row| row.stored))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn scratch_matches_exporter_plane_offsets_and_capacity_recipe() {
        // Independent exporter specs: live/count/prefix/metadata/group/inverse,
        // route partials, output, and two trailing per-expert planes.
        let planes = |hidden: u64, experts: u64, topk: u64, width: u64,
            tile: u64, capacity: u64, atomic: bool| {
            let routes = capacity * topk;
            // One byte/activation plus one four-byte scale word per K32 block.
            let partial = if atomic { 4 } else {
                width.div_ceil(tile) * routes * (tile + tile / 8)
            };
            let mut end = 0u64;
            for size in [4, experts * routes * 4, experts * 4, experts * 8,
                routes * 19 * 4, routes * 4, routes * 4, partial,
                if atomic { capacity } else { routes } * hidden * 4, experts * 4, experts * 4] {
                end = end.div_ceil(16) * 16 + size;
            }
            end
        };
        for (hidden, experts, topk, width, role, coded, tiles) in [
            (5120, 384, 6, 576, 1, true, [128, 128, 128, 192, 192, 192]),
            (4096, 256, 6, 352, 65888, false, [32, 32, 32, 128, 128, 128]),
            (5120, 384, 6, 288, 8, true, [128, 128, 128, 64, 64, 64]),
            (5120, 384, 6, 512, 66048, true, [128; 6]),
            (5120, 384, 6, 448, 65984, true, [128; 6]),
            (5120, 384, 6, 352, 65888, true, [128; 6]),
            (4096, 256, 6, 512, 1, false, [128; 6]),
            (4096, 256, 6, 704, 66240, false, [128; 6]),
            (7168, 384, 6, 608, 66144, false, [128; 6]),
        ] {
            for (capacity, tile) in [1, 16, 80, 256, 1024, 4096].into_iter().zip(tiles) {
                let atomic = !coded && role < 65536 && capacity >= 256;
                let expected = if native_prefill(hidden as usize, width as usize, 121, capacity as u32) {
                    native_prefill_scratch_bytes(hidden as usize, width as usize, experts as usize,
                        topk as usize, capacity as u32).unwrap()
                } else { planes(hidden, experts, topk, width, tile, capacity, atomic) };
                assert_eq!(native_scratch_bytes(hidden as usize, experts as usize, topk as usize,
                    width as usize, role, capacity as u32, coded), Some(expected));
            }
        }
        for (hidden, experts, topk, width, capacity) in [
            (255, 384, 6, 288, 1), (5120, 0, 6, 288, 1),
            (5120, 384, 0, 288, 1), (5120, 384, 6, 31, 1), (5120, 384, 6, 288, 64),
        ] {
            assert_eq!(native_scratch_bytes(hidden, experts, topk, width, 8, capacity, true), None);
        }
        assert_eq!(native_scratch_bytes(5120, usize::MAX, 6, 288, 8, 4096, true), None);
    }

    #[test]
    fn stream_scratch_matches_independent_plane_census() {
        for row in &layouts().native_slices {
            for &capacity in &row.prefill {
                let (h, w, e, pairs) = (row.hidden as u64, row.width as u64,
                    if row.hidden == 5120 { 384u64 } else { 256u64 }, u64::from(capacity) * 6);
                let chunks = pairs.div_ceil(128) + e.min(pairs);
                let mut expected = 0u64;
                for size in [(4 + 3 * e + chunks) * 4, pairs * 4, pairs * 4,
                    chunks * 128 * (w + w / 32).div_ceil(16) * 16, pairs * 4, pairs * 4,
                    pairs * h * 4, e * 4, e * 4] {
                    expected = expected.div_ceil(16) * 16 + size;
                }
                assert!(native_prefill(row.hidden, row.width, row.sm, capacity));
                assert_eq!(native_prefill_scratch_bytes(row.hidden, row.width, e as usize, 6, capacity), Some(expected));
                for decode in [1, 16, 80] { assert!(!native_prefill(row.hidden, row.width, row.sm, decode)); }
            }
        }
        assert!(!native_prefill(7168, 768, 121, 256));
        assert!(!native_prefill(5120, 288, 120, 256));
    }

    #[test]
    fn policy_rejects_capacity_dependent_decode_tiling_and_invalid_tiles() {
        let bundled = include_str!("../../../../scripts/build/mxfp4-layouts.json");
        for failure in 0..4 {
            let mut data: Layouts = serde_json::from_str(bundled).unwrap();
            match failure {
                0 => data.native_slices[0].slices[1] = 64,
                1 => data.native_slices[0].slices[2] = 256,
                2 => data.native_slices[0].capacities[1] = 1,
                _ => { data.native_slices[0].slices.pop(); }
            }
            assert!(validate(&data).is_err());
        }
        assert_eq!(native_slice_width(5120, 288, 120, 1, 192), 192);
        assert_eq!(native_slice_width(5120, 288, 121, 1, 192), 128);
        assert_eq!(native_slice_width(5120, 288, 121, 256, 192), 192);
        assert!(native_prefill(5120, 288, 121, 256));
    }

    #[test]
    fn geometry_selects_exact_storage_or_an_explicit_measurement_hold() {
        assert_eq!(stored_width(5120, 576), Some(576));
        assert_eq!(ownership_unit(2304, 5), 64);
        assert_eq!(ownership_unit(2304, 7), 32);
        assert_eq!(generic_stored_width(6144, 352), 384);
        assert_eq!(generic_stored_width(4096, 352), 352);
        for width in [288, 320, 352, 448, 480, 768, 1152, 2304] {
            assert_eq!(stored_width(5120, width), Some(width));
        }
        for (hidden, widths) in [(4096, vec![288, 320, 352, 416, 672, 704]),
                                (7168, vec![416, 448, 608])] {
            for width in widths {
                assert_eq!(stored_width(hidden, width), Some(width));
            }
        }
        for (hidden, width) in [(0, 288), (255, 288), (5120, 0), (5120, 31), (5120, 8224)] {
            assert_eq!(stored_width(hidden, width), None);
        }
    }
}
