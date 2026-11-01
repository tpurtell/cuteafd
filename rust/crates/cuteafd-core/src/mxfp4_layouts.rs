//! Resident storage and measured recipes shared with native packaging.
use serde::Deserialize;
use std::sync::OnceLock;

#[derive(Deserialize)]
struct StorageHold { hidden: usize, width: usize, stored: usize }
#[derive(Deserialize)]
struct Layouts { schema: u32, storage_holds: Vec<StorageHold> }

/// Layout decisions depend on matrix geometry, never a family or Spark count.
/// Holds retain the measured layout pending an explicit qualification decision.
pub fn stored_width(hidden: usize, width: usize) -> Option<usize> {
    if hidden == 0 || hidden % 256 != 0 || width == 0 || width % 32 != 0 || width > 8192 {
        return None;
    }
    static DATA: OnceLock<Layouts> = OnceLock::new();
    let data = DATA.get_or_init(|| {
        let data: Layouts = serde_json::from_str(include_str!(
            "../../../../scripts/build/mxfp4-layouts.json")).expect("checked MXFP4 layouts");
        assert_eq!(data.schema, 1);
        data
    });
    Some(data.storage_holds.iter().find(|row| row.hidden == hidden && row.width == width)
        .map_or(width, |row| row.stored))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn geometry_selects_exact_storage_or_an_explicit_measurement_hold() {
        assert_eq!(stored_width(5120, 576), Some(640));
        for width in [288, 320, 352, 448, 480, 768, 1152, 2304] {
            assert_eq!(stored_width(5120, width), Some(width));
        }
        for (hidden, widths) in [(4096, vec![288, 320, 352, 416, 672, 704]),
                                (7168, vec![416, 448, 608])] {
            for width in widths {
                assert_eq!(stored_width(hidden, width), Some(width.div_ceil(128) * 128));
            }
        }
        for (hidden, width) in [(0, 288), (255, 288), (5120, 0), (5120, 31), (5120, 8224)] {
            assert_eq!(stored_width(hidden, width), None);
        }
    }
}
