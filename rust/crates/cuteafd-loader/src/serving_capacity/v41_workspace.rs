//! Physical layouts for independently retired V4.1 workspace sets.
use super::WeightArenaLayout;
use anyhow::{ensure, Result};

fn layout(component: &str, planes: impl IntoIterator<Item = (String, usize)>) -> Result<WeightArenaLayout> {
    WeightArenaLayout::new(format!("v41/{component}"), planes.into_iter().map(|(name, bytes)| (name, bytes.max(1))))
}

pub fn hc(rows: usize) -> Result<WeightArenaLayout> {
    ensure!((1..=8256).contains(&rows), "invalid mHC workspace rows");
    layout("hc", [("residual", 40960), ("incoming_pre", 16), ("next_pre", 16),
        ("post", 16), ("comb", 64), ("collapsed", 10240), ("normalized", 10240),
        ("sublayer", 10240), ("output", 40960)].map(|(name, stride)| (name.into(), rows * stride)))
}

pub fn window(rows: usize, unit: usize, layers: &[usize], scratch: usize) -> Result<WeightArenaLayout> {
    ensure!(unit > 0 && unit <= rows && rows <= 8256 && !layers.is_empty(), "invalid window workspace geometry");
    let mut planes = vec![("scratch".into(), scratch), ("alpha".into(), 4),
        ("input".into(), unit * 10240), ("projected".into(), unit * 1024),
        ("normalized".into(), unit * 1024), ("positions".into(), unit * 8),
        ("frequencies".into(), unit * 256)];
    for layer in layers {
        planes.extend([("values", 512), ("scales", 16), ("destinations", 8)]
            .map(|(name, stride)| (format!("layer{layer}.{name}"), rows * stride)));
    }
    layout("window-proposal", planes)
}

pub fn engram(unit: usize, scratch: usize) -> Result<WeightArenaLayout> {
    ensure!((1..=4096).contains(&unit), "invalid Engram workspace unit");
    layout("engram-layer", [("output", unit * 40960), ("residual", unit * 40960),
        ("embeddings", unit * 24 * 512), ("text_mask", unit), ("projected", unit * 51200),
        ("scratch", scratch), ("alpha", 4)].map(|(name, bytes)| (name.into(), bytes)))
}

pub fn projection(rows: usize, k: usize, n: usize, scratch: usize, owned_input: bool) -> Result<WeightArenaLayout> {
    ensure!((1..=4096).contains(&rows) && k > 0 && n > 0, "invalid draft projection geometry");
    let mut planes = vec![("scratch".into(), scratch), ("output".into(), rows * n * 2)];
    if owned_input { planes.push(("input".into(), rows * k * 2)); }
    planes.push(("alpha".into(), 4));
    layout("draft-projection", planes)
}

/// Scratch and data retire together, but small requests can share driver pages.
/// Keep a single backing unless splitting saves physical SM120 bytes.
pub fn projection_backings(rows: usize, k: usize, n: usize, scratch: usize,
    owned_input: bool) -> Result<Vec<WeightArenaLayout>> {
    let combined = projection(rows, k, n, scratch, owned_input)?;
    let mut data = combined.views().iter().filter(|(name, _)| name.as_str() != "scratch")
        .collect::<Vec<_>>();
    data.sort_by_key(|(_, view)| view.offset());
    let split = vec![layout("draft-projection/scratch", [("scratch".into(), scratch)])?,
        layout("draft-projection/data", data.into_iter().map(|(name, view)| (name.clone(), view.bytes())))?];
    if projection_backing_bytes(&split)? < projection_backing_bytes(std::slice::from_ref(&combined))? {
        Ok(split)
    } else {
        Ok(vec![combined])
    }
}

/// Each physical request and its independent driver tail are charged once.
pub fn projection_backing_bytes(backings: &[WeightArenaLayout]) -> Result<usize> {
    backings.iter().try_fold(0usize, |sum, backing| {
        let bytes = crate::placement::planner::sm120_request_backing(backing.bytes() as u64)?;
        sum.checked_add(usize::try_from(bytes)?).ok_or_else(|| anyhow::anyhow!("projection backing overflow"))
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn census9_components_match_payloads_and_independent_backing_tails() -> Result<()> {
        use crate::placement::planner::independent_backing_slack;
        // #1714 census9: V4.1 Flash, one RTX + three Sparks, decode 64,
        // prefill 2752 x 3. Each entry is an independently retired storage set.
        let hc_sets = [64; 4].into_iter().chain([40; 12]).chain([8256; 2])
            .map(hc).collect::<Result<Vec<_>>>()?;
        let layers = (0..40).collect::<Vec<_>>();
        let window_sets = vec![window(64, 64, &layers, 629760)?,
            window(64, 64, &layers, 629760)?, window(8256, 2752, &layers, 22912000)?];
        let engram_sets = [64; 4].into_iter().map(|rows| engram(rows, 587776))
            .chain([2752; 2].into_iter().map(|rows| engram(rows, 27326464)))
            .collect::<Result<Vec<_>>>()?;
        let mut projection_sets = Vec::new();
        for _ in 0..6 {
            for (k, n, scratch) in [(5120, 1280, 146432), (1280, 32768, 918528),
                (5120, 512, 714752), (8192, 5120, 923648)] {
                projection_sets.push(projection(40, k, n, scratch, true)?);
            }
        }
        for (rows, scratch) in [(64, 1509376), (2752, 68356096)] {
            projection_sets.push(projection(rows, 15360, 5120, scratch, true)?);
        }
        for _ in 0..3 {
            for (rows, scratch) in [(64, 629760), (2752, 22912000)] {
                projection_sets.push(projection(rows, 5120, 512, scratch, false)?);
            }
        }
        for (sets, old_count, old_payload, new_payload, old_tail, new_tail) in [
            (hc_sets, 162, 1_944_746_496, 1_944_751_104, 38_535_168, 26_571_776),
            (window_sets, 381, 240_074_252, 240_075_008, 86_664_192, 3_194_624),
            (engram_sets, 42, 894_559_896, 894_560_792, 29_495_296, 3_020_264),
            (projection_sets, 122, 309_213_312, 309_213_312, 18_226_176, 15_590_308),
        ] {
            let old_requests = sets.iter().flat_map(|set| set.views().values()
                .map(|view| view.bytes() as u64)).collect::<Vec<_>>();
            let new_requests = sets.iter().map(|set| set.bytes() as u64).collect::<Vec<_>>();
            assert_eq!(old_requests.len(), old_count);
            assert_eq!(old_requests.iter().sum::<u64>(), old_payload);
            assert_eq!(new_requests.iter().sum::<u64>(), new_payload);
            assert_eq!(independent_backing_slack(old_requests)?, old_tail);
            assert_eq!(independent_backing_slack(new_requests)?, new_tail);
        }
        Ok(())
    }

    #[test]
    fn measured_projection_backings_avoid_small_request_threshold() -> Result<()> {
        use crate::placement::planner::independent_backing_slack;
        let mut old = Vec::new();
        let mut new = Vec::new();
        for _ in 0..6 {
            for (k, n, scratch) in [(5120, 1280, 923648), (1280, 32768, 146432),
                (5120, 512, 714752), (8192, 5120, 918528)] {
                old.push(projection(40, k, n, scratch, true)?);
                new.extend(projection_backings(40, k, n, scratch, true)?);
            }
        }
        for (rows, scratch) in [(64, 1509376), (2752, 68356096)] {
            old.push(projection(rows, 15360, 5120, scratch, true)?);
            new.extend(projection_backings(rows, 15360, 5120, scratch, true)?);
        }
        for _ in 0..3 {
            for (rows, scratch) in [(64, 629760), (2752, 22912000)] {
                old.push(projection(rows, 5120, 512, scratch, false)?);
                new.extend(projection_backings(rows, 5120, 512, scratch, false)?);
            }
        }
        assert_eq!((old.len(), new.len()), (32, 44));
        assert_eq!(old.iter().map(WeightArenaLayout::bytes).sum::<usize>(),
            new.iter().map(WeightArenaLayout::bytes).sum::<usize>());
        assert_eq!(independent_backing_slack(old.iter().map(|b| b.bytes() as u64))?, 24_222_604);
        assert_eq!(independent_backing_slack(new.iter().map(|b| b.bytes() as u64))?, 14_662_588);
        assert_eq!(projection_backing_bytes(&old)? - projection_backing_bytes(&new)?, 9_560_016);
        Ok(())
    }

    #[test]
    fn projection_split_preserves_planes_and_never_increases_physical_charge() -> Result<()> {
        for rows in [1, 16, 40, 64, 256, 2752, 4096] {
            for scratch in [0, 1, (1 << 20) - 1, 1 << 20, (1 << 20) + 1, 22912000] {
                for owned_input in [false, true] {
                    let combined = projection(rows, 5120, 512, scratch, owned_input)?;
                    let backings = projection_backings(rows, 5120, 512, scratch, owned_input)?;
                    assert!(projection_backing_bytes(&backings)? <=
                        projection_backing_bytes(std::slice::from_ref(&combined))?);
                    let planes = backings.iter().flat_map(|b| b.views().iter().map(|(name, view)|
                        (name.clone(), view.bytes()))).collect::<std::collections::BTreeMap<_, _>>();
                    assert_eq!(planes.len(), backings.iter().map(|b| b.views().len()).sum::<usize>());
                    assert_eq!(planes, combined.views().iter().map(|(name, view)|
                        (name.clone(), view.bytes())).collect());
                    for backing in backings {
                        assert!(backing.views().values().all(|v| v.offset() % 256 == 0));
                    }
                }
            }
        }
        Ok(())
    }

    #[test]
    fn packed_components_preserve_planes_and_one_physical_extent() -> Result<()> {
        for rows in [1, 16, 64, 256, 4096] {
            for layout in [hc(rows)?, window(rows, rows, &(0..40).collect::<Vec<_>>(), 12345)?,
                engram(rows, 12345)?, projection(rows, 5120, 512, 12345, true)?,
                projection(rows, 5120, 512, 12345, false)?] {
                let mut planes = layout.views().values().collect::<Vec<_>>();
                planes.sort_by_key(|view| view.offset());
                assert!(planes.iter().all(|view| view.offset() % 256 == 0 && view.bytes() > 0));
                assert!(planes.windows(2).all(|pair| pair[0].offset() + pair[0].bytes() <= pair[1].offset()));
                assert_eq!(planes.last().unwrap().offset() + planes.last().unwrap().bytes(), layout.bytes());
            }
        }
        Ok(())
    }
}
