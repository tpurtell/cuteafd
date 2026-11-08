//! DeepSeek V4 startup attention/tail keys shared by admission and capture.
//! Expert and dSpark execution is eager; target taps and draft KV writes are
//! part of the target segments. Keys can feed #1612's node counter once native
//! launch counts and V4 architecture rates have been measured (#1614).
use crate::placement::inventory::{DeclaredGraphSet, GraphSet, Lifetime};

pub const LOCAL_EXPERTS: usize = usize::MAX;
pub const SKIPPED_EXPERTS: usize = usize::MAX - 1;
pub const TAIL_SEGMENT: usize = usize::MAX;

/// Provisional SM120 executable-overhead seed (~104 KiB), approved for #1614
/// from the #1605/#1612 cross-family observations; not a V4 measurement or rate.
/// V4.1's sealed set is larger: 4,179,623,936 B / 33,808 executables (~120.7 KiB).
/// No nodes or padding are asserted. The first V4 two-RTX capture reconciles
/// this Estimated owner; replace the seed with V4 node/rates once measured.
pub const ESTIMATED_EXECUTABLE_SEED: u64 = 104 * 1024;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize)]
pub struct V4GraphKey {
    pub layer: usize,
    pub rows: usize,
    pub chunked: bool,
    pub attention: &'static str,
    pub table_width: usize,
    pub table_stride: usize,
    pub previous: usize,
    /// Preserve both TP2 variants until native launch arguments prove source
    /// invariance: https://github.com/tpurtell/cuteafd/issues/1624.
    pub tp2_broadcast: bool,
    pub exchange_f32: bool,
}

pub fn c4_width_class(width: usize, units: usize) -> usize {
    width.max(1).next_power_of_two().min(units.max(1))
}

pub fn c4_width_classes(units: usize) -> Vec<usize> {
    let units = units.max(1);
    let mut classes = vec![1usize];
    while let Some(next) = classes.last().and_then(|last| last.checked_mul(2)) {
        if next > units { break; }
        classes.push(next);
    }
    if classes.last().copied() != Some(units) { classes.push(units); }
    classes
}

/// `previous` is the actual previous layer's reduction owner, never a cross
/// product of possible owners. Peer post keys use `usize::from(layer > 0)`.
/// Both plain and continuation compressor launches have distinct keys. Window
/// fallback on compressed layers is possible only in width class one; C4 has
/// no visible group before token four, C128 before token 128.
pub fn v4_serving_graph_keys(ratios: &[usize], units: usize, decode_rows: usize,
    ranks: usize, reductions: &[usize], tp2: bool, exchange_f32: bool) -> Option<Vec<Vec<V4GraphKey>>> {
    if ratios.is_empty() || ratios.len() != reductions.len() || !(1..=2).contains(&ranks)
        || decode_rows == 0 || ratios.iter().any(|ratio| ![0, 4, 128].contains(ratio)) { return None; }
    let classes = c4_width_classes(units);
    let mut keys = vec![Vec::new(); ranks];
    for (rank, bank) in keys.iter_mut().enumerate() {
        for broadcast in [false, true].into_iter().filter(|&broadcast| !broadcast || tp2) {
        for rows in 1..=decode_rows {
            for chunked in [false, true] {
                for &width in &classes {
                    for (layer, &ratio) in ratios.iter().enumerate() {
                        let previous = if rank == 1 { usize::from(layer > 0) }
                            else if layer == 0 { 0 } else { reductions[layer - 1] };
                        let key = V4GraphKey { layer, rows, chunked, attention: "win", table_width: width,
                            table_stride: width, previous, tp2_broadcast: broadcast, exchange_f32: tp2 && exchange_f32 };
                        if ratio == 0 || width == 1 { bank.push(key); }
                        if ratio != 0 {
                            bank.push(V4GraphKey { attention: if ratio == 4 { "c4" } else { "c128" }, ..key });
                        }
                    }
                }
            }
            if rank == 0 {
                bank.push(V4GraphKey { layer: TAIL_SEGMENT, rows, chunked: false, attention: "tail",
                    table_width: 0, table_stride: 0, previous: *reductions.last()?,
                    tp2_broadcast: broadcast, exchange_f32: tp2 && exchange_f32 });
            }
        }
    }
    }
    Some(keys)
}

pub fn v4_declared_graphs(ratios: &[usize], units: usize, decode_rows: usize,
    ranks: usize, reductions: &[usize], tp2: bool, exchange_f32: bool) -> Result<Vec<DeclaredGraphSet<V4GraphKey>>, String> {
    v4_serving_graph_keys(ratios, units, decode_rows, ranks, reductions, tp2, exchange_f32)
        .ok_or_else(|| "invalid V4 startup graph geometry".to_string())?
        .iter().map(|keys| DeclaredGraphSet::new(keys)).collect()
}

/// Count without materializing keys while the planner searches pool sizes.
/// All graph bytes are Ready-resident Estimated driver storage, no growth.
pub fn v4_startup_graphs(ratios: &[usize], units: usize, decode_rows: usize, ranks: usize, tp2: bool) -> Option<GraphSet> {
    if ratios.is_empty() || !(1..=2).contains(&ranks) || decode_rows == 0
        || ratios.iter().any(|ratio| ![0, 4, 128].contains(ratio)) { return None; }
    let classes = c4_width_classes(units).len() as u64;
    let shapes = (decode_rows as u64).checked_mul(2)?.checked_mul(classes)?.checked_mul(if tp2 { 2 } else { 1 })?;
    let segments = (ratios.len() as u64).checked_mul(classes)?
        .checked_add(ratios.iter().filter(|&&ratio| ratio != 0).count() as u64)?;
    let fronts = segments.checked_mul(2)?.checked_mul(decode_rows as u64)?;
    let mut executables = vec![fronts; ranks];
    executables[0] = fronts.checked_add(decode_rows as u64)?;
    if tp2 { for count in &mut executables { *count = count.checked_mul(2)?; } }
    executables.iter().try_for_each(|count| count.checked_mul(ESTIMATED_EXECUTABLE_SEED).map(|_| ()))?;
    Some(GraphSet::new(&executables, ESTIMATED_EXECUTABLE_SEED, 0, 0, shapes, Lifetime::Startup))
}

#[cfg(test)]
mod deepseek_v4_tests {
    use super::*;
    #[test]
    fn deterministic_unique_startup_keys_match_one_and_two_rank_counts() {
        let ratios = [0, 4, 128, 4];
        for ranks in [1, 2] {
            for units in [1, 48, 64, 65, 8192] {
                let reductions = [LOCAL_EXPERTS, 2, 2, LOCAL_EXPERTS];
                let keys = v4_serving_graph_keys(&ratios, units, 16, ranks, &reductions, false, false).unwrap();
                assert_eq!(keys, v4_serving_graph_keys(&ratios, units, 16, ranks, &reductions, false, false).unwrap());
                let declared = v4_declared_graphs(&ratios, units, 16, ranks, &reductions, false, false).unwrap();
                let set = v4_startup_graphs(&ratios, units, 16, ranks, false).unwrap();
                for rank in 0..ranks {
                    declared[rank].verify(keys[rank].iter()).unwrap();
                    assert_eq!(keys[rank].len() as u64, set.ranks[rank].executables);
                    assert_eq!(set.bytes(rank), set.ranks[rank].executables * 106_496);
                    assert_eq!(set.at_ready(rank), set.bytes(rank));
                    assert_eq!(set.growth(rank), 0);
                }
            }
        }
    }
    #[test]
    fn tp2_preserves_broadcast_variants_and_exact_exchange_dtype() {
        for exchange_f32 in [false, true] {
            let banks = v4_serving_graph_keys(&[0, 4, 128], 65, 16, 2,
                &[LOCAL_EXPERTS; 3], true, exchange_f32).unwrap();
            let charge = v4_startup_graphs(&[0, 4, 128], 65, 16, 2, true).unwrap();
            let eager = v4_startup_graphs(&[0, 4, 128], 65, 16, 2, false).unwrap();
            for (rank, bank) in banks.iter().enumerate() {
                let declared = DeclaredGraphSet::new(bank).unwrap();
                declared.verify(bank.iter()).unwrap();
                assert_eq!(bank.len() as u64, charge.ranks[rank].executables);
                assert_eq!(charge.bytes(rank), 2 * eager.bytes(rank));
                for key in bank {
                    assert_eq!(key.exchange_f32, exchange_f32);
                    assert!(declared.contains(&V4GraphKey { tp2_broadcast: !key.tp2_broadcast, ..*key }));
                }
            }
        }
    }

    #[test]
    fn invalid_geometry_refuses_before_enumeration() {
        assert!(v4_serving_graph_keys(&[4], 64, 16, 2, &[], false, false).is_none());
        for (ratios, rows, ranks) in [(vec![], 16, 2), (vec![2], 16, 2),
            (vec![4], 0, 2), (vec![4], 16, 0), (vec![4], 16, 3)] {
            assert!(v4_startup_graphs(&ratios, 64, rows, ranks, false).is_none());
        }
    }

    #[test]
    fn previous_owner_is_exact_and_tail_has_no_geometry() {
        let keys = v4_serving_graph_keys(&[0, 4, 128], 48, 2, 2,
            &[LOCAL_EXPERTS, 2, SKIPPED_EXPERTS], false, false).unwrap();
        for key in &keys[0] {
            assert_eq!(key.previous, match key.layer { 0 => 0, 1 => LOCAL_EXPERTS, 2 => 2, _ => SKIPPED_EXPERTS });
            if key.attention == "tail" { assert_eq!((key.table_width, key.table_stride), (0, 0)); }
        }
        assert!(keys[1].iter().all(|key| key.previous == usize::from(key.layer > 0)));
    }
    #[test]
    fn flash_split_43_layers_and_three_eager_draft_stages() {
        let ratios: Vec<_> = (0..43).map(|layer| if layer < 2 { 0 } else if layer % 2 == 0 { 4 } else { 128 }).collect();
        let set = v4_startup_graphs(&ratios, 8192, 16, 2, false).unwrap();
        assert_eq!(set.ranks.iter().map(|rank| rank.executables).collect::<Vec<_>>(), vec![20592, 20576]);
        assert_eq!(set.ranks.iter().map(|rank| rank.executables).sum::<u64>(), 41168);
    }
}
