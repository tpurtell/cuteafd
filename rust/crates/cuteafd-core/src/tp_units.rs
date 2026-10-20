//! Checkpoint-independent tensor-parallel ownership of aligned intermediate units.
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{collections::BTreeMap, ops::Range};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum UnitRule {
    Contiguous,
    LayerRotation,
    Rotated,
}

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum UnitMapError {
    #[error("TP unit geometry needs positive experts/unit, whole intermediate units, and nonempty contiguous ranks")]
    Geometry,
    #[error("TP unit, expert or rank is outside the geometry")]
    Index,
}

/// Exact-even ownership stays contiguous; uneven units balance across layers
/// (native K32) or experts and layers (H128).
pub fn balanced_rule(unit_columns: usize, units: usize, world: usize) -> UnitRule {
    if world != 0 && units != 0 && units % world == 0 {
        UnitRule::Contiguous
    } else if unit_columns == 32 {
        UnitRule::LayerRotation
    } else {
        UnitRule::Rotated
    }
}

/// Fixed-width kernels keep one contiguous source range per layer. Exact-even
/// native-unit splits do not rotate, preserving their original arithmetic.
pub fn layer_rule(units: usize, world: usize) -> UnitRule {
    if world != 0 && units != 0 && units % world == 0 {
        UnitRule::Contiguous
    } else {
        UnitRule::LayerRotation
    }
}

/// Owner of unit `u` of expert `e` in layer `layer`. All arithmetic is integer
/// and independent of checkpoint payloads, traffic, and installed packages.
pub fn unit_owner(
    rule: UnitRule,
    units: usize,
    expert: usize,
    unit: usize,
    layer: usize,
    world: usize,
) -> Result<usize, UnitMapError> {
    if units == 0 || world == 0 || (world > units && rule != UnitRule::Rotated) {
        return Err(UnitMapError::Geometry);
    }
    if unit >= units {
        return Err(UnitMapError::Index);
    }
    // Reduce before multiplying, including for callers using unbounded expert IDs.
    let modular_sum = |a: usize, b: usize| ((a as u128 + b as u128) % world as u128) as usize;
    if rule == UnitRule::Rotated {
        let expert_offset = ((units as u128 * expert as u128) % world as u128) as usize;
        return Ok(modular_sum(modular_sum(expert_offset, unit), layer));
    }
    let (base, extra) = (units / world, units % world);
    let cut = extra * (base + 1);
    let rank = if unit < cut {
        unit / (base + 1)
    } else {
        extra + (unit - cut) / base
    };
    Ok(if rule == UnitRule::LayerRotation {
        modular_sum(rank, layer)
    } else {
        rank
    })
}

/// A rank's complete unit map, packed in ascending source-column order per expert.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RankUnits {
    pub layer: usize,
    pub rank: usize,
    pub world: usize,
    pub unit_columns: usize,
    pub units_per_expert: usize,
    pub rule: UnitRule,
    pub experts: Vec<Vec<usize>>,
}

impl RankUnits {
    pub fn new(
        rule: UnitRule,
        experts: usize,
        intermediate: usize,
        unit_columns: usize,
        layer: usize,
        world: usize,
        rank: usize,
    ) -> Result<Self, UnitMapError> {
        if experts == 0
            || unit_columns == 0
            || intermediate == 0
            || intermediate % unit_columns != 0
            || world == 0
            || (world > intermediate / unit_columns && rule != UnitRule::Rotated)
        {
            return Err(UnitMapError::Geometry);
        }
        if rank >= world {
            return Err(UnitMapError::Index);
        }
        let units_per_expert = intermediate / unit_columns;
        let experts = (0..experts)
            .map(|expert| {
                (0..units_per_expert)
                    .filter(|&unit| {
                        unit_owner(rule, units_per_expert, expert, unit, layer, world) == Ok(rank)
                    })
                    .collect()
            })
            .collect();
        Ok(Self {
            layer,
            rank,
            world,
            unit_columns,
            units_per_expert,
            rule,
            experts,
        })
    }

    pub fn column_ranges(&self, expert: usize) -> Result<Vec<Range<usize>>, UnitMapError> {
        let units = self.experts.get(expert).ok_or(UnitMapError::Index)?;
        let mut ranges: Vec<Range<usize>> = Vec::new();
        for &unit in units {
            let start = unit * self.unit_columns;
            if let Some(range) = ranges.last_mut().filter(|range| range.end == start) {
                range.end += self.unit_columns;
            } else {
                ranges.push(start..start + self.unit_columns);
            }
        }
        Ok(ranges)
    }

    pub fn width_classes(&self) -> BTreeMap<usize, Vec<usize>> {
        let mut classes = BTreeMap::<usize, Vec<usize>>::new();
        for (expert, units) in self.experts.iter().enumerate() {
            classes
                .entry(units.len() * self.unit_columns)
                .or_default()
                .push(expert);
        }
        classes
    }

    pub fn resident_units(&self) -> usize {
        self.experts.iter().map(Vec::len).sum()
    }

    /// Canonical little-endian seal of geometry and source units, not host paths.
    pub fn fingerprint(&self) -> String {
        let mut hash = Sha256::new();
        hash.update(b"cuteafd-tp-units-v1");
        for value in [
            self.layer,
            self.rank,
            self.world,
            self.unit_columns,
            self.units_per_expert,
            self.experts.len(),
        ] {
            hash.update((value as u64).to_le_bytes());
        }
        for units in &self.experts {
            hash.update((units.len() as u64).to_le_bytes());
            for &unit in units {
                hash.update((unit as u64).to_le_bytes());
            }
        }
        format!("{:x}", hash.finalize())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_unit_has_one_owner_and_each_expert_is_balanced() {
        for rule in [
            UnitRule::Contiguous,
            UnitRule::LayerRotation,
            UnitRule::Rotated,
        ] {
            for (experts, intermediate, unit) in
                [(256, 2048, 128), (384, 2048, 32), (512, 640, 128)]
            {
                for world in 2..=6.min(intermediate / unit) {
                    for layer in [0, 3, 40, 77] {
                        let maps = (0..world)
                            .map(|rank| {
                                RankUnits::new(
                                    rule,
                                    experts,
                                    intermediate,
                                    unit,
                                    layer,
                                    world,
                                    rank,
                                )
                                .unwrap()
                            })
                            .collect::<Vec<_>>();
                        for expert in 0..experts {
                            let mut owned = maps
                                .iter()
                                .flat_map(|map| map.experts[expert].iter().copied())
                                .collect::<Vec<_>>();
                            owned.sort_unstable();
                            assert_eq!(owned, (0..intermediate / unit).collect::<Vec<_>>());
                            let counts = maps
                                .iter()
                                .map(|map| map.experts[expert].len())
                                .collect::<Vec<_>>();
                            assert!(
                                counts.iter().max().unwrap() - counts.iter().min().unwrap() <= 1
                            );
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn prototype_map_fingerprints() {
        // The Python prototype independently computes these complete maps.
        for (rule, experts, intermediate, unit, layer, world, rank, expected) in [
            (UnitRule::Contiguous, 256, 2048, 128, 40, 6, 4,
                "79b5adba7abcd261e5647d3009a1e22b4596cad2dfd2526188b2f3ff9bdb0050"),
            (UnitRule::LayerRotation, 384, 2048, 32, 77, 6, 1,
                "f5b8719aad670d16f996bd88d91d6c654f50aad555dba8993a1df5fd14105f77"),
            (UnitRule::Rotated, 256, 2048, 128, 3, 6, 0,
                "1211457b2cd49f09d118754400217a0db80d3a537958c2996bd6b89e3cefecd3"),
            (UnitRule::Rotated, 512, 640, 128, 77, 4, 3,
                "f44725d538ecc935c4f32fb1321229f5296415b6c6493892f03e02f2c898cfbe"),
        ] {
            assert_eq!(RankUnits::new(rule, experts, intermediate, unit, layer, world, rank)
                .unwrap().fingerprint(), expected);
        }
    }

    #[test]
    fn fixed_prototype_vectors_and_exlr8_reference() {
        for (e, u, layer, world) in [
            (0, 0, 3, 6),
            (7, 15, 40, 6),
            (255, 9, 77, 4),
            (256, 5, 40, 6),
        ] {
            assert_eq!(
                unit_owner(UnitRule::Rotated, 16, e, u, layer, world).unwrap(),
                (16 * e + u + layer) % world
            );
        }
        let rank = RankUnits::new(UnitRule::Rotated, 256, 2048, 128, 3, 6, 0).unwrap();
        assert_eq!(rank.experts[0], [3, 9, 15]);
        assert_eq!(
            rank.column_ranges(0).unwrap(),
            [384..512, 1152..1280, 1920..2048]
        );
        assert_eq!(
            rank.width_classes().keys().copied().collect::<Vec<_>>(),
            [256, 384]
        );
        for unit in [128, 32] {
            let widths = (0..6)
                .map(|r| {
                    RankUnits::new(UnitRule::Contiguous, 384, 2048, unit, 40, 6, r)
                        .unwrap()
                        .resident_units()
                        * unit
                        / 384
                })
                .collect::<Vec<_>>();
            assert_eq!(
                widths,
                if unit == 128 {
                    vec![384, 384, 384, 384, 256, 256]
                } else {
                    vec![352, 352, 352, 352, 320, 320]
                }
            );
        }
    }

    #[test]
    fn rotation_balances_memory_and_seals_source_columns() {
        let totals = (0..6)
            .map(|rank| {
                (0..6)
                    .map(|layer| {
                        RankUnits::new(UnitRule::LayerRotation, 384, 2048, 32, layer, 6, rank)
                            .unwrap()
                            .resident_units()
                    })
                    .sum::<usize>()
            })
            .collect::<Vec<_>>();
        assert!(totals.iter().all(|&total| total == 384 * 64));
        let maps = (0..6)
            .map(|rank| RankUnits::new(UnitRule::Rotated, 256, 2048, 128, 3, 6, rank).unwrap())
            .collect::<Vec<_>>();
        let totals = maps
            .iter()
            .map(RankUnits::resident_units)
            .collect::<Vec<_>>();
        assert_eq!(totals.iter().sum::<usize>(), 256 * 16);
        assert!(totals.iter().max().unwrap() - totals.iter().min().unwrap() <= 1);
        assert_ne!(maps[0].fingerprint(), maps[1].fingerprint());
    }

    #[test]
    fn even_balanced_maps_are_identical_to_contiguous() {
        for (intermediate, world) in [(2048, 2), (2048, 4), (2304, 3),
            (2304, 6), (1536, 6), (3072, 6)] {
            for unit in [32, 128] {
                for layer in [0, 3, 40, 77] {
                    for rank in 0..world {
                        let expected = RankUnits::new(UnitRule::Contiguous, 8,
                            intermediate, unit, layer, world, rank).unwrap();
                        let rule = balanced_rule(unit, intermediate / unit, world);
                        let actual = RankUnits::new(rule, 8, intermediate,
                            unit, layer, world, rank).unwrap();
                        assert_eq!(actual, expected);
                        assert_eq!(actual.fingerprint(), expected.fingerprint());
                        assert_eq!(actual.width_classes(), expected.width_classes());
                        for expert in 0..8 {
                            assert_eq!(actual.column_ranges(expert), expected.column_ranges(expert));
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn invalid_geometry_fails_closed() {
        for (i, u, n, r) in [
            (0, 32, 6, 0),
            (2048, 0, 6, 0),
            (2000, 128, 6, 0),
            (2048, 128, 0, 0),
            (640, 128, 6, 0),
            (2048, 128, 6, 6),
        ] {
            assert!(RankUnits::new(UnitRule::Contiguous, 384, i, u, 0, n, r).is_err());
        }
        assert_eq!(balanced_rule(32, 64, 6), UnitRule::LayerRotation);
        assert_eq!(balanced_rule(128, 16, 6), UnitRule::Rotated);
    }
}
