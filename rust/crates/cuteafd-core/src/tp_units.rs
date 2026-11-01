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

/// Checkpoint-provenanced export coverage, shared by planner and wire admission.
pub fn spark_package_entry(key: &str) -> &'static serde_json::Value {
    static DATA: std::sync::OnceLock<serde_json::Value> = std::sync::OnceLock::new();
    &DATA.get_or_init(|| serde_json::from_str(include_str!(
        "../../../../scripts/build/spark-package-ranges.json")).expect("checked package ranges"))["entries"][key]
}

/// Additive width-keyed native ABI roles never encode Spark count or rank.
pub const NATIVE_WIDTH_ROLE_BASE: u32 = 65536;

pub fn native_width_role(width: usize) -> Option<u32> {
    (width > 0 && width % 32 == 0 && width <= 8192)
        .then(|| NATIVE_WIDTH_ROLE_BASE + width as u32)
}

pub fn native_role_width(role: u32) -> Option<usize> {
    let width = role.checked_sub(NATIVE_WIDTH_ROLE_BASE)? as usize;
    (native_width_role(width) == Some(role)).then_some(width)
}

/// Executable identity and resident geometry, independent of source ownership.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NativeArtifact {
    pub role: u32,
    pub logical: usize,
    pub stored: usize,
    pub name: String,
}

/// One world's distinct adjacent artifacts. Equal worlds retain their ABI names.
pub fn native_spark_artifacts(hidden: usize, intermediate: usize, world: usize)
    -> Result<Vec<NativeArtifact>, UnitMapError> {
    let unit = crate::mxfp4_layouts::ownership_unit(intermediate, world);
    if !(2..=crate::MAX_SPARK_WORLD).contains(&world) || intermediate == 0
        || intermediate % unit != 0 || intermediate / unit < world {
        return Err(UnitMapError::Geometry);
    }
    let units = intermediate / unit;
    let unequal = units % world != 0;
    let mut widths = vec![units.div_ceil(world) * unit];
    if unequal { widths.push(units / world * unit); }
    widths.into_iter().map(|logical| {
        let role = if unequal { native_width_role(logical).ok_or(UnitMapError::Geometry)? }
            else { match world { 2 => 5, 3 => 6, 4 => 1, 6 => 7, 8 => 8,
                _ => return Err(UnitMapError::Geometry) } };
        let stored = crate::mxfp4_layouts::stored_width(hidden, logical).ok_or(UnitMapError::Geometry)?;
        let name = if unequal { format!("spark_w{logical}") }
            else if world == 4 { "spark".to_owned() } else { format!("spark_tp{world}") };
        Ok(NativeArtifact { role, logical, stored, name })
    }).collect()
}

/// A frozen (deprecated) format's admitted Spark worlds: the cards it shipped
/// or was admitted on before it froze. Its packaged worlds may be wider (a
/// historical role or a tail sibling's layout), but a frozen format never
/// admits a new world. `None` for formats that are not frozen.
pub fn frozen_admitted_worlds(family: &str, format: &str) -> Option<Vec<usize>> {
    let format = if format.starts_with("exl3") { "exl3" } else { format.strip_suffix("a4").unwrap_or(format) };
    let entry = spark_package_entry(&format!("{family}:{format}"));
    (entry["deprecated"].as_bool() == Some(true)).then(|| entry["admitted_worlds"].as_array()
        .expect("a frozen package range names its admitted worlds").iter()
        .filter_map(|world| world.as_u64().map(|world| world as usize)).collect())
}

/// Descriptor ABI and stored columns determine code, not rank identity.
pub fn package_width_keys(intermediate: usize, unit: usize, world: usize,
    exl3: bool, native: bool, nvfp4: bool) -> Vec<(bool, usize)> {
    let unit = if native && !exl3 && !nvfp4 { crate::mxfp4_layouts::ownership_unit(intermediate, world) } else { unit };
    if !(1..=crate::MAX_SPARK_WORLD).contains(&world) || unit == 0 || intermediate == 0
        || intermediate % unit != 0 || intermediate / unit < world { return vec![]; }
    let units = intermediate / unit;
    if exl3 && units % world != 0 { return vec![(true, units.div_ceil(world) * unit)]; }
    if nvfp4 && intermediate % (128 * world) != 0 {
        return vec![(false, (units.div_ceil(world) * unit).div_ceil(128) * 128)];
    }
    let mut widths = vec![(false, units / world * unit)];
    if units % world != 0 { widths.push((false, units.div_ceil(world) * unit)); }
    widths
}

pub fn spark_package_worlds(family: &str, format: &str) -> Vec<usize> {
    let format = if format.starts_with("exl3") { "exl3" } else { format.strip_suffix("a4").unwrap_or(format) };
    let entry = spark_package_entry(&format!("{family}:{format}"));
    let seeds = entry["worlds"].as_array().map(|items| items.iter()
        .filter_map(|item| item.as_u64().map(|n| n as usize)).collect::<Vec<_>>()).unwrap_or_default();
    if entry["deprecated"].as_bool() == Some(true) { return seeds; }
    let intermediate = entry["intermediate"].as_u64().unwrap_or(0) as usize;
    let exl3 = format == "exl3";
    let native = matches!(family, "v41" | "dsv4f" | "dsv4p") && !exl3;
    let unit = if native || matches!(family, "mimof" | "mimop") { 32 } else { 128 };
    let available = seeds.into_iter().flat_map(|world|
        package_width_keys(intermediate, unit, world, exl3, native, false)).collect::<std::collections::BTreeSet<_>>();
    (1..=crate::MAX_SPARK_WORLD).filter(|&world| {
        let needed = package_width_keys(intermediate, unit, world, exl3, native, false);
        !needed.is_empty()
            && needed.iter().all(|key| available.contains(key))
    }).collect()
}

/// Wire topology qualification is format-independent; checkpoint admission still
/// checks the selected family's recipe and native whole-K32 ownership.
pub fn spark_ownership_topology(tp: usize, ep: usize) -> bool {
    (ep == 1 && ["v41", "dsv4f", "dsv4p"].iter()
        .any(|family| spark_package_worlds(family, "mxfp4").contains(&tp)))
        || matches!((tp, ep), (2, 2) | (3, 2) | (2, 3))
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

/// One adjacent exact shard, rotated by layer only when native units do not
/// divide evenly. Logical and resident columns are the same interval.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LayerShard {
    pub first: usize,
    pub width: usize,
}

impl LayerShard {
    pub fn new(intermediate: usize, unit: usize, layer: usize, world: usize, rank: usize)
        -> Result<Self, UnitMapError> {
        if unit == 0 || intermediate == 0 || intermediate % unit != 0
            || world == 0 || world > intermediate / unit {
            return Err(UnitMapError::Geometry);
        }
        if rank >= world { return Err(UnitMapError::Index); }
        let units = intermediate / unit;
        let (base, extra) = (units / world, units % world);
        let canonical = if layer_rule(units, world) == UnitRule::LayerRotation {
            ((rank as u128 + world as u128 - (layer % world) as u128) % world as u128) as usize
        } else { rank };
        Ok(Self {
            first: (canonical * base + canonical.min(extra)) * unit,
            width: (base + usize::from(canonical < extra)) * unit,
        })
    }

    pub fn source_range(self) -> Range<usize> { self.first..self.first + self.width }
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
    fn exact_k32_layer_shards_cover_every_geometry_and_world() {
        let tables: &[(usize, &[&[usize]])] = &[
            (2304, &[&[1152, 1152], &[768, 768, 768], &[576; 4],
                &[480, 480, 448, 448, 448], &[384; 6],
                &[352, 352, 320, 320, 320, 320, 320], &[288; 8]]),
            (2048, &[&[1024, 1024], &[704, 672, 672], &[512; 4],
                &[416, 416, 416, 416, 384], &[352, 352, 352, 352, 320, 320],
                &[320, 288, 288, 288, 288, 288, 288], &[256; 8]]),
            (3072, &[&[1536, 1536], &[1024; 3], &[768; 4],
                &[640, 608, 608, 608, 608], &[512; 6],
                &[448, 448, 448, 448, 448, 416, 416], &[384; 8]]),
        ];
        for &(intermediate, table) in tables {
            for (index, &widths) in table.iter().enumerate() {
                let world = index + 2;
                let unequal = intermediate / 32 % world != 0;
                let mut totals = vec![0; world];
                for layer in 0..world {
                    let mut ranges = Vec::new();
                    for rank in 0..world {
                        let shard = LayerShard::new(intermediate, 32, layer, world, rank).unwrap();
                        let canonical = if unequal { (rank + world - layer) % world } else { rank };
                        assert_eq!(shard.width, widths[canonical]);
                        assert_eq!(shard.first, widths[..canonical].iter().sum::<usize>());
                        assert_eq!(shard.first % 32, 0);
                        assert_eq!(shard.width % 32, 0);
                        let map = RankUnits::new(layer_rule(intermediate / 32, world), 3,
                            intermediate, 32, layer, world, rank).unwrap();
                        for expert in 0..3 {
                            assert_eq!(map.column_ranges(expert).unwrap(), [shard.source_range()]);
                        }
                        totals[rank] += shard.width;
                        ranges.push(shard.source_range());
                    }
                    ranges.sort_by_key(|range| range.start);
                    assert_eq!(ranges[0].start, 0);
                    assert_eq!(ranges.last().unwrap().end, intermediate);
                    assert!(ranges.windows(2).all(|pair| pair[0].end == pair[1].start));
                }
                assert_eq!(totals, vec![intermediate; world]);
            }
        }
        for (intermediate, unit, world, rank) in [(0, 32, 2, 0), (2048, 0, 2, 0),
            (2047, 32, 2, 0), (32, 32, 2, 0), (2048, 32, 0, 0), (2048, 32, 2, 2)] {
            assert!(LayerShard::new(intermediate, unit, usize::MAX, world, rank).is_err());
        }
        assert_eq!(LayerShard::new(2048, 32, usize::MAX, 3, 1).unwrap(),
            LayerShard::new(2048, 32, usize::MAX % 3, 3, 1).unwrap());
    }

    #[test]
    fn native_world_artifacts_cover_rotated_shards_and_keep_equal_abi_ids() {
        for (hidden, intermediate) in [(5120, 2304), (4096, 2048), (7168, 3072)] {
            for world in 2..=8 {
                let artifacts = native_spark_artifacts(hidden, intermediate, world).unwrap();
                let unit = crate::mxfp4_layouts::ownership_unit(intermediate, world);
                let unequal = intermediate / unit % world != 0;
                assert_eq!(artifacts.len(), if unequal { 2 } else { 1 });
                for artifact in &artifacts {
                    assert_eq!(artifact.stored, crate::mxfp4_layouts::stored_width(hidden, artifact.logical).unwrap());
                    assert!(artifact.stored <= artifact.logical.div_ceil(128) * 128);
                    if unequal {
                        assert_eq!(native_role_width(artifact.role), Some(artifact.logical));
                        assert_eq!(artifact.name, format!("spark_w{}", artifact.logical));
                    } else {
                        let role = match world { 2 => 5, 3 => 6, 4 => 1, 6 => 7, 8 => 8, _ => unreachable!() };
                        assert_eq!(artifact.role, role);
                        assert_eq!(artifact.name, if world == 4 { "spark".to_owned() } else { format!("spark_tp{world}") });
                    }
                }
                for layer in [0, 1, 17, 63] {
                    for rank in 0..world {
                        let shard = LayerShard::new(intermediate, unit, layer, world, rank).unwrap();
                        assert_eq!(artifacts.iter().filter(|a| a.logical == shard.width).count(), 1);
                    }
                }
            }
        }
        for (hidden, intermediate, world) in [(255, 2304, 4), (5120, 0, 4),
            (5120, 2303, 4), (5120, 2304, 1), (5120, 2304, 9)] {
            assert!(native_spark_artifacts(hidden, intermediate, world).is_err());
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
