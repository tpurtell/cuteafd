//! Routed-expert package layouts this build's packagers produce, and the
//! slice arithmetic the expert staging uses. The planner's placement check
//! reads these, so it accepts exactly the layouts an image can carry.
//!
//! - FP8 (`python/tools/aot/package_fp8_moe_aot.py`, b12x fp8_moe): Spark
//!   layouts tp4, tp2 and tp6 where they split the intermediate into whole
//!   128-row blocks (TP6 of any intermediate of at least six blocks, uneven
//!   blocks padded to the widest); a tp1 coordinator package.
//! - MXFP4 through the same programs (MiMo V2.6 Pro `mimop:fp8`): Spark tp6
//!   and tp2, a tp1 coordinator package.
//! - NVFP4 (ModelOpt) through the same programs (`glmf:nvfp4`, `qwen4:nvfp4`,
//!   `glm:nvfp4`): Spark tp4, tp2, tp3 and tp6 in whole 16-value blocks, a tp1
//!   coordinator package.
//! - EXL3 (`python/tools/aot/package_exl3_aot.py`): Spark worlds 4, 2, 3, and
//!   6 when the intermediate has at least six 128-row blocks; Qwen also has TP1.
//! - DeepSeek native experts (expertd-native MXFP4 / EXL3): 2, 3, 4 and 6.
//!
//! The expert transport (RoCE verbs, TCP) runs one through eight Spark ranks.
//! Package coverage below is independent of transport capacity.

/// Spark worlds the expert transport runs.
pub const TRANSPORT_WORLDS: [usize; 8] = [1, 2, 3, 4, 5, 6, 7, 8];

/// Default Spark layouts of an FP8 (E4M3, 128x128 scales) expert package.
pub fn fp8_spark_worlds(intermediate: usize) -> Vec<usize> {
    let blocks = intermediate / 128;
    [2usize, 4, 6]
        .into_iter()
        .filter(|&tp| intermediate % 128 == 0 && (intermediate % (128 * tp) == 0 || (tp == 6 && blocks >= 6)))
        .collect()
}

/// Default Spark layouts of an MXFP4 package on the fp8_moe programs.
pub fn mxfp4_spark_worlds() -> Vec<usize> {
    vec![2, 6]
}

/// Default Spark layouts of an NVFP4 package on the fp8_moe programs: the
/// transport worlds whose ranks each own a 16-value block.
pub fn nvfp4_spark_worlds(intermediate: usize) -> Vec<usize> {
    [2usize, 3, 4, 6].into_iter().filter(|&tp| intermediate % 16 == 0 && intermediate / 16 >= tp).collect()
}

/// Spark worlds of an EXL3 package (`package_exl3_aot.py` profiles).
pub fn exl3_spark_worlds(intermediate: usize) -> Vec<usize> {
    let blocks = intermediate / 128;
    if intermediate % 128 != 0 || blocks < 2 {
        return Vec::new();
    }
    let mut worlds = vec![2, 3, 4];
    if blocks >= 6 {
        worlds.push(6);
    }
    worlds.retain(|&tp| blocks >= tp);
    worlds
}

/// The stored intermediate slice of the widest rank of `tp`: whole `block`-row
/// blocks, as evenly as the blocks allow, padded to 128 rows. `None` when the
/// intermediate does not split into whole blocks over `tp` ranks.
/// `formats::fp8_experts::Fp8ExpertTensors::slice` stages exactly this width.
pub fn stored_slice(intermediate: usize, block: usize, tp: usize) -> Option<usize> {
    (tp > 0 && block > 0 && intermediate % block == 0 && intermediate / block >= tp)
        .then(|| ((intermediate / block).div_ceil(tp) * block).div_ceil(128) * 128)
}

/// A build recipe is independent of the coordinator's installed package tree.
/// Generic exact-width packages use one program per distinct rank extent;
/// native MXFP4/NVFP4 retain their calibrated padded storage.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SparkRecipe {
    pub slicing: crate::formats::fp8_experts::Slicing,
    pub widths: Vec<usize>,
    pub required_layouts: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum SparkRecipeRefusal {
    #[error("geometry-invalid: intermediate {intermediate} cannot give {world} ranks nonempty whole-{block} row slices")]
    GeometryInvalid { intermediate: usize, world: usize, block: usize },
    #[error("missing Spark package recipe: {package} TP{world}")]
    MissingPackage { package: String, world: usize },
    #[error("unknown Spark package recipe: {0}")]
    UnknownPackage(String),
}

/// Qualified recipes only: optional installation checks cannot widen this set.
/// TP5 support is added per format only after its hardware gate passes.
pub fn spark_recipe(package: &str, intermediate: usize, world: usize) -> Result<SparkRecipe, SparkRecipeRefusal> {
    use crate::formats::fp8_experts::Slicing;
    let key = package.split_whitespace().next().unwrap_or(package);
    let (family, format) = key.split_once(':').ok_or_else(|| SparkRecipeRefusal::UnknownPackage(package.into()))?;
    let native = matches!(family, "v41" | "dsv4f" | "dsv4p");
    let exl3 = format.starts_with("exl3-k") && matches!(family, "v41" | "dsv4f" | "dsv4p" | "glm" | "glmf" | "qwen4");
    let nvfp4 = matches!(format, "nvfp4" | "nvfp4a4") && matches!(family, "v41" | "glm" | "glmf" | "qwen4");
    let mxfp4 = (format == "mxfp4" && native) || (format == "fp8" && matches!(family, "mimof" | "mimop"));
    let fp8 = format == "fp8" && matches!(family, "mimo" | "glm" | "glmf" | "qwen4");
    if !(exl3 || nvfp4 || mxfp4 || fp8) {
        return Err(SparkRecipeRefusal::UnknownPackage(package.into()));
    }
    let block = if nvfp4 && !native && !matches!(world, 1 | 5 | 7 | 8) { 16 } else { 128 };
    if !(1..=8).contains(&world) || intermediate == 0 || intermediate % block != 0 || intermediate / block < world {
        return Err(SparkRecipeRefusal::GeometryInvalid { intermediate, world, block });
    }
    let packaged = if native { matches!(world, 2 | 3 | 4 | 6) }
        else if exl3 { matches!(world, 2 | 3 | 4 | 6) || (family == "qwen4" && world == 1) }
        else if nvfp4 { matches!(world, 2 | 3 | 4 | 6) }
        else if mxfp4 { if family == "mimof" { matches!(world, 2 | 4) } else { matches!(world, 2 | 6) } }
        else { matches!(world, 2 | 4 | 6) && (intermediate % (128 * world) == 0 || world == 6) };
    if !packaged {
        return Err(SparkRecipeRefusal::MissingPackage { package: key.into(), world });
    }
    let exact = exl3 || (!native && intermediate % 128 == 0 && intermediate / 128 >= world);
    let widths = if exact {
        let blocks = intermediate / 128;
        (0..world).map(|rank| (blocks / world + usize::from(rank < blocks % world)) * 128).collect::<Vec<_>>()
    } else {
        vec![stored_slice(intermediate, if native { 128 } else { block }, world)
            .ok_or(SparkRecipeRefusal::GeometryInvalid { intermediate, world, block })?; world]
    };
    let required_layouts = if exl3 {
        (0..world).map(|rank| format!("tp{world}-rank{rank}")).collect()
    } else if exact && widths.iter().any(|width| *width != widths[0]) {
        let mut unique = widths.clone(); unique.sort_unstable_by(|a, b| b.cmp(a)); unique.dedup();
        unique.into_iter().map(|width| format!("tp{world}-w{width}")).collect()
    } else { vec![format!("tp{world}")] };
    Ok(SparkRecipe { slicing: if exact { Slicing::Blocks(128) } else { Slicing::Padded }, widths, required_layouts })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn recipes_preserve_exact_tp6_and_legacy_nvfp4() {
        use crate::formats::fp8_experts::Slicing;
        for package in ["mimo:fp8", "mimop:fp8 (MXFP4)", "glm:nvfp4", "glmf:exl3-k34"] {
            let recipe = spark_recipe(package, 2048, 6).unwrap();
            assert_eq!(recipe.slicing, Slicing::Blocks(128));
            assert_eq!(recipe.widths, [384, 384, 384, 384, 256, 256]);
            let expected = if package.contains("exl3") {
                (0..6).map(|rank| format!("tp6-rank{rank}")).collect::<Vec<_>>()
            } else { vec!["tp6-w384".into(), "tp6-w256".into()] };
            assert_eq!(recipe.required_layouts, expected);
        }
        let recipe = spark_recipe("qwen4:nvfp4", 640, 6).unwrap();
        assert_eq!(recipe.slicing, Slicing::Padded);
        assert_eq!(recipe.widths, [128; 6]);
        assert_eq!(recipe.required_layouts, ["tp6"]);
        assert_eq!(spark_recipe("v41:exl3-k23 (expertd-native)", 2304, 6).unwrap().widths, [384; 6]);
        for package in ["qwen4:nvfp4", "qwen4:exl3-k45", "qwen4:fp8"] {
            for world in [7, 8] {
                assert!(matches!(spark_recipe(package, 640, world), Err(SparkRecipeRefusal::GeometryInvalid { .. })));
            }
            assert!(matches!(spark_recipe(package, 640, 5), Err(SparkRecipeRefusal::MissingPackage { .. })));
        }
    }

    #[test]
    fn package_layouts_follow_the_packagers() {
        // 2048 = 16 blocks: every FP8 layout; Qwen's 640 = 5 blocks: none.
        assert_eq!(fp8_spark_worlds(2048), [2, 4, 6]);
        assert!(fp8_spark_worlds(640).is_empty());
        assert_eq!(exl3_spark_worlds(2048), [2, 3, 4, 6]);
        assert_eq!(exl3_spark_worlds(640), [2, 3, 4]);
        assert_eq!(stored_slice(2048, 128, 6), Some(384));
        assert_eq!(stored_slice(2048, 32, 6), Some(384));
        assert_eq!(stored_slice(2304, 128, 4), Some(640));
        assert_eq!(stored_slice(640, 128, 6), None);
        assert_eq!(stored_slice(2048, 128, 0), None);
        assert_eq!(nvfp4_spark_worlds(640), [2, 3, 4, 6]);
        assert_eq!(stored_slice(640, 16, 6), Some(128));
        assert_eq!(stored_slice(2048, 16, 6), Some(384));
    }
}
