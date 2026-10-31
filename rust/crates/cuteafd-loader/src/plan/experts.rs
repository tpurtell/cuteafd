//! Routed-expert package layouts this build's packagers produce, and the
//! slice arithmetic the expert staging uses. The planner's placement check
//! reads these, so it accepts exactly the layouts an image can carry.
//!
//! Export coverage is checkpoint-provenanced data shared with the packagers.
//! Admission compares descriptor ABI and native-unit width classes, retaining
//! real native-role geometry restrictions and deprecated NVFP4 coverage.

/// Spark worlds the expert transport runs.
pub const TRANSPORT_WORLDS: [usize; 8] = [1, 2, 3, 4, 5, 6, 7, 8];

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
    pub rule: cuteafd_core::tp_units::UnitRule,
    pub required_layouts: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum InstalledSparkRecipeRefusal {
    #[error("missing installed Spark package artifact: {0}")]
    MissingArtifact(std::path::PathBuf),
    #[error("padded Spark storage conflicts with the recipe's unequal exact rank widths")]
    PaddedExactMismatch,
    #[error("Spark rank {rank} selected width {selected}, but the recipe requires {expected}")]
    WidthMismatch { rank: usize, selected: usize, expected: usize },
    #[error("Spark rank {rank} is outside recipe world {world}")]
    InvalidRank { rank: usize, world: usize },
}

impl SparkRecipe {
    fn same_ownership(&self, other: &Self) -> bool {
        let slicing_matches = self.slicing == other.slicing
            || (matches!(self.slicing, crate::formats::fp8_experts::Slicing::Blocks(_))
                && matches!(other.slicing, crate::formats::fp8_experts::Slicing::Blocks(_))
                && self.rule == cuteafd_core::tp_units::UnitRule::Contiguous
                && other.rule == cuteafd_core::tp_units::UnitRule::Contiguous);
        // Different native scale units can describe the same exact contiguous
        // source ranges in mixed-format layers; storage and packages must agree.
        slicing_matches && self.widths == other.widths && self.rule == other.rule
            && self.required_layouts == other.required_layouts
    }

    pub fn rank_for_layer(&self, rank: usize, layer: usize) -> usize {
        if self.rule == cuteafd_core::tp_units::UnitRule::LayerRotation {
            (rank + self.widths.len() - layer % self.widths.len()) % self.widths.len()
        } else { rank }
    }

    pub fn width_for_layer(&self, rank: usize, layer: usize) -> Option<usize> {
        if rank >= self.widths.len() { return None; }
        self.widths.get(self.rank_for_layer(rank, layer)).copied()
    }

    /// Validate an explicit Spark package root, never the coordinator tree.
    /// Installation can narrow a registered recipe but cannot create one.
    pub fn validate_installed(&self, root: &std::path::Path, rank: usize,
        selected_slicing: crate::formats::fp8_experts::Slicing, selected_width: usize)
        -> Result<(), InstalledSparkRecipeRefusal> {
        if matches!(self.slicing, crate::formats::fp8_experts::Slicing::Blocks(_))
            && selected_slicing != self.slicing && self.widths.windows(2).any(|pair| pair[0] != pair[1]) {
            return Err(InstalledSparkRecipeRefusal::PaddedExactMismatch);
        }
        let expected = *self.widths.get(rank).ok_or(InstalledSparkRecipeRefusal::InvalidRank {
            rank, world: self.widths.len(),
        })?;
        if selected_width != expected {
            return Err(InstalledSparkRecipeRefusal::WidthMismatch { rank, selected: selected_width, expected });
        }
        for layout in &self.required_layouts {
            let directory = root.join(layout);
            let artifact = directory.join("libcuteafd_fp8moe.so");
            let installed = if layout.contains("-rank") {
                // EXL3 links one module per capacity beneath each rank layout.
                std::fs::read_dir(&directory).ok().is_some_and(|entries| entries.flatten().any(|entry|
                    entry.file_name().to_str().is_some_and(|name| name.starts_with('m'))
                        && entry.path().join("libcuteafd_exl3.so").is_file()))
            } else { artifact.is_file() };
            if !installed {
                return Err(InstalledSparkRecipeRefusal::MissingArtifact(if layout.contains("-rank") {
                    directory
                } else { artifact }));
            }
        }
        Ok(())
    }
}

/// The named deprecation refusal when any package of `contract` (keys joined
/// by " + ") is frozen and `world` is not one of its admitted worlds.
pub fn frozen_refusal(contract: &str, world: usize) -> Option<SparkRecipeRefusal> {
    contract.split(" + ").find_map(|key| {
        let (family, format) = key.split_once(':')?;
        let worlds = cuteafd_core::tp_units::frozen_admitted_worlds(family, format)?;
        (!worlds.contains(&world)).then(|| SparkRecipeRefusal::Deprecated { package: key.into(), world, worlds })
    })
}

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum SparkRecipeRefusal {
    #[error("geometry-invalid: intermediate {intermediate} cannot give {world} ranks nonempty whole-{block} row slices; native MXFP4 requires equal whole-K32 shards: add per-rank adjacent-width staging and matching native Spark role exports before admitting unequal widths")]
    GeometryInvalid { intermediate: usize, world: usize, block: usize },
    #[error("missing Spark package recipe: {package} TP{world}; export its balanced per-rank widths in package_exl3_aot.py (EXL3) or package_fp8_moe_aot.py (FP8/NVFP4), register spark_recipe, and include those layouts in the Spark image")]
    MissingPackage { package: String, world: usize },
    #[error("NVFP4 is deprecated; Spark worlds {worlds:?}; requested {package} TP{world}")]
    Deprecated { package: String, world: usize, worlds: Vec<usize> },
    #[error("native MXFP4 requires equal whole-K32 shards for intermediate {intermediate}, TP{world}; add native roles taking unequal whole-K32 shards, rotated by layer, and rank-sized staging")]
    NativeUnequal { intermediate: usize, world: usize },
    #[error("unknown Spark package recipe: {0}")]
    UnknownPackage(String),
    #[error("{0}")]
    Placement(cuteafd_core::tp_units::UnitMapError),
}

/// Enumerate recipes from the format policy and checkpoint geometry.
pub fn spark_worlds(package: &str, intermediate: usize) -> Vec<usize> {
    TRANSPORT_WORLDS.into_iter().filter(|&world| spark_recipe(package, intermediate, world).is_ok()).collect()
}

/// Validate every hosted routed operand against its checkpoint family and derive
/// one qualified storage recipe without inspecting any installed package tree.
pub fn checkpoint_spark_recipe(checkpoint: &super::Checkpoint, world: usize, layers: &[usize]) -> anyhow::Result<SparkRecipe> {
    use anyhow::{ensure, Context};
    let family = super::family::detect(checkpoint).context("Spark admission: unsupported checkpoint family")?;
    let model = family.open(checkpoint)?;
    let intermediate = model.spec().moe.as_ref().context("Spark admission: checkpoint has no routed experts")?.intermediate;
    ensure!(!layers.is_empty() && layers.windows(2).all(|pair| pair[0] < pair[1]),
        "Spark admission: hosted layers must be nonempty, sorted and unique");
    for &layer in layers {
        ensure!(model.spec().layers.get(layer).is_some_and(|spec| spec.ffn == super::spec::FfnKind::Moe),
            "Spark admission: hosted layer {layer} is not a routed target layer");
    }
    let modelopt = crate::formats::modelopt::ModelOpt::read(&checkpoint.snapshot, &checkpoint.config)?;
    let required = super::files::required_tensors(checkpoint, super::files::ReadRole::Spark { rank: 0, world })?
        .into_iter().filter(|name| family.classify(model.spec(), name)
            .and_then(|role| role.layer).is_some_and(|layer| layers.binary_search(&layer).is_ok()))
        .collect::<std::collections::BTreeSet<_>>();
    for &layer in layers {
        ensure!(required.iter().any(|name| family.classify(model.spec(), name)
            .is_some_and(|role| role.layer == Some(layer))),
            "Spark admission: hosted layer {layer} has no routed expert headers");
    }
    let tensors = checkpoint.tensors.iter().filter(|tensor| required.contains(&tensor.meta.name))
        .cloned().collect::<Vec<_>>();
    let present = tensors.iter().map(|tensor| tensor.meta.name.as_str()).collect::<std::collections::BTreeSet<_>>();
    for name in &required {
        ensure!(present.contains(name.as_str()), "Spark admission: missing routed expert tensor header {name}");
    }
    let groups = super::format::group_by_stem(&tensors);
    ensure!(!groups.is_empty(), "Spark admission: missing routed expert tensor headers");
    let mut selected = None;
    for (stem, members) in groups {
        let member = members.values().next().context("Spark admission: empty routed operand")?;
        let role = family.classify(model.spec(), &member.meta.name)
            .context("Spark admission: routed operand lost its family role")?;
        let mut operand = super::format::detect(&members)?;
        if let Some(modelopt) = &modelopt {
            modelopt.check(&stem, &operand).map_err(anyhow::Error::msg)
                .with_context(|| format!("Spark admission tensor {stem}"))?;
        }
        model.accepts(&role, &stem, &mut operand).map_err(anyhow::Error::msg)
            .with_context(|| format!("Spark admission tensor {stem}"))?;
        let contract = model.experts(&operand)
            .with_context(|| format!("Spark admission tensor {stem}: no package for {}", operand.label()))?;
        let recipe = spark_recipe(&contract.package, intermediate, world)
            .with_context(|| format!("Spark admission tensor {stem}"))?;
        ensure!(selected.as_ref().is_none_or(|previous: &SparkRecipe| previous.same_ownership(&recipe)),
            "Spark admission tensor {stem}: routed formats require conflicting Spark recipes");
        selected = Some(recipe);
    }
    selected.context("Spark admission: missing routed expert storage recipe")
}

use cuteafd_core::tp_units::package_width_keys;

/// Registered recipes only: optional installation checks cannot widen this set.
/// Step-5 TP6 geometry is registered for qualification; its hardware gate is pending.
/// Package availability is separate from the CPU-qualified transport range.
pub fn spark_recipe(package: &str, intermediate: usize, world: usize) -> Result<SparkRecipe, SparkRecipeRefusal> {
    use crate::formats::fp8_experts::Slicing;
    let key = package.split_whitespace().next().unwrap_or(package);
    let (family, format) = key.split_once(':').ok_or_else(|| SparkRecipeRefusal::UnknownPackage(package.into()))?;
    let native = matches!(family, "v41" | "dsv4f" | "dsv4p");
    let exl3 = format.strip_prefix("exl3-k").is_some_and(|bits| (1..=2).contains(&bits.len())
        && bits.bytes().all(|bit| (b'2'..=b'6').contains(&bit))
        && (bits.len() == 1 || bits.as_bytes()[0] != bits.as_bytes()[1]))
        && matches!(family, "dsv4p" | "glm" | "glmf" | "qwen4" | "kimik3" | "step5");
    let nvfp4 = (matches!(format, "nvfp4" | "nvfp4a4") && matches!(family, "v41" | "glm" | "glmf" | "qwen4"))
        || (family == "step5" && format == "nvfp4");
    let mxfp4 = (format == "mxfp4" && native) || (format == "fp8" && matches!(family, "mimof" | "mimop"));
    let fp8 = format == "fp8" && matches!(family, "mimo" | "glm" | "glmf" | "qwen4" | "step5");
    let bf16 = family == "step5" && format == "bf16";
    if !(exl3 || nvfp4 || mxfp4 || fp8 || bf16) {
        return Err(SparkRecipeRefusal::UnknownPackage(package.into()));
    }
    let block = if nvfp4 { 16 } else if mxfp4 { 32 } else { 128 };
    if !(1..=8).contains(&world) || intermediate == 0 || intermediate % block != 0 || intermediate / block < world {
        return Err(SparkRecipeRefusal::GeometryInvalid { intermediate, world, block });
    }
    // Native family roles and staging require equal whole-K32 shards. Generic
    // MXFP4 tail exports do not make those native roles executable by themselves.
    if native && mxfp4 && (intermediate % world != 0 || intermediate / world % 32 != 0) {
        return Err(SparkRecipeRefusal::NativeUnequal { intermediate, world });
    }
    let coverage_key = format!("{family}:{}", if exl3 { "exl3" } else if nvfp4 { "nvfp4" } else { format });
    let coverage = cuteafd_core::tp_units::spark_package_entry(&coverage_key);
    let worlds = coverage["worlds"].as_array().map(|values| values.iter()
        .filter_map(|value| value.as_u64().map(|value| value as usize)).collect::<Vec<_>>()).unwrap_or_default();
    if coverage["deprecated"].as_bool() == Some(true) && !worlds.contains(&world) {
        return Err(SparkRecipeRefusal::Deprecated { package: key.into(), world, worlds });
    }
    let packaged_intermediate = coverage["intermediate"].as_u64().unwrap_or(0) as usize;
    let required = package_width_keys(intermediate, block, world, exl3, native, nvfp4);
    let available = worlds.iter().flat_map(|&count|
        package_width_keys(packaged_intermediate, block, count, exl3, native, nvfp4))
        .collect::<std::collections::BTreeSet<_>>();
    if required.is_empty() || !required.iter().all(|width| available.contains(width)) {
        return Err(SparkRecipeRefusal::MissingPackage { package: key.into(), world });
    }
    let compact = exl3 && intermediate / block % world != 0;
    // EXL3 gathers whole trellis units per expert. Fixed-width kernels instead
    // rotate contiguous native-unit ranges by layer, retaining NVFP4 padding.
    let rule = if compact { cuteafd_core::tp_units::UnitRule::Rotated }
        else { cuteafd_core::tp_units::layer_rule(intermediate / block, world) };
    let exact = exl3 || (!native && (!nvfp4 || intermediate % (128 * world) == 0));
    let widths = if compact {
        vec![(intermediate / block).div_ceil(world) * block; world]
    } else if exact {
        (0..world).map(|rank| cuteafd_core::tp_units::RankUnits::new(rule, 1, intermediate, block, 0, world, rank)
            .map(|map| map.resident_units() * block).map_err(SparkRecipeRefusal::Placement)).collect::<Result<Vec<_>, _>>()?
    } else {
        vec![stored_slice(intermediate, block, world)
            .ok_or(SparkRecipeRefusal::GeometryInvalid { intermediate, world, block })?; world]
    };
    let required_layouts = if exl3 {
        let tag = if compact { "-compact" } else { "" };
        (0..world).map(|rank| format!("tp{world}{tag}-rank{rank}")).collect()
    } else if exact && widths.iter().any(|width| *width != widths[0]) {
        let mut unique = widths.clone(); unique.sort_unstable_by(|a, b| b.cmp(a)); unique.dedup();
        unique.into_iter().map(|width| format!("tp{world}-w{width}")).collect()
    } else { vec![format!("tp{world}")] };
    Ok(SparkRecipe { slicing: if exact { Slicing::Blocks(block) } else { Slicing::Padded }, widths, rule, required_layouts })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn checkpoint_spark_recipe(checkpoint: &super::super::Checkpoint, world: usize) -> anyhow::Result<SparkRecipe> {
        let family = super::super::family::detect(checkpoint).unwrap();
        let model = family.open(checkpoint)?;
        let layers = model.spec().layers.iter().enumerate()
            .filter_map(|(layer, spec)| (spec.ffn == super::super::spec::FfnKind::Moe).then_some(layer)).collect::<Vec<_>>();
        super::checkpoint_spark_recipe(checkpoint, world, &layers)
    }

    #[test]
    fn balanced_mxfp4_tp6_rotates_one_exact_width_per_layer() {
        use cuteafd_core::tp_units::UnitRule;
        let recipe = spark_recipe("mimop:fp8", 2048, 6).unwrap();
        assert_eq!(recipe.slicing, crate::formats::fp8_experts::Slicing::Blocks(32));
        assert_eq!(recipe.rule, UnitRule::LayerRotation);
        assert_eq!(recipe.widths, [352, 352, 352, 352, 320, 320]);
        assert_eq!(recipe.required_layouts, ["tp6-w352", "tp6-w320"]);
        for layer in [0, 3, 40, 77] {
            let widths = (0..6).map(|rank| recipe.width_for_layer(rank, layer).unwrap()).collect::<Vec<_>>();
            assert_eq!(widths.iter().sum::<usize>(), 2048);
            assert_eq!(widths.iter().filter(|&&width| width == 352).count(), 4);
            for rank in 0..6 { assert_eq!(widths[(rank + layer) % 6], recipe.widths[rank]); }
        }
        for rank in 0..6 {
            assert_eq!((0..6).map(|layer| recipe.width_for_layer(rank, layer).unwrap()).sum::<usize>(), 2048);
        }
        assert_eq!(recipe.width_for_layer(6, 1), None);
        assert_eq!(spark_recipe("mimop:fp8", 2048, 2).unwrap().rule, UnitRule::Contiguous);
    }

    #[test]
    fn balanced_mimo_flash_tp3_rotates_exact_k32_widths() {
        use cuteafd_core::tp_units::UnitRule;
        let recipe = spark_recipe("mimof:fp8", 2048, 3).unwrap();
        assert_eq!(recipe.slicing, crate::formats::fp8_experts::Slicing::Blocks(32));
        assert_eq!(recipe.rule, UnitRule::LayerRotation);
        assert_eq!(recipe.widths, [704, 672, 672]);
        assert_eq!(recipe.required_layouts, ["tp3-w704", "tp3-w672"]);
        for layer in [0, 1, 2, 40, 77] {
            let widths = (0..3).map(|rank| recipe.width_for_layer(rank, layer).unwrap()).collect::<Vec<_>>();
            assert_eq!(widths.iter().sum::<usize>(), 2048);
            for rank in 0..3 { assert_eq!(widths[(rank + layer) % 3], recipe.widths[rank]); }
        }
        for rank in 0..3 {
            assert_eq!((0..3).map(|layer| recipe.width_for_layer(rank, layer).unwrap()).sum::<usize>(), 2048);
        }
        for world in [2, 4] {
            assert_eq!(spark_recipe("mimof:fp8", 2048, world).unwrap().rule, UnitRule::Contiguous);
        }
    }

    #[test]
    fn compact_exl3_recipe_uses_own_maximum_and_preserves_even_exports() {
        use cuteafd_core::tp_units::UnitRule;
        let recipe = spark_recipe("glm:exl3-k45", 2048, 6).unwrap();
        assert_eq!(recipe.rule, UnitRule::Rotated);
        assert_eq!(recipe.widths, [384; 6]);
        assert_eq!(recipe.required_layouts,
            (0..6).map(|rank| format!("tp6-compact-rank{rank}")).collect::<Vec<_>>());
        for (package, intermediate, world) in [("glm:exl3-k45", 2048, 4),
            ("v41:mxfp4", 2304, 6), ("step5:fp8", 1536, 6)] {
            assert_eq!(spark_recipe(package, intermediate, world).unwrap().rule, UnitRule::Contiguous);
        }
        assert_eq!(spark_recipe("glm:exl3-k34", 2048, 6).unwrap().rule, UnitRule::Rotated);
    }

    #[test]
    fn checkpoint_recipes_follow_headers_without_installed_packages() {
        use super::super::{testing as fixture, Checkpoint};
        let mut cases = vec![
            (fixture::mimo_flash_config(), fixture::mimo_flash_tensors(), "mimo:fp8", 2048, 4),
            (fixture::mimo_flash_mopd_config(), fixture::mimo_flash_mopd_tensors(), "mimof:fp8", 2048, 4),
            (fixture::mimo_pro_config(), fixture::mimo_pro_tensors(), "mimop:fp8", 2048, 6),
            (fixture::glm5_config(), fixture::glm5_tensors(|name, n, k| fixture::fp8(name, n, k, None)), "glm:fp8", 2048, 4),
            (fixture::glm5_config(), fixture::glm5_tensors(|name, n, k| fixture::exl3(name, n, k, 4)), "glm:exl3-k45", 2048, 4),
        ];
        let config = fixture::glm5_flash_config(2);
        cases.push((config.clone(), fixture::glm5_flash_tensors(&config), "glmf:fp8", 2048, 4));
        let prefix = "model.language_model.layers.0.mlp.experts.0";
        let tensors = [("gate_proj", 640, 2560), ("up_proj", 640, 2560), ("down_proj", 2560, 640)]
            .into_iter().flat_map(|(projection, n, k)| fixture::exl3(&format!("{prefix}.{projection}"), n, k, 4)).collect();
        cases.push((fixture::qwen4_config(1), tensors, "qwen4:exl3-k45", 640, 1));
        for (config, tensors, package, intermediate, world) in cases {
            let root = tempfile::tempdir().unwrap();
            fixture::write_snapshot(root.path(), &config, &tensors, None);
            let checkpoint = Checkpoint::open(root.path()).unwrap();
            assert_eq!(checkpoint_spark_recipe(&checkpoint, world).unwrap(),
                spark_recipe(package, intermediate, world).unwrap(), "{package}");
        }
    }

    #[test]
    fn checkpoint_recipes_refuse_missing_or_malformed_headers_and_unqualified_worlds() {
        use super::super::{testing as fixture, Checkpoint};
        let root = tempfile::tempdir().unwrap();
        fixture::write_snapshot(root.path(), &fixture::mimo_flash_config(), &fixture::mimo_flash_tensors(), None);
        let mut checkpoint = Checkpoint::open(root.path()).unwrap();
        assert!(format!("{:#}", checkpoint_spark_recipe(&checkpoint, 9).unwrap_err()).contains("geometry-invalid"));
        let index = checkpoint.tensors.iter().position(|tensor| tensor.meta.name.contains(".experts.")
            && tensor.meta.name.ends_with("weight_scale_inv")).unwrap();
        let removed = checkpoint.tensors.remove(index);
        let error = format!("{:#}", checkpoint_spark_recipe(&checkpoint, 4).unwrap_err());
        assert!(error.contains("missing routed expert tensor header") && error.contains(&removed.meta.name), "{error}");
        checkpoint.tensors.insert(index, removed);
        checkpoint.tensors[index].meta.shape = vec![1, 1];
        assert!(checkpoint_spark_recipe(&checkpoint, 4).is_err());
        checkpoint.tensors.clear();
        assert!(checkpoint_spark_recipe(&checkpoint, 4).is_err());
    }

    #[test]
    fn checkpoint_recipes_refuse_conflicting_storage_layouts() {
        use super::super::{testing as fixture, Checkpoint};
        let root = tempfile::tempdir().unwrap();
        let tensors = fixture::glm5_tensors(|name, n, k| {
            if name.ends_with("down_proj") { fixture::exl3(name, n, k, 4) }
            else { fixture::fp8(name, n, k, None) }
        });
        fixture::write_snapshot(root.path(), &fixture::glm5_config(), &tensors, None);
        let error = format!("{:#}", checkpoint_spark_recipe(&Checkpoint::open(root.path()).unwrap(), 4).unwrap_err());
        assert!(error.contains("conflicting Spark recipes"), "{error}");
    }

    #[test]
    fn recipes_preserve_exact_tp6_and_legacy_nvfp4() {
        use crate::formats::fp8_experts::Slicing;
        let step5 = spark_recipe("step5:fp8", 1536, 4).unwrap();
        assert_eq!(step5.slicing, Slicing::Blocks(128));
        assert_eq!(step5.widths, [384; 4]);
        assert_eq!(step5.required_layouts, ["tp4"]);
        let step5_tp6 = spark_recipe("step5:fp8", 1536, 6).unwrap();
        assert_eq!(step5_tp6.slicing, Slicing::Blocks(128));
        assert_eq!(step5_tp6.widths, [256; 6]);
        assert_eq!(step5_tp6.widths.iter().sum::<usize>(), 1536);
        assert_eq!(step5_tp6.required_layouts, ["tp6"]);
        assert_eq!(spark_worlds("step5:fp8", 1536), [4, 5, 6]);
        for world in 0..=8 {
            if !matches!(world, 4 | 5 | 6) { assert!(spark_recipe("step5:fp8", 1536, world).is_err()); }
            if matches!(world, 4 | 5 | 6) {
                assert_eq!(spark_recipe("step5:bf16", 1536, world).unwrap(),
                    spark_recipe("step5:fp8", 1536, world).unwrap());
            } else { assert!(spark_recipe("step5:bf16", 1536, world).is_err()); }
            if matches!(world, 4 | 6) {
                let nvfp4 = spark_recipe("step5:nvfp4", 1536, world).unwrap();
                assert_eq!(nvfp4.slicing, Slicing::Blocks(16));
                assert!(nvfp4.same_ownership(&spark_recipe("step5:fp8", 1536, world).unwrap()));
            } else { assert!(spark_recipe("step5:nvfp4", 1536, world).is_err()); }
            for format in ["nvfp4a4", "mxfp4"] {
                assert!(spark_recipe(&format!("step5:{format}"), 1536, world).is_err());
            }
            // EXL3 K4/K6 follows the shared width-class ranges (step5:exl3):
            // exact H128 slices where twelve blocks split evenly, compact
            // trellis widths elsewhere; one rank cannot hold every expert.
            let exl3 = spark_recipe("step5:exl3-k46", 1536, world);
            if (2..=6).contains(&world) {
                let recipe = exl3.unwrap();
                if 12 % world == 0 { assert_eq!(recipe.widths, vec![1536 / world; world]); }
                else { assert_eq!(recipe.widths, vec![12usize.div_ceil(world) * 128; world]); }
            } else { assert!(exl3.is_err()); }
            assert!(spark_recipe("step5:exl3-k47", 1536, world).is_err());
        }
        assert_eq!(spark_worlds("step5:exl3-k46", 1536), cuteafd_core::tp_units::spark_package_worlds("step5", "exl3"));
        assert_eq!(spark_worlds("step5:exl3-k46", 1536), [2, 3, 4, 5, 6]);
        for package in ["mimo:fp8", "glm:nvfp4", "glmf:exl3-k34", "qwen4:nvfp4"] {
            assert!(spark_recipe(package, 2048, 6).is_ok());
        }
        assert!(spark_worlds("v41:exl3-k23", 2304).is_empty());
        assert!(spark_worlds("dsv4f:exl3-k23", 2048).is_empty());
        assert_eq!(spark_worlds("v41:mxfp4", 2304), [2, 3, 4, 6, 8]);
        assert_eq!(spark_worlds("v41:nvfp4", 2304), [2, 3, 4, 6]);
        for package in ["qwen4:nvfp4", "qwen4:exl3-k45", "qwen4:fp8"] {
            for world in [7, 8] {
                let refusal = spark_recipe(package, 640, world).unwrap_err();
                if package == "qwen4:nvfp4" {
                    assert!(matches!(refusal, SparkRecipeRefusal::Deprecated { .. }));
                } else {
                    assert!(matches!(refusal, SparkRecipeRefusal::GeometryInvalid { .. }));
                }
            }
            match package {
                "qwen4:fp8" => assert!(spark_recipe(package, 640, 5).is_ok()),
                "qwen4:nvfp4" => assert!(matches!(spark_recipe(package, 640, 5), Err(SparkRecipeRefusal::Deprecated { .. }))),
                _ => assert!(matches!(spark_recipe(package, 640, 5), Err(SparkRecipeRefusal::MissingPackage { .. }))),
            }
        }
    }

    /// Frozen NVFP4 admits only the worlds it shipped on, for every family,
    /// with the named deprecation reason; Step NVFP4 shipped only 1 RTX + 4,
    /// although its Spark package also covers TP6 (the BF16 tails' layout).
    #[test]
    fn frozen_formats_refuse_every_world_they_did_not_ship_on() {
        use cuteafd_core::tp_units::frozen_admitted_worlds;
        for (family, admitted) in [("step5", vec![4]), ("glm", vec![2, 3, 4, 6]), ("glmf", vec![2, 3, 4, 5, 6]),
            ("qwen4", vec![1, 2, 3, 4, 6]), ("v41", vec![2, 3, 4, 6])] {
            assert_eq!(frozen_admitted_worlds(family, "nvfp4"), Some(admitted.clone()), "{family}");
            assert_eq!(frozen_admitted_worlds(family, "nvfp4a4"), Some(admitted.clone()), "{family}");
            for world in 1..=8 {
                let refusal = frozen_refusal(&format!("{family}:nvfp4"), world);
                assert_eq!(refusal.is_none(), admitted.contains(&world), "{family} TP{world}");
                if let Some(refusal) = refusal {
                    assert!(refusal.to_string().contains("NVFP4 is deprecated"));
                }
            }
        }
        assert!(spark_recipe("step5:nvfp4", 1536, 6).is_ok(), "the package still covers TP6");
        assert!(frozen_refusal("step5:nvfp4", 6).is_some());
        assert!(frozen_refusal("step5:bf16 + step5:nvfp4", 6).is_some());
        for package in ["step5:fp8", "step5:bf16", "glm:exl3-k45", "v41:mxfp4"] {
            assert!(frozen_refusal(package, 6).is_none(), "{package}");
            let (family, format) = package.split_once(':').unwrap();
            assert!(frozen_admitted_worlds(family, format).is_none());
        }
    }

    #[test]
    fn mixed_native_units_require_identical_exact_source_and_storage_ranges() {
        use crate::formats::fp8_experts::Slicing;
        use cuteafd_core::tp_units::UnitRule;
        let nvfp4 = spark_recipe("step5:nvfp4", 1536, 6).unwrap();
        let bf16 = spark_recipe("step5:bf16", 1536, 6).unwrap();
        assert!(nvfp4.same_ownership(&bf16));
        assert!(bf16.same_ownership(&nvfp4));
        let mut incompatible = bf16.clone();
        incompatible.slicing = Slicing::Padded;
        assert!(!nvfp4.same_ownership(&incompatible));
        incompatible = bf16.clone();
        incompatible.widths[0] += 128;
        assert!(!nvfp4.same_ownership(&incompatible));
        incompatible = bf16.clone();
        incompatible.rule = UnitRule::LayerRotation;
        assert!(!nvfp4.same_ownership(&incompatible));
        incompatible = bf16;
        incompatible.required_layouts = vec!["tp6-w256".into()];
        assert!(!nvfp4.same_ownership(&incompatible));
        let qwen = spark_recipe("qwen4:nvfp4", 640, 2).unwrap();
        assert_eq!(qwen.slicing, Slicing::Padded);
        assert_eq!(qwen.widths, [384, 384]);
    }

    #[test]
    fn installed_packages_cannot_hide_a_padded_rank_or_incomplete_tree() {
        use crate::formats::fp8_experts::Slicing;
        let root = tempfile::tempdir().unwrap();
        let recipe = spark_recipe("mimop:fp8", 2048, 6).unwrap();
        let put = |layout: &str, library: &str| {
            let directory = root.path().join(layout);
            std::fs::create_dir_all(&directory).unwrap();
            std::fs::write(directory.join(library), b"fixture").unwrap();
        };
        put("tp6", "libcuteafd_fp8moe.so");
        put("tp6-w352", "libcuteafd_fp8moe.so");
        assert!(matches!(recipe.validate_installed(root.path(), 0, Slicing::Blocks(32), 352),
            Err(InstalledSparkRecipeRefusal::MissingArtifact(_))));
        put("tp6-w320", "libcuteafd_fp8moe.so");
        assert!(recipe.validate_installed(root.path(), 0, Slicing::Blocks(32), 352).is_ok());
        assert!(recipe.validate_installed(root.path(), 4, Slicing::Blocks(32), 320).is_ok());
        assert!(matches!(recipe.validate_installed(root.path(), 4, Slicing::Blocks(32), 352),
            Err(InstalledSparkRecipeRefusal::WidthMismatch { .. })));
        assert!(matches!(recipe.validate_installed(root.path(), 6, Slicing::Blocks(32), 320),
            Err(InstalledSparkRecipeRefusal::InvalidRank { .. })));
        assert!(matches!(recipe.validate_installed(root.path(), 0, Slicing::Padded, 352),
            Err(InstalledSparkRecipeRefusal::PaddedExactMismatch)));
        let exl3 = spark_recipe("glm:exl3-k45", 2048, 6).unwrap();
        assert!(matches!(exl3.validate_installed(root.path(), 0, Slicing::Blocks(128), 384),
            Err(InstalledSparkRecipeRefusal::MissingArtifact(_))));
        for rank in 0..6 { put(&format!("tp6-compact-rank{rank}/m16"), "libcuteafd_exl3.so"); }
        assert!(exl3.validate_installed(root.path(), 5, Slicing::Blocks(128), 384).is_ok());
    }

    #[test]
    fn native_unit_and_compact_recipes_preserve_supported_worlds() {
        for (package, intermediate, world) in [
            ("glmf:nvfp4", 2048, 5), ("glmf:nvfp4a4", 2048, 5),
            ("qwen4:nvfp4", 640, 6), ("v41:mxfp4", 2304, 4),
        ] {
            assert!(spark_recipe(package, intermediate, world).is_ok());
        }
    }

    #[test]
    fn inventory_cells_use_format_native_units_for_both_rtx_counts() {
        use crate::formats::fp8_experts::Slicing;
        use cuteafd_core::tp_units::{RankUnits, UnitRule};
        let cases: &[(&str, usize, &[usize])] = &[
            ("v41:mxfp4", 2304, &[2, 3, 4, 6, 8]),
            ("v41:nvfp4", 2304, &[2, 3, 4, 6]),
            ("dsv4f:mxfp4", 2048, &[2, 4, 8]),
            ("dsv4p:mxfp4", 3072, &[2, 3, 4, 6, 8]),
            ("dsv4p:exl3-k23", 3072, &[2, 3, 4, 5, 6, 7, 8]),
            ("glm:fp8", 2048, &[2, 3, 4, 5, 6, 7, 8]),
            ("glm:nvfp4", 2048, &[2, 3, 4, 6]),
            ("glm:exl3-k45", 2048, &[2, 3, 4, 5, 6, 7, 8]),
            ("glmf:fp8", 2048, &[2, 3, 4, 5, 6, 7, 8]),
            ("glmf:nvfp4", 2048, &[2, 3, 4, 5, 6]),
            ("glmf:exl3-k34", 2048, &[2, 3, 4, 5, 6, 7, 8]),
            ("mimo:fp8", 2048, &[2, 3, 4, 5, 6, 7, 8]),
            ("mimof:fp8", 2048, &[2, 3, 4, 5, 6, 7, 8]),
            ("mimop:fp8", 2048, &[2, 6, 7, 8]),
            ("qwen4:exl3-k45", 640, &[1, 2, 3, 4]),
            ("qwen4:nvfp4", 640, &[1, 2, 3, 4, 6]),
            ("qwen4:fp8", 640, &[1, 2, 3, 4, 5]),
            ("step5:fp8", 1536, &[4, 5, 6]),
            ("step5:bf16", 1536, &[4, 5, 6]),
            ("step5:nvfp4", 1536, &[4, 6]),
            ("kimik3:exl3-k23", 3072, &[6, 7, 8]),
        ];
        // Spark ownership is independent of the number of coordinator GPUs.
        for rtx in [1, 2] {
            for &(package, intermediate, worlds) in cases {
                for world in 1..=8 {
                    let result = spark_recipe(package, intermediate, world);
                    let context = format!("{package} I{intermediate} TP{world} RTX{rtx}");
                    assert_eq!(result.is_ok(), worlds.contains(&world), "{context}: {result:?}");
                    let Ok(recipe) = result else { continue };
                    let unit = if package.contains("nvfp4") { 16 }
                        else if package.ends_with("mxfp4") || matches!(package, "mimof:fp8" | "mimop:fp8") { 32 }
                        else { 128 };
                    if (intermediate / unit) % world == 0 {
                        assert_eq!(recipe.rule, UnitRule::Contiguous, "{context}");
                    }
                    for layer in 0..world {
                        let maps = (0..world).map(|rank| RankUnits::new(recipe.rule, 8,
                            intermediate, unit, layer, world, rank).unwrap()).collect::<Vec<_>>();
                        assert_eq!(maps.iter().map(RankUnits::resident_units).sum::<usize>(),
                            8 * intermediate / unit, "{context}");
                        for (rank, map) in maps.iter().enumerate() {
                            let stored = recipe.width_for_layer(rank, layer).unwrap();
                            let live = map.width_classes().keys().copied().max().unwrap();
                            assert!(stored >= live, "{context}");
                            if recipe.slicing == Slicing::Padded { assert_eq!(stored % 128, 0, "{context}"); }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn native_mxfp4_admission_matches_equal_k32_reader_and_exporter() {
        for world in [3, 6] {
            assert!(matches!(spark_recipe("dsv4f:mxfp4", 2048, world),
                Err(SparkRecipeRefusal::NativeUnequal { .. })));
        }
        for world in [2, 4] {
            assert!(spark_recipe("dsv4f:mxfp4", 2048, world).is_ok());
        }
        for world in [2, 3, 4, 6] {
            assert!(spark_recipe("dsv4p:mxfp4", 3072, world).is_ok());
        }
        assert!(spark_recipe("mimof:fp8", 2048, 3).is_ok());
        assert!(spark_recipe("mimop:fp8", 2048, 6).is_ok());
    }

    #[test]
    fn recipe_worlds_keep_family_and_format_qualification_separate() {
        for intermediate in [640, 2048, 2304, 3072] {
            for package in ["mimo:fp8", "mimof:fp8", "mimop:fp8", "glm:nvfp4",
                "glmf:nvfp4a4", "glm:exl3-k34", "glmf:exl3-k34", "v41:mxfp4",
                "dsv4p:exl3-k23", "qwen4:exl3-k45", "qwen4:nvfp4"] {
                let worlds = spark_worlds(package, intermediate);
                if package.contains("nvfp4") {
                    assert!(!worlds.iter().any(|world| matches!(world, 7 | 8)));
                    if package != "glmf:nvfp4a4" { assert!(!worlds.contains(&5)); }
                }
                for world in worlds {
                    let recipe = spark_recipe(package, intermediate, world).unwrap();
                    assert_eq!(recipe.widths.len(), world);
                    assert!(recipe.widths.iter().all(|width| *width > 0 && width % 32 == 0));
                    if recipe.rule != cuteafd_core::tp_units::UnitRule::Rotated
                        && matches!(recipe.slicing, crate::formats::fp8_experts::Slicing::Blocks(_)) {
                        assert_eq!(recipe.widths.iter().sum::<usize>(), intermediate);
                    }
                }
            }
        }
        assert_eq!(spark_worlds("mimof:fp8", 2048), [2, 3, 4, 5, 6, 7, 8]);
        assert!(spark_recipe("mimof:fp8", 2048, 3).is_ok());
        assert_eq!(spark_worlds("mimop:fp8", 2048), [2, 6, 7, 8]);
        assert_eq!(spark_worlds("qwen4:exl3-k45", 640), [1, 2, 3, 4]);
        assert_eq!(spark_worlds("qwen4:nvfp4", 640), [1, 2, 3, 4, 6]);
        assert!(matches!(spark_recipe("unknown:fp8", 2048, 4),
            Err(SparkRecipeRefusal::UnknownPackage(_))));
    }

    #[test]
    fn package_layouts_follow_the_packagers() {
        // Only exact-even or explicitly qualified uneven kernels are admitted.
        assert_eq!(spark_worlds("mimo:fp8", 2048), [2, 3, 4, 5, 6, 7, 8]);
        assert_eq!(spark_worlds("qwen4:fp8", 640), [1, 2, 3, 4, 5]);
        assert_eq!(spark_worlds("glm:exl3-k34", 2048), [2, 3, 4, 5, 6, 7, 8]);
        assert_eq!(spark_worlds("qwen4:exl3-k45", 640), [1, 2, 3, 4]);
        assert_eq!(stored_slice(2048, 128, 6), Some(384));
        assert_eq!(stored_slice(2048, 32, 6), Some(384));
        assert_eq!(stored_slice(2304, 128, 4), Some(640));
        assert_eq!(stored_slice(640, 128, 6), None);
        assert_eq!(stored_slice(2048, 128, 0), None);
        assert_eq!(spark_worlds("qwen4:nvfp4", 640), [1, 2, 3, 4, 6]);
        assert_eq!(stored_slice(640, 16, 6), Some(128));
        assert_eq!(stored_slice(2048, 16, 6), Some(384));
    }
}
