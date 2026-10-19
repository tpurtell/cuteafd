//! Routed-expert package layouts this build's packagers produce, and the
//! slice arithmetic the expert staging uses. The planner's placement check
//! reads these, so it accepts exactly the layouts an image can carry.
//!
//! - FP8 (`python/tools/aot/package_fp8_moe_aot.py`, b12x fp8_moe): Spark
//!   layouts tp4, tp2 and tp6 where they split the intermediate into whole
//!   128-row blocks (TP6 of any intermediate of at least six blocks, unequal
//!   exact-width packages); a tp1 coordinator package.
//! - MXFP4 through the same programs (MiMo V2.6 Pro `mimop:fp8`): Spark tp6
//!   and tp2, a tp1 coordinator package.
//! - NVFP4 (ModelOpt) through the same programs (`glmf:nvfp4`, `qwen4:nvfp4`,
//!   `glm:nvfp4`): Spark tp4, tp2, tp3 and tp6 in whole 16-value blocks, a tp1
//!   coordinator package. GLM Flash also qualifies exact-H128 TP5.
//! - EXL3 (`python/tools/aot/package_exl3_aot.py`): Spark worlds 4, 2, 3, and
//!   6 when the intermediate has at least six 128-row blocks; Qwen also has TP1.
//! - DeepSeek native experts (expertd-native MXFP4 / EXL3): 2, 3, 4 and 6.
//!
//! The expert transport (RoCE verbs, TCP) runs one through six Spark ranks.
//! Package coverage below is independent of transport capacity.

/// Spark worlds the expert transport runs.
pub const TRANSPORT_WORLDS: [usize; 6] = [1, 2, 3, 4, 5, 6];

/// Default Spark layouts of an FP8 (E4M3, 128x128 scales) expert package.
pub fn fp8_spark_worlds(intermediate: usize) -> Vec<usize> {
    spark_worlds("mimo:fp8", intermediate)
}

/// Default Spark layouts of an MXFP4 package on the fp8_moe programs.
pub fn mxfp4_spark_worlds() -> Vec<usize> {
    spark_worlds("mimop:fp8", 2048)
}

/// Default Spark layouts of an NVFP4 package on the fp8_moe programs: the
/// transport worlds whose ranks each own a 16-value block.
pub fn nvfp4_spark_worlds(intermediate: usize) -> Vec<usize> {
    spark_worlds("glm:nvfp4", intermediate)
}

/// Spark worlds of an EXL3 package (`package_exl3_aot.py` profiles).
pub fn exl3_spark_worlds(intermediate: usize) -> Vec<usize> {
    spark_worlds("glm:exl3-k34", intermediate)
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

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum SparkRecipeRefusal {
    #[error("geometry-invalid: intermediate {intermediate} cannot give {world} ranks nonempty whole-{block} row slices")]
    GeometryInvalid { intermediate: usize, world: usize, block: usize },
    #[error("missing Spark package recipe: {package} TP{world}")]
    MissingPackage { package: String, world: usize },
    #[error("unknown Spark package recipe: {0}")]
    UnknownPackage(String),
}

/// Enumerate recipes from the format policy and this checkpoint's geometry.
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
        ensure!(selected.as_ref().is_none_or(|previous| previous == &recipe),
            "Spark admission tensor {stem}: routed formats require conflicting Spark recipes");
        selected = Some(recipe);
    }
    selected.context("Spark admission: missing routed expert storage recipe")
}

/// Registered recipes only: optional installation checks cannot widen this set.
/// Step-5 TP6 geometry is registered for qualification; its hardware gate is pending.
/// TP5 support is added per format only after its hardware gate passes.
pub fn spark_recipe(package: &str, intermediate: usize, world: usize) -> Result<SparkRecipe, SparkRecipeRefusal> {
    use crate::formats::fp8_experts::Slicing;
    let key = package.split_whitespace().next().unwrap_or(package);
    let (family, format) = key.split_once(':').ok_or_else(|| SparkRecipeRefusal::UnknownPackage(package.into()))?;
    let native = matches!(family, "v41" | "dsv4f" | "dsv4p");
    let exl3 = format.starts_with("exl3-k") && matches!(family, "v41" | "dsv4f" | "dsv4p" | "glm" | "glmf" | "qwen4" | "kimik3");
    let nvfp4 = matches!(format, "nvfp4" | "nvfp4a4") && matches!(family, "v41" | "glm" | "glmf" | "qwen4");
    let mxfp4 = (format == "mxfp4" && native) || (format == "fp8" && matches!(family, "mimof" | "mimop"));
    let fp8 = format == "fp8" && matches!(family, "mimo" | "glm" | "glmf" | "qwen4" | "step5");
    let bf16 = family == "step5" && format == "bf16";
    if !(exl3 || nvfp4 || mxfp4 || fp8 || bf16) {
        return Err(SparkRecipeRefusal::UnknownPackage(package.into()));
    }
    // V4.1 EXL3 TP6 remains unqualified pending a capped-local fidelity gate.
    // Kimi K3's only Spark layout is TP6 width 512 (#273: TP4 needs 160 GiB/rank).
    if (family == "v41" && exl3 && world == 6) || (family == "kimik3" && world != 6) {
        return Err(SparkRecipeRefusal::MissingPackage { package: key.into(), world });
    }
    let block = if nvfp4 && !native && !matches!(world, 1 | 5) { 16 } else { 128 };
    if !(1..=6).contains(&world) || intermediate == 0 || intermediate % block != 0 || intermediate / block < world {
        return Err(SparkRecipeRefusal::GeometryInvalid { intermediate, world, block });
    }
    let packaged = if family == "step5" { matches!(world, 4 | 6) }
        else if native { matches!(world, 2 | 3 | 4 | 6) }
        else if exl3 { matches!(world, 2 | 3 | 4 | 6) || (family == "qwen4" && world == 1) }
        else if nvfp4 { matches!(world, 2 | 3 | 4 | 6) || (family == "glmf" && world == 5) }
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

    fn checkpoint_spark_recipe(checkpoint: &super::super::Checkpoint, world: usize) -> anyhow::Result<SparkRecipe> {
        let family = super::super::family::detect(checkpoint).unwrap();
        let model = family.open(checkpoint)?;
        let layers = model.spec().layers.iter().enumerate()
            .filter_map(|(layer, spec)| (spec.ffn == super::super::spec::FfnKind::Moe).then_some(layer)).collect::<Vec<_>>();
        super::checkpoint_spark_recipe(checkpoint, world, &layers)
    }

    #[test]
    fn checkpoint_recipes_follow_headers_without_installed_packages() {
        use super::super::{testing as fixture, Checkpoint};
        let mut cases = vec![
            (fixture::mimo_flash_config(), fixture::mimo_flash_tensors(), "mimo:fp8", 2048, 6),
            (fixture::mimo_flash_mopd_config(), fixture::mimo_flash_mopd_tensors(), "mimof:fp8", 2048, 4),
            (fixture::mimo_pro_config(), fixture::mimo_pro_tensors(), "mimop:fp8", 2048, 6),
            (fixture::glm5_config(), fixture::glm5_tensors(|name, n, k| fixture::fp8(name, n, k, None)), "glm:fp8", 2048, 6),
            (fixture::glm5_config(), fixture::glm5_tensors(|name, n, k| fixture::exl3(name, n, k, 4)), "glm:exl3-k45", 2048, 4),
        ];
        let config = fixture::glm5_flash_config(2);
        cases.push((config.clone(), fixture::glm5_flash_tensors(&config), "glmf:fp8", 2048, 6));
        let prefix = "model.language_model.layers.0.mlp.experts.0";
        let tensors = [("gate_proj", 640, 2560), ("up_proj", 640, 2560), ("down_proj", 2560, 640)]
            .into_iter().flat_map(|(projection, n, k)| fixture::exl3(&format!("{prefix}.{projection}"), n, k, 4)).collect();
        cases.push((fixture::qwen4_config(1), tensors, "qwen4:exl3-k45", 640, 3));
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
        assert!(format!("{:#}", checkpoint_spark_recipe(&checkpoint, 3).unwrap_err()).contains("missing Spark package recipe"));
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
        assert_eq!(spark_worlds("step5:fp8", 1536), [4, 6]);
        for world in 0..=8 {
            if !matches!(world, 4 | 6) { assert!(spark_recipe("step5:fp8", 1536, world).is_err()); }
            if matches!(world, 4 | 6) {
                assert_eq!(spark_recipe("step5:bf16", 1536, world).unwrap(),
                    spark_recipe("step5:fp8", 1536, world).unwrap());
            } else { assert!(spark_recipe("step5:bf16", 1536, world).is_err()); }
            for format in ["nvfp4", "nvfp4a4", "mxfp4", "exl3-k34"] {
                assert!(spark_recipe(&format!("step5:{format}"), 1536, world).is_err());
            }
        }
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
        assert!(matches!(spark_recipe("v41:exl3-k23 (expertd-native)", 2304, 6),
            Err(SparkRecipeRefusal::MissingPackage { .. })));
        assert_eq!(spark_recipe("v41:exl3-k23 (expertd-native)", 2304, 3).unwrap().widths, [768; 3]);
        assert_eq!(spark_worlds("v41:exl3-k23", 2304), [2, 3, 4]);
        assert_eq!(spark_worlds("v41:mxfp4", 2304), [2, 3, 4, 6]);
        assert_eq!(spark_worlds("v41:nvfp4", 2304), [2, 3, 4, 6]);
        for package in ["qwen4:nvfp4", "qwen4:exl3-k45", "qwen4:fp8"] {
            for world in [7, 8] {
                assert!(matches!(spark_recipe(package, 640, world), Err(SparkRecipeRefusal::GeometryInvalid { .. })));
            }
            assert!(matches!(spark_recipe(package, 640, 5), Err(SparkRecipeRefusal::MissingPackage { .. })));
        }
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
        put("tp6-w384", "libcuteafd_fp8moe.so");
        assert!(matches!(recipe.validate_installed(root.path(), 0, Slicing::Blocks(128), 384),
            Err(InstalledSparkRecipeRefusal::MissingArtifact(_))));
        put("tp6-w256", "libcuteafd_fp8moe.so");
        assert!(recipe.validate_installed(root.path(), 0, Slicing::Blocks(128), 384).is_ok());
        assert!(recipe.validate_installed(root.path(), 4, Slicing::Blocks(128), 256).is_ok());
        assert!(matches!(recipe.validate_installed(root.path(), 4, Slicing::Blocks(128), 384),
            Err(InstalledSparkRecipeRefusal::WidthMismatch { .. })));
        assert!(matches!(recipe.validate_installed(root.path(), 6, Slicing::Blocks(128), 256),
            Err(InstalledSparkRecipeRefusal::InvalidRank { .. })));
        assert!(matches!(recipe.validate_installed(root.path(), 0, Slicing::Padded, 384),
            Err(InstalledSparkRecipeRefusal::PaddedExactMismatch)));
        let legacy = spark_recipe("qwen4:nvfp4", 640, 6).unwrap();
        assert!(legacy.validate_installed(root.path(), 5, Slicing::Padded, 128).is_ok());
        let exl3 = spark_recipe("glmf:exl3-k34", 2048, 6).unwrap();
        assert!(matches!(exl3.validate_installed(root.path(), 0, Slicing::Blocks(128), 384),
            Err(InstalledSparkRecipeRefusal::MissingArtifact(_))));
        for rank in 0..6 { put(&format!("tp6-rank{rank}/m16"), "libcuteafd_exl3.so"); }
        assert!(exl3.validate_installed(root.path(), 5, Slicing::Blocks(128), 256).is_ok());
    }

    #[test]
    fn glm_flash_nvfp4_tp5_requires_exact_width_packages() {
        use crate::formats::fp8_experts::Slicing;
        for package in ["glmf:nvfp4", "glmf:nvfp4a4"] {
            let recipe = spark_recipe(package, 2048, 5).unwrap();
            assert_eq!(spark_worlds(package, 2048), [2, 3, 4, 5, 6]);
            assert_eq!(recipe.slicing, Slicing::Blocks(128));
            assert_eq!(recipe.widths, [512, 384, 384, 384, 384]);
            assert_eq!(recipe.required_layouts, ["tp5-w512", "tp5-w384"]);
            let root = tempfile::tempdir().unwrap();
            for layout in ["tp5", "tp5-w512"] {
                let directory = root.path().join(layout);
                std::fs::create_dir_all(&directory).unwrap();
                std::fs::write(directory.join("libcuteafd_fp8moe.so"), b"fixture").unwrap();
            }
            assert!(matches!(recipe.validate_installed(root.path(), 0, Slicing::Blocks(128), 512),
                Err(InstalledSparkRecipeRefusal::MissingArtifact(_))));
            let directory = root.path().join("tp5-w384");
            std::fs::create_dir_all(&directory).unwrap();
            std::fs::write(directory.join("libcuteafd_fp8moe.so"), b"fixture").unwrap();
            assert!(recipe.validate_installed(root.path(), 0, Slicing::Blocks(128), 512).is_ok());
            assert!(recipe.validate_installed(root.path(), 4, Slicing::Blocks(128), 384).is_ok());
            assert!(matches!(recipe.validate_installed(root.path(), 0, Slicing::Padded, 512),
                Err(InstalledSparkRecipeRefusal::PaddedExactMismatch)));
        }
        for package in ["glm:nvfp4", "qwen4:nvfp4", "qwen4:nvfp4a4", "glmf:exl3-k34", "glmf:fp8"] {
            assert!(matches!(spark_recipe(package, 2048, 5), Err(SparkRecipeRefusal::MissingPackage { .. })));
        }
    }

    #[test]
    fn recipe_worlds_keep_family_and_format_qualification_separate() {
        for intermediate in [640, 2048, 2304, 3072] {
            for package in ["mimo:fp8", "mimof:fp8", "mimop:fp8", "glm:nvfp4",
                "glmf:nvfp4a4", "glm:exl3-k34", "glmf:exl3-k34", "v41:mxfp4",
                "dsv4f:exl3-k23", "dsv4p:exl3-k23", "qwen4:exl3-k45", "qwen4:nvfp4"] {
                let worlds = spark_worlds(package, intermediate);
                assert!(!worlds.iter().any(|world| matches!(world, 7 | 8)));
                assert_eq!(worlds.contains(&5), package == "glmf:nvfp4a4");
                for world in worlds {
                    let recipe = spark_recipe(package, intermediate, world).unwrap();
                    assert_eq!(recipe.widths.len(), world);
                    assert!(recipe.widths.iter().all(|width| *width > 0 && width % 128 == 0));
                    if matches!(recipe.slicing, crate::formats::fp8_experts::Slicing::Blocks(_)) {
                        assert_eq!(recipe.widths.iter().sum::<usize>(), intermediate);
                    }
                }
            }
        }
        assert_eq!(spark_worlds("mimof:fp8", 2048), [2, 4]);
        assert_eq!(spark_worlds("mimop:fp8", 2048), [2, 6]);
        assert_eq!(spark_worlds("qwen4:exl3-k45", 640), [1, 2, 3, 4]);
        assert_eq!(spark_worlds("qwen4:nvfp4", 640), [2, 3, 4, 6]);
        assert!(matches!(spark_recipe("unknown:fp8", 2048, 4),
            Err(SparkRecipeRefusal::UnknownPackage(_))));
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
