//! Small, metadata-only multi-core checkpoint discovery.
use super::{view::metadata_bytes, MAX_JSON_BYTES};
use anyhow::{ensure, Context, Result};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{collections::{BTreeMap, BTreeSet}, fmt, path::Path, str::FromStr};

pub const MC_MANIFEST: &str = "cuteafd-mc.json";

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum CoreView {
    #[default]
    Original,
    K8Compact,
    K6Compact,
    Fp8Compact,
}
impl CoreView {
    pub fn name(self) -> &'static str {
        match self {
            Self::Original => "original", Self::K8Compact => "k8-compact",
            Self::K6Compact => "k6-compact", Self::Fp8Compact => "fp8-compact",
        }
    }
    pub fn compact(self) -> bool { matches!(self, Self::K8Compact | Self::K6Compact) }
    fn files(self) -> [&'static str; 3] {
        match self {
            Self::Original | Self::Fp8Compact => ["model.safetensors.index.json", "config.json", "quantize_config.json"],
            Self::K8Compact => ["model.k8.safetensors.index.json", "config.k8.json", "quantize_config.k8.json"],
            Self::K6Compact => ["model.k6.safetensors.index.json", "config.k6.json", "quantize_config.k6.json"],
        }
    }
    fn bits(self) -> u8 {
        match self { Self::Original => 16, Self::K8Compact => 8, Self::K6Compact => 6, Self::Fp8Compact => 8 }
    }
}
impl fmt::Display for CoreView {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result { f.write_str(self.name()) }
}
impl FromStr for CoreView {
    type Err = anyhow::Error;
    fn from_str(value: &str) -> Result<Self> {
        match value {
            "original" => Ok(Self::Original), "k8-compact" => Ok(Self::K8Compact),
            "k6-compact" => Ok(Self::K6Compact), "fp8-compact" => Ok(Self::Fp8Compact),
            _ => anyhow::bail!("unknown CORE_VIEW={value:?}; expected original, k8-compact, k6-compact or fp8-compact"),
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct Source { repo: String, revision: String }

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Variant {
    pub name: CoreView,
    pub index: String,
    pub config: String,
    pub quantize: String,
    core_bits: u8,
    pub(super) core_tensors: Vec<String>,
    metadata_sha256: BTreeMap<String, String>,
    fingerprint: String,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Manifest {
    schema: String,
    source: Source,
    original_core_dtype: String,
    shared_inventory_sha256: String,
    variants: Vec<Variant>,
}

fn is_digest(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}
fn safe_text(value: &str) -> bool { !value.is_empty() && !value.contains(['\0', '\n', '\r']) }
fn shard_name(value: &str) -> bool {
    safe_text(value) && value.ends_with(".safetensors") && !value.starts_with('.') && !value.contains(['/', '\\'])
}

impl Variant {
    fn validate(&self, shared: &str) -> Result<()> {
        ensure!(self.name != CoreView::Fp8Compact, "fp8-compact is derived at load, not a stored MC variant");
        let expected = self.name.files();
        ensure!([self.index.as_str(), self.config.as_str(), self.quantize.as_str()] == expected,
            "MC {} references noncanonical metadata filenames", self.name);
        ensure!(self.core_bits == self.name.bits(), "MC {} core_bits mismatch", self.name);
        ensure!(!self.core_tensors.is_empty() && self.core_tensors.iter().all(|s| safe_text(s))
            && self.core_tensors.windows(2).all(|pair| pair[0] < pair[1]),
            "MC {} core_tensors must be nonempty, sorted and unique", self.name);
        ensure!(self.metadata_sha256.len() == 3 && expected.iter().all(|name|
            self.metadata_sha256.get(*name).is_some_and(|digest| is_digest(digest))),
            "MC {} requires exactly three metadata SHA256 digests", self.name);
        ensure!(is_digest(&self.fingerprint) && self.fingerprint == self.expected_fingerprint(shared),
            "MC {} fingerprint mismatch", self.name);
        Ok(())
    }
    fn expected_fingerprint(&self, shared: &str) -> String {
        let mut hash = Sha256::new();
        hash.update(format!("cuteafd-mc-view-v1\0{}\0{shared}\0", self.name));
        for (name, digest) in &self.metadata_sha256 { hash.update(format!("{name}\0{digest}\n")); }
        format!("{:x}", hash.finalize())
    }
    pub(super) fn check_metadata(&self, name: &str, bytes: &[u8]) -> Result<()> {
        ensure!(self.metadata_sha256.get(name).is_some_and(|digest| *digest == format!("{:x}", Sha256::digest(bytes))),
            "MC {} metadata digest mismatch for {name}", self.name);
        Ok(())
    }
    pub(super) fn check_index(&self, weights: &BTreeMap<String, String>, shared: &str) -> Result<()> {
        let core: BTreeSet<_> = self.core_tensors.iter().map(String::as_str).collect();
        ensure!(core.iter().all(|name| weights.contains_key(*name)), "MC {} core tensor missing from selected index", self.name);
        let mut hash = Sha256::new();
        for (name, shard) in weights {
            ensure!(safe_text(name) && shard_name(shard), "MC {} invalid tensor name or shard {name:?}: {shard:?}", self.name);
            if !core.contains(name.as_str()) { hash.update(format!("{name}\0{shard}\n")); }
        }
        ensure!(format!("{:x}", hash.finalize()) == shared, "MC {} shared inventory digest mismatch", self.name);
        // The original exposes no trellis core buffers, even when a shard stores
        // other variants. Routed expert trellises belong to the shared inventory.
        if self.name == CoreView::Original {
            ensure!(core.iter().all(|name| !crate::formats::exl3_storage::EXL3_SUFFIXES.iter()
                .any(|suffix| name.ends_with(&format!(".{suffix}")))),
                "MC original core contains a quantized tensor");
        }
        Ok(())
    }
}

impl Manifest {
    pub(super) fn read(snapshot: &Path) -> Result<Option<(Self, Vec<u8>)>> {
        let path = snapshot.join(MC_MANIFEST);
        let bytes = match metadata_bytes(&path) {
            Ok(bytes) => bytes,
            Err(error) if error.downcast_ref::<std::io::Error>().is_some_and(|e| e.kind() == std::io::ErrorKind::NotFound) => return Ok(None),
            Err(error) => return Err(error),
        };
        ensure!(bytes.len() as u64 <= MAX_JSON_BYTES, "MC manifest exceeds metadata size limit");
        let manifest: Self = serde_json::from_slice(&bytes).context("decoding cuteafd-mc.json")?;
        ensure!(manifest.schema == "cuteafd-mc-v1", "unsupported MC schema {}", manifest.schema);
        ensure!(safe_text(&manifest.source.repo) && safe_text(&manifest.source.revision), "MC requires a pinned source repo and revision");
        ensure!(matches!(manifest.original_core_dtype.as_str(), "BF16" | "F32"), "FP8-origin checkpoints must not be MC or contain quantized core tensors");
        ensure!(is_digest(&manifest.shared_inventory_sha256), "invalid MC shared inventory SHA256");
        ensure!((2..=3).contains(&manifest.variants.len()), "MC requires original and one or two compact views");
        let mut names = BTreeSet::new();
        for variant in &manifest.variants {
            variant.validate(&manifest.shared_inventory_sha256)?;
            ensure!(names.insert(variant.name.name()), "duplicate MC variant {}", variant.name);
        }
        ensure!(names.contains("original"), "MC requires an original view");
        Ok(Some((manifest, bytes)))
    }
    pub(super) fn select(&self, view: CoreView) -> Result<&Variant> {
        let selected = if view == CoreView::Fp8Compact { CoreView::Original } else { view };
        self.variants.iter().find(|variant| variant.name == selected)
            .with_context(|| format!("CORE_VIEW={view} is absent from cuteafd-mc.json"))
    }
    pub(super) fn check_index(&self, variant: &Variant, weights: &BTreeMap<String, String>) -> Result<()> {
        variant.check_index(weights, &self.shared_inventory_sha256)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::plan::checkpoint::CheckpointView;

    fn fixture() -> std::path::PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../scripts/tests/fixtures/mc-v1")
    }
    fn copy_fixture() -> tempfile::TempDir {
        let dir = tempfile::tempdir().unwrap();
        for entry in std::fs::read_dir(fixture()).unwrap() {
            let entry = entry.unwrap();
            std::fs::copy(entry.path(), dir.path().join(entry.file_name())).unwrap();
        }
        dir
    }
    #[test]
    fn cross_language_golden_vectors_and_metadata_only_discovery() -> Result<()> {
        let root = fixture();
        let (manifest, _) = Manifest::read(&root)?.unwrap();
        assert_eq!(manifest.shared_inventory_sha256,
            "1fb4551df22a63f50f22340099e2b495813979999164ec641931f466fe674655");
        for (view, expected) in [
            (CoreView::Original, "4d585770a7aee778ae778d5d7e11dd013e69efd2740654c4a852f6f232d9ed16"),
            (CoreView::K6Compact, "2c998320fea2f7f60a2c8e605ab68e85bd63cf96188b3724e4fccd3a96c2f10e"),
            (CoreView::K8Compact, "86470216c28b0b319efd25d2eba3dc7ac12c2d0a955ec0747f3d2d01558da76b"),
        ] {
            assert_eq!(manifest.select(view)?.fingerprint, expected);
            let selected = CheckpointView::select(&root, view)?;
            assert_eq!(selected.mode(), view);
            assert_eq!(selected.index_name(), view.files()[0]);
            assert!(selected.classification_headers.is_empty());
        }
        Ok(())
    }
    #[test]
    fn selected_view_needs_no_other_metadata_or_shards() -> Result<()> {
        let dir = copy_fixture();
        for view in [CoreView::Original, CoreView::K6Compact] {
            for file in view.files() { std::fs::remove_file(dir.path().join(file))?; }
        }
        CheckpointView::select(dir.path(), CoreView::K8Compact)?;
        assert!(CheckpointView::select(dir.path(), CoreView::Original).is_err());
        Ok(())
    }
    #[test]
    fn missing_view_plain_checkpoint_and_fp8_origin_refusal() -> Result<()> {
        let dir = copy_fixture();
        let path = dir.path().join(MC_MANIFEST);
        let mut manifest: serde_json::Value = serde_json::from_slice(&std::fs::read(&path)?)?;
        manifest["variants"].as_array_mut().unwrap().retain(|v| v["name"] != "k8-compact");
        std::fs::write(&path, manifest.to_string())?;
        assert!(CheckpointView::select(dir.path(), CoreView::K8Compact).unwrap_err().to_string().contains("absent"));
        manifest["original_core_dtype"] = "F8_E4M3".into();
        std::fs::write(&path, manifest.to_string())?;
        assert!(CheckpointView::select(dir.path(), CoreView::Original).unwrap_err().to_string().contains("FP8-origin"));
        std::fs::remove_file(path)?;
        CheckpointView::select(dir.path(), CoreView::Original)?;
        assert!(CheckpointView::select(dir.path(), CoreView::K6Compact).is_err());
        Ok(())
    }
    #[test]
    fn fp8_selection_uses_exact_original_ids_without_opening_alternate_metadata() -> Result<()> {
        let dir = copy_fixture();
        let original = CheckpointView::select(dir.path(), CoreView::Original)?;
        let expected = Manifest::read(dir.path())?.unwrap().0.select(CoreView::Original)?.core_tensors.clone();
        for mode in [CoreView::K6Compact, CoreView::K8Compact] {
            for file in mode.files() { std::fs::remove_file(dir.path().join(file))?; }
        }
        let selected = CheckpointView::select(dir.path(), CoreView::Fp8Compact)?;
        assert_eq!(selected.fp8_tensors(), expected);
        assert_eq!(selected.index_path(), original.index_path());
        assert_ne!(selected.fingerprint(), original.fingerprint());
        for name in selected.weight_map().unwrap().keys() {
            assert_eq!(selected.converts_to_fp8(name), expected.contains(name), "{name}");
        }
        std::fs::remove_file(dir.path().join(MC_MANIFEST))?;
        assert!(CheckpointView::select(dir.path(), CoreView::Fp8Compact).unwrap_err().to_string()
            .contains("requires cuteafd-mc.json"));
        Ok(())
    }
    #[test]
    fn selected_bytes_are_authenticated_and_immutable() -> Result<()> {
        let dir = copy_fixture();
        let selected = CheckpointView::select(dir.path(), CoreView::K8Compact)?;
        std::fs::write(dir.path().join("config.k8.json"), "{}")?;
        assert!(CheckpointView::select(dir.path(), CoreView::K8Compact).unwrap_err().to_string().contains("digest"));
        assert_eq!(selected.config()?["model_type"], "kimi_k3");
        assert!(selected.check_metadata_bytes("config.k8.json", b"{}").is_err());
        Ok(())
    }

    #[test]
    fn only_mc_checkpoints_declare_a_core_view() -> Result<()> {
        let root = fixture();
        for view in [CoreView::Original, CoreView::K6Compact, CoreView::K8Compact, CoreView::Fp8Compact] {
            let selected = CheckpointView::select(&root, view)?;
            assert_eq!(selected.declared_view(), Some(view));
            assert_eq!(selected.fp8_recipe().is_some(), view == CoreView::Fp8Compact);
        }
        // A plain checkpoint (no cuteafd-mc.json) has only its own core: no view, no recipe.
        let plain = tempfile::tempdir()?;
        crate::plan::testing::write_snapshot(plain.path(), &crate::plan::testing::kimi_k3_config(),
            &crate::plan::testing::kimi_k3_tensors(false), None);
        let selected = CheckpointView::select(plain.path(), CoreView::Original)?;
        assert!(!selected.is_mc());
        assert_eq!((selected.declared_view(), selected.fp8_recipe()), (None, None));
        assert!(CheckpointView::select(plain.path(), CoreView::K6Compact).is_err());
        Ok(())
    }

}

#[cfg(test)]
mod family_tests {
    use super::*;
    use crate::plan::{testing, Checkpoint, checkpoint::CheckpointView};

    #[test]
    fn mc_native_classification_follows_family_capability_without_role_payloads() -> Result<()> {
        use crate::plan::{family, testing::t};
        for (architecture, model_type, eager) in [
            ("GlmMoeDsaForCausalLM", "glm_moe_dsa", true),
            ("Glm5NextForConditionalGeneration", "glm5_next", true),
            ("MuseSparkForConditionalGeneration", "muse_spark", true),
            ("Step5ForConditionalGeneration", "step5", false),
        ] {
            let dir = tempfile::tempdir()?;
            let config = serde_json::json!({"model_type":model_type,"architectures":[architecture]});
            let core = "model.language_model.layers.0.self_attn.k_proj.weight";
            let expert = "model.layers.0.mlp.experts.0.gate_proj.weight";
            let shapes = BTreeMap::from([(core.to_owned(), vec![128, 128])]);
            testing::write_mc_snapshot(dir.path(), &config,
                &[t(core, "BF16", &[128,128]), t(expert, "BF16", &[128,128])], &shapes);
            for mode in [CoreView::Original, CoreView::K6Compact, CoreView::K8Compact] {
                let selected = CheckpointView::select(dir.path(), mode)?;
                let inventory = Checkpoint::inventory_view(&selected)?;
                assert_eq!(family::detect(&inventory).unwrap().eager_native_core_headers(), eager);
                assert!(inventory.tensors.is_empty(), "{model_type}: classification is not payload");
                let worker = Checkpoint::open_view_for_role(&selected, "experts", |name| name == expert)?;
                assert_eq!(worker.tensors.len(), 1);
                assert_eq!(worker.tensors[0].meta.name, expert);
                assert_eq!(worker.header(core).is_some(), eager && mode == CoreView::Original,
                    "{model_type}/{mode}: keep only selected native classification headers");
                assert_eq!(worker.shard_bytes, std::fs::metadata(dir.path().join("shared-and-core.safetensors"))?.len());
            }
        }
        Ok(())
    }

    #[test]
    fn kimi_original_k6_k8_reuse_native_and_width_specific_layouts() -> Result<()> {
        use crate::families::kimi_k3::{config::KimiK3Config, core::CoreInventory};
        let dir = tempfile::tempdir()?;
        let config = testing::kimi_k3_config();
        let tensors = testing::kimi_k3_tensors(false);
        let shapes = tensors.iter().filter(|(n, ..)| n.strip_suffix(".weight")
            .is_some_and(|s| crate::families::kimi_k3::core::split(s).is_some()))
            .map(|(n, _, s)| (n.clone(), s.clone())).collect();
        testing::write_mc_snapshot(dir.path(), &config, &tensors, &shapes);
        for (view, bits) in [(CoreView::Original, 16), (CoreView::K6Compact, 6), (CoreView::K8Compact, 8)] {
            let selected = CheckpointView::select(dir.path(), view)?;
            let checkpoint = Checkpoint::open_view(&selected)?;
            let cfg = KimiK3Config::from_hf(&checkpoint.config)?;
            let expected = tensors.iter().map(|(n, _, s)| (n.clone(), s.clone())).collect();
            let core = CoreInventory::from_checkpoint(&cfg, &checkpoint, &expected)?;
            if bits == 16 {
                assert!(core.projections.is_empty() && core.native.is_empty(), "all-native uses ordinary BF16 path");
            } else {
                assert!(!core.projections.is_empty());
                assert!(core.projections.values().all(|p| p.bits == bits));
            }
            crate::families::kimi_k3::weight_layout::LayerLayout::pair_with_core(&cfg, 1,
                crate::placement::LayerMode::HeadSplit, &core)?;
            let role = Checkpoint::open_view_for_role(&selected, "experts", |n| n.contains(".experts."))?;
            assert!(role.tensors.iter().all(|t| t.meta.name.contains(".experts.")));
        }
        let fp8 = Checkpoint::open_view(&CheckpointView::select(dir.path(), CoreView::Fp8Compact)?)?;
        let cfg = KimiK3Config::from_hf(&fp8.config)?;
        let expected = tensors.iter().map(|(name, _, shape)| (name.clone(), shape.clone())).collect();
        let error = CoreInventory::from_checkpoint(&cfg, &fp8, &expected).unwrap_err().to_string();
        assert!(error.contains("lm_head.weight") && error.contains("vocabulary head") && error.contains("Kimi"), "{error}");
        Ok(())
    }

    #[test]
    fn k31_views_preserve_native_indexer_and_charge_selected_core() -> Result<()> {
        use crate::families::kimi_k3::{config::{tests::k31_config, KimiK3Config},
            core::CoreInventory, weight_layout::LayerLayout};
        let dir = tempfile::tempdir()?;
        let config = k31_config();
        let cfg = KimiK3Config::from_hf(&config)?;
        let mut tensors = testing::kimi_k3_tensors(false);
        for (layer, kind) in cfg.attention.iter().enumerate() {
            if *kind != crate::families::kimi_k3::config::KimiAttention::Mla { continue; }
            let root = format!("language_model.model.layers.{layer}.self_attn.indexer");
            for (leaf, shape) in [("wq_b.weight", vec![4096,1536]), ("wk.weight", vec![128,7168]),
                ("weights_proj.weight", vec![32,7168]), ("k_norm.weight", vec![128]), ("k_norm.bias", vec![128])] {
                tensors.push(testing::t(format!("{root}.{leaf}"), "BF16", &shape));
            }
        }
        let shapes = tensors.iter().filter(|(n, ..)| n.strip_suffix(".weight")
            .is_some_and(|s| crate::families::kimi_k3::core::split(s).is_some()))
            .map(|(n, _, s)| (n.clone(), s.clone())).collect();
        testing::write_mc_snapshot(dir.path(), &config, &tensors, &shapes);
        let mut bytes = Vec::new();
        let mut identities = std::collections::BTreeSet::new();
        for (view, bits) in [(CoreView::Original, 16), (CoreView::K8Compact, 8), (CoreView::K6Compact, 6)] {
            let selected = CheckpointView::select(dir.path(), view)?;
            identities.insert(selected.fingerprint().to_owned());
            let checkpoint = Checkpoint::open_view(&selected)?;
            let expected = tensors.iter().map(|(n, _, s)| (n.clone(), s.clone())).collect();
            let core = CoreInventory::from_checkpoint(&cfg, &checkpoint, &expected)?;
            if bits == 16 { assert!(core.projections.is_empty() && core.native.is_empty()); }
            else { assert!(!core.projections.is_empty() && core.projections.values().all(|p| p.bits == bits)); }
            let index: Vec<_> = checkpoint.tensors.iter().filter(|t| t.meta.name.contains(".indexer.")).collect();
            assert_eq!(index.len(), 24*5);
            assert!(index.iter().all(|t| t.meta.dtype == cuteafd_core::DType::Bf16));
            let layouts = LayerLayout::pair_with_core(&cfg, 3, crate::placement::LayerMode::ContextSplit, &core)?;
            bytes.push(layouts.iter().map(|l| l.bytes()).collect::<Result<Vec<_>>>()?.iter().sum::<usize>());
        }
        assert_eq!(identities.len(), 3);
        assert!(bytes[0] > bytes[1] && bytes[1] > bytes[2], "{bytes:?}");
        let fp8 = Checkpoint::open_view(&CheckpointView::select(dir.path(), CoreView::Fp8Compact)?)?;
        let expected = tensors.iter().map(|(name, _, shape)| (name.clone(), shape.clone())).collect();
        let error = CoreInventory::from_checkpoint(&cfg, &fp8, &expected).unwrap_err().to_string();
        assert!(error.contains("lm_head.weight") && error.contains("Kimi"), "{error}");
        let plain = tempfile::tempdir()?;
        testing::write_snapshot(plain.path(), &config, &tensors, None);
        assert!(CheckpointView::select(plain.path(), CoreView::Fp8Compact).unwrap_err().to_string()
            .contains("requires cuteafd-mc.json"));
        assert!(CheckpointView::select(plain.path(), CoreView::K8Compact).is_err());
        Ok(())
    }

    #[test]
    fn muse_original_k6_k8_admit_distinct_exact_core_bytes() -> Result<()> {
        use crate::families::muse_spark::{config::MuseSparkConfig, core::CoreInventory, weight_layout::LayerLayout};
        let dir = tempfile::tempdir()?;
        let mut config = testing::muse_spark_config();
        config.as_object_mut().unwrap().remove("audio_config");
        config.as_object_mut().unwrap().remove("vision_config");
        let cfg = MuseSparkConfig::from_hf(&config)?;
        let (tensors, _) = testing::muse_spark_tensors(&[]);
        let shapes = crate::families::muse_spark::core::expected(&cfg);
        testing::write_mc_snapshot(dir.path(), &config, &tensors, &shapes);
        let mut bytes = Vec::new();
        for (view, bits) in [(CoreView::Original, 16), (CoreView::K6Compact, 6), (CoreView::K8Compact, 8)] {
            let checkpoint = Checkpoint::open_view(&CheckpointView::select(dir.path(), view)?)?;
            let core = CoreInventory::from_checkpoint(&cfg, &checkpoint)?;
            assert!(core.projections.values().all(|p| p.bits == bits));
            if bits == 16 {
                assert!(core.projections.is_empty());
                assert!(!core.native.is_empty());
                assert!(core.native.values().all(|weight| !weight.fp8));
            }
            for ranks in [1, 2] {
                for layer in LayerLayout::ranks(&cfg, 0, ranks, &core)? {
                    for bridge in ["moe.pre_expert", "moe.post_expert"] {
                        assert!(layer.core.contains_key(bridge) || layer.native.contains_key(bridge),
                            "{view}: every bridge must have a bound trellis or native executor");
                        assert!(!layer.operands.contains_key(bridge), "no unbound plain bridge fallback");
                    }
                }
            }
            let pair = LayerLayout::ranks(&cfg, 0, 2, &core)?;
            bytes.push(pair.iter().map(|l| l.operands.values().sum::<usize>()
                + l.core.values().map(|c| c.bytes).sum::<usize>()
                + l.native.values().map(|n| n.bytes().unwrap() as usize).sum::<usize>()).sum::<usize>());
        }
        let selected = CheckpointView::select(dir.path(), CoreView::Original)?;
        let inventory = Checkpoint::inventory_view(&selected)?;
        let full = Checkpoint::open_view(&selected)?;
        let role = crate::plan::files::ReadRole::Spark { rank: 0, world: 4 };
        assert_eq!(crate::plan::files::required_tensors(&inventory, role)?,
            crate::plan::files::required_tensors(&full, role)?);
        let worker = crate::plan::files::open_role_view(&selected, role)?;
        assert!(worker.tensors.iter().all(|t| t.meta.name.contains(".experts.")));
        let worker_core = CoreInventory::from_checkpoint(&cfg, &worker)?;
        assert_eq!(worker_core.native, CoreInventory::from_checkpoint(&cfg, &full)?.native);
        let name = format!("{}0.self_attn.k_proj.weight", crate::families::muse_spark::core::PREFIX);
        assert!(worker.header(&name).is_some());
        assert!(!worker.tensors.iter().any(|t| t.meta.name == name));
        assert!(bytes[0] > bytes[2] && bytes[2] > bytes[1]);
        let fp8 = Checkpoint::open_view(&CheckpointView::select(dir.path(), CoreView::Fp8Compact)?)?;
        let converted = CoreInventory::from_checkpoint(&cfg, &fp8)?;
        assert_eq!(converted.native.len(), shapes.len());
        assert!(converted.native.values().all(|p| p.fp8));
        assert_eq!(converted.head_outputs(&cfg), cfg.vocab_size.next_multiple_of(128));
        let global = crate::families::muse_spark::weight_layout::GlobalLayout::new(&cfg, &converted)?;
        assert_eq!(global.head_bf16, 0);
        assert!(global.head.is_none());
        assert_eq!(global.head_native.as_ref().unwrap().shape, [202112, 8192]);
        let admitted = crate::families::muse_spark::weight_layout::projections(&cfg, &converted, 2, 2752)?;
        assert!(admitted[0].native_head.is_none());
        assert!(admitted[1].native_head.is_some());
        assert!(admitted[1].native_programs.programs.iter().any(|n| n == "native_attention_n202112_k8192_m64_f32"));
        assert!(admitted.iter().all(|rank| rank.native_programs.programs.iter()
            .all(|name| !name.contains("n202112_k8192_m4096"))));
        for rank in 0..2 {
            let layers = (0..cfg.layers).map(|layer| LayerLayout::ranks(&cfg, layer, 2, &converted)
                .map(|v| v[rank].clone())).collect::<Result<Vec<_>>>()?;
            let arena = crate::families::muse_spark::weight_layout::coordinator_arena(rank, 2, &global,
                &layers.iter().collect::<Vec<_>>())?;
            assert!(arena.view("global.head").is_none());
            if rank == 1 {
                assert_eq!(arena.view("lm_head.values").unwrap().bytes(), 202112 * 8192);
                assert_eq!(arena.view("lm_head.scales").unwrap().bytes(), 1579 * 64 * 4);
            } else { assert!(arena.view("lm_head.values").is_none()); }
            assert_eq!(arena.bytes(), arena.payload_bytes() + arena.padding_bytes());
        }
        Ok(())
    }

    #[test]
    fn flash_original_k6_k8_keep_native_fallback_and_exact_widths() -> Result<()> {
        use crate::families::glm5_flash::{GlmNextConfig, core::CoreInventory};
        let dir = tempfile::tempdir()?;
        let mut config = testing::glm5_flash_config(2);
        config["text_config"]["vocab_size"] = 128.into();
        let cfg = GlmNextConfig::from_hf(&config)?;
        let tensors = testing::glm5_flash_tensors(&config);
        let shapes = crate::families::glm5_flash::core::expected(&cfg);
        testing::write_mc_snapshot(dir.path(), &config, &tensors, &shapes);
        for (view, bits) in [(CoreView::Original, 16), (CoreView::K6Compact, 6), (CoreView::K8Compact, 8)] {
            let checkpoint = Checkpoint::open_view(&CheckpointView::select(dir.path(), view)?)?;
            let core = CoreInventory::from_checkpoint(&cfg, &checkpoint)?;
            if bits == 16 {
                assert!(core.projections.is_empty());
                assert!(!core.native.is_empty());
                assert!(core.native.values().all(|weight| !weight.fp8));
            }
            else { assert!(core.projections.values().all(|p| p.bits == bits)); }
            for layer in 0..cfg.layers { core.layer_weights(&cfg, layer, 2, None)?; }
        }
        let selected = CheckpointView::select(dir.path(), CoreView::Fp8Compact)?;
        let checkpoint = Checkpoint::open_view(&selected)?;
        let core = CoreInventory::from_checkpoint(&cfg, &checkpoint)?;
        assert!(core.projections.is_empty());
        assert_eq!(core.native.len(), shapes.len());
        assert!(core.native.iter().all(|(stem, weight)| weight.fp8 && selected.converts_to_fp8(&format!("{stem}.weight"))));
        let arenas = crate::families::glm5_flash::weight_layout::target(&checkpoint, &cfg,
            &vec![crate::placement::LayerMode::HeadSplit; cfg.layers], 2, false, false,
            crate::families::glm5_flash::resident::GlmfHeadFormat::Bf16, None)?;
        for (rank, arena) in arenas.iter().enumerate() {
            for layer in 0..cfg.layers {
                for layout in core.layer_native(&cfg, layer, 2, None)?[rank].values() {
                    let [n, k] = layout.shape;
                    assert_eq!(arena.view(&format!("{}.payload", layout.stem)).unwrap().bytes(), n*k);
                    assert_eq!(arena.view(&format!("{}.scales", layout.stem)).unwrap().bytes(), n/128*(k/128)*4);
                    assert!(arena.view(&format!("{}.weight", layout.stem)).is_none());
                }
            }
            assert_eq!(arena.bytes(), arena.payload_bytes() + arena.padding_bytes());
        }
        let [n, k] = core.native["lm_head"].shape;
        assert_eq!(arenas[0].view("lm_head.payload").unwrap().bytes(), n*k);
        assert_eq!(arenas[0].view("lm_head.scales").unwrap().bytes(), (n/128*(k/128)*4).max(256));
        assert!(arenas.iter().all(|arena| arena.view("lm_head.weight").is_none()));

        // Selecting only one member stages its BF16 siblings without converting them.
        let subset = std::collections::BTreeMap::from([(
            "model.language_model.layers.0.self_attn.q_proj.weight".to_owned(),
            shapes["model.language_model.layers.0.self_attn.q_proj.weight"].clone())]);
        let siblings = tensors.iter().filter_map(|(name, dtype, shape)| {
            if let Some(shape) = shapes.get(name) { return Some((name.clone(), "BF16", shape.clone())); }
            let core_scale = name.strip_suffix(".weight_scale_inv")
                .is_some_and(|stem| shapes.contains_key(&format!("{stem}.weight")));
            (!core_scale).then(|| (name.clone(), *dtype, shape.clone()))
        }).collect::<Vec<_>>();
        testing::write_mc_snapshot(dir.path(), &config, &siblings, &subset);
        let checkpoint = Checkpoint::open_view(&CheckpointView::select(dir.path(), CoreView::Fp8Compact)?)?;
        let core = CoreInventory::from_checkpoint(&cfg, &checkpoint)?;
        assert_eq!(core.native.values().filter(|weight| weight.fp8).count(), 1);
        assert!(core.native["model.language_model.layers.0.self_attn.q_proj"].fp8);
        assert!(!core.native["model.language_model.layers.0.self_attn.k_proj"].fp8);
        Ok(())
    }
}
