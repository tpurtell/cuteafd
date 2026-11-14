//! One immutable dense-core view, selected before inventory or payload reads.
use super::{mc::{Manifest, MC_MANIFEST}, CoreView};
use anyhow::{ensure, Context, Result};
use serde::Deserialize;
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::{collections::BTreeMap, io::Read, path::{Path, PathBuf}, sync::Arc};

/// Changing the converter invalidates only derived FP8 view identities.
pub const FP8_RECIPE: &str = "row-major-e4m3-f32-block128-rne-amax448-v1";

#[derive(Debug, Clone)]
pub struct CheckpointView {
    snapshot: PathBuf,
    mode: CoreView,
    config_name: String,
    index_name: String,
    quant_names: Vec<String>,
    fingerprint: String,
    metadata_digests: BTreeMap<String, [u8; 32]>,
    config: Arc<Value>,
    quantize_config: Option<Arc<Value>>,
    weight_map: Option<Arc<BTreeMap<String, String>>>,
    fp8_tensors: Arc<Vec<String>>,
    pub(super) classification_headers: Arc<Vec<super::CheckpointTensor>>,
}

pub(super) fn metadata_bytes(path: &Path) -> Result<Vec<u8>> {
    let file = std::fs::File::open(path)?;
    ensure!(file.metadata()?.len() <= super::MAX_JSON_BYTES, "{} exceeds metadata size limit", path.display());
    let mut bytes = Vec::new();
    file.take(super::MAX_JSON_BYTES + 1).read_to_end(&mut bytes)?;
    ensure!(bytes.len() as u64 <= super::MAX_JSON_BYTES, "{} exceeds metadata size limit", path.display());
    Ok(bytes)
}

fn check_references(value: &Value, quant_names: &[String], index: &str) -> Result<()> {
    match value {
        Value::Object(object) => for (key, value) in object {
            if key == "per_module_metadata" {
                let reference = value.as_str().context("per_module_metadata must name a metadata file")?;
                let (file, field) = reference.split_once(':').context("per_module_metadata must name file:field")?;
                ensure!(quant_names.iter().any(|name| name == file) && field == "tensor_storage",
                    "core view references unselected quantization metadata {reference}");
            }
            if key == "transformers_weights" && !value.is_null() {
                ensure!(value.as_str() == Some(index), "core view references unselected index {value}");
            }
            check_references(value, quant_names, index)?;
        },
        Value::Array(values) => for value in values { check_references(value, quant_names, index)?; },
        _ => (),
    }
    Ok(())
}

impl CheckpointView {
    pub fn select(snapshot: &Path, mode: CoreView) -> Result<Self> {
        Self::select_recipe(snapshot, mode, FP8_RECIPE)
    }

    fn select_recipe(snapshot: &Path, mode: CoreView, recipe: &str) -> Result<Self> {
        let mc = Manifest::read(snapshot)?;
        let variant = mc.as_ref().map(|(manifest, _)| manifest.select(mode)).transpose()?;
        ensure!(mode == CoreView::Original || variant.is_some(),
            "CORE_VIEW={mode} requires {MC_MANIFEST}; no compact view exists in a plain checkpoint");
        let config_name = variant.map_or("config.json", |v| v.config.as_str());
        let config_bytes = metadata_bytes(&snapshot.join(config_name))
            .with_context(|| format!("selecting CORE_VIEW={mode}: {config_name}"))?;
        let config: Value = serde_json::from_slice(&config_bytes)?;
        let index_name = variant.map(|v| Ok(v.index.clone()))
            .unwrap_or_else(|| super::index_name_from_config(&config))?;
        let quant_names: Vec<String> = if let Some(variant) = variant { vec![variant.quantize.clone()] }
            else { ["quantize_config.json", "quantization_config.json", "hf_quant_config.json"].into_iter()
                .filter(|name| snapshot.join(name).is_file()).map(str::to_owned).collect() };
        let allowed_quant = if variant.is_some() { quant_names.clone() } else {
            ["quantize_config.json", "quantization_config.json", "hf_quant_config.json"].map(str::to_owned).to_vec()
        };
        let fp8_tensors = if mode == CoreView::Fp8Compact {
            variant.context("fp8-compact requires MC original core_tensors")?.core_tensors.clone()
        } else { Vec::new() };
        let mut hash = Sha256::new();
        hash.update(b"cuteafd-core-view-v1\0");
        hash.update(mode.name());
        if mode == CoreView::Fp8Compact { hash.update(recipe.as_bytes()); }
        let mut metadata_digests = BTreeMap::new();
        let mut bind = |name: &str, bytes: &[u8]| {
            hash.update((name.len() as u64).to_le_bytes());
            hash.update(name);
            hash.update((bytes.len() as u64).to_le_bytes());
            hash.update(bytes);
            metadata_digests.insert(name.to_owned(), Sha256::digest(bytes).into());
        };
        if let Some((_, bytes)) = &mc { bind(MC_MANIFEST, bytes); }
        let mut quantize_config = None;
        for name in std::iter::once(config_name).chain(quant_names.iter().map(String::as_str)) {
            let path = snapshot.join(name);
            let bytes = if name == config_name { config_bytes.clone() } else { metadata_bytes(&path)? };
            if let Some(variant) = variant { variant.check_metadata(name, &bytes)?; }
            let value = if name == config_name { config.clone() } else {
                crate::formats::exl3_storage::parse_storage_metadata(&path, &bytes, None)?
            };
            check_references(&value, &allowed_quant, &index_name)?;
            bind(name, &bytes);
            if name != config_name && quantize_config.is_none() { quantize_config = Some(Arc::new(value)); }
        }
        let weight_map = if snapshot.join(&index_name).is_file() {
            #[derive(Deserialize)]
            struct Index { weight_map: BTreeMap<String, String> }
            let bytes = metadata_bytes(&snapshot.join(&index_name))?;
            let index: Index = serde_json::from_slice(&bytes).context("decoding selected weight_map")?;
            if let Some((manifest, _)) = &mc {
                let variant = manifest.select(mode)?;
                variant.check_metadata(&index_name, &bytes)?;
                manifest.check_index(variant, &index.weight_map)?;
            }
            bind(&index_name, &bytes);
            Some(Arc::new(index.weight_map))
        } else {
            ensure!(mc.is_none(), "MC view requires selected index {index_name}");
            None
        };
        Ok(Self { snapshot: snapshot.into(), mode, config_name: config_name.into(), index_name,
            quant_names, fingerprint: format!("{:x}", hash.finalize()), config: Arc::new(config),
            quantize_config, weight_map, fp8_tensors: Arc::new(fp8_tensors), metadata_digests, classification_headers: Arc::default() })
    }

    /// The quant's original tensor IDs are the complete conversion definition.
    pub fn fp8_tensors(&self) -> &[String] { &self.fp8_tensors }
    pub fn converts_to_fp8(&self, name: &str) -> bool {
        self.fp8_tensors.binary_search_by(|tensor| tensor.as_str().cmp(name)).is_ok()
    }

    pub fn snapshot(&self) -> &Path { &self.snapshot }
    pub fn mode(&self) -> CoreView { self.mode }
    pub fn is_mc(&self) -> bool { self.metadata_digests.contains_key(MC_MANIFEST) }
    /// The core view this checkpoint serves, when it declares views at all (an MC checkpoint
    /// with `cuteafd-mc.json`). A plain checkpoint has only its own core: `None`, not
    /// `original`, so nothing downstream labels it with a view it cannot switch.
    pub fn declared_view(&self) -> Option<CoreView> { self.is_mc().then_some(self.mode) }
    /// The FP8 conversion recipe of a load-time `fp8-compact` view, else `None`.
    pub fn fp8_recipe(&self) -> Option<&'static str> { (self.mode == CoreView::Fp8Compact).then_some(FP8_RECIPE) }
    pub fn config_name(&self) -> &str { &self.config_name }
    pub fn index_name(&self) -> &str { &self.index_name }
    pub fn quant_names(&self) -> &[String] { &self.quant_names }
    pub fn fingerprint(&self) -> &str { &self.fingerprint }
    pub fn metadata_sha256(&self, name: &str) -> Result<String> {
        let digest = self.metadata_digests.get(name).context("metadata was not selected")?;
        Ok(digest.iter().map(|byte| format!("{byte:02x}")).collect())
    }
    pub fn config(&self) -> Result<Value> { Ok(self.config.as_ref().clone()) }
    pub(super) fn quantize_config(&self) -> Option<Value> { self.quantize_config.as_deref().cloned() }
    pub(super) fn weight_map(&self) -> Option<&BTreeMap<String, String>> { self.weight_map.as_deref() }
    pub(super) fn native_core_names(&self) -> impl Iterator<Item = &str> {
        // Family detection is config-only; do not open its model before these headers exist.
        let metadata = super::Checkpoint { view: None, snapshot: self.snapshot.clone(),
            config: self.config.as_ref().clone(), quantize_config: None,
            weight_map: BTreeMap::new(), tensors: Vec::new(), missing_shards: Vec::new(), shard_bytes: 0 };
        let eager_core = crate::plan::family::detect(&metadata)
            .is_some_and(|family| family.eager_native_core_headers());
        self.weight_map.iter().flat_map(|weights| weights.keys()).map(String::as_str)
            .filter(move |name| eager_core && !name.contains(".experts.") && !name.contains(".mtp.")
                && (name.ends_with(".weight") || name.ends_with(".weight_scale_inv") || name.ends_with(".qscale_weight")
                    || name.ends_with(".weight_scale") || name.ends_with(".input_scale"))
                && (name.contains(".self_attn.") || name.contains(".attention.") || name.contains(".mlp.") || name.starts_with("lm_head.")
                    || name.starts_with("language_model.lm_head.") || name.starts_with("output.")
                    || name.contains(".shared_experts.") || name.contains(".feed_forward.")))
    }
    pub(crate) fn check_metadata_bytes(&self, name: &str, bytes: &[u8]) -> Result<()> {
        let expected = self.metadata_digests.get(name).context("metadata was not selected")?;
        let actual: [u8; 32] = Sha256::digest(bytes).into();
        ensure!(*expected == actual, "selected metadata {name} changed after core view selection");
        Ok(())
    }
    pub fn index_path(&self) -> PathBuf { self.snapshot.join(&self.index_name) }
}
