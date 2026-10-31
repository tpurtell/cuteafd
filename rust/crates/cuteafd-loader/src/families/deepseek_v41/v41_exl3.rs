//! Validated routed-only EXL3 metadata for the native V4.1 engine.
use crate::formats::exl3_contract::{check_gptqmodel_manifest, check_exllamav3_manifest, check_header_widths};
use crate::formats::exl3_storage::{
    check_quantization_config, derive_storage_map, mcg_marker_shape,
    Exl3Module, Exl3StorageMap, Exl3StorageSource,
};
use crate::SafetensorsTensorMetadata;
use anyhow::{ensure, Context, Result};
use cuteafd_core::DType;
use serde_json::Value;
use std::{
    collections::{BTreeMap, BTreeSet},
    fs::File,
    io::Read,
    ops::Range,
    path::Path,
};

// Large MoE publications carry per-tensor maps exceeding 64 MiB.
const MAX_MANIFEST_BYTES: u64 = 512 * 1024 * 1024;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum V41Exl3ProjectionKind {
    Gate,
    Up,
    Down,
}

/// One projection's on-disk contract, before conversion into native kernel tiles.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct V41Exl3Projection {
    pub name: String,
    pub kind: V41Exl3ProjectionKind,
    pub bits: usize,
    pub input_features: usize,
    pub output_features: usize,
}

impl V41Exl3Projection {
    pub fn trellis_shape(&self) -> [usize; 3] {
        [
            self.input_features / 16,
            self.output_features / 16,
            16 * self.bits,
        ]
    }

    pub fn trellis_bytes(&self) -> usize {
        self.input_features * self.output_features * self.bits / 8
    }

    /// Check the physical safetensors header against the declared projection.
    /// The manifest alone is never proof of the actual checkpoint's layout.
    pub fn validate_tensor(&self, tensor: &SafetensorsTensorMetadata) -> Result<()> {
        let (prefix, suffix) = tensor
            .name
            .rsplit_once('.')
            .context("missing EXL3 tensor suffix")?;
        ensure!(
            prefix == self.name,
            "EXL3 tensor belongs to another projection"
        );
        let (dtype, shape, bytes) = match suffix {
            "trellis" => (
                DType::I16,
                self.trellis_shape().to_vec(),
                self.trellis_bytes(),
            ),
            "suh" => (
                DType::F16,
                vec![self.input_features],
                self.input_features * 2,
            ),
            "svh" => (
                DType::F16,
                vec![self.output_features],
                self.output_features * 2,
            ),
            "mcg" => (DType::I32, vec![], 4),
            _ => anyhow::bail!("unexpected EXL3 tensor {}", tensor.name),
        };
        let shape_ok = if suffix == "mcg" { mcg_marker_shape(&tensor.shape) } else { tensor.shape == shape };
        ensure!(
            tensor.dtype == dtype && shape_ok && tensor.byte_length == bytes as u64,
            "EXL3 manifest/header mismatch for {}",
            tensor.name
        );
        Ok(())
    }

    /// Partition complete H128 rotation blocks. Equal TP4 slices of 2304
    /// channels would split blocks, so the first two ranks own one extra block.
    /// TP2 (9 blocks) and TP3 (6 blocks) divide 18 exactly, so they carry no
    /// padding and no duplicated boundary block.
    pub fn intermediate_partition(&self, world: usize, rank: usize) -> Result<Range<usize>> {
        ensure!(
            (1..=8).contains(&world) && rank < world,
            "invalid EXL3 TP rank/world"
        );
        let intermediate = match self.kind {
            V41Exl3ProjectionKind::Down => self.input_features,
            _ => self.output_features,
        };
        // Whole H128 blocks: V4.1 TP3/TP6 own exactly six/three blocks;
        // uneven geometries put one extra block on the first ranks.
        ensure!(
            intermediate % 128 == 0,
            "EXL3 intermediate axis is not H128 aligned"
        );
        let blocks = intermediate / 128;
        ensure!(blocks >= world, "EXL3 partition would be empty");
        let map = cuteafd_core::tp_units::RankUnits::new(cuteafd_core::tp_units::UnitRule::Contiguous,
            1, intermediate, 128, 0, world, rank)?;
        map.column_ranges(0)?.into_iter().next().context("empty EXL3 partition")
    }
}

/// How a checkpoint names its routed EXL3 projections.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum V41Exl3Naming {
    /// DeepSeek V4.1: `layers.{L}.ffn.experts.{E}.w1|w3|w2` and `mtp.{S}.ffn.experts...`.
    CheckpointNative,
    /// Hugging Face module names (DeepSeek V4 EXL3 publications):
    /// `model.layers.{L}.mlp.experts.{E}.gate_proj|up_proj|down_proj` and
    /// `mtp.{S}.mlp.experts...`.
    HfMlp,
    /// Multimodal Hugging Face checkpoints (GLM 5.3 Flash):
    /// `model.language_model.layers.{L}.mlp.experts.{E}.gate_proj|up_proj|down_proj`,
    /// the native MTP layer after the `layers` backbone layers as draft stage 0.
    HfLanguageModel { layers: usize },
    /// Multimodal Hugging Face checkpoints with a separate MTP module (Qwen 3.8
    /// Flash Next): backbone as [`Self::HfLanguageModel`], draft stage `S` at
    /// `mtp.layers.{S}.mlp.experts.{E}.gate_proj|up_proj|down_proj`.
    HfLanguageModelMtp,
    /// Prefix, routed module and gate/up/down stems discovered from tensor names.
    HeaderDerived { prefix: String, module: String, stems: [String; 3] },
}

impl V41Exl3Naming {
    /// The projection name (without the `.trellis`/`.suh`/... suffix).
    pub fn projection(&self, draft: bool, layer: usize, expert: usize, kind: V41Exl3ProjectionKind) -> String {
        match self {
            Self::CheckpointNative => {
                let stem = match kind {
                    V41Exl3ProjectionKind::Gate => "w1",
                    V41Exl3ProjectionKind::Up => "w3",
                    V41Exl3ProjectionKind::Down => "w2",
                };
                let prefix = if draft { "mtp" } else { "layers" };
                format!("{prefix}.{layer}.ffn.experts.{expert}.{stem}")
            }
            Self::HfMlp => {
                let stem = match kind {
                    V41Exl3ProjectionKind::Gate => "gate_proj",
                    V41Exl3ProjectionKind::Up => "up_proj",
                    V41Exl3ProjectionKind::Down => "down_proj",
                };
                let prefix = if draft { "mtp" } else { "model.layers" };
                format!("{prefix}.{layer}.mlp.experts.{expert}.{stem}")
            }
            Self::HfLanguageModel { layers } => {
                let stem = match kind {
                    V41Exl3ProjectionKind::Gate => "gate_proj",
                    V41Exl3ProjectionKind::Up => "up_proj",
                    V41Exl3ProjectionKind::Down => "down_proj",
                };
                let layer = if draft { layers + layer } else { layer };
                format!("model.language_model.layers.{layer}.mlp.experts.{expert}.{stem}")
            }
            Self::HfLanguageModelMtp => {
                let stem = match kind {
                    V41Exl3ProjectionKind::Gate => "gate_proj",
                    V41Exl3ProjectionKind::Up => "up_proj",
                    V41Exl3ProjectionKind::Down => "down_proj",
                };
                let prefix = if draft { "mtp" } else { "model.language_model" };
                format!("{prefix}.layers.{layer}.mlp.experts.{expert}.{stem}")
            }
            Self::HeaderDerived { prefix, module, stems } => {
                let stem = &stems[match kind {
                    V41Exl3ProjectionKind::Gate => 0,
                    V41Exl3ProjectionKind::Up => 1,
                    V41Exl3ProjectionKind::Down => 2,
                }];
                let prefix = if draft { "mtp" } else { prefix.as_str() };
                format!("{prefix}.{layer}.{module}.experts.{expert}.{stem}")
            }
        }
    }
}

#[derive(Debug)]
pub struct V41Exl3Manifest {
    /// Routed-expert extents of every target and draft layer.
    pub experts: crate::RoutedExpertShape,
    pub naming: V41Exl3Naming,
    pub projections: BTreeMap<String, V41Exl3Projection>,
    /// One checkpoint-wide family, shared by all target and draft layers.
    pub(crate) decoder_tiers: Vec<usize>,
    /// Where the storage layout was read from (always checked against the headers).
    pub storage: Exl3StorageSource,
}

impl V41Exl3Manifest {
    pub fn decoder_tiers(&self) -> &[usize] {
        &self.decoder_tiers
    }
}

pub(crate) fn decoder_family(
    projections: &BTreeMap<String, V41Exl3Projection>,
) -> Result<Vec<usize>> {
    let mut bits: BTreeSet<_> = projections.values().map(|p| p.bits).collect();
    ensure!(
        !bits.is_empty() && bits.iter().all(|b| (2..=6).contains(b)),
        "EXL3 decoder family must contain K2..K6 projections"
    );
    ensure!(bits.len() <= 2,
        "unsupported: EXL3 with {} expert tiers (2 supported); tensors: {}",
        bits.len(), bits.iter().filter_map(|bit| projections.values().find(|p| p.bits == *bit))
            .map(|p| format!("{} (K{})", p.name, p.bits)).collect::<Vec<_>>().join(", "));
    // Native mixed kernels retain an empty adjacent tier for uniform models.
    if bits.len() == 1 {
        let bit = *bits.first().unwrap();
        bits.insert(if bit == 6 { 5 } else { bit + 1 });
    }
    Ok(bits.into_iter().collect())
}

/// The checkpoint's decoder tiers, or `preferred` (the tier pair the
/// family's packages ship) when it covers every projection's bits.
pub(crate) fn decoder_family_preferring(
    projections: &BTreeMap<String, V41Exl3Projection>,
    preferred: &[usize],
) -> Result<Vec<usize>> {
    let family = decoder_family(projections)?;
    let covered = preferred.len() == 2
        && preferred[0] + 1 == preferred[1]
        && preferred.iter().all(|bits| (2..=6).contains(bits))
        && projections.values().all(|p| preferred.contains(&p.bits));
    Ok(if covered { preferred.to_vec() } else { family })
}

pub(crate) fn read_json(path: &Path, limit: u64) -> Result<Value> {
    let bytes = read_bytes(path, limit)?;
    let mut value: Value = serde_json::from_slice(&bytes).with_context(|| format!("parsing {}", path.display()))?;
    normalize_legacy_names(&mut value);
    Ok(value)
}

/// Checkpoints published before the ds41rt -> cuteafd rename carry `ds41rt`
/// keys (`meta.ds41rt`, `ds41rt_*`) and `ds41rt.`-prefixed schema strings; read
/// them under the current names.
pub(crate) fn normalize_legacy_names(value: &mut Value) {
    match value {
        Value::Object(map) => {
            let legacy: Vec<String> = map.keys().filter(|k| *k == "ds41rt" || k.starts_with("ds41rt_")).cloned().collect();
            for key in legacy {
                let renamed = format!("cuteafd{}", &key["ds41rt".len()..]);
                if !map.contains_key(&renamed) {
                    let entry = map.remove(&key).unwrap();
                    map.insert(renamed, entry);
                }
            }
            map.values_mut().for_each(normalize_legacy_names);
        }
        Value::Array(items) => items.iter_mut().for_each(normalize_legacy_names),
        Value::String(text) if text.starts_with("ds41rt.") => {
            *text = format!("cuteafd.{}", &text["ds41rt.".len()..]);
        }
        _ => {}
    }
}

fn is_routed_module(name: &str) -> bool {
    name.split_once(".experts.").is_some_and(|(_, tail)|
        tail.split_once('.').is_some_and(|(expert, _)| expert.parse::<usize>().is_ok()))
}

fn header_naming(routed: &BTreeMap<&str, &Exl3Module>) -> Result<V41Exl3Naming> {
    let mut layouts = BTreeSet::new();
    let mut stems = BTreeSet::new();
    for name in routed.keys().filter(|name| !name.starts_with("mtp.")) {
        let (left, _) = name.split_once(".experts.").context("EXL3 expert ordinal absent")?;
        let (layer, module) = left.rsplit_once('.').context("EXL3 expert module absent")?;
        let (prefix, ordinal) = layer.rsplit_once('.').context("EXL3 layer prefix absent")?;
        ordinal.parse::<usize>().context("EXL3 layer ordinal is not an integer")?;
        ensure!(prefix.ends_with(".layers") || prefix == "layers", "EXL3 layer prefix lacks layers segment");
        layouts.insert((prefix.to_owned(), module.to_owned()));
        stems.insert(name.rsplit_once('.').context("EXL3 projection stem absent")?.1);
    }
    ensure!(layouts.len() == 1, "EXL3 headers must identify one backbone layer prefix/module: {layouts:?}");
    // Gate and up have identical shapes; their standard naming convention fixes the roles.
    let roles = [["gate_proj", "up_proj", "down_proj"], ["w1", "w3", "w2"]]
        .into_iter().find(|roles| roles.iter().copied().collect::<BTreeSet<_>>() == stems)
        .with_context(|| format!("EXL3 headers have unsupported or mixed projection stems: {stems:?}"))?;
    let (prefix, module) = layouts.into_iter().next().unwrap();
    Ok(V41Exl3Naming::HeaderDerived { prefix, module, stems: roles.map(str::to_owned) })
}

/// Routed EXL3 experts of a DeepSeek V4, GLM 5.x or Qwen 3.8 checkpoint with
/// Hugging Face module names. The storage layout is the safetensors headers'
/// (`headers`: every tensor of the checkpoint); a storage map in the snapshot
/// (GPTQModel's `quantize_config.json`, exllamav3's `quantization_config.json`)
/// is read only to check that it agrees. Only routed experts (backbone and
/// draft) are read here; every other tensor stays with the coordinator.
/// `tiers` is the decoder-tier pair of the family's expert packages: a
/// checkpoint whose bits it covers selects it, a uniform one with an empty
/// tier (GLM 5.3 Flash K4 on the K3/K4 package); otherwise the checkpoint's
/// own tiers.
pub(crate) fn read_deepseek_v4_exl3_manifest<'a>(
    snapshot: &Path,
    backbone: crate::RoutedExpertShape,
    headers: impl IntoIterator<Item = &'a SafetensorsTensorMetadata>,
    tiers: &[usize],
) -> Result<V41Exl3Manifest> {
    let config = read_json(&snapshot.join("config.json"), 1024 * 1024)?;
    let derived = derive_storage_map(headers)?;
    // MTP starts after every target layer, which can exceed the routed bound
    // (Step-5's last target layer is dense).
    let target_layers = config.get("text_config").unwrap_or(&config)["num_hidden_layers"].as_u64()
        .map_or(backbone.layers, |layers| (layers as usize).max(backbone.layers));
    let source = check_storage_maps(snapshot, &config, &derived, target_layers)?;
    build_deepseek_v4_manifest(source, &derived, backbone, tiers)
}

/// Which storage map the snapshot carries, checked against `derived`: a
/// GPTQModel `quantize_config.json` (and its `quantization_config.json`
/// copy), an exllamav3 `quantization_config.json`, or none (the compact
/// `config.json` block alone).
fn check_storage_maps(snapshot: &Path, config: &Value, derived: &Exl3StorageMap, backbone_layers: usize) -> Result<Exl3StorageSource> {
    let compact = config.get("quantization_config").context("config.json has no quantization_config")?;
    use sha2::{Digest, Sha256};
    let mut source = None;
    let paths = ["quantize_config.json", "quantization_config.json"].map(|file| snapshot.join(file));
    let present = paths.iter().filter(|path| path.is_file()).collect::<Vec<_>>();
    let first = present.first().map(|path| read_bytes(path, MAX_MANIFEST_BYTES)).transpose()?;
    if let [left, right] = present.as_slice() {
        ensure!(left.metadata()?.len() == right.metadata()?.len(),
            "quantize_config.json and quantization_config.json differ in size");
        let other = read_bytes(right, MAX_MANIFEST_BYTES)?;
        ensure!(Sha256::digest(first.as_ref().unwrap()) == Sha256::digest(&other),
            "quantize_config.json and quantization_config.json differ in SHA256");
    }
    if let Some(path) = present.first() {
        let file = path.file_name().unwrap().to_string_lossy();
        let mut manifest = crate::formats::exl3_storage::parse_storage_metadata(
            path, first.as_ref().unwrap(), Some(derived))?;
        normalize_legacy_names(&mut manifest);
        let flavor = if manifest.get("checkpoint_format").is_some() {
            check_gptqmodel_manifest(compact, &manifest, derived, backbone_layers).with_context(|| format!("checking {file}"))?;
            Exl3StorageSource::GptqModel
        } else {
            check_exllamav3_manifest(compact, &manifest).with_context(|| format!("checking {file}"))?;
            Exl3StorageSource::Exllamav3
        };
        source.get_or_insert(flavor);
    }
    check_header_widths(compact, derived, backbone_layers)?;
    match source {
        Some(source) => Ok(source),
        None => {
            check_quantization_config(compact)?;
            Ok(Exl3StorageSource::Headers)
        }
    }
}

fn read_bytes(path: &Path, limit: u64) -> Result<Vec<u8>> {
    let file = File::open(path).with_context(|| format!("opening {}", path.display()))?;
    let size = file.metadata()?.len();
    ensure!(size <= limit, "{} exceeds metadata size limit", path.display());
    let mut bytes = Vec::with_capacity(usize::try_from(size).context("metadata size exceeds address space")?);
    // Keep the bounded read too: the file may grow after the metadata check.
    file.take(limit + 1).read_to_end(&mut bytes)?;
    ensure!(bytes.len() as u64 <= limit, "{} exceeds metadata size limit", path.display());
    Ok(bytes)
}


fn build_deepseek_v4_manifest(
    storage: Exl3StorageSource,
    derived: &Exl3StorageMap,
    backbone: crate::RoutedExpertShape,
    tiers: &[usize],
) -> Result<V41Exl3Manifest> {
    // Header derivation and optional side-map equality validate every module.
    // Dense core and plain-HF appended MTP belong to coordinator loaders,
    // just as native FP8 MTP tensors do in the original GLM publication.
    let coordinator_mtp = |name: &str| name.strip_prefix("model.layers.")
        .and_then(|tail| tail.split('.').next()?.parse::<usize>().ok())
        .is_some_and(|layer| layer >= backbone.layers);
    let routed: BTreeMap<&str, &Exl3Module> = derived
        .iter()
        .filter(|(name, _)| is_routed_module(name) && !coordinator_mtp(name))
        .map(|(name, module)| (name.as_str(), module))
        .collect();
    ensure!(!routed.is_empty(), "the checkpoint has no routed EXL3 experts (*.mlp.experts.*.trellis)");
    let language_model = routed.keys().any(|name| name.starts_with("model.language_model.layers."));
    let mtp_layers = language_model && routed.keys().any(|name| name.starts_with("mtp.layers."));
    let naming = if mtp_layers {
        V41Exl3Naming::HfLanguageModelMtp
    } else if language_model {
        V41Exl3Naming::HfLanguageModel { layers: backbone.layers }
    } else if routed.keys().filter(|name| !name.starts_with("mtp."))
        .all(|name| name.starts_with("model.layers.") && name.contains(".mlp.experts.")) {
        V41Exl3Naming::HfMlp
    } else {
        header_naming(&routed)?
    };
    // Draft stages use a separate MTP prefix or layers past the backbone.
    let stage_of = |name: &str| -> Option<usize> {
        if mtp_layers {
            name.strip_prefix("mtp.layers.")?.split('.').next()?.parse().ok()
        } else if language_model {
            let layer: usize = name.strip_prefix("model.language_model.layers.")?.split('.').next()?.parse().ok()?;
            layer.checked_sub(backbone.layers)
        } else {
            name.strip_prefix("mtp.")?.split('.').next()?.parse().ok()
        }
    };
    let draft_stages = routed.keys().filter_map(|name| stage_of(name)).max().map_or(0, |stage| stage + 1);
    let draft_experts = routed
        .keys()
        .filter(|name| stage_of(name) == Some(0) && name.contains(".mlp.experts.") && name.ends_with(".gate_proj"))
        .count();
    let (hidden, intermediate) = (backbone.hidden, backbone.intermediate);
    let mut projections = BTreeMap::new();
    for (draft, layers, experts) in [
        (false, backbone.first_layer..backbone.layers, backbone.experts),
        (true, 0..draft_stages, draft_experts),
    ] {
        for layer in layers {
            for expert in 0..experts {
                for kind in [
                    V41Exl3ProjectionKind::Gate,
                    V41Exl3ProjectionKind::Up,
                    V41Exl3ProjectionKind::Down,
                ] {
                    let name = naming.projection(draft, layer, expert, kind);
                    let module = routed.get(name.as_str()).with_context(|| format!("missing projection {name}"))?;
                    let (input, output) = if kind == V41Exl3ProjectionKind::Down {
                        (intermediate, hidden)
                    } else {
                        (hidden, intermediate)
                    };
                    ensure!(
                        (module.input_features, module.output_features) == (input, output),
                        "projection {name} stores {} -> {} features; the model's is {input} -> {output}",
                        module.input_features,
                        module.output_features
                    );
                    ensure!(module.codebook == crate::formats::exl3_storage::Exl3Codebook::Mcg,
                        "projection {name}: routed expert kernels require MCG, found {}", module.codebook.marker());
                    ensure!(
                        (2..=6).contains(&module.bits),
                        "projection {name} is K{}; the expert kernels run K2..K6",
                        module.bits
                    );
                    ensure!(
                        input % 128 == 0 && output % 128 == 0,
                        "projection {name} requires H128-aligned geometry"
                    );
                    projections.insert(
                        name.clone(),
                        V41Exl3Projection { name, kind, bits: module.bits, input_features: input, output_features: output },
                    );
                }
            }
        }
    }
    if let Some(name) = routed.keys().find(|name| !projections.contains_key(**name)) {
        anyhow::bail!(
            "EXL3 manifest includes non-routed or unexpected projections: {name} is not a projection of \
             {} layers x {} experts or {draft_stages} draft stages",
            backbone.layers,
            backbone.experts
        );
    }
    Ok(V41Exl3Manifest {
        experts: crate::RoutedExpertShape { draft_stages, draft_experts, ..backbone },
        naming,
        decoder_tiers: decoder_family_preferring(&projections, tiers)?,
        projections,
        storage,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn deepseek_v4_publication_with(
        bits: impl Fn(&str) -> usize,
        mcg: &[usize],
    ) -> (Value, Value, crate::RoutedExpertShape, Vec<SafetensorsTensorMetadata>) {
        let shape = crate::RoutedExpertShape {
            layers: 2,
            first_layer: 0,
            experts: 3,
            topk: 2,
            hidden: 256,
            intermediate: 384,
            draft_stages: 0,
            draft_experts: 0,
        };
        let compact = serde_json::json!({
            "bits": 2.0, "checkpoint_format": "exl3", "codebook": "mcg", "desc_act": false,
            "format": "exl3", "group_size": -1, "lm_head": false, "meta": {"fallback": null},
            "method": "exl3", "out_scales": "auto", "pack_dtype": "int32", "quant_method": "exl3",
            "module_include": ["^model\\.layers\\.\\d+\\.mlp\\.experts\\.\\d+\\.(?:gate_proj|up_proj|down_proj)$"],
        });
        let header = |name: String, dtype: DType, shape: Vec<usize>, width: u64| SafetensorsTensorMetadata {
            byte_length: width * shape.iter().product::<usize>() as u64,
            name,
            dtype,
            shape,
            byte_offset: 0,
        };
        let mut storage = serde_json::Map::new();
        let mut headers = vec![header("model.norm.weight".into(), DType::Bf16, vec![256], 2)];
        for (draft, layers) in [(false, 2), (true, 1)] {
            for layer in 0..layers {
                for expert in 0..3 {
                    for kind in [V41Exl3ProjectionKind::Gate, V41Exl3ProjectionKind::Up, V41Exl3ProjectionKind::Down] {
                        let name = V41Exl3Naming::HfMlp.projection(draft, layer, expert, kind);
                        let k = bits(&name);
                        let (input, output) = if kind == V41Exl3ProjectionKind::Down { (384, 256) } else { (256, 384) };
                        let tensors = serde_json::json!({
                            format!("{name}.trellis"): {"shape": [input / 16, output / 16, 16 * k], "torch_dtype": "int16"},
                            format!("{name}.suh"): {"shape": [input], "torch_dtype": "float16"},
                            format!("{name}.svh"): {"shape": [output], "torch_dtype": "float16"},
                            format!("{name}.mcg"): {"shape": [], "torch_dtype": "int32"},
                        });
                        storage.insert(name.clone(), serde_json::json!({
                            "bits_per_weight": k, "mcg_multiplier": 3417055213u64,
                            "quant_format": "exl3", "stored_tensors": tensors,
                        }));
                        headers.extend([
                            header(format!("{name}.trellis"), DType::I16, vec![input / 16, output / 16, 16 * k], 2),
                            header(format!("{name}.suh"), DType::F16, vec![input], 2),
                            header(format!("{name}.svh"), DType::F16, vec![output], 2),
                            header(format!("{name}.mcg"), DType::I32, mcg.to_vec(), 4),
                        ]);
                    }
                }
            }
        }
        let mut manifest = compact.clone();
        manifest["tensor_storage"] = Value::Object(storage);
        (serde_json::json!({"quantization_config": compact}), manifest, shape, headers)
    }

    fn deepseek_v4_publication() -> (Value, Value, crate::RoutedExpertShape, Vec<SafetensorsTensorMetadata>) {
        deepseek_v4_publication_with(|_| 2, &[])
    }

    /// Reads the manifest of a snapshot holding `config` and `maps` (file name, contents).
    fn read_publication(
        config: &Value,
        maps: &[(&str, &Value)],
        shape: crate::RoutedExpertShape,
        headers: &[SafetensorsTensorMetadata],
        tiers: &[usize],
    ) -> Result<V41Exl3Manifest> {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("config.json"), serde_json::to_vec(config).unwrap()).unwrap();
        for (file, map) in maps {
            std::fs::write(dir.path().join(file), serde_json::to_vec(map).unwrap()).unwrap();
        }
        read_deepseek_v4_exl3_manifest(dir.path(), shape, headers, tiers)
    }

    #[test]
    fn duplicate_side_maps_match_bytes_before_one_parse() -> Result<()> {
        let (config, manifest, shape, headers) = deepseek_v4_publication();
        read_publication(&config, &[("quantize_config.json", &manifest),
            ("quantization_config.json", &manifest)], shape, &headers, &[])?;
        let mut different = manifest.clone();
        different["unused"] = serde_json::json!(1);
        let error = read_publication(&config, &[("quantize_config.json", &manifest),
            ("quantization_config.json", &different)], shape, &headers, &[]).unwrap_err();
        assert!(format!("{error:#}").contains("differ in size"));
        Ok(())
    }

    #[test]
    fn publication_manifest_over_64_mib_remains_bounded() -> Result<()> {
        use std::io::Write;

        let (config, manifest, shape, headers) = deepseek_v4_publication();
        let dir = tempfile::tempdir()?;
        std::fs::write(dir.path().join("config.json"), serde_json::to_vec(&config)?)?;
        let path = dir.path().join("quantize_config.json");
        let mut file = File::create(&path)?;
        serde_json::to_writer(&mut file, &manifest)?;
        let padding = [b' '; 1024 * 1024];
        for _ in 0..65 {
            file.write_all(&padding)?;
        }
        drop(file);
        assert!(path.metadata()?.len() > 64 * 1024 * 1024);
        let parsed = read_deepseek_v4_exl3_manifest(dir.path(), shape, &headers, &[])?;
        assert_eq!(parsed.projections.len(), headers.len() / 4);
        assert!(read_json(&path, MAX_MANIFEST_BYTES)?["tensor_storage"].is_object());

        // A sparse oversized file exercises the ceiling without allocating its payload.
        File::options().write(true).open(&path)?.set_len(MAX_MANIFEST_BYTES + 1)?;
        let error = read_bytes(&path, MAX_MANIFEST_BYTES).unwrap_err();
        assert!(error.to_string().contains("exceeds metadata size limit"));
        assert!(read_json(&path, MAX_MANIFEST_BYTES)
            .unwrap_err().to_string().contains("exceeds metadata size limit"));
        Ok(())
    }

    #[test]
    fn publication_without_pack_hint_uses_native_header_stems_and_storage() -> Result<()> {
        let (_, mut manifest, mut shape, headers) = deepseek_v4_publication();
        shape.draft_stages = 0;
        shape.draft_experts = 0;
        manifest.as_object_mut().unwrap().remove("pack_dtype");
        let config = serde_json::json!({"quantization_config": {
            "bits": 2.0, "checkpoint_format": "exl3", "format": "exl3", "quant_method": "exl3"}});
        let rename = |name: &str| name.replacen("model.layers", "language_model.model.layers", 1)
            .replace(".mlp.experts.", ".block_sparse_moe.experts.")
            .replace(".gate_proj", ".w1").replace(".up_proj", ".w3").replace(".down_proj", ".w2");
        let headers = headers.into_iter().filter(|t| t.name.starts_with("model.layers.")).map(|mut t| {
            t.name = rename(&t.name); t
        }).collect::<Vec<_>>();
        let storage = manifest["tensor_storage"].as_object().unwrap().iter()
            .filter(|(name, _)| name.starts_with("model.layers.")).map(|(name, entry)| {
                let mut entry = entry.clone();
                entry["stored_tensors"] = Value::Object(entry["stored_tensors"].as_object().unwrap().iter()
                    .map(|(name, tensor)| (rename(name), tensor.clone())).collect());
                (rename(name), entry)
            }).collect();
        manifest["tensor_storage"] = Value::Object(storage);
        manifest["module_include"] = serde_json::json!([
            r"language_model\.model\.layers\.[0-9]+\.block_sparse_moe\.experts\.[0-9]+\.w[123]$"]);
        let parsed = read_publication(&config, &[("quantize_config.json", &manifest)], shape, &headers, &[2, 3])?;
        assert_eq!(parsed.naming, V41Exl3Naming::HeaderDerived {
            prefix: "language_model.model.layers".into(), module: "block_sparse_moe".into(),
            stems: ["w1", "w3", "w2"].map(str::to_owned) });
        for (stem, kind) in [("w1", V41Exl3ProjectionKind::Gate), ("w3", V41Exl3ProjectionKind::Up),
            ("w2", V41Exl3ProjectionKind::Down)] {
            let name = format!("language_model.model.layers.1.block_sparse_moe.experts.2.{stem}");
            assert_eq!(parsed.projections[&name].kind, kind);
        }
        let plan = parsed.residency(crate::V41Exl3Layer::Backbone(1), 2, 1)?;
        assert_eq!(plan.loads[0].tensor, "language_model.model.layers.1.block_sparse_moe.experts.0.w1.trellis");
        for hint in [Value::from("int16"), Value::Null] {
            let mut wrong = manifest.clone();
            wrong["pack_dtype"] = hint;
            assert!(read_publication(&config, &[("quantize_config.json", &wrong)], shape, &headers, &[]).is_err());
        }
        let mut wrong_headers = headers.clone();
        let trellis = wrong_headers.iter_mut().find(|t| t.name.ends_with(".trellis")).unwrap();
        trellis.dtype = DType::I32;
        trellis.byte_length *= 2;
        let error = format!("{:#}", read_publication(&config, &[("quantize_config.json", &manifest)],
            shape, &wrong_headers, &[]).unwrap_err());
        assert!(error.contains("expected i16"), "{error}");
        let mut mixed = headers;
        for tensor in &mut mixed {
            tensor.name = tensor.name.replace(".experts.0.w3.", ".experts.0.up_proj.");
        }
        let error = format!("{:#}", read_publication(&config, &[], shape, &mixed, &[]).unwrap_err());
        assert!(error.contains("unsupported or mixed projection stems"), "{error}");
        Ok(())
    }

    #[test]
    fn arbitrary_header_wrappers_and_expert_modules_drive_residency() -> Result<()> {
        let (_, _, shape, headers) = deepseek_v4_publication();
        let config = serde_json::json!({"quantization_config": {"quant_method": "exl3", "version": "0.0.43",
            "bits": 2, "head_bits": 16, "codebook": "mcg", "scope": "routed_experts_only"}});
        for prefix in ["language_model.model.layers", "custom.wrapper.decoder.layers"] {
            let headers = headers.iter().filter(|tensor| tensor.name.starts_with("model.layers.")).cloned()
                .map(|mut tensor| {
                    tensor.name = tensor.name.replacen("model.layers", prefix, 1)
                        .replace(".mlp.experts.", ".block_sparse_moe.experts.");
                    tensor
                }).collect::<Vec<_>>();
            let parsed = read_publication(&config, &[], shape, &headers, &[2, 3])?;
            assert_eq!(parsed.naming, V41Exl3Naming::HeaderDerived {
                prefix: prefix.into(), module: "block_sparse_moe".into(),
                stems: ["gate_proj", "up_proj", "down_proj"].map(str::to_owned) });
            assert_eq!(parsed.experts.draft_stages, 0);
            assert_eq!(parsed.projections.len(), 2 * 3 * 3);
            let plan = parsed.residency(crate::V41Exl3Layer::Backbone(1), 2, 1)?;
            assert_eq!(plan.loads[0].tensor,
                format!("{prefix}.1.block_sparse_moe.experts.0.gate_proj.trellis"));
            let mut mixed = headers;
            mixed[0].name = mixed[0].name.replacen(prefix, "different.layers", 1);
            assert!(read_publication(&config, &[], shape, &mixed, &[2, 3]).is_err());
        }
        Ok(())
    }

    #[test]
    fn mixed_core_k6_and_routed_k2_preserve_catalog_and_side_map_checks() -> Result<()> {
        for kimi in [false, true] {
            let (config, mut manifest, shape, mut headers) = deepseek_v4_publication();
            let rename = |name: &str| if kimi {
                name.replace("model.layers.", "language_model.model.layers.")
                    .replace(".mlp.experts.", ".block_sparse_moe.experts.")
            } else { name.into() };
            if kimi {
                headers.retain(|tensor| !tensor.name.starts_with("mtp."));
                for tensor in &mut headers { tensor.name = rename(&tensor.name); }
                let entries = manifest["tensor_storage"].as_object().unwrap().iter()
                    .filter(|(name, _)| !name.starts_with("mtp.")).map(|(name, entry)| {
                    let mut entry = entry.clone();
                    entry["stored_tensors"] = serde_json::Value::Object(entry["stored_tensors"].as_object().unwrap()
                        .iter().map(|(name, tensor)| (rename(name), tensor.clone())).collect());
                    (rename(name), entry)
                }).collect();
                manifest["tensor_storage"] = serde_json::Value::Object(entries);
            }
            let core = if kimi { "language_model.model.layers.0.self_attn.q_proj" }
                else { "model.layers.0.self_attn.q_a_proj" };
            let mut stored = serde_json::Map::new();
            for (suffix, dtype, label, dims, width) in [
                ("trellis", DType::I16, "int16", vec![16, 24, 96], 2),
                ("suh", DType::F16, "float16", vec![256], 2),
                ("svh", DType::F16, "float16", vec![384], 2),
                ("mcg", DType::I32, "int32", vec![], 4),
            ] {
                let name = format!("{core}.{suffix}");
                stored.insert(name.clone(), serde_json::json!({"torch_dtype":label,"shape":dims}));
                headers.push(SafetensorsTensorMetadata { name, dtype, byte_offset: 0,
                    byte_length: width * dims.iter().product::<usize>() as u64, shape: dims });
            }
            manifest["tensor_storage"][core] = serde_json::json!({"bits_per_weight":6,
                "mcg_multiplier":3417055213u64,"quant_format":"exl3","stored_tensors":stored});
            let catalog = read_publication(&config, &[("quantize_config.json", &manifest)], shape, &headers, &[2,3])?;
            assert_eq!(catalog.projections.len(), if kimi { 18 } else { 27 });
            assert!(!catalog.projections.contains_key(core));
            assert_eq!(catalog.decoder_tiers(), &[2,3]);
            let mut mismatch = manifest.clone();
            mismatch["tensor_storage"][core]["bits_per_weight"] = serde_json::json!(5);
            let error = read_publication(&config, &[("quantize_config.json", &mismatch)], shape, &headers, &[]).unwrap_err();
            assert!(format!("{error:#}").contains(core));
            let mut incomplete = headers.clone();
            incomplete.retain(|tensor| tensor.name != format!("{core}.svh"));
            assert!(read_publication(&config, &[("quantize_config.json", &manifest)], shape, &incomplete, &[]).is_err());
        }
        Ok(())
    }

    #[test]
    fn glm_mixed_core_head_and_mtp_catalog_needs_no_side_file() -> Result<()> {
        let (mut config, mut manifest, shape, mut headers) = deepseek_v4_publication_with(|_| 4, &[]);
        manifest = serde_json::from_str(&serde_json::to_string(&manifest)?
            .replace("mtp.0.", "model.layers.2."))?;
        for header in &mut headers {
            if let Some(tail) = header.name.strip_prefix("mtp.0.") {
                header.name = format!("model.layers.2.{tail}");
            }
        }
        let compact = serde_json::json!({"quant_method":"exl3", "bits":4.0, "core_bits":6.0,
            "head_bits":6.0, "mtp_bits":4.0, "checkpoint_format":"exl3", "format":"exl3",
            "per_module_metadata":"quantize_config.json:tensor_storage"});
        config["quantization_config"] = compact.clone();
        manifest.as_object_mut().unwrap().extend(compact.as_object().unwrap().clone());
        manifest["lm_head"] = Value::from(true);
        manifest["sym"] = Value::from(true);
        manifest["calibration"] = serde_json::json!({"samples":1426,"tokens":1097737});
        for (stem, bits, n, k) in [("model.layers.0.self_attn.q_a_proj",6,256,256),
            ("model.layers.0.self_attn.kv_a_proj_with_mqa",6,640,256),
            ("model.layers.2.self_attn.q_a_proj",4,256,256), ("model.layers.2.eh_proj",4,256,512),
            ("lm_head",6,512,256)] {
            let mut stored = serde_json::Map::new();
            for (suffix,dtype,label,dims,width) in [
                ("trellis",DType::I16,"int16",vec![k/16,n/16,16*bits],2),
                ("suh",DType::F16,"float16",vec![k],2),
                ("svh",DType::F16,"float16",vec![n],2),
                ("mcg",DType::I32,"int32",vec![1],4)] {
                let name=format!("{stem}.{suffix}");
                stored.insert(name.clone(),serde_json::json!({"torch_dtype":label,"shape":dims}));
                headers.push(SafetensorsTensorMetadata {name,dtype,byte_offset:0,
                    byte_length:width*dims.iter().product::<usize>() as u64,shape:dims});
            }
            manifest["tensor_storage"][stem]=serde_json::json!({"bits_per_weight":bits,
                "mcg_multiplier":3417055213u64,"quant_format":"exl3","stored_tensors":stored});
        }
        let mapped=read_publication(&config,&[("quantize_config.json",&manifest)],shape,&headers,&[3,4])?;
        let native=read_publication(&config,&[],shape,&headers,&[3,4])?;
        assert_eq!(mapped.projections,native.projections);
        assert_eq!(mapped.experts,native.experts);
        assert_eq!(native.storage,Exl3StorageSource::Headers);
        assert_eq!(native.naming, V41Exl3Naming::HfMlp);
        assert_eq!((native.experts.draft_stages, native.experts.draft_experts), (0, 0));
        assert!(native.residency(crate::V41Exl3Layer::Dspark(0), 1, 0).is_err());
        assert!(!native.projections.keys().any(|name| name.starts_with("model.layers.2.")));
        assert_eq!(native.projections.len(),18);
        assert!(native.projections.values().all(|p|p.bits==4));
        let mut wrong=manifest.clone();
        wrong["tensor_storage"]["lm_head"]["bits_per_weight"]=Value::from(4);
        assert!(read_publication(&config,&[("quantize_config.json",&wrong)],shape,&headers,&[]).is_err());
        let mut wrong=config;
        wrong["quantization_config"]["mtp_bits"]=Value::from(6);
        let error=read_publication(&wrong,&[],shape,&headers,&[]).unwrap_err();
        assert!(format!("{error:#}").contains("mtp_bits=6"));
        Ok(())
    }

    #[test]
    fn routed_mul1_is_rejected_after_shared_header_derivation() {
        let (config, _, shape, mut headers) = deepseek_v4_publication();
        let marker = headers.iter_mut().find(|tensor| tensor.name.ends_with(".mcg")).unwrap();
        let stem = marker.name.strip_suffix(".mcg").unwrap().to_owned();
        marker.name = format!("{stem}.mul1");
        let error = read_publication(&config, &[], shape, &headers, &[]).unwrap_err();
        let text = format!("{error:#}");
        assert!(text.contains(&stem) && text.contains("routed expert kernels require MCG"), "{text}");
    }

    #[test]
    fn deepseek_v4_publication_reads_hf_named_experts_and_drafts() {
        let (config, manifest, shape, headers) = deepseek_v4_publication();
        let parsed = read_publication(&config, &[("quantize_config.json", &manifest)], shape, &headers, &[]).unwrap();
        assert_eq!(parsed.storage, Exl3StorageSource::GptqModel);
        assert_eq!(parsed.naming, V41Exl3Naming::HfMlp);
        assert_eq!(parsed.experts.draft_stages, 1);
        assert_eq!(parsed.experts.draft_experts, 3);
        assert_eq!(parsed.projections.len(), (2 + 1) * 3 * 3);
        assert_eq!(parsed.decoder_tiers(), &[2, 3]);
        let down = &parsed.projections["model.layers.1.mlp.experts.2.down_proj"];
        assert_eq!(down.trellis_shape(), [24, 16, 32]);
        assert_eq!(down.intermediate_partition(2, 1).unwrap(), 256..384);
        let plan = parsed
            .residency(crate::V41Exl3Layer::Backbone(1), 2, 0)
            .unwrap();
        assert_eq!((plan.experts, plan.intermediate, plan.intermediate_start), (3, 256, 0));
        assert_eq!(plan.loads[0].tensor, "model.layers.1.mlp.experts.0.gate_proj.trellis");
        assert!(parsed.residency(crate::V41Exl3Layer::Backbone(2), 2, 0).is_err());
        let draft = parsed.residency(crate::V41Exl3Layer::Dspark(0), 1, 0).unwrap();
        assert_eq!(draft.loads[0].tensor, "mtp.0.mlp.experts.0.gate_proj.trellis");
    }

    /// Standard exllamav3 output: the compact block alone, `[1]` markers, and
    /// the same manifest as the GPTQModel publication of the same tensors.
    #[test]
    fn standard_exllamav3_checkpoint_derives_its_storage_map() {
        let (gptq_config, gptq_map, shape, scalar) = deepseek_v4_publication();
        let (_, _, _, vector) = deepseek_v4_publication_with(|_| 2, &[1]);
        let config = serde_json::json!({"quantization_config": {"quant_method": "exl3", "version": "0.0.43",
            "bits": 2, "head_bits": 16, "codebook": "mcg", "scope": "routed_experts_only"}});
        let standard = read_publication(&config, &[], shape, &vector, &[]).unwrap();
        let mapped = read_publication(&gptq_config, &[("quantize_config.json", &gptq_map)], shape, &scalar, &[]).unwrap();
        assert_eq!(standard.storage, Exl3StorageSource::Headers);
        assert_eq!(standard.projections, mapped.projections);
        assert_eq!((standard.experts, &standard.naming), (mapped.experts, &mapped.naming));
        assert_eq!(standard.decoder_tiers(), mapped.decoder_tiers());
        let mcg = vector.iter().find(|t| t.name.ends_with(".mcg")).unwrap();
        standard.projections[mcg.name.rsplit_once('.').unwrap().0].validate_tensor(mcg).unwrap();
        // exllamav3's own map (config block + tensor_storage) agrees too.
        let mut exllamav3 = config["quantization_config"].clone();
        let mut storage = gptq_map["tensor_storage"].clone();
        for entry in storage.as_object_mut().unwrap().values_mut() {
            for tensor in entry["stored_tensors"].as_object_mut().unwrap().values_mut() {
                let dtype = format!("torch.{}", tensor["torch_dtype"].as_str().unwrap());
                *tensor = serde_json::json!({"dtype": dtype, "shape": tensor["shape"]});
            }
            for (name, tensor) in entry["stored_tensors"].as_object_mut().unwrap() {
                if name.ends_with(".mcg") {
                    tensor["shape"] = serde_json::json!([1]);
                }
            }
        }
        exllamav3["tensor_storage"] = storage;
        let read = read_publication(&config, &[("quantization_config.json", &exllamav3)], shape, &vector, &[]).unwrap();
        assert_eq!((read.storage, read.projections.len()), (Exl3StorageSource::Exllamav3, 27));
        // ... but not when its block drifts from config.json.
        let mut drift = exllamav3.clone();
        drift["bits"] = serde_json::json!(3);
        let error = read_publication(&config, &[("quantization_config.json", &drift)], shape, &vector, &[]);
        assert!(format!("{:#}", error.unwrap_err()).contains("disagree at bits"));
        // Uniform K2 runs on the K2/K3 family; a family preference that
        // covers it (an empty upper or lower tier) wins.
        assert_eq!(read_publication(&config, &[], shape, &vector, &[2, 3]).unwrap().decoder_tiers(), &[2, 3]);
        let (_, _, _, k3) = deepseek_v4_publication_with(|_| 3, &[1]);
        assert_eq!(read_publication(&config, &[], shape, &k3, &[]).unwrap().decoder_tiers(), &[3, 4]);
        assert_eq!(read_publication(&config, &[], shape, &k3, &[2, 3]).unwrap().decoder_tiers(), &[2, 3]);
        let (_, _, _, mixed) = deepseek_v4_publication_with(|name| if name.contains("down") { 4 } else { 3 }, &[1]);
        assert_eq!(read_publication(&config, &[], shape, &mixed, &[2, 3]).unwrap().decoder_tiers(), &[3, 4]);
    }

    /// A storage map that disagrees with the headers is an error naming the
    /// module, whichever side changed.
    #[test]
    fn storage_map_disagreeing_with_headers_names_the_module() {
        let (config, manifest, shape, headers) = deepseek_v4_publication();
        let maps = [("quantize_config.json", &manifest)];
        // The checkpoint's markers are [1]; the map says [].
        let (_, _, _, vector) = deepseek_v4_publication_with(|_| 2, &[1]);
        let error = format!("{:#}", read_publication(&config, &maps, shape, &vector, &[]).unwrap_err());
        assert!(error.contains("quantize_config.json disagrees with the safetensors headers at model.layers.0.mlp.experts.0.")
            && error.contains("mcg i32 []") && error.contains("mcg i32 [1]"), "{error}");
        // One projection quantized at K3 in the checkpoint, K2 in the map.
        let target = "model.layers.1.mlp.experts.2.up_proj";
        let (_, _, _, k3) = deepseek_v4_publication_with(|name| if name == target { 3 } else { 2 }, &[]);
        let error = format!("{:#}", read_publication(&config, &maps, shape, &k3, &[]).unwrap_err());
        assert!(error.contains(&format!("at {target}: map K2")) && error.contains("headers K3"), "{error}");
        // A module the map lacks, and one the checkpoint lacks.
        let mut short = manifest.clone();
        short["tensor_storage"].as_object_mut().unwrap().remove(target);
        let error = format!("{:#}", read_publication(&config, &[("quantize_config.json", &short)], shape, &headers, &[])
            .unwrap_err());
        assert!(error.contains(target) && error.contains("does not list"), "{error}");
        let trimmed: Vec<_> = headers.iter().filter(|t| !t.name.starts_with(&format!("{target}."))).cloned().collect();
        let error = format!("{:#}", read_publication(&config, &maps, shape, &trimmed, &[]).unwrap_err());
        assert!(error.contains(target) && error.contains("no such EXL3 tensors"), "{error}");
        // Without a map the same trimmed checkpoint names its missing projection.
        let standard = serde_json::json!({"quantization_config": {"quant_method": "exl3", "bits": 2}});
        let error = format!("{:#}", read_publication(&standard, &[], shape, &trimmed, &[]).unwrap_err());
        assert!(error.contains(&format!("missing projection {target}")), "{error}");
        // A copy beside quantize_config.json must agree as well.
        let mut copy = manifest.clone();
        copy["tensor_storage"][target]["bits_per_weight"] = serde_json::json!(3);
        let error = format!("{:#}", read_publication(&config,
            &[("quantize_config.json", &manifest), ("quantization_config.json", &copy)], shape, &headers, &[])
            .unwrap_err());
        assert!(error.contains("quantize_config.json and quantization_config.json differ in SHA256"), "{error}");
    }

    #[test]
    fn stitched_native_core_metadata_preserves_exl3_contract() {
        let (mut config, mut manifest, shape, headers) = deepseek_v4_publication();
        let native = serde_json::json!({"quant_method": "fp8", "fmt": "e4m3",
            "activation_scheme": "dynamic", "weight_block_size": [128, 128],
            "modules": ["model.layers.0.self_attn.q_a_proj"]});
        let meta = serde_json::json!({"native_core_quantization": native});
        config["quantization_config"]["meta"] = meta.clone();
        manifest["meta"] = meta;
        let parsed = read_publication(&config, &[("quantize_config.json", &manifest),
            ("quantization_config.json", &manifest)], shape, &headers, &[]).unwrap();
        assert_eq!(parsed.projections.len(), 27);
        assert_eq!(parsed.storage, Exl3StorageSource::GptqModel);
        let mut drift = manifest.clone();
        drift["meta"]["native_core_quantization"]["fmt"] = serde_json::json!("e5m2");
        let error = read_publication(&config, &[("quantize_config.json", &drift)], shape, &headers, &[])
            .unwrap_err();
        assert!(format!("{error:#}").contains("disagree at meta"));
        // Native format declarations are metadata, not new EXL3 storage keys.
        manifest["native_core_quantization"] = native;
        let error = read_publication(&config, &[("quantize_config.json", &manifest)], shape, &headers, &[])
            .unwrap_err();
        assert!(format!("{error:#}").contains("unexpected key native_core_quantization"));
    }

    #[test]
    fn deepseek_v4_publication_rejects_contract_drift() {
        let (config, manifest, shape, headers) = deepseek_v4_publication();
        for (pointer, value, expected) in [
            ("/out_scales", serde_json::json!("always"), "storage contract"),
            ("/bits", serde_json::json!(2.5), "integer"),
            ("/surprise", serde_json::json!(1), "unexpected key"),
            (
                "/tensor_storage/model.layers.0.mlp.experts.0.up_proj/mcg_multiplier",
                serde_json::json!(1),
                "MCG multiplier",
            ),
        ] {
            let mut manifest = manifest.clone();
            match manifest.pointer_mut(pointer) {
                Some(slot) => *slot = value,
                None => {
                    manifest[pointer.trim_start_matches('/')] = value;
                }
            }
            let error = read_publication(&config, &[("quantize_config.json", &manifest)], shape, &headers, &[]);
            let error = format!("{:#}", error.unwrap_err());
            assert!(error.contains(expected), "unexpected error: {error}");
        }
        let mut compact = config.clone();
        compact["quantization_config"]["bits"] = serde_json::json!(3.0);
        let error = read_publication(&compact, &[("quantize_config.json", &manifest)], shape, &headers, &[]);
        assert!(format!("{:#}", error.unwrap_err()).contains("disagree"));
        // Unsupported variants stay unsupported without a map, by name.
        let standard = |quant: Value| serde_json::json!({"quantization_config": quant});
        let error = read_publication(&standard(serde_json::json!({"quant_method": "exl3", "bits": 2, "codebook": "mul1"})),
            &[], shape, &headers, &[]).unwrap_err();
        assert!(format!("{error:#}").contains("codebook=\"mul1\""), "{error:#}");
        let mut three_inst = headers.clone();
        three_inst.retain(|t| t.name != "mtp.0.mlp.experts.1.down_proj.mcg");
        let error = read_publication(&standard(serde_json::json!({"quant_method": "exl3", "bits": 2})),
            &[], shape, &three_inst, &[]).unwrap_err();
        assert!(format!("{error:#}").contains("mtp.0.mlp.experts.1.down_proj.trellis: EXL3 3INST codebook"), "{error:#}");
    }

    #[test]
    #[ignore = "requires CUTEAFD_DSV4_EXL3_SNAPSHOT pointing to a DeepSeek V4 EXL3 publication"]
    fn deepseek_v4_publication_checkpoint_catalog() {
        let path = std::env::var_os("CUTEAFD_DSV4_EXL3_SNAPSHOT").expect("CUTEAFD_DSV4_EXL3_SNAPSHOT");
        let catalog = crate::read_expert_catalog(Path::new(&path)).unwrap();
        let manifest = catalog.exl3().unwrap();
        let shape = catalog.routed_experts();
        assert_eq!(manifest.decoder_tiers(), &[2, 3]);
        assert_eq!(
            manifest.projections.len(),
            3 * (shape.layers * shape.experts + shape.draft_stages * shape.draft_experts)
        );
        println!(
            "{} projections, {} layers x {} experts + {} draft stages; geometry {:?}",
            manifest.projections.len(),
            shape.layers,
            shape.experts,
            shape.draft_stages,
            shape.geometry().unwrap()
        );
    }

    #[test]
    fn pro_intermediate_splits_six_ways_on_whole_blocks() {
        for kind in [V41Exl3ProjectionKind::Gate, V41Exl3ProjectionKind::Down] {
            let (input, output) = if kind == V41Exl3ProjectionKind::Down { (3072, 7168) } else { (7168, 3072) };
            let p = V41Exl3Projection {
                name: "test".into(),
                kind,
                bits: 3,
                input_features: input,
                output_features: output,
            };
            for rank in 0..6 {
                assert_eq!(p.intermediate_partition(6, rank).unwrap(), rank * 512..(rank + 1) * 512);
            }
            assert!(p.intermediate_partition(6, 6).is_err());
            assert!(p.intermediate_partition(9, 0).is_err());
        }
    }

    #[test]
    fn retained_transport_worlds_partition_nonempty_h128_blocks_exactly() {
        for intermediate in [640, 2048, 2304, 3072] {
            for kind in [V41Exl3ProjectionKind::Gate, V41Exl3ProjectionKind::Down] {
                let (input_features, output_features) = if kind == V41Exl3ProjectionKind::Down {
                    (intermediate, 5120)
                } else {
                    (5120, intermediate)
                };
                let p = V41Exl3Projection { name: "test".into(), kind, bits: 4,
                    input_features, output_features };
                for world in [1, 5] {
                    if intermediate / 128 < world {
                        assert!(p.intermediate_partition(world, 0).is_err());
                        continue;
                    }
                    let mut cursor = 0;
                    let mut widths = Vec::new();
                    for rank in 0..world {
                        let range = p.intermediate_partition(world, rank).unwrap();
                        assert_eq!(range.start, cursor);
                        assert_eq!(range.start % 128, 0);
                        assert_eq!(range.end % 128, 0);
                        assert!(!range.is_empty());
                        widths.push(range.len());
                        cursor = range.end;
                    }
                    assert_eq!(cursor, intermediate);
                    assert!(widths.iter().max().unwrap() - widths.iter().min().unwrap() <= 128);
                    assert!(p.intermediate_partition(world, world).is_err());
                }
            }
        }
    }

    #[test]
    fn intermediate_2048_splits_six_ways_on_uneven_whole_blocks() {
        // GLM 5.3 / GLM 5.3 Flash / MiMo: 16 H128 blocks -> 3, 3, 3, 3, 2, 2.
        for kind in [V41Exl3ProjectionKind::Up, V41Exl3ProjectionKind::Down] {
            let (input, output) = if kind == V41Exl3ProjectionKind::Down { (2048, 6144) } else { (6144, 2048) };
            let p = V41Exl3Projection { name: "test".into(), kind, bits: 4, input_features: input, output_features: output };
            let mut cursor = 0;
            for (rank, width) in [384, 384, 384, 384, 256, 256].into_iter().enumerate() {
                assert_eq!(p.intermediate_partition(6, rank).unwrap(), cursor..cursor + width);
                cursor += width;
            }
            assert_eq!(cursor, 2048);
        }
    }

    #[test]
    fn aligned_tp_slices_cover_the_complete_rotation_axis() {
        for kind in [
            V41Exl3ProjectionKind::Gate,
            V41Exl3ProjectionKind::Up,
            V41Exl3ProjectionKind::Down,
        ] {
            let (input, output) = if kind == V41Exl3ProjectionKind::Down {
                (2304, 5120)
            } else {
                (5120, 2304)
            };
            let p = V41Exl3Projection {
                name: "test".into(),
                kind,
                bits: 3,
                input_features: input,
                output_features: output,
            };
            for (world, widths) in [
                (1, vec![2304]),
                (2, vec![1152, 1152]),
                // TP3 divides the 18 H128 blocks exactly, so every rank owns
                // six whole blocks with no padding or duplicated boundary.
                (3, vec![768, 768, 768]),
                (4, vec![640, 640, 512, 512]),
                (5, vec![512, 512, 512, 384, 384]),
                (6, vec![384; 6]),
                (7, vec![384, 384, 384, 384, 256, 256, 256]),
                (8, vec![384, 384, 256, 256, 256, 256, 256, 256]),
            ] {
                let mut cursor = 0;
                for (rank, width) in widths.into_iter().enumerate() {
                    let range = p.intermediate_partition(world, rank).unwrap();
                    assert_eq!(range, cursor..cursor + width);
                    assert_eq!(range.start % 128, 0);
                    cursor = range.end;
                }
                assert_eq!(cursor, 2304);
                assert!(p.intermediate_partition(world, world).is_err());
            }
            // Unadmitted worlds and empty shards fail closed.
            for world in [0usize, 9, 19] {
                assert!(
                    p.intermediate_partition(world, 0).is_err(),
                    "EXL3 world {world} must stay unadmitted"
                );
            }
        }
    }

}

#[cfg(test)]
mod qwen_naming_tests {
    use super::*;

    #[test]
    fn qwen_mtp_experts_live_under_mtp_layers() {
        let naming = V41Exl3Naming::HfLanguageModelMtp;
        assert_eq!(naming.projection(false, 47, 511, V41Exl3ProjectionKind::Down),
            "model.language_model.layers.47.mlp.experts.511.down_proj");
        assert_eq!(naming.projection(true, 0, 3, V41Exl3ProjectionKind::Gate), "mtp.layers.0.mlp.experts.3.gate_proj");
    }

    fn k6_core_storage() -> Exl3StorageMap {
        let module = |bits| Exl3Module { codebook: crate::formats::exl3_storage::Exl3Codebook::Mcg, bits,
            input_features: 128, output_features: 128, tensors: BTreeMap::new() };
        BTreeMap::from([
            ("language_model.model.layers.1.block_sparse_moe.experts.0.w1".to_owned(), module(2)),
            ("language_model.model.layers.1.self_attn.q_proj".to_owned(), module(6)),
            ("language_model.model.layers.1.block_sparse_moe.routed_expert_up_proj".to_owned(), module(6)),
            ("language_model.lm_head".to_owned(), module(6)),
        ])
    }

    /// The published Kimi K3 Core-K6 key set: compact `config.json` block and
    /// the top-level `quantize_config.json` keys (tensor_storage elided).
    fn k6_publication() -> (Value, Value) {
        let compact = serde_json::json!({"bits": 2.0, "checkpoint_format": "exl3", "core_bits": 6.0, "format": "exl3",
            "per_module_metadata": "quantize_config.json:tensor_storage", "quant_method": "exl3"});
        let manifest = serde_json::json!({"bits": 2.0, "checkpoint_format": "exl3", "codebook": "mcg",
            "core_bits": 6.0, "desc_act": false, "format": "exl3", "group_size": -1, "method": "exl3",
            "module_include": ["^language_model\\.lm_head$"], "out_scales": "auto", "quant_method": "exl3",
            "meta": {"core_quantization_provenance": {}, "quantization_provenance": {}},
            "tensor_storage": {}});
        (compact, manifest)
    }

    #[test]
    fn k6_core_bits_is_checked_against_core_headers() {
        let (compact, manifest) = k6_publication();
        let storage = k6_core_storage();
        check_gptqmodel_manifest(&compact, &manifest, &storage, usize::MAX).unwrap();
        // A core module at another width disagrees with the declared core_bits.
        let mut wrong = storage.clone();
        wrong.get_mut("language_model.lm_head").unwrap().bits = 4;
        let error = format!("{:#}", check_gptqmodel_manifest(&compact, &manifest, &wrong, usize::MAX).unwrap_err());
        assert!(error.contains("core_bits=6") && error.contains("language_model.lm_head"), "{error}");
        // core_bits with only routed experts in the headers is not a core quant.
        let routed: Exl3StorageMap = storage.iter().filter(|(name, _)| is_routed_module(name))
            .map(|(name, module)| (name.clone(), module.clone())).collect();
        assert!(check_gptqmodel_manifest(&compact, &manifest, &routed, usize::MAX).is_err());
        for bad in [Value::from(6.5), Value::from(0), Value::from("6")] {
            let mut bad_manifest = manifest.clone();
            bad_manifest["core_bits"] = bad.clone();
            let mut bad_compact = compact.clone();
            bad_compact["core_bits"] = bad;
            assert!(check_gptqmodel_manifest(&bad_compact, &bad_manifest, &storage, usize::MAX).is_err());
        }
        // Compact and full metadata still have to agree on core_bits.
        let mut disagree = compact.clone();
        disagree["core_bits"] = Value::from(5.0);
        assert!(check_gptqmodel_manifest(&disagree, &manifest, &storage, usize::MAX).is_err());
        // The pointer may name only the GPTQModel storage map.
        let mut pointer = compact.clone();
        pointer["per_module_metadata"] = Value::from("other.json:tensor_storage");
        assert!(check_gptqmodel_manifest(&pointer, &manifest, &storage, usize::MAX).is_err());
        // Unknown keys are still refused.
        let mut extra = manifest.clone();
        extra["head_core_bits"] = Value::from(6.0);
        let error = format!("{:#}", check_gptqmodel_manifest(&compact, &extra, &storage, usize::MAX).unwrap_err());
        assert!(error.contains("unexpected key head_core_bits"), "{error}");
    }
}
