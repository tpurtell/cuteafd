//! Validated routed-only EXL3 metadata for the native V4.1 engine.
use crate::formats::exl3_storage::{
    check_quantization_config, derive_storage_map, mcg_marker_shape, parse_storage_map, verify_storage_map,
    Exl3Module, Exl3StorageMap, Exl3StorageSource,
};
use crate::{OfficialV41Config, SafetensorsTensorMetadata, OFFICIAL_V41_MODEL_ID};
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

pub const V41_EXL3_SCHEMA: &str = "cuteafd.v41-routed-exl3.v1";
const MAX_MANIFEST_BYTES: u64 = 64 * 1024 * 1024;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum V41Exl3ProjectionKind {
    Gate,
    Up,
    Down,
}

/// Explicit resident layout; paired storage requires ownership-aware execution.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum V41Exl3Partition {
    Disjoint,
    PairedTp4,
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
        self.intermediate_partition_with_layout(world, rank, V41Exl3Partition::Disjoint)
    }

    pub fn intermediate_partition_with_layout(
        &self,
        world: usize,
        rank: usize,
        layout: V41Exl3Partition,
    ) -> Result<Range<usize>> {
        ensure!(
            (1..=8).contains(&world) && rank < world,
            "invalid EXL3 TP rank/world"
        );
        let intermediate = match self.kind {
            V41Exl3ProjectionKind::Down => self.input_features,
            _ => self.output_features,
        };
        // Six ranks are the implicit EXL3 TP6 split of every non-V4.1 geometry:
        // even where the H128 blocks divide (V4 Pro: 24 -> 4 apiece), else the
        // first ranks own the extra blocks (2048: 3, 3, 3, 3, 2, 2), each rank
        // with the export of its own width. V4.1's 2304 keeps its six-rank
        // Spark layouts native.
        ensure!(
            world != 6 || intermediate != 2304,
            "EXL3 six-rank partition is not used for V4.1"
        );
        ensure!(
            intermediate % 128 == 0,
            "EXL3 intermediate axis is not H128 aligned"
        );
        if layout == V41Exl3Partition::PairedTp4 {
            ensure!(
                world == 4 && intermediate == 2304,
                "paired EXL3 layout requires TP4 with 2304 intermediate channels"
            );
            let blocks = &cuteafd_core::EXL3_TP4_RESIDENT_BLOCKS[rank];
            return Ok(blocks.start * 128..blocks.end * 128);
        }
        let blocks = intermediate / 128;
        ensure!(blocks >= world, "EXL3 partition would be empty");
        let count = blocks / world + usize::from(rank < blocks % world);
        let start = rank * (blocks / world) + rank.min(blocks % world);
        Ok(start * 128..(start + count) * 128)
    }
}

/// How the checkpoint stores its MTP (dSpark draft) routed experts.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum V41Exl3MtpExperts {
    /// CUTEAFD-staged snapshots quantize draft experts to EXL3 as well.
    Exl3,
    /// Raw publications keep draft experts at the official native FP4
    /// (MXFP4) source precision; the draft path must load native weights.
    Source,
}

/// How a checkpoint names its routed EXL3 projections.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
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
}

impl V41Exl3Naming {
    /// The projection name (without the `.trellis`/`.suh`/... suffix).
    pub fn projection(self, draft: bool, layer: usize, expert: usize, kind: V41Exl3ProjectionKind) -> String {
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
        }
    }
}

#[derive(Debug)]
pub struct V41Exl3Manifest {
    /// Validated original non-routed model geometry and quantization contract;
    /// `None` for other families, whose catalogs carry only routed experts.
    pub config: Option<OfficialV41Config>,
    /// Routed-expert extents of every target and draft layer.
    pub experts: crate::RoutedExpertShape,
    pub naming: V41Exl3Naming,
    pub projections: BTreeMap<String, V41Exl3Projection>,
    /// One checkpoint-wide family, shared by all target and draft layers.
    pub(crate) decoder_tiers: Vec<usize>,
    /// Retained for validation by the PLE storage path; never silently discarded.
    pub ple_quantization: Option<Value>,
    pub(crate) mtp_experts: V41Exl3MtpExperts,
    /// Where the storage layout was read from (always checked against the headers).
    pub storage: Exl3StorageSource,
}

impl V41Exl3Manifest {
    fn v41(
        config: OfficialV41Config,
        projections: BTreeMap<String, V41Exl3Projection>,
        ple_quantization: Option<Value>,
        mtp_experts: V41Exl3MtpExperts,
    ) -> Result<Self> {
        Ok(Self {
            experts: crate::RoutedExpertShape::of_v41(&config),
            config: Some(config),
            naming: V41Exl3Naming::CheckpointNative,
            decoder_tiers: decoder_family(&projections)?,
            projections,
            ple_quantization,
            // V4.1 publications either carry a GPTQModel map (EXL3 drafts) or
            // are raw (source drafts); the catalog checks every header.
            storage: match mtp_experts {
                V41Exl3MtpExperts::Exl3 => Exl3StorageSource::GptqModel,
                V41Exl3MtpExperts::Source => Exl3StorageSource::Headers,
            },
            mtp_experts,
        })
    }

    pub fn decoder_tiers(&self) -> &[usize] {
        &self.decoder_tiers
    }

    /// True when draft experts stayed at the native FP4 source precision.
    pub fn mtp_experts_are_source(&self) -> bool {
        self.mtp_experts == V41Exl3MtpExperts::Source
    }
}

pub(crate) fn decoder_family(
    projections: &BTreeMap<String, V41Exl3Projection>,
) -> Result<Vec<usize>> {
    let mut bits: BTreeSet<_> = projections.values().map(|p| p.bits).collect();
    ensure!(
        !bits.is_empty() && bits.iter().all(|b| (2..=5).contains(b)),
        "EXL3 decoder family must contain K2..K5 projections"
    );
    // Native mixed kernels retain an empty adjacent tier for uniform models.
    if bits.len() == 1 {
        let bit = *bits.first().unwrap();
        bits.insert(if bit == 5 { 4 } else { bit + 1 });
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
        && preferred.iter().all(|bits| (2..=5).contains(bits))
        && projections.values().all(|p| preferred.contains(&p.bits));
    Ok(if covered { preferred.to_vec() } else { family })
}

pub(crate) fn read_json(path: &Path, limit: u64) -> Result<Value> {
    let mut bytes = Vec::new();
    File::open(path)
        .with_context(|| format!("opening {}", path.display()))?
        .take(limit + 1)
        .read_to_end(&mut bytes)?;
    ensure!(
        bytes.len() as u64 <= limit,
        "{} exceeds metadata size limit",
        path.display()
    );
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

pub fn read_v41_exl3_manifest(snapshot: &Path) -> Result<V41Exl3Manifest> {
    let config = read_json(&snapshot.join("config.json"), 1024 * 1024)?;
    let manifest_path = snapshot.join("quantize_config.json");
    if manifest_path.is_file() {
        let manifest = read_json(&manifest_path, MAX_MANIFEST_BYTES)?;
        parse_manifest(config, &manifest)
    } else {
        parse_raw_publication(config)
    }
}

/// First-class support for raw exllamav3-style V4.1 Flash EXL3 publications
/// that ship only a compact `config.json` quantization block (no
/// quantize_config.json staging manifest). The accepted contract is strict:
/// integer K2..K5 uniform routed experts with the MCG codebook, checkpoint-
/// native tensor naming, source-precision MTP draft experts, and the
/// official FP8 block for every non-routed tensor. Tensor payload layouts
/// are verified against the safetensors headers by the catalog, so the
/// config only needs to establish geometry-independent facts.
fn parse_raw_publication(mut config: Value) -> Result<V41Exl3Manifest> {
    let quant = config["quantization_config"]
        .as_object()
        .context("raw EXL3 publication requires a quantization_config object")?;
    ensure!(
        quant.get("quant_method").and_then(Value::as_str) == Some("exl3"),
        "raw EXL3 publication requires quant_method=exl3"
    );
    let bits = quant["bits"]
        .as_u64()
        .filter(|b| (2..=5).contains(b))
        .context("raw EXL3 publication requires integer bits in 2..=5")? as usize;
    ensure!(
        quant.get("codebook").and_then(Value::as_str) == Some("mcg"),
        "raw EXL3 publication requires the MCG codebook"
    );
    ensure!(
        quant.get("mtp_experts").and_then(Value::as_str) == Some("source"),
        "raw EXL3 publication requires mtp_experts=source"
    );
    // Optional exllamav3 storage hints, when present, must match the only
    // layout the engine consumes.
    for (key, wanted) in [
        ("out_scales", Value::from("never")),
        ("group_size", Value::from(-1)),
        ("desc_act", Value::from(false)),
        ("pack_dtype", Value::from("int32")),
    ] {
        if let Some(found) = quant.get(key) {
            ensure!(
                found == &wanted,
                "raw EXL3 publication has unsupported {key}={found}"
            );
        }
    }
    let native = quant
        .get("non_routed_quantization")
        .context("raw EXL3 publication requires non_routed_quantization")?;
    ensure!(
        native["quant_method"] == "deepseek_v4_fp8"
            && native["fmt"] == "e4m3"
            && native["activation_scheme"] == "dynamic"
            && native["scale_fmt"] == "ue8m0"
            && native["weight_block_size"] == serde_json::json!([32, 32])
            && native["expert_dtype"] == "fp4",
        "raw EXL3 publication has an unsupported non-routed FP8 contract"
    );
    // Every contract key must be understood; reject silent drift.
    for key in quant.keys() {
        ensure!(
            matches!(
                key.as_str(),
                "quant_method"
                    | "bits"
                    | "codebook"
                    | "mtp_experts"
                    | "mtp_experts_start_layer"
                    | "weight_block_size"
                    | "non_routed_quantization"
                    | "out_scales"
                    | "group_size"
                    | "desc_act"
                    | "pack_dtype"
            ),
            "raw EXL3 publication has unexpected quantization_config key {key}"
        );
    }
    let mtp_start_layer = quant
        .get("mtp_experts_start_layer")
        .and_then(Value::as_u64);
    // Substitute the canonical official FP8 block so the strict official
    // config validation covers every non-quantization field.
    config["quantization_config"] = serde_json::json!({
        "quant_method": "fp8",
        "activation_scheme": "dynamic",
        "weight_block_size": [32, 32],
        "scale_fmt": "ue8m0",
        "expert_dtype": "fp4",
    });
    let validated =
        OfficialV41Config::from_json(OFFICIAL_V41_MODEL_ID, &serde_json::to_vec(&config)?)?;
    let text = validated.text();
    if let Some(start_layer) = mtp_start_layer {
        ensure!(
            start_layer == text.num_hidden_layers as u64,
            "raw EXL3 mtp_experts_start_layer {start_layer} disagrees with the architecture"
        );
    }
    let mut projections = BTreeMap::new();
    for layer in 0..text.num_hidden_layers {
        for expert in 0..text.n_routed_experts {
            for (stem, kind) in [
                ("w1", V41Exl3ProjectionKind::Gate),
                ("w3", V41Exl3ProjectionKind::Up),
                ("w2", V41Exl3ProjectionKind::Down),
            ] {
                let (input, output) = if kind == V41Exl3ProjectionKind::Down {
                    (text.moe_intermediate_size, text.hidden_size)
                } else {
                    (text.hidden_size, text.moe_intermediate_size)
                };
                let name = format!("layers.{layer}.ffn.experts.{expert}.{stem}");
                projections.insert(
                    name.clone(),
                    V41Exl3Projection {
                        name,
                        kind,
                        bits,
                        input_features: input,
                        output_features: output,
                    },
                );
            }
        }
    }
    V41Exl3Manifest::v41(validated, projections, None, V41Exl3MtpExperts::Source)
}

/// Hugging Face routed-expert modules (`*.mlp.experts.{E}.*` or `*.ffn.experts.{E}.*`).
fn is_routed_module(name: &str) -> bool {
    name.contains(".mlp.experts.") || name.contains(".ffn.experts.")
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
    let source = check_storage_maps(snapshot, &config, &derived)?;
    build_deepseek_v4_manifest(source, &derived, backbone, tiers)
}

/// Which storage map the snapshot carries, checked against `derived`: a
/// GPTQModel `quantize_config.json` (and its `quantization_config.json`
/// copy), an exllamav3 `quantization_config.json`, or none (the compact
/// `config.json` block alone).
fn check_storage_maps(snapshot: &Path, config: &Value, derived: &Exl3StorageMap) -> Result<Exl3StorageSource> {
    let compact = config.get("quantization_config").context("config.json has no quantization_config")?;
    let mut source = None;
    let mut first: Option<Vec<u8>> = None;
    for file in ["quantize_config.json", "quantization_config.json"] {
        let path = snapshot.join(file);
        if !path.is_file() {
            continue;
        }
        let bytes = read_bytes(&path, MAX_MANIFEST_BYTES)?;
        if first.as_ref().is_some_and(|first| *first == bytes) {
            continue;
        }
        let mut manifest: Value =
            serde_json::from_slice(&bytes).with_context(|| format!("parsing {}", path.display()))?;
        normalize_legacy_names(&mut manifest);
        let flavor = if manifest.get("checkpoint_format").is_some() {
            check_gptqmodel_manifest(compact, &manifest).with_context(|| format!("checking {file}"))?;
            Exl3StorageSource::GptqModel
        } else {
            check_exllamav3_manifest(compact, &manifest).with_context(|| format!("checking {file}"))?;
            Exl3StorageSource::Exllamav3
        };
        let declared = parse_storage_map(file, &manifest["tensor_storage"])?;
        verify_storage_map(file, &declared, derived)?;
        source.get_or_insert(flavor);
        first.get_or_insert(bytes);
    }
    match source {
        Some(source) => Ok(source),
        None => {
            check_quantization_config(compact)?;
            Ok(Exl3StorageSource::Headers)
        }
    }
}

fn read_bytes(path: &Path, limit: u64) -> Result<Vec<u8>> {
    let mut bytes = Vec::new();
    File::open(path)
        .with_context(|| format!("opening {}", path.display()))?
        .take(limit + 1)
        .read_to_end(&mut bytes)?;
    ensure!(bytes.len() as u64 <= limit, "{} exceeds metadata size limit", path.display());
    Ok(bytes)
}

/// GPTQModel's storage contract: the compact `config.json` block repeated,
/// the EXL3 layout fields, and nothing else.
fn check_gptqmodel_manifest(compact: &Value, manifest: &Value) -> Result<()> {
    for field in ["quant_method", "method", "format", "checkpoint_format"] {
        ensure!(
            manifest.get(field).and_then(Value::as_str) == Some("exl3"),
            "EXL3 manifest requires {field}=exl3"
        );
    }
    // `out_scales` only decides whether the quantizer folded output-channel
    // scales into `svh`; the kernels always apply `svh` per channel.
    ensure!(
        manifest["codebook"] == "mcg"
            && matches!(manifest["out_scales"].as_str(), Some("never" | "auto"))
            && manifest["group_size"] == -1
            && manifest["desc_act"] == false
            && manifest["pack_dtype"] == "int32"
            && manifest.get("lm_head").is_none_or(|v| v == false),
        "unsupported EXL3 storage contract"
    );
    ensure!(
        manifest["bits"]
            .as_f64()
            .is_some_and(|v| v.fract() == 0.0 && (2.0..=5.0).contains(&v)),
        "EXL3 base bits must be an integer in 2..5"
    );
    let object = manifest.as_object().context("EXL3 manifest must be an object")?;
    for key in object.keys() {
        ensure!(
            matches!(
                key.as_str(),
                "bits" | "checkpoint_format" | "codebook" | "desc_act" | "format" | "group_size"
                    | "lm_head" | "meta" | "method" | "module_include" | "out_scales"
                    | "pack_dtype" | "quant_method" | "tensor_storage"
            ),
            "EXL3 manifest has unexpected key {key}"
        );
    }
    let compact = compact.as_object().context("missing compact EXL3 config")?;
    ensure!(
        compact.get("quant_method") == Some(&Value::from("exl3")),
        "config must declare EXL3 quantization"
    );
    for (key, value) in compact {
        ensure!(
            manifest.get(key) == Some(value),
            "compact/full EXL3 metadata disagree at {key}"
        );
    }
    Ok(())
}

/// exllamav3's `quantization_config.json`: the `config.json` block plus
/// `tensor_storage`, nothing else.
fn check_exllamav3_manifest(compact: &Value, manifest: &Value) -> Result<()> {
    let manifest = manifest.as_object().context("EXL3 storage map must be an object")?;
    let compact_object = compact.as_object().context("missing compact EXL3 config")?;
    for key in compact_object.keys().chain(manifest.keys()) {
        ensure!(
            key == "tensor_storage" || manifest.get(key) == compact_object.get(key),
            "compact/full EXL3 metadata disagree at {key}"
        );
    }
    ensure!(!compact_object.contains_key("tensor_storage"), "config.json carries an inline tensor_storage");
    check_quantization_config(compact)?;
    Ok(())
}

fn build_deepseek_v4_manifest(
    storage: Exl3StorageSource,
    derived: &Exl3StorageMap,
    backbone: crate::RoutedExpertShape,
    tiers: &[usize],
) -> Result<V41Exl3Manifest> {
    if storage == Exl3StorageSource::GptqModel {
        if let Some(name) = derived.keys().find(|name| !is_routed_module(name)) {
            anyhow::bail!("EXL3 manifest includes non-routed or unexpected projections: {name}");
        }
    }
    let routed: BTreeMap<&str, &Exl3Module> = derived
        .iter()
        .filter(|(name, _)| is_routed_module(name))
        .map(|(name, module)| (name.as_str(), module))
        .collect();
    ensure!(!routed.is_empty(), "the checkpoint has no routed EXL3 experts (*.mlp.experts.*.trellis)");
    let language_model = routed.keys().any(|name| name.starts_with("model.language_model.layers."));
    let mtp_layers = language_model && routed.keys().any(|name| name.starts_with("mtp.layers."));
    let naming = if mtp_layers {
        V41Exl3Naming::HfLanguageModelMtp
    } else if language_model {
        V41Exl3Naming::HfLanguageModel { layers: backbone.layers }
    } else {
        V41Exl3Naming::HfMlp
    };
    // Draft stages: `mtp.{S}` or `mtp.layers.{S}` names, or language-model
    // layers past the backbone.
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
                    ensure!(
                        (2..=5).contains(&module.bits),
                        "projection {name} is K{}; the expert kernels run K2..K5",
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
        config: None,
        experts: crate::RoutedExpertShape { draft_stages, draft_experts, ..backbone },
        naming,
        decoder_tiers: decoder_family_preferring(&projections, tiers)?,
        projections,
        ple_quantization: None,
        mtp_experts: V41Exl3MtpExperts::Exl3,
        storage,
    })
}

fn parse_manifest(mut config: Value, manifest: &Value) -> Result<V41Exl3Manifest> {
    for field in ["quant_method", "method", "format", "checkpoint_format"] {
        ensure!(
            manifest.get(field).and_then(Value::as_str) == Some("exl3"),
            "EXL3 manifest requires {field}=exl3"
        );
    }
    ensure!(
        manifest["codebook"] == "mcg"
            && manifest["out_scales"] == "never"
            && manifest["group_size"] == -1
            && manifest["desc_act"] == false
            && manifest["pack_dtype"] == "int32",
        "unsupported EXL3 storage contract"
    );
    ensure!(
        manifest["bits"]
            .as_u64()
            .is_some_and(|v| (2..=5).contains(&v)),
        "EXL3 base bits must be an integer in 2..5"
    );
    let meta = manifest
        .pointer("/meta/cuteafd")
        .context("missing V4.1 EXL3 metadata")?;
    ensure!(
        meta["schema"] == V41_EXL3_SCHEMA && meta["tensor_naming"] == "checkpoint-native",
        "unsupported V4.1 EXL3 schema or namespace"
    );
    let compact = config["quantization_config"]
        .as_object()
        .context("missing compact EXL3 config")?;
    ensure!(
        !compact.is_empty() && compact.get("quant_method") == Some(&Value::from("exl3")),
        "config must declare EXL3 quantization"
    );
    for (key, value) in compact {
        ensure!(
            manifest.get(key) == Some(value),
            "compact/full EXL3 metadata disagree at {key}"
        );
    }
    config["quantization_config"] = meta["native_quantization_config"].clone();
    let ple_quantization = config
        .as_object_mut()
        .context("config must be an object")?
        .remove("cuteafd_ple_quantization");
    let validated =
        OfficialV41Config::from_json(OFFICIAL_V41_MODEL_ID, &serde_json::to_vec(&config)?)?;
    let text = validated.text();
    let storage = manifest["tensor_storage"]
        .as_object()
        .context("missing EXL3 tensor_storage")?;
    let mut projections = BTreeMap::new();
    for (prefix, layers, experts) in [
        ("layers", text.num_hidden_layers, text.n_routed_experts),
        (
            "mtp",
            validated.dspark_compress_ratios().len(),
            text.dspark_n_routed_experts,
        ),
    ] {
        for layer in 0..layers {
            for expert in 0..experts {
                for (stem, kind) in [
                    ("w1", V41Exl3ProjectionKind::Gate),
                    ("w3", V41Exl3ProjectionKind::Up),
                    ("w2", V41Exl3ProjectionKind::Down),
                ] {
                    let name = format!("{prefix}.{layer}.ffn.experts.{expert}.{stem}");
                    let value = storage
                        .get(&name)
                        .with_context(|| format!("missing projection {name}"))?;
                    let (input, output) = if kind == V41Exl3ProjectionKind::Down {
                        (text.moe_intermediate_size, text.hidden_size)
                    } else {
                        (text.hidden_size, text.moe_intermediate_size)
                    };
                    let projection = parse_projection(&name, kind, input, output, value)?;
                    projections.insert(name, projection);
                }
            }
        }
    }
    ensure!(
        storage.len() == projections.len(),
        "EXL3 manifest includes non-routed or unexpected projections"
    );
    V41Exl3Manifest::v41(validated, projections, ple_quantization, V41Exl3MtpExperts::Exl3)
}

fn parse_projection(
    name: &str,
    kind: V41Exl3ProjectionKind,
    input: usize,
    output: usize,
    value: &Value,
) -> Result<V41Exl3Projection> {
    ensure!(
        value["quant_format"] == "exl3",
        "projection {name} is not EXL3"
    );
    let bits = value["bits_per_weight"]
        .as_u64()
        .filter(|b| (2..=5).contains(b))
        .with_context(|| format!("projection {name} must have integer K2..K5"))?
        as usize;
    ensure!(
        input > 0 && output > 0 && input % 128 == 0 && output % 128 == 0,
        "projection {name} requires H128-aligned geometry"
    );
    let projection = V41Exl3Projection {
        name: name.to_owned(),
        kind,
        bits,
        input_features: input,
        output_features: output,
    };
    let tensors = value["stored_tensors"]
        .as_object()
        .context("missing projection storage")?;
    ensure!(
        tensors.len() == 4,
        "projection {name} must contain exactly trellis/suh/svh/mcg"
    );
    for (suffix, dtype, shape) in [
        ("trellis", "int16", projection.trellis_shape().to_vec()),
        ("suh", "float16", vec![input]),
        ("svh", "float16", vec![output]),
        ("mcg", "int32", vec![]),
    ] {
        let key = format!("{name}.{suffix}");
        let tensor = tensors
            .get(&key)
            .with_context(|| format!("missing {key}"))?;
        let shape_ok = if suffix == "mcg" {
            tensor["shape"].as_array().is_some_and(|dims| dims.is_empty() || dims == &[Value::from(1)])
        } else {
            tensor["shape"] == serde_json::to_value(shape)?
        };
        ensure!(
            tensor["torch_dtype"] == dtype && shape_ok,
            "invalid EXL3 dtype/shape for {key}"
        );
    }
    Ok(projection)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn raw_publication_config(bits: u64) -> Value {
        let mut config: Value =
            serde_json::from_str(include_str!("official-v41-config.json")).unwrap();
        config["quantization_config"] = serde_json::json!({
            "quant_method": "exl3",
            "bits": bits,
            "codebook": "mcg",
            "mtp_experts": "source",
            "mtp_experts_start_layer": 40,
            "weight_block_size": [32, 32],
            "non_routed_quantization": {
                "quant_method": "deepseek_v4_fp8",
                "fmt": "e4m3",
                "activation_scheme": "dynamic",
                "scale_fmt": "ue8m0",
                "weight_block_size": [32, 32],
                "expert_dtype": "fp4",
            },
        });
        config
    }

    #[test]
    fn raw_publication_manifest_is_uniform_backbone_only() {
        let manifest = parse_raw_publication(raw_publication_config(2)).unwrap();
        assert!(manifest.mtp_experts_are_source());
        assert!(manifest.ple_quantization.is_none());
        assert_eq!(manifest.decoder_tiers(), &[2, 3]);
        assert_eq!(manifest.projections.len(), 40 * 384 * 3);
        assert!(manifest
            .projections
            .keys()
            .all(|name| name.starts_with("layers.")));
        let projection = &manifest.projections["layers.39.ffn.experts.383.w1"];
        assert_eq!(projection.bits, 2);
        assert_eq!(projection.trellis_shape(), [320, 144, 32]);
        let down = &manifest.projections["layers.0.ffn.experts.0.w2"];
        assert_eq!(down.trellis_shape(), [144, 320, 32]);
        assert!(manifest
            .config
            .as_ref()
            .unwrap()
            .quantization()
            .quant_method
            .eq_ignore_ascii_case("fp8"));
    }

    /// A miniature DeepSeek V4 EXL3 publication: HF-named routed experts, one
    /// draft stage, `out_scales=auto` and float `bits`, as V4 Pro EXL3 K2 ships.
    /// Returns the config, the GPTQModel storage map, the shape, and the
    /// safetensors headers (MCG marker shaped `mcg`).
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
    fn deepseek_v4_publication_reads_hf_named_experts_and_drafts() {
        let (config, manifest, shape, headers) = deepseek_v4_publication();
        let parsed = read_publication(&config, &[("quantize_config.json", &manifest)], shape, &headers, &[]).unwrap();
        assert!(parsed.config.is_none());
        assert_eq!(parsed.storage, Exl3StorageSource::GptqModel);
        assert_eq!(parsed.naming, V41Exl3Naming::HfMlp);
        assert_eq!(parsed.experts.draft_stages, 1);
        assert_eq!(parsed.experts.draft_experts, 3);
        assert_eq!(parsed.projections.len(), (2 + 1) * 3 * 3);
        assert_eq!(parsed.decoder_tiers(), &[2, 3]);
        assert!(!parsed.mtp_experts_are_source());
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
        assert_eq!((standard.experts, standard.naming), (mapped.experts, mapped.naming));
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
        assert!(error.contains("quantize_config.json disagrees with the safetensors headers at model.layers.0.mlp.experts.0.down_proj")
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
        assert!(error.contains("quantization_config.json") && error.contains("bits_per_weight 3"), "{error}");
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
    fn raw_publication_rejects_contract_drift() {
        for (pointer, value, expected) in [
            ("/quantization_config/bits", serde_json::json!(3.25), "integer bits"),
            (
                "/quantization_config/codebook",
                serde_json::json!("gptq"),
                "MCG codebook",
            ),
            (
                "/quantization_config/mtp_experts",
                serde_json::json!("exl3"),
                "mtp_experts=source",
            ),
            (
                "/quantization_config/non_routed_quantization/fmt",
                serde_json::json!("e5m2"),
                "non-routed FP8",
            ),
        ] {
            let mut config = raw_publication_config(2);
            *config.pointer_mut(pointer).unwrap() = value;
            let error = parse_raw_publication(config).unwrap_err().to_string();
            assert!(error.contains(expected), "unexpected error: {error}");
        }
        let mut extra = raw_publication_config(2);
        extra["quantization_config"]["surprise"] = serde_json::json!(true);
        assert!(parse_raw_publication(extra)
            .unwrap_err()
            .to_string()
            .contains("unexpected quantization_config key"));
        let mut wrong_start = raw_publication_config(2);
        wrong_start["quantization_config"]["mtp_experts_start_layer"] = serde_json::json!(39);
        assert!(parse_raw_publication(wrong_start)
            .unwrap_err()
            .to_string()
            .contains("mtp_experts_start_layer"));
    }

    #[test]
    #[ignore = "requires CUTEAFD_EXL3_RAW_SNAPSHOT pointing to a raw local publication"]
    fn raw_publication_checkpoint_manifest() {
        let path = std::env::var_os("CUTEAFD_EXL3_RAW_SNAPSHOT").expect("CUTEAFD_EXL3_RAW_SNAPSHOT");
        let manifest = read_v41_exl3_manifest(Path::new(&path)).unwrap();
        assert!(manifest.mtp_experts_are_source());
        assert_eq!(manifest.projections.len(), 46_080);
        assert_eq!(manifest.decoder_tiers(), &[2, 3]);
        assert!(manifest
            .projections
            .values()
            .all(|projection| projection.bits == 2));
        let index = read_json(
            &Path::new(&path).join("model.safetensors.index.json"),
            MAX_MANIFEST_BYTES,
        )
        .unwrap();
        let shards: std::collections::BTreeSet<_> = index["weight_map"]
            .as_object()
            .unwrap()
            .values()
            .map(|v| v.as_str().unwrap())
            .collect();
        let mut seen = std::collections::BTreeSet::new();
        for shard in shards {
            for tensor in crate::read_safetensors_metadata(&Path::new(&path).join(shard)).unwrap() {
                if !tensor.name.starts_with("layers.") || !tensor.name.contains(".ffn.experts.") {
                    continue;
                }
                let (prefix, _) = tensor.name.rsplit_once('.').unwrap();
                manifest
                    .projections
                    .get(prefix)
                    .expect("unexpected routed tensor")
                    .validate_tensor(&tensor)
                    .unwrap();
                assert!(seen.insert(tensor.name), "duplicate routed tensor");
            }
        }
        assert_eq!(seen.len(), 4 * manifest.projections.len());
        println!(
            "validated {} raw K2 projections",
            manifest.projections.len()
        );
    }

    #[test]
    #[ignore = "requires CUTEAFD_EXL3_SNAPSHOT pointing to a published local snapshot"]
    fn published_checkpoint_manifest() {
        let path = std::env::var_os("CUTEAFD_EXL3_SNAPSHOT").expect("CUTEAFD_EXL3_SNAPSHOT");
        let manifest = read_v41_exl3_manifest(Path::new(&path)).unwrap();
        assert_eq!(manifest.projections.len(), 47_232);
        assert_eq!(manifest.decoder_tiers(), &[3, 4]);
        // For the published quant the common family equals every old local
        // family, so descriptor layouts and allocation sizes remain unchanged.
        let mut families = BTreeMap::<String, BTreeSet<usize>>::new();
        for p in manifest.projections.values() {
            let prefix = p.name.split(".ffn.experts.").next().unwrap();
            families.entry(prefix.into()).or_default().insert(p.bits);
        }
        assert_eq!(families.len(), 43);
        assert!(families
            .values()
            .all(|bits| bits.iter().copied().collect::<Vec<_>>() == [3, 4]));
        let mut counts = BTreeMap::new();
        for p in manifest.projections.values() {
            *counts.entry(p.bits).or_insert(0usize) += 1;
        }
        assert_eq!(counts, BTreeMap::from([(3, 35_424), (4, 11_808)]));
        let index = read_json(
            &Path::new(&path).join("model.safetensors.index.json"),
            MAX_MANIFEST_BYTES,
        )
        .unwrap();
        let shards: std::collections::BTreeSet<_> = index["weight_map"]
            .as_object()
            .unwrap()
            .values()
            .map(|v| v.as_str().unwrap())
            .collect();
        let mut seen = std::collections::BTreeSet::new();
        for shard in shards {
            for tensor in crate::read_safetensors_metadata(&Path::new(&path).join(shard)).unwrap() {
                if !tensor.name.contains(".ffn.experts.") {
                    continue;
                }
                let (prefix, _) = tensor.name.rsplit_once('.').unwrap();
                manifest
                    .projections
                    .get(prefix)
                    .expect("unexpected routed tensor")
                    .validate_tensor(&tensor)
                    .unwrap();
                assert!(seen.insert(tensor.name), "duplicate routed tensor");
            }
        }
        assert_eq!(seen.len(), 4 * manifest.projections.len());
        println!(
            "validated {} projections; tiers {:?}; PLE override={}",
            manifest.projections.len(),
            counts,
            manifest.ple_quantization.is_some()
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
    fn new_transport_worlds_partition_nonempty_h128_blocks_exactly() {
        for intermediate in [640, 2048, 2304, 3072] {
            for kind in [V41Exl3ProjectionKind::Gate, V41Exl3ProjectionKind::Down] {
                let (input_features, output_features) = if kind == V41Exl3ProjectionKind::Down {
                    (intermediate, 5120)
                } else {
                    (5120, intermediate)
                };
                let p = V41Exl3Projection { name: "test".into(), kind, bits: 4,
                    input_features, output_features };
                for world in [1, 5, 7, 8] {
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
            // Only the admitted EXL3 worlds partition; six-rank Spark layouts
            // stay native, and an empty shard is never produced.
            for world in [0usize, 6, 9, 19] {
                assert!(
                    p.intermediate_partition(world, 0).is_err(),
                    "EXL3 world {world} must stay unadmitted"
                );
            }
        }
    }

    #[test]
    fn validates_each_integer_tier_and_rejects_shape_or_rotation_drift() {
        let name = "layers.0.ffn.experts.0.w1";
        for bits in 2..=5 {
            let mut tensors = serde_json::Map::new();
            for (suffix, dtype, shape) in [
                ("trellis", "int16", vec![320, 144, 16 * bits]),
                ("suh", "float16", vec![5120]),
                ("svh", "float16", vec![2304]),
                ("mcg", "int32", vec![]),
            ] {
                tensors.insert(
                    format!("{name}.{suffix}"),
                    serde_json::json!({"torch_dtype": dtype, "shape": shape}),
                );
            }
            let value = serde_json::json!({"quant_format": "exl3", "bits_per_weight": bits, "stored_tensors": tensors});
            let parse =
                |v: &Value| parse_projection(name, V41Exl3ProjectionKind::Gate, 5120, 2304, v);
            assert_eq!(
                parse(&value).unwrap().trellis_bytes(),
                5120 * 2304 * bits / 8
            );
            for invalid in [
                serde_json::json!(1),
                serde_json::json!(6),
                serde_json::json!(3.25),
            ] {
                let mut bad = value.clone();
                bad["bits_per_weight"] = invalid;
                assert!(parse(&bad).is_err());
            }
            let mut bad = value.clone();
            bad["stored_tensors"][format!("{name}.suh")]["shape"] = serde_json::json!([2304]);
            assert!(parse(&bad).is_err());
            let mut bad = value.clone();
            bad["stored_tensors"][format!("{name}.trellis")]["shape"] =
                serde_json::json!([144, 320, 16 * bits]);
            assert!(parse(&bad).is_err());
        }
    }
}

#[cfg(test)]
mod legacy_name_tests {
    #[test]
    fn ds41rt_publications_read_under_current_names() {
        let mut value = serde_json::json!({"meta": {"ds41rt": {"schema": "ds41rt.v41-routed-exl3.v1"}},
            "ds41rt_ple_quantization": {"schema": "ds41rt.nvfp4-ple.v1"}, "other": "ds41rt-free"});
        super::normalize_legacy_names(&mut value);
        assert_eq!(value["meta"]["cuteafd"]["schema"], super::V41_EXL3_SCHEMA);
        assert_eq!(value["cuteafd_ple_quantization"]["schema"], "cuteafd.nvfp4-ple.v1");
        assert_eq!(value["other"], "ds41rt-free");
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
}
