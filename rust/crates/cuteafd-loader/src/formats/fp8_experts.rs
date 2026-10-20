//! Routed experts kept as the checkpoint's FP8: E4M3 weights with one FP32
//! (or BF16, Qwen 3.8 Flash Next; widened at read) scale per 128x128 block
//! (`weight_scale_inv`), Hugging Face names
//! `model.layers.{l}.mlp.experts.{e}.{gate,up,down}_proj.weight[_scale_inv]`
//! or, for multimodal checkpoints, under `model.language_model.` (MiMo V2
//! Flash; GLM 5.x and GLM 5.3 Flash official FP8). The `fp8` expert family serves them without re-quantization.
//!
//! Tensor-parallel slices split the intermediate `I` over `tp` ranks in whole
//! 128-row blocks, as evenly as the blocks allow (the first ranks own any
//! extra block): rank `r` owns gate/up rows and down columns `[first, first +
//! len)`, with the matching block-scale rows (gate/up `[I/128, H/128]`) or
//! columns (down `[H/128, I/128]`). Every rank stores its slice zero-padded
//! to the widest (TP6 of 2048: 3, 3, 3, 3, 2, 2 blocks stored as 384); where
//! `tp` divides the blocks this is the plain `[r * I/tp, (r + 1) * I/tp)`.
//!
//! MiMo V2.6 Pro stores its experts as MXFP4 instead (`ExpertFormat::Mxfp4`):
//! `weight` U8 `[N, K/2]` (two E2M1 codes per byte, the even element in the
//! low nibble) and `weight_scale` U8 `[N, K/32]` (UE8M0 exponents). The same
//! programs widen them exactly to BF16 (`fp8-mimop` packages). Legacy padded
//! slices split `I` in whole 32-element scale blocks and pad to 128.
//! Exact-even packages use whole H128 blocks. Uneven native MXFP4 packages
//! keep exact K32 widths (352, 352, 352, 352, 320, 320), rotated by absolute
//! layer. Down scale rows alone are padded to their u32 physical pitch.
//!
//! NVIDIA ModelOpt NVFP4 releases (`ExpertFormat::Nvfp4`: GLM 5.3, GLM 5.3
//! Flash, Qwen 3.8 Flash Next) store `weight` U8 `[N, K/2]` (E2M1, even
//! element low), `weight_scale` E4M3 `[N, K/16]` (linear layout) and an FP32
//! `weight_scale_2` per projection (the expert's alpha), plus calibrated
//! FP32 `input_scale`. Native packages quantize activations to E2M1 with
//! E4M3 per-16 scales and apply the checkpoint's W4A4 recipe at every row count.
//! Slices split `I` in whole 16-value scale blocks (TP6 of 2048: 352, 352,
//! 336, 336, 336, 336), zero-padded to one 128-aligned width (384); each
//! projection's scale region is the E4M3 grid of every expert followed by
//! the experts' FP32 alphas and FP32 input scales (read by the W4A4 route).
use crate::catalog::read_safetensors_metadata;
use crate::families::deepseek_v41::v41_catalog::RoutedExpertShape;
use anyhow::{ensure, Context, Result};
use cuteafd_core::DType;
use std::collections::{BTreeMap, HashMap};
use std::os::unix::fs::FileExt;
use std::path::{Path, PathBuf};
use std::sync::Arc;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Fp8Projection {
    Gate,
    Up,
    Down,
}

impl Fp8Projection {
    pub const ALL: [Self; 3] = [Self::Gate, Self::Up, Self::Down];

    fn stem(self) -> &'static str {
        match self {
            Self::Gate => "gate_proj",
            Self::Up => "up_proj",
            Self::Down => "down_proj",
        }
    }
}

/// Storage of the routed expert weights.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ExpertFormat {
    /// Native BF16 matrices without quantization scales.
    Bf16,
    /// E4M3 with FP32 (or BF16) 128x128 block scales (`weight_scale_inv`).
    Fp8Block128,
    /// Packed E2M1 with UE8M0 scales per 32 values along K (`weight_scale`).
    Mxfp4,
    /// ModelOpt NVFP4: packed E2M1, E4M3 scales per 16 values along K
    /// (`weight_scale`) and an FP32 per-tensor `weight_scale_2`.
    Nvfp4,
}

impl ExpertFormat {
    /// Canonical native arithmetic package. NVFP4 always quantizes activations
    /// with the checkpoint's calibrated K16 scales (W4A4), at every row count.
    pub fn package_name(self, geometry: &str) -> String {
        format!("fp8-{geometry}{}", self.package_suffix())
    }

    fn package_suffix(self) -> &'static str {
        match self {
            Self::Fp8Block128 | Self::Mxfp4 => "",
            Self::Nvfp4 => "-nvfp4a4",
            Self::Bf16 => "-bf16-weights",
        }
    }

    /// Whether the weights are packed E2M1 (two codes per byte).
    pub fn packed_fp4(self) -> bool {
        matches!(self, Self::Mxfp4 | Self::Nvfp4)
    }

    /// Values per scale along K: 128 (FP8 blocks), 32 (MXFP4), 16 (NVFP4).
    pub fn group(self) -> usize {
        match self {
            Self::Fp8Block128 => 128,
            Self::Mxfp4 => 32,
            Self::Nvfp4 => 16,
            Self::Bf16 => 128,
        }
    }
}

#[derive(Debug, Clone)]
struct Located {
    shard: String,
    offset: u64,
    bytes: u64,
    dtype: DType,
    shape: Vec<usize>,
}

fn first_selected_layer(shape: RoutedExpertShape, layers: Option<&std::collections::BTreeSet<usize>>) -> usize {
    layers.and_then(|selected| selected.first().copied()).unwrap_or(shape.first_layer)
}

/// An NVFP4 projection's `input_scale` tensor name from its weight name
/// (`....gate_proj.weight` -> `....gate_proj.input_scale`).
fn input_scale_name(weight: &str) -> String {
    format!("{}.input_scale", weight.strip_suffix(".weight").unwrap_or(weight))
}

/// Where every routed FP8 expert tensor lives in the snapshot.
#[derive(Debug, Clone)]
pub struct Fp8ExpertTensors {
    snapshot: PathBuf,
    shape: RoutedExpertShape,
    /// `model.` or `model.language_model.`: where the decoder layers live.
    prefix: String,
    /// Whether draft (MTP) experts live under `mtp.layers.{s}.` (Qwen 3.8 Flash
    /// Next) rather than as decoder layers past the backbone: layer
    /// `shape.layers + s` names them.
    mtp_layers: bool,
    format: ExpertFormat,
    tensors: Arc<HashMap<String, Located>>,
    activations: Vec<cuteafd_core::FfnActivation>,
    slice_layer: usize,
    coordinator: bool,
}

/// Parallel readers in the projection-at-a-time native expert loader.
pub const FP8_LOAD_READERS: usize = 16;

/// Host owners of one projection load. The synchronous H2D owner retains its
/// largest upload until all layers finish; reader buffers drain before upload.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Fp8ProjectionLoad {
    pub weights: usize,
    pub scales: usize,
    pub readers: usize,
    pub reader_staging: usize,
    pub reader_auxiliary: usize,
    pub gate_inputs: usize,
}

impl Fp8ProjectionLoad {
    pub fn pinned_upload(&self) -> usize { self.weights.max(self.scales) }

    pub fn host_bytes(&self) -> Result<usize> {
        self.reader_staging.checked_add(self.reader_auxiliary)
            .and_then(|bytes| bytes.checked_mul(self.readers))
            .and_then(|bytes| bytes.checked_add(self.weights))
            .and_then(|bytes| bytes.checked_add(self.scales))
            .and_then(|bytes| bytes.checked_add(self.gate_inputs))
            .context("FP8 projection host load overflow")
    }
}

/// Layer maxima for sequential projections, including retained upload staging.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Fp8LayerLoad {
    pub resident: usize,
    pub host: usize,
    pub pinned: usize,
}

/// How a package lays a TP rank's intermediate slice out.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Slicing {
    /// Ranks own whole format blocks as evenly as possible; every rank stores
    /// the widest rank's width, zero-padded to 128 (one program per world).
    Padded,
    /// Ranks own whole `g`-row blocks (H128, or native MXFP4 K32) and store
    /// exactly their own rows (a program per distinct width): no weight padding.
    Blocks(usize),
}

/// Use unequal native-unit slices only when every rank's width is packaged.
/// An incomplete Spark tree or CUTEAFD_FP8_EXACT_SLICES=0 selects the legacy
/// padded layout, except exact-even native splits need no width-specific package.
/// NVFP4 retains H128 storage padding unless its exact-even width needs none.
/// The worker validates its width against the planner recipe
/// before allocation; the coordinator package tree never defines Spark slicing.
pub fn exact_layout_directory(directory: &Path, tensors: &Fp8ExpertTensors, tp: usize,
    rank: usize) -> (PathBuf, Slicing) {
    exact_layout_directory_with(directory, tensors, tp, rank,
        std::env::var("CUTEAFD_FP8_EXACT_SLICES").as_deref() != Ok("0"))
}

fn exact_layout_directory_with(directory: &Path, tensors: &Fp8ExpertTensors, tp: usize,
    rank: usize, enabled: bool) -> (PathBuf, Slicing) {
    if tp == 0 || rank >= tp { return (directory.to_path_buf(), Slicing::Padded); }
    let padded = (directory.to_path_buf(), tensors.slicing_for_directory(directory, tp));
    if tensors.needs_nvfp4_padding(tp) { return padded; }
    if !enabled || tp < 2 { return padded; }
    let (Some(package), Some(layout)) = (directory.parent(), directory.file_name().and_then(|name| name.to_str()))
        else { return padded; };
    let exact = tensors.exact_slicing(tp);
    let paths = (0..tp).map(|rank| tensors.rank_width(tp, rank, exact).ok()
        .map(|width| package.join(format!("{layout}-w{width}")))).collect::<Option<Vec<_>>>();
    match paths {
        Some(paths) if paths.iter().all(|path| path.is_dir()) => (paths[rank].clone(), exact),
        _ => padded,
    }
}

/// A packed checkpoint matrix window uploaded directly from a shared read bank.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Fp8MatrixSlice {
    pub source_offset: usize,
    pub source_pitch: usize,
    pub destination_pitch: usize,
    pub width: usize,
    pub rows: usize,
    pub destination_bytes: usize,
}

fn stacked_projection<'a>(name: &'a str, prefix: &str) -> Option<(usize, Fp8Projection, &'a str)> {
    let (layer, rest) = name.strip_prefix(prefix)?.strip_prefix("layers.")?.split_once('.')?;
    let (projection, suffix) = rest.strip_prefix("moe.")?.split_once('.')?;
    let projection = Fp8Projection::ALL.into_iter().find(|p| p.stem() == projection)?;
    ["weight", "weight_scale_inv", "weight_scale", "weight_scale_2", "input_scale"].contains(&suffix)
        .then_some((layer.parse().ok()?, projection, suffix))
}

impl Fp8ExpertTensors {
    pub fn name(&self, layer: usize, expert: usize, projection: Fp8Projection) -> String {
        if self.mtp_layers && layer >= self.shape.layers {
            let stage = layer - self.shape.layers;
            return format!("mtp.layers.{stage}.mlp.experts.{expert}.{}.weight", projection.stem());
        }
        format!("{}layers.{layer}.mlp.experts.{expert}.{}.weight", self.prefix, projection.stem())
    }

    /// Reads the index and every shard header holding routed experts; checks
    /// that each expert tensor present is E4M3 with an FP32 128x128 grid, or
    /// MXFP4 (packed E2M1 U8 with UE8M0 per-32 scales).
    pub fn read(snapshot: &Path, shape: RoutedExpertShape) -> Result<Self> {
        Self::read_selected(snapshot, shape, None)
    }

    /// Remove unused draft headers without rereading backbone shards.
    pub(crate) fn retain_backbone(&mut self) {
        let prefix = self.prefix.clone();
        let end = self.shape.layers;
        Arc::make_mut(&mut self.tensors).retain(|name, _| name.strip_prefix(&prefix)
            .and_then(|name| name.strip_prefix("layers."))
            .and_then(|name| name.split('.').next()?.parse::<usize>().ok())
            .is_some_and(|layer| layer < end));
    }

    /// Open only selected layers, using a selected layer's actual headers to
    /// establish format. MTP-only readers must never probe a backbone shard.
    pub fn read_layers(snapshot: &Path, shape: RoutedExpertShape, layers: &std::collections::BTreeSet<usize>) -> Result<Self> {
        ensure!(!layers.is_empty(), "expert role needs at least one layer");
        Self::read_selected(snapshot, shape, Some(layers))
    }

    fn read_selected(snapshot: &Path, shape: RoutedExpertShape, layers: Option<&std::collections::BTreeSet<usize>>) -> Result<Self> {
        let index = crate::families::deepseek_v41::v41_exl3::read_json(&snapshot.join("model.safetensors.index.json"), 64 * 1024 * 1024)?;
        let weight_map: BTreeMap<String, String> = serde_json::from_value(
            index.get("weight_map").cloned().context("index has no weight_map")?)?;
        let prefix = if weight_map.keys().any(|name| name.starts_with("model.language_model.layers.")) {
            "model.language_model."
        } else {
            "model."
        };
        let mtp_layers = weight_map.keys().any(|name| name.starts_with("mtp.layers.") && name.contains(".mlp.experts."));
        let routed = |name: &str| {
            let layer = name.strip_prefix(prefix).and_then(|n| n.strip_prefix("layers."))
                .and_then(|n| n.split('.').next()).and_then(|n| n.parse::<usize>().ok())
                .or_else(|| name.strip_prefix("mtp.layers.").and_then(|n| n.split('.').next())
                    .and_then(|n| n.parse::<usize>().ok()).and_then(|s| shape.layers.checked_add(s)));
            (name.contains(".mlp.experts.") || stacked_projection(name, prefix).is_some())
                && layer.is_some_and(|layer| layers.is_none_or(|selected| selected.contains(&layer)))
        };
        let shards: std::collections::BTreeSet<&String> =
            weight_map.iter().filter(|(name, _)| routed(name)).map(|(_, shard)| shard).collect();
        let mut tensors = HashMap::new();
        let mut source_names = HashMap::new();
        for shard in shards {
            let needed = weight_map.iter().find(|(name, file)| *file == shard && routed(name))
                .map(|(name, _)| name).context("selected expert shard has no requirement")?;
            for meta in read_safetensors_metadata(&snapshot.join(shard))
                .with_context(|| format!("expert role needs tensor {needed} in shard {shard} at snapshot {}", snapshot.display()))?
            {
                if routed(&meta.name) && weight_map.get(&meta.name) == Some(shard) {
                    if let Some((layer, projection, suffix)) = stacked_projection(&meta.name, prefix) {
                        ensure!(shape.experts > 0 && meta.shape.first() == Some(&shape.experts),
                            "{}: expected expert axis {}, found {:?}", meta.name, shape.experts, meta.shape);
                        ensure!(meta.byte_length % shape.experts as u64 == 0,
                            "{}: expert-axis byte extent is not uniform", meta.name);
                        let bytes = meta.byte_length / shape.experts as u64;
                        for expert in 0..shape.experts {
                            // Metadata views only: each expert retains its original shard and
                            // direct byte range. No payload is rewritten or duplicated.
                            let name = format!("{prefix}layers.{layer}.mlp.experts.{expert}.{}.{suffix}", projection.stem());
                            let offset = meta.byte_offset.checked_add(bytes.checked_mul(expert as u64)
                                .context("expert view byte offset overflow")?).context("expert view offset overflow")?;
                            ensure!(!tensors.contains_key(&name), "{}: duplicate stacked/per-expert storage for {name}", meta.name);
                            source_names.insert(name.clone(), meta.name.clone());
                            tensors.insert(name, Located { shard: shard.clone(), offset, bytes,
                                dtype: meta.dtype.clone(), shape: meta.shape[1..].to_vec() });
                        }
                    } else {
                        ensure!(!tensors.contains_key(&meta.name), "{}: duplicate expert tensor", meta.name);
                        tensors.insert(meta.name.clone(), Located { shard: shard.clone(), offset: meta.byte_offset,
                            bytes: meta.byte_length, dtype: meta.dtype, shape: meta.shape });
                    }
                }
            }
        }
        let checkpoint = crate::plan::checkpoint::Checkpoint::inventory(snapshot)?;
        let activations = match crate::plan::family::detect(&checkpoint) {
            Some(family) if family.id() == "step5" => {
                crate::plan::families::step5::Config::parse(&checkpoint)?.activation
            }
            _ => Vec::new(),
        };
        let mut catalog = Self { snapshot: snapshot.to_path_buf(), shape, prefix: prefix.to_string(), mtp_layers,
            format: ExpertFormat::Fp8Block128, tensors: Arc::new(tensors), activations,
            slice_layer: first_selected_layer(shape, layers), coordinator: false };
        let first_layer = layers.and_then(|selected| selected.first().copied()).unwrap_or(shape.first_layer);
        let first = catalog.name(first_layer, 0, Fp8Projection::Gate);
        if catalog.located(&first)?.dtype == DType::Bf16 {
            catalog.format = ExpertFormat::Bf16;
        }
        if matches!(catalog.located(&first)?.dtype, DType::U8 | DType::I8) {
            // E4M3 scales with a second-level FP32 scale: ModelOpt NVFP4; UE8M0 ones: MXFP4.
            let nvfp4 = catalog.tensors.get(&format!("{first}_scale")).is_some_and(|s| s.dtype == DType::F8E4M3)
                && catalog.tensors.contains_key(&format!("{first}_scale_2"));
            catalog.format = if nvfp4 { ExpertFormat::Nvfp4 } else { ExpertFormat::Mxfp4 };
        }
        for name in catalog.tensors.keys().filter(|name| name.ends_with(".weight")) {
            use crate::plan::checkpoint::ModuleNumerics;
            let source = source_names.get(name).map(String::as_str).unwrap_or(name);
            let numerics = checkpoint.module_numerics_with_header(source, catalog.tensors[name].dtype.clone())?;
            let nvfp4 = catalog.tensors.contains_key(&format!("{name}_scale_2"));
            match numerics {
                ModuleNumerics::W4A4 => ensure!(nvfp4, "{name}: declared W4A4 needs native NVFP4 weights"),
                ModuleNumerics::W8A8 => {
                    ensure!(!nvfp4, "{name}: declared W8A8 cannot execute as W4A4");
                    let qwen = checkpoint.model_type() == Some("qwen4_exp")
                        || checkpoint.text_config().get("model_type").and_then(serde_json::Value::as_str)
                            == Some("qwen4_exp_text");
                    ensure!(qwen && name.starts_with("mtp.layers."),
                        "{name}: declared expert W8A8 needs a both-projection native kernel");
                    // Qwen MTP is an explicitly reported, temporary deviation.
                }
                ModuleNumerics::CheckpointPrecision | ModuleNumerics::W8CheckpointActivations => {
                    ensure!(!nvfp4 || checkpoint.quantization().is_none(),
                        "{name}: NVFP4 storage disagrees with declared {numerics:?}");
                }
            }
        }
        // One expert of the first routed layer fixes the format contract.
        for projection in Fp8Projection::ALL {
            catalog.check(first_layer, 0, projection)?;
        }
        Ok(catalog)
    }

    pub fn shape(&self) -> &RoutedExpertShape {
        &self.shape
    }

    /// Family-declared arithmetic, when it differs from the process geometry's legacy default.
    pub fn activation(&self, layer: usize) -> Option<cuteafd_core::FfnActivation> {
        self.activations.get(layer).copied()
    }

    pub fn format(&self) -> ExpertFormat {
        self.format
    }

    /// Coordinator packages retain their fixed contiguous H128 rank ABI.
    pub fn for_coordinator(&self) -> Self {
        let mut view = self.clone();
        view.coordinator = true;
        view
    }

    fn slice_rule(&self, units: usize, tp: usize) -> cuteafd_core::tp_units::UnitRule {
        if self.coordinator { cuteafd_core::tp_units::UnitRule::Contiguous }
        else { cuteafd_core::tp_units::layer_rule(units, tp) }
    }

    pub fn exact_slicing(&self, _tp: usize) -> Slicing {
        Slicing::Blocks(if self.coordinator { 128 } else { self.block() })
    }

    fn needs_nvfp4_padding(&self, tp: usize) -> bool {
        self.format == ExpertFormat::Nvfp4 && !self.coordinator
            && (tp == 0 || self.shape.intermediate == 0 || self.shape.intermediate % (128 * tp) != 0)
    }

    pub fn unit_map(&self, tp: usize, rank: usize, slicing: Slicing) -> Result<cuteafd_core::tp_units::RankUnits> {
        let block = match slicing { Slicing::Blocks(g) => g, Slicing::Padded => self.block() };
        let rule = self.slice_rule(self.shape.intermediate / block, tp);
        Ok(cuteafd_core::tp_units::RankUnits::new(rule, self.shape.experts,
            self.shape.intermediate, block, self.slice_layer, tp, rank)?)
    }

    pub fn slicing_for_directory(&self, directory: &Path, tp: usize) -> Slicing {
        if self.needs_nvfp4_padding(tp) {
            Slicing::Padded
        } else if directory.file_name().and_then(|name| name.to_str()).is_some_and(|name| name.contains("-w"))
            || (tp > 0 && self.shape.intermediate > 0 && self.shape.intermediate % (self.block() * tp) == 0) {
            self.exact_slicing(tp)
        } else { Slicing::Padded }
    }

    /// Storage format of a layer's first gate, including mixed-format MTP
    /// layers. This only inspects headers; validate_layer checks the full
    /// layer against the selected package before any weight is loaded.
    pub fn layer_format(&self, layer: usize) -> Result<ExpertFormat> {
        let first = self.name(layer, 0, Fp8Projection::Gate);
        match &self.located(&first)?.dtype {
            DType::F8E4M3 => Ok(ExpertFormat::Fp8Block128),
            DType::U8 | DType::I8 => {
                let nvfp4 = self.tensors.get(&format!("{first}_scale"))
                    .is_some_and(|s| s.dtype == DType::F8E4M3)
                    && self.tensors.contains_key(&format!("{first}_scale_2"));
                Ok(if nvfp4 { ExpertFormat::Nvfp4 } else { ExpertFormat::Mxfp4 })
            }
            DType::Bf16 => Ok(ExpertFormat::Bf16),
            dtype => anyhow::bail!("{first}: unsupported expert dtype {dtype:?}"),
        }
    }

    /// A metadata-only view using this layer's format. The layer numbering and
    /// tensor names are preserved so a mixed-format MTP package can own just
    /// the draft layer. Validate all experts before allocating its storage.
    pub fn for_layer(&self, layer: usize) -> Result<Self> {
        let view = self.for_layer_metadata(layer)?;
        view.validate_layer(layer)?;
        Ok(view)
    }

    /// Header-only planning view. Runtime loading uses `for_layer` to validate
    /// every expert before allocating storage.
    pub fn for_layer_metadata(&self, layer: usize) -> Result<Self> {
        let mut view = self.clone();
        view.format = self.layer_format(layer)?;
        view.slice_layer = layer;
        Ok(view)
    }

    /// Contiguous homogeneous packages preserve absolute layer order. Views share
    /// immutable shard metadata, including direct stacked-expert byte ranges.
    pub fn homogeneous_layers(&self, layers: &[usize]) -> Result<Vec<(Self, Vec<usize>)>> {
        ensure!(!layers.is_empty() && layers.windows(2).all(|w| w[0] < w[1]),
            "expert group layers must be nonempty, sorted and unique");
        let mut groups = Vec::<(Self, Vec<usize>)>::new();
        for &layer in layers {
            let format = self.layer_format(layer)?;
            if let Some((view, ids)) = groups.last_mut().filter(|(view, ids)|
                view.format == format && view.activation(ids[0]) == self.activation(layer)
                    && ids.last().is_some_and(|last| last.checked_add(1) == Some(layer))) {
                view.validate_layer(layer)?;
                ids.push(layer);
            } else {
                groups.push((self.for_layer(layer)?, vec![layer]));
            }
        }
        Ok(groups)
    }

    /// Whether `layer` has routed FP8 experts in this snapshot.
    pub fn has_layer(&self, layer: usize) -> bool {
        self.tensors.contains_key(&self.name(layer, 0, Fp8Projection::Gate))
    }

    /// Check every requested expert's tensor headers before committing device
    /// memory to a resident layer set. Some checkpoints have an MTP layer in
    /// a different format from the backbone; one package cannot run both.
    pub fn validate_layer(&self, layer: usize) -> Result<()> {
        for expert in 0..self.shape.experts {
            for projection in Fp8Projection::ALL {
                self.check(layer, expert, projection)?;
                if self.format == ExpertFormat::Nvfp4 {
                    let name = input_scale_name(&self.name(layer, expert, projection));
                    let scale = self.located(&name)?;
                    ensure!(scale.dtype == DType::F32 && scale.bytes == 4,
                        "{name}: expected one FP32 input scale, found {:?} {:?}", scale.dtype, scale.shape);
                }
            }
        }
        Ok(())
    }

    fn dims(&self, projection: Fp8Projection) -> (usize, usize) {
        let (h, i) = (self.shape.hidden, self.shape.intermediate);
        if projection == Fp8Projection::Down { (h, i) } else { (i, h) }
    }

    fn located(&self, name: &str) -> Result<&Located> {
        self.tensors.get(name).with_context(|| format!("checkpoint has no routed expert tensor {name}"))
    }

    fn check(&self, layer: usize, expert: usize, projection: Fp8Projection) -> Result<()> {
        let name = self.name(layer, expert, projection);
        let (rows, cols) = self.dims(projection);
        let weight = self.located(&name)?;
        if self.format == ExpertFormat::Bf16 {
            ensure!(weight.dtype == DType::Bf16 && weight.shape == [rows, cols],
                "{name}: expected native BF16 [{rows}, {cols}], found {:?} {:?}", weight.dtype, weight.shape);
            for suffix in ["_scale_inv", "_scale", "_scale_2"] {
                ensure!(!self.tensors.contains_key(&format!("{name}{suffix}")), "{name}: native BF16 must not have quantization scales");
            }
            let input = input_scale_name(&name);
            ensure!(!self.tensors.contains_key(&input), "{input}: native BF16 must not have quantization input scales");
            return Ok(());
        }
        if self.format == ExpertFormat::Nvfp4 {
            ensure!(weight.dtype == DType::U8 && weight.shape == [rows, cols / 2] && cols % 16 == 0,
                "{name}: expected packed E2M1 U8 [{rows}, {}], found {:?} {:?}", cols / 2, weight.dtype, weight.shape);
            let scale = self.located(&format!("{name}_scale"))?;
            ensure!(scale.dtype == DType::F8E4M3 && scale.shape == [rows, cols / 16],
                "{name}_scale: expected E4M3 [{rows}, {}] (16 values per scale, linear), found {:?} {:?}", cols / 16,
                scale.dtype, scale.shape);
            let alpha = self.located(&format!("{name}_scale_2"))?;
            ensure!(alpha.dtype == DType::F32 && alpha.bytes == 4,
                "{name}_scale_2: expected one FP32 value, found {:?} {:?}", alpha.dtype, alpha.shape);
            return Ok(());
        }
        if self.format == ExpertFormat::Mxfp4 {
            ensure!(matches!(weight.dtype, DType::U8 | DType::I8) && weight.shape == [rows, cols / 2] && cols % 32 == 0,
                "{name}: expected packed E2M1 U8 [{rows}, {}], found {:?} {:?}", cols / 2, weight.dtype, weight.shape);
            let scale = self.located(&format!("{name}_scale"))?;
            ensure!(matches!(scale.dtype, DType::U8 | DType::F8E8M0) && scale.shape == [rows, cols / 32],
                "{name}_scale: expected UE8M0 [{rows}, {}], found {:?} {:?}", cols / 32, scale.dtype, scale.shape);
            return Ok(());
        }
        ensure!(weight.dtype == DType::F8E4M3 && weight.shape == [rows, cols],
            "{name}: expected E4M3 [{rows}, {cols}], found {:?} {:?}", weight.dtype, weight.shape);
        let scale = self.located(&format!("{name}_scale_inv"))?;
        ensure!(matches!(scale.dtype, DType::F32 | DType::Bf16) && scale.shape == [rows.div_ceil(128), cols.div_ceil(128)],
            "{name}_scale_inv: expected FP32 or BF16 [{}, {}] 128x128 block scales, found {:?} {:?}",
            rows.div_ceil(128), cols.div_ceil(128), scale.dtype, scale.shape);
        Ok(())
    }

    /// The stored width of rank `rank`'s slice: every rank of a padded package
    /// stores the widest rank's ([`Self::slice`]); an exact package stores each
    /// rank's own whole `g`-row blocks.
    pub fn rank_width(&self, tp: usize, rank: usize, slicing: Slicing) -> Result<usize> {
        match slicing {
            Slicing::Padded => self.slice(tp),
            Slicing::Blocks(_) => Ok(self.rank_range_with(tp, rank, slicing)?.1),
        }
    }

    /// [`Self::rank_range`] under `slicing`: exact packages own whole `g`-row
    /// blocks (`g` a multiple of the format's block), as evenly as they split.
    pub fn rank_range_with(&self, tp: usize, rank: usize, slicing: Slicing) -> Result<(usize, usize)> {
        let Slicing::Blocks(g) = slicing else { return self.rank_range(tp, rank) };
        ensure!(rank < tp, "rank {rank} of TP{tp}");
        let i = self.shape.intermediate;
        ensure!(g > 0 && g % self.block() == 0 && i % g == 0 && i / g >= tp,
            "intermediate {i} does not split into whole {g}-row blocks over {tp} ranks");
        let rule = self.slice_rule(i / g, tp);
        let map = cuteafd_core::tp_units::RankUnits::new(rule, 1, i, g, self.slice_layer, tp, rank)?;
        let ranges = map.column_ranges(0)?;
        ensure!(ranges.len() == 1, "fixed-width package requires one contiguous range");
        Ok((ranges[0].start, ranges[0].len()))
    }

    /// [`Self::slice_bytes`] of rank `rank` under `slicing`.
    pub fn slice_bytes_with(&self, projection: Fp8Projection, tp: usize, rank: usize, slicing: Slicing)
        -> Result<(usize, usize)> {
        let (rows, cols) = self.dims(projection);
        let slice = self.rank_width(tp, rank, slicing)?;
        let (rows, cols) = if projection == Fp8Projection::Down { (rows, slice) } else { (slice, cols) };
        if self.format == ExpertFormat::Bf16 { return Ok((rows * cols * 2, 0)); }
        if self.format.packed_fp4() {
            let scale_cols = if self.format == ExpertFormat::Mxfp4 { (cols / 32).div_ceil(4) * 4 }
                else { cols / self.format.group() };
            return Ok((rows * cols / 2, rows * scale_cols));
        }
        Ok((rows * cols, rows.div_ceil(128) * cols.div_ceil(128) * 4))
    }

    /// [`Self::scale_region_bytes`] of rank `rank` under `slicing`.
    pub fn scale_region_bytes_with(&self, projection: Fp8Projection, tp: usize, rank: usize, slicing: Slicing)
        -> Result<usize> {
        let experts = self.shape.experts;
        let alphas = if self.format == ExpertFormat::Nvfp4 { experts * 8 } else { 0 };
        Ok(experts * self.slice_bytes_with(projection, tp, rank, slicing)?.1 + alphas)
    }

    /// Allocation recipe used by the native projection loader and its admission.
    pub fn projection_load(&self, layer: usize, projection: Fp8Projection, tp: usize,
        rank: usize, slicing: Slicing) -> Result<Fp8ProjectionLoad> {
        let experts = self.shape.experts;
        ensure!(experts > 0, "FP8 projection load needs experts");
        let (weights, _) = self.slice_bytes_with(projection, tp, rank, slicing)?;
        let weights = weights.checked_mul(experts).context("FP8 projection weights overflow")?;
        let scales = self.scale_region_bytes_with(projection, tp, rank, slicing)?;
        let readers = experts.div_ceil(experts.div_ceil(FP8_LOAD_READERS));
        let mut reader_staging = 0;
        let mut reader_auxiliary = 0;
        for expert in 0..experts {
            let name = self.name(layer, expert, projection);
            let weight = self.located(&name)?;
            if projection == Fp8Projection::Down && self.format != ExpertFormat::Bf16 {
                reader_staging = reader_staging.max(usize::try_from(weight.bytes)?);
            }
            if self.format.packed_fp4() {
                let scale = self.located(&format!("{name}_scale"))?;
                if projection == Fp8Projection::Down {
                    reader_staging = reader_staging.max(usize::try_from(scale.bytes)?);
                }
            } else if self.format == ExpertFormat::Fp8Block128 {
                let scale = self.located(&format!("{name}_scale_inv"))?;
                let (rows, cols) = self.dims(projection);
                let widened = rows.div_ceil(128).checked_mul(cols.div_ceil(128))
                    .and_then(|n| n.checked_mul(4)).context("FP8 scale grid overflow")?;
                // BF16 raw scales and their widened FP32 grid coexist briefly.
                reader_auxiliary = reader_auxiliary.max(usize::try_from(scale.bytes)?
                    .checked_add(widened).context("FP8 reader grid overflow")?);
            }
        }
        let gate_inputs = if self.format == ExpertFormat::Nvfp4 {
            experts.checked_mul(4).context("NVFP4 gate input scales overflow")?
        } else { 0 };
        Ok(Fp8ProjectionLoad { weights, scales, readers, reader_staging,
            reader_auxiliary, gate_inputs })
    }

    /// Sequential native load owners, excluding the final resident device bytes.
    pub fn layer_load(&self, layer: usize, tp: usize, rank: usize,
        slicing: Slicing) -> Result<Fp8LayerLoad> {
        let mut load = Fp8LayerLoad::default();
        for projection in Fp8Projection::ALL {
            let plan = self.projection_load(layer, projection, tp, rank, slicing)?;
            load.resident = load.resident.checked_add(plan.weights)
                .and_then(|bytes| bytes.checked_add(plan.scales))
                .context("FP8 layer resident overflow")?;
            load.host = load.host.max(plan.host_bytes()?);
            load.pinned = load.pinned.max(plan.pinned_upload());
        }
        Ok(load)
    }

    /// One expert's stored weight and scale read extent for an installed rank.
    pub fn weight_read_bytes_with(&self, tp: usize, rank: usize, slicing: Slicing) -> Result<usize> {
        let experts = self.shape.experts;
        ensure!(experts > 0, "expert read extent needs a positive expert count");
        Fp8Projection::ALL.iter().try_fold(0usize, |total, &projection| {
            let (weights, _) = self.slice_bytes_with(projection, tp, rank, slicing)?;
            let region = self.scale_region_bytes_with(projection, tp, rank, slicing)?;
            ensure!(region % experts == 0, "expert scale read extent is not uniform");
            total.checked_add(weights).and_then(|bytes| bytes.checked_add(region / experts))
                .context("expert weight/scale read extent overflow")
        })
    }

    /// Rows of one slice block: the 128x128 scale block (FP8), the 32-wide
    /// UE8M0 block (MXFP4) or the 16-wide E4M3 block (NVFP4).
    pub fn block(&self) -> usize {
        self.format.group()
    }

    /// The stored intermediate slice width of every rank of `tp`: the widest
    /// rank range in whole blocks, padded to 128.
    pub fn slice(&self, tp: usize) -> Result<usize> {
        let (i, block) = (self.shape.intermediate, self.block());
        crate::plan::experts::stored_slice(i, block, tp)
            .with_context(|| format!("intermediate {i} does not split into whole {block}-row blocks over {tp} ranks"))
    }

    /// The intermediate rows `[first, first + len)` rank `rank` of `tp` computes
    /// (the rest of its stored slice is zero padding).
    pub fn rank_range(&self, tp: usize, rank: usize) -> Result<(usize, usize)> {
        ensure!(rank < tp, "rank {rank} of TP{tp}");
        self.slice(tp)?;
        let map = cuteafd_core::tp_units::RankUnits::new(
            self.slice_rule(self.shape.intermediate / self.block(), tp),
            1, self.shape.intermediate, self.block(), self.slice_layer, tp, rank)?;
        let range = map.column_ranges(0)?.remove(0);
        Ok((range.start, range.len()))
    }

    /// Bytes of one expert projection's slice: (weight, scales) as stored
    /// (E4M3 + FP32 128x128 grid, or packed E2M1 + UE8M0 per 32).
    pub fn slice_bytes(&self, projection: Fp8Projection, tp: usize) -> Result<(usize, usize)> {
        let (rows, cols) = self.dims(projection);
        let slice = self.slice(tp)?;
        let (rows, cols) = if projection == Fp8Projection::Down { (rows, slice) } else { (slice, cols) };
        if self.format == ExpertFormat::Bf16 { return Ok((rows * cols * 2, 0)); }
        if self.format.packed_fp4() {
            let scale_cols = if self.format == ExpertFormat::Mxfp4 { (cols / 32).div_ceil(4) * 4 }
                else { cols / self.format.group() };
            return Ok((rows * cols / 2, rows * scale_cols));
        }
        Ok((rows * cols, rows.div_ceil(128) * cols.div_ceil(128) * 4))
    }

    /// Bytes of one projection's scale region for every expert: the scale
    /// grids, then (NVFP4) the experts' FP32 alphas and FP32 input scales.
    pub fn scale_region_bytes(&self, projection: Fp8Projection, tp: usize) -> Result<usize> {
        let experts = self.shape.experts;
        let alphas = if self.format == ExpertFormat::Nvfp4 { experts * 8 } else { 0 };
        Ok(experts * self.slice_bytes(projection, tp)?.1 + alphas)
    }

    /// An NVFP4 projection's `weight_scale_2` (alpha); 1 for the other formats.
    pub fn read_alpha(&self, layer: usize, expert: usize, projection: Fp8Projection) -> Result<f32> {
        self.read_scalar(&format!("{}_scale_2", self.name(layer, expert, projection)))
    }

    /// An NVFP4 projection's static activation scale (`input_scale`, the W4A4
    /// recipe's); 1 for the other formats.
    pub fn read_input_scale(&self, layer: usize, expert: usize, projection: Fp8Projection) -> Result<f32> {
        self.read_scalar(&input_scale_name(&self.name(layer, expert, projection)))
    }

    /// The W4A4 FC1 kernel quantizes a step's activations once, with the gate
    /// projection's `input_scale`, and dequantizes the up half with the up
    /// projection's `input_scale`: the two must be bit-identical or one half is
    /// silently scaled. The shipped NVFP4 checkpoints (GLM 5.3, GLM 5.3 Flash,
    /// Qwen 3.8 Flash Next) are; a repack that is not fails load here, naming
    /// the layer, the expert and both values. `gate` and `up` are the experts'
    /// little-endian FP32 `input_scale` bytes as the loader already read them,
    /// so the check costs no extra reads.
    pub fn check_input_scales(layer: usize, gate: &[u8], up: &[u8]) -> Result<()> {
        ensure!(gate.len() == up.len() && gate.len() % 4 == 0, "NVFP4 layer {layer}: input_scale regions differ in size");
        for (expert, (gate, up)) in gate.chunks_exact(4).zip(up.chunks_exact(4)).enumerate() {
            ensure!(gate == up,
                "NVFP4 layer {layer} expert {expert}: gate input_scale {} and up input_scale {} differ; \
                 the W4A4 FC1 kernel needs them bit-identical",
                f32::from_le_bytes(gate.try_into().unwrap()), f32::from_le_bytes(up.try_into().unwrap()));
        }
        Ok(())
    }

    fn read_scalar(&self, name: &str) -> Result<f32> {
        if self.format != ExpertFormat::Nvfp4 {
            return Ok(1.0);
        }
        let value = f32::from_le_bytes(self.read_scalar_bytes(name)?);
        ensure!(value.is_finite() && value > 0.0, "{name}: {value} is not a positive finite scale");
        Ok(value)
    }

    /// The raw bytes of one FP32 scalar tensor, so a comparison is bit-exact.
    fn read_scalar_bytes(&self, name: &str) -> Result<[u8; 4]> {
        let located = self.located(name)?;
        ensure!(located.dtype == DType::F32 && located.bytes == 4, "{name}: expected one FP32 value");
        let mut bytes = [0u8; 4];
        std::fs::File::open(self.snapshot.join(&located.shard))?.read_exact_at(&mut bytes, located.offset)
            .with_context(|| format!("reading {name}"))?;
        Ok(bytes)
    }

    /// Full projection sizes in the device format: BF16 block scales widen
    /// exactly to FP32; NVFP4 appends one alpha and input scale per expert.
    pub fn projection_bytes(&self, projection: Fp8Projection) -> (usize, usize) {
        let (rows, cols) = self.dims(projection);
        if self.format == ExpertFormat::Bf16 { return (rows * cols * 2, 0); }
        if self.format.packed_fp4() {
            (rows * cols / 2, rows * cols / self.format.group()
                + usize::from(self.format == ExpertFormat::Nvfp4) * 8)
        } else {
            (rows * cols, rows.div_ceil(128) * cols.div_ceil(128) * 4)
        }
    }

    /// One source read feeds every rank's projection. Returns checkpoint bytes
    /// read, which excludes exact BF16-to-FP32 scale widening.
    pub fn read_projection_once(&self, layer: usize, expert: usize, projection: Fp8Projection,
        weight: &mut [u8], scale: &mut [u8]) -> Result<usize> {
        self.check(layer, expert, projection)?;
        let name = self.name(layer, expert, projection);
        let sizes = self.projection_bytes(projection);
        ensure!((weight.len(), scale.len()) == sizes, "{name}: shared projection buffers have the wrong size");
        let w = self.located(&name)?;
        if self.format == ExpertFormat::Bf16 {
            std::fs::File::open(self.snapshot.join(&w.shard))?.read_exact_at(weight, w.offset)?;
            return Ok(weight.len());
        }
        let suffix = if self.format.packed_fp4() { "_scale" } else { "_scale_inv" };
        let s = self.located(&format!("{name}{suffix}"))?;
        let read = |located: &Located, out: &mut [u8]| -> Result<()> {
            std::fs::File::open(self.snapshot.join(&located.shard))?.read_exact_at(out, located.offset)
                .with_context(|| format!("reading shared projection {name}"))
        };
        read(w, weight)?;
        let raw_bytes = usize::try_from(s.bytes)?;
        read(s, &mut scale[..raw_bytes])?;
        if s.dtype == DType::Bf16 {
            // Widen backward in place so the pinned source needs no second grid.
            for index in (0..raw_bytes / 2).rev() {
                let value = u16::from_le_bytes([scale[2 * index], scale[2 * index + 1]]);
                scale[4 * index..4 * index + 4].copy_from_slice(&(u32::from(value) << 16).to_le_bytes());
            }
        }
        let scalars = if self.format == ExpertFormat::Nvfp4 {
            scale[sizes.1 - 8..sizes.1 - 4].copy_from_slice(&self.read_alpha(layer, expert, projection)?.to_le_bytes());
            scale[sizes.1 - 4..].copy_from_slice(&self.read_input_scale(layer, expert, projection)?.to_le_bytes());
            8
        } else { 0 };
        Ok(weight.len() + raw_bytes + scalars)
    }

    /// Weight and scale-grid windows of one shared source projection. NVFP4's
    /// two replicated scalars are separate from the matrix windows.
    pub fn projection_slices(&self, projection: Fp8Projection, tp: usize, rank: usize, slicing: Slicing)
        -> Result<[Fp8MatrixSlice; 2]> {
        let (rows, cols) = self.dims(projection);
        let (first, len) = self.rank_range_with(tp, rank, slicing)?;
        let stored = self.rank_width(tp, rank, slicing)?;
        let bytes = self.slice_bytes_with(projection, tp, rank, slicing)?;
        let matrix = |row_unit: usize, column_unit: usize, element: usize, destination_bytes| {
            let source_pitch = cols.div_ceil(column_unit) * element;
            if projection == Fp8Projection::Down {
                Fp8MatrixSlice { source_offset: first / column_unit * element, source_pitch,
                    destination_pitch: stored.div_ceil(column_unit) * element,
                    width: len / column_unit * element, rows: rows.div_ceil(row_unit), destination_bytes }
            } else {
                Fp8MatrixSlice { source_offset: first / row_unit * source_pitch, source_pitch,
                    destination_pitch: source_pitch, width: source_pitch, rows: len / row_unit, destination_bytes }
            }
        };
        Ok(if self.format == ExpertFormat::Bf16 {
            [matrix(1, 1, 2, bytes.0), Fp8MatrixSlice { source_offset: 0, source_pitch: 0,
                destination_pitch: 0, width: 0, rows: 0, destination_bytes: 0 }]
        } else if self.format.packed_fp4() {
            let mut scale = matrix(1, self.format.group(), 1, bytes.1);
            if self.format == ExpertFormat::Mxfp4 && projection == Fp8Projection::Down {
                scale.destination_pitch = (stored / 32).div_ceil(4) * 4;
            }
            [matrix(1, 2, 1, bytes.0), scale]
        } else {
            [matrix(1, 1, 1, bytes.0), matrix(128, 128, 4, bytes.1)]
        })
    }

    /// `read_slice` for packed FP4 (MXFP4, NVFP4): rank rows `[first, first +
    /// len)` of gate/up (rows) or down (K columns), zero-padded to the stored
    /// slice width.
    #[allow(clippy::too_many_arguments)]
    fn read_mxfp4_slice(&self, name: &str, projection: Fp8Projection, tp: usize, rank: usize, slicing: Slicing,
        weight: &mut [u8], scale: &mut [u8], staging: &mut Vec<u8>) -> Result<()> {
        let (rows, cols) = self.dims(projection);
        let slice = self.rank_width(tp, rank, slicing)?;
        let (first, len) = self.rank_range_with(tp, rank, slicing)?;
        let w = self.located(name)?;
        let s = self.located(&format!("{name}_scale"))?;
        let open = |shard: &str| std::fs::File::open(self.snapshot.join(shard));
        let (w_file, s_file) = (open(&w.shard)?, open(&s.shard)?);
        weight.fill(0);
        scale.fill(0);
        if projection == Fp8Projection::Down {
            // K columns [first, first + len) of every row, into slice-wide rows.
            let group = self.format.group();
            for (located, file, out, per) in [(w, &w_file, &mut *weight, 2usize), (s, &s_file, &mut *scale, group)] {
                staging.resize(located.bytes as usize, 0);
                file.read_exact_at(staging, located.offset).with_context(|| format!("reading {name}"))?;
                let row_out = if per == 32 { (slice / per).div_ceil(4) * 4 } else { slice / per };
                let (row_in, take) = (cols / per, len / per);
                for (row, out) in out.chunks_exact_mut(row_out).enumerate().take(rows) {
                    out[..take].copy_from_slice(&staging[row * row_in + first / per..][..take]);
                }
            }
        } else {
            let (w_row, s_row) = (cols / 2, cols / self.format.group());
            w_file.read_exact_at(&mut weight[..len * w_row], w.offset + (first * w_row) as u64)
                .with_context(|| format!("reading {name}"))?;
            s_file.read_exact_at(&mut scale[..len * s_row], s.offset + (first * s_row) as u64)
                .with_context(|| format!("reading {name}_scale"))?;
        }
        Ok(())
    }

    /// Reads rank `rank`'s slice of one expert projection: the E4M3 weight
    /// (gate/up `[slice, H]`, down `[H, slice]`, zero past the rank's range) and its FP32 block scales.
    /// `staging` is reused scratch for the down projection's column window.
    #[allow(clippy::too_many_arguments)]
    pub fn read_slice(&self, layer: usize, expert: usize, projection: Fp8Projection, tp: usize, rank: usize,
        weight: &mut [u8], scale: &mut [u8], staging: &mut Vec<u8>) -> Result<()> {
        self.read_slice_with(layer, expert, projection, tp, rank, Slicing::Padded, weight, scale, staging)
    }

    /// [`Self::read_slice`] under `slicing`.
    #[allow(clippy::too_many_arguments)]
    pub fn read_slice_with(&self, layer: usize, expert: usize, projection: Fp8Projection, tp: usize, rank: usize,
        slicing: Slicing, weight: &mut [u8], scale: &mut [u8], staging: &mut Vec<u8>) -> Result<()> {
        ensure!(rank < tp, "rank {rank} of TP{tp}");
        self.check(layer, expert, projection)?;
        let name = self.name(layer, expert, projection);
        let (rows, cols) = self.dims(projection);
        let slice = self.rank_width(tp, rank, slicing)?;
        let (weight_bytes, scale_bytes) = self.slice_bytes_with(projection, tp, rank, slicing)?;
        ensure!(weight.len() == weight_bytes && scale.len() == scale_bytes, "{name}: slice buffers of the wrong size");
        if self.format == ExpertFormat::Bf16 {
            let w = self.located(&name)?;
            let file = std::fs::File::open(self.snapshot.join(&w.shard))?;
            let (first, len) = self.rank_range_with(tp, rank, slicing)?;
            weight.fill(0);
            if projection == Fp8Projection::Down {
                for row in 0..rows {
                    let offset = w.offset + ((row * cols + first) * 2) as u64;
                    file.read_exact_at(&mut weight[row * slice * 2..][..len * 2], offset)
                        .with_context(|| format!("reading BF16 {name} row {row}"))?;
                }
            } else {
                file.read_exact_at(&mut weight[..len * cols * 2], w.offset + (first * cols * 2) as u64)
                    .with_context(|| format!("reading BF16 {name}"))?;
            }
            return Ok(());
        }
        if self.format.packed_fp4() {
            return self.read_mxfp4_slice(&name, projection, tp, rank, slicing, weight, scale, staging);
        }
        let w = self.located(&name)?;
        let s = self.located(&format!("{name}_scale_inv"))?;
        let open = |shard: &str| std::fs::File::open(self.snapshot.join(shard));
        let (w_file, s_file) = (open(&w.shard)?, open(&s.shard)?);
        let (first, len) = self.rank_range_with(tp, rank, slicing)?;
        let (scale_rows, scale_cols) = (rows.div_ceil(128), cols.div_ceil(128));
        // The whole block-scale grid (at most a few hundred entries), widened to
        // FP32: Qwen 3.8 Flash Next stores BF16 scales, which FP32 holds exactly.
        let mut raw = vec![0u8; s.bytes as usize];
        s_file.read_exact_at(&mut raw, s.offset)?;
        let grid: Vec<u8> = match s.dtype {
            DType::Bf16 => raw.chunks_exact(2)
                .flat_map(|b| f32::from_bits(u32::from(u16::from_le_bytes([b[0], b[1]])) << 16).to_le_bytes())
                .collect(),
            _ => raw,
        };
        ensure!(grid.len() == scale_rows * scale_cols * 4, "{name}_scale_inv: unexpected grid size");
        // A rank owning fewer blocks than the stored slice keeps zero padding
        // past `len` (zero weights, zero scales).
        if len < slice {
            weight.fill(0);
            scale.fill(0);
        }
        if projection == Fp8Projection::Down {
            // Column window [first, first + len) of every row: read the whole
            // tensor once and gather (down is a third of an expert's bytes).
            staging.resize(w.bytes as usize, 0);
            w_file.read_exact_at(staging, w.offset).with_context(|| format!("reading {name}"))?;
            for (row, out) in weight.chunks_exact_mut(slice).enumerate() {
                out[..len].copy_from_slice(&staging[row * cols + first..][..len]);
            }
            let (window, taken, at) = (slice / 128 * 4, len / 128 * 4, first / 128 * 4);
            for (row, out) in scale.chunks_exact_mut(window).enumerate().take(scale_rows) {
                out[..taken].copy_from_slice(&grid[row * scale_cols * 4 + at..][..taken]);
            }
        } else {
            w_file.read_exact_at(&mut weight[..len * cols], w.offset + (first * cols) as u64)
                .with_context(|| format!("reading {name}"))?;
            let (at, taken) = (first / 128 * scale_cols * 4, len / 128 * scale_cols * 4);
            scale[..taken].copy_from_slice(&grid[at..at + taken]);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn package_names_follow_native_math_not_family() {
        for geometry in ["glm", "glmf", "qwen4", "step5", "future"] {
            assert_eq!(ExpertFormat::Nvfp4.package_name(geometry), format!("fp8-{geometry}-nvfp4a4"));
            assert_eq!(ExpertFormat::Fp8Block128.package_name(geometry), format!("fp8-{geometry}"));
            assert_eq!(ExpertFormat::Mxfp4.package_name(geometry), format!("fp8-{geometry}"));
            assert_eq!(ExpertFormat::Bf16.package_name(geometry), format!("fp8-{geometry}-bf16-weights"));
        }
    }

    fn catalog(format: ExpertFormat, intermediate: usize) -> Fp8ExpertTensors {
        let shape = RoutedExpertShape { layers: 78, first_layer: 3, experts: 256, topk: 8, hidden: 6144, intermediate,
            draft_stages: 0, draft_experts: 0 };
        Fp8ExpertTensors { snapshot: PathBuf::new(), shape, prefix: "model.".into(), mtp_layers: false, format,
            tensors: Arc::new(HashMap::new()), activations: Vec::new(), slice_layer: 0, coordinator: false }
    }

    fn ranges(tensors: &Fp8ExpertTensors, tp: usize) -> Vec<(usize, usize)> {
        (0..tp).map(|rank| tensors.rank_range(tp, rank).unwrap()).collect()
    }

    #[test]
    fn homogeneous_groups_preserve_layer_ids_arithmetic_and_shared_metadata() -> Result<()> {
        let mut tensors = catalog(ExpertFormat::Fp8Block128, 128);
        tensors.shape.experts = 1;
        tensors.shape.hidden = 128;
        tensors.activations = vec![cuteafd_core::FfnActivation::SiluPostClampFp32Product { limit: 7. }; 78];
        tensors.activations[5] = cuteafd_core::FfnActivation::SiluPostClampFp32Product { limit: 10. };
        for layer in 3..8 {
            for projection in Fp8Projection::ALL {
                let name = tensors.name(layer, 0, projection);
                Arc::make_mut(&mut tensors.tensors).insert(name.clone(), Located {
                    shard: "headers-only.safetensors".into(), offset: 0, bytes: 128*128,
                    dtype: if layer == 6 { DType::Bf16 } else { DType::F8E4M3 }, shape: vec![128, 128],
                });
                if layer != 6 {
                    Arc::make_mut(&mut tensors.tensors).insert(format!("{name}_scale_inv"), Located {
                        shard: "headers-only.safetensors".into(), offset: 0, bytes: 4,
                        dtype: DType::F32, shape: vec![1, 1],
                    });
                }
            }
        }
        let groups = tensors.homogeneous_layers(&[3, 4, 5, 6, 7])?;
        assert_eq!(groups.iter().map(|(_, ids)| ids.clone()).collect::<Vec<_>>(),
            vec![vec![3, 4], vec![5], vec![6], vec![7]]);
        assert_eq!(groups[2].0.format(), ExpertFormat::Bf16);
        assert!(groups.iter().all(|(view, _)| Arc::ptr_eq(&view.tensors, &tensors.tensors)));
        assert_eq!(tensors.homogeneous_layers(&[3, 7])?.len(), 2);
        assert!(tensors.homogeneous_layers(&[]).is_err());
        assert!(tensors.homogeneous_layers(&[4, 3]).is_err());
        assert!(tensors.homogeneous_layers(&[3, 3]).is_err());
        Ok(())
    }

    #[test]
    fn native_projection_load_owners_follow_format_and_direct_bf16_slices() -> Result<()> {
        for format in [ExpertFormat::Bf16, ExpertFormat::Fp8Block128,
            ExpertFormat::Mxfp4, ExpertFormat::Nvfp4] {
            let mut tensors = catalog(format, 1536);
            tensors.shape.hidden = 4096;
            tensors.shape.experts = 352;
            for projection in Fp8Projection::ALL {
                let (rows, cols) = tensors.dims(projection);
                let full = tensors.projection_bytes(projection);
                for expert in 0..352 {
                    let name = tensors.name(3, expert, projection);
                    Arc::make_mut(&mut tensors.tensors).insert(name.clone(), Located {
                        shard: "headers-only.safetensors".into(), offset: 0, bytes: full.0 as u64,
                        dtype: if format == ExpertFormat::Bf16 { DType::Bf16 } else { DType::U8 },
                        shape: vec![rows, cols],
                    });
                    if format != ExpertFormat::Bf16 {
                        let suffix = if format.packed_fp4() { "_scale" } else { "_scale_inv" };
                        // Include the BF16 -> FP32 grid owner's overlapping lifetime.
                        let raw = if format == ExpertFormat::Fp8Block128 { full.1 / 2 }
                            else { full.1 - if format == ExpertFormat::Nvfp4 { 8 } else { 0 } };
                        Arc::make_mut(&mut tensors.tensors).insert(format!("{name}{suffix}"), Located {
                            shard: "headers-only.safetensors".into(), offset: 0, bytes: raw as u64,
                            dtype: if format == ExpertFormat::Fp8Block128 { DType::Bf16 } else { DType::U8 },
                            shape: vec![],
                        });
                    }
                }
            }
            for tp in [1, 4, 6] {
                for rank in 0..tp {
                    for slicing in [Slicing::Padded, Slicing::Blocks(128)] {
                        let load = tensors.layer_load(3, tp, rank, slicing)?;
                        assert_eq!(load.resident,
                            tensors.weight_read_bytes_with(tp, rank, slicing)? * 352);
                        for projection in Fp8Projection::ALL {
                            let plan = tensors.projection_load(3, projection, tp, rank, slicing)?;
                            let (weights, _) = tensors.slice_bytes_with(projection, tp, rank, slicing)?;
                            assert_eq!(plan.weights, weights * 352);
                            assert_eq!(plan.scales, tensors.scale_region_bytes_with(projection, tp, rank, slicing)?);
                            assert_eq!(plan.readers, 16);
                            assert_eq!(plan.reader_staging, if projection == Fp8Projection::Down
                                && format != ExpertFormat::Bf16 { tensors.projection_bytes(projection).0 } else { 0 });
                            assert!(load.host >= plan.host_bytes()?);
                            assert!(load.pinned >= plan.pinned_upload());
                            if format == ExpertFormat::Bf16 {
                                assert_eq!(plan.reader_auxiliary, 0);
                                assert_eq!(plan.host_bytes()?, plan.weights);
                                assert_eq!(plan.pinned_upload(), plan.weights);
                            }
                        }
                        if format == ExpertFormat::Bf16 && tp == 6 {
                            assert_eq!(load.pinned, 738_197_504);
                            assert_eq!(load.host, load.pinned);
                        }
                    }
                }
            }
        }
        assert!(Fp8ProjectionLoad { weights: usize::MAX, scales: 1, readers: 1,
            reader_staging: 0, reader_auxiliary: 0, gate_inputs: 0 }.host_bytes().is_err());
        Ok(())
    }

    #[test]
    fn installed_read_bytes_follow_exact_rank_widths_and_checkpoint_scales() {
        for (format, divisor, group, scalar) in [
            (ExpertFormat::Fp8Block128, 1, 128, 0),
            (ExpertFormat::Mxfp4, 2, 32, 0),
            (ExpertFormat::Nvfp4, 2, 16, 8),
        ] {
            let tensors = catalog(format, 2048);
            let hidden = tensors.shape().hidden;
            let widths = [384, 384, 384, 384, 256, 256];
            for (rank, width) in widths.into_iter().enumerate() {
                let weights = hidden * width / divisor;
                let scales = if format == ExpertFormat::Fp8Block128 {
                    hidden.div_ceil(group) * width.div_ceil(group) * 4
                } else { hidden * width / group };
                assert_eq!(tensors.weight_read_bytes_with(6, rank, Slicing::Blocks(128)).unwrap(),
                    3 * (weights + scales + scalar));
            }
            let padded = tensors.weight_read_bytes_with(6, 5, Slicing::Padded).unwrap();
            assert_eq!(padded, tensors.weight_read_bytes_with(6, 0, Slicing::Blocks(128)).unwrap());
            assert!(padded > tensors.weight_read_bytes_with(6, 5, Slicing::Blocks(128)).unwrap());
            assert!(tensors.weight_read_bytes_with(6, 6, Slicing::Blocks(128)).is_err());
        }
    }

    #[test]
    fn stacked_expert_views_preserve_tp6_tp4_weights_and_scale_windows() -> Result<()> {
        use crate::plan::testing::{t, write_snapshot};
        let dir = tempfile::tempdir()?;
        let shape = RoutedExpertShape { layers: 2, first_layer: 0, experts: 2, topk: 1,
            hidden: 128, intermediate: 1536, draft_stages: 0, draft_experts: 0 };
        let mut fixture = Vec::new();
        for layer in 0..2 {
            for projection in Fp8Projection::ALL {
                let (rows, cols) = if projection == Fp8Projection::Down { (128,1536) } else { (1536,128) };
                let name = format!("model.layers.{layer}.moe.{}.weight", projection.stem());
                fixture.push(t(&name, if layer == 0 { "F8_E4M3" } else { "BF16" }, &[2,rows,cols]));
                if layer == 0 { fixture.push(t(format!("{name}_scale_inv"), "F32", &[2,rows/128,cols/128])); }
            }
        }
        write_snapshot(dir.path(), &serde_json::json!({}), &fixture, None);
        let tensors = Fp8ExpertTensors::read(dir.path(), shape)?;
        assert_eq!(tensors.layer_format(0)?, ExpertFormat::Fp8Block128);
        assert_eq!(tensors.layer_format(1)?, ExpertFormat::Bf16);
        for layer in 0..2 {
            let view = tensors.for_layer(layer)?;
            for projection in Fp8Projection::ALL {
                for expert in 0..2 {
                    let located = view.located(&view.name(layer, expert, projection))?;
                    let file = std::fs::File::options().write(true).open(dir.path().join(&located.shard))?;
                    let bytes = (0..located.bytes as usize).map(|i| ((i*17 + i/13 + expert*59)%251) as u8).collect::<Vec<_>>();
                    file.write_all_at(&bytes, located.offset)?;
                    if layer == 0 {
                        let scale = view.located(&format!("{}_scale_inv", view.name(layer,expert,projection)))?;
                        let bytes = (0..scale.bytes as usize).map(|i| (i + expert*71) as u8).collect::<Vec<_>>();
                        file.write_all_at(&bytes, scale.offset)?;
                    }
                }
                for expert in 0..2 {
                    let full = view.projection_bytes(projection);
                    let (mut w,mut s) = (vec![0;full.0],vec![0;full.1]);
                    assert_eq!(view.read_projection_once(layer,expert,projection,&mut w,&mut s)?,full.0+full.1);
                    for (tp,width) in [(6,256),(4,384)] {
                        for rank in 0..tp {
                            assert_eq!(view.rank_range(tp,rank)?,(rank*width,width));
                            let sizes = view.slice_bytes_with(projection,tp,rank,Slicing::Padded)?;
                            if layer == 1 { assert_eq!(sizes.1,0); }
                            let (mut rw,mut rs) = (vec![0;sizes.0],vec![0;sizes.1]);
                            view.read_slice_with(layer,expert,projection,tp,rank,Slicing::Padded,&mut rw,&mut rs,&mut Vec::new())?;
                            for (window,source,actual) in [(view.projection_slices(projection,tp,rank,Slicing::Padded)?[0],&w,rw),
                                (view.projection_slices(projection,tp,rank,Slicing::Padded)?[1],&s,rs)] {
                                let mut expected = vec![0;window.destination_bytes];
                                for row in 0..window.rows {
                                    expected[row*window.destination_pitch..][..window.width].copy_from_slice(
                                        &source[window.source_offset+row*window.source_pitch..][..window.width]);
                                }
                                assert_eq!(actual,expected,"layer {layer} expert {expert} {projection:?} TP{tp} rank {rank}");
                            }
                        }
                    }
                }
            }
        }
        // Leading-axis mismatches fail before a payload read can alias another expert.
        fixture[0].2[0] = 3;
        write_snapshot(dir.path(), &serde_json::json!({}), &fixture, None);
        assert!(Fp8ExpertTensors::read(dir.path(),shape).unwrap_err().to_string().contains("expected expert axis 2"));
        Ok(())
    }

    #[test]
    fn bf16_layer_requires_native_matrices_without_scales() {
        let mut tensors = catalog(ExpertFormat::Bf16, 2048);
        tensors.shape.experts = 1;
        for projection in Fp8Projection::ALL {
            let (rows, cols) = tensors.dims(projection);
            let name = tensors.name(78, 0, projection);
            Arc::make_mut(&mut tensors.tensors).insert(name, Located { shard:"nextn.safetensors".into(),
                offset:0, bytes:(rows * cols * 2) as u64, dtype:DType::Bf16, shape:vec![rows,cols] });
        }
        let view = tensors.for_layer(78).unwrap();
        assert_eq!(view.format(), ExpertFormat::Bf16);
        assert_eq!(view.slice_bytes(Fp8Projection::Gate, 4).unwrap(), (512 * 6144 * 2, 0));
        let name = format!("{}_scale_inv", tensors.name(78, 0, Fp8Projection::Gate));
        Arc::make_mut(&mut tensors.tensors).insert(name, Located { shard:"nextn.safetensors".into(), offset:0,
            bytes:4, dtype:DType::F32, shape:vec![1] });
        assert!(tensors.for_layer(78).unwrap_err().to_string().contains("must not have quantization scales"));
    }

    #[test]
    fn mtp_only_reads_its_format_and_ignores_missing_or_corrupt_backbone() {
        use crate::plan::testing::{t, write_safetensors};
        for nvfp4 in [false, true] {
            let dir = tempfile::tempdir().unwrap();
            std::fs::write(dir.path().join("config.json"), b"{}").unwrap();
            let shape = RoutedExpertShape { layers: 2, first_layer: 0, experts: 1, topk: 1,
                hidden: 128, intermediate: 128, draft_stages: 0, draft_experts: 0 };
            let mut map = BTreeMap::new();
            let mut tensors = Vec::new();
            for p in Fp8Projection::ALL {
                let name = format!("mtp.layers.0.mlp.experts.0.{}.weight", p.stem());
                if nvfp4 {
                    tensors.extend([t(&name, "U8", &[128, 64]), t(format!("{name}_scale"), "F8_E4M3", &[128, 8]),
                        t(format!("{name}_scale_2"), "F32", &[]),
                        t(format!("{}.input_scale", name.strip_suffix(".weight").unwrap()), "F32", &[])]);
                } else {
                    tensors.extend([t(&name, "F8_E4M3", &[128, 128]), t(format!("{name}_scale_inv"), "F32", &[1, 1])]);
                }
            }
            for tensor in &tensors { map.insert(tensor.0.clone(), "mtp.safetensors"); }
            map.insert("model.language_model.layers.0.mlp.experts.0.gate_proj.weight".into(), "backbone.safetensors");
            std::fs::write(dir.path().join("model.safetensors.index.json"),
                serde_json::to_vec(&serde_json::json!({"weight_map": map})).unwrap()).unwrap();
            write_safetensors(&dir.path().join("mtp.safetensors"), &tensors);
            let layers = std::collections::BTreeSet::from([2]);
            let sliced = Fp8ExpertTensors::read_layers(dir.path(), shape, &layers).unwrap();
            assert_eq!(sliced.format(), if nvfp4 { ExpertFormat::Nvfp4 } else { ExpertFormat::Fp8Block128 });
            sliced.validate_layer(2).unwrap();
            std::fs::write(dir.path().join("backbone.safetensors"), b"corrupt").unwrap();
            Fp8ExpertTensors::read_layers(dir.path(), shape, &layers).unwrap().validate_layer(2).unwrap();
            std::fs::remove_file(dir.path().join("mtp.safetensors")).unwrap();
            let error = Fp8ExpertTensors::read_layers(dir.path(), shape, &layers).unwrap_err().to_string();
            assert!(error.contains("mtp.layers.0") && error.contains("mtp.safetensors"), "{error}");
        }
    }

    #[test]
    fn resident_headers_reject_mixed_format_mtp_before_weight_reads() {
        let mut tensors = catalog(ExpertFormat::Nvfp4, 128);
        tensors.shape.hidden = 128;
        tensors.shape.experts = 1;
        tensors.mtp_layers = true;
        for projection in Fp8Projection::ALL {
            let name = tensors.name(3, 0, projection);
            for (name, dtype, shape, bytes) in [
                (name.clone(), DType::U8, vec![128, 64], 8192),
                (format!("{name}_scale"), DType::F8E4M3, vec![128, 8], 1024),
                (format!("{name}_scale_2"), DType::F32, vec![], 4),
                (format!("{}.input_scale", name.strip_suffix(".weight").unwrap()), DType::F32, vec![], 4),
            ] {
                Arc::make_mut(&mut tensors.tensors).insert(name, Located { shard: "not-read.safetensors".into(), offset: 0,
                    bytes, dtype, shape });
            }
        }
        tensors.validate_layer(3).unwrap();
        assert_eq!(tensors.layer_format(3).unwrap(), ExpertFormat::Nvfp4);
        let gate = tensors.name(3, 0, Fp8Projection::Gate);
        let input = format!("{}.input_scale", gate.strip_suffix(".weight").unwrap());
        let scale = Arc::make_mut(&mut tensors.tensors).remove(&input).unwrap();
        assert!(tensors.validate_layer(3).unwrap_err().to_string().contains(&input));
        Arc::make_mut(&mut tensors.tensors).insert(input, scale);
        let mtp = tensors.name(78, 0, Fp8Projection::Gate);
        Arc::make_mut(&mut tensors.tensors).insert(mtp.clone(), Located { shard: "not-read.safetensors".into(), offset: 0,
            bytes: 128 * 128, dtype: DType::F8E4M3, shape: vec![128, 128] });
        assert_eq!(tensors.layer_format(78).unwrap(), ExpertFormat::Fp8Block128);
        assert!(tensors.layer_format(79).is_err());
        let error = tensors.validate_layer(78).unwrap_err().to_string();
        assert!(error.contains(&mtp) && error.contains("expected packed E2M1 U8"), "{error}");
        // A separate package view accepts the complete FP8 draft, without
        // changing the backbone contract or duplicating device weights.
        for projection in Fp8Projection::ALL {
            let name = tensors.name(78, 0, projection);
            Arc::make_mut(&mut tensors.tensors).insert(name.clone(), Located { shard: "not-read.safetensors".into(), offset: 0,
                bytes: 128 * 128, dtype: DType::F8E4M3, shape: vec![128, 128] });
            Arc::make_mut(&mut tensors.tensors).insert(format!("{name}_scale_inv"), Located {
                shard: "not-read.safetensors".into(), offset: 0,
                bytes: 4, dtype: DType::F32, shape: vec![1, 1] });
        }
        let draft = tensors.for_layer(78).unwrap();
        assert_eq!(draft.format(), ExpertFormat::Fp8Block128);
        assert_eq!(tensors.format(), ExpertFormat::Nvfp4);
        assert_eq!(draft.name(78, 0, Fp8Projection::Gate), mtp);
        assert!(draft.validate_layer(3).is_err());
        assert!(tensors.validate_layer(78).is_err());
        let name = format!("{}_scale", tensors.name(3, 0, Fp8Projection::Down));
        Arc::make_mut(&mut tensors.tensors).remove(&name);
        assert!(tensors.validate_layer(3).is_err());
    }

    #[test]
    fn fp8_slices_split_whole_blocks_and_pad_to_the_widest() {
        let fp8 = catalog(ExpertFormat::Fp8Block128, 2048);
        assert_eq!(fp8.slice(4).unwrap(), 512);
        assert_eq!(ranges(&fp8, 4), [(0, 512), (512, 512), (1024, 512), (1536, 512)]);
        // TP6 of 16 blocks: 3, 3, 3, 3, 2, 2, stored 384 wide.
        assert_eq!(fp8.slice(6).unwrap(), 384);
        assert_eq!(ranges(&fp8, 6), [(0, 384), (384, 384), (768, 384), (1152, 384), (1536, 256), (1792, 256)]);
        assert_eq!(fp8.slice_bytes(Fp8Projection::Gate, 6).unwrap(), (384 * 6144, 3 * 48 * 4));
        assert_eq!(fp8.slice_bytes(Fp8Projection::Down, 6).unwrap(), (6144 * 384, 48 * 3 * 4));
        assert!(fp8.slice(17).is_err());
        // Qwen's 640 (5 blocks) over 3: 2, 2, 1 in 256.
        let qwen = catalog(ExpertFormat::Fp8Block128, 640);
        assert_eq!(ranges(&qwen, 3), [(0, 256), (256, 256), (512, 128)]);
        assert_eq!(qwen.slice(3).unwrap(), 256);
    }

    #[test]
    fn nvfp4_slices_keep_their_16_blocks() {
        let nvfp4 = catalog(ExpertFormat::Nvfp4, 2048);
        // TP6 of 128 blocks: 22, 22, 21, 21, 21, 21 stored 384 wide.
        assert_eq!(nvfp4.slice(6).unwrap(), 384);
        assert_eq!(ranges(&nvfp4, 6), [(0, 352), (352, 352), (704, 336), (1040, 336), (1376, 336), (1712, 336)]);
        assert_eq!(nvfp4.slice_bytes(Fp8Projection::Gate, 6).unwrap(), (384 * 6144 / 2, 384 * 6144 / 16));
        assert_eq!(nvfp4.slice_bytes(Fp8Projection::Down, 1).unwrap(), (6144 * 1024, 6144 * 128));
        assert_eq!(nvfp4.scale_region_bytes(Fp8Projection::Down, 1).unwrap(), 256 * 6144 * 128 + 256 * 8);
        // Qwen's 640 = 40 blocks: TP3 14, 13, 13 in 256.
        let qwen = catalog(ExpertFormat::Nvfp4, 640);
        assert_eq!(ranges(&qwen, 3), [(0, 224), (224, 208), (432, 208)]);
        assert_eq!(qwen.slice(3).unwrap(), 256);
        assert_eq!(qwen.slice(2).unwrap(), 384);
    }

    /// Every expert's gate and up `input_scale` bytes, read the way the loader
    /// packs them, handed to `check`.
    fn input_scales<T>(tensors: &Fp8ExpertTensors, check: impl Fn(&[Vec<u8>; 2]) -> T) -> T {
        let read = |projection| (0..tensors.shape().experts)
            .flat_map(|expert| tensors.read_input_scale(0, expert, projection).unwrap().to_le_bytes()).collect();
        check(&[read(Fp8Projection::Gate), read(Fp8Projection::Up)])
    }

    /// A one-layer, `experts`-expert NVFP4 snapshot (`hidden` = `intermediate`
    /// = 128), on disk so the guard reads real `input_scale` bytes.
    fn nvfp4_snapshot(dir: &Path, experts: usize) -> Fp8ExpertTensors {
        use crate::plan::testing::{t, write_snapshot};
        let (hidden, intermediate) = (128usize, 128usize);
        let mut tensors = Vec::new();
        for expert in 0..experts {
            for projection in Fp8Projection::ALL {
                let name = format!("model.layers.0.mlp.experts.{expert}.{}.weight", projection.stem());
                tensors.extend([
                    t(&name, "U8", &[intermediate, hidden / 2]),
                    t(format!("{name}_scale"), "F8_E4M3", &[intermediate, hidden / 16]),
                    t(format!("{name}_scale_2"), "F32", &[]),
                    t(input_scale_name(&name), "F32", &[]),
                ]);
            }
        }
        write_snapshot(dir, &serde_json::json!({}), &tensors, None);
        let shape = RoutedExpertShape { layers: 1, first_layer: 0, experts, topk: 1, hidden, intermediate,
            draft_stages: 0, draft_experts: 0 };
        Fp8ExpertTensors::read(dir, shape).unwrap()
    }

    /// Overwrites every expert's gate/up `input_scale` in the snapshot's shard.
    fn write_input_scales(tensors: &Fp8ExpertTensors, scales: impl Fn(usize, Fp8Projection) -> f32) {
        let shard = tensors.tensors.values().next().unwrap().shard.clone();
        let path = tensors.snapshot.join(&shard);
        let metas = read_safetensors_metadata(&path).unwrap();
        let file = std::fs::File::options().write(true).open(&path).unwrap();
        for expert in 0..tensors.shape.experts {
            for projection in [Fp8Projection::Gate, Fp8Projection::Up] {
                let name = input_scale_name(&tensors.name(0, expert, projection));
                let at = metas.iter().find(|m| m.name == name).unwrap().byte_offset;
                file.write_all_at(&scales(expert, projection).to_le_bytes(), at).unwrap();
            }
        }
    }

    #[test]
    fn nvfp4_gate_up_input_scales_that_agree_load() {
        let dir = tempfile::tempdir().unwrap();
        let tensors = nvfp4_snapshot(dir.path(), 2);
        assert_eq!(tensors.format(), ExpertFormat::Nvfp4);
        // Distinct per expert, gate == up in each: the guard accepts the layer.
        write_input_scales(&tensors, |expert, _| if expert == 0 { 0.25 } else { 1.5 });
        input_scales(&tensors, |t| Fp8ExpertTensors::check_input_scales(0, &t[0], &t[1])).unwrap();
        tensors.validate_layer(0).unwrap();
    }

    #[test]
    fn nvfp4_asymmetric_gate_up_input_scales_fail_the_load() {
        let dir = tempfile::tempdir().unwrap();
        let tensors = nvfp4_snapshot(dir.path(), 2);
        // Expert 0 agrees; expert 1's up half is calibrated at half the gate's.
        write_input_scales(&tensors, |expert, projection| match (expert, projection) {
            (1, Fp8Projection::Up) => 0.5,
            _ => 1.0,
        });
        let error = input_scales(&tensors, |t| Fp8ExpertTensors::check_input_scales(0, &t[0], &t[1])).unwrap_err().to_string();
        assert!(error.contains("layer 0") && error.contains("expert 1"), "{error}");
        assert!(error.contains("gate input_scale 1") && error.contains("up input_scale 0.5"), "{error}");
        assert!(!error.contains("expert 0"), "{error}");
    }

    #[test]
    fn shared_projection_windows_match_rank_reads_for_every_format_and_padding() -> Result<()> {
        use crate::plan::testing::{fp8, mxfp4, nvfp4, write_snapshot};
        for (format, bf16) in [(ExpertFormat::Fp8Block128, false), (ExpertFormat::Fp8Block128, true),
            (ExpertFormat::Mxfp4, false), (ExpertFormat::Nvfp4, false), (ExpertFormat::Bf16, false)] {
            let dir = tempfile::tempdir()?;
            let shape = RoutedExpertShape { layers: 1, first_layer: 0, experts: 2, topk: 1, hidden: 256,
                intermediate: 640, draft_stages: 0, draft_experts: 0 };
            let mut fixtures = Vec::new();
            for expert in 0..shape.experts {
                for (projection, n, k) in [("gate", shape.intermediate, shape.hidden),
                    ("up", shape.intermediate, shape.hidden), ("down", shape.hidden, shape.intermediate)] {
                    let name = format!("model.layers.0.mlp.experts.{expert}.{projection}_proj");
                    let mut tensors = match format {
                        ExpertFormat::Fp8Block128 => fp8(&name, n, k, None),
                        ExpertFormat::Mxfp4 => mxfp4(&name, n, k),
                        ExpertFormat::Nvfp4 => nvfp4(&name, n, k),
                        ExpertFormat::Bf16 => vec![crate::plan::testing::t(format!("{name}.weight"), "BF16", &[n,k])],
                    };
                    if bf16 { tensors[1].1 = "BF16"; }
                    fixtures.extend(tensors);
                }
            }
            write_snapshot(dir.path(), &serde_json::json!({}), &fixtures, None);
            let tensors = Fp8ExpertTensors::read(dir.path(), shape)?;
            for (name, located) in tensors.tensors.iter() {
                let bytes = if name.ends_with("_scale_2") || name.ends_with(".input_scale") {
                    0.25f32.to_le_bytes().to_vec()
                } else {
                    (0..located.bytes as usize).map(|i| ((i * 31 + i / 7 + 11) % 251) as u8).collect()
                };
                std::fs::OpenOptions::new().write(true).open(dir.path().join(&located.shard))?
                    .write_all_at(&bytes, located.offset)?;
            }
            for expert in 0..shape.experts {
                for projection in Fp8Projection::ALL {
                    let sizes = tensors.projection_bytes(projection);
                    let mut full_weight = vec![0; sizes.0];
                    let mut full_scale = vec![0; sizes.1];
                    let read = tensors.read_projection_once(0, expert, projection, &mut full_weight, &mut full_scale)?;
                    let expected = if bf16 { sizes.0 + sizes.1 / 2 } else { sizes.0 + sizes.1 };
                    assert_eq!(read, expected);
                    for slicing in [Slicing::Padded, Slicing::Blocks(128)] {
                        for rank in 0..2 {
                            let sizes = tensors.slice_bytes_with(projection, 2, rank, slicing)?;
                            let mut weight = vec![0; sizes.0];
                            let mut scale = vec![0; sizes.1];
                            tensors.read_slice_with(0, expert, projection, 2, rank, slicing,
                                &mut weight, &mut scale, &mut Vec::new())?;
                            let windows = tensors.projection_slices(projection, 2, rank, slicing)?;
                            for (window, source, expected) in [(windows[0], &full_weight, weight),
                                (windows[1], &full_scale, scale)] {
                                let mut destination = vec![0; window.destination_bytes];
                                for row in 0..window.rows {
                                    let at = window.source_offset + row * window.source_pitch;
                                    destination[row * window.destination_pitch..][..window.width]
                                        .copy_from_slice(&source[at..at + window.width]);
                                }
                                assert_eq!(destination, expected, "{format:?} {projection:?} {slicing:?} rank {rank}");
                            }
                        }
                    }
                    if format == ExpertFormat::Nvfp4 {
                        assert_eq!(&full_scale[full_scale.len() - 8..], &[0, 0, 128, 62, 0, 0, 128, 62]);
                    }
                    assert!(tensors.read_projection_once(0, expert, projection, &mut [], &mut full_scale).is_err());
                }
            }
        }
        Ok(())
    }

    #[test]
    fn balanced_k32_slices_reconstruct_source_bytes_and_pad_scale_rows() -> Result<()> {
        use crate::plan::testing::{mxfp4, write_snapshot};
        let dir = tempfile::tempdir()?;
        let shape = RoutedExpertShape { layers: 1, first_layer: 0, experts: 2, topk: 1,
            hidden: 256, intermediate: 2048, draft_stages: 0, draft_experts: 0 };
        let mut fixtures = Vec::new();
        for expert in 0..shape.experts {
            for (projection, n, k) in [("gate", 2048, 256), ("up", 2048, 256), ("down", 256, 2048)] {
                fixtures.extend(mxfp4(&format!("model.layers.0.mlp.experts.{expert}.{projection}_proj"), n, k));
            }
        }
        write_snapshot(dir.path(), &serde_json::json!({}), &fixtures, None);
        let mut tensors = Fp8ExpertTensors::read(dir.path(), shape)?;
        for located in tensors.tensors.values() {
            let bytes = (0..located.bytes as usize).map(|i| ((i * 17 + i / 13 + 7) % 251) as u8).collect::<Vec<_>>();
            std::fs::OpenOptions::new().write(true).open(dir.path().join(&located.shard))?
                .write_all_at(&bytes, located.offset)?;
        }
        let slicing = Slicing::Blocks(32);
        for (tp, scale_pitch) in [(3, 24), (6, 12)] {
            assert_eq!(tensors.exact_slicing(tp), slicing);
            for layer in [0, 3, 40, 77] {
                tensors.slice_layer = layer;
                for expert in 0..shape.experts {
                    for projection in Fp8Projection::ALL {
                        let (w, s) = tensors.projection_bytes(projection);
                        let mut weights = vec![0; w];
                        let mut scales = vec![0; s];
                        tensors.read_projection_once(0, expert, projection, &mut weights, &mut scales)?;
                        let mut reconstructed = [vec![0; w], vec![0; s]];
                        for rank in 0..tp {
                            let sizes = tensors.slice_bytes_with(projection, tp, rank, slicing)?;
                            let mut weight = vec![0; sizes.0];
                            let mut scale = vec![0; sizes.1];
                            tensors.read_slice_with(0, expert, projection, tp, rank, slicing,
                                &mut weight, &mut scale, &mut Vec::new())?;
                            let windows = tensors.projection_slices(projection, tp, rank, slicing)?;
                            if projection == Fp8Projection::Down {
                                assert_eq!(windows[1].destination_pitch, scale_pitch);
                                assert_eq!(scale.len(), 256 * scale_pitch);
                            }
                            for (index, (window, actual, source)) in windows.into_iter()
                                .zip([weight, scale]).zip([&weights, &scales])
                                .map(|((window, actual), source)| (window, actual, source)).enumerate() {
                                for row in 0..window.rows {
                                    let src = window.source_offset + row * window.source_pitch;
                                    let dst = row * window.destination_pitch;
                                    assert_eq!(&actual[dst..dst + window.width], &source[src..src + window.width]);
                                    assert!(actual[dst + window.width..dst + window.destination_pitch].iter().all(|&b| b == 0));
                                    reconstructed[index][src..src + window.width].copy_from_slice(&actual[dst..dst + window.width]);
                                }
                            }
                        }
                        assert_eq!(reconstructed, [weights, scales]);
                    }
                }
            }
        }
        Ok(())
    }

    #[test]
    fn qwen_coordinator_keeps_fixed_h128_halves_while_sparks_use_native_k16() -> Result<()> {
        let tensors = catalog(ExpertFormat::Nvfp4, 640);
        let root = tempfile::tempdir()?;
        let package = root.path().join("fp8-qwen4-nvfp4a4");
        for name in ["tp2", "tp2-w384", "tp2-w256"] {
            std::fs::create_dir_all(package.join(name))?;
        }
        for layer in 0..8 {
            let mut view = tensors.clone();
            view.slice_layer = layer;
            let coordinator = view.for_coordinator();
            for (rank, expected) in [(0, (0, 384)), (1, (384, 256))] {
                let (directory, slicing) = exact_layout_directory_with(&package.join("tp2"), &coordinator, 2, rank, true);
                assert_eq!(slicing, Slicing::Blocks(128));
                assert_eq!(directory, package.join(format!("tp2-w{}", expected.1)));
                assert_eq!(coordinator.rank_range_with(2, rank, slicing)?, expected);
                assert_eq!(coordinator.unit_map(2, rank, slicing)?.rule,
                    cuteafd_core::tp_units::UnitRule::Contiguous);
                let (directory, slicing) = exact_layout_directory_with(&package.join("tp2"), &view, 2, rank, true);
                assert_eq!((directory, slicing), (package.join("tp2"), Slicing::Padded));
                assert_eq!(view.rank_range_with(2, rank, slicing)?, (rank * 320, 320));
                assert_eq!(view.rank_width(2, rank, slicing)?, 384);
            }
        }
        Ok(())
    }

    #[test]
    fn nvfp4_layer_rotation_reconstructs_native_k16_bytes_with_padded_storage() -> Result<()> {
        use crate::plan::testing::{nvfp4, write_snapshot};
        use crate::plan::experts::spark_recipe;
        for intermediate in [640, 2048] {
            let root = tempfile::tempdir()?;
            let shape = RoutedExpertShape { layers: 1, first_layer: 0, experts: 2,
                topk: 1, hidden: 256, intermediate, draft_stages: 0, draft_experts: 0 };
            let fixtures = (0..shape.experts).flat_map(|expert| {
                [("gate", intermediate, 256), ("up", intermediate, 256), ("down", 256, intermediate)]
                    .into_iter().flat_map(move |(projection, n, k)|
                        nvfp4(&format!("model.layers.0.mlp.experts.{expert}.{projection}_proj"), n, k))
            }).collect::<Vec<_>>();
            write_snapshot(root.path(), &serde_json::json!({}), &fixtures, None);
            let mut tensors = Fp8ExpertTensors::read(root.path(), shape)?;
            for (name, located) in tensors.tensors.iter() {
                let bytes = if name.ends_with("_scale_2") || name.ends_with(".input_scale") {
                    0.25f32.to_le_bytes().to_vec()
                } else { (0..located.bytes as usize).map(|i| ((i * 17 + i / 13 + 7) % 251) as u8).collect() };
                std::fs::OpenOptions::new().write(true).open(root.path().join(&located.shard))?
                    .write_all_at(&bytes, located.offset)?;
            }
            for world in [2, 3, 4, 5, 6] {
                let package = if world == 5 { "glmf:nvfp4" } else { "glm:nvfp4" };
                let recipe = spark_recipe(package, intermediate, world)?;
                for layer in 0..world {
                    tensors.slice_layer = layer;
                    for expert in 0..shape.experts {
                        for projection in Fp8Projection::ALL {
                            let (w, s) = tensors.projection_bytes(projection);
                            let mut source = [vec![0; w], vec![0; s]];
                            let [weight, scale] = &mut source;
                            tensors.read_projection_once(0, expert, projection, weight, scale)?;
                            let mut reconstructed = [vec![0; w], vec![0; s]];
                            for rank in 0..world {
                                let (directory, slicing) = exact_layout_directory_with(Path::new("tp"),
                                    &tensors, world, rank, true);
                                assert_eq!(directory, Path::new("tp"));
                                assert_eq!(slicing, recipe.slicing);
                                let map = tensors.unit_map(world, rank, slicing)?;
                                assert_eq!(map.unit_columns, 16);
                                let live = tensors.rank_range(world, rank)?;
                                assert_eq!(map.column_ranges(expert)?, [live.0..live.0 + live.1]);
                                assert_eq!(tensors.rank_width(world, rank, slicing)?,
                                    recipe.width_for_layer(rank, layer).unwrap());
                                let sizes = tensors.slice_bytes_with(projection, world, rank, slicing)?;
                                let mut actual = [vec![255; sizes.0], vec![255; sizes.1]];
                                let [weight, scale] = &mut actual;
                                tensors.read_slice_with(0, expert, projection, world, rank, slicing,
                                    weight, scale, &mut Vec::new())?;
                                let windows = tensors.projection_slices(projection, world, rank, slicing)?;
                                for (index, window) in windows.into_iter().enumerate() {
                                    let mut expected = vec![0; actual[index].len()];
                                    for row in 0..window.rows {
                                        let src = window.source_offset + row * window.source_pitch;
                                        let dst = row * window.destination_pitch;
                                        expected[dst..dst + window.width].copy_from_slice(&source[index][src..src + window.width]);
                                        reconstructed[index][src..src + window.width].copy_from_slice(&actual[index][dst..dst + window.width]);
                                    }
                                    assert_eq!(actual[index], expected,
                                        "I{intermediate} TP{world} L{layer} r{rank} {projection:?}");
                                }
                            }
                            // Scalar metadata is replicated separately, not part of matrix windows.
                            let start = s - 8;
                            reconstructed[1][start..].copy_from_slice(&source[1][start..]);
                            assert_eq!(reconstructed, source);
                        }
                    }
                }
            }
        }
        Ok(())
    }

    #[test]
    fn exact_even_mxfp4_preserves_source_ranges_at_native_granularity() {
        let tensors = catalog(ExpertFormat::Mxfp4, 2304);
        for tp in [3, 4, 6] {
            assert_eq!(tensors.exact_slicing(tp), Slicing::Blocks(32));
            for rank in 0..tp {
                assert_eq!(tensors.rank_range_with(tp, rank, Slicing::Blocks(32)).unwrap(),
                    (rank * (2304 / tp), 2304 / tp));
            }
        }
    }

    #[test]
    fn mxfp4_slices_keep_their_32_blocks() {
        let mxfp4 = catalog(ExpertFormat::Mxfp4, 2048);
        assert_eq!(mxfp4.slice(6).unwrap(), 384);
        assert_eq!(ranges(&mxfp4, 6), [(0, 352), (352, 352), (704, 352), (1056, 352), (1408, 320), (1728, 320)]);
        assert_eq!(mxfp4.slice(2).unwrap(), 1024);
    }

    #[test]
    fn packaged_world_directory_slicing_matches_coordinator_fingerprints() -> Result<()> {
        use crate::plan::experts::spark_recipe;
        use cuteafd_core::tp_units::RankUnits;
        for (package, format) in [
            ("mimo:fp8", ExpertFormat::Fp8Block128),
            ("mimof:fp8", ExpertFormat::Mxfp4), ("mimop:fp8", ExpertFormat::Mxfp4),
            ("glm:fp8", ExpertFormat::Fp8Block128), ("glmf:fp8", ExpertFormat::Fp8Block128),
            ("qwen4:fp8", ExpertFormat::Fp8Block128), ("step5:fp8", ExpertFormat::Fp8Block128),
            ("step5:bf16", ExpertFormat::Bf16), ("step5:nvfp4", ExpertFormat::Nvfp4),
            ("glm:nvfp4", ExpertFormat::Nvfp4), ("glm:nvfp4a4", ExpertFormat::Nvfp4),
            ("glmf:nvfp4", ExpertFormat::Nvfp4), ("glmf:nvfp4a4", ExpertFormat::Nvfp4),
            ("qwen4:nvfp4", ExpertFormat::Nvfp4), ("qwen4:nvfp4a4", ExpertFormat::Nvfp4),
        ] {
            for intermediate in [640, 1536, 2048, 2304, 3072] {
                let mut tensors = catalog(format, intermediate);
                for world in 1..=6 {
                    let Ok(recipe) = spark_recipe(package, intermediate, world) else { continue };
                    let root = tempfile::tempdir()?;
                    let directory = root.path().join(format!("tp{world}"));
                    std::fs::create_dir(&directory)?;
                    for layout in &recipe.required_layouts {
                        std::fs::create_dir_all(root.path().join(layout))?;
                    }
                    for layer in [0, 1, 40, 77] {
                        tensors.slice_layer = layer;
                        for rank in 0..world {
                            let (selected, slicing) = exact_layout_directory_with(&directory, &tensors, world, rank, true);
                            let worker_slicing = tensors.slicing_for_directory(&selected, world);
                            let context = format!("{package} i{intermediate} tp{world} L{layer} r{rank}");
                            assert_eq!(slicing, worker_slicing, "{context}");
                            assert_eq!(worker_slicing, recipe.slicing, "{context}");
                            let unit = match recipe.slicing { Slicing::Blocks(g) => g, Slicing::Padded => tensors.block() };
                            let coordinator = RankUnits::new(recipe.rule, tensors.shape.experts,
                                intermediate, unit, layer, world, rank)?;
                            let worker = tensors.unit_map(world, rank, worker_slicing)?;
                            assert_eq!(coordinator.fingerprint(), worker.fingerprint(), "{context}");
                            assert_eq!(tensors.rank_width(world, rank, worker_slicing)?,
                                recipe.widths[recipe.rank_for_layer(rank, layer)], "{context}");
                        }
                    }
                }
            }
        }
        Ok(())
    }

    #[test]
    fn exact_even_directory_keeps_source_ranges_and_stored_bytes() -> Result<()> {
        for format in [ExpertFormat::Fp8Block128, ExpertFormat::Mxfp4, ExpertFormat::Nvfp4, ExpertFormat::Bf16] {
            let tensors = catalog(format, 2048);
            for tp in [2, 4] {
                let directory = PathBuf::from(format!("tp{tp}"));
                let slicing = tensors.slicing_for_directory(&directory, tp);
                assert_eq!(slicing, Slicing::Blocks(tensors.block()));
                for rank in 0..tp {
                    assert_eq!(tensors.rank_range_with(tp, rank, slicing)?, tensors.rank_range(tp, rank)?);
                    assert_eq!(tensors.rank_width(tp, rank, slicing)?, tensors.rank_width(tp, rank, Slicing::Padded)?);
                    for projection in Fp8Projection::ALL {
                        assert_eq!(tensors.projection_slices(projection, tp, rank, slicing)?,
                            tensors.projection_slices(projection, tp, rank, Slicing::Padded)?);
                        assert_eq!(tensors.slice_bytes_with(projection, tp, rank, slicing)?,
                            tensors.slice_bytes_with(projection, tp, rank, Slicing::Padded)?);
                    }
                }
                assert_eq!(exact_layout_directory_with(&directory, &tensors, tp, 0, false),
                    (directory, slicing));
            }
            assert_eq!(tensors.slicing_for_directory(Path::new("tp6"), 6), Slicing::Padded);
            assert_eq!(tensors.slicing_for_directory(Path::new("tp0"), 0), Slicing::Padded);
        }
        Ok(())
    }

    #[test]
    fn package_selection_requires_every_rank_width_and_honors_opt_out() {
        for format in [ExpertFormat::Fp8Block128, ExpertFormat::Mxfp4] {
            let tensors = catalog(format, 2048);
            let dir = tempfile::tempdir().unwrap();
            let padded = dir.path().join("tp6");
            std::fs::create_dir(&padded).unwrap();
            for rank in 0..6 {
                assert_eq!(exact_layout_directory_with(&padded, &tensors, 6, rank, true),
                    (padded.clone(), Slicing::Padded));
            }
            let exact = tensors.exact_slicing(6);
            let widest = tensors.rank_width(6, 0, exact).unwrap();
            let narrowest = tensors.rank_width(6, 5, exact).unwrap();
            std::fs::create_dir(dir.path().join(format!("tp6-w{widest}"))).unwrap();
            assert_eq!(exact_layout_directory_with(&padded, &tensors, 6, 0, true),
                (padded.clone(), Slicing::Padded));
            std::fs::create_dir(dir.path().join(format!("tp6-w{narrowest}"))).unwrap();
            for rank in 0..6 {
                let width = tensors.rank_width(6, rank, exact).unwrap();
                assert_eq!(exact_layout_directory_with(&padded, &tensors, 6, rank, true),
                    (dir.path().join(format!("tp6-w{width}")), exact));
                assert_eq!(exact_layout_directory_with(&padded, &tensors, 6, rank, false),
                    (padded.clone(), Slicing::Padded));
            }
            assert_eq!(exact_layout_directory_with(&padded, &tensors, 6, 6, true),
                (padded.clone(), Slicing::Padded));
            assert_eq!(exact_layout_directory_with(&padded, &tensors, 1, 0, true),
                (padded, exact));
        }
    }

    #[test]
    fn exact_block_slices_cover_the_intermediate_without_padding() {
        let tensors = catalog(ExpertFormat::Mxfp4, 2048);
        let exact = Slicing::Blocks(128);
        let widths: Vec<usize> = (0..6).map(|r| tensors.rank_width(6, r, exact).unwrap()).collect();
        assert_eq!(widths, [384, 384, 384, 384, 256, 256]);
        let ranges: Vec<(usize, usize)> = (0..6).map(|r| tensors.rank_range_with(6, r, exact).unwrap()).collect();
        assert_eq!(ranges.iter().map(|r| r.1).sum::<usize>(), 2048);
        assert!(ranges.windows(2).all(|w| w[0].0 + w[0].1 == w[1].0));
        // Padded: every rank stores 384 for 352/320 real rows.
        assert_eq!((0..6).map(|r| tensors.rank_width(6, r, Slicing::Padded).unwrap()).sum::<usize>(), 2304);
        let (w, s) = tensors.slice_bytes_with(Fp8Projection::Down, 6, 5, exact).unwrap();
        assert_eq!((w, s), (6144 * 256 / 2, 6144 * 256 / 32));
        assert!(tensors.rank_range_with(6, 0, Slicing::Blocks(48)).is_err());
    }
}
