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
//! programs widen them exactly to BF16 (`fp8-mimop` packages). Their slices
//! split `I` in whole 32-element scale blocks, as evenly as the blocks allow
//! (TP6 of 2048: 352, 352, 352, 352, 320, 320), and every rank stores its
//! slice zero-padded to one 128-aligned width (384): zero gate/up rows give
//! SiLU(0) * 0 = 0 and zero down columns add nothing, so padding is exact.
//!
//! NVIDIA ModelOpt NVFP4 releases (`ExpertFormat::Nvfp4`: GLM 5.3, GLM 5.3
//! Flash, Qwen 3.8 Flash Next) store `weight` U8 `[N, K/2]` (E2M1, even
//! element low), `weight_scale` E4M3 `[N, K/16]` (linear layout) and an FP32
//! `weight_scale_2` per projection (the expert's alpha; `input_scale` belongs
//! to the W4A4 recipe and is not read). The `fp8-<geometry>-nvfp4` packages
//! widen `e2m1 * e4m3` exactly to BF16 and apply alpha in FP32 (W4A16).
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
    /// E4M3 with FP32 (or BF16) 128x128 block scales (`weight_scale_inv`).
    Fp8Block128,
    /// Packed E2M1 with UE8M0 scales per 32 values along K (`weight_scale`).
    Mxfp4,
    /// ModelOpt NVFP4: packed E2M1, E4M3 scales per 16 values along K
    /// (`weight_scale`) and an FP32 per-tensor `weight_scale_2`.
    Nvfp4,
}

impl ExpertFormat {
    /// The package directory suffix and label (`fp8-glmf`, `fp8-mimop`,
    /// `fp8-glmf-nvfp4`): FP8 and MXFP4 checkpoints have distinct
    /// geometries, NVFP4 releases share theirs with the FP8 ones.
    pub fn package_suffix(self) -> &'static str {
        match self {
            Self::Fp8Block128 | Self::Mxfp4 => "",
            Self::Nvfp4 => "-nvfp4",
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
    tensors: HashMap<String, Located>,
}

/// How a package lays a TP rank's intermediate slice out.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Slicing {
    /// Ranks own whole format blocks as evenly as possible; every rank stores
    /// the widest rank's width, zero-padded to 128 (one program per world).
    Padded,
    /// Ranks own whole `g`-row blocks (`g` a multiple of 128) and store
    /// exactly their own rows (a program per distinct width): no padding.
    Blocks(usize),
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
            name.contains(".mlp.experts.") && layer.is_some_and(|layer| layers.is_none_or(|selected| selected.contains(&layer)))
        };
        let shards: std::collections::BTreeSet<&String> =
            weight_map.iter().filter(|(name, _)| routed(name)).map(|(_, shard)| shard).collect();
        let mut tensors = HashMap::new();
        for shard in shards {
            let needed = weight_map.iter().find(|(name, file)| *file == shard && routed(name))
                .map(|(name, _)| name).context("selected expert shard has no requirement")?;
            for meta in read_safetensors_metadata(&snapshot.join(shard))
                .with_context(|| format!("expert role needs tensor {needed} in shard {shard} at snapshot {}", snapshot.display()))?
            {
                if routed(&meta.name) && weight_map.get(&meta.name) == Some(shard) {
                    tensors.insert(meta.name.clone(), Located {
                        shard: shard.clone(),
                        offset: meta.byte_offset,
                        bytes: meta.byte_length,
                        dtype: meta.dtype,
                        shape: meta.shape,
                    });
                }
            }
        }
        let mut catalog = Self { snapshot: snapshot.to_path_buf(), shape, prefix: prefix.to_string(), mtp_layers,
            format: ExpertFormat::Fp8Block128, tensors };
        let first_layer = layers.and_then(|selected| selected.first().copied()).unwrap_or(shape.first_layer);
        let first = catalog.name(first_layer, 0, Fp8Projection::Gate);
        if matches!(catalog.located(&first)?.dtype, DType::U8 | DType::I8) {
            // E4M3 scales with a second-level FP32 scale: ModelOpt NVFP4; UE8M0 ones: MXFP4.
            let nvfp4 = catalog.tensors.get(&format!("{first}_scale")).is_some_and(|s| s.dtype == DType::F8E4M3)
                && catalog.tensors.contains_key(&format!("{first}_scale_2"));
            catalog.format = if nvfp4 { ExpertFormat::Nvfp4 } else { ExpertFormat::Mxfp4 };
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

    pub fn format(&self) -> ExpertFormat {
        self.format
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
            dtype => anyhow::bail!("{first}: unsupported expert dtype {dtype:?}"),
        }
    }

    /// A metadata-only view using this layer's format. The layer numbering and
    /// tensor names are preserved so a mixed-format MTP package can own just
    /// the draft layer. Validate all experts before allocating its storage.
    pub fn for_layer(&self, layer: usize) -> Result<Self> {
        let mut view = self.clone();
        view.format = self.layer_format(layer)?;
        view.validate_layer(layer)?;
        Ok(view)
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
        ensure!(g > 0 && g % self.block() == 0 && g % 128 == 0 && i % g == 0 && i / g >= tp,
            "intermediate {i} does not split into whole {g}-row blocks over {tp} ranks");
        let blocks = i / g;
        let (base, extra) = (blocks / tp, blocks % tp);
        let first = rank * base + rank.min(extra);
        Ok((first * g, (base + usize::from(rank < extra)) * g))
    }

    /// [`Self::slice_bytes`] of rank `rank` under `slicing`.
    pub fn slice_bytes_with(&self, projection: Fp8Projection, tp: usize, rank: usize, slicing: Slicing)
        -> Result<(usize, usize)> {
        let (rows, cols) = self.dims(projection);
        let slice = self.rank_width(tp, rank, slicing)?;
        let (rows, cols) = if projection == Fp8Projection::Down { (rows, slice) } else { (slice, cols) };
        if self.format.packed_fp4() {
            return Ok((rows * cols / 2, rows * cols / self.format.group()));
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
        let block = self.block();
        let blocks = self.shape.intermediate / block;
        let (base, extra) = (blocks / tp, blocks % tp);
        let first = rank * base + rank.min(extra);
        Ok((first * block, (base + usize::from(rank < extra)) * block))
    }

    /// Bytes of one expert projection's slice: (weight, scales) as stored
    /// (E4M3 + FP32 128x128 grid, or packed E2M1 + UE8M0 per 32).
    pub fn slice_bytes(&self, projection: Fp8Projection, tp: usize) -> Result<(usize, usize)> {
        let (rows, cols) = self.dims(projection);
        let slice = self.slice(tp)?;
        let (rows, cols) = if projection == Fp8Projection::Down { (rows, slice) } else { (slice, cols) };
        if self.format.packed_fp4() {
            return Ok((rows * cols / 2, rows * cols / self.format.group()));
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
        Ok(if self.format.packed_fp4() {
            [matrix(1, 2, 1, bytes.0), matrix(1, self.format.group(), 1, bytes.1)]
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
                let (row_in, row_out, take) = (cols / per, slice / per, len / per);
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

    fn catalog(format: ExpertFormat, intermediate: usize) -> Fp8ExpertTensors {
        let shape = RoutedExpertShape { layers: 78, first_layer: 3, experts: 256, topk: 8, hidden: 6144, intermediate,
            draft_stages: 0, draft_experts: 0 };
        Fp8ExpertTensors { snapshot: PathBuf::new(), shape, prefix: "model.".into(), mtp_layers: false, format,
            tensors: HashMap::new() }
    }

    fn ranges(tensors: &Fp8ExpertTensors, tp: usize) -> Vec<(usize, usize)> {
        (0..tp).map(|rank| tensors.rank_range(tp, rank).unwrap()).collect()
    }

    #[test]
    fn mtp_only_reads_its_format_and_ignores_missing_or_corrupt_backbone() {
        use crate::plan::testing::{t, write_safetensors};
        for nvfp4 in [false, true] {
            let dir = tempfile::tempdir().unwrap();
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
                tensors.tensors.insert(name, Located { shard: "not-read.safetensors".into(), offset: 0,
                    bytes, dtype, shape });
            }
        }
        tensors.validate_layer(3).unwrap();
        assert_eq!(tensors.layer_format(3).unwrap(), ExpertFormat::Nvfp4);
        let gate = tensors.name(3, 0, Fp8Projection::Gate);
        let input = format!("{}.input_scale", gate.strip_suffix(".weight").unwrap());
        let scale = tensors.tensors.remove(&input).unwrap();
        assert!(tensors.validate_layer(3).unwrap_err().to_string().contains(&input));
        tensors.tensors.insert(input, scale);
        let mtp = tensors.name(78, 0, Fp8Projection::Gate);
        tensors.tensors.insert(mtp.clone(), Located { shard: "not-read.safetensors".into(), offset: 0,
            bytes: 128 * 128, dtype: DType::F8E4M3, shape: vec![128, 128] });
        assert_eq!(tensors.layer_format(78).unwrap(), ExpertFormat::Fp8Block128);
        assert!(tensors.layer_format(79).is_err());
        let error = tensors.validate_layer(78).unwrap_err().to_string();
        assert!(error.contains(&mtp) && error.contains("expected packed E2M1 U8"), "{error}");
        // A separate package view accepts the complete FP8 draft, without
        // changing the backbone contract or duplicating device weights.
        for projection in Fp8Projection::ALL {
            let name = tensors.name(78, 0, projection);
            tensors.tensors.insert(name.clone(), Located { shard: "not-read.safetensors".into(), offset: 0,
                bytes: 128 * 128, dtype: DType::F8E4M3, shape: vec![128, 128] });
            tensors.tensors.insert(format!("{name}_scale_inv"), Located {
                shard: "not-read.safetensors".into(), offset: 0,
                bytes: 4, dtype: DType::F32, shape: vec![1, 1] });
        }
        let draft = tensors.for_layer(78).unwrap();
        assert_eq!(draft.format(), ExpertFormat::Fp8Block128);
        assert_eq!(tensors.format(), ExpertFormat::Nvfp4);
        assert_eq!(draft.name(78, 0, Fp8Projection::Gate), mtp);
        assert!(draft.validate_layer(3).is_err());
        assert!(tensors.validate_layer(78).is_err());
        tensors.tensors.remove(&format!("{}_scale", tensors.name(3, 0, Fp8Projection::Down)));
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
            (ExpertFormat::Mxfp4, false), (ExpertFormat::Nvfp4, false)] {
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
                    };
                    if bf16 { tensors[1].1 = "BF16"; }
                    fixtures.extend(tensors);
                }
            }
            write_snapshot(dir.path(), &serde_json::json!({}), &fixtures, None);
            let tensors = Fp8ExpertTensors::read(dir.path(), shape)?;
            for (name, located) in &tensors.tensors {
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
    fn mxfp4_slices_keep_their_32_blocks() {
        let mxfp4 = catalog(ExpertFormat::Mxfp4, 2048);
        assert_eq!(mxfp4.slice(6).unwrap(), 384);
        assert_eq!(ranges(&mxfp4, 6), [(0, 352), (352, 352), (704, 352), (1056, 352), (1408, 320), (1728, 320)]);
        assert_eq!(mxfp4.slice(2).unwrap(), 1024);
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
        assert!(tensors.rank_range_with(6, 0, Slicing::Blocks(96)).is_err());
    }
}
