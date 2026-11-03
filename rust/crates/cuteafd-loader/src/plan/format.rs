//! Storage-format detection for one logical weight (a group of tensors that
//! share a stem, such as `x.weight` + `x.weight_scale_inv`, or the EXL3
//! `x.trellis`/`x.suh`/`x.svh`/`x.mcg` set).
//!
//! Detection is lossless: a [`QuantOperand`] records the element encoding
//! (E4M3 and E5M2 stay distinct), the scale tensor's encoding, direction and
//! block tiling, row segments whose block grid restarts, the logical and
//! stored shapes and the companion tensors. It decides nothing about what a
//! family executes; family contracts accept or reject an operand, and resolve
//! segment layouts only they know (MiMo's fused qkv shards). A group whose
//! tensors do not describe one consistent operand is a [`Malformed`] result
//! naming the tensor and the reason.
//!
//! The descriptor is shaped to carry ModelOpt NVFP4 (PLAN.md Phase 5,
//! `QuantOperand`): E2M1 codes low nibble first, E4M3 scales per 16 in linear
//! layout, an FP32 `weight_scale_2` and an `input_scale`.
use cuteafd_core::DType;
use serde::Serialize;
use std::collections::BTreeMap;

use super::checkpoint::CheckpointTensor;

/// How the stored elements encode the weight.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize)]
#[serde(rename_all = "snake_case", tag = "kind")]
pub enum Encoding {
    Bf16,
    F16,
    F32,
    /// FP8 E4M3 (OCP, finite-only `fn` variant as checkpoints store it).
    E4m3,
    /// FP8 E5M2.
    E5m2,
    /// FP4 E2M1 codes, two per byte along the last axis, the even element in
    /// the low nibble.
    E2m1,
    /// EXL3 trellis codes at `bits` per weight (mcg codebook, Hadamard
    /// sign vectors `suh`/`svh`).
    Exl3 { bits: usize },
    /// Unit-major experts, E4M3-rounded MCG with shared hidden rotation.
    Exlr8 { bits: usize },
    /// Family-independent dense trellis with a file-local rotation id.
    Trellis { k: usize, codebook: TrellisCodebook, rotation: usize },
    Int4 { group: usize },
    /// Integer data (hash-route tables, offsets); `bits` wide.
    Int { bits: usize },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize)]
pub enum TrellisCodebook { E4M3, Fp16 }

/// How a scale tensor stores its values.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ScaleEncoding {
    F32,
    Bf16,
    F16,
    /// Unsigned E8M0 exponents (`F8_E8M0`, or `U8` in MXFP4 checkpoints).
    Ue8m0,
    E4m3,
}

/// How a scale applies. Every checkpoint layout this build knows multiplies
/// (`w = q * s`), including HF's `weight_scale_inv` (named for the
/// quantizer's division); a divisor layout would get its own variant.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ScaleDirection {
    Multiply,
}

/// How the scale grid tiles the weight's rows.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize)]
#[serde(rename_all = "snake_case", tag = "kind")]
pub enum RowTiling {
    /// Uniform blocks of `rows` (the last one may be partial).
    Uniform { rows: usize },
    /// More block rows than a uniform tiling has: the grid restarts at segment
    /// boundaries only the family knows (fused projections, per-head blocks).
    Unresolved { scale_rows: usize },
    /// Segments (row counts, in storage order), each tiled by `block` rows
    /// from its own first row.
    Segments { block: usize, rows: Vec<usize> },
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize)]
pub struct BlockScale {
    /// The member suffix holding the scales (`weight_scale_inv`, `scale`, ...).
    pub tensor: String,
    pub encoding: ScaleEncoding,
    pub direction: ScaleDirection,
    pub rows: RowTiling,
    /// Logical elements per scale along the last (K) axis.
    pub cols: usize,
    /// The stored scale grid shape.
    pub grid: Vec<usize>,
}

/// A FP4 operand's per-tensor second-level scale (ModelOpt `weight_scale_2`)
/// or static activation scale (`input_scale`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize)]
pub struct TensorScale {
    pub encoding: ScaleEncoding,
}

/// One logical weight as the checkpoint stores it.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize)]
pub struct QuantOperand {
    pub encoding: Encoding,
    /// The stored container dtype of the codes (`U8` vs `I8` for packed FP4).
    pub container: String,
    /// The logical (dequantized) shape.
    pub logical: Vec<usize>,
    /// The stored shape of the codes.
    pub stored: Vec<usize>,
    pub scale: Option<BlockScale>,
    pub global_scale: Option<TensorScale>,
    pub input_scale: Option<TensorScale>,
    /// Member suffixes beyond the codes and scales (`suh`, `svh`, `mcg`, `bias`).
    pub companions: Vec<String>,
}

/// A group whose tensors do not form one operand.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, thiserror::Error)]
#[error("{tensor}: {reason}")]
pub struct Malformed {
    pub tensor: String,
    pub reason: String,
}

impl QuantOperand {
    fn plain(encoding: Encoding, meta: &crate::SafetensorsTensorMetadata) -> Self {
        Self {
            encoding,
            container: dtype_label(&meta.dtype),
            logical: meta.shape.clone(),
            stored: meta.shape.clone(),
            scale: None,
            global_scale: None,
            input_scale: None,
            companions: Vec::new(),
        }
    }

    /// Rows x cols (the last two logical dims), when the operand is a matrix.
    pub fn matrix(&self) -> Option<(usize, usize)> {
        match self.logical.as_slice() {
            [rows, cols] => Some((*rows, *cols)),
            _ => None,
        }
    }

    /// A short, distinct label: the element encoding, then the scale encoding
    /// and its tiling. E4M3 and E5M2, FP32 and E8M0 scales, uniform and
    /// segmented grids all label differently.
    pub fn label(&self) -> String {
        if self.companions.iter().any(|s| s == "strided") { return "fp8-strided264/ue8m0".into(); }
        let element = match self.encoding {
            Encoding::Bf16 => return "bf16".into(),
            Encoding::F16 => return "f16".into(),
            Encoding::F32 => return "f32".into(),
            Encoding::Int { .. } => return "int".into(),
            Encoding::Exl3 { bits } => return format!("exl3-k{bits}"),
            Encoding::Exlr8 { bits } => return format!("exlr8-k{bits}"),
            Encoding::Trellis { k, codebook, rotation } => return format!("trellis-k{k}-{codebook:?}-rotation{rotation}"),
            Encoding::Int4 { group } => return format!("int4-g{group}"),
            Encoding::E4m3 => "fp8",
            Encoding::E5m2 => "fp8e5m2",
            Encoding::E2m1 => {
                return match (&self.scale, self.global_scale) {
                    (Some(scale), Some(_)) if scale.encoding == ScaleEncoding::E4m3 => format!("nvfp4-g{}", scale.cols),
                    (Some(scale), None) if scale.encoding == ScaleEncoding::Ue8m0 => format!("mxfp4-g{}", scale.cols),
                    (Some(scale), _) => format!("fp4-{}-g{}", scale_label(scale.encoding), scale.cols),
                    (None, _) => "fp4-unscaled".into(),
                };
            }
        };
        let Some(scale) = &self.scale else {
            return format!("{element}-unscaled");
        };
        let (rows, cols) = self.matrix().unwrap_or((0, 0));
        let encoding = scale_label(scale.encoding);
        match &scale.rows {
            RowTiling::Uniform { rows: r } if *r >= rows && scale.cols >= cols => format!("{element}-tensor/{encoding}"),
            RowTiling::Uniform { rows: 1 } if scale.cols >= cols => format!("{element}-channel/{encoding}"),
            RowTiling::Uniform { rows: r } => format!("{element}-block{r}x{}/{encoding}", scale.cols),
            RowTiling::Unresolved { .. } => format!("{element}-block?x{}/{encoding}-unresolved", scale.cols),
            RowTiling::Segments { block, rows } => {
                format!("{element}-block{block}x{}/{encoding}-segmented{}", scale.cols, rows.len())
            }
        }
    }

    /// E4M3 with `encoding` scales in uniform `block` x `block` tiles.
    pub fn is_fp8_block(&self, block: usize, encodings: &[ScaleEncoding]) -> bool {
        self.encoding == Encoding::E4m3
            && self.scale.as_ref().is_some_and(|s| {
                s.cols == block && matches!(s.rows, RowTiling::Uniform { rows } if rows == block)
                    && encodings.contains(&s.encoding)
            })
    }

    /// E4M3 with one `encoding` scale per row and `group` elements along K.
    pub fn is_fp8_row_groups(&self, group: usize, encodings: &[ScaleEncoding]) -> bool {
        self.encoding == Encoding::E4m3
            && self.scale.as_ref().is_some_and(|s| {
                s.cols == group && matches!(s.rows, RowTiling::Uniform { rows: 1 }) && encodings.contains(&s.encoding)
            })
    }

    /// E4M3 with `encoding` scales whose rows restart at resolved segments.
    pub fn is_fp8_segmented(&self, block: usize, encodings: &[ScaleEncoding]) -> bool {
        self.encoding == Encoding::E4m3
            && self.scale.as_ref().is_some_and(|s| {
                s.cols == block && matches!(&s.rows, RowTiling::Segments { block: b, .. } if *b == block)
                    && encodings.contains(&s.encoding)
            })
    }

    /// Packed E2M1 with UE8M0 scales per `group` along K and no second-level scale.
    pub fn is_mxfp4(&self, group: usize) -> bool {
        self.encoding == Encoding::E2m1
            && self.global_scale.is_none()
            && self.scale.as_ref().is_some_and(|s| {
                s.encoding == ScaleEncoding::Ue8m0 && s.cols == group && matches!(s.rows, RowTiling::Uniform { rows: 1 })
            })
    }

    /// ModelOpt NVFP4: packed E2M1, E4M3 per-16 linear scales, FP32 global scale.
    pub fn is_nvfp4(&self) -> bool {
        self.encoding == Encoding::E2m1
            && self.global_scale.is_some_and(|g| g.encoding == ScaleEncoding::F32)
            && self.scale.as_ref().is_some_and(|s| {
                s.encoding == ScaleEncoding::E4m3 && s.cols == 16 && matches!(s.rows, RowTiling::Uniform { rows: 1 })
            })
    }

    /// MCG trellis bits for the existing routed-expert kernel contracts.
    /// Dense native loaders select their codebook independently from headers.
    pub fn exl3_bits(&self) -> Option<usize> {
        match self.encoding {
            Encoding::Exl3 { bits } if !self.companions.iter().any(|s|s=="mul1") => Some(bits),
            _ => None,
        }
    }

    /// One of the plain encodings, unscaled.
    pub fn is_plain(&self, encodings: &[Encoding]) -> bool {
        self.scale.is_none() && self.global_scale.is_none() && encodings.contains(&self.encoding)
    }

    /// Resolves an [`RowTiling::Unresolved`] grid into `segments` (row counts
    /// in storage order), each tiled by `block` rows. Fails unless the
    /// segments cover the rows and their blocks are exactly the grid's rows.
    pub fn resolve_segments(&mut self, block: usize, segments: Vec<usize>) -> Result<(), String> {
        let rows = self.matrix().map(|(rows, _)| rows).ok_or("not a matrix")?;
        let scale = self.scale.as_mut().ok_or("no scale grid")?;
        let grid_rows = scale.grid.first().copied().unwrap_or(0);
        let total: usize = segments.iter().sum();
        let blocks: usize = segments.iter().map(|s| s.div_ceil(block)).sum();
        if block == 0 || total != rows || blocks != grid_rows {
            return Err(format!(
                "segments {segments:?} of {block}-row blocks do not tile {rows} rows onto {grid_rows} scale rows"
            ));
        }
        if scale.cols != block {
            return Err(format!("{}-column scale blocks, segments use {block}", scale.cols));
        }
        scale.rows = RowTiling::Segments { block, rows: segments };
        Ok(())
    }
}

fn scale_label(encoding: ScaleEncoding) -> &'static str {
    match encoding {
        ScaleEncoding::F32 => "f32",
        ScaleEncoding::Bf16 => "bf16",
        ScaleEncoding::F16 => "f16",
        ScaleEncoding::Ue8m0 => "ue8m0",
        ScaleEncoding::E4m3 => "e4m3",
    }
}

/// Suffixes that belong to the logical weight named by the stem before them.
const SUFFIXES: &[&str] = &[
    "weight_scale_inv",
    // Mistral consolidated checkpoints: FP32 block scales, w = q * s.
    "qscale_weight",
    "weight_scale_2",
    "weight_scale",
    "input_scale",
    "weight",
    "scale",
    "bias",
    "trellis",
    "suh",
    "svh",
    "mcg",
    "mul1",
    "su",
    "sv",
];

/// Splits `a.b.weight_scale_inv` into (`a.b`, `weight_scale_inv`). Names with no
/// known suffix are their own stem.
pub fn split_stem(name: &str) -> (&str, &str) {
    if let Some((stem, suffix)) = name.rsplit_once('.') {
        if SUFFIXES.contains(&suffix) || (suffix.strip_prefix("strided_").is_some_and(|part| part.len() == 5 && part.bytes().all(|b| b.is_ascii_digit())) && stem.ends_with(".engram.embed")) {
            return (stem, suffix);
        }
    }
    (name, "")
}

/// Groups tensors by stem, keeping each group's members keyed by suffix.
pub fn group_by_stem(tensors: &[CheckpointTensor]) -> BTreeMap<String, BTreeMap<String, &CheckpointTensor>> {
    group_tensor_refs(tensors.iter())
}

pub(crate) fn group_tensor_refs<'a>(tensors: impl IntoIterator<Item = &'a CheckpointTensor>)
    -> BTreeMap<String, BTreeMap<String, &'a CheckpointTensor>> {
    let mut groups: BTreeMap<String, BTreeMap<String, &CheckpointTensor>> = BTreeMap::new();
    for tensor in tensors {
        let (stem, suffix) = split_stem(&tensor.meta.name);
        groups
            .entry(stem.to_owned())
            .or_default()
            .insert(suffix.to_owned(), tensor);
    }
    groups
}

fn dtype_label(dtype: &DType) -> String {
    match dtype {
        DType::Unknown(value) => value.clone(),
        other => format!("{other:?}").to_ascii_lowercase(),
    }
}

fn scale_encoding(dtype: &DType) -> Option<ScaleEncoding> {
    Some(match dtype {
        DType::F32 => ScaleEncoding::F32,
        DType::Bf16 => ScaleEncoding::Bf16,
        DType::F16 => ScaleEncoding::F16,
        DType::F8E8M0 => ScaleEncoding::Ue8m0,
        DType::F8E4M3 => ScaleEncoding::E4m3,
        _ => return None,
    })
}

/// The power-of-two block edge whose tiling of `dim` gives exactly `scales`
/// blocks (576 rows in 128-row blocks need 5 scales).
fn block_edge(dim: usize, scales: usize) -> Option<usize> {
    if scales == 1 {
        return Some(dim.max(1));
    }
    [1usize, 16, 32, 64, 128, 256, 512].into_iter().find(|edge| dim.div_ceil(*edge) == scales)
}

/// Detects the storage format of one logical weight.
pub fn detect(members: &BTreeMap<String, &CheckpointTensor>) -> Result<QuantOperand, Malformed> {
    let get = |suffix: &str| members.get(suffix).map(|tensor| &tensor.meta);
    let malformed = |tensor: &str, reason: String| Malformed { tensor: tensor.to_owned(), reason };
    if members.keys().any(|suffix| suffix.starts_with("strided_")) {
        let first = &members.values().next().expect("nonempty strided group").meta;
        let mut rows = 0usize;
        for (number, (suffix, tensor)) in members.iter().enumerate() {
            let meta = &tensor.meta;
            if suffix != &format!("strided_{number:05}") || meta.dtype != DType::U8
                || meta.shape.len() != 2 || meta.shape[0] == 0 || meta.shape[1] != 264
                || meta.byte_length != meta.shape[0] as u64 * 264 {
                return Err(malformed(&meta.name, "strided engram requires contiguous U8 [part_rows, 264] parts".into()));
            }
            rows = rows.checked_add(meta.shape[0]).ok_or_else(|| malformed(&meta.name, "strided engram row overflow".into()))?;
        }
        let mut operand = QuantOperand::plain(Encoding::E4m3, first);
        operand.logical = vec![rows, 256];
        operand.stored = vec![rows, 264];
        operand.scale = Some(BlockScale { tensor: "strided".into(), encoding: ScaleEncoding::Ue8m0,
            direction: ScaleDirection::Multiply, rows: RowTiling::Uniform { rows: 1 },
            cols: 32, grid: vec![rows, 8] });
        operand.companions.push("strided".into());
        return Ok(operand);
    }
    if crate::formats::exl3_storage::EXL3_SUFFIXES.iter().any(|suffix| members.contains_key(*suffix)) {
        return detect_exl3(members);
    }
    let weight = match get("weight") {
        Some(weight) => weight,
        // A group without `weight`: one tensor named by its own stem, or a lone
        // scale (a table-wide scale shared by sharded weights).
        None => match members.values().collect::<Vec<_>>().as_slice() {
            [only] => &only.meta,
            _ => {
                let first = members.values().next().map_or(String::new(), |t| t.meta.name.clone());
                return Err(malformed(&first, format!("members {:?} have no weight tensor", members.keys().collect::<Vec<_>>())));
            }
        },
    };
    // A stacked projection has independent quantization grids/scalars per expert.
    // Validate the expert axis before applying the existing matrix contract.
    if weight.shape.len() == 3 && ["weight_scale_inv", "qscale_weight", "weight_scale", "scale"].iter().any(|suffix|
        members.get(*suffix).is_some_and(|scale| scale.meta.shape.iter().product::<usize>() != 1)) {
        let experts = &weight.shape[0];
        if *experts == 0 {
            return Err(malformed(&weight.name, "empty stacked expert axis".into()));
        }
        let mut views = BTreeMap::new();
        for (suffix, tensor) in members {
            let mut view = (*tensor).clone();
            if view.meta.shape.first() != Some(experts) {
                return Err(malformed(&view.meta.name, format!("expected expert axis {experts}, found {:?}", view.meta.shape)));
            }
            view.meta.shape.remove(0);
            views.insert(suffix.clone(), view);
        }
        let refs = views.iter().map(|(suffix, tensor)| (suffix.clone(), tensor)).collect();
        let mut operand = detect(&refs)?;
        operand.logical.insert(0, *experts);
        operand.stored = weight.shape.clone();
        if let Some(scale) = &mut operand.scale { scale.grid.insert(0, *experts); }
        return Ok(operand);
    }
    let companions: Vec<String> = members.keys().filter(|s| *s == "bias").cloned().collect();
    let mut operand = match &weight.dtype {
        DType::Bf16 => QuantOperand::plain(Encoding::Bf16, weight),
        DType::F16 => QuantOperand::plain(Encoding::F16, weight),
        DType::F32 => QuantOperand::plain(Encoding::F32, weight),
        DType::I16 => QuantOperand::plain(Encoding::Int { bits: 16 }, weight),
        DType::I32 => QuantOperand::plain(Encoding::Int { bits: 32 }, weight),
        DType::I64 => QuantOperand::plain(Encoding::Int { bits: 64 }, weight),
        DType::F8E4M3 | DType::F8E5M2 => {
            let encoding = if weight.dtype == DType::F8E4M3 { Encoding::E4m3 } else { Encoding::E5m2 };
            let mut operand = QuantOperand::plain(encoding, weight);
            let scale = ["weight_scale_inv", "qscale_weight", "scale", "weight_scale"].into_iter().find_map(|s| get(s).map(|m| (s, m)));
            if let Some((suffix, scale)) = scale {
                operand.scale = Some(fp8_scale(weight, suffix, scale)?);
            }
            if get("weight_scale_2").is_some() {
                return Err(malformed(&weight.name, "an FP8 weight with a second-level weight_scale_2".into()));
            }
            operand
        }
        DType::U8 | DType::I8 | DType::F4 => fp4(members, weight)?,
        other => {
            return Err(malformed(&weight.name, format!("unknown element dtype {}", dtype_label(other))));
        }
    };
    if let Some(input) = get("input_scale") {
        let encoding = scale_encoding(&input.dtype)
            .ok_or_else(|| malformed(&input.name, format!("input_scale dtype {}", dtype_label(&input.dtype))))?;
        operand.input_scale = Some(TensorScale { encoding });
    }
    operand.companions = companions;
    if operand.scale.is_none() && !matches!(operand.encoding, Encoding::E4m3 | Encoding::E5m2 | Encoding::E2m1) {
        if let Some(suffix) = ["weight_scale_inv", "qscale_weight", "weight_scale", "scale", "weight_scale_2"].into_iter()
            .find(|s| members.contains_key(*s) && members.len() > 1) {
            return Err(malformed(&members[suffix].meta.name, format!("a scale beside the unquantized {} weight {}",
                operand.label(), weight.name)));
        }
    }
    let known = ["weight", "weight_scale_inv", "qscale_weight", "weight_scale", "scale", "weight_scale_2", "input_scale", "bias", ""];
    if let Some((suffix, tensor)) = members.iter().find(|(s, _)| !known.contains(&s.as_str())) {
        return Err(malformed(&tensor.meta.name, format!("unexpected member .{suffix} beside {}", weight.name)));
    }
    Ok(operand)
}

/// The block scale of an FP8 weight `[N, K]` (or a per-tensor scale).
fn fp8_scale(weight: &crate::SafetensorsTensorMetadata, suffix: &str, scale: &crate::SafetensorsTensorMetadata)
    -> Result<BlockScale, Malformed> {
    let malformed = |reason: String| Malformed { tensor: scale.name.clone(), reason };
    // MXFP8 stores its UE8M0 exponents as U8 (the MXFP4 convention).
    let encoding = match scale.dtype {
        DType::U8 => Some(ScaleEncoding::Ue8m0),
        ref other => scale_encoding(other),
    }.ok_or_else(|| malformed(format!("scale dtype {}", dtype_label(&scale.dtype))))?;
    let numel: usize = scale.shape.iter().product();
    let (rows, cols) = match weight.shape.as_slice() {
        [rows, cols] => (*rows, *cols),
        _ if numel == 1 => {
            // A per-tensor scale of a weight of any rank.
            let n = weight.shape.iter().product();
            return Ok(BlockScale { tensor: suffix.into(), encoding, direction: ScaleDirection::Multiply,
                rows: RowTiling::Uniform { rows: n }, cols: n, grid: scale.shape.clone() });
        }
        other => return Err(malformed(format!("block scales of a rank-{} weight {other:?}", other.len()))),
    };
    let (grid_rows, grid_cols) = match scale.shape.as_slice() {
        [] | [1] => (1, 1),
        [r] if *r == rows => (rows, 1),
        [r, c] => (*r, *c),
        other => return Err(malformed(format!("scale grid {other:?} for weight [{rows}, {cols}]"))),
    };
    let col_edge = block_edge(cols, grid_cols)
        .ok_or_else(|| malformed(format!("{grid_cols} scale columns do not tile {cols} columns in power-of-two blocks")))?;
    let tiling = match block_edge(rows, grid_rows) {
        Some(edge) => RowTiling::Uniform { rows: edge },
        // More block rows than a uniform tiling (but fewer than per-row): the
        // grid restarts at segment boundaries.
        None if grid_rows > rows.div_ceil(col_edge.max(1)) && grid_rows < rows => {
            RowTiling::Unresolved { scale_rows: grid_rows }
        }
        None => return Err(malformed(format!("{grid_rows} scale rows do not tile {rows} rows"))),
    };
    Ok(BlockScale { tensor: suffix.into(), encoding, direction: ScaleDirection::Multiply, rows: tiling,
        cols: col_edge, grid: scale.shape.clone() })
}

/// Packed FP4 (two E2M1 codes per byte): MXFP4 (UE8M0 per 32) or ModelOpt
/// NVFP4 (E4M3 per 16 plus an FP32 `weight_scale_2`).
fn fp4(members: &BTreeMap<String, &CheckpointTensor>, weight: &crate::SafetensorsTensorMetadata)
    -> Result<QuantOperand, Malformed> {
    let get = |suffix: &str| members.get(suffix).map(|tensor| &tensor.meta);
    let malformed = |tensor: &str, reason: String| Malformed { tensor: tensor.to_owned(), reason };
    let Some((suffix, scale)) = ["weight_scale", "scale", "weight_scale_inv"].into_iter().find_map(|s| get(s).map(|m| (s, m)))
    else {
        // Unscaled 8-bit integers are integer data, not FP4.
        let bits = 8;
        let mut operand = QuantOperand::plain(Encoding::Int { bits }, weight);
        if weight.dtype == DType::F4 {
            operand.encoding = Encoding::E2m1;
        }
        return Ok(operand);
    };
    let [rows, packed] = weight.shape.as_slice() else {
        return Err(malformed(&weight.name, format!("packed FP4 weight of rank {} {:?}", weight.shape.len(), weight.shape)));
    };
    let (rows, k) = (*rows, packed * 2);
    let global = get("weight_scale_2");
    let encoding = match (&scale.dtype, global) {
        (DType::F8E8M0 | DType::U8, None) => ScaleEncoding::Ue8m0,
        (DType::F8E4M3, _) => ScaleEncoding::E4m3,
        (other, _) => {
            return Err(malformed(&scale.name, format!("FP4 scale dtype {}{}", dtype_label(other),
                if global.is_some() { " with a weight_scale_2" } else { "" })));
        }
    };
    let [scale_rows, groups] = scale.shape.as_slice() else {
        return Err(malformed(&scale.name, format!("FP4 scale grid {:?} for weight [{rows}, {k}]", scale.shape)));
    };
    if *scale_rows != rows || *groups == 0 || k % groups != 0 {
        return Err(malformed(&scale.name, format!("FP4 scale grid [{scale_rows}, {groups}] does not tile [{rows}, {k}] \
            (one row of scales per weight row, whole groups along K)")));
    }
    let mut operand = QuantOperand::plain(Encoding::E2m1, weight);
    operand.logical = vec![rows, k];
    operand.scale = Some(BlockScale { tensor: suffix.into(), encoding, direction: ScaleDirection::Multiply,
        rows: RowTiling::Uniform { rows: 1 }, cols: k / groups, grid: scale.shape.clone() });
    if let Some(global) = global {
        let numel: usize = global.shape.iter().product();
        let encoding = scale_encoding(&global.dtype).filter(|_| numel == 1).ok_or_else(|| {
            malformed(&global.name, format!("weight_scale_2 must be one value, found {} {:?}",
                dtype_label(&global.dtype), global.shape))
        })?;
        operand.global_scale = Some(TensorScale { encoding });
    }
    Ok(operand)
}

/// An EXL3 projection: `trellis` I16 `[K/16, N/16, 16 * bits]`, `suh` F16
/// `[K]`, `svh` F16 `[N]` and one `mcg`/`mul1` I32 `[]` or `[1]` marker.
/// The marker remains in companions so routed contracts can require MCG.
/// 3INST and packed `su`/`sv` stay unsupported, named by tensor.
fn detect_exl3(members: &BTreeMap<String, &CheckpointTensor>) -> Result<QuantOperand, Malformed> {
    use crate::formats::exl3_storage::{exl3_module, Exl3StorageError};
    let anchor = members.values().next().map_or(String::new(), |t| t.meta.name.clone());
    let stem = split_stem(&anchor).0.to_owned();
    let suffixes: Vec<&str> = members.keys().map(String::as_str).collect();
    let module = exl3_module(&stem, &suffixes, |suffix| members.get(suffix).map(|tensor| &tensor.meta))
        .map_err(|error| match error {
            Exl3StorageError::Malformed { tensor, reason } => Malformed { tensor, reason },
            Exl3StorageError::Unsupported { tensor, variant } => {
                Malformed { tensor, reason: format!("{variant} (expected MCG/MUL1 with unpacked suh/svh)") }
            }
            other => Malformed { tensor: stem.clone(), reason: other.to_string() },
        })?;
    let trellis = &members["trellis"].meta;
    let (k, n, bits) = (module.input_features, module.output_features, module.bits);
    Ok(QuantOperand {
        encoding: Encoding::Exl3 { bits },
        container: "i16".into(),
        logical: vec![n, k],
        stored: trellis.shape.clone(),
        scale: None,
        global_scale: None,
        input_scale: None,
        companions: vec![module.codebook.marker().into(), "suh".into(), "svh".into()],
    })
}
