//! Checkpoint-native coordinator layouts and shared runtime/planner admission.
use super::{Header, files, number};
use anyhow::{Context, ensure};
use serde::Serialize;
use std::collections::BTreeSet;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

mod ownership;
pub mod programs;
pub mod scratch;
pub use ownership::{DenseAxis, DenseExecution, DenseOwnership, DenseRole, DenseSelection, DenseTransform, VocabularyRole, KvBGeometry, KvBLayout};

/// One bounded native absorption step, shared by admission and loading.
pub use crate::shared::dense_read::STEP_BYTES as DENSE_STEP_BYTES;

type Result<T> = std::result::Result<T, DenseError>;

#[derive(Debug, thiserror::Error)]
pub enum DenseError {
    #[error("EXLR8 dense: {0:#}")]
    Invalid(#[from] anyhow::Error),
    #[error("EXLR8 dense: {tensor}: MXFP8 input slice {first}..{end} cuts a K32 scale block")]
    Mxfp8ScaleBoundary { tensor: String, first: usize, end: usize },
    #[error("EXLR8 dense: {tensor}: physical {storage:?} storage contradicts {execution:?} execution axis")]
    StorageExecutionAxis { tensor: String, storage: ShardSide, execution: DenseAxis },
    #[error("EXLR8 dense: {path}: native metadata required: {reason}")]
    NativeMetadata { path: PathBuf, reason: String },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum DenseEncoding {
    Trellis { bits: usize },
    Int4G32,
    Mxfp8,
    BlockFp8,
    RowFp8,
    Bf16,
    F32,
    F16,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum ShardSide {
    Replicated,
    Output,
    Input,
}

#[derive(Debug, Clone, Serialize)]
pub struct DenseRange {
    pub path: PathBuf,
    pub offset: u64,
    pub bytes: usize,
}

impl DenseRange {
    fn slice(&self, first: usize, bytes: usize) -> anyhow::Result<Self> {
        ensure!(
            first
                .checked_add(bytes)
                .is_some_and(|end| end <= self.bytes),
            "dense range slice outside tensor"
        );
        Ok(Self {
            path: self.path.clone(),
            offset: self
                .offset
                .checked_add(first as u64)
                .context("dense file offset overflow")?,
            bytes,
        })
    }

    fn read(&self, out: &mut [u8]) -> anyhow::Result<()> {
        ensure!(
            out.len() == self.bytes,
            "dense range destination length mismatch"
        );
        let mut file = File::open(&self.path)?;
        file.seek(SeekFrom::Start(self.offset))?;
        file.read_exact(out).context("dense range read")
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct DenseWindow {
    pub row_bytes: usize,
    pub rows: std::ops::Range<usize>,
    pub cols: std::ops::Range<usize>,
}

#[derive(Debug, Clone, Serialize)]
pub struct DenseComponent {
    pub suffix: String,
    pub dtype: String,
    pub shape: Vec<usize>,
    pub source: DenseRange,
    /// Strided native bytes packed directly into one device allocation.
    pub window: Option<DenseWindow>,
}

impl DenseComponent {
    pub fn resident_bytes(&self) -> Result<usize> {
        match &self.window {
            Some(w) => product(&[w.rows.len(), w.cols.len()]).map_err(Into::into),
            None => Ok(self.source.bytes),
        }
    }

    /// The same checked row/column windows consumed by the mount and push loaders.
    pub fn read(&self, out: &mut [u8]) -> Result<()> {
        (|| -> anyhow::Result<()> {
            ensure!(out.len() == self.resident_bytes()?, "dense component read length mismatch");
            if let Some(w) = &self.window {
                ensure!(w.row_bytes > 0 && w.rows.start <= w.rows.end && w.cols.start <= w.cols.end
                    && w.cols.end <= w.row_bytes
                    && product(&[w.rows.end, w.row_bytes])? <= self.source.bytes,
                    "dense component window outside stored extent");
                let mut file = File::open(&self.source.path)?;
                for (at, row) in w.rows.clone().enumerate() {
                    let offset = sum([product(&[row, w.row_bytes])?, w.cols.start])?;
                    file.seek(SeekFrom::Start(self.source.offset.checked_add(offset as u64)
                        .context("dense component file offset overflow")?))?;
                    file.read_exact(&mut out[at * w.cols.len()..(at + 1) * w.cols.len()])?;
                }
                Ok(())
            } else { self.source.read(out) }
        })().map_err(Into::into)
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct DenseRecord {
    pub tensor: String,
    pub encoding: DenseEncoding,
    pub shape: Vec<usize>,
    /// Checkpoint physical storage axis, never execution ownership.
    pub side: ShardSide,
    pub ownership: DenseOwnership,
    pub channels_per_shard: usize,
    /// Zero is a private trellis rotation, not an absent rotation.
    pub rotation_id: usize,
    pub rotation: Option<String>,
    pub hessian_solved: bool,
    pub components: Vec<DenseComponent>,
    pub native_blocks: usize,
}

#[derive(Debug, Clone)]
pub struct DenseUnit {
    pub block: usize,
    pub source: Vec<DenseRange>,
}

impl DenseUnit {
    pub fn staging_bytes(&self) -> Result<usize> {
        sum(self.source.iter().map(|r| r.bytes)).map_err(Into::into)
    }

    /// Read complete native blocks through checked source extents.
    pub fn read(&self, staging: &mut [u8]) -> Result<()> {
        (|| -> anyhow::Result<()> {
            ensure!(
                staging.len() == sum(self.source.iter().map(|r| r.bytes))?,
                "dense unit staging length mismatch"
            );
            let mut at = 0;
            for source in &self.source {
                source.read(&mut staging[at..at + source.bytes])?;
                at += source.bytes;
            }
            Ok(())
        })()
        .map_err(Into::into)
    }
}

#[derive(Debug, Clone)]
pub struct DenseSlice {
    pub tensor: String,
    pub absorbed: Option<KvBLayout>,
    pub transform: DenseTransform,
    pub axis: DenseAxis,
    pub first_channel: usize,
    pub channels: usize,
    pub side: ShardSide,
    pub components: Vec<DenseComponent>,
    pub units: Vec<DenseUnit>,
}

#[derive(Debug, Clone)]
pub struct DensePlan {
    pub rank: usize,
    pub world: usize,
    pub tensors: Vec<DenseSlice>,
    /// Sum of exact native operand allocation requests (each at least 256 bytes).
    pub resident_bytes: usize,
    /// Loading-only peak: reusable pinned batch plus retained full-row cache.
    /// Never part of resident_bytes or steady-state device admission.
    pub read_staging_bytes: usize,
}

fn sum(values: impl IntoIterator<Item = usize>) -> anyhow::Result<usize> {
    values.into_iter().try_fold(0usize, |a, b| {
        a.checked_add(b).context("dense byte count overflow")
    })
}

fn product(values: &[usize]) -> anyhow::Result<usize> {
    values.iter().try_fold(1usize, |a, &b| {
        a.checked_mul(b).context("dense shape overflow")
    })
}

/// Native operands retain the loader's minimum size, without padding to pointer
/// alignment: cuteafd_alloc_device_buffer forwards and reports the exact request.
pub fn allocation(bytes: usize) -> usize { bytes.max(256) }

fn lane_allocation(bytes: usize) -> anyhow::Result<usize> {
    allocation(bytes)
        .checked_add(255)
        .map(|v| v / 256 * 256)
        .context("dense allocation overflow")
}

fn component(
    header: &Header,
    path: &Path,
    name: &str,
    suffix: &str,
    dtype: &str,
    shape: &[usize],
) -> anyhow::Result<DenseComponent> {
    let meta = &header.value[name];
    ensure!(
        meta["dtype"] == dtype && meta["shape"] == serde_json::json!(shape),
        "dense component {name} requires {dtype}{shape:?}"
    );
    let size = match dtype {
        "U8" => 1,
        "BF16" | "F16" => 2,
        "F32" => 4,
        "I64" | "U64" => 8,
        _ => anyhow::bail!("unsupported dense component dtype {dtype}"),
    };
    let bytes = product(shape)?
        .checked_mul(size)
        .context("dense extent overflow")?;
    let (offset, extent) = header.extent(name)?;
    ensure!(
        extent == bytes as u64,
        "dense component byte extent mismatch for {name}"
    );
    Ok(DenseComponent {
        window: None,
        suffix: suffix.into(),
        dtype: dtype.into(),
        shape: shape.to_vec(),
        source: DenseRange {
            path: path.to_path_buf(),
            offset,
            bytes,
        },
    })
}

fn inventory_candidates(snapshot: &Path) -> Result<Vec<(PathBuf, Header)>> {
    Ok(inventory_headers(snapshot)?.into_iter().filter_map(|(path, header, vision)|
        (!vision).then_some((path, header))).collect())
}

/// Bundled plain vision remains in checkpoint identity, but is not native dense storage.
pub(super) fn inventory_headers(snapshot: &Path) -> Result<Vec<(PathBuf, Header, bool)>> {
    use crate::plan::{checkpoint::Checkpoint, family, Component};
    // Construct only the configuration view: opening the checkpoint inventory here recurses.
    let checkpoint = Checkpoint { snapshot: snapshot.into(), config: super::read_json(&snapshot.join("config.json"))?,
        quantize_config: None, weight_map: Default::default(), tensors: vec![], missing_shards: vec![], shard_bytes: 0 };
    let family = family::detect(&checkpoint).context("unsupported EXLR8 dense family")?;
    let model = family.open(&checkpoint).map_err(anyhow::Error::new)?;
    let mut paths = Vec::new();
    for path in files(snapshot, "dense")? {
        let header = Header::open(&path)?;
        let entries = header.value.as_object().context("dense header must be an object")?;
        let names: Vec<_> = entries.keys().filter(|name| *name != "__metadata__").collect();
        let vision = model.spec().vision && !names.is_empty() && names.iter().all(|name|
            family.classify(model.spec(), name).is_some_and(|role| role.component == Component::Vision));
        if !vision {
            header.validate().map_err(|error| DenseError::NativeMetadata { path: path.clone(), reason: format!("{error:#}") })?;
        }
        paths.push((path, header, vision));
    }
    Ok(paths)
}

/// Read headers and tiny record tables only. The official checkpoint is not an input.
impl DenseRecord {
    pub fn inventory(snapshot: &Path) -> Result<Vec<Self>> {
        let paths = inventory_candidates(snapshot)?;
        Self::inventory_inner(snapshot, paths).map_err(Into::into)
    }

    fn inventory_inner(snapshot: &Path, paths: Vec<(PathBuf, Header)>) -> anyhow::Result<Vec<Self>> {
        let config = super::read_json(&snapshot.join("config.json"))?;
        let family = ownership::DenseFamily::from_hf(&config)?;
        let config = config.get("text_config").unwrap_or(&config);
        let layers = number(config, "num_hidden_layers")?;
        let mut records = Vec::new();
        let mut names = BTreeSet::new();
        for (path, header) in paths {
            let metadata = &header.value["__metadata__"];
            let layer = metadata["layer"]
                .as_str()
                .map(str::parse::<usize>)
                .transpose()?;
            if layer.is_some_and(|l| l >= layers) {
                continue;
            }
            let entries = header
                .value
                .as_object()
                .context("dense header must be an object")?;
            let mut extents = entries
                .keys()
                .filter(|n| *n != "__metadata__")
                .map(|n| header.extent(n).map(|(start, bytes)| (start, bytes, n)))
                .collect::<anyhow::Result<Vec<_>>>()?;
            extents.sort_by_key(|&(start, _, _)| start);
            let mut end = header.start;
            for (start, bytes, name) in extents {
                ensure!(start >= end, "overlapping dense tensor {name}");
                end = start
                    .checked_add(bytes)
                    .context("dense file extent overflow")?;
            }
            let rotations = metadata["rotations"]
                .as_str()
                .unwrap_or("attn_in,q_resid,mlp_in")
                .split(',')
                .map(str::trim)
                .collect::<Vec<_>>();
            ensure!(
                rotations.iter().all(|r| !r.is_empty())
                    && rotations.iter().collect::<BTreeSet<_>>().len() == rotations.len(),
                "invalid dense rotation names"
            );
            let sources = super::source_names(metadata)?;
            let logical = |stem: &str, weight: bool| {
                super::logical_name(metadata, sources.as_ref(), layer, stem, weight)
            };
            let mut claimed = BTreeSet::new();
            for name in entries.keys().filter(|n| n.ends_with(".record")) {
                component(&header, &path, name, "record", "I64", &[8])?;
                let raw = header.read(&path, name)?;
                let mut r = [0usize; 8];
                for (i, bytes) in raw.chunks_exact(8).enumerate() {
                    r[i] = usize::try_from(i64::from_le_bytes(bytes.try_into().unwrap()))
                        .context("negative or oversized dense record field")?;
                }
                let [code, n, k, bits, side, unit, rotation_id, solved] = r;
                ensure!(
                    n > 0 && k > 0 && solved <= 1,
                    "invalid dense geometry/solver in {name}"
                );
                let side = match side {
                    0 => ShardSide::Replicated,
                    1 => ShardSide::Output,
                    2 => ShardSide::Input,
                    _ => anyhow::bail!("invalid dense shard side in {name}"),
                };
                ensure!(
                    (side == ShardSide::Replicated && unit == 0)
                        || (side != ShardSide::Replicated && unit > 0),
                    "dense shard unit disagrees with side in {name}"
                );
                let extent = if side == ShardSide::Input { k } else { n };
                ensure!(
                    side == ShardSide::Replicated || extent % unit == 0,
                    "dense shard unit does not divide dimension in {name}"
                );
                ensure!(
                    code == 1 || (bits == 0 && rotation_id == 0),
                    "non-trellis dense record has trellis fields in {name}"
                );
                let stem = name.trim_end_matches(".record");
                let mut specs: Vec<(&str, &str, Vec<usize>)> = Vec::new();
                let encoding = match code {
                    1 => {
                        ensure!(
                            (2..=4).contains(&bits)
                                && n % 128 == 0
                                && k % 128 == 0
                                && rotation_id <= rotations.len(),
                            "unsupported dense trellis geometry/rotation in {name}"
                        );
                        let (blocks, other) = if side == ShardSide::Input {
                            (k / 128, n)
                        } else {
                            (n / 128, k)
                        };
                        specs.extend([
                            (
                                "trellis",
                                "U8",
                                vec![blocks, product(&[other, 128, bits])? / 8],
                            ),
                            ("suh", "F16", vec![k]),
                            ("svh", "F16", vec![n]),
                        ]);
                        DenseEncoding::Trellis { bits }
                    }
                    2 | 3 => {
                        ensure!(
                            k % 128 == 0 && n % 32 == 0,
                            "unsupported dense group geometry in {name}"
                        );
                        let packed = if code == 2 { 2 } else { 1 };
                        let payload = if side == ShardSide::Input {
                            vec![k / 128, n, 128 / packed]
                        } else {
                            vec![n, k / packed]
                        };
                        let scales = if side == ShardSide::Input {
                            vec![k / 128, n, 4]
                        } else {
                            vec![n, k / 32]
                        };
                        specs.extend(if code == 2 {
                            vec![("q4", "U8", payload), ("s4", "F16", scales)]
                        } else {
                            vec![("fp8", "U8", payload), ("e8m0", "U8", scales)]
                        });
                        if code == 2 {
                            DenseEncoding::Int4G32
                        } else {
                            DenseEncoding::Mxfp8
                        }
                    }
                    4 => {
                        ensure!(
                            side != ShardSide::Input,
                            "input-sharded block-FP8 needs a native strided slice"
                        );
                        specs.extend([
                            ("fp8", "U8", vec![n, k]),
                            ("scale", "F32", vec![n.div_ceil(128), k.div_ceil(128)]),
                        ]);
                        DenseEncoding::BlockFp8
                    }
                    5 => {
                        ensure!(
                            side != ShardSide::Input,
                            "row-FP8 requires output-major storage"
                        );
                        specs.extend([("fp8", "U8", vec![n, k]), ("scale", "F32", vec![n])]);
                        DenseEncoding::RowFp8
                    }
                    6 | 7 => {
                        ensure!(
                            side != ShardSide::Input,
                            "input-sharded plain dense needs a native strided slice"
                        );
                        specs.push(if code == 6 {
                            ("bf16", "BF16", vec![n, k])
                        } else {
                            ("fp32", "F32", vec![n, k])
                        });
                        if code == 6 {
                            DenseEncoding::Bf16
                        } else {
                            DenseEncoding::F32
                        }
                    }
                    _ => anyhow::bail!(
                        "unsupported dense encoding {code} in {name}: add its native kernel"
                    ),
                };
                let components = specs
                    .into_iter()
                    .map(|(suffix, dtype, shape)| {
                        let key = format!("{stem}.{suffix}");
                        claimed.insert(key.clone());
                        component(&header, &path, &key, suffix, dtype, &shape)
                    })
                    .collect::<anyhow::Result<Vec<_>>>()?;
                let checksum = format!("{stem}.checksum");
                component(
                    &header,
                    &path,
                    &checksum,
                    "checksum",
                    "U64",
                    &[extent.div_ceil(128)],
                )?;
                claimed.extend([name.clone(), checksum]);
                let tensor = logical(stem, true)?;
                ensure!(
                    names.insert(tensor.clone()),
                    "duplicate native dense tensor {tensor}"
                );
                records.push(Self {
                    ownership: family.ownership(&tensor)?,
                    tensor,
                    encoding,
                    shape: vec![n, k],
                    side,
                    channels_per_shard: unit,
                    rotation_id,
                    rotation: (rotation_id > 0).then(|| rotations[rotation_id - 1].into()),
                    hessian_solved: solved == 1,
                    components,
                    native_blocks: extent.div_ceil(128),
                });
            }
            for (name, meta) in entries {
                if name.starts_with("__") || claimed.contains(name) {
                    continue;
                }
                let dtype = meta["dtype"].as_str().context("dense dtype missing")?;
                let encoding = match dtype {
                    "BF16" => DenseEncoding::Bf16,
                    "F16" => DenseEncoding::F16,
                    "F32" => DenseEncoding::F32,
                    _ => anyhow::bail!("unsupported unrecorded dense tensor {name}: {dtype}"),
                };
                let shape: Vec<usize> = serde_json::from_value(meta["shape"].clone())?;
                ensure!(
                    !shape.is_empty() && shape.iter().all(|&v| v > 0),
                    "invalid dense plain tensor {name}"
                );
                let tensor = logical(name, false)?;
                ensure!(
                    names.insert(tensor.clone()),
                    "duplicate native dense tensor {tensor}"
                );
                records.push(Self {
                    ownership: family.ownership(&tensor)?,
                    tensor,
                    encoding,
                    shape: shape.clone(),
                    side: ShardSide::Replicated,
                    channels_per_shard: 0,
                    rotation_id: 0,
                    rotation: None,
                    hessian_solved: false,
                    components: vec![component(&header, &path, name, "plain", dtype, &shape)?],
                    native_blocks: 0,
                });
            }
        }
        ensure!(!records.is_empty(), "empty native dense inventory");
        for record in &records { record.validate_ownership()?; }
        records.sort_by(|a, b| a.tensor.cmp(&b.tensor));
        Ok(records)
    }

    fn channels(&self) -> anyhow::Result<usize> {
        self.shape
            .get(usize::from(self.side == ShardSide::Input))
            .copied()
            .filter(|&n| n > 0)
            .context("invalid dense logical shape")
    }

    fn ranges(&self, first: usize, count: usize) -> anyhow::Result<Vec<DenseComponent>> {
        let extent = self.channels()?;
        self.components
            .iter()
            .map(|component| {
                let replicated = matches!(self.encoding, DenseEncoding::Trellis { .. })
                    && ((component.suffix == "suh" && self.side != ShardSide::Input)
                        || (component.suffix == "svh" && self.side == ShardSide::Input));
                let (begin, bytes) = if replicated {
                    (0, component.source.bytes)
                } else if self.encoding == DenseEncoding::BlockFp8 && component.suffix == "scale" {
                    ensure!(
                        first % 128 == 0 && (count % 128 == 0 || first + count == extent),
                        "unaligned block-FP8 scale slice"
                    );
                    let row = self.shape[1]
                        .div_ceil(128)
                        .checked_mul(4)
                        .context("dense scale stride overflow")?;
                    (
                        product(&[first / 128, row])?,
                        product(&[count.div_ceil(128), row])?,
                    )
                } else {
                    ensure!(
                        component.source.bytes % extent == 0,
                        "dense component stride is not integral"
                    );
                    let stride = component.source.bytes / extent;
                    (product(&[first, stride])?, product(&[count, stride])?)
                };
                let mut sliced = component.clone();
                sliced.source = component.source.slice(begin, bytes)?;
                if !replicated {
                    let blocks = (matches!(self.encoding, DenseEncoding::Trellis { .. })
                        && component.suffix == "trellis")
                        || (self.side == ShardSide::Input
                            && matches!(
                                self.encoding,
                                DenseEncoding::Int4G32 | DenseEncoding::Mxfp8
                            ))
                        || (self.encoding == DenseEncoding::BlockFp8
                            && component.suffix == "scale");
                    sliced.shape[0] = if blocks { count.div_ceil(128) } else { count };
                }
                Ok(sliced)
            })
            .collect()
    }

    pub fn slice(&self, first: usize, channels: usize) -> Result<DenseSlice> {
        (|| -> anyhow::Result<_> {
            let extent = self.channels()?;
            ensure!(
                channels > 0 && first.checked_add(channels).is_some_and(|end| end <= extent),
                "dense channel slice outside tensor"
            );
            ensure!(
                !matches!(self.encoding, DenseEncoding::Trellis { .. } | DenseEncoding::Int4G32 | DenseEncoding::BlockFp8)
                    && self.side != ShardSide::Input
                    || first % 128 == 0 && (channels % 128 == 0 || first + channels == extent),
                "dense slice must contain complete native blocks"
            );
            ensure!(
                self.native_blocks == extent.div_ceil(128),
                "dense native block count differs from channel extent"
            );
            let components = self.ranges(first, channels)?;
            let units = (first..first + channels)
                .step_by(128)
                .map(|at| {
                    let count = (first + channels - at).min(128);
                    let mut ranges = self.ranges(at, count)?;
                    // Keep the native component order used by streaming loaders.
                    if matches!(self.encoding, DenseEncoding::Trellis { .. })
                        && self.side != ShardSide::Input
                    {
                        ranges.swap(1, 2);
                    }
                    Ok(DenseUnit {
                        block: at / 128,
                        source: ranges.into_iter().map(|c| c.source).collect(),
                    })
                })
                .collect::<anyhow::Result<Vec<_>>>()?;
            Ok(DenseSlice {
                absorbed: None,
                transform: self.ownership.transform,
                axis: if self.side == ShardSide::Input { DenseAxis::Input } else { DenseAxis::Output },
                tensor: self.tensor.clone(),
                first_channel: first,
                channels,
                side: self.side,
                components,
                units,
            })
        })()
        .map_err(Into::into)
    }
}

impl DenseRecord {
    /// Select execution channels without changing checkpoint quantization or resident layout.
    pub fn slice_axis(&self, axis: DenseAxis, first: usize, channels: usize) -> Result<DenseSlice> {
        if matches!((self.side, axis), (ShardSide::Output, DenseAxis::Input) | (ShardSide::Input, DenseAxis::Output)) {
            return Err(DenseError::StorageExecutionAxis { tensor: self.tensor.clone(), storage: self.side, execution: axis });
        }
        let end = first.checked_add(channels).ok_or_else(|| anyhow::anyhow!("dense channel overflow"))?;
        if self.encoding == DenseEncoding::Mxfp8 && axis == DenseAxis::Input
            && (first % 32 != 0 || end % 32 != 0) {
            return Err(DenseError::Mxfp8ScaleBoundary { tensor: self.tensor.clone(), first, end });
        }
        let stored = if self.side == ShardSide::Input { DenseAxis::Input } else { DenseAxis::Output };
        if axis == stored && self.native_blocks > 0 { return self.slice(first, channels); }
        (|| -> anyhow::Result<_> {
            let dimension = usize::from(axis == DenseAxis::Input);
            let extent = *self.shape.get(dimension).context("dense slice axis outside shape")?;
            ensure!(channels > 0 && end <= extent, "dense channel slice outside tensor");
            if matches!(self.encoding, DenseEncoding::Trellis { .. }) {
                ensure!(first % 128 == 0 && channels % 128 == 0,
                    "dense trellis slice must contain complete native blocks");
            }
            let mut components = Vec::new();
            for component in &self.components {
                let mut c = component.clone();
                ensure!(c.window.is_none(), "dense component already sliced");
                let vector_axis = match c.suffix.as_str() {
                    "suh" => Some(DenseAxis::Input), "svh" => Some(DenseAxis::Output), _ => None,
                };
                if let Some(vector_axis) = vector_axis {
                    if vector_axis == axis {
                        c.source = c.source.slice(product(&[first, 2])?, product(&[channels, 2])?)?;
                        c.shape = vec![channels];
                    }
                } else if axis == DenseAxis::Output && (self.native_blocks == 0 || first == 0 && channels == extent) {
                    ensure!(c.source.bytes % extent == 0, "dense row stride is not integral");
                    let stride = c.source.bytes / extent;
                    c.source = c.source.slice(product(&[first, stride])?, product(&[channels, stride])?)?;
                    c.shape[0] = channels;
                } else {
                    let (rows, row_bytes, start, width) = match self.encoding {
                        DenseEncoding::Trellis { bits } => {
                            let blocks = self.shape[usize::from(stored == DenseAxis::Input)] / 128;
                            (blocks, product(&[extent, 128, bits])? / 8,
                                product(&[first, 128, bits])? / 8, product(&[channels, 128, bits])? / 8)
                        }
                        DenseEncoding::Mxfp8 | DenseEncoding::Int4G32 if stored == DenseAxis::Input => {
                            let stride = match c.suffix.as_str() {
                                "fp8" => 128, "q4" => 64, "e8m0" => 4, "s4" => 8,
                                _ => anyhow::bail!("unsupported native input-major component"),
                            };
                            c.shape[1] = channels;
                            (self.shape[1] / 128, product(&[extent, stride])?, product(&[first, stride])?, product(&[channels, stride])?)
                        }
                        _ if axis == DenseAxis::Input => {
                            let (divisor, element) = match (self.encoding, c.suffix.as_str()) {
                                (DenseEncoding::Mxfp8, "e8m0") => (32, 1),
                                (DenseEncoding::Int4G32, "q4") => (2, 1),
                                (DenseEncoding::Int4G32, "s4") => (32, 2),
                                (DenseEncoding::BlockFp8, "scale") => (128, 4),
                                (DenseEncoding::RowFp8, "scale") => { components.push(c); continue; },
                                (_, _) => (1, match c.dtype.as_str() { "U8" => 1, "BF16" | "F16" => 2, "F32" => 4,
                                    _ => anyhow::bail!("unsupported native dense dtype") }),
                            };
                            ensure!(first % divisor == 0 && channels % divisor == 0,
                                "dense input slice cuts a scale or packed value block");
                            let rows = c.shape[0];
                            c.shape[1] = channels / divisor;
                            (rows, product(&[extent / divisor, element])?, product(&[first / divisor, element])?,
                                product(&[channels / divisor, element])?)
                        }
                        _ => anyhow::bail!("unsupported native storage/execution slice"),
                    };
                    ensure!(product(&[rows, row_bytes])? == c.source.bytes && start + width <= row_bytes,
                        "dense strided component extent mismatch");
                    if matches!(self.encoding, DenseEncoding::Trellis { .. }) { c.shape[1] = width; }
                    c.window = Some(DenseWindow { row_bytes, rows: 0..rows, cols: start..start + width });
                }
                components.push(c);
            }
            Ok(DenseSlice { tensor: self.tensor.clone(), absorbed: None, transform: self.ownership.transform,
                axis, first_channel: first,
                channels, side: self.side, components, units: vec![] })
        })().map_err(Into::into)
    }
}

impl DensePlan {
    /// Streaming owns one staging buffer; retained/pinned host storage is caller-owned.
    pub fn admit_host(&self, available: usize, reserved: usize) -> Result<usize> {
        (|| -> anyhow::Result<_> {
            let total = sum([self.read_staging_bytes, reserved])?;
            ensure!(
                total <= available,
                "native dense host admission needs {total} bytes, available {available}"
            );
            Ok(total)
        })()
        .map_err(Into::into)
    }

    pub fn for_rank(records: &[DenseRecord], world: usize, rank: usize) -> Result<Self> {
        Self::for_rank_owned(records, world, rank, &std::collections::BTreeMap::new())
    }

    /// `whole_mla` is a planner decision; records only expose eligibility and split facts.
    pub fn for_rank_owned(records: &[DenseRecord], world: usize, rank: usize,
        whole_mla: &std::collections::BTreeMap<usize, usize>) -> Result<Self> {
        for record in records {
            if let DenseExecution::Split(axis) = record.ownership.execution {
                if matches!((record.side, axis), (ShardSide::Output, DenseAxis::Input) | (ShardSide::Input, DenseAxis::Output)) {
                    return Err(DenseError::StorageExecutionAxis { tensor: record.tensor.clone(), storage: record.side, execution: axis });
                }
                if record.encoding == DenseEncoding::Mxfp8 && axis == DenseAxis::Input
                    && world > 0 && record.shape.get(1).is_some_and(|k| k % world == 0 && (k / world) % 32 != 0) {
                    return Err(DenseError::Mxfp8ScaleBoundary { tensor: record.tensor.clone(), first: 0, end: record.shape[1] / world });
                }
            }
        }
        (|| -> anyhow::Result<Self> {
            ensure!(
                (1..=2).contains(&world) && rank < world,
                "dense coordinator rank/world invalid"
            );
            ensure!(!records.is_empty(), "empty dense coordinator plan");
            let mut tensors = Vec::new();
            let mut resident_bytes = 0usize;
            let mut pinned_staging_bytes = 0usize;
            let mut cached_rows_bytes = 0usize;
            for record in records {
                record.validate_ownership()?;
                let owner = record.ownership.whole_mla.and_then(|layer| whole_mla.get(&layer)).copied();
                ensure!(owner.is_none_or(|owner| world == 2 && owner < world), "invalid whole-MLA owner");
                if owner.is_some_and(|owner| owner != rank)
                    || record.ownership.execution == DenseExecution::RankZero && rank != 0 { continue; }
                let (parts, part) = if owner.is_some() { (1, 0) } else { (world, rank) };
                let slice = match record.ownership.execution {
                    DenseExecution::AbsorbHeads(geometry) => {
                        let layout = geometry.layout(parts, part)?;
                        DenseSlice { tensor: record.tensor.clone(), first_channel: layout.first_head,
                            channels: layout.heads, absorbed: Some(layout), transform: DenseTransform::Identity,
                            axis: DenseAxis::Output,
                            side: record.side, components: vec![], units: vec![] }
                    }
                    DenseExecution::Split(axis) => {
                        let extent = *record.shape.get(usize::from(axis == DenseAxis::Input))
                            .context("dense execution axis outside shape")?;
                        ensure!(extent % parts == 0, "dense TP dimension does not divide world");
                        record.slice_axis(axis, part * (extent / parts), extent / parts)?
                    }
                    DenseExecution::RankZero | DenseExecution::Replicated => {
                        if record.native_blocks > 0 { record.slice(0, record.channels()?)? }
                        else { DenseSlice { tensor: record.tensor.clone(), first_channel: 0,
                            channels: record.channels()?, absorbed: None, transform: record.ownership.transform,
                            axis: DenseAxis::Output,
                            side: record.side, components: record.components.clone(), units: vec![] } }
                    }
                };
                if let Some(layout) = &slice.absorbed {
                    for bytes in layout.allocations { resident_bytes = sum([resident_bytes, allocation(bytes)])?; }
                } else {
                    for component in &slice.components {
                        resident_bytes = sum([resident_bytes, slice.transform.allocation_bytes(component)?])?;
                    }
                }
                for unit in &slice.units {
                    pinned_staging_bytes = pinned_staging_bytes.max(unit.staging_bytes()?);
                }
                for component in &slice.components {
                    let window = slice.transform.read_window(component)?;
                    let batch = crate::shared::dense_read::ReadBatch::new(
                        window.rows.len(), window.row_bytes, window.cols.len(), DENSE_STEP_BYTES)?;
                    pinned_staging_bytes = pinned_staging_bytes.max(batch.pinned_bytes());
                    cached_rows_bytes = cached_rows_bytes.max(batch.cached_bytes);
                }
                if slice.absorbed.is_some() {
                    let staged = sum(record.components.iter().map(|c| c.source.bytes))?.min(DENSE_STEP_BYTES);
                    pinned_staging_bytes = pinned_staging_bytes.max(staged.max(crate::shared::dense_read::PINNED_FLOOR_BYTES));
                }
                tensors.push(slice);
            }
            Ok(Self {
                rank,
                world,
                tensors,
                resident_bytes,
                read_staging_bytes: sum([pinned_staging_bytes, cached_rows_bytes])?,
            })
        })()
        .map_err(Into::into)
    }

    /// All lane allocations and caller reservations precede the first weight allocation.
    pub fn admit(
        &self,
        available: usize,
        lane_workspace: usize,
        lanes: usize,
        reserved: usize,
    ) -> Result<usize> {
        (|| -> anyhow::Result<_> {
            ensure!(lanes > 0, "native dense needs at least one execution lane");
            let workspace = lane_allocation(lane_workspace)?
                .checked_mul(lanes)
                .context("dense lane workspace overflow")?;
            let total = sum([self.resident_bytes, workspace, reserved])?;
            ensure!(
                total <= available,
                "native dense admission needs {total} bytes, available {available}"
            );
            Ok(total)
        })()
        .map_err(Into::into)
    }
}

/// Exact resident allocation groups of an already-owned native EXLR8 plan.
/// The same path admits either family's backbone or separately owned vocabulary;
/// selection rejects MTP, while transformed bytes describe the loaded operand.
/// Source staging belongs only to the plan's loading peak.
pub fn resident_native(selection: &DenseSelection, layers: usize, plan: &DensePlan)
    -> Result<Vec<(String, String, u64)>> {
    (|| -> anyhow::Result<_> {
        let mut groups = std::collections::BTreeMap::<(String, String), u64>::new();
        let mut put = |group: &str, format: &str, bytes: usize| -> anyhow::Result<()> {
            ensure!(bytes > 0, "empty native dense allocation in {group}");
            let total = groups.entry((group.into(), format.into())).or_default();
            *total = total.checked_add(u64::try_from(bytes)?).context("native resident group overflow")?;
            Ok(())
        };
        for slice in &plan.tensors {
            let group = selection.resident_group(&slice.tensor, layers)?;
            if let Some(layout) = &slice.absorbed {
                for (format, bytes) in ["fp8", "fp8-scale", "fp8", "fp8-scale"].into_iter().zip(layout.allocations) {
                    put(group, format, allocation(bytes))?;
                }
            } else {
                for component in &slice.components {
                    let format = if slice.transform == DenseTransform::WidenBf16 { "f32".into() }
                        else { component.dtype.to_lowercase() };
                    put(group, &format, slice.transform.allocation_bytes(component)?)?;
                }
            }
        }
        let total = groups.values().try_fold(0u64, |total, &bytes|
            total.checked_add(bytes).context("native resident total overflow"))?;
        ensure!(total == u64::try_from(plan.resident_bytes)?, "native resident groups differ from owned plan");
        Ok(groups.into_iter().map(|((g, f), b)| (g, f, b)).collect())
    })().map_err(Into::into)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use std::io::Write;

    fn write(
        root: &Path,
        fields: [i64; 8],
        parts: Vec<(&str, &str, Vec<usize>, Vec<u8>)>,
    ) -> PathBuf {
        std::fs::create_dir_all(root.join("dense")).unwrap();
        std::fs::write(root.join("config.json"), crate::plan::testing::glm5_config().to_string()).unwrap();
        let mut entries = serde_json::Map::new();
        entries.insert(
            "__metadata__".into(),
            json!({"format":super::super::FORMAT,
            "scheme":super::super::SCHEME,"layer":"0", "source_names": json!({
                "mlp.gate_proj": match fields[4] {
                    2 => "model.layers.0.mlp.down_proj.weight",
                    0 => "model.layers.0.self_attn.q_a_proj.weight",
                    _ => "model.layers.0.mlp.gate_proj.weight",
                },
                "input_layernorm.weight": "model.layers.0.input_layernorm.weight",
            }).to_string()}),
        );
        let mut payload = fields
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect::<Vec<_>>();
        entries.insert(
            "mlp.gate_proj.record".into(),
            json!({"dtype":"I64","shape":[8],"data_offsets":[0,64]}),
        );
        for (suffix, dtype, shape, data) in parts {
            let begin = payload.len();
            payload.extend(data);
            entries.insert(
                format!("mlp.gate_proj.{suffix}"),
                json!({"dtype":dtype,"shape":shape,"data_offsets":[begin,payload.len()]}),
            );
        }
        let header = serde_json::to_vec(&entries).unwrap();
        let path = root.join("dense/layer.safetensors");
        let mut f = File::create(&path).unwrap();
        f.write_all(&(header.len() as u64).to_le_bytes()).unwrap();
        f.write_all(&header).unwrap();
        f.write_all(&payload).unwrap();
        path
    }

    fn trellis(root: &Path, side: i64, unit: i64, rotation: i64) -> PathBuf {
        let payload = (0..16384).map(|i| (i % 251) as u8).collect::<Vec<_>>();
        let suh = vec![17u8; 512];
        let svh = vec![23u8; 256];
        let checksums = vec![0; if side == 2 { 16 } else { 8 }];
        write(
            root,
            [1, 128, 256, 4, side, unit, rotation, 1],
            vec![
                (
                    "trellis",
                    "U8",
                    if side == 2 {
                        vec![2, 8192]
                    } else {
                        vec![1, 16384]
                    },
                    payload,
                ),
                ("suh", "F16", vec![256], suh),
                ("svh", "F16", vec![128], svh),
                (
                    "checksum",
                    "U64",
                    vec![if side == 2 { 2 } else { 1 }],
                    checksums,
                ),
            ],
        )
    }

    #[test]
    fn typed_trellis_inventory_and_input_tp_verify_native_bytes() {
        let root = tempfile::tempdir().unwrap();
        trellis(root.path(), 2, 128, 0);
        let records = DenseRecord::inventory(root.path()).unwrap();
        assert_eq!(records[0].encoding, DenseEncoding::Trellis { bits: 4 });
        assert_eq!(records[0].rotation_id, 0);
        assert_eq!(records[0].rotation, None);
        for rank in 0..2 {
            let plan = DensePlan::for_rank(&records, 2, rank).unwrap();
            assert_eq!(plan.tensors[0].first_channel, rank * 128);
            assert_eq!(plan.resident_bytes, 8192 + 256 + 256);
            assert_eq!(plan.read_staging_bytes, crate::shared::dense_read::PINNED_FLOOR_BYTES);
            let unit = &plan.tensors[0].units[0];
            unit.read(&mut vec![0; unit.staging_bytes().unwrap()])
                .unwrap();
            assert!(unit.read(&mut []).is_err());
            assert_eq!(
                plan.admit(plan.resident_bytes + 512, 1, 2, 0).unwrap(),
                plan.resident_bytes + 512
            );
            assert!(plan.admit(plan.resident_bytes + 511, 1, 2, 0).is_err());
            assert!(plan.admit(usize::MAX, usize::MAX, 2, 0).is_err());
            assert!(plan.admit(usize::MAX, 0, 0, 0).is_err());
        }
        let plan = DensePlan::for_rank(&records, 1, 0).unwrap();
        let unit = &plan.tensors[0].units[0];
        let source = &unit.source[0];
        std::fs::OpenOptions::new().write(true).open(&source.path).unwrap()
            .set_len(source.offset + source.bytes as u64 - 1).unwrap();
        assert!(unit.read(&mut vec![0; unit.staging_bytes().unwrap()]).is_err());
    }

    #[test]
    fn replicated_native_records_keep_block_streaming() {
        let root = tempfile::tempdir().unwrap();
        trellis(root.path(), 0, 0, 0);
        let records = DenseRecord::inventory(root.path()).unwrap();
        assert_eq!(records[0].channels_per_shard, 0);
        assert_eq!(records[0].native_blocks, 1);
        for rank in 0..2 {
            let plan = DensePlan::for_rank(&records, 2, rank).unwrap();
            let unit = &plan.tensors[0].units[0];
            unit.read(&mut vec![0; unit.staging_bytes().unwrap()]).unwrap();
        }
    }

    #[test]
    fn output_trellis_component_order_and_shared_rotation_are_preserved() {
        let root = tempfile::tempdir().unwrap();
        trellis(root.path(), 1, 128, 3);
        let records = DenseRecord::inventory(root.path()).unwrap();
        assert_eq!(records[0].rotation.as_deref(), Some("mlp_in"));
        let p = DensePlan::for_rank(&records, 1, 0).unwrap();
        let unit = &p.tensors[0].units[0];
        unit.read(&mut vec![0; unit.staging_bytes().unwrap()])
            .unwrap();
        assert!(DensePlan::for_rank(&records, 2, 0).is_err());
    }

    #[test]
    fn invalid_record_fields_shapes_extents_and_units_fail_before_payload_reads() {
        for (side, unit, rotation) in [
            (3, 128, 0),
            (2, 0, 0),
            (0, 128, 0),
            (2, 192, 0),
            (2, 128, 4),
        ] {
            let root = tempfile::tempdir().unwrap();
            trellis(root.path(), side, unit, rotation);
            assert!(
                DenseRecord::inventory(root.path()).is_err(),
                "{side}/{unit}/{rotation}"
            );
        }
        let root = tempfile::tempdir().unwrap();
        write(
            root.path(),
            [3, 128, 128, 0, 0, 0, 0, 1],
            vec![
                ("fp8", "U8", vec![128, 128], vec![0; 16383]),
                ("e8m0", "U8", vec![128, 4], vec![127; 512]),
                ("checksum", "U64", vec![1], vec![0; 8]),
            ],
        );
        assert!(DenseRecord::inventory(root.path()).is_err());
        for fields in [
            [9, 128, 128, 0, 0, 0, 0, 0],
            [1, 128, 128, 1, 0, 0, 0, 0],
            [1, -1, 128, 4, 0, 0, 0, 0],
            [3, 128, 128, 4, 0, 0, 0, 0],
            [3, 128, 128, 0, 0, 0, 0, 2],
        ] {
            write(root.path(), fields, vec![]);
            assert!(DenseRecord::inventory(root.path()).is_err());
        }
        assert!(product(&[usize::MAX, 2]).is_err());
        assert_eq!(allocation(usize::MAX), usize::MAX);
        assert!(lane_allocation(usize::MAX).is_err());
        assert!(DensePlan::for_rank(&[], 3, 0).is_err());
    }

    #[test]
    fn side2_int4_blocks_and_slice_are_unit_major() {
        let root = tempfile::tempdir().unwrap();
        let data = vec![0x89; 128 * 256 / 2];
        let scales = vec![0x3c; 128 * 256 / 32 * 2];
        let hashes = vec![0; 16];
        write(
            root.path(),
            [2, 128, 256, 0, 2, 128, 0, 1],
            vec![
                ("q4", "U8", vec![2, 128, 64], data),
                ("s4", "F16", vec![2, 128, 4], scales),
                ("checksum", "U64", vec![2], hashes),
            ],
        );
        let r = DenseRecord::inventory(root.path()).unwrap();
        for rank in 0..2 {
            let p = DensePlan::for_rank(&r, 2, rank).unwrap();
            let u = &p.tensors[0].units[0];
            assert_eq!(u.source[0].bytes, 8192);
            assert_eq!(u.source[1].bytes, 1024);
            u.read(&mut vec![0; u.staging_bytes().unwrap()])
                .unwrap();
        }
    }

    fn rewrite_header(path: &Path, edit: impl FnOnce(&mut serde_json::Value)) {
        let mut h = Header::open(path).unwrap();
        let mut f = File::open(path).unwrap();
        f.seek(SeekFrom::Start(h.start)).unwrap();
        let mut payload = Vec::new();
        f.read_to_end(&mut payload).unwrap();
        edit(&mut h.value);
        let bytes = serde_json::to_vec(&h.value).unwrap();
        let mut f = File::create(path).unwrap();
        f.write_all(&(bytes.len() as u64).to_le_bytes()).unwrap();
        f.write_all(&bytes).unwrap();
        f.write_all(&payload).unwrap();
    }

    #[test]
    fn plain_norms_have_no_native_blocks_or_tp_slice() {
        let root = tempfile::tempdir().unwrap();
        let path = trellis(root.path(), 2, 128, 0);
        rewrite_header(&path, |h| {
            // A plain norm can share a shard with records, but never their extents.
            let end = h["mlp.gate_proj.checksum"]["data_offsets"][1]
                .as_u64()
                .unwrap();
            h["input_layernorm.weight"] =
                json!({"dtype":"BF16","shape":[128],"data_offsets":[end,end+256]});
        });
        std::fs::OpenOptions::new()
            .append(true)
            .open(&path)
            .unwrap()
            .write_all(&[0; 256])
            .unwrap();
        let records = DenseRecord::inventory(root.path()).unwrap();
        let norm = records
            .iter()
            .find(|r| r.tensor.ends_with("input_layernorm.weight"))
            .unwrap();
        assert_eq!(norm.native_blocks, 0);
        for rank in 0..2 {
            let p = DensePlan::for_rank(&records, 2, rank).unwrap();
            let norm = p
                .tensors
                .iter()
                .find(|r| r.tensor.ends_with("input_layernorm.weight"))
                .unwrap();
            assert!(norm.units.is_empty());
            assert_eq!(norm.components[0].shape, [128]);
            assert_eq!(norm.components[0].source.bytes, 256);
            assert_eq!(
                p.admit_host(p.read_staging_bytes + 32, 32).unwrap(),
                p.read_staging_bytes + 32
            );
            assert!(p.admit_host(p.read_staging_bytes + 31, 32).is_err());
            assert!(p.admit_host(usize::MAX, usize::MAX).is_err());
        }
    }

    #[test]
    fn header_overlap_orphan_duplicate_and_bad_rotations_are_rejected() {
        let root = tempfile::tempdir().unwrap();
        for edit in [
            (|h: &mut serde_json::Value| {
                h["orphan"] = json!({"dtype":"U8","shape":[1],"data_offsets":[64,65]})
            }) as fn(&mut serde_json::Value),
            |h| h["mlp.gate_proj.suh"]["data_offsets"] = json!([64, 576]),
            |h| h["__metadata__"]["rotations"] = json!("a,a"),
            |h| h["__metadata__"]["rotations"] = json!("a,,c"),
        ] {
            let path = trellis(root.path(), 2, 128, 0);
            rewrite_header(&path, edit);
            assert!(DenseRecord::inventory(root.path()).is_err());
        }
        let path = trellis(root.path(), 2, 128, 0);
        std::fs::copy(&path, root.path().join("dense/duplicate.safetensors")).unwrap();
        assert!(DenseRecord::inventory(root.path()).is_err());
    }

    #[test]
    fn row_major_formats_verify_partial_last_block_and_tp_shapes() {
        for code in 3..=7 {
            let root = tempfile::tempdir().unwrap();
            let n = if code == 3 { 256 } else { 288 };
            let k = 128;
            let parts = match code {
                3 => vec![
                    ("fp8", "U8", vec![n, k], vec![1; n * k]),
                    ("e8m0", "U8", vec![n, 4], vec![127; n * 4]),
                ],
                4 => vec![
                    ("fp8", "U8", vec![n, k], vec![1; n * k]),
                    (
                        "scale",
                        "F32",
                        vec![n.div_ceil(128), 1],
                        vec![1; n.div_ceil(128) * 4],
                    ),
                ],
                5 => vec![
                    ("fp8", "U8", vec![n, k], vec![1; n * k]),
                    ("scale", "F32", vec![n], vec![1; n * 4]),
                ],
                6 => vec![("bf16", "BF16", vec![n, k], vec![1; n * k * 2])],
                _ => vec![("fp32", "F32", vec![n, k], vec![1; n * k * 4])],
            };
            let hashes = vec![0; n.div_ceil(128) * 8];
            let mut parts = parts;
            parts.push(("checksum", "U64", vec![n.div_ceil(128)], hashes));
            write(
                root.path(),
                [code, n as i64, k as i64, 0, 1, 32, 0, 0],
                parts,
            );
            let records = DenseRecord::inventory(root.path()).unwrap();
            let p = DensePlan::for_rank(&records, 1, 0).unwrap();
            for unit in &p.tensors[0].units {
                unit.read(&mut vec![0; unit.staging_bytes().unwrap()])
                    .unwrap();
            }
            assert_eq!(p.tensors[0].components[0].shape[0], n);
            if code == 4 {
                assert!(DensePlan::for_rank(&records, 2, 0).is_err());
            } else {
                for rank in 0..2 {
                    let p = DensePlan::for_rank(&records, 2, rank).unwrap();
                    assert_eq!(p.tensors[0].components[0].shape, [n / 2, 128]);
                    if code == 3 { assert_eq!(p.tensors[0].components[1].shape, [128, 4]); }
                }
            }
        }
    }

    #[test]
    fn indexer_replicates_while_rank_zero_alone_owns_vocab_and_router() {
        let root = tempfile::tempdir().unwrap();
        trellis(root.path(), 2, 128, 0);
        let mut record = DenseRecord::inventory(root.path()).unwrap().remove(0);
        record.tensor = "model.layers.0.self_attn.indexer.wq_b.weight".into();
        record.ownership = DenseOwnership::new(DenseExecution::Replicated);
        for rank in 0..2 {
            let plan = DensePlan::for_rank(&[record.clone()], 2, rank).unwrap();
            assert_eq!(plan.tensors[0].first_channel, 0);
            assert_eq!(plan.tensors[0].channels, 256);
        }
        for name in [
            "model.embed_tokens.weight",
            "lm_head.weight",
            "model.norm.weight",
            "model.layers.3.mlp.gate.weight",
            "model.layers.3.mlp.gate.e_score_correction_bias",
        ] {
            record.tensor = name.into();
            record.ownership = DenseOwnership::new(DenseExecution::RankZero);
            assert_eq!(
                DensePlan::for_rank(&[record.clone()], 2, 0)
                    .unwrap()
                    .tensors
                    .len(),
                1
            );
            assert!(
                DensePlan::for_rank(&[record.clone()], 2, 1)
                    .unwrap()
                    .tensors
                    .is_empty()
            );
        }
    }

    #[test]
    fn replicated_storage_mxfp8_slices_rows_and_k32_without_conversion() {
        let root = tempfile::tempdir().unwrap();
        write(root.path(), [3, 64, 128, 0, 0, 0, 0, 0], vec![
            ("fp8", "U8", vec![64, 128], (0..8192).map(|i| (i % 251) as u8).collect()),
            ("e8m0", "U8", vec![64, 4], (0..256).map(|i| i as u8).collect()),
            ("checksum", "U64", vec![1], vec![0; 8]),
        ]);
        let mut record = DenseRecord::inventory(root.path()).unwrap().remove(0);
        record.ownership = DenseOwnership::new(DenseExecution::Split(DenseAxis::Output));
        for rank in 0..2 {
            let plan = DensePlan::for_rank(&[record.clone()], 2, rank).unwrap();
            assert_eq!(plan.tensors[0].channels, 32); // Flash beta is N64 -> N32.
            assert_eq!(plan.tensors[0].components[0].shape, [32, 128]);
            let c = &plan.tensors[0].components[0];
            let mut bytes = vec![0; c.resident_bytes().unwrap()];
            c.read(&mut bytes).unwrap();
            assert_eq!(bytes, (rank * 4096..(rank + 1) * 4096).map(|i| (i % 251) as u8).collect::<Vec<_>>());
        }
        let slice = record.slice_axis(DenseAxis::Input, 32, 64).unwrap();
        for (component, stride, first, count) in [(&slice.components[0], 128, 32, 64),
            (&slice.components[1], 4, 1, 2)] {
            let mut bytes = vec![0; component.resident_bytes().unwrap()];
            component.read(&mut bytes).unwrap();
            let raw = &record.components[usize::from(stride == 4)];
            let mut source = vec![0; raw.source.bytes];
            raw.read(&mut source).unwrap();
            let expected: Vec<_> = source.chunks_exact(stride).flat_map(|r| r[first..first + count].iter().copied()).collect();
            assert_eq!(bytes, expected);
        }
        assert!(matches!(record.slice_axis(DenseAxis::Input, 16, 64),
            Err(DenseError::Mxfp8ScaleBoundary { first: 16, end: 80, .. })));
        assert!(matches!(record.slice_axis(DenseAxis::Input, 32, 48),
            Err(DenseError::Mxfp8ScaleBoundary { first: 32, end: 80, .. })));
    }

    #[test]
    fn contradictory_projection_storage_has_typed_refusal() {
        let root = tempfile::tempdir().unwrap();
        trellis(root.path(), 2, 128, 0);
        let mut record = DenseRecord::inventory(root.path()).unwrap().remove(0);
        record.tensor = "model.layers.0.mlp.gate_proj.weight".into();
        record.ownership = DenseOwnership::new(DenseExecution::Split(DenseAxis::Output));
        assert!(matches!(DensePlan::for_rank(&[record], 2, 0),
            Err(DenseError::StorageExecutionAxis { storage: ShardSide::Input, execution: DenseAxis::Output, .. })));
    }

    #[test]
    fn kv_b_plan_charges_only_absorbed_head_operands() {
        let root = tempfile::tempdir().unwrap();
        trellis(root.path(), 0, 0, 0);
        let mut record = DenseRecord::inventory(root.path()).unwrap().remove(0);
        let geometry = KvBGeometry { heads: 64, nope: 192, values: 256, latent: 512 };
        record.tensor = "model.layers.0.self_attn.kv_b_proj.weight".into();
        record.encoding = DenseEncoding::BlockFp8;
        record.shape = vec![64 * 448, 512];
        record.ownership = DenseOwnership::new(DenseExecution::AbsorbHeads(geometry));
        for rank in 0..2 {
            let plan = DensePlan::for_rank(&[record.clone()], 2, rank).unwrap();
            let layout = plan.tensors[0].absorbed.as_ref().unwrap();
            assert_eq!((layout.first_head, layout.heads), (rank * 32, 32));
            assert_eq!(layout.allocations, [32 * 512 * 192, 32 * 512 * 3 * 4,
                32 * 256 * 512, 32 * 256 * 8 * 4]);
            assert_eq!(plan.resident_bytes, 7_798_784);
            assert_eq!(plan.resident_bytes, layout.allocations.iter().sum::<usize>());
            assert!(plan.tensors[0].components.is_empty());
            assert_eq!(resident_native(&DenseSelection::from_hf(&crate::plan::testing::glm5_config()).unwrap(), 1, &plan)
                .unwrap().iter().map(|v| v.2 as usize).sum::<usize>(), plan.resident_bytes);
        }
    }

    #[test]
    fn family_ownership_is_independent_of_physical_side_and_whole_mla_is_external() {
        let glm = ownership::DenseFamily::from_hf(&crate::plan::testing::glm5_config()).unwrap();
        for (leaf, expected) in [
            ("q_a_proj", DenseExecution::Replicated), ("kv_a_proj_with_mqa", DenseExecution::Replicated),
            ("q_b_proj", DenseExecution::Split(DenseAxis::Output)),
            ("o_proj", DenseExecution::Split(DenseAxis::Input)),
            ("indexer.wq_b", DenseExecution::Replicated), ("indexer.wk", DenseExecution::Replicated),
        ] {
            assert_eq!(glm.ownership(&format!("model.layers.0.self_attn.{leaf}.weight")).unwrap().execution, expected);
        }
        let flash = ownership::DenseFamily::from_hf(&crate::plan::testing::glm5_flash_config(2)).unwrap();
        let root = tempfile::tempdir().unwrap();
        trellis(root.path(), 1, 128, 0);
        let mut record = DenseRecord::inventory(root.path()).unwrap().remove(0);
        record.tensor = "model.language_model.layers.1.self_attn.q_b_proj.weight".into();
        record.ownership = flash.ownership(&record.tensor).unwrap();
        assert_eq!(record.ownership.whole_mla, Some(1));
        let owners = std::collections::BTreeMap::from([(1, 1)]);
        let selection = DenseSelection::from_hf(&crate::plan::testing::glm5_flash_config(2)).unwrap();
        let empty = DensePlan::for_rank_owned(&[record.clone()], 2, 0, &owners).unwrap();
        assert!(empty.tensors.is_empty());
        assert!(resident_native(&selection, 2, &empty).unwrap().is_empty());
        let plan = DensePlan::for_rank_owned(&[record], 2, 1, &owners).unwrap();
        assert_eq!(plan.tensors[0].channels, 128);
        assert_eq!(resident_native(&selection, 2, &plan).unwrap().iter()
            .map(|v| v.2 as usize).sum::<usize>(), plan.resident_bytes);
        assert_eq!(flash.ownership("model.language_model.layers.1.self_attn.indexer.wq_b.weight")
            .unwrap().execution, DenseExecution::Replicated);
        assert_eq!(flash.ownership("model.language_model.layers.1.mlp.gate.weight")
            .unwrap().execution, DenseExecution::Replicated);
    }

    #[test]
    fn flash_kernel_required_widening_is_lossless_load_only_and_glm_is_unchanged() {
        let configs: serde_json::Value = serde_json::from_str(include_str!(
            "../../../tests/fixtures/exlr8-dense-configs.json")).unwrap();
        let flash = ownership::DenseFamily::from_hf(&configs["flash"]).unwrap();
        let glm = ownership::DenseFamily::from_hf(&configs["glm"]).unwrap();
        let cfg = crate::families::glm5_flash::GlmNextConfig::from_hf(&configs["flash"]).unwrap();
        let conv = configs["flash"]["text_config"]["linear_attn_config"]["short_conv_kernel_size"].as_u64().unwrap() as usize;
        let width = cfg.kda_heads * cfg.kda_head_dim;
        for (leaf, shape, execution) in [
            ("hc_attn_fn", vec![2 * cfg.hc_mult + cfg.hc_mult * cfg.hc_mult, cfg.hc_mult * cfg.hidden], DenseExecution::Replicated),
            ("hc_ffn_fn", vec![2 * cfg.hc_mult + cfg.hc_mult * cfg.hc_mult, cfg.hc_mult * cfg.hidden], DenseExecution::Replicated),
            ("self_attn.q_conv1d.weight", vec![width, 1, conv], DenseExecution::Split(DenseAxis::Output)),
            ("self_attn.k_conv1d.weight", vec![width, 1, conv], DenseExecution::Split(DenseAxis::Output)),
            ("self_attn.v_conv1d.weight", vec![width, 1, conv], DenseExecution::Split(DenseAxis::Output)),
        ] {
            let tensor = format!("model.language_model.layers.0.{leaf}");
            let ownership = flash.ownership(&tensor).unwrap();
            assert_eq!(ownership.transform, DenseTransform::WidenBf16);
            assert_eq!(ownership.execution, execution);
            let source_bytes = shape.iter().product::<usize>() * 2;
            let record = DenseRecord { tensor, encoding: DenseEncoding::Bf16, shape: shape.clone(), side: ShardSide::Replicated,
                ownership, channels_per_shard: 0, rotation_id: 0, rotation: None, hessian_solved: false, native_blocks: 0,
                components: vec![DenseComponent { suffix: "plain".into(), dtype: "BF16".into(), shape,
                    source: DenseRange { path: "not-read".into(), offset: 0, bytes: source_bytes }, window: None }] };
            for world in [1, 2] { for rank in 0..world {
                let plan = DensePlan::for_rank(&[record.clone()], world, rank).unwrap();
                let slice = &plan.tensors[0];
                let component = &slice.components[0];
                assert_eq!(plan.resident_bytes, (component.resident_bytes().unwrap() * 2).max(256));
                let selection = DenseSelection::from_hf(&configs["flash"]).unwrap();
                let groups = resident_native(&selection, cfg.layers, &plan).unwrap();
                assert_eq!(groups.iter().map(|v| v.2 as usize).sum::<usize>(), plan.resident_bytes);
                assert!(groups.iter().all(|(_, format, _)| format == "f32"));
                let mut invalid_plan = plan.clone();
                invalid_plan.resident_bytes += 1;
                assert!(resident_native(&selection, cfg.layers, &invalid_plan).is_err());
                invalid_plan = plan.clone();
                invalid_plan.tensors[0].tensor = format!("model.language_model.layers.{}.{}", cfg.layers, leaf);
                assert!(resident_native(&selection, cfg.layers, &invalid_plan).is_err());
                assert!(plan.read_staging_bytes > 0);
                assert!(plan.read_staging_bytes <= DENSE_STEP_BYTES);
            } }
            let mut invalid = record;
            invalid.components[0].dtype = "F32".into();
            assert!(DensePlan::for_rank(&[invalid], 1, 0).is_err());
        }
        for leaf in ["input_layernorm.weight", "self_attn.q_a_layernorm.weight", "self_attn.indexer.k_norm.weight"] {
            assert_eq!(glm.ownership(&format!("model.layers.0.{leaf}")).unwrap().transform, DenseTransform::Identity);
        }
        for leaf in ["hc_attn_scale", "hc_ffn_base", "self_attn.A_log", "self_attn.dt_bias", "self_attn.o_norm.weight"] {
            assert_eq!(flash.ownership(&format!("model.language_model.layers.0.{leaf}")).unwrap().transform, DenseTransform::Identity);
        }
        let mut component = DenseComponent { suffix: "plain".into(), dtype: "BF16".into(), shape: vec![1],
            source: DenseRange { path: "not-read".into(), offset: 0, bytes: usize::MAX - 1 }, window: None };
        assert!(DenseTransform::WidenBf16.resident_bytes(&component).is_err());
        component.source.bytes = 3;
        assert!(DenseTransform::WidenBf16.resident_bytes(&component).is_err());
    }

    #[test]
    fn external_drafter_backbone_selection_never_admits_native_mtp() {
        let root = tempfile::tempdir().unwrap();
        trellis(root.path(), 0, 0, 0);
        let template = DenseRecord::inventory(root.path()).unwrap().remove(0);
        let mxfp8 = |tensor: String, n: usize, k: usize| {
            let mut record = template.clone();
            record.tensor = tensor;
            record.encoding = DenseEncoding::Mxfp8;
            record.shape = vec![n, k];
            record.native_blocks = n.div_ceil(128);
            record.rotation = None;
            record.rotation_id = 0;
            record.components = [("fp8", vec![n, k], n * k),
                ("e8m0", vec![n, k / 32], n * k / 32)].into_iter().map(|(suffix, shape, bytes)|
                DenseComponent { suffix: suffix.into(), dtype: "U8".into(), shape,
                    source: DenseRange { path: root.path().join("must-not-read-payload"), offset: 0, bytes },
                    window: None }).collect();
            record
        };
        let mut glm = crate::plan::testing::glm5_config();
        glm["num_hidden_layers"] = json!(78);
        glm["indexer_types"] = json!(vec!["full"; 78]);
        glm["mlp_layer_types"] = json!((0..78).map(|l| if l == 0 { "dense" } else { "sparse" }).collect::<Vec<_>>());
        for (config, prefix, layers, hidden, q) in [
            (glm, "model.", 78, 6144, 2048),
            (crate::plan::testing::glm5_flash_config(45), "model.language_model.", 45, 4096, 1536),
        ] {
            let selection = DenseSelection::from_hf(&config).unwrap();
            let backbone = vec![mxfp8(format!("{prefix}layers.0.self_attn.q_a_proj.weight"), q, hidden)];
            let mut with_mtp = backbone.clone();
            for (leaf, n, k) in [("eh_proj.weight", hidden, 2 * hidden),
                ("self_attn.q_a_proj.weight", q, hidden), ("shared_head.norm.weight", hidden, 128),
                ("shared_head.head.weight", 154880, hidden)] {
                with_mtp.push(mxfp8(format!("{prefix}layers.{layers}.{leaf}"), n, k));
            }
            let names = selection.vocabulary_names();
            assert_eq!(names, [format!("{prefix}embed_tokens.weight"), "lm_head.weight".into()]);
            for (name, role) in names.iter().zip([VocabularyRole::Embedding, VocabularyRole::Head]) {
                assert_eq!(selection.role(name, layers).unwrap(), DenseRole::Vocabulary(role));
            }
            assert_eq!(selection.role(&with_mtp[1].tensor, layers).unwrap(), DenseRole::Excluded);
            // A backbone operand can have exactly the head's shape and encoding.
            let mut row_backbone = backbone[0].clone();
            row_backbone.encoding = DenseEncoding::RowFp8;
            row_backbone.shape = vec![154880, hidden];
            let mut row_head = row_backbone.clone();
            row_head.tensor = names[1].clone();
            let rows = selection.backbone(&[row_backbone.clone(), row_head], layers).unwrap();
            assert_eq!(rows.len(), 1);
            assert_eq!(rows[0].tensor, row_backbone.tensor);
            let selected = selection.backbone(&with_mtp, layers).unwrap();
            assert_eq!(selected.iter().map(|r| &r.tensor).collect::<Vec<_>>(),
                backbone.iter().map(|r| &r.tensor).collect::<Vec<_>>());
            for world in [1, 2] {
                for rank in 0..world {
                    let before = DensePlan::for_rank(&backbone, world, rank).unwrap();
                    let after = DensePlan::for_rank(&selected, world, rank).unwrap();
                    assert_eq!((before.resident_bytes, before.read_staging_bytes),
                        (after.resident_bytes, after.read_staging_bytes));
                    assert!(!after.tensors.iter().any(|slice| slice.tensor.contains(&format!("layers.{layers}."))));
                }
            }
            assert!(selection.backbone(&with_mtp, layers + 1).is_err());
            assert!(selection.backbone(&with_mtp[1..], layers).is_err());
        }
    }

    #[test]
    fn native_inventory_excludes_shipped_glm_mtp_metadata_layer() {
        let root = tempfile::tempdir().unwrap();
        let path = trellis(root.path(), 0, 0, 0);
        let records = DenseRecord::inventory(root.path()).unwrap();
        let mut bytes = std::fs::read(&path).unwrap();
        let size = u64::from_le_bytes(bytes[..8].try_into().unwrap()) as usize;
        let mut header: serde_json::Value = serde_json::from_slice(&bytes[8..8 + size]).unwrap();
        header["__metadata__"]["layer"] = json!("78");
        let header = serde_json::to_vec(&header).unwrap();
        let mut file = File::create(root.path().join("dense/exlr8-dense-078.safetensors")).unwrap();
        file.write_all(&(header.len() as u64).to_le_bytes()).unwrap();
        file.write_all(&header).unwrap();
        file.write_all(&bytes.split_off(8 + size)).unwrap();
        let after = DenseRecord::inventory(root.path()).unwrap();
        assert_eq!(after.len(), records.len());
        for rank in 0..2 {
            let before = DensePlan::for_rank(&records, 2, rank).unwrap();
            let after = DensePlan::for_rank(&after, 2, rank).unwrap();
            assert_eq!((before.resident_bytes, before.read_staging_bytes),
                (after.resident_bytes, after.read_staging_bytes));
        }
    }

    #[test]
    fn vision_shards_require_family_spec_and_mixed_shards_are_refused() {
        let root = tempfile::tempdir().unwrap();
        std::fs::create_dir(root.path().join("dense")).unwrap();
        let mut config = crate::plan::testing::glm5_flash_config(2);
        config["vision_config"] = json!({"hidden_size": 1024});
        let path = root.path().join("dense/auxiliary.safetensors");
        let write = |names: &[&str]| {
            let mut header = serde_json::Map::from_iter([("__metadata__".into(), json!({"format":"pt"}))]);
            for (i, name) in names.iter().enumerate() {
                header.insert((*name).into(), json!({"dtype":"BF16","shape":[1],"data_offsets":[i*2,i*2+2]}));
            }
            let bytes = serde_json::to_vec(&header).unwrap();
            let mut file = File::create(&path).unwrap();
            file.write_all(&(bytes.len() as u64).to_le_bytes()).unwrap();
            file.write_all(&bytes).unwrap();
            file.write_all(&vec![0; names.len()*2]).unwrap();
        };
        std::fs::write(root.path().join("config.json"), config.to_string()).unwrap();
        write(&["model.visual.patch_embed.proj.weight"]);
        assert!(inventory_candidates(root.path()).unwrap().is_empty());
        std::fs::create_dir(root.path().join("experts")).unwrap();
        let index = serde_json::to_vec(&json!({"__metadata__": {"format": "exlr8-v2", "scheme": "exlr8-e4m3-v2"}})).unwrap();
        let mut file = File::create(root.path().join("experts/exlr8-layer-000.index.safetensors")).unwrap();
        file.write_all(&(index.len() as u64).to_le_bytes()).unwrap();
        file.write_all(&index).unwrap();
        let catalog = crate::plan::exlr8::catalog_headers(root.path()).unwrap();
        assert!(catalog.contains(&path));
        let checkpoint = crate::plan::exlr8::inventory(root.path(), config.clone(), |_| true).unwrap();
        assert_eq!(checkpoint.tensors.len(), 1);
        assert_eq!(checkpoint.tensors[0].meta.name, "model.visual.patch_embed.proj.weight");
        write(&["model.visual.patch_embed.proj.weight", "model.language_model.norm.weight"]);
        assert!(matches!(DenseRecord::inventory(root.path()), Err(DenseError::NativeMetadata { path: p, .. }) if p == path));
        assert!(crate::plan::exlr8::catalog_headers(root.path()).is_err());
        write(&["model.visual.patch_embed.proj.weight"]);
        config.as_object_mut().unwrap().remove("vision_config");
        std::fs::write(root.path().join("config.json"), config.to_string()).unwrap();
        assert!(matches!(DenseRecord::inventory(root.path()), Err(DenseError::NativeMetadata { path: p, .. }) if p == path));
        assert!(crate::plan::exlr8::catalog_headers(root.path()).is_err());
    }

    #[test]
    #[ignore = "requires the EXLR8 GLM 5.3 Flash checkpoint"]
    fn real_flash_native_splits_respect_scale_blocks() {
        let snapshot = std::path::PathBuf::from(std::env::var("CUTEAFD_EXLR8_FLASH_SNAPSHOT").expect("snapshot"));
        let vision_path = snapshot.join("dense/vision-bf16.safetensors");
        let vision = Header::open(&vision_path).unwrap();
        let tensors = vision.value.as_object().unwrap().iter().filter(|(name, _)| *name != "__metadata__").collect::<Vec<_>>();
        assert_eq!(tensors.len(), 347);
        assert!(tensors.iter().all(|(_, meta)| meta["dtype"] == "BF16"));
        let candidates = inventory_candidates(&snapshot).unwrap();
        assert!(!candidates.iter().any(|(path, _)| *path == vision_path));
        let records = DenseRecord::inventory(&snapshot).unwrap();
        for world in [1, 2] {
            for rank in 0..world {
                let plan = DensePlan::for_rank(&records, world, rank).unwrap();
                let mut beta = 0;
                for slice in &plan.tensors {
                    let record = records.iter().find(|r| r.tensor == slice.tensor).unwrap();
                    if record.encoding == DenseEncoding::Mxfp8 && slice.axis == DenseAxis::Input {
                        assert_eq!((slice.first_channel % 32, slice.channels % 32), (0, 0), "{}", slice.tensor);
                    }
                    if slice.tensor.ends_with(".self_attn.b_proj.weight") {
                        assert_eq!(slice.channels, 64 / world);
                        beta += 1;
                    }
                }
                assert_eq!(beta, 34);
                eprintln!("Flash TP{world} rank{rank}: {} native slices, 34 beta slices N{}, bytes {}",
                    plan.tensors.len(), 64 / world, plan.resident_bytes);
            }
        }
    }

    /// The real checkpoint's native coordinator residency: TP2 ranks sum to the
    /// census plus the kv_b split's scales (embedding/head on rank 0 only).
    /// Set CUTEAFD_EXLR8_SNAPSHOT to the EXLR8 GLM 5.3 snapshot.
    #[test]
    #[ignore = "requires the EXLR8 GLM 5.3 checkpoint"]
    fn real_checkpoint_native_residency() {
        let snapshot = std::path::PathBuf::from(std::env::var("CUTEAFD_EXLR8_SNAPSHOT").expect("snapshot"));
        let records = DenseRecord::inventory(&snapshot).unwrap();
        let selection = DenseSelection::from_hf(&crate::plan::checkpoint::read_json(&snapshot.join("config.json")).unwrap()).unwrap();
        for world in [1, 2] {
            for rank in 0..world {
                let plan = DensePlan::for_rank(&records, world, rank).unwrap();
                let mut previous_resident = 0usize;
                for record in &records {
                    let rank0 = matches!(record.tensor.as_str(), "model.embed_tokens.weight" | "lm_head.weight" | "model.norm.weight")
                        || record.tensor.ends_with(".mlp.gate.weight") || record.tensor.ends_with(".mlp.gate.e_score_correction_bias");
                    if rank0 && rank != 0 {
                        assert!(!plan.tensors.iter().any(|s| s.tensor == record.tensor));
                        continue;
                    }
                    let slice = plan.tensors.iter().find(|s| s.tensor == record.tensor).unwrap();
                    if record.tensor.ends_with(".self_attn.kv_b_proj.weight") {
                        // Previous public resident_native already charged the absorbed operands.
                        previous_resident += [(64 / world * 512 * 192),
                            (64 / world * 512 * 3) * 4,
                            (64 / world * 256 * 512),
                            (64 / world * 256 * 8) * 4].into_iter().map(|b| b.max(256)).sum::<usize>();
                        continue;
                    }
                    let split = world > 1 && record.side != ShardSide::Replicated && !rank0
                        && !record.tensor.contains(".self_attn.indexer.");
                    let channels = record.channels().unwrap() / if split { world } else { 1 };
                    assert_eq!((slice.first_channel, slice.channels), (if split { rank * channels } else { 0 }, channels), "{}", record.tensor);
                    let previous = if record.native_blocks == 0 { record.components.clone() }
                        else { record.slice(slice.first_channel, channels).unwrap().components };
                    for (actual, expected) in slice.components.iter().zip(&previous) {
                        assert!(actual.window.is_none(), "{}", record.tensor);
                        assert_eq!((&actual.shape, actual.source.offset, actual.source.bytes),
                            (&expected.shape, expected.source.offset, expected.source.bytes), "{}", record.tensor);
                        previous_resident += expected.source.bytes.max(256);
                    }
                }
                let current = resident_native(&selection, 78, &plan).unwrap()
                    .iter().map(|entry| entry.2 as usize).sum::<usize>();
                assert_eq!(current, previous_resident, "GLM public plan bytes changed TP{world} rank{rank}");
                eprintln!("GLM TP{world} rank{rank}: ownership identical; public bytes {previous_resident} -> {current}");
            }
        }
        let total = |groups: &[(String, String, u64)]| groups.iter().map(|g| g.2).sum::<u64>();
        let one = super::resident_native(&selection, 78, &DensePlan::for_rank(&records, 1, 0).unwrap()).unwrap();
        let ranks: Vec<_> = (0..2).map(|r| super::resident_native(&selection, 78, &DensePlan::for_rank(&records, 2, r).unwrap()).unwrap()).collect();
        assert!(ranks[1].iter().all(|(g, _, _)| g != "embedding" && g != "lm_head" && g != "router"));
        let (whole, r0, r1) = (total(&one), total(&ranks[0]), total(&ranks[1]));
        eprintln!("native resident bytes: TP1 {whole}, TP2 {r0} + {r1}");
        // Replicated q_a/kv_a/indexer/norms make the pair larger than one GPU.
        assert!(r0 + r1 > whole && r0 > r1);
        // Census 9.68 GiB logical (kv_b E4M3 bytes identical once split) plus split scales.
        assert!((10_000_000_000..12_500_000_000).contains(&whole), "{whole}");
    }
}
