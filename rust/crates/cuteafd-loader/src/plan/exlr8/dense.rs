//! Checkpoint-native coordinator layouts and shared runtime/planner admission.
use super::{Header, files, number};
use anyhow::{Context, ensure};
use serde::Serialize;
use std::collections::BTreeSet;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

type Result<T> = std::result::Result<T, DenseError>;

#[derive(Debug, thiserror::Error)]
#[error("EXLR8 dense: {0:#}")]
pub struct DenseError(#[from] anyhow::Error);

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
pub struct DenseComponent {
    pub suffix: String,
    pub dtype: String,
    pub shape: Vec<usize>,
    pub source: DenseRange,
}

#[derive(Debug, Clone, Serialize)]
pub struct DenseRecord {
    pub tensor: String,
    pub encoding: DenseEncoding,
    pub shape: Vec<usize>,
    pub side: ShardSide,
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
    /// Sum of separately 256-byte-aligned component allocations.
    pub resident_bytes: usize,
    /// One staging unit, including replicated trellis scale vectors.
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

fn allocation(bytes: usize) -> anyhow::Result<usize> {
    bytes
        .max(256)
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

/// One native dense lane: BF16 rotations/intermediates and decode-sized split-K scratch.
pub fn scratch_bytes(cfg: &crate::families::glm5::GlmDsaConfig, heads: usize, inter: usize, rows: usize) -> usize {
    let (h, q, ih) = (cfg.hidden, cfg.q_lora_rank, cfg.index_heads);
    let max_k = [h, q, heads * cfg.v_head_dim, inter].into_iter().max().unwrap_or(0);
    let max_n = [heads * (cfg.qk_nope_head_dim + cfg.qk_rope_head_dim), h, inter, ih * 128].into_iter().max().unwrap_or(0);
    let splits = max_k.div_ceil(1024);
    [rows * max_k * 2, 64, max_k / 128 * 2, rows.min(64) * splits * max_n * 4, rows * max_n * 4,
        rows * (q + cfg.kv_lora_rank + cfg.qk_rope_head_dim) * 2, rows * heads * 256 * 2, rows * inter * 2,
        rows * inter * 2, rows * inter * 2, rows * (128 + ih) * 2, rows * ih * 128 * 2, rows * heads * cfg.v_head_dim * 2]
        .into_iter().map(|b| b.max(256).next_multiple_of(256)).sum()
}

/// Opt-in MX8 records are per prefill lane, never retained by decode lanes.
pub fn a8_scratch_bytes(cfg: &crate::families::glm5::GlmDsaConfig, heads: usize, inter: usize, rows: usize) -> usize {
    if rows <= 64 { return 0; }
    let max_k = [cfg.hidden, cfg.q_lora_rank, heads * cfg.v_head_dim, inter].into_iter().max().unwrap_or(0);
    (rows.div_ceil(16) * (max_k / 32) * 528).next_multiple_of(256)
}

/// One coordinator's native dense charge: decode scratch, plus every prefill lane's scratch and,
/// with opt-in A8 prefill, its MX8 records. `cuteafd plan` and runtime admission both charge this.
pub fn lane_charge_bytes(cfg: &crate::families::glm5::GlmDsaConfig, gpus: usize, decode_rows: usize,
    prefill_rows: usize, lanes: usize, a8: bool) -> usize {
    let (heads, inter) = (cfg.heads / gpus, cfg.dense_intermediate / gpus);
    let a8 = if a8 { a8_scratch_bytes(cfg, heads, inter, prefill_rows) } else { 0 };
    scratch_bytes(cfg, heads, inter, decode_rows) + lanes * (scratch_bytes(cfg, heads, inter, prefill_rows) + a8)
}

/// Read headers and tiny record tables only. The official checkpoint is not an input.
impl DenseRecord {
    pub fn inventory(snapshot: &Path) -> Result<Vec<Self>> {
        Self::inventory_inner(snapshot).map_err(Into::into)
    }

    fn inventory_inner(snapshot: &Path) -> anyhow::Result<Vec<Self>> {
        let config = super::read_json(&snapshot.join("config.json"))?;
        let layers = number(&config, "num_hidden_layers")?;
        let mut records = Vec::new();
        let mut names = BTreeSet::new();
        for path in files(snapshot, "dense")? {
            let header = Header::open(&path)?;
            header.validate()?;
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
            let logical = |stem: &str, weight: bool| {
                let tail = if weight {
                    format!("{stem}.weight")
                } else {
                    stem.into()
                };
                layer.map_or_else(
                    || {
                        if stem == "lm_head" {
                            tail.clone()
                        } else {
                            format!("model.{tail}")
                        }
                    },
                    |l| {
                        format!(
                            "{}{l}.{tail}",
                            metadata["tensor_prefix"]
                                .as_str()
                                .unwrap_or("model.layers.")
                        )
                    },
                )
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
                let tensor = logical(stem, true);
                ensure!(
                    names.insert(tensor.clone()),
                    "duplicate native dense tensor {tensor}"
                );
                records.push(Self {
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
                let tensor = logical(name, false);
                ensure!(
                    names.insert(tensor.clone()),
                    "duplicate native dense tensor {tensor}"
                );
                records.push(Self {
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
                first % 128 == 0 && (channels % 128 == 0 || first + channels == extent),
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
        (|| -> anyhow::Result<Self> {
            ensure!(
                (1..=2).contains(&world) && rank < world,
                "dense coordinator rank/world invalid"
            );
            ensure!(!records.is_empty(), "empty dense coordinator plan");
            let mut tensors = Vec::new();
            let mut resident_bytes = 0usize;
            let mut read_staging_bytes = 0usize;
            for record in records {
                let rank0 = matches!(
                    record.tensor.as_str(),
                    "model.embed_tokens.weight" | "lm_head.weight" | "model.norm.weight"
                ) || record.tensor.ends_with(".mlp.gate.weight")
                    || record.tensor.ends_with(".mlp.gate.e_score_correction_bias");
                if rank0 && rank != 0 {
                    continue;
                }
                let extent = record.channels()?;
                // The DSA indexer consumes every index head on both coordinators.
                let split = world > 1
                    && record.side != ShardSide::Replicated
                    && !rank0
                    && !record.tensor.contains(".self_attn.indexer.");
                let count = if split {
                    ensure!(
                        extent % world == 0,
                        "dense TP dimension does not divide world"
                    );
                    extent / world
                } else {
                    extent
                };
                if split {
                    ensure!(
                        count % record.native_blocks == 0,
                        "dense TP cuts a checkpoint shard unit"
                    );
                }
                let slice = if record.native_blocks == 0 {
                    DenseSlice {
                        tensor: record.tensor.clone(),
                        first_channel: 0,
                        channels: extent,
                        side: record.side,
                        components: record.components.clone(),
                        units: vec![],
                    }
                } else {
                    record.slice(if split { rank * count } else { 0 }, count)?
                };
                for component in &slice.components {
                    resident_bytes = resident_bytes
                        .checked_add(allocation(component.source.bytes)?)
                        .context("dense residency overflow")?;
                }
                for unit in &slice.units {
                    read_staging_bytes = read_staging_bytes.max(unit.staging_bytes()?);
                }
                for component in slice.components.iter().filter(|_| slice.units.is_empty()) {
                    read_staging_bytes = read_staging_bytes.max(component.source.bytes);
                }
                tensors.push(slice);
            }
            Ok(Self {
                rank,
                world,
                tensors,
                resident_bytes,
                read_staging_bytes,
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
            let workspace = allocation(lane_workspace)?
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

/// Coordinator resident bytes of a native GLM rank, grouped as the official
/// `layout::glm5::resident_owned` groups them (group, format, bytes): every
/// component its own allocation of at least 256 bytes (the loader's sizes), `kv_b` split per head into
/// E4M3 `w_uk`/`w_uv` with one F32 scale per row and 64-wide tile, router on
/// rank 0, vocabulary (embedding + head, one row-FP8 owner) on rank 0.
pub fn resident_native(records: &[DenseRecord], world: usize, rank: usize, layers: usize, heads: usize,
    nope: usize, v: usize, lat: usize) -> Result<Vec<(String, String, u64)>> {
    (|| -> anyhow::Result<_> {
        ensure!(heads % world == 0, "GLM heads do not split over {world} coordinators");
        let plan = DensePlan::for_rank(records, world, rank)?;
        let mut groups = std::collections::BTreeMap::<(String, String), u64>::new();
        let mut put = |group: &str, format: &str, bytes: usize| -> anyhow::Result<()> {
            ensure!(bytes > 0, "empty native dense allocation in {group}");
            *groups.entry((group.into(), format.into())).or_default() += bytes.max(256) as u64;
            Ok(())
        };
        for slice in &plan.tensors {
            let Some(rest) = slice.tensor.strip_prefix("model.layers.") else {
                let group = match slice.tensor.as_str() {
                    "model.embed_tokens.weight" => "embedding", "lm_head.weight" => "lm_head", _ => "norm" };
                for c in &slice.components { put(group, &c.dtype.to_lowercase(), c.source.bytes)?; }
                continue;
            };
            let layer: usize = rest.split('.').next().context("layer index")?.parse()?;
            if layer >= layers { continue; }
            // kv_b lands as its split operands (below), not as a record.
            if slice.tensor.ends_with(".self_attn.kv_b_proj.weight") { continue; }
            let group = if slice.tensor.contains(".self_attn.indexer.") { "indexer" }
                else if slice.tensor.contains(".self_attn.") { "attention" }
                else if slice.tensor.ends_with(".mlp.gate.weight") || slice.tensor.ends_with(".mlp.gate.e_score_correction_bias") { "router" }
                else if slice.tensor.contains(".mlp.") { "dense" } else { "norm" };
            for c in &slice.components { put(group, &c.dtype.to_lowercase(), c.source.bytes)?; }
        }
        let local = heads / world;
        for _ in 0..layers {
            put("attention", "fp8", local * lat * nope)?;
            put("attention", "fp8-scale", local * lat * (nope / 64) * 4)?;
            put("attention", "fp8", local * v * lat)?;
            put("attention", "fp8-scale", local * v * (lat / 64) * 4)?;
        }
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
        std::fs::write(root.join("config.json"), r#"{"num_hidden_layers":1}"#).unwrap();
        let mut entries = serde_json::Map::new();
        entries.insert(
            "__metadata__".into(),
            json!({"format":super::super::FORMAT,
            "scheme":super::super::SCHEME,"layer":"0"}),
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
            assert_eq!(plan.read_staging_bytes, 8192 + 256 + 256);
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
        assert!(allocation(usize::MAX).is_err());
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
    fn a8_lane_records_scale_with_rows_and_vanish_at_decode_width() {
        let cfg = crate::families::glm5::GlmDsaConfig::from_hf(&crate::plan::testing::glm5_config()).unwrap();
        // TP2: widest K is o_proj's 32 heads x 256 = 8192; one MX8 record row is K/32 x 528 B per 16 rows.
        let (heads, inter) = (cfg.heads / 2, cfg.dense_intermediate / 2);
        let full = a8_scratch_bytes(&cfg, heads, inter, 4096);
        assert_eq!(full, 256 * (8192 / 32) * 528);
        assert_eq!(a8_scratch_bytes(&cfg, heads, inter, 2048) * 2, full);
        assert_eq!(a8_scratch_bytes(&cfg, heads, inter, 65), 5 * (8192 / 32) * 528);
        for rows in [0, 1, 63, 64] {
            assert_eq!(a8_scratch_bytes(&cfg, heads, inter, rows), 0, "{rows} rows");
        }
        // Full width: K 16384.
        assert_eq!(a8_scratch_bytes(&cfg, cfg.heads, cfg.dense_intermediate, 4096), 256 * (16384 / 32) * 528);
        // The shared lane charge is decode plus lanes x (prefill scratch [+ records]).
        for (gpus, lanes) in [(1, 1), (2, 3)] {
            let (heads, inter) = (cfg.heads / gpus, cfg.dense_intermediate / gpus);
            let (decode, prefill) = (scratch_bytes(&cfg, heads, inter, 64), scratch_bytes(&cfg, heads, inter, 4096));
            let a8 = a8_scratch_bytes(&cfg, heads, inter, 4096);
            assert_eq!(lane_charge_bytes(&cfg, gpus, 64, 4096, lanes, false), decode + lanes * prefill);
            assert_eq!(lane_charge_bytes(&cfg, gpus, 64, 4096, lanes, true), decode + lanes * (prefill + a8));
        }
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
            if code == 3 {
                for rank in 0..2 {
                    let p = DensePlan::for_rank(&records, 2, rank).unwrap();
                    assert_eq!(p.tensors[0].components[0].shape, [128, 128]);
                    assert_eq!(p.tensors[0].components[1].shape, [128, 4]);
                }
            } else {
                // TP cannot cut a native block, even if it divides the declared shard unit.
                assert!(DensePlan::for_rank(&records, 2, 0).is_err());
            }
        }
    }

    #[test]
    fn indexer_replicates_while_rank_zero_alone_owns_vocab_and_router() {
        let root = tempfile::tempdir().unwrap();
        trellis(root.path(), 2, 128, 0);
        let mut record = DenseRecord::inventory(root.path()).unwrap().remove(0);
        record.tensor = "model.layers.0.self_attn.indexer.wq_b.weight".into();
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

    /// The real checkpoint's native coordinator residency: TP2 ranks sum to the
    /// census plus the kv_b split's scales (embedding/head on rank 0 only).
    /// Set CUTEAFD_EXLR8_SNAPSHOT to the EXLR8 GLM 5.3 snapshot.
    #[test]
    #[ignore = "requires the EXLR8 GLM 5.3 checkpoint"]
    fn real_checkpoint_native_residency() {
        let snapshot = std::path::PathBuf::from(std::env::var("CUTEAFD_EXLR8_SNAPSHOT").expect("snapshot"));
        let records = DenseRecord::inventory(&snapshot).unwrap();
        let total = |groups: &[(String, String, u64)]| groups.iter().map(|g| g.2).sum::<u64>();
        let one = super::resident_native(&records, 1, 0, 78, 64, 192, 256, 512).unwrap();
        let ranks: Vec<_> = (0..2).map(|r| super::resident_native(&records, 2, r, 78, 64, 192, 256, 512).unwrap()).collect();
        assert!(ranks[1].iter().all(|(g, _, _)| g != "embedding" && g != "lm_head" && g != "router"));
        let (whole, r0, r1) = (total(&one), total(&ranks[0]), total(&ranks[1]));
        eprintln!("native resident bytes: TP1 {whole}, TP2 {r0} + {r1}");
        // Replicated q_a/kv_a/indexer/norms make the pair larger than one GPU.
        assert!(r0 + r1 > whole && r0 > r1);
        // Census 9.68 GiB logical (kv_b E4M3 bytes identical once split) plus split scales.
        assert!((10_000_000_000..12_500_000_000).contains(&whole), "{whole}");
    }
}
