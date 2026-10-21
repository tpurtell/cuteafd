//! Deterministic source-byte plans. Offsets come from the existing loader's
//! slicing descriptors; transforms and device-address relocation happen later.

#[cfg(test)]
mod native_tests;
#[cfg(test)]
mod tests;

use nest_push::plan::{Dim, Entry, Extent, Pattern, Plan, Steps, Totals};
use std::{
    collections::{hash_map::DefaultHasher, BTreeMap, HashMap},
    hash::{Hash, Hasher},
    ops::Range,
    path::{Component, Path, PathBuf},
};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum BuildError {
    #[error("push plan: {0}")]
    Layout(String),
    #[error(transparent)]
    Check(#[from] nest_push::plan::PlanError),
    #[error(transparent)]
    Io(#[from] std::io::Error),
    #[error("checkpoint layout: {0:#}")]
    Metadata(#[from] anyhow::Error),
}

type Result<T> = std::result::Result<T, BuildError>;

fn invalid(message: &str) -> BuildError {
    BuildError::Layout(message.into())
}
fn add(a: u64, b: u64) -> Result<u64> {
    a.checked_add(b).ok_or_else(|| invalid("offset overflow"))
}
fn mul(a: u64, b: u64) -> Result<u64> {
    a.checked_mul(b).ok_or_else(|| invalid("extent overflow"))
}

/// A tensor or a metadata-only stacked-expert view from checkpoint headers.
#[derive(Clone, Debug)]
pub struct Source {
    pub path: PathBuf,
    pub offset: u64,
    pub bytes: u64,
}

/// The arena's source-format tensor image. Padding is initialized by the owner,
/// not filled by reading adjacent source bytes. Pitched rows remain pitched.
#[derive(Clone, Copy, Debug)]
pub struct Destination {
    pub rank: u32,
    pub offset: u64,
    pub bytes: u64,
}

#[derive(Debug)]
pub struct CheckedPlan {
    pub plan: Plan,
    pub file_sizes: Vec<u64>,
    pub arenas: Vec<u64>,
    pub totals: Totals,
}

impl CheckedPlan {
    /// The shared client batches automatically, preserving pattern IDs.
    pub fn upload(&self, group: &super::PushGroup) -> std::io::Result<()> {
        group.patterns(self.plan.patterns.clone())?;
        group.entries(self.plan.entries.clone())
    }
}

struct CheckedPattern {
    pattern: Pattern,
    extent: Extent,
    fits_message: bool,
    plan_id: Option<u32>,
}

fn pattern_hash(pattern: &Pattern) -> u64 {
    let mut hash = DefaultHasher::new();
    pattern.run.hash(&mut hash);
    pattern.dims.len().hash(&mut hash);
    for dim in &pattern.dims {
        dim.src_stride.hash(&mut hash);
        dim.dst_stride.hash(&mut hash);
        match &dim.steps {
            Steps::Count(count) => {
                0u8.hash(&mut hash);
                count.hash(&mut hash);
            }
            Steps::Indices(indices) => {
                1u8.hash(&mut hash);
                indices.hash(&mut hash);
            }
        }
    }
    hash.finish()
}

pub struct Builder {
    mount: PathBuf,
    plan: Plan,
    file_ids: BTreeMap<PathBuf, u32>,
    file_sizes: Vec<u64>,
    arenas: Vec<u64>,
    checked_patterns: Vec<CheckedPattern>,
    pattern_ids: HashMap<u64, Vec<usize>>,
}

impl Builder {
    /// `mount` is the SparkNest namespace root, not the snapshot directory.
    /// Paths sent to the hub are relative to it, with no host-specific prefix.
    pub fn new(mount: &Path, arenas: Vec<u64>) -> Self {
        Self {
            mount: mount.into(),
            plan: Plan {
                files: vec![],
                patterns: vec![],
                entries: vec![],
            },
            file_ids: BTreeMap::new(),
            file_sizes: vec![],
            arenas,
            checked_patterns: vec![],
            pattern_ids: HashMap::new(),
        }
    }

    fn file(&mut self, path: &Path) -> Result<u32> {
        if let Some(&id) = self.file_ids.get(path) {
            return Ok(id);
        }
        let relative = path
            .strip_prefix(&self.mount)
            .map_err(|_| invalid("source is outside namespace mount"))?;
        if relative.as_os_str().is_empty()
            || relative
                .components()
                .any(|c| !matches!(c, Component::Normal(_)))
        {
            return Err(invalid("source namespace path is not normalized"));
        }
        let name = relative
            .to_str()
            .ok_or_else(|| invalid("namespace path is not UTF-8"))?;
        let id = u32::try_from(self.plan.files.len()).map_err(|_| invalid("too many files"))?;
        self.file_sizes.push(std::fs::metadata(path)?.len());
        self.plan.files.push(name.into());
        self.file_ids.insert(path.into(), id);
        Ok(id)
    }

    fn checked_pattern(&mut self, pattern: &Pattern) -> Result<usize> {
        self.checked_pattern_with_hash(pattern, pattern_hash(pattern))
    }

    fn checked_pattern_with_hash(&mut self, pattern: &Pattern, hash: u64) -> Result<usize> {
        if let Some(ids) = self.pattern_ids.get(&hash) {
            // Hashes only narrow the lookup; collisions never identify patterns.
            if let Some(&id) = ids.iter().find(|&&id| self.checked_patterns[id].pattern == *pattern) {
                return Ok(id);
            }
        }
        let extent = pattern.check().map_err(BuildError::Layout)?;
        let fits_message = nest_push::proto::encode(pattern).len() + 16 <= nest_push::proto::MAX_MSG;
        let id = self.checked_patterns.len();
        self.checked_patterns.push(CheckedPattern {
            pattern: pattern.clone(),
            extent,
            fits_message,
            plan_id: None,
        });
        self.pattern_ids.entry(hash).or_default().push(id);
        Ok(id)
    }

    fn pattern(&mut self, checked: usize) -> Result<u32> {
        let pattern = &mut self.checked_patterns[checked];
        if let Some(id) = pattern.plan_id {
            return Ok(id);
        }
        let id =
            u32::try_from(self.plan.patterns.len()).map_err(|_| invalid("too many patterns"))?;
        self.plan.patterns.push(pattern.pattern.clone());
        pattern.plan_id = Some(id);
        Ok(id)
    }

    fn entry(
        &mut self,
        source: &Source,
        destination: Destination,
        pattern: Pattern,
        relative: u64,
    ) -> Result<()> {
        let checked = self.checked_pattern(&pattern)?;
        let extent = self.checked_patterns[checked].extent;
        let arena = self
            .arenas
            .get(destination.rank as usize)
            .ok_or_else(|| invalid("destination rank is absent"))?;
        if add(destination.offset, destination.bytes)? > *arena {
            return Err(invalid("destination view exceeds arena"));
        }
        if add(relative, extent.src_len)? > source.bytes || extent.dst_len > destination.bytes {
            return Err(invalid("tensor image exceeds source or destination view"));
        }
        if !self.checked_patterns[checked].fits_message {
            let axis = pattern
                .dims
                .iter()
                .position(|d| matches!(d.steps, Steps::Indices(_)))
                .ok_or_else(|| invalid("oversized pattern has no index list to split"))?;
            if pattern.dims[..axis]
                .iter()
                .any(|d| !matches!(d.steps, Steps::Count(1)))
            {
                return Err(invalid(
                    "oversized interleaved index list needs multipart protocol support",
                ));
            }
            let Steps::Indices(indices) = &pattern.dims[axis].steps else {
                unreachable!()
            };
            // Five bytes per u32 is a conservative postcard bound; reserve room
            // for all dimension metadata and the request/vector envelopes.
            let chunk_len = (nest_push::proto::MAX_MSG - 256) / 5;
            for (chunk, indices) in indices.chunks(chunk_len).enumerate() {
                let mut part = pattern.clone();
                part.dims[axis].steps = Steps::Indices(indices.to_vec());
                let shift = mul(
                    mul(chunk as u64, chunk_len as u64)?,
                    pattern.dims[axis].dst_stride,
                )?;
                self.entry(
                    source,
                    Destination {
                        offset: add(destination.offset, shift)?,
                        bytes: destination
                            .bytes
                            .checked_sub(shift)
                            .ok_or_else(|| invalid("split index list exceeds destination"))?,
                        ..destination
                    },
                    part,
                    relative,
                )?;
            }
            return Ok(());
        }
        let file = self.file(&source.path)?;
        let pattern = self.pattern(checked)?;
        self.plan.entries.push(Entry {
            file,
            pattern,
            src_base: add(source.offset, relative)?,
            dst_base: destination.offset,
            rank: destination.rank,
        });
        Ok(())
    }

    /// Whole tensor, contiguous TP rows, EXLR8 unit component, or replicated
    /// rotation/scalar. Large images are represented by a count, not run lists.
    pub fn contiguous(
        &mut self,
        source: &Source,
        relative: u64,
        bytes: u64,
        destination: Destination,
    ) -> Result<()> {
        if bytes == 0 {
            return Err(invalid("empty contiguous tensor"));
        }
        let chunk = bytes.min(u32::MAX as u64);
        let full = bytes / chunk;
        let dims = if full == 1 {
            vec![]
        } else {
            vec![Dim {
                steps: Steps::Count(
                    u32::try_from(full).map_err(|_| invalid("tensor count exceeds u32"))?,
                ),
                src_stride: chunk,
                dst_stride: chunk,
            }]
        };
        let first = mul(full, chunk)?;
        self.entry(
            source,
            destination,
            Pattern {
                run: chunk as u32,
                dims,
            },
            relative,
        )?;
        if first < bytes {
            self.entry(
                source,
                Destination {
                    offset: add(destination.offset, first)?,
                    bytes: destination
                        .bytes
                        .checked_sub(first)
                        .ok_or_else(|| invalid("destination shorter than tensor"))?,
                    ..destination
                },
                Pattern {
                    run: (bytes - first) as u32,
                    dims: vec![],
                },
                add(relative, first)?,
            )?;
        }
        Ok(())
    }

    /// Source rows with one owned interval, packed or padded to the exact
    /// existing destination pitch (FP8, NVFP4, MXFP4, coordinator TP2).
    pub fn rows(
        &mut self,
        source: &Source,
        relative: u64,
        rows: u32,
        width: u32,
        source_pitch: u64,
        destination_pitch: u64,
        destination: Destination,
    ) -> Result<()> {
        if rows == 0
            || width == 0
            || source_pitch < width as u64
            || destination_pitch < width as u64
        {
            return Err(invalid("empty or overlapping tensor rows"));
        }
        if source_pitch == width as u64 && destination_pitch == width as u64 {
            return self.contiguous(
                source,
                relative,
                mul(rows as u64, width as u64)?,
                destination,
            );
        }
        self.entry(
            source,
            destination,
            Pattern {
                run: width,
                dims: vec![Dim {
                    steps: Steps::Count(rows),
                    src_stride: source_pitch,
                    dst_stride: destination_pitch,
                }],
            },
            relative,
        )
    }

    /// Source tensor sliced evenly by row (axis 0) or column (axis 1).
    pub fn matrix_partition(
        &mut self,
        source: &Source,
        rows: u32,
        axis: usize,
        world: u32,
        rank: u32,
        destination: Destination,
    ) -> Result<()> {
        if rows == 0 || world == 0 || rank >= world || source.bytes % rows as u64 != 0 {
            return Err(invalid("invalid matrix partition"));
        }
        let pitch = source.bytes / rows as u64;
        match axis {
            0 if rows % world == 0 => {
                let bytes = source.bytes / world as u64;
                self.contiguous(source, mul(rank as u64, bytes)?, bytes, destination)
            }
            1 if pitch % world as u64 == 0 => {
                let width = u32::try_from(pitch / world as u64)
                    .map_err(|_| invalid("row width exceeds u32"))?;
                self.rows(
                    source,
                    mul(rank as u64, width as u64)?,
                    rows,
                    width,
                    pitch,
                    width as u64,
                    destination,
                )
            }
            _ => Err(invalid("matrix axis does not divide evenly")),
        }
    }

    /// Gather ascending column intervals into one packed tensor. Equal-sized
    /// native units use one indexed pattern, even for balanced TP ownership.
    pub fn columns(
        &mut self,
        source: &Source,
        rows: u32,
        source_pitch: u64,
        columns: &[Range<u64>],
        destination: Destination,
    ) -> Result<()> {
        if rows == 0
            || columns.is_empty()
            || columns
                .iter()
                .any(|r| r.start >= r.end || r.end > source_pitch)
            || columns.windows(2).any(|w| w[0].end > w[1].start)
        {
            return Err(invalid("invalid tensor column intervals"));
        }
        if columns.len() == 1 {
            let range = &columns[0];
            let width = u32::try_from(range.end - range.start)
                .map_err(|_| invalid("column width exceeds u32"))?;
            return self.rows(
                source,
                range.start,
                rows,
                width,
                source_pitch,
                width as u64,
                destination,
            );
        }
        // GCD recovers the native unit when adjacent owned units were merged
        // into wider intervals by the existing layout descriptor.
        fn gcd(mut a: u64, mut b: u64) -> u64 {
            while b != 0 {
                (a, b) = (b, a % b);
            }
            a
        }
        let unit = columns.iter().fold(0, |unit, range| {
            gcd(gcd(unit, range.start), range.end - range.start)
        });
        let width = u32::try_from(unit).map_err(|_| invalid("column unit exceeds u32"))?;
        let mut indices = Vec::new();
        for range in columns {
            for index in range.start / unit..range.end / unit {
                indices
                    .push(u32::try_from(index).map_err(|_| invalid("column index exceeds u32"))?);
            }
        }
        let packed_pitch = mul(indices.len() as u64, unit)?;
        self.entry(
            source,
            destination,
            Pattern {
                run: width,
                dims: vec![
                    Dim {
                        steps: Steps::Count(rows),
                        src_stride: source_pitch,
                        dst_stride: packed_pitch,
                    },
                    Dim {
                        steps: Steps::Indices(indices),
                        src_stride: unit,
                        dst_stride: unit,
                    },
                ],
            },
            0,
        )
    }

    pub fn v41_source(catalog: &crate::OfficialV41Catalog, name: &str) -> Result<Source> {
        let tensor = catalog.tensor(name)?;
        Ok(Source {
            path: catalog.snapshot().join(&tensor.shard),
            offset: tensor.metadata.byte_offset,
            bytes: tensor.metadata.byte_length,
        })
    }

    pub fn exl3(
        &mut self,
        catalog: &crate::OfficialV41Catalog,
        name: &str,
        units: &cuteafd_core::tp_units::RankUnits,
        expert: usize,
        destination: Destination,
    ) -> Result<()> {
        let gather = catalog.exl3_tensor_gather(name, units, expert)?;
        let columns = gather
            .columns
            .iter()
            .map(|r| r.start as u64..r.end as u64)
            .collect::<Vec<_>>();
        self.columns(
            &Self::v41_source(catalog, name)?,
            u32::try_from(gather.rows).map_err(|_| invalid("too many EXL3 rows"))?,
            gather.source_row_bytes as u64,
            &columns,
            destination,
        )
    }

    /// Place source tensors into the current EXL3 layer's arena offsets.
    /// Generated descriptor words remain the arena owner's responsibility.
    pub fn exl3_residency(
        &mut self,
        catalog: &crate::OfficialV41Catalog,
        residency: &crate::V41Exl3Residency,
        offsets: &[usize],
        destination: Destination,
    ) -> Result<()> {
        if offsets.len() != residency.buffers.len() {
            return Err(invalid("EXL3 buffer offset count mismatch"));
        }
        for (buffer, &offset) in residency.buffers.iter().zip(offsets) {
            if add(offset as u64, buffer.bytes.max(16) as u64)? > destination.bytes {
                return Err(invalid("EXL3 buffer outside arena view"));
            }
        }
        for load in &residency.loads {
            for &(buffer, relative) in &load.destinations {
                let target_buffer = residency
                    .buffers
                    .get(buffer)
                    .ok_or_else(|| invalid("EXL3 destination buffer absent"))?;
                if relative
                    .checked_add(load.bytes)
                    .is_none_or(|end| end > target_buffer.bytes)
                {
                    return Err(invalid("EXL3 load exceeds destination buffer"));
                }
                let target = Destination {
                    rank: destination.rank,
                    offset: add(
                        destination.offset,
                        add(offsets[buffer] as u64, relative as u64)?,
                    )?,
                    bytes: load.bytes as u64,
                };
                if let Some(units) = &residency.unit_map {
                    self.exl3(
                        catalog,
                        &load.tensor,
                        units,
                        load.expert
                            .ok_or_else(|| invalid("EXL3 compact load lacks expert"))?,
                        target,
                    )?;
                } else {
                    let slice =
                        catalog.exl3_tensor_slice(&load.tensor, residency.world, residency.rank)?;
                    self.rows(
                        &Self::v41_source(catalog, &load.tensor)?,
                        slice.column_start_bytes as u64,
                        u32::try_from(slice.rows).map_err(|_| invalid("too many EXL3 rows"))?,
                        u32::try_from(slice.selected_row_bytes)
                            .map_err(|_| invalid("EXL3 width exceeds u32"))?,
                        slice.source_row_bytes as u64,
                        slice.selected_row_bytes as u64,
                        target,
                    )?;
                }
            }
        }
        Ok(())
    }

    /// Preserve the existing V4.1 staging region order and its alignment gaps.
    /// MXFP4 and NVFP4 share the same native row/column partition operation;
    /// NVFP4 scalar regions are replicated verbatim rather than TP sliced.
    pub fn v41_regions(
        &mut self,
        catalog: &crate::OfficialV41Catalog,
        names: &[String],
        ranges: &[Range<usize>],
        world: u32,
        rank: u32,
        destination: Destination,
    ) -> Result<()> {
        if names.len() != ranges.len() || world == 0 || rank >= world {
            return Err(invalid("invalid V4.1 staging regions"));
        }
        for (name, range) in names.iter().zip(ranges) {
            if range.start >= range.end || range.end as u64 > destination.bytes {
                return Err(invalid("V4.1 staging region outside destination"));
            }
            let source = Self::v41_source(catalog, name)?;
            let target = Destination {
                offset: add(destination.offset, range.start as u64)?,
                bytes: (range.end - range.start) as u64,
                ..destination
            };
            let metadata = &catalog.tensor(name)?.metadata;
            if source.bytes == target.bytes {
                self.contiguous(&source, 0, source.bytes, target)?;
            } else {
                let rows = *metadata
                    .shape
                    .first()
                    .ok_or_else(|| invalid("partitioned tensor has no rows"))?;
                let axis = usize::from(name.contains(".w2."));
                self.matrix_partition(
                    &source,
                    u32::try_from(rows).map_err(|_| invalid("too many matrix rows"))?,
                    axis,
                    world,
                    rank,
                    target,
                )?;
            }
        }
        Ok(())
    }

    /// Native planes occupy their final rank regions. Gate/up land as
    /// [up;gate], independently padded; no post-Done fusion copy is needed.
    /// Six NVFP4 scalars per expert go to the admitted host-only tail.
    pub fn v41_resident(
        &mut self,
        catalog: &crate::OfficialV41Catalog,
        names: &[String],
        layout: &super::native::NativePlanes,
        expert: usize,
        world: u32,
        rank: u32,
        destination: Destination,
        scalars: Option<Destination>,
    ) -> Result<()> {
        if world == 0 || rank >= world || expert >= layout.experts
            || destination.bytes < layout.bytes as u64
            || names.len() != if layout.nvfp4 { 12 } else { 6 } {
            return Err(invalid("invalid native resident image"));
        }
        for (slot, name) in names.iter().take(6).enumerate() {
            let down = name.contains(".w2.");
            let gate = name.contains(".w1.");
            if !down && !gate && !name.contains(".w3.") {
                return Err(invalid("native projection name is not W1/W3/W2"));
            }
            let scale = slot >= 3;
            let plane = usize::from(down) * 2 + usize::from(scale);
            let divisor = if scale { if layout.nvfp4 { 16 } else { 32 } } else { 2 };
            let source = Self::v41_source(catalog, name)?;
            let expected = mul(mul(layout.hidden as u64, layout.intermediate as u64)?, world as u64)? / divisor;
            if source.bytes != expected {
                return Err(invalid("native source geometry differs from resident plan"));
            }
            let (offset, bytes) = layout.expert_plane(plane, expert)?;
            let half = if gate { layout.kernel_intermediate * layout.hidden / divisor as usize } else { 0 };
            let source_pitch = if down { layout.intermediate * world as usize / divisor as usize }
                else { layout.hidden / divisor as usize };
            let width = if down { layout.intermediate / divisor as usize } else { source_pitch };
            let source_offset = if down { rank as usize * width }
                else { rank as usize * layout.intermediate * source_pitch };
            let destination_pitch = if down && layout.nvfp4 {
                layout.kernel_intermediate / divisor as usize
            } else { width };
            self.rows(&source, source_offset as u64,
                u32::try_from(if down { layout.hidden } else { layout.intermediate })
                    .map_err(|_| invalid("native rows exceed u32"))?,
                u32::try_from(width).map_err(|_| invalid("native row exceeds u32"))?,
                source_pitch as u64, destination_pitch as u64,
                Destination { offset: add(destination.offset, (offset + half) as u64)?,
                    bytes: (bytes - half) as u64, ..destination })?;
        }
        if layout.nvfp4 {
            let scalars = scalars.ok_or_else(|| invalid("NVFP4 host scalar image absent"))?;
            if scalars.rank != destination.rank || scalars.bytes < layout.scalar_bytes() as u64 {
                return Err(invalid("NVFP4 host scalar image too small"));
            }
            for (slot, name) in names.iter().skip(6).enumerate() {
                let source = Self::v41_source(catalog, name)?;
                if source.bytes != 4 || catalog.tensor(name)?.metadata.dtype != cuteafd_core::DType::F32 {
                    return Err(invalid("NVFP4 scalar must be F32"));
                }
                self.contiguous(&source, 0, 4, Destination {
                    offset: add(scalars.offset, (expert * 24 + slot * 4) as u64)?,
                    bytes: 4, ..scalars })?;
            }
        }
        Ok(())
    }

    pub fn fp8_matrix(
        &mut self,
        source: &Source,
        slice: crate::formats::fp8_experts::Fp8MatrixSlice,
        destination: Destination,
    ) -> Result<()> {
        if slice.destination_bytes as u64 > destination.bytes {
            return Err(invalid("matrix image exceeds destination"));
        }
        self.rows(
            source,
            slice.source_offset as u64,
            u32::try_from(slice.rows).map_err(|_| invalid("too many matrix rows"))?,
            u32::try_from(slice.width).map_err(|_| invalid("matrix width exceeds u32"))?,
            slice.source_pitch as u64,
            slice.destination_pitch as u64,
            destination,
        )
    }

    /// Build native FP8/FP4 projection images using the existing window oracle.
    /// BF16 scale bytes land at half-width offsets; widening runs after Done.
    pub fn fp8_projection(
        &mut self,
        tensors: &crate::formats::fp8_experts::Fp8ExpertTensors,
        layer: usize,
        expert: usize,
        projection: crate::formats::fp8_experts::Fp8Projection,
        world: usize,
        rank: usize,
        slicing: crate::formats::fp8_experts::Slicing,
        weight: Destination,
        scale: Destination,
    ) -> Result<()> {
        use crate::formats::fp8_experts::ExpertFormat;
        let tensors = tensors.for_layer_metadata(layer)?;
        let name = tensors.name(layer, expert, projection);
        let [w, mut s] = tensors.projection_slices(projection, world, rank, slicing)?;
        let (source, _) = tensors.push_source(&name)?;
        self.fp8_matrix(&source, w, weight)?;
        if tensors.format() == ExpertFormat::Bf16 {
            return Ok(());
        }
        let suffix = if tensors.format().packed_fp4() {
            "_scale"
        } else {
            "_scale_inv"
        };
        let (source, dtype) = tensors.push_source(&format!("{name}{suffix}"))?;
        if dtype == cuteafd_core::DType::Bf16 {
            for value in [
                &mut s.source_offset,
                &mut s.source_pitch,
                &mut s.destination_pitch,
                &mut s.width,
                &mut s.destination_bytes,
            ] {
                if *value % 2 != 0 {
                    return Err(invalid("unaligned BF16 scale view"));
                }
                *value /= 2;
            }
        }
        self.fp8_matrix(&source, s, scale)?;
        if tensors.format() == ExpertFormat::Nvfp4 {
            let image = tensors
                .slice_bytes_with(projection, world, rank, slicing)?
                .1 as u64;
            if add(image, 8)? > scale.bytes {
                return Err(invalid("NVFP4 scalar image outside destination"));
            }
            let input = format!(
                "{}.input_scale",
                name.strip_suffix(".weight")
                    .ok_or_else(|| invalid("weight suffix missing"))?
            );
            for (name, offset) in [(format!("{name}_scale_2"), image), (input, image + 4)] {
                let (source, dtype) = tensors.push_source(&name)?;
                if dtype != cuteafd_core::DType::F32 || source.bytes != 4 {
                    return Err(invalid("NVFP4 scalar must be F32"));
                }
                self.contiguous(
                    &source,
                    0,
                    4,
                    Destination {
                        offset: add(scale.offset, offset)?,
                        bytes: 4,
                        ..scale
                    },
                )?;
            }
        }
        Ok(())
    }

    /// Match the resident EXLR8 image, including its deduplicated block planes.
    /// Affine destination groups become one indexed pattern, not one entry per
    /// selected unit; irregular alignment boundaries start a new group.
    pub fn exlr8(
        &mut self,
        snapshot: &Path,
        residency: &crate::plan::exlr8::residency::Residency,
        destination: Destination,
    ) -> Result<()> {
        if residency.resident_bytes as u64 > destination.bytes {
            return Err(invalid("EXLR8 residency outside destination"));
        }
        let index_path = snapshot.join(format!(
            "experts/exlr8-layer-{:03}.index.safetensors",
            residency.layer
        ));
        let headers = crate::read_safetensors_metadata(&index_path)?;
        for (slot, name) in ["signs.in", "signs.out"].iter().enumerate() {
            let tensor = headers
                .iter()
                .find(|t| t.name == *name)
                .ok_or_else(|| invalid("EXLR8 signs absent"))?;
            if tensor.byte_length != residency.hidden as u64 {
                return Err(invalid("EXLR8 sign extent mismatch"));
            }
            self.contiguous(
                &Source {
                    path: index_path.clone(),
                    offset: tensor.byte_offset,
                    bytes: tensor.byte_length,
                },
                0,
                tensor.byte_length,
                Destination {
                    offset: add(destination.offset, residency.signs_offsets[slot] as u64)?,
                    bytes: tensor.byte_length,
                    ..destination
                },
            )?;
        }
        type Key = (PathBuf, usize, usize, i32, usize, usize);
        let mut groups: BTreeMap<Key, Vec<(u64, u64)>> = BTreeMap::new();
        let mut writers = BTreeMap::new();
        for unit in &residency.loads {
            for (component, source) in unit.source.iter().enumerate() {
                let offset = unit.destination[component] as u64;
                if add(offset, source.bytes as u64)? > destination.bytes {
                    return Err(invalid("EXLR8 component outside destination"));
                }
                let identity = (source.path.clone(), source.offset, source.bytes);
                if let Some(previous) = writers.insert(offset, identity.clone()) {
                    if component != 2 || previous != identity {
                        return Err(invalid("conflicting EXLR8 component writers"));
                    }
                    continue;
                }
                groups
                    .entry((
                        source.path.clone(),
                        source.bytes,
                        unit.projection,
                        unit.bits,
                        unit.expert,
                        component,
                    ))
                    .or_default()
                    .push((source.offset, offset));
            }
        }
        for ((path, bytes, _, _, _, _), mut points) in groups {
            points.sort_unstable();
            let mut start = 0;
            while start < points.len() {
                let mut end = start + 1;
                let stride = points
                    .get(end)
                    .and_then(|p| p.1.checked_sub(points[start].1));
                // Plan::check validates bounding extents, so a sparse affine
                // group cannot span another component's writer.
                let adjacent = |a: u64, b: u64| {
                    writers
                        .range((std::ops::Bound::Excluded(a), std::ops::Bound::Excluded(b)))
                        .next()
                        .is_none()
                };
                if let Some(stride) = stride
                    .filter(|s| *s >= bytes as u64 && adjacent(points[start].1, points[end].1))
                {
                    end += 1;
                    while end < points.len()
                        && points[end].1.checked_sub(points[end - 1].1) == Some(stride)
                        && adjacent(points[end - 1].1, points[end].1)
                    {
                        end += 1;
                    }
                }
                let selected = &points[start..end];
                let first = selected[0];
                let last = selected[selected.len() - 1];
                let source = Source {
                    path: path.clone(),
                    offset: first.0,
                    bytes: add(
                        last.0
                            .checked_sub(first.0)
                            .ok_or_else(|| invalid("EXLR8 sources out of order"))?,
                        bytes as u64,
                    )?,
                };
                let target = Destination {
                    offset: add(destination.offset, first.1)?,
                    bytes: add(
                        last.1
                            .checked_sub(first.1)
                            .ok_or_else(|| invalid("EXLR8 destinations out of order"))?,
                        bytes as u64,
                    )?,
                    ..destination
                };
                if selected.len() == 1 {
                    self.contiguous(&source, 0, bytes as u64, target)?;
                } else {
                    fn gcd(mut a: u64, mut b: u64) -> u64 {
                        while b != 0 {
                            (a, b) = (b, a % b);
                        }
                        a
                    }
                    let source_stride = selected.windows(2).fold(0, |s, w| gcd(s, w[1].0 - w[0].0));
                    if source_stride < bytes as u64 {
                        return Err(invalid("overlapping EXLR8 unit sources"));
                    }
                    let indices = selected
                        .iter()
                        .map(|p| {
                            u32::try_from((p.0 - first.0) / source_stride)
                                .map_err(|_| invalid("EXLR8 source index exceeds u32"))
                        })
                        .collect::<Result<Vec<_>>>()?;
                    self.entry(
                        &source,
                        target,
                        Pattern {
                            run: u32::try_from(bytes)
                                .map_err(|_| invalid("EXLR8 unit exceeds u32"))?,
                            dims: vec![Dim {
                                steps: Steps::Indices(indices),
                                src_stride: source_stride,
                                dst_stride: stride.expect("multi-point affine group"),
                            }],
                        },
                        0,
                    )?;
                }
                start = end;
            }
        }
        Ok(())
    }

    pub fn finish(self) -> Result<CheckedPlan> {
        let totals = self.plan.check(&self.file_sizes, &self.arenas)?;
        #[cfg(test)]
        {
            let largest = self
                .plan
                .patterns
                .iter()
                .filter_map(|pattern| {
                    pattern.dims.iter().find_map(|dim| match &dim.steps {
                        Steps::Indices(indices) => {
                            Some((indices.len(), nest_push::proto::encode(pattern).len()))
                        }
                        _ => None,
                    })
                })
                .max()
                .unwrap_or_default();
            eprintln!(
                "push plan: {} entries, {} patterns; max Indices {} items / {} encoded bytes",
                self.plan.entries.len(),
                self.plan.patterns.len(),
                largest.0,
                largest.1
            );
        }
        Ok(CheckedPlan {
            plan: self.plan,
            file_sizes: self.file_sizes,
            arenas: self.arenas,
            totals,
        })
    }
}
