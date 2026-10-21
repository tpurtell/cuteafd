//! Bounded direct reads of packed EXL3 tensors, retaining complete H128 blocks.
use crate::{OfficialV41Catalog, V41Exl3Projection, V41Exl3ProjectionKind};
use anyhow::{ensure, Context, Result};
#[cfg(test)]
use std::{fs::File, os::unix::fs::FileExt};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct V41Exl3TensorSlice {
    pub rows: usize,
    pub source_row_bytes: usize,
    pub column_start_bytes: usize,
    pub selected_row_bytes: usize,
}

impl V41Exl3TensorSlice {
    pub fn bytes(self) -> usize {
        self.rows * self.selected_row_bytes
    }
    /// Verify that a TP2 pair needs exactly one complete source projection.
    /// Replicated rotations may have identical slices; split projections must
    /// cover adjacent ranges, with neither gaps nor extra source bytes.
    pub fn validate_pair(pair: [Self; 2], source_bytes: usize) -> Result<()> {
        let [a, b] = pair;
        ensure!(a.rows > 0 && a.source_row_bytes > 0 && a.rows == b.rows
            && a.source_row_bytes == b.source_row_bytes,
            "incompatible TP2 projection rows");
        ensure!(a.rows.checked_mul(a.source_row_bytes) == Some(source_bytes),
            "TP2 projection source size mismatch");
        for slice in pair {
            ensure!(slice.selected_row_bytes > 0
                && slice.column_start_bytes.checked_add(slice.selected_row_bytes)
                    .is_some_and(|end| end <= slice.source_row_bytes),
                "TP2 projection slice exceeds source row");
        }
        let full = |s: Self| s.column_start_bytes == 0
            && s.selected_row_bytes == s.source_row_bytes;
        ensure!((full(a) && full(b)) || (a.column_start_bytes == 0
            && b.column_start_bytes == a.selected_row_bytes
            && b.column_start_bytes + b.selected_row_bytes == b.source_row_bytes),
            "TP2 projection ranges overlap or leave unread gaps");
        Ok(())
    }

    pub fn scratch_bytes(self) -> usize {
        if self.rows > 1 && self.selected_row_bytes != self.source_row_bytes {
            self.source_row_bytes
        } else {
            0
        }
    }
}

/// Source windows for one expert's ascending H128 units. Gate/up trellis
/// gathers columns of each packed K row; down trellis gathers whole K rows.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct V41Exl3TensorGather {
    pub(crate) rows: usize,
    pub(crate) source_row_bytes: usize,
    pub(crate) columns: Vec<std::ops::Range<usize>>,
    selected_row_bytes: usize,
}

impl V41Exl3TensorGather {
    pub fn bytes(&self) -> usize { self.rows * self.selected_row_bytes }

    pub fn scratch_bytes(&self) -> usize {
        if self.rows > 1 && self.selected_row_bytes != self.source_row_bytes {
            self.source_row_bytes
        } else { 0 }
    }
}

fn tensor_gather(p: &V41Exl3Projection, suffix: &str,
    map: &cuteafd_core::tp_units::RankUnits, expert: usize) -> Result<V41Exl3TensorGather> {
    let down = p.kind == V41Exl3ProjectionKind::Down;
    let intermediate = if down { p.input_features } else { p.output_features };
    ensure!(map.unit_columns == 128 && map.units_per_expert.checked_mul(128) == Some(intermediate),
        "EXL3 source gather requires matching whole-H128 units");
    let units = map.experts.get(expert).context("EXL3 source gather expert out of range")?;
    ensure!(!units.is_empty() && units.iter().all(|&unit| unit < map.units_per_expert)
        && units.windows(2).all(|pair| pair[0] < pair[1]),
        "EXL3 source gather units must be nonempty, sorted and unique");
    let ranges = map.column_ranges(expert)?;
    let (rows, source_row_bytes, columns) = match suffix {
        "trellis" if !down => {
            let tile_bytes = 32 * p.bits;
            (p.input_features / 16, p.output_features / 16 * tile_bytes,
                ranges.into_iter().map(|r| r.start / 16 * tile_bytes..r.end / 16 * tile_bytes).collect())
        }
        "trellis" => {
            let row_bytes = p.output_features / 16 * 32 * p.bits;
            (1, p.trellis_bytes(), ranges.into_iter()
                .map(|r| r.start / 16 * row_bytes..r.end / 16 * row_bytes).collect())
        }
        "suh" if down => (1, p.input_features * 2,
            ranges.into_iter().map(|r| r.start * 2..r.end * 2).collect()),
        "svh" if !down => (1, p.output_features * 2,
            ranges.into_iter().map(|r| r.start * 2..r.end * 2).collect()),
        "suh" => (1, p.input_features * 2, vec![0..p.input_features * 2]),
        "svh" => (1, p.output_features * 2, vec![0..p.output_features * 2]),
        "mcg" => (1, 4, vec![0..4]),
        _ => anyhow::bail!("unsupported EXL3 tensor suffix {suffix}"),
    };
    let selected_row_bytes = columns.iter().map(|r| r.end - r.start).sum();
    Ok(V41Exl3TensorGather { rows, source_row_bytes, columns, selected_row_bytes })
}

fn read_gather(file: &impl crate::expert_reader::PositionedRead, offset: u64, gather: &V41Exl3TensorGather,
    dst: &mut [u8], scratch: &mut [u8]) -> Result<()> {
    ensure!(dst.len() >= gather.bytes() && scratch.len() >= gather.scratch_bytes(),
        "EXL3 gather staging or scratch too small");
    let source_offset = |row: usize, column: usize| {
        row.checked_mul(gather.source_row_bytes).and_then(|r| r.checked_add(column))
            .and_then(|r| u64::try_from(r).ok()).and_then(|r| offset.checked_add(r))
            .context("EXL3 source gather offset overflow")
    };
    if gather.rows == 1 {
        let mut cursor = 0;
        for range in &gather.columns {
            let bytes = range.end - range.start;
            super::v41_catalog::projection_read(file, &mut dst[cursor..cursor + bytes], source_offset(0, range.start)?)?;
            cursor += bytes;
        }
    } else if gather.selected_row_bytes == gather.source_row_bytes {
        super::v41_catalog::projection_read(file, &mut dst[..gather.bytes()], offset)?;
    } else {
        let batch = scratch.len() / gather.source_row_bytes;
        for start in (0..gather.rows).step_by(batch) {
            let rows = batch.min(gather.rows - start);
            super::v41_catalog::projection_read(file, &mut scratch[..rows * gather.source_row_bytes], source_offset(start, 0)?)?;
            for row in 0..rows {
                let mut cursor = (start + row) * gather.selected_row_bytes;
                for range in &gather.columns {
                    let bytes = range.end - range.start;
                    let source = row * gather.source_row_bytes + range.start;
                    dst[cursor..cursor + bytes].copy_from_slice(&scratch[source..source + bytes]);
                    cursor += bytes;
                }
            }
        }
    }
    Ok(())
}

fn tensor_slice(
    p: &V41Exl3Projection,
    suffix: &str,
    world: usize,
    rank: usize,
) -> Result<V41Exl3TensorSlice> {
    let range = p.intermediate_partition(world, rank)?;
    let width = range.end - range.start;
    let down = p.kind == V41Exl3ProjectionKind::Down;
    let contiguous = |full, start, bytes| V41Exl3TensorSlice {
        rows: 1,
        source_row_bytes: full,
        column_start_bytes: start,
        selected_row_bytes: bytes,
    };
    Ok(match suffix {
        "trellis" if !down => {
            let tile_bytes = 16 * p.bits * 2;
            V41Exl3TensorSlice {
                rows: p.input_features / 16,
                source_row_bytes: p.output_features / 16 * tile_bytes,
                column_start_bytes: range.start / 16 * tile_bytes,
                selected_row_bytes: width / 16 * tile_bytes,
            }
        }
        "trellis" => {
            let row_bytes = p.output_features / 16 * 16 * p.bits * 2;
            contiguous(
                p.trellis_bytes(),
                range.start / 16 * row_bytes,
                width / 16 * row_bytes,
            )
        }
        "suh" if down => contiguous(p.input_features * 2, range.start * 2, width * 2),
        "svh" if !down => contiguous(p.output_features * 2, range.start * 2, width * 2),
        "suh" => contiguous(p.input_features * 2, 0, p.input_features * 2),
        "svh" => contiguous(p.output_features * 2, 0, p.output_features * 2),
        "mcg" => contiguous(4, 0, 4),
        _ => anyhow::bail!("unsupported EXL3 tensor suffix {suffix}"),
    })
}

impl OfficialV41Catalog {
    pub fn exl3_tensor_gather(&self, name: &str,
        map: &cuteafd_core::tp_units::RankUnits, expert: usize) -> Result<V41Exl3TensorGather> {
        self.exl3_tensor_gather_metadata(&self.tensor(name)?.metadata, map, expert)
    }

    pub(crate) fn exl3_tensor_gather_metadata(&self, tensor: &crate::SafetensorsTensorMetadata,
        map: &cuteafd_core::tp_units::RankUnits, expert: usize) -> Result<V41Exl3TensorGather> {
        let manifest = self.exl3().context("checkpoint is not routed EXL3")?;
        let (prefix, suffix) = tensor.name.rsplit_once('.').context("missing EXL3 suffix")?;
        let projection = manifest.projections.get(prefix).context("unknown EXL3 projection")?;
        projection.validate_tensor(tensor)?;
        tensor_gather(projection, suffix, map, expert)
    }

    /// Preserve source trellis tiles and rotation bytes without dequantization.
    pub fn read_exl3_tensor_units_into(&self, name: &str,
        map: &cuteafd_core::tp_units::RankUnits, expert: usize,
        dst: &mut [u8], scratch: &mut [u8]) -> Result<usize> {
        let gather = self.exl3_tensor_gather(name, map, expert)?;
        let tensor = self.tensor(name)?;
        let file = self.shard_file(&tensor.shard)?;
        read_gather(&file, tensor.metadata.byte_offset, &gather, dst, scratch)?;
        Ok(gather.bytes())
    }

    pub fn exl3_tensor_slice(
        &self,
        name: &str,
        world: usize,
        rank: usize,
    ) -> Result<V41Exl3TensorSlice> {
        self.exl3_tensor_slice_metadata(&self.tensor(name)?.metadata, world, rank)
    }

    pub(crate) fn exl3_tensor_slice_metadata(&self, tensor: &crate::SafetensorsTensorMetadata,
        world: usize, rank: usize) -> Result<V41Exl3TensorSlice> {
        let manifest = self.exl3().context("checkpoint is not routed EXL3")?;
        let (prefix, suffix) = tensor.name.rsplit_once('.').context("missing EXL3 suffix")?;
        let projection = manifest
            .projections
            .get(prefix)
            .context("unknown EXL3 projection")?;
        projection.validate_tensor(tensor)?;
        tensor_slice(projection, suffix, world, rank)
    }

    pub fn exl3_tensor_bytes(&self, name: &str, world: usize, rank: usize) -> Result<u64> {
        Ok(u64::try_from(
            self.exl3_tensor_slice(name, world, rank)?.bytes(),
        )?)
    }

    /// Caller owns staging and scratch. This copies compressed tiles and stored
    /// rotations only; it never materializes a dequantized weight matrix.
    pub fn read_exl3_tensor_into(
        &self,
        name: &str,
        world: usize,
        rank: usize,
        dst: &mut [u8],
        scratch: &mut [u8],
    ) -> Result<usize> {
        let slice = self.exl3_tensor_slice(name, world, rank)?;
        ensure!(dst.len() >= slice.bytes(), "EXL3 staging buffer too small");
        ensure!(
            scratch.len() >= slice.scratch_bytes(),
            "EXL3 read scratch too small"
        );
        let tensor = self.tensor(name)?;
        let file = self.shard_file(&tensor.shard)?;
        read_slice(&file, tensor.metadata.byte_offset, slice, dst, scratch)?;
        Ok(slice.bytes())
    }
}

fn read_slice(
    file: &impl crate::expert_reader::PositionedRead,
    offset: u64,
    slice: V41Exl3TensorSlice,
    dst: &mut [u8],
    scratch: &mut [u8],
) -> Result<()> {
    if slice.rows == 1 || slice.selected_row_bytes == slice.source_row_bytes {
        super::v41_catalog::projection_read(file,
            &mut dst[..slice.bytes()],
            offset
                .checked_add(slice.column_start_bytes as u64)
                .context("EXL3 contiguous offset overflow")?,
        )?;
        return Ok(());
    }
    let batch = scratch.len() / slice.source_row_bytes;
    ensure!(batch > 0, "EXL3 scratch must hold a source row");
    for start in (0..slice.rows).step_by(batch) {
        let rows = batch.min(slice.rows - start);
        let file_offset = (start as u64)
            .checked_mul(slice.source_row_bytes as u64)
            .and_then(|n| offset.checked_add(n))
            .context("EXL3 row offset overflow")?;
        super::v41_catalog::projection_read(file, &mut scratch[..rows * slice.source_row_bytes], file_offset)?;
        for row in 0..rows {
            let src = row * slice.source_row_bytes + slice.column_start_bytes;
            let dest = (start + row) * slice.selected_row_bytes;
            dst[dest..dest + slice.selected_row_bytes]
                .copy_from_slice(&scratch[src..src + slice.selected_row_bytes]);
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore = "requires CUTEAFD_EXL3_SNAPSHOT pointing to a published local snapshot"]
    fn published_projection_reads_match_full_checkpoint_bytes() {
        let path = std::env::var_os("CUTEAFD_EXL3_SNAPSHOT").expect("CUTEAFD_EXL3_SNAPSHOT");
        let catalog = crate::read_official_v41_catalog(std::path::Path::new(&path),
        )
        .unwrap();
        for kind in [
            V41Exl3ProjectionKind::Gate,
            V41Exl3ProjectionKind::Up,
            V41Exl3ProjectionKind::Down,
        ] {
            for bits in [3, 4] {
                let p = catalog
                    .exl3()
                    .unwrap()
                    .projections
                    .values()
                    .find(|p| p.kind == kind && p.bits == bits)
                    .unwrap();
                for suffix in ["trellis", "suh", "svh", "mcg"] {
                    let name = format!("{}.{suffix}", p.name);
                    let full = catalog.exl3_tensor_slice(&name, 1, 0).unwrap();
                    let mut expected = vec![0; full.bytes()];
                    catalog
                        .read_exl3_tensor_into(
                            &name,
                            1,
                            0,
                            &mut expected,
                            &mut vec![0; full.scratch_bytes()],
                        )
                        .unwrap();
                    for world in [2, 4] {
                        let mut rebuilt = vec![0; expected.len()];
                        for rank in 0..world {
                            let s = catalog
                                .exl3_tensor_slice(&name, world, rank)
                                .unwrap();
                            let mut bytes = vec![0; s.bytes()];
                            catalog
                                .read_exl3_tensor_into(
                                    &name,
                                    world,
                                    rank,
                                    &mut bytes,
                                    &mut vec![0; s.scratch_bytes() * 7],
                                )
                                .unwrap();
                            for row in 0..s.rows {
                                let start = row * s.source_row_bytes + s.column_start_bytes;
                                assert_eq!(
                                    &expected[start..start + s.selected_row_bytes],
                                    &bytes[row * s.selected_row_bytes
                                        ..(row + 1) * s.selected_row_bytes]
                                );
                                rebuilt[start..start + s.selected_row_bytes].copy_from_slice(
                                    &bytes[row * s.selected_row_bytes
                                        ..(row + 1) * s.selected_row_bytes],
                                );
                            }
                        }
                        assert_eq!(rebuilt, expected);
                    }
                }
            }
        }
        println!("K3/K4 gate/up/down packed tensors and rotations reconstruct exactly for disjoint TP2/TP4");
    }

    #[test]
    fn gathered_h128_units_preserve_source_bytes_and_reconstruct_every_projection() -> Result<()> {
        use cuteafd_core::tp_units::RankUnits;
        let dir = tempfile::tempdir()?;
        let path = dir.path().join("gather");
        let file = File::options().create_new(true).read(true).write(true).open(&path)?;
        let reader = crate::expert_reader::ExpertReader::default();
        let cached = reader.open(&path)?;
        let offset = (1u64 << 31) + 64;
        for intermediate in [2048, 2304] {
            for world in [2, 3, 4, 6] {
                for layer in [0, 3, 77] {
                    for kind in [V41Exl3ProjectionKind::Gate, V41Exl3ProjectionKind::Up, V41Exl3ProjectionKind::Down] {
                        for bits in 2..=5 {
                            let down = kind == V41Exl3ProjectionKind::Down;
                            let projection = V41Exl3Projection { name: "test".into(), kind, bits,
                                input_features: if down { intermediate } else { 128 },
                                output_features: if down { 128 } else { intermediate } };
                            for suffix in ["trellis", "suh", "svh", "mcg"] {
                                let full = tensor_slice(&projection, suffix, 1, 0)?;
                                let source: Vec<_> = (0..full.bytes()).map(|i| ((i * 17 + i / 251) % 253) as u8).collect();
                                file.write_all_at(&source, offset)?;
                                let mut rebuilt = vec![0; source.len()];
                                for rank in 0..world {
                                    let rule = cuteafd_core::tp_units::balanced_rule(128, intermediate / 128, world);
                                    let map = RankUnits::new(rule, 4, intermediate, 128, layer, world, rank)?;
                                    let gather = tensor_gather(&projection, suffix, &map, 3)?;
                                    let mut got = vec![0xa5; gather.bytes()];
                                    read_gather(&cached, offset, &gather, &mut got,
                                        &mut vec![0; crate::EXPERT_READ_WINDOW_BYTES.max(gather.scratch_bytes())])?;
                                    let mut legacy = vec![0; gather.bytes()];
                                    read_gather(&file, offset, &gather, &mut legacy,
                                        &mut vec![0; gather.scratch_bytes() * 3])?;
                                    assert_eq!(got, legacy);
                                    let mut cursor = 0;
                                    for row in 0..gather.rows {
                                        for range in &gather.columns {
                                            let first = row * gather.source_row_bytes + range.start;
                                            let bytes = range.end - range.start;
                                            assert_eq!(&got[cursor..cursor + bytes], &source[first..first + bytes]);
                                            rebuilt[first..first + bytes].copy_from_slice(&got[cursor..cursor + bytes]);
                                            cursor += bytes;
                                        }
                                    }
                                    if intermediate / 128 % world == 0 {
                                        let slice = tensor_slice(&projection, suffix, world, rank)?;
                                        let mut legacy = vec![0; slice.bytes()];
                                        read_slice(&file, offset, slice, &mut legacy, &mut vec![0; slice.scratch_bytes()])?;
                                        assert_eq!(got, legacy);
                                    }
                                    if got.len() > 1 {
                                        assert!(read_gather(&file, offset, &gather, &mut got[..1], &mut []).is_err());
                                    }
                                }
                                assert_eq!(rebuilt, source);
                            }
                        }
                    }
                }
            }
        }
        Ok(())
    }

    #[test]
    fn fabric_windows_match_legacy_glm_k4_rows_for_all_six_ranks() -> Result<()> {
        use cuteafd_core::tp_units::RankUnits;
        let root = tempfile::tempdir()?;
        let path = root.path().join("trellis");
        let file = File::options().create_new(true).write(true).read(true).open(&path)?;
        let p = V41Exl3Projection { name: "test".into(), kind: V41Exl3ProjectionKind::Gate,
            bits: 4, input_features: 6144, output_features: 2048 };
        let payload: Vec<u8> = (0..p.trellis_bytes()).map(|i| (i * 31 + i / 251) as u8).collect();
        let offset = (1u64 << 31) + 128;
        file.write_all_at(&payload, offset)?;
        let reader = crate::expert_reader::ExpertReader::default();
        let cached = reader.open(&path)?;
        for rank in 0..6 {
            let map = RankUnits::new(cuteafd_core::tp_units::balanced_rule(128, 16, 6),
                4, 2048, 128, 3, 6, rank)?;
            let gather = tensor_gather(&p, "trellis", &map, 3)?;
            let mut old = vec![0; gather.bytes()];
            let mut new = vec![0; gather.bytes()];
            read_gather(&file, offset, &gather, &mut old, &mut vec![0; gather.scratch_bytes() * 64])?;
            let (_, bytes) = OfficialV41Catalog::count_storage_reads(|| {
                read_gather(&cached, offset, &gather, &mut new, &mut vec![0; crate::EXPERT_READ_WINDOW_BYTES])
            })?;
            assert_eq!(old, new, "rank {rank}");
            assert_eq!(bytes, payload.len(), "I/O pool must preserve caller byte accounting");
        }
        Ok(())
    }

    #[test]
    fn gathered_units_reject_wrong_rotation_axis_and_noncanonical_order() -> Result<()> {
        use cuteafd_core::tp_units::{RankUnits, UnitRule};
        let projection = V41Exl3Projection { name: "test".into(), kind: V41Exl3ProjectionKind::Gate,
            bits: 4, input_features: 128, output_features: 2048 };
        let mut map = RankUnits::new(UnitRule::Rotated, 1, 2048, 128, 3, 6, 0)?;
        assert!(tensor_gather(&projection, "trellis", &map, 0).is_ok());
        map.experts[0].reverse();
        assert!(tensor_gather(&projection, "trellis", &map, 0).is_err());
        map.experts[0] = vec![0, 0];
        assert!(tensor_gather(&projection, "suh", &map, 0).is_err());
        map.experts[0] = vec![16];
        assert!(tensor_gather(&projection, "svh", &map, 0).is_err());
        map.experts[0] = vec![0];
        map.unit_columns = 32;
        assert!(tensor_gather(&projection, "trellis", &map, 0).is_err());
        Ok(())
    }

    #[test]
    fn shared_tp2_union_covers_each_projection_once() -> Result<()> {
        for kind in [V41Exl3ProjectionKind::Gate, V41Exl3ProjectionKind::Up, V41Exl3ProjectionKind::Down] {
            for bits in 2..=5 {
                let (input, output) = if kind == V41Exl3ProjectionKind::Down { (2304, 128) } else { (128, 2304) };
                let projection = V41Exl3Projection { name: "test".into(), kind, bits,
                    input_features: input, output_features: output };
                for suffix in ["trellis", "suh", "svh", "mcg"] {
                    let full = tensor_slice(&projection, suffix, 1, 0)?;
                    let pair = [tensor_slice(&projection, suffix, 2, 0)?, tensor_slice(&projection, suffix, 2, 1)?];
                    V41Exl3TensorSlice::validate_pair(pair, full.bytes())?;
                    let source: Vec<_> = (0..full.bytes()).map(|i| (i % 251) as u8).collect();
                    let dir = tempfile::tempdir()?;
                    let file = File::options().create_new(true).read(true).write(true).open(dir.path().join("tensor"))?;
                    file.write_all_at(&source, 64)?;
                    let mut once = vec![0; source.len()];
                    file.read_exact_at(&mut once, 64)?;
                    for slice in pair {
                        let mut legacy = vec![0; slice.bytes()];
                        read_slice(&file, 64, slice, &mut legacy, &mut vec![0; slice.scratch_bytes() * 3])?;
                        let pitched: Vec<_> = (0..slice.rows).flat_map(|row| {
                            let start = row * slice.source_row_bytes + slice.column_start_bytes;
                            once[start..start + slice.selected_row_bytes].iter().copied()
                        }).collect();
                        assert_eq!(pitched, legacy);
                    }
                }
            }
        }
        Ok(())
    }

    #[test]
    fn shared_tp2_union_rejects_gaps_overlap_and_oversized_slices() {
        let slice = V41Exl3TensorSlice { rows: 4, source_row_bytes: 16,
            column_start_bytes: 0, selected_row_bytes: 8 };
        let right = V41Exl3TensorSlice { column_start_bytes: 8, ..slice };
        assert!(V41Exl3TensorSlice::validate_pair([slice, right], 64).is_ok());
        for start in [0, 7, 9, usize::MAX] {
            assert!(V41Exl3TensorSlice::validate_pair([slice,
                V41Exl3TensorSlice { column_start_bytes: start, ..slice }], 64).is_err());
        }
        assert!(V41Exl3TensorSlice::validate_pair([slice, right], 65).is_err());
    }

    #[test]
    fn packed_tp_slices_reconstruct_full_tensors_past_two_gib() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("packed");
        let file = File::options()
            .create_new(true)
            .read(true)
            .write(true)
            .open(path)
            .unwrap();
        let offset = (1u64 << 31) + 128;
        for kind in [
            V41Exl3ProjectionKind::Gate,
            V41Exl3ProjectionKind::Up,
            V41Exl3ProjectionKind::Down,
        ] {
            let (input, output) = if kind == V41Exl3ProjectionKind::Down {
                (2304, 128)
            } else {
                (128, 2304)
            };
            for bits in 2..=5 {
                let p = V41Exl3Projection {
                    name: "test".into(),
                    kind,
                    bits,
                    input_features: input,
                    output_features: output,
                };
                for suffix in ["trellis", "suh", "svh", "mcg"] {
                    let full = tensor_slice(&p, suffix, 1, 0).unwrap();
                    let payload: Vec<_> = (0..full.bytes())
                        .map(|i| ((i * 17 + i / 251) % 253) as u8)
                        .collect();
                    file.set_len(offset + payload.len() as u64).unwrap();
                    file.write_all_at(&payload, offset).unwrap();
                    for world in [1, 2, 4] {
                        let mut reconstructed = vec![0u8; payload.len()];
                        for rank in 0..world {
                            let s =
                                tensor_slice(&p, suffix, world, rank).unwrap();
                            let mut dst = vec![0xA5; s.bytes()];
                            let mut scratch = vec![0; s.scratch_bytes() * 3];
                            read_slice(&file, offset, s, &mut dst, &mut scratch).unwrap();
                            for row in 0..s.rows {
                                let start = row * s.source_row_bytes + s.column_start_bytes;
                                assert_eq!(
                                    &dst[row * s.selected_row_bytes
                                        ..(row + 1) * s.selected_row_bytes],
                                    &payload[start..start + s.selected_row_bytes]
                                );
                                reconstructed[start..start + s.selected_row_bytes].copy_from_slice(
                                    &dst[row * s.selected_row_bytes
                                        ..(row + 1) * s.selected_row_bytes],
                                );
                            }
                        }
                        assert_eq!(reconstructed, payload);
                    }
                }
            }
        }
    }
}
