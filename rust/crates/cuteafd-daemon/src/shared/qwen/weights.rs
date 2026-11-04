//! Checkpoint-native tensor reads and packing shared by hybrid Qwen families.
use crate::shared::memory::device::{Allocation, Device};
use anyhow::{ensure, Context, Result};
use cuteafd_core::DType;
use cuteafd_ffi::{CuteafdDeviceBuffer, NativeLibrary};
use cuteafd_loader::plan::checkpoint::{Checkpoint, CheckpointTensor};
use std::os::unix::fs::FileExt;

pub(crate) struct WeightReader<'a> {
    pub(crate) library: &'a NativeLibrary,
    pub(crate) checkpoint: &'a Checkpoint,
}

pub(crate) fn f32_bytes(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

#[cfg(test)]
fn widen(bytes: &[u8], dtype: DType) -> Result<Vec<u8>> {
    match dtype {
        DType::Bf16 => {
            ensure!(bytes.len() % 2 == 0, "incomplete BF16 tensor");
            Ok(bytes.chunks_exact(2).flat_map(|b| {
                (u32::from(u16::from_le_bytes([b[0], b[1]])) << 16).to_le_bytes()
            }).collect())
        }
        DType::F32 => {
            ensure!(bytes.len() % 4 == 0, "incomplete FP32 tensor");
            Ok(bytes.to_vec())
        }
        other => anyhow::bail!("expected BF16 or FP32, found {other:?}"),
    }
}

/// Explicit operand conversion, only for checkpoints whose computation rounds
/// FP32 parameters before use. Native BF16 sources remain byte-identical.
pub(crate) fn materialize_bf16(bytes: &[u8], dtype: DType) -> Result<Vec<u8>> {
    match dtype {
        DType::Bf16 => {
            ensure!(bytes.len() % 2 == 0, "incomplete BF16 tensor");
            Ok(bytes.to_vec())
        }
        DType::F32 => {
            ensure!(bytes.len() % 4 == 0, "incomplete FP32 tensor");
            Ok(bytes.chunks_exact(4).flat_map(|b| {
                let value = f32::from_le_bytes(b.try_into().unwrap());
                let rounded = cuteafd_core::dense_attention::OutputPrecision::Bf16.round(value);
                ((rounded.to_bits() >> 16) as u16).to_le_bytes()
            }).collect())
        }
        other => anyhow::bail!("expected BF16 or FP32 operand, found {other:?}"),
    }
}

#[cfg(test)]
fn row_layout(tensors: &[(Vec<u8>, DType, Vec<usize>)], pad_rows: usize) -> Result<(usize, usize)> {
    let first = tensors.first().context("empty Qwen row concatenation")?;
    ensure!(first.2.len() == 2 && first.2[0] > 0 && first.2[1] > 0, "expected nonempty BF16 matrix");
    let cols = first.2[1];
    let mut rows = pad_rows;
    for (bytes, dtype, shape) in tensors {
        ensure!(*dtype == DType::Bf16 && shape.len() == 2 && shape[0] > 0 && shape[1] == cols,
            "Qwen packed rows require BF16 matrices sharing columns");
        let size = shape[0].checked_mul(cols).and_then(|n| n.checked_mul(2)).context("Qwen packed tensor size overflow")?;
        ensure!(bytes.len() == size, "Qwen packed tensor byte count disagrees with shape");
        rows = rows.checked_add(shape[0]).context("Qwen packed row count overflow")?;
    }
    let bytes = rows.checked_mul(cols).and_then(|n| n.checked_mul(2)).context("Qwen packed tensor size overflow")?;
    Ok((cols, bytes))
}

fn tensor<'a>(checkpoint: &'a Checkpoint, name: &str) -> Result<&'a CheckpointTensor> {
    cuteafd_ffi::memory_ledger::tensor(name);
    let at = checkpoint.tensors.binary_search_by(|t| t.meta.name.as_str().cmp(name))
        .map_err(|_| anyhow::anyhow!("checkpoint has no tensor {name}"))?;
    Ok(&checkpoint.tensors[at])
}

fn raw(checkpoint: &Checkpoint, name: &str) -> Result<(Vec<u8>, DType, Vec<usize>)> {
    let tensor = tensor(checkpoint, name)?;
    let size = usize::try_from(tensor.meta.byte_length).context("Qwen tensor byte count exceeds usize")?;
    let mut bytes = vec![0u8; size];
    std::fs::File::open(checkpoint.snapshot.join(&tensor.shard))?
        .read_exact_at(&mut bytes, tensor.meta.byte_offset)
        .with_context(|| format!("reading {name}"))?;
    Ok((bytes, tensor.meta.dtype.clone(), tensor.meta.shape.clone()))
}

impl<'a> WeightReader<'a> {
    pub(crate) fn tensor(&self, name: &str) -> Result<&'a CheckpointTensor> {
        tensor(self.checkpoint, name)
    }

    pub(crate) fn raw(&self, name: &str) -> Result<(Vec<u8>, DType, Vec<usize>)> {
        raw(self.checkpoint, name)
    }

    pub(crate) fn upload(&self, bytes: &[u8]) -> Result<Allocation<'a>> {
        let device = Device { library: self.library, id: self.library.cuda_get_device()? };
        let allocation = Allocation::new(device, bytes.len().max(256))?;
        self.library.copy_h2d(allocation.buffer, bytes)?;
        Ok(allocation)
    }

    /// Native checkpoint bytes, concatenated without changing element precision.
    pub(crate) fn native(&self, names: &[String], widen_bf16: bool) -> Result<Allocation<'a>> {
        use crate::shared::experts::rtx::dense::{DenseLoad, Job, Operand, Rect, Step, STEP_BYTES};
        let device = Device { library: self.library, id: self.library.cuda_get_device()? };
        let mut bytes = 0usize;
        let sources = names.iter().map(|name| -> Result<_> {
            let tensor = &self.tensor(name)?.meta;
            let widen = widen_bf16 && tensor.dtype == DType::Bf16;
            ensure!(!widen_bf16 || matches!(tensor.dtype, DType::Bf16 | DType::F32),
                "{name}: expected BF16 or FP32");
            let width = match tensor.dtype {
                DType::Bf16 => 2,
                DType::F32 => 4,
                DType::F8E4M3 => 1,
                ref dtype => anyhow::bail!("{name}: unsupported native Qwen dtype {dtype:?}"),
            };
            let size = tensor.shape.iter().try_fold(width, |size: usize, &dim|
                size.checked_mul(dim).context("Qwen native tensor shape overflow"))?;
            ensure!(size > 0 && u64::try_from(size)? == tensor.byte_length,
                "{name}: native tensor byte count disagrees with shape");
            let rect = Rect::tensor(self.checkpoint, name, None, None)?;
            let offset = bytes;
            bytes = bytes.checked_add(rect.bytes().checked_mul(if widen { 2 } else { 1 })
                .context("Qwen native tensor size overflow")?).context("Qwen native operand overflow")?;
            Ok((rect, offset, widen))
        }).collect::<Result<Vec<_>>>()?;
        ensure!(bytes > 0, "empty Qwen native operand");
        let out = Allocation::new(device, bytes.max(256))?;
        let load = DenseLoad::mount(self.library);
        for (rect, offset, widen) in sources {
            if !widen {
                let size = rect.bytes();
                let buffer = CuteafdDeviceBuffer { ptr: out.buffer.ptr.cast::<u8>().wrapping_add(offset).cast(),
                    bytes: size, ..out.buffer };
                load.operand(Operand { device: device.id, buffer, lands: vec![(rect, 0)], bytes: size })?;
                continue;
            }
            ensure!(rect.width() <= STEP_BYTES, "Qwen widening row exceeds bounded staging");
            let rows = (STEP_BYTES / rect.width()).max(1);
            for first in (0..rect.rows.len()).step_by(rows) {
                let part = rect.sub(first..(first + rows).min(rect.rows.len()));
                let target_offset = offset + first * rect.width() * 2;
                let dst = CuteafdDeviceBuffer {
                    ptr: out.buffer.ptr.cast::<u8>().wrapping_add(target_offset).cast(),
                    bytes: part.bytes() * 2, ..out.buffer };
                load.step(Step { device: device.id, jobs: vec![Job::Widen { at: 0,
                    bytes: part.bytes(), dst }], rects: vec![part] }, std::ptr::null_mut())?;
            }
        }
        Ok(out)
    }

    pub(crate) fn bf16(&self, name: &str) -> Result<Allocation<'a>> {
        let _memory_format = cuteafd_ffi::memory_ledger::format("bf16");
        ensure!(self.tensor(name)?.meta.dtype == DType::Bf16, "{name}: expected BF16");
        self.native(&[name.into()], false)
    }

    /// Concatenate BF16 matrices in source order, retaining zero padded rows.
    pub(crate) fn rows(&self, names: &[String], pad_rows: usize) -> Result<Allocation<'a>> {
        let _memory_format = cuteafd_ffi::memory_ledger::format("bf16");
        use crate::shared::experts::rtx::dense::{DenseLoad, Operand, Rect};
        let mut cols = None;
        let mut bytes = 0usize;
        let lands = names.iter().map(|name| -> Result<_> {
            let tensor = &self.tensor(name)?.meta;
            ensure!(tensor.dtype == DType::Bf16 && tensor.shape.len() == 2
                && tensor.shape.iter().all(|&dim| dim > 0)
                && cols.is_none_or(|cols| cols == tensor.shape[1]),
                "Qwen packed rows require BF16 matrices sharing columns");
            cols = Some(tensor.shape[1]);
            let rect = Rect::tensor(self.checkpoint, name, None, None)?;
            let size = tensor.shape[0].checked_mul(tensor.shape[1]).and_then(|n| n.checked_mul(2))
                .context("Qwen packed tensor size overflow")?;
            ensure!(rect.bytes() == size, "Qwen packed tensor byte count disagrees with shape");
            let at = bytes;
            bytes = bytes.checked_add(rect.bytes()).context("Qwen packed rows overflow")?;
            Ok((rect, at))
        }).collect::<Result<Vec<_>>>()?;
        let padded = bytes.checked_add(pad_rows.checked_mul(cols.context("empty Qwen row concatenation")?)
            .and_then(|n| n.checked_mul(2)).context("Qwen row padding overflow")?)
            .context("Qwen padded operand overflow")?;
        let device = Device { library: self.library, id: self.library.cuda_get_device()? };
        let out = Allocation::new(device, padded)?;
        self.library.cuda_zero_bytes(out.buffer, out.buffer.bytes)?;
        DenseLoad::mount(self.library).operand(Operand { device: device.id,
            buffer: out.buffer, lands, bytes: padded })?;
        Ok(out)
    }

    /// Native FP32 values stay byte-identical; BF16 values widen exactly.
    pub(crate) fn f32(&self, name: &str) -> Result<Allocation<'a>> {
        let _memory_format = cuteafd_ffi::memory_ledger::format("f32");
        self.native(&[name.into()], true)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore = "requires an allocated CPU build slot and explicit NVMe fixture directory"]
    fn native_records_replay_exact_bytes_widening_and_padded_rows() -> Result<()> {
        use crate::shared::experts::rtx::dense::{recording, testing, RankLayout, STEP_BYTES};
        use cuteafd_ffi::native_library_lifetime_fixture::Fixture;
        use cuteafd_loader::plan::testing::{t, write_snapshot};
        let fixture = Fixture::build()?;
        let library = fixture.load()?;
        fixture.configure_memory(&library, 0, 1 << 30)?;
        let device = Device { library: &library, id: 0 };
        let root = tempfile::tempdir_in(fixture.directory())?;
        write_snapshot(root.path(), &serde_json::json!({}), &[
            t("bf16", "BF16", &[2, 3]), t("fp8", "F8_E4M3", &[2, 3]),
            t("scale_bf16", "BF16", &[1, 3]), t("scale_f32", "F32", &[1, 3])], None);
        let checkpoint = Checkpoint::open(root.path())?;
        for tensor in &checkpoint.tensors {
            let bytes: Vec<_> = (0..tensor.meta.byte_length).map(|i|
                (i as u8).wrapping_mul(29).wrapping_add(tensor.meta.name.len() as u8)).collect();
            std::fs::File::options().write(true).open(root.path().join(&tensor.shard))?
                .write_all_at(&bytes, tensor.meta.byte_offset)?;
        }
        let reader = WeightReader { library: &library, checkpoint: &checkpoint };
        let load = || Ok(vec![reader.bf16("bf16")?, reader.native(&["fp8".into()], false)?,
            reader.native(&["scale_bf16".into(), "scale_f32".into()], true)?,
            reader.native(&["scale_f32".into(), "scale_bf16".into()], true)?,
            reader.rows(&["bf16".into(), "bf16".into()], 2)?, reader.f32("scale_bf16")?]);
        let expected = device.run(load)?;
        let (actual, records) = device.run(|| recording(load))?;
        assert_eq!(records.operands.len(), 5);
        assert_eq!(records.steps.len(), 3);
        assert!(records.steps.iter().all(|step| step.bytes() <= STEP_BYTES));
        // Recording only needs headers: a missing payload must not be opened.
        let shard = checkpoint.snapshot.join(&checkpoint.tensors[0].shard);
        let saved = shard.with_extension("saved");
        std::fs::rename(&shard, &saved)?;
        let header_only = device.run(|| recording(load));
        std::fs::rename(&saved, &shard)?;
        drop(header_only?);
        let layout = RankLayout::new(records.operands, records.steps, 0)?;
        testing::replay(device, layout, root.path(), 1, 0)?;
        let read = |allocation: &Allocation<'_>| -> Result<Vec<u8>> {
            let mut bytes = vec![0; allocation.buffer.bytes];
            library.copy_d2h(&mut bytes, allocation.buffer)?;
            Ok(bytes)
        };
        for (index, (actual, expected)) in actual.iter().zip(&expected).enumerate() {
            // Native allocations have a 256-byte floor; only padded rows define
            // the trailing bytes. Compare the payload, not allocator padding.
            let bytes = [12, 6, 24, 24, 36, 12][index];
            assert_eq!(&read(actual)?[..bytes], &read(expected)?[..bytes], "operand {index}");
        }
        let raw = reader.raw("scale_bf16")?.0;
        let raw_f32 = reader.raw("scale_f32")?.0;
        let mut widened = widen(&raw, DType::Bf16)?;
        widened.extend_from_slice(&raw_f32);
        assert_eq!(&read(&actual[2])?[..24], widened);
        assert_eq!(&read(&actual[4])?[24..36], [0; 12]);
        Ok(())
    }

    #[test]
    fn checkpoint_reads_respect_offsets_and_reject_missing_or_truncated_payloads() -> Result<()> {
        use cuteafd_loader::SafetensorsTensorMetadata;
        let dir = tempfile::tempdir()?;
        std::fs::write(dir.path().join("weights.bin"), [9, 9, 0, 63, 128, 63, 7])?;
        let checkpoint = Checkpoint { snapshot: dir.path().into(), config: serde_json::json!({}),
            quantize_config: None, weight_map: Default::default(), missing_shards: vec![], shard_bytes: 7,
            tensors: vec![CheckpointTensor { shard: "weights.bin".into(), meta: SafetensorsTensorMetadata {
                name: "weight".into(), dtype: DType::Bf16, shape: vec![1, 2], byte_offset: 2, byte_length: 4,
            } }] };
        assert_eq!(raw(&checkpoint, "weight")?, (vec![0, 63, 128, 63], DType::Bf16, vec![1, 2]));
        assert!(raw(&checkpoint, "missing").is_err());
        std::fs::write(dir.path().join("weights.bin"), [9, 9, 0])?;
        assert!(raw(&checkpoint, "weight").is_err());
        Ok(())
    }

    #[test]
    fn widening_preserves_all_bf16_bits_and_native_fp32_bytes() -> Result<()> {
        let bits: Vec<u16> = (0..=u16::MAX).collect();
        let bytes: Vec<u8> = bits.iter().flat_map(|b| b.to_le_bytes()).collect();
        let widened = widen(&bytes, DType::Bf16)?;
        for (value, bytes) in bits.iter().zip(widened.chunks_exact(4)) {
            assert_eq!(u32::from_le_bytes(bytes.try_into().unwrap()), u32::from(*value) << 16);
        }
        assert_eq!(widen(&widened, DType::F32)?, widened);
        assert!(widen(&[0], DType::Bf16).is_err());
        assert!(widen(&[0; 3], DType::F32).is_err());
        Ok(())
    }

    #[test]
    fn bf16_operand_materialization_rounds_ties_to_even_and_preserves_native_bytes() -> Result<()> {
        let native: Vec<_> = (0..=u16::MAX).flat_map(u16::to_le_bytes).collect();
        assert_eq!(materialize_bf16(&native, DType::Bf16)?, native);
        // Independent expected upper words on either side of every finite BF16
        // midpoint, including signed zero, subnormals and finite overflow.
        for upper in 0..=u16::MAX {
            if upper & 0x7f80 == 0x7f80 { continue; }
            for lower in [0, 0x7fff, 0x8000, 0x8001, 0xffff] {
                let bits = (u32::from(upper) << 16) | lower;
                let carry = lower > 0x8000 || (lower == 0x8000 && upper & 1 != 0);
                let expected = upper.wrapping_add(u16::from(carry));
                assert_eq!(materialize_bf16(&bits.to_le_bytes(), DType::F32)?, expected.to_le_bytes());
            }
        }
        for (bits, expected) in [(0x7f80_0000u32, 0x7f80u16), (0xff80_0000, 0xff80),
            (0x7f80_0001, 0x7fc0), (0xff80_0001, 0xffc0)] {
            assert_eq!(materialize_bf16(&bits.to_le_bytes(), DType::F32)?, expected.to_le_bytes());
        }
        assert!(materialize_bf16(&[0], DType::Bf16).is_err());
        assert!(materialize_bf16(&[0;3], DType::F32).is_err());
        assert!(materialize_bf16(&[0;4], DType::F8E4M3).is_err());
        Ok(())
    }

    #[test]
    fn packed_rows_validate_shape_before_index_or_allocation() -> Result<()> {
        let matrix = (vec![0; 12], DType::Bf16, vec![2, 3]);
        assert_eq!(row_layout(&[matrix.clone(), matrix.clone()], 1)?, (3, 30));
        assert!(row_layout(&[], 0).is_err());
        for invalid in [(vec![], DType::Bf16, vec![]), (vec![0; 12], DType::F32, vec![2, 3]),
            (vec![0; 12], DType::Bf16, vec![2, 4]), (vec![0; 11], DType::Bf16, vec![2, 3]),
            (vec![], DType::Bf16, vec![usize::MAX, 3])] {
            assert!(row_layout(&[matrix.clone(), invalid], 0).is_err());
        }
        assert!(row_layout(&[matrix], usize::MAX).is_err());
        Ok(())
    }
}
