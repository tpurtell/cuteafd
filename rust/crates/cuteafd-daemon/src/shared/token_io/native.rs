//! Dormant checkpoint-native vocabulary ownership; the official token I/O is unchanged.
use crate::shared::experts::exlr8::view;
use crate::shared::memory::{DeviceAllocation, HostAllocation};
use anyhow::{Context, Result, ensure};
use cuteafd_ffi::native_dense::{DenseEmbedding, DenseLinear};
use cuteafd_ffi::programs::Programs;
use cuteafd_ffi::{CuteafdDeviceBuffer, CuteafdHostBuffer, NativeLibrary};
use cuteafd_loader::plan::exlr8::dense::{
    DenseEncoding, DensePlan, DenseRecord, DenseSlice, ShardSide,
};

const EMBEDDING: &str = "model.embed_tokens.weight";
const HEAD: &str = "lm_head.weight";

/// Admitted rank-0 payload/scales only; no BF16 table, logits, or per-lane storage.
pub(crate) struct VocabularyPlan {
    dense: DensePlan,
    hidden: usize,
    vocabulary: usize,
}

impl VocabularyPlan {
    pub(crate) fn new(
        records: &[DenseRecord],
        world: usize,
        rank: usize,
        hidden: usize,
        vocabulary: usize,
    ) -> Result<Self> {
        ensure!(
            hidden > 0 && vocabulary > 0,
            "native vocabulary has empty dimensions"
        );
        let mut selected = Vec::new();
        for name in [EMBEDDING, HEAD] {
            let mut matches = records.iter().filter(|r| r.tensor == name);
            let record = matches
                .next()
                .with_context(|| format!("missing native vocabulary {name}"))?;
            ensure!(
                matches.next().is_none(),
                "duplicate native vocabulary {name}"
            );
            ensure!(
                record.encoding == DenseEncoding::RowFp8
                    && record.shape == [vocabulary, hidden]
                    && record.side == ShardSide::Output
                    && record.rotation.is_none()
                    && record.rotation_id == 0,
                "native vocabulary {name} requires unrotated output-major row-FP8"
            );
            let payload = vocabulary
                .checked_mul(hidden)
                .context("native vocabulary extent overflow")?;
            let scales = vocabulary
                .checked_mul(4)
                .context("native vocabulary scale extent overflow")?;
            ensure!(
                record.components.len() == 2,
                "native vocabulary component count"
            );
            for (component, suffix, dtype, shape, bytes) in [
                (
                    &record.components[0],
                    "fp8",
                    "U8",
                    vec![vocabulary, hidden],
                    payload,
                ),
                (
                    &record.components[1],
                    "scale",
                    "F32",
                    vec![vocabulary],
                    scales,
                ),
            ] {
                ensure!(
                    component.suffix == suffix
                        && component.dtype == dtype
                        && component.shape == shape
                        && component.source.bytes == bytes,
                    "native vocabulary {name} component {suffix} layout"
                );
                component
                    .source
                    .offset
                    .checked_add(bytes as u64)
                    .context("native vocabulary file extent overflow")?;
            }
            // Validate the complete native partition even on the non-owning rank.
            record.slice(0, vocabulary)?;
            selected.push(record.clone());
        }
        Ok(Self {
            dense: DensePlan::for_rank(&selected, world, rank)?,
            hidden,
            vocabulary,
        })
    }

    pub(crate) fn resident_bytes(&self) -> usize {
        self.dense.resident_bytes
    }
    pub(crate) fn staging_bytes(&self) -> usize {
        self.dense.read_staging_bytes
    }

    pub(crate) fn admit(
        &self,
        device_available: usize,
        host_available: usize,
        device_reserved: usize,
        host_reserved: usize,
    ) -> Result<()> {
        let device = self
            .resident_bytes()
            .checked_add(device_reserved)
            .context("native vocabulary device admission overflow")?;
        ensure!(
            device <= device_available,
            "native vocabulary device admission needs {device}, available {device_available}"
        );
        self.dense.admit_host(host_available, host_reserved)?;
        Ok(())
    }
}

pub(crate) struct RowFp8Weights<'a> {
    payload: DeviceAllocation<'a>,
    scales: DeviceAllocation<'a>,
}

impl RowFp8Weights<'_> {
    pub(crate) fn payload(&self) -> CuteafdDeviceBuffer {
        self.payload.buffer
    }
    pub(crate) fn scales(&self) -> CuteafdDeviceBuffer {
        self.scales.buffer
    }
}

/// Owns immutable native bytes. Callers must drain all users before releasing this owner.
/// Binding is done before allocation; rank 1 owns no vocabulary or gather/head binding.
pub(crate) struct NativeVocabulary<'a> {
    pub(crate) embedding: RowFp8Weights<'a>,
    pub(crate) head: RowFp8Weights<'a>,
    pub(crate) gather: DenseEmbedding<'a>,
    pub(crate) linear: DenseLinear<'a>,
}

impl<'a> NativeVocabulary<'a> {
    pub(crate) fn load(
        library: &'a NativeLibrary,
        programs: &'a Programs<'a>,
        plan: VocabularyPlan,
        host_available: usize,
        device_reserved: usize,
        host_reserved: usize,
    ) -> Result<Option<Self>> {
        ensure!(
            std::ptr::eq(library, programs.library()),
            "native vocabulary library mismatch"
        );
        if plan.dense.rank != 0 {
            ensure!(
                plan.dense.tensors.is_empty()
                    && plan.resident_bytes() == 0
                    && plan.staging_bytes() == 0,
                "non-owner vocabulary residency"
            );
            return Ok(None);
        }
        let gather = DenseEmbedding::bind(programs, "exlr8_dense_fp8_row_embedding")?;
        let linear = DenseLinear::bind(programs, "exlr8_dense_fp8_row_lm_head")?;
        for geometry in [gather.geometry(), linear.geometry()] {
            ensure!(
                geometry.input_features == plan.hidden as u64
                    && geometry.output_features == plan.vocabulary as u64,
                "native vocabulary checkpoint differs from compiled geometry"
            );
        }
        let host_available = cuteafd_ffi::cgroup_memory_headroom(host_available)?;
        plan.admit(
            library.cuda_memory_info()?.0,
            host_available,
            device_reserved,
            host_reserved,
        )?;
        let mut staging =
            HostAllocation::named(library, plan.staging_bytes(), "native-vocabulary-staging")?;
        let embedding = load_rows(library, &plan.dense.tensors[0], &mut staging)?;
        let head = load_rows(library, &plan.dense.tensors[1], &mut staging)?;
        Ok(Some(Self {
            embedding,
            head,
            gather,
            linear,
        }))
    }
}

impl NativeVocabulary<'_> {
    /// F32 logits `[rows, vocabulary]` of BF16 rows `x [rows, hidden]` through
    /// the checkpoint's row-FP8 head.
    ///
    /// # Safety
    /// `x` and `logits` are live device buffers of these shapes on `stream`'s
    /// device (this owner's) until the stream drains.
    pub(crate) unsafe fn head_logits(&self, x: *const std::ffi::c_void, logits: *mut f32, rows: usize,
        stream: *mut std::ffi::c_void) -> Result<()> {
        // SAFETY: the caller's contract; the head operands are this owner's.
        unsafe { self.linear.launch(x.cast_mut(), self.head.payload().ptr, self.head.scales().ptr,
            std::ptr::null_mut(), 0, logits.cast(), u32::try_from(rows)?, 0, stream) }?;
        Ok(())
    }

    /// BF16 rows `[rows, hidden]` gathered from the row-FP8 embedding for I32 `ids`.
    ///
    /// # Safety
    /// `ids` and `output` are live device buffers of these shapes on `stream`'s
    /// device until the stream drains.
    pub(crate) unsafe fn embed(&self, ids: *mut std::ffi::c_void, output: *mut std::ffi::c_void, rows: usize,
        stream: *mut std::ffi::c_void) -> Result<()> {
        // SAFETY: the caller's contract; the embedding operands are this owner's.
        unsafe { self.gather.launch(ids, self.embedding.payload().ptr, self.embedding.scales().ptr, output,
            u32::try_from(rows)?, stream) }?;
        Ok(())
    }

    /// Device bytes of the payloads and scales.
    pub(crate) fn bytes(&self) -> usize {
        [self.embedding.payload(), self.embedding.scales(), self.head.payload(), self.head.scales()]
            .iter().map(|b| b.bytes).sum()
    }
}

/// Source order is payload then scales. Read the whole block before any upload.
/// The callback is blocking, so staging is reusable and partial owners can drop safely.
fn stream_rows(
    slice: &DenseSlice,
    staging: &mut [u8],
    mut publish: impl FnMut(usize, usize, &[u8]) -> Result<()>,
) -> Result<()> {
    ensure!(
        slice.components.len() == 2 && !slice.units.is_empty(),
        "native vocabulary partition"
    );
    let mut covered = [0usize; 2];
    for (index, unit) in slice.units.iter().enumerate() {
        ensure!(
            unit.block == index && unit.source.len() == 2,
            "native vocabulary block order"
        );
        let bytes = unit.staging_bytes()?;
        ensure!(bytes <= staging.len(), "native vocabulary staging extent");
        unit.read(&mut staging[..bytes])?;
        let mut at = 0;
        for (i, source) in unit.source.iter().enumerate() {
            let component = &slice.components[i].source;
            let offset = usize::try_from(
                source
                    .offset
                    .checked_sub(component.offset)
                    .context("native vocabulary component offset")?,
            )?;
            let end = offset
                .checked_add(source.bytes)
                .context("native vocabulary upload overflow")?;
            ensure!(
                source.path == component.path && offset == covered[i] && end <= component.bytes,
                "native vocabulary component coverage"
            );
            publish(i, offset, &staging[at..at + source.bytes])?;
            covered[i] = end;
            at += source.bytes;
        }
    }
    ensure!(
        covered
            == [
                slice.components[0].source.bytes,
                slice.components[1].source.bytes
            ],
        "native vocabulary incomplete component coverage"
    );
    Ok(())
}

fn load_rows<'a>(
    library: &'a NativeLibrary,
    slice: &DenseSlice,
    staging: &mut HostAllocation<'a>,
) -> Result<RowFp8Weights<'a>> {
    let weights = RowFp8Weights {
        payload: DeviceAllocation::named(
            library,
            slice.components[0].source.bytes,
            "native-vocabulary-payload",
        )?,
        scales: DeviceAllocation::named(
            library,
            slice.components[1].source.bytes,
            "native-vocabulary-scales",
        )?,
    };
    let host = staging.buffer;
    stream_rows(slice, staging.bytes_mut(), |i, offset, bytes| {
        let device = if i == 0 {
            weights.payload()
        } else {
            weights.scales()
        };
        library.copy_host_buffer_h2d(
            view(device, offset, bytes.len())?,
            CuteafdHostBuffer {
                ptr: bytes.as_ptr().cast_mut().cast(),
                bytes: bytes.len(),
                ..host
            },
            bytes.len(),
        )
    })?;
    Ok(weights)
}

#[cfg(test)]
mod tests {
    use super::*;
    use cuteafd_loader::plan::exlr8::dense::{DenseComponent, DenseRange};

    fn record(root: &std::path::Path, name: &str) -> DenseRecord {
        let (n, k) = (256, 8);
        let payload: Vec<_> = (0..n * k).map(|i| (i % 251) as u8).collect();
        let scales = vec![37; n * 4];
        let path = root.join(name);
        std::fs::write(&path, [&payload[..], &scales[..]].concat()).unwrap();
        DenseRecord {
            tensor: name.into(),
            encoding: DenseEncoding::RowFp8,
            shape: vec![n, k],
            side: ShardSide::Output,
            channels_per_shard: 128,
            rotation_id: 0,
            rotation: None,
            hessian_solved: false,
            native_blocks: n.div_ceil(128),
            components: vec![
                DenseComponent {
                    suffix: "fp8".into(),
                    dtype: "U8".into(),
                    shape: vec![n, k],
                    source: DenseRange {
                        path: path.clone(),
                        offset: 0,
                        bytes: payload.len(),
                    },
                },
                DenseComponent {
                    suffix: "scale".into(),
                    dtype: "F32".into(),
                    shape: vec![n],
                    source: DenseRange {
                        path,
                        offset: payload.len() as u64,
                        bytes: scales.len(),
                    },
                },
            ],
        }
    }

    #[test]
    fn native_vocabulary_rank0_admission_and_exact_streaming() -> Result<()> {
        let root = tempfile::tempdir()?;
        let records = [record(root.path(), EMBEDDING), record(root.path(), HEAD)];
        let plan = VocabularyPlan::new(&records, 2, 0, 8, 256)?;
        assert_eq!(plan.resident_bytes(), 6144);
        assert_eq!(plan.staging_bytes(), 1536);
        assert!(plan.admit(6143, 1536, 0, 0).is_err());
        assert!(plan.admit(6144, 1535, 0, 0).is_err());
        assert!(plan.admit(6144, 1536, 1, 0).is_err());
        assert!(plan.admit(6144, 1536, 0, 1).is_err());
        plan.admit(6144, 1536, 0, 0)?;
        let non_owner = VocabularyPlan::new(&records, 2, 1, 8, 256)?;
        assert_eq!(
            (non_owner.resident_bytes(), non_owner.staging_bytes()),
            (0, 0)
        );
        assert!(non_owner.dense.tensors.is_empty());
        let single = VocabularyPlan::new(&records, 1, 0, 8, 256)?;
        assert_eq!(single.resident_bytes(), plan.resident_bytes());
        for slice in &plan.dense.tensors {
            let mut output = [vec![0; 2048], vec![0; 1024]];
            stream_rows(slice, &mut vec![0; plan.staging_bytes()], |i, at, bytes| {
                output[i][at..at + bytes.len()].copy_from_slice(bytes);
                Ok(())
            })?;
            assert_eq!(
                output.concat(),
                std::fs::read(&slice.components[0].source.path)?
            );
        }
        Ok(())
    }

    #[test]
    fn native_vocabulary_rejects_bad_block_order_before_publication() -> Result<()> {
        let root = tempfile::tempdir()?;
        let record = record(root.path(), EMBEDDING);
        let mut slice = record.slice(0, 256)?;
        slice.units[0].block = 1;
        let mut published = 0;
        assert!(
            stream_rows(&slice, &mut vec![0; 1536], |_, _, _| {
                published += 1;
                Ok(())
            })
            .is_err()
        );
        assert_eq!(published, 0);
        let slice = record.slice(0, 256)?;
        assert!(
            stream_rows(&slice, &mut vec![0; 1536], |_, _, _| {
                anyhow::bail!("injected upload failure")
            })
            .is_err()
        );
        Ok(())
    }

    #[test]
    #[ignore = "requires a sealed SM120 native-vocabulary test slot and NVMe TMPDIR"]
    fn native_vocabulary_gpu_embedding_and_head_equivalence() -> Result<()> {
        use std::io::Write;
        const N: usize = 154880;
        const K: usize = 6144;
        let root = tempfile::tempdir()?;
        let mut records = Vec::new();
        for name in [EMBEDDING, HEAD] {
            let path = root.path().join(name);
            let mut file = std::io::BufWriter::new(std::fs::File::create(&path)?);
            let payload = vec![0x38; 128 * K]; // E4M3 1.0, with native F32 row scale 0.5.
            for _ in 0..N / 128 {
                file.write_all(&payload)?;
            }
            let scales = [0.5f32.to_le_bytes(); 128].concat();
            for _ in 0..N / 128 {
                file.write_all(&scales)?;
            }
            file.flush()?;
            let mut item = record(root.path(), "small-fixture");
            item.tensor = name.into();
            item.shape = vec![N, K];
            item.native_blocks = N / 128;
            item.components[0].shape = vec![N, K];
            item.components[0].source = DenseRange {
                path: path.clone(),
                offset: 0,
                bytes: N * K,
            };
            item.components[1].shape = vec![N];
            item.components[1].source = DenseRange {
                path,
                offset: (N * K) as u64,
                bytes: N * 4,
            };
            records.push(item);
        }
        // SAFETY: the sealed runner supplies this engine's pinned native library.
        let library = unsafe {
            NativeLibrary::load(std::env::var_os("CUTEAFD_NATIVE_LIB").context("native library")?)?
        };
        library.cuda_set_device(0)?;
        let programs = library.programs()?.with_manifest(std::path::Path::new(
            &std::env::var_os("CUTEAFD_PROGRAM_MANIFEST").context("native manifest")?,
        ))?;
        let plan = VocabularyPlan::new(&records, 2, 0, K, N)?;
        let owner = NativeVocabulary::load(&library, &programs, plan, 32 << 20, 0, 0)?
            .context("rank0 vocabulary owner")?;
        assert!(
            NativeVocabulary::load(
                &library,
                &programs,
                VocabularyPlan::new(&records, 2, 1, K, N)?,
                0,
                0,
                0
            )?
            .is_none()
        );
        let ids = DeviceAllocation::new(&library, 5 * 4)?;
        library.copy_h2d(
            ids.buffer,
            &[0i32, (N - 1) as i32, 17, -1, N as i32]
                .into_iter()
                .flat_map(i32::to_le_bytes)
                .collect::<Vec<_>>(),
        )?;
        let output = DeviceAllocation::new(&library, 5 * K * 2)?;
        let x = DeviceAllocation::new(&library, K * 2)?;
        library.copy_h2d(x.buffer, &0x3f80u16.to_le_bytes().repeat(K))?;
        let logits = DeviceAllocation::new(&library, N * 4)?;
        let mut embedding = vec![0; 5 * K * 2];
        let mut scores = vec![0; N * 4];
        // SAFETY: all operands match the validated ABI and survive the explicit device drain.
        let launched = unsafe {
            owner
                .gather
                .launch(
                    ids.buffer.ptr,
                    owner.embedding.payload().ptr,
                    owner.embedding.scales().ptr,
                    output.buffer.ptr,
                    5,
                    std::ptr::null_mut(),
                )
                .and_then(|_| {
                    owner.linear.launch(
                        x.buffer.ptr,
                        owner.head.payload().ptr,
                        owner.head.scales().ptr,
                        std::ptr::null_mut(),
                        0,
                        logits.buffer.ptr,
                        1,
                        0,
                        std::ptr::null_mut(),
                    )
                })
        };
        let drained = library.cuda_device_synchronize();
        launched?;
        drained?;
        library.copy_d2h(&mut embedding, output.buffer)?;
        library.copy_d2h(&mut scores, logits.buffer)?;
        for (row, bytes) in embedding.chunks_exact(K * 2).enumerate() {
            let expected = if row < 3 { 0x3f00u16 } else { 0 };
            assert!(
                bytes
                    .chunks_exact(2)
                    .all(|b| u16::from_le_bytes(b.try_into().unwrap()) == expected)
            );
        }
        assert!(
            scores
                .chunks_exact(4)
                .all(|b| f32::from_le_bytes(b.try_into().unwrap()) == 3072.0)
        );
        Ok(())
    }

    #[test]
    fn native_vocabulary_rejects_wrong_types_geometry_and_missing_records() -> Result<()> {
        let root = tempfile::tempdir()?;
        let records = [record(root.path(), EMBEDDING), record(root.path(), HEAD)];
        for change in 0..7 {
            let mut bad = records.clone();
            match change {
                0 => bad[0].encoding = DenseEncoding::Bf16,
                1 => bad[0].shape[1] += 1,
                2 => bad[0].side = ShardSide::Input,
                3 => bad[0].components[1].dtype = "F16".into(),
                4 => bad[0].components[0].source.bytes -= 1,
                5 => bad[0].native_blocks -= 1,
                _ => bad[0].components.swap(0, 1),
            }
            assert!(
                VocabularyPlan::new(&bad, 2, 0, 8, 256).is_err(),
                "case {change}"
            );
        }
        assert!(VocabularyPlan::new(&records[..1], 2, 0, 8, 256).is_err());
        assert!(
            VocabularyPlan::new(
                &[records[0].clone(), records[0].clone(), records[1].clone()],
                2,
                0,
                8,
                256
            )
            .is_err()
        );
        assert!(VocabularyPlan::new(&records, 2, 2, 8, 256).is_err());
        assert!(VocabularyPlan::new(&records, 3, 0, 8, 256).is_err());
        Ok(())
    }
}
