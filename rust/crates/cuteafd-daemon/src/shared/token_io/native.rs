//! Dormant checkpoint-native vocabulary ownership; the official token I/O is unchanged.
use crate::shared::experts::exlr8::view;
use crate::shared::memory::{DeviceAllocation, HostAllocation};
use anyhow::{Context, Result, ensure};
use cuteafd_ffi::native_dense::{DenseEmbedding, DenseLinear};
use cuteafd_ffi::programs::Programs;
use cuteafd_ffi::{CuteafdDeviceBuffer, CuteafdHostBuffer, NativeLibrary};
use cuteafd_loader::plan::exlr8::dense::{
    DenseEncoding, DensePlan, DenseRecord, DenseSelection, DenseSlice, ShardSide,
};

const EMBEDDING: &str = "model.embed_tokens.weight";
const HEAD: &str = "lm_head.weight";

/// Admitted rank-0 payload/scales only; no BF16 table, logits, or per-lane storage.
pub(crate) struct VocabularyPlan {
    dense: DensePlan,
    staging_bytes: usize,
    hidden: usize,
    vocabulary: usize,
}

impl VocabularyPlan {
    pub(crate) fn new(
        records: &[DenseRecord],
        selection: &DenseSelection,
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
        for name in selection.vocabulary_names() {
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
        let dense = DensePlan::for_rank(&selected, world, rank)?;
        // Vocabulary streams native units, not Mount's component batches and row cache.
        let staging_bytes = dense.tensors.iter().flat_map(|slice| &slice.units)
            .try_fold(0usize, |peak, unit| -> Result<usize> {
                Ok(peak.max(unit.staging_bytes()?))
            })?;
        Ok(Self {
            dense,
            staging_bytes,
            hidden,
            vocabulary,
        })
    }

    pub(crate) fn resident_groups(&self, selection: &DenseSelection, layers: usize)
        -> Result<Vec<(String, String, u64)>> {
        Ok(cuteafd_loader::plan::exlr8::dense::resident_native(selection, layers, &self.dense)?)
    }

    pub(crate) fn dense_plan(&self) -> &DensePlan {
        &self.dense
    }

    pub(crate) fn resident_bytes(&self) -> usize {
        self.dense.resident_bytes
    }
    pub(crate) fn staging_bytes(&self) -> usize {
        self.staging_bytes
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
        let host = self.staging_bytes().checked_add(host_reserved)
            .context("native vocabulary host admission overflow")?;
        ensure!(host <= host_available,
            "native vocabulary host admission needs {host}, available {host_available}");
        Ok(())
    }
}

pub(crate) struct RowFp8Weights<'a> {
    payload: DeviceAllocation<'a>,
    scales: DeviceAllocation<'a>,
}

/// One named table's residency, independent of the shared vocabulary lifetime owner.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct RowFp8Residency {
    pub(crate) device: i32,
    pub(crate) bytes: usize,
}

impl RowFp8Residency {
    pub(crate) fn from_buffers(payload: CuteafdDeviceBuffer, scales: CuteafdDeviceBuffer) -> Self {
        Self { device: payload.device_id, bytes: payload.bytes + scales.bytes }
    }
}

impl RowFp8Weights<'_> {
    pub(crate) fn residency(&self) -> RowFp8Residency {
        RowFp8Residency::from_buffers(self.payload(), self.scales())
    }

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
        let resolve = |encoding| -> Result<String> {
            let mut matches = programs.names().filter(|name| name.starts_with("exlr8_dense_"))
                .filter_map(|name| cuteafd_ffi::native_dense::validate_spec(programs.spec(name).ok()?).ok()
                    .filter(|geometry| geometry.encoding == encoding
                        && geometry.input_features == plan.hidden as u64
                        && geometry.output_features == plan.vocabulary as u64)
                    .map(|_| name.to_owned()));
            let name = matches.next().context("native vocabulary has no compiled geometry")?;
            ensure!(matches.next().is_none(), "duplicate native vocabulary kernel geometry");
            Ok(name)
        };
        let gather = DenseEmbedding::bind(programs, &resolve(cuteafd_ffi::native_dense::Encoding::Embedding)?)?;
        let linear = DenseLinear::bind(programs, &resolve(cuteafd_ffi::native_dense::Encoding::RowFp8)?)?;
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

    /// Whole-owner inventory; named embedding/head charges partition this total.
    pub(crate) fn bytes(&self) -> usize {
        self.embedding.residency().bytes + self.head.residency().bytes
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

    fn selection(flash: bool) -> DenseSelection {
        let configs: serde_json::Value = serde_json::from_str(include_str!(
            "../../../../cuteafd-loader/tests/fixtures/exlr8-dense-configs.json")).unwrap();
        DenseSelection::from_hf(&configs[if flash { "flash" } else { "glm" }]).unwrap()
    }

    fn record(root: &std::path::Path, name: &str) -> DenseRecord {
        let (n, k) = (256, 8);
        let payload: Vec<_> = (0..n * k).map(|i| (i % 251) as u8).collect();
        let scales = vec![37; n * 4];
        let path = root.join(name);
        std::fs::write(&path, [&payload[..], &scales[..]].concat()).unwrap();
        DenseRecord {
            tensor: name.into(),
            ownership: cuteafd_loader::plan::exlr8::dense::DenseOwnership::new(
                cuteafd_loader::plan::exlr8::dense::DenseExecution::RankZero),
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
                    window: None,
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
                    window: None,
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
    fn native_embedding_and_head_charges_partition_planned_residency() -> Result<()> {
        use super::super::validate_embedding_residency;
        let root = tempfile::tempdir()?;
        let records = [record(root.path(), EMBEDDING), record(root.path(), HEAD)];
        for flash in [false, true] {
            let mut records = records.clone();
            for (record, name) in records.iter_mut().zip(selection(flash).vocabulary_names()) {
                record.tensor = name.into();
            }
            for world in [1, 2] {
                let plan = VocabularyPlan::new(&records, &selection(flash), world, 0, 8, 256)?;
                let named = plan.dense.tensors.iter().map(|table| {
                    let buffers = table.components.iter().map(|component| CuteafdDeviceBuffer {
                        bytes: component.source.bytes, device_id: 3, ..Default::default()
                    }).collect::<Vec<_>>();
                    RowFp8Residency::from_buffers(buffers[0], buffers[1])
                }).collect::<Vec<_>>();
                let (embedding, head) = (named[0], named[1]);
                assert_eq!(embedding, RowFp8Residency { device: 3, bytes: 3072 });
                assert_eq!(head.bytes, 3072);
                assert_eq!(embedding.bytes + head.bytes, plan.resident_bytes());
                validate_embedding_residency(embedding.device, embedding.bytes, 3, 3072)?;
                for (device, bytes) in [(0, 3072), (3, 0), (3, 256), (3, 6144), (3, 2048)] {
                    assert!(validate_embedding_residency(embedding.device, embedding.bytes, device, bytes).is_err());
                }
                // GLM previously put both tables in the head charge; splitting owners must conserve bytes.
                let body = 12345;
                let before = body + plan.resident_bytes();
                let after = body + head.bytes + embedding.bytes;
                assert_eq!(before, after);
                assert_ne!(after + embedding.bytes, before);
            }
        }
        Ok(())
    }

    #[test]
    fn native_vocabulary_rank0_admission_and_exact_streaming() -> Result<()> {
        let root = tempfile::tempdir()?;
        let records = [record(root.path(), EMBEDDING), record(root.path(), HEAD)];
        let plan = VocabularyPlan::new(&records, &selection(false), 2, 0, 8, 256)?;
        assert_eq!(plan.resident_bytes(), 6144);
        assert_eq!(plan.staging_bytes(), 1536);
        let max_unit_read = plan.dense.tensors.iter().flat_map(|slice| &slice.units)
            .map(|unit| unit.staging_bytes().unwrap()).max().unwrap();
        assert_eq!(plan.staging_bytes(), max_unit_read);
        assert!(plan.staging_bytes() < plan.dense.read_staging_bytes);
        assert!(plan.admit(6143, 1536, 0, 0).is_err());
        assert!(plan.admit(6144, 1535, 0, 0).is_err());
        assert!(plan.admit(6144, 1536, 1, 0).is_err());
        assert!(plan.admit(6144, 1536, 0, 1).is_err());
        assert!(plan.admit(6144, usize::MAX, 0, usize::MAX).is_err());
        plan.admit(6144, 1536, 0, 0)?;
        let non_owner = VocabularyPlan::new(&records, &selection(false), 2, 1, 8, 256)?;
        assert_eq!(
            (non_owner.resident_bytes(), non_owner.staging_bytes()),
            (0, 0)
        );
        assert!(non_owner.dense.tensors.is_empty());
        let single = VocabularyPlan::new(&records, &selection(false), 1, 0, 8, 256)?;
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
    fn flash_vocabulary_selects_only_target_tables_with_native_mtp_present() -> Result<()> {
        const EMBED: &str = "model.language_model.embed_tokens.weight";
        const N: usize = 154880;
        const K: usize = 4096;
        let root = tempfile::tempdir()?;
        let native = |name: &str| {
            let mut item = record(root.path(), name);
            item.tensor = name.into();
            item.shape = vec![N, K];
            item.native_blocks = N / 128;
            item.components[0].shape = vec![N, K];
            item.components[0].source.bytes = N * K;
            item.components[1].shape = vec![N];
            item.components[1].source.offset = (N * K) as u64;
            item.components[1].source.bytes = N * 4;
            item
        };
        let records = vec![native(EMBED), native(HEAD)];
        let mut with_mtp = records.clone();
        let mut eh = native("model.language_model.layers.45.eh_proj.weight");
        eh.encoding = DenseEncoding::Mxfp8;
        eh.shape = vec![K, 2 * K];
        eh.side = ShardSide::Replicated;
        eh.channels_per_shard = 0;
        eh.native_blocks = K / 128;
        eh.components[0].shape = vec![K, 2 * K];
        eh.components[0].source.bytes = 2 * K * K;
        eh.components[1].suffix = "e8m0".into();
        eh.components[1].dtype = "U8".into();
        eh.components[1].shape = vec![K, 2 * K / 32];
        eh.components[1].source.bytes = 2 * K * K / 32;
        with_mtp.push(eh);
        with_mtp.push(native("model.language_model.layers.45.shared_head.head.weight"));
        // A second family prefix cannot substitute for the selected target embedding.
        with_mtp.push(native(EMBEDDING));
        for world in [1, 2] {
            for rank in 0..world {
                let before = VocabularyPlan::new(&records, &selection(true), world, rank, K, N)?;
                let after = VocabularyPlan::new(&with_mtp, &selection(true), world, rank, K, N)?;
                let groups = after.resident_groups(&selection(true), 45)?;
                assert_eq!(groups.iter().map(|v| v.2 as usize).sum::<usize>(), after.resident_bytes());
                assert!(groups.iter().all(|(group, _, _)| group == "embedding" || group == "lm_head"));
                assert_eq!((before.resident_bytes(), before.staging_bytes()),
                    (after.resident_bytes(), after.staging_bytes()));
                let names: Vec<_> = after.dense.tensors.iter().map(|slice| slice.tensor.as_str()).collect();
                assert_eq!(names, if rank == 0 { vec![EMBED, HEAD] } else { vec![] });
                for slice in &after.dense.tensors {
                    let target = records.iter().find(|record| record.tensor == slice.tensor).unwrap();
                    assert!(slice.units.iter().all(|unit| unit.source.iter().all(|source|
                        source.path == target.components[0].source.path)));
                }
            }
        }
        let mtp_only = vec![with_mtp[3].clone(), records[1].clone()];
        assert!(VocabularyPlan::new(&mtp_only, &selection(true), 1, 0, K, N).is_err());
        assert!(VocabularyPlan::new(&with_mtp[1..], &selection(true), 1, 0, K, N).is_err());
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
        let plan = VocabularyPlan::new(&records, &selection(false), 2, 0, K, N)?;
        let owner = NativeVocabulary::load(&library, &programs, plan, 32 << 20, 0, 0)?
            .context("rank0 vocabulary owner")?;
        assert!(
            NativeVocabulary::load(
                &library,
                &programs,
                VocabularyPlan::new(&records, &selection(false), 2, 1, K, N)?,
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
                VocabularyPlan::new(&bad, &selection(false), 2, 0, 8, 256).is_err(),
                "case {change}"
            );
        }
        assert!(VocabularyPlan::new(&records[..1], &selection(false), 2, 0, 8, 256).is_err());
        assert!(
            VocabularyPlan::new(
                &[records[0].clone(), records[0].clone(), records[1].clone()],
                &selection(false),
                2,
                0,
                8,
                256
            )
            .is_err()
        );
        assert!(VocabularyPlan::new(&records, &selection(false), 2, 2, 8, 256).is_err());
        assert!(VocabularyPlan::new(&records, &selection(false), 3, 0, 8, 256).is_err());
        Ok(())
    }
}
