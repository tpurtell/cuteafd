//! Dormant checkpoint-native vocabulary ownership; the official token I/O is unchanged.
use crate::shared::experts::exlr8::view;
use crate::shared::experts::rtx::dense::{self, DenseLoad, Operand, Rect};
use crate::shared::memory::{DeviceAllocation, HostAllocation};
use anyhow::{Context, Result, ensure};
use cuteafd_ffi::native_dense::{DenseEmbedding, DenseLinear};
use cuteafd_ffi::programs::Programs;
use cuteafd_ffi::{CuteafdDeviceBuffer, CuteafdHostBuffer, NativeLibrary};
use cuteafd_loader::plan::exlr8::dense::{
    DenseEncoding, DensePlan, DenseRange, DenseRecord, DenseSelection, DenseSlice, ShardSide,
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
        let staging_bytes = dense.unit_staging_bytes()?;
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

    pub(crate) fn dense_plan(&self) -> &DensePlan { &self.dense }

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
        self.admit_devices(device_available, None, host_available, device_reserved, host_reserved)
    }

    fn admit_devices(&self, head_available: usize, embedding_available: Option<usize>,
        host_available: usize, device_reserved: usize, host_reserved: usize) -> Result<()> {
        let resident = if let Some(available) = embedding_available {
            let embedding = self.dense.tensors.first().context("native vocabulary embedding missing")?
                .components.iter().try_fold(0usize, |sum, c| sum.checked_add(c.source.bytes)
                    .context("native embedding admission overflow"))?;
            ensure!(embedding <= available, "native vocabulary embedding device admission needs {embedding}, available {available}");
            self.resident_bytes().checked_sub(embedding).context("native vocabulary owner partition")?
        } else { self.resident_bytes() };
        let device = resident
            .checked_add(device_reserved)
            .context("native vocabulary device admission overflow")?;
        let device_available = head_available;
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
    source: [DenseRange; 2],
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

    pub(crate) fn host_row(&self, library: &NativeLibrary, token: usize, hidden: usize,
        output: &mut [u8]) -> Result<()> {
        ensure!(hidden.checked_mul(2) == Some(output.len()), "native vocabulary BF16 row width");
        let mut bytes = vec![0u8; hidden];
        let mut scale = [0u8; 4];
        if dense::is_recording() {
            // Recorded destinations have not landed. Only the requested row
            // comes from storage, with the same native decode as the mount.
            read_source_row(&self.source, token, hidden, &mut bytes, &mut scale)?;
        } else {
            let first = token.checked_mul(hidden).context("native vocabulary row offset overflow")?;
            let scale_first = token.checked_mul(4).context("native vocabulary scale offset overflow")?;
            library.copy_d2h(&mut bytes, view(self.payload(), first, hidden)?)?;
            library.copy_d2h(&mut scale, view(self.scales(), scale_first, 4)?)?;
        }
        let scale = f32::from_le_bytes(scale);
        for (out, &b) in output.chunks_exact_mut(2).zip(&bytes) {
            out.copy_from_slice(&super::bf16_rne(super::e4m3_to_f32(b) * scale).to_le_bytes());
        }
        Ok(())
    }
}

fn read_source_row(source: &[DenseRange; 2], token: usize, hidden: usize,
    payload: &mut [u8], scale: &mut [u8; 4]) -> Result<()> {
    use std::os::unix::fs::FileExt;
    ensure!(hidden > 0 && payload.len() == hidden, "native vocabulary row width");
    for (range, width, output) in [(&source[0], hidden, payload), (&source[1], 4, scale.as_mut_slice())] {
        let first = token.checked_mul(width).context("native vocabulary row offset overflow")?;
        ensure!(first.checked_add(width).is_some_and(|end| end <= range.bytes),
            "native vocabulary row outside source");
        let offset = range.offset.checked_add(u64::try_from(first)?)
            .context("native vocabulary source offset overflow")?;
        std::fs::File::open(&range.path)
            .with_context(|| format!("opening {}", range.path.display()))?
            .read_exact_at(output, offset)?;
    }
    Ok(())
}

/// Owns immutable native bytes. Callers must drain all users before releasing this owner.
/// Binding is done before allocation; the head stays on rank 0 while the
/// embedding may reside on a peer selected by admission.
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
        Self::load_with_embedding_device(library, programs, plan, host_available,
            device_reserved, host_reserved, library.cuda_get_device()?)
    }

    pub(crate) fn load_with_embedding_device(
        library: &'a NativeLibrary, programs: &'a Programs<'a>, plan: VocabularyPlan,
        host_available: usize, device_reserved: usize, host_reserved: usize, embedding_device: i32,
    ) -> Result<Option<Self>> {
        Self::load_inner(library, programs, plan, embedding_device,
            Some((host_available, device_reserved, host_reserved)))
    }

    pub(crate) fn load_admitted(library: &'a NativeLibrary, programs: &'a Programs<'a>,
        plan: VocabularyPlan, admission: &cuteafd_loader::placement::families::glm5::admission::NativeLoad,
    ) -> Result<Option<Self>> {
        admission.validate_vocabulary(plan.staging_bytes() as u64)?;
        Self::load_inner(library, programs, plan, library.cuda_get_device()?, None)
    }

    /// Consume an already-admitted vocabulary owner without sampling memory.
    pub(crate) fn load_prepared(library: &'a NativeLibrary, programs: &'a Programs<'a>,
        plan: VocabularyPlan, staging_bytes: u64, embedding_device: i32) -> Result<Option<Self>> {
        ensure!(plan.staging_bytes() as u64 == staging_bytes, "native vocabulary staging differs from admission");
        Self::load_inner(library, programs, plan, embedding_device, None)
    }

    fn load_inner(library: &'a NativeLibrary, programs: &'a Programs<'a>, plan: VocabularyPlan,
        embedding_device: i32, available: Option<(usize, usize, usize)>) -> Result<Option<Self>> {
        use crate::shared::memory::device::Device;
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
        if let Some((host_available, device_reserved, host_reserved)) = available {
            let host_available = cuteafd_ffi::cgroup_memory_headroom(host_available)?;
            let head_device = library.cuda_get_device()?;
            let embedding_available = if embedding_device == head_device { None } else {
                Some(Device { library, id: embedding_device }.run(|| library.cuda_memory_info())?.0)
            };
            plan.admit_devices(library.cuda_memory_info()?.0, embedding_available,
                host_available, device_reserved, host_reserved)?;
        }
        let mut staging =
            HostAllocation::named(library, plan.staging_bytes(), "native-vocabulary-staging")?;
        let embedding = Device { library, id: embedding_device }
            .run(|| load_rows(library, &plan.dense.tensors[0], &mut staging))?;
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

/// Validate the native partition before publishing bytes or deferred records.
fn row_lands(slice: &DenseSlice) -> Result<[Vec<(Rect, usize)>; 2]> {
    ensure!(slice.components.len() == 2 && !slice.units.is_empty(), "native vocabulary partition");
    let mut lands: [Vec<_>; 2] = Default::default();
    let mut covered = [0usize; 2];
    for (index, unit) in slice.units.iter().enumerate() {
        ensure!(unit.block == index && unit.source.len() == 2, "native vocabulary block order");
        for (i, source) in unit.source.iter().enumerate() {
            let component = &slice.components[i].source;
            let offset = usize::try_from(source.offset.checked_sub(component.offset)
                .context("native vocabulary component offset")?)?;
            let end = offset.checked_add(source.bytes).context("native vocabulary upload overflow")?;
            ensure!(source.bytes > 0 && source.path == component.path && offset == covered[i]
                && end <= component.bytes, "native vocabulary component coverage");
            lands[i].push((Rect { name: format!("{}/{}", slice.tensor, slice.components[i].suffix),
                path: source.path.clone(), base: source.offset, row_bytes: source.bytes,
                rows: 0..1, cols: 0..source.bytes }, offset));
            covered[i] = end;
        }
    }
    ensure!(covered == [slice.components[0].source.bytes, slice.components[1].source.bytes],
        "native vocabulary incomplete component coverage");
    Ok(lands)
}

/// Source order is payload then scales. Read the whole block before any upload.
/// The callback is blocking, so staging is reusable and partial owners can drop safely.
fn stream_rows(
    slice: &DenseSlice,
    staging: &mut [u8],
    mut publish: impl FnMut(usize, usize, &[u8]) -> Result<()>,
) -> Result<()> {
    let lands = row_lands(slice)?;
    for (index, unit) in slice.units.iter().enumerate() {
        let bytes = unit.staging_bytes()?;
        ensure!(bytes <= staging.len(), "native vocabulary staging extent");
        unit.read(&mut staging[..bytes])?;
        let mut at = 0;
        for (i, source) in unit.source.iter().enumerate() {
            publish(i, lands[i][index].1, &staging[at..at + source.bytes])?;
            at += source.bytes;
        }
    }
    Ok(())
}

fn record_rows(slice: &DenseSlice, buffers: [CuteafdDeviceBuffer; 2],
    mut publish: impl FnMut(Operand) -> Result<()>) -> Result<()> {
    for (lands, buffer) in row_lands(slice)?.into_iter().zip(buffers) {
        let bytes = lands.iter().map(|(rect, _)| rect.bytes()).sum();
        publish(Operand { device: buffer.device_id, buffer, lands, bytes })?;
    }
    Ok(())
}

pub(crate) fn row_requests(slice: &DenseSlice) -> Result<Vec<(String, usize)>> {
    row_lands(slice)?;
    slice.components.iter().map(|component| Ok((component.suffix.clone(),
        cuteafd_loader::plan::exlr8::dense::DenseTransform::Identity.allocation_bytes(component)?))).collect()
}

fn load_rows<'a>(
    library: &'a NativeLibrary,
    slice: &DenseSlice,
    staging: &mut HostAllocation<'a>,
) -> Result<RowFp8Weights<'a>> {
    let requests = row_requests(slice)?;
    let device = library.cuda_get_device()?;
    let weights = RowFp8Weights {
        payload: DeviceAllocation::named(
            library,
            requests[0].1,
            "native-vocabulary-payload",
        )?,
        scales: DeviceAllocation::named(
            library,
            requests[1].1,
            "native-vocabulary-scales",
        )?,
        source: [slice.components[0].source.clone(), slice.components[1].source.clone()],
    };
    for ((name, bytes), buffer) in requests.iter().zip([weights.payload(), weights.scales()]) {
        ensure!(buffer.bytes == *bytes && buffer.device_id == device && !buffer.ptr.is_null(),
            "{name}: native vocabulary allocation differs from request");
    }
    if dense::is_recording() {
        let load = DenseLoad::mount(library);
        record_rows(slice, [weights.payload(), weights.scales()], |operand| load.operand(operand))?;
        return Ok(weights);
    }
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

    #[test]
    fn compact_native_row_uploads_match_component_requests() -> Result<()> {
        let root = std::path::PathBuf::from(std::env::var_os("HOME").context("HOME")?)
            .join(".cache/cuteafd/builds/glm-weight-inventory-fixtures");
        let fixture = cuteafd_ffi::native_library_lifetime_fixture::Fixture::build_in(&root)?;
        let library = fixture.load()?;
        fixture.configure_memory(&library, 0, 1 << 30)?;
        let root = tempfile::tempdir_in(fixture.directory())?;
        let record = record(root.path(), EMBEDDING);
        let plan = DensePlan::for_rank(&[record], 1, 0)?;
        let slice = &plan.tensors[0];
        let requests = row_requests(slice)?;
        let mut staging = HostAllocation::new(&library, plan.read_staging_bytes.max(256))?;
        let weights = load_rows(&library, slice, &mut staging)?;
        let expected = requests.into_iter().map(|(_, bytes)| bytes).collect::<Vec<_>>();
        assert_eq!(expected, [weights.payload().bytes, weights.scales().bytes]);
        Ok(())
    }

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
    fn split_vocabulary_admission_charges_each_table_once() -> Result<()> {
        let root = tempfile::tempdir()?;
        let records = [record(root.path(), EMBEDDING), record(root.path(), HEAD)];
        let plan = VocabularyPlan::new(&records, &selection(false), 2, 0, 8, 256)?;
        let table = plan.resident_bytes() / 2;
        let host = plan.staging_bytes() + 17;
        plan.admit_devices(table + 23, Some(table), host, 23, 17)?;
        assert!(plan.admit_devices(table + 22, Some(table), host, 23, 17).is_err());
        assert!(plan.admit_devices(table + 23, Some(table - 1), host, 23, 17).is_err());
        assert!(plan.admit_devices(table + 23, Some(table), host - 1, 23, 17).is_err());
        assert!(plan.admit_devices(table + 23, None, host, 23, 17).is_err());
        plan.admit_devices(2 * table + 23, None, host, 23, 17)?;
        Ok(())
    }

    #[test]
    fn native_vocabulary_recording_replays_exact_mount_components() -> Result<()> {
        let root = tempfile::tempdir()?;
        for flash in [false, true] {
            let records = selection(flash).vocabulary_names().into_iter()
                .map(|name| record(root.path(), &name)).collect::<Vec<_>>();
            let plan = VocabularyPlan::new(&records, &selection(flash), 1, 0, 8, 256)?;
            for slice in &plan.dense.tensors {
                let buffers = std::array::from_fn(|i| CuteafdDeviceBuffer {
                    bytes: slice.components[i].source.bytes, device_id: 3, ..Default::default()
                });
                let sink = dense::DenseSink::default();
                record_rows(slice, buffers, |operand| sink.operand(operand))?;
                let operands = sink.take_all().operands;
                assert_eq!(operands.len(), 2);
                let mut mount = [vec![0; buffers[0].bytes], vec![0; buffers[1].bytes]];
                stream_rows(slice, &mut vec![0; plan.staging_bytes()], |i, at, bytes| {
                    mount[i][at..at + bytes.len()].copy_from_slice(bytes);
                    Ok(())
                })?;
                for (operand, expected) in operands.iter().zip(&mount) {
                    assert_eq!(operand.device, 3);
                    let mut replay = vec![0; operand.bytes];
                    for (rect, at) in &operand.lands {
                        rect.read(&mut replay[*at..*at + rect.bytes()])?;
                    }
                    assert_eq!(&replay, expected);
                }
                let mut invalid = slice.clone();
                invalid.units.last_mut().unwrap().block += 1;
                let mut published = 0;
                assert!(record_rows(&invalid, buffers, |_| { published += 1; Ok(()) }).is_err());
                assert_eq!(published, 0);
                // Recording validates metadata, not source availability.
                let mut absent = slice.clone();
                for component in &mut absent.components { component.source.path = root.path().join("absent"); }
                for unit in &mut absent.units {
                    for source in &mut unit.source { source.path = root.path().join("absent"); }
                }
                record_rows(&absent, buffers, |operand| sink.operand(operand))?;
            }
        }
        Ok(())
    }

    #[test]
    fn source_mask_rows_match_mount_stream_with_separate_component_files() -> Result<()> {
        let root = tempfile::tempdir()?;
        let mut item = record(root.path(), EMBEDDING);
        let payload = (0..256 * 8).map(|i| i as u8).collect::<Vec<_>>();
        let scales = (0..256).flat_map(|i| [0.5f32, 1.0, 1.00390625, 0.0, -2.0][i % 5].to_le_bytes())
            .collect::<Vec<_>>();
        for (i, bytes) in [&payload, &scales].into_iter().enumerate() {
            let prefix = 13 + i * 7;
            let path = root.path().join(format!("component-{i}"));
            std::fs::write(&path, [vec![0xa5; prefix], bytes.clone()].concat())?;
            item.components[i].source = DenseRange { path, offset: prefix as u64, bytes: bytes.len() };
        }
        let slice = item.slice(0, 256)?;
        let source = [slice.components[0].source.clone(), slice.components[1].source.clone()];
        let mut mounted = [vec![0; payload.len()], vec![0; scales.len()]];
        stream_rows(&slice, &mut vec![0; 1536], |i, at, bytes| {
            mounted[i][at..at + bytes.len()].copy_from_slice(bytes);
            Ok(())
        })?;
        for token in 0..256 {
            let mut row = [0u8; 8];
            let mut scale = [0u8; 4];
            read_source_row(&source, token, 8, &mut row, &mut scale)?;
            assert_eq!(row, mounted[0][token * 8..(token + 1) * 8]);
            assert_eq!(scale, mounted[1][token * 4..(token + 1) * 4]);
            let decode = |bytes: &[u8], scale: &[u8]| {
                let scale = f32::from_le_bytes(scale.try_into().unwrap());
                bytes.iter().flat_map(|&b| super::super::bf16_rne(
                    super::super::e4m3_to_f32(b) * scale).to_le_bytes()).collect::<Vec<_>>()
            };
            assert_eq!(decode(&row, &scale), decode(&mounted[0][token * 8..(token + 1) * 8],
                &mounted[1][token * 4..(token + 1) * 4]));
        }
        assert!(read_source_row(&source, 256, 8, &mut [0; 8], &mut [0; 4]).is_err());
        assert!(read_source_row(&source, usize::MAX, 8, &mut [0; 8], &mut [0; 4]).is_err());
        assert!(read_source_row(&source, 0, 8, &mut [0; 7], &mut [0; 4]).is_err());
        Ok(())
    }

    #[test]
    #[ignore = "requires an allocated CPU build slot and explicit NVMe fixture directory"]
    fn recorded_mask_row_matches_mounted_device_without_reading_unlanded_bytes() -> Result<()> {
        let fixture = cuteafd_ffi::native_library_lifetime_fixture::Fixture::build()?;
        let library = fixture.load()?;
        fixture.configure_memory(&library, 1, 1 << 30)?;
        fixture.immediate_copies(&library)?;
        let root = tempfile::tempdir_in(fixture.directory())?;
        let item = record(root.path(), EMBEDDING);
        let mut bytes = (0..256 * 8).map(|i| i as u8).collect::<Vec<_>>();
        bytes.extend((0..256).flat_map(|i| [0.5f32, 1.00390625, -2.0][i % 3].to_le_bytes()));
        std::fs::write(&item.components[0].source.path, &bytes)?;
        let slice = item.slice(0, 256)?;
        let mut staging = HostAllocation::named(&library, 1536, "mask-row-test")?;
        let mounted = load_rows(&library, &slice, &mut staging)?;
        let ((recorded, rows), records) = dense::recording(|| {
            let recorded = load_rows(&library, &slice, &mut staging)?;
            // Poison both device tables: recording must use storage, not these.
            library.copy_h2d(recorded.payload(), &vec![0x7f; 256 * 8])?;
            library.copy_h2d(recorded.scales(), &vec![0xff; 256 * 4])?;
            let mut rows = Vec::new();
            for token in 0..256 {
                let mut row = vec![0; 16];
                recorded.host_row(&library, token, 8, &mut row)?;
                rows.push(row);
            }
            Ok((recorded, rows))
        })?;
        assert_eq!(records.operands.len(), 2);
        for (token, expected) in rows.into_iter().enumerate() {
            let mut actual = vec![0; 16];
            mounted.host_row(&library, token, 8, &mut actual)?;
            assert_eq!(actual, expected, "mask token {token}");
        }
        drop(recorded);
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
        drop(owner);
        let stream = crate::shared::memory::device::Stream::new(
            crate::shared::memory::device::Device { library: &library, id: 0 })?;
        let expanded = DeviceAllocation::new(&library, 5 * 4 * K * 2)?;
        for embedding_device in [0, 1] {
            if embedding_device == 1 {
                stream.device.run(|| library.cuda_enable_peer(1))?;
            }
            let vocabulary = std::rc::Rc::new(NativeVocabulary::load_with_embedding_device(
                &library, &programs, VocabularyPlan::new(&records, &selection(false), 2, 0, K, N)?,
                32 << 20, 0, 0, embedding_device)?.context("capture vocabulary")?);
            let embedding = crate::shared::token_io::TokenEmbedding::row_fp8(&library, crate::shared::token_io::EmbedSource {
                path: records[0].components[0].source.path.clone(), offset: 0, vocab: N, hidden: K,
                snapshot: None, tensor_name: EMBEDDING.into(),
            }, vocabulary)?;
            embedding.validate_residency(embedding_device, (N * K + N * 4) as u64)?;
            // SAFETY: all buffers and the table survive capture, replay and the stream drain.
            unsafe {
                library.cuda_graph_begin_capture(stream.raw)?;
                let gathered = embedding.gather(ids.buffer.ptr, std::ptr::null(), 5, 4,
                    std::ptr::null(), expanded.buffer.ptr, Some(output.buffer),
                    crate::shared::peer_split::RankDevice { device: stream.device.id, stream: stream.raw });
                let graph = library.cuda_graph_end_capture(stream.raw)?;
                gathered?;
                let replayed = library.cuda_graph_launch(graph, stream.raw);
                let drained = stream.drain();
                library.cuda_graph_exec_destroy(graph)?;
                replayed?;
                drained?;
            }
            let mut copied = vec![0; expanded.buffer.bytes];
            library.copy_d2h(&mut copied, expanded.buffer)?;
            for (row, bytes) in copied.chunks_exact(4 * K * 2).enumerate() {
                let expected = if row < 3 { 0x3f00u16 } else { 0 };
                assert!(bytes.chunks_exact(2).all(|b| u16::from_le_bytes(b.try_into().unwrap()) == expected));
            }
        }
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
