//! Official coordinator tensors retain their native checkpoint representations.
use crate::shared::memory::{DeviceAllocation, HostAllocation};
use anyhow::{ensure, Context, Result};
use cuteafd_ffi::{CuteafdDeviceBuffer, NativeLibrary};
use cuteafd_loader::OfficialV41Catalog;
use std::collections::{BTreeMap, BTreeSet};
#[path = "v41_tensors/vocabulary_shard.rs"]
mod vocabulary_shard;
pub(crate) use vocabulary_shard::VocabularyShard;

pub(crate) struct NativeRtxTensors<'a> {
    tensors: BTreeMap<String, DeviceAllocation<'a>>,
    resident_bytes: usize,
    mapped_embedding: Option<(HostAllocation<'a>, CuteafdDeviceBuffer)>,
}
impl<'a> NativeRtxTensors<'a> {
    /// Target and dSpark borrow the same immutable table on both CED devices.
    pub fn load_embedding(library: &'a NativeLibrary, catalog: &OfficialV41Catalog,
        placement: crate::shared::memory::EmbedPlacement) -> Result<Self> {
        let _scope = cuteafd_ffi::memory_ledger::scope("embedding");
        let names = ["embed.weight".to_owned()];
        let bytes = Self::plan(catalog, &names)?;
        if placement == crate::shared::memory::EmbedPlacement::Gpu {
            tracing::info!(?placement, device_bytes=bytes, pinned_bytes=0, "V4.1 embedding placement (single residency, RTX0 only)");
            return Self::load(library, catalog, &names, bytes, 16 << 20);
        }
        cuteafd_loader::plan::checkpoint::Checkpoint::open(catalog.snapshot())?
            .require_untied_embedding("embed.weight")?;
        let mut storage = HostAllocation::new(library, bytes)?;
        catalog.coordinator_tensor_reader("embed.weight")?.read_into(0, storage.bytes_mut())?;
        let alias = library.cuda_host_buffer_device_alias(storage.buffer)?;
        tracing::info!(?placement, device_bytes=0, pinned_bytes=bytes, "V4.1 embedding placement (single residency, shared target/dSpark)");
        Ok(Self { tensors: BTreeMap::new(), resident_bytes: 0, mapped_embedding: Some((storage, alias)) })
    }

    pub fn plan(catalog: &OfficialV41Catalog, names: &[String]) -> Result<usize> {
        ensure!(!names.is_empty(), "RTX tensor set is empty");
        let mut seen = BTreeSet::new();
        names.iter().try_fold(0usize, |total, name| {
            ensure!(seen.insert(name), "duplicate RTX tensor {name}");
            ensure!(
                !name.contains(".ffn.experts."),
                "routed expert weights require native expert packing"
            );
            let bytes = usize::try_from(catalog.device_tensor_bytes(name, None)?)?;
            total
                .checked_add(bytes)
                .context("RTX resident tensor budget overflow")
        })
    }
    /// Admission precedes allocation and payload reads; bounded pinned staging is
    /// released after synchronous uploads. Engram tables cannot enter this path.
    pub fn load(
        library: &'a NativeLibrary,
        catalog: &OfficialV41Catalog,
        names: &[String],
        device_budget: usize,
        staging_bytes: usize,
    ) -> Result<Self> {
        let resident_bytes = Self::plan(catalog, names)?;
        ensure!(
            resident_bytes <= device_budget,
            "RTX tensor set exceeds device budget"
        );
        ensure!(
            staging_bytes > 0 && staging_bytes <= 64 * 1024 * 1024,
            "RTX pinned staging must be 1 byte through 64 MiB"
        );
        let mut staging = HostAllocation::new(library, staging_bytes)?;
        let mut tensors = BTreeMap::new();
        for name in names {
            let reader = catalog.coordinator_tensor_reader(name)?;
            let bytes = usize::try_from(reader.bytes())?;
            let allocation = DeviceAllocation::new(library, bytes)?;
            let mut offset = 0;
            while offset < bytes {
                let count = staging_bytes.min(bytes - offset);
                let source = &mut staging.bytes_mut()[..count];
                reader.read_into(offset as u64, source)?;
                let mut destination = allocation.buffer;
                destination.ptr = unsafe { destination.ptr.cast::<u8>().add(offset).cast() };
                destination.bytes = count;
                library.copy_h2d(destination, source)?;
                offset += count;
            }
            tensors.insert(name.clone(), allocation);
        }
        Ok(Self {
            tensors,
            resident_bytes,
            mapped_embedding: None,
        })
    }
    /// Borrowed native representation; never free or retain after the owner drops.
    pub fn get(&self, name: &str) -> Result<CuteafdDeviceBuffer> {
        if name == "embed.weight" {
            if let Some((_, alias)) = &self.mapped_embedding { return Ok(*alias); }
        }
        Ok(self
            .tensors
            .get(name)
            .with_context(|| format!("RTX tensor is not resident: {name}"))?
            .buffer)
    }
    pub fn resident_bytes(&self) -> usize {
        self.resident_bytes
    }
}

/// One vocabulary weight shared by the backbone and dSpark. Each execution
/// owns its handle and workspace; `all` retains only the packed FP8 weight.
pub(crate) struct VocabularyHead<'library> {
    shard: VocabularyShard<'library>,
}

/// E4M3 vocabulary heads (`CUTEAFD_V41_FP8_HEAD`): `draft` projects only the
/// dSpark draft head through an FP8 copy (outputs unchanged, acceptance may
/// move), `all` (default) both heads from one FP8 copy; `off` keeps BF16. Copies use per-row x
/// 128-K FP32 scales (`fp8_linear`, the better of amax and power-of-two per
/// block) and the W8A16 tensor-core GEMV. `all` releases BF16 after packing;
/// `draft` retains BF16 for target projections.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Fp8Head { Off, Draft, All }
pub(crate) fn fp8_head() -> Fp8Head {
    static MODE: std::sync::OnceLock<Fp8Head> = std::sync::OnceLock::new();
    *MODE.get_or_init(|| {
        let mode = match std::env::var("CUTEAFD_V41_FP8_HEAD").as_deref() {
            Ok("off" | "0" | "bf16") => Fp8Head::Off,
            Ok("draft") => Fp8Head::Draft,
            _ => Fp8Head::All,
        };
        tracing::info!(?mode, "V4.1 vocabulary heads");
        mode
    })
}

/// Packs a resident BF16 head, draining before releasing the source. Failed
/// drains quarantine both formats and the native module until process teardown.
fn pack_fp8<'a>(library: &'a NativeLibrary, weight: DeviceAllocation<'a>, rows: usize)
    -> Result<(Option<DeviceAllocation<'a>>, crate::shared::fp8_linear::Fp8Weight<'a>)> {
    ensure!(weight.buffer.bytes == rows * 10240, "FP8 head geometry differs from BF16");
    let stream = library.cuda_stream_create()?;
    let packed = crate::shared::fp8_linear::Fp8Weight::pack(library, weight.buffer.ptr, rows, 5120,
        crate::shared::fp8_linear::Fp8Scales::Best, stream);
    // SAFETY: source and packed destinations remain owned until the pack drains.
    if let Err(error) = unsafe { library.cuda_stream_synchronize(stream) } {
        library.quarantine_module_after_failed_drain();
        std::mem::forget(weight);
        if let Ok(packed) = packed { packed.quarantine(); }
        return Err(error.context("FP8 vocabulary pack drain failed; source and destinations quarantined"));
    }
    // SAFETY: the only queued work has completed.
    unsafe { library.cuda_stream_destroy(stream)?; }
    let packed = packed?;
    let source = if fp8_head() == Fp8Head::All { drop(weight); None } else { Some(weight) };
    Ok((source, packed))
}

/// A vocabulary projection through the FP8 copy when the caller holds scratch
/// for it (see [`fp8_head`]), else the BF16 head.
///
/// # Safety
/// As [`cuteafd_ffi::V41VocabularyProjection::launch`]; `scratch` was sized by
/// [`fp8_scratch`] for at least `rows` rows of this head.
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn project_vocabulary(library: &NativeLibrary, projection: &cuteafd_ffi::V41VocabularyProjection<'_>,
    bf16: Option<CuteafdDeviceBuffer>, fp8: Option<(&crate::shared::fp8_linear::Fp8Weight<'_>, &DeviceAllocation<'_>)>,
    input: CuteafdDeviceBuffer, logits: CuteafdDeviceBuffer, rows: usize, stream: *mut std::ffi::c_void) -> Result<()> {
    match fp8 {
        Some((weight, scratch)) => {
            ensure!(input.bytes >= rows * 10240 && logits.bytes >= rows * weight.n * 4
                && input.device_id == logits.device_id, "FP8 vocabulary projection buffers differ");
            unsafe { weight.apply(library, input.ptr, logits.ptr, true, rows, 0, weight.n, scratch, stream) }
        }
        None => unsafe { projection.launch(input, bf16.context("BF16 vocabulary is not resident; FP8 scratch required")?, logits, rows, stream) },
    }
}

/// GEMV scratch for `rows` rows of an FP8 head copy, when `site` uses one.
pub(crate) fn fp8_scratch<'a>(library: &'a NativeLibrary, fp8: Option<&crate::shared::fp8_linear::Fp8Weight<'_>>,
    rows: usize, site: Fp8Head) -> Result<Option<DeviceAllocation<'a>>> {
    let enabled = match fp8_head() { Fp8Head::Off => false, Fp8Head::Draft => site == Fp8Head::Draft, Fp8Head::All => true };
    match fp8.filter(|_| enabled) {
        Some(weight) => Ok(Some(crate::shared::fp8_linear::scratch(library, rows, &[(weight.k, weight.n)])?)),
        None => Ok(None),
    }
}
impl<'library> VocabularyHead<'library> {
    /// Peak load admission, including BF16 staging and FP8 destinations.
    pub fn plan(catalog: &OfficialV41Catalog) -> Result<usize> {
        VocabularyShard::load_bytes(catalog, 0..129280)
    }
    pub fn resident_bytes(catalog: &OfficialV41Catalog) -> Result<usize> {
        let source = VocabularyShard::device_bytes(catalog, 0..129280)?;
        Ok(match fp8_head() {
            Fp8Head::Off => source,
            Fp8Head::Draft => source + source / 2 + source / 64,
            Fp8Head::All => source / 2 + source / 64,
        })
    }
    pub fn load(
        library: &'library NativeLibrary,
        catalog: &OfficialV41Catalog,
        budget: usize,
        staging_bytes: usize,
    ) -> Result<Self> {
        Ok(Self { shard: VocabularyShard::load(library, catalog, 0..129280, budget, staging_bytes)? })
    }
    pub fn weight(&self) -> Option<CuteafdDeviceBuffer> { self.shard.weight() }
    pub fn device_id(&self) -> i32 { self.shard.device_id() }
    pub fn fp8(&self) -> Option<&crate::shared::fp8_linear::Fp8Weight<'library>> { self.shard.fp8() }
}
