//! Official coordinator tensors retain their native checkpoint representations.
use crate::shared::memory::Tag;
use crate::shared::memory::{DeviceAllocation, HostAllocation, device::Device,
    weight_arena::{DeviceWeightArena, weight_allocation}};
use anyhow::{ensure, Context, Result};
use cuteafd_ffi::{CuteafdDeviceBuffer, NativeLibrary};
use cuteafd_loader::OfficialV41Catalog;
use std::collections::{BTreeMap, BTreeSet};
use cuteafd_core::memory_layout::ArenaTag;
#[path = "v41_tensors/vocabulary_shard.rs"]
mod vocabulary_shard;
pub(crate) use vocabulary_shard::VocabularyShard;
#[cfg(test)]
pub(crate) use vocabulary_shard::tests::packed_head_push_delivers_transient_source_before_arena_pack;

pub(crate) fn allocation<'a>(library: &'a NativeLibrary,
    arena: Option<&DeviceWeightArena<'a>>, name: &str, bytes: usize) -> Result<DeviceAllocation<'a>> {
    Ok(DeviceAllocation::from_allocation(weight_allocation(
        Device { library, id: library.cuda_get_device()? }, arena, name, bytes, ArenaTag::weights("v41/target"))?))
}

/// Checkpoint tensor `name` (an unsharded coordinator tensor) as one
/// rectangle: its rows (`rows` of them, or all), each whole.
pub(crate) fn rect(catalog: &OfficialV41Catalog, name: &str, rows: Option<std::ops::Range<usize>>)
    -> Result<crate::shared::experts::rtx::dense::Rect> {
    let tensor = catalog.tensor(name)?;
    ensure!(matches!(tensor.placement, cuteafd_loader::V41TensorPlacement::CoordinatorRtx),
        "{name}: only unsharded RTX tensors land on the coordinator");
    let total = tensor.metadata.shape.first().copied().filter(|_| tensor.metadata.shape.len() >= 2).unwrap_or(1);
    let bytes = usize::try_from(tensor.metadata.byte_length)?;
    ensure!(total > 0 && bytes % total == 0, "{name}: rows do not divide the tensor");
    let rows = rows.unwrap_or(0..total);
    ensure!(rows.end <= total && !rows.is_empty(), "{name}: rows outside the tensor");
    Ok(crate::shared::experts::rtx::dense::Rect { name: name.into(), path: catalog.snapshot().join(&tensor.shard),
        base: tensor.metadata.byte_offset, row_bytes: bytes / total, rows, cols: 0..bytes / total })
}

/// Row (`axis` 0) or column (`axis` 1) half `rank` of an unsharded
/// coordinator matrix, as one rectangle.
pub(crate) fn half(catalog: &OfficialV41Catalog, name: &str, axis: usize, rank: usize)
    -> Result<crate::shared::experts::rtx::dense::Rect> {
    let tensor = catalog.tensor(name)?;
    let shape = &tensor.metadata.shape;
    ensure!(shape.len() == 2 && axis < 2 && rank < 2 && shape[axis] % 2 == 0, "{name}: no TP2 half on axis {axis}");
    let whole = rect(catalog, name, None)?;
    Ok(if axis == 0 {
        let half = shape[0] / 2;
        whole.sub(rank * half..(rank + 1) * half)
    } else {
        let width = whole.row_bytes / 2;
        crate::shared::experts::rtx::dense::Rect { cols: rank * width..(rank + 1) * width, ..whole }
    })
}

pub(crate) struct NativeRtxTensors<'a> {
    tensors: BTreeMap<String, DeviceAllocation<'a>>,
    resident_bytes: usize,
    mapped_embedding: Option<(HostAllocation<'a>, CuteafdDeviceBuffer, crate::shared::memory::HostLanding)>,
}
impl<'a> NativeRtxTensors<'a> {
    /// Target and dSpark borrow the same immutable table on both CED devices.
    #[cfg(test)]
    pub fn load_embedding(library: &'a NativeLibrary, catalog: &OfficialV41Catalog,
        placement: crate::shared::memory::EmbedPlacement) -> Result<Self> {
        Self::load_embedding_in(library, catalog, placement, None)
    }
    pub fn load_embedding_in(library: &'a NativeLibrary, catalog: &OfficialV41Catalog,
        placement: crate::shared::memory::EmbedPlacement, arena: Option<&DeviceWeightArena<'a>>) -> Result<Self> {
        let names = ["embed.weight".to_owned()];
        let bytes = Self::plan(catalog, &names)?;
        if placement == crate::shared::memory::EmbedPlacement::Gpu {
            tracing::info!(?placement, device_bytes=bytes, pinned_bytes=0, "V4.1 embedding placement (single residency, RTX0 only)");
            return Self::load_in(library, catalog, &names, bytes, 16 << 20, arena);
        }
        cuteafd_loader::plan::checkpoint::Checkpoint::open(catalog.snapshot())?
            .require_untied_embedding("embed.weight")?;
        let mut storage = HostAllocation::new(library, bytes, Tag::embedding("v41/embedding"))?;
        let landing = crate::shared::memory::HostLanding::new(&mut storage);
        crate::shared::experts::rtx::dense::host_operand(crate::shared::experts::rtx::dense::HostOperand {
            device: library.cuda_get_device()?, landing: landing.clone(),
            rect: rect(catalog, "embed.weight", None)?,
        }, crate::shared::experts::rtx::push::PushPolicy::for_snapshot(catalog.snapshot())?)?;
        let alias = library.cuda_host_buffer_device_alias(storage.buffer)?;
        tracing::info!(?placement, device_bytes=0, pinned_bytes=bytes, "V4.1 embedding placement (single residency, shared target/dSpark)");
        Ok(Self { tensors: BTreeMap::new(), resident_bytes: 0, mapped_embedding: Some((storage, alias, landing)) })
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
    #[cfg(test)]
    pub fn load(
        library: &'a NativeLibrary,
        catalog: &OfficialV41Catalog,
        names: &[String],
        device_budget: usize,
        staging_bytes: usize,
    ) -> Result<Self> {
        Self::load_in(library, catalog, names, device_budget, staging_bytes, None)
    }

    pub fn load_in(library: &'a NativeLibrary, catalog: &OfficialV41Catalog, names: &[String],
        device_budget: usize, staging_bytes: usize, arena: Option<&DeviceWeightArena<'a>>) -> Result<Self> {
        let resident_bytes = Self::plan(catalog, names)?;
        ensure!(
            resident_bytes <= device_budget,
            "RTX tensor set exceeds device budget"
        );
        ensure!(
            staging_bytes > 0 && staging_bytes <= 64 * 1024 * 1024,
            "RTX pinned staging must be 1 byte through 64 MiB"
        );
        let _ = staging_bytes;
        let dense = crate::shared::experts::rtx::dense::DenseLoad::mount(library);
        let device = library.cuda_get_device()?;
        let mut tensors = BTreeMap::new();
        for name in names {
            let bytes = usize::try_from(catalog.device_tensor_bytes(name, None)?)?;
            let allocation = DeviceAllocation::from_allocation(weight_allocation(
                Device { library, id: device }, arena, name, bytes, ArenaTag::weights("v41/target"))?);
            dense.operand(crate::shared::experts::rtx::dense::Operand { device, buffer: allocation.buffer,
                lands: vec![(rect(catalog, name, None)?, 0)], bytes })?;
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
            if let Some((_, alias, landing)) = &self.mapped_embedding {
                ensure!(landing.ready.load(std::sync::atomic::Ordering::Acquire),
                    "host embedding is recorded but not committed");
                return Ok(*alias);
            }
        }
        Ok(self
            .tensors
            .get(name)
            .with_context(|| format!("RTX tensor is not resident: {name}"))?
            .buffer)
    }
    /// Header-only geometry for workspace construction before payload commit.
    pub(crate) fn embedding_layout(&self) -> Result<(usize, i32)> {
        if let Some((_, alias, _)) = &self.mapped_embedding {
            return Ok((alias.bytes, alias.device_id));
        }
        let buffer = self.get("embed.weight")?;
        Ok((buffer.bytes, buffer.device_id))
    }
    pub fn resident_bytes(&self) -> usize {
        self.resident_bytes
    }
    pub(crate) fn inventory(&self, inventory: &mut crate::shared::memory::receipts::Inventory<'_>) -> Result<()> {
        for (name, allocation) in &self.tensors { allocation.inventory(inventory, name)?; }
        if let Some((owner, _, _)) = &self.mapped_embedding { inventory.pinned("embed.weight", owner.buffer)?; }
        Ok(())
    }
}

/// One vocabulary weight shared by the backbone and dSpark. Each execution
/// owns its handle and workspace; `all` retains only the packed FP8 weight,
/// `packed` only the exact 12-bit copy.
pub(crate) struct VocabularyHead<'library> {
    shard: VocabularyShard<'library>,
}

/// Vocabulary heads (`CUTEAFD_V41_FP8_HEAD`): `draft` projects only the dSpark
/// draft head through an E4M3 copy (outputs unchanged, acceptance may move),
/// `all` runs both heads from one E4M3 copy; `off` keeps BF16. E4M3 copies
/// use per-row x 128-K FP32 scales (`fp8_linear`, the better of amax and
/// power-of-two per block) and the W8A16 tensor-core GEMV. Unset/`auto` selects
/// `packed`, which keeps one
/// exact 12-bit copy of the BF16 head (`packed_head`): target logits read it
/// exactly, the dSpark drafter its coarse plane. `all` and `packed` release BF16
/// after packing; `draft` retains BF16 for target projections.
pub(crate) use cuteafd_loader::placement::families::deepseek_v41::resident::Fp8Head;
static HEAD_MODE: std::sync::OnceLock<Fp8Head> = std::sync::OnceLock::new();
pub(crate) fn fp8_head() -> Fp8Head {
    *HEAD_MODE.get_or_init(|| publish_head(Fp8Head::configured()))
}

#[cfg(test)]
pub(crate) fn use_golden_fp8_head() -> Result<()> {
    // The pinned operand inventory and its load budgets describe the all-FP8 head.
    let mode = HEAD_MODE.get_or_init(|| publish_head(Fp8Head::All));
    ensure!(*mode == Fp8Head::All, "run the coordinator operand oracle in its own test process");
    Ok(())
}

fn publish_head(mode: Fp8Head) -> Fp8Head {
    cuteafd_bench::context::set_resolved("v41-fp8-head", mode.setting());
    tracing::info!(?mode, "V4.1 vocabulary heads");
    mode
}

/// The resident copy of the vocabulary a projection can read instead of BF16.
#[derive(Clone, Copy)]
pub(crate) enum HeadCopy<'h, 'a> {
    /// E4M3 with FP32 scales per row and 128-wide K block (`draft`, `all`).
    Fp8(&'h crate::shared::fp8_linear::Fp8Weight<'a>),
    /// The exact 12-bit head (`packed`).
    Packed(&'h crate::shared::packed_head::PackedHead<'a>),
}

impl HeadCopy<'_, '_> {
    /// Vocabulary rows and width.
    fn shape(self) -> (usize, usize) {
        match self {
            HeadCopy::Fp8(weight) => (weight.n, weight.k),
            HeadCopy::Packed(head) => (head.n, head.k),
        }
    }
}

/// GEMV scratch of a head copy at one projection site.
pub(crate) struct HeadScratch<'a> {
    allocation: DeviceAllocation<'a>,
    /// The dSpark drafter's site: a packed head is read through its coarse plane.
    coarse: bool,
}

/// Packs a resident BF16 head, draining before releasing the source. Failed
/// drains quarantine both formats and the native module until process teardown.
fn pack_fp8_in<'a>(library: &'a NativeLibrary, weight: DeviceAllocation<'a>, rows: usize,
    arena: Option<&DeviceWeightArena<'a>>)
    -> Result<(Option<DeviceAllocation<'a>>, crate::shared::fp8_linear::Fp8Weight<'a>)> {
    ensure!(weight.buffer.bytes == rows * 10240, "FP8 head geometry differs from BF16");
    let stream = library.cuda_stream_create()?;
    let device = Device { library, id: library.cuda_get_device()? };
    let packed = crate::shared::fp8_linear::Fp8Weight::pack_with(library, weight.buffer.ptr, rows, 5120,
        crate::shared::fp8_linear::Fp8Scales::Best, stream, "head.fp8",
        |name, bytes| weight_allocation(device, arena, name, bytes, ArenaTag::weights("v41/target")));
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

/// Packs a resident BF16 head into the exact 12-bit copy, which verifies itself
/// bit for bit, then releases the BF16 source. A failed drain quarantines the
/// destination, the source and the native module until process teardown.
fn pack_packed_in<'a>(library: &'a NativeLibrary, weight: DeviceAllocation<'a>, rows: usize,
    arena: Option<&DeviceWeightArena<'a>>) -> Result<crate::shared::packed_head::PackedHead<'a>> {
    ensure!(weight.buffer.bytes == rows * 10240, "packed head geometry differs from BF16");
    let stream = library.cuda_stream_create()?;
    let device = Device { library, id: library.cuda_get_device()? };
    let packed = crate::shared::packed_head::PackedHead::pack_with(library, weight.buffer.ptr, rows, 5120, stream,
        |name, bytes| weight_allocation(device, arena, &format!("head.{name}"), bytes, ArenaTag::weights("v41/target")));
    if packed.is_err() && library.is_quarantined_after_failed_drain() {
        // The pack could not prove its reads of the source retired.
        std::mem::forget(weight);
        return packed;
    }
    // SAFETY: the pack drained the stream, which has no other users.
    unsafe { library.cuda_stream_destroy(stream)?; }
    drop(weight);
    packed
}

/// A vocabulary projection through the head copy when the caller holds scratch
/// for it (see [`fp8_head`]), else the BF16 head. A packed copy is read exactly,
/// or through its coarse plane at the drafter's site.
///
/// # Safety
/// As [`cuteafd_ffi::V41VocabularyProjection::launch`]; `scratch` was sized by
/// [`fp8_scratch`] for at least `rows` rows of this head.
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn project_vocabulary(library: &NativeLibrary, projection: Option<&cuteafd_ffi::V41VocabularyProjection<'_>>,
    bf16: Option<CuteafdDeviceBuffer>, copy: Option<(HeadCopy<'_, '_>, &HeadScratch<'_>)>,
    input: CuteafdDeviceBuffer, logits: CuteafdDeviceBuffer, rows: usize, stream: *mut std::ffi::c_void) -> Result<()> {
    let Some((copy, scratch)) = copy else {
        return unsafe { projection.context("BF16 head projection not constructed")?.launch(input, bf16.context("BF16 vocabulary is not resident; head copy scratch required")?,
            logits, rows, stream) };
    };
    ensure!(input.bytes >= rows * 10240 && logits.bytes >= rows * copy.shape().0 * 4
        && input.device_id == logits.device_id, "vocabulary head copy projection buffers differ");
    match copy {
        HeadCopy::Fp8(weight) => unsafe {
            weight.apply(library, input.ptr, logits.ptr, true, rows, 0, weight.n, &scratch.allocation, stream)
        },
        HeadCopy::Packed(head) => unsafe {
            head.apply(library, input.ptr, logits.ptr, true, rows, scratch.coarse, scratch.allocation.buffer, stream)
        },
    }
}

/// [`project_vocabulary`] for target logits: the BF16 head takes the shared target-head route
/// ([`crate::shared::target_head`], tensor cores by default); head copies are unchanged.
///
/// # Safety
/// As [`project_vocabulary`].
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn project_target_vocabulary(library: &NativeLibrary,
    projection: Option<&cuteafd_ffi::V41VocabularyProjection<'_>>, bf16: Option<CuteafdDeviceBuffer>,
    copy: Option<(HeadCopy<'_, '_>, &HeadScratch<'_>)>, input: CuteafdDeviceBuffer,
    logits: CuteafdDeviceBuffer, rows: usize, stream: *mut std::ffi::c_void) -> Result<()> {
    if copy.is_none() && crate::shared::target_head::TargetHead::configured().tensor_cores() {
        let weight = bf16.context("BF16 vocabulary is not resident; head copy scratch required")?;
        return unsafe { projection.context("BF16 target projection not constructed")?.launch_tensor_invariant(input, weight, logits, rows, stream) };
    }
    unsafe { project_vocabulary(library, projection, bf16, copy, input, logits, rows, stream) }
}

/// Whether `site` (`Fp8Head::All` for target logits, `Fp8Head::Draft` for the
/// dSpark drafter) projects through the resident copy under `mode`, and if so
/// whether it reads a packed head's coarse plane (the drafter does).
fn copy_site(mode: Fp8Head, site: Fp8Head) -> Option<bool> {
    let enabled = match mode {
        Fp8Head::Off => false,
        Fp8Head::Draft => site == Fp8Head::Draft,
        Fp8Head::All | Fp8Head::Packed => true,
    };
    enabled.then_some(site == Fp8Head::Draft)
}

/// GEMV scratch for `rows` rows of a head copy, when `site` uses one (see [`copy_site`]).
pub(crate) fn fp8_scratch<'a>(library: &'a NativeLibrary, copy: Option<HeadCopy<'_, '_>>,
    rows: usize, site: Fp8Head) -> Result<Option<HeadScratch<'a>>> {
    let (Some(copy), Some(coarse)) = (copy, copy_site(fp8_head(), site)) else { return Ok(None) };
    let (n, k) = copy.shape();
    let allocation = match copy {
        HeadCopy::Fp8(_) => crate::shared::fp8_linear::scratch(library, rows, &[(k, n)])?,
        HeadCopy::Packed(_) => crate::shared::packed_head::scratch(library, rows, &[(k, n)],
            if coarse { Tag::drafter_workspace("v41/draft-head-projection") }
            else { Tag::workspace("v41/target-head-projection") })?,
    };
    Ok(Some(HeadScratch { allocation, coarse }))
}

#[cfg(test)]
mod head_site_tests {
    use super::*;

    #[test]
    fn planner_and_serve_publish_the_same_resolved_head() {
        for request in ["", "auto", "off", "draft", "all", "packed"] {
            let planned = Fp8Head::parse(request);
            let (served, settings) = cuteafd_bench::context::capture_resolved(|| publish_head(planned));
            assert_eq!(served, planned);
            assert_eq!(settings.0, [("v41-fp8-head".into(), planned.setting().into())]);
            settings.assert_resolved();
            for ranks in [1, 2] {
                let source = 1_323_827_200 / ranks;
                assert_eq!(served.resident(source), planned.resident(source));
                if request.is_empty() || request == "auto" {
                    assert_eq!(served, Fp8Head::Packed);
                    assert_eq!(copy_site(served, Fp8Head::All), Some(false));
                    assert_eq!(copy_site(served, Fp8Head::Draft), Some(true));
                }
            }
        }
    }

    /// `packed` sends both sites through the exact 12-bit copy (BF16 is gone):
    /// target logits exactly, the drafter through the coarse plane. `draft`
    /// keeps target logits on BF16; `off` never uses a copy.
    #[test]
    fn packed_reads_exact_target_and_coarse_draft_sites() {
        let (target, draft) = (Fp8Head::All, Fp8Head::Draft);
        assert_eq!([copy_site(Fp8Head::Packed, target), copy_site(Fp8Head::Packed, draft)], [Some(false), Some(true)]);
        assert_eq!([copy_site(Fp8Head::All, target), copy_site(Fp8Head::All, draft)], [Some(false), Some(true)]);
        assert_eq!([copy_site(Fp8Head::Draft, target), copy_site(Fp8Head::Draft, draft)], [None, Some(true)]);
        assert_eq!([copy_site(Fp8Head::Off, target), copy_site(Fp8Head::Off, draft)], [None, None]);
        // Every mode whose BF16 head is released serves both sites from its copy.
        for mode in [Fp8Head::All, Fp8Head::Packed] {
            assert!(mode.target() && mode.draft());
            assert!(copy_site(mode, target).is_some() && copy_site(mode, draft).is_some());
        }
    }
}
impl<'library> VocabularyHead<'library> {
    pub(crate) fn inventory(&self, inventory: &mut crate::shared::memory::receipts::Inventory<'_>) -> Result<()> {
        self.shard.inventory(inventory)
    }
    /// Peak load admission, including BF16 staging and FP8 destinations.
    pub fn plan(catalog: &OfficialV41Catalog) -> Result<usize> {
        VocabularyShard::load_bytes(catalog, 0..129280)
    }
    pub fn resident_bytes(catalog: &OfficialV41Catalog) -> Result<usize> {
        let source = VocabularyShard::device_bytes(catalog, 0..129280)?;
        Ok(usize::try_from(fp8_head().resident(source as u64).0)?)
    }
    #[cfg(test)]
    pub fn load(
        library: &'library NativeLibrary,
        catalog: &OfficialV41Catalog,
        budget: usize,
        staging_bytes: usize,
    ) -> Result<Self> {
        Self::load_in(library, catalog, budget, staging_bytes, None)
    }
    pub fn load_in(library: &'library NativeLibrary, catalog: &OfficialV41Catalog,
        budget: usize, staging_bytes: usize, arena: Option<&DeviceWeightArena<'library>>) -> Result<Self> {
        Ok(Self { shard: VocabularyShard::load_in(library, catalog, 0..129280, budget, staging_bytes, arena)? })
    }
    pub fn finish_push(&self) -> Result<()> { self.shard.finish_push() }
    pub fn weight(&self) -> Option<CuteafdDeviceBuffer> { self.shard.weight() }
    pub fn device_id(&self) -> i32 { self.shard.device_id() }
    pub fn copy(&self) -> Option<HeadCopy<'_, 'library>> { self.shard.copy() }
}

#[cfg(test)]
impl<'a> NativeRtxTensors<'a> {
    pub(crate) fn hc_fixture(library: &'a NativeLibrary) -> Result<Self> {
        let specs = [("projection", 24 * 20480 * 4), ("scale", 12), ("base", 96), ("norm", 10240)];
        let layout = cuteafd_loader::serving_capacity::WeightArenaLayout::new("v41-hc-fixture",
            specs.iter().map(|&(name, bytes)| (name.to_owned(), bytes)))?;
        let arena = DeviceWeightArena::new(Device { library, id: library.cuda_get_device()? }, layout,
            ArenaTag::weights("v41/hc-fixture"))?;
        let tensors = specs.into_iter().map(|(name, bytes)|
            Ok((name.to_owned(), DeviceAllocation::from_allocation(arena.view(name, bytes)?))))
            .collect::<Result<_>>()?;
        Ok(Self { tensors, resident_bytes: arena.layout().bytes(), mapped_embedding: None })
    }
    pub(crate) fn compressor_fixture(library: &'a NativeLibrary, layer: usize, ratio: usize) -> Result<Self> {
        let mut specs = vec![
            (format!("layers.{layer}.attn.compressor.wkv.weight"), 512 * 10240),
            (format!("layers.{layer}.attn.compressor.norm.weight"), 512 * 2),
        ];
        if ratio == 2 {
            specs.push((format!("layers.{layer}.attn.compressor.wgate.weight"), 512 * 10240));
        }
        specs.push((format!("layers.{layer}.attn.indexer.wk.weight"), 128 * 512 * 2));
        specs.push((format!("layers.{layer}.attn.indexer.k_norm.weight"), 128 * 2));
        let layout = cuteafd_loader::serving_capacity::WeightArenaLayout::new("v41-compressor-fixture",
            specs.iter().map(|(name, bytes)| (name.clone(), *bytes)))?;
        let arena = DeviceWeightArena::new(Device { library, id: library.cuda_get_device()? }, layout,
            ArenaTag::weights("v41/compressor-fixture"))?;
        let tensors = specs.into_iter().map(|(name, bytes)| {
            let view = arena.view(&name, bytes)?;
            Ok((name, DeviceAllocation::from_allocation(view)))
        }).collect::<Result<_>>()?;
        Ok(Self { tensors, resident_bytes: arena.layout().bytes(), mapped_embedding: None })
    }
    /// Every resident tensor (golden digests).
    pub(crate) fn visit(&self, prefix: &str, f: &mut dyn FnMut(String, cuteafd_ffi::CuteafdDeviceBuffer)) {
        for (name, allocation) in &self.tensors { f(format!("{prefix}{name}"), allocation.buffer); }
        if self.mapped_embedding.is_some() {
            f(format!("{prefix}embed.weight"), self.get("embed.weight").expect("committed host embedding"));
        }
    }
}

#[cfg(test)]
mod host_embedding_tests {
    use super::*;

    #[test]
    fn mapped_embedding_alias_is_unavailable_until_committed() -> Result<()> {
        use cuteafd_ffi::native_library_lifetime_fixture::Fixture;
        let fixture = Fixture::build_for_tests()?;
        let library = fixture.load()?;
        fixture.configure_memory(&library, 0, 1 << 30)?;
        let mut owner = HostAllocation::new(&library, 30, Tag::embedding("v41/embedding"))?;
        let landing = crate::shared::memory::HostLanding::new(&mut owner);
        let alias = library.cuda_host_buffer_device_alias(owner.buffer)?;
        let table = NativeRtxTensors { tensors: BTreeMap::new(), resident_bytes: 0,
            mapped_embedding: Some((owner, alias, landing.clone())) };
        assert_eq!(table.embedding_layout()?, (30, alias.device_id));
        assert!(table.get("embed.weight").unwrap_err().to_string().contains("not committed"));
        landing.ready.store(true, std::sync::atomic::Ordering::Release);
        assert_eq!(table.get("embed.weight")?.ptr, alias.ptr);
        assert_eq!(table.resident_bytes(), 0);
        Ok(())
    }
}

#[cfg(test)]
impl VocabularyHead<'_> {
    pub(crate) fn visit(&self, prefix: &str, f: &mut dyn FnMut(String, cuteafd_ffi::CuteafdDeviceBuffer)) {
        self.shard.visit(prefix, f);
    }
}
