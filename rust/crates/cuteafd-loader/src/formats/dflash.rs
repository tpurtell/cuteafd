//! Checkpoint-native MLA DFlash2 geometry shared by admission and loading.
use super::{block_dspark::{DsparkConfigError, DsparkRope, DsparkRotaryPlan}, mla::MlaGeometry};
use serde::Deserialize;
use cuteafd_core::MlaCacheFormat;
use std::path::Path;

type Result<T> = std::result::Result<T, DsparkConfigError>;

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
#[error("DFlash2 top-k vocabulary {vocab} outside native int32 range 16..=2147483647")]
pub struct DflashTopkVocabError { pub vocab: u64 }

/// The shared CUDA selector emits 16 distinct int32 token indices. Chunk
/// coverage is unbounded; only the output count and signed-index ABI remain.
pub fn validate_topk_vocab(vocab: u64) -> std::result::Result<(), DflashTopkVocabError> {
    if !(16..=i32::MAX as u64).contains(&vocab) { return Err(DflashTopkVocabError { vocab }); }
    Ok(())
}

#[derive(Debug, Clone, PartialEq)]
pub struct MlaDflashConfig {
    pub mla: MlaGeometry,
    pub intermediate: usize,
    pub eps: f64,
    pub block: usize,
    pub group: usize,
    pub selector_rank: usize,
    pub selector_top_k: usize,
    pub taps: Vec<usize>,
    pub vocab: usize,
    pub mask_token: u32,
    pub max_context: usize,
    /// None means full context; each sliding layer keeps its own exact window.
    pub windows: Vec<Option<usize>>,
    pub rotary: DsparkRotaryPlan,
    /// The native program set: DFlash2 (convolutions, top-16 selector) or
    /// Mistral Large 4's mtp_dspark (plain layers, RNN Markov head of
    /// `markov_rank`, position-specific mask rows, block-anchored windows).
    pub kind: cuteafd_core::coordinator_programs::MlaDraftKind,
    pub markov_rank: usize,
}

/// Named immutable allocation recipe. Admission and device storage consume the
/// same byte counts; embedding, mask row and vocabulary weights belong to target.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MlaDraftBuffer {
    pub name: String,
    pub bytes: usize,
}

/// Placement-independent owner facts. Device ordinals are bound by the caller;
/// library-scoped pinned stores use a sharing key, not another private charge.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MlaDraftDomain { Device, HostPinned }
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MlaDraftLifetime { Runtime, Library }
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MlaDraftDtype { Bytes, Bf16, F32, I32, U32, I64 }
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MlaDraftOwner {
    pub id: String,
    pub domain: MlaDraftDomain,
    pub bytes: usize,
    pub alignment: usize,
    pub dtype: MlaDraftDtype,
    pub shape: Vec<usize>,
    pub lifetime: MlaDraftLifetime,
    pub shared_key: Option<String>,
    pub staging_for: Option<String>,
    /// Diagnostic host-row/tap inputs create staging lazily; normal device input does not.
    pub on_use: bool,
}
impl MlaDraftOwner {
    pub fn new(id: impl Into<String>, domain: MlaDraftDomain, dtype: MlaDraftDtype,
        shape: Vec<usize>, minimum_bytes: usize) -> Result<Self> {
        let width = match dtype { MlaDraftDtype::Bytes => 1, MlaDraftDtype::Bf16 => 2,
            MlaDraftDtype::F32 | MlaDraftDtype::I32 | MlaDraftDtype::U32 => 4,
            MlaDraftDtype::I64 => 8 };
        if shape.is_empty() || shape.contains(&0) { return Err(unsupported("draft owner has an empty shape")); }
        let bytes = shape.iter().try_fold(width, |n: usize, &d| n.checked_mul(d))
            .ok_or_else(|| unsupported("draft owner byte extent overflow"))?.max(minimum_bytes);
        Ok(Self { id: id.into(), domain, dtype, shape, bytes, alignment: 1,
            lifetime: MlaDraftLifetime::Runtime, shared_key: None, staging_for: None, on_use: false })
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MlaDraftView {
    pub id: String,
    pub backing: String,
    pub offset: usize,
    pub bytes: usize,
}
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MlaDraftPacking { Direct, Uk, Uv, GateUp, UnitScale }
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MlaDraftWeight {
    pub owner: String,
    pub shape: Vec<usize>,
    pub sources: Vec<(String, Vec<usize>)>,
    pub packing: MlaDraftPacking,
}

impl MlaDraftWeight {
    /// Materialize only this destination; source headers remain in checkpoint order.
    pub fn pack(&self, read: &mut impl FnMut(&str, &[usize]) -> anyhow::Result<Vec<u8>>)
        -> anyhow::Result<Vec<u8>> {
        let source = |index: usize, read: &mut dyn FnMut(&str, &[usize]) -> anyhow::Result<Vec<u8>>| {
            let (name, shape) = &self.sources[index];
            let bytes = read(name, shape)?;
            let expected = shape.iter().try_fold(2usize, |n, &d| n.checked_mul(d))
                .ok_or_else(|| anyhow::anyhow!("draft source extent overflow"))?;
            anyhow::ensure!(bytes.len() == expected, "draft source {name} byte extent differs from header");
            Ok::<_, anyhow::Error>(bytes)
        };
        let bytes = match self.packing {
            MlaDraftPacking::Direct => source(0, read)?,
            MlaDraftPacking::GateUp => {
                let mut bytes = source(0, read)?;
                bytes.extend(source(1, read)?);
                bytes
            }
            MlaDraftPacking::UnitScale => {
                let heads = self.shape[1];
                let mut bytes = vec![0u8; heads * 2];
                bytes.extend(std::iter::repeat_n(0x3f80u16.to_le_bytes(), heads).flatten());
                bytes
            }
            MlaDraftPacking::Uk | MlaDraftPacking::Uv => {
                let kv = source(0, read)?;
                let (heads, latent) = (self.shape[0], self.sources[0].1[1]);
                let head_rows = self.sources[0].1[0] / heads;
                let mut bytes = vec![0u8; self.shape.iter().product::<usize>() * 2];
                match self.packing {
                    MlaDraftPacking::Uk => {
                        let nope = self.shape[2];
                        for head in 0..heads { for d in 0..nope { for col in 0..latent {
                            let src = ((head * head_rows + d) * latent + col) * 2;
                            let dst = ((head * latent + col) * nope + d) * 2;
                            bytes[dst..dst + 2].copy_from_slice(&kv[src..src + 2]);
                        } } }
                    }
                    MlaDraftPacking::Uv => {
                        let value = self.shape[1];
                        let nope = head_rows - value;
                        for head in 0..heads {
                            let src = (head * head_rows + nope) * latent * 2;
                            let dst = head * value * latent * 2;
                            bytes[dst..dst + value * latent * 2].copy_from_slice(&kv[src..src + value * latent * 2]);
                        }
                    }
                    _ => unreachable!(),
                }
                bytes
            }
        };
        anyhow::ensure!(bytes.len() == self.shape.iter().product::<usize>() * 2,
            "draft packed destination {} has wrong extent", self.owner);
        Ok(bytes)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MlaDraftRuntimeLayout {
    kind: cuteafd_core::coordinator_programs::MlaDraftKind,
    cache_format: MlaCacheFormat,
    target_hidden: usize,
    target_vocab: usize,
    target_taps: Vec<usize>,
    capacity: crate::families::glm5::draft_representation::GlmDraftCapacity,
    weights: crate::families::glm5::draft_representation::GlmDraftWeightLayout,
    cache: MlaDraftCacheLayout,
    paged_cache: DraftCacheLayout,
    program: cuteafd_core::coordinator_programs::MlaDraftProgramLayout,
    fp8_scratch: Option<crate::families::glm5::draft_representation::GlmDraftFp8ScratchLayout>,
    sm_count: usize,
    fp8_mode: u8,
    buffers: Vec<MlaDraftBuffer>,
    owners: Vec<MlaDraftOwner>,
    packing_sources: Vec<MlaDraftOwner>,
    sync_upload_bytes: usize,
    weight_destinations: Vec<MlaDraftWeight>,
}

/// Only DFlash2 uses the native top-16 selector; RNN-Markov has no selector owner.
pub fn draft_selector_workspace_bytes(kind: cuteafd_core::coordinator_programs::MlaDraftKind,
    drafted: usize) -> Result<usize> {
    use cuteafd_core::coordinator_programs::MlaDraftKind;
    match kind {
        MlaDraftKind::Dflash2 => drafted.checked_mul(64 * 16 * 8)
            .map(|bytes| bytes.max(256)).ok_or_else(|| unsupported("draft selector workspace overflow")),
        MlaDraftKind::Ml4Dspark => Ok(0),
    }
}

impl MlaDraftRuntimeLayout {
    pub const TAP_ROWS: usize = 2048;

    pub fn kind(&self) -> cuteafd_core::coordinator_programs::MlaDraftKind { self.kind }
    pub fn selector_workspace_bytes(&self) -> Result<usize> {
        draft_selector_workspace_bytes(self.kind, self.capacity.block_rows - self.capacity.max_batch_sequences)
    }
    pub fn cache_format(&self) -> MlaCacheFormat { self.cache_format }
    pub fn target_hidden(&self) -> usize { self.target_hidden }
    pub fn target_vocab(&self) -> usize { self.target_vocab }
    pub fn target_taps(&self) -> &[usize] { &self.target_taps }
    pub fn buffers(&self) -> &[MlaDraftBuffer] { &self.buffers }
    pub fn owners(&self) -> &[MlaDraftOwner] { &self.owners }
    pub fn packing_sources(&self) -> &[MlaDraftOwner] { &self.packing_sources }
    pub fn sync_upload_bytes(&self) -> usize { self.sync_upload_bytes }
    /// The loader retains the whole safetensors file. Native MLA reordering also
    /// materializes a source and an output; Vec growth is a labelled heap allowance.
    pub fn loading_host_bytes(&self, checkpoint_file_bytes: u64) -> Result<u64> {
        let upload = (self.sync_upload_bytes as u64).checked_mul(4)
            .ok_or_else(|| unsupported("MLA draft loading heap overflow"))?;
        checkpoint_file_bytes.checked_add(upload)
            .ok_or_else(|| unsupported("MLA draft loading heap overflow"))
    }
    pub fn weight_destinations(&self) -> &[MlaDraftWeight] { &self.weight_destinations }
    pub fn device_buffers(&self) -> Vec<MlaDraftBuffer> {
        if self.kind != cuteafd_core::coordinator_programs::MlaDraftKind::Ml4Dspark {
            return self.buffers.clone();
        }
        self.owners.iter().filter(|o| o.domain == MlaDraftDomain::Device)
            .map(|o| MlaDraftBuffer { name: o.id.clone(), bytes: o.bytes }).collect()
    }
    pub fn allocate_device_with<T, E>(&self, mut allocate: impl FnMut(&MlaDraftBuffer) -> std::result::Result<T, E>)
        -> std::result::Result<Vec<T>, E> {
        self.device_buffers().iter().map(&mut allocate).collect()
    }
    pub fn host_owner(&self, id: &str) -> Result<&MlaDraftOwner> {
        self.owners.iter().find(|o| o.id == id && o.domain == MlaDraftDomain::HostPinned)
            .ok_or_else(|| unsupported("MLA draft unknown host owner"))
    }
    pub fn view(&self, id: impl Into<String>, backing: &str, offset: usize, bytes: usize) -> Result<MlaDraftView> {
        let extent = self.owners.iter().find(|o| o.id == backing).map(|o| o.bytes)
            .or_else(|| self.buffers.iter().find(|o| o.name == backing).map(|o| o.bytes))
            .ok_or_else(|| unsupported("MLA draft view backing missing"))?;
        if !offset.checked_add(bytes).is_some_and(|end| end <= extent) {
            return Err(unsupported("MLA draft view exceeds owner recipe"));
        }
        Ok(MlaDraftView { id: id.into(), backing: backing.into(), offset, bytes })
    }
    /// Preflight and launch consume this same body-selected input/output pair.
    pub fn block_views(&self, first_sequence: usize, rows: usize) -> Result<[MlaDraftView; 2]> {
        let row = self.target_hidden.checked_mul(2).ok_or_else(|| unsupported("draft row overflow"))?;
        let block = self.capacity.block_rows / self.capacity.max_batch_sequences;
        let offset = first_sequence.checked_mul(block).and_then(|n| n.checked_mul(row))
            .ok_or_else(|| unsupported("draft block offset overflow"))?;
        let bytes = rows.checked_mul(row).ok_or_else(|| unsupported("draft block extent overflow"))?;
        let input = if self.kind == cuteafd_core::coordinator_programs::MlaDraftKind::Ml4Dspark { "n" } else { "conv" };
        Ok([self.view("block.input", input, offset, bytes)?, self.view("block.output", "delta", offset, bytes)?])
    }
    pub fn mask_view(&self, sequence: usize) -> Result<MlaDraftView> {
        let block = self.capacity.block_rows / self.capacity.max_batch_sequences;
        let row = self.target_hidden * 2;
        self.view("block.mask", "h", (sequence * block + 1) * row, (block - 1) * row)
    }
    pub fn window_view(&self, layer: usize, slot: usize) -> Result<MlaDraftView> {
        let cache = self.cache.layers.get(layer).ok_or_else(|| unsupported("draft window layer missing"))?;
        if cache.window.is_none() || slot >= self.capacity.context_slots {
            return Err(unsupported("draft window slot outside capacity"));
        }
        let bytes = cache.rows_per_slot * self.paged_cache.record_bytes;
        self.view("cache.window", &format!("layer.{layer}.cache"), slot * bytes, bytes)
    }
    pub fn result_views(&self, sequences: usize) -> Result<[MlaDraftView; 2]> {
        if sequences > self.capacity.max_batch_sequences {
            return Err(unsupported("draft result count outside capacity"));
        }
        let block = self.capacity.block_rows / self.capacity.max_batch_sequences;
        let drafts = if self.kind == cuteafd_core::coordinator_programs::MlaDraftKind::Ml4Dspark { block } else { block - 1 };
        let bytes = sequences * drafts * 4;
        Ok([self.view("result.tokens", "tokens", 0, bytes)?, self.view("result.landing", "host.landing", 0, bytes)?])
    }
    pub fn context_views(&self, first_row: usize, rows: usize) -> Result<[MlaDraftView; 3]> {
        if rows == 0 || rows > 64 { return Err(unsupported("draft context chunk outside m64")); }
        let bytes = rows.checked_mul(self.target_hidden).and_then(|n| n.checked_mul(2))
            .ok_or_else(|| unsupported("draft context extent overflow"))?;
        let offset = first_row.checked_mul(self.target_hidden).and_then(|n| n.checked_mul(2))
            .ok_or_else(|| unsupported("draft context offset overflow"))?;
        let metadata = first_row.checked_mul(8).ok_or_else(|| unsupported("draft metadata offset overflow"))?;
        Ok([self.view("context.input", "fused_norm", offset, bytes)?,
            self.view("context.slots", "context_slots", metadata, rows * 8)?,
            self.view("context.positions", "context_positions", metadata, rows * 8)?])
    }
    pub fn capacity(&self) -> crate::families::glm5::draft_representation::GlmDraftCapacity { self.capacity }
    pub fn weights(&self) -> crate::families::glm5::draft_representation::GlmDraftWeightLayout { self.weights }
    pub fn cache(&self) -> &MlaDraftCacheLayout { &self.cache }
    pub fn paged_cache(&self) -> &DraftCacheLayout { &self.paged_cache }
    pub fn program(&self) -> cuteafd_core::coordinator_programs::MlaDraftProgramLayout { self.program }
    pub fn fp8_scratch(&self) -> Option<&crate::families::glm5::draft_representation::GlmDraftFp8ScratchLayout> {
        self.fp8_scratch.as_ref()
    }
    pub fn sm_count(&self) -> usize { self.sm_count }
    pub fn fp8_mode(&self) -> u8 { self.fp8_mode }

    /// Rebind only the aggregate full-context pool; every native shape and bounded
    /// ring remains identical to the checked checkpoint-derived layout.
    pub fn with_pool_tokens(&self, target_tokens: usize) -> Result<Self> {
        let pages = target_tokens.div_ceil(64);
        if target_tokens == 0 || pages > i32::MAX as usize {
            return Err(unsupported("draft pool token budget outside native page limits"));
        }
        let bytes = pages.checked_mul(64).and_then(|n| n.checked_mul(self.paged_cache.record_bytes))
            .ok_or_else(|| unsupported("draft pool byte overflow"))?;
        let mut next = self.clone();
        next.paged_cache.target_tokens = target_tokens;
        next.paged_cache.pages = pages;
        next.paged_cache.full_bytes = (bytes as u64).checked_mul(next.paged_cache.full_layers as u64)
            .ok_or_else(|| unsupported("draft pool layer total overflow"))?;
        for (index, layer) in next.cache.layers.iter().enumerate() {
            if layer.window().is_none() {
                next.paged_cache.layer_bytes[index] = bytes;
                let name = format!("layer.{index}.cache");
                next.buffers.iter_mut().find(|buffer| buffer.name == name)
                    .ok_or_else(|| unsupported("draft pool allocation recipe missing layer"))?.bytes = bytes.max(256);
                let owner = next.owners.iter_mut().find(|owner| owner.id == name)
                    .ok_or_else(|| unsupported("draft pool owner recipe missing layer"))?;
                owner.bytes = bytes.max(256);
                owner.shape = vec![owner.bytes];
            }
        }
        next.resident_bytes()?;
        Ok(next)
    }

    pub fn window_mark_bytes(&self) -> Result<usize> {
        usize::try_from(self.paged_cache.window_bytes / self.capacity.context_slots as u64)
            .map_err(|_| unsupported("draft window mark overflow"))
    }

    pub fn buffer_bytes(&self) -> Result<u64> {
        self.buffers.iter().try_fold(0u64, |sum, b| sum.checked_add(b.bytes as u64)
            .ok_or_else(|| unsupported("MLA draft buffer total overflow")))
    }

    pub fn resident_bytes(&self) -> Result<u64> {
        if self.kind == cuteafd_core::coordinator_programs::MlaDraftKind::Ml4Dspark {
            return self.buffer_bytes()?.checked_add(self.weights.resident_bytes()
                .map_err(|_| unsupported("MLA draft weight total overflow"))?)
                .ok_or_else(|| unsupported("MLA draft runtime total overflow"));
        }
        self.owners.iter().filter(|owner| owner.domain == MlaDraftDomain::Device && !owner.on_use)
            .try_fold(0u64, |sum, owner| sum.checked_add(owner.bytes as u64)
                .ok_or_else(|| unsupported("MLA draft runtime total overflow")))
    }

    /// Allocation never derives another size or silently rounds beyond admission.
    /// Call only after the solver admits this immutable layout on the target GPU.
    pub fn allocate_with<T, E>(&self, mut allocate: impl FnMut(&MlaDraftBuffer) -> std::result::Result<T, E>)
        -> std::result::Result<Vec<T>, E> {
        self.buffers.iter().map(&mut allocate).collect()
    }

    /// SM-local scratch, bounded rings and the shared full-layer pool stay on the target's
    /// home. Packing is a separate startup peak, never a second resident copy.
    pub fn demands(&self, gpu: u8) -> Result<(crate::placement::draft::DrafterDemand, crate::placement::Demand)> {
        use crate::placement::{draft::DrafterDemand, Demand};
        use cuteafd_core::memory_layout::{Basis, Category};
        let owned = self.owners.iter()
            .filter(|owner| owner.domain == MlaDraftDomain::Device && !owner.on_use)
            .map(|owner| Demand::new(gpu, Category::Drafter,
                format!("MLA DFlash {}", owner.id), owner.bytes as u64, Basis::Formula)).collect();
        let staging = self.weights.max_load_staging;
        let peak = self.resident_bytes()?.checked_add(staging)
            .ok_or_else(|| unsupported("MLA draft startup peak overflow"))?;
        Ok((DrafterDemand::itemized(owned, gpu), Demand::new(gpu, Category::Drafter,
            "MLA DFlash load peak including transient packing", peak, Basis::Formula)))
    }
}

/// One layer's slot-isolated physical cache and page-table capacity. The
/// sliding ring includes an admitted private block; committed rows stay intact
/// until verification decides which rows to retain.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MlaDraftCacheLayer {
    slots: usize,
    context: usize,
    block: usize,
    page_rows: usize,
    window: Option<usize>,
    rows_per_slot: usize,
    table_width: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MlaDraftCacheLayout {
    pub layers: Vec<MlaDraftCacheLayer>,
    pub cache_bytes: u64,
    pub page_table_bytes: u64,
    /// Physical padding and private block rows above retained-record capacity.
    pub margin_bytes: u64,
}

/// Immutable full-layer pool and bounded-window inventory. The target's aggregate
/// token budget is independent of both request count and checkpoint context extent.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DraftCacheLayout {
    pub target_tokens: usize,
    pub page_rows: usize,
    pub pages: usize,
    pub record_bytes: usize,
    pub full_layers: usize,
    pub window_bytes: u64,
    pub full_bytes: u64,
    pub page_table_bytes: u64,
    pub layer_bytes: Vec<usize>,
}

impl DraftCacheLayout {
    pub fn cache_bytes(&self) -> Result<u64> {
        self.full_bytes.checked_add(self.window_bytes)
            .ok_or_else(|| unsupported("draft pool total overflow"))
    }

    /// Owner balancing moves the same inventory runtime allocates, not a per-slot
    /// estimate. The one chosen owner holds every full-layer segment of a page ID.
    pub fn demands(&self, gpu: u8) -> Result<Vec<crate::placement::Demand>> {
        use crate::placement::Demand;
        use cuteafd_core::memory_layout::{Basis, Category};
        Ok(vec![Demand::new(gpu, Category::Drafter, "draft shared full-context KV pool",
                self.full_bytes, Basis::Formula),
            Demand::new(gpu, Category::Drafter, "draft bounded window rings",
                self.window_bytes, Basis::Formula),
            Demand::new(gpu, Category::Drafter, "draft page tables",
                self.page_table_bytes, Basis::Formula)])
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MlaDraftCacheView {
    /// Absolute start, page-aligned relative to the slot's storage origin.
    /// Kernel lengths/queries are relative to this start; RoPE stays absolute.
    pub base_position: usize,
    pub cache_seqlen: usize,
    pub pages: Vec<i32>,
}

/// One block uses absolute positions and the slot's immutable cold-resume origin.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MlaDraftCacheRequest {
    pub slot: usize,
    pub position: usize,
    pub valid_from: usize,
}

/// Checked host metadata for one native m64 launch. Sequence order is preserved
/// across chunks; positions drive RoPE while slots address origin-relative rows.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MlaDraftCacheBatch {
    pub first_sequence: usize,
    pub positions: Vec<i64>,
    pub slots: Vec<i64>,
    pub page_table: Vec<i32>,
    pub page_table_width: usize,
    pub lengths: Vec<i32>,
    pub query_offsets: Vec<i32>,
}

/// A cold continuation binds one immutable storage origin to one slot. Rebind
/// after slot reuse or prefix restore; never move the origin of retained rows.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MlaDraftCacheSlot {
    layer: MlaDraftCacheLayer,
    slot: usize,
    valid_from: usize,
}

/// Logical liveness of the retained MLA records. Prefix restores are cold:
/// rebinding excludes every byte from the previous occupant without clearing KV.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MlaDraftCacheState {
    layout: MlaDraftCacheLayout,
    slots: Vec<Option<MlaDraftFrontier>>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct MlaDraftFrontier {
    origin: usize,
    end: usize,
}

impl MlaDraftCacheState {
    pub fn layer(&self, index: usize) -> Result<&MlaDraftCacheLayer> {
        self.layout.layers.get(index).ok_or_else(|| unsupported("MLA draft unknown cache layer"))
    }

    pub fn new(layout: MlaDraftCacheLayout) -> Result<Self> {
        let first = layout.layers.first().ok_or_else(|| unsupported("MLA draft has no cache layers"))?;
        if layout.layers.iter().any(|layer| layer.slots != first.slots || layer.context != first.context) {
            return Err(unsupported("MLA draft cache layers disagree on slots/context"));
        }
        let slots = vec![None; first.slots];
        Ok(Self { layout, slots })
    }

    /// Both a prefix restore and a newly cold-started drafter use this operation.
    /// Validate every layer before replacing the old slot binding.
    pub fn reset(&mut self, slot: usize, origin: usize) -> Result<()> {
        for layer in &self.layout.layers { layer.bind(slot, origin)?; }
        self.slots[slot] = Some(MlaDraftFrontier { origin, end: origin });
        Ok(())
    }

    /// Bind exact saved metadata after the owner has restored every retained ring row.
    pub fn restore_frontier(&mut self, slot: usize, origin: usize, end: usize) -> Result<()> {
        if end < origin || end > self.layout.layers[0].context {
            return Err(unsupported("MLA draft saved frontier outside context"));
        }
        self.reset(slot, origin)?;
        self.slots[slot] = Some(MlaDraftFrontier { origin, end });
        Ok(())
    }

    pub fn discard(&mut self, slot: usize) -> Result<()> {
        *self.slots.get_mut(slot).ok_or_else(|| unsupported("MLA draft discard outside slots"))? = None;
        Ok(())
    }

    pub fn frontier(&self, slot: usize) -> Result<(usize, usize)> {
        let state = self.slots.get(slot).and_then(Option::as_ref)
            .ok_or_else(|| unsupported("MLA draft slot is not bound"))?;
        Ok((state.origin, state.end))
    }

    /// Validate the entire append before any producer writes; publishing the
    /// frontier is separate and happens only after the ordered context update.
    pub fn context_rows(&self, layer: usize, slot: usize, start: usize, rows: usize) -> Result<Vec<i64>> {
        self.check_append(slot, start, rows)?;
        let (origin, _) = self.frontier(slot)?;
        let finish = start + rows;
        let layer = self.layout.layers.get(layer).ok_or_else(|| unsupported("MLA draft unknown cache layer"))?;
        let bound = layer.bind(slot, origin)?;
        (start..finish).map(|position| i64::try_from(bound.row(position)?)
            .map_err(|_| unsupported("MLA draft physical row exceeds I64"))).collect()
    }

    fn check_append(&self, slot: usize, start: usize, rows: usize) -> Result<()> {
        let (_, end) = self.frontier(slot)?;
        let finish = start.checked_add(rows).ok_or_else(|| unsupported("MLA draft context overflow"))?;
        // A tail-only producer may omit only records no layer can ever read again.
        // Keep the immutable storage origin so its physical rows equal a full append.
        let discarded_gap = start > end && self.layout.layers.iter()
            .all(|layer| layer.window.is_some_and(|window| rows >= window));
        if (start != end && !discarded_gap) || rows == 0 || finish > self.layout.layers[0].context {
            return Err(unsupported("MLA draft context append differs from committed frontier"));
        }
        Ok(())
    }

    /// Physical rows outside the committed window, including private proposal slack.
    /// Zero these at commit/rollback boundaries so whole-owner snapshots are canonical.
    pub fn window_slack(&self, layer: usize, slot: usize) -> Result<Vec<std::ops::Range<usize>>> {
        let (origin, end) = self.frontier(slot)?;
        let layer = self.layout.layers.get(layer).ok_or_else(|| unsupported("MLA draft unknown cache layer"))?;
        let Some(window) = layer.window else { return Ok(Vec::new()); };
        let capacity = layer.rows_per_slot;
        let base = slot * capacity;
        let retained = (end - origin).min(window);
        if retained == 0 { return Ok(vec![base..base + capacity]); }
        let first = (end - origin - retained) % capacity;
        let finish = first + retained;
        let mut slack = Vec::new();
        if finish <= capacity {
            if first > 0 { slack.push(base..base + first); }
            if finish < capacity { slack.push(base + finish..base + capacity); }
        } else {
            slack.push(base + finish - capacity..base + first);
        }
        Ok(slack)
    }

    pub fn commit_context(&mut self, slot: usize, start: usize, rows: usize) -> Result<()> {
        self.check_append(slot, start, rows)?;
        self.slots[slot].as_mut().unwrap().end = start + rows;
        Ok(())
    }

    /// A proposed block cannot move a slot's storage origin or skip committed
    /// context. Rejected blocks leave this logical state unchanged.
    pub fn block_batches(&self, layer: usize, requests: &[MlaDraftCacheRequest]) -> Result<Vec<MlaDraftCacheBatch>> {
        for request in requests {
            if self.frontier(request.slot)? != (request.valid_from, request.position) {
                return Err(unsupported("MLA draft request differs from bound origin/frontier"));
            }
        }
        self.layout.layers.get(layer).ok_or_else(|| unsupported("MLA draft unknown cache layer"))?
            .block_batches(requests)
    }
}

impl MlaDraftCacheLayer {
    pub fn rows_per_slot(&self) -> usize { self.rows_per_slot }
    pub fn table_width(&self) -> usize { self.table_width }
    pub fn slots(&self) -> usize { self.slots }
    pub fn window(&self) -> Option<usize> { self.window }
    pub fn context(&self) -> usize { self.context }
    pub fn block(&self) -> usize { self.block }

    pub fn bind(&self, slot: usize, valid_from: usize) -> Result<MlaDraftCacheSlot> {
        if slot >= self.slots || valid_from >= self.context {
            return Err(unsupported("MLA draft origin outside admitted slot/context"));
        }
        Ok(MlaDraftCacheSlot { layer: self.clone(), slot, valid_from })
    }

    /// Build every chunk before staging any of them, so a bad later sequence
    /// cannot follow a partially enqueued draft. One slot appears at most once.
    pub fn block_batches(&self, requests: &[MlaDraftCacheRequest]) -> Result<Vec<MlaDraftCacheBatch>> {
        use cuteafd_core::coordinator_programs::MlaDraftProgramLayout;
        let per_batch = MlaDraftProgramLayout::ROWS / self.block.max(1);
        if requests.is_empty() || self.block == 0 || per_batch == 0 || requests.len() > self.slots {
            return Err(unsupported("MLA draft batch outside admitted rows/slots"));
        }
        let mut seen = std::collections::BTreeSet::new();
        let views = requests.iter().map(|request| {
            if !seen.insert(request.slot) { return Err(unsupported("MLA draft repeats a slot in one batch")); }
            let bound = self.bind(request.slot, request.valid_from)?;
            let view = bound.block_view(request.position, self.block)?;
            let rows = (request.position..request.position + self.block)
                .map(|position| Ok((position as i64, i64::try_from(bound.row(position)?)
                    .map_err(|_| unsupported("MLA draft physical row exceeds I64"))?)))
                .collect::<Result<Vec<_>>>()?;
            Ok((view, rows))
        }).collect::<Result<Vec<_>>>()?;
        views.chunks(per_batch).enumerate().map(|(chunk, views)| {
            let width = views.iter().map(|(view, _)| view.pages.len()).max().unwrap_or(0);
            let mut batch = MlaDraftCacheBatch { first_sequence: chunk * per_batch,
                positions: Vec::with_capacity(views.len() * self.block),
                slots: Vec::with_capacity(views.len() * self.block),
                page_table: Vec::with_capacity(views.len() * width), page_table_width: width,
                lengths: Vec::with_capacity(views.len()), query_offsets: vec![0] };
            for (view, rows) in views {
                for &(position, slot) in rows { batch.positions.push(position); batch.slots.push(slot); }
                batch.page_table.extend_from_slice(&view.pages);
                batch.page_table.resize(batch.lengths.len() * width + width, 0);
                batch.lengths.push(i32::try_from(view.cache_seqlen)
                    .map_err(|_| unsupported("MLA draft cache length exceeds I32"))?);
                batch.query_offsets.push(batch.positions.len() as i32);
            }
            Ok(batch)
        }).collect()
    }
}

impl MlaDraftCacheSlot {
    pub fn valid_from(&self) -> usize { self.valid_from }

    /// Producer positions remain absolute for RoPE; only storage is rebased.
    pub fn row(&self, position: usize) -> Result<usize> {
        if position < self.valid_from || position >= self.layer.context {
            return Err(unsupported("MLA draft row outside bound origin/context"));
        }
        Ok(self.slot * self.layer.rows_per_slot + (position - self.valid_from) % self.layer.rows_per_slot)
    }

    /// No state is published here. A rejected block can be overwritten without
    /// rolling back the committed ring or changing its live frontier.
    pub fn block_view(&self, anchor: usize, rows: usize) -> Result<MlaDraftCacheView> {
        let layer = &self.layer;
        let end = anchor.checked_add(rows).ok_or_else(|| unsupported("MLA draft block overflow"))?;
        if anchor < self.valid_from || rows == 0 || rows > layer.block || end > layer.context {
            return Err(unsupported("MLA draft block outside bound origin/context"));
        }
        let relative_anchor = anchor - self.valid_from;
        let begin = layer.window.map_or(0, |window| relative_anchor.saturating_sub(window));
        let base = begin / layer.page_rows * layer.page_rows;
        let base_position = self.valid_from + base;
        let pages_per_slot = layer.rows_per_slot / layer.page_rows;
        let first = base / layer.page_rows;
        let count = (end - base_position).div_ceil(layer.page_rows);
        if count > layer.table_width {
            return Err(unsupported("MLA draft view exceeds admitted page table"));
        }
        // Origin-relative pages exclude stale rows before an unaligned cold
        // restore, including full attention (which has no lower-bound mask).
        // Sliding wrapped spans may repeat a page; the window masks its prefix.
        let pages = (first..first + count).map(|page|
            (self.slot * pages_per_slot + page % pages_per_slot) as i32).collect();
        Ok(MlaDraftCacheView { base_position, cache_seqlen: end - base_position, pages })
    }
}

#[derive(Deserialize)]
struct Raw {
    architectures: Vec<String>,
    hidden_size: usize,
    intermediate_size: usize,
    num_hidden_layers: usize,
    num_attention_heads: usize,
    num_key_value_heads: usize,
    q_lora_rank: usize,
    kv_lora_rank: usize,
    qk_nope_head_dim: usize,
    qk_rope_head_dim: usize,
    v_head_dim: usize,
    rms_norm_eps: f64,
    vocab_size: usize,
    draft_vocab_size: usize,
    max_position_embeddings: usize,
    num_target_layers: usize,
    layer_types: Vec<String>,
    sliding_window: Option<usize>,
    use_sliding_window: bool,
    is_causal: bool,
    mla_use_output_gate: bool,
    attention_bias: bool,
    hidden_act: String,
    fc_norm: bool,
    rope_parameters: DsparkRope,
    dflash_config: Draft,
}

#[derive(Deserialize)]
struct Draft {
    attention_mode: String,
    block_size: usize,
    conv_kernel_size: usize,
    conv_group_size: usize,
    selector_rank: usize,
    selector_top_k: usize,
    target_layer_ids: Vec<usize>,
    mask_token_id: u32,
}

fn unsupported(message: &str) -> DsparkConfigError {
    DsparkConfigError::Unsupported(message.into())
}

impl MlaDflashConfig {
    pub fn read(snapshot: &Path) -> Result<Self> {
        Self::from_json(&serde_json::from_slice(&std::fs::read(snapshot.join("config.json"))?)?)
    }

    pub fn from_json(value: &serde_json::Value) -> Result<Self> {
        let raw: Raw = serde_json::from_value(value.clone())?;
        validate_topk_vocab(raw.vocab_size as u64)?;
        let d = raw.dflash_config;
        if !raw.architectures.iter().any(|a| a == "DFlash2DraftModel")
            || d.attention_mode != "mla" || raw.is_causal || raw.mla_use_output_gate
            || raw.attention_bias || raw.hidden_act != "silu" || raw.fc_norm {
            return Err(unsupported("MLA DFlash2 requires noncausal, ungated, bias-free SiLU and fc_norm=false"));
        }
        let mla = MlaGeometry { hidden: raw.hidden_size, heads: raw.num_attention_heads,
            q_rank: raw.q_lora_rank, kv_rank: raw.kv_lora_rank, nope: raw.qk_nope_head_dim,
            rope: raw.qk_rope_head_dim, value: raw.v_head_dim };
        if mla != (MlaGeometry { hidden: 7168, heads: 64, q_rank: 1536,
            kv_rank: 512, nope: 128, rope: 64, value: 128 })
            || raw.num_key_value_heads != mla.heads || raw.rms_norm_eps != 1e-5 {
            return Err(unsupported("MLA draft exports require H7168, 64 heads, Q1536/KV512, NoPE128/RoPE64/V128, eps=1e-5"));
        }
        if raw.intermediate_size == 0 || raw.intermediate_size > 1_000_000
            || raw.intermediate_size % 128 != 0 || raw.num_hidden_layers == 0
            || raw.num_hidden_layers > 1024 || raw.vocab_size == 0 || raw.vocab_size > 1_000_000
            || raw.vocab_size != raw.draft_vocab_size || d.mask_token_id as usize >= raw.vocab_size
            || raw.max_position_embeddings == 0 || raw.max_position_embeddings > 1_048_576
            || raw.num_target_layers == 0 || raw.num_target_layers > 1_000_000
            || d.target_layer_ids.is_empty() || d.target_layer_ids.len() > 1024
            || d.target_layer_ids.iter().any(|&tap| tap >= raw.num_target_layers)
            || d.target_layer_ids.windows(2).any(|pair| pair[0] >= pair[1]) {
            return Err(unsupported("invalid MLA draft dimensions, context, vocabulary or ordered post-layer taps"));
        }
        if d.block_size != 8 || d.conv_kernel_size != 2 || d.conv_group_size != 16
            || d.selector_rank != 256 || d.selector_top_k != 16 {
            return Err(unsupported("DFlash2 requires block8, two-tap group16 convolutions, rank256/top16 selector"));
        }
        if raw.layer_types.len() != raw.num_hidden_layers {
            return Err(unsupported("MLA draft layer_types count differs from num_hidden_layers"));
        }
        let windows = raw.layer_types.iter().map(|kind| match kind.as_str() {
            "full_attention" => Ok(None),
            "sliding_attention" if raw.use_sliding_window && raw.sliding_window == Some(4096) => Ok(Some(4096)),
            _ => Err(unsupported("MLA draft exports support full attention or a native 4096-token window")),
        }).collect::<Result<Vec<_>>>()?;
        let rotary = raw.rope_parameters.plan(mla.rope)?;
        Ok(Self { mla, intermediate: raw.intermediate_size, eps: raw.rms_norm_eps,
            block: d.block_size, group: d.conv_group_size, selector_rank: d.selector_rank,
            selector_top_k: d.selector_top_k, taps: d.target_layer_ids, vocab: raw.vocab_size,
            mask_token: d.mask_token_id, max_context: raw.max_position_embeddings, windows, rotary,
            kind: cuteafd_core::coordinator_programs::MlaDraftKind::Dflash2, markov_rank: 0 })
    }

    /// Mistral Large 4's bundled mtp_dspark, from the checkpoint's own params.json
    /// (`dspark`) and text config (`target`): dense MLA layers with the target's
    /// attention geometry (no q-LoRA) and dense FFN, YaRN without LeRoPE (the
    /// draft's cos|sin table is the target's at alpha 0), FP32 attention scale
    /// 128^-0.5, one block-anchored window per layer.
    pub fn ml4(dspark: &crate::families::mistral_large4::dspark::Ml4DsparkConfig,
        target: &crate::families::mistral_large4::Ml4Config) -> Result<Self> {
        let rope = target.qk_rope_head_dim;
        if (target.hidden, target.heads, target.kv_lora_rank, target.qk_nope_head_dim, rope, target.v_head_dim)
            != (6144, 64, 512, 64, 64, 128) || dspark.depth != 3 || dspark.window != 2048 || dspark.markov_rank != 512 {
            return Err(unsupported("mtp_dspark exports require the Mistral Large 4 geometry, depth 3, window 2048, rank 512"));
        }
        let inv_freq = target.inv_freq(&vec![0.0; rope / 2])
            .map_err(|e| unsupported(&format!("mtp_dspark rotary: {e}")))?;
        Ok(Self { mla: MlaGeometry { hidden: target.hidden, heads: target.heads, q_rank: 0,
                kv_rank: target.kv_lora_rank, nope: target.qk_nope_head_dim, rope, value: target.v_head_dim },
            intermediate: target.dense_intermediate, eps: target.rms_norm_eps, block: dspark.depth, group: 0,
            selector_rank: 0, selector_top_k: 0, taps: dspark.taps.clone(), vocab: target.vocab_size,
            mask_token: 0, max_context: target.max_position_embeddings,
            windows: vec![Some(dspark.window); dspark.layers],
            rotary: DsparkRotaryPlan { inv_freq, attention_factor: 1.0, softmax_factor: 1.0 },
            kind: cuteafd_core::coordinator_programs::MlaDraftKind::Ml4Dspark, markov_rank: dspark.markov_rank })
    }

    fn is_ml4(&self) -> bool { self.kind == cuteafd_core::coordinator_programs::MlaDraftKind::Ml4Dspark }

    /// Drafted tokens per proposal: DFlash2's block holds the anchor (block - 1);
    /// mtp_dspark predicts from the anchor too (block).
    pub fn drafts(&self) -> usize { if self.is_ml4() { self.block } else { self.block - 1 } }

    pub fn attention_scale(&self) -> f32 {
        ((self.mla.nope + self.mla.rope) as f64).sqrt().recip() as f32 * self.rotary.softmax_factor
    }

    /// Admission uses the export's compiled extent, not a live sequence length.
    /// The loaded FFI binding validates these same capacities against its manifest.
    pub fn program_layout(&self, compiled_context: usize)
        -> Result<cuteafd_core::coordinator_programs::MlaDraftProgramLayout> {
        if compiled_context > self.max_context {
            return Err(unsupported("MLA draft export context exceeds checkpoint extent"));
        }
        cuteafd_core::coordinator_programs::MlaDraftProgramLayout::for_kind(self.kind, compiled_context)
            .map_err(|_| unsupported("MLA draft program context outside native extent"))
    }

    /// Exact BF16 source shapes. Embedding and vocabulary head are borrowed
    /// from the target, matching the checkpoint's reference execution.
    pub fn tensor_shapes(&self) -> Vec<(String, Vec<usize>)> {
        if self.is_ml4() {
            // The drafter ships in the target checkpoint under mtp_dspark.* (BF16).
            let dspark = crate::families::mistral_large4::dspark::Ml4DsparkConfig { layers: self.windows.len(),
                depth: self.block, taps: self.taps.clone(), window: self.windows[0].unwrap_or(0),
                markov_rank: self.markov_rank };
            let m = &self.mla;
            return dspark.tensors(m.hidden, m.heads, m.nope + m.rope, m.kv_rank, m.rope, m.nope, m.value,
                self.intermediate, self.vocab);
        }
        let h = self.mla.hidden;
        let mut shapes = vec![
            ("fc.weight".into(), vec![h, self.taps.len() * h]),
            ("hidden_norm.weight".into(), vec![h]),
            ("norm.weight".into(), vec![h]),
            ("candidate_selector.hidden_projection.weight".into(), vec![self.selector_rank, h]),
            ("candidate_selector.predecessor_codebook".into(), vec![self.vocab, self.selector_rank]),
            ("candidate_selector.successor_codebook".into(), vec![self.vocab, self.selector_rank]),
        ];
        for layer in 0..self.windows.len() {
            let p = format!("layers.{layer}");
            for leaf in ["input_layernorm.weight", "post_attention_layernorm.weight"] {
                shapes.push((format!("{p}.{leaf}"), vec![h]));
            }
            for conv in ["attention_conv", "mlp_conv"] {
                shapes.push((format!("{p}.{conv}.kernel_projection.weight"), vec![4 * h / self.group, h]));
                shapes.push((format!("{p}.{conv}.base_kernel"), vec![2, 2, h]));
            }
            for (leaf, shape) in [
                ("q_a_proj", vec![self.mla.q_rank, h]),
                ("kv_a_proj_with_mqa", vec![self.mla.kv_rank + self.mla.rope, h]),
                ("q_a_layernorm", vec![self.mla.q_rank]),
                ("kv_a_layernorm", vec![self.mla.kv_rank]),
                ("q_b_proj", vec![self.mla.heads * (self.mla.nope + self.mla.rope), self.mla.q_rank]),
                ("kv_b_proj", vec![self.mla.heads * (self.mla.nope + self.mla.value), self.mla.kv_rank]),
                ("o_proj", vec![h, self.mla.heads * self.mla.value]),
            ] {
                shapes.push((format!("{p}.self_attn.{leaf}.weight"), shape));
            }
            for (leaf, shape) in [
                ("gate_proj", vec![self.intermediate, h]),
                ("up_proj", vec![self.intermediate, h]),
                ("down_proj", vec![h, self.intermediate]),
            ] {
                shapes.push((format!("{p}.mlp.{leaf}.weight"), shape));
            }
        }
        shapes
    }

    pub fn bf16_weight_bytes(&self) -> Result<u64> {
        self.weight_layout(crate::families::glm5::draft_representation::GlmDraftRepresentation::Bf16Only)?
            .resident_bytes().map_err(|_| unsupported("MLA draft weight total overflow"))
    }

    /// The native MLA projections remain BF16 in both arms. Only the shared
    /// FC, selector projection, convolution projections and MLP matrices use
    /// the optional FP8 representation; embeddings and the head remain borrowed.
    pub fn weight_layout(&self,
        mode: crate::families::glm5::draft_representation::GlmDraftRepresentation)
        -> Result<crate::families::glm5::draft_representation::GlmDraftWeightLayout> {
        use crate::families::glm5::draft_representation::{GlmDraftRepresentation, GlmDraftWeightLayout};
        let overflow = || unsupported("MLA draft weight admission overflow");
        let mut layout = GlmDraftWeightLayout { bf16_values: 0, fp8_values: 0, fp8_scales: 0,
            bf16_auxiliary: 0, bf16_codebooks: 0, max_load_staging: 0 };
        if self.is_ml4() && mode == GlmDraftRepresentation::Fp8Only {
            return Err(unsupported("the mtp_dspark drafter is served as shipped (BF16)"));
        }
        for (name, shape) in self.tensor_shapes() {
            let values = shape.iter().try_fold(1u64, |n, &d| n.checked_mul(d as u64))
                .ok_or_else(overflow)?;
            let bytes = values.checked_mul(2).ok_or_else(overflow)?;
            let codebook = name.ends_with("_codebook") || name.contains("autoregressive_head.embedding")
                || name.contains("autoregressive_head.output");
            let matrix = name == "fc.weight" || name == "candidate_selector.hidden_projection.weight"
                || name.contains(".kernel_projection.") || name.contains(".mlp.");
            let fp8 = matrix && mode == GlmDraftRepresentation::Fp8Only;
            let destination = if fp8 {
                if shape.len() != 2 || shape[0] % 16 != 0 || shape[1] % 128 != 0 {
                    return Err(unsupported("MLA draft FP8 GEMMs require N multiples of 16 and K multiples of 128"));
                }
                layout.fp8_scales = layout.fp8_scales.checked_add(values / 128 * 4)
                    .ok_or_else(overflow)?;
                layout.max_load_staging = layout.max_load_staging.max(bytes);
                &mut layout.fp8_values
            } else if codebook { &mut layout.bf16_codebooks }
            else if shape.len() == 1 || name.ends_with(".base_kernel") { &mut layout.bf16_auxiliary }
            else { &mut layout.bf16_values };
            *destination = destination.checked_add(if fp8 { values } else { bytes })
                .ok_or_else(overflow)?;
        }
        layout.loading_peak_bytes().map_err(|_| overflow())?;
        Ok(layout)
    }

    fn weight_owners(&self, mode: crate::families::glm5::draft_representation::GlmDraftRepresentation)
        -> Result<(Vec<MlaDraftOwner>, Vec<MlaDraftOwner>, usize)> {
        use crate::families::glm5::draft_representation::GlmDraftRepresentation;
        use MlaDraftDtype::{Bf16, Bytes, F32};
        let mut owners = Vec::new();
        let mut sources = Vec::new();
        let mut sync = 0usize;
        let mut add = |id: String, shape: Vec<usize>, matrix: bool, upload: Option<usize>| -> Result<()> {
            let source = MlaDraftOwner::new(format!("{id}.source"), MlaDraftDomain::Device, Bf16, shape.clone(), 256)?;
            sync = sync.max(upload.unwrap_or(source.bytes));
            if matrix && mode == GlmDraftRepresentation::Fp8Only {
                if shape.len() != 2 || shape[0] % 16 != 0 || shape[1] % 128 != 0 {
                    return Err(unsupported("MLA draft FP8 GEMMs require N multiples of 16 and K multiples of 128"));
                }
                let values = source.bytes / 2;
                owners.push(MlaDraftOwner::new(format!("{id}.values"), MlaDraftDomain::Device, Bytes, vec![values], 0)?);
                owners.push(MlaDraftOwner::new(format!("{id}.scales"), MlaDraftDomain::Device, F32,
                    vec![shape[0], shape[1] / 128], 0)?);
                sources.push(MlaDraftOwner { dtype: Bytes, shape: vec![source.bytes], ..source });
            } else {
                owners.push(MlaDraftOwner { id, ..source });
            }
            Ok(())
        };
        let (h, inter, mla) = (self.mla.hidden, self.intermediate, &self.mla);
        // Match the executor's fresh-per-matrix packing order, including concatenations.
        for layer in 0..self.windows.len() {
            let p = format!("weight.layer.{layer}");
            for leaf in ["input_norm", "post_norm"] { add(format!("{p}.{leaf}"), vec![h], false, None)?; }
            for conv in ["attn", "mlp"] {
                add(format!("{p}.{conv}_conv"), vec![4 * h / self.group, h], true, None)?;
                add(format!("{p}.{conv}_base"), vec![2, 2, h], false, None)?;
            }
            for (leaf, shape) in [
                ("qkv_a", vec![mla.q_rank + mla.kv_rank + mla.rope, h]),
                ("q_norm", vec![mla.q_rank]), ("kv_norm", vec![mla.kv_rank]),
                ("q_b", vec![mla.heads * (mla.nope + mla.rope), mla.q_rank]),
                ("uk", vec![mla.heads, mla.kv_rank, mla.nope]),
                ("uv", vec![mla.heads, mla.value, mla.kv_rank]),
                ("output", vec![h, mla.heads * mla.value]),
            ] { add(format!("{p}.attention.{leaf}"), shape, false, None)?; }
            let half = inter.checked_mul(h).and_then(|n| n.checked_mul(2))
                .ok_or_else(|| unsupported("MLA draft gate/up extent overflow"))?;
            add(format!("{p}.gate_up"), vec![2 * inter, h], true, Some(half))?;
            add(format!("{p}.down"), vec![h, inter], true, None)?;
        }
        add("weight.fc".into(), vec![h, self.taps.len() * h], true, None)?;
        for leaf in ["hidden_norm", "norm"] { add(format!("weight.{leaf}"), vec![h], false, None)?; }
        add("weight.projection".into(), vec![self.selector_rank, h], true, None)?;
        for leaf in ["predecessor", "successor"] {
            add(format!("weight.{leaf}"), vec![self.vocab, self.selector_rank], false, None)?;
        }
        Ok((owners, sources, sync))
    }

    /// One checked inventory for the shared executor's common buffers and the
    /// native MLA branch. Native work is chunked, not replicated per layer or
    /// scheduler sequence. Only bounded rings and page tables scale with slots;
    /// full-context records scale with the target's aggregate token budget.
    pub fn runtime_layout(&self,
        mode: crate::families::glm5::draft_representation::GlmDraftRepresentation,
        capacity: crate::families::glm5::draft_representation::GlmDraftCapacity,
        compiled_context: usize, target_pool_tokens: usize, sm_count: usize, fp8_mode: u8, cache_format: MlaCacheFormat) -> Result<MlaDraftRuntimeLayout> {
        use crate::families::glm5::draft_representation::{draft_fp8_scratch_bytes,
            GlmDraftCapacity, GlmDraftFp8ScratchLayout, GlmDraftLinearShape, GlmDraftRepresentation};
        use cuteafd_core::coordinator_programs::MlaDraftProgramLayout;
        validate_topk_vocab(self.vocab as u64)?;
        let invalid = || unsupported("MLA draft runtime geometry/capacity outside native limits");
        if self.is_ml4() {
            return self.ml4_runtime_layout(capacity, compiled_context, target_pool_tokens, sm_count, cache_format);
        }
        if capacity != GlmDraftCapacity::new(capacity.context_slots, capacity.max_batch_sequences, self.block)
            .map_err(|_| invalid())? || capacity.max_batch_sequences > 32
            || !(1..=4096).contains(&sm_count) || fp8_mode > 2 || self.block != 8
            || self.group != 16 || self.selector_rank != 256 || self.selector_top_k != 16
            || self.mla != (MlaGeometry { hidden: 7168, heads: 64, q_rank: 1536,
                kv_rank: 512, nope: 128, rope: 64, value: 128 })
            || self.intermediate == 0 || self.intermediate > 1_000_000
            || self.vocab == 0 || self.vocab > 1_000_000 || self.taps.is_empty() || self.taps.len() > 1024
            || self.windows.is_empty() || self.windows.len() > 1024
            || self.windows.iter().any(|w| !matches!(w, None | Some(4096))) {
            return Err(invalid());
        }
        let program = self.program_layout(compiled_context)?;
        let cache = self.cache_layout(capacity.context_slots, compiled_context, cache_format)?;
        let paged_cache = self.paged_cache_layout(capacity.context_slots, compiled_context, target_pool_tokens, cache_format)?;
        let mut weights = self.weight_layout(mode)?;
        let fp8 = mode == GlmDraftRepresentation::Fp8Only;
        let (h, inter, rows, seq) = (self.mla.hidden, self.intermediate,
            capacity.block_rows, capacity.max_batch_sequences);
        let drafted = seq * (self.block - 1);
        let tap_rows = MlaDraftRuntimeLayout::TAP_ROWS;
        let native_rows = MlaDraftProgramLayout::ROWS;
        use MlaDraftDtype::{Bf16, Bytes, F32, I32, I64, U32};
        let mut buffers = Vec::new();
        let mut owners = Vec::new();
        let mut add = |name: String, dtype, shape: Vec<usize>| -> Result<()> {
            let owner = MlaDraftOwner::new(name.clone(), MlaDraftDomain::Device, dtype, shape, 256)?;
            buffers.push(MlaDraftBuffer { name, bytes: owner.bytes });
            owners.push(owner);
            Ok(())
        };
        for (index, layer) in cache.layers.iter().enumerate() {
            add(format!("layer.{index}.cache"), Bytes, vec![paged_cache.layer_bytes[index]])?;
            add(format!("layer.{index}.page_table"), I32, vec![layer.slots, layer.table_width])?;
        }
        for (name, dtype, dims) in [
            ("taps", Bf16, vec![tap_rows, self.taps.len(), h]),
            ("fused", Bf16, vec![tap_rows, h]), ("fused_norm", Bf16, vec![tap_rows, h]),
            ("context_positions", I64, vec![tap_rows]), ("context_slots", I64, vec![tap_rows]),
            ("inverse_frequencies", F32, vec![self.mla.rope / 2]),
            ("query", Bf16, vec![native_rows, 64, 576]),
            ("q_residual", Bf16, vec![native_rows, 1536]),
            ("attention", Bf16, vec![native_rows, 64, 512]),
            ("lse", F32, vec![native_rows, 64]),
            ("native_scratch", Bytes, vec![program.scratch_bytes()]),
            ("lengths", I32, vec![seq]), ("query_offsets", I32, vec![seq + 1]),
            ("native_slots", I64, vec![native_rows]), ("native_positions", I64, vec![native_rows]),
            ("head_workspace", Bytes, vec![4 << 20]),
            ("h", Bf16, vec![rows, h]), ("n", Bf16, vec![rows, h]),
            ("conv", Bf16, vec![rows, h]), ("dynamic", Bf16, vec![rows, 4 * h / self.group]),
            ("delta", Bf16, vec![rows, h]), ("gate_up", Bf16, vec![rows, 2 * inter]),
            ("act", Bf16, vec![rows, inter]), ("logits", F32, vec![rows, self.vocab]),
            ("unary", F32, vec![drafted, 16]), ("candidates", U32, vec![drafted, 16]),
            ("projected", Bf16, vec![rows, self.selector_rank]), ("anchors", U32, vec![seq]),
            ("ids", U32, vec![rows]), ("tokens", U32, vec![drafted]),
            ("features", Bytes, vec![drafted, 16]), ("positions", I64, vec![rows]),
            ("topk_workspace", Bytes, vec![draft_selector_workspace_bytes(self.kind, drafted)?]),
        ] { add(name.into(), dtype, dims)?; }

        let fp8_scratch = if fp8 {
            let shapes = vec![
                GlmDraftLinearShape { k: (self.taps.len() * h) as u64, n: h as u64 },
                GlmDraftLinearShape { k: h as u64, n: self.selector_rank as u64 },
                GlmDraftLinearShape { k: h as u64, n: (4 * h / self.group) as u64 },
                GlmDraftLinearShape { k: h as u64, n: (2 * inter) as u64 },
                GlmDraftLinearShape { k: inter as u64, n: h as u64 },
            ];
            if shapes.iter().any(|s| s.k % 128 != 0 || s.n % 16 != 0) { return Err(invalid()); }
            // The executor concatenates gate/up before packing; source headers
            // alone undercount this transient when the MLP exceeds FC size.
            weights.max_load_staging = shapes.iter().map(|s| s.k * s.n * 2)
                .max().unwrap_or(0).max(weights.max_load_staging);
            let scratch_rows = tap_rows.max(rows);
            let bytes = shapes.iter().flat_map(|s| (0..=fp8_mode).map(move |m|
                draft_fp8_scratch_bytes(scratch_rows as u64, s.k, s.n, sm_count as u64, m)))
                .max().unwrap_or(0).max(256);
            add("fp8_scratch".into(), Bytes, vec![usize::try_from(bytes).map_err(|_| invalid())?])?;
            Some(GlmDraftFp8ScratchLayout { rows: scratch_rows, shapes })
        } else { None };
        let (weight_owners, packing_sources, sync_upload_bytes) = self.weight_owners(mode)?;
        owners.extend(weight_owners);
        owners.push(MlaDraftOwner::new("landing", MlaDraftDomain::HostPinned, Bytes, vec![drafted * 20], 0)?);
        let metadata = ["context_positions", "context_slots", "positions", "anchors", "ids",
            "native_positions", "native_slots", "lengths", "query_offsets"];
        for buffer in &buffers {
            let diagnostic = buffer.name == "taps";
            if diagnostic || metadata.contains(&buffer.name.as_str()) || buffer.name.ends_with(".page_table") {
                let mut owner = MlaDraftOwner::new(format!("{}.upload", buffer.name),
                    MlaDraftDomain::HostPinned, Bytes, vec![buffer.bytes], 0)?;
                owner.lifetime = MlaDraftLifetime::Library;
                owner.staging_for = Some(buffer.name.clone());
                owner.on_use = diagnostic;
                owners.push(owner);
            }
        }
        let layout = MlaDraftRuntimeLayout { kind: self.kind, cache_format, target_hidden: h, target_vocab: self.vocab,
            target_taps: self.taps.clone(), capacity, weights, cache, paged_cache, program,
            fp8_scratch, sm_count, fp8_mode, buffers, owners, packing_sources, sync_upload_bytes, weight_destinations: Vec::new() };

        layout.resident_bytes()?;
        Ok(layout)
    }

    /// mtp_dspark: the shared cache/paged layouts and native arenas, its own
    /// layer bodies (no convolutions), head (RNN Markov, the target's head and
    /// soft cap) and inputs (position-specific mask rows), BF16 weights. The
    /// cos|sin table covers every compiled position (YaRN without LeRoPE).
    fn ml4_runtime_layout(&self, capacity: crate::families::glm5::draft_representation::GlmDraftCapacity,
        compiled_context: usize, target_pool_tokens: usize, sm_count: usize, cache_format: MlaCacheFormat)
        -> Result<MlaDraftRuntimeLayout> {
        use crate::families::glm5::draft_representation::{GlmDraftCapacity, GlmDraftRepresentation};
        use cuteafd_core::coordinator_programs::MlaDraftProgramLayout;
        let invalid = || unsupported("mtp_dspark runtime geometry/capacity outside native limits");
        if capacity != GlmDraftCapacity::new(capacity.context_slots, capacity.max_batch_sequences, self.block)
            .map_err(|_| invalid())? || capacity.max_batch_sequences > 32 || !(1..=4096).contains(&sm_count)
            || cache_format != MlaCacheFormat::Fp8 || self.windows.iter().any(|w| *w != Some(2048))
            || self.markov_rank != 512 {
            return Err(invalid());
        }
        let program = self.program_layout(compiled_context)?;
        let cache = self.cache_layout(capacity.context_slots, compiled_context, cache_format)?;
        let paged_cache = self.paged_cache_layout(capacity.context_slots, compiled_context, target_pool_tokens, cache_format)?;
        let weights = self.weight_layout(GlmDraftRepresentation::Bf16Only)?;
        let (h, inter, rows, seq) = (self.mla.hidden, self.intermediate, capacity.block_rows,
            capacity.max_batch_sequences);
        let tap_rows = MlaDraftRuntimeLayout::TAP_ROWS;
        let native_rows = MlaDraftProgramLayout::ROWS;
        let mut buffers = Vec::new();
        let mut owners = Vec::new();
        let mut add = |name: String, dimensions: &[usize], dtype: MlaDraftDtype| -> Result<()> {
            let width: usize = match dtype { MlaDraftDtype::Bytes => 1, MlaDraftDtype::Bf16 => 2,
                MlaDraftDtype::F32 | MlaDraftDtype::I32 | MlaDraftDtype::U32 => 4, MlaDraftDtype::I64 => 8 };
            let bytes = dimensions.iter().try_fold(width, |n, &d| n.checked_mul(d))
                .ok_or_else(|| unsupported("mtp_dspark arena multiplication overflow"))?.max(256);
            owners.push(MlaDraftOwner { id: name.clone(), domain: MlaDraftDomain::Device, bytes,
                alignment: 256, dtype, shape: dimensions.to_vec(), lifetime: MlaDraftLifetime::Runtime,
                shared_key: None, staging_for: None, on_use: false });
            buffers.push(MlaDraftBuffer { name, bytes });
            Ok(())
        };
        use MlaDraftDtype::{Bytes, Bf16, F32, I32, U32, I64};
        for (index, layer) in cache.layers.iter().enumerate() {
            add(format!("layer.{index}.cache"), &[paged_cache.layer_bytes[index]], Bytes)?;
            add(format!("layer.{index}.page_table"), &[layer.slots, layer.table_width], I32)?;
        }
        for (name, dims, dtype) in [
            ("taps", vec![tap_rows, self.taps.len(), h], Bf16),
            ("fused", vec![tap_rows, h], Bf16), ("fused_norm", vec![tap_rows, h], Bf16),
            ("context_positions", vec![tap_rows], I64), ("context_slots", vec![tap_rows], I64),
            ("inverse_frequencies", vec![compiled_context, self.mla.rope], F32),
            ("query", vec![native_rows, 64, 576], Bf16),
            ("attention", vec![native_rows, 64, 512], Bf16), ("lse", vec![native_rows, 64], F32),
            ("native_scratch", vec![program.scratch_bytes()], Bytes),
            ("lengths", vec![seq], I32), ("query_offsets", vec![seq + 1], I32),
            ("native_slots", vec![native_rows], I64), ("native_positions", vec![native_rows], I64),
            ("head_workspace", vec![4 << 20], Bytes),
            ("h", vec![rows, h], Bf16), ("n", vec![rows, h], Bf16), ("delta", vec![rows, h], Bf16),
            ("gate_up", vec![rows, 2 * inter], Bf16), ("act", vec![rows, inter], Bf16),
            ("logits", vec![rows, self.vocab], F32), ("markov_norms", vec![self.vocab], F32),
            ("markov_workspace", vec![seq * 296 * 8 + rows * 4 + seq * self.markov_rank * 4], Bytes),
            ("anchors", vec![seq], U32), ("ids", vec![rows], U32),
            ("tokens", vec![seq * self.drafts()], U32), ("positions", vec![rows], I64),
        ] { add(name.into(), &dims, dtype)?; }
        let weight_destinations = self.ml4_weight_destinations()?;
        for weight in &weight_destinations {
            let bytes = weight.shape.iter().try_fold(2usize, |n, &d| n.checked_mul(d))
                .ok_or_else(|| unsupported("mtp_dspark weight destination overflow"))?.max(256);
            owners.push(MlaDraftOwner { id: weight.owner.clone(), domain: MlaDraftDomain::Device, bytes,
                alignment: 256, dtype: Bf16, shape: weight.shape.clone(), lifetime: MlaDraftLifetime::Runtime,
                shared_key: None, staging_for: None, on_use: false });
        }
        // Ordered uploads keep one pinned backing per destination until library teardown.
        // Host-row and put_taps diagnostics are lazy; normal serving supplies device rows.
        let metadata = ["context_positions", "context_slots", "positions", "anchors", "ids",
            "native_positions", "native_slots", "lengths", "query_offsets"];
        let staging: Vec<_> = owners.iter().filter(|o| metadata.contains(&o.id.as_str())
            || o.id.ends_with(".page_table") || matches!(o.id.as_str(), "h" | "taps"))
            .map(|o| MlaDraftOwner { id: format!("host.ordered.{}", o.id), domain: MlaDraftDomain::HostPinned,
                bytes: o.bytes, alignment: 1, dtype: Bytes, shape: vec![o.bytes], lifetime: MlaDraftLifetime::Library,
                shared_key: Some(format!("ordered-h2d/{}", o.id)), staging_for: Some(o.id.clone()),
                on_use: matches!(o.id.as_str(), "h" | "taps") }).collect();
        owners.extend(staging);
        owners.push(MlaDraftOwner { id: "host.landing".into(), domain: MlaDraftDomain::HostPinned,
            bytes: seq * self.drafts() * 4, alignment: 1, dtype: U32, shape: vec![seq, self.drafts()],
            lifetime: MlaDraftLifetime::Runtime, shared_key: None, staging_for: None, on_use: false });
        let largest_upload = owners.iter().filter(|o| o.domain == MlaDraftDomain::Device
            && (o.id.starts_with("weight.") || o.id == "inverse_frequencies"))
            .map(|o| o.bytes).max().unwrap_or(0);
        owners.push(MlaDraftOwner { id: "host.sync_h2d".into(), domain: MlaDraftDomain::HostPinned,
            bytes: largest_upload, alignment: 1, dtype: Bytes, shape: vec![largest_upload],
            lifetime: MlaDraftLifetime::Library, shared_key: Some("library/sync-h2d".into()),
            staging_for: None, on_use: false });
        let destination_bytes = owners.iter().filter(|o| o.id.starts_with("weight."))
            .try_fold(0u64, |sum, o| sum.checked_add(o.bytes as u64))
            .ok_or_else(|| unsupported("mtp_dspark weight owner total overflow"))?;
        let mut weights = weights;
        let source_bytes = weights.resident_bytes().map_err(|_| unsupported("mtp_dspark weight total overflow"))?;
        weights.bf16_auxiliary = weights.bf16_auxiliary.checked_add(destination_bytes.checked_sub(source_bytes)
            .ok_or_else(|| unsupported("mtp_dspark packing lost source bytes"))?)
            .ok_or_else(|| unsupported("mtp_dspark generated weight overflow"))?;
        let layout = MlaDraftRuntimeLayout { kind: self.kind, cache_format, target_hidden: h, target_vocab: self.vocab,
            target_taps: self.taps.clone(), capacity, weights, cache, paged_cache, program,
            fp8_scratch: None, sm_count, fp8_mode: 0, buffers, owners, weight_destinations,
            packing_sources: Vec::new(), sync_upload_bytes: largest_upload };
        layout.resident_bytes()?;
        Ok(layout)
    }

    fn ml4_weight_destinations(&self) -> Result<Vec<MlaDraftWeight>> {
        use MlaDraftPacking::{Direct, GateUp, Uk, UnitScale, Uv};
        let prefix = crate::families::mistral_large4::dspark::PREFIX;
        let mut destinations = Vec::new();
        for (source, shape) in self.tensor_shapes() {
            let name = source.strip_prefix(prefix).ok_or_else(|| unsupported("mtp_dspark tensor prefix"))?;
            if name.ends_with(".attention.wkv_b.weight") {
                for (suffix, packed, packing) in [
                    ("uk", vec![self.mla.heads, self.mla.kv_rank, self.mla.nope], Uk),
                    ("uv", vec![self.mla.heads, self.mla.value, self.mla.kv_rank], Uv),
                ] {
                    destinations.push(MlaDraftWeight { owner: format!("weight.{name}.{suffix}"), shape: packed,
                        sources: vec![(source.clone(), shape.clone())], packing });
                }
            } else if name.ends_with(".feed_forward.w1.weight") {
                destinations.push(MlaDraftWeight { owner: format!("weight.{name}.gate_up"),
                    shape: vec![2 * self.intermediate, self.mla.hidden],
                    sources: vec![(source.clone(), shape.clone()),
                        (source.replace(".w1.weight", ".w3.weight"), shape)], packing: GateUp });
            } else if !name.ends_with(".feed_forward.w3.weight") {
                destinations.push(MlaDraftWeight { owner: format!("weight.{name}"), shape: shape.clone(),
                    sources: vec![(source, shape)], packing: Direct });
            }
        }
        for index in 0..self.windows.len() {
            destinations.push(MlaDraftWeight { owner: format!("weight.backbone.layers.{index}.attention.unit_scale"),
                shape: vec![2, self.mla.heads], sources: Vec::new(), packing: UnitScale });
        }
        Ok(destinations)
    }

    /// Full-context pages come from the target's aggregate token budget. Window
    /// rings retain their existing per-request append slack and table geometry.
    pub fn paged_cache_layout(&self, slots: usize, context: usize, target_tokens: usize, cache_format: MlaCacheFormat)
        -> Result<DraftCacheLayout> {
        let source = self.cache_layout(slots, context, cache_format)?;
        let pages = target_tokens.div_ceil(64);
        if target_tokens == 0 || pages > i32::MAX as usize {
            return Err(unsupported("draft pool token budget outside native page limits"));
        }
        let overflow = || unsupported("draft paged layout overflow");
        let record_bytes = cache_format.record_bytes();
        let full_layer = pages.checked_mul(64).and_then(|n| n.checked_mul(record_bytes))
            .ok_or_else(overflow)?;
        let (mut window_bytes, mut full_bytes, mut full_layers) = (0u64, 0u64, 0usize);
        let mut layer_bytes = Vec::with_capacity(source.layers.len());
        for layer in &source.layers {
            let bytes = if layer.window.is_none() {
                full_layers += 1;
                full_bytes = full_bytes.checked_add(full_layer as u64).ok_or_else(overflow)?;
                full_layer
            } else {
                let bytes = layer.rows_per_slot.checked_mul(slots)
                    .and_then(|n| n.checked_mul(record_bytes)).ok_or_else(overflow)?;
                window_bytes = window_bytes.checked_add(bytes as u64).ok_or_else(overflow)?;
                bytes
            };
            layer_bytes.push(bytes);
        }
        let layout = DraftCacheLayout { target_tokens, page_rows: 64, pages, record_bytes,
            full_layers, window_bytes, full_bytes, page_table_bytes: source.page_table_bytes, layer_bytes };
        layout.cache_bytes()?;
        Ok(layout)
    }

    /// Cache admission shared by planner and runtime. Full layers admit the
    /// selected context (blocks must end inside it); sliding layers additionally
    /// admit a private block and page-offset slack in their virtual tables.
    pub fn cache_layout(&self, slots: usize, context: usize, cache_format: MlaCacheFormat) -> Result<MlaDraftCacheLayout> {
        const PAGE: usize = 64;
        if slots == 0 || context == 0 || context > self.max_context || self.block == 0 {
            return Err(unsupported("MLA draft cache slots/context outside checkpoint extent"));
        }
        let overflow = || unsupported("MLA draft cache admission overflow");
        let record = cache_format.record_bytes() as u64;
        let mut layers = Vec::with_capacity(self.windows.len());
        let (mut cache_bytes, mut page_table_bytes, mut margin_bytes) = (0u64, 0u64, 0u64);
        for &window in &self.windows {
            if window == Some(0) { return Err(unsupported("MLA draft window must be nonzero")); }
            let retained = window.unwrap_or(context).min(context);
            let live = match window {
                Some(w) => w.checked_add(self.block).ok_or_else(overflow)?.min(context),
                None => context,
            };
            let rows_per_slot = live.checked_add(PAGE - 1).ok_or_else(overflow)? / PAGE * PAGE;
            let table_width = match window {
                Some(_) => live.checked_add(PAGE - 1).ok_or_else(overflow)?.div_ceil(PAGE),
                None => rows_per_slot / PAGE,
            };
            let total_rows = rows_per_slot.checked_mul(slots).ok_or_else(overflow)?;
            if total_rows / PAGE > i32::MAX as usize || total_rows > i64::MAX as usize {
                return Err(unsupported("MLA draft cache exceeds native page/row indices"));
            }
            let bytes = (total_rows as u64).checked_mul(record).ok_or_else(overflow)?;
            let margin = ((rows_per_slot - retained) as u64).checked_mul(slots as u64)
                .and_then(|n| n.checked_mul(record)).ok_or_else(overflow)?;
            let tables = table_width.checked_mul(slots).and_then(|n| n.checked_mul(4))
                .ok_or_else(overflow)? as u64;
            cache_bytes = cache_bytes.checked_add(bytes).ok_or_else(overflow)?;
            margin_bytes = margin_bytes.checked_add(margin).ok_or_else(overflow)?;
            page_table_bytes = page_table_bytes.checked_add(tables).ok_or_else(overflow)?;
            layers.push(MlaDraftCacheLayer { slots, context, block: self.block, page_rows: PAGE,
                window, rows_per_slot, table_width });
        }
        Ok(MlaDraftCacheLayout { layers, cache_bytes, page_table_bytes, margin_bytes })
    }

    /// Latent BF16 records only, not block/workspace/staging admission. Full
    /// layers grow to the selected context; sliding layers do not silently grow.
    pub fn cache_bytes(&self, slots: usize, context: usize) -> Result<u64> {
        if slots == 0 || context == 0 || context > self.max_context {
            return Err(unsupported("MLA draft cache slots/context outside checkpoint extent"));
        }
        let rows = self.windows.iter().try_fold(0u64, |sum, window| {
            sum.checked_add(window.unwrap_or(context) as u64)
                .ok_or_else(|| unsupported("MLA draft cache row overflow"))
        })?;
        rows.checked_mul(slots as u64).and_then(|n| n.checked_mul((self.mla.kv_rank + self.mla.rope) as u64))
            .and_then(|n| n.checked_mul(2)).ok_or_else(|| unsupported("MLA draft cache byte overflow"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn topk_vocab_checks_only_native_index_and_output_limits() {
        for vocab in [154880, 163840, 196608, 196609, 202048, 248320, i32::MAX as u64] {
            assert_eq!(validate_topk_vocab(vocab), Ok(()));
        }
        for vocab in [0, 15, i32::MAX as u64 + 1, u64::MAX] {
            assert_eq!(validate_topk_vocab(vocab), Err(DflashTopkVocabError { vocab }));
        }
    }

    fn config() -> serde_json::Value {
        json!({"architectures":["DFlash2DraftModel"], "hidden_size":7168,
            "intermediate_size":14336, "num_hidden_layers":5, "num_attention_heads":64,
            "num_key_value_heads":64, "q_lora_rank":1536, "kv_lora_rank":512,
            "qk_nope_head_dim":128, "qk_rope_head_dim":64, "v_head_dim":128,
            "rms_norm_eps":1e-5, "vocab_size":163840, "draft_vocab_size":163840,
            "max_position_embeddings":1048576, "num_target_layers":93,
            "layer_types":["sliding_attention","sliding_attention","sliding_attention",
                "sliding_attention","full_attention"], "sliding_window":4096,
            "use_sliding_window":true, "is_causal":false, "mla_use_output_gate":false,
            "attention_bias":false, "hidden_act":"silu", "fc_norm":false,
            "rope_parameters":{"rope_type":"yarn", "rope_theta":50000.0, "factor":32.0,
                "original_max_position_embeddings":32768, "beta_fast":32, "beta_slow":1,
                "mscale":1.0, "mscale_all_dim":1.0},
            "dflash_config":{"attention_mode":"mla", "block_size":8,
                "conv_kernel_size":2, "conv_group_size":16, "selector_rank":256,
                "selector_top_k":16, "target_layer_ids":[19,37,66,78,90], "mask_token_id":163592}})
    }

    #[test]
    fn native_mla_weights_and_full_sliding_cache_are_exact() {
        let cfg = MlaDflashConfig::from_json(&config()).unwrap();
        assert_eq!(cfg.windows, vec![Some(4096), Some(4096), Some(4096), Some(4096), None]);
        assert_eq!(cfg.taps, [19,37,66,78,90]);
        assert_eq!(cfg.bf16_weight_bytes().unwrap(), 5_036_945_408);
        assert!(!cfg.tensor_shapes().iter().any(|(name, _)| name.contains("embed_tokens") || name.contains("lm_head")));
        assert_eq!(cfg.cache_bytes(8, 262144).unwrap(), 2_566_914_048);
        assert_eq!(cfg.cache_bytes(8, 1048576).unwrap(), 9_814_671_360);
        assert_eq!(cfg.rotary.attention_factor, 1.0);
        let expected = (192.0f64).sqrt().recip() * (1.0 + 0.1 * 32.0f64.ln()).powi(2);
        assert!((cfg.attention_scale() as f64 - expected).abs() < 2e-8);
        assert!(cfg.cache_bytes(0, 262144).is_err());
        assert!(cfg.cache_bytes(8, 1048577).is_err());
        assert!(cfg.cache_bytes(usize::MAX, 1048576).is_err());
    }

    #[test]
    fn admission_program_layout_matches_native_capacity_at_256k_and_1m() {
        use cuteafd_core::coordinator_programs::MlaDraftProgramLayout;
        let cfg = MlaDflashConfig::from_json(&config()).unwrap();
        for context in [262144, 1048576] {
            let layout = cfg.program_layout(context).unwrap();
            assert_eq!(layout, MlaDraftProgramLayout::new(context).unwrap());
            assert_eq!(layout.scratch_bytes(), 67_371_008);
            assert_eq!(layout.scratch[2], 1_048_576);
        }
        assert!(cfg.program_layout(0).is_err());
        assert!(cfg.program_layout(1048577).is_err());
        let mut shorter = cfg;
        shorter.max_context = 262144;
        assert!(shorter.program_layout(262144).is_ok());
        assert!(shorter.program_layout(1048576).is_err());
    }

    #[test]
    fn selector_workspace_is_head_aware_and_preserves_glm53_dflash2_bytes() {
        use cuteafd_core::coordinator_programs::MlaDraftKind::*;
        use crate::families::glm5::draft_representation::GlmDraftCapacity;
        for sequences in [1, 8, 16, 32] {
            let ml4 = GlmDraftCapacity::new(40, sequences, 3).unwrap();
            let glm53 = GlmDraftCapacity::new(40, sequences, 16).unwrap();
            let drafted = |c: GlmDraftCapacity| c.block_rows - c.max_batch_sequences;
            assert_eq!(draft_selector_workspace_bytes(Ml4Dspark, drafted(ml4)).unwrap(), 0);
            assert_eq!(draft_selector_workspace_bytes(Dflash2, drafted(glm53)).unwrap(),
                sequences * 15 * 8192);
        }
        assert!(draft_selector_workspace_bytes(Dflash2, usize::MAX).is_err());
    }

    #[test]
    fn ml4_owner_and_view_receipts_match_tp1_and_five_spark_min() -> anyhow::Result<()> {
        use std::collections::{BTreeMap, BTreeSet};
        use crate::families::glm5::draft_representation::{GlmDraftCapacity, GlmDraftRepresentation::Bf16Only};
        let cfg = MlaDflashConfig { mla: MlaGeometry { hidden: 6144, heads: 64, q_rank: 0,
            kv_rank: 512, nope: 64, rope: 64, value: 128 }, intermediate: 24576, eps: 1e-5,
            block: 3, group: 0, selector_rank: 512, selector_top_k: 0, taps: vec![1,18,36,54,71],
            vocab: 204800, mask_token: 0, max_context: 1 << 20, windows: vec![Some(2048); 3],
            rotary: DsparkRotaryPlan { inv_freq: vec![1.;32], attention_factor: 1., softmax_factor: 1. },
            kind: cuteafd_core::coordinator_programs::MlaDraftKind::Ml4Dspark, markov_rank: 512 };
        for (label, slots, sequences, device) in [("tp1", 5, 1, 0), ("min-five-sparks", 20, 16, 1)] {
            let layout = cfg.runtime_layout(Bf16Only, GlmDraftCapacity::new(slots, sequences, 3)?,
                1 << 20, 1 << 21, 188, 0, MlaCacheFormat::Fp8)?;
            let allocated: BTreeMap<_, _> = layout.allocate_device_with(|b| Ok::<_, anyhow::Error>((b.name.clone(), b.bytes)))?
                .into_iter().collect();
            let demanded: BTreeMap<_, _> = layout.demands(device)?.0.movable(2)?.conditional[0].demands.iter()
                .map(|d| (d.group.strip_prefix("MLA DFlash ").unwrap().to_string(), d.bytes as usize)).collect();
            assert_eq!(allocated, demanded, "{label}: every owner, not just totals");
            let shipped: Vec<serde_json::Value> = serde_json::from_str(include_str!("../families/mistral_large4/official-tensors.json"))?;
            let mut headers = BTreeMap::new();
            for tensor in shipped {
                let pattern = tensor["pattern"].as_str().unwrap();
                if !pattern.starts_with("mtp_dspark.") { continue; }
                let shape: Vec<_> = tensor["shape"].as_array().unwrap().iter().map(|d| d.as_u64().unwrap() as usize).collect();
                if let Some(axes) = tensor["axes"].as_array().unwrap().first() {
                    for index in axes.as_array().unwrap() { headers.insert(pattern.replacen("{}", &index.to_string(), 1), shape.clone()); }
                } else { headers.insert(pattern.to_string(), shape); }
            }
            let mut sources = BTreeSet::new();
            for destination in layout.weight_destinations() {
                assert_eq!(allocated[&destination.owner], (destination.shape.iter().product::<usize>() * 2).max(256));
                for (name, shape) in &destination.sources { assert_eq!(&headers[name], shape); sources.insert(name.clone()); }
            }
            assert_eq!(sources, headers.keys().cloned().collect());
            assert_eq!(layout.weight_destinations().len(), 41);
            assert_eq!(allocated.iter().filter(|(id, _)| id.ends_with(".unit_scale")).count(), 3);
            for id in ["conv", "dynamic", "unary", "candidates", "projected", "features", "topk_workspace", "q_residual", "unit_scale"] {
                assert!(!allocated.contains_key(id));
            }
            // Every descriptor class used by the executor, including every slot and tail chunk.
            let check = |view: MlaDraftView, backing: &str, offset: usize, bytes: usize| {
                assert_eq!((view.backing.as_str(), view.offset, view.bytes), (backing, offset, bytes));
                let owner = layout.owners().iter().find(|o| o.id == backing).unwrap();
                assert!(offset + bytes <= owner.bytes);
            };
            for first in 0..sequences { for count in 1..=sequences-first {
                if count * 3 > 64 { continue; }
                let [input, output] = layout.block_views(first, count * 3)?;
                check(input, "n", first * 3 * 6144 * 2, count * 3 * 6144 * 2);
                check(output, "delta", first * 3 * 6144 * 2, count * 3 * 6144 * 2);
            } }
            for first in (0..2048).step_by(64) { for rows in [1, 17, 64] {
                let [input, slots, positions] = layout.context_views(first, rows)?;
                check(input, "fused_norm", first * 6144 * 2, rows * 6144 * 2);
                check(slots, "context_slots", first * 8, rows * 8);
                check(positions, "context_positions", first * 8, rows * 8);
            } }
            for sequence in 0..sequences { check(layout.mask_view(sequence)?, "h", (sequence*3+1)*6144*2, 2*6144*2); }
            for layer in 0..3 { for slot in 0..slots {
                check(layout.window_view(layer, slot)?, &format!("layer.{layer}.cache"), slot*2112*656, 2112*656);
            } }
            for count in 1..=sequences {
                let [tokens, landing] = layout.result_views(count)?;
                check(tokens, "tokens", 0, count*3*4); check(landing, "host.landing", 0, count*3*4);
            }
            assert!(layout.block_views(sequences, 3).is_err());
            assert!(layout.context_views(2048, 1).is_err());
            assert!(layout.window_view(0, slots).is_err());
            let ordered: Vec<_> = layout.owners().iter().filter(|o| o.staging_for.is_some()).collect();
            for owner in &ordered {
                assert_eq!(owner.bytes, allocated[owner.staging_for.as_ref().unwrap()]);
                assert_eq!(owner.domain, MlaDraftDomain::HostPinned);
                assert_eq!(owner.lifetime, MlaDraftLifetime::Library);
            }
            let mandatory: usize = ordered.iter().filter(|o| !o.on_use).map(|o| o.bytes).sum();
            let lazy: usize = ordered.iter().filter(|o| o.on_use).map(|o| o.bytes).sum();
            assert_eq!(mandatory, 32768 + (sequences*3*8).max(256) + 4*256 + 2*512 + 3*(slots*34*4).max(256));
            assert_eq!(lazy, 2048*5*6144*2 + sequences*3*6144*2);
            assert_eq!(layout.host_owner("host.sync_h2d")?.bytes, 603_979_776);
            eprintln!("{label}: ordered={mandatory} lazy={lazy} landing={} shared_sync={} device_owners={} device_bytes={}",
                layout.host_owner("host.landing")?.bytes, layout.host_owner("host.sync_h2d")?.bytes,
                allocated.len(), allocated.values().sum::<usize>());
        }
        Ok(())
    }

    #[test]
    fn ml4_packing_recipe_preserves_native_order_and_generated_scales() -> anyhow::Result<()> {
        let source = ("kv".into(), vec![6, 2]);
        let bytes: Vec<u8> = (0u16..12).flat_map(u16::to_le_bytes).collect();
        let mut read = |_: &str, _: &[usize]| Ok(bytes.clone());
        let make = |packing, shape| MlaDraftWeight { owner: "weight.test".into(), shape, sources: vec![source.clone()], packing };
        let words = |bytes: Vec<u8>| bytes.chunks_exact(2).map(|b| u16::from_le_bytes([b[0],b[1]])).collect::<Vec<_>>();
        assert_eq!(words(make(MlaDraftPacking::Uk, vec![2,2,1]).pack(&mut read)?), [0,1,6,7]);
        assert_eq!(words(make(MlaDraftPacking::Uv, vec![2,2,2]).pack(&mut read)?), [2,3,4,5,8,9,10,11]);
        let unit = MlaDraftWeight { owner: "unit".into(), shape: vec![2,2], sources: vec![], packing: MlaDraftPacking::UnitScale };
        assert_eq!(words(unit.pack(&mut read)?), [0,0,0x3f80,0x3f80]);
        let fused = MlaDraftWeight { owner: "fused".into(), shape: vec![12,2], sources: vec![source.clone(),source], packing: MlaDraftPacking::GateUp };
        assert_eq!(fused.pack(&mut read)?, [bytes.clone(),bytes].concat());
        Ok(())
    }

    #[test]
    fn runtime_layout_allocates_exact_admitted_bytes_at_both_contexts_and_sm_counts() {
        use crate::families::glm5::draft_representation::{GlmDraftCapacity, GlmDraftRepresentation::*};
        let cfg = MlaDflashConfig::from_json(&config()).unwrap();
        for context in [262144, 1048576] {
            for sms in [170, 188] {
                for mode in [Bf16Only, Fp8Only] {
                    let capacity = GlmDraftCapacity::new(8, 8, 8).unwrap();
                    let layout = cfg.runtime_layout(mode, capacity, context, context, sms, 2, MlaCacheFormat::Bf16).unwrap();
                    let bridge = crate::families::kimi_k3::draft_bridge_demands(&layout, 4096).unwrap();
                    assert_eq!(bridge.len(), 5);
                    assert_eq!(bridge.iter().find(|d| d.group == "DFlash bridge taps").unwrap().bytes,
                        (4096 * 5 * 7168 * 2) as u64);
                    assert!(bridge.iter().filter(|d| d.group != "DFlash bridge exchange").all(|d| d.gpu == 0));
                    assert!(!bridge.iter().any(|d| d.group.contains("head scratch") || d.group.contains("logits")));

                    let allocated = layout.allocate_with(|b| Ok::<_, ()>((b.name.clone(), b.bytes))).unwrap();
                    let bytes = |name: &str| allocated.iter().find(|(n, _)| n == name).unwrap().1;
                    assert_eq!(bytes("query"), 64 * 64 * 576 * 2);
                    assert_eq!(bytes("q_residual"), 64 * 1536 * 2);
                    assert_eq!(bytes("attention"), 64 * 64 * 512 * 2);
                    assert_eq!(bytes("native_scratch"), 67_371_008);
                    assert_eq!(bytes("taps"), 2048 * 5 * 7168 * 2);
                    assert_eq!(bytes("logits"), 64 * 163840 * 4);
                    assert_eq!(bytes("layer.0.cache"), 8 * 4160 * 1152);
                    assert_eq!(bytes("layer.4.cache"), context * 1152);
                    assert_eq!(bytes("layer.4.page_table"), 8 * context / 64 * 4);
                    assert_eq!(bytes("topk_workspace"), 56 * 64 * 16 * 8);
                    assert_eq!(bytes("topk_workspace"), layout.selector_workspace_bytes().unwrap());
                    assert_eq!(allocated.iter().map(|(_, b)| *b as u64).sum::<u64>(), layout.buffer_bytes().unwrap());
                    assert_eq!(allocated.iter().map(|(n, _)| n).collect::<std::collections::BTreeSet<_>>().len(), allocated.len());
                    assert!(!allocated.iter().any(|(n, _)| n.contains("embed") || n == "mask_row" || n == "head_weights"));
                    let (demand, startup) = layout.demands(1).unwrap();
                    let movable = demand.movable(2).unwrap();
                    assert_eq!(movable.allowed, [1]);
                    assert_eq!(movable.conditional[0].demands.iter().map(|d| d.bytes).sum::<u64>(),
                        layout.resident_bytes().unwrap());
                    assert_eq!(startup.bytes, layout.resident_bytes().unwrap() + layout.weights.max_load_staging);
                    assert_eq!(layout.fp8_scratch.is_some(), mode == Fp8Only);
                    let cache_sum = allocated.iter().filter(|(n, _)| n.ends_with(".cache"))
                        .map(|(_, n)| *n as u64).sum::<u64>();
                    assert_eq!(cache_sum, layout.paged_cache.cache_bytes().unwrap());
                }
            }
        }
    }

    #[test]
    fn named_owners_keep_arenas_weights_pinned_and_exact_packing_sources_separate() {
        use crate::families::glm5::draft_representation::{GlmDraftCapacity, GlmDraftRepresentation::*};
        let mut cfg = MlaDflashConfig::from_json(&config()).unwrap();
        cfg.intermediate = 32768;
        for mode in [Bf16Only, Fp8Only] {
            for sequences in [1, 8, 32] {
                let layout = cfg.runtime_layout(mode, GlmDraftCapacity::new(40, sequences, 8).unwrap(),
                    262144, 128, 170, 2, MlaCacheFormat::Fp8).unwrap();
                let owners = layout.owners();
                assert_eq!(owners.iter().map(|o| &o.id).collect::<std::collections::BTreeSet<_>>().len(), owners.len());
                let owner = |id: &str| owners.iter().find(|o| o.id == id).unwrap();
                for buffer in layout.device_buffers() {
                    assert_eq!(owner(&buffer.name).bytes, buffer.bytes);
                    assert_eq!(owner(&buffer.name).domain, MlaDraftDomain::Device);
                    assert!(!buffer.name.starts_with("weight."));
                }
                assert_eq!(owner("query_offsets").shape, [sequences + 1]);
                assert_eq!(owner("query_offsets").bytes, ((sequences + 1) * 4).max(256));
                assert_eq!(owner("landing").bytes, sequences * 7 * 20);
                assert_eq!(owner("landing").domain, MlaDraftDomain::HostPinned);
                assert_eq!(owner("weight.layer.0.attention.uk").shape, [64, 512, 128]);
                assert_eq!(owner("weight.layer.0.attention.uv").shape, [64, 128, 512]);
                let uploads = owners.iter().filter(|o| o.staging_for.is_some() && !o.on_use).collect::<Vec<_>>();
                assert_eq!(uploads.len(), 9 + cfg.windows.len());
                for upload in uploads {
                    let destination = owner(upload.staging_for.as_deref().unwrap());
                    assert_eq!(upload.bytes, destination.bytes);
                    assert_eq!(upload.lifetime, MlaDraftLifetime::Library);
                    assert_eq!(upload.domain, MlaDraftDomain::HostPinned);
                }
                assert!(owner("taps.upload").on_use);
                assert!(!owners.iter().any(|o| o.id == "h.upload" || o.id.contains("embed") || o.id.contains("head_weights")));
                let sources = layout.packing_sources();
                if mode == Fp8Only {
                    assert_eq!(sources.len(), 4 * cfg.windows.len() + 2);
                    assert_eq!(sources[0].id, "weight.layer.0.attn_conv.source");
                    assert_eq!(sources[2].id, "weight.layer.0.gate_up.source");
                    assert_eq!(sources[2].bytes, 2 * 32768 * 7168 * 2);
                    assert_eq!(sources.last().unwrap().id, "weight.projection.source");
                    assert!(sources.iter().all(|source| source.dtype == MlaDraftDtype::Bytes
                        && source.shape == [source.bytes] && !owners.iter().any(|o| o.id == source.id)));
                    assert_eq!(layout.weights.max_load_staging, sources.iter().map(|s| s.bytes as u64).max().unwrap());
                    assert_eq!(owner("weight.layer.0.gate_up.values").bytes, 2 * 32768 * 7168);
                    assert_eq!(owner("weight.layer.0.gate_up.scales").shape, [65536, 56]);
                    assert!(!owners.iter().any(|o| o.id == "weight.layer.0.gate_up"));
                } else {
                    assert!(sources.is_empty());
                    assert_eq!(owner("weight.layer.0.gate_up").shape, [65536, 7168]);
                }
                assert_eq!(layout.sync_upload_bytes(), 5 * 7168 * 7168 * 2);
                assert_eq!(layout.loading_host_bytes(123).unwrap(), 123 + 4 * layout.sync_upload_bytes() as u64);
                assert!(layout.loading_host_bytes(u64::MAX).is_err());
                let resized = layout.with_pool_tokens(256).unwrap();
                for source in layout.owners() {
                    let next = resized.owners().iter().find(|next| next.id == source.id).unwrap();
                    if source.id == "layer.4.cache" {
                        assert_eq!(next.bytes, 256 * MlaCacheFormat::Fp8.record_bytes());
                        assert_eq!(next.shape, [next.bytes]);
                    } else { assert_eq!(next, source); }
                }
            }
        }
    }

    #[test]
    fn bounded_tail_gap_refusals_are_transactional_and_slack_is_exact() {
        let mut cfg = MlaDflashConfig::from_json(&config()).unwrap();
        for windows in [vec![Some(2048); 3], vec![Some(2048), Some(2049)], vec![Some(2048), None]] {
            cfg.windows = windows.clone();
            let mut state = MlaDraftCacheState::new(cfg.cache_layout(2, 262144, MlaCacheFormat::Fp8).unwrap()).unwrap();
            state.reset(0, 0).unwrap();
            state.reset(1, 513).unwrap();
            let before = state.clone();
            if windows.iter().all(|window| *window == Some(2048)) {
                state.commit_context(0, 6284, 2048).unwrap();
                assert_eq!(state.frontier(0).unwrap(), (0, 8332));
                assert_eq!(state.frontier(1).unwrap(), (513, 513));
                for (layer, layout) in state.layout.layers.iter().enumerate() {
                    let bound = layout.bind(0, 0).unwrap();
                    let retained = (6284..8332).map(|position| bound.row(position).unwrap())
                        .collect::<std::collections::BTreeSet<_>>();
                    let slack = state.window_slack(layer, 0).unwrap().into_iter().flatten()
                        .collect::<std::collections::BTreeSet<_>>();
                    assert_eq!(retained.len(), 2048);
                    assert!(retained.is_disjoint(&slack));
                    assert_eq!(retained.len() + slack.len(), layout.rows_per_slot());
                    let other_slack = state.window_slack(layer, 1).unwrap();
                    assert_eq!(other_slack, [layout.rows_per_slot()..2 * layout.rows_per_slot()]);
                }
                let committed = state.clone();
                for (start, rows) in [(8333, 2047), (8331, 2048), (8332, 0), (usize::MAX, 1), (262144, 2048)] {
                    assert!(state.commit_context(0, start, rows).is_err());
                    assert_eq!(state, committed);
                }
            } else {
                assert!(state.commit_context(0, 6284, 2048).is_err());
                assert_eq!(state, before);
            }
        }
    }

    #[test]
    fn generic_views_preserve_dflash_buffers_and_offsets_without_new_owners() {
        use crate::families::glm5::draft_representation::{GlmDraftCapacity, GlmDraftRepresentation};
        let cfg = MlaDflashConfig::from_json(&config()).unwrap();
        for sequences in [1usize, 8, 16, 32] {
            for sms in [170, 188] {
                for mode in [GlmDraftRepresentation::Bf16Only, GlmDraftRepresentation::Fp8Only] {
                    let layout = cfg.runtime_layout(mode, GlmDraftCapacity::new(sequences, sequences, 8).unwrap(),
                        262144, 262144, sms, 2, MlaCacheFormat::Fp8).unwrap();
                    assert!(!layout.owners().is_empty());
                    assert!(layout.owners().iter().any(|owner| owner.id == "weight.fc" || owner.id == "weight.fc.values"));
                    assert_eq!(layout.resident_bytes().unwrap(), layout.buffer_bytes().unwrap() + layout.weights().resident_bytes().unwrap());
                    assert_eq!(layout.device_buffers(), layout.buffers());
                    assert_eq!(layout.allocate_device_with(|b| Ok::<_, ()>(b.clone())).unwrap(),
                        layout.allocate_with(|b| Ok::<_, ()>(b.clone())).unwrap());
                    for sequence in 0..sequences {
                        let [input, output] = layout.block_views(sequence, 8).unwrap();
                        assert_eq!((input.backing.as_str(), input.offset, input.bytes),
                            ("conv", sequence * 8 * 7168 * 2, 8 * 7168 * 2));
                        assert_eq!((output.backing.as_str(), output.offset, output.bytes),
                            ("delta", sequence * 8 * 7168 * 2, 8 * 7168 * 2));
                        let mask = layout.mask_view(sequence).unwrap();
                        assert_eq!((mask.backing.as_str(), mask.offset, mask.bytes),
                            ("h", (sequence * 8 + 1) * 7168 * 2, 7 * 7168 * 2));
                    }
                    for first in (0..2048).step_by(64) {
                        let [input, slots, positions] = layout.context_views(first, 64).unwrap();
                        assert_eq!((input.offset, input.bytes), (first * 7168 * 2, 64 * 7168 * 2));
                        assert_eq!((slots.offset, slots.bytes), (first * 8, 64 * 8));
                        assert_eq!((positions.offset, positions.bytes), (first * 8, 64 * 8));
                    }
                    assert!(layout.block_views(sequences, 8).is_err());
                    assert!(layout.context_views(2048, 1).is_err());
                    assert!(layout.view("bad", "missing", 0, 1).is_err());
                    assert!(layout.view("overflow", "h", usize::MAX, 1).is_err());
                }
            }
        }
    }

    #[test]
    fn runtime_layout_chunks_native_arenas_and_preserves_per_slot_cache() {
        use crate::families::glm5::draft_representation::{GlmDraftCapacity, GlmDraftRepresentation::Fp8Only};
        let cfg = MlaDflashConfig::from_json(&config()).unwrap();
        let make = |slots, sequences, sms| cfg.runtime_layout(Fp8Only,
            GlmDraftCapacity::new(slots, sequences, 8).unwrap(), 262144, 262144, sms, 2, MlaCacheFormat::Bf16).unwrap();
        let eight = make(8, 8, 188);
        let sixteen = make(16, 16, 188);
        let bytes = |l: &MlaDraftRuntimeLayout, name| l.buffers().iter().find(|b| b.name == name).unwrap().bytes;
        for name in ["query", "q_residual", "attention", "lse", "native_scratch", "layer.4.cache"] {
            assert_eq!(bytes(&eight, name), bytes(&sixteen, name));
        }
        for name in ["h", "gate_up", "logits", "layer.0.cache", "layer.4.page_table"] {
            assert_eq!(2 * bytes(&eight, name), bytes(&sixteen, name));
        }
        assert_eq!(sixteen.capacity.max_batch_sequences, 16);
        assert_eq!(sixteen.cache.layers[0].block_batches(&(0..16).map(|slot|
            MlaDraftCacheRequest { slot, position: 4096, valid_from: 0 }).collect::<Vec<_>>()).unwrap().len(), 2);
        // 170 and 188 SMs can select the same discrete split count. A
        // single-SM fixture proves the query is not replaced by a card constant.
        let smaller = make(8, 8, 1);
        assert_ne!(bytes(&eight, "fp8_scratch"), bytes(&smaller, "fp8_scratch"));
        let mut count = 0;
        let failed = eight.allocate_with(|_| { count += 1; if count == 3 { Err("allocation") } else { Ok(()) } });
        assert_eq!(failed, Err("allocation"));
        assert_eq!(count, 3);
    }

    #[test]
    fn runtime_layout_checks_public_capacity_and_combined_fp8_packing_peak() {
        use crate::families::glm5::draft_representation::{GlmDraftCapacity, GlmDraftRepresentation::*};
        let mut cfg = MlaDflashConfig::from_json(&config()).unwrap();
        let capacity = GlmDraftCapacity::new(8, 8, 8).unwrap();
        let mut bad = capacity;
        bad.block_rows += 1;
        assert!(cfg.runtime_layout(Bf16Only, bad, 262144, 262144, 188, 0, MlaCacheFormat::Bf16).is_err());
        for (context, sms, mode) in [(0,188,0), (1048577,188,0), (262144,0,0), (262144,4097,0), (262144,188,3)] {
            assert!(cfg.runtime_layout(Fp8Only, capacity, context, 262144, sms, mode, MlaCacheFormat::Bf16).is_err());
        }
        cfg.intermediate = 32768;
        let layout = cfg.runtime_layout(Fp8Only, capacity, 262144, 262144, 188, 2, MlaCacheFormat::Bf16).unwrap();
        assert_eq!(layout.weights.max_load_staging, 2 * 2 * 32768 * 7168);
        assert!(layout.fp8_scratch.as_ref().unwrap().shapes.iter().any(|s| s.n == 65536 && s.k == 7168));
        cfg.mla.hidden = usize::MAX;
        assert!(cfg.runtime_layout(Bf16Only, capacity, 262144, 262144, 188, 0, MlaCacheFormat::Bf16).is_err());
    }

    #[test]
    fn precision_inventory_keeps_native_mla_and_codebooks_bf16_without_dual_copies() {
        use crate::families::glm5::draft_representation::GlmDraftRepresentation::{Bf16Only, Fp8Only};
        let cfg = MlaDflashConfig::from_json(&config()).unwrap();
        let bf16 = cfg.weight_layout(Bf16Only).unwrap();
        let fp8 = cfg.weight_layout(Fp8Only).unwrap();
        assert_eq!(bf16.resident_bytes().unwrap(), 5_036_945_408);
        assert_eq!((bf16.fp8_values, bf16.fp8_scales, bf16.max_load_staging), (0, 0, 0));
        assert_eq!(bf16.bf16_codebooks, 2 * cfg.vocab as u64 * cfg.selector_rank as u64 * 2);
        assert_eq!(fp8.bf16_codebooks, bf16.bf16_codebooks);
        assert_eq!(fp8.bf16_auxiliary, bf16.bf16_auxiliary);
        let native_mla = cfg.tensor_shapes().iter().filter(|(name, shape)|
            name.contains(".self_attn.") && shape.len() == 2)
            .map(|(_, shape)| shape.iter().product::<usize>() as u64 * 2).sum::<u64>();
        assert_eq!(fp8.bf16_values, native_mla);
        assert_eq!(bf16.bf16_values - fp8.bf16_values, 2 * fp8.fp8_values);
        assert_eq!(fp8.fp8_scales, fp8.fp8_values / 128 * 4);
        assert_eq!(fp8.max_load_staging, 2 * 7168 * 5 * 7168);
        assert_eq!(fp8.loading_peak_bytes().unwrap(), fp8.resident_bytes().unwrap() + fp8.max_load_staging);
        assert!(fp8.resident_bytes().unwrap() < bf16.resident_bytes().unwrap());
        assert!(fp8.loading_peak_bytes().unwrap() < bf16.resident_bytes().unwrap());
        let mut unsupported = cfg;
        unsupported.intermediate += 1;
        assert!(unsupported.weight_layout(Fp8Only).is_err());
        assert!(unsupported.weight_layout(Bf16Only).is_ok());
    }

    #[test]
    fn compact_runtime_charges_full_pages_windows_and_resize_identically() {
        use crate::families::glm5::draft_representation::{GlmDraftCapacity, GlmDraftRepresentation};
        let cfg = MlaDflashConfig::from_json(&config()).unwrap();
        let capacity = GlmDraftCapacity::new(8, 8, cfg.block).unwrap();
        for weights in [GlmDraftRepresentation::Bf16Only, GlmDraftRepresentation::Fp8Only] {
            let make = |format, tokens| cfg.runtime_layout(weights, capacity, 524288, tokens, 188, 0, format).unwrap();
            let bf16 = make(MlaCacheFormat::Bf16, 524288);
            let fp8 = make(MlaCacheFormat::Fp8, 524288);
            assert_eq!(fp8.cache_format(), MlaCacheFormat::Fp8);
            assert_eq!(fp8.cache().layers, bf16.cache().layers);
            assert_eq!(fp8.paged_cache().page_table_bytes, bf16.paged_cache().page_table_bytes);
            assert_eq!(fp8.program(), bf16.program());
            for (b, f) in bf16.buffers().iter().zip(fp8.buffers()) {
                assert_eq!(b.name, f.name);
                if b.name.ends_with(".cache") { assert_eq!(b.bytes / 1152 * 656, f.bytes); }
                else { assert_eq!(b.bytes, f.bytes); }
            }
            assert_eq!(fp8.window_mark_bytes().unwrap(), 4 * 4160 * 656);
            for tokens in [65, 262144, 1048576] {
                assert_eq!(fp8.with_pool_tokens(tokens).unwrap(), make(MlaCacheFormat::Fp8, tokens));
            }
        }
    }

    #[test]
    fn pooled_full_layers_charge_target_tokens_not_slots_times_context() {
        let cfg = MlaDflashConfig::from_json(&config()).unwrap();
        let tokens = 1 << 20;
        for slots in [1, 8, 20] {
            let layout = cfg.paged_cache_layout(slots, tokens, tokens, MlaCacheFormat::Bf16).unwrap();
            assert_eq!(layout.pages, tokens / 64);
            assert_eq!(layout.full_layers, 1);
            assert_eq!(layout.full_bytes, tokens as u64 * 1152);
            assert_eq!(layout.window_bytes, 4 * slots as u64 * 4160 * 1152);
            assert_eq!(layout.layer_bytes.iter().sum::<usize>() as u64, layout.cache_bytes().unwrap());
            assert_eq!(layout.demands(1).unwrap().iter().map(|d| d.bytes).sum::<u64>(),
                layout.cache_bytes().unwrap() + layout.page_table_bytes);
        }
        let layout = cfg.paged_cache_layout(8, tokens, tokens, MlaCacheFormat::Bf16).unwrap();
        assert_eq!(layout.cache_bytes().unwrap(), 1_361_313_792);
        assert_eq!(cfg.paged_cache_layout(8, 262144, tokens, MlaCacheFormat::Bf16).unwrap().full_bytes, layout.full_bytes);
        assert_eq!(cfg.paged_cache_layout(8, tokens, 65, MlaCacheFormat::Bf16).unwrap().full_bytes, 128 * 1152);
        assert!(cfg.paged_cache_layout(8, tokens, 0, MlaCacheFormat::Bf16).is_err());
        assert!(cfg.paged_cache_layout(8, tokens, usize::MAX, MlaCacheFormat::Bf16).is_err());
    }

    #[test]
    fn aggregate_pool_resize_preserves_native_layout_and_bounded_window_marks() {
        use crate::families::glm5::draft_representation::{GlmDraftCapacity, GlmDraftRepresentation};
        let cfg = MlaDflashConfig::from_json(&config()).unwrap();
        let capacity = GlmDraftCapacity::new(8, 8, cfg.block).unwrap();
        for representation in [GlmDraftRepresentation::Bf16Only, GlmDraftRepresentation::Fp8Only] {
            let initial = cfg.runtime_layout(representation, capacity, 1 << 20, 8 << 20, 188, 0, MlaCacheFormat::Bf16).unwrap();
            for tokens in [65, 262144, 1048576] {
                let resized = initial.with_pool_tokens(tokens).unwrap();
                let native = cfg.runtime_layout(representation, capacity, 1 << 20, tokens, 188, 0, MlaCacheFormat::Bf16).unwrap();
                assert_eq!(resized, native);
                assert_eq!(resized.window_mark_bytes().unwrap(), 4 * 4160 * 1152);
            }
            assert!(initial.with_pool_tokens(0).is_err());
            assert!(initial.with_pool_tokens(usize::MAX).is_err());
        }
    }

    #[test]
    fn admitted_cache_charges_private_blocks_and_virtual_page_slack_per_slot() {
        let cfg = MlaDflashConfig::from_json(&config()).unwrap();
        for (context, tables) in [(262144, 139520), (1048576, 532736)] {
            let layout = cfg.cache_layout(8, context, MlaCacheFormat::Bf16).unwrap();
            assert_eq!(layout.margin_bytes, 2_359_296);
            assert_eq!(layout.cache_bytes, cfg.cache_bytes(8, context).unwrap() + layout.margin_bytes);
            assert_eq!(layout.page_table_bytes, tables);
            let demands = layout.demands(1).unwrap();
            assert_eq!(demands.iter().map(|d| d.bytes).sum::<u64>(), layout.cache_bytes + tables);
            assert_eq!(demands[1].bytes, layout.margin_bytes);
            let movable = crate::placement::draft::DrafterDemand::itemized(demands, 1).movable(2).unwrap();
            assert_eq!(movable.allowed, [1]);
            assert_eq!(movable.conditional[0].demands.iter().map(|d| d.bytes).sum::<u64>(),
                layout.cache_bytes + tables);
            for layer in &layout.layers[..4] {
                assert_eq!((layer.rows_per_slot(), layer.table_width(), layer.slots()), (4160, 66, 8));
            }
            let full = &layout.layers[4];
            assert_eq!(full.rows_per_slot(), context);
            assert_eq!(full.table_width(), context / 64);
            assert!(full.bind(7, 0).unwrap().block_view(context - cfg.block, cfg.block).is_ok());
            assert!(full.bind(7, 0).unwrap().block_view(context - cfg.block + 1, cfg.block).is_err());
            assert!(full.bind(8, 0).is_err());
            assert!(full.bind(0, 0).unwrap().row(context).is_err());
            assert!(full.bind(0, 0).unwrap().block_view(0, 0).is_err());
            assert!(full.bind(0, 0).unwrap().block_view(0, cfg.block + 1).is_err());
            assert!(full.bind(0, 0).unwrap().block_view(usize::MAX, cfg.block).is_err());
        }
        assert!(cfg.cache_layout(0, 262144, MlaCacheFormat::Bf16).is_err());
        assert!(cfg.cache_layout(8, 1048577, MlaCacheFormat::Bf16).is_err());
        assert!(cfg.cache_layout(usize::MAX, 1048576, MlaCacheFormat::Bf16).is_err());
        let small = cfg.cache_layout(2, 7, MlaCacheFormat::Bf16).unwrap();
        assert!(small.layers.iter().all(|l| l.rows_per_slot() == 64));
        assert!(small.layers[0].bind(0, 0).unwrap().block_view(0, 7).is_ok());
        assert!(small.layers[0].bind(0, 0).unwrap().block_view(0, 8).is_err());
    }

    #[test]
    fn full_block_at_window_edge_preserves_live_rows_across_wraps_and_slots() {
        let cfg = MlaDflashConfig::from_json(&config()).unwrap();
        let layout = cfg.cache_layout(3, 262144, MlaCacheFormat::Bf16).unwrap();
        let layer = &layout.layers[0];
        let capacity = layer.rows_per_slot();
        let mut saw_repeated_page = false;
        let tag = |slot: usize, pos: usize| ((slot as u64 + 1) << 32) | pos as u64;
        for base in [0, 4096 - 64, capacity - 64, 2 * capacity - 64, 3 * capacity - 64] {
            for anchor in base..base + 128 {
                let begin = anchor.saturating_sub(4096);
                for slot in 0..3 {
                    let mut ring = vec![u64::MAX; 3 * capacity];
                    for neighbor in 0..3 {
                        let bound = layer.bind(neighbor, 0).unwrap();
                        for pos in begin..anchor {
                            ring[bound.row(pos).unwrap()] = tag(neighbor, pos);
                        }
                    }
                    let bound = layer.bind(slot, 0).unwrap();
                    // Two rejected proposals at the same frontier must both
                    // leave every committed row and every neighboring slot exact.
                    for attempt in 0..2 {
                        for pos in anchor..anchor + cfg.block {
                            ring[bound.row(pos).unwrap()] = (attempt as u64) << 48 | tag(slot, pos);
                        }
                        for neighbor in 0..3 {
                            let neighbor_bound = layer.bind(neighbor, 0).unwrap();
                            for pos in begin..anchor {
                                assert_eq!(ring[neighbor_bound.row(pos).unwrap()], tag(neighbor, pos),
                                    "anchor {anchor}, slot {slot}, neighbor {neighbor}, live row {pos}");
                            }
                        }
                        let view = bound.block_view(anchor, cfg.block).unwrap();
                        assert!(view.pages.len() <= layer.table_width());
                        assert_eq!(view.cache_seqlen, anchor + cfg.block - view.base_position);
                        saw_repeated_page |= view.pages.first() == view.pages.last() && view.pages.len() > 1;
                        for pos in begin..anchor + cfg.block {
                            let relative = pos - view.base_position;
                            let row = view.pages[relative / 64] as usize * 64 + relative % 64;
                            assert_eq!(row, bound.row(pos).unwrap());
                        }
                    }
                }
            }
        }
        assert!(saw_repeated_page, "unaligned wrapped spans exercise virtual page duplication");
    }

    #[test]
    fn cold_prefix_restore_rebases_storage_without_exposing_stale_prefix_rows() {
        let cfg = MlaDflashConfig::from_json(&config()).unwrap();
        let layout = cfg.cache_layout(3, 262144, MlaCacheFormat::Bf16).unwrap();
        for layer in [&layout.layers[0], &layout.layers[4]] {
            let capacity = layer.rows_per_slot();
            assert!(layer.bind(0, 262144).is_err());
            assert!(layer.bind(3, 17).is_err());
            for origin in (0..64).chain([4095, 4096, 4097, 262144 - 16]) {
                let bound = layer.bind(1, origin).unwrap();
                assert_eq!(bound.valid_from(), origin);
                assert_eq!(bound.row(origin).unwrap(), capacity);
                if origin > 0 {
                    assert!(bound.row(origin - 1).is_err());
                    assert!(bound.block_view(origin - 1, cfg.block).is_err());
                }
                // The full layer begins cold at offset zero, not at the
                // absolute origin's offset inside a pre-existing page.
                let cold = bound.block_view(origin, cfg.block).unwrap();
                assert_eq!(cold.base_position, origin);
                assert_eq!(cold.cache_seqlen, cfg.block);
                assert_eq!(cold.pages, [capacity as i32 / 64]);
                let relative_anchors = if layer.window.is_some() {
                    vec![0, 1, 63, 64, 4095, 4096, 4097, 4159, 4160, 8321, 12481]
                } else { vec![0, 1, 63, 64, 127] };
                for offset in relative_anchors {
                    let anchor = origin + offset;
                    if anchor + cfg.block > 262144 { continue; }
                    let begin = layer.window.map_or(origin, |w| anchor.saturating_sub(w).max(origin));
                    let mut ring = vec![usize::MAX; capacity * 3];
                    for position in begin..anchor {
                        ring[bound.row(position).unwrap()] = position;
                    }
                    let view = bound.block_view(anchor, cfg.block).unwrap();
                    assert_eq!((view.base_position - origin) % 64, 0);
                    assert!(view.base_position >= origin);
                    assert_eq!(view.cache_seqlen, anchor + cfg.block - view.base_position);
                    for attempt in 0..2 {
                        for position in anchor..anchor + cfg.block {
                            ring[bound.row(position).unwrap()] = position + attempt * 262144;
                        }
                        for position in begin..anchor {
                            assert_eq!(ring[bound.row(position).unwrap()], position);
                        }
                        // Exactly the attention-visible rows map to this
                        // continuation; rejected proposals preserve live history.
                        for position in begin..anchor + cfg.block {
                            let relative = position - view.base_position;
                            let physical = view.pages[relative / 64] as usize * 64 + relative % 64;
                            assert_eq!(physical, bound.row(position).unwrap());
                            assert_eq!((anchor as i64) - position as i64,
                                ((anchor - origin) as i64) - (position - origin) as i64);
                            assert_ne!(ring[physical], usize::MAX);
                        }
                        assert!(ring[..capacity].iter().all(|&row| row == usize::MAX));
                        assert!(ring[2 * capacity..].iter().all(|&row| row == usize::MAX));
                    }
                }
                // Slot reuse or another cold restore creates a new binding;
                // an old binding keeps its original coordinate system intact.
                let replacement = layer.bind(1, origin + 1).unwrap();
                assert_eq!(replacement.row(origin + 1).unwrap(), capacity);
                assert_eq!(bound.row(origin + 1).unwrap(), capacity + 1);
            }
        }
    }

    #[test]
    fn restored_slot_matches_clean_cold_start_origin_rows_and_reachable_state() {
        let cfg = MlaDflashConfig::from_json(&config()).unwrap();
        for context in [262144, 1048576] {
            let layout = cfg.cache_layout(3, context, MlaCacheFormat::Bf16).unwrap();
            for origin in (0..64).chain([4095, 4096, 4097, context - 16]) {
                let mut restored = MlaDraftCacheState::new(layout.clone()).unwrap();
                restored.reset(1, 0).unwrap();
                restored.commit_context(1, 0, 128).unwrap();
                restored.reset(2, 17).unwrap();
                restored.commit_context(2, 17, 19).unwrap();
                // Previous occupant has both committed and rejected proposal rows.
                // Unwritten backing bytes stay poisoned rather than being cleared.
                let mut reused = std::collections::BTreeMap::new();
                for (index, layer) in layout.layers.iter().enumerate() {
                    let old = layer.bind(1, 0).unwrap();
                    for position in 0..136 {
                        reused.insert((index, old.row(position).unwrap()), usize::MAX);
                    }
                }
                restored.discard(1).unwrap();
                assert!(restored.frontier(1).is_err());
                restored.reset(1, origin).unwrap();
                let mut clean = MlaDraftCacheState::new(layout.clone()).unwrap();
                clean.reset(1, origin).unwrap();
                assert_eq!(restored.frontier(1).unwrap(), (origin, origin));
                assert_eq!(restored.slots[1], clean.slots[1]);
                assert_eq!(restored.frontier(2).unwrap(), (17, 36));
                let cold_request = MlaDraftCacheRequest { slot: 1, position: origin, valid_from: origin };
                for index in 0..layout.layers.len() {
                    let batches = restored.block_batches(index, &[cold_request]).unwrap();
                    assert_eq!(batches, clean.block_batches(index, &[cold_request]).unwrap());
                    assert_eq!(batches[0].lengths, [cfg.block as i32]);
                    assert_eq!(batches[0].positions, (origin..origin + cfg.block).map(|p| p as i64).collect::<Vec<_>>());
                    let view = layout.layers[index].bind(1, origin).unwrap().block_view(origin, cfg.block).unwrap();
                    assert_eq!(view.base_position, origin);
                    // Every cold attention-visible record is produced before
                    // use; old poisoned rows outside this span are unreachable.
                    let mut cold_reused = reused.clone();
                    let mut cold_fresh = std::collections::BTreeMap::new();
                    for (&row, &position) in batches[0].slots.iter().zip(&batches[0].positions) {
                        cold_reused.insert((index, row as usize), position as usize);
                        cold_fresh.insert((index, row as usize), position as usize);
                    }
                    for offset in 0..view.cache_seqlen {
                        let physical = view.pages[offset / 64] as usize * 64 + offset % 64;
                        assert_eq!(cold_reused.get(&(index, physical)), cold_fresh.get(&(index, physical)));
                        assert_eq!(cold_fresh.get(&(index, physical)), Some(&(origin + offset)));
                    }
                }
                let mut fresh = std::collections::BTreeMap::new();
                let committed = (context - origin - cfg.block).min(4201);
                for (index, layer) in layout.layers.iter().enumerate() {
                    if committed > 0 {
                        let rows = restored.context_rows(index, 1, origin, committed).unwrap();
                        assert_eq!(rows, clean.context_rows(index, 1, origin, committed).unwrap());
                        for (offset, row) in rows.into_iter().enumerate() {
                            reused.insert((index, row as usize), origin + offset);
                            fresh.insert((index, row as usize), origin + offset);
                        }
                    }
                    let bound = layer.bind(1, origin).unwrap();
                    if origin > 0 { assert!(bound.row(origin - 1).is_err()); }
                }
                if committed > 0 {
                    restored.commit_context(1, origin, committed).unwrap();
                    clean.commit_context(1, origin, committed).unwrap();
                }
                let anchor = origin + committed;
                let request = MlaDraftCacheRequest { slot: 1, position: anchor, valid_from: origin };
                for (index, layer) in layout.layers.iter().enumerate() {
                    let batches = restored.block_batches(index, &[request]).unwrap();
                    assert_eq!(batches, clean.block_batches(index, &[request]).unwrap());
                    let before = restored.clone();
                    // Native production writes the whole block before attention.
                    for (&row, &position) in batches[0].slots.iter().zip(&batches[0].positions) {
                        reused.insert((index, row as usize), position as usize);
                        fresh.insert((index, row as usize), position as usize);
                    }
                    let view = layer.bind(1, origin).unwrap().block_view(anchor, cfg.block).unwrap();
                    for position in view.base_position..anchor + cfg.block {
                        let offset = position - view.base_position;
                        let physical = view.pages[offset / 64] as usize * 64 + offset % 64;
                        assert_eq!(reused.get(&(index, physical)), fresh.get(&(index, physical)));
                        assert_eq!(fresh.get(&(index, physical)), Some(&position));
                    }
                    assert_eq!(restored, before, "rejected proposal cannot publish a frontier");
                }
                let before = restored.clone();
                assert!(restored.reset(1, context).is_err());
                assert_eq!(restored, before, "bad rebind must preserve the live slot");
                let wrong = MlaDraftCacheRequest { valid_from: origin + 1, ..request };
                assert!(restored.block_batches(0, &[wrong]).is_err());
                assert!(restored.context_rows(0, 1, anchor + 1, 1).is_err());
            }
        }
    }

    #[test]
    fn native_block_batches_chunk_concurrency_without_changing_positions_or_cache_rows() {
        let cfg = MlaDflashConfig::from_json(&config()).unwrap();
        let layout = cfg.cache_layout(16, 262144, MlaCacheFormat::Bf16).unwrap();
        for layer in [&layout.layers[0], &layout.layers[4]] {
            for origin in 0..64 {
                let requests = (0..16).map(|slot| MlaDraftCacheRequest { slot,
                    valid_from: origin + slot * 17, position: origin + slot * 17 + slot * 4097 })
                    .collect::<Vec<_>>();
                let batches = layer.block_batches(&requests).unwrap();
                assert_eq!(batches.len(), 2);
                for batch in batches {
                    assert_eq!(batch.positions.len(), 64);
                    assert_eq!(batch.slots.len(), 64);
                    assert_eq!(batch.query_offsets, [0,8,16,24,32,40,48,56,64]);
                    assert_eq!(batch.lengths.len(), 8);
                    assert_eq!(batch.page_table.len(), 8 * batch.page_table_width);
                    for local in 0..8 {
                        let request = requests[batch.first_sequence + local];
                        let bound = layer.bind(request.slot, request.valid_from).unwrap();
                        let view = bound.block_view(request.position, cfg.block).unwrap();
                        assert_eq!(batch.lengths[local] as usize, view.cache_seqlen);
                        let pages = &batch.page_table[local * batch.page_table_width..][..batch.page_table_width];
                        assert_eq!(&pages[..view.pages.len()], view.pages);
                        assert!(pages[view.pages.len()..].iter().all(|&page| page == 0));
                        for row in 0..cfg.block {
                            let index = local * cfg.block + row;
                            let absolute = request.position + row;
                            assert_eq!(batch.positions[index] as usize, absolute);
                            assert_eq!(batch.slots[index] as usize, bound.row(absolute).unwrap());
                            let relative = absolute - view.base_position;
                            assert_eq!(pages[relative / 64] as usize * 64 + relative % 64,
                                batch.slots[index] as usize);
                        }
                    }
                }
                let partial = layer.block_batches(&requests[..9]).unwrap();
                assert_eq!(partial[1].first_sequence, 8);
                assert_eq!(partial[1].query_offsets, [0,8]);
                assert_eq!(partial[1].positions.len(), 8);
            }
        }
    }

    #[test]
    fn native_block_batches_reject_bad_later_chunks_before_returning_metadata() {
        let cfg = MlaDflashConfig::from_json(&config()).unwrap();
        let layer = &cfg.cache_layout(16, 262144, MlaCacheFormat::Bf16).unwrap().layers[0];
        let requests = (0..16).map(|slot| MlaDraftCacheRequest { slot, position: 4096, valid_from: 17 })
            .collect::<Vec<_>>();
        for defect in 0..5 {
            let mut bad = requests.clone();
            match defect {
                0 => bad[15].slot = 0,
                1 => bad[15].slot = 16,
                2 => bad[15].valid_from = 4097,
                3 => bad[15].position = 262144 - cfg.block + 1,
                _ => bad[15].position = usize::MAX,
            }
            assert!(layer.block_batches(&bad).is_err());
        }
        assert!(layer.block_batches(&[]).is_err());
        let mut too_many = requests.clone(); too_many.push(requests[0]);
        assert!(layer.block_batches(&too_many).is_err());
    }

    #[test]
    fn mla_config_refuses_unsupported_native_contracts() {
        for (field, value) in [
            ("mla_use_output_gate", json!(true)), ("is_causal", json!(true)),
            ("fc_norm", json!(true)), ("attention_bias", json!(true)),
            ("rms_norm_eps", json!(1e-6)), ("num_attention_heads", json!(96)),
            ("qk_rope_head_dim", json!(128)), ("sliding_window", json!(2048)),
            ("num_hidden_layers", json!(6)), ("vocab_size", json!(0)),
            ("intermediate_size", json!(usize::MAX)), ("max_position_embeddings", json!(0)),
        ] {
            let mut raw = config(); raw[field] = value;
            assert!(MlaDflashConfig::from_json(&raw).is_err(), "{field}");
        }
        for (field, value) in [("target_layer_ids", json!([19,19])),
            ("target_layer_ids", json!([37,19])), ("target_layer_ids", json!([93])),
            ("mask_token_id", json!(163840)), ("conv_group_size", json!(0)),
            ("block_size", json!(64)), ("attention_mode", json!("gqa"))] {
            let mut raw = config(); raw["dflash_config"][field] = value;
            assert!(MlaDflashConfig::from_json(&raw).is_err(), "{field}");
        }
    }
}
