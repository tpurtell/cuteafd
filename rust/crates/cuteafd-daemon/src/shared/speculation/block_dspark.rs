//! Shared Qwen3-backbone block dSpark runtime; target lifecycles stay in bindings.
use crate::shared::speculation::dflash::{ContextRow, Draft, DraftSeq, HeadLaunch, ReplayDrafter, ReplayMode, TargetHead,
    TAP_ROWS};
use crate::shared::speculation::dflash::DraftHead;
use cuteafd_loader::families::glm5::draft_representation::GlmDraftRepresentation;
use crate::shared::fp8_linear;
use crate::shared::speculation::weights::{Load, Source, Weight, WeightAllocator};
use crate::shared::memory::DeviceAllocation;
use crate::shared::token_io::TokenEmbedding;
use anyhow::{ensure, Context, Result};
use cuteafd_ffi::programs::{VocabularyHead, VOCABULARY_HEAD_WORKSPACE};
use cuteafd_ffi::{CuteafdDeviceBuffer, NativeLibrary};
use cuteafd_loader::read_safetensors_metadata;
use std::cell::{Cell, RefCell};
use std::collections::HashMap;
use std::ffi::c_void;
use std::path::Path;
use crate::shared::memory::Tag;

use crate::shared::memory::{device, receipts};
use std::rc::Rc;

/// Byte-exact marks for a windowed drafter. The mark stores every physical
/// K/V plane and its canonical origin/frontier; restore never reruns arithmetic.
pub(crate) struct WindowSnapshots<'a> {
    copies: crate::shared::prefix::PrefixCopies<'a>,
    arena: Rc<device::Allocation<'a>>,
    planes: Vec<usize>,
    stride: usize,
    marks: usize,
}

impl<'a> WindowSnapshots<'a> {
    pub fn bytes(planes: &[usize], marks: usize) -> Result<(usize, usize)> {
        ensure!(marks > 0 && !planes.is_empty() && planes.iter().all(|&n| n > 0 && n % 16 == 0),
            "window snapshots require aligned nonempty planes and marks");
        let payload = planes.iter().try_fold(16usize, |sum, &n| sum.checked_add(n))
            .context("window snapshot payload overflow")?;
        let stride = payload.checked_add(255).context("window snapshot alignment overflow")? / 256 * 256;
        Ok((stride, stride.checked_mul(marks).context("window snapshot arena overflow")?))
    }

    /// The caller admits the exact arena returned by `bytes` before construction.
    pub fn new(stream: &device::Stream<'a>, planes: Vec<usize>, marks: usize) -> Result<Self> {
        let (stride, bytes) = Self::bytes(&planes, marks)?;
        let arena = Rc::new(device::Allocation::new(stream.device, bytes,
            Tag::prefix("shared/draft-window-snapshots"))?);
        let copies = crate::shared::prefix::PrefixCopies::new(stream.device.library,
            vec![vec![crate::shared::prefix::Reader { device: stream.device.id, stream: stream.raw }]])?;
        Ok(Self { copies, arena, planes, stride, marks })
    }

    pub fn backing(&self) -> Rc<device::Allocation<'a>> { Rc::clone(&self.arena) }
    pub fn mark_bytes(&self) -> usize { self.stride }
    pub fn inventory(&self, inventory: &mut receipts::Inventory<'_>, name: &str) -> Result<()> {
        self.arena.inventory(inventory, name)?;
        self.copies.allocation_receipts(inventory, &format!("{name}.copies"))?;
        inventory.host(&format!("{name}.planes"), &self.planes)
    }

    pub fn mark(&self, mark: usize) -> Result<CuteafdDeviceBuffer> {
        ensure!(mark < self.marks, "window snapshot mark outside admission");
        Ok(CuteafdDeviceBuffer { ptr: self.arena.buffer.ptr.wrapping_byte_add(mark * self.stride),
            bytes: self.stride, ..self.arena.buffer })
    }

    fn parts(&self, mark: usize, live: &[CuteafdDeviceBuffer], capture: bool)
        -> Result<Vec<(usize, CuteafdDeviceBuffer, CuteafdDeviceBuffer)>> {
        ensure!(live.len() == self.planes.len(), "window snapshot plane inventory differs");
        let saved = self.mark(mark)?;
        for (index, plane) in live.iter().enumerate() {
            let end = (plane.ptr as usize).checked_add(plane.bytes).context("window live plane overflow")?;
            ensure!(live[..index].iter().all(|other| (other.ptr as usize).checked_add(other.bytes)
                .is_some_and(|other_end| end <= other.ptr as usize || other_end <= plane.ptr as usize)),
                "window snapshot live planes overlap");
        }
        let mut offset = 16;
        let mut copies = Vec::with_capacity(live.len());
        for (&live, &bytes) in live.iter().zip(&self.planes) {
            ensure!(live.device_id == saved.device_id && !live.ptr.is_null() && live.bytes == bytes
                && live.ptr as usize % 16 == 0, "window snapshot live plane differs from admission");
            let part = CuteafdDeviceBuffer { ptr: saved.ptr.wrapping_byte_add(offset), bytes, ..saved };
            let a = self.arena.buffer.ptr as usize;
            let b = live.ptr as usize;
            ensure!(a.checked_add(self.arena.buffer.bytes).is_some_and(|end| end <= b)
                || b.checked_add(bytes).is_some_and(|end| end <= a),
                "window snapshot aliases live state");
            copies.push(if capture { (0, part, live) } else { (0, live, part) });
            offset += bytes;
        }
        Ok(copies)
    }

    /// # Safety
    /// Live planes belong to the ordered reader stream, do not overlap, and
    /// their owners survive `drain` and all subsequent restoration copies.
    pub unsafe fn capture(&self, mark: usize, live: &[CuteafdDeviceBuffer], origin: usize, frontier: usize) -> Result<()> {
        ensure!(origin <= frontier, "window snapshot frontier precedes origin");
        let copies = self.parts(mark, live, true)?;
        let mut metadata = [0u8; 16];
        metadata[..8].copy_from_slice(&u64::try_from(origin)?.to_le_bytes());
        metadata[8..].copy_from_slice(&u64::try_from(frontier)?.to_le_bytes());
        let saved = self.mark(mark)?;
        self.copies.ordered(0, |stream| {
            // SAFETY: the ordered uploader retains its host staging until the
            // copy stream retires; mark storage survives every queued consumer.
            unsafe { self.arena.device.library.copy_h2d_ordered(
                CuteafdDeviceBuffer { bytes: 16, ..saved }, &metadata, stream) }
        })?;
        // SAFETY: the caller retains all live planes; parts validated extents.
        unsafe { self.copies.copy_d2d(&copies) }
    }

    /// # Safety
    /// The destination planes satisfy `capture`'s ownership/stream contract.
    /// The mark is initialized by capture or a completed host-tier restore.
    pub unsafe fn restore(&self, mark: usize, live: &[CuteafdDeviceBuffer], frontier: usize)
        -> Result<(usize, usize)> {
        let copies = self.parts(mark, live, false)?;
        self.copies.drain()?;
        let saved = self.mark(mark)?;
        let mut metadata = [0u8; 16];
        self.arena.device.run(|| self.arena.device.library.copy_d2h(&mut metadata,
            CuteafdDeviceBuffer { bytes: 16, ..saved }))?;
        let origin = usize::try_from(u64::from_le_bytes(metadata[..8].try_into()?))?;
        let end = usize::try_from(u64::from_le_bytes(metadata[8..].try_into()?))?;
        ensure!(origin <= end && end == frontier, "window snapshot metadata/frontier differs");
        // SAFETY: caller retains the initialized mark and destination planes.
        unsafe { self.copies.copy_d2d(&copies)?; }
        Ok((origin, end))
    }

    pub fn drain(&self) -> Result<()> { self.copies.drain() }
}

type Dev<'a> = DeviceAllocation<'a>;

struct PagedContext<'a> {
    layout: cuteafd_loader::formats::dflash::DraftCacheLayout,
    state: RefCell<PagedState>,
    tables: Dev<'a>,
}

struct PagedState {
    pool: cuteafd_engine::prefix::DraftPagePool,
    bindings: Vec<Option<cuteafd_engine::prefix::DraftBinding>>,
}

impl PagedState {
    fn binding(&self, slot: usize) -> Result<cuteafd_engine::prefix::DraftBinding> {
        self.bindings.get(slot).copied().flatten().context("unbound block dSpark slot")
    }

    fn discard(&mut self, slot: usize) -> Result<()> {
        let binding = self.bindings.get_mut(slot).context("block dSpark slot outside capacity")?.take();
        if let Some(binding) = binding { self.pool.release(binding)?; }
        Ok(())
    }
}

/// Error paths must retire producers before returning their physical destinations.
struct PageLease<'c, 'a> {
    drafter: &'c DsparkDrafter<'a>,
    reservations: Vec<cuteafd_engine::prefix::DraftReservation>,
}

impl PageLease<'_, '_> {
    fn commit(mut self) -> Result<()> {
        if let Some(context) = &self.drafter.paged {
            let mut state = context.state.borrow_mut();
            while let Some(&reservation) = self.reservations.last() {
                state.pool.commit(reservation, reservation.end() - reservation.start())?;
                self.reservations.pop();
            }
        }
        Ok(())
    }
}

impl Drop for PageLease<'_, '_> {
    fn drop(&mut self) {
        if self.reservations.is_empty() { return; }
        // SAFETY: leases keep all destinations owned until this producer stream retires.
        if unsafe { self.drafter.library.cuda_stream_synchronize(self.drafter.stream) }.is_err() {
            return; // Quarantine on failed retirement rather than recycle live storage.
        }
        if let Some(context) = &self.drafter.paged {
            let mut state = context.state.borrow_mut();
            for reservation in self.reservations.drain(..) { let _ = state.pool.commit(reservation, 0); }
        }
    }
}

/// Most sequences one draft step takes (the Markov kernel's limit).
pub(crate) const MAX_SEQUENCES: usize = cuteafd_loader::formats::block_dspark::MAX_SEQUENCES;

/// Whether `snapshot` holds a Speculators dSpark checkpoint.
pub(crate) fn is_dspark(snapshot: &Path) -> Result<bool> {
    let path = snapshot.join("config.json");
    let config: serde_json::Value = serde_json::from_slice(&std::fs::read(&path)
        .with_context(|| format!("reading drafter {}", path.display()))?)
        .with_context(|| format!("parsing drafter {}", path.display()))?;
    Ok(config["speculators_model_type"] == "dspark" || config["architectures"].as_array()
        .is_some_and(|a| a.iter().any(|v| v == "DSparkDraftModel")))
}

/// The drafter geometry the kernels are written for, read from `config.json`.
#[derive(Debug, Clone)]
pub(crate) struct DsparkConfig {
    pub backbone: cuteafd_loader::formats::block_dspark::DsparkBackbone,
    pub hidden: usize,
    pub intermediate: usize,
    pub layers: usize,
    pub heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub eps: f32,
    pub theta: f32,
    /// Query rows per draft and drafted tokens (sample_from_anchor).
    pub block: usize,
    pub mask_token: u32,
    pub rank: usize,
    /// Target layers whose outputs are tapped (aux ids - 1).
    pub taps: Vec<usize>,
    pub vocab: usize,
    /// Causal attention inside the block.
    pub causal: bool,
    pub window: Option<usize>,
    pub rotary: Option<cuteafd_loader::formats::block_dspark::DsparkRotaryPlan>,
}

impl DsparkConfig {
    pub fn read(snapshot: &Path) -> Result<Self> {
        use cuteafd_loader::formats::block_dspark::{BlockDsparkConfig, DsparkRope};
        let cfg = BlockDsparkConfig::read(snapshot)?;
        ensure!(matches!(cfg.head_dim, 64 | 128), "dSpark kernels take 64/128-wide heads");
        ensure!(cfg.rank == 256, "dSpark kernels take a vanilla rank-256 Markov head");
        let rope_theta = match cfg.rope {
            DsparkRope::Default { rope_theta } | DsparkRope::Yarn { rope_theta, .. } => rope_theta,
        };
        let rotary = if cfg.head_dim == 64 {
            ensure!(matches!(cfg.rope, DsparkRope::Default { .. }), "64-wide dSpark requires default RoPE");
            None
        } else { Some(cfg.rope.plan(cfg.head_dim)?) };
        Ok(Self { backbone: cfg.backbone, hidden: cfg.hidden, intermediate: cfg.intermediate, layers: cfg.layers,
            heads: cfg.heads, kv_heads: cfg.kv_heads, head_dim: cfg.head_dim,
            eps: cfg.eps as f32, theta: rope_theta as f32, block: cfg.block,
            mask_token: cfg.mask_token, rank: cfg.rank, taps: cfg.taps, vocab: cfg.vocab, causal: cfg.causal,
            window: cfg.window, rotary })
    }

    fn context_row(&self, slot: usize, position: usize, slots: usize, capacity: usize) -> Result<i32> {
        ensure!(slot < slots && capacity > 0, "dSpark context slot/capacity outside admission");
        ensure!(self.window.is_some() || position < capacity,
            "full-context dSpark position {position} exceeds admitted {capacity}");
        let row = slot.checked_mul(capacity).and_then(|base| base.checked_add(position % capacity))
            .context("dSpark context row overflow")?;
        Ok(i32::try_from(row)?)
    }

    /// Drafted tokens per draft step (every block row predicts one).
    pub fn drafts(&self) -> usize {
        self.block
    }

    pub(crate) fn kv_width(&self) -> usize {
        self.kv_heads * self.head_dim
    }

    fn q_width(&self) -> usize {
        self.heads * self.head_dim
    }

    pub(crate) fn qkv_width(&self) -> usize {
        self.q_width() + 2 * self.kv_width()
    }
}

fn validate_target_taps(taps: &[super::target::HiddenTap], layers: &[usize], hidden: usize,
    stream: *mut c_void, end: usize, n: usize) -> Result<()> {
    ensure!(taps.len() == layers.len(), "target tap count differs from draft checkpoint");
    for (tap, &layer) in taps.iter().zip(layers) {
        ensure!(tap.layer == layer, "target tap order differs from draft checkpoint");
        ensure!(tap.hidden == hidden && tap.stride >= hidden,
            "target tap hidden width or stride differs from draft checkpoint");
        ensure!(end <= tap.rows, "target tap rows exceed producer span");
        ensure!(tap.stream == stream, "target taps require the drafter's producer stream");
        ensure!(n == 0 || !tap.ptr.is_null(), "target tap has a null device pointer");
        tap.stride.checked_mul(2).context("target tap pitch overflow")?;
        tap.rows.saturating_sub(1).checked_mul(tap.stride).and_then(|v| v.checked_add(hidden))
            .and_then(|v| v.checked_mul(2)).context("target tap byte span overflow")?;
    }
    Ok(())
}

/// The loader's single drafter recipe: the runtime load and admission bind the same owners.
fn weight_recipe<T, M>(snapshot: &Path, tensor: impl FnMut(&str, &[usize]) -> Result<T>,
    matrix: impl FnMut(&[(&str, usize)], usize) -> Result<M>)
    -> Result<cuteafd_loader::formats::block_dspark::BlockDsparkWeights<T, M>> {
    let cfg = cuteafd_loader::formats::block_dspark::BlockDsparkConfig::read(snapshot)?;
    cuteafd_loader::formats::block_dspark::block_dspark_weight_recipe(&cfg, tensor, matrix)
}

struct DraftLayer<'a> {
    input_norm: Dev<'a>,
    post_norm: Dev<'a>,
    /// q | k | v rows; the context update reads the k | v rows.
    qkv: Weight<'a>,
    q_norm: Dev<'a>,
    k_norm: Dev<'a>,
    o: Weight<'a>,
    gate_up: Weight<'a>,
    down: Weight<'a>,
    k_ring: Dev<'a>,
    v_ring: Dev<'a>,
}

struct Workspace<'a> {
    sequences: usize,
    h: Dev<'a>,
    n: Dev<'a>,
    qkv: Dev<'a>,
    q: Dev<'a>,
    k: Dev<'a>,
    v: Dev<'a>,
    attn: Dev<'a>,
    delta: Dev<'a>,
    gate_up: Dev<'a>,
    act: Dev<'a>,
    logits: Dev<'a>,
    anchors: Dev<'a>,
    ids: Dev<'a>,
    tokens: Dev<'a>,
    confidence: Dev<'a>,
    /// Pinned landing of `tokens` then `confidence`: both download behind the
    /// round's one host wait.
    landing: crate::shared::memory::HostAllocation<'a>,
    positions: Dev<'a>,
    tables: Dev<'a>,
    attention_workspace: Dev<'a>,
    markov_workspace: Dev<'a>,
    head: Option<VocabularyHead<'a>>,
    _head_workspace: Dev<'a>,
}

/// The target head inside a draft step.
#[derive(Clone, Copy)]
enum HeadRef<'h> {
    Bf16(*const c_void),
    Launch(&'h HeadLaunch<'h>),
}

#[derive(Clone, Copy)]
enum DraftInput<'r, 'e> {
    Rows(&'r [u8]),
    Table(&'r TokenEmbedding<'e>),
}

/// The tensors a dSpark draft reads (everything but the checkpoint's copies
/// of the target's embedding and LM head).
fn wanted(name: &str) -> bool {
    name != "embed_tokens.weight" && name != "lm_head.weight"
}

/// Header-only admission before any native allocation: the config fits the
/// target and every tensor the drafter reads is BF16 of its shape.
pub(crate) fn check_checkpoint(snapshot: &Path, hidden: usize, vocab: usize, layers: usize) -> Result<()> {
    let cfg = DsparkConfig::read(snapshot)?;
    ensure!(cfg.hidden == hidden && cfg.vocab == vocab && cfg.taps.iter().all(|&l| l < layers),
        "the dSpark drafter (hidden {}, vocab {}, taps {:?}) does not fit this target", cfg.hidden, cfg.vocab,
        cfg.taps);
    let tensors: HashMap<String, (cuteafd_core::DType, Vec<usize>)> =
        read_safetensors_metadata(&snapshot.join("model.safetensors"))?.into_iter()
            .map(|t| (t.name, (t.dtype, t.shape))).collect();
    let (h, kv, inter) = (cfg.hidden, cfg.kv_width(), cfg.intermediate);
    let mut wanted = vec![("fc.weight".to_string(), vec![h, cfg.taps.len() * h]), ("hidden_norm.weight".into(), vec![h]),
        ("norm.weight".into(), vec![h]), ("markov_head.markov_w1.weight".into(), vec![cfg.vocab, cfg.rank]),
        ("markov_head.markov_w2.weight".into(), vec![cfg.vocab, cfg.rank]),
        ("confidence_head.proj.weight".into(), vec![1, h + cfg.rank]), ("confidence_head.proj.bias".into(), vec![1])];
    for l in 0..cfg.layers {
        let (p, a) = (format!("layers.{l}"), format!("layers.{l}.self_attn"));
        wanted.extend([(format!("{p}.input_layernorm.weight"), vec![h]),
            (format!("{p}.post_attention_layernorm.weight"), vec![h]),
            (format!("{a}.q_proj.weight"), vec![cfg.q_width(), h]), (format!("{a}.k_proj.weight"), vec![kv, h]),
            (format!("{a}.v_proj.weight"), vec![kv, h]), (format!("{a}.o_proj.weight"), vec![h, cfg.q_width()]),
            (format!("{a}.q_norm.weight"), vec![cfg.head_dim]), (format!("{a}.k_norm.weight"), vec![cfg.head_dim]),
            (format!("{p}.mlp.gate_proj.weight"), vec![inter, h]), (format!("{p}.mlp.up_proj.weight"), vec![inter, h]),
            (format!("{p}.mlp.down_proj.weight"), vec![h, inter])]);
    }
    for (name, shape) in wanted {
        let (dtype, found) = tensors.get(&name).with_context(|| format!("dSpark checkpoint has no {name}"))?;
        ensure!(*dtype == cuteafd_core::DType::Bf16 && *found == shape,
            "dSpark {name}: {dtype:?} {found:?}, expected BF16 {shape:?}");
    }
    Ok(())
}

/// Reads the drafter's tensors on a thread (while the target loads).
pub(crate) fn prefetch(snapshot: &Path) -> std::thread::JoinHandle<Result<HashMap<String, Vec<u8>>>> {
    let path = snapshot.join("model.safetensors");
    std::thread::spawn(move || {
        use std::os::unix::fs::FileExt;
        let file = std::fs::File::open(&path)?;
        read_safetensors_metadata(&path)?.into_iter().filter(|t| wanted(&t.name)).map(|t| {
            let mut bytes = vec![0u8; t.byte_length as usize];
            file.read_exact_at(&mut bytes, t.byte_offset)?;
            Ok((t.name, bytes))
        }).collect()
    })
}

pub(crate) struct DsparkDrafter<'a> {
    pub(crate) kv_cache: cuteafd_loader::families::qwen4::Qwen4KvCache,
    library: &'a NativeLibrary,
    pub cfg: DsparkConfig,
    stream: *mut c_void,
    /// The draft round's host wait (`--host-wait`).
    round: std::cell::RefCell<crate::shared::memory::BlockingWait<'a>>,
    pub slots: usize,
    pub(crate) max_sequences: usize,
    context_capacity: usize,
    paged: Option<PagedContext<'a>>,
    inv_freq: Option<Dev<'a>>,
    projections: cuteafd_ffi::draft_projection::DraftProjections<'a>,
    fc: Weight<'a>,
    hidden_norm: Dev<'a>,
    norm: Dev<'a>,
    markov_w1: Dev<'a>,
    markov_w2: Dev<'a>,
    /// Row norms of `markov_w2` (FP32): the exact pruning bound of the Markov argmax.
    markov_norms: Dev<'a>,
    confidence_w: Dev<'a>,
    confidence_b: Dev<'a>,
    layers: Vec<DraftLayer<'a>>,
    /// [TAP_ROWS, taps * hidden] BF16: the last step's tapped rows.
    pub taps: Dev<'a>,
    fused: Dev<'a>,
    fused_norm: Dev<'a>,
    context_kv: Dev<'a>,
    context_positions: Dev<'a>,
    context_slots: Dev<'a>,
    mask_row: Vec<u8>,
    target_projection: super::dflash::TargetProjection,
    workspace: RefCell<Option<Workspace<'a>>>,
    representation: GlmDraftRepresentation,
    /// GEMV scratch of the FP8 representation (every context/draft row count).
    fp8_workspace: Option<Dev<'a>>,
    /// How the borrowed BF16 head runs past one draft block (--draft-head).
    head_mode: Cell<DraftHead>,
    /// How the FP8 GEMMs run (--draft-linear), and the latest mode the FP8 scratch serves.
    fp8_rows: Cell<fp8_linear::Fp8Rows>,
    fp8_admitted: fp8_linear::Fp8Rows,
}

fn at(dev: &Dev<'_>, bytes: usize) -> *mut c_void {
    // SAFETY: callers stay inside the allocation.
    unsafe { dev.buffer.ptr.cast::<u8>().add(bytes) }.cast()
}

fn bytes_of<T: Copy>(values: &[T]) -> &[u8] {
    // SAFETY: plain-old-data slices viewed as bytes for host->device copies.
    unsafe { std::slice::from_raw_parts(values.as_ptr().cast(), std::mem::size_of_val(values)) }
}

impl<'a> DsparkDrafter<'a> {
    /// Actual physical CUDA device is supplied by the binding, not inferred from rank.
    /// The loader's request list (`block_dspark_arena_requests`) on one physical device.
    pub fn arena_requests(snapshot: &Path, representation: GlmDraftRepresentation, device: i32)
        -> Result<std::collections::BTreeMap<i32, Vec<(String, usize)>>> {
        ensure!(device >= 0, "negative block dSpark device");
        Ok(std::collections::BTreeMap::from([(device,
            cuteafd_loader::formats::block_dspark::block_dspark_arena_requests(snapshot, representation)?)]))
    }

    pub(crate) fn allocation_receipts(&self, inventory: &mut crate::shared::memory::receipts::Inventory<'_>,
        prefix: &str) -> Result<()> {
        self.fc.allocation_receipts(inventory, &format!("{prefix}.fc"))?;
        for (index, layer) in self.layers.iter().enumerate() {
            for (name, owner) in [("qkv", &layer.qkv), ("o", &layer.o), ("gate_up", &layer.gate_up), ("down", &layer.down)] {
                owner.allocation_receipts(inventory, &format!("{prefix}.layer.{index}.{name}"))?;
            }
            for (name, owner) in [("input_norm", &layer.input_norm), ("post_norm", &layer.post_norm),
                ("q_norm", &layer.q_norm), ("k_norm", &layer.k_norm)] {
                owner.inventory(inventory, &format!("{prefix}.layer.{index}.{name}"))?;
            }
        }
        for (name, owner) in [("hidden_norm", &self.hidden_norm), ("norm", &self.norm), ("markov_w1", &self.markov_w1),
            ("markov_w2", &self.markov_w2), ("markov_norms", &self.markov_norms), ("confidence_w", &self.confidence_w),
            ("confidence_b", &self.confidence_b)] { owner.inventory(inventory, &format!("{prefix}.{name}"))?; }
        if let Some(owner) = &self.inv_freq { owner.inventory(inventory, &format!("{prefix}.inv_freq"))?; }
        Ok(())
    }

    /// Uploads the drafter's tensors (see [`prefetch`]) and allocates `slots`
    /// ring contexts; draft steps take up to `max_sequences` sequences.
    /// `mask_row` is the target embedding of the mask token; `fp8_rows` how the
    /// FP8 GEMMs run.
    #[allow(clippy::too_many_arguments)]
    pub fn load(library: &'a NativeLibrary, programs: &'a cuteafd_ffi::programs::Programs<'a>, target_projection: super::dflash::TargetProjection, snapshot: &Path, tensors: HashMap<String, Vec<u8>>, stream: *mut c_void,
        slots: usize, max_sequences: usize, mask_row: Vec<u8>, representation: GlmDraftRepresentation,
        scales: fp8_linear::Fp8Scales, fp8_rows: fp8_linear::Fp8Rows, kv_cache: cuteafd_loader::families::qwen4::Qwen4KvCache) -> Result<Self> {
        Self::load_with(library, programs, target_projection, snapshot, tensors, stream, slots, max_sequences, mask_row,
            representation, scales, fp8_rows, kv_cache,
            |device, name, bytes| super::weights::standalone(library, device, name, bytes))
    }

    #[allow(clippy::too_many_arguments)]
    pub fn load_with(library: &'a NativeLibrary, programs: &'a cuteafd_ffi::programs::Programs<'a>, target_projection: super::dflash::TargetProjection, snapshot: &Path, tensors: HashMap<String, Vec<u8>>, stream: *mut c_void,
        slots: usize, max_sequences: usize, mask_row: Vec<u8>, representation: GlmDraftRepresentation,
        scales: fp8_linear::Fp8Scales, fp8_rows: fp8_linear::Fp8Rows, kv_cache: cuteafd_loader::families::qwen4::Qwen4KvCache,
        mut allocate: impl FnMut(i32, &str, usize) -> Result<crate::shared::memory::device::Allocation<'a>>) -> Result<Self> {
        let cfg = DsparkConfig::read(snapshot)?;
        let capacity = cfg.window.context("full-context dSpark requires explicit admitted context capacity")?;
        Self::load_admitted_with(library, programs, target_projection, snapshot, tensors, stream, slots, max_sequences, mask_row,
            representation, scales, fp8_rows, capacity, kv_cache, &mut allocate)
    }

    /// Full-context capacity is explicit: reject positions beyond admission,
    /// never silently wrap them into the GLM sliding window.
    #[allow(clippy::too_many_arguments)]
    pub fn load_admitted(library: &'a NativeLibrary, programs: &'a cuteafd_ffi::programs::Programs<'a>, target_projection: super::dflash::TargetProjection, snapshot: &Path, tensors: HashMap<String, Vec<u8>>,
        stream: *mut c_void, slots: usize, max_sequences: usize, mask_row: Vec<u8>,
        representation: GlmDraftRepresentation, scales: fp8_linear::Fp8Scales,
        fp8_rows: fp8_linear::Fp8Rows, context_capacity: usize, kv_cache: cuteafd_loader::families::qwen4::Qwen4KvCache) -> Result<Self> {
        Self::load_admitted_with(library, programs, target_projection, snapshot, tensors, stream, slots, max_sequences, mask_row,
            representation, scales, fp8_rows, context_capacity, kv_cache,
            |device, name, bytes| super::weights::standalone(library, device, name, bytes))
    }

    #[allow(clippy::too_many_arguments)]
    pub fn load_admitted_with(library: &'a NativeLibrary, programs: &'a cuteafd_ffi::programs::Programs<'a>, target_projection: super::dflash::TargetProjection, snapshot: &Path, tensors: HashMap<String, Vec<u8>>,
        stream: *mut c_void, slots: usize, max_sequences: usize, mask_row: Vec<u8>,
        representation: GlmDraftRepresentation, scales: fp8_linear::Fp8Scales,
        fp8_rows: fp8_linear::Fp8Rows, context_capacity: usize, kv_cache: cuteafd_loader::families::qwen4::Qwen4KvCache,
        mut allocate: impl FnMut(i32, &str, usize) -> Result<crate::shared::memory::device::Allocation<'a>>) -> Result<Self> {
        let target_tokens = slots.checked_mul(context_capacity).context("dSpark token budget overflow")?;
        Self::load_paged_with(library, programs, target_projection, snapshot, tensors, stream, slots, max_sequences, mask_row,
            representation, scales, fp8_rows, context_capacity, target_tokens, kv_cache, &mut allocate)
    }

    /// Serving full-context layers use the same aggregate token budget as target KV.
    #[allow(clippy::too_many_arguments)]
    pub fn load_paged(library: &'a NativeLibrary, programs: &'a cuteafd_ffi::programs::Programs<'a>, target_projection: super::dflash::TargetProjection, snapshot: &Path, tensors: HashMap<String, Vec<u8>>,
        stream: *mut c_void, slots: usize, max_sequences: usize, mask_row: Vec<u8>,
        representation: GlmDraftRepresentation, scales: fp8_linear::Fp8Scales,
        fp8_rows: fp8_linear::Fp8Rows, context_capacity: usize, target_tokens: usize,
        kv_cache: cuteafd_loader::families::qwen4::Qwen4KvCache) -> Result<Self> {
        Self::load_paged_with(library, programs, target_projection, snapshot, tensors, stream, slots, max_sequences, mask_row,
            representation, scales, fp8_rows, context_capacity, target_tokens, kv_cache,
            |device, name, bytes| super::weights::standalone(library, device, name, bytes))
    }

    #[allow(clippy::too_many_arguments)]
    pub fn load_paged_with(library: &'a NativeLibrary, programs: &'a cuteafd_ffi::programs::Programs<'a>, target_projection: super::dflash::TargetProjection, snapshot: &Path, tensors: HashMap<String, Vec<u8>>,
        stream: *mut c_void, slots: usize, max_sequences: usize, mask_row: Vec<u8>,
        representation: GlmDraftRepresentation, scales: fp8_linear::Fp8Scales,
        fp8_rows: fp8_linear::Fp8Rows, context_capacity: usize, target_tokens: usize,
        kv_cache: cuteafd_loader::families::qwen4::Qwen4KvCache,
        mut allocate: impl FnMut(i32, &str, usize) -> Result<crate::shared::memory::device::Allocation<'a>>) -> Result<Self> {
        let _memory_scope = cuteafd_ffi::memory_ledger::scope("drafter");
        let cfg = DsparkConfig::read(snapshot)?;
        ensure!(cfg.backbone.is_qwen3(cfg.head_dim), "dSpark backbone needs native arithmetic bindings");
        let h = cfg.hidden;
        let shapes = [(h * cfg.taps.len(), h), (h, cfg.qkv_width()), (h, 2 * cfg.kv_heads * cfg.head_dim),
            (cfg.q_width(), h), (h, 2 * cfg.intermediate), (cfg.intermediate, h)];
        let projections = cuteafd_ffi::draft_projection::DraftProjections::load(programs,
            if representation == GlmDraftRepresentation::Bf16Only { &shapes } else { &[] },
            cuteafd_ffi::draft_projection::batch_rows(TAP_ROWS, cfg.block, max_sequences,
                programs.capacities().decode_rows.unwrap_or(0)))?;
        ensure!(context_capacity > 0 && slots > 0 && max_sequences <= slots,
            "dSpark requires nonzero admitted context slots and capacity");
        ensure!(cfg.window.is_none_or(|window| window == context_capacity),
            "sliding dSpark capacity must equal its checkpoint window");
        ensure!(slots.checked_mul(context_capacity).is_some_and(|rows| rows <= i32::MAX as usize),
            "dSpark context capacity exceeds signed32 native slots");
        ensure!(cfg.hidden.checked_mul(2) == Some(mask_row.len()), "dSpark mask row shape");
        let config = serde_json::from_slice(&std::fs::read(snapshot.join("config.json"))?)?;
        let cache_layout = cuteafd_loader::formats::block_dspark::BlockDsparkConfig::from_json(&config)?
            .paged_cache_layout(slots, context_capacity, target_tokens, kv_cache)?;
        let cache_rows = cache_layout.layer_bytes[0] / cache_layout.record_bytes;
        let residency = cuteafd_loader::families::glm5::draft_representation::block_dspark_paged_resident_bytes(
            &config, slots, max_sequences, library.sm_count()? as u64, representation, fp8_rows.code() as u8,
            context_capacity, target_tokens, kv_cache)?;
        tracing::info!(resident_bytes = residency.owned, scratch_bytes = residency.scratch, kv_cache = %kv_cache, "dSpark shared readiness inventory");
        ensure!((1..=MAX_SEQUENCES).contains(&max_sequences), "dSpark drafts take 1..={MAX_SEQUENCES} sequences");
        let allocate: &mut WeightAllocator<'a, '_> = &mut allocate;
        let load = Load::with_tensors(library, snapshot, tensors)?.with_allocator(allocate);
        let upload = |name: &str, data: &[u8]| -> Result<Dev<'a>> {
            let allocation = load.allocation(name, data.len().max(256))?;
            library.copy_h2d(allocation.buffer, data)?;
            Ok(allocation)
        };
        let zeroed = |n: usize, tag: Tag| -> Result<Dev<'a>> {
            let allocation = DeviceAllocation::new(library, n.max(256), tag)?;
            library.cuda_zero_bytes(allocation.buffer, allocation.buffer.bytes)?;
            Ok(allocation)
        };
        let (h, kv, inter) = (cfg.hidden, cfg.kv_width(), cfg.intermediate);
        let weights = weight_recipe(snapshot, |name, shape| load.tensor(name, shape),
            |parts, cols| load.matrix(parts, cols, representation == GlmDraftRepresentation::Fp8Only, scales, stream))?;
        let layers = weights.layers.into_iter().map(|layer| -> Result<_> {
            Ok(DraftLayer { input_norm: layer.input_norm, post_norm: layer.post_norm, qkv: layer.qkv,
                q_norm: layer.q_norm, k_norm: layer.k_norm, o: layer.o, gate_up: layer.gate_up, down: layer.down,
                k_ring: zeroed(cache_rows * kv_cache.plane_record_bytes(cfg.kv_heads, cfg.head_dim), Tag::drafter_kv("shared/dspark-rings"))?,
                v_ring: zeroed(cache_rows * kv_cache.plane_record_bytes(cfg.kv_heads, cfg.head_dim), Tag::drafter_kv("shared/dspark-rings"))? })
        }).collect::<Result<Vec<_>>>()?;
        let taps = cfg.taps.len() * h;
        let fp8_workspace = match representation {
            GlmDraftRepresentation::Bf16Only => None,
            GlmDraftRepresentation::Fp8Only => {
                let shapes = [(taps, h), (h, cfg.qkv_width()), (h, 2 * kv), (cfg.q_width(), h), (h, 2 * inter),
                    (inter, h)];
                Some(fp8_linear::scratch_rows(library, TAP_ROWS.max(max_sequences * cfg.block), &shapes, fp8_rows)?)
            }
        };
        let markov_norms = load.allocation("markov_norms", cfg.vocab * 4)?;
        load.row_norms(weights.markov_w2.buffer, markov_norms.buffer, cfg.vocab, cfg.rank, stream)?;
        Ok(Self {
            kv_cache,
            library,
            stream,
            round: std::cell::RefCell::new(crate::shared::memory::BlockingWait::new(library)),
            slots,
            max_sequences,
            context_capacity,
            paged: if cfg.window.is_none() {
                use std::hash::{Hash, Hasher};
                let mut identity = std::collections::hash_map::DefaultHasher::new();
                (context_capacity, cfg.hidden, cfg.layers, cfg.kv_heads, cfg.head_dim,
                    &cfg.taps, cache_layout.record_bytes).hash(&mut identity);
                let mut state = PagedState {
                    pool: cuteafd_engine::prefix::DraftPagePool::new(target_tokens, slots,
                        context_capacity, identity.finish())?, bindings: vec![None; slots] };
                for slot in 0..slots { state.bindings[slot] = Some(state.pool.bind(slot, 0)?); }
                Some(PagedContext { tables: zeroed(cache_layout.page_table_bytes as usize, Tag::drafter_workspace("shared/dspark-page-tables"))?,
                    layout: cache_layout, state: RefCell::new(state) })
            } else { None },
            inv_freq: cfg.rotary.as_ref().map(|plan| upload("inv_freq", bytes_of(&plan.inv_freq))).transpose()?,
            fc: weights.fc, hidden_norm: weights.hidden_norm, norm: weights.norm,
            markov_w1: weights.markov_w1, markov_w2: weights.markov_w2, markov_norms,
            confidence_w: weights.confidence_w, confidence_b: weights.confidence_b,
            layers,
            taps: zeroed(TAP_ROWS * taps * 2, Tag::drafter_workspace("shared/dspark-taps"))?,
            fused: zeroed(TAP_ROWS * h * 2, Tag::drafter_workspace("shared/dspark-context-update"))?,
            fused_norm: zeroed(TAP_ROWS * h * 2, Tag::drafter_workspace("shared/dspark-context-update"))?,
            context_kv: zeroed(TAP_ROWS * 2 * kv * 2, Tag::drafter_workspace("shared/dspark-context-update"))?,
            context_positions: zeroed(TAP_ROWS * 8, Tag::drafter_workspace("shared/dspark-context-update"))?,
            context_slots: zeroed(TAP_ROWS * 4, Tag::drafter_workspace("shared/dspark-context-update"))?,
            mask_row,
            target_projection, workspace: RefCell::new(None),
            projections, representation,
            fp8_workspace,
            head_mode: Cell::new(DraftHead::Exact),
            fp8_rows: Cell::new(fp8_rows),
            fp8_admitted: fp8_rows,
            cfg,
        })
    }

    /// How draft steps run the borrowed BF16 head from now on (the FP8 head launcher ignores it).
    pub fn set_draft_head(&self, mode: DraftHead) {
        self.head_mode.set(mode);
    }

    /// Every checkpoint-sourced device owner (weights, FP8 values and scales, the
    /// Markov norms finish), named, for push-versus-mount byte equality.
    #[cfg(test)]
    pub(crate) fn weight_buffers(&self) -> Vec<(String, CuteafdDeviceBuffer)> {
        let mut out = Vec::new();
        let mut weight = |name: String, weight: &Weight<'_>| match weight {
            Weight::Bf16(owner) => out.push((name, owner.buffer)),
            Weight::Fp8(owner) => {
                let (values, scales) = owner.buffers();
                out.push((format!("{name}.values"), values));
                out.push((format!("{name}.scales"), scales));
            }
        };
        weight("fc".into(), &self.fc);
        for (index, layer) in self.layers.iter().enumerate() {
            for (name, owner) in [("qkv", &layer.qkv), ("o", &layer.o), ("gate_up", &layer.gate_up), ("down", &layer.down)] {
                weight(format!("layer.{index}.{name}"), owner);
            }
        }
        for (index, layer) in self.layers.iter().enumerate() {
            for (name, owner) in [("input_norm", &layer.input_norm), ("post_norm", &layer.post_norm),
                ("q_norm", &layer.q_norm), ("k_norm", &layer.k_norm)] {
                out.push((format!("layer.{index}.{name}"), owner.buffer));
            }
        }
        for (name, owner) in [("hidden_norm", &self.hidden_norm), ("norm", &self.norm), ("markov_w1", &self.markov_w1),
            ("markov_w2", &self.markov_w2), ("markov_norms", &self.markov_norms), ("confidence_w", &self.confidence_w),
            ("confidence_b", &self.confidence_b)] {
            out.push((name.into(), owner.buffer));
        }
        out
    }

    /// How the FP8 GEMMs run from now on: the load's mode or one before it.
    pub fn set_draft_linear(&self, mode: fp8_linear::Fp8Rows) -> Result<()> {
        ensure!(mode <= self.fp8_admitted, "the dSpark FP8 scratch was admitted for {:?}, not {mode:?}",
            self.fp8_admitted);
        self.fp8_rows.set(mode);
        Ok(())
    }

    /// `out` [rows, n] = `x` [rows, k] @ rows `first..first + n` of `weight`^T.
    ///
    /// # Safety
    /// Pointers are live device buffers of those shapes.
    #[allow(clippy::too_many_arguments)]
    unsafe fn linear(&self, x: *const c_void, weight: &Weight<'_>, first: usize, out: *mut c_void, rows: usize,
        k: usize, n: usize) -> Result<()> {
        use super::weights::{LegacyProjection, Projection};
        let projection = LegacyProjection { weight, library: self.library, programs: &self.projections,
            scratch: self.fp8_workspace.as_ref(), mode: self.fp8_rows.get() };
        // SAFETY: the caller retains the selected matrix rows and input/output;
        // set_draft_linear keeps the scratch within its admitted mode.
        unsafe { projection.project(x, first, out, rows, k, n, self.stream) }
    }

    pub fn tap_index(&self, layer: usize) -> Option<usize> {
        self.cfg.taps.iter().position(|&l| l == layer)
    }

    /// Taps the mean of the `hc` streams of rows `[first, first + n)` of
    /// `streams` ([rows, hc, hidden] BF16) into tap rows `to..to + n` when
    /// `layer` is tapped.
    pub fn tap_streams_at(&self, layer: usize, streams: *const c_void, hc: usize, first: usize, n: usize, to: usize)
        -> Result<()> {
        let Some(index) = self.tap_index(layer) else { return Ok(()) };
        let (h, width) = (self.cfg.hidden, self.cfg.taps.len() * self.cfg.hidden);
        ensure!(to + n <= TAP_ROWS, "tap rows {to}..{} exceed {TAP_ROWS}", to + n);
        // SAFETY: `streams` holds first + n rows of hc streams; the tap buffer TAP_ROWS rows.
        unsafe {
            self.library.glm_dflash_tap_mean(streams.cast::<u8>().add(first * hc * h * 2).cast(),
                self.taps.buffer.ptr.cast::<u8>().add(to * width * 2).cast(), n, h, hc, width, index * h, self.stream)
        }
    }

    /// Copies ordered, strided target taps without a host transfer. Reads use
    /// the producer stream, so returning permits the target to reuse storage.
    ///
    /// # Safety
    /// Each tap points to its declared live BF16 span on this device and stays
    /// valid until the producer stream reaches these copies.
    pub unsafe fn capture_target_taps(&self, taps: &[super::target::HiddenTap], first: usize,
        n: usize, to: usize) -> Result<()> {
        ensure!(taps.len() == self.cfg.taps.len(), "target tap count differs from draft checkpoint");
        let end = first.checked_add(n).context("target tap row overflow")?;
        let destination_end = to.checked_add(n).context("draft tap row overflow")?;
        ensure!(destination_end <= TAP_ROWS, "draft tap rows exceed capacity");
        let width_bytes = self.cfg.hidden.checked_mul(2).context("tap width overflow")?;
        let pitch = taps.len().checked_mul(width_bytes).context("draft tap pitch overflow")?;
        // Validate the complete set before enqueueing any transfer.
        validate_target_taps(taps, &self.cfg.taps, self.cfg.hidden, self.stream, end, n)?;
        if n == 0 { return Ok(()); }
        let device_id = self.library.cuda_get_device()?;
        for (index, tap) in taps.iter().enumerate() {
            let source_pitch = tap.stride.checked_mul(2).context("target tap pitch overflow")?;
            let offset = first.checked_mul(source_pitch).context("target tap offset overflow")?;
            let source_bytes = (tap.rows - 1) * source_pitch + width_bytes;
            let destination_offset = to * pitch + index * width_bytes;
            // SAFETY: all spans/strides were validated, the caller owns live
            // producer buffers, and source reads precede reuse on that stream.
            unsafe {
                self.library.copy_d2d_2d_async(
                    CuteafdDeviceBuffer { ptr: at(&self.taps, destination_offset),
                        bytes: self.taps.buffer.bytes - destination_offset, ..self.taps.buffer }, pitch,
                    CuteafdDeviceBuffer { ptr: tap.ptr.cast::<u8>().add(offset).cast_mut().cast(),
                        bytes: source_bytes - offset, device_id, ..CuteafdDeviceBuffer::default() },
                    source_pitch, width_bytes, n, self.stream)?;
            }
        }
        Ok(())
    }

    /// Writes the context K/V of committed tapped rows (in tap-row order).
    pub fn update(&self, rows: &[ContextRow]) -> Result<()> {
        if rows.is_empty() {
            return Ok(());
        }
        let (first, last) = (rows.iter().map(|r| r.tap_row).min().unwrap(), rows.iter().map(|r| r.tap_row).max().unwrap());
        let n = last - first + 1;
        ensure!(last < TAP_ROWS, "tap row {last} past {TAP_ROWS}");
        let (mut positions, mut slots) = (vec![0i64; n], vec![-1i32; n]);
        let mut seen = std::collections::HashSet::new();
        let mut groups = std::collections::BTreeMap::<usize, Vec<&ContextRow>>::new();
        for row in rows {
            ensure!(seen.insert(row.tap_row), "duplicate block dSpark tap destination");
            positions[row.tap_row - first] = i64::try_from(row.position)?;
            groups.entry(row.slot).or_default().push(row);
        }
        let mut lease = PageLease { drafter: self, reservations: Vec::new() };
        if let Some(context) = &self.paged {
            // Check all rows before reserving or enqueueing any producer writes.
            let mut state = context.state.borrow_mut();
            for (&slot, group) in &mut groups {
                group.sort_by_key(|row| row.position);
                let binding = state.binding(slot)?;
                let (_, frontier) = state.pool.frontier(binding)?;
                ensure!(group.iter().enumerate().all(|(i, row)| row.position == frontier + i),
                    "block dSpark update must append contiguous canonical rows");
            }
            for (&slot, group) in &groups {
                let binding = state.binding(slot)?;
                let reservation = state.pool.reserve(binding, group[0].position, group.len())?;
                lease.reservations.push(reservation);
                for (row, destination) in group.iter().zip(state.pool.destinations(reservation)?) {
                    slots[row.tap_row - first] = i32::try_from(destination)?;
                }
            }
        } else {
            for row in rows {
                slots[row.tap_row - first] = self.cfg.context_row(row.slot, row.position, self.slots,
                    self.context_capacity)?;
            }
        }
        self.put(&self.context_positions, bytes_of(&positions))?;
        self.put(&self.context_slots, bytes_of(&slots))?;
        let (h, kv) = (self.cfg.hidden, self.cfg.kv_width());
        let width = self.cfg.taps.len() * h;
        let s = self.stream;
        // SAFETY: every buffer holds TAP_ROWS rows of its width; the stream orders the chain.
        unsafe {
            self.linear(at(&self.taps, first * width * 2), &self.fc, 0, self.fused.buffer.ptr, n, width, h)?;
            self.library.glm_dflash_rmsnorm(self.fused.buffer.ptr, self.hidden_norm.buffer.ptr,
                self.fused_norm.buffer.ptr, n, h, self.cfg.eps, s)?;
            let q_rows = self.cfg.q_width();
            for layer in &self.layers {
                self.linear(self.fused_norm.buffer.ptr, &layer.qkv, q_rows, self.context_kv.buffer.ptr, n, h, 2 * kv)?;
                self.library.block_dspark_qk_rope(self.context_kv.buffer.ptr, layer.q_norm.buffer.ptr,
                    layer.k_norm.buffer.ptr, self.context_positions.buffer.ptr, self.context_slots.buffer.ptr,
                    self.inv_freq.as_ref().map_or(std::ptr::null(), |v| v.buffer.ptr.cast_const()), std::ptr::null_mut(), layer.k_ring.buffer.ptr, layer.v_ring.buffer.ptr, n, 0, self.cfg.kv_heads, self.cfg.head_dim,
                    self.cfg.theta, self.cfg.rotary.as_ref().map_or(1.0, |p| p.attention_factor), self.cfg.eps, self.kv_cache == cuteafd_loader::families::qwen4::Qwen4KvCache::Fp8, s)?;
            }
        }
        lease.commit()
    }

    pub(crate) fn reset_context(&self, slot: usize, origin: usize) -> Result<()> {
        ensure!(slot < self.slots && origin <= self.context_capacity, "block dSpark reset outside admission");
        if let Some(context) = &self.paged {
            // SAFETY: retire the previous request before releasing its page ownership.
            unsafe { self.library.cuda_stream_synchronize(self.stream)?; }
            let mut state = context.state.borrow_mut();
            state.discard(slot)?;
            state.bindings[slot] = Some(state.pool.bind(slot, origin)?);
        }
        Ok(())
    }

    pub(crate) fn discard_context(&self, slot: usize) -> Result<()> {
        ensure!(slot < self.slots, "block dSpark discard outside admission");
        if let Some(context) = &self.paged {
            // SAFETY: cancellation cannot recycle destinations still read by queued work.
            unsafe { self.library.cuda_stream_synchronize(self.stream)?; }
            context.state.borrow_mut().discard(slot)?;
        }
        Ok(())
    }

    /// The slot's live draft page reservation (`None` for windowed drafters).
    pub(crate) fn reservation(&self, slot: usize) -> Option<cuteafd_engine::reserve::DraftDemand> {
        let context = self.paged.as_ref()?;
        let state = context.state.borrow();
        let binding = state.binding(slot).ok()?;
        state.pool.reservation(binding, self.cfg.block).ok()
    }
    pub(crate) fn lifetime_rows(&self) -> Option<usize> {
        self.paged.as_ref().map(|context| context.state.borrow().pool.lifetime_rows(self.cfg.block))
    }
    pub(crate) fn reservable(&self) -> Option<usize> {
        self.paged.as_ref().map(|context| context.state.borrow().pool.reservable())
    }
    pub(crate) fn admission(&self, origin: usize) -> Option<cuteafd_engine::reserve::DraftDemand> {
        self.paged.as_ref().map(|_| cuteafd_engine::reserve::DraftDemand {
            origin, headroom: self.cfg.block.div_ceil(cuteafd_engine::prefix::DRAFT_PAGE_ROWS) + 1, held: 0 })
    }

    pub(crate) fn admit_pages(&self, slot: usize, end: usize) -> Result<()> {
        if let Some(context) = &self.paged {
            let mut state = context.state.borrow_mut();
            let binding = state.binding(slot)?;
            state.pool.admit(binding, end, self.cfg.block)?;
        }
        Ok(())
    }

    pub(crate) fn capture_pages(&self, slot: usize, frontier: usize, write_frontier: usize)
        -> Result<cuteafd_engine::prefix::DraftPageAttachment> {
        let context = self.paged.as_ref().context("windowed block dSpark has no full-context pages")?;
        // SAFETY: canonical producers retire before snapshot membership is published.
        unsafe { self.library.cuda_stream_synchronize(self.stream)?; }
        let mut state = context.state.borrow_mut();
        let binding = state.binding(slot)?;
        ensure!(state.pool.frontier(binding)?.1 == frontier, "block dSpark snapshot frontier differs");
        let (snapshot, copies) = state.pool.capture_writable(binding, write_frontier)?;
        let attachment = state.pool.attachment(snapshot)?;
        drop(state);
        if let Err(error) = self.copy_pages(&copies) {
            // SAFETY: failed copies retain destinations until this stream drains.
            if unsafe { self.library.cuda_stream_synchronize(self.stream) }.is_ok() {
                context.state.borrow_mut().pool.release_snapshot(snapshot)?;
            }
            return Err(error);
        }
        // SAFETY: snapshot consumers may run on a different stream after publication.
        unsafe { self.library.cuda_stream_synchronize(self.stream)?; }
        Ok(attachment)
    }

    pub(crate) fn pages_valid(&self, attachment: &cuteafd_engine::prefix::DraftPageAttachment) -> bool {
        self.paged.as_ref().is_some_and(|context|
            context.state.borrow().pool.check_attachment(attachment).is_ok())
    }

    pub(crate) fn pages_revocable(&self, attachment: &cuteafd_engine::prefix::DraftPageAttachment) -> bool {
        self.paged.as_ref().is_some_and(|context|
            context.state.borrow().pool.snapshot_revocable(attachment.snapshot))
    }

    pub(crate) fn retain_pages(&self, attachment: &cuteafd_engine::prefix::DraftPageAttachment) -> Result<()> {
        let context = self.paged.as_ref().context("block dSpark has no page pool")?;
        let mut state = context.state.borrow_mut();
        state.pool.check_attachment(attachment)?;
        Ok(state.pool.retain_snapshot(attachment.snapshot)?)
    }

    pub(crate) fn release_pages(&self, attachment: &cuteafd_engine::prefix::DraftPageAttachment) -> Result<()> {
        let context = self.paged.as_ref().context("block dSpark has no page pool")?;
        // SAFETY: family also retires its host-copy stream before releasing the attachment.
        unsafe { self.library.cuda_stream_synchronize(self.stream)?; }
        let mut state = context.state.borrow_mut();
        state.pool.check_attachment(attachment)?;
        Ok(state.pool.release_snapshot(attachment.snapshot)?)
    }

    pub(crate) fn promote_pages(&self, saved: &cuteafd_engine::prefix::DraftPageAttachment)
        -> Result<cuteafd_engine::prefix::DraftPageAttachment> {
        let context = self.paged.as_ref().context("block dSpark has no page pool")?;
        Ok(context.state.borrow_mut().pool.promote(saved)?)
    }

    pub(crate) fn restore_pages(&self, slot: usize,
        attachment: &cuteafd_engine::prefix::DraftPageAttachment, write_frontier: usize) -> Result<()> {
        let context = self.paged.as_ref().context("block dSpark has no page pool")?;
        ensure!(slot < self.slots, "block dSpark restore outside admitted slots");
        // SAFETY: retire the previous binding before replacing its ownership.
        unsafe { self.library.cuda_stream_synchronize(self.stream)?; }
        let mut state = context.state.borrow_mut();
        state.pool.check_attachment(attachment)?;
        state.pool.pin_snapshot(attachment.snapshot)?;
        state.discard(slot)?;
        let (binding, copies) = match state.pool.restore_writable(slot, attachment.snapshot,
            attachment.frontier, write_frontier) {
            Ok(restored) => restored,
            Err(error) => { state.pool.unpin_snapshot(attachment.snapshot)?; return Err(error.into()); }
        };
        state.bindings[slot] = Some(binding);
        drop(state);
        let copied = self.copy_pages(&copies);
        // SAFETY: retain source pin and destinations through the final restore DMA.
        // A failed drain deliberately quarantines both in the pool.
        unsafe { self.library.cuda_stream_synchronize(self.stream)?; }
        context.state.borrow_mut().pool.unpin_snapshot(attachment.snapshot)?;
        copied
    }

    pub(crate) fn page_bytes(&self) -> usize {
        self.paged.as_ref().map_or(0, |context|
            context.layout.page_rows * context.layout.record_bytes * context.layout.full_layers)
    }

    pub(crate) fn page_segments(&self, page: cuteafd_engine::prefix::DraftPageId)
        -> Result<Vec<cuteafd_hostcache::copy::DeviceRange>> {
        let context = self.paged.as_ref().context("block dSpark has no page pool")?;
        ensure!((page.index as usize) < context.layout.pages, "block dSpark page outside pool");
        let bytes = context.layout.page_rows * context.layout.record_bytes / 2;
        Ok(self.layers.iter().flat_map(|layer| [&layer.k_ring, &layer.v_ring])
            .map(|plane| cuteafd_hostcache::copy::DeviceRange {
                addr: at(plane, page.index as usize * bytes) as u64, bytes }).collect())
    }

    /// Slot-local physical planes, including inactive bytes, for exact window
    /// snapshots. No tap replay or weight-dependent reconstruction is involved.
    pub(crate) fn window_planes(&self, slot: usize) -> Result<Vec<CuteafdDeviceBuffer>> {
        ensure!(self.cfg.window == Some(self.context_capacity) && self.paged.is_none()
            && slot < self.slots, "block dSpark has no admitted window slot");
        let bytes = self.context_capacity.checked_mul(self.kv_cache.plane_record_bytes(
            self.cfg.kv_heads, self.cfg.head_dim)).context("block dSpark window extent overflow")?;
        let offset = slot.checked_mul(bytes).context("block dSpark window offset overflow")?;
        self.layers.iter().flat_map(|layer| [&layer.k_ring, &layer.v_ring]).map(|plane| {
            ensure!(offset.checked_add(bytes).is_some_and(|end| end <= plane.buffer.bytes),
                "block dSpark window plane outside retained backing");
            Ok(CuteafdDeviceBuffer { ptr: at(plane, offset), bytes, ..plane.buffer })
        }).collect()
    }

    fn copy_pages(&self, copies: &[cuteafd_engine::prefix::TailCopy]) -> Result<()> {
        let context = self.paged.as_ref().context("block dSpark has no page pool")?;
        let row_bytes = context.layout.record_bytes / 2;
        let page_bytes = context.layout.page_rows * row_bytes;
        for copy in copies {
            for plane in self.layers.iter().flat_map(|layer| [&layer.k_ring, &layer.v_ring]) {
                let bytes = copy.rows * row_bytes;
                let source = CuteafdDeviceBuffer { ptr: at(plane, copy.from as usize * page_bytes),
                    bytes, ..plane.buffer };
                let destination = CuteafdDeviceBuffer { ptr: at(plane, copy.to as usize * page_bytes),
                    bytes, ..plane.buffer };
                // SAFETY: allocator retains checked page segments until the stream retirement barrier.
                unsafe { self.library.copy_d2d_async(destination, source, bytes, self.stream)?; }
            }
        }
        Ok(())
    }

    fn put(&self, dev: &Dev<'_>, bytes: &[u8]) -> Result<()> {
        ensure!(bytes.len() <= dev.buffer.bytes, "table exceeds its buffer");
        // SAFETY: metadata and its consumers share this drafter's owned stream.
        unsafe { self.library.copy_h2d_ordered(dev.buffer, bytes, self.stream) }
    }

    /// Explicit serving startup admission; other callers retain lazy allocation.
    pub(crate) fn prepare_workspace(&self) -> Result<()> {
        if self.workspace.borrow().is_none() {
            *self.workspace.borrow_mut() = Some(self.workspace(self.max_sequences)?);
        }
        Ok(())
    }

    fn workspace(&self, sequences: usize) -> Result<Workspace<'a>> {
        let _memory_scope = cuteafd_ffi::memory_ledger::scope("drafter/workspace");
        let c = &self.cfg;
        let rows = sequences * c.block;
        let alloc = |bytes: usize| DeviceAllocation::new(self.library, bytes.max(256), Tag::drafter_workspace("shared/dspark"));
        let head_workspace = alloc(VOCABULARY_HEAD_WORKSPACE)?;
        let attention_workspace = {
            let _scope = cuteafd_ffi::memory_ledger::scope("workspace/draft-attention");
            alloc(self.library.block_dspark_attention_workspace(sequences, c.heads, c.kv_heads, c.block,
                self.context_capacity + c.block, c.head_dim)?)?
        };
        Ok(Workspace {
            sequences,
            h: alloc(rows * c.hidden * 2)?,
            n: alloc(rows * c.hidden * 2)?,
            qkv: alloc(rows * c.qkv_width() * 2)?,
            q: alloc(rows * c.q_width() * 2)?,
            k: alloc(rows * c.kv_width() * 2)?,
            v: alloc(rows * c.kv_width() * 2)?,
            attn: alloc(rows * c.q_width() * 2)?,
            delta: alloc(rows * c.hidden * 2)?,
            gate_up: alloc(rows * 2 * c.intermediate * 2)?,
            act: alloc(rows * c.intermediate * 2)?,
            logits: alloc(rows * c.vocab * 4)?,
            anchors: alloc(sequences * 4)?,
            ids: alloc(rows * 4)?,
            tokens: alloc(rows * 4)?,
            confidence: alloc(rows * 4)?,
            landing: crate::shared::memory::HostAllocation::new(self.library, rows * 8, Tag::staging("shared/dspark-landing"))?,
            positions: alloc(rows * 8)?,
            tables: alloc(3 * sequences * 4)?,
            attention_workspace,
            markov_workspace: alloc(self.library.glmf_dspark_markov_workspace(sequences, c.block)?)?,
            head: if self.target_projection == super::dflash::TargetProjection::Bf16 {
                // SAFETY: the retained scratch drops after the borrowed native head.
                Some(unsafe { self.library.vocabulary_head_rows(head_workspace.buffer.ptr, c.hidden as u32,
                    rows as u32, c.vocab as u32, cuteafd_ffi::programs::HeadMode::Gemm)? })
            } else { None },
            _head_workspace: head_workspace,
        })
    }

    fn head_ref<'h>(&self, head: &'h TargetHead<'h>) -> Result<HeadRef<'h>> {
        Ok(match head {
            TargetHead::Bf16(owner) => {
                ensure!(owner.buffer.bytes == self.cfg.vocab * self.cfg.hidden * 2,
                    "dSpark borrows the target BF16 head [{}, {}]", self.cfg.vocab, self.cfg.hidden);
                HeadRef::Bf16(owner.buffer.ptr)
            }
            TargetHead::Bf16Buffer(_) => anyhow::bail!("dSpark requires its retained target head owner"),
            TargetHead::Launch(launch) => HeadRef::Launch(launch),
        })
    }

    /// [`Self::draft`] with the block's input rows gathered on the device
    /// from the target's embedding table.
    pub fn draft_device(&self, sequences: &[DraftSeq], embedding: &TokenEmbedding<'_>, head: TargetHead<'_>)
        -> Result<Vec<Draft>> {
        self.draft_from(sequences, DraftInput::Table(embedding), self.head_ref(&head)?, true)
    }

    pub fn draft_device_diagnostic(&self, sequences: &[DraftSeq], embedding: &TokenEmbedding<'_>,
        head: TargetHead<'_>) -> Result<Vec<Draft>> {
        self.draft_from(sequences, DraftInput::Table(embedding), self.head_ref(&head)?, false)
    }

    fn draft_from(&self, sequences: &[DraftSeq], input: DraftInput<'_, '_>, head: HeadRef<'_>, serving: bool)
        -> Result<Vec<Draft>> {
        ensure!(!matches!(head, HeadRef::Bf16(p) if p.is_null()), "dSpark has no borrowed target head");
        let c = &self.cfg;
        let (s_count, block, h) = (sequences.len(), c.block, c.hidden);
        let rows_ok = match input {
            DraftInput::Rows(r) => r.len() == s_count * h * 2,
            DraftInput::Table(_) => true,
        };
        ensure!(s_count > 0 && s_count <= self.max_sequences && rows_ok, "draft step of {s_count} sequences");
        let rows = s_count * block;
        let mut slot = self.workspace.borrow_mut();
        if slot.as_ref().is_none_or(|w| w.sequences < s_count) {
            *slot = None;
            *slot = Some(self.workspace(self.max_sequences)?);
        }
        let w = slot.as_ref().context("draft workspace")?;
        let mut positions = Vec::with_capacity(rows);
        let mut tables = vec![0i32; 3 * s_count];
        let table_width = self.context_capacity.div_ceil(64);
        let mut pages = vec![0i32; if self.paged.is_some() { s_count * table_width } else { 0 }];
        for (i, seq) in sequences.iter().enumerate() {
            ensure!(seq.slot < self.slots, "ring slot {} of {}", seq.slot, self.slots);
            ensure!(self.cfg.window.is_some() || seq.position <= self.context_capacity,
                "full-context dSpark exceeds admitted context");
            ensure!(seq.position.checked_add(block).is_some_and(|end| end <= i32::MAX as usize),
                "dSpark position exceeds native extent");
            positions.extend((seq.position..seq.position + block).map(|p| p as i64));
            tables[i] = seq.slot as i32;
            tables[s_count + i] = seq.position.saturating_sub(seq.valid_from).min(self.context_capacity) as i32;
            tables[2 * s_count + i] = seq.position as i32;
            if let Some(context) = &self.paged {
                let state = context.state.borrow();
                let binding = state.binding(seq.slot)?;
                let (origin, frontier) = state.pool.frontier(binding)?;
                ensure!(origin == seq.valid_from && frontier == seq.position,
                    "block dSpark draft differs from canonical context frontier");
                let table = state.pool.committed_page_table(binding)?;
                ensure!(table.len() <= table_width, "block dSpark page table exceeds context");
                pages[i * table_width..i * table_width + table.len()].copy_from_slice(&table);
                tables[s_count + i] = i32::try_from(frontier - origin)?;
                tables[2 * s_count + i] = i32::try_from(frontier - origin)?;
            }
        }
        if let Some(context) = &self.paged { self.put(&context.tables, bytes_of(&pages))?; }
        // Eager per-round launch: keep the arena capacity-sized, not this work bound.
        // A future graph path must read changing lengths on-device instead.
        let max_keys = tables[s_count..2 * s_count].iter().copied().max().unwrap() as usize + block;
        match input {
            DraftInput::Rows(anchor_rows) => {
                let mut embed = Vec::with_capacity(rows * h * 2);
                for i in 0..s_count {
                    embed.extend_from_slice(&anchor_rows[i * h * 2..(i + 1) * h * 2]);
                    for _ in 1..block {
                        embed.extend_from_slice(&self.mask_row);
                    }
                }
                self.put(&w.h, &embed)?;
            }
            DraftInput::Table(embedding) => {
                let ids: Vec<u32> = sequences.iter()
                    .flat_map(|seq| std::iter::once(seq.anchor).chain(std::iter::repeat_n(c.mask_token, block - 1)))
                    .collect();
                embedding.embed(&ids, w.ids.buffer, 1, w.h.buffer, self.stream)?;
            }
        }
        self.put(&w.positions, bytes_of(&positions))?;
        self.put(&w.tables, bytes_of(&tables))?;
        let anchors: Vec<u32> = sequences.iter().map(|s| s.anchor).collect();
        self.put(&w.anchors, bytes_of(&anchors))?;
        let (s, l) = (self.stream, self.library);
        let (eps, inter) = (c.eps, c.intermediate);
        // SAFETY: every workspace buffer holds `rows` rows of its width and the
        // weights their checkpoint shapes; the stream orders the chain.
        unsafe {
            l.glm_dflash_rmsnorm(w.h.buffer.ptr, self.layers[0].input_norm.buffer.ptr, w.n.buffer.ptr, rows, h, eps, s)?;
            for (index, layer) in self.layers.iter().enumerate() {
                self.linear(w.n.buffer.ptr, &layer.qkv, 0, w.qkv.buffer.ptr, rows, h, c.qkv_width())?;
                l.block_dspark_qk_rope(w.qkv.buffer.ptr, layer.q_norm.buffer.ptr, layer.k_norm.buffer.ptr,
                    w.positions.buffer.ptr, std::ptr::null(),
                    self.inv_freq.as_ref().map_or(std::ptr::null(), |v| v.buffer.ptr.cast_const()), w.q.buffer.ptr, w.k.buffer.ptr, w.v.buffer.ptr, rows,
                    c.heads, c.kv_heads, c.head_dim, c.theta,
                    c.rotary.as_ref().map_or(1.0, |p| p.attention_factor), eps, false, s)?;
                if let Some(context) = &self.paged {
                    l.block_dspark_attention_paged(w.q.buffer.ptr, w.k.buffer.ptr, w.v.buffer.ptr,
                        layer.k_ring.buffer.ptr, layer.v_ring.buffer.ptr, w.tables.buffer.ptr,
                        at(&w.tables, s_count * 4), at(&w.tables, 2 * s_count * 4), context.tables.buffer.ptr,
                        w.attn.buffer.ptr, w.attention_workspace.buffer.ptr, s_count, block, c.heads,
                        c.kv_heads, self.context_capacity, max_keys, c.causal, c.head_dim, table_width,
                        1.0 / (c.head_dim as f32).sqrt(),
                        self.kv_cache == cuteafd_loader::families::qwen4::Qwen4KvCache::Fp8, s)?;
                } else {
                    l.block_dspark_attention(w.q.buffer.ptr, w.k.buffer.ptr, w.v.buffer.ptr, layer.k_ring.buffer.ptr,
                        layer.v_ring.buffer.ptr, w.tables.buffer.ptr, at(&w.tables, s_count * 4), at(&w.tables, 2 * s_count * 4),
                        w.attn.buffer.ptr, w.attention_workspace.buffer.ptr, s_count, block, c.heads, c.kv_heads, self.context_capacity,
                        max_keys, c.causal, c.head_dim, 1.0 / (c.head_dim as f32).sqrt(), self.kv_cache == cuteafd_loader::families::qwen4::Qwen4KvCache::Fp8, s)?;
                }
                self.linear(w.attn.buffer.ptr, &layer.o, 0, w.delta.buffer.ptr, rows, c.q_width(), h)?;
                l.glmf_dspark_add_rmsnorm(w.h.buffer.ptr, w.delta.buffer.ptr, layer.post_norm.buffer.ptr,
                    w.h.buffer.ptr, w.n.buffer.ptr, rows, h, eps, s)?;
                self.linear(w.n.buffer.ptr, &layer.gate_up, 0, w.gate_up.buffer.ptr, rows, h, 2 * inter)?;
                l.glm_dflash_silu_mul(w.gate_up.buffer.ptr, w.act.buffer.ptr, rows, inter, s)?;
                self.linear(w.act.buffer.ptr, &layer.down, 0, w.delta.buffer.ptr, rows, inter, h)?;
                let next = self.layers.get(index + 1).map_or(self.norm.buffer.ptr, |n| n.input_norm.buffer.ptr);
                l.glmf_dspark_add_rmsnorm(w.h.buffer.ptr, w.delta.buffer.ptr, next, w.h.buffer.ptr, w.n.buffer.ptr,
                    rows, h, eps, s)?;
            }
            match head {
                HeadRef::Bf16(weight) if serving => crate::shared::draft::launch_bf16_head(w.head.as_ref().context("BF16 dSpark head was not prepared before serving")?,
                    w.n.buffer.ptr.cast(), weight.cast(), w.logits.buffer.ptr.cast(), rows, s)?,
                HeadRef::Bf16(weight) => crate::shared::speculation::dflash::launch_draft_head(l, w.head.as_ref().context("BF16 dSpark head was not prepared before serving")?, w.n.buffer.ptr,
                    weight, w.logits.buffer.ptr.cast(), rows, h, c.vocab, s, self.head_mode.get())?,
                HeadRef::Launch(launch) => launch(w.n.buffer.ptr.cast_const(), w.logits.buffer.ptr.cast(), rows, s)?,
            }
            l.glmf_dspark_markov(w.logits.buffer.ptr, self.markov_w1.buffer.ptr, self.markov_w2.buffer.ptr,
                self.markov_norms.buffer.ptr, w.anchors.buffer.ptr, w.tokens.buffer.ptr, w.markov_workspace.buffer.ptr, s_count, block, c.vocab,
                c.rank, s)?;
            l.glmf_dspark_confidence(w.n.buffer.ptr, self.markov_w1.buffer.ptr, self.confidence_w.buffer.ptr,
                self.confidence_b.buffer.ptr, w.anchors.buffer.ptr, w.tokens.buffer.ptr, w.confidence.buffer.ptr,
                s_count, block, h, c.rank, s)?;
            // The round's results download on its stream; one host wait then
            // covers the round and both copies (no second, synchronous drain).
            // SAFETY: the device buffers hold `rows` words; the landing is
            // pinned, owned by the workspace and read only after the wait.
            let landing = |offset: usize| cuteafd_ffi::CuteafdHostBuffer {
                ptr: w.landing.buffer.ptr.cast::<u8>().add(offset).cast(), bytes: rows * 4, ..w.landing.buffer };
            l.copy_d2h_host_buffer_async(landing(0), CuteafdDeviceBuffer { bytes: rows * 4, ..w.tokens.buffer },
                rows * 4, s)?;
            l.copy_d2h_host_buffer_async(landing(rows * 4), CuteafdDeviceBuffer { bytes: rows * 4, ..w.confidence.buffer },
                rows * 4, s)?;
            self.round.borrow_mut().wait(s)?;
        }
        let (tokens, confidence) = w.landing.bytes()[..rows * 8].split_at(rows * 4);
        let word = |b: &[u8], i: usize| u32::from_le_bytes(b[i * 4..i * 4 + 4].try_into().unwrap());
        Ok((0..s_count).map(|i| Draft {
            tokens: (0..block).map(|j| word(&tokens, i * block + j)).collect(),
            features: vec![[0.0; 4]; block],
            confidence: (0..block).map(|j| f32::from_bits(word(&confidence, i * block + j))).collect(),
        }).collect())
    }

    /// The last draft step's final-norm rows [sequences * block, hidden] BF16.
    pub fn last_hidden(&self, sequences: usize) -> Result<Vec<u8>> {
        let slot = self.workspace.borrow();
        let w = slot.as_ref().context("no draft step ran")?;
        let bytes = sequences * self.cfg.block * self.cfg.hidden * 2;
        let mut out = vec![0u8; bytes];
        self.library.copy_d2h(&mut out, CuteafdDeviceBuffer { bytes, ..w.n.buffer })?;
        Ok(out)
    }

    #[cfg(test)]
    pub(crate) fn last_logits(&self, sequences: usize) -> Result<Vec<f32>> {
        let slot = self.workspace.borrow();
        let w = slot.as_ref().context("no draft step ran")?;
        let bytes = sequences.checked_mul(self.cfg.block).and_then(|n| n.checked_mul(self.cfg.vocab))
            .and_then(|n| n.checked_mul(4)).context("draft logit readback overflow")?;
        ensure!(sequences > 0 && bytes <= w.logits.buffer.bytes, "draft logit readback exceeds arena");
        let mut host = vec![0; bytes];
        self.library.copy_d2h(&mut host, CuteafdDeviceBuffer { bytes, ..w.logits.buffer })?;
        Ok(host.chunks_exact(4).map(|word| f32::from_le_bytes(word.try_into().unwrap())).collect())
    }

    /// Copies BF16 tap rows (host, [n, taps * hidden]) into the tap buffer.
    pub fn put_taps(&self, rows: &[u8]) -> Result<()> {
        self.put(&self.taps, rows)
    }
}

impl ReplayDrafter for DsparkDrafter<'_> {
    fn block(&self) -> usize {
        self.cfg.block
    }

    fn drafts(&self) -> usize {
        self.cfg.drafts()
    }

    fn sequences(&self) -> usize {
        self.max_sequences.min(self.slots)
    }

    fn resident_modes(&self) -> Vec<ReplayMode> {
        vec![ReplayMode { name: self.representation.name(), legacy_fp8: None }]
    }

    fn context(&self, taps: &[u8], first: usize) -> Result<()> {
        let n = taps.len() / (self.cfg.taps.len() * self.cfg.hidden * 2);
        self.put_taps(taps)?;
        self.update(&(0..n).map(|r| ContextRow { tap_row: r, slot: 0, position: first + r }).collect::<Vec<_>>())
    }

    fn draft_tokens(&self, seqs: &[(usize, u32, usize)], anchor_rows: &[u8], head: *const c_void)
        -> Result<Vec<Vec<u32>>> {
        let seqs: Vec<DraftSeq> = seqs.iter().map(|&(slot, anchor, position)| DraftSeq { slot, anchor, position, valid_from: 0 }).collect();
        // Replay's caller keeps the target BF16 head live throughout.
        Ok(self.draft_from(&seqs, DraftInput::Rows(anchor_rows), HeadRef::Bf16(head), false)?.into_iter()
            .map(|d| d.tokens).collect())
    }

    fn tap_rows(&self) -> usize {
        TAP_ROWS
    }

    fn last_hidden(&self, sequences: usize) -> Result<Vec<u8>> {
        DsparkDrafter::last_hidden(self, sequences)
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use crate::shared::memory::TEST;
    #[test]
    fn window_snapshot_restore_equals_cold_state_and_survives_slot_reuse() -> Result<()> {
        use device::{Allocation, Device, Stream};
        use cuteafd_ffi::native_library_lifetime_fixture::Fixture;
        use cuteafd_loader::placement::planner::{CoordinatorDevices, Lifetime};
        use cuteafd_core::memory_layout::Category;
        let fixture = Fixture::build_for_tests()?;
        let library = fixture.load()?;
        fixture.configure_memory(&library, 1, 1 << 30)?;
        fixture.immediate_copies(&library)?;
        fixture.quiet_events(&library, true)?;
        let device = Device { library: &library, id: 1 };
        let stream = Stream::new(device)?;
        let planes = [Allocation::new(device, 128, TEST)?, Allocation::new(device, 256, TEST)?];
        let live = planes.iter().map(|plane| plane.buffer).collect::<Vec<_>>();
        let snapshots = WindowSnapshots::new(&stream, vec![128, 256], 2)?;
        // A cold window after wrap, with deterministic BF16/FP8 payload bytes.
        let cold = |frontier: usize, bytes: usize| {
            let rows = bytes / 16;
            let mut ring = vec![0u8; bytes];
            for position in 0..frontier {
                let at = position % rows * 16;
                for (column, value) in ring[at..at + 16].iter_mut().enumerate() {
                    *value = (position.wrapping_mul(37) + column * 11) as u8;
                }
            }
            ring
        };
        let expected = [cold(23, 128), cold(23, 256)];
        for (plane, bytes) in live.iter().zip(&expected) {
            device.run(|| library.copy_h2d(*plane, bytes))?;
        }
        // SAFETY: owned disjoint fixture planes survive the ordered copy owner.
        unsafe { snapshots.capture(0, &live, 0, 23)?; }
        snapshots.drain()?;
        for plane in &live { device.run(|| library.copy_h2d(*plane, &vec![197; plane.bytes]))?; }
        // Another request/snapshot must not overwrite the original mark.
        unsafe { snapshots.capture(1, &live, 23, 31)?; }
        snapshots.drain()?;
        assert!(unsafe { snapshots.restore(0, &live, 24) }.is_err());
        for plane in &live {
            let mut bytes = vec![0; plane.bytes];
            device.run(|| library.copy_d2h(&mut bytes, *plane))?;
            assert_eq!(bytes, vec![197; plane.bytes], "invalid restore must not mutate destinations");
        }
        assert_eq!(unsafe { snapshots.restore(0, &live, 23)? }, (0, 23));
        snapshots.drain()?;
        for (plane, cold) in live.iter().zip(&expected) {
            let mut restored = vec![0; plane.bytes];
            device.run(|| library.copy_d2h(&mut restored, *plane))?;
            assert_eq!(&restored, cold);
        }
        let before = fixture.events()?;
        let mut inventory = receipts::Inventory::new(CoordinatorDevices::new(&[1])?, "window",
            Lifetime::TARGET, Category::Prefix, "draft window marks");
        snapshots.inventory(&mut inventory, "marks")?;
        assert_eq!(inventory.receipts.iter().filter(|r| r.allocation.view.is_none()
            && r.allocation.location == cuteafd_loader::placement::planner::Location::coordinator(0)).count(), 1);
        assert_eq!(fixture.events()?, before, "inventory must not enqueue native work");
        assert!(unsafe { snapshots.capture(2, &live, 0, 23) }.is_err());
        assert!(WindowSnapshots::bytes(&[usize::MAX - 15], 2).is_err());
        Ok(())
    }



    #[test]
    fn block_drafter_header_requests_match_arena_mount_and_push_bytes() -> Result<()> {
        use crate::shared::{experts::rtx::dense, memory::{device::Device, weight_arena::{DeviceWeightArena, weight_allocation}, receipts::Inventory}};
        use cuteafd_loader::{plan::testing::{t, write_safetensors}, serving_capacity::WeightArenaLayout,
            placement::planner::{CoordinatorDevices, Lifetime, Phase}, families::qwen4::Qwen4KvCache};
        use cuteafd_core::memory_layout::Category;
        use std::{collections::BTreeSet, os::unix::fs::FileExt};
        let fixture = cuteafd_ffi::native_library_lifetime_fixture::Fixture::build_for_tests()?;
        let library = fixture.load()?;
        fixture.configure_memory(&library, 1, 1 << 30)?;
        fixture.immediate_copies(&library)?;
        fixture.quiet_events(&library, true)?;
        let device = Device { library: &library, id: 1 };
        for window in [Some(8), None] {
            let root = tempfile::tempdir_in(fixture.directory())?;
            let snapshot = root.path().join("draft");
            std::fs::create_dir_all(&snapshot)?;
            let config = serde_json::json!({"architectures":["DSparkDraftModel"], "speculators_model_type":"dspark",
                "aux_hidden_state_layer_ids":[1], "block_size":2, "confidence_head_with_markov":true,
                "draft_vocab_size":16, "enable_confidence_head":true, "markov_head_type":"vanilla", "markov_rank":256,
                "mask_token_id":0, "sample_from_anchor":true, "sliding_window_non_causal":false,
                "transformer_layer_config":{"attention_bias":false, "head_dim":128, "hidden_act":"silu",
                    "hidden_size":256, "intermediate_size":256, "layer_types":[if window.is_some() { "sliding_attention" } else { "full_attention" }],
                    "max_position_embeddings":64, "model_type":"qwen3", "num_attention_heads":2, "num_hidden_layers":1,
                    "num_key_value_heads":2, "rms_norm_eps":1e-5, "rope_parameters":{"rope_theta":10000.0,"rope_type":"default"},
                    "sliding_window":window, "vocab_size":16}});
            std::fs::write(snapshot.join("config.json"), serde_json::to_vec(&config)?)?;
            let cfg = DsparkConfig::read(&snapshot)?;
            let shapes = RefCell::new(Vec::new());
            weight_recipe(&snapshot, |name, shape| { shapes.borrow_mut().push(t(name,"BF16",shape)); Ok(()) },
                |parts,cols| { for (name,rows) in parts { shapes.borrow_mut().push(t(*name,"BF16",&[*rows,cols])); } Ok(()) })?;
            let path = snapshot.join("model.safetensors");
            write_safetensors(&path,&shapes.into_inner());
            let file = std::fs::OpenOptions::new().write(true).open(&path)?;
            for tensor in read_safetensors_metadata(&path)? {
                let bytes = (0..tensor.byte_length/2).flat_map(|i| {
                    let value = (i % 127) as f32 * 0.015625 - 0.75;
                    ((value.to_bits() >> 16) as u16).to_le_bytes()
                }).collect::<Vec<_>>();
                file.write_all_at(&bytes,tensor.byte_offset)?;
            }
            let shapes = [(256,256),(256,768),(256,512)].into_iter().collect::<BTreeSet<_>>();
            let entries = shapes.into_iter().map(|(k,n)| serde_json::json!({"name":format!("draft_bf16_k{k}_n{n}"),
                "pointers":[{"name":"x"},{"name":"w"},{"name":"out"}], "scalars":[{"type":"int32"}],
                "capacity_rows":TAP_ROWS, "scratch_bytes_at_capacity":{}, "geometry":{"width":k,"outputs":n,
                    "output_dtype":"torch.bfloat16", "reduction":"ordered_k16", "tiles_by_rows":[[16,1,4,16,64,6]]}})).collect::<Vec<_>>();
            fixture.register_programs(&library,&entries)?;
            let manifest = snapshot.join("programs.json");
            std::fs::write(&manifest,serde_json::to_vec(&serde_json::json!({"programs":entries}))?)?;
            let programs = library.programs()?.with_manifest(&manifest)?;
            for representation in [GlmDraftRepresentation::Bf16Only,GlmDraftRepresentation::Fp8Only] {
                let expected = DsparkDrafter::arena_requests(&snapshot,representation,device.id)?.remove(&device.id).unwrap();
                fn run<'a>(device: Device<'a>, programs: &'a cuteafd_ffi::programs::Programs<'a>, snapshot: &Path,
                    representation: GlmDraftRepresentation, arena: Option<&DeviceWeightArena<'a>>, expected: &[(String,usize)])
                    -> Result<DsparkDrafter<'a>> {
                    let requested = RefCell::new(Vec::new());
                    let tensors = if dense::is_recording() { HashMap::new() } else {
                        prefetch(snapshot).join().map_err(|_| anyhow::anyhow!("prefetch panic"))?? };
                    let mut allocate = |id,name:&str,bytes| {
                        assert_eq!(id,device.id); requested.borrow_mut().push((name.to_owned(),bytes));
                        weight_allocation(device,arena,name,bytes, crate::shared::memory::TEST_ARENA)
                    };
                    let cfg = DsparkConfig::read(snapshot)?;
                    let drafter = device.run(|| if cfg.window.is_some() {
                        DsparkDrafter::load_with(device.library,programs,crate::shared::speculation::dflash::TargetProjection::Native,snapshot,tensors,std::ptr::null_mut(),
                            1,1,vec![0;512],representation,fp8_linear::Fp8Scales::Pow2,fp8_linear::Fp8Rows::W8a16,
                            Qwen4KvCache::Bf16,&mut allocate)
                    } else {
                        DsparkDrafter::load_paged_with(device.library,programs,crate::shared::speculation::dflash::TargetProjection::Native,snapshot,tensors,std::ptr::null_mut(),
                            1,1,vec![0;512],representation,fp8_linear::Fp8Scales::Pow2,fp8_linear::Fp8Rows::W8a16,
                            8,64,Qwen4KvCache::Bf16,&mut allocate)
                    })?;
                    assert_eq!(requested.into_inner(),expected);
                    Ok(drafter)
                }
                let baseline = run(device,&programs,&snapshot,representation,None,&expected)?;
                let digest = |drafter:&DsparkDrafter<'_>| -> Result<Vec<(String,Vec<u8>)>> {
                    let mut owners = drafter.weight_buffers();
                    if let Some(inv) = &drafter.inv_freq { owners.push(("inv_freq".into(),inv.buffer)); }
                    owners.into_iter().map(|(name,buffer)| {
                        let mut bytes=vec![0;buffer.bytes]; device.run(|| library.copy_d2h(&mut bytes,buffer))?;
                        assert!(bytes.iter().any(|&b|b!=0),"unfilled {name}"); Ok((name,bytes))
                    }).collect()
                };
                let wanted = digest(&baseline)?;
                for recorded in [false,true] {
                    let arena = DeviceWeightArena::new(device,WeightArenaLayout::new("test/block",expected.clone())?, crate::shared::memory::TEST_ARENA)?;
                    let actual = if recorded {
                        let (draft,records) = dense::recording(|| run(device,&programs,&snapshot,representation,Some(&arena),&expected))?;
                        let sockets=tempfile::tempdir()?;
                        crate::shared::experts::rtx::push::tests::receive_pushed(device,None,root.path(),sockets.path(),&records,None,4,1)?;
                        draft
                    } else { run(device,&programs,&snapshot,representation,Some(&arena),&expected)? };
                    drop(arena);
                    assert_eq!(digest(&actual)?,wanted,"{representation:?} window={window:?} recorded={recorded}");
                    let mut inventory=Inventory::new(CoordinatorDevices::new(&[1])?,"block",
                        Lifetime { first:Phase::DraftLoad,last:Phase::Growth },Category::Drafter,"block weights");
                    actual.allocation_receipts(&mut inventory,"draft")?;
                    assert_eq!(inventory.receipts.iter().filter(|r|r.allocation.view.is_none()).count(),1);
                    assert_eq!(inventory.receipts.iter().filter(|r|r.allocation.view.is_some()).count(),expected.len());
                }
            }
        }
        Ok(())
    }

    fn geometry(window: Option<usize>) -> DsparkConfig {
        DsparkConfig { backbone: cuteafd_loader::formats::block_dspark::DsparkBackbone {
                norm: cuteafd_loader::formats::block_dspark::DsparkNorm::Rms,
                residual: cuteafd_loader::formats::block_dspark::DsparkResidual::Bf16, rotary_dim: 128,
                activation: cuteafd_core::FfnActivation::SiluBf16Product { limit: None } },
            hidden: 5120, intermediate: 10240, layers: 5, heads: 40, kv_heads: 8,
            head_dim: 128, eps: 1e-6, theta: 1e7, block: 7, mask_token: 248077, rank: 256,
            taps: vec![4,16,28,40,52], vocab: 248320, causal: false, window, rotary: None }
    }

    #[test]
    fn draft_target_taps_validate_order_stride_stream_and_span() {
        use super::super::target::HiddenTap;
        let marker = 0u8;
        let pointer = std::ptr::from_ref(&marker).cast();
        let tap = HiddenTap { layer: 4, ptr: pointer, rows: 7, hidden: 5120, stride: 5128,
            stream: std::ptr::null_mut() };
        let taps = [tap, HiddenTap { layer: 16, ..tap }];
        validate_target_taps(&taps, &[4, 16], 5120, std::ptr::null_mut(), 7, 7).unwrap();
        assert!(validate_target_taps(&taps, &[16, 4], 5120, std::ptr::null_mut(), 7, 7).is_err());
        assert!(validate_target_taps(&taps, &[4], 5120, std::ptr::null_mut(), 7, 7).is_err());
        for invalid in [HiddenTap { rows: 6, ..tap }, HiddenTap { hidden: 4096, ..tap },
            HiddenTap { stride: 5119, ..tap }, HiddenTap { ptr: std::ptr::null(), ..tap },
            HiddenTap { stream: std::ptr::from_ref(&marker).cast_mut().cast(), ..tap },
            HiddenTap { stride: usize::MAX, ..tap }] {
            assert!(validate_target_taps(&[invalid], &[4], 5120, std::ptr::null_mut(), 7, 7).is_err());
        }
    }

    #[test]
    fn full_context_destinations_never_wrap() {
        let cfg = geometry(None);
        assert_eq!(cfg.context_row(0, 8191, 2, 8192).unwrap(), 8191);
        assert_eq!(cfg.context_row(1, 8191, 2, 8192).unwrap(), 16383);
        assert!(cfg.context_row(0, 8192, 2, 8192).is_err());
        assert!(cfg.context_row(2, 0, 2, 8192).is_err());
        assert!(cfg.context_row(1, 0, 2, usize::MAX).is_err());
    }

    #[test]
    fn sliding_context_retains_legacy_modulo() {
        let cfg = geometry(Some(2048));
        assert_eq!(cfg.context_row(1, 8193, 2, 2048).unwrap(), 2049);
        assert!(cfg.context_row(0, 0, 2, 0).is_err());
    }
}

impl crate::shared::prefix::cuda_copy::SnapshotOwner for DsparkDrafter<'_> {
    fn snapshot_buffers(&self) -> Result<Vec<cuteafd_ffi::CuteafdDeviceBuffer>> {
        Ok(self.layers.iter().flat_map(|layer| [layer.k_ring.buffer, layer.v_ring.buffer]).collect())
    }
}
