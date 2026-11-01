//! Shared Qwen3-backbone block dSpark runtime; target lifecycles stay in bindings.
use crate::shared::speculation::dflash::{ContextRow, Draft, DraftSeq, HeadLaunch, ReplayDrafter, ReplayMode, TargetHead,
    TAP_ROWS};
use crate::shared::speculation::dflash::DraftHead;
use cuteafd_loader::families::glm5::draft_representation::GlmDraftRepresentation;
use crate::shared::fp8_linear::{self, Fp8Weight};
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
pub(crate) const MAX_SEQUENCES: usize = 32;

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
        Ok(Self { hidden: cfg.hidden, intermediate: cfg.intermediate, layers: cfg.layers,
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

/// Exactly one resident representation of a GEMM weight.
enum Weight<'a> {
    Bf16(Dev<'a>),
    Fp8(Fp8Weight<'a>),
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
    head: VocabularyHead<'a>,
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
    /// Uploads the drafter's tensors (see [`prefetch`]) and allocates `slots`
    /// ring contexts; draft steps take up to `max_sequences` sequences.
    /// `mask_row` is the target embedding of the mask token; `fp8_rows` how the
    /// FP8 GEMMs run.
    #[allow(clippy::too_many_arguments)]
    pub fn load(library: &'a NativeLibrary, snapshot: &Path, tensors: HashMap<String, Vec<u8>>, stream: *mut c_void,
        slots: usize, max_sequences: usize, mask_row: Vec<u8>, representation: GlmDraftRepresentation,
        scales: fp8_linear::Fp8Scales, fp8_rows: fp8_linear::Fp8Rows, kv_cache: cuteafd_loader::families::qwen4::Qwen4KvCache) -> Result<Self> {
        let cfg = DsparkConfig::read(snapshot)?;
        let capacity = cfg.window.context("full-context dSpark requires explicit admitted context capacity")?;
        Self::load_admitted(library, snapshot, tensors, stream, slots, max_sequences, mask_row,
            representation, scales, fp8_rows, capacity, kv_cache)
    }

    /// Full-context capacity is explicit: reject positions beyond admission,
    /// never silently wrap them into the GLM sliding window.
    #[allow(clippy::too_many_arguments)]
    pub fn load_admitted(library: &'a NativeLibrary, snapshot: &Path, tensors: HashMap<String, Vec<u8>>,
        stream: *mut c_void, slots: usize, max_sequences: usize, mask_row: Vec<u8>,
        representation: GlmDraftRepresentation, scales: fp8_linear::Fp8Scales,
        fp8_rows: fp8_linear::Fp8Rows, context_capacity: usize, kv_cache: cuteafd_loader::families::qwen4::Qwen4KvCache) -> Result<Self> {
        let target_tokens = slots.checked_mul(context_capacity).context("dSpark token budget overflow")?;
        Self::load_paged(library, snapshot, tensors, stream, slots, max_sequences, mask_row,
            representation, scales, fp8_rows, context_capacity, target_tokens, kv_cache)
    }

    /// Serving full-context layers use the same aggregate token budget as target KV.
    #[allow(clippy::too_many_arguments)]
    pub fn load_paged(library: &'a NativeLibrary, snapshot: &Path, tensors: HashMap<String, Vec<u8>>,
        stream: *mut c_void, slots: usize, max_sequences: usize, mask_row: Vec<u8>,
        representation: GlmDraftRepresentation, scales: fp8_linear::Fp8Scales,
        fp8_rows: fp8_linear::Fp8Rows, context_capacity: usize, target_tokens: usize,
        kv_cache: cuteafd_loader::families::qwen4::Qwen4KvCache) -> Result<Self> {
        let _memory_scope = cuteafd_ffi::memory_ledger::scope("drafter");
        let cfg = DsparkConfig::read(snapshot)?;
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
        let (resident_bytes, scratch_bytes) = cuteafd_loader::families::glm5::draft_representation::block_dspark_paged_resident_bytes(
            &config, slots, max_sequences, library.sm_count()? as u64, representation, fp8_rows.code() as u8,
            context_capacity, target_tokens, kv_cache)?;
        tracing::info!(resident_bytes, scratch_bytes, kv_cache = %kv_cache, "dSpark shared readiness inventory");
        ensure!((1..=MAX_SEQUENCES).contains(&max_sequences), "dSpark drafts take 1..={MAX_SEQUENCES} sequences");
        let path = snapshot.join("model.safetensors");
        let shapes: HashMap<String, Vec<usize>> = read_safetensors_metadata(&path)?.into_iter()
            .filter(|t| t.dtype == cuteafd_core::DType::Bf16).map(|t| (t.name, t.shape)).collect();
        let bytes = |name: &str, shape: &[usize]| -> Result<&[u8]> {
            let found = shapes.get(name).with_context(|| format!("dSpark checkpoint has no BF16 {name}"))?;
            ensure!(found == shape, "{name}: shape {found:?}, expected {shape:?}");
            tensors.get(name).map(Vec::as_slice).with_context(|| format!("{name} was not read"))
        };
        let upload = |data: &[u8]| -> Result<Dev<'a>> {
            let allocation = DeviceAllocation::new(library, data.len().max(256))?;
            library.copy_h2d(allocation.buffer, data)?;
            Ok(allocation)
        };
        let zeroed = |n: usize| -> Result<Dev<'a>> {
            let allocation = DeviceAllocation::new(library, n.max(256))?;
            library.cuda_zero_bytes(allocation.buffer, allocation.buffer.bytes)?;
            Ok(allocation)
        };
        let (h, kv, inter) = (cfg.hidden, cfg.kv_width(), cfg.intermediate);
        let tensor = |name: &str, shape: &[usize]| bytes(name, shape).and_then(upload);
        let matrix = |source: Dev<'a>, n: usize, k: usize| -> Result<Weight<'a>> {
            match representation {
                GlmDraftRepresentation::Bf16Only => Ok(Weight::Bf16(source)),
                GlmDraftRepresentation::Fp8Only => {
                    let packed = Fp8Weight::pack(library, source.buffer.ptr, n, k, scales, stream);
                    // SAFETY: the packing kernel reads `source`: drain it before the source drops.
                    if let Err(error) = unsafe { library.cuda_stream_synchronize(stream) } {
                        std::mem::forget(source);
                        if let Ok(weight) = packed { weight.quarantine(); }
                        return Err(error.context("dSpark packing owners quarantined after a failed drain"));
                    }
                    let packed = packed?;
                    drop(source);
                    Ok(Weight::Fp8(packed))
                }
            }
        };
        let concat = |parts: &[(&str, usize)], cols: usize| -> Result<Dev<'a>> {
            let total: usize = parts.iter().map(|(_, rows)| rows * cols * 2).sum();
            let allocation = DeviceAllocation::new(library, total)?;
            let mut offset = 0;
            for (name, rows) in parts {
                let data = bytes(name, &[*rows, cols])?;
                library.copy_h2d(CuteafdDeviceBuffer { ptr: at(&allocation, offset), bytes: data.len(),
                    ..allocation.buffer }, data)?;
                offset += data.len();
            }
            Ok(allocation)
        };
        let layers = (0..cfg.layers).map(|l| -> Result<DraftLayer<'a>> {
            let p = format!("layers.{l}");
            let a = format!("{p}.self_attn");
            Ok(DraftLayer {
                input_norm: tensor(&format!("{p}.input_layernorm.weight"), &[h])?,
                post_norm: tensor(&format!("{p}.post_attention_layernorm.weight"), &[h])?,
                qkv: matrix(concat(&[(&format!("{a}.q_proj.weight"), cfg.q_width()), (&format!("{a}.k_proj.weight"), kv),
                    (&format!("{a}.v_proj.weight"), kv)], h)?, cfg.qkv_width(), h)?,
                q_norm: tensor(&format!("{a}.q_norm.weight"), &[cfg.head_dim])?,
                k_norm: tensor(&format!("{a}.k_norm.weight"), &[cfg.head_dim])?,
                o: matrix(tensor(&format!("{a}.o_proj.weight"), &[h, cfg.q_width()])?, h, cfg.q_width())?,
                gate_up: matrix(concat(&[(&format!("{p}.mlp.gate_proj.weight"), inter),
                    (&format!("{p}.mlp.up_proj.weight"), inter)], h)?, 2 * inter, h)?,
                down: matrix(tensor(&format!("{p}.mlp.down_proj.weight"), &[h, inter])?, h, inter)?,
                k_ring: zeroed(cache_rows * kv_cache.plane_record_bytes(cfg.kv_heads, cfg.head_dim))?,
                v_ring: zeroed(cache_rows * kv_cache.plane_record_bytes(cfg.kv_heads, cfg.head_dim))?,
            })
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
        let markov_w2 = tensor("markov_head.markov_w2.weight", &[cfg.vocab, cfg.rank])?;
        let markov_norms = DeviceAllocation::new(library, cfg.vocab * 4)?;
        // SAFETY: both buffers hold `vocab` rows; the stream orders the kernel before any draft.
        unsafe { library.glmf_dspark_markov_norms(markov_w2.buffer.ptr, markov_norms.buffer.ptr, cfg.vocab, cfg.rank,
            stream)? };
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
                Some(PagedContext { tables: zeroed(cache_layout.page_table_bytes as usize)?,
                    layout: cache_layout, state: RefCell::new(state) })
            } else { None },
            inv_freq: cfg.rotary.as_ref().map(|plan| upload(bytes_of(&plan.inv_freq))).transpose()?,
            fc: matrix(tensor("fc.weight", &[h, taps])?, h, taps)?,
            hidden_norm: tensor("hidden_norm.weight", &[h])?,
            norm: tensor("norm.weight", &[h])?,
            markov_w1: tensor("markov_head.markov_w1.weight", &[cfg.vocab, cfg.rank])?,
            markov_w2,
            markov_norms,
            confidence_w: tensor("confidence_head.proj.weight", &[1, h + cfg.rank])?,
            confidence_b: tensor("confidence_head.proj.bias", &[1])?,
            layers,
            taps: zeroed(TAP_ROWS * taps * 2)?,
            fused: zeroed(TAP_ROWS * h * 2)?,
            fused_norm: zeroed(TAP_ROWS * h * 2)?,
            context_kv: zeroed(TAP_ROWS * 2 * kv * 2)?,
            context_positions: zeroed(TAP_ROWS * 8)?,
            context_slots: zeroed(TAP_ROWS * 4)?,
            mask_row,
            workspace: RefCell::new(None),
            representation,
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
        match weight {
            // SAFETY: rows first..first + n lie in the matrix; the caller's contract.
            Weight::Bf16(w) => unsafe { self.library.linear_bf16(x, at(w, first * k * 2), out, rows, k, n, self.stream) },
            Weight::Fp8(w) => {
                let scratch = self.fp8_workspace.as_ref().context("FP8 dSpark scratch was not admitted")?;
                // SAFETY: the scratch covers every shape for up to max(TAP_ROWS, sequences x block) rows
                // in this mode (set_draft_linear keeps it within the admitted one).
                unsafe { w.apply_rows(self.library, x, out, false, rows, first, n, scratch, self.stream,
                    self.fp8_rows.get()) }
            }
        }
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
        let alloc = |bytes: usize| DeviceAllocation::new(self.library, bytes.max(256));
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
            landing: crate::shared::memory::HostAllocation::new(self.library, rows * 8)?,
            positions: alloc(rows * 8)?,
            tables: alloc(3 * sequences * 4)?,
            attention_workspace,
            markov_workspace: alloc(self.library.glmf_dspark_markov_workspace(sequences, c.block)?)?,
            // SAFETY: the workspace buffer lives in the same struct and drops after the head.
            head: unsafe { self.library.vocabulary_head_rows(head_workspace.buffer.ptr, c.hidden as u32,
                rows as u32, c.vocab as u32)? },
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
                HeadRef::Bf16(weight) if serving => crate::shared::draft::launch_bf16_head(&w.head,
                    w.n.buffer.ptr.cast(), weight.cast(), w.logits.buffer.ptr.cast(), rows, s)?,
                HeadRef::Bf16(weight) => crate::shared::speculation::dflash::launch_draft_head(l, &w.head, w.n.buffer.ptr,
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

    fn geometry(window: Option<usize>) -> DsparkConfig {
        DsparkConfig { hidden: 5120, intermediate: 10240, layers: 5, heads: 40, kv_heads: 8,
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
