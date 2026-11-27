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
        Self::with_backing(crate::shared::prefix::Reader { device:stream.device.id,stream:stream.raw },arena,planes,marks)
    }

    pub fn with_backing(reader:crate::shared::prefix::Reader,arena:Rc<device::Allocation<'a>>,
        planes:Vec<usize>,marks:usize)->Result<Self> {
        let (stride,bytes)=Self::bytes(&planes,marks)?;
        ensure!(arena.device.id==reader.device && arena.buffer.bytes==bytes,
            "window mark backing differs from admitted physical owner");
        let copies=crate::shared::prefix::PrefixCopies::new(arena.device.library,vec![vec![reader]])?;
        Ok(Self { copies,arena,planes,stride,marks })
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
        let (origin,end)=self.metadata(mark)?;
        ensure!(end == frontier, "window snapshot metadata/frontier differs");
        // SAFETY: caller retains the initialized mark and destination planes.
        unsafe { self.copies.copy_d2d(&copies)?; }
        Ok((origin, end))
    }

    pub fn metadata(&self,mark:usize)->Result<(usize,usize)> {
        self.copies.drain()?;
        let saved=self.mark(mark)?;
        let mut bytes=[0u8;16];
        self.copies.copy_d2h(0, &mut bytes, CuteafdDeviceBuffer { bytes:16,..saved })?;
        let origin=usize::try_from(u64::from_le_bytes(bytes[..8].try_into()?))?;
        let end=usize::try_from(u64::from_le_bytes(bytes[8..].try_into()?))?;
        ensure!(origin<=end,"window snapshot metadata invalid");
        Ok((origin,end))
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
        .is_some_and(|a| a.iter().any(|v| v == "DSparkDraftModel" || v == "Qwen3DSparkModel")))
}

/// The drafter geometry the kernels are written for, read from `config.json`.
#[derive(Debug, Clone)]
pub(crate) struct DsparkConfig {
    pub backbone: cuteafd_loader::formats::block_dspark::DsparkBackbone,
    pub dialect: cuteafd_loader::formats::block_dspark::DsparkDialect,
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
        ensure!(matches!(cfg.rank, 256 | 512), "dSpark kernels take vanilla rank-256/512 Markov heads");
        let rope_theta = match cfg.rope {
            DsparkRope::Default { rope_theta } | DsparkRope::Yarn { rope_theta, .. } => rope_theta,
        };
        let rotary = if cfg.head_dim == 64 && matches!(cfg.rope, DsparkRope::Default { .. }) {
            None
        } else { Some(cfg.rope.plan(cfg.backbone.rotary_dim)?) };
        Ok(Self { backbone: cfg.backbone, dialect: cfg.dialect, hidden: cfg.hidden, intermediate: cfg.intermediate, layers: cfg.layers,
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
        ensure!(tap.site == cuteafd_core::TapSite::LayerExit(layer), "target tap order differs from draft checkpoint");
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

struct NativeBackbone<'a> {
    norm: cuteafd_ffi::minimax_m3::forward::ForwardProducer<'a>,
    query: cuteafd_ffi::minimax_m3::producers::Producer<'a>,
    key: cuteafd_ffi::minimax_m3::producers::Producer<'a>,
    activation: cuteafd_ffi::minimax_m3::producers::Producer<'a>,
    add: cuteafd_ffi::Bf16Add<'a>,
    wide: Dev<'a>,
    norm_scratch: Dev<'a>,
    gate: Dev<'a>,
    up: Dev<'a>,
    rope: CuteafdDeviceBuffer,
    positions: usize,
    projections: Rc<super::weights::NativeProjectionWorkspace<'a>>,
}

impl<'a> NativeBackbone<'a> {
    fn bind(library: &'a NativeLibrary, programs: &'a cuteafd_ffi::programs::Programs<'a>,
        cfg: &DsparkConfig, rope: CuteafdDeviceBuffer, positions: usize,
        projections: Rc<super::weights::NativeProjectionWorkspace<'a>>) -> Result<Self> {
        use cuteafd_loader::formats::block_dspark::{DsparkNorm, DsparkResidual};
        use cuteafd_ffi::minimax_m3::{forward, producers};
        ensure!(cfg.hidden == 6144 && cfg.heads == 64 && cfg.kv_heads == 4 && cfg.head_dim == 128
            && cfg.backbone.norm == DsparkNorm::Gemma && cfg.backbone.residual == DsparkResidual::Bf16
            && cfg.backbone.rotary_dim == 64 && cfg.eps == 1e-6 && cfg.theta == 1e7
            && cfg.backbone.activation == (cuteafd_core::FfnActivation::SwigluOai { alpha:1.702, beta:1., limit:7. })
            && !cfg.causal && cfg.intermediate == 12288 && positions > 0
            && rope.device_id == library.cuda_get_device()? && !rope.ptr.is_null()
            && positions.checked_mul(256).is_some_and(|bytes| bytes <= rope.bytes),
            "native block drafter arithmetic differs from installed MiniMax programs");
        let alloc = |bytes, name| DeviceAllocation::new(library, bytes, Tag::drafter_workspace(name));
        Ok(Self { norm: forward::ForwardProducer::bind(programs, forward::Kind::ResidualNorm)?,
            query: producers::Producer::bind(programs, producers::Kind::NormRope { heads:64 })?,
            key: producers::Producer::bind(programs, producers::Kind::NormRope { heads:4 })?,
            activation: producers::Producer::bind(programs, producers::Kind::Oai { width:12288 })?,
            add: library.bf16_add()?, wide: alloc(TAP_ROWS * cfg.hidden * 4, "shared/draft-native/norm-input")?,
            norm_scratch: alloc(TAP_ROWS * cfg.q_width() * 2, "shared/draft-native/norm-rope")?,
            gate: alloc(TAP_ROWS * cfg.intermediate * 2, "shared/draft-native/gate")?,
            up: alloc(TAP_ROWS * cfg.intermediate * 2, "shared/draft-native/up")?, rope, positions, projections })
    }

    unsafe fn norm(&self, library: &NativeLibrary, input: *const c_void, weight: CuteafdDeviceBuffer,
        output: *mut c_void, rows: usize, stream: *mut c_void) -> Result<()> {
        ensure!(rows > 0 && rows <= TAP_ROWS, "native draft norm row capacity");
        let buffer = |ptr, bytes| CuteafdDeviceBuffer { ptr, bytes, device_id:self.wide.buffer.device_id, ..Default::default() };
        // SAFETY: retained scratch and caller spans survive the ordered native producers.
        unsafe {
            library.cuda_zero_bytes_async(self.wide.buffer, rows * 6144 * 4, stream)?;
            library.cuda_accumulate_bf16_to_f32_async(buffer(input.cast_mut(),rows * 6144 * 2),
                self.wide.buffer, rows * 6144, stream)?;
            self.norm.norm(cuteafd_ffi::minimax_m3::forward::NormBatch { rows,
                input:self.wide.buffer,weight,output:buffer(output,rows * 6144 * 2) }, stream)?;
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    unsafe fn rope(&self, cfg: &DsparkConfig, input: CuteafdDeviceBuffer, query_norm: CuteafdDeviceBuffer,
        key_norm: CuteafdDeviceBuffer, positions: CuteafdDeviceBuffer, query: CuteafdDeviceBuffer,
        key: CuteafdDeviceBuffer, value: CuteafdDeviceBuffer, rows: usize, context: bool,
        stream: *mut c_void) -> Result<()> {
        use cuteafd_ffi::minimax_m3::producers::NormRopeBatch;
        let library = self.wide.library;
        let q = if context { 0 } else { cfg.q_width() };
        let kv = cfg.kv_width();
        let packed = q + 2 * kv;
        ensure!(rows > 0 && rows <= TAP_ROWS && rows * packed * 2 <= input.bytes,
            "native draft QKV extent differs");
        // SAFETY: distinct retained scratch and packed input; producers and
        // rotary table are retained on the issuing stream by the backbone owner.
        unsafe {
            if !context {
                library.copy_d2d_2d_async(query, q * 2, input, packed * 2, q * 2, rows, stream)?;
                self.query.norm_rope(NormRopeBatch { rows, positions_capacity:self.positions,
                    input:query, weight:query_norm, positions, cos_sin:self.rope,
                    output:self.norm_scratch.buffer, scratch:self.wide.buffer },stream)?;
                library.copy_d2d_async(query,self.norm_scratch.buffer,rows * q * 2,stream)?;
            }
            let k = CuteafdDeviceBuffer { ptr:input.ptr.wrapping_byte_add(q * 2),
                bytes:input.bytes - q * 2, ..input };
            let v = CuteafdDeviceBuffer { ptr:input.ptr.wrapping_byte_add((q + kv) * 2),
                bytes:input.bytes - (q + kv) * 2, ..input };
            library.copy_d2d_2d_async(key,kv * 2,k,packed * 2,kv * 2,rows,stream)?;
            library.copy_d2d_2d_async(value,kv * 2,v,packed * 2,kv * 2,rows,stream)?;
            self.key.norm_rope(NormRopeBatch { rows, positions_capacity:self.positions,
                input:key,weight:key_norm,positions,cos_sin:self.rope,
                output:self.norm_scratch.buffer,scratch:self.wide.buffer },stream)?;
            library.copy_d2d_async(key,self.norm_scratch.buffer,rows * kv * 2,stream)?;
        }
        Ok(())
    }

    unsafe fn add_norm(&self, library:&NativeLibrary, residual:CuteafdDeviceBuffer,
        delta:CuteafdDeviceBuffer, weight:CuteafdDeviceBuffer, output:*mut c_void,
        rows:usize, stream:*mut c_void) -> Result<()> {
        // SAFETY: all spans survive this stream; BF16 addition matches checkpoint residual rounding.
        unsafe {
            self.add.launch(residual,delta,self.norm_scratch.buffer,rows * 6144,stream)?;
            library.copy_d2d_async(residual,self.norm_scratch.buffer,rows * 6144 * 2,stream)?;
            self.norm(library,residual.ptr,weight,output,rows,stream)
        }
    }

    fn inventory(&self, inventory: &mut receipts::Inventory<'_>, prefix: &str) -> Result<()> {
        for (name, owner) in [("norm-input", &self.wide), ("norm-rope", &self.norm_scratch),
            ("gate", &self.gate), ("up", &self.up)] { owner.inventory(inventory,&format!("{prefix}.{name}"))?; }
        self.projections.inventory(inventory,&format!("{prefix}.projections"))
    }
}

enum Matrix<'a> {
    Legacy(Weight<'a>),
    Native(super::weights::NativeProjection<'a>),
}
impl Matrix<'_> {
    fn inventory(&self, inventory: &mut receipts::Inventory<'_>, name: &str) -> Result<()> {
        match self { Self::Legacy(weight) => weight.allocation_receipts(inventory, name),
            Self::Native(weight) => weight.inventory(inventory, name) }
    }
}

struct DraftLayer<'a> {
    input_norm: Dev<'a>,
    post_norm: Dev<'a>,
    /// q | k | v rows; the context update reads the k | v rows.
    qkv: Matrix<'a>,
    q_norm: Dev<'a>,
    k_norm: Dev<'a>,
    o: Matrix<'a>,
    gate_up: Matrix<'a>,
    down: Matrix<'a>,
    k_ring: Dev<'a>,
    v_ring: Dev<'a>,
}

struct MarkovHead<'a> {
    w1:Dev<'a>, w2:Dev<'a>, norms:Dev<'a>, confidence_w:Dev<'a>, confidence_b:Dev<'a>,
}
struct HeadWorkspace<'a> {
    logits:Dev<'a>,tokens:Dev<'a>,confidence:Dev<'a>,markov:Dev<'a>,
    landing:crate::shared::memory::HostAllocation<'a>,
    head:Option<VocabularyHead<'a>>, _head_workspace:Dev<'a>,
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
    output: Option<HeadWorkspace<'a>>,
    anchors: Dev<'a>,
    ids: Dev<'a>,
    positions: Dev<'a>,
    tables: Dev<'a>,
    attention_workspace: Dev<'a>,
}
impl<'a> Workspace<'a> {
    fn output(&self)->Result<&HeadWorkspace<'a>> { self.output.as_ref().context("peer block drafter has no head workspace") }
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
    Launch(&'r dyn Fn(&[u32], CuteafdDeviceBuffer, *mut c_void) -> Result<()>),
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
    native: Option<NativeBackbone<'a>>,
    fc: Matrix<'a>,
    hidden_norm: Dev<'a>,
    norm: Dev<'a>,
    head: Option<MarkovHead<'a>>,
    head_split: Option<super::gqa_tp2::GqaHeadSplit<'a,DsparkDrafter<'a>>>,
    // Target producer streams outlive the family draft; events fence every private wait.
    cooperative_producers:Vec<(crate::shared::peer_split::RankDevice,device::Event<'a>)>,
    physical_rank:usize,
    physical_ranks:usize,
    lead:usize,
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
    // Shared exchange drops before this peer stream, whose device body it borrows.
    peer_stream:Option<device::Stream<'a>>,
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
        if let Some(native) = &self.native { native.inventory(inventory, &format!("{prefix}.native"))?; }
        self.fc.inventory(inventory, &format!("{prefix}.fc"))?;
        for (index, layer) in self.layers.iter().enumerate() {
            for (name, owner) in [("qkv", &layer.qkv), ("o", &layer.o), ("gate_up", &layer.gate_up), ("down", &layer.down)] {
                owner.inventory(inventory, &format!("{prefix}.layer.{index}.{name}"))?;
            }
            for (name, owner) in [("input_norm", &layer.input_norm), ("post_norm", &layer.post_norm),
                ("q_norm", &layer.q_norm), ("k_norm", &layer.k_norm)] {
                owner.inventory(inventory, &format!("{prefix}.layer.{index}.{name}"))?;
            }
        }
        for (name,owner) in [("hidden_norm",&self.hidden_norm),("norm",&self.norm)] { owner.inventory(inventory,&format!("{prefix}.{name}"))?; }
        if let Some(head)=&self.head {
            for (name,owner) in [("markov_w1",&head.w1),("markov_w2",&head.w2),("markov_norms",&head.norms),
                ("confidence_w",&head.confidence_w),("confidence_b",&head.confidence_b)] { owner.inventory(inventory,&format!("{prefix}.{name}"))?; }
        }
        if let Some(owner) = &self.inv_freq { owner.inventory(inventory, &format!("{prefix}.inv_freq"))?; }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn load_tp2_with(library:&'a NativeLibrary,programs:&'a cuteafd_ffi::programs::Programs<'a>,
        target_projection:super::dflash::TargetProjection,snapshot:&Path,slots:usize,sequences:usize,
        kv:cuteafd_loader::families::qwen4::Qwen4KvCache,representation:GlmDraftRepresentation,
        devices:[i32;2],lead_stream:*mut c_void,lead:usize,
        mut allocate:impl FnMut(i32,&str,usize)->Result<device::Allocation<'a>>)->Result<Self> {
        ensure!(lead<2 && library.cuda_get_device()?==devices[lead] && !lead_stream.is_null(),"block TP2 current device differs from lead");
        let peer_stream=device::Stream::new(device::Device { library,id:devices[1-lead] })?;
        let rank_streams=std::array::from_fn(|rank|crate::shared::peer_split::RankDevice {
            device:devices[rank],stream:if rank==lead { lead_stream } else { peer_stream.raw } });
        let slot=cuteafd_loader::formats::block_dspark::BLOCK_TP2_EXCHANGE_BYTES;
        let exchange=Rc::new(crate::shared::peer_split::PeerExchange::new_abortable(library,rank_streams,1,slot,slot)?);
        exchange.trace_identity("block-dspark-tp2-private");
        let cfg=DsparkConfig::read(snapshot)?;
        let window=cfg.window.context("block TP2 requires sliding window")?;
        let mut load=|rank,projection|Self::load_rank_with(library,programs,projection,snapshot,HashMap::new(),
            exchange.stream(rank),slots,sequences,vec![0;cfg.hidden*2],representation,
            fp8_linear::Fp8Scales::Amax,fp8_linear::Fp8Rows::W8a16,window,slots*window,kv,None,
            rank,2,lead,0,&mut allocate);
        let mut body=load(lead,target_projection)?;
        let peer_device=device::Device { library,id:exchange.device(1-lead) };
        let peer=peer_device.own(||load(1-lead,super::dflash::TargetProjection::Native).map(Box::new))?;
        body.prepare_workspace()?;
        peer.device.run(||peer.prepare_workspace())?;
        let bytes=sequences.checked_mul(cfg.block).and_then(|n|n.checked_mul(cfg.hidden)).and_then(|n|n.checked_mul(2))
            .context("block TP2 reduce extent overflow")?;
        body.head_split=Some(super::gqa_tp2::GqaHeadSplit::new(peer,exchange,lead,bytes,
            [Tag::drafter_workspace("shared/block-dspark-reduce"),Tag::transport("shared/block-dspark-reduce")])?);
        body.peer_stream=Some(peer_stream);
        Ok(body)
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
        Self::load_window_paged_with(library,programs,target_projection,snapshot,tensors,stream,slots,
            max_sequences,mask_row,representation,scales,fp8_rows,context_capacity,target_tokens,kv_cache,0,&mut allocate)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn load_window_paged_with(library: &'a NativeLibrary, programs: &'a cuteafd_ffi::programs::Programs<'a>,
        target_projection: super::dflash::TargetProjection, snapshot: &Path, tensors: HashMap<String, Vec<u8>>,
        stream: *mut c_void, slots: usize, max_sequences: usize, mask_row: Vec<u8>,
        representation: GlmDraftRepresentation, scales: fp8_linear::Fp8Scales, fp8_rows: fp8_linear::Fp8Rows,
        context_capacity: usize, target_tokens: usize, kv_cache: cuteafd_loader::families::qwen4::Qwen4KvCache,
        context_window: usize, mut allocate: impl FnMut(i32,&str,usize)->Result<device::Allocation<'a>>) -> Result<Self> {
        Self::load_rank_with(library,programs,target_projection,snapshot,tensors,stream,slots,max_sequences,
            mask_row,representation,scales,fp8_rows,context_capacity,target_tokens,kv_cache,None,
            0,1,0,context_window,&mut allocate)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn load_native_with(library: &'a NativeLibrary, programs: &'a cuteafd_ffi::programs::Programs<'a>,
        target_projection: super::dflash::TargetProjection, snapshot: &Path, stream: *mut c_void,
        slots: usize, max_sequences: usize, mask_row: Vec<u8>, rope: CuteafdDeviceBuffer, positions: usize,
        mut allocate: impl FnMut(i32, &str, usize) -> Result<crate::shared::memory::device::Allocation<'a>>) -> Result<Self> {
        let cfg = DsparkConfig::read(snapshot)?;
        let capacity = cfg.window.context("native block drafter requires a declared sliding window")?;
        Self::load_backbone_with(library, programs, target_projection, snapshot, HashMap::new(), stream,
            slots, max_sequences, mask_row, GlmDraftRepresentation::Bf16Only, fp8_linear::Fp8Scales::Pow2,
            fp8_linear::Fp8Rows::W8a16, capacity, slots.checked_mul(capacity).context("native draft capacity overflow")?,
            cuteafd_loader::families::qwen4::Qwen4KvCache::Bf16, Some((rope,positions)), &mut allocate)
    }

    #[allow(clippy::too_many_arguments)]
    fn load_backbone_with(library: &'a NativeLibrary, programs: &'a cuteafd_ffi::programs::Programs<'a>, target_projection: super::dflash::TargetProjection, snapshot: &Path, tensors: HashMap<String, Vec<u8>>, stream: *mut c_void,
        slots: usize, max_sequences: usize, mask_row: Vec<u8>, representation: GlmDraftRepresentation,
        scales: fp8_linear::Fp8Scales, fp8_rows: fp8_linear::Fp8Rows, context_capacity: usize, target_tokens: usize,
        kv_cache: cuteafd_loader::families::qwen4::Qwen4KvCache, native_rope: Option<(CuteafdDeviceBuffer,usize)>,
        allocate: &mut WeightAllocator<'a, '_>) -> Result<Self> {
        Self::load_rank_with(library,programs,target_projection,snapshot,tensors,stream,slots,max_sequences,
            mask_row,representation,scales,fp8_rows,context_capacity,target_tokens,kv_cache,native_rope,
            0,1,0,0,allocate)
    }

    #[allow(clippy::too_many_arguments)]
    fn load_rank_with(library:&'a NativeLibrary,programs:&'a cuteafd_ffi::programs::Programs<'a>,
        target_projection:super::dflash::TargetProjection,snapshot:&Path,tensors:HashMap<String,Vec<u8>>,
        stream:*mut c_void,slots:usize,max_sequences:usize,mask_row:Vec<u8>,representation:GlmDraftRepresentation,
        scales:fp8_linear::Fp8Scales,fp8_rows:fp8_linear::Fp8Rows,context_capacity:usize,target_tokens:usize,
        kv_cache:cuteafd_loader::families::qwen4::Qwen4KvCache,native_rope:Option<(CuteafdDeviceBuffer,usize)>,
        physical_rank:usize,physical_ranks:usize,lead:usize,context_window:usize,allocate:&mut WeightAllocator<'a,'_>)->Result<Self> {
        let _memory_scope = cuteafd_ffi::memory_ledger::scope("drafter");
        let mut cfg = DsparkConfig::read(snapshot)?;
        if context_window > 0 { cfg.window=Some(context_window); }
        ensure!(matches!(physical_ranks,1|2) && physical_rank<physical_ranks && lead<physical_ranks,
            "invalid block physical rank inventory");
        if physical_ranks==2 {
            ensure!(native_rope.is_none()
                && cfg.window.is_some() && cfg.heads%2==0 && cfg.kv_heads%2==0 && cfg.intermediate%2==0,
                "block TP2 requires GQA with divisible heads/neurons");
            cfg.heads/=2; cfg.kv_heads/=2; cfg.intermediate/=2;
        }
        ensure!(cfg.backbone.is_qwen3(cfg.head_dim) == native_rope.is_none(),
            "dSpark backbone needs its declared native arithmetic bindings");
        let h = cfg.hidden;
        let shapes = [(h * cfg.taps.len(), h), (h, cfg.qkv_width()), (h, 2 * cfg.kv_heads * cfg.head_dim),
            (cfg.q_width(), h), (h, 2 * cfg.intermediate), (cfg.intermediate, h)];
        let native_bf16 = [(h * cfg.taps.len(), h)];
        let projections = cuteafd_ffi::draft_projection::DraftProjections::load(programs,
            if native_rope.is_some() { &native_bf16 } else if representation == GlmDraftRepresentation::Bf16Only { &shapes } else { &[] },
            cuteafd_ffi::draft_projection::batch_rows(TAP_ROWS, cfg.block, max_sequences,
                programs.capacities().decode_rows.unwrap_or(0)))?;
        ensure!(context_capacity > 0 && slots > 0 && max_sequences <= slots,
            "dSpark requires nonzero admitted context slots and capacity");
        ensure!(cfg.window.is_none_or(|window| window == context_capacity),
            "sliding dSpark capacity must equal its checkpoint window");
        ensure!(slots.checked_mul(context_capacity).is_some_and(|rows| rows <= i32::MAX as usize),
            "dSpark context capacity exceeds signed32 native slots");
        ensure!(cfg.hidden.checked_mul(2) == Some(mask_row.len()), "dSpark mask row shape");
        let mut config:serde_json::Value = serde_json::from_slice(&std::fs::read(snapshot.join("config.json"))?)?;
        if context_window > 0 {
            let c = if config["speculators_model_type"] == "dspark" { &mut config["transformer_layer_config"] } else { &mut config };
            c["sliding_window"]=serde_json::json!(context_window);
            c["layer_types"]=serde_json::json!(vec!["sliding_attention";cfg.layers]);
        }
        let cache_layout = cuteafd_loader::formats::block_dspark::BlockDsparkConfig::from_json(&config)?
            .paged_cache_layout(slots, context_capacity, target_tokens, kv_cache)?;
        let cache_rows = cache_layout.layer_bytes[0] / cache_layout.record_bytes;
        let residency = cuteafd_loader::families::glm5::draft_representation::block_dspark_paged_resident_bytes(
            &config, slots, max_sequences, library.sm_count()? as u64, representation, fp8_rows.code() as u8,
            context_capacity, target_tokens, kv_cache)?;
        tracing::info!(resident_bytes = residency.owned, scratch_bytes = residency.scratch, kv_cache = %kv_cache, "dSpark shared readiness inventory");
        ensure!((1..=MAX_SEQUENCES).contains(&max_sequences), "dSpark drafts take 1..={MAX_SEQUENCES} sequences");
        let load = if native_rope.is_some() || physical_ranks==2 { Load::open(library, snapshot)? }
            else { Load::with_tensors(library, snapshot, tensors)? }.with_allocator(allocate);
        let native_shapes = [(h, cfg.q_width()), (h,cfg.kv_width()), (cfg.q_width(),h),
            (h,cfg.intermediate), (cfg.intermediate,h)];
        let native_workspace = native_rope.map(|_| super::weights::NativeProjectionWorkspace::new(library,
            &native_shapes, TAP_ROWS.max(max_sequences * cfg.block)).map(Rc::new)).transpose()?;
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
        let weights=if physical_ranks==2 {
            let source=cuteafd_loader::formats::block_dspark::BlockDsparkConfig::read(snapshot)?;
            cuteafd_loader::formats::block_dspark::block_dspark_rank_recipe(&source,physical_rank,physical_ranks,lead,
                |name,shape|load.tensor(name,shape),|parts,columns,axis| {
                    match axis {
                        Some(axis)=>load.matrix_shard(parts,columns,axis,physical_rank,physical_ranks,representation==GlmDraftRepresentation::Fp8Only,scales,stream),
                        None=>load.matrix(parts,columns,representation==GlmDraftRepresentation::Fp8Only,scales,stream),
                    }.map(Matrix::Legacy)
                })?
        } else {
            let weights = weight_recipe(snapshot, |name, shape| load.tensor(name, shape),
            |parts, cols| {
                if let Some(workspace) = &native_workspace {
                    if parts.len() == 1 && parts[0].0.ends_with("dspark.fc.weight") {
                        return load.tensor(parts[0].0, &[parts[0].1,cols]).map(Weight::Bf16).map(Matrix::Legacy);
                    }
                    return super::weights::NativeProjection::load(&load,programs,parts,cols,Rc::clone(workspace)).map(Matrix::Native);
                }
                load.matrix(parts, cols, representation == GlmDraftRepresentation::Fp8Only, scales, stream).map(Matrix::Legacy)
            })?;
            cuteafd_loader::formats::block_dspark::BlockDsparkRankWeights { layers:weights.layers,fc:weights.fc,
                hidden_norm:weights.hidden_norm,norm:weights.norm,
                head:Some((weights.markov_w1,weights.markov_w2,weights.confidence_w,weights.confidence_b)) }
        };
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
        let head=weights.head.map(|(w1,w2,confidence_w,confidence_b)|->Result<_> {
            let norms=load.allocation("markov_norms",cfg.vocab*4)?;
            load.row_norms(w2.buffer,norms.buffer,cfg.vocab,cfg.rank,stream)?;
            Ok(MarkovHead { w1,w2,norms,confidence_w,confidence_b })
        }).transpose()?;
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
            inv_freq: if native_rope.is_some() { None } else {
                cfg.rotary.as_ref().map(|plan| upload("inv_freq", bytes_of(&plan.inv_freq))).transpose()? },
            fc: weights.fc, hidden_norm: weights.hidden_norm, norm: weights.norm,
            head,head_split:None,cooperative_producers:Vec::new(),physical_rank,physical_ranks,lead,
            layers,
            taps: zeroed(TAP_ROWS * taps * 2, Tag::drafter_workspace("shared/dspark-taps"))?,
            fused: zeroed(TAP_ROWS * h * 2, Tag::drafter_workspace("shared/dspark-context-update"))?,
            fused_norm: zeroed(TAP_ROWS * h * 2, Tag::drafter_workspace("shared/dspark-context-update"))?,
            context_kv: zeroed(TAP_ROWS * 2 * kv * 2, Tag::drafter_workspace("shared/dspark-context-update"))?,
            context_positions: zeroed(TAP_ROWS * 8, Tag::drafter_workspace("shared/dspark-context-update"))?,
            context_slots: zeroed(TAP_ROWS * 4, Tag::drafter_workspace("shared/dspark-context-update"))?,
            mask_row,
            target_projection, workspace: RefCell::new(None),
            projections, representation, native: native_rope.map(|(rope,positions)|
                NativeBackbone::bind(library,programs,&cfg,rope,positions,native_workspace.expect("native workspace"))).transpose()?,
            fp8_workspace,
            head_mode: Cell::new(DraftHead::Exact),
            fp8_rows: Cell::new(fp8_rows),
            fp8_admitted: fp8_rows,
            peer_stream:None,
            cfg,
        })
    }

    fn on_device<T>(&self, enqueue: impl FnOnce() -> Result<T>) -> Result<T> {
        device::Device { library: self.library, id: self.hidden_norm.buffer.device_id }.run(enqueue)
    }

    /// Retained owners including canonical K/V and prepared request scratch.
    pub fn runtime_receipts(&self, devices:cuteafd_loader::placement::planner::CoordinatorDevices<'_>,
        prefix:&str) -> Result<Vec<cuteafd_loader::placement::planner::Receipt>> {
        use cuteafd_loader::placement::planner::{Lifetime,Phase};
        use cuteafd_core::memory_layout::Category;
        let mut inventory = receipts::Inventory::new(devices,prefix,
            Lifetime { first:Phase::DraftLoad,last:Phase::Growth },Category::Drafter,"native draft owners");
        self.allocation_receipts(&mut inventory,"")?;
        for (index,layer) in self.layers.iter().enumerate() {
            layer.k_ring.inventory(&mut inventory,&format!("layer.{index}.k_ring"))?;
            layer.v_ring.inventory(&mut inventory,&format!("layer.{index}.v_ring"))?;
        }
        for (name,owner) in [("taps",&self.taps),("fused",&self.fused),("fused_norm",&self.fused_norm),
            ("context_kv",&self.context_kv),("context_positions",&self.context_positions),("context_slots",&self.context_slots)] {
            owner.inventory(&mut inventory,name)?;
        }
        if let Some(context) = &self.paged { context.tables.inventory(&mut inventory,"context.page-table")?; }
        if let Some(owner) = &self.fp8_workspace { owner.inventory(&mut inventory,"workspace.fp8-linear")?; }
        if let Some(workspace) = self.workspace.borrow().as_ref() {
            for (name,owner) in [("h",&workspace.h),("n",&workspace.n),("qkv",&workspace.qkv),
                ("q",&workspace.q),("k",&workspace.k),("v",&workspace.v),("attn",&workspace.attn),
                ("delta",&workspace.delta),("gate_up",&workspace.gate_up),("act",&workspace.act),
                ("anchors",&workspace.anchors),("ids",&workspace.ids),("positions",&workspace.positions),
                ("tables",&workspace.tables),("attention",&workspace.attention_workspace)] {
                owner.inventory(&mut inventory,&format!("workspace.{name}"))?;
            }
            if let Some(output)=&workspace.output {
                for (name,owner) in [("logits",&output.logits),("tokens",&output.tokens),("confidence",&output.confidence),
                    ("head",&output._head_workspace),("markov",&output.markov)] {
                    owner.inventory(&mut inventory,&format!("workspace.{name}"))?;
                }
                inventory.pinned("workspace.landing",output.landing.buffer)?;
            }
        }
        for receipt in &mut inventory.receipts {
            if receipt.allocation.id.starts_with(&format!("{prefix}..")) {
                receipt.allocation.id = receipt.allocation.id.replacen("..",".",1);
            }
        }
        if let Some(split)=&self.head_split {
            split.reduced.inventory(&mut inventory,"tp2.reduced")?;
            split.peer.device.run(||split.peer_reduced.inventory(&mut inventory,"peer.tp2.reduced"))?;
            let mut transport=receipts::Inventory::new(devices,prefix,
                Lifetime { first:Phase::DraftLoad,last:Phase::Growth },Category::Transport,"block dSpark private exchange");
            for (rank,name,buffer) in split.exchange.allocation_buffers() {
                transport.device(&format!("tp2.exchange.{rank}.{name}"),buffer)?;
            }
            inventory.receipts.extend(transport.receipts);
            inventory.receipts.extend(split.peer.device.run(||split.peer.runtime_receipts(devices,&format!("{prefix}.peer")))?);
        }
        Ok(inventory.receipts)
    }

    /// How draft steps run the borrowed BF16 head (native head launchers ignore it).
    pub fn set_draft_head(&self, mode: DraftHead) {
        self.head_mode.set(mode);
    }

    /// Every checkpoint-sourced device owner (weights, FP8 values and scales, the
    /// Markov norms finish), named, for push-versus-mount byte equality.
    #[cfg(test)]
    pub(crate) fn weight_buffers(&self) -> Vec<(String, CuteafdDeviceBuffer)> {
        let mut out = Vec::new();
        let mut weight = |name: String, weight: &Matrix<'_>| match weight {
            Matrix::Legacy(Weight::Bf16(owner)) => out.push((name, owner.buffer)),
            Matrix::Legacy(Weight::Fp8(owner)) => {
                let (values, scales) = owner.buffers();
                out.push((format!("{name}.values"), values));
                out.push((format!("{name}.scales"), scales));
            }
            Matrix::Native(_) => panic!("legacy weight digest requires a legacy projection"),
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
        for (name,owner) in [("hidden_norm",&self.hidden_norm),("norm",&self.norm)] { out.push((name.into(),owner.buffer)); }
        if let Some(head)=&self.head {
            for (name,owner) in [("markov_w1",&head.w1),("markov_w2",&head.w2),("markov_norms",&head.norms),
                ("confidence_w",&head.confidence_w),("confidence_b",&head.confidence_b)] { out.push((name.into(),owner.buffer)); }
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
    unsafe fn linear(&self, x: *const c_void, weight: &Matrix<'_>, first: usize, out: *mut c_void, rows: usize,
        k: usize, n: usize) -> Result<()> {
        use super::weights::{LegacyProjection, Projection};
        match weight {
            Matrix::Legacy(weight) => {
                let projection = LegacyProjection { weight, library: self.library, programs: &self.projections,
                    scratch: self.fp8_workspace.as_ref(), mode: self.fp8_rows.get() };
                // SAFETY: caller retains matrix/input/output; admitted mode bounds scratch.
                unsafe { projection.project(x, first, out, rows, k, n, self.stream) }
            }
            // SAFETY: same caller contract, native adapter retains records and output scratch.
            Matrix::Native(projection) => unsafe { projection.project(x, first, out, rows, k, n, self.stream) },
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
        self.on_device(|| {
            // SAFETY: caller retains target streams and this owner retains the
            // tap destination/stream; all launches use the owner's device.
            unsafe { self.library.glm_dflash_tap_mean(streams.cast::<u8>().add(first * hc * h * 2).cast(),
                self.taps.buffer.ptr.cast::<u8>().add(to * width * 2).cast(), n, h, hc, width, index * h, self.stream) }
        })
    }

    /// Copy a row span already concatenated in checkpoint tap order. The
    /// producer's ordered stream must be this drafter's stream.
    ///
    /// # Safety
    /// `source` holds initialized BF16 rows of taps*hidden elements and its
    /// owner survives the queued copy; source/destination do not overlap.
    pub unsafe fn capture_concatenated(&self, source:CuteafdDeviceBuffer, first:usize,
        rows:usize, to:usize, stream:*mut c_void) -> Result<()> {
        let pitch = self.cfg.taps.len().checked_mul(self.cfg.hidden).and_then(|n| n.checked_mul(2))
            .context("draft concatenated tap pitch overflow")?;
        let end = first.checked_add(rows).and_then(|n| n.checked_mul(pitch)).context("source tap extent overflow")?;
        ensure!(stream == self.stream && source.device_id == self.taps.buffer.device_id
            && rows > 0 && end <= source.bytes && to.checked_add(rows).is_some_and(|n| n <= TAP_ROWS),
            "concatenated target taps differ from admitted drafter span/stream");
        let input = CuteafdDeviceBuffer { ptr:source.ptr.wrapping_byte_add(first * pitch),bytes:rows * pitch,..source };
        let output = CuteafdDeviceBuffer { ptr:at(&self.taps,to * pitch),bytes:rows * pitch,..self.taps.buffer };
        // SAFETY: caller retains the checked disjoint source; this owner retains taps.
        self.on_device(|| {
            // SAFETY: caller retains the checked disjoint source and this
            // drafter retains its destination and stream on the scoped device.
            unsafe { self.library.copy_d2d_async(output,input,rows * pitch,self.stream) }
        })
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
        let device_id = self.hidden_norm.buffer.device_id;
        self.on_device(|| self.capture_target_taps_on_device(taps, first, n, to, width_bytes, pitch, device_id))
    }

    fn capture_target_taps_on_device(&self, taps: &[super::target::HiddenTap], first: usize,
        n: usize, to: usize, width_bytes: usize, pitch: usize, device_id: i32) -> Result<()> {
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
        self.checked(||self.on_device(|| self.update_on_device(rows)))
    }

    fn update_on_device(&self, rows: &[ContextRow]) -> Result<()> {
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
        self.order_cooperative_producers()?;
        if let Some(split)=&self.head_split {
            if !rows.is_empty() {
                let first=rows.iter().map(|r|r.tap_row).min().unwrap();
                let last=rows.iter().map(|r|r.tap_row).max().unwrap();
                let stride=self.cfg.taps.len()*self.cfg.hidden*2;
                split.exchange.push_to(split.lead,crate::shared::peer_split::DIRECT,
                    self.taps.buffer.ptr.wrapping_byte_add(first*stride),
                    split.peer.taps.buffer.ptr.wrapping_byte_add(first*stride),(last-first+1)*stride)?;
                split.exchange.wait(1-split.lead,crate::shared::peer_split::DIRECT)?;
                split.peer.device.run(||split.peer.update_on_device(rows))?;
                split.exchange.signal(1-split.lead,crate::shared::peer_split::DIRECT)?;
                split.exchange.wait(split.lead,crate::shared::peer_split::DIRECT)?;
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
            if let Some(native) = &self.native {
                ensure!(positions.iter().all(|&p| p >= 0 && (p as usize) < native.positions),
                    "native draft context position exceeds admitted rotary table");
                native.norm(self.library,self.fused.buffer.ptr,self.hidden_norm.buffer,self.fused_norm.buffer.ptr,n,s)?;
            } else {
                self.library.glm_dflash_rmsnorm(self.fused.buffer.ptr, self.hidden_norm.buffer.ptr,
                    self.fused_norm.buffer.ptr, n, h, self.cfg.eps, s)?;
            }
            let q_rows = self.cfg.q_width();
            for layer in &self.layers {
                self.linear(self.fused_norm.buffer.ptr, &layer.qkv, q_rows, self.context_kv.buffer.ptr, n, h, 2 * kv)?;
                if let Some(native) = &self.native {
                    native.rope(&self.cfg,self.context_kv.buffer,layer.q_norm.buffer,layer.k_norm.buffer,
                        self.context_positions.buffer,CuteafdDeviceBuffer::default(),native.gate.buffer,native.up.buffer,n,true,s)?;
                    let row_bytes = kv * 2;
                    for group in groups.values() {
                        let mut offset = 0;
                        while offset < group.len() {
                            let begin = group[offset];
                            let mut count = 1;
                            while offset + count < group.len() {
                                let next = group[offset + count];
                                if next.tap_row != begin.tap_row + count || next.position != begin.position + count
                                    || (begin.position % self.context_capacity) + count >= self.context_capacity { break; }
                                count += 1;
                            }
                            let source = (begin.tap_row - first) * row_bytes;
                            let destination = usize::try_from(self.cfg.context_row(begin.slot,begin.position,self.slots,self.context_capacity)?)? * row_bytes;
                            for (src,dst) in [(&native.gate,&layer.k_ring),(&native.up,&layer.v_ring)] {
                                let from = CuteafdDeviceBuffer { ptr:at(src,source),bytes:count * row_bytes,..src.buffer };
                                let to = CuteafdDeviceBuffer { ptr:at(dst,destination),bytes:count * row_bytes,..dst.buffer };
                                self.library.copy_d2d_async(to,from,count * row_bytes,s)?;
                            }
                            offset += count;
                        }
                    }
                } else {
                    self.library.block_dspark_qk_rope(self.context_kv.buffer.ptr, layer.q_norm.buffer.ptr,
                        layer.k_norm.buffer.ptr, self.context_positions.buffer.ptr, self.context_slots.buffer.ptr,
                        self.inv_freq.as_ref().map_or(std::ptr::null(), |v| v.buffer.ptr.cast_const()), std::ptr::null_mut(), layer.k_ring.buffer.ptr, layer.v_ring.buffer.ptr, n, 0, self.cfg.kv_heads, self.cfg.head_dim,
                        self.cfg.theta, self.cfg.rotary.as_ref().map_or(1.0, |p| p.attention_factor), self.cfg.eps, self.kv_cache == cuteafd_loader::families::qwen4::Qwen4KvCache::Fp8, s)?;
                }
            }
        }
        lease.commit()
    }

    pub(crate) fn reset_context(&self, slot: usize, origin: usize) -> Result<()> {
        self.on_device(|| self.reset_context_on_device(slot, origin))
    }

    fn reset_context_on_device(&self, slot: usize, origin: usize) -> Result<()> {
        if let Some(split)=&self.head_split { split.peer.device.run(||split.peer.reset_context_on_device(slot,origin))?; }
        ensure!(slot < self.slots && origin <= self.context_capacity, "block dSpark reset outside admission");
        if let Some(context) = &self.paged {
            // SAFETY: retire the previous request before releasing its page ownership.
            unsafe { self.library.cuda_stream_synchronize(self.stream)?; }
            let mut state = context.state.borrow_mut();
            state.discard(slot)?;
            state.bindings[slot] = Some(state.pool.bind(slot, origin)?);
        } else if origin == 0 {
            for plane in self.window_planes(slot)?.into_iter().filter(|p|p.device_id==self.hidden_norm.buffer.device_id) {
                // SAFETY: fresh slot admission precedes every new consumer;
                // zero preserves deterministic inactive bytes for exact marks.
                unsafe { self.library.cuda_zero_bytes_async(plane,plane.bytes,self.stream)?; }
            }
        }
        Ok(())
    }

    pub(crate) fn quarantine_pages(&self, slot: usize) {
        if let Some(split) = &self.head_split { split.peer.quarantine_pages(slot); }
        if let Some(context) = &self.paged {
            let mut state = context.state.borrow_mut();
            if let Some(binding) = state.bindings.get(slot).copied().flatten() {
                state.pool.quarantine(binding).expect("live block binding belongs to its pool");
            }
        }
    }

    pub(crate) fn discard_context(&self, slot: usize) -> Result<()> {
        self.on_device(|| self.discard_context_on_device(slot))
    }

    fn discard_context_on_device(&self, slot: usize) -> Result<()> {
        if let Some(split)=&self.head_split { split.peer.device.run(||split.peer.discard_context_on_device(slot))?; }
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

    pub(crate) fn window_readers(&self)->Vec<crate::shared::prefix::Reader> {
        let mut readers=vec![crate::shared::prefix::Reader { device:self.hidden_norm.buffer.device_id,stream:self.stream }];
        if let Some(split)=&self.head_split { readers.extend(split.peer.window_readers()); }
        readers.sort_by_key(|r|r.device); readers
    }

    /// Slot-local physical planes, including inactive bytes, for exact window
    /// snapshots. No tap replay or weight-dependent reconstruction is involved.
    pub(crate) fn window_planes(&self, slot: usize) -> Result<Vec<CuteafdDeviceBuffer>> {
        ensure!(self.cfg.window == Some(self.context_capacity) && self.paged.is_none()
            && slot < self.slots, "block dSpark has no admitted window slot");
        let bytes = self.context_capacity.checked_mul(self.kv_cache.plane_record_bytes(
            self.cfg.kv_heads, self.cfg.head_dim)).context("block dSpark window extent overflow")?;
        let offset = slot.checked_mul(bytes).context("block dSpark window offset overflow")?;
        let mut planes=self.layers.iter().flat_map(|layer| [&layer.k_ring, &layer.v_ring]).map(|plane| {
            ensure!(offset.checked_add(bytes).is_some_and(|end| end <= plane.buffer.bytes),
                "block dSpark window plane outside retained backing");
            Ok(CuteafdDeviceBuffer { ptr: at(plane, offset), bytes, ..plane.buffer })
        }).collect::<Result<Vec<_>>>()?;
        if let Some(split)=&self.head_split { planes.extend(split.peer.device.run(||split.peer.window_planes(slot))?); }
        Ok(planes)
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
        ensure!(dev.buffer.device_id == self.hidden_norm.buffer.device_id,
            "block dSpark upload destination is not on its owner device");
        self.on_device(|| {
            // SAFETY: destination and stream share this retained device; the
            // ordered uploader creates and records its event in this scope.
            unsafe { self.library.copy_h2d_ordered(dev.buffer, bytes, self.stream) }
        })
    }

    pub(crate) fn bind_cooperative_producers(&mut self,producers:&[crate::shared::peer_split::RankDevice])->Result<()> {
        if let Some(split)=&self.head_split {
            ensure!(producers.len()==2 && self.cooperative_producers.is_empty(),"block cooperative producers already bound/missing");
            for (rank,&producer) in producers.iter().enumerate() {
                let event=device::Event::new(device::Device { library:self.library,id:producer.device })?;
                split.exchange.require_cooperative_producer_event(rank,producer,event.device.id,event.raw)?;
                self.cooperative_producers.push((producer,event));
            }
        }
        Ok(())
    }
    fn order_cooperative_producers(&self)->Result<()> {
        if let Some(split)=&self.head_split {
            ensure!(self.cooperative_producers.len()==2,"block TP2 producer events must be bound before exchange waits");
            for (rank,(producer,event)) in self.cooperative_producers.iter().enumerate() {
                split.exchange.require_cooperative_producer_event(rank,*producer,event.device.id,event.raw)?;
                event.device.run(|| {
                    // SAFETY: the target retains this producer stream and this
                    // event through draft retirement; waiter is its local private stream.
                    unsafe {
                        self.library.cuda_event_record(event.raw,producer.stream)?;
                        self.library.cuda_stream_wait_event(split.exchange.stream(rank),event.raw)
                    }
                })?;
            }
        }
        Ok(())
    }

    pub(crate) fn require_live(&self)->Result<()> {
        self.head_split.as_ref().map_or(Ok(()),|s|s.exchange.require_live())
    }
    pub(crate) fn abort(&self)->Result<()> {
        self.head_split.as_ref().map_or(Ok(()),|s|s.exchange.publish_abort())
    }
    pub(crate) fn finish_terminal(&self,drained:bool) {
        if let Some(split)=&self.head_split {
            if split.exchange.terminal_state()==crate::shared::peer_split::TerminalState::Aborting {
                split.exchange.finish_terminal(drained);
            }
        }
    }
    pub(crate) fn checked<T>(&self,work:impl FnOnce()->Result<T>)->Result<T> {
        self.require_live()?;
        match work() {
            Ok(value)=>Ok(value),
            Err(error)=>match self.abort() {
                Ok(())=>Err(error),
                Err(abort)=>Err(error.context(format!("block draft abort publication failed: {abort:#}"))),
            }
        }
    }

    pub(crate) fn drain(&self) -> Result<()> {
        self.on_device(|| {
            // SAFETY: this owner retains its stream and every draft operand through retirement.
            unsafe { self.library.cuda_stream_synchronize(self.stream)?; }
            if let Some(split)=&self.head_split { split.peer.device.run(||split.peer.drain())?; }
            Ok(())
        })
    }

    /// Explicit serving startup admission; other callers retain lazy allocation.
    pub(crate) fn prepare_workspace(&self) -> Result<()> {
        if self.workspace.borrow().is_none() {
            *self.workspace.borrow_mut() = Some(self.workspace(self.max_sequences)?);
        }
        Ok(())
    }

    fn workspace(&self, sequences: usize) -> Result<Workspace<'a>> {
        let device = device::Device { library: self.library, id: self.hidden_norm.buffer.device_id };
        device.run(|| self.workspace_on_device(sequences))
    }

    fn workspace_on_device(&self, sequences: usize) -> Result<Workspace<'a>> {
        let _memory_scope = cuteafd_ffi::memory_ledger::scope("drafter/workspace");
        let c = &self.cfg;
        let rows = sequences * c.block;
        let alloc = |bytes: usize| DeviceAllocation::new(self.library, bytes.max(256), Tag::drafter_workspace("shared/dspark"));
        let output=if self.head.is_some() {
            let head_workspace=alloc(VOCABULARY_HEAD_WORKSPACE)?;
            Some(HeadWorkspace {
                logits:alloc(rows*c.vocab*4)?,tokens:alloc(rows*4)?,confidence:alloc(rows*4)?,
                markov:alloc(self.library.glmf_dspark_markov_workspace(sequences,c.block)?)?,
                landing:crate::shared::memory::HostAllocation::new(self.library,rows*8,Tag::staging("shared/dspark-landing"))?,
                head:if self.target_projection==super::dflash::TargetProjection::Bf16 {
                    // SAFETY: retained head workspace outlives the borrowed vocabulary handle.
                    Some(unsafe { self.library.vocabulary_head_rows(head_workspace.buffer.ptr,c.hidden as u32,
                        rows as u32,c.vocab as u32,cuteafd_ffi::programs::HeadMode::Gemm)? })
                } else { None },_head_workspace:head_workspace,
            })
        } else { None };
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
            output,
            anchors: alloc(sequences * 4)?,
            ids: alloc(rows * 4)?,
            positions: alloc(rows * 8)?,
            tables: alloc(3 * sequences * 4)?,
            attention_workspace,
        })
    }

    fn forward_tp2(&self,w:&Workspace<'a>,split:&super::gqa_tp2::GqaHeadSplit<'a,Self>,
        positions:&[i64],tables:&[i32],sequences:usize,max_keys:usize)->Result<()> {
        use crate::shared::peer_split::DIRECT;
        self.order_cooperative_producers()?;
        let rows=sequences*self.cfg.block; let h=self.cfg.hidden;
        let peer_work=split.peer.workspace.borrow(); let pw=peer_work.as_ref().context("block TP2 peer scratch not prepared")?;
        split.peer.device.run(|| { split.peer.put(&pw.positions,bytes_of(positions))?; split.peer.put(&pw.tables,bytes_of(tables)) })?;
        split.exchange.push_to(split.lead,DIRECT,w.h.buffer.ptr,pw.h.buffer.ptr,rows*h*2)?;
        split.exchange.wait(1-split.lead,DIRECT)?;
        self.norm_first(w,rows)?;
        split.peer.device.run(||split.peer.norm_first(pw,rows))?;
        for index in 0..self.layers.len() {
            self.attention_partial(w,index,sequences,max_keys)?;
            split.peer.device.run(||split.peer.attention_partial(pw,index,sequences,max_keys))?;
            split.reduce([w.delta.buffer,pw.delta.buffer],rows*h)?;
            self.ffn_partial(w,index,rows,split.reduced.buffer)?;
            split.peer.device.run(||split.peer.ffn_partial(pw,index,rows,split.peer_reduced.buffer))?;
            split.reduce([w.delta.buffer,pw.delta.buffer],rows*h)?;
            self.finish_layer(w,index,rows,split.reduced.buffer)?;
            split.peer.device.run(||split.peer.finish_layer(pw,index,rows,split.peer_reduced.buffer))?;
        }
        split.exchange.signal(1-split.lead,DIRECT)?;
        split.exchange.wait(split.lead,DIRECT)
    }

    fn norm_first(&self,w:&Workspace<'_>,rows:usize)->Result<()> {
        // SAFETY: local residual/norm owners cover the admitted block on this retained stream.
        unsafe { self.library.glm_dflash_rmsnorm(w.h.buffer.ptr,self.layers[0].input_norm.buffer.ptr,
            w.n.buffer.ptr,rows,self.cfg.hidden,self.cfg.eps,self.stream) }
    }
    fn attention_partial(&self,w:&Workspace<'_>,index:usize,sequences:usize,max_keys:usize)->Result<()> {
        let c=&self.cfg; let layer=&self.layers[index]; let rows=sequences*c.block;
        // SAFETY: rank-local weights and KV use the checked local-head geometry and retained stream.
        unsafe {
            self.linear(w.n.buffer.ptr,&layer.qkv,0,w.qkv.buffer.ptr,rows,c.hidden,c.qkv_width())?;
            self.library.block_dspark_qk_rope(w.qkv.buffer.ptr,layer.q_norm.buffer.ptr,layer.k_norm.buffer.ptr,
                w.positions.buffer.ptr,std::ptr::null(),self.inv_freq.as_ref().map_or(std::ptr::null(),|v|v.buffer.ptr.cast_const()),
                w.q.buffer.ptr,w.k.buffer.ptr,w.v.buffer.ptr,rows,c.heads,c.kv_heads,c.head_dim,c.theta,
                c.rotary.as_ref().map_or(1.,|p|p.attention_factor),c.eps,false,self.stream)?;
            self.library.block_dspark_attention(w.q.buffer.ptr,w.k.buffer.ptr,w.v.buffer.ptr,layer.k_ring.buffer.ptr,
                layer.v_ring.buffer.ptr,w.tables.buffer.ptr,at(&w.tables,sequences*4),at(&w.tables,sequences*8),
                w.attn.buffer.ptr,w.attention_workspace.buffer.ptr,sequences,c.block,c.heads,c.kv_heads,self.context_capacity,
                max_keys,c.window.unwrap_or(0),c.causal,c.head_dim,1./(c.head_dim as f32).sqrt(),
                self.kv_cache==cuteafd_loader::families::qwen4::Qwen4KvCache::Fp8,self.stream)?;
            self.linear(w.attn.buffer.ptr,&layer.o,0,w.delta.buffer.ptr,rows,c.q_width(),c.hidden)
        }
    }
    fn ffn_partial(&self,w:&Workspace<'_>,index:usize,rows:usize,delta:CuteafdDeviceBuffer)->Result<()> {
        let c=&self.cfg; let layer=&self.layers[index];
        // SAFETY: reduced full-hidden attention updates replicated residual before local-neuron projections.
        unsafe {
            self.library.glmf_dspark_add_rmsnorm(w.h.buffer.ptr,delta.ptr,layer.post_norm.buffer.ptr,
                w.h.buffer.ptr,w.n.buffer.ptr,rows,c.hidden,c.eps,self.stream)?;
            self.linear(w.n.buffer.ptr,&layer.gate_up,0,w.gate_up.buffer.ptr,rows,c.hidden,2*c.intermediate)?;
            self.library.glm_dflash_silu_mul(w.gate_up.buffer.ptr,w.act.buffer.ptr,rows,c.intermediate,self.stream)?;
            self.linear(w.act.buffer.ptr,&layer.down,0,w.delta.buffer.ptr,rows,c.intermediate,c.hidden)
        }
    }
    fn finish_layer(&self,w:&Workspace<'_>,index:usize,rows:usize,delta:CuteafdDeviceBuffer)->Result<()> {
        let next=self.layers.get(index+1).map_or(self.norm.buffer,|layer|layer.input_norm.buffer);
        // SAFETY: shared reduction and next norm are retained full-hidden operands on this rank's stream.
        unsafe { self.library.glmf_dspark_add_rmsnorm(w.h.buffer.ptr,delta.ptr,next.ptr,w.h.buffer.ptr,w.n.buffer.ptr,
            rows,self.cfg.hidden,self.cfg.eps,self.stream) }
    }

    fn head_ref<'h>(&self, head: &'h TargetHead<'h>) -> Result<HeadRef<'h>> {
        Ok(match head {
            TargetHead::Bf16(owner) => {
                ensure!(owner.buffer.bytes == self.cfg.vocab * self.cfg.hidden * 2,
                    "dSpark borrows the target BF16 head [{}, {}]", self.cfg.vocab, self.cfg.hidden);
                HeadRef::Bf16(owner.buffer.ptr)
            }
            TargetHead::Bf16Buffer(buffer) => {
                ensure!(buffer.bytes == self.cfg.vocab * self.cfg.hidden * 2 && !buffer.ptr.is_null(),
                    "dSpark borrowed target head extent differs");
                HeadRef::Bf16(buffer.ptr)
            },
            TargetHead::Launch(launch) => HeadRef::Launch(launch),
        })
    }

    /// [`Self::draft`] with the block's input rows gathered on the device
    /// from the target's embedding table.
    pub fn draft_device(&self, sequences: &[DraftSeq], embedding: &TokenEmbedding<'_>, head: TargetHead<'_>)
        -> Result<Vec<Draft>> {
        self.draft_from(sequences, DraftInput::Table(embedding), self.head_ref(&head)?, true)
    }

    /// Device-driven input binding for a target whose embedding lives on a peer.
    pub fn draft_device_with(&self, sequences: &[DraftSeq],
        input: &dyn Fn(&[u32], CuteafdDeviceBuffer, *mut c_void) -> Result<()>, head: TargetHead<'_>)
        -> Result<Vec<Draft>> {
        self.draft_from(sequences, DraftInput::Launch(input), self.head_ref(&head)?, true)
    }

    pub fn draft_device_diagnostic(&self, sequences: &[DraftSeq], embedding: &TokenEmbedding<'_>,
        head: TargetHead<'_>) -> Result<Vec<Draft>> {
        self.draft_from(sequences, DraftInput::Table(embedding), self.head_ref(&head)?, false)
    }

    fn draft_from(&self, sequences: &[DraftSeq], input: DraftInput<'_, '_>, head: HeadRef<'_>, serving: bool)
        -> Result<Vec<Draft>> {
        self.checked(||self.on_device(|| self.draft_on_device(sequences, input, head, serving)))
    }

    fn draft_on_device(&self, sequences: &[DraftSeq], input: DraftInput<'_, '_>, head: HeadRef<'_>, serving: bool)
        -> Result<Vec<Draft>> {
        ensure!(!matches!(head, HeadRef::Bf16(p) if p.is_null()), "dSpark has no borrowed target head");
        let c = &self.cfg;
        let (s_count, block, h) = (sequences.len(), c.block, c.hidden);
        let rows_ok = match input {
            DraftInput::Rows(r) => r.len() == s_count * h * 2,
            DraftInput::Table(_) | DraftInput::Launch(_) => true,
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
            DraftInput::Launch(launch) => {
                let ids = sequences.iter().flat_map(|seq| std::iter::once(seq.anchor)
                    .chain(std::iter::repeat_n(c.mask_token,block - 1))).collect::<Vec<_>>();
                launch(&ids,w.h.buffer,self.stream)?;
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
            if let Some(split)=&self.head_split {
                self.forward_tp2(w,split,&positions,&tables,s_count,max_keys)?;
            } else {
            if let Some(native) = &self.native {
                ensure!(positions.iter().all(|&p| p >= 0 && (p as usize) < native.positions),
                    "native draft block position exceeds admitted rotary table");
                native.norm(l,w.h.buffer.ptr,self.layers[0].input_norm.buffer,w.n.buffer.ptr,rows,s)?;
            } else {
                l.glm_dflash_rmsnorm(w.h.buffer.ptr, self.layers[0].input_norm.buffer.ptr, w.n.buffer.ptr, rows, h, eps, s)?;
            }
            for (index, layer) in self.layers.iter().enumerate() {
                self.linear(w.n.buffer.ptr, &layer.qkv, 0, w.qkv.buffer.ptr, rows, h, c.qkv_width())?;
                if let Some(native) = &self.native {
                    native.rope(c,w.qkv.buffer,layer.q_norm.buffer,layer.k_norm.buffer,w.positions.buffer,
                        w.q.buffer,w.k.buffer,w.v.buffer,rows,false,s)?;
                } else {
                    l.block_dspark_qk_rope(w.qkv.buffer.ptr, layer.q_norm.buffer.ptr, layer.k_norm.buffer.ptr,
                        w.positions.buffer.ptr, std::ptr::null(),
                        self.inv_freq.as_ref().map_or(std::ptr::null(), |v| v.buffer.ptr.cast_const()), w.q.buffer.ptr, w.k.buffer.ptr, w.v.buffer.ptr, rows,
                        c.heads, c.kv_heads, c.head_dim, c.theta,
                        c.rotary.as_ref().map_or(1.0, |p| p.attention_factor), eps, false, s)?;
                }
                if let Some(context) = &self.paged {
                    l.block_dspark_attention_paged(w.q.buffer.ptr, w.k.buffer.ptr, w.v.buffer.ptr,
                        layer.k_ring.buffer.ptr, layer.v_ring.buffer.ptr, w.tables.buffer.ptr,
                        at(&w.tables, s_count * 4), at(&w.tables, 2 * s_count * 4), context.tables.buffer.ptr,
                        w.attn.buffer.ptr, w.attention_workspace.buffer.ptr, s_count, block, c.heads,
                        c.kv_heads, self.context_capacity, max_keys, c.window.unwrap_or(0), c.causal, c.head_dim, table_width,
                        1.0 / (c.head_dim as f32).sqrt(),
                        self.kv_cache == cuteafd_loader::families::qwen4::Qwen4KvCache::Fp8, s)?;
                } else {
                    l.block_dspark_attention(w.q.buffer.ptr, w.k.buffer.ptr, w.v.buffer.ptr, layer.k_ring.buffer.ptr,
                        layer.v_ring.buffer.ptr, w.tables.buffer.ptr, at(&w.tables, s_count * 4), at(&w.tables, 2 * s_count * 4),
                        w.attn.buffer.ptr, w.attention_workspace.buffer.ptr, s_count, block, c.heads, c.kv_heads, self.context_capacity,
                        max_keys, c.window.unwrap_or(0), c.causal, c.head_dim, 1.0 / (c.head_dim as f32).sqrt(), self.kv_cache == cuteafd_loader::families::qwen4::Qwen4KvCache::Fp8, s)?;
                }
                self.linear(w.attn.buffer.ptr, &layer.o, 0, w.delta.buffer.ptr, rows, c.q_width(), h)?;
                if let Some(native) = &self.native {
                    native.add_norm(l,w.h.buffer,w.delta.buffer,layer.post_norm.buffer,w.n.buffer.ptr,rows,s)?;
                } else {
                    l.glmf_dspark_add_rmsnorm(w.h.buffer.ptr, w.delta.buffer.ptr, layer.post_norm.buffer.ptr,
                        w.h.buffer.ptr, w.n.buffer.ptr, rows, h, eps, s)?;
                }
                self.linear(w.n.buffer.ptr, &layer.gate_up, 0, w.gate_up.buffer.ptr, rows, h, 2 * inter)?;
                if let Some(native) = &self.native {
                    l.copy_d2d_2d_async(native.gate.buffer,inter * 2,w.gate_up.buffer,inter * 4,inter * 2,rows,s)?;
                    let up = CuteafdDeviceBuffer { ptr:at(&w.gate_up,inter * 2),
                        bytes:w.gate_up.buffer.bytes - inter * 2,..w.gate_up.buffer };
                    l.copy_d2d_2d_async(native.up.buffer,inter * 2,up,inter * 4,inter * 2,rows,s)?;
                    native.activation.oai(cuteafd_ffi::minimax_m3::producers::OaiBatch { rows,
                        gate:native.gate.buffer,up:native.up.buffer,output:w.act.buffer },s)?;
                } else { l.glm_dflash_silu_mul(w.gate_up.buffer.ptr, w.act.buffer.ptr, rows, inter, s)?; }
                self.linear(w.act.buffer.ptr, &layer.down, 0, w.delta.buffer.ptr, rows, inter, h)?;
                let next = self.layers.get(index + 1).map_or(self.norm.buffer, |n| n.input_norm.buffer);
                if let Some(native) = &self.native {
                    native.add_norm(l,w.h.buffer,w.delta.buffer,next,w.n.buffer.ptr,rows,s)?;
                } else {
                    l.glmf_dspark_add_rmsnorm(w.h.buffer.ptr, w.delta.buffer.ptr, next.ptr, w.h.buffer.ptr, w.n.buffer.ptr,
                        rows, h, eps, s)?;
                }
            }
            }
            match head {
                HeadRef::Bf16(weight) if serving => crate::shared::draft::launch_bf16_head(w.output()?.head.as_ref().context("BF16 dSpark head was not prepared before serving")?,
                    w.n.buffer.ptr.cast(), weight.cast(), w.output()?.logits.buffer.ptr.cast(), rows, s)?,
                HeadRef::Bf16(weight) => crate::shared::speculation::dflash::launch_draft_head(l, w.output()?.head.as_ref().context("BF16 dSpark head was not prepared before serving")?, w.n.buffer.ptr,
                    weight, w.output()?.logits.buffer.ptr.cast(), rows, h, c.vocab, s, self.head_mode.get())?,
                HeadRef::Launch(launch) => launch(w.n.buffer.ptr.cast_const(), w.output()?.logits.buffer.ptr.cast(), rows, s)?,
            }
            l.glmf_dspark_markov(w.output()?.logits.buffer.ptr, self.head.as_ref().context("block Markov head absent")?.w1.buffer.ptr, self.head.as_ref().context("block Markov head absent")?.w2.buffer.ptr,
                self.head.as_ref().context("block Markov head absent")?.norms.buffer.ptr, w.anchors.buffer.ptr, w.output()?.tokens.buffer.ptr, w.output()?.markov.buffer.ptr, s_count, block, c.vocab,
                c.rank, s)?;
            l.glmf_dspark_confidence(w.n.buffer.ptr, self.head.as_ref().context("block Markov head absent")?.w1.buffer.ptr, self.head.as_ref().context("block confidence head absent")?.confidence_w.buffer.ptr,
                self.head.as_ref().context("block confidence head absent")?.confidence_b.buffer.ptr, w.anchors.buffer.ptr, w.output()?.tokens.buffer.ptr, w.output()?.confidence.buffer.ptr,
                s_count, block, h, c.rank, s)?;
            // The round's results download on its stream; one host wait then
            // covers the round and both copies (no second, synchronous drain).
            // SAFETY: the device buffers hold `rows` words; the landing is
            // pinned, owned by the workspace and read only after the wait.
            let output=w.output()?;
            let landing = |offset: usize| cuteafd_ffi::CuteafdHostBuffer {
                ptr: output.landing.buffer.ptr.cast::<u8>().add(offset).cast(), bytes: rows * 4, ..output.landing.buffer };
            l.copy_d2h_host_buffer_async(landing(0), CuteafdDeviceBuffer { bytes: rows * 4, ..w.output()?.tokens.buffer },
                rows * 4, s)?;
            l.copy_d2h_host_buffer_async(landing(rows * 4), CuteafdDeviceBuffer { bytes: rows * 4, ..w.output()?.confidence.buffer },
                rows * 4, s)?;
            self.round.borrow_mut().wait(s)?;
        }
        let (tokens, confidence) = w.output()?.landing.bytes()[..rows * 8].split_at(rows * 4);
        let word = |b: &[u8], i: usize| u32::from_le_bytes(b[i * 4..i * 4 + 4].try_into().unwrap());
        Ok((0..s_count).map(|i| Draft {
            tokens: (0..block).map(|j| word(&tokens, i * block + j)).collect(),
            features: vec![[0.0; 4]; block],
            confidence: (0..block).map(|j| f32::from_bits(word(&confidence, i * block + j))).collect(),
        }).collect())
    }

    pub(crate) fn stream_raw(&self) -> *mut c_void {
        self.stream
    }

    /// The last draft step's final-norm rows [sequences * block, hidden] BF16.
    pub fn last_hidden(&self, sequences: usize) -> Result<Vec<u8>> {
        let slot = self.workspace.borrow();
        let w = slot.as_ref().context("no draft step ran")?;
        let bytes = sequences * self.cfg.block * self.cfg.hidden * 2;
        let mut out = vec![0u8; bytes];
        // SAFETY: `self.stream` produced `w.n` and remains live on this device.
        unsafe { self.library.copy_d2h(&mut out, CuteafdDeviceBuffer { bytes, ..w.n.buffer }, self.stream)?; }
        Ok(out)
    }

    #[cfg(test)]
    pub(crate) fn last_logits(&self, sequences: usize) -> Result<Vec<f32>> {
        let slot = self.workspace.borrow();
        let w = slot.as_ref().context("no draft step ran")?;
        let bytes = sequences.checked_mul(self.cfg.block).and_then(|n| n.checked_mul(self.cfg.vocab))
            .and_then(|n| n.checked_mul(4)).context("draft logit readback overflow")?;
        ensure!(sequences > 0 && bytes <= w.output()?.logits.buffer.bytes, "draft logit readback exceeds arena");
        let mut host = vec![0; bytes];
        self.library.copy_d2h_quiescent(&mut host, CuteafdDeviceBuffer { bytes, ..w.output()?.logits.buffer })?;
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

    fn context_window(&self) -> Option<usize> { self.cfg.window }

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
    fn block_tp2_native_rank_arenas_match_physical_owner_receipts()->Result<()> {
        use crate::shared::memory::{device::{Device,Stream},weight_arena::DeviceWeightArena};
        use cuteafd_loader::{formats::block_dspark::{BlockDsparkConfig,BlockDsparkRuntimeLayout},
            placement::planner::{CoordinatorDevices,Location}};
        let fixture=cuteafd_ffi::native_library_lifetime_fixture::Fixture::build_for_tests()?;
        let library=fixture.load()?;
        for rank in 0..2 { fixture.configure_memory(&library,rank,1<<30)?; }
        fixture.immediate_copies(&library)?; fixture.quiet_events(&library,true)?;
        fixture.peer_ordering_model(&library)?;
        let root=tempfile::tempdir_in(fixture.directory())?;
        let config=serde_json::json!({"architectures":["DSparkDraftModel"],"speculators_model_type":"dspark",
            "aux_hidden_state_layer_ids":[1],"block_size":2,"confidence_head_with_markov":true,
            "draft_vocab_size":16,"enable_confidence_head":true,"markov_head_type":"vanilla","markov_rank":512,
            "mask_token_id":0,"sample_from_anchor":true,"sliding_window_non_causal":false,
            "transformer_layer_config":{"attention_bias":false,"head_dim":64,"hidden_act":"silu",
                "hidden_size":256,"intermediate_size":256,"layer_types":["sliding_attention"],
                "max_position_embeddings":8192,"model_type":"qwen3","num_attention_heads":4,"num_hidden_layers":1,
                "num_key_value_heads":2,"rms_norm_eps":1e-5,"rope_parameters":{"rope_theta":10000.,"rope_type":"default"},
                "sliding_window":8,"vocab_size":16}});
        std::fs::write(root.path().join("config.json"),serde_json::to_vec(&config)?)?;
        let cfg=BlockDsparkConfig::read(root.path())?;
        let tensors=RefCell::new(Vec::new());
        cuteafd_loader::formats::block_dspark::block_dspark_weight_recipe(&cfg,|name,shape| {
            tensors.borrow_mut().push(cuteafd_loader::plan::testing::t(name,"BF16",shape)); Ok::<_,anyhow::Error>(())
        },|parts,columns| { for (name,rows) in parts { tensors.borrow_mut().push(
            cuteafd_loader::plan::testing::t(*name,"BF16",&[*rows,columns])); } Ok(()) })?;
        cuteafd_loader::plan::testing::write_safetensors(&root.path().join("model.safetensors"),&tensors.into_inner());
        let entries=[(256,256),(256,128),(128,256)].into_iter().map(|(k,n)|serde_json::json!({
            "name":format!("draft_bf16_k{k}_n{n}"),"pointers":[{"name":"x"},{"name":"w"},{"name":"out"}],
            "scalars":[{"type":"int32"}],"capacity_rows":TAP_ROWS,"scratch_bytes_at_capacity":{},
            "geometry":{"width":k,"outputs":n,"output_dtype":"torch.bfloat16","reduction":"ordered_k16",
                "tiles_by_rows":[[16,1,4,16,64,6]]}})).collect::<Vec<_>>();
        fixture.register_programs(&library,&entries)?;
        let manifest=root.path().join("PROGRAMS.json");
        std::fs::write(&manifest,serde_json::to_vec(&serde_json::json!({"programs":entries}))?)?;
        let programs=library.programs()?.with_manifest(&manifest)?;
        let devices=[0,1].map(|id|Device { library:&library,id });
        for lead in [0,1] { for kv in [cuteafd_loader::families::qwen4::Qwen4KvCache::Bf16,
            cuteafd_loader::families::qwen4::Qwen4KvCache::Fp8] {
            let layouts=[0,1].map(|rank|BlockDsparkRuntimeLayout::for_rank(root.path(),2,2,188,
                &format!("test/block/{rank}"),kv,rank,2,lead)).into_iter().collect::<Result<Vec<_>>>()?;
            let arenas=layouts.iter().enumerate().map(|(rank,l)|DeviceWeightArena::new(devices[rank],l.weights.clone(),
                crate::shared::memory::TEST_ARENA)).collect::<Result<Vec<_>>>()?;
            let stream=Stream::new(devices[lead])?;
            let mut draft=devices[lead].own(||DsparkDrafter::load_tp2_with(&library,&programs,
                super::super::dflash::TargetProjection::Native,root.path(),2,2,kv,GlmDraftRepresentation::Bf16Only,[0,1],stream.raw,lead,
                |physical,name,bytes|arenas[physical as usize].view(name,bytes)))?;
            let split=draft.head_split.as_ref().context("missing paired block owner")?;
            assert!(draft.head.is_some() && split.peer.head.is_none());
            assert!(draft.workspace.borrow().as_ref().unwrap().output.is_some());
            assert!(split.peer.workspace.borrow().as_ref().unwrap().output.is_none());
            assert_eq!((draft.cfg.heads,draft.cfg.kv_heads,draft.cfg.intermediate),(2,1,128));
            let receipts=draft.runtime_receipts(CoordinatorDevices::new(&[0,1])?,"draft")?;
            for rank in 0..2 {
                let roots=receipts.iter().filter(|r|r.allocation.location==Location::coordinator(rank as u8)
                    && r.allocation.view.is_none() && r.allocation.group=="weight arena").collect::<Vec<_>>();
                assert_eq!(roots.len(),1,"rank {rank} has duplicate/missing arena");
                assert_eq!(roots[0].allocation.bytes,layouts[rank].weights.bytes() as u64);
                let reduction=receipts.iter().filter(|r|r.allocation.location==Location::coordinator(rank as u8)
                    && r.allocation.view.is_none() && r.allocation.id.ends_with("tp2.reduced")).collect::<Vec<_>>();
                assert_eq!(reduction.len(),1); assert_eq!(reduction[0].allocation.bytes,layouts[rank].reduction_bytes as u64);
                let exchange=receipts.iter().filter(|r|r.allocation.location==Location::coordinator(rank as u8)
                    && r.allocation.view.is_none() && r.allocation.id.contains("tp2.exchange.")).collect::<Vec<_>>();
                assert_eq!(exchange.len(),layouts[rank].exchange_buffers.len());
                assert!(exchange.iter().all(|r|r.allocation.category==cuteafd_core::memory_layout::Category::Transport));
                assert_eq!(exchange.iter().map(|r|r.allocation.bytes).sum::<u64>(),
                    layouts[rank].exchange_buffers.iter().map(|(_,n)|*n as u64).sum::<u64>());
                let actual=receipts.iter().filter(|r|r.allocation.location==Location::coordinator(rank as u8)
                    && r.allocation.view.is_none()).map(|r|r.allocation.bytes).sum::<u64>();
                let planned=layouts[rank].weights.bytes() as u64+layouts[rank].demands(rank as u8).iter().map(|d|d.bytes).sum::<u64>();
                assert_eq!(actual,planned,"rank {rank}, lead {lead}, KV {kv}: every physical backing must be charged once");
            }
            let producers=[Stream::new(devices[0])?,Stream::new(devices[1])?];
            let ranks=[0,1].map(|rank|crate::shared::peer_split::RankDevice { device:rank as i32,stream:producers[rank].raw });
            draft.bind_cooperative_producers(&ranks)?;
            draft.order_cooperative_producers()?;
            let split=draft.head_split.as_ref().unwrap();
            let (producer,event)=&draft.cooperative_producers[1-lead];
            assert!(split.exchange.require_cooperative_producer_event(1-lead,*producer,event.device.id,std::ptr::null_mut()).is_err());
            assert!(split.exchange.require_cooperative_producer_event(1-lead,*producer,1-event.device.id,event.raw).is_err());
            let invalid=ContextRow { tap_row:TAP_ROWS,slot:0,position:0 };
            fixture.quiet_events(&library,false)?;
            let before_invalid=fixture.events()?;
            let error=draft.update(&[invalid]).unwrap_err();
            let after_invalid=fixture.events()?;
            fixture.quiet_events(&library,true)?;
            assert_eq!(after_invalid.bytes().filter(|&event|event==b'p').count(),
                before_invalid.bytes().filter(|&event|event==b'p').count(),
                "invalid update queued a peer push before validation");
            assert!(format!("{error:#}").contains("past"));
            assert_eq!(split.exchange.terminal_state(),crate::shared::peer_split::TerminalState::Aborting);
            // Same sequence as Engine::shutdown: abort again, ordered teardown,
            // terminal completion, then repeated shutdown must remain harmless.
            let steps=RefCell::new(Vec::new());
            let shutdown=||->Result<()> {
                draft.abort()?;
                crate::shared::experts::rtx::pair::teardown(&mut crate::shared::experts::rtx::pair::Teardown {
                    abort:||{ steps.borrow_mut().push("abort"); draft.abort() },
                    ranks:||{ steps.borrow_mut().push("ranks"); draft.drain() },
                    links:||{ steps.borrow_mut().push("links"); Ok(()) },
                    copies:||{ steps.borrow_mut().push("copies"); draft.drain() },
                    graphs:||{ steps.borrow_mut().push("graphs"); Ok(()) },
                    finish:|drained|{ steps.borrow_mut().push("finish"); draft.finish_terminal(drained); },
                })
            };
            shutdown()?;
            assert_eq!(steps.borrow().as_slice(),["abort","ranks","links","copies","graphs","finish"]);
            assert_eq!(split.exchange.terminal_state(),crate::shared::peer_split::TerminalState::Drained);
            draft.abort()?;
            shutdown()?;
            draft.drain()?;
        } }
        Ok(())
    }

    #[test]
    fn admitted_rank_local_windows_restore_all_bytes_without_cross_device_copies()->Result<()> {
        use device::{Allocation,Device,Stream};
        let fixture=cuteafd_ffi::native_library_lifetime_fixture::Fixture::build_for_tests()?;
        let library=fixture.load()?;
        for rank in 0..2 { fixture.configure_memory(&library,rank,1<<30)?; }
        fixture.immediate_copies(&library)?; fixture.quiet_events(&library,true)?;
        let devices=[0,1].map(|id|Device { library:&library,id });
        let streams=[Stream::new(devices[0])?,Stream::new(devices[1])?];
        let planes=[Allocation::new(devices[0],256,TEST)?,Allocation::new(devices[1],256,TEST)?];
        let expected=[(0..256).map(|i|(i*17) as u8).collect::<Vec<_>>(),
            (0..256).map(|i|(i*31+7) as u8).collect::<Vec<_>>()];
        let mut snapshots=Vec::new();
        for rank in 0..2 {
            let (_,bytes)=WindowSnapshots::bytes(&[256],2)?;
            let backing=Rc::new(Allocation::new(devices[rank],bytes,Tag::prefix("test/admitted-window"))?);
            let before=fixture.events()?;
            let owner=WindowSnapshots::with_backing(crate::shared::prefix::Reader {
                device:rank as i32,stream:streams[rank].raw },backing.clone(),vec![256],2)?;
            assert!(Rc::ptr_eq(&owner.backing(),&backing));
            let added=fixture.events()?.strip_prefix(&before).unwrap().to_owned();
            assert!(!added.contains("device-alloc"),"admitted backing allocated twice");
            devices[rank].run(||library.copy_h2d(planes[rank].buffer,&expected[rank]))?;
            // SAFETY: rank-local live planes and backing outlive event-ordered copies.
            unsafe { owner.capture(0,&[planes[rank].buffer],0,4099)?; }
            owner.drain()?;
            devices[rank].run(||library.copy_h2d(planes[rank].buffer,&vec![199;256]))?;
            snapshots.push(owner);
        }
        assert_eq!(snapshots[0].metadata(0)?,snapshots[1].metadata(0)?);
        for rank in 0..2 {
            assert!(unsafe { snapshots[rank].restore(0,&[planes[1-rank].buffer],4099) }.is_err());
            // SAFETY: initialized mark and local owner satisfy retained event fence contract.
            unsafe { snapshots[rank].restore(0,&[planes[rank].buffer],4099)?; }
            snapshots[rank].drain()?;
            let mut actual=vec![0;256];
            devices[rank].run(||library.copy_d2h_quiescent(&mut actual,planes[rank].buffer))?;
            assert_eq!(actual,expected[rank],"rank bytes including inactive region changed");
        }
        let copies=include_str!("../prefix/copy_stream.rs");
        assert!(copies.find("cuda_event_record(event, reader)").unwrap()<copies.find("cuda_stream_wait_event(state.stream, event)").unwrap());
        assert!(copies.find("cuda_event_record(state.landed, state.stream)").unwrap()<copies.find("cuda_stream_wait_event(reader, state.landed)").unwrap());
        Ok(())
    }

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
            device.run(|| library.copy_d2h_quiescent(&mut bytes, *plane))?;
            assert_eq!(bytes, vec![197; plane.bytes], "invalid restore must not mutate destinations");
        }
        assert_eq!(unsafe { snapshots.restore(0, &live, 23)? }, (0, 23));
        snapshots.drain()?;
        for (plane, cold) in live.iter().zip(&expected) {
            let mut restored = vec![0; plane.bytes];
            device.run(|| library.copy_d2h_quiescent(&mut restored, *plane))?;
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
        for (window, rank, dim, yarn) in [(Some(8),256,128,false),(None,256,128,false),
            (Some(8),512,128,false),(None,512,128,false),(Some(8),256,64,true),(None,256,64,true)] {
            let root = tempfile::tempdir_in(fixture.directory())?;
            let snapshot = root.path().join("draft");
            std::fs::create_dir_all(&snapshot)?;
            let config = serde_json::json!({"architectures":["DSparkDraftModel"], "speculators_model_type":"dspark",
                "aux_hidden_state_layer_ids":[1], "block_size":2, "confidence_head_with_markov":true,
                "draft_vocab_size":16, "enable_confidence_head":true, "markov_head_type":"vanilla", "markov_rank":rank,
                "mask_token_id":0, "sample_from_anchor":true, "sliding_window_non_causal":false,
                "transformer_layer_config":{"attention_bias":false, "head_dim":dim, "hidden_act":"silu",
                    "hidden_size":256, "intermediate_size":256, "layer_types":[if window.is_some() { "sliding_attention" } else { "full_attention" }],
                    "max_position_embeddings":64, "model_type":"qwen3", "num_attention_heads":2, "num_hidden_layers":1,
                    "num_key_value_heads":2, "rms_norm_eps":1e-5, "rope_parameters":if yarn {
                        serde_json::json!({"rope_theta":10000.,"rope_type":"yarn","factor":16.,"original_max_position_embeddings":65536})
                    } else { serde_json::json!({"rope_theta":10000.0,"rope_type":"default"}) },
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
            let shapes = [(256,256),(256,cfg.qkv_width()),(cfg.q_width(),256),(256,2*cfg.kv_width()),(256,512)]
                .into_iter().collect::<BTreeSet<_>>();
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
                    device.library.cuda_set_device(0)?;
                    drafter.prepare_workspace()?;
                    assert_eq!(device.library.cuda_get_device()?, 0, "workspace preparation restores caller device");
                    {
                    let workspace = drafter.workspace.borrow();
                    let workspace = workspace.as_ref().context("prepared reversed-device workspace missing")?;
                    let output=workspace.output()?;
                    for owner in [&workspace.h, &workspace.n, &workspace.qkv, &workspace.q,
                        &workspace.k, &workspace.v, &workspace.attn, &workspace.delta, &workspace.gate_up,
                        &workspace.act, &output.logits, &workspace.anchors, &workspace.ids,
                        &output.tokens, &output.confidence, &workspace.positions, &workspace.tables,
                        &workspace.attention_workspace, &output.markov, &output._head_workspace] {
                        assert_eq!(owner.buffer.device_id, device.id, "workspace owner follows drafter, not caller");
                    }
                    }
                    drafter.put(&drafter.context_positions, &17i64.to_le_bytes())?;
                    assert_eq!(device.library.cuda_get_device()?, 0, "ordered put restores caller device");
                    let staging = device.library.ordered_h2d_staging_buffer(drafter.context_positions.buffer,
                        drafter.stream)?.context("ordered upload staging missing on retained device")?;
                    assert!(staging.bytes >= 8);
                    drafter.on_device(|| {
                        assert_eq!(device.library.cuda_get_device()?, device.id);
                        let event = device.library.cuda_event_create()?;
                        // SAFETY: this fixture retains the owner stream/buffer;
                        // explicit event and enqueue run on the scoped device.
                        unsafe {
                            device.library.cuda_zero_bytes_async(drafter.context_slots.buffer, 4, drafter.stream)?;
                            device.library.cuda_event_record(event, drafter.stream)?;
                            device.library.cuda_event_synchronize(event)?;
                            device.library.cuda_event_destroy(event)?;
                        }
                        Ok(())
                    })?;
                    assert_eq!(device.library.cuda_get_device()?, 0, "enqueue restores caller device");
                    drafter.reset_context(0, 0)?;
                    assert_eq!(device.library.cuda_get_device()?, 0, "reset restores caller device");
                    let source = include_str!("block_dspark.rs");
                    assert!(source.contains("self.on_device(|| self.update_on_device(rows))"));
                    assert!(source.contains("self.on_device(|| self.draft_on_device(sequences, input, head, serving))"));
                    Ok(drafter)
                }
                let baseline = run(device,&programs,&snapshot,representation,None,&expected)?;
                let digest = |drafter:&DsparkDrafter<'_>| -> Result<Vec<(String,Vec<u8>)>> {
                    let mut owners = drafter.weight_buffers();
                    if let Some(inv) = &drafter.inv_freq { owners.push(("inv_freq".into(),inv.buffer)); }
                    owners.into_iter().map(|(name,buffer)| {
                        let mut bytes=vec![0;buffer.bytes]; device.run(|| library.copy_d2h_quiescent(&mut bytes,buffer))?;
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

    #[test]
    #[ignore = "installed Lithos parity fixture; requires sealed CUDA programs and reference outputs"]
    fn installed_lithos_dspark_forward_parity_fixture() -> Result<()> {
        use crate::shared::memory::{device::{Device,Stream},weight_arena::DeviceWeightArena};
        use cuteafd_core::memory_layout::ArenaTag;
        use cuteafd_loader::families::qwen4::Qwen4KvCache;
        let Some(snapshot)=std::env::var_os("CUTEAFD_TEST_DRAFT_SNAPSHOT").map(std::path::PathBuf::from)
            .filter(|p|p.join("config.json").is_file()) else {
            eprintln!("SKIP installed Lithos parity fixture: checkpoint absent"); return Ok(());
        };
        let root=std::path::PathBuf::from(std::env::var("CUTEAFD_GOLDEN_FIXTURES")?);
        let golden=root.join("taps");
        let target=std::path::PathBuf::from(std::env::var("CUTEAFD_TEST_TARGET_SNAPSHOT")?);
        let metadata:serde_json::Value=serde_json::from_slice(&std::fs::read(root.join("full-bf16/meta.json"))?)?;
        let positions=metadata["positions"].as_array().context("fixture positions")?.iter()
            .map(|n|n.as_u64().map(|n|n as usize).context("fixture position")).collect::<Result<Vec<_>>>()?;
        let cfg=DsparkConfig::read(&snapshot)?;
        ensure!((cfg.hidden,cfg.heads,cfg.kv_heads,cfg.head_dim,cfg.block,cfg.rank)==(7168,64,16,64,5,256)
            && cfg.taps==[7,23,67,83,91] && cfg.mask_token==163824 && cfg.window.is_none(),"fixture is not native Lithos");
        let tokens=std::fs::read(golden.join("tokens.bin"))?.chunks_exact(4)
            .map(|b|u32::from_le_bytes(b.try_into().unwrap())).collect::<Vec<_>>();
        let context=tokens.len();
        let layers=cfg.taps.iter().map(|l|std::fs::read(golden.join(format!("layer{l:02}.bin"))))
            .collect::<std::io::Result<Vec<_>>>()?;
        // SAFETY: matching sealed library outlives every native owner in this isolated test.
        let library=unsafe { NativeLibrary::load(std::env::var("CUTEAFD_NATIVE_LIB")?)? };
        let manifest=std::path::PathBuf::from(std::env::var("CUTEAFD_PROGRAM_MANIFEST")?);
        let programs=library.programs()?.with_manifest(&manifest)?;
        let device=Device { library:&library,id:std::env::var("CUTEAFD_CACHE_TEST_DEVICE")?.parse()? };
        let stream=Stream::new(device)?;
        let source=super::super::weights::Source::open(&target)?;
        let index:serde_json::Value=serde_json::from_slice(&std::fs::read(target.join("model.safetensors.index.json"))?)?;
        let find=|suffix:&str|->Result<String> {
            let names=index["weight_map"].as_object().context("target index")?.keys()
                .filter(|name|name.ends_with(suffix)).cloned().collect::<Vec<_>>();
            ensure!(names.len()==1,"target operand {suffix} is ambiguous");Ok(names[0].clone())
        };
        let embedding=source.rect(&find("embed_tokens.weight")?,&[cfg.vocab,cfg.hidden])?;
        let head_name=find("lm_head.weight")?;
        let head_requests=vec![(head_name.clone(),cfg.vocab*cfg.hidden*2)];
        let head_arena=DeviceWeightArena::new(device,cuteafd_loader::serving_capacity::WeightArenaLayout::new(
            "test/lithos-head",head_requests)?,ArenaTag::drafter_weights("test/lithos-head"))?;
        let head=device.run(|| {
            let mut allocate=|physical,name:&str,bytes| { ensure!(physical==device.id,"fixture head owner");head_arena.view(name,bytes) };
            super::super::weights::Load::open(&library,&target)?.with_allocator(&mut allocate)
                .tensor(&head_name,&[cfg.vocab,cfg.hidden])
        })?;
        for window in [0,2048] { for kv in [Qwen4KvCache::Bf16,Qwen4KvCache::Fp8] {
            let name=format!("{}-{kv}",if window==0 { "full" } else { "window2048" });
            let reference=root.join(&name);
            let words=|file:&str|->Result<Vec<u32>> { Ok(std::fs::read(reference.join(file))?.chunks_exact(4)
                .map(|b|u32::from_le_bytes(b.try_into().unwrap())).collect()) };
            let expected=words("drafts.bin")?;
            let confidence=words("confidence.bin")?;
            let hidden=std::fs::read(reference.join("hidden.bin"))?;
            let requests=DsparkDrafter::arena_requests(&snapshot,GlmDraftRepresentation::Bf16Only,device.id)?
                .remove(&device.id).context("fixture arena requests")?;
            let arena=DeviceWeightArena::new(device,cuteafd_loader::serving_capacity::WeightArenaLayout::new(
                "test/lithos-draft",requests)?,ArenaTag::drafter_weights("test/lithos-draft"))?;
            let tensors=prefetch(&snapshot).join().map_err(|_|anyhow::anyhow!("fixture prefetch panicked"))??;
            let draft=device.run(||DsparkDrafter::load_window_paged_with(&library,&programs,
                super::super::dflash::TargetProjection::Bf16,&snapshot,tensors,stream.raw,
                1,1,vec![0;cfg.hidden*2],GlmDraftRepresentation::Bf16Only,fp8_linear::Fp8Scales::Amax,
                fp8_linear::Fp8Rows::W8a16,if window==0 {context} else {window},context+128,kv,window,
                |physical,name,bytes| { ensure!(physical==device.id,"fixture draft owner");arena.view(name,bytes) }))?;
            let (mut done,mut max_abs,mut max_relative,mut max_confidence)=(0,0f32,0f32,0f32);
            for (index,&position) in positions.iter().enumerate() {
                while done<position {
                    let n=(position-done).min(TAP_ROWS);
                    let row=cfg.hidden*2;
                    let mut taps=vec![0;n*cfg.taps.len()*row];
                    for r in 0..n { for (tap,layer) in layers.iter().enumerate() {
                        taps[(r*layers.len()+tap)*row..][..row].copy_from_slice(&layer[(done+r)*row..][..row]);
                    } }
                    device.run(|| { draft.put_taps(&taps)?;draft.update(&(0..n).map(|r|ContextRow {
                        tap_row:r,slot:0,position:done+r }).collect::<Vec<_>>())?;draft.drain() })?;
                    done+=n;
                }
                let input=|ids:&[u32],out:CuteafdDeviceBuffer,stream|->Result<()> {
                    for (row,&id) in ids.iter().enumerate() {
                        let mut rect=embedding.clone();rect.rows=id as usize..id as usize+1;
                        let mut bytes=vec![0;cfg.hidden*2];rect.read(&mut bytes)?;
                        let out=crate::shared::prefix::view(out,row*bytes.len(),bytes.len())?;
                        // SAFETY: ordered staging and destination survive this retained stream.
                        unsafe { library.copy_h2d_ordered(out,&bytes,stream)?; }
                    }
                    Ok(())
                };
                let actual=device.run(||draft.draft_device_with(&[DraftSeq {slot:0,anchor:tokens[position],position,valid_from:0}],
                    &input,super::super::dflash::TargetHead::Bf16(&head)))?.remove(0);
                ensure!(actual.tokens==expected[index*cfg.block..][..cfg.block],"{name} position {position} draft tokens differ");
                let confidence_argmax=|values:&[f32]|values.iter().enumerate()
                    .max_by(|(i,a),(j,b)|a.total_cmp(b).then_with(||j.cmp(i))).map(|(i,_)|i);
                let expected_confidence=confidence[index*cfg.block..][..cfg.block].iter()
                    .map(|&bits|f32::from_bits(bits)).collect::<Vec<_>>();
                ensure!(confidence_argmax(&actual.confidence)==confidence_argmax(&expected_confidence),
                    "{name} position {position} confidence argmax differs");
                for (j,&value) in actual.confidence.iter().enumerate() {
                    max_confidence=max_confidence.max((value-f32::from_bits(confidence[index*cfg.block+j])).abs());
                }
                let actual=device.run(||draft.last_hidden(1))?;
                let expected=&hidden[index*cfg.block*cfg.hidden*2..][..actual.len()];
                for (a,b) in actual.chunks_exact(2).zip(expected.chunks_exact(2)) {
                    let a=f32::from_bits((u16::from_le_bytes(a.try_into().unwrap()) as u32)<<16);
                    let b=f32::from_bits((u16::from_le_bytes(b.try_into().unwrap()) as u32)<<16);
                    max_abs=max_abs.max((a-b).abs());max_relative=max_relative.max((a-b).abs()/b.abs().max(1e-3));
                }
            }
            eprintln!("Lithos parity {name}: {} anchors, tokens exact, hidden max_abs={max_abs:.8} max_relative={max_relative:.8}, confidence max_abs={max_confidence:.8}",positions.len());
            ensure!(max_abs.is_finite()&&max_relative.is_finite()&&max_confidence<0.05,"{name} nonfinite or excessive confidence delta");
        } }
        Ok(())
    }

    #[test]
    #[ignore = "requires sealed native programs, MiniMax drafter weights and GPU1"]
    fn installed_minimax_window_ingest_and_marks_are_byte_exact() -> Result<()> {
        use crate::shared::memory::{device::{Allocation, Device, Stream}, weight_arena::DeviceWeightArena};
        use cuteafd_core::memory_layout::ArenaTag;
        // SAFETY: the matching sealed slot library is retained throughout this isolated oracle.
        let library = unsafe { NativeLibrary::load(std::env::var("CUTEAFD_NATIVE_LIB")?)? };
        let manifest = std::path::PathBuf::from(std::env::var("CUTEAFD_PROGRAM_MANIFEST")?);
        let programs = library.programs()?.with_manifest(&manifest)?;
        let snapshot = std::path::PathBuf::from(std::env::var("CUTEAFD_TEST_DRAFT_SNAPSHOT")?);
        let id = std::env::var("CUTEAFD_CACHE_TEST_DEVICE").context("oracle harness visible device missing")?.parse()?;
        let device = Device { library: &library, id };
        let stream = Stream::new(device)?;
        let positions = 65536 + 4096;
        let rope = Allocation::new(device, positions * 256, TEST)?;
        device.run(|| library.copy_h2d(rope.buffer,
            &crate::families::minimax_m3::cache::rotary_bytes(positions, 1e7)))?;
        let cfg = DsparkConfig::read(&snapshot)?;
        let window = cfg.window.context("oracle requires window")?;
        let requests = DsparkDrafter::arena_requests(&snapshot, GlmDraftRepresentation::Bf16Only, id)?
            .remove(&id).context("oracle weight requests")?;
        let arena = DeviceWeightArena::new(device, cuteafd_loader::serving_capacity::WeightArenaLayout::new(
            "window-oracle", requests)?, ArenaTag::drafter_weights("window-oracle"))?;
        let drafter = device.run(|| DsparkDrafter::load_native_with(&library, &programs,
            super::super::dflash::TargetProjection::Bf16, &snapshot, stream.raw,
            1, 1, vec![0; cfg.hidden * 2], rope.buffer, positions,
            |id, name, bytes| { ensure!(id == device.id, "oracle device"); arena.view(name, bytes) }))?;
        let planes = drafter.window_planes(0)?;
        let marks = WindowSnapshots::new(&stream, planes.iter().map(|plane| plane.bytes).collect(), 2)?;
        let pitch = cfg.taps.len() * cfg.hidden * 2;
        let read = || -> Result<Vec<Vec<u8>>> {
            stream.device.run(|| {
                // SAFETY: all producers use this owned stream and planes outlive the readback.
                unsafe { library.cuda_stream_synchronize(stream.raw)?; }
                planes.iter().map(|&plane| {
                    let mut bytes = vec![0; plane.bytes];
                    library.copy_d2h_quiescent(&mut bytes, plane)?;
                    Ok(bytes)
                }).collect()
            })
        };
        let ingest = |start: usize, count: usize| -> Result<()> {
            let taps = (0..count).flat_map(|row| (0..pitch / 2).flat_map(move |column| {
                let value = (((start + row) * 37 + column * 11) % 251) as f32 / 128. - 1.;
                ((value.to_bits() >> 16) as u16).to_le_bytes()
            })).collect::<Vec<_>>();
            drafter.put_taps(&taps)?;
            drafter.update(&(0..count).map(|tap_row| ContextRow {
                tap_row, slot: 0, position: start + tap_row }).collect::<Vec<_>>())
        };
        for end in [17, window, window + 1, 8192, 65536] {
            // 128 is serving's partition; 257 exercises a straddling boundary.
            for chunk in [128, 257] {
                let mut expected = None;
                for skip in [false, true] {
                    drafter.reset_context(0, 0)?;
                    for start in (0..end).step_by(chunk) {
                        let count = chunk.min(end - start);
                        if !skip || super::super::retain_context_chunk(cfg.window, start..start + count, end) {
                            ingest(start, count)?;
                        }
                    }
                    let actual = read()?;
                    if let Some(expected) = &expected { ensure!(&actual == expected,
                        "window ring bytes differ end={end} chunk={chunk}"); }
                    else { expected = Some(actual); }
                    // SAFETY: admitted disjoint rings survive both copy streams and restoration.
                    unsafe { marks.capture(usize::from(skip), &planes, 0, end)?; }
                    marks.drain()?;
                }
                let mut expected_mark = vec![0; marks.mark_bytes()];
                let mut actual_mark = expected_mark.clone();
                device.run(|| {
                    library.copy_d2h_quiescent(&mut expected_mark, marks.mark(0)?)?;
                    library.copy_d2h_quiescent(&mut actual_mark, marks.mark(1)?)
                })?;
                // Padding isn't state; compare origin/frontier and every physical plane.
                let payload = 16 + planes.iter().map(|plane| plane.bytes).sum::<usize>();
                ensure!(expected_mark[..payload] == actual_mark[..payload], "prompt snapshot bytes differ");
                ingest(end, 17)?;
                let expected_turn = read()?;
                // SAFETY: initialized prompt mark and retained planes satisfy snapshot ownership.
                ensure!(unsafe { marks.restore(1, &planes, end)? } == (0, end), "restored frontier differs");
                marks.drain()?;
                ensure!(read()? == expected.as_ref().unwrap().clone(), "restore differs from cold prefix");
                ingest(end, 17)?;
                ensure!(read()? == expected_turn, "turn continuation differs");
                eprintln!("window ingest byte-exact: prompt={end} chunk={chunk} planes={} snapshot={payload}", planes.len());
            }
        }
        Ok(())
    }

    fn geometry(window: Option<usize>) -> DsparkConfig {
        DsparkConfig { backbone: cuteafd_loader::formats::block_dspark::DsparkBackbone {
                norm: cuteafd_loader::formats::block_dspark::DsparkNorm::Rms,
                residual: cuteafd_loader::formats::block_dspark::DsparkResidual::Bf16, rotary_dim: 128,
                activation: cuteafd_core::FfnActivation::SiluBf16Product { limit: None } },
            dialect: cuteafd_loader::formats::block_dspark::DsparkDialect::Qwen3,
            hidden: 5120, intermediate: 10240, layers: 5, heads: 40, kv_heads: 8,
            head_dim: 128, eps: 1e-6, theta: 1e7, block: 7, mask_token: 248077, rank: 256,
            taps: vec![4,16,28,40,52], vocab: 248320, causal: false, window, rotary: None }
    }

    #[test]
    fn draft_target_taps_validate_order_stride_stream_and_span() {
        use super::super::target::HiddenTap;
        let marker = 0u8;
        let pointer = std::ptr::from_ref(&marker).cast();
        let tap = HiddenTap { site: cuteafd_core::TapSite::LayerExit(4), ptr: pointer, rows: 7, hidden: 5120, stride: 5128,
            stream: std::ptr::null_mut() };
        let taps = [tap, HiddenTap { site: cuteafd_core::TapSite::LayerExit(16), ..tap }];
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
