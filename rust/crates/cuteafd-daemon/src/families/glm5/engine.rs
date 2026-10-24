//! GLM 5.x (glm_moe_dsa) coordinator over the exported glm_* programs.
//!
//! One layer: input norm (fused with the previous layer's residual add),
//! MLA producer (writes the FP8 656-byte latent record), DSA index producer
//! and top-k on full-indexer layers (shared layers reuse the last full
//! layer's selection), sparse MLA, W_UV + o_proj, post-attention norm, then
//! the dense MLP or the MoE (router, expert input quantization, shared
//! expert, routed experts). The latent and index caches share page ids.
//!
//! A long Spark prefill chunk runs as two lanes of consecutive rows, each with
//! its own workspace and transport ([`GlmEngine::prefill`]): one lane's Spark
//! wave stays in flight while the other lane's GPU layers run.
use super::weights::{GlmLayer, GlmWeights};
use crate::shared::launch_grid::Fp8QuantizeGrid;
use crate::shared::memory::{DeviceAllocation, HostAllocation};
use crate::shared::token_io::{DeviceLogits, TokenEmbedding};
use crate::shared::spark_intake::{copy_parallel, IntakeMode, SparkIntake, SparkLink};
use cuteafd_transport::expert::{
    SparkExpertWave, EXPERT_PROTOCOL_V2_FLAG_V41_COMPACT_BF16,
};
use cuteafd_transport::{
    ExpertProtocolV2Request, ExpertProtocolV2RowDescriptor, ExpertV2Dtype, ExpertV2SourceKind,
};
use anyhow::{ensure, Context, Result};
use cuteafd_ffi::programs::{Programs, Scalar, VocabularyHead, VOCABULARY_HEAD_WORKSPACE};
use cuteafd_ffi::NativeLibrary;
use cuteafd_loader::families::glm5::GlmDsaConfig;
use crate::shared::peer_split::{PeerExchange, RankDevice, DIRECT};
use std::cell::{Cell, RefCell};
use cuteafd_loader::serving_capacity::{glm_decode_row_bucket,
    glm_decode_table_width, glm_decode_table_widths, GLM_SHORT_ROW_BUCKETS};
use std::ffi::c_void;
use crate::shared::experts::rtx::{pair::Pair, ExpertInput, Routes, PartialDtype, ExchangeDtype, RtxExpertLayer};

#[path = "engine/context.rs"]
mod context;
mod tp2;
mod layers;
#[path = "engine/native_dense.rs"]
pub(crate) mod native_dense;
pub(crate) mod exlr8;
pub(crate) use cuteafd_loader::placement::families::glm5::GlmContext;

type Dev<'a> = DeviceAllocation<'a>;

pub(crate) const PAGE_ROWS: usize = 64;
/// Decode/verify steps run padded to one of these row counts, and their page
/// tables to a power-of-two width of at least 16 pages (or the
/// whole context): a bounded set of decode graph shapes, all captured at
/// startup ([`GlmEngine::warm_decode_graphs`]), so serving never captures.
/// Exact up to 16 rows (one sequence's verify step: no padding at C1), then
/// in steps of 4 / 8 (padding a batched step costs ~1% per row on the GPU).
pub(crate) const ROW_BUCKETS: [usize; 24] = GLM_SHORT_ROW_BUCKETS;

/// Statuses follow the ids in a decode workspace's `select` buffer at a fixed offset.
const SELECT_STATUS_OFFSET: usize = DECODE_ROWS * 4;
pub(crate) const RECORD_BYTES: usize = 656;
const RECORD_PAGE_BYTES: usize = PAGE_ROWS * RECORD_BYTES;
pub(crate) const INDEX_PAGE_BYTES: usize = 8448;
/// Most Spark ranks a step's partials come from (the compact reducer's limit).
const MAX_RANKS: usize = crate::shared::spark_intake::MAX_INTAKE_RANKS;
/// Ranks whose (zero) partials a skipped exchange uploads: the TP4 layout.
const SKIP_RANKS: usize = cuteafd_loader::placement::families::glm5::SKIP_RANKS;
/// Rows of the decode-route programs (`_m64`).
pub(crate) const DECODE_ROWS: usize = 64;
/// Most lanes a long Spark prefill chunk splits into (serve `--prefill-waves`,
/// default 2; 1 is the serial path), and the fewest rows per
/// lane worth another exchange per layer.
pub(crate) const PREFILL_LANES: usize = 4;
const MIN_LANE_ROWS: usize = 256;

/// What precedes a decode segment's residual norm.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum Previous {
    First,
    Delta,
    Planes(usize),
}

/// Everything a captured decode segment bakes in.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct GraphKey {
    layer: usize,
    rows: usize,
    table_width: usize,
    table_stride: usize,
    previous: Previous,
    /// The last segment ends in the head and the greedy selection.
    head: bool,
}

struct GraphExec<'a>(*mut c_void, &'a NativeLibrary);

impl Drop for GraphExec<'_> {
    fn drop(&mut self) {
        // SAFETY: the exec came from end_capture and is destroyed once.
        let _ = unsafe { self.1.cuda_graph_exec_destroy(self.0) };
    }
}

/// Host tables of one step.
#[derive(Clone)]
struct StepTables {
    decode: bool,
    positions: Vec<i64>,
    slots: Vec<i64>,
    /// Prefill: the sequence's pages; decode: one padded row per step row.
    page_table: Vec<i32>,
    table_width: usize,
    table_stride: usize,
    cache_lengths: Vec<i32>,
    /// Selected entries per row the sparse MLA reads (every earlier token
    /// below the index top-k; the selection leads each indices row).
    lengths: Vec<i32>,
    /// Leading rows that are real (the rest pad a decode step to its row
    /// bucket and write only the scratch page): the Spark exchange sends these.
    exchange_rows: usize,
}

impl StepTables {
    fn pad_rows(&mut self, pool_pages: usize, bucket: usize) {
        let scratch = pool_pages as i32;
        while self.positions.len() < bucket {
            self.positions.push(0);
            self.slots.push(i64::from(scratch) * PAGE_ROWS as i64);
            self.cache_lengths.push(1);
            self.lengths.push(1);
            self.page_table.extend(std::iter::repeat_n(scratch, self.table_width));
        }
    }
}

/// A prefill step's sequence pages (its page table); empty for a decode step.
fn placement_pages(tables: &StepTables) -> Vec<u32> {
    if tables.decode { Vec::new() } else { tables.page_table.iter().map(|&p| p as u32).collect() }
}

/// A sequence's pages (shared by the latent and index caches) and length.
#[derive(Debug, Clone)]
pub(crate) struct GlmPlacement {
    pub pages: Vec<u32>,
    pub len: usize,
}

impl GlmPlacement {
    pub fn slot(&self, position: usize) -> Result<i64> {
        let page = *self.pages.get(position / PAGE_ROWS).context("position past the sequence's pages")?;
        Ok(i64::from(page) * PAGE_ROWS as i64 + (position % PAGE_ROWS) as i64)
    }

    /// The first `count` pages as a page-table row.
    fn table(&self, count: usize) -> impl Iterator<Item = i32> + '_ {
        self.pages[..count].iter().map(|&p| p as i32)
    }
}

/// Refcounted pages of the shared latent/index cache pool (goldens and benches; serving takes
/// its pages from the prefix cache's pool).
pub(crate) struct PageAllocator {
    pool: cuteafd_engine::prefix::RefPagePool,
}

impl PageAllocator {
    /// `engine`'s pool: parity half pools under context (page `j` on GPU `j mod 2`).
    pub fn new(engine: &GlmEngine<'_>) -> Self {
        Self { pool: if engine.context_kv() { cuteafd_engine::prefix::RefPagePool::parity(engine.pages, PAGE_ROWS, 0) }
            else { cuteafd_engine::prefix::RefPagePool::new(engine.pages, PAGE_ROWS) } }
    }

    /// Reserves every page a sequence of up to `capacity` tokens needs.
    pub fn admit(&mut self, capacity: usize) -> Result<GlmPlacement> {
        let pages = self.pool.alloc(self.pool.pages_for(capacity)).context("cache pages exhausted")?;
        Ok(GlmPlacement { pages, len: 0 })
    }

    /// A second sequence starting as `source`'s first `len` rows: full pages shared, the
    /// partial tail page copied by the caller (the returned copy).
    pub fn fork(&mut self, source: &GlmPlacement, len: usize, capacity: usize)
        -> Result<(GlmPlacement, Option<cuteafd_engine::prefix::TailCopy>)> {
        let fork = self.pool.fork(&source.pages, len, self.pool.pages_for(capacity)).context("cache pages exhausted")?;
        Ok((GlmPlacement { pages: fork.pages, len: 0 }, fork.copy))
    }

    pub fn release(&mut self, placement: GlmPlacement) {
        self.pool.release(&placement.pages);
    }
}

struct Workspace<'a> {
    rows: usize,
    h: Dev<'a>,
    x: Dev<'a>,
    query: Dev<'a>,
    q_resid: Dev<'a>,
    q_fp8: Dev<'a>,
    head_weights: Dev<'a>,
    indices: Dev<'a>,
    lengths: Dev<'a>,
    attn: Dev<'a>,
    delta: Dev<'a>,
    positions: Dev<'a>,
    slots: Dev<'a>,
    page_table: Dev<'a>,
    cache_lengths: Dev<'a>,
    scratch: Dev<'a>,
    topk_scratch: Dev<'a>,
    logits: Dev<'a>,
    /// MoE: router logits (FP32), routes, wire rows, shared-expert output,
    /// rank partial planes, and their pinned staging.
    router_logits: Dev<'a>,
    route_ids: Dev<'a>,
    route_weights: Dev<'a>,
    wire: Dev<'a>,
    shared: Dev<'a>,
    /// Decode and each prefill lane own an arena; weights remain shared per GPU.
    native_shared: Option<crate::shared::experts::exlr8::Exlr8Execution<'a>>,
    /// Native dense scratch (EXLR8 checkpoint dense operands), when the layers run natively.
    native_dense: Option<native_dense::NativeScratch<'a>>,
    payload: Option<Dev<'a>>,
    router_host: RefCell<HostAllocation<'a>>,
    /// The step's token ids (U32, gathered from the device embedding table).
    ids: Dev<'a>,
    /// Greedy selection of the logits rows inside the decode graph: U32 ids, then U32 statuses.
    select: Dev<'a>,
    /// The cuBLAS LM head (rank 0 only).
    head: Option<VocabularyHead<'a>>,
    _head_workspace: Dev<'a>,
}

/// The second GPU of a two-GPU head split (rank 1): its share of every layer
/// (half the heads' q_b, kv_b and o_proj; half the dense and shared-expert
/// intermediate), its copy of the latent and index caches (the replicated
/// latent projection and DSA indexer write identical records on both GPUs),
/// RoPE table, workspaces and captured decode segments.
pub(crate) struct GlmPeer<'a> {
    pub device: i32,
    pub stream: *mut c_void,
    pub layers: Vec<GlmLayer<'a>>,
    kv: Vec<Dev<'a>>,
    index: Vec<Option<Dev<'a>>>,
    cos_sin: Dev<'a>,
    decode_workspace: RefCell<Option<Workspace<'a>>>,
    lane_workspaces: RefCell<Vec<Workspace<'a>>>,
    graphs: RefCell<std::collections::HashMap<GraphKey, GraphExec<'a>>>,
    /// L2 prefetch of its next layer's weights while rank 0 waits in a decode exchange.
    l2: Option<crate::shared::l2_prefetch::L2Prefetch>,
}

/// Exchange slot of layer `index`, lane `lane`: its attention partials (`ffn`
/// false) or its FFN exchange (dense partials, the shared-expert partial from
/// rank 1, the routed + shared sum from rank 0), by layer parity.
fn slot(index: usize, ffn: bool, lane: usize) -> usize {
    4 * lane + 2 * (index % 2) + usize::from(ffn)
}

fn slot_of(index: usize, ffn: bool, lane: usize) -> usize { slot(index, ffn, lane) }

/// The context prefill plan an attention uses: a unit-driven call selects its
/// unit explicitly (set before either rank's attention for that unit); a
/// single-pass step (`usize::MAX`, set by `step_tables`) uses its lane's plan.
fn context_plan_index(unit: usize, lane: usize) -> usize {
    if unit == usize::MAX { lane } else { unit }
}

/// Borrowed workspaces of the head split's second GPU.
enum PeerWorkspaces<'e, 'a> {
    Decode(std::cell::Ref<'e, Option<Workspace<'a>>>),
    Lanes(std::cell::Ref<'e, Vec<Workspace<'a>>>),
}

impl<'a> PeerWorkspaces<'_, 'a> {
    fn get(&self, lane: usize) -> Result<&Workspace<'a>> {
        match self {
            Self::Decode(w) => w.as_ref().context("peer decode workspace"),
            Self::Lanes(w) => w.get(lane).context("peer lane workspace"),
        }
    }
}

/// A step's logits: on the device (one workspace's rows), or downloaded
/// (rows spanning several prefill lanes).
enum StepLogits {
    Device(DeviceLogits),
    Host(Vec<f32>),
}

pub(crate) struct GlmEngine<'a> {
    pub(crate) spark_bf16: bool,
    /// Per rank: bound native dense programs and the prefill unpack staging
    /// (EXLR8 checkpoint dense operands); empty on the official path.
    pub(crate) native_dense: Vec<Option<native_dense::NativePrograms<'a>>>,
    pub(crate) native_experts: Vec<exlr8::NativeExperts<'a>>,
    quantize_grid: Fp8QuantizeGrid,
    pub library: &'a NativeLibrary,
    pub programs: &'a Programs<'a>,
    pub cfg: GlmDsaConfig,
    pub weights: GlmWeights<'a>,
    pub stream: *mut c_void,
    pub max_context: usize,
    pub prefill_rows: usize,
    pub pages: usize,
    kv: Vec<Dev<'a>>,
    index: Vec<Option<Dev<'a>>>,
    cos_sin: Dev<'a>,
    decode_workspace: RefCell<Option<Workspace<'a>>>,
    /// One prefill workspace per lane (a serial prefill uses the first) and
    /// one Spark transport thread per lane: a lane's request assembly,
    /// post, polling and partial copies run there while this thread queues
    /// GPU work. Without lane transports, or with `--prefill-waves 1`,
    /// prefill is serial on the caller's transport.
    lane_workspaces: RefCell<Vec<Workspace<'a>>>,
    /// Prefill units in flight on the one Spark transport (PLAN "Prefill
    /// contract"): each unit owns a prefill workspace slot (`lane_workspaces`).
    prefill_lanes: usize,
    /// Recorded after a unit's route/wire copies to its pinned staging: the
    /// host waits for these copies only, never the whole stream.
    staged: Cell<*mut c_void>,
    /// Under context, per staging parity: the (layer, unit) whose attention
    /// last filled it. A unit reuses the view only when its same-sequence
    /// predecessor filled it at this layer; otherwise it gathers its history.
    staging_owner: RefCell<[Option<(usize, usize)>; 2]>,
    /// Set by [`Self::context_unit_turn`] for the attention about to run.
    context_reuse: Cell<bool>,
    /// The prefill unit whose attention runs next (its context gather/commit plan).
    context_unit: Cell<usize>,
    /// The one prefill transport: every unit's waves, up to `prefill_lanes`
    /// outstanding, each landing in its own return slot. Decode keeps its own
    /// link (its captured graphs bake one intake's planes).
    pub prefill_link: RefCell<Option<SparkLink<'a>>>,
    /// Its tokio runtime (receives poll on the engine thread).
    pub prefill_runtime: Option<tokio::runtime::Runtime>,
    terminal: Cell<crate::shared::peer_split::TerminalState>,
    /// Prefill steps keep every row's logits (golden scoring); otherwise the
    /// prefill workspace holds logits for at most `DECODE_ROWS` rows.
    pub full_prefill_logits: bool,
    /// Rows one pipelined prefill call streams through its units (`--prefill-span-rows`).
    pub prefill_span_rows: usize,
    /// Benchmarks: MoE layers skip the Spark exchange (see
    /// `EngineArgs::skip_routed_experts`) and reduce zero partials of
    /// [`SKIP_RANKS`] ranks uploaded through this host-mode intake.
    skip: Option<SparkIntake<'a>>,
    /// Host time per phase since the last reset: GPU wait before the expert
    /// request, the Spark exchange, the logits download.
    pub profile: RefCell<[f64; 3]>,
    /// Host seconds inside the exchanges: building and posting requests,
    /// copying received partials into the pinned staging.
    pub exchange_host: RefCell<[f64; 4]>,
    draft_probe: RefCell<Option<crate::shared::draft::probe::RoundProbe<'a>>>,
    graphs: RefCell<std::collections::HashMap<GraphKey, GraphExec<'a>>>,
    graphs_warmed: Cell<bool>,
    graphs_warming: Cell<bool>,
    graph_misses: Cell<u64>,
    /// The DFlash2 drafter; every step taps its target layers.
    pub drafter: Option<crate::shared::speculation::dflash::DflashDrafter<'a>>,
    pub(super) attention_plan: cuteafd_core::coordinator_programs::AttentionReductionPlan,
    /// L2 prefetch of the next layer's weights during decode exchanges.
    pub l2: Option<crate::shared::l2_prefetch::L2Prefetch>,
    /// The token embedding table (resident on this GPU or read from its shard).
    pub embedding: TokenEmbedding<'a>,
    /// Prefill projections over the FP8 weights run W8A8 (E4M3 activations per
    /// row and 128-K block, the official FP8 release's served numerics); false:
    /// W8A16 (bitwise the former BF16 prefill over dequantized weights).
    pub prefill_w8a8: bool,
    /// This engine's GPU (rank 0 of a head split).
    pub device: i32,
    /// Token-split attention (K3): the build's programs, set at construction (each GPU's
    /// pools then hold half the page ids), and once the peer is attached, its buffers and
    /// exchange (dropped before the head split's exchange, whose abort words it borrows).
    context_mode: Option<GlmContext>,
    layer_execution: Option<layers::Execution<'a>>,
    context: Option<context::GlmContextSplit<'a>>,
    /// Per prefill lane (decode: lane 0) and rank: this step's context tables.
    context_units: RefCell<Vec<[context::ContextUnit; 2]>>,
    /// The head split's second GPU and the exchange between the two.
    peer: Option<GlmPeer<'a>>,
    exchange: Option<PeerExchange<'a>>,
    tp2: Option<Pair<'a>>,
    tp2_exchange: ExchangeDtype,
    force_route_check: Cell<bool>,
}

/// Per layer the latent record pool and (full-indexer layers) the index-key
/// pool, zeroed, and the RoPE table, on the current device.
#[allow(clippy::type_complexity)]
fn caches<'a>(library: &'a NativeLibrary, cfg: &GlmDsaConfig, layers: &[GlmLayer<'_>], pages: usize, max_context: usize)
    -> Result<(Vec<Dev<'a>>, Vec<Option<Dev<'a>>>, Dev<'a>)> {
    let _memory_scope = cuteafd_ffi::memory_ledger::scope("kv");
    let zeroed = |bytes: usize| -> Result<Dev<'a>> {
        let allocation = DeviceAllocation::new(library, bytes.max(256))?;
        library.cuda_zero_bytes(allocation.buffer, allocation.buffer.bytes)?;
        Ok(allocation)
    };
    // One page past the pool is the scratch page padded decode rows write.
    let owns = |layer: &GlmLayer<'_>| layer.attention_owner.is_none_or(|owner| {
        layer.range("q_a_norm").is_some() && owner < 2
    });
    let kv = layers.iter().map(|layer| zeroed(if owns(layer) { (pages + 1) * RECORD_PAGE_BYTES } else { 256 })).collect::<Result<Vec<_>>>()?;
    let index = layers.iter()
        .map(|l| (owns(l) && l.full_indexer).then(|| zeroed((pages + 1) * INDEX_PAGE_BYTES)).transpose())
        .collect::<Result<Vec<_>>>()?;
    // cos | sin of position * theta^(-2i/64), FP32 like the reference's inv_freq.
    let dim = cfg.qk_rope_head_dim;
    let mut table = vec![0f32; max_context * dim];
    let inv: Vec<f32> = (0..dim / 2).map(|i| 1.0 / (cfg.rope_theta as f32).powf((2 * i) as f32 / dim as f32)).collect();
    for p in 0..max_context {
        for (i, f) in inv.iter().enumerate() {
            let angle = p as f32 * f;
            table[p * dim + i] = angle.cos();
            table[p * dim + dim / 2 + i] = angle.sin();
        }
    }
    let cos_sin = DeviceAllocation::new(library, table.len() * 4)?;
    library.copy_h2d(cos_sin.buffer, bytes_of(&table))?;
    Ok((kv, index, cos_sin))
}

fn bytes_of<T: Copy>(values: &[T]) -> &[u8] {
    // SAFETY: plain-old-data slices viewed as bytes for host->device copies.
    unsafe { std::slice::from_raw_parts(values.as_ptr().cast(), std::mem::size_of_val(values)) }
}

impl<'a> GlmEngine<'a> {
    #[allow(clippy::too_many_arguments)]
    pub fn new(library: &'a NativeLibrary, programs: &'a Programs<'a>, cfg: GlmDsaConfig,
        weights: GlmWeights<'a>, stream: *mut c_void, max_context: usize, prefill_rows: usize, pages: usize,
        embedding: TokenEmbedding<'a>, prefill_lanes: usize, context_mode: Option<GlmContext>) -> Result<Self> {
        ensure!((1..=4).contains(&prefill_lanes), "GLM prefill lanes must match admitted inventory");
        ensure!(context_mode.is_none() || pages % 2 == 0, "a context pool has two equal halves, not {pages} pages");
        let quantize_grid = Fp8QuantizeGrid::new(library.sm_count()?, None)?;
        ensure!(embedding.hidden() == cfg.hidden, "embedding rows of {} for hidden {}", embedding.hidden(), cfg.hidden);
        let local = if context_mode.is_some() { pages / 2 } else { pages };
        let (kv, index, cos_sin) = caches(library, &cfg, &weights.layers, local, max_context)?;
        let device = library.cuda_get_device()?;
        Ok(Self { quantize_grid, device, native_experts: Vec::new(), native_dense: Vec::new(), peer: None, exchange: None, tp2: None, tp2_exchange: ExchangeDtype::Bf16, force_route_check: Cell::new(false), context_mode, context: None, layer_execution: None,
            context_units: RefCell::new(Vec::new()), library, programs, cfg, weights, stream, max_context, prefill_rows, pages, kv, index, cos_sin,
            decode_workspace: RefCell::new(None), lane_workspaces: RefCell::new(Vec::new()),
            prefill_lanes, staged: Cell::new(library.cuda_event_create_ordering()?),
            staging_owner: RefCell::new([None; 2]), context_reuse: Cell::new(false), context_unit: Cell::new(0), prefill_link: RefCell::new(None), prefill_runtime: None,
            attention_plan: cuteafd_core::coordinator_programs::AttentionReductionPlan::new(DECODE_ROWS, None, None)?,
            terminal: Cell::new(crate::shared::peer_split::TerminalState::Active), spark_bf16: false, full_prefill_logits: false, prefill_span_rows: DEFAULT_PREFILL_SPAN_ROWS, skip: None, l2: None, profile: RefCell::new([0.0; 3]),
            exchange_host: RefCell::new([0.0; 4]), draft_probe: RefCell::new(None),
            graphs: RefCell::new(std::collections::HashMap::new()), graphs_warmed: Cell::new(false), graphs_warming: Cell::new(false), graph_misses: Cell::new(0), drafter: None, embedding, prefill_w8a8: true })
    }

    /// Arm after graph warmup; only explicit verify rounds record signals.
    pub(crate) fn arm_draft_probe(&self) -> Result<()> {
        let dense = self.weights.layers.iter().map(|layer| layer.dense).collect();
        *self.draft_probe.borrow_mut() = Some(crate::shared::draft::probe::RoundProbe::new(
            self.library, dense, self.cfg.topk, DECODE_ROWS)?);
        Ok(())
    }

    pub(crate) fn probe_begin(&self, rows: usize) {
        if let Some(probe) = self.draft_probe.borrow_mut().as_mut() { probe.begin(rows); }
    }

    pub(crate) fn probe_end(&self) {
        if let Some(probe) = self.draft_probe.borrow_mut().as_mut() { probe.end(); }
    }

    /// Read only after the verifier's normal stream synchronization.
    pub(crate) fn probe_finish<T>(&self, observe: impl FnOnce(&crate::shared::draft::routes::RoundRoutes,
        &[Option<f64>]) -> T) -> Option<T> {
        let mut probe = self.draft_probe.borrow_mut();
        let (routes, times) = probe.as_mut()?.finish()?;
        Some(observe(routes, &times))
    }

    /// Every layer's latent record pool (656 B per row, 64-row pages) and, on full-indexer
    /// layers, its index-key pool (8448 B per page: 64 x 128 E4M3 keys, then 64 FP32 scales).
    pub(crate) fn paged_buffers(&self) -> Vec<(cuteafd_ffi::CuteafdDeviceBuffer, Option<cuteafd_ffi::CuteafdDeviceBuffer>)> {
        self.paged_buffers_on(0)
    }

    /// [`Self::paged_buffers`] of rank `rank` (1: the head split's copy on its second GPU).
    pub(crate) fn paged_buffers_on(&self, rank: usize)
        -> Vec<(cuteafd_ffi::CuteafdDeviceBuffer, Option<cuteafd_ffi::CuteafdDeviceBuffer>)> {
        let (kv, index) = match (rank, &self.peer) {
            (1, Some(peer)) => (&peer.kv, &peer.index),
            _ => (&self.kv, &self.index),
        };
        let layers = if rank == 1 { &self.peer.as_ref().unwrap().layers } else { &self.weights.layers };
        kv.iter().zip(index).zip(layers).filter(|(_, layer)| layer.range("q_a_norm").is_some())
            .map(|((kv, index), _)| (kv.buffer, index.as_ref().map(|i| i.buffer))).collect()
    }

    /// GPUs this engine runs on: 2 under a head split.
    pub fn ranks(&self) -> usize {
        1 + usize::from(self.peer.is_some())
    }

    /// Attaches the head split's second GPU: `device` with `stream`, holding
    /// `layers` (every layer's rank-1 share, see `GlmLoader::model`). Enables
    /// peer access both ways, loads the programs there, and allocates its
    /// caches, RoPE table and the exchange (four slots per prefill lane).
    pub fn attach_peer(&mut self, device: i32, stream: *mut c_void, layers: Vec<GlmLayer<'a>>) -> Result<()> {
        ensure!(layers.len() == self.weights.layers.len() && layers.iter().chain(&self.weights.layers).all(|l| l.split),
            "attach_peer needs the head-split shares of every loaded layer");
        let rows = self.prefill_rows.max(DECODE_ROWS);
        let exchange = PeerExchange::new_abortable(self.library, [RankDevice { device: self.device, stream: self.stream },
            RankDevice { device, stream }], 4 * self.prefill_lanes.max(1), rows * self.cfg.hidden * self.tp2_exchange.element_bytes())?;
        let peer = exchange.on(1, || -> Result<GlmPeer<'a>> {
            let selected = cuteafd_core::coordinator_programs::CoordinatorPrograms { family: "glm", split_family: Some("glm2") };
            self.programs.load_matching(|name| selected.contains(name))?;
            let (kv, index, cos_sin) = caches(self.library, &self.cfg, &layers, self.local_pages(), self.max_context)?;
            Ok(GlmPeer { device, stream, layers, kv, index, cos_sin, decode_workspace: RefCell::new(None),
                lane_workspaces: RefCell::new(Vec::new()), graphs: RefCell::new(std::collections::HashMap::new()),
                l2: None })
        })?;
        self.peer = Some(peer);
        self.exchange = Some(exchange);
        Ok(())
    }

    /// Whether the KV pools are page-partitioned (token-split attention).
    pub fn context_kv(&self) -> bool {
        self.context_mode.is_some()
    }

    /// Allocates the token-split buffers and exchange once the peer is attached; `extent`:
    /// the admitted compiled context (the staging the solver charged).
    pub fn attach_context(&mut self, extent: usize, manifest: &serde_json::Value) -> Result<()> {
        let context = self.context_mode.context("this engine was built without context attention")?;
        let exchange = self.exchange()?;
        let abort = match (exchange.abort_word(0)?, exchange.abort_word(1)?) {
            (Some(a), Some(b)) => Some([a, b]),
            _ => None,
        };
        let split = context::GlmContextSplit::new(self, context, extent, manifest, abort)?;
        self.context = Some(split);
        Ok(())
    }

    fn peer(&self) -> Result<&GlmPeer<'a>> {
        self.peer.as_ref().context("no head-split peer")
    }

    fn exchange(&self) -> Result<&PeerExchange<'a>> {
        self.exchange.as_ref().context("no head-split exchange")
    }

    /// The stream of rank `rank`.
    pub(crate) fn stream_of(&self, rank: usize) -> *mut c_void {
        match (rank, &self.peer) {
            (1, Some(peer)) => peer.stream,
            _ => self.stream,
        }
    }

    /// Runs `body` with rank `rank`'s device current (this engine's device again after).
    pub(crate) fn on<T>(&self, rank: usize, body: impl FnOnce() -> Result<T>) -> Result<T> {
        match (rank, &self.peer) {
            (1, Some(peer)) => crate::shared::peer_split::on_device(self.library, peer.device, self.device, body),
            _ => body(),
        }
    }

    /// Receive slot `slot` of rank `rank`.
    fn recv(&self, rank: usize, slot: usize) -> Result<*mut c_void> {
        self.exchange()?.recv(rank, slot)
    }

    /// `name` (a `glm_*` program) for `layer`: its head-split share's (`glm2_*`) when split.
    fn program(layer: &GlmLayer<'_>, name: &str) -> String {
        match (if name.starts_with("glm_ffn_") { layer.split } else { layer.attention_split }, name.strip_prefix("glm_")) {
            (true, Some(rest)) => format!("glm2_{rest}"),
            _ => name.to_string(),
        }
    }

    /// Launches `name` on rank `rank`'s stream.
    fn run_on(&self, rank: usize, name: &str, pointers: &[(&str, *mut c_void)], scalars: &[Scalar]) -> Result<()> {
        let names: Vec<&str> = pointers.iter().map(|(n, _)| *n).collect();
        let program = self.programs.program(name, &names)?;
        let raw: Vec<*mut c_void> = pointers.iter().map(|(_, p)| *p).collect();
        // SAFETY: every pointer names a live allocation of rank `rank`'s GPU sized for
        // the rows in `scalars`; that rank's stream orders all its launches.
        self.on(rank, || unsafe { program.launch(&raw, scalars, self.stream_of(rank)) })
            .with_context(|| format!("{name} with {scalars:?}"))
    }

    /// On rank `rank`: `h += bf16(first + second)` (`deltas` 2; the same operand
    /// order on both GPUs of a head split), `h += first` (1) or nothing (0),
    /// then `x = weight * RMSNorm(h)`.
    #[allow(clippy::too_many_arguments)]
    fn norm_full(&self, rank: usize, w: &Workspace<'_>, weight: *mut c_void, deltas: i32, rows: Scalar,
        first: *mut c_void, second: *mut c_void) -> Result<()> {
        self.run_on(rank, "glm_norm", &[("residual", w.h.buffer.ptr), ("delta0", first), ("delta1", second),
            ("weight", weight), ("out", w.x.buffer.ptr)], &[rows, Scalar::I32(deltas)])
    }

    /// Benchmarks: MoE layers skip the Spark exchange from now on.
    pub fn skip_routed_experts(&mut self) -> Result<()> {
        let rows = self.prefill_rows.max(DECODE_ROWS);
        self.skip = Some(SparkIntake::new(self.library, IntakeMode::Host, SKIP_RANKS, rows, self.cfg.hidden * 2)?);
        Ok(())
    }

    /// Whether MoE layers skip the Spark exchange.
    pub fn skip_routed(&self) -> bool {
        self.skip.is_some()
    }

    fn alloc(&self, bytes: usize) -> Result<Dev<'a>> {
        DeviceAllocation::new(self.library, bytes.max(256))
    }

    fn run(&self, name: &str, pointers: &[(&str, *mut c_void)], scalars: &[Scalar]) -> Result<()> {
        self.run_on(0, name, pointers, scalars)
    }

    fn scratch(&self, name: &str) -> Result<usize> {
        Ok(self.programs.spec(name)?.scratch.get("scratch").copied().unwrap_or(0) as usize)
    }

    pub fn prepare_scoring_prefill(&self) -> Result<()> {
        let count = if self.pipelined() { self.prefill_lanes } else { 1 };
        let mut slots = self.lane_workspaces.borrow_mut();
        while slots.len() < count { slots.push(self.workspace(self.prefill_rows, false)?); }
        self.peer_workspaces(false, count)?;
        Ok(())
    }

    fn workspace(&self, t: usize, decode: bool) -> Result<Workspace<'a>> {
        self.workspace_on(0, t, decode)
    }

    /// Rank `rank`'s workspace for steps of up to `t` rows (rank 1 has no
    /// router, expert or head buffers), allocated on that rank's GPU.
    fn workspace_on(&self, rank: usize, t: usize, decode: bool) -> Result<Workspace<'a>> {
        self.on(rank, || self.workspace_here(rank, t, decode))
    }

    fn workspace_here(&self, rank: usize, t: usize, decode: bool) -> Result<Workspace<'a>> {
        let _memory_scope = cuteafd_ffi::memory_ledger::scope("workspace");
        let (h, heads) = (self.cfg.hidden, self.cfg.heads);
        let (cap, mode) = if decode { ("m64", "decode") } else { ("m4096", "prefill") };
        let lead = rank == 0;
        let lead_only = |bytes: usize| if lead { bytes } else { 256 };
        let mut scratch = 0usize;
        let split = self.weights.layers.first().is_some_and(|l| l.split);
        let (moe, dense) = (self.cfg.moe_intermediate, self.cfg.dense_intermediate);
        // A head split's layers (all of them) run its share's programs.
        let (prefix, moe, dense) = if split { ("glm2", moe / 2, dense / 2) } else { ("glm", moe, dense) };
        // Every layer of a head split runs its share's heads on both GPUs.
        let whole = self.weights.layers.iter().any(|layer| layer.attention_owner.is_some());
        let heads = if split && !whole { heads / 2 } else { heads };
        for name in [format!("glm_index_producer_{cap}"), format!("{prefix}_producer_{cap}"),
            format!("{prefix}_sparse_mla_{mode}_{cap}"), format!("{prefix}_o_{cap}"), format!("{prefix}_ffn_i{moe}_{cap}"),
            format!("{prefix}_ffn_i{dense}_{cap}")] {
            scratch = scratch.max(self.scratch(&name)?);
        }
        let tensor_ffn = format!("{prefix}_ffn_i{dense}_pt_{cap}");
        if self.weights.layers.iter().any(|layer|
            layer.mlp_numerics == cuteafd_loader::plan::checkpoint::ModuleNumerics::W8A8) {
            // Declared W8A8 cannot fall back, including decode graph warmup.
            scratch = scratch.max(self.scratch(&tensor_ffn)?);
        }
        // Optional programs for checkpoint-precision modules in older libraries.
        let mut optional = vec![format!("glm_index_producer_bf16_{cap}"), format!("{prefix}_producer_bf16_{cap}"),
            format!("{prefix}_o_bf16_{cap}"), format!("{prefix}_ffn_i{moe}_bf16_{cap}")];
        if !decode {
            optional.push(format!("{prefix}_ffn_i{dense}_pt_{cap}"));
        }
        for name in optional {
            scratch = scratch.max(self.scratch(&name).unwrap_or(0));
        }
        if whole {
            let extra = cuteafd_loader::placement::families::glm5::step_scratch(&self.cfg, false, decode,
                |name| self.scratch(name).ok().map(|n| n as u64))?;
            scratch = scratch.max(extra.programs as usize);
        }
        let head_workspace = self.alloc(if lead { VOCABULARY_HEAD_WORKSPACE } else { 256 })?;
        let topk = self.alloc(self.scratch(&format!("glm_index_topk_{mode}_{cap}"))?)?;
        self.library.cuda_zero_bytes(topk.buffer, topk.buffer.bytes)?;
        let lengths: Vec<i32> = vec![self.cfg.index_topk as i32; t];
        let lengths_dev = self.alloc(t * 4)?;
        self.library.copy_h2d(lengths_dev.buffer, bytes_of(&lengths))?;
        Ok(Workspace {
            rows: t,
            h: self.alloc(t * h * 2)?,
            x: self.alloc(t * h * 2)?,
            query: self.alloc(t * heads * 576 * 2)?,
            q_resid: self.alloc(t * self.cfg.q_lora_rank * 2)?,
            q_fp8: self.alloc(t * self.cfg.index_heads * self.cfg.index_head_dim)?,
            head_weights: self.alloc(t * self.cfg.index_heads * 4)?,
            indices: self.alloc(t * self.cfg.index_topk * 4)?,
            lengths: lengths_dev,
            attn: self.alloc(t * heads * 512 * 2)?,
            delta: self.alloc(t * h * 2)?,
            positions: self.alloc(t * 8)?,
            slots: self.alloc(t * 8)?,
            page_table: self.alloc(if decode { t * self.pages * 4 } else { self.pages * 4 })?,
            cache_lengths: self.alloc(t * 4)?,
            scratch: self.alloc(scratch)?,
            topk_scratch: topk,
            logits: self.alloc(lead_only(if decode || self.full_prefill_logits { t } else { t.min(DECODE_ROWS) }
                * self.cfg.vocab_size * 4))?,
            router_logits: self.alloc(if lead || self.tp2.is_some() { t * self.cfg.experts * 4 } else { 256 })?,
            route_ids: self.alloc(if lead || self.tp2.is_some() { t * self.cfg.topk * 4 } else { 256 })?,
            route_weights: self.alloc(if lead || self.tp2.is_some() { t * self.cfg.topk * 4 } else { 256 })?,
            wire: self.alloc(if lead || self.tp2.is_some() { t * (h + h / 32) } else { 256 })?,
            shared: self.alloc(t * h * 2)?,
            native_shared: self.native_experts.get(rank)
                .map(|native| native.execution(self.library, self.programs)).transpose()?,
            native_dense: if self.weights.layers.iter().any(|l| l.is_native()) {
                let split = self.weights.layers.first().is_some_and(|l| l.split);
                let attention_split = self.weights.layers.first().is_some_and(|l| l.attention_split);
                Some(native_dense::NativeScratch::new(self, self.cfg.heads / if attention_split { 2 } else { 1 },
                    self.cfg.dense_intermediate / if split { 2 } else { 1 }, t)?)
            } else { None },
            payload: self.tp2.as_ref().map(|tp2| self.alloc(tp2.combine.payload_bytes(t, h))).transpose()?,
            router_host: RefCell::new(HostAllocation::new(self.library, lead_only(t * (self.cfg.topk * 8 + if self.spark_bf16 { h * 2 } else { h + h / 32 })))?),
            ids: self.alloc(t * 4)?,
            select: self.alloc(t * 8)?,
            // SAFETY: the workspace buffer lives in the same struct and drops after the head.
            head: if lead {
                Some(unsafe { self.library.vocabulary_head_rows(head_workspace.buffer.ptr, h as u32, t as u32,
                    self.cfg.vocab_size as u32)? })
            } else {
                None
            },
            _head_workspace: head_workspace,
        })
    }

    fn put<T: Copy>(&self, dev: &Dev<'_>, values: &[T]) -> Result<()> {
        let bytes = bytes_of(values);
        ensure!(bytes.len() <= dev.buffer.bytes, "table exceeds its buffer");
        self.library.copy_h2d(cuteafd_ffi::CuteafdDeviceBuffer { bytes: bytes.len(), ..dev.buffer }, bytes)
    }

    fn download(&self, dev: &Dev<'_>, bytes: usize) -> Result<Vec<u8>> {
        // SAFETY: the engine owns this stream.
        unsafe { self.library.cuda_stream_synchronize(self.stream)? };
        let mut out = vec![0u8; bytes];
        self.library.copy_d2h(&mut out, cuteafd_ffi::CuteafdDeviceBuffer { bytes, ..dev.buffer })?;
        Ok(out)
    }

    /// Prefills a sequence from its length through every resident layer and
    /// returns the last row's logits when all layers are resident.
    /// `on_layer` receives each layer's output rows (BF16 [t, hidden]).
    pub fn prefill(&self, placement: &mut GlmPlacement, tokens: &[u32],
        experts: Option<(&mut SparkLink<'_>, &tokio::runtime::Runtime)>,
        on_layer: Option<&mut dyn FnMut(usize, &[u8]) -> Result<()>>) -> Result<Option<Vec<f32>>> {
        self.prefill_rows_logits(placement, tokens, experts, on_layer, 1)
    }

    /// [`Self::prefill`] leaving the last row's logits on the device.
    pub fn prefill_device(&self, placement: &mut GlmPlacement, tokens: &[u32],
        experts: Option<(&mut SparkLink<'_>, &tokio::runtime::Runtime)>) -> Result<Option<DeviceLogits>> {
        match self.prefill_step(placement, tokens, experts, None, 1)? {
            Some(StepLogits::Device(logits)) => Ok(Some(logits)),
            Some(StepLogits::Host(_)) => anyhow::bail!("the last prefill row's logits span lanes"),
            None => Ok(None),
        }
    }

    /// Downloads a step's logits (counted in the logits phase).
    fn host_logits(&self, logits: Option<StepLogits>) -> Result<Option<Vec<f32>>> {
        let timer = std::time::Instant::now();
        let out = match logits {
            Some(StepLogits::Device(logits)) => Some(logits.to_host(self.library)?),
            Some(StepLogits::Host(logits)) => Some(logits),
            None => None,
        };
        self.profile.borrow_mut()[2] += timer.elapsed().as_secs_f64();
        Ok(out)
    }

    /// Whether a Spark prefill runs as lanes (a lane transport, every layer resident).
    fn pipelined(&self) -> bool {
        self.prefill_lanes > 1 && self.weights.layers.len() == self.cfg.layers && self.prefill_link.borrow().is_some()
    }

    /// Longest chunk one prefill call takes. Pipelined, a call is a continuous
    /// stream of units (`prefill_span_rows`, default 16384 rows), so the Sparks
    /// drain only at span ends, not every `waves x unit` rows.
    pub fn prefill_capacity(&self) -> usize {
        if self.pipelined() { self.prefill_span_rows().max(self.prefill_lanes * self.prefill_rows) }
        else { self.prefill_rows }
    }

    /// Rows one pipelined prefill call streams through its units
    /// (`--prefill-span-rows`; `waves x unit` restores fixed chunks).
    fn prefill_span_rows(&self) -> usize {
        self.prefill_span_rows.clamp(256, self.max_context.max(256))
    }

    fn prefill_tables(&self, placement: &GlmPlacement, start: usize, t: usize) -> Result<StepTables> {
        let used = (start + t).div_ceil(PAGE_ROWS);
        Ok(StepTables {
            decode: false,
            positions: (start..start + t).map(|p| p as i64).collect(),
            slots: (start..start + t).map(|p| placement.slot(p)).collect::<Result<_>>()?,
            page_table: placement.table(used).collect(),
            table_width: used,
            table_stride: 0,
            cache_lengths: (start..start + t).map(|p| (p + 1) as i32).collect(),
            lengths: (start..start + t).map(|p| (p + 1).min(self.cfg.index_topk) as i32).collect(),
            exchange_rows: t,
        })
    }

    /// [`Self::prefill`] returning the logits of the chunk's last `logit_rows` rows.
    /// Without `on_layer`, a Spark chunk of at least 2 x 256 rows runs as
    /// [`PREFILL_LANES`] lanes (see [`Self::step_lanes`]).
    pub fn prefill_rows_logits(&self, placement: &mut GlmPlacement, tokens: &[u32],
        experts: Option<(&mut SparkLink<'_>, &tokio::runtime::Runtime)>,
        on_layer: Option<&mut dyn FnMut(usize, &[u8]) -> Result<()>>, logit_rows: usize) -> Result<Option<Vec<f32>>> {
        let logits = self.prefill_step(placement, tokens, experts, on_layer, logit_rows)?;
        self.host_logits(logits)
    }

    fn prefill_step(&self, placement: &mut GlmPlacement, tokens: &[u32],
        experts: Option<(&mut SparkLink<'_>, &tokio::runtime::Runtime)>,
        on_layer: Option<&mut dyn FnMut(usize, &[u8]) -> Result<()>>, logit_rows: usize) -> Result<Option<StepLogits>> {
        let (start, t) = (placement.len, tokens.len());
        if on_layer.is_none() && self.pipelined() && t >= 2 * MIN_LANE_ROWS && experts.is_some() {
            // Equal page-aligned units of at most `prefill_rows`, a multiple of the
            // waves in flight (PLAN "Prefill contract"); each unit's pages start where
            // the previous unit's end.
            ensure!(start + t <= self.max_context, "prefill of {t} rows at {start} exceeds the context");
            let cuts = crate::shared::prefill_pipeline::prefill_units(t, self.prefill_rows, self.prefill_lanes,
                MIN_LANE_ROWS)?;
            let (mut lanes, mut chunks) = (Vec::new(), Vec::new());
            for rows in cuts {
                lanes.push(self.prefill_tables(placement, start + rows.start, rows.len())?);
                chunks.push(&tokens[rows]);
            }
            let mut link = self.prefill_link.borrow_mut();
            let link = link.as_mut().context("GLM prefill units need the prefill link")?;
            let runtime = self.prefill_runtime.as_ref().context("GLM prefill runtime")?;
            let logits = self.step_lanes(&lanes, &chunks, logit_rows, link, runtime)?;
            placement.len += t;
            return Ok(logits);
        }
        ensure!(t > 0 && t <= self.prefill_rows && start + t <= self.max_context, "prefill of {t} rows at {start}");
        let tables = self.prefill_tables(placement, start, t)?;
        let logits = self.step(&tables, tokens, logit_rows, experts, on_layer)?;
        placement.len += t;
        Ok(logits.map(StepLogits::Device))
    }

    /// Appends each sequence's tokens (one for decode, several for a
    /// speculative verify) at its length in one decode-shaped step; returns
    /// every row's logits. A caller that rejects a suffix sets `len` back:
    /// GLM keeps no recurrent state, and rejected records are overwritten.
    pub fn verify(&self, sequences: &mut [(&mut GlmPlacement, usize)], tokens: &[u32],
        experts: Option<(&mut SparkLink<'_>, &tokio::runtime::Runtime)>,
        on_layer: Option<&mut dyn FnMut(usize, &[u8]) -> Result<()>>) -> Result<Option<Vec<f32>>> {
        let logits = self.verify_device_hooked(sequences, tokens, experts, on_layer)?;
        self.host_logits(logits.map(StepLogits::Device))
    }

    /// [`Self::verify`] leaving every row's logits on the device, with the
    /// decode graph's greedy selection of them.
    pub fn verify_device(&self, sequences: &mut [(&mut GlmPlacement, usize)], tokens: &[u32],
        experts: Option<(&mut SparkLink<'_>, &tokio::runtime::Runtime)>) -> Result<Option<DeviceLogits>> {
        self.verify_device_hooked(sequences, tokens, experts, None)
    }

    fn verify_device_hooked(&self, sequences: &mut [(&mut GlmPlacement, usize)], tokens: &[u32],
        experts: Option<(&mut SparkLink<'_>, &tokio::runtime::Runtime)>,
        on_layer: Option<&mut dyn FnMut(usize, &[u8]) -> Result<()>>) -> Result<Option<DeviceLogits>> {
        let rows: usize = sequences.iter().map(|(_, n)| n).sum();
        ensure!(rows > 0 && rows <= DECODE_ROWS && tokens.len() == rows, "decode step of {rows} rows");
        let mut needed = 1;
        for (placement, count) in sequences.iter() {
            ensure!(placement.len + count <= self.max_context, "decode at {} past the context", placement.len + count - 1);
            needed = needed.max((placement.len + count).div_ceil(PAGE_ROWS));
        }
        let width = self.table_width_bucket(needed);
        let mut tables = StepTables { decode: true, positions: Vec::new(), slots: Vec::new(), page_table: Vec::new(),
            table_width: width, table_stride: width, cache_lengths: Vec::new(), lengths: Vec::new(), exchange_rows: rows };
        for (placement, count) in sequences.iter() {
            let pages = placement.pages.len().min(width);
            for position in placement.len..placement.len + count {
                tables.positions.push(position as i64);
                tables.slots.push(placement.slot(position)?);
                tables.cache_lengths.push((position + 1) as i32);
                // The index top-k selects every earlier token up to its k,
                // leading the row: the sparse MLA reads only those.
                tables.lengths.push((position + 1).min(self.cfg.index_topk) as i32);
                tables.page_table.extend(placement.table(pages));
                tables.page_table.extend(std::iter::repeat_n(0, width - pages));
            }
        }
        let bucket = glm_decode_row_bucket(rows, width)?;
        self.pad_rows(&mut tables, bucket);
        let mut padded = tokens.to_vec();
        padded.resize(bucket, 0);
        // Padded rows' logits follow the real rows'; only those are returned.
        let logits = self.step(&tables, &padded, bucket, experts, on_layer)?
            .map(|logits| DeviceLogits { rows, ..logits });
        for (placement, count) in sequences.iter_mut() {
            placement.len += *count;
        }
        Ok(logits)
    }

    /// The page-table width a decode step whose longest row needs `pages`
    /// pages runs with: a power of two from 16, capped at the
    /// context's pages.
    fn table_width_bucket(&self, pages: usize) -> usize {
        glm_decode_table_width(pages, self.max_context, self.pages)
    }

    /// Every table width [`Self::table_width_bucket`] returns.
    fn table_width_buckets(&self) -> Vec<usize> {
        glm_decode_table_widths(self.max_context, self.pages)
    }

    /// Pads a decode step's tables to `bucket` rows: position 0 of the scratch
    /// page past the pool (its own records and index keys; no sequence reads them).
    fn pad_rows(&self, tables: &mut StepTables, bucket: usize) {
        // Under context the slot id is a logical page id (`rank_tables` sends padded rows
        // to each GPU's scratch row); the pages past the pool are never a real row's.
        tables.pad_rows(self.pages, bucket);
    }

    /// Captures every decode graph serving can replay: one padded step per
    /// (row bucket, table width) over the scratch page (one real row goes to
    /// the Sparks), so no request captures. Returns the graph count.
    pub fn warm_decode_graphs(&self, mut experts: Option<(&mut SparkLink<'_>, &tokio::runtime::Runtime)>) -> Result<usize> {
        let started = std::time::Instant::now();
        let free = |rank: usize| self.on(rank, || self.library.cuda_memory_info().map(|(free, _)| free as i64));
        let before: Vec<i64> = (0..self.ranks()).map(free).collect::<Result<_>>()?;
        let owned = if self.layers_kv() {
            let owners: Vec<_> = self.weights.layers.iter().map(|layer| layer.attention_owner.unwrap()).collect();
            let tp2: Vec<_> = (0..owners.len()).map(|layer| self.tp2_layer(layer)).collect();
            Some(cuteafd_loader::serving_capacity::glm_graphs::glm_owned_startup_graphs(
                self.max_context, self.pages, &owners, &tp2).map_err(anyhow::Error::msg)?)
        } else { None };
        let shapes = owned.as_ref().map(|startup| startup.keys.clone()).unwrap_or_else(||
            cuteafd_loader::serving_capacity::glm_graphs::glm_graph_shapes(self.max_context, self.pages));
        self.graphs_warming.set(true);
        for shape in &shapes {
            let (width, bucket) = (shape.table_width, shape.rows);
            let mut tables = StepTables { decode: true, positions: Vec::new(), slots: Vec::new(),
                page_table: Vec::new(), table_width: width, table_stride: width, cache_lengths: Vec::new(),
                lengths: Vec::new(), exchange_rows: 1 };
            self.pad_rows(&mut tables, bucket);
            let experts = experts.as_mut().map(|(link, runtime)| (&mut **link, *runtime));
            self.step(&tables, &vec![0; bucket], bucket, experts, None)?;
        }
        if let Some(startup) = &owned {
            let execution = self.layer_execution.as_ref().context("GLM layer execution")?;
            for rank in 0..2 {
                let graphs = if rank == 0 { self.graphs.borrow() } else { self.peer()?.graphs.borrow() };
                ensure!(graphs.len() as u64 == startup.inventory.ranks[rank].executables,
                    "GLM rank {rank} captured {} graphs, GraphSet expected {}", graphs.len(), startup.inventory.ranks[rank].executables);
                for shape in &startup.keys {
                    for segment in execution.segments.iter().filter(|segment| segment.rank == rank) {
                        ensure!(graphs.contains_key(&GraphKey { layer: segment.phase.key(segment.layer), rows: shape.rows,
                            table_width: shape.table_width, table_stride: shape.table_width, previous: Previous::First, head: false }),
                            "GLM startup capture missing from GraphSet: {segment:?} {shape:?}");
                    }
                }
            }
            tracing::info!(inventory = ?startup.inventory, "GLM owned startup GraphSet captured");
        }
        // SAFETY: both ranks' streams are live; nothing else is queued.
        unsafe { self.library.cuda_stream_synchronize(self.stream)? };
        if self.peer.is_some() {
            let stream = self.stream_of(1);
            self.on(1, || unsafe { self.library.cuda_stream_synchronize(stream) })?;
        }
        let graphs = self.graphs.borrow().len() + self.peer.as_ref().map_or(0, |p| p.graphs.borrow().len());
        let bytes: Vec<i64> = (0..self.ranks()).map(|rank| Ok(before[rank] - free(rank)?)).collect::<Result<_>>()?;
        self.graphs_warming.set(false);
        self.graphs_warmed.set(true);
        tracing::info!(graphs, ?bytes, widths = ?self.table_width_buckets(), rows = ?ROW_BUCKETS,
            long_rows = ?cuteafd_loader::serving_capacity::GLM_LONG_ROW_BUCKETS,
            elapsed_ms = started.elapsed().as_millis() as u64, "GLM decode graphs captured at startup");
        Ok(graphs)
    }

    fn step(&self, tables: &StepTables, tokens: &[u32], logit_rows: usize,
        experts: Option<(&mut SparkLink<'_>, &tokio::runtime::Runtime)>,
        on_layer: Option<&mut dyn FnMut(usize, &[u8]) -> Result<()>>) -> Result<Option<DeviceLogits>> {
        crate::shared::experts::rtx::pair::submit_scoped(
            || self.require_live(), || {
                crate::shared::decode_graph::fatal_drain(self.retire_tp2(), "GLM submission shutdown");
                Ok(())
            },
            |scope| self.step_inner(scope, tables, tokens, logit_rows, experts, on_layer))
    }

    fn step_inner(&self, scope: &crate::shared::experts::rtx::pair::SubmissionScope<'_>, tables: &StepTables, tokens: &[u32], logit_rows: usize,
        mut experts: Option<(&mut SparkLink<'_>, &tokio::runtime::Runtime)>,
        mut on_layer: Option<&mut dyn FnMut(usize, &[u8]) -> Result<()>>) -> Result<Option<DeviceLogits>> {
        let (h, t) = (self.cfg.hidden, tables.positions.len());
        // Decode steps have their own workspace; a serial prefill uses the first lane's.
        if tables.decode && self.decode_workspace.borrow().is_none() {
            *self.decode_workspace.borrow_mut() = Some(self.workspace(DECODE_ROWS, true)?);
        }
        if !tables.decode && self.lane_workspaces.borrow().is_empty() {
            let first = self.workspace(self.prefill_rows, false)?;
            self.lane_workspaces.borrow_mut().push(first);
        }
        let (decode_workspace, lane_workspaces) = (self.decode_workspace.borrow(), self.lane_workspaces.borrow());
        let w = if tables.decode { decode_workspace.as_ref().context("workspace")? } else { &lane_workspaces[0] };
        ensure!(t <= w.rows && logit_rows <= t, "step exceeds the workspace");
        ensure!(tables.decode || self.full_prefill_logits || logit_rows <= DECODE_ROWS,
            "prefill logits past {DECODE_ROWS} rows need full_prefill_logits");
        // The head split's second GPU: its workspace of the same shape and the same tables
        // (under context, each rank's own: slots, half tables, shard lengths).
        let peer_workspaces = self.peer_workspaces(tables.decode, 1)?;
        let w1 = peer_workspaces.as_ref().map(|p| p.get(0)).transpose()?;
        let ranked = self.step_tables(&[(w, w1, tables)], &[placement_pages(tables)])?;
        let tables = &ranked[0][0];
        ensure!(tokens.len() == t, "{} tokens for a {t}-row step", tokens.len());
        let rows = Scalar::I32(t as i32);
        let cap = if tables.decode { "m64" } else { "m4096" };
        let layers = &self.weights.layers;
        if let Some(pair) = &self.tp2 {
            pair.begin_checks(&[t], self.force_route_check.get())?;
        }
        if self.layer_execution.is_some() {
            self.embedding.embed(tokens, w.ids.buffer, 1, w.h.buffer, self.stream)?;
            return self.layer_step(scope, w, w1.context("owned attention peer")?, &ranked[0],
                t, logit_rows, &mut experts, on_layer);
        }
        if tables.decode && logit_rows > 0 && on_layer.is_none() && layers.len() == self.cfg.layers {
            // The first captured segment gathers the rows from the staged ids;
            // the last runs the head and the greedy selection.
            let gather = self.embedding.device_gather();
            if gather {
                self.embedding.check(tokens)?;
                self.put(&w.ids, tokens)?;
            } else {
                self.embedding.embed(tokens, w.ids.buffer, 1, w.h.buffer, self.stream)?;
            }
            let head = logit_rows == t;
            self.decode_layers(w, w1, &ranked[0], t, &mut experts, gather, head)?;
            if !head {
                self.launch_head(w, t, logit_rows)?;
            }
            if let Some(pair) = &self.tp2 { pair.finish_checks()?; }
            return Ok(Some(self.device_logits(w, logit_rows, head)));
        }
        self.embedding.embed(tokens, w.ids.buffer, 1, w.h.buffer, self.stream)?;
        let bytes = t * h * 2;
        if let Some(w1) = w1 {
            self.exchange()?.push_to(0, DIRECT, w.h.buffer.ptr, w1.h.buffer.ptr, bytes)?;
        }
        self.norm(w, &layers[0], "input_norm", 0, rows)?;
        for (index, layer) in layers.iter().enumerate() {
            if let Some(w1) = w1 {
                self.peer_attention(index, 0, w1, rows, cap, &ranked[0], bytes)?;
            }
            self.attention_unit(0, w, index, layer, rows, cap, &ranked[0], 0)?;
            // h += attention; x = post_attention_layernorm(h)
            let (attended, ffn) = (slot(index, false, 0), slot(index, true, 0));
            match w1 {
                Some(w1) => {
                    self.exchange()?.push(0, attended, w.delta.buffer.ptr, bytes)?;
                    self.exchange()?.wait(0, attended)?;
                    self.norm_full(0, w, layer.ptr("post_norm")?, 2, rows, w.delta.buffer.ptr, self.recv(0, attended)?)?;
                    self.peer_ffn(index, 0, w1, rows, cap, bytes)?;
                }
                None => self.run("glm_norm", &[("residual", w.h.buffer.ptr), ("delta0", w.delta.buffer.ptr),
                    ("delta1", w.delta.buffer.ptr), ("weight", layer.ptr("post_norm")?), ("out", w.x.buffer.ptr)],
                    &[rows, Scalar::I32(1)])?,
            }
            if layer.dense {
                self.ffn(w, layer, self.cfg.dense_intermediate, cap, w.delta.buffer.ptr, rows)?;
                if w1.is_some() {
                    self.exchange()?.push(0, ffn, w.delta.buffer.ptr, bytes)?;
                }
            } else if self.tp2_layer(index) {
                self.moe_front(w, layer, t)?;
                self.tp2_pair(index, 0, w, w1.context("TP2 peer workspace")?, t, cap)?;
            } else if let Some(skip) = &self.skip {
                self.moe_front(w, layer, t)?;
                let ranks = self.moe_skip(w, index, layer, t, cap)?;
                self.reduce(skip.pointers(), w, ranks, t)?;
            } else {
                let (transport, runtime) = experts.as_mut()
                    .with_context(|| format!("layer {index} is an MoE layer; pass Spark peers for its routed experts"))?;
                self.moe(w, index, layer, t, cap, tables.decode, transport, runtime)?;
            }
            // h += ffn; x = next input_layernorm(h) (or the final norm). Under a head split
            // the FFN output is this GPU's partial plus rank 1's (dense), or the routed sum
            // with this GPU's shared-expert half plus rank 1's half (sent to rank 1 first).
            let weight = match layers.get(index + 1) {
                Some(next) => next.ptr("input_norm")?,
                None => self.weights.norm.buffer.ptr,
            };
            if let Some(w1) = w1 {
                if !layer.dense && !self.tp2_layer(index) && index + 1 < layers.len() {
                    self.exchange()?.push(0, ffn, w.delta.buffer.ptr, bytes)?;
                }
                self.peer_post(index, 0, w1, rows, bytes)?;
                self.exchange()?.wait(0, ffn)?;
                if self.tp2_layer(index) { self.tp2_norm(0, w, weight, index, 0, rows)?; }
                else { self.norm_full(0, w, weight, 2, rows, w.delta.buffer.ptr, self.recv(0, ffn)?)?; }
            } else {
                self.run("glm_norm", &[("residual", w.h.buffer.ptr), ("delta0", w.delta.buffer.ptr),
                    ("delta1", w.delta.buffer.ptr), ("weight", weight), ("out", w.x.buffer.ptr)],
                    &[rows, Scalar::I32(1)])?;
            }
            if let Some(drafter) = &self.drafter {
                let n = t.min(crate::shared::speculation::dflash::TAP_ROWS);
                drafter.tap(index, w.h.buffer.ptr, t - n, n)?;
            }
            if let Some(on_layer) = on_layer.as_mut() {
                on_layer(index, &self.download(&w.h, t * h * 2)?)?;
            }
            if let Some(probe) = self.draft_probe.borrow_mut().as_mut() {
                // SAFETY: the engine stream owns this round and its layer events.
                unsafe { probe.mark(index, self.stream)? };
            }
            crate::shared::console::layer_mark(index);
        }
        if let Some(pair) = &self.tp2 { pair.finish_checks()?; }
        if layers.len() < self.cfg.layers || logit_rows == 0 {
            // SAFETY: the engine owns this stream.
            unsafe { self.library.cuda_stream_synchronize(self.stream)? };
            return Ok(None);
        }
        self.launch_head(w, t, logit_rows)?;
        Ok(Some(self.device_logits(w, logit_rows, false)))
    }

    /// Logits of the last `n` of `t` rows of the final norm's output into the workspace's logits,
    /// through the shared BF16 target-head route ([`crate::shared::target_head`]).
    fn launch_head(&self, w: &Workspace<'_>, t: usize, n: usize) -> Result<()> {
        let h = self.cfg.hidden;
        // SAFETY: the final norm's output and the head operands are live buffers of these shapes.
        unsafe {
            let x = w.x.buffer.ptr.cast::<u8>().add((t - n) * h * 2).cast::<std::ffi::c_void>();
            match &self.weights.head {
                super::weights::GlmHead::Bf16(head) => crate::shared::target_head::launch(
                    crate::shared::target_head::TargetHead::configured(), w.head.as_ref().context("LM head")?,
                    x, head.buffer.ptr.cast_const(), w.logits.buffer.ptr.cast(), n, self.stream),
                super::weights::GlmHead::RowFp8(owner) => owner.head_logits(x, w.logits.buffer.ptr.cast(), n, self.stream),
            }
        }
    }

    /// Writes a prefill unit's tables into `w` on `rank`'s stream, ordered after
    /// the work already queued there (a workspace slot reused mid-call).
    fn put_tables_ordered(&self, rank: usize, w: &Workspace<'_>, tables: &StepTables) -> Result<()> {
        let stream = self.stream_of(rank);
        let put = |dev: &Dev<'_>, bytes: &[u8]| -> Result<()> {
            ensure!(bytes.len() <= dev.buffer.bytes, "table exceeds its buffer");
            // SAFETY: a live buffer of this rank; ordered after earlier readers on its stream.
            // The whole buffer: its ordered staging is sized once, never regrown mid-call.
            self.on(rank, || unsafe { self.library.copy_h2d_ordered(dev.buffer, bytes, stream) })
        };
        put(&w.positions, bytes_of(&tables.positions))?;
        put(&w.slots, bytes_of(&tables.slots))?;
        put(&w.page_table, bytes_of(&tables.page_table))?;
        put(&w.cache_lengths, bytes_of(&tables.cache_lengths))?;
        put(&w.lengths, bytes_of(&tables.lengths))
    }

    /// Writes a step's tables into `w` (rank 0's).
    fn put_tables(&self, w: &Workspace<'_>, tables: &StepTables) -> Result<()> {
        self.put(&w.positions, &tables.positions)?;
        self.put(&w.slots, &tables.slots)?;
        self.put(&w.page_table, &tables.page_table)?;
        self.put(&w.cache_lengths, &tables.cache_lengths)?;
        self.put(&w.lengths, &tables.lengths)
    }


    /// Rank 1's workspaces (the decode one, or `lanes` prefill lanes), created on first use.
    fn peer_workspaces(&self, decode: bool, lanes: usize)
        -> Result<Option<PeerWorkspaces<'_, 'a>>> {
        let Some(peer) = &self.peer else { return Ok(None) };
        if decode {
            if peer.decode_workspace.borrow().is_none() {
                *peer.decode_workspace.borrow_mut() = Some(self.workspace_on(1, DECODE_ROWS, true)?);
            }
            return Ok(Some(PeerWorkspaces::Decode(peer.decode_workspace.borrow())));
        }
        {
            let mut slots = peer.lane_workspaces.borrow_mut();
            while slots.len() < lanes {
                slots.push(self.workspace_on(1, self.prefill_rows, false)?);
            }
        }
        Ok(Some(PeerWorkspaces::Lanes(peer.lane_workspaces.borrow())))
    }

    /// Rank 1's attention half of unit (`index`, `lane`): (layer 0: the embedded
    /// rows and input norm), its heads' attention, the attention all-reduce and
    /// post-attention norm. Both ranks queue this front before either queues FFN work.
    #[allow(clippy::too_many_arguments)]
    fn peer_attention(&self, index: usize, lane: usize, w1: &Workspace<'_>, rows: Scalar, cap: &str,
        tables: &[StepTables; 2], bytes: usize) -> Result<()> {
        let peer = self.peer()?;
        let (layer, exchange) = (&peer.layers[index], self.exchange()?);
        if index == 0 {
            exchange.wait(1, DIRECT)?;
            self.norm_full(1, w1, layer.ptr("input_norm")?, 0, rows, w1.delta.buffer.ptr, w1.delta.buffer.ptr)?;
        }
        self.attention_unit(1, w1, index, layer, rows, cap, tables, lane)?;
        let attended = slot(index, false, lane);
        exchange.push(1, attended, w1.delta.buffer.ptr, bytes)?;
        exchange.wait(1, attended)?;
        self.norm_full(1, w1, layer.ptr("post_norm")?, 2, rows, w1.delta.buffer.ptr, self.recv(1, attended)?)
    }

    fn peer_ffn(&self, index: usize, lane: usize, w1: &Workspace<'_>, rows: Scalar,
        cap: &str, bytes: usize) -> Result<()> {
        let peer = self.peer()?;
        let (layer, exchange) = (&peer.layers[index], self.exchange()?);
        let ffn = slot(index, true, lane);
        if layer.dense {
            self.ffn_on(1, w1, layer, self.cfg.dense_intermediate, cap, w1.delta.buffer.ptr, rows)?;
            exchange.push(1, ffn, w1.delta.buffer.ptr, bytes)
        } else {
            self.shared_on(1, index, w1, layer, rows, cap)?;
            if self.tp2_layer(index) { Ok(()) } else { exchange.push(1, ffn, w1.shared.buffer.ptr, bytes) }
        }
    }

    /// Rank 1's FFN half of unit (`index`, `lane`): rank 0's dense partial or
    /// routed + shared sum in, then the next layer's input norm (nothing after
    /// the last layer). The norm adds the same two operands in the same order as
    /// rank 0's.
    fn peer_post(&self, index: usize, lane: usize, w1: &Workspace<'_>, rows: Scalar, bytes: usize) -> Result<()> {
        let peer = self.peer()?;
        let Some(next) = peer.layers.get(index + 1) else { return Ok(()) };
        let ffn = slot(index, true, lane);
        self.exchange()?.wait(1, ffn)?;
        let received = self.recv(1, ffn)?;
        let (first, second) = if peer.layers[index].dense { (w1.delta.buffer.ptr, received) }
            else { (received, w1.shared.buffer.ptr) };
        if self.tp2_layer(index) {
            self.tp2_norm(1, w1, next.ptr("input_norm")?, index, lane, rows)
        } else { self.norm_full(1, w1, next.ptr("input_norm")?, 2, rows, first, second) }
    }

    /// `bytes` at `offset` inside `dev`.
    fn region(dev: &Dev<'_>, offset: usize, bytes: usize) -> cuteafd_ffi::CuteafdDeviceBuffer {
        debug_assert!(offset + bytes <= dev.buffer.bytes);
        cuteafd_ffi::CuteafdDeviceBuffer {
            // SAFETY: callers pass offsets inside the buffer.
            ptr: unsafe { dev.buffer.ptr.cast::<u8>().add(offset) }.cast(),
            bytes,
            ..dev.buffer
        }
    }

    /// Greedy tokens of the first `rows` logits rows into `select`.
    fn select_greedy(&self, w: &Workspace<'_>, rows: usize) -> Result<()> {
        let vocab = self.cfg.vocab_size;
        // SAFETY: the logits rows and the select buffer (ids, then statuses) are live buffers of these shapes.
        unsafe {
            self.library.cuda_logits_greedy_f32_async(w.logits.buffer.ptr, rows, vocab, vocab, w.select.buffer.ptr,
                std::ptr::null_mut(), Self::region(&w.select, SELECT_STATUS_OFFSET, rows * 4).ptr, self.stream)
        }
    }

    /// The first `rows` logits rows as device logits (`greedy`: with the rows'
    /// selection from [`Self::select_greedy`]).
    fn device_logits(&self, w: &Workspace<'_>, rows: usize, greedy: bool) -> DeviceLogits {
        let vocab = self.cfg.vocab_size;
        DeviceLogits { ptr: w.logits.buffer.ptr, rows, vocab, stride: vocab, stream: self.stream,
            greedy: greedy.then(|| (w.select.buffer.ptr.cast_const(), Self::region(&w.select, SELECT_STATUS_OFFSET, rows * 4).ptr
                .cast_const())) }
    }

    /// Decode layers as captured graph segments: each replays the previous
    /// layer's reduce + residual norm, this layer's attention and
    /// post-attention norm, and the dense FFN or the MoE front (router,
    /// select, wire rows); the Spark exchange runs between segments.
    ///
    /// Under a head split each segment exchanges with rank 1's segment of the
    /// same layer (captured on rank 1's stream); rank 1's next segment is
    /// queued before the host waits in this layer's Spark exchange.
    #[allow(clippy::too_many_arguments)]
    fn decode_layers(&self, w: &Workspace<'_>, w1: Option<&Workspace<'_>>, ranked: &[StepTables; 2], t: usize,
        experts: &mut Option<(&mut SparkLink<'_>, &tokio::runtime::Runtime)>, gather: bool, head: bool) -> Result<()> {
        let tables = &ranked[0];
        let rows = Scalar::I32(t as i32);
        let layers = &self.weights.layers;
        let bytes = t * self.cfg.hidden * 2;
        // The decode transport's intake planes (or the skip intake's); the
        // replayed graphs bake them in.
        let planes = match (experts.as_ref(), &self.skip) {
            (_, Some(skip)) => skip.pointers(),
            (Some((transport, _)), None) => transport.intake.pointers(),
            (None, None) => [std::ptr::null(); MAX_RANKS],
        };
        // Previous layer's FFN output: none (first layer), in `delta`, or Spark planes.
        let mut previous = Previous::First;
        for index in 0..=layers.len() {
            let layer = layers.get(index);
            let segment = || -> Result<()> {
                if index == 0 && gather {
                    // SAFETY: the step's ids are in `ids` before the replay; `h` holds its rows.
                    unsafe { self.embedding.gather(w.ids.buffer.ptr, std::ptr::null(), t, 1, std::ptr::null(),
                        w.h.buffer.ptr, self.stream)? };
                }
                if let (0, Some(w1)) = (index, w1) {
                    self.exchange()?.push_to(0, DIRECT, w.h.buffer.ptr, w1.h.buffer.ptr, bytes)?;
                }
                if let Previous::Planes(ranks) = previous {
                    self.reduce(planes, w, ranks, t)?;
                }
                let weight = match layer {
                    Some(layer) => layer.ptr("input_norm")?,
                    None => self.weights.norm.buffer.ptr,
                };
                match (w1, previous, index.checked_sub(1)) {
                    // Rank 1's partial (dense) or shared-expert half in; this GPU's routed +
                    // shared sum out first (rank 1 needs it; not after the last layer).
                    (Some(_), Previous::Delta | Previous::Planes(_), Some(before)) => {
                        let ffn = slot(before, true, 0);
                        if matches!(previous, Previous::Planes(_)) && layer.is_some() {
                            self.exchange()?.push(0, ffn, w.delta.buffer.ptr, bytes)?;
                        }
                        self.exchange()?.wait(0, ffn)?;
                        if self.tp2_layer(before) { self.tp2_norm(0, w, weight, before, 0, rows)?; }
                        else { self.norm_full(0, w, weight, 2, rows, w.delta.buffer.ptr, self.recv(0, ffn)?)?; }
                    }
                    _ => {
                        let deltas = if matches!(previous, Previous::First) { 0 } else { 1 };
                        self.run("glm_norm", &[("residual", w.h.buffer.ptr), ("delta0", w.delta.buffer.ptr),
                            ("delta1", w.delta.buffer.ptr), ("weight", weight), ("out", w.x.buffer.ptr)],
                            &[rows, Scalar::I32(deltas)])?;
                    }
                }
                // `h` now holds the previous layer's output.
                if let (Some(drafter), Some(previous)) = (&self.drafter, index.checked_sub(1)) {
                    drafter.tap(previous, w.h.buffer.ptr, 0, t)?;
                }
                let Some(layer) = layer else {
                    if head {
                        self.launch_head(w, t, t)?;
                        self.select_greedy(w, t)?;
                    }
                    return Ok(());
                };
                self.attention_unit(0, w, index, layer, rows, "m64", ranked, 0)?;
                if w1.is_some() {
                    let attended = slot(index, false, 0);
                    self.exchange()?.push(0, attended, w.delta.buffer.ptr, bytes)?;
                    self.exchange()?.wait(0, attended)?;
                    self.norm_full(0, w, layer.ptr("post_norm")?, 2, rows, w.delta.buffer.ptr, self.recv(0, attended)?)?;
                } else {
                    self.run("glm_norm", &[("residual", w.h.buffer.ptr), ("delta0", w.delta.buffer.ptr),
                        ("delta1", w.delta.buffer.ptr), ("weight", layer.ptr("post_norm")?), ("out", w.x.buffer.ptr)],
                        &[rows, Scalar::I32(1)])?;
                }
                if layer.dense {
                    self.ffn(w, layer, self.cfg.dense_intermediate, "m64", w.delta.buffer.ptr, rows)?;
                    if w1.is_some() {
                        self.exchange()?.push(0, slot(index, true, 0), w.delta.buffer.ptr, bytes)?;
                    }
                    Ok(())
                } else {
                    self.moe_front(w, layer, t)
                }
            };
            let key = GraphKey { layer: index, rows: t, table_width: tables.table_width,
                table_stride: tables.table_stride, previous, head: head && layer.is_none() };
            self.replay_on(0, key, segment)?;
            if let Some(w1) = w1 {
                // Rank 1's segments: layer 0 with rank 0's first, then each next one before
                // the host waits in this layer's exchange.
                if index == 0 {
                    self.peer_segment(0, Previous::First, w1, t, ranked, bytes)?;
                }
                if let Some(_layer) = layer.filter(|_| self.tp2_layer(index)) {
                    self.tp2_pair(index, 0, w, w1, t, "m64")?;
                }
                if let Some(layer) = layer.filter(|_| index + 1 < layers.len()) {
                    let kind = if layer.dense || self.tp2_layer(index) { Previous::Delta }
                        else { Previous::Planes(0) };
                    // Rank 1's next weights into L2 while rank 0 waits in this layer's exchange.
                    if let (false, Some(l2)) = (layer.dense, self.peer()?.l2.as_ref()) {
                        self.on(1, || l2.issue(self.library, index, self.stream_of(1)))?;
                    }
                    self.peer_segment(index + 1, kind, w1, t, ranked, bytes)?;
                }
            }
            previous = match layer {
                None => break,
                Some(layer) if layer.dense || self.tp2_layer(index) => Previous::Delta,
                // Only the step's real rows go to the Sparks; padded rows reduce stale
                // planes into their own (scratch) rows.
                Some(layer) if self.skip.is_some() => Previous::Planes(self.moe_skip(w, index, layer,
                    tables.exchange_rows, "m64")?),
                Some(layer) => {
                    let (transport, runtime) = experts.as_mut()
                        .with_context(|| format!("layer {index} is an MoE layer; pass Spark peers for its routed experts"))?;
                    let ranks = self.moe_exchange(w, index, layer, tables.exchange_rows, "m64", true, transport,
                        runtime)?;
                    Previous::Planes(ranks)
                }
            };
            if let Some(probe) = self.draft_probe.borrow_mut().as_mut() {
                // SAFETY: the engine stream owns this round and its layer events.
                unsafe { probe.mark(index, self.stream)? };
            }
            crate::shared::console::layer_mark(index);
        }
        Ok(())
    }

    /// Rank 1's decode segment of layer `index` (see [`Self::decode_layers`]):
    /// the previous layer's FFN exchange (rank 0's dense partial or routed +
    /// shared sum in) and this layer's input norm, its heads' attention and the
    /// attention all-reduce, then its dense partial or shared-expert half out.
    fn peer_segment(&self, index: usize, previous: Previous, w1: &Workspace<'_>, t: usize, ranked: &[StepTables; 2],
        bytes: usize) -> Result<()> {
        let tables = &ranked[1];
        let peer = self.peer()?;
        let rows = Scalar::I32(t as i32);
        let segment = || -> Result<()> {
            let exchange = self.exchange()?;
            let layer = &peer.layers[index];
            match (previous, index.checked_sub(1)) {
                (Previous::Delta | Previous::Planes(_), Some(before)) => {
                    let ffn = slot(before, true, 0);
                    exchange.wait(1, ffn)?;
                    let received = self.recv(1, ffn)?;
                    let (first, second) = if matches!(previous, Previous::Delta) { (w1.delta.buffer.ptr, received) }
                        else { (received, w1.shared.buffer.ptr) };
                    if self.tp2_layer(before) { self.tp2_norm(1, w1, layer.ptr("input_norm")?, before, 0, rows)?; }
                    else { self.norm_full(1, w1, layer.ptr("input_norm")?, 2, rows, first, second)?; }
                }
                _ => {
                    exchange.wait(1, DIRECT)?;
                    self.norm_full(1, w1, layer.ptr("input_norm")?, 0, rows, w1.delta.buffer.ptr, w1.delta.buffer.ptr)?;
                }
            }
            self.attention_unit(1, w1, index, layer, rows, "m64", ranked, 0)?;
            let (attended, ffn) = (slot(index, false, 0), slot(index, true, 0));
            exchange.push(1, attended, w1.delta.buffer.ptr, bytes)?;
            exchange.wait(1, attended)?;
            self.norm_full(1, w1, layer.ptr("post_norm")?, 2, rows, w1.delta.buffer.ptr, self.recv(1, attended)?)?;
            let out = if layer.dense { w1.delta.buffer.ptr } else { w1.shared.buffer.ptr };
            let intermediate = if layer.dense { self.cfg.dense_intermediate } else { self.cfg.moe_intermediate };
            if layer.dense { self.ffn_on(1, w1, layer, intermediate, "m64", out, rows)?; }
            else { self.shared_on(1, index, w1, layer, rows, "m64")?; }
            if self.tp2_layer(index) { Ok(()) } else { exchange.push(1, ffn, out, bytes) }
        };
        let key = GraphKey { layer: index, rows: t, table_width: tables.table_width, table_stride: tables.table_stride,
            previous, head: false };
        self.replay_on(1, key, segment)
    }

    /// Launches `segment` through a graph captured the first time `key` is seen.
    fn replay(&self, key: GraphKey, segment: impl Fn() -> Result<()>) -> Result<()> {
        self.replay_on(0, key, segment)
    }

    /// [`Self::replay`] on rank `rank`'s stream (its own graphs).
    fn replay_on(&self, rank: usize, key: GraphKey, segment: impl Fn() -> Result<()>) -> Result<()> {
        self.require_live()?;
        let graphs = match (rank, &self.peer) {
            (1, Some(peer)) => &peer.graphs,
            _ => &self.graphs,
        };
        let stream = self.stream_of(rank);
        if let Some(graph) = graphs.borrow().get(&key) {
            // SAFETY: the graph's pointers are persistent engine buffers of that rank.
            return self.on(rank, || unsafe { self.library.cuda_graph_launch(graph.0, stream) });
        }
        if self.graphs_warmed.get() {
            let misses = self.graph_misses.get().saturating_add(1);
            self.graph_misses.set(misses);
            tracing::warn!(rank, ?key, graph_misses = misses,
                "GLM startup graph coverage gap; running segment uncaptured");
            return segment();
        }
        // SAFETY: capture records launches on that rank's stream; nothing in the
        // segment synchronizes the host.
        self.on(rank, || unsafe { self.library.cuda_graph_begin_capture(stream) })?;
        let captured = segment();
        let exec = self.on(rank, || unsafe { self.library.cuda_graph_end_capture(stream) });
        captured?;
        let exec = match exec {
            Ok(exec) => exec,
            Err(error) if !self.graphs_warming.get() => {
                let misses = self.graph_misses.get().saturating_add(1);
                self.graph_misses.set(misses);
                tracing::warn!(rank, ?key, graph_misses = misses, %error,
                    "GLM decode graph capture failed; running segment uncaptured");
                return segment();
            }
            Err(error) => return Err(error),
        };
        let graph = GraphExec(exec, self.library);
        self.on(rank, || unsafe { self.library.cuda_graph_launch(graph.0, stream) })?;
        graphs.borrow_mut().insert(key, graph);
        Ok(())
    }

    /// Per layer, the weights a decode step reads after its routed experts
    /// are out, in read order: the next layer's attention (E4M3 weights and
    /// their scales), post-attention norm, router and shared expert (or dense
    /// MLP); after the last layer the final norm and head.
    pub fn decode_read_order(&self) -> Vec<Vec<crate::shared::l2_prefetch::Range>> {
        self.decode_read_order_on(0)
    }

    /// Rank 1's L2 prefetch (its shares of the next layer's weights) with `budget`
    /// bytes per layer, issued between its decode segments.
    pub fn attach_peer_l2(&mut self, budget: usize) -> Result<()> {
        let order = self.decode_read_order_on(1);
        let l2 = self.on(1, || crate::shared::l2_prefetch::L2Prefetch::new(self.library, budget, &order))?;
        self.peer.as_mut().context("no head-split peer")?.l2 = Some(l2);
        Ok(())
    }

    /// [`Self::decode_read_order`] of rank `rank`'s shares (rank 1 reads no head).
    fn decode_read_order_on(&self, rank: usize) -> Vec<Vec<crate::shared::l2_prefetch::Range>> {
        let layers = match (rank, &self.peer) {
            (1, Some(peer)) => &peer.layers,
            _ => &self.weights.layers,
        };
        (0..layers.len()).map(|i| match layers.get(i + 1) {
            // `operands` takes a weight's E4M3 copy and scales, else its BF16 operand.
            Some(next) => crate::shared::l2_prefetch::operands(&["input_norm", "w_qkv_a", "q_a_norm", "kv_a_norm",
                "w_q_b", "w_iq", "w_ik", "k_norm_w", "k_norm_b", "w_uk", "w_uv", "w_o", "post_norm", "gate",
                "gate.bias", "w_gate_up", "w_down"],
                |n| next.range(n)),
            None if rank == 1 => Vec::new(),
            None => vec![(self.weights.norm.buffer.ptr.cast_const(), self.weights.norm.buffer.bytes),
                self.weights.head.range()],
        }).collect()
    }

    /// After layer `index`'s shared expert is queued in a one-lane decode or
    /// verify step: the L2 prefetch of what the step reads next.
    fn prefetch(&self, index: usize) -> Result<()> {
        match &self.l2 {
            Some(l2) => l2.issue(self.library, index, self.stream),
            None => Ok(()),
        }
    }

    fn shared_on(&self, rank: usize, index: usize, w: &Workspace<'_>, layer: &GlmLayer<'_>,
        rows: Scalar, cap: &str) -> Result<()> {
        if let Some(native) = self.native_experts.get(rank) {
            let Scalar::I32(rows) = rows else { anyhow::bail!("EXLR8 shared rows must be I32"); };
            self.on(rank, || native.shared(self.library, index, w, rows as usize, self.stream_of(rank)))
        } else {
            self.ffn_on(rank, w, layer, self.cfg.moe_intermediate, cap, w.shared.buffer.ptr, rows)
        }
    }

    /// Router, shared expert and the Spark routed experts; leaves
    /// routed + shared in `delta` for the next norm's residual add.
    #[allow(clippy::too_many_arguments)]
    #[allow(clippy::too_many_arguments)]
    fn moe(&self, w: &Workspace<'_>, index: usize, layer: &GlmLayer<'_>, t: usize, cap: &str, decode: bool,
        transport: &mut SparkLink<'_>, runtime: &tokio::runtime::Runtime) -> Result<()> {
        self.moe_front(w, layer, t)?;
        let ranks = self.moe_exchange(w, index, layer, t, cap, decode, transport, runtime)?;
        self.reduce(transport.intake.pointers(), w, ranks, t)
    }

    /// Router scores, the sigmoid top-k selection and the wire rows (device only).
    fn moe_front(&self, w: &Workspace<'_>, layer: &GlmLayer<'_>, t: usize) -> Result<()> {
        self.moe_front_on(0, w, layer, t)
    }

    fn moe_front_on(&self, rank: usize, w: &Workspace<'_>, layer: &GlmLayer<'_>, t: usize) -> Result<()> {
        let (h, topk) = (self.cfg.hidden, self.cfg.topk);
        let rows = Scalar::I32(t as i32);
        self.run_on(rank, "glm_router_scores", &[("x", w.x.buffer.ptr), ("w", layer.ptr("gate")?),
            ("logits", w.router_logits.buffer.ptr)], &[rows])?;
        // SAFETY: logits, bias and route outputs are live buffers of `t` rows.
        self.on(rank, || unsafe {
            self.library.router_select(w.router_logits.buffer.ptr, layer.ptr("gate.bias")?, std::ptr::null(),
                std::ptr::null(), w.route_ids.buffer.ptr, w.route_weights.buffer.ptr, t, self.cfg.experts, topk,
                self.cfg.routed_scale as f32, true, self.stream_of(rank))
        })?;
        if self.spark_bf16 && !self.tp2.as_ref().is_some_and(|p| p.partial() == PartialDtype::F32) { return Ok(()); }
        let grid = self.quantize_grid.blocks(t, h);
        self.run_on(rank, "glm_expert_input_quant", &[("source_ptr", w.x.buffer.ptr), ("values_ptr", w.wire.buffer.ptr),
            // SAFETY: the scale rows follow the payload inside each wire row.
            ("scale_rows_ptr", unsafe { w.wire.buffer.ptr.cast::<u8>().add(h) }.cast()),
            ("scale_mma_ptr", w.delta.buffer.ptr)], &[rows, Scalar::I32(grid as i32)])
    }

    /// Routes and wire rows down, the shared expert, the Spark exchange and
    /// the rank planes' uploads; returns the rank count the reduce needs.
    /// The next exchange's download sync completes the uploads before the
    /// staging is rewritten.
    #[allow(clippy::too_many_arguments)]
    fn moe_exchange(&self, w: &Workspace<'_>, index: usize, layer: &GlmLayer<'_>, t: usize, cap: &str, decode: bool,
        transport: &mut SparkLink<'_>, runtime: &tokio::runtime::Runtime) -> Result<usize> {
        self.moe_stage(w, index, layer, t, cap)?;
        if decode {
            self.prefetch(index)?;
        }
        let wave = self.moe_send(w, index, t, decode, transport)?;
        runtime.block_on(self.moe_land(t, transport, wave))
    }

    /// [`Self::moe_exchange`] without the Spark request (benchmarks): the
    /// stage, then zero partials of [`SKIP_RANKS`] ranks uploaded through
    /// the skip intake's pinned staging (a host-mode wave's cost).
    /// With CUTEAFD_EMULATE_EXCHANGE_US a decode step waits that long after
    /// the shared expert (and the L2 prefetch) as a Spark exchange would.
    fn moe_skip(&self, w: &Workspace<'_>, index: usize, layer: &GlmLayer<'_>, t: usize, cap: &str) -> Result<usize> {
        let skip = self.skip.as_ref().context("routed experts are not skipped")?;
        self.moe_stage(w, index, layer, t, cap)?;
        if cap == "m64" {
            let mark = crate::shared::l2_prefetch::exchange_mark(self.library, self.stream)?;
            self.prefetch(index)?;
            crate::shared::l2_prefetch::exchange_wait(self.library, mark)?;
        }
        skip.upload_zeros(t, self.stream)?;
        Ok(SKIP_RANKS)
    }

    /// Routes and wire rows down to this workspace's pinned staging (the host
    /// waits for them), then the shared expert queued on the GPU; send the
    /// request with [`Self::moe_send`].
    fn moe_stage(&self, w: &Workspace<'_>, layer_index: usize, layer: &GlmLayer<'_>, t: usize, cap: &str) -> Result<()> {
        let (h, topk) = (self.cfg.hidden, self.cfg.topk);
        let layout = crate::shared::experts::staging::RouteStaging::new(t, topk,
            if self.spark_bf16 { h * 2 } else { h + h / 32 })?;
        let route_bytes = layout.route_bytes;
        let staging = w.router_host.borrow_mut();
        let timer = std::time::Instant::now();
        // SAFETY: the pinned regions are large enough; the sync completes them
        // (and this workspace's previous plane uploads, freeing its staging).
        // PROFILE=expert-routes only: this layer-step's MoE input rows (BF16 `x`), downloaded
        // into their own pinned buffer before the existing sync. Off, `input_sample` is `None`.
        let sample = match cuteafd_transport::route_log::input_sample() {
            Some(ordinal) => {
                let rows = t.min(cuteafd_transport::route_log::SAMPLE_ROWS);
                let pinned = HostAllocation::new(self.library, rows * h * 2)?;
                // SAFETY: `x` holds this layer-step's `t` >= `rows` BF16 rows in stream order; the sync below completes it.
                unsafe { self.library.copy_d2h_host_buffer_async(pinned.buffer, w.x.buffer, rows * h * 2, self.stream)?; }
                Some((ordinal, rows, pinned))
            }
            None => None,
        };
        unsafe {
            layout.enqueue(self.library, staging.buffer,
                [w.route_ids.buffer, w.route_weights.buffer,
                    if self.spark_bf16 { w.x.buffer } else { w.wire.buffer }], self.stream)?;
            // Wait for this unit's copies only: other units' queued work keeps
            // the GPU busy while the host builds and posts the request.
            self.library.cuda_event_record(self.staged.get(), self.stream)?;
            self.library.cuda_event_synchronize(self.staged.get())?;
        }
        self.profile.borrow_mut()[0] += timer.elapsed().as_secs_f64();
        if let Some((ordinal, rows, pinned)) = sample {
            let staged = staging.bytes();
            let word = |offset: usize, i: usize| u32::from_le_bytes(staged[offset + i * 4..][..4].try_into().unwrap());
            cuteafd_transport::route_log::save_sample(cuteafd_transport::route_log::InputSample {
                ordinal, layer: layer_index as u32, wave_rows: t as u32, hidden: h as u32, topk: topk as u32,
                kind: if cap == "m64" { cuteafd_transport::ExpertV2SourceKind::Decode } else { cuteafd_transport::ExpertV2SourceKind::Prefill },
                x_bf16: pinned.bytes()[..rows * h * 2].to_vec(),
                ids: (0..rows * topk).map(|i| word(0, i) as i32).collect(),
                weights: (0..rows * topk).map(|i| f32::from_bits(word(route_bytes, i))).collect(),
                step: None,
            });
        }
        // The shared expert runs on the GPU while the host sends and the Sparks compute.
        self.shared_on(0, layer_index, w, layer, Scalar::I32(t as i32), cap)
    }

    /// A prefill unit's wave on the link's transport worker: the routes and
    /// wire rows copy into the wave slot's pinned staging on the stream, the
    /// shared expert queues behind them, and the worker builds and posts the
    /// request once the copies complete. This thread never waits for the GPU.
    fn moe_wave(&self, w: &Workspace<'_>, index: usize, layer: &GlmLayer<'_>, t: usize, cap: &str,
        transport: &mut crate::shared::spark_waves::Waves<'_, '_>) -> Result<crate::shared::spark_waves::WaveTicket> {
        let (h, topk, bf16) = (self.cfg.hidden, self.cfg.topk, self.spark_bf16);
        let layout = crate::shared::experts::staging::RouteStaging::new(t, topk, if bf16 { h * 2 } else { h + h / 32 })?;
        self.route_sample(w, index, t, cap)?;
        // SAFETY: ids, weights and wire rows hold this unit's `t` rows in
        // stream order; the copies read them before later stream work.
        let ticket = unsafe {
            transport.send(t, layout, [w.route_ids.buffer, w.route_weights.buffer,
                if bf16 { w.x.buffer } else { w.wire.buffer }], self.stream,
                Box::new(move |routes, wire| expert_request_parts(routes, wire, index, t, h, topk, false, bf16)))?
        };
        // The shared expert runs on the GPU while the worker sends and the Sparks compute.
        self.shared_on(0, index, w, layer, Scalar::I32(t as i32), cap)?;
        Ok(ticket)
    }

    /// Bytes of the widest prefill wave's pinned staging (routes and wire rows).
    fn wave_staging_bytes(&self) -> Result<usize> {
        let h = self.cfg.hidden;
        Ok(crate::shared::experts::staging::RouteStaging::new(self.prefill_rows, self.cfg.topk,
            if self.spark_bf16 { h * 2 } else { h + h / 32 })?.bytes)
    }

    /// PROFILE=expert-routes only: this layer-step's MoE input rows (BF16 `x`)
    /// into their own pinned buffer, saved once their copy completes.
    fn route_sample(&self, w: &Workspace<'_>, index: usize, t: usize, cap: &str) -> Result<()> {
        let Some(ordinal) = cuteafd_transport::route_log::input_sample() else { return Ok(()) };
        let h = self.cfg.hidden;
        let rows = t.min(cuteafd_transport::route_log::SAMPLE_ROWS);
        let pinned = HostAllocation::new(self.library, rows * h * 2)?;
        let ids = HostAllocation::new(self.library, rows * self.cfg.topk * 8)?;
        // SAFETY: `x`, ids and weights hold this layer-step's rows in stream
        // order; the event below completes the copies (diagnostic only).
        unsafe {
            self.library.copy_d2h_host_buffer_async(pinned.buffer, w.x.buffer, rows * h * 2, self.stream)?;
            let half = cuteafd_ffi::CuteafdHostBuffer { ptr: ids.buffer.ptr.cast::<u8>().add(rows * self.cfg.topk * 4).cast(),
                bytes: rows * self.cfg.topk * 4, ..ids.buffer };
            self.library.copy_d2h_host_buffer_async(ids.buffer, w.route_ids.buffer, rows * self.cfg.topk * 4, self.stream)?;
            self.library.copy_d2h_host_buffer_async(half, w.route_weights.buffer, rows * self.cfg.topk * 4, self.stream)?;
            self.library.cuda_event_record(self.staged.get(), self.stream)?;
            self.library.cuda_event_synchronize(self.staged.get())?;
        }
        let staged = ids.bytes();
        let topk = self.cfg.topk;
        let word = |offset: usize, i: usize| u32::from_le_bytes(staged[offset + i * 4..][..4].try_into().unwrap());
        cuteafd_transport::route_log::save_sample(cuteafd_transport::route_log::InputSample {
            ordinal, layer: index as u32, wave_rows: t as u32, hidden: h as u32, topk: topk as u32,
            kind: if cap == "m64" { cuteafd_transport::ExpertV2SourceKind::Decode } else { cuteafd_transport::ExpertV2SourceKind::Prefill },
            x_bf16: pinned.bytes()[..rows * h * 2].to_vec(),
            ids: (0..rows * topk).map(|i| word(0, i) as i32).collect(),
            weights: (0..rows * topk).map(|i| f32::from_bits(word(rows * topk * 4, i))).collect(),
            step: None,
        });
        Ok(())
    }

    /// One request to every Spark rank from the staged routes and wire rows
    /// ([`Self::moe_stage`]); complete it with [`Self::moe_land`].
    fn moe_send(&self, w: &Workspace<'_>, index: usize, t: usize, decode: bool, transport: &mut SparkLink<'_>)
        -> Result<SparkExpertWave> {
        let timer = std::time::Instant::now();
        let request = {
            let staging = w.router_host.borrow();
            if let Some(probe) = self.draft_probe.borrow_mut().as_mut().filter(|p| p.active()) {
                probe.staged(index, &staging.bytes()[..t * self.cfg.topk * 4]);
            }
            expert_request(staging.bytes(), index, t, self.cfg.hidden, self.cfg.topk, decode, self.spark_bf16)?
        };
        self.exchange_host.borrow_mut()[2] += timer.elapsed().as_secs_f64();
        let post = std::time::Instant::now();
        let wave = transport.dispatch(&request)?;
        self.exchange_host.borrow_mut()[3] += post.elapsed().as_secs_f64();
        self.exchange_host.borrow_mut()[0] += timer.elapsed().as_secs_f64();
        Ok(wave)
    }

    /// Receives `wave`'s BF16 rank partials into the transport's intake
    /// planes; returns the rank count the reduce needs.
    async fn moe_land(&self, t: usize, transport: &mut SparkLink<'_>, wave: SparkExpertWave) -> Result<usize> {
        let ranks = transport.world_size();
        ensure!(ranks <= MAX_RANKS, "{ranks} Spark ranks exceed the reduction planes");
        let timer = std::time::Instant::now();
        transport.receive(wave, t, self.stream).await?;
        self.profile.borrow_mut()[1] += timer.elapsed().as_secs_f64();
        Ok(ranks)
    }

    /// Pipelined Spark prefill of consecutive-row lanes of one sequence, each
    /// with its own workspace and transport thread. Units (layer, lane) run
    /// layer-major, so a lane's attention follows its own previous layer and
    /// the earlier lanes' same layer in stream order: later lanes' MLA and
    /// DSA index top-k read earlier lanes' latent and index records from the
    /// caches, and each lane's shared-indexer layers reuse the `indices` its
    /// own workspace kept from its last full layer. A lane's request is built,
    /// posted and received on its thread while this thread queues the other
    /// lanes' GPU layers. Returns the last `logit_rows` rows' logits.
    fn step_lanes(&self, lanes: &[StepTables], chunks: &[&[u32]], logit_rows: usize,
        link: &mut SparkLink<'_>, runtime: &tokio::runtime::Runtime) -> Result<Option<StepLogits>> {
        crate::shared::experts::rtx::pair::submit_scoped(
            || self.require_live(), || {
                crate::shared::decode_graph::fatal_drain(self.retire_tp2(), "GLM submission shutdown");
                Ok(())
            },
            |scope| self.step_lanes_inner(scope, lanes, chunks, logit_rows, link, runtime))
    }

    fn step_lanes_inner(&self, scope: &crate::shared::experts::rtx::pair::SubmissionScope<'_>, lanes: &[StepTables],
        chunks: &[&[u32]], logit_rows: usize, link: &mut SparkLink<'_>, runtime: &tokio::runtime::Runtime)
        -> Result<Option<StepLogits>> {
        // `lanes` are the call's units in order; at most `waves` are in flight,
        // unit `u` holding workspace slot `u % waves` on both ranks.
        let waves = self.prefill_lanes.min(lanes.len()).max(1);
        ensure!(waves <= link.waves(), "{waves} prefill units in flight exceed the link's {} waves", link.waves());
        {
            let mut slots = self.lane_workspaces.borrow_mut();
            while slots.len() < waves {
                slots.push(self.workspace(self.prefill_rows, false)?);
            }
        }
        let owned = self.lane_workspaces.borrow();
        let workspaces: Vec<&Workspace<'_>> = owned.iter().take(waves).collect();
        let counts: Vec<usize> = lanes.iter().map(|l| l.positions.len()).collect();
        let starts: Vec<usize> = counts.iter().scan(0, |first, &n| {
            let here = *first;
            *first += n;
            Some(here)
        }).collect();
        let total: usize = counts.iter().sum();
        ensure!(logit_rows <= counts.last().copied().unwrap_or(0) && (self.full_prefill_logits || logit_rows <= DECODE_ROWS),
            "prefill logits must lie in the last unit and stay within {DECODE_ROWS} rows without full_prefill_logits");
        ensure!(counts.iter().all(|&n| n <= self.prefill_rows) && chunks.iter().zip(&counts).all(|(c, &n)| c.len() == n),
            "prefill unit exceeds its workspace");
        let peer_workspaces = self.peer_workspaces(false, waves)?;
        let split = peer_workspaces.is_some();
        // Both streams drain once per call: the units' tables are then queued in
        // stream order as each unit starts ([`UnitHooks::begin_unit`]).
        for rank in 0..self.ranks() {
            // SAFETY: the engine owns both streams; nothing queued reads the tables written below.
            self.on(rank, || unsafe { self.library.cuda_stream_synchronize(self.stream_of(rank)) })?;
        }
        let pages = lanes.iter().map(placement_pages).collect::<Vec<_>>();
        let mut ranked = Vec::with_capacity(lanes.len());
        for tables in lanes {
            ranked.push(if self.context.is_some() {
                [self.rank_tables(tables, 0)?, self.rank_tables(tables, 1)?]
            } else { [tables.clone(), tables.clone()] });
        }
        *self.context_units.borrow_mut() = vec![[context::ContextUnit::default(); 2]; lanes.len()];
        if self.layer_execution.is_some() {
            ensure!(lanes.len() <= waves, "owned-attention prefill takes one unit per slot");
            if let Some(pair) = &self.tp2 { pair.begin_checks(&counts, self.force_route_check.get())?; }
            for (unit, w) in workspaces.iter().enumerate() {
                self.put_tables(w, &ranked[unit][0])?;
                if let Some(peers) = &peer_workspaces { self.on(1, || self.put_tables(peers.get(unit)?, &ranked[unit][1]))?; }
                self.embedding.embed(chunks[unit], w.ids.buffer, 1, w.h.buffer, self.stream)?;
            }
            return self.layer_prefill(scope, &workspaces[..lanes.len()],
                peer_workspaces.as_ref().context("owned attention lanes")?, &ranked, logit_rows, link, runtime);
        }
        let layers = &self.weights.layers;
        let cap = "m4096";
        let rows_of = |unit: usize| Scalar::I32(counts[unit] as i32);
        let bytes_of_unit = |unit: usize| counts[unit] * self.cfg.hidden * 2;
        // The drafter taps the call's last TAP_ROWS rows: each unit's part of
        // that window, at its offset among the tap rows.
        let window = total - total.min(crate::shared::speculation::dflash::TAP_ROWS);
        *self.staging_owner.borrow_mut() = [None; 2];
        // Per slot, rank 1's progress when a later unit reuses the slot: its peer
        // workspace is rewritten only after the earlier unit's reads there. Every
        // rank-1 wait already queued has its rank-0 push queued earlier on rank 0's
        // stream, so this cross-GPU wait cannot close a cycle.
        let peer_released: Vec<Option<crate::shared::memory::device::Event<'_>>> = match &self.peer {
            Some(peer) => (0..waves).map(|_| crate::shared::memory::device::Event::new(
                crate::shared::memory::device::Device { library: self.library, id: peer.device }).map(Some))
                .collect::<Result<_>>()?,
            None => (0..waves).map(|_| None).collect(),
        };
        let begin = |unit: usize| -> Result<()> {
            let slot = unit % waves;
            let w = workspaces[slot];
            if unit >= waves {
                // The slot's previous unit committed on rank 0 (stream order). Rank 1's
                // queued work for it must finish before its peer workspace is rewritten.
                if let (Some(event), Some(peer)) = (&peer_released[slot], &self.peer) {
                    // SAFETY: rank 1's stream and this live event of its device.
                    self.on(1, || unsafe { self.library.cuda_event_record(event.raw, peer.stream) })?;
                    // SAFETY: rank 0's stream waits on the retained rank-1 event.
                    unsafe { self.library.cuda_stream_wait_event(self.stream, event.raw)? };
                }
            }
            let ranked = &ranked[unit];
            if unit >= waves { self.put_tables_ordered(0, w, &ranked[0])?; } else { self.put_tables(w, &ranked[0])?; }
            if let Some(peers) = &peer_workspaces {
                let w1 = peers.get(slot)?;
                if unit >= waves {
                    // Rank 1's tables after its previous unit's reads (it waits on rank 0's push below).
                    self.put_tables_ordered(1, w1, &ranked[1])?;
                } else { self.on(1, || self.put_tables(w1, &ranked[1]))?; }
            }
            if self.context.is_some() {
                let start = lanes[unit].positions.first().map_or(0, |&p| p as usize);
                let mut units = self.context_units.borrow_mut();
                for rank in 0..2 {
                    units[unit][rank] = self.context_prefill_tables_on(rank, slot, start, start,
                        counts[unit], &pages[unit], unit >= waves)?;
                }
            }
            self.embedding.embed(chunks[unit], w.ids.buffer, 1, w.h.buffer, self.stream)?;
            // The rank-1 entry copy is published in this unit's first attention,
            // next to its DIRECT wait: Begin admits several slots before any attention.
            Ok(())
        };
        let attention = |index: usize, unit: usize| -> Result<()> {
            let slot = unit % waves;
            // Both ranks' context attention below read this unit's gather/commit plan.
            self.context_unit.set(unit);
            if let Some(peers) = &peer_workspaces {
                if index == 0 {
                    // Publish the DIRECT mailbox only here, right before rank 1's wait on it.
                    self.exchange()?.push_to(0, DIRECT, workspaces[slot].h.buffer.ptr, peers.get(slot)?.h.buffer.ptr,
                        bytes_of_unit(unit))?;
                }
                self.peer_attention(index, slot, peers.get(slot)?, rows_of(unit), cap,
                    &ranked[unit], bytes_of_unit(unit))?;
            }
            let (w, layer, rows) = (workspaces[slot], &layers[index], rows_of(unit));
            if index == 0 {
                self.norm(w, layer, "input_norm", 0, rows)?;
            }
            self.attention_unit(0, w, index, layer, rows, cap, &ranked[unit], slot)?;
            if split {
                let attended = slot_of(index, false, slot);
                self.exchange()?.push(0, attended, w.delta.buffer.ptr, bytes_of_unit(unit))?;
                self.exchange()?.wait(0, attended)?;
                self.norm_full(0, w, layer.ptr("post_norm")?, 2, rows, w.delta.buffer.ptr, self.recv(0, attended)?)?;
                self.peer_ffn(index, slot, peer_workspaces.as_ref().context("peer workspaces")?.get(slot)?,
                    rows, cap, bytes_of_unit(unit))?;
            } else {
                self.run("glm_norm", &[("residual", w.h.buffer.ptr), ("delta0", w.delta.buffer.ptr),
                    ("delta1", w.delta.buffer.ptr), ("weight", layer.ptr("post_norm")?), ("out", w.x.buffer.ptr)],
                    &[rows, Scalar::I32(1)])?;
            }
            if layer.dense {
                self.ffn(w, layer, self.cfg.dense_intermediate, cap, w.delta.buffer.ptr, rows)?;
                if split {
                    self.exchange()?.push(0, slot_of(index, true, slot), w.delta.buffer.ptr, bytes_of_unit(unit))?;
                }
                Ok(())
            } else {
                self.moe_front(w, layer, counts[unit])
            }
        };
        // h += ffn (routed partials reduced first); x = the next input norm.
        let post = |index: usize, unit: usize, planes: Option<([*const u16; MAX_RANKS], usize)>| -> Result<()> {
            let slot = unit % waves;
            let (w, rows) = (workspaces[slot], rows_of(unit));
            if let Some((planes, ranks)) = planes {
                self.reduce(planes, w, ranks, counts[unit])?;
            }
            let weight = match layers.get(index + 1) {
                Some(next) => next.ptr("input_norm")?,
                None => self.weights.norm.buffer.ptr,
            };
            if split {
                let ffn = slot_of(index, true, slot);
                if planes.is_some() && index + 1 < layers.len() {
                    self.exchange()?.push(0, ffn, w.delta.buffer.ptr, bytes_of_unit(unit))?;
                }
                self.peer_post(index, slot, peer_workspaces.as_ref().context("peer workspaces")?.get(slot)?,
                    rows, bytes_of_unit(unit))?;
                self.exchange()?.wait(0, ffn)?;
                if self.tp2_layer(index) { self.tp2_norm(0, w, weight, index, slot, rows)?; }
                else { self.norm_full(0, w, weight, 2, rows, w.delta.buffer.ptr, self.recv(0, ffn)?)?; }
            } else {
                self.run("glm_norm", &[("residual", w.h.buffer.ptr), ("delta0", w.delta.buffer.ptr),
                    ("delta1", w.delta.buffer.ptr), ("weight", weight), ("out", w.x.buffer.ptr)],
                    &[rows, Scalar::I32(1)])?;
            }
            if let Some(drafter) = &self.drafter {
                let first = starts[unit];
                let from = first.max(window);
                if from < first + counts[unit] {
                    drafter.tap_at(index, w.h.buffer.ptr, from - first, first + counts[unit] - from, from - window)?;
                }
            }
            Ok(())
        };
        let unit_list: Vec<crate::shared::prefill_pipeline::Unit> = (0..lanes.len()).map(|unit| {
            crate::shared::prefill_pipeline::Unit { index: unit, sequence: 0,
                rows: starts[unit]..starts[unit] + counts[unit], layers: 0..layers.len(), slot: unit % waves, barrier: false }
        }).collect();
        let peers: Vec<Option<&Workspace<'_>>> = (0..waves).map(|slot| peer_workspaces.as_ref()
            .map(|p| p.get(slot)).transpose()).collect::<Result<_>>()?;
        // TP2 route diagnostics: every unit checks its own rows on its own slot
        // (slot storage exists before any peer wait); the span always closes,
        // also before an executor error propagates.
        if let Some(pair) = &self.tp2 { pair.start_checks(self.force_route_check.get())?; }
        // Each wave's request build, post, polling and partial uploads run on
        // the link's transport worker while this thread queues GPU work.
        let ran = link.with_waves(self.wave_staging_bytes()?, |transport| {
            let mut hooks = UnitHooks { engine: self, workspaces: &workspaces, peers: &peers, layers, counts: &counts,
                begin: &begin, attention: &attention, post: &post, transport, cap, waves };
            crate::shared::prefill_pipeline::run_prefill_units(&mut hooks, &unit_list, waves)
        });
        if let Some(pair) = &self.tp2 { ran.and(pair.finish_span())?; } else { ran?; }
        // The last unit's workspace holds the call's last rows.
        let last = lanes.len() - 1;
        self.prefill_logits(&[workspaces[last % waves]], &[counts[last]], logit_rows)
    }

    /// Keep a last-lane suffix on device for serving; only cross-lane logits need a host join.
    fn prefill_logits(&self, work: &[&Workspace<'_>], counts: &[usize], wanted: usize) -> Result<Option<StepLogits>> {
        ensure!(work.len() == counts.len(), "prefill logits workspace geometry");
        match prefill_logits_plan(counts, wanted)? {
            PrefillLogitsPlan::None => {
                // SAFETY: the engine owns this stream and every queued prefill operand.
                unsafe { self.library.cuda_stream_synchronize(self.stream)? };
                Ok(None)
            }
            PrefillLogitsPlan::Device { lane, rows } => {
                self.launch_head(work[lane], counts[lane], rows)?;
                Ok(Some(StepLogits::Device(self.device_logits(work[lane], rows, false))))
            }
            PrefillLogitsPlan::Host(rows) => {
                let mut logits = Vec::with_capacity(wanted * self.cfg.vocab_size);
                for ((w, &count), rows) in work.iter().zip(counts).zip(rows) {
                    if rows > 0 {
                        self.launch_head(w, count, rows)?;
                        logits.extend(self.device_logits(w, rows, false).to_host(self.library)?);
                    }
                }
                Ok(Some(StepLogits::Host(logits)))
            }
        }
    }

    /// A transport's rank planes + shared expert into `delta`. The next
    /// wave on that transport is dispatched only after its stage's stream
    /// sync, which follows this reduce.
    fn reduce(&self, planes: [*const u16; crate::shared::spark_intake::MAX_INTAKE_RANKS], w: &Workspace<'_>, ranks: usize, t: usize) -> Result<()> {
        // SAFETY: the transport's intake planes, shared and delta are live
        // [t, h] BF16 buffers ordered after the wave's intake.
        unsafe {
            self.library.v41_compact_reducer()?.reduce_planes(planes, ranks as u32, w.shared.buffer.ptr.cast(),
                w.delta.buffer.ptr.cast(), t as u32, self.stream)
        }
    }

    fn norm(&self, w: &Workspace<'_>, layer: &GlmLayer<'_>, weight: &str, deltas: i32, rows: Scalar) -> Result<()> {
        self.run("glm_norm", &[("residual", w.h.buffer.ptr), ("delta0", w.delta.buffer.ptr),
            ("delta1", w.delta.buffer.ptr), ("weight", layer.ptr(weight)?), ("out", w.x.buffer.ptr)],
            &[rows, Scalar::I32(deltas)])
    }

    /// Before unit `lane`'s attention at `index` under context: it reuses the layer
    /// parity's staging view when the last attention on it was the previous unit's
    /// at this same layer (that one gathered the history and wrote its rows
    /// there); otherwise it gathers its own whole history. Units of a call are one
    /// sequence in order, so unit `lane - 1` is the predecessor.
    pub(super) fn context_unit_turn(&self, index: usize, lane: usize) -> Result<()> {
        let mut owner = self.staging_owner.borrow_mut();
        let view = &mut owner[index % 2];
        self.context_reuse.set(lane > 0 && *view == Some((index, lane - 1)));
        *view = Some((index, lane));
        Ok(())
    }

    /// Layer `index`'s attention on `rank` for one unit: `heads` over this rank's
    /// replicated pools, or under context the decode exchange / prefill gather (`lane`
    /// picks the prefill lane's commit blocks).
    #[allow(clippy::too_many_arguments)]
    fn attention_unit(&self, rank: usize, w: &Workspace<'_>, index: usize, layer: &GlmLayer<'_>, rows: Scalar,
        cap: &str, ranked: &[StepTables; 2], lane: usize) -> Result<()> {
        let tables = &ranked[rank];
        if self.context.is_none() {
            return self.attention_on(rank, w, index, layer, rows, cap, tables);
        }
        if tables.decode {
            return self.context_decode_attention(rank, w, index, layer, rows, tables.positions.len(), tables);
        }
        let index_of = context_plan_index(self.context_unit.get(), lane);
        let units = self.context_units.borrow();
        let mut unit = units.get(index_of).copied().with_context(|| format!(
            "context prefill unit {index_of} of {} (lane {lane})", units.len()))?[rank];
        drop(units);
        if self.context_reuse.get() {
            // The staging view already holds this unit's history: its predecessor
            // gathered it at this layer and wrote its own rows there.
            unit.shards = [0; 2];
        }
        self.context_prefill_attention(rank, w, index, layer, rows, cap, tables, lane, unit)
    }

    /// Uploads a step's tables (one per prefill lane, or the decode step) to rank 0's and
    /// rank 1's workspaces once both streams have drained (every wait on them is matched by
    /// a push already queued, so they drain), and returns each lane's per-rank tables:
    /// identical under heads, each GPU's own under context ([`Self::rank_tables`]).
    fn step_tables(&self, units: &[(&Workspace<'_>, Option<&Workspace<'_>>, &StepTables)], pages: &[Vec<u32>])
        -> Result<Vec<[StepTables; 2]>> {
        // Rank 1 always drains first (as the head split always did); under context rank 0 too:
        // its queued gather and commit kernels read the per-step lists rewritten here.
        let drained = if self.context.is_some() { 0..self.ranks() } else { 1..self.ranks() };
        for rank in drained {
            // SAFETY: the engine owns both streams; nothing queued reads these tables after this.
            self.on(rank, || unsafe { self.library.cuda_stream_synchronize(self.stream_of(rank)) })?;
        }
        let mut out = Vec::new();
        let mut context_units = Vec::new();
        // Gather lists are one buffer per rank and every unit's history is a prefix
        // of the sequence's pages: upload the units in order so the last (longest)
        // list stays, and each unit's shard counts select its own prefix of it.
        for (lane, (&(w, w1, tables), pages)) in units.iter().zip(pages).enumerate() {
            let ranked = if self.context.is_some() {
                let start = tables.positions.first().map_or(0, |&p| p as usize);
                if !tables.decode {
                    // Each unit records its full history; at its attention it gathers it
                    // only when its predecessor's rows are not already in the staging view
                    // ([`Self::context_unit_turn`]).
                    context_units.push([0, 1].map(|rank| self.context_prefill_tables(rank, lane, start,
                        start, tables.positions.len(), pages)).into_iter().collect::<Result<Vec<_>>>()?
                        .try_into().map_err(|_| anyhow::anyhow!("two ranks"))?);
                }
                [self.rank_tables(tables, 0)?, self.rank_tables(tables, 1)?]
            } else {
                [tables.clone(), tables.clone()]
            };
            self.put_tables(w, &ranked[0])?;
            if let Some(w1) = w1 {
                self.on(1, || self.put_tables(w1, &ranked[1]))?;
            }
            out.push(ranked);
        }
        *self.context_units.borrow_mut() = context_units;
        // A single-pass step's lanes index their own plans (`attention_unit`'s `lane`).
        self.context_unit.set(usize::MAX);
        Ok(out)
    }

    /// Layer `index`'s attention on rank `rank` into `delta`: under a head split
    /// that rank's heads (the replicated latent record and DSA indexer, its
    /// heads' sparse MLA, W_UV and a partial o_proj sum).
    #[allow(clippy::too_many_arguments)]
    fn attention_on(&self, rank: usize, w: &Workspace<'_>, index: usize, layer: &GlmLayer<'_>, rows: Scalar,
        cap: &str, tables: &StepTables) -> Result<()> {
        let (kv, index_cache, _) = self.rank_caches(rank, index);
        let (kv, index_cache) = (kv.buffer, index_cache.map(|c| c.buffer));
        self.attention_over(rank, w, index, layer, rows, cap, tables, kv, index_cache)
    }

    /// [`Self::attention_on`] over the given latent and index caches (a rank's pools, or a
    /// context prefill's staging view).
    #[allow(clippy::too_many_arguments)]
    fn attention_over(&self, rank: usize, w: &Workspace<'_>, index: usize, layer: &GlmLayer<'_>, rows: Scalar,
        cap: &str, tables: &StepTables, kv: cuteafd_ffi::CuteafdDeviceBuffer,
        index_cache: Option<cuteafd_ffi::CuteafdDeviceBuffer>) -> Result<()> {
        let mode = if tables.decode { "decode" } else { "prefill" };
        let heads = self.cfg.heads / if layer.attention_split { 2 } else { 1 };
        self.producer(rank, w, layer, rows, cap, kv, w.query.buffer.ptr)?;
        if let Some(index_cache) = index_cache {
            self.index_producer(rank, w, layer, rows, cap, index_cache)?;
            self.run_on(rank, &format!("glm_index_topk_{mode}_{cap}"), &[
                ("q_fp8", w.q_fp8.buffer.ptr), ("weights", w.head_weights.buffer.ptr),
                ("index_k_cache", index_cache.ptr), ("page_table", w.page_table.buffer.ptr),
                ("cache_lengths", w.cache_lengths.buffer.ptr), ("output_indices", w.indices.buffer.ptr),
                ("scratch", w.topk_scratch.buffer.ptr)],
                &[rows, Scalar::I32(tables.table_width as i32), Scalar::I32(tables.table_stride as i32)])?;
        }
        // Shared-indexer layers read the previous full layer's `indices`.
        if let (false, Some(kernel)) = (tables.decode, native_mla_prefill()) {
            let scale = ((self.cfg.qk_nope_head_dim + self.cfg.qk_rope_head_dim) as f32).powf(-0.5);
            // SAFETY: query, cache, indices, lengths and the attention output are
            // live buffers of the step's rows on this rank's stream.
            self.on(rank, || unsafe {
                self.library.glm_mla_prefill(w.query.buffer.ptr, kv.ptr, w.indices.buffer.ptr,
                    w.lengths.buffer.ptr, w.attn.buffer.ptr, tables.positions.len(), heads,
                    self.cfg.index_topk, 656, scale * std::f32::consts::LOG2_E, kernel, self.stream_of(rank))
            })?;
            if mla_prefill_check() {
                // SAFETY: as above; the check synchronizes the stream.
                let stats = self.on(rank, || unsafe {
                    self.library.glm_mla_prefill_check(w.query.buffer.ptr, kv.ptr, w.indices.buffer.ptr,
                        w.lengths.buffer.ptr, tables.positions.len(), heads, self.cfg.index_topk, 656,
                        scale * std::f32::consts::LOG2_E, self.stream_of(rank))
                })?;
                print_mla_check(index, &stats);
            }
        } else {
            let decode_scalars = [rows, Scalar::I32(self.attention_plan.rows() as i32)];
            let prefill_scalars = [rows];
            let scalars = if tables.decode { &decode_scalars[..] } else { &prefill_scalars[..] };
            self.run_on(rank, &Self::program(layer, &format!("glm_sparse_mla_{mode}_{cap}")), &[
                ("q", w.query.buffer.ptr), ("kv_cache", kv.ptr), ("indices", w.indices.buffer.ptr),
                ("lengths", w.lengths.buffer.ptr), ("out", w.attn.buffer.ptr), ("scratch", w.scratch.buffer.ptr)],
                scalars)?;
        }
        self.o_proj(rank, w, layer, cap, rows)
    }

    /// Rank `rank`'s latent and (full-indexer layers) index-key pools of layer `index`, and its RoPE table.
    fn rank_caches(&self, rank: usize, index: usize) -> (&Dev<'a>, Option<&Dev<'a>>, &Dev<'a>) {
        match (rank, &self.peer) {
            (1, Some(peer)) => (&peer.kv[index], peer.index[index].as_ref(), &peer.cos_sin),
            _ => (&self.kv[index], self.index[index].as_ref(), &self.cos_sin),
        }
    }

    /// The MLA producer on `rank`: the latent record of each row into `kv` at its slot (a
    /// negative or scratch slot skips nothing: padded rows write the scratch page) and the
    /// rows' absorbed query of this rank's heads into `query` (`[t, heads, 576]`, or rows
    /// `query_pitch` bytes apart through a strided view when this differs).
    #[allow(clippy::too_many_arguments)]
    fn producer(&self, rank: usize, w: &Workspace<'_>, layer: &GlmLayer<'_>, rows: Scalar, cap: &str,
        kv: cuteafd_ffi::CuteafdDeviceBuffer, query: *mut c_void) -> Result<()> {
        if layer.is_native() {
            let context = self.context.is_some() && query != w.query.buffer.ptr;
            return self.native_producer(rank, w, layer, scalar_rows(rows)?, cap, kv, query, context);
        }
        let (_, _, cos_sin) = self.rank_caches(rank, 0);
        let bf16 = layer.range("w_q_b").is_some();
        let variant = if bf16 { format!("glm_producer_bf16_{cap}") } else { format!("glm_producer_{cap}") };
        let mut producer = vec![("x", w.x.buffer.ptr), ("positions", w.positions.buffer.ptr),
            ("kv_slots", w.slots.buffer.ptr), ("cos_sin", cos_sin.buffer.ptr)];
        producer.extend(weight(layer, "w_qkv_a", cap)?);
        producer.extend([("q_a_norm", layer.ptr("q_a_norm")?), ("kv_a_norm", layer.ptr("kv_a_norm")?)]);
        producer.extend(weight(layer, "w_q_b", cap)?);
        producer.extend(weight(layer, "w_uk", cap)?);
        producer.extend([("kv_cache", kv.ptr), ("query", query), ("q_resid", w.q_resid.buffer.ptr),
            ("scratch", w.scratch.buffer.ptr)]);
        let scalars = if bf16 { vec![rows] } else { self.projection_scalars(rows, cap) };
        self.run_on(rank, &Self::program(layer, &variant), &producer, &scalars)
    }

    /// The DSA index producer on `rank`: index keys at the rows' slots in `index_cache`,
    /// the FP8 index query and head weights into the workspace.
    fn index_producer(&self, rank: usize, w: &Workspace<'_>, layer: &GlmLayer<'_>, rows: Scalar, cap: &str,
        index_cache: cuteafd_ffi::CuteafdDeviceBuffer) -> Result<()> {
        if layer.is_native() {
            return self.native_index_producer(rank, w, layer, scalar_rows(rows)?, cap, index_cache);
        }
        let (_, _, cos_sin) = self.rank_caches(rank, 0);
        let bf16 = layer.range("w_iq").is_some();
        let mut pointers = vec![("x", w.x.buffer.ptr), ("q_resid", w.q_resid.buffer.ptr),
            ("positions", w.positions.buffer.ptr), ("index_slots", w.slots.buffer.ptr),
            ("cos_sin", cos_sin.buffer.ptr)];
        pointers.extend(weight(layer, "w_iq", cap)?);
        pointers.extend([("w_ik", layer.ptr("w_ik")?), ("k_norm_w", layer.ptr("k_norm_w")?),
            ("k_norm_b", layer.ptr("k_norm_b")?), ("index_cache", index_cache.ptr),
            ("q_fp8", w.q_fp8.buffer.ptr), ("head_weights", w.head_weights.buffer.ptr),
            ("scratch", w.scratch.buffer.ptr)]);
        let name = if bf16 { format!("glm_index_producer_bf16_{cap}") } else { format!("glm_index_producer_{cap}") };
        let scalars = if bf16 { vec![rows] } else { self.projection_scalars(rows, cap) };
        self.run_on(rank, &name, &pointers, &scalars)
    }

    /// W_UV + o_proj of this rank's heads over `attn` into `delta` (a partial sum under a split).
    fn o_proj(&self, rank: usize, w: &Workspace<'_>, layer: &GlmLayer<'_>, cap: &str, rows: Scalar) -> Result<()> {
        if layer.is_native() {
            return self.native_o_proj(rank, w, layer, scalar_rows(rows)?, cap);
        }
        let bf16 = layer.range("w_o").is_some();
        let mut o = vec![("attn", w.attn.buffer.ptr)];
        o.extend(weight(layer, "w_uv", cap)?);
        o.extend(weight(layer, "w_o", cap)?);
        o.extend([("out", w.delta.buffer.ptr), ("scratch", w.scratch.buffer.ptr)]);
        let name = if bf16 { format!("glm_o_bf16_{cap}") } else { format!("glm_o_{cap}") };
        let scalars = if bf16 { vec![rows] } else { self.projection_scalars(rows, cap) };
        self.run_on(rank, &Self::program(layer, &name), &o, &scalars)
    }

    /// Scalars of the programs over FP8 weights: `rows`, plus the prefill
    /// programs' `fp8_rows` (1: W8A8, 0: W8A16).
    fn projection_scalars(&self, rows: Scalar, cap: &str) -> Vec<Scalar> {
        if cap == "m64" {
            vec![rows]
        } else {
            vec![rows, Scalar::I32(i32::from(self.prefill_w8a8))]
        }
    }

    /// SwiGLU MLP (dense layers, or the shared expert) into `out`.
    fn ffn(&self, w: &Workspace<'_>, layer: &GlmLayer<'_>, intermediate: usize, cap: &str, out: *mut c_void,
        rows: Scalar) -> Result<()> {
        self.ffn_on(0, w, layer, intermediate, cap, out, rows)
    }

    /// [`Self::ffn`] on rank `rank`; a head-split layer runs its slice of the
    /// `intermediate` (a partial sum).
    #[allow(clippy::too_many_arguments)]
    fn ffn_on(&self, rank: usize, w: &Workspace<'_>, layer: &GlmLayer<'_>, intermediate: usize, cap: &str,
        out: *mut c_void, rows: Scalar) -> Result<()> {
        if layer.is_native() && layer.dense {
            let rows = scalar_rows(rows)?;
            let out = cuteafd_ffi::CuteafdDeviceBuffer { ptr: out, bytes: rows * self.cfg.hidden * 2,
                device_id: self.exchange_device(rank)?, flags: 0 };
            return self.native_ffn(rank, w, layer, rows, out);
        }
        let intermediate = intermediate / if layer.split { 2 } else { 1 };
        let declared_w8a8 = layer.mlp_numerics == cuteafd_loader::plan::checkpoint::ModuleNumerics::W8A8;
        if declared_w8a8 || (cap != "m64" && self.prefill_w8a8 && tensor_fp8_prefill()
            && layer.range("w_gate_up_tscale").is_some()) {
            // ModelOpt per-tensor FP8: static W8A8 with the checkpoint's input and weight scales.
            let pointers = [("x", w.x.buffer.ptr), ("w_gate_up_fp8", layer.ptr("w_gate_up_fp8")?),
                ("w_gate_up_tscale", layer.ptr("w_gate_up_tscale")?), ("w_down_fp8", layer.ptr("w_down_fp8")?),
                ("w_down_tscale", layer.ptr("w_down_tscale")?), ("out", out), ("scratch", w.scratch.buffer.ptr)];
            return self.run_on(rank, &Self::program(layer, &format!("glm_ffn_i{intermediate}_pt_{cap}")), &pointers,
                &[rows]);
        }
        let mut pointers = vec![("x", w.x.buffer.ptr)];
        pointers.extend(weight(layer, "w_gate_up", cap)?);
        pointers.extend(weight(layer, "w_down", cap)?);
        pointers.extend([("out", out), ("scratch", w.scratch.buffer.ptr)]);
        if layer.range("w_gate_up").is_some() {
            // A checkpoint's BF16 shared expert on its own weights.
            return self.run_on(rank, &Self::program(layer, &format!("glm_ffn_i{intermediate}_bf16_{cap}")), &pointers,
                &[rows]);
        }
        self.run_on(rank, &Self::program(layer, &format!("glm_ffn_i{intermediate}_{cap}")), &pointers,
            &self.projection_scalars(rows, cap))
    }
}

/// The live row count a program scalar carries.
fn scalar_rows(rows: Scalar) -> Result<usize> {
    match rows { Scalar::I32(n) if n > 0 => Ok(n as usize), other => anyhow::bail!("row scalar {other:?}") }
}

impl<'a> GlmEngine<'a> {
    /// Rank `rank`'s device id.
    fn exchange_device(&self, rank: usize) -> Result<i32> {
        Ok(if rank == 0 { self.device } else { self.peer()?.device })
    }


}

/// Optional static per-tensor prefill route for undeclared FP8 modules.
/// Declared W8A8 modules always use their calibrated static path at every row
/// count; CUTEAFD_GLM_TENSOR_FP8=block does not override that contract.
pub(crate) fn tensor_fp8_prefill() -> bool {
    static TENSOR: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *TENSOR.get_or_init(|| std::env::var("CUTEAFD_GLM_TENSOR_FP8").map_or(true, |v| v != "block"))
}

/// Prefill sparse MLA kernel of glm_mla_prefill.cu (its `kernel` argument), or None for the
/// b12x program: CUTEAFD_MLA_PREFILL=e4m3-p2 (2, the default: E4M3 query, two-term E4M3 P),
/// e4m3 (1: one-term P), e4m3-q2 (3: two-term query), e4m3-q2p2 (4), f16 (0: the F16 kernel)
/// or b12x.
pub(crate) fn native_mla_prefill() -> Option<i32> {
    static KERNEL: std::sync::OnceLock<Option<i32>> = std::sync::OnceLock::new();
    *KERNEL.get_or_init(|| match std::env::var("CUTEAFD_MLA_PREFILL").as_deref() {
        Ok("b12x") => None,
        Ok("f16") => Some(0),
        Ok("e4m3") => Some(1),
        Ok("e4m3-p2") | Err(_) => Some(2),
        Ok("e4m3-q2") => Some(3),
        Ok("e4m3-q2p2") => Some(4),
        Ok(other) => {
            tracing::warn!(value = other, "unknown CUTEAFD_MLA_PREFILL; using e4m3-p2");
            Some(2)
        }
    })
}

/// CUTEAFD_MLA_PREFILL_CHECK=1: after each prefill MLA, run every native kernel on its inputs and
/// print their differences from the two-term E4M3 kernel (diagnostics; synchronizes, allocates).
pub(crate) fn mla_prefill_check() -> bool {
    static CHECK: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *CHECK.get_or_init(|| std::env::var("CUTEAFD_MLA_PREFILL_CHECK").is_ok_and(|v| v == "1"))
}

pub(crate) fn print_mla_check(layer: usize, stats: &[[f64; 3]; 5]) {
    let names = ["f16", "e4m3", "e4m3-p2", "e4m3-q2", "e4m3-q2p2"];
    println!("mla check layer {layer:2} (vs e4m3-q2p2, rms {:.3e}): {}", stats[4][2], (0..4)
        .map(|k| format!("{} rel {:.2e} max {:.2e}", names[k], stats[k][0], stats[k][1]))
        .collect::<Vec<_>>().join(" | "));
}

/// GLM's per-unit hooks over the unchanged lane arithmetic. Unit `u` holds
/// workspace slot `u % waves` (rank 0's and rank 1's) and that slot's
/// exchange lane; its Spark waves go out on the prefill link, one return slot each.
struct UnitHooks<'h, 'e, 'a, 'l, 'r, B, A, P> {
    engine: &'h GlmEngine<'a>,
    workspaces: &'h [&'h Workspace<'e>],
    /// Rank 1's workspace per slot (borrowed once for the call).
    peers: &'h [Option<&'h Workspace<'e>>],
    layers: &'h [GlmLayer<'a>],
    counts: &'h [usize],
    begin: &'h B,
    attention: &'h A,
    post: &'h P,
    transport: &'h mut crate::shared::spark_waves::Waves<'l, 'r>,
    cap: &'static str,
    waves: usize,
}

impl<B, A, P> crate::shared::prefill_pipeline::PrefillHooks for UnitHooks<'_, '_, '_, '_, '_, B, A, P>
where B: Fn(usize) -> Result<()>,
      A: Fn(usize, usize) -> Result<()>,
      P: Fn(usize, usize, Option<([*const u16; MAX_RANKS], usize)>) -> Result<()> {
    type Ticket = crate::shared::spark_waves::WaveTicket;
    fn route(&self, layer: usize) -> crate::shared::prefill_pipeline::FfnRoute {
        if self.layers[layer].dense || self.engine.tp2_layer(layer) {
            crate::shared::prefill_pipeline::FfnRoute::Local
        } else { crate::shared::prefill_pipeline::FfnRoute::Spark }
    }
    fn begin_unit(&mut self, unit: &crate::shared::prefill_pipeline::Unit) -> Result<()> {
        if let Some(pair) = &self.engine.tp2 { pair.begin_unit(unit.slot, unit.index, self.counts[unit.index])?; }
        (self.begin)(unit.index)
    }
    fn attention(&mut self, layer: usize, unit: &crate::shared::prefill_pipeline::Unit) -> Result<()> {
        self.engine.context_unit_turn(layer, unit.index)?;
        (self.attention)(layer, unit.index)
    }
    fn local_ffn(&mut self, layer: usize, unit: &crate::shared::prefill_pipeline::Unit) -> Result<()> {
        let slot = unit.index % self.waves;
        if self.engine.tp2_layer(layer) {
            let w1 = self.peers[slot].context("TP2 peer workspaces")?;
            self.engine.tp2_pair(layer, slot, self.workspaces[slot], w1, self.counts[unit.index], self.cap)?;
        }
        (self.post)(layer, unit.index, None)
    }
    fn send_experts(&mut self, layer: usize, unit: &crate::shared::prefill_pipeline::Unit)
        -> Result<crate::shared::spark_waves::WaveTicket> {
        let w = self.workspaces[unit.index % self.waves];
        let rows = self.counts[unit.index];
        self.engine.moe_wave(w, layer, &self.layers[layer], rows, self.cap, self.transport)
    }
    fn land_experts(&mut self, layer: usize, unit: &crate::shared::prefill_pipeline::Unit,
        wave: crate::shared::spark_waves::WaveTicket) -> Result<()> {
        let slot = wave.slot();
        let timer = std::time::Instant::now();
        let times = self.transport.land(wave, self.engine.stream)?;
        self.engine.profile.borrow_mut()[1] += timer.elapsed().as_secs_f64();
        self.engine.exchange_host.borrow_mut()[0] += times.posted;
        if let Some(probe) = self.engine.draft_probe.borrow_mut().as_mut().filter(|p| p.active()) {
            probe.staged(layer, self.transport.staged_ids(slot)?);
        }
        // The reduce reads the return slot this wave just landed in; the slot
        // takes its next wave only after that reduce.
        (self.post)(layer, unit.index, Some((self.transport.planes(slot), self.transport.ranks())))?;
        self.transport.consumed(slot, self.engine.stream)
    }
    fn end_unit(&mut self, unit: &crate::shared::prefill_pipeline::Unit) -> Result<()> {
        // Drains (diagnostics only) and compares this unit's routes before its slot is reused.
        if let Some(pair) = &self.engine.tp2 { pair.finish_unit(unit.slot, unit.index)?; }
        Ok(())
    }
    fn commit_unit(&mut self, _: &crate::shared::prefill_pipeline::Unit) -> Result<()> { Ok(()) }
}

/// Default rows per continuous prefill span (shared PREFILL_SPAN_ROWS key).
pub(crate) const DEFAULT_PREFILL_SPAN_ROWS: usize = cuteafd_loader::plan::layout::GLM_DEFAULT_PREFILL_SPAN_ROWS as usize;


/// The expert request of `t` staged rows: expert ids (U32) and gate
/// weights (F32) `[t, topk]`, then the FP8 wire rows, as `moe_stage` lays
/// them out in the pinned router staging.
#[allow(clippy::too_many_arguments)]
fn expert_request(staged: &[u8], index: usize, t: usize, h: usize, topk: usize, decode: bool,
    bf16: bool) -> Result<ExpertProtocolV2Request> {
    let layout = crate::shared::experts::staging::RouteStaging::new(t, topk,
        if bf16 { h * 2 } else { h + h / 32 })?;
    let routes = layout.routes(staged)?;
    let mut wire = vec![0u8; layout.wire_bytes];
    copy_parallel(&mut wire, layout.wire(staged)?);
    expert_request_parts(routes, wire.into(), index, t, h, topk, decode, bf16)
}

/// The expert request of `t` rows from its route entries and wire rows
/// (`wire` is shared, e.g. a view of a wave slot's pinned staging).
#[allow(clippy::too_many_arguments)]
fn expert_request_parts(routes: Vec<cuteafd_transport::ExpertProtocolV2RouteEntry>, wire: bytes::Bytes,
    index: usize, t: usize, h: usize, topk: usize, decode: bool, bf16: bool) -> Result<ExpertProtocolV2Request> {
    let kind = if decode { ExpertV2SourceKind::Decode } else { ExpertV2SourceKind::Prefill };
    let mut request = ExpertProtocolV2Request::new_bytes(index as u64 + 1, 17, index as u32, h as u32,
        if bf16 { ExpertV2Dtype::Bf16 } else { ExpertV2Dtype::Fp8E4m3Ue8m0K32 },
        (0..t as u32).map(|row| ExpertProtocolV2RowDescriptor {
            row_id: u64::from(row), source_kind: kind, source_request_id: 1,
            token_position: u64::from(row), route_offset: row * topk as u32, route_count: topk as u32,
        }).collect(),
        routes, wire)?;
    request.header.flags |= EXPERT_PROTOCOL_V2_FLAG_V41_COMPACT_BF16;
    Ok(request)
}

/// A weight's program pointers: the checkpoint's BF16 operand when the layer holds one (the
/// BF16 programs), else its E4M3 copy and FP32 scales (prefill programs read `w_qkv_a`'s
/// per-row scales, K-block major).
fn weight(layer: &GlmLayer<'_>, name: &'static str, cap: &str) -> Result<Vec<(&'static str, *mut c_void)>> {
    if layer.range(name).is_some() {
        return Ok(vec![(name, layer.ptr(name)?)]);
    }
    let (fp8, scale, kscale) = super::weights::fp8_operand_names(name);
    let scale = if cap != "m64" && !kscale.is_empty() { kscale } else { scale };
    Ok(vec![(fp8, layer.ptr(fp8)?), (scale, layer.ptr(scale)?)])
}

#[derive(Debug, PartialEq, Eq)]
enum PrefillLogitsPlan {
    None,
    Device { lane: usize, rows: usize },
    Host(Vec<usize>),
}

fn prefill_logits_plan(counts: &[usize], wanted: usize) -> Result<PrefillLogitsPlan> {
    ensure!(!counts.is_empty() && counts.iter().all(|&rows| rows > 0), "empty prefill lanes");
    let total: usize = counts.iter().sum();
    ensure!(wanted <= total, "prefill logits exceed lane rows");
    if wanted == 0 { return Ok(PrefillLogitsPlan::None); }
    let lane = counts.len() - 1;
    if wanted <= counts[lane] { return Ok(PrefillLogitsPlan::Device { lane, rows: wanted }); }
    let mut first = 0;
    Ok(PrefillLogitsPlan::Host(counts.iter().map(|&rows| {
        let count = (first + rows).saturating_sub((total - wanted).max(first));
        first += rows;
        count
    }).collect()))
}

#[cfg(test)]
mod long_graph_tests {
    use super::*;

    #[test]
    fn native_expert_request_preserves_fp8_rows_and_bf16_response_contract() -> Result<()> {
        for rows in [1, 8, 64, 65] {
            let (hidden, topk) = (128, 2);
            let routes = rows * topk;
            let mut staged = Vec::new();
            for id in 0..routes { staged.extend_from_slice(&((id % 256) as u32).to_le_bytes()); }
            for _ in 0..routes { staged.extend_from_slice(&0.5f32.to_le_bytes()); }
            let wire = vec![0x3c; rows * (hidden + hidden / 32)];
            staged.extend_from_slice(&wire);
            let request = expert_request(&staged, 3, rows, hidden, topk, false, false)?;
            assert_eq!(request.header.hidden_dtype, ExpertV2Dtype::Fp8E4m3Ue8m0K32);
            assert_eq!(request.header.flags, EXPERT_PROTOCOL_V2_FLAG_V41_COMPACT_BF16);
            assert_eq!(request.hidden_payload.as_ref(), wire.as_slice());
            assert_eq!(request.routes.len(), routes);
            assert!(expert_request(&staged[..staged.len() - 1], 3, rows, hidden, topk,
                false, false).is_err());
        }
        Ok(())
    }

    #[test]
    fn prefill_last_lane_logits_stay_on_device_for_every_executor() {
        for counts in [vec![4096], vec![4096, 1024], vec![4096, 4096, 17]] {
            let lane = counts.len() - 1;
            for rows in [1, counts[lane]] {
                assert_eq!(prefill_logits_plan(&counts, rows).unwrap(), PrefillLogitsPlan::Device { lane, rows });
            }
            assert_eq!(prefill_logits_plan(&counts, 0).unwrap(), PrefillLogitsPlan::None);
            if lane > 0 {
                let mut expected = vec![0; counts.len()];
                expected[lane - 1] = 1;
                expected[lane] = counts[lane];
                assert_eq!(prefill_logits_plan(&counts, counts[lane] + 1).unwrap(), PrefillLogitsPlan::Host(expected));
                assert_eq!(prefill_logits_plan(&counts, counts.iter().sum()).unwrap(), PrefillLogitsPlan::Host(counts.clone()));
            }
            assert!(prefill_logits_plan(&counts, counts.iter().sum::<usize>() + 1).is_err());
        }
        assert!(prefill_logits_plan(&[], 0).is_err());
        assert!(prefill_logits_plan(&[4096, 0], 1).is_err());
    }


    #[test]
    fn long_padding_resolves_only_to_sized_scratch_records() {
        let pages = 22_018;
        for real in 1..=64 {
            let width = 1_048_576 / PAGE_ROWS;
            let bucket = glm_decode_row_bucket(real, width).unwrap();
            let mut tables = StepTables { decode: true, positions: vec![200_000; real],
                slots: vec![123 * PAGE_ROWS as i64; real], page_table: vec![123; real * width],
                table_width: width, table_stride: width, cache_lengths: vec![200_001; real],
                lengths: vec![2048; real], exchange_rows: real };
            tables.pad_rows(pages, bucket);
            assert_eq!(tables.exchange_rows, real);
            assert_eq!(&tables.page_table[..real * width], vec![123; real * width]);
            assert_eq!(&tables.positions[..real], vec![200_000; real]);
            assert_eq!(tables.page_table.len(), bucket * width);
            assert!(bucket * width * 4 <= DECODE_ROWS * pages * 4);
            assert!(bucket <= DECODE_ROWS);
            for row in real..bucket {
                assert_eq!(tables.positions[row], 0);
                assert_eq!(tables.cache_lengths[row], 1);
                assert_eq!(tables.lengths[row], 1);
                let slot = tables.slots[row] as usize;
                assert_eq!(slot, pages * PAGE_ROWS);
                assert!(tables.page_table[row * width..(row + 1) * width].iter()
                    .all(|&page| page as usize == pages));
                assert!((slot + 1) * RECORD_BYTES <= (pages + 1) * RECORD_PAGE_BYTES);
                // 128-byte index key plus its 4-byte scale; only position zero is live.
                assert!((slot + 1) * 132 <= (pages + 1) * INDEX_PAGE_BYTES);
            }
        }
        assert_eq!(glm_decode_row_bucket(33, 16384).unwrap(), DECODE_ROWS);
        assert_eq!(INDEX_PAGE_BYTES, PAGE_ROWS * 132);
    }
}

impl Drop for GlmEngine<'_> {
    fn drop(&mut self) {
        if self.tp2.is_some() || self.layer_execution.is_some()
            || (!self.native_experts.is_empty() && self.peer.is_some()) {
            crate::shared::decode_graph::fatal_drain(self.retire_tp2(), "GLM peer engine retirement");
        }
        if !self.native_experts.is_empty() {
            // SAFETY: native shared execution and workspace owners remain live until this drain.
            crate::shared::decode_graph::fatal_drain(unsafe {
                self.library.cuda_stream_synchronize(self.stream)
            }, "GLM EXLR8 engine retirement");
        }
    }
}

#[cfg(test)]
mod prefill_unit_tests {
    #[test]
    fn context_plans_follow_the_unit_for_both_ranks_and_reset_for_single_steps() {
        // A unit-driven call selects the unit on both ranks; a single-pass step
        // falls back to its lane, whatever an earlier call left behind.
        assert_eq!(super::context_plan_index(3, 1), 3);
        assert_eq!(super::context_plan_index(usize::MAX, 0), 0);
        assert_eq!(super::context_plan_index(usize::MAX, 2), 2);
        let source = include_str!("engine.rs");
        let body = source.split("let attention = |index: usize, unit: usize|").nth(1).unwrap()
            .split("let post = |index").next().unwrap();
        let set = body.find("self.context_unit.set(unit)").expect("unit attention selects its plan");
        let peer = body.find("self.peer_attention(").expect("rank 1 attention");
        let own = body.find("self.attention_unit(0,").expect("rank 0 attention");
        assert!(set < peer && set < own, "the plan is selected before either rank's attention");
        let tables = source.split("fn step_tables(").nth(1).unwrap().split("\n    }\n").next().unwrap();
        assert!(tables.contains("self.context_unit.set(usize::MAX)"), "single-pass steps reset the unit cursor");
    }

    /// GLM's hook mapping for TP2 route diagnostics (begin at Begin on the
    /// unit's slot, capture in the TP2 local FFN, finish at End, span closed
    /// after the executor) over the real executor on a long span: every unit
    /// is checked, including units on recycled slots, and a later unit's
    /// corruption fails strict checks naming that unit and slot.
    #[test]
    fn route_checks_cover_every_unit_on_recycled_slots() {
        use crate::shared::experts::rtx::{routes::SlotRouteChecks, RouteCheck};
        use crate::shared::prefill_pipeline::{run_prefill_units, sequence_units, FfnRoute, PrefillHooks, Unit};
        struct Hooks { checks: SlotRouteChecks<Vec<RouteCheck>>, corrupt: Option<usize>, checked: Vec<usize> }
        impl PrefillHooks for Hooks {
            type Ticket = ();
            fn route(&self, layer: usize) -> FfnRoute { if layer % 2 == 0 { FfnRoute::Local } else { FfnRoute::Spark } }
            fn begin_unit(&mut self, unit: &Unit) -> anyhow::Result<()> {
                self.checks.begin(unit.slot, unit.index, unit.rows.len(), true)
            }
            fn attention(&mut self, _: usize, _: &Unit) -> anyhow::Result<()> { Ok(()) }
            fn local_ffn(&mut self, layer: usize, unit: &Unit) -> anyhow::Result<()> {
                let rows = unit.rows.len();
                let ids = vec![0u8; rows * 2 * 8];
                let mut peer = ids.clone();
                if self.corrupt == Some(unit.index) && layer == 2 { peer[3 * 8] ^= 1; }
                self.checks.storage(unit.slot, unit.index)?.unwrap()
                    .push(RouteCheck { layer, rows, topk: 2, ranks: [ids, peer] });
                Ok(())
            }
            fn send_experts(&mut self, _: usize, _: &Unit) -> anyhow::Result<()> { Ok(()) }
            fn land_experts(&mut self, _: usize, _: &Unit, _: ()) -> anyhow::Result<()> { Ok(()) }
            fn end_unit(&mut self, unit: &Unit) -> anyhow::Result<()> {
                let identity = self.checks.finish(unit.slot, unit.index,
                    |storage, rows| { assert!(storage.iter().all(|c| c.rows == rows)); Ok(std::mem::take(storage)) })?;
                assert_eq!(identity.unwrap().layers_checked, 2);
                self.checked.push(unit.index);
                Ok(())
            }
            fn commit_unit(&mut self, _: &Unit) -> anyhow::Result<()> { Ok(()) }
        }
        let run = |corrupt| {
            let units = sequence_units(16384, 4096, 2, 4).unwrap();
            assert!(units.len() > 2 && units.iter().any(|u| u.index >= 2 && u.slot == u.index % 2));
            let mut hooks = Hooks { checks: SlotRouteChecks::new(vec![Vec::new(), Vec::new()], true).unwrap(),
                corrupt, checked: Vec::new() };
            let ran = run_prefill_units(&mut hooks, &units, 2);
            let result = ran.and(hooks.checks.finish_span());
            (result, hooks.checked)
        };
        let (result, checked) = run(None);
        result.unwrap();
        assert_eq!(checked, vec![0, 1, 2, 3], "every unit, including those on recycled slots, is checked");
        let (result, checked) = run(Some(3));
        let error = result.unwrap_err().to_string();
        assert!(error.contains("unit 3, slot 1, layer 2, row 3"), "{error}");
        assert_eq!(checked, vec![0, 1, 2]);
        // The engine wires that mapping, with no single-slot-only gate.
        let source = include_str!("engine.rs");
        assert!(!source.contains(concat!("route_checks = lanes.len() ", "<= waves")));
        let hooks = source.split("impl<B, A, P> crate::shared::prefill_pipeline::PrefillHooks").nth(1).unwrap();
        assert!(hooks.contains("pair.begin_unit(unit.slot, unit.index"));
        assert!(hooks.contains("pair.finish_unit(unit.slot, unit.index)"));
        assert!(source.contains("ran.and(pair.finish_span())"), "the span closes even when the executor fails");
    }

    #[test]
    fn unit_entry_is_published_in_its_first_attention_not_at_begin() {
        // The schedule emits Begin for every live slot before any attention; one
        // DIRECT flag sequence serves all slots, so an entry pushed at Begin could
        // satisfy an earlier unit's wait. Publish beside rank 1's wait instead.
        let source = include_str!("engine.rs");
        let begin = source.split("let begin = |unit: usize|").nth(1).unwrap()
            .split("let attention = |index: usize, unit: usize|").next().unwrap();
        assert!(!begin.contains("push_to(0, DIRECT"), "Begin publishes no DIRECT mailbox");
        let attention = source.split("let attention = |index: usize, unit: usize|").nth(1).unwrap()
            .split("let post = |index").next().unwrap();
        let first = attention.find("if index == 0 {").expect("first-layer entry publish");
        let push = attention.find("push_to(0, DIRECT").expect("unit entry pushed in attention");
        let wait = attention.find("self.peer_attention(").expect("rank 1 waits in peer_attention");
        assert!(first < push && push < wait, "entry push precedes rank 1's wait, inside the first layer only");
        let peer = source.split("fn peer_attention(").nth(1).unwrap().split("\n    }\n").next().unwrap();
        assert!(peer.find("exchange.wait(1, DIRECT)").is_some_and(|w| w < peer.find("attention_unit(1,").unwrap()));
    }
}
