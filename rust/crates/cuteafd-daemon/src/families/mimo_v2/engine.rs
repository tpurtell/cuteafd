//! MiMo V2 (mimo_v2_flash, V2.6 Pro mimo_v2) coordinator over the exported
//! mimo_* (Flash) or mimop_* (V2.6 Pro) programs.
//!
//! One layer: input norm (fused with the previous layer's residual add),
//! the QKV producer (RoPE, KV record), GQA attention, o_proj, the
//! post-attention norm, then the dense MLP or the MoE: router scores (FP32
//! weight as BF16 hi + lo), the native sigmoid top-8 select, FP8 K32 wire
//! rows, then the routed experts in the checkpoint's own FP8 (the `fp8`
//! family): on the Sparks (one BF16 partial per TP rank, summed on the GPU,
//! as GLM) or, with local experts, on this GPU from the TP1 package.
//!
//! KV state: full layers keep one record per token (keys then values of the 4
//! KV heads) in a paged pool shared by sequences (64 rows per page); SWA layers
//! keep a 256-slot ring per sequence (8 KV heads). Full-attention records are
//! int8 by default (an FP32 scale `amax / 127` per 32 dims of each head's key and
//! value: 1440 bytes on V2 Flash) or BF16 (`--kv-cache bf16`: 2560 bytes); the
//! `full_*_kvint8` programs read and write the int8 layout, and int8 prefill widens
//! the sequence's records once into `kv_wide`. SWA records are BF16. An SWA
//! step's records go to a step buffer first; the attention program reads
//! in-step keys from it, older keys from the ring, and commits the step to the
//! ring afterwards.
//!
//! Decode and verify steps launch eagerly, or (`--decode-graphs true`) replay one
//! captured graph per layer segment between the Spark exchanges
//! ([`MimoEngine::decode_layers`]), every 1..=64-row shape captured at startup.
use super::weights::{MimoLayer, MimoWeights};
use super::head::BorrowedHead;
use cuteafd_loader::families::mimo_v2::workspace::MimoPrefillKvShadowPlan;
use crate::shared::experts::fp8::{Fp8Experts, Fp8Layer};
use cuteafd_loader::formats::fp8_experts::Fp8ExpertTensors;
use crate::shared::launch_grid::Fp8QuantizeGrid;
use crate::shared::memory::{DeviceAllocation, HostAllocation};
use crate::shared::memory::device::{Allocation, Device};
use crate::shared::token_io::{DeviceLogits, TokenEmbedding};
use crate::shared::spark_intake::SparkLink;
use cuteafd_transport::expert::EXPERT_PROTOCOL_V2_FLAG_V41_COMPACT_BF16;
use cuteafd_transport::{
    ExpertProtocolV2Request, ExpertProtocolV2RouteEntry, ExpertProtocolV2RowDescriptor, ExpertV2Dtype, ExpertV2SourceKind,
};
use anyhow::{ensure, Context, Result};
use cuteafd_ffi::programs::{Programs, Scalar, VocabularyHead, VOCABULARY_HEAD_WORKSPACE};
use cuteafd_ffi::NativeLibrary;
use cuteafd_loader::families::mimo_v2::{MimoAttention, MimoKvCache, MimoV2Config,
    MimoAttentionWorkspace, MimoPrefillOutput, MimoWorkspaceLayout, MimoWorkspaceOptions};
use crate::shared::peer_split::{PeerExchange, RankDevice, DIRECT};
use std::cell::{Cell, RefCell};
use std::collections::HashMap;
use std::ffi::c_void;
use std::rc::Rc;

type Dev<'a> = DeviceAllocation<'a>;

pub(crate) const PAGE_ROWS: usize = 64;
pub(crate) const RING_ROWS: usize = 256;
/// Rows of the decode-route programs (`_m64`).
pub(crate) const DECODE_ROWS: usize = cuteafd_loader::families::mimo_v2::decode_graph::MIMO_DECODE_ROWS;
/// Expert scratch and transport must fit every decode/verify row shape even
/// when prompts are deliberately processed through a narrower workspace.
pub(super) fn expert_capacity(prefill_rows: usize) -> usize {
    prefill_rows.max(DECODE_ROWS)
}
/// Programs of one layer's attention: the qkv producer, attention, o_proj.
const ATTENTION_PARTS: usize = 3;

/// Most Spark ranks a step's partials come from (the compact reducer's limit).
const MAX_RANKS: usize = 6;

/// What the coordinator sends the Spark ranks as expert input rows.
#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
pub(crate) enum ExpertInput {
    /// FP8 K32 wire rows (E4M3 + UE8M0 per 32; `H + H/32` bytes per row).
    Fp8,
    /// The BF16 rows themselves (`2H` bytes; the ranks need the `-bf16` package).
    Bf16,
    /// BF16 for decode-shaped steps, FP8 wire rows for prefill.
    Bf16Decode,
}

impl ExpertInput {
    pub fn bf16(self, decode: bool) -> bool {
        match self {
            Self::Fp8 => false,
            Self::Bf16 => true,
            Self::Bf16Decode => decode,
        }
    }
}

/// Where the routed experts run.
pub(crate) enum Experts<'a> {
    /// The TP1 FP8 package on the coordinator GPU (resident MoE layers).
    Local(Fp8Experts<'a>),
    /// The TP1 package on the coordinator GPU with a window of resident MoE
    /// layers, loading each missing layer over the oldest (the model's experts
    /// do not fit: MiMo V2.6 Pro's are 495 GiB). For prefill checks; a decode
    /// step would reload every layer.
    Streamed { experts: RefCell<Fp8Experts<'a>>, tensors: &'a Fp8ExpertTensors, window: usize },
    /// No routed experts: MoE layers add zero (coordinator timing only; the
    /// outputs are not the model's).
    Skip,
    /// Spark ranks serving the `fp8` family over RoCE.
    /// `lanes`: further transports to the same ranks for the earlier row lanes
    /// of a pipelined prefill ([`MimoEngine::prefill_capacity`]; `transport`
    /// carries the last lane); empty keeps prefill serial.
    Spark { transport: RefCell<SparkLink<'a>>, lanes: Vec<RefCell<SparkLink<'a>>>, runtime: tokio::runtime::Runtime },
}

/// A dispatched Spark wave and its send-side host phases (seconds).
struct SentWave {
    /// Taken by [`MimoEngine::spark_receive`].
    wave: Option<cuteafd_transport::expert::SparkExpertWave>,
    gpu_wait: f64,
    build: f64,
    dispatch: f64,
    sent: std::time::Instant,
}

#[derive(Debug, thiserror::Error)]
#[error("MiMo head-split submission failed permanently: {cause}; {drain}")]
pub(crate) struct TerminalStepError {
    #[source]
    cause: anyhow::Error,
    drain: String,
}

/// Fewest rows per lane of a pipelined prefill: at least the DFlash tap ring
/// (1024 rows) and the MTP hidden ring, which the last lane alone feeds.
const MIN_LANE_ROWS: usize = 1024;

/// A serving chunk may end at any row, including a short prompt tail. Do not
/// advertise twice the workspace when a tail larger than one workspace could
/// still be too short for the two-lane path's tap window.
fn lane_prefill_capacity(rows: usize, lanes: usize) -> usize {
    if lanes >= 2 && rows >= 2 * MIN_LANE_ROWS { lanes * rows } else { rows }
}

/// Row lanes of a `t`-row pipelined prefill step with `lanes` transports of
/// `rows`-row workspaces: as many as keep every lane at least
/// [`MIN_LANE_ROWS`] (and enough that none exceeds `rows`); 1 is serial.
fn prefill_lane_count(t: usize, rows: usize, lanes: usize) -> usize {
    if lanes < 2 || t < 2 * MIN_LANE_ROWS {
        return 1;
    }
    let mut n = (t / MIN_LANE_ROWS).min(lanes).max(2);
    // The engine splits ceil(t / n) rows per lane; the last one keeps the rest.
    while n > 2 && n > t.div_ceil(rows) && t - (n - 1) * t.div_ceil(n) < MIN_LANE_ROWS {
        n -= 1;
    }
    n
}

fn independent_prefill_rows(capacity: usize, lanes: bool, output: MimoPrefillOutput, mtp: bool,
    rows: [usize; 2]) -> bool {
    // Larger single-request chunks already split across the old row lanes.
    // Preserve those exact row partitions instead of silently joining them.
    lanes && !mtp && output == MimoPrefillOutput::LastRow
        && rows.into_iter().all(|n| n > 0 && n <= capacity && n < 2 * MIN_LANE_ROWS)
}

/// Only attention geometry changes under a head split. Rank 1 runs split
/// target layers exclusively; rank 0 also runs unsplit MTP attention in its
/// decode workspace. Its prefill workspace never runs MTP, but may run an
/// unsplit target layer, so shrink it only when every target layer is split.
pub(super) fn attention_workspace_geometry(rank: usize, ranks: usize, decode: bool,
    all_target_layers_split: bool) -> MimoAttentionWorkspace {
    cuteafd_loader::families::mimo_v2::admission::attention_workspace_geometry(
        rank, ranks, decode, all_target_layers_split)
}

/// Most rows the FP8 LM head program takes (MmaFp8Gemv's M tile).
pub(crate) const FP8_ROWS: i32 = 16;

/// Most rows a decode program reads the E4M3 qkv / o / dense FFN copies for
/// (sparkinfer `MIMO_FP8_ROWS`: one 16-row GEMV tile up to 16 rows, two above),
/// so DFlash verify steps of up to 32 rows stay off the BF16 projections.
pub(crate) const FP8_DECODE_ROWS: i32 = 32;

/// Most rows of one sequence an MTP drafting pass takes (older true rows
/// catch up in passes of their own first).
const MTP_STEP_ROWS: usize = 16;

/// KV splits of the full-attention decode program (its compiled maximum is 32).
const DECODE_SPLITS: i32 = 16;

/// Host tables of one step.
struct StepTables {
    decode: bool,
    positions: Vec<i64>,
    /// Full layers: paged record slot per row.
    slots: Vec<i64>,
    /// SWA layers: ring slot per row (ring * 256 + position % 256).
    ring_slots: Vec<i64>,
    /// Index of the first step row of each row's sequence.
    seq_first: Vec<i32>,
    /// Prefill: the sequence's pages; decode: one padded row per step row.
    page_table: Vec<i32>,
    table_stride: usize,
}

/// The previous layer's FFN output a decode segment starts from.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum Previous {
    /// Layer 0: the embedded rows.
    First,
    /// A dense MLP's output in `delta` (under a head split, plus the other GPU's partial).
    Dense,
    /// The Spark ranks' partials in this many intake planes (the segment reduces them).
    Planes(usize),
    /// Routed experts run on this GPU (local, streamed or skipped): their sum in `delta`.
    Delta,
}

/// Everything a captured decode segment bakes in besides the engine's persistent buffers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct GraphKey {
    layer: usize,
    rows: usize,
    table_stride: usize,
    previous: Previous,
    /// The last segment ends in the LM head and the greedy selection of every row.
    head: bool,
}

struct GraphExec<'a>(*mut c_void, &'a NativeLibrary);

impl Drop for GraphExec<'_> {
    fn drop(&mut self) {
        // SAFETY: the exec came from end_capture and is destroyed once.
        let _ = unsafe { self.1.cuda_graph_exec_destroy(self.0) };
    }
}

/// A sequence's full-attention pages (from the refcounted pool: full pages may be
/// shared with retained prefix snapshots and other sequences, which never write them),
/// its SWA ring and its length.
#[derive(Debug, Clone)]
pub(crate) struct MimoPlacement {
    pub pages: Vec<u32>,
    pub ring: i32,
    pub len: usize,
}

impl MimoPlacement {
    pub fn slot(&self, position: usize) -> Result<i64> {
        let page = *self.pages.get(position / PAGE_ROWS).context("position past the sequence's pages")?;
        Ok(i64::from(page) * PAGE_ROWS as i64 + (position % PAGE_ROWS) as i64)
    }

    pub fn ring_slot(&self, position: usize) -> i64 {
        i64::from(self.ring) * RING_ROWS as i64 + (position % RING_ROWS) as i64
    }

    fn prefill_tables(&self, rows: usize, max_context: usize) -> Result<StepTables> {
        let end = self.len.checked_add(rows).context("paired prefill context overflow")?;
        ensure!(rows > 0 && end <= max_context && self.ring >= 0,
            "paired prefill exceeds context capacity or has no request ring");
        let pages = self.pages.get(..end.div_ceil(PAGE_ROWS))
            .context("paired prefill exceeds the request's admitted pages")?;
        Ok(StepTables {
            decode: false,
            positions: (self.len..end).map(|p| p as i64).collect(),
            slots: (self.len..end).map(|p| self.slot(p)).collect::<Result<_>>()?,
            ring_slots: (self.len..end).map(|p| self.ring_slot(p)).collect(),
            seq_first: vec![0; rows],
            page_table: pages.iter().map(|&page| page as i32).collect(),
            table_stride: 0,
        })
    }
}

/// Pages of the full-attention pool (the refcounted pool the prefix cache shares) and
/// free SWA rings, for callers without a prefix cache (the golden command).
pub(crate) struct Allocator {
    pages: cuteafd_engine::prefix::RefPagePool,
    rings: Vec<i32>,
}

impl Allocator {
    pub fn new(pages: usize, rings: usize) -> Self {
        Self { pages: cuteafd_engine::prefix::RefPagePool::new(pages, PAGE_ROWS), rings: (0..rings as i32).rev().collect() }
    }

    /// Reserves every page a sequence of up to `capacity` tokens needs, and a ring.
    pub fn admit(&mut self, capacity: usize) -> Result<MimoPlacement> {
        let pages = self.pages.alloc(self.pages.pages_for(capacity))?;
        let Some(ring) = self.rings.pop() else {
            self.pages.release(&pages);
            anyhow::bail!("SWA rings exhausted");
        };
        Ok(MimoPlacement { pages, ring, len: 0 })
    }

    /// A second sequence starting as `source`'s first `len` rows: full pages shared, the
    /// partial tail page copied into its own page by the caller (the returned copy).
    pub fn fork(&mut self, source: &MimoPlacement, len: usize, capacity: usize)
        -> Result<(MimoPlacement, Option<cuteafd_engine::prefix::TailCopy>)> {
        let fork = self.pages.fork(&source.pages, len, self.pages.pages_for(capacity))?;
        let Some(ring) = self.rings.pop() else {
            self.pages.release(&fork.pages);
            anyhow::bail!("SWA rings exhausted");
        };
        Ok((MimoPlacement { pages: fork.pages, ring, len: 0 }, fork.copy))
    }

    /// Returns a finished sequence's pages and ring.
    pub fn release(&mut self, placement: MimoPlacement) {
        self.pages.release(&placement.pages);
        self.rings.push(placement.ring);
    }
}

struct Workspace<'a> {
    rows: usize,
    logits_rows: usize,
    h: Dev<'a>,
    x: Dev<'a>,
    query: Dev<'a>,
    attn: Dev<'a>,
    delta: Dev<'a>,
    kv_step: Dev<'a>,
    /// 8-bit KV prefill: BF16 copy of one sequence's full-attention records (`max_context` rows).
    kv_wide: Rc<Dev<'a>>,
    positions: Dev<'a>,
    slots: Dev<'a>,
    step_slots: Dev<'a>,
    ring_slots: Dev<'a>,
    seq_first: Dev<'a>,
    page_table: Dev<'a>,
    scratch: Dev<'a>,
    logits: Dev<'a>,
    router_logits: Dev<'a>,
    route_ids: Dev<'a>,
    route_weights: Dev<'a>,
    wire: Dev<'a>,
    /// Spark exchange: rank partial planes, a zero shared-expert plane (MiMo
    /// has none), and pinned staging for routes, wire rows and partials.
    zero_plane: Dev<'a>,
    router_host: RefCell<HostAllocation<'a>>,
    /// The step's token ids (U32), gathered from the device embedding table.
    ids: Dev<'a>,
    /// Greedy selection of logits rows (MTP drafts): U32 ids, then U32 statuses.
    select: Dev<'a>,
    /// The cuBLAS LM head (rank 0 only).
    head: Option<VocabularyHead<'a>>,
    _head_workspace: Dev<'a>,
}

/// The second GPU of a two-GPU head split (see `MimoV2Config::head_split`):
/// its share of every layer (half the query heads with the KV heads they read,
/// half the dense MLP), its KV records, RoPE tables and workspaces.
pub(crate) struct Peer<'a> {
    pub device: i32,
    pub stream: *mut c_void,
    pub layers: Vec<MimoLayer<'a>>,
    kv: Vec<Rc<Allocation<'a>>>,
    cos_sin_full: Dev<'a>,
    cos_sin_swa: Dev<'a>,
    workspace: RefCell<Option<Workspace<'a>>>,
    /// The earlier prefill lanes' workspaces (see [`MimoEngine::step_lanes`]).
    lane_workspaces: RefCell<Vec<Workspace<'a>>>,
    prefill_kv_wide: RefCell<Option<Rc<Dev<'a>>>>,
    decode_workspace: RefCell<Option<Workspace<'a>>>,
    /// Rank 1's decode segments (see [`MimoEngine::decode_layers`]).
    graphs: RefCell<HashMap<GraphKey, GraphExec<'a>>>,
}

/// Exchange slot of layer `index`'s attention partials (`ffn` false) or its
/// FFN partials / routed-expert sum: by layer parity, so a push for layer `l`
/// never lands on rows the other GPU may still read (it consumed layer `l - 2`'s
/// before it sent layer `l - 1`'s attention partial, which this GPU waited for).
fn slot(index: usize, ffn: bool) -> usize {
    lane_slot(index, ffn, 0)
}

/// Each prefill lane retains its own receive rows until that lane consumes them.
fn lane_slot(index: usize, ffn: bool, lane: usize) -> usize {
    4 * lane + 2 * (index % 2) + usize::from(ffn)
}

/// cos | sin of position * theta^(-2i/dim) for `max_context` positions, FP32
/// like the reference's inv_freq, on the current device.
fn rope_table<'a>(library: &'a NativeLibrary, dim: usize, theta: f64, max_context: usize) -> Result<Dev<'a>> {
    let _memory_scope = cuteafd_ffi::memory_ledger::scope("kv/rope");
    let inv: Vec<f32> = (0..dim / 2).map(|i| 1.0 / (theta as f32).powf((2 * i) as f32 / dim as f32)).collect();
    let mut values = vec![0f32; max_context * dim];
    for p in 0..max_context {
        for (i, f) in inv.iter().enumerate() {
            let angle = p as f32 * f;
            values[p * dim + i] = angle.cos();
            values[p * dim + dim / 2 + i] = angle.sin();
        }
    }
    let allocation = DeviceAllocation::new(library, values.len() * 4)?;
    library.copy_h2d(allocation.buffer, bytes_of(&values))?;
    Ok(allocation)
}

pub(crate) struct MimoEngine<'a> {
    terminal: Cell<crate::shared::peer_split::TerminalState>,
    /// Closing/draining is also normal retirement. Only a failed submission
    /// latches this bit, so a caller cannot swallow a terminal execution error.
    submission_failed: Cell<bool>,
    quantize_grid: Fp8QuantizeGrid,
    pub library: &'a NativeLibrary,
    pub programs: &'a Programs<'a>,
    pub cfg: MimoV2Config,
    pub weights: MimoWeights<'a>,
    pub stream: *mut c_void,
    pub max_context: usize,
    pub prefill_rows: usize,
    prefill_output: MimoPrefillOutput,
    pub pages: usize,
    pub rings: usize,
    /// Program family of the checkpoint's geometry (`mimo`, `mimop`).
    family: &'static str,
    /// This engine's GPU (rank 0 of a head split).
    pub device: i32,
    /// Program family of one GPU's share under a head split (`mimop2`): the
    /// layers' producer, attention, o_proj and dense MLP programs.
    split_family: Option<&'static str>,
    /// The head split's second GPU and both ends of its exchange (rank 0's, the peer's).
    peer: Option<Peer<'a>>,
    exchange: Option<PeerExchange<'a>>,
    /// Per layer: the paged record pool (full) or the rings (SWA).
    kv: Vec<Rc<Allocation<'a>>>,
    cos_sin_full: Dev<'a>,
    cos_sin_swa: Dev<'a>,
    workspace: RefCell<Option<Workspace<'a>>>,
    decode_workspace: RefCell<Option<Workspace<'a>>>,
    /// The earlier prefill lanes (the last one uses `workspace`); the first
    /// admits its own last-row head output for independent request pairs.
    lane_workspaces: RefCell<Vec<Workspace<'a>>>,
    prefill_kv_wide: RefCell<Option<Rc<Dev<'a>>>>,
    experts: Option<Experts<'a>>,
    pub expert_input: ExpertInput,
    /// Host time per phase: GPU wait before the expert request, the Spark exchange.
    pub profile: RefCell<[f64; 2]>,
    /// `CUTEAFD_MIMO_WAVE_TIMING=1`: one line per Spark wave with its host
    /// phases (diagnostics; read once at construction).
    wave_timing: bool,
    /// The DFlash drafter (V2.6 Pro's dflash/): every step taps its target layers.
    pub drafter: Option<super::dflash::MimoDrafter<'a>>,
    /// L2 prefetch of the next layer's weights during decode exchanges.
    pub l2: Option<crate::shared::l2_prefetch::L2Prefetch>,
    /// The native MTP drafter: every step taps the last layer's rows.
    pub mtp: Option<super::mtp::MtpDrafter<'a>>,
    /// The token embedding table (resident on this GPU or read from its shard).
    pub embedding: TokenEmbedding<'a>,
    /// Pinned staging of queued MTP passes' tables (async uploads) and its fill level.
    mtp_staging: RefCell<(HostAllocation<'a>, usize)>,
    /// Prefill qkv and dense-FFN projections run W8A8 (E4M3 activations per
    /// row and 128-K block, the official FP8 release's served numerics); false:
    /// W8A16 (bitwise the former BF16 prefill over dequantized weights).
    pub prefill_w8a8: bool,
    /// Kernel/activation choice over the same immutable FP8 output weight.
    pub output_fp8_decode: bool,
    /// KV record format of every layer (and the MTP rings).
    kv_cache: MimoKvCache,
    /// Decode and verify steps replay one captured graph per layer segment
    /// between the Spark exchanges ([`Self::decode_layers`]); false: eager launches.
    pub decode_graphs: bool,
    /// Rank 0's decode segments, keyed by everything they bake in.
    graphs: RefCell<HashMap<GraphKey, GraphExec<'a>>>,
    /// Frozen preallocation bound for each physical rank; graph storage has
    /// its own contract, separate from modules/libraries/constraints.
    pub(super) graph_storage_plan: Option<cuteafd_loader::families::mimo_v2::decode_graph::MimoDecodeGraphPlan>,
    pub(super) graph_storage_bound_bytes: Option<Vec<u64>>,
    graph_storage_observed_bytes: [Cell<u64>;2],
    /// Graphs captured while serving (after [`Self::capture_decode_graphs`]): each is a
    /// shape the startup capture missed.
    late_captures: Cell<usize>,
}

fn bytes_of<T: Copy>(values: &[T]) -> &[u8] {
    // SAFETY: plain-old-data slices viewed as bytes for host->device copies.
    unsafe { std::slice::from_raw_parts(values.as_ptr().cast(), std::mem::size_of_val(values)) }
}

/// `[rows]`, plus the decode programs' `fp8_rows` (`FP8_DECODE_ROWS` when the layer has the FP8 copy, else 0).
fn fp8_scalars(rows: Scalar, decode: bool, fp8: bool) -> Vec<Scalar> {
    let mut scalars = vec![rows];
    if decode {
        scalars.push(Scalar::I32(if fp8 { FP8_DECODE_ROWS } else { 0 }));
    }
    scalars
}

/// An FP8-only weight's scales: row major for decode programs, K-block major for prefill ones.
fn scale_operand(name: &str, decode: bool) -> Result<&'static str> {
    Ok(match (name, decode) {
        ("w_qkv", true) => "w_qkv_scale",
        ("w_qkv", false) => "w_qkv_kscale",
        ("w_gate_up", true) => "w_gate_up_scale",
        ("w_gate_up", false) => "w_gate_up_kscale",
        ("w_down", true) => "w_down_scale",
        ("w_down", false) => "w_down_kscale",
        ("w_o", true) => "w_o_scale",
        ("w_o", false) => "w_o_kscale",
        _ => anyhow::bail!("unknown MiMo FP8 weight {name}: no scale operand is registered"),
    })
}

fn scale(name: &str, decode: bool, layer: &MimoLayer<'_>) -> Result<(&'static str, *mut c_void)> {
    let operand = scale_operand(name, decode)?;
    Ok((operand, layer.ptr(operand)?))
}

fn kind(attention: MimoAttention) -> &'static str {
    match attention {
        MimoAttention::Full => "full",
        MimoAttention::Sliding => "swa",
    }
}

impl<'a> MimoEngine<'a> {
    #[allow(clippy::too_many_arguments)]
    pub fn new(library: &'a NativeLibrary, programs: &'a Programs<'a>, cfg: MimoV2Config,
        weights: MimoWeights<'a>, stream: *mut c_void, max_context: usize, prefill_rows: usize, pages: usize,
        rings: usize, embedding: TokenEmbedding<'a>, kv_cache: MimoKvCache,
        prefill_output: MimoPrefillOutput) -> Result<Self> {
        let _memory_scope = cuteafd_ffi::memory_ledger::scope("kv");
        let family = cfg.program_family()?;
        // Layers loaded as head-split shares carry half the heads (see `attach_peer`).
        let ranks = if weights.layers.iter().any(|l| l.split) { 2 } else { 1 };
        let share = cfg.head_split(ranks)?;
        let split_family = if ranks > 1 { Some(share.program_family()?) } else { None };
        let device = library.cuda_get_device()?;
        let quantize_grid = Fp8QuantizeGrid::new(library.sm_count()?, None)?;
        ensure!(embedding.hidden() == cfg.hidden, "embedding rows of {} for hidden {}", embedding.hidden(), cfg.hidden);
        ensure!(cfg.rope_dim == 64 && cfg.head_dim == 192 && cfg.v_head_dim == 128 && cfg.window <= RING_ROWS - DECODE_ROWS,
            "the mimo programs are built for 192/128 heads, 64 RoPE dims and a window of at most {}",
            RING_ROWS - DECODE_ROWS);
        let zeroed = |bytes: usize| -> Result<Rc<Allocation<'a>>> {
            let _memory_scope = cuteafd_ffi::memory_ledger::scope("kv");
            let allocation = Rc::new(Allocation::new(Device { library, id: device }, bytes.max(256))?);
            library.cuda_zero_bytes(allocation.buffer, allocation.buffer.bytes)?;
            Ok(allocation)
        };
        let kv = weights.layers.iter().map(|layer| {
            let record = if layer.split { &share } else { &cfg }.record_bytes(layer.attention, kv_cache);
            zeroed(match layer.attention {
                MimoAttention::Full => pages * PAGE_ROWS * record,
                MimoAttention::Sliding => rings * RING_ROWS * record,
            })
        }).collect::<Result<Vec<_>>>()?;
        let table = |theta: f64| rope_table(library, cfg.rope_dim, theta, max_context);
        let (cos_sin_full, cos_sin_swa) = (table(cfg.full_rope_theta)?, table(cfg.swa_rope_theta)?);
        Ok(Self { terminal: Cell::new(crate::shared::peer_split::TerminalState::Active),
            submission_failed: Cell::new(false), quantize_grid, library, programs, cfg, weights, stream,
            max_context, prefill_rows, prefill_output, pages, rings, family, device,
            split_family, peer: None, exchange: None, kv, cos_sin_full,
            cos_sin_swa, workspace: RefCell::new(None), decode_workspace: RefCell::new(None),
            lane_workspaces: RefCell::new(Vec::new()), prefill_kv_wide: RefCell::new(None), experts: None,
            expert_input: ExpertInput::Fp8,
            profile: RefCell::new([0.0; 2]),
            wave_timing: std::env::var("CUTEAFD_MIMO_WAVE_TIMING").is_ok_and(|v| v == "1"), drafter: None, mtp: None, l2: None, embedding,
            mtp_staging: RefCell::new((HostAllocation::new(library, 1 << 20)?, 0)), prefill_w8a8: true,
            output_fp8_decode: true, kv_cache,
            decode_graphs: false, graphs: RefCell::new(HashMap::new()), graph_storage_plan: None, graph_storage_bound_bytes: None,
            graph_storage_observed_bytes: std::array::from_fn(|_| Cell::new(0)), late_captures: Cell::new(0) })
    }

    /// Layer `layer`'s KV storage: the paged record pool (full attention; page `p` holds
    /// rows at `p * PAGE_ROWS * record`) or the SWA rings (ring `r` at `r * RING_ROWS *
    /// record`), with its record bytes.
    pub(crate) fn kv_layer(&self, layer: usize) -> (MimoAttention, cuteafd_ffi::CuteafdDeviceBuffer, usize) {
        self.kv_layer_on(0, layer)
    }

    /// [`Self::kv_layer`] of rank `rank`'s share (0: this GPU, 1: the head split's peer).
    pub(crate) fn kv_layer_on(&self, rank: usize, layer: usize) -> (MimoAttention, cuteafd_ffi::CuteafdDeviceBuffer, usize) {
        let (layers, kv) = match (rank, &self.peer) {
            (1, Some(peer)) => (&peer.layers, &peer.kv),
            _ => (&self.weights.layers, &self.kv),
        };
        let attention = layers[layer].attention;
        (attention, kv[layer].buffer, self.record_bytes(&layers[layer]))
    }

    /// Strong allocation ownership for host snapshot DMA, independent of this
    /// engine's lifetime. Sliding rings are captured into separate mark arenas.
    pub(crate) fn full_kv_owners(&self) -> Vec<Rc<Allocation<'a>>> {
        let mut owners = Vec::new();
        for rank in 0..self.ranks() {
            let (layers, kv) = match (rank, &self.peer) {
                (1, Some(peer)) => (&peer.layers, &peer.kv),
                _ => (&self.weights.layers, &self.kv),
            };
            owners.extend(layers.iter().zip(kv).filter(|(layer, _)| layer.attention == MimoAttention::Full)
                .map(|(_, owner)| Rc::clone(owner)));
        }
        owners
    }

    /// The KV record format.
    pub fn kv_cache(&self) -> MimoKvCache {
        self.kv_cache
    }

    /// Bytes of `layer`'s KV record on the GPU holding it (its KV heads only under a head split).
    fn record_bytes(&self, layer: &MimoLayer<'_>) -> usize {
        let heads = self.cfg.kv_heads(layer.attention) / if layer.split { 2 } else { 1 };
        self.cfg.record_bytes_of(heads, self.kv_cache.of(layer.attention))
    }

    /// GPUs this engine runs on: 2 under a head split.
    pub fn ranks(&self) -> usize {
        1 + usize::from(self.peer.is_some())
    }

    /// Attaches the head split's second GPU: `device` with `stream`, holding
    /// `layers` (every layer's rank-1 share, see `MimoLoader::model`). Enables
    /// peer access both ways, loads the programs there, and allocates its KV
    /// records, RoPE tables and both ends of the exchange.
    pub fn attach_peer(&mut self, device: i32, stream: *mut c_void, layers: Vec<MimoLayer<'a>>, spark: bool) -> Result<()> {
        let _memory_scope = cuteafd_ffi::memory_ledger::scope("transport/peer-split");
        ensure!(self.split_family.is_some() && layers.len() == self.weights.layers.len()
            && layers.iter().all(|l| l.split), "attach_peer needs the head-split shares of every loaded layer");
        let library = self.library;
        let rows = self.prefill_rows.max(DECODE_ROWS);
        let exchange = PeerExchange::new_abortable(library, [RankDevice { device: self.device, stream: self.stream },
            RankDevice { device, stream }], cuteafd_loader::families::mimo_v2::admission::peer_slots(super::admission::transport_lanes(
                spark, &self.cfg)?),
            rows * self.cfg.hidden * 2)?;
        let zeroed = |bytes: usize| -> Result<Rc<Allocation<'a>>> {
            let allocation = Rc::new(Allocation::new(Device { library, id: device }, bytes.max(256))?);
            library.cuda_zero_bytes(allocation.buffer, allocation.buffer.bytes)?;
            Ok(allocation)
        };
        let peer = exchange.on(1, || -> Result<Peer<'a>> {
            let selected = cuteafd_core::coordinator_programs::CoordinatorPrograms {
                family: self.family, split_family: self.split_family,
            };
            self.programs.load_matching(|name| selected.contains(name))?;
            let _kv_scope = cuteafd_ffi::memory_ledger::scope("kv/peer");
            let kv = layers.iter().map(|layer| {
                let record = self.record_bytes(layer);
                zeroed(match layer.attention {
                    MimoAttention::Full => self.pages * PAGE_ROWS * record,
                    MimoAttention::Sliding => self.rings * RING_ROWS * record,
                })
            }).collect::<Result<Vec<_>>>()?;
            let table = |theta: f64| rope_table(library, self.cfg.rope_dim, theta, self.max_context);
            Ok(Peer { device, stream, kv, cos_sin_full: table(self.cfg.full_rope_theta)?,
                cos_sin_swa: table(self.cfg.swa_rope_theta)?, layers, workspace: RefCell::new(None),
                lane_workspaces: RefCell::new(Vec::new()), prefill_kv_wide: RefCell::new(None), decode_workspace: RefCell::new(None), graphs: RefCell::new(HashMap::new()) })
        })?;
        self.peer = Some(peer);
        self.exchange = Some(exchange);
        Ok(())
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
        let Some(peer) = self.peer.as_ref().filter(|_| rank == 1) else { return body() };
        self.library.cuda_set_device(peer.device)?;
        let out = body();
        self.library.cuda_set_device(self.device)?;
        out
    }

    pub(crate) fn is_terminal(&self) -> bool {
        self.terminal.get() != crate::shared::peer_split::TerminalState::Active
    }

    pub(super) fn submission_failed(&self) -> bool { self.submission_failed.get() }

    pub(crate) fn retain_queued_storage(&self) -> bool {
        self.terminal.get() == crate::shared::peer_split::TerminalState::Retained
    }

    /// A callback-owned copy queue failed after engine compute was drained.
    /// Escalate ownership only; never reset counters or make an engine reusable.
    pub(super) fn retain_serving_storage(&self) {
        self.terminal.set(crate::shared::peer_split::TerminalState::Retained);
        if let Some(exchange) = &self.exchange { exchange.finish_terminal(false); }
    }

    pub(crate) fn require_live(&self) -> Result<()> {
        let state=self.terminal.get();
        if state!=crate::shared::peer_split::TerminalState::Active {
            return Err(crate::shared::peer_split::TerminalPeerError::Closed(state).into());
        }
        match &self.exchange { Some(exchange) => exchange.require_live(), None => Ok(()) }
    }

    pub(crate) fn terminal_error(&self) -> anyhow::Error {
        let state = self.terminal.get();
        crate::shared::peer_split::TerminalPeerError::Closed(state).into()
    }

    /// Abort and drain before returning a submission error to a caller that may
    /// release its pages or tear down the engine. This protects engine-owned
    /// queues; serving also retains its sampler/prefix/host owners independently.
    /// A terminal engine can never execute another submission.
    pub(super) fn submit<T>(&self, body: impl FnOnce() -> Result<T>) -> Result<T> {
        if let Err(error) = self.require_live() {
            self.submission_failed.set(true);
            return Err(error);
        }
        let result = body();
        match result {
            Err(cause) => {
                self.submission_failed.set(true);
                let drain = match self.terminal_shutdown() {
                    Ok(()) => "queued streams drained; engine cannot be reused".to_string(),
                    Err(error) => format!("queued owners retained: {error:#}"),
                };
                Err(TerminalStepError { cause, drain }.into())
            }
            result => result,
        }
    }

    /// Close the complete engine before its buffers are released, including
    /// ordinary one-GPU errors. Never reset a wave before QPs quiesce; never
    /// release its pinned/landing owners until all compute and copy streams
    /// drain. Publication failure skips potentially blocked compute drains.
    pub(crate) fn terminal_shutdown(&self) -> Result<()> {
        use crate::shared::peer_split::{TerminalPeerError,TerminalState};
        match self.terminal.get() {
            TerminalState::Drained=>return Ok(()),
            TerminalState::Active=>self.terminal.set(TerminalState::Aborting),
            state=>return Err(TerminalPeerError::Closed(state).into()),
        }
        let mut failures=Vec::new();
        let published=match &self.exchange {
            Some(exchange)=>match exchange.publish_abort() {
                Ok(())=>true,
                Err(error)=>{ failures.push(format!("peer abort publication: {error:#}")); false }
            },
            None=>true,
        };
        if let Err(error)=self.terminal_links(|link|link.terminal_quiesce()) {
            failures.push(format!("Spark QP quiescence: {error:#}"));
        }
        let mut compute_drained = false;
        if published {
            let result=match &self.exchange {
                Some(exchange)=>exchange.drain_compute(),
                None=>self.on(0, || {
                    // SAFETY: the engine retains its stream and every queued
                    // buffer; one-GPU work has no unmatched peer-wait kernel.
                    unsafe { self.library.cuda_stream_synchronize(self.stream) }
                }),
            };
            match result {
                Ok(()) => compute_drained = true,
                Err(error) => failures.push(format!("compute drainage: {error:#}")),
            }
        }
        if compute_drained {
            if let Err(error)=self.terminal_links(|link|link.terminal_drain_copies()) {
                failures.push(format!("Spark upload drainage: {error:#}"));
            }
        } else {
            // A copy stream may wait on a compute event that can no longer
            // become visible. Do not replace a failed abort/drain with another
            // blocking synchronization; retain every upload/landing owner.
            failures.push("Spark upload drainage skipped because compute completion was not proven".into());
        }
        if failures.is_empty() {
            if let Err(error)=self.terminal_links(|link|link.terminal_release()) {
                failures.push(format!("Spark owner release: {error:#}"));
            }
        }
        let drained=failures.is_empty();
        self.terminal.set(if drained { TerminalState::Drained } else { TerminalState::Retained });
        if let Some(exchange)=&self.exchange { exchange.finish_terminal(drained); }
        if drained { Ok(()) } else { Err(TerminalPeerError::Retain(failures.join("; ")).into()) }
    }

    fn terminal_links(&self, mut action: impl FnMut(&mut SparkLink<'a>)->Result<()>) -> Result<()> {
        let mut failures=Vec::new();
        if let Some(Experts::Spark { transport,lanes,.. })=&self.experts {
            for (index,link) in std::iter::once(transport).chain(lanes.iter()).enumerate() {
                let result=link.try_borrow_mut().map_err(anyhow::Error::from)
                    .and_then(|mut link|action(&mut link));
                if let Err(error)=result { failures.push(format!("lane {index}: {error:#}")); }
            }
        }
        ensure!(failures.is_empty(),"{}",failures.join("; "));
        Ok(())
    }

    /// Receive slot `slot` of rank `rank` (null without a head split).
    fn recv(&self, rank: usize, slot: usize) -> *mut c_void {
        self.exchange.as_ref().and_then(|e| e.recv(rank, slot).ok()).unwrap_or(std::ptr::null_mut())
    }

    fn exchange(&self) -> Result<&PeerExchange<'a>> {
        self.exchange.as_ref().context("no head-split exchange")
    }

    /// Queues on `from`'s stream: push `bytes` of `source` into the other GPU's slot `slot`.
    fn push(&self, from: usize, slot: usize, source: *const c_void, bytes: usize) -> Result<()> {
        self.exchange()?.push(from, slot, source, bytes)
    }

    /// Queues on `at`'s stream: wait for the other GPU's next push into slot `slot`.
    fn wait(&self, at: usize, slot: usize) -> Result<()> {
        self.exchange()?.wait(at, slot)
    }


    /// Serves MoE layers from `experts` (without, the engine stops at the first MoE layer).
    pub fn set_experts(&mut self, experts: Experts<'a>) {
        self.experts = Some(experts);
    }

    pub fn has_experts(&self) -> bool {
        self.experts.is_some()
    }

    /// Borrows the immutable target head, including its exact native layout.
    /// Every queued head launch is drained while the engine still owns it.
    pub fn head(&self) -> BorrowedHead<'_, 'a> {
        BorrowedHead { weight: &self.weights.head, programs: self.programs, family: self.family,
            hidden: self.cfg.hidden, vocab: self.cfg.vocab_size }
    }

    fn alloc(&self, bytes: usize) -> Result<Dev<'a>> {
        DeviceAllocation::new(self.library, bytes.max(256))
    }

    /// `mimo_*` program names of this checkpoint's program family (`mimo` for
    /// V2 Flash, `mimop` for V2.6 Pro).
    fn program_name(&self, name: &str, split: bool) -> String {
        let family = if split { self.split_family.unwrap_or(self.family) } else { self.family };
        match name.strip_prefix("mimo_") {
            Some(rest) => format!("{family}_{rest}"),
            None => name.to_string(),
        }
    }

    fn run(&self, name: &str, pointers: &[(&str, *mut c_void)], scalars: &[Scalar]) -> Result<()> {
        self.run_on(0, false, name, pointers, scalars)
    }

    /// Launches `name` on rank `rank`'s stream; `split` picks the head-split share's program.
    fn run_on(&self, rank: usize, split: bool, name: &str, pointers: &[(&str, *mut c_void)], scalars: &[Scalar])
        -> Result<()> {
        let name = &self.program_name(name, split);
        let selected = cuteafd_core::coordinator_programs::CoordinatorPrograms {
            family: self.family, split_family: self.split_family,
        };
        ensure!(selected.contains(name), "MiMo program {name} is outside its preloaded family set");
        let names: Vec<&str> = pointers.iter().map(|(n, _)| *n).collect();
        let program = self.programs.program(name, &names)?;
        let raw: Vec<*mut c_void> = pointers.iter().map(|(_, p)| *p).collect();
        // SAFETY: every pointer names a live allocation of rank `rank`'s GPU sized for
        // the rows in `scalars`; that rank's stream orders all its launches.
        self.on(rank, || unsafe { program.launch(&raw, scalars, self.stream_of(rank)) })
            .with_context(|| format!("{name} with {scalars:?}"))
    }

    /// Rank `rank`'s workspace for steps of up to `t` rows (rank 1 holds the
    /// attention-side buffers only); allocated on that rank's GPU.
    fn workspace(&self, rank: usize, t: usize, decode: bool) -> Result<Workspace<'a>> {
        self.on(rank, || self.workspace_here(rank, t, decode, true))
    }

    /// All consumers use this rank's compute stream. A lane's next widen is
    /// ordered after the prior attention reads; no transport owns this arena.
    /// Every workspace retains an Rc, and published storage is never resized.
    fn shared_prefill_kv_wide(&self, rank: usize, bytes: usize) -> Result<Rc<Dev<'a>>> {
        let _memory_scope = cuteafd_ffi::memory_ledger::scope("workspace/kv-shadow");
        let cell = if rank == 0 { &self.prefill_kv_wide }
            else { &self.peer.as_ref().context("prefill KV shadow peer")?.prefill_kv_wide };
        let mut owned = cell.borrow_mut();
        if let Some(arena) = owned.as_ref() {
            MimoPrefillKvShadowPlan::require_same_extent(arena.buffer.bytes as u64, bytes as u64)?;
            return Ok(Rc::clone(arena));
        }
        let arena = Rc::new(self.alloc(bytes)?);
        *owned = Some(Rc::clone(&arena));
        Ok(arena)
    }

    /// `head`: the LM head and its logits (only the first row lane of a
    /// pipelined prefill skips them).
    fn workspace_here(&self, rank: usize, t: usize, decode: bool, head: bool) -> Result<Workspace<'a>> {
        let _memory_scope = cuteafd_ffi::memory_ledger::scope("workspace");
        let h = self.cfg.hidden;
        let lead = rank == 0;
        let with_head = lead && head;
        let scratch = super::admission::workspace_native_scratch(&self.cfg, self.programs,
            if self.split_family.is_some() { 2 } else { 1 }, rank, decode, self.kv_cache, self.weights.output_fp8)?;
        let spark = lead && matches!(self.experts, Some(Experts::Spark { .. }));
        let attention = attention_workspace_geometry(rank, self.ranks(), decode,
            self.weights.layers.iter().all(|layer| layer.split));
        let options = MimoWorkspaceOptions {
            rows: t as u64, decode, lead, with_head, spark, max_context: self.max_context as u64,
            pool_pages: self.pages as u64, kv_cache: self.kv_cache,
            attention, prefill_output: self.prefill_output, native_scratch_bytes: scratch,
            head_workspace_bytes: VOCABULARY_HEAD_WORKSPACE as u64,
        };
        let layout = MimoWorkspaceLayout::new(&self.cfg, options)?;
        let size = |bytes| usize::try_from(bytes).context("MiMo workspace size does not fit this process");
        let head_workspace = self.alloc(size(layout.head_workspace)?)?;
        let identity: Vec<i64> = (0..t as i64).collect();
        let step_slots = self.alloc(size(layout.step_slots)?)?;
        self.library.copy_h2d(step_slots.buffer, bytes_of(&identity))?;
        Ok(Workspace {
            rows: t,
            logits_rows: size(layout.logits_rows)?,
            h: self.alloc(size(layout.h)?)?,
            x: self.alloc(size(layout.x)?)?,
            query: self.alloc(size(layout.query)?)?,
            attn: self.alloc(size(layout.attn)?)?,
            delta: self.alloc(size(layout.delta)?)?,
            kv_step: self.alloc(size(layout.kv_step)?)?,
            kv_wide: if options.shares_prefill_kv_wide() {
                self.shared_prefill_kv_wide(rank, size(layout.kv_wide)?)?
            } else { Rc::new(self.alloc(size(layout.kv_wide)?)?) },
            positions: self.alloc(size(layout.positions)?)?,
            slots: self.alloc(size(layout.slots)?)?,
            step_slots,
            ring_slots: self.alloc(size(layout.ring_slots)?)?,
            seq_first: self.alloc(size(layout.seq_first)?)?,
            page_table: self.alloc(size(layout.page_table)?)?,
            scratch: self.alloc(size(layout.scratch)?)?,
            logits: self.alloc(size(layout.logits)?)?,
            router_logits: self.alloc(size(layout.router_logits)?)?,
            route_ids: self.alloc(size(layout.route_ids)?)?,
            route_weights: self.alloc(size(layout.route_weights)?)?,
            wire: self.alloc(size(layout.wire)?)?,
            zero_plane: {
                let zero = self.alloc(size(layout.zero_plane)?)?;
                self.library.cuda_zero_bytes(zero.buffer, zero.buffer.bytes)?;
                zero
            },
            router_host: RefCell::new(HostAllocation::new(self.library, size(layout.router_host)?)?),
            ids: self.alloc(size(layout.ids)?)?,
            select: self.alloc(size(layout.select)?)?,
            // SAFETY: the workspace buffer lives in the same struct and drops after the head.
            head: if with_head {
                Some(unsafe { self.library.vocabulary_head_rows(head_workspace.buffer.ptr, h as u32, size(layout.logits_rows)? as u32,
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

    fn inject_media(&self, w: &Workspace<'_>, tables: &StepTables,
        media: Option<&cuteafd_engine::media::RequestMedia>) -> Result<()> {
        let Some(media) = media else { return Ok(()); };
        let start = tables.positions.first().copied().context("media prefill positions")? as usize;
        let mut chunk = cuteafd_engine::media::MediaChunk::default();
        media.write_chunk(start, start + tables.positions.len(), &mut chunk)?;
        if chunk.indices.is_empty() { return Ok(()); }
        // Reuse norm scratch and a table only before their consumers. The FFI adapter
        // drains the injection before restoring seq_first; no new device allocations.
        self.library.embedding_injection()?.inject_host(&chunk.features, &chunk.indices,
            w.x.buffer, w.seq_first.buffer, w.h.buffer, tables.positions.len(), self.cfg.hidden, 1, self.stream)?;
        self.put(&w.seq_first, &tables.seq_first)
    }

    /// Prefills a sequence from its length through every resident layer and
    /// returns the last row's logits when all layers are resident, with teacher forcing: after layer `l`, `forced(l)` (when it
    /// returns rows) replaces the residual before layer `l + 1`, so each
    /// layer's comparison measures that layer alone. `all_logits` returns
    /// every row's logits instead of the last row's.
    pub fn prefill_forced(&self, placement: &mut MimoPlacement, tokens: &[u32], all_logits: bool,
        on_layer: Option<&mut dyn FnMut(usize, &[u8]) -> Result<()>>,
        forced: Option<&dyn Fn(usize) -> Option<Vec<u8>>>) -> Result<Option<Vec<f32>>> {
        self.prefill_device(placement, tokens, all_logits, on_layer, forced)?
            .map(|logits| logits.to_host(self.library)).transpose()
    }

    /// [`Self::prefill_forced`] leaving the logits on the device.
    pub fn prefill_device(&self, placement: &mut MimoPlacement, tokens: &[u32], all_logits: bool,
        on_layer: Option<&mut dyn FnMut(usize, &[u8]) -> Result<()>>,
        forced: Option<&dyn Fn(usize) -> Option<Vec<u8>>>) -> Result<Option<DeviceLogits>> {
        self.prefill_media_device(placement, tokens, all_logits, on_layer, forced, None)
    }

    pub fn prefill_media_device(&self, placement: &mut MimoPlacement, tokens: &[u32], all_logits: bool,
        on_layer: Option<&mut dyn FnMut(usize, &[u8]) -> Result<()>>,
        forced: Option<&dyn Fn(usize) -> Option<Vec<u8>>>, media: Option<&cuteafd_engine::media::RequestMedia>)
        -> Result<Option<DeviceLogits>> {
        ensure!(!all_logits || self.prefill_output == MimoPrefillOutput::AllRows,
            "this MiMo engine admits last-row prefill logits; load an AllRows diagnostic engine for all_logits");
        let (t, start) = (tokens.len(), placement.len);
        let lanes = if self.lanes_ready() && !all_logits && on_layer.is_none() && forced.is_none() {
            prefill_lane_count(t, self.prefill_rows, self.transport_lanes())
        } else {
            1
        };
        let limit = lanes * self.prefill_rows;
        ensure!(t > 0 && t <= limit && start + t <= self.max_context, "prefill of {t} rows at {start}");
        let used = (start + t).div_ceil(PAGE_ROWS);
        let tables = |first: usize, end: usize| -> Result<StepTables> {
            Ok(StepTables {
                decode: false,
                positions: (first..end).map(|p| p as i64).collect(),
                slots: (first..end).map(|p| placement.slot(p)).collect::<Result<_>>()?,
                ring_slots: (first..end).map(|p| placement.ring_slot(p)).collect(),
                seq_first: vec![0; end - first],
                page_table: placement.pages[..used].iter().map(|&page| page as i32).collect(),
                table_stride: 0,
            })
        };
        let logits = if lanes > 1 {
            // Row lanes, each as a consecutive chunk would see the cache.
            let split = t.div_ceil(lanes);
            let bounds: Vec<(usize, usize)> = (0..lanes).map(|i| (i * split, ((i + 1) * split).min(t))).collect();
            let lane_tables = bounds.iter().map(|&(a, b)| tables(start + a, start + b)).collect::<Result<Vec<_>>>()?;
            let steps: Vec<(&StepTables, &[u32])> = lane_tables.iter().zip(&bounds)
                .map(|(tables, &(a, b))| (tables, &tokens[a..b])).collect();
            self.step_lanes(&steps, media)?
        } else {
            self.step(&tables(start, start + t)?, tokens, if all_logits { t } else { 1 }, on_layer, forced, media)?
        };
        placement.len += t;
        Ok(logits)
    }

    /// Rows one prefill step takes: the programs' rows per lane when prefill
    /// runs pipelined in row lanes ([`Self::step_lanes`]).
    pub fn prefill_capacity(&self) -> usize {
        lane_prefill_capacity(self.prefill_rows, self.transport_lanes())
    }

    /// Spark transports, one per prefill row lane (1: serial prefill).
    fn transport_lanes(&self) -> usize {
        match &self.experts {
            Some(Experts::Spark { lanes, .. }) => 1 + lanes.len(),
            _ => 1,
        }
    }

    /// Pipelined prefill is available with a second Spark lane transport,
    /// including when attention heads are split over both coordinator GPUs.
    fn lanes_ready(&self) -> bool {
        self.prefill_capacity() > self.prefill_rows
    }

    /// Pairing reuses the admitted Spark lanes for two chunks below 2048 rows,
    /// preserving each request's existing single-lane arithmetic. Larger chunks
    /// and MTP engines keep their original path. Attention tables stay separate.
    pub fn can_prefill_pair(&self, rows: [usize; 2]) -> bool {
        // MTP keeps its existing route until independent request pairing has
        // its own hidden-ring/catch-up qualification.
        independent_prefill_rows(self.prefill_rows, self.lanes_ready(), self.prefill_output, self.mtp.is_some(), rows)
    }

    pub fn full_prefill_logits(&self) -> bool {
        self.prefill_output == MimoPrefillOutput::AllRows
    }

    /// Allocate the diagnostic head before requests can be admitted.
    pub fn prepare_scoring_prefill(&self) -> Result<()> {
        if self.workspace.borrow().is_none() {
            *self.workspace.borrow_mut() = Some(self.workspace(0, self.prefill_rows, false)?);
        }
        if let Some(peer) = &self.peer {
            if peer.workspace.borrow().is_none() {
                *peer.workspace.borrow_mut() = Some(self.workspace(1, self.prefill_rows, false)?);
            }
        }
        Ok(())
    }

    pub fn prepare_prefill_pair(&mut self) -> Result<()> {
        if self.can_prefill_pair([1, 1]) {
            if let Some(drafter) = &mut self.drafter {
                drafter.prepare_prefill_lanes()?;
            }
        }
        Ok(())
    }

    /// Two independent prefill chunks. Both output pointers remain valid until
    /// the next prefill call; consume both before reusing either workspace.
    /// DFlash taps are separate banks, consumed with `update_lane(0/1)`.
    pub fn prefill_pair_device(&self, requests: [(&mut MimoPlacement, &[u32]); 2])
        -> Result<[Option<DeviceLogits>; 2]> {
        self.prefill_pair_media_device(requests, [None, None])
    }

    pub fn prefill_pair_media_device(&self, requests: [(&mut MimoPlacement, &[u32]); 2],
        media: [Option<&cuteafd_engine::media::RequestMedia>; 2]) -> Result<[Option<DeviceLogits>; 2]> {
        let [(first, first_tokens), (second, second_tokens)] = requests;
        ensure!(self.can_prefill_pair([first_tokens.len(), second_tokens.len()]),
            "independent prefill pair exceeds the admitted lanes");
        ensure!(first.ring != second.ring, "independent prefill requests must own distinct rings");
        let tables = [first.prefill_tables(first_tokens.len(), self.max_context)?,
            second.prefill_tables(second_tokens.len(), self.max_context)?];
        let logits = self.submit(|| self.step_lanes_inner(&[
            (&tables[0], first_tokens), (&tables[1], second_tokens)], true, &media))?;
        let logits: [Option<DeviceLogits>; 2] = logits.try_into()
            .map_err(|_| anyhow::anyhow!("an independent prefill pair returns two outputs"))?;
        first.len += first_tokens.len();
        second.len += second_tokens.len();
        Ok(logits)
    }

    /// A prefill step in two row lanes, layer by layer: lane `i` lands its
    /// previous layer's experts, then queues this layer's attention and router
    /// and sends its wave, so the GPU works on one lane while the other's
    /// wave is out. Each lane is exactly a consecutive prefill chunk (lane 0
    /// writes its KV before lane 1 reads it at every layer). Returns the last
    /// row's logits.
    fn step_lanes(&self, lanes: &[(&StepTables, &[u32])], media: Option<&cuteafd_engine::media::RequestMedia>) -> Result<Option<DeviceLogits>> {
        let media = vec![media; lanes.len()];
        self.submit(|| self.step_lanes_inner(lanes, false, &media).map(|mut logits| logits.pop().flatten()))
    }

    fn step_lanes_inner(&self, lanes: &[(&StepTables, &[u32])], independent: bool,
        media: &[Option<&cuteafd_engine::media::RequestMedia>])
        -> Result<Vec<Option<DeviceLogits>>> {
        let Some(Experts::Spark { transport, lanes: links, runtime }) = &self.experts else {
            anyhow::bail!("pipelined prefill needs Spark experts with lane transports");
        };
        let n = lanes.len();
        ensure!(n >= 2 && n <= links.len() + 1, "{n} prefill lanes need {} lane transports", n.saturating_sub(1));
        let last = n - 1;
        // The first earlier lane admits a last-row head for independent pairs.
        let first_head = self.prefill_output == MimoPrefillOutput::LastRow && self.mtp.is_none();
        while self.lane_workspaces.borrow().len() < last {
            let head = first_head && self.lane_workspaces.borrow().is_empty();
            let workspace = self.on(0, || self.workspace_here(0, self.prefill_rows, false, head))?;
            self.lane_workspaces.borrow_mut().push(workspace);
        }
        if self.workspace.borrow().is_none() {
            *self.workspace.borrow_mut() = Some(self.on(0, || self.workspace_here(0, self.prefill_rows, false, true))?);
        }
        let (earlier, main) = (self.lane_workspaces.borrow(), self.workspace.borrow());
        let w: Vec<&Workspace<'_>> = earlier[..last].iter()
            .chain(std::iter::once(main.as_ref().context("workspace")?)).collect();
        let t: Vec<usize> = lanes.iter().map(|lane| lane.1.len()).collect();
        ensure!(t.iter().zip(&w).all(|(&t, w)| t <= w.rows), "prefill lanes exceed their workspaces");
        let rows: Vec<Scalar> = t.iter().map(|&t| Scalar::I32(t as i32)).collect();
        // SAFETY: the engine owns this stream; the previous step must be done reading the tables.
        unsafe { self.library.cuda_stream_synchronize(self.stream)? };
        for (i, (tables, tokens)) in lanes.iter().enumerate() {
            self.put(&w[i].positions, &tables.positions)?;
            self.put(&w[i].slots, &tables.slots)?;
            self.put(&w[i].ring_slots, &tables.ring_slots)?;
            self.put(&w[i].seq_first, &tables.seq_first)?;
            self.put(&w[i].page_table, &tables.page_table)?;
            self.embedding.embed(tokens, w[i].ids.buffer, 1, w[i].h.buffer, self.stream)?;
            self.inject_media(w[i], tables, media.get(i).copied().flatten())?;
        }
        let peer_workspaces = match &self.peer {
            Some(peer) => {
                while peer.lane_workspaces.borrow().len() < last {
                    let workspace = self.workspace(1, self.prefill_rows, false)?;
                    peer.lane_workspaces.borrow_mut().push(workspace);
                }
                if peer.workspace.borrow().is_none() {
                    *peer.workspace.borrow_mut() = Some(self.workspace(1, self.prefill_rows, false)?);
                }
                Some((peer.lane_workspaces.borrow(), peer.workspace.borrow()))
            }
            None => None,
        };
        let w1: Option<Vec<&Workspace<'_>>> = match &peer_workspaces {
            Some((earlier, main)) => Some(earlier[..last].iter()
                .chain(std::iter::once(main.as_ref().context("peer workspace")?)).collect()),
            None => None,
        };
        let w1 = w1.as_deref();
        if let Some(peers) = w1 {
            // SAFETY: the peer stream belongs to this engine; all previous waits have
            // matching pushes, and its old tables must drain before being overwritten.
            self.on(1, || unsafe { self.library.cuda_stream_synchronize(self.stream_of(1)) })?;
            for (i, (tables, _)) in lanes.iter().enumerate() {
                self.on(1, || {
                    self.put(&peers[i].positions, &tables.positions)?;
                    self.put(&peers[i].slots, &tables.slots)?;
                    self.put(&peers[i].ring_slots, &tables.ring_slots)?;
                    self.put(&peers[i].seq_first, &tables.seq_first)?;
                    self.put(&peers[i].page_table, &tables.page_table)
                })?;
                self.exchange()?.push_to(0, DIRECT, w[i].h.buffer.ptr, peers[i].h.buffer.ptr,
                    t[i] * self.cfg.hidden * 2)?;
            }
        }
        let layers = &self.weights.layers;
        let bf16_input = self.expert_input.bf16(false);
        let mut transports: Vec<_> = links[..last].iter().map(RefCell::borrow_mut)
            .chain(std::iter::once(transport.borrow_mut())).collect();
        let mut inflight: Vec<Option<(usize, SentWave)>> = (0..n).map(|_| None).collect();
        // After a lane's FFN output is in its `delta`: both GPUs' next input
        // norms, then each independent request's drafter taps on rank 0.
        // Consecutive chunks of one request keep the historical last-lane taps.
        let after_ffn = |i: usize, index: usize| -> Result<()> {
            let weight = match layers.get(index + 1) {
                Some(next) => next.ptr("input_norm")?,
                None => self.weights.norm.buffer.ptr,
            };
            if let Some(peers) = w1 {
                self.peer_lane_post(index, i, peers[i], rows[i])?;
            }
            let dense_split = w1.is_some() && layers[index].dense;
            let ffn = lane_slot(index, true, i);
            if dense_split {
                self.wait(0, ffn)?;
            }
            self.norm_on(0, w[i], weight, if dense_split { 2 } else { 1 }, rows[i],
                if dense_split { self.recv(0, ffn) } else { w[i].delta.buffer.ptr })?;
            if independent || i == last {
                if let Some(drafter) = &self.drafter {
                    let n = t[i].min(super::dflash::TAP_ROWS);
                    if independent {
                        drafter.tap_lane(i, index, w[i].h.buffer.ptr, t[i] - n, n)?;
                    } else {
                        drafter.tap(index, w[i].h.buffer.ptr, t[i] - n, n)?;
                    }
                }
            }
            // Independent MTP pairing is excluded; preserve the old single-
            // request path's final-lane hidden ring and catch-up semantics.
            if let (Some(mtp), true) = (&self.mtp, i == last && index + 1 == self.cfg.layers) {
                self.mtp_tap(mtp, w[last], lanes[last].0)?;
            }
            Ok(())
        };
        for i in 0..n {
            self.norm(w[i], layers[0].ptr("input_norm")?, 0, rows[i])?;
        }
        for (index, layer) in layers.iter().enumerate() {
            for i in 0..n {
                if let Some((previous, sent)) = inflight[i].take() {
                    let forward = (w1.is_some() && previous + 1 < layers.len())
                        .then_some(lane_slot(previous, true, i));
                    self.spark_land(w[i], previous, t[i], &mut transports[i], runtime, sent, forward, false)?;
                    after_ffn(i, previous)?;
                }
                if let Some(peers) = w1 {
                    // Both ranks queue (layer, lane) in the same order: lane 0's KV
                    // writes and SWA ring commit precede lane 1's reads. A peer's
                    // attention is queued only when its rank-0 partner can also be
                    // queued, so failed Spark dispatches leave no future peer wait.
                    self.peer_lane_attention(index, i, peers[i], rows[i], lanes[i].0,
                        t[i] * self.cfg.hidden * 2)?;
                }
                self.attention_on(0, w[i], self.kv[index].buffer.ptr, layer, rows[i], "m4096", lanes[i].0)?;
                if w1.is_some() {
                    let attended = lane_slot(index, false, i);
                    self.push(0, attended, w[i].delta.buffer.ptr, t[i] * self.cfg.hidden * 2)?;
                    self.wait(0, attended)?;
                    self.norm_on(0, w[i], layer.ptr("post_norm")?, 2, rows[i], self.recv(0, attended))?;
                } else {
                    self.norm(w[i], layer.ptr("post_norm")?, 1, rows[i])?;
                }
                if layer.dense {
                    self.dense_ffn(0, w[i], layer, rows[i], "m4096", false)?;
                    if w1.is_some() {
                        self.push(0, lane_slot(index, true, i), w[i].delta.buffer.ptr,
                            t[i] * self.cfg.hidden * 2)?;
                    }
                    after_ffn(i, index)?;
                } else {
                    self.moe_front(w[i], layer, t[i], bf16_input)?;
                    inflight[i] = Some((index, self.spark_send(w[i], index, t[i], false, bf16_input, false, &mut transports[i])?));
                }
            }
        }
        for i in 0..n {
            if let Some((previous, sent)) = inflight[i].take() {
                let forward = (w1.is_some() && previous + 1 < layers.len())
                    .then_some(lane_slot(previous, true, i));
                self.spark_land(w[i], previous, t[i], &mut transports[i], runtime, sent, forward, false)?;
                after_ffn(i, previous)?;
            }
        }
        if layers.len() < self.cfg.layers {
            // SAFETY: the engine owns both streams; every queued wait has a matching push.
            unsafe { self.library.cuda_stream_synchronize(self.stream)? };
            if w1.is_some() {
                self.on(1, || unsafe { self.library.cuda_stream_synchronize(self.stream_of(1)) })?;
            }
            return Ok((0..n).map(|_| None).collect());
        }
        let mut logits: Vec<Option<DeviceLogits>> = (0..n).map(|_| None).collect();
        for i in if independent { 0..n } else { last..n } {
            self.launch_head(w[i], t[i], 1, false)?;
            logits[i] = Some(self.device_logits(w[i], 1, false));
        }
        Ok(logits)
    }

    /// Appends each sequence's tokens (one for decode, several for a
    /// speculative verify) at its length in one decode-shaped step; returns
    /// every row's logits. A caller that rejects a suffix sets `len` back:
    /// the 256-slot rings keep every key a later step can still need.
    pub fn verify(&self, sequences: &mut [(&mut MimoPlacement, usize)], tokens: &[u32],
        on_layer: Option<&mut dyn FnMut(usize, &[u8]) -> Result<()>>) -> Result<Option<Vec<f32>>> {
        self.verify_device(sequences, tokens, on_layer)?.map(|logits| logits.to_host(self.library)).transpose()
    }

    /// [`Self::verify`] leaving every row's logits on the device.
    pub fn verify_device(&self, sequences: &mut [(&mut MimoPlacement, usize)], tokens: &[u32],
        on_layer: Option<&mut dyn FnMut(usize, &[u8]) -> Result<()>>) -> Result<Option<DeviceLogits>> {
        self.verify_media_device(sequences, tokens, on_layer, None)
    }

    /// A single-sequence diagnostic step may contain teacher-forced image rows.
    pub fn verify_media_device(&self, sequences: &mut [(&mut MimoPlacement, usize)], tokens: &[u32],
        on_layer: Option<&mut dyn FnMut(usize, &[u8]) -> Result<()>>,
        media: Option<&cuteafd_engine::media::RequestMedia>) -> Result<Option<DeviceLogits>> {
        ensure!(media.is_none() || sequences.len() == 1, "media verification requires one sequence");
        let rows: usize = sequences.iter().map(|(_, n)| n).sum();
        ensure!(rows > 0 && rows <= DECODE_ROWS && tokens.len() == rows, "decode step of {rows} rows");
        // With graphs one stride for every step (they bake it in): the pages of a full context.
        let floor = if self.decode_graphs { self.decode_stride() } else { 1 };
        let stride = sequences.iter().map(|(p, _)| p.pages.len()).max().unwrap_or(1).max(floor);
        let mut tables = StepTables { decode: true, positions: Vec::new(), slots: Vec::new(), ring_slots: Vec::new(),
            seq_first: Vec::new(), page_table: Vec::new(), table_stride: stride };
        for (placement, count) in sequences.iter() {
            let first = tables.positions.len() as i32;
            for position in placement.len..placement.len + count {
                ensure!(position < self.max_context, "decode at {position} past the context");
                tables.positions.push(position as i64);
                tables.slots.push(placement.slot(position)?);
                tables.ring_slots.push(placement.ring_slot(position));
                tables.seq_first.push(first);
                let mut pages: Vec<i32> = placement.pages.iter().map(|&page| page as i32).collect();
                pages.resize(stride, 0);
                tables.page_table.extend(pages);
            }
        }
        let logits = self.step(&tables, tokens, rows, on_layer, None, media)?;
        for (placement, count) in sequences.iter_mut() {
            placement.len += *count;
        }
        Ok(logits)
    }

    fn step(&self, tables: &StepTables, tokens: &[u32], logit_rows: usize,
        on_layer: Option<&mut dyn FnMut(usize, &[u8]) -> Result<()>>,
        forced: Option<&dyn Fn(usize) -> Option<Vec<u8>>>, media: Option<&cuteafd_engine::media::RequestMedia>) -> Result<Option<DeviceLogits>> {
        self.submit(|| self.step_inner(tables, tokens, logit_rows, on_layer, forced, media))
    }

    fn step_inner(&self, tables: &StepTables, tokens: &[u32], logit_rows: usize,
        mut on_layer: Option<&mut dyn FnMut(usize, &[u8]) -> Result<()>>,
        forced: Option<&dyn Fn(usize) -> Option<Vec<u8>>>, media: Option<&cuteafd_engine::media::RequestMedia>) -> Result<Option<DeviceLogits>> {
        let (h, t) = (self.cfg.hidden, tables.positions.len());
        let (cell, capacity) = if tables.decode { (&self.decode_workspace, DECODE_ROWS) } else { (&self.workspace, self.prefill_rows) };
        if cell.borrow().is_none() {
            *cell.borrow_mut() = Some(self.workspace(0, capacity, tables.decode)?);
        }
        let workspace = cell.borrow();
        let w = workspace.as_ref().context("workspace")?;
        ensure!(t <= w.rows && logit_rows <= t && logit_rows <= w.logits_rows && tokens.len() == t,
            "step exceeds the admitted workspace or logits rows");
        // The head split's second GPU: its workspace of the same shape.
        let peer_workspace = match &self.peer {
            Some(peer) => {
                let cell = if tables.decode { &peer.decode_workspace } else { &peer.workspace };
                if cell.borrow().is_none() {
                    *cell.borrow_mut() = Some(self.workspace(1, capacity, tables.decode)?);
                }
                Some((peer, cell.borrow()))
            }
            None => None,
        };
        let split = match &peer_workspace {
            Some((peer, ws)) => Some((*peer, ws.as_ref().context("peer workspace")?)),
            None => None,
        };
        // The tables and ids go up synchronously: the previous step (which no
        // longer ends in a logits download) must be done reading them.
        // SAFETY: the engine owns these streams; every exchange wait queued on
        // them was matched by a push the previous step queued, so they drain.
        unsafe {
            self.library.cuda_stream_synchronize(self.stream)?;
            if let Some((peer, _)) = split {
                self.library.cuda_stream_synchronize(peer.stream)?;
            }
        }
        for (rank, w) in std::iter::once(w).chain(split.map(|(_, w1)| w1)).enumerate() {
            self.on(rank, || {
                self.put(&w.positions, &tables.positions)?;
                self.put(&w.slots, &tables.slots)?;
                self.put(&w.ring_slots, &tables.ring_slots)?;
                self.put(&w.seq_first, &tables.seq_first)?;
                self.put(&w.page_table, &tables.page_table)
            })?;
        }
        if tables.decode && self.decode_graphs && on_layer.is_none() && forced.is_none() && media.is_none() && self.graphable() {
            // The first captured segment gathers the rows from the uploaded ids; the
            // last runs the head and the greedy selection.
            let gather = self.embedding.device_gather();
            if gather {
                self.embedding.check(tokens)?;
                self.put(&w.ids, tokens)?;
            } else {
                self.embedding.embed(tokens, w.ids.buffer, 1, w.h.buffer, self.stream)?;
            }
            let head = logit_rows == t;
            self.decode_layers(w, split.map(|(_, w1)| w1), tables, t, gather, head, true)?;
            if let Some(mtp) = &self.mtp {
                self.mtp_tap(mtp, w, tables)?;
            }
            if !head {
                self.launch_head(w, t, logit_rows, true)?;
            }
            return Ok(Some(self.device_logits(w, logit_rows, head)));
        }
        self.embedding.embed(tokens, w.ids.buffer, 1, w.h.buffer, self.stream)?;
        self.inject_media(w, tables, media)?;
        let rows = Scalar::I32(t as i32);
        let cap = if tables.decode { "m64" } else { "m4096" };
        let layers = &self.weights.layers;
        let bytes = t * h * 2;
        // Rank 1 needs nothing from the host once queued (every input is a push from
        // rank 0), so without teacher forcing (which rewrites its rows from the host
        // mid-step) its next layer is queued before rank 0 blocks in the Spark exchange:
        // once rank 0 forwards the routed sum, rank 1 runs on without waiting for the
        // host to queue its work.
        let ahead = forced.is_none();
        if let Some((_, w1)) = split {
            // The embedded rows to the second GPU's residual stream.
            self.exchange()?.push_to(0, DIRECT, w.h.buffer.ptr, w1.h.buffer.ptr, bytes)?;
            self.peer_layer(0, w1, rows, cap, tables, bytes)?;
        }
        self.norm(w, layers[0].ptr("input_norm")?, 0, rows)?;
        for (index, layer) in layers.iter().enumerate() {
            let last = index + 1 == layers.len();
            if let (Some((_, w1)), false, true) = (split, ahead, index > 0) {
                self.peer_layer(index, w1, rows, cap, tables, bytes)?;
            }
            // h += attention; x = post_attention_layernorm(h); under a head split both
            // GPUs add both partials in the same order (identical residual streams).
            self.attention_on(0, w, self.kv[index].buffer.ptr, layer, rows, cap, tables)?;
            let (attended, ffn) = (slot(index, false), slot(index, true));
            if split.is_some() {
                self.push(0, attended, w.delta.buffer.ptr, bytes)?;
                self.wait(0, attended)?;
                self.norm_on(0, w, layer.ptr("post_norm")?, 2, rows, self.recv(0, attended))?;
                if let (Some((_, w1)), true, false) = (split, ahead, last) {
                    self.peer_layer(index + 1, w1, rows, cap, tables, bytes)?;
                }
            } else {
                self.norm(w, layer.ptr("post_norm")?, 1, rows)?;
            }
            // How the next norm takes the FFN output: one delta, or this GPU's partial plus the other's.
            let mut deltas = 1;
            if layer.dense {
                self.dense_ffn(0, w, layer, rows, cap, tables.decode)?;
                if split.is_some() {
                    self.push(0, ffn, w.delta.buffer.ptr, bytes)?;
                    self.wait(0, ffn)?;
                    deltas = 2;
                }
            } else {
                // The routed experts' sum also goes to the second GPU (not after the last layer).
                let forward = (split.is_some() && !last).then_some(ffn);
                self.moe(w, index, layer, t, tables.decode, forward)?;
            }
            // h += ffn; x = next input_layernorm(h) (or the final norm).
            let weight = match layers.get(index + 1) {
                Some(next) => next.ptr("input_norm")?,
                None => self.weights.norm.buffer.ptr,
            };
            let second = if deltas == 2 { self.recv(0, ffn) } else { w.delta.buffer.ptr };
            self.norm_on(0, w, weight, deltas, rows, second)?;
            if let Some(drafter) = &self.drafter {
                // The step's last TAP_ROWS rows (a prefill's tail holds every
                // context row later drafts can see).
                let n = t.min(super::dflash::TAP_ROWS);
                drafter.tap(index, w.h.buffer.ptr, t - n, n)?;
            }
            if let (Some(mtp), true) = (&self.mtp, index + 1 == self.cfg.layers) {
                self.mtp_tap(mtp, w, tables)?;
            }
            if let Some(on_layer) = on_layer.as_mut() {
                on_layer(index, &self.download(&w.h, t * h * 2)?)?;
            }
            if let Some(rows_forced) = forced.and_then(|f| f(index)) {
                ensure!(rows_forced.len() == t * h * 2, "teacher-forced rows of the wrong size");
                self.library.copy_h2d(cuteafd_ffi::CuteafdDeviceBuffer { bytes: rows_forced.len(), ..w.h.buffer },
                    &rows_forced)?;
                self.norm(w, weight, 0, rows)?;
                if let (Some((peer, w1)), false) = (split, last) {
                    // SAFETY: the engine owns the peer stream (queued only through this layer:
                    // `ahead` is off); drained before the synchronous copy.
                    unsafe { self.library.cuda_stream_synchronize(peer.stream)? };
                    self.library.copy_h2d(cuteafd_ffi::CuteafdDeviceBuffer { bytes: rows_forced.len(), ..w1.h.buffer },
                        &rows_forced)?;
                    self.norm_on(1, w1, peer.layers[index + 1].ptr("input_norm")?, 0, rows, w1.delta.buffer.ptr)?;
                }
            }
            crate::shared::console::layer_mark(index);
        }
        if layers.len() < self.cfg.layers {
            // SAFETY: the engine owns this stream.
            unsafe { self.library.cuda_stream_synchronize(self.stream)? };
            return Ok(None);
        }
        self.launch_head(w, t, logit_rows, tables.decode)?;
        Ok(Some(self.device_logits(w, logit_rows, false)))
    }

    /// Logits of the last `n` of `t` final-norm rows, always through the
    /// immutable head representation. FP8 partitions rows into16-row launches.
    fn launch_head(&self, w: &Workspace<'_>, t: usize, n: usize, _decode: bool) -> Result<()> {
        ensure!(n <= t && n <= w.logits_rows, "head exceeds admitted logits rows");
        let x = Self::region(&w.x, (t - n) *self.cfg.hidden *2, n *self.cfg.hidden *2);
        self.head().launch(w.head.as_ref(), x, w.logits.buffer, n, self.stream)
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

    /// Greedy tokens of the first `rows` logits rows into `select` (U32 ids, then U32 statuses).
    fn select_greedy(&self, w: &Workspace<'_>, rows: usize) -> Result<()> {
        let vocab = self.cfg.vocab_size;
        // SAFETY: the logits rows and the select buffer are live buffers of these shapes.
        unsafe {
            self.library.cuda_logits_greedy_f32_async(w.logits.buffer.ptr, rows, vocab, vocab, w.select.buffer.ptr,
                std::ptr::null_mut(), Self::region(&w.select, rows * 4, rows * 4).ptr, self.stream)
        }
    }

    /// The first `rows` logits rows as device logits (`greedy`: with their selection
    /// from [`Self::select_greedy`]).
    fn device_logits(&self, w: &Workspace<'_>, rows: usize, greedy: bool) -> DeviceLogits {
        let vocab = self.cfg.vocab_size;
        DeviceLogits { ptr: w.logits.buffer.ptr, rows, vocab, stride: vocab, stream: self.stream,
            greedy: greedy.then(|| (w.select.buffer.ptr.cast_const(), Self::region(&w.select, rows * 4, rows * 4).ptr
                .cast_const())) }
    }

    /// Pages of a full context: the page-table stride of every decode step (constant,
    /// so the captured segments serve every sequence length).
    fn decode_stride(&self) -> usize {
        self.max_context.div_ceil(PAGE_ROWS).min(self.pages).max(1)
    }

    /// Decode steps can replay captured segments: every layer resident, and routed
    /// experts for the MoE layers.
    fn graphable(&self) -> bool {
        self.weights.layers.len() == self.cfg.layers
            && (self.experts.is_some() || self.weights.layers.iter().all(|l| l.dense))
    }

    /// What decode segment `index` (0..=layers) starts from: the embedded rows, or
    /// layer `index - 1`'s FFN output.
    fn previous(&self, index: usize) -> Previous {
        let Some(before) = index.checked_sub(1) else { return Previous::First };
        if self.weights.layers[before].dense {
            return Previous::Dense;
        }
        match &self.experts {
            Some(Experts::Spark { transport, .. }) => Previous::Planes(transport.borrow().world_size()),
            _ => Previous::Delta,
        }
    }

    /// Decode layers as captured graph segments, one per layer plus a tail: segment
    /// `i` holds layer `i - 1`'s FFN epilogue (the Spark partials' reduce, under a
    /// head split the FFN exchange) with layer `i`'s input norm, then layer `i`'s
    /// attention, post-attention norm, and its dense MLP or the MoE front (router,
    /// top-k, wire rows); the tail ends in the final norm, the LM head and the greedy
    /// selection. The routed experts run between segments: the host-driven Spark
    /// exchange (routes and rows down, the request out, partials into the intake
    /// planes), or this GPU's experts. A device-driven exchange that needs no host
    /// between segments can capture the same segments back to back as one graph.
    ///
    /// Under a head split each segment exchanges with rank 1's segment of the same
    /// layer (captured on rank 1's stream); rank 1's next segment is queued before
    /// the host waits in this layer's exchange.
    ///
    /// `launch` false only captures the segments (every shape a later step replays
    /// without capturing; see [`Self::capture_decode_graphs`]).
    #[allow(clippy::too_many_arguments)]
    fn decode_layers(&self, w: &Workspace<'_>, w1: Option<&Workspace<'_>>, tables: &StepTables, t: usize,
        gather: bool, head: bool, launch: bool) -> Result<()> {
        let layers = &self.weights.layers;
        let planes = match &self.experts {
            Some(Experts::Spark { transport, .. }) => transport.borrow().intake.pointers(),
            _ => [std::ptr::null(); MAX_RANKS],
        };
        let bf16_input = matches!(self.experts, Some(Experts::Spark { .. })) && self.expert_input.bf16(true);
        for index in 0..=layers.len() {
            let previous = self.previous(index);
            let last = index == layers.len();
            let key = GraphKey { layer: index, rows: t, table_stride: tables.table_stride, previous, head: head && last };
            self.graph(0, key, launch, || self.decode_segment(index, previous, w, w1, tables, t, gather, key.head,
                planes, bf16_input))?;
            if let (true, Previous::Planes(_), Some(Experts::Spark { transport, .. })) = (launch, previous, &self.experts) {
                // The planes are free for the next wave once the stream passes this segment's reduce.
                transport.borrow().intake.consumed(self.stream)?;
            }
            if let Some(w1) = w1 {
                // Rank 1's segments: layer 0 with rank 0's first, then each next one before
                // the host waits in this layer's exchange.
                if index == 0 {
                    self.peer_segment(0, w1, tables, t, launch)?;
                }
                if index + 1 < layers.len() {
                    self.peer_segment(index + 1, w1, tables, t, launch)?;
                }
            }
            if last || !launch || layers[index].dense {
                continue;
            }
            match self.experts.as_ref() {
                Some(Experts::Spark { transport, runtime, .. }) => {
                    let mut transport = transport.borrow_mut();
                    let mut sent = self.spark_send(w, index, t, true, bf16_input, true, &mut transport)?;
                    let receive = self.spark_receive(t, &mut transport, runtime, &mut sent)?;
                    self.wave_line(index, t, &sent, receive, 0.0);
                }
                Some(_) => self.moe_run(w, index, t, true)?,
                None => anyhow::bail!("layer {index} is an MoE layer; pass Spark peers for its routed experts"),
            }
            crate::shared::console::layer_mark(index);
        }
        Ok(())
    }

    /// Rank 0's decode segment `index` (see [`Self::decode_layers`]).
    #[allow(clippy::too_many_arguments)]
    fn decode_segment(&self, index: usize, previous: Previous, w: &Workspace<'_>, w1: Option<&Workspace<'_>>,
        tables: &StepTables, t: usize, gather: bool, head: bool, planes: [*const u16; MAX_RANKS], bf16_input: bool)
        -> Result<()> {
        let layers = &self.weights.layers;
        let rows = Scalar::I32(t as i32);
        let bytes = t * self.cfg.hidden * 2;
        if index == 0 {
            if gather {
                // SAFETY: the step's ids are in `ids` before the replay; `h` holds its rows.
                unsafe { self.embedding.gather(w.ids.buffer.ptr, std::ptr::null(), t, 1, std::ptr::null(),
                    w.h.buffer.ptr, self.stream)? };
            }
            if let Some(w1) = w1 {
                // The embedded rows to the second GPU's residual stream.
                self.exchange()?.push_to(0, DIRECT, w.h.buffer.ptr, w1.h.buffer.ptr, bytes)?;
            }
        }
        if let Previous::Planes(ranks) = previous {
            // SAFETY: the decode transport's intake planes, the zero plane and delta are live
            // [t, h] BF16 buffers; the planes hold the wave's partials once the stream gets here.
            unsafe {
                self.library.v41_compact_reducer()?.reduce_planes(planes, ranks as u32, w.zero_plane.buffer.ptr.cast(),
                    w.delta.buffer.ptr.cast(), t as u32, self.stream)?;
            }
        }
        // h += the previous layer's FFN output; x = this layer's input norm (or the final norm).
        let weight = match layers.get(index) {
            Some(layer) => layer.ptr("input_norm")?,
            None => self.weights.norm.buffer.ptr,
        };
        match (previous, index.checked_sub(1), w1.is_some()) {
            (Previous::First, ..) => self.norm(w, weight, 0, rows)?,
            // This GPU's dense partial plus rank 1's.
            (Previous::Dense, Some(before), true) => {
                let ffn = slot(before, true);
                self.wait(0, ffn)?;
                self.norm_on(0, w, weight, 2, rows, self.recv(0, ffn))?;
            }
            (Previous::Planes(_) | Previous::Delta, Some(before), true) => {
                // The routed experts' sum to rank 1 too (it has no layer after the last).
                if index < layers.len() {
                    self.push(0, slot(before, true), w.delta.buffer.ptr, bytes)?;
                }
                self.norm(w, weight, 1, rows)?;
            }
            _ => self.norm(w, weight, 1, rows)?,
        }
        if let (Some(drafter), Some(before)) = (&self.drafter, index.checked_sub(1)) {
            let n = t.min(super::dflash::TAP_ROWS);
            drafter.tap(before, w.h.buffer.ptr, t - n, n)?;
        }
        let Some(layer) = layers.get(index) else {
            if head {
                self.launch_head(w, t, t, true)?;
                self.select_greedy(w, t)?;
            }
            return Ok(());
        };
        self.attention_on(0, w, self.kv[index].buffer.ptr, layer, rows, "m64", tables)?;
        if w1.is_some() {
            let attended = slot(index, false);
            self.push(0, attended, w.delta.buffer.ptr, bytes)?;
            self.wait(0, attended)?;
            self.norm_on(0, w, layer.ptr("post_norm")?, 2, rows, self.recv(0, attended))?;
        } else {
            self.norm(w, layer.ptr("post_norm")?, 1, rows)?;
        }
        if layer.dense {
            self.dense_ffn(0, w, layer, rows, "m64", true)?;
            // Rank 1 takes this partial in its next segment (it has none after the last layer).
            if w1.is_some() && index + 1 < layers.len() {
                self.push(0, slot(index, true), w.delta.buffer.ptr, bytes)?;
            }
            Ok(())
        } else {
            self.moe_front(w, layer, t, bf16_input)?;
            if matches!(self.experts, Some(Experts::Spark { .. })) {
                // The wave's routes and rows down to the host inside the segment.
                self.stage_routes(w, t, bf16_input)?;
            }
            Ok(())
        }
    }

    /// Rank 1's decode segment of layer `index` (the head split's second GPU, see
    /// [`Self::decode_layers`]): the previous layer's FFN exchange (rank 0's dense
    /// partial or routed sum in) and this layer's input norm, its heads' attention and
    /// the attention all-reduce, then its dense partial out.
    fn peer_segment(&self, index: usize, w1: &Workspace<'_>, tables: &StepTables, t: usize, launch: bool)
        -> Result<()> {
        let peer = self.peer.as_ref().context("no head-split peer")?;
        let previous = match index.checked_sub(1) {
            None => Previous::First,
            Some(before) if peer.layers[before].dense => Previous::Dense,
            Some(_) => Previous::Delta,
        };
        let key = GraphKey { layer: index, rows: t, table_stride: tables.table_stride, previous, head: false };
        self.graph(1, key, launch, || {
            let rows = Scalar::I32(t as i32);
            let bytes = t * self.cfg.hidden * 2;
            let share = &peer.layers[index];
            match index.checked_sub(1) {
                None => {
                    self.wait(1, DIRECT)?;
                    self.norm_on(1, w1, share.ptr("input_norm")?, 0, rows, w1.delta.buffer.ptr)?;
                }
                Some(before) => {
                    let ffn = slot(before, true);
                    self.wait(1, ffn)?;
                    let (deltas, first) = if previous == Previous::Dense { (2, w1.delta.buffer.ptr) }
                        else { (1, self.recv(1, ffn)) };
                    self.norm_full(1, w1, share.ptr("input_norm")?, deltas, rows, first, self.recv(1, ffn))?;
                }
            }
            self.attention_on(1, w1, peer.kv[index].buffer.ptr, share, rows, "m64", tables)?;
            let attended = slot(index, false);
            self.push(1, attended, w1.delta.buffer.ptr, bytes)?;
            self.wait(1, attended)?;
            self.norm_on(1, w1, share.ptr("post_norm")?, 2, rows, self.recv(1, attended))?;
            if share.dense {
                self.dense_ffn(1, w1, share, rows, "m64", true)?;
                self.push(1, slot(index, true), w1.delta.buffer.ptr, bytes)?;
            }
            Ok(())
        })
    }

    /// Launches `segment` on rank `rank`'s stream through the graph captured for `key`,
    /// capturing it first when `key` is new; `launch` false only captures.
    fn graph(&self, rank: usize, key: GraphKey, launch: bool, segment: impl FnOnce() -> Result<()>) -> Result<()> {
        let graphs = match (rank, &self.peer) {
            (1, Some(peer)) => &peer.graphs,
            _ => &self.graphs,
        };
        let stream = self.stream_of(rank);
        if let Some(graph) = graphs.borrow().get(&key) {
            if !launch {
                return Ok(());
            }
            // SAFETY: the graph's pointers are persistent engine buffers of that rank.
            return self.on(rank, || unsafe { self.library.cuda_graph_launch(graph.0, stream) });
        }
        if launch {
            ensure!(self.graph_storage_plan.is_none(),
                "rank {rank} decode graph {key:?} was not captured during pre-admitted startup");
            self.late_captures.set(self.late_captures.get() + 1);
            tracing::debug!(rank, ?key, "MiMo decode segment captured while serving");
        }
        // SAFETY: capture records launches on that rank's stream; nothing in the
        // segment synchronizes the host or allocates.
        self.on(rank, || unsafe { self.library.cuda_graph_begin_capture(stream) })?;
        let captured = segment();
        // SAFETY: ends the capture begun above on the same stream.
        let exec = self.on(rank, || unsafe { self.library.cuda_graph_end_capture(stream) });
        captured?;
        let exec = GraphExec(exec?, self.library);
        if launch {
            // SAFETY: as for a replay.
            self.on(rank, || unsafe { self.library.cuda_graph_launch(exec.0, stream) })?;
        }
        graphs.borrow_mut().insert(key, exec);
        Ok(())
    }

    /// Captures (without running) every decode segment of steps of 1..=`max_rows` rows,
    /// so serving replays them without capturing; returns the graphs captured.
    pub fn capture_decode_graphs(&self, max_rows: usize) -> Result<usize> {
        self.capture_decode_graphs_observed(max_rows, |_| Ok(()))
    }

    /// The startup oracle records each completed shape through this same path.
    fn capture_decode_graphs_observed(&self, max_rows: usize,
        mut observe: impl FnMut(&str) -> Result<()>) -> Result<usize> {
        if !self.decode_graphs || !self.graphable() {
            return Ok(0);
        }
        let max_rows = max_rows.clamp(1, DECODE_ROWS);
        let plan = cuteafd_loader::families::mimo_v2::decode_graph::MimoDecodeGraphPlan::new(
            self.cfg.layers, self.ranks(), max_rows, true)?;
        let admitted = self.graph_storage_plan.as_ref().context("decode graph geometry was not pre-admitted")?;
        let bounds = self.graph_storage_bound_bytes.as_ref().context("decode graph storage was not pre-admitted")?;
        ensure!(bounds.len() == self.ranks() && admitted.segments_per_rank == plan.segments_per_rank
            && admitted.row_shapes >= max_rows, "decode graph admission has different physical geometry");
        if self.decode_workspace.borrow().is_none() {
            *self.decode_workspace.borrow_mut() = Some(self.workspace(0, DECODE_ROWS, true)?);
        }
        if let Some(peer) = &self.peer {
            if peer.decode_workspace.borrow().is_none() {
                *peer.decode_workspace.borrow_mut() = Some(self.workspace(1, DECODE_ROWS, true)?);
            }
        }
        let workspace = self.decode_workspace.borrow();
        let w = workspace.as_ref().context("workspace")?;
        let peer_workspace = self.peer.as_ref().map(|peer| peer.decode_workspace.borrow());
        let w1 = match &peer_workspace {
            Some(ws) => Some(ws.as_ref().context("peer workspace")?),
            None => None,
        };
        let free = |rank: usize| self.on(rank, || Ok(self.library.cuda_memory_info()?.0));
        let free_before_head = free(0)?;
        // The cuBLAS head's first call (handle setup) runs before any capture.
        self.launch_head(w, 1, 1, false)?;
        // SAFETY: the engine owns this stream.
        unsafe { self.library.cuda_stream_synchronize(self.stream)? };
        tracing::info!(bytes = free_before_head.saturating_sub(free(0)?),
            "MiMo eager vocabulary head initialization outside graph storage");
        observe("eager-vocabulary-head-initialization")?;
        let free_before = (0..self.ranks()).map(&free).collect::<Result<Vec<_>>>()?;
        let gather = self.embedding.device_gather();
        let before = self.graphs.borrow().len() + self.peer.as_ref().map_or(0, |p| p.graphs.borrow().len());
        let mut previous_free = free_before.clone();
        for t in 1..=max_rows {
            let tables = StepTables { decode: true, positions: vec![0; t], slots: Vec::new(), ring_slots: Vec::new(),
                seq_first: Vec::new(), page_table: Vec::new(), table_stride: self.decode_stride() };
            for head in cuteafd_loader::families::mimo_v2::decode_graph::MIMO_DECODE_TAIL_HEAD_VARIANTS {
                // Partial-row logit requests use a distinct headless tail key.
                // All preceding layer/peer keys are shared and already captured.
                self.decode_layers(w, w1, &tables, t, gather, head, false)?;
            }
            observe(&format!("decode-graph-row-shape-{t}"))?;
            for rank in 0..self.ranks() {
                let now = free(rank)?;
                let delta = previous_free[rank].saturating_sub(now) as u64;
                previous_free[rank] = now;
                let observed = self.graph_storage_observed_bytes[rank].get().checked_add(delta)
                    .context("decode graph measured storage overflow")?;
                self.graph_storage_observed_bytes[rank].set(observed);
                ensure!(observed <= bounds[rank],
                    "rank {rank} retained decode graph storage {observed} B at {t} rows exceeds the pre-admitted {} B bound",bounds[rank]);
            }
        }
        for rank in 0..self.ranks() {
            let graphs = if rank == 0 { &self.graphs } else { &self.peer.as_ref().context("peer graphs")?.graphs };
            let actual = graphs.borrow().keys().filter(|key| key.rows <= max_rows).count();
            ensure!(actual == plan.executables_per_rank[rank],
                "rank {rank} captured {actual} decode graphs; admitted geometry expects {}",plan.executables_per_rank[rank]);
            let delta = free_before[rank].saturating_sub(free(rank)?) as u64;
            let observed = self.graph_storage_observed_bytes[rank].get();
            tracing::info!(rank, executables=actual, delta_bytes=delta, observed_bytes=observed, bound_bytes=bounds[rank],
                "MiMo retained decode graph storage");
        }
        let after = self.graphs.borrow().len() + self.peer.as_ref().map_or(0, |p| p.graphs.borrow().len());
        let mib = |a: usize, b: usize| a.saturating_sub(b) as f64 / (1u64 << 20) as f64;
        tracing::info!(device_mib = format!("{:.1}", mib(free_before[0], free(0)?)),
            peer_mib = format!("{:.1}", if self.peer.is_some() { mib(free_before[1], free(1)?) } else { 0.0 }),
            "MiMo decode graph memory");
        Ok(after - before)
    }

    /// Decode segments captured while serving (shapes the startup capture missed).
    pub fn late_captures(&self) -> usize {
        self.late_captures.get()
    }

    /// Queues rank 1's share of layer `index` (the head split's second GPU),
    /// mirroring rank 0's pushes in `step` one for one: the embedded rows before
    /// layer 0 (its first wait), then per layer its attention partial out and rank
    /// 0's in, and the dense MLP's partials or the routed experts' sum in. Ends
    /// with the next layer's input norm (nothing after the last layer).
    #[allow(clippy::too_many_arguments)]
    fn peer_layer(&self, index: usize, w1: &Workspace<'_>, rows: Scalar, cap: &str, tables: &StepTables, bytes: usize)
        -> Result<()> {
        let peer = self.peer.as_ref().context("no head-split peer")?;
        let share = &peer.layers[index];
        if index == 0 {
            self.wait(1, DIRECT)?;
            self.norm_on(1, w1, share.ptr("input_norm")?, 0, rows, w1.delta.buffer.ptr)?;
        }
        self.attention_on(1, w1, peer.kv[index].buffer.ptr, share, rows, cap, tables)?;
        let (attended, ffn) = (slot(index, false), slot(index, true));
        self.push(1, attended, w1.delta.buffer.ptr, bytes)?;
        self.wait(1, attended)?;
        self.norm_on(1, w1, share.ptr("post_norm")?, 2, rows, self.recv(1, attended))?;
        let next = peer.layers.get(index + 1);
        if share.dense {
            self.dense_ffn(1, w1, share, rows, cap, tables.decode)?;
            self.push(1, ffn, w1.delta.buffer.ptr, bytes)?;
            self.wait(1, ffn)?;
        } else if next.is_some() {
            self.wait(1, ffn)?;
        }
        let Some(next) = next else { return Ok(()) };
        let (deltas, first) = if share.dense { (2, w1.delta.buffer.ptr) } else { (1, self.recv(1, ffn)) };
        self.norm_full(1, w1, next.ptr("input_norm")?, deltas, rows, first, self.recv(1, ffn))
    }

    /// Rank 1's attention and dense-FFN producer for one prefill lane. The
    /// FFN wait is queued separately, only once rank 0 has queued its dense
    /// partial or landed the routed sum (see [`Self::peer_lane_post`]).
    fn peer_lane_attention(&self, index: usize, lane: usize, w: &Workspace<'_>, rows: Scalar,
        tables: &StepTables, bytes: usize) -> Result<()> {
        let peer = self.peer.as_ref().context("no head-split peer")?;
        let layer = &peer.layers[index];
        if index == 0 {
            self.wait(1, DIRECT)?;
            self.norm_on(1, w, layer.ptr("input_norm")?, 0, rows, w.delta.buffer.ptr)?;
        }
        self.attention_on(1, w, peer.kv[index].buffer.ptr, layer, rows, "m4096", tables)?;
        let attended = lane_slot(index, false, lane);
        self.push(1, attended, w.delta.buffer.ptr, bytes)?;
        self.wait(1, attended)?;
        self.norm_on(1, w, layer.ptr("post_norm")?, 2, rows, self.recv(1, attended))?;
        if layer.dense {
            self.dense_ffn(1, w, layer, rows, "m4096", false)?;
            self.push(1, lane_slot(index, true, lane), w.delta.buffer.ptr, bytes)?;
        }
        Ok(())
    }

    /// Finish rank 1's FFN for a lane, keeping its residual stream identical
    /// to rank 0's, and prepare its next layer's normalized input.
    fn peer_lane_post(&self, index: usize, lane: usize, w: &Workspace<'_>, rows: Scalar) -> Result<()> {
        let peer = self.peer.as_ref().context("no head-split peer")?;
        let dense = peer.layers[index].dense;
        let next = peer.layers.get(index + 1);
        let ffn = lane_slot(index, true, lane);
        if dense || next.is_some() {
            self.wait(1, ffn)?;
        }
        let Some(next) = next else { return Ok(()) };
        let first = if dense { w.delta.buffer.ptr } else { self.recv(1, ffn) };
        self.norm_full(1, w, next.ptr("input_norm")?, if dense { 2 } else { 1 }, rows, first, self.recv(1, ffn))
    }

    /// `residual (h) += delta` when `deltas` is 1, then `x = weight * RMSNorm(h)`.
    fn norm(&self, w: &Workspace<'_>, weight: *mut c_void, deltas: i32, rows: Scalar) -> Result<()> {
        self.norm_full(0, w, weight, deltas, rows, w.delta.buffer.ptr, w.delta.buffer.ptr)
    }

    /// On rank `rank`: `h += bf16(delta + second)` when `deltas` is 2 (the local
    /// `delta` first, so both GPUs of a head split sum identically), `h += delta`
    /// when 1, then `x = weight * RMSNorm(h)`.
    #[allow(clippy::too_many_arguments)]
    fn norm_on(&self, rank: usize, w: &Workspace<'_>, weight: *mut c_void, deltas: i32, rows: Scalar,
        second: *mut c_void) -> Result<()> {
        self.norm_full(rank, w, weight, deltas, rows, w.delta.buffer.ptr, second)
    }

    #[allow(clippy::too_many_arguments)]
    fn norm_full(&self, rank: usize, w: &Workspace<'_>, weight: *mut c_void, deltas: i32, rows: Scalar,
        first: *mut c_void, second: *mut c_void) -> Result<()> {
        self.run_on(rank, false, "mimo_norm", &[("residual", w.h.buffer.ptr), ("delta0", first), ("delta1", second),
            ("weight", weight), ("out", w.x.buffer.ptr)], &[rows, Scalar::I32(deltas)])
    }

    /// Copies this step's last-layer rows (pre-norm) into the MTP hidden ring
    /// at (ring, position % 256); a prefill's last 256 rows.
    fn mtp_tap(&self, mtp: &super::mtp::MtpDrafter<'_>, w: &Workspace<'_>, tables: &StepTables) -> Result<()> {
        use super::mtp::HIDDEN_ROWS;
        let (h, t) = (self.cfg.hidden, tables.positions.len());
        let row = h * 2;
        let first = t.saturating_sub(HIDDEN_ROWS);
        let mut r = first;
        while r < t {
            // A run of rows whose ring slots advance by one (one sequence, no wrap).
            let slot = tables.ring_slots[r] as usize;
            let mut n = 1;
            while r + n < t && tables.ring_slots[r + n] as usize == slot + n && (slot + n) % HIDDEN_ROWS != 0 {
                n += 1;
            }
            let ring = slot / RING_ROWS;
            let dest = (ring * HIDDEN_ROWS + slot % RING_ROWS % HIDDEN_ROWS) * row;
            // SAFETY: rows r..r+n of `w.h` and ring rows dest.. lie inside their buffers; stream-ordered.
            unsafe {
                self.library.copy_d2d_async(
                    cuteafd_ffi::CuteafdDeviceBuffer { ptr: mtp.hidden.buffer.ptr.cast::<u8>().add(dest).cast(),
                        bytes: n * row, ..mtp.hidden.buffer },
                    cuteafd_ffi::CuteafdDeviceBuffer { ptr: w.h.buffer.ptr.cast::<u8>().add(r * row).cast(),
                        bytes: n * row, ..w.h.buffer },
                    n * row, self.stream)?;
            }
            r += n;
        }
        Ok(())
    }

    /// A new sequence in `ring` whose first `len` tokens are processed: MTP
    /// stages start their true rows within a window of the end.
    pub fn mtp_reset(&self, ring: usize, len: usize) {
        if let Some(mtp) = &self.mtp {
            let mut ext = mtp.ext.borrow_mut();
            let start = len.saturating_sub(self.cfg.window + mtp.stages.len() + 1);
            ext[ring] = vec![start; mtp.stages.len()];
        }
    }

    /// Up to `stages` MTP drafts after each sequence's next token (see
    /// `mtp`). Every pass embeds its tokens on the device (later stages read
    /// the earlier stages' drafts there), so the passes queue back to back
    /// and the drafts come back once, at the end.
    pub fn mtp_draft(&self, seqs: &[super::mtp::MtpSeq<'_>], stages: usize) -> Result<Vec<Vec<u32>>> {
        self.submit(|| self.mtp_draft_inner(seqs, stages))
    }

    fn mtp_draft_inner(&self, seqs: &[super::mtp::MtpSeq<'_>], stages: usize) -> Result<Vec<Vec<u32>>> {
        use super::mtp::Token;
        let mtp = self.mtp.as_ref().context("no MTP drafter")?;
        let stages = stages.min(mtp.stages.len());
        ensure!(seqs.len() <= DECODE_ROWS, "MTP drafts of {} sequences", seqs.len());
        // The previous step is done with the staging.
        // SAFETY: the engine owns this stream.
        unsafe { self.library.cuda_stream_synchronize(self.stream)? };
        self.mtp_staging.borrow_mut().1 = 0;
        for k in 0..stages {
            // Catch-up passes over true rows while a sequence's rows exceed the step.
            loop {
                let mut groups = Vec::new();
                let mut rows = 0;
                for seq in seqs {
                    let ext = mtp.ext.borrow()[seq.ring][k];
                    ensure!(ext <= seq.len, "MTP ring {} is ahead of its sequence (reset it at admission)", seq.ring);
                    // True rows are those whose token t_{j+k+1} is known (j <= len - k - 1).
                    // Catch-up stops before row len - 1: the drafting pass needs that row (its
                    // argmax is the draft), so stage 0 must not consume it here.
                    let true_end = seq.len.saturating_sub(k.max(1));
                    if seq.len - ext > MTP_STEP_ROWS && ext < true_end && rows < DECODE_ROWS {
                        let n = (true_end - ext).min(DECODE_ROWS - rows);
                        groups.push((seq.ring, ext, (ext..ext + n).map(|j| Token::Known(seq.tokens[j + k + 1]))
                            .collect::<Vec<_>>()));
                        rows += n;
                    }
                }
                if groups.is_empty() {
                    break;
                }
                self.mtp_pass(mtp, k, &groups, None, seqs)?;
                let mut ext = mtp.ext.borrow_mut();
                for (ring, first, tokens) in &groups {
                    ext[*ring][k] = ext[*ring][k].max(first + tokens.len());
                }
            }
            // The drafting pass: rows ext..len of every sequence (in groups within the step).
            let mut index = 0;
            while index < seqs.len() {
                let mut groups = Vec::new();
                let mut members = Vec::new();
                let mut rows = 0;
                while index < seqs.len() {
                    let seq = &seqs[index];
                    let ext = mtp.ext.borrow()[seq.ring][k];
                    ensure!(seq.tokens.len() > seq.len, "MTP sequence needs its next token");
                    let n = seq.len - ext;
                    if rows + n > DECODE_ROWS && !groups.is_empty() {
                        break;
                    }
                    ensure!(n <= DECODE_ROWS, "MTP stage {k}: {n} pending rows");
                    let tokens: Vec<Token> = (ext..seq.len).map(|j| {
                        let at = j + k + 1;
                        if at <= seq.len { Token::Known(seq.tokens[at]) }
                        else { Token::Draft { stage: at - seq.len - 1, member: index } }
                    }).collect();
                    groups.push((seq.ring, ext, tokens));
                    members.push(index);
                    rows += n;
                    index += 1;
                }
                self.mtp_pass(mtp, k, &groups, Some(&members), seqs)?;
                let mut ext = mtp.ext.borrow_mut();
                for (&i, (ring, _, _)) in members.iter().zip(&groups) {
                    ext[*ring][k] = ext[*ring][k].max(seqs[i].len.saturating_sub(k));
                }
            }
        }
        if stages == 0 {
            return Ok(vec![Vec::new(); seqs.len()]);
        }
        let bytes = self.download(&mtp.ids, (1 + stages) * DECODE_ROWS * 4)?;
        let word = |at: usize| u32::from_le_bytes(bytes[at * 4..at * 4 + 4].try_into().unwrap());
        Ok((0..seqs.len()).map(|i| (0..stages).map(|k| word((1 + k) * DECODE_ROWS + i)).collect()).collect())
    }

    /// Queues `bytes` into `dst` through the MTP staging (waits for the
    /// stream only when the staging is full).
    fn stage_async(&self, dst: cuteafd_ffi::CuteafdDeviceBuffer, bytes: &[u8]) -> Result<()> {
        if bytes.is_empty() {
            return Ok(());
        }
        ensure!(bytes.len() <= dst.bytes, "staged upload exceeds its buffer");
        let mut staging = self.mtp_staging.borrow_mut();
        if staging.1 + bytes.len() > staging.0.buffer.bytes {
            // SAFETY: the engine owns this stream; its queued copies read the staging.
            unsafe { self.library.cuda_stream_synchronize(self.stream)? };
            staging.1 = 0;
        }
        let at = staging.1;
        ensure!(at + bytes.len() <= staging.0.buffer.bytes, "MTP pass inputs exceed the staging buffer");
        staging.0.bytes_mut()[at..at + bytes.len()].copy_from_slice(bytes);
        let source = cuteafd_ffi::CuteafdHostBuffer {
            // SAFETY: `at` lies inside the pinned staging buffer.
            ptr: unsafe { staging.0.buffer.ptr.cast::<u8>().add(at) }.cast(),
            bytes: bytes.len(),
            ..staging.0.buffer
        };
        // SAFETY: the staged bytes stay untouched until the stream drains (see above).
        unsafe { self.library.copy_host_buffer_h2d_async(dst, source, bytes.len(), self.stream)? };
        staging.1 = (at + bytes.len()).div_ceil(16) * 16;
        Ok(())
    }

    /// One MTP stage over `groups` of (ring, first row, tokens t_{j+k+1});
    /// with `members`, each group's last row's draft becomes stage `k`'s draft
    /// of that member (on the device). Queued: no host wait.
    fn mtp_pass(&self, mtp: &super::mtp::MtpDrafter<'_>, k: usize,
        groups: &[(usize, usize, Vec<super::mtp::Token>)], members: Option<&[usize]>,
        seqs: &[super::mtp::MtpSeq<'_>]) -> Result<()> {
        use super::mtp::{Token, HIDDEN_ROWS};
        let stage = &mtp.stages[k];
        let (h, vocab) = (self.cfg.hidden, self.cfg.vocab_size);
        let mut tables = StepTables { decode: true, positions: Vec::new(), slots: Vec::new(), ring_slots: Vec::new(),
            seq_first: Vec::new(), page_table: vec![0], table_stride: 1 };
        let (mut known, mut index) = (Vec::new(), Vec::new());
        for (ring, first, tokens) in groups {
            ensure!(*ring < self.rings, "MTP ring {ring} of {}", self.rings);
            let start = tables.positions.len() as i32;
            for (j, token) in (*first..first + tokens.len()).zip(tokens) {
                tables.positions.push(j as i64);
                tables.slots.push(-1);
                tables.ring_slots.push((ring * RING_ROWS + j % RING_ROWS) as i64);
                tables.seq_first.push(start);
                let r = known.len();
                match *token {
                    Token::Known(id) => {
                        known.push(id);
                        index.push(r as u32);
                    }
                    Token::Draft { stage, member } => {
                        ensure!(stage < k && member < DECODE_ROWS, "MTP stage {k} reads draft {stage} of {member}");
                        known.push(0);
                        index.push(((1 + stage) * DECODE_ROWS + member) as u32);
                    }
                }
            }
        }
        let t = tables.positions.len();
        ensure!(t > 0 && t <= DECODE_ROWS, "MTP pass of {t} rows");
        if self.decode_workspace.borrow().is_none() {
            *self.decode_workspace.borrow_mut() = Some(self.workspace(0, DECODE_ROWS, true)?);
        }
        let workspace = self.decode_workspace.borrow();
        let w = workspace.as_ref().context("workspace")?;
        self.stage_async(w.positions.buffer, bytes_of(&tables.positions))?;
        self.stage_async(w.ring_slots.buffer, bytes_of(&tables.ring_slots))?;
        self.stage_async(w.seq_first.buffer, bytes_of(&tables.seq_first))?;
        self.stage_async(w.page_table.buffer, bytes_of(&tables.page_table))?;
        self.embedding.check(&known)?;
        self.stage_async(mtp.ids.buffer, bytes_of(&known))?;
        self.stage_async(mtp.index.buffer, bytes_of(&index))?;
        let row = h * 2;
        // SAFETY: the ids (known tokens, earlier stages' drafts) and the indices are
        // ordered on this stream before the gather; `embed` holds DECODE_ROWS rows.
        unsafe { self.embedding.embed_device_ids(mtp.ids.buffer, Some((mtp.index.buffer.ptr.cast_const(), &index)),
            t, 1, None, mtp.embed.buffer, self.stream)? };
        if seqs.iter().any(|s| s.media.is_some()) {
            let chunk = super::mtp::embedding_media(seqs, k, groups)?;
            if !chunk.indices.is_empty() {
                self.library.embedding_injection()?.inject_host(&chunk.features, &chunk.indices,
                    w.x.buffer, w.seq_first.buffer, mtp.embed.buffer, t, h, 1, self.stream)?;
                self.stage_async(w.seq_first.buffer, bytes_of(&tables.seq_first))?;
            }
        }
        // The target's hidden rows of the same positions.
        let mut at = 0;
        for (ring, first, tokens) in groups {
            for j in *first..first + tokens.len() {
                let src = (ring * HIDDEN_ROWS + j % HIDDEN_ROWS) * row;
                // SAFETY: one row inside each buffer; stream-ordered.
                unsafe {
                    self.library.copy_d2d_async(
                        cuteafd_ffi::CuteafdDeviceBuffer { ptr: mtp.rows_h.buffer.ptr.cast::<u8>().add(at * row).cast(),
                            bytes: row, ..mtp.rows_h.buffer },
                        cuteafd_ffi::CuteafdDeviceBuffer { ptr: mtp.hidden.buffer.ptr.cast::<u8>().add(src).cast(),
                            bytes: row, ..mtp.hidden.buffer }, row, self.stream)?;
                }
                at += 1;
            }
        }
        let rows = Scalar::I32(t as i32);
        for (source, weight, out) in [(&mtp.embed, &stage.enorm, &mtp.normed_e), (&mtp.rows_h, &stage.hnorm, &mtp.normed_h)] {
            self.run("mimo_norm", &[("residual", source.buffer.ptr), ("delta0", source.buffer.ptr),
                ("delta1", source.buffer.ptr), ("weight", weight.buffer.ptr), ("out", out.buffer.ptr)],
                &[rows, Scalar::I32(0)])?;
        }
        // SAFETY: [t, H] sources into [t, 2H] halves, then eh_proj into the residual stream.
        unsafe {
            self.library.glm_dflash_tap(mtp.normed_e.buffer.ptr, mtp.cat.buffer.ptr, t, h, 2 * h, 0, self.stream)?;
            self.library.glm_dflash_tap(mtp.normed_h.buffer.ptr, mtp.cat.buffer.ptr, t, h, 2 * h, h, self.stream)?;
            self.library.linear_bf16(mtp.cat.buffer.ptr, stage.eh.buffer.ptr, w.h.buffer.ptr, t, 2 * h, h, self.stream)?;
        }
        let layer = &stage.layer;
        self.norm(w, layer.ptr("input_norm")?, 0, rows)?;
        self.attention(w, stage.ring.buffer.ptr, layer, rows, "m64", &tables)?;
        self.norm(w, layer.ptr("post_norm")?, 1, rows)?;
        self.dense_ffn(0, w, layer, rows, "m64", true)?;
        self.norm(w, stage.final_norm.buffer.ptr, 1, rows)?;
        let Some(members) = members else { return Ok(()) };
        ensure!(members.len() == groups.len(), "a member per drafting group");
        self.launch_head(w, t, t, true)?;
        let region = |dev: &Dev<'_>, offset: usize, bytes: usize| cuteafd_ffi::CuteafdDeviceBuffer {
            // SAFETY: callers pass offsets inside the allocation.
            ptr: unsafe { dev.buffer.ptr.cast::<u8>().add(offset) }.cast(), bytes, ..dev.buffer };
        // SAFETY: the logits rows, the select buffer (ids, then statuses) and the
        // drafts region are live buffers of these shapes; stream-ordered.
        unsafe {
            self.library.cuda_logits_greedy_f32_async(w.logits.buffer.ptr, t, vocab, vocab, w.select.buffer.ptr,
                std::ptr::null_mut(), region(&w.select, t * 4, t * 4).ptr, self.stream)?;
            let mut end = 0;
            for ((_, _, tokens), &member) in groups.iter().zip(members) {
                end += tokens.len();
                self.library.copy_d2d_async(region(&mtp.ids, ((1 + k) * DECODE_ROWS + member) * 4, 4),
                    region(&w.select, (end - 1) * 4, 4), 4, self.stream)?;
            }
        }
        Ok(())
    }

    /// `kv`: the layer's paged record pool (full) or its rings (SWA).
    fn attention(&self, w: &Workspace<'_>, kv: *mut c_void, layer: &MimoLayer<'_>, rows: Scalar, cap: &str,
        tables: &StepTables) -> Result<()> {
        self.attention_on(0, w, kv, layer, rows, cap, tables)
    }

    /// Attention of `layer` (rank `rank`'s share under a head split: its heads,
    /// a partial o_proj sum) into `delta`, on that rank's GPU.
    #[allow(clippy::too_many_arguments)]
    fn attention_on(&self, rank: usize, w: &Workspace<'_>, kv: *mut c_void, layer: &MimoLayer<'_>, rows: Scalar,
        cap: &str, tables: &StepTables) -> Result<()> {
        for part in 0..ATTENTION_PARTS {
            self.attention_part(rank, part, w, kv, layer, rows, cap, tables)?;
        }
        Ok(())
    }

    /// Part `part` of [`Self::attention_on`]: 0 the qkv producer, 1 attention,
    /// 2 o_proj (a head split enqueues the two GPUs' parts alternately, so
    /// neither waits for the host to queue the other's whole layer).
    #[allow(clippy::too_many_arguments)]
    fn attention_part(&self, rank: usize, part: usize, w: &Workspace<'_>, kv: *mut c_void, layer: &MimoLayer<'_>,
        rows: Scalar, cap: &str, tables: &StepTables) -> Result<()> {
        let mode = if tables.decode { "decode" } else { "prefill" };
        let k = kind(layer.attention);
        let (full, swa) = match (rank, &self.peer) {
            (1, Some(peer)) => (&peer.cos_sin_full, &peer.cos_sin_swa),
            _ => (&self.cos_sin_full, &self.cos_sin_swa),
        };
        let (slots, cos_sin) = match layer.attention {
            MimoAttention::Full => (w.slots.buffer.ptr, full.buffer.ptr),
            MimoAttention::Sliding => (w.step_slots.buffer.ptr, swa.buffer.ptr),
        };
        let split = layer.split;
        let records = match layer.attention {
            MimoAttention::Full => kv,
            MimoAttention::Sliding => w.kv_step.buffer.ptr,
        };
        let decode = tables.decode;
        if part == 0 {
            let pointers = [("x", w.x.buffer.ptr), ("positions", w.positions.buffer.ptr), ("kv_slots", slots),
                ("cos_sin", cos_sin), ("w_qkv_fp8", layer.ptr("w_qkv_fp8")?), scale("w_qkv", decode, layer)?,
                ("kv_cache", records), ("query", w.query.buffer.ptr), ("scratch", w.scratch.buffer.ptr)];
            let kv = self.kv_cache.of(layer.attention).program_tag();
            return self.run_on(rank, split, &format!("mimo_{k}_producer{kv}_{cap}"), &pointers, &self.w8_scalars(rows, decode));
        }
        let name = format!("mimo_{k}_attention{}_{mode}_{cap}", self.kv_cache.of(layer.attention).program_tag());
        match layer.attention {
            _ if part != 1 => {}
            MimoAttention::Full => {
                let mut scalars = vec![rows, Scalar::I32(tables.table_stride as i32)];
                let mut pointers = vec![("q", w.query.buffer.ptr), ("kv_cache", kv),
                    ("positions", w.positions.buffer.ptr), ("page_table", w.page_table.buffer.ptr)];
                if tables.decode {
                    scalars.push(Scalar::I32(DECODE_SPLITS));
                } else if self.kv_cache != MimoKvCache::Bf16 {
                    // 8-bit prefill: the sequence's keys are widened once into `kv_wide`.
                    let keys = tables.positions.last().map_or(0, |&p| p + 1);
                    ensure!(keys as usize <= self.max_context, "prefill past max_context ({keys} keys)");
                    pointers.push(("kv_wide", w.kv_wide.buffer.ptr));
                    scalars.push(Scalar::I32(keys as i32));
                }
                pointers.extend([("out", w.attn.buffer.ptr), ("scratch", w.scratch.buffer.ptr)]);
                self.run_on(rank, split, &name, &pointers, &scalars)?;
            }
            MimoAttention::Sliding => {
                self.run_on(rank, split, &name, &[("q", w.query.buffer.ptr), ("kv_step", w.kv_step.buffer.ptr),
                    ("ring", kv), ("positions", w.positions.buffer.ptr),
                    ("ring_slots", w.ring_slots.buffer.ptr), ("seq_first", w.seq_first.buffer.ptr),
                    ("sinks", layer.ptr("sinks")?), ("out", w.attn.buffer.ptr), ("scratch", w.scratch.buffer.ptr)],
                    &[rows])?;
            }
        }
        if part != 2 {
            return Ok(());
        }
        if layer.has("w_o_fp8") {
            let pointers = [("attn", w.attn.buffer.ptr), ("w_o_fp8", layer.ptr("w_o_fp8")?),
                scale("w_o", decode, layer)?, ("out", w.delta.buffer.ptr), ("scratch", w.scratch.buffer.ptr)];
            // Prefill retains BF16 activations. A8 output activations are a
            // separate arithmetic/quality gate, independent of QKV/FFN flags.
            let scalars = [rows, Scalar::I32(if decode && self.output_fp8_decode { FP8_DECODE_ROWS } else { 0 })];
            return self.run_on(rank, split, &format!("mimo_o_w8_{cap}"), &pointers, &scalars);
        }
        let mut pointers = vec![("attn", w.attn.buffer.ptr), ("w_o", layer.ptr("w_o")?)];
        if decode {
            pointers.extend([("w_o_fp8", layer.ptr_or("w_o_fp8", "w_o")?), ("w_o_scale", layer.ptr_or("w_o_scale", "w_o")?)]);
        }
        pointers.push(("out", w.delta.buffer.ptr));
        self.run_on(rank, split, &format!("mimo_o_{cap}"), &pointers, &fp8_scalars(rows, decode, layer.has("w_o_fp8")))
    }

    /// The dense SwiGLU MLP (layer 0, MTP layers) over its FP8-only weights into `delta`.
    #[allow(clippy::too_many_arguments)]
    fn dense_ffn(&self, rank: usize, w: &Workspace<'_>, layer: &MimoLayer<'_>, rows: Scalar, cap: &str, decode: bool)
        -> Result<()> {
        let pointers = [("x", w.x.buffer.ptr), ("w_gate_up_fp8", layer.ptr("w_gate_up_fp8")?),
            scale("w_gate_up", decode, layer)?, ("w_down_fp8", layer.ptr("w_down_fp8")?), scale("w_down", decode, layer)?,
            ("out", w.delta.buffer.ptr), ("scratch", w.scratch.buffer.ptr)];
        self.run_on(rank, layer.split, &format!("mimo_ffn_{cap}"), &pointers, &self.w8_scalars(rows, decode))
    }

    /// `[rows, fp8_rows]` of the programs over FP8-only weights: decode rows up
    /// to `FP8_DECODE_ROWS` on the FP8 GEMVs (W8A16 GEMMs above); prefill 1
    /// (W8A8) or 0 (W8A16, `prefill_w8a8` off).
    fn w8_scalars(&self, rows: Scalar, decode: bool) -> [Scalar; 2] {
        [rows, Scalar::I32(if decode { FP8_DECODE_ROWS } else { i32::from(self.prefill_w8a8) })]
    }

    /// Per MoE layer, the weights a decode step reads after its routed
    /// experts are out, in read order: the next layer's attention (E4M3
    /// copies where the decode programs read them), norms, router and dense
    /// MLP; after the last layer the final norm and head.
    pub fn decode_read_order(&self) -> Vec<Vec<crate::shared::l2_prefetch::Range>> {
        let layers = &self.weights.layers;
        (0..layers.len()).map(|i| match layers.get(i + 1) {
            Some(next) => crate::shared::l2_prefetch::operands(&["input_norm", "w_qkv", "sinks", "w_o", "post_norm",
                "w_router", "w_hilo", "gate.bias", "w_gate_up", "w_down"], |n| next.range(n)),
            None => {
                let head = self.weights.head.allocations();
                std::iter::once(&self.weights.norm).chain(head).map(|a| (a.buffer.ptr.cast_const(), a.buffer.bytes))
                    .collect()
            }
        }).collect()
    }

    /// In a decode step with no real exchange (local experts), under
    /// CUTEAFD_EMULATE_EXCHANGE_US (benchmarks): the L2 prefetch and a
    /// Spark-like wait before the experts run.
    fn emulated_exchange(&self, index: usize, decode: bool) -> Result<()> {
        if !decode {
            return Ok(());
        }
        let Some(mark) = crate::shared::l2_prefetch::exchange_mark(self.library, self.stream)? else { return Ok(()) };
        if let Some(l2) = &self.l2 {
            l2.issue(self.library, index, self.stream)?;
        }
        crate::shared::l2_prefetch::exchange_wait(self.library, Some(mark))
    }

    /// Router scores, the sigmoid top-k select and the FP8 wire rows, then the
    /// routed experts (local or Spark); leaves their sum in `delta` for the
    /// next norm's residual add.
    /// With `forward`, the sum is also pushed into that exchange slot of the head split's
    /// second GPU (from the Spark path right after the reduce, before the host wait).
    fn moe(&self, w: &Workspace<'_>, index: usize, layer: &MimoLayer<'_>, t: usize, decode: bool,
        forward: Option<usize>) -> Result<()> {
        let bytes = t * self.cfg.hidden * 2;
        let spark = matches!(self.experts, Some(Experts::Spark { .. }));
        self.moe_local(w, index, layer, t, decode, forward.filter(|_| spark))?;
        match forward {
            Some(slot) if !spark => self.push(0, slot, w.delta.buffer.ptr, bytes),
            _ => Ok(()),
        }
    }

    /// Router scores, the top-k selection and (unless the ranks take BF16
    /// rows) the FP8 wire rows, all on the stream.
    fn moe_front(&self, w: &Workspace<'_>, layer: &MimoLayer<'_>, t: usize, bf16_input: bool) -> Result<()> {
        let (h, topk) = (self.cfg.hidden, self.cfg.topk);
        let rows = Scalar::I32(t as i32);
        self.run("mimo_router_scores", &[("x", w.x.buffer.ptr), layer.router_operand()?,
            ("logits", w.router_logits.buffer.ptr), ("scratch", w.scratch.buffer.ptr)], &[rows])?;
        // SAFETY: logits, bias and route outputs are live buffers of `t` rows.
        unsafe {
            self.library.router_select(w.router_logits.buffer.ptr, layer.ptr("gate.bias")?, std::ptr::null(),
                std::ptr::null(), w.route_ids.buffer.ptr, w.route_weights.buffer.ptr, t, self.cfg.experts, topk,
                self.cfg.routed_scale as f32, true, self.stream)?;
        }
        let grid = self.quantize_grid.blocks(t, h);
        if !bf16_input {
            self.run("mimo_expert_input_quant", &[("source_ptr", w.x.buffer.ptr), ("values_ptr", w.wire.buffer.ptr),
                // SAFETY: the scale rows follow the payload inside each wire row.
                ("scale_rows_ptr", unsafe { w.wire.buffer.ptr.cast::<u8>().add(h) }.cast()),
                ("scale_mma_ptr", w.delta.buffer.ptr)], &[rows, Scalar::I32(grid as i32)])?;
        }
        Ok(())
    }

    fn moe_local(&self, w: &Workspace<'_>, index: usize, layer: &MimoLayer<'_>, t: usize, decode: bool,
        forward: Option<usize>) -> Result<()> {
        let experts = self.experts.as_ref().with_context(|| format!(
            "layer {index} is an MoE layer: pass Spark --peers serving the fp8 family, or --local-experts \
             (run --layers 1 for the dense layer alone)"))?;
        let bf16_input = matches!(experts, Experts::Spark { .. }) && self.expert_input.bf16(decode);
        self.moe_front(w, layer, t, bf16_input)?;
        match experts {
            Experts::Spark { transport, runtime, .. } => {
                self.spark_moe(w, index, t, decode, bf16_input, &mut transport.borrow_mut(), runtime, forward)
            }
            _ => self.moe_run(w, index, t, decode),
        }
    }

    /// The routed experts on this GPU (local, streamed or skipped) after [`Self::moe_front`],
    /// their sum into `delta`.
    fn moe_run(&self, w: &Workspace<'_>, index: usize, t: usize, decode: bool) -> Result<()> {
        let h = self.cfg.hidden;
        let experts = self.experts.as_ref().context("no routed experts")?;
        self.emulated_exchange(index, decode)?;
        match experts {
            Experts::Local(local) => {
                let resident = local.index_of(index)?;
                // Coordinator packages take the BF16 rows themselves (exact, no wire).
                let input = if local.wire_input() { w.wire.buffer.ptr } else { w.x.buffer.ptr };
                // SAFETY: input rows, route ids (u32 = i32 for ids < 256), weights and
                // delta are live buffers of `t` rows on this engine's stream.
                unsafe {
                    local.run(resident, t, input, w.route_ids.buffer.ptr, w.route_weights.buffer.ptr,
                        w.delta.buffer.ptr, self.stream)
                }
            }
            Experts::Streamed { experts, tensors, window } => {
                let mut local = experts.borrow_mut();
                if local.index_of(index).is_err() {
                    // SAFETY: the engine owns this stream; draining it retires every
                    // launch that read the layer about to be evicted.
                    unsafe { self.library.cuda_stream_synchronize(self.stream)? };
                    if local.layers.len() >= (*window).max(1) {
                        local.layers.remove(0);
                    }
                    let started = std::time::Instant::now();
                    local.layers.push(Fp8Layer::load(self.library, tensors, index, 1, 0)?);
                    self.profile.borrow_mut()[1] += started.elapsed().as_secs_f64();
                }
                let resident = local.index_of(index)?;
                let input = if local.wire_input() { w.wire.buffer.ptr } else { w.x.buffer.ptr };
                // SAFETY: as for `Local`; the layer stays resident until a later step evicts it
                // after draining the stream.
                unsafe {
                    local.run(resident, t, input, w.route_ids.buffer.ptr, w.route_weights.buffer.ptr,
                        w.delta.buffer.ptr, self.stream)
                }
            }
            // SAFETY: `delta` holds `t` rows on this engine's stream.
            Experts::Skip => unsafe {
                self.library.cuda_zero_bytes_async(w.delta.buffer, t * h * 2, self.stream)
            },
            Experts::Spark { .. } => anyhow::bail!("Spark experts run through the exchange"),
        }
    }

    /// Routes and wire rows down, one request to every Spark rank, the BF16
    /// rank partials summed into `delta` (GLM's exchange, no shared expert).
    #[allow(clippy::too_many_arguments)]
    fn spark_moe(&self, w: &Workspace<'_>, index: usize, t: usize, decode: bool, bf16_input: bool,
        transport: &mut SparkLink<'_>, runtime: &tokio::runtime::Runtime, forward: Option<usize>) -> Result<()> {
        let sent = self.spark_send(w, index, t, decode, bf16_input, false, transport)?;
        self.spark_land(w, index, t, transport, runtime, sent, forward, true)
    }

    /// The send half of [`Self::spark_moe`]: routes and wire rows down (the
    /// stream drains first), the request out to every rank.
    #[allow(clippy::too_many_arguments)]
    fn spark_send(&self, w: &Workspace<'_>, index: usize, t: usize, decode: bool, bf16_input: bool, staged: bool,
        transport: &mut SparkLink<'_>) -> Result<SentWave> {
        let kind = if decode { ExpertV2SourceKind::Decode } else { ExpertV2SourceKind::Prefill };
        let (h, topk) = (self.cfg.hidden, self.cfg.topk);
        let (route_bytes, wire_bytes) = (t * topk * 4, if bf16_input { t * h * 2 } else { t * (h + h / 32) });
        let (input, dtype) = if bf16_input { (&w.x, ExpertV2Dtype::Bf16) } else { (&w.wire, ExpertV2Dtype::Fp8E4m3Ue8m0K32) };
        let staging = w.router_host.borrow_mut();
        let host = staging.buffer;
        let at = |offset: usize| Self::staging_at(host, offset);
        let timer = std::time::Instant::now();
        // Prefill rows go straight into the transport's registered egress
        // buffer and out from there to every rank; small waves keep a copy.
        let egress = transport.egress(wire_bytes)?;
        // SAFETY: the pinned regions are large enough; the sync completes them.
        unsafe {
            // `staged`: a decode segment already queued these copies into the staging
            // ([`Self::stage_routes`]); only a wave bound for the egress buffer copies again.
            if !staged {
                self.library.copy_d2h_host_buffer_async(at(0), w.route_ids.buffer, route_bytes, self.stream)?;
                self.library.copy_d2h_host_buffer_async(at(route_bytes), w.route_weights.buffer, route_bytes,
                    self.stream)?;
            }
            match egress {
                Some(target) => self.library.copy_d2h_host_buffer_async(target, input.buffer, wire_bytes, self.stream)?,
                None if !staged => self.library.copy_d2h_host_buffer_async(at(2 * route_bytes), input.buffer,
                    wire_bytes, self.stream)?,
                None => {}
            }
        }
        // In a decode step the L2 prefetch queues behind the copies; the host waits for the copies only.
        match self.l2.as_ref().filter(|_| decode) {
            Some(l2) => {
                let mark = crate::shared::l2_prefetch::mark(self.library, self.stream)?;
                l2.issue(self.library, index, self.stream)?;
                crate::shared::l2_prefetch::reached(self.library, mark)?;
            }
            // SAFETY: the engine owns this stream.
            None => unsafe { self.library.cuda_stream_synchronize(self.stream)? },
        }
        let gpu_wait = timer.elapsed().as_secs_f64();
        self.profile.borrow_mut()[0] += gpu_wait;
        let built = std::time::Instant::now();
        let staged = staging.bytes();
        let word = |offset: usize, i: usize| u32::from_le_bytes(staged[offset + i * 4..][..4].try_into().unwrap());
        let routes = (0..t * topk).map(|i| ExpertProtocolV2RouteEntry {
            row_index: (i / topk) as u32, expert_id: word(0, i), gate_weight: f32::from_bits(word(route_bytes, i)),
        }).collect();
        let wire = match egress {
            Some(_) => transport.egress_payload(wire_bytes)?,
            None => staged[2 * route_bytes..2 * route_bytes + wire_bytes].to_vec().into(),
        };
        drop(staging);
        let mut request = ExpertProtocolV2Request::new_bytes(index as u64 + 1, 17, index as u32, h as u32, dtype,
            (0..t as u32).map(|row| ExpertProtocolV2RowDescriptor {
                row_id: u64::from(row), source_kind: kind, source_request_id: 1,
                token_position: u64::from(row), route_offset: row * topk as u32, route_count: topk as u32,
            }).collect(),
            routes, wire)?;
        request.header.flags |= EXPERT_PROTOCOL_V2_FLAG_V41_COMPACT_BF16;
        let ranks = transport.world_size();
        ensure!(ranks <= MAX_RANKS, "{ranks} Spark ranks exceed the reduction planes");
        let build = built.elapsed().as_secs_f64();
        let timer = std::time::Instant::now();
        let wave = transport.dispatch(&request)?;
        Ok(SentWave { wave: Some(wave), gpu_wait, build, dispatch: timer.elapsed().as_secs_f64(), sent: timer })
    }

    /// `offset` bytes into the pinned router staging.
    fn staging_at(host: cuteafd_ffi::CuteafdHostBuffer, offset: usize) -> cuteafd_ffi::CuteafdHostBuffer {
        cuteafd_ffi::CuteafdHostBuffer {
            // SAFETY: ids, weights and wire rows are consecutive inside the pinned buffer.
            ptr: unsafe { host.ptr.cast::<u8>().add(offset) }.cast(),
            bytes: host.bytes - offset,
            ..host
        }
    }

    /// Queues a decode wave's routes and expert input rows into the pinned router
    /// staging, where [`Self::spark_send`] (with `staged`) reads them after its sync.
    fn stage_routes(&self, w: &Workspace<'_>, t: usize, bf16_input: bool) -> Result<()> {
        let (h, topk) = (self.cfg.hidden, self.cfg.topk);
        let (route_bytes, wire_bytes) = (t * topk * 4, if bf16_input { t * h * 2 } else { t * (h + h / 32) });
        let input = if bf16_input { &w.x } else { &w.wire };
        let host = w.router_host.borrow().buffer;
        ensure!(2 * route_bytes + wire_bytes <= host.bytes, "router staging of {} bytes for {t} rows", host.bytes);
        let at = |offset: usize| Self::staging_at(host, offset);
        // SAFETY: the pinned regions hold these bytes (checked above); the host reads them
        // only after the stream passes the copies (spark_send's sync), and the next
        // segment rewrites them only after the request was built from them.
        unsafe {
            self.library.copy_d2h_host_buffer_async(at(0), w.route_ids.buffer, route_bytes, self.stream)?;
            self.library.copy_d2h_host_buffer_async(at(route_bytes), w.route_weights.buffer, route_bytes, self.stream)?;
            self.library.copy_d2h_host_buffer_async(at(2 * route_bytes), input.buffer, wire_bytes, self.stream)
        }
    }

    /// The land half: receives the wave into the transport's planes and
    /// queues their sum into `delta` (forwarded to the head split's second GPU
    /// when `forward`); `drain` waits for the stream after it.
    #[allow(clippy::too_many_arguments)]
    fn spark_land(&self, w: &Workspace<'_>, index: usize, t: usize, transport: &mut SparkLink<'_>,
        runtime: &tokio::runtime::Runtime, sent: SentWave, forward: Option<usize>, drain: bool) -> Result<()> {
        let h = self.cfg.hidden;
        let mut sent = sent;
        let receive = self.spark_receive(t, transport, runtime, &mut sent)?;
        let reduced = std::time::Instant::now();
        // SAFETY: the zero plane and delta are live [t, h] BF16 buffers; the
        // intake planes are ordered after the wave by `receive`.
        unsafe {
            transport.reduce(w.zero_plane.buffer.ptr.cast(), w.delta.buffer.ptr.cast(), t, self.stream)?;
            if let Some(slot) = forward {
                self.push(0, slot, w.delta.buffer.ptr, t * h * 2)?;
            }
            if drain {
                // The next layer's request staging is rewritten only after this drains.
                self.library.cuda_stream_synchronize(self.stream)?;
            }
        }
        self.wave_line(index, t, &sent, receive, reduced.elapsed().as_secs_f64());
        Ok(())
    }

    /// Receives `sent`'s wave into the transport's intake planes (the stream is ordered
    /// after it); returns the host seconds spent receiving.
    fn spark_receive(&self, t: usize, transport: &mut SparkLink<'_>, runtime: &tokio::runtime::Runtime,
        sent: &mut SentWave) -> Result<f64> {
        let waited = std::time::Instant::now();
        let wave = sent.wave.take().context("wave already received")?;
        runtime.block_on(transport.receive(wave, t, self.stream))?;
        self.profile.borrow_mut()[1] += sent.sent.elapsed().as_secs_f64();
        Ok(waited.elapsed().as_secs_f64())
    }

    /// `CUTEAFD_MIMO_WAVE_TIMING=1`: one line per Spark wave.
    fn wave_line(&self, index: usize, t: usize, sent: &SentWave, receive: f64, reduce: f64) {
        if self.wave_timing {
            eprintln!("mimo_wave layer={index} rows={t} gpu_wait_ms={:.3} build_ms={:.3} dispatch_ms={:.3} \
                receive_ms={:.3} reduce_ms={:.3}", sent.gpu_wait * 1e3, sent.build * 1e3, sent.dispatch * 1e3,
                receive * 1e3, reduce * 1e3);
        }
    }
}

#[cfg(test)]
#[path = "terminal_fault_tests.rs"]
mod terminal_fault_tests;

#[cfg(test)]
mod tests {
    use super::{attention_workspace_geometry, expert_capacity, independent_prefill_rows, lane_prefill_capacity, prefill_lane_count,
        scale_operand, MimoAttentionWorkspace, MimoPlacement, MimoPrefillOutput, DECODE_ROWS, MIN_LANE_ROWS, PAGE_ROWS};

    #[test]
    fn fp8_output_scale_operands_match_both_exported_abis() {
        for (decode, expected) in [(true, "w_o_scale"), (false, "w_o_kscale")] {
            let operands = ["attn", "w_o_fp8", scale_operand("w_o", decode).unwrap(), "out", "scratch"];
            assert_eq!(operands, ["attn", "w_o_fp8", expected, "out", "scratch"]);
        }
        for (weight, row, kmajor) in [("w_qkv", "w_qkv_scale", "w_qkv_kscale"),
            ("w_gate_up", "w_gate_up_scale", "w_gate_up_kscale"), ("w_down", "w_down_scale", "w_down_kscale")] {
            assert_eq!(scale_operand(weight, true).unwrap(), row);
            assert_eq!(scale_operand(weight, false).unwrap(), kmajor);
        }
        for decode in [true, false] {
            assert!(scale_operand("w_unregistered", decode).unwrap_err().to_string().contains("no scale operand"));
        }
    }

    #[test]
    fn narrow_prefill_expert_storage_covers_every_decode_descriptor_and_native_program() {
        use cuteafd_ffi::fp8_moe::{Fp8MoeInfo, Fp8MoeWeights};
        let info = Fp8MoeInfo { hidden: 4096, slice: 2048, experts: 256, topk: 8,
            intermediate: 2048, tp: 1, wire_input: false, swiglu_limit: 0.0,
            capacities: vec![16, 64, 4096], weights: Fp8MoeWeights::Fp8 };
        for prefill in [1, 16, 32, 63, 64, 128, 4096] {
            // Local load/scratch, SparkLink negotiation/intake, and warmup
            // all consume this common extent; the native program must fit it.
            let capacity = expert_capacity(prefill);
            let program = info.capacity_for(capacity).unwrap();
            let partial_plane = capacity * info.hidden * 2;
            for decode_rows in 1..=DECODE_ROWS {
                assert!(decode_rows <= capacity && decode_rows <= program);
                assert!(decode_rows * info.hidden * 2 <= partial_plane);
            }
            assert!(prefill <= capacity);
            if prefill >= DECODE_ROWS { assert_eq!(capacity, prefill); }
        }
        assert_eq!(expert_capacity(4096), 4096);
        let narrow_only = Fp8MoeInfo { capacities: vec![16], ..info };
        assert_eq!(narrow_only.capacity_for(expert_capacity(1)), None);
    }

    #[test]
    fn split_prefill_and_peer_workspaces_use_only_their_attention_heads() {
        let split = MimoAttentionWorkspace::PartitionedHeads { ranks: 2 };
        assert_eq!(attention_workspace_geometry(0, 2, false, true), split);
        assert_eq!(attention_workspace_geometry(1, 2, false, true), split);
        assert_eq!(attention_workspace_geometry(1, 2, true, true), split);
        // The attached peer always contains split shares, even if a future
        // target mix still requires global buffers on the leading GPU.
        assert_eq!(attention_workspace_geometry(1, 2, false, false), split);
    }

    #[test]
    fn lead_decode_and_unsplit_target_workspaces_preserve_global_attention() {
        let global = MimoAttentionWorkspace::Global;
        // MTP uses rank 0's decode workspace and whole-model programs.
        assert_eq!(attention_workspace_geometry(0, 2, true, true), global);
        assert_eq!(attention_workspace_geometry(0, 2, false, false), global);
        assert_eq!(attention_workspace_geometry(0, 1, false, true), global);
        assert_eq!(attention_workspace_geometry(0, 1, true, true), global);
    }

    #[test]
    fn advertised_prefill_capacity_accepts_every_prompt_tail() {
        for rows in [1, 512, 1024, 1536, 2047, 2048, 4096] {
            assert_eq!(lane_prefill_capacity(rows, 1), rows);
            for lanes in 2..=4 {
                let capacity = lane_prefill_capacity(rows, lanes);
                for tokens in 1..=capacity {
                    let n = prefill_lane_count(tokens, rows, lanes);
                    assert!((1..=lanes).contains(&n));
                    if tokens > rows {
                        assert!(n >= 2, "{tokens}-row tail exceeds its {rows}-row workspace but cannot use lanes");
                    }
                    if n > 1 {
                        // The engine's split: every lane fits its workspace and keeps the tap window.
                        let split = tokens.div_ceil(n);
                        for i in 0..n {
                            let lane = ((i + 1) * split).min(tokens) - i * split;
                            assert!(lane <= rows && lane >= MIN_LANE_ROWS, "{tokens} rows, {n} lanes of {rows}: {lane}");
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn independent_short_prompts_fit_without_changing_single_prompt_split_threshold() {
        let last = MimoPrefillOutput::LastRow;
        for rows in [[1, 1], [17, 1463], [1023, 1025], [1464, 1463], [2047, 1], [2047, 2047]] {
            assert!(independent_prefill_rows(4096, true, last, false, rows));
            assert!(!independent_prefill_rows(4096, false, last, false, rows));
            assert!(!independent_prefill_rows(4096, true, last, true, rows));
            assert!(!independent_prefill_rows(4096, true, MimoPrefillOutput::AllRows, false, rows));
        }
        for rows in [[0, 1463], [1463, 0], [2048, 1], [1, 4096], [4097, 1463], [1463, 8192]] {
            assert!(!independent_prefill_rows(4096, true, last, false, rows));
        }
        assert_eq!(MIN_LANE_ROWS, 1024);
        assert_eq!(lane_prefill_capacity(1024, 2), 1024);
    }

    #[test]
    fn independent_tables_preserve_different_cached_frontiers_and_private_write_pages() {
        // Both own a shared read-only prefix page but append to distinct tail
        // pages/rings at different cached positions; no packed attention rows.
        let a = MimoPlacement { pages: vec![3, 7], ring: 2, len: PAGE_ROWS };
        let b = MimoPlacement { pages: vec![3, 9], ring: 7, len: PAGE_ROWS + 2 };
        let a_table = a.prefill_tables(3, 128).unwrap();
        let b_table = b.prefill_tables(1, 128).unwrap();
        assert_eq!(a_table.positions, [64, 65, 66]);
        assert_eq!(b_table.positions, [66]);
        assert_eq!(a_table.slots, [448, 449, 450]);
        assert_eq!(b_table.slots, [578]);
        assert_eq!(a_table.page_table, [3, 7]);
        assert_eq!(b_table.page_table, [3, 9]);
        assert_eq!(a_table.ring_slots, [576, 577, 578]);
        assert_eq!(b_table.ring_slots, [1858]);
        assert_eq!(a_table.seq_first, [0, 0, 0]);
        assert_eq!(b_table.seq_first, [0]);
        assert_eq!((a_table.table_stride, b_table.table_stride), (0, 0));
        assert_eq!((a.len, b.len), (64, 66));
        assert!(a.prefill_tables(65, 256).is_err(), "unadmitted pages cannot be queued");
        assert!(b.prefill_tables(1, 66).is_err(), "context overflow cannot be queued");
    }
}
