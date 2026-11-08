//! DeepSeek V4 coordinator: one sequence prefilled from position 0 through the
//! exported programs, routed experts on the Spark ranks.
//!
//! This is the correctness path the serving engine grows from: every layer
//! runs the prototype's op sequence (mHC pre, producer, compressor, index
//! top-k, sparse MLA, wo, mHC post_pre, router, experts + shared FFN, mHC
//! post) and each stage's buffers are sized for the prompt.
use super::metadata::{self, StepTables, INDEX_PAGE_BYTES, MAIN_PAGE_BYTES};
use super::pool::{Placement, PoolShape};
use std::cell::{Cell, RefCell};
use super::weights::{LayerWeights, ModelWeights};
use crate::shared::launch_grid::Fp8QuantizeGrid;
use crate::shared::memory::{DeviceAllocation, HostAllocation, Tag};
use crate::shared::token_io::{DeviceLogits, TokenEmbedding};
use anyhow::{ensure, Context, Result};
use cuteafd_ffi::programs::{Program, Programs, Scalar};
use cuteafd_ffi::NativeLibrary;
use cuteafd_core::coordinator_programs::AttentionReductionPlan;
use cuteafd_loader::families::deepseek_v4::DeepseekV4Config;
use crate::shared::spark_intake::SparkLink;
use cuteafd_transport::expert::EXPERT_PROTOCOL_V2_FLAG_V41_COMPACT_BF16;
use cuteafd_transport::{
    ExpertProtocolV2Request, ExpertProtocolV2RouteEntry, ExpertProtocolV2RowDescriptor, ExpertV2Dtype,
    ExpertV2SourceKind,
};
use crate::shared::peer_split::{PeerExchange, RankDevice, DIRECT};
use std::ffi::c_void;
use std::time::Instant;

mod context;
mod dspark;
mod tp2;
mod prefill;
mod packing;
#[cfg(test)]
mod packed_check;
pub(crate) use tp2::ExchangePolicy;
use tp2::Tp2State;
pub(crate) use dspark::DraftRequest;

/// Per-step and per-launch scratch: lanes, attention planes, router, sampler and expert wire rows.
pub(super) const WORKSPACE: Tag = Tag::workspace("v4/workspace");
/// Per-step table metadata: positions, destination slots, compact page tables.
pub(super) const STEP_TABLES: Tag = Tag::workspace("v4/workspace/step");
/// dSpark drafter per-step scratch: taps, main projection, draft ids and markov state.
pub(super) const DRAFT_WORKSPACE: Tag = Tag::drafter_workspace("v4/draft/workspace");
/// Target KV pools: window rings, C4/C128 page pools and compressor state.
pub(super) const KV: Tag = Tag::kv("v4/kv");
/// Pool-following page tables placement charges as KV records.
pub(super) const KV_RECORDS: Tag = Tag::kv("v4/kv/records");
/// Generated RoPE cos/sin tables.
pub(super) const ROPE: Tag = Tag::weights("v4/rope");
/// Peer/TP2 reply and reduce landings.
pub(super) const REDUCE_LANDING: Tag = Tag::transport("v4/transport/reduce");

#[derive(Clone)]
pub(crate) struct Dev<'a> {
    pub(crate) buffer: cuteafd_ffi::CuteafdDeviceBuffer,
    _owner: std::rc::Rc<DeviceAllocation<'a>>,
}

impl<'a> From<DeviceAllocation<'a>> for Dev<'a> {
    fn from(allocation: DeviceAllocation<'a>) -> Self {
        Self { buffer: allocation.buffer, _owner: std::rc::Rc::new(allocation) }
    }
}

impl<'a> Dev<'a> {
    fn rows(&self, slot: usize, bytes: usize) -> Self {
        let offset = slot * bytes;
        assert!(offset + bytes <= self.buffer.bytes);
        // SAFETY: each slot is a disjoint, bounded view; the shared owner outlives every view.
        let ptr = unsafe { self.buffer.ptr.cast::<u8>().add(offset) }.cast();
        Self { buffer: cuteafd_ffi::CuteafdDeviceBuffer { ptr, bytes, ..self.buffer },
            _owner: std::rc::Rc::clone(&self._owner) }
    }
}

pub(crate) struct Engine<'a> {
    pub library: &'a NativeLibrary,
    pub programs: &'a Programs<'a>,
    pub cfg: DeepseekV4Config,
    pub weights: ModelWeights<'a>,
    pub family: &'static str,
    pub decode_rows: usize,
    pub prefill_rows: usize,
    /// Export row capacity used for program names, independent of live unit cuts.
    pub prefill_kernel_rows: usize,
    pub prefill_waves: usize,
    /// Frozen outer intake; independent of live slots and compiled kernel rows.
    pub prefill_span_rows: usize,
    pub full_prefill_logits: bool,
    pub c128_width: usize,
    /// Longest sequence the exported programs' cache extents cover.
    pub max_context: usize,
    pub stream: *mut c_void,
    quantize_grid: Fp8QuantizeGrid,
    attention_reduction: AttentionReductionPlan,
    pub shape: PoolShape,
    pools: Vec<LayerCache<'a>>,
    rope_window: Dev<'a>,
    rope_compressed: Dev<'a>,
    /// Prefill and decode workspaces, reused across steps.
    prefill_workspace: RefCell<Option<Workspace<'a>>>,
    packed_row_cap: Cell<Option<usize>>,
    packed_programs: crate::shared::prefill_packing::ProgramInventory,
    decode_workspace: RefCell<Option<Workspace<'a>>>,
    pub profile: RefCell<Profile>,
    graphs: RefCell<std::collections::HashMap<GraphKey, GraphExec<'a>>>,
    /// Expert layers resident on this GPU (they skip the Spark exchange).
    pub local: RefCell<Option<super::local::LocalExperts<'a>>>,
    /// The token embedding table (resident on this GPU or read from its shard).
    pub embedding: TokenEmbedding<'a>,
    /// Benchmarks and token gates only: MoE layers run the shared expert alone.
    pub skip_routed: bool,
    /// This engine's GPU (rank 0 of a head split).
    pub device: i32,
    /// Program family of one GPU's share under a head split (`dsv4f2`, `dsv4p2`).
    split_family: Option<&'static str>,
    /// The head split's second GPU and the exchange between the two.
    peer: Option<V4Peer<'a>>,
    exchange: Option<PeerExchange<'a>>,
    tp2: Option<Tp2State<'a>>,
    terminal: Cell<crate::shared::peer_split::TerminalState>,
    exchange_policy: ExchangePolicy,
    /// C4 token-split attention (#58): the build's programs, set at construction (each GPU's C4
    /// pools then hold half the units, plus per-layer scratch pages), and once the peer is
    /// attached, its buffers and exchange (dropped before the head split's exchange, whose abort
    /// words it borrows).
    context_mode: Option<cuteafd_loader::placement::families::deepseek_v4::V4Context>,
    context: Option<context::V4ContextSplit<'a>>,
    /// Per lane and rank: this step's context tables (gather and commit counts).
    context_units: RefCell<Vec<[context::ContextUnit; 2]>>,
    capture_only: std::cell::Cell<bool>,
    declared_graphs: RefCell<Vec<cuteafd_loader::placement::inventory::DeclaredGraphSet<GraphKey>>>,
    graph_captures: std::cell::Cell<usize>,
    expert_graph_captures: std::cell::Cell<usize>,
    pub(super) graph_coverage: crate::shared::decode_graph::census::LaunchCoverage,
    draft_probe: RefCell<Option<crate::shared::draft::probe::RoundProbe<'a>>>,
    expert_usage: Option<cuteafd_ffi::expert_usage::ExpertUsage<'a>>,
}

/// The second GPU of a two-GPU head split (rank 1): its share of every
/// backbone layer (half the heads' `w_q` rows and sinks, half the output
/// groups, half the shared expert), its copy of the caches and compressor
/// state (the replicated mHC, latent projection, compressors and indexer
/// keep them identical), RoPE tables, workspaces and decode graphs.
pub(crate) struct V4Peer<'a> {
    pub device: i32,
    pub stream: *mut c_void,
    pub layers: Vec<LayerWeights<'a>>,
    pools: Vec<LayerCache<'a>>,
    rope_window: Dev<'a>,
    rope_compressed: Dev<'a>,
    prefill_workspace: RefCell<Option<Workspace<'a>>>,
    decode_workspace: RefCell<Option<Workspace<'a>>>,
    graphs: RefCell<std::collections::HashMap<GraphKey, GraphExec<'a>>>,
}

/// Exchange slot of layer `layer`, lane `lane`: its attention partials (`ffn`
/// false) or its FFN exchange (rank 1's shared-expert half, rank 0's routed +
/// shared sum), by layer parity.
fn slot(layer: usize, ffn: bool, lane: usize) -> usize {
    4 * lane + 2 * (layer % 2) + usize::from(ffn)
}

/// `exchange` result for a layer whose experts ran on this GPU.
const LOCAL_EXPERTS: usize = cuteafd_loader::serving_capacity::v4_graphs::LOCAL_EXPERTS;
/// `exchange` result for a layer whose routed experts were skipped.
const SKIPPED_EXPERTS: usize = cuteafd_loader::serving_capacity::v4_graphs::SKIPPED_EXPERTS;

/// Vocabulary logits rows a workspace holds (every decode/verify row; a prefill
/// lands at most this many rows at once and downloads longer spans in chunks).
const LOGIT_ROWS: usize = 64;
/// Rows of a step whose target taps feed the drafter's main KV: every row of
/// a decode step, the last window of a prefill lane.
const TAP_ROWS: usize = metadata::WINDOW;

/// One lane's inputs for a step.
struct RowStep<'s> {
    tables: &'s StepTables,
    tokens: &'s [u32],
}

use cuteafd_loader::serving_capacity::v4_graphs::{V4GraphKey as GraphKey, TAIL_SEGMENT};

trait GraphCoverage {
    fn coverage_segments(self, rank: usize, skip_routed: bool, tp2_layer: bool,
        draft_ready: bool, last_layer: usize, tap_layers: &[usize],
        record: impl FnMut(usize, &'static str));
}

impl GraphCoverage for GraphKey {
    fn coverage_segments(self, rank: usize, skip_routed: bool, tp2_layer: bool,
        draft_ready: bool, last_layer: usize, tap_layers: &[usize],
        mut record: impl FnMut(usize, &'static str)) {
        let lane = if self.attention == "local-experts" || self.attention.starts_with("tp2-") { self.previous } else { 0 };
        let tap = |layer, record: &mut dyn FnMut(usize, &'static str)| {
            if rank == 0 && draft_ready && tap_layers.contains(&layer) { record(lane, "tap"); }
        };
        match self.attention {
            "tail" => {
                if self.previous > 0 { record(lane, "post"); tap(last_layer, &mut record); }
                if draft_ready { record(lane, "draft-kv"); }
                record(lane, "head");
            }
            "local-experts" | "tp2-replicated" | "tp2-broadcast" => {
                if rank == 0 { record(lane, "shared"); }
                record(lane, "partial");
            }
            _ => {
                if (rank == 0 && self.previous > 0) || (rank == 1 && self.layer > 0) {
                    record(lane, "post");
                    if self.layer > 0 { tap(self.layer - 1, &mut record); }
                }
                record(lane, "front"); record(lane, "back");
                if !skip_routed && (rank == 0 || tp2_layer) { record(lane, "router"); }
                if rank == 1 { record(lane, "shared"); }
            }
        }
    }
}

fn startup_capture_allowed(warming: bool, declared: bool) -> bool { warming && declared }

struct GraphExec<'a>(*mut c_void, &'a NativeLibrary);

impl Drop for GraphExec<'_> {
    fn drop(&mut self) {
        // SAFETY: the exec came from end_capture and is destroyed once.
        let _ = unsafe { self.1.cuda_graph_exec_destroy(self.0) };
    }
}

/// Host-visible phases of a step, for profiling.
#[derive(Debug, Clone, Copy)]
pub(crate) enum Phase {
    /// Waiting for the GPU through the router and input quantizer.
    RouterSync,
    Routing,
    /// Dispatch, remote compute and partials landing in device planes.
    Experts,
    /// Waiting for the head and downloading logits.
    Head,
}

#[derive(Debug, Default)]
pub(crate) struct Profile {
    pub seconds: [f64; 4],
}

impl Profile {
    fn add(&mut self, phase: Phase, since: Instant) {
        self.seconds[phase as usize] += since.elapsed().as_secs_f64();
    }

    pub fn report(&self) -> String {
        let names = ["router_sync", "routing", "experts", "head"];
        names.iter().zip(self.seconds).map(|(n, s)| format!("{n} {:.1} ms", s * 1e3)).collect::<Vec<_>>().join(", ")
    }
}

/// What [`Engine::attach_peer`] sizes the second GPU's caches with.
#[derive(Debug, Clone, Copy)]
pub(crate) struct PeerParts {
    pub shape: PoolShape,
    pub tp2: bool,
}

/// Everything an [`Engine`] needs besides its pools and workspaces.
pub(crate) struct EngineParts<'a> {
    pub expert_usage: Option<(cuteafd_core::expert_usage::Geometry, cuteafd_core::expert_usage::Reader)>,
    pub library: &'a NativeLibrary,
    pub programs: &'a Programs<'a>,
    pub cfg: DeepseekV4Config,
    pub weights: ModelWeights<'a>,
    pub family: &'static str,
    pub decode_rows: usize,
    pub prefill_rows: usize,
    /// Export row capacity used for program names, independent of live unit cuts.
    pub prefill_kernel_rows: usize,
    pub prefill_waves: usize,
    /// Frozen outer intake; independent of live slots and compiled kernel rows.
    pub prefill_span_rows: usize,
    pub c128_width: usize,
    pub max_context: usize,
    pub stream: *mut c_void,
    pub sms: Option<usize>,
    pub shape: PoolShape,
    pub embedding: TokenEmbedding<'a>,
    pub skip_routed: bool,
    /// C4 token-split attention's programs (`Some` only when the placement chose `context`).
    pub context: Option<cuteafd_loader::placement::families::deepseek_v4::V4Context>,
}

/// C4 pool pages one GPU allocates: every unit, or under `context` its half plus the scratch
/// pages non-owner compressor rows write (one 64-row page per 64 decode/verify rows).
fn c4_units(shape: PoolShape, context: bool) -> usize {
    if context { shape.units / 2 + metadata::C4_SCRATCH_UNITS } else { shape.units }
}

fn bytes_of<T: Copy>(values: &[T]) -> &[u8] {
    // SAFETY: plain-old-data slices viewed as bytes for host->device copies.
    unsafe { std::slice::from_raw_parts(values.as_ptr().cast(), std::mem::size_of_val(values)) }
}

/// Per-layer caches and compressor state for one sequence.
pub(crate) struct LayerCache<'a> {
    pub main: Dev<'a>,
    pub compressed: Option<Dev<'a>>,
    pub index: Option<Dev<'a>>,
    pub states: Vec<Dev<'a>>,
}

/// What one lane of rows carries from layer to layer: the residual streams,
/// the mHC post/comb of its current layer, its token ids and step tables.
struct UnitSlot<'a> {
    /// This lane's position in its workspace (context tables and commit lists are per lane).
    index: usize,
    stream_a: Dev<'a>,
    stream_b: Dev<'a>,
    post: Dev<'a>,
    comb: Dev<'a>,
    /// Token ids (hash-layer routing).
    tokens: Dev<'a>,
    /// Shared-expert output of the lane's current layer (its post reads it
    /// after the other lane's shared expert ran).
    shared: Dev<'a>,
    /// Persistent TP2 payload: consumed before this lane advances to another layer.
    payload: Option<Dev<'a>>,
    tables: StepBuffers<'a>,
    /// dSpark target taps of the lane's last TAP_ROWS rows, BF16
    /// [admitted tap rows, taps * dim] (empty without a drafter).
    taps: Dev<'a>,
}

/// Step workspace sized for `rows` rows per lane; everything but the lanes is
/// consumed within one (layer, lane) unit and shared.
struct Workspace<'a> {
    slots: Vec<UnitSlot<'a>>,
    y: Dev<'a>,
    query: Dev<'a>,
    q_rank: Dev<'a>,
    attn_out: Dev<'a>,
    delta: Dev<'a>,
    index_query: Dev<'a>,
    index_weights: Dev<'a>,
    selected: Dev<'a>,
    topk_scratch: Dev<'a>,
    logits: Dev<'a>,
    /// Routes: expert ids (U32) and weights (FP32), [rows, topk].
    route_ids: Dev<'a>,
    route_weights: Dev<'a>,
    wire: Dev<'a>,
    scratch: Dev<'a>,
    dummy: Dev<'a>,
    vocab_logits: Dev<'a>,
    /// dSpark: projected taps (BF16) and the projection's FP32 rows, first
    /// tokens and drafts (U32), Markov argmax partials.
    main_x: Dev<'a>,
    main_work: Dev<'a>,
    first_tokens: Dev<'a>,
    drafts: Dev<'a>,
    markov: Dev<'a>,
    /// dSpark: the draft block's token ids (token, then noise) for the gather.
    draft_ids: Dev<'a>,
    /// A head split's sums of the two GPUs' partials ([rows, dim] BF16).
    sum: Dev<'a>,
    // Drops before its workspace below (rank 0 only).
    head: Option<cuteafd_ffi::programs::VocabularyHead<'a>>,
    _head_workspace: Dev<'a>,
}

/// Device copies of one step's tables.
#[derive(Clone)]
struct StepBuffers<'a> {
    /// Rows these buffers hold; tables are copied in per step.
    rows: usize,
    positions: Dev<'a>,
    main_slots: Dev<'a>,
    swa_indices: Dev<'a>,
    swa_lengths: Dev<'a>,
    c4: Vec<Dev<'a>>,
    c128: Vec<Dev<'a>>,
    c4_page_table: Dev<'a>,
    c4_visible: Dev<'a>,
    c4_indexed_lengths: Dev<'a>,
    c128_indices: Dev<'a>,
    c128_lengths: Dev<'a>,
}

impl<'a> Engine<'a> {
    fn alloc(&self, bytes: usize, tag: Tag) -> Result<Dev<'a>> {
        {
            let owner = std::rc::Rc::new(DeviceAllocation::new(self.library, bytes.max(256), tag)?);
            Ok(Dev { buffer: owner.buffer, _owner: owner })
        }
    }

    fn zeroed(&self, bytes: usize, tag: Tag) -> Result<Dev<'a>> {
        let allocation = self.alloc(bytes, tag)?;
        self.library.cuda_zero_bytes(allocation.buffer, allocation.buffer.bytes)?;
        Ok(allocation)
    }

    fn kernel_rows(&self, rows: usize) -> usize {
        if rows == self.prefill_rows { self.prefill_kernel_rows } else { rows }
    }

    fn program(&self, name: &str, pointers: &[&str]) -> Result<Program<'a>> {
        self.programs.program(&format!("{}_{name}", self.family), pointers)
    }

    fn run(&self, name: &str, pointers: &[(&str, *mut c_void)], scalars: &[Scalar]) -> Result<()> {
        self.run_on(0, false, name, pointers, scalars)
    }

    /// Launches `name` on rank `rank`'s stream; `split` picks the head-split
    /// share's program (`{split_family}_{name}`).
    fn run_on(&self, rank: usize, split: bool, name: &str, pointers: &[(&str, *mut c_void)], scalars: &[Scalar])
        -> Result<()> {
        let family = if split { self.split_family.context("no head-split programs")? } else { self.family };
        let names: Vec<&str> = pointers.iter().map(|(n, _)| *n).collect();
        let full_name = format!("{family}_{name}");
        self.packed_programs.check(&full_name)?;
        let program = self.programs.program(&format!("{family}_{name}"), &names)?;
        let raw: Vec<*mut c_void> = pointers.iter().map(|(_, p)| *p).collect();
        // SAFETY: every pointer names a live allocation of rank `rank`'s GPU sized for
        // the rows in `scalars`; that rank's stream orders all its launches.
        self.on(rank, || unsafe { program.launch(&raw, scalars, self.stream_of(rank)) })
            .with_context(|| format!("{name} with scalars {scalars:?}"))
    }

    /// GPUs this engine runs on: 2 under a head split.
    pub(crate) fn ranks(&self) -> usize {
        1 + usize::from(self.peer.is_some())
    }

    /// The stream of rank `rank`.
    pub(crate) fn stream_of(&self, rank: usize) -> *mut c_void {
        match (rank, &self.peer) {
            (1, Some(peer)) => peer.stream,
            _ => self.stream,
        }
    }

    /// Runs `body` with rank `rank`'s device current, preserving the caller's device.
    pub(crate) fn on<T>(&self, rank: usize, body: impl FnOnce() -> Result<T>) -> Result<T> {
        let device = self.peer.as_ref().filter(|_| rank == 1).map_or(self.device, |peer| peer.device);
        crate::shared::peer_split::on_device(self.library, device, body)
    }

    fn peer(&self) -> Result<&V4Peer<'a>> {
        self.peer.as_ref().context("no head-split peer")
    }

    fn exchange(&self) -> Result<&PeerExchange<'a>> {
        self.exchange.as_ref().context("no head-split exchange")
    }

    /// Attaches the head split's second GPU: `device` with `stream`, holding
    /// `layers` (every backbone layer's rank-1 share). Loads the programs there and
    /// allocates its caches, RoPE tables and the exchange (four slots per prefill lane).
    pub fn attach_peer(&mut self, device: i32, stream: *mut c_void, layers: Vec<LayerWeights<'a>>, parts: PeerParts)
        -> Result<()> {
        let _memory_scope = cuteafd_ffi::memory_ledger::scope("peer-split");
        ensure!(self.split_family.is_some() && layers.len() == self.cfg.n_layers && layers.iter().all(|l| l.attention_split && l.ffn_split),
            "attach_peer needs the head-split shares of every backbone layer");
        let slot_bytes = if parts.tp2 || self.exchange_policy == ExchangePolicy::F32 {
            self.exchange_policy.slot_bytes(self.prefill_rows, self.decode_rows, self.cfg.dim)
        } else { self.prefill_rows.max(self.decode_rows) * self.cfg.dim * 2 };
        let exchange = PeerExchange::new_abortable(self.library, [RankDevice { device: self.device, stream: self.stream },
            RankDevice { device, stream }], 4 * self.prefill_waves, slot_bytes,
            self.exchange_policy.decode_bytes(self.decode_rows, self.cfg.dim))?;
        let peer = exchange.on(1, || -> Result<V4Peer<'a>> {
            self.programs.load_matching(|name| self.selected_programs().contains(name))?;
            // The peer's caches and RoPE tables are KV, not exchange storage.
            let _memory_scope = cuteafd_ffi::memory_ledger::scope("kv");
            let table = |compressed: bool| -> Result<Dev<'a>> {
                let values = metadata::rope_table(&self.cfg, compressed, self.max_context.max(metadata::WINDOW));
                let allocation = DeviceAllocation::new(self.library, values.len() * 4, ROPE)?;
                self.library.copy_h2d(allocation.buffer, bytes_of(&values))?;
                Ok(allocation.into())
            };
            let pools = (0..self.cfg.n_layers).map(|l| Self::pool_layer_for(self.library, &self.cfg, parts.shape, l,
                self.context_mode.is_some())).collect::<Result<Vec<_>>>()?;
            Ok(V4Peer { device, stream, layers, pools, rope_window: table(false)?, rope_compressed: table(true)?,
                prefill_workspace: RefCell::new(None), decode_workspace: RefCell::new(None),
                graphs: RefCell::new(std::collections::HashMap::new()) })
        })?;
        self.peer = Some(peer);
        self.exchange = Some(exchange);
        Ok(())
    }

    fn selected_programs(&self) -> cuteafd_core::coordinator_programs::CoordinatorPrograms<'_> {
        cuteafd_core::coordinator_programs::CoordinatorPrograms { family: self.family, split_family: self.split_family }
    }

    /// One arena per dispatch shape; launches within an arena are stream ordered.
    fn scratch_bytes(&self, prefill: bool) -> Result<usize> {
        let mut sizes = Vec::new();
        let rows = if prefill { self.prefill_kernel_rows } else { self.decode_rows };
        for name in self.programs.names().filter(|name| self.selected_programs().workspace_program(name, rows as u64, prefill)) {
            for value in self.programs.spec(name)?.scratch.values() {
                sizes.push((name, *value));
            }
        }
        Ok(usize::try_from(self.selected_programs().shared_scratch(sizes))?)
    }

    fn pool_layer(parts: &EngineParts<'a>, layer: usize) -> Result<LayerCache<'a>> {
        Self::pool_layer_for(parts.library, &parts.cfg, parts.shape, layer, parts.context.is_some())
    }

    /// Layer `layer`'s zeroed caches and compressor state on the current device. Under `context`
    /// the C4 pools hold this GPU's half of the units (local page `id >> 1`); C128 pools, window
    /// rings and carry stay whole on both GPUs.
    fn pool_layer_for(library: &'a NativeLibrary, cfg: &DeepseekV4Config, shape: PoolShape, layer: usize,
        context: bool) -> Result<LayerCache<'a>> {
        let _memory_scope = cuteafd_ffi::memory_ledger::scope("kv");
        let zeroed = |bytes: usize| -> Result<Dev<'a>> {
            let allocation = DeviceAllocation::new(library, bytes.max(256), KV)?;
            library.cuda_zero_bytes(allocation.buffer, allocation.buffer.bytes)?;
            Ok(allocation.into())
        };
        let sequences = shape.sequences;
        let main = zeroed(shape.window_pages() * MAIN_PAGE_BYTES)?;
        let (compressed, index, states) = match cfg.compress_ratios.get(layer).copied().unwrap_or(0) {
            4 => (
                Some(zeroed(c4_units(shape, context) * metadata::compressed_page_bytes(4))?),
                Some(zeroed(c4_units(shape, context) * INDEX_PAGE_BYTES)?),
                vec![
                    zeroed(sequences * 16 * 1024 * 4)?,
                    zeroed(sequences * 16 * 1024 * 4)?,
                    zeroed(sequences * 16 * 256 * 4)?,
                    zeroed(sequences * 16 * 256 * 4)?,
                ],
            ),
            128 => (
                Some(zeroed(shape.units * metadata::compressed_page_bytes(128))?),
                None,
                vec![zeroed(sequences * 256 * 512 * 4)?, zeroed(sequences * 256 * 512 * 4)?],
            ),
            _ => (None, None, Vec::new()),
        };
        Ok(LayerCache { main, compressed, index, states })
    }

    /// Allocates the cache pools and RoPE tables for `parts.shape`.
    pub fn new(parts: EngineParts<'a>) -> Result<Self> {
        let _memory_scope = cuteafd_ffi::memory_ledger::scope("kv");
        let quantize_grid = Fp8QuantizeGrid::new(parts.library.sm_count()?, parts.sms)?;
        ensure!(parts.embedding.hidden() == parts.cfg.dim, "embedding rows of {} for dim {}", parts.embedding.hidden(),
            parts.cfg.dim);
        // Layers past the backbone are the dSpark stages' window caches.
        let stages = parts.weights.dspark.as_ref().map_or(0, |d| d.stages.len());
        let attention_reduction = AttentionReductionPlan::new(parts.decode_rows,
            (stages > 0).then_some(parts.cfg.dspark_block_size), None)?;
        let pools = (0..parts.cfg.n_layers + stages).map(|l| Self::pool_layer(&parts, l)).collect::<Result<Vec<_>>>()?;
        let table = |compressed: bool| -> Result<Dev<'a>> {
            let values = metadata::rope_table(&parts.cfg, compressed, parts.max_context.max(metadata::WINDOW));
            let allocation = DeviceAllocation::new(parts.library, values.len() * 4, ROPE)?;
            parts.library.copy_h2d(allocation.buffer, bytes_of(&values))?;
            Ok(allocation.into())
        };
        let split_family = match (parts.weights.layers.iter().any(|l| l.attention_split || l.ffn_split), parts.family) {
            (false, _) => None,
            (true, "dsv4f") => Some("dsv4f2"),
            (true, "dsv4p") => Some("dsv4p2"),
            (true, other) => anyhow::bail!("no head-split programs for {other}"),
        };
        ensure!(parts.context.is_none() || (split_family.is_some() && parts.shape.units % 2 == 0),
            "C4 context attention needs the head-split programs and an even unit pool, not {} units", parts.shape.units);
        let device = parts.library.cuda_get_device()?;
        let expert_usage = parts.expert_usage.map(|(geometry, reader)| cuteafd_ffi::expert_usage::ExpertUsage::new(
            parts.library, geometry, &[cuteafd_ffi::expert_usage::UsageDevice { logical: 0, cuda: device }], reader)).transpose()?;
        Ok(Self {
            expert_usage,
            device,
            split_family,
            peer: None,
            terminal: Cell::new(crate::shared::peer_split::TerminalState::Active),
            exchange: None,
            tp2: None,
            context_mode: parts.context,
            context: None,
            context_units: RefCell::new(Vec::new()),
            exchange_policy: ExchangePolicy::from_env()?,
            capture_only: std::cell::Cell::new(false),
            declared_graphs: RefCell::new(Vec::new()),
            graph_captures: std::cell::Cell::new(0),
            expert_graph_captures: std::cell::Cell::new(0),
            graph_coverage: crate::shared::decode_graph::census::LaunchCoverage::new(),
            draft_probe: RefCell::new(None),
            rope_window: table(false)?,
            rope_compressed: table(true)?,
            pools,
            library: parts.library,
            programs: parts.programs,
            cfg: parts.cfg,
            weights: parts.weights,
            family: parts.family,
            decode_rows: parts.decode_rows,
            prefill_rows: parts.prefill_rows,
            prefill_kernel_rows: parts.prefill_kernel_rows,
            prefill_waves: parts.prefill_waves,
            prefill_span_rows: parts.prefill_span_rows,
            full_prefill_logits: false,
            c128_width: parts.c128_width,
            max_context: parts.max_context,
            stream: parts.stream,
            quantize_grid,
            attention_reduction,
            shape: parts.shape,
            prefill_workspace: RefCell::new(None),
            packed_row_cap: Cell::new(None),
            packed_programs: Default::default(),
            decode_workspace: RefCell::new(None),
            profile: RefCell::new(Profile::default()),
            graphs: RefCell::new(std::collections::HashMap::new()),
            local: RefCell::new(None),
            embedding: parts.embedding,
            skip_routed: parts.skip_routed,
        })
    }

    /// All-row diagnostics reuse the admitted bounded head buffer, downloading
    /// its rows in LOGIT_ROWS batches; no vocabulary-sized GPU buffer is added.
    pub fn prepare_scoring_prefill(&mut self) -> Result<()> {
        self.prepare_workspaces()?;
        self.full_prefill_logits = true;
        Ok(())
    }

    /// Reserve every upload source before any peer wait can be queued. Growing
    /// pinned storage later may implicitly synchronize the device and deadlock.
    fn reserve_upload(&self, buffer: cuteafd_ffi::CuteafdDeviceBuffer) -> Result<()> {
        let stream = if self.library.cuda_get_device()? == self.device { self.stream }
            else { self.peer()?.stream };
        // SAFETY: startup selects the owning device; the persistent buffer uses only this stream.
        unsafe { self.library.reserve_h2d_ordered(buffer, stream) }
    }

    fn reserve_slot_uploads(&self, lane: &UnitSlot<'_>, lead: bool) -> Result<()> {
        let tables = &lane.tables;
        for buffer in [&lane.tokens, &tables.positions, &tables.main_slots,
            &tables.swa_indices, &tables.swa_lengths, &tables.c4_page_table,
            &tables.c4_visible, &tables.c4_indexed_lengths, &tables.c128_indices,
            &tables.c128_lengths].into_iter().chain(&tables.c4).chain(&tables.c128) {
            self.reserve_upload(buffer.buffer)?;
        }
        if lead && !self.embedding.device_gather() { self.reserve_upload(lane.stream_a.buffer)?; }
        Ok(())
    }

    /// Allocate both shapes on every rank before readiness, even with no local experts.
    pub fn prepare_workspaces(&self) -> Result<()> {
        if self.decode_workspace.borrow().is_none() {
            *self.decode_workspace.borrow_mut() = Some(self.workspace(self.decode_rows, 1, false)?);
        }
        if self.prefill_workspace.borrow().is_none() {
            *self.prefill_workspace.borrow_mut() = Some(self.workspace(self.prefill_rows, self.prefill_waves, true)?);
        }
        if let Some(peer) = &self.peer {
            if peer.decode_workspace.borrow().is_none() {
                *peer.decode_workspace.borrow_mut() = Some(self.workspace_on(1, self.decode_rows, 1, false)?);
            }
            if peer.prefill_workspace.borrow().is_none() {
                *peer.prefill_workspace.borrow_mut() = Some(self.workspace_on(1, self.prefill_rows, self.prefill_waves, true)?);
            }
        }
        Ok(())
    }

    pub fn drain_state(&self) -> Result<()> {
        for rank in 0..self.ranks() {
            self.on(rank, || {
                // SAFETY: all hooks completed and every peer wait has its publication queued.
                unsafe { self.library.cuda_stream_synchronize(self.stream_of(rank)) }
            })?;
        }
        Ok(())
    }

    fn workspace(&self, t: usize, lanes: usize, shared_page_list: bool) -> Result<Workspace<'a>> {
        self.workspace_on(0, t, lanes, shared_page_list)
    }

    /// Rank `rank`'s workspace (rank 1 has router replicas but no drafter or
    /// vocabulary head buffers), allocated on that rank's GPU.
    fn workspace_on(&self, rank: usize, t: usize, lanes: usize, shared_page_list: bool) -> Result<Workspace<'a>> {
        self.on(rank, || self.workspace_here(rank, t, lanes, shared_page_list))
    }

    fn workspace_here(&self, rank: usize, t: usize, lanes: usize, shared_page_list: bool) -> Result<Workspace<'a>> {
        let _memory_scope = cuteafd_ffi::memory_ledger::scope("workspace");
        let lead = rank == 0;
        let lead_only = |bytes: usize| if lead { bytes } else { 256 };
        let h = self.cfg.dim;
        let draft = lead && self.weights.dspark.is_some();
        let draft_bytes = |bytes| if draft { bytes } else { 0 };
        // Rank 1 runs only its head-split share; rank 0 keeps every head for
        // the unsplit dSpark stages.
        let heads = if rank == 1 { self.cfg.n_heads / 2 } else { self.cfg.n_heads };
        let (experts, topk) = (self.cfg.n_routed_experts, self.cfg.n_activated_experts);
        let head_workspace = self.alloc(lead_only(cuteafd_ffi::programs::VOCABULARY_HEAD_WORKSPACE), WORKSPACE)?;
        let route = if shared_page_list { format!("prefill_m{}", self.prefill_kernel_rows) }
            else { format!("decode_m{}", self.decode_rows) };
        let spec = self.programs.spec(&format!("{}_index_topk_{route}", self.family))?;
        let topk_scratch = usize::try_from(*spec.scratch.get("scratch").context("V4 top-k scratch geometry")?)?;
        let plane = |bytes: usize| self.alloc(lanes * bytes, WORKSPACE);
        let stream_a = plane(t * 4 * h * 2)?;
        let stream_b = plane(t * 4 * h * 2)?;
        let post = plane(t * 4 * 4)?;
        let comb = plane(t * 16 * 4)?;
        let shared = plane(t * h * 2)?;
        let payload = self.tp2.as_ref().map(|_| plane(t * h * 4)).transpose()?;
        let lane = |index: usize| -> Result<UnitSlot<'a>> {
            let slot = UnitSlot {
                index,
                payload: payload.as_ref().map(|p| p.rows(index, t * h * 4)),
                stream_a: stream_a.rows(index, t * 4 * h * 2),
                stream_b: stream_b.rows(index, t * 4 * h * 2),
                post: post.rows(index, t * 4 * 4),
                comb: comb.rows(index, t * 16 * 4),
                tokens: self.alloc(t * 4, WORKSPACE)?,
                shared: shared.rows(index, t * h * 2),
                tables: self.step_buffers(t, if shared_page_list { cuteafd_loader::serving_capacity::V4_PACKED_MEMBERS } else { t },
                    if shared_page_list { cuteafd_loader::serving_capacity::V4_PACKED_MEMBERS } else { 1 })?,
                taps: self.alloc(draft_bytes(cuteafd_loader::serving_capacity::deepseek_v4_tap_rows(t, shared_page_list)
                    * self.cfg.dspark_target_layer_ids.len() * h * 2), DRAFT_WORKSPACE)?,
            };
            self.reserve_slot_uploads(&slot, lead)?;
            Ok(slot)
        };
        Ok(Workspace {
            slots: (0..lanes).map(lane).collect::<Result<_>>()?,
            y: self.alloc(t * h * 2, WORKSPACE)?,
            query: self.alloc(t * heads * 512 * 2, WORKSPACE)?,
            q_rank: self.alloc(t * self.cfg.q_lora_rank * 2, WORKSPACE)?,
            attn_out: self.alloc(t * heads * 512 * 2, WORKSPACE)?,
            delta: self.alloc(t * h * 2, WORKSPACE)?,
            index_query: self.alloc(t * self.cfg.index_n_heads * self.cfg.index_head_dim, WORKSPACE)?,
            index_weights: self.alloc(t * self.cfg.index_n_heads * 4, WORKSPACE)?,
            selected: self.alloc(t * self.cfg.index_topk * 4, WORKSPACE)?,
            topk_scratch: self.zeroed(topk_scratch, WORKSPACE)?,
            logits: self.alloc(t * experts * 4, WORKSPACE)?,
            route_ids: self.alloc(t * topk * 4, WORKSPACE)?,
            route_weights: self.alloc(t * topk * 4, WORKSPACE)?,
            wire: self.alloc(t * (h + h / 32), WORKSPACE)?,
            scratch: self.alloc(self.scratch_bytes(shared_page_list)?, WORKSPACE)?,
            dummy: self.zeroed(4096, WORKSPACE)?,
            vocab_logits: self.alloc(lead_only(t.min(LOGIT_ROWS) * self.cfg.vocab_size * 4), WORKSPACE)?,
            main_x: self.alloc(draft_bytes(t.min(TAP_ROWS) * h * 2), DRAFT_WORKSPACE)?,
            main_work: self.alloc(draft_bytes(t.min(TAP_ROWS) * h * 4), DRAFT_WORKSPACE)?,
            first_tokens: self.alloc(draft_bytes(t * 4), DRAFT_WORKSPACE)?,
            drafts: self.alloc(draft_bytes(t * 4), DRAFT_WORKSPACE)?,
            markov: self.alloc(if draft { self.library.dsv4_markov_workspace(t)? } else { 0 }, DRAFT_WORKSPACE)?,
            draft_ids: self.alloc(draft_bytes(t * 4), DRAFT_WORKSPACE)?,
            sum: self.alloc(if self.split_family.is_some() { t * h * 2 } else { 256 }, REDUCE_LANDING)?,
            // SAFETY: the workspace buffer lives in the same struct and drops
            // after the head (field order).
            head: if lead {
                Some(unsafe { self.library.vocabulary_head(head_workspace.buffer.ptr, h as u32, t as u32, crate::shared::target_head::TargetHead::configured().native_mode())? })
            } else {
                None
            },
            _head_workspace: head_workspace,
        })
    }

    /// Persistent table buffers for up to `rows` rows.
    fn step_buffers(&self, rows: usize, page_lists: usize, members: usize) -> Result<StepBuffers<'a>> {
        let _memory_scope = cuteafd_ffi::memory_ledger::scope("workspace/step");
        let ints = |count: usize| self.alloc(count * 4, STEP_TABLES);
        // Each member's compressor headers and 16-byte pointer alignment are private.
        let extra = if members > 1 { cuteafd_loader::serving_capacity::V4_PACKED_TABLE_EXTRA_ROWS } else { 0 };
        let storage_rows = rows + extra;
        let metadata = |_: usize| -> Result<Vec<Dev<'a>>> { (0..9).map(|_| ints(storage_rows + 2)).collect() };
        Ok(StepBuffers {
            rows,
            positions: self.alloc(storage_rows * 8, STEP_TABLES)?,
            main_slots: self.alloc(storage_rows * 8, STEP_TABLES)?,
            swa_indices: ints(storage_rows * metadata::WINDOW)?,
            swa_lengths: ints(storage_rows)?,
            c4: metadata(4)?,
            c128: metadata(128)?,
            c4_page_table: {
                // Pool-following tables are KV records, matching placement::solve.
                let _memory_scope = cuteafd_ffi::memory_ledger::scope("kv/records");
                // Prefill retains a private, aligned page list per packed member.
                self.alloc(page_lists * (self.shape.units + if members > 1 { 3 } else { 0 }) * 4, KV_RECORDS)?
            },
            c4_visible: ints(storage_rows)?,
            c4_indexed_lengths: ints(storage_rows)?,
            c128_indices: ints(storage_rows * self.c128_width)?,
            c128_lengths: ints(storage_rows)?,
        })
    }

    /// Stream-ordered slot uploads retain their pinned source until the copy completes.
    fn upload(&self, buffer: cuteafd_ffi::CuteafdDeviceBuffer, bytes: &[u8]) -> Result<()> {
        // SAFETY: callers select the owning device; all workspace consumers use its engine stream.
        let stream = if self.library.cuda_get_device()? == self.device { self.stream }
            else { self.peer()?.stream };
        unsafe { self.library.copy_h2d_ordered(buffer, bytes, stream) }
    }

    fn fill(&self, m: &StepBuffers<'_>, tables: &StepTables) -> Result<()> {
        ensure!(tables.rows <= m.rows, "step of {} rows exceeds its buffers ({})", tables.rows, m.rows);
        let put = |buffer: &Dev<'_>, bytes: &[u8]| -> Result<()> {
            ensure!(bytes.len() <= buffer.buffer.bytes, "step table exceeds its buffer");
            if !bytes.is_empty() {
                self.upload(buffer.buffer, bytes)?;
            }
            Ok(())
        };
        put(&m.positions, bytes_of(&tables.positions))?;
        put(&m.main_slots, bytes_of(&tables.main_slots))?;
        put(&m.swa_indices, bytes_of(&tables.swa_indices))?;
        put(&m.swa_lengths, bytes_of(&tables.swa_lengths))?;
        for (buffers, entries) in [(&m.c4, &tables.c4_tables), (&m.c128, &tables.c128_tables)] {
            for (buffer, (_, values)) in buffers.iter().zip(entries) {
                put(buffer, bytes_of(values))?;
            }
        }
        put(&m.c4_page_table, bytes_of(&tables.c4_page_table))?;
        put(&m.c4_visible, bytes_of(&tables.c4_visible))?;
        put(&m.c4_indexed_lengths, bytes_of(&tables.c4_indexed_lengths))?;
        put(&m.c128_indices, bytes_of(&tables.c128_indices))?;
        put(&m.c128_lengths, bytes_of(&tables.c128_lengths))
    }

    fn sync(&self) -> Result<()> {
        // SAFETY: the engine owns this stream.
        unsafe { self.library.cuda_stream_synchronize(self.stream) }
    }

    fn download(&self, allocation: &Dev<'_>, bytes: usize) -> Result<Vec<u8>> {
        self.sync()?;
        let mut out = vec![0u8; bytes];
        self.library.copy_d2h(&mut out, cuteafd_ffi::CuteafdDeviceBuffer { bytes, ..allocation.buffer })?;
        Ok(out)
    }

    /// Every layer's caches and compressor state (backbone layers, then the dSpark stages'
    /// window caches), for the prefix cache's copies.
    pub fn caches(&self) -> &[LayerCache<'a>] {
        &self.pools
    }

    /// [`Self::caches`] of rank `rank` (1: the head split's copy, backbone layers only).
    pub fn caches_on(&self, rank: usize) -> &[LayerCache<'a>] {
        match (rank, &self.peer) {
            (1, Some(peer)) => &peer.pools,
            _ => &self.pools,
        }
    }

    /// Longest chunk one [`Self::prefill`] call takes.
    pub fn prefill_capacity(&self) -> usize {
        self.prefill_span_rows
    }

    /// Prefills the next chunk of a sequence (rows continue at its length) and
    /// returns FP32 logits of its last `logit_rows` rows [logit_rows, vocab].
    /// `on_layer` receives each layer's stream.
    ///
    /// Consecutive units reuse their slots as they commit, without draining at chunk boundaries.
    #[allow(clippy::too_many_arguments)]
    pub fn prefill(
        &self,
        placement: &mut Placement,
        tokens: &[u32],
        transports: &mut [SparkLink<'_>],
        runtime: &tokio::runtime::Runtime,
        logit_rows: usize,
        on_layer: Option<&mut dyn FnMut(usize, &[u8]) -> Result<()>>,
    ) -> Result<Vec<f32>> {
        match self.prefill_step(placement, tokens, transports, runtime, logit_rows, on_layer, true)? {
            Logits::Host(logits) => Ok(logits),
            Logits::Device(_) => unreachable!("a downloading prefill"),
        }
    }

    /// [`Self::prefill`] leaving the logits on the device (`None` without
    /// logit rows); the rows must lie in one workspace's worth.
    pub fn prefill_device(&self, placement: &mut Placement, tokens: &[u32], transports: &mut [SparkLink<'_>],
        runtime: &tokio::runtime::Runtime, logit_rows: usize) -> Result<Option<DeviceLogits>> {
        match self.prefill_step(placement, tokens, transports, runtime, logit_rows, None, false)? {
            Logits::Device(logits) => Ok(logits),
            Logits::Host(_) => unreachable!("a device prefill"),
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn prefill_step(
        &self,
        placement: &mut Placement,
        tokens: &[u32],
        transports: &mut [SparkLink<'_>],
        runtime: &tokio::runtime::Runtime,
        logit_rows: usize,
        on_layer: Option<&mut dyn FnMut(usize, &[u8]) -> Result<()>>,
        download: bool,
    ) -> Result<Logits> {
        let _coverage_step = self.graph_coverage.step("prefill", 1, tokens.len());
        let start = placement.len;
        let t = tokens.len();
        ensure!(t > 0 && t <= self.prefill_capacity() && start + t <= self.max_context,
            "prefill span of {t} tokens at {start} exceeds the span or context capacity");
        let _ = runtime; // Prefill waves run on the link's transport worker.
        let logits = self.prefill_units(placement, tokens, transports, logit_rows, on_layer, download)?;
        placement.len += t;
        Ok(logits)
    }

    /// One decode row per sequence: appends each token and returns the
    /// logits rows [rows, vocab] in order.
    pub fn decode(
        &self,
        rows: &mut [(&mut Placement, u32)],
        transport: Option<&mut SparkLink<'_>>,
        runtime: &tokio::runtime::Runtime,
    ) -> Result<Vec<f32>> {
        let logits = self.decode_device(rows, transport, runtime)?.to_host(self.library)?;
        crate::shared::spark_waves::healthy()?;
        Ok(logits)
    }

    /// [`Self::decode`] leaving the logits on the device.
    pub fn decode_device(
        &self,
        rows: &mut [(&mut Placement, u32)],
        transport: Option<&mut SparkLink<'_>>,
        runtime: &tokio::runtime::Runtime,
    ) -> Result<DeviceLogits> {
        let _coverage_step = self.graph_coverage.step("decode", rows.len(), rows.len());
        ensure!(!rows.is_empty() && rows.len() <= self.decode_rows, "decode batch of {} rows", rows.len());
        for (placement, _) in rows.iter() {
            ensure!(placement.len > 0 && placement.len < self.max_context, "decode at {} outside the context", placement.len);
        }
        let tokens: Vec<u32> = rows.iter().map(|(_, token)| *token).collect();
        let steps: Vec<(&Placement, usize)> = rows.iter().map(|(p, _)| (&**p, p.len)).collect();
        let mut tables = metadata::decode_step(&steps, &self.shape, self.cfg.index_topk, self.c128_width)?;
        tables.pad_c4_geometry(self.shape.units)?;
        let tables = &tables;
        let step = RowStep { tables, tokens: &tokens };
        let transports = match transport {
            Some(transport) => std::slice::from_mut(transport),
            None => &mut [],
        };
        let logits = self.step(&[step], transports, runtime, tokens.len(), None, false)?;
        for (placement, _) in rows.iter_mut() {
            placement.len += 1;
        }
        logits.device()
    }

    /// Appends each sequence's tokens (one or more) at its length in one
    /// decode-shaped step and returns the logits of every row in order. Each
    /// length advances by its token count; a caller that rejects a suffix
    /// sets the length back (compressor state is addressed by position, so
    /// the rejected rows' writes are simply overwritten later).
    pub fn verify(
        &self,
        sequences: &mut [(&mut Placement, &[u32])],
        transports: &mut [SparkLink<'_>],
        runtime: &tokio::runtime::Runtime,
    ) -> Result<Vec<f32>> {
        let logits = self.verify_device(sequences, transports, runtime)?.to_host(self.library)?;
        crate::shared::spark_waves::healthy()?;
        Ok(logits)
    }

    /// [`Self::verify`] leaving every row's logits on the device.
    pub fn verify_device(
        &self,
        sequences: &mut [(&mut Placement, &[u32])],
        transports: &mut [SparkLink<'_>],
        runtime: &tokio::runtime::Runtime,
    ) -> Result<DeviceLogits> {
        let tokens: Vec<u32> = sequences.iter().flat_map(|(_, tokens)| tokens.iter().copied()).collect();
        let _coverage_step = self.graph_coverage.step("verify", sequences.len(), tokens.len());
        ensure!(!tokens.is_empty() && tokens.len() <= self.decode_rows, "verify step of {} rows", tokens.len());
        for (placement, tokens) in sequences.iter() {
            ensure!(!tokens.is_empty() && placement.len > 0 && placement.len + tokens.len() <= self.max_context,
                "verify of {} rows at {} outside the context", tokens.len(), placement.len);
        }
        let spans: Vec<(&Placement, usize, usize)> = sequences.iter().map(|(p, t)| (&**p, p.len, t.len())).collect();
        let mut tables = metadata::verify_step(&spans, &self.shape, self.cfg.index_topk, self.c128_width)?;
        tables.pad_c4_geometry(self.shape.units)?;
        let tables = &tables;
        let step = RowStep { tables, tokens: &tokens };
        let logits = self.step(&[step], transports, runtime, tokens.len(), None, false)?;
        for (placement, tokens) in sequences.iter_mut() {
            placement.len += tokens.len();
        }
        logits.device()
    }

    /// Runs every layer for `lanes` (one decode lane, or prefill lanes of
    /// consecutive rows of one sequence) and returns the logits of the last
    /// `logit_rows` rows across the lanes.
    fn step(
        &self,
        lanes: &[RowStep<'_>],
        transports: &mut [SparkLink<'_>],
        runtime: &tokio::runtime::Runtime,
        logit_rows: usize,
        on_layer: Option<&mut dyn FnMut(usize, &[u8]) -> Result<()>>,
        download: bool,
    ) -> Result<Logits> {
        let transports = RefCell::new(transports);
        crate::shared::experts::rtx::pair::submit(
            || self.require_live(), || self.shutdown_submission(&mut transports.borrow_mut()),
            || self.step_inner(lanes, &mut transports.borrow_mut(), runtime, logit_rows, on_layer, download))
    }

    fn step_inner(
        &self,
        lanes: &[RowStep<'_>],
        transports: &mut [SparkLink<'_>],
        runtime: &tokio::runtime::Runtime,
        logit_rows: usize,
        on_layer: Option<&mut dyn FnMut(usize, &[u8]) -> Result<()>>,
        download: bool,
    ) -> Result<Logits> {
        self.begin_route_step();
        let h = self.cfg.dim;
        ensure!(lanes.len() == 1 && lanes[0].tables.decode
            && lanes[0].tokens.len() == lanes[0].tables.rows && logit_rows <= lanes[0].tables.rows,
            "single-step requires one decode batch");
        ensure!(on_layer.is_none() && !download, "decode returns device logits");
        if self.decode_workspace.borrow().is_none() {
            *self.decode_workspace.borrow_mut() = Some(self.workspace(self.decode_rows, 1, false)?);
        }
        let workspace = self.decode_workspace.borrow();
        let w = workspace.as_ref().context("decode workspace")?;
        let step = &lanes[0];
        let lane = &w.slots[0];
        let gather = self.embedding.device_gather();
        self.fill(&lane.tables, step.tables)?;
        self.embedding.check(step.tokens)?;
        self.upload(lane.tokens.buffer, bytes_of(step.tokens))?;
        if !gather {
            self.upload(lane.stream_a.buffer, &self.embedding.host_rows_repeated(step.tokens, 4)?)?;
        }
        self.prepare_route_checks(&[step.tables.rows])?;
        let cap = self.decode_rows;
        let rows_of = |_: usize| Scalar::I32(step.tables.rows as i32);
        let peer_workspace = match &self.peer {
            Some(peer) => {
                if peer.decode_workspace.borrow().is_none() {
                    *peer.decode_workspace.borrow_mut() = Some(self.workspace_on(1, self.decode_rows, 1, false)?);
                }
                Some(peer.decode_workspace.borrow())
            }
            None => None,
        };
        let w1 = peer_workspace.as_ref().and_then(|ws| ws.as_ref());
        if let Some(w1) = w1 {
            // Persistent slots are overwritten on their owning streams after
            // prior readers; metadata staging retains each upload's source.
            self.on(1, || {
                self.fill(&w1.slots[0].tables, step.tables)?;
                self.upload(w1.slots[0].tokens.buffer, bytes_of(step.tokens))
            })?;
            if self.context.is_some() {
                *self.context_units.borrow_mut() = vec![[
                    self.put_context_tables(0, &lane.tables, step.tables, 0, false, &[])?,
                    self.put_context_tables(1, &w1.slots[0].tables, step.tables, 0, false, &[])?,
                ]];
            }
        }
        {
            let (tables, lane) = (lanes[0].tables, &w.slots[0]);
            let (t, rows) = (tables.rows, rows_of(0));
            // Decode waves go to the first transport's exchange; its one
            // return slot's planes are baked into the replayed graphs (they
            // never move). A landed wave is consumed after the replay whose
            // post reduced it.
            let mut decode_link = transports.first_mut().filter(|_| !self.skip_routed);
            let decode_reduction = match decode_link.as_deref_mut() {
                Some(link) => {
                    let exchange = link.exchange()?;
                    ensure!(exchange.waves() == 1, "V4 decode bakes one return slot's planes into its graphs");
                    Some(exchange.reduction(0))
                }
                None if self.skip_routed || self.local_layers() == self.weights.layers.len() =>
                    None,
                None => anyhow::bail!("no transport"),
            };
            let mut unread = false;
            let local_layers = self.local_layers();
            let mut ranks = 0usize;
            for (layer, weights) in self.weights.layers.iter().enumerate() {
                let previous = ranks;
                // One segment: the previous layer's FFN reduce + mHC post, then
                // this layer's attention, router and expert input quantization,
                // replayed as a CUDA graph keyed by everything it bakes in.
                let segment = || -> Result<()> {
                    if layer == 0 && gather {
                        self.gather_streams(lane, t)?;
                    }
                    if let (0, Some(w1)) = (layer, w1) {
                        self.record_peer(0, "entry", DIRECT, true, 0, 0);
                        self.exchange()?.push_to(0, DIRECT, lane.stream_a.buffer.ptr, w1.slots[0].stream_a.buffer.ptr,
                            t * 4 * h * 2)?;
                    }
                    if previous > 0 {
                        self.post_layer(w, lane, 0, previous, decode_reduction.clone(), rows, layer - 1)?;
                        self.tap(w, lane, layer - 1, t)?;
                    }
                    if w1.is_some() {
                        self.attention_split(layer, weights, tables, lane, 0, w, rows, cap)?;
                    } else {
                        self.attention(layer, weights, tables, lane, w, rows, cap)?;
                    }
                    Ok(())
                };
                let key = GraphKey {
                    layer,
                    rows: t,
                    chunked: tables.chunked,
                    attention: self.attention_kind(weights.ratio, tables),
                    table_width: tables.c4_table_width,
                    table_stride: tables.c4_table_stride,
                    previous,
                    tp2_broadcast: false, exchange_f32: false,
                };
                if self.mixed_attention(weights.ratio, tables) {
                    self.coverage_legacy(0, key, crate::shared::decode_graph::census::Launch::Eager, "mixed-c128");
                    segment()?;
                } else { self.replay_on(0, key, segment)?; }
                if std::mem::take(&mut unread) {
                    // The replay's post reduced the previous layer's planes.
                    decode_link.as_deref_mut().context("decode transport")?.exchange()?.consumed(0, self.stream)?;
                }
                // Preserve owner routes outside capture before another router reuses them.
                // Every routed layer rings its ids on the device (read after the step's sync).
                self.probe_ring(layer, w, t, false)?;
                if let Some(w1) = w1 {
                    // Rank 1's segments: layer 0 after rank 0's, then each next one before the
                    // host waits in this layer's exchange.
                    if layer == 0 {
                        self.peer_segment(0, tables, w1, t, cap)?;
                    }
                    self.peer_experts(layer, 0, w1, t, true)?;
                    if layer + 1 < self.weights.layers.len() {
                        self.peer_segment(layer + 1, tables, w1, t, cap)?;
                    }
                }
                let _ = runtime;
                ranks = self.decode_experts(layer, t, w, lane, cap, decode_link.as_deref_mut())?;
                unread = layer >= local_layers && !self.skip_routed;
                self.probe_mark(layer)?;
                crate::shared::console::layer_mark(layer);
            }
            // The tail: the last post, its taps and the drafter's KV, then the head.
            let last = self.weights.layers.len() - 1;
            let key = GraphKey { layer: TAIL_SEGMENT, rows: t, chunked: false, attention: "tail", table_width: 0,
                table_stride: 0, previous: ranks, tp2_broadcast: false, exchange_f32: false };
            self.replay(key, || -> Result<()> {
                if ranks > 0 {
                    self.post_layer(w, lane, 0, ranks, decode_reduction.clone(), rows, last)?;
                    self.tap(w, lane, last, t)?;
                }
                self.write_draft_kv(w, lane, t, cap)?;
                self.head_launch(&lane.stream_a, t, t, 0, w)
            })?;
            if unread {
                decode_link.as_deref_mut().context("decode transport")?.exchange()?.consumed(0, self.stream)?;
            }
            self.finish_route_step()?;
            Ok(Logits::Device(Some(self.device_logits(w, t))))
        }
    }

    /// The step's token embeddings, four copies per row, into the lane's streams.
    fn gather_streams(&self, lane: &UnitSlot<'_>, rows: usize) -> Result<()> {
        // SAFETY: the lane's token ids are copied in before the stream reaches
        // the gather; its streams hold [rows, 4, dim] BF16.
        unsafe { self.embedding.gather(lane.tokens.buffer.ptr, std::ptr::null(), rows, 4, std::ptr::null(),
            lane.stream_a.buffer.ptr, None, crate::shared::peer_split::RankDevice { device: self.device, stream: self.stream }) }
    }

    /// The first `rows` rows of the workspace's vocabulary logits.
    fn device_logits(&self, w: &Workspace<'_>, rows: usize) -> DeviceLogits {
        let vocab = self.cfg.vocab_size;
        DeviceLogits { ptr: w.vocab_logits.buffer.ptr, rows, vocab, stride: vocab, stream: self.stream, greedy: None }
    }

    fn mixed_attention(&self, ratio: usize, tables: &StepTables) -> bool {
        tables.decode && ratio == 128 && self.attention_reduction.mixed_indexed_rows(&tables.c128_lengths)
    }

    /// Which sparse attention a layer runs this step.
    fn attention_kind(&self, ratio: usize, tables: &StepTables) -> &'static str {
        match ratio {
            4 if tables.c4_groups > 0 => "c4",
            128 if tables.c128_groups > 0 => "c128",
            _ => "win",
        }
    }

    fn coverage_launch(&self, segment: &'static str, rank: usize, lane: usize, rows: usize,
        launch: crate::shared::decode_graph::census::Launch, reason: &'static str) {
        if !self.graph_coverage.enabled() { return; }
        let mode = if self.context.is_some() { "context" }
            else if self.peer.is_some() { "heads" } else { "single" };
        self.graph_coverage.record(mode, segment, rank, lane, rows, launch, reason);
    }

    // Legacy graphs combine logical segments. Count their actual composition,
    // not only the containing graph.
    fn coverage_legacy(&self, rank: usize, key: GraphKey,
        launch: crate::shared::decode_graph::census::Launch, reason: &'static str) {
        if !self.graph_coverage.enabled() { return; }
        key.coverage_segments(rank, self.skip_routed, self.tp2_layer(key.layer),
            self.draft_ready(), self.cfg.n_layers - 1, &self.cfg.dspark_target_layer_ids,
            |lane, segment| self.coverage_launch(segment, rank, lane, key.rows, launch, reason));
    }

    /// Replay a startup executable, otherwise run the segment eagerly.
    fn replay(&self, key: GraphKey, segment: impl FnOnce() -> Result<()>) -> Result<()> {
        self.replay_on(0, key, segment)
    }

    /// [`Self::replay`] on rank `rank`'s stream (its own graphs).
    fn replay_on(&self, rank: usize, key: GraphKey, segment: impl FnOnce() -> Result<()>) -> Result<()> {
        self.require_live()?;
        let mut key = key;
        if !self.capture_only.get() {
        if let Some(tp2) = &self.tp2 {
            key.tp2_broadcast = tp2.broadcast();
            key.exchange_f32 = self.exchange_policy.is_f32(key.rows, self.decode_rows);
        }
        }
        // Diagnostic copies must never become permanent graph nodes.
        if self.tp2.as_ref().is_some_and(|t| t.checking.get()) && !self.capture_only.get() {
            self.coverage_legacy(rank, key, crate::shared::decode_graph::census::Launch::Eager, "route-check");
            return segment();
        }
        let graphs = match (rank, &self.peer) {
            (1, Some(peer)) => &peer.graphs,
            _ => &self.graphs,
        };
        let stream = self.stream_of(rank);
        if let Some(graph) = graphs.borrow().get(&key) {
            if self.capture_only.get() { return Ok(()); }
            self.coverage_legacy(rank, key, crate::shared::decode_graph::census::Launch::Replay, "hit");
            // SAFETY: the graph's pointers are persistent engine buffers of that rank.
            return self.on(rank, || unsafe { self.library.cuda_graph_launch(graph.0, stream) });
        }
        // Only the startup warm loop may instantiate executables. Local/TP2
        // experts and any undeclared traffic shape remain eager forever.
        if !startup_capture_allowed(self.capture_only.get(),
            self.declared_graphs.borrow().get(rank).is_some_and(|set| set.contains(&key))) {
            self.coverage_legacy(rank, key, crate::shared::decode_graph::census::Launch::Eager, "outside-startup-set");
            return segment();
        }
        // SAFETY: capture records launches on that rank's stream; nothing in the
        // segment synchronizes the host.
        self.on(rank, || unsafe { self.library.cuda_graph_begin_capture(stream) })?;
        let captured = segment();
        let exec = self.on(rank, || unsafe { self.library.cuda_graph_end_capture(stream) })
            .map(|exec| GraphExec(exec, self.library));
        captured?;
        let exec = exec?;
        self.graph_captures.set(self.graph_captures.get() + 1);
        if key.attention.starts_with("tp2-") {
            self.expert_graph_captures.set(self.expert_graph_captures.get() + 1);
        }
        graphs.borrow_mut().insert(key, exec);
        Ok(())
    }

    /// Rank 1's decode segment of `layer`: the previous layer's FFN close, then this
    /// layer's share (see [`Self::peer_front`]), replayed as its own graph.
    fn peer_segment(&self, layer: usize, tables: &StepTables, w1: &Workspace<'_>, t: usize, cap: usize) -> Result<()> {
        let rows = Scalar::I32(t as i32);
        let lane = &w1.slots[0];
        let segment = || -> Result<()> {
            if layer > 0 {
                self.peer_post(layer - 1, lane, 0, w1, rows)?;
            }
            self.peer_front(layer, tables, lane, 0, w1, rows, cap)
        };
        let key = GraphKey { layer, rows: t, chunked: tables.chunked,
            attention: self.attention_kind(self.peer()?.layers[layer].ratio, tables),
            table_width: tables.c4_table_width, table_stride: tables.c4_table_stride, previous: usize::from(layer > 0), tp2_broadcast: false, exchange_f32: false };
        if self.mixed_attention(self.peer()?.layers[layer].ratio, tables) {
            self.coverage_legacy(1, key, crate::shared::decode_graph::census::Launch::Eager, "mixed-c128");
            segment()
        } else { self.replay_on(1, key, segment) }
    }

    /// The layer's routed partials + shared expert, reduced, then mHC post
    /// into stream a.
    fn post(&self, w: &Workspace<'_>, lane: &UnitSlot<'_>, ranks: usize, rows: Scalar,
        layer: usize) -> Result<()> {
        if ranks == SKIPPED_EXPERTS {
            return self.run("mhc_post", &[
                ("x", lane.shared.buffer.ptr), ("residual", lane.stream_b.buffer.ptr), ("prev_post", lane.post.buffer.ptr),
                ("prev_comb", lane.comb.buffer.ptr), ("out", lane.stream_a.buffer.ptr),
            ], &[rows]);
        }
        if ranks == LOCAL_EXPERTS {
            let output = self.local_output(layer, 0)?;
            return self.run("mhc_post", &[
                ("x", output), ("residual", lane.stream_b.buffer.ptr), ("prev_post", lane.post.buffer.ptr),
                ("prev_comb", lane.comb.buffer.ptr), ("out", lane.stream_a.buffer.ptr),
            ], &[rows]);
        }
        self.run("mhc_post", &[
            ("x", w.delta.buffer.ptr), ("residual", lane.stream_b.buffer.ptr), ("prev_post", lane.post.buffer.ptr),
            ("prev_comb", lane.comb.buffer.ptr), ("out", lane.stream_a.buffer.ptr),
        ], &[rows])
    }

    /// A backbone layer's [`Self::post`] for lane `index`: [`Self::post_split`] under a head split.
    #[allow(clippy::too_many_arguments)]
    fn post_layer(&self, w: &Workspace<'_>, lane: &UnitSlot<'_>, index: usize, ranks: usize,
        reduction: Option<crate::shared::spark_waves::ReplyReduction<'_>>, rows: Scalar, layer: usize) -> Result<()> {
        if let Some(reduction) = reduction.filter(|_| !matches!(ranks, LOCAL_EXPERTS | SKIPPED_EXPERTS)) {
            let Scalar::I32(count) = rows else { unreachable!() };
            // SAFETY: the landed decode slot and workspace outlive this graph and every replay.
            unsafe { reduction.launch(self.library, lane.shared.buffer.ptr.cast(), w.delta.buffer.ptr.cast(),
                count as usize, self.stream).with_context(|| format!("layer {layer} expert reduction"))?; }
        }
        if self.peer.is_some() {
            self.post_split(w, lane, index, ranks, rows, layer)
        } else {
            self.post(w, lane, ranks, rows, layer)
        }
    }

    /// [`Self::post`] on rank 0 under a head split (lane `index`): the routed sum
    /// with this GPU's shared-expert half (or the local layer's output, or the
    /// half alone when routed experts are skipped), sent to rank 1 first (not after
    /// the last layer), plus rank 1's half, then mHC post.
    #[allow(clippy::too_many_arguments)]
    fn post_split(&self, w: &Workspace<'_>, lane: &UnitSlot<'_>, index: usize, ranks: usize, rows: Scalar, layer: usize) -> Result<()> {
        let Scalar::I32(count) = rows else { unreachable!() };
        if self.tp2_layer(layer) && ranks == LOCAL_EXPERTS {
            return self.tp2_post(0, layer, index, w, lane, count as usize);
        }
        let base = match ranks {
            SKIPPED_EXPERTS => lane.shared.buffer.ptr,
            LOCAL_EXPERTS => self.local_output(layer, index)?,
            _ => w.delta.buffer.ptr,
        };
        let (exchange, ffn) = (self.exchange()?, slot(layer, true, index));
        if layer + 1 < self.cfg.n_layers {
            self.record_peer(0, "ffn", ffn, true, layer, index);
            exchange.push(0, ffn, base, count as usize * self.cfg.dim * 2)?;
        }
        self.record_peer(0, "ffn", ffn, false, layer, index);
        exchange.wait(0, ffn)?;
        self.add(0, base, exchange.recv(0, ffn)?, w.sum.buffer.ptr, count as usize)?;
        self.run("mhc_post", &[
            ("x", w.sum.buffer.ptr), ("residual", lane.stream_b.buffer.ptr), ("prev_post", lane.post.buffer.ptr),
            ("prev_comb", lane.comb.buffer.ptr), ("out", lane.stream_a.buffer.ptr),
        ], &[rows])
    }

    /// Rank 1's share of `layer` for its lane `index` (`lane`, `w` its own):
    /// (layer 0: rank 0's residual streams in), its heads' attention and the
    /// attention all-reduce, mHC post_pre, then its shared-expert half out to rank 0.
    #[allow(clippy::too_many_arguments)]
    fn peer_front(&self, layer: usize, tables: &StepTables, lane: &UnitSlot<'_>, index: usize, w: &Workspace<'_>,
        rows: Scalar, cap: usize) -> Result<()> {
        let peer = self.peer()?;
        let weights = &peer.layers[layer];
        let exchange = self.exchange()?;
        if layer == 0 {
            let entry = if tables.decode { DIRECT } else { slot(0, false, index) };
            self.record_peer(1, "entry", entry, false, 0, index);
            exchange.wait(1, entry)?;
        }
        self.attention_front(1, layer, weights, tables, lane, w, rows, cap)?;
        self.attention_sum(1, layer, index, w, rows)?;
        self.attention_back(1, layer, tables.decode, weights, lane, w, rows, cap, w.sum.buffer.ptr)?;
        self.run_on(1, weights.ffn_split, &format!("shared_ffn_m{}", self.kernel_rows(cap)), &[
            ("x", w.y.buffer.ptr), ("w13", weights.ptr("w13")?), ("w13_scale", weights.ptr("w13_scale")?),
            ("w2", weights.ptr("w2")?), ("w2_scale", weights.ptr("w2_scale")?), ("out", lane.shared.buffer.ptr),
            ("scratch", w.scratch.buffer.ptr),
        ], &[rows]).with_context(|| format!("layer {layer} shared expert (rank 1)"))?;
        let Scalar::I32(t) = rows else { unreachable!() };
        if self.tp2_layer(layer) {
            if cap != self.decode_rows {
                self.tp2_experts(1, layer, index, w, t as usize)?;
            }
            Ok(())
        } else {
            self.record_peer(1, "ffn", slot(layer, true, index), true, layer, index);
            exchange.push(1, slot(layer, true, index), lane.shared.buffer.ptr, t as usize * self.cfg.dim * 2)
        }
    }

    fn peer_experts(&self, layer: usize, index: usize, w: &Workspace<'_>, t: usize, decode: bool) -> Result<()> {
        if !self.tp2_layer(layer) { return Ok(()); }
        let run = || self.tp2_experts(1, layer, index, w, t);
        if decode { self.replay_on(1, self.tp2_key(layer, t, index), run) } else { run() }
    }

    fn expert_key(layer: usize, rows: usize, index: usize) -> GraphKey {
        GraphKey { layer, rows, chunked: false, attention: "local-experts",
            table_width: 0, table_stride: 0, previous: index, tp2_broadcast: false, exchange_f32: false }
    }

    /// Rank 1's FFN close of `layer` (not after the last layer): rank 0's routed +
    /// shared sum in, plus its own shared half (the same operands as rank 0's sum),
    /// then mHC post into its stream a.
    fn peer_post(&self, layer: usize, lane: &UnitSlot<'_>, index: usize, w: &Workspace<'_>, rows: Scalar) -> Result<()> {
        if layer + 1 >= self.cfg.n_layers {
            return Ok(());
        }
        let Scalar::I32(count) = rows else { unreachable!() };
        if self.tp2_layer(layer) && !self.skip_routed {
            return self.tp2_post(1, layer, index, w, lane, count as usize);
        }
        let (exchange, ffn) = (self.exchange()?, slot(layer, true, index));
        self.record_peer(1, "ffn", ffn, false, layer, index);
        exchange.wait(1, ffn)?;
        self.add(1, exchange.recv(1, ffn)?, lane.shared.buffer.ptr, w.sum.buffer.ptr, count as usize)?;
        self.run_on(1, false, "mhc_post", &[
            ("x", w.sum.buffer.ptr), ("residual", lane.stream_b.buffer.ptr), ("prev_post", lane.post.buffer.ptr),
            ("prev_comb", lane.comb.buffer.ptr), ("out", lane.stream_a.buffer.ptr),
        ], &[rows])
    }

    /// mHC pre, producer, compressor/indexer, sparse MLA, wo, mHC post_pre,
    /// router scores and expert input quantization (stream a -> stream b, y).
    #[allow(clippy::too_many_arguments)]
    fn attention(
        &self,
        layer: usize,
        weights: &LayerWeights<'_>,
        tables: &StepTables,
        lane: &UnitSlot<'_>,
        w: &Workspace<'_>,
        rows: Scalar,
        cap: usize,
    ) -> Result<()> {
        self.attention_front(0, layer, weights, tables, lane, w, rows, cap)?;
        self.attention_back(0, layer, tables.decode, weights, lane, w, rows, cap, w.delta.buffer.ptr)
    }

    /// [`Self::attention`] on rank 0 under a head split: the front, the attention
    /// all-reduce with rank 1 (lane `index`'s slot), then the back on the sum.
    #[allow(clippy::too_many_arguments)]
    fn attention_split(&self, layer: usize, weights: &LayerWeights<'_>, tables: &StepTables, lane: &UnitSlot<'_>,
        index: usize, w: &Workspace<'_>, rows: Scalar, cap: usize) -> Result<()> {
        self.attention_front(0, layer, weights, tables, lane, w, rows, cap)?;
        self.attention_sum(0, layer, index, w, rows)?;
        self.attention_back(0, layer, tables.decode, weights, lane, w, rows, cap, w.sum.buffer.ptr)
    }

    /// Rank `rank`'s attention partial into `delta` pushed to the other GPU, the
    /// other's waited for, and their sum (local first: the same operands on both
    /// GPUs, so the same bits) into `sum`.
    fn attention_sum(&self, rank: usize, layer: usize, lane: usize, w: &Workspace<'_>, rows: Scalar) -> Result<()> {
        let Scalar::I32(t) = rows else { unreachable!() };
        let (exchange, at) = (self.exchange()?, slot(layer, false, lane));
        let bytes = t as usize * self.cfg.dim * 2;
        self.record_peer(rank, "attention", at, true, layer, lane);
        exchange.push(rank, at, w.delta.buffer.ptr, bytes)?;
        self.record_peer(rank, "attention", at, false, layer, lane);
        exchange.wait(rank, at)?;
        self.add(rank, w.delta.buffer.ptr, exchange.recv(rank, at)?, w.sum.buffer.ptr, t as usize)
    }

    /// `out = bf16(a + b)` over `t` rows on rank `rank` (a sum of two partials: the
    /// same bits whichever GPU adds them).
    fn add(&self, rank: usize, a: *mut c_void, b: *mut c_void, out: *mut c_void, t: usize) -> Result<()> {
        self.exchange()?.add(rank, a, b, out, t * self.cfg.dim)
    }

    /// mHC pre, producer, compressor/indexer, sparse MLA and wo into `delta` on
    /// rank `rank` (under a head split: its heads, a partial wo sum).
    #[allow(clippy::too_many_arguments)]
    fn attention_front(
        &self,
        rank: usize,
        layer: usize,
        weights: &LayerWeights<'_>,
        tables: &StepTables,
        lane: &UnitSlot<'_>,
        w: &Workspace<'_>,
        rows: Scalar,
        cap: usize,
    ) -> Result<()> {
        if self.context.as_ref().is_some_and(|c| c.ordinal(layer).is_some()) {
            if tables.decode {
                return self.context_decode_front(rank, layer, weights, tables, lane, w, rows, cap);
            }
            let unit = self.context_units.borrow().get(lane.index).map(|u| u[rank].clone())
                .context("context step tables")?;
            return self.context_prefill_front(rank, layer, weights, tables, lane, lane.index, w, rows, cap, &unit);
        }
        self.attention_produce(rank, layer, weights, lane, w, rows, cap)?;
        let cache = &self.caches_on(rank)[layer];
        let rope = self.attention_rope(rank, weights.ratio);
        let (attention, indexed_cache, indexed_indices, indexed_lengths) =
            self.compress(rank, layer, weights, cache, None, tables, &lane.tables, w, rope, rows, cap)?;
        debug_assert_eq!(attention, self.attention_kind(weights.ratio, tables));
        self.attention_sparse(rank, weights, tables, lane, w, rows, cap,
            cache.main.buffer.ptr, indexed_cache, indexed_indices, indexed_lengths)?;
        self.attention_project(rank, weights, lane, w, rows, cap)
    }

    fn attention_rope(&self, rank: usize, ratio: usize) -> &Dev<'a> {
        match (rank, &self.peer, ratio) {
            (1, Some(peer), 0) => &peer.rope_window,
            (1, Some(peer), _) => &peer.rope_compressed,
            (_, _, 0) => &self.rope_window,
            _ => &self.rope_compressed,
        }
    }

    /// The replicated mHC/latent producer, with only its query heads split.
    #[allow(clippy::too_many_arguments)]
    fn attention_produce(&self, rank: usize, layer: usize, weights: &LayerWeights<'_>,
        lane: &UnitSlot<'_>, w: &Workspace<'_>, rows: Scalar, cap: usize) -> Result<()> {
        let m = &lane.tables;
        let (pools, rope_window, rope_compressed) = match (rank, &self.peer) {
            (1, Some(peer)) => (&peer.pools, &peer.rope_window, &peer.rope_compressed),
            _ => (&self.pools, &self.rope_window, &self.rope_compressed),
        };
        let cache = &pools[layer];
        let ratio = weights.ratio;
        let rope = if ratio == 0 { rope_window } else { rope_compressed };
        let split = weights.attention_split;
        let a = &lane.stream_a;
        self.run_on(rank, false, "mhc_pre", &[
            ("residual", a.buffer.ptr), ("fn", weights.ptr("attn.fn")?), ("scale", weights.ptr("attn.scale")?),
            ("base", weights.ptr("attn.base")?), ("norm", weights.ptr("attn.norm")?), ("post", lane.post.buffer.ptr),
            ("comb", lane.comb.buffer.ptr), ("y", w.y.buffer.ptr), ("scratch", w.scratch.buffer.ptr),
        ], &[rows])?;
        self.run_on(rank, split, &format!("producer_m{}", self.kernel_rows(cap)), &[
            ("hidden", w.y.buffer.ptr), ("positions", m.positions.buffer.ptr), ("main_slots", m.main_slots.buffer.ptr),
            ("cos_sin", rope.buffer.ptr), ("w_qkv", weights.ptr("w_qkv")?), ("w_qkv_scale", weights.ptr("w_qkv_scale")?),
            ("w_q", weights.ptr("w_q")?), ("w_q_scale", weights.ptr("w_q_scale")?), ("q_norm", weights.ptr("q_norm")?),
            ("kv_norm", weights.ptr("kv_norm")?), ("main_kv_cache", cache.main.buffer.ptr), ("query", w.query.buffer.ptr),
            ("q_rank", w.q_rank.buffer.ptr), ("scratch", w.scratch.buffer.ptr),
        ], &[rows])
    }

    /// Heads arithmetic over either the pool or a gathered context view.
    #[allow(clippy::too_many_arguments)]
    fn attention_sparse(&self, rank: usize, weights: &LayerWeights<'_>, tables: &StepTables,
        lane: &UnitSlot<'_>, w: &Workspace<'_>, rows: Scalar, cap: usize, main: *mut c_void,
        indexed_cache: *mut c_void, indexed_indices: *mut c_void, indexed_lengths: *mut c_void) -> Result<()> {
        let m = &lane.tables;
        let mode = if tables.decode { "decode" } else { "prefill" };
        let attention = self.attention_kind(weights.ratio, tables);
        if self.mixed_attention(weights.ratio, tables) {
            // These exports have different SWA chunk folds. Keep each row on its
            // serial route; common all-window/all-compressed graphs are unchanged.
            let heads = self.cfg.n_heads / if weights.attention_split { 2 } else { 1 };
            let at = |dev: &Dev<'_>, start: usize, count: usize, stride: usize| -> Result<*mut c_void> {
                let end = start.checked_add(count).and_then(|n| n.checked_mul(stride))
                    .context("attention row range overflow")?;
                ensure!(end <= dev.buffer.bytes, "attention row range exceeds its buffer");
                Ok(dev.buffer.ptr.cast::<u8>().wrapping_add(start * stride).cast())
            };
            for range in self.attention_reduction.indexed_ranges(&tables.c128_lengths) {
                let (start, count) = (range.start, range.rows);
                let attention = if range.compressed { "c128" } else { "win" };
                self.run_on(rank, weights.attention_split, &format!("sparse_mla_{mode}_{attention}_m{}", self.kernel_rows(cap)), &[
                    ("q", at(&w.query, start, count, heads * 512 * 2)?),
                    ("swa_cache", main),
                    ("swa_indices", at(&m.swa_indices, start, count, metadata::WINDOW * 4)?),
                    ("swa_lengths", at(&m.swa_lengths, start, count, 4)?),
                    ("indexed_cache", indexed_cache),
                    ("indexed_indices", at(&m.c128_indices, start, count, self.c128_width * 4)?),
                    ("indexed_lengths", at(&m.c128_lengths, start, count, 4)?),
                    ("attn_sink", weights.ptr("attn_sink")?),
                    ("out", at(&w.attn_out, start, count, heads * 512 * 2)?),
                    ("scratch", w.scratch.buffer.ptr),
                ], &[Scalar::I32(i32::try_from(count)?)])?;
            }
            return Ok(());
        }
        self.run_on(rank, weights.attention_split, &format!("sparse_mla_{mode}_{attention}_m{}", self.kernel_rows(cap)), &[
            ("q", w.query.buffer.ptr), ("swa_cache", main), ("swa_indices", m.swa_indices.buffer.ptr),
            ("swa_lengths", m.swa_lengths.buffer.ptr), ("indexed_cache", indexed_cache),
            ("indexed_indices", indexed_indices), ("indexed_lengths", indexed_lengths),
            ("attn_sink", weights.ptr("attn_sink")?), ("out", w.attn_out.buffer.ptr), ("scratch", w.scratch.buffer.ptr),
        ], &[rows])
    }

    fn attention_project(&self, rank: usize, weights: &LayerWeights<'_>, lane: &UnitSlot<'_>,
        w: &Workspace<'_>, rows: Scalar, cap: usize) -> Result<()> {
        let m = &lane.tables;
        let rope = self.attention_rope(rank, weights.ratio);
        self.run_on(rank, weights.attention_split, &format!("wo_m{}", self.kernel_rows(cap)), &[
            ("o", w.attn_out.buffer.ptr), ("positions", m.positions.buffer.ptr), ("cos_sin", rope.buffer.ptr),
            ("wo_a", weights.ptr("wo_a")?), ("wo_a_scale", weights.ptr("wo_a_scale")?), ("wo_b", weights.ptr("wo_b")?),
            ("wo_b_scale", weights.ptr("wo_b_scale")?), ("out", w.delta.buffer.ptr), ("scratch", w.scratch.buffer.ptr),
        ], &[rows])
    }

    /// mHC post_pre over the attention output `x` (stream a -> stream b, y), then on
    /// rank 0 the router scores and expert input quantization.
    #[allow(clippy::too_many_arguments)]
    fn attention_back(&self, rank: usize, layer: usize, decode: bool, weights: &LayerWeights<'_>, lane: &UnitSlot<'_>, w: &Workspace<'_>, rows: Scalar,
        cap: usize, x: *mut c_void) -> Result<()> {
        let (a, b) = (&lane.stream_a, &lane.stream_b);
        self.run_on(rank, false, &format!("mhc_post_pre_m{}", self.kernel_rows(cap)), &[
            ("x", x), ("residual", a.buffer.ptr), ("prev_post", lane.post.buffer.ptr),
            ("prev_comb", lane.comb.buffer.ptr), ("fn", weights.ptr("ffn.fn")?), ("scale", weights.ptr("ffn.scale")?),
            ("base", weights.ptr("ffn.base")?), ("norm", weights.ptr("ffn.norm")?), ("residual_out", b.buffer.ptr),
            ("post", lane.post.buffer.ptr), ("comb", lane.comb.buffer.ptr), ("y", w.y.buffer.ptr), ("scratch", w.scratch.buffer.ptr),
        ], &[rows])?;
        if rank != 0 && !self.tp2_layer_weights(weights) {
            return Ok(());
        }
        let Scalar::I32(t) = rows else { unreachable!() };
        self.run_on(rank, false, "router_scores", &[
            ("x", w.y.buffer.ptr), ("w", weights.ptr("gate")?), ("logits", w.logits.buffer.ptr),
        ], &[rows])?;
        let (bias, tid2eid) = if weights.hash {
            (std::ptr::null_mut(), weights.ptr("gate.tid2eid")?)
        } else {
            (weights.ptr("gate.bias")?, std::ptr::null_mut())
        };
        // SAFETY: logits, routing tables, tokens and route outputs are live
        // device buffers sized for this step's rows.
        self.on(rank, || unsafe {
            self.library.dsv4_router_select(w.logits.buffer.ptr, bias, tid2eid, lane.tokens.buffer.ptr,
                w.route_ids.buffer.ptr, w.route_weights.buffer.ptr, t as usize, self.cfg.n_routed_experts,
                self.cfg.n_activated_experts, self.cfg.route_scale as f32, self.stream_of(rank))?;
            if let Some(usage) = &self.expert_usage {
                let phase = if decode { cuteafd_ffi::expert_usage::UsagePhase::Decode }
                    else { cuteafd_ffi::expert_usage::UsagePhase::Prefill };
                usage.routes(self.device, layer, phase)?.record(w.route_ids.buffer.ptr, t as usize, None, u32::MAX,
                    rank == 0, self.stream_of(rank))?;
            }
            Ok(())
        })?;
        self.quantize_input(rank, w, t as usize)
    }

    fn quantize_input(&self, rank: usize, w: &Workspace<'_>, t: usize) -> Result<()> {
        self.quantize_to(rank, w, &w.wire, t)
    }

    fn quantize_to(&self, rank: usize, w: &Workspace<'_>, wire: &Dev<'_>, t: usize) -> Result<()> {
        let h = self.cfg.dim;
        let grid = self.quantize_grid.blocks(t, h);
        self.run_on(rank, false, "expert_input_quant", &[
            ("source_ptr", w.y.buffer.ptr), ("values_ptr", wire.buffer.ptr),
            // SAFETY: scale rows follow the payload inside each wire row.
            ("scale_rows_ptr", unsafe { wire.buffer.ptr.cast::<u8>().add(h) }.cast()),
            ("scale_mma_ptr", w.dummy.buffer.ptr),
        ], &[Scalar::I32(t as i32), Scalar::I32(grid as i32)])
    }

    #[allow(clippy::too_many_arguments)]
    fn compress(
        &self,
        rank: usize,
        layer: usize,
        weights: &LayerWeights<'_>,
        cache: &LayerCache<'_>,
        staged: Option<(*mut c_void, *mut c_void)>,
        tables: &StepTables,
        m: &StepBuffers<'_>,
        w: &Workspace<'_>,
        rope: &Dev<'_>,
        rows: Scalar,
        cap: usize,
    ) -> Result<(&'static str, *mut c_void, *mut c_void, *mut c_void)> {
        let dummy = w.dummy.buffer.ptr;
        let window = ("win", dummy, dummy, dummy);
        let ratio = weights.ratio;
        if ratio == 0 {
            return Ok(window);
        }
        self.compress_records(rank, layer, weights, cache, staged, tables, m, w, rope, rows, cap)?;
        let groups = if ratio == 4 { tables.c4_groups } else { tables.c128_groups };
        let (compressed, index) = match staged {
            Some(views) => views,
            None => (cache.compressed.as_ref().context("compressed cache")?.buffer.ptr,
                cache.index.as_ref().map_or(std::ptr::null_mut(), |i| i.buffer.ptr)),
        };
        if groups == 0 {
            return Ok(window);
        }
        if ratio == 128 {
            return Ok(("c128", compressed, m.c128_indices.buffer.ptr, m.c128_lengths.buffer.ptr));
        }
        self.index_produce(rank, weights, m, w, rope, rows, cap)?;
        let mode = if tables.decode { "decode" } else { "prefill" };
        self.run_on(rank, false, &format!("index_topk_{mode}_m{}", self.kernel_rows(cap)), &[
            ("q_fp8", w.index_query.buffer.ptr), ("weights", w.index_weights.buffer.ptr),
            ("index_k_cache", index),
            ("page_table", m.c4_page_table.buffer.ptr), ("cache_lengths", m.c4_visible.buffer.ptr),
            ("output_indices", w.selected.buffer.ptr), ("scratch", w.topk_scratch.buffer.ptr),
        ], &[rows, Scalar::I32(tables.c4_table_width as i32), Scalar::I32(tables.c4_table_stride as i32)])?;
        Ok(("c4", compressed, w.selected.buffer.ptr, m.c4_indexed_lengths.buffer.ptr))
    }

    /// Updates replicated carry while the supplied slots choose record ownership. `staged`: a
    /// gathered context prefill's C4 record and index views (logical slots) instead of the pools.
    #[allow(clippy::too_many_arguments)]
    fn compress_records(&self, rank: usize, layer: usize, weights: &LayerWeights<'_>, cache: &LayerCache<'_>,
        staged: Option<(*mut c_void, *mut c_void)>, tables: &StepTables, m: &StepBuffers<'_>, w: &Workspace<'_>,
        rope: &Dev<'_>, rows: Scalar, cap: usize) -> Result<()> {
        let ratio = weights.ratio;
        ensure!(matches!(ratio, 4 | 128), "compressor requires C4 or C128");
        ensure!(staged.is_none() || ratio == 4, "only C4 records are staged");
        let (metadata, names) = if ratio == 4 { (&m.c4, &tables.c4_tables) } else { (&m.c128, &tables.c128_tables) };
        let compressed = match staged { Some((records, _)) => records,
            None => cache.compressed.as_ref().context("compressed cache")?.buffer.ptr };
        let mut pointers: Vec<(&str, *mut c_void)> = vec![("hidden", w.y.buffer.ptr)];
        pointers.extend(names.iter().zip(metadata).map(|((name, _), buffer)| (*name, buffer.buffer.ptr)));
        pointers.extend([
            ("cos_sin", rope.buffer.ptr), ("joint_projection", weights.ptr("joint_projection")?),
            ("main_ape", weights.ptr("main_ape")?), ("main_norm", weights.ptr("main_norm")?),
            ("compressed_cache", compressed), ("main_kv_state", cache.states[0].buffer.ptr),
            ("main_score_state", cache.states[1].buffer.ptr),
        ]);
        if ratio == 4 {
            pointers.extend([
                ("index_ape", weights.ptr("index_ape")?), ("index_norm", weights.ptr("index_norm")?),
                ("index_cache", match staged { Some((_, keys)) => keys,
                    None => cache.index.as_ref().context("index cache")?.buffer.ptr }),
                ("index_kv_state", cache.states[2].buffer.ptr), ("index_score_state", cache.states[3].buffer.ptr),
            ]);
        }
        pointers.push(("scratch", w.scratch.buffer.ptr));
        if tables.chunked {
            // Grid bounds: a row completes at most one group, and there are
            // at most as many sequences as rows.
            self.run_on(rank, false, &format!("compressor_continuation_c{ratio}"), &pointers, &[rows, rows, rows])
        } else if tables.decode {
            self.run_on(rank, false, &format!("compressor_decode_c{ratio}"), &pointers, &[rows])
        } else {
            let completed = names[0].1[0].max(1);
            let program = if tables.start == 0 { "prefill" } else { "continuation" };
            self.run_on(rank, false, &format!("compressor_{program}_c{ratio}"), &pointers,
                &[rows, Scalar::I32(completed), Scalar::I32(1)])
        }
        .with_context(|| format!("layer {layer} compressor"))
    }

    #[allow(clippy::too_many_arguments)]
    fn index_produce(&self, rank: usize, weights: &LayerWeights<'_>, m: &StepBuffers<'_>, w: &Workspace<'_>,
        rope: &Dev<'_>, rows: Scalar, cap: usize) -> Result<()> {
        self.run_on(rank, false, &format!("index_producer_m{}", self.kernel_rows(cap)), &[
            ("q_rank", w.q_rank.buffer.ptr), ("hidden", w.y.buffer.ptr), ("positions", m.positions.buffer.ptr),
            ("cos_sin", rope.buffer.ptr), ("w_q", weights.ptr("index_w_q")?), ("w_q_scale", weights.ptr("index_w_q_scale")?),
            ("w_proj", weights.ptr("index_w_proj")?), ("query", w.index_query.buffer.ptr),
            ("head_weights", w.index_weights.buffer.ptr), ("scratch", w.scratch.buffer.ptr),
        ], &[rows])
    }

    /// One Spark request for `rows` wire rows with `topk` routes each.
    fn expert_request(&self, layer: usize, rows: usize, routes: Vec<ExpertProtocolV2RouteEntry>,
        wire: bytes::Bytes, kind: ExpertV2SourceKind) -> Result<ExpertProtocolV2Request> {
        spark_request(self.cfg.dim, self.cfg.n_activated_experts, layer, rows, routes, wire, kind)
    }

    /// Widest session geometry; opening reply planes never runs an expert layer.
    pub fn session_shape(&self, rows: usize) -> Result<cuteafd_transport::SparkSessionShape> {
        let mut shape = cuteafd_transport::SparkSessionShape::new(u32::try_from(rows)?, self.cfg.dim as u32,
            self.cfg.n_activated_experts as u32, ExpertV2Dtype::Fp8E4m3Ue8m0K32)?;
        shape.flags |= EXPERT_PROTOCOL_V2_FLAG_V41_COMPACT_BF16;
        Ok(shape)
    }

    pub(crate) fn local_layers(&self) -> usize {
        self.tp2.as_ref().map_or_else(|| self.local.borrow().as_ref().map_or(0, |l| l.layers().len()), |t| t.layers.last().map_or(0, |id| id + 1))
    }

    pub(crate) fn local_expert_tp(&self) -> usize { if self.tp2.is_some() { 2 } else { 1 } }

    pub fn install_local(&mut self, local: Option<super::local::LocalExperts<'a>>) -> Result<()> {
        ensure!(local.as_ref().is_none_or(|local| local.layers().iter().copied().eq(0..local.layers().len())),
            "V4 local experts currently require a prefix starting at layer 0");
        *self.local.borrow_mut() = local;
        Ok(())
    }

    pub(crate) fn local_group_rows(&self, rows: usize) -> Result<usize> {
        match &self.tp2 {
            Some(tp2) => tp2.group_rows(rows),
            None => self.local.borrow().as_ref().context("local experts")?.group_rows(rows),
        }
    }

    pub(crate) fn arm_draft_probe(&self) -> Result<()> {
        *self.draft_probe.borrow_mut() = Some(crate::shared::draft::probe::RoundProbe::new(
            self.library, vec![false; self.weights.layers.len()], self.cfg.n_activated_experts, self.decode_rows)?);
        Ok(())
    }

    pub(crate) fn probe_begin(&self, rows: usize) {
        if let Some(probe) = self.draft_probe.borrow_mut().as_mut() { probe.begin(rows); }
    }

    pub(crate) fn probe_end(&self, ok: bool) {
        if let Some(probe) = self.draft_probe.borrow_mut().as_mut() {
            probe.end();
            if !ok { let _ = probe.finish(); }
        }
    }

    pub(crate) fn probe_finish<T>(&self, observe: impl FnOnce(&crate::shared::draft::routes::RoundRoutes,
        &[Option<f64>]) -> T) -> Option<T> {
        let mut probe = self.draft_probe.borrow_mut();
        let (routes, layer_us) = probe.as_mut()?.finish()?;
        Some(observe(routes, &layer_us))
    }

    fn probe_mark(&self, layer: usize) -> Result<()> {
        if let Some(probe) = self.draft_probe.borrow_mut().as_mut() {
            // SAFETY: the event is owner-local, outside capture, and read after token selection drains.
            // The next segment closes this layer's TP2 FFN: shifted post/peer work is residual.
            unsafe { probe.mark(layer, self.stream)? };
        }
        Ok(())
    }

    fn probe_ring(&self, layer: usize, w: &Workspace<'_>, rows: usize, staged: bool) -> Result<()> {
        let mut probe = self.draft_probe.borrow_mut();
        let Some(probe) = probe.as_mut().filter(|probe| probe.wants_ring(staged)) else { return Ok(()) };
        // SAFETY: the owner router just completed in stream order; no graph captures this copy.
        unsafe { probe.ring(self.library, layer, w.route_ids.buffer, rows, self.stream) }
    }

    fn local_output(&self, _layer: usize, _index: usize) -> Result<*mut c_void> {
        Ok(self.local.borrow().as_ref().context("local experts")?.output.buffer.ptr)
    }

    /// The shared expert on the unit's FFN input `y` into the lane's `shared`.
    fn shared_ffn(&self, layer: usize, w: &Workspace<'_>, lane: &UnitSlot<'_>, rows: Scalar, cap: usize,
        weights: &LayerWeights<'_>) -> Result<()> {
        // A head split's layers hold half the shared expert (its program family's share).
        self.run_on(0, weights.ffn_split, &format!("shared_ffn_m{}", self.kernel_rows(cap)), &[
            ("x", w.y.buffer.ptr), ("w13", weights.ptr("w13")?), ("w13_scale", weights.ptr("w13_scale")?),
            ("w2", weights.ptr("w2")?), ("w2_scale", weights.ptr("w2_scale")?), ("out", lane.shared.buffer.ptr),
            ("scratch", w.scratch.buffer.ptr),
        ], &[rows]).with_context(|| format!("layer {layer} shared expert"))
    }

    /// Shared and routed experts of a coordinator-resident layer: routes,
    /// wire rows and results stay on the device, the host only enqueues.
    fn local_experts(&self, layer: usize, t: usize, w: &Workspace<'_>, lane: &UnitSlot<'_>, cap: usize, index: usize) -> Result<()> {
        let timer = Instant::now();
        self.count_route_fallback(layer);
        let run = || -> Result<()> {
            self.shared_ffn(layer, w, lane, Scalar::I32(t as i32), cap, &self.weights.layers[layer])?;
            if self.tp2_layer(layer) {
                self.tp2_experts(0, layer, index, w, t)?;
            } else {
                ensure!(self.peer.is_none(), "TP1 routed backbone experts are forbidden under a head split");
                let mut local = self.local.borrow_mut();
                let local = local.as_mut().context("local experts")?;
                // SAFETY: wire, routes and shared rows are complete in stream order.
                unsafe { local.run(super::local::LocalLayer::Backbone(layer), t, w.wire.buffer.ptr,
                    w.route_ids.buffer.ptr, w.route_weights.buffer.ptr, lane.shared.buffer.ptr, self.stream)?; }
            }
            Ok(())
        };
        if cap == self.decode_rows {
            let key = if self.tp2_layer(layer) { self.tp2_key(layer, t, index) } else { Self::expert_key(layer, t, index) };
            self.replay(key, run)?;
        } else { run()?; }
        self.profile.borrow_mut().add(Phase::Experts, timer);
        Ok(())
    }

    /// Decode graphs held per rank, and the captures served so far. Every C4 layer key takes a
    /// geometry class from `metadata::c4_width_classes`, so the inventory is bounded by the
    /// layer, row, lane and class counts rather than by the context length.
    pub fn graph_inventory(&self) -> (usize, usize, usize) {
        (self.graphs.borrow().len(),
            self.peer.as_ref().map_or(0, |peer| peer.graphs.borrow().len()),
            self.graph_captures.get())
    }

    /// Capture declared attention and tail segments without launching unmatched
    /// peer waits or uninitialized expert routes. Every buffer is persistent.
    pub fn warm_segment_graphs(&self, transports: &mut [SparkLink<'_>], admitted_bytes: &[u64]) -> Result<Vec<i64>> {
        use cuteafd_loader::serving_capacity::v4_graphs;
        self.warm_local_graphs()?;
        self.drain_state()?;
        let ratios: Vec<_> = self.weights.layers.iter().map(|weights| weights.ratio).collect();
        let spark_ranks = match transports.first_mut() {
            Some(link) => link.exchange()?.ranks(), None => 0,
        };
        let reductions: Vec<_> = (0..ratios.len()).map(|layer| {
            if self.skip_routed { SKIPPED_EXPERTS }
            else if layer < self.local_layers() { LOCAL_EXPERTS } else { spark_ranks }
        }).collect();
        let keys = v4_graphs::v4_serving_graph_keys(&ratios, self.shape.units, self.decode_rows,
            self.ranks(), &reductions, self.tp2.is_some(), self.exchange_policy.is_f32(1, self.decode_rows)).context("V4 startup graph geometry")?;
        *self.declared_graphs.borrow_mut() = keys.iter().map(|bank|
            cuteafd_loader::placement::inventory::DeclaredGraphSet::new(bank).map_err(anyhow::Error::msg))
            .collect::<Result<Vec<_>>>()?;
        let charged = v4_graphs::v4_startup_graphs(&ratios, self.shape.units, self.decode_rows,
            self.ranks(), self.tp2.is_some()).context("V4 startup graph charge")?;
        let declared_bytes: Vec<_> = charged.ranks.iter().map(|rank| rank.bytes).collect();
        if admitted_bytes != declared_bytes {
            tracing::warn!(?admitted_bytes, ?declared_bytes,
                "V4 startup graph estimate differs from admitted owner; continuing capture without replanning (#1612, #1614)");
        }
        let expected: usize = keys.iter().map(Vec::len).sum();
        let sample = || (0..self.ranks()).map(|rank| self.on(rank, ||
            Ok(self.library.cuda_allocatable_memory_info()?.0))).collect::<Result<Vec<_>>>();
        let before = sample()?;
        let started = Instant::now();
        tracing::warn!(graphs = expected, seed_bytes = v4_graphs::ESTIMATED_EXECUTABLE_SEED,
            "V4 startup graphs use an Estimated executable seed pending V4 node/rate measurements (#1612, #1614)");
        self.capture_only.set(true);
        let result = (|| -> Result<()> {
            let workspace = self.decode_workspace.borrow();
            let w = workspace.as_ref().context("decode workspace")?;
            for (rank, bank) in keys.iter().enumerate() {
                for &key in bank {
                    let placement = Placement::new(0, (0..self.decode_rows.div_ceil(256).max(1) as u32).collect());
                    let mut tables = if key.chunked {
                        metadata::verify_step(&[(&placement, 0, key.rows)], &self.shape, self.cfg.index_topk, self.c128_width)?
                    } else {
                        let rows: Vec<_> = (0..key.rows).map(|_| (&placement, 0)).collect();
                        metadata::decode_step(&rows, &self.shape, self.cfg.index_topk, self.c128_width)?
                    };
                    tables.c4_groups = usize::from(key.attention == "c4");
                    tables.c128_groups = usize::from(key.attention == "c128");
                    tables.c128_lengths.fill(i32::from(key.attention == "c128"));
                    tables.c4_table_width = key.table_width.max(1);
                    tables.c4_table_stride = key.table_stride.max(1);
                    tables.c4_page_table = vec![0; key.rows * tables.c4_table_stride];
                    let peer_work = self.peer.as_ref().map(|peer| peer.decode_workspace.borrow());
                    let own = if rank == 0 { w } else {
                        peer_work.as_ref().and_then(|work| work.as_ref()).context("peer decode workspace")?
                    };
                    let lane = &own.slots[0];
                    self.on(rank, || self.fill(&lane.tables, &tables))?;
                    if self.context.is_some() && key.layer != TAIL_SEGMENT {
                        self.put_context_tables(rank, &lane.tables, &tables, 0, false, &[])?;
                    }
                    let scalar = Scalar::I32(key.rows as i32);
                    let reduction = if rank == 0 { match transports.first_mut() {
                        Some(link) => Some(link.exchange()?.reduction(0)), None => None,
                    }} else { None };
                    self.replay_on(rank, key, || {
                        if key.layer == TAIL_SEGMENT {
                            let last = self.weights.layers.len() - 1;
                            if key.previous > 0 {
                                self.post_layer(w, lane, 0, key.previous, reduction.clone(), scalar, last)?;
                                self.tap(w, lane, last, key.rows)?;
                            }
                            self.write_draft_kv(w, lane, key.rows, self.decode_rows)?;
                            return self.head_launch(&lane.stream_a, key.rows, key.rows, 0, w);
                        }
                        if rank == 1 {
                            if key.layer > 0 { self.peer_post(key.layer - 1, lane, 0, own, scalar)?; }
                            return self.peer_front(key.layer, &tables, lane, 0, own, scalar, self.decode_rows);
                        }
                        if key.layer == 0 {
                            if self.embedding.device_gather() { self.gather_streams(lane, key.rows)?; }
                            if let Some(work) = peer_work.as_ref().and_then(|work| work.as_ref()) {
                                self.record_peer(0, "entry", DIRECT, true, 0, 0);
                                self.exchange()?.push_to(0, DIRECT, lane.stream_a.buffer.ptr,
                                    work.slots[0].stream_a.buffer.ptr, key.rows * 4 * self.cfg.dim * 2)?;
                            }
                        }
                        if key.previous > 0 {
                            self.post_layer(w, lane, 0, key.previous, reduction.clone(), scalar, key.layer - 1)?;
                            self.tap(w, lane, key.layer - 1, key.rows)?;
                        }
                        let weights = &self.weights.layers[key.layer];
                        if self.peer.is_some() {
                            self.attention_split(key.layer, weights, &tables, lane, 0, w, scalar, self.decode_rows)
                        } else { self.attention(key.layer, weights, &tables, lane, w, scalar, self.decode_rows) }
                    })?;
                    // Captures enqueue no executable launches; drain metadata uploads
                    // before reusing the startup step's staging allocation.
                    self.drain_state()?;
                }
            }
            Ok(())
        })();
        let drained = self.drain_state();
        self.capture_only.set(false);
        result?;
        drained?;
        if self.graph_captures.get() != expected {
            tracing::error!(declared = expected, captured = self.graph_captures.get(),
                "V4 startup capture count differs; missing keys remain eager and actual storage is measured");
        }
        for (rank, declared) in self.declared_graphs.borrow().iter().enumerate() {
            tracing::info!(rank, executables = declared.len(),
                declared_graphs = %declared.fingerprint(rank as u8).map_err(anyhow::Error::msg)?,
                "V4 startup graph key coverage");
            let graphs = if rank == 0 { self.graphs.borrow() } else { self.peer()?.graphs.borrow() };
            if let Err(error) = declared.verify(graphs.keys()) {
                let missing: Vec<_> = keys[rank].iter().filter(|key| !graphs.contains_key(key)).collect();
                let extra: Vec<_> = graphs.keys().filter(|key| !declared.contains(key)).collect();
                tracing::error!(rank, declared = declared.len(), captured = graphs.len(), ?missing, ?extra, %error,
                    "V4 startup key coverage differs; missing keys remain eager and actual storage is measured");
            }
        }
        let after = sample()?;
        let bytes: Vec<_> = before.iter().zip(after).map(|(&before, after)| before as i64 - after as i64).collect();
        tracing::info!(graphs = expected, elapsed_ms = started.elapsed().as_millis() as u64,
            graph_bytes = bytes.iter().sum::<i64>(), graph_bytes_by_owner = ?bytes,
            graph_bytes_charged = charged.ranks.iter().map(|rank| rank.bytes).sum::<u64>(),
            "V4 decode graphs captured at startup");
        for (rank, &measured) in bytes.iter().enumerate() {
            if measured > charged.bytes(rank) as i64 {
                tracing::warn!(rank, measured, estimated = charged.bytes(rank),
                    "V4 startup graph capture exceeds executable estimate (#1612, #1614)");
            }
        }
        Ok(bytes)
    }

    /// Expert kernels intentionally remain eager, including route-source checks.
    pub fn warm_local_graphs(&self) -> Result<()> {
        self.prepare_workspaces()?;
        self.warm_tp2_routers()
    }

    /// A prefill unit's wave on the link's transport worker: routes and wire
    /// rows copy into the wave slot's pinned staging on the stream; the worker
    /// builds (row positions start at `first`) and posts the request once they
    /// complete. This thread does not wait for the GPU.
    fn wave_send(&self, layer: usize, t: usize, w: &Workspace<'_>, first: u64,
        members: Vec<(usize, usize, usize, usize)>, transport: &mut crate::shared::spark_waves::Waves<'_>) -> Result<crate::shared::spark_waves::WaveTicket> {
        let (h, topk) = (self.cfg.dim, self.cfg.n_activated_experts);
        let combined_reply = if members.is_empty() { crate::shared::spark_waves::single_reply_eligible(t, ExpertV2SourceKind::Prefill) }
            else { crate::shared::spark_waves::combined_reply_eligible(members.iter().flat_map(|&(_, rows, _, state)|
                (0..rows).map(move |_| (state as u64 + 1, true)))) };
        let layout = crate::shared::experts::staging::RouteStaging::new(t, topk, h + h / 32)?;
        let timer = Instant::now();
        // SAFETY: ids, weights and wire rows hold this unit's `t` rows in
        // stream order; the copies read them before later stream work.
        let ticket = unsafe {
            transport.send(t, transport.compact_payload(t), layout, [w.route_ids.buffer, w.route_weights.buffer, w.wire.buffer], self.stream, combined_reply,
                Box::new(move |routes, wire| {
                    let mut request = spark_request(h, topk, layer, t, routes, wire, ExpertV2SourceKind::Prefill)?;
                    for row in &mut request.rows { row.token_position += first; }
                    for &(first_row, rows, start, state) in &members {
                        for (offset, row) in request.rows[first_row..first_row + rows].iter_mut().enumerate() {
                            row.source_request_id = state as u64 + 1;
                            row.token_position = (start + offset) as u64;
                        }
                    }
                    Ok(request)
                }))?
        };
        self.profile.borrow_mut().add(Phase::Routing, timer);
        Ok(ticket)
    }

    /// The land half of [`Self::wave_send`]: the stream waits for the wave's
    /// planes; returns the rank count the post's reduce needs.
    fn wave_land(&self, transport: &mut crate::shared::spark_waves::Waves<'_>,
        ticket: crate::shared::spark_waves::WaveTicket) -> Result<usize> {
        let timer = Instant::now();
        transport.land(ticket, self.stream)?;
        self.profile.borrow_mut().add(Phase::Experts, timer);
        Ok(transport.ranks())
    }

    /// Bytes of the widest prefill wave's pinned staging (routes and wire rows).
    fn wave_staging_bytes(&self) -> Result<usize> {
        let h = self.cfg.dim;
        Ok(crate::shared::experts::staging::RouteStaging::new(self.prefill_rows, self.cfg.n_activated_experts,
            h + h / 32)?.bytes)
    }

    /// A decode unit's experts; returns the rank count its post needs
    /// ([`LOCAL_EXPERTS`] for a coordinator-resident layer). A Spark layer's
    /// routes and wire rows copy into the exchange's staging on the stream,
    /// the shared expert runs while the worker posts and the Sparks compute,
    /// and the stream then waits for the partials; the next segment's post
    /// reduces them. The host never waits.
    fn decode_experts(
        &self,
        layer: usize,
        t: usize,
        w: &Workspace<'_>,
        lane: &UnitSlot<'_>,
        cap: usize,
        transport: Option<&mut SparkLink<'_>>,
    ) -> Result<usize> {
        if self.skip_routed {
            self.coverage_launch("shared", 0, lane.index, t,
                crate::shared::decode_graph::census::Launch::Eager, "skip-routed");
            self.shared_ffn(layer, w, lane, Scalar::I32(t as i32), cap, &self.weights.layers[layer])?;
            return Ok(SKIPPED_EXPERTS);
        }
        if layer < self.local_layers() {
            self.local_experts(layer, t, w, lane, cap, 0)?;
            return Ok(LOCAL_EXPERTS);
        }
        let exchange = transport.context("no transport")?.exchange()?;
        let (h, topk) = (self.cfg.dim, self.cfg.n_activated_experts);
        let layout = crate::shared::experts::staging::RouteStaging::new(t, topk, h + h / 32)?;
        let timer = Instant::now();
        // SAFETY: ids, weights and wire rows hold this unit's `t` rows in
        // stream order (the segment's router ran before); the copies read
        // them before later stream work rewrites them.
        let ticket = unsafe {
            exchange.send(t, exchange.compact_payload(t), layout, [w.route_ids.buffer, w.route_weights.buffer, w.wire.buffer], self.stream, false,
                Box::new(move |routes, wire| spark_request(h, topk, layer, t, routes, wire, ExpertV2SourceKind::Decode)))?
        };
        self.profile.borrow_mut().add(Phase::Routing, timer);
        // The shared expert runs on the GPU while the Sparks compute.
        self.coverage_launch("shared", 0, lane.index, t,
            crate::shared::decode_graph::census::Launch::Eager, "spark-host");
        self.shared_ffn(layer, w, lane, Scalar::I32(t as i32), cap, &self.weights.layers[layer])?;
        exchange.land(ticket, self.stream)?;
        Ok(exchange.ranks())
    }

    /// Logits of the last `n` of `t` rows into vocabulary logits rows `at..at + n`.
    fn head_launch(&self, stream: &Dev<'_>, t: usize, n: usize, at: usize, w: &Workspace<'_>) -> Result<()> {
        let h = self.cfg.dim;
        let vocab = self.cfg.vocab_size;
        self.run("mhc_head", &[
            ("residual", stream.buffer.ptr), ("fn", self.weights.head_fn.buffer.ptr),
            ("scale", self.weights.head_scale.buffer.ptr), ("base", self.weights.head_base.buffer.ptr),
            ("norm", self.weights.norm.buffer.ptr), ("collapsed", w.delta.buffer.ptr), ("out", w.y.buffer.ptr),
        ], &[Scalar::I32(t as i32)])?;
        // SAFETY: input, weights and logits are live buffers of the head's
        // shape; the last `n` rows start `t - n` rows into `y`.
        unsafe {
            crate::shared::target_head::launch(crate::shared::target_head::TargetHead::configured(),
                w.head.as_ref().context("LM head")?, w.y.buffer.ptr.cast::<u8>().add((t - n) * h * 2).cast(),
                self.weights.head.buffer.ptr.cast_const(), w.vocab_logits.buffer.ptr.cast::<f32>().add(at * vocab), n,
                self.stream)
        }
    }
}

/// One Spark request for `rows` wire rows (FP8, UE8M0 scales per 32) with
/// `topk` routes each; the response is compact BF16 partials.
fn spark_request(dim: usize, topk: usize, layer: usize, rows: usize, routes: Vec<ExpertProtocolV2RouteEntry>,
    wire: bytes::Bytes, kind: ExpertV2SourceKind) -> Result<ExpertProtocolV2Request> {
    let topk = topk as u32;
    let mut request = ExpertProtocolV2Request::new_bytes(
        layer as u64 + 1, 17, layer as u32, dim as u32, ExpertV2Dtype::Fp8E4m3Ue8m0K32,
        (0..rows as u32).map(|row| ExpertProtocolV2RowDescriptor {
            row_id: u64::from(row), source_kind: kind, source_request_id: 1,
            token_position: u64::from(row), route_offset: row * topk, route_count: topk,
        }).collect(),
        routes, wire,
    )?;
    request.header.flags |= EXPERT_PROTOCOL_V2_FLAG_V41_COMPACT_BF16;
    Ok(request)
}

/// A step's logits: downloaded, or left on the device.
enum Logits {
    Host(Vec<f32>),
    Device(Option<DeviceLogits>),
}

impl Logits {
    fn device(self) -> Result<DeviceLogits> {
        match self {
            Logits::Device(Some(logits)) => Ok(logits),
            _ => anyhow::bail!("the step left no device logits"),
        }
    }
}

#[cfg(test)]
mod coverage_tests {
    use super::GraphCoverage;

    #[test]
    fn captures_require_both_startup_phase_and_declared_identity() {
        for warming in [false, true] {
            for declared in [false, true] {
                assert_eq!(super::startup_capture_allowed(warming, declared), warming && declared);
            }
        }
        let banks = cuteafd_loader::serving_capacity::v4_graphs::v4_serving_graph_keys(
            &[0, 4, 128], 48, 16, 2, &[2; 3], true, false).unwrap();
        for bank in banks {
            let declared = cuteafd_loader::placement::inventory::DeclaredGraphSet::new(&bank).unwrap();
            declared.verify(bank.iter()).unwrap();
            assert!(declared.verify(bank.iter().skip(1)).is_err());
            let extra = super::GraphKey { rows: 17, ..bank[0] };
            assert!(!declared.contains(&extra));
            assert!(declared.verify(bank.iter().chain(std::iter::once(&extra))).is_err());
            for attention in ["local-experts", "tp2-replicated", "tp2-broadcast", "draft"] {
                let eager = super::GraphKey { attention, ..bank[0] };
                assert!(!declared.contains(&eager));
                assert!(!super::startup_capture_allowed(true, declared.contains(&eager)));
            }
        }
    }

    #[test]
    fn legacy_coverage_matches_attention_expert_and_tail_composition() {
        let base = super::GraphKey { layer: 3, rows: 33, chunked: true, attention: "c4",
            table_width: 8, table_stride: 8, previous: 4, tp2_broadcast: false, exchange_f32: false };
        let segments = |key: super::GraphKey, rank, skip, tp2, draft| {
            let mut out = Vec::new();
            key.coverage_segments(rank, skip, tp2, draft, 7, &[2, 7],
                |lane, segment| out.push((lane, segment)));
            out
        };
        assert_eq!(segments(base, 0, false, true, true),
            vec![(0, "post"), (0, "tap"), (0, "front"), (0, "back"), (0, "router")]);
        assert_eq!(segments(base, 1, false, true, true),
            vec![(0, "post"), (0, "front"), (0, "back"), (0, "router"), (0, "shared")]);
        assert_eq!(segments(base, 1, true, false, true),
            vec![(0, "post"), (0, "front"), (0, "back"), (0, "shared")]);
        assert_eq!(segments(super::GraphKey { layer: 0, previous: 0, ..base }, 0, false, false, false),
            vec![(0, "front"), (0, "back"), (0, "router")]);
        for attention in ["local-experts", "tp2-replicated", "tp2-broadcast"] {
            let key = super::GraphKey { attention, previous: 1, ..base };
            assert_eq!(segments(key, 0, false, true, true), vec![(1, "shared"), (1, "partial")]);
            assert_eq!(segments(key, 1, false, true, true), vec![(1, "partial")]);
        }
        let tail = super::GraphKey { attention: "tail", ..base };
        assert_eq!(segments(tail, 0, false, true, true),
            vec![(0, "post"), (0, "tap"), (0, "draft-kv"), (0, "head")]);
        assert_eq!(segments(super::GraphKey { previous: 0, ..tail }, 0, true, false, false), vec![(0, "head")]);
    }

}

#[cfg(test)]
mod session_capacity_tests {
    use super::*;

    #[test]
    fn session_covers_widest_real_decode_and_prefill_wave() -> Result<()> {
        for geometry in [cuteafd_core::ExpertGeometry::DEEPSEEK_V4_FLASH,
            cuteafd_core::ExpertGeometry::DEEPSEEK_V4_PRO] {
            for rows in [64, 640, 2048, 4096] {
                let mut session = cuteafd_transport::SparkSessionShape::new(rows, geometry.hidden,
                    geometry.topk, ExpertV2Dtype::Fp8E4m3Ue8m0K32)?;
                session.flags |= EXPERT_PROTOCOL_V2_FLAG_V41_COMPACT_BF16;
                for kind in [ExpertV2SourceKind::Decode, ExpertV2SourceKind::Prefill] {
                    let routes = (0..rows * geometry.topk).map(|i| ExpertProtocolV2RouteEntry {
                        row_index: i / geometry.topk, expert_id: i % geometry.experts, gate_weight: 0.0,
                    }).collect();
                    let real = spark_request(geometry.hidden as usize, geometry.topk as usize, 24,
                        rows as usize, routes, vec![0; rows as usize * (geometry.hidden as usize * 33 / 32)].into(), kind)?;
                    let actual = cuteafd_transport::SparkSessionShape::from_request(&real)?;
                    assert_eq!(actual.request_wire_bytes, session.request_wire_bytes);
                    assert!(actual.response_wire_bytes <= session.response_wire_bytes);
                }
            }
        }
        Ok(())
    }
}
