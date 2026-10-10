//! DeepSeek V4 coordinator: one sequence prefilled from position 0 through the
//! exported programs, routed experts on the Spark ranks.
//!
//! This is the correctness path the serving engine grows from: every layer
//! runs the prototype's op sequence (mHC pre, producer, compressor, index
//! top-k, sparse MLA, wo, mHC post_pre, router, experts + shared FFN, mHC
//! post) and each stage's buffers are sized for the prompt.
use super::metadata::{self, StepTables, INDEX_PAGE_BYTES, MAIN_PAGE_BYTES};
use super::pool::{Placement, PoolShape};
use std::cell::RefCell;
use super::weights::{LayerWeights, ModelWeights};
use crate::shared::launch_grid::Fp8QuantizeGrid;
use crate::shared::memory::{DeviceAllocation, HostAllocation};
use crate::shared::token_io::{DeviceLogits, TokenEmbedding};
use anyhow::{ensure, Context, Result};
use cuteafd_ffi::programs::{Program, Programs, Scalar};
use cuteafd_ffi::NativeLibrary;
use cuteafd_loader::families::deepseek_v4::DeepseekV4Config;
use crate::shared::spark_intake::SparkLink;
use cuteafd_transport::expert::{SparkExpertWave, EXPERT_PROTOCOL_V2_FLAG_V41_COMPACT_BF16};
use cuteafd_transport::{
    ExpertProtocolV2Request, ExpertProtocolV2RouteEntry, ExpertProtocolV2RowDescriptor, ExpertV2Dtype,
    ExpertV2SourceKind,
};
use crate::shared::peer_split::{PeerExchange, RankDevice, DIRECT};
use std::ffi::c_void;
use std::time::Instant;

mod dspark;
mod tp2;
pub(crate) use tp2::ExchangePolicy;
use tp2::Tp2State;
pub(crate) use dspark::DraftRequest;

type Dev<'a> = DeviceAllocation<'a>;

pub(crate) struct Engine<'a> {
    pub library: &'a NativeLibrary,
    pub programs: &'a Programs<'a>,
    pub cfg: DeepseekV4Config,
    pub weights: ModelWeights<'a>,
    pub family: &'static str,
    pub decode_rows: usize,
    pub prefill_rows: usize,
    pub full_prefill_logits: bool,
    pub c128_width: usize,
    /// Longest sequence the exported programs' cache extents cover.
    pub max_context: usize,
    pub stream: *mut c_void,
    quantize_grid: Fp8QuantizeGrid,
    pub shape: PoolShape,
    pools: Vec<LayerCache<'a>>,
    rope_window: Dev<'a>,
    rope_compressed: Dev<'a>,
    /// Prefill and decode workspaces, reused across steps.
    prefill_workspace: RefCell<Option<Workspace<'a>>>,
    decode_workspace: RefCell<Option<Workspace<'a>>>,
    pub profile: RefCell<Profile>,
    graphs: RefCell<std::collections::HashMap<GraphKey, GraphExec<'a>>>,
    /// Expert layers resident on this GPU (they skip the Spark exchange).
    pub local: RefCell<Option<super::local::LocalExperts<'a>>>,
    /// The token embedding table (resident on this GPU or read from its shard).
    pub embedding: TokenEmbedding<'a>,
    /// Benchmarks and token gates only: MoE layers run the shared expert alone.
    pub skip_routed: bool,
    /// The device-driven Spark exchange for decode and verify steps
    /// (`CUTEAFD_SPARK_DEVICE=1`); without it they use the first transport.
    pub device_link: Option<crate::shared::spark_intake::SparkDeviceLink<'a>>,
    /// This engine's GPU (rank 0 of a head split).
    pub device: i32,
    /// Program family of one GPU's share under a head split (`dsv4f2`, `dsv4p2`).
    split_family: Option<&'static str>,
    /// The head split's second GPU and the exchange between the two.
    peer: Option<V4Peer<'a>>,
    exchange: Option<PeerExchange<'a>>,
    tp2: Option<Tp2State<'a>>,
    exchange_policy: ExchangePolicy,
    capture_only: std::cell::Cell<bool>,
    graph_captures: std::cell::Cell<usize>,
    expert_graph_captures: std::cell::Cell<usize>,
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
const LOCAL_EXPERTS: usize = usize::MAX;
/// `exchange` result for a layer whose routed experts were skipped.
const SKIPPED_EXPERTS: usize = usize::MAX - 1;

/// Lanes a long prefill chunk splits into, and the fewest rows per lane worth
/// a second Spark exchange per layer.
pub(crate) const PREFILL_LANES: usize = 2;

const MIN_LANE_ROWS: usize = 256;
/// Vocabulary logits rows a workspace holds (every decode/verify row; a prefill
/// lands at most this many rows at once and downloads longer spans in chunks).
const LOGIT_ROWS: usize = 64;
/// Rows of a step whose target taps feed the drafter's main KV: every row of
/// a decode step, the last window of a prefill lane.
const TAP_ROWS: usize = metadata::WINDOW;

/// One lane's inputs for a step.
struct LaneStep<'s> {
    tables: &'s StepTables,
    tokens: &'s [u32],
}

/// Everything a captured decode segment bakes in.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct GraphKey {
    layer: usize,
    rows: usize,
    chunked: bool,
    attention: &'static str,
    table_width: usize,
    table_stride: usize,
    previous: usize,
    tp2_broadcast: bool,
    exchange_f32: bool,
}

/// The decode step's tail (last post, taps, drafter KV, head) as a graph key layer.
const TAIL_SEGMENT: usize = usize::MAX;

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
    pub library: &'a NativeLibrary,
    pub programs: &'a Programs<'a>,
    pub cfg: DeepseekV4Config,
    pub weights: ModelWeights<'a>,
    pub family: &'static str,
    pub decode_rows: usize,
    pub prefill_rows: usize,
    pub c128_width: usize,
    pub max_context: usize,
    pub stream: *mut c_void,
    pub sms: Option<usize>,
    pub shape: PoolShape,
    pub embedding: TokenEmbedding<'a>,
    pub skip_routed: bool,
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
struct Lane<'a> {
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
    /// [TAP_ROWS, taps * dim] (empty without a drafter).
    taps: Dev<'a>,
}

/// Step workspace sized for `rows` rows per lane; everything but the lanes is
/// consumed within one (layer, lane) unit and shared.
struct Workspace<'a> {
    lanes: Vec<Lane<'a>>,
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
    /// Pinned staging: routes + wire rows down, rank partials up.
    router_host: HostAllocation<'a>,
    /// A head split's sums of the two GPUs' partials ([rows, dim] BF16).
    sum: Dev<'a>,
    // Drops before its workspace below (rank 0 only).
    head: Option<cuteafd_ffi::programs::VocabularyHead<'a>>,
    _head_workspace: Dev<'a>,
}

/// Device copies of one step's tables.
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
    fn alloc(&self, bytes: usize) -> Result<Dev<'a>> {
        DeviceAllocation::new(self.library, bytes.max(256))
    }

    fn zeroed(&self, bytes: usize) -> Result<Dev<'a>> {
        let allocation = self.alloc(bytes)?;
        self.library.cuda_zero_bytes(allocation.buffer, allocation.buffer.bytes)?;
        Ok(allocation)
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

    /// Runs `body` with rank `rank`'s device current (this engine's device again after).
    pub(crate) fn on<T>(&self, rank: usize, body: impl FnOnce() -> Result<T>) -> Result<T> {
        match (rank, &self.peer) {
            (1, Some(peer)) => crate::shared::peer_split::on_device(self.library, peer.device, self.device, body),
            _ => body(),
        }
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
        ensure!(self.split_family.is_some() && layers.len() == self.cfg.n_layers && layers.iter().all(|l| l.split),
            "attach_peer needs the head-split shares of every backbone layer");
        let slot_bytes = if parts.tp2 || self.exchange_policy == ExchangePolicy::F32 {
            self.exchange_policy.slot_bytes(self.prefill_rows, self.decode_rows, self.cfg.dim)
        } else { self.prefill_rows.max(self.decode_rows) * self.cfg.dim * 2 };
        let exchange = PeerExchange::new(self.library, [RankDevice { device: self.device, stream: self.stream },
            RankDevice { device, stream }], 4 * PREFILL_LANES, slot_bytes)?;
        let peer = exchange.on(1, || -> Result<V4Peer<'a>> {
            self.programs.load_matching(|name| self.selected_programs().contains(name))?;
            // The peer's caches and RoPE tables are KV, not exchange storage.
            let _memory_scope = cuteafd_ffi::memory_ledger::scope("kv");
            let table = |compressed: bool| -> Result<Dev<'a>> {
                let values = metadata::rope_table(&self.cfg, compressed, self.max_context.max(metadata::WINDOW));
                let allocation = DeviceAllocation::new(self.library, values.len() * 4)?;
                self.library.copy_h2d(allocation.buffer, bytes_of(&values))?;
                Ok(allocation)
            };
            let pools = (0..self.cfg.n_layers).map(|l| Self::pool_layer_for(self.library, &self.cfg, parts.shape, l))
                .collect::<Result<Vec<_>>>()?;
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

    /// One shared region for this server's programs; launches are stream ordered.
    fn scratch_bytes(&self) -> Result<usize> {
        let mut sizes = Vec::new();
        for name in self.programs.names().filter(|name| self.selected_programs().contains(name)) {
            for value in self.programs.spec(name)?.scratch.values() {
                sizes.push((name, *value));
            }
        }
        Ok(usize::try_from(self.selected_programs().shared_scratch(sizes))?)
    }

    fn pool_layer(parts: &EngineParts<'a>, layer: usize) -> Result<LayerCache<'a>> {
        Self::pool_layer_for(parts.library, &parts.cfg, parts.shape, layer)
    }

    /// Layer `layer`'s zeroed caches and compressor state on the current device.
    fn pool_layer_for(library: &'a NativeLibrary, cfg: &DeepseekV4Config, shape: PoolShape, layer: usize)
        -> Result<LayerCache<'a>> {
        let _memory_scope = cuteafd_ffi::memory_ledger::scope("kv");
        let zeroed = |bytes: usize| -> Result<Dev<'a>> {
            let allocation = DeviceAllocation::new(library, bytes.max(256))?;
            library.cuda_zero_bytes(allocation.buffer, allocation.buffer.bytes)?;
            Ok(allocation)
        };
        let sequences = shape.sequences;
        let main = zeroed(shape.window_pages() * MAIN_PAGE_BYTES)?;
        let (compressed, index, states) = match cfg.compress_ratios.get(layer).copied().unwrap_or(0) {
            4 => (
                Some(zeroed(shape.units * metadata::compressed_page_bytes(4))?),
                Some(zeroed(shape.units * INDEX_PAGE_BYTES)?),
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
        let pools = (0..parts.cfg.n_layers + stages).map(|l| Self::pool_layer(&parts, l)).collect::<Result<Vec<_>>>()?;
        let table = |compressed: bool| -> Result<Dev<'a>> {
            let values = metadata::rope_table(&parts.cfg, compressed, parts.max_context.max(metadata::WINDOW));
            let allocation = DeviceAllocation::new(parts.library, values.len() * 4)?;
            parts.library.copy_h2d(allocation.buffer, bytes_of(&values))?;
            Ok(allocation)
        };
        let split_family = match (parts.weights.layers.first().is_some_and(|l| l.split), parts.family) {
            (false, _) => None,
            (true, "dsv4f") => Some("dsv4f2"),
            (true, "dsv4p") => Some("dsv4p2"),
            (true, other) => anyhow::bail!("no head-split programs for {other}"),
        };
        Ok(Self {
            device: parts.library.cuda_get_device()?,
            split_family,
            peer: None,
            exchange: None,
            tp2: None,
            exchange_policy: ExchangePolicy::from_env()?,
            capture_only: std::cell::Cell::new(false),
            graph_captures: std::cell::Cell::new(0),
            expert_graph_captures: std::cell::Cell::new(0),
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
            full_prefill_logits: false,
            c128_width: parts.c128_width,
            max_context: parts.max_context,
            stream: parts.stream,
            quantize_grid,
            shape: parts.shape,
            prefill_workspace: RefCell::new(None),
            decode_workspace: RefCell::new(None),
            profile: RefCell::new(Profile::default()),
            graphs: RefCell::new(std::collections::HashMap::new()),
            local: RefCell::new(None),
            embedding: parts.embedding,
            skip_routed: parts.skip_routed,
            device_link: None,
        })
    }

    /// All-row diagnostics reuse the admitted bounded head buffer, downloading
    /// its rows in LOGIT_ROWS batches; no vocabulary-sized GPU buffer is added.
    pub fn prepare_scoring_prefill(&mut self) -> Result<()> {
        *self.prefill_workspace.borrow_mut() = Some(self.workspace(self.prefill_rows, PREFILL_LANES)?);
        if let Some(peer) = &self.peer {
            *peer.prefill_workspace.borrow_mut() = Some(self.workspace_on(1, self.prefill_rows, PREFILL_LANES)?);
        }
        self.full_prefill_logits = true;
        Ok(())
    }

    fn workspace(&self, t: usize, lanes: usize) -> Result<Workspace<'a>> {
        self.workspace_on(0, t, lanes)
    }

    /// Rank `rank`'s workspace (rank 1 has router replicas but no drafter or
    /// vocabulary head buffers), allocated on that rank's GPU.
    fn workspace_on(&self, rank: usize, t: usize, lanes: usize) -> Result<Workspace<'a>> {
        self.on(rank, || self.workspace_here(rank, t, lanes))
    }

    fn workspace_here(&self, rank: usize, t: usize, lanes: usize) -> Result<Workspace<'a>> {
        let _memory_scope = cuteafd_ffi::memory_ledger::scope("workspace");
        let lead = rank == 0;
        let lead_only = |bytes: usize| if lead { bytes } else { 256 };
        let h = self.cfg.dim;
        // Rank 1 runs only its head-split share; rank 0 keeps every head for
        // the unsplit dSpark stages.
        let heads = if rank == 1 { self.cfg.n_heads / 2 } else { self.cfg.n_heads };
        let (experts, topk) = (self.cfg.n_routed_experts, self.cfg.n_activated_experts);
        let head_workspace = self.alloc(lead_only(cuteafd_ffi::programs::VOCABULARY_HEAD_WORKSPACE))?;
        let mut topk_scratch = 0usize;
        for route in [format!("decode_m{}", self.decode_rows), format!("prefill_m{}", self.prefill_rows)] {
            let spec = self.programs.spec(&format!("{}_index_topk_{route}", self.family))?;
            topk_scratch = topk_scratch.max(spec.scratch.get("scratch").copied().unwrap_or(0) as usize);
        }
        let lane = || -> Result<Lane<'a>> {
            Ok(Lane {
                payload: if self.split_family.is_some() { Some(self.alloc(t * h * 4)?) } else { None },
                stream_a: self.alloc(t * 4 * h * 2)?,
                stream_b: self.alloc(t * 4 * h * 2)?,
                post: self.alloc(t * 4 * 4)?,
                comb: self.alloc(t * 16 * 4)?,
                tokens: self.alloc(t * 4)?,
                shared: self.alloc(t * h * 2)?,
                tables: self.step_buffers(t)?,
                taps: self.alloc(t.min(TAP_ROWS) * self.cfg.dspark_target_layer_ids.len() * h * 2)?,
            })
        };
        Ok(Workspace {
            lanes: (0..lanes).map(|_| lane()).collect::<Result<_>>()?,
            y: self.alloc(t * h * 2)?,
            query: self.alloc(t * heads * 512 * 2)?,
            q_rank: self.alloc(t * self.cfg.q_lora_rank * 2)?,
            attn_out: self.alloc(t * heads * 512 * 2)?,
            delta: self.alloc(t * h * 2)?,
            index_query: self.alloc(t * self.cfg.index_n_heads * self.cfg.index_head_dim)?,
            index_weights: self.alloc(t * self.cfg.index_n_heads * 4)?,
            selected: self.alloc(t * self.cfg.index_topk * 4)?,
            topk_scratch: self.zeroed(topk_scratch)?,
            logits: self.alloc(t * experts * 4)?,
            route_ids: self.alloc(t * topk * 4)?,
            route_weights: self.alloc(t * topk * 4)?,
            wire: self.alloc(t * (h + h / 32))?,
            scratch: self.alloc(self.scratch_bytes()?)?,
            dummy: self.zeroed(4096)?,
            vocab_logits: self.alloc(lead_only(t.min(LOGIT_ROWS) * self.cfg.vocab_size * 4))?,
            main_x: self.alloc(t.min(TAP_ROWS) * h * 2)?,
            main_work: self.alloc(t.min(TAP_ROWS) * h * 4)?,
            first_tokens: self.alloc(t * 4)?,
            drafts: self.alloc(t * 4)?,
            markov: self.alloc(self.library.dsv4_markov_workspace(t)?)?,
            draft_ids: self.alloc(t * 4)?,
            router_host: HostAllocation::new(self.library, lead_only(t * (topk * 8 + h + h / 32)))?,
            sum: self.alloc(if self.split_family.is_some() { t * h * 2 } else { 256 })?,
            // SAFETY: the workspace buffer lives in the same struct and drops
            // after the head (field order).
            head: if lead {
                Some(unsafe { self.library.vocabulary_head(head_workspace.buffer.ptr, h as u32, t as u32)? })
            } else {
                None
            },
            _head_workspace: head_workspace,
        })
    }

    /// Persistent table buffers for up to `rows` rows.
    fn step_buffers(&self, rows: usize) -> Result<StepBuffers<'a>> {
        let _memory_scope = cuteafd_ffi::memory_ledger::scope("workspace/step");
        let ints = |count: usize| self.alloc(count * 4);
        let metadata = |_: usize| -> Result<Vec<Dev<'a>>> { (0..9).map(|_| ints(rows + 2)).collect() };
        Ok(StepBuffers {
            rows,
            positions: self.alloc(rows * 8)?,
            main_slots: self.alloc(rows * 8)?,
            swa_indices: ints(rows * metadata::WINDOW)?,
            swa_lengths: ints(rows)?,
            c4: metadata(4)?,
            c128: metadata(128)?,
            c4_page_table: ints(rows.max(1) * self.shape.units)?,
            c4_visible: ints(rows)?,
            c4_indexed_lengths: ints(rows)?,
            c128_indices: ints(rows * self.c128_width)?,
            c128_lengths: ints(rows)?,
        })
    }

    fn fill(&self, m: &StepBuffers<'_>, tables: &StepTables) -> Result<()> {
        ensure!(tables.rows <= m.rows, "step of {} rows exceeds its buffers ({})", tables.rows, m.rows);
        let put = |buffer: &Dev<'_>, bytes: &[u8]| -> Result<()> {
            ensure!(bytes.len() <= buffer.buffer.bytes, "step table exceeds its buffer");
            if !bytes.is_empty() {
                self.library.copy_h2d(cuteafd_ffi::CuteafdDeviceBuffer { bytes: bytes.len(), ..buffer.buffer }, bytes)?;
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
        PREFILL_LANES * self.prefill_rows
    }

    /// Prefills the next chunk of a sequence (rows continue at its length) and
    /// returns FP32 logits of its last `logit_rows` rows [logit_rows, vocab].
    /// `on_layer` receives each layer's stream.
    ///
    /// A long chunk without `on_layer` runs as [`PREFILL_LANES`] lanes of
    /// consecutive rows, so one lane's attention overlaps the other's experts.
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
        let start = placement.len;
        let t = tokens.len();
        let lanes = if on_layer.is_none() && t >= PREFILL_LANES * MIN_LANE_ROWS { PREFILL_LANES } else { 1 };
        let per_lane = t.div_ceil(lanes);
        ensure!(t > 0 && per_lane <= self.prefill_rows && start + t <= self.max_context,
            "prefill chunk of {t} tokens at {start} exceeds {} rows per lane or the {}-token context",
            self.prefill_rows, self.max_context);
        let tables = (0..lanes).map(|lane| {
            let (first, end) = (lane * per_lane, ((lane + 1) * per_lane).min(t));
            metadata::prefill_step(placement, &self.shape, start + first, end - first, self.cfg.index_topk, self.c128_width)
        }).collect::<Result<Vec<_>>>()?;
        let steps: Vec<LaneStep<'_>> = tables.iter().enumerate().map(|(lane, tables)| {
            let (first, end) = (lane * per_lane, ((lane + 1) * per_lane).min(t));
            LaneStep { tables, tokens: &tokens[first..end] }
        }).collect();
        let logits = self.step(&steps, transports, runtime, logit_rows, on_layer, download)?;
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
        self.check_device()?;
        Ok(logits)
    }

    /// [`Self::decode`] leaving the logits on the device.
    pub fn decode_device(
        &self,
        rows: &mut [(&mut Placement, u32)],
        transport: Option<&mut SparkLink<'_>>,
        runtime: &tokio::runtime::Runtime,
    ) -> Result<DeviceLogits> {
        ensure!(!rows.is_empty() && rows.len() <= self.decode_rows, "decode batch of {} rows", rows.len());
        for (placement, _) in rows.iter() {
            ensure!(placement.len > 0 && placement.len < self.max_context, "decode at {} outside the context", placement.len);
        }
        let tokens: Vec<u32> = rows.iter().map(|(_, token)| *token).collect();
        let steps: Vec<(&Placement, usize)> = rows.iter().map(|(p, _)| (&**p, p.len)).collect();
        let tables = metadata::decode_step(&steps, &self.shape, self.cfg.index_topk, self.c128_width)?;
        let step = LaneStep { tables: &tables, tokens: &tokens };
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
        self.check_device()?;
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
        ensure!(!tokens.is_empty() && tokens.len() <= self.decode_rows, "verify step of {} rows", tokens.len());
        for (placement, tokens) in sequences.iter() {
            ensure!(!tokens.is_empty() && placement.len > 0 && placement.len + tokens.len() <= self.max_context,
                "verify of {} rows at {} outside the context", tokens.len(), placement.len);
        }
        let spans: Vec<(&Placement, usize, usize)> = sequences.iter().map(|(p, t)| (&**p, p.len, t.len())).collect();
        let tables = metadata::verify_step(&spans, &self.shape, self.cfg.index_topk, self.c128_width)?;
        let step = LaneStep { tables: &tables, tokens: &tokens };
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
        lanes: &[LaneStep<'_>],
        transports: &mut [SparkLink<'_>],
        runtime: &tokio::runtime::Runtime,
        logit_rows: usize,
        mut on_layer: Option<&mut dyn FnMut(usize, &[u8]) -> Result<()>>,
        download: bool,
    ) -> Result<Logits> {
        self.begin_route_step();
        let h = self.cfg.dim;
        let decode = lanes[0].tables.decode;
        ensure!(!lanes.is_empty(), "a step needs a lane");
        let total: usize = lanes.iter().map(|l| l.tables.rows).sum();
        ensure!(logit_rows <= total && (on_layer.is_none() || lanes.len() == 1)
            && lanes.iter().all(|l| l.tokens.len() == l.tables.rows)
            && (!decode || lanes.len() == 1), "step lanes disagree");
        let slot = if decode { &self.decode_workspace } else { &self.prefill_workspace };
        if slot.borrow().is_none() {
            let (rows, count) = if decode { (self.decode_rows, 1) } else { (self.prefill_rows, PREFILL_LANES) };
            *slot.borrow_mut() = Some(self.workspace(rows, count)?);
        }
        let workspace = slot.borrow();
        let w = workspace.as_ref().context("workspace")?;
        ensure!(lanes.len() <= w.lanes.len(), "{} lanes exceed the workspace", lanes.len());
        // Tables are copied in only after the previous step's last read (the
        // head download synchronized the stream).
        // The decode graph's first segment gathers the streams itself.
        let gather = self.embedding.device_gather();
        for (step, lane) in lanes.iter().zip(&w.lanes) {
            self.fill(&lane.tables, step.tables)?;
            self.embedding.check(step.tokens)?;
            // Token ids feed hash routing and the embedding gather.
            let token_bytes: Vec<u8> = step.tokens.iter().flat_map(|token| token.to_le_bytes()).collect();
            self.library.copy_h2d(cuteafd_ffi::CuteafdDeviceBuffer { bytes: token_bytes.len(), ..lane.tokens.buffer }, &token_bytes)?;
            if !gather {
                // The mHC streams start as four copies of the embedding.
                let rows = self.embedding.host_rows_repeated(step.tokens, 4)?;
                self.library.copy_h2d(cuteafd_ffi::CuteafdDeviceBuffer { bytes: rows.len(), ..lane.stream_a.buffer }, &rows)?;
            } else if !decode {
                self.gather_streams(lane, step.tables.rows)?;
            }
        }
        self.prepare_route_checks(&lanes.iter().map(|l| l.tables.rows).collect::<Vec<_>>())?;
        let cap = if decode { self.decode_rows } else { self.prefill_rows };
        let rows_of = |lane: usize| Scalar::I32(lanes[lane].tables.rows as i32);
        // The head split's second GPU: its workspace of the same shape, the same tables,
        // and each lane's residual streams pushed over (the first decode segment pushes them).
        let peer_workspace = match &self.peer {
            Some(peer) => {
                let cell = if decode { &peer.decode_workspace } else { &peer.prefill_workspace };
                if cell.borrow().is_none() {
                    let (rows, count) = if decode { (self.decode_rows, 1) } else { (self.prefill_rows, PREFILL_LANES) };
                    *cell.borrow_mut() = Some(self.workspace_on(1, rows, count)?);
                }
                Some(cell.borrow())
            }
            None => None,
        };
        let w1 = match &peer_workspace {
            Some(ws) => Some(ws.as_ref().context("peer workspace")?),
            None => None,
        };
        if let Some(w1) = w1 {
            // SAFETY: the engine owns the peer stream; every wait on it was matched by a
            // push rank 0 queued, so it drains before its tables are rewritten.
            unsafe { self.library.cuda_stream_synchronize(self.stream_of(1))? };
            for (step, lane1) in lanes.iter().zip(&w1.lanes) {
                self.on(1, || {
                    self.fill(&lane1.tables, step.tables)?;
                    self.library.copy_h2d(cuteafd_ffi::CuteafdDeviceBuffer {
                        bytes: step.tokens.len() * 4, ..lane1.tokens.buffer
                    }, bytes_of(step.tokens))
                })?;
            }
            if !decode {
                for (index, (step, lane)) in lanes.iter().zip(&w.lanes).enumerate() {
                    self.record_peer(0, "entry", DIRECT, true, 0, index);
                    self.exchange()?.push_to(0, DIRECT, lane.stream_a.buffer.ptr, w1.lanes[index].stream_a.buffer.ptr,
                        step.tables.rows * 4 * h * 2)?;
                }
                for (index, step) in lanes.iter().enumerate() {
                    self.peer_front(0, step.tables, &w1.lanes[index], index, w1, rows_of(index), cap)?;
                }
            }
        }
        // Rank 1 after rank 0 queued unit (layer, lane)'s FFN: that unit's FFN close and the
        // lane's next attention (rank 1 runs a unit ahead of the host's rank-0 work).
        let peer_next = |(layer, lane): (usize, usize)| -> Result<()> {
            let Some(w1) = w1 else { return Ok(()) };
            self.peer_post(layer, &w1.lanes[lane], lane, w1, rows_of(lane))?;
            if layer + 1 < self.weights.layers.len() {
                self.peer_front(layer + 1, lanes[lane].tables, &w1.lanes[lane], lane, w1, rows_of(lane), cap)?;
            }
            Ok(())
        };
        if decode {
            let (tables, lane) = (lanes[0].tables, &w.lanes[0]);
            let (t, rows) = (tables.rows, rows_of(0));
            // Decode waves go to the first transport; its planes are baked into
            // the replayed graphs (they never move).
            let device = self.device_link.as_ref().filter(|_| !self.skip_routed);
            if let Some(link) = device {
                link.arm()?;
            }
            let decode_planes = match (device, transports.first()) {
                (Some(link), _) => link.pointers(),
                (None, Some(transport)) => transport.intake.pointers(),
                (None, None) if self.skip_routed || self.local_layers() == self.weights.layers.len() =>
                    [std::ptr::null(); 6],
                (None, None) => anyhow::bail!("no transport"),
            };
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
                        self.exchange()?.push_to(0, DIRECT, lane.stream_a.buffer.ptr, w1.lanes[0].stream_a.buffer.ptr,
                            t * 4 * h * 2)?;
                    }
                    if previous > 0 {
                        self.post_layer(w, lane, 0, previous, decode_planes, rows, layer - 1)?;
                        self.tap(w, lane, layer - 1, t)?;
                    }
                    if w1.is_some() {
                        self.attention_split(layer, weights, tables, lane, 0, w, rows, cap)?;
                    } else {
                        self.attention(layer, weights, tables, lane, w, rows, cap)?;
                    }
                    // Device exchange: the layer's Spark wave, the shared expert while the
                    // Sparks compute, and the wait for its partials join the segment.
                    match device {
                        Some(link) if layer >= local_layers => self.device_experts(link, layer, t, w, lane, cap),
                        _ => Ok(()),
                    }
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
                if let Some(link) = device.filter(|_| layer >= local_layers) {
                    // Announced before the replay can publish it: the proxy spins for it.
                    link.expect(1);
                }
                self.replay_on(0, key, segment)?;
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
                ranks = match device {
                    Some(link) if layer >= local_layers => link.world_size(),
                    _ => self.decode_experts(layer, t, w, lane, cap, transports.first_mut(), runtime)?,
                };
                crate::shared::console::layer_mark(layer);
            }
            // The tail: the last post, its taps and the drafter's KV, then the head.
            let last = self.weights.layers.len() - 1;
            let key = GraphKey { layer: TAIL_SEGMENT, rows: t, chunked: false, attention: "tail", table_width: 0,
                table_stride: 0, previous: ranks, tp2_broadcast: false, exchange_f32: false };
            self.replay(key, || -> Result<()> {
                if ranks > 0 {
                    self.post_layer(w, lane, 0, ranks, decode_planes, rows, last)?;
                    self.tap(w, lane, last, t)?;
                }
                self.write_draft_kv(w, lane, t, cap)?;
                self.head_launch(&lane.stream_a, t, t, 0, w)
            })?;
            self.finish_route_step()?;
            return Ok(Logits::Device(Some(self.device_logits(w, t))));
        } else {
            // Units run layer-major. A unit's attention needs its own lane's
            // previous layer posted and the previous lane's same layer (KV and
            // compressor state) done. With a transport per lane, each Spark
            // wave stays in flight while the next unit's attention runs and is
            // received after the next wave is dispatched, so the Sparks always
            // hold the next request when they finish one.
            let units: Vec<(usize, usize)> = (0..self.weights.layers.len())
                .flat_map(|layer| (0..lanes.len()).map(move |lane| (layer, lane))).collect();
            let attention = |(layer, lane): (usize, usize)| {
                let weights = &self.weights.layers[layer];
                if w1.is_some() {
                    self.attention_split(layer, weights, lanes[lane].tables, &w.lanes[lane], lane, w, rows_of(lane), cap)
                } else {
                    self.attention(layer, weights, lanes[lane].tables, &w.lanes[lane], w, rows_of(lane), cap)
                }
            };
            let local_layers = self.local_layers();
            let pipelined = transports.len() >= lanes.len() && lanes.len() > 1;
            let planes: Vec<[*const u16; 6]> = transports.iter().map(|t| t.intake.pointers()).collect();
            let post = |(layer, lane): (usize, usize), ranks: usize| {
                let slot = if pipelined { lane } else { 0 };
                let plane = planes.get(slot).copied().unwrap_or([std::ptr::null(); 6]);
                self.post_layer(w, &w.lanes[lane], lane, ranks, plane, rows_of(lane), layer)?;
                self.tap(w, &w.lanes[lane], layer, lanes[lane].tables.rows)
            };
            runtime.block_on(async {
                let mut inflight: Option<((usize, usize), SparkExpertWave)> = None;
                attention(units[0])?;
                for (index, &unit) in units.iter().enumerate() {
                    let (layer, lane) = unit;
                    let t = lanes[lane].tables.rows;
                    if self.skip_routed {
                        self.shared_ffn(layer, w, &w.lanes[lane], rows_of(lane), cap, &self.weights.layers[layer])?;
                        peer_next(unit)?;
                        post(unit, SKIPPED_EXPERTS)?;
                    } else if layer < local_layers {
                        self.local_experts(layer, t, w, &w.lanes[lane], cap, lane)?;
                        peer_next(unit)?;
                        post(unit, LOCAL_EXPERTS)?;
                    } else {
                        let slot = if pipelined { lane } else { 0 };
                        let request = self.stage_request(layer, t, w, ExpertV2SourceKind::Prefill,
                            &mut transports[slot])?;
                        let wave = transports[slot].dispatch(&request)?;
                        // The shared expert runs on the GPU while the Sparks compute.
                        self.shared_ffn(layer, w, &w.lanes[lane], rows_of(lane), cap, &self.weights.layers[layer])?;
                        peer_next(unit)?;
                        if let Some((previous, wave)) = inflight.take() {
                            let slot = if pipelined { previous.1 } else { 0 };
                            let ranks = self.land(&mut transports[slot], wave, lanes[previous.1].tables.rows, w).await?;
                            post(previous, ranks)?;
                        }
                        inflight = Some((unit, wave));
                    }
                    let next = units.get(index + 1).copied();
                    // The next unit's input is this unit's post when both are
                    // on one lane, or without a transport per lane.
                    if !pipelined || next.is_none_or(|(_, next_lane)| next_lane == lane) {
                        if let Some((current, wave)) = inflight.take() {
                            let slot = if pipelined { current.1 } else { 0 };
                            let ranks = self.land(&mut transports[slot], wave, lanes[current.1].tables.rows, w).await?;
                            post(current, ranks)?;
                        }
                    }
                    if let Some(on_layer) = on_layer.as_mut() {
                        on_layer(layer, &self.download(&w.lanes[lane].stream_a, t * 4 * h * 2)?)?;
                    }
                    if let Some(next) = next {
                        attention(next)?;
                    }
                }
                anyhow::Ok(())
            })?;
        }
        for (index, step) in lanes.iter().enumerate() {
            self.write_draft_kv(w, &w.lanes[index], step.tables.rows, cap)?;
        }
        if logit_rows == 0 {
            // The next step rewrites the tables only after this one drains.
            self.sync()?;
            self.finish_route_step()?;
            return Ok(if download { Logits::Host(Vec::new()) } else { Logits::Device(None) });
        }
        let mut logits = Vec::with_capacity(if download { logit_rows * self.cfg.vocab_size } else { 0 });
        let (mut first, mut landed) = (0, 0);
        for (step, lane) in lanes.iter().zip(&w.lanes) {
            let rows = step.tables.rows;
            // Rows of this lane inside the last `logit_rows` of the step.
            let wanted = (first + rows).saturating_sub((total - logit_rows).max(first));
            if wanted > 0 && download {
                // The workspace holds `LOGIT_ROWS` rows: longer spans (golden NLL) land in chunks.
                let mut done = 0;
                while done < wanted {
                    let chunk = (wanted - done).min(LOGIT_ROWS);
                    // Rows `rows - wanted + done ..+ chunk` are the last `chunk` of the first `end`.
                    let end = rows - wanted + done + chunk;
                    self.head_launch(&lane.stream_a, end, chunk, 0, w)?;
                    let timer = Instant::now();
                    let bytes = self.download(&w.vocab_logits, chunk * self.cfg.vocab_size * 4)?;
                    self.profile.borrow_mut().add(Phase::Head, timer);
                    logits.extend(bytes.chunks_exact(4).map(|b| f32::from_le_bytes(b.try_into().unwrap())));
                    done += chunk;
                }
            } else if wanted > 0 {
                // Lanes land their rows one after the other in the logits buffer.
                let capacity = w.vocab_logits.buffer.bytes / (self.cfg.vocab_size * 4);
                ensure!(landed + wanted <= capacity, "{logit_rows} device logit rows exceed the workspace's {capacity}");
                self.head_launch(&lane.stream_a, rows, wanted, landed, w)?;
                landed += wanted;
            }
            first += rows;
        }
        self.finish_route_step()?;
        Ok(if download { Logits::Host(logits) } else { Logits::Device(Some(self.device_logits(w, landed))) })
    }

    /// The step's token embeddings, four copies per row, into the lane's streams.
    fn gather_streams(&self, lane: &Lane<'_>, rows: usize) -> Result<()> {
        // SAFETY: the lane's token ids are copied in before the stream reaches
        // the gather; its streams hold [rows, 4, dim] BF16.
        unsafe { self.embedding.gather(lane.tokens.buffer.ptr, std::ptr::null(), rows, 4, std::ptr::null(),
            lane.stream_a.buffer.ptr, self.stream) }
    }

    /// The first `rows` rows of the workspace's vocabulary logits.
    fn device_logits(&self, w: &Workspace<'_>, rows: usize) -> DeviceLogits {
        let vocab = self.cfg.vocab_size;
        DeviceLogits { ptr: w.vocab_logits.buffer.ptr, rows, vocab, stride: vocab, stream: self.stream, greedy: None }
    }

    /// Which sparse attention a layer runs this step.
    fn attention_kind(&self, ratio: usize, tables: &StepTables) -> &'static str {
        match ratio {
            4 if tables.c4_groups > 0 => "c4",
            128 if tables.c128_groups > 0 => "c128",
            _ => "win",
        }
    }

    /// Launches `segment` through a captured graph for `key`, capturing it the
    /// first time.
    fn replay(&self, key: GraphKey, segment: impl FnOnce() -> Result<()>) -> Result<()> {
        self.replay_on(0, key, segment)
    }

    /// [`Self::replay`] on rank `rank`'s stream (its own graphs).
    fn replay_on(&self, rank: usize, key: GraphKey, segment: impl FnOnce() -> Result<()>) -> Result<()> {
        let mut key = key;
        if let Some(tp2) = &self.tp2 {
            key.tp2_broadcast = tp2.broadcast();
            key.exchange_f32 = self.exchange_policy.is_f32(key.rows, self.decode_rows);
        }
        // Diagnostic copies must never become permanent graph nodes.
        if self.tp2.as_ref().is_some_and(|t| t.checking.get()) && !self.capture_only.get() {
            return segment();
        }
        let graphs = match (rank, &self.peer) {
            (1, Some(peer)) => &peer.graphs,
            _ => &self.graphs,
        };
        let stream = self.stream_of(rank);
        if let Some(graph) = graphs.borrow().get(&key) {
            if self.capture_only.get() { return Ok(()); }
            // SAFETY: the graph's pointers are persistent engine buffers of that rank.
            return self.on(rank, || unsafe { self.library.cuda_graph_launch(graph.0, stream) });
        }
        // SAFETY: capture records launches on that rank's stream; nothing in the
        // segment synchronizes the host.
        self.on(rank, || unsafe { self.library.cuda_graph_begin_capture(stream) })?;
        let captured = segment();
        let exec = self.on(rank, || unsafe { self.library.cuda_graph_end_capture(stream) });
        captured?;
        let exec = exec?;
        if !self.capture_only.get() { self.on(rank, || unsafe { self.library.cuda_graph_launch(exec, stream) })?; }
        self.graph_captures.set(self.graph_captures.get() + 1);
        if key.attention.starts_with("tp2-") {
            self.expert_graph_captures.set(self.expert_graph_captures.get() + 1);
        }
        graphs.borrow_mut().insert(key, GraphExec(exec, self.library));
        Ok(())
    }

    /// Rank 1's decode segment of `layer`: the previous layer's FFN close, then this
    /// layer's share (see [`Self::peer_front`]), replayed as its own graph.
    fn peer_segment(&self, layer: usize, tables: &StepTables, w1: &Workspace<'_>, t: usize, cap: usize) -> Result<()> {
        let rows = Scalar::I32(t as i32);
        let lane = &w1.lanes[0];
        let segment = || -> Result<()> {
            if layer > 0 {
                self.peer_post(layer - 1, lane, 0, w1, rows)?;
            }
            self.peer_front(layer, tables, lane, 0, w1, rows, cap)
        };
        let key = GraphKey { layer, rows: t, chunked: tables.chunked,
            attention: self.attention_kind(self.peer()?.layers[layer].ratio, tables),
            table_width: tables.c4_table_width, table_stride: tables.c4_table_stride, previous: usize::from(layer > 0), tp2_broadcast: false, exchange_f32: false };
        self.replay_on(1, key, segment)
    }

    /// The layer's routed partials + shared expert, reduced, then mHC post
    /// into stream a.
    fn post(&self, w: &Workspace<'_>, lane: &Lane<'_>, ranks: usize, planes: [*const u16; 6], rows: Scalar,
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
        let Scalar::I32(count) = rows else { unreachable!() };
        let reducer = self.library.v41_compact_reducer()?;
        // SAFETY: the transport's intake planes, shared and delta are live
        // [rows, h] BF16 buffers on this device, ordered after the wave's
        // intake and the shared FFN. The next dispatch on that transport
        // follows a full stream sync (`stage_request`).
        unsafe {
            reducer.reduce_planes(planes, ranks as u32, lane.shared.buffer.ptr.cast(),
                w.delta.buffer.ptr.cast(), count as u32, self.stream)
                .with_context(|| format!("layer {layer} expert reduction"))?;
        }
        self.run("mhc_post", &[
            ("x", w.delta.buffer.ptr), ("residual", lane.stream_b.buffer.ptr), ("prev_post", lane.post.buffer.ptr),
            ("prev_comb", lane.comb.buffer.ptr), ("out", lane.stream_a.buffer.ptr),
        ], &[rows])
    }

    /// A backbone layer's [`Self::post`] for lane `index`: [`Self::post_split`] under a head split.
    #[allow(clippy::too_many_arguments)]
    fn post_layer(&self, w: &Workspace<'_>, lane: &Lane<'_>, index: usize, ranks: usize, planes: [*const u16; 6],
        rows: Scalar, layer: usize) -> Result<()> {
        if self.peer.is_some() {
            self.post_split(w, lane, index, ranks, planes, rows, layer)
        } else {
            self.post(w, lane, ranks, planes, rows, layer)
        }
    }

    /// [`Self::post`] on rank 0 under a head split (lane `index`): the routed sum
    /// with this GPU's shared-expert half (or the local layer's output, or the
    /// half alone when routed experts are skipped), sent to rank 1 first (not after
    /// the last layer), plus rank 1's half, then mHC post.
    #[allow(clippy::too_many_arguments)]
    fn post_split(&self, w: &Workspace<'_>, lane: &Lane<'_>, index: usize, ranks: usize, planes: [*const u16; 6],
        rows: Scalar, layer: usize) -> Result<()> {
        let Scalar::I32(count) = rows else { unreachable!() };
        if self.tp2_layer(layer) && ranks == LOCAL_EXPERTS {
            return self.tp2_post(0, layer, index, w, lane, count as usize);
        }
        let base = match ranks {
            SKIPPED_EXPERTS => lane.shared.buffer.ptr,
            LOCAL_EXPERTS => self.local_output(layer, index)?,
            _ => {
                let reducer = self.library.v41_compact_reducer()?;
                // SAFETY: as in `post`.
                unsafe {
                    reducer.reduce_planes(planes, ranks as u32, lane.shared.buffer.ptr.cast(),
                        w.delta.buffer.ptr.cast(), count as u32, self.stream)
                        .with_context(|| format!("layer {layer} expert reduction"))?;
                }
                w.delta.buffer.ptr
            }
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
    fn peer_front(&self, layer: usize, tables: &StepTables, lane: &Lane<'_>, index: usize, w: &Workspace<'_>,
        rows: Scalar, cap: usize) -> Result<()> {
        let peer = self.peer()?;
        let weights = &peer.layers[layer];
        let exchange = self.exchange()?;
        if layer == 0 {
            self.record_peer(1, "entry", DIRECT, false, 0, index);
            exchange.wait(1, DIRECT)?;
        }
        self.attention_front(1, layer, weights, tables, lane, w, rows, cap)?;
        self.attention_sum(1, layer, index, w, rows)?;
        self.attention_back(1, weights, lane, w, rows, cap, w.sum.buffer.ptr)?;
        self.run_on(1, true, &format!("shared_ffn_m{cap}"), &[
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
    fn peer_post(&self, layer: usize, lane: &Lane<'_>, index: usize, w: &Workspace<'_>, rows: Scalar) -> Result<()> {
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
        lane: &Lane<'_>,
        w: &Workspace<'_>,
        rows: Scalar,
        cap: usize,
    ) -> Result<()> {
        self.attention_front(0, layer, weights, tables, lane, w, rows, cap)?;
        self.attention_back(0, weights, lane, w, rows, cap, w.delta.buffer.ptr)
    }

    /// [`Self::attention`] on rank 0 under a head split: the front, the attention
    /// all-reduce with rank 1 (lane `index`'s slot), then the back on the sum.
    #[allow(clippy::too_many_arguments)]
    fn attention_split(&self, layer: usize, weights: &LayerWeights<'_>, tables: &StepTables, lane: &Lane<'_>,
        index: usize, w: &Workspace<'_>, rows: Scalar, cap: usize) -> Result<()> {
        self.attention_front(0, layer, weights, tables, lane, w, rows, cap)?;
        self.attention_sum(0, layer, index, w, rows)?;
        self.attention_back(0, weights, lane, w, rows, cap, w.sum.buffer.ptr)
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
        lane: &Lane<'_>,
        w: &Workspace<'_>,
        rows: Scalar,
        cap: usize,
    ) -> Result<()> {
        let m = &lane.tables;
        let (pools, rope_window, rope_compressed) = match (rank, &self.peer) {
            (1, Some(peer)) => (&peer.pools, &peer.rope_window, &peer.rope_compressed),
            _ => (&self.pools, &self.rope_window, &self.rope_compressed),
        };
        let cache = &pools[layer];
        let ratio = weights.ratio;
        let rope = if ratio == 0 { rope_window } else { rope_compressed };
        let mode = if tables.decode { "decode" } else { "prefill" };
        let split = weights.split;
        let a = &lane.stream_a;
        self.run_on(rank, false, "mhc_pre", &[
            ("residual", a.buffer.ptr), ("fn", weights.ptr("attn.fn")?), ("scale", weights.ptr("attn.scale")?),
            ("base", weights.ptr("attn.base")?), ("norm", weights.ptr("attn.norm")?), ("post", lane.post.buffer.ptr),
            ("comb", lane.comb.buffer.ptr), ("y", w.y.buffer.ptr), ("scratch", w.scratch.buffer.ptr),
        ], &[rows])?;
        self.run_on(rank, split, &format!("producer_m{cap}"), &[
            ("hidden", w.y.buffer.ptr), ("positions", m.positions.buffer.ptr), ("main_slots", m.main_slots.buffer.ptr),
            ("cos_sin", rope.buffer.ptr), ("w_qkv", weights.ptr("w_qkv")?), ("w_qkv_scale", weights.ptr("w_qkv_scale")?),
            ("w_q", weights.ptr("w_q")?), ("w_q_scale", weights.ptr("w_q_scale")?), ("q_norm", weights.ptr("q_norm")?),
            ("kv_norm", weights.ptr("kv_norm")?), ("main_kv_cache", cache.main.buffer.ptr), ("query", w.query.buffer.ptr),
            ("q_rank", w.q_rank.buffer.ptr), ("scratch", w.scratch.buffer.ptr),
        ], &[rows])?;
        let (attention, indexed_cache, indexed_indices, indexed_lengths) =
            self.compress(rank, layer, weights, cache, tables, m, w, rope, rows, cap)?;
        debug_assert_eq!(attention, self.attention_kind(ratio, tables));
        self.run_on(rank, split, &format!("sparse_mla_{mode}_{attention}_m{cap}"), &[
            ("q", w.query.buffer.ptr), ("swa_cache", cache.main.buffer.ptr), ("swa_indices", m.swa_indices.buffer.ptr),
            ("swa_lengths", m.swa_lengths.buffer.ptr), ("indexed_cache", indexed_cache),
            ("indexed_indices", indexed_indices), ("indexed_lengths", indexed_lengths),
            ("attn_sink", weights.ptr("attn_sink")?), ("out", w.attn_out.buffer.ptr), ("scratch", w.scratch.buffer.ptr),
        ], &[rows])?;
        self.run_on(rank, split, &format!("wo_m{cap}"), &[
            ("o", w.attn_out.buffer.ptr), ("positions", m.positions.buffer.ptr), ("cos_sin", rope.buffer.ptr),
            ("wo_a", weights.ptr("wo_a")?), ("wo_a_scale", weights.ptr("wo_a_scale")?), ("wo_b", weights.ptr("wo_b")?),
            ("wo_b_scale", weights.ptr("wo_b_scale")?), ("out", w.delta.buffer.ptr), ("scratch", w.scratch.buffer.ptr),
        ], &[rows])
    }

    /// mHC post_pre over the attention output `x` (stream a -> stream b, y), then on
    /// rank 0 the router scores and expert input quantization.
    #[allow(clippy::too_many_arguments)]
    fn attention_back(&self, rank: usize, weights: &LayerWeights<'_>, lane: &Lane<'_>, w: &Workspace<'_>, rows: Scalar,
        cap: usize, x: *mut c_void) -> Result<()> {
        let (a, b) = (&lane.stream_a, &lane.stream_b);
        self.run_on(rank, false, &format!("mhc_post_pre_m{cap}"), &[
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
        let (groups, metadata, names) = if ratio == 4 {
            (tables.c4_groups, &m.c4, &tables.c4_tables)
        } else {
            (tables.c128_groups, &m.c128, &tables.c128_tables)
        };
        let compressed = cache.compressed.as_ref().context("compressed cache")?.buffer.ptr;
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
                ("index_cache", cache.index.as_ref().context("index cache")?.buffer.ptr),
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
        .with_context(|| format!("layer {layer} compressor"))?;
        if groups == 0 {
            return Ok(window);
        }
        if ratio == 128 {
            return Ok(("c128", compressed, m.c128_indices.buffer.ptr, m.c128_lengths.buffer.ptr));
        }
        self.run_on(rank, false, &format!("index_producer_m{cap}"), &[
            ("q_rank", w.q_rank.buffer.ptr), ("hidden", w.y.buffer.ptr), ("positions", m.positions.buffer.ptr),
            ("cos_sin", rope.buffer.ptr), ("w_q", weights.ptr("index_w_q")?), ("w_q_scale", weights.ptr("index_w_q_scale")?),
            ("w_proj", weights.ptr("index_w_proj")?), ("query", w.index_query.buffer.ptr),
            ("head_weights", w.index_weights.buffer.ptr), ("scratch", w.scratch.buffer.ptr),
        ], &[rows])?;
        let mode = if tables.decode { "decode" } else { "prefill" };
        self.run_on(rank, false, &format!("index_topk_{mode}_m{cap}"), &[
            ("q_fp8", w.index_query.buffer.ptr), ("weights", w.index_weights.buffer.ptr),
            ("index_k_cache", cache.index.as_ref().context("index cache")?.buffer.ptr),
            ("page_table", m.c4_page_table.buffer.ptr), ("cache_lengths", m.c4_visible.buffer.ptr),
            ("output_indices", w.selected.buffer.ptr), ("scratch", w.topk_scratch.buffer.ptr),
        ], &[rows, Scalar::I32(tables.c4_table_width as i32), Scalar::I32(tables.c4_table_stride as i32)])?;
        Ok(("c4", compressed, w.selected.buffer.ptr, m.c4_indexed_lengths.buffer.ptr))
    }

    /// Router download, shared expert launch, host routing, Spark exchange
    /// and plane uploads; returns the Spark rank count.
    #[allow(clippy::too_many_arguments)]
    /// One Spark request for `rows` wire rows with `topk` routes each.
    fn expert_request(&self, layer: usize, rows: usize, routes: Vec<ExpertProtocolV2RouteEntry>,
        wire: bytes::Bytes, kind: ExpertV2SourceKind) -> Result<ExpertProtocolV2Request> {
        spark_request(self.cfg.dim, self.cfg.n_activated_experts, layer, rows, routes, wire, kind)
    }

    /// A decode unit's Spark layer on the device exchange: the wave goes out
    /// from the stream, the shared expert runs while the Sparks compute, and
    /// the stream waits for the partials (the next segment's post reduces).
    fn device_experts(&self, link: &crate::shared::spark_intake::SparkDeviceLink<'_>, layer: usize, t: usize,
        w: &Workspace<'_>, lane: &Lane<'_>, cap: usize) -> Result<()> {
        let (h, topk) = (self.cfg.dim, self.cfg.n_activated_experts);
        let sized = |dev: &Dev<'_>, bytes: usize| cuteafd_ffi::CuteafdDeviceBuffer { bytes, ..dev.buffer };
        // SAFETY: routes and wire rows are complete in stream order (the
        // segment's router ran before); the previous wave on this link was
        // collected and reduced earlier on the stream.
        unsafe {
            link.dispatch(layer, t, DEVICE_DECODE, sized(&w.route_ids, t * topk * 4),
                sized(&w.route_weights, t * topk * 4), sized(&w.wire, t * (h + h / 32)), self.stream)?;
        }
        self.shared_ffn(layer, w, lane, Scalar::I32(t as i32), cap, &self.weights.layers[layer])?;
        // SAFETY: the dispatch above is this wait's wave.
        unsafe { link.collect(self.stream) }
    }

    /// Connects the device-driven exchange for decode/verify steps (see
    /// [`Self::device_link`]); its transport is warmed with a full wave.
    pub fn attach_device_link(&mut self, peers: &[std::net::SocketAddr], executors: &[u64],
        config: cuteafd_transport::TcpTransportConfig) -> Result<()> {
        let (dim, topk, experts, rows) =
            (self.cfg.dim, self.cfg.n_activated_experts, self.cfg.n_routed_experts, self.decode_rows);
        let warm = spark_request(dim, topk, self.cfg.n_layers - 1, rows, (0..rows * topk).map(|i|
            ExpertProtocolV2RouteEntry { row_index: (i / topk) as u32, expert_id: (i % experts) as u32,
                gate_weight: 0.0 }).collect(), vec![0u8; rows * (dim + dim / 32)].into(), ExpertV2SourceKind::Decode)?;
        let build: cuteafd_transport::expert::DeviceBuild = Box::new(move |wave, routes, wire| {
            let kind = if wave.kind == DEVICE_DECODE { ExpertV2SourceKind::Decode } else { ExpertV2SourceKind::Prefill };
            spark_request(dim, topk, wave.layer as usize, wave.rows as usize, routes, wire, kind)
        });
        self.device_link = Some(crate::shared::spark_intake::SparkDeviceLink::new(self.library, self.device, peers,
            executors, rows, topk, dim + dim / 32, dim * 2, config, Some(warm), build)?);
        Ok(())
    }

    /// After a decode/verify step's results were read: the device exchange's
    /// waves of that step all succeeded (otherwise the step must be discarded).
    pub fn check_device(&self) -> Result<()> {
        self.device_link.as_ref().map_or(Ok(()), |link| link.check())
    }

    /// Connects every Spark rank and registers full-size buffers with one
    /// prefill-sized request of zero rows, so the first real request does not
    /// pay for connection setup (about 0.6 s).
    pub fn warm_transport(&self, transport: &mut SparkLink<'_>, runtime: &tokio::runtime::Runtime) -> Result<()> {
        let (rows, h, experts, topk) = (self.prefill_rows, self.cfg.dim, self.cfg.n_routed_experts, self.cfg.n_activated_experts);
        let routes = (0..rows * topk).map(|i| ExpertProtocolV2RouteEntry {
            row_index: (i / topk) as u32, expert_id: (i % experts) as u32, gate_weight: 0.0,
        }).collect();
        let request = self.expert_request(self.cfg.n_layers - 1, rows, routes, vec![0u8; rows * (h + h / 32)].into(),
            ExpertV2SourceKind::Prefill)?;
        let wave = transport.dispatch(&request)?;
        runtime.block_on(transport.receive(wave, rows, self.stream))?;
        self.sync()
    }

    pub(crate) fn local_layers(&self) -> usize {
        self.tp2.as_ref().map_or_else(|| self.local.borrow().as_ref().map_or(0, |l| l.layers()), |t| t.layers.end)
    }

    pub fn install_local(&mut self, local: Option<super::local::LocalExperts<'a>>) {
        *self.local.borrow_mut() = local;
    }

    fn local_output(&self, _layer: usize, _index: usize) -> Result<*mut c_void> {
        Ok(self.local.borrow().as_ref().context("local experts")?.output.buffer.ptr)
    }

    /// The shared expert on the unit's FFN input `y` into the lane's `shared`.
    fn shared_ffn(&self, layer: usize, w: &Workspace<'_>, lane: &Lane<'_>, rows: Scalar, cap: usize,
        weights: &LayerWeights<'_>) -> Result<()> {
        // A head split's layers hold half the shared expert (its program family's share).
        self.run_on(0, weights.split, &format!("shared_ffn_m{cap}"), &[
            ("x", w.y.buffer.ptr), ("w13", weights.ptr("w13")?), ("w13_scale", weights.ptr("w13_scale")?),
            ("w2", weights.ptr("w2")?), ("w2_scale", weights.ptr("w2_scale")?), ("out", lane.shared.buffer.ptr),
            ("scratch", w.scratch.buffer.ptr),
        ], &[rows]).with_context(|| format!("layer {layer} shared expert"))
    }

    /// Shared and routed experts of a coordinator-resident layer: routes,
    /// wire rows and results stay on the device, the host only enqueues.
    fn local_experts(&self, layer: usize, t: usize, w: &Workspace<'_>, lane: &Lane<'_>, cap: usize, index: usize) -> Result<()> {
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

    /// Every local-expert row shape is captured at startup, without executing
    /// uninitialized routes or unmatched peer waits. Pointer ownership is fixed.
    pub fn warm_local_graphs(&self) -> Result<()> {
        if self.local_layers() == 0 { return Ok(()); }
        if self.decode_workspace.borrow().is_none() {
            *self.decode_workspace.borrow_mut() = Some(self.workspace(self.decode_rows, 1)?);
        }
        if let Some(peer) = &self.peer {
            if peer.decode_workspace.borrow().is_none() {
                *peer.decode_workspace.borrow_mut() = Some(self.workspace_on(1, self.decode_rows, 1)?);
            }
        }
        self.warm_tp2_routers()?;
        self.capture_only.set(true);
        let result = (|| {
            let work = self.decode_workspace.borrow();
            let w = work.as_ref().context("decode workspace")?;
            for t in 1..=self.decode_rows {
                for layer in 0..self.local_layers() {
                    self.local_experts(layer, t, w, &w.lanes[0], self.decode_rows, 0)?;
                    if self.tp2_layer(layer) {
                        let work = self.peer()?.decode_workspace.borrow();
                        self.peer_experts(layer, 0, work.as_ref().context("peer decode workspace")?, t, true)?;
                    }
                }
            }
            Ok(())
        })();
        self.capture_only.set(false);
        result
    }

    /// Waits for the unit's routes and wire rows and builds its Spark request.
    fn stage_request(&self, layer: usize, t: usize, w: &Workspace<'_>, kind: ExpertV2SourceKind,
        transport: &mut SparkLink<'_>) -> Result<ExpertProtocolV2Request> {
        let (h, topk) = (self.cfg.dim, self.cfg.n_activated_experts);
        let timer = Instant::now();
        let (route_bytes, wire_bytes) = (t * topk * 4, t * (h + h / 32));
        let host = w.router_host.buffer;
        let at = |offset: usize| cuteafd_ffi::CuteafdHostBuffer {
            // SAFETY: ids, weights and wire rows are consecutive inside the pinned buffer.
            ptr: unsafe { host.ptr.cast::<u8>().add(offset) }.cast(),
            bytes: host.bytes - offset,
            ..host
        };
        // Prefill rows go straight into this transport's registered egress
        // buffer and out from there to every rank; small waves keep a copy.
        let egress = transport.egress(wire_bytes)?;
        // SAFETY: the pinned regions are large enough; the sync below completes them.
        unsafe {
            self.library.copy_d2h_host_buffer_async(at(0), w.route_ids.buffer, route_bytes, self.stream)?;
            self.library.copy_d2h_host_buffer_async(at(route_bytes), w.route_weights.buffer, route_bytes, self.stream)?;
            let target = egress.unwrap_or(at(2 * route_bytes));
            self.library.copy_d2h_host_buffer_async(target, w.wire.buffer, wire_bytes, self.stream)?;
        }
        self.sync()?;
        self.profile.borrow_mut().add(Phase::RouterSync, timer);
        let timer = Instant::now();
        let staged = w.router_host.bytes();
        let word = |offset: usize, i: usize| u32::from_le_bytes(staged[offset + i * 4..][..4].try_into().unwrap());
        let routes = (0..t * topk).map(|i| ExpertProtocolV2RouteEntry {
            row_index: (i / topk) as u32,
            expert_id: word(0, i),
            gate_weight: f32::from_bits(word(route_bytes, i)),
        }).collect();
        let wire = match egress {
            Some(_) => transport.egress_payload(wire_bytes)?,
            None => staged[2 * route_bytes..2 * route_bytes + wire_bytes].to_vec().into(),
        };
        let request = self.expert_request(layer, t, routes, wire, kind);
        self.profile.borrow_mut().add(Phase::Routing, timer);
        request
    }

    /// Receives a unit's partial rows into its transport's intake planes; the
    /// reduce that reads them is ordered after the intake on the stream.
    async fn land(&self, transport: &mut SparkLink<'_>, wave: SparkExpertWave, t: usize, _w: &Workspace<'_>)
        -> Result<usize> {
        let timer = Instant::now();
        transport.receive(wave, t, self.stream).await?;
        self.profile.borrow_mut().add(Phase::Experts, timer);
        Ok(transport.world_size())
    }

    /// A decode unit's experts; returns the rank count its post needs
    /// ([`LOCAL_EXPERTS`] for a coordinator-resident layer).
    #[allow(clippy::too_many_arguments)]
    fn decode_experts(
        &self,
        layer: usize,
        t: usize,
        w: &Workspace<'_>,
        lane: &Lane<'_>,
        cap: usize,
        transport: Option<&mut SparkLink<'_>>,
        runtime: &tokio::runtime::Runtime,
    ) -> Result<usize> {
        if self.skip_routed {
            self.shared_ffn(layer, w, lane, Scalar::I32(t as i32), cap, &self.weights.layers[layer])?;
            return Ok(SKIPPED_EXPERTS);
        }
        if layer < self.local_layers() {
            self.local_experts(layer, t, w, lane, cap, 0)?;
            return Ok(LOCAL_EXPERTS);
        }
        let transport = transport.context("no transport")?;
        // Decode rows poll without the prefill spin quantum.
        let request = self.stage_request(layer, t, w, ExpertV2SourceKind::Decode, transport)?;
        let wave = transport.dispatch(&request)?;
        // The shared expert runs on the GPU while the Sparks compute.
        self.shared_ffn(layer, w, lane, Scalar::I32(t as i32), cap, &self.weights.layers[layer])?;
        runtime.block_on(self.land(transport, wave, t, w))
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
            w.head.as_ref().context("LM head")?.launch(w.y.buffer.ptr.cast::<u8>().add((t - n) * h * 2).cast(), self.weights.head.buffer.ptr.cast(),
                w.vocab_logits.buffer.ptr.cast::<f32>().add(at * vocab), n as u32, self.stream)
        }
    }
}

/// [`cuteafd_transport::expert::DeviceWave::kind`] of decode/verify waves.
const DEVICE_DECODE: u32 = 0;

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
