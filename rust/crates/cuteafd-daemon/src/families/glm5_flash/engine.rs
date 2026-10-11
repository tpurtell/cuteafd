//! GLM 5.3 Flash (glm5_next) coordinator over the exported glmf_* programs.
//!
//! Four mHC streams (BF16 `[rows, 4, H]`) run through every layer: the
//! attention site's collapse and input norm (`mhc_pre`, or the previous
//! layer's fused `mhc_post_pre`), the attention sublayer, `mhc_post_pre`
//! back into the streams and onto the FFN site, the FFN, and the next fused
//! post/pre (`mhc_post` after the last layer, then the stream-mean head).
//!
//! Attention: KDA layers keep per-sequence recurrent state (FP32, or BF16 with
//! `--kda-state bf16`, `[64, 128, 128]`) and short-conv state (the last three q/k/v inputs) in
//! slot pools, one slot per sequence shared by every KDA layer (every KDA
//! layer's pool back to back, so `glmf_kda_commit` reaches all of them in one
//! launch). A speculative verify step (`verify_spec`) leaves that state alone
//! and records each row's replay inputs; `commit` then applies the accepted
//! rows with the recurrent step's own arithmetic. MLA layers
//! keep FP8 528-byte latent records in 64-row pages, and their DSA indexer
//! keeps one FP8 key per completed 4-token pool in pool pages (64 pools per
//! page). The pooled keys are computed from the pool's BF16 keys and gates:
//! with `--index-cache keys` every token's key | gate row is kept beside its
//! record (512 B per token and MLA layer); with `--index-cache compact` only
//! the rows of each sequence's open pool are (at most three, in a per-slot
//! tail that the KDA slot regions carry, so marks and slot copies include it),
//! and a speculative step records its rows for `commit` to rebuild the tails.
//! Both give the same pooled keys bit for bit. The indexer selects every
//! earlier token up to 2051 tokens; past that the top 512 pools
//! (glmf_index_topk) expand to tokens plus the open tail pool.
//!
//! FFN: dense SwiGLU (clamped at 10), or the MoE: FP32 router logits, the
//! native sigmoid top-8 select, the shared expert, and routed experts from
//! the checkpoint's FP8/NVFP4 on this GPU (the TP1 package, all routed layers
//! resident by default; a paging window only when explicitly requested for
//! diagnostics) or on the Sparks.
//!
//! Two-GPU head split ([`GlmfEngine::attach_peer`], `--split-device`): each GPU runs half
//! the KDA and MLA heads (its KDA state, its MLA queries, a partial o_proj) and half the
//! dense / shared-expert intermediate (a partial sum); the mHC streams, the MLA latent
//! records and the DSA indexer are computed on both (identical bits), and the partials meet
//! over peer memory (`shared::peer_split`): each GPU pushes its partial, waits for the
//! other's and adds the two (the same bits in either order), so both residual streams stay
//! identical. Router, routed experts, LM head and drafter stay on rank 0; rank 1 is queued a
//! layer ahead of rank 0's expert exchange.
use crate::shared::decode_graph::{GraphBank, GraphOwner, GraphStats};
use super::packing;
use super::weights::{GlmfLayer, GlmfWeights};
use crate::shared::peer_split::{PeerExchange, RankDevice, DIRECT};
use super::head::GlmfHead;
use crate::families::glm5::dflash::TargetHead;
use crate::shared::experts::fp8::{Fp8Experts, Fp8Layer};
use crate::shared::launch_grid::Fp8QuantizeGrid;
use crate::shared::memory::{DeviceAllocation, HostAllocation};
use crate::shared::token_io::{DeviceLogits, TokenEmbedding, TokenSelector};
use anyhow::{ensure, Context, Result};
use cuteafd_ffi::programs::{Programs, Scalar, VocabularyHead};
use cuteafd_ffi::NativeLibrary;
use cuteafd_loader::formats::fp8_experts::Fp8ExpertTensors;
use cuteafd_loader::families::glm5_flash::{GlmNextAttention, GlmNextConfig};
use cuteafd_loader::serving_capacity::{glmf_lane_bytes, glmf_step_scratch, glmf_step_workspaces, glmf_table_pages,
    glmf_temporary_bytes, GlmfKdaState, GlmfScratchOptions, GlmfStepShape};
use crate::shared::spark_intake::SparkLink;
use cuteafd_transport::expert::{SparkExpertWave, EXPERT_PROTOCOL_V2_FLAG_V41_COMPACT_BF16};
use cuteafd_transport::{
    ExpertProtocolV2Request, ExpertProtocolV2RouteEntry, ExpertProtocolV2RowDescriptor, ExpertV2Dtype, ExpertV2SourceKind,
};
use std::cell::RefCell;
use std::ffi::c_void;
use std::rc::Rc;

type Dev<'a> = DeviceAllocation<'a>;

pub(crate) const PAGE_ROWS: usize = 64;
/// Rows of the decode-route programs (`_m64`).
pub(crate) const DECODE_ROWS: usize = 64;
/// Rows of the wide decode-route programs (`_m128`): with `--decode-rows 128`, a decode or verify
/// step of more than [`DECODE_ROWS`] rows runs them (fewer rows keep the `_m64` programs).
pub(crate) const WIDE_DECODE_ROWS: usize = 128;
/// The prefill programs' capacity.
const PREFILL_CAP: &str = "m4096";
/// Selected-slot row width of the sparse MLA programs (2048 + 3, padded to 64).
pub(crate) const SPARSE_TOPK: usize = 2112;
/// Most decode rows the FP8 GEMVs take (the programs' FP8_ROWS).
const FP8_ROWS: i32 = 16;
/// FP8 latent record bytes (512 E4M3 + 4 FP32 group scales).
pub(crate) const RECORD_BYTES: usize = 528;

/// The decode program capacity a step of `rows` rows runs: `m64`, or the wide `m128` past
/// [`DECODE_ROWS`] rows.
pub(crate) fn decode_cap(rows: usize) -> &'static str {
    if rows > DECODE_ROWS { "m128" } else { "m64" }
}

/// Whether `cap` is a decode program capacity (`m64`, `m128`), not the prefill programs'.
fn is_decode(cap: &str) -> bool {
    cap != PREFILL_CAP
}

/// Rows a speculative step of decode capacity `cap` records per layer: `REPLAY_ROWS` of the fork's
/// `_glmf_kernels.py` for the `_m64` programs, the wide programs' 128 rows for `_m128`.
fn replay_rows_of(cap: &str) -> usize {
    if cap == "m128" { WIDE_DECODE_ROWS } else { DECODE_ROWS }
}

/// Bytes of one KDA layer's replay record of `rows` rows (`kda_replay_layout`): k | decay | v
/// FP32 per row and head, beta FP32, the q/k/v in-projection row BF16.
pub(crate) fn replay_bytes(heads: usize, channels: usize, rows: usize) -> usize {
    rows * heads * 3 * 128 * 4 + rows * heads * 4 + rows * channels * 2
}

/// The replay commit `name` for records of `rows` rows: itself at the `_m64` programs' 64, its wide
/// `_m128` variant past them.
fn commit_for(name: &str, rows: usize) -> String {
    if rows > DECODE_ROWS { format!("{name}_m{WIDE_DECODE_ROWS}") } else { name.to_string() }
}

/// Waves the sparse MLA decode's split planner allows (SparkInfer's `_CEIL_WAVES_MAX`), and the
/// decode kernel's CTAs per row at one split (64 heads in blocks of 16).
const SPARSE_MLA_WAVES: usize = 3;
const SPARSE_MLA_HEAD_BLOCKS: usize = 4;

/// The most rows a verify step schedules, up to `decode_rows`: the drafts' budget. Past the `_m64`
/// programs it is the largest row count whose one-split sparse MLA launch (4 CTAs a row, one CTA
/// per SM) fits three waves of the GPU's `sms` multiprocessors, read from the device: one row more
/// starts a nearly empty fourth wave. On an RTX 5090 (170 SMs) the wide sparse MLA takes 313 us a
/// layer at 127 rows (508 CTAs) and 407 us at 128 (512 CTAs). So 127 rows on 170 SMs, 128 on 188,
/// 99 on 132; never fewer than the `_m64` programs' 64, and `decode_rows` itself at 64. The wide
/// sparse MLA plans its 128-row bucket at one split (`full_launch_splits=1` at export; at 188 SMs
/// the planner's own plan), so an object exported on either card keeps 4 CTAs a row.
pub(crate) fn verify_budget(decode_rows: usize, sms: usize) -> usize {
    cuteafd_loader::serving_capacity::glmf_graphs::verify_budget(decode_rows, sms)
}

/// Hands the rows an even share of `verify_rows` leaves over to the first sequences, one each.
/// `limits` are the drafts each sequence may add after its next token, `room` the even share's:
/// 127 rows at 16 sequences give 6 each and leave 15 rows, so 15 sequences may draft 7. A
/// sequence takes one only with the full `room` and `more(i)` (it speculates, and its tokens and
/// capacity allow another draft). A share that leaves nothing over changes nothing (64 or 128
/// rows at 16 sequences).
pub(crate) fn hand_out_remainder(limits: &mut [usize], room: usize, verify_rows: usize, more: impl Fn(usize) -> bool) {
    let mut left = verify_rows.saturating_sub(limits.len() * (room + 1));
    for (i, limit) in limits.iter_mut().enumerate() {
        if left == 0 {
            break;
        }
        if *limit == room && more(i) {
            *limit += 1;
            left -= 1;
        }
    }
}

/// Bytes per KDA layer over `kda_heads` heads, as [`Caches::new`] allocates them: one sequence's
/// recurrent state `[heads, 128, 128]` (FP32, or BF16 with `--kda-state bf16`) and BF16 conv window
/// (the last three q/k/v inputs), the two regions `slot_regions_on` hands out per layer and a prefix
/// mark copies, and the layer's speculative replay record of `decode_rows` rows (`--decode-rows`).
pub(crate) fn kda_layer_bytes(cfg: &GlmNextConfig, kda_heads: usize, state: KdaState, decode_rows: usize)
    -> (usize, usize, usize) {
    let d = kda_heads * cfg.kda_head_dim;
    (d * cfg.kda_head_dim * state.bytes(), 3 * 3 * d * 2, replay_bytes(kda_heads, 3 * d, decode_rows))
}
/// One token's BF16 DSA index key | gate row (128 keys after k_norm, 128 gates).
pub(crate) const KEY_BYTES: usize = 512;
/// One sequence's index tail in one MLA layer (`TAIL_BYTES` of the fork's `_glmf_kernels.py`):
/// an i32 row count (0..=3), 12 zero bytes, then the key | gate rows of its open pool, zero
/// past the count.
pub(crate) const TAIL_BYTES: usize = 16 + 3 * KEY_BYTES;

/// The DSA index cache (`--index-cache`).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, clap::ValueEnum)]
pub(crate) enum IndexCache {
    /// Every token's BF16 key | gate row beside its latent record: 11,804 B per token.
    #[default]
    Keys,
    /// The pooled keys alone, plus a tail of at most three BF16 key | gate rows per sequence and
    /// MLA layer (`glmf_index_producer_c_*`, `glmf_kda_commit_c`): 6,172 B per token, the same
    /// pooled keys bit for bit. A two-GPU head split keeps `keys`.
    Compact,
}

impl From<IndexCache> for cuteafd_loader::serving_capacity::GlmfIndexCache {
    fn from(cache: IndexCache) -> Self {
        match cache {
            IndexCache::Keys => Self::Keys,
            IndexCache::Compact => Self::Compact,
        }
    }
}

/// Storage of the KDA recurrent state (`--kda-state`). Every program computes in FP32; a BF16
/// state is rounded to nearest even after every decode, verify and commit row (after the row's
/// read-out, so a verify committed at k rows stores the bits of k serial steps) and in the chunked
/// prefill where each window of tiles stores it (`bf16-tile`: after every 16-row tile).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, clap::ValueEnum)]
pub(crate) enum KdaState {
    #[default]
    F32,
    Bf16,
    /// BF16 with the chunked prefill rounded after every 16-row tile.
    #[value(name = "bf16-tile")]
    Bf16Tile,
}

impl KdaState {
    /// The `--kda-state` value.
    pub(crate) fn name(self) -> &'static str {
        match self {
            Self::F32 => "f32",
            Self::Bf16 => "bf16",
            Self::Bf16Tile => "bf16-tile",
        }
    }

    /// Bytes of one state element.
    pub(crate) fn bytes(self) -> usize {
        if self == Self::F32 { 4 } else { 2 }
    }

    /// The `glmf_kda_*` program of capacity `cap` (`m64`, `m4096`) for this state.
    pub(crate) fn program(self, cap: &str) -> String {
        match (self, cap) {
            (Self::F32, _) => format!("kda_{cap}"),
            (Self::Bf16Tile, "m4096") => format!("kda_s16t_{cap}"),
            _ => format!("kda_s16_{cap}"),
        }
    }

    /// The verify-by-replay commit program for this state.
    pub(crate) fn commit_program(self) -> &'static str {
        if self == Self::F32 { "kda_commit" } else { "kda_commit_s16" }
    }

    /// The commit with the compact index cache (`--index-cache compact`), which also rebuilds the
    /// index tails in the same launch, for this state.
    pub(crate) fn compact_commit_program(self) -> &'static str {
        if self == Self::F32 { "kda_commit_c" } else { "kda_commit_c_s16" }
    }
}

impl From<KdaState> for GlmfKdaState {
    fn from(state: KdaState) -> Self {
        match state {
            KdaState::F32 => Self::F32,
            KdaState::Bf16 => Self::Bf16,
            KdaState::Bf16Tile => Self::Bf16Tile,
        }
    }
}

/// Where the KDA speculative replay records live (`--replay-records`).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash, clap::ValueEnum)]
pub(crate) enum ReplayRecords {
    /// Their own allocation beside the KDA state.
    #[default]
    Own,
    /// In the prefill lanes' shared scratch, which no decode step reads. A record lives only
    /// from a speculative verify to its commit, and nothing prefills in between (the serving
    /// loop verifies, selects, emits and commits before its next prefill round; `RecordGuard`
    /// refuses a commit otherwise). One GPU, with the step workspaces allocated before the KV
    /// pool (the measured admission).
    Shared,
}

/// With the replay records in the prefill lanes' scratch, a prefill between a speculative verify
/// and its commit would overwrite the records the commit reads. Every use of rank 0's prefill
/// lanes and every speculative verify is counted, and a commit of records a prefill may have
/// overwritten fails instead of applying them.
#[derive(Debug, Default)]
pub(crate) struct RecordGuard {
    shared: bool,
    /// Uses of the prefill lanes so far, and their count when a speculative verify last recorded.
    prefills: std::cell::Cell<u64>,
    recorded_at: std::cell::Cell<Option<u64>>,
}

impl RecordGuard {
    fn new(records: ReplayRecords) -> Self {
        Self { shared: records == ReplayRecords::Shared, ..Self::default() }
    }

    /// The prefill lanes (and their scratch) are in use.
    fn prefilled(&self) {
        self.prefills.set(self.prefills.get() + 1);
    }

    /// A speculative verify recorded its rows.
    fn recorded(&self) {
        self.recorded_at.set(Some(self.prefills.get()));
    }

    /// Whether a commit may read the records: always with records of their own; with shared ones,
    /// only when no prefill has run since the last speculative verify.
    fn check_commit(&self) -> Result<()> {
        ensure!(!self.shared || self.recorded_at.get() == Some(self.prefills.get()),
            "a prefill ran between a speculative verify and its commit, over the replay records the commit \
            reads (--replay-records shared): commit every verify before the next prefill");
        Ok(())
    }
}

const MAX_RANKS: usize = 6;
/// Lanes a long Spark prefill chunk splits into by default, and at most (`--prefill-lanes`; each
/// lane's GPU layers run while the other lanes' Spark waves are in flight, one transport per
/// lane), and the fewest rows per lane worth another exchange per layer.
pub(crate) const DEFAULT_PREFILL_LANES: usize = 2;
pub(crate) const MAX_PREFILL_LANES: usize = 4;
const MIN_LANE_ROWS: usize = 256;

/// The longest chunk `lanes` lanes of `rows` rows take. Multi-lane splits must land on MLA page
/// boundaries. A narrow workspace still accepts a single unpadded tail, but cannot advertise
/// unusable padding.
fn prefill_lane_capacity(lanes: usize, rows: usize) -> usize {
    rows.max(lanes * (rows / PAGE_ROWS) * PAGE_ROWS)
}

/// How a chunk of `tokens` rows runs on up to `lanes` lanes of `rows` rows: (lanes, rows per
/// lane). One lane per `MIN_LANE_ROWS` rows, at most `lanes`, and at least as many as the rows
/// need once each lane's share is rounded up to whole MLA pages: every lane but the last holds
/// the same number of pages, so each lane's pools and pages start where the previous lane's end.
/// Two lanes keep the rule they always had: two from twice `MIN_LANE_ROWS` rows (or when one
/// lane cannot hold the chunk), cut at the middle rounded up to a page.
fn prefill_lane_plan(tokens: usize, lanes: usize, rows: usize) -> Result<(usize, usize)> {
    ensure!(tokens > 0 && tokens <= prefill_lane_capacity(lanes, rows),
        "prefill of {tokens} tokens exceeds {lanes} lanes of {rows} rows");
    let mut wanted = (tokens / MIN_LANE_ROWS).clamp(1, lanes.max(1)).max(tokens.div_ceil(rows.max(1)));
    let per_lane = loop {
        let per_lane = if wanted == 1 { tokens } else { tokens.div_ceil(wanted).next_multiple_of(PAGE_ROWS) };
        if per_lane <= rows || wanted >= lanes {
            break per_lane;
        }
        wanted += 1;
    };
    ensure!(per_lane <= rows, "prefill lane of {per_lane} tokens exceeds {rows} rows");
    Ok((tokens.div_ceil(per_lane), per_lane))
}
const HC: usize = 4;

/// The longest chunk a lone prefill runs as one step: pipelined Spark lanes ([`prefill_lane_plan`])
/// cut a chunk into two lanes from twice `MIN_LANE_ROWS` rows; a serial prefill takes `rows`.
fn single_pass_rows(pipelined: bool, lanes: usize, rows: usize) -> usize {
    if pipelined && lanes > 1 { (2 * MIN_LANE_ROWS - 1).min(rows) } else { rows }
}

/// What a packed pass's `on_logits` returns for one sequence ([`GlmfEngine::prefill_packed`]):
/// `Err` is a failure the pass shares (the engine's), which ends it for every sequence;
/// `Ok(Err)` is that sequence's own (its grammar, its sampling), which fails it alone.
pub(crate) type PackedOutcome = Result<Result<()>>;

/// A packed pass once its step has run: the step wrote every sequence's KDA state and pages, so
/// every placement advances first, then each sequence's `head(i)` logits go to `on_logits` in
/// order. A sequence's own failure does not stop the others'; it is returned among the outcomes,
/// and its caller releases or resets that placement.
fn packed_tail<L>(sequences: &mut [(&mut GlmfPlacement, &[u32])], mut head: impl FnMut(usize) -> Result<L>,
    on_logits: &mut dyn FnMut(usize, L) -> PackedOutcome) -> Result<Vec<Result<()>>> {
    for (placement, tokens) in sequences.iter_mut() {
        placement.len += tokens.len();
        placement.kda_len = placement.len;
    }
    (0..sequences.len()).map(|i| on_logits(i, head(i)?)).collect()
}

/// CUTEAFD_GLMF_PREFILL_LANES: (lanes on: unset or not `1`, lanes with a `--layers` subset: `subset`).
fn lane_setting() -> (bool, bool) {
    let setting = std::env::var("CUTEAFD_GLMF_PREFILL_LANES");
    (setting.as_ref().map_or(true, |v| v != "1"), setting.as_ref().is_ok_and(|v| v == "subset"))
}

/// Whether an engine over `layers` of `total` with `experts` prefills in `lanes` Spark lanes (as
/// [`GlmfEngine::prefill_capacity`] finds once it exists): lanes on, every layer resident (or a
/// subset with `subset`) and a Spark transport per lane.
pub(crate) fn prefill_pipelines(layers: usize, total: usize, experts: Option<&Experts<'_>>, lanes: usize) -> bool {
    let (on, subset) = lane_setting();
    on && (layers == total || subset)
        && matches!(experts, Some(Experts::Spark { transports, .. }) if transports.borrow().len() >= lanes)
}

/// Routed experts on this GPU from the TP1 package. All layers stay resident
/// unless an explicit diagnostic paging window was requested.
pub(crate) struct LocalExperts<'a> {
    pub library: &'a NativeLibrary,
    pub tensors: &'a Fp8ExpertTensors,
    pub experts: RefCell<Fp8Experts<'a>>,
    /// Explicit diagnostic paging; absent for the fully resident serving path.
    pub window: Option<usize>,
    pub loads: RefCell<usize>,
}

impl LocalExperts<'_> {
    fn index_of(&self, layer: usize) -> Result<usize> {
        let mut experts = self.experts.borrow_mut();
        if let Ok(index) = experts.index_of(layer) {
            return Ok(index);
        }
        let window = self.window.context("local expert layer missing from the admitted resident set")?;
        if experts.layers.len() >= window {
            experts.layers.remove(0);
        }
        let started = std::time::Instant::now();
        experts.layers.push(Fp8Layer::load(self.library, self.tensors, layer, 1, 0)?);
        *self.loads.borrow_mut() += 1;
        tracing::debug!(layer, elapsed_ms = started.elapsed().as_millis() as u64, "FP8 expert layer loaded");
        Ok(experts.layers.len() - 1)
    }
}

/// Routed experts on this GPU from the EXL3 checkpoint (the coordinator's
/// `exl3-glmf-k<tiers>/rtx-tp1` package): a window of resident layers,
/// reloaded as the step walks the layers. The package reads FP8 K32 wire
/// rows (as the Sparks do) and its reducer adds the shared expert.
pub(crate) struct LocalExl3<'a> {
    pub library: &'a NativeLibrary,
    pub native_lib: std::path::PathBuf,
    pub catalog: &'a cuteafd_loader::OfficialV41Catalog,
    pub resident: RefCell<Option<(std::ops::Range<usize>, crate::families::deepseek_v4::local::LocalExperts<'a>)>>,
    pub window: usize,
    /// Layers past the engine's last one never load.
    pub layers: usize,
    pub max_rows: usize,
    pub budget: usize,
    pub loads: RefCell<usize>,
}

impl LocalExl3<'_> {
    /// Makes `layer` resident (with the next `window - 1` layers).
    fn ensure(&self, layer: usize, stream: *mut c_void) -> Result<()> {
        if self.resident.borrow().as_ref().is_some_and(|(range, _)| range.contains(&layer)) {
            return Ok(());
        }
        // SAFETY: the engine owns this stream; the old window's launches drain first.
        unsafe { self.library.cuda_stream_synchronize(stream)? };
        *self.resident.borrow_mut() = None;
        let range = layer..(layer + self.window).min(self.layers);
        let started = std::time::Instant::now();
        let local = crate::families::deepseek_v4::local::LocalExperts::load_range(self.library, &self.native_lib, self.catalog, 0,
            range.clone(), self.max_rows, self.budget, stream)?
            .context("no coordinator EXL3 package for this checkpoint (build glmf:exl3-k<tiers>)")?;
        let range = layer..layer + local.layers();
        ensure!(range.contains(&layer), "EXL3 expert layer {layer} does not fit the budget");
        *self.loads.borrow_mut() += range.len();
        tracing::debug!(?range, elapsed_ms = started.elapsed().as_millis() as u64, "EXL3 expert window resident");
        *self.resident.borrow_mut() = Some((range, local));
        Ok(())
    }
}

/// Where the routed experts run.
pub(crate) enum Experts<'a> {
    Local(LocalExperts<'a>),
    LocalExl3(LocalExl3<'a>),
    /// Spark ranks serving the routed experts over RoCE (one BF16 partial per
    /// rank); one transport per prefill lane (decode uses the first).
    Spark { transports: RefCell<Vec<SparkLink<'a>>>, runtime: tokio::runtime::Runtime },
    /// Profiling only: the router, the wire rows and the shared expert run,
    /// the routed experts contribute nothing (the coordinator's own work).
    Skip,
}

/// Per-program GPU times (CUTEAFD_GLMF_PROFILE_OPS=1): timing events around
/// every launch, read back by [`GlmfEngine::op_profile`].
#[derive(Default)]
pub(crate) struct OpTimes {
    pending: Vec<(String, *mut c_void, *mut c_void)>,
    pool: Vec<*mut c_void>,
    pub totals: std::collections::BTreeMap<String, (f64, usize)>,
}

/// Borrowed workspaces of the head split's second GPU.
enum PeerWorkspaces<'e, 'a> {
    One(std::cell::Ref<'e, Option<Workspace<'a>>>),
    Lanes(std::cell::Ref<'e, Vec<Workspace<'a>>>),
}

impl<'a> PeerWorkspaces<'_, 'a> {
    fn get(&self, lane: usize) -> Result<&Workspace<'a>> {
        match self {
            Self::One(w) => w.as_ref().context("peer workspace"),
            Self::Lanes(w) => w.get(lane).context("peer lane workspace"),
        }
    }
}

/// Host tables of one step.
#[derive(Default)]
struct StepTables {
    decode: bool,
    eager: bool,
    /// Ordinary rows before startup graph padding; zero for all-masked warm-up.
    real_rows: usize,
    positions: Vec<i64>,
    /// MLA latent record slot per row (also the row's token-key slot).
    kv_slots: Vec<i64>,
    /// KDA state slot per row.
    kda_slots: Vec<i32>,
    /// First step row of each row's sequence.
    seq_first: Vec<i32>,
    /// Pool index-cache slot of the pool a row completes, else -1.
    pool_slots: Vec<i64>,
    /// Complete pools each row sees.
    cache_lengths: Vec<i32>,
    /// Record pages and pool pages: one shared table (prefill) or one padded row per step row.
    page_table: Vec<i32>,
    pool_table: Vec<i32>,
    page_stride: usize,
    pool_stride: usize,
    /// Pool-table columns the top-k reads.
    pool_width: usize,
    /// Whether any row sees more than 2051 tokens (the pool top-k runs).
    long: bool,
    /// A speculative verify: KDA state stays, replay rows are recorded.
    spec: bool,
    /// A packed prefill's sequences ([`GlmfEngine::prefill_packed`]): every per-sequence program
    /// runs once per segment over its rows, with its own tables. Empty: one sequence (or a decode
    /// step, whose programs take every row's sequence from the tables).
    segments: Vec<packing::Segment>,
}

/// Rows a per-sequence program runs over: one packed sequence's, or the whole step's.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Span {
    first: usize,
    rows: usize,
}

impl Span {
    fn rows(&self) -> Scalar {
        Scalar::I32(self.rows as i32)
    }
}

impl StepTables {
    /// The rows each per-sequence program runs over: every packed sequence's, or the whole step.
    fn spans(&self) -> Vec<Span> {
        if self.segments.is_empty() {
            vec![Span { first: 0, rows: self.kv_slots.len() }]
        } else {
            self.segments.iter().map(|s| Span { first: s.first, rows: s.rows }).collect()
        }
    }
}

/// One sequence's rows for the DSA indexer, its pool top-k and the selection
/// ([`GlmfEngine::index_parts`]).
struct IndexPart {
    first: usize,
    rows: usize,
    /// A row sees more than the dense context: the pool top-k runs.
    long: bool,
    /// The sequence's record-page table (one shared row in a prefill step).
    page_table: *mut c_void,
    /// A packed sequence's pool-page table and the columns its top-k reads; None: the step's
    /// own (`w.pool_table`, `tables.pool_width`), which only the long pool top-k reads.
    pool: Option<(*mut c_void, usize)>,
}

/// Row `first` of `buffer`'s rows of `row_bytes` bytes (the buffer itself for row 0).
fn row_at(buffer: &Dev<'_>, first: usize, row_bytes: usize) -> *mut c_void {
    buffer.buffer.ptr.wrapping_byte_add(first * row_bytes)
}

/// DSA indexer heads: the index query and head-weight rows (`glmf_temporary_bytes`'s q_fp8 and
/// head_weights).
const INDEX_HEADS: usize = 32;

/// Tokens per DSA index pool, and pools per pool-cache page.
pub(crate) const KPOOL: usize = 4;
const POOL_PAGE_TOKENS: usize = KPOOL * PAGE_ROWS;

/// MLA pages per allocation unit: a unit is four consecutive 64-row MLA pages (256 tokens) and
/// the pool-cache page of the same index (64 pools of 4 tokens), so one refcounted unit index
/// names every paged byte of 256 positions (the prefix cache's page).
pub(crate) const UNIT_PAGES: usize = KPOOL;
pub(crate) const UNIT_ROWS: usize = POOL_PAGE_TOKENS;

/// A sequence's allocation units (and the MLA and pool pages they expand to), its KDA state
/// slot, its length and the rows its KDA state holds.
#[derive(Debug, Clone)]
pub(crate) struct GlmfPlacement {
    pub units: Vec<u32>,
    pub pages: Vec<i32>,
    pub pool_pages: Vec<i32>,
    pub slot: i32,
    pub len: usize,
    /// Rows the KDA recurrent and conv state has consumed: `len` after a prefill or a plain
    /// verify; a speculative verify leaves it until the caller commits the kept rows.
    pub kda_len: usize,
}

impl GlmfPlacement {
    /// A fresh sequence over `units` with KDA state slot `slot`.
    pub fn new(units: Vec<u32>, slot: i32) -> Self {
        let pages = units.iter().flat_map(|&u| (0..UNIT_PAGES as i32).map(move |i| u as i32 * UNIT_PAGES as i32 + i))
            .collect();
        let pool_pages = units.iter().map(|&u| u as i32).collect();
        Self { units, pages, pool_pages, slot, len: 0, kda_len: 0 }
    }

    pub fn record(&self, position: usize) -> Result<i64> {
        let page = *self.pages.get(position / PAGE_ROWS).context("position past the sequence's pages")?;
        Ok(i64::from(page) * PAGE_ROWS as i64 + (position % PAGE_ROWS) as i64)
    }

    /// Pool-cache slot of the pool `position` completes, or -1.
    pub fn pool_slot(&self, position: usize) -> Result<i64> {
        if position % KPOOL != KPOOL - 1 {
            return Ok(-1);
        }
        let page = *self.pool_pages.get(position / POOL_PAGE_TOKENS).context("position past the pool pages")?;
        Ok(i64::from(page) * PAGE_ROWS as i64 + ((position / KPOOL) % PAGE_ROWS) as i64)
    }
}

/// Refcounted allocation units and free KDA state slots (goldens and benches; serving takes
/// its units from the prefix cache's pool).
pub(crate) struct Allocator {
    units: cuteafd_engine::prefix::RefPagePool,
    slots: Vec<i32>,
}

impl Allocator {
    /// Over an engine of `pages` MLA pages (a whole number of units) and `slots` KDA slots.
    pub fn new(pages: usize, slots: usize) -> Self {
        Self::with_reserved(pages, slots, 0)
    }

    /// [`Self::new`] that never hands out the first `reserved` units (pool-page prefix marks keep
    /// unit 0 zeroed: see `GLMF_POOL_MARK_RESERVED_UNITS`).
    pub fn with_reserved(pages: usize, slots: usize, reserved: usize) -> Self {
        Self { units: cuteafd_engine::prefix::RefPagePool::with_reserved(pages / UNIT_PAGES, UNIT_ROWS, reserved),
            slots: (0..slots as i32).rev().collect() }
    }

    /// Reserves every unit a sequence of up to `capacity` tokens needs and a state slot (the
    /// engine zeroes the slot and maps the pool pages before the first step).
    pub fn admit(&mut self, capacity: usize) -> Result<GlmfPlacement> {
        let slot = self.slots.pop().context("KDA state slots exhausted")?;
        match self.units.alloc(self.units.pages_for(capacity)) {
            Ok(units) => Ok(GlmfPlacement::new(units, slot)),
            Err(error) => {
                self.slots.push(slot);
                Err(error).context("cache pages exhausted")
            }
        }
    }

    /// A second sequence starting as `source`'s first `len` rows: full units shared, the
    /// partial tail unit copied by the caller (the returned copy), and its own KDA slot.
    pub fn fork(&mut self, source: &GlmfPlacement, len: usize, capacity: usize)
        -> Result<(GlmfPlacement, Option<cuteafd_engine::prefix::TailCopy>)> {
        let slot = self.slots.pop().context("KDA state slots exhausted")?;
        match self.units.fork(&source.units, len, self.units.pages_for(capacity)) {
            Ok(fork) => Ok((GlmfPlacement::new(fork.pages, slot), fork.copy)),
            Err(error) => {
                self.slots.push(slot);
                Err(error).context("cache pages exhausted")
            }
        }
    }

    /// `n` units for a pool-page prefix mark, ascending as the prefix cache hands them out.
    pub fn take_units(&mut self, n: usize) -> Result<Vec<u32>> {
        let mut units = self.units.alloc(n).context("cache pages exhausted")?;
        units.sort_unstable();
        Ok(units)
    }

    pub fn release_units(&mut self, units: &[u32]) {
        self.units.release(units);
    }

    /// A spare KDA state slot (a speculative verify's backup).
    pub fn spare_slot(&mut self) -> Result<i32> {
        self.slots.pop().context("KDA state slots exhausted")
    }

    pub fn release_slot(&mut self, slot: i32) {
        self.slots.push(slot);
    }

    pub fn release(&mut self, placement: GlmfPlacement) {
        self.units.release(&placement.units);
        self.slots.push(placement.slot);
    }
}

/// Extra per-GPU storage of `lanes` prefill lanes: all peer receive slots and every retained
/// workspace's delta.
pub(crate) fn fp32_partial_reserve(lanes: usize, prefill_rows: usize, hidden: usize) -> u64 {
    partial_reserve(lanes, prefill_rows, hidden, 4)
}

pub(crate) fn partial_reserve(lanes: usize, prefill_rows: usize, hidden: usize, bytes: usize) -> u64 {
    (((4 * lanes + lanes + 1) * prefill_rows.max(DECODE_ROWS) + DECODE_ROWS)
        * hidden * bytes.saturating_sub(2)) as u64
}

/// Workspace deltas are already in the exact union; only the enlarged exchange slots of `lanes`
/// prefill lanes remain.
pub(crate) fn partial_exchange_reserve(lanes: usize, prefill_rows: usize, hidden: usize, bytes: usize) -> u64 {
    (4 * lanes * prefill_rows.max(DECODE_ROWS) * hidden * bytes.saturating_sub(2)) as u64
}

/// Two additional parity slots per lane hold normalized heads until the peer consumes them.
pub(crate) fn output_shard_reserve(lanes: usize, prefill_rows: usize, hidden: usize) -> u64 {
    (2 * lanes * prefill_rows.max(DECODE_ROWS) * hidden * 2) as u64
}

/// A step workspace: one lane's own buffers over the temporaries of one attention call. Prefill
/// lanes run on one stream and use the temporaries only inside one unit's attention call (the
/// lane's x, delta, shared-expert, route and wire rows carry everything across its expert
/// exchange), so a GPU's lanes share one set of temporaries; decode keeps its own, since its
/// graphs hold its pointers. Sizes: `cuteafd_loader::serving_capacity::glmf_lane_bytes`.
struct Workspace<'a> {
    rows: usize,
    /// Zero rows: rank 1's partial of a dense MLP rank 0 runs whole (ModelOpt NVFP4); rank 1 of a
    /// head split only.
    zero: Option<Dev<'a>>,
    /// The sum of a head split's two partials (the peer add writes a disjoint buffer); head split only.
    sum: Option<Dev<'a>>,
    streams: [Dev<'a>; 2],
    post: Dev<'a>,
    comb: Dev<'a>,
    x: Dev<'a>,
    delta: Dev<'a>,
    shared: Dev<'a>,
    /// The routed experts' partial: local experts only.
    routed: Option<Dev<'a>>,
    positions: Dev<'a>,
    kv_slots: Dev<'a>,
    kda_slots: Dev<'a>,
    seq_first: Dev<'a>,
    pool_slots: Dev<'a>,
    cache_lengths: Dev<'a>,
    page_table: Dev<'a>,
    pool_table: Dev<'a>,
    /// The step's token ids (U32, gathered from the device embedding table).
    ids: Dev<'a>,
    /// Greedy selection of the logits rows inside the decode graph: U32 ids, then U32 statuses.
    select: Dev<'a>,
    router_logits: Dev<'a>,
    route_ids: Dev<'a>,
    route_weights: Dev<'a>,
    wire: Dev<'a>,
    router_host: RefCell<HostAllocation<'a>>,
    /// Shared by every prefill lane of this GPU (decode: its own).
    temps: Rc<Temporaries<'a>>,
}

/// The temporaries of one attention call: produced and consumed inside it, on the one stream
/// that orders every lane's programs. Sizes: `cuteafd_loader::serving_capacity::glmf_temporary_bytes`.
struct Temporaries<'a> {
    rows: usize,
    query: Dev<'a>,
    q_resid: Dev<'a>,
    latent: Dev<'a>,
    q_fp8: Dev<'a>,
    head_weights: Dev<'a>,
    pools: Dev<'a>,
    indices: Dev<'a>,
    lengths: Dev<'a>,
    scratch: Dev<'a>,
    /// The pool top-k's scratch: zeroed once, restored by every launch.
    topk_scratch: Dev<'a>,
    logits: Dev<'a>,
    /// The LM head (rank 0 only).
    head: Option<VocabularyHead<'a>>,
    _head_workspace: Dev<'a>,
}

impl<'a> std::ops::Deref for Workspace<'a> {
    type Target = Temporaries<'a>;

    /// A lane's view of its GPU's temporaries (`w.scratch`, `w.query`, ...).
    fn deref(&self) -> &Temporaries<'a> {
        &self.temps
    }
}

impl Workspace<'_> {
    /// The head split's sum of the two partials.
    fn sum_ptr(&self) -> Result<*mut c_void> {
        Ok(self.sum.as_ref().context("the partials' sum exists only under a head split")?.buffer.ptr)
    }

    /// Rank 1's zero rows.
    fn zero_ptr(&self) -> Result<*mut c_void> {
        Ok(self.zero.as_ref().context("zero rows exist only on a head split's second GPU")?.buffer.ptr)
    }

    /// The local routed experts' partial.
    fn routed_ptr(&self) -> Result<*mut c_void> {
        Ok(self.routed.as_ref().context("the routed partial exists only with local experts")?.buffer.ptr)
    }
}

/// What a GPU's step workspaces depend on: the engine's configuration and where its experts run.
/// Every buffer's size comes from `cuteafd_loader::serving_capacity::glmf_*`, the arithmetic the
/// planner charges.
pub(crate) struct StepPlan<'p, 'a> {
    library: &'a NativeLibrary,
    programs: &'a Programs<'a>,
    cfg: &'p GlmNextConfig,
    scratch: GlmfScratchOptions,
    /// Rank 0's shape (rank 1 is not the lead and runs no experts).
    shape: GlmfStepShape,
    /// Rows of the decode workspace (`--decode-rows`): it runs the `_m64` programs, and with 128 the
    /// wide `_m128` ones too.
    decode_rows: usize,
    /// Bytes of KDA replay records rank 0's prefill scratch holds (`--replay-records shared`): the
    /// scratch is at least this large. 0 with records of their own.
    shared_records: u64,
}

/// The engine settings a step plan depends on besides its layers and experts.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct StepSettings {
    pub kda_fp32_partials: bool,
    pub kda_output_shard: bool,
    pub kda_prefill_expanded: bool,
    pub full_prefill_logits: bool,
    /// Sizes the page tables (`glmf_table_pages`).
    pub max_context: usize,
    /// The DSA index cache: the compact one runs the `index_producer_c` programs, whose scratch
    /// (the producer's, then the step's key | gate rows) the steps' scratch holds.
    pub index_cache: IndexCache,
    /// The KDA recurrent state's storage: which KDA programs the steps launch (`--kda-state`).
    pub kda_state: KdaState,
    /// Where the KDA replay records live.
    pub replay_records: ReplayRecords,
    /// The most rows of one decode or verify step (`--decode-rows`: [`DECODE_ROWS`], or
    /// [`WIDE_DECODE_ROWS`] with the wide programs).
    pub decode_rows: usize,
}

impl<'p, 'a> StepPlan<'p, 'a> {
    /// The plan of an engine over `layers` with `experts` and `settings` (rank 0's shape).
    pub(crate) fn new(library: &'a NativeLibrary, programs: &'a Programs<'a>, cfg: &'p GlmNextConfig,
        layers: &[GlmfLayer<'_>], experts: Option<&Experts<'_>>, settings: StepSettings) -> Self {
        let split = layers.first().is_some_and(|l| l.split);
        let (table_pages, table_pool_pages) = glmf_table_pages(settings.max_context as u64);
        // As `GlmfEngine::new` resolves it: without an MLA layer there is no index cache.
        let index_compact = settings.index_cache == IndexCache::Compact
            && layers.iter().any(|layer| layer.attention == GlmNextAttention::Mla);
        // Every KDA layer's record of the decode rows, over this GPU's KDA heads (as `Caches::new`).
        let kda_heads = cfg.kda_heads / if split { 2 } else { 1 };
        let kda_layers = layers.iter().filter(|layer| layer.attention == GlmNextAttention::Kda).count();
        let shared_records = if settings.replay_records == ReplayRecords::Shared {
            (kda_layers * replay_bytes(kda_heads, 3 * kda_heads * cfg.kda_head_dim, settings.decode_rows)) as u64
        } else { 0 };
        Self {
            library,
            programs,
            cfg,
            scratch: GlmfScratchOptions { split, kda_w8: layers.iter().any(|layer| layer.has("w_in_fp8")),
                kda_fp32_partials: settings.kda_fp32_partials, kda_output_shard: settings.kda_output_shard,
                kda_prefill_expanded: settings.kda_prefill_expanded, index_compact,
                kda_state: settings.kda_state.into() },
            shape: GlmfStepShape {
                lead: true,
                split,
                local_experts: matches!(experts, Some(Experts::Local(_))),
                spark: matches!(experts, Some(Experts::Spark { .. })),
                partial_bytes: if settings.kda_fp32_partials { 4 } else { 2 },
                output_shard: settings.kda_output_shard,
                full_prefill_logits: settings.full_prefill_logits,
                table_pages,
                table_pool_pages,
            },
            decode_rows: settings.decode_rows,
            shared_records,
        }
    }

    /// Routed experts as the plan's shape will have them (a plan made before its experts exist:
    /// the admission's).
    pub(crate) fn with_experts(mut self, local: bool, spark: bool) -> Self {
        self.shape.local_experts = local;
        self.shape.spark = spark;
        self
    }

    /// Device bytes of rank `rank`'s step workspaces: the decode workspace of the plan's decode rows,
    /// and `lanes` prefill lanes of `prefill_rows` rows over their one set of temporaries (what
    /// `decode_workspace_of` and `prefill_lanes_of` allocate).
    pub(crate) fn workspace_bytes(&self, rank: usize, prefill_rows: usize, lanes: usize) -> Result<u64> {
        let lookup = |name: &str| self.programs.spec(name).ok()
            .map(|spec| spec.scratch.get("scratch").copied().unwrap_or(0));
        let decode_scratch = glmf_step_scratch(lookup, self.cfg, self.scratch, self.decode_rows as u64, true)?;
        let mut prefill_scratch = glmf_step_scratch(lookup, self.cfg, self.scratch, prefill_rows as u64, false)?;
        if rank == 0 {
            // The prefill scratch also holds the replay records (`--replay-records shared`).
            prefill_scratch.programs = prefill_scratch.programs.max(self.shared_records);
        }
        Ok(glmf_step_workspaces(self.cfg, lanes, prefill_rows as u64, self.decode_rows as u64, &self.shape(rank),
            decode_scratch, prefill_scratch).device_bytes())
    }

    fn key(&self) -> (GlmfScratchOptions, GlmfStepShape, usize, u64) {
        (self.scratch, self.shape, self.decode_rows, self.shared_records)
    }
}

/// A GPU's step workspaces allocated before its engine (eager start-up): the decode workspace and
/// every prefill lane, so the KV pool is sized from the memory they leave.
pub(crate) struct StepWorkspaces<'a> {
    key: (GlmfScratchOptions, GlmfStepShape, usize, u64),
    rows: usize,
    decode: Workspace<'a>,
    lanes: Vec<Workspace<'a>>,
}

/// Rank 0's prefill lanes' shared temporaries, whose scratch holds the KDA replay records with
/// `--replay-records shared` (the caches keep it alive with them).
pub(crate) struct PrefillScratch<'a>(Rc<Temporaries<'a>>);

impl<'a> StepWorkspaces<'a> {
    /// The decode workspace and `lanes` prefill lanes of `rows` rows of `plan`, on the current device.
    pub(crate) fn allocate(plan: &StepPlan<'_, 'a>, rows: usize, lanes: usize) -> Result<Self> {
        let decode = plan.lane(0, plan.decode_rows, true, Rc::new(plan.temporaries(0, plan.decode_rows, true)?))?;
        let temps = Rc::new(plan.temporaries(0, rows, false)?);
        let lanes = (0..lanes).map(|_| plan.lane(0, rows, false, temps.clone())).collect::<Result<_>>()?;
        Ok(Self { key: plan.key(), rows, decode, lanes })
    }

    /// The prefill lanes' shared temporaries (their scratch).
    pub(crate) fn prefill_scratch(&self) -> Option<PrefillScratch<'a>> {
        self.lanes.first().map(|lane| PrefillScratch(lane.temps.clone()))
    }
}

impl<'a> StepPlan<'_, 'a> {
    fn shape(&self, rank: usize) -> GlmfStepShape {
        if rank == 0 { self.shape } else { GlmfStepShape { lead: false, local_experts: false, spark: false, ..self.shape } }
    }

    fn alloc(&self, bytes: u64) -> Result<Dev<'a>> {
        DeviceAllocation::new(self.library, usize::try_from(bytes)?.max(256))
    }

    fn zeroed(&self, bytes: u64) -> Result<Dev<'a>> {
        let allocation = self.alloc(bytes)?;
        self.library.cuda_zero_bytes(allocation.buffer, allocation.buffer.bytes)?;
        Ok(allocation)
    }

    /// Rank `rank`'s temporaries for steps of up to `rows` rows, on the current device.
    fn temporaries(&self, rank: usize, rows: usize, decode: bool) -> Result<Temporaries<'a>> {
        let _memory_scope = cuteafd_ffi::memory_ledger::scope("workspace");
        let shape = self.shape(rank);
        let lookup = |name: &str| self.programs.spec(name).ok()
            .map(|spec| spec.scratch.get("scratch").copied().unwrap_or(0));
        let mut scratch = glmf_step_scratch(lookup, self.cfg, self.scratch, rows as u64, decode)?;
        if rank == 0 && !decode {
            // The prefill scratch also holds the replay records (`--replay-records shared`).
            scratch.programs = scratch.programs.max(self.shared_records);
        }
        let bytes = glmf_temporary_bytes(self.cfg, rows as u64, decode, &shape, scratch);
        let head_workspace = self.alloc(bytes.head_workspace)?;
        Ok(Temporaries {
            rows,
            query: self.alloc(bytes.query)?,
            q_resid: self.alloc(bytes.q_resid)?,
            latent: self.alloc(bytes.latent)?,
            q_fp8: self.alloc(bytes.q_fp8)?,
            head_weights: self.alloc(bytes.head_weights)?,
            pools: self.alloc(bytes.pools)?,
            indices: self.alloc(bytes.indices)?,
            lengths: self.alloc(bytes.lengths)?,
            scratch: self.alloc(bytes.scratch)?,
            topk_scratch: self.zeroed(bytes.topk_scratch)?,
            logits: self.alloc(bytes.logits)?,
            // SAFETY: the workspace buffer lives in the same struct and drops after the head.
            head: if shape.lead {
                Some(unsafe { self.library.vocabulary_head_rows(head_workspace.buffer.ptr, self.cfg.hidden as u32,
                    rows as u32, self.cfg.vocab_size as u32)? })
            } else {
                None
            },
            _head_workspace: head_workspace,
        })
    }

    /// Rank `rank`'s lane buffers for steps of up to `rows` rows over `temps`, on the current device.
    fn lane(&self, rank: usize, rows: usize, decode: bool, temps: Rc<Temporaries<'a>>) -> Result<Workspace<'a>> {
        let _memory_scope = cuteafd_ffi::memory_ledger::scope("workspace");
        ensure!(rows <= temps.rows, "a lane of {rows} rows over temporaries of {}", temps.rows);
        let bytes = glmf_lane_bytes(self.cfg, rows as u64, decode, &self.shape(rank));
        Ok(Workspace {
            rows,
            zero: bytes.zero.map(|b| self.zeroed(b)).transpose()?,
            sum: bytes.sum.map(|b| self.alloc(b)).transpose()?,
            streams: [self.alloc(bytes.streams)?, self.alloc(bytes.streams)?],
            post: self.alloc(bytes.post)?,
            comb: self.alloc(bytes.comb)?,
            x: self.alloc(bytes.x)?,
            delta: self.alloc(bytes.delta)?,
            shared: self.alloc(bytes.shared)?,
            routed: bytes.routed.map(|b| self.alloc(b)).transpose()?,
            positions: self.alloc(bytes.positions)?,
            kv_slots: self.alloc(bytes.kv_slots)?,
            kda_slots: self.alloc(bytes.kda_slots)?,
            seq_first: self.alloc(bytes.seq_first)?,
            pool_slots: self.alloc(bytes.pool_slots)?,
            cache_lengths: self.alloc(bytes.cache_lengths)?,
            page_table: self.alloc(bytes.page_table)?,
            pool_table: self.alloc(bytes.pool_table)?,
            ids: self.alloc(bytes.ids)?,
            select: self.alloc(bytes.select)?,
            router_logits: self.alloc(bytes.router_logits)?,
            route_ids: self.alloc(bytes.route_ids)?,
            route_weights: self.alloc(bytes.route_weights)?,
            wire: self.alloc(bytes.wire)?,
            router_host: RefCell::new(HostAllocation::new(self.library, usize::try_from(bytes.router_host)?)?),
            temps,
        })
    }
}

/// ModelOpt NVFP4 dense MLPs (nvidia/GLM-5.3-Flash-NVFP4 layers 0-2) on the
/// `fp8-glmfdense-nvfp4` package: one always-selected expert (ids 0, weight 1),
/// so the route sum is the MLP output exactly.
pub(crate) struct DenseNvfp4<'a> {
    pub module: cuteafd_ffi::fp8_moe::Fp8MoeModule,
    pub scratch: crate::shared::memory::DeviceAllocation<'a>,
    pub ids: crate::shared::memory::DeviceAllocation<'a>,
    pub weights: crate::shared::memory::DeviceAllocation<'a>,
}

impl<'a> DenseNvfp4<'a> {
    /// Loads the package at `directory` with scratch, ids and weights for `rows` rows.
    pub fn load(library: &'a NativeLibrary, directory: &std::path::Path, cfg: &GlmNextConfig, rows: usize)
        -> Result<Self> {
        let _memory_scope = cuteafd_ffi::memory_ledger::scope("weights/dense-nvfp4");
        // SAFETY: a trusted package for the current device; the engine drains its
        // stream before dropping it.
        let module = unsafe { cuteafd_ffi::fp8_moe::Fp8MoeModule::load(directory) }
            .with_context(|| format!("dense NVFP4 package {} (glmfdense:nvfp4)", directory.display()))?;
        let info = module.info().clone();
        ensure!(info.experts == 1 && info.topk == 1 && info.hidden == cfg.hidden
            && info.intermediate == cfg.dense_intermediate && !info.wire_input
            && matches!(info.weights, cuteafd_ffi::fp8_moe::Fp8MoeWeights::Nvfp4 { .. }),
            "{} ({info:?}) is not the dense NVFP4 MLP package", directory.display());
        let top = info.capacity_for(rows).with_context(|| format!("dense NVFP4 package has no capacity for {rows} rows"))?;
        let scratch = crate::shared::memory::DeviceAllocation::new(library, module.scratch_bytes(top)?.max(256))?;
        let ids = crate::shared::memory::DeviceAllocation::new(library, rows * 4)?;
        library.copy_h2d(ids.buffer, &vec![0u8; rows * 4])?;
        let weights = crate::shared::memory::DeviceAllocation::new(library, rows * 4)?;
        let ones: Vec<u8> = (0..rows).flat_map(|_| 1f32.to_le_bytes()).collect();
        library.copy_h2d(weights.buffer, &ones)?;
        Ok(Self { module, scratch, ids, weights })
    }
}

/// One MLA layer's DSA index cache: the per-token keys | gates (BF16 [record slots, 256];
/// `--index-cache keys` only) and the FP8 pool-key pages.
struct IndexLayer<'a> {
    keys: Option<Dev<'a>>,
    pools: Dev<'a>,
}

/// One MLA layer's paged buffers: latent records (528 B per row) and, with `--index-cache
/// keys`, the indexer's token keys (512 B per row), both in 64-row MLA pages, then the
/// pool-key cache (8448 B per pool page).
#[derive(Debug, Clone, Copy)]
pub(crate) struct PagedLayer {
    pub records: cuteafd_ffi::CuteafdDeviceBuffer,
    pub keys: Option<cuteafd_ffi::CuteafdDeviceBuffer>,
    pub pools: cuteafd_ffi::CuteafdDeviceBuffer,
}

/// One GPU's caches. Per MLA layer (None for KDA): the latent record pool and its DSA index
/// cache. Every KDA layer's pools back to back: FP32 or BF16 recurrent state `[layers, slots,
/// heads, 128, 128]`, BF16 conv state `[layers, slots, 3, 3D]` and the speculative replay records
/// (`replay_bytes` of the decode rows per layer), over this GPU's KDA heads. With the compact index
/// cache, every MLA layer's sequence tails `[MLA layers, slots, TAIL_BYTES]` and speculative
/// key | gate records `[MLA layers, decode rows, 256]` (BF16). A speculative step records at its
/// programs' capacity (`replay_rows_of`): an `_m64` step packs 64-row records layer by layer from
/// the start, as the `_m64` commits read them, and an `_m128` step 128-row ones. The
/// `glmf_kda_commit` tables (slot, first row, kept rows per sequence) and the logical page of each
/// pool-cache page within its sequence.
struct Caches<'a> {
    kv: Vec<Option<Dev<'a>>>,
    index: Vec<Option<IndexLayer<'a>>>,
    kda_state: Dev<'a>,
    kda_conv: Dev<'a>,
    /// The replay records: their own allocation, or a region of rank 0's prefill scratch.
    kda_replay: cuteafd_ffi::CuteafdDeviceBuffer,
    _kda_replay_owner: RecordsOwner<'a>,
    /// Compact index cache: (tails, speculative key | gate records).
    index_tails: Option<(Dev<'a>, Dev<'a>)>,
    commit_tables: Dev<'a>,
    pool_logical: Dev<'a>,
    /// KDA heads of this GPU.
    kda_heads: usize,
}

/// What keeps the replay records' memory: their own allocation, or the prefill temporaries
/// whose scratch holds them.
enum RecordsOwner<'a> {
    Own { _records: Dev<'a> },
    Shared { _temporaries: Rc<Temporaries<'a>> },
}

impl<'a> Caches<'a> {
    /// Zeroed caches on the current device for `layers` with `kda_heads` KDA heads, the replay
    /// records and commit tables for steps of up to `decode_rows` rows. The replay records take their
    /// own allocation, or the start of `records`' scratch (written by every speculative verify before
    /// its commit reads them, so they start unzeroed).
    #[allow(clippy::too_many_arguments)]
    fn new(library: &'a NativeLibrary, cfg: &GlmNextConfig, layers: &[GlmfLayer<'_>], pages: usize, pool_pages: usize,
        slots: usize, kda_heads: usize, index_cache: IndexCache, kda_state: KdaState, decode_rows: usize,
        records: Option<PrefillScratch<'a>>) -> Result<Self> {
        let zeroed = |bytes: usize| -> Result<Dev<'a>> {
            let allocation = DeviceAllocation::new(library, bytes.max(256))?;
            library.cuda_zero_bytes(allocation.buffer, allocation.buffer.bytes)?;
            Ok(allocation)
        };
        let (state, conv, replay) = kda_layer_bytes(cfg, kda_heads, kda_state, decode_rows);
        let keys = index_cache == IndexCache::Keys;
        let (mut kv, mut index, mut kda_layers, mut mla_layers) = (Vec::new(), Vec::new(), 0, 0);
        for layer in layers {
            match layer.attention {
                GlmNextAttention::Mla => {
                    kv.push(Some(zeroed(pages * PAGE_ROWS * RECORD_BYTES)?));
                    index.push(Some(IndexLayer {
                        keys: if keys { Some(zeroed(pages * PAGE_ROWS * KEY_BYTES)?) } else { None },
                        pools: zeroed(pool_pages * PAGE_ROWS * 132)?,
                    }));
                    mla_layers += 1;
                }
                GlmNextAttention::Kda => {
                    kv.push(None);
                    index.push(None);
                    kda_layers += 1;
                }
            }
        }
        let index_tails = if keys || mla_layers == 0 { None } else {
            Some((zeroed(mla_layers * slots * TAIL_BYTES)?, zeroed(mla_layers * decode_rows * KEY_BYTES)?))
        };
        let replay = kda_layers * replay;
        let (kda_replay, owner) = match records {
            Some(PrefillScratch(temps)) => {
                let scratch = temps.scratch.buffer;
                ensure!(scratch.bytes >= replay.max(256), "{replay} bytes of replay records do not fit the \
                    {}-byte prefill scratch", scratch.bytes);
                (cuteafd_ffi::CuteafdDeviceBuffer { bytes: replay.max(256), ..scratch },
                    RecordsOwner::Shared { _temporaries: temps })
            }
            None => {
                let own = zeroed(replay)?;
                (own.buffer, RecordsOwner::Own { _records: own })
            }
        };
        Ok(Self { kv, index, kda_state: zeroed(kda_layers * slots * state)?,
            kda_conv: zeroed(kda_layers * slots * conv)?,
            kda_replay, _kda_replay_owner: owner, index_tails,
            commit_tables: zeroed(3 * decode_rows * 4)?, pool_logical: zeroed(pool_pages * 4)?, kda_heads })
    }
}

/// The second GPU of a two-GPU head split (rank 1): its share of every layer, its caches
/// (its KDA heads' state; the replicated MLA records and DSA keys), workspaces and captured
/// decode segments.
pub(crate) struct GlmfPeer<'a> {
    pub device: i32,
    pub stream: *mut c_void,
    pub layers: Vec<GlmfLayer<'a>>,
    caches: Caches<'a>,
    decode_workspace: RefCell<Option<Workspace<'a>>>,
    /// Its prefill lanes (a serial prefill runs in the first).
    lane_workspaces: RefCell<Vec<Workspace<'a>>>,
    graphs: RefCell<GraphBank<GraphKey, GraphExec<'a>>>,
    /// L2 prefetch of its next layer's weights while it waits for rank 0's expert exchange.
    l2: Option<crate::shared::l2_prefetch::L2Prefetch>,
}

/// Exchange slot of layer `index`, lane `lane`: its attention partials (`ffn` false) or its
/// FFN exchange (dense partials, the shared-expert half from rank 1, the routed + shared sum
/// from rank 0), by layer parity.
fn slot(index: usize, ffn: bool, lane: usize) -> usize {
    4 * lane + 2 * (index % 2) + usize::from(ffn)
}

/// The normalized-heads slot of attention exchange slot `output_slot` with `lanes` prefill lanes
/// (after every lane's four exchange slots).
fn norm_slot(lanes: usize, output_slot: usize) -> usize {
    4 * lanes + (output_slot / 4) * 2 + (output_slot % 4) / 2
}

/// Output token rows: rank 0 owns the leading ceil half, rank 1 the remaining rows.
fn output_rows(rows: usize, rank: usize) -> (usize, usize) {
    let first = rows.div_ceil(2);
    if rank == 0 { (0, first) } else { (first, rows - first) }
}

pub(crate) struct GlmfEngine<'a> {
    quantize_grid: Fp8QuantizeGrid,
    pub library: &'a NativeLibrary,
    pub programs: &'a Programs<'a>,
    pub cfg: GlmNextConfig,
    pub weights: GlmfWeights<'a>,
    pub stream: *mut c_void,
    pub max_context: usize,
    /// Rows of one prefill lane (and of a serial prefill chunk).
    pub prefill_rows: usize,
    /// Prefill lanes (1..=[`MAX_PREFILL_LANES`]): a Spark prefill chunk runs in up to this many.
    pub prefill_lane_count: usize,
    pub pages: usize,
    pub slots: usize,
    /// The most rows of one decode or verify step (`--decode-rows`): [`DECODE_ROWS`], or
    /// [`WIDE_DECODE_ROWS`], where a step of more than 64 rows runs the wide `_m128` programs (fewer
    /// rows keep the `_m64` ones). The decode workspace, the replay records and the commit tables
    /// hold this many rows.
    pub decode_rows: usize,
    /// The most rows a verify step schedules, the drafts' budget: `decode_rows`, or with the wide
    /// programs this GPU's whole sparse MLA waves ([`verify_budget`]: 127 on an RTX 5090).
    pub verify_rows: usize,
    /// The row buckets padded decode steps run at (startup graphs), from `verify_rows`.
    buckets: DecodeBuckets,
    /// Rows per layer of the replay records the last speculative step wrote (its programs'
    /// capacity, `replay_rows_of`): the commit that follows reads records of that many rows.
    replay_rows: std::cell::Cell<usize>,
    /// Per layer: its index among the KDA layers (None for MLA).
    kda_ordinal: Vec<Option<usize>>,
    /// Per layer: its index among the MLA layers (None for KDA).
    mla_ordinal: Vec<Option<usize>>,
    /// The DSA index cache the caches were built for.
    pub index_cache: IndexCache,
    /// Prefix mark arena slots the KV admission reserved on the layout above, which the prefix
    /// cache allocates.
    pub mark_slots: usize,
    /// Where the replay records live, and the check that no prefill overwrites shared ones
    /// between a verify and its commit.
    pub replay_records: ReplayRecords,
    records: RecordGuard,
    /// This GPU's caches (rank 0 of a head split).
    caches: Caches<'a>,
    /// This engine's GPU (rank 0 of a head split).
    pub device: i32,
    /// The head split's second GPU and the exchange between the two.
    peer: Option<GlmfPeer<'a>>,
    exchange: Option<PeerExchange<'a>>,
    /// The drafter (DFlash2 or dSpark): every step taps its target layers.
    pub drafter: Option<super::dspark::Drafter<'a>>,
    /// L2 prefetch of the next layer's weights during decode exchanges.
    pub l2: Option<crate::shared::l2_prefetch::L2Prefetch>,
    /// Host copy of the caches' pool-page map (a shared pool page sits at the same logical
    /// page in every sequence that holds it).
    pool_logical_host: RefCell<Vec<i32>>,
    /// Pool-cache pages: one per allocation unit (`pages / UNIT_PAGES`).
    pub pool_pages: usize,
    /// Columns of a step row's MLA page table and pool-page table (`glmf_table_pages`): one
    /// sequence of `max_context` tokens, so the tables are sized before the pool.
    table_pages: usize,
    table_pool_pages: usize,
    decode_workspace: RefCell<Option<Workspace<'a>>>,
    /// Prefill lanes of `prefill_rows` rows over one set of temporaries (pipelined Spark prefill;
    /// a serial prefill runs in the first).
    lane_workspaces: RefCell<Vec<Workspace<'a>>>,
    experts: Option<Experts<'a>>,
    dense_nvfp4: Option<DenseNvfp4<'a>>,
    /// Host seconds: GPU wait before expert exchanges, the exchanges.
    /// Host seconds per phase: GPU work until each expert exchange (waiting
    /// for the routes and wire rows), the Spark exchanges, and the head (final
    /// norm, vocabulary projection, logits download).
    pub profile: RefCell<[f64; 3]>,
    /// Prefill steps keep every row's logits (golden scoring); otherwise the
    /// prefill workspace holds logits for at most `DECODE_ROWS` rows.
    pub full_prefill_logits: bool,
    /// Captured decode segments (CUTEAFD_GLMF_GRAPHS=0 runs decode eagerly), within the graph
    /// budget when one is set.
    graphs: RefCell<GraphBank<GraphKey, GraphExec<'a>>>,
    /// Executables evicted from either rank's cache (rank, executable): destroyed at the next
    /// decode step, once every rank's stream has drained.
    use_graphs: bool,
    /// Every serving decode graph captured at startup (`CUTEAFD_GLMF_STARTUP_GRAPHS`, on unless 0);
    /// a graph budget (`--graph-budget-mib`) captures lazily within it instead.
    pub(crate) startup_graphs: bool,
    warming_graphs: std::cell::Cell<bool>,
    logged_graph_shapes: RefCell<std::collections::HashSet<(usize, usize, bool, GraphGeometry)>>,
    /// Spark prefill runs in lanes (CUTEAFD_GLMF_PREFILL_LANES, default on).
    lanes: bool,
    /// CUTEAFD_GLMF_PREFILL_LANES=subset: lanes even with a `--layers`
    /// subset (timing runs against loopback ranks that hold only those layers).
    subset_lanes: bool,
    /// Opt-in matched-token/route evidence; never adds a device readback.
    split_audit: bool,
    /// Recorded after a layer's routes and wire rows reach the host staging.
    routes_ready: *mut c_void,
    /// The shared draft policy's per-step signals (layer events, routes), armed by
    /// `serve-glmf` under the shared policy and active only around its verify steps.
    draft_probe: RefCell<Option<super::draft_probe::RoundProbe<'a>>>,
    /// A decode segment is being captured: host-side probe work stays out of the graph.
    capturing: std::cell::Cell<bool>,
    ops: Option<RefCell<OpTimes>>,
    /// Prefill projections that run block-FP8 GEMMs (the layers need FP8 copies).
    pub fp8_prefill: Fp8Prefill,
    pub kda_fp32_partials: bool,
    pub kda_output_shard: bool,
    pub kda_prefill_expanded: bool,
    /// The KDA recurrent state's storage (its programs, slot and mark bytes).
    pub kda_state: KdaState,
    /// The token embedding table (resident on this GPU or read from its shard).
    pub embedding: TokenEmbedding<'a>,
    /// The serving loop's token selector when start-up made it before the KV pool.
    selector: RefCell<Option<TokenSelector<'a>>>,
}

/// Which prefill projections run W8A8 block-FP8 GEMMs (E4M3 activations per
/// row and 128-K block): the MLA projections, the dense and shared-expert MLPs
/// (FP8-only weights: W8A16 otherwise), the KDA in-projection and o_proj
/// (per-row x 128-K copies: BF16 otherwise).
#[derive(Debug, Default, Clone, Copy)]
pub(crate) struct Fp8Prefill {
    pub mla: bool,
    pub ffn: bool,
    /// KDA projections that run FP8 (bit 0 the in-projection, bit 1 o_proj).
    pub kda_bits: i32,
}

// Serving uses exact canonical row buckets; teacher-forced probe spans run eagerly
// and do not enlarge the startup graph set. Counters still count every capture.
use crate::shared::decode_graph::{check_bucket_thresholds, masked_row, ProjectionThreshold};

pub(crate) use cuteafd_loader::serving_capacity::glmf_graphs::{DecodeBuckets, GraphGeometry, decode_strides, serving_graph_shapes, serving_graph_reserve};
#[allow(unused_imports)]
use cuteafd_loader::serving_capacity::glmf_graphs::{graph_geometries, graph_reserve_bytes, MIN_PAGE_STRIDE, PLAIN_DECODE_BUCKETS, SPEC_DECODE_BUCKETS};
// Inclusive arithmetic crossovers audited against the pinned AOT exporter.
// Full/half-head shapes share these thresholds; MoE projections use real rows.
const DECODE_PROJECTION_THRESHOLDS: &[ProjectionThreshold] = &[
    ProjectionThreshold { name: "index.iq[4096,1536].bf16", skinny_rows: 8 },
    ProjectionThreshold { name: "index.ik[288,4096].bf16", skinny_rows: 160 },
    ProjectionThreshold { name: "kda.in[24896|12576,4096].bf16", skinny_rows: 8 },
    ProjectionThreshold { name: "kda.out[4096,8192|4096].bf16", skinny_rows: 8 },
    ProjectionThreshold { name: "kda.in.out.fp8", skinny_rows: 16 },
    ProjectionThreshold { name: "kda.half.in.out.fp8.wide", skinny_rows: 32 },
    ProjectionThreshold { name: "mla.qkv_a[2048,4096].fp8", skinny_rows: 16 },
    ProjectionThreshold { name: "mla.q_b[16384|8192,1536].fp8", skinny_rows: 16 },
    ProjectionThreshold { name: "mla.out[4096,16384|8192].fp8", skinny_rows: 16 },
    ProjectionThreshold { name: "dense.gate_up[24576|12288,4096].fp8", skinny_rows: 16 },
    ProjectionThreshold { name: "dense.down[4096,12288|6144].fp8", skinny_rows: 16 },
];

// `--decode-rows 128`: a step of 65 to 128 rows runs the `_m128` programs. They keep the `_m64`
// programs' arithmetic crossovers above (the same `_Fp8Switch` and BF16 projection routes, the
// 128 x 128 prefill route only from 512 rows, the decode KDA structures, the index producers' 8 and
// 160 rows), so their capacity is the one new threshold: no bucket may pad a step of up to 64 rows
// into them, whose sparse MLA plans its rows as its own bucket (FR-G.6).
const WIDE_DECODE_THRESHOLDS: &[ProjectionThreshold] = &[
    ProjectionThreshold { name: "glmf.decode_capacity[m64|m128]", skinny_rows: DECODE_ROWS },
];

/// Refuses bucket sets that would pad a step across an audited arithmetic crossover: the plain
/// buckets serving `sequences` reaches and, with `speculation`, the speculative ones.
fn check_decode_thresholds(buckets: &DecodeBuckets, sequences: usize, speculation: bool) -> Result<()> {
    let thresholds: Vec<ProjectionThreshold> =
        DECODE_PROJECTION_THRESHOLDS.iter().chain(WIDE_DECODE_THRESHOLDS).copied().collect();
    let cap = buckets.bucket(sequences.clamp(16, DECODE_ROWS), false);
    let plain: Vec<_> = buckets.plain.iter().copied().filter(|&rows| rows <= cap).collect();
    check_bucket_thresholds(&plain, &thresholds)?;
    if speculation { check_bucket_thresholds(&buckets.spec, &thresholds)?; }
    Ok(())
}
// WP9 2026-10-07, RTX PRO 6000 SM120, runtime99c: each new LM lane
// adds 71,942,144 B beyond tracked buffers; drafter adds 68,269,888 B. This
// measured/calibrated allowance is not exact cuBLAS allocator ownership.
#[allow(unused_imports)]
use cuteafd_loader::serving_capacity::glmf_graphs::WORKSPACE_RUNTIME_OVERHEAD_BYTES;

/// Per rank, the device bytes its step workspaces take before the KV pool, for an admission that
/// sizes the pool before they exist: the decode workspace and `lanes` prefill lanes of `prefill_rows`
/// rows over their one set of shared temporaries (`plan`'s arithmetic, which the engine allocates),
/// plus the measured untracked runtime memory per workspace, the drafter's included. A plan with
/// all-row prefill logits (`--full-prefill-logits`) gives the exact union scoring runs in: a serial
/// prefill runs in lane 0, and the logits live once, in the lanes' shared temporaries.
pub(crate) fn workspace_reserve(plan: &StepPlan<'_, '_>, prefill_rows: usize, lanes: usize, peer: bool,
    drafter: bool) -> Result<Vec<u64>> {
    (0..if peer { 2 } else { 1 }).map(|rank| {
        // The planner separately admits drafter storage. Its runtime overhead is additional.
        Ok(workspace_reserve_bytes(plan.workspace_bytes(rank, prefill_rows, lanes)?, lanes, rank == 0 && drafter))
    }).collect()
}

/// A rank's `tracked` workspace bytes with the untracked runtime memory of its decode workspace,
/// `lanes` prefill lanes and (`drafter`) the drafter's.
fn workspace_reserve_bytes(tracked: u64, lanes: usize, drafter: bool) -> u64 {
    tracked + cuteafd_loader::serving_capacity::glmf_graphs::workspace_runtime_overhead(lanes, drafter)
}

/// The prefill lanes a serving engine creates before readiness: `lanes` (`--prefill-lanes`) when
/// Spark prefill runs in lanes (`CUTEAFD_GLMF_PREFILL_LANES=1` turns them off, `subset` keeps them
/// with a `--layers` subset), else one serial workspace.
fn prefill_workspace_count(spark: bool, complete: bool, value: Option<&str>, lanes: usize) -> usize {
    if spark && value != Some("1") && (complete || value == Some("subset")) { lanes } else { 1 }
}
pub(crate) fn configured_prefill_lanes(spark: bool, complete: bool, lanes: usize) -> usize {
    prefill_workspace_count(spark, complete, std::env::var("CUTEAFD_GLMF_PREFILL_LANES").ok().as_deref(), lanes)
}

fn startup_graph_policy(graphs: Option<&str>, startup: Option<&str>) -> bool {
    graphs != Some("0") && startup != Some("0")
}
pub(crate) fn startup_graphs_enabled() -> bool {
    startup_graph_policy(std::env::var("CUTEAFD_GLMF_GRAPHS").ok().as_deref(),
        std::env::var("CUTEAFD_GLMF_STARTUP_GRAPHS").ok().as_deref())
}

pub(crate) use cuteafd_loader::serving_capacity::glmf_graphs::StartupGraphReserve;

/// Admission policy lives in the loader; CUDA error classification stays at the edge.
pub(crate) fn admit_beside_decode_graphs(startup: Option<StartupGraphReserve>,
    admit: impl FnMut(Option<u64>) -> Result<usize>) -> Result<(usize, bool)> {
    let result = cuteafd_loader::serving_capacity::glmf_graphs::admit_beside_decode_graphs(
        startup, admit, crate::shared::memory_report::kv_shortfall,
        |retry, refused| retry.context(format!("admitting the GLM Flash KV pool with lazily captured \
            decode graphs, after the startup set's admission was refused ({refused:#})")));
    if let (Some(graphs), Ok((tokens, false))) = (startup, &result) {
        tracing::warn!(startup_graph_bytes = graphs.reserve, graph_allowance_bytes = graphs.allowance,
            lazy_pool_tokens = tokens, "GLM Flash startup decode graphs refused: capturing lazily within the graph allowance");
    }
    result
}

/// A startup capture's tables: `rows` masked rows over `geometry`.
fn startup_tables(rows: usize, spec: bool, geometry: GraphGeometry) -> StepTables {
    let mut tables = StepTables { decode: true, spec, long: geometry.long, pool_width: geometry.pool_width,
        page_stride: geometry.page_stride, pool_stride: geometry.pool_stride, ..Default::default() };
    pad_decode_tables(&mut tables, rows);
    tables
}

fn pad_decode_tables(tables: &mut StepTables, bucket: usize) {
    while tables.kv_slots.len() < bucket {
        let row = masked_row(tables.kv_slots.len());
        tables.positions.push(row.position);
        tables.kv_slots.push(row.kv_slot);
        tables.pool_slots.push(row.pool_slot);
        tables.kda_slots.push(row.state_slot);
        tables.seq_first.push(row.seq_first);
        tables.cache_lengths.push(row.cache_length);
        tables.page_table.extend(std::iter::repeat_n(0, tables.page_stride));
        tables.pool_table.extend(std::iter::repeat_n(0, tables.pool_stride));
    }
}

/// What a captured decode segment baked in.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct GraphKey {
    segment: usize,
    rows: usize,
    spec: bool,
    long: bool,
    pool_width: usize,
    page_stride: usize,
    pool_stride: usize,
}

impl GraphKey {
    /// Segment `segment` of a decode step of `rows` rows over `tables`, keyed by its geometry
    /// ([`GraphGeometry::keyed`]: a short step keys no pool-table width or stride).
    fn new(segment: usize, rows: usize, tables: &StepTables) -> Self {
        let geometry = GraphGeometry::keyed(tables.pool_width, tables.page_stride, tables.pool_stride, tables.long);
        Self { segment, rows, spec: tables.spec, long: geometry.long, pool_width: geometry.pool_width,
            page_stride: geometry.page_stride, pool_stride: geometry.pool_stride }
    }
}

/// What the decode graph caches did and hold.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub(crate) struct GraphCounts {
    pub stats: GraphStats,
    /// Executables held, the step shapes they belong to (segment 0's), and their charged bytes.
    pub held: usize,
    pub shapes: usize,
    pub bytes: u64,
}

type GraphExec<'a> = GraphOwner<'a, ()>;

fn bytes_of<T: Copy>(values: &[T]) -> &[u8] {
    // SAFETY: plain-old-data slices viewed as bytes for host->device copies.
    unsafe { std::slice::from_raw_parts(values.as_ptr().cast(), std::mem::size_of_val(values)) }
}

fn audit_token_hash(tokens: &[u32]) -> u64 {
    tokens.iter().flat_map(|token| token.to_le_bytes()).fold(0xcbf2_9ce4_8422_2325,
        |hash, byte| (hash ^ u64::from(byte)).wrapping_mul(0x0000_0100_0000_01b3))
}

fn audit_route_counts(routes: &[ExpertProtocolV2RouteEntry]) -> std::collections::BTreeMap<u32, usize> {
    let mut counts = std::collections::BTreeMap::new();
    for route in routes {
        *counts.entry(route.expert_id).or_insert(0) += 1;
    }
    counts
}

/// A Spark wave of `layer`: `rows` rows of E4M3 input with a UE8M0 scale per 32 values
/// (`hidden + hidden / 32` bytes each) and `topk` routes per row, answered with compact BF16
/// partials. Serving waves (`spark_dispatch`) and the start-up warm-up ([`spark_warmup_request`])
/// both take this format.
fn spark_request(hidden: usize, topk: usize, layer: usize, rows: usize, routes: Vec<ExpertProtocolV2RouteEntry>,
    wire: Vec<u8>, kind: ExpertV2SourceKind) -> Result<ExpertProtocolV2Request> {
    let topk = topk as u32;
    let mut request = ExpertProtocolV2Request::new(layer as u64 + 1, 17, layer as u32, hidden as u32,
        ExpertV2Dtype::Fp8E4m3Ue8m0K32,
        (0..rows as u32).map(|row| ExpertProtocolV2RowDescriptor {
            row_id: u64::from(row), source_kind: kind, source_request_id: 1,
            token_position: u64::from(row), route_offset: row * topk, route_count: topk,
        }).collect(),
        routes, wire)?;
    request.header.flags |= EXPERT_PROTOCOL_V2_FLAG_V41_COMPACT_BF16;
    Ok(request)
}

/// The wave that warms a Spark transport at start-up (as GLM-5, MiMo and DeepSeek V4 warm theirs):
/// `rows` zero rows of prefill at the first MoE layer, every gate zero and the experts in turn. Each
/// rank's session sizes its rings for the request that opens it, and the transport drops and
/// reconnects a rank whose rings a later request outgrows, so a transport warmed with the most rows
/// any of its waves carries never reconnects while serving.
pub(crate) fn spark_warmup_request(cfg: &GlmNextConfig, rows: usize) -> Result<ExpertProtocolV2Request> {
    let layer = (0..cfg.layers).find(|&layer| !cfg.dense[layer]).context("no MoE layer to warm the Spark transports")?;
    let (h, topk) = (cfg.hidden, cfg.topk);
    let routes = (0..rows * topk).map(|i| ExpertProtocolV2RouteEntry {
        row_index: (i / topk) as u32, expert_id: (i % cfg.experts) as u32, gate_weight: 0.0,
    }).collect();
    spark_request(h, topk, layer, rows, routes, vec![0; rows * (h + h / 32)], ExpertV2SourceKind::Prefill)
}

impl<'a> GlmfEngine<'a> {
    /// `index_cache`: the DSA index cache (`compact` needs at least one MLA layer to matter;
    /// without one there is no index cache, and the engine records `keys`). `decode_rows`: the most
    /// rows of one decode or verify step, [`DECODE_ROWS`] or [`WIDE_DECODE_ROWS`].
    #[allow(clippy::too_many_arguments)]
    pub fn new(library: &'a NativeLibrary, programs: &'a Programs<'a>, cfg: GlmNextConfig, weights: GlmfWeights<'a>,
        stream: *mut c_void, max_context: usize, prefill_rows: usize, prefill_lane_count: usize, pages: usize,
        slots: usize, embedding: TokenEmbedding<'a>, index_cache: IndexCache, kda_state: KdaState, decode_rows: usize,
        records: Option<PrefillScratch<'a>>) -> Result<Self> {
        let _memory_scope = cuteafd_ffi::memory_ledger::scope("kv");
        let sms = library.sm_count()?;
        let quantize_grid = Fp8QuantizeGrid::new(sms, None)?;
        ensure!(decode_rows == DECODE_ROWS || decode_rows == WIDE_DECODE_ROWS,
            "decode steps of up to {decode_rows} rows: the programs take {DECODE_ROWS} or {WIDE_DECODE_ROWS}");
        let verify_rows = verify_budget(decode_rows, sms);
        tracing::info!(decode_rows, verify_rows, sms, "GLM Flash decode and verify step rows");
        ensure!(embedding.hidden() == cfg.hidden, "embedding rows of {} for hidden {}", embedding.hidden(), cfg.hidden);
        ensure!((1..=MAX_PREFILL_LANES).contains(&prefill_lane_count) && prefill_rows > 0,
            "{prefill_lane_count} prefill lanes of {prefill_rows} rows (1 to {MAX_PREFILL_LANES} lanes)");
        ensure!(cfg.hc_mult == HC && cfg.kv_lora_rank == 512 && cfg.kda_head_dim == 128 && cfg.heads == 64,
            "the glmf programs are built for 4 mHC streams, a 512 latent, 64 MLA heads and 128-wide KDA heads");
        // Whole allocation units: four MLA pages and one pool page each.
        let pages = pages.max(1).next_multiple_of(UNIT_PAGES);
        let pool_pages = pages / UNIT_PAGES;
        let ordinals = |kind: GlmNextAttention| -> Vec<Option<usize>> {
            let mut count = 0;
            weights.layers.iter().map(|layer| (layer.attention == kind).then(|| {
                count += 1;
                count - 1
            })).collect()
        };
        let (kda_ordinal, mla_ordinal) = (ordinals(GlmNextAttention::Kda), ordinals(GlmNextAttention::Mla));
        // No MLA layer, no index cache: the compact one has nothing to hold.
        let index_cache = if mla_ordinal.iter().flatten().next().is_some() { index_cache } else { IndexCache::Keys };
        // A head split's shares hold half the KDA heads (and their state).
        let split = weights.layers.first().is_some_and(|l| l.split);
        ensure!(!split || index_cache == IndexCache::Keys, "a head split keeps the per-token index keys");
        // The head split's share programs (`glmf2_*`) exist at 64 decode rows only.
        ensure!(!split || decode_rows == DECODE_ROWS,
            "a head split takes decode steps of up to {DECODE_ROWS} rows (--decode-rows {DECODE_ROWS})");
        let kda_heads = cfg.kda_heads / if split { 2 } else { 1 };
        ensure!(records.is_none() || !split, "a head split keeps the replay records of its own (--replay-records own)");
        let replay_records = if records.is_some() { ReplayRecords::Shared } else { ReplayRecords::Own };
        let caches = Caches::new(library, &cfg, &weights.layers, pages, pool_pages, slots, kda_heads, index_cache,
            kda_state, decode_rows, records)?;
        let device = library.cuda_get_device()?;
        let (table_pages, table_pool_pages) = glmf_table_pages(max_context as u64);
        let (table_pages, table_pool_pages) = (usize::try_from(table_pages)?, usize::try_from(table_pool_pages)?);
        Ok(Self { quantize_grid, library, programs, cfg, weights, stream, max_context, prefill_rows, prefill_lane_count,
            pages, slots, decode_rows, verify_rows, buckets: DecodeBuckets::new(verify_rows),
            replay_rows: std::cell::Cell::new(DECODE_ROWS),
            kda_ordinal, mla_ordinal, index_cache, replay_records, records: RecordGuard::new(replay_records), caches,
            device, peer: None, exchange: None, drafter: None, pool_logical_host: RefCell::new(vec![0; pool_pages]), pool_pages,
            table_pages, table_pool_pages, decode_workspace: RefCell::new(None),
            lane_workspaces: RefCell::new(Vec::new()),
            experts: None, dense_nvfp4: None, profile: RefCell::new([0.0; 3]), graphs: RefCell::new(GraphBank::new(None)),

            use_graphs: std::env::var("CUTEAFD_GLMF_GRAPHS").map_or(true, |v| v != "0"),
            startup_graphs: startup_graphs_enabled(),
            warming_graphs: std::cell::Cell::new(false), logged_graph_shapes: RefCell::new(std::collections::HashSet::new()),
            lanes: lane_setting().0,
            subset_lanes: lane_setting().1,
            split_audit: std::env::var("CUTEAFD_GLMF_SPLIT_AUDIT").is_ok_and(|v| v == "1"),
            full_prefill_logits: false, routes_ready: library.cuda_event_create_ordering()?,
            draft_probe: RefCell::new(None), capturing: std::cell::Cell::new(false),
            ops: std::env::var("CUTEAFD_GLMF_PROFILE_OPS").is_ok_and(|v| v == "1").then(RefCell::default),
            fp8_prefill: Fp8Prefill::default(), kda_fp32_partials: false,
            kda_output_shard: false, mark_slots: 0,
            kda_prefill_expanded: false, kda_state, l2: None, embedding, selector: RefCell::new(None) })
    }

    /// Decode graphs captured lazily as steps arrive, at their exact rows, instead of the startup set
    /// (the KV admission found no room for it: `admit_beside_decode_graphs`).
    pub(crate) fn capture_graphs_lazily(&mut self) {
        self.startup_graphs = false;
    }

    /// Own every ordinary text/media workspace before readiness: both ranks' decode workspaces and
    /// the prefill lanes a Spark prefill runs in (`--prefill-lanes`, over one set of temporaries;
    /// otherwise the serial workspace, lane 0), and the drafter's. No shared BLAS handles.
    pub fn prepare_serving_workspaces(&self) -> Result<()> {
        let lanes = if self.pipelined() { self.prefill_lane_count } else { 1 };
        for rank in 0..self.ranks() {
            drop(self.decode_workspace_of(rank)?);
            drop(self.prefill_lanes_of(rank, lanes)?);
        }
        if let Some(drafter) = &self.drafter { drafter.prepare_workspace()?; }
        self.synchronize()
    }

    /// Capture the complete serving key set on masked rows, without expert traffic. The bucket sets
    /// pass the projection thresholds before readiness in every graph mode.
    pub fn warm_decode_graphs(&self, sequences: usize, speculation: bool) -> Result<usize> {
        if !self.use_graphs { return Ok(0); }
        check_decode_thresholds(&self.buckets, sequences, speculation)?;
        tracing::info!(plain_rows = ?self.buckets.plain, spec_rows = ?self.buckets.spec, decode_rows = self.decode_rows,
            verify_rows = self.verify_rows, startup = self.startup_graphs,
            "GLM Flash decode buckets pass the projection thresholds");
        if !self.startup_graphs { return Ok(0); }
        let shapes = serving_graph_shapes(self.max_context, self.pages, self.cfg.dense_context(), sequences, speculation,
            &self.buckets);
        let segments = self.weights.layers.len() + 1;
        let expected = shapes.len() * (segments + self.peer.as_ref().map_or(0, |p| p.layers.len()));
        tracing::info!(shapes = shapes.len(), graphs = expected, plain_rows = ?self.buckets.plain,
            spec_rows = ?self.buckets.spec, "GLM Flash startup decode graph admission");
        // Allocate fixed workspaces before measuring the graph executables' physical memory.
        for rank in 0..self.ranks() {
            drop(self.decode_workspace_of(rank)?);
        }
        self.synchronize()?;
        let free = || (0..self.ranks()).map(|rank| self.on(rank,
            || self.library.cuda_physical_memory_info().map(|(free, _)| free as i64))).collect::<Result<Vec<_>>>();
        let before = free()?;
        let started = std::time::Instant::now();
        self.warming_graphs.set(true);
        let captured = (|| -> Result<()> {
            for &(rows, spec, geometry) in &shapes {
                self.step(&startup_tables(rows, spec, geometry), &vec![0; rows], rows, None, None, None, None)?;
            }
            self.synchronize()
        })();
        self.warming_graphs.set(false);
        captured?;
        let graphs = self.graphs.borrow().len() + self.peer.as_ref().map_or(0, |p| p.graphs.borrow().len());
        ensure!(graphs == expected, "startup captured {graphs} graphs, expected {expected}");
        for rank in 0..self.ranks() {
            let end = if rank == 0 { segments } else { segments - 1 };
            let keys: Vec<_> = shapes.iter().flat_map(|&(rows, spec, geometry)| {
                let tables = startup_tables(rows, spec, geometry);
                (0..end).map(move |segment| GraphKey::new(segment, rows, &tables))
            }).collect();
            self.graphs_of(rank).borrow_mut().seal_startup(&keys)?;
        }
        for &(rows, spec, geometry) in &shapes {
            let tables = startup_tables(rows, spec, geometry);
            for rank in 0..self.ranks() {
                let graphs = if rank == 0 { &self.graphs } else { &self.peer()?.graphs };
                let end = if rank == 0 { segments } else { segments - 1 };
                for segment in 0..end {
                    ensure!(graphs.borrow().contains(&GraphKey::new(segment, rows, &tables)),
                        "startup graph coverage missing");
                }
            }
        }
        let bytes: Vec<i64> = before.into_iter().zip(free()?).map(|(before, after)| before - after).collect();
        tracing::info!(graphs, shapes = shapes.len(), ?bytes, elapsed_ms = started.elapsed().as_millis() as u64,
            "GLM Flash decode graphs captured at startup");
        Ok(graphs)
    }

    /// One-GPU real-row byte gate; plain steps restore the complete persistent state. With the wide
    /// programs the speculative steps also pad 65 rows and the verify budget less one into the
    /// budget's bucket (65 -> 127 and 126 -> 127 on an RTX 5090).
    pub fn check_decode_padding(&self, tokens: &[u32]) -> Result<()> {
        ensure!(self.ranks() == 1 && self.use_graphs && self.startup_graphs,
            "padding check needs one GPU, graphs and CUTEAFD_GLMF_STARTUP_GRAPHS=1");
        let mut spec_rows = vec![3, 9, 17, 33];
        if self.verify_rows > DECODE_ROWS {
            spec_rows.extend([DECODE_ROWS + 1, self.verify_rows - 1]);
        }
        let needed = 32 + spec_rows.iter().copied().max().unwrap_or(0);
        ensure!(tokens.len() >= needed && self.max_context >= needed, "padding check needs {needed} tokens of context");
        let mut placement = GlmfPlacement::new(vec![0], 0);
        self.prefill_device(&mut placement, &tokens[..32])?;
        let caches = self.caches_of(0);
        // Every persistent byte a step can write: KDA state, conv state and replay records, the
        // compact index cache's tails and records, and the paged MLA records, keys and pools.
        let mut buffers = vec![caches.kda_state.buffer, caches.kda_conv.buffer, caches.kda_replay];
        if let Some((tails, records)) = &caches.index_tails {
            buffers.extend([tails.buffer, records.buffer]);
        }
        for layer in self.paged_buffers() {
            buffers.extend([layer.records, layer.pools]);
            buffers.extend(layer.keys);
        }
        let snapshot = || -> Result<Vec<Vec<u8>>> {
            self.synchronize()?;
            buffers.iter().map(|&buffer| {
                let mut bytes = vec![0; buffer.bytes];
                self.library.copy_d2h(&mut bytes, buffer)?;
                Ok(bytes)
            }).collect()
        };
        let restore = |state: &[Vec<u8>]| -> Result<()> {
            self.synchronize()?;
            for (&buffer, bytes) in buffers.iter().zip(state) { self.library.copy_h2d(buffer, bytes)?; }
            Ok(())
        };
        let original = placement.clone();
        let before = snapshot()?;
        for rows in [5, 9] {
            let input = &tokens[32..32 + rows];
            restore(&before)?;
            placement = original.clone();
            let exact = self.decode_step(&mut [(&mut placement, rows)], input, None, false, None, None, true)?
                .context("padding check needs all layers")?.to_host(self.library)?;
            let exact_state = snapshot()?;
            let exact_placement = (placement.len, placement.kda_len);
            restore(&before)?;
            placement = original.clone();
            let padded = self.verify_device(&mut [(&mut placement, rows)], input, false)?
                .context("padding check needs all layers")?.to_host(self.library)?;
            let bucket = self.buckets.bucket(rows, false);
            ensure!(exact.len() == padded.len() && exact.iter().zip(&padded).all(|(a, b)| a.to_bits() == b.to_bits()),
                "plain decode padding {rows}->{bucket} changed real-row logits");
            ensure!(snapshot()? == exact_state && (placement.len, placement.kda_len) == exact_placement,
                "plain decode padding {rows}->{bucket} changed persistent state");
            tracing::info!(rows, bucket, bytes = exact.len() * 4,
                "GLM Flash plain padded decode real-row logits and persistent state byte-exact");
        }
        restore(&before)?;
        placement = original;
        for rows in spec_rows {
            let start = placement.len;
            let input = &tokens[32..32 + rows];
            let mut ignore = |_: usize, _: &[u8]| Ok(());
            let plain = self.decode_step(&mut [(&mut placement, rows)], input, Some(&mut ignore), true, None, None, true)?
                .context("padding check needs all layers")?.to_host(self.library)?;
            placement.len = start;
            let padded = self.verify_device(&mut [(&mut placement, rows)], input, true)?
                .context("padding check needs all layers")?.to_host(self.library)?;
            placement.len = start;
            let bucket = self.buckets.bucket(rows, true);
            ensure!(plain.len() == padded.len() && plain.iter().zip(&padded).all(|(a, b)| a.to_bits() == b.to_bits()),
                "decode padding {rows}->{bucket} changed real-row logits");
            tracing::info!(rows, bucket, programs = decode_cap(bucket), bytes = plain.len() * 4,
                "GLM Flash padded decode real-row logits byte-exact");
        }
        self.synchronize()
    }

    /// Serves MoE layers from `experts` (without, the engine stops at the first MoE layer).
    /// The one-expert NVFP4 package for ModelOpt NVFP4 dense MLPs.
    pub fn set_dense_nvfp4(&mut self, dense: DenseNvfp4<'a>) {
        self.dense_nvfp4 = Some(dense);
    }

    pub fn set_experts(&mut self, experts: Experts<'a>) {
        self.experts = Some(experts);
    }

    pub fn experts(&self) -> Option<&Experts<'a>> {
        self.experts.as_ref()
    }

    /// Attaches the head split's second GPU: `device` with `stream`, holding `layers` (every
    /// layer's rank-1 share, see `GlmfLoader::model`). Loads the programs there, allocates its
    /// caches and the exchange (four slots per prefill lane, six with the KDA output shard).
    pub fn attach_peer(&mut self, device: i32, stream: *mut c_void, layers: Vec<GlmfLayer<'a>>) -> Result<()> {
        ensure!(layers.len() == self.weights.layers.len() && layers.iter().chain(&self.weights.layers).all(|l| l.split),
            "attach_peer needs the head-split shares of every loaded layer");
        // The share programs (`glmf2_*`) exist at 64 decode rows only.
        ensure!(self.decode_rows == DECODE_ROWS,
            "a head split takes decode steps of up to {DECODE_ROWS} rows (--decode-rows {DECODE_ROWS})");
        let rows = self.prefill_rows.max(DECODE_ROWS);
        let exchange = PeerExchange::new(self.library, [RankDevice { device: self.device, stream: self.stream },
            RankDevice { device, stream }], if self.kda_output_shard { 6 } else { 4 } * self.prefill_lane_count,
            rows * self.cfg.hidden * self.partial_bytes())?;
        // Both GPUs run the indexer; per-rank index tails are not built yet.
        ensure!(self.index_cache == IndexCache::Keys, "a head split keeps the per-token index keys (--index-cache keys)");
        let peer = exchange.on(1, || -> Result<GlmfPeer<'a>> {
            // Its own and the shares' programs.
            self.programs.load_matching(|name| super::glmf_startup_program(name, true, self.decode_rows > DECODE_ROWS))?;
            let _memory_scope = cuteafd_ffi::memory_ledger::scope("kv");
            let caches = Caches::new(self.library, &self.cfg, &layers, self.pages, self.pool_pages, self.slots,
                self.caches.kda_heads, self.index_cache, self.kda_state, self.decode_rows, None)?;
            Ok(GlmfPeer { device, stream, layers, caches,
                decode_workspace: RefCell::new(None), lane_workspaces: RefCell::new(Vec::new()),
                graphs: RefCell::new(GraphBank::new(self.graphs.borrow().budget())), l2: None })
        })?;
        self.peer = Some(peer);
        self.exchange = Some(exchange);
        // Native kernels load lazily on first launch, and a lazy load can wait for the device:
        // queued a layer ahead, rank 1 would then wait for its own stream, which waits on a
        // push the host has not queued yet. Load the native MLA prefill on both GPUs now.
        for rank in 0..2 {
            self.on(rank, || self.warm_mla_prefill(rank))?;
        }
        self.synchronize()
    }

    /// One masked row of the native MLA prefill on rank `rank` (loads its kernel there).
    fn warm_mla_prefill(&self, rank: usize) -> Result<()> {
        let Some(kernel) = crate::families::glm5::engine::native_mla_prefill() else { return Ok(()) };
        let heads = self.cfg.heads / 2;
        let q = self.alloc(heads * self.cfg.kv_lora_rank * 2)?;
        let kv = self.alloc(PAGE_ROWS * RECORD_BYTES)?;
        let indices = self.alloc(SPARSE_TOPK * 4)?;
        self.library.copy_h2d(indices.buffer, &vec![0xFFu8; SPARSE_TOPK * 4])?;
        let lengths = self.alloc(4)?;
        self.library.cuda_zero_bytes(lengths.buffer, 256)?;
        let out = self.alloc(heads * self.cfg.kv_lora_rank * 2)?;
        // SAFETY: every buffer above is live and sized for one row of `heads` heads; the
        // stream drains before they drop.
        unsafe {
            self.library.glm_mla_prefill(q.buffer.ptr, kv.buffer.ptr, indices.buffer.ptr, lengths.buffer.ptr,
                out.buffer.ptr, 1, heads, SPARSE_TOPK, RECORD_BYTES, 1.0, kernel, self.stream_of(rank))?;
            self.library.cuda_stream_synchronize(self.stream_of(rank))
        }
    }

    /// GPUs this engine runs on: 2 under a head split.
    pub fn ranks(&self) -> usize {
        1 + usize::from(self.peer.is_some())
    }

    fn peer(&self) -> Result<&GlmfPeer<'a>> {
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

    fn caches_of(&self, rank: usize) -> &Caches<'a> {
        match (rank, &self.peer) {
            (1, Some(peer)) => &peer.caches,
            _ => &self.caches,
        }
    }

    /// Drains every rank's stream.
    pub(crate) fn synchronize(&self) -> Result<()> {
        for rank in 0..self.ranks() {
            // SAFETY: the engine owns these streams.
            self.on(rank, || unsafe { self.library.cuda_stream_synchronize(self.stream_of(rank)) })?;
        }
        Ok(())
    }

    /// Before a sequence's first step: zeroes its KDA state and maps its pool pages.
    fn start(&self, placement: &GlmfPlacement) -> Result<()> {
        self.map_pools(placement)?;
        self.reset_slot(placement.slot)
    }

    /// Records each of `placement`'s pool pages' logical page (the index expansion reads it).
    /// A restored sequence maps its pages before its first step as a fresh one does; shared
    /// pages keep the value they have.
    pub fn map_pools(&self, placement: &GlmfPlacement) -> Result<()> {
        let mut host = self.pool_logical_host.borrow_mut();
        let mut changed = false;
        for (logical, &page) in placement.pool_pages.iter().enumerate() {
            let page = usize::try_from(page)?;
            ensure!(page < self.pool_pages, "pool page {page} out of range");
            changed |= std::mem::replace(&mut host[page], logical as i32) != logical as i32;
        }
        if changed {
            // Entries other sequences' queued steps read keep their values.
            for rank in 0..self.ranks() {
                let map = self.caches_of(rank).pool_logical.buffer;
                self.on(rank, || self.library.copy_h2d(cuteafd_ffi::CuteafdDeviceBuffer { bytes: host.len() * 4, ..map },
                    bytes_of(&host[..])))?;
            }
        }
        Ok(())
    }

    /// Copies a sequence's KDA recurrent and conv state (and, with the compact index cache, its
    /// index tails) from slot `from` to slot `to` on the engine stream (the backup a
    /// speculative verify restores before replaying its accepted rows).
    pub fn copy_slot(&self, from: i32, to: i32) -> Result<()> {
        let (from, to) = (usize::try_from(from)?, usize::try_from(to)?);
        ensure!(from < self.slots && to < self.slots && from != to, "KDA slots {from} -> {to} out of range");
        for rank in 0..self.ranks() {
            for (at_from, at_to) in self.slot_regions_on(rank, from).into_iter().zip(self.slot_regions_on(rank, to)) {
                // SAFETY: both slot regions are live and disjoint; the rank's stream orders the copy.
                self.on(rank, || unsafe {
                    self.library.copy_d2d_async(at_to, at_from, at_from.bytes, self.stream_of(rank))
                })?;
            }
        }
        Ok(())
    }

    /// Every KDA layer's recurrent and conv state regions of `slot`, then (compact index cache)
    /// every MLA layer's index tail (rank 0's under a head split).
    pub(crate) fn slot_regions(&self, slot: usize) -> Vec<cuteafd_ffi::CuteafdDeviceBuffer> {
        self.slot_regions_on(0, slot)
    }

    /// [`Self::slot_regions`] on rank `rank` (its KDA heads).
    pub(crate) fn slot_regions_on(&self, rank: usize, slot: usize) -> Vec<cuteafd_ffi::CuteafdDeviceBuffer> {
        let layers = self.kda_ordinal.iter().flatten().count().max(1);
        let caches = self.caches_of(rank);
        let mut out = Vec::new();
        let tails = caches.index_tails.as_ref().map(|(tails, _)| (tails, self.mla_ordinal.iter().flatten().count()));
        for (pool, layers) in [(&caches.kda_state, layers), (&caches.kda_conv, layers)].into_iter().chain(tails) {
            let per = pool.buffer.bytes / layers / self.slots;
            for layer in 0..layers {
                out.push(cuteafd_ffi::CuteafdDeviceBuffer {
                    // SAFETY: layer < layers and slot < slots: the region lies inside the pool.
                    ptr: unsafe { pool.buffer.ptr.cast::<u8>().add((layer * self.slots + slot) * per) }.cast(),
                    bytes: per,
                    ..pool.buffer
                });
            }
        }
        out
    }

    /// Every MLA layer's paged buffers: latent records (528 B per row), indexer token keys
    /// (512 B per row; `--index-cache keys` only), both in 64-row MLA pages, and the pool-key
    /// cache (8448 B per pool page).
    pub(crate) fn paged_buffers(&self) -> Vec<PagedLayer> {
        self.paged_buffers_on(0)
    }

    /// [`Self::paged_buffers`] of rank `rank` (1: the head split's identical copy).
    pub(crate) fn paged_buffers_on(&self, rank: usize) -> Vec<PagedLayer> {
        let caches = self.caches_of(rank);
        caches.kv.iter().zip(&caches.index).filter_map(|(kv, index)| match (kv, index) {
            (Some(kv), Some(index)) => Some(PagedLayer { records: kv.buffer, keys: index.keys.as_ref().map(|k| k.buffer),
                pools: index.pools.buffer }),
            _ => None,
        }).collect()
    }

    /// Zeroes a sequence's KDA recurrent and conv state, and its index tails (before its first
    /// step).
    pub fn reset_slot(&self, slot: i32) -> Result<()> {
        let slot = usize::try_from(slot)?;
        ensure!(slot < self.slots, "KDA slot {slot} out of range");
        for rank in 0..self.ranks() {
            for region in self.slot_regions_on(rank, slot) {
                self.on(rank, || self.library.cuda_zero_bytes(region, region.bytes))?;
            }
        }
        Ok(())
    }

    /// Every KDA layer's recurrent then conv state of `slot` (host copy; checks): the recurrent
    /// state first (FP32 or BF16, rank by rank under a head split), then the BF16 conv state, then
    /// (compact index cache) every MLA layer's index tail.
    pub fn slot_state(&self, slot: i32) -> Result<Vec<u8>> {
        let slot = usize::try_from(slot)?;
        ensure!(slot < self.slots, "KDA slot {slot} out of range");
        self.synchronize()?;
        let kda = self.kda_ordinal.iter().flatten().count().max(1);
        let (mut state, mut conv, mut tails) = (Vec::new(), Vec::new(), Vec::new());
        for rank in 0..self.ranks() {
            let regions = self.slot_regions_on(rank, slot);
            // `slot_regions_on`: every KDA layer's recurrent region, every KDA layer's conv region,
            // then every MLA layer's index tail.
            let (recurrent, rest) = regions.split_at(kda);
            let (window, tail) = rest.split_at(kda);
            for (out, regions) in [(&mut state, recurrent), (&mut conv, window), (&mut tails, tail)] {
                for &region in regions {
                    let mut bytes = vec![0u8; region.bytes];
                    self.on(rank, || self.library.copy_d2h(&mut bytes, region))?;
                    out.extend(bytes);
                }
            }
        }
        state.extend(conv);
        state.extend(tails);
        Ok(state)
    }

    /// After a speculative verify step (`verify_spec`): applies each
    /// sequence's first `keep` rows (from step row `first`) to the KDA state
    /// of `slot` in every layer, as serial steps over those rows would have,
    /// and (compact index cache, the same launch) rebuilds its index tail in
    /// every MLA layer from the old tail and the kept rows' keys and gates.
    /// Callers set the committed placements' `kda_len` to their kept length. The records are the last
    /// speculative step's: `kda_commit*` reads them after an `_m64` step, `kda_commit*_m128` after a
    /// wide one.
    pub fn commit(&self, sequences: &[(i32, usize, usize)]) -> Result<()> {
        if sequences.is_empty() {
            return Ok(());
        }
        self.records.check_commit()?;
        let recorded = self.replay_rows.get();
        ensure!(sequences.len() <= self.decode_rows && sequences.iter().all(|&(slot, first, keep)|
            slot >= 0 && (slot as usize) < self.slots && first + keep <= recorded), "commit of {sequences:?}");
        let n = sequences.len();
        let mut tables = vec![0i32; 3 * n];
        for (i, &(slot, first, keep)) in sequences.iter().enumerate() {
            tables[i] = slot;
            tables[n + i] = first as i32;
            tables[2 * n + i] = keep as i32;
        }
        let layers = self.kda_ordinal.iter().flatten().count();
        let split = self.peer.is_some();
        for rank in 0..self.ranks() {
            let caches = self.caches_of(rank);
            if rank == 1 {
                // SAFETY: the engine owns the peer stream; drained before its tables are rewritten.
                self.on(1, || unsafe { self.library.cuda_stream_synchronize(self.stream_of(1)) })?;
            }
            self.on(rank, || self.put(&caches.commit_tables, &tables))?;
            let mut pointers = vec![("state", caches.kda_state.buffer.ptr),
                ("conv_state", caches.kda_conv.buffer.ptr), ("replay", caches.kda_replay.ptr),
                ("tables", caches.commit_tables.buffer.ptr)];
            let mut scalars = vec![Scalar::I32(n as i32), Scalar::I32(layers as i32), Scalar::I32(self.slots as i32)];
            // The state's commit; with the compact index cache, the one that also rebuilds the tails.
            let program = match &caches.index_tails {
                Some((tails, records)) => {
                    pointers.extend([("tails", tails.buffer.ptr), ("index_replay", records.buffer.ptr)]);
                    scalars.push(Scalar::I32(self.mla_ordinal.iter().flatten().count() as i32));
                    self.kda_state.compact_commit_program()
                }
                None => self.kda_state.commit_program(),
            };
            self.run_on(rank, split, &commit_for(program, recorded), &pointers, &scalars)?;
        }
        Ok(())
    }

    fn alloc(&self, bytes: usize) -> Result<Dev<'a>> {
        DeviceAllocation::new(self.library, bytes.max(256))
    }

    fn run(&self, name: &str, pointers: &[(&str, *mut c_void)], scalars: &[Scalar]) -> Result<()> {
        self.run_on(0, false, name, pointers, scalars)
    }

    /// Launches `glmf_{name}` (`split`: a head split's share, `glmf2_{name}`) on rank `rank`'s
    /// stream (rank 0's launches timed when profiling ops).
    fn run_on(&self, rank: usize, split: bool, name: &str, pointers: &[(&str, *mut c_void)], scalars: &[Scalar])
        -> Result<()> {
        let name = format!("{}_{name}", if split { "glmf2" } else { "glmf" });
        let names: Vec<&str> = pointers.iter().map(|(n, _)| *n).collect();
        let program = self.programs.program(&name, &names)?;
        let raw: Vec<*mut c_void> = pointers.iter().map(|(_, p)| *p).collect();
        let stream = self.stream_of(rank);
        // SAFETY: every pointer names a live allocation of rank `rank`'s GPU sized for the rows
        // in `scalars`; that rank's stream orders all its launches.
        let launch = || self.on(rank, || unsafe { program.launch(&raw, scalars, stream) })
            .with_context(|| format!("{name} with {scalars:?}"));
        if rank == 0 { self.timed(&name, launch) } else { launch() }
    }

    /// Runs `body` (stream work) between two timing events when profiling ops.
    fn timed<T>(&self, label: &str, body: impl FnOnce() -> Result<T>) -> Result<T> {
        let Some(ops) = &self.ops else { return body() };
        let event = |ops: &mut OpTimes| -> Result<*mut c_void> {
            match ops.pool.pop() {
                Some(event) => Ok(event),
                None => self.library.cuda_event_create(),
            }
        };
        let (start, end) = {
            let mut ops = ops.borrow_mut();
            (event(&mut ops)?, event(&mut ops)?)
        };
        // SAFETY: both events are live timing events; the engine owns the stream.
        unsafe { self.library.cuda_event_record(start, self.stream)? };
        let out = body()?;
        // SAFETY: as above.
        unsafe { self.library.cuda_event_record(end, self.stream)? };
        ops.borrow_mut().pending.push((label.to_owned(), start, end));
        Ok(out)
    }

    /// Drains the stream and returns the per-label GPU milliseconds and
    /// launches accumulated since the last call (empty unless profiling ops).
    pub fn op_profile(&self) -> Result<std::collections::BTreeMap<String, (f64, usize)>> {
        let Some(ops) = &self.ops else { return Ok(Default::default()) };
        // SAFETY: the engine owns this stream.
        unsafe { self.library.cuda_stream_synchronize(self.stream)? };
        self.flush_ops()?;
        Ok(std::mem::take(&mut ops.borrow_mut().totals))
    }

    /// Adds host wall time since `since` under `label` when profiling ops.
    fn host_op(&self, label: &str, since: std::time::Instant) {
        if let Some(ops) = &self.ops {
            let mut ops = ops.borrow_mut();
            let entry = ops.totals.entry(label.to_owned()).or_default();
            entry.0 += since.elapsed().as_secs_f64() * 1e3;
            entry.1 += 1;
        }
    }

    fn flush_ops(&self) -> Result<()> {
        let Some(ops) = &self.ops else { return Ok(()) };
        let mut ops = ops.borrow_mut();
        let pending = std::mem::take(&mut ops.pending);
        for (label, start, end) in pending {
            // SAFETY: both events were recorded on the drained stream.
            let ms = unsafe { self.library.cuda_event_elapsed_ms(start, end)? };
            let entry = ops.totals.entry(label).or_default();
            entry.0 += f64::from(ms);
            entry.1 += 1;
            ops.pool.extend([start, end]);
        }
        Ok(())
    }

    fn scratch(&self, name: &str) -> Result<usize> {
        Ok(self.programs.spec(&format!("glmf_{name}"))?.scratch.get("scratch").copied().unwrap_or(0) as usize)
    }

    /// Every prefill workspace teacher-forced scoring runs in, before readiness (`--full-prefill-logits`
    /// admits them): both ranks' prefill lanes, as a Spark prefill takes them. A serial prefill, a
    /// short chunk and the prefix captures run in lane 0, so there is no third workspace, and the
    /// all-row logits live once, in the lanes' shared temporaries.
    pub fn prepare_scoring_prefill(&self) -> Result<()> {
        let lanes = if self.pipelined() { self.prefill_lane_count } else { 1 };
        for rank in 0..self.ranks() {
            drop(self.prefill_lanes_of(rank, lanes)?);
        }
        Ok(())
    }

    /// This engine's step plan: what its workspaces hold, given its configuration and experts.
    fn step_plan(&self) -> StepPlan<'_, 'a> {
        StepPlan::new(self.library, self.programs, &self.cfg, &self.weights.layers, self.experts.as_ref(),
            StepSettings { kda_fp32_partials: self.kda_fp32_partials, kda_output_shard: self.kda_output_shard,
                kda_prefill_expanded: self.kda_prefill_expanded, full_prefill_logits: self.full_prefill_logits,
                max_context: self.max_context, index_cache: self.index_cache, kda_state: self.kda_state,
                replay_records: self.replay_records, decode_rows: self.decode_rows })
    }

    /// Installs step workspaces allocated before this engine (`StepWorkspaces`), which must be
    /// what its step plan holds.
    pub(crate) fn install_workspaces(&self, workspaces: StepWorkspaces<'a>) -> Result<()> {
        ensure!(self.peer.is_none() && workspaces.key == self.step_plan().key() && workspaces.rows == self.prefill_rows,
            "step workspaces allocated for another plan");
        ensure!(self.decode_workspace.borrow().is_none() && self.lane_workspaces.borrow().is_empty(),
            "step workspaces already allocated");
        *self.decode_workspace.borrow_mut() = Some(workspaces.decode);
        *self.lane_workspaces.borrow_mut() = workspaces.lanes;
        Ok(())
    }

    /// The token selector made at start-up (eager admission), for the serving loop to take.
    pub(crate) fn set_selector(&self, selector: TokenSelector<'a>) {
        *self.selector.borrow_mut() = Some(selector);
    }

    pub(crate) fn take_selector(&self) -> Option<TokenSelector<'a>> {
        self.selector.borrow_mut().take()
    }

    /// Rank `rank`'s decode workspace (its own temporaries) of `decode_rows` rows, allocated on that
    /// rank's GPU on first use.
    fn decode_workspace_of(&self, rank: usize) -> Result<std::cell::Ref<'_, Option<Workspace<'a>>>> {
        let cell = match (rank, &self.peer) {
            (1, Some(peer)) => &peer.decode_workspace,
            _ => &self.decode_workspace,
        };
        if cell.borrow().is_none() {
            let workspace = self.on(rank, || {
                let plan = self.step_plan();
                plan.lane(rank, self.decode_rows, true, Rc::new(plan.temporaries(rank, self.decode_rows, true)?))
            })?;
            *cell.borrow_mut() = Some(workspace);
        }
        Ok(cell.borrow())
    }

    /// Rank `rank`'s prefill lanes, at least `count` of `prefill_rows` rows over one set of
    /// temporaries, allocated on that rank's GPU on first use.
    fn prefill_lanes_of(&self, rank: usize, count: usize) -> Result<std::cell::Ref<'_, Vec<Workspace<'a>>>> {
        let cell = match (rank, &self.peer) {
            (1, Some(peer)) => &peer.lane_workspaces,
            _ => {
                // Their scratch may hold the replay records: a commit after this sees it.
                self.records.prefilled();
                &self.lane_workspaces
            }
        };
        if cell.borrow().len() < count {
            let mut lanes = cell.borrow_mut();
            self.on(rank, || -> Result<()> {
                let plan = self.step_plan();
                let temps = match lanes.first() {
                    Some(lane) => lane.temps.clone(),
                    None => Rc::new(plan.temporaries(rank, self.prefill_rows, false)?),
                };
                while lanes.len() < count {
                    lanes.push(plan.lane(rank, self.prefill_rows, false, temps.clone())?);
                }
                Ok(())
            })?;
        }
        Ok(cell.borrow())
    }

    /// Streams start as four copies of each token's embedding. With the
    /// device table the ids go up and the rows are gathered into every stream
    /// slot (unless `defer_gather`: the decode graph's first segment gathers);
    /// otherwise the shard's rows go up once (into the second stream buffer)
    /// and are copied into each stream slot on the device.
    fn load_streams(&self, w: &Workspace<'_>, tokens: &[u32], defer_gather: bool) -> Result<()> {
        ensure!(!tokens.is_empty() && tokens.len() <= w.rows, "{} tokens exceed the workspace", tokens.len());
        if self.embedding.device_gather() {
            let host = std::time::Instant::now();
            self.embedding.check(tokens)?;
            self.put(&w.ids, tokens)?;
            self.host_op("host: token ids upload", host);
            if !defer_gather {
                self.timed("embedding gather", || self.gather_streams(w, tokens.len()))?;
            }
            return Ok(());
        }
        let embed = self.embedding.host_rows(tokens)?;
        let row = self.cfg.hidden * 2;
        let t = tokens.len();
        let host = std::time::Instant::now();
        let staged = cuteafd_ffi::CuteafdDeviceBuffer { bytes: embed.len(), ..w.streams[1].buffer };
        self.library.copy_h2d(staged, &embed)?;
        self.host_op("host: embedding upload", host);
        self.timed("stream expansion", || {
            for s in 0..HC {
                let dst = cuteafd_ffi::CuteafdDeviceBuffer {
                    // SAFETY: slot `s` of row 0 lies inside the [t, 4, H] stream buffer.
                    ptr: unsafe { w.streams[0].buffer.ptr.cast::<u8>().add(s * row) }.cast(),
                    bytes: w.streams[0].buffer.bytes - s * row,
                    ..w.streams[0].buffer
                };
                // SAFETY: both buffers are live workspace buffers of at least `t` pitched
                // rows; the upload above completed before the copies are queued.
                unsafe { self.library.copy_device_rows_async(dst, staged, row, t, HC * row, row, self.stream)? };
            }
            Ok(())
        })
    }

    /// The device table's rows of the `t` staged ids, four copies each, into stream buffer 0.
    fn gather_streams(&self, w: &Workspace<'_>, t: usize) -> Result<()> {
        // SAFETY: the ids are on the device (a completed copy) and the streams hold t x 4 rows.
        unsafe { self.embedding.gather(w.ids.buffer.ptr, std::ptr::null(), t, HC, std::ptr::null(),
            w.streams[0].buffer.ptr, self.stream) }
    }

    fn inject_media(&self, w: &Workspace<'_>, tables: &StepTables,
        media: Option<&cuteafd_engine::media::RequestMedia>) -> Result<()> {
        let Some(media) = media.filter(|m| !m.spans().is_empty()) else { return Ok(()); };
        let start = usize::try_from(*tables.positions.first().context("media positions")?)?;
        let mut chunk = cuteafd_engine::media::MediaChunk::default();
        media.write_chunk(start, start + tables.positions.len(), &mut chunk)?;
        if chunk.indices.is_empty() { return Ok(()); }
        // Reuse the second stream and id scratch before consumers. The adapter
        // drains injection before ids are restored; graph pointers/shapes stay fixed.
        self.library.embedding_injection()?.inject_host(&chunk.features, &chunk.indices,
            w.streams[1].buffer, w.ids.buffer, w.streams[0].buffer,
            tables.positions.len(), self.cfg.hidden, HC, self.stream)?;
        Ok(())
    }

    /// Greedy tokens of the first `rows` logits rows into `select`.
    fn select_greedy(&self, w: &Workspace<'_>, rows: usize) -> Result<()> {
        let vocab = self.cfg.vocab_size;
        // SAFETY: the logits rows and the select buffer (ids, then statuses) are live buffers of these shapes.
        unsafe {
            self.library.cuda_logits_greedy_f32_async(w.logits.buffer.ptr, rows, vocab, vocab, w.select.buffer.ptr,
                std::ptr::null_mut(), w.select.buffer.ptr.cast::<u8>().add(rows * 4).cast(), self.stream)
        }
    }

    /// The first `rows` logits rows as device logits (`greedy`: with the rows'
    /// selection from [`Self::select_greedy`]).
    fn device_logits(&self, w: &Workspace<'_>, rows: usize, greedy: bool) -> DeviceLogits {
        let vocab = self.cfg.vocab_size;
        DeviceLogits { ptr: w.logits.buffer.ptr, rows, vocab, stride: vocab, stream: self.stream,
            // SAFETY: the statuses follow the ids inside the rows x 8-byte select buffer.
            greedy: greedy.then(|| (w.select.buffer.ptr.cast_const(),
                unsafe { w.select.buffer.ptr.cast::<u8>().add(rows * 4) }.cast_const().cast())) }
    }

    fn put<T: Copy>(&self, dev: &Dev<'_>, values: &[T]) -> Result<()> {
        let bytes = bytes_of(values);
        ensure!(bytes.len() <= dev.buffer.bytes, "table exceeds its buffer");
        self.library.copy_h2d(cuteafd_ffi::CuteafdDeviceBuffer { bytes: bytes.len(), ..dev.buffer }, bytes)
    }

    fn download(&self, dev: &Dev<'_>, bytes: usize) -> Result<Vec<u8>> {
        ensure!(bytes <= dev.buffer.bytes, "download of {bytes} bytes exceeds {}-byte buffer", dev.buffer.bytes);
        // SAFETY: the engine owns this stream.
        unsafe { self.library.cuda_stream_synchronize(self.stream)? };
        let mut out = vec![0u8; bytes];
        self.library.copy_d2h(&mut out, cuteafd_ffi::CuteafdDeviceBuffer { bytes, ..dev.buffer })?;
        Ok(out)
    }

    /// Per-row positions, record and pool slots, and the pools each row sees.
    fn rows(&self, placement: &GlmfPlacement, positions: std::ops::Range<usize>, first: i32, tables: &mut StepTables)
        -> Result<()> {
        for position in positions {
            ensure!(position < self.max_context, "position {position} past the context {}", self.max_context);
            tables.positions.push(position as i64);
            tables.kv_slots.push(placement.record(position)?);
            tables.pool_slots.push(placement.pool_slot(position)?);
            tables.kda_slots.push(placement.slot);
            tables.seq_first.push(first);
            tables.cache_lengths.push(((position + 1) / KPOOL) as i32);
            tables.long |= position + 1 > self.cfg.dense_context();
            tables.pool_width = tables.pool_width.max((position + 1).div_ceil(POOL_PAGE_TOKENS));
        }
        Ok(())
    }

    /// Prefills a sequence from its length through every resident layer and
    /// returns the last row's logits when all layers are resident.
    /// `on_layer` receives each layer's output streams (BF16 [t, 4, hidden]).
    /// With `forced`, `forced(l)` (when it returns rows) replaces the streams
    /// after layer `l`, so each layer's comparison measures that layer alone.
    /// With `all_logits`, returns every row's logits instead of the last.
    pub fn prefill_forced(&self, placement: &mut GlmfPlacement, tokens: &[u32],
        on_layer: Option<&mut dyn FnMut(usize, &[u8]) -> Result<()>>,
        forced: Option<&dyn Fn(usize) -> Option<Vec<u8>>>, all_logits: bool) -> Result<Option<Vec<f32>>> {
        self.prefill_step(placement, tokens, on_layer, forced, all_logits, false, None)?
            .map(|logits| logits.into_host(self.library)).transpose()
    }

    /// [`Self::prefill`] leaving the last row's logits on the device.
    pub fn prefill_device(&self, placement: &mut GlmfPlacement, tokens: &[u32]) -> Result<Option<DeviceLogits>> {
        self.prefill_step(placement, tokens, None, None, false, true, None)?.map(StepLogits::device).transpose()
    }

    /// Media changes only the gathered streams, not token ids or KDA/MLA positions.
    pub fn prefill_media_device(&self, placement: &mut GlmfPlacement, tokens: &[u32],
        media: &cuteafd_engine::media::RequestMedia) -> Result<Option<DeviceLogits>> {
        self.prefill_step(placement, tokens, None, None, false, true, Some(media))?.map(StepLogits::device).transpose()
    }

    /// Keep ordered all-row logits and media injection on the admitted scoring path.
    pub(crate) fn prefill_scoring_media(&self, placement: &mut GlmfPlacement, tokens: &[u32],
        media: &cuteafd_engine::media::RequestMedia) -> Result<Option<crate::shared::probe::ScoreLogits>> {
        self.prefill_step(placement, tokens, None, None, true, false, Some(media))?
            .map(|logits| Ok(crate::shared::probe::ScoreLogits::Host {
                values: logits.into_host(self.library)?, vocab: self.cfg.vocab_size,
            })).transpose()
    }

    fn prefill_step(&self, placement: &mut GlmfPlacement, tokens: &[u32],
        on_layer: Option<&mut dyn FnMut(usize, &[u8]) -> Result<()>>,
        forced: Option<&dyn Fn(usize) -> Option<Vec<u8>>>, all_logits: bool, device: bool, media: Option<&cuteafd_engine::media::RequestMedia>)
        -> Result<Option<StepLogits>> {
        let (t, start) = (tokens.len(), placement.len);
        if self.split_audit && t >= 256 {
            tracing::info!(rows = t, start, token_hash = format_args!("{:016x}", audit_token_hash(tokens)),
                "GLM Flash split audit prefill");
        }
        if on_layer.is_none() && forced.is_none() && self.pipelined() {
            return self.prefill_lanes(placement, tokens, all_logits, device, media);
        }
        Ok(self.prefill_one(placement, tokens, on_layer, forced, all_logits, media)?.map(StepLogits::Device))
    }

    /// One serial prefill pass of `tokens` (at most `prefill_rows`) in the first lane's workspace.
    fn prefill_one(&self, placement: &mut GlmfPlacement, tokens: &[u32],
        on_layer: Option<&mut dyn FnMut(usize, &[u8]) -> Result<()>>,
        forced: Option<&dyn Fn(usize) -> Option<Vec<u8>>>, all_logits: bool,
        media: Option<&cuteafd_engine::media::RequestMedia>) -> Result<Option<DeviceLogits>> {
        let (t, start) = (tokens.len(), placement.len);
        ensure!(t > 0 && t <= self.prefill_rows && start + t <= self.max_context, "prefill of {t} rows at {start}");
        if start == 0 {
            self.start(placement)?;
        }
        let mut tables = StepTables { page_table: self.columns(&placement.pages, self.table_pages),
            pool_table: self.columns(&placement.pool_pages, self.table_pool_pages), ..Default::default() };
        self.rows(placement, start..start + t, 0, &mut tables)?;
        let logits = self.step(&tables, tokens, if all_logits { t } else { 1 }, on_layer, forced, None, media)?;
        placement.len += t;
        placement.kda_len = placement.len;
        Ok(logits)
    }

    /// A serial prefill pass whatever the lanes (the reference `--lane-check` holds lanes to):
    /// every row's logits with `all_logits`, else the last row's.
    pub(crate) fn prefill_serial(&self, placement: &mut GlmfPlacement, tokens: &[u32], all_logits: bool)
        -> Result<Option<Vec<f32>>> {
        self.prefill_one(placement, tokens, None, None, all_logits, None)?
            .map(|logits| logits.to_host(self.library)).transpose()
    }

    /// The rows of each lane a pipelined prefill of `tokens` rows runs in, in order.
    pub(crate) fn prefill_cuts(&self, tokens: usize) -> Result<Vec<usize>> {
        let (_, per_lane) = prefill_lane_plan(tokens, self.prefill_lane_count, self.prefill_rows)?;
        Ok((0..tokens).step_by(per_lane).map(|first| per_lane.min(tokens - first)).collect())
    }

    /// Whether prefill runs as Spark lanes (a transport per lane, every layer resident).
    /// CUTEAFD_GLMF_PREFILL_LANES=1 keeps the serial one-workspace prefill (A/B runs).
    fn pipelined(&self) -> bool {
        self.lanes && (self.weights.layers.len() == self.cfg.layers || self.subset_lanes) && matches!(&self.experts,
            Some(Experts::Spark { transports, .. }) if transports.borrow().len() >= self.prefill_lane_count)
    }

    /// Longest chunk one prefill call takes: a lane of `prefill_rows` rows
    /// each when Spark prefill is pipelined.
    pub fn prefill_capacity(&self) -> usize {
        if self.pipelined() { prefill_lane_capacity(self.prefill_lane_count, self.prefill_rows) } else { self.prefill_rows }
    }

    /// A Spark prefill chunk as up to `prefill_lane_count` lanes of consecutive
    /// rows (see [`Self::step_lanes`]).
    fn prefill_lanes(&self, placement: &mut GlmfPlacement, tokens: &[u32], all_logits: bool, device: bool, media: Option<&cuteafd_engine::media::RequestMedia>)
        -> Result<Option<StepLogits>> {
        let (start, t) = (placement.len, tokens.len());
        // Lanes split at a multiple of 64 rows (an MLA page), so each lane's
        // pools and pages start where the previous lane's end.
        let cuts = self.prefill_cuts(t)?;
        ensure!(t > 0 && cuts.iter().all(|&n| n <= self.prefill_rows) && start + t <= self.max_context,
            "prefill of {t} rows at {start} exceeds {} rows per lane or the context", self.prefill_rows);
        if start == 0 {
            self.start(placement)?;
        }
        let mut steps = Vec::new();
        let mut first = 0;
        for n in cuts {
            let mut tables = StepTables { page_table: self.columns(&placement.pages, self.table_pages),
                pool_table: self.columns(&placement.pool_pages, self.table_pool_pages), ..Default::default() };
            self.rows(placement, start + first..start + first + n, 0, &mut tables)?;
            steps.push((tables, &tokens[first..first + n]));
            first += n;
        }
        let logits = self.step_lanes(&steps, if all_logits { t } else { 1 }, device, media)?;
        placement.len += t;
        placement.kda_len = placement.len;
        Ok(logits)
    }

    /// What one packed prefill pass ([`Self::prefill_packed`]) may hold: the first prefill lane's
    /// rows, one sequence's table columns, and the drafter's tap rows.
    pub(crate) fn packed_limits(&self) -> packing::Limits {
        packing::Limits { rows: self.prefill_rows,
            chunk_rows: single_pass_rows(self.pipelined(), self.prefill_lane_count, self.prefill_rows),
            table_pages: self.table_pages, table_pool_pages: self.table_pool_pages,
            taps: self.drafter.as_ref().map(|_| crate::families::glm5::dflash::TAP_ROWS),
            dense_context: self.cfg.dense_context(), max_context: self.max_context }
    }

    /// Whether a packed prefill can run at all: one GPU and every layer resident.
    pub(crate) fn packs_prefill(&self) -> bool {
        self.peer.is_none() && self.weights.layers.len() == self.cfg.layers
    }

    /// One prefill pass over the next chunk of each of `sequences` (`packing`): their rows back to
    /// back in the first prefill lane's workspace, each sequence's mHC sites, router scores, KDA
    /// layers, DSA indexer, top-k and selection run over its rows alone, everything else over all
    /// rows, and one Spark wave per MoE layer. `on_logits(i, logits)` then receives sequence `i`'s
    /// last row's logits, in order (valid until it returns), and returns a [`PackedOutcome`].
    /// Once the step has run every placement advances, whatever the callbacks return. Returns the
    /// layout (each sequence's drafter tap rows) and each sequence's own outcome.
    pub fn prefill_packed(&self, sequences: &mut [(&mut GlmfPlacement, &[u32])],
        on_logits: &mut dyn FnMut(usize, DeviceLogits) -> PackedOutcome)
        -> Result<(Vec<packing::Segment>, Vec<Result<()>>)> {
        ensure!(self.packs_prefill(), "a packed prefill runs on one GPU with every layer");
        let chunks: Vec<(usize, usize)> = sequences.iter().map(|(p, tokens)| (p.len, tokens.len())).collect();
        let segments = packing::plan(&chunks, &self.packed_limits())?;
        let mut tables = StepTables::default();
        for ((placement, tokens), segment) in sequences.iter().zip(&segments) {
            if placement.len == 0 {
                self.start(placement)?;
            }
            // `seq_first` stays 0: every program that reads it runs per sequence.
            self.rows(placement, placement.len..placement.len + tokens.len(), 0, &mut tables)?;
            tables.page_table.extend_from_slice(placement.pages.get(..segment.page_columns)
                .context("a packed sequence's positions past its pages")?);
            tables.pool_table.extend_from_slice(placement.pool_pages.get(..segment.pool_columns)
                .context("a packed sequence's positions past its pool pages")?);
        }
        tables.segments = segments.clone();
        let tokens: Vec<u32> = sequences.iter().flat_map(|(_, tokens)| tokens.iter().copied()).collect();
        self.step(&tables, &tokens, 1, None, None, None, None)?;
        let lanes = self.prefill_lanes_of(0, 1)?;
        let w = lanes.first().context("prefill workspace")?;
        let outcomes = packed_tail(sequences, |i| {
            // The sequence's last normalized row through the head alone, as its own pass's.
            let timer = std::time::Instant::now();
            self.logits(w, segments[i].end_row(), 1, false)?;
            self.profile.borrow_mut()[2] += timer.elapsed().as_secs_f64();
            Ok(self.device_logits(w, 1, false))
        }, on_logits)?;
        Ok((segments, outcomes))
    }

    pub fn prefill(&self, placement: &mut GlmfPlacement, tokens: &[u32],
        on_layer: Option<&mut dyn FnMut(usize, &[u8]) -> Result<()>>) -> Result<Option<Vec<f32>>> {
        self.prefill_forced(placement, tokens, on_layer, None, false)
    }

    /// Appends each sequence's tokens (one for decode, several for a verify)
    /// at its length in one decode-shaped step; returns every row's logits.
    /// KDA state advances in place: a caller rejecting a suffix must replay
    /// (or verify with [`Self::verify_spec`] and commit what it keeps).
    pub fn verify(&self, sequences: &mut [(&mut GlmfPlacement, usize)], tokens: &[u32],
        on_layer: Option<&mut dyn FnMut(usize, &[u8]) -> Result<()>>) -> Result<Option<Vec<f32>>> {
        self.decode_step(sequences, tokens, on_layer, false, None, None, true)?.map(|l| l.to_host(self.library)).transpose()
    }

    pub fn verify_trace(&self, sequences: &mut [(&mut GlmfPlacement, usize)], tokens: &[u32],
        on_layer: &mut dyn FnMut(usize, &[u8]) -> Result<()>, dir: &std::path::Path) -> Result<Option<Vec<f32>>> {
        self.decode_step(sequences, tokens, Some(on_layer), false, Some(dir), None, true)?
            .map(|l| l.to_host(self.library)).transpose()
    }

    /// Diagnostic snapshot inside a verify's layer callback. The callback
    /// already disables graphs and synchronizes the layer's output download.
    /// Router buffers still belong to this layer, even though mHC has prepared
    /// the next attention input. Ordinary serving never calls this method.
    pub fn trace_decode_layer(&self, layer: usize, rows: usize, streams: &[u8], dir: &std::path::Path)
        -> Result<()> {
        std::fs::create_dir_all(dir)?;
        std::fs::write(dir.join("streams.bin"), streams)?;
        let workspace = self.decode_workspace.borrow();
        let w = workspace.as_ref().context("decode trace without a workspace")?;
        for (name, buffer, bytes) in [("ffn.bin", &w.delta, rows * self.cfg.hidden * 2),
            ("next_input.bin", &w.x, rows * self.cfg.hidden * 2)] {
            std::fs::write(dir.join(name), self.download(buffer, bytes)?)?;
        }
        if !self.weights.layers[layer].dense {
            for (name, buffer, bytes) in [("route_ids.bin", &w.route_ids, rows * self.cfg.topk * 4),
                ("route_weights.bin", &w.route_weights, rows * self.cfg.topk * 4),
                ("router_logits.bin", &w.router_logits, rows * self.cfg.experts * 4),
                ("wire.bin", &w.wire, rows * (self.cfg.hidden + self.cfg.hidden / 32)),
                ("shared.bin", &w.shared, rows * self.cfg.hidden * 2)] {
                std::fs::write(dir.join(name), self.download(buffer, bytes)?)?;
            }
        }
        Ok(())
    }

    /// A speculative verify: as [`Self::verify`], but the KDA state stays at
    /// every sequence's start and each row's replay inputs are recorded; the
    /// caller then passes every sequence's kept rows to [`Self::commit`]
    /// (MLA records past a sequence's kept length are rewritten by later
    /// steps). Placements advance by all rows; callers set the kept length.
    pub fn verify_spec(&self, sequences: &mut [(&mut GlmfPlacement, usize)], tokens: &[u32])
        -> Result<Option<Vec<f32>>> {
        self.decode_step(sequences, tokens, None, true, None, None, true)?.map(|l| l.to_host(self.library)).transpose()
    }

    /// Physical row count of an ordinary serving verify (lazy graphs use exact rows).
    pub(crate) fn serving_decode_rows(&self, rows: usize, spec: bool) -> usize {
        if self.use_graphs && self.startup_graphs { self.buckets.bucket(rows, spec) } else { rows }
    }

    /// [`Self::verify`] (`spec`: [`Self::verify_spec`]) leaving every row's
    /// logits on the device, with the decode graph's greedy selection of them.
    pub fn verify_device(&self, sequences: &mut [(&mut GlmfPlacement, usize)], tokens: &[u32], spec: bool)
        -> Result<Option<DeviceLogits>> {
        self.decode_step(sequences, tokens, None, spec, None, None, false)
    }

    /// Teacher-forced scoring may append image rows with decode geometry.
    pub fn verify_media_device(&self, sequences: &mut [(&mut GlmfPlacement, usize)], tokens: &[u32],
        media: &cuteafd_engine::media::RequestMedia) -> Result<Option<DeviceLogits>> {
        ensure!(sequences.len() == 1, "media scoring needs one sequence");
        self.decode_step(sequences, tokens, None, false, None, Some(media), true)
    }

    fn decode_step(&self, sequences: &mut [(&mut GlmfPlacement, usize)], tokens: &[u32],
        on_layer: Option<&mut dyn FnMut(usize, &[u8]) -> Result<()>>, spec: bool,
        trace: Option<&std::path::Path>, media: Option<&cuteafd_engine::media::RequestMedia>, eager: bool) -> Result<Option<DeviceLogits>> {
        let rows: usize = sequences.iter().map(|(_, n)| n).sum();
        ensure!(rows > 0 && rows <= self.decode_rows && tokens.len() == rows,
            "decode step of {rows} rows (--decode-rows {})", self.decode_rows);
        // Power-of-two strides and widths bound the graphs a growing batch captures.
        let live_tokens = sequences.iter().map(|(p, count)| p.len + count).max().unwrap_or(1);
        let allocated = |n, unit| crate::shared::context::decode_allocation_units(n, live_tokens, unit);
        let (page_stride, pool_stride) = decode_strides(
            allocated(sequences.iter().map(|(p, _)| p.pages.len()).max().unwrap_or(1), PAGE_ROWS),
            allocated(sequences.iter().map(|(p, _)| p.pool_pages.len()).max().unwrap_or(1), UNIT_ROWS),
            (self.pages, self.pool_pages), (self.table_pages, self.table_pool_pages));
        let mut tables = StepTables { decode: true, real_rows: rows, eager: eager || media.is_some() || trace.is_some() || on_layer.is_some(),
            page_stride, pool_stride, spec, ..Default::default() };
        for (placement, count) in sequences.iter() {
            if placement.len == 0 {
                self.start(placement)?;
            }
            let first = tables.kv_slots.len() as i32;
            self.rows(placement, placement.len..placement.len + count, first, &mut tables)?;
            for _ in 0..*count {
                let mut pages = placement.pages.clone();
                pages.resize(page_stride, 0);
                tables.page_table.extend(pages);
                let mut pools = placement.pool_pages.clone();
                pools.resize(pool_stride, 0);
                tables.pool_table.extend(pools);
            }
        }
        tables.pool_width = tables.pool_width.next_power_of_two().min(pool_stride);
        if spec {
            // The commit that follows reads this step's records.
            self.records.recorded();
        }
        let graphed = self.use_graphs && !tables.eager && on_layer.is_none() && trace.is_none()
            && media.is_none_or(|m| m.spans().is_empty());
        let bucket = if graphed { self.serving_decode_rows(rows, spec) } else { rows };
        if spec {
            // The step records at its programs' capacity; the commit that follows reads that many rows.
            self.replay_rows.set(replay_rows_of(decode_cap(bucket)));
        }
        let logits = if graphed {
            pad_decode_tables(&mut tables, bucket);
            let mut padded = tokens.to_vec();
            padded.resize(bucket, 0);
            self.step(&tables, &padded, bucket, None, None, None, None)?.map(|mut logits| {
                // Keep the bucket-sized greedy status offset while exposing only real rows.
                logits.rows = rows;
                logits
            })
        } else {
            self.step(&tables, tokens, rows, on_layer, None, trace, media)?
        };
        for (placement, count) in sequences.iter_mut() {
            placement.len += *count;
            if !spec {
                placement.kda_len = placement.len;
            }
        }
        Ok(logits)
    }

    /// A sequence's pages as one table row of `columns` columns: positions past `max_context`
    /// are never stepped, so pages past the table's columns are never read.
    fn columns(&self, pages: &[i32], columns: usize) -> Vec<i32> {
        pages[..pages.len().min(columns)].to_vec()
    }

    /// Writes a step's tables into `w` (on the current device).
    fn put_tables(&self, w: &Workspace<'_>, tables: &StepTables) -> Result<()> {
        self.put(&w.positions, &tables.positions)?;
        self.put(&w.kv_slots, &tables.kv_slots)?;
        self.put(&w.kda_slots, &tables.kda_slots)?;
        self.put(&w.seq_first, &tables.seq_first)?;
        self.put(&w.pool_slots, &tables.pool_slots)?;
        self.put(&w.cache_lengths, &tables.cache_lengths)?;
        self.put(&w.page_table, &tables.page_table)?;
        self.put(&w.pool_table, &tables.pool_table)
    }

    /// Writes a step's tables into rank 1's `w1` after its stream drained (every wait on it is
    /// matched by a push rank 0 queued earlier, so it drains).
    fn peer_tables(&self, w1: &Workspace<'_>, tables: &StepTables) -> Result<()> {
        // SAFETY: the engine owns the peer stream.
        self.on(1, || unsafe { self.library.cuda_stream_synchronize(self.stream_of(1)) })?;
        self.on(1, || self.put_tables(w1, tables))
    }

    /// Rank 1's workspaces (the decode one, or `lanes` prefill lanes: a serial prefill runs in the
    /// first), created on first use; None without a head split.
    fn peer_workspaces(&self, decode: bool, lanes: Option<usize>) -> Result<Option<PeerWorkspaces<'_, 'a>>> {
        if self.peer.is_none() {
            return Ok(None);
        }
        Ok(Some(match lanes {
            None if decode => PeerWorkspaces::One(self.decode_workspace_of(1)?),
            lanes => PeerWorkspaces::Lanes(self.prefill_lanes_of(1, lanes.unwrap_or(1))?),
        }))
    }

    fn precise_attention(&self, layer: &GlmfLayer<'_>) -> bool {
        self.kda_fp32_partials && layer.split &&
            layer.attention == GlmNextAttention::Kda && layer.has("w_in_fp8")
    }

    fn output_shard_attention(&self, layer: &GlmfLayer<'_>) -> bool {
        self.kda_output_shard && layer.split &&
            layer.attention == GlmNextAttention::Kda && layer.has("w_in_fp8")
    }

    fn partial_bytes(&self) -> usize {
        if self.kda_fp32_partials { 4 } else { 2 }
    }

    fn full_kda_norm(&self, w: &Workspace<'_>) -> *mut c_void {
        w.scratch.buffer.ptr.wrapping_byte_add(w.scratch.buffer.bytes - w.rows * self.cfg.hidden * 4)
    }

    /// Share the missing heads only for each rank's output token rows, project
    /// those rows with the full K reduction, then share the finished rows.
    /// Separate slots keep norm inputs live through both joins. Each token is
    /// projected once and each output is rounded once.
    fn complete_output_shard(&self, rank: usize, w: &Workspace<'_>, layer: &GlmfLayer<'_>,
        output_slot: usize, t: usize, cap: &str) -> Result<*mut c_void> {
        let exchange = self.exchange()?;
        let h = self.cfg.hidden;
        let (first, owned) = output_rows(t, rank);
        let sent_first = if rank == 0 { owned } else { 0 };
        let heads_slot = norm_slot(self.prefill_lane_count, output_slot);
        exchange.push(rank, heads_slot, w.delta.buffer.ptr.wrapping_byte_add(sent_first * h * 2),
            (t - owned) * h * 2)?;
        exchange.wait(rank, heads_slot)?;
        let peer_norm = exchange.recv(rank, heads_slot)?;
        let norm = w.delta.buffer.ptr.wrapping_byte_add(first * h * 2);
        let (a, b) = if rank == 0 { (norm, peer_norm) } else { (peer_norm, norm) };
        let full = self.full_kda_norm(w);
        let expanded = if self.kda_prefill_expanded && !is_decode(cap) { "_expanded" } else { "" };
        if owned != 0 {
            self.run_on(rank, true, "join_heads", &[("a", a), ("b", b), ("out", full)],
                &[Scalar::I32(owned as i32)])?;
            // Select the projection route from the global batch width, so a
            // 64-row verification split into 32 + 32 retains the full-head TMA math.
            self.run_on(rank, true, &format!("kda_output_rows{expanded}_{cap}"),
                &[("x", full), ("w_fp8", layer.ptr("w_o_fp8")?),
                ("w_kscale", layer.ptr("w_o_kscale")?), ("out", w.delta.buffer.ptr),
                ("scratch", w.scratch.buffer.ptr)],
                &[Scalar::I32(owned as i32), Scalar::I32(t as i32)])?;
        }
        // Zero-owned ranks still publish: both peers advance each slot's sequence.
        exchange.push(rank, output_slot, w.delta.buffer.ptr, owned * h * 2)?;
        exchange.wait(rank, output_slot)?;
        let peer_output = exchange.recv(rank, output_slot)?;
        let (a, b) = if rank == 0 { (w.delta.buffer.ptr, peer_output) } else { (peer_output, w.delta.buffer.ptr) };
        self.run_on(rank, true, "join_rows",
            &[("a", a), ("b", b), ("out", w.sum_ptr()?)],
            &[Scalar::I32(t.div_ceil(2) as i32), Scalar::I32((t / 2) as i32)])?;
        w.sum_ptr()
    }

    /// Rank 0's attention partial out, rank 1's in, their sum into `sum`.
    /// Without a head split, returns `delta` itself.
    fn meet_attention(&self, w: &Workspace<'_>, slot: usize, t: usize, layer: &GlmfLayer<'_>, cap: &str) -> Result<*mut c_void> {
        let Some(exchange) = &self.exchange else { return Ok(w.delta.buffer.ptr) };
        let (h, precise) = (self.cfg.hidden, self.precise_attention(layer));
        if self.output_shard_attention(layer) {
            return self.complete_output_shard(0, w, layer, slot, t, cap);
        } else if precise {
            // Both FP32 copies overlap. Sum in rank order on both GPUs and
            // round once, preserving the unsplit projection's output precision.
            exchange.push(0, slot, w.delta.buffer.ptr, t * h * 4)?;
            exchange.wait(0, slot)?;
            self.run_on(0, true, "add_fp32",
                &[("a", w.delta.buffer.ptr), ("b", exchange.recv(0, slot)?),
                ("out", w.sum_ptr()?)], &[Scalar::I32(t as i32)])?;
        } else {
            exchange.push(0, slot, w.delta.buffer.ptr, t * h * 2)?;
            exchange.wait(0, slot)?;
            exchange.add(0, w.delta.buffer.ptr, exchange.recv(0, slot)?, w.sum_ptr()?, t * h)?;
        }
        w.sum_ptr()
    }

    /// Rank 0's side of layer `index`'s FFN exchange (lane `lane`): its FFN output in `delta`
    /// (its dense partial, or the routed + shared-half sum) out unless this is the last of
    /// `layers` (rank 1 stops there), rank 1's dense partial or shared-expert half in, their
    /// sum into `sum`. Without a head split, `delta` itself.
    fn meet_ffn(&self, w: &Workspace<'_>, index: usize, lane: usize, layers: usize, t: usize) -> Result<*mut c_void> {
        let Some(exchange) = &self.exchange else { return Ok(w.delta.buffer.ptr) };
        let (h, slot) = (self.cfg.hidden, slot(index, true, lane));
        if index + 1 < layers {
            exchange.push(0, slot, w.delta.buffer.ptr, t * h * 2)?;
        }
        exchange.wait(0, slot)?;
        exchange.add(0, w.delta.buffer.ptr, exchange.recv(0, slot)?, w.sum_ptr()?, t * h)?;
        w.sum_ptr()
    }

    /// Rank 1's FFN output buffer of `layer`: its dense partial (`delta`; zero rows when rank
    /// 0 runs the MLP whole) or shared-expert half (`shared`).
    fn peer_ffn_out(w1: &Workspace<'_>, layer: &GlmfLayer<'_>) -> Result<*mut c_void> {
        match (layer.dense, layer.has("w_gate_up_fp8")) {
            (true, true) => Ok(w1.delta.buffer.ptr),
            (true, false) => w1.zero_ptr(),
            (false, _) => Ok(w1.shared.buffer.ptr),
        }
    }

    /// Rank 1's attention half of unit (`index`, `lane`): (layer 0: rank 0's streams in and the
    /// attention-site collapse), its heads' attention, the attention all-reduce (rank 0's
    /// operand order) and the FFN-site collapse, then its dense partial or shared-expert half,
    /// pushed to rank 0.
    fn peer_attention(&self, index: usize, lane: usize, w1: &Workspace<'_>, t: usize, cap: &str, tables: &StepTables)
        -> Result<()> {
        let (peer, exchange) = (self.peer()?, self.exchange()?);
        let layer = &peer.layers[index];
        let (h, rows) = (self.cfg.hidden, Scalar::I32(t as i32));
        if index == 0 {
            exchange.wait(1, DIRECT)?;
            self.pre_on(1, w1, &w1.streams[0], layer, rows)?;
        }
        self.attention(1, w1, index, layer, rows, cap, tables, None)?;
        let attended = slot(index, false, lane);
        let precise = self.precise_attention(layer);
        let sum = if self.output_shard_attention(layer) {
            self.complete_output_shard(1, w1, layer, attended, t, cap)?
        } else {
            exchange.push(1, attended, w1.delta.buffer.ptr, t * h * if precise { 4 } else { 2 })?;
            exchange.wait(1, attended)?;
            if precise {
                self.run_on(1, true, "add_fp32",
                    &[("a", exchange.recv(1, attended)?), ("b", w1.delta.buffer.ptr),
                    ("out", w1.sum_ptr()?)], &[rows])?;
            } else {
                exchange.add(1, exchange.recv(1, attended)?, w1.delta.buffer.ptr, w1.sum_ptr()?, t * h)?;
            }
            w1.sum_ptr()?
        };
        self.post_pre_on(1, w1, sum, 0, layer, "ffn", "post_norm", rows, cap)?;
        let out = Self::peer_ffn_out(w1, layer)?;
        // A dense MLP rank 0 runs whole leaves rank 1's zero rows as its partial.
        match (layer.dense, layer.has("w_gate_up_fp8")) {
            (true, false) => {}
            (true, true) => self.ffn_on(1, w1, layer, self.cfg.dense_intermediate, cap, out, rows)?,
            (false, _) => self.ffn_on(1, w1, layer, self.cfg.moe_intermediate, cap, out, rows)?,
        }
        exchange.push(1, slot(index, true, lane), out, t * h * 2)
    }

    /// Rank 1's FFN exchange of unit (`index`, `lane`): rank 0's dense partial or routed +
    /// shared sum in, summed with its own half in rank 0's operand order, then the next
    /// layer's attention-site collapse (nothing after the last layer).
    fn peer_post(&self, index: usize, lane: usize, w1: &Workspace<'_>, t: usize, cap: &str) -> Result<()> {
        let (peer, exchange) = (self.peer()?, self.exchange()?);
        let Some(next) = peer.layers.get(index + 1) else { return Ok(()) };
        let ffn = slot(index, true, lane);
        exchange.wait(1, ffn)?;
        exchange.add(1, exchange.recv(1, ffn)?, Self::peer_ffn_out(w1, &peer.layers[index])?, w1.sum_ptr()?,
            t * self.cfg.hidden)?;
        self.post_pre_on(1, w1, w1.sum_ptr()?, 1, next, "attn", "input_norm", Scalar::I32(t as i32), cap)
    }

    /// Rank 1's decode segment of layer `index` (see [`Self::decode_graphed`]): the previous
    /// layer's FFN exchange and this layer's attention-site collapse (layer 0: rank 0's
    /// streams), its attention half and FFN half.
    fn peer_segment(&self, index: usize, w1: &Workspace<'_>, t: usize, tables: &StepTables) -> Result<()> {
        let key = GraphKey::new(index, t, tables);
        self.replay_on(1, key, || {
            if let Some(previous) = index.checked_sub(1) {
                self.peer_post(previous, 0, w1, t, decode_cap(t))?;
            }
            self.peer_attention(index, 0, w1, t, decode_cap(t), tables)
        })
    }

    fn step(&self, tables: &StepTables, tokens: &[u32], logit_rows: usize,
        mut on_layer: Option<&mut dyn FnMut(usize, &[u8]) -> Result<()>>,
        forced: Option<&dyn Fn(usize) -> Option<Vec<u8>>>, trace: Option<&std::path::Path>, media: Option<&cuteafd_engine::media::RequestMedia>) -> Result<Option<DeviceLogits>> {
        let (h, t) = (self.cfg.hidden, tables.kv_slots.len());
        // Decode runs in its own workspace, a serial prefill in the first prefill lane.
        let (decode_workspace, prefill_lanes);
        let w = if tables.decode {
            decode_workspace = self.decode_workspace_of(0)?;
            decode_workspace.as_ref().context("decode workspace")?
        } else {
            prefill_lanes = self.prefill_lanes_of(0, 1)?;
            prefill_lanes.first().context("prefill workspace")?
        };
        ensure!(t <= w.rows && logit_rows <= t, "step exceeds the workspace");
        ensure!(tables.decode || self.full_prefill_logits || logit_rows <= DECODE_ROWS,
            "prefill logits past {DECODE_ROWS} rows need full_prefill_logits");
        self.put_tables(w, tables)?;
        // The head split's second GPU: its workspace of the same shape and the same tables.
        let peer_workspaces = self.peer_workspaces(tables.decode, None)?;
        let w1 = peer_workspaces.as_ref().map(|p| p.get(0)).transpose()?;
        if let Some(w1) = w1 {
            ensure!(forced.is_none(), "teacher-forced prefill runs without a head split");
            self.peer_tables(w1, tables)?;
        }
        let row = h * 2;
        ensure!(tokens.len() == t, "{} tokens for a {t}-row step", tokens.len());
        let graphed = self.use_graphs && tables.decode && !tables.eager && on_layer.is_none() && forced.is_none()
            && media.is_none_or(|m| m.spans().is_empty());
        self.load_streams(w, tokens, graphed)?;
        self.inject_media(w, tables, media)?;
        if media.is_some_and(|m| !m.spans().is_empty()) { self.put(&w.ids, tokens)?; }
        let rows = Scalar::I32(t as i32);
        if graphed {
            return self.decode_graphed(w, w1, tables, t, rows, logit_rows);
        }
        let cap = if tables.decode { decode_cap(t) } else { PREFILL_CAP };
        let layers = &self.weights.layers;
        // The mHC sites and the router scores run per sequence in a packed prefill (one span: the
        // whole step).
        let spans = tables.spans();
        ensure!(w1.is_none() || tables.segments.is_empty(), "a packed prefill runs on one GPU");
        if let Some(w1) = w1 {
            // The streams to the second GPU, which runs a unit ahead of the host's rank-0
            // work (all its inputs are pushes from rank 0).
            self.exchange()?.push_to(0, DIRECT, w.streams[0].buffer.ptr, w1.streams[0].buffer.ptr, t * HC * row)?;
            self.peer_attention(0, 0, w1, t, cap, tables)?;
        }
        let mut cur = 0usize;
        for span in &spans {
            self.pre_at(0, w, &w.streams[cur], &layers[0], span.first, span.rows())?;
        }
        for (index, layer) in layers.iter().enumerate() {
            if let Some(dir) = trace {
                let dir = dir.join(format!("layer{index:02}"));
                std::fs::create_dir_all(&dir)?;
                for (name, buffer, bytes) in [("attention_input.bin", &w.x, t * h * 2),
                    ("attention_post.bin", &w.post, t * HC * 4), ("attention_comb.bin", &w.comb, t * HC * HC * 4)] {
                    std::fs::write(dir.join(name), self.download(buffer, bytes)?)?;
                }
            }
            self.attention(0, w, index, layer, rows, cap, tables,
                trace.map(|dir| dir.join(format!("layer{index:02}"))).as_deref())?;
            if let Some(dir) = trace {
                let dir = dir.join(format!("layer{index:02}"));
                let dtype = if self.precise_attention(layer) { "float32" } else { "bfloat16" };
                std::fs::write(dir.join("attention.bin"), self.download(&w.delta,
                    t * h * if dtype == "float32" { 4 } else { 2 })?)?;
                std::fs::write(dir.join("attention_meta.json"), serde_json::to_vec(&serde_json::json!({
                    "rows": t, "hidden": h, "dtype": dtype,
                    "kind": if self.output_shard_attention(layer) { "normalized_heads" } else { "projection" } }))?)?;
                if layer.attention == GlmNextAttention::Kda {
                    let d = self.caches.kda_heads * self.cfg.kda_head_dim;
                    // The in-projection's output width (q|k|v, f_a, g_a, b), whatever its weight format.
                    let p = 3 * d + 2 * self.cfg.kda_head_dim + self.caches.kda_heads;
                    // Decode KDA AOT layout: BF16 in-projection, f|gate,
                    // convolved q|k|v, recurrent output and gated-norm output;
                    // each region is 1024-byte aligned in the pinned manifest.
                    let bytes: usize = [t * p * 2, t * 4 * d, t * 6 * d, t * 2 * d, t * 2 * d]
                        .iter().map(|n| n.next_multiple_of(1024)).sum();
                    std::fs::write(dir.join("kda_scratch.bin"), self.download(&w.scratch, bytes)?)?;
                    std::fs::write(dir.join("kda_meta.json"), serde_json::to_vec(&serde_json::json!({
                        "rows": t, "width": d, "in_width": p, "alignment": 1024 }))?)?;
                }
            }
            // Attention back into the streams (a head split: the two partials' sum), then the
            // FFN site's collapse + norm.
            let attended = self.meet_attention(w, slot(index, false, 0), t, layer, cap)?;
            for span in &spans {
                self.post_pre_at(0, w, attended, cur, layer, "ffn", "post_norm", span.first, span.rows(), cap)?;
            }
            cur ^= 1;
            if let Some(w1) = w1 {
                // Rank 1: this layer's FFN exchange, then the next layer's attention.
                self.peer_post(index, 0, w1, t, cap)?;
                if index + 1 < layers.len() {
                    self.peer_attention(index + 1, 0, w1, t, cap, tables)?;
                }
            }
            if let Some(dir) = trace {
                std::fs::write(dir.join(format!("layer{index:02}/ffn_input.bin")), self.download(&w.x, t * h * 2)?)?;
            }
            if layer.dense {
                self.ffn(w, layer, self.cfg.dense_intermediate, cap, w.delta.buffer.ptr, rows)?;
            } else {
                self.moe(w, index, layer, t, rows, cap, tables.decode, &spans)?;
            }
            let out = self.meet_ffn(w, index, 0, layers.len(), t)?;
            match layers.get(index + 1) {
                Some(next) => {
                    for span in &spans {
                        self.post_pre_at(0, w, out, cur, next, "attn", "input_norm", span.first, span.rows(), cap)?;
                    }
                    cur ^= 1;
                }
                None => {
                    for span in &spans {
                        self.mhc_post_at(w, out, cur, span)?;
                    }
                    cur ^= 1;
                }
            }
            if let Some(drafter) = &self.drafter {
                if tables.segments.is_empty() {
                    let n = t.min(crate::families::glm5::dflash::TAP_ROWS);
                    drafter.tap_streams(index, w.streams[cur].buffer.ptr, HC, t - n, n)?;
                } else {
                    // Each packed sequence's last rows, as its own pass taps them, at its tap rows.
                    for s in tables.segments.iter().filter(|s| s.tap_rows > 0) {
                        drafter.tap_streams_at(index, w.streams[cur].buffer.ptr, HC, s.end_row() - s.tap_rows,
                            s.tap_rows, s.tap_offset)?;
                    }
                }
            }
            if let Some(on_layer) = on_layer.as_mut() {
                on_layer(index, &self.download(&w.streams[cur], t * HC * row)?)?;
            }
            if let Some(rows_forced) = forced.and_then(|f| f(index)) {
                ensure!(rows_forced.len() == t * HC * row, "teacher-forced streams of the wrong size");
                self.library.copy_h2d(cuteafd_ffi::CuteafdDeviceBuffer { bytes: rows_forced.len(),
                    ..w.streams[cur].buffer }, &rows_forced)?;
                if let Some(next) = layers.get(index + 1) {
                    self.pre(w, &w.streams[cur], next, rows)?;
                }
            }
            crate::shared::console::layer_mark(index);
            self.probe_mark(index);
        }
        if layers.len() < self.cfg.layers {
            self.synchronize()?;
            return Ok(None);
        }
        let timer = std::time::Instant::now();
        self.run("head", &[("streams", w.streams[cur].buffer.ptr), ("weight", self.weights.norm.buffer.ptr),
            ("out", w.x.buffer.ptr)], &[rows])?;
        if !tables.segments.is_empty() {
            // A packed prefill: the caller takes each sequence's last row's logits in turn
            // ([`Self::prefill_packed`]) from the normalized rows.
            self.profile.borrow_mut()[2] += timer.elapsed().as_secs_f64();
            return Ok(None);
        }
        self.logits(w, t, logit_rows, tables.decode)?;
        self.profile.borrow_mut()[2] += timer.elapsed().as_secs_f64();
        Ok(Some(self.device_logits(w, logit_rows, false)))
    }

    /// The last layer's mHC post of `span`'s rows: the FFN output `x` back into streams `cur`, into
    /// the other stream buffer.
    fn mhc_post_at(&self, w: &Workspace<'_>, x: *mut c_void, cur: usize, span: &Span) -> Result<()> {
        let (h, first) = (self.cfg.hidden, span.first);
        ensure!(first == 0 || self.peer.is_none(), "per-sequence rows run on one GPU");
        self.run("mhc_post", &[("x", x.wrapping_byte_add(first * h * 2)),
            ("residual", row_at(&w.streams[cur], first, HC * h * 2)), ("prev_post", row_at(&w.post, first, HC * 4)),
            ("prev_comb", row_at(&w.comb, first, HC * HC * 4)),
            ("out", row_at(&w.streams[cur ^ 1], first, HC * h * 2))], &[span.rows()])
    }

    /// A decode step as captured segments: segment `i` posts layer `i - 1`'s
    /// FFN output into the streams with layer `i`'s attention-site collapse,
    /// then runs layer `i` up to its routed experts, which run (local or on
    /// the Sparks) between segments. Streams start and end in buffer 0.
    ///
    /// Under a head split each segment exchanges with rank 1's segment of the same layer
    /// (captured on rank 1's stream); rank 1's next segment is queued before the host waits in
    /// this layer's expert exchange.
    fn decode_graphed(&self, w: &Workspace<'_>, w1: Option<&Workspace<'_>>, tables: &StepTables, t: usize, rows: Scalar,
        logit_rows: usize) -> Result<Option<DeviceLogits>> {
        self.release_retired_graphs()?;
        let layers = &self.weights.layers;
        // Every layer resident: the last segment ends in the head and the greedy selection.
        let head = layers.len() == self.cfg.layers && logit_rows == t;
        let gather = self.embedding.device_gather();
        // The `_m64` programs up to 64 rows, the wide `_m128` ones past them (the rows are in the
        // graph key, so one capacity per captured segment).
        let cap = decode_cap(t);
        for index in 0..=layers.len() {
            let key = GraphKey::new(index, t, tables);
            self.replay(key, || -> Result<()> {
                // Layer `index - 1`'s output streams land in buffer 0 first thing.
                let tap = || match (&self.drafter, index.checked_sub(1)) {
                    (Some(drafter), Some(previous)) => drafter.tap_streams(previous, w.streams[0].buffer.ptr, HC, 0, t),
                    _ => Ok(()),
                };
                // Layer `index - 1`'s FFN output (a head split: with rank 1's half).
                let out = match index.checked_sub(1) {
                    Some(previous) => self.meet_ffn(w, previous, 0, layers.len(), t)?,
                    None => w.delta.buffer.ptr,
                };
                let Some(layer) = layers.get(index) else {
                    self.run("mhc_post", &[("x", out), ("residual", w.streams[1].buffer.ptr),
                        ("prev_post", w.post.buffer.ptr), ("prev_comb", w.comb.buffer.ptr),
                        ("out", w.streams[0].buffer.ptr)], &[rows])?;
                    tap()?;
                    if head {
                        self.run("head", &[("streams", w.streams[0].buffer.ptr),
                            ("weight", self.weights.norm.buffer.ptr), ("out", w.x.buffer.ptr)], &[rows])?;
                        self.logits(w, t, t, true)?;
                        self.select_greedy(w, t)?;
                    }
                    return Ok(());
                };
                if index == 0 {
                    if gather {
                        self.gather_streams(w, t)?;
                    }
                    if let Some(w1) = w1 {
                        self.exchange()?.push_to(0, DIRECT, w.streams[0].buffer.ptr, w1.streams[0].buffer.ptr,
                            t * HC * self.cfg.hidden * 2)?;
                    }
                    self.pre(w, &w.streams[0], layer, rows)?;
                } else {
                    self.post_pre_on(0, w, out, 1, layer, "attn", "input_norm", rows, cap)?;
                    tap()?;
                }
                self.attention(0, w, index, layer, rows, cap, tables, None)?;
                let attended = self.meet_attention(w, slot(index, false, 0), t, layer, cap)?;
                self.post_pre_on(0, w, attended, 0, layer, "ffn", "post_norm", rows, cap)?;
                if layer.dense {
                    self.ffn(w, layer, self.cfg.dense_intermediate, cap, w.delta.buffer.ptr, rows)
                } else if self.startup_graphs {
                    // Real width is runtime data, not part of a bucket graph key.
                    Ok(())
                } else {
                    self.moe_front(w, index, layer, t, rows, cap, &[Span { first: 0, rows: t }])
                }
            })?;
            if let Some(w1) = w1 {
                // Rank 1's segments: layer 0 with rank 0's first, then each next one before the
                // host waits in this layer's expert exchange.
                if index == 0 && !layers.is_empty() {
                    self.peer_segment(0, w1, t, tables)?;
                }
                if index + 1 < layers.len() {
                    // Rank 1's next weights into L2 while it waits for this layer's exchange.
                    if let (false, Some(l2)) = (layers[index].dense, self.peer()?.l2.as_ref()) {
                        self.on(1, || l2.issue(self.library, index, self.stream_of(1)))?;
                    }
                    self.peer_segment(index + 1, w1, t, tables)?;
                }
            }
            if layers.get(index).is_some_and(|layer| !layer.dense) {
                if self.warming_graphs.get() {
                    // Masked startup rows need no routed result; preserve peer event ordering.
                    self.exchange_window(index, true, true)?;
                } else if self.startup_graphs {
                    // A padded step's real rows run the router, the expert wire and the shared expert
                    // at the step's capacity (a bucket past 64 rows holds more than 64 real rows); the
                    // padding tail of the decode workspace's `delta` is cleared after the expert work.
                    crate::shared::decode_graph::real_row_moe(tables.real_rows, t, |real| {
                        self.moe(w, index, &layers[index], real, Scalar::I32(real as i32), cap, true,
                            &[Span { first: 0, rows: real }])
                    }, |tail| {
                        let offset = tail.start * self.cfg.hidden * 2;
                        let bytes = tail.len() * self.cfg.hidden * 2;
                        anyhow::ensure!(offset + bytes <= w.delta.buffer.bytes, "MoE tail exceeds delta");
                        // SAFETY: this subrange belongs to the live delta allocation;
                        // the same stream orders the clear after all quantizer scratch use.
                        unsafe {
                            let buffer = cuteafd_ffi::CuteafdDeviceBuffer {
                                ptr: w.delta.buffer.ptr.cast::<u8>().add(offset).cast(),
                                bytes, ..w.delta.buffer
                            };
                            self.library.cuda_zero_bytes_async(buffer, bytes, self.stream)
                        }
                    })?;
                } else {
                    // The router ran inside this layer's segment: its ids are still in place.
                    self.probe_ring(index, w, t, matches!(self.experts, Some(Experts::Spark { .. })))?;
                    self.moe_experts(w, index, &layers[index], t, rows, cap, true)?;
                }
            }
            if index < layers.len() {
                crate::shared::console::layer_mark(index);
                self.probe_mark(index);
            }
        }
        if layers.len() < self.cfg.layers {
            self.synchronize()?;
            return Ok(None);
        }
        if !head {
            let timer = std::time::Instant::now();
            self.run("head", &[("streams", w.streams[0].buffer.ptr), ("weight", self.weights.norm.buffer.ptr),
                ("out", w.x.buffer.ptr)], &[rows])?;
            self.logits(w, t, logit_rows, tables.decode)?;
            self.profile.borrow_mut()[2] += timer.elapsed().as_secs_f64();
        }
        Ok(Some(self.device_logits(w, logit_rows, head)))
    }

    /// The vocabulary projection of the last `logit_rows` normalized rows into
    /// `w.logits` through the one resident head: BF16, or the FP8 head
    /// (--fp8-head) in 16-row spans for every row count.
    fn logits(&self, w: &Workspace<'_>, t: usize, logit_rows: usize, _decode: bool) -> Result<()> {
        let h = self.cfg.hidden;
        // SAFETY: rows t - logit_rows.. of the normalized rows lie inside `w.x`.
        let x = unsafe { w.x.buffer.ptr.cast::<u8>().add((t - logit_rows) * h * 2) }.cast::<c_void>();
        match &self.weights.head {
            GlmfHead::Fp8 { .. } => self.timed("glmf_head_fp8", || {
                // SAFETY: `x` holds `logit_rows` normalized rows and `w.logits` their
                // FP32 logits (the workspace's logit capacity); the engine stream orders them.
                unsafe { self.weights.head.launch_fp8(self.programs, x, w.logits.buffer.ptr.cast(), logit_rows, h,
                    self.cfg.vocab_size, self.stream) }
            }),
            // SAFETY: the head's input and operands are live buffers of these shapes.
            GlmfHead::Bf16(head) => unsafe {
                w.head.as_ref().context("LM head")?.launch(x.cast(), head.buffer.ptr.cast(), w.logits.buffer.ptr.cast(), logit_rows as u32,
                    self.stream)
            },
        }
    }

    /// The target head as the DFlash drafter borrows it (the same resident copy).
    pub fn draft_head(&self) -> TargetHead<'_> {
        match &self.weights.head {
            GlmfHead::Bf16(head) => TargetHead::Bf16(head),
            fp8 @ GlmfHead::Fp8 { .. } => TargetHead::Launch(Box::new(move |x, logits, rows, stream| {
                // SAFETY: the drafter passes its live normalized rows and logits
                // workspace for `rows` rows, on the stream it launches the head on.
                unsafe { fp8.launch_fp8(self.programs, x, logits, rows, self.cfg.hidden, self.cfg.vocab_size, stream) }
            })),
        }
    }

    /// Launches `segment` through a graph captured the first time `key` is seen.
    fn replay(&self, key: GraphKey, segment: impl FnOnce() -> Result<()>) -> Result<()> {
        self.replay_on(0, key, segment)
    }

    /// [`Self::replay`] on rank `rank`'s stream (its own graphs). A capture measures its
    /// executable's device bytes (free memory before the capture and after its instantiation);
    /// executables evicted past the graph budget retire until the next decode step.
    fn replay_on(&self, rank: usize, key: GraphKey, segment: impl FnOnce() -> Result<()>) -> Result<()> {
        let graphs = self.graphs_of(rank);
        let stream = self.stream_of(rank);
        let cached = graphs.borrow_mut().launch(&key).map(|graph| graph.raw);
        if let Some(exec) = cached {
            // SAFETY: the graph's pointers are persistent engine buffers of that rank.
            return self.on(rank, || unsafe { self.library.cuda_graph_launch(exec, stream) });
        }
        let geometry = GraphGeometry { pool_width: key.pool_width, page_stride: key.page_stride,
            pool_stride: key.pool_stride, long: key.long };
        // Lazily captured graphs (a graph budget) capture unseen shapes by design.
        if !self.warming_graphs.get() && self.startup_graphs
            && self.logged_graph_shapes.borrow_mut().insert((rank, key.rows, key.spec, geometry)) {
            tracing::warn!(rank, rows = key.rows, spec = key.spec, long = key.long, pool_width = key.pool_width,
                page_stride = key.page_stride, pool_stride = key.pool_stride, "GLM Flash unseen serving graph geometry");
        }
        let calibrating = graphs.borrow().calibrating();
        let free = || self.on(rank, || self.library.cuda_memory_info().map(|(free, _)| free));
        let before = if calibrating { Some(free()?) } else { None };
        // SAFETY: capture records this stream's launches; nothing in a segment
        // synchronizes the host or allocates.
        self.on(rank, || unsafe { self.library.cuda_graph_begin_capture(stream) })?;
        self.capturing.set(true);
        let captured = segment();
        self.capturing.set(false);
        // SAFETY: ends the capture begun above on the same stream.
        let exec = self.on(rank, || unsafe { self.library.cuda_graph_end_capture(stream) });
        captured?;
        // SAFETY: persistent engine storage is drained before bank destruction.
        let exec = unsafe { GraphOwner::new(self.library, if rank == 1 { self.peer()?.device } else { self.device }, exec?, ())? };
        let measured = match before {
            Some(before) => Some(before.saturating_sub(free()?) as u64),
            None => None,
        };
        // SAFETY: the new graph reads and writes persistent engine buffers.
        self.on(rank, || unsafe { self.library.cuda_graph_launch(exec.raw, stream) })?;
        let (recaptured, evicted, held, stats) = {
            let mut graphs = graphs.borrow_mut();
            let recaptured = graphs.seen(&key);
            let retired = graphs.retired_len();
            graphs.insert(key, exec, measured);
            (recaptured, graphs.retired_len() - retired, graphs.bytes(), graphs.stats())
        };
        if recaptured {
            tracing::debug!(rank, ?key, recaptures = stats.recaptures, "decode graph recaptured");
        }
        if calibrating && !graphs.borrow().calibrating() {
            tracing::info!(rank, executables = stats.captures, each_bytes = graphs.borrow().each(),
                budget = ?graphs.borrow().budget(), "decode graph size calibrated at the first eviction");
        }
        if evicted > 0 {
            tracing::info!(rank, evicted, held = graphs.borrow().len(), held_bytes = held,
                budget = ?graphs.borrow().budget(),
                captures = stats.captures, recaptures = stats.recaptures, evictions = stats.evictions,
                "decode graphs past the budget retire");
        }
        Ok(())
    }

    fn graphs_of(&self, rank: usize) -> &RefCell<GraphBank<GraphKey, GraphExec<'a>>> {
        match (rank, &self.peer) {
            (1, Some(peer)) => &peer.graphs,
            _ => &self.graphs,
        }
    }

    /// Destroys retired decode graphs once every rank's stream has drained (none of them can be
    /// in flight). Called between decode steps: a sync inside one could wait on a peer push the
    /// host has not queued yet.
    fn release_retired_graphs(&self) -> Result<()> {
        if (0..self.ranks()).all(|rank| self.graphs_of(rank).borrow().retired_len() == 0) {
            return Ok(());
        }
        self.synchronize()?;
        for rank in 0..self.ranks() {
            let retired = self.graphs_of(rank).borrow_mut().drain_retired(|| Ok(()))?;
            self.on(rank, || { drop(retired); Ok(()) })?;
        }
        Ok(())
    }

    /// The decode graph budget of every rank (None: unbounded).
    pub(crate) fn set_graph_budget(&self, budget: Option<u64>) {
        for rank in 0..self.ranks() {
            self.graphs_of(rank).borrow_mut().set_budget(budget);
        }
    }

    /// Decode graph captures, recaptures and evictions, and the measured bytes held (every rank).
    pub(crate) fn graph_stats(&self) -> GraphCounts {
        (0..self.ranks()).fold(GraphCounts::default(), |total, rank| {
            let graphs = self.graphs_of(rank).borrow();
            let stats = graphs.stats();
            GraphCounts {
                stats: GraphStats { captures: total.stats.captures + stats.captures,
                    recaptures: total.stats.recaptures + stats.recaptures,
                    evictions: total.stats.evictions + stats.evictions },
                held: total.held + graphs.len(),
                shapes: total.shapes + graphs.count(|key| key.segment == 0),
                bytes: total.bytes + graphs.bytes(),
            }
        })
    }

    /// Attention-site collapse and input norm of `layer` from `streams`.
    fn pre(&self, w: &Workspace<'_>, streams: &Dev<'_>, layer: &GlmfLayer<'_>, rows: Scalar) -> Result<()> {
        self.pre_on(0, w, streams, layer, rows)
    }

    /// [`Self::pre`] on rank `rank`.
    fn pre_on(&self, rank: usize, w: &Workspace<'_>, streams: &Dev<'_>, layer: &GlmfLayer<'_>, rows: Scalar)
        -> Result<()> {
        self.pre_at(rank, w, streams, layer, 0, rows)
    }

    /// [`Self::pre_on`] over the `rows` rows from row `first` (one packed sequence's), the
    /// program's scratch from its start as for any other call.
    fn pre_at(&self, rank: usize, w: &Workspace<'_>, streams: &Dev<'_>, layer: &GlmfLayer<'_>, first: usize,
        rows: Scalar) -> Result<()> {
        let h = self.cfg.hidden;
        self.run_on(rank, false, "mhc_pre", &[("residual", row_at(streams, first, HC * h * 2)),
            ("fn", layer.ptr("attn.fn")?), ("scale", layer.ptr("attn.scale")?), ("base", layer.ptr("attn.base")?),
            ("norm", layer.ptr("input_norm")?), ("post", row_at(&w.post, first, HC * 4)),
            ("comb", row_at(&w.comb, first, HC * HC * 4)), ("y", row_at(&w.x, first, h * 2)),
            ("scratch", w.scratch.buffer.ptr)], &[rows])
    }

    /// On rank `rank`: the sublayer output `x` (`delta`, or a head split's `sum`) back into
    /// streams `cur` (into the other buffer), then the `site` collapse of `layer` normalized
    /// by its `norm`.
    #[allow(clippy::too_many_arguments)]
    fn post_pre_on(&self, rank: usize, w: &Workspace<'_>, x: *mut c_void, cur: usize, layer: &GlmfLayer<'_>,
        site: &str, norm: &str, rows: Scalar, cap: &str) -> Result<()> {
        self.post_pre_at(rank, w, x, cur, layer, site, norm, 0, rows, cap)
    }

    /// [`Self::post_pre_on`] over the `rows` rows from row `first` (one packed sequence's, on one
    /// GPU: `x` holds BF16 rows), the program's scratch from its start as for any other call.
    #[allow(clippy::too_many_arguments)]
    fn post_pre_at(&self, rank: usize, w: &Workspace<'_>, x: *mut c_void, cur: usize, layer: &GlmfLayer<'_>,
        site: &str, norm: &str, first: usize, rows: Scalar, cap: &str) -> Result<()> {
        let h = self.cfg.hidden;
        ensure!(first == 0 || self.peer.is_none(), "per-sequence rows run on one GPU");
        self.run_on(rank, false, &format!("mhc_post_pre_{cap}"), &[("x", x.wrapping_byte_add(first * h * 2)),
            ("residual", row_at(&w.streams[cur], first, HC * h * 2)), ("prev_post", row_at(&w.post, first, HC * 4)),
            ("prev_comb", row_at(&w.comb, first, HC * HC * 4)), ("fn", layer.ptr(&format!("{site}.fn"))?),
            ("scale", layer.ptr(&format!("{site}.scale"))?), ("base", layer.ptr(&format!("{site}.base"))?),
            ("norm", layer.ptr(norm)?), ("residual_out", row_at(&w.streams[cur ^ 1], first, HC * h * 2)),
            ("post", row_at(&w.post, first, HC * 4)), ("comb", row_at(&w.comb, first, HC * HC * 4)),
            ("y", row_at(&w.x, first, h * 2)), ("scratch", w.scratch.buffer.ptr)], &[rows])
    }

    /// Layer `index`'s attention on rank `rank` into `delta` (a head split: that rank's heads,
    /// a partial o_proj sum).
    #[allow(clippy::too_many_arguments)]
    fn attention(&self, rank: usize, w: &Workspace<'_>, index: usize, layer: &GlmfLayer<'_>, rows: Scalar, cap: &str,
        tables: &StepTables, trace: Option<&std::path::Path>) -> Result<()> {
        match layer.attention {
            // The chunked recurrence above 64 rows holds one sequence: a packed prefill runs the
            // layer once per sequence, over its rows (one span: the whole step).
            GlmNextAttention::Kda => tables.spans().iter().try_for_each(|span| {
                ensure!(span.first == 0 || (rank == 0 && self.peer.is_none()), "per-sequence rows run on one GPU");
                self.kda_on(rank, w, index, layer, if tables.segments.is_empty() { rows } else { span.rows() },
                    span.first, cap, tables.spec)
            }),
            GlmNextAttention::Mla => self.mla_on(rank, w, index, layer, rows, cap, tables, trace),
        }
    }

    /// Layer `index`'s KDA on rank `rank` (a head split: its heads and their state), over the
    /// `rows` rows from row `first` (one packed sequence's: its `seq_first` entries are 0).
    #[allow(clippy::too_many_arguments)]
    fn kda_on(&self, rank: usize, w: &Workspace<'_>, index: usize, layer: &GlmfLayer<'_>, rows: Scalar, first: usize,
        cap: &str, spec: bool) -> Result<()> {
        let ordinal = self.kda_ordinal[index].context("KDA layer without a state pool")?;
        let at = |pool: cuteafd_ffi::CuteafdDeviceBuffer, per: usize| -> *mut c_void {
            // SAFETY: ordinal < KDA layers, so the layer's region lies inside the pool.
            unsafe { pool.ptr.cast::<u8>().add(ordinal * per) }.cast()
        };
        let caches = self.caches_of(rank);
        let d = caches.kda_heads * self.cfg.kda_head_dim;
        let conv_state = at(caches.kda_conv.buffer, self.slots * 3 * 3 * d * 2);
        let state = at(caches.kda_state.buffer, self.slots * d * self.cfg.kda_head_dim * self.kda_state.bytes());
        // Records of the step's programs' rows, layer by layer (an `_m64` step's 64-row records, a wide
        // step's 128-row ones), as the commit that follows reads them.
        let replay = at(caches.kda_replay, replay_bytes(caches.kda_heads, 3 * d, replay_rows_of(cap)));
        let decode = is_decode(cap);
        // Row `first`'s input, state slot, sequence start and output. The output rows are FP32 only
        // for a head split's FP32 partials (`kda_w8_f32`); packed rows run on one GPU, in BF16.
        let h = self.cfg.hidden;
        let out_bytes = if self.precise_attention(layer) { 4 } else { 2 };
        let rows_at = [row_at(&w.x, first, h * 2), row_at(&w.kda_slots, first, 4), row_at(&w.seq_first, first, 4),
            w.delta.buffer.ptr.wrapping_byte_add(first * h * out_bytes)];
        if layer.has("w_in_fp8") {
            return self.kda_w8(rank, w, layer, rows, rows_at, cap, spec, [conv_state, state, replay]);
        }
        let [x, slots, seq_first, out] = rows_at;
        let mut pointers = vec![("x", x), ("w_in", layer.ptr("w_in")?)];
        // Decode programs read per-row scales [N, K/128]; prefill ones K-block major.
        let (in_scale, o_scale) = if decode { ("w_in_scale", "w_o_scale") } else { ("w_in_kscale", "w_o_kscale") };
        pointers.extend([("w_in_fp8", layer.ptr_or("w_in_fp8", "w_in")?), (in_scale, layer.ptr_or(in_scale, "w_in")?)]);
        pointers.extend([("w_fg", layer.ptr("w_fg")?), ("conv_w", layer.ptr("conv_w")?), ("a_log", layer.ptr("a_log")?),
            ("dt_bias", layer.ptr("dt_bias")?), ("o_norm", layer.ptr("o_norm")?), ("w_o", layer.ptr("w_o")?)]);
        pointers.extend([("w_o_fp8", layer.ptr_or("w_o_fp8", "w_o")?), (o_scale, layer.ptr_or(o_scale, "w_o")?)]);
        pointers.extend([("conv_state", conv_state), ("state", state), ("slots", slots), ("seq_first", seq_first),
            ("out", out)]);
        let mut scalars = self.fp8_scalars(rows, decode, layer.has(if decode { "w_in_fp8" } else { "w_in_kscale" }));
        if !decode && layer.has("w_in_kscale") {
            // Prefill fp8_rows bits: 1 the in-projection, 2 o_proj.
            scalars[1] = Scalar::I32(self.fp8_prefill.kda_bits);
        }
        if decode {
            pointers.push(("replay", replay));
            scalars.push(Scalar::I32(i32::from(spec)));
        } else {
            ensure!(!spec, "speculative steps are decode-shaped");
        }
        pointers.push(("scratch", w.scratch.buffer.ptr));
        self.run_on(rank, layer.split, &self.kda_state.program(cap), &pointers, &scalars)
    }

    /// KDA over the layer's only (FP8, per-row x 128-K, K-major scales) in/out
    /// projections: `kda_w8_{cap}`. Decode rows up to 32 on half heads (16 on full heads) run the FP8 GEMV, wider
    /// verify steps W8A16; prefill runs W8A8 on the `--fp8-prefill kda-*` bits, else W8A16. `rows_at`: the
    /// rows' input, state slots, sequence starts and output.
    #[allow(clippy::too_many_arguments)]
    fn kda_w8(&self, rank: usize, w: &Workspace<'_>, layer: &GlmfLayer<'_>, rows: Scalar,
        [x, slots, seq_first, out]: [*mut c_void; 4], cap: &str, spec: bool,
        [conv_state, state, replay]: [*mut c_void; 3]) -> Result<()> {
        let decode = is_decode(cap);
        let mut pointers = vec![("x", x), ("w_in_fp8", layer.ptr("w_in_fp8")?),
            ("w_in_kscale", layer.ptr("w_in_kscale")?), ("w_fg", layer.ptr("w_fg")?), ("conv_w", layer.ptr("conv_w")?),
            ("a_log", layer.ptr("a_log")?), ("dt_bias", layer.ptr("dt_bias")?), ("o_norm", layer.ptr("o_norm")?),
            ("w_o_fp8", layer.ptr("w_o_fp8")?), ("w_o_kscale", layer.ptr("w_o_kscale")?), ("conv_state", conv_state),
            ("state", state), ("slots", slots), ("seq_first", seq_first), ("out", out)];
        let mut scalars = vec![rows, Scalar::I32(if decode {
            if layer.split { 32 } else { FP8_ROWS }
        } else { self.fp8_prefill.kda_bits })];
        if decode {
            pointers.push(("replay", replay));
            scalars.push(Scalar::I32(i32::from(spec)));
        } else {
            ensure!(!spec, "speculative steps are decode-shaped");
        }
        pointers.push(("scratch", w.scratch.buffer.ptr));
        let dtype = if self.output_shard_attention(layer) { "_norm" }
            else if self.precise_attention(layer) { "_f32" } else { "" };
        let expanded = if self.kda_prefill_expanded && !is_decode(cap) { "_expanded" } else { "" };
        let name = format!("kda_w8{dtype}{expanded}_{cap}");
        self.run_on(rank, layer.split, &name, &pointers, &scalars)
    }

    /// `[rows, fp8]`: the decode programs' `fp8_rows` (16 when the layer has
    /// the FP8 copy, else 0), or the prefill programs' `fp8` switch (1: rows
    /// past the skinny GEMV run the block-FP8 GEMMs).
    fn fp8_scalars(&self, rows: Scalar, decode: bool, fp8: bool) -> Vec<Scalar> {
        vec![rows, Scalar::I32(match (fp8, decode) {
            (false, _) => 0,
            (true, true) => FP8_ROWS,
            (true, false) => 1,
        })]
    }

    /// SwiGLU MLP (dense layer or shared expert) of intermediate `inter` into `out`.
    fn ffn(&self, w: &Workspace<'_>, layer: &GlmfLayer<'_>, inter: usize, cap: &str, out: *mut c_void, rows: Scalar)
        -> Result<()> {
        self.ffn_on(0, w, layer, inter, cap, out, rows)
    }

    /// [`Self::ffn`] on rank `rank`: a head-split layer runs its half of the intermediate (a
    /// partial sum); a ModelOpt NVFP4 dense MLP runs whole on rank 0.
    #[allow(clippy::too_many_arguments)]
    fn ffn_on(&self, rank: usize, w: &Workspace<'_>, layer: &GlmfLayer<'_>, inter: usize, cap: &str, out: *mut c_void,
        rows: Scalar) -> Result<()> {
        if layer.has("nvfp4_w1") {
            ensure!(rank == 0, "NVFP4 dense MLPs run on rank 0");
            let dense = self.dense_nvfp4.as_ref().context("an NVFP4 dense layer needs the fp8-glmfdense-nvfp4 package")?;
            let Scalar::I32(rows) = rows else { anyhow::bail!("row count scalar") };
            let pointers = [w.x.buffer.ptr, dense.ids.buffer.ptr, dense.weights.buffer.ptr, layer.ptr("nvfp4_w1")?,
                layer.ptr("nvfp4_s1")?, layer.ptr("nvfp4_w3")?, layer.ptr("nvfp4_s3")?, layer.ptr("nvfp4_w2")?,
                layer.ptr("nvfp4_s2")?, out, dense.scratch.buffer.ptr];
            // SAFETY: the input rows, the layer's NVFP4 operands, the constant ids/weights
            // (the expert rows: the widest prefill lane or decode step), the output and the
            // scratch are live device buffers used on the engine stream.
            return self.timed("ffn (NVFP4 dense)", || unsafe {
                dense.module.launch(&pointers, usize::try_from(rows)?, self.stream)
            });
        }
        let decode = is_decode(cap);
        let pointers = [("x", w.x.buffer.ptr), ("w_gate_up_fp8", layer.ptr("w_gate_up_fp8")?),
            ("w_gate_up_scale", layer.ptr("w_gate_up_scale")?), ("w_down_fp8", layer.ptr("w_down_fp8")?),
            ("w_down_scale", layer.ptr("w_down_scale")?), ("out", out), ("scratch", w.scratch.buffer.ptr)];
        // FP8-only weights: decode rows up to `fp8_rows` on the GEMV; prefill W8A8 or W8A16.
        let inter = inter / if layer.split { 2 } else { 1 };
        self.run_on(rank, layer.split, &format!("ffn_i{inter}_{cap}"), &pointers,
            &self.fp8_scalars(rows, decode, decode || self.fp8_prefill.ffn))
    }

    /// The rows and tables the DSA indexer, top-k and selection run over: each packed sequence's
    /// (its rows, and its page tables at their offsets in `w`'s table row), or the whole step's.
    fn index_parts(&self, w: &Workspace<'_>, tables: &StepTables) -> Vec<IndexPart> {
        if tables.segments.is_empty() {
            return vec![IndexPart { first: 0, rows: tables.positions.len(), long: tables.long,
                page_table: w.page_table.buffer.ptr, pool: None }];
        }
        tables.segments.iter().map(|s| IndexPart { first: s.first, rows: s.rows, long: s.long,
            page_table: row_at(&w.page_table, s.page_offset, 4),
            pool: Some((row_at(&w.pool_table, s.pool_offset, 4), s.pool_columns)) }).collect()
    }

    /// Layer `index`'s MLA on rank `rank` (a head split: the replicated latent record and DSA
    /// indexer, its heads' queries, sparse MLA, W_UV and a partial o_proj).
    #[allow(clippy::too_many_arguments)]
    fn mla_on(&self, rank: usize, w: &Workspace<'_>, index: usize, layer: &GlmfLayer<'_>, rows: Scalar, cap: &str,
        tables: &StepTables, trace: Option<&std::path::Path>) -> Result<()> {
        let t = tables.positions.len();
        let trace = trace.filter(|_| rank == 0);
        let caches = self.caches_of(rank);
        let split = layer.split;
        let heads = self.cfg.heads / if split { 2 } else { 1 };
        // These scratch layouts are diagnostic-only and match the pinned AOT
        // decode programs. Stop at the first MLA layer; later layers' inputs
        // already differ and cannot identify the original numerical cause.
        let trace = trace.filter(|_| tables.decode && index == 3);
        let mode = if tables.decode { "decode" } else { "prefill" };
        let cache = caches.kv[index].as_ref().context("MLA layer without a record pool")?.buffer.ptr;
        let index_cache = caches.index[index].as_ref().context("MLA layer without an index cache")?;
        let pool_cache = &index_cache.pools;
        let decode = tables.decode;
        // FP8-only weights: decode rows up to `fp8_rows` on the GEMV; prefill W8A8 or W8A16.
        let fp8 = decode || self.fp8_prefill.mla;
        let mut pointers = vec![("x", w.x.buffer.ptr), ("kv_slots", w.kv_slots.buffer.ptr),
            ("w_qkv_a_fp8", layer.ptr("w_qkv_a_fp8")?), ("w_qkv_a_scale", layer.ptr("w_qkv_a_scale")?),
            ("q_a_norm", layer.ptr("q_a_norm")?), ("kv_a_norm", layer.ptr("kv_a_norm")?),
            ("w_q_b_fp8", layer.ptr("w_q_b_fp8")?), ("w_q_b_scale", layer.ptr("w_q_b_scale")?)];
        pointers.extend([("w_uk", layer.ptr("w_uk")?), ("kv_cache", cache), ("query", w.query.buffer.ptr),
            ("q_resid", w.q_resid.buffer.ptr), ("scratch", w.scratch.buffer.ptr)]);
        self.run_on(rank, split, &format!("mla_producer_{cap}"), &pointers, &self.fp8_scalars(rows, decode, fp8))?;
        if let Some(dir) = trace {
            let qkv_width = self.cfg.q_lora_rank + self.cfg.kv_lora_rank;
            let q_width = self.cfg.heads * self.cfg.qk_nope_dim;
            let bytes = (t * qkv_width * 2).next_multiple_of(1024)
                + (t * q_width * 2).next_multiple_of(1024);
            for (name, buffer, bytes) in [("mla_producer_scratch.bin", &w.scratch, bytes),
                ("mla_query.bin", &w.query, t * self.cfg.heads * self.cfg.kv_lora_rank * 2),
                ("mla_q_resid.bin", &w.q_resid, t * self.cfg.q_lora_rank * 2)] {
                std::fs::write(dir.join(name), self.download(buffer, bytes)?)?;
            }
            std::fs::write(dir.join("mla_meta.json"), serde_json::to_vec(&serde_json::json!({
                "rows": t, "qkv_width": qkv_width, "q_width": q_width, "alignment": 1024 }))?)?;
        }
        // The DSA indexer, its pool top-k and the selection read one sequence's tables: a packed
        // prefill runs them once per sequence, over its rows and its tables (one part: the step).
        let (h, q, pools) = (self.cfg.hidden, self.cfg.q_lora_rank, self.cfg.index_topk / KPOOL);
        ensure!(rank == 0 || tables.segments.is_empty(), "a packed prefill runs on one GPU");
        for part in self.index_parts(w, tables) {
            let (first, rows) = (part.first, if tables.segments.is_empty() { rows } else { Scalar::I32(part.rows as i32) });
            let (x, q_resid) = (row_at(&w.x, first, h * 2), row_at(&w.q_resid, first, q * 2));
            let (q_fp8, head_weights) = (row_at(&w.q_fp8, first, INDEX_HEADS * 128), row_at(&w.head_weights, first,
                INDEX_HEADS * 4));
            let (pool_slots, positions) = (row_at(&w.pool_slots, first, 8), row_at(&w.positions, first, 8));
            match (&index_cache.keys, &caches.index_tails) {
                (Some(keys), _) => self.run_on(rank, false, &format!("index_producer_{cap}"), &[("x", x),
                    ("q_resid", q_resid), ("slots", row_at(&w.kv_slots, first, 8)), ("pool_slots", pool_slots),
                    ("w_iq", layer.ptr("w_iq")?), ("w_ik", layer.ptr("w_ik")?), ("k_norm_w", layer.ptr("k_norm_w")?),
                    ("k_norm_b", layer.ptr("k_norm_b")?), ("ape", layer.ptr("ape")?), ("token_keys", keys.buffer.ptr),
                    ("index_cache", pool_cache.buffer.ptr), ("q_fp8", q_fp8), ("head_weights", head_weights),
                    ("scratch", w.scratch.buffer.ptr)], &[rows])?,
                (None, Some((tails, records))) => {
                    // This MLA layer's tails `[slots, TAIL_BYTES]` and speculative key | gate record.
                    let ordinal = self.mla_ordinal[index].context("MLA layer without an ordinal")?;
                    let recorded = replay_rows_of(cap);
                    ensure!(!tables.spec || t <= recorded, "a speculative step of {t} rows exceeds the replay record");
                    // SAFETY: ordinal < MLA layers and the records hold `decode_rows` >= `recorded` rows per
                    // layer: both regions lie inside their pools.
                    let (tails, record) = unsafe { (tails.buffer.ptr.cast::<u8>().add(ordinal * self.slots * TAIL_BYTES),
                        records.buffer.ptr.cast::<u8>().add(ordinal * recorded * KEY_BYTES)) };
                    self.run_on(rank, false, &format!("index_producer_c_{cap}"), &[("x", x), ("q_resid", q_resid),
                        ("pool_slots", pool_slots), ("positions", positions),
                        ("kda_slots", row_at(&w.kda_slots, first, 4)), ("seq_first", row_at(&w.seq_first, first, 4)),
                        ("w_iq", layer.ptr("w_iq")?), ("w_ik", layer.ptr("w_ik")?), ("k_norm_w", layer.ptr("k_norm_w")?),
                        ("k_norm_b", layer.ptr("k_norm_b")?), ("ape", layer.ptr("ape")?), ("tails", tails.cast()),
                        ("replay", record.cast()), ("index_cache", pool_cache.buffer.ptr), ("q_fp8", q_fp8),
                        ("head_weights", head_weights), ("scratch", w.scratch.buffer.ptr)],
                        &[rows, Scalar::I32(i32::from(tables.spec))])?
                }
                (None, None) => anyhow::bail!("MLA layer {index} has neither token keys nor index tails"),
            }
            let pools_at = row_at(&w.pools, first, pools * 4);
            // A packed step is long when any of its sequences is: its short parts skip the top-k.
            if tables.long {
                if part.long {
                    self.run_on(rank, false, &format!("index_topk_{mode}_{cap}"), &[("q_fp8", q_fp8),
                        ("weights", head_weights), ("index_k_cache", pool_cache.buffer.ptr),
                        ("page_table", part.pool.map_or(w.pool_table.buffer.ptr, |(table, _)| table)),
                        ("cache_lengths", row_at(&w.cache_lengths, first, 4)), ("output_indices", pools_at),
                        ("scratch", w.topk_scratch.buffer.ptr)],
                        &[rows, part.pool.map_or(Scalar::I32(tables.pool_width.max(1) as i32),
                            |(_, width)| Scalar::I32(width.max(1) as i32)), Scalar::I32(tables.pool_stride as i32)])?;
                }
            }
            self.run_on(rank, false, "index_expand", &[("positions", positions), ("pools", pools_at),
                ("pool_logical", caches.pool_logical.buffer.ptr), ("page_table", part.page_table),
                ("indices", row_at(&w.indices, first, SPARSE_TOPK * 4)), ("lengths", row_at(&w.lengths, first, 4))],
                &[rows, Scalar::I32(tables.page_stride as i32)])?;
        }
        if let Some(dir) = trace {
            for (name, buffer, bytes) in [("mla_indices.bin", &w.indices, t * SPARSE_TOPK * 4),
                ("mla_lengths.bin", &w.lengths, t * 4)] {
                std::fs::write(dir.join(name), self.download(buffer, bytes)?)?;
            }
            let kv = caches.kv[index].as_ref().context("MLA trace without a record pool")?;
            std::fs::write(dir.join("mla_kv.bin"), self.download(kv, kv.buffer.bytes)?)?;
        }
        if let (false, Some(kernel)) = (tables.decode, crate::families::glm5::engine::native_mla_prefill()) {
            let scale = (self.cfg.qk_nope_dim as f32).powf(-0.5);
            let stream = self.stream_of(rank);
            let launch = || self.on(rank, || {
                // SAFETY: query, record cache, indices, lengths and the latent output
                // are live buffers of the step's rows on this rank's stream.
                unsafe {
                    self.library.glm_mla_prefill(w.query.buffer.ptr, cache, w.indices.buffer.ptr, w.lengths.buffer.ptr,
                        w.latent.buffer.ptr, tables.positions.len(), heads, SPARSE_TOPK, RECORD_BYTES,
                        scale * std::f32::consts::LOG2_E, kernel, stream)
                }
            });
            if rank == 0 { self.timed("glm_mla_prefill (native)", launch)? } else { launch()? }
            if rank == 0 && crate::families::glm5::engine::mla_prefill_check() {
                // SAFETY: as above; the check synchronizes the stream.
                let stats = unsafe {
                    self.library.glm_mla_prefill_check(w.query.buffer.ptr, cache, w.indices.buffer.ptr,
                        w.lengths.buffer.ptr, tables.positions.len(), heads, SPARSE_TOPK, RECORD_BYTES,
                        scale * std::f32::consts::LOG2_E, self.stream)
                }?;
                crate::families::glm5::engine::print_mla_check(index, &stats);
            }
        } else {
            self.run_on(rank, split, &format!("sparse_mla_{mode}_{cap}"), &[("q", w.query.buffer.ptr), ("kv_cache", cache),
                ("indices", w.indices.buffer.ptr), ("lengths", w.lengths.buffer.ptr), ("out", w.latent.buffer.ptr),
                ("scratch", w.scratch.buffer.ptr)], &[rows])?;
        }
        if let Some(dir) = trace {
            std::fs::write(dir.join("mla_latent.bin"), self.download(&w.latent,
                t * self.cfg.heads * self.cfg.kv_lora_rank * 2)?)?;
            std::fs::write(dir.join("mla_sparse_scratch.bin"), self.download(&w.scratch,
                self.scratch(&format!("sparse_mla_decode_{cap}"))?)?)?;
        }
        let pointers = [("attn", w.latent.buffer.ptr), ("w_uv", layer.ptr("w_uv")?),
            ("w_o_fp8", layer.ptr("w_o_fp8")?), ("w_o_scale", layer.ptr("w_o_scale")?),
            ("out", w.delta.buffer.ptr), ("scratch", w.scratch.buffer.ptr)];
        self.run_on(rank, split, &format!("o_{cap}"), &pointers, &self.fp8_scalars(rows, decode, fp8))?;
        if let Some(dir) = trace {
            std::fs::write(dir.join("mla_values.bin"), self.download(&w.scratch,
                t * self.cfg.heads * self.cfg.v_head_dim * 2)?)?;
        }
        Ok(())
    }

    /// Router, shared expert and routed experts; leaves `bf16(routed + shared)` in `delta`.
    /// `spans`: the router scores' rows (each packed sequence's, or the whole step).
    #[allow(clippy::too_many_arguments)]
    fn moe(&self, w: &Workspace<'_>, index: usize, layer: &GlmfLayer<'_>, t: usize, rows: Scalar, cap: &str,
        decode: bool, spans: &[Span]) -> Result<()> {
        self.moe_front(w, index, layer, t, rows, cap, spans)?;
        self.moe_experts(w, index, layer, t, rows, cap, decode)
    }

    /// Router logits, the sigmoid top-8, the shared expert (into `shared`)
    /// and, for wire-fed experts, the FP8 K32 wire rows. No host sync. The router scores, a skinny
    /// GEMV up to 160 rows, run over each of `spans` (each packed sequence's rows, or the whole
    /// step); the rest over every row.
    #[allow(clippy::too_many_arguments)]
    fn moe_front(&self, w: &Workspace<'_>, index: usize, layer: &GlmfLayer<'_>, t: usize, rows: Scalar, cap: &str,
        spans: &[Span]) -> Result<()> {
        let (h, topk) = (self.cfg.hidden, self.cfg.topk);
        let experts = self.experts.as_ref().with_context(|| format!(
            "layer {index} is an MoE layer: pass --local-experts (FP8 package) or Spark --peers \
             (run --layers 3 for the dense layers alone)"))?;
        for span in spans {
            let rows = if spans.len() == 1 && span.rows == t { rows } else { span.rows() };
            self.run("router_scores", &[("x", row_at(&w.x, span.first, h * 2)), ("w", layer.ptr("gate")?),
                ("logits", row_at(&w.router_logits, span.first, self.cfg.experts * 4))], &[rows])?;
        }
        self.timed("router_select", || {
            // SAFETY: logits, bias and route outputs are live buffers of `t` rows.
            unsafe {
                self.library.router_select(w.router_logits.buffer.ptr, layer.ptr("gate.bias")?, std::ptr::null(),
                    std::ptr::null(), w.route_ids.buffer.ptr, w.route_weights.buffer.ptr, t, self.cfg.experts, topk,
                    self.cfg.routed_scale as f32, true, self.stream)
            }
        })?;
        self.probe_ring(index, w, t, matches!(experts, Experts::Spark { .. }))?;
        if !matches!(experts, Experts::Local(_)) {
            let grid = self.quantize_grid.blocks(t, h);
            self.run("expert_input_quant", &[("source_ptr", w.x.buffer.ptr), ("values_ptr", w.wire.buffer.ptr),
                // SAFETY: the scale rows follow the payload inside each wire row.
                ("scale_rows_ptr", unsafe { w.wire.buffer.ptr.cast::<u8>().add(h) }.cast()),
                ("scale_mma_ptr", w.delta.buffer.ptr)], &[rows, Scalar::I32(grid as i32)])?;
        }
        Ok(())
    }

    /// Per MoE layer, the weights a decode step reads after its routed
    /// experts are out, in read order: the next layer's attention site and
    /// attention (E4M3 copies where the decode programs read them), FFN site,
    /// router and shared expert; after the last layer the final norm and head.
    pub fn decode_read_order(&self) -> Vec<Vec<crate::shared::l2_prefetch::Range>> {
        self.decode_read_order_on(0)
    }

    /// Rank 1's L2 prefetch (its shares of the next layer's weights) with `budget` bytes per
    /// layer, issued before each of its decode segments that waits on an expert exchange.
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
            Some(next) => {
                let attention: &[&str] = match next.attention {
                    GlmNextAttention::Kda if next.has("w_in_fp8") => &["w_in_fp8", "w_in_kscale", "w_fg", "conv_w",
                        "a_log", "dt_bias", "o_norm", "w_o_fp8", "w_o_kscale"],
                    GlmNextAttention::Kda => &["w_in", "w_fg", "conv_w", "a_log", "dt_bias", "o_norm", "w_o"],
                    GlmNextAttention::Mla => &["w_qkv_a_fp8", "w_qkv_a_scale", "q_a_norm", "kv_a_norm", "w_q_b_fp8",
                        "w_q_b_scale", "w_iq", "w_ik", "k_norm_w", "k_norm_b", "ape", "w_uk", "w_uv", "w_o_fp8",
                        "w_o_scale"],
                };
                let names: Vec<&str> = ["attn.fn", "attn.scale", "attn.base", "input_norm"].iter().chain(attention)
                    .chain(&["ffn.fn", "ffn.scale", "ffn.base", "post_norm", "gate", "gate.bias", "w_gate_up_fp8",
                        "w_gate_up_scale", "w_down_fp8", "w_down_scale"])
                    .copied().collect();
                crate::shared::l2_prefetch::operands(&names, |n| next.range(n))
            }
            None if rank == 1 => Vec::new(),
            None => {
                std::iter::once(&self.weights.norm).chain(self.weights.head.allocations())
                    .map(|a| (a.buffer.ptr.cast_const(), a.buffer.bytes)).collect()
            }
        }).collect()
    }

    /// After layer `index`'s shared expert is queued in a one-lane decode or
    /// verify step (`decode`): the L2 prefetch of what the step reads next;
    /// with no real exchange (`local`), only under CUTEAFD_EMULATE_EXCHANGE_US
    /// (benchmarks), with a Spark-like wait.
    fn exchange_window(&self, index: usize, decode: bool, local: bool) -> Result<()> {
        if !decode {
            return Ok(());
        }
        let mark = if local { crate::shared::l2_prefetch::exchange_mark(self.library, self.stream)? } else { None };
        if local && mark.is_none() {
            return Ok(());
        }
        if let Some(l2) = &self.l2 {
            l2.issue(self.library, index, self.stream)?;
        }
        crate::shared::l2_prefetch::exchange_wait(self.library, mark)
    }

    /// The routed experts of layer `index` (the front ran); leaves
    /// `bf16(routed + shared)` in `delta`.
    /// The shared expert runs here, after the routes are on their way (on the
    /// Spark path while the ranks compute).
    #[allow(clippy::too_many_arguments)]
    fn moe_experts(&self, w: &Workspace<'_>, index: usize, layer: &GlmfLayer<'_>, t: usize, rows: Scalar, cap: &str,
        decode: bool) -> Result<()> {
        let h = self.cfg.hidden;
        let experts = self.experts.as_ref().context("MoE layer without experts")?;
        let shared = || {
            self.ffn(w, layer, self.cfg.moe_intermediate, cap, w.shared.buffer.ptr, rows)?;
            self.exchange_window(index, decode, !matches!(experts, Experts::Spark { .. }))
        };
        match experts {
            Experts::Skip => {
                self.ffn(w, layer, self.cfg.moe_intermediate, cap, w.delta.buffer.ptr, rows)?;
                return self.exchange_window(index, decode, true);
            }
            Experts::Local(local) => {
                shared()?;
                let resident = local.index_of(index)?;
                let fp8 = local.experts.borrow();
                ensure!(!fp8.wire_input(), "the coordinator FP8 package takes BF16 rows");
                // SAFETY: input rows, route ids, weights and the output are live
                // buffers of `t` rows on this engine's stream.
                unsafe {
                    fp8.run(resident, t, w.x.buffer.ptr, w.route_ids.buffer.ptr, w.route_weights.buffer.ptr,
                        w.routed_ptr()?, self.stream)?;
                }
                if local.window.is_some() {
                    // Diagnostic paging may drop this layer before the stream drains.
                    // SAFETY: the engine owns this stream.
                    unsafe { self.library.cuda_stream_synchronize(self.stream)? };
                }
            }
            Experts::LocalExl3(local) => {
                shared()?;
                local.ensure(index, self.stream)?;
                let mut resident = local.resident.borrow_mut();
                let (_, experts) = resident.as_mut().context("EXL3 window")?;
                // SAFETY: wire rows, routes and the shared-expert rows are complete in
                // stream order; the output is copied before the window can change.
                unsafe {
                    experts.run(crate::families::deepseek_v4::local::LocalLayer::Backbone(index), t, w.wire.buffer.ptr,
                        w.route_ids.buffer.ptr, w.route_weights.buffer.ptr, w.shared.buffer.ptr, self.stream)?;
                    self.library.copy_d2d_async(w.delta.buffer, experts.output.buffer, t * h * 2, self.stream)?;
                }
                return Ok(());
            }
            Experts::Spark { transports, runtime } => {
                // The compact reducer adds the shared expert plane to the rank partials.
                let mut transports = transports.borrow_mut();
                let transport = transports.first_mut().context("no Spark transport")?;
                let wave = self.spark_dispatch(w, index, t, decode, transport, shared)?;
                return runtime.block_on(self.spark_land(w, t, transport, wave));
            }
        }
        self.run("add", &[("a", w.routed_ptr()?), ("b", w.shared.buffer.ptr), ("out", w.delta.buffer.ptr)], &[rows])
    }

    /// The draft probe's layer mark: the end of `index` on the engine stream.
    fn probe_mark(&self, index: usize) {
        if let Some(probe) = self.draft_probe.borrow_mut().as_mut() {
            // SAFETY: the engine's own stream, live for the engine's lifetime.
            unsafe { probe.mark(index, self.stream) };
        }
    }

    /// Queue layer `index`'s router ids into the draft probe's ring right
    /// behind the router (`spark`: the layer's ids are also staged).
    fn probe_ring(&self, index: usize, w: &Workspace<'_>, t: usize, spark: bool) -> Result<()> {
        if self.capturing.get() { return Ok(()); }
        let mut probe = self.draft_probe.borrow_mut();
        let Some(probe) = probe.as_mut().filter(|p| p.wants_ring(spark)) else { return Ok(()) };
        // SAFETY: the router just wrote `t` rows of ids into `route_ids` on this stream; the ring
        // is read after the step's sync.
        unsafe { probe.ring(self.library, index, w.route_ids.buffer, t, self.stream) }
    }

    /// Arm the shared draft policy's per-step signals for verify steps of up
    /// to `rows` rows (`serve-glmf` under the shared policy only).
    pub(crate) fn arm_draft_probe(&self, rows: usize) -> Result<()> {
        let dense = self.weights.layers.iter().map(|layer| layer.dense).collect();
        *self.draft_probe.borrow_mut() = Some(super::draft_probe::RoundProbe::new(self.library, dense,
            self.cfg.topk, rows)?);
        Ok(())
    }

    /// Start recording a verify step of `rows` rows (with the draft probe armed).
    pub(crate) fn probe_begin(&self, rows: usize) {
        if let Some(probe) = self.draft_probe.borrow_mut().as_mut() { probe.begin(rows); }
    }

    /// Stop recording after the step and its token selection (which drains the
    /// engine stream); a failed step's signals are dropped.
    pub(crate) fn probe_end(&self, ok: bool) {
        if let Some(probe) = self.draft_probe.borrow_mut().as_mut() {
            probe.end();
            if !ok { let _ = probe.finish(); }
        }
    }

    /// The probed step's routes and layer µs, once its stream drained, handed
    /// to `observe` (nothing when the step's signals are incomplete).
    pub(crate) fn probe_finish<T>(&self, observe: impl FnOnce(&crate::shared::draft::routes::RoundRoutes,
        &[Option<f64>]) -> T) -> Option<T> {
        let mut probe = self.draft_probe.borrow_mut();
        let (routes, layer_us) = probe.as_mut()?.finish()?;
        Some(observe(routes, &layer_us))
    }

    /// Ring and staged-id agreement under `CUTEAFD_GLMF_ROUTE_RING_CHECK`.
    pub(crate) fn probe_ring_check(&self) -> Option<super::draft_probe::RingCheck> {
        self.draft_probe.borrow().as_ref().map(|probe| probe.ring_check)
    }

    /// Routes and wire rows down and one request to every Spark rank; the
    /// shared expert (`shared`) queues behind the copies and runs while the
    /// ranks compute. Complete with [`Self::spark_land`].
    fn spark_dispatch(&self, w: &Workspace<'_>, index: usize, t: usize, decode: bool, transport: &mut SparkLink<'_>,
        shared: impl FnOnce() -> Result<()>) -> Result<SparkExpertWave> {
        let kind = if decode { ExpertV2SourceKind::Decode } else { ExpertV2SourceKind::Prefill };
        let (h, topk) = (self.cfg.hidden, self.cfg.topk);
        let (route_bytes, wire_bytes) = (t * topk * 4, t * (h + h / 32));
        let staging = w.router_host.borrow_mut();
        let host = staging.buffer;
        let at = |offset: usize| cuteafd_ffi::CuteafdHostBuffer {
            // SAFETY: ids, weights and wire rows are consecutive inside the pinned buffer.
            ptr: unsafe { host.ptr.cast::<u8>().add(offset) }.cast(),
            bytes: host.bytes - offset,
            ..host
        };
        let timer = std::time::Instant::now();
        // SAFETY: the pinned regions are large enough; the sync completes them.
        unsafe {
            self.library.copy_d2h_host_buffer_async(at(0), w.route_ids.buffer, route_bytes, self.stream)?;
            self.library.copy_d2h_host_buffer_async(at(route_bytes), w.route_weights.buffer, route_bytes, self.stream)?;
            self.library.copy_d2h_host_buffer_async(at(2 * route_bytes), w.wire.buffer, wire_bytes, self.stream)?;
            self.library.cuda_event_record(self.routes_ready, self.stream)?;
        }
        // The shared expert queues behind the copies and runs during the exchange;
        // the host waits for the copies only (they also order after this
        // workspace's previous plane uploads, which frees its pinned plane staging).
        shared()?;
        // SAFETY: the event was recorded on this engine's stream above.
        unsafe { self.library.cuda_event_synchronize(self.routes_ready)? };
        self.profile.borrow_mut()[0] += timer.elapsed().as_secs_f64();
        let staged = staging.bytes();
        if let Some(probe) = self.draft_probe.borrow_mut().as_mut().filter(|p| p.active()) {
            probe.staged(index, &staged[..route_bytes]);
        }
        let word = |offset: usize, i: usize| u32::from_le_bytes(staged[offset + i * 4..][..4].try_into().unwrap());
        let routes = (0..t * topk).map(|i| ExpertProtocolV2RouteEntry {
            row_index: (i / topk) as u32, expert_id: word(0, i), gate_weight: f32::from_bits(word(route_bytes, i)),
        }).collect();
        let wire = staged[2 * route_bytes..2 * route_bytes + wire_bytes].to_vec();
        drop(staging);
        let request = spark_request(h, topk, index, t, routes, wire, kind)?;
        if self.split_audit && !decode && t >= 256 {
            let counts = audit_route_counts(&request.routes);
            tracing::info!(layer = index, rows = t, distinct = counts.len(),
                max_routes = counts.values().max().copied().unwrap_or(0),
                histogram = %serde_json::to_string(&counts)?, "GLM Flash split audit routes");
        }
        transport.dispatch(&request)
    }

    /// Receives `wave`'s BF16 rank partials into its transport's intake planes
    /// and sums them with the shared expert into `delta`.
    async fn spark_land(&self, w: &Workspace<'_>, t: usize, transport: &mut SparkLink<'_>, wave: SparkExpertWave)
        -> Result<()> {
        let ranks = transport.world_size();
        ensure!(ranks <= MAX_RANKS, "{ranks} Spark ranks exceed the reduction planes");
        let timer = std::time::Instant::now();
        transport.receive(wave, t, self.stream).await?;
        self.profile.borrow_mut()[1] += timer.elapsed().as_secs_f64();
        // SAFETY: the shared-expert plane and `delta` are live [t, h] BF16
        // buffers; the planes are ordered after the wave by `receive`.
        unsafe { transport.reduce(w.shared.buffer.ptr.cast(), w.delta.buffer.ptr.cast(), t, self.stream) }
    }

    /// Pipelined Spark prefill of consecutive-row lanes of one sequence (each
    /// with its own workspace and transport). Units (layer, lane) run
    /// layer-major: a lane's attention needs its own previous layer posted and
    /// the previous lane's same layer done, which stream order gives (KDA
    /// state and conv windows, MLA records and DSA token/pool keys all pass
    /// through the engine's caches). Each lane's Spark wave stays in flight
    /// while the other lane's GPU layers run and is received after the next
    /// wave is dispatched, so the ranks hold the next request when they finish
    /// one. Returns the logits of the last `logit_rows` rows across the lanes.
    fn step_lanes(&self, lanes: &[(StepTables, &[u32])], logit_rows: usize, device: bool, media: Option<&cuteafd_engine::media::RequestMedia>)
        -> Result<Option<StepLogits>> {
        let Some(Experts::Spark { transports, runtime }) = &self.experts else {
            anyhow::bail!("pipelined prefill needs Spark experts");
        };
        let mut transports = transports.borrow_mut();
        ensure!(transports.len() >= lanes.len(), "{} lanes need as many Spark transports", lanes.len());
        let workspaces = self.prefill_lanes_of(0, lanes.len())?;
        let total: usize = lanes.iter().map(|(t, _)| t.kv_slots.len()).sum();
        ensure!(logit_rows <= total && (self.full_prefill_logits || logit_rows <= DECODE_ROWS),
            "prefill logits past {DECODE_ROWS} rows need full_prefill_logits");
        for ((tables, tokens), w) in lanes.iter().zip(workspaces.iter()) {
            let t = tables.kv_slots.len();
            ensure!(t <= w.rows, "lane of {t} rows exceeds its workspace");
            self.put_tables(w, tables)?;
            self.load_streams(w, tokens, false)?;
            self.inject_media(w, tables, media)?;
            if media.is_some_and(|m| !m.spans().is_empty()) { self.put(&w.ids, tokens)?; }
        }
        let layers = &self.weights.layers;
        let cap = "m4096";
        let rows_of = |lane: usize| Scalar::I32(lanes[lane].0.kv_slots.len() as i32);
        let count_of = |lane: usize| lanes[lane].0.kv_slots.len();
        // The head split's second GPU: each lane's tables and streams, then every lane's
        // layer-0 attention (rank 1 runs a unit ahead of the host's rank-0 work).
        let peer_workspaces = self.peer_workspaces(false, Some(lanes.len()))?;
        if let Some(peers) = &peer_workspaces {
            // SAFETY: the engine owns the peer stream; drained before its tables are rewritten.
            self.on(1, || unsafe { self.library.cuda_stream_synchronize(self.stream_of(1)) })?;
            for (lane, ((tables, _), w)) in lanes.iter().zip(workspaces.iter()).enumerate() {
                let w1 = peers.get(lane)?;
                self.on(1, || self.put_tables(w1, tables))?;
                self.exchange()?.push_to(0, DIRECT, w.streams[0].buffer.ptr, w1.streams[0].buffer.ptr,
                    count_of(lane) * HC * self.cfg.hidden * 2)?;
            }
            for (lane, (tables, _)) in lanes.iter().enumerate() {
                self.peer_attention(0, lane, peers.get(lane)?, count_of(lane), cap, tables)?;
            }
        }
        // After rank 0 queued unit (layer, lane)'s FFN: rank 1's FFN exchange of that unit and
        // its lane's next attention.
        let peer_next = |(layer, lane): (usize, usize)| -> Result<()> {
            let Some(peers) = &peer_workspaces else { return Ok(()) };
            let w1 = peers.get(lane)?;
            self.peer_post(layer, lane, w1, count_of(lane), cap)?;
            if layer + 1 < layers.len() {
                self.peer_attention(layer + 1, lane, w1, count_of(lane), cap, &lanes[lane].0)?;
            }
            Ok(())
        };
        // The drafter taps the chunk's last TAP_ROWS rows: lane `lane`'s part of
        // that window, at its offset in the tap rows.
        let tap_rows = total.min(crate::families::glm5::dflash::TAP_ROWS);
        let lane_first: Vec<usize> = lanes.iter().scan(0, |first, (t, _)| {
            let here = *first;
            *first += t.kv_slots.len();
            Some(here)
        }).collect();
        // Streams sit in buffer 0 at a layer's start and in buffer 1 mid-layer.
        let attention = |(layer, lane): (usize, usize)| -> Result<()> {
            let (w, weights, rows) = (&workspaces[lane], &layers[layer], rows_of(lane));
            let t = lanes[lane].0.kv_slots.len();
            if layer == 0 {
                self.pre(w, &w.streams[0], weights, rows)?;
            }
            self.attention(0, w, layer, weights, rows, cap, &lanes[lane].0, None)?;
            let attended = self.meet_attention(w, slot(layer, false, lane), t, weights, "m4096")?;
            self.post_pre_on(0, w, attended, 0, weights, "ffn", "post_norm", rows, cap)?;
            if weights.dense {
                self.ffn(w, weights, self.cfg.dense_intermediate, cap, w.delta.buffer.ptr, rows)
            } else {
                self.moe_front(w, layer, weights, t, rows, cap, &[Span { first: 0, rows: t }])
            }
        };
        let post = |(layer, lane): (usize, usize)| -> Result<()> {
            let (w, rows) = (&workspaces[lane], rows_of(lane));
            let out = self.meet_ffn(w, layer, lane, layers.len(), count_of(lane))?;
            match layers.get(layer + 1) {
                Some(next) => self.post_pre_on(0, w, out, 1, next, "attn", "input_norm", rows, cap)?,
                None => self.run("mhc_post", &[("x", out), ("residual", w.streams[1].buffer.ptr),
                    ("prev_post", w.post.buffer.ptr), ("prev_comb", w.comb.buffer.ptr),
                    ("out", w.streams[0].buffer.ptr)], &[rows])?,
            }
            if let Some(drafter) = &self.drafter {
                let (first, t) = (lane_first[lane], lanes[lane].0.kv_slots.len());
                let window = total - tap_rows;
                let from = first.max(window);
                if from < first + t {
                    drafter.tap_streams_at(layer, w.streams[0].buffer.ptr, HC, from - first, first + t - from,
                        from - window)?;
                }
            }
            Ok(())
        };
        let units: Vec<(usize, usize)> = (0..layers.len()).flat_map(|l| (0..lanes.len()).map(move |k| (l, k))).collect();
        runtime.block_on(async {
            let mut inflight: Option<((usize, usize), SparkExpertWave)> = None;
            attention(units[0])?;
            for (index, &unit) in units.iter().enumerate() {
                let (layer, lane) = unit;
                let w = &workspaces[lane];
                if layers[layer].dense {
                    peer_next(unit)?;
                    post(unit)?;
                } else {
                    let t = lanes[lane].0.kv_slots.len();
                    let shared = || self.ffn(w, &layers[layer], self.cfg.moe_intermediate, cap, w.shared.buffer.ptr,
                        rows_of(lane));
                    let wave = self.spark_dispatch(w, layer, t, false, &mut transports[lane], shared)?;
                    peer_next(unit)?;
                    if let Some((previous, wave)) = inflight.take() {
                        let t = lanes[previous.1].0.kv_slots.len();
                        self.spark_land(&workspaces[previous.1], t, &mut transports[previous.1], wave).await?;
                        post(previous)?;
                    }
                    inflight = Some((unit, wave));
                }
                let next = units.get(index + 1).copied();
                // The next unit reads this unit's post when both are on one lane.
                if next.is_none_or(|(_, next_lane)| next_lane == lane) {
                    if let Some((current, wave)) = inflight.take() {
                        let t = lanes[current.1].0.kv_slots.len();
                        self.spark_land(&workspaces[current.1], t, &mut transports[current.1], wave).await?;
                        post(current)?;
                    }
                }
                if let Some(next) = next {
                    attention(next)?;
                }
            }
            anyhow::Ok(())
        })?;
        if logit_rows == 0 {
            self.synchronize()?;
            return Ok(None);
        }
        let timer = std::time::Instant::now();
        let last = lanes.len() - 1;
        if device {
            // The last rows sit in the last lane: their logits stay in its workspace.
            let t = lanes[last].0.kv_slots.len();
            ensure!(logit_rows <= t, "device logits of {logit_rows} rows across prefill lanes");
            let w = &workspaces[last];
            self.run("head", &[("streams", w.streams[0].buffer.ptr), ("weight", self.weights.norm.buffer.ptr),
                ("out", w.x.buffer.ptr)], &[Scalar::I32(t as i32)])?;
            self.logits(w, t, logit_rows, false)?;
            self.profile.borrow_mut()[2] += timer.elapsed().as_secs_f64();
            return Ok(Some(StepLogits::Device(self.device_logits(w, logit_rows, false))));
        }
        let mut logits = Vec::with_capacity(logit_rows * self.cfg.vocab_size);
        for (((tables, _), w), &first) in lanes.iter().zip(workspaces.iter()).zip(&lane_first) {
            let t = tables.kv_slots.len();
            // Rows of this lane inside the last `logit_rows` of the chunk.
            let wanted = (first + t).saturating_sub((total - logit_rows).max(first));
            if wanted == 0 {
                continue;
            }
            self.run("head", &[("streams", w.streams[0].buffer.ptr), ("weight", self.weights.norm.buffer.ptr),
                ("out", w.x.buffer.ptr)], &[Scalar::I32(t as i32)])?;
            self.logits(w, t, wanted, false)?;
            let bytes = self.download(&w.logits, wanted * self.cfg.vocab_size * 4)?;
            logits.extend(bytes.chunks_exact(4).map(|b| f32::from_le_bytes(b.try_into().unwrap())));
        }
        self.profile.borrow_mut()[2] += timer.elapsed().as_secs_f64();
        Ok(Some(StepLogits::Host(logits)))
    }
}

/// A prefill's logits: on the device (one workspace), or gathered across lanes on the host.
pub(crate) enum StepLogits {
    Device(DeviceLogits),
    Host(Vec<f32>),
}

impl StepLogits {
    fn into_host(self, library: &NativeLibrary) -> Result<Vec<f32>> {
        match self {
            Self::Device(logits) => logits.to_host(library),
            Self::Host(logits) => Ok(logits),
        }
    }

    fn device(self) -> Result<DeviceLogits> {
        match self {
            Self::Device(logits) => Ok(logits),
            Self::Host(_) => anyhow::bail!("prefill lanes left their logits on the host"),
        }
    }
}

impl Drop for GlmfEngine<'_> {
    fn drop(&mut self) {
        // SAFETY: the engine owns this stream and its resident weights. Drain
        // queued work, including a failed step, before their storage drops.
        crate::shared::decode_graph::fatal_drain(self.synchronize(), "GLM Flash engine ranks");
        unsafe {
            let _ = self.library.cuda_event_destroy(self.routes_ready);
        }
        if let Some(ops) = self.ops.take() {
            let ops = ops.into_inner();
            for event in ops.pool.into_iter().chain(ops.pending.into_iter().flat_map(|(_, a, b)| [a, b])) {
                // SAFETY: as above; no launch references these events any more.
                let _ = unsafe { self.library.cuda_event_destroy(event) };
            }
        }
    }
}

/// [`DecodeBuckets::bucket`] of the 64-row sets; serving reads the engine's sets. (Test code starts
/// here: the source checks read the engine up to the first `#[cfg(test)]` item.)
#[cfg(test)]
fn decode_bucket(rows: usize, spec: bool) -> usize {
    if spec { SPEC_DECODE_BUCKETS.into_iter().find(|&bucket| bucket >= rows).unwrap_or(rows) }
    else { PLAIN_DECODE_BUCKETS.into_iter().find(|&bucket| bucket >= rows).unwrap_or(rows) }
}

#[cfg(test)]
mod prefill_lane_tests {
    use super::{prefill_lane_capacity, prefill_lane_plan, KdaState, DEFAULT_PREFILL_LANES, MAX_PREFILL_LANES,
        MIN_LANE_ROWS, PAGE_ROWS};

    #[test]
    fn kda_state_selects_its_programs_and_element_size() {
        assert_eq!((KdaState::F32.program("m64"), KdaState::F32.program("m4096"), KdaState::F32.commit_program()),
            ("kda_m64".to_string(), "kda_m4096".to_string(), "kda_commit"));
        assert_eq!((KdaState::Bf16.program("m64"), KdaState::Bf16.program("m4096"), KdaState::Bf16.commit_program()),
            ("kda_s16_m64".to_string(), "kda_s16_m4096".to_string(), "kda_commit_s16"));
        // Per-tile rounding changes only the chunked prefill; decode and commit are per row either way.
        assert_eq!((KdaState::Bf16Tile.program("m64"), KdaState::Bf16Tile.program("m4096"),
            KdaState::Bf16Tile.commit_program()),
            ("kda_s16_m64".to_string(), "kda_s16t_m4096".to_string(), "kda_commit_s16"));
        assert_eq!([KdaState::F32.bytes(), KdaState::Bf16.bytes(), KdaState::Bf16Tile.bytes()], [4, 2, 2]);
        // 34 KDA layers x 16 slots x 64 heads x 128 x 128: 2.28 GB in FP32, 1.14 GB in BF16.
        assert_eq!(34 * 16 * 64 * 128 * 128 * KdaState::F32.bytes(), 2_281_701_376);
        assert_eq!(34 * 16 * 64 * 128 * 128 * KdaState::Bf16.bytes(), 1_140_850_688);
        // With the compact index cache the commit also rebuilds the index tails: its own stems.
        assert_eq!([KdaState::F32, KdaState::Bf16, KdaState::Bf16Tile].map(KdaState::compact_commit_program),
            ["kda_commit_c", "kda_commit_c_s16", "kda_commit_c_s16"]);
        // The loader's step scratch (`GlmfScratchOptions::kda_state`) charges the programs the steps launch.
        for state in [KdaState::F32, KdaState::Bf16, KdaState::Bf16Tile] {
            let loader = super::GlmfKdaState::from(state);
            assert_eq!((loader.program("m64"), loader.program("m4096"), loader.bytes()),
                (state.program("m64"), state.program("m4096"), state.bytes() as u64));
        }
    }

    #[test]
    fn split_audit_counts_routes_and_hashes_token_bytes_in_order() {
        let routes: Vec<_> = [7, 2, 7, 9, 2, 7].into_iter().enumerate().map(|(i, expert_id)|
            super::ExpertProtocolV2RouteEntry { row_index: i as u32 / 2, expert_id, gate_weight: 0.5 }).collect();
        assert_eq!(super::audit_route_counts(&routes), [(2, 2), (7, 3), (9, 1)].into_iter().collect());
        assert!(super::audit_route_counts(&[]).is_empty());
        assert_eq!(super::audit_token_hash(&[]), 0xcbf2_9ce4_8422_2325);
        assert_eq!(super::audit_token_hash(&[1, 2]), 0xc9c2_8939_c996_68c6);
        assert_ne!(super::audit_token_hash(&[1, 2]), super::audit_token_hash(&[2, 1]));
    }

    #[test]
    fn decode_projection_registry_matches_audited_pinned_sources() {
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../third_party/sparkinfer/b12x");
        // Any exporter change requires re-auditing shapes and arithmetic routes.
        for (path, expected) in [
            ("integration/cuteafd/glmf.py", 0x4949_583b_7962_90f6_u64),
            ("integration/cuteafd/_glm_kernels.py", 0xabb3_5df9_3a4c_9796),
            ("integration/cuteafd/_fp8_weights.py", 0xb186_4a9c_36b1_7b30),
            ("integration/cuteafd/dsv4_mhc.py", 0x6b2a_c5a5_1dfa_46c5),
            ("integration/cuteafd/_common.py", 0x74ad_1b69_f557_a0a5),
            ("gemm/bf16_gemv/_skinny.py", 0x26cd_c0f9_cb63_d3a9),
        ] {
            let bytes = std::fs::read(root.join(path)).unwrap();
            let hash = bytes.iter().fold(0xcbf2_9ce4_8422_2325_u64,
                |hash, &byte| (hash ^ u64::from(byte)).wrapping_mul(0x0000_0100_0000_01b3));
            assert_eq!(hash, expected, "re-audit GLM decode projection thresholds after {path} source drift");
        }
        assert_eq!(super::DECODE_PROJECTION_THRESHOLDS.iter().map(|p| p.skinny_rows).collect::<Vec<_>>(),
            [8, 160, 8, 8, 16, 32, 16, 16, 16, 16, 16]);
        // The 64-row sets, and with the wide programs the speculative set ending at the verify budget (127 on
        // 170 SMs, 128 on 188, 99 on 132), pass at every serving width; the `_m128` programs add only
        // their capacity (`WIDE_DECODE_THRESHOLDS`).
        let default = super::DecodeBuckets::new(super::DECODE_ROWS);
        assert_eq!((default.plain.as_slice(), default.spec.as_slice()),
            (&super::PLAIN_DECODE_BUCKETS[..], &super::SPEC_DECODE_BUCKETS[..]));
        for verify in [super::DECODE_ROWS, 99, 127, 128] {
            let buckets = super::DecodeBuckets::new(verify);
            assert_eq!(buckets.spec.last(), Some(&verify));
            for sequences in 1..=super::DECODE_ROWS {
                super::check_decode_thresholds(&buckets, sequences, false).unwrap();
                super::check_decode_thresholds(&buckets, sequences, true).unwrap();
            }
        }
        let error = super::check_bucket_thresholds(&[1, 4, 16], super::DECODE_PROJECTION_THRESHOLDS)
            .unwrap_err().to_string();
        assert!(error.contains("index.iq"));
        assert!(super::check_bucket_thresholds(&[1, 4, 8, 16, 64], super::DECODE_PROJECTION_THRESHOLDS).is_err());
        // A set that pads a step of up to 64 rows into the `_m128` programs is refused.
        let error = super::check_bucket_thresholds(&[32, 72], super::WIDE_DECODE_THRESHOLDS).unwrap_err().to_string();
        assert!(error.contains("m64|m128"), "{error}");
        let crossing = super::DecodeBuckets { plain: super::PLAIN_DECODE_BUCKETS.to_vec(),
            spec: vec![2, 4, 8, 16, 32, 72, 127] };
        let error = super::check_decode_thresholds(&crossing, 16, true).unwrap_err().to_string();
        assert!(error.contains("m64|m128"), "{error}");
    }

    #[test]
    fn scoring_reserve_is_the_configured_lanes_with_a_serial_prefill_in_lane_0() {
        assert_eq!(super::partial_exchange_reserve(2, 4096, 4096, 4), 268_435_456);
        assert_eq!(super::partial_exchange_reserve(2, 4096, 4096, 2), 0);
        assert_eq!(super::partial_exchange_reserve(4, 2048, 4096, 4), super::partial_exchange_reserve(2, 4096, 4096, 4));
        // Whole workspaces kept a serial one beside the lanes (1 + lanes past one lane); shared lanes
        // run a serial prefill in lane 0, so scoring's union is the configured lanes' (with all-row
        // logits in their temporaries: `step_workspaces_reproduce_...`), one allowance per workspace.
        for lanes in 1..=MAX_PREFILL_LANES {
            for drafter in [false, true] {
                assert_eq!(super::workspace_reserve_bytes(10_000, lanes, drafter),
                    10_000 + super::WORKSPACE_RUNTIME_OVERHEAD_BYTES * (1 + lanes + usize::from(drafter)) as u64);
            }
        }
    }

    /// The startup set's fallback, against an admission modelled on a 5090 at 131,072 tokens and 16
    /// sequences: 49,408 tokens of 11,804 B fit beside the planner's 1.5 GiB graph allowance, and the
    /// startup set needs 3,388,063,576 B more (28,060 graphs).
    #[test]
    fn startup_graphs_fall_back_to_lazy_capture_only_on_a_real_shortfall() {
        use super::{admit_beside_decode_graphs, StartupGraphReserve};
        use crate::shared::memory_report::NoKvRoom;
        let allowance = 1_610_612_736;
        let startup = Some(StartupGraphReserve { reserve: allowance + 3_388_063_576, allowance });
        // Whole 256-token units of 11,804 B in the room past the fixed costs and `extra`.
        let modelled = |room: u64| move |extra: u64| -> anyhow::Result<usize> {
            let tokens = room.saturating_sub(extra) / 11_804 / 256 * 256;
            anyhow::ensure!(tokens >= 256, NoKvRoom { free_after_reserve: vec![room as i64 - extra as i64] });
            Ok(tokens as usize)
        };
        let calls = std::cell::RefCell::new(Vec::new());
        // A planned admission: the startup set's bytes above the allowance, nothing for lazy capture.
        let counted = |room: u64| { let admit = modelled(room); let calls = &calls;
            move |graphs: Option<u64>| { calls.borrow_mut().push(graphs);
                admit(graphs.map_or(0, |bytes| bytes.saturating_sub(allowance))) } };
        let set = Some(allowance + 3_388_063_576);
        // Fits: startup capture, one admission keeping the startup set.
        let (tokens, at_startup) = admit_beside_decode_graphs(startup, counted(8_000_000_000)).unwrap();
        assert!(at_startup && tokens == 390_656);
        assert_eq!(calls.take(), [set]);
        // A real shortfall: no room beside the startup set, 49,408 tokens with lazily captured graphs.
        let (tokens, at_startup) = admit_beside_decode_graphs(startup, counted(49_408 * 11_804)).unwrap();
        assert!(!at_startup && tokens == 49_408);
        assert_eq!(calls.take(), [set, None]);
        // No room either way: the startup admission's refusal stands.
        let error = admit_beside_decode_graphs(startup, counted(1_000)).unwrap_err();
        assert!(error.to_string().contains(&format!("{}", 1_000 - 3_388_063_576_i64)), "{error}");
        assert_eq!(calls.take(), [set, None]);
        // Any other refusal is not a shortfall: no retry.
        let failing = |_: Option<u64>| -> anyhow::Result<usize> { anyhow::bail!("CUDA error 2") };
        assert!(admit_beside_decode_graphs(startup, failing).unwrap_err().to_string().contains("CUDA error"));
        // A startup set within the allowance adds nothing to retry without; startup capture off runs lazily.
        let within = Some(StartupGraphReserve { reserve: allowance, allowance });
        assert!(admit_beside_decode_graphs(within, counted(1_000)).is_err());
        assert_eq!(calls.take(), [Some(allowance)]);
        assert_eq!(admit_beside_decode_graphs(None, counted(49_408 * 11_804)).unwrap(), (49_408, false));
        assert_eq!(calls.take(), [None]);
    }

    /// The lazy retry after each refusal for want of memory: a pool falls back to lazy capture; a retry
    /// short of memory too (either kind) keeps the startup refusal; a retry that fails for another
    /// reason returns its own error, with the startup refusal as context. Same startup set as above.
    #[test]
    fn the_lazy_retry_keeps_the_startup_refusal_only_when_it_is_short_too() {
        use super::{admit_beside_decode_graphs, StartupGraphReserve};
        use crate::shared::memory_report::{kv_shortfall, NoKvRoom};
        use cuteafd_core::serving_capacity::CapacityError;
        // The modelled launch above (49,408 tokens of 11,804 B fit beside the allowance alone).
        fn no_room() -> anyhow::Error {
            NoKvRoom { free_after_reserve: vec![583_212_032 - 3_388_063_576] }.into()
        }
        // An automatic pool's one-unit check (`GpuMemoryBudget::admit`), as a 5090 refused it at 131,072
        // tokens and 16 sequences with host embedding; the retry then admitted 91,904 tokens.
        fn over_budget() -> anyhow::Error {
            CapacityError::GpuBudgetExceeded { device: 0, required: 36_016_220_192, budget: 33_711_521_792,
                shortfall: 2_304_698_400 }.into()
        }
        // The retry's refreshed memory sample or checkpoint read failing: none is a shortfall.
        let others: [fn() -> anyhow::Error; 4] = [
            || CapacityError::Invalid("invalid GPU budget or physical memory sample").into(),
            || CapacityError::Overflow("GPU allocation budget").into(),
            || anyhow::anyhow!("cuteafd_cuda_memory_info returned status 1: unspecified launch failure"),
            || anyhow::Error::from(std::io::Error::from(std::io::ErrorKind::NotFound)).context("opening config.json"),
        ];
        let allowance = 1_610_612_736;
        let startup = Some(StartupGraphReserve { reserve: allowance + 3_388_063_576, allowance });
        let calls = std::cell::RefCell::new(Vec::new());
        // The admission keeping the startup set is refused with `first`; the retry answers `retry`.
        let scripted = |first: anyhow::Error, retry: anyhow::Result<usize>| {
            let mut answers = vec![retry, Err(first)];
            let calls = &calls;
            move |graphs: Option<u64>| { calls.borrow_mut().push(graphs); answers.pop().expect("at most two admissions") }
        };
        let both = [Some(allowance + 3_388_063_576), None];
        let chain = |error: &anyhow::Error| error.chain().map(|cause| cause.to_string()).collect::<Vec<_>>();
        let refusals: [(fn() -> anyhow::Error, usize); 2] = [(no_room, 49_408), (over_budget, 91_904)];
        for (refusal, pool) in refusals {
            // A pool with lazily captured graphs: lazy capture.
            let admitted = admit_beside_decode_graphs(startup, scripted(refusal(), Ok(pool))).unwrap();
            assert_eq!(admitted, (pool, false));
            assert_eq!(calls.take(), both);
            // Short either way: the startup refusal, unchanged.
            for (short, _) in refusals {
                let error = admit_beside_decode_graphs(startup, scripted(refusal(), Err(short()))).unwrap_err();
                assert!(kv_shortfall(&error));
                assert_eq!(chain(&error), chain(&refusal()));
                assert_eq!(calls.take(), both);
            }
            // Another failure: the retry's own error, whole, beneath the startup refusal as context.
            for other in others {
                let error = admit_beside_decode_graphs(startup, scripted(refusal(), Err(other()))).unwrap_err();
                assert!(!kv_shortfall(&error), "{error:#}");
                assert_eq!(&chain(&error)[1..], chain(&other()).as_slice());
                assert!(error.to_string().contains(&format!("({:#})", refusal())), "{error:#}");
                assert_eq!(calls.take(), both);
            }
        }
    }

    #[test]
    fn startup_defaults_honor_both_explicit_disables() {
        assert!(super::startup_graph_policy(None, None));
        assert!(super::startup_graph_policy(Some("1"), Some("1")));
        assert!(!super::startup_graph_policy(Some("0"), None));
        assert!(!super::startup_graph_policy(None, Some("0")));
        assert_eq!(super::prefill_workspace_count(true, true, None, 2), 2);
        assert_eq!(super::prefill_workspace_count(true, true, None, 4), 4);
        assert_eq!(super::prefill_workspace_count(true, true, Some("1"), 4), 1);
        assert_eq!(super::prefill_workspace_count(true, false, None, 2), 1);
        assert_eq!(super::prefill_workspace_count(true, false, Some("subset"), 2), 2);
        assert_eq!(super::prefill_workspace_count(false, true, None, 2), 1);
    }

    #[test]
    fn step_workspaces_reproduce_the_frozen_accounting_and_share_the_lanes_temporaries() {
        use cuteafd_loader::families::glm5_flash::{GlmNextConfig, GlmNextAttention};
        use cuteafd_loader::serving_capacity::{glmf_lane_bytes, glmf_step_workspaces, glmf_temporary_bytes, GlmfScratch,
            GlmfStepShape};
        let cfg = GlmNextConfig { vocab_size: 154880, hidden: 4096, layers: 2,
            attention: vec![GlmNextAttention::Kda, GlmNextAttention::Mla], dense: vec![false; 2],
            dense_intermediate: 12288, experts: 288, topk: 8, moe_intermediate: 2048,
            routed_scale: 2.5, swiglu_limit: 10.0, rms_norm_eps: 1e-5, hc_mult: 4,
            kda_heads: 64, kda_head_dim: 128, heads: 64, q_lora_rank: 1536, kv_lora_rank: 512,
            qk_nope_dim: 256, v_head_dim: 256, index_topk: 2048, index_kpool: 4, eos: vec![] };
        // Frozen99c PROGRAMS.json capacities; actual ledger counts, not scaled guesses.
        let decode = GlmfScratch { programs: 26_214_400, topk: 8_653_824 };
        let prefill = GlmfScratch { programs: 782_236_672, topk: 558_007_296 };
        let shape = GlmfStepShape { lead: true, split: false, local_experts: true, spark: false, partial_bytes: 2,
            output_shard: false, full_prefill_logits: false, table_pages: 32768, table_pool_pages: 8192 };
        // A whole workspace per lane, with the head-split rows every workspace used to hold: the
        // frozen accounting of the layout before the lanes shared their temporaries.
        let whole = |rows: u64, step_decode: bool, scratch: GlmfScratch| {
            let mut lane = glmf_lane_bytes(&cfg, rows, step_decode, &shape);
            (lane.zero, lane.sum) = (Some(rows * 4096 * 2), Some(rows * 4096 * 2));
            lane.device_bytes() + glmf_temporary_bytes(&cfg, rows, step_decode, &shape, scratch).device_bytes()
        };
        assert_eq!(whole(4096, false, prefill), 2_486_583_296);
        assert_eq!(whole(64, true, decode), 106_421_504);
        let tracked = 2 * whole(4096, false, prefill) + whole(64, true, decode);
        assert_eq!(tracked, 5_079_588_096);
        assert_eq!(tracked - 5_068_061_409, 11_526_687);
        // Two lanes over one set of temporaries, without a head split's rows: 2.13 GB less.
        let shared = glmf_step_workspaces(&cfg, 2, 4096, 64, &shape, decode, prefill).device_bytes();
        assert_eq!(shared, 2_950_470_912);
        let spark = GlmfStepShape { local_experts: false, spark: true, ..shape };
        assert_eq!(glmf_step_workspaces(&cfg, 2, 4096, 64, &spark, decode, prefill).device_bytes(), 2_882_837_760);
        let reserve = super::workspace_reserve_bytes(shared, 2, true);
        assert_eq!(reserve, shared + 4 * super::WORKSPACE_RUNTIME_OVERHEAD_BYTES);
        assert!(reserve >= shared + 3 * 71_942_144 + 68_269_888);
        // The programs' scratch keeps its capacity whatever a lane's rows.
        for rows in [1, 64, 512, 4096] {
            let temps = glmf_temporary_bytes(&cfg, rows, false, &shape, prefill);
            assert_eq!((temps.scratch, temps.topk_scratch), (782_236_672, 558_007_296));
        }
        let full = GlmfStepShape { full_prefill_logits: true, ..shape };
        assert_eq!(glmf_temporary_bytes(&cfg, 4096, false, &full, prefill).logits
            - glmf_temporary_bytes(&cfg, 4096, false, &shape, prefill).logits, (4096 - 64) * 154_880 * 4);
        // Rank 1 holds no head, logits or router rows (each floored to 256 bytes when allocated),
        // with all-row logits or without.
        let peer = GlmfStepShape { lead: false, ..shape };
        let temps = glmf_temporary_bytes(&cfg, 4096, false, &peer, prefill);
        assert_eq!((temps.logits, temps.head_workspace), (0, 0));
        assert_eq!(glmf_temporary_bytes(&cfg, 4096, false, &GlmfStepShape { full_prefill_logits: true, ..peer }, prefill),
            temps);
        assert!(glmf_lane_bytes(&cfg, 4096, false, &peer).device_bytes()
            < glmf_lane_bytes(&cfg, 4096, false, &shape).device_bytes());
        // Scoring's exact union (`--full-prefill-logits`): the same lanes, a serial prefill in lane 0,
        // the all-row logits once in their shared temporaries. Whole workspaces held them three
        // times (a serial workspace and two lanes) at four runtime allowances.
        let scoring = glmf_step_workspaces(&cfg, 2, 4096, 64, &full, decode, prefill).device_bytes();
        assert_eq!(scoring - shared, (4096 - 64) * 154_880 * 4);
        assert_eq!(super::workspace_reserve_bytes(scoring, 2, false), scoring + 3 * super::WORKSPACE_RUNTIME_OVERHEAD_BYTES);
    }

    #[test]
    fn million_token_graphs_add_only_three_long_geometries() {
        use super::*;
        let pages = 2_097_152 / PAGE_ROWS;
        for (verify, expected) in [(64, 300), (128, 330)] {
            let buckets = DecodeBuckets::new(verify);
            let baseline = serving_graph_shapes(131_072, pages, 2051, 16, true, &buckets);
            let extended = serving_graph_shapes(1_048_576, pages, 2051, 16, true, &buckets);
            assert_eq!(extended.len(), expected);
            assert!(baseline.iter().all(|key| extended.contains(key)));
            for allocated in [131_072usize, 262_144, 524_288, 1_048_576] {
                for live in [1usize, 8192, 131_072, 131_073, 200_000, 300_000, 600_000] {
                    if live > allocated { continue; }
                    let units = crate::shared::context::decode_allocation_units(
                        allocated.div_ceil(UNIT_ROWS), live, UNIT_ROWS);
                    let (page_stride, pool_stride) = decode_strides(units * UNIT_PAGES, units,
                        (pages, pages / UNIT_PAGES), (16384, 4096));
                    let geometry = GraphGeometry::keyed(live.div_ceil(UNIT_ROWS).next_power_of_two()
                        .min(pool_stride), page_stride, pool_stride, live > 2051);
                    for &(rows, spec, _) in &baseline {
                        assert!(extended.contains(&(rows, spec, geometry)), "{allocated}/{live}/{geometry:?}");
                    }
                }
            }
            let reserves = serving_graph_reserve(1_048_576, 2_097_152, 2051, 16, true, 45, true, &buckets);
            let old = serving_graph_reserve(131_072, 2_097_152, 2051, 16, true, 45, true, &buckets);
            assert!(reserves.iter().zip(old).all(|(new, old)| new - old < (512 << 20)));
        }
    }

    #[test]
    fn startup_graph_shapes_cover_all_admitted_capacities_and_positions() {
        use super::{decode_bucket, glmf_table_pages, serving_graph_shapes, DecodeBuckets, GraphGeometry, GraphKey,
            StepTables, DECODE_ROWS, MIN_PAGE_STRIDE, UNIT_PAGES, UNIT_ROWS};
        let buckets = DecodeBuckets::new(DECODE_ROWS);
        for (context, pages, dense) in [(32768usize, 4096usize, 2051usize), (777, 28, 99), (3000, 36, 2051)] {
            let shapes = serving_graph_shapes(context, pages, dense, 16, true, &buckets);
            let set: std::collections::HashSet<_> = shapes.iter().copied().collect();
            assert_eq!(shapes.len(), set.len());
            let (table_pages, table_pools) = glmf_table_pages(context as u64);
            for capacity in 1..=context.min(pages / UNIT_PAGES * UNIT_ROWS) {
                let units = capacity.div_ceil(UNIT_ROWS);
                let mut lengths = vec![1, capacity, dense.min(capacity), (dense + 1).min(capacity)];
                for bit in 0..usize::BITS {
                    let Some(boundary) = UNIT_ROWS.checked_shl(bit) else { break };
                    if boundary >= capacity { break; }
                    lengths.extend([boundary, boundary + 1]);
                }
                for len in lengths {
                    // The decode step's tables: power-of-two strides (the page stride from its floor), within the
                    // pool and a table row; the top-k width within the pool stride.
                    let page_stride = (units * UNIT_PAGES).next_power_of_two().max(MIN_PAGE_STRIDE).min(pages)
                        .min(table_pages as usize);
                    let pool_stride = units.next_power_of_two().min(pages / UNIT_PAGES)
                        .min(table_pools as usize);
                    let pool_width = len.div_ceil(UNIT_ROWS).next_power_of_two().min(pool_stride);
                    let long = len > dense;
                    for (rows, spec) in [(1, false), (3, false), (10, false), (16, false), (2, true), (10, true), (64, true)] {
                        let tables = StepTables { decode: true, spec, long, pool_width, page_stride, pool_stride,
                            ..Default::default() };
                        let key = GraphKey::new(0, decode_bucket(rows, spec), &tables);
                        let geometry = GraphGeometry { pool_width: key.pool_width, page_stride: key.page_stride,
                            pool_stride: key.pool_stride, long: key.long };
                        // A short step's key holds no pool-table width or stride.
                        assert_eq!((key.pool_width, key.pool_stride), if long { (pool_width, pool_stride) } else { (0, 0) });
                        assert!(set.contains(&(key.rows, spec, geometry)), "missing {rows}/{spec}/{geometry:?}");
                    }
                }
            }
        }
        // 14 geometries of 32,768 tokens (four short ones, one per page stride, and ten long ones), each
        // at 4 plain and 6 speculative buckets.
        assert_eq!(serving_graph_shapes(32768, 4096, 2051, 16, true, &buckets).len(), 140);
        assert_eq!(serving_graph_shapes(32768, 4096, 2051, 16, false, &buckets).len(), 56);
    }

    #[test]
    fn speculation_buckets_and_pre_kv_reserve_match_serving_policy() {
        for rows in 1..=64 {
            let bucket = super::decode_bucket(rows, true);
            assert!(bucket >= rows && bucket <= 2 * rows);
        }
        assert_eq!(super::decode_bucket(8, true), 8);
        assert_eq!(super::decode_bucket(32, true), 32);
        assert_eq!(super::decode_bucket(5, false), 8);
        assert_eq!(super::decode_bucket(8, false), 8);
        assert_eq!(super::decode_bucket(9, false), 16);
        assert_eq!(super::decode_bucket(17, false), 32);
        let buckets = super::DecodeBuckets::new(super::DECODE_ROWS);
        for sequences in [8, 16] {
            let shapes = super::serving_graph_shapes(32768, 32768, 2051, sequences, true, &buckets);
            assert_eq!(shapes.len() * 46, 6440);
            let reserve = super::serving_graph_reserve(32768, 2_097_152, 2051, sequences, true, 45, true, &buckets);
            assert_eq!(reserve, [1_198_944_016, super::graph_reserve_bytes(6300)]);
            assert_eq!(super::serving_graph_shapes(32768, 32768, 2051, sequences, false, &buckets).len() * 46, 2576);
        }
        assert_eq!(super::serving_graph_shapes(32768, 32768, 2051, 32, true, &buckets).len() * 46, 7084);
        assert_eq!(super::serving_graph_shapes(32768, 32768, 2051, 64, true, &buckets).len() * 46, 7728);
    }

    #[test]
    fn decode_padding_masks_storage_and_preserves_real_tables() {
        for (rows, bucket) in [(3, 4), (9, 16), (17, 32), (33, 64)] {
            let mut tables = super::StepTables { decode: true, page_stride: 8, pool_stride: 2,
                positions: vec![17; rows], kv_slots: vec![19; rows], kda_slots: vec![2; rows],
                seq_first: (0..rows as i32).collect(), pool_slots: vec![-1; rows], cache_lengths: vec![4; rows],
                page_table: vec![3; rows * 8], pool_table: vec![1; rows * 2], ..Default::default() };
            super::pad_decode_tables(&mut tables, bucket);
            assert_eq!(&tables.positions[..rows], vec![17; rows]);
            assert_eq!(&tables.kv_slots[..rows], vec![19; rows]);
            assert_eq!(&tables.positions[rows..], vec![-1; bucket - rows]);
            assert_eq!(&tables.kv_slots[rows..], vec![-1; bucket - rows]);
            assert_eq!(&tables.kda_slots[rows..], vec![-1; bucket - rows]);
            assert_eq!(&tables.cache_lengths[rows..], vec![0; bucket - rows]);
            assert_eq!(tables.page_table.len(), bucket * 8);
            assert_eq!(tables.pool_table.len(), bucket * 2);
            assert_eq!(super::decode_bucket(rows, true), bucket);
        }
    }

    #[test]
    fn output_token_rows_cover_odd_batches_and_zero_owned_rank() {
        for rows in [1, 22, 63, 64, 512, 513, 4096] {
            let lead = super::output_rows(rows, 0);
            let peer = super::output_rows(rows, 1);
            assert_eq!(lead.0, 0);
            assert_eq!(peer.0, lead.1);
            assert_eq!(lead.1 + peer.1, rows);
            // Norm rows sent + completed output rows equal the original
            // BF16 partial's bytes, even for an odd or one-token batch.
            for (_, owned) in [lead, peer] {
                assert_eq!((rows - owned) * 4096 * 2 + owned * 4096 * 2, rows * 4096 * 2);
            }
        }
        assert_eq!(super::output_rows(1, 1), (1, 0));
    }

    #[test]
    fn output_shard_norm_slots_isolate_every_lane_and_layer_parity() {
        for lanes in 1..=MAX_PREFILL_LANES {
            let mut heads = std::collections::BTreeSet::new();
            let mut existing = std::collections::BTreeSet::new();
            for lane in 0..lanes {
                for layer in 0..2 {
                    existing.insert(super::slot(layer, false, lane));
                    existing.insert(super::slot(layer, true, lane));
                    let slot = super::norm_slot(lanes, super::slot(layer, false, lane));
                    assert_eq!(slot, super::norm_slot(lanes, super::slot(layer + 2, false, lane)));
                    assert!(heads.insert(slot));
                }
            }
            assert!(heads.is_disjoint(&existing));
            // The exchange holds six slots per lane with the output shard.
            assert_eq!(heads, (4 * lanes..6 * lanes).collect(), "{lanes} lanes");
        }
    }

    #[test]
    fn every_advertised_prefill_tail_fits_its_lane_workspaces() {
        for lanes in 1..=MAX_PREFILL_LANES {
            for rows in [1, 63, 64, 65, 96, 127, 128, 255, 256, 511, 512, 1024, 1536, 2047, 2048, 4096] {
                let capacity = prefill_lane_capacity(lanes, rows);
                for tokens in 1..=capacity {
                    let (used, per_lane) = prefill_lane_plan(tokens, lanes, rows).unwrap();
                    assert!(used <= lanes && per_lane <= rows, "lanes={lanes}, rows={rows}, tokens={tokens}");
                    let starts: Vec<_> = (0..tokens).step_by(per_lane).collect();
                    assert_eq!(starts.len(), used);
                    assert_eq!(starts.iter().map(|&s| per_lane.min(tokens - s)).sum::<usize>(), tokens);
                    assert!(starts.iter().all(|&s| per_lane.min(tokens - s) <= rows));
                    assert!(starts.iter().skip(1).all(|s| s % PAGE_ROWS == 0));
                }
                assert!(prefill_lane_plan(capacity + 1, lanes, rows).is_err());
            }
        }
    }

    #[test]
    fn two_lanes_keep_their_cuts_for_every_chunk() {
        // The rule before the lane count was a setting: identical cuts give identical bits.
        let before = |tokens: usize, rows: usize| {
            let lanes = if tokens <= rows && tokens < 2 * MIN_LANE_ROWS { 1 } else { 2 };
            if lanes == 1 { tokens } else { tokens.div_ceil(lanes).next_multiple_of(PAGE_ROWS) }
        };
        for rows in [64, 128, 256, 2048, 4096] {
            for tokens in 1..=prefill_lane_capacity(2, rows) {
                assert_eq!(prefill_lane_plan(tokens, 2, rows).unwrap().1, before(tokens, rows), "{tokens} of {rows}");
            }
        }
    }

    #[test]
    fn four_lanes_of_2048_hold_a_chunk_of_8192() {
        assert_eq!(prefill_lane_capacity(4, 2048), 8192);
        assert_eq!(prefill_lane_plan(8192, 4, 2048).unwrap(), (4, 2048));
        assert_eq!(prefill_lane_plan(4096, 4, 2048).unwrap(), (4, 1024));
        // One lane per 256 rows, cut on 64-row pages.
        assert_eq!(prefill_lane_plan(1000, 4, 2048).unwrap(), (3, 384));
        assert_eq!(prefill_lane_plan(511, 4, 2048).unwrap(), (1, 511));
        // One lane: the chunk is the lane.
        assert_eq!(prefill_lane_capacity(1, 4096), 4096);
        assert_eq!(prefill_lane_plan(4096, 1, 4096).unwrap(), (1, 4096));
        assert!(prefill_lane_plan(4097, 1, 4096).is_err());
    }

    #[test]
    fn the_workspace_arithmetic_uses_the_engine_geometry() {
        use cuteafd_loader::serving_capacity::{GLMF_DECODE_ROWS, GLMF_HEAD_WORKSPACE, GLMF_SPARSE_TOPK,
            GLMF_WIDE_DECODE_ROWS};
        assert_eq!(GLMF_DECODE_ROWS, super::DECODE_ROWS as u64);
        assert_eq!(GLMF_WIDE_DECODE_ROWS, super::WIDE_DECODE_ROWS as u64);
        assert_eq!(GLMF_SPARSE_TOPK, super::SPARSE_TOPK as u64);
        assert_eq!(GLMF_HEAD_WORKSPACE, cuteafd_ffi::programs::VOCABULARY_HEAD_WORKSPACE as u64);
    }

    #[test]
    fn single_pass_rows_are_the_chunks_the_lane_plan_leaves_in_one_lane() {
        for (lanes, rows) in [(2, 4096), (4, 2048), (3, 1024), (2, 256), (1, 4096)] {
            let single = super::single_pass_rows(true, lanes, rows);
            for tokens in 1..=prefill_lane_capacity(lanes, rows) {
                let (cut, _) = prefill_lane_plan(tokens, lanes, rows).unwrap();
                assert_eq!(cut == 1, tokens <= single, "{lanes} lanes of {rows}: {tokens} tokens in {cut} lanes");
            }
        }
        assert_eq!(super::single_pass_rows(false, 2, 4096), 4096);
    }

    #[test]
    fn narrow_workspace_splits_the_short_tool_prompt() {
        assert_eq!(prefill_lane_plan(214, 2, 128).unwrap(), (2, 128));
        assert_eq!(prefill_lane_plan(1, 2, 1).unwrap(), (1, 1));
        assert!(prefill_lane_plan(0, 2, 128).is_err());
        assert!(prefill_lane_plan(1, 2, 0).is_err());
        // Keep the qualified default's lane threshold and advertised width.
        assert_eq!(prefill_lane_capacity(DEFAULT_PREFILL_LANES, 4096), 8192);
        assert_eq!(prefill_lane_plan(511, 2, 4096).unwrap(), (1, 511));
        assert_eq!(prefill_lane_plan(512, 2, 4096).unwrap(), (2, 256));
    }
}

#[cfg(test)]
mod wide_decode_tests {
    use super::{commit_for, decode_cap, hand_out_remainder, is_decode, replay_bytes, replay_rows_of, verify_budget,
        DecodeBuckets, KdaState, DECODE_ROWS, KEY_BYTES, WIDE_DECODE_ROWS};

    /// Steps of up to 64 rows keep the `_m64` programs, their 64-row records and commits; wider steps
    /// run the `_m128` programs, record 128 rows per layer and commit with the `_m128` commits.
    #[test]
    fn wide_steps_run_the_m128_programs_and_their_commits() {
        assert_eq!([1, 16, 63, 64].map(decode_cap), ["m64"; 4]);
        assert_eq!([65, 100, 127, 128].map(decode_cap), ["m128"; 4]);
        assert!(is_decode("m64") && is_decode("m128") && !is_decode("m4096"));
        assert_eq!((replay_rows_of("m64"), replay_rows_of("m128")), (DECODE_ROWS, WIDE_DECODE_ROWS));
        for state in [KdaState::F32, KdaState::Bf16, KdaState::Bf16Tile] {
            // Either capacity's KDA program is the state's own.
            assert_eq!(state.program(decode_cap(100)), state.program("m64").replace("m64", "m128"));
            for base in [state.commit_program(), state.compact_commit_program()] {
                assert_eq!(commit_for(base, replay_rows_of(decode_cap(64))), base);
                assert_eq!(commit_for(base, replay_rows_of(decode_cap(65))), format!("{base}_m128"));
            }
        }
        assert_eq!(commit_for("kda_commit_c_s16", 128), "kda_commit_c_s16_m128");
        // One layer's record over 64 KDA heads: 9,453,568 B at 64 rows, twice that at 128.
        assert_eq!(replay_bytes(64, 3 * 64 * 128, 64), 9_453_568);
        assert_eq!(replay_bytes(64, 3 * 64 * 128, 128), 2 * 9_453_568);
    }

    /// The engine's caches hold what the measured and planned admissions and the planner charge
    /// (`glm_flash_rank_cache_geometry_rows`): `Caches::new`'s KDA records of every layer, the compact
    /// index cache's key | gate records and the commit tables, at either decode rows; and the KDA records
    /// alone (`glm_flash_kda_replay_bytes_rows`) where `--replay-records shared` places them.
    #[test]
    fn the_caches_hold_the_records_the_admissions_charge() {
        use cuteafd_loader::families::glm5_flash::{GlmNextAttention, GlmNextConfig};
        use cuteafd_loader::serving_capacity::{glm_flash_kda_replay_bytes_rows, glm_flash_rank_cache_geometry_rows,
            GlmfIndexCache};
        let cfg = GlmNextConfig { vocab_size: 154880, hidden: 4096, layers: 45,
            attention: (0..45).map(|i| if i % 4 == 3 { GlmNextAttention::Mla } else { GlmNextAttention::Kda }).collect(),
            dense: vec![false; 45], dense_intermediate: 12288, experts: 288, topk: 8, moe_intermediate: 2048,
            routed_scale: 2.5, swiglu_limit: 10.0, rms_norm_eps: 1e-5, hc_mult: 4, kda_heads: 64, kda_head_dim: 128,
            heads: 64, q_lora_rank: 1536, kv_lora_rank: 512, qk_nope_dim: 256, v_head_dim: 256, index_topk: 2048,
            index_kpool: 4, eos: vec![] };
        let (kda, mla) = (34, 11);
        for rows in [DECODE_ROWS, WIDE_DECODE_ROWS] {
            // Shared records: `StepPlan::new`'s and `Caches::new`'s region, what the measured admission and
            // the planner take out of the state.
            assert_eq!(glm_flash_kda_replay_bytes_rows(&cfg, 45, 1, rows as u64).unwrap(),
                (kda * replay_bytes(64, 3 * 64 * 128, rows)) as u64);
            for (index, compact) in [(GlmfIndexCache::Keys, false), (GlmfIndexCache::Compact, true)] {
                let geometry = glm_flash_rank_cache_geometry_rows(&cfg, 45, 1, index, 4, rows as u64).unwrap();
                let rank = &geometry.ranks[0];
                let records = kda * replay_bytes(64, 3 * 64 * 128, rows) + if compact { mla * rows * KEY_BYTES } else { 0 };
                assert_eq!(rank.speculative_replay_bytes, records as u64, "{rows} rows, {index:?}");
                assert_eq!(rank.fixed_state_bytes, (3 * rows * 4) as u64);
            }
        }
    }

    /// The verify budget: the largest row count whose one-split wide sparse MLA (4 CTAs a row) fits
    /// three waves of the GPU's SMs, at least 64; 64 decode rows keep 64.
    #[test]
    fn the_verify_budget_keeps_the_wide_sparse_mla_in_three_waves() {
        assert_eq!(verify_budget(WIDE_DECODE_ROWS, 170), 127);
        assert_eq!(verify_budget(WIDE_DECODE_ROWS, 188), 128);
        assert_eq!(verify_budget(WIDE_DECODE_ROWS, 132), 99);
        for sms in [86, 100, 132, 170, 188, 200] {
            let rows = verify_budget(WIDE_DECODE_ROWS, sms);
            assert!(4 * rows <= 3 * sms && (rows == WIDE_DECODE_ROWS || 4 * (rows + 1) > 3 * sms), "{sms} SMs");
        }
        // 85 SMs or fewer keep the 64-row programs' 64, and so does --decode-rows 64 on any GPU.
        assert_eq!(verify_budget(WIDE_DECODE_ROWS, 84), DECODE_ROWS);
        for sms in [48, 84, 170, 188] {
            assert_eq!(verify_budget(DECODE_ROWS, sms), DECODE_ROWS);
        }
    }

    /// The rows an even share of the budget leaves over go to the first sequences that can draft once
    /// more: 127 rows at 16 sequences, 15 sequences draft 7 and one 6.
    #[test]
    fn the_remainder_goes_to_the_first_sequences_that_can_draft_again() {
        let room_of = |rows: usize, sequences: usize| (rows / sequences).max(1) - 1;
        let room = room_of(127, 16);
        assert_eq!(room, 6);
        let mut limits = vec![room; 16];
        hand_out_remainder(&mut limits, room, 127, |_| true);
        assert_eq!(limits.iter().filter(|&&limit| limit == 7).count(), 15);
        assert_eq!(limits.iter().map(|limit| limit + 1).sum::<usize>(), 127);
        // A sequence that cannot draft again (no speculation, too few tokens or too little capacity
        // left) or was cut below the room keeps its limit; the next ones take the rows.
        let mut limits = vec![room; 16];
        limits[1] = 2;
        hand_out_remainder(&mut limits, room, 127, |i| i > 2);
        assert_eq!(&limits[..4], &[6, 2, 6, 7]);
        assert!(limits.iter().map(|limit| limit + 1).sum::<usize>() <= 127);
        // A share that leaves nothing over changes nothing: 64 or 128 rows at 16 sequences.
        for rows in [64, 128] {
            let room = room_of(rows, 16);
            let mut limits = vec![room; 16];
            hand_out_remainder(&mut limits, room, rows, |_| true);
            assert_eq!(limits, vec![room; 16]);
        }
        let mut limits = vec![126];
        hand_out_remainder(&mut limits, 126, 127, |_| true);
        assert_eq!(limits, [126]);
    }

    /// The speculative buckets end at the verify budget; every step pads within its program capacity.
    #[test]
    fn wide_buckets_end_at_the_verify_budget() {
        let wide = DecodeBuckets::new(127);
        assert_eq!(wide.spec, [2, 4, 8, 16, 32, 64, 127]);
        assert_eq!(wide.plain, [1, 4, 8, 16, 32, 64]);
        assert_eq!(DecodeBuckets::new(128).spec.last(), Some(&128));
        assert_eq!(DecodeBuckets::new(99).spec.last(), Some(&99));
        for rows in 1..=127 {
            let bucket = wide.bucket(rows, true);
            assert!(bucket >= rows && bucket <= 127, "{rows}");
            assert_eq!(decode_cap(bucket), decode_cap(rows), "{rows}");
            if rows <= DECODE_ROWS {
                assert_eq!(bucket, super::decode_bucket(rows, true));
            }
        }
        assert_eq!([65, 100, 126, 127].map(|rows| wide.bucket(rows, true)), [127; 4]);
        for rows in 1..=DECODE_ROWS {
            assert_eq!(wide.bucket(rows, false), super::decode_bucket(rows, false));
        }
    }

    /// The startup set grows by one speculative bucket per geometry: at 16 sequences and the 2,097,152-
    /// token pool target, 50 -> 55 shapes at 8,192 tokens, 140 -> 154 at 32,768 and 270 -> 297 at
    /// 131,072 (the geometries of the graph keys); at 8,192 tokens its reserve grows by 230 graphs
    /// (40,422,684 B). Each added shape is the budget's bucket, keyed as its own shape by its capture's
    /// padded tables.
    #[test]
    fn the_startup_set_grows_by_one_speculative_bucket_per_geometry() {
        use super::{graph_reserve_bytes, serving_graph_reserve, serving_graph_shapes, startup_tables, GraphKey, PAGE_ROWS};
        let pages = 2_097_152usize.div_ceil(PAGE_ROWS);
        let (narrow, wide) = (DecodeBuckets::new(DECODE_ROWS), DecodeBuckets::new(127));
        for (context, before, after) in [(8_192usize, 50usize, 55usize), (32_768, 140, 154), (131_072, 270, 297)] {
            let narrow_shapes = serving_graph_shapes(context, pages, 2051, 16, true, &narrow);
            let wide_shapes = serving_graph_shapes(context, pages, 2051, 16, true, &wide);
            assert_eq!((narrow_shapes.len(), wide_shapes.len()), (before, after));
            let added: Vec<_> = wide_shapes.iter().filter(|shape| !narrow_shapes.contains(shape)).copied().collect();
            assert_eq!(added.len(), after - before);
            for (rows, spec, geometry) in added {
                assert_eq!((rows, spec), (127, true));
                let tables = startup_tables(rows, spec, geometry);
                assert_eq!(GraphKey::new(0, rows, &tables), GraphKey { segment: 0, rows, spec, long: geometry.long,
                    pool_width: geometry.pool_width, page_stride: geometry.page_stride, pool_stride: geometry.pool_stride });
            }
            // The plain set is unchanged.
            assert_eq!(serving_graph_shapes(context, pages, 2051, 16, false, &wide),
                serving_graph_shapes(context, pages, 2051, 16, false, &narrow));
        }
        let reserve = |buckets: &DecodeBuckets| serving_graph_reserve(8_192, 2_097_152, 2051, 16, true, 45, false,
            buckets)[0];
        assert_eq!(reserve(&narrow), graph_reserve_bytes(50 * 46));
        assert_eq!(reserve(&narrow), 471_335_704);
        assert_eq!(reserve(&wide) - reserve(&narrow), 40_422_684);
    }

    /// A padded wide step (65, 100 or 126 -> 127 rows) keeps its real tables, masks the rest, runs the
    /// router, wire and experts on its real rows only and then clears exactly `real..bucket` of the
    /// decode workspace's `delta`, which holds 128 rows.
    #[test]
    fn padded_wide_steps_keep_padding_out_of_the_router_and_off_the_expert_wire() {
        let buckets = DecodeBuckets::new(127);
        for rows in [65usize, 100, 126] {
            let bucket = buckets.bucket(rows, true);
            assert_eq!(bucket, 127);
            let mut tables = super::StepTables { decode: true, spec: true, real_rows: rows, page_stride: 8,
                pool_stride: 2, positions: vec![17; rows], kv_slots: vec![19; rows], kda_slots: vec![2; rows],
                seq_first: (0..rows as i32).collect(), pool_slots: vec![-1; rows], cache_lengths: vec![4; rows],
                page_table: vec![3; rows * 8], pool_table: vec![1; rows * 2], ..Default::default() };
            super::pad_decode_tables(&mut tables, bucket);
            assert_eq!(&tables.positions[..rows], vec![17; rows]);
            assert_eq!(&tables.kv_slots[..rows], vec![19; rows]);
            assert!(tables.positions[rows..].iter().chain(&tables.kv_slots[rows..]).all(|&v| v == -1));
            assert!(tables.kda_slots[rows..].iter().all(|&slot| slot == -1));
            assert_eq!((tables.kv_slots.len(), tables.page_table.len(), tables.real_rows), (bucket, bucket * 8, rows));
            let (mut ran, mut cleared) = (None, None);
            crate::shared::decode_graph::real_row_moe(tables.real_rows, bucket, |real| { ran = Some(real); Ok(()) },
                |tail| { cleared = Some(tail); Ok(()) }).unwrap();
            assert_eq!((ran, cleared), (Some(rows), Some(rows..bucket)));
            assert!(bucket <= WIDE_DECODE_ROWS && decode_cap(bucket) == "m128");
        }
    }
}

#[cfg(test)]
mod graph_key_tests {
    use super::{check_bucket_thresholds, check_decode_thresholds, decode_bucket, decode_strides, glmf_table_pages,
        graph_geometries, graph_reserve_bytes, serving_graph_reserve, serving_graph_shapes, startup_tables,
        DecodeBuckets, GraphGeometry, GraphKey, StepTables, DECODE_PROJECTION_THRESHOLDS, DECODE_ROWS, MIN_PAGE_STRIDE,
        PLAIN_DECODE_BUCKETS, SPEC_DECODE_BUCKETS, UNIT_PAGES, UNIT_ROWS};
    use std::collections::HashSet;

    fn tables(long: bool, pool_width: usize, page_stride: usize, pool_stride: usize) -> StepTables {
        StepTables { decode: true, long, pool_width, page_stride, pool_stride, spec: true, ..Default::default() }
    }

    #[test]
    fn short_steps_share_graphs_whatever_their_pool_top_k_shape() {
        // The pool top-k (the only reader of the width and the pool table stride) runs only when a
        // row sees past the dense context.
        let key = |t: &StepTables| GraphKey::new(3, 40, t);
        assert_eq!(key(&tables(false, 1, 64, 16)), key(&tables(false, 4, 64, 32)));
        assert_eq!((key(&tables(false, 8, 64, 16)).pool_width, key(&tables(false, 8, 64, 16)).pool_stride), (0, 0));
        assert_ne!(key(&tables(true, 1, 64, 16)), key(&tables(true, 4, 64, 16)));
        assert_ne!(key(&tables(true, 4, 64, 16)), key(&tables(true, 4, 64, 32)));
        // The page table stride reaches every step (the index expansion reads it).
        assert_ne!(key(&tables(false, 1, 64, 16)), key(&tables(false, 1, 128, 16)));
        assert_ne!(key(&tables(false, 1, 64, 16)), key(&tables(true, 1, 64, 16)));
        assert_ne!(GraphKey::new(3, 40, &tables(false, 1, 64, 16)), GraphKey::new(4, 40, &tables(false, 1, 64, 16)));
        assert_ne!(GraphKey::new(3, 40, &tables(false, 1, 64, 16)), GraphKey::new(3, 32, &tables(false, 1, 64, 16)));
    }

    #[test]
    fn decode_strides_are_powers_of_two_from_a_floor_within_the_pool_and_the_table() {
        let (pool, columns) = ((1 << 20, 1 << 18), (2048, 512));
        // The page stride starts at the floor; the pool stride (keyed only by long steps) has none.
        assert_eq!(decode_strides(5, 2, pool, columns), (MIN_PAGE_STRIDE, 2));
        assert_eq!(decode_strides(64, 16, pool, columns), (64, 16));
        assert_eq!(decode_strides(65, 17, pool, columns), (128, 32));
        assert_eq!(decode_strides(132, 33, pool, columns), (256, 64));
        // A row holds one sequence of the context: 131,072 tokens are 2,048 pages, 512 pool pages.
        assert_eq!(decode_strides(4000, 1000, pool, columns), (2048, 512));
        // A tiny pool, or a short context's table row, caps the strides below the floor.
        assert_eq!(decode_strides(5, 2, (32, 8), columns), (32, 2));
        assert_eq!(decode_strides(5, 2, pool, table(777)), (16, 2));
        // A long step's sequence holds more than the 2,051-token dense context: at least 9 units (36
        // pages), so its page stride is at least the floor already.
        let units = 2052usize.div_ceil(UNIT_ROWS);
        assert_eq!((units * UNIT_PAGES).next_power_of_two(), MIN_PAGE_STRIDE);
        assert_eq!(decode_strides(units * UNIT_PAGES, units, pool, columns), (64, 16));
    }

    /// A table row's (page, pool page) columns for `context` tokens.
    fn table(context: usize) -> (usize, usize) {
        let (pages, pools) = glmf_table_pages(context as u64);
        (pages as usize, pools as usize)
    }

    /// The startup geometries as `work/p0` enumerated them before the shorter keys: every key held
    /// the pool top-k's width and stride, and strides started at one unit.
    fn full_keys(context: usize, pages: usize, dense: usize) -> Vec<GraphGeometry> {
        let pools = pages / UNIT_PAGES;
        let mut geometries = Vec::new();
        for units in 1..=context.div_ceil(UNIT_ROWS).min(pools) {
            let pool_stride = units.next_power_of_two().min(pools);
            let page_stride = (units * UNIT_PAGES).next_power_of_two().min(pages);
            let capacity = (units * UNIT_ROWS).min(context);
            let mut width = 1;
            while width / 2 * UNIT_ROWS < capacity {
                let low = if width == 1 { 1 } else { width / 2 * UNIT_ROWS + 1 };
                let high = (width * UNIT_ROWS).min(capacity);
                for long in [false, true] {
                    if (!long && low <= high.min(dense)) || (long && low.max(dense + 1) <= high) {
                        let geometry = GraphGeometry { pool_width: width.min(pool_stride), page_stride, pool_stride, long };
                        if !geometries.contains(&geometry) { geometries.push(geometry); }
                    }
                }
                width *= 2;
            }
        }
        geometries
    }

    /// The startup set before and after, 16 sequences with drafts: long steps (the only ones that run
    /// the pool top-k) keep every key, short steps collapse onto page strides from 64 pages with no
    /// pool-table width or stride, every shape keeps the row buckets the threshold registry passes,
    /// and each capture's padded tables key it as its own shape.
    #[test]
    fn shorter_keys_shrink_the_startup_set_and_change_no_launch() {
        let dense = 2051;
        // The 64-row bucket sets (`--decode-rows 64`).
        let buckets = DecodeBuckets::new(DECODE_ROWS);
        // The reserve's pool target (2,097,152 tokens: 32,768 pages) at three contexts, then smaller
        // pools and short contexts, whose pool or table row caps the floors.
        for (context, pages, before_shapes, after_shapes) in [(8192usize, 32768usize, 230usize, 50usize),
            (32768, 32768, 400, 140), (131_072, 32768, 610, 270), (32768, 4096, 400, 140), (131_072, 2048, 610, 270),
            (4096, 64, 160, 20), (3000, 36, 160, 20), (2048, 32768, 100, 10)] {
            let (table_pages, _) = table(context);
            let (before, after) = (full_keys(context, pages, dense), graph_geometries(context, pages, dense));
            let long = |set: &[GraphGeometry]| set.iter().copied().filter(|g| g.long).collect::<Vec<_>>();
            assert_eq!(long(&after), long(&before), "{context}/{pages}: long keys moved");
            let short: HashSet<_> = after.iter().copied().filter(|g| !g.long).collect();
            let collapsed: HashSet<_> = before.iter().filter(|g| !g.long).map(|g| GraphGeometry { pool_width: 0,
                page_stride: g.page_stride.max(MIN_PAGE_STRIDE).min(pages).min(table_pages), pool_stride: 0,
                long: false }).collect();
            assert_eq!((short.len(), &short), (after.len() - long(&after).len(), &collapsed), "{context}/{pages}");
            let shapes = serving_graph_shapes(context, pages, dense, 16, true, &buckets);
            assert_eq!((before.len() * 10, shapes.len()), (before_shapes, after_shapes), "{context}/{pages}");
            // Rows: per geometry, exactly the buckets `check_decode_thresholds` passes for 16 sequences.
            check_decode_thresholds(&buckets, 16, true).unwrap();
            let plain: Vec<_> = PLAIN_DECODE_BUCKETS.into_iter()
                .filter(|&rows| rows <= decode_bucket(16usize.clamp(16, DECODE_ROWS), false)).collect();
            for geometry in &after {
                for (spec, buckets) in [(false, &plain[..]), (true, &SPEC_DECODE_BUCKETS[..])] {
                    let rows: Vec<_> = shapes.iter().filter(|s| s.2 == *geometry && s.1 == spec).map(|s| s.0).collect();
                    assert_eq!(rows, buckets);
                    check_bucket_thresholds(&rows, DECODE_PROJECTION_THRESHOLDS).unwrap();
                }
            }
            // The capture's masked tables: `rows` rows of the geometry's strides, keyed as the shape.
            for &(rows, spec, geometry) in &shapes {
                let tables = startup_tables(rows, spec, geometry);
                assert_eq!((tables.kv_slots.len(), tables.page_table.len(), tables.pool_table.len(), tables.real_rows),
                    (rows, rows * geometry.page_stride, rows * geometry.pool_stride, 0));
                assert!(tables.positions.iter().all(|&p| p == -1) && tables.kv_slots.iter().all(|&s| s == -1));
                for segment in [0, 45] {
                    assert_eq!(GraphKey::new(segment, rows, &tables), GraphKey { segment, rows, spec,
                        long: geometry.long, pool_width: geometry.pool_width, page_stride: geometry.page_stride,
                        pool_stride: geometry.pool_stride });
                }
            }
        }
        // The startup reserve on one GPU (46 segments a shape) at the engine's 146,459 B a graph.
        for (context, before, after) in [(8192, 1_926_552_328, 471_335_704), (32768, 3_300_923_584, 1_198_944_016),
            (131_072, 4_998_676_312u64, 2_249_933_800u64)] {
            assert_eq!(graph_reserve_bytes(full_keys(context, 32768, dense).len() * 10 * 46), before);
            assert_eq!(serving_graph_reserve(context, 2_097_152, dense, 16, true, 45, false, &buckets), [after]);
        }
    }

    /// A short step's key leaves out the pool table's width and stride, so the pool top-k, which runs
    /// only when a row is long, must stay the one launch that reads them. Re-derive the key (and
    /// this check) if another launch reads the pool table or those scalars.
    #[test]
    fn only_the_long_pool_top_k_reads_what_short_keys_leave_out() {
        let source = include_str!("engine.rs");
        let source = &source[..source.find("\n#[cfg(test)]\n").unwrap()];
        let reads = ["pool_table.buffer", "Scalar::I32(tables.pool_width", "Scalar::I32(tables.pool_stride"];
        let topk = source.find("&format!(\"index_topk_").unwrap();
        let guard = source[..topk].rfind("if tables.long {").unwrap();
        let end = topk + source[topk..].find("\"index_expand\"").unwrap();
        // `if tables.long {` opens right before the launch and closes after its scalars.
        assert!(!source[guard..topk].contains('}'));
        for read in reads {
            assert_eq!(source.matches(read).count(), 1, "{read}");
            let at = source.find(read).unwrap();
            assert!(topk < at && at < end, "{read} outside the long pool top-k launch");
        }
        let last = reads.iter().map(|read| source.find(read).unwrap()).max().unwrap();
        assert!(source[last..end].lines().any(|line| line.trim() == "}"));
    }
}

#[cfg(test)]
mod replay_record_tests {
    use super::{replay_bytes, RecordGuard, ReplayRecords, DECODE_ROWS, WIDE_DECODE_ROWS};

    #[test]
    fn shared_records_commit_only_what_no_prefill_overwrote() {
        let guard = RecordGuard::new(ReplayRecords::Shared);
        // The serving loop: prefill rounds, then a verify and its commit.
        guard.prefilled();
        guard.recorded();
        guard.check_commit().unwrap();
        // A second commit of the same records (nothing ran in between) reads them intact.
        guard.check_commit().unwrap();
        // A prefill between a verify and its commit overwrote the records.
        guard.recorded();
        guard.prefilled();
        let error = guard.check_commit().unwrap_err().to_string();
        assert!(error.contains("between a speculative verify and its commit"), "{error}");
        // The next verify records afresh.
        guard.recorded();
        guard.check_commit().unwrap();
        // A commit with no verify at all reads nothing it could trust.
        assert!(RecordGuard::new(ReplayRecords::Shared).check_commit().is_err());
    }

    #[test]
    fn records_of_their_own_are_never_refused() {
        let guard = RecordGuard::new(ReplayRecords::Own);
        guard.recorded();
        guard.prefilled();
        guard.check_commit().unwrap();
        assert!(RecordGuard::new(ReplayRecords::Own).check_commit().is_ok());
    }

    /// One KDA layer's records over a GPU's 64 KDA heads: 9,453,568 B; 34 layers 321,421,312 B,
    /// the loader's `glm_flash_kda_replay_bytes`, which the 4,096-row KDA prefill programs'
    /// 782,236,672-byte scratch holds. With `--decode-rows 128` they hold 128 rows: 642,842,624 B
    /// (`glm_flash_kda_replay_bytes_rows`), which the same scratch still holds.
    #[test]
    fn the_records_fit_the_prefill_scratch() {
        assert_eq!(replay_bytes(64, 3 * 64 * 128, DECODE_ROWS), 9_453_568);
        assert_eq!(34 * replay_bytes(64, 3 * 64 * 128, DECODE_ROWS), 321_421_312);
        assert!(34 * replay_bytes(64, 3 * 64 * 128, DECODE_ROWS) < 782_236_672);
        assert_eq!(34 * replay_bytes(64, 3 * 64 * 128, WIDE_DECODE_ROWS), 642_842_624);
        assert!(34 * replay_bytes(64, 3 * 64 * 128, WIDE_DECODE_ROWS) < 782_236_672);
    }
}

#[cfg(test)]
mod spark_warmup_tests {
    use super::{prefill_lane_capacity, prefill_lane_plan, prefill_workspace_count, spark_request, spark_warmup_request,
        DECODE_ROWS, MAX_PREFILL_LANES};
    use cuteafd_loader::families::glm5_flash::{GlmNextAttention, GlmNextConfig};
    use cuteafd_transport::expert::EXPERT_PROTOCOL_V2_FLAG_V41_COMPACT_BF16;
    use cuteafd_transport::{ExpertProtocolV2Request, ExpertProtocolV2RouteEntry, ExpertV2Dtype, ExpertV2SourceKind,
        EXPERT_PROTOCOL_V2_RESPONSE_HEADER_LEN};
    use std::collections::BTreeSet;

    /// GLM 5.3 Flash: 45 layers (0-2 dense), 288 routed experts, top-8, hidden 4096.
    fn glm53_flash() -> GlmNextConfig {
        GlmNextConfig { vocab_size: 154880, hidden: 4096, layers: 45,
            attention: (0..45).map(|i| if i % 4 == 3 { GlmNextAttention::Mla } else { GlmNextAttention::Kda }).collect(),
            dense: (0..45).map(|i| i < 3).collect(), dense_intermediate: 12288, experts: 288, topk: 8,
            moe_intermediate: 2048, routed_scale: 2.5, swiglu_limit: 10.0, rms_norm_eps: 1e-5, hc_mult: 4,
            kda_heads: 64, kda_head_dim: 128, heads: 64, q_lora_rank: 1536, kv_lora_rank: 512, qk_nope_dim: 256,
            v_head_dim: 256, index_topk: 2048, index_kpool: 4, eos: vec![] }
    }

    /// The answer bytes a rank's session reserves for `request` (the transport's
    /// `verbs_host_expected_response_wire_bytes` without a reduced-precision answer): the header,
    /// then a row index and a partial row at least BF16-wide per row.
    fn response_bytes(request: &ExpertProtocolV2Request) -> usize {
        let header = &request.header;
        EXPERT_PROTOCOL_V2_RESPONSE_HEADER_LEN + header.row_count as usize
            * (4 + (header.hidden_row_stride_bytes as usize).max(2 * header.hidden_dim as usize))
    }

    /// A serving wave of `t` rows at the first MoE layer, as `spark_dispatch` builds one.
    fn wave(cfg: &GlmNextConfig, t: usize, kind: ExpertV2SourceKind) -> ExpertProtocolV2Request {
        let routes = (0..t * cfg.topk).map(|i| ExpertProtocolV2RouteEntry { row_index: (i / cfg.topk) as u32,
            expert_id: (7 * i % cfg.experts) as u32, gate_weight: 0.125 }).collect();
        spark_request(cfg.hidden, cfg.topk, 3, t, routes, vec![0x38; t * (cfg.hidden + cfg.hidden / 32)], kind).unwrap()
    }

    /// The warm-up is a full lane of prefill in the serving format: 4,096 zero rows at the first MoE
    /// layer (3), every gate zero, all 288 experts routed, eight distinct a row. Each rank's session
    /// then keeps 17,858,656 request and 33,570,912 answer bytes a slot: at the transport's default
    /// eight 4 KiB-aligned slots, 411,500,544 B of rings, what the coordinator's `transport/rdma-rings`
    /// ledger held per session once a 4,096-row lane had opened it (3,292,004,352 B for two lanes of
    /// four ranks).
    #[test]
    fn the_warmup_is_a_full_lane_of_zero_gates_in_the_serving_format() {
        let cfg = glm53_flash();
        let warm = spark_warmup_request(&cfg, 4096).unwrap();
        let serving = wave(&cfg, 4096, ExpertV2SourceKind::Prefill);
        assert_eq!((&warm.header, &warm.rows), (&serving.header, &serving.rows));
        assert_eq!((warm.header.layer_id, warm.header.row_count, warm.header.hidden_dtype, warm.header.flags),
            (3, 4096, ExpertV2Dtype::Fp8E4m3Ue8m0K32, EXPERT_PROTOCOL_V2_FLAG_V41_COMPACT_BF16));
        assert!(warm.rows.iter().all(|row| row.source_kind == ExpertV2SourceKind::Prefill));
        assert!(warm.routes.iter().all(|route| route.gate_weight == 0.0));
        assert!(warm.routes.chunks(8).enumerate().all(|(row, routes)| routes.iter().all(|r| r.row_index == row as u32)
            && routes.iter().map(|r| r.expert_id).collect::<BTreeSet<_>>().len() == 8));
        assert_eq!(warm.routes.iter().map(|r| r.expert_id).collect::<BTreeSet<_>>(), (0..288).collect());
        assert!(warm.hidden_payload.len() == 4096 * (4096 + 128) && warm.hidden_payload.iter().all(|&b| b == 0));
        let (request, answer) = (warm.wire_stats().wire_bytes, response_bytes(&warm));
        assert_eq!((request, answer), (17_858_656, 33_570_912));
        assert_eq!(8 * (request.next_multiple_of(4096) + answer.next_multiple_of(4096)), 411_500_544);
    }

    /// No wave of a warmed transport is larger than its warm-up: decode and verify steps and prefill
    /// lanes or serial chunks of up to the warmed rows fit the slots it opened, so `post` never drops
    /// and reconnects a warmed rank while serving.
    #[test]
    fn no_wave_outgrows_a_warmed_session() {
        let cfg = glm53_flash();
        for rows in [2048, 4096] {
            let warm = spark_warmup_request(&cfg, rows).unwrap();
            let (request, answer) = (warm.wire_stats().wire_bytes, response_bytes(&warm));
            for t in [1, 2, DECODE_ROWS - 1, DECODE_ROWS, 128, 1023, 1024, 2048, 2072, 2109, 2176, 4095, 4096]
                .into_iter().filter(|&t| t <= rows) {
                for kind in [ExpertV2SourceKind::Decode, ExpertV2SourceKind::Prefill] {
                    let serving = wave(&cfg, t, kind);
                    assert!(serving.wire_stats().wire_bytes <= request && response_bytes(&serving) <= answer,
                        "{t} rows of {kind:?} after a {rows}-row warm-up");
                }
            }
            let largest = wave(&cfg, rows, ExpertV2SourceKind::Prefill);
            assert_eq!((largest.wire_stats().wire_bytes, response_bytes(&largest)), (request, answer));
        }
    }

    /// Start-up warms each transport a prefill runs on (`configured_prefill_lanes`, which agrees with
    /// `GlmfEngine::pipelined`): every lane's while lanes run (1 to 4), else transport 0, which also
    /// carries decode, verify and serial chunks. No chunk the engine takes uses another transport or
    /// puts more rows on one than the warm-up sent.
    #[test]
    fn the_warmup_covers_every_transport_a_prefill_uses() {
        let rows = 4096;
        for lanes in 1..=MAX_PREFILL_LANES {
            for (setting, complete) in [(None, true), (Some("1"), true), (None, false), (Some("subset"), false),
                (Some("subset"), true)] {
                let warmed = prefill_workspace_count(true, complete, setting, lanes);
                // `lane_setting`: lanes run unless set to 1, over every layer or a subset with `subset`.
                let pipelined = setting != Some("1") && (complete || setting == Some("subset"));
                assert_eq!(warmed, if pipelined { lanes } else { 1 }, "{lanes} lanes, {setting:?}, {complete}");
                let capacity = if pipelined { prefill_lane_capacity(lanes, rows) } else { rows };
                for tokens in (1..=capacity).step_by(61).chain([capacity]) {
                    let (used, per_lane) = if pipelined { prefill_lane_plan(tokens, lanes, rows).unwrap() }
                        else { (1, tokens) };
                    assert!(used <= warmed && per_lane <= rows, "{tokens} tokens, {lanes} lanes, {setting:?}");
                }
            }
        }
    }
}

#[cfg(test)]
mod packed_step_tests {
    use super::{packed_tail, packing, GlmfPlacement, Span, StepTables};
    use anyhow::{anyhow, ensure, Result};
    use std::collections::HashMap;

    /// The device as a packed pass leaves it: each KDA slot's consumed rows (its KDA state and
    /// its MLA/index pages), written by the step for every sequence before any callback.
    #[derive(Default)]
    struct Device {
        rows: HashMap<i32, Vec<u32>>,
    }

    impl Device {
        fn step(&mut self, sequences: &[(&mut GlmfPlacement, &[u32])]) {
            for (placement, tokens) in sequences {
                self.rows.entry(placement.slot).or_default().extend_from_slice(tokens);
            }
        }

        /// A decode step: it writes at the placement's length, which must be what the device holds.
        fn decode(&mut self, placement: &mut GlmfPlacement, token: u32) -> Result<()> {
            let rows = self.rows.entry(placement.slot).or_default();
            ensure!(rows.len() == placement.len && placement.kda_len == placement.len,
                "decode at {} (KDA {}) over {} rows", placement.len, placement.kda_len, rows.len());
            rows.push(token);
            placement.len += 1;
            placement.kda_len = placement.len;
            Ok(())
        }
    }

    /// Three prompts in one packed pass (the second resumed at 64 rows), the second of which fails
    /// on its own (its grammar allows no token).
    fn pass(device: &mut Device, head: impl FnMut(usize) -> Result<usize>)
        -> (Vec<GlmfPlacement>, Result<Vec<Result<()>>>, Vec<usize>) {
        let chunks: [Vec<u32>; 3] = [(0..40).collect(), (100..110).collect(), vec![7]];
        let mut placements: Vec<GlmfPlacement> = (0..3).map(|i| GlmfPlacement::new(vec![i], i as i32)).collect();
        device.rows.insert(1, (0..64).collect());
        placements[1].len = 64;
        placements[1].kda_len = 64;
        let mut called = Vec::new();
        let outcomes = {
            let mut sequences: Vec<(&mut GlmfPlacement, &[u32])> = placements.iter_mut().zip(&chunks)
                .map(|(p, c)| (p, c.as_slice())).collect();
            device.step(&sequences);
            packed_tail(&mut sequences, head, &mut |i, row| {
                assert_eq!(row, i, "each sequence gets its own head row");
                called.push(i);
                Ok(if i == 1 { Err(anyhow!("grammar allows no target token")) } else { Ok(()) })
            })
        };
        (placements, outcomes, called)
    }

    #[test]
    fn a_member_that_fails_alone_leaves_its_neighbours_to_finish_and_decode() {
        let mut device = Device::default();
        let (mut placements, outcomes, called) = pass(&mut device, Ok);
        // Every callback ran; the failure is the second member's alone, reported afterwards.
        assert_eq!(called, [0, 1, 2]);
        let outcomes = outcomes.unwrap();
        assert_eq!(outcomes.iter().map(Result::is_ok).collect::<Vec<_>>(), [true, false, true]);
        assert_eq!(outcomes[1].as_ref().unwrap_err().to_string(), "grammar allows no target token");
        // Every placement, the failed one's too, matches what the step wrote.
        for p in &placements {
            assert_eq!((p.len, p.kda_len), (device.rows[&p.slot].len(), device.rows[&p.slot].len()), "slot {}", p.slot);
        }
        assert_eq!(placements.iter().map(|p| p.len).collect::<Vec<_>>(), [40, 74, 1]);
        // The healthy members decode on top of their own rows, and only those.
        for token in [900, 901] {
            device.decode(&mut placements[0], token).unwrap();
            device.decode(&mut placements[2], token + 10).unwrap();
        }
        assert_eq!(device.rows[&0], (0..40).chain([900, 901]).collect::<Vec<_>>());
        assert_eq!(device.rows[&2], [7, 910, 911]);
        // The failed member's caller releases it: a fresh placement over a reset slot starts clean.
        device.rows.remove(&1);
        let mut fresh = GlmfPlacement::new(vec![1], 1);
        device.decode(&mut fresh, 5).unwrap();
    }

    #[test]
    fn a_shared_failure_ends_the_pass_with_every_placement_where_the_step_left_it() {
        let mut device = Device::default();
        let (placements, outcomes, called) = pass(&mut device,
            |i| if i == 2 { Err(anyhow!("head failed")) } else { Ok(i) });
        // The head's failure is the pass's: it ends it (every member fails and is released).
        assert_eq!(outcomes.unwrap_err().to_string(), "head failed");
        assert_eq!(called, [0, 1]);
        for p in &placements {
            assert_eq!((p.len, p.kda_len), (device.rows[&p.slot].len(), device.rows[&p.slot].len()), "slot {}", p.slot);
        }
    }

    #[test]
    fn a_step_of_one_sequence_runs_every_per_sequence_program_once_over_all_rows() {
        // Decode, verify, serial and lane prefill steps carry no segments: one span of every row,
        // so each per-sequence launch gets the whole step's rows from row 0, as before packing.
        for rows in [1, 7, 64, 127, 4096] {
            let tables = StepTables { kv_slots: vec![0; rows], ..Default::default() };
            assert_eq!(tables.spans(), [Span { first: 0, rows }]);
        }
    }

    #[test]
    fn a_packed_step_runs_them_once_per_sequence_over_its_own_rows() {
        let limits = packing::Limits { rows: 4096, chunk_rows: 511, table_pages: 2048, table_pool_pages: 512,
            taps: Some(2048), dense_context: 2051, max_context: 131_072 };
        let segments = packing::plan(&[(0, 40), (2100, 100), (0, 1)], &limits).unwrap();
        let tables = StepTables { kv_slots: vec![0; 141], segments, ..Default::default() };
        assert_eq!(tables.spans(), [Span { first: 0, rows: 40 }, Span { first: 40, rows: 100 },
            Span { first: 140, rows: 1 }]);
    }
}
