//! Qwen 3.8 Flash Next (qwen4_exp) coordinator over the exported qwen4_* programs.
//!
//! Four hyper-connection streams (BF16 `[rows, 4, H]`) run through every
//! layer: the attention site's input (`hc_pre`, or the previous layer's fused
//! `hc_post_pre`), the attention sublayer, `hc_post_pre` into the streams and
//! onto the MLP site, the MoE, and the next fused post/pre (`hc_post` after
//! the last layer, then the stream mixer `head` and lm_head). PLE (layer 1)
//! adds its n-gram features to the streams before that layer's attention site.
//!
//! Attention: Gated DeltaNet layers keep per-sequence FP32 recurrent state
//! and short-conv state (the last three q/k/v inputs) in slot pools, one slot
//! per sequence shared by every GDN layer (and the PLE conv state); full
//! attention layers keep BF16 K/V records in 64-row pages, per-token raw
//! index keys beside them, and one pooled index key per completed 4-token
//! block in pool pages (64 blocks per page). Up to 2051 visible tokens every
//! token is attended; past that the top 512 blocks (qwen4_index_topk) expand
//! to tokens plus the open tail block.
//!
//! MoE: FP32 router logits, the native softmax top-10 (logits and weights
//! rounded to BF16 as the reference), the shared expert with its sigmoid
//! gate, and routed experts on this GPU (FP8/NVFP4 packages fully resident
//! by default, EXL3 packages with their admitted resident window) or on the Sparks.
use super::weights::{Qwen4Head, Qwen4Layer, Qwen4Weights};
use crate::shared::experts::fp8::Fp8Experts;
use crate::shared::experts::rtx::owner_tp2::OwnerTp2;
use crate::shared::launch_grid::Fp8QuantizeGrid;
use crate::shared::memory::{HostAllocation, Tag};
use crate::shared::qwen::{GdnBank, LayerStateMap, REPLAY_ROWS};
use crate::shared::memory::device::{Allocation, Device, DeviceOwner};
use std::rc::Rc;
use crate::shared::token_io::{DeviceLogits, TokenEmbedding};
use anyhow::{ensure, Context, Result};
use cuteafd_ffi::programs::{Programs, Scalar, VocabularyHead, VOCABULARY_HEAD_WORKSPACE};
use cuteafd_ffi::NativeLibrary;
use cuteafd_loader::formats::fp8_experts::Fp8ExpertTensors;
use cuteafd_loader::families::qwen4::{NgramHistory, Qwen4Attention, Qwen4Config};
use crate::shared::spark_intake::SparkLink;
use cuteafd_transport::expert::EXPERT_PROTOCOL_V2_FLAG_V41_COMPACT_BF16;
use cuteafd_transport::{
    ExpertProtocolV2Request, ExpertProtocolV2RouteEntry, ExpertProtocolV2RowDescriptor, ExpertV2Dtype, ExpertV2SourceKind,
};
use std::cell::{Cell, RefCell};
use crate::shared::decode_graph::GraphBank;
use std::ffi::c_void;
use std::sync::Arc;

type Dev<'a> = Rc<Allocation<'a>>;

pub(crate) const PAGE_ROWS: usize = 64;
/// Views are 256-byte aligned; every record-set part is `PAGE_ROWS` rows of a
/// 256-byte multiple (`record bytes` is 1040 or 2048), so the offsets hold.
const INDEX_DIM_BYTES: usize = INDEX_DIM * 2;

/// One pooled record set's physical request: the KV records, then the token
/// index keys and the pool index keys at aligned offsets. One arena per record
/// set, so admission prices a single cudaMalloc rounding for the set instead of
/// one per part (`qwen4.rs` `pooled_backing`).
fn record_arena_bytes(pages: usize, pool_pages: usize, record_bytes: usize) -> Result<(usize, usize, usize)> {
    let bytes = |dim: usize, stride: usize| dim.checked_mul(PAGE_ROWS).and_then(|rows| rows.checked_mul(stride))
        .ok_or_else(|| anyhow::anyhow!("Qwen record arena overflow"));
    Ok((bytes(pages, record_bytes)?, bytes(pages, INDEX_DIM_BYTES)?, bytes(pool_pages, INDEX_DIM_BYTES)?))
}

/// Rows of the decode-shaped programs (`_m64`).
pub(crate) const DECODE_ROWS: usize = cuteafd_loader::families::qwen4::VERIFY_ROWS;
use cuteafd_loader::serving_capacity::qwen_graphs::{QwenGraphShapeKey, QWEN_MIN_PAGE_STRIDE};
/// FP8 decode projection and head span capacity (matches fork `FP8_DECODE_ROWS`).
const FP8_PROJECTION_SKINNY_ROWS: usize = 64;

/// Serving graphs are on unless `CUTEAFD_QWEN4_GRAPHS=0`.
pub(super) fn graphs_enabled(graphs: Option<&str>) -> bool {
    graphs != Some("0")
}

/// Selected-slot row width of the sparse attention (2048 + 3, padded to 64).
pub(crate) const SPARSE_TOPK: usize = 2112;
/// BF16 index-key width, independent of the K/V record format.
pub(crate) const INDEX_DIM: usize = 128;
const MAX_RANKS: usize = crate::shared::spark_intake::MAX_INTAKE_RANKS;
const HC: usize = 4;
/// Tokens per QSA index block, and blocks per pool-cache page.
pub(crate) const BLOCK: usize = 4;
const POOL_PAGE_TOKENS: usize = BLOCK * PAGE_ROWS;
/// Record pages per allocation unit: a unit is four consecutive 64-row record pages (256 tokens)
/// and the pool-cache page of the same index (64 blocks of 4 tokens), so one refcounted unit
/// index names every paged byte of 256 positions (the prefix cache's page).
pub(crate) const UNIT_PAGES: usize = BLOCK;
pub(crate) const UNIT_ROWS: usize = POOL_PAGE_TOKENS;
/// Rows of the PLE conv state ((taps - 1) x dilation).
const PLE_STATE_ROWS: usize = 9;
/// Most rows one launch of the E4M3 head (`qwen4_head_fp8`) takes; wider
/// logits calls run it in spans of this many rows.
pub(crate) const FP8_HEAD_ROWS: usize = 64;

/// `(first row, rows)` spans of at most [`FP8_HEAD_ROWS`] covering `rows` logits rows.
pub(crate) fn fp8_head_spans(rows: usize) -> impl Iterator<Item = (usize, usize)> {
    (0..rows).step_by(FP8_HEAD_ROWS).map(move |first| (first, FP8_HEAD_ROWS.min(rows - first)))
}

/// Widest checkpoint-native wire geometry, with no expert layer or dummy rows.
pub(crate) fn spark_session_shape(cfg: &Qwen4Config, rows: usize, bf16: bool)
    -> Result<cuteafd_transport::SparkSessionShape> {
    let dtype = if bf16 { ExpertV2Dtype::Bf16 } else { ExpertV2Dtype::Fp8E4m3Ue8m0K32 };
    let mut shape = cuteafd_transport::SparkSessionShape::new(rows as u32, cfg.hidden as u32, cfg.topk as u32, dtype)?;
    shape.flags |= EXPERT_PROTOCOL_V2_FLAG_V41_COMPACT_BF16;
    Ok(shape)
}

/// Target rows a sequence may hold for its MTP canonical history before the
/// next draft step (the MTP stash, per state slot).
pub(crate) const MTP_PENDING_ROWS: usize = 64;

fn remote_front_plane_bytes(rows: usize, hidden: usize, topk: usize, bf16: bool) -> [usize; 4] {
    let input = if bf16 { hidden * 2 } else { hidden + hidden / 32 };
    [rows * input, rows * topk * 4, rows * topk * 4, rows * hidden * 2]
        .map(|bytes| bytes.next_multiple_of(16))
}

/// The MTP layer stays resident locally even when backbone experts run on Sparks.
pub(crate) enum MtpExperts<'a> {
    Whole(crate::shared::experts::rtx::pair::Whole<'a>),
    Fp8(Fp8Experts<'a>),
    Exl3(RefCell<crate::families::deepseek_v4::local::LocalExperts<'a>>),
}

/// Resident routed layers, independent of any Spark fallback.
pub(crate) enum LocalRouted<'a> {
    Whole(crate::shared::experts::rtx::pair::Whole<'a>),
    Tp2(RefCell<OwnerTp2<'a>>),
}
impl LocalRouted<'_> {
    pub(crate) fn layers(&self) -> Vec<usize> {
        match self { Self::Whole(whole) => whole.layers.clone(),
            Self::Tp2(pair) => pair.borrow().experts[0].layers().to_vec() }
    }
    fn wire(&self) -> bool {
        match self { Self::Whole(whole) => whole.uses_fp8_input(), Self::Tp2(pair) => pair.borrow().wire }
    }
    fn group_rows(&self, rows: usize) -> Result<usize> {
        match self {
            Self::Whole(whole) => whole.group_rows(rows),
            Self::Tp2(pair) => {
                let pair = pair.borrow();
                let group = pair.experts[0].group_rows(rows)?;
                ensure!(pair.experts[1].group_rows(rows)? == group, "Qwen TP2 grouping differs");
                Ok(group)
            }
        }
    }
    fn weight_read_bytes(&self) -> Result<usize> {
        match self { Self::Whole(whole) => whole.weight_read_bytes(),
            Self::Tp2(pair) => pair.borrow().experts.iter().map(|rank| rank.weight_read_bytes())
                .collect::<Result<Vec<_>>>()?.into_iter().max().context("Qwen TP2 ranks missing") }
    }
}

/// All-local and all-Spark are endpoints of the same admitted hybrid.
pub(crate) struct HybridExperts<'a> {
    pub local: Option<LocalRouted<'a>>,
    pub local_layers: Vec<usize>,
    pub transport: Vec<RefCell<SparkLink<'a>>>,
    pub mtp: Option<MtpExperts<'a>>,
    /// A zero-local dual still needs abort words for its attention hops.
    pub terminal: Option<crate::shared::peer_split::PeerExchange<'a>>,
}
impl HybridExperts<'_> {
    pub(crate) fn local_layer(&self, layer: usize) -> bool { self.local_layers.binary_search(&layer).is_ok() }
    fn with_peer<T>(&self, body: impl FnOnce(&crate::shared::peer_split::PeerExchange<'_>) -> Result<T>) -> Result<Option<T>> {
        match &self.local {
            Some(LocalRouted::Tp2(pair)) => body(&pair.borrow().exchange).map(Some),
            _ => self.terminal.as_ref().map(body).transpose(),
        }
    }
}

pub(crate) enum Experts<'a> {
    Hybrid(HybridExperts<'a>),
    /// Plumbing diagnostics, never a production expert source.
    SharedOnly,
}

/// Appends one row's page list padded with zeros (or cut) to `stride` entries,
/// without a per-row allocation.
fn extend_padded(table: &mut Vec<i32>, pages: &[i32], stride: usize) {
    let kept = pages.len().min(stride);
    table.extend_from_slice(&pages[..kept]);
    table.resize(table.len() + stride - kept, 0);
}

/// Host tables of one step.
#[derive(Default, Clone)]
struct StepTables {
    decode: bool,
    /// A speculative verify: GDN and PLE record replay inputs instead of
    /// advancing their state (`commit` applies the accepted rows).
    spec: bool,
    positions: Vec<i64>,
    rope_positions: Vec<[i32; 3]>,
    block_rope_positions: Vec<[i32; 3]>,
    /// K/V record slot per row (also the row's raw index-key slot).
    kv_slots: Vec<i64>,
    /// GDN / PLE state slot per row.
    slots: Vec<i32>,
    /// First step row of each row's sequence.
    seq_first: Vec<i32>,
    /// Pool index-cache slot of the block a row completes, else -1.
    pool_slots: Vec<i64>,
    /// Complete blocks each row sees.
    cache_lengths: Vec<i32>,
    /// Record pages and pool pages: one shared table (prefill) or one padded row per step row.
    page_table: Vec<i32>,
    pool_table: Vec<i32>,
    /// Row stride of the page tables (0: every row reads one shared table).
    page_stride: usize,
    pool_stride: usize,
    /// Pages of the shared table (prefill) or of each padded row (decode).
    page_width: usize,
    pool_width: usize,
    /// Whether any row sees more than 2051 tokens (the block top-k runs).
    long: bool,
    /// PLE table rows [rows, 16].
    ple_ids: Vec<i64>,
}

impl crate::shared::mapped_table::RowSource for StepTables {
    fn units(&self) -> usize { 1 }
    fn rows(&self, unit: usize) -> &[i64] { assert_eq!(unit, 0); &self.ple_ids }
}
impl crate::shared::mapped_table::RowSource for Vec<StepTables> {
    fn units(&self) -> usize { self.len() }
    fn rows(&self, unit: usize) -> &[i64] { &self[unit].ple_ids }
}

impl StepTables {
    fn unit(&self, rows: std::ops::Range<usize>, ple_rows: usize, dense_context: usize) -> Self {
        let n = rows.len();
        let positions = self.positions[rows.clone()].to_vec();
        Self { long: positions.iter().any(|&p| p as usize + 1 > dense_context), positions,
            rope_positions: self.rope_positions[rows.clone()].to_vec(),
            block_rope_positions: self.block_rope_positions[rows.clone()].to_vec(),
            kv_slots: self.kv_slots[rows.clone()].to_vec(), slots: self.slots[rows.clone()].to_vec(),
            seq_first: vec![0; n], pool_slots: self.pool_slots[rows.clone()].to_vec(),
            cache_lengths: self.cache_lengths[rows.clone()].to_vec(),
            page_table: self.page_table.clone(), pool_table: self.pool_table.clone(),
            page_width: self.page_width, pool_width: self.pool_width,
            ple_ids: self.ple_ids[rows.start * ple_rows..rows.end * ple_rows].to_vec(),
            ..Default::default() }
    }
}

/// A sequence's allocation units (and the record and pool pages they expand to), its GDN/PLE
/// state slot, its length, the rows its state holds and its n-gram history.
#[derive(Debug, Clone)]
pub(crate) struct Qwen4Placement {
    pub units: Vec<u32>,
    pub pages: Vec<i32>,
    pub pool_pages: Vec<i32>,
    pub slot: i32,
    pub len: usize,
    /// Rows the GDN recurrent/conv and PLE conv state has consumed: `len` after a prefill or a
    /// plain verify; a speculative verify leaves it until the kept rows are committed and the
    /// placement rewound ([`Qwen4Engine::rewind`]).
    pub state_len: usize,
    /// Contiguous canonical MTP rows; proposals and pending hidden rows do not count.
    pub mtp_frontier: std::cell::Cell<usize>,
    pub mtp_snapshot_pending: std::cell::Cell<bool>,
    pub history: NgramHistory,
    /// Request metadata from native ids/span grids, recomputed on prefix restore.
    /// Never radix keys; rotary coordinates never change logical cache rows.
    pub rope: cuteafd_loader::families::qwen4::RopePositions,
    pub media: Option<cuteafd_engine::media::RequestMedia>,
}

impl Qwen4Placement {
    /// A fresh sequence over `units` with state slot `slot` and n-gram history `history`.
    pub fn new(units: Vec<u32>, slot: i32, history: NgramHistory) -> Self {
        let pages = units.iter().flat_map(|&u| (0..UNIT_PAGES as i32).map(move |i| u as i32 * UNIT_PAGES as i32 + i))
            .collect();
        let pool_pages = units.iter().map(|&u| u as i32).collect();
        Self { units, pages, pool_pages, slot, len: 0, state_len: 0, mtp_frontier: Default::default(), mtp_snapshot_pending: Default::default(), history, rope: Default::default(), media: None }
    }

    /// Appends freshly reserved units (a decode top-up) with their record and pool pages.
    pub fn extend(&mut self, units: &[u32]) {
        self.pages.extend(units.iter().flat_map(|&u| (0..UNIT_PAGES as i32).map(move |i| u as i32 * UNIT_PAGES as i32 + i)));
        self.pool_pages.extend(units.iter().map(|&u| u as i32));
        self.units.extend_from_slice(units);
    }

    pub(super) fn packed_table_entries(&self) -> [usize; 2] {
        [self.pages.len(), self.pool_pages.len()]
    }

    pub fn record(&self, position: usize) -> Result<i64> {
        let page = *self.pages.get(position / PAGE_ROWS).context("position past the sequence's pages")?;
        Ok(i64::from(page) * PAGE_ROWS as i64 + (position % PAGE_ROWS) as i64)
    }

    /// Pool-cache slot of the block `position` completes, or -1.
    pub fn pool_slot(&self, position: usize) -> Result<i64> {
        if position % BLOCK != BLOCK - 1 {
            return Ok(-1);
        }
        let page = *self.pool_pages.get(position / POOL_PAGE_TOKENS).context("position past the pool pages")?;
        Ok(i64::from(page) * PAGE_ROWS as i64 + ((position / BLOCK) % PAGE_ROWS) as i64)
    }
}

/// Pack media rows in the same order as native token gathers (MTP uses p+1).
fn embedding_media(groups: &[(&Qwen4Placement, usize, usize)]) -> Result<cuteafd_engine::media::MediaChunk> {
    let mut packed = cuteafd_engine::media::MediaChunk::default();
    let mut offset = 0usize;
    for &(placement, start, rows) in groups {
        if let Some(media) = placement.media.as_ref().filter(|media| media.needed(start, start + rows).next().is_some()) {
            let mut chunk = cuteafd_engine::media::MediaChunk::default();
            media.write_chunk(start, start + rows, &mut chunk)?;
            packed.indices.extend(chunk.indices.iter().map(|&index| index + offset as u32));
            packed.features.extend(chunk.features);
        }
        offset += rows;
    }
    Ok(packed)
}

fn mtp_embedding_media(groups: &[MtpGroup<'_>]) -> Result<cuteafd_engine::media::MediaChunk> {
    let rows: Vec<_> = groups.iter().flat_map(|group| group.rows.iter()
        .map(move |row| (group.placement, row.position + 1, 1))).collect();
    embedding_media(&rows)
}

/// The n-gram history after `tokens` (the PLE hash's context is a pure function of the token ids:
/// their last `ngram_size - 1`, EOS before the first).
pub(crate) fn history_of(cfg: &Qwen4Config, tokens: &[u32]) -> NgramHistory {
    ngram_history(cfg.eos, cfg.ngram_size, tokens)
}

fn ngram_history(eos: u32, ngram_size: usize, tokens: &[u32]) -> NgramHistory {
    let context = ngram_size - 1;
    let mut history = vec![eos; context];
    history.extend_from_slice(&tokens[tokens.len().saturating_sub(context)..]);
    NgramHistory(history.split_off(history.len() - context))
}

/// Actual future host owners against the admitted recipe owners (see
/// `Qwen4Engine::admit_host_memory`); `units` is the engine's pool pages.
pub(super) fn validate_admitted_host(host: &cuteafd_loader::placement::families::qwen4::QwenHost,
    tables: &[(String, u64)], prefix_bytes: u64, units: u64) -> Result<()> {
    let mut logits = 0u64;
    for (name, bytes) in tables {
        let parts: Vec<_> = name.split('/').collect();
        let bound = match parts.as_slice() {
            ["qwen4", "owner-0", "mtp-prefill", "pinned-logits"] => host.mtp_pinned_logits,
            ["qwen4", _, _, "pinned-logits"] => {
                logits = logits.checked_add(*bytes).context("Qwen pinned logits overflow")?;
                continue;
            }
            ["qwen4", owner, kind, "staging"] => {
                let owner: usize = owner.strip_prefix("owner-").context("Qwen upload owner")?.parse()?;
                let bounds = host.uploads.get(owner).context("Qwen upload owner was not admitted")?;
                match *kind {
                    "prefill" => bounds[0],
                    "decode" => bounds[1],
                    "mtp-prefill" => host.mtp_upload.context("Qwen MTP prefill uploads were not admitted")?,
                    other => anyhow::bail!("unadmitted Qwen upload owner {other}"),
                }.at(units).context("Qwen upload bound overflow")?
            }
            ["staging", "mapped-table-worker"] => host.ple.context("Qwen PLE was not admitted")?.worker,
            ["mapped-table", "row-cache"] => host.ple.context("Qwen PLE was not admitted")?.row_cache,
            _ => anyhow::bail!("unadmitted Qwen host owner {name}"),
        };
        ensure!(*bytes <= bound, "Qwen host owner {name} needs {bytes} bytes, admitted {bound}");
    }
    ensure!(logits <= host.pinned_logits,
        "Qwen pinned logits need {logits} bytes, admitted {}", host.pinned_logits);
    ensure!(prefix_bytes <= host.prefix_quota,
        "Qwen host prefix needs {prefix_bytes} bytes, admitted {}", host.prefix_quota);
    tracing::info!(owners = tables.len(), prefix_bytes, "Qwen host owners within the admitted recipe");
    Ok(())
}

#[cfg(test)]
mod tests {
    use cuteafd_loader::families::qwen4::NgramHasher;

    #[test]
    fn a_decode_top_up_matches_a_placement_built_over_all_its_units() {
        let history = || NgramHasher::from_config(1000, 20_000_000, 3, 2, 0, 1234, 7).start();
        let mut grown = super::Qwen4Placement::new(vec![4, 0], 1, history());
        grown.extend(&[6, 2]);
        let whole = super::Qwen4Placement::new(vec![4, 0, 6, 2], 1, history());
        assert_eq!((&grown.units, &grown.pages, &grown.pool_pages), (&whole.units, &whole.pages, &whole.pool_pages));
    }

    #[test]
    fn mtp_source_charge_uses_scheduler_verify_budget() -> anyhow::Result<()> {
        use super::{DECODE_ROWS, MTP_DEFERRED_STEPS};
        use cuteafd_core::{DraftControl, DraftHistory, DrafterSpec, HeadScore};
        assert_eq!(DECODE_ROWS, cuteafd_loader::families::qwen4::VERIFY_ROWS);
        let spec = DrafterSpec { block: MTP_DEFERRED_STEPS, head: HeadScore::None, copy: 0 };
        for control in [DraftControl::History, DraftControl::Fixed(MTP_DEFERRED_STEPS)] {
            for requests in 1..=DECODE_ROWS {
                let plan: Vec<_> = (0..requests as u64).map(|id| (id, MTP_DEFERRED_STEPS)).collect();
                let depths = DraftHistory::new(spec, control).plan(&plan, DECODE_ROWS);
                assert!(depths.iter().map(|depth| depth + 1).sum::<usize>() <= DECODE_ROWS);
                assert!(depths.iter().all(|&depth| depth <= MTP_DEFERRED_STEPS));
            }
        }
        // Unbudgeted, a full lane of full chains would not fit.
        assert!(DECODE_ROWS * (MTP_DEFERRED_STEPS + 1) > DECODE_ROWS);
        Ok(())
    }

    #[test]
    fn padded_page_rows_match_clone_and_resize() {
        for (pages, stride) in [(vec![3, 1], 4), (vec![7, 8, 9], 3), (vec![5, 6, 7], 2), (vec![], 2)] {
            let mut table = vec![-1];
            super::extend_padded(&mut table, &pages, stride);
            let mut expected = pages.clone();
            expected.resize(stride, 0);
            assert_eq!(table[1..], expected[..]);
        }
    }

    #[test]
    fn serving_graphs_default_on_and_respect_the_explicit_opt_out() {
        use super::graphs_enabled;
        assert!(graphs_enabled(None));
        assert!(graphs_enabled(Some("1")));
        assert!(!graphs_enabled(Some("0")));
    }

    #[test]
    fn graph_keys_retain_only_baked_attention_scalars() {
        use super::*;
        let tables = StepTables { spec: true, long: true, page_width: 64, page_stride: 64,
            pool_width: 8, pool_stride: 16, ..Default::default() };
        let mut changed = tables.clone();
        changed.pool_width = 32;
        for attention in [Some(Qwen4Attention::Full), Some(Qwen4Attention::Gdn), None] {
            let key = |tables: &StepTables| GraphKey::new(3, 8, attention, tables);
            assert_eq!(key(&tables), key(&changed));
            let mut geometry = tables.clone();
            geometry.page_width *= 2;
            geometry.page_stride *= 2;
            geometry.pool_stride *= 2;
            assert_eq!(key(&tables) == key(&geometry), attention != Some(Qwen4Attention::Full));
            let mut spec = tables.clone();
            spec.spec = false;
            assert_eq!(key(&tables) == key(&spec), attention.is_none());
        }
        let mut short = tables.clone();
        short.long = false;
        let mut other = short.clone();
        other.pool_stride *= 2;
        assert_eq!(GraphKey::new(3, 8, Some(Qwen4Attention::Full), &short),
            GraphKey::new(3, 8, Some(Qwen4Attention::Full), &other));
        assert_ne!(GraphKey::new(3, 8, Some(Qwen4Attention::Gdn), &other),
            GraphKey::new(3, 8, None, &other));
        // Exact live row counts key distinctly (no bucketing).
        for rows in [1, 3, 5, 9, 17, 25, 33, 37, 64] {
            assert_eq!(GraphKey::new(3, rows, Some(Qwen4Attention::Gdn), &short).rows, rows);
        }
    }

    #[test]
    fn media_gathers_pack_sequences_and_shift_mtp_by_one_native_row() {
        use cuteafd_engine::media::{EmbeddingCache, ImageKey, MediaSpan, RequestMedia};
        let key = ImageKey([7;32]);
        let mut cache = EmbeddingCache::new(16);
        let pin = cache.reserve(key, 16).unwrap();
        let payload: Vec<u8> = (0..16).collect();
        let lease = cache.complete(key, std::sync::Arc::from(payload.clone())).unwrap();
        let mut media = RequestMedia::new(vec![MediaSpan { start: 2, len: 4, key: key.into() }], 2, 8).unwrap();
        media.attach(lease).unwrap(); drop(pin);
        let mut image = super::Qwen4Placement::new(vec![0], 0, super::NgramHistory(vec![0]));
        image.media = Some(media);
        let text = super::Qwen4Placement::new(vec![1], 1, super::NgramHistory(vec![0]));
        let packed = super::embedding_media(&[(&text, 0, 2), (&image, 1, 6)]).unwrap();
        assert_eq!(packed.indices, [3,4,5,6]); assert_eq!(packed.features, payload);
        // MTP row p embeds token p+1, including the first image row.
        let shifted = super::mtp_embedding_media(&[super::MtpGroup { placement: &image, rows: vec![
            super::MtpRow { position: 1, token: 248056, source: 0 },
            super::MtpRow { position: 4, token: 248056, source: 1 },
        ] }]).unwrap();
        assert_eq!(shifted.indices, [0,1]); assert_eq!(shifted.features, [0,1,2,3,12,13,14,15]);
        assert!(super::embedding_media(&[(&image, 8, 2)]).unwrap().indices.is_empty());
    }

    #[test]
    fn fp8_head_spans_cover_every_logits_row_once() {
        for rows in 1..=4096 {
            let mut next = 0;
            for (first, n) in super::fp8_head_spans(rows) {
                assert_eq!(first, next);
                assert!((1..=super::FP8_HEAD_ROWS).contains(&n));
                next += n;
            }
            assert_eq!(next, rows);
        }
        assert_eq!(super::fp8_head_spans(0).count(), 0);
    }

    #[test]
    fn image_rotary_metadata_leaves_logical_slots_and_native_ple_history_unchanged() {
        use cuteafd_loader::families::qwen4::{ImageSpan, NgramHistory, RopePositions};
        let ids = [10, 248053, 248056, 248056, 248056, 248056, 248056, 248056, 248054];
        let images = [ImageSpan { start: 2, grid: [1, 4, 6] }];
        let hasher = NgramHasher::from_config(248_320, 20_000_000, 3, 8, 0, 1234, 248044);
        let mut placement = super::Qwen4Placement::new(vec![2], 0, NgramHistory(vec![248044; 2]));
        let before: Vec<_> = (0..ids.len()).map(|row| (placement.record(row).unwrap(),
            placement.pool_slot(row).unwrap())).collect();
        placement.rope = RopePositions::new(&ids, &images, 248056, 2).unwrap();
        assert_eq!(placement.rope.at(7).unwrap(), [2, 3, 4]);
        assert_eq!(placement.rope.at(8).unwrap(), [5; 3]);
        for (row, expected) in before.into_iter().enumerate() {
            assert_eq!((placement.record(row).unwrap(), placement.pool_slot(row).unwrap()), expected);
        }
        let mut history = hasher.start();
        hasher.hash(&mut history, &ids, &mut Vec::new()).unwrap();
        assert_eq!(super::ngram_history(248044, 3, &ids), history);
        assert_eq!(history.0, vec![248056, 248054]);
        // Prefix restoration installs rebuilt request metadata, not cached radix ids.
        placement.rope = RopePositions::new(&ids, &images, 248056, 2).unwrap();
        assert_eq!(placement.rope.at(4).unwrap(), [2, 2, 4]);
        assert_eq!(placement.rope.at(ids.len()).unwrap(), [6; 3]);
    }

    #[test]
    fn history_after_tokens_is_what_the_hash_leaves() {
        for ngram in [2usize, 3, 5] {
            let hasher = NgramHasher::from_config(1000, 20_000_000, ngram, 2, 0, 1234, 7);
            for len in [0usize, 1, 2, 3, 9] {
                let tokens: Vec<u32> = (0..len as u32).map(|t| if t == 4 { 7 } else { 100 + t }).collect();
                let mut history = hasher.start();
                hasher.hash(&mut history, &tokens, &mut Vec::new()).unwrap();
                assert_eq!(super::ngram_history(7, ngram, &tokens), history, "ngram {ngram} len {len}");
            }
        }
    }
}

/// Refcounted allocation units and free state slots (goldens and benches; serving takes its
/// units from the prefix cache's pool).
pub(crate) struct Allocator {
    units: cuteafd_engine::prefix::RefPagePool,
    slots: Vec<i32>,
    cfg: Qwen4Config,
}

impl Allocator {
    /// Over an engine of `pages` record pages (a whole number of units) and `slots` state slots.
    pub fn new(pages: usize, slots: usize, cfg: &Qwen4Config) -> Self {
        Self { units: cuteafd_engine::prefix::RefPagePool::new(pages / UNIT_PAGES, UNIT_ROWS),
            slots: (0..slots as i32).rev().collect(), cfg: cfg.clone() }
    }

    /// Reserves every unit a sequence of up to `capacity` tokens needs and a state slot (the
    /// engine zeroes the slot and maps the pool pages before the first step).
    pub fn admit(&mut self, capacity: usize) -> Result<Qwen4Placement> {
        let slot = self.slots.pop().context("state slots exhausted")?;
        match self.units.alloc(self.units.pages_for(capacity)) {
            Ok(units) => Ok(Qwen4Placement::new(units, slot, history_of(&self.cfg, &[]))),
            Err(error) => {
                self.slots.push(slot);
                Err(error).context("cache pages exhausted")
            }
        }
    }

    /// A second sequence starting as `source`'s first `len` rows (`tokens`): full units shared,
    /// the partial tail unit copied by the caller (the returned copy), its own state slot.
    pub fn fork(&mut self, source: &Qwen4Placement, tokens: &[u32], capacity: usize)
        -> Result<(Qwen4Placement, Option<cuteafd_engine::prefix::TailCopy>)> {
        let slot = self.slots.pop().context("state slots exhausted")?;
        match self.units.fork(&source.units, tokens.len(), self.units.pages_for(capacity)) {
            Ok(fork) => {
                let mut placement = Qwen4Placement::new(fork.pages, slot, history_of(&self.cfg, tokens));
                placement.rope = source.rope.clone();
                placement.media = source.media.clone();
                Ok((placement, fork.copy))
            },
            Err(error) => {
                self.slots.push(slot);
                Err(error).context("cache pages exhausted")
            }
        }
    }

    pub fn release(&mut self, placement: Qwen4Placement) {
        self.units.release(&placement.units);
        self.slots.push(placement.slot);
    }
}

struct Workspace<'a> {
    rows: usize,
    slot_rows: usize,
    row_offset: Cell<usize>,
    position: Cell<usize>,
    units: Cell<bool>,
    streams: [Dev<'a>; 2],
    inject: Dev<'a>,
    x: Dev<'a>,
    delta: Dev<'a>,
    shared: Dev<'a>,
    routed: Dev<'a>,
    positions: Dev<'a>,
    rope_positions: Dev<'a>,
    block_rope_positions: Dev<'a>,
    kv_slots: Dev<'a>,
    slots: Dev<'a>,
    seq_first: Dev<'a>,
    local_seq_first: Option<Dev<'a>>,
    table_view: Cell<Option<usize>>,
    pool_slots: Dev<'a>,
    cache_lengths: Dev<'a>,
    page_table: Dev<'a>,
    pool_table: Dev<'a>,
    ple_ids: Dev<'a>,
    /// Mapped PLE table: the step's gathered rows ([rows x 16, row bytes]) and
    /// the identity ids the program reads them with.
    ple_rows: Option<(Dev<'a>, Dev<'a>)>,
    query: Dev<'a>,
    gate: Dev<'a>,
    index_q: Dev<'a>,
    attn: Dev<'a>,
    blocks: Dev<'a>,
    indices: Dev<'a>,
    lengths: Dev<'a>,
    scratch: Dev<'a>,
    topk_scratch: Dev<'a>,
    logits: Dev<'a>,
    logit_rows: usize,
    logit_offset: Cell<usize>,
    /// Terminal target rows retained until the span's canonical MTP hand-off.
    terminal_streams: Option<Dev<'a>>,
    terminal_valid: Cell<bool>,
    router_logits: Dev<'a>,
    route_ids: Dev<'a>,
    route_weights: Dev<'a>,
    wire: Dev<'a>,
    /// MTP steps: the source row of each row's feedback, and the greedy draft
    /// (U32 token, FP32 logit) of each head row.
    hidden_rows: Dev<'a>,
    argmax: Dev<'a>,
    /// The step's token ids (U32, gathered from the device embedding table;
    /// MTP chain steps: indices into the previous step's drafts).
    ids: Dev<'a>,
    /// Greedy selection of the logits rows inside the decode graph: U32 ids, then U32 statuses.
    select: Dev<'a>,
    /// Immutable pinned sources through every queued metadata read.
    staging: RefCell<crate::shared::token_table::RetainedUploads<'a>>,
    /// Pinned landing of the logits rows.
    logits_host: RefCell<HostAllocation<'a>>,
    head: Option<VocabularyHead<'a>>,
    _head_workspace: Dev<'a>,
}

/// Address pair for a sequence's contiguous metadata. The caller borrows its
/// workspace through submission; the engine retains that workspace through drain.
#[derive(Clone, Copy)]
pub(super) struct AttentionTables {
    pages: *mut c_void,
    pools: *mut c_void,
    width: usize,
}

impl Workspace<'_> {
    fn row_table(&self, buffer: &Dev<'_>, width: usize) -> *mut c_void {
        buffer.buffer.ptr.wrapping_byte_add(self.table_view.get().unwrap_or(0) * width)
    }

    fn first_table(&self) -> *mut c_void {
        if self.table_view.get().is_some() {
            self.local_seq_first.as_ref().expect("prefill member first plane").buffer.ptr
        } else { self.seq_first.buffer.ptr }
    }

    fn attention_tables(&self, [page_offset, pool_offset]: [usize; 2], width: usize) -> AttentionTables {
        AttentionTables {
            pages: self.page_table.buffer.ptr.wrapping_byte_add(page_offset * size_of::<i32>()),
            pools: self.pool_table.buffer.ptr.wrapping_byte_add(pool_offset * size_of::<i32>()),
            width,
        }
    }

    fn live(&self, buffer: &Dev<'_>, stride: usize) -> *mut c_void {
        self.live_buffer(buffer, stride).ptr
    }
    fn unit_offset(slot: usize, unit_rows: usize) -> usize { slot * unit_rows }

    fn live_buffer(&self, buffer: &Dev<'_>, stride: usize) -> cuteafd_ffi::CuteafdDeviceBuffer {
        let offset = self.row_offset.get() * stride;
        assert!(offset < buffer.buffer.bytes);
        cuteafd_ffi::CuteafdDeviceBuffer { ptr: buffer.buffer.ptr.wrapping_byte_add(offset),
            bytes: buffer.buffer.bytes - offset, ..buffer.buffer }
    }
}

/// One MTP row: the pair (target or draft pre-mixer streams at `position`,
/// the token at `position + 1`), its streams read from row `source` of the step's source.
#[derive(Debug, Clone, Copy)]
pub(crate) struct MtpRow {
    pub position: usize,
    pub token: u32,
    pub source: i32,
}

/// A sequence's rows of an MTP step (contiguous, in position order).
pub(crate) struct MtpGroup<'p> {
    pub placement: &'p Qwen4Placement,
    pub rows: Vec<MtpRow>,
}

/// The tokens an MTP step embeds.
#[derive(Debug, Clone, Copy)]
pub(crate) enum MtpTokens<'t> {
    /// Host token ids, one per row.
    Host(&'t [u32]),
    /// Chain steps: row `r` embeds draft `index[r]` of the deferred step `step`.
    Drafts { step: usize, index: &'t [u32] },
}

/// What an MTP step with head rows hands back.
#[derive(Debug, Clone, Copy)]
pub(crate) enum MtpOut {
    /// Each head row's greedy draft (token, logit) now, and with `logits`
    /// the head rows' FP32 logits (synchronizes).
    Download { logits: bool },
    /// The drafts stay on the device as step `step` of the draft cycle
    /// ([`Qwen4Engine::mtp_drafts`] reads them back).
    Defer { step: usize },
}

/// Draft steps one cycle can defer (the most drafts a sequence verifies).
pub(crate) const MTP_DEFERRED_STEPS: usize = DECODE_ROWS;

/// Where an MTP step reads its rows' streams.
#[derive(Debug, Clone, Copy)]
pub(crate) enum MtpSource {
    /// The stash of target rows (row `slot * MTP_PENDING_ROWS + i`).
    Pending,
    /// The previous MTP step's output streams in the same workspace.
    Chain,
    /// The workspace's final target streams of its last step (prefill).
    Target,
    /// A caller buffer of `[rows, 4, H]` streams (golden checks).
    Buffer(*mut c_void),
}

fn validate_layer_owners(owners: &[usize], devices: usize) -> Result<()> {
    ensure!(!owners.is_empty() && matches!(devices, 1 | 2), "invalid Qwen attention owners");
    ensure!(owners.iter().all(|&owner| owner < devices), "Qwen attention owner out of range");
    if devices == 1 { return Ok(()); }
    cuteafd_loader::placement::families::qwen4::check_dual_layer_owners(owners)
}

pub(crate) struct Qwen4Engine<'a> {
    terminal: Cell<crate::shared::peer_split::TerminalState>,
    pub(crate) spark_bf16: bool,
    quantize_grids: Vec<Fp8QuantizeGrid>,
    active_owner: Cell<usize>,
    pub library: &'a NativeLibrary,
    pub programs: &'a Programs<'a>,
    pub cfg: Qwen4Config,
    pub weights: Qwen4Weights<'a>,
    pub ple: Option<super::ple::PleTable<'a>>,
    pub stream: *mut c_void,
    pub max_context: usize,
    pub prefill_rows: usize,
    pub(super) prefill_shape: Option<cuteafd_loader::serving_capacity::PrefillShape>,
    pub(super) host_prefix_config: Option<cuteafd_hostcache::config::Config>,
    host_memory_admitted: bool,
    /// Host owners the recipe admitted before allocation; None without a solve.
    pub(super) admitted_host: Option<cuteafd_loader::placement::families::qwen4::QwenHost>,
    request_id: Cell<u64>,
    pub pages: usize,
    pub kv_format: cuteafd_loader::families::qwen4::Qwen4KvCache,
    pub kv_record_bytes: usize,
    pub slots: usize,
    /// Per full layer: the K/V record pool.
    kv: Vec<Option<Dev<'a>>>,
    /// One global layer map; recurrent pools are compact only within an owner.
    state_map: LayerStateMap,
    gdn_banks: Vec<DeviceOwner<'a, GdnBank<'a>>>,
    /// Per full layer: raw per-token index keys (BF16 [record slots, 128]) and pooled block keys.
    index: Vec<Option<(Dev<'a>, Dev<'a>)>>,
    /// PLE conv state pool (BF16 [slots, 9, 4H]) and its speculative replay record ([64, 4H]).
    ple_state: Option<Dev<'a>>,
    ple_replay: Option<Dev<'a>>,
    /// Mapped PLE table: the current step's rows, gathering until the PLE layer.
    ple_pending: RefCell<Option<crate::shared::mapped_table::PendingRows>>,
    /// The MTP layer's K/V records and index caches, and the stash of target
    /// pre-mixer stream rows awaiting the MTP (BF16 [slots, 64, 4, H]).
    mtp_kv: Option<(Dev<'a>, Dev<'a>, Dev<'a>)>,
    mtp_pending: Option<Dev<'a>>,
    /// Where the last step left its final streams (decode, stream buffer).
    last_streams: std::cell::Cell<(bool, usize)>,
    last_stream_offset: Cell<usize>,
    /// Where the last MTP step left its output streams (decode, stream buffer).
    mtp_streams: std::cell::Cell<(bool, usize)>,
    /// Pool-cache pages: one per allocation unit (`pages / UNIT_PAGES`).
    pub pool_pages: usize,
    workspace: RefCell<Option<Workspace<'a>>>,
    mtp_prefill_workspace: RefCell<Option<Workspace<'a>>>,
    decode_workspace: RefCell<Option<Workspace<'a>>>,
    peer_workspace: RefCell<Option<Workspace<'a>>>,
    peer_decode_workspace: RefCell<Option<Workspace<'a>>>,
    workspaces_prepared: Cell<bool>,
    experts: Option<Experts<'a>>,
    hops: Option<crate::shared::peer_split::hop::HopLink<'a>>,
    /// Host seconds: GPU wait before expert exchanges, the exchanges.
    pub profile: RefCell<[f64; 2]>,
    graphs: RefCell<Vec<GraphBank<GraphKey, GraphExec<'a>>>>,
    use_graphs: bool,
    /// The bounded graph cache is configured for this launch: lazy exact-row
    /// capture and miss accounting are on. Cleared only by `CUTEAFD_QWEN4_GRAPHS=0`.
    graph_capture: Cell<bool>,
    graph_cycles: RefCell<Option<crate::shared::decode_graph::device::EventCycles<'a>>>,
    /// Observer seeded and startup graph initialization completed.
    graph_warmed: Cell<bool>,
    /// Prefill programs over FP8-only projections quantize their activations
    /// (W8A8, `fp8_rows` 1) instead of W8A16 (`--fp8-prefill-w8a8`).
    pub w8a8_prefill: bool,
    packed_row_cap: Cell<Option<usize>>,
    /// L2 prefetch of the next layer's weights during decode exchanges.
    pub l2: Option<crate::shared::l2_prefetch::L2Prefetch>,
    /// The token embedding table (resident on this GPU or read from its shard).
    pub embedding: TokenEmbedding<'a>,
    /// Active decode/verify row capacity (`64` or `128`).
    pub decode_rows: usize,
    /// Deferred MTP drafts of a cycle: U32 [MTP_DEFERRED_STEPS, decode_rows].
    mtp_drafts: Dev<'a>,
    pub(super) attention_plan: cuteafd_core::coordinator_programs::AttentionReductionPlan,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct GraphKey {
    segment: usize,
    rows: usize,
    spec: bool,
    head: bool,
    long: bool,
    page_width: usize,
    page_stride: usize,
    pool_stride: usize,
}

impl GraphKey {
    fn new(segment: usize, rows: usize, attention: Option<Qwen4Attention>, tables: &StepTables) -> Self {
        let full = attention == Some(Qwen4Attention::Full);
        Self { segment, rows, spec: attention.is_some() && tables.spec, head: attention.is_none(),
            long: full && tables.long,
            page_width: if full { tables.page_width } else { 0 },
            page_stride: if full { tables.page_stride } else { 0 },
            pool_stride: if full && tables.long { tables.pool_stride } else { 0 } }
    }
}

struct GraphExec<'a>(*mut c_void, Device<'a>);

impl Drop for GraphExec<'_> {
    fn drop(&mut self) {
        if self.1.library.is_quarantined_after_failed_drain() { return; }
        // SAFETY: the executable graph is owned here and its execution stream drained.
        let _ = self.1.run(|| unsafe { self.1.library.cuda_graph_exec_destroy(self.0) });
    }
}

fn bytes_of<T: Copy>(values: &[T]) -> &[u8] {
    // SAFETY: plain-old-data slices viewed as bytes for host->device copies.
    unsafe { std::slice::from_raw_parts(values.as_ptr().cast(), std::mem::size_of_val(values)) }
}

/// Where the step's streams live after each layer (for callbacks and forcing).
type LayerHook<'h> = Option<&'h mut dyn FnMut(usize, &[u8]) -> Result<()>>;
type Forced<'h> = Option<&'h dyn Fn(usize) -> Option<Vec<u8>>>;

impl<'a> Qwen4Engine<'a> {
    pub(crate) fn decode_cap(&self) -> &'static str {
        if self.decode_rows <= DECODE_ROWS { "m64" } else { "m128" }
    }

    fn is_decode_cap(cap: &str) -> bool {
        matches!(cap, "m64" | "m128")
    }

    #[allow(clippy::too_many_arguments)]
    pub fn new(library: &'a NativeLibrary, programs: &'a Programs<'a>, cfg: Qwen4Config, weights: Qwen4Weights<'a>,
        ple: Option<super::ple::PleTable<'a>>, stream: *mut c_void, max_context: usize, prefill_rows: usize,
        pages: usize, slots: usize, kv_format: cuteafd_loader::families::qwen4::Qwen4KvCache, embedding: TokenEmbedding<'a>) -> Result<Self> {
        Self::new_with_decode_rows(library, programs, cfg, weights, ple, stream, max_context, prefill_rows, pages, slots, DECODE_ROWS, kv_format, embedding)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn new_with_decode_rows(library: &'a NativeLibrary, programs: &'a Programs<'a>, cfg: Qwen4Config, weights: Qwen4Weights<'a>,
        ple: Option<super::ple::PleTable<'a>>, stream: *mut c_void, max_context: usize, prefill_rows: usize,
        pages: usize, slots: usize, decode_rows: usize, kv_format: cuteafd_loader::families::qwen4::Qwen4KvCache, embedding: TokenEmbedding<'a>) -> Result<Self> {
        let device = Device { library, id: library.cuda_get_device()? };
        let owners = vec![0; weights.layers.len()];
        Self::new_placed_with_decode_rows(library, cfg, weights, ple, &owners, &[(device, programs, stream)],
            max_context, prefill_rows, pages, slots, decode_rows, kv_format, embedding)
    }

    /// Whole-width attention owners; head, embedding and MTP remain on rank zero.
    /// Execution is enabled separately, after TP2 exchange and graph admission.
    #[allow(clippy::too_many_arguments)]
    pub fn new_placed(library: &'a NativeLibrary, cfg: Qwen4Config, weights: Qwen4Weights<'a>,
        ple: Option<super::ple::PleTable<'a>>, owners: &[usize],
        ranks: &[(Device<'a>, &'a Programs<'a>, *mut c_void)], max_context: usize,
        prefill_rows: usize, pages: usize, slots: usize,
        kv_format: cuteafd_loader::families::qwen4::Qwen4KvCache, embedding: TokenEmbedding<'a>) -> Result<Self> {
        Self::new_placed_with_decode_rows(library, cfg, weights, ple, owners, ranks, max_context, prefill_rows,
            pages, slots, DECODE_ROWS, kv_format, embedding)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn new_placed_with_decode_rows(library: &'a NativeLibrary, cfg: Qwen4Config, weights: Qwen4Weights<'a>,
        ple: Option<super::ple::PleTable<'a>>, owners: &[usize],
        ranks: &[(Device<'a>, &'a Programs<'a>, *mut c_void)], max_context: usize,
        prefill_rows: usize, pages: usize, slots: usize, decode_rows: usize,
        kv_format: cuteafd_loader::families::qwen4::Qwen4KvCache, embedding: TokenEmbedding<'a>) -> Result<Self> {
        let decode_rows = cuteafd_loader::families::qwen4::decode_capacity(decode_rows);
        let decode_cap = if decode_rows <= DECODE_ROWS { "m64" } else { "m128" };
        let gdn_commit_program = if decode_rows <= DECODE_ROWS { "qwen4_gdn_commit" } else { "qwen4_gdn_commit_m128" };
        ensure!(!ranks.is_empty() && ranks.len() <= 2, "Qwen supports one or two attention owners");
        ensure!(ranks.iter().all(|(device, _, _)| std::ptr::eq(device.library, library)),
            "Qwen attention owners use different native libraries");
        ensure!(ranks.len() == 1 || ranks[0].0.id != ranks[1].0.id, "duplicate Qwen owner device");
        let (home, programs, stream) = ranks[0];
        let _home_scope = home.enter()?;
        let _memory_scope = cuteafd_ffi::memory_ledger::scope("kv");
        validate_layer_owners(owners, ranks.len())?;
        let quantize_grids = ranks.iter().map(|(device, _, _)|
            device.run(|| Ok(Fp8QuantizeGrid::new(library.sm_count()?, None)?))).collect::<Result<Vec<_>>>()?;
        ensure!(embedding.hidden() == cfg.hidden, "embedding rows of {} for hidden {}", embedding.hidden(), cfg.hidden);
        cfg.check_programs()?;
        for &(_, programs, _) in ranks {
            super::programs::validate_all(&cfg, programs, max_context)?;
        }
        let kv_record_bytes = kv_format.record_bytes(cfg.kv_heads, cfg.head_dim);
        let kv_suffix = if kv_format == cuteafd_loader::families::qwen4::Qwen4KvCache::Fp8 { "_kv_fp8" } else { "" };
        for (owner, &(_, owner_programs, _)) in ranks.iter().enumerate() {
            for cap in [decode_cap, "m4096"] {
                for layer in weights.layers.iter().enumerate()
                    .filter(|(global, _)| owners[*global] == owner).map(|(_, layer)| layer)
                    .chain(weights.mtp.iter().filter(|_| owner == 0).map(|m| &m.layer))
                    .filter(|layer| layer.attention == Qwen4Attention::Full) {
                    owner_programs.spec(&format!("qwen4_sparse_gqa{kv_suffix}_{cap}"))?;
                    let stem = if Self::w8(layer) { "attn_producer_w8" } else { "attn_producer" };
                    owner_programs.spec(&format!("qwen4_{stem}{kv_suffix}_{cap}"))?;
                }
            }
        }
        let zeroed = |bytes: usize, tag: Tag| -> Result<Dev<'a>> {
            let allocation = Rc::new(Allocation::new(Device { library, id: library.cuda_get_device()? }, bytes.max(256), tag)?);
            library.cuda_zero_bytes(allocation.buffer, allocation.buffer.bytes)?;
            Ok(allocation)
        };
        // Whole allocation units: four record pages and one pool page each.
        let pages = pages.max(1).next_multiple_of(UNIT_PAGES);
        let pool_pages = pages / UNIT_PAGES;
        let (mut kv, mut index) = (Vec::new(), Vec::new());
        let kinds: Vec<_> = weights.layers.iter().map(|l| l.attention).collect();
        let state_map = LayerStateMap::new(&kinds, owners, ranks.len())?;
        for (global, layer) in weights.layers.iter().enumerate() {
            let device = ranks[owners[global]].0;
            ensure!(layer.device()? == device.id, "Qwen layer {global} weights are not on their attention owner");
            let _layer_scope = device.enter()?;
            match layer.attention {
                Qwen4Attention::Full => {
                    let (records, keys, pooled) = record_arena_bytes(pages, pool_pages, kv_record_bytes)?;
                    let arena = zeroed(records + keys + pooled, Tag::kv("qwen4/kv-arena"))?;
                    kv.push(Some(Rc::new(Allocation::view(&arena, 0, records)?)));
                    index.push(Some((Rc::new(Allocation::view(&arena, records, keys)?),
                        Rc::new(Allocation::view(&arena, records + keys, pooled)?))));
                }
                Qwen4Attention::Gdn => {
                    kv.push(None);
                    index.push(None);
                }
            }
        }
        let gdn_banks = ranks.iter().enumerate().map(|(owner, &(device, programs, stream))|
            device.own(|| GdnBank::new_with_replay_rows(library, programs, stream, cfg.gdn_geometry(), state_map.gdn_layers[owner], slots, gdn_commit_program, decode_rows)))
            .collect::<Result<Vec<_>>>()?;
        ensure!(weights.head.allocations().iter().all(|a| a.device.id == home.id)
            && weights.mixer.iter().all(|a| a.device.id == home.id), "Qwen head must stay on owner zero");
        if let Some(mtp) = &weights.mtp {
            ensure!(mtp.layer.device()? == home.id, "Qwen MTP must stay on owner zero");
        }
        let ple_owner = cfg.ple_layers.first().filter(|&&index| index < owners.len()).map(|&index| owners[index]);
        if let (Some(table), Some(owner)) = (&ple, ple_owner) {
            ensure!(table.device == ranks[owner].0.id,
                "Qwen PLE table must load on its attention owner");
        }
        let (ple_state, ple_replay) = if weights.layers.len() > cfg.ple_layers.first().copied().unwrap_or(usize::MAX) {
            let _ple_scope = ranks[ple_owner.context("Qwen PLE state owner")?].0.enter()?;
            (Some(zeroed(slots * PLE_STATE_ROWS * cfg.hc_width() * 2, Tag::kv("qwen4/ple-state"))?),
             Some(zeroed(decode_rows * cfg.hc_width() * 2, Tag::kv("qwen4/ple-replay"))?))
        } else {
            (None, None)
        };
        // The MTP layer's own attention cache and its per-slot stash of target
        // rows (snapshotted with the rest of a slot's state) are drafter state.
        let (mtp_kv, mtp_pending) = if weights.mtp.is_some() {
            let (records, keys, pooled) = record_arena_bytes(pages, pool_pages, kv_record_bytes)?;
            let arena = zeroed(records + keys + pooled, Tag::drafter_kv("qwen4/mtp-kv-arena"))?;
            (Some((Rc::new(Allocation::view(&arena, 0, records)?),
                Rc::new(Allocation::view(&arena, records, keys)?),
                Rc::new(Allocation::view(&arena, records + keys, pooled)?))),
             Some(zeroed(slots * MTP_PENDING_ROWS * HC * cfg.hidden * 2, Tag::drafter_kv("qwen4/mtp-stash"))?))
        } else {
            (None, None)
        };
        let graphs = (0..gdn_banks.len()).map(|_| GraphBank::new()).collect::<Vec<_>>();
        Ok(Self { terminal: Cell::new(crate::shared::peer_split::TerminalState::Active), quantize_grids, active_owner: Cell::new(0), library, programs, cfg, weights, ple, stream, max_context, prefill_rows, prefill_shape: None, host_prefix_config: None, host_memory_admitted: false, admitted_host: None, request_id: Cell::new(1), pages, slots, kv_format, kv_record_bytes, kv, state_map, gdn_banks, index, ple_state, ple_replay, ple_pending: RefCell::new(None),
            mtp_kv, mtp_pending,
            last_streams: std::cell::Cell::new((false, 0)), last_stream_offset: Cell::new(0), mtp_streams: std::cell::Cell::new((false, 0)),
            pool_pages, workspace: RefCell::new(None), mtp_prefill_workspace: RefCell::new(None),
            decode_workspace: RefCell::new(None), peer_workspace: RefCell::new(None),
            peer_decode_workspace: RefCell::new(None), workspaces_prepared: Cell::new(false), experts: None, hops: None, profile: RefCell::new([0.0; 2]),
            graphs: RefCell::new(graphs),
            use_graphs: graphs_enabled(std::env::var("CUTEAFD_QWEN4_GRAPHS").ok().as_deref()),
            graph_capture: Cell::new(false),
            graph_cycles: RefCell::new(None),
            graph_warmed: Cell::new(false),
            w8a8_prefill: false, spark_bf16: false,
            packed_row_cap: Cell::new(None),
            l2: None, embedding,
            decode_rows,
            mtp_drafts: zeroed(MTP_DEFERRED_STEPS * decode_rows * 4, Tag::drafter_workspace("qwen4/mtp-drafts"))?,
            attention_plan: cuteafd_core::coordinator_programs::AttentionReductionPlan::new(DECODE_ROWS, None, None)? })
    }

    /// The resolved capture shape: one or two layer owners, and whether Sparks
    /// are attached (a 0-Spark card captures no onboarding graphs; a TP1 card
    /// captures no TP2 peer segments).
    fn graph_shape(&self) -> QwenGraphShapeKey {
        QwenGraphShapeKey::resolve(self.gdn_banks.len(),
            self.has_sparks())
    }

    /// Attach the per-owner graph banks: one bounded cache per physical device,
    /// charged to the family's graph reserve before KV sizing. Returns each
    /// device's resolved reserve.
    pub fn configure_graph_cache(&self) -> Result<Vec<u64>> {
        use crate::shared::decode_graph::device::{Basis, Cycle, Preference, Site};
        let shape = self.graph_shape();
        let mut reserves = Vec::new();
        for (owner, bank) in self.graphs.borrow_mut().iter_mut().enumerate() {
            let resolved = cuteafd_loader::serving_capacity::qwen_graphs::qwen_resolve_reserve(shape, owner)
                .map_err(anyhow::Error::msg)?;
            bank.attach_device(self.library, self.gdn_banks[owner].device.id, resolved.bytes, Site {
                name: "qwen4.segment", cycle: Cycle::Decode, preference: Preference::Adaptive,
                bootstrap: cuteafd_loader::serving_capacity::qwen_graphs::QWEN_GRAPH_BYTES_PER_GRAPH,
                basis: Basis::Bytes,
                spans_devices: super::graph_spanning::spans("qwen4.segment"),
            })?;
            crate::shared::decode_graph::device::set_reserve_policy(self.gdn_banks[owner].device.id, resolved)?;
            tracing::info!(rank = owner, reserve = resolved.bytes, source = resolved.source,
                soft_cap = resolved.base_bytes, growth_headroom = ?resolved.growth_headroom,
                bootstrap = cuteafd_loader::serving_capacity::qwen_graphs::QWEN_GRAPH_BYTES_PER_GRAPH,
                shape = ?shape, "Qwen exact-row graph reserve");
            reserves.push(resolved.bytes);
        }
        if self.use_graphs && reserves.iter().any(|&bytes| bytes > 0) {
            *self.graph_cycles.borrow_mut() = Some(crate::shared::decode_graph::device::EventCycles::new(self.library, &self.gdn_banks.iter().map(|bank| (bank.device.id, bank.stream)).collect::<Vec<_>>())?);
        }
        self.graph_capture.set(self.use_graphs);
        Ok(reserves)
    }

    /// Visit every owner's bank (the shared observer's pass interface).
    fn visit_graphs(&self, visit: &mut crate::shared::decode_graph::device::GraphVisitor<'_>) -> Result<()> {
        let mut graphs = self.graphs.borrow_mut();
        for (owner, bank) in graphs.iter_mut().enumerate() {
            visit(self.gdn_banks[owner].device.id, "qwen4.segment",
                super::graph_spanning::spans("qwen4.segment"), bank)?;
        }
        Ok(())
    }

    fn begin_graph_cycle(&self) -> Result<()> {
        let mut cycles = self.graph_cycles.borrow_mut();
        let Some(cycles) = cycles.as_mut() else { return Ok(()) };
        // SAFETY: graph operands are borrowed persistent engine storage.
        unsafe { cycles.begin_persistent(|visit| self.visit_graphs(visit)) }
    }

    fn graph_completed(&self) -> Result<()> {
        let mut cycles = self.graph_cycles.borrow_mut();
        let Some(cycles) = cycles.as_mut() else { return Ok(()) };
        for bank in self.graphs.borrow_mut().iter_mut() { bank.cache_tick(); }
        // SAFETY: all owners' launches for the submitted step are queued.
        unsafe { cycles.complete() }
    }

    /// Attribute the whole step's replay work to its cycle.
    fn graph_cycle(&self, tables: &StepTables) {
        use crate::shared::decode_graph::device::Cycle;
        let cycle = if !tables.decode { Cycle::Prefill }
            else if tables.spec { Cycle::Verify }
            else { Cycle::Decode };
        for bank in self.graphs.borrow_mut().iter_mut() { bank.cache_cycle(cycle); }
    }

    pub fn set_experts(&mut self, experts: Experts<'a>) {
        self.experts = Some(experts);
    }

    pub fn install_hybrid(&mut self, mut experts: HybridExperts<'a>,
        placement: &cuteafd_loader::placement::Placement, spec: cuteafd_loader::placement::HopSpec) -> Result<()> {
        let planned = placement.layers.iter().enumerate().filter_map(|(layer, assignment)|
            (assignment.experts != cuteafd_loader::placement::ExpertHome::Spark).then_some(layer)).collect::<Vec<_>>();
        ensure!(experts.local_layers == planned
            && experts.local.as_ref().map_or_else(Vec::new, LocalRouted::layers) == planned,
            "Qwen resident expert IDs differ from admitted homes");
        if self.gdn_banks.len() == 2 {
            let ranks = [0, 1].map(|rank| crate::shared::peer_split::RankDevice {
                device: self.gdn_banks[rank].device.id, stream: self.gdn_banks[rank].stream });
            let modes: Vec<_> = self.state_map.layers.iter().map(|home| cuteafd_loader::placement::LayerMode::Whole {
                gpu: home.owner as u8, ffn: cuteafd_loader::placement::FfnMode::Owner }).collect();
            ensure!(placement.layers.iter().map(|layer| layer.mode).eq(modes.iter().copied()),
                "Qwen attention owners differ from admission");
            let hops = cuteafd_loader::placement::plan_hops(&modes, &spec)?;
            ensure!(placement.hops == hops && hops.len() == 2, "Qwen cutover and exit must be admitted");
            if experts.local.is_none() {
                experts.terminal = Some(crate::shared::peer_split::PeerExchange::new_abortable(
                    self.library, ranks, 1, 256, 256)?);
            }
            let mut link = crate::shared::peer_split::hop::HopLink::new(self.library, ranks, &hops,
                spec, self.decode_rows * spec.row_bytes as usize)?;
            experts.with_peer(|peer| link.attach_abort([
                peer.abort_word(0)?.context("Qwen rank zero abort")?,
                peer.abort_word(1)?.context("Qwen rank one abort")?]))?.context("Qwen dual terminal owner")?;
            self.hops = Some(link);
        }
        self.experts = Some(Experts::Hybrid(experts));
        Ok(())
    }

    pub(crate) fn has_sparks(&self) -> bool {
        matches!(&self.experts, Some(Experts::Hybrid(experts)) if !experts.transport.is_empty())
    }
    pub(crate) fn spark_layer(&self, layer: usize) -> bool {
        layer < self.weights.layers.len()
            && matches!(&self.experts, Some(Experts::Hybrid(experts)) if !experts.local_layer(layer))
    }
    fn spark_link(&self, owner: usize) -> Option<&RefCell<SparkLink<'a>>> {
        match &self.experts { Some(Experts::Hybrid(experts)) => experts.transport.get(owner), _ => None }
    }

    /// Admit all still-lazy host consumers, including a frozen serving cache quota.
    pub(super) fn admit_host_memory(&mut self, args: &super::EngineArgs) -> Result<()> {
        ensure!(!self.host_memory_admitted, "Qwen host memory was already admitted");
        let (config, scoring, headroom) = match &args.planner_host_prefix {
            Some((prefix, lanes)) => {
                let prefix = prefix.with_scoring(self.decode_rows,
                    self.prefill_rows, self.cfg.vocab_size)?;
                let layout = super::prefix::Qwen4Prefix::geometry(self)?;
                (prefix.resolve_host_config(layout, self.max_context, *lanes)?, prefix.scoring_host_bytes,
                    prefix.host_cache_headroom_bytes)
            }
            None => (None, cuteafd_core::memory_layout::host_logits_bytes(
                self.decode_rows as u64,
                self.prefill_rows as u64, self.cfg.vocab_size as u64).context("Qwen host scoring overflow")?, 0),
        };
        let tables = self.future_host_workspaces()?;
        match &self.admitted_host {
            // The recipe charged every host owner against one host sample before
            // allocation; check the actual owners fit what it admitted.
            Some(host) => self.validate_admitted_host(host, &tables, config.as_ref().map_or(0, |c| c.bytes))?,
            None => {
                let views: Vec<_> = tables.iter().map(|(name, bytes)| (name.as_str(), *bytes)).collect();
                crate::shared::memory::budget::admit_host_memory(config.as_ref().map_or(0, |c| c.bytes),
                    &views, scoring, headroom)?;
            }
        }
        self.host_prefix_config = config;
        self.host_memory_admitted = true;
        Ok(())
    }

    /// Actual future host owners against the admitted recipe owners: retained
    /// uploads within their page-linear bounds, pinned logits within the
    /// admitted landing, the PLE table's own staging and cache exactly, and the
    /// resolved host prefix within its admitted quota.
    fn validate_admitted_host(&self, host: &cuteafd_loader::placement::families::qwen4::QwenHost,
        tables: &[(String, u64)], prefix_bytes: u64) -> Result<()> {
        // One pool unit per 256-token pool page, the host curves' page domain.
        validate_admitted_host(host, tables, prefix_bytes, self.pool_pages as u64)
    }

    fn future_host_workspaces(&self) -> Result<Vec<(String, u64)>> {
        use super::admission::HostWorkspaceBytes;
        let mut demands = Vec::new();
        if let Some(ple) = &self.ple {
            demands.push(("staging/mapped-table-worker".to_owned(), ple.mapped().host_workspace_bytes() as u64));
            demands.push(("mapped-table/row-cache".to_owned(), ple.mapped().cache_bytes() as u64));
        }
        let ple_ids = self.ple.as_ref().map(|ple| ple.mapped()).is_none();
        for owner in 0..self.gdn_banks.len() {
            for decode in [false, true] {
                if self.workspace_slot(owner, decode)?.borrow().is_some() { continue; }
                let rows = if decode { self.decode_rows } else { self.prefill_shape
                    .map_or(self.prefill_rows, |shape| shape.unit_rows as usize) };
                let logits = if decode { self.decode_rows } else { 1 };
                let host = HostWorkspaceBytes::new(&self.cfg, rows, decode, self.pages, self.pool_pages,
                    if decode { 0 } else { self.prefill_shape.map_or(0, |shape| shape.waves as usize) },
                    self.has_sparks(), self.weights.mtp.is_some(), ple_ids, logits, self.gdn_banks.len() == 1)?;
                let label = format!("qwen4/owner-{owner}/{}", if decode { "decode" } else { "prefill" });
                for (kind, bytes) in [("staging", host.staging), ("pinned-logits", host.logits)] {
                    if bytes > 0 { demands.push((format!("{label}/{kind}"), bytes)); }
                }
            }
        }
        if self.prefill_shape.is_some() && self.weights.mtp.is_some() && self.mtp_prefill_workspace.borrow().is_none() {
            let host = HostWorkspaceBytes::mtp(&self.cfg, self.mtp_prefill_rows(),
                self.pages, self.pool_pages, self.prefill_rows.div_ceil(self.mtp_prefill_rows()))?;
            for (kind, bytes) in [("staging", host.staging), ("pinned-logits", host.logits)] {
                demands.push((format!("qwen4/owner-0/mtp-prefill/{kind}"), bytes));
            }
        }
        Ok(demands)
    }

    /// Size every unit-table upload before a request can queue prefill work.
    pub(crate) fn prepare_prefill_units(&self) -> Result<()> {
        let Some(shape) = self.prefill_shape else { return Ok(()) };
        let shape = shape.validate()?;
        for owner in 0..self.gdn_banks.len() {
            self.on_owner(owner, || {
                let slot = self.workspace_slot(owner, false)?;
                if slot.borrow().is_none() {
                    *slot.borrow_mut() = Some(self.workspace_rows(shape.unit_rows as usize, false,
                        1, shape.slot_rows() as usize,
                        if owner == 0 && self.weights.mtp.is_some() { self.prefill_rows } else { 0 }, shape.waves as usize)?);
                }
                Ok(())
            })?;
        }
        if self.weights.mtp.is_some() && self.mtp_prefill_workspace.borrow().is_none() {
            *self.mtp_prefill_workspace.borrow_mut() = Some(self.on_owner(0, ||
                self.workspace(self.mtp_prefill_rows(), false, 1))?);
        }
        Ok(())
    }

    /// Retain every admitted serving workspace and private Lt plan before capture or Ready.
    pub(crate) fn prepare_serving_workspaces(&self) -> Result<()> {
        self.prepare_prefill_units()?;
        for owner in 0..self.gdn_banks.len() {
            for decode in [false, true] {
                self.on_owner(owner, || {
                    let slot = self.workspace_slot(owner, decode)?;
                    if slot.borrow().is_none() {
                        let rows = if decode { self.decode_rows } else { self.prefill_rows };
                        let logits = if decode { self.decode_rows } else { 1 };
                        *slot.borrow_mut() = Some(self.workspace(rows, decode, logits)?);
                    }
                    Ok(())
                })?;
            }
        }
        self.drain_state()?;
        self.workspaces_prepared.set(true);
        Ok(())
    }

    fn allow_workspace_creation(&self) -> Result<()> {
        ensure!(!self.workspaces_prepared.get(), "Qwen serving workspace was not prepared before Ready");
        Ok(())
    }

    /// Seed each physical device's graph memory observer before lazy first-use
    /// exact-row capture opens. Startup prewarm and pins are empty: `qwen4.segment`
    /// is rank-local (`Scope::Local`) on both 1 and 2 GPUs, so every exact row
    /// shape captures lazily on first serving use under the shared default donor rule.
    pub fn warm_decode_graphs(&self, sequences: usize, _speculation: bool) -> Result<usize> {
        ensure!(!self.graph_warmed.get(), "Qwen decode graphs already initialized");
        if !self.use_graphs { return Ok(0); }
        ensure!(self.graph_capture.get(), "Qwen graphs must be configured before initialization");
        cuteafd_core::concurrency::validate("qwen4", sequences as u64)?;
        self.drain_state()?;
        for bank in &self.gdn_banks {
            crate::shared::decode_graph::device::observe_device(self.library, bank.device.id)?;
        }
        self.graph_warmed.set(true);
        tracing::info!(shape = ?self.graph_shape(), "Qwen exact-row decode graphs ready (0 startup prewarm shapes; lazy capture)");
        Ok(0)
    }

    /// Compare exact-row graphed decode/verify logits and persistent cache/state
    /// bytes (including unused storage) against ungraphed eager execution across
    /// non-bucket row counts, both short and long contexts.
    /// Run only before serving: this diagnostic owns its allocator and state slots.
    pub fn check_decode_graphs(&self, tokens: &[u32]) -> Result<()> {
        ensure!(self.use_graphs && self.graph_capture.get() && self.weights.layers.len() == self.cfg.layers,
            "Qwen exact-row graph check needs all layers and configured graphs");
        ensure!(tokens.len() >= 69 && self.max_context >= 69 && self.slots >= 10 && self.pages >= 40,
            "Qwen exact-row graph check needs 69 tokens, ten slots and forty pages");
        let snapshot = |buffers: &[cuteafd_ffi::CuteafdDeviceBuffer]| -> Result<Vec<Vec<u8>>> {
            self.drain_state()?;
            buffers.iter().map(|&buffer| {
                let mut bytes = vec![0; buffer.bytes];
                Device { library: self.library, id: buffer.device_id }.run(|| {
                    self.library.copy_d2h(&mut bytes, buffer)
                })?;
                Ok(bytes)
            }).collect()
        };
        let restore = |buffers: &[cuteafd_ffi::CuteafdDeviceBuffer], saved: &[Vec<u8>]| -> Result<()> {
            self.drain_state()?;
            for (&buffer, bytes) in buffers.iter().zip(saved) {
                Device { library: self.library, id: buffer.device_id }.run(|| {
                    self.library.copy_h2d(buffer, bytes)
                })?;
            }
            Ok(())
        };
        let mut cases: Vec<(usize, usize, usize, bool)> = vec![
            (32, 1, 1, false), (32, 5, 1, false), (32, 9, 1, false),
            (32, 1, 3, true), (32, 1, 5, true), (32, 1, 9, true),
            (32, 1, 17, true), (32, 1, 25, true), (32, 1, 33, true), (32, 1, 37, true),
        ];
        if tokens.len() >= 2140 && self.max_context >= 2140 && self.pages >= 40 {
            cases.extend([(2100, 1, 1, false), (2100, 1, 5, true), (2100, 1, 13, true)]);
        }
        for (prefill, sequences, width, spec) in cases {
            let mut allocator = Allocator::new(self.pages, self.slots, &self.cfg);
            let mut placements: Vec<_> = (0..sequences).map(|_| allocator.admit(prefill + width + 1)).collect::<Result<_>>()?;
            for placement in &mut placements {
                self.prefill_device(placement, &tokens[..prefill], None, None, 1)?;
            }
            let original = placements.clone();
            let mut buffers = self.persistent_state_buffers();
            for [kv, keys, pools] in self.paged_buffers() {
                buffers.extend([kv, keys, pools]);
            }
            let before = snapshot(&buffers)?;
            let input = &tokens[prefill..prefill + width];
            let mut groups: Vec<_> = placements.iter_mut().map(|p| (p, input)).collect();
            let eager = self.verify_device_ungraphed(&mut groups, spec)?
                .context("Qwen graph check eager logits")?.to_host(self.library)?;
            let eager_state = snapshot(&buffers)?;
            // Run graphed twice from the identical restored state: first pass captures
            // (or donor-updates at the W1 soft cap), second pass replays the cached graph.
            let graph_hits = || crate::shared::decode_graph::device::snapshots().iter()
                .flat_map(|snapshot| snapshot.cycles.values()).map(|counters| counters.hits).sum::<u64>();
            let mut pass1_hits = 0u64;
            for pass in 0..2 {
                restore(&buffers, &before)?;
                placements = original.clone();
                if pass == 1 {
                    pass1_hits = graph_hits();
                }
                let mut groups: Vec<_> = placements.iter_mut().map(|p| (p, input)).collect();
                let logits = self.verify_device(&mut groups, spec)?.context("Qwen graph check logits")?;
                let graphed = logits.to_host(self.library)?;
                let rows = sequences * width;
                let (ids, statuses) = logits.greedy.context("Qwen graphed greedy selection missing")?;
                let device = self.library.cuda_get_device()?;
                let mut selected = vec![0u8; rows * 4];
                let mut status = vec![0u8; rows * 4];
                for (ptr, bytes) in [(ids, &mut selected), (statuses, &mut status)] {
                    self.library.copy_d2h(bytes, cuteafd_ffi::CuteafdDeviceBuffer {
                        ptr: ptr.cast_mut(), bytes: rows * 4, device_id: device, ..Default::default()
                    })?;
                }
                ensure!(status.iter().all(|&byte| byte == 0), "Qwen graphed greedy selection/status failed");
                for (row, id) in selected.chunks_exact(4).enumerate() {
                    let values = &graphed[row * self.cfg.vocab_size..][..self.cfg.vocab_size];
                    let expected = (1..values.len()).fold(0, |best, i| if values[i] > values[best] { i } else { best });
                    ensure!(u32::from_le_bytes(id.try_into()?) as usize == expected, "Qwen graphed greedy id differs");
                }
                ensure!(eager.len() == graphed.len() && eager.iter().zip(&graphed).all(|(a, b)| a.to_bits() == b.to_bits()),
                    "Qwen exact-row graph (prefill={prefill}, rows={rows}, spec={spec}, pass={pass}) changed logits");
                ensure!(snapshot(&buffers)? == eager_state,
                    "Qwen exact-row graph (prefill={prefill}, rows={rows}, spec={spec}, pass={pass}) changed persistent cache/state bytes");
            }
            ensure!(graph_hits() > pass1_hits,
                "Qwen exact-row graph check did not replay graphs for prefill={prefill} rows={} spec={spec}",
                sequences * width);
            println!("qwen4 exact-row graph check (prefill={prefill}, rows={}, spec={spec}): 0 state bytes differ, logits bit-identical, graphs={}, hits={}",
                sequences * width, self.captured_graphs(), graph_hits());
        }
        self.drain_state()
    }

    /// Enable lazy graph capture/replay in golden wide-batch verification without
    /// running the startup prewarm pass over unrelated narrow shapes.
    pub(super) fn enable_lazy_graphs_for_golden(&self) -> Result<()> {
        if !self.use_graphs { return Ok(()); }
        ensure!(self.graph_capture.get(), "Qwen graphs must be configured before enabling lazy capture");
        for bank in &self.gdn_banks {
            crate::shared::decode_graph::device::observe_device(self.library, bank.device.id)?;
        }
        self.graph_warmed.set(true);
        Ok(())
    }
    /// Logical record traffic, not measured DRAM traffic (GQA reuse and L2 can reduce it).
    /// Called only for the opt-in per-cycle trace, after the device work completes.
    pub(crate) fn verify_attention_trace(&self, contexts: &[Vec<usize>], rows: usize) -> crate::shared::speculation_trace::Deferred {
        let full_layers = self.weights.layers.iter().filter(|l| l.attention == Qwen4Attention::Full).count();
        let records: usize = contexts.iter().flatten().map(|&n| n.min(SPARSE_TOPK)).sum();
        let hint = self.attention_plan.rows();
        let splits = if hint == 1 { 64 } else if hint <= 4 { 32 } else { 16 };
        let program = self.kv_stem("sparse_gqa");
        let record_bytes = self.kv_record_bytes;
        let cap = self.decode_cap();
        crate::shared::speculation_trace::Deferred::new(move || serde_json::json!({"program": format!("qwen4_{program}_{cap}"),
            "route": "selected_split", "physical_rows": rows, "splits": splits,
            "qk_tile": [16, 32, 16], "pv_tile": [16, 256, 16], "kv_warps": 2,
            "record_bytes": record_bytes, "target_full_layers": full_layers,
            "logical_records_per_layer_upper_bound": records,
            "logical_record_bytes_upper_bound": records as u64 * record_bytes as u64 * full_layers as u64,
            "traffic_scope": "target verify; exact for dense contexts, upper bound for sparse; excludes index/query, padding and MTP draft; not DRAM bytes"}))
    }

    pub fn captured_graphs(&self) -> usize {
        self.graphs.borrow().iter().map(|bank| bank.len()).sum()
    }

    pub fn experts(&self) -> Option<&Experts<'a>> {
        self.experts.as_ref()
    }

    /// Per MoE layer, the weights a decode step reads after its routed
    /// experts, in read order: the next layer's attention site, attention
    /// (the E4M3 weights and scales of FP8-only projections), MLP site,
    /// router and shared expert; after the last layer the mixer and LM head.
    pub fn decode_read_order(&self) -> Vec<Vec<crate::shared::l2_prefetch::Range>> {
        let layers = &self.weights.layers;
        (0..layers.len()).map(|i| match layers.get(i + 1) {
            Some(next) => {
                let attention: &[&str] = match next.attention {
                    Qwen4Attention::Gdn => &["w_in", "conv_w", "a_log", "dt_bias", "norm_w", "w_out"],
                    Qwen4Attention::Full => &["w_in", "q_norm", "k_norm", "iq_norm", "ik_norm", "w_o"],
                };
                let names: Vec<&str> = ["attn.norm", "attn.w_di", "attn.w_up"].iter().chain(attention)
                    .chain(&["mlp.norm", "mlp.w_di", "mlp.w_up", "gate", "shared.w_gate_up", "shared.w_down"])
                    .copied().collect();
                crate::shared::l2_prefetch::operands(&names, |n| next.range(n))
            }
            None => self.weights.mixer.iter().chain(self.weights.head.allocations())
                .map(|a| (a.buffer.ptr.cast_const(), a.buffer.bytes)).collect(),
        }).collect()
    }

    /// After layer `index`'s shared expert is queued in a decode step: the
    /// L2 prefetch of what the step reads next; with local experts only under
    /// CUTEAFD_EMULATE_EXCHANGE_US (benchmarks), with a Spark-like wait.
    fn exchange_window(&self, index: usize, decode: bool, local: bool) -> Result<()> {
        if !decode {
            return Ok(());
        }
        let mark = if local { crate::shared::l2_prefetch::exchange_mark(self.library, self.execution_stream())? } else { None };
        if local && mark.is_none() {
            return Ok(());
        }
        if let Some(l2) = &self.l2 {
            l2.issue(self.library, index, self.execution_stream())?;
        }
        crate::shared::l2_prefetch::exchange_wait(self.library, mark)
    }

    /// Before a sequence's first step: zeroes its state slot.
    pub(crate) fn start(&self, placement: &Qwen4Placement) -> Result<()> {
        self.reset_slot(placement.slot)
    }

    /// Every full-attention layer's paged buffers (the MTP layer's last, when loaded): K/V
    /// records (2048 B per row) and raw index keys (256 B per row), both in 64-row record pages,
    /// and pooled block keys (256 B per block, 64 blocks per pool page).
    pub(crate) fn paged_buffers(&self) -> Vec<[cuteafd_ffi::CuteafdDeviceBuffer; 3]> {
        let mut out: Vec<_> = self.kv.iter().zip(&self.index).filter_map(|(kv, index)| match (kv, index) {
            (Some(kv), Some((keys, pools))) => Some([kv.buffer, keys.buffer, pools.buffer]),
            _ => None,
        }).collect();
        if let Some((kv, keys, pools)) = &self.mtp_kv {
            out.push([kv.buffer, keys.buffer, pools.buffer]);
        }
        out
    }

    pub(crate) fn snapshot_owners(&self) -> Vec<Rc<Allocation<'a>>> {
        let mut owners = Vec::new();
        for (kv, index) in self.kv.iter().zip(&self.index) {
            if let (Some(kv), Some((keys, pools))) = (kv, index) {
                owners.extend([kv.clone(), keys.clone(), pools.clone()]);
            }
        }
        if let Some((kv, keys, pools)) = &self.mtp_kv {
            owners.extend([kv.clone(), keys.clone(), pools.clone()]);
        }
        for bank in &self.gdn_banks {
            owners.extend([&bank.conv, &bank.state].into_iter().filter_map(|p| p.as_ref().cloned()));
        }
        owners.extend(self.ple_state.as_ref().cloned());
        owners.extend(self.mtp_pending.as_ref().cloned());
        owners
    }

    /// Every execution owner's (device, stream): the streams that read and write its KV pages
    /// and GDN state (prefix copies fence against them).
    pub(crate) fn owner_streams(&self) -> Vec<(i32, *mut c_void)> {
        self.gdn_banks.iter().map(|bank| (bank.device.id, bank.stream)).collect()
    }

    pub(crate) fn state_stream(&self, device: i32) -> Result<*mut c_void> {
        self.gdn_banks.iter().find(|bank| bank.device.id == device)
            .map(|bank| bank.stream).context("Qwen snapshot device without an execution stream")
    }

    pub(crate) fn require_live(&self) -> Result<()> {
        use crate::shared::peer_split::{TerminalPeerError, TerminalState};
        if self.terminal.get() != TerminalState::Active {
            return Err(TerminalPeerError::Closed(self.terminal.get()).into());
        }
        if let Some(Experts::Hybrid(experts)) = &self.experts {
            experts.with_peer(|peer| peer.require_live())?;
        }
        Ok(())
    }

    fn shutdown_submission(&self) -> Result<()> {
        use crate::shared::{experts::rtx::pair::{teardown, Teardown}, peer_split::TerminalState};
        if self.terminal.get() == TerminalState::Drained { return Ok(()); }
        self.terminal.set(TerminalState::Aborting);
        let result = teardown(&mut Teardown {
            abort: || {
                if let Some(Experts::Hybrid(experts)) = &self.experts {
                    experts.with_peer(|peer| {
                        if peer.terminal_state() != TerminalState::Aborting {
                            let _scope = self.gdn_banks[0].device.enter()?;
                            peer.publish_abort()?;
                        }
                        Ok(())
                    })?;
                }
                Ok(())
            },
            ranks: || self.drain_state(),
            links: || {
                if let Some(Experts::Hybrid(experts)) = &self.experts { for link in &experts.transport { link.try_borrow_mut()?.terminal_quiesce()?; } }
                Ok(())
            },
            copies: || {
                if let Some(Experts::Hybrid(experts)) = &self.experts { for link in &experts.transport { link.try_borrow()?.terminal_drain_copies()?; } }
                Ok(())
            },
            graphs: || {
                if let Some(Experts::Hybrid(experts)) = &self.experts { for link in &experts.transport { link.try_borrow_mut()?.terminal_release()?; } }
                // The ranks above drained every consumer; captured storage is persistent engine storage.
                for bank in self.graphs.borrow_mut().iter_mut() { bank.cache_clear_drained(); }
                Ok(())
            },
            finish: |drained| {
                self.terminal.set(if drained { TerminalState::Drained } else { TerminalState::Retained });
                if let Some(Experts::Hybrid(experts)) = &self.experts { let _ = experts.with_peer(|peer| { peer.finish_terminal(drained); Ok(()) }); }
            },
        });
        crate::shared::decode_graph::fatal_drain(result, "Qwen submission shutdown"); Ok(())
    }

    /// Request-local cancellation (#1817, `pair::submit_recoverable`). A
    /// host-gather timeout is raised at its consumer before that consumer
    /// queued anything, but earlier layers of the step are already queued.
    /// With one owner and coordinator-local experts they are plain stream work:
    /// drop the step's unconsumed table tickets, drain the execution stream,
    /// then the table's upload stream (returning its copied slots), and
    /// observe the graph boundary a capture in the step may have opened.
    /// TP2 and Spark layouts hold peer or Spark waits that only the terminal
    /// abort words can release, so they refuse and retire as before (#1842).
    /// Packed metadata pressure is earlier: at most one ordered table copy
    /// has issued, and its upload fence precedes this stream's drain.
    fn cancel_submission(&self, kind: crate::shared::experts::rtx::pair::Recovery) -> Result<()> {
        use crate::shared::experts::rtx::pair::Recovery;
        if kind == Recovery::HostGather {
            ensure!(self.gdn_banks.len() == 1 && self.hops.is_none()
                && !self.has_sparks(),
                "Qwen host-gather recovery needs one owner without peer or Spark work in flight");
        }
        drop(self.ple_pending.borrow_mut().take());
        self.drain_state()?;
        if let Some(ple) = &self.ple { ple.mapped().drain()?; }
        // An early return skipped the step's own cursor resets; no later step may
        // inherit a packed member view, unit slot offset or stale MTP terminal rows.
        for slot in [&self.workspace, &self.mtp_prefill_workspace, &self.decode_workspace,
            &self.peer_workspace, &self.peer_decode_workspace] {
            if let Some(w) = slot.try_borrow()?.as_ref() {
                w.table_view.set(None); w.row_offset.set(0); w.logit_offset.set(0); w.position.set(0);
                w.units.set(false); w.terminal_valid.set(false);
            }
        }
        self.last_stream_offset.set(0);
        if let Some(cycles) = self.graph_cycles.borrow_mut().as_mut() { cycles.abandon(); }
        Ok(())
    }

    pub(crate) fn drain_state(&self) -> Result<()> {
        let mut failures = Vec::new();
        for bank in &self.gdn_banks {
            // SAFETY: each bank owns its execution stream and live state allocations.
            if let Err(error) = bank.device.run(|| unsafe { self.library.cuda_stream_synchronize(bank.stream) }) {
                failures.push(format!("owner {}: {error:#}", bank.device.id));
            }
        }
        ensure!(failures.is_empty(), "{}", failures.join("; ")); Ok(())
    }

    /// A partial-model diagnostic has no logits readback to retire its rows.
    fn drain_partial_step(&self) -> Result<()> {
        drop(self.ple_pending.borrow_mut().take());
        self.drain_state()?;
        Ok(())
    }

    /// `bytes` at `offset` inside `pool`.
    fn region(pool: &Dev<'_>, offset: usize, bytes: usize) -> cuteafd_ffi::CuteafdDeviceBuffer {
        debug_assert!(offset + bytes <= pool.buffer.bytes);
        cuteafd_ffi::CuteafdDeviceBuffer {
            // SAFETY: callers pass offsets inside the pool.
            ptr: unsafe { pool.buffer.ptr.cast::<u8>().add(offset) }.cast(),
            bytes,
            ..pool.buffer
        }
    }

    /// Every per-slot state region of `slot` (GDN conv + recurrent state per
    /// GDN layer, PLE conv state).
    pub(crate) fn slot_regions(&self, slot: usize) -> Vec<cuteafd_ffi::CuteafdDeviceBuffer> {
        let mut regions = Vec::new();
        for layer in &self.state_map.layers {
            if let Some(ord) = layer.gdn_ordinal {
                let bank = &self.gdn_banks[layer.owner];
                regions.extend(bank.slot_regions(ord, slot));
            }
        }
        if let Some(ple) = &self.ple_state {
            let per = ple.buffer.bytes / self.slots;
            regions.push(Self::region(ple, slot * per, per));
        }
        regions
    }

    pub(crate) fn snapshot_regions(&self, slot: usize) -> Vec<cuteafd_ffi::CuteafdDeviceBuffer> {
        let mut regions = self.slot_regions(slot);
        if let Some(pending) = &self.mtp_pending {
            let row = HC * self.cfg.hidden * 2;
            regions.push(Self::region(pending, slot * MTP_PENDING_ROWS * row, row));
        }
        regions
    }

    pub(crate) fn mtp_keep_last_pending(&self, slot: i32, last: usize) -> Result<()> {
        let pending = self.mtp_pending.as_ref().context("MTP stash absent")?;
        let slot = usize::try_from(slot)?;
        ensure!(slot < self.slots && last < MTP_PENDING_ROWS, "invalid MTP snapshot pending row");
        if last > 0 {
            let row = HC * self.cfg.hidden * 2;
            let dst = Self::region(pending, slot * MTP_PENDING_ROWS * row, row);
            let src = Self::region(pending, (slot * MTP_PENDING_ROWS + last) * row, row);
            // SAFETY: disjoint rows in retained stash storage, ordered after canonical forwards.
            unsafe { crate::shared::memory::device::copy_device_ordered(self.library, dst, src, row, self.execution_stream())?; }
        }
        Ok(())
    }

    fn persistent_state_buffers(&self) -> Vec<cuteafd_ffi::CuteafdDeviceBuffer> {
        let mut buffers: Vec<_> = self.gdn_banks.iter().flat_map(|bank|
            [&bank.conv, &bank.state].into_iter().filter_map(|pool| pool.as_ref().map(|p| p.buffer))).collect();
        buffers.extend(self.ple_state.as_ref().map(|p| p.buffer));
        buffers
    }

    /// Global GDN layer's conv/state/replay pointers inside its owner's bank.
    fn gdn_pools(&self, index: usize) -> Result<[*mut c_void; 3]> {
        let layer = self.state_map.layers.get(index).context("GDN global layer out of range")?;
        let ord = layer.gdn_ordinal.context("GDN layer without a state pool")?;
        let bank = &self.gdn_banks[layer.owner];
        bank.pools(ord)
    }

    /// Zeroes a sequence's state (before its first step).
    pub fn reset_slot(&self, slot: i32) -> Result<()> {
        let slot = usize::try_from(slot)?;
        ensure!(slot < self.slots, "state slot {slot} out of range");
        for region in self.slot_regions(slot) {
            Device { library: self.library, id: region.device_id }.run(||
                self.library.cuda_zero_bytes(region, region.bytes))?;
        }
        Ok(())
    }

    fn execution_stream(&self) -> *mut c_void {
        self.gdn_banks[self.active_owner.get()].stream
    }

    fn execution_programs(&self) -> &'a Programs<'a> {
        self.gdn_banks[self.active_owner.get()].programs
    }

    fn on_owner<T>(&self, owner: usize, work: impl FnOnce() -> Result<T>) -> Result<T> {
        let bank = self.gdn_banks.get(owner).context("Qwen execution owner out of range")?;
        // Single-device engines retain their startup CUDA device, as before hybrid
        // dispatch. Only multi-owner execution needs per-hook device switching.
        if self.gdn_banks.len() == 1 { return work(); }
        struct Restore<'c>(&'c Cell<usize>, usize);
        impl Drop for Restore<'_> {
            fn drop(&mut self) { self.0.set(self.1); }
        }
        let _restore = Restore(&self.active_owner, self.active_owner.replace(owner));
        bank.device.run(work)
    }

    fn workspace_slot(&self, owner: usize, decode: bool) -> Result<&RefCell<Option<Workspace<'a>>>> {
        match (owner, decode) {
            (0, false) => Ok(&self.workspace), (0, true) => Ok(&self.decode_workspace),
            (1, false) => Ok(&self.peer_workspace), (1, true) => Ok(&self.peer_decode_workspace),
            _ => anyhow::bail!("Qwen workspace owner out of range"),
        }
    }

    /// Per-step scratch of the executing step's workspace.
    fn alloc(&self, bytes: usize) -> Result<Dev<'a>> {
        self.alloc_tagged(bytes, Tag::workspace("qwen4/step-workspace"))
    }

    fn alloc_tagged(&self, bytes: usize, tag: Tag) -> Result<Dev<'a>> {
        Ok(Rc::new(Allocation::new(Device { library: self.library, id: self.library.cuda_get_device()? }, bytes.max(256), tag)?))
    }

    fn run(&self, name: &str, pointers: &[(&str, *mut c_void)], scalars: &[Scalar]) -> Result<()> {
        let names: Vec<&str> = pointers.iter().map(|(n, _)| *n).collect();
        let program = self.execution_programs().program(name, &names)?;
        let raw: Vec<*mut c_void> = pointers.iter().map(|(_, p)| *p).collect();
        // SAFETY: every pointer names a live allocation sized for the rows in
        // `scalars`; the stream orders all launches of this engine.
        unsafe { program.launch(&raw, scalars, self.execution_stream()) }.with_context(|| format!("{name} with {scalars:?}"))
    }

    fn scratch(&self, name: &str) -> Result<usize> {
        Ok(self.execution_programs().spec(name)?.scratch.get("scratch").copied().unwrap_or(0) as usize)
    }

    fn workspace(&self, t: usize, decode: bool, logit_rows: usize) -> Result<Workspace<'a>> {
        self.workspace_rows(t, decode, logit_rows, t, 0, 0)
    }

    fn workspace_rows(&self, t: usize, decode: bool, logit_rows: usize, span: usize, terminal_rows: usize, units: usize)
        -> Result<Workspace<'a>> {
        self.allow_workspace_creation()?;
        let _memory_scope = cuteafd_ffi::memory_ledger::scope("workspace");
        let h = self.cfg.hidden;
        let cap = if decode { self.decode_cap() } else { "m4096" };
        let ple = if self.ple.as_ref().is_some_and(|p| p.fp8) { "qwen4_ple_fp8" } else { "qwen4_ple_bf16" };
        let mut scratch = 0;
        for name in ["qwen4_hc_pre".to_string(), "qwen4_hc_post_pre".into(), "qwen4_head".into(),
            "qwen4_shared".into(), ple.into(), "qwen4_mtp_feedback".into(), format!("qwen4_gdn_{cap}"),
            format!("qwen4_{}_{cap}", self.kv_stem("attn_producer")), format!("qwen4_gdn_w8_{cap}"),
            format!("qwen4_{}_{cap}", self.kv_stem("attn_producer_w8")), format!("qwen4_attn_o_w8_{cap}"),
            format!("qwen4_{}_{cap}", self.kv_stem("sparse_gqa")), format!("qwen4_attn_o_{cap}")] {
            if let Ok(bytes) = self.scratch(&name) {
                scratch = usize::max(scratch, bytes);
            }
        }
        let topk_scratch = self.scratch(&format!("qwen4_index_topk_{cap}")).unwrap_or(0);
        let blocks = self.cfg.index_budget / BLOCK;
        let head_workspace = self.alloc(VOCABULARY_HEAD_WORKSPACE)?;
        let tables = super::admission::StepTableBytes::new(t, decode, self.pages, self.pool_pages, self.cfg.ple_rows())?;
        let host = if !decode && self.prefill_shape.is_some() && units == 0 {
            super::admission::HostWorkspaceBytes::mtp(&self.cfg, t, self.pages, self.pool_pages, self.prefill_rows.div_ceil(t))?
        } else {
            super::admission::HostWorkspaceBytes::new(&self.cfg, t, decode, self.pages, self.pool_pages,
                units, self.has_sparks(), self.weights.mtp.is_some(),
                self.ple.as_ref().map(|ple| ple.mapped()).is_none(), logit_rows, self.gdn_banks.len() == 1)?
        };
        let (topk, heads, hd) = (self.cfg.topk, self.cfg.heads, self.cfg.head_dim);
        Ok(Workspace {
            rows: t, slot_rows: span, row_offset: Cell::new(0), position: Cell::new(0), units: Cell::new(false),
            streams: [self.alloc(span * HC * h * 2)?, self.alloc(span * HC * h * 2)?],
            inject: self.alloc(span * HC * 2)?,
            x: self.alloc(span * h * 2)?,
            delta: self.alloc(span * h * 2)?,
            shared: self.alloc(span * h * 2)?,
            routed: self.alloc(t * h * 2)?,
            positions: self.alloc(tables.positions)?,
            rope_positions: self.alloc(tables.rope_positions)?,
            block_rope_positions: self.alloc(tables.block_rope_positions)?,
            kv_slots: self.alloc(tables.kv_slots)?,
            slots: self.alloc(tables.slots)?,
            seq_first: self.alloc(tables.seq_first)?,
            local_seq_first: if decode { None } else {
                let zero = self.alloc(tables.seq_first)?;
                self.library.cuda_zero_bytes(zero.buffer, zero.buffer.bytes)?;
                Some(zero)
            },
            table_view: Cell::new(None),
            pool_slots: self.alloc(tables.pool_slots)?,
            cache_lengths: self.alloc(tables.cache_lengths)?,
            page_table: self.alloc(tables.page_table)?,
            pool_table: self.alloc(tables.pool_table)?,
            ple_ids: self.alloc(tables.ple_ids)?,
            ple_rows: match self.ple.as_ref() {
                Some(ple) => {
                    let n = t * self.cfg.ple_rows();
                    let local = self.alloc(n * 8)?;
                    self.put(&local, &(0..n as i64).collect::<Vec<i64>>())?;
                    Some((self.alloc_tagged(n * ple.row_bytes, Tag::staging("qwen4/ple-gather-rows"))?, local))
                }
                None => None,
            },
            query: self.alloc(t * heads * hd * 2)?,
            gate: self.alloc(t * heads * hd * 2)?,
            index_q: self.alloc(t * self.cfg.index_heads * INDEX_DIM * 2)?,
            attn: self.alloc(t * heads * hd * 2)?,
            blocks: self.alloc(t * blocks * 4)?,
            indices: self.alloc(t * SPARSE_TOPK * 4)?,
            lengths: self.alloc(t * 4)?,
            scratch: self.alloc(scratch)?,
            topk_scratch: {
                let zero = self.alloc(topk_scratch)?;
                self.library.cuda_zero_bytes(zero.buffer, zero.buffer.bytes)?;
                zero
            },
            logits: self.alloc(logit_rows * self.cfg.vocab_size * 4)?,
            logit_rows, logit_offset: Cell::new(0),
            terminal_streams: (terminal_rows > 0).then(|| self.alloc(terminal_rows * HC * h * 2)).transpose()?,
            terminal_valid: Cell::new(false),
            router_logits: self.alloc(t * self.cfg.experts * 4)?,
            route_ids: self.alloc(t * topk * 4)?,
            route_weights: self.alloc(t * topk * 4)?,
            wire: self.alloc(t * (h + h / 32))?,
            hidden_rows: self.alloc(t * 4)?,
            argmax: self.alloc(logit_rows * 8)?,
            ids: self.alloc(tables.ids)?,
            select: self.alloc(logit_rows * 8)?,
            logits_host: RefCell::new(HostAllocation::new(self.library,
                usize::try_from(host.logits)?, Tag::staging("qwen4/logits-host"))?),
            staging: RefCell::new(crate::shared::token_table::RetainedUploads::new(
                Device { library: self.library, id: self.library.cuda_get_device()? }, host.uploads)?),
            // Target and MTP logits return to owner zero; peer workspaces never project.
            // SAFETY: the workspace buffer lives in the same struct and drops after the head.
            head: if self.active_owner.get() == 0 && matches!(self.weights.head, Qwen4Head::Bf16(_)) {
                Some(unsafe { self.library.vocabulary_head_rows(head_workspace.buffer.ptr, h as u32, logit_rows as u32,
                    self.cfg.vocab_size as u32, crate::shared::target_head::TargetHead::configured().native_mode())? })
            } else { None },
            _head_workspace: head_workspace,
        })
    }

    fn put<T: Copy>(&self, dev: &Dev<'_>, values: &[T]) -> Result<()> {
        let bytes = bytes_of(values);
        ensure!(bytes.len() <= dev.buffer.bytes, "table exceeds its buffer");
        if bytes.is_empty() {
            return Ok(());
        }
        self.library.copy_h2d(cuteafd_ffi::CuteafdDeviceBuffer { bytes: bytes.len(), ..dev.buffer }, bytes)
    }

    /// Stage the step's ids for a device gather from either table placement.
    /// With `defer_gather`, the decode graph's first segment gathers the rows.
    fn stage_embedding(&self, w: &Workspace<'_>, tokens: &[u32], copies: usize, out: &Dev<'_>, defer_gather: bool)
        -> Result<()> {
        self.embedding.check(tokens)?;
        self.stage_table(w, &w.ids, tokens)?;
        if !defer_gather {
            // SAFETY: the ids are staged on this stream; `out` holds the rows.
            unsafe { self.embedding.gather(w.ids.buffer.ptr, std::ptr::null(), tokens.len(), copies,
                std::ptr::null(), w.live(out, copies * self.cfg.hidden * 2), None, crate::shared::peer_split::RankDevice {
                    device: self.gdn_banks[self.active_owner.get()].device.id, stream: self.execution_stream() })? };
        }
        Ok(())
    }

    /// Queues `bytes` into `dst` through the workspace's pinned staging.
    fn stage(&self, w: &Workspace<'_>, dst: cuteafd_ffi::CuteafdDeviceBuffer, bytes: &[u8]) -> Result<()> {
        // SAFETY: this workspace and all table readers share the execution
        // stream; the arena retains each immutable source through its read.
        unsafe { w.staging.borrow_mut().upload(dst, bytes, self.execution_stream()) }
    }

    fn stage_table<T: Copy>(&self, w: &Workspace<'_>, dev: &Dev<'_>, values: &[T]) -> Result<()> {
        self.stage(w, dev.buffer, bytes_of(values))
    }

    fn download(&self, dev: &Dev<'_>, bytes: usize) -> Result<Vec<u8>> {
        // SAFETY: the engine owns this stream.
        unsafe { self.library.cuda_stream_synchronize(self.execution_stream())? };
        let mut out = vec![0u8; bytes];
        self.library.copy_d2h(&mut out, cuteafd_ffi::CuteafdDeviceBuffer { bytes, ..dev.buffer })?;
        Ok(out)
    }

    /// Read the PLE rows of `tokens[resume..]` (after `history`, the n-gram
    /// history at `resume`) into the table's row cache ahead of its gathers,
    /// in the units the prefill runs: each scheduler chunk (ends `chunk_ends`)
    /// cut as `prefill_units` cuts it, `waves` deep. None without a mapped table.
    pub fn read_ahead_ple(&self, history: &NgramHistory, tokens: &[u32], resume: usize, chunk_ends: &[usize])
        -> Option<cuteafd_loader::LookaheadHandle> {
        let (ple, mapped) = self.ple.as_ref().map(|p| (p, p.mapped()))?;
        if resume >= tokens.len() { return None; }
        let units = match self.ple_units(resume, tokens.len(), chunk_ends) {
            Ok(units) => units,
            Err(error) => { tracing::warn!("PLE look-ahead not started: {error:#}"); return None; }
        };
        let (hasher, mut history) = (ple.hasher.clone(), history.clone());
        let source = cuteafd_loader::formats::mapped_table::token_table::HashedSource {
            tokens: tokens[resume..].to_vec(),
            hash: move |tokens: &[u32], rows: &mut Vec<u64>| {
                let mut ids = Vec::with_capacity(tokens.len() * hasher.heads());
                hasher.hash(&mut history, tokens, &mut ids)?;
                rows.extend(ids.into_iter().map(|id| id as u64));
                Ok(())
            },
        };
        Some(mapped.start_lookahead(self.request_id.get(), resume as u64, Box::new(source), units))
    }

    /// The prefill units of `tokens[resume..end]` (absolute ends) and their depth.
    fn ple_units(&self, resume: usize, end: usize, chunk_ends: &[usize]) -> Result<cuteafd_loader::TableUnitPlan> {
        let (unit_rows, waves) = self.prefill_shape.map_or((self.prefill_rows, 1),
            |shape| (shape.unit_rows as usize, shape.waves as usize));
        let (mut ends, mut calls) = (Vec::new(), Vec::new());
        let mut at = resume;
        for &chunk_end in chunk_ends.iter().filter(|&&e| e > resume).chain(std::iter::once(&end)) {
            if chunk_end <= at { continue; }
            // Each scheduler chunk is one prefill call.
            calls.push(ends.len());
            if self.prefill_shape.is_some() {
                let units = crate::shared::prefill_pipeline::sequence_units(chunk_end - at, unit_rows, 1, 1)?;
                ends.extend(units.iter().map(|unit| (at + unit.rows.end) as u64));
            } else { ends.push(chunk_end as u64); }
            at = chunk_end;
        }
        Ok(cuteafd_loader::TableUnitPlan { ends, depth: waves.max(1), calls })
    }

    /// Read the PLE rows `tokens` add after `history` into the row cache now.
    pub fn prefetch_ple_decode(&self, history: &[u32], tokens: &[u32]) {
        let Some((ple, mapped)) = self.ple.as_ref().map(|p| (p, p.mapped())) else { return };
        if tokens.is_empty() { return; }
        let mut ids = Vec::with_capacity(tokens.len() * self.cfg.ple_rows());
        if let Err(error) = ple.hasher.hash(&mut history_of(&self.cfg, history), tokens, &mut ids)
            .and_then(|()| mapped.prefetch_decode(&ids)) {
            tracing::warn!("PLE decode prefetch: {error:#}");
        }
    }

    /// Logs the mapped PLE table's stats since the previous call.
    pub fn log_table_stats(&self, event: &str) {
        if let Some(mapped) = self.ple.as_ref().map(|p| p.mapped()) {
            mapped.log_interval(event);
        }
    }

    /// Per-row positions, record and pool slots, blocks seen, PLE rows.
    fn rows(&self, placement: &mut Qwen4Placement, tokens: &[u32], first: i32, tables: &mut StepTables) -> Result<()> {
        let start = placement.len;
        for (i, _) in tokens.iter().enumerate() {
            let position = start + i;
            ensure!(position < self.max_context, "position {position} past the context {}", self.max_context);
            tables.positions.push(position as i64);
            tables.rope_positions.push(placement.rope.at(position)?);
            tables.block_rope_positions.push(placement.rope.at(position - position % BLOCK)?);
            tables.kv_slots.push(placement.record(position)?);
            tables.pool_slots.push(placement.pool_slot(position)?);
            tables.slots.push(placement.slot);
            tables.seq_first.push(first);
            tables.cache_lengths.push(((position + 1) / BLOCK) as i32);
            tables.long |= position + 1 > self.cfg.dense_context();
            tables.pool_width = tables.pool_width.max((position + 1).div_ceil(POOL_PAGE_TOKENS));
        }
        if let Some(ple) = &self.ple {
            // These are native embedding ids, including image placeholders, not
            // prefix-cache radix keys. The reference uses ple_input_ids=input_ids.
            ple.hasher.hash(&mut placement.history, tokens, &mut tables.ple_ids)?;
        }
        Ok(())
    }

    /// Prefills a sequence from its length through every resident layer and
    /// returns the logits of the last `logit_rows` rows when all layers are
    /// resident. `on_layer` receives each layer's output streams (BF16 [t, 4,
    /// H]); with `forced`, `forced(l)` (when it returns rows) replaces the
    /// streams after layer `l`, so each layer's comparison measures that layer alone.
    pub fn prefill_forced(&self, placement: &mut Qwen4Placement, tokens: &[u32], on_layer: LayerHook<'_>,
        forced: Forced<'_>, logit_rows: usize) -> Result<Option<Vec<f32>>> {
        self.prefill_device(placement, tokens, on_layer, forced, logit_rows)?
            .map(|logits| logits.to_host(self.library)).transpose()
    }

    /// [`Self::prefill_forced`] leaving the logits on the device.
    pub fn prefill_device(&self, placement: &mut Qwen4Placement, tokens: &[u32], on_layer: LayerHook<'_>,
        forced: Forced<'_>, logit_rows: usize) -> Result<Option<DeviceLogits>> {
        if self.prefill_shape.is_some() && on_layer.is_none() && forced.is_none() {
            return crate::shared::experts::rtx::pair::submit_recoverable(|| self.require_live(), |kind| self.cancel_submission(kind), || self.shutdown_submission(),
                |_| self.prefill_units(placement, tokens, logit_rows));
        }
        self.last_stream_offset.set(0);
        let (t, start) = (tokens.len(), placement.len);
        ensure!(t > 0 && t <= self.prefill_rows.min(4096) && start + t <= self.max_context, "prefill of {t} rows at {start}");
        if start == 0 {
            self.start(placement)?;
        }
        let mut tables = StepTables { page_table: placement.pages.clone(), pool_table: placement.pool_pages.clone(),
            page_stride: 0, pool_stride: 0, page_width: placement.pages.len(), ..Default::default() };
        self.rows(placement, tokens, 0, &mut tables)?;
        let media = embedding_media(&[(placement, start, t)])?;
        let logits = self.step(&Arc::new(tables), tokens, logit_rows.clamp(1, t), on_layer, forced, &media, true)?;
        placement.len += t;
        placement.state_len = placement.len;
        Ok(logits)
    }

    pub(crate) fn prefill_scoring(&self, placement: &mut Qwen4Placement, tokens: &[u32], logit_rows: usize,
        position: usize, probe: &crate::shared::probe::ProbeRef)
        -> Result<Option<crate::shared::probe::ScoreLogits>> {
        if logit_rows <= 1 {
            return Ok(self.prefill_device(placement, tokens, None, None, logit_rows)?
                .map(crate::shared::probe::ScoreLogits::Device));
        }
        self.prefill_consumer(placement, tokens, None, None, logit_rows,
            crate::shared::probe::ScoringRows { consumer: probe, position })
    }

    /// Preserve diagnostic layer hooks and transformer cuts while consuming bounded head tiles.
    pub(crate) fn prefill_consumer(&self, placement: &mut Qwen4Placement, tokens: &[u32],
        on_layer: LayerHook<'_>, forced: Forced<'_>, logit_rows: usize,
        scoring: crate::shared::probe::ScoringRows<'_>) -> Result<Option<crate::shared::probe::ScoreLogits>> {
        ensure!(logit_rows > 0 && logit_rows <= tokens.len(), "Qwen scoring logits extent");
        crate::shared::experts::rtx::pair::submit_recoverable(|| self.require_live(), |kind| self.cancel_submission(kind), || self.shutdown_submission(), |_| {
            if self.prefill_shape.is_some() && on_layer.is_none() && forced.is_none() {
                if self.prefill_units_with(placement, tokens, logit_rows, Some(scoring))?.is_none() { return Ok(None); }
            } else {
                if self.prefill_device(placement, tokens, on_layer, forced, 1)?.is_none() { return Ok(None); }
                let workspace = self.workspace.borrow();
                let w = workspace.as_ref().context("Qwen scoring workspace")?;
                let (_, cur) = self.last_streams.get();
                self.on_owner(0, || self.record_head(w, &w.streams[cur], tokens.len(), logit_rows, scoring))?;
            }
            Ok(Some(crate::shared::probe::ScoreLogits::Recorded))
        })
    }

    fn prefill_units(&self, placement: &mut Qwen4Placement, tokens: &[u32], logit_rows: usize)
        -> Result<Option<DeviceLogits>> {
        self.prefill_units_with(placement, tokens, logit_rows, None)
    }

    fn prefill_units_with(&self, placement: &mut Qwen4Placement, tokens: &[u32], logit_rows: usize,
        scoring: Option<crate::shared::probe::ScoringRows<'_>>) -> Result<Option<DeviceLogits>> {
        use crate::shared::prefill_pipeline::{sequence_units, run_prefill_units};
        let shape = self.prefill_shape.context("Qwen prefill unit shape")?.validate()?;
        ensure!(self.experts.is_some(), "Qwen prefill units need admitted experts");
        let (t, start) = (tokens.len(), placement.len);
        validate_prefill_span(t, start, self.prefill_rows, self.max_context)?;
        let logit_rows = logit_rows.clamp(1, t);
        if start == 0 { self.start(placement)?; }
        let mut tables = StepTables { page_table: placement.pages.clone(), pool_table: placement.pool_pages.clone(),
            page_width: placement.pages.len(), ..Default::default() };
        self.rows(placement, tokens, 0, &mut tables)?;
        let waves = shape.waves as usize;
        // Keep the cuts independent of wave count for same-cut identity checks.
        let mut units = sequence_units(t, shape.unit_rows as usize, 1, self.weights.layers.len())?;
        for unit in &mut units { unit.slot = unit.index % waves; }
        let retained_rows = if scoring.is_some() { 1 } else { logit_rows };
        for owner in 0..self.gdn_banks.len() {
            self.on_owner(owner, || {
                let slot = self.workspace_slot(owner, false)?;
                let replace = slot.borrow().as_ref().is_none_or(|w| w.rows != shape.unit_rows as usize
                    || w.slot_rows != shape.slot_rows() as usize || w.logit_rows < retained_rows);
                if replace {
                    self.allow_workspace_creation()?;
                    self.drain_state()?;
                    *slot.borrow_mut() = Some(self.workspace_rows(shape.unit_rows as usize, false,
                        retained_rows, shape.slot_rows() as usize,
                        if owner == 0 && self.weights.mtp.is_some() { self.prefill_rows } else { 0 }, shape.waves as usize)?);
                }
                Ok(())
            })?;
        }
        self.prepare_prefill_units()?;
        let root = self.workspace.borrow();
        let peer = self.peer_workspace.borrow();
        let w = root.as_ref().context("Qwen prefill workspace")?;
        let work = [Some(w), peer.as_ref()];
        for w in work.iter().flatten() { w.terminal_valid.set(false); w.units.set(true); }
        let unit_tables: Arc<Vec<_>> = Arc::new(units.iter().map(|unit| tables.unit(unit.rows.clone(),
            if self.ple.is_some() { self.cfg.ple_rows() } else { 0 }, self.cfg.dense_context())).collect());
        let ple_batch = self.ple.as_ref().map(|ple| ple.mapped().batch(Arc::clone(&unit_tables)
            as Arc<dyn crate::shared::mapped_table::RowSource>));
        let mut links = match &self.experts {
            Some(Experts::Hybrid(experts)) => experts.transport.iter().map(RefCell::borrow_mut).collect::<Vec<_>>(),
            _ => Vec::new(),
        };
        let mut transports = Vec::new();
        for link in &mut links {
            let exchange = link.exchange()?;
            ensure!(exchange.fits(self.wave_staging_bytes(shape.unit_rows as usize)?),
                "Spark exchange staging is narrower than a prefill unit");
            transports.push(exchange);
        }
        let mut hooks = UnitHooks { engine: self, transports, work, placement, tokens,
            tables: unit_tables, ple_batch, current: vec![[0; 2]; units.len()],
            entered: vec![false; units.len()], pending: (0..units.len()).map(|_| None).collect(), logit_rows, scoring };
        let result = run_prefill_units(&mut hooks, &units, waves);
        for w in work.iter().flatten() {
            w.row_offset.set(0); w.logit_offset.set(0); w.position.set(0); w.units.set(false);
        }
        self.last_stream_offset.set(0);
        result?;
        w.terminal_valid.set(w.terminal_streams.is_some());
        placement.len += t;
        placement.state_len = placement.len;
        tracing::debug!(rows = t, unit_rows = shape.unit_rows, waves, units = units.len(), "Qwen prefill units");
        if self.weights.layers.len() < self.cfg.layers {
            self.drain_partial_step()?;
            return Ok(None);
        }
        Ok(Some(self.device_logits(w, retained_rows, false)))
    }

    pub(crate) fn configure_packing(&self, mode: crate::shared::prefill_packing::PackingMode) -> Result<()> {
        use crate::shared::prefill_packing::{PackingMode, log_mode};
        if mode == PackingMode::Unrestricted { self.packed_row_cap.set(None); return Ok(()); }
        let rows = self.prefill_shape.map_or(self.prefill_rows, |s| s.unit_rows as usize);
        let cap = match &self.experts {
            Some(Experts::Hybrid(experts)) if experts.local.is_none() => self.spark_link(0)
                .context("Qwen remote grouping")?.borrow().installed_grouping()?.iter()
                .map(|report| report.packed_row_cap(self.spark_bf16)).min().unwrap_or(0).min(rows),
            Some(Experts::SharedOnly) => rows,
            // The shared RTX trait has no arithmetic-route boundary metadata.
            _ => 0,
        };
        let cap = cap.min(self.coordinator_packed_cap(rows)?);
        self.packed_row_cap.set(Some(cap));
        log_mode(mode, cap);
        Ok(())
    }

    fn coordinator_packed_cap(&self, rows: usize) -> Result<usize> {
        let inventory = super::programs::packed_inventory(
            self.programs.names().filter(|name| name.starts_with("qwen4_")))?;
        let cap = crate::shared::prefill_packing::coordinator_row_cap(self.programs, inventory, rows)?;
        Ok(super::programs::ple_packed_cap(cap, self.ple.is_some()))
    }

    pub(crate) fn packed_limits(&self) -> Option<crate::shared::prefill_packing::Limits> {
        if self.gdn_banks.len() != 1 || self.hops.is_some() { return None; }
        let rows = self.prefill_shape.map_or(self.prefill_rows, |s| s.unit_rows as usize);
        Some(crate::shared::prefill_packing::Limits {
            rows: self.packed_row_cap.get().unwrap_or(rows),
            chunk_rows: if self.weights.mtp.is_some() && self.prefill_shape.is_none() {
                MTP_PENDING_ROWS.min(rows)
            } else { rows },
            max_context: self.max_context, taps: None,
        })
    }

    pub(super) fn packed_table_budget(&self) -> crate::shared::prefill_packing::TableBudget {
        // Admission charges decode_rows full-pool metadata rows. A sequence can
        // use no more than one such row, regardless of shared physical pages.
        crate::shared::prefill_packing::TableBudget([self.decode_rows * self.pages, self.decode_rows * self.pool_pages])
    }

    pub(crate) fn prefill_packed(&self, sequences: &mut [crate::shared::qwen::scheduler::PrefillSeq<'_,
        Qwen4Placement, super::speculate::MtpSeq>],
        on_logits: &mut dyn FnMut(usize, DeviceLogits) -> crate::shared::prefill_packing::MemberOutcome)
        -> Result<Vec<Result<()>>> {
        // Capacity refusal is request-local. The scheduler applies these bounds
        // before forming the group; direct callers must also check before any
        // mutation, CUDA submission, or terminal guard.
        let limits = self.packed_limits().context("Qwen packing requires one attention owner")?;
        let segments = crate::shared::prefill_packing::plan(&sequences.iter()
            .map(|s| (s.start, s.tokens.len())).collect::<Vec<_>>(), &limits)?;
        let offsets = self.packed_table_budget().offsets(sequences.iter().map(|s| s.placement.packed_table_entries()))?;
        ensure!(sequences.iter().enumerate().all(|(i, s)| s.start == s.placement.len
            && !sequences[..i].iter().any(|p| p.placement.slot == s.placement.slot)),
            "Qwen packed frontier or state-slot alias");
        crate::shared::experts::rtx::pair::submit_recoverable(|| self.require_live(), |kind| self.cancel_submission(kind), || self.shutdown_submission(), |_| {
            ensure!(self.execution_stream() == self.gdn_banks[0].stream,
                "Qwen packed tables must share the decode execution stream");
            // A sole lane issues all readers and writers in this stream's order.
            // Exclusively borrow the decode owner through target + MTP submission;
            // verify cannot submit reentrantly from a head callback. Captured
            // pointers stay fixed and step_inner rewrites both planes before replay.
            let mut decode = self.decode_workspace.try_borrow_mut()?;
            if decode.is_none() { *decode = Some(self.workspace(self.decode_rows, true, self.decode_rows)?); }
            let decode = decode.as_ref().context("Qwen packed decode metadata owner")?;
            let (mut pages, mut pools) = (Vec::new(), Vec::new());
            for s in sequences.iter() {
                pages.extend_from_slice(&s.placement.pages);
                pools.extend_from_slice(&s.placement.pool_pages);
            }
            // Only the borrowed owner stages the page IDs. If the previous
            // sources have not retired, fail this pack before touching state,
            // PLE or Spark. A first successful upload can still be in flight.
            self.stage_table(decode, &decode.page_table, &pages).map_err(packed_upload_error)?;
            self.stage_table(decode, &decode.pool_table, &pools).map_err(packed_upload_error)?;
            let mut tables: Vec<StepTables> = Vec::with_capacity(sequences.len());
            for (i, s) in sequences.iter_mut().enumerate() {
                if s.start == 0 { self.start(s.placement)?; }
                let mut table = StepTables { page_width: s.placement.pages.len(), ..Default::default() };
                self.rows(s.placement, s.tokens, 0, &mut table)?;
                ensure!(table.positions.len() == segments[i].rows, "Qwen packed rows differ");
                tables.push(table);
            }
            let tokens = sequences.iter().flat_map(|s| s.tokens.iter().copied()).collect::<Vec<_>>();
            if self.prefill_shape.is_some() { self.prepare_prefill_units()?; }
            if self.workspace.borrow().is_none() { *self.workspace.borrow_mut() = Some(self.workspace(self.prefill_rows, false, 1)?); }
            let workspace = self.workspace.borrow();
            let w = workspace.as_ref().context("Qwen packed workspace")?;
            ensure!(tokens.len() <= w.rows, "Qwen pack exceeds admitted unit");
            w.row_offset.set(0); w.logit_offset.set(0); w.units.set(false); w.terminal_valid.set(false);
            self.last_stream_offset.set(0);
            let mut all = StepTables::default();
            let views = offsets.into_iter().zip(&tables).map(|(offsets, table)|
                decode.attention_tables(offsets, table.page_width)).collect::<Vec<_>>();
            for (s, t) in segments.iter().zip(&tables) {
                all.positions.extend_from_slice(&t.positions);
                all.rope_positions.extend_from_slice(&t.rope_positions);
                all.block_rope_positions.extend_from_slice(&t.block_rope_positions);
                all.kv_slots.extend_from_slice(&t.kv_slots);
                all.slots.extend_from_slice(&t.slots);
                all.seq_first.extend(std::iter::repeat_n(s.first as i32, s.rows));
                all.pool_slots.extend_from_slice(&t.pool_slots);
                all.cache_lengths.extend_from_slice(&t.cache_lengths);
                all.ple_ids.extend_from_slice(&t.ple_ids);
            }
            w.table_view.set(None);
            let all = Arc::new(all);
            self.stage_step_tables(w, &all)?;
            if let Some(mapped) = self.ple.as_ref().map(|p| p.mapped()) {
                drop(self.ple_pending.borrow_mut().take());
                *self.ple_pending.borrow_mut() = Some(mapped.begin(Arc::clone(&all)
                    as Arc<dyn crate::shared::mapped_table::RowSource>, false)?);
            }
            self.stage_embedding(w, &tokens, HC, &w.streams[0], false)?;
            use crate::shared::prefill_pipeline::{run_prefill_units, Unit, UnitSegment};
            let unit = Unit { index: 0, sequence: 0, rows: 0..tokens.len(),
                layers: 0..self.weights.layers.len(), slot: 0, barrier: false,
                segments: segments.iter().enumerate().map(|(i, s)| UnitSegment {
                    sequence: i, rows: s.first..s.end_row(), tokens: s.start..s.start + s.rows,
                    prompt_end: sequences[i].next.is_none(), snapshot: false, head: true,
                }).collect() };
            let run = |transport| -> Result<usize> {
                let mut hooks = PackedHooks { engine: self, w, tables: &tables, segments: &segments,
                    views: &views, transport, cur: 0, entered: false };
                // One queued pack is one shared unit; cross-pack lookahead belongs to the scheduler.
                run_prefill_units(&mut hooks, &[unit], 1)?;
                Ok(hooks.cur)
            };
            let cur = match self.spark_link(0) {
                Some(transport) => {
                    let mut link = transport.borrow_mut();
                    run(Some(link.exchange()?))?
                }
                _ => run(None)?,
            };
            w.table_view.set(None);
            self.last_streams.set((false, cur)); self.last_stream_offset.set(0);
            if let Some(terminal) = &w.terminal_streams {
                // SAFETY: packed rows fit the existing terminal span and share its producer stream.
                unsafe { crate::shared::memory::device::copy_device_ordered(self.library, terminal.buffer,
                    w.streams[cur].buffer, tokens.len() * HC * self.cfg.hidden * 2, self.execution_stream())?; }
                w.terminal_valid.set(true);
            }
            for s in sequences.iter_mut() { s.placement.len += s.tokens.len(); s.placement.state_len = s.placement.len; }
            let result = crate::shared::prefill_packing::finish_members(sequences.len(), |i| {
                let s = &segments[i];
                w.row_offset.set(s.first); w.position.set(s.start);
                self.head(w, &w.streams[cur], s.rows, Scalar::I32(s.rows as i32), 1)?;
                Ok(self.device_logits(w, 1, false))
            }, on_logits);
            w.row_offset.set(0);
            let outcomes = result?;
            drop(workspace);
            super::speculate::prefill_packed(self, sequences, &segments, &views)?;
            Ok(outcomes)
        })
    }

    pub(crate) fn mtp_prefill_rows(&self) -> usize {
        self.prefill_shape.map_or(self.prefill_rows, |shape| shape.unit_rows as usize)
    }

    pub fn prefill(&self, placement: &mut Qwen4Placement, tokens: &[u32]) -> Result<Option<Vec<f32>>> {
        self.prefill_forced(placement, tokens, None, None, 1)
    }

    /// Appends each sequence's tokens (one for decode, several for a verify)
    /// at its length in one decode-shaped step; returns every row's logits.
    /// GDN and PLE state advance in place: a caller rejecting a suffix must replay.
    pub fn verify(&self, sequences: &mut [(&mut Qwen4Placement, &[u32])], on_layer: LayerHook<'_>)
        -> Result<Option<Vec<f32>>> {
        self.verify_step(sequences, on_layer, false, true)?.map(|logits| logits.to_host(self.library)).transpose()
    }

    /// [`Self::verify`] as a speculative step: the GDN and PLE state stay as
    /// they were and each row's replay inputs are recorded; [`Self::commit`]
    /// then applies each sequence's accepted rows (and the caller rewinds the
    /// placements with [`Self::rewind`]). K/V records past the accepted rows
    /// are overwritten when those positions come again.
    pub fn verify_spec(&self, sequences: &mut [(&mut Qwen4Placement, &[u32])], on_layer: LayerHook<'_>)
        -> Result<Option<Vec<f32>>> {
        self.verify_step(sequences, on_layer, true, true)?.map(|logits| logits.to_host(self.library)).transpose()
    }

    /// [`Self::verify`] (`spec`: [`Self::verify_spec`]) leaving every row's
    /// logits on the device, with the decode graph's greedy selection of them.
    pub fn verify_device(&self, sequences: &mut [(&mut Qwen4Placement, &[u32])], spec: bool)
        -> Result<Option<DeviceLogits>> {
        self.verify_step(sequences, None, spec, true)
    }

    /// Diagnostic probes retain exact row geometry and never capture serving graphs.
    pub fn verify_device_ungraphed(&self, sequences: &mut [(&mut Qwen4Placement, &[u32])], spec: bool)
        -> Result<Option<DeviceLogits>> {
        self.verify_step(sequences, None, spec, false)
    }

    fn verify_step(&self, sequences: &mut [(&mut Qwen4Placement, &[u32])], on_layer: LayerHook<'_>,
        spec: bool, graphs: bool) -> Result<Option<DeviceLogits>> {
        // A packed head callback cannot re-enter decode before the pack's MTP
        // has submitted its last read of the borrowed planes.
        drop(self.decode_workspace.try_borrow()?);
        let rows: usize = sequences.iter().map(|(_, t)| t.len()).sum();
        ensure!(rows > 0 && rows <= self.decode_rows, "decode step of {rows} rows");
        let tokens: Vec<u32> = sequences.iter().flat_map(|(_, t)| t.iter().copied()).collect();
        let live_tokens = sequences.iter().map(|(p, t)| p.len + t.len()).max().unwrap_or(1);
        let allocated = |n, unit| crate::shared::context::decode_allocation_units(n, live_tokens, unit);
        let page_stride = allocated(sequences.iter().map(|(p, _)| p.pages.len()).max().unwrap_or(1), PAGE_ROWS)
            .next_power_of_two().max(QWEN_MIN_PAGE_STRIDE).min(self.pages);
        let pool_stride = allocated(sequences.iter().map(|(p, _)| p.pool_pages.len()).max().unwrap_or(1), UNIT_ROWS)
            .next_power_of_two().min(self.pool_pages);
        let mut tables = StepTables { decode: true, spec, page_stride, pool_stride, page_width: page_stride,
            ..Default::default() };
        for (placement, tokens) in sequences.iter_mut() {
            if placement.len == 0 {
                self.start(placement)?;
            }
            let first = tables.kv_slots.len() as i32;
            self.rows(placement, tokens, first, &mut tables)?;
            for _ in 0..tokens.len() {
                extend_padded(&mut tables.page_table, &placement.pages, page_stride);
                extend_padded(&mut tables.pool_table, &placement.pool_pages, pool_stride);
            }
        }
        tables.pool_width = tables.pool_width.next_power_of_two().min(pool_stride);
        let media = if sequences.iter().any(|(p, t)| p.media.as_ref()
            .is_some_and(|media| media.needed(p.len, p.len + t.len()).next().is_some())) {
            let media_rows: Vec<_> = sequences.iter().map(|(p, t)| (&**p, p.len, t.len())).collect();
            embedding_media(&media_rows)?
        } else { Default::default() };
        let logits = self.step(&Arc::new(tables), &tokens, rows, on_layer, None, &media, graphs)?;
        for (placement, tokens) in sequences.iter_mut() {
            placement.len += tokens.len();
            if !spec {
                placement.state_len = placement.len;
            }
        }
        Ok(logits)
    }

    /// Applies each sequence's accepted rows of the last speculative step to
    /// the GDN and PLE state: `(state slot, first step row, accepted rows)`.
    pub fn commit(&self, accepted: &[(i32, usize, usize)]) -> Result<()> {
        let n = accepted.len();
        if n == 0 {
            return Ok(());
        }
        ensure!(n <= self.decode_rows, "commit of {n} sequences");
        let mut table = vec![0i32; 3 * n];
        for (i, &(slot, first, keep)) in accepted.iter().enumerate() {
            ensure!(first + keep <= self.decode_rows && usize::try_from(slot).is_ok_and(|s| s < self.slots),
                "commit of rows {first}+{keep} in slot {slot}");
            (table[i], table[n + i], table[2 * n + i]) = (slot, i32::try_from(first)?, i32::try_from(keep)?);
        }
        // Each bank uploads its commit table in order on its own stream (GdnBank::commit),
        // behind that owner's verify work and the previous commit's readers.
        let i32s = |v: usize| -> Result<Scalar> { Ok(Scalar::I32(i32::try_from(v)?)) };
        for bank in &self.gdn_banks {
            bank.device.run(|| bank.commit(&table, n))?;
        }
        if let (Some(state), Some(replay)) = (&self.ple_state, &self.ple_replay) {
            let bank = self.gdn_banks.iter().find(|bank| bank.device.id == state.device.id)
                .context("Qwen PLE commit owner")?;
            let ple_commit = if self.decode_rows <= DECODE_ROWS { "qwen4_ple_commit" } else { "qwen4_ple_commit_m128" };
            bank.device.run(|| {
                let program = bank.programs.program(ple_commit, &["conv_state", "replay", "tables"])?;
                // SAFETY: PLE pools and this bank's uploaded commit table belong to the same owner.
                unsafe { program.launch(&[state.buffer.ptr, replay.buffer.ptr, bank.commit_tables.buffer.ptr],
                    &[i32s(n)?], bank.stream) }
            })?;
        }
        Ok(())
    }

    /// After a speculative step: `placement` keeps the `kept` tokens it verified
    /// from `start`, whose n-gram history was `history` before the step. The caller commits
    /// the kept rows to the state ([`Self::commit`]) with this rewind.
    pub fn rewind(&self, placement: &mut Qwen4Placement, start: usize, history: NgramHistory, kept: &[u32])
        -> Result<()> {
        placement.len = start + kept.len();
        placement.state_len = placement.len;
        placement.history = history;
        if let Some(ple) = &self.ple {
            ple.hasher.hash(&mut placement.history, kept, &mut Vec::new())?;
        }
        Ok(())
    }

    pub(crate) fn last_target_decode(&self) -> bool {
        self.last_streams.get().0
    }

    /// Copies target stream rows of the last step (`decode`: the decode
    /// workspace, else the prefill one) into the MTP stash: `(state slot,
    /// first step row, rows, first stash row)` per sequence.
    pub fn mtp_stash(&self, decode: bool, rows: &[(i32, usize, usize, usize)]) -> Result<()> {
        let pending = self.mtp_pending.as_ref().context("MTP is not loaded")?;
        let (last_decode, cur) = self.last_streams.get();
        ensure!(last_decode == decode, "the stash reads the last step's workspace");
        let workspace = if decode { self.decode_workspace.borrow() } else { self.workspace.borrow() };
        let w = workspace.as_ref().context("no step ran")?;
        let row = HC * self.cfg.hidden * 2;
        for &(slot, first, count, at) in rows {
            let slot = usize::try_from(slot)?;
            ensure!(slot < self.slots && at + count <= MTP_PENDING_ROWS && first + count <= if decode { w.rows } else { self.prefill_rows },
                "MTP stash of rows {first}+{count} at {at}");
            if count == 0 {
                continue;
            }
            let dst = Self::region(pending, (slot * MTP_PENDING_ROWS + at) * row, count * row);
            let source = if !decode && w.terminal_valid.get() {
                w.terminal_streams.as_ref().context("missing terminal target rows")?
            } else { &w.streams[cur] };
            let src = Self::region(source, first * row, count * row);
            // SAFETY: both regions lie inside live buffers; the stream orders the copy.
            unsafe { crate::shared::memory::device::copy_device_ordered(self.library, dst, src, count * row, self.execution_stream())? };
        }
        Ok(())
    }

    /// One MTP step over `groups` (each sequence's rows contiguous) in the
    /// decode workspace (`decode`, at most 64 rows) or the prefill one,
    /// embedding `tokens`. Rows read their streams from `source`; the output
    /// streams stay in the workspace for a following [`MtpSource::Chain`]
    /// step. With `heads` (step rows), each head row's greedy draft is
    /// returned (token, logit; and with `logits` the head rows' FP32 logits)
    /// or deferred on the device, per `out`. Deferred and head-less steps
    /// queue without a host wait while the staging lasts.
    #[allow(clippy::type_complexity, clippy::too_many_arguments)]
    pub fn mtp_step(&self, decode: bool, groups: &[MtpGroup<'_>], source: MtpSource, heads: &[usize],
        tokens: MtpTokens<'_>, output: MtpOut) -> Result<(Vec<(u32, f32)>, Option<Vec<f32>>)> {
        self.mtp_step_with_tables(decode, groups, source, heads, tokens, output, None)
    }

    #[allow(clippy::type_complexity, clippy::too_many_arguments)]
    pub(super) fn mtp_step_with_tables(&self, decode: bool, groups: &[MtpGroup<'_>], source: MtpSource,
        heads: &[usize], tokens: MtpTokens<'_>, output: MtpOut, packed: Option<AttentionTables>)
        -> Result<(Vec<(u32, f32)>, Option<Vec<f32>>)> {
        if packed.is_some() {
            ensure!(!decode && self.gdn_banks.len() == 1 && self.execution_stream() == self.gdn_banks[0].stream
                && self.decode_workspace.try_borrow().is_err(), "Qwen MTP requires the packed decode-table lease");
        } else if decode {
            drop(self.decode_workspace.try_borrow()?);
        }
        let mtp = self.weights.mtp.as_ref().context("MTP is not loaded (--mtp)")?;
        let (kv, keys, blocks) = self.mtp_kv.as_ref().context("MTP pools")?;
        let (h, vocab) = (self.cfg.hidden, self.cfg.vocab_size);
        let t: usize = groups.iter().map(|g| g.rows.len()).sum();
        let capacity = if decode { self.decode_rows } else { self.mtp_prefill_rows() };
        let token_rows = match tokens { MtpTokens::Host(t) => t.len(), MtpTokens::Drafts { index, .. } => index.len() };
        ensure!(t > 0 && t <= capacity && heads.len() <= t && token_rows == t, "MTP step of {t} rows");
        let mut tables = StepTables { decode, ..Default::default() };
        let mut hidden_rows = Vec::with_capacity(t);
        if decode {
            let live_tokens = groups.iter().flat_map(|g| g.rows.iter().map(|row| row.position + 1))
                .max().unwrap_or(1);
            let stride = |n: usize, unit: usize, total: usize|
                crate::shared::context::decode_allocation_units(n, live_tokens, unit).next_power_of_two().min(total);
            tables.page_stride = stride(groups.iter().map(|g| g.placement.pages.len()).max().unwrap_or(1),
                PAGE_ROWS, self.pages);
            tables.pool_stride = stride(groups.iter().map(|g| g.placement.pool_pages.len()).max().unwrap_or(1),
                UNIT_ROWS, self.pool_pages);
            tables.page_width = tables.page_stride;
        } else {
            ensure!(groups.len() == 1, "a prefill-shaped MTP step takes one sequence");
            if packed.is_none() {
                tables.page_table = groups[0].placement.pages.clone();
                tables.pool_table = groups[0].placement.pool_pages.clone();
            }
            tables.page_width = groups[0].placement.pages.len();
        }
        for group in groups {
            let first = tables.kv_slots.len() as i32;
            let placement = group.placement;
            for row in &group.rows {
                let position = row.position;
                ensure!(position < self.max_context, "MTP position {position} past the context");
                tables.positions.push(position as i64);
                tables.rope_positions.push(placement.rope.at(position)?);
                tables.block_rope_positions.push(placement.rope.at(position - position % BLOCK)?);
                tables.kv_slots.push(placement.record(position)?);
                tables.pool_slots.push(placement.pool_slot(position)?);
                tables.slots.push(placement.slot);
                tables.seq_first.push(first);
                tables.cache_lengths.push(((position + 1) / BLOCK) as i32);
                tables.long |= position + 1 > self.cfg.dense_context();
                tables.pool_width = tables.pool_width.max((position + 1).div_ceil(POOL_PAGE_TOKENS));
                hidden_rows.push(match source {
                    MtpSource::Pending => i32::try_from(usize::try_from(placement.slot)? * MTP_PENDING_ROWS)?
                        + row.source,
                    _ => row.source,
                });
                if decode {
                    extend_padded(&mut tables.page_table, &placement.pages, tables.page_stride);
                    extend_padded(&mut tables.pool_table, &placement.pool_pages, tables.pool_stride);
                }
            }
        }
        if decode {
            tables.pool_width = tables.pool_width.next_power_of_two().min(tables.pool_stride);
        }
        let separate = !decode && self.prefill_shape.is_some();
        let slot = if decode { &self.decode_workspace } else if separate { &self.mtp_prefill_workspace } else { &self.workspace };
        if slot.borrow().is_none() {
            *slot.borrow_mut() = Some(self.workspace(capacity, decode, if decode { self.decode_rows } else { 1 })?);
        }
        let workspace = slot.borrow();
        let w = workspace.as_ref().context("workspace")?;
        ensure!(heads.len() <= w.logit_rows, "{} MTP head rows exceed the workspace's {}", heads.len(), w.logit_rows);
        // Source streams, and the buffer the feedback writes (never the source).
        let (src, dst) = match source {
            MtpSource::Pending => (self.mtp_pending.as_ref().context("MTP stash")?.buffer.ptr, 0),
            MtpSource::Buffer(ptr) => (ptr, 0),
            MtpSource::Chain => {
                let (d, cur) = self.mtp_streams.get();
                ensure!(d == decode, "an MTP chain step follows an MTP step in the same workspace");
                (w.live(&w.streams[cur], HC * self.cfg.hidden * 2), cur ^ 1)
            }
            MtpSource::Target => {
                let (d, cur) = self.last_streams.get();
                ensure!(d == decode, "the MTP reads the last target step of the same workspace");
                if separate {
                    let target = self.workspace.borrow();
                    let target = target.as_ref().context("no target prefill ran")?;
                    let source = if target.terminal_valid.get() {
                        target.terminal_streams.as_ref().context("missing terminal target rows")?
                    } else { &target.streams[cur] };
                    (source.buffer.ptr, 0)
                } else { (w.streams[cur].buffer.ptr, cur ^ 1) }
            }
        };
        self.stage_table(w, &w.positions, &tables.positions)?;
        self.stage_table(w, &w.rope_positions, &tables.rope_positions)?;
        self.stage_table(w, &w.block_rope_positions, &tables.block_rope_positions)?;
        self.stage_table(w, &w.kv_slots, &tables.kv_slots)?;
        self.stage_table(w, &w.slots, &tables.slots)?;
        self.stage_table(w, &w.seq_first, &tables.seq_first)?;
        self.stage_table(w, &w.pool_slots, &tables.pool_slots)?;
        self.stage_table(w, &w.cache_lengths, &tables.cache_lengths)?;
        self.stage_table(w, &w.page_table, &tables.page_table)?;
        self.stage_table(w, &w.pool_table, &tables.pool_table)?;
        self.stage_table(w, &w.hidden_rows, &hidden_rows)?;
        match tokens {
            MtpTokens::Host(tokens) => self.stage_embedding(w, tokens, 1, &w.x, false)?,
            MtpTokens::Drafts { step, index } => {
                ensure!(step < MTP_DEFERRED_STEPS && index.iter().all(|&i| (i as usize) < self.decode_rows),
                    "MTP chain step reads draft step {step}");
                self.stage_table(w, &w.ids, index)?;
                let drafts = Self::region(&self.mtp_drafts, step * self.decode_rows * 4, self.decode_rows * 4);
                // SAFETY: the drafts of `step` and the staged indices are ordered on this
                // stream before the gather; `x` holds t rows.
                unsafe { self.embedding.embed_device_ids(drafts, Some((w.ids.buffer.ptr.cast_const(), index)), t, 1,
                    None, w.live_buffer(&w.x, self.cfg.hidden * 2), self.execution_stream())? };
            }
        }
        if matches!(tokens, MtpTokens::Host(_)) && groups.iter().any(|group| group.placement.media.as_ref()
            .is_some_and(|media| group.rows.iter().any(|row| media.needed(row.position + 1, row.position + 2).next().is_some()))) {
            let media = mtp_embedding_media(groups)?;
            self.inject_media(w, &media, &tables.seq_first, &w.x, t, 1)?;
        }
        let rows = Scalar::I32(t as i32);
        let cap = if decode { self.decode_cap() } else { "m4096" };
        self.run("qwen4_mtp_feedback", &[("hidden", src), ("hidden_rows", w.hidden_rows.buffer.ptr),
            ("embed", w.live(&w.x, self.cfg.hidden * 2)), ("norm_hidden", mtp.norm_hidden.buffer.ptr),
            ("norm_embed", mtp.norm_embed.buffer.ptr), ("fc_hidden", mtp.fc_hidden.buffer.ptr),
            ("fc_embed", mtp.fc_embed.buffer.ptr), ("streams", w.live(&w.streams[dst], HC * self.cfg.hidden * 2)),
            ("delta", w.live(&w.delta, self.cfg.hidden * 2)), ("inject", w.live(&w.inject, HC * 2)), ("scratch", w.scratch.buffer.ptr)],
            &[rows])?;
        // The embedding branch joins every stream (unit injection), then the attention site.
        self.post_pre(w, dst, &mtp.layer, "attn", rows)?;
        let pages = packed.unwrap_or_else(|| w.attention_tables([0; 2], tables.page_width));
        self.attend(w, &mtp.layer, kv.buffer.ptr, keys.buffer.ptr, blocks.buffer.ptr, rows, cap, &tables, pages)?;
        self.post_pre(w, dst ^ 1, &mtp.layer, "mlp", rows)?;
        self.moe(w, self.cfg.layers, &mtp.layer, t, rows, decode)?;
        let mut out = dst;
        self.post(w, &mut out, rows)?;
        self.mtp_streams.set((decode, out));
        if heads.is_empty() {
            return Ok((Vec::new(), None));
        }
        let [norm, down, up] = &mtp.mixer;
        self.run("qwen4_head", &[("streams", w.live(&w.streams[out], HC * self.cfg.hidden * 2)), ("norm", norm.buffer.ptr),
            ("w_down", down.buffer.ptr), ("w_up", up.buffer.ptr), ("out", w.live(&w.x, self.cfg.hidden * 2)),
            ("scratch", w.scratch.buffer.ptr)], &[rows])?;
        // The head rows, gathered contiguously into `delta`.
        let row = h * 2;
        for (i, &r) in heads.iter().enumerate() {
            ensure!(r < t, "MTP head row {r} of {t}");
            // SAFETY: row r of `x` and row i of `delta` lie inside [t, H] buffers.
            unsafe {
                crate::shared::memory::device::copy_device_ordered(self.library, Self::region(&w.delta, i * row, row), Self::region(&w.x, r * row, row),
                    row, self.execution_stream())?;
            }
        }
        let n = heads.len();
        ensure!(n <= self.decode_rows, "{n} MTP head rows");
        // The target's head (one representation shared with the drafts).
        self.logits(w, w.live(&w.delta, self.cfg.hidden * 2), n)?;
        // SAFETY: the logits and the argmax outputs are live buffers of these shapes.
        unsafe {
            self.library.cuda_logits_argmax_checked_f32_async(
                cuteafd_ffi::CuteafdDeviceBuffer { bytes: n * vocab * 4, ..w.logits.buffer },
                Self::region(&w.argmax, 0, n * 4), Self::region(&w.argmax, n * 4, n * 4), n, vocab, self.execution_stream())?;
        }
        let logits = match output {
            MtpOut::Defer { step } => {
                ensure!(step < MTP_DEFERRED_STEPS, "deferred MTP step {step}");
                let at = Self::region(&self.mtp_drafts, step * self.decode_rows * 4, n * 4);
                // SAFETY: both regions are live; the stream orders the copy after the argmax.
                unsafe { crate::shared::memory::device::copy_device_ordered(self.library, at, Self::region(&w.argmax, 0, n * 4), n * 4, self.execution_stream())? };
                return Ok((Vec::new(), None));
            }
            MtpOut::Download { logits } => logits,
        };
        let best = self.download(&w.argmax, n * 8)?;
        let word = |i: usize| u32::from_le_bytes(best[i * 4..i * 4 + 4].try_into().unwrap());
        let drafts = (0..n).map(|i| (word(i), f32::from_bits(word(n + i)))).collect();
        let logits = if logits { Some(self.download_logits(w, n)?) } else { None };
        Ok((drafts, logits))
    }

    /// The deferred drafts of a cycle's steps: `counts[s]` drafts of step `s`
    /// (synchronizes).
    pub fn mtp_drafts(&self, counts: &[usize]) -> Result<Vec<Vec<u32>>> {
        ensure!(counts.len() <= MTP_DEFERRED_STEPS && counts.iter().all(|&n| n <= self.decode_rows), "deferred drafts");
        let bytes = self.download(&self.mtp_drafts, counts.len() * self.decode_rows * 4)?;
        Ok(counts.iter().enumerate().map(|(s, &n)| (0..n).map(|i| {
            let at = (s * self.decode_rows + i) * 4;
            u32::from_le_bytes(bytes[at..at + 4].try_into().unwrap())
        }).collect()).collect())
    }

    /// The last MTP step's output streams (BF16 [rows, 4, H]).
    pub fn mtp_output(&self, rows: usize) -> Result<Vec<u8>> {
        let (decode, cur) = self.mtp_streams.get();
        let workspace = if decode { self.decode_workspace.borrow() }
            else if self.prefill_shape.is_some() { self.mtp_prefill_workspace.borrow() } else { self.workspace.borrow() };
        let w = workspace.as_ref().context("no MTP step ran")?;
        self.download(&w.streams[cur], rows * HC * self.cfg.hidden * 2)
    }

    fn inject_media(&self, w: &Workspace<'_>, media: &cuteafd_engine::media::MediaChunk,
        seq_first: &[i32], out: &Dev<'_>, rows: usize, copies: usize) -> Result<()> {
        if media.indices.is_empty() { return Ok(()); }
        // Scratch is consumed before norm/feedback; injection drains before the
        // reused sequence table is restored. Graph storage stays unchanged.
        self.library.embedding_injection()?.inject_host(&media.features, &media.indices,
            w.live_buffer(&w.delta, self.cfg.hidden * 2), w.seq_first.buffer, w.live_buffer(out, copies * self.cfg.hidden * 2), rows, self.cfg.hidden, copies, self.execution_stream())?;
        self.put(&w.seq_first, seq_first)
    }



    fn step(&self, tables: &Arc<StepTables>, tokens: &[u32], logit_rows: usize, on_layer: LayerHook<'_>,
        forced: Forced<'_>, media: &cuteafd_engine::media::MediaChunk, graphs: bool) -> Result<Option<DeviceLogits>> {
        crate::shared::experts::rtx::pair::submit_recoverable(|| self.require_live(), |kind| self.cancel_submission(kind), || self.shutdown_submission(), |_| {
            self.begin_graph_cycle()?;
            let output = self.step_inner(tables, tokens, logit_rows, on_layer, forced, media, graphs)?;
            self.graph_completed()?;
            Ok(output)
        })
    }

    fn step_inner(&self, tables: &Arc<StepTables>, tokens: &[u32], logit_rows: usize, mut on_layer: LayerHook<'_>,
        forced: Forced<'_>, media: &cuteafd_engine::media::MediaChunk, graphs: bool) -> Result<Option<DeviceLogits>> {
        self.last_stream_offset.set(0);
        self.graph_cycle(tables);
        if let Some(w) = self.workspace.borrow().as_ref() {
            w.row_offset.set(0); w.logit_offset.set(0); w.units.set(false);
            w.terminal_valid.set(false);
        }
        ensure!(self.gdn_banks.len() == 1 || self.hops.is_some(),
            "Qwen dual owner execution needs TP2 exchange and owner graph admission");
        if self.gdn_banks.len() == 2 {
            return self.step_placed(tables, tokens, logit_rows, on_layer, forced, media, graphs);
        }
        let (h, t) = (self.cfg.hidden, tables.kv_slots.len());
        let (slot, capacity) = if tables.decode { (&self.decode_workspace, self.decode_rows) } else { (&self.workspace, self.prefill_rows) };
        if slot.borrow().as_ref().is_some_and(|w| w.logit_rows < logit_rows || w.rows < capacity) {
            self.allow_workspace_creation()?;
            *slot.borrow_mut() = None;
        }
        if slot.borrow().is_none() {
            let rows = if tables.decode { self.decode_rows } else { logit_rows.max(1) };
            *slot.borrow_mut() = Some(self.workspace(capacity, tables.decode, rows)?);
        }
        let workspace = slot.borrow();
        let w = workspace.as_ref().context("workspace")?;
        w.position.set(tables.positions.first().copied().unwrap_or(0) as usize);
        ensure!(t <= w.rows && logit_rows <= t, "step exceeds the workspace");
        self.stage_table(w, &w.positions, &tables.positions)?;
        self.stage_table(w, &w.rope_positions, &tables.rope_positions)?;
        self.stage_table(w, &w.block_rope_positions, &tables.block_rope_positions)?;
        self.stage_table(w, &w.kv_slots, &tables.kv_slots)?;
        self.stage_table(w, &w.slots, &tables.slots)?;
        self.stage_table(w, &w.seq_first, &tables.seq_first)?;
        self.stage_table(w, &w.pool_slots, &tables.pool_slots)?;
        self.stage_table(w, &w.cache_lengths, &tables.cache_lengths)?;
        self.stage_table(w, &w.page_table, &tables.page_table)?;
        self.stage_table(w, &w.pool_table, &tables.pool_table)?;
        match self.ple.as_ref().map(|p| p.mapped()) {
            // Gathers while the layers before the PLE layer are enqueued (`finish_ple`).
            Some(mapped) => {
                // Cancel unsubmitted rows without waiting on page faults.
                drop(self.ple_pending.borrow_mut().take());
                *self.ple_pending.borrow_mut() = Some(mapped.begin(Arc::clone(tables) as Arc<dyn crate::shared::mapped_table::RowSource>, tables.decode)?);
            }
            None => self.stage_table(w, &w.ple_ids, &tables.ple_ids)?,
        }
        // Streams start as four copies of the embedding.
        let row = h * 2;
        ensure!(tokens.len() == t, "{} tokens for a {t}-row step", tokens.len());
        let graphed = graphs && self.use_graphs && self.graph_capture.get() && tables.decode
            && on_layer.is_none() && forced.is_none() && media.indices.is_empty();
        self.stage_embedding(w, tokens, HC, &w.streams[0], graphed)?;
        self.inject_media(w, media, &tables.seq_first, &w.streams[0], t, HC)?;
        let rows = Scalar::I32(t as i32);
        if graphed {
            return self.decode_graphed(w, tables, t, rows, logit_rows);
        }
        let cap = if tables.decode { self.decode_cap() } else { "m4096" };
        let layers = &self.weights.layers;
        let mut cur = 0usize;
        let spec = tables.spec;
        if self.cfg.ple_layers.contains(&0) {
            self.finish_ple(w)?;
        }
        self.enter(w, &mut cur, None, &layers[0], 0, rows, spec)?;
        for (index, layer) in layers.iter().enumerate() {
            match layer.attention {
                Qwen4Attention::Gdn => self.gdn(w, index, layer, rows, cap, spec)?,
                Qwen4Attention::Full => self.full(w, index, layer, rows, cap, tables)?,
            }
            // Attention back into the streams, then the MLP site's input.
            self.post_pre(w, cur, layer, "mlp", rows)?;
            cur ^= 1;
            self.moe(w, index, layer, t, rows, tables.decode)?;
            let forced_rows = forced.and_then(|f| f(index));
            match layers.get(index + 1) {
                Some(next) => {
                    let has_ple = self.cfg.ple_layers.contains(&(index + 1));
                    if has_ple || on_layer.is_some() || forced_rows.is_some() {
                        // Materialize this layer's output first (PLE and hooks see it).
                        self.post(w, &mut cur, rows)?;
                        if let Some(on_layer) = on_layer.as_mut() {
                            on_layer(index, &self.download(&w.streams[cur], t * HC * row)?)?;
                        }
                        if let Some(rows_forced) = forced_rows {
                            ensure!(rows_forced.len() == t * HC * row, "teacher-forced streams of the wrong size");
                            self.library.copy_h2d(cuteafd_ffi::CuteafdDeviceBuffer { bytes: rows_forced.len(),
                                ..w.streams[cur].buffer }, &rows_forced)?;
                        }
                        if has_ple {
                            self.finish_ple(w)?;
                        }
                        self.enter(w, &mut cur, None, next, index + 1, rows, spec)?;
                    } else {
                        self.enter(w, &mut cur, Some(()), next, index + 1, rows, spec)?;
                    }
                }
                None => {
                    self.post(w, &mut cur, rows)?;
                    if let Some(on_layer) = on_layer.as_mut() {
                        on_layer(index, &self.download(&w.streams[cur], t * HC * row)?)?;
                    }
                }
            }
            crate::shared::console::layer_mark(index);
        }
        self.last_streams.set((tables.decode, cur));
        if layers.len() < self.cfg.layers {
            self.drain_partial_step()?;
            return Ok(None);
        }
        self.head(w, &w.streams[cur], t, rows, logit_rows)?;
        Ok(Some(self.device_logits(w, logit_rows, false)))
    }

    fn stage_step_tables(&self, w: &Workspace<'_>, tables: &StepTables) -> Result<()> {
        self.stage_table(w, &w.positions, &tables.positions)?;
        self.stage_table(w, &w.rope_positions, &tables.rope_positions)?;
        self.stage_table(w, &w.block_rope_positions, &tables.block_rope_positions)?;
        self.stage_table(w, &w.kv_slots, &tables.kv_slots)?;
        self.stage_table(w, &w.slots, &tables.slots)?;
        self.stage_table(w, &w.seq_first, &tables.seq_first)?;
        self.stage_table(w, &w.pool_slots, &tables.pool_slots)?;
        self.stage_table(w, &w.cache_lengths, &tables.cache_lengths)?;
        self.stage_table(w, &w.page_table, &tables.page_table)?;
        self.stage_table(w, &w.pool_table, &tables.pool_table)?;
        if self.ple.as_ref().map(|p| p.mapped()).is_none() {
            self.stage_table(w, &w.ple_ids, &tables.ple_ids)?;
        }
        Ok(())
    }

    /// Unfused ownership transitions: post on the old owner, hop, then pre on
    /// the new owner. Same-owner boundaries retain the fused post/pre path.
    #[allow(clippy::too_many_arguments)]
    fn step_placed(&self, tables: &Arc<StepTables>, tokens: &[u32], logit_rows: usize, mut on_layer: LayerHook<'_>,
        forced: Forced<'_>, media: &cuteafd_engine::media::MediaChunk, graphs: bool) -> Result<Option<DeviceLogits>> {
        let graphed = graphs && self.use_graphs && tables.decode
            && on_layer.is_none() && forced.is_none() && media.indices.is_empty();
        let hops = self.hops.as_ref().context("Qwen dual residual hops")?;
        let (t, h) = (tables.kv_slots.len(), self.cfg.hidden);
        ensure!(tokens.len() == t && logit_rows <= t, "invalid Qwen owner step rows");
        let capacity = if tables.decode { self.decode_rows } else { self.prefill_rows };
        for owner in 0..2 {
            self.on_owner(owner, || {
                let slot = self.workspace_slot(owner, tables.decode)?;
                if slot.borrow().as_ref().is_some_and(|w| w.logit_rows < logit_rows || w.rows < capacity) {
                    self.allow_workspace_creation()?;
                    self.drain_state()?;
                    *slot.borrow_mut() = None;
                }
                if slot.borrow().is_none() {
                    *slot.borrow_mut() = Some(self.workspace(capacity, tables.decode,
                        if tables.decode { self.decode_rows } else { logit_rows.max(1) })?);
                }
                let workspace = slot.borrow();
                let w = workspace.as_ref().context("Qwen owner workspace")?;
                ensure!(t <= w.rows, "Qwen owner step exceeds workspace");
                self.stage_step_tables(w, tables)
            })?;
        }
        let root = self.workspace_slot(0, tables.decode)?.borrow();
        let peer = self.workspace_slot(1, tables.decode)?.borrow();
        let work = [root.as_ref().context("root workspace")?, peer.as_ref().context("peer workspace")?];
        if let Some(mapped) = self.ple.as_ref().map(|p| p.mapped()) {
            drop(self.ple_pending.borrow_mut().take());
            *self.ple_pending.borrow_mut() = Some(mapped.begin(Arc::clone(tables) as Arc<dyn crate::shared::mapped_table::RowSource>, tables.decode)?);
        }
        self.on_owner(0, || {
            self.stage_embedding(work[0], tokens, HC, &work[0].streams[0], graphed)?;
            self.inject_media(work[0], media, &tables.seq_first, &work[0].streams[0], t, HC)
        })?;
        let rows = Scalar::I32(i32::try_from(t)?);
        let cap = if tables.decode { self.decode_cap() } else { "m4096" };
        if graphed { return self.decode_placed(work, tables, t, rows, logit_rows); }
        let mut cur = [0; 2];
        let layers = &self.weights.layers;
        self.on_owner(0, || {
            if self.cfg.ple_layers.contains(&0) { self.finish_ple(work[0])?; }
            self.enter(work[0], &mut cur[0], None, &layers[0], 0, rows, tables.spec)
        })?;
        for (index, layer) in layers.iter().enumerate() {
            let owner = self.state_map.layers[index].owner;
            let w = work[owner];
            self.on_owner(owner, || {
                match layer.attention {
                    Qwen4Attention::Gdn => self.gdn(w, index, layer, rows, cap, tables.spec)?,
                    Qwen4Attention::Full => self.full(w, index, layer, rows, cap, tables)?,
                }
                self.post_pre(w, cur[owner], layer, "mlp", rows)?;
                cur[owner] ^= 1;
                self.moe(w, index, layer, t, rows, tables.decode)
            })?;
            let next = layers.get(index + 1);
            let next_owner = next.map(|_| self.state_map.layers[index + 1].owner);
            let forced_rows = forced.and_then(|force| force(index));
            let materialize = next.is_none() || next_owner != Some(owner) || on_layer.is_some()
                || forced_rows.is_some() || self.cfg.ple_layers.contains(&(index + 1));
            self.on_owner(owner, || {
                if materialize {
                    self.post(w, &mut cur[owner], rows)?;
                    if let Some(hook) = on_layer.as_mut() { hook(index, &self.download(&w.streams[cur[owner]], t * HC * h * 2)?)?; }
                    if let Some(bytes) = forced_rows {
                        ensure!(bytes.len() == t * HC * h * 2, "invalid forced owner streams");
                        self.library.copy_h2d(cuteafd_ffi::CuteafdDeviceBuffer { bytes: bytes.len(),
                            ..w.streams[cur[owner]].buffer }, &bytes)?;
                    }
                }
                Ok(())
            })?;
            crate::shared::console::layer_mark(index);
            if let Some(next) = next {
                let next_owner = next_owner.context("Qwen next attention owner")?;
                if next_owner != owner {
                    let ticket = hops.send(0, 0, w.streams[cur[owner]].buffer.ptr, t)?;
                    let landed = hops.land(ticket)?;
                    self.on_owner(next_owner, || {
                        let dst = work[next_owner].streams[cur[next_owner]].buffer;
                        let src = cuteafd_ffi::CuteafdDeviceBuffer { ptr: landed, ..dst };
                        // SAFETY: hop wait precedes this same-owner copy into persistent streams.
                        unsafe { crate::shared::memory::device::copy_device_ordered(self.library, dst, src, t * HC * h * 2, self.execution_stream()) }
                    })?;
                }
                self.on_owner(next_owner, || {
                    if self.cfg.ple_layers.contains(&(index + 1)) { self.finish_ple(work[next_owner])?; }
                    self.enter(work[next_owner], &mut cur[next_owner], (!materialize).then_some(()), next,
                        index + 1, rows, tables.spec)
                })?;
            }
        }
        let ticket = hops.send(1, 0, work[1].streams[cur[1]].buffer.ptr, t)?;
        let landed = hops.land(ticket)?;
        self.on_owner(0, || {
            let dst = work[0].streams[cur[0]].buffer;
            let src = cuteafd_ffi::CuteafdDeviceBuffer { ptr: landed, ..dst };
            // SAFETY: exit wait precedes the head/MTP owner-zero stream copy.
            unsafe { crate::shared::memory::device::copy_device_ordered(self.library, dst, src, t * HC * h * 2, self.execution_stream())?; }
            self.last_streams.set((tables.decode, cur[0]));
            if layers.len() < self.cfg.layers { self.drain_partial_step()?; return Ok(None); }
            self.head(work[0], &work[0].streams[cur[0]], t, rows, logit_rows)?;
            Ok(Some(self.device_logits(work[0], logit_rows, false)))
        })
    }

    fn decode_placed(&self, work: [&Workspace<'_>; 2], tables: &StepTables, t: usize,
        rows: Scalar, logit_rows: usize) -> Result<Option<DeviceLogits>> {
        let layers = &self.weights.layers;
        let n = layers.len();
        let hops = self.hops.as_ref().context("Qwen graph residual hops")?;
        let head = n == self.cfg.layers && logit_rows == t;
        let key = |segment| GraphKey::new(segment, t, layers.get(segment).map(|layer| layer.attention), tables);
        let mut cur = [0; 2];
        for (index, layer) in layers.iter().enumerate() {
            let owner = self.state_map.layers[index].owner;
            let w = work[owner];
            let boundary = index > 0 && self.state_map.layers[index - 1].owner != owner;
            if boundary {
                self.on_owner(0, || {
                    let mut c = cur[0];
                    self.post(work[0], &mut c, rows)
                })?;
                cur[0] ^= 1;
                let ticket = hops.send(0, 0, work[0].streams[cur[0]].buffer.ptr, t)?;
                let landed = hops.land(ticket)?;
                self.on_owner(1, || {
                    let dst = w.streams[cur[1]].buffer;
                    let src = cuteafd_ffi::CuteafdDeviceBuffer { ptr: landed, ..dst };
                    // SAFETY: cutover land orders this copy before owner-one attention graph.
                    unsafe { crate::shared::memory::device::copy_device_ordered(self.library, dst, src, t * HC * self.cfg.hidden * 2, self.execution_stream()) }
                })?;
            }
            self.on_owner(owner, || {
                if self.cfg.ple_layers.contains(&index) { self.finish_ple(w)?; }
                let start = cur[owner];
                self.replay(key(index), || {
                    let mut c = start;
                    if index == 0 && self.embedding.device_gather() {
                        // SAFETY: staged token ids and persistent root streams cover this graph bucket.
                        unsafe { self.embedding.gather(w.ids.buffer.ptr, std::ptr::null(), t, HC,
                            std::ptr::null(), w.live(&w.streams[0], HC * self.cfg.hidden * 2), None, crate::shared::peer_split::RankDevice {
                                device: self.gdn_banks[self.active_owner.get()].device.id, stream: self.execution_stream() })?; }
                    }
                    if index == 0 || boundary {
                        self.enter(w, &mut c, None, layer, index, rows, tables.spec)?;
                    } else if self.cfg.ple_layers.contains(&index) {
                        self.post(w, &mut c, rows)?;
                        self.enter(w, &mut c, None, layer, index, rows, tables.spec)?;
                    } else {
                        self.enter(w, &mut c, Some(()), layer, index, rows, tables.spec)?;
                    }
                    match layer.attention {
                        Qwen4Attention::Gdn => self.gdn(w, index, layer, rows, self.decode_cap(), tables.spec)?,
                        Qwen4Attention::Full => self.full(w, index, layer, rows, self.decode_cap(), tables)?,
                    }
                    self.post_pre(w, c, layer, "mlp", rows)?;
                    // Routed experts run between segments (and outside capture).
                    Ok(())
                })?;
                if index == 0 || boundary { cur[owner] ^= 1; }
                self.moe_front(w, index, layer, t, rows)?;
                self.moe_experts(w, index, t, rows, true)?;
                Ok(())
            })?;
            crate::shared::console::layer_mark(index);
        }
        self.on_owner(1, || {
            let mut c = cur[1];
            self.post(work[1], &mut c, rows)
        })?;
        cur[1] ^= 1;
        let ticket = hops.send(1, 0, work[1].streams[cur[1]].buffer.ptr, t)?;
        let landed = hops.land(ticket)?;
        self.on_owner(0, || {
            let dst = work[0].streams[cur[0]].buffer;
            let src = cuteafd_ffi::CuteafdDeviceBuffer { ptr: landed, ..dst };
            // SAFETY: exit land precedes persistent root stream copy/head graph.
            unsafe { crate::shared::memory::device::copy_device_ordered(self.library, dst, src, t * HC * self.cfg.hidden * 2, self.execution_stream())?; }
            if head {
                self.replay(key(n), || {
                    self.head(work[0], &work[0].streams[cur[0]], t, rows, t)?;
                    self.select_greedy(work[0], t)
                })?;
            }
            self.last_streams.set((true, cur[0]));
            if n < self.cfg.layers { self.drain_partial_step()?; return Ok(None); }
            if !head { self.head(work[0], &work[0].streams[cur[0]], t, rows, logit_rows)?; }
            Ok(Some(self.device_logits(work[0], logit_rows, head)))
        })
    }

    /// The stream mixer and lm_head over the last `logit_rows` rows, and with
    /// `greedy` the rows' greedy tokens into `select` (decode graphs).
    fn mix_head(&self, w: &Workspace<'_>, streams: &Dev<'_>, rows: Scalar) -> Result<()> {
        let [norm, down, up] = &self.weights.mixer;
        self.run("qwen4_head", &[("streams", w.live(streams, HC * self.cfg.hidden * 2)), ("norm", norm.buffer.ptr),
            ("w_down", down.buffer.ptr), ("w_up", up.buffer.ptr), ("out", w.live(&w.x, self.cfg.hidden * 2)),
            ("scratch", w.scratch.buffer.ptr)], &[rows])
    }

    fn head(&self, w: &Workspace<'_>, streams: &Dev<'_>, t: usize, rows: Scalar, logit_rows: usize) -> Result<()> {
        self.mix_head(w, streams, rows)?;
        ensure!(logit_rows <= t, "{logit_rows} logits rows of a {t}-row step");
        let x = w.live(&w.x, self.cfg.hidden * 2).cast::<u8>()
            .wrapping_add((t - logit_rows) * self.cfg.hidden * 2).cast();
        self.logits(w, x, logit_rows)
    }

    fn record_head(&self, w: &Workspace<'_>, streams: &Dev<'_>, rows: usize, logit_rows: usize,
        scoring: crate::shared::probe::ScoringRows<'_>) -> Result<()> {
        ensure!(logit_rows > 0 && logit_rows <= rows, "Qwen scoring head extent");
        self.mix_head(w, streams, Scalar::I32(i32::try_from(rows)?))?;
        // The ordinary prefill plane can be just one row. Consume each tile
        // before reuse without growing that admitted plane or changing unit cuts.
        w.logit_offset.set(0);
        for span in crate::shared::probe::head_spans(logit_rows, w.logit_rows.min(DECODE_ROWS))? {
            let x = w.live(&w.x, self.cfg.hidden * 2).cast::<u8>()
                .wrapping_add((rows - logit_rows + span.start) * self.cfg.hidden * 2).cast();
            self.logits(w, x, span.len())?;
            scoring.record(self.library, span.start, &self.device_logits(w, span.len(), false))?;
        }
        Ok(())
    }

    /// FP32 logits of `rows` BF16 head inputs at `x` (rows of `[*, H]`) into the
    /// workspace's logits rows, through the one resident LM head: the BF16 head
    /// GEMM, or the E4M3 `qwen4_head_fp8` in spans of 16 rows.
    fn logits(&self, w: &Workspace<'_>, x: *mut c_void, rows: usize) -> Result<()> {
        let (h, vocab) = (self.cfg.hidden, self.cfg.vocab_size);
        ensure!(rows <= w.logit_rows, "{rows} logits rows exceed the workspace's {}", w.logit_rows);
        if rows == 0 {
            return Ok(());
        }
        match &self.weights.head {
            // SAFETY: `x` holds `rows` head inputs; the head and the logits rows are live buffers of these shapes.
            Qwen4Head::Bf16(head) => unsafe {
                crate::shared::target_head::launch(crate::shared::target_head::TargetHead::configured(), w.head.as_ref().context("BF16 head not constructed")?,
                    x.cast_const(), head.buffer.ptr.cast_const(), Self::region(&w.logits, w.logit_offset.get() * vocab * 4, rows * vocab * 4).ptr.cast(), rows, self.execution_stream())
            },
            Qwen4Head::Fp8 { values, scales } => {
                for (first, n) in fp8_head_spans(rows) {
                    let input = x.cast::<u8>().wrapping_add(first * h * 2).cast();
                    let out = Self::region(&w.logits, (w.logit_offset.get() + first) * vocab * 4, n * vocab * 4).ptr;
                    self.run("qwen4_head_fp8", &[("x", input), ("w_fp8", values.buffer.ptr),
                        ("scale", scales.buffer.ptr), ("logits", out)], &[Scalar::I32(n as i32)])?;
                }
                Ok(())
            }
        }
    }

    /// Greedy tokens of the first `rows` logits rows into `select`.
    fn select_greedy(&self, w: &Workspace<'_>, rows: usize) -> Result<()> {
        let vocab = self.cfg.vocab_size;
        // SAFETY: the logits rows and the select buffer (ids, then statuses) are live buffers of these shapes.
        unsafe {
            self.library.cuda_logits_greedy_f32_async(w.logits.buffer.ptr, rows, vocab, vocab, w.select.buffer.ptr,
                std::ptr::null_mut(), Self::region(&w.select, rows * 4, rows * 4).ptr, self.execution_stream())
        }
    }

    /// The first `rows` logits rows as device logits (`greedy`: with the rows'
    /// selection from [`Self::select_greedy`]).
    fn device_logits(&self, w: &Workspace<'_>, rows: usize, greedy: bool) -> DeviceLogits {
        let vocab = self.cfg.vocab_size;
        DeviceLogits { ptr: w.logits.buffer.ptr, rows, vocab, stride: vocab, stream: self.execution_stream(),
            greedy: greedy.then(|| (w.select.buffer.ptr.cast_const(), Self::region(&w.select, rows * 4, rows * 4).ptr
                .cast_const())) }
    }

    /// The first `rows` logits rows through the pinned landing buffer.
    fn download_logits(&self, w: &Workspace<'_>, rows: usize) -> Result<Vec<f32>> {
        let n = rows * self.cfg.vocab_size;
        let host = w.logits_host.borrow();
        ensure!(n * 4 <= host.buffer.bytes, "logits rows exceed the landing buffer");
        // SAFETY: the pinned buffer holds n floats; the sync completes the copy before the read.
        unsafe {
            self.library.copy_d2h_host_buffer_async(host.buffer, w.logits.buffer, n * 4, self.execution_stream())?;
            self.library.cuda_stream_synchronize(self.execution_stream())?;
            Ok(std::slice::from_raw_parts(host.buffer.ptr.cast::<f32>(), n).to_vec())
        }
    }

    /// A decode step as captured segments: segment `i` finishes layer `i - 1`
    /// (its MoE output into the streams) and runs layer `i` up to its routed
    /// experts, which run (local or on the Sparks) between segments. Every
    /// post flips the stream buffer: segment 0 flips once (the MLP site),
    /// later layers twice (their attention entry, then the MLP site), and the
    /// final segment once; the parity is the same every step, so replays
    /// track it without running the closures.
    fn decode_graphed(&self, w: &Workspace<'_>, tables: &StepTables, t: usize, rows: Scalar, logit_rows: usize)
        -> Result<Option<DeviceLogits>> {
        let layers = &self.weights.layers;
        // Every layer resident: the last segment ends in the head and the greedy selection.
        let head = layers.len() == self.cfg.layers && logit_rows == t;
        let gather = self.embedding.device_gather();
        let mut cur = 0usize;
        for index in 0..=layers.len() {
            let key = GraphKey::new(index, t, layers.get(index).map(|layer| layer.attention), tables);
            let start = cur;
            if self.cfg.ple_layers.contains(&index) {
                self.finish_ple(w)?;
            }
            let segment = || -> Result<()> {
                let mut c = start;
                let Some(layer) = layers.get(index) else {
                    self.post(w, &mut c, rows)?;
                    if head {
                        self.head(w, &w.streams[c], t, rows, t)?;
                        self.select_greedy(w, t)?;
                    }
                    return Ok(());
                };
                let spec = tables.spec;
                if index == 0 && gather {
                    // SAFETY: the step's ids are staged before the replay; the streams hold its rows.
                    unsafe { self.embedding.gather(w.ids.buffer.ptr, std::ptr::null(), t, HC, std::ptr::null(),
                        w.live(&w.streams[0], HC * self.cfg.hidden * 2), None, crate::shared::peer_split::RankDevice {
                            device: self.gdn_banks[self.active_owner.get()].device.id, stream: self.execution_stream() })? };
                }
                if index == 0 {
                    self.enter(w, &mut c, None, layer, index, rows, spec)?;
                } else if self.cfg.ple_layers.contains(&index) {
                    self.post(w, &mut c, rows)?;
                    self.enter(w, &mut c, None, layer, index, rows, spec)?;
                } else {
                    self.enter(w, &mut c, Some(()), layer, index, rows, spec)?;
                }
                match layer.attention {
                    Qwen4Attention::Gdn => self.gdn(w, index, layer, rows, self.decode_cap(), spec)?,
                    Qwen4Attention::Full => self.full(w, index, layer, rows, self.decode_cap(), tables)?,
                }
                self.post_pre(w, c, layer, "mlp", rows)?;
                // Routed experts run between segments (and outside capture).
                Ok(())
            };
            if index == layers.len() && !head { segment()?; } else { self.replay(key, segment)?; }
            cur ^= if index == 0 || index == layers.len() { 1 } else { 0 };
            if index < layers.len() {
                self.moe_front(w, index, &layers[index], t, rows)?;
                self.moe_experts(w, index, t, rows, true)?;
                crate::shared::console::layer_mark(index);
            }
        }
        self.last_streams.set((true, cur));
        if layers.len() < self.cfg.layers {
            self.drain_partial_step()?;
            return Ok(None);
        }
        if !head {
            self.head(w, &w.streams[cur], t, rows, logit_rows)?;
        }
        Ok(Some(self.device_logits(w, logit_rows, head)))
    }

    fn replay(&self, key: GraphKey, segment: impl Fn() -> Result<()>) -> Result<()> {
        self.require_live()?;
        let owner = self.active_owner.get();
        if let Some(graph) = self.graphs.borrow_mut()[owner].cached_launch(&key).map(|graph| graph.0) {
            // SAFETY: the graph's pointers are persistent engine buffers.
            unsafe { crate::shared::decode_graph::launch(&self.library, graph, self.execution_stream())?; }
            return self.ple_consumed(&key);
        }
        // Graphs unconfigured, or a step before initialization: run the segment
        // ungraphed without recording a miss.
        if !self.graph_capture.get() || !self.graph_warmed.get() {
            segment()?;
            return self.ple_consumed(&key);
        }
        let ticket = self.graphs.borrow_mut()[owner].cache_ticket(&key, false, None);
        let Some(mut ticket) = ticket else {
            // The policy ran it eagerly (reserve/headroom or no headroom counted):
            // record the shape once so the next trace can inspect it.
            self.miss_graph(key);
            segment()?;
            return self.ple_consumed(&key);
        };
        // SAFETY: capture records this stream's launches; nothing in a segment
        // synchronizes the host or allocates.
        if let Err(error) = unsafe { crate::shared::decode_graph::begin_capture(&self.library, self.execution_stream(), &ticket) } {
            if self.graphs.borrow_mut()[owner].cache_ticket_failed(ticket, &error) {
                tracing::warn!(%error, "Qwen graph capture unavailable; running eagerly");
            }
            segment()?;
            return self.ple_consumed(&key);
        }
        let captured = segment();
        // SAFETY: ends the capture begun above on the same stream.
        let exec = unsafe { crate::shared::decode_graph::end_capture(&self.library, self.execution_stream(), &mut ticket) };
        if let Err(error) = captured {
            self.graphs.borrow_mut()[owner].cache_ticket_failed(ticket, exec.as_ref().err().unwrap_or(&error));
            if let Ok((raw, _, _, _, _)) = exec {
                // SAFETY: the unlaunched executable owns no queued work.
                unsafe { let _ = self.library.cuda_graph_exec_destroy(raw); }
            }
            return Err(error);
        }
        let (raw, nodes, width, spans_devices, inspection_error) = match exec {
            Ok(info) => info,
            Err(error) => {
                if self.graphs.borrow_mut()[owner].cache_ticket_failed(ticket, &error) {
                    tracing::warn!(%error, "Qwen graph instantiation unavailable; running eagerly");
                }
                segment()?;
                return self.ple_consumed(&key);
            }
        };
        if let Some(error) = inspection_error {
            static WARNED: std::sync::Once = std::sync::Once::new();
            WARNED.call_once(|| tracing::warn!(site = "qwen4.segment", %error,
                "graph topology unknown; retaining executable without chain reuse"));
        }
        crate::shared::decode_graph::census::capture_key("qwen4", "decode",
            self.graphs.as_ptr() as usize, owner as i32, Some(0), &key, raw);
        let graph = GraphExec(raw, Device { library: self.library, id: self.library.cuda_get_device()? });
        self.graphs.borrow_mut()[owner].cache_commit(key, graph, ticket, nodes as u64, width, spans_devices);
        let graph = self.graphs.borrow()[owner].get(&key).context("inserted Qwen graph")?.0;
        // SAFETY: the new graph reads and writes persistent engine buffers.
        unsafe { crate::shared::decode_graph::launch(&self.library, graph, self.execution_stream())?; }
        self.ple_consumed(&key)
    }

    /// The PLE reader of a graphed layer segment has just been launched: its
    /// mapped rows are consumed, and no later layer is queued yet.
    fn ple_consumed(&self, key: &GraphKey) -> Result<()> {
        if self.cfg.ple_layers.contains(&key.segment) {
            if let Some(ple) = &self.ple {
                // SAFETY: this segment is the last work queued on the stream.
                unsafe { ple.mapped().consumed(self.execution_stream())?; }
            }
        }
        Ok(())
    }

    /// Record and warn about one ungraphed decode shape. The warning names the
    /// family, the shape key and the engine's request context exactly once per
    /// variant; every occurrence joins the process-wide `/v1/stats`
    /// `graph_missing_variants` inventory that `probe::graph_capture_stats`
    /// publishes.
    fn miss_graph(&self, key: GraphKey) {
        let variant = format!("rows={} spec={} head={} long={} page_width={} page_stride={} pool_stride={}",
            key.rows, key.spec, key.head, key.long, key.page_width, key.page_stride, key.pool_stride);
        crate::shared::decode_graph::note_eager("qwen4", &variant, || format!(
            "segment={} max_context={} slots={} pages={}",
            key.segment, self.max_context, self.slots, self.pages));
    }

    fn site<'l>(layer: &'l Qwen4Layer<'_>, site: &str) -> Result<[*mut c_void; 3]> {
        Ok([layer.ptr(&format!("{site}.norm"))?, layer.ptr(&format!("{site}.w_di"))?,
            layer.ptr(&format!("{site}.w_up"))?])
    }

    /// Into layer `index`'s attention site: with `posted` = None the streams
    /// in `cur` are this layer's input (PLE first when it has one, then
    /// `hc_pre`); with Some the previous MoE output is still in `delta` and
    /// `hc_post_pre` posts it (into the other buffer) and enters.
    #[allow(clippy::too_many_arguments)]
    fn enter(&self, w: &Workspace<'_>, cur: &mut usize, posted: Option<()>, layer: &Qwen4Layer<'_>, index: usize,
        rows: Scalar, spec: bool) -> Result<()> {
        let [norm, di, up] = Self::site(layer, "attn")?;
        if posted.is_some() {
            self.post_pre(w, *cur, layer, "attn", rows)?;
            *cur ^= 1;
            return Ok(());
        }
        if self.cfg.ple_layers.contains(&index) {
            self.ple(w, &w.streams[*cur], layer, rows, spec)?;
        }
        self.run("qwen4_hc_pre", &[("residual", w.live(&w.streams[*cur], HC * self.cfg.hidden * 2)), ("norm", norm), ("w_di", di),
            ("w_up", up), ("y", w.live(&w.x, self.cfg.hidden * 2)), ("inject", w.live(&w.inject, HC * 2)), ("scratch", w.scratch.buffer.ptr)],
            &[rows])
    }

    /// `delta` (the finished sublayer's output) into streams `cur` (written
    /// to the other buffer), then `site`'s input from them.
    fn post_pre(&self, w: &Workspace<'_>, cur: usize, layer: &Qwen4Layer<'_>, site: &str, rows: Scalar) -> Result<()> {
        let [norm, di, up] = Self::site(layer, site)?;
        self.run("qwen4_hc_post_pre", &[("x", w.live(&w.delta, self.cfg.hidden * 2)), ("residual", w.live(&w.streams[cur], HC * self.cfg.hidden * 2)),
            ("inject", w.live(&w.inject, HC * 2)), ("norm", norm), ("w_di", di), ("w_up", up),
            ("residual_out", w.live(&w.streams[cur ^ 1], HC * self.cfg.hidden * 2)), ("y", w.live(&w.x, self.cfg.hidden * 2)), ("scratch", w.scratch.buffer.ptr)],
            &[rows])
    }

    /// `delta` into streams `cur` alone (written to the other buffer).
    fn post(&self, w: &Workspace<'_>, cur: &mut usize, rows: Scalar) -> Result<()> {
        self.run("qwen4_hc_post", &[("x", w.live(&w.delta, self.cfg.hidden * 2)), ("residual", w.live(&w.streams[*cur], HC * self.cfg.hidden * 2)),
            ("inject", w.live(&w.inject, HC * 2)), ("out", w.live(&w.streams[*cur ^ 1], HC * self.cfg.hidden * 2))], &[rows])?;
        *cur ^= 1;
        Ok(())
    }

    /// Late landing, outside graph capture: preceding-layer work is already
    /// queued before waiting for this consumer's rows.
    fn finish_ple(&self, w: &Workspace<'_>) -> Result<()> {
        let (Some(mapped), Some((rows, _))) = (self.ple.as_ref().map(|p| p.mapped()), &w.ple_rows) else {
            return Ok(());
        };
        let pending = self.ple_pending.borrow_mut().take().context("the step's PLE rows were not gathered")?;
        // SAFETY: the engine owns this stream; the workspace's rows outlive the step.
        unsafe { mapped.finish(pending, rows.buffer, self.execution_stream())? };
        Ok(())
    }

    fn ple(&self, w: &Workspace<'_>, streams: &Dev<'_>, layer: &Qwen4Layer<'_>, rows: Scalar, spec: bool)
        -> Result<()> {
        let table = self.ple.as_ref().context("PLE layer without the n-gram table")?;
        let state = self.ple_state.as_ref().context("PLE conv state")?;
        let replay = self.ple_replay.as_ref().context("PLE replay record")?;
        let name = if table.fp8 { "qwen4_ple_fp8" } else { "qwen4_ple_bf16" };
        // Mapped: the step's rows were gathered in id order, so identity ids read them.
        let (gathered, local) = w.ple_rows.as_ref().context("PLE layer without gathered-row workspace")?;
        let (ids, gathered) = (local.buffer.ptr, gathered.buffer.ptr);
        self.run(name, &[("streams", w.live(streams, HC * self.cfg.hidden * 2)), ("ids", ids), ("table", gathered),
            ("scale", layer.ptr("ple.scale")?), ("w_kv", layer.ptr("ple.w_kv")?),
            ("norm_key", layer.ptr("ple.norm_key")?), ("norm_query", layer.ptr("ple.norm_query")?),
            ("norm_conv", layer.ptr("ple.norm_conv")?), ("conv_w", layer.ptr("ple.conv_w")?),
            ("conv_state", state.buffer.ptr), ("slots", w.slots.buffer.ptr), ("seq_first", w.seq_first.buffer.ptr),
            ("replay", replay.buffer.ptr), ("scratch", w.scratch.buffer.ptr)], &[rows, Scalar::I32(i32::from(spec))])?;
        // SAFETY: the PLE program is this destination's final reader.
        unsafe { table.mapped().consumed(self.execution_stream()) }
    }

    /// Layers whose projections are held FP8-only (`--fp8-decode`) take the
    /// `_w8` programs in every step shape; the prefill programs then add the
    /// W8A8 switch (`fp8_rows`).
    fn w8(layer: &Qwen4Layer<'_>) -> bool {
        layer.has("w_in_fp8")
    }

    /// The step scalars of a program over `layer`'s projections: `rows`,
    /// plus `fp8_rows` for FP8-only prefill programs.
    fn w8_scalars(&self, layer: &Qwen4Layer<'_>, rows: Scalar, cap: &str) -> Vec<Scalar> {
        let mut scalars = vec![rows];
        if Self::w8(layer) && !Self::is_decode_cap(cap) {
            scalars.push(Scalar::I32(i32::from(self.w8a8_prefill)));
        }
        scalars
    }

    /// `name`'s BF16 operand, or with FP8-only projections its E4M3 values and scales.
    fn projection(layer: &Qwen4Layer<'_>, name: &'static str) -> Result<Vec<(&'static str, *mut c_void)>> {
        if Self::w8(layer) {
            let (q, s) = cuteafd_loader::families::qwen4::resident::fp8_operands(name).context("projection")?;
            Ok(vec![(q, layer.ptr(q)?), (s, layer.ptr(s)?)])
        } else {
            Ok(vec![(name, layer.ptr(name)?)])
        }
    }

    fn kv_stem(&self, stem: &str) -> String {
        if self.kv_format == cuteafd_loader::families::qwen4::Qwen4KvCache::Fp8 { format!("{stem}_kv_fp8") } else { stem.into() }
    }

    fn kv_program(&self, layer: &Qwen4Layer<'_>, cap: &str) -> String {
        let stem = if Self::w8(layer) { "attn_producer_w8" } else { "attn_producer" };
        format!("qwen4_{}_{cap}", self.kv_stem(stem))
    }

    fn program(layer: &Qwen4Layer<'_>, stem: &str, cap: &str) -> String {
        if Self::w8(layer) { format!("qwen4_{stem}_w8_{cap}") } else { format!("qwen4_{stem}_{cap}") }
    }

    fn gdn(&self, w: &Workspace<'_>, index: usize, layer: &Qwen4Layer<'_>, rows: Scalar, cap: &str, spec: bool)
        -> Result<()> {
        let [conv, state, replay] = self.gdn_pools(index)?;
        let mut pointers = vec![("x", w.live(&w.x, self.cfg.hidden * 2))];
        pointers.extend(Self::projection(layer, "w_in")?);
        pointers.extend([("conv_w", layer.ptr("conv_w")?), ("a_log", layer.ptr("a_log")?),
            ("dt_bias", layer.ptr("dt_bias")?), ("norm_w", layer.ptr("norm_w")?)]);
        pointers.extend(Self::projection(layer, "w_out")?);
        pointers.extend([("conv_state", conv), ("state", state), ("slots", w.row_table(&w.slots, 4)),
            ("seq_first", w.first_table()), ("out", w.live(&w.delta, self.cfg.hidden * 2))]);
        // Decode capacities record speculative replay inputs (spec) or advance the state.
        let decode = Self::is_decode_cap(cap);
        if decode {
            pointers.push(("replay", replay));
        }
        pointers.push(("scratch", w.scratch.buffer.ptr));
        let name = Self::program(layer, "gdn", cap);
        if decode {
            self.run(&name, &pointers, &[rows, Scalar::I32(i32::from(spec))])
        } else {
            ensure!(!spec, "speculative steps take the decode programs");
            self.run(&name, &pointers, &self.w8_scalars(layer, rows, cap))
        }
    }

    fn full(&self, w: &Workspace<'_>, index: usize, layer: &Qwen4Layer<'_>, rows: Scalar, cap: &str,
        tables: &StepTables) -> Result<()> {
        let cache = self.kv[index].as_ref().context("full attention layer without a record pool")?;
        let (keys, blocks) = self.index[index].as_ref().context("full attention layer without an index cache")?;
        let pages = w.attention_tables([0; 2], tables.page_width);
        self.attend(w, layer, cache.buffer.ptr, keys.buffer.ptr, blocks.buffer.ptr, rows, cap, tables, pages)
    }

    fn packed_full(&self, w: &Workspace<'_>, index: usize, layer: &Qwen4Layer<'_>, rows: Scalar,
        tables: &StepTables, pages: AttentionTables) -> Result<()> {
        let cache = self.kv[index].as_ref().context("full attention layer without a record pool")?;
        let (keys, blocks) = self.index[index].as_ref().context("full attention layer without an index cache")?;
        self.attend(w, layer, cache.buffer.ptr, keys.buffer.ptr, blocks.buffer.ptr, rows, "m4096", tables, pages)
    }

    /// Full attention of `layer` over the record pool `cache`, raw index keys
    /// `keys` and pooled block keys `blocks` (a target layer's or the MTP layer's).
    #[allow(clippy::too_many_arguments)]
    fn attend(&self, w: &Workspace<'_>, layer: &Qwen4Layer<'_>, cache: *mut c_void, keys: *mut c_void,
        blocks: *mut c_void, rows: Scalar, cap: &str, tables: &StepTables, pages: AttentionTables) -> Result<()> {
        let mut pointers = vec![("x", w.live(&w.x, self.cfg.hidden * 2))];
        pointers.extend(Self::projection(layer, "w_in")?);
        pointers.extend([("q_norm", layer.ptr("q_norm")?), ("k_norm", layer.ptr("k_norm")?),
            ("iq_norm", layer.ptr("iq_norm")?), ("ik_norm", layer.ptr("ik_norm")?),
            ("positions", w.row_table(&w.positions, 8)), ("rope_positions", w.row_table(&w.rope_positions, 12)),
            ("block_rope_positions", w.row_table(&w.block_rope_positions, 12)), ("kv_slots", w.row_table(&w.kv_slots, 8)),
            ("pool_slots", w.row_table(&w.pool_slots, 8)), ("kv_cache", cache), ("token_keys", keys),
            ("index_cache", blocks), ("query", w.query.buffer.ptr), ("gate", w.gate.buffer.ptr),
            ("index_q", w.index_q.buffer.ptr), ("scratch", w.scratch.buffer.ptr)]);
        self.run(&self.kv_program(layer, cap), &pointers, &self.w8_scalars(layer, rows, cap))?;
        if tables.long {
            self.run(&format!("qwen4_index_topk_{cap}"), &[("index_q", w.index_q.buffer.ptr),
                ("positions", w.row_table(&w.positions, 8)), ("index_cache", blocks),
                ("page_table", pages.pools), ("output_indices", w.blocks.buffer.ptr),
                ("scratch", w.topk_scratch.buffer.ptr)], &[rows, Scalar::I32(tables.pool_stride as i32)])?;
        }
        self.run("qwen4_index_expand", &[("positions", w.row_table(&w.positions, 8)), ("blocks", w.blocks.buffer.ptr),
            ("indices", w.indices.buffer.ptr), ("lengths", w.lengths.buffer.ptr)], &[rows])?;
        self.run(&format!("qwen4_{}_{cap}", self.kv_stem("sparse_gqa")), &[("query", w.query.buffer.ptr), ("kv_cache", cache),
            ("positions", w.row_table(&w.positions, 8)), ("page_table", pages.pages),
            ("indices", w.indices.buffer.ptr), ("out", w.attn.buffer.ptr), ("scratch", w.scratch.buffer.ptr)],
            &[rows, Scalar::I32(pages.width as i32), Scalar::I32(tables.page_stride as i32),
                Scalar::I32(self.attention_plan.rows() as i32)])?;
        let mut pointers = vec![("attn", w.attn.buffer.ptr), ("gate", w.gate.buffer.ptr)];
        pointers.extend(Self::projection(layer, "w_o")?);
        pointers.extend([("out", w.live(&w.delta, self.cfg.hidden * 2)), ("scratch", w.scratch.buffer.ptr)]);
        self.run(&Self::program(layer, "attn_o", cap), &pointers, &self.w8_scalars(layer, rows, cap))
    }

    /// Router, shared expert and routed experts; leaves `bf16(routed + shared)` in `delta`.
    fn moe(&self, w: &Workspace<'_>, index: usize, layer: &Qwen4Layer<'_>, t: usize, rows: Scalar, decode: bool)
        -> Result<()> {
        self.moe_front(w, index, layer, t, rows)?;
        self.moe_experts(w, index, t, rows, decode)
    }

    /// Router logits, the softmax top-10, the shared expert (into `shared`)
    /// and, for wire-fed experts, the FP8 K32 wire rows. No host sync.
    fn moe_front(&self, w: &Workspace<'_>, index: usize, layer: &Qwen4Layer<'_>, t: usize, rows: Scalar)
        -> Result<()> {
        let h = self.cfg.hidden;
        let experts = self.experts.as_ref().with_context(|| format!(
            "layer {index} needs routed experts: pass --local-experts, Spark --peers or --shared-only"))?;
        self.run("qwen4_router_scores", &[("x", w.live(&w.x, self.cfg.hidden * 2)), ("w", layer.ptr("gate")?),
            ("logits", w.router_logits.buffer.ptr)], &[rows])?;
        // SAFETY: logits and route outputs are live buffers of `t` rows.
        unsafe {
            self.library.router_select_softmax(w.router_logits.buffer.ptr, w.route_ids.buffer.ptr,
                w.route_weights.buffer.ptr, t, self.cfg.experts, self.cfg.topk, 1.0, true, self.execution_stream())?;
        }
        if !self.spark_layer(index) { self.shared(w, layer, rows)?; }
        let wire = match experts {
            Experts::Hybrid(experts) if index < self.cfg.layers && experts.local_layer(index) =>
                experts.local.as_ref().context("Qwen resident expert owner")?.wire(),
            Experts::Hybrid(experts) if index == self.cfg.layers => match &experts.mtp {
                Some(MtpExperts::Whole(whole)) => whole.uses_fp8_input(),
                other => matches!(other, Some(MtpExperts::Exl3(_))),
            },
            Experts::Hybrid(_) => !self.spark_bf16,
            Experts::SharedOnly => false,
        };
        if wire {
            let grid = self.quantize_grids[self.active_owner.get()].blocks(t, h);
            self.run("qwen4_expert_input_quant", &[("source_ptr", w.live(&w.x, self.cfg.hidden * 2)), ("values_ptr", w.wire.buffer.ptr),
                // SAFETY: the scale rows follow the payload inside each wire row.
                ("scale_rows_ptr", unsafe { w.wire.buffer.ptr.cast::<u8>().add(h) }.cast()),
                ("scale_mma_ptr", w.live(&w.delta, self.cfg.hidden * 2))], &[rows, Scalar::I32(grid as i32)])?;
        }
        Ok(())
    }

    fn shared(&self, w: &Workspace<'_>, layer: &Qwen4Layer<'_>, rows: Scalar) -> Result<()> {
        self.run("qwen4_shared", &[("x", w.live(&w.x, self.cfg.hidden * 2)), ("w_gate_up", layer.ptr("shared.w_gate_up")?),
            ("w_down", layer.ptr("shared.w_down")?), ("out", w.live(&w.shared, self.cfg.hidden * 2)), ("scratch", w.scratch.buffer.ptr)],
            &[rows])
    }

    /// The routed experts of layer `index` (the front ran); leaves
    /// `bf16(routed + shared)` in `delta`.
    fn moe_experts(&self, w: &Workspace<'_>, index: usize, t: usize, rows: Scalar, decode: bool) -> Result<()> {
        let h = self.cfg.hidden;
        match self.experts.as_ref().context("MoE layer without experts")? {
            Experts::Hybrid(experts) if index < self.cfg.layers => {
                if !experts.local_layer(index) {
                    let mut link = self.spark_link(0).context("Qwen remote layer has no landing link")?.borrow_mut();
                    let exchange = link.exchange()?;
                    if self.active_owner.get() == 0 {
                        let wave = self.wave_send(w, index, t, rows, decode, exchange, None)?;
                        return self.wave_land(w, t, wave, exchange);
                    }
                    let landing = self.workspace_slot(0, decode)?.borrow();
                    let landing = landing.as_ref().context("Qwen Spark landing workspace")?;
                    self.remote_front(1, w, landing, index, t, rows)?;
                    let wave = self.on_owner(0, || self.wave_send(landing, index, t, rows, decode, exchange, None))?;
                    self.on_owner(0, || self.wave_land(landing, t, wave, exchange))?;
                    return self.remote_return(1, landing, w, t);
                }
                self.exchange_window(index, decode, true)?;
                let local = experts.local.as_ref().context("Qwen resident layer has no owner")?;
                let input = if local.wire() { crate::shared::experts::rtx::ExpertInput::Fp8K32(w.wire.buffer.ptr) }
                    else { crate::shared::experts::rtx::ExpertInput::Bf16(w.live(&w.x, h * 2)) };
                let routes = crate::shared::experts::rtx::Routes { ids: w.route_ids.buffer.ptr, weights: w.route_weights.buffer.ptr };
                // SAFETY: owner-local operands persist and the serialized backend combines
                // before another layer/unit can reuse its workspace.
                return unsafe { match local {
                    LocalRouted::Whole(whole) => whole.enqueue(index, t, input, routes,
                        w.live_buffer(&w.shared, h * 2), w.live_buffer(&w.delta, h * 2), self.execution_stream(),
                        &mut || crate::shared::decode_graph::fatal_drain(self.shutdown_submission(), "Qwen whole abort")),
                    LocalRouted::Tp2(pair) => pair.borrow_mut().enqueue(self.active_owner.get(), index, t,
                        match input { crate::shared::experts::rtx::ExpertInput::Fp8K32(ptr)
                            | crate::shared::experts::rtx::ExpertInput::Bf16(ptr) => ptr }, routes,
                        w.live(&w.shared, h * 2), w.live(&w.delta, h * 2)),
                } };
            }
            Experts::Hybrid(experts) => {
                ensure!(index == self.cfg.layers, "unknown Qwen expert layer {index}");
                match experts.mtp.as_ref().context("Qwen independent MTP expert owner missing")? {
                    MtpExperts::Whole(whole) => {
                        // SAFETY: target and MTP share one serialized owner-zero stream,
                        // and the shared scratch/output outlive both consumers.
                        return unsafe { whole.enqueue(index, t, crate::shared::experts::rtx::ExpertInput::Fp8K32(w.wire.buffer.ptr),
                            crate::shared::experts::rtx::Routes { ids: w.route_ids.buffer.ptr, weights: w.route_weights.buffer.ptr },
                            w.live_buffer(&w.shared, h * 2), w.live_buffer(&w.delta, h * 2), self.execution_stream(),
                            &mut || crate::shared::decode_graph::fatal_drain(self.shutdown_submission(), "Qwen MTP abort")) };
                    }
                    MtpExperts::Fp8(experts) => {
                        let resident = experts.index_of(index)?;
                        // SAFETY: independent MTP input/routes/output live on the head-owner stream.
                        unsafe { experts.run(resident, t, w.live(&w.x, h * 2), w.route_ids.buffer.ptr,
                            w.route_weights.buffer.ptr, w.routed.buffer.ptr, self.execution_stream())?; }
                    }
                    MtpExperts::Exl3(experts) => {
                        let mut experts = experts.borrow_mut();
                        // SAFETY: the full-width MTP scratch and operands persist until stream drainage.
                        unsafe {
                            experts.run(crate::families::deepseek_v4::local::LocalLayer::Stage(0), t,
                                w.wire.buffer.ptr, w.route_ids.buffer.ptr, w.route_weights.buffer.ptr,
                                w.live(&w.shared, h * 2), self.execution_stream())?;
                            crate::shared::memory::device::copy_device_ordered(self.library,
                                w.live_buffer(&w.delta, h * 2), experts.output.buffer, t * h * 2, self.execution_stream())?;
                        }
                        return Ok(());
                    }
                }
            }
            Experts::SharedOnly => {
                // SAFETY: both BF16 planes persist on the same ordered owner stream.
                unsafe { crate::shared::memory::device::copy_device_ordered(self.library,
                    w.live_buffer(&w.delta, h * 2), w.live_buffer(&w.shared, h * 2), t * h * 2, self.execution_stream())?; }
                return Ok(());
            }
        }
        self.run("qwen4_add", &[("a", w.routed.buffer.ptr), ("b", w.live(&w.shared, self.cfg.hidden * 2)), ("out", w.live(&w.delta, self.cfg.hidden * 2))],
            &[rows])
    }

    fn remote_front(&self, owner: usize, source: &Workspace<'_>, landing: &Workspace<'_>,
        layer: usize, rows: usize, scalar: Scalar) -> Result<()> {
        ensure!(owner == 1 && self.active_owner.get() == owner, "Qwen remote front owner");
        self.shared(source, &self.weights.layers[layer], scalar)?;
        let Some(Experts::Hybrid(experts)) = &self.experts else { anyhow::bail!("Qwen hybrid missing") };
        experts.with_peer(|peer| {
            // Protect the landing scratch from a previous unit's reader before
            // peer writes it; every edge is on the existing owner streams.
            peer.signal(0, 0)?;
            peer.wait(1, 0)?;
            let input = if self.spark_bf16 { (source.live_buffer(&source.x, self.cfg.hidden * 2),
                landing.live_buffer(&landing.x, self.cfg.hidden * 2)) }
                else { (source.wire.buffer, landing.wire.buffer) };
            let buffers = [input, (source.route_ids.buffer, landing.route_ids.buffer),
                (source.route_weights.buffer, landing.route_weights.buffer),
                (source.live_buffer(&source.shared, self.cfg.hidden * 2), landing.live_buffer(&landing.shared, self.cfg.hidden * 2))];
            let bytes = remote_front_plane_bytes(rows, self.cfg.hidden, self.cfg.topk, self.spark_bf16);
            for ((source, destination), bytes) in buffers.iter().zip(bytes) {
                ensure!(bytes <= source.bytes && bytes <= destination.bytes, "Qwen remote plane exceeds workspace");
            }
            let mut index = 0;
            let planes = buffers.map(|(source, destination)| {
                let bytes = bytes[index] as u64;
                index += 1;
                cuteafd_ffi::peer_exchange::PeerPlane {
                    source: source.ptr, destination: destination.ptr, rows: 1, row_bytes: bytes,
                    source_pitch: bytes, destination_pitch: bytes }
            });
            peer.push_planes(1, 0, &planes)?;
            peer.wait(0, 0)
        })?.context("Qwen remote front needs peer owner")?;
        Ok(())
    }

    fn remote_return(&self, owner: usize, landing: &Workspace<'_>, destination: &Workspace<'_>, rows: usize) -> Result<()> {
        ensure!(owner == 1, "Qwen remote return owner");
        let Some(Experts::Hybrid(experts)) = &self.experts else { anyhow::bail!("Qwen hybrid missing") };
        experts.with_peer(|peer| {
            peer.push_to(0, crate::shared::peer_split::DIRECT, landing.live(&landing.delta, self.cfg.hidden * 2),
                destination.live(&destination.delta, self.cfg.hidden * 2), rows * self.cfg.hidden * 2)?;
            peer.wait(1, crate::shared::peer_split::DIRECT)
        })?.context("Qwen remote return needs peer owner")?;
        Ok(())
    }

    /// A wave on the link's exchange: router outputs and wire rows copy into
    /// the wave slot's pinned staging, the shared expert (and in a decode
    /// step the L2 prefetch) queues behind them, and the worker builds and
    /// posts the request once the copies complete. This thread does not wait
    /// for the GPU. Complete with [`Self::wave_land`].
    fn wave_send(&self, w: &Workspace<'_>, index: usize, t: usize, rows: Scalar, decode: bool,
        exchange: &mut crate::shared::spark_waves::Waves<'_>, origins: Option<crate::shared::prefill_pipeline::ReplyOrigins>) -> Result<crate::shared::spark_waves::WaveTicket> {
        let kind = if decode { ExpertV2SourceKind::Decode } else { ExpertV2SourceKind::Prefill };
        let (h, topk, bf16) = (self.cfg.hidden, self.cfg.topk, self.spark_bf16);
        let layout = crate::shared::experts::staging::RouteStaging::new(t, topk, if bf16 { h * 2 } else { h + h / 32 })?;
        ensure!(exchange.ranks() <= MAX_RANKS, "{} Spark ranks exceed the reduction planes", exchange.ranks());
        let sized = |buffer: cuteafd_ffi::CuteafdDeviceBuffer, bytes: usize| cuteafd_ffi::CuteafdDeviceBuffer { bytes, ..buffer };
        let input = if bf16 { w.live_buffer(&w.x, h * 2) } else { w.wire.buffer };
        let request_id = self.request_id.get();
        self.request_id.set(request_id.checked_add(1).context("Qwen Spark request id overflow")?);
        let position = w.position.get();
        // SAFETY: ids, weights and input rows hold this wave's `t` rows in
        // stream order; the copies read them before later stream work.
        let combined_reply = origins.as_ref().map_or_else(|| crate::shared::spark_waves::single_reply_eligible(t, kind),
            crate::shared::prefill_pipeline::ReplyOrigins::combined_reply_eligible);
        let ticket = unsafe {
            exchange.send(t, exchange.compact_payload(t), layout, [sized(w.route_ids.buffer, layout.route_bytes),
                sized(w.route_weights.buffer, layout.route_bytes), sized(input, layout.wire_bytes)],
                self.execution_stream(), combined_reply, Box::new(move |routes, wire| {
                    let mut request = ExpertProtocolV2Request::new_bytes(request_id, 17, index as u32, h as u32,
                        if bf16 { ExpertV2Dtype::Bf16 } else { ExpertV2Dtype::Fp8E4m3Ue8m0K32 },
                        (0..t as u32).map(|row| ExpertProtocolV2RowDescriptor {
                            row_id: u64::from(row), source_kind: kind, source_request_id: 1,
                            token_position: (position + row as usize) as u64, route_offset: row * topk as u32,
                            route_count: topk as u32,
                        }).collect(),
                        routes, wire)?;
                    if let Some(origins) = origins { origins.apply(&mut request)?; }
                    request.header.flags |= EXPERT_PROTOCOL_V2_FLAG_V41_COMPACT_BF16;
                    Ok(request)
                }))?
        };
        // The shared expert runs during the exchange (the L2 prefetch behind it).
        if self.state_map.layers[index].owner == self.active_owner.get() {
            self.shared(w, &self.weights.layers[index], rows)?;
        }
        self.exchange_window(index, decode, false)?;
        Ok(ticket)
    }

    /// The land half of [`Self::wave_send`]: the stream waits for the wave's
    /// planes and sums them with the shared expert into `delta`.
    fn wave_land(&self, w: &Workspace<'_>, t: usize, wave: crate::shared::spark_waves::WaveTicket,
        exchange: &mut crate::shared::spark_waves::Waves<'_>) -> Result<()> {
        let slot = wave.slot();
        exchange.land(wave, self.execution_stream())?;
        // SAFETY: the shared-expert plane and `delta` are live [t, h] BF16
        // buffers; the slot's planes are ordered after the wave by `land`.
        unsafe {
            exchange.reduce(slot, w.live(&w.shared, self.cfg.hidden * 2).cast(),
                w.live(&w.delta, self.cfg.hidden * 2).cast(), t, self.execution_stream())
        }
    }

    /// Bytes of the widest prefill wave's pinned staging (routes and wire rows).
    fn wave_staging_bytes(&self, rows: usize) -> Result<usize> {
        let h = self.cfg.hidden;
        Ok(crate::shared::experts::staging::RouteStaging::new(rows, self.cfg.topk,
            if self.spark_bf16 { h * 2 } else { h + h / 32 })?.bytes)
    }
}

/// A queued pack uses the shared executor; its immutable metadata is uploaded once.
struct PackedHooks<'e, 'w, 't, 'a, 'r, 'n> {
    engine: &'e Qwen4Engine<'a>,
    w: &'w Workspace<'a>,
    tables: &'t [StepTables],
    segments: &'t [crate::shared::prefill_packing::Segment],
    views: &'t [AttentionTables],
    transport: Option<&'r mut crate::shared::spark_waves::Waves<'n>>,
    cur: usize,
    entered: bool,
}

impl PackedHooks<'_, '_, '_, '_, '_, '_> {
    fn aggregate(&self) {
        self.w.row_offset.set(0);
        self.w.table_view.set(None);
        self.w.position.set(0);
    }
    fn finish(&mut self, layer: usize, rows: Scalar) -> Result<()> {
        self.aggregate();
        if let Some(next) = self.engine.weights.layers.get(layer + 1) {
            let ple = self.engine.cfg.ple_layers.contains(&(layer + 1));
            if ple {
                self.engine.post(self.w, &mut self.cur, rows)?;
                self.engine.finish_ple(self.w)?;
            }
            self.engine.enter(self.w, &mut self.cur, (!ple).then_some(()), next, layer + 1, rows, false)
        } else { self.engine.post(self.w, &mut self.cur, rows) }
    }
}

impl crate::shared::prefill_pipeline::PrefillHooks for PackedHooks<'_, '_, '_, '_, '_, '_> {
    type Ticket = crate::shared::spark_waves::WaveTicket;
    fn route(&self, layer: usize) -> crate::shared::prefill_pipeline::FfnRoute {
        if self.engine.spark_layer(layer) { crate::shared::prefill_pipeline::FfnRoute::Spark }
        else { crate::shared::prefill_pipeline::FfnRoute::Local }
    }
    fn begin_unit(&mut self, _: &crate::shared::prefill_pipeline::Unit) -> Result<()> { Ok(()) }
    fn attention(&mut self, index: usize, unit: &crate::shared::prefill_pipeline::Unit) -> Result<()> {
        let rows = Scalar::I32(unit.rows.len() as i32);
        if !self.entered {
            if self.engine.cfg.ple_layers.contains(&0) { self.engine.finish_ple(self.w)?; }
            self.engine.enter(self.w, &mut self.cur, None, &self.engine.weights.layers[0], 0, rows, false)?;
            self.entered = true;
        }
        let layer = &self.engine.weights.layers[index];
        for ((s, table), &view) in self.segments.iter().zip(self.tables).zip(self.views) {
            self.w.row_offset.set(s.first); self.w.position.set(s.start); self.w.table_view.set(Some(s.first));
            let n = Scalar::I32(s.rows as i32);
            match layer.attention {
                Qwen4Attention::Gdn => self.engine.gdn(self.w, index, layer, n, "m4096", false)?,
                Qwen4Attention::Full => self.engine.packed_full(self.w, index, layer, n, table, view)?,
            }
        }
        self.aggregate();
        self.engine.post_pre(self.w, self.cur, layer, "mlp", rows)?;
        self.cur ^= 1;
        Ok(())
    }
    fn local_ffn(&mut self, index: usize, unit: &crate::shared::prefill_pipeline::Unit) -> Result<()> {
        let rows = Scalar::I32(unit.rows.len() as i32);
        self.engine.moe(self.w, index, &self.engine.weights.layers[index], unit.rows.len(), rows, false)?;
        self.finish(index, rows)
    }
    fn send_experts(&mut self, index: usize, unit: &crate::shared::prefill_pipeline::Unit) -> Result<Self::Ticket> {
        let rows = Scalar::I32(unit.rows.len() as i32);
        self.engine.moe_front(self.w, index, &self.engine.weights.layers[index], unit.rows.len(), rows)?;
        self.engine.wave_send(self.w, index, unit.rows.len(), rows, false,
            self.transport.as_mut().context("packed Spark exchange")?, unit.reply_origins()?)
    }
    fn land_experts(&mut self, index: usize, unit: &crate::shared::prefill_pipeline::Unit, ticket: Self::Ticket) -> Result<()> {
        self.engine.wave_land(self.w, unit.rows.len(), ticket, self.transport.as_mut().context("packed Spark exchange")?)?;
        self.finish(index, Scalar::I32(unit.rows.len() as i32))
    }
    fn end_unit(&mut self, _: &crate::shared::prefill_pipeline::Unit) -> Result<()> { self.aggregate(); Ok(()) }
    fn commit_unit(&mut self, _: &crate::shared::prefill_pipeline::Unit) -> Result<()> { Ok(()) }
}

/// Live rows survive hooks in slot planes; all tables and scratch are shared
/// and rewritten on the execution stream before the hook that consumes them.
struct UnitHooks<'e, 'w, 'p, 't, 'a, 'r, 'n> {
    engine: &'e Qwen4Engine<'a>,
    transports: Vec<&'r mut crate::shared::spark_waves::Waves<'n>>,
    work: [Option<&'w Workspace<'a>>; 2],
    placement: &'p Qwen4Placement,
    tokens: &'t [u32],
    tables: Arc<Vec<StepTables>>,
    ple_batch: Option<Arc<crate::shared::mapped_table::RowBatch>>,
    current: Vec<[usize; 2]>,
    entered: Vec<bool>,
    pending: Vec<Option<crate::shared::mapped_table::PendingRows>>,
    logit_rows: usize,
    scoring: Option<crate::shared::probe::ScoringRows<'t>>,
}

impl<'a, 'w> UnitHooks<'_, 'w, '_, '_, 'a, '_, '_> {
    fn owner(&self, layer: usize) -> usize { self.engine.state_map.layers[layer].owner }
    fn workspace(&self, owner: usize) -> Result<&'w Workspace<'a>> { self.work[owner].context("Qwen unit owner workspace") }
    fn select(&self, owner: usize, unit: &crate::shared::prefill_pipeline::Unit) -> Result<()> {
        let shape = self.engine.prefill_shape.context("Qwen prefill shape")?;
        let w = self.workspace(owner)?;
        w.row_offset.set(Workspace::unit_offset(unit.slot, shape.unit_rows as usize));
        w.position.set(self.tables[unit.index].positions[0] as usize);
        self.engine.stage_step_tables(w, &self.tables[unit.index])
    }
    fn enter(&mut self, index: usize, unit: &crate::shared::prefill_pipeline::Unit, posted: bool) -> Result<()> {
        let owner = self.owner(index);
        let engine = self.engine;
        engine.on_owner(owner, || {
            // The current hook (or cutover hop) already selected this unit.
            // Entering the next same-owner layer must not upload its tables again.
            if engine.cfg.ple_layers.contains(&index) {
                *engine.ple_pending.borrow_mut() = self.pending[unit.index].take();
                engine.finish_ple(self.workspace(owner)?)?;
            }
            let w = self.work[owner].context("Qwen unit entry workspace")?;
            engine.enter(w, &mut self.current[unit.index][owner], posted.then_some(()),
                &engine.weights.layers[index], index, Scalar::I32(unit.rows.len() as i32), false)
        })?;
        self.entered[unit.index] = true;
        Ok(())
    }
    fn hop(&self, ordinal: usize, from: usize, to: usize, unit: &crate::shared::prefill_pipeline::Unit) -> Result<()> {
        let engine = self.engine;
        let src = self.workspace(from)?;
        let dst = self.workspace(to)?;
        let stride = HC * engine.cfg.hidden * 2;
        let hops = engine.hops.as_ref().context("Qwen unit residual hops")?;
        let ticket = hops.send(ordinal, 0, src.live(&src.streams[self.current[unit.index][from]], stride), unit.rows.len())?;
        let landed = hops.land(ticket)?;
        engine.on_owner(to, || {
            self.select(to, unit)?;
            let output = dst.live_buffer(&dst.streams[self.current[unit.index][to]], stride);
            let input = cuteafd_ffi::CuteafdDeviceBuffer { ptr: landed, ..output };
            // SAFETY: the admitted hop wait precedes the owner-local slot copy.
            unsafe { crate::shared::memory::device::copy_device_ordered(engine.library, output, input,
                unit.rows.len() * stride, engine.execution_stream()) }
        })
    }
    fn finish_ffn(&mut self, layer: usize, unit: &crate::shared::prefill_pipeline::Unit) -> Result<()> {
        let owner = self.owner(layer);
        let next = (layer + 1 < self.engine.weights.layers.len()).then_some(layer + 1);
        let materialize = next.is_none_or(|next| self.owner(next) != owner || self.engine.cfg.ple_layers.contains(&next));
        if materialize {
            self.engine.on_owner(owner, || self.engine.post(self.work[owner].context("Qwen unit post workspace")?,
                &mut self.current[unit.index][owner], Scalar::I32(unit.rows.len() as i32)))?;
        }
        if let Some(next) = next {
            if self.owner(next) != owner { self.hop(0, owner, self.owner(next), unit)?; }
            self.enter(next, unit, !materialize)
        } else {
            if owner != 0 { self.hop(1, owner, 0, unit)?; }
            Ok(())
        }
    }
}

impl crate::shared::prefill_pipeline::PrefillHooks for UnitHooks<'_, '_, '_, '_, '_, '_, '_> {
    type Ticket = crate::shared::spark_waves::WaveTicket;
    fn route(&self, layer: usize) -> crate::shared::prefill_pipeline::FfnRoute {
        if self.engine.spark_layer(layer) { crate::shared::prefill_pipeline::FfnRoute::Spark }
        else { crate::shared::prefill_pipeline::FfnRoute::Local }
    }
    fn begin_unit(&mut self, unit: &crate::shared::prefill_pipeline::Unit) -> Result<()> {
        let engine = self.engine;
        if let Some(mapped) = engine.ple.as_ref().map(|ple| ple.mapped()) {
            let batch = self.ple_batch.as_ref().context("Qwen PLE prefill batch")?;
            for index in unit.index..(unit.index + mapped.depth()).min(self.pending.len()) {
                if self.pending[index].is_none() { self.pending[index] = Some(mapped.begin_unit(batch, index, false)?); }
            }
        }
        engine.on_owner(0, || {
            self.select(0, unit)?;
            let w = self.workspace(0)?;
            engine.stage_embedding(w, &self.tokens[unit.rows.clone()], HC, &w.streams[0], false)?;
            let start = self.tables[unit.index].positions[0] as usize;
            let media = embedding_media(&[(self.placement, start, unit.rows.len())])?;
            engine.inject_media(w, &media, &self.tables[unit.index].seq_first, &w.streams[0], unit.rows.len(), HC)
        })
    }
    fn attention(&mut self, layer: usize, unit: &crate::shared::prefill_pipeline::Unit) -> Result<()> {
        let owner = self.owner(layer);
        self.engine.on_owner(owner, || {
            self.select(owner, unit)?;
            if !self.entered[unit.index] { self.enter(layer, unit, false)?; }
            let w = self.work[owner].context("Qwen unit attention workspace")?;
            let rows = Scalar::I32(unit.rows.len() as i32);
            let weights = &self.engine.weights.layers[layer];
            match weights.attention {
                Qwen4Attention::Gdn => self.engine.gdn(w, layer, weights, rows, "m4096", false)?,
                Qwen4Attention::Full => self.engine.full(w, layer, weights, rows, "m4096", &self.tables[unit.index])?,
            }
            self.engine.post_pre(w, self.current[unit.index][owner], weights, "mlp", rows)?;
            self.current[unit.index][owner] ^= 1;
            Ok(())
        })
    }
    fn local_ffn(&mut self, layer: usize, unit: &crate::shared::prefill_pipeline::Unit) -> Result<()> {
        let owner = self.owner(layer);
        self.engine.on_owner(owner, || {
            self.select(owner, unit)?;
            self.engine.moe(self.workspace(owner)?, layer, &self.engine.weights.layers[layer],
                unit.rows.len(), Scalar::I32(unit.rows.len() as i32), false)
        })?;
        self.finish_ffn(layer, unit)
    }
    fn send_experts(&mut self, layer: usize, unit: &crate::shared::prefill_pipeline::Unit) -> Result<Self::Ticket> {
        let owner = self.owner(layer);
        self.engine.on_owner(owner, || {
            self.select(owner, unit)?;
            let w = self.work[owner].context("Qwen unit send workspace")?;
            let rows = Scalar::I32(unit.rows.len() as i32);
            self.engine.moe_front(w, layer, &self.engine.weights.layers[layer], unit.rows.len(), rows)?;
            if owner == 1 {
                self.engine.on_owner(0, || self.select(0, unit))?;
                self.engine.remote_front(owner, w, self.work[0].context("Qwen unit landing workspace")?,
                    layer, unit.rows.len(), rows)?;
            }
            self.engine.on_owner(0, || self.engine.wave_send(self.work[0].context("Qwen unit landing workspace")?,
                layer, unit.rows.len(), rows, false, self.transports.get_mut(0)
                    .context("Qwen unit Spark landing link")?, unit.reply_origins()?))
        })
    }
    fn land_experts(&mut self, layer: usize, unit: &crate::shared::prefill_pipeline::Unit, ticket: Self::Ticket) -> Result<()> {
        let owner = self.owner(layer);
        self.engine.on_owner(owner, || {
            self.select(owner, unit)?;
            let w = self.work[owner].context("Qwen unit land workspace")?;
            if owner == 1 { self.engine.on_owner(0, || self.select(0, unit))?; }
            self.engine.on_owner(0, || self.engine.wave_land(self.work[0].context("Qwen unit landing workspace")?,
                unit.rows.len(), ticket, self.transports.get_mut(0).context("Qwen unit Spark landing link")?))?;
            if owner == 1 { self.engine.remote_return(owner, self.work[0].context("Qwen unit landing workspace")?, w, unit.rows.len())?; }
            Ok(())
        })?;
        self.finish_ffn(layer, unit)
    }
    fn end_unit(&mut self, unit: &crate::shared::prefill_pipeline::Unit) -> Result<()> {
        let engine = self.engine;
        engine.on_owner(0, || {
            self.select(0, unit)?;
            let w = self.workspace(0)?;
            let cur = self.current[unit.index][0];
            if let Some(terminal) = &w.terminal_streams {
                let stride = HC * engine.cfg.hidden * 2;
                let bytes = unit.rows.len() * stride;
                let src = w.live_buffer(&w.streams[cur], stride);
                let dst = Qwen4Engine::region(terminal, unit.rows.start * stride, bytes);
                // SAFETY: unit owns its slot and distinct admitted terminal span.
                unsafe { crate::shared::memory::device::copy_device_ordered(engine.library, dst, src, bytes, engine.execution_stream())?; }
            }
            let first = self.tokens.len() - self.logit_rows;
            let count = unit.rows.end.saturating_sub(unit.rows.start.max(first));
            if count > 0 && engine.weights.layers.len() == engine.cfg.layers {
                let offset = unit.rows.start.max(first) - first;
                if let Some(scoring) = self.scoring {
                    engine.record_head(w, &w.streams[cur], unit.rows.len(), count,
                        crate::shared::probe::ScoringRows { position: scoring.position + offset, ..scoring })?;
                } else {
                    w.logit_offset.set(offset);
                    engine.head(w, &w.streams[cur], unit.rows.len(), Scalar::I32(unit.rows.len() as i32), count)?;
                }
            }
            engine.last_streams.set((false, cur));
            Ok(())
        })
    }
    fn commit_unit(&mut self, _: &crate::shared::prefill_pipeline::Unit) -> Result<()> { Ok(()) }
}

fn packed_upload_error(error: anyhow::Error) -> anyhow::Error {
    use crate::shared::experts::rtx::pair::{RecoverableStepError, Recovery};
    if error.is::<crate::shared::token_table::UploadPressure>() {
        RecoverableStepError::new(Recovery::PackedMetadata, error.to_string()).into()
    } else {
        error
    }
}

fn validate_prefill_span(rows: usize, start: usize, admitted_span: usize, max_context: usize) -> Result<()> {
    ensure!(rows > 0 && rows <= admitted_span
        && start.checked_add(rows).is_some_and(|end| end <= max_context),
        "prefill of {rows} rows at {start} exceeds admitted span {admitted_span} or context {max_context}");
    Ok(())
}

#[cfg(test)]
mod gather_recovery_tests {
    /// #1817: every Qwen submission boundary takes the request-local path,
    /// and host-gather recovery is refused unless one owner holds no peer or
    /// Spark work (TP2 and Spark retire terminally; #1842).
    #[test]
    fn qwen_boundaries_recover_host_gathers_only_without_peer_or_spark_work() {
        let source = include_str!("engine.rs");
        let body = source.split("\n#[cfg(test)]\nmod gather_recovery_tests").next().unwrap();
        assert!(!body.contains("pair::submit(") && !body.contains("pair::submit_scoped("));
        assert_eq!(body.matches("pair::submit_recoverable(|| self.require_live(), |kind| self.cancel_submission(kind), || self.shutdown_submission(),").count(), 4);
        let cancel = body.split("    fn cancel_submission(").nth(1).unwrap().split("    pub(crate) fn drain_state(").next().unwrap();
        assert!(cancel.contains("self.gdn_banks.len() == 1 && self.hops.is_none()"));
        assert!(cancel.contains("!self.has_sparks()"));
        // Drop tickets, drain the consumers, then the table's own copies, then the graph boundary.
        let order = ["self.ple_pending.borrow_mut().take()", "self.drain_state()?", "ple.mapped().drain()?",
            "w.terminal_valid.set(false)", "cycles.abandon()"];
        let at: Vec<_> = order.iter().map(|needle| cancel.find(needle).unwrap_or_else(|| panic!("{needle}"))).collect();
        assert!(at.windows(2).all(|pair| pair[0] < pair[1]), "cancellation order {at:?}");
    }

    #[test]
    fn qwen_graph_boundary_records_without_a_host_drain() {
        let source = include_str!("engine.rs");
        let completed = source.split("    fn graph_completed(").nth(1).unwrap()
            .split("    /// Attribute the whole step's replay work").next().unwrap();
        assert!(completed.contains("cycles.complete()"));
        assert!(!completed.contains("drain_state") && !completed.contains("observe_completed"));
        let step = source.split("    fn step(").nth(1).unwrap().split("    fn step_inner(").next().unwrap();
        assert!(step.find("self.begin_graph_cycle()?").unwrap() < step.find("self.step_inner(").unwrap());
        assert!(step.find("self.step_inner(").unwrap() < step.find("self.graph_completed()?").unwrap());
    }
}

#[cfg(test)]
mod prefill_unit_tests {
    use super::*;
    use crate::shared::prefill_pipeline::{prefill_schedule, sequence_units, FfnRoute, Step};
    use std::ops::Range;

    #[test]
    fn streamed_head_tiles_respect_the_existing_one_row_plane() -> Result<()> {
        for rows in [1usize, 63, 64, 65, 511, 4096, 8192] {
            for wanted in [1, rows.div_ceil(2), rows] {
                for capacity in [1, 64] {
                    let first = rows - wanted;
                    let mut positions = Vec::new();
                    for unit in sequence_units(rows, 2048, 1, 2)? {
                        let begin = unit.rows.start.max(first);
                        let count = unit.rows.end.saturating_sub(begin);
                        if count == 0 { continue; }
                        for span in crate::shared::probe::head_spans(count, capacity)? {
                            assert!(span.len() <= capacity);
                            positions.extend(span.map(|row| 1000 + begin - first + row));
                        }
                    }
                    assert_eq!(positions, (1000..1000 + wanted).collect::<Vec<_>>());
                }
            }
        }
        Ok(())
    }

    #[test]
    fn remote_front_planes_align_odd_decode_and_prefill_rows() {
        for rows in [1usize, 41, 2752] {
            for bf16 in [false, true] {
                let bytes = remote_front_plane_bytes(rows, 2560, 10, bf16);
                assert!(bytes.iter().all(|bytes| bytes % 16 == 0));
                assert_eq!(bytes[1], (rows * 40).next_multiple_of(16));
                assert_eq!(bytes[2], bytes[1]);
                assert!(bytes[1] >= rows * 40 && bytes[1] < rows * 40 + 16);
            }
        }
    }

    #[test]
    fn dual_hybrid_uses_one_spark_link_on_the_landing_owner() {
        let source = include_str!("hybrid.rs");
        let links = source.split_once("        let mut transport = Vec::new();").unwrap().1
            .split_once("        Ok(engine::HybridExperts").unwrap().0;
        assert_eq!(links.matches("self.spark_link(args, &peers)").count(), 1);
        assert!(links.contains("devices[0].run("));
        assert!(!links.contains("for device in devices"));
        assert_eq!(cuteafd_loader::plan::layout::spark_endpoints("qwen4", 1), 1);
        let source = include_str!("engine.rs");
        let begin = source.split_once("    fn remote_front(").unwrap().1.split_once("    fn remote_return(").unwrap().0;
        assert!(begin.contains("peer.push_planes(1, 0"));
        assert!(begin.contains("peer.wait(0, 0)"));
        assert!(!begin.contains("synchronize"));
    }

    #[test]
    fn single_owner_hooks_do_not_switch_the_cuda_device() {
        let source = include_str!("engine.rs");
        let owner = source.split_once("    fn on_owner<T>").unwrap().1
            .split_once("    fn workspace_slot(").unwrap().0;
        let single = owner.find("if self.gdn_banks.len() == 1 { return work(); }").unwrap();
        assert!(owner.find("self.gdn_banks.get(owner)").unwrap() < single,
            "invalid owner IDs must still refuse");
        assert!(single < owner.find("self.active_owner.replace(owner)").unwrap());
        assert!(single < owner.find("bank.device.run(work)").unwrap(),
            "single-device nested hooks must bypass CUDA get/set calls");
    }

    #[test]
    fn all_spark_units_upload_tables_only_at_the_original_five_hook_sites() {
        let source = include_str!("engine.rs");
        let hooks = source.split_once("impl<'a, 'w> UnitHooks<").unwrap().1
            .split_once("fn validate_prefill_span(").unwrap().0;
        let enter = hooks.split_once("    fn enter(").unwrap().1.split_once("    fn hop(").unwrap().0;
        assert!(!enter.contains("self.select("), "same-owner enter must not upload tables");
        for name in ["begin_unit", "attention", "send_experts", "land_experts", "end_unit"] {
            let body = hooks.split_once(&format!("    fn {name}(")).unwrap().1
                .split_once("\n    fn ").map_or_else(|| hooks.split_once(&format!("    fn {name}(")).unwrap().1,
                    |(body, _)| body);
            let remote_only = body.matches("|| self.select(0, unit)").count();
            assert_eq!(body.matches("self.select(").count() - remote_only, 1, "{name}");
        }
        let attention = hooks.split_once("    fn attention(").unwrap().1.split_once("    fn local_ffn(").unwrap().0;
        assert!(attention.find("self.select(").unwrap() < attention.find("self.enter(").unwrap());
        // Dev all-Spark: one begin/end per unit, attention/send/land per layer.
        // Pin each unit's ordered upload sites, including a fourth, 41-row tail.
        for (span, lengths) in [(8192, vec![2752, 2752, 2688]),
            (8297, vec![2752, 2752, 2752, 41])] {
            for waves in [1, 2, 3] {
                let mut units = sequence_units(span, 2752, 1, 48).unwrap();
                for unit in &mut units { unit.slot = unit.index % waves; }
                assert_eq!(units.iter().map(|unit| unit.rows.len()).collect::<Vec<_>>(), lengths);
                let schedule = prefill_schedule(&vec![FfnRoute::Spark; 48], &units, waves).unwrap();
                let mut actual = vec![Vec::new(); units.len()];
                for step in schedule {
                    let (unit, site, layer) = match step {
                        Step::Begin { unit } => (unit, "begin", None),
                        Step::Attention { unit, layer } => (unit, "attention", Some(layer)),
                        Step::Send { unit, layer } => (unit, "send", Some(layer)),
                        Step::Land { unit, layer } => (unit, "land", Some(layer)),
                        Step::End { unit } => (unit, "end", None),
                        Step::Commit { .. } => continue,
                        Step::Local { .. } => panic!("all-Spark unit dispatched a local FFN"),
                    };
                    actual[unit].push((site, layer, units[unit].rows.len()));
                }
                for (unit, sites) in actual.iter().enumerate() {
                    let rows = lengths[unit];
                    let mut expected = vec![("begin", None, rows)];
                    for layer in 0..48 {
                        expected.extend([("attention", Some(layer), rows),
                            ("send", Some(layer), rows), ("land", Some(layer), rows)]);
                    }
                    expected.push(("end", None, rows));
                    assert_eq!(*sites, expected, "span={span} waves={waves} unit={unit}");
                }
                assert_eq!(actual.iter().map(Vec::len).sum::<usize>(),
                    if span == 8297 { 584 } else { 438 });
            }
        }
    }

    #[test]
    fn streamed_scoring_preserves_units_and_consumes_head_before_reuse() {
        let source = include_str!("engine.rs");
        let units = source.split_once("    fn prefill_units_with(").unwrap().1
            .split_once("    pub(crate) fn configure_packing").unwrap().0;
        assert!(units.contains("let retained_rows = if scoring.is_some() { 1 } else { logit_rows }"));
        assert!(units.contains("sequence_units(t, shape.unit_rows as usize, 1, self.weights.layers.len())"));
        let head = source.split_once("    fn record_head(").unwrap().1.split_once("    /// FP32 logits").unwrap().0;
        assert!(head.contains("head_spans(logit_rows, w.logit_rows.min(DECODE_ROWS))"));
        assert!(head.find("self.mix_head").unwrap() < head.find("head_spans").unwrap());
        assert!(head.find("self.logits(").unwrap() < head.find("scoring.record(").unwrap());
        assert!(!head.contains("workspace("));
        let hooks = source.split_once("struct UnitHooks<").unwrap().1
            .split_once("fn validate_prefill_span(").unwrap().0;
        assert!(hooks.contains("engine.record_head(w, &w.streams[cur]"));
    }

    #[test]
    fn unit_tables_keep_absolute_addresses_and_reset_sequence_rows() -> Result<()> {
        for (start, count) in [0usize, 63, 255, 1023, 2047, 4093].into_iter()
            .flat_map(|start| [4096, 8192, 8256].map(|count| (start, count))) {
            let ple_rows = 16;
            let mut whole = StepTables { page_table: vec![9, 4, 7, 2], pool_table: vec![3, 8],
                page_width: 4, pool_width: 2, ..Default::default() };
            for row in 0..count {
                let position = start + row;
                whole.positions.push(position as i64);
                whole.rope_positions.push([position as i32, 2 * position as i32, 3 * position as i32]);
                whole.block_rope_positions.push([(position / BLOCK * BLOCK) as i32; 3]);
                whole.kv_slots.push((700 + position) as i64);
                whole.pool_slots.push(if position % BLOCK == BLOCK - 1 { (position / BLOCK) as i64 } else { -1 });
                whole.slots.push(5);
                whole.seq_first.push(0);
                whole.cache_lengths.push(((position + 1) / BLOCK) as i32);
                whole.ple_ids.extend((0..ple_rows).map(|head| (position * ple_rows + head) as i64));
            }
            for width in [512, 1024, 2048, 2752, 4096] {
                for unit in sequence_units(count, width, 1, 48)? {
                    let rows = unit.rows;
                    let local = whole.unit(rows.clone(), ple_rows, 2051);
                    assert_eq!(local.positions, whole.positions[rows.clone()]);
                    assert_eq!(local.rope_positions, whole.rope_positions[rows.clone()]);
                    assert_eq!(local.block_rope_positions, whole.block_rope_positions[rows.clone()]);
                    assert_eq!(local.kv_slots, whole.kv_slots[rows.clone()]);
                    assert_eq!(local.pool_slots, whole.pool_slots[rows.clone()]);
                    assert_eq!(local.slots, whole.slots[rows.clone()]);
                    assert_eq!(local.cache_lengths, whole.cache_lengths[rows.clone()]);
                    assert_eq!(local.ple_ids, whole.ple_ids[rows.start * ple_rows..rows.end * ple_rows]);
                    assert_eq!(local.page_table, whole.page_table);
                    assert_eq!(local.pool_table, whole.pool_table);
                    assert_eq!((local.page_width, local.pool_width), (whole.page_width, whole.pool_width));
                    assert_eq!((local.page_stride, local.pool_stride), (0, 0));
                    assert_eq!(local.long, local.positions.iter().any(|&p| p + 1 > 2051));
                    assert_eq!(local.seq_first, vec![0; rows.len()]);
                    // A whole-step view would leave negative first-row indices
                    // at every nonzero cut; GDN/PLE must see a new local row 0.
                    if rows.start > 0 { assert_ne!(local.seq_first[0], whole.seq_first[rows.start] - rows.start as i32); }
                    assert!(!local.decode && !local.spec);
                }
            }
        }
        Ok(())
    }

    #[test]
    fn prefill_workspace_allocations_equal_shared_admission_for_all_wave_counts() -> Result<()> {
        use cuteafd_loader::serving_capacity::{PrefillShape, qwen_graphs::{qwen_admission, QwenAdmissionInputs}};
        let mut config = cuteafd_loader::plan::testing::qwen4_config(48);
        config["text_config"]["ple_layer_ids"] = serde_json::json!([2]);
        config["text_config"]["ple_embed_dim"] = 2560.into();
        config["text_config"]["ngram_size"] = 3.into();
        config["text_config"]["heads_per_ngram"] = 8.into();
        config["text_config"]["mtp_num_hidden_layers"] = 1.into();
        let cfg = Qwen4Config::from_hf(&config)?;
        let (h, heads, hd, topk) = (cfg.hidden as u64, cfg.heads as u64, cfg.head_dim as u64, cfg.topk as u64);
        for (rows, waves) in [(4096, 1), (4096, 2), (2752, 3), (2048, 4), (512, 4)] {
            let shape = PrefillShape { unit_rows: rows, waves }.validate()?;
            for mtp in [false, true] {
                for mapped in [None, Some(160u64), Some(320)] {
                    let span = shape.slot_rows();
                    // Mirror workspace_rows allocations, independently of the loader's
                    // workspace formula: live planes use span, all hook scratch uses rows.
                    let mut buffers = vec![span * 4 * h * 2, span * 4 * h * 2, span * 4 * 2,
                        span * h * 2, span * h * 2, span * h * 2, rows * h * 2,
                        rows * heads * hd * 2, rows * heads * hd * 2,
                        rows * cfg.index_heads as u64 * INDEX_DIM as u64 * 2, rows * heads * hd * 2,
                        rows * (cfg.index_budget / BLOCK) as u64 * 4, rows * SPARSE_TOPK as u64 * 4,
                        rows * 4, 0, 0, cfg.vocab_size as u64 * 4,
                        rows * cfg.experts as u64 * 4, rows * topk * 4, rows * topk * 4,
                        rows * (h + h / 32), rows * 4, 8, 8, VOCABULARY_HEAD_WORKSPACE as u64];
                    let tables = super::super::admission::StepTableBytes::new(rows as usize, false, 0, 0, cfg.ple_rows())?;
                    buffers.extend([tables.positions, tables.rope_positions, tables.block_rope_positions,
                        tables.kv_slots, tables.slots, tables.seq_first, tables.pool_slots,
                        tables.cache_lengths, tables.page_table, tables.pool_table, tables.ple_ids, tables.ids]
                        .map(|bytes| bytes as u64));
                    buffers.push(rows * 4); // Immutable sequence-local zero origins for packed attention.
                    if let Some(width) = mapped {
                        buffers.extend([rows * cfg.ple_rows() as u64 * 8, rows * cfg.ple_rows() as u64 * width]);
                    }
                    if mtp { buffers.push(span * 4 * h * 2); }
                    let runtime = buffers.into_iter().map(|bytes| bytes.max(256)).sum::<u64>();
                    let admitted = qwen_admission(&QwenAdmissionInputs {
                        cfg: &cfg, layers: 48, mtp,
                        kv_format: cuteafd_loader::families::qwen4::Qwen4KvCache::Fp8, manifest: None,
                        prefill_rows: span, prefill_shape: Some(shape), slots: 8, mark_bytes: 0,
                        ple: mapped.map(|width| (width, width == 160)),
                        future_expert_bytes: 0, headroom: 0, decode_rows: 64,
                    })?;
                    let planned = admitted.items.iter().find(|item| item.1 == "prefill step").unwrap().2;
                    assert_eq!(planned, runtime, "{shape:?}, MTP {mtp}, PLE {mapped:?}");
                }
            }
        }
        Ok(())
    }

    #[test]
    fn diagnostic_prefill_cannot_exceed_its_admitted_terminal_and_logit_span() -> Result<()> {
        let admitted = 16384;
        validate_prefill_span(admitted, 7, admitted, 32768)?;
        assert!(validate_prefill_span(admitted + 1, 7, admitted, 32768).is_err());
        assert!(validate_prefill_span(512, usize::MAX, admitted, usize::MAX).is_err());
        assert!(validate_prefill_span(0, 0, admitted, 32768).is_err());
        Ok(())
    }

    const LIVE: [&str; 6] = ["streams0", "streams1", "inject", "x", "delta", "shared"];

    /// Replay the executor's actual order with Qwen's post/pre writes in
    /// landing, including its next-layer x surviving another unit's hooks.
    fn replay(span: usize, width: usize, waves: usize, zero_plane: Option<&str>, scratch_alias: bool)
        -> Result<Option<String>> {
        let mut units = sequence_units(span, width, 1, 48)?;
        for unit in &mut units { unit.slot = unit.index % waves; }
        let routes = vec![FfnRoute::Spark; 48];
        let mut current = vec![0usize; units.len()];
        let mut entered = vec![false; units.len()];
        let mut live = Vec::<(usize, &'static str, Range<usize>)>::new();
        for step in prefill_schedule(&routes, &units, waves)? {
            let (index, mut writes) = match step {
                Step::Begin { unit } => (unit, vec!["streams0"]),
                Step::Attention { unit, .. } => {
                    let mut writes = vec!["delta"];
                    if !entered[unit] { writes.extend(["inject", "x"]); }
                    current[unit] ^= 1;
                    writes.extend([LIVE[current[unit]], "inject", "x"]);
                    (unit, writes)
                }
                Step::Send { unit, .. } => (unit, vec!["shared"]),
                Step::Land { unit, layer } => {
                    current[unit] ^= 1;
                    entered[unit] = true;
                    let mut writes = vec!["delta", LIVE[current[unit]]];
                    if layer + 1 < 48 { writes.extend(["inject", "x"]); }
                    (unit, writes)
                }
                Step::End { unit } => (unit, vec!["x"]),
                Step::Commit { unit } => { live.retain(|(owner, _, _)| *owner != unit); continue; }
                Step::Local { .. } => unreachable!(),
            };
            // Hook-local tables, query/gate/index/attn, routes/wire and scratch
            // stay separate from all six live planes, at row zero.
            writes.extend(["tables", "query", "gate", "index_q", "attn", "blocks", "indices",
                "lengths", "scratch", "topk_scratch", "router_logits", "route_ids", "route_weights", "wire"]);
            for plane in writes {
                let (plane, first) = if plane == "scratch" && scratch_alias { ("x", 0) }
                    else { (plane, if LIVE.contains(&plane) && zero_plane != Some(plane) {
                        Workspace::unit_offset(units[index].slot, width)
                    } else { 0 }) };
                let range = first..first + units[index].rows.len();
                if let Some((owner, _, _)) = live.iter().find(|(owner, held, rows)| *owner != index
                    && *held == plane && rows.start < range.end && range.start < rows.end) {
                    return Ok(Some(format!("{step:?} overwrites unit {owner} {plane}")));
                }
                if LIVE.contains(&plane) {
                    live.retain(|(owner, held, _)| *owner != index || *held != plane);
                    live.push((index, plane, range));
                }
            }
        }
        Ok(None)
    }

    #[test]
    fn real_schedule_preserves_all_qwen_live_planes_and_catches_shared_rows() -> Result<()> {
        for (span, width) in [(1, 512), (513, 512), (4096, 512), (4096, 2048), (3971, 1024),
            (8192, 4096), (8192, 2048), (8256, 2752), (16513, 2752), (16385, 2048)] {
            for waves in 1..=4 {
                assert_eq!(replay(span, width, waves, None, false)?, None, "{span}/{width}/{waves}");
            }
        }
        for plane in LIVE {
            let hazard = replay(4096, 1024, 2, Some(plane), false)?.expect("row-zero negative control");
            assert!(hazard.contains(plane), "{hazard}");
        }
        assert!(replay(4096, 1024, 2, None, true)?.unwrap().contains("x"));
        Ok(())
    }
}

impl Drop for Qwen4Engine<'_> {
    fn drop(&mut self) {
        crate::shared::decode_graph::fatal_drain(self.shutdown_submission(), "Qwen engine retirement");
    }
}

#[cfg(test)]
mod packed_route_fixture {
    use super::*;
    use super::super::{speculate, prefix};
    use crate::shared::qwen::scheduler::PrefillSeq;
    use clap::Parser;

    #[derive(Parser)]
    struct Args {
        #[command(flatten)]
        engine: super::super::EngineArgs,
    }

    fn regions(engine: &Qwen4Engine<'_>, slot: i32) -> Result<Vec<Vec<u8>>> {
        engine.slot_regions(slot as usize).into_iter().map(|buffer| {
            // SAFETY: the engine retains the region and its owning execution stream.
            unsafe { engine.library.cuda_stream_synchronize(engine.state_stream(buffer.device_id)?)?; }
            let mut bytes = vec![0; buffer.bytes];
            engine.library.copy_d2h(&mut bytes, buffer)?;
            Ok(bytes)
        }).collect()
    }

    #[test]
    fn shared_128k_prefix_uses_physical_units_once_and_fits_decode_metadata() -> Result<()> {
        let shared = 131_072 / UNIT_ROWS;
        let members = 16;
        let private = 3;
        let pool = 1_122_560 / UNIT_ROWS;
        let entries = [(shared + private) * UNIT_PAGES, shared + private];
        assert_eq!((shared + members * private) * UNIT_ROWS, 143_360);
        assert!(shared + members * private < pool && members * (shared + private) > pool);
        let old = crate::shared::prefill_packing::TableBudget([pool * UNIT_PAGES, pool]);
        assert!(!old.fits(std::iter::repeat_n(entries, members)));
        let decode = crate::shared::prefill_packing::TableBudget([DECODE_ROWS * pool * UNIT_PAGES, DECODE_ROWS * pool]);
        let offsets = decode.offsets(std::iter::repeat_n(entries, members))?;
        assert_eq!(offsets[15], [15 * entries[0], 15 * entries[1]]);
        let shape = super::super::admission::StepTableBytes::new(DECODE_ROWS, true, pool * UNIT_PAGES, pool, 0)?;
        assert_eq!(shape.page_table + shape.pool_table, 5_612_800);
        assert!(members * (entries[0] + entries[1]) * size_of::<i32>() < shape.page_table + shape.pool_table);
        Ok(())
    }

    #[test]
    #[ignore = "requires sealed Qwen native artifact, official reference and GPU"]
    fn packed_native_genuine_rows() -> Result<()> {
        #[derive(Parser)]
        struct Cli { #[command(flatten)] args: super::super::GoldenArgs }
        let snapshot = std::env::var("CUTEAFD_FIXTURE_MODEL")?;
        let golden = std::env::var("CUTEAFD_FIXTURE_REFERENCE")?;
        let args = Cli::try_parse_from(["fixture", "--snapshot", &snapshot, "--golden", &golden,
            "--local-experts", "--mtp", "3", "--pool-tokens", "65536", "--slots", "16",
            "--packed-check", "8", "--prefill-packing", "unrestricted"])?;
        tokio::runtime::Builder::new_current_thread().enable_all().build()?
            .block_on(super::super::run_golden(args.args))
    }

    #[test]
    #[ignore = "requires sealed Qwen native artifact, checkpoint, reference and GPU"]
    fn packed_totals_at_most_64_discriminator() -> Result<()> {
        let snapshot = std::env::var("CUTEAFD_FIXTURE_MODEL")?;
        let golden = std::env::var("CUTEAFD_FIXTURE_REFERENCE")?;
        let args = Args::try_parse_from(["fixture", "--snapshot", &snapshot,
            "--local-experts", "--mtp", "3", "--pool-tokens", "65536",
            "--slots", "16", "--prefill-packing", "unrestricted"])?;
        let windows = super::super::fidelity_windows(std::path::Path::new(&golden), None, false)?;
        let opened = super::super::open(&args.engine)?;
        opened.with_engine(&args.engine, |engine| {
            let mut names = Vec::new();
            for (layer, home) in engine.state_map.layers.iter().enumerate() {
                if home.gdn_ordinal.is_some() {
                    names.push(format!("rank{} layer{layer} GDN.conv", home.owner));
                    names.push(format!("rank{} layer{layer} GDN.recurrent", home.owner));
                }
            }
            if engine.ple_state.is_some() {
                names.push(format!("rank0 layer{} PLE.conv", engine.cfg.ple_layers[0]));
            }
            let mut every_equal = true;
            let shapes: &[&[usize]] = if std::env::var_os("CUTEAFD_FIXTURE_REVERSE").is_some() {
                &[&[29, 9, 7], &[40, 24]]
            } else { &[&[7, 9], &[7, 17], &[7, 9, 29], &[24, 40]] };
            for &lengths in shapes {
                let total: usize = lengths.iter().sum();
                let mut allocator = Allocator::new(engine.pages, engine.slots, &engine.cfg);
                let mut packed = lengths.iter().map(|&n| allocator.admit(n + 1)).collect::<Result<Vec<_>>>()?;
                let mut alone = lengths.iter().map(|&n| allocator.admit(n + 1)).collect::<Result<Vec<_>>>()?;
                let tokens = lengths.iter().enumerate().map(|(i, &n)| {
                    let source = &windows[i].tokens;
                    (0..n).map(|j| source[(i * 97 + j) % source.len()]).collect::<Vec<_>>()
                }).collect::<Vec<_>>();
                let mut drafts = vec![speculate::MtpSeq::default(); lengths.len()];
                let mut logits = vec![Vec::new(); lengths.len()];
                let mut sequences = packed.iter_mut().zip(&tokens).zip(&mut drafts).map(|((placement, tokens), draft)| {
                    PrefillSeq { placement, tokens, start: 0, next: None, draft, neural: true }
                }).collect::<Vec<_>>();
                for result in engine.prefill_packed(&mut sequences, &mut |i, head| {
                    logits[i] = head.row_host(engine.library, 0)?;
                    Ok(Ok(()))
                })? { result?; }
                drop(sequences);
                for i in 0..lengths.len() {
                    let head = engine.prefill_device(&mut alone[i], &tokens[i], None, None, 1)?
                        .context("singleton full-model head")?.row_host(engine.library, 0)?;
                    let mut draft = speculate::MtpSeq::default();
                    speculate::prefill_chunk(engine, &alone[i], 0, &tokens[i], None, &mut draft)?;
                    let equal = logits[i].iter().map(|x| x.to_bits()).eq(head.iter().map(|x| x.to_bits()));
                    let state = prefix::packed_state(engine, &packed[i])? == prefix::packed_state(engine, &alone[i])?;
                    let a = regions(engine, packed[i].slot)?;
                    let b = regions(engine, alone[i].slot)?;
                    ensure!(a.len() == names.len() && b.len() == names.len(), "state labels drifted");
                    let first = a.iter().zip(&b).enumerate().find(|(_, (a, b))| a != b);
                    let ple = names.iter().position(|n| n.contains("PLE.conv"))
                        .map(|p| a[p] == b[p]);
                    if let Some((plane, (a, b))) = first {
                        let byte = a.iter().zip(b).position(|(a, b)| a != b).unwrap();
                        println!("discriminator total{total} member{i}: first differing tensor {} byte{byte}", names[plane]);
                    } else if !state {
                        println!("discriminator total{total} member{i}: target slot planes equal; first differing canonical plane is paged KV/index or MTP stash/history");
                    }
                    let (kl, _) = crate::shared::golden_scores::scored_kl(&head, &logits[i], head.len())?;
                    println!("discriminator total{total} member{i} rows{}: logits_bytes={equal} canonical_state_bytes={state} PLE.conv_equal={ple:?} KL={kl:.9}", lengths[i]);
                    every_equal &= equal && state;
                }
            }
            println!("discriminator all totals <=64 byte-identical: {every_equal}");
            Ok(())
        })
    }
}

#[cfg(test)]
mod owner_state_tests {
    use super::*;

    #[test]
    fn owner_tp2_cutover_and_exit_queues_drain() -> Result<()> {
        use crate::shared::peer_split::order::{check, Schedule};
        for layers in 2..49 {
            for cutover in 1..layers {
                let mut schedule = Schedule::default();
                for _step in 0..3 {
                    for layer in 0..layers {
                        let owner = usize::from(layer >= cutover);
                        if layer == cutover {
                            schedule.push(0, "hop", 0, "cutover send");
                            schedule.wait(1, "hop", 0, "cutover land");
                        }
                        schedule.push(owner, "experts", 0, "owner input/routes");
                        schedule.wait(1 - owner, "experts", 0, "peer input/routes");
                        schedule.push(1 - owner, "experts", 1, "peer partial");
                        schedule.wait(owner, "experts", 1, "owner partial");
                    }
                    schedule.push(1, "hop", 1, "exit send");
                    schedule.wait(0, "hop", 1, "head exit land");
                    schedule.end_request();
                }
                check(&schedule).map_err(|error| anyhow::anyhow!(error.to_string()))?;
                if matches!(layers, 2 | 4 | 8) {
                    crate::shared::peer_split::order::check_submission_faults(&schedule)
                        .map_err(|error| anyhow::anyhow!(error.to_string()))?;
                }
            }
        }
        Ok(())
    }

    #[test]
    fn whole_layer_owners_require_one_contiguous_cutover() -> Result<()> {
        for layers in 2..49 {
            for cutover in 1..layers {
                let owners: Vec<_> = (0..layers).map(|layer| usize::from(layer >= cutover)).collect();
                validate_layer_owners(&owners, 2)?;
            }
        }
        for owners in [vec![], vec![0, 0], vec![1, 1], vec![1, 0], vec![0, 1, 0], vec![0, 1, 2]] {
            assert!(validate_layer_owners(&owners, 2).is_err(), "accepted {owners:?}");
        }
        validate_layer_owners(&[0, 0, 0], 1)?;
        assert!(validate_layer_owners(&[0, 1], 1).is_err());
        Ok(())
    }

    #[test]
    fn global_layer_state_map_keeps_owner_local_gdn_ordinals() -> Result<()> {
        use Qwen4Attention::{Full, Gdn};
        let kinds = [Gdn, Gdn, Full, Gdn, Full, Gdn, Gdn, Full];
        for cutover in 1..kinds.len() {
            let owners: Vec<_> = (0..kinds.len()).map(|i| usize::from(i >= cutover)).collect();
            let map = LayerStateMap::new(&kinds, &owners, 2)?;
            let mut counts = [0; 2];
            for (global, home) in map.layers.iter().enumerate() {
                assert_eq!(home.owner, owners[global]);
                if kinds[global] == Gdn {
                    assert_eq!(home.gdn_ordinal, Some(counts[home.owner]));
                    counts[home.owner] += 1;
                } else { assert_eq!(home.gdn_ordinal, None); }
            }
            assert_eq!(map.gdn_layers, counts);
            assert_eq!(counts.iter().sum::<usize>(), 5);
        }
        assert!(LayerStateMap::new(&kinds, &[0], 2).is_err());
        assert!(LayerStateMap::new(&[Gdn], &[2], 2).is_err());
        Ok(())
    }
}
