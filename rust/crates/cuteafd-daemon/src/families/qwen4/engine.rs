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
use crate::shared::experts::fp8::{Fp8Experts, Fp8Layer};
use crate::shared::launch_grid::Fp8QuantizeGrid;
use crate::shared::memory::HostAllocation;
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
use crate::shared::decode_graph::{check_bucket_thresholds, masked_row, real_row_moe, ProjectionThreshold};
use std::ffi::c_void;

type Dev<'a> = Rc<Allocation<'a>>;

pub(crate) const PAGE_ROWS: usize = 64;
/// Rows of the decode-shaped programs (`_m64`).
pub(crate) const DECODE_ROWS: usize = 64;
use cuteafd_loader::serving_capacity::qwen_graphs::{QWEN_PLAIN_BUCKETS as PLAIN_BUCKETS, QWEN_SPEC_BUCKETS as SPEC_BUCKETS};
// Keep this registry aligned with the pinned fork; script contracts check its source thresholds.
const GLM_BF16_SKINNY_ROWS: usize = 8;
const QWEN_WIDE_SKINNY_ROWS: usize = 24;
const QWEN_MEDIUM_SKINNY_ROWS: usize = 64;
const QWEN_SMALL_SKINNY_ROWS: usize = 160;
const FP8_PROJECTION_SKINNY_ROWS: usize = 16;
const DECODE_PROJECTION_THRESHOLDS: &[ProjectionThreshold] = &[
    ProjectionThreshold { name: "gdn.in_projection.bf16", skinny_rows: GLM_BF16_SKINNY_ROWS },
    ProjectionThreshold { name: "gdn.out_projection.bf16", skinny_rows: GLM_BF16_SKINNY_ROWS },
    ProjectionThreshold { name: "attention.in_projection.bf16", skinny_rows: GLM_BF16_SKINNY_ROWS },
    ProjectionThreshold { name: "attention.out_projection.bf16", skinny_rows: GLM_BF16_SKINNY_ROWS },
    ProjectionThreshold { name: "hc.down_inject", skinny_rows: QWEN_SMALL_SKINNY_ROWS },
    ProjectionThreshold { name: "head.mixer_down", skinny_rows: QWEN_SMALL_SKINNY_ROWS },
    ProjectionThreshold { name: "ple.kv", skinny_rows: QWEN_WIDE_SKINNY_ROWS },
    ProjectionThreshold { name: "router.scores", skinny_rows: QWEN_MEDIUM_SKINNY_ROWS },
    ProjectionThreshold { name: "shared.gate_up", skinny_rows: QWEN_MEDIUM_SKINNY_ROWS },
    ProjectionThreshold { name: "mtp.feedback", skinny_rows: QWEN_WIDE_SKINNY_ROWS },
    ProjectionThreshold { name: "gdn.projections.fp8", skinny_rows: FP8_PROJECTION_SKINNY_ROWS },
    ProjectionThreshold { name: "attention.projections.fp8", skinny_rows: FP8_PROJECTION_SKINNY_ROWS },
];

pub(super) fn startup_graphs_enabled(graphs: Option<&str>, startup: Option<&str>) -> bool {
    graphs != Some("0") && startup.map_or(true, |value| value == "1")
}

fn decode_bucket(rows: usize, spec: bool) -> usize {
    cuteafd_loader::serving_capacity::qwen_graphs::qwen_decode_bucket(rows, spec)
}

pub(crate) fn copy_row_limit(rows: usize, sequences: usize) -> usize {
    SPEC_BUCKETS.iter().copied().filter(|&bucket| bucket <= rows && bucket >= sequences)
        .last().unwrap_or(sequences)
}

/// Selected-slot row width of the sparse attention (2048 + 3, padded to 64).
pub(crate) const SPARSE_TOPK: usize = 2112;
/// BF16 index-key width, independent of the K/V record format.
pub(crate) const INDEX_DIM: usize = 128;
const MAX_RANKS: usize = 6;
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
pub(crate) const FP8_HEAD_ROWS: usize = 16;

/// `(first row, rows)` spans of at most [`FP8_HEAD_ROWS`] covering `rows` logits rows.
pub(crate) fn fp8_head_spans(rows: usize) -> impl Iterator<Item = (usize, usize)> {
    (0..rows).step_by(FP8_HEAD_ROWS).map(move |first| (first, FP8_HEAD_ROWS.min(rows - first)))
}
/// Rows a speculative step records per GDN layer (the fork's `REPLAY_ROWS`).
pub(crate) const REPLAY_ROWS: usize = 64;
/// Target rows a sequence may hold for its MTP canonical history before the
/// next draft step (the MTP stash, per state slot).
pub(crate) const MTP_PENDING_ROWS: usize = 64;

/// Bytes of one GDN layer's replay record (`gdn_replay_layout` in the fork):
/// normalized keys and values, decay and beta, and the conv inputs.
fn gdn_replay_bytes(cfg: &Qwen4Config) -> usize {
    let (kh, vh, d) = (cfg.gdn_key_heads, cfg.gdn_value_heads, cfg.gdn_head_dim);
    let bytes = REPLAY_ROWS * (kh * d * 4 + vh * d * 4 + vh * 2 * 4 + cfg.gdn_conv_width() * 2);
    bytes.div_ceil(1024) * 1024
}

/// Routed experts on this GPU from the TP1 package. All layers stay resident
/// unless an explicit diagnostic paging window was requested.
pub(crate) struct LocalExperts<'a> {
    pub library: &'a NativeLibrary,
    pub tensors: &'a Fp8ExpertTensors,
    pub experts: RefCell<Fp8Experts<'a>>,
    /// Mixed-format MTP owns its FP8 package separately from NVFP4 target layers.
    pub mtp_experts: Option<Fp8Experts<'a>>,
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
/// `exl3-qwen4-k45/rtx-tp1` package): a window of resident layers. The
/// package reads FP8 K32 wire rows and its reducer adds the shared expert.
pub(crate) struct LocalExl3<'a> {
    pub library: &'a NativeLibrary,
    pub native_lib: std::path::PathBuf,
    pub catalog: &'a cuteafd_loader::OfficialV41Catalog,
    pub resident: RefCell<Option<(std::ops::Range<usize>, crate::families::deepseek_v4::local::LocalExperts<'a>)>>,
    pub window: usize,
    pub layers: usize,
    /// The MTP layer's experts (draft stage 0) stay resident with every window;
    /// expert layer `layers` names them.
    pub mtp: bool,
    pub max_rows: usize,
    pub budget: usize,
    pub loads: RefCell<usize>,
}

impl LocalExl3<'_> {
    pub(super) fn ensure(&self, layer: usize, stream: *mut c_void) -> Result<()> {
        if self.resident.borrow().as_ref().is_some_and(|(range, _)| range.contains(&layer) || layer == self.layers) {
            ensure!(layer < self.layers || self.mtp, "MTP experts are not loaded");
            return Ok(());
        }
        let layer = if layer == self.layers { 0 } else { layer };
        // SAFETY: the engine owns this stream; the old window's launches drain first.
        unsafe { self.library.cuda_stream_synchronize(stream)? };
        *self.resident.borrow_mut() = None;
        let range = layer..(layer + self.window).min(self.layers);
        let started = std::time::Instant::now();
        let local = crate::families::deepseek_v4::local::LocalExperts::load_range(self.library, &self.native_lib, self.catalog,
            usize::from(self.mtp), range.clone(), self.max_rows, self.budget, stream)?
            .context("no coordinator EXL3 package for this checkpoint (build qwen4:exl3-k45)")?;
        let range = layer..layer + local.layers();
        ensure!(range.contains(&layer), "EXL3 expert layer {layer} does not fit the budget");
        *self.loads.borrow_mut() += range.len();
        tracing::debug!(?range, elapsed_ms = started.elapsed().as_millis() as u64, "EXL3 expert window resident");
        *self.resident.borrow_mut() = Some((range, local));
        Ok(())
    }
}

/// The MTP layer stays resident locally even when backbone experts run on Sparks.
pub(crate) enum MtpExperts<'a> {
    Fp8(Fp8Experts<'a>),
    Exl3(RefCell<crate::families::deepseek_v4::local::LocalExperts<'a>>),
}

/// Where the routed experts run.
pub(crate) enum Experts<'a> {
    Local(LocalExperts<'a>),
    LocalExl3(LocalExl3<'a>),
    Tp2 { routed: RefCell<QwenTp2<'a>>, mtp: Option<MtpExperts<'a>> },
    /// Spark ranks over RoCE (one BF16 partial plane per rank), plus a local draft layer.
    Spark { transport: RefCell<SparkLink<'a>>, runtime: tokio::runtime::Runtime,
        mtp: Option<MtpExperts<'a>> },
    /// No routed experts (plumbing tests only: the MoE output is the shared expert alone).
    SharedOnly,
}

/// Host tables of one step.
#[derive(Default)]
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
        Self { units, pages, pool_pages, slot, len: 0, state_len: 0, history, rope: Default::default(), media: None }
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

#[cfg(test)]
mod tests {
    use cuteafd_loader::families::qwen4::NgramHasher;

    #[test]
    fn startup_graphs_default_on_but_respect_both_explicit_opt_outs() {
        use super::startup_graphs_enabled;
        assert!(startup_graphs_enabled(None, None));
        assert!(startup_graphs_enabled(None, Some("1")));
        assert!(!startup_graphs_enabled(None, Some("0")));
        assert!(!startup_graphs_enabled(Some("0"), None));
        assert!(!startup_graphs_enabled(Some("0"), Some("1")));
    }

    #[test]
    fn serving_graph_counts_and_modes_cover_the_qualified_layout() {
        use super::*;
        assert_eq!(graph_geometries(32768, 512, 2051).len(), 40);
        let shapes = serving_graph_shapes(32768, 512, 2051, 16, true);
        assert_eq!(shapes.len() * 49, 21560);
        let modes: std::collections::HashSet<_> = shapes.iter().map(|&(rows, spec, _)| (rows, spec)).collect();
        assert_eq!(modes, [(1, false), (4, false), (8, false), (16, false),
            (2, true), (4, true), (8, true), (16, true), (24, true), (32, true), (64, true)].into());
        assert_eq!(serving_graph_shapes(32768, 512, 2051, 16, false).len() * 49, 7840);
        assert_eq!(graph_geometries(8192, 128, 2051).len(), 23);
    }

    #[test]
    fn decode_buckets_preserve_registered_arithmetic_routes() {
        use super::*;
        check_bucket_thresholds(PLAIN_BUCKETS, DECODE_PROJECTION_THRESHOLDS).unwrap();
        check_bucket_thresholds(SPEC_BUCKETS, DECODE_PROJECTION_THRESHOLDS).unwrap();
        for (real, spec, bucket) in [(5, false, 8), (9, false, 16), (3, true, 4),
            (5, true, 8), (9, true, 16), (17, true, 24), (25, true, 32), (33, true, 64)] {
            assert_eq!(decode_bucket(real, spec), bucket);
            for projection in DECODE_PROJECTION_THRESHOLDS {
                assert_eq!(real <= projection.skinny_rows, bucket <= projection.skinny_rows, "{}", projection.name);
            }
        }
        assert_eq!(decode_bucket(2, true), 2);
        assert_eq!(decode_bucket(24, true), 24);
    }

    #[test]
    fn checkpoint_full_graphs_add_only_one_long_geometry() {
        use super::*;
        let pages = 2_097_152 / PAGE_ROWS;
        let old = serving_graph_shapes(131_072, pages, 2051, 16, true);
        let full = serving_graph_shapes(262_144, pages, 2051, 16, true);
        assert_eq!(old.len(), 671);
        assert_eq!(full.len(), 682);
        assert!(old.iter().all(|key| full.contains(key)));
        for allocated in [131_072usize, 262_144] {
            for live in [1usize, 8192, 131_072, 131_073, 200_000, 262_144] {
                if live > allocated { continue; }
                let units = crate::shared::context::decode_allocation_units(
                    allocated.div_ceil(UNIT_ROWS), live, UNIT_ROWS);
                let pool_stride = units.next_power_of_two();
                let geometry = GraphGeometry { pool_width: live.div_ceil(UNIT_ROWS).next_power_of_two()
                    .min(pool_stride), pool_stride, page_stride: (units * UNIT_PAGES).next_power_of_two(),
                    long: live > 2051 };
                assert!(full.contains(&(64, true, geometry)), "{allocated}/{live}/{geometry:?}");
            }
        }
    }

    #[test]
    fn startup_geometries_cover_all_reachable_lengths_and_allocations() {
        use super::*;
        for (context, pages) in [(1, 4), (255, 4), (257, 8), (4096, 64), (8192, 128), (32768, 512), (8192, 100)] {
            let geometries = graph_geometries(context, pages, 2051);
            for units in 1..=context.div_ceil(UNIT_ROWS).min(pages / UNIT_PAGES) {
                let pool_stride = units.next_power_of_two().min(pages / UNIT_PAGES);
                let page_stride = (units * UNIT_PAGES).next_power_of_two().min(pages);
                for length in 1..=(units * UNIT_ROWS).min(context) {
                    let geometry = GraphGeometry { pool_width: length.div_ceil(UNIT_ROWS).next_power_of_two().min(pool_stride),
                        pool_stride, page_stride, long: length > 2051 };
                    assert!(geometries.contains(&geometry), "context {context} units {units} length {length}: {geometry:?}");
                }
            }
            let unique: std::collections::HashSet<_> = geometries.iter().collect();
            assert_eq!(unique.len(), geometries.len());
        }
    }

    #[test]
    fn padding_masks_every_tail_and_preserves_native_real_rows() {
        use super::*;
        for real in 1..=64 {
            for ple_rows in [0, 16] {
                let mut tables = StepTables { page_stride: 4, pool_stride: 2,
                    positions: vec![17; real], rope_positions: vec![[2, 3, 4]; real],
                    block_rope_positions: vec![[1, 2, 3]; real], kv_slots: vec![29; real],
                    pool_slots: vec![7; real], slots: vec![2; real], seq_first: vec![0; real],
                    cache_lengths: vec![4; real], page_table: vec![3; real * 4], pool_table: vec![5; real * 2],
                    ple_ids: vec![9; real * ple_rows], ..Default::default() };
                let mut tokens = vec![248056; real];
                let bucket = decode_bucket(real, true);
                pad_decode_tables(&mut tables, &mut tokens, bucket, ple_rows);
                assert_eq!(tokens.len(), bucket);
                assert_eq!(&tokens[..real], vec![248056; real]);
                assert_eq!(&tables.positions[..real], vec![17; real]);
                assert_eq!(&tables.ple_ids[..real * ple_rows], vec![9; real * ple_rows]);
                for row in real..bucket {
                    assert_eq!((tokens[row], tables.positions[row], tables.kv_slots[row], tables.pool_slots[row],
                        tables.slots[row], tables.seq_first[row], tables.cache_lengths[row]),
                        (0, -1, -1, -1, -1, row as i32, 0));
                    assert_eq!(tables.rope_positions[row], [0; 3]);
                    assert_eq!(tables.block_rope_positions[row], [0; 3]);
                }
                assert!(tables.page_table[real * 4..].iter().all(|&p| p == 0));
                assert!(tables.pool_table[real * 2..].iter().all(|&p| p == 0));
                assert!(tables.ple_ids[real * ple_rows..].iter().all(|&p| p == 0));
                assert_eq!(tables.page_table.len(), bucket * 4);
                assert_eq!(tables.pool_table.len(), bucket * 2);
                assert_eq!(tables.ple_ids.len(), bucket * ple_rows);
            }
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
    router_logits: Dev<'a>,
    route_ids: Dev<'a>,
    route_weights: Dev<'a>,
    wire: Dev<'a>,
    router_host: RefCell<HostAllocation<'a>>,
    /// MTP steps: the source row of each row's feedback, and the greedy draft
    /// (U32 token, FP32 logit) of each head row.
    hidden_rows: Dev<'a>,
    argmax: Dev<'a>,
    /// The step's token ids (U32, gathered from the device embedding table;
    /// MTP chain steps: indices into the previous step's drafts).
    ids: Dev<'a>,
    /// Greedy selection of the logits rows inside the decode graph: U32 ids, then U32 statuses.
    select: Dev<'a>,
    /// Pinned staging of a step's host tables and input rows (async uploads)
    /// and its fill level.
    staging: RefCell<(HostAllocation<'a>, usize)>,
    /// Pinned landing of the logits rows.
    logits_host: RefCell<HostAllocation<'a>>,
    head: VocabularyHead<'a>,
    _head_workspace: Dev<'a>,
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct LayerStateHome {
    owner: usize,
    gdn_ordinal: Option<usize>,
}

#[derive(Debug, PartialEq, Eq)]
struct LayerStateMap {
    layers: Vec<LayerStateHome>,
    gdn_layers: Vec<usize>,
}

impl LayerStateMap {
    fn new(kinds: &[Qwen4Attention], owners: &[usize], devices: usize) -> Result<Self> {
        ensure!(devices > 0 && kinds.len() == owners.len(), "invalid Qwen layer ownership map");
        let mut gdn_layers = vec![0; devices];
        let mut layers = Vec::with_capacity(kinds.len());
        for (&kind, &owner) in kinds.iter().zip(owners) {
            ensure!(owner < devices, "Qwen layer owner {owner} outside {devices} devices");
            let gdn_ordinal = if kind == Qwen4Attention::Gdn {
                let ordinal = gdn_layers[owner];
                gdn_layers[owner] += 1;
                Some(ordinal)
            } else { None };
            layers.push(LayerStateHome { owner, gdn_ordinal });
        }
        Ok(Self { layers, gdn_layers })
    }
}

fn validate_layer_owners(owners: &[usize], devices: usize) -> Result<()> {
    ensure!(!owners.is_empty() && matches!(devices, 1 | 2), "invalid Qwen attention owners");
    ensure!(owners.iter().all(|&owner| owner < devices), "Qwen attention owner out of range");
    if devices == 1 { return Ok(()); }
    cuteafd_loader::placement::families::qwen4::check_dual_layer_owners(owners)
}

/// Compact recurrent pools on one owner. Global layer ids never index these directly.
struct GdnBank<'a> {
    library: &'a NativeLibrary,
    programs: &'a Programs<'a>,
    stream: *mut c_void,
    layers: usize,
    conv: Option<Dev<'a>>,
    state: Option<Dev<'a>>,
    replay: Option<Dev<'a>>,
    commit_tables: Dev<'a>,
}

impl<'a> GdnBank<'a> {
    fn new(library: &'a NativeLibrary, programs: &'a Programs<'a>, stream: *mut c_void,
        cfg: &Qwen4Config, layers: usize, slots: usize) -> Result<Self> {
        let zeroed = |bytes: usize| -> Result<Dev<'a>> {
            let allocation = Rc::new(Allocation::new(Device { library, id: library.cuda_get_device()? }, bytes.max(256))?);
            library.cuda_zero_bytes(allocation.buffer, allocation.buffer.bytes)?;
            Ok(allocation)
        };
        let (conv, state, replay) = if layers > 0 {
            (Some(zeroed(layers * slots * Qwen4Engine::conv_slot_bytes(cfg))?),
             Some(zeroed(layers * slots * Qwen4Engine::state_slot_bytes(cfg))?),
             Some(zeroed(layers * gdn_replay_bytes(cfg))?))
        } else { (None, None, None) };
        Ok(Self { library, programs, stream, layers, conv, state, replay,
            commit_tables: zeroed(3 * DECODE_ROWS * 4)? })
    }

    fn commit(&self, table: &[i32], sequences: usize, slots: usize) -> Result<()> {
        // SAFETY: this bank owns the stream; its previous table readers retire before upload.
        unsafe { self.library.cuda_stream_synchronize(self.stream)? };
        self.library.copy_h2d(cuteafd_ffi::CuteafdDeviceBuffer {
            bytes: std::mem::size_of_val(table), ..self.commit_tables.buffer }, bytes_of(table))?;
        if let (Some(conv), Some(state), Some(replay)) = (&self.conv, &self.state, &self.replay) {
            let program = self.programs.program("qwen4_gdn_commit", &["state", "conv_state", "replay", "tables"])?;
            let scalars = [sequences, self.layers, slots].map(|n| i32::try_from(n).map(Scalar::I32));
            let scalars = scalars.into_iter().collect::<std::result::Result<Vec<_>, _>>()?;
            // SAFETY: the live owner-local pools match the compact layer/slot geometry.
            unsafe { program.launch(&[state.buffer.ptr, conv.buffer.ptr, replay.buffer.ptr,
                self.commit_tables.buffer.ptr], &scalars, self.stream)? };
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy)]
struct Tp2Rows {
    input: usize,
    ids: usize,
    weights: usize,
    bytes: usize,
    partial_bytes: usize,
}

impl Tp2Rows {
    fn new(hidden: usize, topk: usize, rows: usize, wire: bool,
        partial: crate::shared::experts::rtx::PartialDtype) -> Result<Self> {
        ensure!(hidden > 0 && topk > 0 && rows > 0 && (!wire || hidden % 32 == 0),
            "invalid Qwen TP2 row geometry");
        let mul = |a: usize, b: usize| a.checked_mul(b).context("Qwen TP2 row extent overflow");
        let align = |bytes: usize| bytes.checked_add(15).map(|n| n / 16 * 16)
            .context("Qwen TP2 row padding overflow");
        let input = mul(rows, if wire { hidden.checked_add(hidden / 32).context("Qwen wire stride overflow")? }
            else { mul(hidden, 2)? })?;
        let ids = align(input)?;
        let route_bytes = mul(mul(rows, topk)?, 4)?;
        let weights = ids.checked_add(align(route_bytes)?).context("Qwen TP2 routes overflow")?;
        let bytes = weights.checked_add(align(route_bytes)?).context("Qwen TP2 payload overflow")?;
        let element = match partial { crate::shared::experts::rtx::PartialDtype::F32 => 4,
            crate::shared::experts::rtx::PartialDtype::Bf16 => 2 };
        Ok(Self { input, ids, weights, bytes, partial_bytes: mul(mul(rows, hidden)?, element)? })
    }
}

/// TP2 routed halves reduce only onto the whole-attention owner. The next
/// broadcast cannot overtake that owner's sum, so both rank workspaces are
/// serialized without a host synchronization on the request path.
pub(crate) struct QwenTp2<'a> {
    library: &'a NativeLibrary,
    ranks: [Device<'a>; 2],
    experts: std::mem::ManuallyDrop<[Box<dyn crate::shared::experts::rtx::RtxExpertLayer + 'a>; 2]>,
    exchange: std::mem::ManuallyDrop<crate::shared::peer_split::PeerExchange<'a>>,
    combine: [cuteafd_ffi::RtxTp2Combine<'a>; 2],
    send: [Allocation<'a>; 2],
    reduced: [Allocation<'a>; 2],
    hidden: usize,
    topk: usize,
    max_rows: usize,
    wire: bool,
    layout: Tp2Rows,
}

impl<'a> QwenTp2<'a> {
    fn new(library: &'a NativeLibrary, ranks: [crate::shared::peer_split::RankDevice; 2],
        experts: [Box<dyn crate::shared::experts::rtx::RtxExpertLayer + 'a>; 2],
        hidden: usize, topk: usize, max_rows: usize, wire: bool) -> Result<Self> {
        use crate::shared::experts::rtx::{PartialDtype, RtxShard};
        ensure!(experts[0].partial() == experts[1].partial()
            && experts[0].layers() == experts[1].layers(), "Qwen TP2 halves disagree");
        for rank in 0..2 {
            ensure!(experts[rank].shard() == RtxShard::Tp2 { rank: rank as u8 }
                && experts[rank].device() == ranks[rank].device, "Qwen TP2 rank ownership mismatch");
        }
        let layout = Tp2Rows::new(hidden, topk, max_rows, wire, experts[0].partial())?;
        let devices = ranks.map(|rank| Device { library, id: rank.device });
        let zero = |rank: usize, bytes| -> Result<Allocation<'a>> {
            let allocation = Allocation::new(devices[rank], bytes)?;
            devices[rank].run(|| library.cuda_zero_bytes(allocation.buffer, bytes))?;
            Ok(allocation)
        };
        let _scope = devices[0].enter()?;
        let exchange = crate::shared::peer_split::PeerExchange::new_abortable(library, ranks, 2,
            layout.bytes.max(layout.partial_bytes).checked_add(15).context("Qwen TP2 slot overflow")? / 16 * 16)?;
        let combine = [devices[0].run(|| library.rtx_tp2_combine())?, devices[1].run(|| library.rtx_tp2_combine())?];
        let send = [zero(0, layout.bytes)?, zero(1, layout.bytes)?];
        let result_bytes = max_rows.checked_mul(hidden).and_then(|n| n.checked_mul(2))
            .context("Qwen TP2 reduction extent overflow")?;
        let reduced = [zero(0, result_bytes)?, zero(1, result_bytes)?];
        let mut tp2 = Self { library, ranks: devices, experts: std::mem::ManuallyDrop::new(experts),
            exchange: std::mem::ManuallyDrop::new(exchange), combine, send, reduced,
            hidden, topk, max_rows, wire, layout };
        // Prime before any peer wait: CUDA LAZY initialization may synchronize
        // the device, which would deadlock after a not-yet-published peer flag.
        for rank in 0..2 {
            let input = tp2.input(tp2.send[rank].buffer.ptr);
            let routes = tp2.routes(tp2.send[rank].buffer.ptr, layout);
            let stream = tp2.exchange.stream(rank);
            let dtype = match tp2.experts[rank].partial() { PartialDtype::F32 => cuteafd_ffi::RtxPartialDtype::F32,
                PartialDtype::Bf16 => cuteafd_ffi::RtxPartialDtype::Bf16 };
            devices[rank].run(|| {
                // SAFETY: zeroed input/routes cover max_rows, and no peer waits exist yet.
                unsafe { tp2.experts[rank].prime(input, routes, stream)?; }
                let partial = cuteafd_ffi::CuteafdDeviceBuffer { ptr: tp2.experts[rank].output(),
                    bytes: layout.partial_bytes, device_id: devices[rank].id, ..Default::default() };
                library.cuda_zero_bytes(partial, partial.bytes)?;
                let shared = zero(rank, result_bytes)?;
                // SAFETY: initialize geometry-generic sum/add on disjoint retained buffers.
                unsafe {
                    tp2.combine[rank].sum(partial.ptr, partial.ptr, tp2.reduced[rank].buffer.ptr.cast(),
                        max_rows * hidden, dtype, stream)?;
                    library.peer_add_bf16(tp2.reduced[rank].buffer.ptr, tp2.reduced[rank].buffer.ptr,
                        shared.buffer.ptr, max_rows * hidden, stream)?;
                    library.cuda_stream_synchronize(stream)?;
                }
                Ok(())
            })?;
        }
        Ok(tp2)
    }

    fn input(&self, ptr: *mut c_void) -> crate::shared::experts::rtx::ExpertInput {
        if self.wire { crate::shared::experts::rtx::ExpertInput::Fp8K32(ptr) }
        else { crate::shared::experts::rtx::ExpertInput::Bf16(ptr) }
    }

    fn routes(&self, ptr: *mut c_void, layout: Tp2Rows) -> crate::shared::experts::rtx::Routes {
        crate::shared::experts::rtx::Routes { ids: ptr.cast::<u8>().wrapping_add(layout.ids).cast(),
            weights: ptr.cast::<u8>().wrapping_add(layout.weights).cast() }
    }

    /// # Safety
    /// Input/routes/shared/output belong to owner, cover rows, are disjoint
    /// except the read-only operands, and stay live until both streams drain.
    unsafe fn enqueue(&mut self, owner: usize, layer: usize, rows: usize, input: *const c_void,
        routes: crate::shared::experts::rtx::Routes, shared: *const c_void, output: *mut c_void) -> Result<()> {
        self.exchange.require_live()?;
        // SAFETY: forwards the retained input/output and ordering contract.
        let result = unsafe { self.enqueue_live(owner, layer, rows, input, routes, shared, output) };
        if result.is_err() {
            let aborted = self.exchange.publish_abort();
            let drained = self.exchange.drain_compute();
            let complete = aborted.is_ok() && drained.is_ok();
            self.exchange.finish_terminal(complete);
            if !complete { self.library.quarantine_module_after_failed_drain(); }
        }
        result
    }

    unsafe fn enqueue_live(&mut self, owner: usize, layer: usize, rows: usize, input: *const c_void,
        routes: crate::shared::experts::rtx::Routes, shared: *const c_void, output: *mut c_void) -> Result<()> {
        use crate::shared::experts::rtx::PartialDtype;
        ensure!(owner < 2 && (1..=self.max_rows).contains(&rows), "invalid Qwen TP2 owner/rows");
        let peer = 1 - owner;
        let extent = Tp2Rows::new(self.hidden, self.topk, rows, self.wire, self.experts[owner].partial())?;
        let ptr = self.send[owner].buffer.ptr;
        let stream = self.exchange.stream(owner);
        self.ranks[owner].run(|| {
            for (source, offset, bytes) in [(input, 0, extent.input), (routes.ids.cast_const(), extent.ids, rows * self.topk * 4),
                (routes.weights.cast_const(), extent.weights, rows * self.topk * 4)] {
                let dst = cuteafd_ffi::CuteafdDeviceBuffer { ptr: ptr.cast::<u8>().wrapping_add(offset).cast(),
                    bytes, device_id: self.ranks[owner].id, ..Default::default() };
                let src = cuteafd_ffi::CuteafdDeviceBuffer { ptr: source.cast_mut(), ..dst };
                // SAFETY: documented owner inputs are live and destination spans are nonoverlapping.
                unsafe { self.library.copy_d2d_async(dst, src, bytes, stream)?; }
            }
            Ok(())
        })?;
        self.exchange.push(owner, 0, ptr, extent.bytes)?;
        self.exchange.wait(peer, 0)?;
        let local_input = self.input(ptr);
        let local_routes = self.routes(ptr, extent);
        let peer_ptr = self.exchange.recv(peer, 0)?;
        let peer_input = self.input(peer_ptr);
        let peer_routes = self.routes(peer_ptr, extent);
        // SAFETY: broadcast is ordered before the peer and owner input packing before local work.
        unsafe {
            self.experts[owner].enqueue(layer, rows, local_input, local_routes, stream)?;
            self.experts[peer].enqueue(layer, rows, peer_input, peer_routes, self.exchange.stream(peer))?;
        }
        self.exchange.push(peer, 1, self.experts[peer].output(), extent.partial_bytes)?;
        self.exchange.wait(owner, 1)?;
        let mine = self.experts[owner].output();
        let theirs = self.exchange.recv(owner, 1)?;
        let [rank0, rank1] = if owner == 0 { [mine, theirs] } else { [theirs, mine] };
        let dtype = match self.experts[owner].partial() { PartialDtype::F32 => cuteafd_ffi::RtxPartialDtype::F32,
            PartialDtype::Bf16 => cuteafd_ffi::RtxPartialDtype::Bf16 };
        self.ranks[owner].run(|| {
            // SAFETY: rank partials arrived on owner; reduction/output storage is disjoint and retained.
            unsafe {
                self.combine[owner].sum(rank0, rank1, self.reduced[owner].buffer.ptr.cast(), rows * self.hidden, dtype, stream)?;
                self.library.peer_add_bf16(self.reduced[owner].buffer.ptr, shared, output, rows * self.hidden, stream)
            }
        })
    }
}

impl Drop for QwenTp2<'_> {
    fn drop(&mut self) {
        use crate::shared::peer_split::TerminalState;
        let complete = match self.exchange.terminal_state() {
            TerminalState::Drained => true,
            TerminalState::Active => {
                // Publish abort before draining: a failed enqueue can leave a
                // peer wait queued whose producer was never reached.
                let aborted = self.exchange.publish_abort();
                let drained = self.exchange.drain_compute();
                let complete = aborted.is_ok() && drained.is_ok();
                self.exchange.finish_terminal(complete);
                if !complete { tracing::error!(?aborted, ?drained, "Qwen TP2 streams failed to drain"); }
                complete
            }
            _ => false,
        };
        if !complete {
            self.library.quarantine_module_after_failed_drain();
            return;
        }
        // SAFETY: both compute streams drained; each retained component drops exactly once.
        unsafe {
            std::mem::ManuallyDrop::drop(&mut self.experts);
            std::mem::ManuallyDrop::drop(&mut self.exchange);
        }
    }
}

pub(crate) struct Qwen4Engine<'a> {
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
    /// Where the last MTP step left its output streams (decode, stream buffer).
    mtp_streams: std::cell::Cell<(bool, usize)>,
    /// Logical page of each pool-cache page within its sequence, and its host copy (a shared
    /// pool page sits at the same logical page in every sequence that holds it).
    pool_logical: Dev<'a>,
    pool_logical_host: RefCell<Vec<i32>>,
    /// Pool-cache pages: one per allocation unit (`pages / UNIT_PAGES`).
    pub pool_pages: usize,
    workspace: RefCell<Option<Workspace<'a>>>,
    decode_workspace: RefCell<Option<Workspace<'a>>>,
    peer_workspace: RefCell<Option<Workspace<'a>>>,
    peer_decode_workspace: RefCell<Option<Workspace<'a>>>,
    experts: Option<Experts<'a>>,
    hops: Option<crate::shared::peer_split::hop::HopLink<'a>>,
    /// Host seconds: GPU wait before expert exchanges, the exchanges.
    pub profile: RefCell<[f64; 2]>,
    graphs: RefCell<std::collections::HashMap<GraphKey, GraphExec<'a>>>,
    use_graphs: bool,
    startup_graphs: bool,
    warming_graphs: Cell<bool>,
    /// Prefill programs over FP8-only projections quantize their activations
    /// (W8A8, `fp8_rows` 1) instead of W8A16 (`--fp8-prefill-w8a8`).
    pub w8a8_prefill: bool,
    pub full_prefill_logits: bool,
    /// Recorded after a Spark exchange's device-to-host copies: the host
    /// waits on it while the shared expert runs behind it.
    routes_ready: *mut c_void,
    /// L2 prefetch of the next layer's weights during decode exchanges.
    pub l2: Option<crate::shared::l2_prefetch::L2Prefetch>,
    /// The token embedding table (resident on this GPU or read from its shard).
    pub embedding: TokenEmbedding<'a>,
    /// Deferred MTP drafts of a cycle: U32 [MTP_DEFERRED_STEPS, DECODE_ROWS].
    mtp_drafts: Dev<'a>,
}

type GraphGeometry = cuteafd_loader::serving_capacity::qwen_graphs::QwenGraphGeometry;

fn graph_geometries(context: usize, pages: usize, dense: usize) -> Vec<GraphGeometry> {
    cuteafd_loader::serving_capacity::qwen_graphs::qwen_graph_geometries(context, pages, dense)
}

pub(super) fn serving_graph_count(context: usize, pool_tokens: usize, dense: usize,
    sequences: usize, speculation: bool, layers: usize) -> Result<usize> {
    let pages = cuteafd_loader::serving_capacity::qwen_graphs::qwen_pool_pages(pool_tokens)
        .context("Qwen graph page count overflow")?;
    serving_graph_shapes(context, pages, dense, sequences, speculation).len()
        .checked_mul(layers.checked_add(1).context("Qwen graph segment count overflow")?)
        .context("Qwen graph count overflow")
}

fn serving_graph_shapes(context: usize, pages: usize, dense: usize, sequences: usize, speculation: bool)
    -> Vec<(usize, bool, GraphGeometry)> {
    cuteafd_loader::serving_capacity::qwen_graphs::qwen_serving_graph_shapes(context, pages, dense, sequences, speculation)
}

fn pad_decode_tables(tables: &mut StepTables, tokens: &mut Vec<u32>, bucket: usize, ple_rows: usize) {
    while tables.kv_slots.len() < bucket {
        let row = masked_row(tables.kv_slots.len());
        tokens.push(0);
        tables.positions.push(row.position);
        tables.rope_positions.push([0; 3]);
        tables.block_rope_positions.push([0; 3]);
        tables.kv_slots.push(row.kv_slot);
        tables.pool_slots.push(row.pool_slot);
        tables.slots.push(row.state_slot);
        tables.seq_first.push(row.seq_first);
        tables.cache_lengths.push(row.cache_length);
        tables.page_table.extend(std::iter::repeat_n(0, tables.page_stride));
        tables.pool_table.extend(std::iter::repeat_n(0, tables.pool_stride));
        tables.ple_ids.extend(std::iter::repeat_n(0, ple_rows));
    }
}

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
    #[allow(clippy::too_many_arguments)]
    pub fn new(library: &'a NativeLibrary, programs: &'a Programs<'a>, cfg: Qwen4Config, weights: Qwen4Weights<'a>,
        ple: Option<super::ple::PleTable<'a>>, stream: *mut c_void, max_context: usize, prefill_rows: usize,
        pages: usize, slots: usize, kv_format: cuteafd_loader::families::qwen4::Qwen4KvCache, embedding: TokenEmbedding<'a>) -> Result<Self> {
        let device = Device { library, id: library.cuda_get_device()? };
        let owners = vec![0; weights.layers.len()];
        Self::new_placed(library, cfg, weights, ple, &owners, &[(device, programs, stream)],
            max_context, prefill_rows, pages, slots, kv_format, embedding)
    }

    /// Whole-width attention owners; head, embedding and MTP remain on rank zero.
    /// Execution is enabled separately, after TP2 exchange and graph admission.
    #[allow(clippy::too_many_arguments)]
    pub fn new_placed(library: &'a NativeLibrary, cfg: Qwen4Config, weights: Qwen4Weights<'a>,
        ple: Option<super::ple::PleTable<'a>>, owners: &[usize],
        ranks: &[(Device<'a>, &'a Programs<'a>, *mut c_void)], max_context: usize,
        prefill_rows: usize, pages: usize, slots: usize,
        kv_format: cuteafd_loader::families::qwen4::Qwen4KvCache, embedding: TokenEmbedding<'a>) -> Result<Self> {
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
        let kv_record_bytes = kv_format.record_bytes(cfg.kv_heads, cfg.head_dim);
        let kv_suffix = if kv_format == cuteafd_loader::families::qwen4::Qwen4KvCache::Fp8 { "_kv_fp8" } else { "" };
        for (owner, &(_, owner_programs, _)) in ranks.iter().enumerate() {
            for cap in ["m64", "m4096"] {
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
        let zeroed = |bytes: usize| -> Result<Dev<'a>> {
            let allocation = Rc::new(Allocation::new(Device { library, id: library.cuda_get_device()? }, bytes.max(256))?);
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
                    kv.push(Some(zeroed(pages * PAGE_ROWS * kv_record_bytes)?));
                    index.push(Some((zeroed(pages * PAGE_ROWS * INDEX_DIM * 2)?,
                        zeroed(pool_pages * PAGE_ROWS * INDEX_DIM * 2)?)));
                }
                Qwen4Attention::Gdn => {
                    kv.push(None);
                    index.push(None);
                }
            }
        }
        let gdn_banks = ranks.iter().enumerate().map(|(owner, &(device, programs, stream))|
            device.own(|| GdnBank::new(library, programs, stream, &cfg, state_map.gdn_layers[owner], slots)))
            .collect::<Result<Vec<_>>>()?;
        ensure!(weights.head.allocations().iter().all(|a| a.device.id == home.id)
            && weights.mixer.iter().all(|a| a.device.id == home.id), "Qwen head must stay on owner zero");
        if let Some(mtp) = &weights.mtp {
            ensure!(mtp.layer.device()? == home.id, "Qwen MTP must stay on owner zero");
        }
        let ple_owner = cfg.ple_layers.first().filter(|&&index| index < owners.len()).map(|&index| owners[index]);
        if let (Some(table), Some(owner)) = (&ple, ple_owner) {
            ensure!(table.scale.buffer.device_id == ranks[owner].0.id,
                "Qwen PLE table must load on its attention owner");
        }
        let (ple_state, ple_replay) = if weights.layers.len() > cfg.ple_layers.first().copied().unwrap_or(usize::MAX) {
            let _ple_scope = ranks[ple_owner.context("Qwen PLE state owner")?].0.enter()?;
            (Some(zeroed(slots * PLE_STATE_ROWS * cfg.hc_width() * 2)?),
             Some(zeroed(REPLAY_ROWS * cfg.hc_width() * 2)?))
        } else {
            (None, None)
        };
        let (mtp_kv, mtp_pending) = if weights.mtp.is_some() {
            (Some((zeroed(pages * PAGE_ROWS * kv_record_bytes)?, zeroed(pages * PAGE_ROWS * INDEX_DIM * 2)?,
                zeroed(pool_pages * PAGE_ROWS * INDEX_DIM * 2)?)),
             Some(zeroed(slots * MTP_PENDING_ROWS * HC * cfg.hidden * 2)?))
        } else {
            (None, None)
        };
        let pool_logical = zeroed(pool_pages * 4)?;
        Ok(Self { quantize_grids, active_owner: Cell::new(0), library, programs, cfg, weights, ple, stream, max_context, prefill_rows, pages, slots, kv_format, kv_record_bytes, kv, state_map, gdn_banks, index, ple_state, ple_replay, ple_pending: RefCell::new(None),
            mtp_kv, mtp_pending,
            last_streams: std::cell::Cell::new((false, 0)), mtp_streams: std::cell::Cell::new((false, 0)), pool_logical,
            pool_logical_host: RefCell::new(vec![0; pool_pages]), pool_pages, workspace: RefCell::new(None),
            decode_workspace: RefCell::new(None), peer_workspace: RefCell::new(None),
            peer_decode_workspace: RefCell::new(None), experts: None, hops: None, profile: RefCell::new([0.0; 2]),
            graphs: RefCell::new(std::collections::HashMap::new()),
            use_graphs: std::env::var("CUTEAFD_QWEN4_GRAPHS").map_or(true, |v| v != "0"),
            startup_graphs: false,
            warming_graphs: Cell::new(false), w8a8_prefill: false, full_prefill_logits: false,
            routes_ready: library.cuda_event_create_ordering()?, l2: None, embedding,
            mtp_drafts: zeroed(MTP_DEFERRED_STEPS * DECODE_ROWS * 4)? })
    }

    pub(super) fn enable_startup_graphs(&mut self) {
        self.startup_graphs = startup_graphs_enabled(
            std::env::var("CUTEAFD_QWEN4_GRAPHS").ok().as_deref(),
            std::env::var("CUTEAFD_QWEN4_STARTUP_GRAPHS").ok().as_deref());
    }

    pub fn set_experts(&mut self, experts: Experts<'a>) {
        self.experts = Some(experts);
    }

    pub fn set_tp2(&mut self,
        experts: [Box<dyn crate::shared::experts::rtx::RtxExpertLayer + 'a>; 2],
        wire: bool, mtp: Option<MtpExperts<'a>>, placement: &cuteafd_loader::placement::Placement,
        spec: cuteafd_loader::placement::HopSpec) -> Result<()> {
        ensure!(self.gdn_banks.len() == 2 && spec.head_gpu == 0 && spec.entry_gpu == 0,
            "Qwen TP2 needs two attention owners and the head on owner zero");
        let ranks = [0, 1].map(|rank| crate::shared::peer_split::RankDevice {
            device: self.gdn_banks[rank].device.id, stream: self.gdn_banks[rank].stream });
        let modes: Vec<_> = self.state_map.layers.iter().map(|home| cuteafd_loader::placement::LayerMode::Whole {
            gpu: home.owner as u8, ffn: cuteafd_loader::placement::FfnMode::Owner }).collect();
        ensure!(placement.layers.iter().all(|layer| layer.experts == cuteafd_loader::placement::ExpertHome::RtxTp2)
            && placement.tp2.as_ref().is_some_and(|range| range.first == 0 && range.layers == modes.len()),
            "Qwen Spark-free dual execution requires TP2 halves at every routed layer");
        ensure!(placement.layers.iter().map(|layer| layer.mode).eq(modes.iter().copied()),
            "Qwen executor attention owners disagree with placement");
        let hops = cuteafd_loader::placement::plan_hops(&modes, &spec);
        ensure!(placement.hops == hops && hops.len() == 2, "Qwen placement must charge cutover and exit");
        let _scope = self.gdn_banks[0].device.enter()?;
        for bank in &self.gdn_banks {
            bank.device.run(|| { bank.programs.load_matching(|name| name.starts_with("qwen4_")).map(|_| ()) })?;
        }
        let routed = QwenTp2::new(self.library, ranks, experts, self.cfg.hidden, self.cfg.topk,
            self.prefill_rows.max(DECODE_ROWS), wire)?;
        let link = crate::shared::peer_split::hop::HopLink::new(self.library, ranks, &hops, spec)?;
        self.hops = Some(link);
        self.experts = Some(Experts::Tp2 { routed: RefCell::new(routed), mtp });
        Ok(())
    }

    /// Pre-capture every reachable bucket/geometry on no-storage rows, without expert traffic.
    pub fn warm_decode_graphs(&self, sequences: usize, speculation: bool) -> Result<usize> {
        if !self.use_graphs || !self.startup_graphs { return Ok(0); }
        ensure!(sequences <= *PLAIN_BUCKETS.last().unwrap(),
            "Qwen startup graphs support at most 16 concurrent sequences");
        check_bucket_thresholds(PLAIN_BUCKETS, DECODE_PROJECTION_THRESHOLDS)?;
        check_bucket_thresholds(SPEC_BUCKETS, DECODE_PROJECTION_THRESHOLDS)?;
        let shapes = serving_graph_shapes(self.max_context, self.pages, self.cfg.dense_context(), sequences, speculation);
        let segments = self.weights.layers.len() + if self.gdn_banks.len() == 2 { 3 } else { 1 };
        let expected = shapes.len() * segments;
        tracing::info!(graphs = expected, shapes = shapes.len(), plain_rows = ?PLAIN_BUCKETS, spec_rows = ?SPEC_BUCKETS,
            "Qwen startup decode graph admission");
        if self.decode_workspace.borrow().is_none() {
            *self.decode_workspace.borrow_mut() = Some(self.on_owner(0, || self.workspace(DECODE_ROWS, true, DECODE_ROWS))?);
        }
        if self.gdn_banks.len() == 2 && self.peer_decode_workspace.borrow().is_none() {
            *self.peer_decode_workspace.borrow_mut() = Some(self.on_owner(1, || self.workspace(DECODE_ROWS, true, DECODE_ROWS))?);
        }
        self.drain_state()?;
        let sample = || self.gdn_banks.iter().map(|bank| bank.device.run(|| {
            Ok(self.library.cuda_physical_memory_info()?.0)
        })).collect::<Result<Vec<_>>>();
        let before = sample()?;
        let started = std::time::Instant::now();
        self.warming_graphs.set(true);
        let captured = (|| -> Result<()> {
            for &(rows, spec, geometry) in &shapes {
                let mut tables = StepTables { decode: true, spec, long: geometry.long,
                    pool_width: geometry.pool_width, page_stride: geometry.page_stride,
                    page_width: geometry.page_stride, pool_stride: geometry.pool_stride, ..Default::default() };
                let mut tokens = Vec::new();
                pad_decode_tables(&mut tables, &mut tokens, rows, self.ple.as_ref().map_or(0, |_| self.cfg.ple_rows()));
                self.step(&tables, &tokens, rows, None, None, &Default::default(), true)?;
            }
            // SAFETY: queued captures/replays drain before reporting memory or publishing readiness.
            self.drain_state()
        })();
        let drained = self.drain_state();
        self.warming_graphs.set(false);
        captured?;
        drained?;
        let graphs = self.captured_graphs();
        ensure!(graphs == expected, "Qwen startup captured {graphs} graphs, expected {expected}");
        for &(rows, spec, geometry) in &shapes {
            for segment in 0..segments {
                ensure!(self.graphs.borrow().contains_key(&GraphKey { segment, rows, spec,
                    long: geometry.long, pool_width: geometry.pool_width, page_stride: geometry.page_stride,
                    pool_stride: geometry.pool_stride }), "Qwen startup graph coverage missing");
            }
        }
        let after = sample()?;
        let bytes: Vec<_> = before.iter().zip(&after).map(|(&before, &after)| before as i64 - after as i64).collect();
        tracing::info!(graphs, elapsed_ms = started.elapsed().as_millis() as u64,
            graph_bytes = bytes.iter().sum::<i64>(), graph_bytes_by_owner = ?bytes,
            "Qwen decode graphs captured at startup");
        Ok(graphs)
    }

    /// Compare real logits and persistent cache/state bytes, including unused storage.
    /// Run only before serving: this diagnostic owns its allocator and state slots.
    pub fn check_decode_padding(&self, tokens: &[u32]) -> Result<()> {
        ensure!(self.use_graphs && self.startup_graphs && self.weights.layers.len() == self.cfg.layers,
            "Qwen padding check needs all layers and startup graphs");
        ensure!(tokens.len() >= 69 && self.max_context >= 69 && self.slots >= 10 && self.pages >= 40,
            "Qwen padding check needs 69 tokens, ten slots and forty pages");
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
        for (sequences, width, spec) in [(5, 1, false), (9, 1, false),
            (1, 3, true), (1, 5, true), (1, 9, true), (1, 17, true), (1, 25, true), (1, 33, true)] {
            let mut allocator = Allocator::new(self.pages, self.slots, &self.cfg);
            let mut placements: Vec<_> = (0..sequences).map(|_| allocator.admit(65)).collect::<Result<_>>()?;
            for placement in &mut placements {
                self.prefill_device(placement, &tokens[..32], None, None, 1)?;
            }
            let original = placements.clone();
            let mut buffers = self.persistent_state_buffers();
            for [kv, keys, pools] in self.paged_buffers() {
                // Snapshot complete pools: any masked-row write, even outside the live sequences, fails.
                buffers.extend([kv, keys, pools]);
            }
            let before = snapshot(&buffers)?;
            let input = &tokens[32..32 + width];
            let mut groups: Vec<_> = placements.iter_mut().map(|p| (p, input)).collect();
            let plain = self.verify_device_ungraphed(&mut groups, spec)?
                .context("Qwen padding check logits")?.to_host(self.library)?;
            let plain_state = snapshot(&buffers)?;
            restore(&buffers, &before)?;
            placements = original;
            let mut groups: Vec<_> = placements.iter_mut().map(|p| (p, input)).collect();
            let logits = self.verify_device(&mut groups, spec)?.context("Qwen padding check logits")?;
            let padded = logits.to_host(self.library)?;
            let rows = sequences * width;
            let (ids, statuses) = logits.greedy.context("Qwen padded greedy selection missing")?;
            let device = self.library.cuda_get_device()?;
            let mut selected = vec![0u8; rows * 4];
            let mut status = vec![0u8; rows * 4];
            for (ptr, bytes) in [(ids, &mut selected), (statuses, &mut status)] {
                self.library.copy_d2h(bytes, cuteafd_ffi::CuteafdDeviceBuffer {
                    ptr: ptr.cast_mut(), bytes: rows * 4, device_id: device, ..Default::default()
                })?;
            }
            ensure!(status.iter().all(|&byte| byte == 0), "Qwen padded greedy selection/status failed");
            for (row, id) in selected.chunks_exact(4).enumerate() {
                let values = &padded[row * self.cfg.vocab_size..][..self.cfg.vocab_size];
                let expected = (1..values.len()).fold(0, |best, i| if values[i] > values[best] { i } else { best });
                ensure!(u32::from_le_bytes(id.try_into()?) as usize == expected, "Qwen padded greedy id differs");
            }
            ensure!(plain.len() == padded.len() && plain.iter().zip(&padded).all(|(a, b)| a.to_bits() == b.to_bits()),
                "Qwen decode padding {rows}->{} changed real-row logits", decode_bucket(rows, spec));
            ensure!(snapshot(&buffers)? == plain_state,
                "Qwen decode padding {rows}->{} changed persistent cache/state bytes", decode_bucket(rows, spec));
            tracing::info!(rows, spec, bucket = decode_bucket(rows, spec), bytes = plain.len() * 4,
                "Qwen padded decode logits and cache/state byte-exact");
        }
        // Exercise the serving trim helper before comparing exact-width arithmetic.
        let mut sequences = vec![tokens[32..69].to_vec()];
        let before_rows = sequences[0].len();
        let limit = self.copy_verify_row_limit(before_rows, sequences.len(), false);
        super::serve::trim_copy_rows(&mut sequences, limit);
        let input = sequences[0].as_slice();
        ensure!(before_rows == 37 && input.len() == 32 && input == &tokens[32..64],
            "Qwen copy trim byte gate must preserve the 37->32 proposal prefix");
        let mut allocator = Allocator::new(self.pages, self.slots, &self.cfg);
        let mut placement = allocator.admit(69)?;
        self.prefill_device(&mut placement, &tokens[..32], None, None, 1)?;
        let original = placement.clone();
        let mut buffers = self.persistent_state_buffers();
        for [kv, keys, pools] in self.paged_buffers() { buffers.extend([kv, keys, pools]); }
        let before = snapshot(&buffers)?;
        let exact = self.verify_device_ungraphed(&mut [(&mut placement, input)], true)?
            .context("Qwen exact copy-trim logits")?.to_host(self.library)?;
        let exact_state = snapshot(&buffers)?;
        restore(&buffers, &before)?;
        placement = original;
        let trimmed = self.verify_device(&mut [(&mut placement, input)], true)?
            .context("Qwen bucketed copy-trim logits")?.to_host(self.library)?;
        ensure!(exact.len() == trimmed.len() && exact.iter().zip(&trimmed).all(|(a, b)| a.to_bits() == b.to_bits()),
            "Qwen copy trim 37->32 changed exact-width real-row logits");
        ensure!(snapshot(&buffers)? == exact_state, "Qwen copy trim 37->32 changed persistent cache/state bytes");
        tracing::info!(before_rows, rows = input.len(), bucket = decode_bucket(input.len(), true),
            bytes = exact.len() * 4, "Qwen copy trim logits and cache/state byte-exact");
        self.drain_state()
    }

    /// Copy proposals may shrink to a lower spec bucket, never dropping a sequence.
    pub(crate) fn copy_verify_row_limit(&self, rows: usize, sequences: usize, diagnostic: bool) -> usize {
        if self.startup_graphs && self.use_graphs && !diagnostic {
            copy_row_limit(rows, sequences)
        } else { rows }
    }

    /// Physical row extent for an ordinary serving verify; diagnostics are ungraphed.
    pub(crate) fn verify_bucket_rows(&self, rows: usize, spec: bool, diagnostic: bool) -> usize {
        if self.startup_graphs && self.use_graphs && !diagnostic { decode_bucket(rows, spec) } else { rows }
    }

    pub fn captured_graphs(&self) -> usize {
        self.graphs.borrow().len()
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

    /// Before a sequence's first step: zeroes its state slot and maps its pool pages.
    pub(crate) fn start(&self, placement: &Qwen4Placement) -> Result<()> {
        self.map_pools(placement)?;
        self.reset_slot(placement.slot)
    }

    /// Records each of `placement`'s pool pages' logical page (the index expansion reads it).
    /// A restored sequence maps its pages before its first step as a fresh one does; shared
    /// pages keep the value they have.
    pub fn map_pools(&self, placement: &Qwen4Placement) -> Result<()> {
        let mut host = self.pool_logical_host.borrow_mut();
        let mut changed = false;
        for (logical, &page) in placement.pool_pages.iter().enumerate() {
            let page = usize::try_from(page)?;
            ensure!(page < self.pool_pages, "pool page {page} out of range");
            changed |= std::mem::replace(&mut host[page], logical as i32) != logical as i32;
        }
        if changed {
            // Entries other sequences' queued steps read keep their values.
            self.library.copy_h2d(cuteafd_ffi::CuteafdDeviceBuffer { bytes: host.len() * 4, ..self.pool_logical.buffer },
                bytes_of(&host[..]))?;
        }
        Ok(())
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

    /// Retain every backing allocation the host tier can read or restore.
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
        owners
    }

    pub(crate) fn state_stream(&self, device: i32) -> Result<*mut c_void> {
        self.gdn_banks.iter().find(|bank| bank.device.id == device)
            .map(|bank| bank.stream).context("Qwen snapshot device without an execution stream")
    }

    pub(crate) fn drain_state(&self) -> Result<()> {
        for bank in &self.gdn_banks {
            // SAFETY: each bank owns its execution stream and live state allocations.
            bank.device.run(|| unsafe { self.library.cuda_stream_synchronize(bank.stream) })?;
        }
        Ok(())
    }

    fn conv_slot_bytes(cfg: &Qwen4Config) -> usize {
        (cfg.conv_kernel - 1) * cfg.gdn_conv_width() * 2
    }

    fn state_slot_bytes(cfg: &Qwen4Config) -> usize {
        cfg.gdn_value_heads * cfg.gdn_head_dim * cfg.gdn_head_dim * 4
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
        let (conv, state) = (Self::conv_slot_bytes(&self.cfg), Self::state_slot_bytes(&self.cfg));
        for layer in &self.state_map.layers {
            if let Some(ord) = layer.gdn_ordinal {
                let bank = &self.gdn_banks[layer.owner];
                if let (Some(c), Some(s)) = (&bank.conv, &bank.state) {
                    regions.push(Self::region(c, (ord * self.slots + slot) * conv, conv));
                    regions.push(Self::region(s, (ord * self.slots + slot) * state, state));
                }
            }
        }
        if let Some(ple) = &self.ple_state {
            let per = ple.buffer.bytes / self.slots;
            regions.push(Self::region(ple, slot * per, per));
        }
        regions
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
        let (Some(c), Some(s), Some(r)) = (&bank.conv, &bank.state, &bank.replay) else {
            anyhow::bail!("GDN layer without state pools")
        };
        Ok([Self::region(c, ord * self.slots * Self::conv_slot_bytes(&self.cfg), 0).ptr,
            Self::region(s, ord * self.slots * Self::state_slot_bytes(&self.cfg), 0).ptr,
            Self::region(r, ord * gdn_replay_bytes(&self.cfg), 0).ptr])
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

    fn alloc(&self, bytes: usize) -> Result<Dev<'a>> {
        Ok(Rc::new(Allocation::new(Device { library: self.library, id: self.library.cuda_get_device()? }, bytes.max(256))?))
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
        let _memory_scope = cuteafd_ffi::memory_ledger::scope("workspace");
        let h = self.cfg.hidden;
        let cap = if decode { "m64" } else { "m4096" };
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
        let table_rows = if decode { t } else { 1 };
        let head_workspace = self.alloc(VOCABULARY_HEAD_WORKSPACE)?;
        let spark = matches!(self.experts, Some(Experts::Spark { .. }));
        let (topk, heads, hd) = (self.cfg.topk, self.cfg.heads, self.cfg.head_dim);
        Ok(Workspace {
            rows: t,
            streams: [self.alloc(t * HC * h * 2)?, self.alloc(t * HC * h * 2)?],
            inject: self.alloc(t * HC * 2)?,
            x: self.alloc(t * h * 2)?,
            delta: self.alloc(t * h * 2)?,
            shared: self.alloc(t * h * 2)?,
            routed: self.alloc(t * h * 2)?,
            positions: self.alloc(t * 8)?,
            rope_positions: self.alloc(t * 12)?,
            block_rope_positions: self.alloc(t * 12)?,
            kv_slots: self.alloc(t * 8)?,
            slots: self.alloc(t * 4)?,
            seq_first: self.alloc(t * 4)?,
            pool_slots: self.alloc(t * 8)?,
            cache_lengths: self.alloc(t * 4)?,
            page_table: self.alloc(table_rows * self.pages * 4)?,
            pool_table: self.alloc(table_rows * self.pool_pages * 4)?,
            ple_ids: self.alloc(t * self.cfg.ple_rows() * 8)?,
            ple_rows: match self.ple.as_ref().filter(|p| p.mapped().is_some()) {
                Some(ple) => {
                    let n = t * self.cfg.ple_rows();
                    let local = self.alloc(n * 8)?;
                    self.put(&local, &(0..n as i64).collect::<Vec<i64>>())?;
                    Some((self.alloc(n * ple.row_bytes)?, local))
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
            logit_rows,
            router_logits: self.alloc(t * self.cfg.experts * 4)?,
            route_ids: self.alloc(t * topk * 4)?,
            route_weights: self.alloc(t * topk * 4)?,
            wire: self.alloc(t * (h + h / 32))?,
            router_host: RefCell::new(HostAllocation::new(self.library,
                if spark { t * (topk * 8 + h + h / 32) } else { 256 })?),
            hidden_rows: self.alloc(t * 4)?,
            argmax: self.alloc(logit_rows * 8)?,
            ids: self.alloc(t * 4)?,
            select: self.alloc(logit_rows * 8)?,
            logits_host: RefCell::new(HostAllocation::new(self.library, logit_rows * self.cfg.vocab_size * 4)?),
            staging: RefCell::new((HostAllocation::new(self.library, 16 * 16
                + t * (8 * 3 + 4 * 4 + 24 + self.cfg.ple_rows() * 8 + (HC + 1) * h * 2)
                + table_rows * (self.pages + self.pool_pages) * 4)?, 0)),
            // SAFETY: the workspace buffer lives in the same struct and drops after the head.
            head: unsafe { self.library.vocabulary_head_rows(head_workspace.buffer.ptr, h as u32, logit_rows as u32,
                self.cfg.vocab_size as u32)? },
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

    /// Starts a step's staged uploads (the previous step's have drained).
    fn begin_staging(&self, w: &Workspace<'_>) -> Result<()> {
        // SAFETY: the engine owns this stream; its earlier copies read the staging bytes.
        unsafe { self.library.cuda_stream_synchronize(self.execution_stream())? };
        w.staging.borrow_mut().1 = 0;
        Ok(())
    }

    /// Like [`Self::begin_staging`], but keeps appending to the staging (no
    /// host wait) while `bytes` more still fit: queued steps (MTP draft
    /// chains) then run back to back.
    fn continue_staging(&self, w: &Workspace<'_>, bytes: usize) -> Result<()> {
        let (capacity, used) = { let staging = w.staging.borrow(); (staging.0.buffer.bytes, staging.1) };
        if used + bytes > capacity {
            self.begin_staging(w)?;
        }
        Ok(())
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
                std::ptr::null(), out.buffer.ptr, self.execution_stream())? };
        }
        Ok(())
    }

    /// Queues `bytes` into `dst` through the workspace's pinned staging.
    fn stage(&self, w: &Workspace<'_>, dst: cuteafd_ffi::CuteafdDeviceBuffer, bytes: &[u8]) -> Result<()> {
        if bytes.is_empty() {
            return Ok(());
        }
        ensure!(bytes.len() <= dst.bytes, "staged upload exceeds its buffer");
        let mut staging = w.staging.borrow_mut();
        let at = staging.1;
        ensure!(at + bytes.len() <= staging.0.buffer.bytes, "step inputs exceed the staging buffer");
        staging.0.bytes_mut()[at..at + bytes.len()].copy_from_slice(bytes);
        let source = cuteafd_ffi::CuteafdHostBuffer {
            // SAFETY: `at` lies inside the pinned staging buffer.
            ptr: unsafe { staging.0.buffer.ptr.cast::<u8>().add(at) }.cast(),
            bytes: bytes.len(),
            ..staging.0.buffer
        };
        // SAFETY: the staged bytes stay untouched until `begin_staging` drains the stream.
        unsafe { self.library.copy_host_buffer_h2d_async(dst, source, bytes.len(), self.execution_stream())? };
        staging.1 = (at + bytes.len()).div_ceil(16) * 16;
        Ok(())
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

    /// Advises the page cache of a mapped PLE table about the rows of
    /// `tokens` appended after `history` (a later step's, e.g. the next
    /// prefill chunk); best effort: never blocks, failures only log.
    pub fn prefetch_ple(&self, history: &NgramHistory, tokens: &[u32]) {
        let Some((ple, mapped)) = self.ple.as_ref().and_then(|p| p.mapped().map(|m| (p, m))) else { return };
        let mut ids = Vec::with_capacity(tokens.len() * self.cfg.ple_rows());
        if let Err(error) = ple.hasher.hash(&mut history.clone(), tokens, &mut ids).and_then(|()| mapped.prefetch(&ids)) {
            tracing::warn!("PLE prefetch: {error:#}");
        }
    }

    /// Logs the mapped PLE table's stats since the previous call.
    pub fn log_table_stats(&self, event: &str) {
        if let Some(mapped) = self.ple.as_ref().and_then(|p| p.mapped()) {
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
        let (t, start) = (tokens.len(), placement.len);
        ensure!(t > 0 && t <= self.prefill_rows && start + t <= self.max_context, "prefill of {t} rows at {start}");
        if start == 0 {
            self.start(placement)?;
        }
        let mut tables = StepTables { page_table: placement.pages.clone(), pool_table: placement.pool_pages.clone(),
            page_stride: 0, pool_stride: 0, page_width: placement.pages.len(), ..Default::default() };
        self.rows(placement, tokens, 0, &mut tables)?;
        let media = embedding_media(&[(placement, start, t)])?;
        let logits = self.step(&tables, tokens, logit_rows.clamp(1, t), on_layer, forced, &media, true)?;
        placement.len += t;
        placement.state_len = placement.len;
        Ok(logits)
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
        let rows: usize = sequences.iter().map(|(_, t)| t.len()).sum();
        ensure!(rows > 0 && rows <= DECODE_ROWS, "decode step of {rows} rows");
        let mut tokens: Vec<u32> = sequences.iter().flat_map(|(_, t)| t.iter().copied()).collect();
        let live_tokens = sequences.iter().map(|(p, t)| p.len + t.len()).max().unwrap_or(1);
        let allocated = |n, unit| crate::shared::context::decode_allocation_units(n, live_tokens, unit);
        let page_stride = allocated(sequences.iter().map(|(p, _)| p.pages.len()).max().unwrap_or(1), PAGE_ROWS)
            .next_power_of_two().min(self.pages);
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
                let mut pages = placement.pages.clone();
                pages.resize(page_stride, 0);
                tables.page_table.extend(pages);
                let mut pools = placement.pool_pages.clone();
                pools.resize(pool_stride, 0);
                tables.pool_table.extend(pools);
            }
        }
        tables.pool_width = tables.pool_width.next_power_of_two().min(pool_stride);
        let media = if sequences.iter().any(|(p, t)| p.media.as_ref()
            .is_some_and(|media| media.needed(p.len, p.len + t.len()).next().is_some())) {
            let media_rows: Vec<_> = sequences.iter().map(|(p, t)| (&**p, p.len, t.len())).collect();
            embedding_media(&media_rows)?
        } else { Default::default() };
        let bucketed = self.startup_graphs && self.use_graphs && graphs && on_layer.is_none() && media.indices.is_empty();
        if bucketed {
            pad_decode_tables(&mut tables, &mut tokens, decode_bucket(rows, spec), self.ple.as_ref().map_or(0, |_| self.cfg.ple_rows()));
        }
        let physical_rows = tokens.len();
        let mut logits = self.step(&tables, &tokens, physical_rows, on_layer, None, &media, graphs)?;
        if let Some(logits) = logits.as_mut() {
            // Greedy statuses remain after the physical bucket's ids, not the real rows.
            logits.rows = rows;
        }
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
        ensure!(n <= DECODE_ROWS, "commit of {n} sequences");
        let mut table = vec![0i32; 3 * n];
        for (i, &(slot, first, keep)) in accepted.iter().enumerate() {
            ensure!(first + keep <= REPLAY_ROWS && usize::try_from(slot).is_ok_and(|s| s < self.slots),
                "commit of rows {first}+{keep} in slot {slot}");
            (table[i], table[n + i], table[2 * n + i]) = (slot, i32::try_from(first)?, i32::try_from(keep)?);
        }
        // SAFETY: the engine owns this stream; the previous commit's table is consumed.
        unsafe { self.library.cuda_stream_synchronize(self.execution_stream())? };
        let i32s = |v: usize| -> Result<Scalar> { Ok(Scalar::I32(i32::try_from(v)?)) };
        for bank in &self.gdn_banks {
            bank.device.run(|| bank.commit(&table, n, self.slots))?;
        }
        if let (Some(state), Some(replay)) = (&self.ple_state, &self.ple_replay) {
            let bank = self.gdn_banks.iter().find(|bank| bank.device.id == state.device.id)
                .context("Qwen PLE commit owner")?;
            bank.device.run(|| {
                let program = bank.programs.program("qwen4_ple_commit", &["conv_state", "replay", "tables"])?;
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
            ensure!(slot < self.slots && at + count <= MTP_PENDING_ROWS && first + count <= w.rows,
                "MTP stash of rows {first}+{count} at {at}");
            if count == 0 {
                continue;
            }
            let dst = Self::region(pending, (slot * MTP_PENDING_ROWS + at) * row, count * row);
            let src = Self::region(&w.streams[cur], first * row, count * row);
            // SAFETY: both regions lie inside live buffers; the stream orders the copy.
            unsafe { self.library.copy_d2d_async(dst, src, count * row, self.execution_stream())? };
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
        let mtp = self.weights.mtp.as_ref().context("MTP is not loaded (--mtp)")?;
        let (kv, keys, blocks) = self.mtp_kv.as_ref().context("MTP pools")?;
        let (h, vocab) = (self.cfg.hidden, self.cfg.vocab_size);
        let t: usize = groups.iter().map(|g| g.rows.len()).sum();
        let capacity = if decode { DECODE_ROWS } else { self.prefill_rows };
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
            tables.page_table = groups[0].placement.pages.clone();
            tables.pool_table = groups[0].placement.pool_pages.clone();
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
                    let mut pages = placement.pages.clone();
                    pages.resize(tables.page_stride, 0);
                    tables.page_table.extend(pages);
                    let mut pools = placement.pool_pages.clone();
                    pools.resize(tables.pool_stride, 0);
                    tables.pool_table.extend(pools);
                }
            }
        }
        if decode {
            tables.pool_width = tables.pool_width.next_power_of_two().min(tables.pool_stride);
        }
        let slot = if decode { &self.decode_workspace } else { &self.workspace };
        if slot.borrow().is_none() {
            *slot.borrow_mut() = Some(self.workspace(capacity, decode, if decode { DECODE_ROWS } else { 1 })?);
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
                (w.streams[cur].buffer.ptr, cur ^ 1)
            }
            MtpSource::Target => {
                let (d, cur) = self.last_streams.get();
                ensure!(d == decode, "the MTP reads the last target step of the same workspace");
                (w.streams[cur].buffer.ptr, cur ^ 1)
            }
        };
        // Staged bytes (16-byte aligned tables, then the embedding rows at most).
        let staged = 16 * 16 + t * (8 * 3 + 4 * 4 + 24) + (tables.page_table.len() + tables.pool_table.len()) * 4
            + t * h * 2;
        self.continue_staging(w, staged)?;
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
                ensure!(step < MTP_DEFERRED_STEPS && index.iter().all(|&i| (i as usize) < DECODE_ROWS),
                    "MTP chain step reads draft step {step}");
                self.stage_table(w, &w.ids, index)?;
                let drafts = Self::region(&self.mtp_drafts, step * DECODE_ROWS * 4, DECODE_ROWS * 4);
                // SAFETY: the drafts of `step` and the staged indices are ordered on this
                // stream before the gather; `x` holds t rows.
                unsafe { self.embedding.embed_device_ids(drafts, Some((w.ids.buffer.ptr.cast_const(), index)), t, 1,
                    None, w.x.buffer, self.execution_stream())? };
            }
        }
        if matches!(tokens, MtpTokens::Host(_)) && groups.iter().any(|group| group.placement.media.as_ref()
            .is_some_and(|media| group.rows.iter().any(|row| media.needed(row.position + 1, row.position + 2).next().is_some()))) {
            let media = mtp_embedding_media(groups)?;
            self.inject_media(w, &media, &tables.seq_first, &w.x, t, 1)?;
        }
        let rows = Scalar::I32(t as i32);
        let cap = if decode { "m64" } else { "m4096" };
        self.run("qwen4_mtp_feedback", &[("hidden", src), ("hidden_rows", w.hidden_rows.buffer.ptr),
            ("embed", w.x.buffer.ptr), ("norm_hidden", mtp.norm_hidden.buffer.ptr),
            ("norm_embed", mtp.norm_embed.buffer.ptr), ("fc_hidden", mtp.fc_hidden.buffer.ptr),
            ("fc_embed", mtp.fc_embed.buffer.ptr), ("streams", w.streams[dst].buffer.ptr),
            ("delta", w.delta.buffer.ptr), ("inject", w.inject.buffer.ptr), ("scratch", w.scratch.buffer.ptr)],
            &[rows])?;
        // The embedding branch joins every stream (unit injection), then the attention site.
        self.post_pre(w, dst, &mtp.layer, "attn", rows)?;
        self.attend(w, &mtp.layer, kv.buffer.ptr, keys.buffer.ptr, blocks.buffer.ptr, rows, cap, &tables)?;
        self.post_pre(w, dst ^ 1, &mtp.layer, "mlp", rows)?;
        self.moe(w, self.cfg.layers, &mtp.layer, t, rows, decode)?;
        let mut out = dst;
        self.post(w, &mut out, rows)?;
        self.mtp_streams.set((decode, out));
        if heads.is_empty() {
            return Ok((Vec::new(), None));
        }
        let [norm, down, up] = &mtp.mixer;
        self.run("qwen4_head", &[("streams", w.streams[out].buffer.ptr), ("norm", norm.buffer.ptr),
            ("w_down", down.buffer.ptr), ("w_up", up.buffer.ptr), ("out", w.x.buffer.ptr),
            ("scratch", w.scratch.buffer.ptr)], &[rows])?;
        // The head rows, gathered contiguously into `delta`.
        let row = h * 2;
        for (i, &r) in heads.iter().enumerate() {
            ensure!(r < t, "MTP head row {r} of {t}");
            // SAFETY: row r of `x` and row i of `delta` lie inside [t, H] buffers.
            unsafe {
                self.library.copy_d2d_async(Self::region(&w.delta, i * row, row), Self::region(&w.x, r * row, row),
                    row, self.execution_stream())?;
            }
        }
        let n = heads.len();
        ensure!(n <= DECODE_ROWS, "{n} MTP head rows");
        // The target's head (one representation shared with the drafts).
        self.logits(w, w.delta.buffer.ptr, n)?;
        // SAFETY: the logits and the argmax outputs are live buffers of these shapes.
        unsafe {
            self.library.cuda_logits_argmax_checked_f32_async(
                cuteafd_ffi::CuteafdDeviceBuffer { bytes: n * vocab * 4, ..w.logits.buffer },
                Self::region(&w.argmax, 0, n * 4), Self::region(&w.argmax, n * 4, n * 4), n, vocab, self.execution_stream())?;
        }
        let logits = match output {
            MtpOut::Defer { step } => {
                ensure!(step < MTP_DEFERRED_STEPS, "deferred MTP step {step}");
                let at = Self::region(&self.mtp_drafts, step * DECODE_ROWS * 4, n * 4);
                // SAFETY: both regions are live; the stream orders the copy after the argmax.
                unsafe { self.library.copy_d2d_async(at, Self::region(&w.argmax, 0, n * 4), n * 4, self.execution_stream())? };
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
        ensure!(counts.len() <= MTP_DEFERRED_STEPS && counts.iter().all(|&n| n <= DECODE_ROWS), "deferred drafts");
        let bytes = self.download(&self.mtp_drafts, counts.len() * DECODE_ROWS * 4)?;
        Ok(counts.iter().enumerate().map(|(s, &n)| (0..n).map(|i| {
            let at = (s * DECODE_ROWS + i) * 4;
            u32::from_le_bytes(bytes[at..at + 4].try_into().unwrap())
        }).collect()).collect())
    }

    /// The last MTP step's output streams (BF16 [rows, 4, H]).
    pub fn mtp_output(&self, rows: usize) -> Result<Vec<u8>> {
        let (decode, cur) = self.mtp_streams.get();
        let workspace = if decode { self.decode_workspace.borrow() } else { self.workspace.borrow() };
        let w = workspace.as_ref().context("no MTP step ran")?;
        self.download(&w.streams[cur], rows * HC * self.cfg.hidden * 2)
    }

    fn inject_media(&self, w: &Workspace<'_>, media: &cuteafd_engine::media::MediaChunk,
        seq_first: &[i32], out: &Dev<'_>, rows: usize, copies: usize) -> Result<()> {
        if media.indices.is_empty() { return Ok(()); }
        // Scratch is consumed before norm/feedback; injection drains before the
        // reused sequence table is restored. Graph storage stays unchanged.
        self.library.embedding_injection()?.inject_host(&media.features, &media.indices,
            w.delta.buffer, w.seq_first.buffer, out.buffer, rows, self.cfg.hidden, copies, self.execution_stream())?;
        self.put(&w.seq_first, seq_first)
    }

    pub fn prepare_scoring_prefill(&mut self) -> Result<()> {
        *self.workspace.borrow_mut() = Some(self.workspace(self.prefill_rows, false, self.prefill_rows)?);
        self.full_prefill_logits = true;
        Ok(())
    }

    fn step(&self, tables: &StepTables, tokens: &[u32], logit_rows: usize, mut on_layer: LayerHook<'_>,
        forced: Forced<'_>, media: &cuteafd_engine::media::MediaChunk, graphs: bool) -> Result<Option<DeviceLogits>> {
        ensure!(self.gdn_banks.len() == 1 || self.hops.is_some(),
            "Qwen dual owner execution needs TP2 exchange and owner graph admission");
        if self.gdn_banks.len() == 2 {
            return self.step_placed(tables, tokens, logit_rows, on_layer, forced, media, graphs);
        }
        let (h, t) = (self.cfg.hidden, tables.kv_slots.len());
        let (slot, capacity) = if tables.decode { (&self.decode_workspace, DECODE_ROWS) } else { (&self.workspace, self.prefill_rows) };
        if slot.borrow().as_ref().is_some_and(|w| w.logit_rows < logit_rows) {
            *slot.borrow_mut() = None;
        }
        if slot.borrow().is_none() {
            let rows = if tables.decode { DECODE_ROWS } else { logit_rows.max(1) };
            *slot.borrow_mut() = Some(self.workspace(capacity, tables.decode, rows)?);
        }
        let workspace = slot.borrow();
        let w = workspace.as_ref().context("workspace")?;
        ensure!(t <= w.rows && logit_rows <= t, "step exceeds the workspace");
        self.begin_staging(w)?;
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
        match self.ple.as_ref().and_then(|p| p.mapped()) {
            // Gathers while the layers before the PLE layer are enqueued (`finish_ple`).
            Some(mapped) => {
                // A failed step's rows are dropped (their gather waited) first.
                drop(self.ple_pending.borrow_mut().take());
                *self.ple_pending.borrow_mut() = Some(mapped.begin(&tables.ple_ids, tables.decode)?);
            }
            None => self.stage_table(w, &w.ple_ids, &tables.ple_ids)?,
        }
        // Streams start as four copies of the embedding.
        let row = h * 2;
        ensure!(tokens.len() == t, "{} tokens for a {t}-row step", tokens.len());
        let graphed = graphs && self.use_graphs && tables.decode && on_layer.is_none() && forced.is_none() && media.indices.is_empty();
        self.stage_embedding(w, tokens, HC, &w.streams[0], graphed)?;
        self.inject_media(w, media, &tables.seq_first, &w.streams[0], t, HC)?;
        let rows = Scalar::I32(t as i32);
        if graphed {
            return self.decode_graphed(w, tables, t, rows, logit_rows);
        }
        let cap = if tables.decode { "m64" } else { "m4096" };
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
            // SAFETY: the engine owns this stream.
            unsafe { self.library.cuda_stream_synchronize(self.execution_stream())? };
            return Ok(None);
        }
        self.head(w, &w.streams[cur], t, rows, logit_rows)?;
        Ok(Some(self.device_logits(w, logit_rows, false)))
    }

    fn stage_step_tables(&self, w: &Workspace<'_>, tables: &StepTables) -> Result<()> {
        self.begin_staging(w)?;
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
        if self.ple.as_ref().and_then(|p| p.mapped()).is_none() {
            self.stage_table(w, &w.ple_ids, &tables.ple_ids)?;
        }
        Ok(())
    }

    /// Unfused ownership transitions: post on the old owner, hop, then pre on
    /// the new owner. Same-owner boundaries retain the fused post/pre path.
    #[allow(clippy::too_many_arguments)]
    fn step_placed(&self, tables: &StepTables, tokens: &[u32], logit_rows: usize, mut on_layer: LayerHook<'_>,
        forced: Forced<'_>, media: &cuteafd_engine::media::MediaChunk, graphs: bool) -> Result<Option<DeviceLogits>> {
        let graphed = graphs && self.use_graphs && tables.decode
            && on_layer.is_none() && forced.is_none() && media.indices.is_empty();
        let hops = self.hops.as_ref().context("Qwen dual residual hops")?;
        let (t, h) = (tables.kv_slots.len(), self.cfg.hidden);
        ensure!(tokens.len() == t && logit_rows <= t, "invalid Qwen owner step rows");
        let capacity = if tables.decode { DECODE_ROWS } else { self.prefill_rows };
        for owner in 0..2 {
            self.on_owner(owner, || {
                let slot = self.workspace_slot(owner, tables.decode)?;
                if slot.borrow().as_ref().is_some_and(|w| w.logit_rows < logit_rows) {
                    self.drain_state()?;
                    *slot.borrow_mut() = None;
                }
                if slot.borrow().is_none() {
                    *slot.borrow_mut() = Some(self.workspace(capacity, tables.decode,
                        if tables.decode { DECODE_ROWS } else { logit_rows.max(1) })?);
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
        if let Some(mapped) = self.ple.as_ref().and_then(|p| p.mapped()) {
            drop(self.ple_pending.borrow_mut().take());
            *self.ple_pending.borrow_mut() = Some(mapped.begin(&tables.ple_ids, tables.decode)?);
        }
        self.on_owner(0, || {
            self.stage_embedding(work[0], tokens, HC, &work[0].streams[0], graphed)?;
            self.inject_media(work[0], media, &tables.seq_first, &work[0].streams[0], t, HC)
        })?;
        let rows = Scalar::I32(i32::try_from(t)?);
        let cap = if tables.decode { "m64" } else { "m4096" };
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
            if let Some(next) = next {
                let next_owner = next_owner.context("Qwen next attention owner")?;
                if next_owner != owner {
                    let ticket = hops.send(0, 0, w.streams[cur[owner]].buffer.ptr, t)?;
                    let landed = hops.land(ticket)?;
                    self.on_owner(next_owner, || {
                        let dst = work[next_owner].streams[cur[next_owner]].buffer;
                        let src = cuteafd_ffi::CuteafdDeviceBuffer { ptr: landed, ..dst };
                        // SAFETY: hop wait precedes this same-owner copy into persistent streams.
                        unsafe { self.library.copy_d2d_async(dst, src, t * HC * h * 2, self.execution_stream()) }
                    })?;
                }
                self.on_owner(next_owner, || {
                    if self.cfg.ple_layers.contains(&(index + 1)) { self.finish_ple(work[next_owner])?; }
                    self.enter(work[next_owner], &mut cur[next_owner], (!materialize).then_some(()), next,
                        index + 1, rows, tables.spec)
                })?;
            }
            crate::shared::console::layer_mark(index);
        }
        let ticket = hops.send(1, 0, work[1].streams[cur[1]].buffer.ptr, t)?;
        let landed = hops.land(ticket)?;
        self.on_owner(0, || {
            let dst = work[0].streams[cur[0]].buffer;
            let src = cuteafd_ffi::CuteafdDeviceBuffer { ptr: landed, ..dst };
            // SAFETY: exit wait precedes the head/MTP owner-zero stream copy.
            unsafe { self.library.copy_d2d_async(dst, src, t * HC * h * 2, self.execution_stream())?; }
            self.last_streams.set((tables.decode, cur[0]));
            if layers.len() < self.cfg.layers { self.drain_state()?; return Ok(None); }
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
        let real_rows = if self.startup_graphs {
            tables.positions.iter().take_while(|&&position| position >= 0).count()
        } else { t };
        ensure!(tables.positions[real_rows..].iter().all(|&position| position < 0), "Qwen decode mask tail");
        let expert_rows = Scalar::I32(i32::try_from(real_rows)?);
        let key = |segment| GraphKey { segment, rows: t, spec: tables.spec, long: tables.long,
            pool_width: tables.pool_width, page_stride: tables.page_stride, pool_stride: tables.pool_stride };
        let mut cur = [0; 2];
        for (index, layer) in layers.iter().enumerate() {
            let owner = self.state_map.layers[index].owner;
            let w = work[owner];
            let boundary = index > 0 && self.state_map.layers[index - 1].owner != owner;
            if boundary {
                self.on_owner(0, || self.replay(key(n + 1), || {
                    let mut c = cur[0];
                    self.post(work[0], &mut c, rows)
                }))?;
                cur[0] ^= 1;
                let ticket = hops.send(0, 0, work[0].streams[cur[0]].buffer.ptr, t)?;
                let landed = hops.land(ticket)?;
                self.on_owner(1, || {
                    let dst = w.streams[cur[1]].buffer;
                    let src = cuteafd_ffi::CuteafdDeviceBuffer { ptr: landed, ..dst };
                    // SAFETY: cutover land orders this copy before owner-one attention graph.
                    unsafe { self.library.copy_d2d_async(dst, src, t * HC * self.cfg.hidden * 2, self.execution_stream()) }
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
                            std::ptr::null(), w.streams[0].buffer.ptr, self.execution_stream())?; }
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
                        Qwen4Attention::Gdn => self.gdn(w, index, layer, rows, "m64", tables.spec)?,
                        Qwen4Attention::Full => self.full(w, index, layer, rows, "m64", tables)?,
                    }
                    self.post_pre(w, c, layer, "mlp", rows)?;
                    if self.startup_graphs { Ok(()) } else { self.moe_front(w, index, layer, t, rows) }
                })?;
                if index == 0 || boundary { cur[owner] ^= 1; }
                let clear_tail = |tail: std::ops::Range<usize>| -> Result<()> {
                    let buffer = Self::region(&w.delta, tail.start * self.cfg.hidden * 2, tail.len() * self.cfg.hidden * 2);
                    // SAFETY: suffix zeroing precedes graph consumption on the owner stream.
                    unsafe { self.library.cuda_zero_bytes_async(buffer, buffer.bytes, self.execution_stream()) }
                };
                if self.warming_graphs.get() { ensure!(real_rows == 0, "Qwen startup real expert rows"); clear_tail(0..t)?; }
                else if self.startup_graphs {
                    real_row_moe(real_rows, t, |real| {
                        self.moe_front(w, index, layer, real, expert_rows)?;
                        self.moe_experts(w, index, real, expert_rows, true)
                    }, clear_tail)?;
                } else { self.moe_experts(w, index, t, rows, true)?; }
                Ok(())
            })?;
            if !self.warming_graphs.get() { crate::shared::console::layer_mark(index); }
        }
        self.on_owner(1, || self.replay(key(n + 2), || {
            let mut c = cur[1];
            self.post(work[1], &mut c, rows)
        }))?;
        cur[1] ^= 1;
        let ticket = hops.send(1, 0, work[1].streams[cur[1]].buffer.ptr, t)?;
        let landed = hops.land(ticket)?;
        self.on_owner(0, || {
            let dst = work[0].streams[cur[0]].buffer;
            let src = cuteafd_ffi::CuteafdDeviceBuffer { ptr: landed, ..dst };
            // SAFETY: exit land precedes persistent root stream copy/head graph.
            unsafe { self.library.copy_d2d_async(dst, src, t * HC * self.cfg.hidden * 2, self.execution_stream())?; }
            self.replay(key(n), || {
                if head {
                    self.head(work[0], &work[0].streams[cur[0]], t, rows, t)?;
                    self.select_greedy(work[0], t)?;
                }
                Ok(())
            })?;
            self.last_streams.set((true, cur[0]));
            if n < self.cfg.layers { self.drain_state()?; return Ok(None); }
            if !head { self.head(work[0], &work[0].streams[cur[0]], t, rows, logit_rows)?; }
            Ok(Some(self.device_logits(work[0], logit_rows, head)))
        })
    }

    /// The stream mixer and lm_head over the last `logit_rows` rows, and with
    /// `greedy` the rows' greedy tokens into `select` (decode graphs).
    fn head(&self, w: &Workspace<'_>, streams: &Dev<'_>, t: usize, rows: Scalar, logit_rows: usize) -> Result<()> {
        let h = self.cfg.hidden;
        let [norm, down, up] = &self.weights.mixer;
        self.run("qwen4_head", &[("streams", streams.buffer.ptr), ("norm", norm.buffer.ptr),
            ("w_down", down.buffer.ptr), ("w_up", up.buffer.ptr), ("out", w.x.buffer.ptr),
            ("scratch", w.scratch.buffer.ptr)], &[rows])?;
        ensure!(logit_rows <= t, "{logit_rows} logits rows of a {t}-row step");
        // SAFETY: row t - logit_rows of the [t, H] mixer output lies inside `x`.
        let x = unsafe { w.x.buffer.ptr.cast::<u8>().add((t - logit_rows) * h * 2) }.cast();
        self.logits(w, x, logit_rows)
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
                w.head.launch(x.cast(), head.buffer.ptr.cast(), w.logits.buffer.ptr.cast(), rows as u32, self.execution_stream())
            },
            Qwen4Head::Fp8 { values, scales } => {
                for (first, n) in fp8_head_spans(rows) {
                    let input = x.cast::<u8>().wrapping_add(first * h * 2).cast();
                    let out = Self::region(&w.logits, first * vocab * 4, n * vocab * 4).ptr;
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
        // Bucket tails are a contiguous suffix; expert batches must retain native geometry.
        let real_rows = if self.startup_graphs {
            tables.positions.iter().take_while(|&&position| position >= 0).count()
        } else { t };
        ensure!(tables.positions[real_rows..].iter().all(|&position| position < 0),
            "Qwen decode mask is not a contiguous tail");
        let expert_rows = Scalar::I32(real_rows as i32);
        let mut cur = 0usize;
        for index in 0..=layers.len() {
            let key = GraphKey { segment: index, rows: t, spec: tables.spec, long: tables.long, pool_width: tables.pool_width,
                page_stride: tables.page_stride, pool_stride: tables.pool_stride };
            let start = cur;
            if self.cfg.ple_layers.contains(&index) {
                self.finish_ple(w)?;
            }
            self.replay(key, || -> Result<()> {
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
                        w.streams[0].buffer.ptr, self.execution_stream())? };
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
                    Qwen4Attention::Gdn => self.gdn(w, index, layer, rows, "m64", spec)?,
                    Qwen4Attention::Full => self.full(w, index, layer, rows, "m64", tables)?,
                }
                self.post_pre(w, c, layer, "mlp", rows)?;
                if self.startup_graphs { Ok(()) } else { self.moe_front(w, index, layer, t, rows) }
            })?;
            cur ^= if index == 0 || index == layers.len() { 1 } else { 0 };
            if index < layers.len() {
                if self.startup_graphs {
                    let clear_tail = |tail: std::ops::Range<usize>| -> Result<()> {
                        let tail = Self::region(&w.delta, tail.start * self.cfg.hidden * 2,
                            tail.len() * self.cfg.hidden * 2);
                        // SAFETY: the masked suffix is inside persistent delta; the next graph
                        // consumes it on this same stream after the zero and expert output.
                        unsafe { self.library.cuda_zero_bytes_async(tail, tail.bytes, self.execution_stream()) }
                    };
                    if self.warming_graphs.get() {
                        ensure!(real_rows == 0, "Qwen startup MoE rows must all be masked");
                        clear_tail(0..t)?;
                    } else {
                        real_row_moe(real_rows, t, |real_rows| {
                            self.moe_front(w, index, &layers[index], real_rows, expert_rows)?;
                            self.moe_experts(w, index, real_rows, expert_rows, true)
                        }, clear_tail)?;
                    }
                } else if !self.warming_graphs.get() {
                    self.moe_experts(w, index, t, rows, true)?;
                }
                if !self.warming_graphs.get() { crate::shared::console::layer_mark(index); }
            }
        }
        self.last_streams.set((true, cur));
        if layers.len() < self.cfg.layers {
            // SAFETY: the engine owns this stream.
            unsafe { self.library.cuda_stream_synchronize(self.execution_stream())? };
            return Ok(None);
        }
        if !head {
            self.head(w, &w.streams[cur], t, rows, logit_rows)?;
        }
        Ok(Some(self.device_logits(w, logit_rows, head)))
    }

    fn replay(&self, key: GraphKey, segment: impl FnOnce() -> Result<()>) -> Result<()> {
        if let Some(graph) = self.graphs.borrow().get(&key) {
            // SAFETY: the graph's pointers are persistent engine buffers.
            return unsafe { self.library.cuda_graph_launch(graph.0, self.execution_stream()) };
        }
        ensure!(!self.startup_graphs || self.warming_graphs.get(),
            "Qwen serving graph was not captured at startup: {key:?}");
        // SAFETY: capture records this stream's launches; nothing in a segment
        // synchronizes the host or allocates.
        unsafe { self.library.cuda_graph_begin_capture(self.execution_stream())? };
        let captured = segment();
        // SAFETY: ends the capture begun above on the same stream.
        let device = Device { library: self.library, id: self.library.cuda_get_device()? };
        let exec = unsafe { self.library.cuda_graph_end_capture(self.execution_stream()) }
            .map(|exec| GraphExec(exec, device));
        captured?;
        let exec = exec?;
        // SAFETY: the new graph reads and writes persistent engine buffers.
        unsafe { self.library.cuda_graph_launch(exec.0, self.execution_stream())? };
        self.graphs.borrow_mut().insert(key, exec);
        Ok(())
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
        self.run("qwen4_hc_pre", &[("residual", w.streams[*cur].buffer.ptr), ("norm", norm), ("w_di", di),
            ("w_up", up), ("y", w.x.buffer.ptr), ("inject", w.inject.buffer.ptr), ("scratch", w.scratch.buffer.ptr)],
            &[rows])
    }

    /// `delta` (the finished sublayer's output) into streams `cur` (written
    /// to the other buffer), then `site`'s input from them.
    fn post_pre(&self, w: &Workspace<'_>, cur: usize, layer: &Qwen4Layer<'_>, site: &str, rows: Scalar) -> Result<()> {
        let [norm, di, up] = Self::site(layer, site)?;
        self.run("qwen4_hc_post_pre", &[("x", w.delta.buffer.ptr), ("residual", w.streams[cur].buffer.ptr),
            ("inject", w.inject.buffer.ptr), ("norm", norm), ("w_di", di), ("w_up", up),
            ("residual_out", w.streams[cur ^ 1].buffer.ptr), ("y", w.x.buffer.ptr), ("scratch", w.scratch.buffer.ptr)],
            &[rows])
    }

    /// `delta` into streams `cur` alone (written to the other buffer).
    fn post(&self, w: &Workspace<'_>, cur: &mut usize, rows: Scalar) -> Result<()> {
        self.run("qwen4_hc_post", &[("x", w.delta.buffer.ptr), ("residual", w.streams[*cur].buffer.ptr),
            ("inject", w.inject.buffer.ptr), ("out", w.streams[*cur ^ 1].buffer.ptr)], &[rows])?;
        *cur ^= 1;
        Ok(())
    }

    /// Mapped PLE table: waits for the step's rows and queues their upload
    /// (outside any graph capture, before the PLE layer's work).
    fn finish_ple(&self, w: &Workspace<'_>) -> Result<()> {
        let (Some(mapped), Some((rows, _))) = (self.ple.as_ref().and_then(|p| p.mapped()), &w.ple_rows) else {
            return Ok(());
        };
        let pending = self.ple_pending.borrow_mut().take().context("the step's PLE rows were not gathered")?;
        // SAFETY: the engine owns this stream; the workspace's rows outlive the step.
        unsafe { mapped.finish(pending, rows.buffer, self.execution_stream())? };
        Ok(())
    }

    fn ple(&self, w: &Workspace<'_>, streams: &Dev<'_>, layer: &Qwen4Layer<'_>, rows: Scalar, spec: bool)
        -> Result<()> {
        let table = self.ple.as_ref().context("PLE layer without the n-gram table (--table-placement)")?;
        let state = self.ple_state.as_ref().context("PLE conv state")?;
        let replay = self.ple_replay.as_ref().context("PLE replay record")?;
        let name = if table.fp8 { "qwen4_ple_fp8" } else { "qwen4_ple_bf16" };
        // Mapped: the step's rows were gathered in id order, so identity ids read them.
        let (ids, gathered) = match &w.ple_rows {
            Some((gathered, local)) => (local.buffer.ptr, gathered.buffer.ptr),
            None => (w.ple_ids.buffer.ptr, table.table),
        };
        self.run(name, &[("streams", streams.buffer.ptr), ("ids", ids), ("table", gathered),
            ("scale", table.scale.buffer.ptr), ("w_kv", layer.ptr("ple.w_kv")?),
            ("norm_key", layer.ptr("ple.norm_key")?), ("norm_query", layer.ptr("ple.norm_query")?),
            ("norm_conv", layer.ptr("ple.norm_conv")?), ("conv_w", layer.ptr("ple.conv_w")?),
            ("conv_state", state.buffer.ptr), ("slots", w.slots.buffer.ptr), ("seq_first", w.seq_first.buffer.ptr),
            ("replay", replay.buffer.ptr), ("scratch", w.scratch.buffer.ptr)], &[rows, Scalar::I32(i32::from(spec))])
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
        if Self::w8(layer) && cap != "m64" {
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
        let mut pointers = vec![("x", w.x.buffer.ptr)];
        pointers.extend(Self::projection(layer, "w_in")?);
        pointers.extend([("conv_w", layer.ptr("conv_w")?), ("a_log", layer.ptr("a_log")?),
            ("dt_bias", layer.ptr("dt_bias")?), ("norm_w", layer.ptr("norm_w")?)]);
        pointers.extend(Self::projection(layer, "w_out")?);
        pointers.extend([("conv_state", conv), ("state", state), ("slots", w.slots.buffer.ptr),
            ("seq_first", w.seq_first.buffer.ptr), ("out", w.delta.buffer.ptr)]);
        // Decode capacities record speculative replay inputs (spec) or advance the state.
        let decode = cap == "m64";
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
        self.attend(w, layer, cache.buffer.ptr, keys.buffer.ptr, blocks.buffer.ptr, rows, cap, tables)
    }

    /// Full attention of `layer` over the record pool `cache`, raw index keys
    /// `keys` and pooled block keys `blocks` (a target layer's or the MTP layer's).
    #[allow(clippy::too_many_arguments)]
    fn attend(&self, w: &Workspace<'_>, layer: &Qwen4Layer<'_>, cache: *mut c_void, keys: *mut c_void,
        blocks: *mut c_void, rows: Scalar, cap: &str, tables: &StepTables) -> Result<()> {
        let mut pointers = vec![("x", w.x.buffer.ptr)];
        pointers.extend(Self::projection(layer, "w_in")?);
        pointers.extend([("q_norm", layer.ptr("q_norm")?), ("k_norm", layer.ptr("k_norm")?),
            ("iq_norm", layer.ptr("iq_norm")?), ("ik_norm", layer.ptr("ik_norm")?),
            ("positions", w.positions.buffer.ptr), ("rope_positions", w.rope_positions.buffer.ptr),
            ("block_rope_positions", w.block_rope_positions.buffer.ptr), ("kv_slots", w.kv_slots.buffer.ptr),
            ("pool_slots", w.pool_slots.buffer.ptr), ("kv_cache", cache), ("token_keys", keys),
            ("index_cache", blocks), ("query", w.query.buffer.ptr), ("gate", w.gate.buffer.ptr),
            ("index_q", w.index_q.buffer.ptr), ("scratch", w.scratch.buffer.ptr)]);
        self.run(&self.kv_program(layer, cap), &pointers, &self.w8_scalars(layer, rows, cap))?;
        if tables.long {
            self.run(&format!("qwen4_index_topk_{cap}"), &[("index_q", w.index_q.buffer.ptr),
                ("positions", w.positions.buffer.ptr), ("index_cache", blocks),
                ("page_table", w.pool_table.buffer.ptr), ("output_indices", w.blocks.buffer.ptr),
                ("scratch", w.topk_scratch.buffer.ptr)], &[rows, Scalar::I32(tables.pool_stride as i32)])?;
        }
        self.run("qwen4_index_expand", &[("positions", w.positions.buffer.ptr), ("blocks", w.blocks.buffer.ptr),
            ("indices", w.indices.buffer.ptr), ("lengths", w.lengths.buffer.ptr)], &[rows])?;
        self.run(&format!("qwen4_{}_{cap}", self.kv_stem("sparse_gqa")), &[("query", w.query.buffer.ptr), ("kv_cache", cache),
            ("positions", w.positions.buffer.ptr), ("page_table", w.page_table.buffer.ptr),
            ("indices", w.indices.buffer.ptr), ("out", w.attn.buffer.ptr), ("scratch", w.scratch.buffer.ptr)],
            &[rows, Scalar::I32(tables.page_width as i32), Scalar::I32(tables.page_stride as i32)])?;
        let mut pointers = vec![("attn", w.attn.buffer.ptr), ("gate", w.gate.buffer.ptr)];
        pointers.extend(Self::projection(layer, "w_o")?);
        pointers.extend([("out", w.delta.buffer.ptr), ("scratch", w.scratch.buffer.ptr)]);
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
        self.run("qwen4_router_scores", &[("x", w.x.buffer.ptr), ("w", layer.ptr("gate")?),
            ("logits", w.router_logits.buffer.ptr)], &[rows])?;
        // SAFETY: logits and route outputs are live buffers of `t` rows.
        unsafe {
            self.library.router_select_softmax(w.router_logits.buffer.ptr, w.route_ids.buffer.ptr,
                w.route_weights.buffer.ptr, t, self.cfg.experts, self.cfg.topk, 1.0, true, self.execution_stream())?;
        }
        // Spark layers run the shared expert during the exchange (spark_moe).
        if !matches!(experts, Experts::Spark { .. }) || index == self.cfg.layers {
            self.shared(w, layer, rows)?;
        }
        let tp2_wire = match experts {
            Experts::Tp2 { routed, mtp } => if index < self.cfg.layers { routed.borrow().wire }
                else { matches!(mtp, Some(MtpExperts::Exl3(_))) },
            _ => false,
        };
        if tp2_wire || matches!(experts, Experts::LocalExl3(_) | Experts::Spark { .. }) {
            let grid = self.quantize_grids[self.active_owner.get()].blocks(t, h);
            self.run("qwen4_expert_input_quant", &[("source_ptr", w.x.buffer.ptr), ("values_ptr", w.wire.buffer.ptr),
                // SAFETY: the scale rows follow the payload inside each wire row.
                ("scale_rows_ptr", unsafe { w.wire.buffer.ptr.cast::<u8>().add(h) }.cast()),
                ("scale_mma_ptr", w.delta.buffer.ptr)], &[rows, Scalar::I32(grid as i32)])?;
        }
        Ok(())
    }

    fn shared(&self, w: &Workspace<'_>, layer: &Qwen4Layer<'_>, rows: Scalar) -> Result<()> {
        self.run("qwen4_shared", &[("x", w.x.buffer.ptr), ("w_gate_up", layer.ptr("shared.w_gate_up")?),
            ("w_down", layer.ptr("shared.w_down")?), ("out", w.shared.buffer.ptr), ("scratch", w.scratch.buffer.ptr)],
            &[rows])
    }

    /// The routed experts of layer `index` (the front ran); leaves
    /// `bf16(routed + shared)` in `delta`.
    fn moe_experts(&self, w: &Workspace<'_>, index: usize, t: usize, rows: Scalar, decode: bool) -> Result<()> {
        let h = self.cfg.hidden;
        match self.experts.as_ref().context("MoE layer without experts")? {
            Experts::Tp2 { routed, mtp } => {
                if index < self.cfg.layers {
                    let mut routed = routed.borrow_mut();
                    let input = if routed.wire { w.wire.buffer.ptr } else { w.x.buffer.ptr };
                    // SAFETY: all owner buffers persist and were produced on this layer's stream.
                    return unsafe { routed.enqueue(self.active_owner.get(), index, t, input,
                        crate::shared::experts::rtx::Routes { ids: w.route_ids.buffer.ptr, weights: w.route_weights.buffer.ptr },
                        w.shared.buffer.ptr, w.delta.buffer.ptr) };
                }
                match mtp.as_ref().context("Qwen TP2 MTP experts missing")? {
                    MtpExperts::Fp8(experts) => {
                        let resident = experts.index_of(index)?;
                        // SAFETY: MTP owner-zero buffers and resident package persist on this stream.
                        unsafe { experts.run(resident, t, w.x.buffer.ptr, w.route_ids.buffer.ptr,
                            w.route_weights.buffer.ptr, w.routed.buffer.ptr, self.execution_stream())?; }
                    }
                    MtpExperts::Exl3(experts) => {
                        let mut experts = experts.borrow_mut();
                        // SAFETY: MTP owner-zero wire/routes/shared and output are stream ordered.
                        unsafe {
                            experts.run(crate::families::deepseek_v4::local::LocalLayer::Stage(0), t,
                                w.wire.buffer.ptr, w.route_ids.buffer.ptr, w.route_weights.buffer.ptr,
                                w.shared.buffer.ptr, self.execution_stream())?;
                            self.library.copy_d2d_async(w.delta.buffer, experts.output.buffer, t * h * 2, self.execution_stream())?;
                        }
                        return Ok(());
                    }
                }
            }
            Experts::Local(local) => {
                self.exchange_window(index, decode, true)?;
                let target;
                let fp8 = if let Some(draft) = local.mtp_experts.as_ref()
                    .filter(|draft| draft.layers.iter().any(|layer| layer.layer == index)) {
                    draft
                } else {
                    local.index_of(index)?;
                    target = local.experts.borrow();
                    &*target
                };
                let resident = fp8.index_of(index)?;
                ensure!(!fp8.wire_input(), "the coordinator FP8 package takes BF16 rows");
                // SAFETY: input rows, route ids, weights and the output are live
                // buffers of `t` rows on this engine's stream.
                unsafe {
                    fp8.run(resident, t, w.x.buffer.ptr, w.route_ids.buffer.ptr, w.route_weights.buffer.ptr,
                        w.routed.buffer.ptr, self.execution_stream())?;
                }
                if local.window.is_some() {
                    // Diagnostic paging may drop this layer before the stream drains.
                    // SAFETY: the engine owns this stream.
                    unsafe { self.library.cuda_stream_synchronize(self.execution_stream())? };
                }
            }
            Experts::LocalExl3(local) => {
                self.exchange_window(index, decode, true)?;
                local.ensure(index, self.execution_stream())?;
                let mut resident = local.resident.borrow_mut();
                let (_, experts) = resident.as_mut().context("EXL3 window")?;
                let layer = if index == self.cfg.layers {
                    crate::families::deepseek_v4::local::LocalLayer::Stage(0)
                } else {
                    crate::families::deepseek_v4::local::LocalLayer::Backbone(index)
                };
                // SAFETY: wire rows, routes and the shared-expert rows are complete in
                // stream order; the output is copied before the window can change.
                unsafe {
                    experts.run(layer, t, w.wire.buffer.ptr,
                        w.route_ids.buffer.ptr, w.route_weights.buffer.ptr, w.shared.buffer.ptr, self.execution_stream())?;
                    self.library.copy_d2d_async(w.delta.buffer, experts.output.buffer, t * h * 2, self.execution_stream())?;
                }
                return Ok(());
            }
            Experts::Spark { transport, runtime, mtp } => {
                if index < self.cfg.layers {
                    return self.spark_moe(w, index, t, rows, decode, &mut transport.borrow_mut(), runtime);
                }
                ensure!(index == self.cfg.layers, "unknown Qwen expert layer {index}");
                match mtp.as_ref().context("Spark backbone needs coordinator-local MTP experts (--mtp)")? {
                    MtpExperts::Fp8(experts) => {
                        let resident = experts.index_of(index)?;
                        // SAFETY: MTP input/routes and its output are live on this stream.
                        unsafe { experts.run(resident, t, w.x.buffer.ptr, w.route_ids.buffer.ptr,
                            w.route_weights.buffer.ptr, w.routed.buffer.ptr, self.execution_stream())? };
                    }
                    MtpExperts::Exl3(experts) => {
                        let mut experts = experts.borrow_mut();
                        // SAFETY: the resident draft owns its workspace; wire/routes/shared
                        // and the copied output are ordered on the engine's stream.
                        unsafe {
                            experts.run(crate::families::deepseek_v4::local::LocalLayer::Stage(0), t,
                                w.wire.buffer.ptr, w.route_ids.buffer.ptr, w.route_weights.buffer.ptr,
                                w.shared.buffer.ptr, self.execution_stream())?;
                            self.library.copy_d2d_async(w.delta.buffer, experts.output.buffer, t * h * 2, self.execution_stream())?;
                        }
                        return Ok(());
                    }
                }
            }
            Experts::SharedOnly => {
                // SAFETY: both are live [t, H] BF16 buffers ordered on the stream.
                unsafe { self.library.copy_d2d_async(w.delta.buffer, w.shared.buffer, t * h * 2, self.execution_stream())? };
                return Ok(());
            }
        }
        self.run("qwen4_add", &[("a", w.routed.buffer.ptr), ("b", w.shared.buffer.ptr), ("out", w.delta.buffer.ptr)],
            &[rows])
    }

    /// Routes and wire rows down, one request to every Spark rank, the BF16
    /// rank partials and the shared expert summed into `delta`.
    #[allow(clippy::too_many_arguments)]
    fn spark_moe(&self, w: &Workspace<'_>, index: usize, t: usize, rows: Scalar, decode: bool,
        transport: &mut SparkLink<'_>,
        runtime: &tokio::runtime::Runtime) -> Result<()> {
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
            self.library.copy_d2h_host_buffer_async(at(0), w.route_ids.buffer, route_bytes, self.execution_stream())?;
            self.library.copy_d2h_host_buffer_async(at(route_bytes), w.route_weights.buffer, route_bytes, self.execution_stream())?;
            self.library.copy_d2h_host_buffer_async(at(2 * route_bytes), w.wire.buffer, wire_bytes, self.execution_stream())?;
            self.library.cuda_event_record(self.routes_ready, self.execution_stream())?;
        }
        // The shared expert queues behind the copies and runs during the
        // exchange (the L2 prefetch behind it); the host waits for the copies only.
        self.shared(w, &self.weights.layers[index], rows)?;
        self.exchange_window(index, decode, false)?;
        // SAFETY: the event was recorded on this engine's stream above.
        unsafe { self.library.cuda_event_synchronize(self.routes_ready)? };
        self.profile.borrow_mut()[0] += timer.elapsed().as_secs_f64();
        let staged = staging.bytes();
        let word = |offset: usize, i: usize| u32::from_le_bytes(staged[offset + i * 4..][..4].try_into().unwrap());
        let routes = (0..t * topk).map(|i| ExpertProtocolV2RouteEntry {
            row_index: (i / topk) as u32, expert_id: word(0, i), gate_weight: f32::from_bits(word(route_bytes, i)),
        }).collect();
        let wire = staged[2 * route_bytes..2 * route_bytes + wire_bytes].to_vec();
        drop(staging);
        let mut request = ExpertProtocolV2Request::new(index as u64 + 1, 17, index as u32, h as u32,
            ExpertV2Dtype::Fp8E4m3Ue8m0K32,
            (0..t as u32).map(|row| ExpertProtocolV2RowDescriptor {
                row_id: u64::from(row), source_kind: kind, source_request_id: 1,
                token_position: u64::from(row), route_offset: row * topk as u32, route_count: topk as u32,
            }).collect(),
            routes, wire)?;
        request.header.flags |= EXPERT_PROTOCOL_V2_FLAG_V41_COMPACT_BF16;
        let ranks = transport.world_size();
        ensure!(ranks <= MAX_RANKS, "{ranks} Spark ranks exceed the reduction planes");
        let timer = std::time::Instant::now();
        runtime.block_on(async {
            let wave = transport.dispatch(&request)?;
            transport.receive(wave, t, self.execution_stream()).await
        })?;
        self.profile.borrow_mut()[1] += timer.elapsed().as_secs_f64();
        // SAFETY: the shared-expert plane and `delta` are live [t, h] BF16
        // buffers; the intake planes are ordered after the wave by `receive`.
        unsafe {
            transport.reduce(w.shared.buffer.ptr.cast(), w.delta.buffer.ptr.cast(), t, self.execution_stream())?;
            self.library.cuda_stream_synchronize(self.execution_stream())
        }
    }
}

impl Drop for Qwen4Engine<'_> {
    fn drop(&mut self) {
        if let Err(error) = self.drain_state() {
            self.library.quarantine_module_after_failed_drain();
            tracing::error!(%error, "Qwen owner streams failed to drain; retaining device storage");
            return;
        }
        if let Some(bank) = self.gdn_banks.first() {
            // SAFETY: the root owner owns this ordering event; all streams have drained.
            let _ = bank.device.run(|| unsafe { self.library.cuda_event_destroy(self.routes_ready) });
        }
    }
}

#[cfg(test)]
mod owner_state_tests {
    use super::*;

    #[test]
    fn tp2_payloads_pad_routes_and_check_extents() -> Result<()> {
        use crate::shared::experts::rtx::PartialDtype;
        for wire in [false, true] {
            for rows in [1, 3, 64, 4096] {
                let p = Tp2Rows::new(2560, 10, rows, wire, PartialDtype::F32)?;
                assert_eq!(p.ids % 16, 0);
                assert_eq!(p.weights % 16, 0);
                assert_eq!(p.bytes % 16, 0);
                assert!(p.ids >= p.input && p.weights >= p.ids + rows * 40);
                assert_eq!(p.partial_bytes, rows * 2560 * 4);
            }
        }
        assert!(Tp2Rows::new(usize::MAX, 10, 64, false, PartialDtype::F32).is_err());
        assert!(Tp2Rows::new(2561, 10, 64, true, PartialDtype::F32).is_err());
        Ok(())
    }

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
                }
                check(&schedule).map_err(|error| anyhow::anyhow!(error.to_string()))?;
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
