//! Full 42-layer Hy4-flash coordinator forward pass (1x RTX TP1 and 2x RTX TP2)
//! supporting golden single-sequence verification and batched multi-sequence serving
//! through the shared `run_prefill_units` (`PrefillHooks`) executor and zero-host-hop
//! `DeviceLogits` vocabulary head handoff.
use super::cache::{Hy4CacheRank, Hy4Placement, Hy4Prefix};
use super::graphs::{Hy4Graphs, Key as GraphKey, Section as GraphSection};
use super::native::Hy4NativeScratch;
use super::programs::{Hy4Programs, MlaSplitPlan};
use super::weights::{Hy4EngramHost, Hy4EngramWave, Hy4RankWeights};
use crate::shared::decode_graph::device::{self as graph_device, Cycle as GraphCycle};
use crate::shared::distributed_head::Handoff;
use crate::shared::experts::rtx::exl3::Exl3Tp2;
use crate::shared::experts::rtx::{ExpertInput, Routes, RtxExpertLayer};
use crate::shared::experts::staging::RouteStaging;
use crate::shared::launch_grid::Fp8QuantizeGrid;
use crate::shared::memory::device::{Allocation, Device, Event, Stream};
use crate::shared::memory::receipts::Inventory;
use crate::shared::memory::Tag;
use crate::shared::prefill_pipeline::{run_prefill_units, FfnRoute, PrefillHooks, Unit, UnitSegment};
use crate::shared::spark_intake::SparkLink;
use crate::shared::spark_waves::{self, WaveTicket, Waves};
use crate::shared::target_head::{self, TargetHead};
use crate::shared::token_io::DeviceLogits;
use crate::shared::token_table::{enqueue as enqueue_table_h2d, CopyPlane};
use anyhow::{ensure, Context, Result};
use cuteafd_ffi::programs::{Program, Programs, Scalar};
use cuteafd_ffi::{Bf16Add, CuteafdDeviceBuffer, F32Add};
use cuteafd_loader::families::hy4_flash::engram::advance_tail;
use cuteafd_loader::families::hy4_flash::execution::{mla_split_scratch_bytes, PAGE_ROWS};
use cuteafd_loader::placement::families::hy4_flash::DECODE_ROWS;
use cuteafd_loader::families::hy4_flash::{Hy4FlashConfig, Hy4KvCache};
use cuteafd_loader::formats::mapped_table::ReadPriority;
use cuteafd_loader::formats::trellis_core::ProjectionScratchLayout;
use cuteafd_loader::plan::checkpoint::Checkpoint;
use cuteafd_transport::{
    expert::EXPERT_PROTOCOL_V2_FLAG_V41_COMPACT_BF16, ExpertProtocolV2Request,
    ExpertProtocolV2RouteEntry, ExpertProtocolV2RowDescriptor, ExpertV2Dtype, ExpertV2SourceKind,
};
use std::cell::{Cell, RefCell};
use std::collections::{BTreeMap, BTreeSet};
use std::rc::Rc;

const H: usize = 3072;
const HC: usize = 4;
const Q_LORA: usize = 1024;
const KV_A_DIM: usize = 576;
const KV_A_PADDED: usize = 640;
const LATENT_H: usize = 1536;
const EXPERTS: usize = 512;
const TOPK: usize = 16;
const IDX_HEADS: usize = 16;
const IDX_DIM: usize = 128;
const CANDIDATE_TOPK: usize = 2048;
/// Cap `idx_scores` at 64 MiB (16M floats); long-context prefill tiles rows automatically.
const MAX_IDX_SCORE_FLOATS: usize = 16 * 1024 * 1024;

/// Which rows of a batch compute terminal `hc_head` + `lm_head` FP32 logits.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum HeadMode {
    /// Skip terminal `hc_head` + `lm_head` (intermediate prefill units).
    None,
    /// Compute logits only for the final token row of each sequence in the batch.
    LastPerSeq,
    /// Compute logits for every row in `first_row..total_rows` in `<= 64`-row `DeviceLogits` tiles.
    Suffix(usize),
}

/// Per-rank scratch workspace for one Hy4-flash prefill or decode unit.
struct RankScratch<'a> {
    ids: Allocation<'a>,
    positions_i64: Allocation<'a>,
    positions_i32: Allocation<'a>,
    dsa_slots_i64: Allocation<'a>,
    swa_slots_i64: Allocation<'a>,
    r_a: Allocation<'a>,
    r_b: Allocation<'a>,
    r_head_in: Allocation<'a>,
    post: Allocation<'a>,
    post_ffn: Allocation<'a>,
    y: Allocation<'a>,
    delta: Allocation<'a>,
    delta_f32: Allocation<'a>,
    delta_peer_f32: Allocation<'a>,
    delta_sum_f32: Allocation<'a>,
    delta_norm: Allocation<'a>,
    // Engram (one staging slot per Engram layer so all 4 upload once before the layer loop)
    engram_raw: Vec<Allocation<'a>>,
    engram_masked: Vec<Allocation<'a>>,
    engram_normed: Allocation<'a>,
    engram_val: Allocation<'a>,
    engram_keys: Allocation<'a>,
    // MLA
    q_a: Allocation<'a>,
    kv_a: Allocation<'a>,
    kv_a_padded: Allocation<'a>,
    q_resid: Allocation<'a>,
    q_raw: Allocation<'a>,
    gate: Allocation<'a>,
    query: Allocation<'a>,
    attn_out: Allocation<'a>,
    lse_out: Allocation<'a>,
    uv_out: Allocation<'a>,
    attn_scratch: Allocation<'a>,
    dsa_page_table: Allocation<'a>,
    swa_page_table: Allocation<'a>,
    swa_seqlens: Allocation<'a>,
    dsa_cu_seqlens: Allocation<'a>,
    swa_cu_seqlens: Allocation<'a>,
    // Indexer
    idx_q_raw: Allocation<'a>,
    idx_q_rot: Allocation<'a>,
    idx_w_raw: Allocation<'a>,
    idx_w_out: Allocation<'a>,
    idx_tk_raw: Allocation<'a>,
    idx_tk: Allocation<'a>,
    idx_group_tok: Allocation<'a>,
    idx_group_pos: Allocation<'a>,
    idx_group_slots: Allocation<'a>,
    idx_scores: Allocation<'a>,
    idx_score_floats: usize,
    idx_limits: Allocation<'a>,
    idx_cand_slots: Allocation<'a>,
    idx_cand_lens: Allocation<'a>,
    // FFN / LatentMoE
    ffn_gate: Allocation<'a>,
    ffn_up: Allocation<'a>,
    ffn_act: Allocation<'a>,
    router_logits: Allocation<'a>,
    router_select: Allocation<'a>,
    route_ids: Allocation<'a>,
    route_weights: Allocation<'a>,
    latent_in: Allocation<'a>,
    wire_in: Allocation<'a>,
    routed_latent: Allocation<'a>,
    routed_out: Allocation<'a>,
    shared_out: Allocation<'a>,
    shared_f32: Allocation<'a>,
    // Sharded head logits tile (<=64 rows * local_vocab F32)
    logits_tile: Allocation<'a>,
    // Full gathered FP32 logits tile on rank 0 (<=64 rows * vocab F32)
    head_logits_f32: Option<Allocation<'a>>,
    // Compact-core serialized scratch (`k8-compact`, `k6-compact`, `fp8-compact`)
    native: Option<Hy4NativeScratch<'a>>,
}

impl<'a> RankScratch<'a> {
    #[allow(clippy::too_many_arguments)]
    fn new(
        device: Device<'a>,
        rank: usize,
        tp: usize,
        max_rows: usize,
        max_seq_pages: usize,
        max_sequences: usize,
        engram_count: usize,
        vocab: usize,
        local_vocab: usize,
        attn_scratch_bytes: usize,
        native_layout: Option<&ProjectionScratchLayout>,
        native_head_layout: Option<&ProjectionScratchLayout>,
    ) -> Result<Self> {
        let alloc = |bytes: usize, label: &'static str| {
            Allocation::new(device, bytes.max(256), Tag::workspace(label))
        };
        let max_heads = 64 / tp;
        let max_inter = 13824 / tp;
        let max_seq_groups = max_seq_pages * (PAGE_ROWS / 4);
        let idx_score_floats = max_rows
            .saturating_mul(max_seq_groups.max(1))
            .min(MAX_IDX_SCORE_FLOATS)
            .max(max_seq_groups.max(1));
        let wire_row = LATENT_H + LATENT_H / 32;
        let native = if native_layout.is_some() || native_head_layout.is_some() {
            Some(Hy4NativeScratch::allocate(device, native_layout, native_head_layout)?)
        } else {
            None
        };
        let mut engram_raw = Vec::with_capacity(engram_count);
        let mut engram_masked = Vec::with_capacity(engram_count);
        for _ in 0..engram_count {
            engram_raw.push(alloc(max_rows * 16 * 96 * 2, "engram-raw")?);
            engram_masked.push(alloc(max_rows, "engram-masked")?);
        }
        let head_logits_f32 = if rank == 0 {
            Some(alloc(64 * vocab * 4, "head-logits-f32")?)
        } else {
            None
        };
        let max_batch_groups = max_rows / 4 + max_sequences + 2;
        let tp2_f32_bytes = if tp == 2 { max_rows * H * 4 } else { 256 };
        Ok(Self {
            ids: alloc(max_rows * 4, "ids")?,
            positions_i64: alloc(max_rows * 8, "pos-i64")?,
            positions_i32: alloc(max_rows * 4, "pos-i32")?,
            dsa_slots_i64: alloc(max_rows * 8, "dsa-slots-i64")?,
            swa_slots_i64: alloc(max_rows * 8, "swa-slots-i64")?,
            r_a: alloc(max_rows * HC * H * 2, "r-a")?,
            r_b: alloc(max_rows * HC * H * 2, "r-b")?,
            r_head_in: alloc(64 * HC * H * 2, "r-head-in")?,
            post: alloc(max_rows * HC * 4, "post")?,
            post_ffn: alloc(max_rows * HC * 4, "post-ffn")?,
            y: alloc(max_rows * H * 2, "y")?,
            delta: alloc(max_rows * H * 2, "delta")?,
            delta_f32: alloc(tp2_f32_bytes, "delta-f32")?,
            delta_peer_f32: alloc(tp2_f32_bytes, "delta-peer-f32")?,
            delta_sum_f32: alloc(tp2_f32_bytes, "delta-sum-f32")?,
            delta_norm: alloc(max_rows * H * 2, "delta-norm")?,
            engram_raw,
            engram_masked,
            engram_normed: alloc(max_rows * 1536 * 2, "engram-normed")?,
            engram_val: alloc(max_rows * H * 2, "engram-val")?,
            engram_keys: alloc(max_rows * HC * H * 2, "engram-keys")?,
            q_a: alloc(max_rows * Q_LORA * 2, "q-a")?,
            kv_a: alloc(max_rows * KV_A_DIM * 2, "kv-a")?,
            kv_a_padded: alloc(max_rows * KV_A_PADDED * 2, "kv-a-padded")?,
            q_resid: alloc(max_rows * Q_LORA * 2, "q-resid")?,
            q_raw: alloc(max_rows * max_heads * 256 * 2, "q-raw")?,
            gate: alloc(max_rows * max_heads * 256 * 2, "gate")?,
            query: alloc(max_rows * max_heads * 576 * 2, "query")?,
            attn_out: alloc(max_rows * max_heads * 512 * 2, "attn-out")?,
            lse_out: alloc(max_rows * max_heads * 4, "lse-out")?,
            uv_out: alloc(max_rows * max_heads * 256 * 2, "uv-out")?,
            attn_scratch: alloc(attn_scratch_bytes, "attn-scratch")?,
            dsa_page_table: alloc(max_rows * max_seq_pages.max(16) * 4, "dsa-page-table")?,
            swa_page_table: alloc(max_sequences * max_seq_pages.max(16) * 4, "swa-page-table")?,
            swa_seqlens: alloc(max_sequences * 4, "swa-seqlens")?,
            dsa_cu_seqlens: alloc((max_rows + 1) * 4, "dsa-cu-seqlens")?,
            swa_cu_seqlens: alloc((max_sequences + 1) * 4, "swa-cu-seqlens")?,
            idx_q_raw: alloc(max_rows * IDX_HEADS * IDX_DIM * 2, "idx-q-raw")?,
            idx_q_rot: alloc(max_rows * IDX_HEADS * IDX_DIM * 2, "idx-q-rot")?,
            idx_w_raw: alloc(max_rows * IDX_HEADS * 2, "idx-w-raw")?,
            idx_w_out: alloc(max_rows * IDX_HEADS * 4, "idx-w-out")?,
            idx_tk_raw: alloc(max_rows * IDX_DIM * 2, "idx-tk-raw")?,
            idx_tk: alloc(max_rows * IDX_DIM * 2, "idx-tk")?,
            idx_group_tok: alloc((max_rows + 4 * max_sequences) * IDX_DIM * 2, "idx-group-tok")?,
            idx_group_pos: alloc(max_batch_groups * 8, "idx-group-pos")?,
            idx_group_slots: alloc(max_batch_groups * 8, "idx-group-slots")?,
            idx_scores: alloc(idx_score_floats * 4, "idx-scores")?,
            idx_score_floats,
            idx_limits: alloc(max_rows * 4, "idx-limits")?,
            idx_cand_slots: alloc(max_rows * CANDIDATE_TOPK * 4, "idx-cand-slots")?,
            idx_cand_lens: alloc(max_rows * 4, "idx-cand-lens")?,
            ffn_gate: alloc(max_rows * max_inter * 2, "ffn-gate")?,
            ffn_up: alloc(max_rows * max_inter * 2, "ffn-up")?,
            ffn_act: alloc(max_rows * max_inter * 2, "ffn-act")?,
            router_logits: alloc(max_rows * EXPERTS * 4, "router-logits")?,
            router_select: alloc(max_rows * EXPERTS * 4, "router-select")?,
            route_ids: alloc(max_rows * TOPK * 4, "route-ids")?,
            route_weights: alloc(max_rows * TOPK * 4, "route-weights")?,
            latent_in: alloc(max_rows * LATENT_H * 2, "latent-in")?,
            wire_in: alloc(max_rows * wire_row, "wire-in")?,
            routed_latent: alloc(max_rows * LATENT_H * 2, "routed-latent")?,
            routed_out: alloc(max_rows * H * 2, "routed-out")?,
            shared_out: alloc(max_rows * H * 2, "shared-out")?,
            shared_f32: alloc(max_rows * H * 4, "shared-f32")?,
            logits_tile: alloc(64 * local_vocab * 4, "logits-tile")?,
            head_logits_f32,
            native,
        })
    }
}

fn launch_proj_out(
    r: &RankRuntime<'_>,
    stem: &str,
    bf16_prog: &Program<'_>,
    rows: usize,
    input: CuteafdDeviceBuffer,
    output: CuteafdDeviceBuffer,
) -> Result<()> {
    if let Some(linear) = r.weights.native.get(stem) {
        let scratch = r.scratch.native.as_ref().and_then(|n| n.scratch.as_ref()).context("Hy4 native core missing scratch")?;
        return linear.launch(rows, input, output, scratch, r.stream.raw);
    }
    // SAFETY: caller ensures device active and all buffers live through stream drain.
    unsafe {
        bf16_prog.launch(
            &[input.ptr, r.weights.weight(stem)?.ptr, output.ptr],
            &[Scalar::I32(rows as i32)],
            r.stream.raw,
        )
    }
}

fn launch_proj_kv_a(
    r: &RankRuntime<'_>,
    stem: &str,
    rows: usize,
    input: CuteafdDeviceBuffer,
    output_576: CuteafdDeviceBuffer,
) -> Result<()> {
    if let Some(linear) = r.weights.native.get(stem) {
        let scratch = r.scratch.native.as_ref().and_then(|n| n.scratch.as_ref()).context("Hy4 native core missing scratch")?;
        linear.launch(rows, input, r.scratch.kv_a_padded.buffer, scratch, r.stream.raw)?;
        // SAFETY: kv_a_padded [rows, 640] BF16 and output_576 [rows, 576] BF16 are disjoint admitted buffers.
        return unsafe {
            r.device.library.copy_d2d_2d_async(
                output_576,
                KV_A_DIM * 2,
                r.scratch.kv_a_padded.buffer,
                KV_A_PADDED * 2,
                KV_A_DIM * 2,
                rows,
                r.stream.raw,
            )
        };
    }
    // SAFETY: caller ensures device active and all buffers live through stream drain.
    unsafe {
        r.programs.proj_kv_a.launch(
            &[input.ptr, r.weights.weight(stem)?.ptr, output_576.ptr],
            &[Scalar::I32(rows as i32)],
            r.stream.raw,
        )
    }
}

fn launch_proj_split(
    r: &RankRuntime<'_>,
    gpus: usize,
    stem: &str,
    bf16_prog: &Program<'_>,
    rows: usize,
    input: CuteafdDeviceBuffer,
    out_tp1_bf16: CuteafdDeviceBuffer,
    out_tp2_f32: CuteafdDeviceBuffer,
) -> Result<()> {
    if let Some(linear) = r.weights.native.get(stem) {
        let scratch = r.scratch.native.as_ref().and_then(|n| n.scratch.as_ref()).context("Hy4 native core missing scratch")?;
        return if gpus == 1 {
            linear.launch(rows, input, out_tp1_bf16, scratch, r.stream.raw)
        } else {
            linear.partial(rows, input, out_tp2_f32, scratch, r.stream.raw)
        };
    }
    let out = if gpus == 1 { out_tp1_bf16.ptr } else { out_tp2_f32.ptr };
    // SAFETY: caller ensures device active and all buffers live through stream drain.
    unsafe {
        bf16_prog.launch(
            &[input.ptr, r.weights.weight(stem)?.ptr, out],
            &[Scalar::I32(rows as i32)],
            r.stream.raw,
        )
    }
}

fn launch_rank_lm_head(
    r: &RankRuntime<'_>,
    count: usize,
    x_buf: CuteafdDeviceBuffer,
    out_buf: CuteafdDeviceBuffer,
) -> Result<()> {
    if let Some(native_head) = r.weights.native.get("lm_head") {
        let head_scratch = r
            .scratch
            .native
            .as_ref()
            .and_then(|n| n.head_scratch.as_ref())
            .context("Hy4 native lm_head missing scratch")?;
        return native_head.launch(count, x_buf, out_buf, head_scratch, r.stream.raw);
    }
    let head = r.weights.head.as_ref().context("missing VocabularyHead")?;
    let lm_head_w = r.weights.weight("lm_head")?;
    // SAFETY: caller ensures device active and buffers live through stream drain/handoff.
    unsafe {
        target_head::launch(
            TargetHead::configured(),
            head,
            x_buf.ptr,
            lm_head_w.ptr,
            out_buf.ptr.cast(),
            count,
            r.stream.raw,
        )
    }
}

struct RankRuntime<'a> {
    /// Must be first so CUDA graphs drain and destroy before any captured storage
    /// or `RankCompletion` event in `onboarded_experts` drops.
    graphs: Hy4Graphs<'a>,
    device: Device<'a>,
    stream: Rc<Stream<'a>>,
    peer_ready: Event<'a>,
    programs: Hy4Programs<'a>,
    mla_splits: MlaSplitPlan,
    weights: Hy4RankWeights<'a>,
    cache: Rc<Hy4CacheRank<'a>>,
    scratch: RankScratch<'a>,
    bf16_add: Bf16Add<'a>,
    f32_add: F32Add<'a>,
    quantize_grid: Fp8QuantizeGrid,
    onboarded_experts: RefCell<Option<Exl3Tp2<'a>>>,
}

impl<'a> RankRuntime<'a> {
    fn engram_compute(&self, p: &str, slot: usize, rows: usize, use_a: bool) -> Result<()> {
        let r_cur = if use_a { self.scratch.r_a.buffer } else { self.scratch.r_b.buffer };
        let ep = format!("{p}.engram_layer");
        let s = [Scalar::I32(rows as i32)];
        // SAFETY: admitted arena weight views & rank scratch live through stream drain.
        unsafe {
            self.programs.engram_head_norm.launch(
                &[
                    self.scratch.engram_raw[slot].buffer.ptr,
                    self.weights.buf(&format!("{ep}.lookup_per_head_norms.packed"))?.ptr,
                    self.scratch.engram_normed.buffer.ptr,
                ],
                &s, self.stream.raw,
            )?;
            self.programs.proj_engram_value.launch(
                &[self.scratch.engram_normed.buffer.ptr, self.weights.weight(&format!("{ep}.value_proj"))?.ptr, self.scratch.engram_val.buffer.ptr],
                &s, self.stream.raw,
            )?;
            self.programs.proj_engram_key.launch(
                &[self.scratch.engram_normed.buffer.ptr, self.weights.buf(&format!("{ep}.key_projs.packed.weight"))?.ptr, self.scratch.engram_keys.buffer.ptr],
                &s, self.stream.raw,
            )?;
            self.programs.engram_gate_add.launch(
                &[
                    r_cur.ptr,
                    self.scratch.engram_keys.buffer.ptr,
                    self.weights.buf(&format!("{ep}.key_projs.packed.bias"))?.ptr,
                    self.scratch.engram_val.buffer.ptr,
                    self.weights.buf(&format!("{ep}.value_proj.bias"))?.ptr,
                    self.weights.buf(&format!("{ep}.norm_key.packed"))?.ptr,
                    self.weights.buf(&format!("{ep}.norm_query.packed"))?.ptr,
                    self.weights.buf(&format!("{ep}.unified_pre_gamma"))?.ptr,
                    self.weights.buf(&format!("{ep}.unified_per_hc_scales.packed"))?.ptr,
                    self.scratch.engram_masked[slot].buffer.ptr,
                ],
                &s, self.stream.raw,
            )
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn attn_pre(
        &self,
        p: &str,
        layer: usize,
        is_sparse: bool,
        local_heads: usize,
        rows: usize,
        use_a: bool,
        kv_cache_buf: CuteafdDeviceBuffer,
    ) -> Result<()> {
        let r_cur = if use_a { self.scratch.r_a.buffer } else { self.scratch.r_b.buffer };
        let s_rows = [Scalar::I32(rows as i32)];
        let inv_freq = if is_sparse { self.weights.inv_freq_dsa.buffer.ptr } else { self.weights.inv_freq_swa.buffer.ptr };
        let kv_slots_ptr = if is_sparse {
            self.scratch.dsa_slots_i64.buffer.ptr
        } else {
            self.scratch.swa_slots_i64.buffer.ptr
        };
        let (w_uk, _) = self.weights.kv_b_absorbed(layer, local_heads)?;
        let (proj_qb, proj_gate, q_abs) = if is_sparse {
            (&self.programs.proj_q_b_dsa, &self.programs.proj_linear_gate_dsa, &self.programs.mla_q_absorb_dsa)
        } else {
            (&self.programs.proj_q_b_swa, &self.programs.proj_linear_gate_swa, &self.programs.mla_q_absorb_swa)
        };
        // SAFETY: all arena weight views, cache planes, and scratch stay live through stream drain.
        unsafe {
            self.programs.ihc_pre.launch(
                &[
                    r_cur.ptr,
                    self.weights.buf(&format!("{p}.hc_attn_layer.hc_pre.hc_fn"))?.ptr,
                    self.weights.buf(&format!("{p}.hc_attn_layer.hc_pre.hc_scale"))?.ptr,
                    self.weights.buf(&format!("{p}.hc_attn_layer.hc_pre.hc_base"))?.ptr,
                    self.weights.buf(&format!("{p}.input_layernorm.weight"))?.ptr,
                    self.scratch.post.buffer.ptr,
                    self.scratch.y.buffer.ptr,
                ],
                &s_rows, self.stream.raw,
            )?;
            launch_proj_out(
                self,
                &format!("{p}.self_attn.q_a_proj"),
                &self.programs.proj_q_a,
                rows,
                self.scratch.y.buffer,
                self.scratch.q_a.buffer,
            )?;
            launch_proj_kv_a(
                self,
                &format!("{p}.self_attn.kv_a_proj_with_mqa"),
                rows,
                self.scratch.y.buffer,
                self.scratch.kv_a.buffer,
            )?;
            self.programs.mla_rank_norm_pack.launch(
                &[
                    self.scratch.q_a.buffer.ptr,
                    self.scratch.kv_a.buffer.ptr,
                    self.weights.buf(&format!("{p}.self_attn.q_a_layernorm.weight"))?.ptr,
                    self.weights.buf(&format!("{p}.self_attn.kv_a_layernorm.weight"))?.ptr,
                    kv_slots_ptr,
                    self.scratch.positions_i64.buffer.ptr,
                    inv_freq,
                    self.scratch.q_resid.buffer.ptr,
                    kv_cache_buf.ptr,
                ],
                &s_rows, self.stream.raw,
            )?;
            launch_proj_out(
                self,
                &format!("{p}.self_attn.q_b_proj"),
                proj_qb,
                rows,
                self.scratch.q_resid.buffer,
                self.scratch.q_raw.buffer,
            )?;
            launch_proj_out(
                self,
                &format!("{p}.self_attn.linear_gate"),
                proj_gate,
                rows,
                self.scratch.y.buffer,
                self.scratch.gate.buffer,
            )?;
            q_abs.launch(
                &[self.scratch.q_raw.buffer.ptr, w_uk, self.scratch.positions_i64.buffer.ptr, inv_freq, self.scratch.query.buffer.ptr],
                &s_rows, self.stream.raw,
            )
        }
    }

    fn dsa_index_wk(&self, p: &str, rows: usize) -> Result<()> {
        let ip = format!("{p}.self_attn.compressed_indexer");
        let s_rows = [Scalar::I32(rows as i32)];
        // SAFETY: arena weight views and scratch live through stream drain.
        unsafe {
            self.programs.proj_indexer_wk.launch(
                &[self.scratch.y.buffer.ptr, self.weights.weight(&format!("{ip}.token_key_projection"))?.ptr, self.scratch.idx_tk_raw.buffer.ptr],
                &s_rows, self.stream.raw,
            )?;
            let tk_w = self.weights.buf(&format!("{ip}.token_key_norm.weight"))?.ptr;
            self.programs.rms_norm_128.launch(
                &[self.scratch.idx_tk_raw.buffer.ptr, tk_w, tk_w, tk_w, tk_w, tk_w, self.scratch.idx_tk.buffer.ptr],
                &s_rows, self.stream.raw,
            )
        }
    }

    fn dsa_index_q(&self, p: &str, rows: usize) -> Result<()> {
        let ip = format!("{p}.self_attn.compressed_indexer");
        let s_rows = [Scalar::I32(rows as i32)];
        let inv_freq = self.weights.inv_freq_dsa.buffer.ptr;
        // SAFETY: arena weight views and scratch live through stream drain.
        unsafe {
            self.programs.proj_indexer_wq_b.launch(
                &[self.scratch.q_resid.buffer.ptr, self.weights.weight(&format!("{ip}.query_projection"))?.ptr, self.scratch.idx_q_raw.buffer.ptr],
                &s_rows, self.stream.raw,
            )?;
            self.programs.proj_indexer_weights.launch(
                &[self.scratch.y.buffer.ptr, self.weights.weight(&format!("{ip}.head_weight_projection"))?.ptr, self.scratch.idx_w_raw.buffer.ptr],
                &s_rows, self.stream.raw,
            )?;
            self.programs.indexer_query_rope.launch(
                &[
                    self.scratch.idx_q_raw.buffer.ptr,
                    self.scratch.idx_w_raw.buffer.ptr,
                    self.scratch.positions_i64.buffer.ptr,
                    inv_freq,
                    self.scratch.idx_q_rot.buffer.ptr,
                    self.scratch.idx_w_out.buffer.ptr,
                ],
                &s_rows, self.stream.raw,
            )
        }
    }

    fn dsa_attn(&self, kv_cache_buf: CuteafdDeviceBuffer, rows: usize, decode: bool) -> Result<()> {
        // SAFETY: cache and scratch live through stream drain.
        unsafe {
            self.programs.mla_dsa_decode.launch(
                &[
                    self.scratch.query.buffer.ptr,
                    kv_cache_buf.ptr,
                    self.scratch.idx_cand_slots.buffer.ptr,
                    self.scratch.idx_cand_lens.buffer.ptr,
                    self.scratch.dsa_cu_seqlens.buffer.ptr,
                    self.scratch.attn_out.buffer.ptr,
                    self.scratch.lse_out.buffer.ptr,
                    self.scratch.attn_scratch.buffer.ptr,
                ],
                &[Scalar::I32(rows as i32), Scalar::I32(rows as i32), Scalar::I64(CANDIDATE_TOPK as i64),
                    Scalar::I32(if decode { self.mla_splits.dsa } else { 1 })],
                self.stream.raw,
            )
        }
    }

    fn swa_attn(
        &self,
        kv_cache_buf: CuteafdDeviceBuffer,
        rows: usize,
        members_len: usize,
        swa_pt_stride: usize,
        decode: bool,
    ) -> Result<()> {
        let (swa_prog, splits) = if decode {
            (&self.programs.mla_swa_decode, self.mla_splits.swa)
        } else {
            (&self.programs.mla_swa_extend, 1)
        };
        // SAFETY: cache and scratch live through stream drain.
        unsafe {
            swa_prog.launch(
                &[
                    self.scratch.query.buffer.ptr,
                    kv_cache_buf.ptr,
                    self.scratch.swa_page_table.buffer.ptr,
                    self.scratch.swa_seqlens.buffer.ptr,
                    self.scratch.swa_cu_seqlens.buffer.ptr,
                    self.scratch.attn_out.buffer.ptr,
                    self.scratch.lse_out.buffer.ptr,
                    self.scratch.attn_scratch.buffer.ptr,
                ],
                &[
                    Scalar::I32(rows as i32),
                    Scalar::I32(members_len as i32),
                    Scalar::I64(swa_pt_stride as i64),
                    Scalar::I32(splits),
                ],
                self.stream.raw,
            )
        }
    }

    fn attn_out(
        &self,
        p: &str,
        layer: usize,
        is_sparse: bool,
        gpus: usize,
        local_heads: usize,
        rows: usize,
    ) -> Result<()> {
        let (_, w_uv) = self.weights.kv_b_absorbed(layer, local_heads)?;
        let (sink_uv, proj_o) = if is_sparse {
            (&self.programs.mla_sink_gate_uv_dsa, &self.programs.proj_o_dsa)
        } else {
            (&self.programs.mla_sink_gate_uv_swa, &self.programs.proj_o_swa)
        };
        let s_rows = [Scalar::I32(rows as i32)];
        // SAFETY: arena weights and scratch live through stream drain.
        unsafe {
            sink_uv.launch(
                &[
                    self.scratch.attn_out.buffer.ptr,
                    w_uv,
                    self.scratch.gate.buffer.ptr,
                    self.scratch.lse_out.buffer.ptr,
                    self.weights.buf(&format!("{p}.self_attn.learnable_sink_param"))?.ptr,
                    self.scratch.uv_out.buffer.ptr,
                ],
                &s_rows, self.stream.raw,
            )?;
        }
        launch_proj_split(
            self,
            gpus,
            &format!("{p}.self_attn.o_proj"),
            proj_o,
            rows,
            self.scratch.uv_out.buffer,
            self.scratch.delta.buffer,
            self.scratch.delta_f32.buffer,
        )
    }

    fn post_attn_ihc(&self, p: &str, rows: usize, use_a: bool) -> Result<()> {
        let (r_in, r_out) = if use_a {
            (self.scratch.r_a.buffer, self.scratch.r_b.buffer)
        } else {
            (self.scratch.r_b.buffer, self.scratch.r_a.buffer)
        };
        let s_rows = [Scalar::I32(rows as i32)];
        // SAFETY: arena weights and scratch live through stream drain.
        unsafe {
            let ppn = self.weights.buf(&format!("{p}.self_attn.post_proj_norm.weight"))?.ptr;
            self.programs.rms_norm_3072.launch(
                &[self.scratch.delta.buffer.ptr, ppn, ppn, ppn, ppn, ppn, self.scratch.delta_norm.buffer.ptr],
                &s_rows, self.stream.raw,
            )?;
            self.programs.ihc_post_pre.launch(
                &[
                    self.scratch.delta_norm.buffer.ptr,
                    r_in.ptr,
                    self.scratch.post.buffer.ptr,
                    self.weights.buf(&format!("{p}.hc_mlp_layer.hc_pre.hc_fn"))?.ptr,
                    self.weights.buf(&format!("{p}.hc_mlp_layer.hc_pre.hc_scale"))?.ptr,
                    self.weights.buf(&format!("{p}.hc_mlp_layer.hc_pre.hc_base"))?.ptr,
                    self.weights.buf(&format!("{p}.post_attention_layernorm.weight"))?.ptr,
                    r_out.ptr,
                    self.scratch.post_ffn.buffer.ptr,
                    self.scratch.y.buffer.ptr,
                ],
                &s_rows, self.stream.raw,
            )
        }
    }

    fn dense_ffn(&self, p: &str, gpus: usize, rows: usize) -> Result<()> {
        let s_rows = [Scalar::I32(rows as i32)];
        launch_proj_out(
            self,
            &format!("{p}.mlp.gate_proj"),
            &self.programs.proj_dense_gate_up,
            rows,
            self.scratch.y.buffer,
            self.scratch.ffn_gate.buffer,
        )?;
        launch_proj_out(
            self,
            &format!("{p}.mlp.up_proj"),
            &self.programs.proj_dense_gate_up,
            rows,
            self.scratch.y.buffer,
            self.scratch.ffn_up.buffer,
        )?;
        // SAFETY: arena weights and scratch live through stream drain.
        unsafe {
            self.programs.post_silu_split_dense.launch(
                &[self.scratch.ffn_gate.buffer.ptr, self.scratch.ffn_up.buffer.ptr, self.scratch.ffn_act.buffer.ptr],
                &s_rows, self.stream.raw,
            )?;
        }
        launch_proj_split(
            self,
            gpus,
            &format!("{p}.mlp.down_proj"),
            &self.programs.proj_dense_down,
            rows,
            self.scratch.ffn_act.buffer,
            self.scratch.delta.buffer,
            self.scratch.delta_f32.buffer,
        )
    }

    fn moe_router_shared(&self, p: &str, gpus: usize, rows: usize, routed_scaling_factor: f32) -> Result<()> {
        let s_rows = [Scalar::I32(rows as i32)];
        // SAFETY: all buffers live through stream drain.
        unsafe {
            self.programs.proj_router.launch(
                &[self.scratch.y.buffer.ptr, self.weights.weight(&format!("{p}.mlp.gate"))?.ptr, self.scratch.router_logits.buffer.ptr],
                &s_rows, self.stream.raw,
            )?;
            let bias = self.weights.buf(&format!("{p}.mlp.gate.e_score_correction_bias"))?.ptr;
            self.device.library.copy_d2d_async(
                self.scratch.router_select.buffer,
                self.scratch.router_logits.buffer,
                rows * EXPERTS * 4,
                self.stream.raw,
            )?;
            self.device.library.router_select(
                self.scratch.router_select.buffer.ptr,
                bias,
                std::ptr::null(),
                std::ptr::null(),
                self.scratch.route_ids.buffer.ptr,
                self.scratch.route_weights.buffer.ptr,
                rows,
                EXPERTS,
                TOPK,
                routed_scaling_factor,
                true,
                self.stream.raw,
            )?;
            self.programs.route_weights.launch(
                &[self.scratch.router_logits.buffer.ptr, bias, self.scratch.route_ids.buffer.ptr, self.scratch.route_weights.buffer.ptr],
                &s_rows, self.stream.raw,
            )?;
            launch_proj_out(
                self,
                &format!("{p}.mlp.routed_expert_down_proj"),
                &self.programs.proj_latent_down,
                rows,
                self.scratch.y.buffer,
                self.scratch.latent_in.buffer,
            )?;
            let grid = self.quantize_grid.blocks(rows, LATENT_H);
            self.programs.expert_input_quant.launch(
                &[
                    self.scratch.latent_in.buffer.ptr,
                    self.scratch.wire_in.buffer.ptr,
                    self.scratch.wire_in.buffer.ptr.wrapping_byte_add(LATENT_H),
                    self.scratch.wire_in.buffer.ptr,
                ],
                &[Scalar::I32(rows as i32), Scalar::I32(grid as i32)],
                self.stream.raw,
            )?;
            launch_proj_out(
                self,
                &format!("{p}.mlp.shared_experts.gate_proj"),
                &self.programs.proj_shared_gate_up,
                rows,
                self.scratch.y.buffer,
                self.scratch.ffn_gate.buffer,
            )?;
            launch_proj_out(
                self,
                &format!("{p}.mlp.shared_experts.up_proj"),
                &self.programs.proj_shared_gate_up,
                rows,
                self.scratch.y.buffer,
                self.scratch.ffn_up.buffer,
            )?;
            self.programs.post_silu_split_shared.launch(
                &[self.scratch.ffn_gate.buffer.ptr, self.scratch.ffn_up.buffer.ptr, self.scratch.ffn_act.buffer.ptr],
                &s_rows, self.stream.raw,
            )?;
            launch_proj_split(
                self,
                gpus,
                &format!("{p}.mlp.shared_experts.down_proj"),
                &self.programs.proj_shared_down,
                rows,
                self.scratch.ffn_act.buffer,
                self.scratch.shared_out.buffer,
                self.scratch.shared_f32.buffer,
            )
        }
    }

    fn local_exl3(&self, layer: usize, gpus: usize, rows: usize, skip_routed_experts: bool) -> Result<()> {
        // SAFETY: all buffers live through stream drain.
        unsafe {
            if !skip_routed_experts {
                let mut guard = self.onboarded_experts.borrow_mut();
                if let Some(experts) = guard.as_mut() {
                    experts.enqueue(
                        layer,
                        rows,
                        ExpertInput::Fp8K32(self.scratch.wire_in.buffer.ptr),
                        Routes {
                            ids: self.scratch.route_ids.buffer.ptr,
                            weights: self.scratch.route_weights.buffer.ptr,
                        },
                        self.stream.raw,
                        &mut || (),
                    )?;
                    let exp_out = CuteafdDeviceBuffer {
                        ptr: experts.output(),
                        bytes: rows * LATENT_H * 4,
                        device_id: self.device.id,
                        flags: 0,
                    };
                    if gpus == 1 {
                        self.device.library.cuda_f32_to_bf16_async(
                            exp_out,
                            self.scratch.routed_latent.buffer,
                            rows * LATENT_H,
                            self.stream.raw,
                        )?;
                    } else {
                        self.device.library.copy_d2d_async(
                            self.scratch.delta_f32.buffer,
                            exp_out,
                            rows * LATENT_H * 4,
                            self.stream.raw,
                        )?;
                    }
                } else {
                    anyhow::bail!(
                        "layer {layer}: no onboarded hy4f:exl3-k45-mul1 expert package loaded (pass --expert-package or --skip-routed-experts for non-expert diagnostics)"
                    );
                }
            } else {
                self.device.library.cuda_zero_bytes_async(self.scratch.routed_latent.buffer, rows * LATENT_H * 2, self.stream.raw)?;
                if gpus == 2 {
                    self.device.library.cuda_zero_bytes_async(self.scratch.delta_f32.buffer, rows * LATENT_H * 4, self.stream.raw)?;
                }
            }
        }
        Ok(())
    }

    fn ffn_combine_post(&self, p: &str, is_dense: bool, rows: usize, use_a: bool) -> Result<()> {
        if !is_dense {
            launch_proj_out(
                self,
                &format!("{p}.mlp.routed_expert_up_proj"),
                &self.programs.proj_latent_up,
                rows,
                self.scratch.routed_latent.buffer,
                self.scratch.routed_out.buffer,
            )?;
            // SAFETY: routed_out and shared_out finished in stream order into disjoint delta.
            unsafe {
                self.bf16_add.launch(
                    self.scratch.routed_out.buffer,
                    self.scratch.shared_out.buffer,
                    self.scratch.delta.buffer,
                    rows * H,
                    self.stream.raw,
                )?;
            }
        }
        let ffn_norm_name = if is_dense {
            format!("{p}.mlp.post_fc2_norm.weight")
        } else {
            format!("{p}.mlp.combine_output_norm.weight")
        };
        let (r_in, r_out) = if use_a {
            (self.scratch.r_a.buffer, self.scratch.r_b.buffer)
        } else {
            (self.scratch.r_b.buffer, self.scratch.r_a.buffer)
        };
        let s_rows = [Scalar::I32(rows as i32)];
        // SAFETY: arena weights and scratch live through stream drain.
        unsafe {
            let nw = self.weights.buf(&ffn_norm_name)?.ptr;
            self.programs.rms_norm_3072.launch(
                &[self.scratch.delta.buffer.ptr, nw, nw, nw, nw, nw, self.scratch.delta_norm.buffer.ptr],
                &s_rows, self.stream.raw,
            )?;
            self.programs.ihc_post.launch(
                &[self.scratch.delta_norm.buffer.ptr, r_in.ptr, self.scratch.post_ffn.buffer.ptr, r_out.ptr],
                &s_rows, self.stream.raw,
            )
        }
    }
}

/// Active batch metadata staged once per unit in `begin_batch` and consumed across layers.
#[derive(Clone, Debug)]
struct ActiveMember {
    sequence: usize,
    slot: usize,
    pos_start: usize,
    row_start: usize,
    row_len: usize,
    next_engram_tail: [u32; 2],
}

struct ActiveBatch {
    rows: usize,
    max_seq_end: usize,
    pt_stride: usize,
    swa_pt_stride: usize,
    all_single_row: bool,
    graphed: bool,
    ffn_fused_layer: Option<usize>,
    post_attn_deferred: bool,
    decode_attention: bool,
    dump_rows: usize,
    use_a: bool,
    members: Vec<ActiveMember>,
    engram_wave: Hy4EngramWave,
}

/// Coordinator Hy4-flash execution engine (1x RTX TP1 or 2x RTX TP2, 0-Spark or Spark-backed).
pub(crate) struct Hy4Engine<'a> {
    pub(crate) cfg: Hy4FlashConfig,
    pub(crate) max_rows: usize,
    pub(crate) decode_rows: usize,
    pub(crate) prefill_waves: usize,
    pub(crate) max_pages: usize,
    pub(crate) max_sequences: usize,
    pub(crate) max_context: usize,
    pub(crate) engram: Hy4EngramHost,
    pub(crate) skip_routed_experts: bool,
    pub(crate) dump_dir: Option<std::path::PathBuf>,
    max_seq_pages: usize,
    graphs_enabled: Cell<bool>,
    swa_pt_host: Vec<i32>,
    ranks: Vec<RankRuntime<'a>>,
    head_handoff: Option<Handoff<'a>>,
    spark: Option<SparkLink<'a>>,
    onboarded_layers: BTreeSet<usize>,
    next_identity: u64,
    placement_version: u64,
    dsa_index_of: BTreeMap<usize, usize>,
    swa_index_of: BTreeMap<usize, usize>,
    engram_index_of: BTreeMap<usize, usize>,
    golden_placement: Hy4Placement,
    terminal: bool,
}

impl<'a> Hy4Engine<'a> {
    /// Conservative upper bound on per-rank `RankScratch` + head/selector workspace bytes
    /// (excluding target weights, routed experts, and KV/SWA/mark caches) for KV pool sizing.
    pub(crate) fn estimated_rank_workspace_bytes(
        cfg: &Hy4FlashConfig,
        tp: usize,
        max_rows: usize,
        decode_rows: usize,
        max_seq_pages: usize,
        max_sequences: usize,
    ) -> u64 {
        let max_heads = 64 / tp;
        let max_inter = 13824 / tp;
        let max_seq_groups = max_seq_pages * (PAGE_ROWS / 4);
        let idx_score_floats = max_rows
            .saturating_mul(max_seq_groups.max(1))
            .min(MAX_IDX_SCORE_FLOATS)
            .max(max_seq_groups.max(1));
        let wire_row = LATENT_H + LATENT_H / 32;
        let tp2_f32 = if tp == 2 { 3 * max_rows * H * 4 } else { 0 };
        let split_k_cap = mla_split_scratch_bytes(decode_rows, tp);
        let engram_count = cfg.engram_layers.len();
        let sum = max_rows * (4 + 8 + 4 + 8 + 8)
            + 2 * max_rows * HC * H * 2
            + 64 * HC * H * 2
            + 2 * max_rows * HC * 4
            + 3 * max_rows * H * 2
            + tp2_f32
            + engram_count * (max_rows * 16 * 96 * 2 + max_rows)
            + max_rows * (1536 * 2 + H * 2 + HC * H * 2)
            + max_rows * (Q_LORA * 4 + KV_A_DIM * 2 + KV_A_PADDED * 2)
            + max_rows * max_heads * ((256 * 3 + 576 + 512) * 2 + 4)
            + split_k_cap
            + max_rows * max_seq_pages.max(16) * 4
            + max_sequences * max_seq_pages.max(16) * 4
            + max_rows * (IDX_HEADS * IDX_DIM * 4 + IDX_HEADS * 6 + IDX_DIM * 4)
            + (max_rows + 4 * max_sequences) * IDX_DIM * 2
            + idx_score_floats * 4
            + max_rows * (CANDIDATE_TOPK * 4 + 8)
            + 3 * max_rows * max_inter * 2
            + max_rows * (EXPERTS * 8 + TOPK * 8 + LATENT_H * 4 + wire_row + H * 8)
            + 64 * cfg.vocab_size * 8;
        // Include 256 MiB for optional native-core projection scratch + CUDA allocation tails.
        (sum as u64) + (256u64 << 20)
    }

    /// Single-sequence constructor used by `hy4f-golden`.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        devices: &[Device<'a>],
        programs: &'a Programs<'a>,
        checkpoint: &Checkpoint,
        cfg: Hy4FlashConfig,
        kv_format: Hy4KvCache,
        max_rows: usize,
        max_pages: usize,
        skip_routed_experts: bool,
        onboarded_experts: Vec<Option<Exl3Tp2<'a>>>,
    ) -> Result<Self> {
        let max_context = max_pages * PAGE_ROWS;
        Self::new_serving(
            devices,
            programs,
            checkpoint,
            cfg,
            kv_format,
            max_rows,
            max_rows.min(DECODE_ROWS as usize),
            1,
            max_pages,
            1,
            max_context,
            0,
            skip_routed_experts,
            onboarded_experts,
        )
    }

    /// Serving & multi-sequence constructor with exact prefix mark slots and COW page pool.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new_serving(
        devices: &[Device<'a>],
        programs: &'a Programs<'a>,
        checkpoint: &Checkpoint,
        cfg: Hy4FlashConfig,
        kv_format: Hy4KvCache,
        unit_rows: usize,
        decode_rows: usize,
        prefill_waves: usize,
        max_pages: usize,
        max_sequences: usize,
        max_context: usize,
        mark_slots: usize,
        skip_routed_experts: bool,
        mut onboarded_experts: Vec<Option<Exl3Tp2<'a>>>,
    ) -> Result<Self> {
        ensure!((1..=DECODE_ROWS as usize).contains(&decode_rows),
            "Hy4-flash split-K decode exports cover at most {DECODE_ROWS} rows; got {decode_rows}");
        let max_rows = unit_rows.max(decode_rows);
        let gpus = devices.len();
        ensure!(matches!(gpus, 1 | 2), "Hy4Engine supports 1 or 2 GPUs, got {gpus}");
        ensure!(onboarded_experts.len() == gpus, "onboarded_experts rank count mismatch");
        ensure!(max_sequences > 0 && max_pages > 0 && unit_rows > 0, "invalid Hy4Engine extents");
        if gpus == 2 {
            let probed = crate::shared::peer_split::probed_device(
                devices[0].library,
                devices[0].id,
                Some(devices[1].id),
            )?;
            ensure!(
                probed == Some(devices[1].id),
                "Hy4Engine TP2 requires bidirectional GPU-to-GPU peer access between devices {} and {}",
                devices[0].id,
                devices[1].id
            );
        }

        let prefill_waves = prefill_waves.max(1);
        // Prefill scoring can project a full head tile, independently of the
        // configured decode-row budget.
        let head_rows = max_rows.min(DECODE_ROWS as usize);
        let weights_all = Hy4RankWeights::load_all(devices, programs, checkpoint, &cfg, max_rows, head_rows)?;
        // SAFETY: checkpoint shard files remain immutable for the duration of the engine.
        let engram = unsafe { Hy4EngramHost::open(devices[0].library, checkpoint, &cfg, unit_rows, prefill_waves, decode_rows)? };
        let max_seq_pages = max_context.div_ceil(PAGE_ROWS).max(16).min(max_pages);

        let mut ranks = Vec::with_capacity(gpus);
        for (rank, weights) in weights_all.into_iter().enumerate() {
            let device = devices[rank];
            let stream = Rc::new(Stream::new(device)?);
            let peer_ready = Event::new(device)?;
            let progs = device.run(|| Hy4Programs::load(programs, gpus, kv_format))?;
            let sms = device.run(|| device.library.sm_count())?;
            let mla_splits = MlaSplitPlan::new(&progs, max_context, decode_rows, sms)?;
            // Split-capable launches never exceed the admitted decode/verify rows.
            let attn_scratch = mla_split_scratch_bytes(decode_rows, gpus);
            let cache = Rc::new(Hy4CacheRank::new(
                vec![stream.clone()],
                &cfg,
                kv_format,
                max_pages,
                max_sequences,
                max_rows,
                mark_slots,
            )?);
            let native_layout = weights.native.scratch_layout(max_rows)?;
            let native_head_layout = weights.native.head_scratch_layout(head_rows)?;
            let scratch = RankScratch::new(
                device,
                rank,
                gpus,
                max_rows,
                max_seq_pages,
                max_sequences,
                cfg.engram_layers.len(),
                cfg.vocab_size,
                weights.local_vocab,
                attn_scratch,
                native_layout.as_ref(),
                native_head_layout.as_ref(),
            )?;
            let bf16_add = device.run(|| device.library.bf16_add())?;
            let f32_add = device.run(|| device.library.f32_add())?;
            let quantize_grid = Fp8QuantizeGrid::new(sms, None)?;
            ranks.push(RankRuntime {
                graphs: Hy4Graphs::new(),
                device,
                stream,
                peer_ready,
                programs: progs,
                mla_splits,
                weights,
                cache,
                scratch,
                bf16_add,
                f32_add,
                quantize_grid,
                onboarded_experts: RefCell::new(onboarded_experts[rank].take()),
            });
        }

        let head_handoff = if gpus == 2 {
            Some(Handoff::new(ranks[0].stream.clone(), ranks[1].stream.clone())?)
        } else {
            None
        };

        let dsa_index_of: BTreeMap<usize, usize> = cfg
            .sparse_layers()
            .into_iter()
            .enumerate()
            .map(|(i, l)| (l, i))
            .collect();
        let swa_index_of: BTreeMap<usize, usize> = cfg
            .swa_layers()
            .into_iter()
            .enumerate()
            .map(|(i, l)| (l, i))
            .collect();
        let engram_index_of: BTreeMap<usize, usize> = cfg
            .engram_layers
            .iter()
            .copied()
            .enumerate()
            .map(|(i, l)| (l, i))
            .collect();
        let golden_placement = Hy4Placement::new((0..max_pages as u32).collect(), 0, cfg.engram_pad_id);

        let onboarded_layers: BTreeSet<usize> = cfg.routed_layers().into_iter().collect();
        let swa_pt_host = vec![0i32; max_sequences * max_seq_pages.max(16)];
        Ok(Self {
            cfg,
            max_rows,
            decode_rows,
            prefill_waves,
            max_pages,
            max_sequences,
            max_context,
            engram,
            skip_routed_experts,
            dump_dir: None,
            max_seq_pages: max_seq_pages.max(16),
            graphs_enabled: Cell::new(false),
            swa_pt_host,
            ranks,
            head_handoff,
            spark: None,
            onboarded_layers,
            next_identity: 1,
            placement_version: 1,
            dsa_index_of,
            swa_index_of,
            engram_index_of,
            golden_placement,
            terminal: false,
        })
    }

    /// Attach a bootstrapped Spark expert link for routed layers not onboarded onto the RTX GPUs.
    pub(crate) fn attach_spark(&mut self, link: SparkLink<'a>, onboarded_layers: impl IntoIterator<Item = usize>) {
        self.spark = Some(link);
        self.onboarded_layers = onboarded_layers.into_iter().collect();
    }

    /// Orderly terminal retirement: quiesce Spark link, drain device streams and caches, clear graphs, release link.
    pub(crate) fn terminate(&mut self) {
        if self.terminal {
            return;
        }
        self.terminal = true;
        if let Some(spark) = &mut self.spark {
            let _ = spark.terminal_quiesce();
        }
        let drained = self.drain();
        if drained.is_err() {
            self.ranks[0].device.library.quarantine_module_after_failed_drain();
        }
        crate::shared::decode_graph::fatal_drain(drained, "Hy4-flash terminal owners");
        for r in &self.ranks {
            r.graphs.clear_drained();
        }
        if let Some(spark) = &mut self.spark {
            let _ = spark.terminal_drain_copies();
            let _ = spark.terminal_release();
        }
    }

    pub(crate) fn golden_mla_splits(&mut self, dsa: bool, swa: bool) {
        for rank in &mut self.ranks {
            if !dsa { rank.mla_splits.dsa = 1; }
            if !swa { rank.mla_splits.swa = 1; }
            tracing::info!(device = rank.device.id, dsa = rank.mla_splits.dsa,
                swa = rank.mla_splits.swa, "Hy4 golden MLA splits");
        }
    }

    pub(crate) fn configure_graphs(
        &mut self,
        layout: Option<cuteafd_loader::serving_capacity::hy4_graphs::Hy4GraphLayout>,
    ) -> Result<()> {
        for (rank, r) in self.ranks.iter_mut().enumerate() {
            r.graphs.attach(r.device.library, r.device.id, rank, layout, self.decode_rows)?;
        }
        self.graphs_enabled.set(true);
        Ok(())
    }

    pub(crate) fn graphs_active(&self) -> bool {
        self.graphs_enabled.get() && self.ranks.iter().all(|r| r.graphs.attached())
    }

    pub(crate) fn retained_graph_rows(&self, rows: usize) -> Vec<(i32, usize)> {
        self.ranks
            .iter()
            .map(|r| (r.device.id, r.graphs.retained_rows(rows)))
            .collect()
    }

    pub(crate) fn warming_graphs<T>(&mut self, work: impl FnOnce(&mut Self) -> Result<T>) -> Result<T> {
        for r in &self.ranks {
            r.graphs.warming(true);
        }
        let result = work(self);
        for r in &self.ranks {
            r.graphs.warming(false);
        }
        result
    }

    pub(crate) fn without_graphs<T>(&mut self, work: impl FnOnce(&mut Self) -> Result<T>) -> Result<T> {
        let prev = self.graphs_enabled.replace(false);
        let result = work(self);
        self.graphs_enabled.set(prev);
        result
    }

    pub(crate) fn observe_graph_startup(&self) -> Result<()> {
        self.drain()?;
        for r in &self.ranks {
            graph_device::observe_device(r.device.library, r.device.id)?;
        }
        Ok(())
    }

    fn set_graph_cycle(&self, priority: ReadPriority) {
        let cycle = match priority {
            ReadPriority::Decode | ReadPriority::DecodeFirst => GraphCycle::Decode,
            ReadPriority::Prefill | ReadPriority::PrefillFirst => GraphCycle::Prefill,
        };
        for r in &self.ranks {
            r.graphs.cycle(cycle);
        }
    }

    fn graph_completed(&self) -> Result<()> {
        if !self.graphs_active() {
            return Ok(());
        }
        if !self.ranks.iter().any(|r| r.graphs.needs_boundary()) {
            for r in &self.ranks {
                r.graphs.tick();
            }
            return Ok(());
        }
        self.drain()?;
        // SAFETY: every rank stream is drained; graph owners hold no allocations and
        // engine weights/cache/scratch outlive all banks (cleared in terminate()).
        unsafe {
            graph_device::observe_completed(self.ranks[0].device.library, |visit| {
                for r in &self.ranks {
                    r.graphs.visit(visit)?;
                }
                Ok(())
            })?;
        }
        for r in &self.ranks {
            r.graphs.observed();
        }
        Ok(())
    }

    /// Warm one decode row geometry at `pos` on disposable startup cache slots/pages.
    /// Pass 0 runs eager first-touch + `observe_graph_startup`; pass 1 captures + observes.
    pub(crate) fn warm_row_shape(&mut self, rows: usize, pos: usize) -> Result<()> {
        ensure!(rows > 0 && rows <= self.decode_rows, "Hy4 warm rows {rows} out of range");
        let n_seq = rows.min(self.max_sequences);
        let base_per = rows / n_seq;
        let rem = rows % n_seq;
        let max_per = base_per + usize::from(rem > 0);
        let pages_needed = (pos + max_per).div_ceil(PAGE_ROWS).max(1);
        if pos + max_per > self.max_context || pages_needed > self.max_pages {
            return Ok(());
        }
        let shared_pages: Vec<u32> = (0..pages_needed as u32).collect();
        let pad_id = self.cfg.engram_pad_id;
        let layers = self.cfg.layers;
        let token_bufs: Vec<Vec<u32>> = (0..n_seq)
            .map(|i| vec![0u32; base_per + usize::from(i < rem)])
            .collect();
        let sources: Vec<&[u32]> = token_bufs.iter().map(Vec::as_slice).collect();
        let mut segments = Vec::with_capacity(n_seq);
        let mut cursor = 0usize;
        for (i, buf) in token_bufs.iter().enumerate() {
            segments.push(UnitSegment {
                sequence: i,
                rows: cursor..cursor + buf.len(),
                tokens: pos..pos + buf.len(),
                prompt_end: false,
                snapshot: false,
                head: true,
            });
            cursor += buf.len();
        }
        let unit = Unit {
            index: 0,
            sequence: 0,
            segments,
            rows: 0..rows,
            layers: 0..layers,
            slot: 0,
            barrier: false,
        };
        for pass in 0..2 {
            self.reset()?;
            let mut placements: Vec<Hy4Placement> = (0..n_seq)
                .map(|slot| {
                    let mut p = Hy4Placement::new(shared_pages.clone(), slot, pad_id);
                    p.len = pos;
                    p.state_len = pos;
                    p
                })
                .collect();
            self.run_units(
                &mut placements,
                &sources,
                std::slice::from_ref(&unit),
                ReadPriority::Decode,
                |_| HeadMode::Suffix(0),
                &mut |_, _, _| Ok(()),
            )?;
            if pass == 0 {
                self.observe_graph_startup()?;
            }
        }
        self.reset()
    }

    pub(crate) fn library(&self) -> &'a cuteafd_ffi::NativeLibrary {
        self.ranks[0].device.library
    }

    pub(crate) fn lead_device_id(&self) -> i32 {
        self.ranks[0].device.id
    }

    pub(crate) fn prefix(&self) -> Result<Hy4Prefix<'a>> {
        let caches: Vec<Rc<Hy4CacheRank<'a>>> = self.ranks.iter().map(|r| r.cache.clone()).collect();
        Hy4Prefix::new(caches, self.cfg.engram_pad_id)
    }

    pub(crate) fn dump_state(&self, placement: &Hy4Placement) -> Result<Vec<super::cache::Hy4RankStateSnapshot>> {
        self.ranks.iter().map(|r| r.cache.dump_state(placement)).collect()
    }

    pub(crate) fn mark_slots(&self) -> usize {
        self.ranks[0].cache.mark_slots
    }

    pub(crate) fn require_live(&self) -> Result<()> {
        ensure!(!self.terminal, "Hy4-flash execution is terminal");
        Ok(())
    }

    pub(crate) fn drain(&self) -> Result<()> {
        for r in &self.ranks {
            r.stream.drain()?;
            r.cache.drain()?;
        }
        self.engram.drain()
    }

    pub(crate) fn inventory(&self, inventory: &mut Inventory<'_>) -> Result<()> {
        for r in &self.ranks {
            r.cache.inventory(inventory)?;
        }
        Ok(())
    }

    fn dump_bf16(&self, name: &str, buf: CuteafdDeviceBuffer, elems: usize) -> Result<()> {
        let Some(dir) = &self.dump_dir else { return Ok(()); };
        let r0 = &self.ranks[0];
        r0.stream.drain()?;
        std::fs::create_dir_all(dir)?;
        let mut raw = vec![0u8; elems * 2];
        r0.device.run(|| {
            r0.device.library.copy_d2h(&mut raw, CuteafdDeviceBuffer { bytes: elems * 2, ..buf })
        })?;
        let f32_bytes: Vec<u8> = raw
            .chunks_exact(2)
            .flat_map(|b| (u32::from(u16::from_le_bytes([b[0], b[1]])) << 16).to_le_bytes())
            .collect();
        std::fs::write(dir.join(format!("{name}.f32")), f32_bytes)?;
        Ok(())
    }

    fn dump_f32(&self, name: &str, buf: CuteafdDeviceBuffer, elems: usize) -> Result<()> {
        let Some(dir) = &self.dump_dir else { return Ok(()); };
        let r0 = &self.ranks[0];
        r0.stream.drain()?;
        std::fs::create_dir_all(dir)?;
        let mut raw = vec![0u8; elems * 4];
        r0.device.run(|| {
            r0.device.library.copy_d2h(&mut raw, CuteafdDeviceBuffer { bytes: elems * 4, ..buf })
        })?;
        std::fs::write(dir.join(format!("{name}.f32")), raw)?;
        Ok(())
    }

    pub(crate) fn reset(&mut self) -> Result<()> {
        for r in &mut self.ranks {
            r.cache.zero_all()?;
        }
        self.golden_placement = Hy4Placement::new((0..self.max_pages as u32).collect(), 0, self.cfg.engram_pad_id);
        Ok(())
    }

    /// Device-driven GPU-to-GPU P2P sum of `scratch.delta_f32 [rows, width]` across TP2 ranks,
    /// followed by native `egress` (`svh` + BF16 narrow for EXL3, or BF16 narrow for FP8/BF16)
    /// into `scratch.delta [rows, width]` on both ranks with zero host hop.
    fn reduce_tp2_f32_to_bf16(&mut self, rows: usize, width: usize, egress_stem: Option<&str>) -> Result<()> {
        if self.ranks.len() == 1 {
            return Ok(());
        }
        let elems = rows * width;
        let bytes = elems * 4;
        for r in &mut self.ranks {
            r.peer_ready.record(&r.stream)?;
        }
        let ev0 = self.ranks[0].peer_ready.raw;
        let ev1 = self.ranks[1].peer_ready.raw;
        let src0 = CuteafdDeviceBuffer { bytes, ..self.ranks[0].scratch.delta_f32.buffer };
        let src1 = CuteafdDeviceBuffer { bytes, ..self.ranks[1].scratch.delta_f32.buffer };
        self.ranks[0].device.run(|| unsafe {
            self.ranks[0].device.library.cuda_stream_wait_event(self.ranks[0].stream.raw, ev1)?;
            self.ranks[0].device.library.copy_peer_async(
                CuteafdDeviceBuffer { bytes, ..self.ranks[0].scratch.delta_peer_f32.buffer },
                src1,
                bytes,
                self.ranks[0].stream.raw,
            )
        })?;
        self.ranks[1].device.run(|| unsafe {
            self.ranks[1].device.library.cuda_stream_wait_event(self.ranks[1].stream.raw, ev0)?;
            self.ranks[1].device.library.copy_peer_async(
                CuteafdDeviceBuffer { bytes, ..self.ranks[1].scratch.delta_peer_f32.buffer },
                src0,
                bytes,
                self.ranks[1].stream.raw,
            )
        })?;
        for r in &mut self.ranks {
            r.peer_ready.record(&r.stream)?;
        }
        let cp0 = self.ranks[0].peer_ready.raw;
        let cp1 = self.ranks[1].peer_ready.raw;
        let waits = [cp1, cp0];
        for (idx, r) in self.ranks.iter().enumerate() {
            let wait_ev = waits[idx];
            r.device.run(|| {
                // SAFETY: peer copy into delta_peer_f32 completed before wait_ev; all three buffers are disjoint.
                unsafe {
                    r.device.library.cuda_stream_wait_event(r.stream.raw, wait_ev)?;
                    r.f32_add.launch(
                        r.scratch.delta_f32.buffer,
                        r.scratch.delta_peer_f32.buffer,
                        r.scratch.delta_sum_f32.buffer,
                        elems,
                        r.stream.raw,
                    )?;
                }
                if let Some(linear) = egress_stem.and_then(|s| r.weights.native.get(s)) {
                    linear.egress(rows, r.scratch.delta_sum_f32.buffer, r.scratch.delta.buffer, r.stream.raw)
                } else {
                    // SAFETY: delta_sum_f32 sum finished in stream order before narrowing into disjoint delta.
                    unsafe {
                        r.device.library.cuda_f32_to_bf16_async(
                            r.scratch.delta_sum_f32.buffer,
                            r.scratch.delta.buffer,
                            elems,
                            r.stream.raw,
                        )
                    }
                }
            })?;
        }
        Ok(())
    }

    /// Stage metadata, Engram lookups, and token embeddings for a multi-sequence unit.
    fn begin_batch(
        &mut self,
        placements: &[Hy4Placement],
        initial_starts: &[usize],
        sources: &[&[u32]],
        unit: &Unit,
        priority: ReadPriority,
    ) -> Result<ActiveBatch> {
        let rows = unit.rows.len();
        ensure!(rows > 0 && rows <= self.max_rows, "Hy4 unit rows {rows} outside 1..={}", self.max_rows);
        ensure!(priority != ReadPriority::Decode || rows <= self.decode_rows,
            "Hy4 decode/verify uses {rows} rows, exceeds admitted {}-row split-K owner", self.decode_rows);
        ensure!(!unit.segments.is_empty(), "Hy4 unit must carry explicit segments");
        ensure!(unit.segments.len() <= self.max_sequences, "Hy4 batch exceeds max_sequences {}", self.max_sequences);

        let swa_ring_pages = self.ranks[0].cache.swa_ring_pages;
        let swa_ring_rows = self.ranks[0].cache.swa_ring_rows;

        let mut flat_tokens = Vec::with_capacity(rows);
        let mut pos_i64 = Vec::with_capacity(rows * 8);
        let mut pos_i32 = Vec::with_capacity(rows * 4);
        let mut dsa_slots_i64 = Vec::with_capacity(rows * 8);
        let mut swa_slots_i64 = Vec::with_capacity(rows * 8);
        let mut limits_bytes = Vec::with_capacity(rows * 4);
        let mut swa_seqlens_bytes = Vec::with_capacity(unit.segments.len() * 4);
        let mut swa_cu_bytes = Vec::with_capacity((unit.segments.len() + 1) * 4);
        let mut g_pos_bytes = Vec::new();
        let mut g_slots_bytes = Vec::new();
        let mut engram_members: Vec<([u32; 2], &[u32])> = Vec::with_capacity(unit.segments.len());

        let mut max_seq_end = 0usize;
        let mut members = Vec::with_capacity(unit.segments.len());
        let mut row_cursor = 0usize;

        for seg in &unit.segments {
            let p = placements
                .get(seg.sequence)
                .with_context(|| format!("Hy4 segment sequence {} out of range", seg.sequence))?;
            let src = sources
                .get(seg.sequence)
                .with_context(|| format!("Hy4 segment tokens {} out of range", seg.sequence))?;
            ensure!(
                seg.rows.start == row_cursor && seg.rows.len() == seg.tokens.len() && !seg.rows.is_empty(),
                "Hy4 segment rows/tokens mismatch"
            );
            let pos_start = p.len;
            ensure!(
                seg.tokens.start == pos_start,
                "Hy4 segment start {} differs from placement frontier {}",
                seg.tokens.start,
                pos_start
            );
            let base_start = *initial_starts
                .get(seg.sequence)
                .with_context(|| format!("Hy4 initial start {} out of range", seg.sequence))?;
            let slice_start = seg
                .tokens
                .start
                .checked_sub(base_start)
                .context("Hy4 segment start precedes submission origin")?;
            let tok_slice = src
                .get(slice_start..slice_start + seg.tokens.len())
                .context("Hy4 segment token slice out of bounds")?;
            let n = tok_slice.len();
            let pos_end = pos_start + n;
            ensure!(
                pos_end <= self.max_context && pos_end.div_ceil(PAGE_ROWS) <= p.units.len(),
                "Hy4 sequence extent {pos_end} exceeds context {} or reserved pages {}",
                self.max_context,
                p.units.len()
            );
            ensure!(p.slot < self.max_sequences, "Hy4 sequence slot {} out of range", p.slot);
            max_seq_end = max_seq_end.max(pos_end);
            swa_cu_bytes.extend_from_slice(&(row_cursor as i32).to_le_bytes());
            swa_seqlens_bytes.extend_from_slice(&(pos_end as i32).to_le_bytes());
            flat_tokens.extend_from_slice(tok_slice);

            for (offset, _) in tok_slice.iter().enumerate() {
                let pos = pos_start + offset;
                let logical_page = pos / PAGE_ROWS;
                let local_row = pos % PAGE_ROWS;
                let phys_page = p.units[logical_page] as usize;
                let dsa_slot = phys_page * PAGE_ROWS + local_row;
                let swa_slot = p.slot * swa_ring_rows + (pos % swa_ring_rows);
                pos_i64.extend_from_slice(&(pos as i64).to_le_bytes());
                pos_i32.extend_from_slice(&(pos as i32).to_le_bytes());
                dsa_slots_i64.extend_from_slice(&(dsa_slot as i64).to_le_bytes());
                swa_slots_i64.extend_from_slice(&(swa_slot as i64).to_le_bytes());
                limits_bytes.extend_from_slice(&((pos / 4) as i32).to_le_bytes());
            }

            let pending = pos_start % 4;
            let complete = (pending + n) / 4;
            if complete > 0 {
                let first_gid = (pos_start - pending) / 4;
                for g in 0..complete {
                    let gid = first_gid + g;
                    let logical_page = gid / (PAGE_ROWS / 4);
                    let local_group = gid % (PAGE_ROWS / 4);
                    let phys_page = p.units[logical_page] as usize;
                    let phys_group_slot = phys_page * (PAGE_ROWS / 4) + local_group;
                    g_pos_bytes.extend_from_slice(&((gid * 4) as i64).to_le_bytes());
                    g_slots_bytes.extend_from_slice(&(phys_group_slot as i64).to_le_bytes());
                }
            }

            let next_tail = advance_tail(p.engram_tail, tok_slice, tok_slice.len());
            engram_members.push((p.engram_tail, tok_slice));
            members.push(ActiveMember {
                sequence: seg.sequence,
                slot: p.slot,
                pos_start,
                row_start: row_cursor,
                row_len: n,
                next_engram_tail: next_tail,
            });
            row_cursor += n;
        }
        swa_cu_bytes.extend_from_slice(&(rows as i32).to_le_bytes());

        let graphed = priority == ReadPriority::Decode
            && self.graphs_active()
            && !self.skip_routed_experts
            && self.dump_dir.is_none();
        let all_single_row = members.len() == rows && rows <= 64;
        let pt_stride = max_seq_end.div_ceil(PAGE_ROWS).max(1);
        let swa_pt_stride = if graphed && all_single_row {
            self.max_seq_pages
        } else {
            pt_stride
        };
        let mut dsa_pt_bytes = Vec::with_capacity(rows * pt_stride * 4);
        let swa_pt_slice_bytes: Vec<u8> = if graphed && all_single_row {
            let need_ints = if members.len() == 1 {
                pt_stride
            } else {
                (members.len() - 1) * swa_pt_stride + pt_stride
            };
            for (b, m) in members.iter().enumerate() {
                let p = &placements[m.sequence];
                let seq_pages = (m.pos_start + m.row_len).div_ceil(PAGE_ROWS);
                let mut row_pt = Vec::with_capacity(pt_stride * 4);
                let swa_row = &mut self.swa_pt_host[b * swa_pt_stride..b * swa_pt_stride + pt_stride];
                for lp in 0..pt_stride {
                    let phys = if lp < seq_pages { p.units[lp] as i32 } else { 0i32 };
                    row_pt.extend_from_slice(&phys.to_le_bytes());
                    swa_row[lp] = (m.slot * swa_ring_pages + (lp % swa_ring_pages)) as i32;
                }
                for _ in 0..m.row_len {
                    dsa_pt_bytes.extend_from_slice(&row_pt);
                }
            }
            self.swa_pt_host[..need_ints]
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect()
        } else {
            let mut swa_pt_bytes = Vec::with_capacity(members.len() * pt_stride * 4);
            for m in &members {
                let p = &placements[m.sequence];
                let seq_pages = (m.pos_start + m.row_len).div_ceil(PAGE_ROWS);
                let mut row_pt = Vec::with_capacity(pt_stride * 4);
                for lp in 0..pt_stride {
                    let phys = if lp < seq_pages { p.units[lp] as i32 } else { 0i32 };
                    row_pt.extend_from_slice(&phys.to_le_bytes());
                    let swa_phys = (m.slot * swa_ring_pages + (lp % swa_ring_pages)) as i32;
                    swa_pt_bytes.extend_from_slice(&swa_phys.to_le_bytes());
                }
                for _ in 0..m.row_len {
                    dsa_pt_bytes.extend_from_slice(&row_pt);
                }
            }
            swa_pt_bytes
        };
        let cu_singleton: Vec<u8> = (0..=rows).flat_map(|i| (i as i32).to_le_bytes()).collect();

        // Issue background FP8 Engram gather + BF16 decode across all Engram layers in this unit
        // BEFORE launching embed/layer 0 so host I/O overlaps with GPU execution.
        let engram_wave = self.engram.begin_wave(&unit.layers, &engram_members, priority)?;

        // Embed tokens on rank 0 into r_a [rows, 4, H] BF16, replicate GPU-to-GPU to rank 1 if TP2.
        {
            let r0 = &mut self.ranks[0];
            let emb = r0.weights.embedding.as_ref().context("rank 0 missing TokenEmbedding")?;
            r0.device.run(|| {
                emb.embed(&flat_tokens, r0.scratch.ids.buffer, HC, r0.scratch.r_a.buffer, r0.stream.raw)
            })?;
            r0.peer_ready.record(&r0.stream)?;
        }
        if self.ranks.len() == 2 {
            let bytes = rows * HC * H * 2;
            let ev0 = self.ranks[0].peer_ready.raw;
            let src0 = CuteafdDeviceBuffer { bytes, ..self.ranks[0].scratch.r_a.buffer };
            self.ranks[1].device.run(|| unsafe {
                self.ranks[1].device.library.cuda_stream_wait_event(self.ranks[1].stream.raw, ev0)?;
                self.ranks[1].device.library.copy_peer_async(
                    CuteafdDeviceBuffer { bytes, ..self.ranks[1].scratch.r_a.buffer },
                    src0,
                    bytes,
                    self.ranks[1].stream.raw,
                )
            })?;
        }

        // Upload unit metadata ONCE on each rank's non-blocking stream; Engram layers land
        // from pinned staging right before each Engram layer runs.
        for r in &self.ranks {
            r.device.run(|| unsafe {
                let s = r.stream.raw;
                r.device.library.copy_h2d_ordered(r.scratch.positions_i64.buffer, &pos_i64, s)?;
                r.device.library.copy_h2d_ordered(r.scratch.positions_i32.buffer, &pos_i32, s)?;
                r.device.library.copy_h2d_ordered(r.scratch.dsa_slots_i64.buffer, &dsa_slots_i64, s)?;
                r.device.library.copy_h2d_ordered(r.scratch.swa_slots_i64.buffer, &swa_slots_i64, s)?;
                r.device.library.copy_h2d_ordered(r.scratch.dsa_page_table.buffer, &dsa_pt_bytes, s)?;
                r.device.library.copy_h2d_ordered(r.scratch.swa_page_table.buffer, &swa_pt_slice_bytes, s)?;
                r.device.library.copy_h2d_ordered(r.scratch.dsa_cu_seqlens.buffer, &cu_singleton, s)?;
                r.device.library.copy_h2d_ordered(r.scratch.swa_cu_seqlens.buffer, &swa_cu_bytes, s)?;
                r.device.library.copy_h2d_ordered(r.scratch.swa_seqlens.buffer, &swa_seqlens_bytes, s)?;
                r.device.library.copy_h2d_ordered(r.scratch.idx_limits.buffer, &limits_bytes, s)?;
                if !g_pos_bytes.is_empty() {
                    r.device.library.copy_h2d_ordered(r.scratch.idx_group_pos.buffer, &g_pos_bytes, s)?;
                    r.device.library.copy_h2d_ordered(r.scratch.idx_group_slots.buffer, &g_slots_bytes, s)?;
                }
                Ok(())
            })?;
        }

        let dump_rows = if self.dump_dir.is_some() && members.len() == 1 && members[0].pos_start == 0 {
            rows.min(16)
        } else {
            0
        };
        if dump_rows > 0 {
            let dir = self.dump_dir.as_ref().unwrap();
            std::fs::create_dir_all(dir)?;
            let tok_bytes: Vec<u8> = flat_tokens[..dump_rows].iter().flat_map(|t| t.to_le_bytes()).collect();
            std::fs::write(dir.join("tokens.u32"), tok_bytes)?;
            self.dump_bf16("embed", self.ranks[0].scratch.r_a.buffer, dump_rows * HC * H)?;
        }

        Ok(ActiveBatch {
            rows,
            max_seq_end,
            pt_stride,
            swa_pt_stride,
            all_single_row,
            graphed,
            ffn_fused_layer: None,
            post_attn_deferred: false,
            decode_attention: priority == ReadPriority::Decode,
            dump_rows,
            use_a: true,
            members,
            engram_wave,
        })
    }

    /// Run Engram (when present) + attention iHC pre + absorbed MLA + indexer/SWA + sink/gate/UV
    /// + o_proj + TP2 reduce + post_proj_norm + fused iHC post/pre for `layer`.
    /// On TP1 graphed local-FFN layers (`batch.graphed && gpus == 1 && local_ffn_route`),
    /// fuses the rank-local FFN + post-FFN iHC into the same section graph and marks `batch.ffn_fused_layer`.
    fn run_layer_attention(&mut self, layer: usize, batch: &mut ActiveBatch, local_ffn_route: bool) -> Result<()> {
        let p = format!("model.layers.{layer}");
        let is_sparse = self.cfg.sparse_layer[layer];
        let is_dense = self.cfg.dense_layer[layer];
        let attn = self.cfg.attention(layer);
        let gpus = self.ranks.len();
        let local_heads = attn.heads / gpus;
        let rows = batch.rows;
        let use_a = batch.use_a;
        let engram_slot = self.engram_index_of.get(&layer).copied();

        // (A) Engram host-to-device staging (always eager outside graph capture: variable lease ptr + TP2 P2P).
        if let Some(slot) = engram_slot {
            let (mut lease, raw_bytes, masked_bytes) = batch.engram_wave.take_layer(layer)?;
            {
                let r0 = &mut self.ranks[0];
                let result = r0.device.run(|| {
                    let pinned = lease.slot_mut().context("empty Hy4 Engram lease")?;
                    // SAFETY: pinned staging lease stays retained through copied_on; rank-0
                    // engram_raw[slot] and engram_masked[slot] are consumed in r0.stream order.
                    let copied = unsafe {
                        enqueue_table_h2d(
                            r0.device.library,
                            pinned.bytes.buffer(),
                            &[
                                CopyPlane {
                                    destination: r0.scratch.engram_raw[slot].buffer,
                                    offset: 0,
                                    bytes: raw_bytes,
                                },
                                CopyPlane {
                                    destination: r0.scratch.engram_masked[slot].buffer,
                                    offset: raw_bytes,
                                    bytes: masked_bytes,
                                },
                            ],
                            r0.stream.raw,
                        )
                    };
                    // SAFETY: r0.stream contains all reads from this pinned source.
                    let fenced = unsafe { pinned.bytes.copied_on(r0.stream.raw) };
                    batch.engram_wave.retain_copy(lease);
                    if copied.is_err() || fenced.is_err() {
                        if r0.stream.drain().is_err() {
                            r0.device.library.quarantine_module_after_failed_drain();
                        }
                    }
                    copied?;
                    fenced?;
                    Ok(())
                });
                result?;
                if gpus == 2 {
                    r0.peer_ready.record(&r0.stream)?;
                }
            }
            if gpus == 2 {
                let ev0 = self.ranks[0].peer_ready.raw;
                let src_raw = CuteafdDeviceBuffer { bytes: raw_bytes, ..self.ranks[0].scratch.engram_raw[slot].buffer };
                let src_mask = CuteafdDeviceBuffer { bytes: masked_bytes, ..self.ranks[0].scratch.engram_masked[slot].buffer };
                self.ranks[1].device.run(|| unsafe {
                    self.ranks[1].device.library.cuda_stream_wait_event(self.ranks[1].stream.raw, ev0)?;
                    self.ranks[1].device.library.copy_peer_async(
                        CuteafdDeviceBuffer { bytes: raw_bytes, ..self.ranks[1].scratch.engram_raw[slot].buffer },
                        src_raw,
                        raw_bytes,
                        self.ranks[1].stream.raw,
                    )?;
                    self.ranks[1].device.library.copy_peer_async(
                        CuteafdDeviceBuffer { bytes: masked_bytes, ..self.ranks[1].scratch.engram_masked[slot].buffer },
                        src_mask,
                        masked_bytes,
                        self.ranks[1].stream.raw,
                    )
                })?;
            }
        }

        // Eager indexer group-pool + score/topk helper shared by graphed & eager DSA paths.
        let run_dsa_indexer_eager = |r: &RankRuntime<'a>,
                                     dsa_idx: usize,
                                     max_groups: usize,
                                     pt_stride: usize,
                                     graphed: bool|
         -> Result<()> {
            let idx_cache = r.cache.dsa_index_layer(dsa_idx)?;
            let ip = format!("{p}.self_attn.compressed_indexer");
            let inv_freq = r.weights.inv_freq_dsa.buffer.ptr;
            // Assemble completed 4-token groups across all batch members into idx_group_tok.
            r.device.run(|| unsafe {
                let mut batch_groups = 0usize;
                for m in &batch.members {
                    let pending = r.cache.pending_counts.borrow()[m.slot][dsa_idx];
                    ensure!(
                        pending == m.pos_start % 4,
                        "Hy4 DSA layer {layer} slot {} pending count {pending} != {}",
                        m.slot,
                        m.pos_start % 4
                    );
                    let total_avail = pending + m.row_len;
                    let complete = total_avail / 4;
                    let new_pending = total_avail % 4;
                    let pending_buf = r.cache.pending_key_slot(m.slot, dsa_idx)?;
                    let tok_src_base = m.row_start * IDX_DIM * 2;
                    if complete > 0 {
                        let group_dst_off = batch_groups * 4 * IDX_DIM * 2;
                        if pending > 0 {
                            let dst_p = CuteafdDeviceBuffer {
                                ptr: r.scratch.idx_group_tok.buffer.ptr.cast::<u8>().wrapping_add(group_dst_off).cast(),
                                bytes: pending * IDX_DIM * 2,
                                ..r.scratch.idx_group_tok.buffer
                            };
                            r.device.library.copy_d2d_async(dst_p, pending_buf, pending * IDX_DIM * 2, r.stream.raw)?;
                        }
                        let copy_from_cur = complete * 4 - pending;
                        let dst_cur = CuteafdDeviceBuffer {
                            ptr: r.scratch.idx_group_tok.buffer.ptr.cast::<u8>().wrapping_add(group_dst_off + pending * IDX_DIM * 2).cast(),
                            bytes: copy_from_cur * IDX_DIM * 2,
                            ..r.scratch.idx_group_tok.buffer
                        };
                        let src_cur = CuteafdDeviceBuffer {
                            ptr: r.scratch.idx_tk.buffer.ptr.cast::<u8>().wrapping_add(tok_src_base).cast(),
                            bytes: copy_from_cur * IDX_DIM * 2,
                            ..r.scratch.idx_tk.buffer
                        };
                        r.device.library.copy_d2d_async(dst_cur, src_cur, copy_from_cur * IDX_DIM * 2, r.stream.raw)?;
                        batch_groups += complete;
                    }
                    if new_pending > 0 {
                        if complete == 0 {
                            let dst = CuteafdDeviceBuffer {
                                ptr: pending_buf.ptr.cast::<u8>().wrapping_add(pending * IDX_DIM * 2).cast(),
                                bytes: m.row_len * IDX_DIM * 2,
                                ..pending_buf
                            };
                            let src = CuteafdDeviceBuffer {
                                ptr: r.scratch.idx_tk.buffer.ptr.cast::<u8>().wrapping_add(tok_src_base).cast(),
                                bytes: m.row_len * IDX_DIM * 2,
                                ..r.scratch.idx_tk.buffer
                            };
                            r.device.library.copy_d2d_async(dst, src, m.row_len * IDX_DIM * 2, r.stream.raw)?;
                        } else {
                            let tail_off = tok_src_base + (complete * 4 - pending) * IDX_DIM * 2;
                            let src = CuteafdDeviceBuffer {
                                ptr: r.scratch.idx_tk.buffer.ptr.cast::<u8>().wrapping_add(tail_off).cast(),
                                bytes: new_pending * IDX_DIM * 2,
                                ..r.scratch.idx_tk.buffer
                            };
                            r.device.library.copy_d2d_async(pending_buf, src, new_pending * IDX_DIM * 2, r.stream.raw)?;
                        }
                    }
                    r.cache.pending_counts.borrow_mut()[m.slot][dsa_idx] = new_pending;
                }
                if batch_groups > 0 {
                    r.programs.indexer_pool_group.launch(
                        &[
                            r.scratch.idx_group_tok.buffer.ptr,
                            r.weights.weight(&format!("{ip}.key_value_projection"))?.ptr,
                            r.weights.weight(&format!("{ip}.key_gate_projection"))?.ptr,
                            r.weights.buf(&format!("{ip}.intra_group_ape"))?.ptr,
                            r.weights.buf(&format!("{ip}.key_norm.weight"))?.ptr,
                            r.scratch.idx_group_pos.buffer.ptr,
                            r.scratch.idx_group_slots.buffer.ptr,
                            inv_freq,
                            idx_cache.ptr,
                        ],
                        &[Scalar::I32(batch_groups as i32)],
                        r.stream.raw,
                    )?;
                }
                Ok(())
            })?;

            if batch.max_seq_end > CANDIDATE_TOPK && max_groups > 0 {
                if graphed {
                    r.graphs.run(
                        GraphKey { layer, rows, section: GraphSection::DsaIndexQ },
                        &r.stream,
                        || r.device.run(|| r.dsa_index_q(&p, rows)),
                    )?;
                } else {
                    r.device.run(|| r.dsa_index_q(&p, rows))?;
                }
                r.device.run(|| unsafe {
                    let tile_cap = (r.scratch.idx_score_floats / max_groups).max(1);
                    for row_start in (0..rows).step_by(tile_cap) {
                        let tile_rows = (rows - row_start).min(tile_cap);
                        let q_rot_ptr = r.scratch.idx_q_rot.buffer.ptr.cast::<u8>().wrapping_add(row_start * IDX_HEADS * IDX_DIM * 2).cast();
                        let w_out_ptr = r.scratch.idx_w_out.buffer.ptr.cast::<u8>().wrapping_add(row_start * IDX_HEADS * 4).cast();
                        let pt_ptr = r.scratch.dsa_page_table.buffer.ptr.cast::<u8>().wrapping_add(row_start * pt_stride * 4).cast();
                        let lim_ptr = r.scratch.idx_limits.buffer.ptr.cast::<u8>().wrapping_add(row_start * 4).cast();
                        let pos_ptr = r.scratch.positions_i32.buffer.ptr.cast::<u8>().wrapping_add(row_start * 4).cast();
                        let cand_ptr = r.scratch.idx_cand_slots.buffer.ptr.cast::<u8>().wrapping_add(row_start * CANDIDATE_TOPK * 4).cast();
                        let len_ptr = r.scratch.idx_cand_lens.buffer.ptr.cast::<u8>().wrapping_add(row_start * 4).cast();
                        r.programs.indexer_score_groups.launch(
                            &[
                                q_rot_ptr,
                                w_out_ptr,
                                idx_cache.ptr,
                                pt_ptr,
                                lim_ptr,
                                r.scratch.idx_scores.buffer.ptr,
                            ],
                            &[Scalar::I32(tile_rows as i32), Scalar::I32(max_groups as i32), Scalar::I32(pt_stride as i32)],
                            r.stream.raw,
                        )?;
                        r.programs.indexer_topk_expand.launch(
                            &[
                                r.scratch.idx_scores.buffer.ptr,
                                pos_ptr,
                                pt_ptr,
                                cand_ptr,
                                len_ptr,
                            ],
                            &[Scalar::I32(tile_rows as i32), Scalar::I32(max_groups as i32), Scalar::I32(pt_stride as i32)],
                            r.stream.raw,
                        )?;
                    }
                    Ok(())
                })
            } else {
                r.device.run(|| unsafe {
                    r.programs.indexer_topk_expand.launch(
                        &[
                            r.scratch.idx_scores.buffer.ptr,
                            r.scratch.positions_i32.buffer.ptr,
                            r.scratch.dsa_page_table.buffer.ptr,
                            r.scratch.idx_cand_slots.buffer.ptr,
                            r.scratch.idx_cand_lens.buffer.ptr,
                        ],
                        &[Scalar::I32(rows as i32), Scalar::I32(1), Scalar::I32(pt_stride as i32)],
                        r.stream.raw,
                    )
                })
            }
        };

        let max_groups = batch.max_seq_end / 4;
        let pt_stride = batch.pt_stride;
        let swa_pt_stride = batch.swa_pt_stride;
        let members_len = batch.members.len();
        let all_single_row = batch.all_single_row;
        let decode_attention = batch.decode_attention;
        let routed_scale = self.cfg.routed_scaling_factor as f32;
        let skip_routed = self.skip_routed_experts;

        // Graphed decode fast-path:
        //   Case 1 (TP1 + local FFN): fuse entire SWA layer (1 graph) or DSA pre + post (2 graphs).
        //   Case 2 (TP2 or Spark-routed): rank-local attention sections, defer post-attn iHC to FfnFrontend.
        if batch.graphed {
            if gpus == 1 && local_ffn_route {
                let r = &self.ranks[0];
                let kv_cache_buf = if is_sparse {
                    r.cache.dsa_kv_layer(self.dsa_index_of[&layer])?
                } else {
                    r.cache.swa_layer_bank(self.swa_index_of[&layer])?
                };
                if !is_sparse {
                    if all_single_row {
                        r.graphs.run(
                            GraphKey { layer, rows, section: GraphSection::SwaLocal },
                            &r.stream,
                            || {
                                r.device.run(|| {
                                    if let Some(slot) = engram_slot {
                                        r.engram_compute(&p, slot, rows, use_a)?;
                                    }
                                    r.attn_pre(&p, layer, false, local_heads, rows, use_a, kv_cache_buf)?;
                                    r.swa_attn(kv_cache_buf, rows, members_len, swa_pt_stride, decode_attention)?;
                                    r.attn_out(&p, layer, false, 1, local_heads, rows)?;
                                    r.post_attn_ihc(&p, rows, use_a)?;
                                    if is_dense {
                                        r.dense_ffn(&p, 1, rows)?;
                                    } else {
                                        r.moe_router_shared(&p, 1, rows, routed_scale)?;
                                        r.local_exl3(layer, 1, rows, skip_routed)?;
                                    }
                                    r.ffn_combine_post(&p, is_dense, rows, !use_a)
                                })
                            },
                        )?;
                    } else {
                        r.graphs.run(
                            GraphKey { layer, rows, section: GraphSection::SwaExtendPre },
                            &r.stream,
                            || {
                                r.device.run(|| {
                                    if let Some(slot) = engram_slot {
                                        r.engram_compute(&p, slot, rows, use_a)?;
                                    }
                                    r.attn_pre(&p, layer, false, local_heads, rows, use_a, kv_cache_buf)
                                })
                            },
                        )?;
                        r.device.run(|| r.swa_attn(kv_cache_buf, rows, members_len, swa_pt_stride, decode_attention))?;
                        r.graphs.run(
                            GraphKey { layer, rows, section: GraphSection::SwaExtendPost },
                            &r.stream,
                            || {
                                r.device.run(|| {
                                    r.attn_out(&p, layer, false, 1, local_heads, rows)?;
                                    r.post_attn_ihc(&p, rows, use_a)?;
                                    if is_dense {
                                        r.dense_ffn(&p, 1, rows)?;
                                    } else {
                                        r.moe_router_shared(&p, 1, rows, routed_scale)?;
                                        r.local_exl3(layer, 1, rows, skip_routed)?;
                                    }
                                    r.ffn_combine_post(&p, is_dense, rows, !use_a)
                                })
                            },
                        )?;
                    }
                } else {
                    let dsa_idx = self.dsa_index_of[&layer];
                    r.graphs.run(
                        GraphKey { layer, rows, section: GraphSection::DsaPre },
                        &r.stream,
                        || {
                            r.device.run(|| {
                                if let Some(slot) = engram_slot {
                                    r.engram_compute(&p, slot, rows, use_a)?;
                                }
                                r.attn_pre(&p, layer, true, local_heads, rows, use_a, kv_cache_buf)?;
                                r.dsa_index_wk(&p, rows)
                            })
                        },
                    )?;
                    run_dsa_indexer_eager(r, dsa_idx, max_groups, pt_stride, true)?;
                    r.graphs.run(
                        GraphKey { layer, rows, section: GraphSection::DsaLocalPost },
                        &r.stream,
                        || {
                            r.device.run(|| {
                                r.dsa_attn(kv_cache_buf, rows, decode_attention)?;
                                r.attn_out(&p, layer, true, 1, local_heads, rows)?;
                                r.post_attn_ihc(&p, rows, use_a)?;
                                if is_dense {
                                    r.dense_ffn(&p, 1, rows)?;
                                } else {
                                    r.moe_router_shared(&p, 1, rows, routed_scale)?;
                                    r.local_exl3(layer, 1, rows, skip_routed)?;
                                }
                                r.ffn_combine_post(&p, is_dense, rows, !use_a)
                            })
                        },
                    )?;
                }
                // Residual ping-pong flipped twice inside the fused layer; skip separate local_ffn.
                batch.ffn_fused_layer = Some(layer);
                return Ok(());
            }

            // Case 2: TP2 or Spark-routed layer — rank-local attention sections, peer join outside capture.
            for r in &self.ranks {
                let kv_cache_buf = if is_sparse {
                    r.cache.dsa_kv_layer(self.dsa_index_of[&layer])?
                } else {
                    r.cache.swa_layer_bank(self.swa_index_of[&layer])?
                };
                if !is_sparse {
                    if all_single_row {
                        r.graphs.run(
                            GraphKey { layer, rows, section: GraphSection::SwaAttn },
                            &r.stream,
                            || {
                                r.device.run(|| {
                                    if let Some(slot) = engram_slot {
                                        r.engram_compute(&p, slot, rows, use_a)?;
                                    }
                                    r.attn_pre(&p, layer, false, local_heads, rows, use_a, kv_cache_buf)?;
                                    r.swa_attn(kv_cache_buf, rows, members_len, swa_pt_stride, decode_attention)?;
                                    r.attn_out(&p, layer, false, gpus, local_heads, rows)
                                })
                            },
                        )?;
                    } else {
                        r.graphs.run(
                            GraphKey { layer, rows, section: GraphSection::SwaExtendPre },
                            &r.stream,
                            || {
                                r.device.run(|| {
                                    if let Some(slot) = engram_slot {
                                        r.engram_compute(&p, slot, rows, use_a)?;
                                    }
                                    r.attn_pre(&p, layer, false, local_heads, rows, use_a, kv_cache_buf)
                                })
                            },
                        )?;
                        r.device.run(|| r.swa_attn(kv_cache_buf, rows, members_len, swa_pt_stride, decode_attention))?;
                        r.graphs.run(
                            GraphKey { layer, rows, section: GraphSection::SwaExtendPost },
                            &r.stream,
                            || r.device.run(|| r.attn_out(&p, layer, false, gpus, local_heads, rows)),
                        )?;
                    }
                } else {
                    let dsa_idx = self.dsa_index_of[&layer];
                    r.graphs.run(
                        GraphKey { layer, rows, section: GraphSection::DsaPre },
                        &r.stream,
                        || {
                            r.device.run(|| {
                                if let Some(slot) = engram_slot {
                                    r.engram_compute(&p, slot, rows, use_a)?;
                                }
                                r.attn_pre(&p, layer, true, local_heads, rows, use_a, kv_cache_buf)?;
                                r.dsa_index_wk(&p, rows)
                            })
                        },
                    )?;
                    run_dsa_indexer_eager(r, dsa_idx, max_groups, pt_stride, true)?;
                    r.graphs.run(
                        GraphKey { layer, rows, section: GraphSection::DsaAttnOut },
                        &r.stream,
                        || {
                            r.device.run(|| {
                                r.dsa_attn(kv_cache_buf, rows, decode_attention)?;
                                r.attn_out(&p, layer, true, gpus, local_heads, rows)
                            })
                        },
                    )?;
                }
            }
            self.reduce_tp2_f32_to_bf16(rows, H, Some(&format!("{p}.self_attn.o_proj")))?;
            // Defer post-attention norm + iHC post/pre into FfnFrontend so both fold into one graph.
            batch.post_attn_deferred = true;
            return Ok(());
        }

        // Eager path (prefill, ungraphed decode/scoring, dump_dir, or skip_routed_experts).
        if let Some(slot) = engram_slot {
            for r in &self.ranks {
                r.device.run(|| r.engram_compute(&p, slot, rows, use_a))?;
            }
            if batch.dump_rows > 0 && layer <= 3 {
                let r_cur = if use_a { self.ranks[0].scratch.r_a.buffer } else { self.ranks[0].scratch.r_b.buffer };
                self.dump_bf16(&format!("l{layer:02}_after_engram"), r_cur, batch.dump_rows * HC * H)?;
            }
        }
        for r in &self.ranks {
            let kv_cache_buf = if is_sparse {
                r.cache.dsa_kv_layer(self.dsa_index_of[&layer])?
            } else {
                r.cache.swa_layer_bank(self.swa_index_of[&layer])?
            };
            r.device.run(|| r.attn_pre(&p, layer, is_sparse, local_heads, rows, use_a, kv_cache_buf))?;
            if is_sparse {
                let dsa_idx = self.dsa_index_of[&layer];
                r.device.run(|| r.dsa_index_wk(&p, rows))?;
                run_dsa_indexer_eager(r, dsa_idx, max_groups, pt_stride, false)?;
                r.device.run(|| r.dsa_attn(kv_cache_buf, rows, decode_attention))?;
            } else {
                r.device.run(|| r.swa_attn(kv_cache_buf, rows, members_len, swa_pt_stride, decode_attention))?;
            }
            r.device.run(|| r.attn_out(&p, layer, is_sparse, gpus, local_heads, rows))?;
        }
        self.reduce_tp2_f32_to_bf16(rows, H, Some(&format!("{p}.self_attn.o_proj")))?;
        if batch.dump_rows > 0 && layer <= 3 {
            let s0 = &self.ranks[0].scratch;
            let dump_rows = batch.dump_rows;
            self.dump_bf16(&format!("l{layer:02}_attn_y"), s0.y.buffer, dump_rows * H)?;
            self.dump_bf16(&format!("l{layer:02}_attn_q_resid"), s0.q_resid.buffer, dump_rows * Q_LORA)?;
            self.dump_bf16(&format!("l{layer:02}_attn_query"), s0.query.buffer, dump_rows * local_heads * 576)?;
            self.dump_bf16(&format!("l{layer:02}_attn_out"), s0.attn_out.buffer, dump_rows * local_heads * 512)?;
            self.dump_f32(&format!("l{layer:02}_attn_lse"), s0.lse_out.buffer, dump_rows * local_heads)?;
            self.dump_bf16(&format!("l{layer:02}_attn_uv"), s0.uv_out.buffer, dump_rows * local_heads * 256)?;
            self.dump_bf16(&format!("l{layer:02}_attn_delta"), s0.delta.buffer, dump_rows * H)?;
        }

        // (C) post_proj_norm + fused attention iHC post + FFN iHC pre
        for r in &self.ranks {
            r.device.run(|| r.post_attn_ihc(&p, rows, use_a))?;
        }
        batch.use_a = !batch.use_a;
        Ok(())
    }

    /// Run MoE router + latent down + FP8 K32 quant + shared expert on all ranks (used by both local & Spark FFN).
    fn run_moe_router_and_shared(&mut self, layer: usize, batch: &mut ActiveBatch) -> Result<()> {
        let p = format!("model.layers.{layer}");
        let gpus = self.ranks.len();
        let rows = batch.rows;
        let routed_scale = self.cfg.routed_scaling_factor as f32;
        let deferred = batch.post_attn_deferred;
        let use_a = batch.use_a;
        if batch.graphed {
            for r in &self.ranks {
                r.graphs.run(
                    GraphKey { layer, rows, section: GraphSection::FfnFrontend },
                    &r.stream,
                    || {
                        r.device.run(|| {
                            if deferred {
                                r.post_attn_ihc(&p, rows, use_a)?;
                            }
                            r.moe_router_shared(&p, gpus, rows, routed_scale)
                        })
                    },
                )?;
            }
            if deferred {
                batch.post_attn_deferred = false;
                batch.use_a = !batch.use_a;
            }
            return Ok(());
        }
        for r in &self.ranks {
            r.device.run(|| {
                if deferred {
                    r.post_attn_ihc(&p, rows, use_a)?;
                }
                r.moe_router_shared(&p, gpus, rows, routed_scale)
            })?;
        }
        if deferred {
            batch.post_attn_deferred = false;
            batch.use_a = !batch.use_a;
        }
        Ok(())
    }

    /// Finish post-FFN combine (`latent_up` + shared add + norm + iHC post) for dense or MoE layer.
    /// When `reduce_routed_tp2 == false` on TP2, rank 0's `routed_latent` (from Spark) is replicated to rank 1.
    fn finish_ffn_combine_and_post(&mut self, layer: usize, batch: &mut ActiveBatch, reduce_routed_tp2: bool) -> Result<()> {
        let p = format!("model.layers.{layer}");
        let gpus = self.ranks.len();
        let rows = batch.rows;
        let is_dense = self.cfg.dense_layer[layer];

        if is_dense {
            self.reduce_tp2_f32_to_bf16(rows, H, Some(&format!("{p}.mlp.down_proj")))?;
        } else if gpus == 2 {
            if reduce_routed_tp2 {
                self.reduce_tp2_f32_to_bf16(rows, LATENT_H, None)?;
                for r in &self.ranks {
                    r.device.run(|| unsafe {
                        r.device.library.copy_d2d_async(r.scratch.routed_latent.buffer, r.scratch.delta.buffer, rows * LATENT_H * 2, r.stream.raw)
                    })?;
                }
            } else {
                // Spark reply landed on rank 0's routed_latent [rows, 1536] BF16: replicate to rank 1 via P2P.
                let bytes = rows * LATENT_H * 2;
                let (ev0, src0) = {
                    let r0 = &mut self.ranks[0];
                    r0.peer_ready.record(&r0.stream)?;
                    (r0.peer_ready.raw, CuteafdDeviceBuffer { bytes, ..r0.scratch.routed_latent.buffer })
                };
                self.ranks[1].device.run(|| unsafe {
                    self.ranks[1].device.library.cuda_stream_wait_event(self.ranks[1].stream.raw, ev0)?;
                    self.ranks[1].device.library.copy_peer_async(
                        CuteafdDeviceBuffer { bytes, ..self.ranks[1].scratch.routed_latent.buffer },
                        src0,
                        bytes,
                        self.ranks[1].stream.raw,
                    )
                })?;
            }
            for r in &self.ranks {
                r.device.run(|| unsafe {
                    r.device.library.copy_d2d_async(r.scratch.delta_f32.buffer, r.scratch.shared_f32.buffer, rows * H * 4, r.stream.raw)
                })?;
            }
            self.reduce_tp2_f32_to_bf16(rows, H, Some(&format!("{p}.mlp.shared_experts.down_proj")))?;
            for r in &self.ranks {
                r.device.run(|| unsafe {
                    r.device.library.copy_d2d_async(r.scratch.shared_out.buffer, r.scratch.delta.buffer, rows * H * 2, r.stream.raw)
                })?;
            }
        }

        let use_a = batch.use_a;
        for r in &self.ranks {
            if batch.graphed {
                r.graphs.run(
                    GraphKey { layer, rows, section: GraphSection::FfnFinish },
                    &r.stream,
                    || r.device.run(|| r.ffn_combine_post(&p, is_dense, rows, use_a)),
                )?;
            } else {
                r.device.run(|| r.ffn_combine_post(&p, is_dense, rows, use_a))?;
            }
        }
        batch.use_a = !batch.use_a;

        if batch.dump_rows > 0 {
            let dump_rows = batch.dump_rows;
            let r_cur = if batch.use_a { self.ranks[0].scratch.r_a.buffer } else { self.ranks[0].scratch.r_b.buffer };
            self.dump_bf16(&format!("l{layer:02}_out"), r_cur, dump_rows * HC * H)?;
            if layer <= 3 {
                let s0 = &self.ranks[0].scratch;
                self.dump_bf16(&format!("l{layer:02}_ffn_y"), s0.y.buffer, dump_rows * H)?;
                self.dump_bf16(&format!("l{layer:02}_ffn_delta"), s0.delta.buffer, dump_rows * H)?;
                self.dump_bf16(&format!("l{layer:02}_ffn_delta_norm"), s0.delta_norm.buffer, dump_rows * H)?;
                if !is_dense {
                    self.dump_bf16(&format!("l{layer:02}_ffn_routed_latent"), s0.routed_latent.buffer, dump_rows * LATENT_H)?;
                    self.dump_bf16(&format!("l{layer:02}_ffn_shared_out"), s0.shared_out.buffer, dump_rows * H)?;
                }
            }
        }
        Ok(())
    }

    /// Run dense FFN (layer 0) or local onboarded LatentMoE (layers 1..=41) + post-FFN norm + iHC post.
    fn run_layer_local_ffn(&mut self, layer: usize, batch: &mut ActiveBatch) -> Result<()> {
        if batch.ffn_fused_layer == Some(layer) {
            batch.ffn_fused_layer = None;
            return Ok(());
        }
        let p = format!("model.layers.{layer}");
        let gpus = self.ranks.len();
        let rows = batch.rows;
        let deferred = batch.post_attn_deferred;
        let use_a = batch.use_a;
        let routed_scale = self.cfg.routed_scaling_factor as f32;
        let skip_routed = self.skip_routed_experts;

        if self.cfg.dense_layer[layer] {
            for r in &self.ranks {
                if batch.graphed {
                    r.graphs.run(
                        GraphKey { layer, rows, section: GraphSection::FfnFrontend },
                        &r.stream,
                        || {
                            r.device.run(|| {
                                if deferred {
                                    r.post_attn_ihc(&p, rows, use_a)?;
                                }
                                r.dense_ffn(&p, gpus, rows)
                            })
                        },
                    )?;
                } else {
                    r.device.run(|| {
                        if deferred {
                            r.post_attn_ihc(&p, rows, use_a)?;
                        }
                        r.dense_ffn(&p, gpus, rows)
                    })?;
                }
            }
            if deferred {
                batch.post_attn_deferred = false;
                batch.use_a = !batch.use_a;
            }
            return self.finish_ffn_combine_and_post(layer, batch, true);
        }

        if batch.graphed {
            for r in &self.ranks {
                r.graphs.run(
                    GraphKey { layer, rows, section: GraphSection::FfnFrontend },
                    &r.stream,
                    || {
                        r.device.run(|| {
                            if deferred {
                                r.post_attn_ihc(&p, rows, use_a)?;
                            }
                            r.moe_router_shared(&p, gpus, rows, routed_scale)?;
                            r.local_exl3(layer, gpus, rows, skip_routed)
                        })
                    },
                )?;
            }
            if deferred {
                batch.post_attn_deferred = false;
                batch.use_a = !batch.use_a;
            }
            return self.finish_ffn_combine_and_post(layer, batch, true);
        }

        self.run_moe_router_and_shared(layer, batch)?;
        for r in &self.ranks {
            r.device.run(|| r.local_exl3(layer, gpus, rows, skip_routed))?;
        }
        self.finish_ffn_combine_and_post(layer, batch, true)
    }

    /// Project a `<= 64`-row tile of `r0.scratch.y[y_row_off .. y_row_off + count]` through the
    /// vocabulary head into `r0.scratch.head_logits_f32 [count, vocab]` with zero host hop.
    fn project_head_tile(&self, y_row_off: usize, count: usize) -> Result<DeviceLogits> {
        ensure!(count > 0 && count <= 64, "Hy4 head tile count {count} outside 1..=64");
        let vocab = self.cfg.vocab_size;
        let r0 = &self.ranks[0];
        let head_out = r0
            .scratch
            .head_logits_f32
            .as_ref()
            .context("rank 0 missing head_logits_f32")?
            .buffer;
        let x0_buf = CuteafdDeviceBuffer {
            ptr: r0.scratch.y.buffer.ptr.cast::<u8>().wrapping_add(y_row_off * H * 2).cast(),
            bytes: count * H * 2,
            ..r0.scratch.y.buffer
        };
        if self.ranks.len() == 1 {
            let out_buf = CuteafdDeviceBuffer {
                bytes: count * vocab * 4,
                ..head_out
            };
            r0.device.run(|| launch_rank_lm_head(r0, count, x0_buf, out_buf))?;
        } else {
            let r1 = &self.ranks[1];
            let handoff = self.head_handoff.as_ref().context("TP2 head handoff missing")?;
            let local_v = r0.weights.local_vocab;
            let x1_buf = CuteafdDeviceBuffer {
                bytes: count * H * 2,
                ..r1.scratch.y.buffer
            };
            let out0_buf = CuteafdDeviceBuffer {
                bytes: count * local_v * 4,
                ..r0.scratch.logits_tile.buffer
            };
            let out1_buf = CuteafdDeviceBuffer {
                bytes: count * local_v * 4,
                ..r1.scratch.logits_tile.buffer
            };
            handoff.project(
                x0_buf,
                x1_buf,
                || launch_rank_lm_head(r0, count, x0_buf, out0_buf),
                || launch_rank_lm_head(r1, count, x1_buf, out1_buf),
                || {
                    let dst_pitch = vocab * 4;
                    let src_pitch = local_v * 4;
                    // SAFETY: rank 0 and rank 1 projections finished before gather; disjoint columns.
                    unsafe {
                        r0.device.library.copy_d2d_2d_async(
                            head_out,
                            dst_pitch,
                            out0_buf,
                            src_pitch,
                            src_pitch,
                            count,
                            r0.stream.raw,
                        )?;
                    }
                    let dst_right = CuteafdDeviceBuffer {
                        ptr: head_out.ptr.cast::<u8>().wrapping_add(src_pitch).cast(),
                        bytes: head_out.bytes - src_pitch,
                        ..head_out
                    };
                    handoff.copy_columns(dst_right, out1_buf, count, dst_pitch, src_pitch, src_pitch)
                },
            )?;
        }
        Ok(DeviceLogits {
            ptr: head_out.ptr,
            rows: count,
            vocab,
            stride: vocab,
            stream: r0.stream.raw,
            greedy: None,
        })
    }

    /// Run terminal `hc_head` + sharded FP32 `lm_head` -> zero-host-hop `DeviceLogits` tiles.
    fn finish_head(
        &self,
        batch: &ActiveBatch,
        mode: HeadMode,
        on_tile: &mut dyn FnMut(usize, DeviceLogits) -> Result<()>,
    ) -> Result<()> {
        let r0 = &self.ranks[0];
        let r_cur = if batch.use_a { r0.scratch.r_a.buffer } else { r0.scratch.r_b.buffer };
        match mode {
            HeadMode::None => Ok(()),
            HeadMode::LastPerSeq => {
                let count = batch.members.len();
                ensure!(count > 0 && count <= 64, "Hy4 LastPerSeq batch {count} exceeds 64-row head");
                let hc_src_ptr = if count == 1 {
                    let last = batch.rows - 1;
                    r_cur.ptr.cast::<u8>().wrapping_add(last * HC * H * 2).cast()
                } else {
                    let row_bytes = HC * H * 2;
                    r0.device.run(|| {
                        for (idx, m) in batch.members.iter().enumerate() {
                            let last = m.row_start + m.row_len - 1;
                            let src = CuteafdDeviceBuffer {
                                ptr: r_cur.ptr.cast::<u8>().wrapping_add(last * row_bytes).cast(),
                                bytes: row_bytes,
                                ..r_cur
                            };
                            let dst = CuteafdDeviceBuffer {
                                ptr: r0.scratch.r_head_in.buffer.ptr.cast::<u8>().wrapping_add(idx * row_bytes).cast(),
                                bytes: row_bytes,
                                ..r0.scratch.r_head_in.buffer
                            };
                            // SAFETY: r_cur and r_head_in are disjoint admitted buffers on rank 0.
                            unsafe { r0.device.library.copy_d2d_async(dst, src, row_bytes, r0.stream.raw)?; }
                        }
                        Ok(())
                    })?;
                    r0.scratch.r_head_in.buffer.ptr
                };
                r0.device.run(|| unsafe {
                    r0.programs.hc_head.launch(
                        &[
                            hc_src_ptr,
                            r0.weights.buf("model.hc_head.hc_head_fn")?.ptr,
                            r0.weights.buf("model.hc_head.hc_head_scale")?.ptr,
                            r0.weights.buf("model.hc_head.hc_head_base")?.ptr,
                            r0.weights.buf("model.norm.weight")?.ptr,
                            r0.scratch.delta_norm.buffer.ptr,
                            r0.scratch.y.buffer.ptr,
                        ],
                        &[Scalar::I32(count as i32)],
                        r0.stream.raw,
                    )
                })?;
                let logits = self.project_head_tile(0, count)?;
                on_tile(0, logits)
            }
            HeadMode::Suffix(first_row) => {
                ensure!(first_row <= batch.rows, "Hy4 head suffix {first_row} > rows {}", batch.rows);
                let total = batch.rows - first_row;
                if total == 0 {
                    return Ok(());
                }
                let hc_src_ptr = r_cur.ptr.cast::<u8>().wrapping_add(first_row * HC * H * 2).cast();
                r0.device.run(|| unsafe {
                    r0.programs.hc_head.launch(
                        &[
                            hc_src_ptr,
                            r0.weights.buf("model.hc_head.hc_head_fn")?.ptr,
                            r0.weights.buf("model.hc_head.hc_head_scale")?.ptr,
                            r0.weights.buf("model.hc_head.hc_head_base")?.ptr,
                            r0.weights.buf("model.norm.weight")?.ptr,
                            r0.scratch.delta_norm.buffer.ptr,
                            r0.scratch.y.buffer.ptr,
                        ],
                        &[Scalar::I32(total as i32)],
                        r0.stream.raw,
                    )
                })?;
                if batch.dump_rows > 0 && first_row == 0 {
                    self.dump_bf16("hc_head_y", r0.scratch.y.buffer, batch.dump_rows * H)?;
                }
                for tile_off in (0..total).step_by(64) {
                    let count = (total - tile_off).min(64);
                    let logits = self.project_head_tile(tile_off, count)?;
                    on_tile(first_row + tile_off, logits)?;
                }
                Ok(())
            }
        }
    }

    /// Execute one or more causal units through the shared `run_prefill_units` (`PrefillHooks`) driver.
    pub(crate) fn run_units(
        &mut self,
        placements: &mut [Hy4Placement],
        sources: &[&[u32]],
        units: &[Unit],
        priority: ReadPriority,
        head_mode: impl Fn(&Unit) -> HeadMode,
        on_tile: &mut dyn FnMut(&Unit, usize, DeviceLogits) -> Result<()>,
    ) -> Result<()> {
        self.require_live()?;
        if self.graphs_active() {
            self.set_graph_cycle(priority);
        }
        let initial_starts: Vec<usize> = placements.iter().map(|p| p.len).collect();
        let kind = match priority {
            ReadPriority::Decode => ExpertV2SourceKind::Decode,
            _ => ExpertV2SourceKind::Prefill,
        };
        let mut spark = self.spark.take();
        let result = (|| {
            let waves = spark.as_mut().map(|s| s.exchange()).transpose()?;
            struct Hooks<'e, 'a, 'p, 'w, F> {
                engine: &'e mut Hy4Engine<'a>,
                waves: Option<&'w mut Waves<'a>>,
                placements: &'p mut [Hy4Placement],
                initial_starts: &'p [usize],
                sources: &'p [&'p [u32]],
                priority: ReadPriority,
                kind: ExpertV2SourceKind,
                head_mode: F,
                on_tile: &'p mut dyn FnMut(&Unit, usize, DeviceLogits) -> Result<()>,
                active: Option<ActiveBatch>,
            }
            impl<F: Fn(&Unit) -> HeadMode> PrefillHooks for Hooks<'_, '_, '_, '_, F> {
                type Ticket = Option<WaveTicket>;
                fn route(&self, layer: usize) -> FfnRoute {
                    if self.engine.cfg.dense_layer[layer]
                        || self.waves.is_none()
                        || self.engine.onboarded_layers.contains(&layer)
                    {
                        FfnRoute::Local
                    } else {
                        FfnRoute::Spark
                    }
                }
                fn begin_unit(&mut self, unit: &Unit) -> Result<()> {
                    let batch = self.engine.begin_batch(self.placements, self.initial_starts, self.sources, unit, self.priority)?;
                    self.active = Some(batch);
                    Ok(())
                }
                fn attention(&mut self, layer: usize, _unit: &Unit) -> Result<()> {
                    let local_route = self.route(layer) == FfnRoute::Local;
                    let batch = self.active.as_mut().context("Hy4 attention without active unit")?;
                    self.engine.run_layer_attention(layer, batch, local_route)
                }
                fn local_ffn(&mut self, layer: usize, _unit: &Unit) -> Result<()> {
                    let batch = self.active.as_mut().context("Hy4 FFN without active unit")?;
                    self.engine.run_layer_local_ffn(layer, batch)
                }
                fn send_experts(&mut self, layer: usize, unit: &Unit) -> Result<Option<WaveTicket>> {
                    let batch = self.active.as_mut().context("Hy4 send_experts without active unit")?;
                    self.engine.run_moe_router_and_shared(layer, batch)?;
                    let waves = self.waves.as_mut().context("Hy4 Spark route without active SparkLink")?;
                    let rows = batch.rows;
                    let identity = self.engine.next_identity;
                    self.engine.next_identity = identity.checked_add(1).context("Hy4 Spark identity exhausted")?;
                    let version = self.engine.placement_version;
                    let dtype = ExpertV2Dtype::Fp8E4m3Ue8m0K32;
                    let staging = RouteStaging::new(rows, TOPK, dtype.row_bytes(LATENT_H)?)?;
                    let kind = self.kind;
                    let origins: Vec<(u64, u64)> = unit
                        .segments
                        .iter()
                        .flat_map(|seg| seg.tokens.clone().map(move |pos| (seg.sequence as u64 + 1, pos as u64)))
                        .collect();
                    let packed = unit.reply_origins()?;
                    let combined = if kind == ExpertV2SourceKind::Prefill {
                        packed
                            .as_ref()
                            .map_or_else(|| spark_waves::single_reply_eligible(rows, kind), |o| o.combined_reply_eligible())
                    } else {
                        spark_waves::single_reply_eligible(rows, kind)
                    };
                    let source = unit.sequence as u64 + 1;
                    let start = unit.rows.start;
                    let r0 = &self.engine.ranks[0];
                    let ticket = r0.device.run(|| {
                        // SAFETY: admitted rank-0 scratch owns route_ids/route_weights/wire_in on r0.stream.
                        unsafe {
                            waves.send(
                                rows,
                                rows * LATENT_H * 2,
                                staging,
                                [
                                    r0.scratch.route_ids.buffer,
                                    r0.scratch.route_weights.buffer,
                                    r0.scratch.wire_in.buffer,
                                ],
                                r0.stream.raw,
                                combined,
                                Box::new(move |routes, wire| {
                                    let mut req = build_hy4_spark_request(identity, version, layer, source, start, rows, dtype, routes, wire)?;
                                    if let Some(packed) = packed {
                                        packed.apply(&mut req)?;
                                    }
                                    ensure!(origins.len() == req.rows.len(), "Hy4 Spark origin count differs");
                                    for (row, &(src_id, pos)) in req.rows.iter_mut().zip(&origins) {
                                        row.source_request_id = src_id;
                                        row.token_position = pos;
                                        row.source_kind = kind;
                                    }
                                    Ok(req)
                                }),
                            )
                        }
                    })?;
                    Ok(Some(ticket))
                }
                fn land_experts(&mut self, layer: usize, _unit: &Unit, ticket: Option<WaveTicket>) -> Result<()> {
                    let ticket = ticket.context("Hy4 land_experts missing WaveTicket")?;
                    let slot = ticket.slot();
                    let batch = self.active.as_mut().context("Hy4 land_experts without active unit")?;
                    let rows = batch.rows;
                    let waves = self.waves.as_mut().context("Hy4 Spark land without active SparkLink")?;
                    let r0 = &self.engine.ranks[0];
                    r0.device.run(|| {
                        waves.land(ticket, r0.stream.raw)?;
                        // SAFETY: wave landed on r0.stream before reduction into rank-0 routed_latent.
                        unsafe {
                            waves.reduce(
                                slot,
                                std::ptr::null(),
                                r0.scratch.routed_latent.buffer.ptr.cast(),
                                rows,
                                r0.stream.raw,
                            )
                        }
                    })?;
                    self.engine.finish_ffn_combine_and_post(layer, batch, false)
                }
                fn end_unit(&mut self, unit: &Unit) -> Result<()> {
                    let batch = self.active.as_ref().context("Hy4 end_unit without active unit")?;
                    let mode = if unit.layers.end == self.engine.cfg.layers {
                        (self.head_mode)(unit)
                    } else {
                        HeadMode::None
                    };
                    self.engine.finish_head(batch, mode, &mut |row_off, logits| (self.on_tile)(unit, row_off, logits))
                }
                fn commit_unit(&mut self, _unit: &Unit) -> Result<()> {
                    let batch = self.active.take().context("Hy4 commit_unit without active unit")?;
                    for m in &batch.members {
                        let p = &mut self.placements[m.sequence];
                        p.len = m.pos_start + m.row_len;
                        p.state_len = p.len;
                        p.engram_tail = m.next_engram_tail;
                    }
                    Ok(())
                }
            }
            let mut hooks = Hooks {
                engine: self,
                waves,
                placements,
                initial_starts: &initial_starts,
                sources,
                priority,
                kind,
                head_mode,
                on_tile,
                active: None,
            };
            run_prefill_units(&mut hooks, units, 1)?;
            if spark.is_some() {
                spark_waves::healthy()?;
            }
            Ok(())
        })();
        self.spark = spark;
        // Failed captures can still grow driver storage; observe at the completed boundary
        // on both success and failure before terminal retirement.
        let completed = if priority == ReadPriority::Decode && self.graphs_active() {
            self.graph_completed()
        } else {
            Ok(())
        };
        let result = result.and(completed);
        if result.is_err() {
            self.terminate();
        }
        result
    }

    /// Run a single-token decode step (`ReadPriority::Decode`) through layers `0..=last_layer` (golden API).
    pub(crate) fn forward_decode_step(
        &mut self,
        token: u32,
        last_layer: usize,
        compute_head: bool,
    ) -> Result<Vec<f32>> {
        ensure!(last_layer < self.cfg.layers, "last_layer {last_layer} >= {}", self.cfg.layers);
        let pos_start = self.golden_placement.len;
        let unit = Unit {
            index: 0,
            sequence: 0,
            segments: vec![UnitSegment {
                sequence: 0,
                rows: 0..1,
                tokens: pos_start..pos_start + 1,
                prompt_end: true,
                snapshot: false,
                head: compute_head,
            }],
            rows: pos_start..pos_start + 1,
            layers: 0..last_layer + 1,
            slot: 0,
            barrier: false,
        };
        let vocab = self.cfg.vocab_size;
        let mut all_logits = if compute_head && last_layer + 1 == self.cfg.layers {
            vec![0f32; vocab]
        } else {
            Vec::new()
        };
        let mut placement = [self.golden_placement.clone()];
        let library = self.library();
        let mode = if compute_head { HeadMode::Suffix(0) } else { HeadMode::None };
        let tok = [token];
        self.run_units(
            &mut placement,
            &[&tok],
            &[unit],
            ReadPriority::Decode,
            |_| mode,
            &mut |_, _, logits| {
                let tile = logits.to_host(library)?;
                all_logits.copy_from_slice(&tile);
                Ok(())
            },
        )?;
        self.golden_placement = placement.into_iter().next().unwrap();
        if !compute_head || last_layer + 1 < self.cfg.layers {
            self.drain()?;
        }
        Ok(all_logits)
    }

    /// Run a contiguous single-sequence chunk through layers `0..=last_layer` (golden API).
    pub(crate) fn forward_chunk(&mut self, tokens: &[u32], last_layer: usize) -> Result<Vec<f32>> {
        self.forward_chunk_opts(tokens, last_layer, true)
    }

    /// Run a contiguous single-sequence chunk through layers `0..=last_layer`, optionally skipping
    /// the terminal vocabulary head when `compute_head == false` (prefix prefill before `N=1` decode).
    pub(crate) fn forward_chunk_opts(
        &mut self,
        tokens: &[u32],
        last_layer: usize,
        compute_head: bool,
    ) -> Result<Vec<f32>> {
        let rows = tokens.len();
        ensure!(rows > 0 && rows <= self.max_rows, "Hy4 chunk rows {rows} outside 1..={}", self.max_rows);
        ensure!(last_layer < self.cfg.layers, "last_layer {last_layer} >= {}", self.cfg.layers);
        let pos_start = self.golden_placement.len;
        let unit = Unit {
            index: 0,
            sequence: 0,
            segments: vec![UnitSegment {
                sequence: 0,
                rows: 0..rows,
                tokens: pos_start..pos_start + rows,
                prompt_end: true,
                snapshot: false,
                head: compute_head,
            }],
            rows: pos_start..pos_start + rows,
            layers: 0..last_layer + 1,
            slot: 0,
            barrier: false,
        };
        let vocab = self.cfg.vocab_size;
        let mut all_logits = if compute_head && last_layer + 1 == self.cfg.layers {
            vec![0f32; rows * vocab]
        } else {
            Vec::new()
        };
        let mut placement = [self.golden_placement.clone()];
        let library = self.library();
        let mode = if compute_head { HeadMode::Suffix(0) } else { HeadMode::None };
        self.run_units(
            &mut placement,
            &[tokens],
            &[unit],
            ReadPriority::Prefill,
            |_| mode,
            &mut |_, row_off, logits| {
                let tile = logits.to_host(library)?;
                let dst = row_off * vocab;
                all_logits[dst..dst + tile.len()].copy_from_slice(&tile);
                Ok(())
            },
        )?;
        self.golden_placement = placement.into_iter().next().unwrap();
        if !compute_head || last_layer + 1 < self.cfg.layers {
            self.drain()?;
        }
        Ok(all_logits)
    }

    /// Return the active 4-stream residual `[rows, 4, H]` F32 from rank 0 (for layer-0 boundary checks).
    pub(crate) fn read_residual_f32(&self, rows: usize) -> Result<Vec<f32>> {
        let r0 = &self.ranks[0];
        r0.stream.drain()?;
        let bytes = rows * HC * H * 2;
        let mut raw = vec![0u8; bytes];
        r0.device.run(|| {
            r0.device.library.copy_d2h(&mut raw, CuteafdDeviceBuffer { bytes, ..r0.scratch.r_a.buffer })
        })?;
        Ok(raw
            .chunks_exact(2)
            .map(|b| f32::from_bits(u32::from(u16::from_le_bytes([b[0], b[1]])) << 16))
            .collect())
    }
}

impl Drop for Hy4Engine<'_> {
    fn drop(&mut self) {
        self.terminate();
    }
}

#[allow(clippy::too_many_arguments)]
fn build_hy4_spark_request(
    identity: u64,
    version: u64,
    layer: usize,
    source: u64,
    start: usize,
    rows: usize,
    dtype: ExpertV2Dtype,
    routes: Vec<ExpertProtocolV2RouteEntry>,
    wire: bytes::Bytes,
) -> Result<ExpertProtocolV2Request> {
    ensure!(
        identity > 0 && version > 0 && source > 0 && (1..42).contains(&layer) && (1..=4096).contains(&rows),
        "Hy4 Spark request geometry/identity differs (layer={layer}, rows={rows})"
    );
    let descriptors = (0..rows)
        .map(|row| ExpertProtocolV2RowDescriptor {
            row_id: (start + row) as u64,
            source_kind: ExpertV2SourceKind::Prefill,
            source_request_id: source,
            token_position: (start + row) as u64,
            route_offset: (row * TOPK) as u32,
            route_count: TOPK as u32,
        })
        .collect();
    let mut request = ExpertProtocolV2Request::new_bytes(
        identity,
        version,
        u32::try_from(layer)?,
        LATENT_H as u32,
        dtype,
        descriptors,
        routes,
        wire,
    )?;
    request.header.flags |= EXPERT_PROTOCOL_V2_FLAG_V41_COMPACT_BF16;
    Ok(request)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hy4_spark_request_uses_latent_1536_k16_fp8k32_compact_bf16() -> Result<()> {
        let routes: Vec<ExpertProtocolV2RouteEntry> = (0..32)
            .map(|i| ExpertProtocolV2RouteEntry {
                row_index: i / 16,
                expert_id: i,
                gate_weight: 0.0625,
            })
            .collect();
        let req = build_hy4_spark_request(
            7,
            1,
            15,
            3,
            100,
            2,
            ExpertV2Dtype::Fp8E4m3Ue8m0K32,
            routes,
            vec![0u8; 2 * (1536 + 1536 / 32)].into(),
        )?;
        assert_eq!(req.header.hidden_dim, 1536);
        assert_eq!(req.header.hidden_dtype, ExpertV2Dtype::Fp8E4m3Ue8m0K32);
        assert_eq!(req.header.hidden_row_stride_bytes, 1584);
        assert_eq!(req.rows[0].route_count, 16);
        assert_eq!(req.rows[1].route_offset, 16);
        assert_ne!(req.header.flags & EXPERT_PROTOCOL_V2_FLAG_V41_COMPACT_BF16, 0);
        assert!(build_hy4_spark_request(7, 1, 0, 3, 0, 1, ExpertV2Dtype::Fp8E4m3Ue8m0K32, vec![], vec![].into()).is_err());
        Ok(())
    }
}
