//! Full 42-layer Hy4-flash coordinator forward pass (1x RTX TP1 and 2x RTX TP2)
//! supporting golden single-sequence verification and batched multi-sequence serving
//! through the shared `run_prefill_units` (`PrefillHooks`) executor and zero-host-hop
//! `DeviceLogits` vocabulary head handoff.
use super::cache::{slice_buf, Hy4CacheRank, Hy4Placement, Hy4Prefix};
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
use crate::shared::memory::{HostAllocation, Tag};
use crate::shared::memory::weight_arena::DeviceArena;
use crate::shared::peer_split::{
    context::{self, ContextExchange},
    PeerExchange, RankDevice, DIRECT,
};
use crate::shared::prefill_pipeline::{run_prefill_units, FfnRoute, PrefillHooks, Unit, UnitSegment};
use crate::shared::spark_intake::SparkLink;
use crate::shared::spark_waves::{self, WaveTicket, Waves};
use crate::shared::target_head::{self, TargetHead};
use crate::shared::token_io::DeviceLogits;
use crate::shared::token_table::{enqueue as enqueue_table_h2d, CopyPlane};
use anyhow::{ensure, Context, Result};
use cuteafd_core::memory_layout::{arena_class, ArenaTag};
use cuteafd_ffi::expert_usage::{ExpertUsage, UsageDevice, UsagePhase, UsageStorage};
use cuteafd_ffi::peer_exchange::PeerPlane;
use cuteafd_ffi::programs::{Program, Programs, Scalar};
use cuteafd_ffi::{Bf16Add, Bf16F32Add, CuteafdDeviceBuffer, CuteafdHostBuffer, F32Add};
use cuteafd_loader::families::hy4_flash::engram::advance_tail;
use cuteafd_loader::families::hy4_flash::execution::{
    context_chunk_rows, context_exchange_rows, hy4_context_buffers, hy4_dsa_partial_scratch_bytes, hy4_peer_ffn_routed_offset,
    hy4_peer_slot_sizes, mla_split_scratch_bytes, wave_metadata_layout_for_rank,
    SharedScratchLayout, WaveScratchLayout, PAGE_ROWS, WAVE_METADATA,
};
use cuteafd_loader::placement::AttentionPlacement;
use cuteafd_loader::placement::families::hy4_flash::DECODE_ROWS;
use cuteafd_loader::families::hy4_flash::{Hy4FlashConfig, Hy4KvCache};
use cuteafd_loader::formats::mapped_table::ReadPriority;
use cuteafd_loader::formats::trellis_core::ProjectionScratchLayout;
use cuteafd_loader::plan::checkpoint::Checkpoint;
use cuteafd_loader::serving_capacity::WeightArenaLayout;
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
const IDX_KEY_BYTES: usize = IDX_DIM * 2;
const CANDIDATE_TOPK: usize = 2048;
const GROUP_TOPK: usize = 512;
const PEER_SLOT_ATTN: usize = 0;
const PEER_SLOT_FFN: usize = 1;

/// Number of complete 4-token compressed-indexer groups in `[0, pos / 4)` owned by parity `rank`.
fn local_complete_groups(pos: usize, rank: usize) -> usize {
    let total_groups = pos / 4;
    let full_pairs = total_groups / 32;
    let rem = total_groups % 32;
    if rank == 0 {
        full_pairs * 16 + rem.min(16)
    } else {
        full_pairs * 16 + rem.saturating_sub(16)
    }
}

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

fn routed_f32_input(
    rank: &RankRuntime<'_>,
    rows: usize,
    skip_routed: bool,
) -> Result<CuteafdDeviceBuffer> {
    if skip_routed {
        Ok(rank.shared.delta_f32.buffer)
    } else {
        let experts = rank.onboarded_experts.borrow();
        Ok(CuteafdDeviceBuffer {
            ptr: experts.as_ref().context("Hy4 TP2 routed expert output missing")?.output(),
            bytes: rows * LATENT_H * 4,
            device_id: rank.device.id,
            flags: 0,
        })
    }
}

/// One packed per-wave scratch arena per coordinator GPU rank holding every buffer
/// that remains live across `PrefillHooks` boundaries for that wave slot.
struct WaveScratch<'a> {
    _arena: DeviceArena<'a, arena_class::ReadWrite>,
    positions_i64: Allocation<'a>,
    positions_i32: Allocation<'a>,
    dsa_slots_i64: Allocation<'a>,
    swa_slots_i64: Allocation<'a>,
    r_a: Allocation<'a>,
    r_b: Allocation<'a>,
    post_ffn: Allocation<'a>,
    y: Allocation<'a>,
    shared_out: Allocation<'a>,
    shared_f32: Allocation<'a>,
    dsa_page_table: Allocation<'a>,
    swa_page_table: Allocation<'a>,
    swa_seqlens: Allocation<'a>,
    dsa_cu_seqlens: Allocation<'a>,
    swa_cu_seqlens: Allocation<'a>,
    idx_limits: Allocation<'a>,
    idx_group_pos: Allocation<'a>,
    idx_group_slots: Allocation<'a>,
    engram_raw: Vec<Allocation<'a>>,
    engram_masked: Vec<Allocation<'a>>,
}

impl<'a> WaveScratch<'a> {
    #[allow(clippy::too_many_arguments)]
    fn new(
        device: Device<'a>,
        cfg: &Hy4FlashConfig,
        rank: usize,
        slot: usize,
        tp: usize,
        max_rows: usize,
        max_seq_pages: usize,
        max_sequences: usize,
    ) -> Result<Self> {
        let layout = WaveScratchLayout::new(cfg, rank, slot, tp, max_rows, max_seq_pages, max_sequences)?;
        let arena = DeviceArena::new(
            device,
            layout.arena.clone(),
            ArenaTag::workspace("hy4_flash/wave-scratch"),
        )?;
        let view = |name: &str| -> Result<Allocation<'a>> {
            let bytes = arena
                .layout()
                .view(name)
                .with_context(|| format!("Hy4 wave scratch missing view {name}"))?
                .bytes();
            arena.view(name, bytes)
        };
        let mut engram_raw = Vec::with_capacity(cfg.engram_layers.len());
        let mut engram_masked = Vec::with_capacity(cfg.engram_layers.len());
        for i in 0..cfg.engram_layers.len() {
            engram_raw.push(view(&format!("engram_raw.{i}"))?);
            engram_masked.push(view(&format!("engram_masked.{i}"))?);
        }
        Ok(Self {
            positions_i64: view("positions_i64")?,
            positions_i32: view("positions_i32")?,
            dsa_slots_i64: view("dsa_slots_i64")?,
            swa_slots_i64: view("swa_slots_i64")?,
            r_a: view("r_a")?,
            r_b: view("r_b")?,
            post_ffn: view("post_ffn")?,
            y: view("y")?,
            shared_out: view("shared_out")?,
            shared_f32: view("shared_f32")?,
            dsa_page_table: view("dsa_page_table")?,
            swa_page_table: view("swa_page_table")?,
            swa_seqlens: view("swa_seqlens")?,
            dsa_cu_seqlens: view("dsa_cu_seqlens")?,
            swa_cu_seqlens: view("swa_cu_seqlens")?,
            idx_limits: view("idx_limits")?,
            idx_group_pos: view("idx_group_pos")?,
            idx_group_slots: view("idx_group_slots")?,
            engram_raw,
            engram_masked,
            _arena: arena,
        })
    }

    fn inventory(&self, inventory: &mut Inventory<'_>, rank: usize, slot: usize) -> Result<()> {
        let prefix = format!("hy4_flash.wave.{rank}.{slot}");
        for (name, alloc) in [
            ("positions_i64", &self.positions_i64),
            ("positions_i32", &self.positions_i32),
            ("dsa_slots_i64", &self.dsa_slots_i64),
            ("swa_slots_i64", &self.swa_slots_i64),
            ("r_a", &self.r_a),
            ("r_b", &self.r_b),
            ("post_ffn", &self.post_ffn),
            ("y", &self.y),
            ("shared_out", &self.shared_out),
            ("shared_f32", &self.shared_f32),
            ("dsa_page_table", &self.dsa_page_table),
            ("swa_page_table", &self.swa_page_table),
            ("swa_seqlens", &self.swa_seqlens),
            ("dsa_cu_seqlens", &self.dsa_cu_seqlens),
            ("swa_cu_seqlens", &self.swa_cu_seqlens),
            ("idx_limits", &self.idx_limits),
            ("idx_group_pos", &self.idx_group_pos),
            ("idx_group_slots", &self.idx_group_slots),
        ] {
            alloc.inventory(inventory, &format!("{prefix}.{name}"))?;
        }
        for (i, alloc) in self.engram_raw.iter().enumerate() {
            alloc.inventory(inventory, &format!("{prefix}.engram_raw.{i}"))?;
        }
        for (i, alloc) in self.engram_masked.iter().enumerate() {
            alloc.inventory(inventory, &format!("{prefix}.engram_masked.{i}"))?;
        }
        Ok(())
    }
}

struct MetadataUpload<'a> {
    backing: HostAllocation<'a>,
    layout: WeightArenaLayout,
    completion: Vec<Event<'a>>,
    pending: bool,
}

impl<'a> MetadataUpload<'a> {
    fn new(devices: &[Device<'a>], layout: WeightArenaLayout) -> Result<Self> {
        let completion = devices.iter().copied().map(Event::new).collect::<Result<_>>()?;
        let backing = HostAllocation::new(devices[0].library, layout.bytes(), Tag::staging("hy4_flash/wave-metadata"))?;
        Ok(Self { backing, layout, completion, pending: false })
    }

    fn upload(&mut self, ranks: &[RankRuntime<'a>], slot: usize, planes: [&[u8]; WAVE_METADATA.len()]) -> Result<()> {
        ensure!(!self.pending, "Hy4 wave metadata still in flight");
        ensure!(ranks.len() == self.completion.len(), "Hy4 metadata rank count changed");
        let mut sources = [CuteafdHostBuffer::default(); WAVE_METADATA.len()];
        let mut lengths = [0; WAVE_METADATA.len()];
        let source = self.backing.buffer;
        let bytes = self.backing.bytes_mut();
        let mut count = 0;
        for (name, contents) in WAVE_METADATA.into_iter().zip(planes) {
            if contents.is_empty() {
                continue;
            }
            let view = self.layout.view(name).with_context(|| format!("Hy4 missing metadata view {name}"))?;
            ensure!(contents.len() <= view.bytes(), "Hy4 metadata {name} exceeds its admitted view");
            let at = view.offset();
            bytes[at..at + contents.len()].copy_from_slice(contents);
            sources[count] = CuteafdHostBuffer { ptr: source.ptr.wrapping_byte_add(at), bytes: contents.len(), ..source };
            lengths[count] = contents.len();
            count += 1;
        }
        self.pending = true;
        for (r, event) in ranks.iter().zip(&mut self.completion) {
            let wave = &r.waves[slot];
            let mut dst = [CuteafdDeviceBuffer::default(); WAVE_METADATA.len()];
            let bufs = [wave.positions_i64.buffer, wave.positions_i32.buffer, wave.dsa_slots_i64.buffer,
                wave.swa_slots_i64.buffer, wave.dsa_page_table.buffer, wave.swa_page_table.buffer,
                wave.dsa_cu_seqlens.buffer, wave.swa_cu_seqlens.buffer, wave.swa_seqlens.buffer,
                wave.idx_limits.buffer, wave.idx_group_pos.buffer, wave.idx_group_slots.buffer];
            for (target, buffer) in dst.iter_mut().zip(bufs.into_iter().zip(planes).filter_map(|(buffer, plane)| {
                (!plane.is_empty()).then_some(buffer)
            })) {
                *target = buffer;
            }
            r.device.run(|| {
                // SAFETY: the pinned source is private to this slot and stays immutable
                // until both events complete (or terminal retirement drains the streams).
                unsafe { r.device.library.copy_host_buffers_h2d_batch_async(
                    &dst[..count], &sources[..count], &lengths[..count], r.stream.raw,
                ) }
            })?;
            event.record(&r.stream)?;
        }
        Ok(())
    }

    fn retire(&mut self) -> Result<()> {
        if self.pending {
            for event in &self.completion {
                // SAFETY: these are this slot's recorded source-completion events.
                event.device.run(|| unsafe { event.device.library.cuda_event_synchronize(event.raw) })?;
            }
            self.pending = false;
        }
        Ok(())
    }
}

/// Optional per-rank ContextSplit DSA scratch views (allocated inside `SharedScratch`).
struct ContextScratch<'a> {
    cand_scores: Allocation<'a>,
    cand_indices: Allocation<'a>,
    partial_own: Allocation<'a>,
    lse_own: Allocation<'a>,
    partial_peer: Allocation<'a>,
    lse_peer: Allocation<'a>,
    draft_slots: Allocation<'a>,
}

/// One packed hook-local scratch arena per coordinator GPU rank, shared across all wave slots.
struct SharedScratch<'a> {
    _arena: DeviceArena<'a, arena_class::ReadWrite>,
    ids: Allocation<'a>,
    r_head_in: Allocation<'a>,
    post: Allocation<'a>,
    delta: Allocation<'a>,
    delta_f32: Allocation<'a>,
    delta_sum_f32: Allocation<'a>,
    delta_norm: Allocation<'a>,
    engram_normed: Allocation<'a>,
    engram_val: Allocation<'a>,
    engram_keys: Allocation<'a>,
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
    idx_q_raw: Allocation<'a>,
    idx_q_rot: Allocation<'a>,
    idx_w_raw: Allocation<'a>,
    idx_w_out: Allocation<'a>,
    idx_tk_raw: Allocation<'a>,
    idx_tk: Allocation<'a>,
    idx_group_tok: Allocation<'a>,
    idx_scores: Allocation<'a>,
    idx_score_floats: usize,
    idx_cand_slots: Allocation<'a>,
    idx_cand_lens: Allocation<'a>,
    ctx: Option<ContextScratch<'a>>,
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
    logits_tile: Allocation<'a>,
    head_logits_f32: Option<Allocation<'a>>,
    native: Option<Hy4NativeScratch<'a>>,
}

impl<'a> SharedScratch<'a> {
    #[allow(clippy::too_many_arguments)]
    fn new(
        device: Device<'a>,
        cfg: &Hy4FlashConfig,
        rank: usize,
        tp: usize,
        max_rows: usize,
        decode_rows: usize,
        max_seq_pages: usize,
        max_sequences: usize,
        local_vocab: usize,
        attn_scratch_bytes: usize,
        context_split: bool,
        native_layout: Option<&ProjectionScratchLayout>,
        native_head_layout: Option<&ProjectionScratchLayout>,
    ) -> Result<Self> {
        let layout = SharedScratchLayout::new_with_context(
            cfg,
            rank,
            tp,
            max_rows,
            decode_rows,
            max_seq_pages,
            max_sequences,
            local_vocab,
            context_split,
        )?;
        let arena = DeviceArena::new(
            device,
            layout.arena.clone(),
            ArenaTag::workspace("hy4_flash/rank-scratch"),
        )?;
        let view = |name: &str| -> Result<Allocation<'a>> {
            let bytes = arena
                .layout()
                .view(name)
                .with_context(|| format!("Hy4 shared scratch missing view {name}"))?
                .bytes();
            arena.view(name, bytes)
        };
        let attn_scratch = view("attn_scratch")?;
        ensure!(
            attn_scratch_bytes <= attn_scratch.buffer.bytes,
            "Hy4 MLA scratch {attn_scratch_bytes} exceeds admitted {}",
            attn_scratch.buffer.bytes
        );
        let ctx = if context_split {
            Some(ContextScratch {
                cand_scores: view("ctx_cand_scores")?,
                cand_indices: view("ctx_cand_indices")?,
                partial_own: view("ctx_partial_own")?,
                lse_own: view("ctx_lse_own")?,
                partial_peer: view("ctx_partial_peer")?,
                lse_peer: view("ctx_lse_peer")?,
                draft_slots: view("ctx_draft_slots")?,
            })
        } else {
            None
        };
        let native = if native_layout.is_some() || native_head_layout.is_some() {
            Some(super::native::allocate_native_scratch(device, native_layout, native_head_layout)?)
        } else {
            None
        };
        let head_logits_f32 = if rank == 0 {
            Some(view("head_logits_f32")?)
        } else {
            None
        };
        Ok(Self {
            ids: view("ids")?,
            r_head_in: view("r_head_in")?,
            post: view("post")?,
            delta: view("delta")?,
            delta_f32: view("delta_f32")?,
            delta_sum_f32: view("delta_sum_f32")?,
            delta_norm: view("delta_norm")?,
            engram_normed: view("engram_normed")?,
            engram_val: view("engram_val")?,
            engram_keys: view("engram_keys")?,
            q_a: view("q_a")?,
            kv_a: view("kv_a")?,
            kv_a_padded: view("kv_a_padded")?,
            q_resid: view("q_resid")?,
            q_raw: view("q_raw")?,
            gate: view("gate")?,
            query: view("query")?,
            attn_out: view("attn_out")?,
            lse_out: view("lse_out")?,
            uv_out: view("uv_out")?,
            attn_scratch,
            idx_q_raw: view("idx_q_raw")?,
            idx_q_rot: view("idx_q_rot")?,
            idx_w_raw: view("idx_w_raw")?,
            idx_w_out: view("idx_w_out")?,
            idx_tk_raw: view("idx_tk_raw")?,
            idx_tk: view("idx_tk")?,
            idx_group_tok: view("idx_group_tok")?,
            idx_scores: view("idx_scores")?,
            idx_score_floats: layout.idx_score_floats,
            idx_cand_slots: view("idx_cand_slots")?,
            idx_cand_lens: view("idx_cand_lens")?,
            ctx,
            ffn_gate: view("ffn_gate")?,
            ffn_up: view("ffn_up")?,
            ffn_act: view("ffn_act")?,
            router_logits: view("router_logits")?,
            router_select: view("router_select")?,
            route_ids: view("route_ids")?,
            route_weights: view("route_weights")?,
            latent_in: view("latent_in")?,
            wire_in: view("wire_in")?,
            routed_latent: view("routed_latent")?,
            routed_out: view("routed_out")?,
            logits_tile: view("logits_tile")?,
            head_logits_f32,
            native,
            _arena: arena,
        })
    }

    fn inventory(&self, inventory: &mut Inventory<'_>, rank: usize) -> Result<()> {
        let prefix = format!("hy4_flash.scratch.{rank}");
        for (name, alloc) in [
            ("ids", &self.ids),
            ("r_head_in", &self.r_head_in),
            ("post", &self.post),
            ("delta", &self.delta),
            ("delta_f32", &self.delta_f32),
            ("delta_sum_f32", &self.delta_sum_f32),
            ("delta_norm", &self.delta_norm),
            ("engram_normed", &self.engram_normed),
            ("engram_val", &self.engram_val),
            ("engram_keys", &self.engram_keys),
            ("q_a", &self.q_a),
            ("kv_a", &self.kv_a),
            ("kv_a_padded", &self.kv_a_padded),
            ("q_resid", &self.q_resid),
            ("q_raw", &self.q_raw),
            ("gate", &self.gate),
            ("query", &self.query),
            ("attn_out", &self.attn_out),
            ("lse_out", &self.lse_out),
            ("uv_out", &self.uv_out),
            ("attn_scratch", &self.attn_scratch),
            ("idx_q_raw", &self.idx_q_raw),
            ("idx_q_rot", &self.idx_q_rot),
            ("idx_w_raw", &self.idx_w_raw),
            ("idx_w_out", &self.idx_w_out),
            ("idx_tk_raw", &self.idx_tk_raw),
            ("idx_tk", &self.idx_tk),
            ("idx_group_tok", &self.idx_group_tok),
            ("idx_scores", &self.idx_scores),
            ("idx_cand_slots", &self.idx_cand_slots),
            ("idx_cand_lens", &self.idx_cand_lens),
            ("ffn_gate", &self.ffn_gate),
            ("ffn_up", &self.ffn_up),
            ("ffn_act", &self.ffn_act),
            ("router_logits", &self.router_logits),
            ("router_select", &self.router_select),
            ("route_ids", &self.route_ids),
            ("route_weights", &self.route_weights),
            ("latent_in", &self.latent_in),
            ("wire_in", &self.wire_in),
            ("routed_latent", &self.routed_latent),
            ("routed_out", &self.routed_out),
            ("logits_tile", &self.logits_tile),
        ] {
            alloc.inventory(inventory, &format!("{prefix}.{name}"))?;
        }
        if let Some(ctx) = &self.ctx {
            for (name, alloc) in [
                ("ctx_cand_scores", &ctx.cand_scores),
                ("ctx_cand_indices", &ctx.cand_indices),
                ("ctx_partial_own", &ctx.partial_own),
                ("ctx_lse_own", &ctx.lse_own),
                ("ctx_partial_peer", &ctx.partial_peer),
                ("ctx_lse_peer", &ctx.lse_peer),
                ("ctx_draft_slots", &ctx.draft_slots),
            ] {
                alloc.inventory(inventory, &format!("{prefix}.{name}"))?;
            }
        }
        if let Some(head) = &self.head_logits_f32 {
            head.inventory(inventory, &format!("{prefix}.head_logits_f32"))?;
        }
        if let Some(native) = &self.native {
            native.inventory(inventory, rank)?;
        }
        Ok(())
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
        let scratch = r.shared.native.as_ref().and_then(|n| n.scratch.as_ref()).context("Hy4 native core missing scratch")?;
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
        let scratch = r.shared.native.as_ref().and_then(|n| n.scratch.as_ref()).context("Hy4 native core missing scratch")?;
        linear.launch(rows, input, r.shared.kv_a_padded.buffer, scratch, r.stream.raw)?;
        // SAFETY: kv_a_padded [rows, 640] BF16 and output_576 [rows, 576] BF16 are disjoint admitted buffers.
        return unsafe {
            r.device.library.copy_d2d_2d_async(
                output_576,
                KV_A_DIM * 2,
                r.shared.kv_a_padded.buffer,
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
        let scratch = r.shared.native.as_ref().and_then(|n| n.scratch.as_ref()).context("Hy4 native core missing scratch")?;
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
            .shared
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
    programs: Hy4Programs<'a>,
    mla_splits: MlaSplitPlan,
    weights: Hy4RankWeights<'a>,
    cache: Rc<Hy4CacheRank<'a>>,
    shared: SharedScratch<'a>,
    waves: Vec<WaveScratch<'a>>,
    bf16_add: Bf16Add<'a>,
    f32_add: F32Add<'a>,
    bf16_f32_add: Bf16F32Add<'a>,
    quantize_grid: Fp8QuantizeGrid,
    onboarded_experts: RefCell<Option<Exl3Tp2<'a>>>,
}

impl<'a> RankRuntime<'a> {
    fn engram_compute(&self, p: &str, slot: usize, wave_slot: usize, rows: usize, use_a: bool) -> Result<()> {
        let w = &self.waves[wave_slot];
        let r_cur = if use_a { w.r_a.buffer } else { w.r_b.buffer };
        let ep = format!("{p}.engram_layer");
        let s = [Scalar::I32(rows as i32)];
        // SAFETY: admitted arena weight views & rank scratch live through stream drain.
        unsafe {
            self.programs.engram_head_norm.launch(
                &[
                    w.engram_raw[slot].buffer.ptr,
                    self.weights.buf(&format!("{ep}.lookup_per_head_norms.packed"))?.ptr,
                    self.shared.engram_normed.buffer.ptr,
                ],
                &s, self.stream.raw,
            )?;
            self.programs.proj_engram_value.launch(
                &[self.shared.engram_normed.buffer.ptr, self.weights.weight(&format!("{ep}.value_proj"))?.ptr, self.shared.engram_val.buffer.ptr],
                &s, self.stream.raw,
            )?;
            self.programs.proj_engram_key.launch(
                &[self.shared.engram_normed.buffer.ptr, self.weights.buf(&format!("{ep}.key_projs.packed.weight"))?.ptr, self.shared.engram_keys.buffer.ptr],
                &s, self.stream.raw,
            )?;
            self.programs.engram_gate_add.launch(
                &[
                    r_cur.ptr,
                    self.shared.engram_keys.buffer.ptr,
                    self.weights.buf(&format!("{ep}.key_projs.packed.bias"))?.ptr,
                    self.shared.engram_val.buffer.ptr,
                    self.weights.buf(&format!("{ep}.value_proj.bias"))?.ptr,
                    self.weights.buf(&format!("{ep}.norm_key.packed"))?.ptr,
                    self.weights.buf(&format!("{ep}.norm_query.packed"))?.ptr,
                    self.weights.buf(&format!("{ep}.unified_pre_gamma"))?.ptr,
                    self.weights.buf(&format!("{ep}.unified_per_hc_scales.packed"))?.ptr,
                    w.engram_masked[slot].buffer.ptr,
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
        wave_slot: usize,
        rows: usize,
        use_a: bool,
        kv_cache_buf: CuteafdDeviceBuffer,
    ) -> Result<()> {
        let w = &self.waves[wave_slot];
        let r_cur = if use_a { w.r_a.buffer } else { w.r_b.buffer };
        let s_rows = [Scalar::I32(rows as i32)];
        let inv_freq = if is_sparse { self.weights.inv_freq_dsa.buffer.ptr } else { self.weights.inv_freq_swa.buffer.ptr };
        let kv_slots_ptr = if is_sparse {
            w.dsa_slots_i64.buffer.ptr
        } else {
            w.swa_slots_i64.buffer.ptr
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
                    self.shared.post.buffer.ptr,
                    w.y.buffer.ptr,
                ],
                &s_rows, self.stream.raw,
            )?;
            launch_proj_out(
                self,
                &format!("{p}.self_attn.q_a_proj"),
                &self.programs.proj_q_a,
                rows,
                w.y.buffer,
                self.shared.q_a.buffer,
            )?;
            launch_proj_kv_a(
                self,
                &format!("{p}.self_attn.kv_a_proj_with_mqa"),
                rows,
                w.y.buffer,
                self.shared.kv_a.buffer,
            )?;
            self.programs.mla_rank_norm_pack.launch(
                &[
                    self.shared.q_a.buffer.ptr,
                    self.shared.kv_a.buffer.ptr,
                    self.weights.buf(&format!("{p}.self_attn.q_a_layernorm.weight"))?.ptr,
                    self.weights.buf(&format!("{p}.self_attn.kv_a_layernorm.weight"))?.ptr,
                    kv_slots_ptr,
                    w.positions_i64.buffer.ptr,
                    inv_freq,
                    self.shared.q_resid.buffer.ptr,
                    kv_cache_buf.ptr,
                ],
                &s_rows, self.stream.raw,
            )?;
            launch_proj_out(
                self,
                &format!("{p}.self_attn.q_b_proj"),
                proj_qb,
                rows,
                self.shared.q_resid.buffer,
                self.shared.q_raw.buffer,
            )?;
            launch_proj_out(
                self,
                &format!("{p}.self_attn.linear_gate"),
                proj_gate,
                rows,
                w.y.buffer,
                self.shared.gate.buffer,
            )?;
            q_abs.launch(
                &[self.shared.q_raw.buffer.ptr, w_uk, w.positions_i64.buffer.ptr, inv_freq, self.shared.query.buffer.ptr],
                &s_rows, self.stream.raw,
            )
        }
    }

    fn dsa_index_wk(&self, p: &str, wave_slot: usize, rows: usize) -> Result<()> {
        let w = &self.waves[wave_slot];
        let ip = format!("{p}.self_attn.compressed_indexer");
        let s_rows = [Scalar::I32(rows as i32)];
        // SAFETY: arena weight views and scratch live through stream drain.
        unsafe {
            self.programs.proj_indexer_wk.launch(
                &[w.y.buffer.ptr, self.weights.weight(&format!("{ip}.token_key_projection"))?.ptr, self.shared.idx_tk_raw.buffer.ptr],
                &s_rows, self.stream.raw,
            )?;
            let tk_w = self.weights.buf(&format!("{ip}.token_key_norm.weight"))?.ptr;
            self.programs.rms_norm_128.launch(
                &[self.shared.idx_tk_raw.buffer.ptr, tk_w, tk_w, tk_w, tk_w, tk_w, self.shared.idx_tk.buffer.ptr],
                &s_rows, self.stream.raw,
            )
        }
    }

    fn dsa_index_q(&self, p: &str, wave_slot: usize, rows: usize) -> Result<()> {
        let w = &self.waves[wave_slot];
        let ip = format!("{p}.self_attn.compressed_indexer");
        let s_rows = [Scalar::I32(rows as i32)];
        let inv_freq = self.weights.inv_freq_dsa.buffer.ptr;
        // SAFETY: arena weight views and scratch live through stream drain.
        unsafe {
            self.programs.proj_indexer_wq_b.launch(
                &[self.shared.q_resid.buffer.ptr, self.weights.weight(&format!("{ip}.query_projection"))?.ptr, self.shared.idx_q_raw.buffer.ptr],
                &s_rows, self.stream.raw,
            )?;
            self.programs.proj_indexer_weights.launch(
                &[w.y.buffer.ptr, self.weights.weight(&format!("{ip}.head_weight_projection"))?.ptr, self.shared.idx_w_raw.buffer.ptr],
                &s_rows, self.stream.raw,
            )?;
            self.programs.indexer_query_rope.launch(
                &[
                    self.shared.idx_q_raw.buffer.ptr,
                    self.shared.idx_w_raw.buffer.ptr,
                    w.positions_i64.buffer.ptr,
                    inv_freq,
                    self.shared.idx_q_rot.buffer.ptr,
                    self.shared.idx_w_out.buffer.ptr,
                ],
                &s_rows, self.stream.raw,
            )
        }
    }

    fn dsa_attn_context_partials(
        &self,
        rank: usize,
        assembled_query: *mut std::ffi::c_void,
        kv_cache_buf: CuteafdDeviceBuffer,
        wave_slot: usize,
        rows: usize,
        decode: bool,
    ) -> Result<()> {
        let ctx_progs = self.programs.context.as_ref().context("Hy4 ContextSplit programs missing")?;
        let ctx = self.shared.ctx.as_ref().context("Hy4 ContextSplit scratch missing")?;
        let w = &self.waves[wave_slot];
        let splits = if decode { self.mla_splits.dsa_partial } else { 2 };
        // Every rank computes ALL 32 heads over its local parity pages by launching both h0 and h16:
        // rank 0 keeps h0 locally and pushes h16; rank 1 keeps h16 locally and pushes h0.
        let (own_prog, peer_prog) = if rank == 0 {
            (&ctx_progs.dsa_partial_h0, &ctx_progs.dsa_partial_h16)
        } else {
            (&ctx_progs.dsa_partial_h16, &ctx_progs.dsa_partial_h0)
        };
        let scalars = [
            Scalar::I32(rows as i32),
            Scalar::I64(CANDIDATE_TOPK as i64),
            Scalar::I32(splits),
        ];
        // SAFETY: assembled_query, local kv_cache, cand_slots/lens, and partial buffers live through stream drain.
        unsafe {
            own_prog.launch(
                &[
                    assembled_query,
                    kv_cache_buf.ptr,
                    self.shared.idx_cand_slots.buffer.ptr,
                    self.shared.idx_cand_lens.buffer.ptr,
                    w.dsa_cu_seqlens.buffer.ptr,
                    ctx.partial_own.buffer.ptr,
                    ctx.lse_own.buffer.ptr,
                    self.shared.attn_scratch.buffer.ptr,
                ],
                &scalars,
                self.stream.raw,
            )?;
            peer_prog.launch(
                &[
                    assembled_query,
                    kv_cache_buf.ptr,
                    self.shared.idx_cand_slots.buffer.ptr,
                    self.shared.idx_cand_lens.buffer.ptr,
                    w.dsa_cu_seqlens.buffer.ptr,
                    ctx.partial_peer.buffer.ptr,
                    ctx.lse_peer.buffer.ptr,
                    self.shared.attn_scratch.buffer.ptr,
                ],
                &scalars,
                self.stream.raw,
            )
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn dsa_context_combine(
        &self,
        rank: usize,
        p: &str,
        received_partial: *mut std::ffi::c_void,
        received_lse: *mut std::ffi::c_void,
        chunk_start: usize,
        chunk_rows: usize,
    ) -> Result<()> {
        let ctx_progs = self.programs.context.as_ref().context("Hy4 ContextSplit programs missing")?;
        let ctx = self.shared.ctx.as_ref().context("Hy4 ContextSplit scratch missing")?;
        let [o0, o1] = context::combine_order(rank, ctx.partial_own.buffer.ptr, received_partial);
        let [l0, l1] = context::combine_order(rank, ctx.lse_own.buffer.ptr, received_lse);
        let sink = self.weights.buf(&format!("{p}.self_attn.learnable_sink_param"))?.ptr;
        let out_ptr = self
            .shared
            .attn_out
            .buffer
            .ptr
            .cast::<u8>()
            .wrapping_add(chunk_start * 16 * 512 * 2)
            .cast();
        // SAFETY: ContextExchange::wait_partial ordered received_partial/lse before this kernel;
        // combine_order guarantees fixed GPU0-then-GPU1 order and applies natural-log sink once.
        unsafe {
            ctx_progs.lse_combine2.launch(
                &[o0, l0, o1, l1, sink, out_ptr],
                &[Scalar::I32(chunk_rows as i32)],
                self.stream.raw,
            )
        }
    }

    fn dsa_context_gate_uv_o(
        &self,
        p: &str,
        layer: usize,
        local_heads: usize,
        rows: usize,
    ) -> Result<()> {
        let ctx_progs = self.programs.context.as_ref().context("Hy4 ContextSplit programs missing")?;
        let (_, w_uv) = self.weights.kv_b_absorbed(layer, local_heads)?;
        // SAFETY: attn_out already has sink folded by lse_combine2; mla_gate_uv_dsa applies UV + sigmoid(gate) only.
        unsafe {
            ctx_progs.mla_gate_uv_dsa.launch(
                &[
                    self.shared.attn_out.buffer.ptr,
                    w_uv,
                    self.shared.gate.buffer.ptr,
                    self.shared.uv_out.buffer.ptr,
                ],
                &[Scalar::I32(rows as i32)],
                self.stream.raw,
            )?;
        }
        launch_proj_split(
            self,
            2,
            &format!("{p}.self_attn.o_proj"),
            &self.programs.proj_o_dsa,
            rows,
            self.shared.uv_out.buffer,
            self.shared.delta.buffer,
            self.shared.delta_f32.buffer,
        )
    }

    fn dsa_attn(&self, kv_cache_buf: CuteafdDeviceBuffer, wave_slot: usize, rows: usize, decode: bool) -> Result<()> {
        let w = &self.waves[wave_slot];
        // SAFETY: cache and scratch live through stream drain.
        unsafe {
            self.programs.mla_dsa_decode.launch(
                &[
                    self.shared.query.buffer.ptr,
                    kv_cache_buf.ptr,
                    self.shared.idx_cand_slots.buffer.ptr,
                    self.shared.idx_cand_lens.buffer.ptr,
                    w.dsa_cu_seqlens.buffer.ptr,
                    self.shared.attn_out.buffer.ptr,
                    self.shared.lse_out.buffer.ptr,
                    self.shared.attn_scratch.buffer.ptr,
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
        wave_slot: usize,
        rows: usize,
        members_len: usize,
        swa_pt_stride: usize,
        decode: bool,
    ) -> Result<()> {
        let w = &self.waves[wave_slot];
        let (swa_prog, splits) = if decode {
            (&self.programs.mla_swa_decode, self.mla_splits.swa)
        } else {
            (&self.programs.mla_swa_extend, 1)
        };
        // SAFETY: cache and scratch live through stream drain.
        unsafe {
            swa_prog.launch(
                &[
                    self.shared.query.buffer.ptr,
                    kv_cache_buf.ptr,
                    w.swa_page_table.buffer.ptr,
                    w.swa_seqlens.buffer.ptr,
                    w.swa_cu_seqlens.buffer.ptr,
                    self.shared.attn_out.buffer.ptr,
                    self.shared.lse_out.buffer.ptr,
                    self.shared.attn_scratch.buffer.ptr,
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
                    self.shared.attn_out.buffer.ptr,
                    w_uv,
                    self.shared.gate.buffer.ptr,
                    self.shared.lse_out.buffer.ptr,
                    self.weights.buf(&format!("{p}.self_attn.learnable_sink_param"))?.ptr,
                    self.shared.uv_out.buffer.ptr,
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
            self.shared.uv_out.buffer,
            self.shared.delta.buffer,
            self.shared.delta_f32.buffer,
        )
    }

    fn post_attn_ihc(&self, p: &str, wave_slot: usize, rows: usize, use_a: bool) -> Result<()> {
        let w = &self.waves[wave_slot];
        let (r_in, r_out) = if use_a {
            (w.r_a.buffer, w.r_b.buffer)
        } else {
            (w.r_b.buffer, w.r_a.buffer)
        };
        let s_rows = [Scalar::I32(rows as i32)];
        // SAFETY: arena weights and scratch live through stream drain.
        unsafe {
            let ppn = self.weights.buf(&format!("{p}.self_attn.post_proj_norm.weight"))?.ptr;
            self.programs.rms_norm_3072.launch(
                &[self.shared.delta.buffer.ptr, ppn, ppn, ppn, ppn, ppn, self.shared.delta_norm.buffer.ptr],
                &s_rows, self.stream.raw,
            )?;
            self.programs.ihc_post_pre.launch(
                &[
                    self.shared.delta_norm.buffer.ptr,
                    r_in.ptr,
                    self.shared.post.buffer.ptr,
                    self.weights.buf(&format!("{p}.hc_mlp_layer.hc_pre.hc_fn"))?.ptr,
                    self.weights.buf(&format!("{p}.hc_mlp_layer.hc_pre.hc_scale"))?.ptr,
                    self.weights.buf(&format!("{p}.hc_mlp_layer.hc_pre.hc_base"))?.ptr,
                    self.weights.buf(&format!("{p}.post_attention_layernorm.weight"))?.ptr,
                    r_out.ptr,
                    w.post_ffn.buffer.ptr,
                    w.y.buffer.ptr,
                ],
                &s_rows, self.stream.raw,
            )
        }
    }

    fn dense_ffn(&self, p: &str, gpus: usize, wave_slot: usize, rows: usize) -> Result<()> {
        let w = &self.waves[wave_slot];
        let s_rows = [Scalar::I32(rows as i32)];
        launch_proj_out(
            self,
            &format!("{p}.mlp.gate_proj"),
            &self.programs.proj_dense_gate_up,
            rows,
            w.y.buffer,
            self.shared.ffn_gate.buffer,
        )?;
        launch_proj_out(
            self,
            &format!("{p}.mlp.up_proj"),
            &self.programs.proj_dense_gate_up,
            rows,
            w.y.buffer,
            self.shared.ffn_up.buffer,
        )?;
        // SAFETY: arena weights and scratch live through stream drain.
        unsafe {
            self.programs.post_silu_split_dense.launch(
                &[self.shared.ffn_gate.buffer.ptr, self.shared.ffn_up.buffer.ptr, self.shared.ffn_act.buffer.ptr],
                &s_rows, self.stream.raw,
            )?;
        }
        launch_proj_split(
            self,
            gpus,
            &format!("{p}.mlp.down_proj"),
            &self.programs.proj_dense_down,
            rows,
            self.shared.ffn_act.buffer,
            self.shared.delta.buffer,
            self.shared.delta_f32.buffer,
        )
    }

    fn moe_router_shared(
        &self,
        p: &str,
        gpus: usize,
        wave_slot: usize,
        rows: usize,
        routed_scaling_factor: f32,
    ) -> Result<()> {
        self.moe_router(p, wave_slot, rows, routed_scaling_factor)?;
        self.shared_ffn(p, gpus, wave_slot, rows)
    }

    fn moe_router(&self, p: &str, wave_slot: usize, rows: usize, routed_scaling_factor: f32) -> Result<()> {
        let w = &self.waves[wave_slot];
        let s_rows = [Scalar::I32(rows as i32)];
        // SAFETY: all buffers live through stream drain.
        unsafe {
            self.programs.proj_router.launch(
                &[w.y.buffer.ptr, self.weights.weight(&format!("{p}.mlp.gate"))?.ptr, self.shared.router_logits.buffer.ptr],
                &s_rows, self.stream.raw,
            )?;
            let bias = self.weights.buf(&format!("{p}.mlp.gate.e_score_correction_bias"))?.ptr;
            self.device.library.copy_d2d_async(
                self.shared.router_select.buffer,
                self.shared.router_logits.buffer,
                rows * EXPERTS * 4,
                self.stream.raw,
            )?;
            self.device.library.router_select(
                self.shared.router_select.buffer.ptr,
                bias,
                std::ptr::null(),
                std::ptr::null(),
                self.shared.route_ids.buffer.ptr,
                self.shared.route_weights.buffer.ptr,
                rows,
                EXPERTS,
                TOPK,
                routed_scaling_factor,
                true,
                self.stream.raw,
            )?;
            self.programs.route_weights.launch(
                &[self.shared.router_logits.buffer.ptr, bias, self.shared.route_ids.buffer.ptr, self.shared.route_weights.buffer.ptr],
                &s_rows, self.stream.raw,
            )?;
            launch_proj_out(
                self,
                &format!("{p}.mlp.routed_expert_down_proj"),
                &self.programs.proj_latent_down,
                rows,
                w.y.buffer,
                self.shared.latent_in.buffer,
            )?;
            let grid = self.quantize_grid.blocks(rows, LATENT_H);
            self.programs.expert_input_quant.launch(
                &[
                    self.shared.latent_in.buffer.ptr,
                    self.shared.wire_in.buffer.ptr,
                    self.shared.wire_in.buffer.ptr.wrapping_byte_add(LATENT_H),
                    self.shared.wire_in.buffer.ptr,
                ],
                &[Scalar::I32(rows as i32), Scalar::I32(grid as i32)],
                self.stream.raw,
            )?;
        }
        Ok(())
    }

    fn shared_ffn(&self, p: &str, gpus: usize, wave_slot: usize, rows: usize) -> Result<()> {
        let w = &self.waves[wave_slot];
        let s_rows = [Scalar::I32(rows as i32)];
        // SAFETY: all buffers live through stream drain.
        unsafe {
            launch_proj_out(
                self,
                &format!("{p}.mlp.shared_experts.gate_proj"),
                &self.programs.proj_shared_gate_up,
                rows,
                w.y.buffer,
                self.shared.ffn_gate.buffer,
            )?;
            launch_proj_out(
                self,
                &format!("{p}.mlp.shared_experts.up_proj"),
                &self.programs.proj_shared_gate_up,
                rows,
                w.y.buffer,
                self.shared.ffn_up.buffer,
            )?;
            self.programs.post_silu_split_shared.launch(
                &[self.shared.ffn_gate.buffer.ptr, self.shared.ffn_up.buffer.ptr, self.shared.ffn_act.buffer.ptr],
                &s_rows, self.stream.raw,
            )?;
            launch_proj_split(
                self,
                gpus,
                &format!("{p}.mlp.shared_experts.down_proj"),
                &self.programs.proj_shared_down,
                rows,
                self.shared.ffn_act.buffer,
                w.shared_out.buffer,
                w.shared_f32.buffer,
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
                        ExpertInput::Fp8K32(self.shared.wire_in.buffer.ptr),
                        Routes {
                            ids: self.shared.route_ids.buffer.ptr,
                            weights: self.shared.route_weights.buffer.ptr,
                        },
                        self.stream.raw,
                        &mut || (),
                    )?;
                    if gpus == 1 {
                        let exp_out = CuteafdDeviceBuffer {
                            ptr: experts.output(),
                            bytes: rows * LATENT_H * 4,
                            device_id: self.device.id,
                            flags: 0,
                        };
                        self.device.library.cuda_f32_to_bf16_async(
                            exp_out,
                            self.shared.routed_latent.buffer,
                            rows * LATENT_H,
                            self.stream.raw,
                        )?;
                    }
                } else {
                    anyhow::bail!(
                        "layer {layer}: no onboarded hy4f:exl3-k45-mul1 expert package loaded (pass --expert-package or --skip-routed-experts for non-expert diagnostics)"
                    );
                }
            } else {
                self.device.library.cuda_zero_bytes_async(self.shared.routed_latent.buffer, rows * LATENT_H * 2, self.stream.raw)?;
                if gpus == 2 {
                    self.device.library.cuda_zero_bytes_async(self.shared.delta_f32.buffer, rows * LATENT_H * 4, self.stream.raw)?;
                }
            }
        }
        Ok(())
    }

    fn ffn_combine_post(
        &self,
        p: &str,
        is_dense: bool,
        wave_slot: usize,
        rows: usize,
        use_a: bool,
    ) -> Result<()> {
        let w = &self.waves[wave_slot];
        if !is_dense {
            launch_proj_out(
                self,
                &format!("{p}.mlp.routed_expert_up_proj"),
                &self.programs.proj_latent_up,
                rows,
                self.shared.routed_latent.buffer,
                self.shared.routed_out.buffer,
            )?;
            // SAFETY: routed_out and shared_out finished in stream order into disjoint delta.
            unsafe {
                self.bf16_add.launch(
                    self.shared.routed_out.buffer,
                    w.shared_out.buffer,
                    self.shared.delta.buffer,
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
            (w.r_a.buffer, w.r_b.buffer)
        } else {
            (w.r_b.buffer, w.r_a.buffer)
        };
        let s_rows = [Scalar::I32(rows as i32)];
        // SAFETY: arena weights and scratch live through stream drain.
        unsafe {
            let nw = self.weights.buf(&ffn_norm_name)?.ptr;
            self.programs.rms_norm_3072.launch(
                &[self.shared.delta.buffer.ptr, nw, nw, nw, nw, nw, self.shared.delta_norm.buffer.ptr],
                &s_rows, self.stream.raw,
            )?;
            self.programs.ihc_post.launch(
                &[self.shared.delta_norm.buffer.ptr, r_in.ptr, w.post_ffn.buffer.ptr, r_out.ptr],
                &s_rows, self.stream.raw,
            )
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn reduce_f32_to_bf16_local(
        &self,
        rank: usize,
        rows: usize,
        width: usize,
        egress_stem: Option<&str>,
        local_f32: CuteafdDeviceBuffer,
        peer_f32_ptr: *mut std::ffi::c_void,
        out_bf16: CuteafdDeviceBuffer,
    ) -> Result<()> {
        let elems = rows * width;
        let bytes = elems * 4;
        ensure!(local_f32.bytes >= bytes && !peer_f32_ptr.is_null(), "Hy4 TP2 local reduce operand invalid");
        let local = CuteafdDeviceBuffer { bytes, ..local_f32 };
        let peer = CuteafdDeviceBuffer {
            ptr: peer_f32_ptr,
            bytes,
            device_id: self.device.id,
            flags: 0,
        };
        let (a, b) = if rank == 0 { (local, peer) } else { (peer, local) };
        // SAFETY: PeerExchange::wait ordered peer_f32_ptr before this kernel; all operands are disjoint.
        unsafe {
            self.f32_add.launch(a, b, self.shared.delta_sum_f32.buffer, elems, self.stream.raw)?;
        }
        if let Some(linear) = egress_stem.and_then(|s| self.weights.native.get(s)) {
            linear.egress(rows, self.shared.delta_sum_f32.buffer, out_bf16, self.stream.raw)
        } else {
            // SAFETY: f32_add precedes narrowing into the disjoint BF16 consumer buffer.
            unsafe {
                self.device.library.cuda_f32_to_bf16_async(
                    self.shared.delta_sum_f32.buffer,
                    out_bf16,
                    elems,
                    self.stream.raw,
                )
            }
        }
    }

    /// Narrow just the wire partial, before native svh egress. The destination
    /// is the existing final BF16 output view, reused after the exchange.
    fn narrow_peer_partial(&self, rows: usize, width: usize, src: CuteafdDeviceBuffer,
        dst: CuteafdDeviceBuffer) -> Result<()> {
        // SAFETY: both buffers belong to this rank; the projection producer precedes
        // this conversion, and the peer push precedes overwriting dst on this stream.
        unsafe { self.device.library.cuda_f32_to_bf16_async(src, dst, rows * width, self.stream.raw) }
    }

    fn reduce_peer_bf16(&self, rank: usize, rows: usize, width: usize, stem: Option<&str>,
        local_and_output: CuteafdDeviceBuffer, peer: *mut std::ffi::c_void) -> Result<()> {
        let elems = rows * width;
        let peer = CuteafdDeviceBuffer { ptr: peer, bytes: elems * 2, device_id: self.device.id, flags: 0 };
        let (a, b) = if rank == 0 { (local_and_output, peer) } else { (peer, local_and_output) };
        // SAFETY: the peer wait orders both partials; the F32 sum is disjoint from them.
        unsafe { self.bf16_f32_add.launch(a, b, self.shared.delta_sum_f32.buffer, elems, self.stream.raw)?; }
        if let Some(linear) = stem.and_then(|s| self.weights.native.get(s)) {
            linear.egress(rows, self.shared.delta_sum_f32.buffer, local_and_output, self.stream.raw)
        } else {
            self.narrow_peer_partial(rows, width, self.shared.delta_sum_f32.buffer, local_and_output)
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn finish_deferred_ffn_local(
        &self,
        rank: usize,
        gpus: usize,
        prev: DeferredFfn,
        wave_slot: usize,
        rows: usize,
        use_a: bool,
        skip_routed: bool,
        ffn_recv_ptr: *mut std::ffi::c_void,
        routed_recv_offset: usize,
    ) -> Result<()> {
        let prev_p = format!("model.layers.{}", prev.layer);
        if gpus == 2 {
            if prev.is_dense {
                self.reduce_f32_to_bf16_local(
                    rank,
                    rows,
                    H,
                    Some(&format!("{prev_p}.mlp.down_proj")),
                    self.shared.delta_f32.buffer,
                    ffn_recv_ptr,
                    self.shared.delta.buffer,
                )?;
            } else {
                if prev.reduce_routed_tp2 {
                    let routed_in = routed_f32_input(self, rows, skip_routed)?;
                    self.reduce_f32_to_bf16_local(
                        rank,
                        rows,
                        LATENT_H,
                        None,
                        routed_in,
                        ffn_recv_ptr.wrapping_byte_add(routed_recv_offset),
                        self.shared.routed_latent.buffer,
                    )?;
                }
                self.reduce_f32_to_bf16_local(
                    rank,
                    rows,
                    H,
                    Some(&format!("{prev_p}.mlp.shared_experts.down_proj")),
                    self.waves[wave_slot].shared_f32.buffer,
                    ffn_recv_ptr,
                    self.waves[wave_slot].shared_out.buffer,
                )?;
            }
        }
        self.ffn_combine_post(&prev_p, prev.is_dense, wave_slot, rows, use_a)
    }

    #[allow(clippy::too_many_arguments)]
    fn record_expert_usage_local(
        &self,
        usage: Option<&ExpertUsage<'a>>,
        count_expert_usage: bool,
        skip_routed: bool,
        is_dense: bool,
        layer: usize,
        phase: UsagePhase,
        rows: usize,
    ) -> Result<()> {
        if !count_expert_usage || skip_routed || is_dense {
            return Ok(());
        }
        let usage = usage.context("Hy4 expert usage owner missing")?;
        let tally = usage.routes(self.device.id, layer, phase)?;
        // SAFETY: canonical router wrote route_ids on this stream; engine drains before dropping counters.
        unsafe { tally.record(self.shared.route_ids.buffer.ptr, rows, None, u32::MAX, true, self.stream.raw) }
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

#[derive(Clone, Copy, Debug)]
struct DeferredFfn {
    layer: usize,
    is_dense: bool,
    reduce_routed_tp2: bool,
}

struct ActiveBatch {
    wave_slot: usize,
    rows: usize,
    max_seq_end: usize,
    pt_stride: usize,
    ctx_pt_stride: usize,
    ctx_slot_counter: usize,
    swa_pt_stride: usize,
    all_single_row: bool,
    graphed: bool,
    ffn_fused_layer: Option<usize>,
    post_attn_deferred: bool,
    reduce_attn_pending: bool,
    prev_ffn_deferred: Option<DeferredFfn>,
    final_layer: usize,
    rank0_only_tail: bool,
    decode_attention: bool,
    usage_phase: UsagePhase,
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
    context_split: bool,
    graphs_enabled: Cell<bool>,
    swa_pt_host: Vec<i32>,
    metadata_uploads: Vec<Vec<MetadataUpload<'a>>>,
    ranks: Vec<RankRuntime<'a>>,
    // ContextExchange borrows PeerExchange's abort words, so it must drop before peer_exchange.
    context_exchange: Option<ContextExchange<'a>>,
    peer_exchange: Option<PeerExchange<'a>>,
    ffn_routed_recv_offset: usize,
    head_handoff: Option<Handoff<'a>>,
    spark: Option<SparkLink<'a>>,
    onboarded_layers: BTreeSet<usize>,
    next_identity: u64,
    placement_version: u64,
    dsa_index_of: BTreeMap<usize, usize>,
    swa_index_of: BTreeMap<usize, usize>,
    engram_index_of: BTreeMap<usize, usize>,
    golden_placement: Hy4Placement,
    expert_usage: Option<ExpertUsage<'a>>,
    count_expert_usage: bool,
    terminal: bool,
    // Provisional key rows; the committed incomplete group remains untouched
    // until target acceptance tells us which prefix of each member survived.
    verify_keys: Option<Vec<Allocation<'a>>>,
    verifying: bool,
    draft: Option<super::draft::Drafter<'a>>,
}

impl<'a> Hy4Engine<'a> {
    /// Single-sequence constructor used by `hy4f-golden`.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        devices: &[Device<'a>],
        programs: &'a Programs<'a>,
        checkpoint: &Checkpoint,
        cfg: Hy4FlashConfig,
        kv_format: Hy4KvCache,
        attention: AttentionPlacement,
        max_rows: usize,
        decode_rows: usize,
        prefill_waves: usize,
        max_pages: usize,
        skip_routed_experts: bool,
        onboarded_experts: Vec<Option<Exl3Tp2<'a>>>,
    ) -> Result<Self> {
        let context_split = devices.len() == 2 && attention == AttentionPlacement::Context;
        let max_context = max_pages * if context_split { 2 * PAGE_ROWS } else { PAGE_ROWS };
        Self::new_serving(
            devices,
            programs,
            None,
            checkpoint,
            cfg,
            kv_format,
            attention,
            max_rows,
            decode_rows,
            prefill_waves,
            max_pages,
            1,
            max_context,
            max_context,
            0,
            skip_routed_experts,
            onboarded_experts,
            0,
            Default::default(),
        )
    }

    /// Serving & multi-sequence constructor with exact prefix mark slots, COW page pool,
    /// and packed per-wave scratch arenas matching `Hy4Inputs` admission.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new_serving(
        devices: &[Device<'a>],
        programs: &'a Programs<'a>,
        mut preloaded_programs: Option<Vec<Hy4Programs<'a>>>,
        checkpoint: &Checkpoint,
        cfg: Hy4FlashConfig,
        kv_format: Hy4KvCache,
        attention: AttentionPlacement,
        unit_rows: usize,
        decode_rows: usize,
        prefill_waves: usize,
        max_pages: usize,
        max_sequences: usize,
        max_context: usize,
        admitted_context: usize,
        mark_slots: usize,
        skip_routed_experts: bool,
        mut onboarded_experts: Vec<Option<Exl3Tp2<'a>>>,
        draft_peer_rows: usize,
        media: crate::shared::vision::push::Contribution,
    ) -> Result<Self> {
        ensure!((1..=DECODE_ROWS as usize).contains(&decode_rows),
            "Hy4-flash split-K decode exports cover at most {DECODE_ROWS} rows; got {decode_rows}");
        let max_rows = unit_rows.max(decode_rows);
        let gpus = devices.len();
        ensure!(matches!(gpus, 1 | 2), "Hy4Engine supports 1 or 2 GPUs, got {gpus}");
        let context_split = match (gpus, attention) {
            (1, AttentionPlacement::Heads) => false,
            (2, AttentionPlacement::Heads) => false,
            (2, AttentionPlacement::Context) => true,
            _ => anyhow::bail!("Hy4Engine does not support attention placement {attention:?} on {gpus} GPU(s)"),
        };
        ensure!(onboarded_experts.len() == gpus, "onboarded_experts rank count mismatch");
        ensure!(
            max_sequences > 0 && max_pages > 0 && unit_rows > 0 && (1..=4).contains(&prefill_waves),
            "invalid Hy4Engine extents"
        );
        if let Some(pre) = &preloaded_programs {
            ensure!(pre.len() == gpus, "preloaded Hy4Programs rank count mismatch");
        }
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
        let weights_all = Hy4RankWeights::load_all(devices, programs, checkpoint, &cfg, max_rows, head_rows, media)?;
        // SAFETY: checkpoint shard files remain immutable for the duration of the engine.
        let engram = unsafe { Hy4EngramHost::open(devices[0].library, checkpoint, &cfg, unit_rows, prefill_waves, decode_rows)? };
        let max_seq_pages = admitted_context.div_ceil(PAGE_ROWS).max(16);

        let metadata_uploads = if context_split {
            (0..gpus)
                .map(|rank| {
                    (0..prefill_waves)
                        .map(|slot| {
                            let wave = WaveScratchLayout::new(&cfg, rank, slot, gpus, max_rows, max_seq_pages, max_sequences)?;
                            MetadataUpload::new(&[devices[rank]], wave_metadata_layout_for_rank(rank, slot, &wave.arena)?)
                        })
                        .collect::<Result<Vec<_>>>()
                })
                .collect::<Result<Vec<_>>>()?
        } else {
            vec![(0..prefill_waves)
                .map(|slot| {
                    let wave = WaveScratchLayout::new(&cfg, 0, slot, gpus, max_rows, max_seq_pages, max_sequences)?;
                    MetadataUpload::new(devices, wave_metadata_layout_for_rank(0, slot, &wave.arena)?)
                })
                .collect::<Result<Vec<_>>>()?]
        };
        let mut ranks = Vec::with_capacity(gpus);
        for (rank, weights) in weights_all.into_iter().enumerate() {
            let device = devices[rank];
            let stream = Rc::new(Stream::new(device)?);
            let progs = match preloaded_programs.as_mut() {
                Some(pre) => pre.remove(0),
                None => device.run(|| Hy4Programs::load(programs, gpus, kv_format, context_split))?,
            };
            let sms = device.run(|| device.library.sm_count())?;
            let mla_splits = MlaSplitPlan::new(&progs, max_context, decode_rows, sms)?;
            if let Some(ctx) = &progs.context {
                let width = context_exchange_rows(max_rows, decode_rows);
                for program in [&ctx.dsa_partial_h0, &ctx.dsa_partial_h16, &ctx.lse_combine2,
                    &ctx.indexer_scored_topk, &ctx.indexer_merge_expand_owned] {
                    ensure!(width <= program.spec().capacity_rows as usize,
                        "{} cannot cover admitted {width}-row context exchange", program.spec().name);
                }
            }
            // Split-capable launches never exceed the admitted decode/verify rows.
            let attn_scratch = mla_split_scratch_bytes(decode_rows, gpus)
                .max(if context_split { hy4_dsa_partial_scratch_bytes(context_exchange_rows(max_rows, decode_rows)) } else { 0 });
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
            let shared = SharedScratch::new(
                device,
                &cfg,
                rank,
                gpus,
                max_rows,
                decode_rows,
                max_seq_pages,
                max_sequences,
                weights.local_vocab,
                attn_scratch,
                context_split,
                native_layout.as_ref(),
                native_head_layout.as_ref(),
            )?;
            let mut waves = Vec::with_capacity(prefill_waves);
            for slot in 0..prefill_waves {
                waves.push(WaveScratch::new(
                    device,
                    &cfg,
                    rank,
                    slot,
                    gpus,
                    max_rows,
                    max_seq_pages,
                    max_sequences,
                )?);
            }
            let bf16_add = device.run(|| device.library.bf16_add())?;
            let f32_add = device.run(|| device.library.f32_add())?;
            let bf16_f32_add = device.run(|| device.library.bf16_f32_add())?;
            let quantize_grid = Fp8QuantizeGrid::new(sms, None)?;
            ranks.push(RankRuntime {
                graphs: Hy4Graphs::new(),
                device,
                stream,
                programs: progs,
                mla_splits,
                weights,
                cache,
                shared,
                waves,
                bf16_add,
                f32_add,
                bf16_f32_add,
                quantize_grid,
                onboarded_experts: RefCell::new(onboarded_experts[rank].take()),
            });
        }

        let peer_rows = max_rows.max(draft_peer_rows);
        let ffn_routed_recv_offset = hy4_peer_ffn_routed_offset(&cfg, peer_rows);
        let peer_exchange = if gpus == 2 {
            let sizes = hy4_peer_slot_sizes(&cfg, peer_rows);
            let decode_lane_bytes = (decode_rows * HC * H * 2)
                .max(decode_rows * (H + LATENT_H) * 4)
                .max(draft_peer_rows.min(96) * H * 4);
            let exchange = PeerExchange::new_abortable_sized(
                devices[0].library,
                [
                    RankDevice { device: ranks[0].device.id, stream: ranks[0].stream.raw },
                    RankDevice { device: ranks[1].device.id, stream: ranks[1].stream.raw },
                ],
                &sizes,
                decode_lane_bytes,
            )?;
            exchange.trace_identity("hy4_flash.tp2");
            Some(exchange)
        } else {
            None
        };
        let context_exchange = if context_split {
            let peer = peer_exchange.as_ref().context("Hy4 ContextSplit requires TP2 PeerExchange")?;
            let abort = [
                peer.abort_word(0)?.context("Hy4 rank 0 exchange not abortable")?,
                peer.abort_word(1)?.context("Hy4 rank 1 exchange not abortable")?,
            ];
            let buffers = hy4_context_buffers(&cfg, max_rows, decode_rows);
            Some(ContextExchange::new(
                devices[0].library,
                [
                    RankDevice { device: ranks[0].device.id, stream: ranks[0].stream.raw },
                    RankDevice { device: ranks[1].device.id, stream: ranks[1].stream.raw },
                ],
                &buffers,
                Some(abort),
            )?)
        } else {
            None
        };
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
        let golden_pages = if context_split { max_pages * 2 } else { max_pages };
        let golden_placement = Hy4Placement::new((0..golden_pages as u32).collect(), 0, cfg.engram_pad_id);

        let onboarded_layers: BTreeSet<usize> = cfg.routed_layers().into_iter().collect();
        let swa_pt_host = vec![0i32; max_sequences * max_seq_pages.max(16)];
        let engine = Self {
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
            context_split,
            graphs_enabled: Cell::new(false),
            swa_pt_host,
            metadata_uploads,
            ranks,
            context_exchange,
            peer_exchange,
            ffn_routed_recv_offset,
            head_handoff,
            spark: None,
            onboarded_layers,
            next_identity: 1,
            placement_version: 1,
            dsa_index_of,
            swa_index_of,
            engram_index_of,
            golden_placement,
            expert_usage: None,
            count_expert_usage: false,
            terminal: false,
            verify_keys: None,
            verifying: false,
            draft: None,
        };
        engine.assert_disjoint_wave_engram_buffers()?;
        Ok(engine)
    }

    pub(crate) fn prefill_waves(&self) -> usize {
        self.prefill_waves
    }

    pub(crate) fn golden_engram_tail(&self) -> [u32; 2] {
        self.golden_placement.engram_tail
    }

    /// Assert that across every rank and every live wave slot, `engram_raw` and `engram_masked`
    /// device byte ranges (and backing `WaveScratch` arenas) are strictly disjoint.
    pub(crate) fn assert_disjoint_wave_engram_buffers(&self) -> Result<()> {
        for (rank, r) in self.ranks.iter().enumerate() {
            let mut spans: Vec<(usize, usize, String)> = Vec::new();
            let mut arena_spans: Vec<(usize, usize, usize)> = Vec::new();
            for (slot, w) in r.waves.iter().enumerate() {
                let abase = w.positions_i64.buffer.ptr as usize;
                let aend = abase.checked_add(w._arena.layout().bytes()).context("wave arena ptr overflow")?;
                arena_spans.push((abase, aend, slot));
                for (i, buf) in w.engram_raw.iter().enumerate() {
                    let start = buf.buffer.ptr as usize;
                    let end = start.checked_add(buf.buffer.bytes).context("engram_raw ptr overflow")?;
                    ensure!(start != 0 && buf.buffer.bytes > 0, "rank {rank} wave {slot} engram_raw.{i} null/empty");
                    spans.push((start, end, format!("rank{rank}.wave{slot}.engram_raw.{i}")));
                }
                for (i, buf) in w.engram_masked.iter().enumerate() {
                    let start = buf.buffer.ptr as usize;
                    let end = start.checked_add(buf.buffer.bytes).context("engram_masked ptr overflow")?;
                    ensure!(start != 0 && buf.buffer.bytes > 0, "rank {rank} wave {slot} engram_masked.{i} null/empty");
                    spans.push((start, end, format!("rank{rank}.wave{slot}.engram_masked.{i}")));
                }
            }
            arena_spans.sort_by_key(|&(s, _, _)| s);
            for pair in arena_spans.windows(2) {
                ensure!(
                    pair[0].1 <= pair[1].0,
                    "rank {rank} wave arenas overlap: wave {} [{:#x}..{:#x}) vs wave {} [{:#x}..{:#x})",
                    pair[0].2, pair[0].0, pair[0].1, pair[1].2, pair[1].0, pair[1].1
                );
            }
            spans.sort_by_key(|&(s, _, _)| s);
            for pair in spans.windows(2) {
                ensure!(
                    pair[0].1 <= pair[1].0,
                    "per-wave Engram buffers overlap: {} [{:#x}..{:#x}) vs {} [{:#x}..{:#x})",
                    pair[0].2, pair[0].0, pair[0].1, pair[1].2, pair[1].0, pair[1].1
                );
            }
        }
        Ok(())
    }

    /// Human-readable device pointer ranges for per-wave `engram_raw` and `engram_masked` buffers.
    pub(crate) fn wave_engram_buffer_summary(&self) -> Vec<String> {
        let mut out = Vec::new();
        for (rank, r) in self.ranks.iter().enumerate() {
            for (slot, w) in r.waves.iter().enumerate() {
                let raws: Vec<String> = w
                    .engram_raw
                    .iter()
                    .map(|b| format!("{:#x}+{}", b.buffer.ptr as usize, b.buffer.bytes))
                    .collect();
                let masks: Vec<String> = w
                    .engram_masked
                    .iter()
                    .map(|b| format!("{:#x}+{}", b.buffer.ptr as usize, b.buffer.bytes))
                    .collect();
                out.push(format!(
                    "rank={rank} wave={slot} engram_raw=[{}] engram_masked=[{}]",
                    raws.join(","),
                    masks.join(",")
                ));
            }
        }
        out
    }

    /// Attach a bootstrapped Spark expert link for routed layers not onboarded onto the RTX GPUs.
    pub(crate) fn allocate_verify_keys(&mut self) -> Result<()> {
        ensure!(self.verify_keys.is_none(), "Hy4 DFly2 verify keys already allocated");
        self.verify_keys = Some(self.ranks.iter().map(|rank| {
            Allocation::new(rank.device,
                cuteafd_loader::families::hy4_flash::draft::verify_key_bytes(&self.cfg, self.decode_rows),
                Tag::workspace("hy4_flash/dfly2/verify-keys"))
        }).collect::<Result<Vec<_>>>()?);
        Ok(())
    }

    pub(crate) fn set_provisional(&mut self, enabled: bool) -> Result<()> {
        ensure!(!enabled || self.verify_keys.is_some(), "Hy4 speculative verify is not admitted");
        self.verifying = enabled;
        Ok(())
    }

    pub(crate) fn commit_provisional_keys(
        &self, slot: usize, start: usize, first_row: usize, keep: usize,
    ) -> Result<()> {
        let keys = self.verify_keys.as_ref().context("Hy4 missing provisional DSA keys")?;
        ensure!(keep > 0 && first_row + keep <= self.decode_rows, "Hy4 accepted verify extent differs");
        let pending_range = accepted_pending_copy(start, keep);
        for (rank, owner) in self.ranks.iter().zip(keys) {
            rank.device.run(|| {
                let mut counts = rank.cache.pending_counts.borrow_mut();
                for (layer, count) in counts[slot].iter_mut().enumerate() {
                    ensure!(*count == start % 4, "Hy4 committed DSA frontier moved during verify");
                    if let Some((src, dst, rows)) = pending_range {
                        let src_off = (layer * self.decode_rows + first_row + src) * IDX_KEY_BYTES;
                        let src = slice_buf(owner.buffer, src_off, rows * IDX_KEY_BYTES)?;
                        let dst = slice_buf(rank.cache.pending_key_slot(slot, layer)?, dst * IDX_KEY_BYTES, src.bytes)?;
                        // SAFETY: both views are on this stream's device. The
                        // committed owner and provisional source are disjoint.
                        unsafe { rank.device.library.copy_d2d_async(dst, src, src.bytes, rank.stream.raw)?; }
                    }
                    *count = (start + keep) % 4;
                }
                Ok(())
            })?;
        }
        Ok(())
    }

    pub(super) fn attach_drafter(&mut self, draft: super::draft::Drafter<'a>) -> Result<()> {
        self.allocate_verify_keys()?;
        self.draft = Some(draft);
        Ok(())
    }

    pub(super) fn invalidate_draft(&self, slot: usize) {
        if let Some(draft) = &self.draft { draft.invalidate(slot); }
    }

    pub(super) fn draft_valid(&self, placement: &Hy4Placement) -> bool {
        self.draft.as_ref().is_some_and(|d| d.valid(placement.slot, placement.len))
    }

    pub(super) fn draft_propose(&self, placement: &Hy4Placement, bonus: u32, depth: usize) -> Result<Vec<u32>> {
        if depth == 0 || !self.draft_valid(placement) { return Ok(Vec::new()); }
        let batch = [super::draft::Member { slot: placement.slot, pos: placement.len, bonus, depth,
            rope_delta: placement.rope.delta() }];
        Ok(self.draft_batch(&batch)?.pop().unwrap_or_default())
    }

    pub(super) fn draft_batch(&self, members: &[super::draft::Member]) -> Result<Vec<Vec<u32>>> {
        let Some(draft) = &self.draft else { return Ok(Vec::new()); };
        let r0 = &self.ranks[0];
        let embedding = r0.weights.embedding.as_ref().context("DFly2 target embedding missing")?;
        let streams: Vec<&Stream<'a>> = self.ranks.iter().map(|r| r.stream.as_ref()).collect();
        let graphs_vec: Vec<&Hy4Graphs<'a>> = self.ranks.iter().map(|r| &r.graphs).collect();
        let graphs = self.graphs_active().then_some(graphs_vec.as_slice());
        let ropes: Vec<CuteafdDeviceBuffer> = self.ranks.iter().map(|r| r.weights.inv_freq_dsa.buffer).collect();
        if graphs.is_some() {
            self.set_graph_cycle(ReadPriority::Decode);
        }
        let result = draft.propose(
            &streams,
            embedding,
            members,
            graphs,
            &ropes,
            self.peer_exchange.as_ref(),
            self.context_split,
            |rank, layer| self.ranks[rank].cache.dsa_kv_layer(self.dsa_index_of[&layer]),
            |hidden, rows| self.project_vocab_tile(0, hidden, rows),
        );
        // Successful proposals settle with their following target round. On an
        // error there may not be a target round to settle an aborted capture.
        if graphs.is_some() && result.is_err() {
            let _ = self.graph_completed();
        }
        result
    }

    pub(super) fn accept_draft(&self, slot: usize, row: usize, pos: usize) -> Result<()> {
        if let Some(draft) = &self.draft {
            let streams: Vec<&Stream<'a>> = self.ranks.iter().map(|r| r.stream.as_ref()).collect();
            draft.accept(slot, row, pos, &streams)?;
        }
        Ok(())
    }

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
        if let Some(exchange) = &self.peer_exchange {
            let _ = exchange.publish_abort();
        }
        let drained = self.drain();
        if let Some(exchange) = &self.peer_exchange {
            exchange.finish_terminal(drained.is_ok());
        }
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

    pub(crate) fn configure_graphs(
        &mut self,
        layout: Option<cuteafd_loader::serving_capacity::hy4_graphs::Hy4GraphLayout>,
    ) -> Result<()> {
        let tp2_dfly2 = self.draft.as_ref().is_some_and(|d| d.tp() == 2);
        for (rank, r) in self.ranks.iter_mut().enumerate() {
            r.graphs.attach(r.device.library, r.device.id, rank, layout, tp2_dfly2)?;
        }
        self.graphs_enabled.set(true);
        Ok(())
    }

    pub(crate) fn graphs_active(&self) -> bool {
        self.graphs_enabled.get() && self.ranks.iter().all(|r| r.graphs.attached())
    }

    pub(super) fn retained_draft_graph_rows(&self, members: usize) -> usize {
        self.draft.as_ref().map_or(0, |draft| {
            // Lead rank owns the graphed DraftFusion and DraftCorrection sections
            // (plus DraftLayer on TP1); TP2 keeps the 5-layer draft body eager on
            // both ranks so rank 1 preserves 100% of its bounded target graph cache.
            let rows = members * draft.cfg.block_rows();
            self.ranks[0].graphs.retained_draft_rows(rows)
        })
    }

    pub(crate) fn retained_graph_rows(&self, rows: usize) -> Vec<(i32, usize)> {
        self.ranks
            .iter()
            .map(|r| (r.device.id, r.graphs.retained_rows(rows)))
            .collect()
    }

    pub(crate) fn without_graphs<T>(&mut self, work: impl FnOnce(&mut Self) -> Result<T>) -> Result<T> {
        let prev = self.graphs_enabled.replace(false);
        let result = work(self);
        self.graphs_enabled.set(prev);
        result
    }

    fn set_graph_cycle(&self, priority: ReadPriority) {
        let cycle = match priority {
            ReadPriority::Decode | ReadPriority::DecodeFirst if self.verifying => GraphCycle::Verify,
            ReadPriority::Decode | ReadPriority::DecodeFirst => GraphCycle::Decode,
            ReadPriority::Prefill | ReadPriority::PrefillFirst => GraphCycle::Prefill,
        };
        for r in &self.ranks {
            r.graphs.cycle(cycle);
        }
    }

    pub(super) fn graph_completed(&self) -> Result<()> {
        if !self.graphs_active() {
            return Ok(());
        }
        if !self.ranks.iter().any(|r| r.graphs.needs_boundary()) {
            for r in &self.ranks {
                r.graphs.tick();
            }
            return Ok(());
        }
        let wait = std::time::Instant::now();
        self.drain()?;
        graph_device::record_boundary_drain(self.ranks.iter().map(|r| r.device.id), wait.elapsed());
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

    pub(crate) fn library(&self) -> &'a cuteafd_ffi::NativeLibrary {
        self.ranks[0].device.library
    }

    pub(crate) fn lead_device_id(&self) -> i32 {
        self.ranks[0].device.id
    }

    pub(crate) fn prefix(&self) -> Result<Hy4Prefix<'a>> {
        let caches: Vec<Rc<Hy4CacheRank<'a>>> = self.ranks.iter().map(|r| r.cache.clone()).collect();
        Hy4Prefix::new(caches, self.cfg.engram_pad_id, self.context_split)
    }

    pub(crate) fn dump_state(&self, placement: &Hy4Placement) -> Result<Vec<super::cache::Hy4RankStateSnapshot>> {
        self.ranks
            .iter()
            .enumerate()
            .map(|(rank, r)| r.cache.dump_state(placement, self.context_split.then_some(rank)))
            .collect()
    }

    pub(crate) fn mark_slots(&self) -> usize {
        self.ranks[0].cache.mark_slots
    }

    pub(crate) fn require_live(&self) -> Result<()> {
        ensure!(!self.terminal, "Hy4-flash execution is terminal");
        if let Some(exchange) = &self.peer_exchange {
            exchange.require_live()?;
        }
        Ok(())
    }

    pub(crate) fn drain(&self) -> Result<()> {
        for r in &self.ranks {
            r.stream.drain()?;
            r.cache.drain()?;
        }
        self.engram.drain()
    }

    pub(crate) fn configure_expert_usage(&mut self, geometry: Option<&cuteafd_core::expert_usage::Geometry>,
        reader: cuteafd_core::expert_usage::Reader) -> Result<()> {
        if let Some(geometry) = geometry {
            self.expert_usage = Some(ExpertUsage::new(self.library(), geometry.clone(),
                &[UsageDevice { logical: 0, cuda: self.lead_device_id() }], reader)?);
        }
        Ok(())
    }

    pub(crate) fn enable_expert_usage(&mut self) {
        self.count_expert_usage = self.expert_usage.is_some();
    }

    fn record_expert_usage(&self, layer: usize, batch: &ActiveBatch) -> Result<()> {
        if !self.count_expert_usage || self.skip_routed_experts || self.cfg.dense_layer[layer] {
            return Ok(());
        }
        let rank = &self.ranks[0];
        let usage = self.expert_usage.as_ref().context("Hy4 expert usage owner missing")?;
        let tally = usage.routes(rank.device.id, layer, batch.usage_phase)?;
        // SAFETY: the canonical router wrote the exact row width on this stream;
        // the engine drains all ranks before releasing counters and route storage.
        rank.device.run(|| unsafe { tally.record(rank.shared.route_ids.buffer.ptr,
            batch.rows, None, u32::MAX, true, rank.stream.raw) })
    }

    pub(crate) fn inventory(&self, weights_inv: &mut Inventory<'_>, retained_inv: &mut Inventory<'_>) -> Result<()> {
        if let Some(usage) = &self.expert_usage {
            for owner in usage.allocation_receipts() {
                match owner.storage {
                    UsageStorage::Device(buffer) => retained_inv.device(&owner.owner, buffer)?,
                    UsageStorage::Pinned(buffer) => retained_inv.pinned(&owner.owner, buffer)?,
                }
            }
        }
        if let Some(draft) = &self.draft {
            draft.inventory(weights_inv, retained_inv)?;
        }
        if let Some(keys) = &self.verify_keys {
            for (rank, keys) in keys.iter().enumerate() {
                retained_inv.device(&format!("hy4_flash/dfly2/verify-keys/gpu{rank}"), keys.buffer)?;
            }
        }
        for group in &self.metadata_uploads {
            for upload in group {
                retained_inv.pinned(upload.layout.id(), upload.backing.buffer)?;
            }
        }
        if let Some(exchange) = &self.peer_exchange {
            for (rank, name, buf) in exchange.allocation_buffers() {
                retained_inv.device(&format!("hy4_flash.peer.{rank}.{name}"), buf)?;
            }
        }
        if let Some(exchange) = &self.context_exchange {
            for (rank, buf) in exchange.allocation_buffers().into_iter().enumerate() {
                retained_inv.device(&format!("hy4_flash.context_exchange.{rank}"), buf)?;
            }
        }
        for (rank, r) in self.ranks.iter().enumerate() {
            r.weights.inventory(weights_inv, retained_inv)?;
            r.cache.inventory(retained_inv)?;
            r.shared.inventory(retained_inv, rank)?;
            for (slot, wave) in r.waves.iter().enumerate() {
                wave.inventory(retained_inv, rank, slot)?;
            }
            if let Some(experts) = r.onboarded_experts.borrow().as_ref() {
                experts.allocation_receipts(weights_inv, &format!("hy4_flash.experts.{rank}"))?;
            }
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
        let golden_pages = if self.context_split { self.max_pages * 2 } else { self.max_pages };
        self.golden_placement = Hy4Placement::new((0..golden_pages as u32).collect(), 0, self.cfg.engram_pad_id);
        Ok(())
    }

    fn peer_exchange(&self) -> Result<&PeerExchange<'a>> {
        self.peer_exchange.as_ref().context("Hy4 TP2 PeerExchange missing")
    }

    fn peer_push_two_planes(
        &self,
        from: usize,
        slot: usize,
        p0: (*mut std::ffi::c_void, *const std::ffi::c_void, usize),
        p1: (*mut std::ffi::c_void, *const std::ffi::c_void, usize),
    ) -> Result<()> {
        let exchange = self.peer_exchange()?;
        let decode_cap = (self.decode_rows * HC * H * 2).max(self.decode_rows * (H + LATENT_H) * 4);
        if p0.2 + p1.2 <= decode_cap {
            let planes = [
                PeerPlane {
                    destination: p0.0,
                    source: p0.1,
                    rows: 1,
                    row_bytes: p0.2 as u64,
                    destination_pitch: p0.2 as u64,
                    source_pitch: p0.2 as u64,
                },
                PeerPlane {
                    destination: p1.0,
                    source: p1.1,
                    rows: 1,
                    row_bytes: p1.2 as u64,
                    destination_pitch: p1.2 as u64,
                    source_pitch: p1.2 as u64,
                },
            ];
            return exchange.push_planes(from, slot, &planes);
        }
        let r = &self.ranks[from];
        let peer_dev = self.ranks[1 - from].device.id;
        r.device.run(|| unsafe {
            for (dst, src, bytes) in [p0, p1] {
                if bytes == 0 {
                    continue;
                }
                let dst_buf = CuteafdDeviceBuffer { ptr: dst, bytes, device_id: peer_dev, flags: 0 };
                let src_buf = CuteafdDeviceBuffer { ptr: src.cast_mut(), bytes, device_id: r.device.id, flags: 0 };
                r.device.library.copy_d2d_async(dst_buf, src_buf, bytes, r.stream.raw)?;
            }
            Ok(())
        })?;
        exchange.signal(from, slot)
    }

    /// Push both ranks' attention F32 partials on `PEER_SLOT_ATTN` outside capture, then wait both.
    fn exchange_attn_tp2(&self, rows: usize) -> Result<()> {
        if self.ranks.len() == 1 {
            return Ok(());
        }
        let bytes = rows * H * 4;
        let exchange = self.peer_exchange()?;
        exchange.push(0, PEER_SLOT_ATTN, self.ranks[0].shared.delta_f32.buffer.ptr, bytes)?;
        exchange.push(1, PEER_SLOT_ATTN, self.ranks[1].shared.delta_f32.buffer.ptr, bytes)?;
        exchange.wait(0, PEER_SLOT_ATTN)?;
        exchange.wait(1, PEER_SLOT_ATTN)
    }

    /// Push FFN partials on `PEER_SLOT_FFN` outside capture (combining local routed + shared into
    /// one `push_planes` exchange on MoE layers), then wait.
    fn exchange_ffn_tp2(
        &self,
        wave_slot: usize,
        rows: usize,
        is_dense: bool,
        reduce_routed_tp2: bool,
        rank0_only_consumer: bool,
    ) -> Result<()> {
        if self.ranks.len() == 1 {
            return Ok(());
        }
        let exchange = self.peer_exchange()?;
        let shared_bytes = rows * H * 4;
        if is_dense {
            if !rank0_only_consumer {
                exchange.push(0, PEER_SLOT_FFN, self.ranks[0].shared.delta_f32.buffer.ptr, shared_bytes)?;
            }
            exchange.push(1, PEER_SLOT_FFN, self.ranks[1].shared.delta_f32.buffer.ptr, shared_bytes)?;
        } else if reduce_routed_tp2 {
            let routed_bytes = rows * LATENT_H * 4;
            let skip = self.skip_routed_experts;
            let routed0 = routed_f32_input(&self.ranks[0], rows, skip)?;
            let routed1 = routed_f32_input(&self.ranks[1], rows, skip)?;
            if !rank0_only_consumer {
                let dst1 = exchange.recv(1, PEER_SLOT_FFN)?;
                self.peer_push_two_planes(
                    0,
                    PEER_SLOT_FFN,
                    (dst1, self.ranks[0].waves[wave_slot].shared_f32.buffer.ptr, shared_bytes),
                    (dst1.wrapping_byte_add(self.ffn_routed_recv_offset), routed0.ptr, routed_bytes),
                )?;
            }
            let dst0 = exchange.recv(0, PEER_SLOT_FFN)?;
            self.peer_push_two_planes(
                1,
                PEER_SLOT_FFN,
                (dst0, self.ranks[1].waves[wave_slot].shared_f32.buffer.ptr, shared_bytes),
                (dst0.wrapping_byte_add(self.ffn_routed_recv_offset), routed1.ptr, routed_bytes),
            )?;
        } else {
            // Spark MoE: rank 0 sends both shared_f32 and BF16 routed_latent in one exchange; rank 1 sends shared_f32.
            if !rank0_only_consumer {
                let dst1 = exchange.recv(1, PEER_SLOT_FFN)?;
                let routed_bf16_bytes = rows * LATENT_H * 2;
                self.peer_push_two_planes(
                    0,
                    PEER_SLOT_FFN,
                    (dst1, self.ranks[0].waves[wave_slot].shared_f32.buffer.ptr, shared_bytes),
                    (self.ranks[1].shared.routed_latent.buffer.ptr, self.ranks[0].shared.routed_latent.buffer.ptr, routed_bf16_bytes),
                )?;
            }
            exchange.push(1, PEER_SLOT_FFN, self.ranks[1].waves[wave_slot].shared_f32.buffer.ptr, shared_bytes)?;
        }
        exchange.wait(0, PEER_SLOT_FFN)?;
        if !rank0_only_consumer {
            exchange.wait(1, PEER_SLOT_FFN)?;
        }
        Ok(())
    }

    /// Same TP2 FFN exchange, but prefill sends BF16 partials (the native F32
    /// accumulator and post-reduction egress remain unchanged). Decode/verify
    /// retain their existing graph-stable F32 receive planes.
    fn exchange_ffn_prefill(&self, wave_slot: usize, rows: usize, is_dense: bool,
        reduce_routed: bool, rank0_only: bool) -> Result<()> {
        let exchange = self.peer_exchange()?;
        let shared_bytes = rows * H * 2;
        let routed_bytes = rows * LATENT_H * 2;
        for (rank, r) in self.ranks.iter().enumerate() {
            let (src, dst) = if is_dense { (r.shared.delta_f32.buffer, r.shared.delta.buffer) }
                else { (r.waves[wave_slot].shared_f32.buffer, r.waves[wave_slot].shared_out.buffer) };
            r.device.run(|| {
                r.narrow_peer_partial(rows, H, src, dst)?;
                if !is_dense && reduce_routed {
                    r.narrow_peer_partial(rows, LATENT_H, routed_f32_input(r, rows, self.skip_routed_experts)?,
                        r.shared.routed_latent.buffer)?;
                }
                Ok(())
            })?;
            if rank == 0 && rank0_only { continue; }
            if is_dense || (!reduce_routed && rank == 1) {
                exchange.push(rank, PEER_SLOT_FFN, dst.ptr, shared_bytes)?;
            } else {
                let recv = exchange.recv(1 - rank, PEER_SLOT_FFN)?;
                let routed_dst = if reduce_routed { recv.wrapping_byte_add(self.ffn_routed_recv_offset) }
                    else { self.ranks[1].shared.routed_latent.buffer.ptr };
                self.peer_push_two_planes(rank, PEER_SLOT_FFN,
                    (recv, dst.ptr, shared_bytes),
                    (routed_dst, r.shared.routed_latent.buffer.ptr, routed_bytes))?;
            }
        }
        exchange.wait(0, PEER_SLOT_FFN)?;
        if !rank0_only { exchange.wait(1, PEER_SLOT_FFN)?; }
        Ok(())
    }

    /// Stage metadata, Engram lookups, and token embeddings for a multi-sequence unit.
    #[allow(clippy::too_many_arguments)]
    fn begin_batch(
        &mut self,
        placements: &[Hy4Placement],
        initial_starts: &[usize],
        staged_frontiers: &mut [usize],
        staged_engram_tails: &mut [[u32; 2]],
        sources: &[&[u32]],
        unit: &Unit,
        priority: ReadPriority,
        head_mode: HeadMode,
    ) -> Result<ActiveBatch> {
        let wave_slot = unit.slot;
        ensure!(wave_slot < self.prefill_waves, "Hy4 unit wave slot {wave_slot} >= {}", self.prefill_waves);
        let rows = unit.rows.len();
        ensure!(rows > 0 && rows <= self.max_rows, "Hy4 unit rows {rows} outside 1..={}", self.max_rows);
        ensure!(priority != ReadPriority::Decode || rows <= self.decode_rows,
            "Hy4 decode/verify uses {rows} rows, exceeds admitted {}-row split-K owner", self.decode_rows);
        ensure!(!unit.segments.is_empty(), "Hy4 unit must carry explicit segments");
        ensure!(unit.segments.len() <= self.max_sequences, "Hy4 batch exceeds max_sequences {}", self.max_sequences);

        if let Some(draft) = &self.draft {
            for segment in &unit.segments {
                if let Some(p) = placements.get(segment.sequence) { draft.invalidate(p.slot); }
            }
        }

        let swa_ring_pages = self.ranks[0].cache.swa_ring_pages;
        let swa_ring_rows = self.ranks[0].cache.swa_ring_rows;

        let mut flat_tokens = Vec::with_capacity(rows);
        // Rope coordinates are [rows, 3] (T, H, W) interleaved M-RoPE; KV and SWA
        // slots, indexer limits and group ids stay slot-based (`pos`).
        let mut pos_i64 = Vec::with_capacity(rows * 24);
        let mut media_rows: Vec<(usize, usize, usize)> = Vec::new();
        let mut pos_i32 = Vec::with_capacity(rows * 4);
        let mut dsa_slots_i64 = Vec::with_capacity(rows * 8);
        let mut dsa_slots_ctx: [Vec<u8>; 2] = if self.context_split {
            [Vec::with_capacity(rows * 8), Vec::with_capacity(rows * 8)]
        } else {
            [Vec::new(), Vec::new()]
        };
        let mut swa_slots_i64 = Vec::with_capacity(rows * 8);
        let mut limits_bytes = Vec::with_capacity(rows * 4);
        let mut limits_ctx: [Vec<u8>; 2] = if self.context_split {
            [Vec::with_capacity(rows * 4), Vec::with_capacity(rows * 4)]
        } else {
            [Vec::new(), Vec::new()]
        };
        let mut swa_seqlens_bytes = Vec::with_capacity(unit.segments.len() * 4);
        let mut swa_cu_bytes = Vec::with_capacity((unit.segments.len() + 1) * 4);
        let mut g_pos_bytes = Vec::new();
        let mut g_slots_bytes = Vec::new();
        let mut g_slots_ctx: [Vec<u8>; 2] = [Vec::new(), Vec::new()];
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
            let (pos_start, pos_end, cur_tail, next_tail) = stage_segment_frontier_and_tail(
                seg.sequence,
                seg.tokens.start,
                tok_slice,
                staged_frontiers,
                staged_engram_tails,
            )?;
            let n = tok_slice.len();
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

            if p.media.as_ref().is_some_and(|media| media.needed(pos_start, pos_start + n).next().is_some()) {
                media_rows.push((seg.sequence, pos_start, row_cursor));
            }
            for (offset, _) in tok_slice.iter().enumerate() {
                let pos = pos_start + offset;
                let logical_page = pos / PAGE_ROWS;
                let local_row = pos % PAGE_ROWS;
                let raw_page = p.units[logical_page] as usize;
                let swa_slot = p.slot * swa_ring_rows + (pos % swa_ring_rows);
                for axis in p.rope_at(pos)? {
                    pos_i64.extend_from_slice(&axis.to_le_bytes());
                }
                pos_i32.extend_from_slice(&(pos as i32).to_le_bytes());
                swa_slots_i64.extend_from_slice(&(swa_slot as i64).to_le_bytes());
                if self.context_split {
                    let owner = context::page_owner(logical_page);
                    ensure!(raw_page & 1 == owner, "Hy4 ContextSplit logical page {logical_page} has wrong parity id {raw_page}");
                    let local_slot = ((raw_page >> 1) * PAGE_ROWS + local_row) as i64;
                    for rank in 0..2 {
                        let slot_val = if rank == owner { local_slot } else { -1i64 };
                        dsa_slots_ctx[rank].extend_from_slice(&slot_val.to_le_bytes());
                        limits_ctx[rank].extend_from_slice(&(local_complete_groups(pos, rank) as i32).to_le_bytes());
                    }
                } else {
                    let dsa_slot = raw_page * PAGE_ROWS + local_row;
                    dsa_slots_i64.extend_from_slice(&(dsa_slot as i64).to_le_bytes());
                    limits_bytes.extend_from_slice(&((pos / 4) as i32).to_le_bytes());
                }
            }

            let pending = pos_start % 4;
            let complete = (pending + n) / 4;
            if complete > 0 {
                let first_gid = (pos_start - pending) / 4;
                for g in 0..complete {
                    let gid = first_gid + g;
                    let logical_page = gid / (PAGE_ROWS / 4);
                    let local_group = gid % (PAGE_ROWS / 4);
                    let raw_page = p.units[logical_page] as usize;
                    // A pooled group carries the rope phase of its first token slot.
                    for axis in p.rope_at(gid * 4)? {
                        g_pos_bytes.extend_from_slice(&axis.to_le_bytes());
                    }
                    if self.context_split {
                        let owner = context::page_owner(logical_page);
                        ensure!(raw_page & 1 == owner, "Hy4 ContextSplit group {gid} page {logical_page} has wrong parity id {raw_page}");
                        let local_group_slot = ((raw_page >> 1) * (PAGE_ROWS / 4) + local_group) as i64;
                        for rank in 0..2 {
                            let val = if rank == owner { local_group_slot } else { -1i64 };
                            g_slots_ctx[rank].extend_from_slice(&val.to_le_bytes());
                        }
                    } else {
                        let phys_group_slot = raw_page * (PAGE_ROWS / 4) + local_group;
                        g_slots_bytes.extend_from_slice(&(phys_group_slot as i64).to_le_bytes());
                    }
                }
            }

            engram_members.push((cur_tail, tok_slice));
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
            && wave_slot == 0
            && rows <= self.decode_rows
            && self.graphs_active()
            && !self.skip_routed_experts
            && self.dump_dir.is_none();
        let all_single_row = members.len() == rows && rows <= self.decode_rows;
        let pt_stride = max_seq_end.div_ceil(PAGE_ROWS).max(1);
        let ctx_pt_stride = if self.context_split { pt_stride.div_ceil(2).max(1) } else { pt_stride };
        let swa_pt_stride = if graphed && all_single_row {
            self.max_seq_pages
        } else {
            pt_stride
        };
        let mut dsa_pt_bytes = if self.context_split { Vec::new() } else { Vec::with_capacity(rows * pt_stride * 4) };
        let mut dsa_pt_ctx: [Vec<u8>; 2] = if self.context_split {
            [Vec::with_capacity(rows * ctx_pt_stride * 4), Vec::with_capacity(rows * ctx_pt_stride * 4)]
        } else {
            [Vec::new(), Vec::new()]
        };
        let swa_pt_slice_bytes: Vec<u8> = if graphed && all_single_row {
            let need_ints = if members.len() == 1 {
                pt_stride
            } else {
                (members.len() - 1) * swa_pt_stride + pt_stride
            };
            for (b, m) in members.iter().enumerate() {
                let p = &placements[m.sequence];
                let seq_pages = (m.pos_start + m.row_len).div_ceil(PAGE_ROWS);
                let swa_row = &mut self.swa_pt_host[b * swa_pt_stride..b * swa_pt_stride + pt_stride];
                for lp in 0..pt_stride {
                    swa_row[lp] = (m.slot * swa_ring_pages + (lp % swa_ring_pages)) as i32;
                }
                if self.context_split {
                    for rank in 0..2 {
                        let mut row_pt = Vec::with_capacity(ctx_pt_stride * 4);
                        for local_p in 0..ctx_pt_stride {
                            let lp = local_p * 2 + rank;
                            let phys = if lp < seq_pages { (p.units[lp] >> 1) as i32 } else { 0i32 };
                            row_pt.extend_from_slice(&phys.to_le_bytes());
                        }
                        for _ in 0..m.row_len {
                            dsa_pt_ctx[rank].extend_from_slice(&row_pt);
                        }
                    }
                } else {
                    let mut row_pt = Vec::with_capacity(pt_stride * 4);
                    for lp in 0..pt_stride {
                        let phys = if lp < seq_pages { p.units[lp] as i32 } else { 0i32 };
                        row_pt.extend_from_slice(&phys.to_le_bytes());
                    }
                    for _ in 0..m.row_len {
                        dsa_pt_bytes.extend_from_slice(&row_pt);
                    }
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
                for lp in 0..pt_stride {
                    let swa_phys = (m.slot * swa_ring_pages + (lp % swa_ring_pages)) as i32;
                    swa_pt_bytes.extend_from_slice(&swa_phys.to_le_bytes());
                }
                if self.context_split {
                    for rank in 0..2 {
                        let mut row_pt = Vec::with_capacity(ctx_pt_stride * 4);
                        for local_p in 0..ctx_pt_stride {
                            let lp = local_p * 2 + rank;
                            let phys = if lp < seq_pages { (p.units[lp] >> 1) as i32 } else { 0i32 };
                            row_pt.extend_from_slice(&phys.to_le_bytes());
                        }
                        for _ in 0..m.row_len {
                            dsa_pt_ctx[rank].extend_from_slice(&row_pt);
                        }
                    }
                } else {
                    let mut row_pt = Vec::with_capacity(pt_stride * 4);
                    for lp in 0..pt_stride {
                        let phys = if lp < seq_pages { p.units[lp] as i32 } else { 0i32 };
                        row_pt.extend_from_slice(&phys.to_le_bytes());
                    }
                    for _ in 0..m.row_len {
                        dsa_pt_bytes.extend_from_slice(&row_pt);
                    }
                }
            }
            swa_pt_bytes
        };
        let cu_singleton: Vec<u8> = (0..=rows).flat_map(|i| (i as i32).to_le_bytes()).collect();

        // Issue background FP8 Engram gather + BF16 decode across all Engram layers in this unit
        // BEFORE launching embed/layer 0 so host I/O overlaps with GPU execution.
        let engram_wave = self.engram.begin_wave(&unit.layers, &engram_members, priority)?;

        if self.context_split {
            for rank in 0..2 {
                self.metadata_uploads[rank][wave_slot].upload(&self.ranks[rank..rank + 1], wave_slot, [
                    &pos_i64, &pos_i32, &dsa_slots_ctx[rank], &swa_slots_i64,
                    &dsa_pt_ctx[rank], &swa_pt_slice_bytes, &cu_singleton, &swa_cu_bytes,
                    &swa_seqlens_bytes, &limits_ctx[rank], &g_pos_bytes, &g_slots_ctx[rank],
                ])?;
            }
        } else {
            self.metadata_uploads[0][wave_slot].upload(&self.ranks, wave_slot, [
                &pos_i64, &pos_i32, &dsa_slots_i64, &swa_slots_i64,
                &dsa_pt_bytes, &swa_pt_slice_bytes, &cu_singleton, &swa_cu_bytes,
                &swa_seqlens_bytes, &limits_bytes, &g_pos_bytes, &g_slots_bytes,
            ])?;
        }

        // Embed tokens on rank 0 into waves[wave_slot].r_a [rows, 4, H] BF16, replicate GPU-to-GPU to rank 1 if TP2.
        {
            let r0 = &self.ranks[0];
            let emb = r0.weights.embedding.as_ref().context("rank 0 missing TokenEmbedding")?;
            // Image rows: the gathered placeholder embedding is replaced on all four
            // iHC streams by the tower's cached BF16 features, as in SGLang's
            // prepare_mm_inputs before the stream expansion. r_b and y are dead
            // until layer 0 writes them, so they stage the feature rows and indices.
            let mut chunk = cuteafd_engine::media::MediaChunk::default();
            for &(sequence, start, row) in &media_rows {
                let p = &placements[sequence];
                let media = p.media.as_ref().context("Hy4 media rows without request media")?;
                let n = members.iter().find(|m| m.row_start == row).context("Hy4 media member")?.row_len;
                let mut part = cuteafd_engine::media::MediaChunk::default();
                media.write_chunk(start, start + n, &mut part)?;
                chunk.indices.extend(part.indices.iter().map(|&index| index + row as u32));
                chunk.features.extend(part.features);
            }
            let wave = &r0.waves[wave_slot];
            r0.device.run(|| {
                emb.embed(&flat_tokens, r0.shared.ids.buffer, HC, wave.r_a.buffer, r0.stream.raw)?;
                if !chunk.indices.is_empty() {
                    r0.device.library.embedding_injection()?.inject_host(&chunk.features, &chunk.indices,
                        wave.r_b.buffer, wave.y.buffer, wave.r_a.buffer, rows, H, HC, r0.stream.raw)?;
                }
                Ok(())
            })?;
        }
        if self.ranks.len() == 2 {
            let bytes = rows * HC * H * 2;
            let exchange = self.peer_exchange()?;
            exchange.push_to(
                0,
                DIRECT,
                self.ranks[0].waves[wave_slot].r_a.buffer.ptr,
                self.ranks[1].waves[wave_slot].r_a.buffer.ptr,
                bytes,
            )?;
            exchange.wait(1, DIRECT)?;
        }

        let dump_rows = if self.dump_dir.is_some() && members.len() == 1
            && (members[0].pos_start == 0 || priority == ReadPriority::Decode) {
            rows.min(16)
        } else {
            0
        };
        if dump_rows > 0 {
            let dir = self.dump_dir.as_ref().unwrap();
            std::fs::create_dir_all(dir)?;
            let tok_bytes: Vec<u8> = flat_tokens[..dump_rows].iter().flat_map(|t| t.to_le_bytes()).collect();
            std::fs::write(dir.join("tokens.u32"), tok_bytes)?;
            self.dump_bf16("embed", self.ranks[0].waves[wave_slot].r_a.buffer, dump_rows * HC * H)?;
        }
        let final_layer = unit.layers.end.saturating_sub(1);
        let rank0_only_tail = self.ranks.len() == 2
            && unit.layers.end == self.cfg.layers
            && head_mode != HeadMode::None
            && dump_rows == 0;

        Ok(ActiveBatch {
            wave_slot,
            rows,
            max_seq_end,
            pt_stride,
            ctx_pt_stride,
            ctx_slot_counter: 0,
            swa_pt_stride,
            all_single_row,
            graphed,
            ffn_fused_layer: None,
            post_attn_deferred: false,
            reduce_attn_pending: false,
            prev_ffn_deferred: None,
            final_layer,
            rank0_only_tail,
            decode_attention: priority == ReadPriority::Decode,
            usage_phase: match priority {
                ReadPriority::Decode | ReadPriority::DecodeFirst => UsagePhase::Decode,
                ReadPriority::Prefill | ReadPriority::PrefillFirst => UsagePhase::Prefill,
            },
            dump_rows,
            use_a: true,
            members,
            engram_wave,
        })
    }

    /// 2-GPU ContextSplit DSA body after `DsaPre` has written parity-owned KV rows,
    /// absorbed each rank's 16-head query half, and projected `idx_tk`.
    /// Runs group pooling, query exchange, scored top-k + candidate exchange + owned merge,
    /// dual `h0`+`h16` partials on both ranks, partial exchange, optional DFly2 dual recall
    /// capture, `lse_combine2(heads=16, has_sink=True)`, sink-free `mla_gate_uv_dsa`, and `o_proj`.
    fn run_dsa_context_split_body(
        &self,
        layer: usize,
        dsa_idx: usize,
        p: &str,
        batch: &mut ActiveBatch,
    ) -> Result<()> {
        let ctx_ex = self
            .context_exchange
            .as_ref()
            .context("Hy4 ContextSplit missing ContextExchange")?;
        let wave_slot = batch.wave_slot;
        let rows = batch.rows;
        let ctx_pt_stride = batch.ctx_pt_stride;
        let ip = format!("{p}.self_attn.compressed_indexer");
        let kv_caches = [
            self.ranks[0].cache.dsa_kv_layer(dsa_idx)?,
            self.ranks[1].cache.dsa_kv_layer(dsa_idx)?,
        ];
        let idx_caches = [
            self.ranks[0].cache.dsa_index_layer(dsa_idx)?,
            self.ranks[1].cache.dsa_index_layer(dsa_idx)?,
        ];

        // 1. Pool completed 4-token groups into each rank's parity-owned `dsa_index` (skips slot < 0).
        for (rank, r) in self.ranks.iter().enumerate() {
            let w = &r.waves[wave_slot];
            let idx_cache = idx_caches[rank];
            let inv_freq = r.weights.inv_freq_dsa.buffer.ptr;
            // SAFETY: key scratch, provisional keys and parity group cache stay owned through stream drain.
            r.device.run(|| unsafe {
                if self.verifying {
                    let keys = self.verify_keys.as_ref().context("Hy4 missing provisional keys")?;
                    let dst = slice_buf(keys[rank].buffer, dsa_idx * self.decode_rows * IDX_KEY_BYTES, rows * IDX_KEY_BYTES)?;
                    r.device.library.copy_d2d_async(dst, r.shared.idx_tk.buffer, dst.bytes, r.stream.raw)?;
                }
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
                                ptr: r.shared.idx_group_tok.buffer.ptr.cast::<u8>().wrapping_add(group_dst_off).cast(),
                                bytes: pending * IDX_DIM * 2,
                                ..r.shared.idx_group_tok.buffer
                            };
                            r.device.library.copy_d2d_async(dst_p, pending_buf, pending * IDX_DIM * 2, r.stream.raw)?;
                        }
                        let copy_from_cur = complete * 4 - pending;
                        let dst_cur = CuteafdDeviceBuffer {
                            ptr: r.shared.idx_group_tok.buffer.ptr.cast::<u8>().wrapping_add(group_dst_off + pending * IDX_DIM * 2).cast(),
                            bytes: copy_from_cur * IDX_DIM * 2,
                            ..r.shared.idx_group_tok.buffer
                        };
                        let src_cur = CuteafdDeviceBuffer {
                            ptr: r.shared.idx_tk.buffer.ptr.cast::<u8>().wrapping_add(tok_src_base).cast(),
                            bytes: copy_from_cur * IDX_DIM * 2,
                            ..r.shared.idx_tk.buffer
                        };
                        r.device.library.copy_d2d_async(dst_cur, src_cur, copy_from_cur * IDX_DIM * 2, r.stream.raw)?;
                        batch_groups += complete;
                    }
                    if !self.verifying && new_pending > 0 {
                        if complete == 0 {
                            let dst = CuteafdDeviceBuffer {
                                ptr: pending_buf.ptr.cast::<u8>().wrapping_add(pending * IDX_DIM * 2).cast(),
                                bytes: m.row_len * IDX_DIM * 2,
                                ..pending_buf
                            };
                            let src = CuteafdDeviceBuffer {
                                ptr: r.shared.idx_tk.buffer.ptr.cast::<u8>().wrapping_add(tok_src_base).cast(),
                                bytes: m.row_len * IDX_DIM * 2,
                                ..r.shared.idx_tk.buffer
                            };
                            r.device.library.copy_d2d_async(dst, src, m.row_len * IDX_DIM * 2, r.stream.raw)?;
                        } else {
                            let tail_off = tok_src_base + (complete * 4 - pending) * IDX_DIM * 2;
                            let src = CuteafdDeviceBuffer {
                                ptr: r.shared.idx_tk.buffer.ptr.cast::<u8>().wrapping_add(tail_off).cast(),
                                bytes: new_pending * IDX_DIM * 2,
                                ..r.shared.idx_tk.buffer
                            };
                            r.device.library.copy_d2d_async(pending_buf, src, new_pending * IDX_DIM * 2, r.stream.raw)?;
                        }
                    }
                    if !self.verifying {
                        r.cache.pending_counts.borrow_mut()[m.slot][dsa_idx] = new_pending;
                    }
                }
                if batch_groups > 0 {
                    r.programs.indexer_pool_group.launch(
                        &[
                            r.shared.idx_group_tok.buffer.ptr,
                            r.weights.weight(&format!("{ip}.key_value_projection"))?.ptr,
                            r.weights.weight(&format!("{ip}.key_gate_projection"))?.ptr,
                            r.weights.buf(&format!("{ip}.intra_group_ape"))?.ptr,
                            r.weights.buf(&format!("{ip}.key_norm.weight"))?.ptr,
                            w.idx_group_pos.buffer.ptr,
                            w.idx_group_slots.buffer.ptr,
                            inv_freq,
                            idx_cache.ptr,
                        ],
                        &[Scalar::I32(batch_groups as i32)],
                        r.stream.raw,
                    )?;
                }
                Ok(())
            })?;
        }

        // 2. Indexer query projection + RoPE when any row exceeds 2048 tokens.
        let need_score = batch.max_seq_end > CANDIDATE_TOPK;
        if need_score {
            for r in &self.ranks {
                if batch.graphed {
                    r.graphs.run(
                        GraphKey { layer, rows, section: GraphSection::DsaIndexQ },
                        &r.stream,
                        || r.device.run(|| r.dsa_index_q(p, wave_slot, rows)),
                    )?;
                } else {
                    r.device.run(|| r.dsa_index_q(p, wave_slot, rows))?;
                }
            }
        }

        let write_draft = i32::from(self.draft.as_ref().is_some_and(|d| d.captures_layer(layer)));
        let query_half_bytes = 16 * KV_A_DIM * 2;
        let ctx_chunk = context_chunk_rows(self.max_rows, self.decode_rows, batch.usage_phase == UsagePhase::Decode);

        // 3. Chunked ContextSplit exchange + dual partials + combine (1 pass on decode/verify,
        // otherwise one pass per admitted prefill tile; MLA internally uses 64-row scratch).
        for chunk_start in (0..rows).step_by(ctx_chunk) {
            let c_rows = (rows - chunk_start).min(ctx_chunk);
            let slot = if batch.graphed {
                context::slot(dsa_idx, 0)
            } else {
                let s = batch.ctx_slot_counter % 2;
                batch.ctx_slot_counter += 1;
                s
            };

            // (a) Query exchange: each rank pushes its 16-head absorbed query half -> assembled [c_rows, 32, 576].
            for rank in 0..2 {
                let q_ptr = self.ranks[rank]
                    .shared
                    .query
                    .buffer
                    .ptr
                    .cast::<u8>()
                    .wrapping_add(chunk_start * query_half_bytes)
                    .cast();
                ctx_ex.push_query(rank, slot, q_ptr, query_half_bytes, c_rows)?;
            }
            let assembled_q = [ctx_ex.wait_query(0, slot)?, ctx_ex.wait_query(1, slot)?];

            // (b) Scored top-511 candidates + candidate exchange (skipped when max_seq_end <= 2048).
            let (cand_s, cand_i) = if need_score {
                for rank in 0..2 {
                    let r = &self.ranks[rank];
                    let ctx_progs = r.programs.context.as_ref().context("Hy4 ContextSplit programs missing")?;
                    let ctx = r.shared.ctx.as_ref().context("Hy4 ContextSplit scratch missing")?;
                    let w = &r.waves[wave_slot];
                    let max_lg = local_complete_groups(batch.max_seq_end.saturating_sub(1), rank).max(1);
                    let tile_cap = (r.shared.idx_score_floats / max_lg).max(1);
                    // SAFETY: all operand slices lie inside admitted rank/wave arenas.
                    r.device.run(|| unsafe {
                        for sub in (0..c_rows).step_by(tile_cap) {
                            let t_rows = (c_rows - sub).min(tile_cap);
                            let abs_row = chunk_start + sub;
                            let q_rot_ptr = r.shared.idx_q_rot.buffer.ptr.cast::<u8>().wrapping_add(abs_row * IDX_HEADS * IDX_DIM * 2).cast();
                            let w_out_ptr = r.shared.idx_w_out.buffer.ptr.cast::<u8>().wrapping_add(abs_row * IDX_HEADS * 4).cast();
                            let pt_ptr = w.dsa_page_table.buffer.ptr.cast::<u8>().wrapping_add(abs_row * ctx_pt_stride * 4).cast();
                            let lim_ptr = w.idx_limits.buffer.ptr.cast::<u8>().wrapping_add(abs_row * 4).cast();
                            let pos_ptr = w.positions_i32.buffer.ptr.cast::<u8>().wrapping_add(abs_row * 4).cast();
                            let cs_ptr = ctx.cand_scores.buffer.ptr.cast::<u8>().wrapping_add(sub * GROUP_TOPK * 4).cast();
                            let ci_ptr = ctx.cand_indices.buffer.ptr.cast::<u8>().wrapping_add(sub * GROUP_TOPK * 4).cast();
                            r.programs.indexer_score_groups.launch(
                                &[q_rot_ptr, w_out_ptr, idx_caches[rank].ptr, pt_ptr, lim_ptr, r.shared.idx_scores.buffer.ptr],
                                &[Scalar::I32(t_rows as i32), Scalar::I32(max_lg as i32), Scalar::I32(ctx_pt_stride as i32)],
                                r.stream.raw,
                            )?;
                            ctx_progs.indexer_scored_topk.launch(
                                &[r.shared.idx_scores.buffer.ptr, pos_ptr, lim_ptr, cs_ptr, ci_ptr],
                                &[
                                    Scalar::I32(t_rows as i32),
                                    Scalar::I32(max_lg as i32),
                                    Scalar::I32(2),
                                    Scalar::I32(rank as i32),
                                ],
                                r.stream.raw,
                            )?;
                        }
                        Ok(())
                    })?;
                    ctx_ex.push_candidates(
                        rank,
                        slot,
                        ctx.cand_scores.buffer.ptr,
                        ctx.cand_indices.buffer.ptr,
                        c_rows,
                    )?;
                }
                let (s0, i0) = ctx_ex.wait_candidates(0, slot)?;
                let (s1, i1) = ctx_ex.wait_candidates(1, slot)?;
                ([s0, s1], [i0, i1])
            } else {
                let b0 = ctx_ex.buffers(0, slot)?;
                let b1 = ctx_ex.buffers(1, slot)?;
                ([b0.scores, b1.scores], [b0.indices, b1.indices])
            };

            // (c) Bitonic merge + owned slot expansion (plus sparse global draft_slots when tapped).
            for rank in 0..2 {
                let r = &self.ranks[rank];
                let ctx_progs = r.programs.context.as_ref().context("Hy4 ContextSplit programs missing")?;
                let ctx = r.shared.ctx.as_ref().context("Hy4 ContextSplit scratch missing")?;
                let w = &r.waves[wave_slot];
                let pos_ptr = w.positions_i32.buffer.ptr.cast::<u8>().wrapping_add(chunk_start * 4).cast();
                let pt_ptr = w.dsa_page_table.buffer.ptr.cast::<u8>().wrapping_add(chunk_start * ctx_pt_stride * 4).cast();
                // SAFETY: ContextExchange::wait_candidates ordered cand_s/cand_i; outputs live in rank scratch.
                r.device.run(|| unsafe {
                    ctx_progs.indexer_merge_expand_owned.launch(
                        &[
                            cand_s[rank],
                            cand_i[rank],
                            pos_ptr,
                            pt_ptr,
                            r.shared.idx_cand_slots.buffer.ptr,
                            r.shared.idx_cand_lens.buffer.ptr,
                            ctx.draft_slots.buffer.ptr,
                        ],
                        &[
                            Scalar::I32(c_rows as i32),
                            Scalar::I32(ctx_pt_stride as i32),
                            Scalar::I32(rank as i32),
                            Scalar::I32(write_draft),
                        ],
                        r.stream.raw,
                    )
                })?;
            }

            // (d) Dual h0 + h16 partials over each rank's parity-owned local pages (32 heads total/rank).
            for rank in 0..2 {
                let r = &self.ranks[rank];
                let q_ptr = assembled_q[rank];
                let kv_buf = kv_caches[rank];
                let decode_attn = batch.decode_attention;
                if batch.graphed {
                    r.graphs.run(
                        GraphKey { layer, rows, section: GraphSection::DsaPartial },
                        &r.stream,
                        || r.device.run(|| r.dsa_attn_context_partials(rank, q_ptr, kv_buf, wave_slot, c_rows, decode_attn)),
                    )?;
                } else {
                    r.device.run(|| r.dsa_attn_context_partials(rank, q_ptr, kv_buf, wave_slot, c_rows, decode_attn))?;
                }
            }

            // (e) Partial exchange: push peer's 16-head partial + base-2 LSE, wait for our 16-head partial.
            for rank in 0..2 {
                let ctx = self.ranks[rank].shared.ctx.as_ref().context("Hy4 ContextSplit scratch missing")?;
                ctx_ex.push_partial(
                    rank,
                    slot,
                    ctx.partial_peer.buffer.ptr,
                    ctx.lse_peer.buffer.ptr,
                    c_rows,
                )?;
            }
            let (recv_p0, recv_l0) = ctx_ex.wait_partial(0, slot)?;
            let (recv_p1, recv_l1) = ctx_ex.wait_partial(1, slot)?;
            let recv_p = [recv_p0, recv_p1];
            let recv_l = [recv_l0, recv_l1];

            // (f) DFly2 dual recall capture on participating drafter ranks: wait_partial(rank, slot)
            // guarantees the peer's merge_expand_owned already populated its ctx.draft_slots.
            if write_draft != 0 {
                if let Some(draft) = &self.draft {
                    let ctx0 = self.ranks[0].shared.ctx.as_ref().context("Hy4 rank 0 ContextSplit scratch missing")?;
                    let ctx1 = self.ranks[1].shared.ctx.as_ref().context("Hy4 rank 1 ContextSplit scratch missing")?;
                    let chunk_end = chunk_start + c_rows;
                    for rank in 0..draft.tp() {
                        let r = &self.ranks[rank];
                        let (local_ctx, peer_ctx) = if rank == 0 { (ctx0, ctx1) } else { (ctx1, ctx0) };
                        r.device.run(|| {
                            for member in &batch.members {
                                let end = member.row_start + member.row_len;
                                let start = if self.verifying { member.row_start } else { end - 1 };
                                for row in start.max(chunk_start)..end.min(chunk_end) {
                                    let local_row = row - chunk_start;
                                    let dst_row = if self.verifying { row } else { member.slot };
                                    let src_local = slice_buf(local_ctx.draft_slots.buffer, local_row * CANDIDATE_TOPK * 4, CANDIDATE_TOPK * 4)?;
                                    let src_peer = slice_buf(peer_ctx.draft_slots.buffer, local_row * CANDIDATE_TOPK * 4, CANDIDATE_TOPK * 4)?;
                                    draft.capture_recall(rank, self.verifying, dst_row, layer, src_local, &r.stream)?;
                                    draft.capture_recall_peer(rank, self.verifying, dst_row, layer, src_peer, &r.stream)?;
                                }
                            }
                            Ok(())
                        })?;
                    }
                }
            }

            // (g) Combine GPU0-then-GPU1 partials with natural-log sink; fuse gate_uv + o_proj when graphed.
            for rank in 0..2 {
                let r = &self.ranks[rank];
                let rp = recv_p[rank];
                let rl = recv_l[rank];
                if batch.graphed {
                    r.graphs.run(
                        GraphKey { layer, rows, section: GraphSection::DsaAttnOut },
                        &r.stream,
                        || {
                            r.device.run(|| {
                                r.dsa_context_combine(rank, p, rp, rl, 0, rows)?;
                                r.dsa_context_gate_uv_o(p, layer, 16, rows)
                            })
                        },
                    )?;
                } else {
                    r.device.run(|| r.dsa_context_combine(rank, p, rp, rl, chunk_start, c_rows))?;
                }
            }
        }

        if !batch.graphed {
            for r in &self.ranks {
                r.device.run(|| r.dsa_context_gate_uv_o(p, layer, 16, rows))?;
            }
        }
        Ok(())
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
        let wave_slot = batch.wave_slot;

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
                                    destination: r0.waves[wave_slot].engram_raw[slot].buffer,
                                    offset: 0,
                                    bytes: raw_bytes,
                                },
                                CopyPlane {
                                    destination: r0.waves[wave_slot].engram_masked[slot].buffer,
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
            }
            if gpus == 2 {
                let aligned_masked = masked_bytes.div_ceil(16) * 16;
                self.peer_push_two_planes(
                    0,
                    DIRECT,
                    (
                        self.ranks[1].waves[wave_slot].engram_raw[slot].buffer.ptr,
                        self.ranks[0].waves[wave_slot].engram_raw[slot].buffer.ptr,
                        raw_bytes,
                    ),
                    (
                        self.ranks[1].waves[wave_slot].engram_masked[slot].buffer.ptr,
                        self.ranks[0].waves[wave_slot].engram_masked[slot].buffer.ptr,
                        aligned_masked,
                    ),
                )?;
                self.peer_exchange()?.wait(1, DIRECT)?;
            }
        }

        // Eager indexer group-pool + score/topk helper shared by graphed & eager DSA paths.
        let run_dsa_indexer_eager = |r: &RankRuntime<'a>,
                                     dsa_idx: usize,
                                     max_groups: usize,
                                     pt_stride: usize,
                                     graphed: bool|
         -> Result<()> {
            let w = &r.waves[wave_slot];
            let idx_cache = r.cache.dsa_index_layer(dsa_idx)?;
            let ip = format!("{p}.self_attn.compressed_indexer");
            let inv_freq = r.weights.inv_freq_dsa.buffer.ptr;
            // SAFETY: key scratch, provisional keys and the packed group's cache
            // entries all remain owned across the stream's writes.
            r.device.run(|| unsafe {
                if self.verifying {
                    let keys = self.verify_keys.as_ref().context("Hy4 missing provisional keys")?;
                    let rank = usize::from(r.device.id != self.ranks[0].device.id);
                    let dst = slice_buf(keys[rank].buffer, dsa_idx * self.decode_rows * IDX_KEY_BYTES, rows * IDX_KEY_BYTES)?;
                    r.device.library.copy_d2d_async(dst, r.shared.idx_tk.buffer, dst.bytes, r.stream.raw)?;
                }
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
                                ptr: r.shared.idx_group_tok.buffer.ptr.cast::<u8>().wrapping_add(group_dst_off).cast(),
                                bytes: pending * IDX_DIM * 2,
                                ..r.shared.idx_group_tok.buffer
                            };
                            r.device.library.copy_d2d_async(dst_p, pending_buf, pending * IDX_DIM * 2, r.stream.raw)?;
                        }
                        let copy_from_cur = complete * 4 - pending;
                        let dst_cur = CuteafdDeviceBuffer {
                            ptr: r.shared.idx_group_tok.buffer.ptr.cast::<u8>().wrapping_add(group_dst_off + pending * IDX_DIM * 2).cast(),
                            bytes: copy_from_cur * IDX_DIM * 2,
                            ..r.shared.idx_group_tok.buffer
                        };
                        let src_cur = CuteafdDeviceBuffer {
                            ptr: r.shared.idx_tk.buffer.ptr.cast::<u8>().wrapping_add(tok_src_base).cast(),
                            bytes: copy_from_cur * IDX_DIM * 2,
                            ..r.shared.idx_tk.buffer
                        };
                        r.device.library.copy_d2d_async(dst_cur, src_cur, copy_from_cur * IDX_DIM * 2, r.stream.raw)?;
                        batch_groups += complete;
                    }
                    if !self.verifying && new_pending > 0 {
                        if complete == 0 {
                            let dst = CuteafdDeviceBuffer {
                                ptr: pending_buf.ptr.cast::<u8>().wrapping_add(pending * IDX_DIM * 2).cast(),
                                bytes: m.row_len * IDX_DIM * 2,
                                ..pending_buf
                            };
                            let src = CuteafdDeviceBuffer {
                                ptr: r.shared.idx_tk.buffer.ptr.cast::<u8>().wrapping_add(tok_src_base).cast(),
                                bytes: m.row_len * IDX_DIM * 2,
                                ..r.shared.idx_tk.buffer
                            };
                            r.device.library.copy_d2d_async(dst, src, m.row_len * IDX_DIM * 2, r.stream.raw)?;
                        } else {
                            let tail_off = tok_src_base + (complete * 4 - pending) * IDX_DIM * 2;
                            let src = CuteafdDeviceBuffer {
                                ptr: r.shared.idx_tk.buffer.ptr.cast::<u8>().wrapping_add(tail_off).cast(),
                                bytes: new_pending * IDX_DIM * 2,
                                ..r.shared.idx_tk.buffer
                            };
                            r.device.library.copy_d2d_async(pending_buf, src, new_pending * IDX_DIM * 2, r.stream.raw)?;
                        }
                    }
                    if !self.verifying {
                        r.cache.pending_counts.borrow_mut()[m.slot][dsa_idx] = new_pending;
                    }
                }
                if batch_groups > 0 {
                    r.programs.indexer_pool_group.launch(
                        &[
                            r.shared.idx_group_tok.buffer.ptr,
                            r.weights.weight(&format!("{ip}.key_value_projection"))?.ptr,
                            r.weights.weight(&format!("{ip}.key_gate_projection"))?.ptr,
                            r.weights.buf(&format!("{ip}.intra_group_ape"))?.ptr,
                            r.weights.buf(&format!("{ip}.key_norm.weight"))?.ptr,
                            w.idx_group_pos.buffer.ptr,
                            w.idx_group_slots.buffer.ptr,
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
                        || r.device.run(|| r.dsa_index_q(&p, wave_slot, rows)),
                    )?;
                } else {
                    r.device.run(|| r.dsa_index_q(&p, wave_slot, rows))?;
                }
                r.device.run(|| unsafe {
                    let tile_cap = (r.shared.idx_score_floats / max_groups).max(1);
                    for row_start in (0..rows).step_by(tile_cap) {
                        let tile_rows = (rows - row_start).min(tile_cap);
                        let q_rot_ptr = r.shared.idx_q_rot.buffer.ptr.cast::<u8>().wrapping_add(row_start * IDX_HEADS * IDX_DIM * 2).cast();
                        let w_out_ptr = r.shared.idx_w_out.buffer.ptr.cast::<u8>().wrapping_add(row_start * IDX_HEADS * 4).cast();
                        let pt_ptr = w.dsa_page_table.buffer.ptr.cast::<u8>().wrapping_add(row_start * pt_stride * 4).cast();
                        let lim_ptr = w.idx_limits.buffer.ptr.cast::<u8>().wrapping_add(row_start * 4).cast();
                        let pos_ptr = w.positions_i32.buffer.ptr.cast::<u8>().wrapping_add(row_start * 4).cast();
                        let cand_ptr = r.shared.idx_cand_slots.buffer.ptr.cast::<u8>().wrapping_add(row_start * CANDIDATE_TOPK * 4).cast();
                        let len_ptr = r.shared.idx_cand_lens.buffer.ptr.cast::<u8>().wrapping_add(row_start * 4).cast();
                        r.programs.indexer_score_groups.launch(
                            &[
                                q_rot_ptr,
                                w_out_ptr,
                                idx_cache.ptr,
                                pt_ptr,
                                lim_ptr,
                                r.shared.idx_scores.buffer.ptr,
                            ],
                            &[Scalar::I32(tile_rows as i32), Scalar::I32(max_groups as i32), Scalar::I32(pt_stride as i32)],
                            r.stream.raw,
                        )?;
                        r.programs.indexer_topk_expand.launch(
                            &[
                                r.shared.idx_scores.buffer.ptr,
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
                            r.shared.idx_scores.buffer.ptr,
                            w.positions_i32.buffer.ptr,
                            w.dsa_page_table.buffer.ptr,
                            r.shared.idx_cand_slots.buffer.ptr,
                            r.shared.idx_cand_lens.buffer.ptr,
                        ],
                        &[Scalar::I32(rows as i32), Scalar::I32(1), Scalar::I32(pt_stride as i32)],
                        r.stream.raw,
                    )
                })
            }?;
            if let Some(draft) = &self.draft {
                let rank_idx = usize::from(r.device.id != self.ranks[0].device.id);
                if rank_idx < draft.tp() {
                    r.device.run(|| {
                        for member in &batch.members {
                            let end = member.row_start + member.row_len;
                            let start = if self.verifying { member.row_start } else { end - 1 };
                            for row in start..end {
                                let src = slice_buf(r.shared.idx_cand_slots.buffer, row * CANDIDATE_TOPK * 4, CANDIDATE_TOPK * 4)?;
                                draft.capture_recall(rank_idx, self.verifying, if self.verifying { row } else { member.slot }, layer, src, &r.stream)?;
                            }
                        }
                        Ok(())
                    })?;
                }
            }
            Ok(())
        };

        let max_groups = batch.max_seq_end / 4;
        let pt_stride = batch.pt_stride;
        let swa_pt_stride = batch.swa_pt_stride;
        let members_len = batch.members.len();
        let all_single_row = batch.all_single_row;
        let decode_attention = batch.decode_attention;
        let routed_scale = self.cfg.routed_scaling_factor as f32;
        let skip_routed = self.skip_routed_experts;
        let usage = self.expert_usage.as_ref();
        let count_usage = self.count_expert_usage;
        let usage_phase = batch.usage_phase;
        let prev_ffn = batch.prev_ffn_deferred.take();
        let prev_use_a = batch.use_a;
        let use_a = if prev_ffn.is_some() {
            batch.use_a = !batch.use_a;
            batch.use_a
        } else {
            use_a
        };
        let ffn_recv_ptrs = if gpus == 2 {
            let ex = self.peer_exchange()?;
            [ex.recv(0, PEER_SLOT_FFN)?, ex.recv(1, PEER_SLOT_FFN)?]
        } else {
            [std::ptr::null_mut(); 2]
        };
        let routed_recv_offset = self.ffn_routed_recv_offset;

        // Graphed decode fast-path:
        //   Case 1 (TP1 + local FFN): fuse entire SWA layer (1 graph) or DSA pre + post (2 graphs).
        //   Case 2 (TP2 or Spark-routed): fuse previous layer's deferred FFN finish into this layer's
        //           rank-local attention section; PeerExchange push/wait stays outside capture.
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
                                        r.engram_compute(&p, slot, wave_slot, rows, use_a)?;
                                    }
                                    r.attn_pre(&p, layer, false, local_heads, wave_slot, rows, use_a, kv_cache_buf)?;
                                    r.swa_attn(kv_cache_buf, wave_slot, rows, members_len, swa_pt_stride, decode_attention)?;
                                    r.attn_out(&p, layer, false, 1, local_heads, rows)?;
                                    r.post_attn_ihc(&p, wave_slot, rows, use_a)?;
                                    if is_dense {
                                        r.dense_ffn(&p, 1, wave_slot, rows)?;
                                    } else {
                                        r.moe_router_shared(&p, 1, wave_slot, rows, routed_scale)?;
                                        r.local_exl3(layer, 1, rows, skip_routed)?;
                                        r.record_expert_usage_local(usage, count_usage, skip_routed, false, layer, usage_phase, rows)?;
                                    }
                                    r.ffn_combine_post(&p, is_dense, wave_slot, rows, !use_a)
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
                                        r.engram_compute(&p, slot, wave_slot, rows, use_a)?;
                                    }
                                    r.attn_pre(&p, layer, false, local_heads, wave_slot, rows, use_a, kv_cache_buf)
                                })
                            },
                        )?;
                        r.device.run(|| r.swa_attn(kv_cache_buf, wave_slot, rows, members_len, swa_pt_stride, decode_attention))?;
                        r.graphs.run(
                            GraphKey { layer, rows, section: GraphSection::SwaExtendPost },
                            &r.stream,
                            || {
                                r.device.run(|| {
                                    r.attn_out(&p, layer, false, 1, local_heads, rows)?;
                                    r.post_attn_ihc(&p, wave_slot, rows, use_a)?;
                                    if is_dense {
                                        r.dense_ffn(&p, 1, wave_slot, rows)?;
                                    } else {
                                        r.moe_router_shared(&p, 1, wave_slot, rows, routed_scale)?;
                                        r.local_exl3(layer, 1, rows, skip_routed)?;
                                        r.record_expert_usage_local(usage, count_usage, skip_routed, false, layer, usage_phase, rows)?;
                                    }
                                    r.ffn_combine_post(&p, is_dense, wave_slot, rows, !use_a)
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
                                    r.engram_compute(&p, slot, wave_slot, rows, use_a)?;
                                }
                                r.attn_pre(&p, layer, true, local_heads, wave_slot, rows, use_a, kv_cache_buf)?;
                                r.dsa_index_wk(&p, wave_slot, rows)
                            })
                        },
                    )?;
                    run_dsa_indexer_eager(r, dsa_idx, max_groups, pt_stride, true)?;
                    r.graphs.run(
                        GraphKey { layer, rows, section: GraphSection::DsaLocalPost },
                        &r.stream,
                        || {
                            r.device.run(|| {
                                r.dsa_attn(kv_cache_buf, wave_slot, rows, decode_attention)?;
                                r.attn_out(&p, layer, true, 1, local_heads, rows)?;
                                r.post_attn_ihc(&p, wave_slot, rows, use_a)?;
                                if is_dense {
                                    r.dense_ffn(&p, 1, wave_slot, rows)?;
                                } else {
                                    r.moe_router_shared(&p, 1, wave_slot, rows, routed_scale)?;
                                    r.local_exl3(layer, 1, rows, skip_routed)?;
                                    r.record_expert_usage_local(usage, count_usage, skip_routed, false, layer, usage_phase, rows)?;
                                }
                                r.ffn_combine_post(&p, is_dense, wave_slot, rows, !use_a)
                            })
                        },
                    )?;
                }
                // Residual ping-pong flipped twice inside the fused layer; skip separate local_ffn.
                batch.ffn_fused_layer = Some(layer);
                return Ok(());
            }

            // Case 2: TP2 or Spark-routed layer — rank-local attention sections, peer join outside capture.
            if !is_sparse {
                if all_single_row {
                    for (rank, r) in self.ranks.iter().enumerate() {
                        let kv_cache_buf = r.cache.swa_layer_bank(self.swa_index_of[&layer])?;
                        let ffn_recv = ffn_recv_ptrs[rank];
                        r.graphs.run(
                            GraphKey { layer, rows, section: GraphSection::SwaAttn },
                            &r.stream,
                            || {
                                r.device.run(|| {
                                    if let Some(prev) = prev_ffn {
                                        r.finish_deferred_ffn_local(
                                            rank, gpus, prev, wave_slot, rows, prev_use_a, skip_routed, ffn_recv, routed_recv_offset,
                                        )?;
                                    }
                                    if let Some(slot) = engram_slot {
                                        r.engram_compute(&p, slot, wave_slot, rows, use_a)?;
                                    }
                                    r.attn_pre(&p, layer, false, local_heads, wave_slot, rows, use_a, kv_cache_buf)?;
                                    r.swa_attn(kv_cache_buf, wave_slot, rows, members_len, swa_pt_stride, decode_attention)?;
                                    r.attn_out(&p, layer, false, gpus, local_heads, rows)
                                })
                            },
                        )?;
                    }
                } else {
                    for (rank, r) in self.ranks.iter().enumerate() {
                        let kv_cache_buf = r.cache.swa_layer_bank(self.swa_index_of[&layer])?;
                        let ffn_recv = ffn_recv_ptrs[rank];
                        r.graphs.run(
                            GraphKey { layer, rows, section: GraphSection::SwaExtendPre },
                            &r.stream,
                            || {
                                r.device.run(|| {
                                    if let Some(prev) = prev_ffn {
                                        r.finish_deferred_ffn_local(
                                            rank, gpus, prev, wave_slot, rows, prev_use_a, skip_routed, ffn_recv, routed_recv_offset,
                                        )?;
                                    }
                                    if let Some(slot) = engram_slot {
                                        r.engram_compute(&p, slot, wave_slot, rows, use_a)?;
                                    }
                                    r.attn_pre(&p, layer, false, local_heads, wave_slot, rows, use_a, kv_cache_buf)
                                })
                            },
                        )?;
                    }
                    for r in &self.ranks {
                        let kv_cache_buf = r.cache.swa_layer_bank(self.swa_index_of[&layer])?;
                        r.device.run(|| r.swa_attn(kv_cache_buf, wave_slot, rows, members_len, swa_pt_stride, decode_attention))?;
                    }
                    for r in &self.ranks {
                        r.graphs.run(
                            GraphKey { layer, rows, section: GraphSection::SwaExtendPost },
                            &r.stream,
                            || r.device.run(|| r.attn_out(&p, layer, false, gpus, local_heads, rows)),
                        )?;
                    }
                }
            } else {
                let dsa_idx = self.dsa_index_of[&layer];
                // Phase-by-phase across ranks so GPU 1 starts DsaPre before GPU 0 queues eager indexer D2D copies.
                for (rank, r) in self.ranks.iter().enumerate() {
                    let kv_cache_buf = r.cache.dsa_kv_layer(dsa_idx)?;
                    let ffn_recv = ffn_recv_ptrs[rank];
                    r.graphs.run(
                        GraphKey { layer, rows, section: GraphSection::DsaPre },
                        &r.stream,
                        || {
                            r.device.run(|| {
                                if let Some(prev) = prev_ffn {
                                    r.finish_deferred_ffn_local(
                                        rank, gpus, prev, wave_slot, rows, prev_use_a, skip_routed, ffn_recv, routed_recv_offset,
                                    )?;
                                }
                                if let Some(slot) = engram_slot {
                                    r.engram_compute(&p, slot, wave_slot, rows, use_a)?;
                                }
                                r.attn_pre(&p, layer, true, local_heads, wave_slot, rows, use_a, kv_cache_buf)?;
                                r.dsa_index_wk(&p, wave_slot, rows)
                            })
                        },
                    )?;
                }
                if self.context_split {
                    self.run_dsa_context_split_body(layer, dsa_idx, &p, batch)?;
                } else {
                    for r in &self.ranks {
                        run_dsa_indexer_eager(r, dsa_idx, max_groups, pt_stride, true)?;
                    }
                    for r in &self.ranks {
                        let kv_cache_buf = r.cache.dsa_kv_layer(dsa_idx)?;
                        r.graphs.run(
                            GraphKey { layer, rows, section: GraphSection::DsaAttnOut },
                            &r.stream,
                            || {
                                r.device.run(|| {
                                    r.dsa_attn(kv_cache_buf, wave_slot, rows, decode_attention)?;
                                    r.attn_out(&p, layer, true, gpus, local_heads, rows)
                                })
                            },
                        )?;
                    }
                }
            }
            self.exchange_attn_tp2(rows)?;
            // Defer attention F32 reduce + post-attention norm + iHC post/pre into FfnFrontend.
            batch.post_attn_deferred = true;
            batch.reduce_attn_pending = gpus == 2;
            return Ok(());
        }

        // Eager path (prefill, ungraphed decode/scoring, dump_dir, or skip_routed_experts).
        if let Some(prev) = prev_ffn {
            for (rank, r) in self.ranks.iter().enumerate() {
                let ffn_recv = ffn_recv_ptrs[rank];
                r.device.run(|| {
                    r.finish_deferred_ffn_local(
                        rank, gpus, prev, wave_slot, rows, prev_use_a, skip_routed, ffn_recv, routed_recv_offset,
                    )
                })?;
            }
        }
        if let Some(slot) = engram_slot {
            for r in &self.ranks {
                r.device.run(|| r.engram_compute(&p, slot, wave_slot, rows, use_a))?;
            }
            if batch.dump_rows > 0 && layer <= 3 {
                let w0 = &self.ranks[0].waves[wave_slot];
                let r_cur = if use_a { w0.r_a.buffer } else { w0.r_b.buffer };
                self.dump_bf16(&format!("l{layer:02}_after_engram"), r_cur, batch.dump_rows * HC * H)?;
            }
        }
        if is_sparse && self.context_split {
            let dsa_idx = self.dsa_index_of[&layer];
            for r in &self.ranks {
                let kv_cache_buf = r.cache.dsa_kv_layer(dsa_idx)?;
                r.device.run(|| {
                    r.attn_pre(&p, layer, true, local_heads, wave_slot, rows, use_a, kv_cache_buf)?;
                    r.dsa_index_wk(&p, wave_slot, rows)
                })?;
            }
            self.run_dsa_context_split_body(layer, dsa_idx, &p, batch)?;
        } else {
            for r in &self.ranks {
                let kv_cache_buf = if is_sparse {
                    r.cache.dsa_kv_layer(self.dsa_index_of[&layer])?
                } else {
                    r.cache.swa_layer_bank(self.swa_index_of[&layer])?
                };
                r.device.run(|| r.attn_pre(&p, layer, is_sparse, local_heads, wave_slot, rows, use_a, kv_cache_buf))?;
                if is_sparse {
                    let dsa_idx = self.dsa_index_of[&layer];
                    r.device.run(|| r.dsa_index_wk(&p, wave_slot, rows))?;
                    run_dsa_indexer_eager(r, dsa_idx, max_groups, pt_stride, false)?;
                    r.device.run(|| r.dsa_attn(kv_cache_buf, wave_slot, rows, decode_attention))?;
                } else {
                    r.device.run(|| r.swa_attn(kv_cache_buf, wave_slot, rows, members_len, swa_pt_stride, decode_attention))?;
                }
                r.device.run(|| r.attn_out(&p, layer, is_sparse, gpus, local_heads, rows))?;
            }
        }
        if gpus == 2 {
            let bf16_wire = batch.usage_phase == UsagePhase::Prefill;
            let exchange = self.peer_exchange()?;
            if bf16_wire {
                for (rank, r) in self.ranks.iter().enumerate() {
                    r.device.run(|| r.narrow_peer_partial(rows, H, r.shared.delta_f32.buffer, r.shared.delta.buffer))?;
                    exchange.push(rank, PEER_SLOT_ATTN, r.shared.delta.buffer.ptr, rows * H * 2)?;
                }
                exchange.wait(0, PEER_SLOT_ATTN)?;
                exchange.wait(1, PEER_SLOT_ATTN)?;
            } else {
                self.exchange_attn_tp2(rows)?;
            }
            let attn_recv = [exchange.recv(0, PEER_SLOT_ATTN)?, exchange.recv(1, PEER_SLOT_ATTN)?];
            let o_stem = format!("{p}.self_attn.o_proj");
            for (rank, r) in self.ranks.iter().enumerate() {
                r.device.run(|| {
                    if bf16_wire {
                        r.reduce_peer_bf16(rank, rows, H, Some(&o_stem), r.shared.delta.buffer, attn_recv[rank])
                    } else {
                        r.reduce_f32_to_bf16_local(rank, rows, H, Some(&o_stem), r.shared.delta_f32.buffer,
                            attn_recv[rank], r.shared.delta.buffer)
                    }
                })?;
            }
        }
        if batch.dump_rows > 0 && layer <= 3 {
            let s0 = &self.ranks[0].shared;
            let w0 = &self.ranks[0].waves[wave_slot];
            let dump_rows = batch.dump_rows;
            self.dump_bf16(&format!("l{layer:02}_attn_y"), w0.y.buffer, dump_rows * H)?;
            self.dump_bf16(&format!("l{layer:02}_attn_q_resid"), s0.q_resid.buffer, dump_rows * Q_LORA)?;
            self.dump_bf16(&format!("l{layer:02}_attn_query"), s0.query.buffer, dump_rows * local_heads * 576)?;
            self.dump_bf16(&format!("l{layer:02}_attn_out"), s0.attn_out.buffer, dump_rows * local_heads * 512)?;
            self.dump_f32(&format!("l{layer:02}_attn_lse"), s0.lse_out.buffer, dump_rows * local_heads)?;
            self.dump_bf16(&format!("l{layer:02}_attn_uv"), s0.uv_out.buffer, dump_rows * local_heads * 256)?;
            self.dump_bf16(&format!("l{layer:02}_attn_delta"), s0.delta.buffer, dump_rows * H)?;
        }

        // (C) post_proj_norm + fused attention iHC post + FFN iHC pre
        for r in &self.ranks {
            r.device.run(|| r.post_attn_ihc(&p, wave_slot, rows, use_a))?;
        }
        batch.use_a = !batch.use_a;
        Ok(())
    }

    /// Run post-attention and router; include the shared FFN only for local experts.
    fn run_moe_frontend(&mut self, layer: usize, batch: &mut ActiveBatch, local_experts: bool) -> Result<()> {
        let p = format!("model.layers.{layer}");
        let o_stem = format!("{p}.self_attn.o_proj");
        let gpus = self.ranks.len();
        let rows = batch.rows;
        let wave_slot = batch.wave_slot;
        let routed_scale = self.cfg.routed_scaling_factor as f32;
        let deferred = batch.post_attn_deferred;
        let reduce_attn = batch.reduce_attn_pending;
        let use_a = batch.use_a;
        let usage = self.expert_usage.as_ref();
        let count_usage = self.count_expert_usage;
        let skip_routed = self.skip_routed_experts;
        let usage_phase = batch.usage_phase;
        let attn_recv = if reduce_attn {
            let ex = self.peer_exchange()?;
            [ex.recv(0, PEER_SLOT_ATTN)?, ex.recv(1, PEER_SLOT_ATTN)?]
        } else {
            [std::ptr::null_mut(); 2]
        };
        if batch.graphed {
            for (rank, r) in self.ranks.iter().enumerate() {
                let recv_ptr = attn_recv[rank];
                r.graphs.run(
                    GraphKey { layer, rows, section: if local_experts { GraphSection::FfnFrontend } else { GraphSection::SparkRouter } },
                    &r.stream,
                    || {
                        r.device.run(|| {
                            if reduce_attn {
                                r.reduce_f32_to_bf16_local(
                                    rank, rows, H, Some(&o_stem), r.shared.delta_f32.buffer, recv_ptr, r.shared.delta.buffer,
                                )?;
                            }
                            if deferred {
                                r.post_attn_ihc(&p, wave_slot, rows, use_a)?;
                            }
                            if local_experts {
                                r.moe_router_shared(&p, gpus, wave_slot, rows, routed_scale)?;
                            } else if rank == 0 {
                                r.moe_router(&p, wave_slot, rows, routed_scale)?;
                            }
                            if rank == 0 {
                                r.record_expert_usage_local(usage, count_usage, skip_routed, false, layer, usage_phase, rows)?;
                            }
                            Ok(())
                        })
                    },
                )?;
            }
            if deferred {
                batch.post_attn_deferred = false;
                batch.use_a = !batch.use_a;
            }
            batch.reduce_attn_pending = false;
            return Ok(());
        }
        for (rank, r) in self.ranks.iter().enumerate() {
            let recv_ptr = attn_recv[rank];
            r.device.run(|| {
                if reduce_attn {
                    r.reduce_f32_to_bf16_local(
                        rank, rows, H, Some(&o_stem), r.shared.delta_f32.buffer, recv_ptr, r.shared.delta.buffer,
                    )?;
                }
                if deferred {
                    r.post_attn_ihc(&p, wave_slot, rows, use_a)?;
                }
                if local_experts {
                    r.moe_router_shared(&p, gpus, wave_slot, rows, routed_scale)?;
                } else if rank == 0 {
                    r.moe_router(&p, wave_slot, rows, routed_scale)?;
                }
                Ok(())
            })?;
        }
        if deferred {
            batch.post_attn_deferred = false;
            batch.use_a = !batch.use_a;
        }
        batch.reduce_attn_pending = false;
        Ok(())
    }

    /// Enqueue the shared FFN after the Spark request has been staged, before landing its reply.
    fn run_spark_shared_ffn(&self, layer: usize, batch: &ActiveBatch) -> Result<()> {
        let p = format!("model.layers.{layer}");
        let rows = batch.rows;
        let gpus = self.ranks.len();
        for r in &self.ranks {
            let work = || r.device.run(|| r.shared_ffn(&p, gpus, batch.wave_slot, rows));
            if batch.graphed {
                r.graphs.run(GraphKey { layer, rows, section: GraphSection::SparkShared }, &r.stream, work)?;
            } else {
                work()?;
            }
        }
        Ok(())
    }

    /// Finish post-FFN combine (`latent_up` + shared add + norm + iHC post) for dense or MoE layer.
    /// On TP2 graphed decode, exchanges FFN partials via `PeerExchange` and defers rank-local reduction
    /// + combine + iHC post into the next layer's attention section (or terminal `Tail` graph).
    fn finish_ffn_combine_and_post(&mut self, layer: usize, batch: &mut ActiveBatch, reduce_routed_tp2: bool) -> Result<()> {
        let gpus = self.ranks.len();
        let rows = batch.rows;
        let is_dense = self.cfg.dense_layer[layer];
        let wave_slot = batch.wave_slot;
        let rank0_only = batch.rank0_only_tail && layer == batch.final_layer;

        let bf16_wire = gpus == 2 && batch.usage_phase == UsagePhase::Prefill;
        if bf16_wire {
            self.exchange_ffn_prefill(wave_slot, rows, is_dense, reduce_routed_tp2, rank0_only)?;
        } else if gpus == 2 {
            self.exchange_ffn_tp2(wave_slot, rows, is_dense, reduce_routed_tp2, rank0_only)?;
        }

        let deferred_ffn = DeferredFfn { layer, is_dense, reduce_routed_tp2 };
        if batch.graphed && gpus == 2 {
            batch.prev_ffn_deferred = Some(deferred_ffn);
            return Ok(());
        }

        let use_a = batch.use_a;
        let skip_routed = self.skip_routed_experts;
        let ffn_recv = if gpus == 2 {
            let ex = self.peer_exchange()?;
            [ex.recv(0, PEER_SLOT_FFN)?, ex.recv(1, PEER_SLOT_FFN)?]
        } else {
            [std::ptr::null_mut(); 2]
        };
        let routed_off = self.ffn_routed_recv_offset;
        for (rank, r) in self.ranks.iter().enumerate() {
            if rank0_only && rank == 1 {
                continue;
            }
            let recv_ptr = ffn_recv[rank];
            if batch.graphed {
                r.graphs.run(
                    GraphKey { layer, rows, section: GraphSection::FfnFinish },
                    &r.stream,
                    || {
                        r.device.run(|| {
                            r.finish_deferred_ffn_local(
                                rank, gpus, deferred_ffn, wave_slot, rows, use_a, skip_routed, recv_ptr, routed_off,
                            )
                        })
                    },
                )?;
            } else if bf16_wire {
                r.device.run(|| {
                    let p = format!("model.layers.{layer}");
                    if is_dense {
                        r.reduce_peer_bf16(rank, rows, H, Some(&format!("{p}.mlp.down_proj")), r.shared.delta.buffer, recv_ptr)?;
                    } else {
                        if reduce_routed_tp2 {
                            r.reduce_peer_bf16(rank, rows, LATENT_H, None, r.shared.routed_latent.buffer,
                                recv_ptr.wrapping_byte_add(routed_off))?;
                        }
                        r.reduce_peer_bf16(rank, rows, H, Some(&format!("{p}.mlp.shared_experts.down_proj")),
                            r.waves[wave_slot].shared_out.buffer, recv_ptr)?;
                    }
                    r.ffn_combine_post(&p, is_dense, wave_slot, rows, use_a)
                })?;
            } else {
                r.device.run(|| {
                    r.finish_deferred_ffn_local(
                        rank, gpus, deferred_ffn, wave_slot, rows, use_a, skip_routed, recv_ptr, routed_off,
                    )
                })?;
            }
        }
        batch.use_a = !batch.use_a;

        if batch.dump_rows > 0 {
            let dump_rows = batch.dump_rows;
            let w0 = &self.ranks[0].waves[wave_slot];
            let r_cur = if batch.use_a { w0.r_a.buffer } else { w0.r_b.buffer };
            self.dump_bf16(&format!("l{layer:02}_out"), r_cur, dump_rows * HC * H)?;
            if layer <= 3 {
                let s0 = &self.ranks[0].shared;
                let w0 = &self.ranks[0].waves[wave_slot];
                self.dump_bf16(&format!("l{layer:02}_ffn_y"), w0.y.buffer, dump_rows * H)?;
                self.dump_bf16(&format!("l{layer:02}_ffn_delta"), s0.delta.buffer, dump_rows * H)?;
                self.dump_bf16(&format!("l{layer:02}_ffn_delta_norm"), s0.delta_norm.buffer, dump_rows * H)?;
                if !is_dense {
                    self.dump_bf16(&format!("l{layer:02}_ffn_routed_latent"), s0.routed_latent.buffer, dump_rows * LATENT_H)?;
                    self.dump_bf16(&format!("l{layer:02}_ffn_shared_out"), w0.shared_out.buffer, dump_rows * H)?;
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
        let o_stem = format!("{p}.self_attn.o_proj");
        let gpus = self.ranks.len();
        let rows = batch.rows;
        let wave_slot = batch.wave_slot;
        let deferred = batch.post_attn_deferred;
        let reduce_attn = batch.reduce_attn_pending;
        let use_a = batch.use_a;
        let routed_scale = self.cfg.routed_scaling_factor as f32;
        let skip_routed = self.skip_routed_experts;
        let usage = self.expert_usage.as_ref();
        let count_usage = self.count_expert_usage;
        let usage_phase = batch.usage_phase;
        let attn_recv = if reduce_attn {
            let ex = self.peer_exchange()?;
            [ex.recv(0, PEER_SLOT_ATTN)?, ex.recv(1, PEER_SLOT_ATTN)?]
        } else {
            [std::ptr::null_mut(); 2]
        };

        if self.cfg.dense_layer[layer] {
            for (rank, r) in self.ranks.iter().enumerate() {
                let recv_ptr = attn_recv[rank];
                if batch.graphed {
                    r.graphs.run(
                        GraphKey { layer, rows, section: GraphSection::FfnFrontend },
                        &r.stream,
                        || {
                            r.device.run(|| {
                                if reduce_attn {
                                    r.reduce_f32_to_bf16_local(
                                        rank, rows, H, Some(&o_stem), r.shared.delta_f32.buffer, recv_ptr, r.shared.delta.buffer,
                                    )?;
                                }
                                if deferred {
                                    r.post_attn_ihc(&p, wave_slot, rows, use_a)?;
                                }
                                r.dense_ffn(&p, gpus, wave_slot, rows)
                            })
                        },
                    )?;
                } else {
                    r.device.run(|| {
                        if reduce_attn {
                            r.reduce_f32_to_bf16_local(
                                rank, rows, H, Some(&o_stem), r.shared.delta_f32.buffer, recv_ptr, r.shared.delta.buffer,
                            )?;
                        }
                        if deferred {
                            r.post_attn_ihc(&p, wave_slot, rows, use_a)?;
                        }
                        r.dense_ffn(&p, gpus, wave_slot, rows)
                    })?;
                }
            }
            if deferred {
                batch.post_attn_deferred = false;
                batch.use_a = !batch.use_a;
            }
            batch.reduce_attn_pending = false;
            return self.finish_ffn_combine_and_post(layer, batch, true);
        }

        if batch.graphed {
            for (rank, r) in self.ranks.iter().enumerate() {
                let recv_ptr = attn_recv[rank];
                r.graphs.run(
                    GraphKey { layer, rows, section: GraphSection::FfnFrontend },
                    &r.stream,
                    || {
                        r.device.run(|| {
                            if reduce_attn {
                                r.reduce_f32_to_bf16_local(
                                    rank, rows, H, Some(&o_stem), r.shared.delta_f32.buffer, recv_ptr, r.shared.delta.buffer,
                                )?;
                            }
                            if deferred {
                                r.post_attn_ihc(&p, wave_slot, rows, use_a)?;
                            }
                            r.moe_router_shared(&p, gpus, wave_slot, rows, routed_scale)?;
                            r.local_exl3(layer, gpus, rows, skip_routed)?;
                            if rank == 0 {
                                r.record_expert_usage_local(usage, count_usage, skip_routed, false, layer, usage_phase, rows)?;
                            }
                            Ok(())
                        })
                    },
                )?;
            }
            if deferred {
                batch.post_attn_deferred = false;
                batch.use_a = !batch.use_a;
            }
            batch.reduce_attn_pending = false;
            return self.finish_ffn_combine_and_post(layer, batch, true);
        }

        self.run_moe_frontend(layer, batch, true)?;
        for r in &self.ranks {
            r.device.run(|| r.local_exl3(layer, gpus, rows, skip_routed))?;
        }
        self.finish_ffn_combine_and_post(layer, batch, true)
    }

    /// Project a `<= 64`-row tile of `r0.waves[wave_slot].y[y_row_off .. y_row_off + count]` through the
    /// vocabulary head into `r0.shared.head_logits_f32 [count, vocab]` with zero host hop.
    fn project_head_tile(&self, wave_slot: usize, y_row_off: usize, count: usize) -> Result<DeviceLogits> {
        ensure!(count > 0 && count <= 64, "Hy4 head tile count {count} outside 1..=64");
        let r0 = &self.ranks[0];
        let input = slice_buf(r0.waves[wave_slot].y.buffer, y_row_off * H * 2, count * H * 2)?;
        self.project_vocab_tile(wave_slot, input, count)
    }

    fn project_vocab_tile(&self, wave_slot: usize, x0_buf: CuteafdDeviceBuffer, count: usize) -> Result<DeviceLogits> {
        let vocab = self.cfg.vocab_size;
        let r0 = &self.ranks[0];
        let head_out = r0.shared.head_logits_f32.as_ref().context("rank 0 missing head_logits_f32")?.buffer;
        if self.ranks.len() == 1 {
            let out_buf = CuteafdDeviceBuffer {
                bytes: count * vocab * 4,
                ..head_out
            };
            r0.device.run(|| launch_rank_lm_head(r0, count, x0_buf, out_buf))?;
        } else {
            let r1 = &self.ranks[1];
            let w1 = &r1.waves[wave_slot];
            let handoff = self.head_handoff.as_ref().context("TP2 head handoff missing")?;
            let local_v = r0.weights.local_vocab;
            let x1_buf = CuteafdDeviceBuffer {
                bytes: count * H * 2,
                ..w1.y.buffer
            };
            let out0_buf = CuteafdDeviceBuffer {
                bytes: count * local_v * 4,
                ..r0.shared.logits_tile.buffer
            };
            let out1_buf = CuteafdDeviceBuffer {
                bytes: count * local_v * 4,
                ..r1.shared.logits_tile.buffer
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

    fn capture_draft_hidden(&self, batch: &ActiveBatch, first: usize, packed: bool) -> Result<()> {
        let Some(draft) = &self.draft else { return Ok(()); };
        let r = &self.ranks[0];
        r.device.run(|| {
            for (index, member) in batch.members.iter().enumerate() {
                let end = member.row_start + member.row_len;
                let start = if self.verifying { member.row_start.max(first) } else { end - 1 };
                if start < first { continue; }
                for row in start..end {
                    let src = if packed { index } else { row - first };
                    draft.capture_hidden(self.verifying, if self.verifying { row } else { member.slot },
                        slice_buf(r.waves[batch.wave_slot].y.buffer, src * H * 2, H * 2)?, &r.stream)?;
                }
                if !self.verifying { draft.publish(member.slot, member.pos_start + member.row_len); }
            }
            Ok(())
        })
    }

    /// Run terminal `hc_head` + sharded FP32 `lm_head` -> zero-host-hop `DeviceLogits` tiles.
    fn finish_head(
        &self,
        batch: &mut ActiveBatch,
        mode: HeadMode,
        on_tile: &mut dyn FnMut(usize, DeviceLogits) -> Result<()>,
    ) -> Result<()> {
        let wave_slot = batch.wave_slot;
        let gpus = self.ranks.len();
        let r0 = &self.ranks[0];
        let w0 = &r0.waves[wave_slot];

        // On TP2 graphed decode with full-batch Suffix(0), fuse the final layer's deferred FFN finish
        // with terminal `hc_head` into one rank-0 `Tail` graph.
        if let Some(prev) = batch.prev_ffn_deferred.take() {
            let prev_use_a = batch.use_a;
            batch.use_a = !batch.use_a;
            let ffn_recv0 = if gpus == 2 {
                self.peer_exchange()?.recv(0, PEER_SLOT_FFN)?
            } else {
                std::ptr::null_mut()
            };
            let skip_routed = self.skip_routed_experts;
            let routed_off = self.ffn_routed_recv_offset;
            if batch.graphed && gpus == 2 && mode == HeadMode::Suffix(0) && batch.rows > 0 {
                let rows = batch.rows;
                let r_cur = if batch.use_a { w0.r_a.buffer } else { w0.r_b.buffer };
                r0.graphs.run(
                    GraphKey { layer: prev.layer, rows, section: GraphSection::Tail },
                    &r0.stream,
                    || {
                        r0.device.run(|| {
                            r0.finish_deferred_ffn_local(
                                0, gpus, prev, wave_slot, rows, prev_use_a, skip_routed, ffn_recv0, routed_off,
                            )?;
                            // SAFETY: arena weights and scratch live through stream drain.
                            unsafe {
                                r0.programs.hc_head.launch(
                                    &[
                                        r_cur.ptr,
                                        r0.weights.buf("model.hc_head.hc_head_fn")?.ptr,
                                        r0.weights.buf("model.hc_head.hc_head_scale")?.ptr,
                                        r0.weights.buf("model.hc_head.hc_head_base")?.ptr,
                                        r0.weights.buf("model.norm.weight")?.ptr,
                                        r0.shared.delta_norm.buffer.ptr,
                                        w0.y.buffer.ptr,
                                    ],
                                    &[Scalar::I32(rows as i32)],
                                    r0.stream.raw,
                                )
                            }
                        })
                    },
                )?;
                self.capture_draft_hidden(batch, 0, false)?;
                if batch.dump_rows > 0 {
                    self.dump_bf16("hc_head_y", w0.y.buffer, batch.dump_rows * H)?;
                }
                for tile_off in (0..rows).step_by(64) {
                    let count = (rows - tile_off).min(64);
                    let logits = self.project_head_tile(wave_slot, tile_off, count)?;
                    on_tile(tile_off, logits)?;
                }
                return Ok(());
            }
            let ffn_recv = if gpus == 2 {
                let ex = self.peer_exchange()?;
                [ex.recv(0, PEER_SLOT_FFN)?, ex.recv(1, PEER_SLOT_FFN)?]
            } else {
                [std::ptr::null_mut(); 2]
            };
            let rank0_only = batch.rank0_only_tail && prev.layer == batch.final_layer;
            for (rank, r) in self.ranks.iter().enumerate() {
                if rank0_only && rank == 1 {
                    continue;
                }
                r.device.run(|| {
                    r.finish_deferred_ffn_local(
                        rank, gpus, prev, wave_slot, batch.rows, prev_use_a, skip_routed, ffn_recv[rank], routed_off,
                    )
                })?;
            }
        }

        let r_cur = if batch.use_a { w0.r_a.buffer } else { w0.r_b.buffer };
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
                                ptr: r0.shared.r_head_in.buffer.ptr.cast::<u8>().wrapping_add(idx * row_bytes).cast(),
                                bytes: row_bytes,
                                ..r0.shared.r_head_in.buffer
                            };
                            // SAFETY: r_cur and r_head_in are disjoint admitted buffers on rank 0.
                            unsafe { r0.device.library.copy_d2d_async(dst, src, row_bytes, r0.stream.raw)?; }
                        }
                        Ok(())
                    })?;
                    r0.shared.r_head_in.buffer.ptr
                };
                r0.device.run(|| unsafe {
                    r0.programs.hc_head.launch(
                        &[
                            hc_src_ptr,
                            r0.weights.buf("model.hc_head.hc_head_fn")?.ptr,
                            r0.weights.buf("model.hc_head.hc_head_scale")?.ptr,
                            r0.weights.buf("model.hc_head.hc_head_base")?.ptr,
                            r0.weights.buf("model.norm.weight")?.ptr,
                            r0.shared.delta_norm.buffer.ptr,
                            w0.y.buffer.ptr,
                        ],
                        &[Scalar::I32(count as i32)],
                        r0.stream.raw,
                    )
                })?;
                self.capture_draft_hidden(batch, 0, true)?;
                let logits = self.project_head_tile(wave_slot, 0, count)?;
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
                            r0.shared.delta_norm.buffer.ptr,
                            w0.y.buffer.ptr,
                        ],
                        &[Scalar::I32(total as i32)],
                        r0.stream.raw,
                    )
                })?;
                self.capture_draft_hidden(batch, first_row, false)?;
                if batch.dump_rows > 0 && first_row == 0 {
                    self.dump_bf16("hc_head_y", w0.y.buffer, batch.dump_rows * H)?;
                }
                for tile_off in (0..total).step_by(64) {
                    let count = (total - tile_off).min(64);
                    let logits = self.project_head_tile(wave_slot, tile_off, count)?;
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
        ensure!(!self.verifying || (units.len() == 1 && units[0].rows.len() <= self.decode_rows),
            "Hy4 provisional verification requires one admitted decode unit");
        let initial_starts: Vec<usize> = placements.iter().map(|p| p.len).collect();
        let staged_frontiers = initial_starts.clone();
        let staged_engram_tails: Vec<[u32; 2]> = placements.iter().map(|p| p.engram_tail).collect();
        let run_waves = self.prefill_waves;
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
                staged_frontiers: Vec<usize>,
                staged_engram_tails: Vec<[u32; 2]>,
                sources: &'p [&'p [u32]],
                priority: ReadPriority,
                kind: ExpertV2SourceKind,
                head_mode: F,
                on_tile: &'p mut dyn FnMut(&Unit, usize, DeviceLogits) -> Result<()>,
                active: Vec<Option<ActiveBatch>>,
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
                    let slot = unit.slot;
                    ensure!(
                        slot < self.active.len() && self.active[slot].is_none(),
                        "Hy4 begin_unit slot {slot} invalid or already occupied"
                    );
                    let mode = if unit.layers.end == self.engine.cfg.layers {
                        (self.head_mode)(unit)
                    } else {
                        HeadMode::None
                    };
                    let batch = self.engine.begin_batch(
                        self.placements,
                        self.initial_starts,
                        &mut self.staged_frontiers,
                        &mut self.staged_engram_tails,
                        self.sources,
                        unit,
                        self.priority,
                        mode,
                    )?;
                    self.active[slot] = Some(batch);
                    Ok(())
                }
                fn attention(&mut self, layer: usize, unit: &Unit) -> Result<()> {
                    let local_route = self.route(layer) == FfnRoute::Local;
                    let batch = self
                        .active
                        .get_mut(unit.slot)
                        .and_then(Option::as_mut)
                        .context("Hy4 attention without active unit")?;
                    self.engine.run_layer_attention(layer, batch, local_route)
                }
                fn local_ffn(&mut self, layer: usize, unit: &Unit) -> Result<()> {
                    let batch = self
                        .active
                        .get_mut(unit.slot)
                        .and_then(Option::as_mut)
                        .context("Hy4 FFN without active unit")?;
                    self.engine.run_layer_local_ffn(layer, batch)?;
                    if !batch.graphed {
                        self.engine.record_expert_usage(layer, batch)?;
                    }
                    Ok(())
                }
                fn send_experts(&mut self, layer: usize, unit: &Unit) -> Result<Option<WaveTicket>> {
                    let batch = self
                        .active
                        .get_mut(unit.slot)
                        .and_then(Option::as_mut)
                        .context("Hy4 send_experts without active unit")?;
                    self.engine.run_moe_frontend(layer, batch, false)?;
                    if self.engine.ranks.len() == 2 && self.engine.prefill_waves > 1 {
                        // Protect rank 1's PEER_SLOT_ATTN read before another wave starts attention(layer).
                        let exchange = self.engine.peer_exchange()?;
                        exchange.signal(1, DIRECT)?;
                        exchange.wait(0, DIRECT)?;
                    }
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
                                    r0.shared.route_ids.buffer,
                                    r0.shared.route_weights.buffer,
                                    r0.shared.wire_in.buffer,
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
                    // Waves::send copies route_ids, route_weights and wire_in to slot-owned staging
                    // in stream order, then publishes staged. The shared FFN reads wave.y and writes
                    // ffn_gate/up/act, native projection scratch and wave.shared_out/shared_f32 only.
                    // Queue it now so Spark work overlaps it, before land/reduce and TP2 FFN exchange.
                    self.engine.run_spark_shared_ffn(layer, batch)?;
                    if !batch.graphed {
                        self.engine.record_expert_usage(layer, batch)?;
                    }
                    Ok(Some(ticket))
                }
                fn land_experts(&mut self, layer: usize, unit: &Unit, ticket: Option<WaveTicket>) -> Result<()> {
                    let ticket = ticket.context("Hy4 land_experts missing WaveTicket")?;
                    let slot = ticket.slot();
                    let batch = self
                        .active
                        .get_mut(unit.slot)
                        .and_then(Option::as_mut)
                        .context("Hy4 land_experts without active unit")?;
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
                                r0.shared.routed_latent.buffer.ptr.cast(),
                                rows,
                                r0.stream.raw,
                            )
                        }
                    })?;
                    self.engine.finish_ffn_combine_and_post(layer, batch, false)
                }
                fn end_unit(&mut self, unit: &Unit) -> Result<()> {
                    let batch = self
                        .active
                        .get_mut(unit.slot)
                        .and_then(Option::as_mut)
                        .context("Hy4 end_unit without active unit")?;
                    let mode = if unit.layers.end == self.engine.cfg.layers {
                        (self.head_mode)(unit)
                    } else {
                        HeadMode::None
                    };
                    self.engine.finish_head(batch, mode, &mut |row_off, logits| (self.on_tile)(unit, row_off, logits))
                }
                fn commit_unit(&mut self, unit: &Unit) -> Result<()> {
                    let batch = self
                        .active
                        .get_mut(unit.slot)
                        .and_then(Option::take)
                        .context("Hy4 commit_unit without active unit")?;
                    // End-of-unit boundary: only wait for the source DMA, not the layers.
                    for group in &mut self.engine.metadata_uploads {
                        group[unit.slot].retire()?;
                    }
                    for m in &batch.members {
                        let p = &mut self.placements[m.sequence];
                        p.len = m.pos_start + m.row_len;
                        if !self.engine.verifying {
                            p.state_len = p.len;
                            p.engram_tail = m.next_engram_tail;
                        }
                    }
                    Ok(())
                }
            }
            let mut hooks = Hooks {
                engine: self,
                waves,
                placements,
                initial_starts: &initial_starts,
                staged_frontiers,
                staged_engram_tails,
                sources,
                priority,
                kind,
                head_mode,
                on_tile,
                active: (0..run_waves).map(|_| None).collect(),
            };
            run_prefill_units(&mut hooks, units, run_waves)?;
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

    /// Run a contiguous single-sequence span through layers `0..=last_layer` across `self.prefill_waves`
    /// wave slots (multi-wave golden verification).
    pub(crate) fn forward_span(&mut self, tokens: &[u32], unit_rows: usize, last_layer: usize) -> Result<Vec<f32>> {
        use crate::shared::prefill_pipeline::sequence_units;
        ensure!(!tokens.is_empty() && unit_rows > 0 && unit_rows <= self.max_rows, "invalid Hy4 span geometry");
        ensure!(last_layer < self.cfg.layers, "last_layer {last_layer} >= {}", self.cfg.layers);
        let start = self.golden_placement.len;
        let mut units = sequence_units(tokens.len(), unit_rows, self.prefill_waves, last_layer + 1)?;
        let last_idx = units.len() - 1;
        for (i, u) in units.iter_mut().enumerate() {
            let u_start = start + u.rows.start;
            let u_end = start + u.rows.end;
            u.rows = u_start..u_end;
            u.segments = vec![UnitSegment {
                sequence: 0,
                rows: 0..u.rows.len(),
                tokens: u_start..u_end,
                prompt_end: i == last_idx,
                snapshot: false,
                head: last_layer + 1 == self.cfg.layers,
            }];
        }
        let vocab = self.cfg.vocab_size;
        let mut all_logits = if last_layer + 1 == self.cfg.layers {
            vec![0f32; tokens.len() * vocab]
        } else {
            Vec::new()
        };
        let mut placement = [self.golden_placement.clone()];
        let library = self.library();
        let mode = if last_layer + 1 == self.cfg.layers {
            HeadMode::Suffix(0)
        } else {
            HeadMode::None
        };
        self.run_units(
            &mut placement,
            &[tokens],
            &units,
            ReadPriority::Prefill,
            |_| mode,
            &mut |u, row_off, logits| {
                let tile = logits.to_host(library)?;
                let dst = (u.rows.start - start + row_off) * vocab;
                all_logits[dst..dst + tile.len()].copy_from_slice(&tile);
                Ok(())
            },
        )?;
        self.golden_placement = placement.into_iter().next().unwrap();
        if last_layer + 1 < self.cfg.layers {
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

    /// Prove byte-identity of logits and `engram_tail` across multi-wave single-sequence execution
    /// AND concurrent two-sequence execution (`prompt_a` in wave `k`, `prompt_b` in wave `k+1`)
    /// versus solo `waves=1` execution, while asserting disjoint per-wave Engram buffers.
    pub(crate) fn verify_concurrent_two_seq_waves(
        &mut self,
        prompt_a: &[u32],
        prompt_b: &[u32],
        unit_rows: usize,
        last_layer: usize,
    ) -> Result<TwoSeqWaveProof> {
        use crate::shared::prefill_pipeline::sequence_units;
        ensure!(
            self.prefill_waves >= 2 && self.max_sequences >= 2,
            "verify_concurrent_two_seq_waves requires prefill_waves >= 2 and max_sequences >= 2"
        );
        ensure!(
            !prompt_a.is_empty() && !prompt_b.is_empty() && unit_rows > 0 && unit_rows <= self.max_rows,
            "invalid prompt or unit_rows for wave verification"
        );
        ensure!(last_layer + 1 == self.cfg.layers, "verify_concurrent_two_seq_waves requires full 42 layers");
        self.assert_disjoint_wave_engram_buffers()?;
        let buffer_summary = self.wave_engram_buffer_summary();

        let pages_a = prompt_a.len().div_ceil(PAGE_ROWS);
        let pages_b = prompt_b.len().div_ceil(PAGE_ROWS);
        ensure!(
            pages_a + pages_b <= self.max_pages,
            "two-sequence wave proof needs {} pages, engine has {}",
            pages_a + pages_b,
            self.max_pages
        );
        let vocab = self.cfg.vocab_size;
        let pad_id = self.cfg.engram_pad_id;
        let layers = self.cfg.layers;
        let library = self.library();

        let make_solo_units = |len: usize, waves: usize| -> Result<Vec<Unit>> {
            let mut units = sequence_units(len, unit_rows, waves, layers)?;
            let last_idx = units.len() - 1;
            for (i, u) in units.iter_mut().enumerate() {
                let span = u.rows.clone();
                u.segments = vec![UnitSegment {
                    sequence: 0,
                    rows: 0..span.len(),
                    tokens: span,
                    prompt_end: i == last_idx,
                    snapshot: false,
                    head: true,
                }];
            }
            Ok(units)
        };

        // 1. Prompt A solo at waves=1 (one unit per run_units call = strictly 1 wave in flight).
        self.reset()?;
        let units_a1 = make_solo_units(prompt_a.len(), 1)?;
        let mut p_a1 = [Hy4Placement::new((0..pages_a as u32).collect(), 0, pad_id)];
        let mut logits_a_solo = vec![0f32; prompt_a.len() * vocab];
        for mut unit in units_a1 {
            unit.index = 0;
            unit.slot = 0;
            let src = &prompt_a[unit.rows.clone()];
            self.run_units(
                &mut p_a1,
                &[src],
                &[unit],
                ReadPriority::Prefill,
                |_| HeadMode::Suffix(0),
                &mut |u, row_off, logits| {
                    let tile = logits.to_host(library)?;
                    let dst = (u.rows.start + row_off) * vocab;
                    logits_a_solo[dst..dst + tile.len()].copy_from_slice(&tile);
                    Ok(())
                },
            )?;
        }
        let tail_a_solo = p_a1[0].engram_tail;

        // 2. Prompt B solo at waves=1 (one unit per run_units call = strictly 1 wave in flight).
        self.reset()?;
        let units_b1 = make_solo_units(prompt_b.len(), 1)?;
        let mut p_b1 = [Hy4Placement::new((pages_a as u32..(pages_a + pages_b) as u32).collect(), 1, pad_id)];
        let mut logits_b_solo = vec![0f32; prompt_b.len() * vocab];
        for mut unit in units_b1 {
            unit.index = 0;
            unit.slot = 0;
            let src = &prompt_b[unit.rows.clone()];
            self.run_units(
                &mut p_b1,
                &[src],
                &[unit],
                ReadPriority::Prefill,
                |_| HeadMode::Suffix(0),
                &mut |u, row_off, logits| {
                    let tile = logits.to_host(library)?;
                    let dst = (u.rows.start + row_off) * vocab;
                    logits_b_solo[dst..dst + tile.len()].copy_from_slice(&tile);
                    Ok(())
                },
            )?;
        }
        let tail_b_solo = p_b1[0].engram_tail;

        // 3. Prompt A solo at waves=W (cross-wave boundary tail + logits check).
        self.reset()?;
        let units_aw = make_solo_units(prompt_a.len(), self.prefill_waves)?;
        ensure!(units_aw.len() >= 2, "prompt A must span >= 2 units at unit_rows={unit_rows} to cross wave boundary");
        let mut p_aw = [Hy4Placement::new((0..pages_a as u32).collect(), 0, pad_id)];
        let mut logits_a_mw = vec![0f32; prompt_a.len() * vocab];
        self.run_units(
            &mut p_aw,
            &[prompt_a],
            &units_aw,
            ReadPriority::Prefill,
            |_| HeadMode::Suffix(0),
            &mut |u, row_off, logits| {
                let tile = logits.to_host(library)?;
                let dst = (u.rows.start + row_off) * vocab;
                logits_a_mw[dst..dst + tile.len()].copy_from_slice(&tile);
                Ok(())
            },
        )?;
        let tail_a_mw = p_aw[0].engram_tail;
        let diff_a_mw = count_f32_bit_diffs(&logits_a_solo, &logits_a_mw);
        ensure!(
            diff_a_mw == 0 && tail_a_mw == tail_a_solo,
            "single-sequence multi-wave (waves={}) diverged from waves=1: diff_elems={diff_a_mw} tail={tail_a_mw:?} != {tail_a_solo:?}",
            self.prefill_waves
        );

        // 4. Prompt A and Prompt B concurrently in interleaved waves (A in wave k, B in wave k+1).
        self.reset()?;
        let a_ranges = crate::shared::prefill_pipeline::prefill_units(prompt_a.len(), unit_rows, self.prefill_waves, 256)?;
        let b_ranges = crate::shared::prefill_pipeline::prefill_units(prompt_b.len(), unit_rows, self.prefill_waves, 256)?;
        let mut interleaved = Vec::with_capacity(a_ranges.len() + b_ranges.len());
        let max_len = a_ranges.len().max(b_ranges.len());
        let mut idx = 0usize;
        for step in 0..max_len {
            if let Some(span) = a_ranges.get(step).cloned() {
                interleaved.push(Unit {
                    index: idx,
                    sequence: 0,
                    segments: vec![UnitSegment {
                        sequence: 0,
                        rows: 0..span.len(),
                        tokens: span.clone(),
                        prompt_end: step + 1 == a_ranges.len(),
                        snapshot: false,
                        head: true,
                    }],
                    rows: span,
                    layers: 0..layers,
                    slot: idx % self.prefill_waves,
                    barrier: false,
                });
                idx += 1;
            }
            if let Some(span) = b_ranges.get(step).cloned() {
                interleaved.push(Unit {
                    index: idx,
                    sequence: 1,
                    segments: vec![UnitSegment {
                        sequence: 1,
                        rows: 0..span.len(),
                        tokens: span.clone(),
                        prompt_end: step + 1 == b_ranges.len(),
                        snapshot: false,
                        head: true,
                    }],
                    rows: span,
                    layers: 0..layers,
                    slot: idx % self.prefill_waves,
                    barrier: false,
                });
                idx += 1;
            }
        }
        let mut placements = [
            Hy4Placement::new((0..pages_a as u32).collect(), 0, pad_id),
            Hy4Placement::new((pages_a as u32..(pages_a + pages_b) as u32).collect(), 1, pad_id),
        ];
        let mut logits_a_conc = vec![0f32; prompt_a.len() * vocab];
        let mut logits_b_conc = vec![0f32; prompt_b.len() * vocab];
        self.run_units(
            &mut placements,
            &[prompt_a, prompt_b],
            &interleaved,
            ReadPriority::Prefill,
            |_| HeadMode::Suffix(0),
            &mut |u, row_off, logits| {
                let tile = logits.to_host(library)?;
                let dst = (u.rows.start + row_off) * vocab;
                if u.sequence == 0 {
                    logits_a_conc[dst..dst + tile.len()].copy_from_slice(&tile);
                } else {
                    logits_b_conc[dst..dst + tile.len()].copy_from_slice(&tile);
                }
                Ok(())
            },
        )?;
        let tail_a_conc = placements[0].engram_tail;
        let tail_b_conc = placements[1].engram_tail;
        let diff_a_conc = count_f32_bit_diffs(&logits_a_solo, &logits_a_conc);
        let diff_b_conc = count_f32_bit_diffs(&logits_b_solo, &logits_b_conc);
        ensure!(
            diff_a_conc == 0 && tail_a_conc == tail_a_solo,
            "concurrent 2-seq multi-wave prompt A diverged: diff_elems={diff_a_conc} tail={tail_a_conc:?} != {tail_a_solo:?}"
        );
        ensure!(
            diff_b_conc == 0 && tail_b_conc == tail_b_solo,
            "concurrent 2-seq multi-wave prompt B diverged: diff_elems={diff_b_conc} tail={tail_b_conc:?} != {tail_b_solo:?}"
        );

        Ok(TwoSeqWaveProof {
            gpus: self.ranks.len(),
            waves: self.prefill_waves,
            unit_rows,
            prompt_a_len: prompt_a.len(),
            prompt_b_len: prompt_b.len(),
            units_a: a_ranges.len(),
            units_b: b_ranges.len(),
            logits_a_solo,
            logits_b_solo,
            tail_a: tail_a_solo,
            tail_b: tail_b_solo,
            buffer_summary,
        })
    }

    /// Return the active 4-stream residual `[rows, 4, H]` F32 from rank 0 (for layer-0 boundary checks).
    pub(crate) fn read_residual_f32(&self, rows: usize) -> Result<Vec<f32>> {
        let r0 = &self.ranks[0];
        r0.stream.drain()?;
        let bytes = rows * HC * H * 2;
        let mut raw = vec![0u8; bytes];
        r0.device.run(|| {
            r0.device.library.copy_d2h(&mut raw, CuteafdDeviceBuffer { bytes, ..r0.waves[0].r_a.buffer })
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

pub(crate) struct TwoSeqWaveProof {
    pub(crate) gpus: usize,
    pub(crate) waves: usize,
    pub(crate) unit_rows: usize,
    pub(crate) prompt_a_len: usize,
    pub(crate) prompt_b_len: usize,
    pub(crate) units_a: usize,
    pub(crate) units_b: usize,
    pub(crate) logits_a_solo: Vec<f32>,
    pub(crate) logits_b_solo: Vec<f32>,
    pub(crate) tail_a: [u32; 2],
    pub(crate) tail_b: [u32; 2],
    pub(crate) buffer_summary: Vec<String>,
}

fn count_f32_bit_diffs(a: &[f32], b: &[f32]) -> usize {
    a.iter().zip(b.iter()).filter(|(x, y)| x.to_bits() != y.to_bits()).count() + a.len().abs_diff(b.len())
}

/// Advance staged frontier and rolling Engram tail on `begin_unit` before any predecessor wave commits.
fn stage_segment_frontier_and_tail(
    sequence: usize,
    token_start: usize,
    tok_slice: &[u32],
    staged_frontiers: &mut [usize],
    staged_engram_tails: &mut [[u32; 2]],
) -> Result<(usize, usize, [u32; 2], [u32; 2])> {
    let pos_start = *staged_frontiers
        .get(sequence)
        .with_context(|| format!("Hy4 staged frontier sequence {sequence} out of range"))?;
    ensure!(
        token_start == pos_start,
        "Hy4 segment start {token_start} differs from staged frontier {pos_start}"
    );
    let cur_tail = *staged_engram_tails
        .get(sequence)
        .with_context(|| format!("Hy4 staged Engram tail sequence {sequence} out of range"))?;
    let pos_end = pos_start
        .checked_add(tok_slice.len())
        .context("Hy4 staged frontier overflow")?;
    let next_tail = advance_tail(cur_tail, tok_slice, tok_slice.len());
    staged_frontiers[sequence] = pos_end;
    staged_engram_tails[sequence] = next_tail;
    Ok((pos_start, pos_end, cur_tail, next_tail))
}

#[allow(clippy::too_many_arguments)]
/// `src` is local to this verify member; `dst` is inside its live pending group.
fn accepted_pending_copy(start: usize, accepted: usize) -> Option<(usize, usize, usize)> {
    let old = start % 4;
    let new = (start + accepted) % 4;
    match new {
        0 => None,
        _ if old + accepted < 4 => Some((0, old, accepted)),
        _ => Some((accepted - new, 0, new)),
    }
}

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
    fn partial_verify_commits_only_the_accepted_indexer_and_engram_tail() {
        // Exercise every C4 alignment and every rejection position, including
        // history-copy extensions that cross several compressed groups.
        for start in 0..8 {
            for proposed in 1..=16 {
                let tokens = (0..start + proposed).map(|i| i as u32 + 10).collect::<Vec<_>>();
                let old = advance_tail([0, 0], &tokens[..start], start);
                for keep in 1..=proposed {
                    let mut pending = [0u32; 3];
                    pending[..start % 4].copy_from_slice(&tokens[start - start % 4..start]);
                    if let Some((src, dst, rows)) = accepted_pending_copy(start, keep) {
                        pending[dst..dst + rows].copy_from_slice(&tokens[start + src..start + src + rows]);
                    }
                    let end = start + keep;
                    assert_eq!(&pending[..end % 4], &tokens[end - end % 4..end]);
                    assert_eq!(advance_tail(old, &tokens[start..], keep), advance_tail([0, 0], &tokens[..end], end));
                }
            }
        }
    }

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

    #[test]
    fn staged_engram_tails_advance_on_stage_across_waves_1_2_3() -> Result<()> {
        use crate::shared::prefill_pipeline::{prefill_schedule, sequence_units, Step};
        use cuteafd_loader::families::hy4_flash::engram::{Hy4EngramHasher, Hy4EngramHistory};

        let pad = 0u32;
        let prompt_a: Vec<u32> = (0..1536u32).map(|i| 100 + (i * 37) % 1000).collect();
        let prompt_b: Vec<u32> = (0..1536u32).map(|i| 2000 + (i * 53) % 1000).collect();
        let unit_rows = 512usize;
        let layers = 4usize;
        let routes = vec![FfnRoute::Local, FfnRoute::Spark, FfnRoute::Local, FfnRoute::Spark];

        // Synthetic Engram hasher to prove hashed row IDs across wave boundaries match unchunked hashing.
        let mut vocab_map = BTreeMap::new();
        vocab_map.insert(1usize, vec![997u64; 16]);
        let cfg = Hy4FlashConfig {
            vocab_size: 120_832,
            hidden: 3072,
            layers: 42,
            intermediate: 13_824,
            max_position: 262_144,
            rms_norm_eps: 1e-5,
            hc_mult: 4,
            hc_eps: 1e-6,
            hc_magnitude: 2.0,
            q_lora_rank: 1024,
            kv_lora_rank: 512,
            qk_nope_dim: 192,
            qk_rope_dim: 64,
            v_head_dim: 256,
            sparse_heads: 32,
            swa_heads: 64,
            swa_qk_nope_dim: 192,
            swa_qk_rope_dim: 64,
            swa_v_head_dim: 256,
            sliding_window: 512,
            sparse_rope_theta: 10_000_000.0,
            swa_rope_theta: 10_000.0,
            sparse_layer: (0..42).map(|i| i % 4 == 0).collect(),
            indexer_heads: 16,
            indexer_head_dim: 128,
            indexer_compress_ratio: 4,
            indexer_group_topk: 512,
            indexer_candidate_topk: 2048,
            dense_layer: (0..42).map(|i| i == 0).collect(),
            experts: 512,
            topk: 16,
            moe_intermediate: 1536,
            routed_expert_hidden: 1536,
            shared_experts: 1,
            n_group: 1,
            topk_group: 1,
            norm_topk_prob: true,
            routed_scaling_factor: 3.998,
            swiglu_limit: 10.0,
            engram_layers: vec![1],
            engram_vocab_sizes: vocab_map,
            engram_max_ngram_size: 3,
            engram_n_embed_per_ngram: 768,
            engram_n_head_per_ngram: 8,
            engram_pad_id: pad,
            engram_seed: 10_007,
            engram_fp8_embedding: true,
            engram_lookup_table_size: 4096,
            image_token_id: 120_120,
            video_token_id: 120_124,
            eos: vec![120_025],
        };
        let lookup: Vec<i32> = (0..4096i32).collect();
        let hasher = Hy4EngramHasher::new(&cfg, 1, lookup)?;
        let mut expected_rows_a = Vec::new();
        let mut hist_a = Hy4EngramHistory(vec![pad, pad]);
        hasher.hash(&mut hist_a, &prompt_a, &mut expected_rows_a)?;
        let mut expected_rows_b = Vec::new();
        let mut hist_b = Hy4EngramHistory(vec![pad, pad]);
        hasher.hash(&mut hist_b, &prompt_b, &mut expected_rows_b)?;

        // (1) Single-sequence across waves=1, 2, 3: Begin(unit 1) executes BEFORE Commit(unit 0).
        let mut ref_records = Vec::new();
        let mut ref_final_tail = [0u32; 2];
        for waves in [1usize, 2, 3] {
            let mut units = sequence_units(prompt_a.len(), unit_rows, waves, layers)?;
            for u in &mut units {
                let span = u.rows.clone();
                u.segments = vec![UnitSegment {
                    sequence: 0,
                    rows: 0..span.len(),
                    tokens: span,
                    prompt_end: false,
                    snapshot: false,
                    head: false,
                }];
            }
            let steps = prefill_schedule(&routes, &units, waves)?;
            let mut committed_tail = [pad, pad];
            let mut staged_frontiers = vec![0usize];
            let mut staged_tails = vec![[pad, pad]];
            let mut in_flight: Vec<Option<[u32; 2]>> = vec![None; waves];
            let mut unit_records: Vec<([u32; 2], Vec<u32>)> = Vec::new();
            let mut hashed_rows = Vec::new();
            let mut saw_uncommitted_overlap = false;

            for step in steps {
                match step {
                    Step::Begin { unit } => {
                        let u = &units[unit];
                        let seg = &u.segments[0];
                        let tok_slice = &prompt_a[seg.tokens.clone()];
                        if unit > 0 && committed_tail == [pad, pad] {
                            saw_uncommitted_overlap = true;
                        }
                        let (_, _, cur_tail, next_tail) = stage_segment_frontier_and_tail(
                            seg.sequence,
                            seg.tokens.start,
                            tok_slice,
                            &mut staged_frontiers,
                            &mut staged_tails,
                        )?;
                        assert!(in_flight[u.slot].is_none(), "wave slot collision at unit {unit}");
                        in_flight[u.slot] = Some(next_tail);
                        unit_records.push((cur_tail, tok_slice.to_vec()));
                        let mut h = Hy4EngramHistory(cur_tail.to_vec());
                        hasher.hash(&mut h, tok_slice, &mut hashed_rows)?;
                    }
                    Step::Commit { unit } => {
                        let u = &units[unit];
                        committed_tail = in_flight[u.slot].take().expect("active wave on commit");
                    }
                    _ => {}
                }
            }
            if waves > 1 {
                assert!(saw_uncommitted_overlap, "waves={waves} must begin unit 1 before unit 0 commits");
            }
            assert_eq!(hashed_rows, expected_rows_a, "waves={waves} hashed Engram rows diverged across wave boundaries");
            assert_eq!(committed_tail, [prompt_a[prompt_a.len() - 2], prompt_a[prompt_a.len() - 1]]);
            if waves == 1 {
                ref_records = unit_records;
                ref_final_tail = committed_tail;
            } else {
                assert_eq!(unit_records, ref_records, "waves={waves} staged (cur_tail, tokens) diverged from waves=1");
                assert_eq!(committed_tail, ref_final_tail, "waves={waves} committed engram_tail diverged from waves=1");
            }
        }

        // (2) Two sequences interleaved (prompt A in wave k, prompt B in wave k+1) at waves=2 and 3.
        for waves in [2usize, 3] {
            let mut units = Vec::new();
            let mut idx = 0usize;
            for step in 0..(prompt_a.len() / unit_rows) {
                let span = step * unit_rows..(step + 1) * unit_rows;
                for seq in [0usize, 1] {
                    units.push(Unit {
                        index: idx,
                        sequence: seq,
                        segments: vec![UnitSegment {
                            sequence: seq,
                            rows: 0..span.len(),
                            tokens: span.clone(),
                            prompt_end: false,
                            snapshot: false,
                            head: false,
                        }],
                        rows: span.clone(),
                        layers: 0..layers,
                        slot: idx % waves,
                        barrier: false,
                    });
                    idx += 1;
                }
            }
            let steps = prefill_schedule(&routes, &units, waves)?;
            let mut committed = [[pad, pad]; 2];
            let mut staged_frontiers = vec![0usize; 2];
            let mut staged_tails = vec![[pad, pad]; 2];
            let mut in_flight: Vec<Option<(usize, [u32; 2])>> = vec![None; waves];
            let mut hashed = [Vec::new(), Vec::new()];
            let sources = [&prompt_a[..], &prompt_b[..]];

            for step in steps {
                match step {
                    Step::Begin { unit } => {
                        let u = &units[unit];
                        let seg = &u.segments[0];
                        let tok_slice = &sources[seg.sequence][seg.tokens.clone()];
                        let (_, _, cur_tail, next_tail) = stage_segment_frontier_and_tail(
                            seg.sequence,
                            seg.tokens.start,
                            tok_slice,
                            &mut staged_frontiers,
                            &mut staged_tails,
                        )?;
                        assert!(in_flight[u.slot].is_none());
                        in_flight[u.slot] = Some((seg.sequence, next_tail));
                        let mut h = Hy4EngramHistory(cur_tail.to_vec());
                        hasher.hash(&mut h, tok_slice, &mut hashed[seg.sequence])?;
                    }
                    Step::Commit { unit } => {
                        let (seq, tail) = in_flight[units[unit].slot].take().expect("active wave");
                        committed[seq] = tail;
                    }
                    _ => {}
                }
            }
            assert_eq!(hashed[0], expected_rows_a);
            assert_eq!(hashed[1], expected_rows_b);
            assert_eq!(committed[0], [prompt_a[prompt_a.len() - 2], prompt_a[prompt_a.len() - 1]]);
            assert_eq!(committed[1], [prompt_b[prompt_b.len() - 2], prompt_b[prompt_b.len() - 1]]);
        }
        Ok(())
    }

    #[test]
    fn loader_admitted_attn_scratch_covers_runtime_requirement() -> Result<()> {
        let cfg = Hy4FlashConfig {
            vocab_size: 120_832,
            hidden: 3072,
            layers: 42,
            intermediate: 13_824,
            max_position: 262_144,
            rms_norm_eps: 1e-5,
            hc_mult: 4,
            hc_eps: 1e-6,
            hc_magnitude: 2.0,
            q_lora_rank: 1024,
            kv_lora_rank: 512,
            qk_nope_dim: 192,
            qk_rope_dim: 64,
            v_head_dim: 256,
            sparse_heads: 32,
            swa_heads: 64,
            swa_qk_nope_dim: 192,
            swa_qk_rope_dim: 64,
            swa_v_head_dim: 256,
            sliding_window: 512,
            sparse_rope_theta: 10_000_000.0,
            swa_rope_theta: 10_000.0,
            sparse_layer: (0..42).map(|i| i % 4 == 0).collect(),
            indexer_heads: 16,
            indexer_head_dim: 128,
            indexer_compress_ratio: 4,
            indexer_group_topk: 512,
            indexer_candidate_topk: 2048,
            dense_layer: (0..42).map(|i| i == 0).collect(),
            experts: 512,
            topk: 16,
            moe_intermediate: 1536,
            routed_expert_hidden: 1536,
            shared_experts: 1,
            n_group: 1,
            topk_group: 1,
            norm_topk_prob: true,
            routed_scaling_factor: 3.998,
            swiglu_limit: 10.0,
            engram_layers: vec![1, 11, 21, 31],
            engram_vocab_sizes: BTreeMap::new(),
            engram_max_ngram_size: 3,
            engram_n_embed_per_ngram: 768,
            engram_n_head_per_ngram: 8,
            engram_pad_id: 120_002,
            engram_seed: 10_007,
            engram_fp8_embedding: true,
            engram_lookup_table_size: 120_818,
            image_token_id: 120_120,
            video_token_id: 120_124,
            eos: vec![120_025],
        };
        for tp in [1usize, 2] {
            for decode_rows in [1usize, 32, 64] {
                let required = mla_split_scratch_bytes(decode_rows, tp);
                for unit_rows in [1usize, 64, 128, 512, 2752, 4096] {
                    let capacity_rows = unit_rows.max(decode_rows);
                    for rank in 0..tp {
                        let layout = SharedScratchLayout::new(
                            &cfg,
                            rank,
                            tp,
                            capacity_rows,
                            decode_rows,
                            64,
                            16,
                            cfg.vocab_size / tp,
                        )?;
                        let admitted = layout
                            .arena
                            .view("attn_scratch")
                            .context("missing attn_scratch view")?
                            .bytes();
                        assert!(
                            admitted >= required,
                            "tp={tp} rank={rank} unit_rows={unit_rows} decode_rows={decode_rows}: loader admitted attn_scratch {admitted} < runtime required {required}"
                        );
                    }
                }
            }
        }
        Ok(())
    }

    #[test]
    fn context_split_local_complete_groups_matches_parity_partition() {
        for pos in 0..=512usize {
            let total_complete = pos / 4;
            let mut expected = [0usize; 2];
            for g in 0..total_complete {
                let logical_page = g / (PAGE_ROWS / 4);
                expected[context::page_owner(logical_page)] += 1;
            }
            assert_eq!(local_complete_groups(pos, 0), expected[0], "pos={pos} rank=0");
            assert_eq!(local_complete_groups(pos, 1), expected[1], "pos={pos} rank=1");
            assert_eq!(
                local_complete_groups(pos, 0) + local_complete_groups(pos, 1),
                total_complete,
                "pos={pos} parity sum"
            );
        }
    }

    #[test]
    fn tp2_spark_routed_skips_rank1_duplicate_router_and_quant() {
        // Pin #2604 item 7: only rank 0 runs the Spark router, latent down and quant.
        let prod = include_str!("execution.rs").split("#[cfg(test)]").next().unwrap();
        let frontend = prod.split("fn run_moe_frontend(").nth(1).unwrap()
            .split("fn run_spark_shared_ffn(").next().unwrap();
        assert_eq!(frontend.matches("else if rank == 0 {").count(), 2);
        assert_eq!(frontend.matches("r.moe_router(&p, wave_slot, rows, routed_scale)?;").count(), 2);
    }
    #[test]
    fn spark_shared_runs_between_send_and_land() {
        let prod = include_str!("execution.rs").split("#[cfg(test)]").next().unwrap();
        let hook = prod.split("fn send_experts(").nth(1).unwrap()
            .split("fn land_experts(").next().unwrap();
        let router = hook.find("run_moe_frontend(layer, batch, false)").unwrap();
        let send = hook.find("waves.send(").unwrap();
        let shared = hook.find("run_spark_shared_ffn(layer, batch)").unwrap();
        assert!(router < send && send < shared);
        assert!(prod.contains("section: GraphSection::SparkShared"));
    }

}
