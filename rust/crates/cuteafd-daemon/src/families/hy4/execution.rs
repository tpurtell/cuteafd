//! Full 78-layer Hy4 (`hy_v4_vl`) coordinator forward pass (1x RTX TP1 and 2x RTX TP2)
//! supporting golden verification and batched multi-sequence serving through the shared
//! `run_prefill_units` (`PrefillHooks`) executor and zero-host-hop `DeviceLogits` head handoff.
//!
//! Architecture contract (#2550, verified against `python/reference/families/hy4/model.py`):
//! - 78 uniform gated DSA MLA layers (64H TP1 / 32H TP2, `q_lora=2048`, `qa_kva_eps=1e-6`,
//!   `kv_lora=512`, `qk=192+64`, `v=256`, learnable F32 sinks, elementwise sigmoid gate,
//!   adjacent-pair RoPE),
//! - 4-stream iHC (`H=6144`, FP32 mixing, NO sandwich norms: no `post_proj_norm`,
//!   no `post_fc2_norm`, no `combine_output_norm`),
//! - 32-head per-token DSA IndexCache: 21 full layers (`0, 1, 5, 9, ..., 77`) run
//!   `indexer.{wq_b, wk, weights_proj}` -> `index_post` -> `index_topk` (or ContextSplit
//!   `scored_index_topk_decode` + candidate exchange + `dsa_candidate_merge`) writing into
//!   `WaveScratch.{idx_cand_slots, idx_cand_lens}`; the 57 shared layers reuse that wave's
//!   selection directly with zero indexer compute,
//! - layer 0 dense FFN (`I=18432`, unclamped SiLU), layers 1..=77 direct MoE at `H=6144`
//!   (256 experts top-8, `I=2048`, `routed_scaling_factor=2.827`, `swiglu_limit=10.0`,
//!   1 shared expert `I=2048`, NO LatentMoE),
//! - pages-only exact prefix cache (`MarkStore::None`), rank-local CUDA graphs.
use super::cache::{slice_buf, Hy4CacheRank, Hy4Placement, Hy4Prefix};
use super::graphs::{Hy4Graphs, Key as GraphKey, Section as GraphSection};
use super::programs::{Hy4Programs, MlaSplitPlan};
use super::weights::Hy4RankWeights;
use crate::families::hy4_common::native::{Hy4NativeScratch, HY4_NATIVE_TAGS};
use crate::shared::decode_graph::device::{self as graph_device, Cycle as GraphCycle};
use crate::shared::distributed_head::Handoff;
use crate::shared::experts::rtx::exl3::Exl3Tp2;
use crate::shared::experts::rtx::{ExpertInput, Routes, RtxExpertLayer};
use crate::shared::experts::staging::RouteStaging;
use crate::shared::launch_grid::Fp8QuantizeGrid;
use crate::shared::memory::device::{Allocation, Device, Event, Stream};
use crate::shared::memory::receipts::Inventory;
use crate::shared::memory::weight_arena::DeviceArena;
use crate::shared::memory::{HostAllocation, Tag};
use crate::shared::peer_split::{
    context::{self, ContextExchange},
    PeerExchange, RankDevice, DIRECT,
};
use crate::shared::prefill_pipeline::{run_prefill_units, FfnRoute, PrefillHooks, Unit, UnitSegment};
use crate::shared::spark_intake::SparkLink;
use crate::shared::spark_waves::{self, WaveTicket, Waves};
use crate::shared::target_head::{self, TargetHead};
use crate::shared::token_io::DeviceLogits;
use anyhow::{ensure, Context, Result};
use cuteafd_core::memory_layout::{arena_class, ArenaTag};
use cuteafd_ffi::expert_usage::{ExpertUsage, UsageDevice, UsagePhase, UsageStorage};
use cuteafd_ffi::peer_exchange::PeerPlane;
use cuteafd_ffi::programs::{Program, Programs, Scalar};
use cuteafd_ffi::{Bf16Add, CuteafdDeviceBuffer, CuteafdHostBuffer, F32Add};
use cuteafd_loader::families::hy4::execution::{
    hy4_context_buffers, hy4_dsa_partial_scratch_bytes, hy4_peer_ffn_routed_offset,
    hy4_peer_slot_sizes, mla_split_scratch_bytes, wave_metadata_layout_for_rank, IndexTopkScratch,
    SharedScratchLayout, WaveScratchLayout, WAVE_METADATA,
};
use cuteafd_loader::families::hy4::{Hy4Config, Hy4KvCache};
use cuteafd_loader::families::hy4_common::PAGE_ROWS;
use cuteafd_loader::formats::mapped_table::ReadPriority;
use cuteafd_loader::formats::trellis_core::ProjectionScratchLayout;
use cuteafd_loader::placement::families::hy4::DECODE_ROWS;
use cuteafd_loader::placement::AttentionPlacement;
use cuteafd_loader::plan::checkpoint::Checkpoint;
use cuteafd_loader::serving_capacity::WeightArenaLayout;
use cuteafd_transport::{
    expert::EXPERT_PROTOCOL_V2_FLAG_V41_COMPACT_BF16, ExpertProtocolV2Request,
    ExpertProtocolV2RouteEntry, ExpertProtocolV2RowDescriptor, ExpertV2Dtype, ExpertV2SourceKind,
};
use std::cell::{Cell, RefCell};
use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;
use std::rc::Rc;

const H: usize = 6144;
const HC: usize = 4;
const Q_LORA: usize = 2048;
const KV_A_DIM: usize = 576;
const KV_A_PADDED: usize = 640;
const EXPERTS: usize = 256;
const TOPK: usize = 8;
const IDX_HEADS: usize = 32;
const IDX_DIM: usize = 128;
const CANDIDATE_TOPK: usize = 2048;
const PEER_SLOT_ATTN: usize = 0;
const PEER_SLOT_FFN: usize = 1;

/// Which rows of a batch compute terminal `hc_head` + `lm_head` FP32 logits.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum HeadMode {
    None,
    LastPerSeq,
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
            ptr: experts
                .as_ref()
                .context("Hy4 TP2 routed expert output missing")?
                .output(),
            bytes: rows * H * 4,
            device_id: rank.device.id,
            flags: 0,
        })
    }
}

/// One packed per-wave scratch arena per coordinator GPU rank holding every buffer
/// that remains live across `PrefillHooks` boundaries for that wave slot, including
/// `idx_cand_slots` & `idx_cand_lens` for multi-wave-safe IndexCache sharing.
struct WaveScratch<'a> {
    _arena: DeviceArena<'a, arena_class::ReadWrite>,
    positions_i64: Allocation<'a>,
    positions_i32: Allocation<'a>,
    dsa_slots_i64: Allocation<'a>,
    r_a: Allocation<'a>,
    r_b: Allocation<'a>,
    post_ffn: Allocation<'a>,
    y: Allocation<'a>,
    shared_out: Allocation<'a>,
    shared_f32: Allocation<'a>,
    dsa_page_table: Allocation<'a>,
    idx_seq_page_tables: Allocation<'a>,
    dsa_cu_seqlens: Allocation<'a>,
    idx_cache_lens: Allocation<'a>,
    idx_cand_slots: Allocation<'a>,
    idx_cand_lens: Allocation<'a>,
}

impl<'a> WaveScratch<'a> {
    #[allow(clippy::too_many_arguments)]
    fn new(
        device: Device<'a>,
        cfg: &Hy4Config,
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
            ArenaTag::workspace("hy4/wave-scratch"),
        )?;
        let view = |name: &str| -> Result<Allocation<'a>> {
            let bytes = arena
                .layout()
                .view(name)
                .with_context(|| format!("Hy4 wave scratch missing view {name}"))?
                .bytes();
            arena.view(name, bytes)
        };
        Ok(Self {
            positions_i64: view("positions_i64")?,
            positions_i32: view("positions_i32")?,
            dsa_slots_i64: view("dsa_slots_i64")?,
            r_a: view("r_a")?,
            r_b: view("r_b")?,
            post_ffn: view("post_ffn")?,
            y: view("y")?,
            shared_out: view("shared_out")?,
            shared_f32: view("shared_f32")?,
            dsa_page_table: view("dsa_page_table")?,
            idx_seq_page_tables: view("idx_seq_page_tables")?,
            dsa_cu_seqlens: view("dsa_cu_seqlens")?,
            idx_cache_lens: view("idx_cache_lens")?,
            idx_cand_slots: view("idx_cand_slots")?,
            idx_cand_lens: view("idx_cand_lens")?,
            _arena: arena,
        })
    }

    fn inventory(&self, inventory: &mut Inventory<'_>, rank: usize, slot: usize) -> Result<()> {
        let prefix = format!("hy4.wave.{rank}.{slot}");
        for (name, alloc) in [
            ("positions_i64", &self.positions_i64),
            ("positions_i32", &self.positions_i32),
            ("dsa_slots_i64", &self.dsa_slots_i64),
            ("r_a", &self.r_a),
            ("r_b", &self.r_b),
            ("post_ffn", &self.post_ffn),
            ("y", &self.y),
            ("shared_out", &self.shared_out),
            ("shared_f32", &self.shared_f32),
            ("dsa_page_table", &self.dsa_page_table),
            ("idx_seq_page_tables", &self.idx_seq_page_tables),
            ("dsa_cu_seqlens", &self.dsa_cu_seqlens),
            ("idx_cache_lens", &self.idx_cache_lens),
            ("idx_cand_slots", &self.idx_cand_slots),
            ("idx_cand_lens", &self.idx_cand_lens),
        ] {
            alloc.inventory(inventory, &format!("{prefix}.{name}"))?;
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
        let backing = HostAllocation::new(
            devices[0].library,
            layout.bytes(),
            Tag::staging("hy4/wave-metadata"),
        )?;
        Ok(Self {
            backing,
            layout,
            completion,
            pending: false,
        })
    }

    fn upload(
        &mut self,
        ranks: &[RankRuntime<'a>],
        slot: usize,
        planes: [&[u8]; WAVE_METADATA.len()],
    ) -> Result<()> {
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
            let view = self
                .layout
                .view(name)
                .with_context(|| format!("Hy4 missing metadata view {name}"))?;
            ensure!(
                contents.len() <= view.bytes(),
                "Hy4 metadata {name} exceeds its admitted view"
            );
            let at = view.offset();
            bytes[at..at + contents.len()].copy_from_slice(contents);
            sources[count] = CuteafdHostBuffer {
                ptr: source.ptr.wrapping_byte_add(at),
                bytes: contents.len(),
                ..source
            };
            lengths[count] = contents.len();
            count += 1;
        }
        self.pending = true;
        for (r, event) in ranks.iter().zip(&mut self.completion) {
            let wave = &r.waves[slot];
            let mut dst = [CuteafdDeviceBuffer::default(); WAVE_METADATA.len()];
            let bufs = [
                wave.positions_i64.buffer,
                wave.positions_i32.buffer,
                wave.dsa_slots_i64.buffer,
                wave.dsa_page_table.buffer,
                wave.dsa_cu_seqlens.buffer,
                wave.idx_cache_lens.buffer,
                wave.idx_seq_page_tables.buffer,
            ];
            for (target, buffer) in dst.iter_mut().zip(
                bufs.into_iter()
                    .zip(planes)
                    .filter_map(|(buffer, plane)| (!plane.is_empty()).then_some(buffer)),
            ) {
                *target = buffer;
            }
            r.device.run(|| {
                // SAFETY: pinned source is private to this slot and stays immutable until events complete.
                unsafe {
                    r.device.library.copy_host_buffers_h2d_batch_async(
                        &dst[..count],
                        &sources[..count],
                        &lengths[..count],
                        r.stream.raw,
                    )
                }
            })?;
            event.record(&r.stream)?;
        }
        Ok(())
    }

    fn retire(&mut self) -> Result<()> {
        if self.pending {
            for event in &self.completion {
                // SAFETY: slot's recorded completion events.
                event
                    .device
                    .run(|| unsafe { event.device.library.cuda_event_synchronize(event.raw) })?;
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
    merged_scores: Allocation<'a>,
    merged_indices: Allocation<'a>,
    partial_own: Allocation<'a>,
    lse_own: Allocation<'a>,
    partial_peer: Allocation<'a>,
    lse_peer: Allocation<'a>,
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
    idx_k_raw: Allocation<'a>,
    idx_w_raw: Allocation<'a>,
    idx_q_fp8: Allocation<'a>,
    idx_w_out: Allocation<'a>,
    idx_topk_scratch: Allocation<'a>,
    ctx: Option<ContextScratch<'a>>,
    ffn_gate: Allocation<'a>,
    ffn_up: Allocation<'a>,
    ffn_act: Allocation<'a>,
    router_logits: Allocation<'a>,
    router_select: Allocation<'a>,
    route_ids: Allocation<'a>,
    route_weights: Allocation<'a>,
    wire_in: Allocation<'a>,
    routed_out: Allocation<'a>,
    logits_tile: Allocation<'a>,
    head_logits_f32: Option<Allocation<'a>>,
    native: Option<Hy4NativeScratch<'a>>,
}

impl<'a> SharedScratch<'a> {
    #[allow(clippy::too_many_arguments)]
    fn new(
        device: Device<'a>,
        stream: &Stream<'a>,
        cfg: &Hy4Config,
        rank: usize,
        tp: usize,
        max_rows: usize,
        decode_rows: usize,
        local_vocab: usize,
        attn_scratch_bytes: usize,
        context_split: bool,
        index_topk: IndexTopkScratch,
        native_layout: Option<&ProjectionScratchLayout>,
        native_head_layout: Option<&ProjectionScratchLayout>,
    ) -> Result<Self> {
        let layout = SharedScratchLayout::new(
            cfg,
            rank,
            tp,
            max_rows,
            decode_rows,
            local_vocab,
            context_split,
            index_topk,
        )?;
        let arena = DeviceArena::new(
            device,
            layout.arena.clone(),
            ArenaTag::workspace("hy4/rank-scratch"),
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
        let idx_topk_scratch = view("idx_topk_scratch")?;
        // `glm_index_topk` / `scored_index_topk_decode` require zero-filled scratch once at allocation.
        device.run(|| unsafe {
            device
                .library
                .cuda_zero_bytes_async(idx_topk_scratch.buffer, idx_topk_scratch.buffer.bytes, stream.raw)
        })?;
        stream.drain()?;

        let ctx = if context_split {
            Some(ContextScratch {
                cand_scores: view("ctx_cand_scores")?,
                cand_indices: view("ctx_cand_indices")?,
                merged_scores: view("ctx_merged_scores")?,
                merged_indices: view("ctx_merged_indices")?,
                partial_own: view("ctx_partial_own")?,
                lse_own: view("ctx_lse_own")?,
                partial_peer: view("ctx_partial_peer")?,
                lse_peer: view("ctx_lse_peer")?,
            })
        } else {
            None
        };
        let native = if native_layout.is_some() || native_head_layout.is_some() {
            Some(Hy4NativeScratch::allocate_with(
                device,
                HY4_NATIVE_TAGS,
                native_layout,
                native_head_layout,
            )?)
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
            idx_k_raw: view("idx_k_raw")?,
            idx_w_raw: view("idx_w_raw")?,
            idx_q_fp8: view("idx_q_fp8")?,
            idx_w_out: view("idx_w_out")?,
            idx_topk_scratch,
            ctx,
            ffn_gate: view("ffn_gate")?,
            ffn_up: view("ffn_up")?,
            ffn_act: view("ffn_act")?,
            router_logits: view("router_logits")?,
            router_select: view("router_select")?,
            route_ids: view("route_ids")?,
            route_weights: view("route_weights")?,
            wire_in: view("wire_in")?,
            routed_out: view("routed_out")?,
            logits_tile: view("logits_tile")?,
            head_logits_f32,
            native,
            _arena: arena,
        })
    }

    fn inventory(&self, inventory: &mut Inventory<'_>, rank: usize) -> Result<()> {
        let prefix = format!("hy4.scratch.{rank}");
        for (name, alloc) in [
            ("ids", &self.ids),
            ("r_head_in", &self.r_head_in),
            ("post", &self.post),
            ("delta", &self.delta),
            ("delta_f32", &self.delta_f32),
            ("delta_sum_f32", &self.delta_sum_f32),
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
            ("idx_k_raw", &self.idx_k_raw),
            ("idx_w_raw", &self.idx_w_raw),
            ("idx_q_fp8", &self.idx_q_fp8),
            ("idx_w_out", &self.idx_w_out),
            ("idx_topk_scratch", &self.idx_topk_scratch),
            ("ffn_gate", &self.ffn_gate),
            ("ffn_up", &self.ffn_up),
            ("ffn_act", &self.ffn_act),
            ("router_logits", &self.router_logits),
            ("router_select", &self.router_select),
            ("route_ids", &self.route_ids),
            ("route_weights", &self.route_weights),
            ("wire_in", &self.wire_in),
            ("routed_out", &self.routed_out),
            ("logits_tile", &self.logits_tile),
        ] {
            alloc.inventory(inventory, &format!("{prefix}.{name}"))?;
        }
        if let Some(ctx) = &self.ctx {
            for (name, alloc) in [
                ("ctx_cand_scores", &ctx.cand_scores),
                ("ctx_cand_indices", &ctx.cand_indices),
                ("ctx_merged_scores", &ctx.merged_scores),
                ("ctx_merged_indices", &ctx.merged_indices),
                ("ctx_partial_own", &ctx.partial_own),
                ("ctx_lse_own", &ctx.lse_own),
                ("ctx_partial_peer", &ctx.partial_peer),
                ("ctx_lse_peer", &ctx.lse_peer),
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
        let scratch = r
            .shared
            .native
            .as_ref()
            .and_then(|n| n.scratch.as_ref())
            .context("Hy4 native core missing scratch")?;
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
        let scratch = r
            .shared
            .native
            .as_ref()
            .and_then(|n| n.scratch.as_ref())
            .context("Hy4 native core missing scratch")?;
        linear.launch(rows, input, r.shared.kv_a_padded.buffer, scratch, r.stream.raw)?;
        // SAFETY: kv_a_padded [rows, 640] BF16 and output_576 [rows, 576] BF16 are disjoint.
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

#[allow(clippy::too_many_arguments)]
fn launch_proj_split(
    r: &RankRuntime<'_>,
    gpus: usize,
    stem: &str,
    bf16_prog: &Program<'_>,
    rows: usize,
    input: CuteafdDeviceBuffer,
    out_bf16: CuteafdDeviceBuffer,
    out_f32: CuteafdDeviceBuffer,
) -> Result<()> {
    if let Some(linear) = r.weights.native.get(stem) {
        let scratch = r
            .shared
            .native
            .as_ref()
            .and_then(|n| n.scratch.as_ref())
            .context("Hy4 native core missing scratch")?;
        return if gpus == 1 {
            linear.launch(rows, input, out_bf16, scratch, r.stream.raw)
        } else {
            linear.partial(rows, input, out_f32, scratch, r.stream.raw)
        };
    }
    let out = if gpus == 1 { out_bf16 } else { out_f32 };
    // SAFETY: caller ensures device active and all buffers live through stream drain.
    unsafe {
        bf16_prog.launch(
            &[input.ptr, r.weights.weight(stem)?.ptr, out.ptr],
            &[Scalar::I32(rows as i32)],
            r.stream.raw,
        )
    }
}

fn launch_rank_lm_head(
    r: &RankRuntime<'_>,
    count: usize,
    input: CuteafdDeviceBuffer,
    out_f32: CuteafdDeviceBuffer,
) -> Result<()> {
    if let Some(linear) = r.weights.native.get("lm_head") {
        let scratch = r
            .shared
            .native
            .as_ref()
            .and_then(|n| n.head_scratch.as_ref())
            .context("Hy4 native head missing head_scratch")?;
        return linear.launch(count, input, out_f32, scratch, r.stream.raw);
    }
    let head = r.weights.head.as_ref().context("Hy4 rank missing BF16 VocabularyHead")?;
    let head_w = r.weights.weight("lm_head")?;
    // SAFETY: weights, input, and out_f32 are admitted device buffers alive through stream drain.
    unsafe {
        target_head::launch(
            TargetHead::configured(),
            head,
            input.ptr,
            head_w.ptr,
            out_f32.ptr.cast(),
            count,
            r.stream.raw,
        )
    }
}

struct RankRuntime<'a> {
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
    quantize_grid: Fp8QuantizeGrid,
    onboarded_experts: RefCell<Option<Exl3Tp2<'a>>>,
}

impl<'a> RankRuntime<'a> {
    /// Attention iHC pre + MLA Q/KV projections + rank_norm_pack + q_b + q_absorb + linear_gate.
    /// When `full_indexer` is true, also runs `indexer.{wq_b, wk, weights_proj}` + `index_post`.
    #[allow(clippy::too_many_arguments)]
    fn attn_pre(
        &self,
        p: &str,
        layer: usize,
        full_idx: Option<usize>,
        local_heads: usize,
        wave_slot: usize,
        rows: usize,
        use_a: bool,
        kv_cache_buf: CuteafdDeviceBuffer,
    ) -> Result<()> {
        let w = &self.waves[wave_slot];
        let r_cur = if use_a { w.r_a.buffer } else { w.r_b.buffer };
        let s_rows = [Scalar::I32(rows as i32)];
        let (w_uk, _) = self.weights.kv_b_absorbed(layer, local_heads)?;
        let inv_freq = self.weights.inv_freq.buffer.ptr;

        // SAFETY: arena weights, cache, and scratch live through stream drain.
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
                &s_rows,
                self.stream.raw,
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
                    w.dsa_slots_i64.buffer.ptr,
                    w.positions_i64.buffer.ptr,
                    inv_freq,
                    self.shared.q_resid.buffer.ptr,
                    kv_cache_buf.ptr,
                ],
                &s_rows,
                self.stream.raw,
            )?;
            launch_proj_out(
                self,
                &format!("{p}.self_attn.q_b_proj"),
                &self.programs.proj_q_b_dsa,
                rows,
                self.shared.q_resid.buffer,
                self.shared.q_raw.buffer,
            )?;
            self.programs.mla_q_absorb_dsa.launch(
                &[
                    self.shared.q_raw.buffer.ptr,
                    w_uk,
                    w.positions_i64.buffer.ptr,
                    inv_freq,
                    self.shared.query.buffer.ptr,
                ],
                &s_rows,
                self.stream.raw,
            )?;
            launch_proj_out(
                self,
                &format!("{p}.self_attn.linear_gate"),
                &self.programs.proj_linear_gate_dsa,
                rows,
                w.y.buffer,
                self.shared.gate.buffer,
            )?;
            if let Some(fidx) = full_idx {
                let ip = format!("{p}.self_attn.indexer");
                let idx_cache = self.cache.dsa_index_layer(fidx)?;
                self.programs.proj_indexer_wq_b.launch(
                    &[
                        self.shared.q_resid.buffer.ptr,
                        self.weights.weight(&format!("{ip}.wq_b"))?.ptr,
                        self.shared.idx_q_raw.buffer.ptr,
                    ],
                    &s_rows,
                    self.stream.raw,
                )?;
                self.programs.proj_indexer_wk.launch(
                    &[
                        w.y.buffer.ptr,
                        self.weights.weight(&format!("{ip}.wk"))?.ptr,
                        self.shared.idx_k_raw.buffer.ptr,
                    ],
                    &s_rows,
                    self.stream.raw,
                )?;
                self.programs.proj_indexer_weights.launch(
                    &[
                        w.y.buffer.ptr,
                        self.weights.weight(&format!("{ip}.weights_proj"))?.ptr,
                        self.shared.idx_w_raw.buffer.ptr,
                    ],
                    &s_rows,
                    self.stream.raw,
                )?;
                self.programs.index_post.launch(
                    &[
                        self.shared.idx_q_raw.buffer.ptr,
                        self.shared.idx_k_raw.buffer.ptr,
                        self.shared.idx_w_raw.buffer.ptr,
                        w.positions_i64.buffer.ptr,
                        w.dsa_slots_i64.buffer.ptr,
                        inv_freq,
                        self.weights.buf(&format!("{ip}.k_norm.weight"))?.ptr,
                        self.weights.buf(&format!("{ip}.k_norm.bias"))?.ptr,
                        self.shared.idx_q_fp8.buffer.ptr,
                        self.shared.idx_w_out.buffer.ptr,
                        idx_cache.ptr,
                    ],
                    &s_rows,
                    self.stream.raw,
                )?;
            }
        }
        Ok(())
    }

    fn dsa_attn(
        &self,
        kv_cache_buf: CuteafdDeviceBuffer,
        wave_slot: usize,
        rows: usize,
        decode_attention: bool,
    ) -> Result<()> {
        let w = &self.waves[wave_slot];
        let splits = if decode_attention { self.mla_splits.dsa } else { 1 };
        // SAFETY: arena scratch and wave IndexCache selection live through stream drain.
        unsafe {
            self.programs.mla_dsa_decode.launch(
                &[
                    self.shared.query.buffer.ptr,
                    kv_cache_buf.ptr,
                    w.idx_cand_slots.buffer.ptr,
                    w.idx_cand_lens.buffer.ptr,
                    w.dsa_cu_seqlens.buffer.ptr,
                    self.shared.attn_out.buffer.ptr,
                    self.shared.lse_out.buffer.ptr,
                    self.shared.attn_scratch.buffer.ptr,
                ],
                &[
                    Scalar::I32(rows as i32),
                    Scalar::I32(rows as i32),
                    Scalar::I64(CANDIDATE_TOPK as i64),
                    Scalar::I32(splits),
                ],
                self.stream.raw,
            )
        }
    }

    /// ContextSplit dual 32-head partials (`h0` + `h32`) over parity-owned pages for `c_rows <= decode_rows`.
    #[allow(clippy::too_many_arguments)]
    fn dsa_attn_context_partials(
        &self,
        rank: usize,
        assembled_q: *mut std::ffi::c_void,
        kv_cache_buf: CuteafdDeviceBuffer,
        wave_slot: usize,
        chunk_start: usize,
        c_rows: usize,
        decode_attention: bool,
    ) -> Result<()> {
        let ctx_progs = self
            .programs
            .context
            .as_ref()
            .context("Hy4 ContextSplit programs missing")?;
        let ctx = self
            .shared
            .ctx
            .as_ref()
            .context("Hy4 ContextSplit scratch missing")?;
        let w = &self.waves[wave_slot];
        let splits = if decode_attention {
            self.mla_splits.dsa_partial
        } else {
            2
        };
        let (own_prog, peer_prog) = if rank == 0 {
            (&ctx_progs.dsa_partial_h0, &ctx_progs.dsa_partial_h32)
        } else {
            (&ctx_progs.dsa_partial_h32, &ctx_progs.dsa_partial_h0)
        };
        let slots_ptr = w
            .idx_cand_slots
            .buffer
            .ptr
            .cast::<u8>()
            .wrapping_add(chunk_start * CANDIDATE_TOPK * 4)
            .cast();
        let lens_ptr = w
            .idx_cand_lens
            .buffer
            .ptr
            .cast::<u8>()
            .wrapping_add(chunk_start * 4)
            .cast();
        let scalars = [
            Scalar::I32(c_rows as i32),
            Scalar::I64(CANDIDATE_TOPK as i64),
            Scalar::I32(splits),
        ];
        // SAFETY: assembled_q came from ContextExchange::wait_query; scratch lives in rank arena.
        unsafe {
            own_prog.launch(
                &[
                    assembled_q,
                    kv_cache_buf.ptr,
                    slots_ptr,
                    lens_ptr,
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
                    assembled_q,
                    kv_cache_buf.ptr,
                    slots_ptr,
                    lens_ptr,
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

    fn dsa_context_combine(
        &self,
        rank: usize,
        p: &str,
        recv_p: *mut std::ffi::c_void,
        recv_l: *mut std::ffi::c_void,
        chunk_start: usize,
        c_rows: usize,
    ) -> Result<()> {
        let ctx_progs = self
            .programs
            .context
            .as_ref()
            .context("Hy4 ContextSplit programs missing")?;
        let ctx = self
            .shared
            .ctx
            .as_ref()
            .context("Hy4 ContextSplit scratch missing")?;
        let (o0, l0, o1, l1) = if rank == 0 {
            (ctx.partial_own.buffer.ptr, ctx.lse_own.buffer.ptr, recv_p, recv_l)
        } else {
            (recv_p, recv_l, ctx.partial_own.buffer.ptr, ctx.lse_own.buffer.ptr)
        };
        let out_ptr = self
            .shared
            .attn_out
            .buffer
            .ptr
            .cast::<u8>()
            .wrapping_add(chunk_start * 32 * 512 * 2)
            .cast();
        let sink = self
            .weights
            .buf(&format!("{p}.self_attn.learnable_sink_param"))?
            .ptr;
        // SAFETY: partials finished on stream before combine; disjoint output slice.
        unsafe {
            ctx_progs.lse_combine2.launch(
                &[o0, l0, o1, l1, sink, out_ptr],
                &[Scalar::I32(c_rows as i32)],
                self.stream.raw,
            )
        }
    }

    fn dsa_context_gate_uv_o(&self, p: &str, layer: usize, rows: usize) -> Result<()> {
        let ctx_progs = self
            .programs
            .context
            .as_ref()
            .context("Hy4 ContextSplit programs missing")?;
        let (_, w_uv) = self.weights.kv_b_absorbed(layer, 32)?;
        // SAFETY: arena weights and scratch live through stream drain.
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

    fn attn_out(&self, p: &str, layer: usize, gpus: usize, local_heads: usize, rows: usize) -> Result<()> {
        let (_, w_uv) = self.weights.kv_b_absorbed(layer, local_heads)?;
        let sink = self
            .weights
            .buf(&format!("{p}.self_attn.learnable_sink_param"))?
            .ptr;
        // SAFETY: arena weights and scratch live through stream drain.
        unsafe {
            self.programs.mla_sink_gate_uv_dsa.launch(
                &[
                    self.shared.attn_out.buffer.ptr,
                    w_uv,
                    self.shared.gate.buffer.ptr,
                    self.shared.lse_out.buffer.ptr,
                    sink,
                    self.shared.uv_out.buffer.ptr,
                ],
                &[Scalar::I32(rows as i32)],
                self.stream.raw,
            )?;
        }
        launch_proj_split(
            self,
            gpus,
            &format!("{p}.self_attn.o_proj"),
            &self.programs.proj_o_dsa,
            rows,
            self.shared.uv_out.buffer,
            self.shared.delta.buffer,
            self.shared.delta_f32.buffer,
        )
    }

    /// Post-attention iHC update + FFN iHC pre (Hy4 has NO `post_proj_norm`).
    fn post_attn_ihc(&self, p: &str, wave_slot: usize, rows: usize, use_a: bool) -> Result<()> {
        let w = &self.waves[wave_slot];
        let (r_in, r_out) = if use_a {
            (w.r_a.buffer, w.r_b.buffer)
        } else {
            (w.r_b.buffer, w.r_a.buffer)
        };
        // SAFETY: arena weights and scratch live through stream drain.
        unsafe {
            self.programs.ihc_post_pre.launch(
                &[
                    self.shared.delta.buffer.ptr,
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
                &[Scalar::I32(rows as i32)],
                self.stream.raw,
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
        // SAFETY: scratch buffers live through stream drain.
        unsafe {
            self.programs.post_silu_split_dense.launch(
                &[
                    self.shared.ffn_gate.buffer.ptr,
                    self.shared.ffn_up.buffer.ptr,
                    self.shared.ffn_act.buffer.ptr,
                ],
                &s_rows,
                self.stream.raw,
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

    /// Direct MoE router + FP8 K32 quant on `w.y` (`H=6144`, no LatentMoE) + shared expert FFN.
    fn moe_router_shared(
        &self,
        p: &str,
        gpus: usize,
        wave_slot: usize,
        rows: usize,
        routed_scaling_factor: f32,
        needs_router: bool,
    ) -> Result<()> {
        let w = &self.waves[wave_slot];
        let s_rows = [Scalar::I32(rows as i32)];
        // SAFETY: all buffers live through stream drain.
        unsafe {
            if needs_router {
                self.programs.proj_router.launch(
                    &[
                        w.y.buffer.ptr,
                        self.weights.weight(&format!("{p}.mlp.gate"))?.ptr,
                        self.shared.router_logits.buffer.ptr,
                    ],
                    &s_rows,
                    self.stream.raw,
                )?;
                let bias = self
                    .weights
                    .buf(&format!("{p}.mlp.gate.e_score_correction_bias"))?
                    .ptr;
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
                    &[
                        self.shared.router_logits.buffer.ptr,
                        bias,
                        self.shared.route_ids.buffer.ptr,
                        self.shared.route_weights.buffer.ptr,
                    ],
                    &s_rows,
                    self.stream.raw,
                )?;
                let grid = self.quantize_grid.blocks(rows, H);
                self.programs.expert_input_quant.launch(
                    &[
                        w.y.buffer.ptr,
                        self.shared.wire_in.buffer.ptr,
                        self.shared.wire_in.buffer.ptr.wrapping_byte_add(H),
                        self.shared.wire_in.buffer.ptr,
                    ],
                    &[Scalar::I32(rows as i32), Scalar::I32(grid as i32)],
                    self.stream.raw,
                )?;
            }
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
                &[
                    self.shared.ffn_gate.buffer.ptr,
                    self.shared.ffn_up.buffer.ptr,
                    self.shared.ffn_act.buffer.ptr,
                ],
                &s_rows,
                self.stream.raw,
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

    fn local_exl3(
        &self,
        layer: usize,
        gpus: usize,
        rows: usize,
        skip_routed_experts: bool,
    ) -> Result<()> {
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
                            bytes: rows * H * 4,
                            device_id: self.device.id,
                            flags: 0,
                        };
                        self.device.library.cuda_f32_to_bf16_async(
                            exp_out,
                            self.shared.routed_out.buffer,
                            rows * H,
                            self.stream.raw,
                        )?;
                    }
                } else {
                    anyhow::bail!(
                        "Hy4 layer {layer}: no onboarded hy4:exl3-k45-mul1 expert package loaded \
                         (pass --peers, --expert-package, or --skip-routed-experts)"
                    );
                }
            } else {
                self.device.library.cuda_zero_bytes_async(
                    self.shared.routed_out.buffer,
                    rows * H * 2,
                    self.stream.raw,
                )?;
                if gpus == 2 {
                    self.device.library.cuda_zero_bytes_async(
                        self.shared.delta_f32.buffer,
                        rows * H * 4,
                        self.stream.raw,
                    )?;
                }
            }
        }
        Ok(())
    }

    /// Combine routed + shared (if sparse) and apply `ihc_post` directly (Hy4 has NO sandwich norms).
    fn ffn_combine_post(
        &self,
        is_dense: bool,
        wave_slot: usize,
        rows: usize,
        use_a: bool,
    ) -> Result<()> {
        let w = &self.waves[wave_slot];
        if !is_dense {
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
        let (r_in, r_out) = if use_a {
            (w.r_a.buffer, w.r_b.buffer)
        } else {
            (w.r_b.buffer, w.r_a.buffer)
        };
        // SAFETY: scratch buffers live through stream drain.
        unsafe {
            self.programs.ihc_post.launch(
                &[
                    self.shared.delta.buffer.ptr,
                    r_in.ptr,
                    w.post_ffn.buffer.ptr,
                    r_out.ptr,
                ],
                &[Scalar::I32(rows as i32)],
                self.stream.raw,
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
        ensure!(
            local_f32.bytes >= bytes && !peer_f32_ptr.is_null(),
            "Hy4 TP2 local reduce operand invalid"
        );
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
            self.f32_add
                .launch(a, b, self.shared.delta_sum_f32.buffer, elems, self.stream.raw)?;
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
                        H,
                        None,
                        routed_in,
                        ffn_recv_ptr.wrapping_byte_add(routed_recv_offset),
                        self.shared.routed_out.buffer,
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
        self.ffn_combine_post(prev.is_dense, wave_slot, rows, use_a)
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
        // SAFETY: router wrote route_ids on this stream; engine drains before dropping counters.
        unsafe {
            tally.record(
                self.shared.route_ids.buffer.ptr,
                rows,
                None,
                u32::MAX,
                true,
                self.stream.raw,
            )
        }
    }
}

#[derive(Clone, Debug)]
struct ActiveMember {
    sequence: usize,
    slot: usize,
    pos_start: usize,
    row_start: usize,
    row_len: usize,
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
    pt_stride: usize,
    ctx_pt_stride: usize,
    ctx_slot_counter: usize,
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
}

/// Coordinator Hy4 execution engine (1x RTX TP1 or 2x RTX TP2, 0-Spark or Spark-backed).
pub(crate) struct Hy4Engine<'a> {
    pub(crate) cfg: Hy4Config,
    pub(crate) max_rows: usize,
    pub(crate) decode_rows: usize,
    pub(crate) prefill_waves: usize,
    pub(crate) max_pages: usize,
    pub(crate) max_sequences: usize,
    pub(crate) max_context: usize,
    pub(crate) skip_routed_experts: bool,
    pub(crate) dump_dir: Option<std::path::PathBuf>,
    context_split: bool,
    admitted_pages: usize,
    graphs_enabled: Cell<bool>,
    graph_cycles: RefCell<Option<graph_device::EventCycles<'a>>>,
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
    full_index_of: BTreeMap<usize, usize>,
    golden_placement: Hy4Placement,
    expert_usage: Option<ExpertUsage<'a>>,
    count_expert_usage: bool,
    terminal: bool,
}

impl<'a> Hy4Engine<'a> {
    /// Single-sequence constructor used by `hy4-golden`.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        devices: &[Device<'a>],
        programs: &'a Programs<'a>,
        manifest_path: Option<&Path>,
        checkpoint: &Checkpoint,
        cfg: Hy4Config,
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
            manifest_path,
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
            skip_routed_experts,
            onboarded_experts,
            Default::default(),
        )
    }

    /// Serving & multi-sequence constructor with pages-only COW pool and packed scratch arenas
    /// matching `Hy4Inputs` admission.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new_serving(
        devices: &[Device<'a>],
        programs: &'a Programs<'a>,
        mut preloaded_programs: Option<Vec<Hy4Programs<'a>>>,
        manifest_path: Option<&Path>,
        checkpoint: &Checkpoint,
        cfg: Hy4Config,
        kv_format: Hy4KvCache,
        attention: AttentionPlacement,
        unit_rows: usize,
        decode_rows: usize,
        prefill_waves: usize,
        max_pages: usize,
        max_sequences: usize,
        max_context: usize,
        admitted_context: usize,
        skip_routed_experts: bool,
        mut onboarded_experts: Vec<Option<Exl3Tp2<'a>>>,
        media: crate::shared::vision::push::Contribution,
    ) -> Result<Self> {
        ensure!(
            (1..=DECODE_ROWS as usize).contains(&decode_rows),
            "Hy4 split-K decode exports cover at most {DECODE_ROWS} rows; got {decode_rows}"
        );
        let max_rows = unit_rows.max(decode_rows);
        let gpus = devices.len();
        ensure!(matches!(gpus, 1 | 2), "Hy4Engine supports 1 or 2 GPUs, got {gpus}");
        let context_split = match (gpus, attention) {
            (1, AttentionPlacement::Heads) => false,
            (2, AttentionPlacement::Heads) => false,
            (2, AttentionPlacement::Context) => true,
            _ => anyhow::bail!(
                "Hy4Engine does not support attention placement {attention:?} on {gpus} GPU(s)"
            ),
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
        let head_rows = max_rows.min(DECODE_ROWS as usize);
        let index_topk = IndexTopkScratch::resolve(manifest_path, gpus, context_split, admitted_context)?;
        let weights_all =
            Hy4RankWeights::load_all(devices, programs, checkpoint, &cfg, max_rows, head_rows, media)?;
        let max_seq_pages = admitted_context.div_ceil(PAGE_ROWS).max(16);

        let metadata_uploads = if context_split {
            (0..gpus)
                .map(|rank| {
                    (0..prefill_waves)
                        .map(|slot| {
                            let wave = WaveScratchLayout::new(
                                &cfg,
                                rank,
                                slot,
                                gpus,
                                max_rows,
                                max_seq_pages,
                                max_sequences,
                            )?;
                            MetadataUpload::new(
                                &[devices[rank]],
                                wave_metadata_layout_for_rank(rank, slot, &wave.arena)?,
                            )
                        })
                        .collect::<Result<Vec<_>>>()
                })
                .collect::<Result<Vec<_>>>()?
        } else {
            vec![(0..prefill_waves)
                .map(|slot| {
                    let wave = WaveScratchLayout::new(
                        &cfg,
                        0,
                        slot,
                        gpus,
                        max_rows,
                        max_seq_pages,
                        max_sequences,
                    )?;
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
            let attn_scratch = mla_split_scratch_bytes(decode_rows, gpus)
                .max(if context_split { hy4_dsa_partial_scratch_bytes(decode_rows) } else { 0 });
            let cache = Rc::new(Hy4CacheRank::new(
                vec![stream.clone()],
                &cfg,
                kv_format,
                max_pages,
                max_sequences,
            )?);
            let native_layout = weights.native.scratch_layout(max_rows)?;
            let native_head_layout = weights.native.head_scratch_layout(head_rows)?;
            let shared = SharedScratch::new(
                device,
                &stream,
                &cfg,
                rank,
                gpus,
                max_rows,
                decode_rows,
                weights.local_vocab,
                attn_scratch,
                context_split,
                index_topk,
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
                quantize_grid,
                onboarded_experts: RefCell::new(onboarded_experts[rank].take()),
            });
        }

        let ffn_routed_recv_offset = hy4_peer_ffn_routed_offset(&cfg, max_rows);
        let peer_exchange = if gpus == 2 {
            let sizes = hy4_peer_slot_sizes(&cfg, max_rows);
            let decode_lane_bytes = (decode_rows * HC * H * 2).max(decode_rows * 2 * H * 4);
            let exchange = PeerExchange::new_abortable_sized(
                devices[0].library,
                [
                    RankDevice {
                        device: ranks[0].device.id,
                        stream: ranks[0].stream.raw,
                    },
                    RankDevice {
                        device: ranks[1].device.id,
                        stream: ranks[1].stream.raw,
                    },
                ],
                &sizes,
                decode_lane_bytes,
            )?;
            exchange.trace_identity("hy4.tp2");
            Some(exchange)
        } else {
            None
        };
        let context_exchange = if context_split {
            let peer = peer_exchange
                .as_ref()
                .context("Hy4 ContextSplit requires TP2 PeerExchange")?;
            let abort = [
                peer.abort_word(0)?
                    .context("Hy4 rank 0 exchange not abortable")?,
                peer.abort_word(1)?
                    .context("Hy4 rank 1 exchange not abortable")?,
            ];
            let buffers = hy4_context_buffers(&cfg, decode_rows);
            Some(ContextExchange::new(
                devices[0].library,
                [
                    RankDevice {
                        device: ranks[0].device.id,
                        stream: ranks[0].stream.raw,
                    },
                    RankDevice {
                        device: ranks[1].device.id,
                        stream: ranks[1].stream.raw,
                    },
                ],
                &buffers,
                Some(abort),
            )?)
        } else {
            None
        };
        let head_handoff = if gpus == 2 {
            Some(Handoff::new(
                ranks[0].stream.clone(),
                ranks[1].stream.clone(),
            )?)
        } else {
            None
        };

        let mut full_index_of = BTreeMap::new();
        for (idx, &layer) in cfg.full_indexer_layers().iter().enumerate() {
            full_index_of.insert(layer, idx);
        }

        let golden_units: Vec<u32> = if context_split {
            (0..max_pages * 2).map(|p| p as u32).collect()
        } else {
            (0..max_pages).map(|p| p as u32).collect()
        };
        let golden_placement = Hy4Placement::new(golden_units, 0);

        Ok(Self {
            cfg,
            max_rows,
            decode_rows,
            prefill_waves,
            max_pages,
            max_sequences,
            max_context,
            skip_routed_experts,
            dump_dir: None,
            context_split,
            admitted_pages: index_topk.admitted_pages,
            graphs_enabled: Cell::new(false),
            graph_cycles: RefCell::new(None),
            metadata_uploads,
            ranks,
            context_exchange,
            peer_exchange,
            ffn_routed_recv_offset,
            head_handoff,
            spark: None,
            onboarded_layers: BTreeSet::new(),
            next_identity: 1,
            placement_version: 1,
            full_index_of,
            golden_placement,
            expert_usage: None,
            count_expert_usage: false,
            terminal: false,
        })
    }

    pub(crate) fn library(&self) -> &'a cuteafd_ffi::NativeLibrary {
        self.ranks[0].device.library
    }

    pub(crate) fn prefill_waves(&self) -> usize {
        self.prefill_waves
    }

    pub(crate) fn gpus(&self) -> usize {
        self.ranks.len()
    }

    pub(crate) fn attach_spark(&mut self, link: SparkLink<'a>, onboarded_layers: Vec<usize>) {
        self.spark = Some(link);
        self.onboarded_layers = onboarded_layers.into_iter().collect();
    }

    pub(crate) fn configure_expert_usage(
        &mut self,
        geometry: Option<&cuteafd_core::expert_usage::Geometry>,
        reader: cuteafd_core::expert_usage::Reader,
    ) -> Result<()> {
        let Some(geometry) = geometry else {
            return Ok(());
        };
        let r0 = &self.ranks[0];
        let usage = ExpertUsage::new(
            r0.device.library,
            geometry.clone(),
            &[UsageDevice {
                logical: 0,
                cuda: r0.device.id,
            }],
            reader,
        )?;
        self.expert_usage = Some(usage);
        Ok(())
    }

    pub(crate) fn enable_expert_usage(&mut self) {
        self.count_expert_usage = self.expert_usage.is_some();
    }

    pub(crate) fn configure_graphs(
        &mut self,
        policies: &[cuteafd_loader::placement::graph_reserve::Resolved],
    ) -> Result<()> {
        ensure!(policies.len() == self.ranks.len(), "Hy4 graph policy rank count mismatch");
        for (r, &policy) in self.ranks.iter_mut().zip(policies) {
            r.graphs.attach(r.device.library, r.device.id, policy)?;
        }
        let streams: Vec<(i32, *mut std::ffi::c_void)> = self
            .ranks
            .iter()
            .map(|r| (r.device.id, r.stream.raw))
            .collect();
        *self.graph_cycles.borrow_mut() = Some(graph_device::EventCycles::new(
            self.ranks[0].device.library,
            &streams,
        )?);
        self.graphs_enabled.set(true);
        Ok(())
    }

    pub(crate) fn graphs_active(&self) -> bool {
        self.graphs_enabled.get() && self.ranks.iter().all(|r| r.graphs.attached())
    }

    pub(crate) fn without_graphs<T>(&self, run: impl FnOnce(&Self) -> Result<T>) -> Result<T> {
        let prev = self.graphs_enabled.replace(false);
        let out = run(self);
        self.graphs_enabled.set(prev);
        out
    }

    pub(crate) fn without_graphs_mut<T>(
        &mut self,
        run: impl FnOnce(&mut Self) -> Result<T>,
    ) -> Result<T> {
        let prev = self.graphs_enabled.replace(false);
        let out = run(self);
        self.graphs_enabled.set(prev);
        out
    }

    pub(crate) fn retained_graph_rows(&self, rows: usize) -> usize {
        self.ranks.iter().map(|r| r.graphs.retained_rows(rows)).sum()
    }

    fn set_graph_cycle(&self, priority: ReadPriority) {
        let cycle = match priority {
            ReadPriority::Decode | ReadPriority::DecodeFirst => GraphCycle::Decode,
            _ => GraphCycle::Prefill,
        };
        for r in &self.ranks {
            r.graphs.cycle(cycle);
        }
    }

    fn begin_graph_cycle(&self) -> Result<()> {
        if !self.graphs_active() {
            return Ok(());
        }
        let mut cycles = self.graph_cycles.borrow_mut();
        let Some(cycles) = cycles.as_mut() else {
            return Ok(());
        };
        // SAFETY: every Hy4 graph captures only persistent engine weights, cache, scratch, and EXL3 owners.
        unsafe {
            cycles.begin_persistent(|visit| {
                for r in &self.ranks {
                    r.graphs.visit(visit)?;
                }
                Ok(())
            })
        }
    }

    fn graph_completed(&self) -> Result<()> {
        if !self.graphs_active() {
            return Ok(());
        }
        let mut cycles = self.graph_cycles.borrow_mut();
        let Some(cycles) = cycles.as_mut() else {
            return Ok(());
        };
        for r in &self.ranks {
            r.graphs.tick();
        }
        // SAFETY: every rank's launches for the submitted decode step are queued on its stream.
        unsafe { cycles.complete() }
    }

    fn abandon_graph_cycle(&self) {
        if let Some(cycles) = self.graph_cycles.borrow_mut().as_mut() {
            cycles.abandon();
        }
    }

    pub(crate) fn terminate(&mut self) {
        if self.terminal {
            return;
        }
        self.terminal = true;
        self.abandon_graph_cycle();
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
        crate::shared::decode_graph::fatal_drain(drained, "Hy4 terminal owners");
        for r in &self.ranks {
            r.graphs.clear_drained();
        }
        for uploads in &mut self.metadata_uploads {
            for u in uploads {
                let _ = u.retire();
            }
        }
        if let Some(spark) = &mut self.spark {
            let _ = spark.terminal_drain_copies();
            let _ = spark.terminal_release();
        }
    }

    pub(crate) fn require_live(&self) -> Result<()> {
        ensure!(!self.terminal, "Hy4Engine already terminated");
        if let Some(ex) = &self.peer_exchange {
            ex.require_live()?;
        }
        Ok(())
    }

    pub(crate) fn prefix(&self) -> Result<Hy4Prefix<'a>> {
        Hy4Prefix::new(
            self.ranks.iter().map(|r| r.cache.clone()).collect(),
            self.context_split,
        )
    }

    pub(crate) fn drain(&self) -> Result<()> {
        for r in &self.ranks {
            r.stream.drain()?;
            r.cache.drain()?;
        }
        Ok(())
    }

    pub(crate) fn reset_golden(&mut self) -> Result<()> {
        for r in &self.ranks {
            r.cache.zero_all()?;
        }
        self.golden_placement.len = 0;
        Ok(())
    }

    pub(crate) fn dump_state(&self, placement: &Hy4Placement) -> Result<Vec<super::cache::Hy4RankStateSnapshot>> {
        self.ranks
            .iter()
            .enumerate()
            .map(|(rank, r)| r.cache.dump_state(placement, self.context_split.then_some(rank)))
            .collect()
    }

    fn peer_exchange(&self) -> Result<&PeerExchange<'a>> {
        self.peer_exchange
            .as_ref()
            .context("Hy4 TP2 PeerExchange missing")
    }

    /// Record every retained target weight arena and request/cache/scratch/transport owner into `inventory`.
    pub(crate) fn inventory(
        &self,
        weights_inv: &mut Inventory<'_>,
        retained_inv: &mut Inventory<'_>,
    ) -> Result<()> {
        if let Some(usage) = &self.expert_usage {
            for owner in usage.allocation_receipts() {
                match owner.storage {
                    UsageStorage::Device(buffer) => retained_inv.device(&owner.owner, buffer)?,
                    UsageStorage::Pinned(buffer) => retained_inv.pinned(&owner.owner, buffer)?,
                }
            }
        }
        for group in &self.metadata_uploads {
            for upload in group {
                retained_inv.pinned(upload.layout.id(), upload.backing.buffer)?;
            }
        }
        if let Some(ex) = &self.peer_exchange {
            for (rank, name, buf) in ex.allocation_buffers() {
                retained_inv.device(&format!("hy4.peer.{rank}.{name}"), buf)?;
            }
        }
        if let Some(ctx) = &self.context_exchange {
            for (rank, buf) in ctx.allocation_buffers().into_iter().enumerate() {
                retained_inv.device(&format!("hy4.context_exchange.{rank}"), buf)?;
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
                experts.allocation_receipts(weights_inv, &format!("hy4.experts.{rank}"))?;
            }
        }
        Ok(())
    }

    fn dump_bf16(&self, name: &str, buf: CuteafdDeviceBuffer, elems: usize) -> Result<()> {
        let Some(dir) = &self.dump_dir else {
            return Ok(());
        };
        let r0 = &self.ranks[0];
        r0.stream.drain()?;
        let mut raw = vec![0u8; elems * 2];
        r0.device.run(|| {
            r0.device
                .library
                .copy_d2h(&mut raw, slice_buf(buf, 0, elems * 2)?)
        })?;
        let mut out = Vec::with_capacity(elems * 4);
        for chunk in raw.chunks_exact(2) {
            let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
            let f = f32::from_bits(u32::from(bits) << 16);
            out.extend_from_slice(&f.to_le_bytes());
        }
        std::fs::write(dir.join(format!("{name}.f32")), out)?;
        Ok(())
    }

    fn dump_f32(&self, name: &str, buf: CuteafdDeviceBuffer, elems: usize) -> Result<()> {
        let Some(dir) = &self.dump_dir else {
            return Ok(());
        };
        let r0 = &self.ranks[0];
        r0.stream.drain()?;
        let mut raw = vec![0u8; elems * 4];
        r0.device.run(|| {
            r0.device
                .library
                .copy_d2h(&mut raw, slice_buf(buf, 0, elems * 4)?)
        })?;
        std::fs::write(dir.join(format!("{name}.f32")), raw)?;
        Ok(())
    }

    fn peer_push_two_planes(
        &self,
        from: usize,
        slot: usize,
        p0: (*mut std::ffi::c_void, *const std::ffi::c_void, usize),
        p1: (*mut std::ffi::c_void, *const std::ffi::c_void, usize),
    ) -> Result<()> {
        let exchange = self.peer_exchange()?;
        let decode_cap = (self.decode_rows * HC * H * 2).max(self.decode_rows * (H + H) * 4);
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
                let dst_buf = CuteafdDeviceBuffer {
                    ptr: dst,
                    bytes,
                    device_id: peer_dev,
                    flags: 0,
                };
                let src_buf = CuteafdDeviceBuffer {
                    ptr: src.cast_mut(),
                    bytes,
                    device_id: r.device.id,
                    flags: 0,
                };
                r.device
                    .library
                    .copy_d2d_async(dst_buf, src_buf, bytes, r.stream.raw)?;
            }
            Ok(())
        })?;
        exchange.signal(from, slot)
    }

    fn exchange_attn_tp2(&self, rows: usize) -> Result<()> {
        if self.ranks.len() < 2 {
            return Ok(());
        }
        let ex = self.peer_exchange()?;
        let bytes = rows * H * 4;
        ex.push(0, PEER_SLOT_ATTN, self.ranks[0].shared.delta_f32.buffer.ptr, bytes)?;
        ex.push(1, PEER_SLOT_ATTN, self.ranks[1].shared.delta_f32.buffer.ptr, bytes)?;
        ex.wait(0, PEER_SLOT_ATTN)?;
        ex.wait(1, PEER_SLOT_ATTN)
    }

    fn exchange_ffn_tp2(
        &self,
        wave_slot: usize,
        rows: usize,
        is_dense: bool,
        reduce_routed_tp2: bool,
        rank0_only: bool,
    ) -> Result<()> {
        if self.ranks.len() < 2 {
            return Ok(());
        }
        let ex = self.peer_exchange()?;
        let h_f32_bytes = rows * H * 4;
        if is_dense {
            let p0 = self.ranks[0].shared.delta_f32.buffer.ptr;
            let p1 = self.ranks[1].shared.delta_f32.buffer.ptr;
            if !rank0_only {
                ex.push(0, PEER_SLOT_FFN, p0, h_f32_bytes)?;
            }
            ex.push(1, PEER_SLOT_FFN, p1, h_f32_bytes)?;
            ex.wait(0, PEER_SLOT_FFN)?;
            if !rank0_only {
                ex.wait(1, PEER_SLOT_FFN)?;
            }
            return Ok(());
        }
        let s0 = self.ranks[0].waves[wave_slot].shared_f32.buffer.ptr;
        let s1 = self.ranks[1].waves[wave_slot].shared_f32.buffer.ptr;
        if reduce_routed_tp2 {
            let r0_in = routed_f32_input(&self.ranks[0], rows, self.skip_routed_experts)?.ptr;
            let r1_in = routed_f32_input(&self.ranks[1], rows, self.skip_routed_experts)?.ptr;
            let off = self.ffn_routed_recv_offset;
            if !rank0_only {
                let dst1 = ex.recv(1, PEER_SLOT_FFN)?;
                self.peer_push_two_planes(
                    0,
                    PEER_SLOT_FFN,
                    (dst1, s0, h_f32_bytes),
                    (dst1.wrapping_byte_add(off), r0_in, h_f32_bytes),
                )?;
            }
            let dst0 = ex.recv(0, PEER_SLOT_FFN)?;
            self.peer_push_two_planes(
                1,
                PEER_SLOT_FFN,
                (dst0, s1, h_f32_bytes),
                (dst0.wrapping_byte_add(off), r1_in, h_f32_bytes),
            )?;
            ex.wait(0, PEER_SLOT_FFN)?;
            if !rank0_only {
                ex.wait(1, PEER_SLOT_FFN)?;
            }
        } else {
            // Spark experts reduced directly into rank 0's `routed_out` ([rows, H] BF16):
            // replicate to rank 1 via DIRECT (unless rank0_only on final layer) alongside shared FFN.
            if !rank0_only {
                ex.push_to(
                    0,
                    DIRECT,
                    self.ranks[0].shared.routed_out.buffer.ptr,
                    self.ranks[1].shared.routed_out.buffer.ptr,
                    rows * H * 2,
                )?;
                ex.wait(1, DIRECT)?;
                ex.push(0, PEER_SLOT_FFN, s0, h_f32_bytes)?;
            }
            ex.push(1, PEER_SLOT_FFN, s1, h_f32_bytes)?;
            ex.wait(0, PEER_SLOT_FFN)?;
            if !rank0_only {
                ex.wait(1, PEER_SLOT_FFN)?;
            }
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn begin_batch(
        &mut self,
        placements: &[Hy4Placement],
        initial_starts: &[usize],
        staged_frontiers: &mut [usize],
        sources: &[&[u32]],
        unit: &Unit,
        priority: ReadPriority,
        head_mode: HeadMode,
    ) -> Result<ActiveBatch> {
        let wave_slot = unit.slot;
        let rows = unit.rows.len();
        ensure!(
            rows > 0 && rows <= self.max_rows,
            "Hy4 unit rows {rows} outside 1..={}",
            self.max_rows
        );
        ensure!(
            wave_slot < self.prefill_waves,
            "Hy4 wave slot {wave_slot} outside 0..{}",
            self.prefill_waves
        );

        for uploads in &mut self.metadata_uploads {
            uploads[wave_slot].retire()?;
        }

        let mut flat_tokens = Vec::with_capacity(rows);
        let mut pos_i64 = Vec::with_capacity(rows * 24);
        let mut pos_i32 = Vec::with_capacity(rows * 4);
        let mut dsa_slots_i64 = if self.context_split { Vec::new() } else { Vec::with_capacity(rows * 8) };
        let mut dsa_slots_ctx: [Vec<u8>; 2] = if self.context_split {
            [Vec::with_capacity(rows * 8), Vec::with_capacity(rows * 8)]
        } else {
            [Vec::new(), Vec::new()]
        };
        let mut idx_cache_lens = if self.context_split { Vec::new() } else { Vec::with_capacity(rows * 4) };
        let mut idx_cache_lens_ctx: [Vec<u8>; 2] = if self.context_split {
            [Vec::with_capacity(rows * 4), Vec::with_capacity(rows * 4)]
        } else {
            [Vec::new(), Vec::new()]
        };
        let mut idx_cand_lens_host = if self.context_split { Vec::new() } else { Vec::with_capacity(rows * 4) };

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
            let pos_start = staged_frontiers[seg.sequence];
            ensure!(
                pos_start == seg.tokens.start,
                "Hy4 segment start {} != staged frontier {pos_start}",
                seg.tokens.start
            );
            let n = tok_slice.len();
            let pos_end = pos_start + n;
            staged_frontiers[seg.sequence] = pos_end;
            ensure!(
                pos_end <= self.max_context && pos_end.div_ceil(PAGE_ROWS) <= p.units.len(),
                "Hy4 sequence extent {pos_end} exceeds context {} or reserved pages {}",
                self.max_context,
                p.units.len()
            );
            ensure!(p.slot < self.max_sequences, "Hy4 sequence slot {} out of range", p.slot);
            max_seq_end = max_seq_end.max(pos_end);
            flat_tokens.extend_from_slice(tok_slice);

            for offset in 0..n {
                let pos = pos_start + offset;
                let logical_page = pos / PAGE_ROWS;
                let local_row = pos % PAGE_ROWS;
                let raw_page = p.units[logical_page] as usize;
                // Text M-RoPE has equal (T, H, W) = (pos, pos, pos).
                for _ in 0..3 {
                    pos_i64.extend_from_slice(&(pos as i64).to_le_bytes());
                }
                pos_i32.extend_from_slice(&(pos as i32).to_le_bytes());
                if self.context_split {
                    let owner = context::page_owner(logical_page);
                    ensure!(
                        raw_page & 1 == owner,
                        "Hy4 ContextSplit logical page {logical_page} has wrong parity id {raw_page}"
                    );
                    let local_slot = ((raw_page >> 1) * PAGE_ROWS + local_row) as i64;
                    for rank in 0..2 {
                        let slot_val = if rank == owner { local_slot } else { -1i64 };
                        dsa_slots_ctx[rank].extend_from_slice(&slot_val.to_le_bytes());
                        let vis = context::shard_rows(pos + 1, PAGE_ROWS, rank) as i32;
                        idx_cache_lens_ctx[rank].extend_from_slice(&vis.to_le_bytes());
                    }
                } else {
                    let dsa_slot = raw_page * PAGE_ROWS + local_row;
                    dsa_slots_i64.extend_from_slice(&(dsa_slot as i64).to_le_bytes());
                    let vis = (pos + 1) as i32;
                    idx_cache_lens.extend_from_slice(&vis.to_le_bytes());
                    idx_cand_lens_host.extend_from_slice(&vis.min(CANDIDATE_TOPK as i32).to_le_bytes());
                }
            }

            members.push(ActiveMember {
                sequence: seg.sequence,
                slot: p.slot,
                pos_start,
                row_start: row_cursor,
                row_len: n,
            });
            row_cursor += n;
        }

        let pt_stride = max_seq_end.div_ceil(PAGE_ROWS).max(1);
        let ctx_pt_stride = if self.context_split {
            pt_stride.div_ceil(2).max(1)
        } else {
            pt_stride
        };
        let mut dsa_pt_bytes = if self.context_split {
            Vec::new()
        } else {
            Vec::with_capacity(rows * pt_stride * 4)
        };
        let mut seq_pt_bytes = if self.context_split {
            Vec::new()
        } else {
            Vec::with_capacity(members.len() * pt_stride * 4)
        };
        let mut dsa_pt_ctx: [Vec<u8>; 2] = if self.context_split {
            [
                Vec::with_capacity(rows * ctx_pt_stride * 4),
                Vec::with_capacity(rows * ctx_pt_stride * 4),
            ]
        } else {
            [Vec::new(), Vec::new()]
        };

        for m in &members {
            let p = &placements[m.sequence];
            let seq_pages = (m.pos_start + m.row_len).div_ceil(PAGE_ROWS);
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
                seq_pt_bytes.extend_from_slice(&row_pt);
                for _ in 0..m.row_len {
                    dsa_pt_bytes.extend_from_slice(&row_pt);
                }
            }
        }
        let cu_singleton: Vec<u8> = (0..=rows).flat_map(|i| (i as i32).to_le_bytes()).collect();

        if self.context_split {
            for rank in 0..2 {
                self.metadata_uploads[rank][wave_slot].upload(
                    &self.ranks[rank..rank + 1],
                    wave_slot,
                    [
                        &pos_i64,
                        &pos_i32,
                        &dsa_slots_ctx[rank],
                        &dsa_pt_ctx[rank],
                        &cu_singleton,
                        &idx_cache_lens_ctx[rank],
                        &[],
                    ],
                )?;
            }
        } else {
            self.metadata_uploads[0][wave_slot].upload(
                &self.ranks,
                wave_slot,
                [
                    &pos_i64,
                    &pos_i32,
                    &dsa_slots_i64,
                    &dsa_pt_bytes,
                    &cu_singleton,
                    &idx_cache_lens,
                    &seq_pt_bytes,
                ],
            )?;
            // Non-ContextSplit `idx_cand_lens` is `min(pos + 1, 2048)` for every layer.
            for r in &self.ranks {
                let dst = r.waves[wave_slot].idx_cand_lens.buffer;
                r.device.run(|| unsafe {
                    r.device
                        .library
                        .copy_h2d_ordered(dst, &idx_cand_lens_host, r.stream.raw)
                })?;
            }
        }

        // Embed tokens on rank 0 into waves[wave_slot].r_a [rows, 4, H] BF16, replicate to rank 1 if TP2.
        {
            let r0 = &self.ranks[0];
            let emb = r0
                .weights
                .embedding
                .as_ref()
                .context("Hy4 rank 0 missing TokenEmbedding")?;
            let wave = &r0.waves[wave_slot];
            r0.device.run(|| {
                emb.embed(&flat_tokens, r0.shared.ids.buffer, HC, wave.r_a.buffer, r0.stream.raw)
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

        let graphed = priority == ReadPriority::Decode
            && wave_slot == 0
            && rows <= self.decode_rows
            && self.graphs_active()
            && !self.skip_routed_experts
            && self.dump_dir.is_none();

        let dump_rows = if self.dump_dir.is_some()
            && members.len() == 1
            && (members[0].pos_start == 0 || priority == ReadPriority::Decode)
        {
            rows.min(16)
        } else {
            0
        };
        if dump_rows > 0 {
            let dir = self.dump_dir.as_ref().unwrap();
            std::fs::create_dir_all(dir)?;
            let tok_bytes: Vec<u8> = flat_tokens[..dump_rows]
                .iter()
                .flat_map(|t| t.to_le_bytes())
                .collect();
            std::fs::write(dir.join("tokens.u32"), tok_bytes)?;
            self.dump_bf16(
                "embed",
                self.ranks[0].waves[wave_slot].r_a.buffer,
                dump_rows * HC * H,
            )?;
        }

        let final_layer = unit.layers.end.saturating_sub(1);
        let rank0_only_tail = self.ranks.len() == 2
            && unit.layers.end == self.cfg.layers
            && head_mode != HeadMode::None
            && dump_rows == 0;

        Ok(ActiveBatch {
            wave_slot,
            rows,
            pt_stride,
            ctx_pt_stride,
            ctx_slot_counter: 0,
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
        })
    }

    /// Run `hy4*_index_topk_{decode,prefill}` on rank `r` for a full IndexCache layer (TP1 or TP2-Heads).
    fn run_full_indexer_topk(r: &RankRuntime<'_>, full_idx: usize, batch: &ActiveBatch) -> Result<()> {
        let w = &r.waves[batch.wave_slot];
        let idx_cache = r.cache.dsa_index_layer(full_idx)?;
        if batch.decode_attention && batch.rows <= r.programs.index_topk_decode.spec().capacity_rows as usize {
            r.device.run(|| unsafe {
                r.programs.index_topk_decode.launch(
                    &[
                        r.shared.idx_q_fp8.buffer.ptr,
                        r.shared.idx_w_out.buffer.ptr,
                        idx_cache.ptr,
                        w.dsa_page_table.buffer.ptr,
                        w.idx_cache_lens.buffer.ptr,
                        w.idx_cand_slots.buffer.ptr,
                        r.shared.idx_topk_scratch.buffer.ptr,
                    ],
                    &[
                        Scalar::I32(batch.rows as i32),
                        Scalar::I32(batch.pt_stride as i32),
                        Scalar::I32(batch.pt_stride as i32),
                    ],
                    r.stream.raw,
                )
            })
        } else {
            r.device.run(|| unsafe {
                for (seq_idx, m) in batch.members.iter().enumerate() {
                    let q_ptr = r
                        .shared
                        .idx_q_fp8
                        .buffer
                        .ptr
                        .cast::<u8>()
                        .wrapping_add(m.row_start * IDX_HEADS * IDX_DIM)
                        .cast();
                    let w_ptr = r
                        .shared
                        .idx_w_out
                        .buffer
                        .ptr
                        .cast::<u8>()
                        .wrapping_add(m.row_start * IDX_HEADS * 4)
                        .cast();
                    let pt_ptr = w
                        .idx_seq_page_tables
                        .buffer
                        .ptr
                        .cast::<u8>()
                        .wrapping_add(seq_idx * batch.pt_stride * 4)
                        .cast();
                    let len_ptr = w
                        .idx_cache_lens
                        .buffer
                        .ptr
                        .cast::<u8>()
                        .wrapping_add(m.row_start * 4)
                        .cast();
                    let out_ptr = w
                        .idx_cand_slots
                        .buffer
                        .ptr
                        .cast::<u8>()
                        .wrapping_add(m.row_start * CANDIDATE_TOPK * 4)
                        .cast();
                    let seq_width = (m.pos_start + m.row_len).div_ceil(PAGE_ROWS).max(1);
                    r.programs.index_topk_prefill.launch(
                        &[
                            q_ptr,
                            w_ptr,
                            idx_cache.ptr,
                            pt_ptr,
                            len_ptr,
                            out_ptr,
                            r.shared.idx_topk_scratch.buffer.ptr,
                        ],
                        &[
                            Scalar::I32(m.row_len as i32),
                            Scalar::I32(seq_width as i32),
                            Scalar::I32(seq_width as i32),
                        ],
                        r.stream.raw,
                    )?;
                }
                Ok(())
            })
        }
    }

    /// 2-GPU ContextSplit DSA body after `attn_pre` wrote parity-owned KV (and `index_post` on full layers).
    ///
    /// - Full IndexCache layers (`full_idx.is_some()`): run `scored_index_topk_decode` + candidate
    ///   exchange + `dsa_candidate_merge` into `WaveScratch.{idx_cand_slots, idx_cand_lens}`.
    /// - Shared IndexCache layers (`full_idx.is_none()`): skip scoring and candidate exchange
    ///   completely, reusing `WaveScratch.{idx_cand_slots, idx_cand_lens}` from the preceding full layer.
    fn run_dsa_context_split_body(
        &self,
        layer: usize,
        full_idx: Option<usize>,
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
        let kv_caches = [
            self.ranks[0].cache.dsa_kv_layer(layer)?,
            self.ranks[1].cache.dsa_kv_layer(layer)?,
        ];
        let idx_caches = match full_idx {
            Some(fidx) => Some([
                self.ranks[0].cache.dsa_index_layer(fidx)?,
                self.ranks[1].cache.dsa_index_layer(fidx)?,
            ]),
            None => None,
        };
        let query_half_bytes = 32 * KV_A_DIM * 2;
        let ctx_chunk = self.decode_rows;

        for chunk_start in (0..rows).step_by(ctx_chunk) {
            let c_rows = (rows - chunk_start).min(ctx_chunk);
            let slot = if batch.graphed {
                context::slot(layer, 0)
            } else {
                let s = batch.ctx_slot_counter % 2;
                batch.ctx_slot_counter += 1;
                s
            };

            // (a) Query exchange: each rank pushes its 32-head absorbed query -> assembled [c_rows, 64, 576].
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

            // (b) Full IndexCache layers only: scored top-2048 + candidate exchange + merge into WaveScratch.
            if let Some(caches) = idx_caches {
                for rank in 0..2 {
                    let r = &self.ranks[rank];
                    let ctx_progs = r
                        .programs
                        .context
                        .as_ref()
                        .context("Hy4 ContextSplit programs missing")?;
                    let ctx = r
                        .shared
                        .ctx
                        .as_ref()
                        .context("Hy4 ContextSplit scratch missing")?;
                    let w = &r.waves[wave_slot];
                    let q_ptr = r
                        .shared
                        .idx_q_fp8
                        .buffer
                        .ptr
                        .cast::<u8>()
                        .wrapping_add(chunk_start * IDX_HEADS * IDX_DIM)
                        .cast();
                    let w_ptr = r
                        .shared
                        .idx_w_out
                        .buffer
                        .ptr
                        .cast::<u8>()
                        .wrapping_add(chunk_start * IDX_HEADS * 4)
                        .cast();
                    let pt_ptr = w
                        .dsa_page_table
                        .buffer
                        .ptr
                        .cast::<u8>()
                        .wrapping_add(chunk_start * ctx_pt_stride * 4)
                        .cast();
                    let len_ptr = w
                        .idx_cache_lens
                        .buffer
                        .ptr
                        .cast::<u8>()
                        .wrapping_add(chunk_start * 4)
                        .cast();
                    r.device.run(|| unsafe {
                        ctx_progs.scored_index_topk_decode.launch(
                            &[
                                q_ptr,
                                w_ptr,
                                caches[rank].ptr,
                                pt_ptr,
                                len_ptr,
                                ctx.cand_indices.buffer.ptr,
                                ctx.cand_scores.buffer.ptr,
                                r.shared.idx_topk_scratch.buffer.ptr,
                            ],
                            &[
                                Scalar::I32(c_rows as i32),
                                Scalar::I32(ctx_pt_stride as i32),
                                Scalar::I32(ctx_pt_stride as i32),
                                Scalar::I32(2),
                                Scalar::I32(rank as i32),
                                Scalar::I32(self.admitted_pages as i32),
                            ],
                            r.stream.raw,
                        )
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
                let cand_s = [s0, s1];
                let cand_i = [i0, i1];
                for rank in 0..2 {
                    let r = &self.ranks[rank];
                    let ctx_progs = r
                        .programs
                        .context
                        .as_ref()
                        .context("Hy4 ContextSplit programs missing")?;
                    let ctx = r
                        .shared
                        .ctx
                        .as_ref()
                        .context("Hy4 ContextSplit scratch missing")?;
                    let w = &r.waves[wave_slot];
                    let pt_ptr = w
                        .dsa_page_table
                        .buffer
                        .ptr
                        .cast::<u8>()
                        .wrapping_add(chunk_start * ctx_pt_stride * 4)
                        .cast();
                    let local_slots_ptr = w
                        .idx_cand_slots
                        .buffer
                        .ptr
                        .cast::<u8>()
                        .wrapping_add(chunk_start * CANDIDATE_TOPK * 4)
                        .cast();
                    let local_lens_ptr = w
                        .idx_cand_lens
                        .buffer
                        .ptr
                        .cast::<u8>()
                        .wrapping_add(chunk_start * 4)
                        .cast();
                    r.device.run(|| unsafe {
                        ctx_progs.dsa_candidate_merge.launch(
                            &[
                                cand_s[rank],
                                cand_i[rank],
                                ctx.merged_scores.buffer.ptr,
                                ctx.merged_indices.buffer.ptr,
                                pt_ptr,
                                local_slots_ptr,
                                local_lens_ptr,
                            ],
                            &[
                                Scalar::I32(c_rows as i32),
                                Scalar::I32(ctx_pt_stride as i32),
                                Scalar::I32(rank as i32),
                            ],
                            r.stream.raw,
                        )
                    })?;
                }
            }

            // (c) Dual h0 + h32 partials over each rank's parity-owned local pages.
            for rank in 0..2 {
                let r = &self.ranks[rank];
                let q_ptr = assembled_q[rank];
                let kv_buf = kv_caches[rank];
                let decode_attn = batch.decode_attention;
                if batch.graphed {
                    r.graphs.run(
                        GraphKey {
                            layer,
                            rows,
                            section: GraphSection::DsaPartial,
                        },
                        &r.stream,
                        || {
                            r.device.run(|| {
                                r.dsa_attn_context_partials(
                                    rank,
                                    q_ptr,
                                    kv_buf,
                                    wave_slot,
                                    0,
                                    c_rows,
                                    decode_attn,
                                )
                            })
                        },
                    )?;
                } else {
                    r.device.run(|| {
                        r.dsa_attn_context_partials(
                            rank,
                            q_ptr,
                            kv_buf,
                            wave_slot,
                            chunk_start,
                            c_rows,
                            decode_attn,
                        )
                    })?;
                }
            }

            // (d) Partial exchange: push peer's 32-head partial + base-2 LSE, wait for ours.
            for rank in 0..2 {
                let ctx = self.ranks[rank]
                    .shared
                    .ctx
                    .as_ref()
                    .context("Hy4 ContextSplit scratch missing")?;
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

            // (e) Combine GPU0-then-GPU1 partials with learnable sink; fuse gate_uv + o_proj when graphed.
            for rank in 0..2 {
                let r = &self.ranks[rank];
                let rp = recv_p[rank];
                let rl = recv_l[rank];
                if batch.graphed {
                    r.graphs.run(
                        GraphKey {
                            layer,
                            rows,
                            section: GraphSection::DsaAttnOut,
                        },
                        &r.stream,
                        || {
                            r.device.run(|| {
                                r.dsa_context_combine(rank, p, rp, rl, 0, rows)?;
                                r.dsa_context_gate_uv_o(p, layer, rows)
                            })
                        },
                    )?;
                } else {
                    r.device
                        .run(|| r.dsa_context_combine(rank, p, rp, rl, chunk_start, c_rows))?;
                }
            }
        }

        if !batch.graphed {
            for r in &self.ranks {
                r.device.run(|| r.dsa_context_gate_uv_o(p, layer, rows))?;
            }
        }
        Ok(())
    }

    fn run_layer_attention(
        &mut self,
        layer: usize,
        batch: &mut ActiveBatch,
        local_ffn_route: bool,
    ) -> Result<()> {
        let p = format!("model.layers.{layer}");
        let gpus = self.ranks.len();
        let local_heads = self.cfg.heads / gpus;
        let rows = batch.rows;
        let wave_slot = batch.wave_slot;
        let full_idx = self.full_index_of.get(&layer).copied();
        let decode_attention = batch.decode_attention;
        let is_dense = self.cfg.dense_layer[layer];
        let routed_scale = self.cfg.routed_scaling_factor as f32;
        let skip_routed = self.skip_routed_experts;
        let usage = self.expert_usage.as_ref();
        let count_usage = self.count_expert_usage;
        let usage_phase = batch.usage_phase;
        let routed_recv_offset = self.ffn_routed_recv_offset;

        let prev_ffn = batch.prev_ffn_deferred.take();
        let prev_use_a = batch.use_a;
        if prev_ffn.is_some() {
            batch.use_a = !batch.use_a;
        }
        let use_a = batch.use_a;
        let ffn_recv_ptrs = if prev_ffn.is_some() && gpus == 2 {
            let ex = self.peer_exchange()?;
            [ex.recv(0, PEER_SLOT_FFN)?, ex.recv(1, PEER_SLOT_FFN)?]
        } else {
            [std::ptr::null_mut(); 2]
        };

        if batch.graphed {
            let pre_section = if full_idx.is_some() {
                GraphSection::DsaFullPre
            } else {
                GraphSection::DsaSharedPre
            };
            // Case 1: TP1 local FFN (dense layer 0 or onboarded MoE).
            if gpus == 1 && local_ffn_route {
                let r = &self.ranks[0];
                let kv_cache_buf = r.cache.dsa_kv_layer(layer)?;
                if let Some(fidx) = full_idx {
                    r.graphs.run(
                        GraphKey { layer, rows, section: pre_section },
                        &r.stream,
                        || {
                            r.device.run(|| {
                                if let Some(prev) = prev_ffn {
                                    r.finish_deferred_ffn_local(
                                        0, 1, prev, wave_slot, rows, prev_use_a, skip_routed,
                                        std::ptr::null_mut(), routed_recv_offset,
                                    )?;
                                }
                                r.attn_pre(&p, layer, Some(fidx), local_heads, wave_slot, rows, use_a, kv_cache_buf)
                            })
                        },
                    )?;
                    Self::run_full_indexer_topk(r, fidx, batch)?;
                    r.graphs.run(
                        GraphKey { layer, rows, section: GraphSection::DsaLocalPost },
                        &r.stream,
                        || {
                            r.device.run(|| {
                                r.dsa_attn(kv_cache_buf, wave_slot, rows, decode_attention)?;
                                r.attn_out(&p, layer, 1, local_heads, rows)?;
                                r.post_attn_ihc(&p, wave_slot, rows, use_a)?;
                                if is_dense {
                                    r.dense_ffn(&p, 1, wave_slot, rows)?;
                                } else {
                                    r.moe_router_shared(&p, 1, wave_slot, rows, routed_scale, true)?;
                                    r.local_exl3(layer, 1, rows, skip_routed)?;
                                    r.record_expert_usage_local(
                                        usage, count_usage, skip_routed, false, layer, usage_phase, rows,
                                    )?;
                                }
                                r.ffn_combine_post(is_dense, wave_slot, rows, !use_a)
                            })
                        },
                    )?;
                } else {
                    // Shared IndexCache layer: fuse entire layer into ONE graph (zero host joins!).
                    r.graphs.run(
                        GraphKey { layer, rows, section: GraphSection::DsaLocalPost },
                        &r.stream,
                        || {
                            r.device.run(|| {
                                if let Some(prev) = prev_ffn {
                                    r.finish_deferred_ffn_local(
                                        0, 1, prev, wave_slot, rows, prev_use_a, skip_routed,
                                        std::ptr::null_mut(), routed_recv_offset,
                                    )?;
                                }
                                r.attn_pre(&p, layer, None, local_heads, wave_slot, rows, use_a, kv_cache_buf)?;
                                r.dsa_attn(kv_cache_buf, wave_slot, rows, decode_attention)?;
                                r.attn_out(&p, layer, 1, local_heads, rows)?;
                                r.post_attn_ihc(&p, wave_slot, rows, use_a)?;
                                if is_dense {
                                    r.dense_ffn(&p, 1, wave_slot, rows)?;
                                } else {
                                    r.moe_router_shared(&p, 1, wave_slot, rows, routed_scale, true)?;
                                    r.local_exl3(layer, 1, rows, skip_routed)?;
                                    r.record_expert_usage_local(
                                        usage, count_usage, skip_routed, false, layer, usage_phase, rows,
                                    )?;
                                }
                                r.ffn_combine_post(is_dense, wave_slot, rows, !use_a)
                            })
                        },
                    )?;
                }
                batch.ffn_fused_layer = Some(layer);
                return Ok(());
            }

            // Case 2: TP2 or Spark-routed layer.
            for (rank, r) in self.ranks.iter().enumerate() {
                let kv_cache_buf = r.cache.dsa_kv_layer(layer)?;
                let ffn_recv = ffn_recv_ptrs[rank];
                r.graphs.run(
                    GraphKey { layer, rows, section: pre_section },
                    &r.stream,
                    || {
                        r.device.run(|| {
                            if let Some(prev) = prev_ffn {
                                r.finish_deferred_ffn_local(
                                    rank, gpus, prev, wave_slot, rows, prev_use_a, skip_routed,
                                    ffn_recv, routed_recv_offset,
                                )?;
                            }
                            r.attn_pre(&p, layer, full_idx, local_heads, wave_slot, rows, use_a, kv_cache_buf)
                        })
                    },
                )?;
            }
            if self.context_split {
                self.run_dsa_context_split_body(layer, full_idx, &p, batch)?;
            } else {
                if let Some(fidx) = full_idx {
                    for r in &self.ranks {
                        Self::run_full_indexer_topk(r, fidx, batch)?;
                    }
                }
                for r in &self.ranks {
                    let kv_cache_buf = r.cache.dsa_kv_layer(layer)?;
                    r.graphs.run(
                        GraphKey { layer, rows, section: GraphSection::DsaAttnOut },
                        &r.stream,
                        || {
                            r.device.run(|| {
                                r.dsa_attn(kv_cache_buf, wave_slot, rows, decode_attention)?;
                                r.attn_out(&p, layer, gpus, local_heads, rows)
                            })
                        },
                    )?;
                }
            }
            self.exchange_attn_tp2(rows)?;
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
                        rank, gpus, prev, wave_slot, rows, prev_use_a, skip_routed,
                        ffn_recv, routed_recv_offset,
                    )
                })?;
            }
        }
        if self.context_split {
            for r in &self.ranks {
                let kv_cache_buf = r.cache.dsa_kv_layer(layer)?;
                r.device.run(|| {
                    r.attn_pre(&p, layer, full_idx, local_heads, wave_slot, rows, use_a, kv_cache_buf)
                })?;
            }
            self.run_dsa_context_split_body(layer, full_idx, &p, batch)?;
        } else {
            for r in &self.ranks {
                let kv_cache_buf = r.cache.dsa_kv_layer(layer)?;
                r.device.run(|| {
                    r.attn_pre(&p, layer, full_idx, local_heads, wave_slot, rows, use_a, kv_cache_buf)
                })?;
                if let Some(fidx) = full_idx {
                    Self::run_full_indexer_topk(r, fidx, batch)?;
                }
                r.device.run(|| {
                    r.dsa_attn(kv_cache_buf, wave_slot, rows, decode_attention)?;
                    r.attn_out(&p, layer, gpus, local_heads, rows)
                })?;
            }
        }
        if gpus == 2 {
            self.exchange_attn_tp2(rows)?;
            let attn_recv = [
                self.peer_exchange()?.recv(0, PEER_SLOT_ATTN)?,
                self.peer_exchange()?.recv(1, PEER_SLOT_ATTN)?,
            ];
            let o_stem = format!("{p}.self_attn.o_proj");
            for (rank, r) in self.ranks.iter().enumerate() {
                r.device.run(|| {
                    r.reduce_f32_to_bf16_local(
                        rank,
                        rows,
                        H,
                        Some(&o_stem),
                        r.shared.delta_f32.buffer,
                        attn_recv[rank],
                        r.shared.delta.buffer,
                    )
                })?;
            }
        }
        if batch.dump_rows > 0 && layer <= 3 {
            let s0 = &self.ranks[0].shared;
            let w0 = &self.ranks[0].waves[wave_slot];
            let dr = batch.dump_rows;
            self.dump_bf16(&format!("l{layer:02}_attn_y"), w0.y.buffer, dr * H)?;
            self.dump_bf16(&format!("l{layer:02}_attn_q_resid"), s0.q_resid.buffer, dr * Q_LORA)?;
            self.dump_bf16(&format!("l{layer:02}_attn_query"), s0.query.buffer, dr * local_heads * 576)?;
            self.dump_bf16(&format!("l{layer:02}_attn_out"), s0.attn_out.buffer, dr * local_heads * 512)?;
            self.dump_f32(&format!("l{layer:02}_attn_lse"), s0.lse_out.buffer, dr * local_heads)?;
            self.dump_bf16(&format!("l{layer:02}_attn_uv"), s0.uv_out.buffer, dr * local_heads * 256)?;
            self.dump_bf16(&format!("l{layer:02}_attn_delta"), s0.delta.buffer, dr * H)?;
        }

        for r in &self.ranks {
            r.device.run(|| r.post_attn_ihc(&p, wave_slot, rows, use_a))?;
        }
        batch.use_a = !batch.use_a;
        Ok(())
    }

    fn run_moe_router_and_shared(
        &mut self,
        layer: usize,
        batch: &mut ActiveBatch,
        local_experts: bool,
    ) -> Result<()> {
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
                    GraphKey { layer, rows, section: GraphSection::FfnFrontend },
                    &r.stream,
                    || {
                        r.device.run(|| {
                            if reduce_attn {
                                r.reduce_f32_to_bf16_local(
                                    rank, rows, H, Some(&o_stem),
                                    r.shared.delta_f32.buffer, recv_ptr, r.shared.delta.buffer,
                                )?;
                            }
                            if deferred {
                                r.post_attn_ihc(&p, wave_slot, rows, use_a)?;
                            }
                            r.moe_router_shared(&p, gpus, wave_slot, rows, routed_scale, rank == 0 || local_experts)?;
                            if rank == 0 {
                                r.record_expert_usage_local(
                                    usage, count_usage, skip_routed, false, layer, usage_phase, rows,
                                )?;
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
                        rank, rows, H, Some(&o_stem),
                        r.shared.delta_f32.buffer, recv_ptr, r.shared.delta.buffer,
                    )?;
                }
                if deferred {
                    r.post_attn_ihc(&p, wave_slot, rows, use_a)?;
                }
                r.moe_router_shared(&p, gpus, wave_slot, rows, routed_scale, rank == 0 || local_experts)
            })?;
        }
        if deferred {
            batch.post_attn_deferred = false;
            batch.use_a = !batch.use_a;
        }
        batch.reduce_attn_pending = false;
        Ok(())
    }

    fn finish_ffn_combine_and_post(
        &mut self,
        layer: usize,
        batch: &mut ActiveBatch,
        reduce_routed_tp2: bool,
    ) -> Result<()> {
        let gpus = self.ranks.len();
        let rows = batch.rows;
        let is_dense = self.cfg.dense_layer[layer];
        let wave_slot = batch.wave_slot;
        let rank0_only = batch.rank0_only_tail && layer == batch.final_layer;

        if gpus == 2 {
            self.exchange_ffn_tp2(wave_slot, rows, is_dense, reduce_routed_tp2, rank0_only)?;
        }

        let deferred_ffn = DeferredFfn {
            layer,
            is_dense,
            reduce_routed_tp2,
        };
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
                                rank, gpus, deferred_ffn, wave_slot, rows, use_a, skip_routed,
                                recv_ptr, routed_off,
                            )
                        })
                    },
                )?;
            } else {
                r.device.run(|| {
                    r.finish_deferred_ffn_local(
                        rank, gpus, deferred_ffn, wave_slot, rows, use_a, skip_routed,
                        recv_ptr, routed_off,
                    )
                })?;
            }
        }
        batch.use_a = !batch.use_a;

        if batch.dump_rows > 0 {
            let dr = batch.dump_rows;
            let w0 = &self.ranks[0].waves[wave_slot];
            let r_cur = if batch.use_a { w0.r_a.buffer } else { w0.r_b.buffer };
            self.dump_bf16(&format!("l{layer:02}_out"), r_cur, dr * HC * H)?;
            if layer <= 3 {
                let s0 = &self.ranks[0].shared;
                self.dump_bf16(&format!("l{layer:02}_ffn_y"), w0.y.buffer, dr * H)?;
                self.dump_bf16(&format!("l{layer:02}_ffn_delta"), s0.delta.buffer, dr * H)?;
                if !is_dense {
                    self.dump_bf16(&format!("l{layer:02}_ffn_routed_out"), s0.routed_out.buffer, dr * H)?;
                    self.dump_bf16(&format!("l{layer:02}_ffn_shared_out"), w0.shared_out.buffer, dr * H)?;
                }
            }
        }
        Ok(())
    }

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
                                        rank, rows, H, Some(&o_stem),
                                        r.shared.delta_f32.buffer, recv_ptr, r.shared.delta.buffer,
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
                                rank, rows, H, Some(&o_stem),
                                r.shared.delta_f32.buffer, recv_ptr, r.shared.delta.buffer,
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
                                    rank, rows, H, Some(&o_stem),
                                    r.shared.delta_f32.buffer, recv_ptr, r.shared.delta.buffer,
                                )?;
                            }
                            if deferred {
                                r.post_attn_ihc(&p, wave_slot, rows, use_a)?;
                            }
                            r.moe_router_shared(&p, gpus, wave_slot, rows, routed_scale, true)?;
                            r.local_exl3(layer, gpus, rows, skip_routed)?;
                            if rank == 0 {
                                r.record_expert_usage_local(
                                    usage, count_usage, skip_routed, false, layer, usage_phase, rows,
                                )?;
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

        self.run_moe_router_and_shared(layer, batch, true)?;
        for r in &self.ranks {
            r.device.run(|| r.local_exl3(layer, gpus, rows, skip_routed))?;
        }
        self.finish_ffn_combine_and_post(layer, batch, true)
    }

    fn record_expert_usage(&self, layer: usize, batch: &ActiveBatch) -> Result<()> {
        if !self.count_expert_usage || self.skip_routed_experts || self.cfg.dense_layer[layer] {
            return Ok(());
        }
        let r0 = &self.ranks[0];
        r0.device.run(|| {
            r0.record_expert_usage_local(
                self.expert_usage.as_ref(),
                self.count_expert_usage,
                self.skip_routed_experts,
                false,
                layer,
                batch.usage_phase,
                batch.rows,
            )
        })
    }

    fn project_head_tile(&self, wave_slot: usize, y_row_off: usize, count: usize) -> Result<DeviceLogits> {
        ensure!(count > 0 && count <= 64, "Hy4 head tile count {count} outside 1..=64");
        let r0 = &self.ranks[0];
        let input = slice_buf(r0.waves[wave_slot].y.buffer, y_row_off * H * 2, count * H * 2)?;
        let vocab = self.cfg.vocab_size;
        let head_out = r0
            .shared
            .head_logits_f32
            .as_ref()
            .context("Hy4 rank 0 missing head_logits_f32")?
            .buffer;
        if self.ranks.len() == 1 {
            let out_buf = CuteafdDeviceBuffer {
                bytes: count * vocab * 4,
                ..head_out
            };
            r0.device.run(|| launch_rank_lm_head(r0, count, input, out_buf))?;
        } else {
            let r1 = &self.ranks[1];
            let w1 = &r1.waves[wave_slot];
            let handoff = self
                .head_handoff
                .as_ref()
                .context("Hy4 TP2 head handoff missing")?;
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
                input,
                x1_buf,
                || launch_rank_lm_head(r0, count, input, out0_buf),
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
                    GraphKey {
                        layer: prev.layer,
                        rows,
                        section: GraphSection::Tail,
                    },
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
                                        r0.shared.delta.buffer.ptr,
                                        w0.y.buffer.ptr,
                                    ],
                                    &[Scalar::I32(rows as i32)],
                                    r0.stream.raw,
                                )
                            }
                        })
                    },
                )?;
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
                        rank, gpus, prev, wave_slot, batch.rows, prev_use_a, skip_routed,
                        ffn_recv[rank], routed_off,
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
                            unsafe {
                                r0.device.library.copy_d2d_async(dst, src, row_bytes, r0.stream.raw)?;
                            }
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
                            r0.shared.delta.buffer.ptr,
                            w0.y.buffer.ptr,
                        ],
                        &[Scalar::I32(count as i32)],
                        r0.stream.raw,
                    )
                })?;
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
                            r0.shared.delta.buffer.ptr,
                            w0.y.buffer.ptr,
                        ],
                        &[Scalar::I32(total as i32)],
                        r0.stream.raw,
                    )
                })?;
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
        let decode_cycle = matches!(priority, ReadPriority::Decode | ReadPriority::DecodeFirst)
            && self.graphs_active();
        if self.graphs_active() {
            self.set_graph_cycle(priority);
        }
        if decode_cycle {
            self.begin_graph_cycle()?;
        }
        let initial_starts: Vec<usize> = placements.iter().map(|p| p.len).collect();
        let staged_frontiers = initial_starts.clone();
        let run_waves = self.prefill_waves;
        let kind = match priority {
            ReadPriority::Decode | ReadPriority::DecodeFirst => ExpertV2SourceKind::Decode,
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
                    self.engine.run_moe_router_and_shared(layer, batch, false)?;
                    if self.engine.ranks.len() == 2 && self.engine.prefill_waves > 1 {
                        let exchange = self.engine.peer_exchange()?;
                        exchange.signal(1, DIRECT)?;
                        exchange.wait(0, DIRECT)?;
                    }
                    let waves = self
                        .waves
                        .as_mut()
                        .context("Hy4 Spark route without active SparkLink")?;
                    let rows = batch.rows;
                    let identity = self.engine.next_identity;
                    self.engine.next_identity =
                        identity.checked_add(1).context("Hy4 Spark identity exhausted")?;
                    let version = self.engine.placement_version;
                    let dtype = ExpertV2Dtype::Fp8E4m3Ue8m0K32;
                    let staging = RouteStaging::new(rows, TOPK, dtype.row_bytes(H)?)?;
                    let kind = self.kind;
                    let origins: Vec<(u64, u64)> = unit
                        .segments
                        .iter()
                        .flat_map(|seg| {
                            seg.tokens
                                .clone()
                                .map(move |pos| (seg.sequence as u64 + 1, pos as u64))
                        })
                        .collect();
                    let packed = unit.reply_origins()?;
                    let combined = if kind == ExpertV2SourceKind::Prefill {
                        packed.as_ref().map_or_else(
                            || spark_waves::single_reply_eligible(rows, kind),
                            |o| o.combined_reply_eligible(),
                        )
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
                                rows * H * 2,
                                staging,
                                [
                                    r0.shared.route_ids.buffer,
                                    r0.shared.route_weights.buffer,
                                    r0.shared.wire_in.buffer,
                                ],
                                r0.stream.raw,
                                combined,
                                Box::new(move |routes, wire| {
                                    let mut req = build_hy4_spark_request(
                                        identity, version, layer, source, start, rows, kind, dtype, routes, wire,
                                    )?;
                                    if let Some(packed) = packed {
                                        packed.apply(&mut req)?;
                                    }
                                    ensure!(
                                        origins.len() == req.rows.len(),
                                        "Hy4 Spark origin count differs"
                                    );
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
                    if !batch.graphed {
                        self.engine.record_expert_usage(layer, batch)?;
                    }
                    Ok(Some(ticket))
                }
                fn land_experts(
                    &mut self,
                    layer: usize,
                    unit: &Unit,
                    ticket: Option<WaveTicket>,
                ) -> Result<()> {
                    let ticket = ticket.context("Hy4 land_experts missing WaveTicket")?;
                    let slot = ticket.slot();
                    let batch = self
                        .active
                        .get_mut(unit.slot)
                        .and_then(Option::as_mut)
                        .context("Hy4 land_experts without active unit")?;
                    let rows = batch.rows;
                    let waves = self
                        .waves
                        .as_mut()
                        .context("Hy4 Spark land without active SparkLink")?;
                    let r0 = &self.engine.ranks[0];
                    r0.device.run(|| {
                        waves.land(ticket, r0.stream.raw)?;
                        // SAFETY: wave landed on r0.stream before reduction into rank-0 routed_out.
                        unsafe {
                            waves.reduce(
                                slot,
                                std::ptr::null(),
                                r0.shared.routed_out.buffer.ptr.cast(),
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
                    self.engine.finish_head(batch, mode, &mut |row_off, logits| {
                        (self.on_tile)(unit, row_off, logits)
                    })
                }
                fn commit_unit(&mut self, unit: &Unit) -> Result<()> {
                    let batch = self
                        .active
                        .get_mut(unit.slot)
                        .and_then(Option::take)
                        .context("Hy4 commit_unit without active unit")?;
                    for group in &mut self.engine.metadata_uploads {
                        group[unit.slot].retire()?;
                    }
                    for m in &batch.members {
                        self.placements[m.sequence].len = m.pos_start + m.row_len;
                    }
                    Ok(())
                }
            }
            let mut hooks = Hooks {
                engine: self,
                waves: None,
                placements,
                initial_starts: &initial_starts,
                staged_frontiers,
                sources,
                priority,
                kind,
                head_mode,
                on_tile,
                active: (0..run_waves).map(|_| None).collect(),
            };
            if let Some(mut w) = waves {
                hooks.waves = Some(&mut w);
                run_prefill_units(&mut hooks, units, run_waves)?;
            } else {
                run_prefill_units(&mut hooks, units, run_waves)?;
            }
            if spark.is_some() {
                spark_waves::healthy()?;
            }
            Ok(())
        })();
        self.spark = spark;
        let result = match result {
            Ok(()) if decode_cycle => self.graph_completed(),
            Ok(()) => Ok(()),
            Err(err) => {
                if decode_cycle {
                    self.abandon_graph_cycle();
                }
                Err(err)
            }
        };
        if result.is_err() {
            self.terminate();
        }
        result
    }

    /// Single-sequence golden helper: forward `tokens` from `start_pos` through `0..=last_layer`
    /// and read back FP32 logits `[tokens.len(), vocab]` when `last_layer == cfg.layers - 1`.
    pub(crate) fn forward_golden(
        &mut self,
        tokens: &[u32],
        start_pos: usize,
        last_layer: usize,
        unit_rows: usize,
        decode: bool,
    ) -> Result<Vec<f32>> {
        ensure!(!tokens.is_empty(), "Hy4 golden tokens empty");
        ensure!(last_layer < self.cfg.layers, "Hy4 last_layer {last_layer} out of range");
        let vocab = self.cfg.vocab_size;
        let compute_head = last_layer == self.cfg.layers - 1;
        let mut logits = if compute_head {
            vec![0f32; tokens.len() * vocab]
        } else {
            Vec::new()
        };
        let mut placements = [self.golden_placement.clone()];
        placements[0].len = start_pos;
        let priority = if decode {
            ReadPriority::Decode
        } else {
            ReadPriority::Prefill
        };
        let lib = self.library();
        let head_mode = if compute_head {
            HeadMode::Suffix(0)
        } else {
            HeadMode::None
        };
        if decode {
            // Execute one 1-token decode step per `run_units` call so causal token-to-token
            // KV writes and per-step EventCycles graph boundaries match serving exactly.
            for off in 0..tokens.len() {
                let s = start_pos + off;
                let e = s + 1;
                let unit = Unit {
                    index: 0,
                    sequence: 0,
                    segments: vec![UnitSegment {
                        sequence: 0,
                        rows: 0..1,
                        tokens: s..e,
                        prompt_end: e == start_pos + tokens.len(),
                        snapshot: false,
                        head: compute_head,
                    }],
                    rows: s..e,
                    layers: 0..last_layer + 1,
                    slot: 0,
                    barrier: false,
                };
                self.run_units(
                    &mut placements,
                    &[&tokens[off..off + 1]],
                    &[unit],
                    priority,
                    |_| head_mode,
                    &mut |u, row_off, tile| {
                        let global_off = u.rows.start - start_pos + row_off;
                        let dst = &mut logits[global_off * vocab..(global_off + tile.rows) * vocab];
                        let host = tile.to_host(lib)?;
                        dst.copy_from_slice(&host);
                        Ok(())
                    },
                )?;
            }
        } else {
            let step = unit_rows.min(self.max_rows).max(1);
            let mut units = Vec::new();
            for (idx, off) in (0..tokens.len()).step_by(step).enumerate() {
                let len = (tokens.len() - off).min(step);
                let s = start_pos + off;
                let e = s + len;
                units.push(Unit {
                    index: idx,
                    sequence: 0,
                    segments: vec![UnitSegment {
                        sequence: 0,
                        rows: 0..len,
                        tokens: s..e,
                        prompt_end: e == start_pos + tokens.len(),
                        snapshot: false,
                        head: compute_head,
                    }],
                    rows: s..e,
                    layers: 0..last_layer + 1,
                    slot: idx % self.prefill_waves,
                    barrier: false,
                });
            }
            self.run_units(
                &mut placements,
                &[tokens],
                &units,
                priority,
                |_| head_mode,
                &mut |unit, row_off, tile| {
                    let global_off = unit.rows.start - start_pos + row_off;
                    let dst = &mut logits[global_off * vocab..(global_off + tile.rows) * vocab];
                    let host = tile.to_host(lib)?;
                    dst.copy_from_slice(&host);
                    Ok(())
                },
            )?;
        }
        self.golden_placement = placements.into_iter().next().unwrap();
        Ok(logits)
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
    source_kind: ExpertV2SourceKind,
    dtype: ExpertV2Dtype,
    routes: Vec<ExpertProtocolV2RouteEntry>,
    wire: bytes::Bytes,
) -> Result<ExpertProtocolV2Request> {
    ensure!(
        identity > 0 && version > 0 && source > 0 && (1..78).contains(&layer) && (1..=4096).contains(&rows),
        "Hy4 Spark request geometry/identity differs (layer={layer}, rows={rows})"
    );
    let descriptors = (0..rows)
        .map(|row| ExpertProtocolV2RowDescriptor {
            row_id: (start + row) as u64,
            source_kind,
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
        H as u32,
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
    #[test]
    fn hy4_graph_boundary_uses_event_cycles_without_host_drain() {
        let source = include_str!("execution.rs")
            .split("\n#[cfg(test)]\nmod tests")
            .next()
            .unwrap();
        assert!(
            source.contains("EventCycles::new("),
            "Hy4 must construct shared EventCycles across coordinator rank streams"
        );
        let begin = source
            .split("    fn begin_graph_cycle(")
            .nth(1)
            .unwrap()
            .split("    fn graph_completed(")
            .next()
            .unwrap();
        assert!(begin.contains("cycles.begin_persistent("));
        let completed = source
            .split("    fn graph_completed(")
            .nth(1)
            .unwrap()
            .split("    fn abandon_graph_cycle(")
            .next()
            .unwrap();
        assert!(completed.contains("cycles.complete()"));
        assert!(
            !completed.contains("self.drain()") && !completed.contains("observe_completed"),
            "Hy4 graph_completed must use EventCycles::complete without host drain or observe_completed"
        );
        let run = source
            .split("    pub(crate) fn run_units(")
            .nth(1)
            .unwrap()
            .split("    pub(crate) fn forward_golden(")
            .next()
            .unwrap();
        assert!(run.find("self.begin_graph_cycle()?").unwrap() < run.find("run_prefill_units(").unwrap());
        assert!(run.find("run_prefill_units(").unwrap() < run.find("self.graph_completed()").unwrap());
        assert!(run.contains("self.abandon_graph_cycle()"));
    }
}
