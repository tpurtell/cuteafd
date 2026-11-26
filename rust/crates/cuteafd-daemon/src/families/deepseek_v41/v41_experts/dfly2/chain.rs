//! V4.1 DFly2 speculative forward pass on one decoder GPU (with optional
//! TP2 routed experts and 2-RTX sharded vocabulary head).
//!
//! All temporary activations live in one packed `DeviceArena` (`v41/dfly2/workspace`)
//! on the decoder GPU (plus a tiny `v41/dfly2/peer-vocab` arena on GPU0 when split),
//! charged once by admission.

use super::{Dfly2ExpertStages, Dfly2TapStore, Dfly2Weights};
use crate::families::deepseek_v41::v41_experts::dspark::tp2;
use crate::families::deepseek_v41::v41_experts::{ExpertExecution, ExpertLayer, ExpertWeights};
use crate::families::deepseek_v41::v41_tensors::{
    fp8_head, fp8_scratch, project_vocabulary, Fp8Head, HeadScratch, NativeRtxTensors,
    VocabularyHead, VocabularyShard,
};
use crate::shared::experts::rtx::draft::pair::Pair;
use crate::shared::memory::device::{Allocation, Device, Event, Stream};
use crate::shared::memory::weight_arena::DeviceArena;
use crate::shared::memory::{HostAllocation, Tag};
use anyhow::{ensure, Context, Result};
use cuteafd_core::memory_layout::{arena_class, ArenaTag};
use cuteafd_ffi::{
    CuteafdDeviceBuffer, NativeLibrary, V41AttentionOps, V41Dfly2Blas, V41Dfly2Ops,
    V41Dfly2Request, V41GroupedOutput, V41Hc, V41Router, V41VocabularyProjection,
    V41_GROUPED_OUTPUT_WORKSPACE,
};
use cuteafd_loader::serving_capacity::WeightArenaLayout;
use std::collections::BTreeMap;
use std::ffi::c_void;

const HIDDEN: usize = 5120;
const INTER: usize = 2304;
const VOCAB: usize = 129280;
const SPLIT_SLOTS: usize = 192; // 192 * 64 * 514 * 4 = 25.26 MiB (10 parts for <=16 rows, 4 parts for <=48 rows)

fn slice_buf(buf: CuteafdDeviceBuffer, offset: usize, bytes: usize) -> Result<CuteafdDeviceBuffer> {
    ensure!(
        offset <= buf.bytes && bytes <= buf.bytes - offset,
        "DFly2 workspace slice out of bounds ({offset} + {bytes} > {})",
        buf.bytes
    );
    Ok(CuteafdDeviceBuffer {
        ptr: unsafe { buf.ptr.cast::<u8>().add(offset).cast() },
        bytes,
        ..buf
    })
}

fn split_parts(rows: usize) -> usize {
    if rows <= 16 {
        10
    } else if rows <= 48 {
        4
    } else {
        1
    }
}

pub(crate) enum Dfly2Vocab<'w, 'a> {
    Single(&'w VocabularyHead<'a>),
    Split([&'w VocabularyShard<'a>; 2]),
}

enum VocabRunner<'w, 'a> {
    Single {
        head: &'w VocabularyHead<'a>,
        projection: Option<V41VocabularyProjection<'a>>,
        fp8_scratch: Option<HeadScratch<'a>>,
    },
    Split {
        devices: [Device<'a>; 2],
        shards: [&'w VocabularyShard<'a>; 2],
        split: usize,
        peer_stream: Stream<'a>,
        peer_in_ready: Event<'a>,
        peer_out_ready: Event<'a>,
        peer_ops: V41Dfly2Ops<'a>,
        peer_proj: Option<V41VocabularyProjection<'a>>,
        peer_fp8_scratch: Option<HeadScratch<'a>>,
        home_proj: Option<V41VocabularyProjection<'a>>,
        home_fp8_scratch: Option<HeadScratch<'a>>,
        peer_input: Allocation<'a>,
        peer_logits: Allocation<'a>,
        peer_tile_scratch: Allocation<'a>,
        peer_ids: Allocation<'a>,
        peer_max: Allocation<'a>,
        peer_sum: Allocation<'a>,
        _peer_arena: DeviceArena<'a, arena_class::ReadWrite>,
    },
}

enum ExpertRunner<'w, 'a> {
    Full([ExpertExecution<'w, 'a>; 3]),
    Tp2 {
        pair: Pair<'a>,
        inputs: [Allocation<'a>; 3],
        shared: Allocation<'a>,
    },
}

struct Workspace<'a> {
    _arena: DeviceArena<'a, arena_class::ReadWrite>,
    anchor_ids: Allocation<'a>,
    curr_tokens: Allocation<'a>,
    all_tokens: Allocation<'a>,
    all_probs: Allocation<'a>,
    positions: Allocation<'a>,
    block_positions: Allocation<'a>,
    freqs: Allocation<'a>,
    slot_positions: Allocation<'a>,
    slot_freqs: Allocation<'a>,
    requests_dev: Allocation<'a>,
    prev_hidden_batch: Allocation<'a>,
    block_embed: Allocation<'a>,
    anchor_embed: Allocation<'a>,
    norm_a: Allocation<'a>,
    norm_b: Allocation<'a>,
    concat_10240: Allocation<'a>,
    fused: Allocation<'a>,
    hc_res_a: Allocation<'a>,
    hc_res_b: Allocation<'a>,
    hc_pre_a: Allocation<'a>,
    hc_pre_b: Allocation<'a>,
    hc_post: Allocation<'a>,
    hc_comb: Allocation<'a>,
    hc_scratch: Allocation<'a>,
    hc_collapsed: Allocation<'a>,
    hc_normed: Allocation<'a>,
    q_a: Allocation<'a>,
    q_n: Allocation<'a>,
    q_full: Allocation<'a>,
    kv_raw: Allocation<'a>,
    draft_kv: Allocation<'a>,
    attn_out: Allocation<'a>,
    attn_split_scratch: Allocation<'a>,
    ctx_alive: Allocation<'a>,
    wo_a_out: Allocation<'a>,
    sublayer_out: Allocation<'a>,
    grouped_ws: Allocation<'a>,
    router_scores: Allocation<'a>,
    shared_gate: Allocation<'a>,
    shared_up: Allocation<'a>,
    shared_act: Allocation<'a>,
    draft_hidden: Allocation<'a>,
    block_keys: Allocation<'a>,
    hc_gate: Allocation<'a>,
    hc_up: Allocation<'a>,
    hc_act: Allocation<'a>,
    hc_swiglu_delta: Allocation<'a>,
    hc_acorr_delta: Allocation<'a>,
    block_gate_logits: Allocation<'a>,
    acorr_combined: Allocation<'a>,
    blas_ws: Allocation<'a>,
    vocab_ws: Allocation<'a>,
    vocab_logits: Allocation<'a>,
    vocab_tile_scratch: Allocation<'a>,
    local_ids: Allocation<'a>,
    local_max: Allocation<'a>,
    local_sum: Allocation<'a>,
    recv_ids: Allocation<'a>,
    recv_max: Allocation<'a>,
    recv_sum: Allocation<'a>,
}

fn home_arena_layout(
    max_members: usize,
    block_size: usize,
    local_vocab: usize,
) -> Result<WeightArenaLayout> {
    let r = max_members;
    let b = block_size;
    let d = b - 1;
    let m = r * b;
    let mut e = BTreeMap::new();
    let mut add = |name: &str, bytes: usize| {
        e.insert(name.to_string(), bytes.max(256));
    };
    add("anchor_ids", r * 4);
    add("curr_tokens", r * 4);
    add("all_tokens", r * d * 4);
    add("all_probs", r * d * 4);
    add("positions", r * 8);
    add("block_positions", m * 8);
    add("freqs", m * 256);
    add("slot_positions", r * 8);
    add("slot_freqs", r * 256);
    add("requests_dev", 3 * r * std::mem::size_of::<V41Dfly2Request>());
    add("prev_hidden_batch", r * HIDDEN * 2);
    add("block_embed", m * HIDDEN * 2);
    add("anchor_embed", r * HIDDEN * 2);
    add("norm_a", r * HIDDEN * 2);
    add("norm_b", r * HIDDEN * 2);
    add("concat_10240", r * 2 * HIDDEN * 2);
    add("fused", r * HIDDEN * 2);
    add("hc_res_a", m * 4 * HIDDEN * 2);
    add("hc_res_b", m * 4 * HIDDEN * 2);
    add("hc_pre_a", m * 4 * 4);
    add("hc_pre_b", m * 4 * 4);
    add("hc_post", m * 4 * 4);
    add("hc_comb", m * 16 * 4);
    add("hc_scratch", m * 8000);
    add("hc_collapsed", m * HIDDEN * 2);
    add("hc_normed", m * HIDDEN * 2);
    add("q_a", m * 1280 * 2);
    add("q_n", m * 1280 * 2);
    add("q_full", m * 64 * 512 * 2);
    add("kv_raw", m * 512 * 2);
    add("draft_kv", m * 512 * 2);
    add("attn_out", m * 64 * 512 * 2);
    add("attn_split_scratch", SPLIT_SLOTS * 64 * 514 * 4);
    add("ctx_alive", r);
    add("wo_a_out", m * 8192 * 2);
    add("sublayer_out", m * HIDDEN * 2);
    add("grouped_ws", V41_GROUPED_OUTPUT_WORKSPACE);
    add("router_scores", m * 128 * 4);
    add("shared_gate", m * INTER * 2);
    add("shared_up", m * INTER * 2);
    add("shared_act", m * INTER * 2);
    add("draft_hidden", m * HIDDEN * 2);
    add("block_keys", r * b * 512 * 2);
    add("hc_gate", r * HIDDEN * 2);
    add("hc_up", r * HIDDEN * 2);
    add("hc_act", r * HIDDEN * 2);
    add("hc_swiglu_delta", r * HIDDEN * 2);
    add("hc_acorr_delta", r * HIDDEN * 2);
    add("block_gate_logits", r * 64 * 2);
    add("acorr_combined", r * 64 * 512 * 2);
    add("blas_ws", V41Dfly2Blas::WORKSPACE_BYTES);
    add("vocab_ws", V41VocabularyProjection::WORKSPACE_BYTES);
    add("vocab_logits", r * local_vocab * 4);
    add("vocab_tile_scratch", r * V41Dfly2Ops::VOCAB_TILE_SCRATCH_PER_ROW);
    add("local_ids", r * 4);
    add("local_max", r * 4);
    add("local_sum", r * 4);
    add("recv_ids", r * 4);
    add("recv_max", r * 4);
    add("recv_sum", r * 4);
    WeightArenaLayout::new("v41/dfly2/workspace", e)
}

fn peer_vocab_layout(max_members: usize, split: usize) -> Result<WeightArenaLayout> {
    let r = max_members;
    let mut e = BTreeMap::new();
    let mut add = |name: &str, bytes: usize| {
        e.insert(name.to_string(), bytes.max(256));
    };
    add("peer_input", r * HIDDEN * 2);
    add("peer_vocab_ws", V41VocabularyProjection::WORKSPACE_BYTES);
    add("peer_logits", r * split * 4);
    add("peer_tile_scratch", r * V41Dfly2Ops::VOCAB_TILE_SCRATCH_PER_ROW);
    add("peer_ids", r * 4);
    add("peer_max", r * 4);
    add("peer_sum", r * 4);
    WeightArenaLayout::new("v41/dfly2/peer-vocab", e)
}

fn draft_head_scratch_bytes(library: &NativeLibrary, rows: usize, vocab_cols: usize) -> Result<usize> {
    Ok(match fp8_head() {
        Fp8Head::Off => 0,
        Fp8Head::Packed => library.packed_head_workspace(rows, HIDDEN, vocab_cols)?.max(256),
        Fp8Head::Draft | Fp8Head::All => library.fp8_w8a16_workspace(rows, HIDDEN, vocab_cols)?.max(256),
    })
}

struct PendingDfly2<'a> {
    count: usize,
    _library: &'a NativeLibrary,
}

pub(crate) struct Dfly2Chain<'weights, 'library> {
    device: Device<'library>,
    stream: Stream<'library>,
    done: Event<'library>,
    weights: &'weights Dfly2Weights<'library>,
    embedding: &'weights NativeRtxTensors<'library>,
    vocab: VocabRunner<'weights, 'library>,
    experts: ExpertRunner<'weights, 'library>,
    dfly2_ops: V41Dfly2Ops<'library>,
    blas: V41Dfly2Blas<'library>,
    attn_ops: V41AttentionOps<'library>,
    hc: V41Hc<'library>,
    router: V41Router<'library>,
    grouped: V41GroupedOutput<'library>,
    ws: Workspace<'library>,
    host_upload: HostAllocation<'library>,
    host_download: HostAllocation<'library>,
    max_members: usize,
    block_size: usize,
    pending: Option<PendingDfly2<'library>>,
}

impl<'weights, 'library> Dfly2Chain<'weights, 'library> {
    pub fn single_device_bytes(
        library: &NativeLibrary,
        max_members: usize,
        block_size: usize,
    ) -> Result<usize> {
        let cap = Dfly2Weights::expert_capacity(max_members as u32, block_size, false)?;
        let home = home_arena_layout(max_members, block_size, VOCAB)?.bytes();
        let expert = ExpertWeights::plan_execution(
            ExpertLayer::Dspark { stage: 0 },
            library,
            cap,
            cap,
            false,
        )?
        .total()?
        .checked_mul(3)
        .context("DFly2 full expert workspace overflow")?;
        let head = draft_head_scratch_bytes(library, max_members, VOCAB)?;
        home.checked_add(expert)
            .and_then(|n| n.checked_add(head))
            .context("DFly2 single lane device_bytes overflow")
    }

    pub fn distributed_device_bytes(
        library: &NativeLibrary,
        max_members: usize,
        block_size: usize,
        tp2_experts: bool,
        split: usize,
    ) -> Result<[usize; 2]> {
        let cap = Dfly2Weights::expert_capacity(max_members as u32, block_size, tp2_experts)?;
        let home_arena = home_arena_layout(max_members, block_size, VOCAB - split)?.bytes();
        let peer_arena = peer_vocab_layout(max_members, split)?.bytes();
        let head0 = Device { library, id: 0 }.run(|| draft_head_scratch_bytes(library, max_members, split))?;
        let head1 = Device { library, id: 1 }.run(|| draft_head_scratch_bytes(library, max_members, VOCAB - split))?;
        let [exp0, exp1] = if tp2_experts {
            let mut b = Pair::device_bytes(library, cap, tp2::LAYOUT)?;
            b[1] += cap as usize * (10240 * 2 + 12 * 2);
            b
        } else {
            let e = ExpertWeights::plan_execution(
                ExpertLayer::Dspark { stage: 0 },
                library,
                cap,
                cap,
                false,
            )?
            .total()?
            .checked_mul(3)
            .context("DFly2 full expert workspace overflow")?;
            [0, e]
        };
        Ok([
            peer_arena + head0 + exp0,
            home_arena + head1 + exp1,
        ])
    }

    pub fn new(
        weights: &'weights Dfly2Weights<'library>,
        embedding: &'weights NativeRtxTensors<'library>,
        vocab: Dfly2Vocab<'weights, 'library>,
        max_members: usize,
    ) -> Result<Self> {
        let device = weights.device();
        let library = weights.library();
        let block_size = weights.block_size();
        let (table_bytes, table_device) = embedding.embedding_layout()?;
        ensure!(table_bytes == VOCAB * HIDDEN * 2, "invalid shared embedding extent");
        if table_device != device.id {
            device.run(|| library.cuda_enable_peer(table_device))?;
        }
        let local_vocab = match &vocab {
            Dfly2Vocab::Single(_) => VOCAB,
            Dfly2Vocab::Split(shards) => shards[1].tokens().len(),
        };
        let layout = home_arena_layout(max_members, block_size, local_vocab)?;
        let arena = device.run(|| {
            DeviceArena::new(device, layout.clone(), ArenaTag::drafter_workspace("v41/dfly2/workspace"))
        })?;
        let view = |name: &str| -> Result<Allocation<'library>> {
            let bytes = layout.view(name).with_context(|| format!("missing DFly2 view {name}"))?.bytes();
            arena.view(name, bytes)
        };
        let ws = Workspace {
            anchor_ids: view("anchor_ids")?,
            curr_tokens: view("curr_tokens")?,
            all_tokens: view("all_tokens")?,
            all_probs: view("all_probs")?,
            positions: view("positions")?,
            block_positions: view("block_positions")?,
            freqs: view("freqs")?,
            slot_positions: view("slot_positions")?,
            slot_freqs: view("slot_freqs")?,
            requests_dev: view("requests_dev")?,
            prev_hidden_batch: view("prev_hidden_batch")?,
            block_embed: view("block_embed")?,
            anchor_embed: view("anchor_embed")?,
            norm_a: view("norm_a")?,
            norm_b: view("norm_b")?,
            concat_10240: view("concat_10240")?,
            fused: view("fused")?,
            hc_res_a: view("hc_res_a")?,
            hc_res_b: view("hc_res_b")?,
            hc_pre_a: view("hc_pre_a")?,
            hc_pre_b: view("hc_pre_b")?,
            hc_post: view("hc_post")?,
            hc_comb: view("hc_comb")?,
            hc_scratch: view("hc_scratch")?,
            hc_collapsed: view("hc_collapsed")?,
            hc_normed: view("hc_normed")?,
            q_a: view("q_a")?,
            q_n: view("q_n")?,
            q_full: view("q_full")?,
            kv_raw: view("kv_raw")?,
            draft_kv: view("draft_kv")?,
            attn_out: view("attn_out")?,
            attn_split_scratch: view("attn_split_scratch")?,
            ctx_alive: view("ctx_alive")?,
            wo_a_out: view("wo_a_out")?,
            sublayer_out: view("sublayer_out")?,
            grouped_ws: view("grouped_ws")?,
            router_scores: view("router_scores")?,
            shared_gate: view("shared_gate")?,
            shared_up: view("shared_up")?,
            shared_act: view("shared_act")?,
            draft_hidden: view("draft_hidden")?,
            block_keys: view("block_keys")?,
            hc_gate: view("hc_gate")?,
            hc_up: view("hc_up")?,
            hc_act: view("hc_act")?,
            hc_swiglu_delta: view("hc_swiglu_delta")?,
            hc_acorr_delta: view("hc_acorr_delta")?,
            block_gate_logits: view("block_gate_logits")?,
            acorr_combined: view("acorr_combined")?,
            blas_ws: view("blas_ws")?,
            vocab_ws: view("vocab_ws")?,
            vocab_logits: view("vocab_logits")?,
            vocab_tile_scratch: view("vocab_tile_scratch")?,
            local_ids: view("local_ids")?,
            local_max: view("local_max")?,
            local_sum: view("local_sum")?,
            recv_ids: view("recv_ids")?,
            recv_max: view("recv_max")?,
            recv_sum: view("recv_sum")?,
            _arena: arena,
        };

        let vocab_runner = match vocab {
            Dfly2Vocab::Single(head) => device.run(|| {
                let fp8 = fp8_scratch(library, head.copy(), max_members, Fp8Head::Draft)?;
                let proj = if fp8.is_some() {
                    None
                } else {
                    Some(unsafe { library.v41_vocabulary_head(ws.vocab_ws.buffer, crate::shared::target_head::TargetHead::configured().native_mode())? })
                };
                Ok(VocabRunner::Single { head, projection: proj, fp8_scratch: fp8 })
            })?,
            Dfly2Vocab::Split(shards) => {
                let split = shards[0].tokens().end;
                ensure!(
                    shards[0].tokens() == (0..split) && shards[1].tokens() == (split..VOCAB),
                    "DFly2 split vocab shards do not partition {VOCAB}"
                );
                let devices = [Device { library, id: 0 }, Device { library, id: 1 }];
                ensure!(device.id == 1, "DFly2 split vocab expects decoder GPU1");
                for rank in 0..2 {
                    devices[rank].run(|| library.cuda_enable_peer(devices[1 - rank].id))?;
                }
                let p_layout = peer_vocab_layout(max_members, split)?;
                let peer_arena = devices[0].run(|| {
                    DeviceArena::new(
                        devices[0],
                        p_layout.clone(),
                        ArenaTag::drafter_workspace("v41/dfly2/peer-vocab"),
                    )
                })?;
                let pview = |name: &str| -> Result<Allocation<'library>> {
                    let bytes = p_layout.view(name).context("missing peer vocab view")?.bytes();
                    peer_arena.view(name, bytes)
                };
                let peer_ws = pview("peer_vocab_ws")?;
                let (peer_fp8_scratch, peer_proj, peer_ops) = devices[0].run(|| {
                    let fp8 = fp8_scratch(library, shards[0].copy(), max_members, Fp8Head::Draft)?;
                    let proj = if fp8.is_some() {
                        None
                    } else {
                        Some(unsafe { library.v41_vocabulary_shard(peer_ws.buffer, split, crate::shared::target_head::TargetHead::configured().native_mode())? })
                    };
                    let ops = library.v41_dfly2_ops()?;
                    Ok((fp8, proj, ops))
                })?;
                let (home_fp8_scratch, home_proj) = devices[1].run(|| {
                    let fp8 = fp8_scratch(library, shards[1].copy(), max_members, Fp8Head::Draft)?;
                    let proj = if fp8.is_some() {
                        None
                    } else {
                        Some(unsafe {
                            library.v41_vocabulary_shard(ws.vocab_ws.buffer, VOCAB - split, crate::shared::target_head::TargetHead::configured().native_mode())?
                        })
                    };
                    Ok((fp8, proj))
                })?;
                let peer_input = pview("peer_input")?;
                let peer_logits = pview("peer_logits")?;
                let peer_tile_scratch = pview("peer_tile_scratch")?;
                let peer_ids = pview("peer_ids")?;
                let peer_max = pview("peer_max")?;
                let peer_sum = pview("peer_sum")?;
                let _keep_peer_ws = peer_ws; // view backed by peer_arena
                VocabRunner::Split {
                    devices,
                    shards,
                    split,
                    peer_stream: Stream::new(devices[0])?,
                    peer_in_ready: Event::new(devices[1])?,
                    peer_out_ready: Event::new(devices[0])?,
                    peer_ops,
                    _peer_arena: peer_arena,
                    peer_input,
                    peer_logits,
                    peer_tile_scratch,
                    peer_ids,
                    peer_max,
                    peer_sum,
                    peer_proj,
                    peer_fp8_scratch,
                    home_proj,
                    home_fp8_scratch,
                }
            }
        };

        let cap = Dfly2Weights::expert_capacity(max_members as u32, block_size, weights.is_tp2())?;
        let experts = device.run(|| match weights.experts() {
            Dfly2ExpertStages::Full(stages) => {
                let per_stage = ExpertWeights::plan_execution(
                    ExpertLayer::Dspark { stage: 0 },
                    library,
                    cap,
                    cap,
                    false,
                )?
                .total()?;
                Ok(ExpertRunner::Full([
                    stages[0].execution_external(cap, cap, per_stage)?,
                    stages[1].execution_external(cap, cap, per_stage)?,
                    stages[2].execution_external(cap, cap, per_stage)?,
                ]))
            }
            Dfly2ExpertStages::Tp2(pair_weights) => {
                let pair = Pair::new(pair_weights.clone(), cap, tp2::LAYOUT)?;
                let c = cap as usize;
                let inputs = [
                    Allocation::new(device, c * 10240, Tag::drafter_workspace("v41/dfly2-tp2"))?,
                    Allocation::new(device, c * 12, Tag::drafter_workspace("v41/dfly2-tp2"))?,
                    Allocation::new(device, c * 12, Tag::drafter_workspace("v41/dfly2-tp2"))?,
                ];
                let shared = Allocation::new(device, c * 10240, Tag::drafter_workspace("v41/dfly2-tp2"))?;
                Ok(ExpertRunner::Tp2 { pair, inputs, shared })
            }
        })?;

        let (dfly2_ops, blas, attn_ops, hc, router, grouped) = device.run(|| {
            Ok((
                library.v41_dfly2_ops()?,
                unsafe { library.v41_dfly2_blas(ws.blas_ws.buffer)? },
                library.v41_attention_ops()?,
                library.v41_hc()?,
                library.v41_router()?,
                unsafe { library.v41_grouped_output(ws.grouped_ws.buffer)? },
            ))
        })?;

        let upload_bytes = max_members * 4
            + max_members * 8
            + max_members * block_size * 8
            + 3 * max_members * std::mem::size_of::<V41Dfly2Request>()
            + max_members * block_size * 8;
        let download_bytes = max_members * (block_size - 1) * 8;
        let host_upload = device.run(|| HostAllocation::new(library, upload_bytes, Tag::staging("v41/dfly2")))?;
        let host_download = device.run(|| HostAllocation::new(library, download_bytes, Tag::staging("v41/dfly2")))?;

        Ok(Self {
            device,
            stream: Stream::new(device)?,
            done: Event::new(device)?,
            weights,
            embedding,
            vocab: vocab_runner,
            experts,
            ws,
            dfly2_ops,
            blas,
            attn_ops,
            hc,
            router,
            grouped,
            host_upload,
            host_download,
            max_members,
            block_size,
            pending: None,
        })
    }

    pub fn device(&self) -> Device<'library> { self.device }
    pub fn draft_width(&self) -> usize { self.block_size - 1 }
    pub fn stream(&self) -> *mut c_void { self.stream.raw }

    pub fn cancel_pending(&mut self) {
        if self.pending.take().is_some() {
            let _ = self.stream.drain();
            if let VocabRunner::Split { peer_stream, .. } = &self.vocab {
                let _ = peer_stream.drain();
            }
        }
    }

    /// Launch one asynchronous DFly2 draft batch for `items` = `(request_id, anchor_id, committed_end)`.
    /// `context_reqs[stage][i]` describes target layer `[28, 32, 36][stage]`'s committed window ring
    /// and source 20 paged KV pool with `selected` pointing to `taps.recall_row(id, stage)`.
    ///
    /// # Safety
    /// All target window rings, source pages, and `taps` rows referenced by `context_reqs`
    /// must remain live and immutable until `poll()` returns `Some(...)` or `cancel_pending()`.
    pub unsafe fn begin(
        &mut self,
        taps: &Dfly2TapStore<'library>,
        items: &[(u64, u32, u64)],
        context_reqs: [&[V41Dfly2Request]; 3],
    ) -> Result<()> {
        ensure!(self.pending.is_none(), "DFly2 chain already has a pending proposal");
        let r = items.len();
        ensure!(
            r >= 1 && r <= self.max_members
                && context_reqs.iter().all(|c| c.len() == r),
            "invalid DFly2 batch size {r} (max {})",
            self.max_members
        );
        let b = self.block_size;
        let d = b - 1;
        let m = r * b;
        let lib = self.device.library;
        let s = self.stream.raw;
        let w = self.weights;

        // Pack all H2D metadata into pinned host_upload in one layout:
        // [anchor_ids: r*4] [positions: r*8] [block_positions: m*8]
        // [slot_positions: b*r*8] [requests_dev: 3*r*80]
        let req_size = std::mem::size_of::<V41Dfly2Request>();
        let off_pos = r * 4;
        let off_bpos = off_pos + r * 8;
        let off_spos = off_bpos + m * 8;
        let off_reqs = off_spos + b * r * 8;
        {
            let buf = self.host_upload.bytes_mut();
            for (i, &(_, anchor, pos)) in items.iter().enumerate() {
                buf[i * 4..i * 4 + 4].copy_from_slice(&anchor.to_ne_bytes());
                buf[off_pos + i * 8..off_pos + i * 8 + 8].copy_from_slice(&pos.to_ne_bytes());
                for slot in 0..b {
                    let row = i * b + slot;
                    let p = pos + slot as u64;
                    buf[off_bpos + row * 8..off_bpos + row * 8 + 8].copy_from_slice(&p.to_ne_bytes());
                    let sidx = slot * r + i;
                    buf[off_spos + sidx * 8..off_spos + sidx * 8 + 8].copy_from_slice(&p.to_ne_bytes());
                }
            }
            for stage in 0..3 {
                let stage_off = off_reqs + stage * r * req_size;
                let src = unsafe {
                    std::slice::from_raw_parts(
                        context_reqs[stage].as_ptr().cast::<u8>(),
                        r * req_size,
                    )
                };
                buf[stage_off..stage_off + r * req_size].copy_from_slice(src);
            }
        }

        let device = self.device;
        device.run(|| -> Result<()> {
            // 1. Async metadata uploads + gather prev_hidden rows.
            unsafe {
                let up = self.host_upload.bytes();
                lib.copy_h2d_async(self.ws.anchor_ids.buffer, &up[0..r * 4], s)?;
                lib.copy_h2d_async(self.ws.positions.buffer, &up[off_pos..off_pos + r * 8], s)?;
                lib.copy_h2d_async(self.ws.block_positions.buffer, &up[off_bpos..off_bpos + m * 8], s)?;
                lib.copy_h2d_async(self.ws.requests_dev.buffer, &up[off_reqs..off_reqs + 3 * r * req_size], s)?;
            }
            for (i, &(id, _, _)) in items.iter().enumerate() {
                let src = taps.prev_hidden_row(id)?;
                let dst = slice_buf(self.ws.prev_hidden_batch.buffer, i * HIDDEN * 2, HIDDEN * 2)?;
                unsafe { lib.copy_d2d_async(dst, src, HIDDEN * 2, s)?; }
            }

            // 2. Embed block + MTP fusion + seed 4-stream mHC.
            let embed_table = self.embedding.get("embed.weight")?;
            unsafe {
                self.dfly2_ops.embed_block(
                    embed_table,
                    self.ws.anchor_ids.buffer,
                    self.ws.block_embed.buffer,
                    Some(self.ws.anchor_embed.buffer),
                    r,
                    b,
                    w.config().mask_token_id,
                    s,
                )?;
                self.attn_ops.norm(
                    self.ws.anchor_embed.buffer,
                    w.tensor("model.mtp_fusion.enorm.weight")?,
                    None,
                    self.ws.norm_a.buffer,
                    r as u32,
                    HIDDEN as u32,
                    s,
                )?;
                self.attn_ops.norm(
                    self.ws.prev_hidden_batch.buffer,
                    w.tensor("model.mtp_fusion.hnorm.weight")?,
                    None,
                    self.ws.norm_b.buffer,
                    r as u32,
                    HIDDEN as u32,
                    s,
                )?;
                self.dfly2_ops.concat_bf16(
                    self.ws.norm_a.buffer,
                    self.ws.norm_b.buffer,
                    self.ws.concat_10240.buffer,
                    r,
                    HIDDEN,
                    HIDDEN,
                    s,
                )?;
                self.blas.gemm_bf16(
                    self.ws.concat_10240.buffer.ptr,
                    w.tensor("model.mtp_fusion.eh_proj.weight")?.ptr,
                    self.ws.fused.buffer.ptr,
                    r,
                    2 * HIDDEN,
                    HIDDEN,
                    s,
                )?;
                self.dfly2_ops.mtp_fuse_and_seed_hc(
                    self.ws.block_embed.buffer,
                    self.ws.fused.buffer,
                    w.tensor("model.mtp_fusion.gate")?,
                    self.ws.positions.buffer,
                    self.ws.hc_res_a.buffer,
                    self.ws.hc_pre_a.buffer,
                    r,
                    b,
                    s,
                )?;
                // Block RoPE frequencies (YaRN base 160000, identical to target layer 28).
                self.attn_ops.backbone_frequencies(
                    self.ws.block_positions.buffer,
                    self.ws.freqs.buffer,
                    m as u32,
                    28,
                    s,
                )?;
            }

            // 3. Three MoE layers (paired with target layers 28, 32, 36).
            let mut res_in = self.ws.hc_res_a.buffer;
            let mut res_out = self.ws.hc_res_b.buffer;
            let mut pre_in = self.ws.hc_pre_a.buffer;
            let mut pre_out = self.ws.hc_pre_b.buffer;
            let backbone_parts = split_parts(m);
            let backbone_partial = (backbone_parts > 1).then_some(self.ws.attn_split_scratch.buffer);

            for stage in 0..3 {
                let p = format!("model.layers.{stage}");
                // --- Attention sublayer ---
                unsafe {
                    self.hc_begin(
                        res_in,
                        pre_in,
                        w.tensor(&format!("{p}.hc_attn_fn"))?,
                        w.tensor(&format!("{p}.hc_attn_scale"))?,
                        w.tensor(&format!("{p}.hc_attn_base"))?,
                        w.tensor(&format!("{p}.attn_norm.weight"))?,
                        pre_out,
                        m,
                    )?;
                    self.blas.gemm_bf16(
                        self.ws.hc_normed.buffer.ptr,
                        w.tensor(&format!("{p}.attn.wq_a.weight"))?.ptr,
                        self.ws.q_a.buffer.ptr,
                        m,
                        HIDDEN,
                        1280,
                        s,
                    )?;
                    self.attn_ops.norm(
                        self.ws.q_a.buffer,
                        w.tensor(&format!("{p}.attn.q_norm.weight"))?,
                        None,
                        self.ws.q_n.buffer,
                        m as u32,
                        1280,
                        s,
                    )?;
                    self.blas.gemm_bf16(
                        self.ws.q_n.buffer.ptr,
                        w.tensor(&format!("{p}.attn.wq_b.weight"))?.ptr,
                        self.ws.q_full.buffer.ptr,
                        m,
                        1280,
                        64 * 512,
                        s,
                    )?;
                    self.attn_ops.rope(
                        self.ws.q_full.buffer,
                        self.ws.freqs.buffer,
                        self.ws.q_full.buffer,
                        m as u32,
                        64,
                        false,
                        s,
                    )?;
                    self.blas.gemm_bf16(
                        self.ws.hc_normed.buffer.ptr,
                        w.tensor(&format!("{p}.attn.wkv.weight"))?.ptr,
                        self.ws.kv_raw.buffer.ptr,
                        m,
                        HIDDEN,
                        512,
                        s,
                    )?;
                    self.attn_ops.norm(
                        self.ws.kv_raw.buffer,
                        w.tensor(&format!("{p}.attn.kv_norm.weight"))?,
                        Some(self.ws.freqs.buffer),
                        self.ws.draft_kv.buffer,
                        m as u32,
                        512,
                        s,
                    )?;
                    let stage_reqs = slice_buf(
                        self.ws.requests_dev.buffer,
                        stage * r * req_size,
                        r * req_size,
                    )?;
                    self.dfly2_ops.attention(
                        self.ws.q_full.buffer,
                        Some(self.ws.draft_kv.buffer),
                        w.tensor(&format!("{p}.attn.attn_sink"))?,
                        stage_reqs,
                        self.ws.attn_out.buffer,
                        None,
                        backbone_partial,
                        r,
                        b,
                        backbone_parts,
                        true,
                        s,
                    )?;
                    self.attn_ops.rope(
                        self.ws.attn_out.buffer,
                        self.ws.freqs.buffer,
                        self.ws.attn_out.buffer,
                        m as u32,
                        64,
                        true,
                        s,
                    )?;
                    self.grouped.launch(
                        self.ws.attn_out.buffer,
                        w.tensor(&format!("{p}.attn.wo_a.weight"))?,
                        self.ws.wo_a_out.buffer,
                        m as u32,
                        s,
                    )?;
                    self.blas.gemm_bf16(
                        self.ws.wo_a_out.buffer.ptr,
                        w.tensor(&format!("{p}.attn.wo_b.weight"))?.ptr,
                        self.ws.sublayer_out.buffer.ptr,
                        m,
                        8192,
                        HIDDEN,
                        s,
                    )?;
                    self.hc.post(
                        self.ws.sublayer_out.buffer,
                        res_in,
                        self.ws.hc_post.buffer,
                        self.ws.hc_comb.buffer,
                        res_out,
                        m,
                        s,
                    )?;
                }
                std::mem::swap(&mut res_in, &mut res_out);
                std::mem::swap(&mut pre_in, &mut pre_out);

                // --- MoE FFN sublayer ---
                unsafe {
                    self.hc_begin(
                        res_in,
                        pre_in,
                        w.tensor(&format!("{p}.hc_ffn_fn"))?,
                        w.tensor(&format!("{p}.hc_ffn_scale"))?,
                        w.tensor(&format!("{p}.hc_ffn_base"))?,
                        w.tensor(&format!("{p}.ffn_norm.weight"))?,
                        pre_out,
                        m,
                    )?;
                    // Shared BF16 expert (with swiglu_limit=10.0) into expert wave's shared buffer.
                    let (exp_inputs, exp_shared) = match &self.experts {
                        ExpertRunner::Full(execs) => (
                            execs[stage].inputs(),
                            execs[stage].shared.as_ref().context("missing full expert shared")?.buffer,
                        ),
                        ExpertRunner::Tp2 { inputs, shared, .. } => (
                            inputs.each_ref().map(|a| a.buffer),
                            shared.buffer,
                        ),
                    };
                    lib.copy_d2d_async(exp_inputs[0], self.ws.hc_normed.buffer, m * HIDDEN * 2, s)?;
                    let gate_bias = w.tensor(&format!("{p}.ffn.gate.bias"))?;
                    self.router.launch(
                        exp_inputs[0],
                        w.tensor(&format!("{p}.ffn.gate.weight"))?,
                        gate_bias,
                        gate_bias,
                        None,
                        self.ws.router_scores.buffer,
                        exp_inputs[1],
                        exp_inputs[2],
                        m,
                        128,
                        s,
                    )?;
                    self.blas.gemm_bf16(
                        exp_inputs[0].ptr,
                        w.tensor(&format!("{p}.ffn.shared_experts.w1.weight"))?.ptr,
                        self.ws.shared_gate.buffer.ptr,
                        m,
                        HIDDEN,
                        INTER,
                        s,
                    )?;
                    self.blas.gemm_bf16(
                        exp_inputs[0].ptr,
                        w.tensor(&format!("{p}.ffn.shared_experts.w3.weight"))?.ptr,
                        self.ws.shared_up.buffer.ptr,
                        m,
                        HIDDEN,
                        INTER,
                        s,
                    )?;
                    self.dfly2_ops.swiglu(
                        self.ws.shared_gate.buffer,
                        self.ws.shared_up.buffer,
                        self.ws.shared_act.buffer,
                        m,
                        INTER,
                        w.config().swiglu_limit,
                        s,
                    )?;
                    self.blas.gemm_bf16(
                        self.ws.shared_act.buffer.ptr,
                        w.tensor(&format!("{p}.ffn.shared_experts.w2.weight"))?.ptr,
                        exp_shared.ptr,
                        m,
                        INTER,
                        HIDDEN,
                        s,
                    )?;
                    let ffn_out = match &mut self.experts {
                        ExpertRunner::Full(execs) => {
                            execs[stage].launch_on(m as u32, true, s)?;
                            execs[stage].output().context("missing full expert output")?
                        }
                        ExpertRunner::Tp2 { pair, inputs, shared } => {
                            pair.enqueue(stage, m as u32, inputs.each_ref().map(|a| a.buffer), Some(shared.buffer), s)?;
                            pair.output()
                        }
                    };
                    self.hc.post(
                        ffn_out,
                        res_in,
                        self.ws.hc_post.buffer,
                        self.ws.hc_comb.buffer,
                        res_out,
                        m,
                        s,
                    )?;
                }
                std::mem::swap(&mut res_in, &mut res_out);
                std::mem::swap(&mut pre_in, &mut pre_out);
            }

            // 4. Final mHC collapse + final_norm -> draft_hidden [R, B, 5120].
            unsafe {
                self.hc.pre(res_in, pre_in, self.ws.hc_collapsed.buffer, m, s)?;
                self.attn_ops.norm(
                    self.ws.hc_collapsed.buffer,
                    w.tensor("model.final_norm.weight")?,
                    None,
                    self.ws.draft_hidden.buffer,
                    m as u32,
                    HIDDEN as u32,
                    s,
                )?;
            }

            // 5. Sequential hidden_correction + attentive_correction + shared LM head.
            let ac_reqs = slice_buf(self.ws.requests_dev.buffer, 2 * r * req_size, r * req_size)?;
            let ac_parts = split_parts(r);
            let ac_partial = (ac_parts > 1).then_some(self.ws.attn_split_scratch.buffer);

            // Slot 0: compute and store block_keys[:, 0] using (h_0, embed(anchor)).
            unsafe {
                lib.copy_h2d_async(self.ws.slot_positions.buffer, &self.host_upload.bytes()[off_spos..off_spos + r * 8], s)?;
            }
            unsafe {
                self.attn_ops.backbone_frequencies(
                    self.ws.slot_positions.buffer,
                    self.ws.slot_freqs.buffer,
                    r as u32,
                    28,
                    s,
                )?;
                self.dfly2_ops.slice_slot(
                    self.ws.draft_hidden.buffer,
                    self.ws.fused.buffer,
                    r,
                    b,
                    0,
                    HIDDEN,
                    s,
                )?;
                self.attn_ops.norm(
                    self.ws.fused.buffer,
                    w.tensor("model.hidden_correction.hidden_norm.weight")?,
                    None,
                    self.ws.norm_a.buffer,
                    r as u32,
                    HIDDEN as u32,
                    s,
                )?;
                self.attn_ops.norm(
                    self.ws.anchor_embed.buffer,
                    w.tensor("model.hidden_correction.embed_norm.weight")?,
                    None,
                    self.ws.norm_b.buffer,
                    r as u32,
                    HIDDEN as u32,
                    s,
                )?;
                self.dfly2_ops.concat_bf16(
                    self.ws.norm_a.buffer,
                    self.ws.norm_b.buffer,
                    self.ws.concat_10240.buffer,
                    r,
                    HIDDEN,
                    HIDDEN,
                    s,
                )?;
                self.blas.gemm_bf16(
                    self.ws.concat_10240.buffer.ptr,
                    w.tensor("model.hidden_correction.attention.kv_in.weight")?.ptr,
                    self.ws.kv_raw.buffer.ptr,
                    r,
                    2 * HIDDEN,
                    512,
                    s,
                )?;
                self.attn_ops.norm(
                    self.ws.kv_raw.buffer,
                    w.tensor("model.hidden_correction.attention.kv_norm.weight")?,
                    Some(self.ws.slot_freqs.buffer),
                    self.ws.draft_kv.buffer,
                    r as u32,
                    512,
                    s,
                )?;
                self.dfly2_ops.scatter_kv_slot(
                    self.ws.draft_kv.buffer,
                    self.ws.block_keys.buffer,
                    r,
                    b,
                    0,
                    s,
                )?;
            }

            // Slots 1..B: sequential correction + attentive correction + LM head.
            for slot in 1..b {
                unsafe {
                    let pos_off = off_spos + slot * r * 8;
                    lib.copy_h2d_async(self.ws.slot_positions.buffer, &self.host_upload.bytes()[pos_off..pos_off + r * 8], s)?;
                }
                unsafe {
                    self.attn_ops.backbone_frequencies(
                        self.ws.slot_positions.buffer,
                        self.ws.slot_freqs.buffer,
                        r as u32,
                        28,
                        s,
                    )?;
                    self.dfly2_ops.slice_slot(
                        self.ws.draft_hidden.buffer,
                        self.ws.fused.buffer,
                        r,
                        b,
                        slot,
                        HIDDEN,
                        s,
                    )?;
                    let prev_embed = if slot == 1 {
                        self.ws.anchor_embed.buffer
                    } else {
                        self.dfly2_ops.embed_tokens(
                            embed_table,
                            self.ws.curr_tokens.buffer,
                            self.ws.anchor_embed.buffer,
                            r,
                            s,
                        )?;
                        self.ws.anchor_embed.buffer
                    };
                    self.attn_ops.norm(
                        self.ws.fused.buffer,
                        w.tensor("model.hidden_correction.hidden_norm.weight")?,
                        None,
                        self.ws.norm_a.buffer,
                        r as u32,
                        HIDDEN as u32,
                        s,
                    )?;
                    self.attn_ops.norm(
                        prev_embed,
                        w.tensor("model.hidden_correction.embed_norm.weight")?,
                        None,
                        self.ws.norm_b.buffer,
                        r as u32,
                        HIDDEN as u32,
                        s,
                    )?;
                    self.dfly2_ops.concat_bf16(
                        self.ws.norm_a.buffer,
                        self.ws.norm_b.buffer,
                        self.ws.concat_10240.buffer,
                        r,
                        HIDDEN,
                        HIDDEN,
                        s,
                    )?;

                    // SwiGLU correction (unclamped: limit = 0.0).
                    self.blas.gemm_bf16(
                        self.ws.concat_10240.buffer.ptr,
                        w.tensor("model.hidden_correction.gate_proj.weight")?.ptr,
                        self.ws.hc_gate.buffer.ptr,
                        r,
                        2 * HIDDEN,
                        HIDDEN,
                        s,
                    )?;
                    self.blas.gemm_bf16(
                        self.ws.concat_10240.buffer.ptr,
                        w.tensor("model.hidden_correction.up_proj.weight")?.ptr,
                        self.ws.hc_up.buffer.ptr,
                        r,
                        2 * HIDDEN,
                        HIDDEN,
                        s,
                    )?;
                    self.dfly2_ops.swiglu(
                        self.ws.hc_gate.buffer,
                        self.ws.hc_up.buffer,
                        self.ws.hc_act.buffer,
                        r,
                        HIDDEN,
                        0.0,
                        s,
                    )?;
                    self.blas.gemm_bf16(
                        self.ws.hc_act.buffer.ptr,
                        w.tensor("model.hidden_correction.down_proj.weight")?.ptr,
                        self.ws.hc_swiglu_delta.buffer.ptr,
                        r,
                        HIDDEN,
                        HIDDEN,
                        s,
                    )?;

                    // Attentive correction: Q, block key, layer-36 context read, gated block read.
                    self.blas.gemm_bf16(
                        self.ws.concat_10240.buffer.ptr,
                        w.tensor("model.hidden_correction.attention.q_in.weight")?.ptr,
                        self.ws.q_a.buffer.ptr,
                        r,
                        2 * HIDDEN,
                        1280,
                        s,
                    )?;
                    self.attn_ops.norm(
                        self.ws.q_a.buffer,
                        w.tensor("model.hidden_correction.attention.q_norm.weight")?,
                        None,
                        self.ws.q_n.buffer,
                        r as u32,
                        1280,
                        s,
                    )?;
                    self.blas.gemm_bf16(
                        self.ws.q_n.buffer.ptr,
                        w.tensor("model.hidden_correction.attention.wq_b.weight")?.ptr,
                        self.ws.q_full.buffer.ptr,
                        r,
                        1280,
                        64 * 512,
                        s,
                    )?;
                    self.attn_ops.rope(
                        self.ws.q_full.buffer,
                        self.ws.slot_freqs.buffer,
                        self.ws.q_full.buffer,
                        r as u32,
                        64,
                        false,
                        s,
                    )?;
                    self.blas.gemm_bf16(
                        self.ws.concat_10240.buffer.ptr,
                        w.tensor("model.hidden_correction.attention.kv_in.weight")?.ptr,
                        self.ws.kv_raw.buffer.ptr,
                        r,
                        2 * HIDDEN,
                        512,
                        s,
                    )?;
                    self.attn_ops.norm(
                        self.ws.kv_raw.buffer,
                        w.tensor("model.hidden_correction.attention.kv_norm.weight")?,
                        Some(self.ws.slot_freqs.buffer),
                        self.ws.draft_kv.buffer,
                        r as u32,
                        512,
                        s,
                    )?;
                    self.dfly2_ops.scatter_kv_slot(
                        self.ws.draft_kv.buffer,
                        self.ws.block_keys.buffer,
                        r,
                        b,
                        slot,
                        s,
                    )?;
                    self.dfly2_ops.attention(
                        self.ws.q_full.buffer,
                        None,
                        w.tensor("model.hidden_correction.attention.attn_sink")?,
                        ac_reqs,
                        self.ws.attn_out.buffer,
                        Some(self.ws.ctx_alive.buffer),
                        ac_partial,
                        r,
                        1,
                        ac_parts,
                        true,
                        s,
                    )?;
                    self.blas.gemm_bf16(
                        self.ws.concat_10240.buffer.ptr,
                        w.tensor("model.hidden_correction.attention.block_gate.weight")?.ptr,
                        self.ws.block_gate_logits.buffer.ptr,
                        r,
                        2 * HIDDEN,
                        64,
                        s,
                    )?;
                    self.dfly2_ops.block_gated_add(
                        self.ws.q_full.buffer,
                        self.ws.block_keys.buffer,
                        self.ws.attn_out.buffer,
                        self.ws.ctx_alive.buffer,
                        self.ws.block_gate_logits.buffer,
                        w.tensor("model.hidden_correction.attention.block_rel_bias")?,
                        w.tensor("model.hidden_correction.attention.block_sink")?,
                        w.tensor("model.hidden_correction.attention.block_gate_bias")?,
                        self.ws.acorr_combined.buffer,
                        r,
                        b,
                        slot,
                        s,
                    )?;
                    self.attn_ops.rope(
                        self.ws.acorr_combined.buffer,
                        self.ws.slot_freqs.buffer,
                        self.ws.acorr_combined.buffer,
                        r as u32,
                        64,
                        true,
                        s,
                    )?;
                    self.grouped.launch(
                        self.ws.acorr_combined.buffer,
                        w.tensor("model.hidden_correction.attention.wo_a.weight")?,
                        self.ws.wo_a_out.buffer,
                        r as u32,
                        s,
                    )?;
                    self.blas.gemm_bf16(
                        self.ws.wo_a_out.buffer.ptr,
                        w.tensor("model.hidden_correction.attention.wo_b.weight")?.ptr,
                        self.ws.hc_acorr_delta.buffer.ptr,
                        r,
                        8192,
                        HIDDEN,
                        s,
                    )?;
                    // corrected = h_slot + swiglu_delta + acorr_delta (in norm_a buffer).
                    self.dfly2_ops.add3_bf16(
                        self.ws.fused.buffer,
                        self.ws.hc_swiglu_delta.buffer,
                        self.ws.hc_acorr_delta.buffer,
                        self.ws.norm_a.buffer,
                        r * HIDDEN,
                        s,
                    )?;

                    // Shared vocabulary head -> argmax + CandidateProbability.
                    self.project_and_select_slot(r, d, slot - 1)?;
                }
            }

            // 6. Single async D2H of all_tokens [R, D] and all_probs [R, D].
            let tok_bytes = r * d * 4;
            let (dst_tok, dst_prob) = self.host_download.bytes_mut()[..2 * tok_bytes].split_at_mut(tok_bytes);
            let src_tok = slice_buf(self.ws.all_tokens.buffer, 0, tok_bytes)?;
            let src_prob = slice_buf(self.ws.all_probs.buffer, 0, tok_bytes)?;
            unsafe {
                lib.copy_d2h_async(dst_tok, src_tok, s)?;
                lib.copy_d2h_async(dst_prob, src_prob, s)?;
                lib.cuda_event_record(self.done.raw, s)?;
            }
            Ok(())
        })?;

        self.pending = Some(PendingDfly2 { count: r, _library: lib });
        Ok(())
    }

    /// Poll an asynchronous DFly2 draft. Returns `(tokens [R * D], probs [R * D])`
    /// in request-major `[r * D + step]` order when ready.
    pub fn poll(&mut self) -> Result<Option<(Vec<u32>, Vec<f32>)>> {
        let Some(pending) = &self.pending else {
            anyhow::bail!("no pending DFly2 proposal");
        };
        let ready = self.device.run(|| unsafe { self.device.library.cuda_event_query(self.done.raw) })?;
        if !ready {
            return Ok(None);
        }
        let r = pending.count;
        self.pending = None;
        let d = self.block_size - 1;
        let tok_bytes = r * d * 4;
        let bytes = self.host_download.bytes();
        let tokens: Vec<u32> = bytes[..tok_bytes]
            .chunks_exact(4)
            .map(|c| u32::from_ne_bytes(c.try_into().unwrap()))
            .collect();
        let probs: Vec<f32> = bytes[tok_bytes..2 * tok_bytes]
            .chunks_exact(4)
            .map(|c| f32::from_ne_bytes(c.try_into().unwrap()))
            .collect();
        Ok(Some((tokens, probs)))
    }

    /// Synchronous convenience wrapper for single-lane `propose` and tests.
    ///
    /// # Safety
    /// Same lifetime contract as [`Self::begin`].
    pub unsafe fn propose_sync(
        &mut self,
        taps: &Dfly2TapStore<'library>,
        items: &[(u64, u32, u64)],
        context_reqs: [&[V41Dfly2Request]; 3],
    ) -> Result<(Vec<u32>, Vec<f32>)> {
        unsafe { self.begin(taps, items, context_reqs)? };
        self.stream.drain()?;
        self.poll()?.context("DFly2 synchronous draft did not complete after drain")
    }

    #[allow(clippy::too_many_arguments)]
    unsafe fn hc_begin(
        &self,
        res_in: CuteafdDeviceBuffer,
        pre_in: CuteafdDeviceBuffer,
        hc_fn: CuteafdDeviceBuffer,
        hc_scale: CuteafdDeviceBuffer,
        hc_base: CuteafdDeviceBuffer,
        norm_w: CuteafdDeviceBuffer,
        pre_out: CuteafdDeviceBuffer,
        rows: usize,
    ) -> Result<()> {
        let s = self.stream.raw;
        unsafe {
            let fused = self.hc.try_begin(
                res_in,
                hc_fn,
                hc_scale,
                hc_base,
                pre_in,
                norm_w,
                pre_out,
                self.ws.hc_post.buffer,
                self.ws.hc_comb.buffer,
                self.ws.hc_normed.buffer,
                self.ws.hc_scratch.buffer,
                rows,
                s,
            )?;
            if !fused {
                self.hc.mixes_workspace(
                    res_in,
                    hc_fn,
                    hc_scale,
                    hc_base,
                    pre_out,
                    self.ws.hc_post.buffer,
                    self.ws.hc_comb.buffer,
                    self.ws.hc_scratch.buffer,
                    rows,
                    s,
                )?;
                self.hc.pre(res_in, pre_in, self.ws.hc_collapsed.buffer, rows, s)?;
                self.attn_ops.norm(
                    self.ws.hc_collapsed.buffer,
                    norm_w,
                    None,
                    self.ws.hc_normed.buffer,
                    rows as u32,
                    HIDDEN as u32,
                    s,
                )?;
            }
        }
        Ok(())
    }

    unsafe fn project_and_select_slot(&self, r: usize, drafts: usize, slot_idx: usize) -> Result<()> {
        let lib = self.device.library;
        let s = self.stream.raw;
        match &self.vocab {
            VocabRunner::Single { head, projection, fp8_scratch } => unsafe {
                project_vocabulary(
                    lib,
                    projection.as_ref(),
                    head.weight(),
                    head.copy().zip(fp8_scratch.as_ref()),
                    self.ws.norm_a.buffer,
                    self.ws.vocab_logits.buffer,
                    r,
                    s,
                )?;
                self.dfly2_ops.shard_stats_f32(
                    self.ws.vocab_logits.buffer,
                    self.ws.vocab_tile_scratch.buffer,
                    self.ws.local_ids.buffer,
                    self.ws.local_max.buffer,
                    self.ws.local_sum.buffer,
                    r,
                    VOCAB,
                    s,
                )?;
                self.dfly2_ops.finalize_slot_f32(
                    self.ws.local_ids.buffer,
                    self.ws.local_sum.buffer,
                    self.ws.curr_tokens.buffer,
                    self.ws.all_tokens.buffer,
                    self.ws.all_probs.buffer,
                    r,
                    drafts,
                    slot_idx,
                    s,
                )?;
            },
            VocabRunner::Split {
                devices,
                shards,
                split,
                peer_stream,
                peer_in_ready,
                peer_out_ready,
                peer_ops,
                peer_input,
                peer_logits,
                peer_tile_scratch,
                peer_ids,
                peer_max,
                peer_sum,
                peer_proj,
                peer_fp8_scratch,
                home_proj,
                home_fp8_scratch,
                ..
            } => unsafe {
                // 1. Copy corrected hidden [R, 5120] from GPU1 -> GPU0, signal peer.
                lib.copy_peer_async(peer_input.buffer, self.ws.norm_a.buffer, r * HIDDEN * 2, s)?;
                lib.cuda_event_record(peer_in_ready.raw, s)?;
                // 2. Launch GPU0 shard [0, split) -> (id0, max0, sum0) -> copy 12 B/req to GPU1.
                devices[0].run(|| {
                    lib.cuda_stream_wait_event(peer_stream.raw, peer_in_ready.raw)?;
                    project_vocabulary(
                        lib,
                        peer_proj.as_ref(),
                        shards[0].weight(),
                        shards[0].copy().zip(peer_fp8_scratch.as_ref()),
                        peer_input.buffer,
                        peer_logits.buffer,
                        r,
                        peer_stream.raw,
                    )?;
                    peer_ops.shard_stats_f32(
                        peer_logits.buffer,
                        peer_tile_scratch.buffer,
                        peer_ids.buffer,
                        peer_max.buffer,
                        peer_sum.buffer,
                        r,
                        *split,
                        peer_stream.raw,
                    )?;
                    lib.copy_peer_async(self.ws.recv_ids.buffer, peer_ids.buffer, r * 4, peer_stream.raw)?;
                    lib.copy_peer_async(self.ws.recv_max.buffer, peer_max.buffer, r * 4, peer_stream.raw)?;
                    lib.copy_peer_async(self.ws.recv_sum.buffer, peer_sum.buffer, r * 4, peer_stream.raw)?;
                    lib.cuda_event_record(peer_out_ready.raw, peer_stream.raw)?;
                    Ok(())
                })?;
                // 3. Concurrently launch GPU1 shard [split, 129280) -> merge on GPU1.
                project_vocabulary(
                    lib,
                    home_proj.as_ref(),
                    shards[1].weight(),
                    shards[1].copy().zip(home_fp8_scratch.as_ref()),
                    self.ws.norm_a.buffer,
                    self.ws.vocab_logits.buffer,
                    r,
                    s,
                )?;
                self.dfly2_ops.shard_stats_f32(
                    self.ws.vocab_logits.buffer,
                    self.ws.vocab_tile_scratch.buffer,
                    self.ws.local_ids.buffer,
                    self.ws.local_max.buffer,
                    self.ws.local_sum.buffer,
                    r,
                    VOCAB - *split,
                    s,
                )?;
                lib.cuda_stream_wait_event(s, peer_out_ready.raw)?;
                self.dfly2_ops.merge_stats_f32(
                    self.ws.recv_ids.buffer,
                    self.ws.recv_max.buffer,
                    self.ws.recv_sum.buffer,
                    self.ws.local_ids.buffer,
                    self.ws.local_max.buffer,
                    self.ws.local_sum.buffer,
                    self.ws.curr_tokens.buffer,
                    self.ws.all_tokens.buffer,
                    self.ws.all_probs.buffer,
                    r,
                    *split,
                    drafts,
                    slot_idx,
                    s,
                )?;
            },
        }
        Ok(())
    }
}

impl Drop for Dfly2Chain<'_, '_> {
    fn drop(&mut self) {
        self.cancel_pending();
        let _ = self.stream.drain();
        if let VocabRunner::Split { peer_stream, .. } = &self.vocab {
            let _ = peer_stream.drain();
        }
    }
}
