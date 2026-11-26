//! Hy4 DFly2 proposals over the committed target's latent KV and recall.
//!
//! Supports both whole GPU0 execution (`tp = 1`) and head/MLP-split TP2 execution
//! (`tp = 2`) across both coordinator GPUs:
//! - Attention heads (16/16) and MLP neurons (4608/4608) split across both GPUs.
//! - Each rank gathers directly from its own target DSA KV cache (zero KV copy).
//! - Norms, `q_a_proj`, and `kv_a_proj_with_mqa` are replicated; `mtp_fusion`,
//!   `hidden_correction`, `final_norm`, and token argmax stay on lead rank 0.
//! - Reuses the target's existing device-driven `PeerExchange` (`PEER_SLOT_ATTN`,
//!   `PEER_SLOT_FFN`, `DIRECT`) strictly outside rank-local graph capture and
//!   `shared::dense_trellis::Projection::{core_into, reduced_f32}` for K8 F32
//!   partial reduction/egress.
use super::cache::slice_buf;
use super::graphs::{Hy4Graphs, Section};
use super::native::dequant_kv_b_trellis_absorbed;
use super::weights::{pack_kv_b_absorbed, read_tensor_bytes};
use crate::shared::dense_trellis::{
    codebook, CoreWeightLayout, CoreWeights, Projection, ProjectionLayout, ProjectionScratch,
};
use crate::shared::experts::rtx::dense::DenseLoad;
use crate::shared::memory::device::{Allocation, Device, Stream};
use crate::shared::memory::weight_arena::{DeviceArena, DeviceWeightArena};
use crate::shared::peer_split::{PeerExchange, DIRECT};
use crate::shared::token_io::{DeviceLogits, TokenEmbedding};
use anyhow::{ensure, Context, Result};
use cuteafd_core::memory_layout::{ArenaTag, Tag};
use cuteafd_ffi::dense_trellis::TrellisDType;
use cuteafd_ffi::programs::{Program, Programs, Scalar};
use cuteafd_ffi::{Bf16Add, CuteafdDeviceBuffer as Buffer, F32Add};
use cuteafd_loader::families::hy4_flash::{
    draft::{
        attention_scratch_bytes, matrix_split, trellis_id, Admission, RankAdmission, MAX_BATCH,
        MLP_WIDTH, PREFIX,
    },
    Hy4DflyConfig, Hy4FlashConfig, Hy4KvCache,
};
use cuteafd_loader::formats::dense_trellis::TrellisSide;
use cuteafd_loader::formats::trellis_core::CoreSplit;
use cuteafd_loader::plan::Checkpoint;
use std::{cell::Cell, collections::BTreeMap};

pub(super) const H: usize = 3072;
const TOPK: usize = 2048;
const KV_WIDTH: usize = 576;
const HEADS: usize = 32;
const PEER_SLOT_ATTN: usize = 0;
const PEER_SLOT_FFN: usize = 1;

fn weight_arena_name(rank: usize) -> &'static str {
    if rank == 0 {
        "hy4_flash/dfly2/gpu0"
    } else {
        "hy4_flash/dfly2/gpu1"
    }
}

fn workspace_arena_name(rank: usize, tp: usize) -> &'static str {
    match (tp, rank) {
        (1, _) => "hy4_flash/dfly2/workspace",
        (_, 0) => "hy4_flash/dfly2/workspace/gpu0",
        _ => "hy4_flash/dfly2/workspace/gpu1",
    }
}

fn trellis_tag(rank: usize, tp: usize) -> Tag {
    Tag::drafter_workspace(match (tp, rank) {
        (1, _) => "hy4_flash/dfly2/trellis",
        (_, 0) => "hy4_flash/dfly2/trellis/gpu0",
        _ => "hy4_flash/dfly2/trellis/gpu1",
    })
}

fn slice_bf16_matrix(raw: &[u8], shape: &[usize], split: CoreSplit, rank: usize, tp: usize) -> Result<Vec<u8>> {
    if tp == 1 || split == CoreSplit::Whole {
        return Ok(raw.to_vec());
    }
    ensure!(shape.len() == 2, "DFly2 split matrix must be 2D, got {shape:?}");
    let [rows, cols] = [shape[0], shape[1]];
    ensure!(raw.len() == rows * cols * 2, "DFly2 BF16 matrix byte length mismatch");
    match split {
        CoreSplit::Output => {
            ensure!(rows % tp == 0, "DFly2 output rows {rows} not divisible by TP{tp}");
            let span = raw.len() / tp;
            Ok(raw[rank * span..(rank + 1) * span].to_vec())
        }
        CoreSplit::Input => {
            ensure!(cols % tp == 0, "DFly2 input cols {cols} not divisible by TP{tp}");
            let local_cols = cols / tp;
            let row_bytes = cols * 2;
            let local_bytes = local_cols * 2;
            let col_off = rank * local_bytes;
            let mut out = Vec::with_capacity(rows * local_bytes);
            for r in 0..rows {
                let start = r * row_bytes + col_off;
                out.extend_from_slice(&raw[start..start + local_bytes]);
            }
            Ok(out)
        }
        CoreSplit::Whole => unreachable!(),
    }
}

fn layer_matrix_suffix(name: &str) -> Option<&str> {
    let rest = name.strip_prefix(PREFIX)?.strip_suffix(".weight")?;
    let (stage, suffix) = rest.split_once('.')?;
    stage.chars().all(|c| c.is_ascii_digit()).then_some(suffix)
}

struct DraftRank<'a> {
    rank: usize,
    tp: usize,
    local_heads: usize,
    local_mlp: usize,
    device: Device<'a>,
    weights: BTreeMap<String, Allocation<'a>>,
    workspace: BTreeMap<String, Allocation<'a>>,
    linear: BTreeMap<String, (CoreWeights<'a>, Projection<'a>)>,
    trellis: Option<ProjectionScratch<'a>>,
    programs: BTreeMap<&'static str, Program<'a>>,
    add: Bf16Add<'a>,
    f32_add: Option<F32Add<'a>>,
    has_peer_recall: bool,
}

impl<'a> DraftRank<'a> {
    #[allow(clippy::too_many_arguments)]
    fn new(
        device: Device<'a>,
        programs: &'a Programs<'a>,
        checkpoint: &Checkpoint,
        text: &Hy4FlashConfig,
        cfg: &Hy4DflyConfig,
        admission: RankAdmission,
        format: Hy4KvCache,
        slots: usize,
        tp: usize,
    ) -> Result<Self> {
        let rank = admission.rank;
        ensure!(matches!(tp, 1 | 2) && rank < tp, "invalid DFly2 rank {rank}/{tp}");
        let local_heads = HEADS / tp;
        let local_mlp = MLP_WIDTH / tp;
        let extent = cfg.block_rows() * slots;
        let layout = admission.weights;
        let arena = DeviceWeightArena::new(
            device,
            layout.arena.clone(),
            ArenaTag::drafter_weights(weight_arena_name(rank)),
        )?;
        let mut weights = BTreeMap::new();
        for (name, view) in layout.arena.views() {
            let stem = name.rsplit_once('.').context("invalid DFly2 arena name")?.0;
            if layout.native.contains_key(stem) {
                continue;
            }
            let dst = arena.view(name, view.bytes())?;
            if let Some(projection) = layout.absorbed.get(stem) {
                dequant_kv_b_trellis_absorbed(
                    device,
                    programs,
                    checkpoint,
                    projection,
                    rank,
                    tp,
                    local_heads,
                    text.qk_nope_dim,
                    text.v_head_dim,
                    text.kv_lora_rank,
                    dst.buffer,
                )?;
            } else {
                let (raw, _, shape) = read_tensor_bytes(checkpoint, name)?;
                let bytes = if name.ends_with(".kv_b_proj.weight") {
                    let sliced = slice_bf16_matrix(&raw, &shape, CoreSplit::Output, rank, tp)?;
                    pack_kv_b_absorbed(
                        &sliced,
                        local_heads,
                        text.qk_nope_dim,
                        text.v_head_dim,
                        text.kv_lora_rank,
                    )?
                } else if name.ends_with(".learnable_sink_param") {
                    let elem_span = raw.len() / tp;
                    raw[rank * elem_span..(rank + 1) * elem_span]
                        .chunks_exact(2)
                        .flat_map(|v| {
                            ((u16::from_le_bytes([v[0], v[1]]) as u32) << 16).to_le_bytes()
                        })
                        .collect()
                } else if let Some(suffix) = layer_matrix_suffix(name) {
                    slice_bf16_matrix(&raw, &shape, matrix_split(suffix), rank, tp)?
                } else {
                    raw
                };
                ensure!(
                    bytes.len() == view.bytes(),
                    "{name}: DFly2 rank {rank} upload extent {} differs from admission {}",
                    bytes.len(),
                    view.bytes()
                );
                device.run(|| device.library.copy_h2d(dst.buffer, &bytes))?;
            }
            weights.insert(name.clone(), dst);
        }
        let mut linear = BTreeMap::new();
        let dense = DenseLoad::mount(device.library);
        for (stem, admitted) in &layout.native {
            let core = CoreWeightLayout::from_shared(admitted.clone());
            let side = if core.side == TrellisSide::Input { 2 } else { 1 };
            let w = device.run(|| {
                core.load_recorded_with(
                    device,
                    checkpoint,
                    &dense,
                    std::ptr::null_mut(),
                    |name, bytes| arena.view(name, bytes),
                )
            })?;
            let projection = device.run(|| {
                Projection::load_geometry_bits(
                    programs,
                    core.shape[1],
                    core.shape[0],
                    side,
                    core.bits,
                    codebook(&w),
                    extent,
                    TrellisDType::Bf16,
                )
            })?;
            linear.insert(stem.clone(), (w, projection));
        }
        let trellis = layout
            .trellis_scratch(extent)?
            .map(|layout| {
                ProjectionLayout {
                    extents: layout.extents,
                    rows: layout.rows,
                }
                .allocate_packed(device, trellis_tag(rank, tp))
            })
            .transpose()?;

        let mut bound = BTreeMap::new();
        let prefix = if tp == 1 { "hy4f" } else { "hy4f2" };
        let norm: &[&str] = &["x", "weight", "bias", "cos", "sin", "positions", "out"];
        let proj: &[&str] = &["x", "w", "out"];
        let silu: &[&str] = &["gate", "up", "out"];
        for (key, suffix, args) in [
            ("norm", "rms_norm_3072", norm),
            ("q_a", "q_a", proj),
            ("kv_a", "kv_a", proj),
            ("q_b", "q_b_dsa", proj),
            ("gate", "linear_gate_dsa", proj),
            ("o", "o_proj_dsa", proj),
            ("gate_up", "draft_proj_gate_up", proj),
            ("down", "draft_proj_down", proj),
            ("silu", "draft_silu", silu),
            (
                "q_absorb",
                "mla_q_absorb_dsa",
                &["q_raw", "w_uk", "positions", "inv_freq", "query"][..],
            ),
            (
                "sink_uv",
                "mla_sink_gate_uv_dsa",
                &["attn", "w_uv", "gate", "lse", "sink", "out"][..],
            ),
        ] {
            bound.insert(key, programs.program(&format!("{prefix}_{suffix}"), args)?);
        }
        if rank == 0 {
            for (key, suffix, args) in [
                ("join", "draft_proj_join", proj),
                ("correction_down", "draft_proj_correction_down", proj),
                ("correction_silu", "draft_correction_silu", silu),
                (
                    "fusion",
                    "draft_gated_residual",
                    &["residual", "delta", "gate", "out"][..],
                ),
            ] {
                bound.insert(key, programs.program(&format!("hy4f_{suffix}"), args)?);
            }
        }
        let f = format.name();
        bound.insert(
            "gather",
            programs.program(
                &format!("hy4f_draft_gather_{f}"),
                &["pool", "table", "staging"],
            )?,
        );
        bound.insert(
            "pack",
            programs.program(
                &format!("{prefix}_mla_rank_norm_pack_{f}"),
                &[
                    "q_a",
                    "kv_a",
                    "q_norm",
                    "kv_norm",
                    "slots",
                    "positions",
                    "inv_freq",
                    "q_resid",
                    "kv_cache",
                ],
            )?,
        );
        bound.insert(
            "attention",
            programs.program(
                &format!("{prefix}_draft_attention_{f}"),
                &[
                    "query",
                    "kv_cache",
                    "page_table",
                    "cache_seqlens",
                    "cu_seqlens_q",
                    "attn",
                    "lse",
                    "scratch",
                ],
            )?,
        );
        let attn = usize::try_from(
            *bound["attention"]
                .spec()
                .scratch
                .get("scratch")
                .unwrap_or(&0),
        )?;
        ensure!(
            attn <= attention_scratch_bytes(MAX_BATCH, tp),
            "DFly2 TP{tp} attention scratch {attn} exceeds admission"
        );
        let run = DeviceArena::new(
            device,
            admission.workspace,
            ArenaTag::drafter_workspace(workspace_arena_name(rank, tp)),
        )?;
        let views = run
            .layout()
            .views()
            .iter()
            .map(|(name, v)| Ok((name.clone(), run.view(name, v.bytes())?)))
            .collect::<Result<BTreeMap<_, _>>>()?;
        let has_peer_recall = views.contains_key("tap_recall_peer");
        let add = device.run(|| device.library.bf16_add())?;
        let f32_add = if tp == 2 {
            Some(device.run(|| device.library.f32_add())?)
        } else {
            None
        };
        Ok(Self {
            rank,
            tp,
            local_heads,
            local_mlp,
            device,
            weights,
            workspace: views,
            linear,
            trellis,
            programs: bound,
            add,
            f32_add,
            has_peer_recall,
        })
    }

    fn w(&self, name: &str) -> Result<Buffer> {
        Ok(self
            .weights
            .get(name)
            .with_context(|| format!("missing DFly2 rank {} tensor {name}", self.rank))?
            .buffer)
    }

    fn b(&self, name: &str) -> Buffer {
        self.workspace[name].buffer
    }

    fn launch(
        &self,
        key: &str,
        buffers: &[Buffer],
        scalars: &[Scalar],
        stream: &Stream<'a>,
    ) -> Result<()> {
        // SAFETY: all caller operands are bounded live views of this rank's admitted
        // target or DFly2 arenas and use the same ordered stream.
        unsafe {
            self.programs[key].launch(
                &buffers.iter().map(|b| b.ptr).collect::<Vec<_>>(),
                scalars,
                stream.raw,
            )
        }
    }

    fn rows(&self, key: &str, buffers: &[Buffer], rows: usize, stream: &Stream<'a>) -> Result<()> {
        self.launch(key, buffers, &[Scalar::I32(rows as i32)], stream)
    }

    fn proj(
        &self,
        key: &str,
        stem: &str,
        from: Buffer,
        to: Buffer,
        rows: usize,
        stream: &Stream<'a>,
    ) -> Result<()> {
        if let Some((weights, p)) = self.linear.get(stem) {
            if key == "kv_a" {
                let padded = self.b("kv_padded");
                p.local(
                    rows,
                    from,
                    weights,
                    self.trellis
                        .as_ref()
                        .context("DFly2 K8 workspace missing")?,
                    padded,
                    stream.raw,
                )?;
                // SAFETY: disjoint admitted [rows, 640] and [rows, 576] BF16 planes.
                unsafe {
                    self.device.library.copy_d2d_2d_async(
                        to,
                        KV_WIDTH * 2,
                        padded,
                        640 * 2,
                        KV_WIDTH * 2,
                        rows,
                        stream.raw,
                    )
                }
            } else {
                p.local(
                    rows,
                    from,
                    weights,
                    self.trellis
                        .as_ref()
                        .context("DFly2 K8 workspace missing")?,
                    to,
                    stream.raw,
                )
            }
        } else {
            self.rows(
                key,
                &[from, self.w(&format!("{stem}.weight"))?, to],
                rows,
                stream,
            )
        }
    }

    fn proj_partial_f32(
        &self,
        key: &str,
        stem: &str,
        from: Buffer,
        to_f32: Buffer,
        rows: usize,
        stream: &Stream<'a>,
    ) -> Result<()> {
        if let Some((weights, p)) = self.linear.get(stem) {
            p.core_into(
                rows,
                from,
                weights,
                self.trellis
                    .as_ref()
                    .context("DFly2 K8 workspace missing")?,
                to_f32,
                stream.raw,
            )
        } else {
            self.rows(
                key,
                &[from, self.w(&format!("{stem}.weight"))?, to_f32],
                rows,
                stream,
            )
        }
    }

    /// Rank-local F32 partial reduction + egress (BF16 narrowing or K8 `svh` via
    /// `shared::dense_trellis::Projection::reduced_f32`) in physical rank order `[0, 1]`.
    fn reduce_f32_to_bf16(
        &self,
        stem: &str,
        rows: usize,
        peer_f32_ptr: *mut std::ffi::c_void,
        out_bf16: Buffer,
        stream: &Stream<'a>,
    ) -> Result<()> {
        let elems = rows * H;
        let bytes = elems * 4;
        let local_f32 = self.b("delta_f32");
        ensure!(
            local_f32.bytes >= bytes && !peer_f32_ptr.is_null(),
            "DFly2 TP2 F32 reduce operand invalid"
        );
        let local = Buffer { bytes, ..local_f32 };
        let peer = Buffer {
            ptr: peer_f32_ptr,
            bytes,
            device_id: self.device.id,
            flags: 0,
        };
        let (rank0, rank1) = if self.rank == 0 {
            (local, peer)
        } else {
            (peer, local)
        };
        let f32_add = self.f32_add.as_ref().context("DFly2 TP2 F32Add missing")?;
        let sum_f32 = self.b("delta_sum_f32");
        if let Some((weights, p)) = self.linear.get(stem) {
            p.reduced_f32(f32_add, rows, rank0, rank1, sum_f32, weights, out_bf16, stream.raw)
        } else {
            // SAFETY: PeerExchange::wait ordered peer_f32_ptr before this kernel; all operands are disjoint.
            unsafe {
                f32_add.launch(rank0, rank1, sum_f32, elems, stream.raw)?;
                self.device
                    .library
                    .cuda_f32_to_bf16_async(sum_f32, out_bf16, elems, stream.raw)
            }
        }
    }

    fn norm(
        &self,
        name: &str,
        from: Buffer,
        to: Buffer,
        rows: usize,
        stream: &Stream<'a>,
    ) -> Result<()> {
        let w = self.w(name)?;
        self.rows("norm", &[from, w, w, w, w, w, to], rows, stream)
    }

    fn copy(&self, dst: Buffer, src: Buffer, stream: &Stream<'a>) -> Result<()> {
        ensure!(dst.bytes <= src.bytes, "DFly2 copy source is too short");
        // SAFETY: disjoint bounded views ordered on this rank's stream.
        unsafe {
            self.device
                .library
                .copy_d2d_async(dst, src, dst.bytes, stream.raw)
        }
    }

    fn upload(&self, dst: Buffer, src: &[u8], stream: &Stream<'a>) -> Result<()> {
        // SAFETY: the FFI retains staging until the ordered transfer completes.
        unsafe { self.device.library.copy_h2d_ordered(dst, src, stream.raw) }
    }

    fn add(
        &self,
        dst: Buffer,
        a: Buffer,
        b: Buffer,
        rows: usize,
        stream: &Stream<'a>,
    ) -> Result<()> {
        // SAFETY: all views are disjoint, live and ordered on this rank.
        unsafe { self.add.launch(a, b, dst, rows * H, stream.raw) }
    }

    fn join_rows(
        &self,
        joined: Buffer,
        left: Buffer,
        right: Buffer,
        rows: usize,
        stream: &Stream<'a>,
    ) -> Result<()> {
        let width = H * 2;
        // SAFETY: each source is contiguous, and the two disjoint pitched destinations are [rows, 2H].
        unsafe {
            self.device.library.copy_d2d_2d_async(
                joined,
                width * 2,
                left,
                width,
                width,
                rows,
                stream.raw,
            )?;
            self.device.library.copy_d2d_2d_async(
                slice_buf(joined, width, joined.bytes - width)?,
                width * 2,
                right,
                width,
                width,
                rows,
                stream.raw,
            )
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn ffn_whole(
        &self,
        prefix: &str,
        gate_up: &str,
        down: &str,
        silu: &str,
        src: Buffer,
        dst: Buffer,
        rows: usize,
        width: usize,
        stream: &Stream<'a>,
    ) -> Result<()> {
        let bytes = rows * width * 2;
        let gate = slice_buf(self.b("gate_ffn"), 0, bytes)?;
        let up = slice_buf(self.b("up_ffn"), 0, bytes)?;
        let act = slice_buf(self.b("act"), 0, bytes)?;
        self.proj(
            gate_up,
            &format!("{prefix}.gate_proj"),
            src,
            gate,
            rows,
            stream,
        )?;
        self.proj(gate_up, &format!("{prefix}.up_proj"), src, up, rows, stream)?;
        self.rows(silu, &[gate, up, act], rows, stream)?;
        self.proj(down, &format!("{prefix}.down_proj"), act, dst, rows, stream)
    }

    fn ffn_partial_tp2(
        &self,
        prefix: &str,
        src: Buffer,
        dst_f32: Buffer,
        rows: usize,
        stream: &Stream<'a>,
    ) -> Result<()> {
        let bytes = rows * self.local_mlp * 2;
        let gate = slice_buf(self.b("gate_ffn"), 0, bytes)?;
        let up = slice_buf(self.b("up_ffn"), 0, bytes)?;
        let act = slice_buf(self.b("act"), 0, bytes)?;
        self.proj(
            "gate_up",
            &format!("{prefix}.gate_proj"),
            src,
            gate,
            rows,
            stream,
        )?;
        self.proj("gate_up", &format!("{prefix}.up_proj"), src, up, rows, stream)?;
        self.rows("silu", &[gate, up, act], rows, stream)?;
        self.proj_partial_f32("down", &format!("{prefix}.down_proj"), act, dst_f32, rows, stream)
    }

    fn recall(&self, layers: usize, verify: bool, row: usize, stage: usize) -> Result<Buffer> {
        slice_buf(
            self.b(if verify {
                "verify_recall"
            } else {
                "tap_recall"
            }),
            (row * layers + stage) * TOPK * 4,
            TOPK * 4,
        )
    }

    fn recall_peer(&self, layers: usize, verify: bool, row: usize, stage: usize) -> Result<Buffer> {
        ensure!(self.has_peer_recall, "DFly2 peer recall buffer not admitted");
        slice_buf(
            self.b(if verify {
                "verify_recall_peer"
            } else {
                "tap_recall_peer"
            }),
            (row * layers + stage) * TOPK * 4,
            TOPK * 4,
        )
    }

    fn hidden(&self, verify: bool, row: usize) -> Result<Buffer> {
        slice_buf(
            self.b(if verify {
                "verify_hidden"
            } else {
                "tap_hidden"
            }),
            row * H * 2,
            H * 2,
        )
    }

    fn inventory(
        &self,
        weights: &mut crate::shared::memory::receipts::Inventory<'_>,
        retained: &mut crate::shared::memory::receipts::Inventory<'_>,
    ) -> Result<()> {
        let prefix = if self.tp == 1 {
            String::new()
        } else {
            format!("rank{}.", self.rank)
        };
        for (name, allocation) in &self.weights {
            allocation.inventory(weights, &format!("{prefix}{name}"))?;
        }
        for (stem, (allocation, _)) in &self.linear {
            allocation
                .payload
                .inventory(weights, &format!("{prefix}{stem}.trellis"))?;
            allocation
                .suh
                .inventory(weights, &format!("{prefix}{stem}.suh"))?;
            allocation
                .svh
                .inventory(weights, &format!("{prefix}{stem}.svh"))?;
        }
        for (name, allocation) in &self.workspace {
            allocation.inventory(retained, &format!("{prefix}{name}"))?;
        }
        if let Some(scratch) = &self.trellis {
            scratch.allocation_receipts(retained, &trellis_id(self.rank, self.tp))?;
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn attention_core(
        &self,
        p: &str,
        a: &str,
        rows: usize,
        pt_width: usize,
        rope: Buffer,
        stream: &Stream<'a>,
    ) -> Result<()> {
        let state = self.b("state");
        let norm = self.b("norm");
        let kv = self.b("kv");
        self.norm(&format!("{p}.input_layernorm.weight"), state, norm, rows, stream)?;
        self.proj("q_a", &format!("{a}.q_a_proj"), norm, self.b("q_a"), rows, stream)?;
        self.proj(
            "kv_a",
            &format!("{a}.kv_a_proj_with_mqa"),
            norm,
            self.b("kv_a"),
            rows,
            stream,
        )?;
        self.proj(
            "gate",
            &format!("{a}.linear_gate"),
            norm,
            self.b("gate"),
            rows,
            stream,
        )?;
        self.rows(
            "pack",
            &[
                self.b("q_a"),
                self.b("kv_a"),
                self.w(&format!("{a}.q_a_layernorm.weight"))?,
                self.w(&format!("{a}.kv_a_layernorm.weight"))?,
                self.b("write_slots"),
                self.b("positions"),
                rope,
                self.b("q_norm"),
                kv,
            ],
            rows,
            stream,
        )?;
        self.proj(
            "q_b",
            &format!("{a}.q_b_proj"),
            self.b("q_norm"),
            self.b("q_b"),
            rows,
            stream,
        )?;
        let kv_b = self.w(&format!("{a}.kv_b_proj.weight"))?;
        let uk_bytes = self.local_heads * 192 * 512 * 2;
        let uk = slice_buf(kv_b, 0, uk_bytes)?;
        let uv = slice_buf(kv_b, uk_bytes, kv_b.bytes - uk_bytes)?;
        self.rows(
            "q_absorb",
            &[
                self.b("q_b"),
                uk,
                self.b("positions"),
                rope,
                self.b("query"),
            ],
            rows,
            stream,
        )?;
        self.launch(
            "attention",
            &[
                self.b("query"),
                kv,
                self.b("page_table"),
                self.b("lengths"),
                self.b("cu"),
                self.b("attn"),
                self.b("lse"),
                self.b("attn_scratch"),
            ],
            &[
                Scalar::I32(rows as i32),
                Scalar::I32(rows as i32),
                Scalar::I64(pt_width as i64),
                Scalar::I32(1),
            ],
            stream,
        )?;
        self.rows(
            "sink_uv",
            &[
                self.b("attn"),
                uv,
                self.b("gate"),
                self.b("lse"),
                self.w(&format!("{a}.learnable_sink_param"))?,
                self.b("uv"),
            ],
            rows,
            stream,
        )
    }
}

pub(super) struct Drafter<'a> {
    pub cfg: Hy4DflyConfig,
    format: Hy4KvCache,
    ranks: Vec<DraftRank<'a>>,
    frontiers: Vec<Cell<Option<usize>>>,
    verify_rows: usize,
    batch_capacity: usize,
}

impl<'a> Drafter<'a> {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        devices: &[Device<'a>],
        programs: &'a Programs<'a>,
        checkpoint: &Checkpoint,
        text: &Hy4FlashConfig,
        cfg: Hy4DflyConfig,
        admission: Admission,
        format: Hy4KvCache,
        slots: usize,
        verify_rows: usize,
    ) -> Result<Self> {
        let tp = admission.tp;
        ensure!(
            matches!(tp, 1 | 2) && devices.len() >= tp && admission.ranks.len() == tp,
            "DFly2 TP{tp} requires at least {tp} coordinator devices"
        );
        let mut ranks = Vec::with_capacity(tp);
        for rank_admission in admission.ranks {
            let idx = rank_admission.rank;
            let device = devices[idx];
            let rank = device.run(|| {
                DraftRank::new(
                    device,
                    programs,
                    checkpoint,
                    text,
                    &cfg,
                    rank_admission,
                    format,
                    slots,
                    tp,
                )
            })?;
            ranks.push(rank);
        }
        Ok(Self {
            cfg,
            format,
            ranks,
            frontiers: (0..slots).map(|_| Cell::new(None)).collect(),
            verify_rows,
            batch_capacity: slots,
        })
    }

    pub(super) fn tp(&self) -> usize {
        self.ranks.len()
    }

    pub(super) fn owner_devices(&self) -> Vec<i32> {
        self.ranks.iter().map(|r| r.device.id).collect()
    }

    pub(super) fn inventory(
        &self,
        weights: &mut crate::shared::memory::receipts::Inventory<'_>,
        retained: &mut crate::shared::memory::receipts::Inventory<'_>,
    ) -> Result<()> {
        for rank in &self.ranks {
            rank.inventory(weights, retained)?;
        }
        Ok(())
    }

    pub fn valid(&self, slot: usize, pos: usize) -> bool {
        self.frontiers[slot].get() == Some(pos)
    }
    pub fn invalidate(&self, slot: usize) {
        self.frontiers[slot].set(None);
    }
    pub fn publish(&self, slot: usize, pos: usize) {
        self.frontiers[slot].set(Some(pos));
    }

    pub fn captures_layer(&self, target_layer: usize) -> bool {
        self.cfg.target_index_layers.contains(&target_layer)
    }

    pub fn capture_recall(
        &self,
        rank: usize,
        verify: bool,
        row: usize,
        target_layer: usize,
        src: Buffer,
        stream: &Stream<'a>,
    ) -> Result<()> {
        let Some(local) = self.ranks.get(rank) else {
            return Ok(());
        };
        if let Some(stage) = self
            .cfg
            .target_index_layers
            .iter()
            .position(|&l| l == target_layer)
        {
            let dst = local.recall(self.cfg.layers, verify, row, stage)?;
            local.copy(dst, src, stream)?;
        }
        Ok(())
    }

    pub fn capture_recall_peer(
        &self,
        rank: usize,
        verify: bool,
        row: usize,
        target_layer: usize,
        src_on_peer: Buffer,
        stream: &Stream<'a>,
    ) -> Result<()> {
        let Some(local) = self.ranks.get(rank) else {
            return Ok(());
        };
        if let Some(stage) = self
            .cfg
            .target_index_layers
            .iter()
            .position(|&l| l == target_layer)
        {
            let dst = local.recall_peer(self.cfg.layers, verify, row, stage)?;
            local.copy(dst, src_on_peer, stream)?;
        }
        Ok(())
    }

    pub fn capture_hidden(
        &self,
        verify: bool,
        row: usize,
        src: Buffer,
        stream: &Stream<'a>,
    ) -> Result<()> {
        let lead = &self.ranks[0];
        lead.copy(lead.hidden(verify, row)?, src, stream)
    }

    pub fn accept(
        &self,
        slot: usize,
        row: usize,
        pos: usize,
        streams: &[&Stream<'a>],
    ) -> Result<()> {
        ensure!(
            row < self.verify_rows,
            "DFly2 verify tap row outside admission"
        );
        ensure!(
            streams.len() >= self.ranks.len(),
            "DFly2 accept requires one stream per drafter rank"
        );
        for (idx, local) in self.ranks.iter().enumerate() {
            let stream = streams[idx];
            local.device.run(|| {
                if idx == 0 {
                    local.copy(local.hidden(false, slot)?, local.hidden(true, row)?, stream)?;
                }
                for stage in 0..self.cfg.layers {
                    local.copy(
                        local.recall(self.cfg.layers, false, slot, stage)?,
                        local.recall(self.cfg.layers, true, row, stage)?,
                        stream,
                    )?;
                    if local.has_peer_recall {
                        local.copy(
                            local.recall_peer(self.cfg.layers, false, slot, stage)?,
                            local.recall_peer(self.cfg.layers, true, row, stage)?,
                            stream,
                        )?;
                    }
                }
                Ok(())
            })?;
        }
        self.publish(slot, pos);
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub fn propose(
        &self,
        streams: &[&Stream<'a>],
        embedding: &TokenEmbedding<'a>,
        members: &[Member],
        graphs: Option<&[&Hy4Graphs<'a>]>,
        ropes: &[Buffer],
        exchange: Option<&PeerExchange<'a>>,
        context_split: bool,
        cache: impl Fn(usize, usize) -> Result<Buffer>,
        mut head: impl FnMut(Buffer, usize) -> Result<DeviceLogits>,
    ) -> Result<Vec<Vec<u32>>> {
        if members.is_empty() {
            return Ok(Vec::new());
        }
        let tp = self.ranks.len();
        ensure!(
            streams.len() >= tp && ropes.len() >= tp && graphs.is_none_or(|g| g.len() >= tp),
            "DFly2 TP{tp} proposal streams/ropes/graphs incomplete"
        );
        ensure!(
            members.len() <= self.batch_capacity,
            "DFly2 batch exceeds admission"
        );
        ensure!(
            members.iter().all(|m| m.depth > 0
                && m.depth <= self.cfg.num_speculative_tokens
                && self.valid(m.slot, m.pos)),
            "DFly2 batch contains a stale tap or invalid depth"
        );
        let count = members.len();
        let depth = members.iter().map(|m| m.depth).max().unwrap();
        let width = self.cfg.block_rows();
        let rows = count * width;
        let record = self.format.record_bytes();
        let pt_width = TOPK + width;
        let meta = BatchMeta::new(members, width, self.cfg.mask_token_id)?;
        let row = |b: Buffer, i: usize| slice_buf(b, i * H * 2, H * 2);

        let r0 = &self.ranks[0];
        let s0 = streams[0];
        let g0 = graphs.map(|g| g[0]);
        let mut completed0 = s0.cancellation_guard("Hy4 DFly2 proposal rank 0");
        let mut completed1 = (tp == 2).then(|| streams[1].cancellation_guard("Hy4 DFly2 proposal rank 1"));

        let run_rank = |rank_idx: usize,
                        section: Section,
                        stage: usize,
                        work: &mut dyn FnMut() -> Result<()>|
         -> Result<()> {
            let local = &self.ranks[rank_idx];
            let stream = streams[rank_idx];
            local.device.run(|| match graphs {
                Some(g) => g[rank_idx].draft(rows, stage, section, stream, work),
                None => work(),
            })
        };

        // 1. Stage metadata to every participating drafter rank first so GPU1
        //    overlaps its H2D copies (and stage-0 KV gather on TP2) with GPU0's
        //    embedding + MTP fusion before the DIRECT state broadcast.
        let pos_bytes = words(&meta.positions);
        let write_bytes = words(&meta.write_slots);
        let table_bytes = words(&meta.table);
        let len_bytes = words(&meta.lengths);
        let cu_bytes = words(&(0..=rows as i32).collect::<Vec<_>>());
        let anchor_bytes = words(&meta.anchors);
        for (rank_idx, local) in self.ranks.iter().enumerate() {
            let stream = streams[rank_idx];
            local.device.run(|| {
                for (name, data) in [
                    ("positions", &pos_bytes),
                    ("write_slots", &write_bytes),
                    ("page_table", &table_bytes),
                    ("lengths", &len_bytes),
                    ("cu", &cu_bytes),
                ] {
                    local.upload(local.b(name), data, stream)?;
                }
                if rank_idx == 0 {
                    local.upload(local.b("anchor_ids"), &anchor_bytes, stream)?;
                }
                ensure!(
                    local.b("kv").bytes >= count * pt_width * record,
                    "DFly2 rank {rank_idx} KV staging is short"
                );
                Ok(())
            })?;
        }

        // 2. Lead rank embedding + MTP fusion -> `r0.state`.
        r0.device.run(|| {
            let state = r0.b("state");
            let anchor = r0.b("anchor");
            let features = r0.b("features");
            embedding.embed(&meta.ids, r0.b("ids"), 1, state, s0.raw)?;
            for (i, member) in members.iter().enumerate() {
                r0.copy(row(anchor, i)?, row(state, i * width)?, s0)?;
                r0.copy(row(features, i)?, r0.hidden(false, member.slot)?, s0)?;
            }
            Ok(())
        })?;
        run_rank(0, Section::DraftFusion, 0, &mut || {
            let f = format!("{PREFIX}mtp_fusion");
            let state = r0.b("state");
            let residual = r0.b("residual");
            let delta = r0.b("delta");
            let join = r0.b("joined");
            let left = r0.b("join_left");
            let right = r0.b("join_right");
            let anchor = r0.b("anchor");
            let features = r0.b("features");
            r0.norm(&format!("{f}.enorm.weight"), anchor, left, count, s0)?;
            r0.norm(&format!("{f}.hnorm.weight"), features, right, count, s0)?;
            r0.join_rows(join, left, right, count, s0)?;
            r0.proj("join", &format!("{f}.eh_proj"), join, delta, count, s0)?;
            r0.rows(
                "fusion",
                &[anchor, delta, r0.w(&format!("{f}.gate"))?, residual],
                count,
                s0,
            )?;
            for i in 0..count {
                r0.copy(row(state, i * width)?, row(residual, i)?, s0)?;
            }
            Ok(())
        })?;

        // 3. On TP2, overlap GPU1's stage-0 target KV gather with GPU0's fusion,
        //    then broadcast fused `state` from rank 0 -> rank 1 via PeerExchange DIRECT.
        if tp == 2 {
            let r1 = &self.ranks[1];
            let s1 = streams[1];
            let target0 = self.cfg.target_kv_layers[0];
            let layer_cache = cache(1, target0)?;
            let peer_cache = context_split.then(|| cache(0, target0)).transpose()?;
            let kv1 = r1.b("kv");
            r1.device.run(|| {
                for (i, member) in members.iter().enumerate() {
                    let staging = slice_buf(kv1, i * pt_width * record, pt_width * record)?;
                    r1.launch(
                        "gather",
                        &[
                            layer_cache,
                            r1.recall(self.cfg.layers, false, member.slot, 0)?,
                            staging,
                        ],
                        &[
                            Scalar::I32(meta.recall[i] as i32),
                            Scalar::I32(1),
                            Scalar::I32(0),
                        ],
                        s1,
                    )?;
                    if let Some(peer_pool) = peer_cache {
                        r1.launch(
                            "gather",
                            &[
                                peer_pool,
                                r1.recall_peer(self.cfg.layers, false, member.slot, 0)?,
                                staging,
                            ],
                            &[
                                Scalar::I32(meta.recall[i] as i32),
                                Scalar::I32(1),
                                Scalar::I32(0),
                            ],
                            s1,
                        )?;
                    }
                }
                Ok(())
            })?;
            let ex = exchange.context("DFly2 TP2 requires target PeerExchange")?;
            let state_bytes = rows * H * 2;
            ex.push_to(
                0,
                DIRECT,
                r0.b("state").ptr,
                r1.b("state").ptr,
                state_bytes,
            )?;
            ex.wait(1, DIRECT)?;
        }

        // 4. Five absorbed-MLA + SwiGLU DFly2 layers.
        if tp == 1 {
            let kv = r0.b("kv");
            let state = r0.b("state");
            let residual = r0.b("residual");
            let norm = r0.b("norm");
            let delta = r0.b("delta");
            for (stage, &target) in self.cfg.target_kv_layers.iter().enumerate() {
                let p = format!("{PREFIX}{stage}");
                let a = format!("{p}.self_attn");
                let layer_cache = cache(0, target)?;
                let peer_cache = context_split.then(|| cache(1, target)).transpose()?;
                r0.device.run(|| {
                    for (i, member) in members.iter().enumerate() {
                        let staging = slice_buf(kv, i * pt_width * record, pt_width * record)?;
                        r0.launch(
                            "gather",
                            &[
                                layer_cache,
                                r0.recall(self.cfg.layers, false, member.slot, stage)?,
                                staging,
                            ],
                            &[
                                Scalar::I32(meta.recall[i] as i32),
                                Scalar::I32(1),
                                Scalar::I32(0),
                            ],
                            s0,
                        )?;
                        if let Some(peer_pool) = peer_cache {
                            r0.launch(
                                "gather",
                                &[
                                    peer_pool,
                                    r0.recall_peer(self.cfg.layers, false, member.slot, stage)?,
                                    staging,
                                ],
                                &[
                                    Scalar::I32(meta.recall[i] as i32),
                                    Scalar::I32(1),
                                    Scalar::I32(0),
                                ],
                                s0,
                            )?;
                        }
                    }
                    Ok(())
                })?;
                run_rank(0, Section::DraftLayer, stage, &mut || {
                    r0.attention_core(&p, &a, rows, pt_width, ropes[0], s0)?;
                    r0.proj("o", &format!("{a}.o_proj"), r0.b("uv"), delta, rows, s0)?;
                    r0.norm(&format!("{a}.post_proj_norm.weight"), delta, norm, rows, s0)?;
                    r0.add(residual, state, norm, rows, s0)?;
                    r0.norm(
                        &format!("{p}.post_attention_layernorm.weight"),
                        residual,
                        norm,
                        rows,
                        s0,
                    )?;
                    r0.ffn_whole(
                        &format!("{p}.mlp"),
                        "gate_up",
                        "down",
                        "silu",
                        norm,
                        delta,
                        rows,
                        MLP_WIDTH,
                        s0,
                    )?;
                    r0.add(state, residual, delta, rows, s0)?;
                    if stage + 1 == self.cfg.layers {
                        r0.norm(&format!("{PREFIX}final_norm.weight"), state, norm, rows, s0)?;
                    }
                    Ok(())
                })?;
            }
        } else {
            let ex = exchange.context("DFly2 TP2 requires target PeerExchange")?;
            let attn_recv = [ex.recv(0, PEER_SLOT_ATTN)?, ex.recv(1, PEER_SLOT_ATTN)?];
            let ffn_recv = [ex.recv(0, PEER_SLOT_FFN)?, ex.recv(1, PEER_SLOT_FFN)?];
            let partial_bytes = rows * H * 4;
            for (stage, &target) in self.cfg.target_kv_layers.iter().enumerate() {
                let p = format!("{PREFIX}{stage}");
                let a = format!("{p}.self_attn");
                let prev_down = (stage > 0).then(|| format!("{PREFIX}{}.mlp.down_proj", stage - 1));
                let o_stem = format!("{a}.o_proj");

                // (a) Per-rank target KV gather outside capture (local half + P2P peer half under
                //     ContextSplit; rank 1 stage 0 already ran overlapped with GPU0's DraftFusion).
                for (rank_idx, local) in self.ranks.iter().enumerate() {
                    if stage == 0 && rank_idx == 1 {
                        continue;
                    }
                    let stream = streams[rank_idx];
                    let layer_cache = cache(rank_idx, target)?;
                    let peer_cache = context_split.then(|| cache(1 - rank_idx, target)).transpose()?;
                    let kv = local.b("kv");
                    local.device.run(|| {
                        for (i, member) in members.iter().enumerate() {
                            let staging = slice_buf(kv, i * pt_width * record, pt_width * record)?;
                            local.launch(
                                "gather",
                                &[
                                    layer_cache,
                                    local.recall(self.cfg.layers, false, member.slot, stage)?,
                                    staging,
                                ],
                                &[
                                    Scalar::I32(meta.recall[i] as i32),
                                    Scalar::I32(1),
                                    Scalar::I32(0),
                                ],
                                stream,
                            )?;
                            if let Some(peer_pool) = peer_cache {
                                local.launch(
                                    "gather",
                                    &[
                                        peer_pool,
                                        local.recall_peer(self.cfg.layers, false, member.slot, stage)?,
                                        staging,
                                    ],
                                    &[
                                        Scalar::I32(meta.recall[i] as i32),
                                        Scalar::I32(1),
                                        Scalar::I32(0),
                                    ],
                                    stream,
                                )?;
                            }
                        }
                        Ok(())
                    })?;
                }

                // (b) Rank-local attention (fuses prior layer's deferred FFN reduce/add).
                //     Keep the 5-layer TP2 body eager so 21 extra per-width draft graphs
                //     do not evict the 42-layer target's bounded 204/202 MiB graph cache.
                for rank_idx in 0..2 {
                    let local = &self.ranks[rank_idx];
                    let stream = streams[rank_idx];
                    let rope = ropes[rank_idx];
                    let recv_ffn = ffn_recv[rank_idx];
                    local.device.run(|| {
                        if let Some(prev) = prev_down.as_deref() {
                            let delta = local.b("delta");
                            local.reduce_f32_to_bf16(prev, rows, recv_ffn, delta, stream)?;
                            local.add(local.b("state"), local.b("residual"), delta, rows, stream)?;
                        }
                        local.attention_core(&p, &a, rows, pt_width, rope, stream)?;
                        local.proj_partial_f32(
                            "o",
                            &o_stem,
                            local.b("uv"),
                            local.b("delta_f32"),
                            rows,
                            stream,
                        )
                    })?;
                }

                // (c) Device-driven PeerExchange on PEER_SLOT_ATTN.
                ex.push(0, PEER_SLOT_ATTN, self.ranks[0].b("delta_f32").ptr, partial_bytes)?;
                ex.push(1, PEER_SLOT_ATTN, self.ranks[1].b("delta_f32").ptr, partial_bytes)?;
                ex.wait(0, PEER_SLOT_ATTN)?;
                ex.wait(1, PEER_SLOT_ATTN)?;

                // (d) Rank-local attention reduce + norms + split MLP partial (eager to
                //     avoid polluting GPU0/GPU1 target graph reserves).
                for rank_idx in 0..2 {
                    let local = &self.ranks[rank_idx];
                    let stream = streams[rank_idx];
                    let recv_attn = attn_recv[rank_idx];
                    local.device.run(|| {
                        let state = local.b("state");
                        let residual = local.b("residual");
                        let norm = local.b("norm");
                        let delta = local.b("delta");
                        local.reduce_f32_to_bf16(&o_stem, rows, recv_attn, delta, stream)?;
                        local.norm(&format!("{a}.post_proj_norm.weight"), delta, norm, rows, stream)?;
                        local.add(residual, state, norm, rows, stream)?;
                        local.norm(
                            &format!("{p}.post_attention_layernorm.weight"),
                            residual,
                            norm,
                            rows,
                            stream,
                        )?;
                        local.ffn_partial_tp2(
                            &format!("{p}.mlp"),
                            norm,
                            local.b("delta_f32"),
                            rows,
                            stream,
                        )
                    })?;
                }

                // (e) Device-driven PeerExchange on PEER_SLOT_FFN.
                //     On the final layer, only rank 0 consumes the result (rank0-only tail).
                let last_layer = stage + 1 == self.cfg.layers;
                if !last_layer {
                    ex.push(0, PEER_SLOT_FFN, self.ranks[0].b("delta_f32").ptr, partial_bytes)?;
                }
                ex.push(1, PEER_SLOT_FFN, self.ranks[1].b("delta_f32").ptr, partial_bytes)?;
                ex.wait(0, PEER_SLOT_FFN)?;
                if !last_layer {
                    ex.wait(1, PEER_SLOT_FFN)?;
                }
            }

            // (f) Rank-0-only final layer FFN reduce + residual + final_norm (3 tiny kernels, eager).
            let last_stage = self.cfg.layers - 1;
            let last_down = format!("{PREFIX}{last_stage}.mlp.down_proj");
            r0.device.run(|| {
                let state = r0.b("state");
                let residual = r0.b("residual");
                let norm = r0.b("norm");
                let delta = r0.b("delta");
                r0.reduce_f32_to_bf16(&last_down, rows, ffn_recv[0], delta, s0)?;
                r0.add(state, residual, delta, rows, s0)?;
                r0.norm(&format!("{PREFIX}final_norm.weight"), state, norm, rows, s0)
            })?;
        }

        // 5. Lead-rank sequential hidden_correction + shared sharded lm_head + argmax.
        let c = format!("{PREFIX}hidden_correction");
        let norm = r0.b("norm");
        let residual = r0.b("residual");
        let delta = r0.b("delta");
        let join = r0.b("joined");
        let left = r0.b("join_left");
        let right = r0.b("join_right");
        let anchor = r0.b("anchor");
        let features = r0.b("features");
        for step in 0..depth {
            r0.device.run(|| {
                let mut body = || {
                    for i in 0..count {
                        r0.copy(row(features, i)?, row(norm, i * width + step + 1)?, s0)?;
                    }
                    let previous = if step == 0 {
                        r0.b("anchor_ids")
                    } else {
                        slice_buf(r0.b("greedy_ids"), (step - 1) * count * 4, count * 4)?
                    };
                    // SAFETY: previous IDs and output are live and ordered on the lead stream.
                    unsafe {
                        embedding.gather(
                            previous.ptr,
                            std::ptr::null(),
                            count,
                            1,
                            std::ptr::null(),
                            anchor.ptr,
                            None,
                            crate::shared::peer_split::RankDevice {
                                device: r0.device.id,
                                stream: s0.raw,
                            },
                        )?;
                    }
                    r0.norm(&format!("{c}.hidden_norm.weight"), features, left, count, s0)?;
                    r0.norm(&format!("{c}.embed_norm.weight"), anchor, right, count, s0)?;
                    r0.join_rows(join, left, right, count, s0)?;
                    r0.ffn_whole(
                        &c,
                        "join",
                        "correction_down",
                        "correction_silu",
                        join,
                        delta,
                        count,
                        H,
                        s0,
                    )?;
                    r0.add(residual, features, delta, count, s0)?;
                    Ok(())
                };
                match g0 {
                    Some(graphs) => graphs.draft(rows, step, Section::DraftCorrection, s0, body),
                    None => body(),
                }
            })?;
            let logits = head(slice_buf(residual, 0, count * H * 2)?, count)?;
            ensure!(
                logits.rows == count && logits.vocab == logits.stride && logits.stream == s0.raw,
                "DFly2 shared head returned the wrong batch or stream"
            );
            r0.device.run(|| unsafe {
                let buffer = Buffer {
                    ptr: logits.ptr.cast_mut(),
                    bytes: count * logits.vocab * 4,
                    ..r0.b("greedy_ids")
                };
                r0.device.library.cuda_logits_argmax_checked_f32_async(
                    buffer,
                    slice_buf(r0.b("greedy_ids"), step * count * 4, count * 4)?,
                    slice_buf(r0.b("greedy_scores"), step * count * 4, count * 4)?,
                    count,
                    logits.vocab,
                    s0.raw,
                )
            })?;
        }
        r0.device.run(|| s0.drain())?;
        completed0.complete = true;
        if let Some(mut c1) = completed1.take() {
            self.ranks[1].device.run(|| streams[1].drain())?;
            c1.complete = true;
        }
        let mut ids = vec![0u8; depth * count * 4];
        let bytes = ids.len();
        r0.device.run(|| {
            r0.device
                .library
                .copy_d2h(&mut ids, slice_buf(r0.b("greedy_ids"), 0, bytes)?)
        })?;
        members
            .iter()
            .enumerate()
            .map(|(member, m)| {
                (0..m.depth)
                    .map(|step| {
                        let off = (step * count + member) * 4;
                        let token = u32::from_le_bytes(ids[off..off + 4].try_into().unwrap());
                        ensure!(
                            (token as usize) < 120_832,
                            "DFly2 invalid draft token {token}"
                        );
                        Ok(token)
                    })
                    .collect()
            })
            .collect()
    }
}

#[derive(Clone, Copy, Debug)]
pub(super) struct Member {
    pub slot: usize,
    pub pos: usize,
    pub bonus: u32,
    pub depth: usize,
    /// Text rope offset past the prompt's image spans (0 for text-only prompts):
    /// drafted rows are text rows at rope `pos + query + rope_delta` on every axis.
    pub rope_delta: i64,
}

struct BatchMeta {
    ids: Vec<u32>,
    anchors: Vec<u32>,
    positions: Vec<i64>,
    write_slots: Vec<i64>,
    table: Vec<i32>,
    lengths: Vec<i32>,
    recall: Vec<usize>,
}

use crate::shared::draft::dfly2::words;

impl BatchMeta {
    fn new(members: &[Member], block: usize, mask: u32) -> Result<Self> {
        ensure!(
            (2..=6).contains(&block) && members.len() <= MAX_BATCH,
            "invalid DFly2 batch extent"
        );
        let stride = TOPK + block;
        let rows = members.len() * block;
        let mut out = Self {
            ids: vec![mask; rows],
            anchors: Vec::with_capacity(members.len()),
            positions: Vec::with_capacity(rows),
            write_slots: Vec::with_capacity(rows),
            table: vec![-1; rows * stride],
            lengths: Vec::with_capacity(rows),
            recall: Vec::with_capacity(members.len()),
        };
        for (i, member) in members.iter().enumerate() {
            ensure!(member.pos > 0, "DFly2 requires a committed target row");
            let recall = if member.pos <= TOPK {
                member.pos
            } else {
                TOPK - 4 + (member.pos - 1) % 4 + 1
            };
            let base = i * stride;
            out.ids[i * block] = member.bonus;
            out.anchors.push(member.bonus);
            out.recall.push(recall);
            for query in 0..block {
                let row = i * block + query;
                // [T, H, W] rope coordinates; text rows carry equal axes.
                out.positions.extend([(member.pos + query) as i64 + member.rope_delta; 3]);
                out.write_slots.push((base + TOPK + query) as i64);
                out.lengths.push((recall + query + 1) as i32);
                let dst = &mut out.table[row * stride..(row + 1) * stride];
                for (entry, slot) in dst.iter_mut().take(recall).zip(base..) {
                    *entry = slot as i32;
                }
                for (entry, slot) in dst[recall..].iter_mut().take(query + 1).zip(base + TOPK..) {
                    *entry = slot as i32;
                }
            }
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mixed_context_batch_has_disjoint_causal_latent_regions() -> Result<()> {
        for count in [1, 2, 16, MAX_BATCH] {
            let members = (0..count)
                .map(|i| Member {
                    slot: i,
                    pos: [7, 2048, 2049, 2064][i % 4],
                    bonus: i as u32 + 10,
                    depth: i % 5 + 1,
                    rope_delta: 0,
                })
                .collect::<Vec<_>>();
            let meta = BatchMeta::new(&members, 6, 120009)?;
            let stride = TOPK + 6;
            for (i, m) in members.iter().enumerate() {
                let recall = [7, 2048, 2045, 2048][i % 4];
                assert_eq!(meta.recall[i], recall);
                for offset in 0..6 {
                    let row = i * 6 + offset;
                    let table = &meta.table[row * stride..(row + 1) * stride];
                    let live = recall + offset + 1;
                    assert_eq!(meta.ids[row], if offset == 0 { m.bonus } else { 120009 });
                    assert_eq!(meta.positions[3 * row..3 * row + 3], [(m.pos + offset) as i64; 3]);
                    assert_eq!(meta.lengths[row], live as i32);
                    assert_eq!(table[0], (i * stride) as i32);
                    assert_eq!(table[live - 1] as i64, meta.write_slots[row]);
                    assert!(table[live..].iter().all(|&slot| slot == -1));
                }
            }
        }
        Ok(())
    }

    #[test]
    fn bf16_matrix_slicing_splits_output_rows_and_input_columns() -> Result<()> {
        // 4 rows x 4 cols of distinct u16 elements
        let raw: Vec<u8> = (0u16..16).flat_map(u16::to_le_bytes).collect();
        let out0 = slice_bf16_matrix(&raw, &[4, 4], CoreSplit::Output, 0, 2)?;
        let out1 = slice_bf16_matrix(&raw, &[4, 4], CoreSplit::Output, 1, 2)?;
        assert_eq!(out0, (0u16..8).flat_map(u16::to_le_bytes).collect::<Vec<_>>());
        assert_eq!(out1, (8u16..16).flat_map(u16::to_le_bytes).collect::<Vec<_>>());

        let in0 = slice_bf16_matrix(&raw, &[4, 4], CoreSplit::Input, 0, 2)?;
        let in1 = slice_bf16_matrix(&raw, &[4, 4], CoreSplit::Input, 1, 2)?;
        let exp0: Vec<u8> = [0u16, 1, 4, 5, 8, 9, 12, 13]
            .into_iter()
            .flat_map(u16::to_le_bytes)
            .collect();
        let exp1: Vec<u8> = [2u16, 3, 6, 7, 10, 11, 14, 15]
            .into_iter()
            .flat_map(u16::to_le_bytes)
            .collect();
        assert_eq!(in0, exp0);
        assert_eq!(in1, exp1);
        assert_eq!(
            layer_matrix_suffix("model.dfly_layers.2.self_attn.o_proj.weight"),
            Some("self_attn.o_proj")
        );
        assert_eq!(
            layer_matrix_suffix("model.dfly_layers.mtp_fusion.eh_proj.weight"),
            None
        );
        Ok(())
    }
}
