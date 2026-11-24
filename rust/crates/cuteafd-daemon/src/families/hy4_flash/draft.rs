//! Hy4 DFly2 proposals over the committed target's latent KV and recall.
use super::cache::slice_buf;
use super::native::dequant_kv_b_trellis_absorbed;
use super::graphs::{Hy4Graphs, Section};
use super::weights::{pack_kv_b_absorbed, read_tensor_bytes};
use crate::shared::dense_trellis::{
    codebook, CoreWeightLayout, CoreWeights, Projection, ProjectionLayout, ProjectionScratch,
};
use crate::shared::experts::rtx::dense::DenseLoad;
use crate::shared::memory::device::{Allocation, Device, Stream};
use crate::shared::memory::weight_arena::{DeviceArena, DeviceWeightArena};
use crate::shared::token_io::{DeviceLogits, TokenEmbedding};
use anyhow::{ensure, Context, Result};
use cuteafd_core::memory_layout::{ArenaTag, Tag};
use cuteafd_ffi::dense_trellis::TrellisDType;
use cuteafd_ffi::programs::{Program, Programs, Scalar};
use cuteafd_ffi::{Bf16Add, CuteafdDeviceBuffer as Buffer};
use cuteafd_loader::families::hy4_flash::{
    draft::{Admission, ARENA_ID, ATTENTION_SCRATCH, MAX_BATCH, MLP_WIDTH, PREFIX},
    Hy4DflyConfig, Hy4FlashConfig, Hy4KvCache,
};
use cuteafd_loader::plan::Checkpoint;
use std::{cell::Cell, collections::BTreeMap};

pub(super) const H: usize = 3072;
const TOPK: usize = 2048;
const KV_WIDTH: usize = 576;
const HEADS: usize = 32;

pub(super) struct Drafter<'a> {
    pub cfg: Hy4DflyConfig,
    format: Hy4KvCache,
    device: Device<'a>,
    weights: BTreeMap<String, Allocation<'a>>,
    workspace: BTreeMap<String, Allocation<'a>>,
    linear: BTreeMap<String, (CoreWeights<'a>, Projection<'a>)>,
    trellis: Option<ProjectionScratch<'a>>,
    programs: BTreeMap<&'static str, Program<'a>>,
    add: Bf16Add<'a>,
    frontiers: Vec<Cell<Option<usize>>>,
    verify_rows: usize,
    batch_capacity: usize,
    has_peer_recall: bool,
}

impl<'a> Drafter<'a> {
    pub fn new(
        device: Device<'a>,
        programs: &'a Programs<'a>,
        checkpoint: &Checkpoint,
        text: &Hy4FlashConfig,
        cfg: Hy4DflyConfig,
        admission: Admission,
        format: Hy4KvCache,
        slots: usize,
        verify_rows: usize,
    ) -> Result<Self> {
        let extent = cfg.block_rows() * slots;
        let layout = admission.weights;
        let arena = DeviceWeightArena::new(
            device,
            layout.arena.clone(),
            ArenaTag::drafter_weights(ARENA_ID),
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
                    0,
                    1,
                    HEADS,
                    text.qk_nope_dim,
                    text.v_head_dim,
                    text.kv_lora_rank,
                    dst.buffer,
                )?;
            } else {
                let (raw, _, _) = read_tensor_bytes(checkpoint, name)?;
                let bytes = if name.ends_with(".kv_b_proj.weight") {
                    pack_kv_b_absorbed(
                        &raw,
                        HEADS,
                        text.qk_nope_dim,
                        text.v_head_dim,
                        text.kv_lora_rank,
                    )?
                } else if name.ends_with(".learnable_sink_param") {
                    raw.chunks_exact(2)
                        .flat_map(|v| {
                            ((u16::from_le_bytes([v[0], v[1]]) as u32) << 16).to_le_bytes()
                        })
                        .collect()
                } else {
                    raw
                };
                ensure!(bytes.len() == view.bytes(), "{name}: DFly2 upload extent differs from admission");
                device.run(|| device.library.copy_h2d(dst.buffer, &bytes))?;
            }
            weights.insert(name.clone(), dst);
        }
        let mut linear = BTreeMap::new();
        let dense = DenseLoad::mount(device.library);
        for (stem, admitted) in &layout.native {
            let core = CoreWeightLayout::from_shared(admitted.clone());
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
                    1,
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
                .allocate_packed(device, Tag::drafter_workspace("hy4_flash/dfly2/trellis"))
            })
            .transpose()?;

        let mut bound = BTreeMap::new();
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
            ("join", "draft_proj_join", proj),
            ("correction_down", "draft_proj_correction_down", proj),
            ("silu", "draft_silu", silu),
            ("correction_silu", "draft_correction_silu", silu),
            (
                "fusion",
                "draft_gated_residual",
                &["residual", "delta", "gate", "out"][..],
            ),
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
            bound.insert(key, programs.program(&format!("hy4f_{suffix}"), args)?);
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
                &format!("hy4f_mla_rank_norm_pack_{f}"),
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
                &format!("hy4f_draft_attention_{f}"),
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
            attn <= ATTENTION_SCRATCH * MAX_BATCH,
            "DFly2 attention scratch exceeds admission"
        );
        let workspace = admission.workspace;
        let run = DeviceArena::new(
            device,
            workspace,
            ArenaTag::drafter_workspace("hy4_flash/dfly2/workspace"),
        )?;
        let views = run
            .layout()
            .views()
            .iter()
            .map(|(name, v)| Ok((name.clone(), run.view(name, v.bytes())?)))
            .collect::<Result<BTreeMap<_, _>>>()?;
        let has_peer_recall = views.contains_key("tap_recall_peer");
        Ok(Self {
            cfg,
            format,
            device,
            weights,
            workspace: views,
            linear,
            trellis,
            programs: bound,
            add: device.run(|| device.library.bf16_add())?,
            frontiers: (0..slots).map(|_| Cell::new(None)).collect(),
            verify_rows,
            batch_capacity: slots,
            has_peer_recall,
        })
    }

    fn w(&self, name: &str) -> Result<Buffer> {
        Ok(self
            .weights
            .get(name)
            .with_context(|| format!("missing DFly2 tensor {name}"))?
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
        // SAFETY: all caller operands are bounded live views of the admitted
        // rank-0 target or DFly2 arenas and use the same ordered stream.
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
        // SAFETY: disjoint bounded views ordered on the single drafter stream.
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
        // SAFETY: all views are disjoint, live and ordered on rank 0.
        unsafe { self.add.launch(a, b, dst, rows * H, stream.raw) }
    }

    pub(super) fn inventory(
        &self, weights: &mut crate::shared::memory::receipts::Inventory<'_>,
        retained: &mut crate::shared::memory::receipts::Inventory<'_>,
    ) -> Result<()> {
        for (name, allocation) in &self.weights {
            allocation.inventory(weights, name)?;
        }
        for (stem, (allocation, _)) in &self.linear {
            allocation.payload.inventory(weights, &format!("{stem}.trellis"))?;
            allocation.suh.inventory(weights, &format!("{stem}.suh"))?;
            allocation.svh.inventory(weights, &format!("{stem}.svh"))?;
        }
        for (name, allocation) in &self.workspace {
            allocation.inventory(retained, name)?;
        }
        if let Some(scratch) = &self.trellis {
            scratch.allocation_receipts(retained, "hy4_flash/dfly2/trellis")?;
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

    fn recall(&self, verify: bool, row: usize, stage: usize) -> Result<Buffer> {
        slice_buf(
            self.b(if verify {
                "verify_recall"
            } else {
                "tap_recall"
            }),
            (row * self.cfg.layers + stage) * TOPK * 4,
            TOPK * 4,
        )
    }
    fn recall_peer(&self, verify: bool, row: usize, stage: usize) -> Result<Buffer> {
        ensure!(self.has_peer_recall, "DFly2 peer recall buffer not admitted");
        slice_buf(
            self.b(if verify {
                "verify_recall_peer"
            } else {
                "tap_recall_peer"
            }),
            (row * self.cfg.layers + stage) * TOPK * 4,
            TOPK * 4,
        )
    }
    pub fn captures_layer(&self, target_layer: usize) -> bool {
        self.cfg.target_index_layers.contains(&target_layer)
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
    pub fn capture_recall(
        &self,
        verify: bool,
        row: usize,
        target_layer: usize,
        src: Buffer,
        stream: &Stream<'a>,
    ) -> Result<()> {
        if let Some(stage) = self
            .cfg
            .target_index_layers
            .iter()
            .position(|&l| l == target_layer)
        {
            self.copy(self.recall(verify, row, stage)?, src, stream)?;
        }
        Ok(())
    }
    pub fn capture_recall_peer(
        &self,
        verify: bool,
        row: usize,
        target_layer: usize,
        src_on_peer: Buffer,
        stream: &Stream<'a>,
    ) -> Result<()> {
        if let Some(stage) = self
            .cfg
            .target_index_layers
            .iter()
            .position(|&l| l == target_layer)
        {
            self.copy(self.recall_peer(verify, row, stage)?, src_on_peer, stream)?;
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
        self.copy(self.hidden(verify, row)?, src, stream)
    }
    pub fn accept(&self, slot: usize, row: usize, pos: usize, stream: &Stream<'a>) -> Result<()> {
        ensure!(
            row < self.verify_rows,
            "DFly2 verify tap row outside admission"
        );
        self.copy(self.hidden(false, slot)?, self.hidden(true, row)?, stream)?;
        for stage in 0..self.cfg.layers {
            self.copy(
                self.recall(false, slot, stage)?,
                self.recall(true, row, stage)?,
                stream,
            )?;
            if self.has_peer_recall {
                self.copy(
                    self.recall_peer(false, slot, stage)?,
                    self.recall_peer(true, row, stage)?,
                    stream,
                )?;
            }
        }
        self.publish(slot, pos);
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub fn propose(
        &self,
        stream: &Stream<'a>,
        embedding: &TokenEmbedding<'a>,
        members: &[Member],
        graphs: Option<&Hy4Graphs<'a>>,
        rope: Buffer,
        cache: impl Fn(usize) -> Result<Buffer>,
        cache_peer: Option<&dyn Fn(usize) -> Result<Buffer>>,
        mut head: impl FnMut(Buffer, usize) -> Result<DeviceLogits>,
    ) -> Result<Vec<Vec<u32>>> {
        if members.is_empty() {
            return Ok(Vec::new());
        }
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
        self.device.run(|| {
            let mut completed = stream.cancellation_guard("Hy4 DFly2 proposal batch");
            let run = |section, stage, work: &mut dyn FnMut() -> Result<()>| {
                match graphs {
                    Some(graphs) => graphs.draft(rows, stage, section, stream, work),
                    None => work(),
                }
            };
            let row = |b: Buffer, i: usize| slice_buf(b, i * H * 2, H * 2);
            let state = self.b("state");
            let residual = self.b("residual");
            let norm = self.b("norm");
            let delta = self.b("delta");
            let join = self.b("joined");
            let left = self.b("join_left");
            let right = self.b("join_right");
            let anchor = self.b("anchor");
            let features = self.b("features");
            embedding.embed(&meta.ids, self.b("ids"), 1, state, stream.raw)?;
            for (i, member) in members.iter().enumerate() {
                self.copy(row(anchor, i)?, row(state, i * width)?, stream)?;
                self.copy(row(features, i)?, self.hidden(false, member.slot)?, stream)?;
            }
            run(Section::DraftFusion, 0, &mut || {
                let f = format!("{PREFIX}mtp_fusion");
                self.norm(&format!("{f}.enorm.weight"), anchor, left, count, stream)?;
                self.norm(&format!("{f}.hnorm.weight"), features, right, count, stream)?;
                self.join_rows(join, left, right, count, stream)?;
                self.proj("join", &format!("{f}.eh_proj"), join, delta, count, stream)?;
                self.rows(
                    "fusion",
                    &[anchor, delta, self.w(&format!("{f}.gate"))?, residual],
                    count,
                    stream,
                )?;
                for i in 0..count {
                    self.copy(row(state, i * width)?, row(residual, i)?, stream)?;
                }
                Ok(())
            })?;

            for (name, data) in [
                ("positions", words(&meta.positions)),
                ("write_slots", words(&meta.write_slots)),
                ("page_table", words(&meta.table)),
                ("lengths", words(&meta.lengths)),
                ("cu", words(&(0..=rows as i32).collect::<Vec<_>>())),
                ("anchor_ids", words(&meta.anchors)),
            ] {
                self.upload(self.b(name), &data, stream)?;
            }
            let kv = self.b("kv");
            ensure!(
                kv.bytes >= count * pt_width * record,
                "DFly2 KV staging is short"
            );
            for (stage, &target) in self.cfg.target_kv_layers.iter().enumerate() {
                let p = format!("{PREFIX}{stage}");
                let a = format!("{p}.self_attn");
                let layer_cache = cache(target)?;
                let layer_cache_peer = cache_peer.map(|f| f(target)).transpose()?;
                for (i, member) in members.iter().enumerate() {
                    let staging = slice_buf(kv, i * pt_width * record, pt_width * record)?;
                    self.launch(
                        "gather",
                        &[
                            layer_cache,
                            self.recall(false, member.slot, stage)?,
                            staging,
                        ],
                        &[
                            Scalar::I32(meta.recall[i] as i32),
                            Scalar::I32(1),
                            Scalar::I32(0),
                        ],
                        stream,
                    )?;
                    if let Some(peer_pool) = layer_cache_peer {
                        self.launch(
                            "gather",
                            &[
                                peer_pool,
                                self.recall_peer(false, member.slot, stage)?,
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
                run(Section::DraftLayer, stage, &mut || {
                    self.norm(
                        &format!("{p}.input_layernorm.weight"),
                        state,
                        norm,
                        rows,
                        stream,
                    )?;
                    self.proj(
                        "q_a",
                        &format!("{a}.q_a_proj"),
                        norm,
                        self.b("q_a"),
                        rows,
                        stream,
                    )?;
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
                    let uk_bytes = HEADS * 192 * 512 * 2;
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
                    )?;
                    self.proj(
                        "o",
                        &format!("{a}.o_proj"),
                        self.b("uv"),
                        delta,
                        rows,
                        stream,
                    )?;
                    self.norm(
                        &format!("{a}.post_proj_norm.weight"),
                        delta,
                        norm,
                        rows,
                        stream,
                    )?;
                    self.add(residual, state, norm, rows, stream)?;
                    self.norm(
                        &format!("{p}.post_attention_layernorm.weight"),
                        residual,
                        norm,
                        rows,
                        stream,
                    )?;
                    self.ffn(
                        &format!("{p}.mlp"),
                        "gate_up",
                        "down",
                        "silu",
                        norm,
                        delta,
                        rows,
                        MLP_WIDTH,
                        stream,
                    )?;
                    self.add(state, residual, delta, rows, stream)?;
                    if stage + 1 == self.cfg.layers {
                        self.norm(
                            &format!("{PREFIX}final_norm.weight"),
                            state,
                            norm,
                            rows,
                            stream,
                        )?;
                    }
                    Ok(())
                })?;
            }
            let c = format!("{PREFIX}hidden_correction");
            for step in 0..depth {
                run(Section::DraftCorrection, step, &mut || {
                    for i in 0..count {
                        self.copy(row(features, i)?, row(norm, i * width + step + 1)?, stream)?;
                    }
                    let previous = if step == 0 {
                        self.b("anchor_ids")
                    } else {
                        slice_buf(self.b("greedy_ids"), (step - 1) * count * 4, count * 4)?
                    };
                    // SAFETY: previous IDs and output are live and ordered on the proposal stream.
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
                                device: self.device.id,
                                stream: stream.raw,
                            },
                        )?;
                    }
                    self.norm(
                        &format!("{c}.hidden_norm.weight"),
                        features,
                        left,
                        count,
                        stream,
                    )?;
                    self.norm(
                        &format!("{c}.embed_norm.weight"),
                        anchor,
                        right,
                        count,
                        stream,
                    )?;
                    self.join_rows(join, left, right, count, stream)?;
                    self.ffn(
                        &c,
                        "join",
                        "correction_down",
                        "correction_silu",
                        join,
                        delta,
                        count,
                        H,
                        stream,
                    )?;
                    self.add(residual, features, delta, count, stream)?;
                    Ok(())
                })?;
                let logits = head(slice_buf(residual, 0, count * H * 2)?, count)?;
                ensure!(
                    logits.rows == count
                        && logits.vocab == logits.stride
                        && logits.stream == stream.raw,
                    "DFly2 shared head returned the wrong batch or stream"
                );
                // SAFETY: F32 head rows and output spans are on rank 0, ordered and live.
                unsafe {
                    let buffer = Buffer {
                        ptr: logits.ptr.cast_mut(),
                        bytes: count * logits.vocab * 4,
                        ..self.b("greedy_ids")
                    };
                    self.device.library.cuda_logits_argmax_checked_f32_async(
                        buffer,
                        slice_buf(self.b("greedy_ids"), step * count * 4, count * 4)?,
                        slice_buf(self.b("greedy_scores"), step * count * 4, count * 4)?,
                        count,
                        logits.vocab,
                        stream.raw,
                    )?;
                }
            }
            stream.drain()?;
            completed.complete = true;
            let mut ids = vec![0u8; depth * count * 4];
            let bytes = ids.len();
            self.device
                .library
                .copy_d2h(&mut ids, slice_buf(self.b("greedy_ids"), 0, bytes)?)?;
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
        })
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
    fn ffn(
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
}

#[derive(Clone, Copy, Debug)]
pub(super) struct Member {
    pub slot: usize,
    pub pos: usize,
    pub bonus: u32,
    pub depth: usize,
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

fn words<T, const N: usize>(input: &[T]) -> Vec<u8>
where
    T: LeBytes<N>,
{
    input.iter().flat_map(LeBytes::le_bytes).collect()
}
trait LeBytes<const N: usize> {
    fn le_bytes(&self) -> [u8; N];
}
impl LeBytes<4> for i32 {
    fn le_bytes(&self) -> [u8; 4] {
        self.to_le_bytes()
    }
}
impl LeBytes<4> for u32 {
    fn le_bytes(&self) -> [u8; 4] {
        self.to_le_bytes()
    }
}
impl LeBytes<8> for i64 {
    fn le_bytes(&self) -> [u8; 8] {
        self.to_le_bytes()
    }
}

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
                out.positions.push((member.pos + query) as i64);
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
                    assert_eq!(meta.positions[row], (m.pos + offset) as i64);
                    assert_eq!(meta.lengths[row], live as i32);
                    assert_eq!(table[0], (i * stride) as i32);
                    assert_eq!(table[live - 1] as i64, meta.write_slots[row]);
                    assert!(table[live..].iter().all(|&slot| slot == -1));
                }
            }
        }
        Ok(())
    }
}
