//! Header-admitted packed target weight arena for Hy4 (`hy_v4_vl`).
//!
//! One `DeviceWeightArena` per physical coordinator GPU holds all target weights,
//! globals (`embed_tokens`, `hc_head`, `norm`, `lm_head`), and `ROPE_INV_FREQ`.
use crate::families::hy4_common::native::{
    dequant_kv_b_trellis_absorbed_with, Hy4NativeCore, HY4_NATIVE_TAGS,
};
use crate::families::hy4_common::weights::{dense_rect, inv_freq_bytes, kv_b_absorbed_ptrs};
use crate::shared::experts::rtx::dense::{DenseLoad, Job, Operand, Step};
use crate::shared::memory::{
    budget::admit_host_memory,
    device::{Allocation, Device},
    receipts::Inventory,
    weight_arena::DeviceWeightArena,
    Tag,
};
use crate::shared::target_head::TargetHead;
use crate::shared::token_io::{EmbedPlacement, EmbedSource, TokenEmbedding};
use anyhow::{ensure, Context, Result};
use cuteafd_core::memory_layout::ArenaTag;
use cuteafd_core::DType;
use cuteafd_ffi::programs::{Programs, VocabularyHead};
use cuteafd_ffi::CuteafdDeviceBuffer;
use cuteafd_loader::families::hy4::weight_layout::{
    fp8_role, target_arenas, target_mount_load, CoordinatorWeights, ROPE_INV_FREQ,
};
use cuteafd_loader::families::hy4::Hy4Config;
use cuteafd_loader::families::hy4_common::core_split;
use cuteafd_loader::placement::families::hy4 as hy4_placement;
use cuteafd_loader::placement::AttentionPlacement;
use cuteafd_loader::plan::checkpoint::Checkpoint;
use cuteafd_loader::plan::families::hy4 as hy4_plan;
use std::collections::BTreeMap;

const HEAD_WS_BYTES: usize = cuteafd_ffi::programs::VOCABULARY_HEAD_WORKSPACE;

/// One coordinator rank's packed target weights, RoPE table, embedding, and vocabulary head.
pub(crate) struct Hy4RankWeights<'a> {
    pub(crate) device: Device<'a>,
    pub(crate) rank: usize,
    pub(crate) gpus: usize,
    /// Per-layer tensor shard rank/world (1/0 for TP1 or 2-GPU `layers`; rank/2 for `heads`/`context`).
    pub(crate) layer_split_rank: usize,
    pub(crate) layer_split_ranks: usize,
    pub(crate) layout: CoordinatorWeights,
    /// Single packed backing for all target weights & globals on this GPU.
    pub(crate) arena: DeviceWeightArena<'a>,
    /// Named BF16/F32 views inside `arena` (zero separate per-tensor `cudaMalloc`).
    pub(crate) views: BTreeMap<String, Allocation<'a>>,
    /// Compact-core native linears (`k8-compact`, `k6-compact`, `fp8-compact`) backed by `arena`.
    pub(crate) native: Hy4NativeCore<'a>,
    pub(crate) embedding: Option<TokenEmbedding<'a>>,
    pub(crate) inv_freq: Allocation<'a>,
    pub(crate) head_ws: Option<Allocation<'a>>,
    pub(crate) head: Option<VocabularyHead<'a>>,
    pub(crate) local_vocab: usize,
}

impl<'a> Hy4RankWeights<'a> {
    /// Allocate each rank's packed target weight arena and record (or mount) all
    /// non-dequantized target tensors into the active dense loader.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn record_all(
        devices: &[Device<'a>],
        programs: &'a Programs<'a>,
        checkpoint: &Checkpoint,
        cfg: &Hy4Config,
        attention: AttentionPlacement,
        max_rows: usize,
        max_head_rows: usize,
    ) -> Result<Vec<Self>> {
        let gpus = devices.len();
        ensure!(matches!(gpus, 1 | 2), "Hy4 supports 1 or 2 coordinator GPUs, not {gpus}");
        let model = hy4_plan::Model::open(checkpoint)?;
        let modes = hy4_placement::layer_modes(cfg, gpus, attention);
        let layouts = target_arenas(checkpoint, cfg, &model.core, &modes, gpus)?;
        ensure!(layouts.len() == gpus, "Hy4 target_arenas rank count mismatch");
        let whole_layers = gpus == 1 || attention == AttentionPlacement::Layers;

        let library = devices[0].library;
        let dense = DenseLoad::mount(library);
        let mount = target_mount_load(checkpoint, gpus)?;
        let mount_owners = [
            ("hy4/dense-mount-source", 2 * mount.pinned_source),
            ("hy4/dense-mount-rows", 2 * mount.full_row_cache),
            ("hy4/dense-mount-tables", 2 * mount.pinned_tables),
        ];
        admit_host_memory(0, &mount_owners, 0, 0)?;
        dense.admit_mount(mount)?;

        let mut ranks = Vec::with_capacity(gpus);
        for (rank, layout) in layouts.into_iter().enumerate() {
            let device = devices[rank];
            let layer_split_ranks = if whole_layers { 1 } else { gpus };
            let layer_split_rank = if whole_layers { 0 } else { rank };
            let arena = DeviceWeightArena::new(
                device,
                layout.arena.clone(),
                ArenaTag::weights("hy4/target"),
            )?;

            let native = Hy4NativeCore::load_with(
                device,
                programs,
                checkpoint,
                &layout.core,
                &layout.fp8,
                fp8_role,
                HY4_NATIVE_TAGS,
                max_rows,
                &arena,
            )?;
            let claimed_by_native = |name: &str| -> bool {
                for suffix in [".trellis", ".suh", ".svh"] {
                    if let Some(stem) = name.strip_suffix(suffix) {
                        if layout.core.contains_key(stem) {
                            return true;
                        }
                    }
                }
                for suffix in [".payload", ".scales"] {
                    if let Some(stem) = name.strip_suffix(suffix) {
                        if layout.fp8.contains_key(stem) {
                            return true;
                        }
                    }
                }
                false
            };

            let mut views = BTreeMap::new();
            let mut embedding = None;
            let mut inv_freq = None;

            for (name, view_meta) in layout.arena.views() {
                if claimed_by_native(name) {
                    continue;
                }
                if name == ROPE_INV_FREQ {
                    let bytes = inv_freq_bytes(cfg.qk_rope_dim, cfg.rope_theta);
                    let alloc = arena.view(name, bytes.len())?;
                    device.run(|| device.library.copy_h2d(alloc.buffer, &bytes))?;
                    inv_freq = Some(alloc);
                    continue;
                }
                if name == "model.embed_tokens.weight" {
                    let entry = checkpoint
                        .tensors
                        .iter()
                        .find(|t| t.meta.name == *name)
                        .context("Hy4 missing model.embed_tokens.weight")?;
                    let source = EmbedSource::new(
                        &checkpoint.snapshot,
                        &entry.shard,
                        &entry.meta,
                        cfg.hidden,
                    )?;
                    let (emb, ()) = device.run(|| {
                        TokenEmbedding::load_with(
                            device.library,
                            source,
                            EmbedPlacement::Gpu,
                            || Ok(()),
                            |_, bytes| arena.view("model.embed_tokens.weight", bytes),
                        )
                    })?;
                    embedding = Some(emb);
                    continue;
                }

                let alloc = arena.view(name, view_meta.bytes())?;
                let stem = name.strip_suffix(".weight").unwrap_or(name);
                if stem.ends_with(".self_attn.kv_b_proj") {
                    if !model.core.contains_key(stem) {
                        let header = checkpoint
                            .header(name)
                            .context("Hy4 BF16 KV-B header missing")?;
                        ensure!(
                            header.dtype == DType::Bf16 && header.shape.len() == 2,
                            "bad BF16 KV-B shape for {name}"
                        );
                        let nope = cfg.qk_nope_dim;
                        let v = cfg.v_head_dim;
                        let lat = cfg.kv_lora_rank;
                        let rect = dense_rect(
                            checkpoint,
                            name,
                            layer_split_rank,
                            layer_split_ranks,
                            core_split,
                        )?;
                        let heads = rect.rows.len() / (nope + v);
                        let uk_bytes = heads * lat * nope * 2;
                        let uv_bytes = heads * v * lat * 2;
                        ensure!(
                            rect.rows.len() == heads * (nope + v)
                                && rect.width() == lat * 2
                                && rect.bytes() == uk_bytes + uv_bytes
                                && alloc.buffer.bytes == uk_bytes + uv_bytes,
                            "Hy4 KV-B source or destination extent differs for {name}"
                        );
                        dense.step(
                            Step {
                                device: device.id,
                                rects: vec![rect],
                                jobs: vec![Job::Absorb {
                                    heads,
                                    nope,
                                    v,
                                    lat,
                                    uk: CuteafdDeviceBuffer {
                                        bytes: uk_bytes,
                                        ..alloc.buffer
                                    },
                                    uv: CuteafdDeviceBuffer {
                                        ptr: alloc.buffer.ptr.wrapping_byte_add(uk_bytes),
                                        bytes: uv_bytes,
                                        ..alloc.buffer
                                    },
                                }],
                            },
                            std::ptr::null_mut(),
                        )?;
                    }
                } else {
                    let rect = dense_rect(
                        checkpoint,
                        name,
                        layer_split_rank,
                        layer_split_ranks,
                        core_split,
                    )?;
                    let bytes = rect.bytes();
                    ensure!(
                        bytes == alloc.buffer.bytes,
                        "Hy4 {name}: source {bytes} and arena {} sizes differ",
                        alloc.buffer.bytes
                    );
                    dense.operand(Operand {
                        device: device.id,
                        buffer: alloc.buffer,
                        lands: vec![(rect, 0)],
                        bytes,
                    })?;
                }
                views.insert(name.clone(), alloc);
            }

            let inv_freq = inv_freq.context("Hy4 target arena missing inv_freq view")?;
            let local_vocab = cfg.vocab_size / layer_split_ranks;
            let (head_ws, head) = if views.contains_key("lm_head.weight") {
                let ws = Allocation::new(
                    device,
                    HEAD_WS_BYTES,
                    Tag::workspace("hy4/head-cublas-ws"),
                )?;
                let h = device.run(|| {
                    // SAFETY: `ws` stays alive in `Hy4RankWeights` for the full lifetime of `head`.
                    unsafe {
                        device.library.vocabulary_head_rows(
                            ws.buffer.ptr,
                            cfg.hidden as u32,
                            max_head_rows.min(64) as u32,
                            local_vocab as u32,
                            TargetHead::configured().native_mode(),
                        )
                    }
                })?;
                (Some(ws), Some(h))
            } else {
                (None, None)
            };

            ranks.push(Self {
                device,
                rank,
                gpus,
                layer_split_rank,
                layer_split_ranks,
                layout,
                arena,
                views,
                native,
                embedding,
                inv_freq,
                head_ws,
                head,
                local_vocab,
            });
        }
        dense.release();
        Ok(ranks)
    }

    /// Convert compact EXL3 `kv_b_proj` projections serially into their absorbed BF16 views.
    pub(crate) fn finish_kv_b(
        ranks: &[Self],
        programs: &'a Programs<'a>,
        checkpoint: &Checkpoint,
        cfg: &Hy4Config,
    ) -> Result<()> {
        let model = hy4_plan::Model::open(checkpoint)?;
        for r in ranks {
            for (name, dst) in &r.views {
                let Some(stem) = name
                    .strip_suffix(".weight")
                    .filter(|stem| stem.ends_with(".self_attn.kv_b_proj"))
                else {
                    continue;
                };
                let Some(proj) = model.core.get(stem) else {
                    continue;
                };
                let local_heads =
                    proj.shape[0] / r.layer_split_ranks / (cfg.qk_nope_dim + cfg.v_head_dim);
                dequant_kv_b_trellis_absorbed_with(
                    r.device,
                    programs,
                    checkpoint,
                    proj,
                    r.layer_split_rank,
                    r.layer_split_ranks,
                    local_heads,
                    cfg.qk_nope_dim,
                    cfg.v_head_dim,
                    cfg.kv_lora_rank,
                    HY4_NATIVE_TAGS,
                    dst.buffer,
                )?;
            }
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn load_all(
        devices: &[Device<'a>],
        programs: &'a Programs<'a>,
        checkpoint: &Checkpoint,
        cfg: &Hy4Config,
        attention: AttentionPlacement,
        max_rows: usize,
        max_head_rows: usize,
        media: crate::shared::vision::push::Contribution,
    ) -> Result<Vec<Self>> {
        let ranks = crate::shared::experts::rtx::push::load_alone_with_media(
            &checkpoint.snapshot,
            devices,
            media,
            || Self::record_all(devices, programs, checkpoint, cfg, attention, max_rows, max_head_rows),
        )?;
        Self::finish_kv_b(&ranks, programs, checkpoint, cfg)?;
        Ok(ranks)
    }

    pub(crate) fn buf(&self, name: &str) -> Result<CuteafdDeviceBuffer> {
        self.views
            .get(name)
            .map(|a| a.buffer)
            .with_context(|| format!("Hy4 rank {} missing weight view {name}", self.rank))
    }

    pub(crate) fn weight(&self, stem: &str) -> Result<CuteafdDeviceBuffer> {
        self.buf(&format!("{stem}.weight"))
    }

    /// Return `(w_uk [local_heads, 512, 192], w_uv [local_heads, 256, 512])` pointers inside `kv_b_proj`.
    pub(crate) fn kv_b_absorbed(
        &self,
        layer: usize,
        local_heads: usize,
    ) -> Result<(*mut std::ffi::c_void, *mut std::ffi::c_void)> {
        let buf = self.weight(&format!("model.layers.{layer}.self_attn.kv_b_proj"))?;
        kv_b_absorbed_ptrs(buf, local_heads, 192, 256, 512)
    }

    /// Record the packed target weight arena (one physical backing + all named views)
    /// and optional BF16 head workspace into `inventory`.
    pub(crate) fn inventory(
        &self,
        weights_inv: &mut Inventory<'_>,
        retained_inv: &mut Inventory<'_>,
    ) -> Result<()> {
        let rank = self.rank;
        for (name, alloc) in &self.views {
            alloc.inventory(weights_inv, &format!("rank{rank}.{name}"))?;
        }
        self.inv_freq
            .inventory(weights_inv, &format!("rank{rank}.{ROPE_INV_FREQ}"))?;
        if let Some(emb) = &self.embedding {
            emb.inventory(
                weights_inv,
                &format!("rank{rank}.embed_tokens"),
                weights_inv.lifetime(),
            )?;
        }
        self.native.inventory(weights_inv, rank)?;
        if let Some(ws) = &self.head_ws {
            ws.inventory(retained_inv, &format!("hy4.head.{rank}.workspace"))?;
        }
        Ok(())
    }
}
