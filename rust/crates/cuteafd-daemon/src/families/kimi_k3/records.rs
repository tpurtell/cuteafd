//! The Kimi coordinator's dense operands as shared push records: allocated
//! exactly as the host packer allocates them (the same `LayerLayout` and
//! `GlobalLayout` extents), filled either through the mount at once or by the
//! coordinator's push ranks. Every transform runs on the GPU:
//! - class 2: BF16/F32 row and column windows land unchanged (norms, router,
//!   KDA input/forget/conv/dt_bias/norm/output, MLA projections and norms,
//!   FFN halves, the embedding and the head);
//! - class 3: kv_b absorbed per head (`Job::Absorb`), A_log's padding checked
//!   before its active heads upload (`Job::ALog`), residual scores multiplied
//!   in F32 (`Job::Product`) and K6 trellis tiles reordered into the native
//!   shard (`Job::TrellisTiles`). Each K6 codebook marker is read and checked
//!   while planning, before any record exists.
use super::*;
use crate::shared::dense_trellis::{CoreWeightLayout, CoreWeights};
use crate::shared::experts::rtx::dense::{DenseLoad, Job, Operand, Rect, Step, STEP_BYTES};
use cuteafd_loader::{plan::Checkpoint, formats::exl3_storage::exl3_module};

/// One coordinator GPU's records target.
pub(crate) struct KimiRecords<'l, 'a> {
    pub checkpoint: &'l Checkpoint,
    pub cfg: &'l KimiK3Config,
    pub core: &'l CoreInventory,
    pub dense: &'l DenseLoad<'a>,
    /// (device, load stream) of each coordinator GPU.
    pub streams: [(i32, *mut std::ffi::c_void); 2],
}

/// Rows `rows` and byte columns `cols` of a checkpoint tensor (rows of the
/// first dimension; 1-D and 0-D tensors are one row).
fn rect(checkpoint: &Checkpoint, name: &str, rows: Option<std::ops::Range<usize>>,
    cols: Option<std::ops::Range<usize>>) -> Result<Rect> {
    let at = checkpoint.tensors.binary_search_by(|t| t.meta.name.as_str().cmp(name))
        .map_err(|_| anyhow::anyhow!("Kimi coordinator checkpoint lacks {name}"))?;
    let tensor = &checkpoint.tensors[at];
    ensure!(checkpoint.weight_map.get(name) == Some(&tensor.shard), "Kimi tensor header differs from checkpoint index: {name}");
    let total = if tensor.meta.shape.len() >= 2 { tensor.meta.shape[0] } else { 1 };
    let row_bytes = usize::try_from(tensor.meta.byte_length)? / total;
    let (rows, cols) = (rows.unwrap_or(0..total), cols.unwrap_or(0..row_bytes));
    ensure!(rows.end <= total && cols.end <= row_bytes && !rows.is_empty() && !cols.is_empty(),
        "{name}: rectangle outside the tensor");
    Ok(Rect { name: name.into(), path: checkpoint.snapshot.join(&tensor.shard), base: tensor.meta.byte_offset,
        row_bytes, rows, cols })
}

fn meta<'c>(checkpoint: &'c Checkpoint, name: &str) -> Result<&'c cuteafd_loader::SafetensorsTensorMetadata> {
    let at = checkpoint.tensors.binary_search_by(|t| t.meta.name.as_str().cmp(name))
        .map_err(|_| anyhow::anyhow!("Kimi coordinator checkpoint lacks {name}"))?;
    Ok(&checkpoint.tensors[at].meta)
}

fn expect(checkpoint: &Checkpoint, name: &str, dtype: DType, shape: &[usize]) -> Result<()> {
    let tensor = meta(checkpoint, name)?;
    ensure!(tensor.dtype == dtype && tensor.shape == shape, "Kimi tensor {name} expected {dtype:?} {shape:?}");
    Ok(())
}

/// A view `bytes` long at `offset` into `buffer`.
fn view(buffer: CuteafdDeviceBuffer, offset: usize, bytes: usize) -> Result<CuteafdDeviceBuffer> {
    ensure!(offset + bytes <= buffer.bytes, "Kimi record view outside its operand");
    // SAFETY: `offset + bytes` lies inside the live allocation (checked above).
    Ok(CuteafdDeviceBuffer { ptr: unsafe { buffer.ptr.cast::<u8>().add(offset) }.cast(), bytes, ..buffer })
}

impl<'a> KimiRecords<'_, 'a> {
    fn allocate(&self, device: Device<'a>, bytes: usize) -> Result<Allocation<'a>> {
        ensure!(bytes > 0, "Kimi operand is empty");
        let _scope = cuteafd_ffi::memory_ledger::scope("weights/kimi coordinator weights");
        Allocation::new(device, bytes)
    }

    /// An allocation of exactly `bytes` whose `lands` fill it unchanged.
    fn landed(&self, device: Device<'a>, lands: Vec<(Rect, usize)>, bytes: usize) -> Result<Allocation<'a>> {
        ensure!(lands.iter().map(|(r, _)| r.bytes()).sum::<usize>() == bytes, "Kimi landed operand coverage");
        let allocation = self.allocate(device, bytes)?;
        self.dense.operand(Operand { device: device.id, buffer: allocation.buffer, lands, bytes })?;
        Ok(allocation)
    }

    fn step(&self, device: Device<'a>, rects: Vec<Rect>, jobs: Vec<Job>) -> Result<()> {
        let stream = self.streams.iter().find(|(id, _)| *id == device.id).context("Kimi load stream for device")?.1;
        self.dense.step(Step { device: device.id, rects, jobs }, stream)
    }

    /// A whole BF16/F32 tensor, landed unchanged.
    fn whole(&self, device: Device<'a>, name: &str, dtype: DType, shape: &[usize]) -> Result<Allocation<'a>> {
        expect(self.checkpoint, name, dtype, shape)?;
        let r = rect(self.checkpoint, name, None, None)?;
        let bytes = r.bytes();
        self.landed(device, vec![(r, 0)], bytes)
    }

    /// The F32 residual score `norm * proj` (`w8::residual_score_weight`),
    /// multiplied on the GPU from both staged BF16 vectors.
    fn score(&self, device: Device<'a>, norm: &str, proj: &str) -> Result<Allocation<'a>> {
        let hidden = self.cfg.hidden;
        expect(self.checkpoint, norm, DType::Bf16, &[hidden])?;
        expect(self.checkpoint, proj, DType::Bf16, &[1, hidden])?;
        let out = self.allocate(device, hidden * 4)?;
        let rects = vec![rect(self.checkpoint, norm, None, None)?, rect(self.checkpoint, proj, None, None)?];
        self.step(device, rects, vec![Job::Product { at: 0, with: hidden * 2, bytes: hidden * 2, dst: out.buffer }])?;
        Ok(out)
    }

    /// Residual site `(base, residual, following)` as (score, norm) on `device`.
    fn residual(&self, device: Device<'a>, base: &str, residual: &str, following: &str)
        -> Result<(Allocation<'a>, Allocation<'a>)> {
        let score = self.score(device, &format!("{base}.{residual}_norm.weight"), &format!("{base}.{residual}_proj.weight"))?;
        let norm = self.whole(device, &format!("{base}.{following}.weight"), DType::Bf16, &[self.cfg.hidden])?;
        Ok((score, norm))
    }

    /// A BF16 projection share as the host packer slices it: `owners.len()`
    /// row shares (output side) or column shares (input side); `index` is this
    /// owner's share.
    fn bf16_share(&self, device: Device<'a>, name: &str, shape: [usize; 2], side: TrellisSide, ranks: usize,
        index: usize) -> Result<Allocation<'a>> {
        expect(self.checkpoint, name, DType::Bf16, &shape)?;
        let [n, k] = shape;
        let r = match side {
            TrellisSide::Output => { ensure!(n % ranks == 0, "{name}: row share");
                rect(self.checkpoint, name, Some(index * n / ranks..(index + 1) * n / ranks), None)? }
            TrellisSide::Input => { ensure!(k % ranks == 0, "{name}: column share");
                let share = k * 2 / ranks;
                rect(self.checkpoint, name, None, Some(index * share..(index + 1) * share))? }
        };
        let bytes = r.bytes();
        self.landed(device, vec![(r, 0)], bytes)
    }

    /// A K6 projection's native shard: the codebook marker read and checked
    /// now, the trellis rows staged a K16 window at a time and reordered on
    /// the GPU into `layout`'s side, `suh`/`svh` slices landed unchanged.
    fn core(&self, device: Device<'a>, layout: &CoreWeightLayout) -> Result<CoreWeights<'a>> {
        let stem = &layout.stem;
        let suffixes = ["trellis", "suh", "svh", layout.codebook.marker()];
        let metadata = suffixes.iter().map(|&s| Ok((s, meta(self.checkpoint, &format!("{stem}.{s}"))?.clone())))
            .collect::<Result<BTreeMap<_, _>>>()?;
        let module = exl3_module(stem, &suffixes, |s| metadata.get(s))?;
        ensure!(module.codebook == layout.codebook && module.bits == 6, "{stem}: K6 codebook/bits differ from admission");
        // The marker is four bytes: checked on the host while planning.
        let marker = read_tensor(self.checkpoint, &format!("{stem}.{}", layout.codebook.marker()))?;
        ensure!(marker.bytes.as_slice() == layout.codebook.multiplier().to_le_bytes(),
            "{stem}: unsupported {} marker payload", layout.codebook.marker());
        let (n, k) = (module.output_features, module.input_features);
        let [local_n, local_k] = layout.shape;
        let (n0, k0) = match layout.side {
            TrellisSide::Output => (layout.rank * local_n, 0),
            TrellisSide::Input => (0, layout.rank * local_k),
        };
        ensure!(match layout.side { TrellisSide::Output => local_k == k && local_n * layout.ranks == n,
            TrellisSide::Input => local_n == n && local_k * layout.ranks == k }
            && local_n % 128 == 0 && local_k % 128 == 0, "{stem}: K6 shard geometry");
        let tile_bytes = 32 * module.bits;
        let payload = self.allocate(device, local_n * local_k * module.bits / 8)?;
        let trellis = format!("{stem}.trellis");
        // Checkpoint [K16][N16][tile]: rows are K16, this shard's tiles are a
        // contiguous N16 column window of each row.
        let row_bytes = (n / 16) * tile_bytes;
        let cols = (n0 / 16) * tile_bytes..(n0 + local_n) / 16 * tile_bytes;
        let width = cols.len();
        let k16 = local_k / 16;
        let per = (STEP_BYTES / width).max(1);
        let mut kr = 0;
        while kr < k16 {
            let rows = per.min(k16 - kr);
            let source = rect(self.checkpoint, &trellis, Some(k0 / 16 + kr..k0 / 16 + kr + rows), Some(cols.clone()))?;
            ensure!(source.row_bytes == row_bytes, "{trellis}: trellis row extent");
            self.step(device, vec![source], vec![Job::TrellisTiles { rows, k0: kr, tiles: local_n / 16, tile_bytes,
                k16, input: layout.side == TrellisSide::Input, dst: payload.buffer }])?;
            kr += rows;
        }
        let vector = |suffix: &str, start: usize, count: usize| -> Result<Allocation<'a>> {
            let name = format!("{stem}.{suffix}");
            let r = rect(self.checkpoint, &name, None, Some(start * 2..(start + count) * 2))?;
            let bytes = r.bytes();
            self.landed(device, vec![(r, 0)], bytes)
        };
        let suh = vector("suh", k0, local_k)?;
        let svh = vector("svh", n0, local_n)?;
        ensure!(payload.buffer.bytes + suh.buffer.bytes + svh.buffer.bytes == layout.bytes,
            "{stem}: K6 shard differs from admitted bytes");
        Ok(CoreWeights { payload, suh, svh, shape: layout.shape, side: layout.side, bits: layout.bits, codebook: layout.codebook })
    }

    /// One rank's global weights, allocated from `layout`.
    pub fn global(&self, device: Device<'a>, layout: &GlobalLayout) -> Result<GlobalWeights<'a>> {
        let mut operands = BTreeMap::new();
        let [vocab, hidden] = [layout.vocab, layout.hidden];
        let core_head = if let Some(head) = &layout.core_head {
            Some(self.core(device, head)?)
        } else {
            let (name, key) = if layout.rank == 0 { ("language_model.model.embed_tokens.weight", "embedding") }
                else { ("language_model.lm_head.weight", "head") };
            operands.insert(key, self.whole(device, name, DType::Bf16, &[vocab, hidden])?);
            None
        };
        if layout.rank == 1 {
            let (score, norm) = self.residual(device, "language_model.model", "output_attn_res", "norm")?;
            operands.insert("output.score", score);
            operands.insert("output.norm", norm);
        }
        ensure!(operands.len() == layout.operands.len() && layout.operands.iter().all(|(name, bytes)|
            operands.get(name).is_some_and(|a: &Allocation<'_>| a.buffer.bytes == *bytes)),
            "Kimi recorded global operands differ from admission");
        Ok(GlobalWeights { operands, core_head })
    }

    /// One layer's weights on both ranks, allocated from `layouts` (the
    /// accepted admission extents), as `PackedLayer::pair_with_core` packs them.
    pub fn layer(&self, devices: [Device<'a>; 2], layer: usize, layouts: &[LayerLayout; 2])
        -> Result<[LayerWeights<'a>; 2]> {
        let cfg = self.cfg;
        let mut ops: [BTreeMap<&'static str, Allocation<'a>>; 2] = Default::default();
        let mut core: [BTreeMap<&'static str, CoreWeights<'a>>; 2] = Default::default();
        let base = format!("language_model.model.layers.{layer}");
        for rank in 0..2 {
            for (score, norm, residual, following) in [("attention.score", "attention.norm", "self_attention_res", "input_layernorm"),
                ("ffn.score", "ffn.norm", "mlp_res", "post_attention_layernorm")] {
                let (s, n) = self.residual(devices[rank], &base, residual, following)?;
                ops[rank].insert(score, s);
                ops[rank].insert(norm, n);
            }
        }
        // FFN: fused BF16 halves (gate rows, up rows), or K6 shared experts.
        if layouts[0].operands.contains_key("ffn.gate_up") {
            let dense = cfg.dense[layer];
            let width = if dense { cfg.dense_intermediate } else { cfg.moe_intermediate * cfg.shared_experts };
            let prefix = if dense { format!("{base}.mlp") } else { format!("{base}.block_sparse_moe.shared_experts") };
            for rank in 0..2 {
                let rows = rank * width / 2..(rank + 1) * width / 2;
                let mut lands = Vec::new();
                let mut at = 0;
                for leaf in ["gate", "up"] {
                    let name = format!("{prefix}.{leaf}_proj.weight");
                    expect(self.checkpoint, &name, DType::Bf16, &[width, cfg.hidden])?;
                    let r = rect(self.checkpoint, &name, Some(rows.clone()), None)?;
                    let bytes = r.bytes();
                    lands.push((r, at));
                    at += bytes;
                }
                ops[rank].insert("ffn.gate_up", self.landed(devices[rank], lands, at)?);
                ops[rank].insert("ffn.down", self.bf16_share(devices[rank], &format!("{prefix}.down_proj.weight"),
                    [cfg.hidden, width], TrellisSide::Input, 2, rank)?);
            }
        } else {
            let root = format!("{base}.block_sparse_moe.shared_experts");
            let width = cfg.moe_intermediate * cfg.shared_experts;
            for (name, leaf, shape, side) in [("ffn.gate", "gate_proj", [width, cfg.hidden], TrellisSide::Output),
                ("ffn.up", "up_proj", [width, cfg.hidden], TrellisSide::Output),
                ("ffn.down", "down_proj", [cfg.hidden, width], TrellisSide::Input)] {
                self.projection(devices, layouts, &mut ops, &mut core, name, &format!("{root}.{leaf}"), shape, side, &[0, 1])?;
            }
        }
        if !cfg.dense[layer] {
            let owner = routed_owner(cfg, layer)?;
            let device = devices[owner];
            let root = format!("{base}.block_sparse_moe");
            if layouts[owner].core.contains_key("latent.down") {
                for (name, leaf, shape) in [("latent.down", "routed_expert_down_proj", [cfg.latent, cfg.hidden]),
                    ("latent.up", "routed_expert_up_proj", [cfg.hidden, cfg.latent])] {
                    self.projection(devices, layouts, &mut ops, &mut core, name, &format!("{root}.{leaf}"), shape,
                        TrellisSide::Output, &[owner])?;
                }
            } else {
                ops[owner].insert("latent.down", self.whole(device, &format!("{root}.routed_expert_down_proj.weight"),
                    DType::Bf16, &[cfg.latent, cfg.hidden])?);
                ops[owner].insert("latent.up", self.whole(device, &format!("{root}.routed_expert_up_proj.weight"),
                    DType::Bf16, &[cfg.hidden, cfg.latent])?);
            }
            ops[owner].insert("latent.norm", self.whole(device, &format!("{root}.routed_expert_norm.weight"),
                DType::Bf16, &[cfg.latent])?);
            ops[owner].insert("router.weight", self.whole(device, &format!("{root}.gate.weight"), DType::Bf16,
                &[cfg.experts, cfg.hidden])?);
            ops[owner].insert("router.bias", self.whole(device, &format!("{root}.gate.e_score_correction_bias"),
                DType::F32, &[cfg.experts])?);
        }
        let shares = layouts.each_ref().map(|layout| layout.attention);
        match cfg.attention[layer] {
            KimiAttention::Kda => self.kda(devices, layer, layouts, &mut ops, &mut core)?,
            KimiAttention::Mla => self.mla(devices, layer, layouts, &shares, &mut ops, &mut core)?,
        }
        let [o0, o1] = ops;
        let [c0, c1] = core;
        let pair = [LayerWeights { operands: o0, attention: shares[0], core: c0 },
            LayerWeights { operands: o1, attention: shares[1], core: c1 }];
        for (layout, weights) in layouts.iter().zip(&pair) { layout.check_owned(weights)?; }
        Ok(pair)
    }

    #[allow(clippy::too_many_arguments)]
    fn projection(&self, devices: [Device<'a>; 2], layouts: &[LayerLayout; 2],
        ops: &mut [BTreeMap<&'static str, Allocation<'a>>; 2], core: &mut [BTreeMap<&'static str, CoreWeights<'a>>; 2],
        name: &'static str, stem: &str, shape: [usize; 2], side: TrellisSide, owners: &[usize]) -> Result<()> {
        ensure!(!owners.is_empty() && owners.iter().all(|&r| r < 2), "Kimi projection owners");
        if layouts[owners[0]].core.contains_key(name) {
            for &owner in owners {
                let layout = layouts[owner].core.get(name).context("Kimi K6 owner layout")?;
                core[owner].insert(name, self.core(devices[owner], layout)?);
            }
        } else {
            for (index, &owner) in owners.iter().enumerate() {
                ops[owner].insert(name, self.bf16_share(devices[owner], &format!("{stem}.weight"), shape, side,
                    owners.len(), index)?);
            }
        }
        Ok(())
    }

    /// A KDA operand from the shared plan: pieces land unchanged; A_log's
    /// raw vector is staged whole, checked, and its active heads uploaded.
    fn kda_operand(&self, devices: [Device<'a>; 2], layer: usize, operand: cuteafd_loader::kda::KdaOperand)
        -> Result<[Allocation<'a>; 2]> {
        use cuteafd_loader::kda::{KdaError, KdaRole};
        let name = |role: KdaRole| kda::tensor_name(layer, role).context("Kimi KDA role absent");
        let plan = kda::geometry(self.cfg).plan(operand, 2, |role| {
            let tensor = meta(self.checkpoint, &name(role).map_err(|e| KdaError::Source(e.to_string()))?)
                .map_err(|e| KdaError::Source(e.to_string()))?;
            Ok((tensor.dtype.clone(), tensor.shape.clone()))
        })?;
        let element = if plan.dtype == DType::F32 { 4 } else { 2 };
        let mut out = Vec::new();
        for (rank, pieces) in plan.ranks.iter().enumerate() {
            let bytes = pieces.iter().map(|p| p.elements()).sum::<usize>() * element;
            if let Some(heads) = plan.a_log {
                let tensor = name(KdaRole::ALog)?;
                let raw = usize::try_from(meta(self.checkpoint, &tensor)?.byte_length)?;
                let allocation = self.allocate(devices[rank], bytes)?;
                self.step(devices[rank], vec![rect(self.checkpoint, &tensor, None, Some(0..raw))?],
                    vec![Job::ALog { heads }, Job::Upload { at: pieces[0].rows.start * 4, bytes, dst: allocation.buffer }])?;
                out.push(allocation);
                continue;
            }
            ensure!(pieces.iter().all(|p| p.source == plan.dtype), "Kimi KDA operands keep their checkpoint dtype");
            let mut lands = Vec::new();
            for piece in pieces {
                let tensor = name(piece.role)?;
                let shape = &meta(self.checkpoint, &tensor)?.shape;
                let r = if shape.len() == 1 {
                    rect(self.checkpoint, &tensor, None, Some(piece.rows.start * element..piece.rows.end * element))?
                } else {
                    rect(self.checkpoint, &tensor, Some(piece.rows.clone()),
                        Some(piece.window.start * element..piece.window.end * element))?
                };
                lands.push((r, piece.at * element));
            }
            out.push(self.landed(devices[rank], lands, bytes)?);
        }
        <[_; 2]>::try_from(out).map_err(|_| anyhow::anyhow!("Kimi KDA ranks"))
    }

    fn kda(&self, devices: [Device<'a>; 2], layer: usize, layouts: &[LayerLayout; 2],
        ops: &mut [BTreeMap<&'static str, Allocation<'a>>; 2], core: &mut [BTreeMap<&'static str, CoreWeights<'a>>; 2])
        -> Result<()> {
        use cuteafd_loader::kda::KdaOperand;
        let fused = layouts[0].operands.contains_key("kda.input");
        for (operand, name) in [(KdaOperand::WIn, "kda.input"), (KdaOperand::WFg, "kda.forget"),
            (KdaOperand::ConvW, "kda.conv"), (KdaOperand::ALog, "kda.a_log"), (KdaOperand::DtBias, "kda.dt_bias"),
            (KdaOperand::ONorm, "kda.norm"), (KdaOperand::WO, "kda.output")] {
            if !layouts[0].operands.contains_key(name) || (operand == KdaOperand::WO && !fused) { continue; }
            let [a, b] = self.kda_operand(devices, layer, operand)?;
            ops[0].insert(name, a);
            ops[1].insert(name, b);
        }
        if !fused {
            let root = format!("language_model.model.layers.{layer}.self_attn");
            let cfg = self.cfg;
            let width = cfg.kda_heads * cfg.kda_head_dim;
            let fa = format!("{root}.f_a_proj.weight");
            let beta = format!("{root}.b_proj.weight");
            expect(self.checkpoint, &fa, DType::Bf16, &[cfg.kda_head_dim, cfg.hidden])?;
            expect(self.checkpoint, &beta, DType::Bf16, &[cfg.kda_heads, cfg.hidden])?;
            for rank in 0..2 {
                let a = rect(self.checkpoint, &fa, None, None)?;
                let b = rect(self.checkpoint, &beta, Some(rank * cfg.kda_heads / 2..(rank + 1) * cfg.kda_heads / 2), None)?;
                let (at, bytes) = (a.bytes(), a.bytes() + b.bytes());
                ops[rank].insert("kda.fab", self.landed(devices[rank], vec![(a, 0), (b, at)], bytes)?);
            }
            for (name, leaf, shape, side) in [("kda.q", "q_proj", [width, cfg.hidden], TrellisSide::Output),
                ("kda.k", "k_proj", [width, cfg.hidden], TrellisSide::Output),
                ("kda.v", "v_proj", [width, cfg.hidden], TrellisSide::Output),
                ("kda.gate", "g_proj", [width, cfg.hidden], TrellisSide::Output),
                ("kda.output", "o_proj", [cfg.hidden, width], TrellisSide::Input)] {
                self.projection(devices, layouts, ops, core, name, &format!("{root}.{leaf}"), shape, side, &[0, 1])?;
            }
        }
        Ok(())
    }

    fn mla(&self, devices: [Device<'a>; 2], layer: usize, layouts: &[LayerLayout; 2], shares: &[AttentionShare; 2],
        ops: &mut [BTreeMap<&'static str, Allocation<'a>>; 2], core: &mut [BTreeMap<&'static str, CoreWeights<'a>>; 2])
        -> Result<()> {
        let cfg = self.cfg;
        let owners: Vec<usize> = (0..2).filter(|&r| shares[r].present()).collect();
        ensure!(!owners.is_empty(), "Kimi MLA layer without an owner");
        let root = format!("language_model.model.layers.{layer}.self_attn");
        let fused = layouts[owners[0]].operands.contains_key("mla.qkv_a");
        let (heads, nope, v, lat) = (cfg.heads, cfg.qk_nope_dim, cfg.v_head_dim, cfg.kv_lora_rank);
        let local = heads / owners.len();
        let kv_b = format!("{root}.kv_b_proj.weight");
        expect(self.checkpoint, &kv_b, DType::Bf16, &[heads * (nope + v), lat])?;
        for (index, &owner) in owners.iter().enumerate() {
            let device = devices[owner];
            if layouts[owner].operands.contains_key("mla.q_norm") {
                ops[owner].insert("mla.q_norm", self.whole(device, &format!("{root}.q_a_layernorm.weight"), DType::Bf16,
                    &[cfg.q_lora_rank])?);
            }
            if layouts[owner].operands.contains_key("mla.kv_norm") {
                ops[owner].insert("mla.kv_norm", self.whole(device, &format!("{root}.kv_a_layernorm.weight"), DType::Bf16,
                    &[lat])?);
            }
            // kv_b absorbed per head on the GPU, whole heads per step.
            let uk = self.allocate(device, local * lat * nope * 2)?;
            let uv = self.allocate(device, local * v * lat * 2)?;
            let per = (STEP_BYTES / ((nope + v) * lat * 2)).max(1);
            let mut h = 0;
            while h < local {
                let count = per.min(local - h);
                let first = index * local + h;
                self.step(device, vec![rect(self.checkpoint, &kv_b, Some(first * (nope + v)..(first + count) * (nope + v)), None)?],
                    vec![Job::Absorb { heads: count, nope, v, lat, uk: view(uk.buffer, h * lat * nope * 2, count * lat * nope * 2)?,
                        uv: view(uv.buffer, h * v * lat * 2, count * v * lat * 2)? }])?;
                h += count;
            }
            ops[owner].insert("mla.uk", uk);
            ops[owner].insert("mla.uv", uv);
        }
        if fused {
            for (index, &owner) in owners.iter().enumerate() {
                let device = devices[owner];
                let q_a = format!("{root}.q_a_proj.weight");
                let kv_a = format!("{root}.kv_a_proj_with_mqa.weight");
                expect(self.checkpoint, &q_a, DType::Bf16, &[cfg.q_lora_rank, cfg.hidden])?;
                expect(self.checkpoint, &kv_a, DType::Bf16, &[lat + cfg.qk_rope_dim, cfg.hidden])?;
                let (a, b) = (rect(self.checkpoint, &q_a, None, None)?, rect(self.checkpoint, &kv_a, None, None)?);
                let (at, bytes) = (a.bytes(), a.bytes() + b.bytes());
                ops[owner].insert("mla.qkv_a", self.landed(device, vec![(a, 0), (b, at)], bytes)?);
                let qdim = nope + cfg.qk_rope_dim;
                ops[owner].insert("mla.q_b", self.bf16_share(device, &format!("{root}.q_b_proj.weight"),
                    [heads * qdim, cfg.q_lora_rank], TrellisSide::Output, owners.len(), index)?);
                ops[owner].insert("mla.gate", self.bf16_share(device, &format!("{root}.g_proj.weight"),
                    [heads * v, cfg.hidden], TrellisSide::Output, owners.len(), index)?);
                ops[owner].insert("mla.output", self.bf16_share(device, &format!("{root}.o_proj.weight"),
                    [cfg.hidden, heads * v], TrellisSide::Input, owners.len(), index)?);
            }
        } else {
            for (name, leaf, shape, side, replicated) in [
                ("mla.q_a", "q_a_proj", [cfg.q_lora_rank, cfg.hidden], TrellisSide::Output, true),
                ("mla.kv_a", "kv_a_proj_with_mqa", [lat + cfg.qk_rope_dim, cfg.hidden], TrellisSide::Output, true),
                ("mla.q_b", "q_b_proj", [heads * (nope + cfg.qk_rope_dim), cfg.q_lora_rank], TrellisSide::Output, false),
                ("mla.gate", "g_proj", [heads * v, cfg.hidden], TrellisSide::Output, false),
                ("mla.output", "o_proj", [cfg.hidden, heads * v], TrellisSide::Input, false),
            ] {
                let stem = format!("{root}.{leaf}");
                if replicated {
                    for &owner in &owners {
                        self.projection(devices, layouts, ops, core, name, &stem, shape, side, &[owner])?;
                    }
                } else { self.projection(devices, layouts, ops, core, name, &stem, shape, side, &owners)?; }
            }
        }
        Ok(())
    }
}

impl LayerLayout {
    /// Recorded owners match the admitted extents exactly (as `check` does for packed bytes).
    fn check_owned(&self, weights: &LayerWeights<'_>) -> Result<()> {
        ensure!(self.attention == weights.attention && self.operands.len() == weights.operands.len()
            && self.core.len() == weights.core.len(), "Kimi recorded operand inventory differs from admission");
        for (name, bytes) in &self.operands {
            ensure!(weights.operands.get(name).is_some_and(|a| a.buffer.bytes == *bytes),
                "Kimi recorded operand {name} differs from admission");
        }
        for (name, layout) in &self.core {
            let owned = weights.core.get(name).context("Kimi recorded core operand absent")?;
            ensure!(owned.shape == layout.shape && owned.side == layout.side && owned.codebook == layout.codebook
                && owned.payload.buffer.bytes + owned.suh.buffer.bytes + owned.svh.buffer.bytes == layout.bytes,
                "Kimi recorded K6 operand {name} differs from admission");
        }
        Ok(())
    }
}
