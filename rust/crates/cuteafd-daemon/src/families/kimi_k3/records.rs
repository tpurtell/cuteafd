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
use crate::shared::experts::rtx::dense::{DenseLoad, Job, Operand, Rect, Step, TRANSFORM_BYTES};
use cuteafd_loader::plan::Checkpoint;
use crate::shared::memory::weight_arena::DeviceWeightArena;
use cuteafd_core::memory_layout::ArenaTag;

/// One coordinator GPU's records target.
pub(crate) struct KimiRecords<'l, 'a> {
    pub checkpoint: &'l Checkpoint,
    pub cfg: &'l KimiK3Config,
    pub core: &'l CoreInventory,
    pub dense: &'l DenseLoad<'a>,
    /// (device, load stream) of each coordinator GPU.
    pub streams: Vec<(i32, *mut std::ffi::c_void)>,
    pub arenas: std::rc::Rc<BTreeMap<i32, DeviceWeightArena<'a>>>,
    pub prefix: String,
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

impl<'l, 'a> KimiRecords<'l, 'a> {
    pub fn new(checkpoint: &'l Checkpoint, cfg: &'l KimiK3Config, core: &'l CoreInventory,
        dense: &'l DenseLoad<'a>, streams: Vec<(i32, *mut std::ffi::c_void)>) -> Self {
        Self { checkpoint, cfg, core, dense, streams, arenas: Default::default(), prefix: String::new() }
    }

    fn scoped(&self, prefix: String, devices: &[Device<'a>], tensors: Vec<Vec<(String, usize)>>) -> Result<Self> {
        let arenas = if self.arenas.is_empty() {
            devices.iter().zip(tensors).map(|(&device, tensors)| {
                let layout = cuteafd_loader::serving_capacity::WeightArenaLayout::new(
                    format!("kimi/{prefix}/{}", device.id), tensors)?;
                Ok((device.id, DeviceWeightArena::new(device, layout, ArenaTag::weights("kimi/coordinator"))?))
            }).collect::<Result<BTreeMap<_, _>>>()?.into()
        } else { self.arenas.clone() };
        Ok(Self { checkpoint: self.checkpoint, cfg: self.cfg, core: self.core, dense: self.dense,
            streams: self.streams.clone(), arenas, prefix })
    }

    fn allocate(&self, device: Device<'a>, name: &str, bytes: usize) -> Result<Allocation<'a>> {
        let arena = self.arenas.get(&device.id).context("Kimi device weight arena absent")?;
        crate::shared::memory::weight_arena::weight_allocation(device, Some(arena),
            &format!("{}.{name}", self.prefix), bytes, ArenaTag::weights("kimi/coordinator"))
    }

    /// An allocation of exactly `bytes` whose `lands` fill it unchanged.
    fn landed(&self, device: Device<'a>, key: &str, lands: Vec<(Rect, usize)>, bytes: usize) -> Result<Allocation<'a>> {
        ensure!(lands.iter().map(|(r, _)| r.bytes()).sum::<usize>() == bytes, "Kimi landed operand coverage");
        let allocation = self.allocate(device, key, bytes)?;
        self.dense.operand(Operand { device: device.id, buffer: allocation.buffer, lands, bytes })?;
        Ok(allocation)
    }

    fn step(&self, device: Device<'a>, rects: Vec<Rect>, jobs: Vec<Job>) -> Result<()> {
        let stream = self.streams.iter().find(|(id, _)| *id == device.id).context("Kimi load stream for device")?.1;
        self.dense.step(Step { device: device.id, rects, jobs }, stream)
    }

    /// A whole BF16/F32 tensor, landed unchanged.
    fn whole(&self, device: Device<'a>, key: &str, name: &str, dtype: DType, shape: &[usize]) -> Result<Allocation<'a>> {
        expect(self.checkpoint, name, dtype, shape)?;
        let r = rect(self.checkpoint, name, None, None)?;
        let bytes = r.bytes();
        self.landed(device, key, vec![(r, 0)], bytes)
    }

    /// The F32 residual score `norm * proj` (`w8::residual_score_weight`),
    /// multiplied on the GPU from both staged BF16 vectors.
    fn score(&self, device: Device<'a>, key: &str, norm: &str, proj: &str) -> Result<Allocation<'a>> {
        let hidden = self.cfg.hidden;
        expect(self.checkpoint, norm, DType::Bf16, &[hidden])?;
        expect(self.checkpoint, proj, DType::Bf16, &[1, hidden])?;
        let out = self.allocate(device, key, hidden * 4)?;
        let rects = vec![rect(self.checkpoint, norm, None, None)?, rect(self.checkpoint, proj, None, None)?];
        self.step(device, rects, vec![Job::Product { at: 0, with: hidden * 2, bytes: hidden * 2, dst: out.buffer }])?;
        Ok(out)
    }

    /// Residual site `(base, residual, following)` as (score, norm) on `device`.
    fn residual(&self, device: Device<'a>, score: &str, norm: &str, base: &str, residual: &str, following: &str)
        -> Result<(Allocation<'a>, Allocation<'a>)> {
        let score = self.score(device, score, &format!("{base}.{residual}_norm.weight"), &format!("{base}.{residual}_proj.weight"))?;
        let norm = self.whole(device, norm, &format!("{base}.{following}.weight"), DType::Bf16, &[self.cfg.hidden])?;
        Ok((score, norm))
    }

    /// A BF16 projection share as the host packer slices it: `owners.len()`
    /// row shares (output side) or column shares (input side); `index` is this
    /// owner's share.
    fn bf16_share(&self, device: Device<'a>, key: &str, name: &str, shape: [usize; 2], side: TrellisSide, ranks: usize,
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
        self.landed(device, key, vec![(r, 0)], bytes)
    }

    /// A K6 projection's native shard: the codebook marker read and checked
    /// now, the trellis rows staged a K16 window at a time and reordered on
    /// the GPU into `layout`'s side, `suh`/`svh` slices landed unchanged.
    fn core(&self, device: Device<'a>, key: &str, layout: &CoreWeightLayout) -> Result<CoreWeights<'a>> {
        let stream = self.streams.iter().find(|(id, _)| *id == device.id)
            .context("Kimi load stream for device")?.1;
        layout.load_recorded_with(device, self.checkpoint, self.dense, stream, |name, bytes| {
            let suffix = name.strip_prefix(&format!("{}.", layout.stem)).context("Kimi K6 member key")?;
            self.allocate(device, &format!("{key}.{suffix}"), bytes)
        })
    }

    /// One rank's global weights, allocated from `layout`.
    pub fn global(&self, device: Device<'a>, layout: &GlobalLayout) -> Result<GlobalWeights<'a>> {
        let ranks = self.streams.len();
        ensure!(matches!(ranks, 1 | 2) && layout.rank < ranks
            && (ranks == 1 || self.streams[0].0 != self.streams[1].0)
            && self.streams[layout.rank].0 == device.id,
            "Kimi global physical owner inventory");
        let expected = GlobalLayout::ranks_with_core(self.cfg, self.core, ranks)?;
        ensure!(expected[layout.rank] == *layout, "Kimi recorded global differs from admitted geometry");
        let scoped = self.scoped("global".into(), &[device], vec![layout.shared_layout().arena_tensors()?])?;
        let mut operands = BTreeMap::new();
        let [vocab, hidden] = [layout.vocab, layout.hidden];
        let core_head = layout.core_head.as_ref().map(|head| scoped.core(device, "head", head)).transpose()?;
        for (key, name) in [("embedding", "language_model.model.embed_tokens.weight"),
            ("head", "language_model.lm_head.weight")] {
            if layout.operands.contains_key(key) {
                operands.insert(key, scoped.whole(device, key, name, DType::Bf16, &[vocab, hidden])?);
            }
        }
        if layout.operands.contains_key("output.norm") {
            let (score, norm) = scoped.residual(device, "output.score", "output.norm", "language_model.model", "output_attn_res", "norm")?;
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
        self.layer_ranks(&devices, layer, layouts)?.try_into().map_err(|_| anyhow::anyhow!("Kimi pair records"))
    }

    pub fn layer_ranks(&self, devices: &[Device<'a>], layer: usize, layouts: &[LayerLayout])
        -> Result<Vec<LayerWeights<'a>>> {
        let ranks = devices.len();
        ensure!(matches!(ranks, 1 | 2) && layouts.len() == ranks
            && self.streams.len() == ranks && (ranks == 1 || devices[0].id != devices[1].id),
            "Kimi recorded physical owner inventory");
        ensure!(devices.iter().zip(&self.streams).all(|(device, (id, _))| device.id == *id),
            "Kimi recorded load streams differ from physical owners");
        let cfg = self.cfg;
        let mode = if ranks == 1 {
            LayerMode::Whole { gpu: 0, ffn: cuteafd_loader::placement::FfnMode::Owner }
        } else if layouts.iter().all(|layout| layout.attention == AttentionShare::Split) {
            LayerMode::HeadSplit
        } else {
            let owner = layouts.iter().position(|layout| layout.attention == AttentionShare::Whole)
                .context("Kimi recorded layer lacks whole attention owner")?;
            LayerMode::Whole { gpu: owner as u8, ffn: cuteafd_loader::placement::FfnMode::Split }
        };
        ensure!(LayerLayout::ranks_with_core(cfg, layer, mode, self.core, ranks)? == layouts,
            "Kimi recorded layers differ from admitted geometry");
        let prefix = format!("layers.{layer}");
        let scoped = self.scoped(prefix.clone(), devices, layouts.iter()
            .map(|layout| layout.shared_layout().arena_tensors(&prefix)).collect::<Result<_>>()?)?;
        let mut ops: Vec<BTreeMap<&'static str, Allocation<'a>>> = (0..ranks).map(|_| BTreeMap::new()).collect();
        let mut core: Vec<BTreeMap<&'static str, CoreWeights<'a>>> = (0..ranks).map(|_| BTreeMap::new()).collect();
        let base = format!("language_model.model.layers.{layer}");
        for rank in 0..ranks {
            for (score, norm, residual, following) in [("attention.score", "attention.norm", "self_attention_res", "input_layernorm"),
                ("ffn.score", "ffn.norm", "mlp_res", "post_attention_layernorm")] {
                let (s, n) = scoped.residual(devices[rank], score, norm, &base, residual, following)?;
                ops[rank].insert(score, s);
                ops[rank].insert(norm, n);
            }
        }
        // FFN: fused BF16 halves (gate rows, up rows), or K6 shared experts.
        if layouts[0].operands.contains_key("ffn.gate_up") {
            let dense = cfg.dense[layer];
            let width = if dense { cfg.dense_intermediate } else { cfg.moe_intermediate * cfg.shared_experts };
            let prefix = if dense { format!("{base}.mlp") } else { format!("{base}.block_sparse_moe.shared_experts") };
            for rank in 0..ranks {
                let rows = rank * width / ranks..(rank + 1) * width / ranks;
                let mut lands = Vec::new();
                let mut at = 0;
                for leaf in ["gate", "up"] {
                    let name = format!("{prefix}.{leaf}_proj.weight");
                    expect(scoped.checkpoint, &name, DType::Bf16, &[width, cfg.hidden])?;
                    let r = rect(scoped.checkpoint, &name, Some(rows.clone()), None)?;
                    let bytes = r.bytes();
                    lands.push((r, at));
                    at += bytes;
                }
                ops[rank].insert("ffn.gate_up", scoped.landed(devices[rank], "ffn.gate_up", lands, at)?);
                ops[rank].insert("ffn.down", scoped.bf16_share(devices[rank], "ffn.down", &format!("{prefix}.down_proj.weight"),
                    [cfg.hidden, width], TrellisSide::Input, ranks, rank)?);
            }
        } else {
            let root = format!("{base}.block_sparse_moe.shared_experts");
            let width = cfg.moe_intermediate * cfg.shared_experts;
            for (name, leaf, shape, side) in [("ffn.gate", "gate_proj", [width, cfg.hidden], TrellisSide::Output),
                ("ffn.up", "up_proj", [width, cfg.hidden], TrellisSide::Output),
                ("ffn.down", "down_proj", [cfg.hidden, width], TrellisSide::Input)] {
                scoped.projection(devices, layouts, &mut ops, &mut core, name, &format!("{root}.{leaf}"), shape, side, &(0..ranks).collect::<Vec<_>>())?;
            }
        }
        if !cfg.dense[layer] {
            let owner = cuteafd_loader::families::kimi_k3::weight_layout::routed_owner_for(cfg, layer, ranks)?;
            let device = devices[owner];
            let root = format!("{base}.block_sparse_moe");
            if layouts[owner].core.contains_key("latent.down") {
                for (name, leaf, shape) in [("latent.down", "routed_expert_down_proj", [cfg.latent, cfg.hidden]),
                    ("latent.up", "routed_expert_up_proj", [cfg.hidden, cfg.latent])] {
                    scoped.projection(devices, layouts, &mut ops, &mut core, name, &format!("{root}.{leaf}"), shape,
                        TrellisSide::Output, &[owner])?;
                }
            } else {
                ops[owner].insert("latent.down", scoped.whole(device, "latent.down", &format!("{root}.routed_expert_down_proj.weight"),
                    DType::Bf16, &[cfg.latent, cfg.hidden])?);
                ops[owner].insert("latent.up", scoped.whole(device, "latent.up", &format!("{root}.routed_expert_up_proj.weight"),
                    DType::Bf16, &[cfg.hidden, cfg.latent])?);
            }
            ops[owner].insert("latent.norm", scoped.whole(device, "latent.norm", &format!("{root}.routed_expert_norm.weight"),
                DType::Bf16, &[cfg.latent])?);
            ops[owner].insert("router.weight", scoped.whole(device, "router.weight", &format!("{root}.gate.weight"), DType::Bf16,
                &[cfg.experts, cfg.hidden])?);
            ops[owner].insert("router.bias", scoped.whole(device, "router.bias", &format!("{root}.gate.e_score_correction_bias"),
                DType::F32, &[cfg.experts])?);
        }
        let shares = layouts.iter().map(|layout| layout.attention).collect::<Vec<_>>();
        match cfg.attention[layer] {
            KimiAttention::Kda => scoped.kda(devices, layer, layouts, &mut ops, &mut core)?,
            KimiAttention::Mla => scoped.mla(devices, layer, layouts, &shares, &mut ops, &mut core)?,
        }
        let native = layouts.iter().zip(devices).map(|(layout, &device)| {
            layout.native.iter().map(|(&name, layout)| {
                use crate::shared::native_attention::{attention_records, AttentionRecord, NativeAttentionWeights};
                use crate::shared::memory::DeviceAllocation;
                let selected = scoped.checkpoint.projection(&layout.stem)?;
                let [n, k] = layout.shape;
                let (values, scales) = device.run(|| Ok((
                    DeviceAllocation::from_allocation(scoped.allocate(device, &format!("{name}.values"), (n * k * if layout.fp8 { 1 } else { 2 }).max(256))?),
                    if layout.fp8 { Some(DeviceAllocation::from_allocation(scoped.allocate(device, &format!("{name}.scales"), (n.div_ceil(128) * k.div_ceil(128) * 4).max(256))?)) } else { None },
                )))?;
                let records = attention_records(&scoped.checkpoint.snapshot, &selected, scoped.checkpoint.projection_view(&layout.stem),
                    layout, cuteafd_loader::formats::native_attention::Fp8Role::exact("projection", true), crate::shared::fp8_linear::Fp8Scales::Amax, values.buffer,
                    scales.as_ref().map(|scales| scales.buffer))?;
                let stream = scoped.streams.iter().find(|(id, _)| *id == device.id)
                    .context("Kimi native attention load stream")?.1;
                let admitted = scoped.core.native_mount_load(devices.len())?;
                for record in records {
                    match record {
                        AttentionRecord::Operand(record) => {
                            admitted.validate(record.mount_load()?)?;
                            scoped.dense.operand(record)?;
                        }
                        AttentionRecord::Step(record) => {
                            admitted.validate(record.mount_load()?)?;
                            scoped.dense.step(record, stream)?;
                        }
                    }
                }
                Ok((name, NativeAttentionWeights { layout: layout.clone(), values, scales }))
            }).collect::<Result<BTreeMap<_, _>>>()
        }).collect::<Result<Vec<_>>>()?;
        let weights = ops.into_iter().zip(core).zip(shares).zip(native)
            .map(|(((operands, core), attention), native)| LayerWeights { operands, attention, core, native })
            .collect::<Vec<_>>();
        for (layout, weights) in layouts.iter().zip(&weights) { layout.check_owned(weights)?; }
        Ok(weights)
    }

    #[allow(clippy::too_many_arguments)]
    fn projection(&self, devices: &[Device<'a>], layouts: &[LayerLayout],
        ops: &mut [BTreeMap<&'static str, Allocation<'a>>], core: &mut [BTreeMap<&'static str, CoreWeights<'a>>],
        name: &'static str, stem: &str, shape: [usize; 2], side: TrellisSide, owners: &[usize]) -> Result<()> {
        ensure!(!owners.is_empty() && owners.iter().all(|&r| r < devices.len()), "Kimi projection owners");
        if layouts[owners[0]].core.contains_key(name) {
            for &owner in owners {
                let layout = layouts[owner].core.get(name).context("Kimi K6 owner layout")?;
                core[owner].insert(name, self.core(devices[owner], name, layout)?);
            }
        } else if !layouts[owners[0]].native.contains_key(name) {
            for (index, &owner) in owners.iter().enumerate() {
                ops[owner].insert(name, self.bf16_share(devices[owner], name, &format!("{stem}.weight"), shape, side,
                    owners.len(), index)?);
            }
        }
        Ok(())
    }

    /// A KDA operand from the shared plan: pieces land unchanged; A_log's
    /// raw vector is staged whole, checked, and its active heads uploaded.
    fn kda_operand(&self, devices: &[Device<'a>], layer: usize, key: &str, operand: cuteafd_loader::kda::KdaOperand)
        -> Result<Vec<Allocation<'a>>> {
        use cuteafd_loader::kda::{KdaError, KdaRole};
        let name = |role: KdaRole| kda::tensor_name(layer, role).context("Kimi KDA role absent");
        let plan = kda::geometry(self.cfg).plan(operand, devices.len(), |role| {
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
                let allocation = self.allocate(devices[rank], key, bytes)?;
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
            out.push(self.landed(devices[rank], key, lands, bytes)?);
        }
        Ok(out)
    }

    fn kda(&self, devices: &[Device<'a>], layer: usize, layouts: &[LayerLayout],
        ops: &mut [BTreeMap<&'static str, Allocation<'a>>], core: &mut [BTreeMap<&'static str, CoreWeights<'a>>])
        -> Result<()> {
        use cuteafd_loader::kda::KdaOperand;
        let ranks = devices.len();
        let fused = layouts[0].operands.contains_key("kda.input");
        for (operand, name) in [(KdaOperand::WIn, "kda.input"), (KdaOperand::WFg, "kda.forget"),
            (KdaOperand::ConvW, "kda.conv"), (KdaOperand::ALog, "kda.a_log"), (KdaOperand::DtBias, "kda.dt_bias"),
            (KdaOperand::ONorm, "kda.norm"), (KdaOperand::WO, "kda.output")] {
            if !layouts[0].operands.contains_key(name) || (operand == KdaOperand::WO && !fused) { continue; }
            for (rank, allocation) in self.kda_operand(devices, layer, name, operand)?.into_iter().enumerate() {
                ops[rank].insert(name, allocation);
            }
        }
        if !fused {
            let root = format!("language_model.model.layers.{layer}.self_attn");
            let cfg = self.cfg;
            let width = cfg.kda_heads * cfg.kda_head_dim;
            let fa = format!("{root}.f_a_proj.weight");
            let beta = format!("{root}.b_proj.weight");
            expect(self.checkpoint, &fa, DType::Bf16, &[cfg.kda_head_dim, cfg.hidden])?;
            expect(self.checkpoint, &beta, DType::Bf16, &[cfg.kda_heads, cfg.hidden])?;
            for rank in 0..ranks {
                let a = rect(self.checkpoint, &fa, None, None)?;
                let b = rect(self.checkpoint, &beta, Some(rank * cfg.kda_heads / ranks..(rank + 1) * cfg.kda_heads / ranks), None)?;
                let (at, bytes) = (a.bytes(), a.bytes() + b.bytes());
                ops[rank].insert("kda.fab", self.landed(devices[rank], "kda.fab", vec![(a, 0), (b, at)], bytes)?);
            }
            for (name, leaf, shape, side) in [("kda.q", "q_proj", [width, cfg.hidden], TrellisSide::Output),
                ("kda.k", "k_proj", [width, cfg.hidden], TrellisSide::Output),
                ("kda.v", "v_proj", [width, cfg.hidden], TrellisSide::Output),
                ("kda.gate", "g_proj", [width, cfg.hidden], TrellisSide::Output),
                ("kda.output", "o_proj", [cfg.hidden, width], TrellisSide::Input)] {
                self.projection(devices, layouts, ops, core, name, &format!("{root}.{leaf}"), shape, side, &(0..ranks).collect::<Vec<_>>())?;
            }
        }
        Ok(())
    }

    fn mla(&self, devices: &[Device<'a>], layer: usize, layouts: &[LayerLayout], shares: &[AttentionShare],
        ops: &mut [BTreeMap<&'static str, Allocation<'a>>], core: &mut [BTreeMap<&'static str, CoreWeights<'a>>])
        -> Result<()> {
        let cfg = self.cfg;
        let owners: Vec<usize> = (0..devices.len()).filter(|&r| shares[r].present()).collect();
        ensure!(!owners.is_empty(), "Kimi MLA layer without an owner");
        let root = format!("language_model.model.layers.{layer}.self_attn");
        let fused = layouts[owners[0]].operands.contains_key("mla.qkv_a");
        let (heads, nope, v, lat) = (cfg.heads, cfg.qk_nope_dim, cfg.v_head_dim, cfg.kv_lora_rank);
        let local = heads / owners.len();
        if cfg.dsa.is_some() {
            use cuteafd_loader::families::kimi_k3::dsa::IndexerOperand;
            for operand in [IndexerOperand::Query, IndexerOperand::Norm, IndexerOperand::Bias] {
                let (name, shape) = operand.specification(cfg, layer)?;
                for &owner in &owners {
                    ops[owner].insert(operand.arena_name(), self.whole(devices[owner],
                        operand.arena_name(), &name, DType::Bf16, &shape)?);
                }
            }
            let (key, key_shape) = IndexerOperand::Key.specification(cfg, layer)?;
            let (heads, head_shape) = IndexerOperand::Heads.specification(cfg, layer)?;
            expect(self.checkpoint, &key, DType::Bf16, &key_shape)?;
            expect(self.checkpoint, &heads, DType::Bf16, &head_shape)?;
            for &owner in &owners {
                let key_rows = rect(self.checkpoint, &key, None, None)?;
                let head_rows = rect(self.checkpoint, &heads, None, None)?;
                let at = key_rows.bytes();
                let bytes = at.checked_add(head_rows.bytes()).context("Kimi index projection extent overflow")?;
                ops[owner].insert("mla.indexer.kw", self.landed(devices[owner], "mla.indexer.kw",
                    vec![(key_rows, 0), (head_rows, at)], bytes)?);
            }
        }
        let kv_b = format!("{root}.kv_b_proj.weight");
        expect(self.checkpoint, &kv_b, DType::Bf16, &[heads * (nope + v), lat])?;
        for (index, &owner) in owners.iter().enumerate() {
            let device = devices[owner];
            if layouts[owner].operands.contains_key("mla.q_norm") {
                ops[owner].insert("mla.q_norm", self.whole(device, "mla.q_norm", &format!("{root}.q_a_layernorm.weight"), DType::Bf16,
                    &[cfg.q_lora_rank])?);
            }
            if layouts[owner].operands.contains_key("mla.kv_norm") {
                ops[owner].insert("mla.kv_norm", self.whole(device, "mla.kv_norm", &format!("{root}.kv_a_layernorm.weight"), DType::Bf16,
                    &[lat])?);
            }
            // kv_b absorbed per head on the GPU, whole heads per step.
            let uk = self.allocate(device, "mla.uk", local * lat * nope * 2)?;
            let uv = self.allocate(device, "mla.uv", local * v * lat * 2)?;
            let per = (TRANSFORM_BYTES / ((nope + v) * lat * 2)).max(1);
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
                ops[owner].insert("mla.qkv_a", self.landed(device, "mla.qkv_a", vec![(a, 0), (b, at)], bytes)?);
                let qdim = nope + cfg.qk_rope_dim;
                ops[owner].insert("mla.q_b", self.bf16_share(device, "mla.q_b", &format!("{root}.q_b_proj.weight"),
                    [heads * qdim, cfg.q_lora_rank], TrellisSide::Output, owners.len(), index)?);
                ops[owner].insert("mla.gate", self.bf16_share(device, "mla.gate", &format!("{root}.g_proj.weight"),
                    [heads * v, cfg.hidden], TrellisSide::Output, owners.len(), index)?);
                ops[owner].insert("mla.output", self.bf16_share(device, "mla.output", &format!("{root}.o_proj.weight"),
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
            && self.core.len() == weights.core.len() && self.native.len() == weights.native.len(),
            "Kimi recorded operand inventory differs from admission");
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
        for (name, layout) in &self.native {
            let owned = weights.native.get(name).context("Kimi recorded native operand absent")?;
            ensure!(owned.layout == *layout
                && (owned.values.buffer.bytes + owned.scales.as_ref().map_or(0, |s| s.buffer.bytes)) as u64 == layout.bytes()?,
                "Kimi recorded native operand {name} differs from admission");
        }
        Ok(())
    }
}

#[cfg(test)]
mod native_record_tests {
    use super::*;
    use crate::shared::native_attention::{attention_records, AttentionRecord};
    use cuteafd_loader::{formats::{native_attention::{NativeLayout, NativeProjection}, trellis_core::CoreSplit},
        plan::checkpoint::{CoreView, CheckpointTensor, ProjectionStorage, SelectedProjection},
        SafetensorsTensorMetadata};

    #[test]
    fn native_mount_estimate_covers_actual_qb_and_o_records() -> Result<()> {
        for (leaf, shape, split) in [("q_b_proj", [18432, 1536], CoreSplit::Output),
            ("o_proj", [7168, 12288], CoreSplit::Input)] {
            let stem = format!("language_model.model.layers.0.self_attn.{leaf}");
            let selected = SelectedProjection { storage: ProjectionStorage::Native {
                dtype: DType::Bf16, shape: shape.to_vec() }, source_bytes: (shape[0] * shape[1] * 2) as u64,
                tensors: vec![CheckpointTensor { shard: "not-opened.safetensors".into(),
                    meta: SafetensorsTensorMetadata { name: format!("{stem}.weight"), dtype: DType::Bf16,
                        shape: shape.to_vec(), byte_offset: 4096, byte_length: (shape[0] * shape[1] * 2) as u64 } }] };
            for fp8 in [false, true] {
                let core = CoreInventory { native: BTreeMap::from([(stem.clone(), NativeProjection {
                    shape, fp8, split })]), ..Default::default() };
                for ranks in [1, 2] { for rank in 0..ranks {
                    let mut local = shape;
                    match split { CoreSplit::Input => local[1] /= ranks,
                        CoreSplit::Output => local[0] /= ranks, CoreSplit::Whole => {} }
                    let layout = NativeLayout { stem: stem.clone(), shape: local, fp8, rank, ranks, split };
                    let values = CuteafdDeviceBuffer { ptr: 0x1000usize as *mut _,
                        bytes: local[0] * local[1] * if fp8 { 1 } else { 2 }, ..Default::default() };
                    let scales = fp8.then(|| CuteafdDeviceBuffer { ptr: 0x2000usize as *mut _,
                        bytes: local[0] / 128 * (local[1] / 128) * 4, ..Default::default() });
                    let mode = if fp8 { CoreView::Fp8Compact } else { CoreView::Original };
                    let records = attention_records(std::path::Path::new("/not-opened"), &selected, mode,
                        &layout, cuteafd_loader::formats::native_attention::Fp8Role::exact("projection", true), crate::shared::fp8_linear::Fp8Scales::Amax, values, scales)?;
                    let mut actual = cuteafd_loader::shared::dense_read::MountLoad::default();
                    for record in records {
                        actual.merge(match record {
                            AttentionRecord::Operand(record) => record.mount_load()?,
                            AttentionRecord::Step(record) => record.mount_load()?,
                        });
                    }
                    core.native_mount_load(ranks)?.validate(actual)?;
                    assert_eq!(actual.pinned_tables > 0, fp8);
                    assert_eq!(actual.full_row_cache > 0, ranks == 2 && split == CoreSplit::Input);
                } }
            }
        }
        Ok(())
    }
}
