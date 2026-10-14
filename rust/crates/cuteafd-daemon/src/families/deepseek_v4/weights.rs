//! DeepSeek V4 coordinator weights, built from raw checkpoint bytes exactly as
//! b12x.integration.cuteafd.weights.WEIGHT_SOURCES prescribes: every operand is
//! checkpoint bytes (row-concatenated where listed) except block-FP8 scales,
//! which the scale-prep program re-lays into MMA tile order.
use crate::shared::memory::DeviceAllocation;
use anyhow::{ensure, Context, Result};
use cuteafd_ffi::programs::{Programs, Scalar};
use cuteafd_ffi::NativeLibrary;
use cuteafd_loader::families::deepseek_v4::{DeepseekV4Config,
    resident::{layer_sources, resident_operands, scale_extents, share, Prep, Share, WeightSplit}};
use cuteafd_loader::OfficialV41Catalog;
use std::collections::HashMap;
use std::ffi::c_void;
use std::os::unix::fs::FileExt;
use crate::shared::peer_split::{slice_2d, Axis};

pub(crate) struct LayerWeights<'a> {
    pub ratio: usize,
    pub hash: bool,
    /// Head/group slices use the split attention programs. Owned attention
    /// keeps whole projections even when its FFN remains split.
    pub attention_split: bool,
    /// Shared-expert intermediate slices use the split FFN programs.
    pub ffn_split: bool,
    /// Device operands; routing adds `gate.bias` (FP32, score layers) or
    /// `gate.tid2eid` (I32 [vocab, topk], hash layers).
    operands: HashMap<&'static str, DeviceAllocation<'a>>,
}

impl LayerWeights<'_> {
    #[cfg(feature = "dev-v4-layers")]
    pub(super) fn graph_bindings(&self) -> Vec<(usize, usize)> {
        let mut operands = self.operands.iter().collect::<Vec<_>>();
        operands.sort_unstable_by_key(|(name, _)| **name);
        operands.into_iter().map(|(_, value)| (value.buffer.ptr as usize, value.buffer.bytes)).collect()
    }

    pub fn ptr(&self, operand: &str) -> Result<*mut c_void> {
        Ok(self.operands.get(operand).with_context(|| format!("layer has no weight {operand}"))?.buffer.ptr)
    }
}

/// The dSpark drafter (`mtp.*`): three window-only blocks, the target-tap
/// projection of stage 0 and the head extras of the last stage.
pub(crate) struct DsparkWeights<'a> {
    pub stages: Vec<LayerWeights<'a>>,
    /// Block-FP8 [dim, taps * dim] and its raw UE8M0 128x128 block scales.
    pub main_proj: DeviceAllocation<'a>,
    pub main_proj_scale: DeviceAllocation<'a>,
    pub main_norm: DeviceAllocation<'a>,
    pub head_fn: DeviceAllocation<'a>,
    pub head_scale: DeviceAllocation<'a>,
    pub head_base: DeviceAllocation<'a>,
    pub norm: DeviceAllocation<'a>,
    /// Markov head, BF16 [vocab, rank] each.
    pub markov_w1: DeviceAllocation<'a>,
    pub markov_w2: DeviceAllocation<'a>,
}

pub(crate) struct ModelWeights<'a> {
    pub layers: Vec<LayerWeights<'a>>,
    pub dspark: Option<DsparkWeights<'a>>,
    pub head: DeviceAllocation<'a>,
    pub head_fn: DeviceAllocation<'a>,
    pub head_scale: DeviceAllocation<'a>,
    pub head_base: DeviceAllocation<'a>,
    pub norm: DeviceAllocation<'a>,
}

pub(crate) struct WeightLoader<'a, 'p> {
    pub library: &'a NativeLibrary,
    pub catalog: &'a OfficialV41Catalog,
    pub programs: &'p Programs<'a>,
    pub family: &'static str,
    pub stream: *mut c_void,
    /// This loader's device (rank 0 of a head split).
    pub device: i32,
    /// The other GPU of a head split (rank 1), if any.
    pub peers: Vec<crate::shared::peer_split::RankDevice>,
}

impl<'a> WeightLoader<'a, '_> {
    fn read(&self, names: &[String]) -> Result<Vec<u8>> {
        cuteafd_ffi::memory_ledger::tensor(names.first().map_or("", String::as_str));
        let mut bytes = Vec::new();
        for name in names {
            let tensor = self.catalog.tensor(name)?;
            let start = bytes.len();
            bytes.resize(start + tensor.metadata.byte_length as usize, 0);
            std::fs::File::open(self.catalog.snapshot().join(&tensor.shard))?
                .read_exact_at(&mut bytes[start..], tensor.metadata.byte_offset)
                .with_context(|| format!("reading {name}"))?;
        }
        Ok(bytes)
    }

    pub fn upload(&self, bytes: &[u8]) -> Result<DeviceAllocation<'a>> {
        let allocation = DeviceAllocation::new(self.library, bytes.len().max(16))?;
        self.library.copy_h2d(allocation.buffer, bytes)?;
        Ok(allocation)
    }

    pub fn tensor(&self, name: &str) -> Result<DeviceAllocation<'a>> {
        cuteafd_ffi::memory_ledger::tensor(name);
        self.upload(&self.read(&[name.to_string()])?)
    }

    fn scale(&self, raw: &[u8], n: usize, k: usize, groups: usize) -> Result<DeviceAllocation<'a>> {
        ensure!(n % 128 == 0 && k % 128 == 0, "block-FP8 weights need 128-multiple extents");
        let (n_blocks, k_blocks) = (groups * n / 128, k / 128);
        ensure!(raw.len() == n_blocks * k_blocks, "scale bytes {} do not cover {n_blocks}x{k_blocks} blocks", raw.len());
        let source = self.upload(raw)?;
        let output = DeviceAllocation::new(self.library, n_blocks * k_blocks * 512)?;
        let program = self.programs.program(&format!("{}_block_fp8_scale_prep", self.family), &["scale", "scale_mma"])?;
        // SAFETY: both buffers are live device allocations sized for the grid;
        // the stream is synchronized before `source` drops.
        unsafe {
            program.launch(&[source.buffer.ptr, output.buffer.ptr],
                &[Scalar::I32(n_blocks as i32), Scalar::I32(k_blocks as i32)], self.stream)?;
            self.library.cuda_stream_synchronize(self.stream)?;
        }
        Ok(output)
    }

    pub fn layer(&self, cfg: &DeepseekV4Config, layer: usize) -> Result<Vec<LayerWeights<'a>>> {
        use cuteafd_loader::placement::{FfnMode, LayerMode};
        let mode = if self.peers.is_empty() { LayerMode::Whole { gpu: 0, ffn: FfnMode::Owner } }
            else { LayerMode::HeadSplit };
        self.layer_mode(cfg, layer, mode)
    }

    /// Explicit frozen owner mode selected before uploading this layer.
    fn layer_mode(&self, cfg: &DeepseekV4Config, layer: usize,
        mode: cuteafd_loader::placement::LayerMode) -> Result<Vec<LayerWeights<'a>>> {
        let ratio = *cfg.compress_ratios.get(layer).context("V4 weight layer past config")?;
        let layout = WeightSplit::for_mode(mode, ratio, self.ranks())?;
        if self.peers.is_empty() {
            return Ok(vec![self.block(cfg, &format!("layers.{layer}"), ratio, cfg.is_hash_layer(layer))?]);
        }
        self.block_shared(cfg, &format!("layers.{layer}"), ratio, cfg.is_hash_layer(layer), layout)
    }

    /// GPUs of the head split this loader fills (1: no split).
    pub fn ranks(&self) -> usize {
        1 + self.peers.len()
    }

    /// Runs `body` with rank `rank`'s device current and its load stream.
    fn on_rank<T>(&self, rank: usize, body: impl FnOnce(*mut c_void) -> Result<T>) -> Result<T> {
        if rank == 0 {
            return body(self.stream);
        }
        let peer = self.peers.get(rank - 1).with_context(|| format!("no rank {rank}"))?;
        crate::shared::peer_split::on_device(self.library, peer.device, self.device, || body(peer.stream))
    }

    /// [`Self::scale`] on `stream` (the current device's).
    fn scale_on(&self, raw: &[u8], n: usize, k: usize, groups: usize, stream: *mut c_void) -> Result<DeviceAllocation<'a>> {
        ensure!(n % 128 == 0 && k % 128 == 0, "block-FP8 weights need 128-multiple extents");
        let (n_blocks, k_blocks) = (groups * n / 128, k / 128);
        ensure!(raw.len() == n_blocks * k_blocks, "scale bytes {} do not cover {n_blocks}x{k_blocks} blocks", raw.len());
        let source = self.upload(raw)?;
        let output = DeviceAllocation::new(self.library, n_blocks * k_blocks * 512)?;
        let program = self.programs.program(&format!("{}_block_fp8_scale_prep", self.family), &["scale", "scale_mma"])?;
        // SAFETY: both buffers are live device allocations sized for the grid;
        // the stream is synchronized before `source` drops.
        unsafe {
            program.launch(&[source.buffer.ptr, output.buffer.ptr],
                &[Scalar::I32(n_blocks as i32), Scalar::I32(k_blocks as i32)], stream)?;
            self.library.cuda_stream_synchronize(stream)?;
        }
        Ok(output)
    }

    /// One block's operands over the head split's ranks (see [`share`]): each
    /// tensor read once, every rank's slice uploaded on its GPU (scales re-laid
    /// by that GPU's scale-prep program at the slice's extents).
    fn block_shared(&self, cfg: &DeepseekV4Config, prefix: &str, ratio: usize, hash: bool,
        layout: WeightSplit) -> Result<Vec<LayerWeights<'a>>> {
        let ranks = self.ranks();
        let mut operands: Vec<HashMap<&'static str, DeviceAllocation<'a>>> = (0..ranks).map(|_| HashMap::new()).collect();
        for source in layer_sources(cfg, ratio) {
            let kind = share(source.operand, layout.attention, layout.ffn);
            let names: Vec<String> = source.tensors.iter().map(|t| format!("{prefix}.{t}")).collect();
            // Per rank, the concatenation of each tensor's slice.
            let mut parts: Vec<Vec<u8>> = vec![Vec::new(); ranks];
            for name in &names {
                let raw = self.read(std::slice::from_ref(name))?;
                let shape = self.catalog.tensor(name)?.metadata.shape.clone();
                for (rank, part) in parts.iter_mut().enumerate() {
                    if layout.operand(source.operand, rank).is_none() { continue; }
                    match kind {
                        Share::All => part.extend_from_slice(&raw),
                        Share::Rows => {
                            ensure!(raw.len() % ranks == 0 && shape.first().is_some_and(|r| r % ranks == 0),
                                "{name}: {shape:?} does not split by rows over {ranks} GPUs");
                            part.extend_from_slice(&raw[rank * raw.len() / ranks..(rank + 1) * raw.len() / ranks]);
                        }
                        Share::Cols => {
                            ensure!(shape.len() == 2 && shape[1] % ranks == 0 && raw.len() == shape[0] * shape[1],
                                "{name}: {shape:?} does not split by byte columns over {ranks} GPUs");
                            part.extend(slice_2d(&raw, shape[0], shape[1], 1, Axis::Cols, rank, ranks));
                        }
                    }
                }
            }
            for (rank, part) in parts.into_iter().enumerate() {
                if layout.operand(source.operand, rank).is_none() { continue; }
                let allocation = self.on_rank(rank, |stream| match source.prep {
                    Prep::Raw => self.upload(&part),
                    Prep::Scale { n, k, groups } => {
                        let (n, k, groups) = scale_extents(source.operand, kind, n, k, groups, ranks);
                        self.scale_on(&part, n, k, groups, stream)
                    }
                })?;
                operands[rank].insert(source.operand, allocation);
            }
        }
        if hash {
            // The checkpoint stores I64 expert ids; the router reads I32.
            let raw = self.read(&[format!("{prefix}.ffn.gate.tid2eid")])?;
            let ids = raw.chunks_exact(8).map(|b| i64::from_le_bytes(b.try_into().unwrap()))
                .map(|id| i32::try_from(id).map(i32::to_le_bytes)).collect::<Result<Vec<_>, _>>()?;
            for (rank, operands) in operands.iter_mut().enumerate() {
                operands.insert("gate.tid2eid", self.on_rank(rank, |_| self.upload(ids.as_flattened()))?);
            }
        } else {
            let raw = self.read(&[format!("{prefix}.ffn.gate.bias")])?;
            for (rank, operands) in operands.iter_mut().enumerate() {
                operands.insert("gate.bias", self.on_rank(rank, |_| self.upload(&raw))?);
            }
        }
        Ok(operands.into_iter().map(|operands| LayerWeights {
            ratio, hash, attention_split: layout.attention, ffn_split: layout.ffn, operands,
        }).collect())
    }

    /// One block's operands from the checkpoint names under `prefix`.
    fn block(&self, cfg: &DeepseekV4Config, prefix: &str, ratio: usize, hash: bool) -> Result<LayerWeights<'a>> {
        let mut operands = HashMap::new();
        for source in layer_sources(cfg, ratio) {
            let names: Vec<String> = source.tensors.iter().map(|t| format!("{prefix}.{t}")).collect();
            let raw = self.read(&names)?;
            let allocation = match source.prep {
                Prep::Raw => self.upload(&raw)?,
                Prep::Scale { n, k, groups } => self.scale(&raw, n, k, groups)?,
            };
            operands.insert(source.operand, allocation);
        }
        if hash {
            // The checkpoint stores I64 expert ids; the router reads I32.
            let raw = self.read(&[format!("{prefix}.ffn.gate.tid2eid")])?;
            let ids = raw.chunks_exact(8).map(|b| i64::from_le_bytes(b.try_into().unwrap()))
                .map(|id| i32::try_from(id).map(i32::to_le_bytes)).collect::<Result<Vec<_>, _>>()?;
            operands.insert("gate.tid2eid", self.upload(ids.as_flattened())?);
        } else {
            operands.insert("gate.bias", self.tensor(&format!("{prefix}.ffn.gate.bias"))?);
        }
        Ok(LayerWeights { ratio, hash, attention_split: false, ffn_split: false, operands })
    }

    /// The drafter, when the checkpoint carries one (`mtp.0.main_proj`).
    fn dspark(&self, cfg: &DeepseekV4Config) -> Result<Option<DsparkWeights<'a>>> {
        if cfg.dspark_block_size == 0 || self.catalog.tensor("mtp.0.main_proj.weight").is_err() {
            return Ok(None);
        }
        let stages = (0..3).map(|stage| self.block(cfg, &format!("mtp.{stage}"), 0, false))
            .collect::<Result<Vec<_>>>()?;
        let last = stages.len() - 1;
        Ok(Some(DsparkWeights {
            stages,
            main_proj: self.tensor("mtp.0.main_proj.weight")?,
            main_proj_scale: self.tensor("mtp.0.main_proj.scale")?,
            main_norm: self.tensor("mtp.0.main_norm.weight")?,
            head_fn: self.tensor(&format!("mtp.{last}.hc_head_fn"))?,
            head_scale: self.tensor(&format!("mtp.{last}.hc_head_scale"))?,
            head_base: self.tensor(&format!("mtp.{last}.hc_head_base"))?,
            norm: self.tensor(&format!("mtp.{last}.norm.weight"))?,
            markov_w1: self.tensor(&format!("mtp.{last}.markov_head.markov_w1.weight"))?,
            markov_w2: self.tensor(&format!("mtp.{last}.markov_head.markov_w2.weight"))?,
        }))
    }

    /// Every layer (rank 0's shares under a head split, the other rank's in the
    /// second value), the drafter and the head.
    #[allow(clippy::type_complexity)]
    pub fn model(&self, cfg: &DeepseekV4Config, dspark: bool) -> Result<(ModelWeights<'a>, Vec<Vec<LayerWeights<'a>>>)> {
        use cuteafd_loader::placement::{FfnMode, LayerMode};
        let modes = vec![if self.ranks() == 1 { LayerMode::Whole { gpu: 0, ffn: FfnMode::Owner } }
            else { LayerMode::HeadSplit }; cfg.n_layers];
        self.model_modes(cfg, dspark, &modes)
    }

    /// Load exactly the pre-admission modes; reject incomplete headers before
    /// any weight upload and verify final per-operand sizes on every rank.
    #[allow(clippy::type_complexity)]
    pub fn model_modes(&self, cfg: &DeepseekV4Config, dspark: bool,
        modes: &[cuteafd_loader::placement::LayerMode]) -> Result<(ModelWeights<'a>, Vec<Vec<LayerWeights<'a>>>)> {
        let inventory = resident_operands(cfg, modes, self.ranks(), dspark,
            self.catalog.tensors().iter().map(|t| &t.metadata))?;
        let mut shares: Vec<Vec<LayerWeights<'a>>> = (0..self.ranks()).map(|_| Vec::new()).collect();
        for (layer, &mode) in modes.iter().enumerate() {
            for (rank, (share, part)) in shares.iter_mut().zip(self.layer_mode(cfg, layer, mode)?).enumerate() {
                let expected = inventory.iter().filter(|i| i.layer == Some(layer) && i.rank == rank).collect::<Vec<_>>();
                ensure!(part.operands.len() == expected.len(), "V4 layer {layer} rank {rank}: operand inventory differs");
                for item in expected {
                    let allocation = part.operands.get(item.operand.as_str())
                        .with_context(|| format!("V4 layer {layer} rank {rank}: missing {}", item.operand))?;
                    ensure!(allocation.buffer.bytes as u64 == item.bytes,
                        "V4 layer {layer} rank {rank} {}: loaded bytes {}, header inventory {}",
                        item.operand, allocation.buffer.bytes, item.bytes);
                }
                share.push(part);
            }
        }
        let mut shares = shares.into_iter();
        let layers = shares.next().context("rank 0")?;
        let model = ModelWeights {
            layers,
            dspark: if dspark { self.dspark(cfg)? } else { None },
            head: self.tensor("head.weight")?,
            head_fn: self.tensor("hc_head_fn")?,
            head_scale: self.tensor("hc_head_scale")?,
            head_base: self.tensor("hc_head_base")?,
            norm: self.tensor("norm.weight")?,
        };
        if let Some(draft) = &model.dspark {
            for (stage, weights) in draft.stages.iter().enumerate() {
                let expected = inventory.iter().filter(|item| item.draft_stage == Some(stage)).count();
                ensure!(weights.operands.len() == expected,
                    "V4 dSpark stage {stage}: operand inventory differs");
            }
        }
        for item in inventory.iter().filter(|item| item.layer.is_none() && item.group != "embedding") {
            let allocation = if let Some(stage) = item.draft_stage {
                let weights = &model.dspark.as_ref().context("V4 inventory expects dSpark")?.stages[stage];
                weights.operands.get(item.operand.as_str()).context("V4 dSpark operand absent")?
            } else {
                match item.operand.as_str() {
                    "head.weight" => &model.head,
                    "hc_head_fn" => &model.head_fn,
                    "hc_head_scale" => &model.head_scale,
                    "hc_head_base" => &model.head_base,
                    "norm.weight" => &model.norm,
                    name => {
                        let draft = model.dspark.as_ref().context("V4 inventory expects dSpark head")?;
                        match name {
                            "mtp.0.main_proj.weight" => &draft.main_proj,
                            "mtp.0.main_proj.scale" => &draft.main_proj_scale,
                            "mtp.0.main_norm.weight" => &draft.main_norm,
                            "mtp.2.hc_head_fn" => &draft.head_fn,
                            "mtp.2.hc_head_scale" => &draft.head_scale,
                            "mtp.2.hc_head_base" => &draft.head_base,
                            "mtp.2.norm.weight" => &draft.norm,
                            "mtp.2.markov_head.markov_w1.weight" => &draft.markov_w1,
                            "mtp.2.markov_head.markov_w2.weight" => &draft.markov_w2,
                            _ => anyhow::bail!("unknown V4 root operand {name}"),
                        }
                    }
                }
            };
            ensure!(allocation.buffer.bytes as u64 == item.bytes,
                "V4 root/dSpark {:?} {}: loaded bytes {}, header inventory {}",
                item.draft_stage, item.operand, allocation.buffer.bytes, item.bytes);
        }
        Ok((model, shares.collect()))
    }

}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn v4_owned_attention_does_not_unsplit_shared_ffn_weights() {
        for (operand, axis) in [("w_q", Share::Rows), ("w_q_scale", Share::Rows),
            ("attn_sink", Share::Rows), ("wo_a", Share::Rows), ("wo_a_scale", Share::Rows),
            ("wo_b", Share::Cols), ("wo_b_scale", Share::Cols)] {
            assert_eq!(share(operand, true, true), axis);
            assert_eq!(share(operand, false, true), Share::All);
        }
        for (operand, axis) in [("w13", Share::Rows), ("w13_scale", Share::Rows),
            ("w2", Share::Cols), ("w2_scale", Share::Cols)] {
            assert_eq!(share(operand, true, true), axis);
            assert_eq!(share(operand, false, true), axis);
            assert_eq!(share(operand, true, false), Share::All);
        }
        for operand in ["w_qkv", "w_qkv_scale", "joint_projection", "index_w_q", "gate"] {
            for attention_split in [false, true] {
                for ffn_split in [false, true] {
                    assert_eq!(share(operand, attention_split, ffn_split), Share::All);
                }
            }
        }
    }

    #[test]
    fn v4_modes_load_attention_only_on_its_owner() -> Result<()> {
        use cuteafd_loader::placement::{FfnMode, LayerMode};
        for owner in 0..2 {
            let layout = WeightSplit::for_mode(LayerMode::Whole { gpu: owner, ffn: FfnMode::Split }, 4, 2)?;
            for operand in ["attn.fn", "attn.norm", "w_qkv", "w_qkv_scale", "w_q", "attn_sink",
                "wo_a", "wo_a_scale", "wo_b", "joint_projection", "main_norm", "index_norm", "index_w_q",
                "ffn.fn", "ffn.norm", "w13", "w13_scale", "w2", "w2_scale", "gate"] {
                for rank in 0..2 {
                    let ffn = operand.starts_with("ffn.") || operand.starts_with("gate")
                        || matches!(operand, "w13" | "w13_scale" | "w2" | "w2_scale");
                    assert_eq!(layout.operand(operand, rank).is_some(), ffn || rank == usize::from(owner),
                        "{operand} on rank {rank}");
                }
            }
            for operand in ["gate.bias", "gate.tid2eid", "ffn.fn", "ffn.norm"] {
                assert_eq!(layout.operand(operand, 0), Some(Share::All));
                assert_eq!(layout.operand(operand, 1), Some(Share::All));
            }
            assert_eq!(layout.operand("w13", 0), Some(Share::Rows));
            assert_eq!(layout.operand("w2", 1), Some(Share::Cols));
            assert_eq!(layout.operand("wo_a", usize::from(owner)), Some(Share::All));
        }
        assert!(WeightSplit::for_mode(LayerMode::ContextSplit, 128, 2).is_err());
        assert!(WeightSplit::for_mode(LayerMode::ContextSplit, 4, 1).is_err());
        assert!(WeightSplit::for_mode(LayerMode::Whole { gpu: 1, ffn: FfnMode::Owner }, 4, 2).is_err());
        assert!(WeightSplit::for_mode(LayerMode::Whole { gpu: 0, ffn: FfnMode::Split }, 128, 2).is_err());
        assert_eq!(WeightSplit::for_mode(LayerMode::ContextSplit, 4, 2)?,
            WeightSplit::for_mode(LayerMode::HeadSplit, 4, 2)?);
        Ok(())
    }

    #[test]
    fn v4_whole_attention_preserves_output_scale_groups() {
        for groups in [8, 16] {
            assert_eq!(scale_extents("wo_a_scale", Share::All, 1024, 4096, groups, 2),
                (1024, 4096, groups));
            assert_eq!(scale_extents("wo_a_scale", Share::Rows, 1024, 4096, groups, 2),
                (1024, 4096, groups / 2));
        }
        assert_eq!(scale_extents("w13_scale", Share::Rows, 4096, 4096, 1, 2), (2048, 4096, 1));
        assert_eq!(scale_extents("w2_scale", Share::Cols, 4096, 2048, 1, 2), (4096, 1024, 1));
        assert_eq!(scale_extents("w_q_scale", Share::Rows, 32768, 1024, 1, 2), (16384, 1024, 1));
    }
}
