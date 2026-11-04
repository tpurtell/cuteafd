//! DeepSeek V4 coordinator weights, built from raw checkpoint bytes exactly as
//! b12x.integration.cuteafd.weights.WEIGHT_SOURCES prescribes: every operand is
//! checkpoint bytes (row-concatenated where listed) except block-FP8 scales,
//! which the scale-prep program re-lays into MMA tile order.
use crate::shared::memory::DeviceAllocation;
use anyhow::{ensure, Context, Result};
use cuteafd_ffi::programs::Programs;
use cuteafd_ffi::NativeLibrary;
use cuteafd_loader::families::deepseek_v4::{DeepseekV4Config,
    resident::{layer_sources, resident_operands, scale_extents, share, Prep, Share, WeightSplit}};
use cuteafd_loader::OfficialV41Catalog;
use std::collections::HashMap;
use std::ffi::c_void;
use crate::shared::experts::rtx::dense::{DenseLoad, Job, Operand, Rect, Step, STEP_BYTES};

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
    fn rectangles(&self, names: &[String], kind: Share, rank: usize) -> Result<Vec<Rect>> {
        names.iter().map(|name| {
            cuteafd_ffi::memory_ledger::tensor(name);
            let tensor = self.catalog.tensor(name)?;
            let meta = &tensor.metadata;
            let rows = if meta.shape.len() >= 2 { meta.shape[0] } else { 1 };
            let bytes = usize::try_from(meta.byte_length)?;
            ensure!(rows > 0 && bytes % rows == 0, "{name}: invalid native row extent");
            let row_bytes = bytes / rows;
            let (rows, cols) = match kind {
                Share::All => (0..rows, 0..row_bytes),
                Share::Rows => {
                    ensure!(rows % self.ranks() == 0, "{name}: rows do not split over ranks");
                    let part = rows / self.ranks();
                    (rank * part..(rank + 1) * part, 0..row_bytes)
                }
                Share::Cols => {
                    ensure!(meta.shape.len() == 2 && meta.shape[1] % self.ranks() == 0
                        && bytes == meta.shape[0] * meta.shape[1], "{name}: invalid byte column split");
                    let width = row_bytes / self.ranks();
                    (0..rows, rank * width..(rank + 1) * width)
                }
            };
            ensure!(!rows.is_empty() && !cols.is_empty(), "{name}: empty native source");
            Ok(Rect { name: name.clone(), path: self.catalog.snapshot().join(&tensor.shard),
                base: meta.byte_offset, row_bytes, rows, cols })
        }).collect()
    }

    fn native(&self, rects: Vec<Rect>) -> Result<DeviceAllocation<'a>> {
        let mut bytes = 0usize;
        let lands = rects.into_iter().map(|rect| {
            let at = bytes;
            bytes = bytes.checked_add(rect.bytes()).context("V4 operand size overflow")?;
            Ok((rect, at))
        }).collect::<Result<Vec<_>>>()?;
        let allocation = DeviceAllocation::new(self.library, bytes.max(16))?;
        DenseLoad::mount(self.library).operand(Operand { device: allocation.buffer.device_id,
            buffer: allocation.buffer, lands, bytes })?;
        Ok(allocation)
    }

    pub fn tensor(&self, name: &str) -> Result<DeviceAllocation<'a>> {
        self.native(self.rectangles(&[name.into()], Share::All, 0)?)
    }

    fn scale_on(&self, rects: Vec<Rect>, n: usize, k: usize, groups: usize,
        stream: *mut c_void) -> Result<DeviceAllocation<'a>> {
        ensure!(n % 128 == 0 && k % 128 == 0, "block-FP8 weights need 128-multiple extents");
        let n_blocks = groups.checked_mul(n / 128).context("V4 scale rows overflow")?;
        let k_blocks = k / 128;
        let bytes = n_blocks.checked_mul(k_blocks).context("V4 scale grid overflow")?;
        ensure!(bytes > 0 && bytes <= STEP_BYTES && rects.iter().map(Rect::bytes).sum::<usize>() == bytes,
            "V4 scale grid exceeds bounded staging or its headers");
        let output = DeviceAllocation::new(self.library, bytes.checked_mul(512).context("V4 expanded scale overflow")?)?;
        // Load the existing export before recording so missing capabilities fail
        // before any push group starts; replay resolves it on this rank's device.
        self.programs.program(&format!("{}_block_fp8_scale_prep", self.family), &["scale", "scale_mma"])?;
        DenseLoad::mount(self.library).step(Step { device: output.buffer.device_id, rects,
            jobs: vec![Job::BlockFp8Scales { family: self.family, n_blocks, k_blocks, dst: output.buffer }] }, stream)?;
        Ok(output)
    }

    fn ids(&self, name: &str, stream: *mut c_void) -> Result<DeviceAllocation<'a>> {
        let meta = &self.catalog.tensor(name)?.metadata;
        ensure!(meta.dtype == cuteafd_core::DType::I64, "{name}: hash router requires I64 IDs");
        let rects = self.rectangles(&[name.into()], Share::All, 0)?;
        let rect = &rects[0];
        ensure!(rect.width() % 8 == 0 && rect.width() <= STEP_BYTES, "{name}: invalid ID row extent");
        let out = DeviceAllocation::new(self.library, rect.bytes() / 2)?;
        let rows = STEP_BYTES / rect.width();
        for first in (0..rect.rows.len()).step_by(rows) {
            let part = rect.sub(first..(first + rows).min(rect.rows.len()));
            let offset = first * rect.width() / 2;
            let dst = cuteafd_ffi::CuteafdDeviceBuffer { ptr: out.buffer.ptr.cast::<u8>().wrapping_add(offset).cast(),
                bytes: part.bytes() / 2, ..out.buffer };
            DenseLoad::mount(self.library).step(Step { device: out.buffer.device_id,
                rects: vec![part], jobs: vec![Job::NarrowI64I32 { dst }] }, stream)?;
        }
        Ok(out)
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
        crate::shared::peer_split::on_device(self.library, peer.device, || body(peer.stream))
    }

    /// One block's checkpoint rectangles over the head split's ranks (see
    /// [`share`]); each GPU prepares scales at its own slice's extents.
    fn block_shared(&self, cfg: &DeepseekV4Config, prefix: &str, ratio: usize, hash: bool,
        layout: WeightSplit) -> Result<Vec<LayerWeights<'a>>> {
        let ranks = self.ranks();
        let mut operands: Vec<HashMap<&'static str, DeviceAllocation<'a>>> = (0..ranks).map(|_| HashMap::new()).collect();
        for source in layer_sources(cfg, ratio) {
            let kind = share(source.operand, layout.attention, layout.ffn);
            let names: Vec<String> = source.tensors.iter().map(|t| format!("{prefix}.{t}")).collect();
            for (rank, operands) in operands.iter_mut().enumerate() {
                if layout.operand(source.operand, rank).is_none() { continue; }
                let rects = self.rectangles(&names, kind, rank)?;
                let allocation = self.on_rank(rank, |stream| match source.prep {
                    Prep::Raw => self.native(rects),
                    Prep::Scale { n, k, groups } => {
                        let (n, k, groups) = scale_extents(source.operand, kind, n, k, groups, ranks);
                        self.scale_on(rects, n, k, groups, stream)
                    }
                })?;
                operands.insert(source.operand, allocation);
            }
        }
        for (rank, operands) in operands.iter_mut().enumerate() {
            let name = format!("{prefix}.ffn.gate.{}", if hash { "tid2eid" } else { "bias" });
            let allocation = self.on_rank(rank, |stream| if hash { self.ids(&name, stream) }
                else { self.tensor(&name) })?;
            operands.insert(if hash { "gate.tid2eid" } else { "gate.bias" }, allocation);
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
            let rects = self.rectangles(&names, Share::All, 0)?;
            let allocation = match source.prep {
                Prep::Raw => self.native(rects)?,
                Prep::Scale { n, k, groups } => self.scale_on(rects, n, k, groups, self.stream)?,
            };
            operands.insert(source.operand, allocation);
        }
        if hash {
            operands.insert("gate.tid2eid", self.ids(&format!("{prefix}.ffn.gate.tid2eid"), self.stream)?);
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
