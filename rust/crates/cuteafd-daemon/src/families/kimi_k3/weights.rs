//! Checkpoint-native layer packing and device owners for the admitted W9 engine.
//! Pack one layer at a time; allocation starts only after the caller's shared
//! placement admission and an exact comparison with that layer's charged bytes.
use crate::shared::{memory::device::{Allocation, Device}, placement::layers::AttentionShare};
use anyhow::{ensure, Context, Result};
use cuteafd_core::{DType, memory_layout::{Basis, Category}};
use cuteafd_ffi::{CuteafdDeviceBuffer, KimiKdaWeights, KimiMlaWeights};
use cuteafd_loader::{families::kimi_k3::{config::{KimiAttention, KimiK3Config}, kda, mla::{self, MlaOperand},
    w8::{self, ResidualWeightSite}}, kda::{KdaOperand, KdaTensor}, placement::{Demand, LayerMode}, LoadedTensor};
use std::collections::BTreeMap;

/// Global tensors follow the shared entry/head ranks: one embedding on rank 0,
/// one untied full vocabulary head and output mix/norm on rank 1.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct GlobalLayout {
    rank: usize,
    hidden: usize,
    vocab: usize,
    operands: BTreeMap<&'static str, usize>,
}

impl GlobalLayout {
    pub(crate) fn pair(cfg: &KimiK3Config) -> Result<[Self; 2]> {
        ensure!(!cfg.tie_embeddings, "Kimi global weights require the checkpoint's untied head");
        let matrix = extent(&[cfg.vocab_size, cfg.hidden, 2])?;
        Ok([Self { rank: 0, hidden: cfg.hidden, vocab: cfg.vocab_size,
                operands: BTreeMap::from([("embedding", matrix)]) },
            Self { rank: 1, hidden: cfg.hidden, vocab: cfg.vocab_size,
                operands: BTreeMap::from([("head", matrix), ("output.score", extent(&[cfg.hidden, 4])?),
                    ("output.norm", extent(&[cfg.hidden, 2])?)]) }])
    }

    pub(crate) fn bytes(&self) -> Result<usize> {
        self.operands.values().try_fold(0usize, |sum, bytes| sum.checked_add(*bytes)
            .context("Kimi global weight extent overflow"))
    }

    pub(crate) fn demand(&self) -> Result<Demand> {
        Ok(Demand::new(self.rank as u8, Category::Weights, "Kimi global weights", self.bytes()? as u64, Basis::Exact))
    }

    pub(crate) fn pack(&self, cfg: &KimiK3Config, mut read: impl FnMut(&str) -> Result<LoadedTensor>) -> Result<PackedGlobal> {
        ensure!(Self::pair(cfg)?[self.rank] == *self, "Kimi global packing differs from admitted geometry");
        let name = if self.rank == 0 { "language_model.model.embed_tokens.weight" } else { "language_model.lm_head.weight" };
        let tensor = read(name)?;
        check(&tensor, DType::Bf16, &[self.vocab, self.hidden])?;
        let mut operands = BTreeMap::from([(if self.rank == 0 { "embedding" } else { "head" }, tensor.bytes)]);
        if self.rank == 1 {
            let output = w8::load_residual(cfg, ResidualWeightSite::Output, &mut read)?;
            operands.insert("output.score", output.score);
            operands.insert("output.norm", output.norm.bytes);
        }
        let packed = PackedGlobal { rank: self.rank, operands };
        self.check(&packed)?;
        Ok(packed)
    }

    fn check(&self, packed: &PackedGlobal) -> Result<()> {
        ensure!(packed.rank == self.rank && packed.operands.len() == self.operands.len()
            && self.operands.iter().all(|(name, bytes)| packed.operands.get(name).is_some_and(|data| data.len() == *bytes)),
            "Kimi packed global operands differ from admission");
        Ok(())
    }

    pub(super) fn upload<'a>(&self, device: Device<'a>, packed: PackedGlobal) -> Result<GlobalWeights<'a>> {
        self.check(&packed)?;
        let operands = packed.operands.into_iter().map(|(name, bytes)| {
            let _scope = cuteafd_ffi::memory_ledger::scope("weights/kimi global weights");
            let allocation = Allocation::new(device, bytes.len())?;
            ensure!(allocation.buffer.bytes == bytes.len() && allocation.buffer.device_id == device.id,
                "Kimi global allocation differs from admitted extent/device");
            device.run(|| device.library.copy_h2d(allocation.buffer, &bytes))?;
            Ok((name, allocation))
        }).collect::<Result<_>>()?;
        Ok(GlobalWeights { operands })
    }
}

pub(crate) struct PackedGlobal {
    rank: usize,
    operands: BTreeMap<&'static str, Vec<u8>>,
}

pub(crate) struct GlobalWeights<'a> {
    operands: BTreeMap<&'static str, Allocation<'a>>,
}

impl GlobalWeights<'_> {
    pub(crate) fn buffer(&self, name: &str) -> Result<CuteafdDeviceBuffer> {
        Ok(self.operands.get(name).with_context(|| format!("Kimi global operand {name} absent on this rank"))?.buffer)
    }
}

/// Exact resident operand extents, computed without reading tensor payloads.
/// Shared startup admission charges these bytes; packing independently checks
/// every operand before any device allocation can begin.
#[derive(Debug, PartialEq, Eq)]
pub(crate) struct LayerLayout {
    operands: BTreeMap<&'static str, usize>,
    pub(crate) attention: AttentionShare,
}

impl LayerLayout {
    pub(crate) fn pair(cfg: &KimiK3Config, layer: usize, mode: LayerMode) -> Result<[Self; 2]> {
        let kind = *cfg.attention.get(layer).context("Kimi attention layer out of range")?;
        ensure!(cfg.layers == cfg.attention.len() && cfg.dense.len() == cfg.layers,
            "Kimi dense/attention inventories differ");
        ensure!(mode != LayerMode::ContextSplit, "Kimi context attention has no query/LSE exchange ABI");
        ensure!(!matches!(mode, LayerMode::Whole { ffn: cuteafd_loader::placement::FfnMode::Owner, .. }),
            "Kimi FFNs require TP2 intermediate slices");
        ensure!(kind != KimiAttention::Kda || mode == LayerMode::HeadSplit,
            "Kimi KDA requires TP2 head-split placement");
        let shares = AttentionShare::bind(mode, 2)?;
        let intermediate = if cfg.dense[layer] { cfg.dense_intermediate }
            else { extent(&[cfg.moe_intermediate, cfg.shared_experts])? };
        ensure!(intermediate > 0 && intermediate % 2 == 0, "Kimi FFN intermediate does not divide TP2");
        let local = intermediate / 2;
        let sum = |a: usize, b: usize| a.checked_add(b).context("Kimi weight extent overflow");
        let mut pair = std::array::from_fn(|rank| Self { operands: BTreeMap::new(), attention: shares[rank] });
        for rank in &mut pair {
            for name in ["attention.score", "ffn.score"] { rank.insert(name, &[cfg.hidden, 4])?; }
            for name in ["attention.norm", "ffn.norm"] { rank.insert(name, &[cfg.hidden, 2])?; }
            rank.insert("ffn.gate_up", &[2, local, cfg.hidden, 2])?;
            rank.insert("ffn.down", &[cfg.hidden, local, 2])?;
            if !cfg.dense[layer] {
                for name in ["latent.down", "latent.up"] { rank.insert(name, &[cfg.latent, cfg.hidden, 2])?; }
                rank.insert("latent.norm", &[cfg.latent, 2])?;
                rank.insert("router.weight", &[cfg.experts, cfg.hidden, 2])?;
                rank.insert("router.bias", &[cfg.experts, 4])?;
            }
            match kind {
                KimiAttention::Kda => {
                    kda::geometry(cfg).validate(2)?;
                    let heads = cfg.kda_heads / 2;
                    let width = extent(&[heads, cfg.kda_head_dim])?;
                    let input = sum(sum(extent(&[4, width])?, cfg.kda_head_dim)?, heads)?;
                    rank.insert("kda.input", &[input, cfg.hidden, 2])?;
                    rank.insert("kda.forget", &[width, cfg.kda_head_dim, 2])?;
                    rank.insert("kda.conv", &[3, width, cfg.kda_conv, 4])?;
                    rank.insert("kda.a_log", &[heads, 4])?;
                    rank.insert("kda.dt_bias", &[width, 4])?;
                    rank.insert("kda.norm", &[cfg.kda_head_dim, 4])?;
                    rank.insert("kda.output", &[cfg.hidden, width, 2])?;
                }
                KimiAttention::Mla if rank.attention.present() => {
                    ensure!(cfg.heads > 0 && cfg.heads % 2 == 0, "Kimi MLA heads do not divide TP2");
                    let heads = if rank.attention == AttentionShare::Whole { cfg.heads } else { cfg.heads / 2 };
                    let input = sum(cfg.q_lora_rank, sum(cfg.kv_lora_rank, cfg.qk_rope_dim)?)?;
                    rank.insert("mla.qkv_a", &[input, cfg.hidden, 2])?;
                    rank.insert("mla.q_norm", &[cfg.q_lora_rank, 2])?;
                    rank.insert("mla.kv_norm", &[cfg.kv_lora_rank, 2])?;
                    rank.insert("mla.q_b", &[heads, sum(cfg.qk_nope_dim, cfg.qk_rope_dim)?, cfg.q_lora_rank, 2])?;
                    rank.insert("mla.uk", &[heads, cfg.kv_lora_rank, cfg.qk_nope_dim, 2])?;
                    rank.insert("mla.uv", &[heads, cfg.v_head_dim, cfg.kv_lora_rank, 2])?;
                    rank.insert("mla.gate", &[heads, cfg.v_head_dim, cfg.hidden, 2])?;
                    rank.insert("mla.output", &[cfg.hidden, heads, cfg.v_head_dim, 2])?;
                }
                KimiAttention::Mla => {}
            }
        }
        Ok(pair)
    }

    fn insert(&mut self, name: &'static str, dimensions: &[usize]) -> Result<()> {
        ensure!(self.operands.insert(name, extent(dimensions)?).is_none(), "duplicate Kimi operand {name}");
        Ok(())
    }

    pub(crate) fn bytes(&self) -> Result<usize> {
        self.operands.values().try_fold(0usize, |n, size| n.checked_add(*size)
            .context("Kimi layer admission extent overflow"))
    }

    pub(super) fn attention_bytes(&self) -> Result<usize> {
        self.operands.iter().filter(|(name, _)| name.starts_with("mla.") || name.starts_with("kda."))
            .try_fold(0usize, |n, (_, bytes)| n.checked_add(*bytes).context("Kimi attention extent overflow"))
    }

    fn check(&self, packed: &PackedLayer) -> Result<()> {
        ensure!(self.attention == packed.attention && self.operands.len() == packed.operands.len(),
            "Kimi packed operand inventory differs from native admission");
        for (name, bytes) in &self.operands {
            ensure!(packed.operands.get(name).is_some_and(|data| data.len() == *bytes),
                "Kimi packed operand {name} differs from native admission");
        }
        Ok(())
    }
}

fn extent(dimensions: &[usize]) -> Result<usize> {
    ensure!(dimensions.iter().all(|n| *n > 0), "Kimi weight extent is empty");
    dimensions.iter().try_fold(1usize, |n, dim| n.checked_mul(*dim)).context("Kimi weight extent overflow")
}

/// Byte counts describe the packed operands, not the checkpoint's pre-pack
/// representation (the residual score product, for example, is F32).
pub(crate) struct PackedLayer {
    operands: BTreeMap<&'static str, Vec<u8>>,
    attention: AttentionShare,
}

impl PackedLayer {
    pub(crate) fn bytes(&self) -> Result<usize> {
        self.operands.values().try_fold(0usize, |total, bytes|
            total.checked_add(bytes.len()).context("Kimi packed layer size overflow"))
    }

    /// Both ranks are packed together so a source operand is read once, rather
    /// than rereading every tensor separately for each coordinator GPU.
    pub(crate) fn pair(cfg: &KimiK3Config, layer: usize, mode: LayerMode,
        mut read: impl FnMut(&str) -> Result<LoadedTensor>) -> Result<[Self; 2]> {
        let layouts = LayerLayout::pair(cfg, layer, mode)?;
        let kind = cfg.attention[layer];
        let shares = layouts.each_ref().map(|layout| layout.attention);
        let mut pair = std::array::from_fn(|rank| Self {
            operands: BTreeMap::new(), attention: shares[rank] });
        for (site, score, norm) in [
            (ResidualWeightSite::Attention(layer), "attention.score", "attention.norm"),
            (ResidualWeightSite::Ffn(layer), "ffn.score", "ffn.norm"),
        ] {
            let weights = w8::load_residual(cfg, site, &mut read)?;
            for rank in &mut pair {
                rank.operands.insert(score, weights.score.clone());
                rank.operands.insert(norm, weights.norm.bytes.clone());
            }
        }
        for (rank, slice) in pair.iter_mut().zip(w8::pack_ffn(cfg, layer, 2, &mut read)?) {
            rank.operands.insert("ffn.gate_up", slice.gate_up);
            rank.operands.insert("ffn.down", slice.down);
        }
        if !cfg.dense[layer] {
            let latent = w8::load_latent(cfg, layer, &mut read)?;
            let base = format!("language_model.model.layers.{layer}.block_sparse_moe.gate");
            let router = read(&format!("{base}.weight"))?;
            let bias = read(&format!("{base}.e_score_correction_bias"))?;
            check(&router, DType::Bf16, &[cfg.experts, cfg.hidden])?;
            check(&bias, DType::F32, &[cfg.experts])?;
            for rank in &mut pair {
                for (name, bytes) in [("latent.down", &latent.down.bytes), ("latent.norm", &latent.norm.bytes),
                    ("latent.up", &latent.up.bytes), ("router.weight", &router.bytes), ("router.bias", &bias.bytes)] {
                    rank.operands.insert(name, bytes.clone());
                }
            }
        }
        match kind {
            KimiAttention::Kda => {
                for (operand, name) in [(KdaOperand::WIn, "kda.input"), (KdaOperand::WFg, "kda.forget"),
                    (KdaOperand::ConvW, "kda.conv"), (KdaOperand::ALog, "kda.a_log"),
                    (KdaOperand::DtBias, "kda.dt_bias"), (KdaOperand::ONorm, "kda.norm"), (KdaOperand::WO, "kda.output")] {
                    let slices = kda::pack(cfg, layer, operand, 2, |name| {
                        let tensor = read(name)?;
                        Ok(KdaTensor { dtype: tensor.info.dtype, shape: tensor.info.shape, bytes: tensor.bytes })
                    })?;
                    for (rank, slice) in pair.iter_mut().zip(slices) { rank.operands.insert(name, slice.bytes); }
                }
            }
            KimiAttention::Mla => {
                let owners: Vec<_> = shares.iter().enumerate().filter(|(_, share)| share.present())
                    .map(|(rank, _)| rank).collect();
                for (operand, name) in [(MlaOperand::QkvA, "mla.qkv_a"), (MlaOperand::QNorm, "mla.q_norm"),
                    (MlaOperand::KvNorm, "mla.kv_norm"), (MlaOperand::QB, "mla.q_b"), (MlaOperand::Uk, "mla.uk"),
                    (MlaOperand::Uv, "mla.uv"), (MlaOperand::Gate, "mla.gate"), (MlaOperand::Output, "mla.output")] {
                    let slices = mla::pack(cfg, layer, operand, owners.len(), &mut read)?;
                    for (&owner, slice) in owners.iter().zip(slices) { pair[owner].operands.insert(name, slice.bytes); }
                }
            }
        }
        for (layout, packed) in layouts.iter().zip(&pair) { layout.check(packed)?; }
        Ok(pair)
    }
}

fn check(tensor: &LoadedTensor, dtype: DType, shape: &[usize]) -> Result<()> {
    let element = if dtype == DType::F32 { 4usize } else { 2 };
    let bytes = shape.iter().try_fold(element, |size, &dim| size.checked_mul(dim))
        .context("Kimi tensor extent overflow")?;
    ensure!(tensor.info.dtype == dtype && tensor.info.shape == shape && tensor.bytes.len() == bytes,
        "Kimi tensor {} expected {dtype:?} {shape:?} with {bytes} bytes", tensor.info.name);
    Ok(())
}

pub(crate) struct LayerWeights<'a> {
    operands: BTreeMap<&'static str, Allocation<'a>>,
    pub(crate) attention: AttentionShare,
}

impl<'a> LayerWeights<'a> {
    /// `charged_bytes` is the exact weight demand accepted by shared startup
    /// admission, not available CUDA bytes or an independent family budget.
    /// Uploads are synchronous; no queued work can outlive error cleanup here.
    pub(crate) fn upload(device: Device<'a>, packed: PackedLayer, layout: &LayerLayout,
        charged_bytes: usize) -> Result<Self> {
        layout.check(&packed)?;
        ensure!(layout.bytes()? == charged_bytes, "Kimi layer layout differs from admitted weight bytes");
        let operands = packed.operands.into_iter().map(|(name, bytes)| {
            ensure!(!bytes.is_empty(), "Kimi operand {name} is empty");
            let _scope = cuteafd_ffi::memory_ledger::scope("weights/kimi layer weights");
            let allocation = Allocation::new(device, bytes.len())?;
            device.run(|| device.library.copy_h2d(allocation.buffer, &bytes))?;
            Ok((name, allocation))
        }).collect::<Result<_>>()?;
        Ok(Self { operands, attention: packed.attention })
    }

    pub(crate) fn buffer(&self, name: &str) -> Result<CuteafdDeviceBuffer> {
        Ok(self.operands.get(name).with_context(|| format!("Kimi layer has no resident operand {name}"))?.buffer)
    }

    pub(crate) fn kda(&self) -> Result<KimiKdaWeights> {
        ensure!(self.attention == AttentionShare::Split, "Kimi KDA weights are not TP2 split");
        Ok(KimiKdaWeights { input: self.buffer("kda.input")?, forget: self.buffer("kda.forget")?,
            convolution: self.buffer("kda.conv")?, a_log: self.buffer("kda.a_log")?,
            dt_bias: self.buffer("kda.dt_bias")?, norm: self.buffer("kda.norm")?, output: self.buffer("kda.output")? })
    }

    pub(crate) fn mla(&self) -> Result<KimiMlaWeights> {
        ensure!(self.attention.present(), "Kimi MLA weights absent on this nonowner");
        Ok(KimiMlaWeights { qkv_a: self.buffer("mla.qkv_a")?, q_a_norm: self.buffer("mla.q_norm")?,
            kv_a_norm: self.buffer("mla.kv_norm")?, q_b: self.buffer("mla.q_b")?, uk: self.buffer("mla.uk")?,
            uv: self.buffer("mla.uv")?, gate: self.buffer("mla.gate")?, output: self.buffer("mla.output")? })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cuteafd_core::{TensorInfo, TensorRole};
    use cuteafd_loader::placement::FfnMode;

    fn cfg() -> KimiK3Config {
        let mut cfg = KimiK3Config::from_hf(&cuteafd_loader::plan::testing::kimi_k3_config()).unwrap();
        cfg.hidden = 4; cfg.heads = 4; cfg.q_lora_rank = 3; cfg.kv_lora_rank = 2;
        cfg.qk_nope_dim = 2; cfg.qk_rope_dim = 1; cfg.v_head_dim = 1;
        cfg.dense_intermediate = 8; cfg.latent = 2; cfg.experts = 4;
        cfg.moe_intermediate = 4; cfg.shared_experts = 2;
        cfg
    }

    fn tensor(name: &str) -> Result<LoadedTensor> {
        let (dtype, shape) = if name.ends_with("e_score_correction_bias") { (DType::F32, vec![4]) }
        else { (DType::Bf16, if name.ends_with("q_a_proj.weight") { vec![3, 4] }
            else if name.ends_with("q_a_layernorm.weight") { vec![3] }
            else if name.ends_with("kv_a_layernorm.weight") { vec![2] }
            else if name.ends_with("kv_a_proj_with_mqa.weight") { vec![3, 4] }
            else if name.ends_with("q_b_proj.weight") { vec![12, 3] }
            else if name.ends_with("kv_b_proj.weight") { vec![12, 2] }
            else if name.ends_with("g_proj.weight") || name.ends_with("o_proj.weight") { vec![4, 4] }
            else if name.ends_with("routed_expert_down_proj.weight") { vec![2, 4] }
            else if name.ends_with("routed_expert_up_proj.weight") { vec![4, 2] }
            else if name.ends_with("routed_expert_norm.weight") { vec![2] }
            else if name.ends_with("gate.weight") { vec![4, 4] }
            else if name.ends_with("down_proj.weight") { vec![4, 8] }
            else if name.ends_with("gate_proj.weight") || name.ends_with("up_proj.weight") { vec![8, 4] }
            else if name.ends_with("_res_proj.weight") { vec![1, 4] }
            else { vec![4] }) };
        let count: usize = shape.iter().product();
        let bytes = if dtype == DType::F32 { vec![0; count * 4] }
            else { std::iter::repeat_n(0x3f80u16, count).flat_map(u16::to_le_bytes).collect() };
        Ok(LoadedTensor { info: TensorInfo { name: name.into(), file: "test".into(), dtype, shape,
            byte_offset: 0, byte_length: bytes.len() as u64, role: TensorRole::Other, layer_id: None,
            expert_id: None, is_quantization_metadata: false }, source_path: "test".into(), bytes,
            elapsed_micros: 0, sha256: String::new() })
    }

    #[test]
    fn global_native_extents_follow_entry_and_terminal_without_extra_copies() -> Result<()> {
        let mut cfg = cfg();
        cfg.vocab_size = 8;
        let pair = GlobalLayout::pair(&cfg)?;
        assert_eq!(pair[0].operands, BTreeMap::from([("embedding", 64)]));
        assert_eq!(pair[1].operands, BTreeMap::from([("head", 64), ("output.score", 16), ("output.norm", 8)]));
        let mut names = Vec::new();
        for (rank, layout) in pair.iter().enumerate() {
            let packed = layout.pack(&cfg, |name| {
                names.push(name.to_string());
                let mut value = tensor(name)?;
                if name.ends_with("embed_tokens.weight") || name.ends_with("lm_head.weight") {
                    value.info.shape = vec![8, 4];
                    value.bytes = std::iter::repeat_n(0x3f80u16, 32).flat_map(u16::to_le_bytes).collect();
                    value.info.byte_length = value.bytes.len() as u64;
                }
                Ok(value)
            })?;
            layout.check(&packed)?;
            assert_eq!(layout.bytes()?, packed.operands.values().map(Vec::len).sum::<usize>());
            let demand = layout.demand()?;
            assert_eq!(demand.gpu, rank as u8);
            assert_eq!(demand.category, Category::Weights);
            assert_eq!(demand.basis, Basis::Exact);
            assert_eq!(demand.bytes, layout.bytes()? as u64);
            if rank == 1 {
                let scores = packed.operands["output.score"].chunks_exact(4)
                    .map(|bytes| f32::from_le_bytes(bytes.try_into().unwrap())).collect::<Vec<_>>();
                assert_eq!(scores, vec![1.; 4]);
            }
            assert!(pair[1 - rank].check(&packed).is_err());
        }
        assert_eq!(names, ["language_model.model.embed_tokens.weight", "language_model.lm_head.weight",
            "language_model.model.output_attn_res_norm.weight", "language_model.model.output_attn_res_proj.weight",
            "language_model.model.norm.weight"]);
        for invalid in 0..3 {
            let mut broken = cfg.clone();
            match invalid { 0 => broken.tie_embeddings = true, 1 => broken.hidden = usize::MAX, _ => broken.vocab_size = 0 }
            assert!(GlobalLayout::pair(&broken).is_err());
        }
        Ok(())
    }

    #[test]
    fn global_packing_rejects_bad_native_operands_even_with_equal_total_bytes() -> Result<()> {
        let mut cfg = cfg();
        cfg.vocab_size = 1;
        let pair = GlobalLayout::pair(&cfg)?;
        for case in 0..3 {
            assert!(pair[0].pack(&cfg, |name| {
                let mut value = tensor(name)?;
                value.info.shape = vec![1, 4];
                match case { 0 => value.info.dtype = DType::F32, 1 => value.info.shape = vec![4, 1], _ => { value.bytes.pop(); } }
                Ok(value)
            }).is_err());
        }
        let mut packed = pair[1].pack(&cfg, |name| {
            let mut value = tensor(name)?;
            if name.ends_with("lm_head.weight") { value.info.shape = vec![1, 4]; }
            Ok(value)
        })?;
        packed.operands.get_mut("head").unwrap().pop();
        packed.operands.get_mut("output.norm").unwrap().push(0);
        assert_eq!(pair[1].bytes()?, packed.operands.values().map(Vec::len).sum::<usize>());
        assert!(pair[1].check(&packed).is_err());
        Ok(())
    }

    #[test]
    fn owned_mla_omits_nonowner_attention_but_retains_tp_ffn_and_replicated_moe() -> Result<()> {
        let cfg = cfg();
        for owner in 0..2 {
            let pair = PackedLayer::pair(&cfg, 3, LayerMode::Whole { gpu: owner, ffn: FfnMode::Split }, tensor)?;
            let full = &pair[usize::from(owner)];
            let absent = &pair[usize::from(1 - owner)];
            assert_eq!(full.attention, AttentionShare::Whole);
            assert_eq!(absent.attention, AttentionShare::Absent);
            assert!(!absent.operands.keys().any(|key| key.starts_with("mla.")));
            assert_eq!(full.operands["mla.q_b"].len(), 12 * 3 * 2);
            assert_eq!(full.operands["mla.uk"].len(), 4 * 2 * 2 * 2);
            for key in ["ffn.gate_up", "ffn.down", "attention.score", "ffn.norm", "router.weight", "router.bias",
                "latent.down", "latent.norm", "latent.up"] {
                assert_eq!(full.operands[key], absent.operands[key]);
            }
            assert_eq!(absent.operands["attention.score"].len(), 4 * 4);
            assert_eq!(absent.operands["ffn.gate_up"].len(), 2 * 4 * 4 * 2);
            assert_eq!(full.bytes()?, full.operands.values().map(Vec::len).sum::<usize>());
        }
        Ok(())
    }

    #[test]
    fn kda_native_f32_operands_and_column_output_use_shared_tp2_packer() -> Result<()> {
        let mut cfg = cfg();
        cfg.hidden = 128; cfg.kda_heads = 16; cfg.kda_head_dim = 128;
        let width = cfg.kda_heads * cfg.kda_head_dim;
        let pair = PackedLayer::pair(&cfg, 0, LayerMode::HeadSplit, |name| {
            let (dtype, shape) = if let Some(leaf) = name.split("self_attn.").nth(1) {
                match leaf {
                    "q_proj.weight" | "k_proj.weight" | "v_proj.weight" | "g_proj.weight" =>
                        (DType::Bf16, vec![width, cfg.hidden]),
                    "f_a_proj.weight" => (DType::Bf16, vec![128, cfg.hidden]),
                    "f_b_proj.weight" => (DType::Bf16, vec![width, 128]),
                    "b_proj.weight" => (DType::Bf16, vec![cfg.kda_heads, cfg.hidden]),
                    "q_conv1d.weight" | "k_conv1d.weight" | "v_conv1d.weight" =>
                        (DType::F32, vec![width, 1, 4]),
                    "A_log" => (DType::F32, vec![cfg.kda_heads]),
                    "dt_bias" => (DType::F32, vec![width]),
                    "o_norm.weight" => (DType::F32, vec![128]),
                    "o_proj.weight" => (DType::Bf16, vec![cfg.hidden, width]),
                    _ => anyhow::bail!("unexpected KDA tensor {name}"),
                }
            } else if name.ends_with("down_proj.weight") { (DType::Bf16, vec![cfg.hidden, 8]) }
            else if name.ends_with("gate_proj.weight") || name.ends_with("up_proj.weight") {
                (DType::Bf16, vec![8, cfg.hidden])
            } else if name.ends_with("_res_proj.weight") { (DType::Bf16, vec![1, cfg.hidden]) }
            else { (DType::Bf16, vec![cfg.hidden]) };
            let mut value = tensor("test.weight")?;
            value.info.name = name.into();
            value.info.dtype = dtype.clone();
            value.info.shape = shape.clone();
            value.bytes = vec![0; shape.iter().product::<usize>() * if dtype == DType::F32 { 4 } else { 2 }];
            value.info.byte_length = value.bytes.len() as u64;
            Ok(value)
        })?;
        for rank in pair {
            assert_eq!(rank.attention, AttentionShare::Split);
            assert_eq!(rank.operands["kda.norm"].len(), 128 * 4);
            assert_eq!(rank.operands["kda.a_log"].len(), 8 * 4);
            assert_eq!(rank.operands["kda.dt_bias"].len(), width / 2 * 4);
            assert_eq!(rank.operands["kda.output"].len(), cfg.hidden * width / 2 * 2);
            assert!(!rank.operands.keys().any(|key| key.starts_with("latent.") || key.starts_with("router.")));
        }
        Ok(())
    }

    #[test]
    fn admitted_extents_match_packing_and_reject_equal_total_operand_swaps() -> Result<()> {
        let cfg = cfg();
        let mode = LayerMode::Whole { gpu: 1, ffn: FfnMode::Split };
        let layouts = LayerLayout::pair(&cfg, 3, mode)?;
        let mut packed = PackedLayer::pair(&cfg, 3, mode, tensor)?;
        for (layout, packed) in layouts.iter().zip(&packed) {
            assert_eq!(layout.bytes()?, packed.bytes()?);
            layout.check(packed)?;
        }
        let before = packed[0].bytes()?;
        packed[0].operands.get_mut("ffn.gate_up").unwrap().pop();
        packed[0].operands.get_mut("ffn.down").unwrap().push(0);
        assert_eq!(before, packed[0].bytes()?);
        assert!(layouts[0].check(&packed[0]).is_err());
        for change in 0..3 {
            let mut invalid = cfg.clone();
            match change {
                0 => invalid.hidden = usize::MAX,
                1 => invalid.shared_experts = usize::MAX,
                _ => invalid.moe_intermediate = 3,
            }
            if change == 2 { invalid.shared_experts = 1; }
            assert!(PackedLayer::pair(&invalid, 3, mode, |_| unreachable!()).is_err());
        }
        Ok(())
    }

    #[test]
    fn explicit_head_split_and_invalid_source_or_placement_fail_closed() -> Result<()> {
        let cfg = cfg();
        let pair = PackedLayer::pair(&cfg, 3, LayerMode::HeadSplit, tensor)?;
        assert!(pair.iter().all(|rank| rank.attention == AttentionShare::Split));
        assert_eq!(pair[0].operands["mla.q_b"].len(), 6 * 3 * 2);
        assert!(PackedLayer::pair(&cfg, 3, LayerMode::ContextSplit, |_| unreachable!()).is_err());
        assert!(PackedLayer::pair(&cfg, 3, LayerMode::Whole { gpu: 0, ffn: FfnMode::Owner },
            |_| unreachable!()).is_err());
        assert!(PackedLayer::pair(&cfg, 0, LayerMode::Whole { gpu: 0, ffn: FfnMode::Split },
            |_| unreachable!()).is_err());
        assert!(PackedLayer::pair(&cfg, cfg.layers, LayerMode::HeadSplit, |_| unreachable!()).is_err());
        assert!(PackedLayer::pair(&cfg, 3, LayerMode::HeadSplit, |name| {
            let mut value = tensor(name)?;
            if name.ends_with("e_score_correction_bias") { value.bytes.pop(); }
            Ok(value)
        }).is_err());
        Ok(())
    }
}
