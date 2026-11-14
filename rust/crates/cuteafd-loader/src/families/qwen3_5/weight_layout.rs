//! Checkpoint-native dense operands and one immutable arena per weight scope.
use super::Qwen35Config;
use crate::{families::qwen::HybridAttention, plan::checkpoint::Checkpoint, serving_capacity::WeightArenaLayout};
use anyhow::{ensure, Context, Result};
use cuteafd_core::DType;

pub const PREFIX: &str = "model.language_model.";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Storage { Bf16, F32, NativeNorm, Fp8, Scales }

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DenseFormat { Bf16, Fp8 }

impl DenseFormat {
    pub fn detect(checkpoint: &Checkpoint) -> Result<Self> {
        let name = format!("{PREFIX}layers.0.mlp.gate_proj.weight");
        let tensor = checkpoint.tensors.iter().find(|t| t.meta.name == name).context("missing first dense gate")?;
        match tensor.meta.dtype {
            DType::Bf16 => Ok(Self::Bf16),
            DType::F8E4M3 => Ok(Self::Fp8),
            ref dtype => anyhow::bail!("unsupported dense projection format {dtype:?}"),
        }
    }
    pub fn fp8(self) -> bool { self == Self::Fp8 }
}

#[derive(Debug, PartialEq, Eq)]
pub struct Operand {
    pub name: &'static str,
    pub tensors: Vec<String>,
    pub storage: Storage,
}

pub fn layer_operands(prefix: &str, attention: HybridAttention) -> Vec<Operand> {
    let mut out = Vec::new();
    let mut add = |name, tensors: &[&str], storage| out.push(Operand { name,
        tensors: tensors.iter().map(|t| format!("{prefix}.{t}")).collect(), storage });
    for (operand, tensor) in [("attn.norm", "input_layernorm.weight"), ("ffn.norm", "post_attention_layernorm.weight"),
        ("ffn.w_down", "mlp.down_proj.weight")] {
        add(operand, &[tensor], Storage::Bf16);
    }
    add("ffn.w_gate_up", &["mlp.gate_proj.weight", "mlp.up_proj.weight"], Storage::Bf16);
    match attention {
        HybridAttention::Gdn => {
            add("w_in", &["linear_attn.in_proj_qkv.weight", "linear_attn.in_proj_z.weight",
                "linear_attn.in_proj_b.weight", "linear_attn.in_proj_a.weight"], Storage::Bf16);
            for (operand, tensor) in [("conv_w", "conv1d.weight"), ("a_log", "A_log"), ("dt_bias", "dt_bias")] {
                add(operand, &[&format!("linear_attn.{tensor}")], Storage::F32);
            }
            add("norm_w", &["linear_attn.norm.weight"], Storage::NativeNorm);
            add("w_out", &["linear_attn.out_proj.weight"], Storage::Bf16);
        }
        HybridAttention::Full => {
            add("w_in", &["self_attn.q_proj.weight", "self_attn.k_proj.weight", "self_attn.v_proj.weight"], Storage::Bf16);
            for (operand, tensor) in [("q_norm", "q_norm.weight"), ("k_norm", "k_norm.weight"), ("w_o", "o_proj.weight")] {
                add(operand, &[&format!("self_attn.{tensor}")], Storage::Bf16);
            }
        }
    }
    out
}

pub fn native_operands(prefix: &str, attention: HybridAttention, format: DenseFormat) -> Vec<Operand> {
    let specs = layer_operands(prefix, attention);
    if !format.fp8() { return specs; }
    let mut out = Vec::new();
    for mut spec in specs {
        let projection = matches!(spec.name, "w_in" | "w_out" | "w_o" | "ffn.w_gate_up" | "ffn.w_down");
        if !projection { out.push(spec); continue; }
        if attention == HybridAttention::Gdn && spec.name == "w_in" {
            let gates = spec.tensors.split_off(2);
            out.push(Operand { name: "w_ba", tensors: gates, storage: Storage::Bf16 });
        }
        let (weight, scales) = match spec.name {
            "w_in" if attention == HybridAttention::Gdn => ("w_qkvz_fp8", "w_qkvz_scale"),
            "w_in" => ("w_in_fp8", "w_in_scale"),
            "w_out" => ("w_out_fp8", "w_out_scale"),
            "w_o" => ("w_o_fp8", "w_o_scale"),
            "ffn.w_gate_up" => ("ffn.w_gate_up_fp8", "ffn.w_gate_up_scale"),
            "ffn.w_down" => ("ffn.w_down_fp8", "ffn.w_down_scale"),
            _ => unreachable!(),
        };
        out.push(Operand { name: scales, tensors: spec.tensors.iter()
            .map(|name| format!("{}_scale_inv", name)).collect(), storage: Storage::Scales });
        out.push(Operand { name: weight, tensors: spec.tensors, storage: Storage::Fp8 });
    }
    out
}

pub fn validate_native_matrices<'t>(tensors: impl IntoIterator<Item = (&'t DType, &'t [usize], u64)>,
    storage: Storage) -> Result<()> {
    let mut cols = None;
    for (dtype, shape, bytes) in tensors {
        ensure!(shape.len() == 2 && shape[0] > 0 && shape[1] > 0, "native projection requires nonempty matrices");
        ensure!(cols.is_none_or(|previous| previous == shape[1]), "native packed matrix columns differ");
        cols = Some(shape[1]);
        let width = match (storage, dtype) {
            (Storage::Fp8, DType::F8E4M3) => 1,
            (Storage::Scales, DType::Bf16) => 2,
            (Storage::Scales, DType::F32) => 4,
            _ => anyhow::bail!("native matrix dtype disagrees with operand storage"),
        };
        let size = shape[0].checked_mul(shape[1]).and_then(|n| n.checked_mul(width))
            .context("native matrix byte size overflow")?;
        ensure!(u64::try_from(size)? == bytes, "native matrix payload disagrees with shape");
    }
    ensure!(cols.is_some(), "empty native matrix packing");
    Ok(())
}

fn operand_bytes(checkpoint: &Checkpoint, spec: &Operand) -> Result<usize> {
    let mut headers = Vec::new();
    let mut bytes = 0usize;
    for name in &spec.tensors {
        let at = checkpoint.tensors.binary_search_by(|t| t.meta.name.as_str().cmp(name))
            .map_err(|_| anyhow::anyhow!("checkpoint has no tensor {name}"))?;
        let tensor = &checkpoint.tensors[at].meta;
        ensure!(match spec.storage {
            Storage::Bf16 => tensor.dtype == DType::Bf16,
            Storage::F32 | Storage::Scales => matches!(tensor.dtype, DType::Bf16 | DType::F32),
            Storage::Fp8 => tensor.dtype == DType::F8E4M3,
            Storage::NativeNorm => matches!(tensor.dtype, DType::Bf16 | DType::F32),
        }, "{name}: native dense operand dtype differs from admitted projection format");
        let widen = matches!(spec.storage, Storage::F32 | Storage::Scales) && tensor.dtype == DType::Bf16;
        let size = usize::try_from(tensor.byte_length)?.checked_mul(if widen { 2 } else { 1 })
            .context("dense Qwen operand size overflow")?;
        bytes = bytes.checked_add(size).context("dense Qwen packed operand overflow")?;
        headers.push(tensor);
    }
    if matches!(spec.storage, Storage::Fp8 | Storage::Scales) {
        validate_native_matrices(headers.iter().map(|t| (&t.dtype, t.shape.as_slice(), t.byte_length)), spec.storage)?;
    }
    ensure!(bytes > 0, "empty dense Qwen operand");
    Ok(bytes.max(256))
}

pub fn layer_requests(checkpoint: &Checkpoint, prefix: &str, attention: HybridAttention, format: DenseFormat)
    -> Result<Vec<(String, usize)>> {
    native_operands(prefix, attention, format).into_iter().map(|spec| {
        let bytes = operand_bytes(checkpoint, &spec)?;
        Ok((format!("{prefix}/{}", spec.name), bytes))
    }).collect()
}

/// Plan final operands, including separate embedding/head owners even when tied.
pub fn target_arena(checkpoint: &Checkpoint, cfg: &Qwen35Config, norm_dtype: DType,
    format: DenseFormat) -> Result<WeightArenaLayout> {
    let admitted = crate::placement::families::qwen3_5::NativeWeights::native(checkpoint, cfg, norm_dtype)?;
    ensure!(DenseFormat::detect(checkpoint)? == format, "dense Qwen format differs from checkpoint");
    let mut requests = Vec::new();
    for (index, &attention) in cfg.attention.iter().enumerate() {
        let layer = layer_requests(checkpoint, &format!("{PREFIX}layers.{index}"), attention, format)?;
        ensure!(layer.iter().map(|(_, bytes)| *bytes as u64).sum::<u64>() == admitted.layers[index],
            "dense Qwen layer layout differs from admitted payload");
        requests.extend(layer);
    }
    requests.extend([
        (format!("{PREFIX}norm.weight"), usize::try_from(admitted.norm)?),
        ("head.weight".into(), usize::try_from(admitted.head)?),
        ("embedding.table".into(), usize::try_from(admitted.embedding)?),
    ]);
    WeightArenaLayout::new("qwen3_5/target/gpu0", requests)
}

/// Whole layers retain their native operands; globals live only on rank zero.
pub fn placed_arenas(target: &WeightArenaLayout, owners: &[u8]) -> Result<[WeightArenaLayout; 2]> {
    ensure!(!owners.is_empty() && owners.iter().all(|&rank| rank < 2), "invalid dense layer owners");
    let mut requests = [Vec::new(), Vec::new()];
    let mut views = target.views().iter().collect::<Vec<_>>();
    views.sort_by_key(|(_, view)| view.offset());
    for (name, view) in views {
        let rank = if let Some(rest) = name.strip_prefix(&format!("{PREFIX}layers.")) {
            let (index, _) = rest.split_once('/').context("dense layer arena view")?;
            usize::from(*owners.get(index.parse::<usize>()?).context("dense arena layer owner")?)
        } else { 0 };
        requests[rank].push((name.clone(), view.bytes()));
    }
    let [first, second] = requests;
    Ok([WeightArenaLayout::new("qwen3_5/target/gpu0", first)?,
        WeightArenaLayout::new("qwen3_5/target/gpu1", second)?])
}

/// Native MTP shares the target's embedding and head, not its weight ownership.
pub fn mtp_arena(checkpoint: &Checkpoint, cfg: &Qwen35Config, format: DenseFormat)
    -> Result<WeightArenaLayout> {
    ensure!(cfg.mtp_layers == 1, "dense MTP requires one checkpoint-native layer");
    let mut requests = layer_requests(checkpoint, "mtp.layers.0", HybridAttention::Full, format)?;
    for (name, shape) in [("fc", vec![cfg.hidden, 2 * cfg.hidden]), ("norm", vec![cfg.hidden]),
        ("pre_fc_norm_hidden", vec![cfg.hidden]), ("pre_fc_norm_embedding", vec![cfg.hidden])] {
        let name = format!("mtp.{name}.weight");
        let at = checkpoint.tensors.binary_search_by(|t| t.meta.name.cmp(&name))
            .map_err(|_| anyhow::anyhow!("checkpoint has no tensor {name}"))?;
        let tensor = &checkpoint.tensors[at].meta;
        ensure!(tensor.dtype == DType::Bf16 && tensor.shape == shape, "{name}: native MTP shape/dtype differs");
        requests.push((name, usize::try_from(tensor.byte_length)?.max(256)));
    }
    let layout = WeightArenaLayout::new("qwen3_5/mtp/gpu0", requests)?;
    let admitted = crate::placement::families::qwen3_5::DrafterBytes::native_mtp_payload(checkpoint, cfg)?;
    ensure!(layout.payload_bytes() as u64 == admitted.weights.0, "dense MTP layout differs from admitted payload");
    Ok(layout)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn planner_and_runtime_share_the_installed_operand_recipe() -> Result<()> {
        let (directory, _, _, _) = super::super::tests::native_planner_fixture();
        let checkpoint = Checkpoint::open(directory.path())?;
        let cfg = Qwen35Config::from_hf(&checkpoint.config)?;
        let format = DenseFormat::detect(&checkpoint)?;
        let layout = target_arena(&checkpoint, &cfg, DType::F32, format)?;
        let payload = crate::placement::families::qwen3_5::NativeWeights::native(&checkpoint, &cfg, DType::F32)?;
        assert_eq!(layout.payload_bytes() as u64,
            payload.layers.iter().sum::<u64>() + payload.norm + payload.head + payload.embedding);
        assert_eq!(layout.views().len(), 3 * 10 + 8 + 3);
        assert_eq!(layout.view("model.language_model.layers.0/norm_w").unwrap().bytes(), 512);
        assert!(mtp_arena(&checkpoint, &cfg, format).is_err());
        Ok(())
    }
}
