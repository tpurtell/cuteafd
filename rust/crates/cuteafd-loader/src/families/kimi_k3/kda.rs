//! Kimi K3 tensor names onto the shared KDA packer's roles (`crate::kda`).
use cuteafd_core::DType;

use super::config::{KimiAttention, KimiK3Config};
use crate::kda::{KdaError, KdaGeometry, KdaOperand, KdaOutputGate, KdaRole, KdaTensor};

/// The shared KDA geometry of a Kimi K3 checkpoint: full-rank output gate,
/// F32 `o_norm` as shipped.
pub fn geometry(cfg: &KimiK3Config) -> KdaGeometry {
    KdaGeometry { hidden: cfg.hidden, heads: cfg.kda_heads, head_dim: cfg.kda_head_dim, conv_kernel: cfg.kda_conv,
        output_gate: KdaOutputGate::FullRank, o_norm_dtype: DType::F32 }
}

/// The checkpoint tensor holding `role` of KDA layer `layer` (0-based), or
/// `None` for roles Kimi K3 does not ship (the low-rank output gate).
pub fn tensor_name(layer: usize, role: KdaRole) -> Option<String> {
    let leaf = match role {
        KdaRole::Q => "q_proj.weight", KdaRole::K => "k_proj.weight", KdaRole::V => "v_proj.weight",
        KdaRole::Fa => "f_a_proj.weight", KdaRole::Fb => "f_b_proj.weight", KdaRole::G => "g_proj.weight",
        KdaRole::Beta => "b_proj.weight", KdaRole::ConvQ => "q_conv1d.weight", KdaRole::ConvK => "k_conv1d.weight",
        KdaRole::ConvV => "v_conv1d.weight", KdaRole::ALog => "A_log", KdaRole::DtBias => "dt_bias",
        KdaRole::ONorm => "o_norm.weight", KdaRole::O => "o_proj.weight",
        KdaRole::Ga | KdaRole::Gb => return None,
    };
    Some(format!("language_model.model.layers.{layer}.self_attn.{leaf}"))
}

/// Pack one KDA layer's `operand` for `ranks` coordinator GPUs. `read` returns
/// a checkpoint tensor's dtype, shape and bytes by name.
pub fn pack(cfg: &KimiK3Config, layer: usize, operand: KdaOperand, ranks: usize,
    mut read: impl FnMut(&str) -> anyhow::Result<KdaTensor>) -> Result<Vec<KdaTensor>, KdaError> {
    if cfg.attention.get(layer) != Some(&KimiAttention::Kda) {
        return Err(KdaError::Source(format!("Kimi K3 layer {layer} is not a KDA layer")));
    }
    geometry(cfg).pack(operand, ranks, |role| {
        let name = tensor_name(layer, role)
            .ok_or_else(|| KdaError::Source(format!("Kimi K3 ships no {role:?} (full-rank output gate)")))?;
        read(&name).map_err(|error| KdaError::Source(format!("{name}: {error:#}")))
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::families::kimi_k3::config::tests::release_config;

    #[test]
    fn release_layer_packs_from_kimi_names_with_native_dtypes() {
        let cfg = KimiK3Config::from_hf(&release_config()).unwrap();
        let (h, d) = (cfg.hidden, cfg.kda_heads * cfg.kda_head_dim);
        let read = |name: &str| -> anyhow::Result<KdaTensor> {
            let leaf = name.rsplit_once("self_attn.").unwrap().1;
            let (dtype, shape): (DType, Vec<usize>) = match leaf {
                "q_proj.weight" | "k_proj.weight" | "v_proj.weight" | "g_proj.weight" => (DType::Bf16, vec![d, h]),
                "f_a_proj.weight" => (DType::Bf16, vec![128, h]),
                "f_b_proj.weight" => (DType::Bf16, vec![d, 128]),
                "b_proj.weight" => (DType::Bf16, vec![96, h]),
                "q_conv1d.weight" | "k_conv1d.weight" | "v_conv1d.weight" => (DType::F32, vec![d, 1, 4]),
                "A_log" => (DType::F32, vec![128]),
                "dt_bias" => (DType::F32, vec![d]),
                "o_norm.weight" => (DType::F32, vec![128]),
                "o_proj.weight" => (DType::Bf16, vec![h, d]),
                other => anyhow::bail!("unexpected {other}"),
            };
            let element = if dtype == DType::F32 { 4 } else { 2 };
            let count: usize = shape.iter().product();
            let mut bytes = vec![0u8; count * element];
            if leaf == "A_log" {
                for (i, chunk) in bytes.chunks_exact_mut(4).take(96).enumerate() { chunk.copy_from_slice(&(i as f32 + 1.0).to_le_bytes()); }
            }
            Ok(KdaTensor { dtype, shape, bytes })
        };
        // Layer 0 is KDA; layer 3 (1-based 4) is MLA.
        let w_in = pack(&cfg, 0, KdaOperand::WIn, 2, read).unwrap();
        assert_eq!(w_in[0].shape, vec![3 * 6144 + 128 + 6144 + 48, h]);
        let a_log = pack(&cfg, 0, KdaOperand::ALog, 2, read).unwrap();
        assert_eq!(a_log[1].bytes[..4], 49.0f32.to_le_bytes());
        let o_norm = pack(&cfg, 0, KdaOperand::ONorm, 2, read).unwrap();
        assert_eq!(o_norm[0].dtype, DType::F32);
        assert!(pack(&cfg, 3, KdaOperand::WIn, 2, read).unwrap_err().to_string().contains("not a KDA layer"));
    }
}
