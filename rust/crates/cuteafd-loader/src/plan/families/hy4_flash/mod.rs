//! Hy4-flash (`HYV4FlashForConditionalGeneration`, `hy_v4_flash`):
//! 42-layer hybrid MLA (12 compressed-indexer DSA + 30 SWA-512) with 4-stream
//! identity hyper-connections (iHC), 4-layer host-mapped FP8 Engram
//! (`engram-fp8-layer-*-part-*.safetensors`), LatentMoE (41 routed layers ×
//! 512 EXL3 K4 MUL1 experts, top-16, 1536 latent width, 1 shared expert,
//! SwiGLU limit 10.0), optional 5-layer DFly2 drafter and 27-layer ViT (#2478).
//!
//! Validates strictly from the checkpoint's own `config.json`, safetensors
//! tensor headers, and `cuteafd-mc.json` view / `engram_fp8` declarations.
use std::collections::BTreeMap;

use anyhow::{ensure, Result};

use crate::families::hy4_flash::weight_layout::{core_contract, HEAD_STEM};
use crate::families::hy4_flash::{Hy4DflyConfig, Hy4FlashConfig, Hy4VisionConfig};
use crate::formats::trellis_core::CoreProjection;
use crate::plan::{
    checkpoint::Checkpoint,
    family::*,
    format::{Encoding, QuantOperand},
    names::indexed,
    spec::*,
};

mod placement;
#[cfg(test)]
pub(crate) mod tests;

pub struct Hy4Flash;
pub static HY4_FLASH: Hy4Flash = Hy4Flash;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Storage {
    Bf16,
    F32,
    I32,
    /// Host-mapped Engram table (`engram-fp8-layer-*-part-*.safetensors`
    /// declared in `cuteafd-mc.json`, 100-byte strided E4M3 + F32 scale rows).
    MappedEngram,
}

#[derive(Debug, Clone)]
pub(crate) struct Contract {
    pub role: TensorRole,
    pub shape: Vec<usize>,
    pub storage: Storage,
    pub companions: Vec<&'static str>,
    pub bare: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ExpertStorage {
    Exl3 { bits: usize, mul1: bool },
    Bf16,
}

pub struct Model {
    pub cfg: Hy4FlashConfig,
    spec: ModelSpec,
    contracts: BTreeMap<String, Contract>,
    pub core: BTreeMap<String, CoreProjection>,
    pub(crate) routed_bytes: u64,
    pub(crate) expert_storage: BTreeMap<usize, ExpertStorage>,
}

impl Model {
    pub fn open(checkpoint: &Checkpoint) -> Result<Self> {
        let cfg = Hy4FlashConfig::from_hf(&checkpoint.config)?;
        cfg.release_geometry()?;
        let dfly = Hy4DflyConfig::from_hf(&checkpoint.config, &cfg)?;
        let vision = Hy4VisionConfig::from_hf(&checkpoint.config)?;
        let (h, v, hc) = (cfg.hidden, cfg.vocab_size, cfg.hc_mult);

        let mut contracts = BTreeMap::new();
        let mut add = |stem: String, role: TensorRole, shape: Vec<usize>, storage: Storage, companions: Vec<&'static str>, bare: bool| {
            contracts.insert(stem, Contract { role, shape, storage, companions, bare });
        };

        // Globals
        add("model.embed_tokens".into(), TensorRole::new(Component::Embedding), vec![v, h], Storage::Bf16, vec![], false);
        add(HEAD_STEM.into(), TensorRole::new(Component::LmHead), vec![v, h], Storage::Bf16, vec![], false);
        add("model.norm".into(), TensorRole::new(Component::Norm), vec![h], Storage::Bf16, vec![], false);
        add("model.hc_head.hc_head_fn".into(), TensorRole::new(Component::HyperConnection), vec![hc, hc * h], Storage::F32, vec![], true);
        add("model.hc_head.hc_head_base".into(), TensorRole::new(Component::HyperConnection), vec![hc], Storage::F32, vec![], true);
        add("model.hc_head.hc_head_scale".into(), TensorRole::new(Component::HyperConnection), vec![1], Storage::F32, vec![], true);

        for layer in 0..cfg.layers {
            let base = format!("model.layers.{layer}");
            let at = |c| TensorRole::layer(c, layer);
            let attn = cfg.attention(layer);

            // Layer norms + 4-stream iHC
            add(format!("{base}.input_layernorm"), at(Component::Norm), vec![h], Storage::Bf16, vec![], false);
            add(format!("{base}.post_attention_layernorm"), at(Component::Norm), vec![h], Storage::Bf16, vec![], false);
            for tag in ["hc_attn_layer", "hc_mlp_layer"] {
                let p = format!("{base}.{tag}.hc_pre");
                add(format!("{p}.hc_fn"), at(Component::HyperConnection), vec![2 * hc, hc * h], Storage::F32, vec![], true);
                add(format!("{p}.hc_base"), at(Component::HyperConnection), vec![2 * hc], Storage::F32, vec![], true);
                add(format!("{p}.hc_scale"), at(Component::HyperConnection), vec![2], Storage::F32, vec![], true);
            }

            // Attention
            let a = format!("{base}.self_attn");
            add(format!("{a}.q_a_proj"), at(Component::Attention), vec![cfg.q_lora_rank, h], Storage::Bf16, vec![], false);
            add(format!("{a}.q_a_layernorm"), at(Component::Attention), vec![cfg.q_lora_rank], Storage::Bf16, vec![], false);
            add(format!("{a}.q_b_proj"), at(Component::Attention), vec![attn.heads * attn.qk_head_dim(), cfg.q_lora_rank], Storage::Bf16, vec![], false);
            add(format!("{a}.kv_a_proj_with_mqa"), at(Component::Attention), vec![cfg.kv_lora_rank + attn.qk_rope_dim, h], Storage::Bf16, vec![], false);
            add(format!("{a}.kv_a_layernorm"), at(Component::Attention), vec![cfg.kv_lora_rank], Storage::Bf16, vec![], false);
            add(format!("{a}.kv_b_proj"), at(Component::Attention), vec![attn.heads * (attn.qk_nope_dim + attn.v_head_dim), cfg.kv_lora_rank], Storage::Bf16, vec![], false);
            add(format!("{a}.linear_gate"), at(Component::Attention), vec![attn.heads * attn.v_head_dim, h], Storage::Bf16, vec![], false);
            add(format!("{a}.o_proj"), at(Component::Attention), vec![h, attn.heads * attn.v_head_dim], Storage::Bf16, vec![], false);
            add(format!("{a}.post_proj_norm"), at(Component::Attention), vec![h], Storage::Bf16, vec![], false);
            add(format!("{a}.learnable_sink_param"), at(Component::Attention), vec![attn.heads], Storage::F32, vec![], true);

            // Compressed indexer (DSA layers)
            if attn.sparse {
                let d = cfg.indexer_head_dim;
                let n = cfg.indexer_heads;
                let ip = format!("{a}.compressed_indexer");
                add(format!("{ip}.query_projection"), at(Component::Indexer), vec![n * d, cfg.q_lora_rank], Storage::Bf16, vec![], false);
                add(format!("{ip}.head_weight_projection"), at(Component::Indexer), vec![n, h], Storage::Bf16, vec![], false);
                add(format!("{ip}.token_key_projection"), at(Component::Indexer), vec![d, h], Storage::Bf16, vec![], false);
                add(format!("{ip}.token_key_norm"), at(Component::Indexer), vec![d], Storage::Bf16, vec![], false);
                add(format!("{ip}.key_norm"), at(Component::Indexer), vec![d], Storage::F32, vec![], false);
                add(format!("{ip}.key_value_projection"), at(Component::Indexer), vec![d, d], Storage::F32, vec![], false);
                add(format!("{ip}.key_gate_projection"), at(Component::Indexer), vec![d, d], Storage::F32, vec![], false);
                add(format!("{ip}.intra_group_ape"), at(Component::Indexer), vec![cfg.indexer_compress_ratio, d], Storage::F32, vec![], true);
            }

            // FFN: dense vs LatentMoE
            if cfg.dense_layer[layer] {
                add(format!("{base}.mlp.gate_proj"), at(Component::DenseFfn), vec![cfg.intermediate, h], Storage::Bf16, vec![], false);
                add(format!("{base}.mlp.up_proj"), at(Component::DenseFfn), vec![cfg.intermediate, h], Storage::Bf16, vec![], false);
                add(format!("{base}.mlp.down_proj"), at(Component::DenseFfn), vec![h, cfg.intermediate], Storage::Bf16, vec![], false);
                add(format!("{base}.mlp.post_fc2_norm"), at(Component::DenseFfn), vec![h], Storage::Bf16, vec![], false);
            } else {
                let shared = cfg.moe_intermediate * cfg.shared_experts;
                add(format!("{base}.mlp.gate"), at(Component::Router), vec![cfg.experts, h], Storage::Bf16, vec![], false);
                add(format!("{base}.mlp.gate.e_score_correction_bias"), at(Component::Router), vec![cfg.experts], Storage::F32, vec![], true);
                add(format!("{base}.mlp.routed_expert_down_proj"), at(Component::LatentProjection), vec![cfg.routed_expert_hidden, h], Storage::Bf16, vec![], false);
                add(format!("{base}.mlp.routed_expert_up_proj"), at(Component::LatentProjection), vec![h, cfg.routed_expert_hidden], Storage::Bf16, vec![], false);
                add(format!("{base}.mlp.combine_output_norm"), at(Component::LatentProjection), vec![h], Storage::Bf16, vec![], false);
                add(format!("{base}.mlp.shared_experts.gate_proj"), at(Component::SharedExpert), vec![shared, h], Storage::Bf16, vec![], false);
                add(format!("{base}.mlp.shared_experts.up_proj"), at(Component::SharedExpert), vec![shared, h], Storage::Bf16, vec![], false);
                add(format!("{base}.mlp.shared_experts.down_proj"), at(Component::SharedExpert), vec![h, shared], Storage::Bf16, vec![], false);
            }

            // Engram layers (1, 11, 21, 31)
            if let Some(rows) = cfg.engram_rows(layer) {
                let ep = format!("{base}.engram_layer");
                let w = cfg.engram_head_width();
                let concat = cfg.engram_concat_width();
                add(
                    format!("{ep}.multi_head_embedding.embedding"),
                    at(Component::MappedTable),
                    vec![usize::try_from(rows)?, w],
                    Storage::MappedEngram,
                    vec![],
                    false,
                );
                add(format!("{ep}.lookup_table"), at(Component::TableProjection), vec![cfg.engram_lookup_table_size], Storage::I32, vec![], true);
                add(format!("{ep}.unified_pre_gamma"), at(Component::TableProjection), vec![h], Storage::Bf16, vec![], true);
                add(format!("{ep}.value_proj"), at(Component::TableProjection), vec![h, concat], Storage::Bf16, vec!["bias"], false);
                for j in 0..cfg.engram_total_heads() {
                    add(format!("{ep}.lookup_per_head_norms.{j}"), at(Component::TableProjection), vec![w], Storage::Bf16, vec![], false);
                }
                for j in 0..hc {
                    add(format!("{ep}.key_projs.{j}"), at(Component::TableProjection), vec![h, concat], Storage::Bf16, vec!["bias"], false);
                    add(format!("{ep}.norm_key.{j}"), at(Component::TableProjection), vec![h], Storage::Bf16, vec![], false);
                    add(format!("{ep}.norm_query.{j}"), at(Component::TableProjection), vec![h], Storage::Bf16, vec![], false);
                    add(format!("{ep}.unified_per_hc_scales.{j}"), at(Component::TableProjection), vec![h], Storage::Bf16, vec![], true);
                }
            }
        }
        drop(add);

        let routed_bytes = checkpoint.tensors.iter()
            .filter(|t| t.meta.name.contains(".mlp.experts."))
            .map(|t| t.meta.byte_length)
            .sum();

        let mut expert_storage = BTreeMap::new();
        for layer in cfg.routed_layers() {
            expert_storage.insert(layer, detect_expert_storage(checkpoint, layer)?);
        }

        // Read EXL3 compact core projections (K8/K6 MUL1) on target + DFly2 layers.
        let mut expected: BTreeMap<String, Vec<usize>> = contracts.iter()
            .filter(|(_, c)| c.shape.len() == 2 && matches!(c.storage, Storage::Bf16) && !c.bare)
            .map(|(stem, c)| (format!("{stem}.weight"), c.shape.clone()))
            .collect();
        if let Some(dfly_cfg) = &dfly {
            let attn = cfg.attention(0);
            let dfly_mlp = 9216usize;
            for l in 0..dfly_cfg.layers {
                let p = format!("model.dfly_layers.{l}");
                for (sub, shape) in [
                    ("self_attn.q_a_proj", vec![cfg.q_lora_rank, h]),
                    ("self_attn.q_b_proj", vec![attn.heads * attn.qk_head_dim(), cfg.q_lora_rank]),
                    ("self_attn.kv_a_proj_with_mqa", vec![cfg.kv_lora_rank + attn.qk_rope_dim, h]),
                    ("self_attn.kv_b_proj", vec![attn.heads * (attn.qk_nope_dim + attn.v_head_dim), cfg.kv_lora_rank]),
                    ("self_attn.linear_gate", vec![attn.heads * attn.v_head_dim, h]),
                    ("self_attn.o_proj", vec![h, attn.heads * attn.v_head_dim]),
                    ("mlp.gate_proj", vec![dfly_mlp, h]),
                    ("mlp.up_proj", vec![dfly_mlp, h]),
                    ("mlp.down_proj", vec![h, dfly_mlp]),
                ] {
                    expected.insert(format!("{p}.{sub}.weight"), shape);
                }
            }
        }
        let core = crate::formats::trellis_core::read_projections(
            checkpoint,
            &expected,
            |name| !name.contains(".mlp.experts.") && !name.starts_with("vit."),
            core_contract,
        )?;

        // Structural presence check on every target contract member.
        for (stem, contract) in &contracts {
            if core.contains_key(stem) {
                continue;
            }
            let name = if contract.bare { stem.clone() } else { format!("{stem}.weight") };
            ensure!(
                checkpoint.contains_tensor(&name),
                "Hy4-flash checkpoint lacks {name} ({:?})",
                contract.role.component
            );
            for companion in &contract.companions {
                let comp = format!("{stem}.{companion}");
                ensure!(
                    checkpoint.contains_tensor(&comp),
                    "Hy4-flash checkpoint lacks {comp} ({:?})",
                    contract.role.component
                );
            }
        }

        let _ = crate::families::hy4_flash::engram::mapped_engram_bytes(checkpoint, &cfg)?;
        let spec = build_spec(&cfg, dfly.as_ref(), vision.is_some(), &expert_storage, checkpoint);

        Ok(Self {
            cfg,
            spec,
            contracts,
            core,
            routed_bytes,
            expert_storage,
        })
    }

    pub(crate) fn routed(&self) -> Vec<usize> {
        self.cfg.routed_layers()
    }
}

pub fn detect_expert_storage(checkpoint: &Checkpoint, layer: usize) -> Result<ExpertStorage> {
    let trellis_name = format!("model.layers.{layer}.mlp.experts.0.gate_proj.trellis");
    if let Some(header) = checkpoint.header(&trellis_name) {
        let bits = header.shape.get(2).map(|rows| rows / 16).filter(|bits| (1..=8).contains(bits))
            .ok_or_else(|| anyhow::anyhow!("{trellis_name}: EXL3 trellis shape {:?} encodes no width", header.shape))?;
        let mul1 = checkpoint.contains_tensor(&format!("model.layers.{layer}.mlp.experts.0.gate_proj.mul1"));
        return Ok(ExpertStorage::Exl3 { bits, mul1 });
    }
    let bf16_name = format!("model.layers.{layer}.mlp.experts.0.gate_proj.weight");
    if let Some(header) = checkpoint.header(&bf16_name) {
        ensure!(header.dtype == cuteafd_core::DType::Bf16,
            "{bf16_name}: unsupported Hy4-flash routed expert dtype {:?}", header.dtype);
        return Ok(ExpertStorage::Bf16);
    }
    anyhow::bail!("Hy4-flash checkpoint lacks routed expert headers at layer {layer} ({trellis_name} or {bf16_name})")
}

fn build_spec(
    cfg: &Hy4FlashConfig,
    dfly: Option<&Hy4DflyConfig>,
    vision: bool,
    expert_storage: &BTreeMap<usize, ExpertStorage>,
    checkpoint: &Checkpoint,
) -> ModelSpec {
    let layers = (0..cfg.layers).map(|layer| {
        let attn = cfg.attention(layer);
        LayerSpec {
            attention: if attn.sparse {
                AttentionKind::MlaDsa { indexer: true }
            } else {
                AttentionKind::Mla { rope: true, output_gate: true }
            },
            ffn: if cfg.dense_layer[layer] {
                FfnKind::Dense { intermediate: cfg.intermediate }
            } else {
                FfnKind::Moe
            },
            rope: Some(RopeSpec {
                dims: attn.qk_rope_dim,
                theta: if attn.sparse { cfg.sparse_rope_theta } else { cfg.swa_rope_theta },
            }),
        }
    }).collect();

    let mut formats = BTreeMap::<String, Vec<usize>>::new();
    for (&layer, &storage) in expert_storage {
        let label = match storage {
            ExpertStorage::Exl3 { bits, mul1: true } => format!("EXL3 K{bits} MUL1"),
            ExpertStorage::Exl3 { bits, mul1: false } => format!("EXL3 K{bits} MCG"),
            ExpertStorage::Bf16 => "BF16".to_string(),
        };
        formats.entry(label).or_default().push(layer);
    }

    let mut notes = vec![
        format!(
            "attention: hybrid MLA ({} DSA layers {:?} with {} heads + {}-head compressed indexer C{} top-{}/{}; \
             {} SWA-{} layers with {} heads; kv_lora_rank {}, qk {}/{}, v {}, learnable F32 sinks, sigmoid output gate, sandwich norm)",
            cfg.sparse_layers().len(), cfg.sparse_layers(), cfg.sparse_heads, cfg.indexer_heads,
            cfg.indexer_compress_ratio, cfg.indexer_group_topk, cfg.indexer_candidate_topk,
            cfg.swa_layers().len(), cfg.sliding_window, cfg.swa_heads,
            cfg.kv_lora_rank, cfg.qk_nope_dim, cfg.qk_rope_dim, cfg.v_head_dim
        ),
        format!(
            "residuals: {}-stream identity hyper-connections (iHC, eps {}, magnitude {}) + hc_head collapse before final RMSNorm",
            cfg.hc_mult, cfg.hc_eps, cfg.hc_magnitude
        ),
        format!(
            "LatentMoE: {} routed layers (layer 0 dense {}), {}->{}, {} experts top-{} (sigmoid + bias, scale {}), \
             {} shared expert ({}), SwiGLU clamp {}; storage: {}",
            cfg.routed_layers().len(), cfg.intermediate, cfg.hidden, cfg.routed_expert_hidden,
            cfg.experts, cfg.topk, cfg.routed_scaling_factor, cfg.shared_experts, cfg.moe_intermediate,
            cfg.swiglu_limit,
            formats.iter().map(|(label, ls)| format!("{label} ({} layers)", ls.len())).collect::<Vec<_>>().join("; ")
        ),
        format!(
            "Engram: 4 host-mapped FP8 tables at layers {:?} ({} heads x {} dims, {} total rows); \
             storage: engram_fp8 strided shards declared in cuteafd-mc.json (61.5 GiB, bitwise-exact to BF16 roundtrip)",
            cfg.engram_layers, cfg.engram_total_heads(), cfg.engram_head_width(),
            cfg.engram_layers.iter().filter_map(|&l| cfg.engram_rows(l)).sum::<u64>(),
        ),
    ];
    if let Some(d) = dfly {
        notes.push(format!(
            "bundled DFly2 drafter (optional, follow-up): {} layers (kv taps {:?}, index taps {:?}, top-{})",
            d.layers, d.target_kv_layers, d.target_index_layers, d.target_index_topk
        ));
    }

    ModelSpec {
        family: "hy4_flash",
        architecture: checkpoint.architectures().first().cloned().unwrap_or_default(),
        hidden: cfg.hidden,
        vocab: cfg.vocab_size,
        layers,
        moe: Some(MoeSpec {
            experts: cfg.experts,
            top_k: cfg.topk,
            intermediate: cfg.moe_intermediate,
            shared_experts: cfg.shared_experts,
            shared_intermediate: cfg.moe_intermediate * cfg.shared_experts,
            scoring: "sigmoid + bias".into(),
            routed_scaling: Some(cfg.routed_scaling_factor),
            groups: Some((cfg.n_group, cfg.topk_group)),
        }),
        speculator: dfly.map(|d| SpeculatorSpec::BundledDspark { layers: d.layers }),
        tables: vec![MappedTableSpec { name: "engram".into(), layers: cfg.engram_layers.clone() }],
        vision,
        notes,
    }
}

impl Family for Hy4Flash {
    fn id(&self) -> &'static str { "hy4_flash" }
    fn runtime(&self) -> RuntimeStatus { RuntimeStatus::Serving }
    fn attention_precision(&self) -> crate::plan::attention_precision::PrecisionSupport {
        crate::plan::attention_precision::MLA_BF16_ONLY
    }
    fn decode_lanes(&self) -> crate::plan::decode_lanes::LaneSupport {
        crate::plan::decode_lanes::LaneSupport::One {
            reason: "Hy4-flash serving starts on one decode lane; a second lane waits for a measured two-lane A/B (#2478)",
        }
    }
    fn decode_rows(&self) -> crate::plan::decode_rows::DecodeRowSupport {
        crate::plan::decode_rows::DecodeRowSupport::fixed(
            crate::placement::families::hy4_flash::DECODE_ROWS,
            7,
            "Hy4-flash split-K decode exports cover at most 64 rows (hy4f / hy4f2 MLA)",
        )
    }
    fn spark_decode(&self) -> Option<crate::plan::spark_executions::SparkDecode> {
        Some(crate::plan::spark_executions::SparkDecode { rows: 64 })
    }
    fn eager_native_core_headers(&self) -> bool { true }
    fn eager_expert_header(&self, name: &str) -> bool {
        let Some(tail) = name.strip_prefix("model.layers.") else { return false; };
        let Some((layer, tail)) = tail.split_once(".mlp.experts.") else { return false; };
        layer.parse::<usize>().is_ok()
            && matches!(
                tail,
                "0.gate_proj.trellis" | "0.gate_proj.mul1" | "0.gate_proj.weight"
                    | "0.up_proj.trellis" | "0.down_proj.trellis"
            )
    }
    fn fp8_native_mtp_layers(&self, config: &serde_json::Value) -> Option<(&'static str, std::ops::Range<usize>)> {
        let layers = config.get("dfly_num_hidden_layers").and_then(serde_json::Value::as_u64)? as usize;
        (layers > 0).then_some(("model.dfly_layers.", 0..layers))
    }
    fn detect(&self, checkpoint: &Checkpoint) -> bool {
        checkpoint.architectures().iter().any(|a| a == "HYV4FlashForConditionalGeneration")
            || checkpoint.model_type() == Some("hy_v4_flash")
    }
    fn open(&self, checkpoint: &Checkpoint) -> Result<Box<dyn FamilyModel>, ConfigError> {
        Ok(Box::new(Model::open(checkpoint).map_err(ConfigError::from_anyhow)?))
    }
    fn classify(&self, spec: &ModelSpec, name: &str) -> Option<TensorRole> {
        if name.starts_with("vit.") {
            return Some(TensorRole::new(Component::Vision));
        }
        if name.starts_with("model.dfly_layers.") {
            return Some(TensorRole::new(Component::Speculator));
        }
        if name.starts_with("lm_head.") {
            return Some(TensorRole::new(Component::LmHead));
        }
        let rest = name.strip_prefix("model.")?;
        if rest.starts_with("embed_tokens.") {
            return Some(TensorRole::new(Component::Embedding));
        }
        if rest.starts_with("norm.") {
            return Some(TensorRole::new(Component::Norm));
        }
        if rest.starts_with("hc_head.") {
            return Some(TensorRole::new(Component::HyperConnection));
        }
        let (layer, rest) = indexed(rest, "layers.")?;
        if layer >= spec.layers.len() {
            return None;
        }
        if let Some(expert) = rest.strip_prefix("mlp.experts.")
            .and_then(|r| r.split_once('.'))
            .and_then(|(e, _)| e.parse::<usize>().ok())
        {
            return Some(TensorRole::expert(Component::RoutedExpert, layer, expert));
        }
        let component = if rest.starts_with("self_attn.compressed_indexer.") {
            Component::Indexer
        } else if rest.starts_with("self_attn.") {
            Component::Attention
        } else if rest.starts_with("hc_attn_layer.") || rest.starts_with("hc_mlp_layer.") {
            Component::HyperConnection
        } else if rest.starts_with("input_layernorm.") || rest.starts_with("post_attention_layernorm.") {
            Component::Norm
        } else if rest.starts_with("engram_layer.multi_head_embedding.embedding.") {
            Component::MappedTable
        } else if rest.starts_with("engram_layer.") {
            Component::TableProjection
        } else if rest.starts_with("mlp.gate.") {
            Component::Router
        } else if rest.starts_with("mlp.shared_experts.") {
            Component::SharedExpert
        } else if rest.starts_with("mlp.routed_expert_") || rest.starts_with("mlp.combine_output_norm.") {
            Component::LatentProjection
        } else if rest.starts_with("mlp.") {
            Component::DenseFfn
        } else {
            return None;
        };
        Some(TensorRole::layer(component, layer))
    }
    fn component_hint(&self, component: Component) -> Option<Hint> {
        let how = match component {
            Component::Attention => "Export Hy4-flash absorbed MLA (H=32 DSA / H=64 SWA-512, scale=1/16, F32 learnable sinks folded into W_UV + sigmoid linear_gate + o_proj + post_proj_norm); see #2478.",
            Component::Indexer => "Export Hy4-flash 16-head compressed indexer (4-token gated pooling + intra_group_ape + top-511 group recall + current group -> 2048 candidate slots); see #2478.",
            Component::HyperConnection => "Export 4-stream iHC pre/post kernels and MhcHeadKernel(H=3072) in sparkinfer-glmrt; see #2478.",
            Component::RoutedExpert => "Wire hy4f EXL3 K4 MUL1 routed expert package (H1536, I1536, E512, K16, swiglu_limit=10.0) on the MUL1 branches (#415 / #2538); see #2478.",
            Component::MappedTable | Component::TableProjection => "Wire FP8-only Hy4EngramTable + shared look-ahead and 4-stream GPU Engram gate kernel; see #2478.",
            Component::Vision => "Optional 27-layer ViT vision tower is a follow-up item on #2478; text serving runs with --vision off.",
            Component::Speculator => "Optional 5-layer DFly2 drafter is a follow-up item on #2478; text serving runs without speculation first.",
            _ => "Implement Hy4-flash text forward, golden parity, and serve-hy4f over the shared scheduler; see #2478.",
        };
        Some(Hint {
            what: format!("Hy4-flash {} execution is planned (#2478)", component.label()),
            how: how.into(),
        })
    }
    fn optional(&self, component: Component) -> bool {
        matches!(component, Component::Vision | Component::Speculator)
    }
    fn expert_catalog(&self) -> bool {
        true
    }
}

impl FamilyModel for Model {
    fn spec(&self) -> &ModelSpec { &self.spec }
    fn contract_admission(&self) -> bool { true }
    fn native_serving_admission(&self) -> bool { true }
    fn cache_geometry(
        &self,
        options: crate::serving_capacity::CacheOptions,
    ) -> Result<Option<crate::serving_capacity::FamilyCacheGeometry>, crate::serving_capacity::CacheGeometryError> {
        use crate::families::hy4_flash::execution::{CacheLayout, PAGE_ROWS};
        use crate::families::hy4_flash::Hy4KvCache;
        use crate::placement::{families::hy4_flash as placement_hy4, AttentionPlacement, LayerMode};
        use crate::serving_capacity::{
            CacheGeometryError, FamilyCacheGeometry, KvPlacement, RankCacheGeometry,
        };
        if !matches!(options.coordinator_ranks, 1 | 2) {
            return Err(CacheGeometryError::Unsupported {
                family: "hy4_flash",
                what: "requires one or two coordinator GPUs",
            });
        }
        if options.native_mtp_layers > 0 {
            return Err(CacheGeometryError::Unsupported {
                family: "hy4_flash",
                what: "optional DFly2 drafter is reserved separately from target cache",
            });
        }
        let gpus = options.coordinator_ranks;
        let attention = placement_hy4::HY4_FLASH
            .check_attention(options.attention_placement, gpus, false)
            .map_err(|_| CacheGeometryError::Unsupported {
                family: "hy4_flash",
                what: "unsupported attention placement",
            })?;
        let modes = placement_hy4::layer_modes(&self.cfg, gpus, attention);
        let kv_format = match options.mla_kv {
            cuteafd_core::MlaCacheFormat::Bf16 => Hy4KvCache::Bf16,
            cuteafd_core::MlaCacheFormat::Fp8 => Hy4KvCache::Fp8,
        };
        let context_split = modes.iter().any(|m| *m == LayerMode::ContextSplit);
        let logical_unit_rows = if context_split { 2 * PAGE_ROWS as u64 } else { PAGE_ROWS as u64 };
        let placement = if gpus == 1 {
            KvPlacement::SingleDevice
        } else {
            match attention {
                AttentionPlacement::Context => KvPlacement::PartitionedContext,
                AttentionPlacement::Heads => KvPlacement::Replicated,
                AttentionPlacement::Layers => KvPlacement::PartitionedLayers,
            }
        };
        let mut ranks = Vec::with_capacity(gpus);
        for gpu in 0..gpus {
            let owned = |sparse: bool| -> usize {
                modes
                    .iter()
                    .enumerate()
                    .filter(|&(layer, mode)| {
                        self.cfg.sparse_layer[layer] == sparse
                            && match *mode {
                                LayerMode::Whole { gpu: owner, .. } => usize::from(owner) == gpu,
                                LayerMode::HeadSplit | LayerMode::ContextSplit => true,
                            }
                    })
                    .count()
            };
            let owned_dsa = owned(true);
            let owned_swa = owned(false);
            let layout = CacheLayout::new(&self.cfg, kv_format, 1, owned_dsa, owned_swa)
                .map_err(|_| CacheGeometryError::Overflow("hy4_flash cache layout"))?;
            let per_dsa = (layout.dsa_kv_page_bytes + layout.dsa_index_page_bytes) as u64;
            ranks.push(RankCacheGeometry {
                persistent_unit_bytes: (owned_dsa as u64)
                    .checked_mul(per_dsa)
                    .ok_or(CacheGeometryError::Overflow("hy4_flash persistent unit"))?,
                active_state_per_sequence_bytes: (layout.active_swa_ring_bytes
                    + layout.active_indexer_pending_bytes) as u64,
                retained_mark_bytes: layout.mark_bytes_per_slot as u64,
                ..Default::default()
            });
        }
        Ok(Some(FamilyCacheGeometry {
            logical_unit_rows,
            placement,
            ranks,
        }))
    }
    fn activation(&self, layer: usize, shared: bool) -> Option<cuteafd_core::FfnActivation> {
        (layer < self.cfg.layers).then(|| self.cfg.activation(layer, shared))
    }
    fn accepts(&self, role: &TensorRole, stem: &str, operand: &mut QuantOperand) -> Result<(), String> {
        // Selected EXL3 compact core projections (K8/K6 MUL1 or MCG) on target and DFly2.
        if let Some(projection) = self.core.get(stem) {
            return super::require(
                matches!(operand.encoding, Encoding::Exl3 { bits } if bits == projection.bits)
                    && operand.logical == [projection.stored_outputs, projection.shape[1]],
                || format!(
                    "{stem}: expected selected header-native EXL3 K{} [{}, {}], found {}",
                    projection.bits, projection.stored_outputs, projection.shape[1], super::describe(operand)
                ),
            );
        }

        // Optional Vision and DFly2 components (validated structurally by dtype/presence; follow-up).
        if matches!(role.component, Component::Vision | Component::Speculator) {
            return super::require(
                operand.is_plain(&[Encoding::Bf16, Encoding::F32]) && operand.input_scale.is_none(),
                || format!("{stem}: expected plain BF16/F32 optional {:?} tensor, found {}", role.component, super::describe(operand)),
            );
        }

        // Routed experts: EXL3 K4 MUL1 (or BF16), per-expert [1536, 1536].
        if role.component == Component::RoutedExpert {
            let layer = role.layer.ok_or_else(|| format!("{stem}: routed expert without layer"))?;
            let leaf = super::leaf(stem);
            let expected = match leaf {
                "gate_proj" | "up_proj" => [self.cfg.moe_intermediate, self.cfg.routed_expert_hidden],
                "down_proj" => [self.cfg.routed_expert_hidden, self.cfg.moe_intermediate],
                other => return Err(format!("{stem}: unknown Hy4-flash routed projection {other}")),
            };
            let storage = self.expert_storage.get(&layer).copied()
                .ok_or_else(|| format!("{stem}: layer {layer} is not a routed layer"))?;
            return match storage {
                ExpertStorage::Exl3 { bits, mul1 } => super::require(
                    matches!(operand.encoding, Encoding::Exl3 { bits: b } if b == bits)
                        && operand.logical == expected
                        && operand.companions.iter().any(|c| c == if mul1 { "mul1" } else { "mcg" }),
                    || format!(
                        "{stem}: expected EXL3 K{bits} {} {expected:?}, found {}",
                        if mul1 { "MUL1" } else { "MCG" },
                        super::describe(operand)
                    ),
                ),
                ExpertStorage::Bf16 => super::require(
                    operand.is_plain(&[Encoding::Bf16]) && operand.logical == expected,
                    || format!("{stem}: expected BF16 {expected:?}, found {}", super::describe(operand)),
                ),
            };
        }

        let contract = self.contracts.get(stem)
            .ok_or_else(|| format!("{stem}: outside the Hy4-flash tensor contract ({})", super::describe(operand)))?;
        super::require(
            contract.role.component == role.component && contract.role.layer == role.layer,
            || format!("{stem}: classified {:?}, contract says {:?}", role.component, contract.role.component),
        )?;
        let companions: Vec<&str> = operand.companions.iter().map(String::as_str).collect();
        let encoding = match contract.storage {
            Storage::Bf16 | Storage::MappedEngram => Encoding::Bf16,
            Storage::F32 => Encoding::F32,
            Storage::I32 => Encoding::Int { bits: 32 },
        };
        super::require(
            operand.logical == contract.shape
                && operand.is_plain(&[encoding])
                && operand.input_scale.is_none()
                && companions == contract.companions,
            || format!(
                "{stem}: expected {:?} {:?}{}, found {}",
                contract.storage,
                contract.shape,
                if contract.companions.is_empty() { String::new() } else { format!(" with {:?}", contract.companions) },
                super::describe(operand)
            ),
        )
    }
    fn experts(&self, operand: &QuantOperand) -> Option<ExpertContract> {
        match operand.encoding {
            Encoding::Exl3 { bits: 4 } => {
                let is_mul1 = operand.companions.iter().any(|c| c == "mul1");
                Some(ExpertContract {
                    package: if is_mul1 { "hy4f:exl3-k45-mul1".into() } else { "hy4f:exl3-k45".into() },
                    block: 128,
                    // 1536 / 128 = 12 blocks -> matches spark-package-ranges.json hy4f:exl3 worlds 1..=6.
                    spark_worlds: vec![1, 2, 3, 4, 5, 6],
                    local: if is_mul1 {
                        Ok("Hy4-flash RTX EXL3 K4 MUL1 onboarding (whole 1x or TP2 768+768)".into())
                    } else {
                        Err("Hy4-flash routed experts ship MUL1 only; MCG has no hy4f package".into())
                    },
                })
            }
            _ => None,
        }
    }
    fn placement(
        &self,
        input: &crate::plan::PlanInput,
        report: &mut crate::plan::PlanReport,
        options: &mut crate::plan::layout::LayoutOptions,
        requested: crate::plan::ExpertPlacement,
    ) -> Option<anyhow::Result<cuteafd_core::memory_layout::MemoryLayout>> {
        Some(Ok(placement::layout(self, input, report, options, requested)))
    }
}
