//! Hy4 (`HYV4VLForConditionalGeneration`, `hy_v4_vl`):
//! 78-layer uniform gated DSA MLA (64 heads, `q_lora_rank=2048`, `kv_lora_rank=512`,
//! `qk=192+64`, `v=256`, F32 learnable sinks, sigmoid output gate, `qa_kva_eps=1e-6`,
//! adjacent-pair RoPE), 4-stream identity hyper-connections (iHC, FP32 mixing),
//! 32-head per-token DSA IndexCache (21 full layers `0, 1, 5, 9, ..., 77`; 57 shared
//! layers reuse the preceding full layer's top-2048), layer 0 dense FFN (18432),
//! layers 1..=77 direct MoE (256 routed experts top-8, intermediate 2048,
//! `routed_scaling_factor=2.827`, `swiglu_limit=10.0`, 1 shared expert 2048;
//! no Engram, no LatentMoE, no SWA, no sandwich norms), FP32 untied LM head,
//! optional 5-layer DFly2 drafter (`[61, 65, 69, 73, 77]`), and 40-layer ViT (#2550).
//!
//! Validates strictly from the checkpoint's own `config.json`, safetensors
//! tensor headers, and `cuteafd-mc.json` view declarations.
use std::collections::BTreeMap;

use anyhow::{ensure, Result};

use crate::families::hy4::weight_layout::core_contract;
use crate::families::hy4::{Hy4Config, Hy4DflyConfig, Hy4ExpertStorage, Hy4VisionConfig};
use crate::families::hy4_common::{detect_expert_storage, HEAD_STEM};
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

pub struct Hy4;
pub static HY4: Hy4 = Hy4;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Storage {
    Bf16,
    F32,
}

#[derive(Debug, Clone)]
pub(crate) struct Contract {
    pub role: TensorRole,
    pub shape: Vec<usize>,
    pub storage: Storage,
    pub companions: Vec<&'static str>,
    pub bare: bool,
}

pub struct Model {
    pub cfg: Hy4Config,
    spec: ModelSpec,
    contracts: BTreeMap<String, Contract>,
    pub core: BTreeMap<String, CoreProjection>,
    pub(crate) routed_bytes: u64,
    pub(crate) expert_storage: BTreeMap<usize, Hy4ExpertStorage>,
}

impl Model {
    pub fn open(checkpoint: &Checkpoint) -> Result<Self> {
        let cfg = Hy4Config::from_hf(&checkpoint.config)?;
        cfg.release_geometry()?;
        let dfly = Hy4DflyConfig::from_hf(&checkpoint.config, "Hy4", cfg.layers, cfg.vocab_size)?;
        let vision = Hy4VisionConfig::from_hf(&checkpoint.config, "Hy4")?;
        let (h, v, hc) = (cfg.hidden, cfg.vocab_size, cfg.hc_mult);

        let mut contracts = BTreeMap::new();
        let mut add = |stem: String,
                       role: TensorRole,
                       shape: Vec<usize>,
                       storage: Storage,
                       companions: Vec<&'static str>,
                       bare: bool| {
            contracts.insert(
                stem,
                Contract {
                    role,
                    shape,
                    storage,
                    companions,
                    bare,
                },
            );
        };

        // Globals
        add(
            "model.embed_tokens".into(),
            TensorRole::new(Component::Embedding),
            vec![v, h],
            Storage::Bf16,
            vec![],
            false,
        );
        add(
            HEAD_STEM.into(),
            TensorRole::new(Component::LmHead),
            vec![v, h],
            Storage::Bf16,
            vec![],
            false,
        );
        add(
            "model.norm".into(),
            TensorRole::new(Component::Norm),
            vec![h],
            Storage::Bf16,
            vec![],
            false,
        );
        add(
            "model.hc_head.hc_head_fn".into(),
            TensorRole::new(Component::HyperConnection),
            vec![hc, hc * h],
            Storage::F32,
            vec![],
            true,
        );
        add(
            "model.hc_head.hc_head_base".into(),
            TensorRole::new(Component::HyperConnection),
            vec![hc],
            Storage::F32,
            vec![],
            true,
        );
        add(
            "model.hc_head.hc_head_scale".into(),
            TensorRole::new(Component::HyperConnection),
            vec![1],
            Storage::F32,
            vec![],
            true,
        );

        for layer in 0..cfg.layers {
            let base = format!("model.layers.{layer}");
            let at = |c| TensorRole::layer(c, layer);

            // Layer norms + 4-stream iHC (no sandwich norms)
            add(format!("{base}.input_layernorm"), at(Component::Norm), vec![h], Storage::Bf16, vec![], false);
            add(
                format!("{base}.post_attention_layernorm"),
                at(Component::Norm),
                vec![h],
                Storage::Bf16,
                vec![],
                false,
            );
            for tag in ["hc_attn_layer", "hc_mlp_layer"] {
                let p = format!("{base}.{tag}.hc_pre");
                add(format!("{p}.hc_fn"), at(Component::HyperConnection), vec![2 * hc, hc * h], Storage::F32, vec![], true);
                add(format!("{p}.hc_base"), at(Component::HyperConnection), vec![2 * hc], Storage::F32, vec![], true);
                add(format!("{p}.hc_scale"), at(Component::HyperConnection), vec![2], Storage::F32, vec![], true);
            }

            // Uniform 64-head gated DSA MLA
            let a = format!("{base}.self_attn");
            add(format!("{a}.q_a_proj"), at(Component::Attention), vec![cfg.q_lora_rank, h], Storage::Bf16, vec![], false);
            add(format!("{a}.q_a_layernorm"), at(Component::Attention), vec![cfg.q_lora_rank], Storage::Bf16, vec![], false);
            add(
                format!("{a}.q_b_proj"),
                at(Component::Attention),
                vec![cfg.heads * cfg.qk_head_dim(), cfg.q_lora_rank],
                Storage::Bf16,
                vec![],
                false,
            );
            add(
                format!("{a}.kv_a_proj_with_mqa"),
                at(Component::Attention),
                vec![cfg.kv_lora_rank + cfg.qk_rope_dim, h],
                Storage::Bf16,
                vec![],
                false,
            );
            add(format!("{a}.kv_a_layernorm"), at(Component::Attention), vec![cfg.kv_lora_rank], Storage::Bf16, vec![], false);
            add(
                format!("{a}.kv_b_proj"),
                at(Component::Attention),
                vec![cfg.heads * (cfg.qk_nope_dim + cfg.v_head_dim), cfg.kv_lora_rank],
                Storage::Bf16,
                vec![],
                false,
            );
            add(
                format!("{a}.linear_gate"),
                at(Component::Attention),
                vec![cfg.heads * cfg.v_head_dim, h],
                Storage::Bf16,
                vec![],
                false,
            );
            add(
                format!("{a}.o_proj"),
                at(Component::Attention),
                vec![h, cfg.heads * cfg.v_head_dim],
                Storage::Bf16,
                vec![],
                false,
            );
            add(
                format!("{a}.learnable_sink_param"),
                at(Component::Attention),
                vec![cfg.heads],
                Storage::F32,
                vec![],
                true,
            );

            // 32-head per-token DSA indexer (only on the 21 full IndexCache layers)
            if cfg.is_full_indexer(layer) {
                let d = cfg.indexer_head_dim;
                let n = cfg.indexer_heads;
                let ip = format!("{a}.indexer");
                add(format!("{ip}.wq_b"), at(Component::Indexer), vec![n * d, cfg.q_lora_rank], Storage::Bf16, vec![], false);
                add(format!("{ip}.wk"), at(Component::Indexer), vec![d, h], Storage::Bf16, vec![], false);
                add(format!("{ip}.k_norm"), at(Component::Indexer), vec![d], Storage::Bf16, vec!["bias"], false);
                add(format!("{ip}.weights_proj"), at(Component::Indexer), vec![n, h], Storage::Bf16, vec![], false);
            }

            // FFN: dense layer 0 or direct MoE layers 1..=77
            if cfg.dense_layer[layer] {
                add(format!("{base}.mlp.gate_proj"), at(Component::DenseFfn), vec![cfg.intermediate, h], Storage::Bf16, vec![], false);
                add(format!("{base}.mlp.up_proj"), at(Component::DenseFfn), vec![cfg.intermediate, h], Storage::Bf16, vec![], false);
                add(format!("{base}.mlp.down_proj"), at(Component::DenseFfn), vec![h, cfg.intermediate], Storage::Bf16, vec![], false);
            } else {
                let shared = cfg.moe_intermediate * cfg.shared_experts;
                add(format!("{base}.mlp.gate"), at(Component::Router), vec![cfg.experts, h], Storage::Bf16, vec![], false);
                add(
                    format!("{base}.mlp.gate.e_score_correction_bias"),
                    at(Component::Router),
                    vec![cfg.experts],
                    Storage::F32,
                    vec![],
                    true,
                );
                add(
                    format!("{base}.mlp.shared_experts.gate_proj"),
                    at(Component::SharedExpert),
                    vec![shared, h],
                    Storage::Bf16,
                    vec![],
                    false,
                );
                add(
                    format!("{base}.mlp.shared_experts.up_proj"),
                    at(Component::SharedExpert),
                    vec![shared, h],
                    Storage::Bf16,
                    vec![],
                    false,
                );
                add(
                    format!("{base}.mlp.shared_experts.down_proj"),
                    at(Component::SharedExpert),
                    vec![h, shared],
                    Storage::Bf16,
                    vec![],
                    false,
                );
            }
        }
        drop(add);

        let routed_bytes = checkpoint
            .tensors
            .iter()
            .filter(|t| t.meta.name.contains(".mlp.experts."))
            .map(|t| t.meta.byte_length)
            .sum();

        let mut expert_storage = BTreeMap::new();
        for layer in cfg.routed_layers() {
            expert_storage.insert(layer, detect_expert_storage(checkpoint, "Hy4", layer)?);
        }

        // Read EXL3 compact core projections (K8/K6 MUL1) on target + DFly2 layers.
        let mut expected: BTreeMap<String, Vec<usize>> = contracts
            .iter()
            .filter(|(_, c)| c.shape.len() == 2 && matches!(c.storage, Storage::Bf16) && !c.bare)
            .map(|(stem, c)| (format!("{stem}.weight"), c.shape.clone()))
            .collect();
        if let Some(dfly_cfg) = &dfly {
            let dfly_mlp = cfg.intermediate;
            for l in 0..dfly_cfg.layers {
                let p = format!("model.dfly_layers.{l}");
                for (sub, shape) in [
                    ("self_attn.q_a_proj", vec![cfg.q_lora_rank, h]),
                    ("self_attn.q_b_proj", vec![cfg.heads * cfg.qk_head_dim(), cfg.q_lora_rank]),
                    ("self_attn.kv_a_proj_with_mqa", vec![cfg.kv_lora_rank + cfg.qk_rope_dim, h]),
                    (
                        "self_attn.kv_b_proj",
                        vec![cfg.heads * (cfg.qk_nope_dim + cfg.v_head_dim), cfg.kv_lora_rank],
                    ),
                    ("self_attn.linear_gate", vec![cfg.heads * cfg.v_head_dim, h]),
                    ("self_attn.o_proj", vec![h, cfg.heads * cfg.v_head_dim]),
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
            let name = if contract.bare {
                stem.clone()
            } else {
                format!("{stem}.weight")
            };
            ensure!(
                checkpoint.contains_tensor(&name),
                "Hy4 checkpoint lacks {name} ({:?})",
                contract.role.component
            );
            for companion in &contract.companions {
                let comp = format!("{stem}.{companion}");
                ensure!(
                    checkpoint.contains_tensor(&comp),
                    "Hy4 checkpoint lacks {comp} ({:?})",
                    contract.role.component
                );
            }
        }

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

fn build_spec(
    cfg: &Hy4Config,
    dfly: Option<&Hy4DflyConfig>,
    vision: bool,
    expert_storage: &BTreeMap<usize, Hy4ExpertStorage>,
    checkpoint: &Checkpoint,
) -> ModelSpec {
    let layers = (0..cfg.layers)
        .map(|layer| LayerSpec {
            attention: AttentionKind::MlaDsa {
                indexer: cfg.is_full_indexer(layer),
            },
            ffn: if cfg.dense_layer[layer] {
                FfnKind::Dense {
                    intermediate: cfg.intermediate,
                }
            } else {
                FfnKind::Moe
            },
            rope: Some(RopeSpec {
                dims: cfg.qk_rope_dim,
                theta: cfg.rope_theta,
            }),
        })
        .collect();

    let mut formats = BTreeMap::<String, Vec<usize>>::new();
    for (&layer, &storage) in expert_storage {
        let label = match storage {
            Hy4ExpertStorage::Exl3 { bits, mul1: true } => format!("EXL3 K{bits} MUL1"),
            Hy4ExpertStorage::Exl3 { bits, mul1: false } => format!("EXL3 K{bits} MCG"),
            Hy4ExpertStorage::Bf16PerExpert => "BF16 (per-expert)".to_string(),
            Hy4ExpertStorage::Bf16Fused => "BF16 (fused gate_up_proj/down_proj)".to_string(),
        };
        formats.entry(label).or_default().push(layer);
    }

    let mut notes = vec![
        format!(
            "attention: uniform gated DSA MLA ({} layers x {} heads, q_lora {}, kv_lora {}, qk {}/{}, v {}, \
             learnable F32 sinks, sigmoid output gate, qa_kva_eps={}, adjacent-pair RoPE theta={}; \
             IndexCache: {} full layers {:?} with {}-head dim-{} top-{}, {} shared layers reusing preceding full selection)",
            cfg.layers,
            cfg.heads,
            cfg.q_lora_rank,
            cfg.kv_lora_rank,
            cfg.qk_nope_dim,
            cfg.qk_rope_dim,
            cfg.v_head_dim,
            cfg.qa_kva_eps,
            cfg.rope_theta,
            cfg.index_cache.full_count(),
            cfg.full_indexer_layers(),
            cfg.indexer_heads,
            cfg.indexer_head_dim,
            cfg.indexer_topk,
            cfg.index_cache.shared_count(),
        ),
        format!(
            "residuals: {}-stream identity hyper-connections (iHC, eps {}, magnitude {}) + hc_head collapse before final RMSNorm",
            cfg.hc_mult, cfg.hc_eps, cfg.hc_magnitude
        ),
        format!(
            "direct MoE (no LatentMoE, no Engram, no SWA, no sandwich norms): {} routed layers (layer 0 dense {}), \
             H={}, I={}, {} experts top-{} (sigmoid + bias, norm_topk_prob={}, scale {}), {} shared expert ({}), \
             SwiGLU clamp {} on routed only; storage: {}",
            cfg.routed_layers().len(),
            cfg.intermediate,
            cfg.hidden,
            cfg.moe_intermediate,
            cfg.experts,
            cfg.topk,
            cfg.norm_topk_prob,
            cfg.routed_scaling_factor,
            cfg.shared_experts,
            cfg.moe_intermediate,
            cfg.swiglu_limit,
            formats
                .iter()
                .map(|(label, ls)| format!("{label} ({} layers)", ls.len()))
                .collect::<Vec<_>>()
                .join("; "),
        ),
    ];
    if let Some(d) = dfly {
        notes.push(format!(
            "bundled DFly2 drafter (optional): {} dense layers (kv taps {:?}, index taps {:?}, top-{}, block_size={:?})",
            d.layers, d.target_kv_layers, d.target_index_layers, d.target_index_topk, d.block_size
        ));
    }

    ModelSpec {
        family: "hy4",
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
        tables: Vec::new(),
        vision,
        notes,
    }
}

impl Family for Hy4 {
    fn id(&self) -> &'static str {
        "hy4"
    }
    fn runtime(&self) -> RuntimeStatus {
        RuntimeStatus::Serving
    }
    fn attention_precision(&self) -> crate::plan::attention_precision::PrecisionSupport {
        crate::plan::attention_precision::MLA_BF16_ONLY
    }
    fn decode_lanes(&self) -> crate::plan::decode_lanes::LaneSupport {
        crate::plan::decode_lanes::LaneSupport::One {
            reason: "Hy4 serving starts on one decode lane; a second lane waits for a measured two-lane A/B (#2550, #744)",
        }
    }
    fn decode_rows(&self) -> crate::plan::decode_rows::DecodeRowSupport {
        crate::plan::decode_rows::DecodeRowSupport::fixed(
            crate::placement::families::hy4::DECODE_ROWS,
            0,
            "Hy4 split-K decode exports cover at most 64 rows per step",
        )
    }
    fn spark_decode(&self) -> Option<crate::plan::spark_executions::SparkDecode> {
        Some(crate::plan::spark_executions::SparkDecode {
            rows: crate::placement::families::hy4::DECODE_ROWS,
        })
    }
    fn eager_native_core_headers(&self) -> bool {
        true
    }
    fn eager_expert_header(&self, name: &str) -> bool {
        let Some(tail) = name.strip_prefix("model.layers.") else {
            return false;
        };
        let Some((layer, tail)) = tail.split_once(".mlp.experts.") else {
            return false;
        };
        layer.parse::<usize>().is_ok()
            && matches!(
                tail,
                "0.gate_proj.trellis"
                    | "0.gate_proj.mul1"
                    | "0.gate_proj.weight"
                    | "0.up_proj.trellis"
                    | "0.down_proj.trellis"
                    | "gate_up_proj"
                    | "down_proj"
            )
    }
    fn fp8_native_mtp_layers(&self, config: &serde_json::Value) -> Option<(&'static str, std::ops::Range<usize>)> {
        let layers = config.get("dfly_num_hidden_layers").and_then(serde_json::Value::as_u64)? as usize;
        (layers > 0).then_some(("model.dfly_layers.", 0..layers))
    }
    fn detect(&self, checkpoint: &Checkpoint) -> bool {
        checkpoint.architectures().iter().any(|a| a == "HYV4VLForConditionalGeneration")
            || checkpoint.model_type() == Some("hy_v4_vl")
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
        if let Some(after) = rest.strip_prefix("mlp.experts.") {
            if let Some((expert, _)) = after
                .split_once('.')
                .and_then(|(e, tail)| e.parse::<usize>().ok().map(|id| (id, tail)))
            {
                return Some(TensorRole::expert(Component::RoutedExpert, layer, expert));
            }
            if matches!(after, "gate_up_proj" | "down_proj") {
                return Some(TensorRole::layer(Component::RoutedExpert, layer));
            }
        }
        let component = if rest.starts_with("self_attn.indexer.") {
            Component::Indexer
        } else if rest.starts_with("self_attn.") {
            Component::Attention
        } else if rest.starts_with("hc_attn_layer.") || rest.starts_with("hc_mlp_layer.") {
            Component::HyperConnection
        } else if rest.starts_with("input_layernorm.") || rest.starts_with("post_attention_layernorm.") {
            Component::Norm
        } else if rest.starts_with("mlp.gate.") {
            Component::Router
        } else if rest.starts_with("mlp.shared_experts.") {
            Component::SharedExpert
        } else if rest.starts_with("mlp.") {
            Component::DenseFfn
        } else {
            return None;
        };
        Some(TensorRole::layer(component, layer))
    }
    fn component_hint(&self, component: Component) -> Option<Hint> {
        let how = match component {
            Component::Attention => "Export Hy4 absorbed gated MLA (64H TP1 / 32H TP2, q_lora=2048, qa_kva_eps=1e-6, kv_lora=512, F32 sinks + sigmoid linear_gate, interleaved tail-64 RoPE) via hy4_mla.py; see #2550.",
            Component::Indexer => "Export Hy4IndexPost (32H x 128, LayerNorm eps=1e-5, tail-64 interleaved RoPE -> 8448 B/page) and reuse compile_glm_index_topk_aot with IndexCache reuse across the 57 shared layers; see #2550.",
            Component::HyperConnection => "Export 4-stream iHC pre/post/post_pre and MhcHeadKernel at H=6144 (Hy4IhcGeometry(hidden=6144)) in hy4_ihc.py; see #2550.",
            Component::RoutedExpert => "Disambiguate ExpertGeometry::HY4 from GLM5 (same 6144/256/8/2048 shape, swiglu_limit=10.0 vs None) and export hy4:exl3-k45-mul1 for SM121 TP3..6 and SM120 RTX TP1/TP2; see #2550.",
            Component::Vision => "Optional 40-layer ViT vision tower is a follow-up item on #2550; text serving runs with --vision off.",
            Component::Speculator => "Optional 5-layer DFly2 drafter (taps [61, 65, 69, 73, 77], dense I=18432) is a follow-up item on #2550; text serving runs without speculation first.",
            _ => "Wire Hy4 text forward, golden parity vs hy4-v1_20261122_own-text-bf16-f32, and serve-hy4 once wrldsuksgo2mars/Hy4-EXL3-K4-MC publishes; see #2550.",
        };
        Some(Hint {
            what: format!("Hy4 {} execution is planned (#2550)", component.label()),
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
    fn spec(&self) -> &ModelSpec {
        &self.spec
    }
    fn contract_admission(&self) -> bool {
        true
    }
    fn native_serving_admission(&self) -> bool {
        true
    }
    fn cache_geometry(
        &self,
        options: crate::serving_capacity::CacheOptions,
    ) -> Result<Option<crate::serving_capacity::FamilyCacheGeometry>, crate::serving_capacity::CacheGeometryError> {
        use crate::families::hy4::execution::CacheLayout;
        use crate::families::hy4::Hy4KvCache;
        use crate::families::hy4_common::PAGE_ROWS;
        use crate::placement::{families::hy4 as placement_hy4, AttentionPlacement, LayerMode};
        use crate::serving_capacity::{
            CacheGeometryError, FamilyCacheGeometry, KvPlacement, RankCacheGeometry,
        };
        if !matches!(options.coordinator_ranks, 1 | 2) {
            return Err(CacheGeometryError::Unsupported {
                family: "hy4",
                what: "requires one or two coordinator GPUs",
            });
        }
        if options.native_mtp_layers > 0 {
            return Err(CacheGeometryError::Unsupported {
                family: "hy4",
                what: "optional DFly2 drafter is reserved separately from target cache",
            });
        }
        let gpus = options.coordinator_ranks;
        let attention = placement_hy4::HY4
            .check_attention(options.attention_placement, gpus, gpus == 2)
            .map_err(|_| CacheGeometryError::Unsupported {
                family: "hy4",
                what: "unsupported attention placement",
            })?;
        let modes = placement_hy4::layer_modes(&self.cfg, gpus, attention);
        let kv_format = match options.mla_kv {
            cuteafd_core::MlaCacheFormat::Bf16 => Hy4KvCache::Bf16,
            cuteafd_core::MlaCacheFormat::Fp8 => Hy4KvCache::Fp8,
        };
        let context_split = modes.iter().any(|m| *m == LayerMode::ContextSplit);
        let logical_unit_rows = if context_split {
            2 * PAGE_ROWS as u64
        } else {
            PAGE_ROWS as u64
        };
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
            let mut owned_kv = 0usize;
            let mut owned_full_idx = 0usize;
            for (layer, &mode) in modes.iter().enumerate() {
                let owns = match mode {
                    LayerMode::Whole { gpu: owner, .. } => usize::from(owner) == gpu,
                    LayerMode::HeadSplit | LayerMode::ContextSplit => true,
                };
                if !owns {
                    continue;
                }
                owned_kv += 1;
                if self.cfg.is_full_indexer(layer) {
                    owned_full_idx += 1;
                }
            }
            let layout = CacheLayout::new(kv_format, owned_kv, owned_full_idx)
                .map_err(|_| CacheGeometryError::Overflow("hy4 cache layout"))?;
            let persistent_unit_bytes = layout
                .persistent_unit_bytes()
                .ok_or(CacheGeometryError::Overflow("hy4 persistent unit"))?;
            ranks.push(RankCacheGeometry {
                persistent_unit_bytes,
                active_state_per_sequence_bytes: 0,
                retained_mark_bytes: 0,
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
                || {
                    format!(
                        "{stem}: expected selected header-native EXL3 K{} [{}, {}], found {}",
                        projection.bits,
                        projection.stored_outputs,
                        projection.shape[1],
                        super::describe(operand)
                    )
                },
            );
        }

        // Optional Vision and DFly2 components (validated structurally by dtype/presence).
        if matches!(role.component, Component::Vision | Component::Speculator) {
            return super::require(
                operand.is_plain(&[Encoding::Bf16, Encoding::F32]) && operand.input_scale.is_none(),
                || {
                    format!(
                        "{stem}: expected plain BF16/F32 optional {:?} tensor, found {}",
                        role.component,
                        super::describe(operand)
                    )
                },
            );
        }

        // Routed experts: EXL3 K4 MUL1 (quant) or BF16 fused/per-expert (source).
        if role.component == Component::RoutedExpert {
            let layer = role
                .layer
                .ok_or_else(|| format!("{stem}: routed expert without layer"))?;
            let storage = self
                .expert_storage
                .get(&layer)
                .copied()
                .ok_or_else(|| format!("{stem}: layer {layer} is not a routed layer"))?;
            let leaf = super::leaf(stem);
            return match storage {
                Hy4ExpertStorage::Bf16Fused => {
                    let expected = match leaf {
                        "gate_up_proj" => vec![self.cfg.experts, 2 * self.cfg.moe_intermediate, self.cfg.hidden],
                        "down_proj" => vec![self.cfg.experts, self.cfg.hidden, self.cfg.moe_intermediate],
                        other => return Err(format!("{stem}: unknown Hy4 fused routed projection {other}")),
                    };
                    super::require(
                        operand.is_plain(&[Encoding::Bf16]) && operand.logical == expected,
                        || format!("{stem}: expected BF16 fused {expected:?}, found {}", super::describe(operand)),
                    )
                }
                Hy4ExpertStorage::Exl3 { bits, mul1 } => {
                    let expected = match leaf {
                        "gate_proj" | "up_proj" => [self.cfg.moe_intermediate, self.cfg.hidden],
                        "down_proj" => [self.cfg.hidden, self.cfg.moe_intermediate],
                        other => return Err(format!("{stem}: unknown Hy4 routed projection {other}")),
                    };
                    super::require(
                        matches!(operand.encoding, Encoding::Exl3 { bits: b } if b == bits)
                            && operand.logical == expected
                            && operand
                                .companions
                                .iter()
                                .any(|c| c == if mul1 { "mul1" } else { "mcg" }),
                        || {
                            format!(
                                "{stem}: expected EXL3 K{bits} {} {expected:?}, found {}",
                                if mul1 { "MUL1" } else { "MCG" },
                                super::describe(operand)
                            )
                        },
                    )
                }
                Hy4ExpertStorage::Bf16PerExpert => {
                    let expected = match leaf {
                        "gate_proj" | "up_proj" => [self.cfg.moe_intermediate, self.cfg.hidden],
                        "down_proj" => [self.cfg.hidden, self.cfg.moe_intermediate],
                        other => return Err(format!("{stem}: unknown Hy4 routed projection {other}")),
                    };
                    super::require(
                        operand.is_plain(&[Encoding::Bf16]) && operand.logical == expected,
                        || format!("{stem}: expected BF16 {expected:?}, found {}", super::describe(operand)),
                    )
                }
            };
        }

        let contract = self.contracts.get(stem).ok_or_else(|| {
            format!(
                "{stem}: outside the Hy4 tensor contract ({})",
                super::describe(operand)
            )
        })?;
        super::require(
            contract.role.component == role.component && contract.role.layer == role.layer,
            || {
                format!(
                    "{stem}: classified {:?}, contract says {:?}",
                    role.component, contract.role.component
                )
            },
        )?;
        let companions: Vec<&str> = operand.companions.iter().map(String::as_str).collect();
        let encoding = match contract.storage {
            Storage::Bf16 => Encoding::Bf16,
            Storage::F32 => Encoding::F32,
        };
        super::require(
            operand.logical == contract.shape
                && operand.is_plain(&[encoding])
                && operand.input_scale.is_none()
                && companions == contract.companions,
            || {
                format!(
                    "{stem}: expected {:?} {:?}{}, found {}",
                    contract.storage,
                    contract.shape,
                    if contract.companions.is_empty() {
                        String::new()
                    } else {
                        format!(" with {:?}", contract.companions)
                    },
                    super::describe(operand)
                )
            },
        )
    }
    fn experts(&self, operand: &QuantOperand) -> Option<ExpertContract> {
        match operand.encoding {
            Encoding::Exl3 { bits: 4 } => {
                let is_mul1 = operand.companions.iter().any(|c| c == "mul1");
                let package = if is_mul1 {
                    "hy4:exl3-k45-mul1".to_string()
                } else {
                    "hy4:exl3-k45".to_string()
                };
                let spark_worlds = crate::plan::experts::spark_worlds(&package, self.cfg.moe_intermediate);
                Some(ExpertContract {
                    package,
                    block: 128,
                    spark_worlds,
                    local: if is_mul1 {
                        Ok("Hy4 RTX EXL3 K4 MUL1 onboarding (whole 2048 or TP2 1024+1024)".into())
                    } else {
                        Err("Hy4 routed experts ship MUL1 only; MCG has no hy4 package".into())
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
