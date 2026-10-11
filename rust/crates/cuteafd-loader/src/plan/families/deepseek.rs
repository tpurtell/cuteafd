//! DeepSeek V4 and V4.1: compressed MLA with indexers, mHC hyper-connections,
//! DeepSeek-native tensor names (optionally with HF-named EXL3 experts).
use anyhow::Result;
use serde_json::Value;

use super::{describe, require};
use crate::families::deepseek_v4::DeepseekV4Config;
use crate::plan::checkpoint::{opt_usize_field, usize_field, Checkpoint};
use crate::plan::family::{ConfigError, ExpertContract, Family, FamilyModel, Hint, RuntimeStatus};
use crate::plan::format::{Encoding, QuantOperand, ScaleEncoding};
use crate::plan::names::{indexed, indexed_tail};
use crate::plan::spec::*;

pub struct DeepSeek {
    id: &'static str,
    architecture: &'static str,
    runtime: RuntimeStatus,
}

pub static DEEPSEEK_V41: DeepSeek = DeepSeek {
    id: "deepseek_v41",
    architecture: "DeepseekV41ForCausalLM",
    runtime: RuntimeStatus::Serving,
};
pub static DEEPSEEK_V4: DeepSeek = DeepSeek {
    id: "deepseek_v4",
    architecture: "DeepseekV4ForCausalLM",
    runtime: RuntimeStatus::Serving,
};

fn usize_list(config: &Value, key: &str) -> Vec<usize> {
    config
        .get(key)
        .and_then(Value::as_array)
        .map(|values| values.iter().filter_map(Value::as_u64).map(|v| v as usize).collect())
        .unwrap_or_default()
}

impl Family for DeepSeek {
    fn id(&self) -> &'static str {
        self.id
    }
    fn runtime(&self) -> RuntimeStatus {
        self.runtime
    }
    fn detect(&self, checkpoint: &Checkpoint) -> bool {
        checkpoint.architectures().iter().any(|arch| arch == self.architecture)
    }

    fn open(&self, checkpoint: &Checkpoint) -> Result<Box<dyn FamilyModel>, ConfigError> {
        let spec = if self.id == "deepseek_v4" {
            v4_spec(checkpoint).map_err(ConfigError::from_anyhow)?
        } else {
            v41_spec(checkpoint).map_err(ConfigError::from_anyhow)?
        };
        let v4_config = if self.id == "deepseek_v4" {
            Some(DeepseekV4Config::read(&checkpoint.snapshot, dspark_stages(checkpoint))
                .map_err(ConfigError::from_anyhow)?)
        } else { None };
        Ok(Box::new(DeepSeekModel { id: self.id, spec, v4_config, config: checkpoint.config.clone() }))
    }

    fn classify(&self, _spec: &ModelSpec, name: &str) -> Option<TensorRole> {
        use Component::*;
        match name {
            "embed.weight" => return Some(TensorRole::new(Embedding)),
            "head.weight" => return Some(TensorRole::new(LmHead)),
            "norm.weight" => return Some(TensorRole::new(Norm)),
            _ => {}
        }
        if name.starts_with("hc_head_") {
            return Some(TensorRole::new(HyperConnection));
        }
        if name.starts_with("vision.") || name.starts_with("aligner.") || name.starts_with("image_") {
            return Some(TensorRole::new(Vision));
        }
        // EXL3 publications store routed experts with HF names.
        if let Some((layer, rest)) = indexed(name, "model.layers.") {
            if let Some((expert, _)) = indexed(rest, "mlp.experts.") {
                return Some(TensorRole::expert(RoutedExpert, layer, expert));
            }
            return Some(TensorRole::layer(Other, layer));
        }
        if let Some((stage, rest)) = indexed_tail(name, "mtp.") {
            if let Some((expert, _)) = indexed(rest, "ffn.experts.")
                .or_else(|| indexed(rest, "mlp.experts."))
            {
                return Some(TensorRole::expert(SpeculatorExpert, stage, expert));
            }
            return Some(TensorRole::layer(Speculator, stage));
        }
        let (layer, rest) = indexed(name, "layers.")?;
        let component = if let Some((expert, _)) = indexed(rest, "ffn.experts.") {
            return Some(TensorRole::expert(RoutedExpert, layer, expert));
        } else if rest.starts_with("attn.compressor.") || rest.starts_with("attn.indexer.compressor.") {
            Compressor
        } else if rest.starts_with("attn.indexer.") {
            Indexer
        } else if rest.starts_with("attn.") {
            Attention
        } else if rest.starts_with("ffn.gate.") {
            Router
        } else if rest.starts_with("ffn.shared_experts.") {
            SharedExpert
        } else if rest.starts_with("hc_") {
            HyperConnection
        } else if rest.starts_with("engram.embed.") {
            MappedTable
        } else if rest.starts_with("engram.") {
            TableProjection
        } else if rest.ends_with("norm.weight") {
            Norm
        } else {
            Other
        };
        Some(TensorRole::layer(component, layer))
    }

    fn expert_catalog(&self) -> bool {
        self.id == "deepseek_v4"
    }

    fn optional(&self, component: Component) -> bool {
        self.id == "deepseek_v4"
            && matches!(component, Component::Speculator | Component::SpeculatorExpert | Component::Vision)
    }

    fn component_hint(&self, component: Component) -> Option<Hint> {
        if self.id == "deepseek_v4" && component == Component::RoutedExpert {
            return Some(Hint {
                what: "DeepSeek V4 routed experts in NVFP4".into(),
                how: "Spark expertd-native serves native MXFP4 and EXL3 K2-K4 experts at the checkpoint \
                      geometry (ExpertGeometry, read_expert_catalog; EXL3 packages exl3-dsv4f|dsv4p-k<tiers> \
                      from CUTEAFD_EXPERT_FAMILIES=dsv4p:exl3-k23). ModelOpt NVFP4 needs the V4.1 NVFP4 \
                      contract (loader v41_nvfp4) generalized the same way.".into(),
            });
        }
        if self.runtime == RuntimeStatus::Serving {
            return None;
        }
        let (what, how) = match component {
            Component::Attention | Component::Compressor | Component::Indexer => (
                "DeepSeek V4 compressed MLA (ratios 4/128 alternating), per-layer compressor and indexer",
                "Map each component against the official inference/model.py. V4.1's attention stack (rust daemon v41_attention_*, v41_compressor, v41_index_*, \
                 native/families/deepseek_v41/) is the template; V4 differs in ratio schedule (4 and 128 \
                 instead of 2 and 1), has an indexer compressor per ratio-4 layer, no CED encoder/decoder \
                 KV sharing, and FP8 128x128 blocks instead of 32x32. b12x kernels: \
                 attention/compressed_sparse_mla, dsv4_compressor, dsa_indexer. The legacy ds4rt engine \
                 (../ds4rt) served this family; its real_full attention path is the numerical reference.",
            ),
            Component::Router => (
                "hash routing on layers 0..num_hash_layers (ffn.gate.tid2eid) plus sqrtsoftplus noaux_tc",
                "tid2eid maps token id to its 6 experts for the first layers (no scores); later layers use \
                 V4.1's router with gate.bias. Add a hash-route path to the router kernel launch.",
            ),
            Component::RoutedExpert | Component::SpeculatorExpert => (
                "routed experts at this family's geometry",
                "Spark expertd-native and the V4.1 AOT exports are specialized to hidden 5120 / \
                 intermediate 2304 / 384 experts / top-6. Parameterize python/tools/export_b12x_v41_* \
                 by (hidden, intermediate, experts, top_k, format) and select the variant from the \
                 model spec; V4 Flash is 4096/2048/256/6 MXFP4, V4 Pro EXL3 K2 is 7168/3072/384/6 \
                 (b12x moe fused_moe Trellis).",
            ),
            Component::HyperConnection => (
                "mHC with hc_head_* output mixing",
                "V4.1 mHC kernels (v41_hc) apply; V4 adds the hc_head_{fn,base,scale} output head.",
            ),
            Component::Speculator => (
                "three-stage dSpark drafter",
                "Same structure as V4.1 dSpark (markov_head, confidence_head, main_proj) at this \
                 family's width; reuse v41_dspark with geometry parameters.",
            ),
            _ => return None,
        };
        Some(Hint { what: what.into(), how: how.into() })
    }
}

fn dedup(values: &[usize]) -> Vec<usize> {
    let mut out: Vec<usize> = values.to_vec();
    out.sort_unstable();
    out.dedup();
    out
}

/// dSpark stages the checkpoint carries (`mtp.{s}.*`): V4's HF config says
/// one while every V4 checkpoint ships three.
fn dspark_stages(checkpoint: &Checkpoint) -> usize {
    checkpoint
        .tensor_names()
        .filter_map(|name| indexed_tail(name, "mtp.").map(|(index, _)| index))
        .max()
        .map_or(0, |max| max + 1)
}

/// V4's spec from the runtime's own reader (`DeepseekV4Config::read`:
/// `inference/config.json` when the snapshot has one, else the mapped HF
/// config), so serve-dsv4 and the plan see the same schedule.
fn v4_spec(checkpoint: &Checkpoint) -> Result<ModelSpec> {
    let stages = dspark_stages(checkpoint);
    let cfg = DeepseekV4Config::read(&checkpoint.snapshot, stages)?;
    let layers = (0..cfg.n_layers)
        .map(|layer| {
            let ratio = cfg.compress_ratios[layer];
            LayerSpec {
                attention: AttentionKind::CompressedMla { ratio, indexer: ratio == 4 },
                ffn: FfnKind::Moe,
                rope: Some(RopeSpec {
                    dims: cfg.rope_head_dim,
                    theta: if ratio == 0 { cfg.rope_theta } else { cfg.compress_rope_theta },
                }),
            }
        })
        .collect();
    let mut notes = vec![format!("compress ratios {:?}", dedup(&cfg.compress_ratios)), format!("mHC width {}", cfg.hc_mult)];
    if cfg.n_hash_layers > 0 {
        notes.push(format!("hash-routed layers 0..{} (ffn.gate.tid2eid)", cfg.n_hash_layers));
    }
    if checkpoint.snapshot.join("inference/config.json").is_file() {
        notes.push("configuration from inference/config.json (the runtime's reader)".into());
    }
    Ok(ModelSpec {
        family: "deepseek_v4",
        architecture: checkpoint.architectures().first().cloned().unwrap_or_default(),
        hidden: cfg.dim,
        vocab: cfg.vocab_size,
        layers,
        moe: Some(MoeSpec {
            experts: cfg.n_routed_experts,
            top_k: cfg.n_activated_experts,
            intermediate: cfg.moe_inter_dim,
            shared_experts: cfg.n_shared_experts,
            shared_intermediate: cfg.moe_inter_dim,
            scoring: cfg.score_func.clone(),
            routed_scaling: Some(cfg.route_scale),
            groups: None,
        }),
        speculator: (stages > 0).then(|| SpeculatorSpec::Dspark {
            stages,
            experts: cfg.n_routed_experts,
            top_k: cfg.n_activated_experts,
            target_layers: cfg.dspark_target_layer_ids.clone(),
        }),
        tables: Vec::new(),
        vision: checkpoint.config.get("vision_config").is_some(),
        notes,
    })
}

/// V4.1's spec from its HF config. serve-native validates the pinned official
/// configuration (`OfficialV41Config`); the planner describes V4.1
/// publications (EXL3, NVFP4) whose configs extend it, reading the same keys
/// without defaults for the schedule.
fn v41_spec(checkpoint: &Checkpoint) -> Result<ModelSpec> {
    let text = checkpoint.text_config();
    let layers = usize_field(text, "num_hidden_layers")?;
    let ratios = usize_list(text, "compress_ratios");
    anyhow::ensure!(ratios.len() >= layers, "compress_ratios has {} entries for {layers} layers", ratios.len());
    let index_sources = usize_list(text, "index_source_layer_ids");
    let hash_layers = opt_usize_field(text, "num_hash_layers").unwrap_or(0);
    // A layer carries its own indexer when its tensors say so; V4.1 publishes
    // the owners explicitly.
    let has_indexer = |layer: usize| {
        let prefix = format!("layers.{layer}.attn.indexer.");
        checkpoint.tensor_names().any(|name| name.starts_with(&prefix)) || index_sources.contains(&layer)
    };
    let layer_specs = (0..layers)
        .map(|layer| LayerSpec {
            attention: AttentionKind::CompressedMla { ratio: ratios[layer], indexer: has_indexer(layer) },
            ffn: FfnKind::Moe,
            rope: None,
        })
        .collect();
    let stages = dspark_stages(checkpoint);
    let speculator = (stages > 0).then(|| SpeculatorSpec::Dspark {
        stages,
        experts: opt_usize_field(text, "dspark_n_routed_experts")
            .or_else(|| opt_usize_field(text, "n_routed_experts"))
            .unwrap_or(0),
        top_k: opt_usize_field(text, "dspark_num_experts_per_tok")
            .or_else(|| opt_usize_field(text, "num_experts_per_tok"))
            .unwrap_or(0),
        target_layers: usize_list(text, "dspark_target_layer_ids"),
    });
    let engram = usize_list(text, "engram_layer_ids");
    let mut tables = Vec::new();
    if !engram.is_empty() {
        tables.push(MappedTableSpec { name: "engram".into(), layers: engram });
    }
    let mut notes = vec![format!("compress ratios {:?}", dedup(&ratios[..layers]))];
    if let Some(hc) = opt_usize_field(text, "hc_mult") {
        notes.push(format!("mHC width {hc}"));
    }
    if hash_layers > 0 {
        notes.push(format!("hash-routed layers 0..{hash_layers} (ffn.gate.tid2eid)"));
    }
    if let Some(sources) = text.get("kv_source_layer_ids") {
        notes.push(format!("CED KV sources {sources}"));
    }
    Ok(ModelSpec {
        family: "deepseek_v41",
        architecture: "DeepseekV41ForCausalLM".into(),
        hidden: usize_field(text, "hidden_size")?,
        vocab: usize_field(text, "vocab_size")?,
        layers: layer_specs,
        moe: Some(MoeSpec {
            experts: usize_field(text, "n_routed_experts")?,
            top_k: usize_field(text, "num_experts_per_tok")?,
            intermediate: usize_field(text, "moe_intermediate_size")?,
            shared_experts: opt_usize_field(text, "n_shared_experts").unwrap_or(0),
            shared_intermediate: usize_field(text, "moe_intermediate_size")?,
            scoring: text.get("scoring_func").and_then(Value::as_str).unwrap_or("softmax").into(),
            routed_scaling: text.get("routed_scaling_factor").and_then(Value::as_f64),
            groups: None,
        }),
        speculator,
        tables,
        vision: checkpoint.config.get("vision_config").is_some(),
        notes,
    })
}

struct DeepSeekModel {
    id: &'static str,
    spec: ModelSpec,
    v4_config: Option<DeepseekV4Config>,
    config: Value,
}

/// The routed projection shape `[N, K]` a stem names: w1/w3 (gate/up) are
/// `[I, H]`, w2 (down) `[H, I]`.
pub(crate) fn routed_shape(stem: &str, hidden: usize, intermediate: usize) -> Option<[usize; 2]> {
    match super::leaf(stem) {
        "w1" | "w3" | "gate_proj" | "up_proj" => Some([intermediate, hidden]),
        "w2" | "down_proj" => Some([hidden, intermediate]),
        _ => None,
    }
}

impl DeepSeekModel {
    fn geometry(&self) -> Option<&'static str> {
        let moe = self.spec.moe.as_ref()?;
        cuteafd_core::ExpertGeometry {
            hidden: self.spec.hidden as u32,
            experts: moe.experts as u32,
            topk: moe.top_k as u32,
            intermediate: moe.intermediate as u32,
            layers: self.spec.layers.len() as u32,
        }
        .family()
    }

    fn routed(&self, stem: &str, operand: &QuantOperand) -> Result<(), String> {
        let moe = self.spec.moe.as_ref().ok_or("no MoE geometry")?;
        let geometry = self.geometry().ok_or_else(|| format!("no expert kernels at hidden {} / intermediate {} / {} \
            experts / top-{} (cuteafd_core::ExpertGeometry)", self.spec.hidden, moe.intermediate, moe.experts, moe.top_k))?;
        let shape = routed_shape(stem, self.spec.hidden, moe.intermediate)
            .ok_or_else(|| format!("not a routed projection (w1/w2/w3 or gate/up/down_proj)"))?;
        require(operand.logical == shape, || format!("{geometry} experts are {shape:?}, found {}", describe(operand)))?;
        let nvfp4 = self.id == "deepseek_v41" && operand.is_nvfp4();
        require(operand.is_mxfp4(32) || matches!(operand.exl3_bits(), Some(2..=4)) || nvfp4, || {
            format!("{geometry} experts run MXFP4 (E2M1 + UE8M0 per 32), EXL3 K2-K4{}; found {}",
                if self.id == "deepseek_v41" { " or ModelOpt NVFP4" } else { "" }, describe(operand))
        })
    }

    /// Block-FP8 coordinator weights: E4M3 with UE8M0 scales, 32x32 (V4.1)
    /// or 128x128 with 128-multiple extents (V4's scale prep).
    fn coordinator(&self, operand: &QuantOperand) -> Result<(), String> {
        if operand.is_plain(&[Encoding::Bf16, Encoding::F32]) || matches!(operand.encoding, Encoding::Int { .. }) {
            return Ok(());
        }
        let block = if self.id == "deepseek_v4" { 128 } else { 32 };
        let extents = operand.matrix().is_some_and(|(n, k)| n % block == 0 && k % block == 0);
        require(operand.is_fp8_block(block, &[ScaleEncoding::Ue8m0]) && extents, || {
            format!("coordinator weights are BF16, FP32, integer or E4M3 with UE8M0 {block}x{block} block scales over \
                {block}-multiple extents; found {}", describe(operand))
        })
    }
}

impl FamilyModel for DeepSeekModel {
    fn spec(&self) -> &ModelSpec {
        &self.spec
    }

    fn cache_geometry(&self, options: crate::serving_capacity::CacheOptions)
        -> Result<Option<crate::serving_capacity::FamilyCacheGeometry>, crate::serving_capacity::CacheGeometryError> {
        use crate::serving_capacity::{deepseek_v41_cache_geometry, deepseek_v4_cache_geometry, CacheGeometryError};
        if let Some(cfg) = &self.v4_config {
            deepseek_v4_cache_geometry(cfg, options.coordinator_ranks, options.prefill_rows,
                options.native_mtp_layers).map(Some)
        } else if options.native_mtp_layers == 0 {
            deepseek_v41_cache_geometry(&self.config, options.coordinator_ranks).map(Some)
        } else {
            Err(CacheGeometryError::Unsupported {
                family: "deepseek_v41", what: "dSpark state is a separate reservation, not a target cache",
            })
        }
    }

    fn accepts(&self, role: &TensorRole, stem: &str, operand: &mut QuantOperand) -> Result<(), String> {
        match (self.id, role.component) {
            (_, Component::RoutedExpert) => self.routed(stem, operand),
            ("deepseek_v4", Component::Speculator | Component::SpeculatorExpert) => {
                Err("serve-dsv4 does not run the dSpark drafter".into())
            }
            ("deepseek_v4", Component::Vision) => Err("serve-dsv4 takes no images".into()),
            (_, Component::SpeculatorExpert) => self.routed(stem, operand),
            (_, Component::MappedTable) => require(
                operand.is_fp8_row_groups(32, &[ScaleEncoding::Ue8m0]) || operand.is_nvfp4(),
                || format!("engram rows are E4M3 with UE8M0 1x32 scales or NVFP4; found {}", describe(operand)),
            ),
            _ => self.coordinator(operand),
        }
    }

    fn experts(&self, operand: &QuantOperand) -> Option<ExpertContract> {
        let geometry = self.geometry()?;
        let format = match operand.exl3_bits() {
            Some(bits) => format!("exl3-k{bits}"),
            None if operand.is_mxfp4(32) => "mxfp4".into(),
            None if operand.is_nvfp4() => "nvfp4".into(),
            None => return None,
        };
        let serve = if self.id == "deepseek_v4" { "serve-dsv4" } else { "serve-native" };
        Some(ExpertContract {
            package: format!("{geometry}:{format} (expertd-native)"),
            block: 128,
            // Native packages remain bounded independently of transport capacity.
            spark_worlds: vec![2, 3, 4, 6],
            local: Err(format!("{serve} runs routed experts on 2, 3, 4 or 6 Spark ranks (its local expert layers \
                supplement them)")),
        })
    }
}
