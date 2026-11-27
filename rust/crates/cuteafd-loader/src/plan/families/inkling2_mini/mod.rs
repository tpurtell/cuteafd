//! Inkling 2 Mini (`Inkling2LoopForConditionalGeneration`, `inkling2_mini`):
//! looped 24-layer GQA + SiTU MoE transformer (prelude 0..6, core 6..18 looped
//! 3 passes with rank-64 cross-pass message controllers, coda 18..24 -> 48 visits
//! and 48 distinct KV + short-conv cache slots: 8 global paged + 40 local SWA-1024),
//! native NVFP4 W4A4 (layers 0..20) + MXFP8 W8A8 (layers 20..24), bundled 3-layer
//! recurrent dSpark drafter (`mtp.safetensors`, taps `[5, 23, 41, 48]`), and
//! bundled 23-layer PE-Lang ViT (#2701). TP1 only (1x RTX, 0 Sparks).
use std::collections::BTreeMap;

use anyhow::{ensure, Result};

use crate::families::inkling2::{Inkling2MtpConfig, Inkling2VisionConfig};
use crate::families::inkling2_mini::weight_layout::{
    CONTROLLER_STREAMS, CROSS_PASS_KV_QUERIES, CROSS_PASS_QUERIES,
};
use crate::families::inkling2_mini::Inkling2MiniConfig;
use crate::plan::families::inkling2::{expert_stem, expert_storage, ExpertStorage, TEXT, VISION};
use crate::plan::{
    checkpoint::Checkpoint,
    family::*,
    format::{self, QuantOperand, ScaleEncoding},
    names::indexed,
    spec::*,
};

mod placement;
#[cfg(test)]
pub(crate) mod tests;

pub struct Inkling2Mini;
pub static INKLING2_MINI: Inkling2Mini = Inkling2Mini;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Storage {
    Bf16,
    F32,
    Experts,
}

#[derive(Debug, Clone)]
struct Contract {
    role: TensorRole,
    shape: Vec<usize>,
    storage: Storage,
    companions: Vec<&'static str>,
    scale_member: bool,
    bare: bool,
}

pub(super) struct Model {
    pub(crate) cfg: Inkling2MiniConfig,
    spec: ModelSpec,
    contracts: BTreeMap<String, Contract>,
    draft_shapes: BTreeMap<String, Vec<usize>>,
    pub(crate) routed_bytes: u64,
    pub(crate) expert_storage: Vec<ExpertStorage>,
    pub(crate) has_vision: bool,
    pub(crate) has_mtp: bool,
}

fn text(name: &str) -> Option<&str> {
    name.strip_prefix(TEXT)
}

impl Model {
    fn open(checkpoint: &Checkpoint) -> Result<Self> {
        let cfg = Inkling2MiniConfig::from_hf(&checkpoint.config)?;
        cfg.release_geometry()?;
        let vision = Inkling2VisionConfig::from_hf(&checkpoint.config)?;
        let mtp = Inkling2MtpConfig::from_hf_with_target(
            &checkpoint.config,
            cfg.text.hidden,
            cfg.total_visits(),
        )?;
        let t = &cfg.text;
        let lp = &cfg.loop_cfg;
        let (h, v, d) = (t.hidden, t.vocab_size, t.head_dim);

        let mut contracts = BTreeMap::new();
        let mut add = |stem: String,
                       role: TensorRole,
                       shape: Vec<usize>,
                       storage: Storage,
                       bare: bool| {
            let scale_member = stem.ends_with("_l2_clip");
            let bias = stem.starts_with(VISION)
                && (stem.ends_with(".ln_pre")
                    || stem.ends_with(".ln_1")
                    || stem.ends_with(".ln_2")
                    || stem.contains(".self_attn.")
                    || stem.contains(".mlp."));
            contracts.insert(
                stem,
                Contract {
                    role,
                    shape,
                    storage,
                    companions: if bias { vec!["bias"] } else { Vec::new() },
                    scale_member,
                    bare,
                },
            );
        };

        add(format!("{TEXT}embed"), TensorRole::new(Component::Embedding), vec![v, h], Storage::Bf16, false);
        add(format!("{TEXT}unembed"), TensorRole::new(Component::LmHead), vec![v, h], Storage::Bf16, false);
        for norm in ["embed_norm", "norm"] {
            add(format!("{TEXT}{norm}"), TensorRole::new(Component::Norm), vec![h], Storage::Bf16, false);
        }
        add(format!("{TEXT}attn_res.output_query"), TensorRole::new(Component::ResidualMix), vec![h], Storage::F32, true);
        add(
            CROSS_PASS_QUERIES.into(),
            TensorRole::new(Component::ResidualMix),
            vec![2 * lp.core_layers(), h],
            Storage::F32,
            true,
        );
        add(
            CROSS_PASS_KV_QUERIES.into(),
            TensorRole::new(Component::ResidualMix),
            vec![2 * lp.core_layers(), h],
            Storage::F32,
            true,
        );

        for layer in 0..t.layers {
            let base = format!("{TEXT}layers.{layer}");
            let at = |c| TensorRole::layer(c, layer);
            let local = t.local[layer];
            let a = format!("{base}.attn");
            for (proj, n, k) in [
                ("wq_du", t.heads * d, h),
                ("wr_du", t.heads * t.d_rel, h),
                ("wk_dv", t.kv_heads * d, h),
                ("wv_dv", t.kv_heads * d, h),
                ("gate_attn", t.heads * d, h),
                ("attn_sink_proj", t.heads, h),
                ("wo_ud", h, t.heads * d),
            ] {
                add(format!("{a}.{proj}"), at(Component::Attention), vec![n, k], Storage::Bf16, false);
            }
            add(format!("{a}.q_l2_clip"), at(Component::Attention), vec![t.heads, d], Storage::Bf16, false);
            add(format!("{a}.rel_l2_clip"), at(Component::Attention), vec![t.heads, t.d_rel], Storage::Bf16, false);
            add(
                format!("{a}.rel_logits_proj.proj"),
                at(Component::Attention),
                vec![t.d_rel, if local { t.window } else { t.rel_extent }],
                Storage::Bf16,
                true,
            );
            for name in ["k", "v"] {
                let stem = format!("{a}.{name}_sconv");
                if local {
                    add(stem, at(Component::Attention), vec![t.kv_heads * d, 1, t.sconv_kernel], Storage::Bf16, false);
                } else {
                    add(format!("{stem}.static_w"), at(Component::Attention), vec![t.sconv_kernel, t.kv_heads * d], Storage::Bf16, true);
                    add(format!("{stem}.weight_proj"), at(Component::Attention), vec![t.kv_heads * t.sconv_kernel, h], Storage::Bf16, true);
                }
            }
            for name in ["attn_sconv", "mlp_sconv"] {
                add(format!("{base}.{name}"), at(Component::Attention), vec![h, 1, t.sconv_kernel], Storage::Bf16, false);
            }
            let norms: &[&str] = if local {
                &["attn_norm"]
            } else {
                &["q_input_norm", "k_input_norm", "v_input_norm"]
            };
            for norm in norms.iter().chain(["mlp_norm"].iter()) {
                add(format!("{base}.{norm}"), at(Component::Norm), vec![h], Storage::Bf16, false);
            }
            add(format!("{base}.mlp.gate"), at(Component::Router), vec![t.experts, h], Storage::Bf16, false);
            add(
                expert_stem(layer, "w13"),
                TensorRole::layer(Component::RoutedExpert, layer),
                vec![t.experts, 2 * t.intermediate, h],
                Storage::Experts,
                false,
            );
            add(
                expert_stem(layer, "w2"),
                TensorRole::layer(Component::RoutedExpert, layer),
                vec![t.experts, h, t.intermediate],
                Storage::Experts,
                false,
            );
            add(format!("{TEXT}attn_res.mlp_queries.{layer}"), at(Component::ResidualMix), vec![h], Storage::F32, true);
            if layer > 0 {
                add(
                    format!("{TEXT}attn_res.attn_queries.{}", layer - 1),
                    at(Component::ResidualMix),
                    if local { vec![h] } else { vec![3, h] },
                    Storage::F32,
                    true,
                );
            }
            if lp.is_core_layer(layer) {
                for stream in CONTROLLER_STREAMS {
                    let c = format!("{base}.controllers.{stream}");
                    add(format!("{c}.down_weight"), at(Component::ResidualMix), vec![lp.gate_rank, 2 * h], Storage::Bf16, true);
                    add(format!("{c}.forget_weight"), at(Component::ResidualMix), vec![h, lp.gate_rank], Storage::Bf16, true);
                    add(format!("{c}.gain_weight"), at(Component::ResidualMix), vec![h, lp.gate_rank], Storage::Bf16, true);
                }
            }
        }

        if let Some(vc) = &vision {
            let role = TensorRole::new(Component::Vision);
            let (vh, vi) = (vc.hidden, vc.intermediate);
            let tower = format!("{VISION}backbone");
            add(format!("{tower}.embeddings.patch_embedding"), role.clone(), vec![vh, 3 * vc.patch * vc.patch], Storage::Bf16, false);
            add(format!("{tower}.embeddings.position_embedding"), role.clone(), vec![vc.posemb_grid * vc.posemb_grid, vh], Storage::Bf16, true);
            for name in ["class_embedding", "cls_position_embedding"] {
                add(format!("{tower}.embeddings.{name}"), role.clone(), vec![vh], Storage::Bf16, true);
            }
            add(format!("{tower}.ln_pre"), role.clone(), vec![vh], Storage::Bf16, false);
            for layer in 0..vc.layers {
                let base = format!("{tower}.layers.{layer}");
                for norm in ["ln_1", "ln_2"] {
                    add(format!("{base}.{norm}"), role.clone(), vec![vh], Storage::Bf16, false);
                }
                for scale in ["ls_1", "ls_2"] {
                    add(format!("{base}.{scale}.gamma"), role.clone(), vec![vh], Storage::Bf16, true);
                }
                for proj in ["q_proj", "k_proj", "v_proj", "out_proj"] {
                    add(format!("{base}.self_attn.{proj}"), role.clone(), vec![vh, vh], Storage::Bf16, false);
                }
                add(format!("{base}.mlp.c_fc"), role.clone(), vec![vi, vh], Storage::Bf16, false);
                add(format!("{base}.mlp.c_proj"), role.clone(), vec![vh, vi], Storage::Bf16, false);
            }
            add(format!("{VISION}proj_in"), role.clone(), vec![vc.projector_hidden, vc.merge * vc.merge * vh], Storage::Bf16, false);
            add(format!("{VISION}proj_out"), role, vec![h, vc.projector_hidden], Storage::Bf16, false);
        }
        drop(add);

        for layer in 0..t.layers {
            contracts
                .get_mut(&format!("{TEXT}layers.{layer}.mlp.gate"))
                .expect("router contract")
                .companions
                .push("bias");
        }

        let routed_bytes = checkpoint
            .tensors
            .iter()
            .filter(|tensor| tensor.meta.name.contains(".mlp.experts."))
            .map(|tensor| tensor.meta.byte_length)
            .sum();
        let expert_storage = (0..t.layers)
            .map(|layer| expert_storage(checkpoint, layer))
            .collect::<Result<Vec<_>>>()?;

        let draft_shapes = if mtp.is_some() {
            use crate::formats::block_dspark::recurrent::{source_shapes, RecurrentBlockConfig};
            source_shapes(&RecurrentBlockConfig::from_hf(&checkpoint.config)?)?
                .into_iter()
                .map(|(name, shape)| (format!("mtp.{name}"), shape))
                .collect::<BTreeMap<_, _>>()
        } else {
            BTreeMap::new()
        };

        let tower_present = checkpoint.tensor_names().any(|name| name.starts_with(VISION));
        for (stem, contract) in &contracts {
            if contract.role.component == Component::Vision && !tower_present {
                continue;
            }
            match contract.storage {
                Storage::Experts => {
                    let layer = contract.role.layer.expect("routed experts are per layer");
                    ensure!(
                        matches!(expert_storage[layer], ExpertStorage::Nvfp4 | ExpertStorage::Mxfp8),
                        "Inkling 2 Mini layer {layer}: only native NVFP4/MXFP8 routed experts are supported"
                    );
                    ensure!(
                        checkpoint.contains_tensor(&format!("{stem}_weight")),
                        "Inkling 2 Mini checkpoint lacks {stem}_weight (routed_expert)"
                    );
                }
                _ => {
                    let name = if contract.scale_member {
                        format!("{stem}.scale")
                    } else if contract.bare {
                        stem.clone()
                    } else {
                        format!("{stem}.weight")
                    };
                    ensure!(
                        checkpoint.contains_tensor(&name),
                        "Inkling 2 Mini checkpoint lacks {name} ({:?})",
                        contract.role.component
                    );
                    for companion in &contract.companions {
                        let comp = format!("{stem}.{companion}");
                        ensure!(
                            checkpoint.contains_tensor(&comp),
                            "Inkling 2 Mini checkpoint lacks {comp} ({:?})",
                            contract.role.component
                        );
                    }
                }
            }
        }

        let has_vision = vision.is_some() && tower_present;
        let has_mtp = mtp.is_some();
        let spec = build_spec(&cfg, vision.is_some(), mtp.as_ref(), &expert_storage, checkpoint);
        Ok(Self {
            cfg,
            spec,
            contracts,
            draft_shapes,
            routed_bytes,
            expert_storage,
            has_vision,
            has_mtp,
        })
    }

    fn contract(&self, stem: &str) -> Option<&Contract> {
        self.contracts
            .get(stem)
            .or_else(|| stem.strip_suffix(".weight").and_then(|s| self.contracts.get(s)))
    }
}

fn build_spec(
    cfg: &Inkling2MiniConfig,
    vision: bool,
    mtp: Option<&Inkling2MtpConfig>,
    storage: &[ExpertStorage],
    checkpoint: &Checkpoint,
) -> ModelSpec {
    let t = &cfg.text;
    let lp = &cfg.loop_cfg;
    let layers = (0..t.layers)
        .map(|layer| LayerSpec {
            attention: if t.local[layer] {
                AttentionKind::SlidingGqa {
                    heads: t.heads,
                    kv_heads: t.kv_heads,
                    head_dim: t.head_dim,
                    window: t.window,
                    sinks: true,
                }
            } else {
                AttentionKind::Gqa {
                    heads: t.heads,
                    kv_heads: t.kv_heads,
                    head_dim: t.head_dim,
                }
            },
            ffn: FfnKind::Moe,
            rope: None,
        })
        .collect();

    let mut formats = BTreeMap::<String, Vec<usize>>::new();
    for (layer, st) in storage.iter().enumerate() {
        let name = match st {
            ExpertStorage::Mxfp8 => "MXFP8 W8A8".to_string(),
            ExpertStorage::Nvfp4 => "native NVFP4 W4A4 (per-token amax, fouroversix_mae)".to_string(),
            ExpertStorage::Exl3 { bits } => format!("EXL3 K{bits}"),
        };
        formats.entry(name).or_default().push(layer);
    }

    let mut notes = vec![
        format!(
            "looped backbone (HF transformers@d8f10a82): {} physical layers (prelude 0..{}, core {}..{} x {} passes, coda {}..{}) -> \
             {} total visits, {} distinct KV + short-conv cache slots ({} local SWA-{} + {} global paged slots {:?}); \
             passes 1..{} restart from pre-core state with rank-{} cross-pass controllers (attn/k/v/mlp)",
            t.layers,
            lp.core_start,
            lp.core_start,
            lp.core_end,
            lp.num_passes,
            lp.core_end,
            t.layers,
            cfg.total_visits(),
            cfg.cache_slots(),
            cfg.local_cache_slots(),
            t.window,
            cfg.global_cache_slots().len(),
            cfg.global_cache_slots(),
            lp.num_passes,
            lp.gate_rank,
        ),
        format!(
            "attention: GQA {}/{} x {} (group size {}) on every layer; {} physical local SWA-{} and {} physical global layers {:?} \
             with relative-position bias (d_rel {}, extent {}), per-token per-head sinks, sigmoid output gate, q/k L2 clip {:.3}, \
             r L2 clip {}, short K/V convolutions (kernel {}; dynamic taps on global layers)",
            t.heads,
            t.kv_heads,
            t.head_dim,
            t.group(),
            t.local_layers(),
            t.window,
            t.global_layers().len(),
            t.global_layers(),
            t.d_rel,
            t.rel_extent,
            t.qk_l2_clip,
            t.rel_l2_clip,
            t.sconv_kernel,
        ),
        format!(
            "residuals: softmax mixes over {}-layer blocks (F32 queries) + cross-pass 2-block message mix (middle_cycle_cross_pass_*queries); \
             per-layer attn_sconv/mlp_sconv",
            t.res_block
        ),
        format!(
            "MoE: sigmoid + bias top-{} of {}, log-sigmoid softmax weights x {}; SiTU (gate softcap {}, up softcap {}); \
             fused interleaved w13 (even=gate, odd=up); route weight before down-projection quantization",
            t.topk, t.experts, t.route_scale, t.gate_softcap, t.up_softcap
        ),
        format!(
            "routed expert storage by physical layer: {}",
            formats
                .iter()
                .map(|(name, ls)| {
                    let (first, last) = (ls[0], ls[ls.len() - 1]);
                    if ls.len() == last - first + 1 {
                        format!("{name} layers {first}..={last}")
                    } else {
                        format!("{name} layers {ls:?}")
                    }
                })
                .collect::<Vec<_>>()
                .join("; ")
        ),
        format!(
            "head: logits trimmed to {} of {} stored rows, softcap {}; TP1 only (1x RTX, 0 Sparks)",
            t.unpadded_vocab_size, t.vocab_size, t.final_softcap
        ),
    ];
    if let Some((floor, alpha)) = t.log_scaling {
        notes.push(format!(
            "global layers scale q and r by 1 + {alpha} ln(max(1, (pos + 1) / {floor}))"
        ));
    }
    if let Some(mtp) = mtp {
        notes.push(format!(
            "bundled dSpark drafter (mtp.safetensors): {} dense layers (local {:?}, window {}), block {}, \
             rnn_hidden Markov head rank {}, visit feature taps {:?}",
            mtp.layers,
            mtp.local
                .iter()
                .enumerate()
                .filter(|(_, l)| **l)
                .map(|(i, _)| i)
                .collect::<Vec<_>>(),
            mtp.window,
            mtp.block,
            mtp.markov_rank,
            mtp.target_layers
        ));
    }

    ModelSpec {
        family: "inkling2_mini",
        architecture: checkpoint.architectures().first().cloned().unwrap_or_default(),
        hidden: t.hidden,
        vocab: t.vocab_size,
        layers,
        moe: Some(MoeSpec {
            experts: t.experts,
            top_k: t.topk,
            intermediate: t.intermediate,
            shared_experts: 0,
            shared_intermediate: 0,
            scoring: "sigmoid + bias for selection; log-sigmoid softmax weights".into(),
            routed_scaling: Some(t.route_scale),
            groups: None,
        }),
        speculator: mtp.map(|m| SpeculatorSpec::BundledDspark { layers: m.layers }),
        tables: Vec::new(),
        vision,
        notes,
    }
}

impl Family for Inkling2Mini {
    fn id(&self) -> &'static str {
        "inkling2_mini"
    }

    fn runtime(&self) -> RuntimeStatus {
        RuntimeStatus::Serving
    }

    fn detect(&self, checkpoint: &Checkpoint) -> bool {
        checkpoint
            .architectures()
            .iter()
            .any(|a| a == "Inkling2LoopForConditionalGeneration")
            || checkpoint.model_type() == Some("inkling2_loop")
    }

    fn eager_expert_header(&self, name: &str) -> bool {
        let Some(tail) = name.strip_prefix("model.llm.layers.") else {
            return false;
        };
        let Some((layer, tail)) = tail.split_once(".mlp.experts.") else {
            return false;
        };
        layer.parse::<usize>().is_ok() && matches!(tail, "w13_weight" | "w2_weight")
    }

    fn decode_lanes(&self) -> crate::plan::decode_lanes::LaneSupport {
        crate::plan::decode_lanes::LaneSupport::One {
            reason: "Inkling 2 Mini serves TP1 locally on one RTX GPU with one decode lane (#2701)",
        }
    }

    fn decode_rows(&self) -> crate::plan::decode_rows::DecodeRowSupport {
        crate::plan::decode_rows::DecodeRowSupport::fixed(
            128,
            7,
            "Inkling 2 Mini coordinator decode programs cover 128 rows (missing inkling2_mini_*_m256 export)",
        )
    }

    fn open(&self, checkpoint: &Checkpoint) -> Result<Box<dyn FamilyModel>, ConfigError> {
        Ok(Box::new(Model::open(checkpoint).map_err(ConfigError::from_anyhow)?))
    }

    fn classify(&self, spec: &ModelSpec, name: &str) -> Option<TensorRole> {
        if name.starts_with(VISION) {
            return Some(TensorRole::new(Component::Vision));
        }
        if name.starts_with("mtp.") {
            return Some(TensorRole::new(Component::Speculator));
        }
        let rest = text(name)?;
        if rest.starts_with("embed.") {
            return Some(TensorRole::new(Component::Embedding));
        }
        if rest.starts_with("embed_norm.") || rest.starts_with("norm.") {
            return Some(TensorRole::new(Component::Norm));
        }
        if rest.starts_with("unembed.") {
            return Some(TensorRole::new(Component::LmHead));
        }
        if rest == "middle_cycle_cross_pass_queries" || rest == "middle_cycle_cross_pass_kv_queries" {
            return Some(TensorRole::new(Component::ResidualMix));
        }
        if let Some(query) = rest.strip_prefix("attn_res.") {
            if query == "output_query" {
                return Some(TensorRole::new(Component::ResidualMix));
            }
            let (kind, index) = query.split_once('.')?;
            let index: usize = index.parse().ok()?;
            let layer = if kind == "attn_queries" {
                index + 1
            } else if kind == "mlp_queries" {
                index
            } else {
                return None;
            };
            return (layer < spec.layers.len()).then(|| TensorRole::layer(Component::ResidualMix, layer));
        }
        let (layer, rest) = indexed(rest, "layers.")?;
        if layer >= spec.layers.len() {
            return None;
        }
        if rest.starts_with("mlp.experts.") {
            return Some(TensorRole::layer(Component::RoutedExpert, layer));
        }
        let component = if rest.starts_with("mlp.gate.") {
            Component::Router
        } else if rest.starts_with("controllers.") {
            Component::ResidualMix
        } else if rest.starts_with("attn.") || rest.starts_with("attn_sconv.") || rest.starts_with("mlp_sconv.") {
            Component::Attention
        } else if rest.ends_with("_norm.weight") || rest == "mlp_norm.weight" || rest == "attn_norm.weight" {
            Component::Norm
        } else {
            return None;
        };
        Some(TensorRole::layer(component, layer))
    }

    fn component_hint(&self, component: Component) -> Option<Hint> {
        let how = match component {
            Component::Attention => {
                "Missing kernels/exports in sparkinfer-glmrt: add INKLING2_MINI geometry (H=2304, heads=24, kv_heads=8, \
                 head_dim=128, group=3, window=1024, rel_extent=1024, log_floor=8192) to inkling2_attention.py + _inkling2_kernels.py, \
                 and export H=2304/heads=24 variants for inkling2_clip.py, inkling2_convolution*.py, and inkling2_glue.py."
            }
            Component::ResidualMix => {
                "Missing kernels/exports in sparkinfer-glmrt: generalize inkling2_residual.py to H=2304 with 4-query cross-pass mix, \
                 and add new inkling2_loop_controller.py (pair_rms_norm + blend_rms_norm for gate_rank=64 controllers on layers 6..18)."
            }
            Component::RoutedExpert => {
                "Missing kernels/exports in sparkinfer-glmrt: register inkling2_mini_nvfp4 (H=2304, I=1152, E=128, K=8, fouroversix_mae, \
                 SiTU 4/25) and inkling2_mini_mxfp8 (UE8M0, SiTU 4/25) in fp8_moe.py, and export SM120 TP1 packages in package_fp8_moe_aot.py."
            }
            Component::Router => {
                "Missing export: inkling2_router.py for E=128, topk=8, route_scale=8.0, H=2304 and output gate width 3072."
            }
            Component::Vision => {
                "Missing export: extend export_b12x_vision_aot.py for PE-Lang ViT proj_out=[2304, 4096] and wire through shared media path with own arena."
            }
            Component::Speculator => {
                "Missing export: extend inkling_drafter.py for 3-layer DSpark/Markov (H=2304, heads=24, kv_heads=8, I=9216, target_visits=[5,23,41,48])."
            }
            _ => {
                "Missing engine: implement cuteafd-daemon/src/families/inkling2_mini/ (48-visit looped forward, 48 KV+conv cache slots, \
                 GPU streaming dense load, packed arenas, packed prefill, exact-row graphs with #2697 soft-cap, exact 48-slot prefix cache)."
            }
        };
        Some(Hint {
            what: format!("Inkling 2 Mini {} execution is planned (#2701)", component.label()),
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

    fn activation(&self, _layer: usize, _shared: bool) -> Option<cuteafd_core::FfnActivation> {
        Some(self.cfg.activation())
    }

    fn contract_admission(&self) -> bool {
        true
    }

    fn native_serving_admission(&self) -> bool {
        true
    }

    fn accepts(&self, role: &TensorRole, stem: &str, operand: &mut QuantOperand) -> Result<(), String> {
        if role.component == Component::Speculator {
            let name = if self.draft_shapes.contains_key(stem) {
                stem.to_owned()
            } else {
                format!("{stem}.weight")
            };
            let shape = self
                .draft_shapes
                .get(&name)
                .ok_or_else(|| format!("{stem}: outside the Inkling 2 Mini recurrent drafter tensor contract"))?;
            let bias = self.draft_shapes.contains_key(&format!("{stem}.bias"));
            return super::require(
                operand.logical == *shape
                    && operand.is_plain(&[format::Encoding::Bf16])
                    && operand.input_scale.is_none()
                    && operand.companions == if bias { vec!["bias"] } else { Vec::<&str>::new() },
                || format!("{stem}: expected recurrent BF16 {shape:?}, found {}", super::describe(operand)),
            );
        }

        let contract = self.contract(stem).ok_or_else(|| {
            format!(
                "{stem}: outside the Inkling 2 Mini tensor contract ({})",
                super::describe(operand)
            )
        })?;
        super::require(
            contract.role.component == role.component && contract.role.layer == role.layer,
            || {
                format!(
                    "{stem}: classified {:?}, the contract says {:?}",
                    role.component, contract.role.component
                )
            },
        )?;

        let companions = operand.companions.iter().map(String::as_str).collect::<Vec<_>>();
        match contract.storage {
            Storage::Bf16 => super::require(
                operand.logical == contract.shape
                    && operand.is_plain(&[format::Encoding::Bf16])
                    && operand.input_scale.is_none()
                    && companions == contract.companions,
                || {
                    format!(
                        "{stem}: expected BF16 {:?}{}, found {}",
                        contract.shape,
                        if contract.companions.is_empty() {
                            String::new()
                        } else {
                            format!(" with {:?}", contract.companions)
                        },
                        super::describe(operand)
                    )
                },
            ),
            Storage::F32 => super::require(
                operand.logical == contract.shape
                    && operand.is_plain(&[format::Encoding::F32])
                    && operand.input_scale.is_none()
                    && companions == contract.companions,
                || format!("{stem}: expected F32 {:?}, found {}", contract.shape, super::describe(operand)),
            ),
            Storage::Experts => {
                let layer = contract.role.layer.expect("routed experts are per layer");
                let [experts, n, k] = contract.shape[..] else {
                    unreachable!("fused expert contract is [E, N, K]")
                };
                match self.expert_storage[layer] {
                    ExpertStorage::Mxfp8 => super::require(
                        operand.logical == contract.shape
                            && operand.is_fp8_row_groups(32, &[ScaleEncoding::Ue8m0])
                            && operand.global_scale.is_none()
                            && operand.input_scale.is_none(),
                        || {
                            format!(
                                "{stem}: expected MXFP8 E4M3 [{experts}, {n}, {k}] with UE8M0 scales per 32, found {}",
                                super::describe(operand)
                            )
                        },
                    ),
                    ExpertStorage::Nvfp4 => super::require(
                        operand.logical == contract.shape && operand.is_inkling_nvfp4(),
                        || {
                            format!(
                                "{stem}: expected native NVFP4 [{experts}, {n}, {k}] (E2M1 U8, E4M3 g16 scales, F32 amax + denom), found {}",
                                super::describe(operand)
                            )
                        },
                    ),
                    ExpertStorage::Exl3 { bits } => Err(format!(
                        "{stem}: Inkling 2 Mini layer {layer} EXL3 K{bits} is unsupported; use native NVFP4/MXFP8"
                    )),
                }
            }
        }
    }

    fn experts(&self, operand: &QuantOperand) -> Option<ExpertContract> {
        let (package, block) = if operand.is_fp8_row_groups(32, &[ScaleEncoding::Ue8m0]) {
            ("inkling2_mini:mxfp8".to_owned(), 32)
        } else if operand.is_inkling_nvfp4() {
            ("inkling2_mini:nvfp4a4".to_owned(), 16)
        } else {
            return None;
        };
        Some(ExpertContract {
            spark_worlds: Vec::new(),
            local: Ok(format!(
                "Inkling 2 Mini TP1 RTX local package ({package}; 0 Sparks)"
            )),
            package,
            block,
        })
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
