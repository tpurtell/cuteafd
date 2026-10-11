//! Qwen 3.8 Flash Next (qwen4_exp): Gated DeltaNet linear attention with a
//! full-attention layer every fourth (with an indexer), fused expert tensors,
//! shared expert with a sigmoid gate, hyper-connections and PLE n-gram tables.
use super::{bf16, bf16_or_f32, describe, leaf, require};
use crate::families::qwen4::{Qwen4Attention, Qwen4Config};
use crate::plan::checkpoint::Checkpoint;
use crate::plan::experts::{exl3_spark_worlds, fp8_spark_worlds, nvfp4_spark_worlds};
use crate::plan::family::{ConfigError, ExpertContract, Family, FamilyModel, Hint, RuntimeStatus};
use crate::plan::format::{Encoding, QuantOperand, ScaleEncoding};
use crate::plan::names::indexed;
use crate::plan::spec::*;

pub struct Qwen;
pub static QWEN4: Qwen = Qwen;

impl Family for Qwen {
    fn id(&self) -> &'static str {
        "qwen4"
    }
    fn runtime(&self) -> RuntimeStatus {
        RuntimeStatus::Serving
    }

    fn expert_catalog(&self) -> bool {
        true
    }

    fn optional(&self, component: Component) -> bool {
        // Text serving runs without the native MTP layer and the vision tower.
        matches!(component, Component::Speculator | Component::SpeculatorExpert | Component::Vision)
    }
    fn detect(&self, checkpoint: &Checkpoint) -> bool {
        checkpoint
            .architectures()
            .iter()
            .any(|arch| arch == "Qwen4ExpForConditionalGeneration" || arch == "Qwen4ExpForCausalLM")
    }

    fn open(&self, checkpoint: &Checkpoint) -> Result<Box<dyn FamilyModel>, ConfigError> {
        let cfg = Qwen4Config::from_hf(&checkpoint.config).map_err(ConfigError::from_anyhow)?;
        let layers = (0..cfg.layers)
            .map(|layer| match cfg.attention[layer] {
                Qwen4Attention::Gdn => LayerSpec { attention: AttentionKind::GatedDeltaNet, ffn: FfnKind::Moe, rope: None },
                Qwen4Attention::Full => LayerSpec {
                    attention: AttentionKind::Gqa { heads: cfg.heads, kv_heads: cfg.kv_heads, head_dim: cfg.head_dim },
                    ffn: FfnKind::Moe,
                    rope: Some(RopeSpec { dims: cfg.rope_dim, theta: cfg.rope_theta }),
                },
            })
            .collect();
        let spec = ModelSpec {
            family: "qwen4",
            architecture: checkpoint.architectures().first().cloned().unwrap_or_default(),
            hidden: cfg.hidden,
            vocab: cfg.vocab_size,
            layers,
            moe: Some(MoeSpec {
                experts: cfg.experts,
                top_k: cfg.topk,
                intermediate: cfg.moe_intermediate,
                shared_experts: 1,
                shared_intermediate: cfg.shared_intermediate,
                scoring: "softmax".into(),
                routed_scaling: None,
                groups: None,
            }),
            speculator: (cfg.mtp_layers > 0).then_some(SpeculatorSpec::NativeMtp { layers: cfg.mtp_layers }),
            tables: if cfg.ple_layers.is_empty() {
                Vec::new()
            } else {
                vec![MappedTableSpec { name: "ple-ngram".into(), layers: cfg.ple_layers.clone() }]
            },
            vision: checkpoint.config.get("vision_config").is_some(),
            notes: {
                let mut notes = vec![format!(
                    "hyper-connections {} (low rank {}), indexer budget {}",
                    cfg.hc_count, cfg.hc_lowrank, cfg.index_budget
                )];
                notes.extend(representation_notes(&cfg));
                notes
            },
        };
        let programs = cfg.check_programs().map_err(|e| format!("{e:#}"));
        let vision = vision_geometry(&checkpoint.config);
        Ok(Box::new(QwenModel { cfg, spec, programs, vision }))
    }

    fn classify(&self, _spec: &ModelSpec, name: &str) -> Option<TensorRole> {
        use Component::*;
        if name == "lm_head.weight" {
            return Some(TensorRole::new(LmHead));
        }
        if name.starts_with("mtp.") {
            if name.contains(".mlp.experts.") {
                return Some(TensorRole::new(SpeculatorExpert));
            }
            return Some(TensorRole::new(Speculator));
        }
        if name.starts_with("model.visual.") || name.starts_with("visual.") {
            return Some(TensorRole::new(Vision));
        }
        let name = name.strip_prefix("model.language_model.")?;
        match name {
            "embed_tokens.weight" => return Some(TensorRole::new(Embedding)),
            "norm.weight" => return Some(TensorRole::new(Norm)),
            _ => {}
        }
        if name.starts_with("hyper_connection_mixer.") {
            return Some(TensorRole::new(HyperConnection));
        }
        let (layer, rest) = indexed(name, "layers.")?;
        let component = if rest.starts_with("mlp.experts.") {
            // Fused [experts, ...] tensors: one tensor per projection per layer.
            RoutedExpert
        } else if rest.starts_with("ple.ple_embedding.") {
            MappedTable
        } else if rest.starts_with("ple.") {
            TableProjection
        } else if rest.starts_with("self_attn.indexer.") {
            Indexer
        } else if rest.starts_with("self_attn.") || rest.starts_with("linear_attn.") {
            Attention
        } else if rest.starts_with("mlp.gate.") {
            Router
        } else if rest.starts_with("mlp.shared_expert") {
            SharedExpert
        } else if rest.contains("hyper_connection.") {
            HyperConnection
        } else if rest.ends_with("layernorm.weight") || rest.ends_with("norm.weight") {
            Norm
        } else {
            Other
        };
        Some(TensorRole::layer(component, layer))
    }

    fn component_hint(&self, component: Component) -> Option<Hint> {
        let (what, how) = match component {
            Component::RoutedExpert => (
                "512 experts top-10 (hidden 2560, intermediate 640, SiLU unclamped) in a format without a \
                 routed kernel family"
                    .to_string(),
                "Executable: EXL3 K4/K5 (qwen4:exl3-k45, python/tools/aot/package_exl3_aot.py --geometry qwen4; \
                 exl3_cross_sm121.py for Sparks) and FP8 128x128 blocks (qwen4:fp8, native/cmake/shared/fp8_moe.cmake). \
                 BF16 fused [512, 1280, 2560] experts: serve the FP8 or EXL3 K4.25 publication, or add a BF16 \
                 routed family. ModelOpt NVFP4 (E2M1 + E4M3 per-16 scales + FP32 global) runs W4A16 as qwen4:nvfp4 \
                 (fp8_moe weights=nvfp4)."
                    .to_string(),
            ),
            Component::Speculator | Component::SpeculatorExpert => (
                "native MTP layer (full attention + 512 experts, hyper-connection feedback) is not run".to_string(),
                "serve-qwen4 verifies copy-window drafts only; MTP needs the mtp.* weights as one more qwen4 \
                 layer fed by fc_hidden/fc_embedding (see ../qflashrt quantization/qwen_model.py mtp_feedback) \
                 and its experts served (FP8/EXL3 packages)."
                    .to_string(),
            ),
            Component::Vision => (
                "Qwen resident BF16 ViT with native-id M-RoPE".to_string(),
                format!("27 blocks, H1152, 16 heads of 72, I4304, patch16x16x2, merge2; output H2560. \
                    Explicit VISION=auto/rtx/spark admits up to {} image tokens (detail=low 256); LM qualification remains required.",
                    crate::media::QWEN_MAX_IMAGE_TOKENS),
            ),
            Component::MappedTable | Component::TableProjection => (
                "PLE n-gram table in a format other than BF16 or E4M3 with one scale".to_string(),
                "The qwen4_ple_{bf16,fp8} programs gather 16 rows of 160 per token from a pinned host (or GPU) \
                 table; add a gather variant for the new format."
                    .to_string(),
            ),
            _ => (
                format!("{} tensors in a format other than BF16", component.label()),
                "The qwen4 coordinator programs take BF16 weights; dequantize at load (weights.rs) or add \
                 an FP8 program variant."
                    .to_string(),
            ),
        };
        Some(Hint { what, how })
    }
}

const GIB: f64 = (1u64 << 30) as f64;

/// The selectable resident representations (serve-qwen4 `--fp8-decode`, `--mtp-fp8-head`):
/// one format per weight, never BF16 plus an FP8 copy.
fn representation_notes(cfg: &Qwen4Config) -> Vec<String> {
    use crate::families::qwen4::resident::{resident_bytes, Qwen4Representation};

    let mtp = cfg.mtp_layers > 0;
    let all = resident_bytes(cfg, cfg.layers, mtp, Qwen4Representation { fp8_projections: true, fp8_head: true });
    let gib = |bytes: usize| bytes as f64 / GIB;
    vec![
        format!("GDN/attention in-out projections ({} layers{}): checkpoint BF16 {:.2} GiB resident by default; \
            --fp8-decode true holds them as E4M3 + FP32 128x128 block scales only ({:.2} GiB; qwen4_*_w8 programs: \
            16-row GEMV, W8A16 above and in prefill, --fp8-prefill-w8a8 for E4M3 activations), converting one \
            BF16 matrix at a time (<= {:.0} MiB device staging)",
            cfg.layers, if mtp { " + MTP" } else { "" }, gib(all.projections_bf16), gib(all.projections_fp8),
            all.max_device_staging as f64 / (1u64 << 20) as f64),
        format!("lm_head (shared by target and MTP drafts): BF16 {:.2} GiB by default; --mtp-fp8-head true holds \
            one E4M3 head with per-row x 128-K scales ({:.2} GiB, quantized on the host) and runs every logits \
            row through qwen4_head_fp8 in 16-row spans", gib(all.head_bf16), gib(all.head_fp8)),
    ]
}

struct QwenModel {
    cfg: Qwen4Config,
    spec: ModelSpec,
    /// `Qwen4Config::check_programs`: the shapes the qwen4 programs are built for.
    programs: Result<(), String>,
    vision: Result<(), String>,
}

impl FamilyModel for QwenModel {
    fn spec(&self) -> &ModelSpec {
        &self.spec
    }

    fn cache_geometry(&self, options: crate::serving_capacity::CacheOptions)
        -> Result<Option<crate::serving_capacity::FamilyCacheGeometry>, crate::serving_capacity::CacheGeometryError> {
        use crate::serving_capacity::{qwen_cache_geometry, CacheGeometryError};
        if options.coordinator_ranks != 1 || options.native_mtp_layers > 1 {
            return Err(CacheGeometryError::Unsupported { family: "qwen4", what: "only one coordinator and at most one native MTP layer execute" });
        }
        qwen_cache_geometry(&self.cfg, self.cfg.layers, options.native_mtp_layers == 1, options.qwen_kv).map(Some)
    }

    fn accepts(&self, role: &TensorRole, stem: &str, operand: &mut QuantOperand) -> Result<(), String> {
        let name = leaf(stem);
        match role.component {
            Component::Speculator | Component::SpeculatorExpert => {
                return Err("the MTP drafter is not part of the plain serve path".into());
            }
            Component::Vision => {
                self.vision.clone()?;
                let shape = vision_shape(stem).ok_or_else(|| format!("{stem} is not read by the resident Qwen tower"))?;
                return require(operand.is_plain(&[Encoding::Bf16]) && operand.logical == shape, ||
                    format!("{stem} needs BF16 {shape:?}, found {}", describe(operand)));
            },
            _ => {}
        }
        self.programs.clone()?;
        match role.component {
            Component::RoutedExpert => {
                let moe = self.spec.moe.as_ref().ok_or("no MoE geometry")?;
                let shape = super::deepseek::routed_shape(stem, self.spec.hidden, moe.intermediate)
                    .ok_or("not a routed projection (gate/up/down_proj)")?;
                require(operand.logical == shape, || format!("experts are {shape:?}, found {}", describe(operand)))?;
                require(operand.is_fp8_block(128, &[ScaleEncoding::F32, ScaleEncoding::Bf16])
                    || matches!(operand.exl3_bits(), Some(4..=5))
                    || operand.is_nvfp4() && operand.scale.as_ref().is_some_and(|s| s.cols == 16), || {
                    format!("routed experts run EXL3 K4/K5 (qwen4:exl3-k45), E4M3 with 128x128 scales (qwen4:fp8) or \
                        ModelOpt NVFP4 group 16 (qwen4:nvfp4); found {}", describe(operand))
                })
            }
            // The n-gram table: BF16 or E4M3 shards (`ngram_embedding.shard_*`)
            // sharing one table scale (`ngram_embedding.weight_scale`), and its
            // integer hash parameters.
            Component::MappedTable => require(
                operand.is_plain(&[Encoding::Bf16, Encoding::E4m3, Encoding::F32])
                    || matches!(operand.encoding, Encoding::Int { .. }),
                || format!("PLE table data must be BF16, unscaled E4M3 shards, FP32 or integer; found {}", describe(operand)),
            ),
            _ if matches!(name, "conv1d" | "A_log" | "dt_bias") => bf16_or_f32(operand, name),
            _ => bf16(operand, "qwen4 coordinator tensors"),
        }
    }

    fn experts(&self, operand: &QuantOperand) -> Option<ExpertContract> {
        let moe = self.spec.moe.as_ref()?;
        let geometry = cuteafd_core::ExpertGeometry {
            hidden: self.spec.hidden as u32,
            experts: moe.experts as u32,
            topk: moe.top_k as u32,
            intermediate: moe.intermediate as u32,
            layers: 0,
        };
        if !geometry.same_shape(&cuteafd_core::ExpertGeometry::QWEN4) {
            return None;
        }
        if operand.exl3_bits().is_some() {
            return Some(ExpertContract {
                package: "qwen4:exl3-k45".into(),
                block: 128,
                spark_worlds: std::iter::once(1).chain(exl3_spark_worlds(moe.intermediate)).collect(),
                local: Ok("serve-qwen4 --local-experts (TP1 EXL3 package)".into()),
            });
        }
        if operand.is_nvfp4() {
            return Some(ExpertContract {
                package: "qwen4:nvfp4".into(),
                block: 16,
                spark_worlds: nvfp4_spark_worlds(moe.intermediate),
                local: Ok("serve-qwen4 --local-experts (fp8-qwen4-nvfp4 tp1)".into()),
            });
        }
        operand.is_fp8_block(128, &[ScaleEncoding::F32, ScaleEncoding::Bf16]).then(|| ExpertContract {
            package: "qwen4:fp8".into(),
            block: 128,
            spark_worlds: fp8_spark_worlds(moe.intermediate),
            local: Ok("serve-qwen4 --local-experts (fp8-qwen4 tp1)".into()),
        })
    }
}

fn vision_geometry(config: &serde_json::Value) -> Result<(), String> {
    let v = config.get("vision_config").ok_or("checkpoint has no vision_config")?;
    for (name, expected) in [("depth", 27), ("hidden_size", 1152), ("intermediate_size", 4304),
        ("num_heads", 16), ("num_position_embeddings", 2304), ("out_hidden_size", 2560),
        ("patch_size", 16), ("temporal_patch_size", 2), ("spatial_merge_size", 2)] {
        require(v[name].as_u64() == Some(expected), ||
            format!("vision_config.{name} needs {expected}; add a Qwen tower exporter/kernel"))?;
    }
    require(config["model_type"] == "qwen4_exp" && config["text_config"]["hidden_size"] == 2560
        && config["text_config"]["hc_count"] == 4 && v["hidden_act"] == "gelu_pytorch_tanh"
        && v["deepstack_visual_indexes"].as_array().is_some_and(Vec::is_empty), ||
        "Qwen tower config/merger geometry unsupported".into())
}

fn vision_shape(stem: &str) -> Option<Vec<usize>> {
    let name = stem.strip_prefix("model.visual.").or_else(|| stem.strip_prefix("visual."))?;
    let shape: &[usize] = match name {
        "patch_embed.proj" => &[1152, 3, 2, 16, 16],
        "pos_embed" => &[2304, 1152],
        "merger.norm" => &[1152],
        "merger.linear_fc1" => &[4608, 4608],
        "merger.linear_fc2" => &[2560, 4608],
        _ => {
            let (block, rest) = indexed(name, "blocks.")?;
            if block >= 27 { return None; }
            match rest {
                "attn.qkv" => &[3456, 1152],
                "attn.proj" => &[1152, 1152],
                "mlp.linear_fc1" => &[4304, 1152],
                "mlp.linear_fc2" => &[1152, 4304],
                "norm1" | "norm2" => &[1152],
                _ => return None,
            }
        }
    };
    Some(shape.to_vec())
}
