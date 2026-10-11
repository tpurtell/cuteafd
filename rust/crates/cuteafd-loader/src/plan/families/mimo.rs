//! Xiaomi MiMo V2: hybrid full / sliding-window GQA with attention sinks,
//! sigmoid top-8 experts without shared experts, dense-FFN MTP layers.
//!
//! serve-mimo runs MiMo checkpoints: V2 Flash on the `mimo` programs with the
//! checkpoint's FP8 experts (`mimo:fp8`, Spark tp2/tp4/tp6, coordinator tp1)
//! and V2.6 Pro on the `mimop` programs with MXFP4 experts (`mimop:fp8`,
//! Spark tp2/tp6, coordinator tp1); V2.6 Flash MOPD uses `mimof` and MXFP4
//! experts (`mimof:fp8`, Spark tp2/tp4). The plan reads the configuration with the
//! runtime's own reader (`MimoV2Config`) and checks every tensor against what
//! `MimoLoader` stages.
use super::deepseek::routed_shape;
use super::{bf16, bf16_or_fp8_block128, describe, leaf, require};
use crate::families::mimo_v2::{checkpoint_tp, FusedQkvLayout, MimoAttention, MimoV2Config};
use crate::plan::checkpoint::Checkpoint;
use crate::plan::experts::spark_worlds;
use crate::plan::family::{ConfigError, ExpertContract, Family, FamilyModel, Hint, RuntimeStatus};
use crate::plan::format::{Encoding, QuantOperand, RowTiling, ScaleEncoding};
use crate::plan::names::indexed;
use crate::plan::spec::*;

const GIB: f64 = (1u64 << 30) as f64;

pub struct MiMo;
pub static MIMO_V2: MiMo = MiMo;

const ARCHITECTURES: &[&str] = &["MiMoV2FlashForCausalLM", "MiMoV2ForCausalLM"];

/// The model spec of a MiMo configuration, from the runtime reader: one
/// layer per `cfg.attention` / `cfg.dense` entry, with the sinks, window and
/// RoPE serve-mimo uses.
pub fn spec_from(cfg: &MimoV2Config, checkpoint: &Checkpoint) -> ModelSpec {
    let layers = (0..cfg.layers)
        .map(|layer| {
            let attention = cfg.attention[layer];
            LayerSpec {
                attention: match attention {
                    MimoAttention::Full => {
                        AttentionKind::Gqa { heads: cfg.heads, kv_heads: cfg.full_kv_heads, head_dim: cfg.head_dim }
                    }
                    MimoAttention::Sliding => AttentionKind::SlidingGqa {
                        heads: cfg.heads,
                        kv_heads: cfg.swa_kv_heads,
                        head_dim: cfg.head_dim,
                        window: cfg.window,
                        sinks: cfg.swa_sinks,
                    },
                },
                ffn: if cfg.dense[layer] {
                    FfnKind::Dense { intermediate: cfg.dense_intermediate }
                } else {
                    FfnKind::Moe
                },
                rope: Some(RopeSpec { dims: cfg.rope_dim, theta: cfg.rope_theta(attention) }),
            }
        })
        .collect();
    let mtp = checkpoint
        .tensor_names()
        .filter_map(|name| indexed(name, "model.mtp.layers.").map(|(i, _)| i))
        .max()
        .map_or(0, |max| max + 1);
    let mut notes = vec![format!(
        "rotary {} of {} dims (theta full {}, SWA {}), attention value scale {}, sinks full {} / SWA {}",
        cfg.rope_dim, cfg.head_dim, cfg.full_rope_theta, cfg.swa_rope_theta, cfg.v_scale, cfg.full_sinks, cfg.swa_sinks
    )];
    if checkpoint.tensor_names().any(|name| name.ends_with("self_attn.qkv_proj.weight")) {
        let tp = checkpoint_tp(&checkpoint.snapshot).map_or("?".to_string(), |tp| tp.to_string());
        notes.push(format!("fused qkv_proj stored TP{tp}-interleaved ([q|k|v] per row shard, own 128x128 grid per \
            shard); the engine de-interleaves it (FusedQkvLayout)"));
    }
    if let Some(router) = checkpoint.tensors.iter().find(|t| t.meta.name.ends_with("mlp.gate.weight")) {
        notes.push(format!("router weight {:?}", router.meta.dtype));
    }
    if checkpoint.snapshot.join("dflash").join("config.json").exists() {
        notes.push("dflash/: DFlash block drafter (serve-mimo --draft; not planned: its own qwen3-style config)".into());
    }
    if let Ok(family) = cfg.program_family() {
        notes.push(format!("coordinator programs: family {family} (CUTEAFD_ENABLE_MIMO_AOT, \
            CUTEAFD_MIMO_GEOMETRIES={family})"));
    }
    if crate::families::mimo_v2::weight_policy::default_policy(checkpoint, cfg)
        == crate::families::mimo_v2::weight_policy::MimoDefaultPolicy::Fp8 {
        notes.push("measured MiMo default: single-copy FP8 target head/O and DFlash drafter; native QKV/FFN unchanged. --weight-policy checkpoint keeps source formats; explicit per-weight flags override.".into());
        if let Ok(memory) = crate::families::mimo_v2::weight_policy::qualified_projection_memory(cfg) {
            notes.push(format!("default target head/O across coordinator ranks: checkpoint source {} B ({:.3} GiB), selected FP8 resident {} B ({:.3} GiB), maximum drained packing source {} B ({:.3} GiB); these costs use the runtime projection descriptor and exclude other weights, optional DFlash and runtime state",
                memory.source_bytes, memory.source_bytes as f64 / GIB,
                memory.resident_bytes, memory.resident_bytes as f64 / GIB,
                memory.max_load_staging, memory.max_load_staging as f64 / GIB));
        }
    } else {
        notes.push("default resident formats follow checkpoint tensors; explicit single-copy head/O/drafter conversions remain configurable".into());
    }
    notes.push("component bytes, owner totals and weight-only placement budgets below describe checkpoint source storage; they are not complete resident-memory admission. Runtime admission counts selected representations, loading phases, optional DFlash, caches and workspaces separately".into());
    ModelSpec {
        family: "mimo_v2",
        architecture: checkpoint.architectures().first().cloned().unwrap_or_default(),
        hidden: cfg.hidden,
        vocab: cfg.vocab_size,
        layers,
        moe: Some(MoeSpec {
            experts: cfg.experts,
            top_k: cfg.topk,
            intermediate: cfg.moe_intermediate,
            shared_experts: 0,
            shared_intermediate: 0,
            scoring: "sigmoid".into(),
            routed_scaling: Some(cfg.routed_scale),
            groups: None,
        }),
        speculator: (mtp > 0).then_some(SpeculatorSpec::NativeMtp { layers: mtp }),
        tables: Vec::new(),
        vision: checkpoint.config.get("vision_config").is_some(),
        notes,
    }
}

impl Family for MiMo {
    fn id(&self) -> &'static str {
        "mimo_v2"
    }
    fn runtime(&self) -> RuntimeStatus {
        RuntimeStatus::Serving
    }
    fn detect(&self, checkpoint: &Checkpoint) -> bool {
        checkpoint.architectures().iter().any(|arch| ARCHITECTURES.contains(&arch.as_str()))
    }

    fn open(&self, checkpoint: &Checkpoint) -> Result<Box<dyn FamilyModel>, ConfigError> {
        let cfg = MimoV2Config::from_hf(&checkpoint.config).map_err(ConfigError::from_anyhow)?;
        let spec = spec_from(&cfg, checkpoint);
        // What serve-mimo checks before it stages anything: the program
        // geometry and sinks on SWA layers only (`MimoLoader::block`).
        let programs = cfg.program_family().map_err(|e| format!("{e:#}")).and_then(|family| {
            require(!cfg.full_sinks && cfg.swa_sinks, || format!("the {family} programs take sinks on SWA layers \
                only (config: full {} / SWA {})", cfg.full_sinks, cfg.swa_sinks))?;
            Ok(family)
        });
        let checkpoint_tp = checkpoint_tp(&checkpoint.snapshot).map_err(|e| format!("{e:#}"));
        let vision = vision_geometry(&checkpoint.config, cfg.hidden);
        let audio = if checkpoint.tensors.is_empty() {
            Err("audio headers have not been read for this role".into())
        } else {
            crate::media::audio_tower::AudioTowerPlan::from_snapshot(&checkpoint.snapshot,
                crate::media::audio_tower::AudioStorage::Fp32).map_err(|e| e.to_string())
        };
        Ok(Box::new(MimoModel { cfg, spec, programs, checkpoint_tp, vision, audio }))
    }

    fn expert_catalog(&self) -> bool {
        true
    }

    fn optional(&self, component: Component) -> bool {
        // Text serving runs without the MTP layers (serve-mimo --mtp runs them)
        // and without the vision and audio towers.
        matches!(component, Component::Speculator | Component::Vision | Component::Audio)
    }
    fn classify(&self, _spec: &ModelSpec, name: &str) -> Option<TensorRole> {
        use Component::*;
        match name {
            "lm_head.weight" => return Some(TensorRole::new(LmHead)),
            "model.embed_tokens.weight" => return Some(TensorRole::new(Embedding)),
            "model.norm.weight" => return Some(TensorRole::new(Norm)),
            _ => {}
        }
        if let Some((layer, _)) = indexed(name, "model.mtp.layers.") {
            return Some(TensorRole::layer(Speculator, layer));
        }
        if name.starts_with("audio") || name.starts_with("speech_") || name.starts_with("model.audio") {
            return Some(TensorRole::new(Audio));
        }
        if name.starts_with("visual.") || name.starts_with("model.visual.") {
            return Some(TensorRole::new(Vision));
        }
        let (layer, rest) = indexed(name, "model.layers.")?;
        if let Some((expert, _)) = indexed(rest, "mlp.experts.") {
            return Some(TensorRole::expert(RoutedExpert, layer, expert));
        }
        let component = if rest.starts_with("self_attn.") {
            Attention
        } else if rest.starts_with("mlp.gate.") {
            Router
        } else if rest.starts_with("mlp.") {
            DenseFfn
        } else if rest.ends_with("layernorm.weight") {
            Norm
        } else {
            Other
        };
        Some(TensorRole::layer(component, layer))
    }

    fn component_hint_for(&self, spec: &ModelSpec, component: Component, formats: &[String]) -> Option<Hint> {
        let moe = spec.moe.as_ref()?;
        let (h, i, e, k) = (spec.hidden, moe.intermediate, moe.experts, moe.top_k);
        let mxfp4 = formats.iter().any(|f| f.starts_with("mxfp4"));
        match component {
            Component::RoutedExpert if mxfp4 && h == 4096 => Some(Hint {
                what: format!("sigmoid top-{k} MXFP4 routed experts: H {h}, I {i}, {e} experts; no shared expert"),
                how: format!("Exact family `mimof:fp8` (E2M1 packed U8 [N,K/2], UE8M0 U8 [N,K/32]); \
                    fp8-mimof tp1 coordinator and tp2/tp4 Spark, package_fp8_moe_aot.py --geometry mimof. \
                    TP2 owns {} rows per rank; TP4 owns {}. SM121 uses MXFP8 x MXFP4 large-row \
                    prefill; FP8_EXPERT_PREFILL=w8a16 keeps BF16 down.", i / 2, i / 4),
            }),
            Component::RoutedExpert if mxfp4 => {
                // TP6: whole 32-blocks per rank (352/320 of 2048), stored zero-padded to 128.
                let widest = (i / 32).div_ceil(6) * 32;
                let padded = widest.div_ceil(128) * 128;
                let per_rank = |slice: usize| (spec.moe_layers() * e * 3 * slice * h) as f64 * (0.5 + 1.0 / 32.0) / GIB;
                Some(Hint {
                    what: format!("sigmoid top-{k} routed experts, no shared expert, MXFP4 (packed E2M1 U8 [N, K/2], \
                        even element low nibble, UE8M0 U8 [N, K/32]): H {h}, I {i}, {e} experts"),
                    how: format!("Exact family `mimop:fp8` (b12x fp8_moe weights=mxfp4: E2M1 x 2^(s-127); \
                        packages fp8-mimop tp1 coordinator, tp6/tp2 Spark, \
                        python/tools/aot/package_fp8_moe_aot.py --geometry mimop [--cross-sm121]). Spark layout TP6 \
                        over six ranks: whole 32-blocks per rank ({widest}/{} rows) zero-padded to {padded}, \
                        {:.1} GiB per rank (TP2xEP3: {:.1} GiB, but a decode step reads all of a row's experts \
                        that land on one EP group). The SM121 Spark package streams MXFP8 x MXFP4 gate/up \
                        above 640 live rows, and its down projection quantizes the BF16 SwiGLU rows per K32 to \
                        MXFP8 for block-scaled MXFP8 x MXFP4 MMAs (FP8_EXPERT_PREFILL=w8a16 keeps BF16 down); \
                        smaller row counts use the grouped route.",
                        widest - 32, per_rank(padded), per_rank(i / 2) / 3.0),
                })
            }
            Component::RoutedExpert => Some(Hint {
                what: format!("sigmoid top-{k} routed experts, no shared expert (H {h}, I {i}, {e} experts)"),
                how: format!("Exact family `mimo:fp8` (b12x fp8_moe over E4M3 + FP32 128x128 scales; packages \
                    fp8-mimo tp1/tp2/tp4/tp6) for H {h} / I {i} / {e} experts / top-{k}; TP6 owns whole 128-blocks \
                    ({}/{} rows) zero-padded to {}; UE8M0 requantization costs +0.011 nats mean NLL (V2 Flash), \
                    EXL3 is the compact alternative.",
                    (i / 128).div_ceil(6) * 128, i / 128 / 6 * 128, (i / 128).div_ceil(6) * 128),
            }),
            // Pro's router weight is BF16 (the bias FP32); Flash's is FP32.
            Component::Router if formats.iter().any(|f| f == "bf16") => Some(Hint {
                what: "sigmoid noaux_tc router with FP32 e_score_correction_bias, BF16 router weight".into(),
                how: "mimof/mimop_router_scores (BF16 weight, FP32 accumulation) then cuteafd_router_select \
                    (sigmoid, normalized, no routed scaling).".into(),
            }),
            _ => self.component_hint(component),
        }
    }

    fn component_hint(&self, component: Component) -> Option<Hint> {
        let (what, how) = match component {
            Component::Attention => (
                "GQA full attention plus 128-token sliding-window GQA with learned sink bias",
                "serve-mimo runs mimo (V2 Flash), mimof (V2.6 Flash MOPD) and mimop (V2.6 Pro) programs (b12x integration \
                 mimo_{full,swa}_{producer,attention}, mimo_full_*_kvint8, mimo_o): int8 full-attention KV \
                 records (FP32 scale per 32 dims; --kv-cache bf16 for BF16) in paged pools, BF16 SWA rings, \
                 sinks on SWA layers only, QK 192 / V 128, NeoX RoPE on 64 dims. MimoLoader stages BF16 or E4M3 \
                 with FP32 128x128 grids, restarting per 192-row head (Flash's full-layer k_proj) or per \
                 checkpoint TP shard (V2.6 fused qkv_proj, FusedQkvLayout); another layout needs a loader \
                 segment rule first.",
            ),
            Component::RoutedExpert => (
                "sigmoid top-8 routed experts, no shared expert",
                "Packages: mimo:fp8 (V2 Flash geometry, E4M3 + FP32 128x128 scales; Spark tp2/tp4/tp6, \
                 coordinator tp1) and mimop:fp8 (V2.6 Pro geometry, MXFP4: E2M1 U8 [N, K/2] + UE8M0 U8 [N, K/32]; \
                 Spark tp2/tp6, coordinator tp1). Another format or geometry needs an fp8_moe export \
                 (python/tools/aot/package_fp8_moe_aot.py --geometry) and Fp8ExpertTensors staging.",
            ),
            Component::Router => (
                "sigmoid noaux_tc router with e_score_correction_bias (router weight FP32 on Flash)",
                "mimo_router_scores (FP32 weight as BF16 hi + lo, FP32 sums) then cuteafd_router_select \
                 (sigmoid, normalized, no routed scaling).",
            ),
            Component::Vision => (
                "MiMo V2.6 resident BF16 ViT with 1-D LM positions and per-image spatial merger",
                "28 blocks, H1280, Q32/KV8, QK64, I4608, patch16x16x2, merge2; output H4096/H6144. \
                 The resident tower validates all headers before admitted weight/scratch allocation; \
                 serve-mimo --vision rtx[:gpu] enables the local path. LM image qualification remains required; audio/video stay disabled.",
            ),
            Component::Speculator => (
                "MTP layers (SWA attention with sinks, dense FFN, eh_proj/enorm/hnorm)",
                "serve-mimo --mtp N runs them on the SWA and dense FFN programs with the eh_proj fusion \
                 (BF16 [H, 2H]); serving runs without them.",
            ),
            _ => return None,
        };
        Some(Hint { what: what.into(), how: how.into() })
    }
}

struct MimoModel {
    cfg: MimoV2Config,
    spec: ModelSpec,
    /// The coordinator program family (`mimo`, `mimop`), or why none runs this geometry.
    programs: Result<&'static str, String>,
    /// The checkpoint's tensor-parallel degree (fused qkv row shards).
    checkpoint_tp: Result<usize, String>,
    vision: Result<(), String>,
    audio: Result<crate::media::audio_tower::AudioTowerPlan, String>,
}

fn vision_geometry(config: &serde_json::Value, hidden: usize) -> Result<(), String> {
    let v = config.get("vision_config").ok_or("checkpoint has no vision_config")?;
    for (name, expected) in [("depth", 28), ("hidden_size", 1280), ("intermediate_size", 4608),
        ("num_heads", 32), ("num_key_value_heads", 8), ("out_hidden_size", hidden),
        ("patch_size", 16), ("temporal_patch_size", 2), ("spatial_merge_size", 2)] {
        require(v[name].as_u64() == Some(expected as u64), ||
            format!("vision_config.{name} needs {expected}; add a tower exporter/kernel"))?;
    }
    require([4096, 6144].contains(&hidden) && v["hidden_act"] == "silu", ||
        "MiMo vision output/activation geometry unsupported".into())?;
    Ok(())
}

fn vision_shape(stem: &str, hidden: usize) -> Option<Vec<usize>> {
    let name = stem.strip_prefix("visual.")?;
    let shape: &[usize] = match name {
        "patch_embed.proj" => &[1280, 3, 2, 16, 16],
        "merger.ln_q" => &[1280],
        "merger.mlp.0" => &[5120, 5120],
        "merger.mlp.2" => return Some(vec![hidden, 5120]),
        _ => {
            let (block, rest) = indexed(name, "blocks.")?;
            if block >= 28 { return None; }
            match rest {
                "attn.qkv" => &[3072, 1280],
                "attn.qkv.bias" => &[3072],
                "attn.proj" => &[1280, 2048],
                "attn.proj.bias" => &[1280],
                "mlp.gate_proj" | "mlp.up_proj" => &[4608, 1280],
                "mlp.gate_proj.bias" | "mlp.up_proj.bias" => &[4608],
                "mlp.down_proj" => &[1280, 4608],
                "mlp.down_proj.bias" | "norm1" | "norm2" => &[1280],
                "attn.sinks" => &[32],
                _ => return None,
            }
        }
    };
    Some(shape.to_vec())
}

impl MimoModel {
    /// Which attention a layer's tensors serve: MTP layers are SWA layers.
    fn attention(&self, role: &TensorRole) -> MimoAttention {
        match (role.component, role.layer) {
            (Component::Speculator, _) => MimoAttention::Sliding,
            (_, Some(layer)) if layer < self.cfg.layers => self.cfg.attention[layer],
            _ => MimoAttention::Sliding,
        }
    }

    /// A projection `rows_fp8` stages: BF16, or E4M3 with FP32 scales in
    /// 128-column blocks whose rows are uniform 128-row blocks or, for
    /// `per_head` (a full layer's `k_proj`), 192-row heads of a 128- and a
    /// 64-row block. The decode copies need K % 128 == 0.
    fn projection(&self, name: &str, operand: &mut QuantOperand, per_head: bool) -> Result<(), String> {
        let (rows, cols) = operand.matrix().ok_or_else(|| format!("{name} must be a matrix, found {}", describe(operand)))?;
        require(cols % 128 == 0, || format!("{name}: FP8 decode copies need K % 128 == 0, found {}", describe(operand)))?;
        let unresolved = operand.scale.as_ref().is_some_and(|s| matches!(s.rows, RowTiling::Unresolved { .. }));
        if per_head && operand.encoding == Encoding::E4m3 && unresolved {
            let head = self.cfg.head_dim;
            require(rows % head == 0, || format!("{name}: {rows} rows are not whole {head}-row heads"))?;
            operand.resolve_segments(128, vec![head; rows / head]).map_err(|e| format!("{name}: {e}"))?;
            return require(operand.is_fp8_segmented(128, &[ScaleEncoding::F32]), || {
                format!("{name}: per-head blocks need FP32 scales, found {}", describe(operand))
            });
        }
        bf16_or_fp8_block128(operand, name)
    }

    /// V2.6 Pro's fused `qkv_proj`: E4M3 row shards `[q | k | v]` per
    /// checkpoint TP rank, each with its own 128x128 FP32 grid
    /// (`MimoLoader::fused_qkv`, `FusedQkvLayout`).
    fn fused_qkv(&self, attention: MimoAttention, operand: &mut QuantOperand) -> Result<(), String> {
        let tp = self.checkpoint_tp.clone().map_err(|e| format!("fused qkv_proj needs the index's tp_size: {e}"))?;
        let layout = FusedQkvLayout::new(&self.cfg, attention, tp).map_err(|e| format!("{e:#}"))?;
        let expected = [layout.rows(), self.cfg.hidden];
        require(operand.encoding == Encoding::E4m3 && operand.logical == expected, || {
            format!("fused qkv_proj for checkpoint TP{tp} is E4M3 {expected:?} with FP32 [{}, {}] scales, found {}",
                layout.scale_rows(), self.cfg.hidden.div_ceil(128), describe(operand))
        })?;
        let segments = vec![layout.q + layout.k + layout.v; layout.shards];
        operand.resolve_segments(128, segments).map_err(|e| format!("fused qkv_proj (TP{tp}): {e}"))?;
        require(operand.is_fp8_segmented(128, &[ScaleEncoding::F32]), || {
            format!("fused qkv_proj needs FP32 scales, found {}", describe(operand))
        })?;
        layout.program_segments(&self.cfg).map(|_| ()).map_err(|e| format!("{e:#}"))
    }

    fn routed(&self, family: &str, stem: &str, operand: &QuantOperand) -> Result<(), String> {
        let shape = routed_shape(stem, self.cfg.hidden, self.cfg.moe_intermediate)
            .ok_or("not a routed projection (gate/up/down_proj)")?;
        require(operand.logical == shape, || format!("experts are {shape:?}, found {}", describe(operand)))?;
        match family {
            // `Fp8ExpertTensors::check`: E4M3 with FP32 (or BF16) 128x128 grids.
            "mimo" => require(operand.is_fp8_block(128, &[ScaleEncoding::F32, ScaleEncoding::Bf16]), || {
                format!("mimo:fp8 runs E4M3 experts with 128x128 FP32 scales; found {}", describe(operand))
            }),
            // MXFP4: packed E2M1 U8 [N, K/2] with UE8M0 [N, K/32].
            _ => require(operand.is_mxfp4(32), || {
                format!("{family}:fp8 runs MXFP4 experts (E2M1 + UE8M0 per 32); found {}", describe(operand))
            }),
        }
    }

    fn geometry(&self) -> Option<&'static str> {
        let geometry = cuteafd_core::ExpertGeometry {
            hidden: self.cfg.hidden as u32,
            experts: self.cfg.experts as u32,
            topk: self.cfg.topk as u32,
            intermediate: self.cfg.moe_intermediate as u32,
            layers: 0,
        };
        [(cuteafd_core::ExpertGeometry::MIMO_V2_FLASH,
            if self.cfg.program_family().ok() == Some("mimof") { "mimof" } else { "mimo" }),
            (cuteafd_core::ExpertGeometry::MIMO_V26_PRO, "mimop")]
            .into_iter()
            .find_map(|(shape, family)| geometry.same_shape(&shape).then_some(family))
    }
}

impl FamilyModel for MimoModel {
    fn spec(&self) -> &ModelSpec {
        &self.spec
    }

    fn cache_geometry(&self, options: crate::serving_capacity::CacheOptions)
        -> Result<Option<crate::serving_capacity::FamilyCacheGeometry>, crate::serving_capacity::CacheGeometryError> {
        use crate::serving_capacity::{mimo_cache_geometry, CacheGeometryError};
        let available = match &self.spec.speculator {
            Some(SpeculatorSpec::NativeMtp { layers }) => *layers,
            _ => 0,
        };
        if options.native_mtp_layers > available {
            return Err(CacheGeometryError::Unsupported { family: "mimo_v2", what: "requested native MTP stages exceed checkpoint tensors" });
        }
        mimo_cache_geometry(&self.cfg, self.cfg.layers, options.coordinator_ranks, options.mimo_kv,
            options.native_mtp_layers).map(Some)
    }

    fn accepts(&self, role: &TensorRole, stem: &str, operand: &mut QuantOperand) -> Result<(), String> {
        if role.component == Component::Audio {
            let plan = self.audio.as_ref().map_err(Clone::clone)?;
            let read = plan.reads().iter().find(|(name, _)| name.strip_suffix(".weight").unwrap_or(name) == stem)
                .map(|(_, read)| read).ok_or_else(|| format!("{stem} is not read by the bundled MiMo audio tower"))?;
            return require(operand.is_plain(&[Encoding::Bf16, Encoding::F32]) && operand.logical == read.metadata.shape, ||
                format!("{stem} needs plain BF16/FP32 {:?}; found {}", read.metadata.shape, describe(operand)));
        }
        if role.component == Component::Vision {
            self.vision.clone()?;
            let shape = vision_shape(stem, self.cfg.hidden).ok_or_else(||
                format!("{stem} is not read by the resident MiMo tower"))?;
            return require(operand.is_plain(&[Encoding::Bf16]) && operand.logical == shape, ||
                format!("{stem} needs BF16 {shape:?}, found {}", describe(operand)));
        }
        let family = self.programs.clone()?;
        let name = leaf(stem);
        let attention = self.attention(role);
        match (role.component, name) {
            (Component::RoutedExpert, _) => {
                let geometry = self.geometry().ok_or_else(|| format!("no MiMo expert package at hidden {} / \
                    intermediate {} / {} experts / top-{}", self.cfg.hidden, self.cfg.moe_intermediate,
                    self.cfg.experts, self.cfg.topk))?;
                require(geometry == family, || format!("{geometry} experts beside {family} programs"))?;
                self.routed(family, stem, operand)
            }
            (Component::Embedding, _) => bf16(operand, "the embedding"),
            (Component::LmHead, _) => {
                require(operand.matrix().is_some_and(|(_, k)| k % 128 == 0), || {
                    format!("the LM head's FP8 copy needs K % 128 == 0, found {}", describe(operand))
                })?;
                bf16_or_fp8_block128(operand, "lm_head")
            }
            (Component::Attention | Component::Speculator, "qkv_proj") => self.fused_qkv(attention, operand),
            (Component::Attention | Component::Speculator, "k_proj") => {
                self.projection(name, operand, attention == MimoAttention::Full)
            }
            (Component::Attention | Component::Speculator | Component::DenseFfn,
                "q_proj" | "v_proj" | "o_proj" | "gate_proj" | "up_proj" | "down_proj") => {
                self.projection(name, operand, false)
            }
            (Component::Attention | Component::Speculator, "attention_sink_bias") => {
                require(attention == MimoAttention::Sliding, || "sinks on a full-attention layer: the mimo programs \
                    take sinks on SWA layers only".into())?;
                bf16(operand, "attention_sink_bias")
            }
            (Component::Speculator, "eh_proj") => {
                bf16(operand, "eh_proj")?;
                let expected = [self.cfg.hidden, 2 * self.cfg.hidden];
                require(operand.logical == expected, || format!("eh_proj is BF16 {expected:?}, found {}", describe(operand)))
            }
            (Component::Norm | Component::Speculator, _) if name.ends_with("layernorm") || name.ends_with("norm") => {
                bf16(operand, name)
            }
            // FP32 [E, H] (V2 Flash: split into BF16 hi + lo) or BF16 (V2.6 Pro).
            (Component::Router, "gate") => require(
                operand.is_plain(&[if self.cfg.router_fp32 { Encoding::F32 } else { Encoding::Bf16 }])
                    && operand.logical == [self.cfg.experts, self.cfg.hidden],
                || format!("the selected program router weight is {:?} [{}, {}], found {}",
                    if self.cfg.router_fp32 { Encoding::F32 } else { Encoding::Bf16 }, self.cfg.experts, self.cfg.hidden,
                    describe(operand)),
            ),
            (Component::Router, "e_score_correction_bias") => require(operand.is_plain(&[Encoding::F32]), || {
                format!("the router bias is FP32, found {}", describe(operand))
            }),
            _ => Err(format!("{name} is not read by the MiMo loader")),
        }
    }

    fn experts(&self, operand: &QuantOperand) -> Option<ExpertContract> {
        let family = self.geometry()?;
        match family {
            "mimo" if operand.is_fp8_block(128, &[ScaleEncoding::F32, ScaleEncoding::Bf16]) => Some(ExpertContract {
                package: "mimo:fp8".into(),
                block: 128,
                spark_worlds: spark_worlds("mimo:fp8", self.cfg.moe_intermediate),
                local: Ok("serve-mimo --local-experts (fp8-mimo tp1)".into()),
            }),
            "mimof" if operand.is_mxfp4(32) => Some(ExpertContract {
                package: "mimof:fp8 (MXFP4)".into(),
                block: 32,
                spark_worlds: spark_worlds("mimof:fp8", self.cfg.moe_intermediate),
                local: Ok("serve-mimo --local-experts (fp8-mimof tp1)".into()),
            }),
            "mimop" if operand.is_mxfp4(32) => Some(ExpertContract {
                package: "mimop:fp8 (MXFP4)".into(),
                block: 32,
                spark_worlds: spark_worlds("mimop:fp8", self.cfg.moe_intermediate),
                local: Ok("serve-mimo --local-experts (fp8-mimop tp1)".into()),
            }),
            _ => None,
        }
    }
}
