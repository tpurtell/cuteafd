//! Header-native target core. Routed experts and the unused native MTP layer
//! are deliberately outside this inventory; DFlash2 owns its external drafter.
use super::{GlmNextAttention, GlmNextConfig};
use crate::{formats::trellis_core::{read_projections, CoreContract, CoreWeightLayout}, plan::Checkpoint};
pub use crate::formats::trellis_core::{CoreProjection, CoreSplit};
use anyhow::{ensure, Result};
use std::collections::BTreeMap;

const PREFIX: &str = "model.language_model.layers.";

fn attention_leaves(kind: GlmNextAttention) -> [&'static str; 4] {
    match kind {
        GlmNextAttention::Kda => ["q_proj", "k_proj", "v_proj", "o_proj"],
        GlmNextAttention::Mla => ["q_a_proj", "kv_a_proj_with_mqa", "q_b_proj", "o_proj"],
    }
}

pub fn split(cfg: &GlmNextConfig, stem: &str) -> Option<CoreSplit> {
    if stem == "lm_head" { return Some(CoreSplit::Whole); }
    let (layer, leaf) = stem.strip_prefix(PREFIX)?.split_once('.')?;
    let layer = layer.parse::<usize>().ok()?;
    if layer >= cfg.layers { return None; }
    if let Some(leaf) = leaf.strip_prefix("self_attn.") {
        if !attention_leaves(cfg.attention[layer]).contains(&leaf) { return None; }
        return Some(match leaf {
            "q_a_proj" | "kv_a_proj_with_mqa" => CoreSplit::Whole,
            "o_proj" => CoreSplit::Input,
            _ => CoreSplit::Output,
        });
    }
    let mlp = if cfg.dense[layer] { "mlp." } else { "mlp.shared_experts." };
    match leaf.strip_prefix(mlp)? {
        "gate_proj" | "up_proj" => Some(CoreSplit::Output),
        "down_proj" => Some(CoreSplit::Input),
        _ => None,
    }
}

pub fn expected(cfg: &GlmNextConfig) -> BTreeMap<String, Vec<usize>> {
    let mut shapes = BTreeMap::from([("lm_head.weight".into(), vec![cfg.vocab_size, cfg.hidden])]);
    for layer in 0..cfg.layers {
        let root = format!("{PREFIX}{layer}");
        let attention = match cfg.attention[layer] {
            GlmNextAttention::Kda => {
                let d = cfg.kda_heads * cfg.kda_head_dim;
                [("q_proj", d, cfg.hidden), ("k_proj", d, cfg.hidden),
                    ("v_proj", d, cfg.hidden), ("o_proj", cfg.hidden, d)]
            }
            GlmNextAttention::Mla => [
                ("q_a_proj", cfg.q_lora_rank, cfg.hidden),
                ("kv_a_proj_with_mqa", cfg.kv_lora_rank, cfg.hidden),
                ("q_b_proj", cfg.heads * cfg.qk_nope_dim, cfg.q_lora_rank),
                ("o_proj", cfg.hidden, cfg.heads * cfg.v_head_dim),
            ],
        };
        for (leaf, n, k) in attention {
            shapes.insert(format!("{root}.self_attn.{leaf}.weight"), vec![n, k]);
        }
        let (mlp, inter) = if cfg.dense[layer] { ("mlp", cfg.dense_intermediate) }
            else { ("mlp.shared_experts", cfg.moe_intermediate) };
        for (leaf, n, k) in [("gate_proj", inter, cfg.hidden), ("up_proj", inter, cfg.hidden),
            ("down_proj", cfg.hidden, inter)] {
            shapes.insert(format!("{root}.{mlp}.{leaf}.weight"), vec![n, k]);
        }
    }
    shapes
}

#[derive(Debug, Default, Clone)]
pub struct CoreInventory { pub projections: BTreeMap<String, CoreProjection> }

impl CoreInventory {
    pub fn from_checkpoint(cfg: &GlmNextConfig, checkpoint: &Checkpoint) -> Result<Self> {
        let selected = |name: &str| name.starts_with("lm_head.") || name.strip_prefix(PREFIX)
            .and_then(|rest| rest.split_once('.'))
            .is_some_and(|(layer, leaf)| layer.parse::<usize>().is_ok_and(|layer| layer < cfg.layers)
                && !leaf.contains(".experts."));
        let projections = read_projections(checkpoint, &expected(cfg), selected,
            |stem| split(cfg, stem).map(|split| CoreContract { split, bits: 6, padded_outputs: false }))?;
        let inventory = Self { projections };
        for layer in 0..cfg.layers {
            let root = format!("{PREFIX}{layer}");
            let attention = attention_leaves(cfg.attention[layer]);
            let mlp = if cfg.dense[layer] { "mlp" } else { "mlp.shared_experts" };
            for (module, leaves) in [("self_attn", &attention[..]),
                (mlp, &["gate_proj", "up_proj", "down_proj"][..])] {
                let present = leaves.iter().filter(|leaf|
                    inventory.projections.contains_key(&format!("{root}.{module}.{leaf}"))).count();
                ensure!(present == 0 || present == leaves.len(),
                    "{root}.{module}: incomplete native GLM Flash operator group");
            }
        }
        Ok(inventory)
    }

    /// Remove a fused scratch class only when every served group is native.
    pub fn scratch_mask(&self, cfg: &GlmNextConfig, layers: usize) -> crate::serving_capacity::GlmfCoreScratch {
        let has = |layer, leaf| self.projections.contains_key(&format!("{PREFIX}{layer}.{leaf}"));
        crate::serving_capacity::GlmfCoreScratch {
            kda: (0..layers).filter(|&l| cfg.attention[l] == GlmNextAttention::Kda).all(|l| has(l, "self_attn.q_proj")),
            mla: (0..layers).filter(|&l| cfg.attention[l] == GlmNextAttention::Mla).all(|l| has(l, "self_attn.q_a_proj")),
            ffn: (0..layers).all(|l| has(l, if cfg.dense[l] { "mlp.down_proj" } else { "mlp.shared_experts.down_proj" })),
        }
    }

    pub fn get_weight(&self, name: &str) -> Option<&CoreProjection> {
        self.projections.get(name.strip_suffix(".weight").unwrap_or(name))
    }

    /// Whole MLA follows its selected owner. KDA and FFNs retain their TP split.
    pub fn layer_weights(&self, cfg: &GlmNextConfig, layer: usize, ranks: usize, owner: Option<u8>)
        -> Result<Vec<BTreeMap<String, CoreWeightLayout>>> {
        use crate::formats::dense_trellis::TrellisSide;
        ensure!(layer < cfg.layers && matches!(ranks, 1 | 2)
            && owner.is_none_or(|gpu| ranks == 2 && usize::from(gpu) < ranks
                && cfg.attention[layer] == GlmNextAttention::Mla), "GLM Flash core ownership");
        let root = format!("{PREFIX}{layer}.");
        let mut layouts = (0..ranks).map(|_| BTreeMap::new()).collect::<Vec<_>>();
        for (stem, projection) in self.projections.range(root.clone()..) {
            let Some(leaf) = stem.strip_prefix(&root) else { break; };
            let attention = leaf.starts_with("self_attn.");
            let world = if projection.split == CoreSplit::Whole || attention && owner.is_some() { 1 } else { ranks };
            let side = if projection.split == CoreSplit::Input && world > 1 { TrellisSide::Input }
                else { TrellisSide::Output };
            for (rank, layouts) in layouts.iter_mut().enumerate() {
                if attention && owner.is_some_and(|gpu| usize::from(gpu) != rank) { continue; }
                layouts.insert(leaf.to_owned(), CoreWeightLayout::new(projection, side,
                    if world == 1 { 0 } else { rank }, world)?);
            }
        }
        Ok(layouts)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{plan::checkpoint::CheckpointTensor, SafetensorsTensorMetadata};
    use cuteafd_core::DType;

    fn fixture() -> (GlmNextConfig, Checkpoint) {
        let mut config = crate::plan::testing::glm5_flash_config(45);
        config["text_config"]["vocab_size"] = serde_json::json!(154880);
        config["text_config"]["layer_types"] = serde_json::json!((0..45).map(|l|
            if l % 4 == 3 { "deepseek_sparse_attention" } else { "linear_attention" }).collect::<Vec<_>>());
        config["text_config"]["mlp_layer_types"] = serde_json::json!((0..45).map(|l|
            if l < 3 { "dense" } else { "sparse" }).collect::<Vec<_>>());
        config["quantization_config"] = serde_json::json!({"quant_method":"exl3", "format":"exl3",
            "checkpoint_format":"exl3", "bits":4.0, "core_bits":6.0, "mtp_bits":4.0, "head_bits":6.0,
            "per_module_metadata":"quantize_config.json:tensor_storage"});
        let cfg = GlmNextConfig::from_hf(&config).unwrap();
        let mut tensors = Vec::new();
        // The quant finalizer writes packed_specs: [K/16,N/16,16*bits] I16,
        // F16 input/output rotations and an I32 MCG marker. No side file needed.
        for (name, shape) in expected(&cfg) {
            let stem = name.strip_suffix(".weight").unwrap();
            let (n, k) = (shape[0], shape[1]);
            for (suffix, dtype, shape, width) in [("trellis", DType::I16, vec![k/16,n/16,96], 2),
                ("suh", DType::F16, vec![k], 2), ("svh", DType::F16, vec![n], 2),
                ("mcg", DType::I32, vec![1], 4)] {
                tensors.push(CheckpointTensor { shard: "fixture.safetensors".into(), meta: SafetensorsTensorMetadata {
                    name: format!("{stem}.{suffix}"), dtype, byte_offset: 0,
                    byte_length: width * shape.iter().product::<usize>() as u64, shape } });
            }
        }
        let checkpoint = Checkpoint { snapshot: Default::default(), config, quantize_config: None,
            weight_map: tensors.iter().map(|t| (t.meta.name.clone(), t.shard.clone())).collect(), tensors,
            missing_shards: Vec::new(), shard_bytes: 0 };
        (cfg, checkpoint)
    }

    fn complete_fixture() -> (GlmNextConfig, Checkpoint) {
        let (cfg, mut checkpoint) = fixture();
        let mut add = |name: String, dtype: DType, shape: Vec<usize>| {
            let width = match dtype { DType::Bf16 | DType::F16 | DType::I16 => 2, _ => 4 };
            checkpoint.tensors.push(CheckpointTensor { shard: "fixture.safetensors".into(),
                meta: SafetensorsTensorMetadata { name, dtype, byte_offset: 0,
                    byte_length: width * shape.iter().product::<usize>() as u64, shape } });
        };
        add("model.language_model.embed_tokens.weight".into(), DType::Bf16, vec![154880, 4096]);
        add("model.language_model.norm.weight".into(), DType::Bf16, vec![4096]);
        for layer in 0..46 {
            let p = format!("{PREFIX}{layer}");
            for norm in ["input_layernorm", "post_attention_layernorm"] {
                add(format!("{p}.{norm}.weight"), DType::Bf16, vec![4096]);
            }
            if layer < 45 { for site in ["attn", "ffn"] {
                add(format!("{p}.hc_{site}_fn"), DType::Bf16, vec![24, 16384]);
                add(format!("{p}.hc_{site}_base"), DType::F32, vec![24]);
                add(format!("{p}.hc_{site}_scale"), DType::F32, vec![3]);
            } }
            if layer >= 3 {
                add(format!("{p}.mlp.gate.weight"), DType::Bf16, vec![288, 4096]);
                add(format!("{p}.mlp.gate.e_score_correction_bias"), DType::F32, vec![288]);
                for expert in 0..288 { for (leaf, n, k) in [("gate_proj", 2048, 4096),
                    ("up_proj", 2048, 4096), ("down_proj", 4096, 2048)] {
                    let stem = format!("{p}.mlp.experts.{expert}.{leaf}");
                    for (suffix, dtype, shape) in [("trellis", DType::I16, vec![k/16,n/16,64]),
                        ("suh", DType::F16, vec![k]), ("svh", DType::F16, vec![n]),
                        ("mcg", DType::I32, vec![1])] {
                        add(format!("{stem}.{suffix}"), dtype, shape);
                    }
                } }
            }
            if layer < 45 && cfg.attention[layer] == GlmNextAttention::Kda {
                for (leaf, shape) in [("b_proj.weight", vec![64,4096]),
                    ("f_a_proj.weight", vec![128,4096]), ("g_a_proj.weight", vec![128,4096]),
                    ("f_b_proj.weight", vec![8192,128]), ("g_b_proj.weight", vec![8192,128]),
                    ("o_norm.weight", vec![128])] {
                    add(format!("{p}.self_attn.{leaf}"), DType::Bf16, shape);
                }
                for kind in ["q", "k", "v"] {
                    add(format!("{p}.self_attn.{kind}_conv1d.weight"), DType::Bf16, vec![8192,1,4]);
                }
                add(format!("{p}.self_attn.A_log"), DType::F32, vec![64]);
                add(format!("{p}.self_attn.dt_bias"), DType::F32, vec![8192]);
            } else {
                for (leaf, shape) in [("q_a_layernorm.weight", vec![1536]),
                    ("kv_a_layernorm.weight", vec![512]), ("kv_b_proj.weight", vec![32768,512]),
                    ("indexer.wq_b.weight", vec![4096,1536]), ("indexer.wk.weight", vec![128,4096]),
                    ("indexer.weights_proj.weight", vec![32,4096]), ("indexer.k_norm.weight", vec![128]),
                    ("indexer.k_norm.bias", vec![128]), ("indexer.index_kpool_compress_ape", vec![4,128]),
                    ("indexer.index_kpool_compress_gate", vec![128,4096])] {
                    add(format!("{p}.self_attn.{leaf}"), DType::Bf16, shape);
                }
            }
            if layer == 45 {
                for (leaf, n, k) in [("self_attn.q_a_proj",1536,4096),
                    ("self_attn.q_b_proj",16384,1536), ("self_attn.kv_a_proj_with_mqa",512,4096),
                    ("self_attn.o_proj",4096,16384), ("mlp.shared_experts.gate_proj",2048,4096),
                    ("mlp.shared_experts.up_proj",2048,4096), ("mlp.shared_experts.down_proj",4096,2048),
                    ("eh_proj",4096,8192)] {
                    for (suffix, dtype, shape) in [("trellis", DType::I16, vec![k/16,n/16,64]),
                        ("suh", DType::F16, vec![k]), ("svh", DType::F16, vec![n]),
                        ("mcg", DType::I32, vec![1])] {
                        add(format!("{p}.{leaf}.{suffix}"), dtype, shape);
                    }
                }
                for norm in ["enorm", "hnorm", "shared_head.norm"] {
                    add(format!("{p}.{norm}.weight"), DType::Bf16, vec![4096]);
                }
            }
        }
        for (leaf, shape) in crate::plan::families::glm::glm_flash_vision_tensors() {
            add(format!("model.visual.{leaf}"), DType::Bf16, shape);
        }
        checkpoint.tensors.sort_by(|a,b| a.meta.name.cmp(&b.meta.name));
        checkpoint.weight_map = checkpoint.tensors.iter().map(|t| (t.meta.name.clone(), t.shard.clone())).collect();
        checkpoint.shard_bytes = checkpoint.tensors.iter().map(|t| t.meta.byte_length).sum();
        (cfg, checkpoint)
    }

    fn executable_package(cfg: &GlmNextConfig, core: &CoreInventory) -> Result<serde_json::Value> {
        use crate::formats::{dense_trellis::TrellisSide, trellis_core::CoreWeightLayout};
        use serde_json::json;
        let mut weights = Vec::new();
        for ranks in [1, 2] { for owner in if ranks == 1 { vec![None] } else { vec![None, Some(0), Some(1)] } {
            for layer in 0..cfg.layers {
                let owner = owner.filter(|_| cfg.attention[layer] == GlmNextAttention::Mla);
                weights.extend(core.layer_weights(cfg, layer, ranks, owner)?.into_iter().flat_map(BTreeMap::into_values));
            }
        } }
        weights.push(CoreWeightLayout::new(core.get_weight("lm_head").unwrap(), TrellisSide::Output, 0, 1)?);
        let mut policies = BTreeMap::new();
        let mut programs = BTreeMap::new();
        for weight in weights {
            let [n, k] = weight.shape;
            let side = if weight.side == TrellisSide::Input { 2 } else { 1 };
            let shape = format!("k{k}_n{n}_s{side}_mcg");
            let routes = [format!("input_{shape}"), format!("core_{shape}"), format!("output_{shape}")];
            policies.insert(shape.clone(), json!({"shape":shape,"input_features":k,"output_features":n,
                "sharded_side":side,"codebook":"mcg","native_input":routes[0],"native_program":routes[1],
                "output_program":routes[2]}));
            let geometries = [json!({"features":k,"input_dtype":"bf16","max_rows":4096}),
                json!({"abi_version":1,"input_features":k,"output_features":n,"bits":6,"codebook":"mcg",
                    "sharded_side":side,"rotation_count":1,"max_rows":4096,"split_k_blocks":8,
                    "splits":k.div_ceil(1024),"ctas_per_sm":4}),
                json!({"features":n,"output_dtype":if weight.stem == "lm_head" { "f32" } else { "bf16" },"max_rows":4096})];
            let pointers = [vec!["input", "suh", "rotated", "input_block_scales"],
                vec!["rotated_inputs", "payload", "input_block_scales", "partial", "output"], vec!["input", "svh", "output"]];
            for i in 0..3 {
                let name = format!("dense_trellis_k6_{}", routes[i]);
                programs.insert(name.clone(), json!({"name":name,"capacity_rows":4096,"geometry":geometries[i],
                    "pointers":pointers[i].iter().map(|name| json!({"name":name})).collect::<Vec<_>>(),
                    "scalars":vec![json!({"type":"int32"}); if i == 1 { 3 } else { 2 }],
                    "scratch_bytes_at_capacity":if i == 1 { json!({"partial":4096*k.div_ceil(1024)*n*4}) } else { json!({}) }}));
            }
        }
        for (ranks, prefix) in [(1, "glmf"), (2, "glmf2")] {
            for (cap, mode, decode) in [("m64", "decode", true), ("m4096", "prefill", false)] {
                let mut names = vec![format!("{prefix}_mhc_pre"), format!("{prefix}_mhc_post_pre_{cap}"),
                    format!("{prefix}_kda_{cap}"), format!("{prefix}_kda_w8_{cap}"),
                    format!("{prefix}_kda_w8_expanded_{cap}"), format!("{prefix}_kda_w8_f32_{cap}"),
                    format!("{prefix}_kda_w8_f32_expanded_{cap}"), format!("{prefix}_mla_producer_{cap}"),
                    format!("{prefix}_index_producer_{cap}"), format!("{prefix}_index_producer_c_{cap}"),
                    format!("{prefix}_index_topk_{mode}_{cap}"), format!("{prefix}_sparse_mla_{mode}_{cap}"),
                    format!("{prefix}_o_{cap}")];
                for inter in [cfg.dense_intermediate / ranks, cfg.moe_intermediate / ranks] {
                    names.push(format!("{prefix}_ffn_i{inter}_{cap}"));
                }
                names.extend(super::super::core_admission::route_stage_names(cfg, ranks, false, decode, 64));
                for name in names { programs.insert(name.clone(), json!({"name":name,"scratch_bytes_at_capacity":{"scratch":4096}})); }
            }
        }
        Ok(json!({"capacities":{"max_context":1_048_576,"decode_rows":64,"prefill_rows":4096},
            "families":{"dense_trellis_k6":{"abi_version":1,"row_policies":policies.into_values().collect::<Vec<_>>()}},
            "programs":programs.into_values().collect::<Vec<_>>()}))
    }

    #[test]
    fn published_finalizer_manifest_has_executable_native_min_and_max_plans() -> Result<()> {
        use crate::plan::{testing, ExpertPlacement, PlanOptions};
        use serde_json::json;
        let (cfg, mut checkpoint) = complete_fixture();
        checkpoint.config["vision_config"] = json!({"depth":24,"hidden_size":1024,"intermediate_size":4096,
            "num_heads":16,"out_hidden_size":4096,"patch_size":14,"temporal_patch_size":2,"spatial_merge_size":2,
            "projection_intermediate_size":10240,"in_channels":3,"attention_bias":true,"hidden_act":"silu",
            "rms_norm_eps":1e-5,"swiglu_limit":10.0});
        let root = tempfile::tempdir()?;
        // One sparse shard per layer avoids a giant safetensors header; no payload is read.
        let mut shards: BTreeMap<String, Vec<testing::Tensor>> = BTreeMap::new();
        for tensor in &checkpoint.tensors {
            let layer = tensor.meta.name.strip_prefix(PREFIX).and_then(|s| s.split_once('.')).map(|(l, _)| l).unwrap_or("shared");
            let dtype = match tensor.meta.dtype { DType::Bf16 => "BF16", DType::F32 => "F32", DType::F16 => "F16", DType::I16 => "I16", DType::I32 => "I32", _ => unreachable!() };
            shards.entry(format!("layer-{layer}.safetensors")).or_default().push((tensor.meta.name.clone(), dtype, tensor.meta.shape.clone()));
        }
        let mut weight_map = serde_json::Map::new();
        for (name, tensors) in shards {
            testing::write_safetensors(&root.path().join(&name), &tensors);
            for (tensor, ..) in tensors { weight_map.insert(tensor, json!(name)); }
        }
        std::fs::write(root.path().join("config.json"), checkpoint.config.to_string())?;
        std::fs::write(root.path().join("model.safetensors.index.json"), json!({"weight_map":weight_map}).to_string())?;
        let share = root.path().join("share");
        std::fs::create_dir_all(&share)?;
        let manifest = executable_package(&cfg, &CoreInventory::from_checkpoint(&cfg, &checkpoint)?)?;
        let path = share.join("PROGRAMS.json");
        std::fs::write(&path, manifest.to_string())?;
        for tp in [1, 2] { for capacity in crate::placement::inventory::exl3_capacities(4096) {
            let dir = share.join(format!("exl3/exl3-glmf-k34/rtx-tp{tp}/m{capacity}"));
            std::fs::create_dir_all(&dir)?;
            std::fs::write(dir.join("v41_exl3.json"), json!({"hidden":4096,"intermediate":2048/tp,"experts":288,
                "capacity":capacity,"input_format":"e4m3_k32","trellis_lut":{"bytes":32},
                "buffers":{"scratch":{"bytes":capacity*16,"allocation":"scratch","dtype":"f32","zero_on_create":false},
                    "state":{"bytes":256,"allocation":"state","dtype":"i32","zero_on_create":true}}}).to_string())?;
        } }
        let draft = root.path().join("draft");
        std::fs::create_dir_all(&draft)?;
        std::fs::write(draft.join("config.json"), json!({"hidden_size":4096,"intermediate_size":12288,"num_hidden_layers":5,
            "num_attention_heads":32,"num_key_value_heads":8,"head_dim":128,"vocab_size":154880,
            "dflash_config":{"target_layer_ids":[0,8,16,24,32],"conv_group_size":16,"selector_rank":256,"block_size":8}}).to_string())?;
        let input = testing::BudgetedPlanInput::open(root.path())?;
        assert_eq!(input.checkpoint().tensors.len(), 151198);
        assert_eq!(input.checkpoint().tensors.iter().map(|t| t.meta.byte_length).sum::<u64>(), 165621841672);
        for (gpus, placement) in [(1, ExpertPlacement::Sparks { ranks: 2 }), (2, ExpertPlacement::Local)] {
            let report = input.plan(&PlanOptions { placement, attention_placement: Some(crate::placement::AttentionPlacement::Heads),
                layout: Some(crate::plan::layout::LayoutOptions { rtx_bytes: vec![crate::placement::inventory::PRO_TOTAL_BYTES; gpus],
                    head_split: gpus == 2, host_memory: Some((128 << 30, 128 << 30)), context_tokens: 262144,
                    pool_tokens: Some(262144), workspace_manifest: Some(path.clone()), glmf_drafter_snapshot: Some(draft.clone()),
                    weight_budget_bytes: 80 << 30, ..Default::default() }), ..Default::default() })?;
            assert!(report.executable(), "{}", crate::plan::render(&report));
            let layout = report.memory_layout.as_ref().unwrap();
            assert!(layout.devices.iter().take(gpus).all(|d| d.items.iter().any(|i| i.group == "native K6 core")));
            assert_eq!(layout.devices.iter().filter(|d| d.kind == cuteafd_core::memory_layout::DeviceKind::Spark).count(), if gpus == 1 { 2 } else { 0 });
        }
        Ok(())
    }

    #[test]
    fn published_finalizer_manifest_and_native_residency() -> Result<()> {
        let (cfg, checkpoint) = complete_fixture();
        assert_eq!(checkpoint.tensors.len(), 151198);
        assert_eq!(checkpoint.shard_bytes, 165621841672);
        assert_eq!(CoreInventory::from_checkpoint(&cfg, &checkpoint)?.projections.len(), 316);
        for ranks in [1,2] {
            let repr = super::super::resident::GlmfRepresentation::default();
            let resident = super::super::resident::resident_weights(&checkpoint, &cfg, cfg.layers, ranks, repr)
                .map_err(anyhow::Error::msg)?;
            assert_eq!(resident.len(), ranks);
            assert!(resident[0].embedding > 0);
            assert!(resident.iter().all(|r| r.weights > 0));
            if ranks == 2 { assert_eq!(resident[1].embedding, 0); }
        }
        Ok(())
    }

    #[test]
    fn published_target_core_has_316_header_native_owners_and_no_kv_padding() -> Result<()> {
        let (cfg, checkpoint) = fixture();
        let core = CoreInventory::from_checkpoint(&cfg, &checkpoint)?;
        assert_eq!(core.projections.len(), 316);
        assert_eq!(core.get_weight("lm_head").unwrap().shape, [154880, 4096]);
        for layer in 0..cfg.layers {
            if cfg.attention[layer] == GlmNextAttention::Mla {
                let kv = core.get_weight(&format!("{PREFIX}{layer}.self_attn.kv_a_proj_with_mqa")).unwrap();
                assert_eq!(kv.shape, [512, 4096]);
                assert_eq!(kv.stored_outputs, 512);
            }
            for ranks in [1, 2] {
                let layouts = core.layer_weights(&cfg, layer, ranks, None)?;
                assert!(layouts.iter().all(|rank| rank.len() == 7));
                for rank in layouts { for layout in rank.values() {
                    assert_eq!(layout.bytes, layout.shape[0]*layout.shape[1]*6/8
                        + 2*(layout.shape[0]+layout.shape[1]));
                } }
            }
        }
        Ok(())
    }

    #[test]
    fn scratch_masks_preserve_fused_storage_for_mixed_groups() -> Result<()> {
        let (cfg, checkpoint) = fixture();
        let mut core = CoreInventory::from_checkpoint(&cfg, &checkpoint)?;
        let all = core.scratch_mask(&cfg, cfg.layers);
        assert!(all.kda && all.mla && all.ffn);
        core.projections.retain(|name, _| !name.starts_with(&format!("{PREFIX}0.self_attn.")));
        let mixed = core.scratch_mask(&cfg, cfg.layers);
        assert!(!mixed.kda && mixed.mla && mixed.ffn);
        core.projections.retain(|name, _| !name.starts_with(&format!("{PREFIX}3.self_attn.")));
        core.projections.retain(|name, _| !name.starts_with(&format!("{PREFIX}1.mlp.")));
        let mixed = core.scratch_mask(&cfg, cfg.layers);
        assert!(!mixed.kda && !mixed.mla && !mixed.ffn);
        let empty = core.scratch_mask(&cfg, 0);
        assert!(empty.kda && empty.mla && empty.ffn);
        Ok(())
    }

    #[test]
    fn rejects_partial_groups_wrong_bits_and_quantized_keep_operands() {
        let (cfg, checkpoint) = fixture();
        for suffix in ["q_proj", "f_a_proj"] {
            let (_, mut changed) = fixture();
            let stem = format!("{PREFIX}0.self_attn.{suffix}");
            if suffix == "q_proj" {
                changed.tensors.retain(|t| !t.meta.name.starts_with(&stem));
                changed.weight_map.retain(|name, _| !name.starts_with(&stem));
            } else {
                let source = format!("{PREFIX}0.self_attn.q_proj");
                let extra = changed.tensors.iter().filter(|t| t.meta.name.starts_with(&source)).cloned()
                    .map(|mut t| { t.meta.name = t.meta.name.replacen(&source, &stem, 1); t }).collect::<Vec<_>>();
                changed.tensors.extend(extra);
            }
            assert!(CoreInventory::from_checkpoint(&cfg, &changed).is_err());
        }
        let mut changed = checkpoint;
        let tensor = changed.tensors.iter_mut().find(|t| t.meta.name.ends_with("q_proj.trellis")).unwrap();
        tensor.meta.shape[2] = 64;
        tensor.meta.byte_length = tensor.meta.shape.iter().product::<usize>() as u64 * 2;
        assert!(CoreInventory::from_checkpoint(&cfg, &changed).is_err());
    }
}
