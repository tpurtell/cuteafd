//! Header-native Muse Spark core trellis inventory: the dense matrices an
//! EXL3 Core-K6 publication may store as K6 MCG trellis. Everything outside
//! this keep-list (norms, residual gates, router, embedding, towers) keeps
//! its checkpoint precision. The attention output gate (`self_attn.gate_proj`)
//! is on the list but optional: a publication may keep it BF16.
use super::config::MuseSparkConfig;
use crate::{formats::trellis_core::{read_projections, CoreContract}, plan::Checkpoint};
pub use crate::formats::trellis_core::{CoreProjection, CoreSplit};
use anyhow::{ensure, Context, Result};
pub use crate::formats::native_attention::{NativeLayout, NativeProjection};
use std::collections::BTreeMap;

pub const PREFIX: &str = "model.language_model.layers.";
pub const HEAD: &str = "lm_head";
/// Required attention operands, resolved independently in the selected index.
const ATTENTION: [&str; 4] = ["q_proj", "k_proj", "v_proj", "o_proj"];

/// TP2 ownership of each eligible projection: attention heads split; the MoE
/// frontend splits its outputs over E and the backend its inputs (single
/// residency, a rank-ordered K-split reduction). The LM head stays whole.
pub fn split(cfg: &MuseSparkConfig, stem: &str) -> Option<CoreSplit> {
    if stem == HEAD { return Some(CoreSplit::Whole); }
    let (layer, leaf) = stem.strip_prefix(PREFIX)?.split_once('.')?;
    if layer.parse::<usize>().ok()? >= cfg.layers { return None; }
    match leaf {
        "self_attn.q_proj" | "self_attn.k_proj" | "self_attn.v_proj" | "self_attn.gate_proj" => Some(CoreSplit::Output),
        "self_attn.o_proj" => Some(CoreSplit::Input),
        "mlp.pre_expert_proj" => Some(CoreSplit::Output),
        "mlp.post_expert_proj" => Some(CoreSplit::Input),
        _ => None,
    }
}

/// Logical [N, K] of every keep-list matrix.
pub fn expected(cfg: &MuseSparkConfig) -> BTreeMap<String, Vec<usize>> {
    let (h, q, kv, e) = (cfg.hidden, cfg.heads * cfg.head_dim, cfg.kv_heads * cfg.head_dim, cfg.expert_hidden);
    let mut shapes = BTreeMap::from([(format!("{HEAD}.weight"), vec![cfg.vocab_size, h])]);
    for layer in 0..cfg.layers {
        let root = format!("{PREFIX}{layer}");
        for (leaf, n, k) in [("self_attn.q_proj", q, h), ("self_attn.k_proj", kv, h), ("self_attn.v_proj", kv, h),
            ("self_attn.o_proj", h, q), ("self_attn.gate_proj", q, h),
            ("mlp.pre_expert_proj", e, h), ("mlp.post_expert_proj", h, e)] {
            shapes.insert(format!("{root}.{leaf}.weight"), vec![n, k]);
        }
    }
    shapes
}

#[derive(Debug, Default, Clone)]
pub struct CoreInventory {
    pub projections: BTreeMap<String, CoreProjection>,
    pub native: BTreeMap<String, NativeProjection>,
}

impl CoreInventory {
    pub fn from_checkpoint(cfg: &MuseSparkConfig, checkpoint: &Checkpoint) -> Result<Self> {
        let selected = |name: &str| name.starts_with("lm_head.") || name.strip_prefix(PREFIX)
            .and_then(|rest| rest.split_once('.'))
            .is_some_and(|(layer, leaf)| layer.parse::<usize>().is_ok_and(|layer| layer < cfg.layers)
                && !leaf.starts_with("mlp.experts."));
        // A trellis head is stored at whole 128-row output blocks (202,112 for the
        // logical 202,048 vocabulary); the engine trims logits before the softcap.
        checkpoint.validate_fp8_roles("Muse", |name| {
            let stem = name.strip_suffix(".weight").unwrap_or(name);
            let role = if stem.ends_with("lm_head") { "vocabulary head" }
                else if stem.contains(".self_attn.") { "attention projection" }
                else if stem.contains(".mlp.") { "FFN projection" } else { "non-core" };
            crate::formats::native_attention::Fp8Role::exact(role, stem.contains(".self_attn.") && matches!(stem.rsplit('.').next(), Some("q_proj" | "k_proj" | "v_proj" | "o_proj")))
        })?;
        let projections = read_projections(checkpoint, &expected(cfg), selected,
            |stem| split(cfg, stem).map(|split| CoreContract { family: "Muse", widths: &[4, 6, 8], split, view_varying: true, padded_outputs: stem == HEAD }))?;
        if let Some(head) = projections.get(HEAD) {
            ensure!(matches!(head.bits, 6 | 8), "Muse {HEAD}: target head requires K6/K8 trellis or BF16");
        } else {
            ensure!(!["weight_packed", "trellis"].iter().any(|suffix|
                checkpoint.contains_tensor(&format!("{HEAD}.{suffix}"))),
                "Muse {HEAD}: conflicting or unsupported selected target head storage");
            // Index-only planner inventories do not open ordinary native headers.
            // Resolve only the selected view's indexed shard, never another view.
            let name = format!("{HEAD}.weight");
            let header = if let Some(tensor) = checkpoint.tensors.iter().find(|t| t.meta.name == name) {
                tensor.meta.clone()
            } else {
                let shard = checkpoint.weight_map.get(&name).context("Muse selected BF16 head is not indexed")?;
                crate::read_safetensors_metadata(&checkpoint.snapshot.join(shard))?.into_iter()
                    .find(|meta| meta.name == name).context("Muse selected BF16 head header is missing")?
            };
            ensure!(header.dtype == cuteafd_core::DType::Bf16 && header.shape == [cfg.vocab_size, cfg.hidden],
                "Muse {HEAD}: selected native target head must be BF16 [{}, {}]", cfg.vocab_size, cfg.hidden);
        }
        let mut inventory = Self { projections, native: BTreeMap::new() };
        for (name, shape) in expected(cfg) {
            let stem = name.strip_suffix(".weight").unwrap();
            if !stem.contains(".self_attn.") || stem.ends_with(".gate_proj")
                || !checkpoint.contains_tensor(&name) { continue; }
            let selected = checkpoint.projection(stem)?;
            let crate::plan::checkpoint::ProjectionStorage::Native { ref dtype, shape: ref actual } = selected.storage else { continue; };
            ensure!(*actual == shape, "{stem}: native Muse attention shape {actual:?}, expected {shape:?}");
            ensure!(matches!(dtype, cuteafd_core::DType::Bf16 | cuteafd_core::DType::F8E4M3),
                "{stem}: native Muse attention requires BF16 or block E4M3");
            if *dtype == cuteafd_core::DType::F8E4M3 {
                if checkpoint.converts_to_fp8(&format!("{stem}.weight")) {
                    return Err(crate::formats::native_attention::UnsupportedFp8Conversion {
                        tensor: name.clone(), found: "E4M3".into(),
                    }.into());
                }
                crate::formats::native_attention::fp8_scale(&selected, stem)?;
            }
            inventory.native.insert(stem.into(), NativeProjection {
                shape: [shape[0], shape[1]], split: split(cfg, stem).context("Muse attention split")?,
                fp8: *dtype == cuteafd_core::DType::F8E4M3
                    || checkpoint.converts_to_fp8(&format!("{stem}.weight")),
            });
        }
        for layer in 0..cfg.layers {
            let root = format!("{PREFIX}{layer}");
            for (module, leaves) in [("self_attn", &ATTENTION[..]), ("mlp", &["pre_expert_proj", "post_expert_proj"][..])] {
                let present = leaves.iter().filter(|leaf|
                    inventory.projections.contains_key(&format!("{root}.{module}.{leaf}"))).count();
                let mixed_complete = module == "self_attn" && leaves.iter().all(|leaf| {
                    let stem = format!("{root}.{module}.{leaf}");
                    inventory.projections.contains_key(&stem) || inventory.native.contains_key(&stem)
                });
                ensure!(present == 0 || present == leaves.len() || mixed_complete,
                    "{root}.{module}: incomplete native Muse Spark operator group");
            }
        }
        Ok(inventory)
    }

    /// The selected target head: padded for header-native trellis, logical for BF16.
    /// `from_checkpoint` validates the native operand, never a fallback from another view.
    pub fn head_outputs(&self, cfg: &MuseSparkConfig) -> usize {
        self.get_weight(HEAD).map_or(cfg.vocab_size, |head| head.stored_outputs)
    }

    pub fn get_weight(&self, name: &str) -> Option<&CoreProjection> {
        self.projections.get(name.strip_suffix(".weight").unwrap_or(name))
    }

    /// Whether every layer's attention and MoE projections are native K6.
    pub fn complete(&self, cfg: &MuseSparkConfig) -> bool {
        (0..cfg.layers).all(|layer| ["self_attn.q_proj", "mlp.pre_expert_proj"].iter()
            .all(|leaf| self.projections.contains_key(&format!("{PREFIX}{layer}.{leaf}"))))
    }
}

#[cfg(test)]
mod head_tests {
    use super::*;
    use crate::plan::testing;

    #[test]
    fn original_attention_uses_whole_and_tp2_native_shapes() -> Result<()> {
        use super::super::weight_layout::LayerLayout;
        use cuteafd_ffi::draft_projection::ProjectionOutput;
        let cfg = MuseSparkConfig::from_hf(&testing::muse_spark_config())?;
        let dir = tempfile::tempdir()?;
        let mut tensors = vec![testing::t("lm_head.weight", "BF16", &[cfg.vocab_size, cfg.hidden])];
        for (leaf, n) in [("q_proj", 8192), ("k_proj", 1024), ("v_proj", 1024), ("o_proj", 8192)] {
            tensors.push(testing::t(format!("{PREFIX}0.self_attn.{leaf}.weight"), "BF16", &[n, 8192]));
        }
        testing::write_snapshot(dir.path(), &testing::muse_spark_config(), &tensors, None);
        let inventory = CoreInventory::from_checkpoint(&cfg, &Checkpoint::open(dir.path())?)?;
        assert_eq!(inventory.native.len(), 4);
        for ranks in [1, 2] {
            for layout in LayerLayout::ranks(&cfg, 0, ranks, &inventory)? {
                let shapes = [("attn.q", [8192 / ranks, 8192]), ("attn.k", [1024 / ranks, 8192]),
                    ("attn.v", [1024 / ranks, 8192]), ("attn.o", [8192, 8192 / ranks])];
                for (name, shape) in shapes {
                    let native = &layout.native[name];
                    assert_eq!(native.shape, shape);
                    assert!(!native.fp8);
                    let output = if ranks == 2 && name == "attn.o" { ProjectionOutput::F32 }
                        else { ProjectionOutput::Bf16 };
                    let suffix = if output == ProjectionOutput::F32 { "_f32" } else { "" };
                    assert_eq!(output.program_name(shape[1], shape[0]),
                        format!("draft_bf16_k{}_n{}{suffix}", shape[1], shape[0]));
                }
            }
        }
        Ok(())
    }

    #[test]
    fn selected_head_is_validated_without_cross_view_fallback() -> Result<()> {
        let cfg = MuseSparkConfig::from_hf(&testing::muse_spark_config())?;
        for bits in [6, 8] {
            let dir = tempfile::tempdir()?;
            let tensors = testing::exl3(HEAD, cfg.vocab_size.next_multiple_of(128), cfg.hidden, bits);
            testing::write_snapshot(dir.path(), &testing::muse_spark_config(), &tensors, None);
            let checkpoint = Checkpoint::open(dir.path())?;
            let inventory = CoreInventory::from_checkpoint(&cfg, &checkpoint)?;
            assert_eq!(inventory.head_outputs(&cfg), cfg.vocab_size.next_multiple_of(128));
            assert_eq!(inventory.get_weight(HEAD).unwrap().bits, bits);
        }
        let dir = tempfile::tempdir()?;
        testing::write_snapshot(dir.path(), &testing::muse_spark_config(),
            &[testing::t("lm_head.weight", "BF16", &[cfg.vocab_size, cfg.hidden])], None);
        let mut checkpoint = Checkpoint::open(dir.path())?;
        let inventory = CoreInventory::from_checkpoint(&cfg, &checkpoint)?;
        assert_eq!(inventory.head_outputs(&cfg), cfg.vocab_size);
        assert!(inventory.get_weight(HEAD).is_none());
        let mut index_only = Checkpoint::open(dir.path())?;
        index_only.tensors.clear();
        assert_eq!(CoreInventory::from_checkpoint(&cfg, &index_only)?.head_outputs(&cfg), cfg.vocab_size);
        index_only.weight_map.insert("lm_head.weight_packed".into(), "absent.safetensors".into());
        assert!(CoreInventory::from_checkpoint(&cfg, &index_only).is_err());
        for (dtype, shape) in [(cuteafd_core::DType::F32, vec![cfg.vocab_size, cfg.hidden]),
            (cuteafd_core::DType::Bf16, vec![cfg.vocab_size + 64, cfg.hidden])] {
            checkpoint.tensors[0].meta.dtype = dtype;
            checkpoint.tensors[0].meta.shape = shape;
            assert!(CoreInventory::from_checkpoint(&cfg, &checkpoint).is_err());
        }
        checkpoint.tensors.clear();
        checkpoint.weight_map.clear();
        assert!(CoreInventory::from_checkpoint(&cfg, &checkpoint).is_err());
        Ok(())
    }
}
