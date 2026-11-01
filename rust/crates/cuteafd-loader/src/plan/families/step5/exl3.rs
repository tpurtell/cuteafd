//! Step-5 EXL3 publications (`wrldsuksgo2mars/Step-5-Preview-EXL3-K4-Core-K6`):
//! the header-native dense core and per-expert routed storage. Everything is
//! derived from `config.json`, `quantize_config.json` (cross-check only) and the
//! safetensors headers; nothing is read from side files.
use super::Config;
use crate::formats::trellis_core::{read_projections, CoreContract, CoreProjection, CoreSplit};
use crate::plan::checkpoint::Checkpoint;
use anyhow::{ensure, Context, Result};
use std::collections::BTreeMap;

/// The checkpoint's declared EXL3 widths, cross-checked against every header.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Widths {
    /// Target attention q/k/v/o, shared and dense FFNs.
    pub core: usize,
    pub head: usize,
    /// Every MTP-stage projection, including each stage's own vocabulary head.
    pub mtp: usize,
}

/// `config.json` declares the publication; tensors carry the actual widths.
pub fn is_exl3(checkpoint: &Checkpoint) -> bool {
    quantization(checkpoint).is_some_and(|q| q["quant_method"] == "exl3")
}

fn quantization(checkpoint: &Checkpoint) -> Option<&serde_json::Value> {
    checkpoint.config.get("quantization_config")
        .or_else(|| checkpoint.config.get("text_config").and_then(|t| t.get("quantization_config")))
}

pub fn widths(checkpoint: &Checkpoint) -> Result<Widths> {
    let q = quantization(checkpoint).context("Step-5 EXL3 requires quantization_config")?;
    let int = |key: &str, default: Option<usize>| -> Result<usize> {
        match q.get(key) {
            None => default.with_context(|| format!("Step-5 EXL3 requires quantization_config.{key}")),
            Some(v) => {
                let bits = v.as_f64().filter(|b| b.fract() == 0.0 && (2.0..=6.0).contains(b))
                    .with_context(|| format!("Step-5 EXL3 {key} must be an integer K2..K6, found {v}"))?;
                Ok(bits as usize)
            }
        }
    };
    let core = int("core_bits", None)?;
    Ok(Widths { core, head: int("head_bits", Some(core))?, mtp: int("mtp_bits", Some(core))? })
}

/// Stem and [N,K] of every quantizable dense projection the recipe may store
/// as EXL3. Stage-local heads are dense projections like any other MTP matrix.
pub fn expected(cfg: &Config) -> BTreeMap<String, Vec<usize>> {
    let h = cfg.hidden;
    let mut shapes = BTreeMap::from([("lm_head.weight".to_owned(), vec![cfg.vocab, h])]);
    for layer in 0..cfg.layers + cfg.mtp {
        let root = format!("model.layers.{layer}");
        for (leaf, n, k) in [("q_proj", cfg.heads * cfg.dim, h), ("k_proj", cfg.groups * cfg.dim, h),
            ("v_proj", cfg.groups * cfg.dim, h), ("o_proj", h, cfg.heads * cfg.dim)] {
            shapes.insert(format!("{root}.self_attn.{leaf}.weight"), vec![n, k]);
        }
        let (mlp, inter) = if cfg.moe[layer] { ("share_expert", cfg.shared) } else { ("mlp", cfg.dense) };
        for (leaf, n, k) in [("gate_proj", inter, h), ("up_proj", inter, h), ("down_proj", h, inter)] {
            shapes.insert(format!("{root}.{mlp}.{leaf}.weight"), vec![n, k]);
        }
        if layer >= cfg.layers {
            shapes.insert(format!("{root}.eh_proj.weight"), vec![h, 2 * h]);
            shapes.insert(format!("{root}.transformer.shared_head.output.weight"), vec![cfg.vocab, h]);
        }
    }
    shapes
}

/// How the current two-GPU Step engine owns a projection: heads split q/k/v
/// by output rows and o_proj by input columns; FFNs and heads stay whole.
pub fn split(cfg: &Config, stem: &str) -> Option<CoreSplit> {
    if stem == "lm_head" { return Some(CoreSplit::Whole); }
    let (layer, leaf) = stem.strip_prefix("model.layers.")?.split_once('.')?;
    let layer = layer.parse::<usize>().ok()?;
    if layer >= cfg.layers + cfg.mtp { return None; }
    if layer >= cfg.layers {
        return matches!(leaf, "self_attn.q_proj" | "self_attn.k_proj" | "self_attn.v_proj" | "self_attn.o_proj"
            | "mlp.gate_proj" | "mlp.up_proj" | "mlp.down_proj" | "eh_proj"
            | "transformer.shared_head.output").then_some(CoreSplit::Whole);
    }
    let ffn = if cfg.moe[layer] { "share_expert." } else { "mlp." };
    match leaf {
        "self_attn.q_proj" | "self_attn.k_proj" | "self_attn.v_proj" => Some(CoreSplit::Output),
        "self_attn.o_proj" => Some(CoreSplit::Input),
        _ => leaf.strip_prefix(ffn).filter(|leaf| matches!(*leaf, "gate_proj" | "up_proj" | "down_proj"))
            .map(|_| CoreSplit::Whole),
    }
}

/// The recipe's routed width for a target layer: `expert_bits_by_layer`
/// overrides `expert_bits` (falling back to `bits`). Headers stay authoritative.
pub fn declared_expert_bits(checkpoint: &Checkpoint, layer: usize) -> Option<usize> {
    let q = quantization(checkpoint)?;
    q.get("expert_bits_by_layer").and_then(|m| m.get(layer.to_string()))
        .or_else(|| q.get("expert_bits")).or_else(|| q.get("bits"))
        .and_then(serde_json::Value::as_f64).filter(|b| b.fract() == 0.0).map(|b| b as usize)
}

#[derive(Debug, Clone, Default)]
pub struct CoreInventory {
    pub projections: BTreeMap<String, CoreProjection>,
}

impl CoreInventory {
    /// Every target, head and MTP dense EXL3 projection, each at its declared
    /// width. Index-only role inventories read the needed headers once.
    pub fn from_checkpoint(cfg: &Config, checkpoint: &Checkpoint) -> Result<Self> {
        if !is_exl3(checkpoint) { return Ok(Self::default()); }
        let widths = widths(checkpoint)?;
        let shapes = expected(cfg);
        let selected = |name: &str| !name.contains(".experts.") && (name.starts_with("lm_head.")
            || name.starts_with("model.layers."));
        let projections = read_projections(checkpoint, &shapes, selected, |stem| {
            let split = split(cfg, stem)?;
            let mtp = stem.strip_prefix("model.layers.").and_then(|r| r.split('.').next())
                .and_then(|l| l.parse::<usize>().ok()).is_some_and(|l| l >= cfg.layers);
            let bits = if mtp { widths.mtp } else if stem == "lm_head" { widths.head } else { widths.core };
            Some(CoreContract { split, bits, padded_outputs: false })
        })?;
        let inventory = Self { projections };
        inventory.validate_groups(cfg)?;
        Ok(inventory)
    }

    pub fn get(&self, stem: &str) -> Option<&CoreProjection> {
        self.projections.get(stem.strip_suffix(".weight").unwrap_or(stem))
    }

    /// The engine runs complete operator groups natively or not at all.
    pub fn validate_groups(&self, cfg: &Config) -> Result<()> {
        for layer in 0..cfg.layers + cfg.mtp {
            let root = format!("model.layers.{layer}");
            let ffn = if cfg.moe[layer] { "share_expert" } else { "mlp" };
            for (module, leaves) in [("self_attn", &["q_proj", "k_proj", "v_proj", "o_proj"][..]),
                (ffn, &["gate_proj", "up_proj", "down_proj"][..])] {
                let present = leaves.iter().filter(|leaf| self.projections
                    .contains_key(&format!("{root}.{module}.{leaf}"))).count();
                ensure!(present == 0 || present == leaves.len(),
                    "Step-5 {root}.{module} mixes native and EXL3 storage; complete operator groups required");
            }
        }
        Ok(())
    }

    /// Native-dense core bytes by role, for plan residency reporting.
    pub fn bytes(&self, mtp_from: usize) -> (u64, u64) {
        self.projections.values().fold((0, 0), |(target, mtp), p| {
            let draft = p.stem.strip_prefix("model.layers.").and_then(|r| r.split('.').next())
                .and_then(|l| l.parse::<usize>().ok()).is_some_and(|l| l >= mtp_from);
            if draft { (target, mtp + p.source_bytes) } else { (target + p.source_bytes, mtp) }
        })
    }
}
