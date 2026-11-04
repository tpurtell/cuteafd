//! Native V4 operand and residency contract shared by header-only admission and
//! the CUDA loader. Attention ownership never moves the split FFN operands.
use super::DeepseekV4Config;
use anyhow::{ensure, Context, Result};
use crate::{placement::{FfnMode, LayerMode, ModeBytes}, SafetensorsTensorMetadata};
use std::collections::BTreeMap;

#[derive(Debug, Clone, Copy)]
pub enum Prep {
    Raw,
    /// Block-FP8 scales for a weight [groups*n, k].
    Scale { n: usize, k: usize, groups: usize },
}

pub struct Source {
    pub operand: &'static str,
    pub tensors: &'static [&'static str],
    pub prep: Prep,
}

pub fn layer_sources(cfg: &DeepseekV4Config, ratio: usize) -> Vec<Source> {
    let (h, q, heads, g, r) = (cfg.dim, cfg.q_lora_rank, cfg.n_heads, cfg.o_groups, cfg.o_lora_rank);
    let w = heads * 512 / g;
    let inter = cfg.moe_inter_dim;
    let mut sources = vec![
        Source { operand: "attn.fn", tensors: &["hc_attn_fn"], prep: Prep::Raw },
        Source { operand: "attn.scale", tensors: &["hc_attn_scale"], prep: Prep::Raw },
        Source { operand: "attn.base", tensors: &["hc_attn_base"], prep: Prep::Raw },
        Source { operand: "attn.norm", tensors: &["attn_norm.weight"], prep: Prep::Raw },
        Source { operand: "ffn.fn", tensors: &["hc_ffn_fn"], prep: Prep::Raw },
        Source { operand: "ffn.scale", tensors: &["hc_ffn_scale"], prep: Prep::Raw },
        Source { operand: "ffn.base", tensors: &["hc_ffn_base"], prep: Prep::Raw },
        Source { operand: "ffn.norm", tensors: &["ffn_norm.weight"], prep: Prep::Raw },
        Source { operand: "w_qkv", tensors: &["attn.wq_a.weight", "attn.wkv.weight"], prep: Prep::Raw },
        Source { operand: "w_qkv_scale", tensors: &["attn.wq_a.scale", "attn.wkv.scale"],
            prep: Prep::Scale { n: q + 512, k: h, groups: 1 } },
        Source { operand: "w_q", tensors: &["attn.wq_b.weight"], prep: Prep::Raw },
        Source { operand: "w_q_scale", tensors: &["attn.wq_b.scale"], prep: Prep::Scale { n: heads * 512, k: q, groups: 1 } },
        Source { operand: "q_norm", tensors: &["attn.q_norm.weight"], prep: Prep::Raw },
        Source { operand: "kv_norm", tensors: &["attn.kv_norm.weight"], prep: Prep::Raw },
        Source { operand: "attn_sink", tensors: &["attn.attn_sink"], prep: Prep::Raw },
        Source { operand: "wo_a", tensors: &["attn.wo_a.weight"], prep: Prep::Raw },
        Source { operand: "wo_a_scale", tensors: &["attn.wo_a.scale"], prep: Prep::Scale { n: r, k: w, groups: g } },
        Source { operand: "wo_b", tensors: &["attn.wo_b.weight"], prep: Prep::Raw },
        Source { operand: "wo_b_scale", tensors: &["attn.wo_b.scale"], prep: Prep::Scale { n: h, k: g * r, groups: 1 } },
        Source { operand: "w13", tensors: &["ffn.shared_experts.w1.weight", "ffn.shared_experts.w3.weight"], prep: Prep::Raw },
        Source { operand: "w13_scale", tensors: &["ffn.shared_experts.w1.scale", "ffn.shared_experts.w3.scale"],
            prep: Prep::Scale { n: 2 * inter, k: h, groups: 1 } },
        Source { operand: "w2", tensors: &["ffn.shared_experts.w2.weight"], prep: Prep::Raw },
        Source { operand: "w2_scale", tensors: &["ffn.shared_experts.w2.scale"], prep: Prep::Scale { n: h, k: inter, groups: 1 } },
        Source { operand: "gate", tensors: &["ffn.gate.weight"], prep: Prep::Raw },
    ];
    if ratio == 4 {
        sources.extend([
            Source { operand: "joint_projection", tensors: &["attn.compressor.wkv.weight", "attn.compressor.wgate.weight",
                "attn.indexer.compressor.wkv.weight", "attn.indexer.compressor.wgate.weight"], prep: Prep::Raw },
            Source { operand: "index_ape", tensors: &["attn.indexer.compressor.ape"], prep: Prep::Raw },
            Source { operand: "index_norm", tensors: &["attn.indexer.compressor.norm.weight"], prep: Prep::Raw },
            Source { operand: "index_w_q", tensors: &["attn.indexer.wq_b.weight"], prep: Prep::Raw },
            Source { operand: "index_w_q_scale", tensors: &["attn.indexer.wq_b.scale"],
                prep: Prep::Scale { n: cfg.index_n_heads * cfg.index_head_dim, k: q, groups: 1 } },
            Source { operand: "index_w_proj", tensors: &["attn.indexer.weights_proj.weight"], prep: Prep::Raw },
        ]);
    } else if ratio == 128 {
        sources.push(Source { operand: "joint_projection",
            tensors: &["attn.compressor.wkv.weight", "attn.compressor.wgate.weight"], prep: Prep::Raw });
    }
    if ratio != 0 {
        sources.extend([
            Source { operand: "main_ape", tensors: &["attn.compressor.ape"], prep: Prep::Raw },
            Source { operand: "main_norm", tensors: &["attn.compressor.norm.weight"], prep: Prep::Raw },
        ]);
    }
    sources
}

/// How a head split shares an operand's checkpoint tensors.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Share {
    /// Every rank holds the whole operand.
    All,
    /// Rows split evenly (per tensor, concatenated): heads, wo groups, shared-expert intermediate.
    Rows,
    /// Columns split evenly: wo_b over the groups, the shared expert's w2 over the intermediate.
    Cols,
}

/// Attention and FFN ownership are independent for whole-attention layers.
pub fn share(operand: &str, attention_split: bool, ffn_split: bool) -> Share {
    match operand {
        "w_q" | "w_q_scale" | "attn_sink" | "wo_a" | "wo_a_scale" if attention_split => Share::Rows,
        "wo_b" | "wo_b_scale" if attention_split => Share::Cols,
        "w13" | "w13_scale" if ffn_split => Share::Rows,
        "w2" | "w2_scale" if ffn_split => Share::Cols,
        _ => Share::All,
    }
}

/// A mode's checkpoint layout, resolved before uploading any operands.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WeightSplit {
    pub attention: bool,
    pub ffn: bool,
    pub owner: Option<usize>,
}

impl WeightSplit {
    pub fn for_mode(mode: crate::placement::LayerMode, ratio: usize, ranks: usize) -> Result<Self> {
        use crate::placement::{FfnMode, LayerMode};
        ensure!(ranks == 1 || ranks == 2, "V4 weights need one or two ranks");
        Ok(match mode {
            LayerMode::HeadSplit => {
                ensure!(ranks == 2, "V4 head split needs two ranks");
                Self { attention: true, ffn: true, owner: None }
            }
            LayerMode::ContextSplit => {
                ensure!(ranks == 2 && ratio == 4, "V4 context weights need two ranks and C4");
                Self { attention: true, ffn: true, owner: None }
            }
            LayerMode::Whole { gpu, ffn } => {
                let owner = usize::from(gpu);
                ensure!(owner < ranks && (ranks == 1 || ratio == 4), "V4 attention owner or ratio unsupported");
                ensure!((ranks == 1 && ffn == FfnMode::Owner) || (ranks == 2 && ffn == FfnMode::Split),
                    "V4 owned attention keeps the existing FFN placement");
                Self { attention: false, ffn: ffn == FfnMode::Split, owner: Some(owner) }
            }
        })
    }

    pub fn operand(self, operand: &str, rank: usize) -> Option<Share> {
        let ffn = ffn_operand(operand);
        if !ffn && self.owner.is_some_and(|owner| owner != rank) {
            None
        } else {
            Some(share(operand, self.attention, self.ffn))
        }
    }
}

pub fn scale_extents(operand: &str, kind: Share, n: usize, k: usize, groups: usize, ranks: usize)
    -> (usize, usize, usize) {
    match (operand, kind) {
        ("wo_a_scale", Share::Rows) => (n, k, groups / ranks),
        (_, Share::Rows) => (n / ranks, k, groups),
        (_, Share::Cols) => (n, k / ranks, groups),
        _ => (n, k, groups),
    }
}


/// Operands needed by both FFN halves, independent of attention placement.
pub fn ffn_operand(operand: &str) -> bool {
    operand.starts_with("ffn.") || operand.starts_with("gate")
        || matches!(operand, "w13" | "w13_scale" | "w2" | "w2_scale")
}

/// One final device allocation, after concatenation, slicing and scale prep.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResidentOperand {
    pub rank: usize,
    pub layer: Option<usize>,
    pub draft_stage: Option<usize>,
    pub operand: String,
    pub group: String,
    pub format: String,
    pub bytes: u64,
    pub attention: bool,
}

fn operand_bytes(source: &Source, prefix: &str, kind: Share, ranks: usize,
    headers: &BTreeMap<&str, &SafetensorsTensorMetadata>) -> Result<(u64, String)> {
    let mut raw = 0u64;
    let mut dtype = None;
    for suffix in source.tensors {
        let name = format!("{prefix}.{suffix}");
        let tensor = headers.get(name.as_str()).with_context(|| format!("V4 resident tensor missing: {name}"))?;
        if !matches!(kind, Share::All) {
            ensure!(tensor.byte_length % ranks as u64 == 0, "{name}: uneven byte split");
            match kind {
                Share::Rows => ensure!(tensor.shape.first().is_some_and(|n| n % ranks == 0),
                    "{name}: uneven row split"),
                Share::Cols => ensure!(tensor.shape.len() == 2 && tensor.shape[1] % ranks == 0
                    && tensor.byte_length == (tensor.shape[0] as u64).checked_mul(tensor.shape[1] as u64)
                        .context("V4 column extent overflow")?, "{name}: invalid byte-column split"),
                Share::All => unreachable!(),
            }
        }
        let bytes = tensor.byte_length / if kind == Share::All { 1 } else { ranks as u64 };
        raw = raw.checked_add(bytes).context("V4 resident operand overflow")?;
        let format = format!("{:?}", tensor.dtype).to_ascii_lowercase();
        ensure!(dtype.as_ref().is_none_or(|previous| previous == &format), "{prefix}: mixed operand dtypes");
        dtype = Some(format);
    }
    Ok(match source.prep {
        Prep::Raw => (raw.max(16), dtype.context("empty V4 operand")?),
        Prep::Scale { n, k, groups } => {
            let (n, k, groups) = scale_extents(source.operand, kind, n, k, groups, ranks);
            ensure!(n % 128 == 0 && k % 128 == 0, "{prefix}: invalid MMA scale extents");
            let expected = (groups as u64).checked_mul(n as u64 / 128)
                .and_then(|n| n.checked_mul(k as u64 / 128)).context("V4 scale grid overflow")?;
            ensure!(raw == expected, "{prefix}.{}: scale bytes {raw} do not cover {expected} blocks", source.operand);
            (raw.checked_mul(512).context("V4 scale expansion overflow")?, "fp8-mma-scale".into())
        }
    })
}

/// Exact uploaded operands. Extra checkpoint heads (confidence heads, etc.)
/// are not resident merely because they are classified as speculator tensors.
pub fn resident_operands<'a>(cfg: &DeepseekV4Config, modes: &[LayerMode], ranks: usize, dspark: bool,
    tensors: impl IntoIterator<Item = &'a SafetensorsTensorMetadata>) -> Result<Vec<ResidentOperand>> {
    ensure!(modes.len() == cfg.n_layers, "V4 weight mode count differs from config");
    ensure!(cfg.compress_ratios.len() == cfg.n_layers, "V4 ratio count differs from config");
    let layouts = modes.iter().zip(&cfg.compress_ratios).map(|(&mode, &ratio)|
        WeightSplit::for_mode(mode, ratio, ranks)).collect::<Result<Vec<_>>>()?;
    let headers: BTreeMap<_, _> = tensors.into_iter().map(|t| (t.name.as_str(), t)).collect();
    let mut out = Vec::new();
    let mut block = |prefix: &str, layer: Option<usize>, draft_stage: Option<usize>, ratio, hash, layout: WeightSplit, count| -> Result<()> {
        for source in layer_sources(cfg, ratio) {
            for rank in 0..count {
                let Some(kind) = layout.operand(source.operand, rank) else { continue };
                let (bytes, format) = operand_bytes(&source, prefix, kind, count, &headers)?;
                out.push(ResidentOperand { rank, layer, draft_stage, operand: source.operand.into(),
                    group: if layer.is_some() { "target" } else { "speculator" }.into(),
                    format, bytes, attention: !ffn_operand(source.operand) });
            }
        }
        let suffix = if hash { "ffn.gate.tid2eid" } else { "ffn.gate.bias" };
        let name = format!("{prefix}.{suffix}");
        let tensor = headers.get(name.as_str()).with_context(|| format!("V4 resident tensor missing: {name}"))?;
        if hash { ensure!(tensor.byte_length % 8 == 0, "{name}: incomplete I64 expert ids"); }
        for rank in 0..count {
            out.push(ResidentOperand { rank, layer, draft_stage, operand: if hash { "gate.tid2eid" } else { "gate.bias" }.into(),
                group: if layer.is_some() { "target" } else { "speculator" }.into(),
                format: if hash { "i32".into() } else { format!("{:?}", tensor.dtype).to_ascii_lowercase() },
                bytes: (tensor.byte_length / if hash { 2 } else { 1 }).max(16), attention: false });
        }
        Ok(())
    };
    for (layer, &layout) in layouts.iter().enumerate() {
        block(&format!("layers.{layer}"), Some(layer), None, cfg.compress_ratios[layer], cfg.is_hash_layer(layer), layout, ranks)?;
    }
    let mut names = vec![("embed.weight", "embedding"), ("head.weight", "lm_head"),
        ("hc_head_fn", "hyper_connection"), ("hc_head_scale", "hyper_connection"),
        ("hc_head_base", "hyper_connection"), ("norm.weight", "norm")];
    if dspark && cfg.dspark_block_size > 0 && headers.contains_key("mtp.0.main_proj.weight") {
        let whole = WeightSplit::for_mode(LayerMode::Whole { gpu: 0, ffn: FfnMode::Owner }, 0, 1)?;
        for stage in 0..3 { block(&format!("mtp.{stage}"), None, Some(stage), 0, false, whole, 1)?; }
        names.extend([("mtp.0.main_proj.weight", "speculator"), ("mtp.0.main_proj.scale", "speculator"),
            ("mtp.0.main_norm.weight", "speculator"), ("mtp.2.hc_head_fn", "speculator"),
            ("mtp.2.hc_head_scale", "speculator"), ("mtp.2.hc_head_base", "speculator"),
            ("mtp.2.norm.weight", "speculator"), ("mtp.2.markov_head.markov_w1.weight", "speculator"),
            ("mtp.2.markov_head.markov_w2.weight", "speculator")]);
    }
    for (name, group) in names {
        let tensor = headers.get(name).with_context(|| format!("V4 resident tensor missing: {name}"))?;
        out.push(ResidentOperand { rank: 0, layer: None, draft_stage: None, operand: name.into(), group: group.into(),
            format: format!("{:?}", tensor.dtype).to_ascii_lowercase(), bytes: tensor.byte_length.max(16), attention: false });
    }
    Ok(out)
}

/// C4 attention costs, independent of the invariant FFN/head/drafter baseline.
/// Feeding these to the shared solver balances real owner-local weights.
pub fn attention_costs<'a>(cfg: &DeepseekV4Config, ranks: usize,
    tensors: impl IntoIterator<Item = &'a SafetensorsTensorMetadata>) -> Result<Vec<ModeBytes>> {
    let tensors = tensors.into_iter().collect::<Vec<_>>();
    let modes = vec![if ranks == 1 { LayerMode::Whole { gpu: 0, ffn: FfnMode::Owner } }
        else { LayerMode::HeadSplit }; cfg.n_layers];
    let split = resident_operands(cfg, &modes, ranks, false, tensors.iter().copied())?;
    let mut modes_whole = modes;
    if ranks == 2 {
        for (layer, &ratio) in cfg.compress_ratios.iter().enumerate() {
            if ratio == 4 { modes_whole[layer] = LayerMode::Whole { gpu: 0, ffn: FfnMode::Split }; }
        }
    }
    let whole = resident_operands(cfg, &modes_whole, ranks, false, tensors)?;
    let mut costs = vec![ModeBytes::default(); cfg.n_layers];
    for item in split.iter().filter(|i| i.attention) {
        let layer = item.layer.context("V4 attention without layer")?;
        if cfg.compress_ratios[layer] == 4 {
            costs[layer].split[item.rank] = costs[layer].split[item.rank].checked_add(item.bytes)
                .context("V4 split attention weight overflow")?;
        }
    }
    for item in whole.iter().filter(|i| i.attention && i.rank == 0) {
        let layer = item.layer.context("V4 attention without layer")?;
        if cfg.compress_ratios[layer] == 4 {
            costs[layer].whole = costs[layer].whole.checked_add(item.bytes)
                .context("V4 whole attention weight overflow")?;
        }
    }
    Ok(costs)
}

#[cfg(test)]
mod tests {
    use super::*;
    use cuteafd_core::DType;

    fn fixture() -> (DeepseekV4Config, Vec<SafetensorsTensorMetadata>) {
        let cfg = DeepseekV4Config::from_model_args(&serde_json::json!({
            "vocab_size":129280,"dim":4096,"moe_inter_dim":2048,"n_layers":4,"n_hash_layers":1,
            "n_heads":64,"n_routed_experts":256,"n_shared_experts":1,"n_activated_experts":6,
            "score_func":"sqrtsoftplus","route_scale":1.5,"swiglu_limit":10.0,"q_lora_rank":1024,
            "head_dim":512,"rope_head_dim":64,"o_groups":8,"o_lora_rank":1024,"window_size":128,
            "original_seq_len":65536,"rope_theta":10000,"rope_factor":16,"beta_fast":32,"beta_slow":1,
            "index_n_heads":64,"index_head_dim":128,"index_topk":512,"hc_mult":4,"hc_sinkhorn_iters":20,
            "compress_rope_theta":160000,"compress_ratios":[0,4,128,4],"dspark_block_size":5
        }), 3).unwrap();
        let mut tensors = Vec::new();
        let mut add = |name: String, dtype, shape: Vec<usize>, byte_length| {
            tensors.push(SafetensorsTensorMetadata { name, dtype, shape, byte_offset: 0, byte_length });
        };
        for (prefix, ratio, hash) in (0..cfg.n_layers).map(|layer|
            (format!("layers.{layer}"), cfg.compress_ratios[layer], cfg.is_hash_layer(layer)))
            .chain((0..3).map(|stage| (format!("mtp.{stage}"), 0, false))) {
            for source in layer_sources(&cfg, ratio) {
                for &suffix in source.tensors {
                    let (dtype, shape, bytes) = match source.prep {
                        Prep::Raw => (DType::F8E4M3, vec![256, 256], 256 * 256),
                        Prep::Scale { n, k, groups } => {
                            let rows = groups * n / 128 / source.tensors.len();
                            (DType::F8E8M0, vec![rows, k / 128], (rows * k / 128) as u64)
                        }
                    };
                    add(format!("{prefix}.{suffix}"), dtype, shape, bytes);
                }
            }
            if hash { add(format!("{prefix}.ffn.gate.tid2eid"), DType::I64, vec![64, 6], 64 * 6 * 8); }
            else { add(format!("{prefix}.ffn.gate.bias"), DType::F32, vec![256], 1024); }
        }
        for name in ["embed.weight", "head.weight", "hc_head_fn", "hc_head_scale", "hc_head_base", "norm.weight",
            "mtp.0.main_proj.weight", "mtp.0.main_proj.scale", "mtp.0.main_norm.weight",
            "mtp.2.hc_head_fn", "mtp.2.hc_head_scale", "mtp.2.hc_head_base", "mtp.2.norm.weight",
            "mtp.2.markov_head.markov_w1.weight", "mtp.2.markov_head.markov_w2.weight",
            "mtp.2.confidence_head.weight"] {
            add(name.into(), DType::Bf16, vec![256], 512);
        }
        (cfg, tensors)
    }

    #[test]
    fn whole_c4_keeps_ffn_halves_and_removes_every_nonowner_attention_operand() -> Result<()> {
        let (cfg, tensors) = fixture();
        let split_modes = vec![LayerMode::HeadSplit; cfg.n_layers];
        let split = resident_operands(&cfg, &split_modes, 2, true, &tensors)?;
        for owner in 0..2 {
            let mut modes = split_modes.clone();
            modes[1] = LayerMode::Whole { gpu: owner, ffn: FfnMode::Split };
            let whole = resident_operands(&cfg, &modes, 2, true, &tensors)?;
            assert_eq!(whole.iter().filter(|i| i.layer != Some(1) || !i.attention).collect::<Vec<_>>(),
                split.iter().filter(|i| i.layer != Some(1) || !i.attention).collect::<Vec<_>>());
            assert!(whole.iter().filter(|i| i.layer == Some(1) && i.attention).all(|i| i.rank == owner as usize));
            for item in whole.iter().filter(|i| i.layer == Some(1) && i.attention) {
                let previous = split.iter().find(|i| i.rank == item.rank && i.layer == item.layer
                    && i.operand == item.operand).unwrap();
                let share = share(&item.operand, true, true);
                assert_eq!(item.bytes, previous.bytes * if share == Share::All { 1 } else { 2 });
            }
        }
        Ok(())
    }

    #[test]
    fn header_inventory_matches_native_scale_and_hash_conversion_and_exact_drafter_subset() -> Result<()> {
        let (cfg, tensors) = fixture();
        let inventory = resident_operands(&cfg, &vec![LayerMode::HeadSplit; cfg.n_layers], 2, true, &tensors)?;
        let scale = inventory.iter().find(|i| i.layer == Some(1) && i.rank == 0 && i.operand == "w_q_scale").unwrap();
        let header = tensors.iter().find(|t| t.name == "layers.1.attn.wq_b.scale").unwrap();
        assert_eq!(scale.bytes, header.byte_length / 2 * 512);
        let ids = inventory.iter().find(|i| i.layer == Some(0) && i.operand == "gate.tid2eid").unwrap();
        assert_eq!(ids.bytes, 64 * 6 * 4);
        assert_eq!(ids.format, "i32");
        assert!(inventory.iter().filter(|i| i.group == "speculator").all(|i| i.rank == 0));
        assert!(!inventory.iter().any(|i| i.operand.contains("confidence_head")));
        let raw_scale = inventory.iter().find(|i| i.operand == "mtp.0.main_proj.scale").unwrap();
        assert_eq!(raw_scale.bytes, 512);
        let off = resident_operands(&cfg, &vec![LayerMode::HeadSplit; cfg.n_layers], 2, false, &tensors)?;
        assert!(!off.iter().any(|i| i.group == "speculator"));
        Ok(())
    }

    #[test]
    fn c4_mode_costs_are_real_inventory_not_hidden_baseline() -> Result<()> {
        let (cfg, tensors) = fixture();
        let costs = attention_costs(&cfg, 2, &tensors)?;
        for (layer, cost) in costs.iter().enumerate() {
            if cfg.compress_ratios[layer] == 4 {
                assert!(cost.whole > cost.split[0]);
                assert!(cost.whole < cost.split.iter().sum());
                assert_eq!(cost.split[0], cost.split[1]);
            } else { assert_eq!(*cost, ModeBytes::default()); }
        }
        Ok(())
    }

    #[test]
    fn explicit_weight_demands_equal_upload_inventory_under_both_owners() -> Result<()> {
        let (cfg, tensors) = fixture();
        let weights = WeightInventory::from_headers(&cfg, 2, true, &tensors)?;
        assert!(weights.fixed.iter().any(|d| d.gpu == 0 && d.group == "V4 speculator weights"));
        assert!(!weights.fixed.iter().any(|d| d.gpu == 1 && d.group == "V4 speculator weights"));
        for owner in 0..2 {
            let mut modes = vec![LayerMode::HeadSplit; cfg.n_layers];
            modes[1] = LayerMode::Whole { gpu: owner, ffn: FfnMode::Split };
            let inventory = resident_operands(&cfg, &modes, 2, true, &tensors)?;
            let totals = (0..2).map(|rank| inventory.iter().filter(|i|
                i.rank == rank && i.group != "embedding").map(|i| i.bytes).sum()).collect::<Vec<u64>>();
            assert_eq!(weights.loaded_bytes(&modes, 2)?, totals);
        }
        Ok(())
    }

    #[test]
    fn startup_keeps_host_owners_out_of_gpu_and_sums_retained_weights_with_scratch() {
        use crate::shared::dense_read::STEP_BYTES;
        let (cfg, tensors) = fixture();
        for ranks in [1, 2] {
            for dspark in [false, true] {
                let weights = WeightInventory::from_headers(&cfg, ranks, dspark, &tensors).unwrap();
                let modes = vec![if ranks == 1 { LayerMode::Whole { gpu: 0, ffn: FfnMode::Owner } }
                    else { LayerMode::HeadSplit }; cfg.n_layers];
                let resident = weights.loaded_bytes(&modes, ranks).unwrap();
                let phases = weights.startup.device_phases(ranks).unwrap();
                assert_eq!(phases.len(), ranks);
                for rank in 0..ranks {
                    assert_eq!(phases[rank].bytes, resident[rank] + STEP_BYTES as u64);
                }
                assert_eq!(weights.startup.host.iter().filter(|item| item.name == "weight-push").count(), ranks);
                assert!(weights.startup.host_bytes().unwrap() >= ranks as u64 * 128 * 1024 * 1024);
                assert!(weights.startup.host.iter().any(|item| item.name == "staging/coordinator-dense-status" && item.bytes == 4));
            }
        }
    }

    #[test]
    fn inventory_rejects_aggregate_weight_overflow() {
        let (cfg, mut tensors) = fixture();
        tensors.iter_mut().find(|t| t.name == "layers.1.hc_attn_fn").unwrap().byte_length = u64::MAX - 1024;
        let error = attention_costs(&cfg, 2, &tensors).unwrap_err();
        assert!(error.to_string().contains("attention weight overflow"), "{error:#}");
        tensors.iter_mut().find(|t| t.name == "layers.1.hc_attn_fn").unwrap().byte_length = 65_536;
        tensors.iter_mut().find(|t| t.name == "layers.0.hc_ffn_fn").unwrap().byte_length = u64::MAX - 1024;
        let error = WeightInventory::from_headers(&cfg, 2, false, &tensors).unwrap_err();
        assert!(error.to_string().contains("invariant weight overflow"), "{error:#}");
    }

    #[test]
    fn inventory_refuses_missing_headers_bad_scales_modes_and_column_shapes() {
        let (cfg, tensors) = fixture();
        let modes = vec![LayerMode::HeadSplit; cfg.n_layers];
        assert!(resident_operands(&cfg, &modes[..3], 2, false, &tensors).is_err());
        let mut missing = tensors.clone();
        missing.retain(|t| t.name != "layers.1.attn.indexer.weights_proj.weight");
        assert!(resident_operands(&cfg, &modes, 2, false, &missing).is_err());
        let mut malformed = tensors.clone();
        malformed.iter_mut().find(|t| t.name == "layers.1.attn.wq_b.scale").unwrap().byte_length += 2;
        assert!(resident_operands(&cfg, &modes, 2, false, &malformed).is_err());
        let mut malformed = tensors;
        malformed.iter_mut().find(|t| t.name == "layers.1.attn.wo_b.weight").unwrap().shape[1] = 255;
        assert!(resident_operands(&cfg, &modes, 2, false, &malformed).is_err());
        let mut modes = modes;
        modes[2] = LayerMode::Whole { gpu: 1, ffn: FfnMode::Split };
        assert!(resident_operands(&cfg, &modes, 2, false, &missing).is_err());
    }
}

/// Invariant weights are named fixed demands (head/drafter only on lead,
/// shared FFN on both); C4 attention is a per-mode layer charge, not baseline.
#[derive(Debug, Clone)]
pub struct WeightInventory {
    pub c4_attention: Vec<ModeBytes>,
    pub fixed: Vec<crate::placement::Demand>,
    pub startup: StartupInventory,
}

/// Loading-only owners; host mappings must never consume the GPU budget.
#[derive(Debug, Clone, Default)]
pub struct StartupInventory {
    pub device: Vec<StartupItem>,
    pub host: Vec<StartupItem>,
}

#[derive(Debug, Clone)]
pub struct StartupItem {
    pub rank: usize,
    pub name: &'static str,
    pub bytes: u64,
}

impl StartupInventory {
    fn from_operands(operands: &[ResidentOperand], ranks: usize) -> Result<Self> {
        let mut inventory = Self::default();
        for rank in 0..ranks {
            let mut retained = 0u64;
            let mut staged = 0u64;
            let mut scratch = 0;
            let mut status = 0;
            for item in operands.iter().filter(|item| item.rank == rank) {
                // Embedding is already subtracted from the admission baseline.
                if item.group != "embedding" {
                    retained = retained.checked_add(item.bytes).context("V4 startup weights overflow")?;
                }
                let source = if item.format == "fp8-mma-scale" {
                    scratch = crate::shared::dense_read::STEP_BYTES as u64;
                    item.bytes / 512
                } else if item.operand == "gate.tid2eid" {
                    status = 4;
                    item.bytes.checked_mul(2).context("V4 startup IDs overflow")?
                } else { 0 };
                staged = staged.checked_add(source).context("V4 startup staging overflow")?;
            }
            inventory.device.extend([
                StartupItem { rank, name: "weights/coordinator-dense", bytes: retained },
                StartupItem { rank, name: "staging/coordinator-dense-device", bytes: scratch },
            ]);
            inventory.host.extend([
                StartupItem { rank, name: "staging/coordinator-dense", bytes: staged },
                StartupItem { rank, name: "staging/coordinator-dense-status", bytes: status },
                StartupItem { rank, name: "weight-push", bytes:
                    (cuteafd_ffi::push_ring::DEFAULT_SLOTS * cuteafd_ffi::push_ring::DEFAULT_SLOT_BYTES) as u64 },
            ]);
        }
        Ok(inventory)
    }

    pub fn device_bytes(&self, rank: usize) -> Result<u64> {
        self.device.iter().filter(|item| item.rank == rank).try_fold(0u64, |sum, item|
            sum.checked_add(item.bytes).context("V4 startup device overflow"))
    }

    pub fn device_phases(&self, ranks: usize) -> Result<Vec<crate::placement::Demand>> {
        use cuteafd_core::memory_layout::{Basis, Category};
        (0..ranks).map(|rank| Ok(crate::placement::Demand::new(rank as u8, Category::Weights,
            "V4 coordinator dense loading", self.device_bytes(rank)?, Basis::Exact))).collect()
    }

    pub fn host_bytes(&self) -> Result<u64> {
        self.host.iter().try_fold(0u64, |sum, item|
            sum.checked_add(item.bytes).context("V4 startup host overflow"))
    }

    pub fn notes(&self) -> Result<Vec<String>> {
        let mut notes = self.device.iter().map(|item| format!(
            "V4 startup device rtx{} {}: {} bytes (loading only)", item.rank, item.name, item.bytes))
            .chain(self.host.iter().map(|item| format!(
                "V4 startup host rank{} {}: {} bytes (loading only)", item.rank, item.name, item.bytes)))
            .collect::<Vec<_>>();
        notes.push(format!("V4 startup host bound: {} bytes; conservative staged source footprint, exact physical extents admitted at receive", self.host_bytes()?));
        Ok(notes)
    }
}

impl WeightInventory {
    pub fn from_headers<'a>(cfg: &DeepseekV4Config, ranks: usize, dspark: bool,
        tensors: impl IntoIterator<Item = &'a SafetensorsTensorMetadata>) -> Result<Self> {
        use cuteafd_core::memory_layout::{Basis, Category};
        let tensors = tensors.into_iter().collect::<Vec<_>>();
        let modes = vec![if ranks == 1 { LayerMode::Whole { gpu: 0, ffn: FfnMode::Owner } }
            else { LayerMode::HeadSplit }; cfg.n_layers];
        let operands = resident_operands(cfg, &modes, ranks, dspark, tensors.iter().copied())?;
        let c4_attention = attention_costs(cfg, ranks, tensors)?;
        let mut groups = BTreeMap::<(usize, String), u64>::new();
        for item in &operands {
            if item.group == "embedding" || item.layer.is_some_and(|layer|
                item.attention && cfg.compress_ratios[layer] == 4) { continue; }
            let bytes = groups.entry((item.rank, format!("V4 {} weights", item.group))).or_default();
            *bytes = bytes.checked_add(item.bytes).context("V4 invariant weight overflow")?;
        }
        let fixed = groups.into_iter().map(|((gpu, group), bytes)|
            crate::placement::Demand::new(gpu as u8, Category::Weights, group, bytes, Basis::Exact)
                .with_resident_weights(crate::placement::ResidentWeights(bytes))).collect();
        let startup = StartupInventory::from_operands(&operands, ranks)?;
        Ok(Self { c4_attention, fixed, startup })
    }

    /// Final uploaded non-embedding bytes for the selected owners. A post-load
    /// CUDA sample is rebased by these exact bytes to avoid charging twice.
    pub fn loaded_bytes(&self, modes: &[LayerMode], ranks: usize) -> Result<Vec<u64>> {
        ensure!(modes.len() == self.c4_attention.len(), "V4 loaded weight mode count differs");
        let mut bytes = vec![0u64; ranks];
        for demand in &self.fixed {
            let gpu = usize::from(demand.gpu);
            ensure!(gpu < ranks, "V4 fixed weight on missing rank");
            bytes[gpu] = bytes[gpu].checked_add(demand.bytes).context("V4 fixed weight overflow")?;
        }
        for (&mode, cost) in modes.iter().zip(&self.c4_attention) {
            match mode {
                LayerMode::HeadSplit | LayerMode::ContextSplit => {
                    ensure!(ranks == 2, "V4 split weights need two ranks");
                    for rank in 0..ranks { bytes[rank] = bytes[rank].checked_add(cost.split[rank])
                        .context("V4 split weight overflow")?; }
                }
                LayerMode::Whole { gpu, .. } => {
                    let gpu = usize::from(gpu);
                    ensure!(gpu < ranks, "V4 weight owner absent");
                    bytes[gpu] = bytes[gpu].checked_add(cost.whole).context("V4 owned weight overflow")?;
                }
            }
        }
        Ok(bytes)
    }
}
