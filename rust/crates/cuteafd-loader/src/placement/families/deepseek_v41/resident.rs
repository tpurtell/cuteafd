//! V4.1's default CED placement and native resident representations. These
//! formulas mirror the startup owners, including packed scale copies.
use crate::plan::Checkpoint;
use cuteafd_core::memory_layout::{Basis, Category, Item};
use std::collections::BTreeMap;

pub const DEFAULT_POOL_TOKENS: u64 = 2 * 1_048_576;

/// One mode parser for checkpoint residency, startup packing and native execution
/// (`CUTEAFD_V41_FP8_HEAD`): `off` keeps the BF16 head; `draft` adds an E4M3 copy
/// for dSpark; `all` (default) keeps only the E4M3 copy, for both; `packed` keeps
/// only the exact 12-bit copy of the BF16 head (`cuteafd_core::packed_head`):
/// target logits read it exactly, dSpark its coarse plane.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Fp8Head { Off, Draft, All, Packed }

/// Vocabulary rows are hidden_size (5120) wide.
const HEAD_WIDTH: u64 = 5120;

impl Fp8Head {
    pub fn parse(mode: &str) -> Self {
        match mode {
            "off" | "0" | "bf16" => Self::Off,
            "draft" => Self::Draft,
            "packed" => Self::Packed,
            _ => Self::All,
        }
    }

    pub fn configured() -> Self {
        Self::parse(&std::env::var("CUTEAFD_V41_FP8_HEAD").unwrap_or_default())
    }

    /// Target logits project through a resident copy (and its scratch), not BF16.
    pub fn target(self) -> bool { matches!(self, Self::All | Self::Packed) }
    /// dSpark projects through a resident copy.
    pub fn draft(self) -> bool { self != Self::Off }

    /// One E4M3 value and FP32 scales per 128-K block.
    pub fn packed_bytes(source: u64) -> u64 { source / 2 + source / 64 }

    /// The exact 12-bit copy of `source` BF16 bytes: both planes, bases, row
    /// pointers, an exception table of one entry per 1024 weights and the
    /// pack's status, as the loader allocates them.
    pub fn packed12_bytes(source: u64) -> u64 {
        cuteafd_core::packed_head::PackedHeadLayout::bytes(source / (2 * HEAD_WIDTH), HEAD_WIDTH)
    }

    /// Peak while packing: the BF16 source and the copy's destination.
    pub fn load_bytes(self, source: u64) -> u64 {
        source + match self {
            Self::Off => 0,
            Self::Draft | Self::All => Self::packed_bytes(source),
            Self::Packed => Self::packed12_bytes(source),
        }
    }

    pub fn resident(self, source: u64) -> (u64, &'static str) {
        match self {
            Self::Off => (source, "bf16"),
            Self::Draft => (self.load_bytes(source), "bf16+fp8-row128"),
            Self::All => (Self::packed_bytes(source), "fp8-row128"),
            Self::Packed => (Self::packed12_bytes(source), "bf16-packed12"),
        }
    }
}

/// Native MXFP4 packer storage from the measured storage policy. Full RTX
/// experts and independently packed TP slices each retain their own scale bases.
pub fn native_expert_bytes(experts: u64, hidden: u64, intermediate: u64) -> u64 {
    crate::formats::scale_codes::expert_bytes(experts, hidden, intermediate)
        .expect("validated V4.1 coded expert geometry")
}

/// NVFP4 native planes mirror `Nvfp4Side::plane_sizes`: fused FC1, FC2,
/// F8_128x4 scale planes and four per-expert FP32 vectors. TP3/TP6 are exact;
/// TP4 uses the default 576->640 kernel extent.
pub fn nvfp4_expert_bytes(experts: u64, hidden: u64, intermediate: u64) -> u64 {
    let stored = intermediate.div_ceil(128) * 128;
    experts * (3 * hidden * stored / 2
        + (2 * stored).div_ceil(128) * 128 * (hidden / 16).div_ceil(4) * 4
        + hidden.div_ceil(128) * 128 * (stored / 16).div_ceil(4) * 4 + 16)
}

pub fn packed_expert_bytes(package: &str, experts: u64, hidden: u64, intermediate: u64) -> Option<u64> {
    if package.starts_with("v41:mxfp4") {
        Some(native_expert_bytes(experts, hidden, intermediate))
    } else if package.starts_with("v41:nvfp4") {
        Some(nvfp4_expert_bytes(experts, hidden, intermediate))
    } else { None }
}

/// Default backbone/dSpark weights, without routed backbone experts. Vision
/// is admitted separately by the encoder placement. Target embedding lives on RTX0, target normalization
/// on the decoder (RTX1 under CED), and vocabulary is evenly partitioned
/// in the default single-copy FP8 representation.
/// dSpark weights stay on the decoder; default experts are full copies there.
/// Unsupported native format changes return None instead of an exact claim.
pub fn resident_weights(checkpoint: &Checkpoint, ranks: usize, dspark: bool)
    -> Option<Vec<Vec<Item>>> {
    resident_headers(&checkpoint.config, checkpoint.tensors.iter().map(|tensor| &tensor.meta), ranks, dspark)
}

/// The serving catalog and the offline checkpoint enumerate the same source
/// headers; neither caller needs to reopen payloads or reconstruct allocations.
pub fn resident_headers<'a>(config: &serde_json::Value,
    tensors: impl IntoIterator<Item = &'a crate::SafetensorsTensorMetadata>,
    ranks: usize, dspark: bool) -> Option<Vec<Vec<Item>>> {
    let config = config.get("text_config").unwrap_or(config);
    let tensors = tensors.into_iter().collect::<Vec<_>>();
    if ![1, 2].contains(&ranks) || config["num_hidden_layers"].as_u64() != Some(40)
        || config["hidden_size"].as_u64() != Some(5120)
        || tensors.iter().any(|t| {
            let name = &t.name;
            let backbone_expert = name.starts_with("layers.") && name.contains(".ffn.experts.")
                || name.starts_with("model.layers.") && name.contains(".mlp.experts.");
            !backbone_expert && (dspark || !name.starts_with("mtp."))
                && !name.contains(".engram.embed.")
                && (name.ends_with("weight_scale_2") || name.contains(".qweight") || name.ends_with(".trellis"))
        }) {
        return None;
    }
    let decoder = ranks - 1;
    let head_mode = Fp8Head::configured();
    let mut totals: Vec<BTreeMap<(Category, String, String), u64>> = vec![BTreeMap::new(); ranks];
    let mut add = |rank: usize, category, group: &str, format: &str, bytes: u64| {
        *totals[rank].entry((category, group.into(), format.into())).or_default() += bytes;
    };
    let mut draft_experts = false;
    for tensor in tensors {
        let name = tensor.name.as_str();
        let mut bytes = tensor.byte_length;
        let format = format!("{:?}", tensor.dtype).to_ascii_lowercase();
        if name.starts_with("mtp.") {
            if !dspark { continue; }
            if name.contains(".ffn.experts.") { draft_experts = true; continue; }
            // Auxiliary tensors all remain live, but only stage zero's main
            // projection is executed/packed; attention/shared matrices pack at
            // every stage. Grouped wo_a uses its stored n*k/groups extent.
            if name.ends_with(".weight") && format.contains("e4m3")
                && (!name.ends_with(".main_proj.weight") || name == "mtp.0.main_proj.weight") {
                bytes += tensor.shape.iter().product::<usize>() as u64 / 32;
            }
            add(decoder, Category::Drafter, "dspark-auxiliary", &format, bytes);
            continue;
        }
        if name == "head.weight" {
            let (resident, head_format) = head_mode.resident(bytes / ranks as u64);
            for rank in 0..ranks { add(rank, Category::Weights, "lm_head", head_format, resident); }
            continue;
        }
        if name == "embed.weight" { add(0, Category::Embedding, "embedding", &format, bytes); continue; }
        if name == "norm.weight" { add(decoder, Category::Weights, "norm", &format, bytes); continue; }
        if name.starts_with("vision.") || name.starts_with("aligner.") || name.starts_with("image_") {
            continue;
        }
        let Some((layer, rest)) = name.strip_prefix("layers.").and_then(|s| s.split_once('.')) else { continue; };
        let layer: usize = layer.parse().ok()?;
        if layer >= 40 || rest.contains("ffn.experts.") || rest.starts_with("engram.embed.") { continue; }
        let source = [2, 8, 14, 20].contains(&layer);
        if rest.starts_with("attn.compressor.") && !source { continue; }
        if rest.starts_with("attn.indexer.") {
            if rest.contains(".wq_b.") || rest.contains(".weights_proj.") {
                if ![2, 8, 14, 20, 24, 28, 32, 36].contains(&layer) { continue; }
            } else if !source { continue; }
        }
        let (category, group) = if rest.starts_with("engram.") { (Category::Tables, "engram-projections") }
            else if rest.starts_with("ffn.shared_experts.") { (Category::Weights, "shared_expert") }
            else if rest.starts_with("ffn.gate.") { (Category::Weights, "router") }
            else if rest.starts_with("hc_") || rest.ends_with("norm.weight") { (Category::Weights, "norm-hc") }
            else if rest.starts_with("attn.indexer.") { (Category::Weights, "indexer") }
            else if rest.starts_with("attn.compressor.") { (Category::Weights, "compressor") }
            else { (Category::Weights, "attention") };
        if name.ends_with(".weight") && format.contains("e4m3") {
            bytes += tensor.shape.iter().product::<usize>() as u64 / 32;
        }
        if ranks == 2 && group == "shared_expert" {
            // Shared TP2 copies retain the packed scales only; checkpoint
            // scale tensors are transient. Three FP8 matrices per rank.
            if name.ends_with(".scale") { continue; }
            for rank in 0..2 { add(rank, category, group, &format, bytes / 2); }
        } else { add(if ranks == 2 && layer >= 20 { 1 } else { 0 }, category, group, &format, bytes); }
    }
    if draft_experts {
        let experts = config["dspark_n_routed_experts"].as_u64()?;
        let intermediate = config["moe_intermediate_size"].as_u64()?;
        add(decoder, Category::Drafter, "dspark-experts", "mxfp4-packed", 3 * native_expert_bytes(experts, 5120, intermediate));
    }
    Some(totals.into_iter().map(|rank| rank.into_iter().map(|((category, group, format), bytes)|
        Item::new(category, group, format, bytes, Basis::Formula)).collect()).collect())
}

/// Target snapshots always live on RTX0, even with partitioned live caches.
/// dSpark snapshots follow its decoder-owned windows (RTX1 under CED).
pub fn prefix_bytes(entries: u64, dspark: bool, ranks: usize) -> Vec<u64> {
    let slots = if entries == 0 { 0 } else { 2 * entries + 2 };
    prefix_arena_bytes(slots, dspark, ranks)
}

pub fn prefix_arena_bytes(slots: u64, dspark: bool, ranks: usize) -> Vec<u64> {
    let mut bytes = vec![0; ranks];
    let target = (40 * (128 * 528 + 8) + 4 * 4096u64).div_ceil(256) * 256;
    bytes[0] = slots * target;
    if dspark { bytes[ranks - 1] += slots * 3 * 128 * 528; }
    bytes
}

/// Persistent dSpark windows and their prefill source staging, shared across
/// lane workspaces. All three stages reside on the decoder GPU.
pub fn dspark_cache_bytes(slots: u64, prefill_rows: u64) -> u64 {
    3 * (slots * 128 * 528 + prefill_rows * 1024 + 384)
}

/// Engram tables themselves remain host-mapped. Each lane retains one device
/// gather upload; its two gates follow layers 1/14, both on RTX0 under CED.
/// Gate GEMM scratch is part of the family's workspace reservation.
pub fn engram_staging_bytes(prefill_rows: u64, lanes: u64) -> u64 {
    lanes * prefill_rows * (24 * (256 + 8 + 512) + 1)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn checkpoint(tensors: Vec<crate::plan::checkpoint::CheckpointTensor>) -> Checkpoint {
        Checkpoint {
            snapshot: Default::default(), quantize_config: None, weight_map: Default::default(),
            missing_shards: vec![], shard_bytes: 0,
            config: serde_json::json!({"num_hidden_layers":40,"hidden_size":5120}), tensors,
        }
    }

    fn tensor(name: &str, dtype: cuteafd_core::DType, shape: &[usize], bytes: u64)
        -> crate::plan::checkpoint::CheckpointTensor {
        crate::plan::checkpoint::CheckpointTensor {
            shard: "model.safetensors".into(), meta: crate::SafetensorsTensorMetadata {
                name: name.into(), dtype, shape: shape.to_vec(), byte_offset: 0, byte_length: bytes,
            },
        }
    }

    #[test]
    fn routed_quantization_does_not_change_native_backbone_residency() {
        use cuteafd_core::DType;
        for name in ["layers.0.ffn.experts.w1.weight_scale_2",
            "layers.0.ffn.experts.w1.qweight", "layers.0.ffn.experts.w1.trellis",
            "model.layers.0.mlp.experts.w1.qweight", "layers.1.engram.embed.qweight"] {
            let checkpoint = checkpoint(vec![
                tensor("layers.0.attn.wq_a.weight", DType::F8E4M3, &[3200], 3200),
                tensor(name, DType::U8, &[64], 64),
            ]);
            let items = resident_weights(&checkpoint, 1, false).unwrap();
            assert_eq!(items[0].iter().map(|item| item.bytes).sum::<u64>(), 3300, "{name}");
        }
        for name in ["layers.0.attn.wq_a.qweight", "layers.0.attn.wq_a.weight_scale_2",
            "mtp.0.attn.wq_a.trellis"] {
            let checkpoint = checkpoint(vec![tensor(name, DType::U8, &[64], 64)]);
            assert!(resident_weights(&checkpoint, 1, true).is_none(), "{name}");
            assert_eq!(resident_weights(&checkpoint, 1, false).is_some(), name.starts_with("mtp."), "{name}");
        }
    }

    #[test]
    fn draft_packs_only_the_executed_main_projection_and_stored_grouped_extent() {
        use cuteafd_core::DType;
        let mut tensors = Vec::new();
        for stage in 0..3 {
            tensors.push(tensor(&format!("mtp.{stage}.main_proj.weight"), DType::F8E4M3,
                &[5120, 15360], 5120 * 15360));
            tensors.push(tensor(&format!("mtp.{stage}.attn.wo_a.weight"), DType::F8E4M3,
                &[8192, 4096], 8192 * 4096));
        }
        let draft_checkpoint = checkpoint(tensors);
        let expected = 3 * (5120 * 15360 + 8192 * 4096)
            + 5120 * 15360 / 32 + 3 * 8192 * 4096 / 32;
        for ranks in 1..=2 {
            let items = resident_weights(&draft_checkpoint, ranks, true).unwrap();
            assert_eq!(items[ranks - 1].iter().map(|item| item.bytes).sum::<u64>(), expected);
            assert!(items[..ranks - 1].iter().all(Vec::is_empty));
            assert!(resident_weights(&draft_checkpoint, ranks, false).unwrap().iter().all(Vec::is_empty));
        }
        let checkpoint = checkpoint(vec![tensor("layers.0.attn.wo_a.weight", DType::F8E4M3,
            &[8192, 4096], 8192 * 4096)]);
        let items = resident_weights(&checkpoint, 1, false).unwrap();
        assert_eq!(items[0][0].bytes, 8192 * 4096 + 8192 * 4096 / 32);
    }

    #[test]
    fn snapshot_and_expert_placement_keep_physical_owners() {
        let (experts, hidden, logical) = (384u64, 5120u64, 576u64);
        let stored = cuteafd_core::mxfp4_layouts::stored_width(hidden as usize, logical as usize).unwrap() as u64;
        let payload = 3 * hidden * stored / 2;
        let scales = 2 * stored * (1 + 2 * (hidden / 128))
            + hidden * (1 + 2 * logical.div_ceil(128));
        assert_eq!(native_expert_bytes(experts, hidden, logical), experts * (payload + scales));
        let one = prefix_bytes(20, true, 1);
        let two = prefix_bytes(20, true, 2);
        assert_eq!(one[0], two.iter().sum::<u64>());
        assert_eq!(two[1], 42 * 3 * 128 * 528);
        assert_eq!(prefix_bytes(0, true, 2), vec![0, 0]);
        assert_eq!(dspark_cache_bytes(16, 4096), 15_828_096);
    }

    #[test]
    fn tp3_and_tp6_charge_whole_native_and_nvfp4_planes() {
        for (tp, width) in [(3, 768), (6, 384)] {
            let native = native_expert_bytes(384, 5120, width);
            // FC2 carries one row base on every rank; gate/up bases partition.
            assert_eq!(native * tp, native_expert_bytes(384, 5120, 2304) + (tp - 1) * 384 * 5120);
            let nvfp4 = nvfp4_expert_bytes(384, 5120, width);
            assert_eq!(nvfp4, 384 * (3 * 5120 * width * 9 / 16 + 16));
            assert_eq!(nvfp4 * tp, nvfp4_expert_bytes(384, 5120, 2304) + (tp - 1) * 384 * 16);
        }
        assert_eq!(nvfp4_expert_bytes(384, 5120, 576), nvfp4_expert_bytes(384, 5120, 640));
    }

    #[test]
    fn vocabulary_modes_share_execution_residency_and_startup_semantics() {
        let source = 1_323_827_200;
        let packed = 682_598_400;
        let packed12 = 997_395_712;
        for (text, mode) in [("off", Fp8Head::Off), ("0", Fp8Head::Off),
            ("bf16", Fp8Head::Off), ("draft", Fp8Head::Draft),
            ("", Fp8Head::All), ("all", Fp8Head::All), ("other", Fp8Head::All),
            ("packed", Fp8Head::Packed)] {
            assert_eq!(Fp8Head::parse(text), mode);
            assert_eq!(mode.target(), matches!(mode, Fp8Head::All | Fp8Head::Packed));
            assert_eq!(mode.draft(), mode != Fp8Head::Off);
            assert_eq!(mode.load_bytes(source), source + match mode {
                Fp8Head::Off => 0, Fp8Head::Draft | Fp8Head::All => packed, Fp8Head::Packed => packed12,
            });
            assert_eq!(mode.resident(source).0, match mode {
                Fp8Head::Off => source, Fp8Head::Draft => source + packed, Fp8Head::All => packed,
                Fp8Head::Packed => packed12,
            });
        }
    }

    /// `packed` charges exactly what the loader allocates for each shard: 12 bits
    /// per weight, bases, row pointers, the exception table (one entry per 1024
    /// weights) and status, each region 256-byte aligned.
    #[test]
    fn packed_head_charges_its_exact_allocation_per_shard() {
        use cuteafd_core::packed_head::PackedHeadLayout;
        let source = 129280 * 5120 * 2;
        for ranks in [1u64, 2] {
            let rows = 129280 / ranks as usize;
            let layout = PackedHeadLayout::new(rows, 5120).unwrap();
            let (bytes, format) = Fp8Head::Packed.resident(source / ranks);
            assert_eq!((bytes, format), (layout.total_bytes() as u64, "bf16-packed12"));
            assert_eq!(Fp8Head::Packed.load_bytes(source / ranks), source / ranks + bytes);
            assert_eq!(layout.capacity as u64, rows as u64 * 5120 / 1024);
        }
        assert_eq!(Fp8Head::Packed.resident(source).0, 997_395_712);
        assert_eq!(Fp8Head::Packed.resident(source / 2).0, 498_698_240);
        // Between the E4M3 copy and BF16: 1.5 bytes per weight plus under half a percent.
        let per_weight = Fp8Head::Packed.resident(source).0 as f64 / (129280.0 * 5120.0);
        assert!(per_weight > 1.5 && per_weight < 1.5 * 1.005, "{per_weight}");
    }

    #[test]
    fn vocabulary_counts_single_fp8_copy_and_explicit_overrides() {
        let source = 129280 * 5120 * 2;
        let packed = 129280 * (5120 + 40 * 4);
        assert_eq!(Fp8Head::parse("").resident(source), (packed, "fp8-row128"));
        assert_eq!(Fp8Head::parse("all").resident(source), (packed, "fp8-row128"));
        assert_eq!(Fp8Head::parse("off").resident(source), (source, "bf16"));
        assert_eq!(Fp8Head::parse("draft").resident(source), (source + packed, "bf16+fp8-row128"));
        assert_eq!(Fp8Head::parse("all").resident(source / 2).0 * 2, packed);
    }

    #[test]
    fn resident_weights_match_ced_and_discard_transient_shared_scales() {
        use crate::plan::checkpoint::CheckpointTensor;
        use crate::SafetensorsTensorMetadata;
        use cuteafd_core::DType;
        let tensor = |name: &str, dtype, elements: usize, bytes: u64| CheckpointTensor {
            shard: "model.safetensors".into(), meta: SafetensorsTensorMetadata {
                name: name.into(), dtype, shape: vec![elements], byte_offset: 0, byte_length: bytes,
            },
        };
        let checkpoint = Checkpoint {
            snapshot: Default::default(), quantize_config: None, weight_map: Default::default(), missing_shards: vec![], shard_bytes: 0,
            config: serde_json::json!({"num_hidden_layers":40,"hidden_size":5120}),
            tensors: vec![tensor("embed.weight", DType::Bf16, 100, 200),
                tensor("head.weight", DType::Bf16, 100, 200),
                tensor("norm.weight", DType::Bf16, 10, 20),
                tensor("layers.0.attn.wq_a.weight", DType::F8E4M3, 3200, 3200),
                tensor("layers.20.attn.wq_a.weight", DType::F8E4M3, 6400, 6400),
                tensor("layers.0.ffn.shared_experts.w1.weight", DType::F8E4M3, 3200, 3200),
                tensor("layers.0.ffn.shared_experts.w1.scale", DType::F8E8M0, 100, 100),
                tensor("layers.1.engram.embed.weight", DType::Bf16, 100000, 200000)],
        };
        let single = resident_weights(&checkpoint, 1, false).unwrap();
        let dual = resident_weights(&checkpoint, 2, false).unwrap();
        let sum = |items: &[Item]| items.iter().map(|i| i.bytes).sum::<u64>();
        assert_eq!(sum(&single[0]), 200 + 103 + 20 + 3300 + 6600 + 3300 + 100);
        assert_eq!(sum(&dual[0]), 200 + 51 + 3300 + 1650);
        assert_eq!(sum(&dual[1]), 51 + 20 + 6600 + 1650);
    }
}

/// Exact resident scratch owned by the V4.1 9216-patch tower, excluding weights.
pub fn vision_scratch_bytes() -> u64 {
    let n = 9216u64;
    n*588*2 + n*2048*6 + n*6144 + n*5632*2 + n*2816*2
        + (n*3072).max(1024*5120)*4 + n*16*128*4 + n*16*128*2
        + 1024*9216*2 + 3*1024*5120*2 + 16*4 + 4*1024*1024
}
