//! V4.1's default CED placement and native resident representations. These
//! formulas mirror the startup owners, including packed scale copies.
use crate::plan::Checkpoint;
use cuteafd_core::memory_layout::{Basis, Category, Item};
use std::collections::BTreeMap;

pub const DEFAULT_POOL_TOKENS: u64 = 2 * 1_048_576;

/// Native MXFP4 packer storage, including the padding of each intermediate
/// slice. This applies to full RTX experts and each independently packed TP
/// Spark/RTX slice; dividing a full layer's bytes misses TP4's 576->640 pad.
pub fn native_expert_bytes(experts: u64, hidden: u64, intermediate: u64) -> u64 {
    experts * hidden * intermediate.div_ceil(128) * 128 * 51 / 32
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
    let config = checkpoint.config.get("text_config").unwrap_or(&checkpoint.config);
    if ![1, 2].contains(&ranks) || config["num_hidden_layers"].as_u64() != Some(40)
        || config["hidden_size"].as_u64() != Some(5120)
        || checkpoint.tensors.iter().any(|t| t.meta.name.ends_with("weight_scale_2")
            || t.meta.name.contains(".qweight") || t.meta.name.ends_with(".trellis")) {
        return None;
    }
    let decoder = ranks - 1;
    let head_mode = std::env::var("CUTEAFD_V41_FP8_HEAD").unwrap_or_default();
    let mut totals: Vec<BTreeMap<(Category, String, String), u64>> = vec![BTreeMap::new(); ranks];
    let mut add = |rank: usize, category, group: &str, format: &str, bytes: u64| {
        *totals[rank].entry((category, group.into(), format.into())).or_default() += bytes;
    };
    let mut draft_experts = false;
    for tensor in &checkpoint.tensors {
        let name = tensor.meta.name.as_str();
        let mut bytes = tensor.meta.byte_length;
        let format = format!("{:?}", tensor.meta.dtype).to_ascii_lowercase();
        if name.starts_with("mtp.") {
            if !dspark { continue; }
            if name.contains(".ffn.experts.") { draft_experts = true; continue; }
            // NativeRtxTensors retains all auxiliary tensors, then each FP8
            // matrix adds its MMA scale layout without releasing source scales.
            if name.ends_with(".weight") && format.contains("e4m3") {
                bytes += tensor.meta.shape.iter().product::<usize>() as u64 / 32;
            }
            add(decoder, Category::Drafter, "dspark-auxiliary", &format, bytes);
            continue;
        }
        if name == "head.weight" {
            let (resident, head_format) = vocabulary_bytes(bytes / ranks as u64, &head_mode);
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
            bytes += tensor.meta.shape.iter().product::<usize>() as u64 / 32;
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

// Mirror the vocabulary packer: one E4M3 value and FP32 scales per 128-K
// block. Draft-only retains the BF16 target; all mode releases it after packing.
fn vocabulary_bytes(source: u64, mode: &str) -> (u64, &'static str) {
    let packed = source / 2 + source / 64;
    match mode {
        "off" | "0" | "bf16" => (source, "bf16"),
        "draft" => (source + packed, "bf16+fp8-row128"),
        _ => (packed, "fp8-row128"),
    }
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
    #[test]
    fn snapshot_and_expert_placement_keep_physical_owners() {
        assert_eq!(native_expert_bytes(384, 5120, 576), 2_005_401_600);
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
            assert_eq!(native * tp, native_expert_bytes(384, 5120, 2304));
            let nvfp4 = nvfp4_expert_bytes(384, 5120, width);
            assert_eq!(nvfp4, 384 * (3 * 5120 * width * 9 / 16 + 16));
            assert_eq!(nvfp4 * tp, nvfp4_expert_bytes(384, 5120, 2304) + (tp - 1) * 384 * 16);
        }
        assert_eq!(nvfp4_expert_bytes(384, 5120, 576), nvfp4_expert_bytes(384, 5120, 640));
    }

    #[test]
    fn vocabulary_counts_single_fp8_copy_and_explicit_overrides() {
        let source = 129280 * 5120 * 2;
        let packed = 129280 * (5120 + 40 * 4);
        assert_eq!(vocabulary_bytes(source, ""), (packed, "fp8-row128"));
        assert_eq!(vocabulary_bytes(source, "all"), (packed, "fp8-row128"));
        assert_eq!(vocabulary_bytes(source, "off"), (source, "bf16"));
        assert_eq!(vocabulary_bytes(source, "draft"), (source + packed, "bf16+fp8-row128"));
        assert_eq!(vocabulary_bytes(source / 2, "all").0 * 2, packed);
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
