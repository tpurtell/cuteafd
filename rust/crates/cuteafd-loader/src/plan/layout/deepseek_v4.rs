//! V4 resident tensor bytes and ownership, matching WeightLoader's head split.
use crate::plan::{Checkpoint, Component};

pub(super) fn resident_weights(checkpoint: &Checkpoint, ranks: usize) -> Option<Vec<Vec<(String, String, u64)>>> {
    let family = crate::plan::family::detect(checkpoint)?;
    let model = family.open(checkpoint).ok()?;
    let mut groups = vec![std::collections::BTreeMap::<(String, String), u64>::new(); ranks];
    for tensor in &checkpoint.tensors {
        let name = &tensor.meta.name;
        let role = family.classify(model.spec(), name)?;
        if matches!(role.component, Component::RoutedExpert | Component::SpeculatorExpert | Component::MappedTable | Component::Vision) {
            continue;
        }
        // Only the target and dSpark operands that the runtime uploads.
        if role.component == Component::Other { continue; }
        let mut bytes = tensor.meta.byte_length;
        let block_scale = name.ends_with(".scale") && !name.contains("main_proj");
        let format = if block_scale { bytes *= 512; "fp8-mma-scale".into() }
            else if name.ends_with("tid2eid") { bytes /= 2; "i32".into() }
            else { format!("{:?}", tensor.meta.dtype).to_ascii_lowercase() };
        let shared = name.starts_with("layers.");
        let sharded = shared && ["attn.wq_b.", "attn.attn_sink", "attn.wo_a.", "attn.wo_b.",
            "ffn.shared_experts.w1.", "ffn.shared_experts.w2.", "ffn.shared_experts.w3."]
            .iter().any(|part| name.contains(part));
        let group = role.component.label().to_string();
        for (rank, target) in groups.iter_mut().enumerate() {
            if rank > 0 && !shared { continue; }
            *target.entry((group.clone(), format.to_string())).or_default() += if sharded { bytes / ranks as u64 } else { bytes };
        }
    }
    Some(groups.into_iter().map(|group| group.into_iter().map(|((group,format),bytes)| (group,format,bytes)).collect()).collect())
}
