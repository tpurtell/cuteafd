//! GLM coordinator target and drafter records through the shared push ranks.
use crate::shared::{experts::rtx::dense as coordinator_dense, memory::device::Device};
use anyhow::Result;

/// Each GPU's dense plan and destination, ranks `sparks..`.
#[allow(clippy::type_complexity)]
pub(crate) fn plan_ranks<'a>(snapshot: &std::path::Path, records: coordinator_dense::Records,
    devices: &[Device<'a>], sparks: usize)
    -> Result<(Vec<crate::shared::experts::rtx::push::DensePlan>, Vec<coordinator_dense::DenseArena<'a>>)> {
    coordinator_dense::plan_records(snapshot, records, devices, sparks)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::shared::speculation::{dflash::DflashConfig, weights::Source};
    use cuteafd_ffi::CuteafdDeviceBuffer;
    use cuteafd_loader::plan::testing::{t, write_safetensors};

    #[test]
    fn glm_dflash_source_plan_joins_target_and_draft_repositories_on_one_rank() -> Result<()> {
        let root = tempfile::tempdir()?;
        let target = root.path().join("models--glm53/snapshots/target");
        let draft = root.path().join("models--glm53-dflash2/snapshots/draft");
        std::fs::create_dir_all(&target)?;
        std::fs::create_dir_all(&draft)?;
        let cfg = DflashConfig { hidden: 128, intermediate: 128, layers: 2,
            heads: 1, kv_heads: 1, head_dim: 128, eps: 1e-6, theta: 10000., block: 8,
            group: 16, mask_token: 0, rank: 256, taps: vec![0], vocab: 16,
            window: 2048, row_window: false, sample_from_anchor: false };
        let shapes = cfg.tensor_shapes();
        write_safetensors(&draft.join("model.safetensors"), &shapes.iter()
            .map(|(name, shape)| t(name.clone(), "BF16", shape)).collect::<Vec<_>>());
        write_safetensors(&target.join("model.safetensors"), &[t("model.norm.weight", "BF16", &[128])]);
        let draft_source = Source::open(&draft)?;
        let target_source = Source::open(&target)?;
        let rects = std::iter::once(target_source.rect("model.norm.weight", &[128])?)
            .chain(shapes.iter().map(|(name, shape)| draft_source.rect(name, shape))
                .collect::<Result<Vec<_>>>()?);
        let operands = rects.map(|rect| {
            let bytes = rect.bytes();
            coordinator_dense::Operand { device: 1, bytes, lands: vec![(rect, 0)],
                buffer: CuteafdDeviceBuffer { bytes, device_id: 1, ..Default::default() } }
        }).collect();
        let layout = coordinator_dense::RankLayout::new(operands, Vec::new(), 1)?;
        let plan = layout.checked(root.path(), 8, 7)?;
        let draft_bytes = shapes.iter().map(|(_, shape)| shape.iter().product::<usize>() * 2).sum::<usize>();
        assert_eq!(plan.totals.rank_bytes[7], (256 + draft_bytes) as u64);
        assert_eq!(plan.plan.files.len(), 2);
        assert!(plan.plan.files.iter().any(|file| file.starts_with("models--glm53/")));
        assert!(plan.plan.files.iter().any(|file| file.starts_with("models--glm53-dflash2/")));
        assert!(plan.plan.entries.iter().all(|entry| entry.rank == 7));
        let names = layout.operands.iter().flat_map(|(_, o)| o.lands.iter().map(|(r, _)| &r.name))
            .collect::<std::collections::BTreeSet<_>>();
        assert_eq!(names.len(), shapes.len() + 1);
        for (name, _) in &shapes { assert!(names.contains(name), "missing draft source {name}"); }
        Ok(())
    }
}
