//! Kimi K3 (`kimi_k3` wrapping `text_config.model_type=kimi_linear`) checkpoint readers.
pub mod allocation;
pub mod config;
pub mod context;
pub mod core;
pub mod kda;
pub mod mla;
pub mod reference;
pub mod spark;
pub mod w8;
pub mod weight_layout;

pub const COMPILED_CONTEXT: usize = 1_048_576;

/// Shared MLA default for CPU planning, serving and golden startup. KDA keeps
/// its independent head split on a pair; explicit MLA placements always win.
pub fn resolve_attention(requested: Option<crate::placement::AttentionPlacement>, ranks: usize)
    -> crate::placement::AttentionPlacement {
    requested.unwrap_or(if ranks == 2 { crate::placement::AttentionPlacement::Context }
        else { crate::placement::AttentionPlacement::Layers })
}
/// Default context for a BF16 (K2-class) core: one full-length conversation
/// would leave little pool beside the BF16 core weights.
pub const DEFAULT_CONTEXT: usize = 524_288;

/// The default API/allocation context from the checkpoint's own core: a
/// header-native K6 core frees the memory for the full compiled 1M extent with
/// the full pool and onboard experts (TJ, 2026-10-22); BF16 cores keep 512K.
pub fn default_context(core: &core::CoreInventory) -> usize {
    if core.projections.is_empty() { DEFAULT_CONTEXT } else { COMPILED_CONTEXT }
}
pub const SPARK_DEFAULT_BUDGET_BYTES: u64 = 108 << 30;

/// TP6 retained EXL3 arenas/private capacities plus 4096-row K32 input/routes
/// (#762 census). GlmFlashK64 does not retain the m16 specialization.
pub const SPARK_SLOT_BYTES: u64 = 844_696_852;

/// Kimi's typed Spark owners on its EXL3 K64 ladder.
pub fn spark_executions(waves: u64, unit_rows: u64) -> anyhow::Result<cuteafd_core::SparkExecutions> {
    let exports = crate::plan::spark_executions::SparkBackend::Exl3(cuteafd_core::Exl3RowPolicy::GlmFlashK64).exports();
    crate::plan::spark_executions::admit_family("kimi_k3", &exports, None, unit_rows, waves)
}

/// One Kimi owner's bytes (the shared EXL3 owner census at latent width).
pub fn spark_owner_bytes(extent: cuteafd_core::OwnerExtent) -> anyhow::Result<cuteafd_core::OwnerBytes> {
    crate::plan::layout::spark_slots::exl3_owner_bytes("kimi_k3", cuteafd_core::ExpertGeometry::KIMI_K3, extent)
}

/// The Spark execution reserve used before placement and in its ledger: the
/// runtime residual plus the typed owners (one prefill scratch at the admitted
/// unit's export, the decode owner, per-unit route staging times the waves).
pub fn spark_reserve_bytes(waves: u64, unit_rows: u64) -> anyhow::Result<u64> {
    let owners = spark_executions(waves, unit_rows)?.workspace_bytes(spark_owner_bytes)?
        .and_then(|bytes| bytes.total()).ok_or_else(|| anyhow::anyhow!("Kimi Spark owner reserve overflow"))?;
    cuteafd_core::memory_layout::spark_runtime_reserve_bytes(0)
        .and_then(|runtime| runtime.checked_add(owners))
        .ok_or_else(|| anyhow::anyhow!("Kimi Spark runtime reserve overflow"))
}

/// Target taps for one prefill call (`rows` lane rows) while the common MLA
/// drafter consumes 2048-row slices.
pub fn draft_bridge_demands(layout: &crate::formats::dflash::MlaDraftRuntimeLayout, rows: usize)
    -> anyhow::Result<Vec<crate::placement::Demand>> {
    draft_bridge_demands_for_ranks(layout, rows, 2)
}

pub fn draft_bridge_demands_for_ranks(layout: &crate::formats::dflash::MlaDraftRuntimeLayout,
    rows: usize, ranks: usize) -> anyhow::Result<Vec<crate::placement::Demand>> {
    draft_bridge_inventory(layout.target_hidden() as u64, layout.target_vocab() as u64,
        layout.target_taps().len() as u64, layout.capacity().block_rows as u64, rows, ranks)
}

fn draft_bridge_inventory(hidden: u64, vocab: u64, taps: u64, block_rows: u64, rows: usize,
    ranks: usize) -> anyhow::Result<Vec<crate::placement::Demand>> {
    use anyhow::{ensure, Context};
    ensure!(matches!(ranks, 1 | 2) && rows > 0, "Kimi draft bridge physical rank/row geometry");
    // Both routes stage the same inputs. Only the remote route owns exchange controls;
    // the common drafter already owns its hidden/logit/head workspace on the head device.
    let mut demands = crate::placement::draft::remote_dflash_bridge_demands(
        hidden, vocab, taps, block_rows, 1)?.demands;
    if ranks == 1 { demands.retain(|d| d.group != "DFlash bridge exchange"); }
    let tap_bytes = (rows as u64).checked_mul(taps).and_then(|n| n.checked_mul(hidden))
        .and_then(|n| n.checked_mul(2)).context("Kimi bridge tap extent overflow")?;
    demands.iter_mut().find(|d| d.group == "DFlash bridge taps")
        .context("Kimi bridge tap inventory missing")?.bytes = tap_bytes.max(16);
    Ok(demands)
}

#[cfg(test)]
mod bridge_tests {
    use super::*;

    #[test]
    fn local_bridge_retains_only_actual_staging_owners() -> anyhow::Result<()> {
        for (hidden, vocab, taps, block, rows) in [(7168, 163840, 5, 16, 8256), (16, 32, 2, 8, 64)] {
            let remote = draft_bridge_inventory(hidden, vocab, taps, block, rows, 2)?;
            let local = draft_bridge_inventory(hidden, vocab, taps, block, rows, 1)?;
            assert_eq!(local.len(), 3);
            assert!(local.iter().all(|d| d.gpu == 0));
            assert_eq!(local, remote.iter().filter(|d| d.group != "DFlash bridge exchange")
                .cloned().collect::<Vec<_>>());
            assert_eq!(local.iter().find(|d| d.group == "DFlash bridge taps").unwrap().bytes,
                rows as u64 * taps * hidden * 2);
            assert_eq!(remote.iter().filter(|d| d.group == "DFlash bridge exchange").count(), 2);
        }
        for ranks in [0, 3] { assert!(draft_bridge_inventory(16, 32, 2, 8, 64, ranks).is_err()); }
        assert!(draft_bridge_inventory(16, 32, 2, 8, 0, 1).is_err());
        assert!(draft_bridge_inventory(16, 32, 2, 8, usize::MAX, 1).is_err());
        Ok(())
    }
}

/// The shared probe retains one GPU0 route arena; its pinned mirror is host memory.
pub fn draft_probe_demand(layers: usize, topk: usize, rows: usize) -> anyhow::Result<crate::placement::Demand> {
    use anyhow::Context;
    use crate::placement::Demand;
    use cuteafd_core::memory_layout::{Basis, Category};
    let bytes = layers.checked_mul(topk).and_then(|n| n.checked_mul(rows))
        .and_then(|n| n.checked_mul(4)).context("Kimi draft probe extent overflow")?;
    Ok(Demand::new(0, Category::Workspace, "draft policy route arena", bytes.max(256) as u64, Basis::Formula))
}

/// Validate sealed role artifacts at runtime, never during checkpoint planning.
pub fn validate_expert_inventory(path: &std::path::Path, root: &std::path::Path) -> anyhow::Result<()> {
    use crate::plan::checkpoint::read_json;
    use anyhow::{ensure, Context};
    let programs = read_json(path)?;
    ensure!(programs["schema"] == 1 && programs["programs"].is_array(), "invalid PROGRAMS.json inventory");
    let package = read_json(&root.join("manifest.json"))?;
    ensure!(package["schema"] == "cuteafd.exl3-package.v1" && package["geometry"] == "kimik3",
        "Kimi K3 EXL3 package schema/geometry mismatch");
    let revision = package["sparkinfer_revision"].as_str().filter(|s| s.len() == 40)
        .context("Kimi K3 package lacks source revision")?;
    let lock: serde_json::Value = serde_json::from_str(include_str!(concat!(env!("CARGO_MANIFEST_DIR"),
        "/../../../third_party/sparkinfer.lock.json")))?;
    ensure!(lock["revision"] == revision, "Kimi K3 package/source revision mismatch");
    let (layouts, output) = match package["role"].as_str() {
        Some("spark") => {
            // A scoped WIP package need only carry its declared worlds, but each
            // world must have all recipe ranks and capacities before native load.
            let variants = package["variants"].as_array().context("Kimi K3 package lacks variants")?;
            let worlds = variants.iter().filter_map(|entry| entry["directory"].as_str())
                .filter_map(|directory| directory.strip_prefix("tp")?.split('-').next()?.parse::<usize>().ok())
                .collect::<std::collections::BTreeSet<_>>();
            ensure!(!worlds.is_empty(), "Kimi K3 Spark package lacks rank layouts");
            let mut layouts = Vec::new();
            for world in worlds {
                let recipe = crate::plan::experts::spark_recipe("kimik3:exl3-k23", 3072, world)?;
                for (rank, layout) in recipe.required_layouts.into_iter().enumerate() {
                    layouts.push((layout, recipe.widths[rank]));
                }
            }
            (layouts, "bf16")
        },
        Some("coordinator") => {
            ensure!(programs["sparkinfer_revision"] == revision, "Kimi K3 program/package revision mismatch");
            let inventory = programs["programs"].as_array().unwrap();
            for name in ["kimik3_router_scores", "kimik3_expert_input_quant", "kimik3_latent_down",
                "kimik3_latent_up", "kimik3_rmsnorm_h3584"] {
                let entries = inventory.iter().filter(|p| p["name"] == name).collect::<Vec<_>>();
                ensure!(entries.len() == 1 && entries[0]["object_sha256"].as_str().is_some_and(|s| s.len() == 64),
                    "Kimi K3 program inventory requires one sealed {name}");
                let geometry = &entries[0]["geometry"];
                match name {
                    "kimik3_router_scores" => ensure!(geometry["hidden"] == 7168 && geometry["experts"] == 896
                        && geometry["top_k"] == 16, "Kimi router geometry mismatch"),
                    "kimik3_latent_down" => ensure!(geometry["width"] == 7168 && geometry["outputs"] == 3584,
                        "Kimi latent down geometry mismatch"),
                    "kimik3_latent_up" => ensure!(geometry["width"] == 3584 && geometry["outputs"] == 7168,
                        "Kimi latent up geometry mismatch"),
                    _ => ensure!(geometry["hidden"] == 3584, "Kimi latent input geometry mismatch"),
                }
            }
            (vec![("rtx-tp1".into(), 3072), ("rtx-tp2".into(), 1536)], "fp32")
        }
        _ => anyhow::bail!("Kimi K3 EXL3 package must declare coordinator or spark role"),
    };
    let variants = package["variants"].as_array().context("Kimi K3 package lacks variants")?;
    for (layout, width) in layouts {
        for capacity in cuteafd_core::Exl3RowPolicy::for_geometry(cuteafd_core::ExpertGeometry::KIMI_K3).capacities(4096)? {
            let directory = format!("{layout}/m{capacity}");
            let entries = variants.iter().filter(|v| v["directory"] == directory).collect::<Vec<_>>();
            ensure!(entries.len() == 1, "Kimi K3 package requires one {directory}");
            let meta = read_json(&root.join(&directory).join("v41_exl3.json"))?;
            for entry in [entries[0], &meta] {
                ensure!(entry["intermediate"] == width && entry["experts"] == 896 && entry["top_k"] == 16
                    && entry["capacity"] == capacity && entry["bits"] == serde_json::json!([2, 3])
                    && entry["output_dtype"] == output, "Kimi K3 package geometry mismatch: {directory}");
            }
            ensure!(meta["sparkinfer_revision"] == revision && meta["hidden"] == 3584
                && meta["activation"] == "situ" && meta["descriptor_bits"] == 10
                && meta["descriptor_rows"].as_u64().is_none_or(|n| n == if layout.contains("-compact-") { 7 } else { 3 })
                && meta["native_info_version"].as_u64().is_none_or(|n| n == if layout.contains("-compact-") { 3 } else { 2 })
                && meta["decode_schedule"].is_null(), "Kimi K3 EXL3 native contract mismatch: {directory}");
            for artifact in ["libcuteafd_exl3.so", "routes/libv41_exl3_routes.so", "trellis_lut.bin"] {
                ensure!(root.join(&directory).join(artifact).is_file(),
                    "Kimi K3 package missing {directory}/{artifact}");
            }
        }
    }
    Ok(())
}
