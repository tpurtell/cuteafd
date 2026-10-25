//! Dump the planner's per-rank compact EXL3 ownership for a checkpoint layer.
//!
//! Env: REPLAY_SNAPSHOT (snapshot dir), REPLAY_OUTPUT (json path), REPLAY_WORLD
//! (Spark world), REPLAY_LAYER (backbone layer index), REPLAY_RANKS (comma list).
//! The output is the harness fixture consumed by
//! `benchmarks/benchmark_mixed_trellis.py --compact-fixture`.
use anyhow::{Context, Result};
use cuteafd_loader::{read_expert_catalog, V41Exl3Layer};
use serde_json::json;
use std::path::Path;

#[test]
fn replay_layer_ownership() -> Result<()> {
    let Ok(snapshot) = std::env::var("REPLAY_SNAPSHOT") else {
        // Without a checkpoint the generator is a no-op, so the test only runs
        // where a fixture is being captured.
        return Ok(());
    };
    let output = std::env::var("REPLAY_OUTPUT")?;
    let world: usize = std::env::var("REPLAY_WORLD")?.parse()?;
    let layer: usize = std::env::var("REPLAY_LAYER")?.parse()?;
    let ranks: Vec<usize> = std::env::var("REPLAY_RANKS")?
        .split(',').map(|v| v.trim().parse()).collect::<std::result::Result<_, _>>()?;
    let catalog = read_expert_catalog(Path::new(&snapshot))?;
    let manifest = catalog.exl3().context("EXL3 checkpoint")?;
    let shape = manifest.experts;
    if std::env::var("REPLAY_DEBUG").is_ok() {
        println!("shape={shape:?} tiers={:?}", manifest.decoder_tiers());
        println!("projections={} first={:?}", manifest.projections.len(),
            manifest.projections.keys().take(4).collect::<Vec<_>>());
    }
    let mut out_ranks = Vec::new();
    for &rank in &ranks {
        let mut arena_bytes = 0usize;
        for index in shape.first_layer..shape.layers {
            arena_bytes += manifest
                .balanced_residency(V41Exl3Layer::Backbone(index), world, rank)?
                .device_arena_layout()?
                .1;
        }
        let plan = manifest.balanced_residency(V41Exl3Layer::Backbone(layer), world, rank)?;
        let map = plan.unit_map.as_ref().context("compact ownership map")?;
        let descriptor = &plan
            .buffers
            .iter()
            .find(|b| b.name == "descriptor_map")
            .context("descriptor map")?
            .initial_words;
        let combined = &plan
            .buffers
            .iter()
            .find(|b| b.name == "global_to_combined")
            .context("combined map")?
            .initial_words;
        let stride = plan.experts * plan.tiers.len();
        let experts: Vec<_> = map
            .experts
            .iter()
            .enumerate()
            .map(|(expert, units)| {
                let projections: Vec<_> = ["gate", "up", "down"]
                    .into_iter()
                    .enumerate()
                    .map(|(projection, name)| {
                        let packed = descriptor[projection * stride + expert] as usize;
                        let tier = packed >> 9;
                        json!({"projection":name,"tier":tier,"bits":plan.tiers[tier],
                            "tier_local":packed & 511,
                            "payload_word_offset":descriptor[(4 + projection) * stride + expert]})
                    })
                    .collect();
                json!({"global_expert":expert,"combined_local":combined[expert],
                    "h128_units":units,"width":units.len()*128,"projections":projections})
            })
            .collect();
        out_ranks.push(json!({"rank":rank,"aot_layout":plan.aot_layout(),
            "compiled_max_intermediate":plan.intermediate,"expert_count":plan.experts,
            "all_layers_device_arena_bytes":arena_bytes,
            "projection_counts":plan.projection_counts,
            "unit_map_fingerprint":map.fingerprint(),"experts":experts}));
    }
    let evidence = json!({
        "source":"read_expert_catalog + V41Exl3Manifest::balanced_residency (real planner)",
        "snapshot":snapshot,"layer":layer,"spark_world":world,
        "hidden":shape.hidden,"intermediate":shape.intermediate,
        "expert_geometry":shape.experts,"top_k":shape.topk,
        "tiers":manifest.decoder_tiers(),"ranks":out_ranks});
    std::fs::write(&output, serde_json::to_vec_pretty(&evidence)?)?;
    println!("ownership replay written to {output}");
    Ok(())
}
