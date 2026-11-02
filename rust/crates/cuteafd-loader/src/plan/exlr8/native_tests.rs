use super::*;

#[test]
fn family_geometry_uses_nested_flash_config_and_excludes_mtp() -> Result<()> {
    let mut flash = crate::plan::testing::glm5_flash_config(45);
    flash["text_config"]["mlp_layer_types"] = serde_json::json!((0..45)
        .map(|layer| if layer < 3 { "dense" } else { "sparse" }).collect::<Vec<_>>());
    flash["text_config"]["first_k_dense_replace"] = serde_json::json!(3);
    assert_eq!(family_geometry(&flash)?, (45, 3, KernelFamily::GlmFlash));
    let mut invalid = flash.clone();
    invalid["text_config"]["n_routed_experts"] = serde_json::json!(256);
    assert!(family_geometry(&invalid).is_err());
    assert!(AdapterPending::Flash.hint().what.contains("adapter is not wired"));
    Ok(())
}

#[test]
fn flash_native_stages_charge_separate_projected_inputs_once() -> Result<()> {
    let cfg = crate::families::glm5_flash::GlmNextConfig::from_hf(&crate::plan::testing::glm5_flash_config(45))?;
    for ranks in [1, 2] {
        for rows in [1, 64, 128, 4096] {
            let stage = FlashStageLayout::new(&cfg, rows, ranks, false, 128, 4096)?;
            assert_eq!(stage.extents["decay"], rows * cfg.kda_heads / ranks * cfg.kda_head_dim * 2);
            assert_eq!(stage.extents["output_gate"], stage.extents["decay"]);
            assert_eq!(stage.extents["beta"], (rows * cfg.kda_heads / ranks * 2).max(256));
            assert_eq!(stage.extents["aux"], rows * 128 * 2);
            let old = crate::families::glm5_flash::core_admission::CoreStageLayout::new(&cfg, rows, ranks, false, 4096)?;
            assert_eq!(stage.bytes()?, old.bytes()? + ["decay", "output_gate", "beta", "aux"].iter()
                .map(|name| stage.extents[name]).sum::<usize>());
        }
    }
    let linear = LinearScratchLayout { rotated: 256, table: 256, block_scales: 256, partial: 256, records: None, acc: 256 };
    let stage = FlashStageLayout::new(&cfg, 64, 2, true, 128, 4096)?;
    let rank = FlashRankLayout { linear: [linear.clone(), linear.clone()], stages: [stage.clone(), stage.clone()],
        retained_rows: 64 + 3 * 4096, receive_suffix: 4 * 3 * 4096 * cfg.hidden * 2, shared_workspace: 12345 };
    assert_eq!(rank.bytes(cfg.hidden)?, 12345 + (2 * linear.bytes()? + 2 * stage.bytes()?
        + rank.retained_rows * cfg.hidden * 8 + rank.receive_suffix) as u64);
    let mut overflow = rank; overflow.retained_rows = usize::MAX;
    assert!(overflow.bytes(cfg.hidden).is_err());
    Ok(())
}

#[test]
#[ignore = "requires CUTEAFD_EXLR8_FLASH_SNAPSHOT published checkpoint headers"]
fn published_flash_native_facts_match_resident_bridge_and_owned_mla() -> Result<()> {
    let snapshot = std::path::PathBuf::from(std::env::var_os("CUTEAFD_EXLR8_FLASH_SNAPSHOT")
        .context("CUTEAFD_EXLR8_FLASH_SNAPSHOT required")?);
    let config: Value = serde_json::from_slice(&std::fs::read(snapshot.join("config.json"))?)?;
    let cfg = crate::families::glm5_flash::GlmNextConfig::from_hf(&config)?;
    let metadata = MetadataInput::open(&snapshot, "k4")?;
    let records = DenseRecord::inventory(&snapshot)?;
    let routed = cfg.dense.iter().enumerate().filter_map(|(layer, &dense)| (!dense).then_some(layer)).collect::<Vec<_>>();
    for ranks in [1, 2] {
        let shared = crate::plan::exlr8::shared_expert_resident(&metadata, &routed, ranks)?;
        let facts = DenseFacts::new(records.clone(), DenseSelection::from_hf(&config)?, cfg.layers, shared)?;
        let mut owners = BTreeMap::new();
        for owned in [false, true] {
            if owned && ranks == 1 { continue; }
            if owned {
                for (index, layer) in cfg.attention.iter().enumerate().filter_map(|(layer, kind)|
                    (*kind == crate::families::glm5_flash::GlmNextAttention::Mla).then_some(layer)).enumerate() {
                    owners.insert(layer, index % ranks);
                }
            }
            let plans = facts.plans(&owners)?;
            let resident = crate::families::glm5_flash::resident::native_resident(&facts.selection, cfg.layers, &plans, &facts.shared_bytes)?;
            for (rank, plan) in plans.iter().enumerate() {
                assert_eq!(resident[rank].weights + resident[rank].embedding,
                    plan.resident_bytes as u64 + facts.shared_bytes[rank]);
                assert!(plan.tensors.iter().all(|slice| !slice.tensor.contains(".layers.45.")));
                for (&layer, &owner) in &owners {
                    let prefix = format!("model.language_model.layers.{layer}.self_attn.");
                    assert_eq!(plan.tensors.iter().any(|slice| slice.tensor.starts_with(&prefix)), rank == owner);
                }
            }
        }
    }
    Ok(())
}
