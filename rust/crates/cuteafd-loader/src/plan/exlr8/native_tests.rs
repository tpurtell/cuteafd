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
            assert_eq!(stage.extents["index_query"], rows * 32 * 128 * 2);
            assert_eq!(stage.extents["index_packed"], rows * 288 * 2);
            assert_eq!(stage.bytes()?, old.bytes()? + ["decay", "output_gate", "beta", "aux", "index_query", "index_packed"].iter()
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

fn index_stage_fixture(prefix: &str, capacity: usize, compact: bool) -> Value {
    use serde_json::json;
    let replay = if capacity == 128 { 128 } else { 64 };
    let mut pointers = vec![
        ("iq", "bfloat16", "[rows,32,128]".to_string(), "in"),
        ("kw", "bfloat16", "[rows,288]".into(), "in"),
    ];
    let inputs = if compact {
        vec![("pool_slots", "int64"), ("positions", "int64"), ("kda_slots", "int32"), ("seq_first", "int32")]
    } else { vec![("slots", "int64"), ("pool_slots", "int64")] };
    pointers.extend(inputs.into_iter().map(|(name, dtype)| (name, dtype, "[rows]".into(), "in")));
    pointers.extend([
        ("k_norm_w", "bfloat16", "[128]".into(), "in"),
        ("k_norm_b", "bfloat16", "[128]".into(), "in"),
        ("ape", "bfloat16", "[4,128]".into(), "in"),
    ]);
    if compact {
        pointers.extend([("tails", "uint8", "[slots,1552]".into(), "inout"),
            ("replay", "bfloat16", format!("[{replay},256]"), "out")]);
    } else { pointers.push(("token_keys", "bfloat16", "[record_slots,256]".into(), "inout")); }
    pointers.extend([
        ("index_cache", "uint8", "[pool_pages,8448]".into(), "inout"),
        ("q_fp8", "float8_e4m3fn", "[rows,32,128]".into(), "out"),
        ("head_weights", "float32", "[rows,32]".into(), "out"),
    ]);
    if compact { pointers.push(("scratch", "uint8", "[index_keys_scratch]".into(), "scratch")); }
    let mut geometry = json!({"hidden":4096,"index_heads":32,"index_head_dim":128,"index_kpool":4,
        "page_rows":64,"max_rows":capacity,"index_norm_eps":1e-6,"projected_inputs":true});
    if compact { geometry["tail_bytes"] = json!(1552); geometry["replay_rows"] = json!(replay); }
    else { geometry["token_key_row_bytes"] = json!(512); }
    let kind = if compact { "c" } else { "keys" };
    json!({"name":format!("{prefix}_index_projected_{kind}_m{capacity}"),"op":"index_producer",
        "capacity_rows":capacity,"geometry":geometry,
        "pointers":pointers.into_iter().map(|(name,dtype,shape,role)| json!({"name":name,"dtype":dtype,"shape":shape,"role":role})).collect::<Vec<_>>(),
        "scalars":if compact { json!([{"name":"rows","type":"int32"},{"name":"spec","type":"int32"}]) }
            else { json!([{"name":"rows","type":"int32"}]) },
        "scratch_bytes_at_capacity":if compact { json!({"scratch":capacity*512}) } else { json!({}) }})
}

#[test]
fn flash_index_admission_selects_effective_cache_without_fallback() -> Result<()> {
    use crate::serving_capacity::GlmfIndexCache::{Compact, Keys};
    for prefix in ["glmf_stage", "glmf2_stage"] {
        for capacity in [64, 128, 4096] {
            for (cache, other, compact) in [(Compact, Keys, true), (Keys, Compact, false)] {
                let entry = index_stage_fixture(prefix, capacity, compact);
                let manifest = serde_json::json!({"programs":[entry.clone()]});
                let (name, scratch) = flash_index_stage(&manifest, prefix, capacity, cache)?;
                assert_eq!(name, entry["name"].as_str().unwrap());
                assert_eq!(scratch, if compact { capacity * 512 } else { 0 });
                assert!(flash_index_stage(&manifest, prefix, capacity, other).is_err());
                let mut invalid = manifest.clone();
                invalid["programs"][0]["geometry"]["index_heads"] = serde_json::json!(16);
                assert!(flash_index_stage(&invalid, prefix, capacity, cache).is_err());
                let duplicate = serde_json::json!({"programs":[entry.clone(),entry]});
                assert!(flash_index_stage(&duplicate, prefix, capacity, cache).is_err());
            }
        }
    }
    Ok(())
}

#[test]
#[ignore = "requires CUTEAFD_FLASH_STAGE_ABIS with actual TP1/TP2 exporter listings"]
fn exported_flash_index_admission_selects_effective_cache() -> Result<()> {
    use crate::serving_capacity::GlmfIndexCache::{Compact, Keys};
    let paths = std::env::var_os("CUTEAFD_FLASH_STAGE_ABIS").context("actual stage ABI paths required")?;
    let mut checked = BTreeSet::new();
    for path in std::env::split_paths(&paths) {
        let manifest: Value = serde_json::from_slice(&std::fs::read(path)?)?;
        let prefix = if manifest["programs"].as_array().context("stage programs")?.iter()
            .any(|entry| entry["name"].as_str().is_some_and(|name| name.starts_with("glmf2_stage_"))) {
            "glmf2_stage"
        } else { "glmf_stage" };
        for capacity in [64, 128, 4096] {
            for cache in [Compact, Keys] {
                let (name, scratch) = flash_index_stage(&manifest, prefix, capacity, cache)?;
                assert_eq!(scratch, if cache == Compact { capacity * 512 } else { 0 });
                checked.insert(name);
            }
        }
    }
    assert_eq!(checked.len(), 12);
    Ok(())
}

#[test]
#[ignore = "requires CUTEAFD_EXLR8_FLASH_SNAPSHOT published checkpoint headers"]
fn published_flash_native_facts_match_resident_bridge_and_owned_mla() -> Result<()> {
    let snapshot = std::path::PathBuf::from(std::env::var_os("CUTEAFD_EXLR8_FLASH_SNAPSHOT")
        .context("CUTEAFD_EXLR8_FLASH_SNAPSHOT required")?);
    let config: Value = serde_json::from_slice(&std::fs::read(snapshot.join("config.json"))?)?;
    let cfg = crate::families::glm5_flash::GlmNextConfig::from_hf(&config)?;
    let catalog = crate::families::deepseek_v41::v41_catalog::read_expert_catalog(&snapshot)?;
    assert_eq!(catalog.routed_experts().layers, cfg.layers);
    assert_eq!(catalog.routed_experts().first_layer, 3);
    assert_eq!(catalog.routed_experts().experts, cfg.experts);
    assert_eq!(catalog.routed_experts().topk, cfg.topk);
    assert_eq!(catalog.routed_experts().hidden, cfg.hidden);
    assert_eq!(catalog.routed_experts().intermediate, cfg.moe_intermediate);
    assert_eq!((catalog.routed_experts().draft_stages, catalog.routed_experts().draft_experts), (0, 0));
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
