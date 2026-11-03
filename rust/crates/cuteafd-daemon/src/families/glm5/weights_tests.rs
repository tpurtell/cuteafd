//! GLM 5.3 coordinator operands on a patterned synthetic checkpoint, through
//! the fake native DSO with a two-GPU head split. Every operand's bytes are
//! pinned to a golden digest taken from the loader before the landing
//! refactor, so the refactor (and push landings built on it) must reproduce
//! the mount load bit for bit. `w_ik` is excluded: the fake DSO does not
//! implement the FP8 dequantization that fills it.
use super::*;
use cuteafd_ffi::native_library_lifetime_fixture::Fixture;
use cuteafd_loader::plan::testing as fixture;
use sha2::{Digest, Sha256};
use std::os::unix::fs::FileExt;

#[test]
fn tp2_peer_router_names_match_installed_checkpoint_headers() -> Result<()> {
    let home = std::env::var_os("HF_HOME").map(std::path::PathBuf::from)
        .unwrap_or_else(|| std::path::PathBuf::from(std::env::var_os("HOME").unwrap()).join(".cache/huggingface"));
    let models = [
        ("zai-org--GLM-5.3", "aca966e4e02791568aa6a4ced368624b3d897f42", "model.safetensors.index.json", true),
        ("mistralai--Mistral-Large-4-1T-A52B", "f12dabdc0c1e99e7f7924461a6f308859b05af4d", "consolidated.safetensors.index.json", false),
    ];
    for (model, revision, index_file, bias) in models {
        let snapshot = home.join("hub").join(format!("models--{model}")).join("snapshots").join(revision);
        if !snapshot.join("config.json").is_file() {
            eprintln!("checkpoint headers unavailable: {}", snapshot.display());
            continue;
        }
        let cfg = GlmDsaConfig::read(&snapshot)?;
        let index: serde_json::Value = serde_json::from_slice(&std::fs::read(snapshot.join(index_file))?)?;
        let mut headers = HashMap::new();
        for layer in cfg.first_moe_layer..cfg.layers {
            let names: Vec<_> = peer_router_names(&cfg, layer).collect();
            assert_eq!(names.len(), if bias { 2 } else { 1 });
            assert_eq!(names[0].0, "gate");
            for (operand, name) in names {
                let shard = index["weight_map"][&name].as_str().with_context(|| format!("missing peer router {name}"))?;
                if !headers.contains_key(shard) {
                    headers.insert(shard.to_owned(), cuteafd_loader::read_safetensors_metadata(&snapshot.join(shard))?);
                }
                let tensor = headers[shard].iter().find(|t| t.name == name).with_context(|| format!("missing header {name}"))?;
                assert_eq!(tensor.shape, if operand == "gate" { vec![cfg.experts, cfg.hidden] } else { vec![cfg.experts] });
                assert_eq!(tensor.dtype, if operand == "gate" { DType::Bf16 } else { DType::F32 });
            }
        }
    }
    Ok(())
}

#[test]
fn attention_residency_does_not_depend_on_optional_operand_names() {
    for owner in [None, Some(0), Some(1)] {
        for rank in 0..2 {
            let resident = owner.is_none_or(|owner| usize::from(owner) == rank);
            let layer = GlmLayer {
                dense: false, full_indexer: false,
                mlp_numerics: cuteafd_loader::plan::checkpoint::ModuleNumerics::CheckpointPrecision,
                split: true, attention_split: owner.is_none(), attention_owner: owner,
                attention_resident: resident, operands: HashMap::new(), native: HashMap::new(),
                core: Default::default(), native_projections: HashMap::new(),
            };
            assert_eq!(layer.holds_attention(), resident, "owner={owner:?}, rank={rank}");
            assert!(layer.range("q_a_norm").is_none());
            assert!(layer.range("w_uk").is_none());
        }
    }
}

#[test]
fn attention_residency_matches_installed_variant_checkpoint_headers() -> Result<()> {
    let home = std::env::var_os("HF_HOME").map(std::path::PathBuf::from)
        .unwrap_or_else(|| std::path::PathBuf::from(std::env::var_os("HOME").unwrap()).join(".cache/huggingface"));
    for (model, revision, index_file, q_lora) in [
        ("zai-org--GLM-5.3", "aca966e4e02791568aa6a4ced368624b3d897f42", "model.safetensors.index.json", true),
        ("mistralai--Mistral-Large-4-1T-A52B", "f12dabdc0c1e99e7f7924461a6f308859b05af4d", "consolidated.safetensors.index.json", false),
    ] {
        let snapshot = home.join("hub").join(format!("models--{model}")).join("snapshots").join(revision);
        if !snapshot.join("config.json").is_file() {
            eprintln!("checkpoint headers unavailable: {}", snapshot.display());
            continue;
        }
        let cfg = GlmDsaConfig::read(&snapshot)?;
        let index: serde_json::Value = serde_json::from_slice(&std::fs::read(snapshot.join(index_file))?)?;
        let mut headers = HashMap::new();
        for layer in 0..cfg.layers {
            // Both variants load KV-B on the attention owner, but ML4 has no q-LoRA norm.
            assert_eq!(cfg.weight(layer, Op::QANorm).is_some(), q_lora);
            let name = cfg.weight(layer, Op::KvB).context("resident attention KV-B")?;
            let shard = index["weight_map"][&name].as_str().with_context(|| format!("missing attention {name}"))?;
            if !headers.contains_key(shard) {
                headers.insert(shard.to_owned(), cuteafd_loader::read_safetensors_metadata(&snapshot.join(shard))?);
            }
            let tensor = headers[shard].iter().find(|t| t.name == name).with_context(|| format!("missing header {name}"))?;
            assert_eq!(tensor.shape, vec![cfg.heads * (cfg.nope_dim(layer) + cfg.v_head_dim), cfg.kv_lora_rank],
                "{model} layer {layer}: {name}");
        }
    }
    Ok(())
}

/// Deterministic, finite payloads: FP32 grids are positive normals, BF16 and
/// E4M3 bytes vary with the tensor and position.
fn fill(root: &std::path::Path) -> Result<()> {
    let catalog = cuteafd_loader::read_expert_catalog(root)?;
    for tensor in catalog.tensors() {
        let meta = &tensor.metadata;
        let seed = Sha256::digest(meta.name.as_bytes());
        let bytes: Vec<u8> = match meta.dtype {
            DType::F32 => (0..meta.byte_length / 4).flat_map(|i| {
                let v = 0.5 + ((i as u32).wrapping_mul(2654435761) ^ u32::from(seed[0])) as f32 / u32::MAX as f32;
                v.to_le_bytes()
            }).collect(),
            DType::Bf16 => (0..meta.byte_length / 2).flat_map(|i| {
                let v = (((i as u32).wrapping_mul(40503) ^ u32::from(seed[1])) % 2000) as f32 / 1000.0 - 1.0;
                ((v.to_bits() >> 16) as u16).to_le_bytes()
            }).collect(),
            _ => (0..meta.byte_length).map(|i| (i as u8).wrapping_mul(31) ^ seed[(i % 32) as usize] & 0x77).collect(),
        };
        std::fs::File::options().write(true).open(root.join(&tensor.shard))?
            .write_all_at(&bytes, meta.byte_offset)?;
    }
    Ok(())
}

fn digests(library: &NativeLibrary, layers: &[Vec<GlmLayer<'_>>], norm: &DeviceAllocation<'_>,
    head: &DeviceAllocation<'_>) -> Result<std::collections::BTreeMap<String, String>> {
    let mut out = std::collections::BTreeMap::new();
    let mut put = |key: String, buffer: CuteafdDeviceBuffer| -> Result<()> {
        let mut bytes = vec![0u8; buffer.bytes];
        library.copy_d2h(&mut bytes, buffer)?;
        out.insert(key, format!("{:x}", Sha256::digest(&bytes)));
        Ok(())
    };
    for (rank, layers) in layers.iter().enumerate() {
        for (index, layer) in layers.iter().enumerate() {
            for (name, allocation) in &layer.operands {
                if *name == "w_ik" { continue; }
                put(format!("rank{rank}.layer{index}.{name}"), allocation.buffer)?;
            }
        }
    }
    put("norm".into(), norm.buffer)?;
    put("head".into(), head.buffer)?;
    Ok(out)
}

/// Set CUTEAFD_WRITE_GLM5_OPERAND_GOLDEN=1 to regenerate (only from a loader
/// already known to be correct).
#[test]
#[ignore = "requires an allocated CPU build slot and explicit NVMe fixture directory"]
fn head_split_operands_match_the_mount_golden() -> Result<()> {
    let fixture_dso = Fixture::build()?;
    let library = fixture_dso.load()?;
    for device in 0..2 { fixture_dso.configure_memory(&library, device, 64 << 30)?; }
    let root = tempfile::tempdir_in(fixture_dso.directory())?;
    let tensors = fixture::glm5_tensors(|name, n, k| (0..256).flat_map(|expert|
        fixture::fp8(&name.replace(".experts.0.", &format!(".experts.{expert}.")), n, k, None)).collect());
    fixture::write_snapshot(root.path(), &fixture::glm5_config(), &tensors, None);
    fill(root.path())?;
    let mut catalog = cuteafd_loader::read_expert_catalog(root.path())?;
    catalog.select_draft(cuteafd_loader::plan::spec::DraftOwnership::None)?;
    let cfg = GlmDsaConfig::read(root.path())?;
    let stream = std::ptr::null_mut();
    let loader = GlmLoader { library: &library, catalog: &catalog, stream, device: 0,
        peers: vec![crate::shared::peer_split::RankDevice { device: 1, stream }], push: None };
    let (model, shares) = loader.model(&cfg, cfg.layers, None, false)?;
    let layers: Vec<Vec<GlmLayer<'_>>> = std::iter::once(model.layers).chain(shares).collect();
    let actual = digests(&library, &layers, &model.norm, model.head.bf16()?)?;
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src/families/glm5/weights_golden.json");
    if std::env::var_os("CUTEAFD_WRITE_GLM5_OPERAND_GOLDEN").is_some() {
        std::fs::write(&path, serde_json::to_vec_pretty(&actual)?)?;
    }
    let golden: std::collections::BTreeMap<String, String> = serde_json::from_slice(&std::fs::read(&path)?)?;
    assert_eq!(actual.keys().collect::<Vec<_>>(), golden.keys().collect::<Vec<_>>(), "operand inventory changed");
    for (key, digest) in &golden {
        assert_eq!(&actual[key], digest, "{key} differs from the mount golden");
    }
    Ok(())
}

/// The same operands pushed: the loader allocates and records class-2
/// operands, each GPU's plan replays through a ring into them, and the
/// deferred transforms run after delivery. Every operand matches the mount golden.
#[test]
#[ignore = "requires an allocated CPU build slot and explicit NVMe fixture directory"]
fn pushed_head_split_operands_match_the_mount_golden() -> Result<()> {
    let fixture_dso = Fixture::build()?;
    let library = fixture_dso.load()?;
    for device in 0..2 { fixture_dso.configure_memory(&library, device, 64 << 30)?; }
    let root = tempfile::tempdir_in(fixture_dso.directory())?;
    let tensors = fixture::glm5_tensors(|name, n, k| (0..256).flat_map(|expert|
        fixture::fp8(&name.replace(".experts.0.", &format!(".experts.{expert}.")), n, k, None)).collect());
    fixture::write_snapshot(root.path(), &fixture::glm5_config(), &tensors, None);
    fill(root.path())?;
    let mut catalog = cuteafd_loader::read_expert_catalog(root.path())?;
    catalog.select_draft(cuteafd_loader::plan::spec::DraftOwnership::None)?;
    let cfg = GlmDsaConfig::read(root.path())?;
    let stream = std::ptr::null_mut();
    let sink = std::rc::Rc::new(super::landing::PushSink::default());
    let loader = GlmLoader { library: &library, catalog: &catalog, stream, device: 0,
        peers: vec![crate::shared::peer_split::RankDevice { device: 1, stream }], push: Some(sink.clone()) };
    let (model, shares) = loader.model(&cfg, cfg.layers, None, false)?;
    let (records, deferred) = sink.take();
    let (sparks, world) = (6usize, 8usize);
    for gpu in 0..2 {
        use crate::shared::experts::rtx::dense::{DenseArena, RankLayout};
        let device = crate::shared::memory::device::Device { library: &library, id: gpu };
        let layout = RankLayout::new(records.clone(), Vec::new(), gpu)?;
        let checked = layout.checked(root.path(), world, sparks + gpu as usize)?;
        let expected = crate::shared::memory::weight_arena::ExpectedCoverage::from_plan(&checked, (sparks + gpu as usize) as u32)?;
        device.run(|| -> Result<()> {
            let ring = cuteafd_ffi::push_ring::RegisteredRing::new(&library, 8, cuteafd_ffi::push_ring::DEFAULT_SLOT_BYTES)?;
            let fake = crate::shared::experts::rtx::dense::testing::FakeRank::new(&checked, root.path(), (sparks + gpu as usize) as u32, &ring)?;
            let arena = DenseArena::new(device, layout, expected)?;
            crate::shared::memory::push_consumer::consume(device, arena, ring, fake, 8)
        })?;
    }
    for job in &deferred {
        crate::shared::memory::device::Device { library: &library, id: job.device }.run(|| job.run(&library, stream))?;
    }
    let layers: Vec<Vec<GlmLayer<'_>>> = std::iter::once(model.layers).chain(shares).collect();
    let actual = digests(&library, &layers, &model.norm, model.head.bf16()?)?;
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src/families/glm5/weights_golden.json");
    let golden: std::collections::BTreeMap<String, String> = serde_json::from_slice(&std::fs::read(&path)?)?;
    for (key, digest) in &golden {
        assert_eq!(&actual[key], digest, "pushed {key} differs from the mount golden");
    }
    Ok(())
}

fn retained_weight_plan(inventory: &cuteafd_loader::placement::families::glm5::weights::WeightInventory)
    -> Result<cuteafd_loader::placement::planner::ResolvedPlan> {
    use cuteafd_loader::placement::planner::{self, *};
    let facts = Facts {phase_order: cuteafd_loader::placement::planner::PhaseOrder::DRAFT_THEN_TARGET, allocations: inventory.owners.iter().map(|o| o.allocation.clone()).collect(),
        deferred: vec![], sequential: vec![], reservations: vec![], allowances: vec![] };
    let budgets: Vec<_> = (0..inventory.ranks).map(|rank| Location::coordinator(rank as u8))
        .chain([Location::PINNED]).enumerate().map(|(index, location)| Budget {
            id: format!("weights.fixture.{index}"), locations: vec![location],
            physical_bytes: 1 << 34, logical_bytes: 1 << 34, headroom: 0,
            baseline: Baseline::Forecast { occupied: 0, source: "CPU retained-weight fixture".into(), assumed: false },
        }).collect();
    let prefill = PrefillShape { rows: 128, waves: 1 };
    Ok(ResolvedPlan { request: Request { context: 64, pool: Pin::Exact(64), onboard: Pin::Auto,
            prefill: Pin::Exact(prefill), spark_reply_format: None, operation: Operation::Serving },
        decisions: Decisions { spark_reply_format: None, attention: vec![], onboard_layers: vec![], expert_homes: Default::default(),
            embedding: None, prefill, lanes: 1 }, pool_tokens: 64,
        phases: planner::evaluate(&facts, &budgets)?, budgets, facts })
}

#[test]
#[ignore = "requires an allocated CPU build slot and explicit NVMe fixture directory"]
fn retained_weight_owners_match_actual_loader_and_physical_embedding() -> Result<()> {
    use crate::shared::token_io::{EmbedPlacement, EmbedSource, TokenEmbedding};
    use cuteafd_loader::placement::{families::glm5::weights::WeightInventory, planner::{self, *}};
    let fixture_dso = Fixture::build()?;
    let library = fixture_dso.load()?;
    for device in 0..2 { fixture_dso.configure_memory(&library, device, 1 << 34)?; }
    // Allocation-only checks: the fake DSO does not qualify forward numerics.
    for (ranks, owned, host, native_shared, core, first_device) in [
        (1, false, false, false, false, 0), (2, false, false, false, false, 0),
        (2, true, true, false, false, 0), (2, false, true, true, false, 0),
        (2, true, false, false, true, 0), (1, false, false, false, false, 1),
        (2, true, true, false, false, 1), (2, false, false, false, false, 1),
    ] {
        let physical_devices = if ranks == 2 { vec![first_device, 1 - first_device] } else { vec![first_device] };
        library.cuda_set_device(first_device)?;
        let root = tempfile::tempdir_in(fixture_dso.directory())?;
        let (mut config, mut tensors) = fixture::glm5_compact_weights();
        if core {
            config["vocab_size"] = serde_json::json!(256);
            let cfg = GlmDsaConfig::from_hf(&config)?;
            tensors = tensors.into_iter().filter(|t| !t.0.ends_with("_scale_inv"))
                .flat_map(|(name, dtype, mut shape)| {
                    if ["lm_head.weight", "model.embed_tokens.weight"].contains(&name.as_str()) { shape[0] = 256; }
                    if let Some(stem) = name.strip_suffix(".weight").filter(|stem|
                        cuteafd_loader::families::glm5::core::split(&cfg, stem).is_some()) {
                        let n = if stem.ends_with("kv_a_proj_with_mqa") { shape[0].next_multiple_of(128) } else { shape[0] };
                        fixture::exl3(stem, n, shape[1], 6)
                    } else {
                        vec![(name, if dtype == "F8_E4M3" { "BF16" } else { dtype }, shape)]
                    }
                }).collect();
        }
        fixture::write_snapshot(root.path(), &config, &tensors, None);
        let mut catalog = cuteafd_loader::read_expert_catalog(root.path())?;
        catalog.select_draft(cuteafd_loader::plan::spec::DraftOwnership::None)?;
        if core {
            for tensor in catalog.tensors().iter().filter(|t| t.metadata.name.ends_with(".mcg")) {
                std::fs::File::options().write(true).open(root.path().join(&tensor.shard))?
                    .write_all_at(&0xcbac1fedu32.to_le_bytes(), tensor.metadata.byte_offset)?;
            }
        }
        let cfg = GlmDsaConfig::from_hf(&config)?;
        let checkpoint = cuteafd_loader::plan::Checkpoint::coordinator(root.path(), false, false)?;
        let owners = owned.then_some([0, 1]);
        let home = if host { EmbeddingHome::HostPinned } else { EmbeddingHome::Device(0) };
        let mut inventory = WeightInventory::from_checkpoint(&checkpoint, ranks, cfg.layers,
            owners.as_ref().map(|o| o.as_slice()), bf16_programs(), native_shared, home)?;
        let stream = std::ptr::null_mut();
        let loader = GlmLoader { library: &library, catalog: &catalog, stream, device: first_device,
            peers: (ranks == 2).then_some(crate::shared::peer_split::RankDevice { device: 1 - first_device, stream })
                .into_iter().collect(), push: None };
        let tensor = catalog.tensor("model.embed_tokens.weight")?;
        let source = EmbedSource::new(root.path(), &tensor.shard, &tensor.metadata, cfg.hidden)?;
        let (embedding, (model, mut peers)) = TokenEmbedding::load(&library, source,
            if host { EmbedPlacement::Host } else { EmbedPlacement::Gpu },
            || loader.model(&cfg, cfg.layers, owners.as_ref().map(|o| o.as_slice()), native_shared))?;
        if ranks == 2 {
            loader.peer_routers(&cfg, &mut peers[0], 1..2)?;
            inventory.add_peer_routers(&[1])?;
        }
        let plan = retained_weight_plan(&inventory)?;
        let receipts = model.allocation_receipts(&peers, &embedding, &physical_devices)?;
        planner::reconcile(&plan, Phase::Warmup, &receipts)?;
        assert!(model.allocation_receipts(&peers, &embedding, &[]).is_err());
        assert!(model.allocation_receipts(&peers, &embedding, &[-1]).is_err());
        if ranks == 2 {
            assert!(model.allocation_receipts(&peers, &embedding, &[first_device, first_device]).is_err());
            assert!(model.allocation_receipts(&peers, &embedding, &[1 - first_device, first_device]).is_err());
        } else {
            assert!(model.allocation_receipts(&peers, &embedding, &[1 - first_device]).is_err());
        }
        assert!(embedding.allocation_receipt("fixture", "fixture", Location::PINNED, first_device).is_err());
        assert!(embedding.allocation_receipt("fixture", "fixture", Location::coordinator(0), 1 - first_device).is_err());
        let table = receipts.iter().find(|r| r.allocation.id == "glm.weights.embedding").unwrap();
        assert_eq!(table.allocation.location, if host { Location::PINNED } else { Location::coordinator(0) });
        assert_ne!(table.address, 0);
        let mut missing = receipts.clone();
        missing.pop();
        assert!(planner::reconcile(&plan, Phase::Warmup, &missing).is_err());
        let mut extra = receipts.clone();
        extra[0].allocation.id.push_str(".extra");
        assert!(planner::reconcile(&plan, Phase::Warmup, &extra).is_err());
        let mut changed = receipts.clone();
        changed[0].allocation.bytes += 1;
        changed[1].allocation.bytes -= 1;
        assert!(planner::reconcile(&plan, Phase::Warmup, &changed).is_err());
        let mut wrong_device = receipts;
        wrong_device[0].allocation.location = Location::coordinator(7);
        assert!(planner::reconcile(&plan, Phase::Warmup, &wrong_device).is_err());
    }
    Ok(())
}

fn vocabulary_manifest() -> serde_json::Value {
    use serde_json::json;
    let pointer = |name, dtype, shape, role| json!({"name":name,"dtype":dtype,"shape":shape,"role":role});
    let scalars = json!([{"name":"rows","type":"int32"},{"name":"cta_limit","type":"int32"}]);
    json!({"programs":[{
        "name":"exlr8_dense_fp8_row_embedding_k128_n256","capacity_rows":4096,
        "geometry":{"abi_version":1,"max_rows":4096,"hidden":128,"vocabulary":256,"ctas_per_sm":4},
        "pointers":[pointer("ids","int32","[rows]","in"),
            pointer("payload","uint8","[256,128]","in"),
            pointer("scales","float32","[256]","in"),
            pointer("output","bfloat16","[rows,128]","out")],
        "scalars":scalars,"scratch_bytes_at_capacity":{}
    },{
        "name":"exlr8_dense_fp8_row_k128_n256_s1","capacity_rows":4096,
        "geometry":{"abi_version":1,"max_rows":4096,"input_features":128,"output_features":256,
            "sharded_side":1,"ctas_per_sm":4,"splits":1,"split_k":128,"segments":1,
            "encoding":"fp8_row","core":"tile"},
        "pointers":[pointer("x","bfloat16","[rows,128]","in"),
            pointer("payload","uint8","[32768]","in"),
            pointer("scales","float32","[256]","in"),
            pointer("partial","float32","[0]","scratch"),
            pointer("output","float32","[rows,256]","out")],
        "scalars":scalars,"scratch_bytes_at_capacity":{"partial":0}
    }]})
}

fn vocabulary_records(root: &std::path::Path)
    -> Result<Vec<cuteafd_loader::plan::exlr8::dense::DenseRecord>> {
    use cuteafd_loader::plan::exlr8::dense::*;
    let mut records = Vec::new();
    for name in ["model.embed_tokens.weight", "lm_head.weight", "model.norm.weight"] {
        let norm = name == "model.norm.weight";
        let path = root.join(name);
        let payload = if norm { vec![0; 256] } else { vec![0x38; 256 * 128] };
        let scales = if norm { vec![] } else { 0.5f32.to_le_bytes().repeat(256) };
        std::fs::write(&path, [&payload[..], &scales[..]].concat())?;
        let shape = if norm { vec![128] } else { vec![256, 128] };
        let mut components = vec![DenseComponent { suffix: if norm {"bf16"} else {"fp8"}.into(),
            dtype: if norm {"BF16"} else {"U8"}.into(), shape: shape.clone(), window: None,
            source: DenseRange { path: path.clone(), offset: 0, bytes: payload.len() } }];
        if !norm {
            components.push(DenseComponent { suffix: "scale".into(), dtype: "F32".into(), shape: vec![256],
                window: None, source: DenseRange { path, offset: payload.len() as u64, bytes: scales.len() } });
        }
        records.push(DenseRecord { tensor: name.into(),
            encoding: if norm {DenseEncoding::Bf16} else {DenseEncoding::RowFp8}, shape,
            side: if norm {ShardSide::Replicated} else {ShardSide::Output},
            ownership: DenseOwnership::new(DenseExecution::RankZero),
            channels_per_shard: if norm {0} else {128}, rotation_id: 0, rotation: None,
            hessian_solved: false, components, native_blocks: if norm {0} else {2} });
    }
    Ok(records)
}

#[test]
#[ignore = "requires an allocated CPU build slot and explicit NVMe fixture directory"]
fn native_vocabulary_retained_owner_sharing_and_load_history_reconcile() -> Result<()> {
    use crate::shared::token_io::{EmbedSource, TokenEmbedding, native::{NativeVocabulary, VocabularyPlan}};
    use cuteafd_ffi::memory_ledger::{AllocationAudit, AllocationEvent, Space};
    use cuteafd_loader::plan::exlr8::dense::{DensePlan, DenseSelection};
    use cuteafd_loader::placement::{families::glm5::weights::WeightInventory, planner::{self, *}};
    use std::rc::Rc;
    let fixture_dso = Fixture::build()?;
    let library = fixture_dso.load()?;
    fixture_dso.native_vocabulary_programs(&library)?;
    for device in 0..2 { fixture_dso.configure_memory(&library, device, 16 << 20)?; }
    let root = tempfile::tempdir_in(fixture_dso.directory())?;
    let path = root.path().join("programs.json");
    std::fs::write(&path, serde_json::to_vec(&vocabulary_manifest())?)?;
    let programs = library.programs()?.with_manifest(&path)?;
    let records = vocabulary_records(root.path())?;
    let selection = DenseSelection::from_hf(&fixture::glm5_config())?;
    for physical_devices in [vec![0], vec![1], vec![0, 1], vec![1, 0]] {
        let ranks = physical_devices.len();
        library.cuda_set_device(physical_devices[0])?;
        let audit = AllocationAudit::start();
        let vocabulary_plan = || VocabularyPlan::new(&records, &selection, ranks, 0, 128, 256);
        let staging_bytes = vocabulary_plan()?.staging_bytes();
        assert_eq!(staging_bytes, 128 * (128 + 4));
        // Missing geometry and admission failures must allocate nothing.
        let metadata_only = library.programs()?;
        assert!(NativeVocabulary::load(&library, &metadata_only, vocabulary_plan()?, 1 << 20, 0, 0).is_err());
        assert!(NativeVocabulary::load(&library, &programs, vocabulary_plan()?, staging_bytes - 1, 0, 0).is_err());
        assert!(NativeVocabulary::load(&library, &programs, vocabulary_plan()?, 1 << 20, 16 << 20, 0).is_err());
        if ranks == 2 {
            library.cuda_set_device(physical_devices[1])?;
            assert!(NativeVocabulary::load(&library, &metadata_only,
                VocabularyPlan::new(&records, &selection, ranks, 1, 128, 256)?, 0, 0, 0)?.is_none());
            library.cuda_set_device(physical_devices[0])?;
        }
        assert!(audit.events().is_empty());
        let owner = Rc::new(NativeVocabulary::load(&library, &programs, vocabulary_plan()?, 1 << 20, 0, 0)?
            .context("fixture vocabulary owner")?);
        assert_eq!(owner.bytes(), 2 * (256 * 128 + 256 * 4));
        let source = || EmbedSource { path: records[0].components[0].source.path.clone(), offset: 0,
            vocab: 256, hidden: 128, snapshot: None, tensor_name: records[0].tensor.clone() };
        let embedding = TokenEmbedding::row_fp8(&library, source(), owner.clone())?;
        let mut model = GlmWeights { layers: vec![], norm: DeviceAllocation::new(&library, 256)?,
            head: GlmHead::RowFp8(owner.clone()), rope_alpha: None };
        let peers = (1..ranks).map(|_| Vec::new()).collect::<Vec<_>>();
        let plans = (0..ranks).map(|rank| DensePlan::for_rank(&records, ranks, rank)).collect::<std::result::Result<Vec<_>, _>>()?;
        let inventory = WeightInventory::from_native(&selection, 1, &plans)?;
        let mut plan = retained_weight_plan(&inventory)?;
        let receipts = model.allocation_receipts(&peers, &embedding, &physical_devices)?;
        assert_eq!(receipts.len(), 6); // Four native owners, one placeholder and the norm.
        assert_eq!(receipts.iter().map(|r| r.allocation.bytes).sum::<u64>(), 68096);
        assert!(receipts.iter().all(|r| r.allocation.location == Location::coordinator(0)));
        assert_eq!(receipts.iter().map(|r| r.address).collect::<std::collections::BTreeSet<_>>().len(), 6);
        assert_eq!(Rc::strong_count(&owner), 3);
        planner::reconcile(&plan, Phase::Warmup, &receipts)?;
        // One shared pinned stream buffer is created and freed during Load;
        // its observed address/extent comes from the allocator, not the forecast.
        let events = audit.events();
        assert_eq!(events.len(), 8);
        let staging = match events[0] {
            AllocationEvent::Allocated(record) => record,
            _ => anyhow::bail!("staging was not the first native allocation"),
        };
        assert_eq!((staging.key.space, staging.key.scope, staging.bytes),
            (Space::Pinned, "native-vocabulary-staging", staging_bytes));
        let staging_spec = Allocation::buffer("fixture.vocabulary.staging", Location::PINNED,
            staging_bytes as u64, Lifetime { first: Phase::Load, last: Phase::Load },
            cuteafd_core::memory_layout::Category::Staging, "native vocabulary load");
        let observed = Receipt { allocation: Allocation::buffer("fixture.vocabulary.staging", Location::PINNED,
            staging.bytes as u64, staging_spec.lifetime, staging_spec.category, "native vocabulary load"),
            address: staging.address as u64 };
        plan.facts.allocations.push(staging_spec);
        plan.phases = planner::evaluate(&plan.facts, &plan.budgets)?;
        let history = events.iter().map(|event| -> Result<OwnerEvent> {
            let (record, allocated) = match event {
                AllocationEvent::Allocated(record) => (record, true),
                AllocationEvent::Freed(record) => (record, false),
            };
            let receipt = if record.key.space == Space::Pinned {
                ensure!(record.address == staging.address && record.bytes == staging_bytes, "unknown staging event");
                observed.clone()
            } else {
                ensure!(record.key.space == Space::Device && record.key.device == physical_devices[0], "unknown device event");
                let receipt = receipts.iter().find(|r| r.address == record.address as u64).context("unowned native allocation")?;
                ensure!(receipt.allocation.bytes == record.bytes as u64, "native allocation extent mismatch");
                receipt.clone()
            };
            Ok(if allocated { OwnerEvent::Allocated { receipt, phase: Phase::Load } }
                else { OwnerEvent::Freed { receipt, phase: Phase::Load } })
        }).collect::<Result<Vec<_>>>()?;
        planner::reconcile_events(&plan, Phase::Warmup, &history, &[])?;
        let without_free = history.iter().filter(|event| matches!(event, OwnerEvent::Allocated { .. })).cloned().collect::<Vec<_>>();
        assert!(planner::reconcile_events(&plan, Phase::Warmup, &without_free, &[]).is_err());
        for variant in 0..7 {
            let mut changed = receipts.clone();
            match variant {
                0 => { changed.pop(); }
                1 => { let mut extra = changed[0].clone(); extra.allocation.id.push_str(".extra"); changed.push(extra); }
                2 => { changed[0].allocation.bytes += 1; }
                3 => { changed[0].allocation.bytes += 1; changed[1].allocation.bytes -= 1; }
                4 => { changed[0].address = 0; }
                5 => { changed[0].allocation.location = Location::coordinator(1); }
                _ => { changed[0].address = changed[1].address; }
            }
            assert!(planner::reconcile(&plan, Phase::Warmup, &changed).is_err(), "mismatch {variant}");
        }
        // Equal geometry is not ownership: another real constructor makes a
        // distinct Rc and cannot replace the embedding's retained head owner.
        let other = Rc::new(NativeVocabulary::load(&library, &programs, vocabulary_plan()?, 1 << 20, 0, 0)?
            .context("second fixture vocabulary")?);
        model.head = GlmHead::RowFp8(other.clone());
        assert!(model.allocation_receipts(&peers, &embedding, &physical_devices).is_err());
        model.head = GlmHead::RowFp8(owner.clone());
        drop(other);
        let saved = model.norm.buffer.device_id;
        model.norm.buffer.device_id = 1 - saved;
        assert!(model.allocation_receipts(&peers, &embedding, &physical_devices).is_err());
        model.norm.buffer.device_id = saved;
        drop((model, embedding, owner));
        for &device in &physical_devices {
            library.cuda_set_device(device)?;
            assert_eq!(library.cuda_memory_info()?.0, 16 << 20);
        }
    }
    Ok(())
}

#[test]
fn native_decode_prefetch_keeps_projection_order_before_budget_and_peer_has_no_head() {
    use crate::shared::l2_prefetch::{prefix, Range};
    let range = |id: usize| -> Range { (std::ptr::without_provenance::<c_void>(id), 1 << 20) };
    let expected = ["q_a", "kv_a", "q_b", "index_q", "index_k", "index_w", "o", "gate", "up", "down"];
    for rank in 0..2 {
        let base = 100 * (rank + 1);
        let ordered = decode_ranges(|_| None, |name| {
            let index = expected.iter().position(|&p| p == name).unwrap();
            vec![range(base + index)]
        });
        assert_eq!(ordered, (0..expected.len()).map(|i| range(base + i)).collect::<Vec<_>>());
        assert_eq!(prefix(&ordered, (2 << 20) + (1 << 19)),
            [range(base), range(base + 1), (range(base + 2).0, 1 << 19)]);
        let head = head_read_order(rank, range(1), || vec![range(2), range(3), range(4)]);
        if rank == 0 { assert_eq!(head, [range(1), range(2), range(3), range(4)]); }
        else { assert!(head.is_empty()); }
    }
    assert!(head_read_order(1, range(1), || panic!("rank 1 touched rank-0 head")).is_empty());
    // Ordinary FP8 ordering remains unchanged, including payload then scales.
    let names = ["input_norm", "w_qkv_a", "q_a_norm", "kv_a_norm", "w_q_b", "w_iq", "w_ik",
        "k_norm_w", "k_norm_b", "w_uk", "w_uv", "w_o", "post_norm", "gate", "gate.bias", "w_gate_up", "w_down"];
    let fp8 = |name: &str| match name {
        "w_qkv_a_fp8" => Some(range(10)), "w_qkv_a_scale" => Some(range(11)),
        _ => names.iter().position(|&n| n == name).map(|i| range(20 + i)),
    };
    assert_eq!(decode_ranges(fp8, |_| Vec::new()), crate::shared::l2_prefetch::operands(&names, fp8));
}

#[test]
fn exlr8_prefetch_enumerates_each_resident_encoding_in_read_order() {
    use crate::shared::l2_prefetch::Range;
    for rank in 0..2 {
        for components in [vec!["suh", "trellis", "svh"], vec!["q4", "s4"],
            vec!["fp8", "e8m0"], vec!["bf16"]] {
            let expected = components.iter().enumerate().map(|(i, _)|
                (std::ptr::without_provenance::<c_void>(100 * (rank + 1) + i), 1 << 20))
                .collect::<Vec<Range>>();
            // Reverse insertion cannot change read order, unlike HashMap iteration.
            let resident = components.iter().zip(&expected).rev().map(|(component, &range)|
                (crate::shared::experts::exlr8::dense_weights::key("q_a", component), range)).collect::<HashMap<_, _>>();
            let actual = native_projection_ranges("q_a", |key| resident.get(key).copied());
            assert_eq!(actual, expected);
            assert_eq!(crate::shared::l2_prefetch::prefix(&actual, 1 << 19),
                [(expected[0].0, 1 << 19)]);
            assert!(native_projection_ranges("kv_a", |key| resident.get(key).copied()).is_empty());
        }
    }
}

#[test]
fn k6_prefetch_enumerates_both_transform_vectors_and_payload() {
    let buffer = |id| CuteafdDeviceBuffer { ptr: std::ptr::without_provenance_mut::<c_void>(id),
        bytes: 1 << 20, device_id: 0, flags: 0 };
    let ranges = trellis_ranges(buffer(2), buffer(1), buffer(3));
    assert_eq!(ranges, [1, 2, 3].map(|id| (buffer(id).ptr.cast_const(), buffer(id).bytes)));
    assert_eq!(crate::shared::l2_prefetch::prefix(&ranges, 3 << 20), ranges);
}
