//! Installed-header replay and compact real uploads share production dispatch.
use super::*;
use super::super::engine;
use crate::shared::token_io::{EmbedPlacement, EmbedSource, TokenEmbedding};
use cuteafd_core::memory_layout::{Category, MemoryKind};
use cuteafd_ffi::native_library_lifetime_fixture::Fixture;
use cuteafd_loader::placement::families::glm5::weights::WeightInventory;
use cuteafd_loader::placement::planner::{Allocation, CoordinatorDevices, EmbeddingHome, Lifetime};
use cuteafd_loader::plan::testing as fixture;
use std::{collections::BTreeMap, path::PathBuf};

fn installed(model: &str, revision: &str) -> Result<PathBuf> {
    let home = std::env::var_os("HF_HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            PathBuf::from(std::env::var_os("HOME").unwrap()).join(".cache/huggingface")
        });
    let root = home
        .join("hub")
        .join(format!("models--{model}"))
        .join("snapshots")
        .join(revision);
    ensure!(
        root.join("config.json").is_file(),
        "required checkpoint missing: {}",
        root.display()
    );
    Ok(root)
}

fn signatures(
    owners: impl IntoIterator<Item = Allocation>,
) -> BTreeMap<String, (cuteafd_loader::placement::planner::Location, u64, String)> {
    let mut out = BTreeMap::new();
    for owner in owners {
        assert!(
            out.insert(
                owner.id.clone(),
                (owner.location, owner.bytes, format!("{:?}", owner.category))
            )
            .is_none(),
            "duplicate {}",
            owner.id
        );
    }
    out
}

fn replay(
    catalog: &OfficialV41Catalog,
    cfg: &GlmDsaConfig,
    checkpoint: &cuteafd_loader::plan::Checkpoint,
    ranks: usize,
    owners: Option<&[u8]>,
    native_bf16: bool,
    home: EmbeddingHome,
) -> Result<Vec<Allocation>> {
    let core = CoreInventory::from_checkpoint(cfg, checkpoint)?;
    let mut requests = Vec::new();
    for layer in 0..cfg.layers {
        routes::layer_routes(
            catalog,
            cfg,
            layer,
            checkpoint,
            &core,
            ranks,
            owners.map(|o| o[layer]),
            false,
            native_bf16,
            |route| {
                requests.extend(route.requests);
                Ok(())
            },
        )?;
    }
    routes::global_routes(catalog, cfg, &core, |route| {
        requests.extend(route.requests);
        Ok(())
    })?;
    let tensor = catalog.tensor(cfg.embedding())?;
    let source = EmbedSource::new(
        catalog.snapshot(),
        &tensor.shard,
        &tensor.metadata,
        cfg.hidden,
    )?;
    requests.push(source.allocation_request("glm.weights.embedding", home)?);
    Ok(requests)
}

#[test]
fn installed_variant_weight_inventory_equals_production_requests() -> Result<()> {
    for (model, revision) in [
        (
            "zai-org--GLM-5.3",
            "aca966e4e02791568aa6a4ced368624b3d897f42",
        ),
        (
            "nvidia--GLM-5.3-NVFP4",
            "e3b8f9f2fad29727dfa95e766c5cac18ea1414bf",
        ),
        (
            "wrldsuksgo2mars--GLM-5.3-EXL3-K4-Core-K6",
            "5a4e83a09a11f3145332ed632d28a9894ea88225",
        ),
        (
            "wrldsuksgo2mars--Mistral-Large-4-EXL3-K4",
            "9c354cee78cd6feddb057549b8edde6e3b319822",
        ),
    ] {
        let root = installed(model, revision)?;
        let checkpoint = cuteafd_loader::plan::Checkpoint::open(&root)?;
        let catalog = cuteafd_loader::read_expert_catalog(&root)?;
        let cfg = GlmDsaConfig::read(&root)?;
        for ranks in [1, 2] {
            for whole in [false, true] {
                if whole && ranks == 1 {
                    continue;
                }
                let owners = (0..cfg.layers).map(|l| (l % 2) as u8).collect::<Vec<_>>();
                let owners = whole.then_some(owners.as_slice());
                for native_bf16 in [false, true] {
                    for home in [EmbeddingHome::Device(0), EmbeddingHome::HostPinned] {
                        let expected = WeightInventory::from_checkpoint(
                            &checkpoint,
                            ranks,
                            cfg.layers,
                            owners,
                            native_bf16,
                            false,
                            home,
                        )?;
                        let actual = replay(
                            &catalog,
                            &cfg,
                            &checkpoint,
                            ranks,
                            owners,
                            native_bf16,
                            home,
                        )?;
                        assert_eq!(
                            signatures(expected.owners.into_iter().map(|o| o.allocation)),
                            signatures(actual),
                            "{model} ranks={ranks} whole={whole} bf16={native_bf16} home={home:?}"
                        );
                    }
                }
            }
        }
    }
    Ok(())
}

fn arena_charge(components: &[cuteafd_loader::placement::planner::Component], id: &str,
    location: cuteafd_loader::placement::planner::Location, bytes: u64, logical: u64,
    kind: MemoryKind) -> Result<u64> {
    use cuteafd_loader::placement::planner::{Activation, ComponentKind};
    let roots = components.iter().filter_map(|component| match &component.kind {
        ComponentKind::Allocation { template, deferred, .. } if template.id == id => {
            assert!(!deferred);
            assert_eq!(component.activation, Activation::always());
            assert_eq!(component.bytes.at(1).unwrap(), bytes);
            Some(template)
        }
        _ => None,
    }).collect::<Vec<_>>();
    assert_eq!(roots.len(), 1, "one physical root {id}");
    let root = roots[0];
    assert_eq!((root.location, root.bytes, root.lifetime), (location, bytes, Lifetime::RESIDENT));
    assert!(root.view.is_none());
    assert_eq!((root.category, root.group.as_str()), (kind.category(), "weight arena"));
    let claims = components.iter().filter_map(|component| match &component.kind {
        ComponentKind::ResidentWeight(owner) if owner.id == id => {
            assert_eq!(component.activation, Activation::always());
            assert_eq!(component.bytes.at(1).unwrap(), logical);
            Some(owner)
        }
        _ => None,
    }).collect::<Vec<_>>();
    assert_eq!(claims.len(), 1, "one logical claim {id}");
    assert_eq!((claims[0].location, claims[0].weights.0), (location, logical));
    let tails = components.iter().filter_map(|component| match &component.kind {
        ComponentKind::AllocationGranularity(owner) if owner.backings.iter().any(|b| b.backing == id) => {
            assert_eq!(component.activation, Activation::always());
            assert_eq!(component.bytes.at(1).unwrap(), owner.bytes().unwrap());
            Some(owner)
        }
        _ => None,
    }).collect::<Vec<_>>();
    assert_eq!(tails.len(), 1, "one derived tail {id}");
    let tail = tails[0];
    assert_eq!(tail.id, format!("{id}.tail"));
    assert_eq!((tail.location, tail.lifetime), (location, root.lifetime));
    assert_eq!(tail.backings.len(), 1);
    assert_eq!(tail.backings[0].requests, [bytes]);
    assert!(bytes > 1 << 20, "installed immutable backing uses the large-request rule");
    let expected = bytes.div_ceil(2 << 20) * (2 << 20) - bytes;
    assert_eq!(tail.bytes()?, expected);
    Ok(expected)
}

fn arena_inventory(label: &str, inventory: &WeightInventory) -> Result<()> {
    use cuteafd_loader::placement::planner::{ComponentKind, Location, View};
    let components = inventory.components(1)?;
    for (rank, arena) in inventory.coordinator_arenas()?.iter().enumerate() {
        let location = Location::coordinator(rank as u8);
        let owners = inventory.owners.iter().filter(|owner| owner.allocation.location == location
            && !owner.allocation.id.contains(".shared.layer.")).collect::<Vec<_>>();
        let payload = owners.iter().map(|owner| owner.allocation.bytes).sum::<u64>();
        assert_eq!(arena.payload_bytes() as u64, payload);
        assert_eq!(arena.views().len(), owners.len());
        let mut regions = arena.views().values().copied().collect::<Vec<_>>();
        regions.sort_by_key(|view| view.offset());
        let mut end = 0;
        for view in regions {
            assert_eq!(view.offset() % 256, 0);
            assert!(view.offset() >= end);
            end = view.offset() + view.bytes();
        }
        assert_eq!(end, arena.bytes());
        let mut tail = arena_charge(&components, arena.id(), location, arena.bytes() as u64, payload, MemoryKind::Weights)?;
        let mut backing = arena.bytes() as u64;
        let mut old = payload;
        let mut roots = 1;
        for owner in owners {
            let matches = components.iter().filter_map(|component| match &component.kind {
                ComponentKind::Allocation { template, .. } if template.id == owner.allocation.id => Some(template),
                _ => None,
            }).collect::<Vec<_>>();
            let mut expected = owner.allocation.clone();
            expected.view = Some(View { backing: arena.id().into(),
                offset: arena.view(&expected.id).context("target alias recipe")?.offset() as u64 });
            assert_eq!(matches, [&expected], "noncharging alias preserves its physical operand");
        }
        let shared = inventory.owners.iter().filter(|owner| owner.allocation.location == location
            && owner.allocation.id.contains(".shared.layer.")).collect::<Vec<_>>();
        if !shared.is_empty() {
            let layout = cuteafd_loader::serving_capacity::WeightArenaLayout::new(format!("glm/shared/{rank}"),
                shared.iter().map(|owner| (owner.allocation.id.clone(), owner.allocation.bytes as usize)))?;
            tail += arena_charge(&components, layout.id(), location, layout.bytes() as u64,
                layout.payload_bytes() as u64, MemoryKind::Weights)?;
            backing += layout.bytes() as u64;
            old += layout.payload_bytes() as u64;
            roots += 1;
            for owner in shared {
                let aliases = components.iter().filter_map(|component| match &component.kind {
                    ComponentKind::Allocation { template, .. } if template.id == owner.allocation.id => Some(template),
                    _ => None,
                }).collect::<Vec<_>>();
                let mut expected = owner.allocation.clone();
                expected.view = Some(View { backing: layout.id().into(),
                    offset: layout.view(&expected.id).context("shared alias recipe")?.offset() as u64 });
                assert_eq!(aliases, [&expected], "shared layers alias the single rank backing");
            }
        }
        assert_eq!(components.iter().filter(|component| matches!(&component.kind,
            ComponentKind::Allocation { template, .. } if template.location == location && template.view.is_none())).count(), roots);
        assert_eq!(components.iter().filter(|component| matches!(&component.kind,
            ComponentKind::ResidentWeight(owner) if owner.location == location)).count(), roots);
        assert_eq!(components.iter().filter(|component| matches!(&component.kind,
            ComponentKind::AllocationGranularity(owner) if owner.location == location)).count(), roots);
        eprintln!("ARENA_CHARGE {label} rank={rank} roots={roots} old={old} B={backing} tail={tail} new={} delta={} old_rounding=no",
            backing + tail, backing + tail - old);
    }
    for owner in inventory.owners.iter().filter(|owner| owner.allocation.location == Location::PINNED) {
        let found = components.iter().filter_map(|component| match &component.kind {
            ComponentKind::Allocation { template, .. } if template.id == owner.allocation.id => Some(template),
            _ => None,
        }).collect::<Vec<_>>();
        assert_eq!(found, [&owner.allocation], "pinned embedding stays a separate owner");
    }
    Ok(())
}

#[test]
fn installed_arena_cpu_inventory() -> Result<()> {
    for (model, revision) in [
        ("wrldsuksgo2mars--GLM-5.3-EXL3-K4-Core-K6", "5a4e83a09a11f3145332ed632d28a9894ea88225"),
        ("wrldsuksgo2mars--Mistral-Large-4-EXL3-K4", "9c354cee78cd6feddb057549b8edde6e3b319822"),
    ] {
        let root = installed(model, revision)?;
        let checkpoint = cuteafd_loader::plan::Checkpoint::open(&root)?;
        let cfg = GlmDsaConfig::read(&root)?;
        for ranks in [1, 2] {
            let owners = cuteafd_loader::placement::families::glm5::attention_owners(&cfg, cfg.layers)?;
            for whole in [false, true] {
                if whole && ranks == 1 { continue; }
                for home in [EmbeddingHome::Device(0), EmbeddingHome::HostPinned] {
                    let inventory = WeightInventory::from_checkpoint(&checkpoint, ranks, cfg.layers,
                        whole.then_some(owners.as_slice()), false, false, home)?;
                    arena_inventory(&format!("{model} tp{ranks} whole={whole} embed={home:?}"), &inventory)?;
                }
            }
        }
    }
    let root = installed("kindlingai--glm-5.3-exlr8-k2-k3-k4", "a2b599e06c59f2ccb01aaa44e6f182ba21e39f80")?;
    let cfg = GlmDsaConfig::read(&root)?;
    let config = serde_json::from_slice(&std::fs::read(root.join("config.json"))?)?;
    let selection = cuteafd_loader::plan::exlr8::dense::DenseSelection::from_hf(&config)?;
    for ranks in [1, 2] {
        let records = (0..ranks).map(|rank|
            cuteafd_loader::plan::exlr8::dense::requests::NativeDenseRecords::new(&root, cfg.layers, ranks, rank, &BTreeMap::new()))
            .collect::<Result<Vec<_>>>()?;
        let plans = records.iter().map(|r| r.plan.clone()).collect::<Vec<_>>();
        let mut inventory = WeightInventory::from_native(&selection, cfg.layers, &plans)?;
        let all = cuteafd_loader::plan::exlr8::dense::DenseRecord::inventory(&root)?;
        let vocabulary = (0..ranks).map(|rank| crate::shared::token_io::native::VocabularyPlan::new(
            &all, &selection, ranks, rank, cfg.hidden, cfg.vocab_size)).collect::<Result<Vec<_>>>()?;
        inventory.extend_native(&selection, cfg.layers, &vocabulary.iter().map(|p| p.dense_plan()).collect::<Vec<_>>())?;
        let shared = cuteafd_loader::plan::exlr8::residency::MetadataInput::open(&root, "up1")?;
        let plans = (0..ranks).map(|rank| (cfg.first_moe_layer..cfg.layers).map(|layer|
            shared.plan(layer, ranks, rank, cuteafd_loader::plan::exlr8::residency::ExpertSet::Shared)
                .map_err(anyhow::Error::from)).collect::<Result<Vec<_>>>()).collect::<Result<Vec<_>>>()?;
        inventory.extend_shared(&plans)?;
        arena_inventory(&format!("GLM-EXLR8-up1 tp{ranks} context-or-heads"), &inventory)?;
    }
    Ok(())
}

#[test]
fn installed_glm_and_ml4_draft_arena_charges_equal_shared_load_recipes() -> Result<()> {
    use cuteafd_loader::{families::glm5::{draft_physical::{GlmDflashConfig, GlmDflashPhysicalLayout,
        gqa_checkpoint_arena_requests}, draft_representation::{GlmDraftCapacity, GlmDraftRepresentation}},
        families::qwen4::Qwen4KvCache, placement::{families::glm5::{admission::DraftFacts, ml4_draft_layout},
        planner::{ComponentKind, Location}}, serving_capacity::WeightArenaLayout};
    let snapshot = installed("incoai--GLM-5.3-DFlash2", "425aa615ce320caac34400208b30808c8f14f76c")?;
    let cfg = GlmDflashConfig::read(&snapshot)?;
    for representation in [GlmDraftRepresentation::Bf16Only, GlmDraftRepresentation::Fp8Only] {
        let capacity = GlmDraftCapacity::new(20, 16, cfg.block)?;
        let layout = GlmDflashPhysicalLayout::read(&snapshot, representation, capacity, 188, 0, Qwen4KvCache::Fp8)?;
        let logical = layout.resident_weights().0;
        let requests = gqa_checkpoint_arena_requests(&snapshot, representation, 0, 1)?;
        let arena = WeightArenaLayout::new("glm/draft/0", requests)?;
        assert_eq!(arena.payload_bytes() as u64, logical);
        let facts = DraftFacts::gqa(&snapshot, representation, capacity, 188, 0, Qwen4KvCache::Fp8)?;
        assert!(!facts.owners().iter().any(|owner| layout.weight_owners().any(|weight|
            owner.id == format!("glm.draft.{}", weight.id))), "no retired per-tensor charge");
        for home in facts.homes(2) {
            let components = facts.components(home, 1)?;
            let id = format!("glm/draft/{home}");
            let tail = arena_charge(&components, &id, Location::coordinator(home), arena.bytes() as u64, logical, MemoryKind::DrafterWeights)?;
            assert_eq!(components.iter().filter(|component| matches!(&component.kind,
                ComponentKind::ResidentWeight(_))).count(), 1);
            eprintln!("ARENA_CHARGE GLM-draft-{} home={home} old={logical} B={} tail={tail} new={} delta={} old_rounding=no",
                representation.name(), arena.bytes(), arena.bytes() as u64 + tail, arena.bytes() as u64 + tail - logical);
        }
    }
    let snapshot = installed("wrldsuksgo2mars--Mistral-Large-4-EXL3-K4",
        "9c354cee78cd6feddb057549b8edde6e3b319822")?;
    let layout = ml4_draft_layout(&snapshot, 20, 16, 16384, 188)?;
    let arena = WeightArenaLayout::new("glm/draft/0", layout.immutable_arena_requests())?;
    let logical = layout.owners().iter().map(|owner| owner.resident_weights.0).sum::<u64>();
    let generated = layout.owners().iter().find(|owner| owner.id == "markov_norms")
        .context("generated immutable ML4 norms")?;
    assert_eq!(generated.resident_weights.0, 0);
    let old = logical + generated.bytes as u64;
    assert_eq!(arena.payload_bytes() as u64, old);
    let facts = DraftFacts::ml4(&layout)?;
    assert!(!facts.owners().iter().any(|owner| layout.immutable_arena_requests().iter().any(|(name, _)|
        owner.id == format!("glm.draft.{name}"))), "no retired ML4 immutable charge");
    for home in facts.homes(2) {
        let components = facts.components(home, 1)?;
        let id = format!("glm/draft/{home}");
        let tail = arena_charge(&components, &id, Location::coordinator(home), arena.bytes() as u64, logical, MemoryKind::DrafterWeights)?;
        assert_eq!(components.iter().filter(|component| matches!(&component.kind,
            ComponentKind::ResidentWeight(_))).count(), 1);
        eprintln!("ARENA_CHARGE ML4-bundled-draft home={home} old={old} logical={logical} B={} tail={tail} new={} delta={} old_rounding=no",
            arena.bytes(), arena.bytes() as u64 + tail, arena.bytes() as u64 + tail - old);
    }
    Ok(())
}

#[test]
fn installed_k6_max_streaming_staging_covers_measured_pools() -> Result<()> {
    use crate::shared::experts::rtx::dense::{HostStaging, Rect};
    for (model, revision, observed, old_dense) in [
        ("wrldsuksgo2mars--GLM-5.3-EXL3-K4-Core-K6", "5a4e83a09a11f3145332ed632d28a9894ea88225",
            [240060416usize, 27942912], 11891580928usize),
    ] {
        let root = installed(model, revision)?;
        let checkpoint = cuteafd_loader::plan::Checkpoint::open(&root)?;
        let cfg = GlmDsaConfig::read(&root)?;
        let core = CoreInventory::from_checkpoint(&cfg, &checkpoint)?;
        let mut rects: [Vec<Vec<Rect>>; 2] = Default::default();
        let mut add = |rank: usize, layout: cuteafd_loader::formats::trellis_core::CoreWeightLayout| -> Result<()> {
            let outputs = core.get_weight(&layout.stem).context("native projection")?.stored_outputs;
            let layout = CoreWeightLayout::from_shared(layout);
            rects[rank].extend(layout.payload_rects(&checkpoint, outputs)?.into_iter().map(|(_, rect)| vec![rect]));
            Ok(())
        };
        for layer in 0..cfg.layers {
            for (rank, layouts) in core.layer_weights(&cfg, layer, 2, None)?.into_iter().enumerate() {
                for layout in layouts.into_values() { add(rank, layout)?; }
            }
        }
        if let Some(head) = core.get_weight(cfg.head()) {
            add(0, cuteafd_loader::formats::trellis_core::CoreWeightLayout::new(head,
                cuteafd_loader::formats::dense_trellis::TrellisSide::Output, 0, 1)?)?;
        }
        let charges = rects.iter().map(|steps| HostStaging::for_steps(steps.iter().map(Vec::as_slice), 0)
            .and_then(HostStaging::bytes)).collect::<Result<Vec<_>>>()?;
        for rank in 0..2 {
            ensure!(charges[rank] >= observed[rank], "{model} rank {rank}: {} < measured {}", charges[rank], observed[rank]);
        }
        ensure!(charges.iter().sum::<usize>() < old_dense / 2, "{model}: streaming bound must improve substantially");
        eprintln!("{model} native K6 max streaming staging bytes {charges:?}; measured pools {observed:?}; old full dense bound {old_dense}");
    }
    Ok(())
}

#[test]
fn official_ml4_native_inventory_keeps_expert_package_refusal() -> Result<()> {
    let root = installed(
        "mistralai--Mistral-Large-4-1T-A52B",
        "f12dabdc0c1e99e7f7924461a6f308859b05af4d",
    )?;
    let checkpoint = cuteafd_loader::plan::Checkpoint::open(&root)?;
    let cfg = GlmDsaConfig::read(&root)?;
    let inventory = WeightInventory::from_checkpoint(
        &checkpoint,
        1,
        cfg.layers,
        None,
        false,
        false,
        EmbeddingHome::Device(0),
    )?;
    let loader = cuteafd_loader::read_expert_catalog(&root).unwrap_err();
    assert!(!inventory.owners.is_empty());
    let core = CoreInventory::from_checkpoint(&cfg, &checkpoint)?;
    assert!(core.projections.is_empty());
    assert!(core.native.values().any(|layout| layout.fp8));
    assert!(core.native.values().any(|layout| !layout.fp8));
    assert!(loader.to_string().contains("no Spark expert package"));
    Ok(())
}

#[test]
fn original_ml4_native_inventory_equals_routes_and_single_arena_charges() -> Result<()> {
    let root = tempfile::tempdir()?;
    let mut tensors = fixture::mistral_large4_tensors(None, false);
    let experts = tensors.iter().filter_map(|(name, _, shape)| {
        let stem = name.strip_suffix(".weight")?;
        stem.contains(".experts.").then(|| (stem.to_owned(), shape.clone()))
    }).collect::<Vec<_>>();
    tensors.retain(|(name, ..)| !name.contains(".experts."));
    for (stem, shape) in experts {
        tensors.extend(fixture::exl3(&stem, shape[0], shape[1], 4));
    }
    let (mut quant, mut manifest) = fixture::mistral_large4_exl3_config(&tensors);
    for metadata in [&mut quant, &mut manifest] {
        for key in ["core_bits", "head_bits", "lm_head"] {
            metadata.as_object_mut().unwrap().remove(key);
        }
    }
    let mut config = fixture::mistral_large4_config();
    config.as_object_mut().unwrap().remove("transformers_weights");
    config["quantization_config"] = quant;
    fixture::write_snapshot(root.path(), &config, &tensors, None);
    fixture::write_quantize_config(root.path(), &manifest);
    // The native contract validates bundled drafter headers against Mistral's params.
    std::fs::write(root.path().join("params.json"), serde_json::to_vec(&serde_json::json!({
        "softmax_tanh_gating": 30.0,
        "mtp": {"n_layers": 3, "depth": 3, "ragged_attention": "2048,2048,2048",
            "target_activation_layers": "1,18,36,54,71", "norm_strategy": "eagle",
            "dspark": {"markov_rank": 512, "head_type": "rnn", "draft_input": "position_specific"}}
    }))?)?;
    let cfg = GlmDsaConfig::from_hf(&config)?;
    let checkpoint = cuteafd_loader::plan::Checkpoint::open(root.path())?;
    let catalog = cuteafd_loader::read_expert_catalog(root.path())?;
    let attention = (0..cfg.layers).map(|layer| (layer % 2) as u8).collect::<Vec<_>>();
    for ranks in [1, 2] {
        for whole in [false, true] {
            if whole && ranks == 1 { continue; }
            let owners = whole.then_some(attention.as_slice());
            for home in [EmbeddingHome::Device(0), EmbeddingHome::HostPinned] {
                let inventory = WeightInventory::from_checkpoint(&checkpoint, ranks, cfg.layers,
                    owners, true, false, home)?;
                let actual = replay(&catalog, &cfg, &checkpoint, ranks, owners, true, home)?;
                assert_eq!(signatures(inventory.owners.iter().map(|owner| owner.allocation.clone())),
                    signatures(actual), "original ML4 tp{ranks} whole={whole} embed={home:?}");
                arena_inventory(&format!("original ML4 tp{ranks} whole={whole} embed={home:?}"), &inventory)?;
            }
        }
    }
    Ok(())
}

fn compact_dso() -> Result<(Fixture, NativeLibrary)> {
    let root = PathBuf::from(std::env::var_os("HOME").context("HOME")?)
        .join(".cache/cuteafd/builds/glm-weight-inventory-fixtures");
    let fixture = Fixture::build_in(&root)?;
    let library = fixture.load()?;
    for device in [0, 1] {
        fixture.configure_memory(&library, device, 1 << 30)?;
    }
    fixture.quiet_events(&library, true)?;
    Ok((fixture, library))
}

#[test]
fn whole_attention_owner1_allocates_on_its_physical_device() -> Result<()> {
    compact_real_load(Some(&[1, 0]), false, EmbedPlacement::Gpu)
}

#[test]
fn compact_layer_global_and_embedding_uploads_equal_inventory() -> Result<()> {
    compact_real_load(None, false, EmbedPlacement::Gpu)?;
    compact_real_load(None, false, EmbedPlacement::Host)
}

fn compact_real_load(owners: Option<&[u8]>, core: bool, placement: EmbedPlacement) -> Result<()> {
    let (dso, library) = compact_dso()?;
    let root = tempfile::tempdir_in(dso.directory())?;
    let (mut config, mut tensors) = fixture::glm5_compact_weights();
    if !core && bf16_programs() {
        tensors.retain(|t| !t.0.ends_with("_scale_inv"));
        for (_, dtype, _) in &mut tensors {
            if *dtype == "F8_E4M3" {
                *dtype = "BF16";
            }
        }
    } else if !core {
        // Exercise retained calibrated activation scales on the dense MLP.
        config["quantization_config"]["producer"] = serde_json::json!({"name":"modelopt"});
        config["quantization_config"]["quant_algo"] = serde_json::json!("MIXED_PRECISION");
        let mut declared = serde_json::Map::new();
        for op in ["gate", "up", "down"] {
            let stem = format!("model.layers.0.mlp.{op}_proj");
            tensors.retain(|t| t.0 != format!("{stem}.weight_scale_inv"));
            tensors.push(fixture::t(format!("{stem}.weight_scale"), "F32", &[]));
            tensors.push(fixture::t(format!("{stem}.input_scale"), "F32", &[]));
            declared.insert(stem, serde_json::json!({"quant_algo":"FP8"}));
        }
        config["quantization_config"]["quantized_layers"] = declared.into();
    }
    if core {
        config["vocab_size"] = serde_json::json!(256);
        for (name, _, shape) in &mut tensors {
            if name == "lm_head.weight" || name == "model.embed_tokens.weight" {
                shape[0] = 256;
            }
        }
        let cfg = GlmDsaConfig::from_hf(&config)?;
        tensors = tensors
            .into_iter()
            .filter(|t| !t.0.ends_with("_scale_inv"))
            .flat_map(|(name, dtype, shape)| {
                if let Some(stem) = name.strip_suffix(".weight").filter(|stem| {
                    cuteafd_loader::families::glm5::core::split(&cfg, stem).is_some()
                }) {
                    fixture::exl3(
                        stem,
                        if stem.ends_with("kv_a_proj_with_mqa") {
                            shape[0].next_multiple_of(128)
                        } else {
                            shape[0]
                        },
                        shape[1],
                        6,
                    )
                } else {
                    vec![(name, if dtype == "F8_E4M3" { "BF16" } else { dtype }, shape)]
                }
            })
            .collect();
    }
    fixture::write_snapshot(root.path(), &config, &tensors, None);
    if core {
        use std::io::{Seek, SeekFrom, Write};
        let headers = cuteafd_loader::plan::Checkpoint::open(root.path())?;
        for tensor in &headers.tensors {
            if tensor.meta.name.ends_with(".mcg") {
                let mut file = std::fs::OpenOptions::new()
                    .write(true)
                    .open(root.path().join(&tensor.shard))?;
                file.seek(SeekFrom::Start(tensor.meta.byte_offset))?;
                file.write_all(&0xcbac1fedu32.to_le_bytes())?;
            }
        }
    }
    let mut catalog = cuteafd_loader::read_expert_catalog(root.path())?;
    catalog.select_draft(cuteafd_loader::plan::spec::DraftOwnership::None)?;
    let cfg = GlmDsaConfig::read(root.path())?;
    let checkpoint = cuteafd_loader::plan::Checkpoint::open(root.path())?;
    let layout_inventory = WeightInventory::from_checkpoint(&checkpoint, 2, cfg.layers, owners,
        bf16_programs(), false, if placement == EmbedPlacement::Host { EmbeddingHome::HostPinned }
            else { EmbeddingHome::Device(0) })?;
    let arenas = layout_inventory.coordinator_arenas()?.into_iter().enumerate().map(|(rank, layout)| {
        Ok((rank as i32, crate::shared::memory::weight_arena::DeviceWeightArena::new(
            Device { library: &library, id: rank as i32 }, layout, crate::shared::memory::TEST_ARENA)?))
    }).collect::<Result<BTreeMap<_, _>>>()?;
    let loader = GlmLoader {
        library: &library,
        catalog: &catalog,
        stream: std::ptr::null_mut(),
        device: 0,
        peers: vec![crate::shared::peer_split::RankDevice {
            device: 1,
            stream: std::ptr::null_mut(),
        }],
        arenas: None, allocation_prefix: String::new(), push: None,
    };
    let loader = GlmLoader { arenas: Some(std::rc::Rc::new(arenas)), ..loader };
    let source = catalog.tensor(cfg.embedding())?;
    let source = EmbedSource::new(root.path(), &source.shard, &source.metadata, cfg.hidden)?;
    let (embedding, (weights, peers)) = TokenEmbedding::load_with(&library, source, placement, || {
        loader.model(&cfg, cfg.layers, owners, false)
    }, |_, bytes| loader.allocate_owner("embedding", bytes))?;
    let devices = CoordinatorDevices::new(&[0, 1])?;
    let mut inventory = crate::shared::memory::receipts::Inventory::new(
        devices,
        "glm.weights",
        Lifetime::RESIDENT,
        Category::Weights,
        "target weights",
    );
    for (rank, layers) in std::iter::once(&weights.layers)
        .chain(peers.iter())
        .enumerate()
    {
        for (layer, weights) in layers.iter().enumerate() {
            for (name, buffer) in weights.allocation_buffers() {
                inventory.device(&format!("gpu.{rank}.layer.{layer}.{name}"), buffer)?;
            }
        }
    }
    inventory.device("gpu.0.norm", weights.norm.buffer)?;
    match &weights.head {
        GlmHead::Bf16(head) => inventory.device("gpu.0.head", head.buffer)?,
        GlmHead::K6(head) => {
            for (name, buffer) in [
                ("payload", head.payload.buffer),
                ("suh", head.suh.buffer),
                ("svh", head.svh.buffer),
            ] {
                inventory.device(&format!("gpu.0.core.head.{name}"), buffer)?;
            }
        }
        _ => anyhow::bail!("unexpected native vocabulary"),
    }
    let mut physical = crate::shared::memory::receipts::Inventory::new(
        devices, "glm.target", Lifetime::RESIDENT, Category::Weights, "target arena fixture");
    for (rank, layers) in std::iter::once(&weights.layers).chain(peers.iter()).enumerate() {
        for (layer, weights) in layers.iter().enumerate() {
            weights.allocation_receipts(&mut physical, &format!("rank.{rank}.layer.{layer}"))?;
        }
    }
    weights.norm.inventory(&mut physical, "norm")?;
    match &weights.head {
        GlmHead::Bf16(head) => head.inventory(&mut physical, "head")?,
        GlmHead::K6(head) => {
            head.payload.inventory(&mut physical, "head.payload")?;
            head.suh.inventory(&mut physical, "head.suh")?;
            head.svh.inventory(&mut physical, "head.svh")?;
        }
        _ => anyhow::bail!("unexpected native vocabulary"),
    }
    let backings = physical.receipts.iter().filter(|r| r.allocation.view.is_none()).collect::<Vec<_>>();
    assert_eq!(backings.len(), 2);
    for arena in layout_inventory.coordinator_arenas()? {
        let backing = backings.iter().find(|r| r.allocation.id == arena.id()).context("missing target arena")?;
        assert_eq!(backing.allocation.bytes, arena.bytes() as u64);
    }
    assert!(physical.receipts.iter().filter(|r| r.allocation.view.is_some()).all(|r|
        r.allocation.view.as_ref().unwrap().offset % 256 == 0));
    let mut actual = inventory
        .receipts
        .into_iter()
        .map(|r| r.allocation)
        .collect::<Vec<_>>();
    let embedding_receipts = embedding.allocation_receipts(devices, "glm.fixture.embedding")?;
    let mut table = embedding_receipts.iter().find(|r|
        r.allocation.id == "glm.fixture.embedding.table").context("embedding table view")?.allocation.clone();
    table.id = "glm.weights.embedding".into();
    table.category = Category::Embedding;
    actual.push(table);
    for arena in layout_inventory.coordinator_arenas()? {
        if placement == EmbedPlacement::Gpu && arena.id().ends_with("/0") {
            let backing = embedding_receipts.iter().find(|r| r.allocation.id == arena.id())
                .context("embedding lost target backing")?;
            assert_eq!(backing.allocation.bytes, arena.bytes() as u64);
        }
    }
    let expected = WeightInventory::from_checkpoint(
        &checkpoint,
        2,
        cfg.layers,
        owners,
        bf16_programs(),
        false,
        if placement == EmbedPlacement::Host {
            EmbeddingHome::HostPinned
        } else {
            EmbeddingHome::Device(0)
        },
    )?;
    assert_eq!(
        signatures(expected.owners.into_iter().map(|o| o.allocation)),
        signatures(actual)
    );
    Ok(())
}

#[test]
fn ml4_alpha_records_without_reads_and_decodes_only_after_group_commit() -> Result<()> {
    use crate::shared::experts::rtx::{dense, push};
    use std::os::unix::fs::FileExt;
    let (fixture, library) = compact_dso()?;
    fixture.immediate_copies(&library)?;
    let root = tempfile::tempdir_in(fixture.directory())?;
    let (config, mut tensors) = fixture::glm5_compact_weights();
    let mut cfg = GlmDsaConfig::from_hf(&config)?;
    cfg.variant = GlmDsaConfig::from_hf(&fixture::mistral_large4_config())?.variant;
    tensors.push(fixture::t("lerope.alpha", "BF16", &[cfg.qk_rope_head_dim / 2]));
    fixture::write_snapshot(root.path(), &config, &tensors, None);
    let catalog = cuteafd_loader::read_expert_catalog(root.path())?;
    let tensor = catalog.tensor("lerope.alpha")?;
    let bits = (0..cfg.qk_rope_head_dim / 2).map(|i|
        [0x0000u16, 0x8000, 0x3f80, 0xbf80, 0x0001, 0x7f80, 0x7fc1][i % 7]).collect::<Vec<_>>();
    let bytes = bits.iter().flat_map(|value| value.to_le_bytes()).collect::<Vec<_>>();
    std::fs::File::options().write(true).open(root.path().join(&tensor.shard))?
        .write_all_at(&bytes, tensor.metadata.byte_offset)?;
    let loader = GlmLoader { library: &library, catalog: &catalog, device: 0,
        stream: std::ptr::null_mut(), peers: vec![], arenas: None,
        allocation_prefix: String::new(), push: None };
    let mount = loader.rope_alpha(&cfg)?;
    let timing = crate::shared::memory::load_timing::Split::start();
    let (pushed, records) = dense::recording(|| loader.rope_alpha(&cfg))?;
    assert_eq!(timing.read_bytes(), 0, "alpha planning must not mount-read its payload");
    assert!(pushed.values().unwrap_err().to_string().contains("before load-group commit"));
    assert_eq!(records.host_operands.len(), 1);
    assert!(records.operands.is_empty() && records.steps.is_empty());
    let sockets = tempfile::tempdir_in(fixture.directory())?;
    let device = Device { library: &library, id: 0 };
    device.run(|| push::tests::receive_pushed(device, None, root.path(), sockets.path(), &records, None, 0, 0))?;
    let actual = pushed.values()?.iter().map(|value| value.to_bits()).collect::<Vec<_>>();
    assert_eq!(actual, bits.iter().map(|&value| u32::from(value) << 16).collect::<Vec<_>>());
    assert_eq!(actual, mount.values()?.iter().map(|value| value.to_bits()).collect::<Vec<_>>());
    Ok(())
}

#[test]
fn rotary_tables_are_exact_target_arena_views_with_no_new_device_allocation() -> Result<()> {
    let (fixture, library) = compact_dso()?;
    fixture.immediate_copies(&library)?;
    for ml4 in [false, true] {
        let root = tempfile::tempdir_in(fixture.directory())?;
        let (config, mut tensors) = fixture::glm5_compact_weights();
        let mut cfg = GlmDsaConfig::from_hf(&config)?;
        if ml4 {
            cfg.variant = GlmDsaConfig::from_hf(&fixture::mistral_large4_config())?.variant;
            tensors.push(fixture::t("lerope.alpha", "BF16", &[cfg.qk_rope_head_dim / 2]));
        }
        fixture::write_snapshot(root.path(), &config, &tensors, None);
        let catalog = cuteafd_loader::read_expert_catalog(root.path())?;
        if ml4 {
            use std::os::unix::fs::FileExt;
            let tensor = catalog.tensor("lerope.alpha")?;
            let alpha = (0..cfg.qk_rope_head_dim / 2).flat_map(|i|
                (0x3f00u16 + i as u16).to_le_bytes()).collect::<Vec<_>>();
            std::fs::File::options().write(true).open(root.path().join(&tensor.shard))?
                .write_all_at(&alpha, tensor.metadata.byte_offset)?;
        }
        let checkpoint = cuteafd_loader::plan::Checkpoint::open(root.path())?;
        for ranks in [1, 2] {
            let mut inventory = WeightInventory::from_checkpoint(&checkpoint, ranks, cfg.layers, None,
                false, false, EmbeddingHome::Device(0))?;
            let bytes = inventory.add_rotary(64, cfg.qk_rope_head_dim)? as usize;
            inventory.add_rotary(64, cfg.qk_rope_head_dim)?;
            assert!(inventory.add_rotary(65, cfg.qk_rope_head_dim).is_err());
            let components = inventory.components(1)?;
            for rank in 0..ranks {
                assert!(components.iter().any(|component| matches!(&component.kind,
                    cuteafd_loader::placement::planner::ComponentKind::Allocation { template, .. }
                        if template.id == format!("glm.weights.gpu.{rank}.rope")
                            && template.bytes == bytes as u64 && template.view.as_ref()
                                .is_some_and(|view| view.backing == format!("glm/coordinator/{rank}")))));
            }
            let arenas = inventory.coordinator_arenas()?.into_iter().enumerate().map(|(rank, layout)| {
                Ok((rank as i32, crate::shared::memory::weight_arena::DeviceWeightArena::new(
                    Device { library: &library, id: rank as i32 }, layout, crate::shared::memory::TEST_ARENA)?))
            }).collect::<Result<BTreeMap<_, _>>>()?;
            let loader = GlmLoader { library: &library, catalog: &catalog, device: 0,
                stream: std::ptr::null_mut(), peers: vec![], arenas: Some(std::rc::Rc::new(arenas)),
                allocation_prefix: String::new(), push: None };
            let alpha = ml4.then(|| loader.rope_alpha(&cfg)).transpose()?;
            let active = 37;
            let expected = if let Some(variant) = cfg.variant.ml4() {
                variant.rope_table(&alpha.as_ref().unwrap().values()?, active)?
            } else {
                let dim = cfg.qk_rope_head_dim;
                let mut table = vec![0f32; active * dim];
                let inv = (0..dim / 2).map(|i|
                    1.0 / (cfg.rope_theta as f32).powf((2 * i) as f32 / dim as f32)).collect::<Vec<_>>();
                for position in 0..active {
                    for (i, frequency) in inv.iter().enumerate() {
                        let angle = position as f32 * frequency;
                        table[position * dim + i] = angle.cos();
                        table[position * dim + dim / 2 + i] = angle.sin();
                    }
                }
                table
            };
            fixture.quiet_events(&library, false)?;
            let before = fixture.events()?.matches('D').count();
            for rank in 0..ranks {
                Device { library: &library, id: rank as i32 }.run(|| -> Result<()> {
                    let rotary = loader.allocate("rope", bytes)?;
                    engine::fill_rotary(&library, &cfg, alpha.as_ref(), active, &rotary)?;
                    assert_eq!(rotary.buffer.bytes, bytes, "retain full admitted extent");
                    let mut actual = vec![0; expected.len() * 4];
                    library.copy_d2h(&mut actual, rotary.buffer)?;
                    assert_eq!(actual, expected.iter().flat_map(|v| v.to_ne_bytes()).collect::<Vec<_>>());
                    assert!(engine::fill_rotary(&library, &cfg, alpha.as_ref(), 65, &rotary).is_err());
                    Ok(())
                })?;
            }
            assert_eq!(fixture.events()?.matches('D').count(), before, "rotary views must not allocate");
        }
    }
    Ok(())
}

#[test]
fn peer_routers_use_one_target_backing_and_replay_identical_push_bytes() -> Result<()> {
    use crate::shared::experts::rtx::{dense, push};
    use std::os::unix::fs::FileExt;
    let (fixture, library) = compact_dso()?;
    fixture.immediate_copies(&library)?;
    let root = tempfile::tempdir_in(fixture.directory())?;
    let (config, tensors) = fixture::glm5_compact_weights();
    fixture::write_snapshot(root.path(), &config, &tensors, None);
    let catalog = cuteafd_loader::read_expert_catalog(root.path())?;
    let cfg = GlmDsaConfig::read(root.path())?;
    let checkpoint = cuteafd_loader::plan::Checkpoint::open(root.path())?;
    let range = cfg.first_moe_layer..cfg.layers;
    let mut inventory = WeightInventory::from_checkpoint(&checkpoint, 2, cfg.layers, None,
        false, false, EmbeddingHome::Device(0))?;
    inventory.add_peer_routers(&range.clone().collect::<Vec<_>>())?;
    let components = inventory.components(1)?;
    let peer_arena = inventory.coordinator_arenas()?.remove(1);
    assert!(components.iter().any(|component| matches!(&component.kind,
        cuteafd_loader::placement::planner::ComponentKind::Allocation { template, .. }
            if template.id == "glm.weights.gpu.1.layer.1.operand.gate"
                && template.view.as_ref().is_some_and(|view| view.backing == peer_arena.id()))));
    let arenas = inventory.coordinator_arenas()?.into_iter().enumerate().map(|(rank, layout)| {
        Ok((rank as i32, crate::shared::memory::weight_arena::DeviceWeightArena::new(
            Device { library: &library, id: rank as i32 }, layout, crate::shared::memory::TEST_ARENA)?))
    }).collect::<Result<BTreeMap<_, _>>>()?;
    for index in range.clone() {
        for (_, name) in super::peer_router_names(&cfg, index) {
            let tensor = catalog.tensor(&name)?;
            let bytes = (0..tensor.metadata.byte_length).map(|i| (i as u8).wrapping_mul(17).wrapping_add(3)).collect::<Vec<_>>();
            std::fs::File::options().write(true).open(root.path().join(&tensor.shard))?
                .write_all_at(&bytes, tensor.metadata.byte_offset)?;
        }
    }
    let sink = std::rc::Rc::new(super::landing::PushSink::default());
    let loader = GlmLoader { library: &library, catalog: &catalog, device: 0,
        stream: std::ptr::null_mut(), peers: vec![crate::shared::peer_split::RankDevice {
            device: 1, stream: std::ptr::null_mut() }], arenas: Some(std::rc::Rc::new(arenas)),
        allocation_prefix: String::new(), push: Some(sink.clone()) };
    let (_, mut peers) = loader.model(&cfg, cfg.layers, None, false)?;
    let _ = sink.take();
    fixture.quiet_events(&library, false)?;
    let before = fixture.events()?.matches('D').count();
    let timing = crate::shared::memory::load_timing::Split::start();
    let ((), mut records) = dense::recording(|| loader.peer_routers(&cfg, &mut peers[0], range.clone()))?;
    records.extend(sink.take().0);
    assert_eq!(timing.read_bytes(), 0, "router recording must not mount-read payloads");
    assert_eq!(fixture.events()?.matches('D').count(), before, "router replicas are arena views");
    let sockets = tempfile::tempdir_in(fixture.directory())?;
    let device = Device { library: &library, id: 1 };
    device.run(|| push::tests::receive_pushed(device, None, root.path(), sockets.path(), &records, None, 0, 1))?;
    for index in range {
        for (key, name) in super::peer_router_names(&cfg, index) {
            let land = super::landing::Land::whole(&catalog, &name, 0)?;
            let expected = super::landing::read(&catalog, &[land.clone()], land.bytes())?;
            let buffer = peers[0][index].allocation_buffers().into_iter()
                .find(|(name, _)| name == &format!("operand.{key}"))
                .context("peer router allocation")?.1;
            let mut actual = vec![0; expected.len()];
            library.copy_d2h(&mut actual, buffer)?;
            assert_eq!(actual, expected, "peer router {name}");
        }
    }
    Ok(())
}

#[test]
fn compact_k6_payload_vectors_and_head_equal_inventory() -> Result<()> {
    compact_real_load(None, true, EmbedPlacement::Gpu)
}

#[test]
fn glm_and_ml4_k6_min_max_push_plans_preserve_packed_bytes() -> Result<()> {
    use crate::shared::experts::rtx::dense::{self, Job, RankLayout};
    use cuteafd_loader::formats::{dense_trellis::TrellisSide, trellis_core::CoreWeightLayout as SharedLayout};
    use std::os::unix::fs::FileExt;
    let (dso, library) = compact_dso()?;
    for ml4 in [false, true] {
        let root = tempfile::tempdir_in(dso.directory())?;
        let (config, mut tensors) = fixture::glm5_compact_weights();
        let mut cfg = GlmDsaConfig::from_hf(&config)?;
        cfg.vocab_size = 256;
        if ml4 {
            cfg.variant = GlmDsaConfig::from_hf(&fixture::mistral_large4_config())?.variant;
            cfg.q_lora_rank = 0;
        }
        for (name, shape) in cuteafd_loader::families::glm5::core::expected(&cfg) {
            let stem = name.strip_suffix(".weight").context("core stem")?;
            tensors.retain(|t| t.0 != name && t.0 != format!("{name}_scale_inv"));
            tensors.extend(fixture::exl3(stem, shape[0].next_multiple_of(128), shape[1], 6));
        }
        fixture::write_snapshot(root.path(), &config, &tensors, None);
        let checkpoint = cuteafd_loader::plan::Checkpoint::open(root.path())?;
        for tensor in &checkpoint.tensors {
            if tensor.meta.name.ends_with(".mcg") || tensor.meta.name.ends_with(".trellis")
                || tensor.meta.name.ends_with(".suh") || tensor.meta.name.ends_with(".svh") {
                let bytes = if tensor.meta.name.ends_with(".mcg") { 0xcbac1fedu32.to_le_bytes().to_vec() }
                    else { (0..tensor.meta.byte_length).map(|i| (i as u8).wrapping_mul(31).wrapping_add(7)).collect() };
                std::fs::OpenOptions::new().write(true).open(root.path().join(&tensor.shard))?
                    .write_all_at(&bytes, tensor.meta.byte_offset)?;
            }
        }
        let catalog = cuteafd_loader::read_expert_catalog(root.path())?;
        let core = CoreInventory::from_checkpoint(&cfg, &checkpoint)?;
        for ranks in [1, 2] {
            let sink = std::rc::Rc::new(super::landing::PushSink::default());
            let loader = GlmLoader { library: &library, catalog: &catalog, device: 0,
                stream: std::ptr::null_mut(), peers: (1..ranks).map(|rank| crate::shared::peer_split::RankDevice {
                    device: rank as i32, stream: std::ptr::null_mut() }).collect(), arenas: None, allocation_prefix: String::new(), push: Some(sink.clone()) };
            let mut layouts = Vec::new();
            for layer in 0..cfg.layers {
                for (rank, weights) in core.layer_weights(&cfg, layer, ranks, None)?.into_iter().enumerate() {
                    layouts.extend(weights.into_values().map(|layout| (rank, layout)));
                }
            }
            layouts.push((0, SharedLayout::new(core.get_weight(cfg.head()).context("K6 head")?, TrellisSide::Output, 0, 1)?));
            let arenas = (0..ranks).map(|rank| {
                let mut tensors = Vec::new();
                for (_, layout) in layouts.iter().filter(|(owner, _)| *owner == rank) {
                    let [n, k] = layout.shape;
                    for (suffix, bytes) in [("payload", n * k * layout.bits / 8), ("suh", k * 2), ("svh", n * 2)] {
                        tensors.push((format!("glm.weights.gpu.{rank}.{}.{suffix}", layout.stem), bytes));
                    }
                }
                let layout = cuteafd_loader::serving_capacity::WeightArenaLayout::new(
                    format!("glm/coordinator/{rank}"), tensors)?;
                Ok((rank as i32, crate::shared::memory::weight_arena::DeviceWeightArena::new(
                    Device { library: &library, id: rank as i32 }, layout, crate::shared::memory::TEST_ARENA)?))
            }).collect::<Result<BTreeMap<_, _>>>()?;
            let loader = GlmLoader { arenas: Some(std::rc::Rc::new(arenas)), ..loader };
            let mut weights = Vec::new();
            let mut expected = Vec::new();
            for (rank, layout) in &layouts {
                let packed = CoreWeightLayout::from_shared(layout.clone()).read(|name|
                    crate::shared::checkpoint_read::read_tensor(&checkpoint, name))?;
                expected.push(packed.shard);
                weights.push(loader.core_weights(layout.clone(), &checkpoint, *rank, &layout.stem)?);
            }
            let (records, deferred) = sink.take();
            assert!(deferred.is_empty());
            assert_eq!(records.operands.len(), layouts.len() * 2);
            assert_eq!(records.steps.iter().flat_map(|s| &s.jobs).filter(|j| matches!(j, Job::TrellisTiles { .. })).count(), layouts.len());
            // Building the push ranks must retain both payload transforms and vector lands.
            for rank in 0..ranks {
                let layout = RankLayout::new(records.operands.clone(), records.steps.clone(), rank as i32)?;
                let checked = layout.checked(root.path(), ranks + 6, rank + 6)?;
                assert!(checked.totals.rank_bytes[rank + 6] > 0);
                assert!(!layout.steps.is_empty() && !layout.operands.is_empty());
                assert!(layout.host_staging.bytes()? > 0);
            }
            // Replay the actual recorded recipes through the mount fallback. This also
            // catches an eager upload: the original packer is the independent oracle.
            dense::fill_records_by_mount(&library, &records, |_| std::ptr::null_mut())?;
            for ((weights, expected), (_, layout)) in weights.iter().zip(&expected).zip(&layouts) {
                assert_eq!(weights.payload.buffer.bytes + weights.suh.buffer.bytes + weights.svh.buffer.bytes, layout.bytes);
                for (buffer, bytes) in [(weights.payload.buffer, &expected.payload),
                    (weights.suh.buffer, &expected.suh), (weights.svh.buffer, &expected.svh)] {
                    let mut actual = vec![0; buffer.bytes];
                    library.copy_d2h(&mut actual, buffer)?;
                    assert_eq!(&actual, bytes, "{} ranks={ranks} ml4={ml4}", layout.stem);
                }
            }
        }
    }
    Ok(())
}

#[test]
fn mixed_mlp_arithmetic_is_refused_before_upload() -> Result<()> {
    let root = tempfile::tempdir()?;
    let (mut config, tensors) = fixture::glm5_compact_weights();
    config["quantization_config"] = serde_json::json!({"quant_method":"fp8", "producer":{"name":"modelopt"},
        "quant_algo":"MIXED_PRECISION", "quantized_layers":{
            "model.layers.0.mlp.up_proj":{"quant_algo":"FP8"}
    }});
    fixture::write_snapshot(root.path(), &config, &tensors, None);
    let checkpoint = cuteafd_loader::plan::Checkpoint::open(root.path())?;
    let catalog = cuteafd_loader::read_expert_catalog(root.path())?;
    let cfg = GlmDsaConfig::read(root.path())?;
    let core = CoreInventory::from_checkpoint(&cfg, &checkpoint)?;
    let mut published = 0;
    let result = routes::layer_routes(
        &catalog,
        &cfg,
        0,
        &checkpoint,
        &core,
        2,
        None,
        false,
        false,
        |_| {
            published += 1;
            Ok(())
        },
    );
    assert!(
        result
            .unwrap_err()
            .to_string()
            .contains("mixed declared MLP arithmetic")
    );
    assert_eq!(published, 0);
    Ok(())
}

#[test]
fn ml4_non_native_index_query_is_refused_before_upload() -> Result<()> {
    let root = tempfile::tempdir()?;
    let (mut config, tensors) = fixture::glm5_compact_weights();
    // Native contract stays identical; the variant selects ML4's no-conversion rule.
    let official = installed(
        "mistralai--Mistral-Large-4-1T-A52B",
        "f12dabdc0c1e99e7f7924461a6f308859b05af4d",
    )?;
    let ml4 = GlmDsaConfig::read(&official)?.variant;
    config["vocab_size"] = serde_json::json!(64);
    let mut cfg = GlmDsaConfig::from_hf(&config)?;
    cfg.variant = ml4;
    let glm = GlmDsaConfig::from_hf(&config)?;
    let mut renamed = Vec::new();
    for layer in 0..cfg.layers {
        for (key, old) in glm.plain_owners(layer) {
            if let Some((_, new)) = cfg.plain_owners(layer).into_iter().find(|(k, _)| *k == key) {
                if let Some((_, dtype, shape)) = tensors.iter().find(|t| t.0 == old) {
                    renamed.push(fixture::t(new, dtype, shape));
                }
            }
        }
        for op in [Op::QueryScaleS, Op::QueryScaleB] {
            renamed.push(fixture::t(
                cfg.weight(layer, op).unwrap(),
                "BF16",
                &[cfg.heads],
            ));
        }
        for op in [Op::KvA, Op::QB, Op::O] {
            let old = glm.weight(layer, op).unwrap();
            let shape = &tensors.iter().find(|t| t.0 == old).unwrap().2;
            renamed.extend(fixture::exl3(
                &cfg.stem(layer, op).unwrap(),
                shape[0].next_multiple_of(128),
                shape[1],
                6,
            ));
        }
        let kv = cfg.weight(layer, Op::KvB).unwrap();
        renamed.push(fixture::t(
            kv,
            "BF16",
            &[
                cfg.heads * (cfg.nope_dim(layer) + cfg.v_head_dim),
                cfg.kv_lora_rank,
            ],
        ));
        if cfg.indexers[layer] == GlmIndexer::Full {
            renamed.push(fixture::t(
                cfg.weight(layer, Op::IndexQ).unwrap(),
                "BF16",
                &[256, 256],
            ));
        }
    }
    // Inventory validates core shapes from config, so construct its complete core
    // descriptors with the same config rather than admitting an incomplete group.
    for (name, shape) in cuteafd_loader::families::glm5::core::expected(&cfg) {
        if !name.contains("indexer") && name != "output.weight" {
            let stem = name.strip_suffix(".weight").unwrap();
            renamed.retain(|t| !t.0.starts_with(&format!("{stem}.")));
            renamed.extend(fixture::exl3(
                stem,
                shape[0].next_multiple_of(128),
                shape[1],
                6,
            ));
        }
    }
    // The native core object is independent of the catalog's missing future routes.
    renamed.extend(tensors);
    fixture::write_snapshot(root.path(), &config, &renamed, None);
    let checkpoint = cuteafd_loader::plan::Checkpoint::open(root.path())?;
    let catalog = cuteafd_loader::read_expert_catalog(root.path())?;
    let core = CoreInventory::from_checkpoint(&cfg, &checkpoint)?;
    let result = routes::layer_routes(
        &catalog,
        &cfg,
        0,
        &checkpoint,
        &core,
        1,
        None,
        false,
        false,
        |_| Ok(()),
    );
    assert!(
        result
            .unwrap_err()
            .to_string()
            .contains("shipped FP8 block-128 form only")
    );
    Ok(())
}

#[test]
fn compact_ml4_query_vectors_and_native_index_equal_real_layer_requests() -> Result<()> {
    let (dso, library) = compact_dso()?;
    let root = tempfile::tempdir_in(dso.directory())?;
    let (config, mut tensors) = fixture::glm5_compact_weights();
    let glm = GlmDsaConfig::from_hf(&config)?;
    let mut cfg = glm.clone();
    cfg.variant = GlmDsaConfig::from_hf(&fixture::mistral_large4_config())?.variant;
    for layer in 0..cfg.layers {
        for (key, name) in cfg.plain_owners(layer) {
            let old = glm
                .plain_owners(layer)
                .into_iter()
                .find(|(k, _)| *k == key)
                .map(|(_, n)| n);
            let shape = old
                .and_then(|old| tensors.iter().find(|t| t.0 == old).map(|t| t.2.clone()))
                .unwrap_or_else(|| {
                    vec![if key.starts_with("query_scale") {
                        cfg.heads
                    } else if key.starts_with("index_scale") {
                        cfg.index_heads
                    } else {
                        cfg.hidden
                    }]
                });
            tensors.push(fixture::t(name, "BF16", &shape));
        }
        for op in [
            Op::Gate,
            Op::Up,
            Op::Down,
            Op::IndexK,
            Op::IndexWeights,
            Op::Router,
        ] {
            let (Some(old), Some(new)) = (glm.weight(layer, op), cfg.weight(layer, op)) else {
                continue;
            };
            let copies = tensors
                .iter()
                .filter(|t| t.0 == old || t.0 == format!("{old}_scale_inv"))
                .map(|(name, dtype, shape)| (name.replacen(&old, &new, 1), *dtype, shape.clone()))
                .collect::<Vec<_>>();
            tensors.extend(copies);
        }
        tensors.push(fixture::t(
            cfg.weight(layer, Op::KvB).unwrap(),
            "BF16",
            &[
                cfg.heads * (cfg.nope_dim(layer) + cfg.v_head_dim),
                cfg.kv_lora_rank,
            ],
        ));
        if cfg.indexers[layer] == GlmIndexer::Full {
            tensors.extend(fixture::fp8(
                &cfg.stem(layer, Op::IndexQ).unwrap(),
                cfg.index_heads * cfg.index_head_dim,
                cfg.hidden,
                None,
            ));
        }
    }
    for (name, shape) in cuteafd_loader::families::glm5::core::expected(&cfg) {
        if name.contains(".attention.") && !name.contains("indexer") {
            tensors.extend(fixture::exl3(
                name.strip_suffix(".weight").unwrap(),
                shape[0].next_multiple_of(128),
                shape[1],
                6,
            ));
        }
    }
    fixture::write_snapshot(root.path(), &config, &tensors, None);
    let checkpoint = cuteafd_loader::plan::Checkpoint::open(root.path())?;
    use std::os::unix::fs::FileExt;
    for tensor in &checkpoint.tensors {
        if tensor.meta.name.ends_with(".mcg") {
            std::fs::OpenOptions::new()
                .write(true)
                .open(root.path().join(&tensor.shard))?
                .write_all_at(&0xcbac1fedu32.to_le_bytes(), tensor.meta.byte_offset)?;
        }
    }
    let catalog = cuteafd_loader::read_expert_catalog(root.path())?;
    let core = CoreInventory::from_checkpoint(&cfg, &checkpoint)?;
    let loader = GlmLoader {
        library: &library,
        catalog: &catalog,
        stream: std::ptr::null_mut(),
        device: 0,
        peers: vec![crate::shared::peer_split::RankDevice {
            device: 1,
            stream: std::ptr::null_mut(),
        }],
        arenas: None, allocation_prefix: String::new(), push: None,
    };
    let mut requests = Vec::new();
    routes::layer_routes(
        &catalog,
        &cfg,
        0,
        &checkpoint,
        &core,
        2,
        None,
        false,
        bf16_programs(),
        |r| {
            requests.extend(r.requests);
            Ok(())
        },
    )?;
    let arenas = (0..2).map(|rank| {
        let tensors = requests.iter().filter(|request| request.location ==
            cuteafd_loader::placement::planner::Location::coordinator(rank as u8))
            .map(|request| (request.id.clone(), request.bytes as usize));
        let layout = cuteafd_loader::serving_capacity::WeightArenaLayout::new(
            format!("glm/coordinator/{rank}"), tensors)?;
        Ok((rank, crate::shared::memory::weight_arena::DeviceWeightArena::new(
            Device { library: &library, id: rank }, layout, crate::shared::memory::TEST_ARENA)?))
    }).collect::<Result<BTreeMap<_, _>>>()?;
    let loader = GlmLoader { arenas: Some(std::rc::Rc::new(arenas)), ..loader };
    let layers = loader.layer(&cfg, 0, &checkpoint, &core, None, false)?;
    let mut actual = crate::shared::memory::receipts::Inventory::new(
        CoordinatorDevices::new(&[0, 1])?,
        "glm.weights",
        Lifetime::RESIDENT,
        Category::Weights,
        "target weights",
    );
    for (rank, layer) in layers.iter().enumerate() {
        for (name, buffer) in layer.allocation_buffers() {
            actual.device(&format!("gpu.{rank}.layer.0.{name}"), buffer)?;
        }
    }
    assert_eq!(
        signatures(requests),
        signatures(actual.receipts.into_iter().map(|r| r.allocation))
    );
    Ok(())
}

#[test]
fn installed_native_dense_shared_and_vocabulary_equal_reader_requests() -> Result<()> {
    use crate::shared::experts::exlr8::{Exlr8Weights, dense_weights};
    use cuteafd_loader::placement::planner::Location;
    use cuteafd_loader::plan::exlr8::{
        dense::{DensePlan, DenseRecord, DenseSelection},
        residency::{ExpertSet, MetadataInput},
    };
    let root = installed(
        "kindlingai--glm-5.3-exlr8-k2-k3-k4",
        "a2b599e06c59f2ccb01aaa44e6f182ba21e39f80",
    )?;
    let cfg = GlmDsaConfig::read(&root)?;
    let config = serde_json::from_slice(&std::fs::read(root.join("config.json"))?)?;
    let selection = DenseSelection::from_hf(&config)?;
    let records = DenseRecord::inventory(&root)?;
    let shared = MetadataInput::open(&root, "up1")?;
    for world in [1, 2] {
        let dense = (0..world)
            .map(|rank| {
                cuteafd_loader::plan::exlr8::dense::requests::NativeDenseRecords::new(
                    &root,
                    cfg.layers,
                    world,
                    rank,
                    &BTreeMap::new(),
                )
            })
            .collect::<Result<Vec<_>>>()?;
        let mut expected = WeightInventory::from_native(
            &selection,
            cfg.layers,
            &dense.iter().map(|r| r.plan.clone()).collect::<Vec<_>>(),
        )?;
        let mut actual = Vec::new();
        let mut add = |rank: usize, name: String, bytes: usize| {
            actual.push(Allocation::buffer(
                format!("glm.weights.gpu.{rank}.{name}"),
                Location::coordinator(rank as u8),
                bytes as u64,
                Lifetime::RESIDENT,
                if name.starts_with("vocabulary.embedding.") {
                    Category::Embedding
                } else {
                    Category::Weights
                },
                "target weights",
            ))
        };
        for (rank, records) in dense.iter().enumerate() {
            for (layer, record) in records.layers.iter().enumerate() {
                dense_weights::layer_requests(record, &[], |request| {
                    for (name, bytes) in request.owners {
                        let key = GlmLoader::native_layer_owner(&cfg, layer, &format!("layers.{layer}.{name}"))?;
                        add(rank, key, bytes);
                    }
                    Ok(())
                })?;
            }
            if rank == 0 {
                add(
                    rank,
                    "norm".into(),
                    cuteafd_loader::plan::exlr8::dense::DenseTransform::Identity
                        .allocation_bytes(&records.norm)?,
                );
            }
        }
        let selected = records
            .iter()
            .filter(|record| selection.vocabulary_names().contains(&record.tensor))
            .cloned()
            .collect::<Vec<_>>();
        let vocab = (0..world)
            .map(|rank| DensePlan::for_rank(&selected, world, rank).map_err(anyhow::Error::from))
            .collect::<Result<Vec<_>>>()?;
        expected.extend_native(&selection, cfg.layers, &vocab.iter().collect::<Vec<_>>())?;
        for (rank, plan) in vocab.iter().enumerate() {
            for slice in &plan.tensors {
                let table = if slice.tensor == cfg.embedding() {
                    "embedding"
                } else {
                    "head"
                };
                for (suffix, bytes) in cuteafd_loader::plan::exlr8::dense::requests::row_requests(slice)? {
                    add(rank, format!("vocabulary.{table}.{suffix}"), bytes);
                }
            }
        }
        actual.push(EmbedSource::placeholder_request(
            "glm.weights.embedding",
            0,
        )?);
        let plans = (0..world)
            .map(|rank| {
                (cfg.first_moe_layer..cfg.layers)
                    .map(|layer| {
                        shared
                            .plan(layer, world, rank, ExpertSet::Shared)
                            .map_err(anyhow::Error::from)
                    })
                    .collect::<Result<Vec<_>>>()
            })
            .collect::<Result<Vec<_>>>()?;
        expected.extend_shared(&plans)?;
        let before = expected.owners.len();
        assert!(expected.extend_shared(&plans).is_err());
        assert_eq!(expected.owners.len(), before, "duplicate append is atomic");
        if world == 2 {
            let mut peer = expected.clone();
            peer.add_peer_routers(&[cfg.first_moe_layer])?;
            for key in ["gate", "gate.bias"] {
                let source = expected.owners.iter().find(|o| {
                    o.allocation.id
                        == format!(
                            "glm.weights.gpu.0.layer.{}.operand.{key}",
                            cfg.first_moe_layer
                        )
                });
                if let Some(source) = source {
                    let target = peer
                        .owners
                        .iter()
                        .find(|o| {
                            o.allocation.id
                                == format!(
                                    "glm.weights.gpu.1.layer.{}.operand.{key}",
                                    cfg.first_moe_layer
                                )
                        })
                        .context("native peer router")?;
                    assert_eq!(source.allocation.bytes, target.allocation.bytes);
                }
            }
        }
        for plan in plans.iter().flatten() {
            actual.push(Exlr8Weights::shared_request(plan)?);
        }
        assert_eq!(
            signatures(expected.owners.into_iter().map(|o| o.allocation)),
            signatures(actual),
            "native world={world}"
        );
    }
    Ok(())
}

#[test]
fn compact_native_shared_arenas_equal_requests() -> Result<()> {
    use crate::shared::experts::exlr8::Exlr8Weights;
    use cuteafd_loader::plan::exlr8::residency::{ExpertSet, Residency};
    let (dso, library) = compact_dso()?;
    let root = tempfile::tempdir_in(dso.directory())?;
    fixture::exlr8::fixture(root.path(), false);
    for world in [1, 2] {
        for rank in 0..world {
            let plan = Residency::plan(root.path(), "mixed", 3, world, rank, ExpertSet::Shared)?;
            let request = Exlr8Weights::shared_request(&plan)?;
            Device {
                library: &library,
                id: rank as i32,
            }
            .run(|| -> Result<()> {
                let weights = Exlr8Weights::load(&library, plan, 1 << 30)?;
                let devices = CoordinatorDevices::new(&[0, 1])?;
                let mut inventory = crate::shared::memory::receipts::Inventory::new(
                    devices,
                    &format!("glm.weights.gpu.{rank}.shared"),
                    Lifetime::RESIDENT,
                    Category::Weights,
                    "shared weights",
                );
                weights.allocation_receipts(&mut inventory, "layer.3")?;
                let actual = inventory
                    .receipts
                    .into_iter()
                    .find(|r| r.allocation.view.is_none())
                    .context("shared backing receipt")?;
                assert_eq!(signatures([request]), signatures([actual.allocation]));
                Ok(())
            })?;
        }
    }
    Ok(())
}
