//! Installed checkpoint/package allocations through the real serving constructors.
use super::*;
use anyhow::{ensure, Context, Result};
use cuteafd_core::{memory_layout::Category, ExpertGeometry};
use cuteafd_loader::{families::glm5::GlmDsaConfig, placement::{families::glm5::{self,
    local_physical::{LocalPhysicalInput, LocalPhysicalSource, LocalStorage}},
    planner::{CoordinatorDevices, Lifetime, Location}}, plan::{self, exlr8::{PrefillActivations,
    residency::{MetadataInput, ExpertSet}}}};
use crate::shared::{memory::{device::{Device, Stream}, receipts::Inventory,
    weight_arena::{ArenaLayout, LoadingArena, ReadyArena, ExpectedCoverage}}, peer_split::RankDevice,
    experts::{exlr8::Exlr8Weights, exl3::{Exl3Weights, execution::{Exl3Modules, Exl3ModulePaths}}}};
use std::{collections::BTreeMap, path::{Path, PathBuf}, rc::Rc};

#[test]
fn glm_expert_census_resides_weights_before_kv_only() -> Result<()> {
    use cuteafd_loader::placement::planner::{Allocation, Phase, PhaseOrder, Receipt, View};
    use crate::families::glm5::engine::resident_expert_weights;
    let owner = |id: &str| Receipt { allocation: Allocation::buffer(id, Location::coordinator(0), 64,
        Lifetime::TARGET, Category::Experts, "resident expert owners"), address: 0x10000 };
    let mut root = owner("experts.exl3.gpu.1.selection");
    root.allocation.category = Category::Weights;
    root.allocation.group = "weight arena".into();
    let mut alias = owner("glm.expert.routed.rank.0.weight.3.0");
    alias.allocation.view = Some(View { backing: root.allocation.id.clone(), offset: 0 });
    let push = owner("glm.expert.routed.rank.0.weight.7.0.backing");
    let shared = owner("glm.expert.shared.rank.1.layer.4.backing");
    let workspace = owner("glm.expert.routed.rank.0.execution.0.weight.3");
    let pair = owner("glm.expert.routed.routes.0");
    let mut receipts = vec![root, alias, push, shared, workspace, pair];
    let mut invalid = receipts.clone();
    invalid[1].allocation.view.as_mut().unwrap().backing = "missing".into();
    assert!(resident_expert_weights(&mut invalid).is_err());
    assert!(invalid.iter().all(|r| r.allocation.lifetime == Lifetime::TARGET), "failed census mutates lifetimes");
    invalid = receipts.clone();
    invalid[4].allocation.view = invalid[1].allocation.view.clone();
    assert!(resident_expert_weights(&mut invalid).is_err(), "workspace alias silently becomes a weight");
    resident_expert_weights(&mut receipts)?;
    for (index, receipt) in receipts.iter().enumerate() {
        let lifetime = receipt.allocation.lifetime;
        assert_eq!(lifetime, if index < 4 { Lifetime::RESIDENT } else { Lifetime::TARGET });
        assert_eq!(lifetime.contains(Phase::Load, PhaseOrder::TARGET_THEN_DRAFT), index < 4);
        assert!(lifetime.contains(Phase::Ready, PhaseOrder::TARGET_THEN_DRAFT));
    }
    resident_expert_weights(&mut receipts)?;
    Ok(())
}

fn installed_snapshot(repo: &str) -> Result<PathBuf> {
    let home = std::env::var_os("HF_HOME").map(PathBuf::from).unwrap_or_else(||
        PathBuf::from(std::env::var_os("HOME").expect("HOME")).join(".cache/huggingface"));
    let model = home.join("hub").join(format!("models--{}", repo.replace('/', "--")));
    let revision = std::fs::read_to_string(model.join("refs/main"))
        .with_context(|| format!("required installed checkpoint {repo} missing"))?;
    Ok(model.join("snapshots").join(revision.trim()))
}

fn ready<'a>(device: Device<'a>, sizes: Vec<usize>) -> Result<Rc<ReadyArena<'a>>> {
    let admitted = sizes.iter().try_fold(0usize, |sum, &bytes| sum.checked_add(bytes)
        .context("test arena size overflow"))? + sizes.len() * 256;
    Ok(Rc::new(LoadingArena::new(device, ArenaLayout::new(sizes, admitted)?,
        ExpectedCoverage::default())?.done()?.initialize(|_| Ok(()))?))
}

fn modules<'a>(fixture: &cuteafd_ffi::native_library_lifetime_fixture::Fixture,
    device: Device<'a>, package: &Path, rows: usize) -> Result<Vec<Exl3Modules<'a>>> {
    cuteafd_core::Exl3RowPolicy::Nearest.capacities(rows)?.into_iter().map(|capacity| device.run(|| {
        let directory = package.join(format!("m{capacity}"));
        let meta: serde_json::Value = serde_json::from_slice(&std::fs::read(directory.join("v41_exl3.json"))?)?;
        let value = |key: &str| -> Result<u32> { Ok(u32::try_from(meta[key].as_u64().with_context(|| format!("missing {key}"))?)?) };
        let bits = meta["bits"].as_array().context("missing EXL3 bits")?;
        let mut words = [0; 16];
        words[..7].copy_from_slice(&[meta["native_info_version"].as_u64().unwrap_or(2) as u32,
            value("hidden")?, value("intermediate")?, value("experts")?, value("capacity")?, value("top_k")?, bits.len() as u32]);
        for (index, (object, slots)) in [(0, "pointer_slots"), (0, "scalar_slots"),
            (1, "pointer_slots"), (1, "scalar_slots")].into_iter().enumerate() {
            words[7 + index] = u32::try_from(meta["objects"][object][slots].as_array()
                .with_context(|| format!("missing EXL3 object {object} {slots}"))?.len())?;
        }
        for (index, bit) in bits.iter().enumerate() { words[11 + index] = bit.as_u64().context("EXL3 bit")? as u32; }
        words[15] = 4;
        fixture.configure_exl3(device.library, &words)?;
        // SAFETY: only the private CPU DSO is loaded; installed assets/metadata remain unmodified.
        unsafe { Exl3Modules::load_from(device.library, &directory, crate::shared::experts::exl3::execution::Exl3InputFormat::Fp8K32,
            Exl3ModulePaths { core: fixture.module_path().into(), routes: Some(fixture.module_path().into()) }) }
    })).collect()
}

/// Default-tier qualification uses real header parsing, mount/push loaders,
/// execution workspaces and pair receipts, without installed checkpoints or CUDA.
#[test]
fn glm_exl3_local_physical_equals_serving() -> Result<()> {
    use cuteafd_loader::plan::testing::{exl3, exl3_compact, glm5_compact_weights, write_snapshot};
    use crate::shared::experts::layer::ExpertLayer;
    use sha2::{Digest, Sha256};
    const CHILD: &str = "CUTEAFD_GLM_ARENA_PHYSICAL_CHILD";
    if std::env::var_os(CHILD).is_none() {
        let output = std::process::Command::new(std::env::current_exe()?)
            .args(["shared::experts::rtx::local_physical_tests::glm_exl3_local_physical_equals_serving", "--exact", "--nocapture"])
            .env(CHILD, "1").output()?;
        ensure!(output.status.success() && String::from_utf8_lossy(&output.stdout).contains("1 passed; 0 failed"),
            "physical equality: {}\n{}", String::from_utf8_lossy(&output.stdout), String::from_utf8_lossy(&output.stderr));
        return Ok(());
    }
    let (mut config, _) = glm5_compact_weights();
    config["num_hidden_layers"] = 3.into();
    config["indexer_types"] = serde_json::json!(["full", "shared", "shared"]);
    config["mlp_layer_types"] = serde_json::json!(["dense", "sparse", "sparse"]);
    config["quantization_config"] = exl3_compact(4);
    let cfg = GlmDsaConfig::from_hf(&config)?;
    let geometry = ExpertGeometry { hidden: cfg.hidden as u32, experts: cfg.experts as u32,
        intermediate: cfg.moe_intermediate as u32, topk: cfg.topk as u32, layers: cfg.layers as u32 };
    cuteafd_core::set_expert_geometry(geometry).map_err(|g| anyhow::anyhow!("unexpected geometry {g:?}"))?;
    let root = tempfile::tempdir()?;
    let tensors = [1, 2].into_iter().flat_map(|layer| (0..cfg.experts).flat_map(move |expert|
        ["gate_proj", "up_proj", "down_proj"].into_iter().flat_map(move |projection|
            exl3(&format!("model.layers.{layer}.mlp.experts.{expert}.{projection}"), 256, 256, 3 + expert))))
        .collect::<Vec<_>>();
    write_snapshot(root.path(), &config, &tensors, None);
    let path = root.path().join("model-00001-of-00001.safetensors");
    let file = std::fs::OpenOptions::new().write(true).open(&path)?;
    use std::os::unix::fs::FileExt;
    let trellis_byte = |offset: usize| offset.wrapping_mul(17).wrapping_add(offset / 97) as u8;
    for tensor in cuteafd_loader::read_safetensors_metadata(&path)? {
        if tensor.name.ends_with(".mcg") { file.write_all_at(&0xcbac1fedu32.to_le_bytes(), tensor.byte_offset)?; }
        if tensor.name.ends_with(".trellis") {
            let bytes = (0..usize::try_from(tensor.byte_length)?).map(trellis_byte).collect::<Vec<_>>();
            file.write_all_at(&bytes, tensor.byte_offset)?;
        }
    }
    let catalog = cuteafd_loader::read_expert_catalog(root.path())?;
    let fixture = cuteafd_ffi::native_library_lifetime_fixture::Fixture::build_for_tests()?;
    let library = fixture.load()?;
    fixture.immediate_copies(&library)?;
    fixture.peer_abort_initialization(&library)?;
    fixture.strict_device_free(&library, true)?;
    for rank in 0..2 { fixture.configure_memory(&library, rank, 1usize << 30)?; }
    let lock: serde_json::Value = serde_json::from_str(include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"), "/../../../third_party/sparkinfer.lock.json")))?;
    for world in [1, 2] {
        let package = root.path().join(format!("tp{world}"));
        for capacity in cuteafd_core::Exl3RowPolicy::Nearest.capacities(256)? {
            let directory = package.join(format!("m{capacity}"));
            std::fs::create_dir_all(&directory)?;
            std::fs::write(directory.join("lut"), [0u8; 16])?;
            let meta = serde_json::json!({
                "schema":"cuteafd.v41-exl3-aot.v1", "sparkinfer_revision":lock["revision"],
                "hidden":cfg.hidden,"intermediate":cfg.moe_intermediate/world,"experts":cfg.experts,
                "top_k":cfg.topk,"capacity":capacity,"bits":[3,4],"output_dtype":"fp32",
                "swiglu_limit":geometry.swiglu_limit(),"direct":true,"sms":1,"blocks_per_sm":1,
                "decode_capacity":capacity <= 80,"tile":[64,128,64,128],
                "objects":[{"label":"v41_exl3_core","pointer_slots":["gate_suh_ptr"],"scalar_slots":["active_m"]},
                    {"label":"v41_exl3_sum","pointer_slots":["output_ptr"],"scalar_slots":["active_m"]}],
                "trellis_lut":{"file":"lut","bytes":16,"sha256":format!("{:x}",Sha256::digest([0u8;16]))},
                "buffers":{
                    "output":{"bytes":capacity as usize*cfg.hidden*4,"dtype":"torch.float32","allocation":"output","zero_on_create":false},
                    "fc2":{"bytes":capacity*32,"dtype":"torch.float32","allocation":"fc2","zero_on_create":false},
                    "state":{"bytes":16,"dtype":"torch.int32","allocation":"state","zero_on_create":true}}
            });
            std::fs::write(directory.join("v41_exl3.json"), serde_json::to_vec(&meta)?)?;
        }
        let local = glm5::LocalInventory { package: package.clone(), format: glm5::GlmLocalFormat::Exl3,
            experts: vec![], backend_workspace: 0, extra_workspace: [0;2], world, unit_limit: 0 };
        for physical in [[0, 1], [1, 0]] {
        let coordinator = CoordinatorDevices::new(&physical[..world])?;
        let devices = physical.map(|id| Device { library: &library, id });
        let streams = [Stream::new(devices[0])?, Stream::new(devices[1])?];
        let ranks = [0, 1].map(|rank| RankDevice { device: physical[rank], stream: streams[rank].raw });
        for rows in [64, 256] {
            let mut selected = LocalPhysicalInput { layers: vec![1, 2], max_rows: rows, lanes: 2,
                storage: LocalStorage::Mount, route_checks: true, peer_diagnostics: false,
                activations: PrefillActivations::A16 };
            let source = LocalPhysicalSource::read(&cfg, &catalog, &local, &selected, None)?;
            for layers in [vec![1], vec![1, 2], vec![2]] {
                selected.layers = layers;
                for storage in [LocalStorage::Mount, LocalStorage::Push] {
                    selected.storage = storage;
                    let expected = source.inventory(&selected)?;
                    let budgets = [usize::try_from(expected.backend_bytes(0)?)?, usize::try_from(expected.backend_bytes(1)?)?];
                    let weights = if storage == LocalStorage::Push {
                        let mut ranks = Vec::new();
                        for rank in 0..world {
                            let layouts = selected.layers.iter().map(|&layer| catalog.exl3().unwrap()
                                .residency(cuteafd_loader::V41Exl3Layer::Backbone(layer), world, rank)).collect::<Result<Vec<_>>>()?;
                            let arena = ready(devices[rank], layouts.iter().map(|layout| layout.device_arena_layout().map(|(_, bytes)| bytes))
                                .collect::<std::result::Result<Vec<_>, _>>()?)?;
                            ranks.push(layouts.into_iter().enumerate().map(|(index, layout)| {
                                let layer = if world == 1 { ExpertLayer::BackboneFull { layer: selected.layers[index] } }
                                    else { ExpertLayer::BackboneTp2 { layer: selected.layers[index], rank } };
                                Exl3Weights::from_push(arena.clone(), index, layout, Exl3Weights::plan(&catalog, layer)?)
                            }).collect::<Result<Vec<_>>>()?);
                        }
                        Some(ranks)
                    } else { None };
                    let owners = if world == 1 {
                        vec![if let Some(mut weights) = weights {
                            exl3::Exl3Tp2::pushed_whole_weights(devices[0], &catalog, &package, selected.layers.clone(), rows,
                                budgets[0], RtxSelectionPolicy::CanonicalDecode, weights.remove(0))?
                        } else { exl3::Exl3Tp2::load_whole_weights(devices[0], &catalog, &package, selected.layers.clone(), rows,
                            budgets[0], RtxSelectionPolicy::CanonicalDecode)? }]
                    } else if let Some(weights) = weights {
                        exl3::Exl3Tp2::pushed_pair_weights(devices, &catalog, geometry, [&package, &package], selected.layers.clone(), rows,
                            budgets, RtxSelectionPolicy::CanonicalDecode, weights.try_into().map_err(|_| anyhow::anyhow!("rank count"))?)?.into()
                    } else { exl3::Exl3Tp2::load_pair_weights(devices, &catalog, geometry, [&package, &package], selected.layers.clone(), rows,
                        budgets, RtxSelectionPolicy::CanonicalDecode)?.into() };
                    let mut backend = owners.iter().enumerate().map(|(rank, owner)|
                        owner.execution_with_modules(Some(modules(&fixture, devices[rank], &package, rows)?))
                            .map(|execution| Box::new(execution) as Box<dyn RtxExpertLayer>)).collect::<Result<Vec<_>>>()?;
                    let mut actual = Inventory::new(coordinator, "glm", Lifetime::TARGET,
                        Category::Experts, "resident expert owners");
                    // Check the backend's complete physical sum before pair routes/checks are added.
                    for (rank, executor) in backend.iter().enumerate() {
                        let mut receipts = Inventory::new(coordinator, "glm", Lifetime::TARGET,
                            Category::Experts, "resident expert owners");
                        executor.allocation_receipts(&mut receipts, &format!("expert.routed.rank.{rank}"))?;
                        assert_eq!(receipts.receipts.iter().filter(|r| r.allocation.view.is_none()).map(|r| r.allocation.bytes).sum::<u64>(),
                            budgets[rank] as u64, "backend bytes omit a shared root or include pair owners");
                        if world == 2 && storage == LocalStorage::Mount {
                            let layer = selected.layers[0];
                            let layout = catalog.exl3().unwrap().residency(cuteafd_loader::V41Exl3Layer::Backbone(layer), 2, rank)?;
                            let tensor = format!("model.layers.{layer}.mlp.experts.0.gate_proj.trellis");
                            let job = layout.loads.iter().find(|job| job.tensor == tensor).context("missing trellis load")?;
                            let slice = catalog.exl3_tensor_slice(&tensor, 2, rank)?;
                            assert!(slice.rows > 1 && slice.source_row_bytes > slice.selected_row_bytes);
                            for &(index, offset) in &job.destinations {
                                let id = format!("glm.expert.routed.rank.{rank}.weight.{layer}.{index}");
                                let receipt = receipts.receipts.iter().find(|r| r.allocation.id == id).context("missing trellis receipt")?;
                                let buffer = cuteafd_ffi::CuteafdDeviceBuffer {
                                    ptr: (receipt.address as *mut u8).wrapping_add(offset).cast(), bytes: slice.bytes(),
                                    device_id: physical[rank], flags: 0,
                                };
                                let mut actual = vec![0; slice.bytes()];
                                devices[rank].run(|| library.copy_d2h(&mut actual, buffer))?;
                                // Each compact destination row skips the other rank's source columns.
                                for row in 0..slice.rows {
                                    let source = row * slice.source_row_bytes + slice.column_start_bytes;
                                    let expected = (source..source + slice.selected_row_bytes).map(trellis_byte).collect::<Vec<_>>();
                                    assert_eq!(&actual[row * slice.selected_row_bytes..(row + 1) * slice.selected_row_bytes],
                                        expected.as_slice(), "TP2 trellis rank {rank} row {row} physical {physical:?}");
                                }
                            }
                        }
                    }
                    if world == 1 {
                        backend[0].allocation_receipts(&mut actual, "expert.routed.rank.0")?;
                    } else {
                        let pair = pair::Pair::new(&library, ranks, [backend.remove(0), backend.remove(0)], cfg.hidden, cfg.topk,
                            rows, 64, selected.lanes, ExchangeDtype::Bf16)?;
                        assert!(pair.start_checks(true)?);
                        pair.allocation_receipts(&mut actual, "expert.routed")?;
                    }
                    crate::families::glm5::engine::resident_expert_weights(&mut actual.receipts)?;
                    assert!(actual.receipts.iter().any(|r| r.allocation.lifetime == Lifetime::RESIDENT));
                    assert!(actual.receipts.iter().any(|r| r.allocation.lifetime == Lifetime::TARGET));
                    let host = actual.receipts.iter().filter(|r| r.allocation.location == Location::HOST).collect::<Vec<_>>();
                    assert_eq!(host.len(), usize::from(world == 2));
                    if let Some(active) = host.first() {
                        assert_eq!(active.allocation.id, "glm.expert.routed.active");
                        assert_eq!(active.allocation.bytes, (selected.lanes * std::mem::size_of::<Option<usize>>()) as u64);
                        assert_eq!(active.allocation.lifetime, Lifetime::TARGET);
                        assert!(active.allocation.view.is_none() && active.address != 0);
                    }
                    let actual_len = actual.receipts.len() - host.len();
                    let actual = actual.receipts.into_iter().filter(|r| r.allocation.location != Location::HOST)
                        .map(|r| (r.allocation.id.clone(), r.allocation)).collect::<BTreeMap<_, _>>();
                    let expected_len = expected.owners.len() + expected.views.len();
                    let expected = coordinator.remap_weight_arenas(&expected.owners.into_iter().chain(expected.views).collect::<Vec<_>>())?
                        .into_iter().map(|a| (a.id.clone(), a)).collect::<BTreeMap<_, _>>();
                    assert_eq!(actual.len(), actual_len, "duplicate runtime IDs");
                    assert_eq!(expected.len(), expected_len, "duplicate planner IDs");
                    assert_eq!(actual, expected, "TP{world} {storage:?} {rows} {:?} {physical:?}", selected.layers);
                }
            }
        }
        }
    }
    Ok(())
}

#[test]
fn glm_fp8_package_physical_weights_equal_mount_and_push() -> Result<()> {
    use crate::shared::{experts::fp8::{Fp8Layer, regions}, memory::push_fp8::RankPlan};
    use cuteafd_loader::{formats::fp8_experts::ExpertFormat, plan::testing::glm5_compact_weights};
    let fixture = cuteafd_ffi::native_library_lifetime_fixture::Fixture::build_for_tests()?;
    let library = fixture.load()?;
    fixture.immediate_copies(&library)?;
    fixture.strict_device_free(&library, true)?;
    let root = tempfile::tempdir()?;
    let catalog = crate::shared::memory::push_fp8::tests::fixture(root.path(), ExpertFormat::Nvfp4, false)?;
    let tensors = catalog.fp8().context("NVFP4 fixture headers missing")?.for_coordinator();
    let shape = tensors.shape();
    let (config, _) = glm5_compact_weights();
    let mut cfg = GlmDsaConfig::from_hf(&config)?;
    cfg.hidden = shape.hidden;
    cfg.experts = shape.experts;
    cfg.topk = shape.topk;
    cfg.moe_intermediate = shape.intermediate;
    cfg.first_moe_layer = shape.first_layer;
    cfg.layers = shape.layers;
    let package = root.path().join("package/tp2");
    std::fs::create_dir_all(&package)?;
    std::fs::write(package.parent().unwrap().join("manifest.json"), serde_json::to_vec(&serde_json::json!({
        "layouts":{"tp2":{"capacities":[{"capacity":64,"scratch_bytes":256}]}}
    }))?)?;
    let local = glm5::LocalInventory { package: package.clone(), format: glm5::GlmLocalFormat::Nvfp4 { w4a4:true },
        experts:vec![], backend_workspace:0, extra_workspace:[0;2], world:2, unit_limit:0 };
    let mut selected = LocalPhysicalInput { layers:vec![1,2], max_rows:64, lanes:2,
        storage:LocalStorage::Mount, route_checks:true, peer_diagnostics:false, activations:PrefillActivations::A16 };
    let source = LocalPhysicalSource::read(&cfg, &catalog, &local, &selected, None)?;
    for physical in [[0, 1], [1, 0]] {
    let coordinator = CoordinatorDevices::new(&physical)?;
    for layers in [vec![1], vec![1,2], vec![2]] {
        selected.layers = layers;
        for storage in [LocalStorage::Mount, LocalStorage::Push] {
            selected.storage = storage;
            let expected = source.inventory(&selected)?;
            let mut actual = Inventory::new(coordinator, "glm", Lifetime::RESIDENT,
                Category::Experts, "resident expert owners");
            for rank in 0..2 {
                let device = Device { library:&library, id:physical[rank] };
                let (_, slicing) = cuteafd_loader::formats::fp8_experts::exact_layout_directory(&package, &tensors, 2, rank);
                let weights = if storage == LocalStorage::Mount {
                    let arena = crate::shared::experts::weight_layout::arena(device, "fp8",
                        regions(&tensors, &selected.layers, 2, rank, slicing)?)?;
                    device.run(|| selected.layers.iter().map(|&layer| Fp8Layer::load_into(&library,
                        &tensors.for_layer_metadata(layer)?, layer, 2, rank, slicing, &arena)).collect::<Result<Vec<_>>>())?
                } else {
                    let plan = RankPlan::from_tensors(&tensors,
                        &selected.layers.iter().map(|&layer| (layer,slicing)).collect::<Vec<_>>(), 2, rank, usize::MAX)?;
                    let sizes = plan.layers.iter().map(|layer| layer.planes.iter().map(|plane| plane.end).max().unwrap()
                        - layer.planes.iter().filter(|plane| !plane.is_empty()).map(|plane| plane.start).min().unwrap())
                        .collect::<Vec<_>>();
                    let arena = ready(device, sizes)?;
                    plan.layers.iter().enumerate().map(|(index, layer)| Fp8Layer::from_push(arena.clone(), index, layer,
                        plan.layout.region(index)?.start)).collect::<Result<Vec<_>>>()?
                };
                for weight in &weights {
                    weight.allocation_receipts(&mut actual, &format!("expert.routed.rank.{rank}.weight.{}",weight.layer))?;
                }
                let actual_bytes = actual.receipts.iter().filter(|r| r.allocation.location == Location::coordinator(rank as u8)
                    && r.allocation.view.is_none()).map(|r| r.allocation.bytes).sum::<u64>();
                let workspace = expected.owners.iter().filter(|owner| owner.location == Location::coordinator(rank as u8)
                    && (owner.id.ends_with(".scratch") || owner.id.ends_with(".output"))).map(|owner| owner.bytes).sum::<u64>();
                assert_eq!(expected.backend_bytes(rank)?, actual_bytes + workspace,
                    "FP8 backend sum drops shared root or includes pair/check owners");
            }
            let actual_len = actual.receipts.len();
            let actual = actual.receipts.into_iter().map(|r| (r.allocation.id.clone(),r.allocation)).collect::<BTreeMap<_,_>>();
            let expected = coordinator.remap_weight_arenas(&expected.owners.into_iter().chain(expected.views).collect::<Vec<_>>())?
                .into_iter().filter(|a| a.id.starts_with("experts.") || a.id.contains(".weight."))
                .map(|a| (a.id.clone(),a)).collect::<BTreeMap<_,_>>();
            assert_eq!(actual.len(),actual_len,"duplicate FP8 runtime IDs");
            assert_eq!(actual,expected,"FP8 {storage:?} {:?} {physical:?}",selected.layers);
        }
    }
    }
    Ok(())
}

#[derive(Debug, thiserror::Error)]
#[error("set CUTEAFD_GLM_EXL3_WORKSPACE_MANIFEST to the share/PROGRAMS.json of a built GLM EXL3 TP2 installation, including its adjacent lib/exl3 module package")]
struct MissingGlmExl3Input;

#[test]
fn installed_glm_local_physical_equals_serving() -> Result<()> {
    let manifest = std::env::var_os("CUTEAFD_DEPLOYMENT_TEST_PROGRAMS").map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(std::env::var_os("HOME").expect("HOME"))
            .join(".cache/cuteafd/builds/placement-inputs/current/PROGRAMS.json"));
    installed_local_physical_equals_serving(true, manifest,
        "shared::experts::rtx::local_physical_tests::installed_glm_local_physical_equals_serving", false)
}

/// Run from rust/ with a complete installation built at the pinned SparkInfer revision
/// (planner-only manifests are insufficient):
/// CUTEAFD_GLM_EXL3_WORKSPACE_MANIFEST=/path/to/installation/share/PROGRAMS.json \
/// cargo test -p cuteafd-daemon installed_glm_exl3_local_physical_equals_serving -- --ignored --nocapture
#[test]
#[ignore = "requires a built GLM EXL3 TP2 module; set CUTEAFD_GLM_EXL3_WORKSPACE_MANIFEST"]
fn installed_glm_exl3_local_physical_equals_serving() -> Result<()> {
    let manifest = PathBuf::from(std::env::var_os("CUTEAFD_GLM_EXL3_WORKSPACE_MANIFEST")
        .ok_or(MissingGlmExl3Input)?);
    installed_local_physical_equals_serving(false, manifest,
        "shared::experts::rtx::local_physical_tests::installed_glm_exl3_local_physical_equals_serving", true)
}

fn installed_local_physical_equals_serving(native: bool, manifest: PathBuf,
    test_name: &str, ignored: bool) -> Result<()> {
    ensure!(manifest.is_file(), "required installed PROGRAMS.json missing: {}", manifest.display());
    const CHILD: &str = "CUTEAFD_GLM_LOCAL_PHYSICAL_CHILD";
    if std::env::var_os(CHILD).is_none() {
        let mut command = std::process::Command::new(std::env::current_exe()?);
        command.args([test_name, "--exact", "--nocapture"]);
        if ignored { command.arg("--ignored"); }
        let output = command.env(CHILD, "1").output()?;
        let stdout = String::from_utf8_lossy(&output.stdout);
        let stderr = String::from_utf8_lossy(&output.stderr);
        ensure!(output.status.success() && stdout.contains("1 passed; 0 failed"), "physical equality: {stdout}\n{stderr}");
        println!("{stdout}");
        return Ok(());
    }
    let peak_rss = || -> Result<u64> {
        std::fs::read_to_string("/proc/self/status")?.lines().find_map(|line| line.strip_prefix("VmHWM:"))
            .and_then(|value| value.split_whitespace().next()).context("missing RSS high-water mark")?
            .parse::<u64>().map_err(Into::into)
    };
    cuteafd_core::set_expert_geometry(ExpertGeometry::GLM5).map_err(|g| anyhow::anyhow!("unexpected geometry {g:?}"))?;
    let root = PathBuf::from(std::env::var_os("HOME").expect("HOME")).join(".cache/cuteafd/builds/glm-local-expert-physical");
    let fixture = cuteafd_ffi::native_library_lifetime_fixture::Fixture::build_in(&root.join("fixtures"))?;
    let library = fixture.load()?;
    fixture.extent_only(&library)?;
    fixture.quiet_events(&library, true)?;
    for rank in 0..2 { fixture.configure_memory(&library, rank, 96usize << 30)?; }
    let devices = [Device { library: &library, id: 0 }, Device { library: &library, id: 1 }];
    let streams = [Stream::new(devices[0])?, Stream::new(devices[1])?];
    let ranks = [0, 1].map(|rank| RankDevice { device: rank as i32, stream: streams[rank].raw });
    let repo = if native { "kindlingai/glm-5.3-exlr8-k2-k3-k4" }
        else { "wrldsuksgo2mars/GLM-5.3-EXL3-K4-Core-K6" };
    let metadata: serde_json::Value = serde_json::from_slice(&std::fs::read(&manifest)
        .with_context(|| format!("required installed PROGRAMS.json missing: {}", manifest.display()))?)?;
    fixture.register_programs(&library, metadata["programs"].as_array().context("installed programs missing")?)?;
    let programs = library.programs()?.with_manifest(&manifest)?;
    let snapshot = installed_snapshot(repo)?;
    let cfg = GlmDsaConfig::read(&snapshot)?;
    let input = plan::PlanInput::open(&snapshot)?;
    let options = plan::PlanOptions { vision: plan::MediaMode::Off, audio: plan::MediaMode::Off,
        attention_placement: Some(cuteafd_loader::placement::AttentionPlacement::Context),
        layout: Some(plan::layout::LayoutOptions { rtx_bytes: vec![cuteafd_loader::placement::inventory::PRO_TOTAL_BYTES; 2],
            workspace_manifest: Some(manifest.clone()), glm5_drafter_snapshot: Some(installed_snapshot("incoai/GLM-5.3-DFlash2")?),
            ..Default::default() }), ..Default::default() };
    let started = std::time::Instant::now();
    let report = if native { input.exlr8_native_plan("up1", &options, "dflash2")?.report } else { input.plan(&options)? };
    ensure!(started.elapsed().as_secs() < 30, "installed planner exceeded 30s");
    let count = report.solver_placement.as_ref().with_context(|| format!("max card did not resolve placement: {:?}; {:?}", report.config_error, report.hints))?.onboard_layers;
    ensure!(count > 0, "max card selected no local experts");
    let rows = report.prefill_unit_rows.context("max card unit rows missing")? as usize;
    let lanes = plan::resolve_decode_lanes("glm5", options.decode_lanes,
        options.layout.as_ref().unwrap().resolved_concurrency("glm5"))? as usize;
    let catalog = input.expert_catalog()?;
    let exlr8 = if native { Some(MetadataInput::open(&snapshot, "up1")?) } else { None };
    let local = if native { glm5::exlr8_inventory(&snapshot, "up1", &manifest, cfg.first_moe_layer + count,
        2, rows as u64, lanes as u64, false, PrefillActivations::A16)? } else {
        glm5::local_inventory(&cfg, catalog, &manifest, cfg.first_moe_layer + count, rows as u64, lanes as u64, false)? };
    let mut selected = LocalPhysicalInput { layers: (cfg.first_moe_layer..cfg.first_moe_layer + count).collect(),
        max_rows: rows, lanes, storage: LocalStorage::Mount, route_checks: true, peer_diagnostics: false,
        activations: PrefillActivations::A16 };
    let source = LocalPhysicalSource::read(&cfg, catalog, &local, &selected, exlr8.as_ref())?;
    let mut empty = selected.clone();
    empty.layers.clear();
    let empty = source.inventory(&empty)?;
    ensure!(empty.owners.is_empty() && empty.views.is_empty(), "empty prefix retained owners");
    let mut invalid = selected.clone();
    invalid.layers.push(cfg.layers);
    ensure!(source.inventory(&invalid).is_err(), "source accepted an unparsed layer");
    let baseline_rss = peak_rss()?;
    for storage in [LocalStorage::Mount, LocalStorage::Push] {
        selected.storage = storage;
        let expected = source.inventory(&selected)?;
        let budgets = [usize::try_from(expected.backend_bytes(0)?)?, usize::try_from(expected.backend_bytes(1)?)?];
        let backend: [Box<dyn RtxExpertLayer>; 2] = if native {
            let weights = [0, 1].map(|rank| -> Result<Vec<Exlr8Weights<'_>>> {
                let layouts = selected.layers.iter().map(|&layer| exlr8.as_ref().unwrap().plan(layer, 2, rank, ExpertSet::Routed))
                    .collect::<std::result::Result<Vec<_>, _>>()?;
                let arena = if storage == LocalStorage::Push { Some(ready(devices[rank], layouts.iter().map(|l| l.resident_bytes).collect())?) } else { None };
                layouts.into_iter().enumerate().map(|(index, layout)| devices[rank].run(|| {
                    if let Some(arena) = &arena { let budget = Exlr8Weights::plan(&layout)?;
                        Exlr8Weights::from_push(arena.clone(), index, layout, budget)
                    } else { let (weight, records) = dense::recording(|| Exlr8Weights::load(&library, layout, budgets[rank]))?;
                        ensure!(!records.operands.is_empty(), "EXLR8 payload recording did not run"); Ok(weight) }
                })).collect()
            });
            let [left, right] = weights;
            let owners = exlr8::Exlr8RtxWeights::pushed_pair(devices, &programs, selected.layers.clone(), rows, budgets,
                PrefillActivations::A16, [left?, right?])?;
            let [left, right] = owners;
            [Box::new(left.execution()?), Box::new(right.execution()?)]
        } else {
            let packages = [&*local.package, &*local.package];
            let owners = if storage == LocalStorage::Mount {
                let (weights, records) = dense::recording(|| exl3::Exl3Tp2::load_pair_weights(devices, catalog, ExpertGeometry::GLM5,
                    packages, selected.layers.clone(), rows, budgets, RtxSelectionPolicy::CanonicalDecode))?;
                ensure!(!records.operands.is_empty(), "EXL3 payload recording did not run"); weights
            } else {
                let weights = [0, 1].map(|rank| -> Result<Vec<Exl3Weights<'_>>> {
                    let layouts = selected.layers.iter().map(|&layer| catalog.exl3().context("EXL3 catalog")?
                        .residency(cuteafd_loader::V41Exl3Layer::Backbone(layer), 2, rank)).collect::<Result<Vec<_>>>()?;
                    let arena = ready(devices[rank], layouts.iter().map(|l| l.device_arena_layout().map(|(_, bytes)| bytes)).collect::<std::result::Result<Vec<_>, _>>()?)?;
                    layouts.into_iter().enumerate().map(|(index, layout)| { let budget = Exl3Weights::plan(catalog,
                        crate::shared::experts::layer::ExpertLayer::BackboneTp2 { layer: selected.layers[index], rank })?;
                        Exl3Weights::from_push(arena.clone(), index, layout, budget) }).collect()
                });
                let [left, right] = weights;
                exl3::Exl3Tp2::pushed_pair_weights(devices, catalog, ExpertGeometry::GLM5, packages,
                    selected.layers.clone(), rows, budgets, RtxSelectionPolicy::CanonicalDecode, [left?, right?])?
            };
            [Box::new(owners[0].execution_with_modules(Some(modules(&fixture, devices[0], &local.package, rows)?))?),
                Box::new(owners[1].execution_with_modules(Some(modules(&fixture, devices[1], &local.package, rows)?))?)]
        };
        let pair = pair::Pair::new(&library, ranks, backend, cfg.hidden, cfg.topk, rows, 64, lanes, ExchangeDtype::Bf16)?;
        assert!(pair.start_checks(true)?);
        let mut actual = Inventory::new(CoordinatorDevices::new(&[0, 1])?, "glm", Lifetime::TARGET, Category::Experts, "resident expert owners");
        pair.allocation_receipts(&mut actual, "expert.routed")?;
        crate::families::glm5::engine::resident_expert_weights(&mut actual.receipts)?;
        if !native && storage == LocalStorage::Mount {
            let layouts = (0..2).map(|rank| crate::shared::experts::weight_layout::layout(
                devices[rank].id, "exl3", crate::shared::experts::exl3::regions(catalog,
                    selected.layers.iter().map(|&layer| crate::shared::experts::layer::ExpertLayer::BackboneTp2 { layer, rank }))?))
                .collect::<Result<Vec<_>>>()?;
            crate::shared::experts::weight_layout::assert_receipts(&actual.receipts, &layouts)?;
        }
        let actual_count = actual.receipts.iter().filter(|r| r.allocation.location != Location::HOST).count();
        let actual: BTreeMap<_, _> = actual.receipts.into_iter().filter(|r| r.allocation.location != Location::HOST)
            .map(|r| (r.allocation.id.clone(), r.allocation)).collect();
        let expected_count = expected.owners.len() + expected.views.len();
        let expected: BTreeMap<_, _> = expected.owners.into_iter().chain(expected.views).map(|a| (a.id.clone(), a)).collect();
        assert_eq!(actual.len(), actual_count, "duplicate serving receipt IDs");
        assert_eq!(expected.len(), expected_count, "duplicate planned owner IDs");
        assert_eq!(actual, expected, "{repo} {storage:?}, onboard={count}, rows={rows}, lanes={lanes}");
        drop(pair);
        fixture.finish_ordered_uploads(&library)?;
        assert_eq!(fixture.mapped_bytes(&library)?, 0, "virtual extents leaked");
        ensure!(peak_rss()?.saturating_sub(baseline_rss) < 128 * 1024,
            "allocation equality RSS grew by more than 128 MiB");
        println!("{repo} {storage:?}: onboard={count}, rows={rows}, lanes={lanes}, {} equal owners/views", actual.len());
    }
    Ok(())
}
