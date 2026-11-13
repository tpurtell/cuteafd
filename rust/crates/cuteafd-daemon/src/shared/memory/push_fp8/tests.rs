use super::*;
use cuteafd_loader::plan::testing::{fp8, mimo_flash_config, mxfp4, nvfp4, write_snapshot};
use std::{fs::File, os::unix::fs::FileExt};

pub(crate) fn fixture(
    root: &Path,
    format: ExpertFormat,
    bf16_scales: bool,
) -> Result<OfficialV41Catalog> {
    let mut config = mimo_flash_config();
    config["hidden_size"] = 128.into();
    config["n_routed_experts"] = 64.into();
    config["num_experts_per_tok"] = 2.into();
    config["moe_intermediate_size"] = 1024.into();
    config["num_hidden_layers"] = 3.into();
    config["hybrid_layer_pattern"] = serde_json::json!([0, 1, 1]);
    config["moe_layer_freq"] = serde_json::json!([0, 1, 1]);
    if format == ExpertFormat::Nvfp4 {
        config["quantization_config"]["producer"] = serde_json::json!({"name": "modelopt"});
        config["quantization_config"]["config_groups"] = serde_json::json!({"group_0": {
            "targets": ["Linear"],
            "weights": {"dynamic": false, "group_size": 16, "num_bits": 4, "type": "float"},
            "input_activations": {"dynamic": false, "group_size": 16, "num_bits": 4, "type": "float"}
        }});
    }
    let mut tensors = Vec::new();
    for expert in 0..64 {
        for (projection, rows, columns) in
            [("gate", 1024, 128), ("up", 1024, 128), ("down", 128, 1024)]
        {
            let name = format!("model.layers.1.mlp.experts.{expert}.{projection}_proj");
            tensors.extend(match format {
                ExpertFormat::Nvfp4 => nvfp4(&name, rows, columns),
                ExpertFormat::Mxfp4 => mxfp4(&name, rows, columns),
                _ => fp8(&name, rows, columns, None),
            });
        }
    }
    // A later BF16 package shares the same rank arena as the quantized group.
    for expert in 0..64 {
        for (projection, rows, columns) in
            [("gate", 1024, 128), ("up", 1024, 128), ("down", 128, 1024)]
        {
            tensors.push(cuteafd_loader::plan::testing::t(
                format!("model.layers.2.mlp.experts.{expert}.{projection}_proj.weight"),
                "BF16",
                &[rows, columns],
            ));
        }
    }
    if bf16_scales {
        for tensor in &mut tensors {
            if tensor.0.ends_with("_scale_inv") {
                tensor.1 = "BF16";
            }
        }
    }
    write_snapshot(root, &config, &tensors, None);
    let catalog = cuteafd_loader::read_expert_catalog(root)?;
    let tensors = catalog.fp8().context("fixture native tensors absent")?;
    for layer in [1, 2] {
        let tensors = tensors.for_layer_metadata(layer)?;
        for projection in Fp8Projection::ALL {
            for expert in 0..64 {
                for name in source_names(&tensors, layer, expert, projection) {
                    let (source, dtype) = tensors.push_source(&name)?;
                    let image = if dtype == DType::F32 && source.bytes == 4 {
                        // Input values must match gate/up, while alpha tails differ.
                        let value = if name.ends_with("input_scale") {
                            0.5 + expert as f32 / 64.0
                        } else {
                            1.0 + expert as f32 / 64.0 + projection as u8 as f32 / 8.0
                        };
                        value.to_le_bytes().to_vec()
                    } else if name.ends_with("_scale_inv") {
                        let value = (1.0 + expert as f32 / 64.0).to_bits();
                        let bytes = if dtype == DType::Bf16 {
                            ((value >> 16) as u16).to_le_bytes().to_vec()
                        } else {
                            value.to_le_bytes().to_vec()
                        };
                        bytes
                            .into_iter()
                            .cycle()
                            .take(source.bytes as usize)
                            .collect()
                    } else {
                        (0..source.bytes)
                            .map(|i| {
                                (i.wrapping_mul(17) + expert as u64 * 7 + projection as u8 as u64)
                                    as u8
                            })
                            .collect()
                    };
                    File::options()
                        .write(true)
                        .open(source.path)?
                        .write_all_at(&image, source.offset)?;
                }
            }
        }
    }
    Ok(catalog)
}

fn landed(checked: &CheckedPlan, root: &Path, rank: usize) -> Result<Vec<u8>> {
    let mut image = vec![0; checked.arenas[rank] as usize];
    for entry in checked.plan.entries.iter().rev() {
        let file = File::open(root.join(&checked.plan.files[entry.file as usize]))?;
        let pattern = &checked.plan.patterns[entry.pattern as usize];
        for (source, destination) in pattern.runs().into_iter().rev() {
            let start = (entry.dst_base + destination) as usize;
            file.read_exact_at(
                &mut image[start..start + pattern.run as usize],
                entry.src_base + source,
            )?;
        }
    }
    Ok(image)
}

/// The host model of the post-Done spreading: packed grids (no BF16 here)
/// spread to their row pitch with zero padding, as `push_initialize` does.
fn spread_packed(plan: &RankPlan, image: &mut [u8]) {
    for layer in &plan.layers {
        let push_initialize::Layer::Fp8 { expand, .. } = &layer.initialize else { continue };
        for (region, rows, width, pitch) in expand {
            let packed = image[region.start..region.start + rows * width].to_vec();
            let out = &mut image[region.start..region.start + rows * pitch];
            out.fill(0);
            for (row, bytes) in packed.chunks_exact(*width).enumerate() {
                out[row * pitch..row * pitch + width].copy_from_slice(bytes);
            }
        }
    }
}

pub(crate) fn compare(
    plan: &RankPlan,
    catalog: &OfficialV41Catalog,
    world: usize,
    rank: usize,
    image: &[u8],
) -> Result<()> {
    compare_tensors(plan, catalog.fp8().unwrap(), world, rank, image)
}

fn compare_tensors(plan: &RankPlan, tensors: &Fp8ExpertTensors, world: usize, rank: usize,
    image: &[u8]) -> Result<()> {
    for layer in &plan.layers {
        let tensors = tensors.for_layer_metadata(layer.layer)?;
        for (index, projection) in Fp8Projection::ALL.into_iter().enumerate() {
            let (w, s) = tensors.slice_bytes_with(projection, world, rank, layer.slicing)?;
            let mut weights = vec![0; w];
            let mut scales = vec![0; s];
            for expert in 0..tensors.shape().experts {
                tensors.read_slice_with(
                    layer.layer,
                    expert,
                    projection,
                    world,
                    rank,
                    layer.slicing,
                    &mut weights,
                    &mut scales,
                    &mut Vec::new(),
                )?;
                let start = layer.planes[2 * index].start + expert * w;
                assert_eq!(
                    &image[start..start + w],
                    weights,
                    "weight {projection:?} expert {expert}"
                );
                let start = layer.planes[2 * index + 1].start + expert * s;
                assert_eq!(
                    &image[start..start + s],
                    scales,
                    "scale {projection:?} expert {expert}"
                );
                if tensors.format() == ExpertFormat::Nvfp4 {
                    let start = layer.planes[2 * index + 1].start + tensors.shape().experts * s;
                    assert_eq!(
                        &image[start + expert * 4..start + (expert + 1) * 4],
                        tensors
                            .read_alpha(layer.layer, expert, projection)?
                            .to_le_bytes()
                    );
                    let start = start + tensors.shape().experts * 4;
                    assert_eq!(
                        &image[start + expert * 4..start + (expert + 1) * 4],
                        tensors
                            .read_input_scale(layer.layer, expert, projection)?
                            .to_le_bytes()
                    );
                }
            }
        }
    }
    Ok(())
}

#[test]
fn coordinator_whole_and_tp2_selected_planes_match_mount_after_spark_ranks() -> Result<()> {
    use crate::shared::memory::push_coordinator::CoordinatorExpertPlan;
    for format in [ExpertFormat::Fp8Block128, ExpertFormat::Mxfp4, ExpertFormat::Nvfp4] {
        let root = tempfile::tempdir()?;
        let catalog = fixture(root.path(), format, false)?;
        let tensors = catalog.fp8().unwrap().for_coordinator();
        for world in [1, 2] {
            for shard in 0..world {
                for layer in [1, 2] {
                    let admitted = tensors.for_layer_metadata(layer)?.layer_load(layer, world, shard,
                        tensors.exact_slicing(world))?.resident;
                    let plan = if world == 1 {
                        CoordinatorExpertPlan::whole_native(&catalog, &[layer], admitted)?
                    } else { CoordinatorExpertPlan::new(&catalog, &[layer], shard, admitted)? };
                    assert_eq!(plan.layout.bytes(), admitted);
                    assert_eq!(plan.layers, [layer]);
                    let fp8 = plan.fp8.as_ref().unwrap();
                    let group_rank = 6 + shard;
                    let checked = plan.checked(&catalog, root.path(), 8, group_rank)?;
                    assert!(checked.plan.entries.iter().all(|entry| entry.rank == group_rank as u32));
                    let mut image = landed(&checked, root.path(), group_rank)?;
                    spread_packed(fp8, &mut image);
                    compare_tensors(fp8, &tensors, world, shard, &image)?;
                    assert!(if world == 1 {
                        CoordinatorExpertPlan::whole_native(&catalog, &[layer], admitted - 1)
                    } else { CoordinatorExpertPlan::new(&catalog, &[layer], shard, admitted - 1) }.is_err());
                }
            }
        }
    }
    Ok(())
}

#[test]
fn qwen_absolute_draft_is_a_full_checkpoint_set_on_either_physical_gpu() -> Result<()> {
    use crate::shared::{experts::layer::ExpertLayer, memory::push_coordinator::CoordinatorExpertPlan};
    use cuteafd_loader::plan::testing::{qwen4_config, t};
    let root = tempfile::tempdir()?;
    let mut config = qwen4_config(1);
    config["quantization_config"] = serde_json::json!({"quant_method": "fp8", "weight_block_size": [128, 128]});
    config["text_config"]["hidden_size"] = 128.into();
    config["text_config"]["num_experts"] = 64.into();
    config["text_config"]["moe_intermediate_size"] = 1024.into();
    config["text_config"]["mtp_num_hidden_layers"] = 1.into();
    let mut headers = Vec::new();
    for expert in 0..64 {
        for (projection, rows, cols) in [("gate", 1024, 128), ("up", 1024, 128), ("down", 128, 1024)] {
            headers.extend(fp8(&format!("model.language_model.layers.0.mlp.experts.{expert}.{projection}_proj"), rows, cols, None));
            headers.push(t(format!("mtp.layers.0.mlp.experts.{expert}.{projection}_proj.weight"), "BF16", &[rows, cols]));
        }
    }
    write_snapshot(root.path(), &config, &headers, None);
    let catalog = cuteafd_loader::read_expert_catalog(root.path())?;
    let tensors = catalog.fp8().unwrap().for_coordinator();
    for layer in [0, 1] {
        let view = tensors.for_layer_metadata(layer)?;
        for expert in 0..64 { for projection in Fp8Projection::ALL {
            for name in source_names(&view, layer, expert, projection) {
                let (source, dtype) = view.push_source(&name)?;
                let image = if dtype == DType::F32 {
                    1.25f32.to_le_bytes().into_iter().cycle().take(source.bytes as usize).collect::<Vec<_>>()
                } else {
                    (0..source.bytes).map(|i| (i + 7 * expert as u64 + 31 * layer as u64) as u8).collect()
                };
                File::options().write(true).open(source.path)?.write_all_at(&image, source.offset)?;
            }
        }}
    }
    // A role-filtered draft catalog has no backbone headers, but keeps absolute ID 1.
    let draft = cuteafd_loader::read_qwen4_mtp_expert_catalog(root.path(), 1)?;
    for gpu in [0, 1] {
        let backbone = CoordinatorExpertPlan::native_selected(&catalog,
            &[ExpertLayer::BackboneTp2 { layer: 0, rank: gpu }], gpu, usize::MAX)?;
        for source in [&catalog, &draft] {
            let full = CoordinatorExpertPlan::native_selected(source,
                &[ExpertLayer::BackboneFull { layer: 1 }], gpu, usize::MAX)?;
            assert_eq!((full.gpu, full.world, full.layers.as_slice()), (gpu, 1, &[1][..]));
            assert!(CoordinatorExpertPlan::native_selected(source, &[ExpertLayer::BackboneFull { layer: 1 }],
                gpu, full.layout.bytes() - 1).is_err());
            for (plan, source, world, shard) in [(&backbone, &catalog, 2, gpu), (&full, source, 1, 0)] {
                let checked = plan.checked(source, root.path(), 8, 6 + gpu)?;
                let mut image = landed(&checked, root.path(), 6 + gpu)?;
                spread_packed(plan.fp8.as_ref().unwrap(), &mut image);
                // The backbone source is shared by both independently retired sets.
                compare_tensors(plan.fp8.as_ref().unwrap(), &tensors, world, shard, &image)?;
            }
        }
        for selections in [vec![ExpertLayer::BackboneFull { layer: 1 }, ExpertLayer::BackboneFull { layer: 1 }],
            vec![ExpertLayer::BackboneFull { layer: 1 }, ExpertLayer::BackboneFull { layer: 0 }],
            vec![ExpertLayer::BackboneTp2 { layer: 0, rank: gpu }, ExpertLayer::BackboneFull { layer: 1 }],
            vec![ExpertLayer::BackboneTp2 { layer: 0, rank: 1 - gpu }],
            vec![ExpertLayer::Dspark { stage: 0 }], vec![ExpertLayer::DsparkTp2 { stage: 0, rank: gpu }]] {
            assert!(CoordinatorExpertPlan::native_selected(&catalog, &selections, gpu, usize::MAX).is_err());
        }
    }
    Ok(())
}

#[test]
fn planar_rank_images_match_mount_at_tp4_tp6_with_native_scalar_tails() -> Result<()> {
    for format in [ExpertFormat::Fp8Block128, ExpertFormat::Nvfp4] {
        let root = tempfile::tempdir()?;
        let catalog = fixture(root.path(), format, false)?;
        for world in [4, 6] {
            for slicing in [Slicing::Padded, catalog.fp8().unwrap().exact_slicing(world)] {
                for rank in 0..world {
                    let admitted = catalog
                        .fp8()
                        .unwrap()
                        .layer_load(1, world, rank, slicing)?
                        .resident;
                    let plan = RankPlan::new(&catalog, &[(1, slicing)], world, rank, admitted)?;
                    assert_eq!(plan.layout.bytes(), admitted);
                    assert!(
                        RankPlan::new(&catalog, &[(1, slicing)], world, rank, admitted - 1)
                            .is_err()
                    );
                    let checked = plan.checked(&catalog, root.path(), world, rank)?;
                    let mut image = landed(&checked, root.path(), rank)?;
                    spread_packed(&plan, &mut image);
                    compare(&plan, &catalog, world, rank, &image)?;
                    let good = Offer {
                        rank,
                        host: format!("rank-{rank}"),
                        arena_bytes: admitted as u64,
                        placement: plan.placement.clone(),
                    };
                    let assignment = cuteafd_loader::placement::SparkAssignment {
                        rank: rank as u32,
                        resident_bytes: admitted as u64,
                        slices: vec![plan.layers[0].slice.clone()],
                        io: Default::default(),
                    };
                    super::super::push_exl3::validate_offer(
                        &good,
                        rank,
                        &assignment,
                        &plan.placement,
                        admitted as u64,
                    )?;
                    let mut bad = good;
                    bad.placement[0].fingerprint.push('x');
                    assert!(super::super::push_exl3::validate_offer(
                        &bad,
                        rank,
                        &assignment,
                        &plan.placement,
                        admitted as u64
                    )
                    .is_err());
                }
            }
        }
    }
    Ok(())
}

#[test]
#[ignore = "requires an allocated CPU build slot and explicit NVMe fixture directory"]
fn ready_native_views_match_mount_and_retain_one_arena() -> Result<()> {
    use crate::shared::memory::{
        device::Device,
        weight_arena::{ExpectedCoverage, LoadingArena},
    };
    use cuteafd_ffi::native_library_lifetime_fixture::Fixture;
    use std::rc::Rc;
    let native = Fixture::build()?;
    let library = native.load()?;
    for (format, bf16) in [
        (ExpertFormat::Fp8Block128, false),
        (ExpertFormat::Fp8Block128, true),
        (ExpertFormat::Nvfp4, false),
        (ExpertFormat::Mxfp4, false),
    ] {
        let root = tempfile::tempdir_in(native.directory())?;
        let catalog = fixture(root.path(), format, bf16)?;
        // MXFP4 serves exact native-unit slices (MiMo V2.6 Pro): TP6 down
        // widths of 5-6 K/32 scale bytes pad to an 8-byte row, landed packed.
        let slicing = if format == ExpertFormat::Mxfp4 {
            catalog.fp8().unwrap().for_layer_metadata(1)?.exact_slicing(6)
        } else { Slicing::Padded };
        let layers = [(1, slicing), (2, Slicing::Padded)];
        let plan = RankPlan::new(&catalog, &layers, 6, 5, usize::MAX)?;
        if format == ExpertFormat::Mxfp4 {
            let push_initialize::Layer::Fp8 { expand, .. } = &plan.layers[0].initialize else { unreachable!() };
            assert!(!expand.is_empty() && expand.iter().all(|(_, _, w, p)| w < p), "padded down scales land packed");
            let checked = plan.checked(&catalog, root.path(), 6, 5)?;
            let pad = checked.plan.patterns.iter().filter(|p| p.dims.iter().any(|d| d.dst_stride > u64::from(p.run))
                && p.run < 16).count();
            assert_eq!(pad, 0, "no scale run lands with a destination pad");
        }
        let checked = plan.checked(&catalog, root.path(), 6, 5)?;
        let image = landed(&checked, root.path(), 5)?;
        let mut arena = LoadingArena::new(
            Device {
                library: &library,
                id: 0,
            },
            plan.layout.clone(),
            ExpectedCoverage::from_plan(&checked, 5)?, crate::shared::memory::TEST_ARENA
        )?;
        // Reverse run order as the transport may deliver arbitrary fragments.
        for entry in checked.plan.entries.iter().rev() {
            let pattern = &checked.plan.patterns[entry.pattern as usize];
            for (_, offset) in pattern.runs().into_iter().rev() {
                let start = entry.dst_base + offset;
                let bytes = &image[start as usize..start as usize + pattern.run as usize];
                library.copy_h2d(arena.destination(start as usize, bytes.len())?, bytes)?;
                arena.copied(start, bytes.len() as u64)?;
            }
        }
        let initializers = plan
            .layers
            .into_iter()
            .map(|layer| layer.initialize)
            .collect::<Vec<_>>();
        let ready = Rc::new(push_initialize::initialize(
            arena.done()?,
            &initializers,
            None,
            0,
        )?);
        let mut actual = vec![0; image.len()];
        let plan = RankPlan::new(&catalog, &layers, 6, 5, usize::MAX)?;
        for region in 0..plan.layers.len() {
            library.copy_d2h(
                &mut actual[plan.layout.region(region)?],
                ready.region(region)?.buffer(),
            )?;
        }
        compare(&plan, &catalog, 6, 5, &actual)?;
        let before = native.events()?;
        // The expert adapter's cross-layer retention is covered by the service
        // replay. Keep this initializer test usable in memory-only test crates.
        let layer = ready.clone();
        let other = ready.clone();
        assert_eq!(
            layer.region(0)?.buffer().bytes,
            plan.layout.region(0)?.len()
        );
        assert_eq!(
            other.region(1)?.buffer().bytes,
            plan.layout.region(1)?.len()
        );
        drop(ready);
        assert!(!native.events()?[before.len()..].contains('d'));
        drop(other);
        assert!(!native.events()?[before.len()..].contains('d'));
        drop(layer);
        assert_eq!(native.events()?[before.len()..].matches('d').count(), 1);
    }
    Ok(())
}
#[test]
fn equivalent_exact_and_padded_recipes_have_identical_placement_seals() -> Result<()> {
    let root = tempfile::tempdir()?;
    let catalog = fixture(root.path(), ExpertFormat::Fp8Block128, false)?;
    let tensors = catalog.fp8().unwrap();
    for rank in 0..4 {
        let padded = RankPlan::new(&catalog, &[(1, Slicing::Padded)], 4, rank, usize::MAX)?;
        let exact = RankPlan::new(
            &catalog,
            &[(1, tensors.exact_slicing(4))],
            4,
            rank,
            usize::MAX,
        )?;
        assert_eq!(padded.placement, exact.placement);
    }
    Ok(())
}

#[test]
#[ignore = "requires installed native expert headers selected by CUTEAFD_PUSH_FP8"]
fn installed_native_complete_rank_plans_fit_admission_and_control_frames() -> Result<()> {
    let snapshot = std::path::PathBuf::from(
        std::env::var_os("CUTEAFD_PUSH_FP8")
            .context("set CUTEAFD_PUSH_FP8 to the installed snapshot")?,
    );
    let catalog = cuteafd_loader::read_expert_catalog(&snapshot)?;
    let tensors = catalog
        .fp8()
        .context("installed checkpoint has no native expert headers")?;
    let shape = catalog.routed_experts();
    let root = cuteafd_loader::push_load::namespace(&snapshot)
        .map(|(root, _)| root)
        .unwrap_or_else(|_| snapshot.clone());
    for world in [4, 6] {
        for rank in 0..world {
            for exact in [false, true] {
                let layers = (shape.first_layer..shape.layers)
                    .map(|layer| {
                        let view = tensors.for_layer_metadata(layer)?;
                        Ok((
                            layer,
                            if exact {
                                view.exact_slicing(world)
                            } else {
                                Slicing::Padded
                            },
                        ))
                    })
                    .collect::<Result<Vec<_>>>()?;
                let admitted = layers.iter().try_fold(0usize, |total, &(layer, slicing)| {
                    let bytes = tensors
                        .for_layer_metadata(layer)?
                        .layer_load(layer, world, rank, slicing)?
                        .resident;
                    total
                        .checked_add(bytes)
                        .context("installed admission overflow")
                })?;
                let plan = RankPlan::new(&catalog, &layers, world, rank, admitted)?;
                assert_eq!(plan.layout.bytes(), admitted);
                let checked = plan.checked(&catalog, &root, world, rank)?;
                let offer = Offer {
                    rank,
                    host: "rank-host".into(),
                    arena_bytes: admitted as u64,
                    placement: plan.placement.clone(),
                };
                let frame =
                    serde_json::to_vec(&serde_json::json!({"message":"offer", "offer":offer}))?;
                assert!(
                    frame.len() + 1 <= 64 * 1024,
                    "native offer exceeds control bound"
                );
                // Padded slices leave gaps between every row run (MiMo TP6
                // Down: 2.4M per layer); delivery coverage must stay small.
                // Every expected footprint builds; padded TP6 also delivers.
                crate::shared::memory::weight_arena::ExpectedCoverage::from_plan(&checked, rank as u32)?;
                let peak = if world == 6 && !exact && rank < 2 {
                    crate::shared::memory::weight_arena::real_tests::simulate(&checked, rank, "native")?
                } else { 0 };
                assert!(peak < 100_000, "TP{world} rank {rank} exact={exact}: {peak} coverage intervals");
                eprintln!("native TP{world} rank {rank} exact={exact}: {} layers, {admitted} device bytes, {} entries, {} host scratch, {} offer bytes, peak {peak} coverage intervals",
                    layers.len(), checked.plan.entries.len(), plan.host_scratch, frame.len() + 1);
            }
        }
    }
    Ok(())
}

#[test]
fn coordinator_suffix_preserves_fp8_and_nvfp4_spark_plans() -> Result<()> {
    for format in [ExpertFormat::Fp8Block128, ExpertFormat::Nvfp4] {
        let root = tempfile::tempdir()?;
        let catalog = fixture(root.path(), format, false)?;
        for world in [4, 6] {
            let plans = (0..world).map(|rank| {
                let plan = RankPlan::new(&catalog, &[(1, Slicing::Padded), (2, Slicing::Padded)],
                    world, rank, usize::MAX)?;
                Ok((plan.layout.bytes() as u64, plan.placement.clone(),
                    plan.checked(&catalog, root.path(), world, rank)?))
            }).collect::<Result<Vec<_>>>()?;
            super::super::push_exl3::tests::coordinator_suffix(root.path(), &catalog, &plans,
                |assignments, offers, coordinator| checked_plan(root.path(), &catalog, assignments, offers, coordinator))?;
        }
    }
    Ok(())
}

#[test]
#[ignore = "requires installed MiniMax headers, a CPU build slot and explicit NVMe fixture directory"]
fn installed_minimax_nvfp4_push_and_mount_match_with_synthetic_payloads() -> Result<()> {
    use cuteafd_ffi::native_library_lifetime_fixture::Fixture;
    use cuteafd_loader::{RoutedExpertShape,
        plan::{checkpoint::Checkpoint, testing::t}};
    use crate::shared::memory::{device::Device, weight_arena::{ExpectedCoverage, LoadingArena}};
    let snapshot = Path::new("/mnt/sparknest/hf-home/hub/models--MiniMaxAI--MiniMax-M3.1-Flash/snapshots/54e4e69dfc452b1e94a426c24a4fdeeaa8e3ad5e");
    let installed = Checkpoint::open(snapshot)?;
    let stem = "language_model.model.layers.3.block_sparse_moe.experts.0.";
    let headers = installed.tensors.iter().filter(|tensor| tensor.meta.name.starts_with(stem))
        .map(|tensor| {
            let dtype = match tensor.meta.dtype {
                DType::U8 => "U8", DType::F8E4M3 => "F8_E4M3", DType::F32 => "F32",
                ref other => panic!("unexpected vendor NVFP4 dtype {other:?}"),
            };
            t(tensor.meta.name.clone(), dtype, &tensor.meta.shape)
        }).collect::<Vec<_>>();
    assert_eq!(headers.len(), 12, "use one actual expert's complete vendor headers");
    let native = Fixture::build()?;
    let library = native.load()?;
    let root = tempfile::tempdir_in(native.directory())?;
    // Replicate the same expert's headers to the smallest aligned planar group.
    // Payload holes stay sparse; only expert zero receives nonzero matrices.
    let headers = (0..32).flat_map(|expert| headers.iter().map(move |(name, dtype, shape)|
        t(name.replace(".experts.0.", &format!(".experts.{expert}.")), dtype, shape))).collect::<Vec<_>>();
    write_snapshot(root.path(), &installed.config, &headers, None);
    // Keep the vendor's names, dtypes and full matrix dimensions. Only the
    // number of fixture experts is reduced; no vendor payload is read.
    let shape = RoutedExpertShape { layers: 60, first_layer: 3, experts: 32, topk: 1,
        hidden: 6144, intermediate: 3072, draft_stages: 0, draft_experts: 0 };
    let tensors = Fp8ExpertTensors::read_layers(root.path(), shape, &[3].into())?;
    assert_eq!(tensors.format(), ExpertFormat::Nvfp4);
    for expert in 0..32 {
    for projection in Fp8Projection::ALL {
        let names = tensors.projection_source_names(3, expert, projection);
        assert!(names[0].ends_with(".weight_packed"));
        assert!(names[1].ends_with(".weight_scale"));
        assert!(names[2].ends_with(".weight_global_scale"));
        assert!(names[3].ends_with(".input_global_scale"));
        for (index, name) in names.iter().enumerate() {
            if expert != 0 && index < 2 { continue; }
            let (source, _) = tensors.push_source(name)?;
            let image = if index >= 2 {
                // Non-unit divisors catch a missing or double reciprocal.
                (if index == 3 { 8.0f32 } else { 3.0f32 + projection as u8 as f32 }).to_le_bytes().to_vec()
            } else { (0..source.bytes).map(|i| (i.wrapping_mul(17) + projection as u8 as u64 * 7) as u8).collect() };
            File::options().write(true).open(source.path)?.write_all_at(&image, source.offset)?;
        }
    }
    }
    for rank in 0..4 {
        let slicing = tensors.exact_slicing(4);
        let admitted = tensors.layer_load(3, 4, rank, slicing)?.resident;
        let plan = RankPlan::from_tensors(&tensors, &[(3, slicing)], 4, rank, admitted)?;
        let mut builder = Builder::new(root.path(), vec![admitted as u64]);
        plan.append_tensors(&mut builder, &tensors, 0, 4, rank, 0)?;
        let checked = builder.finish()?;
        let image = landed(&checked, root.path(), 0)?;
        let device = Device { library: &library, id: 0 };
        let mut arena = LoadingArena::new(device, plan.layout.clone(),
            ExpectedCoverage::from_plan(&checked, 0)?, crate::shared::memory::TEST_ARENA)?;
        for entry in &checked.plan.entries {
            let pattern = &checked.plan.patterns[entry.pattern as usize];
            for (_, offset) in pattern.runs() {
                let start = entry.dst_base + offset;
                let bytes = &image[start as usize..start as usize + pattern.run as usize];
                library.copy_h2d(arena.destination(start as usize, bytes.len())?, bytes)?;
                arena.copied(start, bytes.len() as u64)?;
            }
        }
        let ready = push_initialize::initialize(arena.done()?, &[plan.layers.into_iter().next().unwrap().initialize], None, 0)?;
        let mut actual = vec![0; image.len()];
        library.copy_d2h(&mut actual, ready.region(0)?.buffer())?;
        let plan = RankPlan::from_tensors(&tensors, &[(3, slicing)], 4, rank, admitted)?;
        compare_tensors(&plan, &tensors, 4, rank, &actual)?;
        // The per-expert builder lands the same expert-zero source bytes,
        // with its scalar pair adjacent rather than in rank-wide scalar vectors.
        for (index, projection) in Fp8Projection::ALL.into_iter().enumerate() {
            let (w, s) = tensors.slice_bytes_with(projection, 4, rank, slicing)?;
            let mut builder = Builder::new(root.path(), vec![(w + s + 8) as u64]);
            builder.fp8_projection(&tensors, 3, 0, projection, 4, rank, slicing,
                Destination { rank: 0, offset: 0, bytes: w as u64 },
                Destination { rank: 0, offset: w as u64, bytes: (s + 8) as u64 })?;
            let single = landed(&builder.finish()?, root.path(), 0)?;
            let weight = &plan.layers[0].planes[2 * index];
            let scale = &plan.layers[0].planes[2 * index + 1];
            assert_eq!(&single[..w], &image[weight.start..weight.start + w]);
            assert_eq!(&single[w..w + s], &image[scale.start..scale.start + s]);
            let alpha = scale.start + 32 * s;
            let input = alpha + 32 * 4;
            assert_eq!(&single[w + s..w + s + 4], &image[alpha..alpha + 4]);
            assert_eq!(&single[w + s + 4..], &image[input..input + 4]);
        }
    }
    Ok(())
}

// Frozen pre-extraction arithmetic: this oracle must not call the shared helper.
fn legacy_host_scratch(plan: &RankPlan) -> usize {
    plan.layers.iter().map(|layer| match &layer.initialize {
        push_initialize::Layer::Fp8 { widen, scalars, inputs, expand, .. } => {
            let widening = widen.iter().map(|range|
                range.len().min(1024 * 1024) + range.len().min(1024 * 1024) / 2).max().unwrap_or(0);
            let scalar = scalars.iter().map(|range| range.len()).max().unwrap_or(0);
            let inputs = inputs.as_ref().map_or(0, |(gate, up, _)| gate.len() + up.len());
            let expanding = expand.iter().map(|(_, _, width, pitch)|
                (1024 * 1024 / pitch).max(1) * (width + pitch)).max().unwrap_or(0);
            widening.max(scalar).max(inputs).max(expanding)
        }
        _ => 0,
    }).max().unwrap_or(0)
}

#[test]
fn extracted_native_scratch_matches_legacy_for_all_storage_geometries() -> Result<()> {
    for (format, bf16_scales) in [(ExpertFormat::Fp8Block128, false),
        (ExpertFormat::Fp8Block128, true), (ExpertFormat::Nvfp4, false), (ExpertFormat::Mxfp4, false)] {
        let root = tempfile::tempdir()?;
        let catalog = fixture(root.path(), format, bf16_scales)?;
        for world in [1, 2, 3, 4, 6, 8] {
            for rank in 0..world {
                for exact in [false, true] {
                    let tensors = catalog.fp8().unwrap();
                    let layers = [1, 2].into_iter().map(|layer| Ok((layer,
                        if exact { tensors.for_layer_metadata(layer)?.exact_slicing(world) }
                        else { Slicing::Padded }))).collect::<Result<Vec<_>>>()?;
                    let plan = RankPlan::new(&catalog, &layers, world, rank, usize::MAX)?;
                    assert_eq!(plan.host_scratch, legacy_host_scratch(&plan),
                        "{format:?} bf16_scales={bf16_scales} TP{world} rank{rank} exact={exact}");
                }
            }
        }
    }
    Ok(())
}

fn installed_snapshot(repo: &str) -> Result<std::path::PathBuf> {
    let home = std::env::var_os("HF_HOME").map(std::path::PathBuf::from).unwrap_or_else(||
        std::path::PathBuf::from(std::env::var_os("HOME").unwrap()).join(".cache/huggingface"));
    let model = home.join("hub").join(format!("models--{}", repo.replace('/', "--")));
    let revision = std::fs::read_to_string(model.join("refs/main"))?;
    Ok(model.join("snapshots").join(revision.trim()))
}

#[test]
#[ignore = "requires installed native family headers; header-only, no CUDA or payload reads"]
fn installed_native_scratch_matches_legacy() -> Result<()> {
    // Kimi's supported Spark path is EXL3, so its official FP8 catalog is not
    // a native push candidate and is covered by the EXL3 loader tests instead.
    for repo in ["zai-org/GLM-5.3", "zai-org/GLM-5.3-Flash", "nvidia/GLM-5.3-NVFP4",
        "nvidia/GLM-5.3-Flash-NVFP4", "stepfun-ai/Step-5-Preview",
        "Qwen/Qwen3.8-Flash-Next-FP8", "nvidia/Qwen3.8-Flash-Next-NVFP4",
        "XiaomiMiMo/MiMo-V2.6-Pro-MOPD", "MiniMaxAI/MiniMax-M3.1-Flash"] {
        let snapshot = installed_snapshot(repo)?;
        let catalog = cuteafd_loader::read_expert_catalog(&snapshot)?;
        let tensors = catalog.fp8().with_context(|| format!("{repo}: native headers absent"))?;
        let layer = tensors.shape().first_layer;
        for world in [1, 2, 4, 6] {
            for rank in 0..world {
                for exact in [false, true] {
                    let slicing = if exact { tensors.for_layer_metadata(layer)?.exact_slicing(world) }
                        else { Slicing::Padded };
                    let plan = RankPlan::new(&catalog, &[(layer, slicing)], world, rank, usize::MAX)?;
                    assert_eq!(plan.host_scratch, legacy_host_scratch(&plan),
                        "{repo} TP{world} rank{rank} exact={exact}");
                }
            }
        }
        eprintln!("{repo}: legacy scratch exact; native header geometry TP1/2/4/6");
    }
    Ok(())
}
