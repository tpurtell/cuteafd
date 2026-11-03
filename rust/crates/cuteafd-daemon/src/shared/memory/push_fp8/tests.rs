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
            ExpectedCoverage::from_plan(&checked, 5)?,
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
