use anyhow::Result;
use cuteafd_loader::serving_capacity::WeightArenaLayout;
use super::weight_layout::*;

fn check(layout: &WeightArenaLayout, regions: &[(String, usize)]) {
    let mut end = 0;
    for (name, bytes) in regions {
        let view = layout.view(name).unwrap();
        assert_eq!(view.offset() % 256, 0);
        assert!(view.offset() >= end);
        assert_eq!(view.bytes(), *bytes);
        end = view.offset() + bytes;
    }
    assert_eq!(end, layout.bytes(), "no final tail rounding");
    assert_eq!(regions.len(), layout.views().len());
    assert!(layout.views().keys().all(|name| !name.contains("raw_scale") && !name.contains("staging")));
}

#[test]
fn full_backing_peak_refuses_before_allocation() -> Result<()> {
    // Previously layer 0 admitted 257 + 1024, then layer 1 admitted
    // 257 + 257: both fit 1281, but the early full backing does not.
    let regions = vec![("early".into(), 257), ("late".into(), 257)];
    let layout = layout(0, "fixture", regions)?;
    assert!(257 + 1024 <= 1281 && 257 + 257 <= 1281);
    assert!(admit(&layout, 1024, 0, 1281).is_err());
    assert!(admit(&layout, 1024, 33, layout.bytes() + 1024 + 32).is_err());
    admit(&layout, 1024, 33, layout.bytes() + 1024 + 33)?;
    assert!(admit(&layout, usize::MAX, 0, usize::MAX).is_err());
    Ok(())
}

#[test]
fn selected_prefix_retains_earlier_maximum_staging_and_padding() -> Result<()> {
    let first = layout(0, "fixture", [("first".into(), 257)])?;
    let union = layout(0, "fixture", [("first".into(), 257), ("last".into(), 257)])?;
    let budget = 1600;
    assert!(fits(&first, 1024, 0, budget)?);
    // A later small staging request fits after the earlier payload, but the
    // earlier maximum staging must still fit alongside the complete arena.
    assert!(257 + 257 <= budget);
    assert!(!fits(&union, 1024, 0, budget)?);
    assert!(fits(&union, 1024, 31, union.bytes() + 1024 + 31)?);
    assert!(fits(&union, usize::MAX, 0, usize::MAX).is_err());
    Ok(())
}

#[test]
fn exl3_selected_tp1_tp2_regions_preserve_composite_layers() -> Result<()> {
    use crate::shared::experts::{exl3, layer::ExpertLayer};
    use cuteafd_loader::plan::testing::{qwen4_config, qwen4_exl3, write_snapshot};
    let root = tempfile::tempdir()?;
    let (tensors, _) = qwen4_exl3(2, 4);
    write_snapshot(root.path(), &qwen4_config(2), &tensors, None);
    let catalog = cuteafd_loader::read_expert_catalog(root.path())?;
    for world in [1, 2] {
        for rank in 0..world {
            let selections = (0..2).map(|layer| if world == 1 {
                ExpertLayer::BackboneFull { layer }
            } else { ExpertLayer::BackboneTp2 { layer, rank } }).collect::<Vec<_>>();
            let regions = exl3::regions(&catalog, selections.iter().copied())?;
            let planned = layout(rank as i32, "exl3", regions.clone())?;
            check(&planned, &regions);
            for (selection, (_, bytes)) in selections.into_iter().zip(&regions) {
                assert_eq!(*bytes, exl3::planned_residency(&catalog, selection)?.device_arena_layout()?.1);
            }
        }
    }
    Ok(())
}

#[test]
fn native_final_planes_exclude_raw_scales_and_preserve_vector_tail() -> Result<()> {
    use cuteafd_loader::push_load::native::NativePlanes;
    for nvfp4 in [false, true] {
        for world in [1, 2] {
            for rank in 0..world {
                let width = 2048 / world;
                let kernel = if nvfp4 { width } else {
                    cuteafd_core::mxfp4_layouts::stored_width(5120, width).unwrap()
                };
                let planes = NativePlanes::new(5120, width, kernel, 3, nvfp4)?;
                let regions = planes.planes.into_iter().chain(planes.vectors)
                    .filter(|(_, bytes)| *bytes > 0).enumerate()
                    .map(|(index, (_, bytes))| (format!("layer.0.{index}"), bytes)).collect::<Vec<_>>();
                let planned = layout(rank as i32, "native", regions.clone())?;
                check(&planned, &regions);
                assert_eq!(planned.bytes(), planes.bytes);
                assert_eq!(regions.len(), if nvfp4 { 8 } else { 4 });
            }
        }
    }
    Ok(())
}

#[test]
fn native_plane_receipts_charge_each_rank_backing_once() -> Result<()> {
    use crate::shared::{experts::layer::ResidentPlane, memory::receipts::Inventory};
    use cuteafd_loader::{placement::planner::{CoordinatorDevices, Lifetime}, push_load::native::NativePlanes};
    let fixture = cuteafd_ffi::native_library_lifetime_fixture::Fixture::build_for_tests()?;
    let library = fixture.load()?;
    fixture.strict_device_free(&library, true)?;
    for nvfp4 in [false, true] {
        let mut layouts = Vec::new();
        let mut inventory = Inventory::new(CoordinatorDevices::new(&[0, 1])?, "fixture",
            Lifetime::TARGET, cuteafd_core::memory_layout::Category::Experts, "expert fixture");
        let mut owners = Vec::new();
        for rank in 0..2 {
            let planes = NativePlanes::new(256, 256, 256, 3, nvfp4)?;
            let regions = planes.planes.into_iter().chain(planes.vectors).filter(|(_, bytes)| *bytes > 0)
                .enumerate().map(|(index, (_, bytes))| (format!("layer.0.{index}"), bytes)).collect::<Vec<_>>();
            let device = crate::shared::memory::device::Device { library: &library, id: rank };
            let arena = arena(device, "native", regions.clone())?;
            layouts.push(arena.layout().clone());
            for (name, bytes) in regions {
                let plane = ResidentPlane::from(crate::shared::memory::DeviceAllocation::from_allocation(arena.view(&name, bytes)?));
                plane.allocation_receipts(&mut inventory, &format!("rank.{rank}.{name}"))?;
                owners.push(plane);
            }
        }
        assert_receipts(&inventory.receipts, &layouts)?;
        assert_eq!(backing_bytes(&inventory.receipts)?, layouts.iter().map(|l| l.bytes() as u64).sum::<u64>());
        if nvfp4 {
            assert!(layouts.iter().all(|l| l.padding_bytes() > 0));
            assert!(backing_bytes(&inventory.receipts)? > layouts.iter().map(|l| l.payload_bytes() as u64).sum());
        }
        inventory.receipts.retain(|r| r.allocation.id != layouts[0].id());
        assert!(assert_receipts(&inventory.receipts, &layouts).is_err(), "missing physical root must fail");
        drop(owners);
    }
    assert_eq!(fixture.device_free_mismatch(&library)?.0, 0);
    Ok(())
}

#[test]
#[ignore = "CPU inventory of installed checkpoints; run explicitly with --nocapture"]
fn installed_format_inventory() -> Result<()> {
    use crate::shared::experts::{exl3, layer::ExpertLayer};
    use anyhow::Context;
    use cuteafd_loader::{formats::fp8_experts::Slicing,
        plan::exlr8::residency::{ExpertSet, Residency}, push_load::native::NativePlanes};
    use std::path::PathBuf;
    fn snapshot(repo: &str) -> Result<PathBuf> {
        let root = PathBuf::from(std::env::var_os("HF_HOME").unwrap_or_else(|| "/mnt/sparknest/hf-home".into()))
            .join("hub").join(format!("models--{}", repo.replace('/', "--")));
        let revision = std::fs::read_to_string(root.join("refs/main"))?;
        Ok(root.join("snapshots").join(revision.trim()))
    }
    fn report(label: &str, regions: Vec<(String, usize)>) -> Result<()> {
        let layout = layout(0, label, regions.clone())?;
        let large = regions.iter().filter(|(_, n)| *n > 1 << 20).map(|(_, n)| *n).collect::<Vec<_>>();
        let rounded = regions.iter().map(|(_, n)| if *n > 1 << 20 { n.next_multiple_of(2 << 20) } else { *n }).sum::<usize>();
        println!("INVENTORY {label}: n={} sum={} large_n={} large_sum={} sm120_rounded={} arena={} padding={} tail={}",
            regions.len(), layout.payload_bytes(), large.len(), large.iter().sum::<usize>(), rounded,
            layout.bytes(), layout.padding_bytes(), layout.bytes() % 256);
        Ok(())
    }
    let root = snapshot("wrldsuksgo2mars/GLM-5.3-Flash-EXL3-K3.25-Core-FP8")?;
    let catalog = cuteafd_loader::read_expert_catalog(&root)?;
    for world in [1, 2] {
        let selections = (catalog.routed_experts().first_layer..catalog.routed_experts().layers).map(|layer| if world == 1 {
            ExpertLayer::BackboneFull { layer }
        } else { ExpertLayer::BackboneTp2 { layer, rank: 0 } });
        report(&format!("exl3-tp{world}"), exl3::regions(&catalog, selections)?)?;
    }
    for (label, repo) in [("fp8", "zai-org/GLM-5.3-Flash"),
        ("mxfp4-package", "XiaomiMiMo/MiMo-V2.6-Pro-MOPD"),
        ("nvfp4-package", "nvidia/GLM-5.3-Flash-NVFP4")] {
        let root = snapshot(repo)?;
        let catalog = cuteafd_loader::read_expert_catalog(&root)?;
        let tensors = catalog.fp8().context("package inventory catalog missing")?;
        let layers = (0..catalog.routed_experts().layers).filter(|&layer| tensors.has_layer(layer)).collect::<Vec<_>>();
        for world in [1, 2] {
            report(&format!("{label}-tp{world}"), cuteafd_loader::serving_capacity::fp8_expert_regions(tensors, &layers, world, 0, Slicing::Padded)?)?;
        }
    }
    let root = snapshot("kindlingai/glm-5.3-flash-exlr8-k2-k3-k4")?;
    let catalog = cuteafd_loader::read_expert_catalog(&root)?;
    for world in [1, 2, 6] {
        let regions = (catalog.routed_experts().first_layer..catalog.routed_experts().layers)
            .map(|layer| {
                let plan = Residency::plan(&root, "k3", layer, world, 0, ExpertSet::Routed)?;
                Ok((format!("layer.{layer}"), plan.resident_bytes))
            }).collect::<Result<Vec<_>>>()?;
        report(&format!("exlr8-tp{world}"), regions)?;
    }
    for (label, repo, nvfp4) in [("native-mxfp4", "deepseek-ai/DeepSeek-V4.1-Flash-1017", false),
        ("native-nvfp4", "nvidia/DeepSeek-V4.1-Flash-NVFP4", true)] {
        let root = snapshot(repo)?;
        let catalog = cuteafd_loader::read_expert_catalog(&root)?;
        let shape = catalog.routed_experts();
        for world in [1, 2] {
            let width = shape.intermediate / world;
            let kernel = if nvfp4 { width.next_multiple_of(128) } else {
                cuteafd_core::mxfp4_layouts::stored_width(shape.hidden, width).context("native width")?
            };
            let planes = NativePlanes::with_scale_codes(shape.hidden, width, kernel, shape.experts, nvfp4,
                !nvfp4 && catalog.mxfp4_scale_codes())?;
            let regions = (shape.first_layer..shape.layers).flat_map(|layer| planes.planes.into_iter().chain(planes.vectors)
                .filter(|(_, bytes)| *bytes > 0).enumerate().map(move |(index, (_, bytes))|
                    (format!("layer.{layer}.{index}"), bytes))).collect();
            report(&format!("{label}-tp{world}"), regions)?;
        }
    }
    Ok(())
}
