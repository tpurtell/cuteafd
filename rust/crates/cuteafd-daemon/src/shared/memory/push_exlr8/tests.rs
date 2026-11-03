use super::*;
use anyhow::Context;
use cuteafd_loader::plan::testing::exlr8::fixture;
use std::{fs::File, os::unix::fs::FileExt};

#[test]
fn mixed_rank_plans_land_mount_bytes_and_reject_changed_placement() -> Result<()> {
    for padded in [false, true] {
        let root = tempfile::tempdir()?;
        fixture(root.path(), padded);
        for world in [4, 6] {
            for rank in 0..world {
                let plan = RankPlan::new(root.path(), "mixed", &[3], world, rank, usize::MAX)?;
                let checked = plan.checked(root.path(), root.path(), world, rank)?;
                let mut landed = vec![0; plan.layout.bytes()];
                let mut coverage = vec![false; landed.len()];
                for entry in &checked.plan.entries {
                    let pattern = &checked.plan.patterns[entry.pattern as usize];
                    let file = File::open(root.path().join(&checked.plan.files[entry.file as usize]))?;
                    for (source, destination) in pattern.runs() {
                        let first = (entry.dst_base + destination) as usize;
                        let bytes = pattern.run as usize;
                        assert!(coverage[first..first + bytes].iter().all(|v| !v));
                        coverage[first..first + bytes].fill(true);
                        file.read_exact_at(&mut landed[first..first + bytes], entry.src_base + source)?;
                    }
                }
                let residency = &plan.residencies[0];
                for unit in &residency.loads {
                    let mut mount = vec![0; unit.staging_bytes()];
                    unit.read(&mut mount)?;
                    let mut cursor = 0;
                    for (source, &destination) in unit.source.iter().zip(&unit.destination) {
                        assert_eq!(landed[destination..destination + source.bytes], mount[cursor..cursor + source.bytes]);
                        cursor += source.bytes;
                    }
                }
                let assignment = cuteafd_loader::placement::SparkAssignment {
                    rank: rank as u32, resident_bytes: plan.layout.bytes() as u64,
                    slices: plan.placement.iter().map(|p| p.slice.clone()).collect(), io: Default::default(),
                };
                let mut offer = Offer { rank, host: "fixture".into(), arena_bytes: plan.layout.logical_bytes() as u64,
                    placement: plan.placement.clone() };
                super::super::push_exl3::validate_offer(&offer, rank, &assignment, &plan.placement, offer.arena_bytes)?;
                offer.placement[0].fingerprint.push('0');
                assert!(super::super::push_exl3::validate_offer(&offer, rank, &assignment,
                    &plan.placement, offer.arena_bytes).is_err());
            }
        }
    }
    Ok(())
}

#[test]
fn coordinator_native_worlds_preserve_exact_payload_residency() -> Result<()> {
    use crate::shared::memory::push_coordinator::CoordinatorExpertPlan;
    for padded in [false, true] {
        let root = tempfile::tempdir()?;
        fixture(root.path(), padded);
        let catalog = fixture_catalog(root.path())?;
        for world in [1, 2] {
            for rank in 0..world {
                let residency = Residency::plan(root.path(), "mixed", 3, world, rank, ExpertSet::Routed)?;
                let plan = CoordinatorExpertPlan::exlr8(&catalog, "mixed", &[3], world, rank,
                    residency.resident_bytes)?;
                assert_eq!(plan.layout.bytes(), residency.resident_bytes);
                assert_eq!((plan.exlr8[0].world, plan.exlr8[0].rank), (world, rank));
                assert_eq!(plan.exlr8[0].compact_unit_limit(), residency.units.div_ceil(world));
                let mut builder = Builder::new(root.path(), vec![0; world - 1].into_iter()
                    .chain([plan.logical_bytes()]).collect());
                // Group rank and native TP rank are independent (the coordinator may follow Sparks).
                plan.append_at(&mut builder, &catalog, (world - 1) as u32, 0)?;
                let checked = builder.finish()?;
                assert!(!checked.plan.entries.is_empty());
                assert!(checked.plan.entries.iter().all(|entry| entry.rank == (world - 1) as u32));
                assert!(CoordinatorExpertPlan::exlr8(&catalog, "mixed", &[3], world, rank,
                    residency.resident_bytes - 1).is_err());
            }
        }
        for (world, rank) in [(0, 0), (1, 1), (2, 2), (3, 0)] {
            assert!(CoordinatorExpertPlan::exlr8(&catalog, "mixed", &[3], world, rank, usize::MAX).is_err());
        }
    }
    Ok(())
}

#[test]
fn installed_exlr8_rank_plans_match_admission_when_supplied() -> Result<()> {
    let Some(snapshot) = std::env::var_os("CUTEAFD_PUSH_EXLR8") else { return Ok(()); };
    let snapshot = Path::new(&snapshot);
    for world in [4, 6] {
        for rank in 0..world {
            let plan = RankPlan::new(snapshot, "up1", &[3], world, rank, usize::MAX)?;
            let checked = plan.checked(snapshot, snapshot, world, rank)?;
            ensure!(!checked.plan.entries.is_empty(), "installed EXLR8 plan is empty");
            assert_eq!(plan.layout.bytes(), plan.residencies[0].resident_bytes);
            eprintln!("EXLR8 TP{world} rank {rank}: resident={} entries={} patterns={}",
                plan.layout.bytes(), checked.plan.entries.len(), checked.plan.patterns.len());
        }
    }
    Ok(())
}

#[test]
fn coordinator_suffix_preserves_mixed_exlr8_spark_plans() -> Result<()> {
    for padded in [false, true] {
        let root = tempfile::tempdir()?;
        fixture(root.path(), padded);
        let catalog = fixture_catalog(root.path())?;
        for world in [4, 6] {
            let plans = (0..world).map(|rank| {
                let plan = RankPlan::new(root.path(), "mixed", &[3], world, rank, usize::MAX)?;
                Ok((plan.layout.bytes() as u64, plan.placement.clone(),
                    plan.checked(root.path(), root.path(), world, rank)?))
            }).collect::<Result<Vec<_>>>()?;
            super::super::push_exl3::tests::coordinator_suffix(root.path(), &catalog, &plans,
                |assignments, offers, coordinator| checked_plan(root.path(), root.path(), &catalog,
                    "mixed", &[3], assignments, offers, coordinator))?;
        }
    }
    Ok(())
}

#[test]
fn onboarded_tp2_destinations_join_the_exlr8_group_without_changing_residency() -> Result<()> {
    use crate::shared::memory::push_coordinator::CoordinatorExpertPlan;
    for padded in [false, true] {
        let root = tempfile::tempdir()?;
        fixture(root.path(), padded);
        fixture_catalog(root.path())?;
        // Complete the checkpoint inventory while selecting sparse onboarded layers.
        let manifest_path = root.path().join("manifests/exlr8-manifest-mixed.json");
        let mut manifest: serde_json::Value = serde_json::from_slice(&std::fs::read(&manifest_path)?)?;
        for layer in 4..8 {
            manifest["layers"][layer.to_string()] = manifest["layers"]["3"].clone();
            for suffix in ["index", "k2", "k3", "k4"] {
                let source = root.path().join(format!("experts/exlr8-layer-003.{suffix}.safetensors"));
                let mut image = std::fs::read(source)?;
                let header_bytes = u64::from_le_bytes(image[..8].try_into()?) as usize;
                let mut header: serde_json::Value = serde_json::from_slice(&image[8..8 + header_bytes])?;
                header["__metadata__"]["layer"] = layer.to_string().into();
                let encoded = serde_json::to_vec(&header)?;
                assert!(encoded.len() <= header_bytes);
                image[8..8 + header_bytes].fill(b' ');
                image[8..8 + encoded.len()].copy_from_slice(&encoded);
                std::fs::write(root.path().join(format!("experts/exlr8-layer-{layer:03}.{suffix}.safetensors")), image)?;
            }
        }
        std::fs::write(manifest_path, serde_json::to_vec(&manifest)?)?;
        let config_path = root.path().join("config.json");
        let mut config: serde_json::Value = serde_json::from_slice(&std::fs::read(&config_path)?)?;
        config["num_hidden_layers"] = 8.into();
        config["indexer_types"] = serde_json::json!(vec!["full"; 8]);
        config["mlp_layer_types"] = serde_json::json!(["dense", "dense", "dense", "sparse", "sparse", "sparse", "sparse", "sparse"]);
        std::fs::write(config_path, serde_json::to_vec(&config)?)?;
        let catalog = fixture_catalog(root.path())?;
        for (layers, gpu) in [(vec![], 0), (vec![7, 5], 0), (vec![5, 5], 0), (vec![5, 7], 2)] {
            assert!(CoordinatorExpertPlan::exlr8(&catalog, "mixed", &layers, 2, gpu, usize::MAX).is_err());
        }
        for sparks in [0, 4, 6] {
            let mut assignments = Vec::new();
            let mut offers = Vec::new();
            let mut spark_entries = Vec::new();
            for rank in 0..sparks {
                let plan = RankPlan::new(root.path(), "mixed", &[3], sparks, rank, usize::MAX)?;
                assignments.push(cuteafd_loader::placement::SparkAssignment { rank: rank as u32,
                    resident_bytes: plan.layout.bytes() as u64,
                    slices: plan.placement.iter().map(|p| p.slice.clone()).collect(), io: Default::default() });
                offers.push(Offer { rank, host: "fixture".into(), arena_bytes: plan.layout.logical_bytes() as u64,
                    placement: plan.placement.clone() });
                let checked = plan.checked(root.path(), root.path(), sparks, rank)?;
                for entry in &checked.plan.entries {
                    spark_entries.push((entry.rank, entry.src_base, entry.dst_base,
                        checked.plan.files[entry.file as usize].clone(),
                        checked.plan.patterns[entry.pattern as usize].clone()));
                }
            }
            let base = 256;
            for gpu in 0..2 {
                let resident = [5, 7].into_iter().map(|layer| Residency::plan(root.path(), "mixed", layer, 2, gpu, ExpertSet::Routed)
                    .map(|r| r.resident_bytes)).collect::<std::result::Result<Vec<_>, _>>()?.into_iter().sum::<usize>();
                let plan = CoordinatorExpertPlan::exlr8(&catalog, "mixed", &[5, 7], 2, gpu, resident)?;
                assert_eq!(plan.layers, [5, 7]);
                assert_eq!(plan.exlr8.len(), 2);
                assert_eq!(plan.layout.bytes(), resident);
                assert_eq!(plan.logical_bytes(), resident as u64);
                let retained = plan.exlr8.iter().map(Residency::host_bytes).sum::<usize>();
                let descriptors = plan.exlr8.iter().map(Residency::descriptor_host_bytes)
                    .collect::<Result<Vec<_>>>()?.into_iter().max().unwrap();
                assert_eq!(plan.exlr8_host_bytes()?, retained + descriptors);
                let host_peak = retained + descriptors + 128 * (1 << 20);
                let host_plans = [Some(plan)];
                assert_eq!(CoordinatorExpertPlan::admit_host(&host_plans.iter().map(Option::as_ref).collect::<Vec<_>>(), &[], 128, 1 << 20,
                    host_peak as u64)?, host_peak);
                assert!(CoordinatorExpertPlan::admit_host(&host_plans.iter().map(Option::as_ref).collect::<Vec<_>>(), &[], 128, 1 << 20,
                    host_peak as u64 - 1).is_err());
                assert!(CoordinatorExpertPlan::exlr8(&catalog, "mixed", &[5, 7], 2, gpu,
                    resident - 1).is_err());
                offers.push(Offer { rank: sparks + gpu, host: "coordinator".into(),
                    arena_bytes: base + host_plans[0].as_ref().unwrap().logical_bytes(), placement: Vec::new() });
            }
            let host_plans = (0..2).map(|gpu| CoordinatorExpertPlan::exlr8(&catalog, "mixed", &[5, 7], 2, gpu,
                usize::MAX).map(Some)).collect::<Result<Vec<_>>>()?;
            let host_peak = 2 * 128 * (1 << 20) + host_plans.iter().map(|plan|
                plan.as_ref().unwrap().exlr8_host_bytes()).collect::<Result<Vec<_>>>()?.into_iter().sum::<usize>();
            assert_eq!(CoordinatorExpertPlan::admit_host(&host_plans.iter().map(Option::as_ref).collect::<Vec<_>>(), &[], 128, 1 << 20, host_peak as u64)?, host_peak);
            assert!(CoordinatorExpertPlan::admit_host(&host_plans.iter().map(Option::as_ref).collect::<Vec<_>>(), &[], 128, 1 << 20, host_peak as u64 - 1).is_err());
            let coordinator = super::super::push_exl3::CoordinatorRanks { ranks: 2,
                append: Box::new(move |builder, catalog, offers| {
                    let catalog = catalog.context("fixture catalog")?;
                    for (gpu, offer) in offers.iter().enumerate() {
                        let plan = CoordinatorExpertPlan::exlr8(catalog, "mixed", &[5, 7], 2, gpu, usize::MAX)?;
                        assert_eq!(offer.arena_bytes, base + plan.logical_bytes());
                        plan.append_at(builder, catalog, offer.rank as u32, base)?;
                    }
                    Ok(())
                }) };
            let checked = checked_plan(root.path(), root.path(), &catalog, "mixed", &[3],
                &assignments, &offers, Some(&coordinator))?;
            assert_eq!(checked.arenas, offers.iter().map(|o| o.arena_bytes).collect::<Vec<_>>());
            for rank in 0..sparks + 2 {
                assert!(checked.plan.entries.iter().any(|e| e.rank == rank as u32));
            }
            for entry in checked.plan.entries.iter().filter(|e| e.rank >= sparks as u32) {
                assert!(entry.dst_base >= base);
                assert!(["exlr8-layer-005", "exlr8-layer-007"].iter()
                    .any(|name| checked.plan.files[entry.file as usize].contains(name)));
            }
            // The Spark sources and destinations are unchanged by the two new ranks.
            let actual = checked.plan.entries.iter().filter(|e| e.rank < sparks as u32)
                .map(|e| (e.rank, e.src_base, e.dst_base, checked.plan.files[e.file as usize].clone(),
                    checked.plan.patterns[e.pattern as usize].clone())).collect::<Vec<_>>();
            assert_eq!(actual, spark_entries);
        }
    }
    Ok(())
}

pub(crate) fn fixture_catalog(root: &Path) -> Result<cuteafd_loader::OfficialV41Catalog> {
    let config: serde_json::Value = serde_json::from_slice(&std::fs::read(root.join("config.json"))?)?;
    Ok(cuteafd_loader::OfficialV41Catalog::exlr8_transport_fixture(root,
        cuteafd_loader::RoutedExpertShape {
            layers: config["num_hidden_layers"].as_u64().context("fixture layers")? as usize,
            first_layer: 3, experts: 2, topk: 1, hidden: 128, intermediate: 384,
            draft_stages: 0, draft_experts: 0,
        }))
}
