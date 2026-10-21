use super::*;
use std::{fs::File, os::unix::fs::FileExt};
use tempfile::TempDir;

pub(super) fn land(plan: &CheckedPlan, mount: &Path) -> Vec<Vec<u8>> {
    let mut arenas = plan
        .arenas
        .iter()
        .map(|&n| vec![0; n as usize])
        .collect::<Vec<_>>();
    for entry in plan.plan.entries.iter().rev() {
        let file = File::open(mount.join(&plan.plan.files[entry.file as usize])).unwrap();
        let pattern = &plan.plan.patterns[entry.pattern as usize];
        for (source, destination) in pattern.runs().into_iter().rev() {
            let start = (entry.dst_base + destination) as usize;
            file.read_exact_at(
                &mut arenas[entry.rank as usize][start..start + pattern.run as usize],
                entry.src_base + source,
            )
            .unwrap();
        }
    }
    arenas
}

#[test]
fn file_id_reuse_preserves_first_seen_order_and_checks_new_sources() {
    let root = TempDir::new().unwrap();
    let a = root.path().join("a");
    let b = root.path().join("b");
    std::fs::write(&a, [0u8; 16]).unwrap();
    std::fs::write(&b, [1u8; 16]).unwrap();
    let mut builder = Builder::new(root.path(), vec![16]);
    assert_eq!(builder.file(&a).unwrap(), 0);
    assert_eq!(builder.file(&a).unwrap(), 0);
    assert_eq!(builder.file(&b).unwrap(), 1);
    assert_eq!(builder.file(&a).unwrap(), 0);
    assert_eq!(builder.plan.files, ["a", "b"]);
    assert_eq!(builder.file_sizes, [16, 16]);
    assert!(builder.file(&root.path().join("../outside")).is_err());
    assert!(builder.file(&root.path().join("missing")).is_err());
    assert_eq!(builder.plan.files.len(), 2);
    let source = Source { path: a, offset: 0, bytes: 16 };
    for offset in [0, 4] {
        builder.contiguous(&source, offset, 4, Destination { rank: 0, offset, bytes: 4 }).unwrap();
    }
    assert!(builder.contiguous(&source, 14, 4,
        Destination { rank: 0, offset: 8, bytes: 4 }).is_err());
    let checked = builder.finish().unwrap();
    assert_eq!(checked.plan.entries.len(), 2);
    assert!(checked.plan.entries.iter().all(|entry| entry.file == 0));
}

#[test]
fn resolved_exl3_loads_match_per_destination_plan_bytes() {
    use crate::plan::testing::{qwen4_config, qwen4_exl3, write_snapshot};
    let directory = TempDir::new().unwrap();
    let (tensors, _) = qwen4_exl3(1, 4);
    let tensors: Vec<_> = tensors.into_iter().filter(|tensor| !tensor.0.contains(".experts.")
        || tensor.0.contains(".experts.0.")).collect();
    let mut config = qwen4_config(1);
    config["text_config"]["num_experts"] = serde_json::json!(1);
    config["text_config"]["num_experts_per_tok"] = serde_json::json!(1);
    write_snapshot(directory.path(), &config, &tensors, None);
    let catalog = crate::read_expert_catalog(directory.path()).unwrap();
    for world in [2, 5] {
        let residency = catalog.exl3().unwrap()
            .balanced_residency(crate::V41Exl3Layer::Backbone(0), world, 0).unwrap();
        let (offsets, bytes) = residency.device_arena_layout().unwrap();
        let destination = Destination { rank: 0, offset: 0, bytes: bytes as u64 };
        let mut actual = Builder::new(directory.path(), vec![bytes as u64]);
        actual.exl3_residency(&catalog, &residency, &offsets, destination).unwrap();
        let mut expected = Builder::new(directory.path(), vec![bytes as u64]);
        for load in &residency.loads {
            for &(buffer, relative) in &load.destinations {
                let target = Destination { offset: (offsets[buffer] + relative) as u64,
                    bytes: load.bytes as u64, ..destination };
                if let Some(units) = &residency.unit_map {
                    expected.exl3(&catalog, &load.tensor, units, load.expert.unwrap(), target).unwrap();
                } else {
                    let slice = catalog.exl3_tensor_slice(&load.tensor, world, 0).unwrap();
                    expected.rows(&Builder::v41_source(&catalog, &load.tensor).unwrap(),
                        slice.column_start_bytes as u64, slice.rows as u32,
                        slice.selected_row_bytes as u32, slice.source_row_bytes as u64,
                        slice.selected_row_bytes as u64, target).unwrap();
                }
            }
        }
        assert_eq!(serde_json::to_vec(&actual.finish().unwrap().plan).unwrap(),
            serde_json::to_vec(&expected.finish().unwrap().plan).unwrap(), "TP{world}");
    }
}

#[test]
fn packed_balanced_columns_and_exact_scalars() {
    let root = TempDir::new().unwrap();
    let bytes = (0..192).map(|i| i as u8).collect::<Vec<_>>();
    let path = root.path().join("weights");
    std::fs::write(&path, &bytes).unwrap();
    let source = Source {
        path,
        offset: 0,
        bytes: bytes.len() as u64,
    };
    let mut builder = Builder::new(root.path(), vec![40; 6]);
    for rank in 0..6 {
        builder
            .columns(
                &source,
                3,
                64,
                &[0..4, 12..20],
                Destination {
                    rank,
                    offset: 0,
                    bytes: 36,
                },
            )
            .unwrap();
        builder
            .contiguous(
                &source,
                4,
                4,
                Destination {
                    rank,
                    offset: 36,
                    bytes: 4,
                },
            )
            .unwrap();
    }
    let plan = builder.finish().unwrap();
    assert_eq!(plan.plan.entries.len(), 12);
    assert_eq!(plan.plan.patterns.len(), 2);
    let expected = (0..3)
        .flat_map(|row| {
            bytes[row * 64..row * 64 + 4]
                .iter()
                .chain(&bytes[row * 64 + 12..row * 64 + 20])
                .copied()
        })
        .chain(bytes[4..8].iter().copied())
        .collect::<Vec<_>>();
    for arena in land(&plan, root.path()) {
        assert_eq!(arena, expected);
    }
}

#[test]
fn pitched_padding_partitions_and_namespace_bounds() {
    let root = TempDir::new().unwrap();
    let path = root.path().join("weights");
    let bytes = (0..96).map(|i| i as u8).collect::<Vec<_>>();
    std::fs::write(&path, &bytes).unwrap();
    let source = Source {
        path,
        offset: 0,
        bytes: 96,
    };
    for axis in 0..2 {
        let mut builder = Builder::new(root.path(), vec![24; 4]);
        for rank in 0..4 {
            builder
                .matrix_partition(
                    &source,
                    8,
                    axis,
                    4,
                    rank,
                    Destination {
                        rank,
                        offset: 0,
                        bytes: 24,
                    },
                )
                .unwrap();
        }
        let plan = builder.finish().unwrap();
        for (rank, arena) in land(&plan, root.path()).into_iter().enumerate() {
            let expected = if axis == 0 {
                bytes[rank * 24..rank * 24 + 24].to_vec()
            } else {
                bytes
                    .chunks_exact(12)
                    .flat_map(|r| r[rank * 3..rank * 3 + 3].iter().copied())
                    .collect()
            };
            assert_eq!(arena, expected);
        }
    }
    let mut builder = Builder::new(root.path(), vec![40]);
    builder
        .rows(
            &source,
            3,
            8,
            3,
            12,
            5,
            Destination {
                rank: 0,
                offset: 0,
                bytes: 40,
            },
        )
        .unwrap();
    let arena = land(&builder.finish().unwrap(), root.path()).remove(0);
    for (row, output) in arena.chunks_exact(5).enumerate() {
        assert_eq!(&output[..3], &bytes[row * 12 + 3..row * 12 + 6]);
        assert_eq!(&output[3..], &[0, 0]);
    }
    let mut builder = Builder::new(root.path(), vec![8]);
    assert!(builder
        .contiguous(
            &source,
            93,
            4,
            Destination {
                rank: 0,
                offset: 0,
                bytes: 4
            }
        )
        .is_err());
    let outside = Source {
        path: root.path().join("../weights"),
        ..source.clone()
    };
    assert!(builder
        .contiguous(
            &outside,
            0,
            4,
            Destination {
                rank: 0,
                offset: 0,
                bytes: 4
            }
        )
        .is_err());
    builder
        .contiguous(
            &source,
            0,
            4,
            Destination {
                rank: 0,
                offset: 0,
                bytes: 4,
            },
        )
        .unwrap();
    builder
        .contiguous(
            &source,
            4,
            4,
            Destination {
                rank: 0,
                offset: 2,
                bytes: 4,
            },
        )
        .unwrap();
    assert!(matches!(
        builder.finish(),
        Err(BuildError::Check(nest_push::plan::PlanError::Overlap(..)))
    ));
}

pub(super) fn snapshot(variable: &str) -> Option<PathBuf> {
    std::env::var_os(variable).map(PathBuf::from)
}

pub(super) fn real_pattern_headroom(plan: &CheckedPlan, label: &str) {
    let (id, pattern) = plan.plan.patterns.iter().enumerate()
        .max_by_key(|(_, p)| nest_push::proto::encode(*p).len()).unwrap();
    let bytes = nest_push::proto::encode(pattern).len();
    let indices = pattern.dims.iter().filter_map(|d| match &d.steps {
        Steps::Indices(v) => Some(v.len()), _ => None,
    }).max().unwrap_or(0);
    assert!(bytes < 100 * 1024, "{label}: real pattern {id} needs {bytes} bytes; approaching socket ceiling");
    eprintln!("real pattern headroom {label}: pattern {id}, {indices} indices, {bytes} encoded bytes");
}

#[test]
fn real_exl3_tp4_tp6_bytes_match_existing_gathers() {
    let Some(snapshot) = snapshot("CUTEAFD_PUSH_EXL3") else {
        return;
    };
    let catalog = crate::read_expert_catalog(&snapshot).unwrap();
    let shape = catalog.routed_experts();
    let layer = shape.first_layer;
    // Manifest naming differs by family; existing tensor names are canonical.
    let names = catalog
        .tensors()
        .iter()
        .filter(|t| {
            t.metadata.name.contains(&format!(".{layer}."))
                && t.metadata.name.contains(".experts.0.")
        })
        .map(|t| t.metadata.name.clone())
        .collect::<Vec<_>>();
    assert!(!names.is_empty());
    for world in [4, 6] {
        let mut jobs = Vec::new();
        let mut sizes = vec![0u64; world];
        for rank in 0..world {
            let units = cuteafd_core::tp_units::RankUnits::new(
                cuteafd_core::tp_units::balanced_rule(128, shape.intermediate / 128, world),
                shape.experts,
                shape.intermediate,
                128,
                layer,
                world,
                rank,
            )
            .unwrap();
            for name in &names {
                let gather = catalog.exl3_tensor_gather(name, &units, 0).unwrap();
                let destination = Destination {
                    rank: rank as u32,
                    offset: sizes[rank],
                    bytes: gather.bytes() as u64,
                };
                sizes[rank] += destination.bytes;
                jobs.push((units.clone(), name, destination, gather));
            }
        }
        let resident = (0..world)
            .map(|rank| {
                catalog
                    .exl3()
                    .unwrap()
                    .balanced_residency(crate::V41Exl3Layer::Backbone(layer), world, rank)
                    .unwrap()
            })
            .collect::<Vec<_>>();
        let layouts = resident
            .iter()
            .map(|r| r.device_arena_layout().unwrap())
            .collect::<Vec<_>>();
        let mut resident_builder =
            Builder::new(&snapshot, layouts.iter().map(|(_, n)| *n as u64).collect());
        for (rank, (residency, (offsets, bytes))) in resident.iter().zip(&layouts).enumerate() {
            resident_builder
                .exl3_residency(
                    &catalog,
                    residency,
                    offsets,
                    Destination {
                        rank: rank as u32,
                        offset: 0,
                        bytes: *bytes as u64,
                    },
                )
                .unwrap();
        }
        let resident_plan = resident_builder.finish().unwrap();
        real_pattern_headroom(&resident_plan, "EXL3 serving residency");
        for (rank, residency) in resident.iter().enumerate() {
            let expected = cuteafd_core::tp_units::RankUnits::new(
                cuteafd_core::tp_units::balanced_rule(128, shape.intermediate / 128, world),
                shape.experts, shape.intermediate, 128, layer, world, rank).unwrap();
            if let Some(actual) = &residency.unit_map { assert_eq!(actual, &expected); }
        }
        let entry = resident_plan.plan.entries.iter().max_by_key(|e|
            nest_push::proto::encode(&resident_plan.plan.patterns[e.pattern as usize]).len()).unwrap();
        let tensor = catalog.tensors().iter().find(|t|
            t.shard == resident_plan.plan.files[entry.file as usize]
            && entry.src_base >= t.metadata.byte_offset
            && entry.src_base < t.metadata.byte_offset + t.metadata.byte_length).unwrap();
        eprintln!("EXL3 TP{world} largest serving pattern tensor {}", tensor.metadata.name);
        eprintln!(
            "EXL3 TP{world} full layer: {} checked entries, {} patterns",
            resident_plan.plan.entries.len(),
            resident_plan.plan.patterns.len()
        );
        let mut builder = Builder::new(&snapshot, sizes);
        for (units, name, destination, _) in &jobs {
            builder
                .exl3(&catalog, name, units, 0, *destination)
                .unwrap();
        }
        let plan = builder.finish().unwrap();
            real_pattern_headroom(&plan, "native checkpoint");
        assert_eq!(plan.plan.entries.len(), jobs.len());
        let landed = land(&plan, &snapshot);
        for (units, name, target, gather) in jobs {
            let mut expected = vec![0; gather.bytes()];
            let mut scratch = vec![0; gather.scratch_bytes()];
            catalog
                .read_exl3_tensor_units_into(name, &units, 0, &mut expected, &mut scratch)
                .unwrap();
            assert_eq!(
                expected,
                landed[target.rank as usize]
                    [target.offset as usize..(target.offset + target.bytes) as usize],
                "TP{world} {name}"
            );
        }
        eprintln!(
            "EXL3 TP{world}: {} entries, {} patterns, byte-exact",
            plan.plan.entries.len(),
            plan.plan.patterns.len()
        );
    }
}

#[test]
fn real_fp8_nvfp4_tp4_tp6_bytes_match_existing_windows() {
    use crate::formats::fp8_experts::{ExpertFormat, Fp8ExpertTensors, Fp8Projection, Slicing};
    for variable in [
        "CUTEAFD_PUSH_FP8",
        "CUTEAFD_PUSH_FP8_TP6",
        "CUTEAFD_PUSH_NVFP4",
        "CUTEAFD_PUSH_MXFP4",
    ] {
        let Some(snapshot) = snapshot(variable) else {
            continue;
        };
        let config: serde_json::Value =
            serde_json::from_slice(&std::fs::read(snapshot.join("config.json")).unwrap()).unwrap();
        let config = config.get("text_config").unwrap_or(&config);
        let get = |names: &[&str]| names.iter().find_map(|n| config[n].as_u64()).unwrap() as usize;
        let shape = crate::RoutedExpertShape {
            layers: get(&["num_hidden_layers"]),
            first_layer: config["first_k_dense_replace"]
                .as_u64()
                .map(|v| v as usize)
                .or_else(|| {
                    config["moe_layer_freq"]
                        .as_array()
                        .and_then(|v| v.iter().position(|n| n.as_u64() == Some(1)))
                })
                .unwrap_or(0),
            experts: get(&["n_routed_experts", "num_experts"]),
            topk: get(&["num_experts_per_tok"]),
            hidden: get(&["hidden_size"]),
            intermediate: get(&["moe_intermediate_size"]),
            draft_stages: 0,
            draft_experts: 0,
        };
        let layer = shape.first_layer;
        let tensors = Fp8ExpertTensors::read_layers(&snapshot, shape, &[layer].into()).unwrap();
        for world in [4, 6] {
            if tensors.rank_range_with(world, 0, Slicing::Padded).is_err() {
                eprintln!("{variable} TP{world}: unsupported native block geometry");
                assert_eq!(variable, "CUTEAFD_PUSH_FP8");
                continue;
            }
            for slicing in [Slicing::Padded, tensors.exact_slicing(world)] {
                for projection in Fp8Projection::ALL {
                    let mut jobs = Vec::new();
                    let mut sizes = Vec::new();
                    for rank in 0..world {
                        let (w, matrix_scales) = tensors
                            .slice_bytes_with(projection, world, rank, slicing)
                            .unwrap();
                        let s = matrix_scales
                            + usize::from(tensors.format() == ExpertFormat::Nvfp4) * 8;
                        jobs.push((
                            Destination {
                                rank: rank as u32,
                                offset: 0,
                                bytes: w as u64,
                            },
                            Destination {
                                rank: rank as u32,
                                offset: w as u64,
                                bytes: s as u64,
                            },
                        ));
                        sizes.push((w + s) as u64);
                    }
                    let mut builder = Builder::new(&snapshot, sizes);
                    for (rank, (w, s)) in jobs.iter().enumerate() {
                        builder
                            .fp8_projection(
                                &tensors, layer, 0, projection, world, rank, slicing, *w, *s,
                            )
                            .unwrap();
                    }
                    let plan = builder.finish().unwrap();
            real_pattern_headroom(&plan, "native checkpoint");
                    let mut landed = land(&plan, &snapshot);
                    let name = tensors.name(layer, 0, projection);
                    let scale_name = format!(
                        "{name}{}",
                        if tensors.format().packed_fp4() {
                            "_scale"
                        } else {
                            "_scale_inv"
                        }
                    );
                    let (_, dtype) = tensors.push_source(&scale_name).unwrap();
                    for (rank, (w, s)) in jobs.iter().enumerate() {
                        let mut weight = vec![0; w.bytes as usize];
                        let mut scale = vec![0; s.bytes as usize];
                        let matrix_scales =
                            scale.len() - usize::from(tensors.format() == ExpertFormat::Nvfp4) * 8;
                        tensors
                            .read_slice_with(
                                layer,
                                0,
                                projection,
                                world,
                                rank,
                                slicing,
                                &mut weight,
                                &mut scale[..matrix_scales],
                                &mut Vec::new(),
                            )
                            .unwrap();
                        if dtype == cuteafd_core::DType::Bf16 {
                            let image = &mut landed[rank][w.bytes as usize..];
                            for i in (0..s.bytes as usize / 4).rev() {
                                let raw =
                                    u16::from_le_bytes(image[2 * i..2 * i + 2].try_into().unwrap());
                                image[4 * i..4 * i + 4]
                                    .copy_from_slice(&(u32::from(raw) << 16).to_le_bytes());
                            }
                        }
                        if tensors.format() == ExpertFormat::Nvfp4 {
                            let n = scale.len();
                            scale[n - 8..n - 4].copy_from_slice(
                                &tensors
                                    .read_alpha(layer, 0, projection)
                                    .unwrap()
                                    .to_le_bytes(),
                            );
                            scale[n - 4..].copy_from_slice(
                                &tensors
                                    .read_input_scale(layer, 0, projection)
                                    .unwrap()
                                    .to_le_bytes(),
                            );
                        }
                        assert_eq!(
                            weight,
                            landed[rank][..w.bytes as usize],
                            "{variable} TP{world} {rank} {projection:?} weights"
                        );
                        assert_eq!(
                            scale,
                            landed[rank][w.bytes as usize..],
                            "{variable} TP{world} {rank} {projection:?} scales"
                        );
                    }
                    eprintln!(
                        "{variable} TP{world} {slicing:?} {projection:?}: {} entries byte-exact",
                        plan.plan.entries.len()
                    );
                }
            }
        }
    }
}

#[test]
fn real_exlr8_tp4_tp6_plan_counts_and_verified_unit_bytes() {
    use crate::plan::exlr8::residency::{ExpertSet, Residency};
    let Some(snapshot) = snapshot("CUTEAFD_PUSH_EXLR8") else {
        return;
    };
    for world in [4, 6] {
        for rank in 0..world {
            let residency =
                Residency::plan(&snapshot, "up1", 3, world, rank, ExpertSet::Routed).unwrap();
            let mut sizes = vec![0; world];
            sizes[rank] = residency.resident_bytes as u64;
            let mut builder = Builder::new(&snapshot, sizes);
            builder
                .exlr8(
                    &snapshot,
                    &residency,
                    Destination {
                        rank: rank as u32,
                        offset: 0,
                        bytes: residency.resident_bytes as u64,
                    },
                )
                .unwrap();
            let plan = builder.finish().unwrap();
            real_pattern_headroom(&plan, "native checkpoint");
            eprintln!(
                "EXLR8 TP{world} rank {rank} layer 3: {} entries, {} patterns",
                plan.plan.entries.len(),
                plan.plan.patterns.len()
            );
            // Read one selected expert rather than a whole resident layer. All
            // entries still pass the full-rank bounds and disjointness gate.
            let expert = residency.loads[0].expert;
            for unit in residency.loads.iter().filter(|u| u.expert == expert) {
                let mut verified = vec![0; unit.staging_bytes()];
                unit.read_verified(&mut verified).unwrap();
                let mut cursor = 0;
                for (component, source) in unit.source.iter().enumerate() {
                    let dst = unit.destination[component] as u64;
                    let entry = plan
                        .plan
                        .entries
                        .iter()
                        .find(|e| {
                            let p = &plan.plan.patterns[e.pattern as usize];
                            e.dst_base <= dst && dst < e.dst_base + p.check().unwrap().dst_len
                        })
                        .unwrap();
                    let pattern = &plan.plan.patterns[entry.pattern as usize];
                    let run = pattern
                        .runs()
                        .into_iter()
                        .find(|(_, d)| entry.dst_base + d == dst)
                        .unwrap();
                    assert_eq!(pattern.run as usize, source.bytes);
                    let mut landed = vec![0; source.bytes];
                    File::open(snapshot.join(&plan.plan.files[entry.file as usize]))
                        .unwrap()
                        .read_exact_at(&mut landed, entry.src_base + run.0)
                        .unwrap();
                    assert_eq!(landed, verified[cursor..cursor + source.bytes]);
                    cursor += source.bytes;
                }
            }
        }
    }
}

#[test]
fn oversized_index_list_splits_without_expanding_outer_rows() {
    let root = TempDir::new().unwrap();
    let path = root.path().join("weights");
    let bytes = (0..400_000).map(|n| n as u8).collect::<Vec<_>>();
    std::fs::write(&path, &bytes).unwrap();
    let source = Source {
        path,
        offset: 0,
        bytes: bytes.len() as u64,
    };
    let pattern = Pattern {
        run: 1,
        dims: vec![Dim {
            steps: Steps::Indices((0..400_000).collect()),
            src_stride: 1,
            dst_stride: 1,
        }],
    };
    assert!(nest_push::proto::encode(&pattern).len() > 1 << 20);
    let destination = Destination {
        rank: 0,
        offset: 0,
        bytes: source.bytes,
    };
    let mut builder = Builder::new(root.path(), vec![source.bytes]);
    builder
        .entry(&source, destination, pattern.clone(), 0)
        .unwrap();
    let plan = builder.finish().unwrap();
    assert!(plan.plan.entries.len() > 1);
    assert!(plan
        .plan
        .patterns
        .iter()
        .all(|p| nest_push::proto::encode(p).len() + 16 <= nest_push::proto::MAX_MSG));
    assert_eq!(land(&plan, root.path())[0], bytes);
    let mut interleaved = pattern;
    interleaved.dims.insert(
        0,
        Dim {
            steps: Steps::Count(2),
            src_stride: source.bytes,
            dst_stride: source.bytes,
        },
    );
    let mut builder = Builder::new(root.path(), vec![2 * source.bytes]);
    let error = builder
        .entry(
            &Source {
                bytes: 2 * source.bytes,
                ..source
            },
            Destination {
                bytes: 2 * destination.bytes,
                ..destination
            },
            interleaved,
            0,
        )
        .unwrap_err();
    assert!(error.to_string().contains("multipart protocol support"));
    assert!(builder.plan.entries.is_empty(), "do not expand outer rows");
}
