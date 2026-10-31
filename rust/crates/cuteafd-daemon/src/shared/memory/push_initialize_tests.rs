use super::*;
use crate::shared::memory::weight_arena::{ArenaLayout, ExpectedCoverage, LoadingArena};
use cuteafd_ffi::native_library_lifetime_fixture::Fixture;
use cuteafd_loader::push_load::{
    builder::{Builder, Destination, Source},
    native::NativePlanes,
};

#[test]
fn calibrated_vectors_match_mount_operations_and_reject_bad_scalars() -> Result<()> {
    let image = [0.5f32, 0.5, 0.25, 2., 2., 4.]
        .into_iter()
        .flat_map(f32::to_le_bytes)
        .collect::<Vec<_>>();
    let v = nvfp4_vectors(&image, 1)?;
    assert_eq!(v, [vec![1.], vec![1.], vec![0.5], vec![0.25]]);
    for (index, value) in [
        (0, f32::NAN),
        (1, 0.75),
        (2, 0.),
        (3, f32::INFINITY),
        (4, 3.),
        (5, -1.),
    ] {
        let mut bad = image.clone();
        bad[index * 4..index * 4 + 4].copy_from_slice(&value.to_le_bytes());
        assert!(nvfp4_vectors(&bad, 1).is_err());
    }
    assert!(nvfp4_vectors(&image[..20], 1).is_err());
    Ok(())
}

fn delivered<'a>(
    device: Device<'a>,
    image: &[u8],
    destination_bytes: usize,
) -> Result<DeliveredArena<'a>> {
    let directory = tempfile::tempdir_in(std::env::var("CUTEAFD_NATIVE_LIFETIME_FIXTURE_DIR")?)?;
    let path = directory.path().join("source");
    std::fs::write(&path, image)?;
    let mut builder = Builder::new(directory.path(), vec![destination_bytes as u64]);
    builder.contiguous(
        &Source {
            path,
            offset: 0,
            bytes: image.len() as u64,
        },
        0,
        image.len() as u64,
        Destination {
            rank: 0,
            offset: 0,
            bytes: image.len() as u64,
        },
    )?;
    let plan = builder.finish()?;
    let mut loading = LoadingArena::new(
        device,
        ArenaLayout::new([destination_bytes], destination_bytes)?,
        ExpectedCoverage::from_plan(&plan, 0)?,
    )?;
    device.run(|| {
        device
            .library
            .copy_h2d(loading.destination(0, image.len())?, image)
    })?;
    loading.copied(0, image.len() as u64)?;
    loading.done()
}

#[test]
#[ignore = "requires an allocated CPU build slot and explicit NVMe fixture directory"]
fn fp8_scale_widening_and_generated_padding_are_exact() -> Result<()> {
    let fixture = Fixture::build()?;
    let library = fixture.load()?;
    let device = Device {
        library: &library,
        id: 0,
    };
    let halves = [
        0u16, 0x8000, 0x3f80, 0x7fc1, 0xffff, 1, 0x7f80, 0x0080, 0xbf80,
    ];
    let image = halves
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect::<Vec<_>>();
    let arena = delivered(device, &image, halves.len() * 4 + 16)?;
    arena.zero_padding()?;
    widen_scales(&arena, 0..halves.len() * 4, 12)?;
    let ready = arena.initialize(|_| Ok(()))?;
    let mut output = vec![0xa5; halves.len() * 4 + 16];
    library.copy_d2h(&mut output, ready.region(0)?.buffer())?;
    let expected = halves
        .iter()
        .flat_map(|v| (u32::from(*v) << 16).to_le_bytes())
        .chain([0; 16])
        .collect::<Vec<_>>();
    assert_eq!(output, expected);
    Ok(())
}

#[test]
#[ignore = "requires an allocated CPU build slot and explicit NVMe fixture directory"]
fn padded_rows_coalesce_and_failed_zero_drains_before_free() -> Result<()> {
    for (enqueue, drain) in [(0, 0), (1, 0), (1, 1)] {
        let fixture = Fixture::build()?;
        let library = fixture.load()?;
        let device = Device {
            library: &library,
            id: 0,
        };
        let directory =
            tempfile::tempdir_in(std::env::var("CUTEAFD_NATIVE_LIFETIME_FIXTURE_DIR")?)?;
        let path = directory.path().join("rows");
        std::fs::write(&path, [7; 12])?;
        let mut builder = Builder::new(directory.path(), vec![24]);
        builder.rows(
            &Source {
                path,
                offset: 0,
                bytes: 12,
            },
            0,
            4,
            3,
            3,
            6,
            Destination {
                rank: 0,
                offset: 0,
                bytes: 24,
            },
        )?;
        let plan = builder.finish()?;
        let mut loading = LoadingArena::new(
            device,
            ArenaLayout::new([24], 24)?,
            ExpectedCoverage::from_plan(&plan, 0)?,
        )?;
        for row in 0..4 {
            library.copy_h2d(loading.destination(row * 6, 3)?, &[7; 3])?;
            loading.copied((row * 6) as u64, 3)?;
        }
        let arena = loading.done()?;
        fixture.configure_pack(&library, enqueue, drain)?;
        let result = arena.zero_padding();
        assert_eq!(result.is_ok(), enqueue == 0 && drain == 0);
        if result.is_ok() {
            let mut actual = vec![0; 24];
            library.copy_d2h(&mut actual, arena.region(0)?)?;
            assert_eq!(actual, [7, 7, 7, 0, 0, 0].repeat(4));
        }
        drop(arena);
        let events = fixture.events()?;
        assert_eq!(
            events.matches('Z').count(),
            1,
            "one pitched enqueue per plane"
        );
        assert_eq!(events.matches('S').count(), 1, "one guaranteed drain");
        if drain == 0 {
            assert!(events.find('S').unwrap() < events.find('d').unwrap());
        } else {
            assert!(library.is_quarantined_after_failed_drain());
            assert!(!events.contains('d'));
        }
    }
    Ok(())
}

#[test]
#[ignore = "requires an allocated CPU build slot and explicit NVMe fixture directory"]
fn device_binding_failures_quarantine_every_post_done_drain() -> Result<()> {
    for operation in 0..3 {
        for id in 0..2 {
            let fixture = Fixture::build()?;
            let library = fixture.load()?;
            let device = Device { library: &library, id };
            let mut arena = Some(delivered(device, &[7; 4], 8)?);
            let transforms = Transforms::new(device, 256, 256)?;
            let completed = if operation == 2 { if id == 0 { 1 } else { 2 } } else { 0 };
            fixture.configure_bind_after(&library, completed)?;
            let result = match operation {
                0 => arena.as_ref().unwrap().zero_padding(),
                1 => transforms.synchronize(),
                _ => initialize(arena.take().unwrap(), &[], None, 0).map(drop),
            };
            assert!(result.is_err());
            assert!(library.is_quarantined_after_failed_drain());
            // A later successful bind must not free storage with unproved completion.
            fixture.configure_bind_after(&library, -1)?;
            drop(transforms);
            drop(arena);
            assert!(!fixture.events()?.contains('d'));
        }
    }
    Ok(())
}

#[test]
#[ignore = "requires an allocated CPU build slot and explicit NVMe fixture directory"]
fn exl3_semantics_reject_nonfinite_and_upload_exact_descriptors() -> Result<()> {
    use cuteafd_loader::{V41Exl3Layer, V41Exl3Load, V41Exl3ResidentBuffer};
    let fixture = Fixture::build()?;
    let library = fixture.load()?;
    let device = Device {
        library: &library,
        id: 0,
    };
    let residency = V41Exl3Residency {
        layer: V41Exl3Layer::Backbone(0),
        world: 1,
        rank: 0,
        intermediate_start: 0,
        experts: 1,
        intermediate: 128,
        tiers: vec![4],
        projection_counts: vec![[1; 3]],
        unit_map: None,
        buffers: vec![
            V41Exl3ResidentBuffer {
                name: "rotation".into(),
                bytes: 8,
                initial_words: vec![],
            },
            V41Exl3ResidentBuffer {
                name: "descriptor".into(),
                bytes: 16,
                initial_words: vec![1, -1, 128, 0],
            },
        ],
        loads: vec![V41Exl3Load {
            tensor: "projection.su".into(),
            expert: Some(0),
            bytes: 8,
            destinations: vec![(0, 0)],
        }],
    };
    let (_, bytes) = residency.device_arena_layout()?;
    let source = [0x3c00u16, 0x0000, 0x8000, 0x7bff]
        .into_iter()
        .flat_map(u16::to_le_bytes)
        .collect::<Vec<_>>();
    let arena = delivered(device, &source, bytes)?;
    let ready = initialize(
        arena,
        &[Layer::Exl3 {
            region: 0,
            residency: &residency,
        }],
        None,
        0,
    )?;
    let mut actual = vec![0xa5; 272];
    library.copy_d2h(&mut actual, ready.region(0)?.buffer())?;
    assert_eq!(&actual[..8], source);
    assert!(actual[8..256].iter().all(|&v| v == 0));
    assert_eq!(
        &actual[256..],
        [1i32, -1, 128, 0]
            .into_iter()
            .flat_map(i32::to_le_bytes)
            .collect::<Vec<_>>()
    );
    for invalid in [0x7c00u16, 0xfc00, 0x7e01] {
        let mut source = source.clone();
        source[..2].copy_from_slice(&invalid.to_le_bytes());
        let arena = delivered(device, &source, bytes)?;
        assert!(initialize(
            arena,
            &[Layer::Exl3 {
                region: 0,
                residency: &residency
            }],
            None,
            0
        )
        .is_err());
    }
    Ok(())
}

#[test]
#[ignore = "requires an allocated CPU build slot and explicit NVMe fixture directory"]
fn native_out_of_place_planes_match_mount_reference() -> Result<()> {
    let fixture = Fixture::build()?;
    let library = fixture.load()?;
    let device = Device {
        library: &library,
        id: 0,
    };
    for nv in [false, true] {
        for (width, stored) in [(384usize, 384), (576, 640), (576, 576)] {
            if !nv && stored == 576 {
                continue;
            }
            let layout = NativePlanes::new(5120, width, stored, 1, nv)?;
            let mut raw = vec![0u8; layout.bytes];
            for plane in 0..4 {
                let (offset, bytes) = layout.expert_plane(plane, 0)?;
                let divisor = if plane % 2 == 0 {
                    2
                } else if nv {
                    16
                } else {
                    32
                };
                let cols = if plane < 2 {
                    5120 / divisor
                } else if nv {
                    layout.kernel_intermediate / divisor
                } else {
                    width / divisor
                };
                let rows = if plane < 2 {
                    2 * layout.kernel_intermediate
                } else {
                    5120
                };
                for row in 0..rows {
                    for col in 0..cols {
                        let valid = if plane < 2 {
                            row % layout.kernel_intermediate < width
                        } else {
                            col < width / divisor
                        };
                        if valid {
                            raw[offset + row * cols + col] =
                                ((row * 37 + col * 13 + plane * 71) % 251 + 1) as u8;
                        }
                    }
                }
                assert!(rows * cols <= bytes);
            }
            let arena = delivered(device, &raw, raw.len())?;
            let scratch = Transforms::new(device, layout.scratch_bytes(), layout.scratch_bytes())?;
            let scalars = if nv {
                [0.5f32, 0.5, 0.25, 2., 2., 4.]
                    .into_iter()
                    .flat_map(f32::to_le_bytes)
                    .collect::<Vec<_>>()
            } else {
                vec![]
            };
            scratch.native(&arena, 0, &layout, &scalars)?;
            let mut actual = vec![0; raw.len()];
            library.copy_d2h(&mut actual, arena.region(0)?)?;
            for plane in 0..4 {
                let (offset, bytes) = layout.expert_plane(plane, 0)?;
                let expected = if nv && plane % 2 == 0 {
                    raw[offset..offset + bytes].to_vec()
                } else if nv {
                    let (rows, cols) = if plane == 1 {
                        (2 * layout.kernel_intermediate, 320)
                    } else {
                        (5120, layout.kernel_intermediate / 16)
                    };
                    let mut out = vec![0; bytes];
                    for row in 0..rows {
                        for col in 0..cols {
                            let at = (row / 128) * cols * 128
                                + (col / 4) * 512
                                + (row % 32) * 16
                                + ((row / 32) % 4) * 4
                                + col % 4;
                            out[at] = raw[offset + row * cols + col];
                        }
                    }
                    out
                } else {
                    reference_mxfp4(
                        &raw[offset..offset + bytes],
                        plane,
                        width,
                        layout.kernel_intermediate,
                        bytes,
                    )
                };
                assert_eq!(
                    &actual[offset..offset + bytes],
                    expected,
                    "nv={nv} width={width} plane={plane}"
                );
            }
            if nv {
                for ((offset, _), value) in layout.vectors.iter().zip([1f32, 1., 0.5, 0.25]) {
                    assert_eq!(&actual[*offset..*offset + 4], value.to_le_bytes());
                }
            }
            drop(scratch);
        }
    }
    Ok(())
}

#[test]
#[ignore = "requires an allocated CPU build slot and explicit NVMe fixture directory"]
fn native_transform_failures_drain_or_quarantine_before_free() -> Result<()> {
    for drain in [0, 1] {
        let fixture = Fixture::build()?;
        let library = fixture.load()?;
        let device = Device {
            library: &library,
            id: 0,
        };
        let layout = NativePlanes::new(5120, 384, 384, 1, false)?;
        let arena = delivered(device, &vec![0; layout.bytes], layout.bytes)?;
        let transforms = Transforms::new(device, layout.scratch_bytes(), layout.scratch_bytes())?;
        fixture.configure_pack(&library, 1, drain)?;
        assert!(transforms.native(&arena, 0, &layout, &[]).is_err());
        drop(arena);
        drop(transforms);
        let events = fixture.events()?;
        assert!(events.contains('Q') && events.contains('S'));
        if drain != 0 {
            assert!(library.is_quarantined_after_failed_drain());
            assert!(!events.contains('d') && !events.contains('T'));
        } else {
            assert!(!library.is_quarantined_after_failed_drain());
            assert!(events.find('S').unwrap() < events.find('d').unwrap());
            assert_eq!(events.matches('d').count(), 2);
        }
    }
    Ok(())
}

#[test]
#[ignore = "requires an allocated CPU build slot and explicit NVMe fixture directory"]
fn exlr8_relocated_and_compact_descriptors_match_mount() -> Result<()> {
    use cuteafd_loader::plan::exlr8::residency::{ExpertSet, Residency};
    let fixture = Fixture::build()?;
    let library = fixture.load()?;
    let device = Device {
        library: &library,
        id: 0,
    };
    for set in [ExpertSet::Routed, ExpertSet::Shared] {
        let residency = Residency {
            layer: 0,
            world: 1,
            rank: 0,
            hidden: 256,
            experts: 1,
            units: 1,
            set,
            set_id: String::new(),
            manifest: String::new(),
            loads: vec![],
            signs: [vec![1; 256], vec![255; 256]],
            signs_offsets: [0, 256],
            pointers_offset: 512,
            bits_offset: 768,
            compact_offsets: [1024, 1280],
            resident_bytes: 1536,
            payload_bytes: 0,
            staging_bytes: 0,
        };
        let arena = delivered(device, &[1; 512], residency.resident_bytes)?;
        let base = arena.region(0)?.ptr as u64;
        let (pointers, bits) = residency.descriptors(base)?;
        let ready = initialize(
            arena,
            &[Layer::Exlr8 {
                region: 0,
                residency: &residency,
            }],
            None,
            0,
        )?;
        let mut actual = vec![0; residency.resident_bytes];
        library.copy_d2h(&mut actual, ready.region(0)?.buffer())?;
        assert_eq!(
            &actual[512..512 + pointers.len() * 8],
            pointers
                .into_iter()
                .flat_map(u64::to_le_bytes)
                .collect::<Vec<_>>()
        );
        assert_eq!(
            &actual[768..768 + bits.len() * 4],
            bits.into_iter()
                .flat_map(i32::to_le_bytes)
                .collect::<Vec<_>>()
        );
        if set == ExpertSet::Routed {
            assert_eq!(&actual[1024..1028], 1i32.to_le_bytes());
        } else {
            assert!(actual[1024..].iter().all(|&v| v == 0));
        }
    }
    Ok(())
}

#[test]
#[ignore = "requires an allocated CPU build slot and explicit NVMe fixture directory"]
fn fp8_header_plans_widen_pad_and_validate_like_mount() -> Result<()> {
    use cuteafd_loader::{
        formats::fp8_experts::{ExpertFormat, Fp8ExpertTensors, Fp8Projection, Slicing},
        plan::testing::{fp8, nvfp4, write_snapshot},
        RoutedExpertShape,
    };
    use std::os::unix::fs::FileExt;
    let fixture = Fixture::build()?;
    let library = fixture.load()?;
    let device = Device {
        library: &library,
        id: 0,
    };
    for nv in [false, true] {
        let directory =
            tempfile::tempdir_in(std::env::var("CUTEAFD_NATIVE_LIFETIME_FIXTURE_DIR")?)?;
        let shape = RoutedExpertShape {
            layers: 1,
            first_layer: 0,
            experts: 1,
            topk: 1,
            hidden: 256,
            intermediate: 640,
            draft_stages: 0,
            draft_experts: 0,
        };
        let mut metadata = Vec::new();
        for (name, n, k) in [("gate", 640, 256), ("up", 640, 256), ("down", 256, 640)] {
            let name = format!("model.layers.0.mlp.experts.0.{name}_proj");
            let mut values = if nv {
                nvfp4(&name, n, k)
            } else {
                fp8(&name, n, k, None)
            };
            if !nv {
                values[1].1 = "BF16";
            }
            metadata.extend(values);
        }
        write_snapshot(directory.path(), &serde_json::json!({}), &metadata, None);
        let tensors = Fp8ExpertTensors::read(directory.path(), shape)?;
        for (name, _, _) in metadata {
            let (source, _) = tensors.push_source(&name)?;
            let bytes = if name.ends_with("_scale_2") || name.ends_with(".input_scale") {
                0.25f32.to_le_bytes().to_vec()
            } else {
                (0..source.bytes)
                    .map(|i| ((i * 31 + 17) % 251) as u8)
                    .collect()
            };
            std::fs::OpenOptions::new()
                .write(true)
                .open(source.path)?
                .write_all_at(&bytes, source.offset)?;
        }
        assert_eq!(tensors.format() == ExpertFormat::Nvfp4, nv);
        for rank in 0..2 {
            let mut cursor = 0;
            let mut scales = Vec::new();
            let mut jobs = Vec::new();
            for projection in Fp8Projection::ALL {
                let (w, s) = tensors.slice_bytes_with(projection, 2, rank, Slicing::Padded)?;
                jobs.push((projection, cursor, w, s));
                scales.push(cursor + w..cursor + w + s + usize::from(nv) * 8);
                cursor += w + s + usize::from(nv) * 8;
            }
            let recipe = Layer::fp8(
                &tensors,
                0,
                0,
                2,
                rank,
                Slicing::Padded,
                scales.clone().try_into().unwrap(),
            )?;
            let mut builder = Builder::new(directory.path(), vec![cursor as u64]);
            let mut expected = vec![0; cursor];
            for ((projection, offset, w, s), scale) in jobs.iter().zip(&scales) {
                builder.fp8_projection(
                    &tensors,
                    0,
                    0,
                    *projection,
                    2,
                    rank,
                    Slicing::Padded,
                    Destination {
                        rank: 0,
                        offset: *offset as u64,
                        bytes: *w as u64,
                    },
                    Destination {
                        rank: 0,
                        offset: scale.start as u64,
                        bytes: scale.len() as u64,
                    },
                )?;
                let (before, after) = expected[*offset..*offset + w + s].split_at_mut(*w);
                tensors.read_slice_with(
                    0,
                    0,
                    *projection,
                    2,
                    rank,
                    Slicing::Padded,
                    before,
                    after,
                    &mut Vec::new(),
                )?;
                if nv {
                    expected[scale.start + s..scale.end].copy_from_slice(
                        &[0.25f32; 2]
                            .into_iter()
                            .flat_map(f32::to_le_bytes)
                            .collect::<Vec<_>>(),
                    );
                }
            }
            let plan = builder.finish()?;
            let mut arena = LoadingArena::new(
                device,
                ArenaLayout::new([cursor], cursor)?,
                ExpectedCoverage::from_plan(&plan, 0)?,
            )?;
            // Deliberately reverse delivery to exercise zero generation without
            // depending on source order or zero-filled fake device allocations.
            for entry in plan.plan.entries.iter().rev() {
                let file = std::fs::File::open(
                    directory.path().join(&plan.plan.files[entry.file as usize]),
                )?;
                let pattern = &plan.plan.patterns[entry.pattern as usize];
                for (src, dst) in pattern.runs().into_iter().rev() {
                    let mut bytes = vec![0; pattern.run as usize];
                    file.read_exact_at(&mut bytes, entry.src_base + src)?;
                    library.copy_h2d(
                        arena.destination(usize::try_from(entry.dst_base + dst)?, bytes.len())?,
                        &bytes,
                    )?;
                    arena.copied(entry.dst_base + dst, bytes.len() as u64)?;
                }
            }
            let ready = initialize(arena.done()?, &[recipe], None, 0)?;
            let mut actual = vec![0; cursor];
            library.copy_d2h(&mut actual, ready.region(0)?.buffer())?;
            assert_eq!(actual, expected, "nv={nv} rank={rank}");
        }
    }
    for value in [0., -1., f32::NAN, f32::INFINITY] {
        let arena = delivered(device, &value.to_le_bytes(), 4)?;
        assert!(initialize(
            arena,
            &[Layer::Fp8 {
                widen: vec![],
                scalars: vec![0..4],
                inputs: None,
                expand: vec![],
            }],
            None,
            0
        )
        .is_err());
    }
    Ok(())
}

// Independent forward (row,col)->tile/lane oracle, unlike the native inverse
// mapping from a packed lane index. Every valid raw word writes one final word.
fn reference_mxfp4(raw: &[u8], plane: usize, width: usize, padded: usize, bytes: usize) -> Vec<u8> {
    let scales = plane % 2 == 1;
    let element = if scales { 1 } else { 4 };
    let divisor = if scales { 32 } else { 8 };
    let rows = if plane < 2 { 2 * padded } else { 5120 };
    let k = if plane < 2 { 5120 } else { width };
    let cols = k / divisor;
    let kt = k.div_ceil(128);
    let mut output = vec![0; bytes];
    for row in 0..rows {
        for col in 0..cols {
            let tile = row / 256 * kt + col / (128 / divisor);
            let local_row = row % 256;
            let local_col = col % (128 / divisor);
            let lane = if scales {
                local_row * 4 + local_col
            } else {
                ((local_row % 32) / 8)
                    + ((local_row % 8 * 4 + local_col % 4) << 2)
                    + ((local_row / 32) << 7)
                    + ((local_col / 4) << 10)
            };
            let target = (tile * if scales { 1024 } else { 4096 } + lane) * element;
            let source = (row * cols + col) * element;
            output[target..target + element].copy_from_slice(&raw[source..source + element]);
        }
    }
    output
}
