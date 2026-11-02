use super::tests::{land, snapshot, real_pattern_headroom};
use super::*;

#[test]
fn real_native_resident_planes_match_mount_staging_after_plan_fusion() {
    for (variable, nv) in [("CUTEAFD_PUSH_V41", false), ("CUTEAFD_PUSH_V41_NVFP4", true)] {
        let Some(root) = snapshot(variable) else { continue; };
        let catalog = crate::read_official_v41_catalog(&root).unwrap();
        for world in [1usize, 4, 6] {
            for rank in 0..world {
                let selection = if world == 1 { crate::V41ExpertSelection::BackboneFull { layer: 0, expert: 0 } }
                    else { crate::V41ExpertSelection::BackboneTp { layer: 0, expert: 0, world, rank } };
                let (names, ranges, raw, width) = if nv {
                    let staging = catalog.nvfp4_expert_staging(selection).unwrap();
                    let mut raw = vec![0; staging.staging_bytes()];
                    staging.read_into(&mut raw, &mut vec![0; staging.minimum_read_scratch_bytes()]).unwrap();
                    (staging.tensor_names().to_vec(), staging.tensor_ranges().to_vec(), raw, staging.intermediate_size())
                } else {
                    let staging = catalog.expert_staging(selection).unwrap();
                    let mut raw = vec![0; staging.staging_bytes()];
                    staging.read_into(&mut raw, &mut vec![0; staging.minimum_read_scratch_bytes()]).unwrap();
                    (staging.tensor_names().to_vec(), staging.tensor_ranges().to_vec(), raw, staging.intermediate_size())
                };
                let hidden = catalog.config().text().hidden_size;
                let stored = if nv { width.div_ceil(128) * 128 } else {
                    cuteafd_core::mxfp4_layouts::stored_width(hidden, width).unwrap()
                };
                let layout = super::super::native::NativePlanes::new(hidden, width, stored, 1, nv).unwrap();
                let total = layout.bytes + layout.scalar_bytes();
                let mut builder = Builder::new(&root, vec![total as u64]);
                builder.v41_resident(&catalog, &names, &layout, 0, world as u32, rank as u32,
                    Destination { rank: 0, offset: 0, bytes: layout.bytes as u64 },
                    nv.then_some(Destination { rank: 0, offset: layout.bytes as u64, bytes: layout.scalar_bytes() as u64 })).unwrap();
                let plan = builder.finish().unwrap();
                real_pattern_headroom(&plan, "native final-plane placement");
                let landed = land(&plan, &root).remove(0);
                let mut expected = vec![0; total];
                for slot in 0..6 {
                    let down = names[slot].contains(".w2.");
                    let gate = names[slot].contains(".w1.");
                    let scale = slot >= 3;
                    let plane = usize::from(down) * 2 + usize::from(scale);
                    let divisor = if scale { if nv { 16 } else { 32 } } else { 2 };
                    let (offset, _) = layout.expert_plane(plane, 0).unwrap();
                    let half = if gate { layout.kernel_intermediate * hidden / divisor } else { 0 };
                    let row = if down { width / divisor } else { hidden / divisor };
                    let pitch = if down && nv { layout.kernel_intermediate / divisor } else { row };
                    for (index, source) in raw[ranges[slot].clone()].chunks_exact(row).enumerate() {
                        let start = offset + half + index * pitch;
                        expected[start..start + row].copy_from_slice(source);
                    }
                }
                if nv { for slot in 0..6 {
                    expected[layout.bytes + slot * 4..layout.bytes + slot * 4 + 4]
                        .copy_from_slice(&raw[ranges[6 + slot].clone()]);
                }}
                assert_eq!(landed, expected, "nv={nv} TP{world} rank={rank}");
                assert_eq!(plan.plan.entries.len(), if nv { 12 } else { 6 });
                eprintln!("native nv={nv} TP{world} rank={rank}: scratch={} bytes, final-plane image byte exact", layout.scratch_bytes());
            }
        }
    }
}

// Independent inverse map from the six-source mount packer. The plane path
// below uses the forward tile map over the fused, independently padded halves.
fn mount_packed(raw: &[u8], ranges: &[std::ops::Range<usize>], plane: usize,
    layout: &super::super::native::NativePlanes) -> Vec<u8> {
    let scales = plane % 2 == 1;
    let element = if scales { 1 } else { 4 };
    let divisor = if scales { 32 } else { 8 };
    let k = if plane < 2 { layout.hidden } else { layout.intermediate };
    let n = if plane < 2 { layout.intermediate } else { layout.hidden };
    let mut output = vec![0; layout.planes[plane].1];
    for (index, destination) in output.chunks_exact_mut(element).enumerate() {
        if layout.kernel_intermediate == layout.intermediate && layout.intermediate % 128 != 0 {
            let columns = k / divisor;
            let half_elements = layout.kernel_intermediate * columns;
            let half = if plane < 2 { index / half_elements } else { 0 };
            let index = if plane < 2 { index % half_elements } else { index };
            let nt = index / (128 * columns);
            let index = index % (128 * columns);
            let rows = (n - nt * 128).min(128);
            let block = if scales { 4 } else { 16 };
            let kt = index / (rows * block);
            let local = index % (rows * block);
            let cols = (columns - kt * block).min(block);
            let (row, col) = if scales { (local / cols, local % cols) } else {
                let n32 = rows / 32;
                (((local / 128) % n32) * 32 + (local % 4) * 8 + (local / 16) % 8,
                    local / (128 * n32) * 4 + (local / 4) % 4)
            };
            let slot = (if plane < 2 { if half == 0 { 1 } else { 0 } } else { 2 })
                + if scales { 3 } else { 0 };
            let source = ranges[slot].start + ((nt * 128 + row) * columns + kt * block + col) * element;
            destination.copy_from_slice(&raw[source..source + element]);
            continue;
        }
        let tile_elements = if scales { 1024 } else { 4096 };
        let tile = index / tile_elements;
        let local = index % tile_elements;
        let nt = tile / k.div_ceil(128);
        let kt = tile % k.div_ceil(128);
        let (mut row, col) = if scales { (nt * 256 + local / 4, kt * 4 + local % 4) }
            else { (nt * 256 + ((local >> 7) & 7) * 32 + (local & 3) * 8
                + (((local >> 2) & 31) >> 2), kt * 16 + (local >> 10) * 4 + ((local >> 2) & 3)) };
        let mut slot = if plane < 2 { 1 } else { 2 };
        if plane < 2 && row >= layout.kernel_intermediate { row -= layout.kernel_intermediate; slot = 0; }
        if scales { slot += 3; }
        if row < n && col < k / divisor {
            let start = ranges[slot].start + (row * (k / divisor) + col) * element;
            destination.copy_from_slice(&raw[start..start + element]);
        }
    }
    output
}

fn plane_packed(raw: &[u8], plane: usize,
    layout: &super::super::native::NativePlanes) -> Vec<u8> {
    let scales = plane % 2 == 1;
    let element = if scales { 1 } else { 4 };
    let divisor = if scales { 32 } else { 8 };
    let rows = if plane < 2 { 2 * layout.kernel_intermediate } else { layout.hidden };
    let k = if plane < 2 { layout.hidden } else { layout.intermediate };
    let cols = k / divisor;
    let mut output = vec![0; raw.len()];
    for row in 0..rows { for col in 0..cols {
        if layout.kernel_intermediate == layout.intermediate && layout.intermediate % 128 != 0 {
            let half = if plane < 2 { layout.kernel_intermediate } else { rows };
            let half_index = row / half;
            let local_row = row % half;
            let nt = local_row / 128;
            let tile_rows = (half - nt * 128).min(128);
            let block = if scales { 4 } else { 16 };
            let kt = col / block;
            let tile_cols = (cols - kt * block).min(block);
            let (r, c) = (local_row % 128, col % block);
            // Flatten the independent [K32, N32, N8, K4, N4] tensor permutation.
            let lane = if scales { r * tile_cols + c } else {
                ((((c / 4 * (tile_rows / 32) + r / 32) * 8 + r % 8) * 4
                    + c % 4) * 4) + (r % 32) / 8
            };
            let target = (half_index * half * cols + nt * 128 * cols
                + kt * tile_rows * block + lane) * element;
            let source = (row * cols + col) * element;
            output[target..target + element].copy_from_slice(&raw[source..source + element]);
            continue;
        }
        let tile = row / 256 * k.div_ceil(128) + col / (128 / divisor);
        let r = row % 256;
        let c = col % (128 / divisor);
        let lane = if scales { r * 4 + c } else {
            (r % 32) / 8 + ((r % 8 * 4 + c % 4) << 2) + ((r / 32) << 7) + ((c / 4) << 10)
        };
        let target = (tile * if scales { 1024 } else { 4096 } + lane) * element;
        let source = (row * cols + col) * element;
        output[target..target + element].copy_from_slice(&raw[source..source + element]);
    }}
    output
}

fn fused_mount(raw: &[u8], ranges: &[std::ops::Range<usize>],
    layout: &super::super::native::NativePlanes) -> Vec<u8> {
    let mut expected = vec![0; layout.bytes];
    for slot in 0..6 {
        let down = slot % 3 == 2;
        let gate = slot % 3 == 0;
        let divisor = if slot >= 3 { 32 } else { 2 };
        let plane = usize::from(down) * 2 + usize::from(slot >= 3);
        let (offset, _) = layout.expert_plane(plane, 0).unwrap();
        let half = if gate { layout.kernel_intermediate * layout.hidden / divisor } else { 0 };
        expected[offset + half..offset + half + ranges[slot].len()].copy_from_slice(&raw[ranges[slot].clone()]);
    }
    expected
}

#[test]
fn native_mount_and_plane_pack_maps_are_byte_exact() {
    for width in [384, 576] {
        let stored = cuteafd_core::mxfp4_layouts::stored_width(5120, width).unwrap();
        let layout = super::super::native::NativePlanes::new(5120, width, stored, 1, false).unwrap();
        let mut raw = Vec::new();
        let mut ranges = Vec::new();
        for slot in 0..6 {
            let start = raw.len();
            let bytes = 5120 * width / if slot >= 3 { 32 } else { 2 };
            raw.extend((0..bytes).map(|i| ((i * 37 + slot * 53) % 255 + 1) as u8));
            ranges.push(start..raw.len());
        }
        let fused = fused_mount(&raw, &ranges, &layout);
        for plane in 0..4 {
            let (offset, bytes) = layout.expert_plane(plane, 0).unwrap();
            let mount = mount_packed(&raw, &ranges, plane, &layout);
            let pushed = plane_packed(&fused[offset..offset + bytes], plane, &layout);
            assert_eq!(mount, pushed, "width {width} plane {plane}");
            assert!(mount.iter().any(|&b| b != 0), "source not exercised");
            if stored == width {
                assert!(mount.iter().all(|&b| b != 0), "exact plane has a byte hole");
            }
        }
    }
}

#[test]
#[ignore = "requires an allocated CPU build slot and explicit NVMe fixture directory"]
fn live_native_tp4_sample_packed_planes_match_mount() {
    let root = snapshot("CUTEAFD_PUSH_V41").expect("explicit live checkpoint required");
    let catalog = crate::read_official_v41_catalog(&root).unwrap();
    assert!(catalog.nvfp4().is_none());
    let width = catalog.config().text().moe_intermediate_size / 4;
    let stored = cuteafd_core::mxfp4_layouts::stored_width(5120, width).unwrap();
    let layout = super::super::native::NativePlanes::new(5120, width, stored, 1, false).unwrap();
    let mut seed = 57320261025u64;
    let mut checked = 0;
    for layer in 19..40 {
        let mut experts = vec![0, 1, 191, 382, 383];
        while experts.len() < 8 {
            seed ^= seed << 13; seed ^= seed >> 7; seed ^= seed << 17;
            let expert = seed as usize % 384;
            if !experts.contains(&expert) { experts.push(expert); }
        }
        for rank in 0..4 { for &expert in &experts {
            let selection = crate::V41ExpertSelection::BackboneTp { layer, expert, world: 4, rank };
            let staging = catalog.expert_staging(selection).unwrap();
            let mut raw = vec![0; staging.staging_bytes()];
            staging.read_into(&mut raw, &mut vec![0; staging.minimum_read_scratch_bytes()]).unwrap();
            let mut builder = Builder::new(&root, vec![layout.bytes as u64]);
            builder.v41_resident(&catalog, staging.tensor_names(), &layout, 0, 4, rank as u32,
                Destination { rank: 0, offset: 0, bytes: layout.bytes as u64 }, None).unwrap();
            let landed = land(&builder.finish().unwrap(), &root).remove(0);
            assert_eq!(landed, fused_mount(&raw, staging.tensor_ranges(), &layout),
                "source image layer{layer} rank{rank} expert{expert}");
            for plane in 0..4 {
                let (offset, bytes) = layout.expert_plane(plane, 0).unwrap();
                assert_eq!(mount_packed(&raw, staging.tensor_ranges(), plane, &layout),
                    plane_packed(&landed[offset..offset + bytes], plane, &layout),
                    "packed layer{layer} rank{rank} expert{expert} plane{plane}");
            }
            checked += 1;
        }
        eprintln!("live MXFP4 layer={layer} rank={rank}: experts={experts:?}, four packed planes byte exact");
        }
    }
    assert_eq!(checked, 4 * 21 * 8);
    eprintln!("live MXFP4 sampled packed byte identity: {checked} experts, {} bytes, no launch vectors in MXFP4", checked * layout.bytes);
}

#[test]
fn real_v41_native_tp4_tp6_and_coordinator_tp2_match_staging() {
    let Some(snapshot) = snapshot("CUTEAFD_PUSH_V41") else {
        return;
    };
    let catalog = crate::read_official_v41_catalog(&snapshot).unwrap();
    for world in [4, 6] {
        let mut sizes = Vec::new();
        let mut staging = Vec::new();
        for rank in 0..world {
            let plan = catalog
                .expert_staging(crate::V41ExpertSelection::BackboneTp {
                    layer: 0,
                    expert: 0,
                    world,
                    rank,
                })
                .unwrap();
            sizes.push(plan.staging_bytes() as u64);
            staging.push(plan);
        }
        let mut builder = Builder::new(&snapshot, sizes.clone());
        for (rank, plan) in staging.iter().enumerate() {
            builder
                .v41_regions(
                    &catalog,
                    plan.tensor_names(),
                    plan.tensor_ranges(),
                    world as u32,
                    rank as u32,
                    Destination {
                        rank: rank as u32,
                        offset: 0,
                        bytes: sizes[rank],
                    },
                )
                .unwrap();
        }
        let plan = builder.finish().unwrap();
        real_pattern_headroom(&plan, "V4.1 native / coordinator");
        let landed = land(&plan, &snapshot);
        for (rank, staging) in staging.iter().enumerate() {
            let mut expected = vec![0; staging.staging_bytes()];
            let mut scratch = vec![0; staging.minimum_read_scratch_bytes()];
            staging.read_into(&mut expected, &mut scratch).unwrap();
            assert_eq!(expected, landed[rank], "V41 MXFP4 TP{world} rank {rank}");
        }
        eprintln!(
            "V41 MXFP4 TP{world}: {} entries byte-exact",
            plan.plan.entries.len()
        );
    }
    let tensor = catalog
        .tensors()
        .iter()
        .find(|t| {
            matches!(t.placement, crate::V41TensorPlacement::CoordinatorRtx)
                && t.metadata.shape.len() == 2
                && t.metadata.shape.iter().all(|n| n % 2 == 0)
                && t.metadata.byte_length < 64 << 20
        })
        .unwrap();
    let name = &tensor.metadata.name;
    let source = Builder::v41_source(&catalog, name).unwrap();
    for axis in 0..2 {
        let mut builder = Builder::new(&snapshot, vec![source.bytes / 2; 2]);
        for rank in 0..2 {
            builder
                .matrix_partition(
                    &source,
                    tensor.metadata.shape[0] as u32,
                    axis,
                    2,
                    rank,
                    Destination {
                        rank,
                        offset: 0,
                        bytes: source.bytes / 2,
                    },
                )
                .unwrap();
        }
        let plan = builder.finish().unwrap();
        real_pattern_headroom(&plan, "V4.1 native / coordinator");
        for (rank, landed) in land(&plan, &snapshot).iter().enumerate() {
            let mut expected = vec![0; landed.len()];
            let mut scratch = vec![0; source.bytes as usize / tensor.metadata.shape[0]];
            catalog
                .read_coordinator_tp2_into(name, axis, rank, &mut expected, &mut scratch)
                .unwrap();
            assert_eq!(
                &expected, landed,
                "coordinator {name} TP2 axis{axis} rank{rank}"
            );
        }
    }
    let mut builder = Builder::new(&snapshot, vec![source.bytes]);
    builder
        .contiguous(
            &source,
            0,
            source.bytes,
            Destination {
                rank: 0,
                offset: 0,
                bytes: source.bytes,
            },
        )
        .unwrap();
    let plan = builder.finish().unwrap();
    real_pattern_headroom(&plan, "coordinator whole");
    let landed = land(&plan, &snapshot).remove(0);
    let mut expected = vec![0; source.bytes as usize];
    catalog.read_projection_once(name, &mut expected).unwrap();
    assert_eq!(landed, expected);
    eprintln!("Coordinator whole and TP2 axes: byte-exact");
}

#[test]
fn real_v41_nvfp4_tp4_tp6_match_staging() {
    let Some(snapshot) = snapshot("CUTEAFD_PUSH_V41_NVFP4") else {
        return;
    };
    let catalog = crate::read_official_v41_catalog(&snapshot).unwrap();
    for world in [4, 6] {
        let staging = (0..world)
            .map(|rank| {
                catalog
                    .nvfp4_expert_staging(crate::V41ExpertSelection::BackboneTp {
                        layer: 0,
                        expert: 0,
                        world,
                        rank,
                    })
                    .unwrap()
            })
            .collect::<Vec<_>>();
        let sizes = staging
            .iter()
            .map(|s| s.staging_bytes() as u64)
            .collect::<Vec<_>>();
        let mut builder = Builder::new(&snapshot, sizes.clone());
        for (rank, s) in staging.iter().enumerate() {
            builder
                .v41_regions(
                    &catalog,
                    s.tensor_names(),
                    s.tensor_ranges(),
                    world as u32,
                    rank as u32,
                    Destination {
                        rank: rank as u32,
                        offset: 0,
                        bytes: sizes[rank],
                    },
                )
                .unwrap();
        }
        let plan = builder.finish().unwrap();
        real_pattern_headroom(&plan, "V4.1 native / coordinator");
        for (s, landed) in staging.iter().zip(land(&plan, &snapshot)) {
            let mut expected = vec![0; s.staging_bytes()];
            let mut scratch = vec![0; s.minimum_read_scratch_bytes()];
            s.read_into(&mut expected, &mut scratch).unwrap();
            assert_eq!(landed, expected);
        }
        eprintln!(
            "V41 NVFP4 TP{world}: {} entries byte-exact",
            plan.plan.entries.len()
        );
    }
}
