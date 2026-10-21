//! Coordinator (RTX TP2) push plans on installed checkpoint headers: the two
//! coordinator GPUs are two ranks after the Spark ranks of one group, and their
//! EXL3 expert halves use the coordinator's legacy TP2 residency, not the
//! Sparks' balanced one.
use super::tests::{land, real_pattern_headroom, snapshot};
use super::*;

#[test]
fn real_exl3_coordinator_tp2_halves_match_mount_staging() {
    let Some(root) = snapshot("CUTEAFD_PUSH_EXL3") else { return; };
    let catalog = crate::read_expert_catalog(&root).unwrap();
    let manifest = catalog.exl3().unwrap();
    let layer = catalog.routed_experts().first_layer;
    // Spark ranks first, coordinator GPUs last: rank ids S and S + 1.
    let sparks = 6u32;
    let residencies = (0..2).map(|rank| manifest.residency(crate::V41Exl3Layer::Backbone(layer), 2, rank).unwrap())
        .collect::<Vec<_>>();
    assert!(residencies.iter().all(|r| r.unit_map.is_none()), "coordinator TP2 keeps the legacy residency");
    let layouts = residencies.iter().map(|r| r.device_arena_layout().unwrap()).collect::<Vec<_>>();
    let mut arenas = vec![0u64; sparks as usize];
    arenas.extend(layouts.iter().map(|(_, bytes)| *bytes as u64));
    let mut builder = Builder::new(&root, arenas);
    for (gpu, (residency, (offsets, bytes))) in residencies.iter().zip(&layouts).enumerate() {
        builder.exl3_residency(&catalog, residency, offsets, Destination {
            rank: sparks + gpu as u32, offset: 0, bytes: *bytes as u64 }).unwrap();
    }
    let plan = builder.finish().unwrap();
    real_pattern_headroom(&plan, "EXL3 coordinator TP2");
    assert!(plan.plan.entries.iter().all(|e| e.rank >= sparks));
    let landed = land(&plan, &root);
    for (gpu, (residency, (offsets, _))) in residencies.iter().zip(&layouts).enumerate() {
        let arena = &landed[sparks as usize + gpu];
        let mut scratch = vec![0; residency.scratch_bytes(&catalog).unwrap()];
        for (index, load) in residency.loads.iter().enumerate() {
            let mut expected = vec![0; load.bytes];
            residency.read_into(&catalog, index, &mut expected, &mut scratch).unwrap();
            for &(buffer, offset) in &load.destinations {
                let start = offsets[buffer] + offset;
                assert_eq!(&arena[start..start + load.bytes], &expected[..],
                    "coordinator TP2 rank {gpu} {} buffer {buffer}", load.tensor);
            }
        }
    }
    eprintln!("EXL3 coordinator TP2 layer {layer}: {} entries, {} patterns, byte-exact",
        plan.plan.entries.len(), plan.plan.patterns.len());
}
