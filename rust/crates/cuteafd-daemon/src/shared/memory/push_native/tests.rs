use super::*;
use std::os::unix::fs::FileExt;

/// Replay real shard offsets without reading weight payloads or allocating a
/// device arena. Each completed scale step contributes its entire code/base
/// write footprint; padding is initialized separately, never counted as a job.
#[test]
fn real_1017_coded_scale_plan_file_order_has_bounded_staging_and_no_holes() -> Result<()> {
    use cuteafd_ffi::push_ring::{DEFAULT_SLOT_BYTES, DEFAULT_SLOTS};
    use cuteafd_loader::push_load::{plan::{Entry, FileEntries, RunWalk, Steps}, slot};
    use std::collections::{BTreeMap, BTreeSet, VecDeque};

    let Some(snapshot) = std::env::var_os("CUTEAFD_SCALE_CODES_LAYOUT_SNAPSHOT") else { return Ok(()); };
    let snapshot = std::path::PathBuf::from(snapshot);
    let catalog = cuteafd_loader::read_official_v41_catalog(&snapshot)?;
    let shape = catalog.routed_experts();
    ensure!(catalog.mxfp4_scale_codes() && catalog.nvfp4().is_none(), "requires coded native V4.1");
    assert_eq!((shape.hidden, shape.intermediate, shape.experts, shape.layers), (5120, 2304, 384, 40));
    let budget = 512usize << 20;

    struct Plane {
        rows: Vec<u8>,
        columns: usize,
        bytes: usize,
    }
    impl Plane {
        fn new(hidden: usize, logical: usize, stored: usize, down: bool) -> Self {
            let mut rows = vec![0; if down { hidden } else { 2 * stored }];
            if !down {
                rows[logical..stored].fill(2);
                rows[stored + logical..].fill(2);
            }
            let columns = if down { logical / 32 } else { hidden / 32 };
            let bytes = rows.len() * (1 + 2 * columns.div_ceil(4));
            Self { rows, columns, bytes }
        }
        fn write(&mut self, first: usize, count: usize) {
            assert!(self.rows[first..first + count].iter().all(|&n| n == 0),
                "overlapping transform rows or a transform wrote FC1 padding");
            self.rows[first..first + count].fill(1);
        }
        fn complete(&self) -> bool { self.rows.iter().all(|&n| n != 0) }
        fn check_bytes(&self) {
            let tiles = self.columns.div_ceil(4);
            let code_bytes = self.rows.len() * 2 * tiles;
            let mut writes = vec![0u8; self.bytes];
            // Expand row ownership independently into every lane-major code
            // byte and row base. Duplicate writes and unwritten padding fail.
            for group in (0..self.rows.len()).step_by(256) {
                let tile_rows = (self.rows.len() - group).min(256);
                let mut first = group;
                while first < group + tile_rows {
                    let owner = self.rows[first];
                    assert_ne!(owner, 0, "generated plane contains an unwritten row");
                    let end = (first + 1..group + tile_rows).find(|&row| self.rows[row] != owner)
                        .unwrap_or(group + tile_rows);
                    for tile in 0..tiles {
                        let start = (group * tiles + tile * tile_rows + first % 256) * 2;
                        let range = &mut writes[start..start + (end - first) * 2];
                        assert!(range.iter().all(|&n| n == 0), "overlapping generated code bytes");
                        range.fill(owner);
                    }
                    let bases = &mut writes[code_bytes + first..code_bytes + end];
                    assert!(bases.iter().all(|&n| n == 0), "overlapping generated row bases");
                    bases.fill(owner);
                    first = end;
                }
            }
            assert!(writes.iter().all(|&n| n != 0), "generated plane has byte holes");
        }
    }

    for world in [3, 4, 6] {
        // The exact golden onboards eight whole layers and leaves TP3 rank 1
        // with layers 8..39; the card-only TP4/TP6 screen missed this route.
        let rank = if world == 3 { 1 } else { world - 1 };
        let logical = shape.intermediate / world;
        let stored = cuteafd_core::mxfp4_layouts::stored_width(shape.hidden, logical).context("stored width")?;
        let planes = NativePlanes::with_scale_codes(shape.hidden, logical, stored, shape.experts, false, true)?;
        let first_layer = if world == 3 { 8 } else { shape.first_layer };
        let slices = (first_layer..shape.layers).map(|layer| SparkSlice {
            unit_map: None, layer: layer as u32, first: (rank * logical) as u32,
            width: logical as u32, stored_width: stored as u32, tiers: vec![], projection_counts: vec![],
        }).collect::<Vec<_>>();
        let plan = RankPlan::new(&catalog, &slices, world, rank, slices.len() * planes.bytes)?;
        let checked = plan.checked(&catalog, &snapshot, world, rank)?;
        let tail = plan.scales.base(&plan.layout);
        assert!(plan.scales.host_peak()? <= budget);
        if world == 3 { assert_eq!(plan.scales.host_staging().largest_step, 122880); }
        let mut expected = BTreeMap::new();
        for (region, _) in slices.iter().enumerate() {
            let base = plan.layout.region(region)?.start;
            for expert in 0..shape.experts {
                for plane in [1, 3] {
                    let (offset, bytes) = planes.expert_plane(plane, expert)?;
                    let down = plane == 3;
                    let rows = if down { shape.hidden } else { 2 * stored };
                    let columns = if down { logical / 32 } else { shape.hidden / 32 };
                    assert_eq!(bytes, rows * (1 + 2 * columns.div_ceil(4)));
                    assert!(expected.insert(base + offset, (bytes, down)).is_none());
                }
            }
        }
        assert_eq!(plan.scales.steps.len(), slices.len() * shape.experts * 3);
        let mut files = BTreeMap::<u32, Vec<&Entry>>::new();
        for entry in &checked.plan.entries { files.entry(entry.file).or_default().push(entry); }
        let mut separated = false;
        for entries in files.values_mut() {
            entries.sort_by_key(|entry| entry.src_base);
            let mut previous_end = 0;
            let mut weight_start = None;
            let mut scale_start = None;
            for entry in entries {
                let extent = checked.plan.patterns[entry.pattern as usize].check().map_err(anyhow::Error::msg)?;
                assert!(entry.src_base >= previous_end, "source tensor extents interleave inside a file");
                previous_end = entry.src_base + extent.src_len;
                if entry.dst_base < tail {
                    if scale_start.is_some_and(|at| entry.src_base.saturating_sub(at) > (1 << 20)) { separated = true; }
                    weight_start.get_or_insert(entry.src_base);
                } else {
                    if weight_start.is_some_and(|at| entry.src_base.saturating_sub(at) > (1 << 20)) { separated = true; }
                    scale_start.get_or_insert(entry.src_base);
                }
            }
        }
        assert!(separated, "real layout did not exercise scales distant from weights");

        // Replay the scale-only source projection through the actual packer:
        // 16 MiB read windows, coalesced gaps, and 1 MiB receiver slots. Omitting
        // resident weight payload makes scales denser, not cheaper to stage.
        let extents = checked.plan.patterns.iter().map(|p| p.check().ok()).collect::<Vec<_>>();
        let zeros = vec![0u8; DEFAULT_SLOT_BYTES];
        let mut packed = BTreeMap::new();
        for (&file, entries) in &files {
            let entries = entries.iter().filter(|entry| entry.dst_base >= tail)
                .map(|entry| (**entry).clone()).collect::<Vec<_>>();
            let index = FileEntries::new(&entries, &extents);
            let mut ranges = Vec::<(u64, u64)>::new();
            for (first, end) in index.ranges() {
                if let Some(last) = ranges.last_mut().filter(|last| first <= last.1.saturating_add(1 << 20)) {
                    last.1 = last.1.max(end);
                } else { ranges.push((first, end)); }
            }
            let (mut slots, mut packer) = (Vec::new(), slot::Packer::new(vec![0; DEFAULT_SLOT_BYTES]));
            for (first, end) in ranges {
                for from in (first..end).step_by(16 << 20) {
                    let to = (from + (16 << 20)).min(end);
                    for entry in index.overlapping(from, to) {
                        let pattern = &checked.plan.patterns[entry.pattern as usize];
                        let mut coalesced = Vec::<(u64, usize)>::new();
                        for (_, dst, len) in RunWalk::new(pattern, entry, from, to) {
                            if let Some(last) = coalesced.last_mut().filter(|last| last.0 + last.1 as u64 == dst) {
                                last.1 += len as usize;
                            } else { coalesced.push((dst, len as usize)); }
                        }
                        for (mut dst, mut len) in coalesced {
                            while len > 0 {
                                let took = packer.push(dst, &zeros[..len.min(zeros.len())]);
                                if took == 0 {
                                    let (buffer, _, _) = packer.finish();
                                    let parsed = slot::parse(&buffer, plan.scales.logical_bytes(&plan.layout))?;
                                    slots.push(slot::Fragments::new(&buffer, parsed).collect::<Vec<_>>());
                                    packer = slot::Packer::new(buffer);
                                } else { dst += took as u64; len -= took; }
                            }
                        }
                    }
                }
            }
            if !packer.is_empty() {
                let (buffer, _, _) = packer.finish();
                let parsed = slot::parse(&buffer, plan.scales.logical_bytes(&plan.layout))?;
                slots.push(slot::Fragments::new(&buffer, parsed).collect::<Vec<_>>());
            }
            packed.insert(file, slots);
        }
        for sources in [6, 8] {
            let mut streams = (0..sources).map(|_| VecDeque::new()).collect::<Vec<_>>();
            for (&file, slots) in &packed { streams[file as usize % sources].extend(slots.iter()); }
            let mut pending = Vec::new();
            let (mut opened, mut complete) = (BTreeMap::<usize, usize>::new(), BTreeSet::new());
            let (mut retained, mut peak_bytes, mut peak_steps, mut max_slot_steps) = (0usize, 0usize, 0usize, 0usize);
            while streams.iter().any(|stream| !stream.is_empty()) || !pending.is_empty() {
                for stream in &mut streams {
                    if pending.len() == DEFAULT_SLOTS { break; }
                    if let Some(slot) = stream.pop_front() { pending.push(slot); }
                }
                if pending.len() < DEFAULT_SLOTS && streams.iter().any(|s| !s.is_empty()) { continue; }
                // A reversed full ring retains incomplete old steps while
                // later slots finish others. No byte payload or GPU is needed.
                while let Some(fragments) = pending.pop() {
                    let mut touched = BTreeSet::new();
                    for &(dst, len) in fragments {
                        let relative = dst - tail;
                        let index = plan.scales.steps.partition_point(|(start, _)| *start <= relative) - 1;
                        let (start, step) = &plan.scales.steps[index];
                        let need = step.rects.iter().map(|rect| rect.bytes()).sum::<usize>();
                        assert!(relative + u64::from(len) <= start + need as u64);
                        assert!(!complete.contains(&index));
                        touched.insert(index);
                        if !opened.contains_key(&index) { retained += need; }
                        *opened.entry(index).or_default() += len as usize;
                        peak_bytes = peak_bytes.max(retained);
                        peak_steps = peak_steps.max(opened.len());
                        assert!(retained <= plan.scales.host_staging().staging_bytes);
                        assert!(opened[&index] <= need);
                        if opened[&index] == need {
                            opened.remove(&index);
                            retained -= need;
                            assert!(complete.insert(index));
                        }
                    }
                    max_slot_steps = max_slot_steps.max(touched.len());
                }
            }
            assert!(opened.is_empty());
            assert_eq!(complete.len(), plan.scales.steps.len());
            assert!(max_slot_steps > 2, "small steps per slot were not exercised");
            assert!(peak_steps > 8, "old eight-step guard was not exercised");
            eprintln!("TP{world} rank{rank}, sources={sources}, reversed128-slot windows: peak_steps={peak_steps}, peak_bytes={peak_bytes}, max_slot_steps={max_slot_steps}, admitted={}", plan.scales.host_peak()?);
            let mut queues = (0..sources).map(|_| VecDeque::new()).collect::<Vec<_>>();
            for (&file, entries) in &files {
                for &entry in entries { queues[file as usize % sources].push_back((entry, 0u64)); }
            }
            let mut open = BTreeMap::<usize, usize>::new();
            let mut completed = BTreeSet::new();
            let mut partial = BTreeMap::<usize, Plane>::new();
            let mut finished = BTreeSet::new();
            let (mut peak, mut weights) = (0, 0usize);
            while queues.iter().any(|queue| !queue.is_empty()) {
                for queue in &mut queues {
                    let Some((entry, consumed)) = queue.front_mut() else { continue; };
                    if entry.dst_base < tail {
                        weights += 1;
                        queue.pop_front();
                        continue;
                    }
                    let pattern = &checked.plan.patterns[entry.pattern as usize];
                    let extent = pattern.check().map_err(anyhow::Error::msg)?;
                    // Coalesce dense destination runs, preserving file/row
                    // order. A 16 KiB fragment also splits TP6's 60 KiB scales,
                    // so every width exercises simultaneously open sources.
                    let mut dense = u64::from(pattern.run);
                    for dim in pattern.dims.iter().rev() {
                        let Steps::Count(count) = &dim.steps else { anyhow::bail!("unexpected scale index list"); };
                        assert_eq!(dim.dst_stride, dense);
                        dense *= u64::from(*count);
                    }
                    assert_eq!(dense, extent.bytes);
                    let relative = entry.dst_base - tail + *consumed;
                    let index = plan.scales.steps.partition_point(|(start, _)| *start <= relative) - 1;
                    let (start, step) = &plan.scales.steps[index];
                    assert_eq!(step.rects.len(), 1, "scale step spans independent source tensors");
                    let need = step.rects[0].bytes();
                    assert!(relative + dense - *consumed <= start + need as u64);
                    assert!(!completed.contains(&index), "source replay reopened a completed scale step");
                    let bytes = (dense - *consumed).min(16 << 10) as usize;
                    let delivered = open.entry(index).or_default();
                    assert_eq!(relative - start, *delivered as u64, "scale source has delivery holes");
                    *delivered += bytes;
                    peak = peak.max(open.len());
                    assert!(open.len() <= sources, "serial replay holds more than one step per source");
                    let retained: usize = open.keys().map(|&index| plan.scales.steps[index].1.rects.iter()
                        .map(|rect| rect.bytes()).sum::<usize>()).sum();
                    assert!(retained <= plan.scales.host_staging().staging_bytes, "pinned scale budget exceeded");
                    *consumed += bytes as u64;
                    if open[&index] == need {
                        open.remove(&index);
                        assert!(completed.insert(index));
                        for output in &step.outputs {
                            let rect = &step.rects[output.rect];
                            let &(expected_bytes, down) = expected.get(&output.offset).context("unplanned coded plane")?;
                            assert_eq!(output.bytes, expected_bytes);
                            assert_eq!(output.at, 0);
                            assert!(!finished.contains(&output.offset), "plane written after completion");
                            let plane = partial.entry(output.offset)
                                .or_insert_with(|| Plane::new(shape.hidden, logical, stored, down));
                            assert_eq!((plane.bytes, plane.rows.len(), plane.columns),
                                (output.bytes, output.stored_rows, rect.cols.len()));
                            let first = if rect.name.ends_with(".w1.scale") { stored } else { 0 };
                            assert_eq!(output.start_row, first);
                            assert_eq!(rect.rows.len(), if down { shape.hidden } else { logical });
                            plane.write(output.start_row, rect.rows.len());
                            if plane.complete() {
                                plane.check_bytes();
                                partial.remove(&output.offset);
                                assert!(finished.insert(output.offset));
                            }
                        }
                    }
                    if *consumed == dense { queue.pop_front(); }
                }
            }
            assert!(weights > 0 && peak > 1, "source interleaving was not exercised");
            assert!(open.is_empty() && partial.is_empty(), "Done left incomplete steps or planes");
            assert_eq!(completed.len(), plan.scales.steps.len());
            assert_eq!(finished, expected.keys().copied().collect());
            eprintln!("coded V4.1 TP{world} rank{rank}, {sources} file-ordered sources: {} steps, {} planes, peak_open={peak}, peak_pinned_with_table={} <= {budget}, no byte holes",
                completed.len(), finished.len(), peak * plan.scales.host_staging().largest_step + (1 << 20));
        }
    }
    Ok(())
}

#[test]
#[ignore = "requires an allocated CPU build slot and explicit NVMe fixture directory"]
fn static_scale_tail_dispatch_accepts_out_of_order_validation() -> Result<()> {
    use super::super::{device::Device, weight_arena::ExpectedCoverage};
    use cuteafd_ffi::native_library_lifetime_fixture::Fixture;

    let fixture = Fixture::build()?;
    let library = fixture.load()?;
    let device = Device { library: &library, id: 0 };
    let plan = ScalePlan::default();
    let layout = ArenaLayout::new([256], 256)?.with_validation(4, 4)?;
    let base = plan.base(&layout);
    assert!(base > layout.logical_bytes() as u64);
    let tail = 0xcbac1fedu32.to_le_bytes();
    for tail_first in [false, true] {
        let expected = ExpectedCoverage::spans(&[(0, 4), (256, 4)]);
        let mut arena = plan.destination(device, layout.clone(), expected)?;
        assert!(arena.staged.is_none());
        // The validation tail is ordinary resident-plan data, not a scale
        // staging destination. Either source may land first.
        if tail_first {
            arena.check_fragment(256, 4)?;
            arena.copy_validation(256, &tail)?;
        }
        arena.check_fragment(0, 4)?;
        arena.copied(0, 4)?;
        if !tail_first {
            arena.check_fragment(256, 4)?;
            arena.copy_validation(256, &tail)?;
        }
        // A bogus address remains invalid even after every expected byte lands;
        // waiting for Done cannot create a scale staging plan.
        assert_eq!(arena.check_fragment(base, 4).unwrap_err().to_string(), "scale tail absent");
        assert_eq!(arena.copy_validation(base, &tail).unwrap_err().to_string(), "scale tail absent");
        let delivered = arena.done()?;
        assert_eq!(delivered.validation(), &tail);
    }
    let expected = ExpectedCoverage::spans(&[(0, 4), (256, 4)]);
    let mut incomplete = plan.destination(device, layout, expected)?;
    incomplete.copied(0, 4)?;
    assert!(incomplete.done().is_err());
    Ok(())
}

#[test]
fn native_plane_geometry_keeps_tp4_tp6_mount_admission() -> Result<()> {
    for nvfp4 in [false, true] {
        for world in [4, 6] {
            let logical: usize = 2304 / world;
            let stored = if nvfp4 { logical.div_ceil(128) * 128 } else {
                cuteafd_core::mxfp4_layouts::stored_width(5120, logical).unwrap()
            };
            let planes = NativePlanes::new(5120, logical, stored, 384, nvfp4)?;
            let resident = cuteafd_loader::placement::families::deepseek_v41::resident_bytes(
                5120, stored as u64, 384, nvfp4)?;
            assert_eq!(planes.bytes as u64, resident);
            let layout = ArenaLayout::new([planes.bytes, planes.bytes], planes.bytes * 2)?;
            assert_eq!(layout.region(1)?.start, planes.bytes);
            assert!(ArenaLayout::new([planes.bytes], planes.bytes - 1).is_err());
        }
    }
    Ok(())
}

#[test]
#[ignore = "requires an allocated CPU build slot and explicit NVMe fixture directory"]
fn installed_native_rank_plan_tp4_tp6_rejects_changed_worker_placement() -> Result<()> {
    let root = std::env::var_os("CUTEAFD_PUSH_V41").context("set CUTEAFD_PUSH_V41")?;
    let root = Path::new(&root);
    let catalog = cuteafd_loader::read_expert_catalog(root)?;
    let geometry = catalog.routed_experts();
    let nvfp4 = catalog.nvfp4().is_some();
    for world in [4, 6] {
        let mut plans = Vec::new();
        for rank in 0..world {
            let width = geometry.intermediate as usize / world;
            let stored = if nvfp4 { width.div_ceil(128) * 128 } else {
                cuteafd_core::mxfp4_layouts::stored_width(geometry.hidden, width).unwrap()
            };
            let slice = SparkSlice { unit_map: None, layer: 0, first: (rank * width) as u32,
                width: width as u32, stored_width: stored as u32, tiers: vec![], projection_counts: vec![] };
            let resident = NativePlanes::with_scale_codes(geometry.hidden, width, stored,
                geometry.experts, nvfp4, !nvfp4 && catalog.mxfp4_scale_codes())?.bytes;
            let plan = RankPlan::new(&catalog, &[slice.clone()], world, rank, resident)?;
            let checked = plan.checked(&catalog, root, world, rank)?;
            checked.plan.check(&checked.file_sizes, &checked.arenas)?;
            assert_eq!(plan.placement[0].slice, slice);
            let assignment = cuteafd_loader::placement::SparkAssignment { rank: rank as u32,
                resident_bytes: resident as u64, slices: vec![slice.clone()], io: Default::default() };
            let offer = Offer { rank, host: format!("rank-{rank}"),
                arena_bytes: plan.logical_bytes(), placement: plan.placement.clone() };
            super::super::push_exl3::validate_offer(&offer, rank, &assignment, &plan.placement, offer.arena_bytes)?;
            let mut bad = offer.clone();
            bad.placement[0].slice.first += 32;
            assert!(super::super::push_exl3::validate_offer(&bad, rank, &assignment, &plan.placement, offer.arena_bytes).is_err());
            bad = offer.clone();
            bad.placement[0].fingerprint.push('0');
            assert!(super::super::push_exl3::validate_offer(&bad, rank, &assignment, &plan.placement, offer.arena_bytes).is_err());
            let mut changed = slice.clone();
            changed.first += 32;
            assert!(RankPlan::new(&catalog, &[changed], world, rank, resident).is_err());
            let mut changed = slice;
            changed.width -= 32;
            assert!(RankPlan::new(&catalog, &[changed], world, rank, resident).is_err());
            assert!(RankPlan::new(&catalog, &[plan.placement[0].slice.clone()], world, rank, resident - 1).is_err());
            eprintln!("native TP{world} rank={rank}: checked {} entries, resident={resident}, scratch={}",
                checked.plan.entries.len(), plan.scratch);
            plans.push((resident as u64, plan.placement.clone(), checked));
        }
        super::super::push_exl3::tests::coordinator_suffix(root, &catalog, &plans,
            |assignments, offers, coordinator| checked_plan(root, &catalog, assignments, offers, coordinator))?;
    }
    Ok(())
}

#[test]
fn coordinator_suffix_preserves_native_spark_plans() -> Result<()> {
    use cuteafd_loader::plan::testing::{t, write_snapshot};
    let root = tempfile::tempdir()?;
    let config = serde_json::json!({"model_type":"deepseek_v4"});
    let args = serde_json::json!({
        "vocab_size":64,"dim":256,"moe_inter_dim":768,"n_layers":3,"n_heads":4,
        "n_routed_experts":2,"n_shared_experts":1,"n_activated_experts":1,
        "score_func":"sqrtsoftplus","route_scale":1.5,"swiglu_limit":10.0,
        "q_lora_rank":128,"head_dim":128,"rope_head_dim":64,"o_groups":1,
        "o_lora_rank":128,"window_size":128,"original_seq_len":65536,
        "rope_theta":10000,"rope_factor":16,"beta_fast":32,"beta_slow":1,
        "index_n_heads":4,"index_head_dim":128,"index_topk":512,"hc_mult":4,
        "hc_sinkhorn_iters":20,"compress_rope_theta":160000,"compress_ratios":[0,0,0]
    });
    let mut tensors = Vec::new();
    for layer in 0..3 {
        for expert in 0..2 {
            for (projection, rows, cols) in [("w1", 768, 256), ("w3", 768, 256), ("w2", 256, 768)] {
                let prefix = format!("layers.{layer}.ffn.experts.{expert}.{projection}");
                tensors.push(t(format!("{prefix}.weight"), "I8", &[rows, cols / 2]));
                tensors.push(t(format!("{prefix}.scale"), "F8_E8M0", &[rows, cols / 32]));
            }
        }
    }
    write_snapshot(root.path(), &config, &tensors, None);
    std::fs::create_dir(root.path().join("inference"))?;
    std::fs::write(root.path().join("inference/config.json"), serde_json::to_vec(&args)?)?;
    let catalog = cuteafd_loader::read_expert_catalog(root.path())?;
    for world in [3, 4, 5, 6, 7] {
        let artifacts = cuteafd_core::tp_units::native_spark_artifacts(256, 768, world)?;
        let plans = (0..world).map(|rank| {
            let slices = (0..3).map(|layer| {
                let shard = cuteafd_core::tp_units::LayerShard::new(768, 32, layer, world, rank)?;
                let artifact = artifacts.iter().find(|a| a.logical == shard.width).context("missing native artifact")?;
                Ok(SparkSlice { unit_map: None, layer: layer as u32, first: shard.first as u32,
                    width: artifact.logical as u32, stored_width: artifact.stored as u32, tiers: vec![], projection_counts: vec![] })
            }).collect::<Result<Vec<_>>>()?;
            let bytes = slices.iter().map(|slice|
                NativePlanes::new(256, slice.width as usize, slice.stored_width as usize, 2, false).map(|p| p.bytes))
                .collect::<Result<Vec<_>>>()?.into_iter().sum();
            let plan = RankPlan::new(&catalog, &slices, world, rank, bytes)?;
            Ok((plan.layout.bytes() as u64, plan.placement.clone(),
                plan.checked(&catalog, root.path(), world, rank)?))
        }).collect::<Result<Vec<_>>>()?;
        super::super::push_exl3::tests::coordinator_suffix(root.path(), &catalog, &plans,
            |assignments, offers, coordinator| checked_plan(root.path(), &catalog, assignments, offers, coordinator))?;
    }
    Ok(())
}

/// The coordinator's native TP2 halves (a coordinator rank of any push group)
/// land the mount's own BackboneTp2 staging bytes in each expert's plane
/// slots before the post-Done pack: W3 then W1 rows of the half (each padded
/// to the kernel width) and the W2 column half. Raw packages also land scales
/// directly; coded scales use a staged source tail and have GPU exactness tests.
/// Synthetic MXFP4 headers, plus the checkpoint when CUTEAFD_PUSH_V41 is set.
#[test]
fn coordinator_native_tp2_halves_land_the_mount_staging_bytes() -> Result<()> {
    use cuteafd_loader::plan::testing::{t, write_snapshot};
    let synthetic = tempfile::tempdir()?;
    let config = serde_json::json!({"model_type":"deepseek_v4"});
    let args = serde_json::json!({
        "vocab_size":64,"dim":256,"moe_inter_dim":512,"n_layers":2,"n_heads":4,
        "n_routed_experts":2,"n_shared_experts":1,"n_activated_experts":1,
        "score_func":"sqrtsoftplus","route_scale":1.5,"swiglu_limit":10.0,
        "q_lora_rank":128,"head_dim":128,"rope_head_dim":64,"o_groups":1,
        "o_lora_rank":128,"window_size":128,"original_seq_len":65536,
        "rope_theta":10000,"rope_factor":16,"beta_fast":32,"beta_slow":1,
        "index_n_heads":4,"index_head_dim":128,"index_topk":512,"hc_mult":4,
        "hc_sinkhorn_iters":20,"compress_rope_theta":160000,"compress_ratios":[0, 0]
    });
    let mut tensors = Vec::new();
    for layer in 0..2 {
        for expert in 0..2 {
            for (projection, rows, cols) in [("w1", 512, 256), ("w3", 512, 256), ("w2", 256, 512)] {
                let prefix = format!("layers.{layer}.ffn.experts.{expert}.{projection}");
                tensors.push(t(format!("{prefix}.weight"), "I8", &[rows, cols / 2]));
                tensors.push(t(format!("{prefix}.scale"), "F8_E8M0", &[rows, cols / 32]));
            }
        }
    }
    write_snapshot(synthetic.path(), &config, &tensors, None);
    std::fs::create_dir(synthetic.path().join("inference"))?;
    std::fs::write(synthetic.path().join("inference/config.json"), serde_json::to_vec(&args)?)?;
    // Distinct bytes everywhere, so a misplaced run cannot match.
    for entry in std::fs::read_dir(synthetic.path())? {
        let path = entry?.path();
        if path.extension().is_some_and(|e| e == "safetensors") {
            let mut bytes = std::fs::read(&path)?;
            let header = 8 + u64::from_le_bytes(bytes[..8].try_into()?) as usize;
            for (i, b) in bytes[header..].iter_mut().enumerate() { *b = (i.wrapping_mul(131) % 251) as u8; }
            std::fs::write(&path, bytes)?;
        }
    }
    let mut roots = vec![(synthetic.path().to_path_buf(), vec![0usize, 1])];
    if let Some(root) = std::env::var_os("CUTEAFD_PUSH_V41") { roots.push((root.into(), vec![0, 7])); }
    for (root, layers) in roots {
        let catalog = cuteafd_loader::read_expert_catalog(&root)?;
        for (expert_world, gpu) in [(1, 0), (2, 0), (2, 1)] {
            use super::super::push_coordinator::CoordinatorExpertPlan;
            let make = |budget| if expert_world == 1 {
                CoordinatorExpertPlan::whole_native(&catalog, &layers, budget)
            } else { CoordinatorExpertPlan::new(&catalog, &layers, gpu, budget) };
            let plan = make(usize::MAX)?;
            let admitted = plan.layout.bytes();
            assert!(admitted > 0);
            assert_eq!(make(admitted)?.layout.bytes(), admitted);
            assert!(make(admitted - 1).is_err());
            assert!(!plan.native.is_empty() && plan.validation.bytes() == 0);
            let world = 6;
            let checked = plan.checked(&catalog, &root, world, 4 + gpu)?;
            checked.plan.check(&checked.file_sizes, &checked.arenas)?;
            // Land every run of the plan into a host image of the rank.
            let mut image = vec![0u8; plan.layout.logical_bytes()];
            let mut files = std::collections::HashMap::new();
            for entry in checked.plan.entries.iter().filter(|e| e.rank as usize == 4 + gpu) {
                // Generated planes are tested through the production GPU path;
                // this host image only verifies directly delivered operands.
                if entry.dst_base >= plan.scales.base(&plan.layout) { continue; }
                let file = files.entry(entry.file).or_insert_with(||
                    std::fs::File::open(root.join(&checked.plan.files[entry.file as usize])).unwrap());
                let pattern = &checked.plan.patterns[entry.pattern as usize];
                for (src, dst) in pattern.runs() {
                    let at = (entry.dst_base + dst) as usize;
                    file.read_exact_at(&mut image[at..at + pattern.run as usize], entry.src_base + src)?;
                }
            }
            for (region, (&layer, planes)) in layers.iter().zip(&plan.native).enumerate() {
                let base = plan.layout.region(region)?.start;
                for expert in [0, planes.experts - 1] {
                    let selection = if expert_world == 1 {
                        cuteafd_loader::V41ExpertSelection::BackboneFull { layer, expert }
                    } else { cuteafd_loader::V41ExpertSelection::BackboneTp2 { layer, expert, rank: gpu } };
                    let staging = catalog.expert_staging(selection)?;
                    let mut staged = vec![0u8; staging.staging_bytes()];
                    let mut scratch = vec![0u8; staging.minimum_read_scratch_bytes().max(1 << 20)];
                    staging.read_into(&mut staged, &mut scratch)?;
                    let tensor = |slot: usize| &staged[staging.tensor_ranges()[slot].clone()];
                    for (plane, (first, second)) in [(0, (1, 0)), (1, (4, 3))] {
                        if planes.scale_codes && plane == 1 { continue; }
                        let (offset, bytes) = planes.expert_plane(plane, expert)?;
                        let slot = &image[base + offset..base + offset + bytes];
                        let half = bytes / 2;
                        let (a, b) = (tensor(first), tensor(second));
                        assert_eq!(&slot[..a.len()], a, "layer {layer} expert {expert} plane {plane} up rows");
                        assert_eq!(&slot[half..half + b.len()], b, "layer {layer} expert {expert} plane {plane} gate rows");
                    }
                    for (plane, source) in [(2, 2), (3, 5)] {
                        if planes.scale_codes && plane == 3 { continue; }
                        let (offset, bytes) = planes.expert_plane(plane, expert)?;
                        let expected = tensor(source);
                        assert_eq!(&image[base + offset..base + offset + expected.len()], expected,
                            "layer {layer} expert {expert} plane {plane}");
                        assert!(expected.len() <= bytes);
                    }
                }
            }
            eprintln!("{}: gpu{gpu} native TP2 halves land mount staging ({} entries)", root.display(),
                checked.plan.entries.len());
        }
    }
    Ok(())
}


#[test]
fn coordinator_coded_scales_and_family_dense_staging_are_independent_host_owners() -> Result<()> {
    use crate::shared::{memory::push_coordinator::CoordinatorExpertPlan,
        experts::rtx::dense::{HostStaging, STEP_BYTES}};
    let root = tempfile::tempdir()?;
    cuteafd_loader::plan::testing::write_v41_snapshot(root.path());
    let catalog = cuteafd_loader::read_official_v41_catalog(root.path())?;
    for world in [1, 2] {
        let plans = (0..world).map(|rank| if world == 1 {
            CoordinatorExpertPlan::whole_native(&catalog, &[0], usize::MAX)
        } else { CoordinatorExpertPlan::new(&catalog, &[0], rank, usize::MAX) }).collect::<Result<Vec<_>>>()?;
        for plan in &plans {
            let geometry = plan.scales.host_staging();
            assert!(geometry.largest_step > 0 && geometry.largest_step <= STEP_BYTES);
            assert_eq!(geometry.table_bytes, 1 << 20);
            assert_eq!(plan.host_bytes()?, geometry.bytes()?);
        }
        let dense = vec![HostStaging { staging_bytes: 8 * STEP_BYTES, largest_step: STEP_BYTES,
            table_bytes: 2 << 20 }; world];
        let refs = plans.iter().map(Some).collect::<Vec<_>>();
        let peak = world * 128 * (1 << 20) + plans.iter().map(|plan| plan.host_bytes()).sum::<Result<usize>>()?
            + dense.iter().map(|staging| staging.bytes()).sum::<Result<usize>>()?;
        assert_eq!(CoordinatorExpertPlan::admit_host(&refs, &dense, 128, 1 << 20, peak as u64)?, peak);
        assert!(CoordinatorExpertPlan::admit_host(&refs, &dense, 128, 1 << 20, peak as u64 - 1).is_err());
    }
    Ok(())
}
