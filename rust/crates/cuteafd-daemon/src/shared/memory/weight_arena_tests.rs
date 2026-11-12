use super::*;
use cuteafd_loader::push_load::{
    builder::{Builder, Destination},
    plan::{Entry, Pattern, RunWalk},
};
use std::collections::{BTreeMap, VecDeque};

#[test]
fn static_weight_views_retain_backing_and_match_tensor_checksums() -> Result<()> {
    use cuteafd_ffi::native_library_lifetime_fixture::Fixture;
    use sha2::{Digest, Sha256};
    let fixture = Fixture::build_for_tests()?;
    let library = fixture.load()?;
    fixture.immediate_copies(&library)?;
    fixture.configure_memory(&library, 1, 1 << 20)?;
    fixture.strict_device_free(&library, true)?;
    let device = Device { library: &library, id: 1 };
    let tensors = [("payload".to_string(), vec![3u8; 513]), ("scales".into(), vec![17u8; 258])];
    let layout = cuteafd_loader::serving_capacity::WeightArenaLayout::new("test/core/1",
        tensors.iter().map(|(name, data)| (name.clone(), data.len())))?;
    let arena = DeviceWeightArena::new(device, layout, crate::shared::memory::TEST_ARENA)?;
    assert_eq!(arena.layout().padding_bytes(), 255);
    assert!(weight_allocation(Device { library: &library, id: 0 }, Some(&arena), "payload", 513, crate::shared::memory::TEST_ARENA).is_err());
    assert!(arena.view("absent", 513).is_err());
    assert!(arena.view("payload", 512).is_err());
    let views = tensors.iter().map(|(name, data)| arena.view(name, data.len())).collect::<Result<Vec<_>>>()?;
    assert_eq!(views[1].buffer.ptr as usize - views[0].buffer.ptr as usize, 768);
    for ((_, data), view) in tensors.iter().zip(&views) {
        let tensor = Allocation::new(device, data.len(), crate::shared::memory::TEST)?;
        device.run(|| {
            library.copy_h2d(tensor.buffer, data)?;
            library.copy_h2d(view.buffer, data)?;
            let mut old = vec![0; data.len()];
            let mut packed = vec![0; data.len()];
            library.copy_d2h(&mut old, tensor.buffer)?;
            library.copy_d2h(&mut packed, view.buffer)?;
            assert_eq!(Sha256::digest(&old), Sha256::digest(&packed));
            assert_eq!(&packed, data);
            Ok(())
        })?;
    }
    drop(arena);
    assert_eq!(device.run(|| library.cuda_physical_memory_info())?.0, (1 << 20) - 1026,
        "views retain the full physical request");
    assert_eq!(views[0].weight_backing().unwrap().id, "test/core/1");
    drop(views);
    assert_eq!(device.run(|| library.cuda_physical_memory_info())?.0, 1 << 20);
    let events = fixture.events()?;
    assert_eq!(events.matches('D').count(), 3);
    assert_eq!(events.matches('d').count(), 3);
    assert_eq!(fixture.device_free_mismatch(&library)?.0, 0);
    Ok(())
}

#[test]
fn static_weight_receipts_share_one_backing_across_inventories() -> Result<()> {
    use cuteafd_loader::placement::planner::{CoordinatorDevices, Lifetime, PhaseOrder};
    use cuteafd_core::memory_layout::Category;
    use super::super::receipts::{Census, Inventory};
    let fixture = cuteafd_ffi::native_library_lifetime_fixture::Fixture::build_for_tests()?;
    let library = fixture.load()?;
    fixture.strict_device_free(&library, true)?;
    let device = Device { library: &library, id: 1 };
    let layout = cuteafd_loader::serving_capacity::WeightArenaLayout::new("test/core/1",
        [("first".into(), 257), ("second".into(), 2)])?;
    let arena = DeviceWeightArena::new(device, layout, crate::shared::memory::TEST_ARENA)?;
    let first = arena.view("first", 257)?;
    let second = super::super::DeviceAllocation::from_allocation(arena.view("second", 2)?);
    let devices = [1];
    let devices = CoordinatorDevices::new(&devices)?;
    let mut one = Inventory::new(devices, "one", Lifetime::TARGET, Category::Weights, "fixture");
    let mut two = Inventory::new(devices, "two", Lifetime::TARGET, Category::Weights, "fixture");
    first.inventory(&mut one, "first")?;
    second.inventory(&mut two, "second")?;
    drop(arena);
    let mut census = Census::default();
    census.owners(one.receipts);
    census.owners(two.receipts);
    census.normalize(PhaseOrder::DRAFT_THEN_TARGET)?;
    assert_eq!(census.receipts.len(), 3);
    let backings = census.receipts.iter().filter(|r| r.allocation.view.is_none()).collect::<Vec<_>>();
    assert_eq!(backings.len(), 1);
    assert_eq!((backings[0].allocation.id.as_str(), backings[0].allocation.bytes), ("test/core/1", 514));
    for receipt in census.receipts.iter().filter(|r| r.allocation.view.is_some()) {
        assert_eq!(receipt.allocation.view.as_ref().unwrap().backing, "test/core/1");
    }
    assert_eq!(census.receipts[2].allocation.view.as_ref().unwrap().offset, 512);
    drop(first);
    assert_eq!(fixture.events()?.matches('d').count(), 0);
    drop(second);
    assert_eq!(fixture.events()?.matches('d').count(), 1);
    assert_eq!(fixture.device_free_mismatch(&library)?.0, 0);
    Ok(())
}

#[test]
fn static_weight_partial_load_and_failed_drain_keep_owner_contract() -> Result<()> {
    for failed_drain in [false, true] {
        let fixture = cuteafd_ffi::native_library_lifetime_fixture::Fixture::build_for_tests()?;
        let library = fixture.load()?;
        fixture.strict_device_free(&library, true)?;
        let device = Device { library: &library, id: 1 };
        let load = || -> Result<()> {
            let layout = cuteafd_loader::serving_capacity::WeightArenaLayout::new("test/core/1",
                [("first".into(), 257), ("second".into(), 2)])?;
            let arena = DeviceWeightArena::new(device, layout, crate::shared::memory::TEST_ARENA)?;
            let first = arena.view("first", 257)?;
            if failed_drain { library.quarantine_module_after_failed_drain(); }
            // A later tensor's rejected extent unwinds the partially loaded views.
            let _second = arena.view("second", 3)?;
            drop(first);
            Ok(())
        };
        assert!(load().is_err());
        assert_eq!(fixture.events()?.matches('D').count(), 1);
        assert_eq!(fixture.events()?.matches('d').count(), usize::from(!failed_drain));
        assert_eq!(fixture.device_free_mismatch(&library)?.0, 0);
        assert_eq!(library.cuda_get_device()?, 0);
    }
    Ok(())
}

struct FileWalk<'a> {
    walks: Vec<RunWalk<'a>>,
    heads: std::collections::BinaryHeap<std::cmp::Reverse<(u64, u64, u32, usize)>>,
}
impl<'a> FileWalk<'a> {
    fn new(entries: Vec<&'a Entry>, patterns: &'a [Pattern]) -> Self {
        let mut walks = entries
            .iter()
            .map(|entry| RunWalk::new(&patterns[entry.pattern as usize], entry, 0, u64::MAX))
            .collect::<Vec<_>>();
        let mut heads = std::collections::BinaryHeap::new();
        for (index, walk) in walks.iter_mut().enumerate() {
            if let Some((source, destination, bytes)) = walk.next() {
                heads.push(std::cmp::Reverse((source, destination, bytes, index)));
            }
        }
        Self { walks, heads }
    }
    // Merge all entry cursors by file offset, including replicated rotations.
    // Adjacent destination runs combine before touching the coverage tree.
    fn batch(&mut self, expected: &ExpectedCoverage, coverage: &mut Coverage) -> Result<bool> {
        let mut remaining = 4 * 1024 * 1024u64;
        let mut pending: Option<(u64, u64)> = None;
        while remaining > 0 {
            if let Some(std::cmp::Reverse((_, destination, bytes, index))) = self.heads.pop() {
                if let Some((source, next, bytes)) = self.walks[index].next() {
                    self.heads
                        .push(std::cmp::Reverse((source, next, bytes, index)));
                }
                let bytes = u64::from(bytes);
                match &mut pending {
                    Some((start, len)) if *start + *len == destination => *len += bytes,
                    _ => {
                        if let Some((start, len)) = pending.take() {
                            expected.record(coverage, start, len)?;
                        }
                        pending = Some((destination, bytes));
                    }
                }
                remaining = remaining.saturating_sub(bytes);
            } else {
                break;
            }
        }
        if let Some((start, len)) = pending {
            expected.record(coverage, start, len)?;
        }
        Ok(!self.heads.is_empty())
    }
}

#[test]
fn real_glm_full_plan_interleaved_sources_and_file_replay() -> Result<()> {
    full_exl3_plan("CUTEAFD_PUSH_EXL3")
}

fn full_exl3_plan(variable: &str) -> Result<()> {
    let Some(snapshot) = std::env::var_os(variable) else {
        return Ok(());
    };
    let snapshot = std::path::PathBuf::from(snapshot);
    let catalog = cuteafd_loader::read_expert_catalog(&snapshot)?;
    let shape = catalog.routed_experts();
    let started = std::time::Instant::now();
    let mut peak = 0;
    for rank in 0..6 {
        let residencies = (shape.first_layer..shape.layers)
            .map(|layer| {
                catalog.exl3().unwrap().balanced_residency(
                    cuteafd_loader::V41Exl3Layer::Backbone(layer),
                    6,
                    rank,
                )
            })
            .collect::<Result<Vec<_>>>()?;
        let layouts = residencies
            .iter()
            .map(|r| r.device_arena_layout())
            .collect::<std::result::Result<Vec<_>, _>>()?;
        let arena = ArenaLayout::new(layouts.iter().map(|(_, n)| *n), usize::MAX)?;
        let mut sizes = vec![0; 6];
        sizes[rank] = arena.bytes() as u64;
        let mut builder = Builder::new(&snapshot, sizes);
        for (index, (residency, (offsets, bytes))) in residencies.iter().zip(&layouts).enumerate() {
            builder.exl3_residency(
                &catalog,
                residency,
                offsets,
                Destination {
                    rank: rank as u32,
                    offset: arena.region(index)?.start as u64,
                    bytes: *bytes as u64,
                },
            )?;
        }
        let plan = builder.finish()?;
        peak = peak.max(simulate(&plan, rank, variable)?);
    }
    eprintln!("{variable} full coverage simulation: peak {peak} intervals, conservative tree bound {} bytes, elapsed {:?}", peak*128, started.elapsed());
    Ok(())
}

/// Deliver a rank's plan from six then eight interleaved sources in file
/// order plus one whole-file replay; returns the peak coverage intervals.
pub(crate) fn simulate(
    plan: &cuteafd_loader::push_load::builder::CheckedPlan,
    rank: usize,
    label: &str,
) -> Result<usize> {
    let expected = ExpectedCoverage::from_plan(plan, rank as u32)?;
    let mut peak = 0;
    let mut files: BTreeMap<u32, Vec<&Entry>> = BTreeMap::new();
    for entry in &plan.plan.entries {
        files.entry(entry.file).or_default().push(entry);
    }
    for sources in [6, 8] {
        let mut queues = (0..sources).map(|_| VecDeque::new()).collect::<Vec<_>>();
        for (&file, entries) in &files {
            queues[file as usize % sources]
                .push_back(FileWalk::new(entries.clone(), &plan.plan.patterns));
        }
        let mut coverage = Coverage::default();
        while queues.iter().any(|q| !q.is_empty()) {
            for queue in &mut queues {
                if let Some(file) = queue.front_mut() {
                    if !file.batch(&expected, &mut coverage)? {
                        queue.pop_front();
                    }
                }
            }
        }
        assert!(expected.complete(&coverage));
        let before = coverage.bytes;
        let mut replay = FileWalk::new(
            files.first_key_value().unwrap().1.clone(),
            &plan.plan.patterns,
        );
        while replay.batch(&expected, &mut coverage)? {}
        assert_eq!(coverage.bytes, before);
        peak = peak.max(coverage.peak);
        eprintln!("{label} TP6 rank{rank}, {sources} sources + file replay: peak {} intervals, conservative tree bound {} bytes", coverage.peak, coverage.peak*128);
    }
    Ok(peak)
}

#[test]
fn real_v41_full_plan_interleaved_sources_and_file_replay() -> Result<()> {
    let Some(snapshot) = std::env::var_os("CUTEAFD_PUSH_V41") else {
        return Ok(());
    };
    let snapshot = std::path::PathBuf::from(snapshot);
    let catalog = cuteafd_loader::read_official_v41_catalog(&snapshot)?;
    let shape = catalog.routed_experts();
    for rank in 0..6 {
        let mut sizes = vec![0; 6];
        sizes[rank] = u64::MAX;
        let mut builder = Builder::new(&snapshot, sizes);
        let mut cursor = 0u64;
        for layer in shape.first_layer..shape.layers {
            for expert in 0..shape.experts {
                let staging =
                    catalog.expert_staging(cuteafd_loader::V41ExpertSelection::BackboneTp {
                        layer,
                        expert,
                        world: 6,
                        rank,
                    })?;
                builder.v41_regions(
                    &catalog,
                    staging.tensor_names(),
                    staging.tensor_ranges(),
                    6,
                    rank as u32,
                    Destination {
                        rank: rank as u32,
                        offset: cursor,
                        bytes: staging.staging_bytes() as u64,
                    },
                )?;
                cursor = cursor
                    .checked_add(staging.staging_bytes() as u64)
                    .context("V4.1 arena overflow")?;
            }
        }
        let mut plan = builder.finish()?;
        plan.arenas[rank] = cursor;
        plan.plan.check(&plan.file_sizes, &plan.arenas)?;
        simulate(&plan, rank, "V4.1 full-layer native")?;
    }
    Ok(())
}

#[test]
fn v41_workspace_components_allocate_once_and_retain_independent_generations() -> Result<()> {
    use cuteafd_loader::serving_capacity::v41_workspace;
    use cuteafd_core::memory_layout::ArenaTag;
    let fixture = cuteafd_ffi::native_library_lifetime_fixture::Fixture::build_for_tests()?;
    let library = fixture.load()?;
    fixture.immediate_copies(&library)?;
    fixture.strict_device_free(&library, true)?;
    for rank in 0..2 {
        library.cuda_set_device(rank)?;
        let mut layouts = vec![v41_workspace::hc(1)?, v41_workspace::window(1, 1, &[0, 1], 256)?,
            v41_workspace::engram(1, 256)?];
        layouts.extend(v41_workspace::projection_backings(1, 5120, 512, 256, true)?);
        let split = v41_workspace::projection_backings(40, 5120, 512, 714752, true)?;
        assert_eq!(split.len(), 2);
        assert_ne!(split[0].id(), split[1].id());
        layouts.extend(split);
        for layout in layouts {
            let before = fixture.events()?;
            let arena = workspace_arena(&library, layout.clone(), ArenaTag::workspace("test/v41-workspace"))?;
            assert_eq!(arena.layout().bytes(), layout.bytes());
            let views = layout.views().iter().map(|(name, view)| arena.view(name, view.bytes()))
                .collect::<Result<Vec<_>>>()?;
            for (index, view) in views.iter().enumerate() {
                assert_eq!(view.buffer.ptr as usize % 256, 0);
                let bytes = vec![index as u8; view.buffer.bytes.min(32)];
                library.copy_h2d(view.buffer, &bytes)?;
                let mut actual = vec![0; bytes.len()];
                library.copy_d2h(&mut actual, view.buffer)?;
                assert_eq!(actual, bytes);
            }
            assert_eq!(fixture.events()?.matches('D').count(), before.matches('D').count() + 1);
            drop(arena);
            assert_eq!(fixture.events()?.matches('d').count(), before.matches('d').count());
            drop(views);
            assert_eq!(fixture.events()?.matches('d').count(), before.matches('d').count() + 1);
        }
    }
    assert_eq!(fixture.device_free_mismatch(&library)?.0, 0);
    Ok(())
}

/// One physical backing per arena, recorded once under the arena's tag; every
/// view reports that tag, and the arena-less standalone tag never applies to a view.
#[test]
fn arena_views_inherit_one_backing_tag() -> Result<()> {
    use cuteafd_core::memory_layout::{arena_class, ArenaTag, MemoryKind};
    use cuteafd_ffi::{memory_ledger, native_library_lifetime_fixture::Fixture};
    let fixture = Fixture::build_for_tests()?;
    let library = fixture.load()?;
    fixture.configure_memory(&library, 1, 1 << 20)?;
    let device = Device { library: &library, id: 0 };
    let layout = cuteafd_loader::serving_capacity::WeightArenaLayout::new("test/kv-arena/0",
        [("pages".to_string(), 4096), ("index".to_string(), 512)])?;
    let audit = memory_ledger::AllocationAudit::begin();
    let start = audit.checkpoint();
    let arena: DeviceArena<'_, arena_class::FixedPool> = DeviceArena::new(device, layout, ArenaTag::kv("test/kv-arena"))?;
    let views = [arena.view("pages", 4096)?, arena.view("index", 512)?];
    let history = audit.finish();
    // The audit sees every thread; this test's owner is unique to it.
    let backings = history.events[start..].iter().filter_map(|event| match event {
        memory_ledger::AllocationEvent::Allocated(record)
            if record.key.tag.owner == "test/kv-arena" => Some(*record),
        _ => None }).collect::<Vec<_>>();
    assert_eq!(backings.len(), 1, "an arena records exactly one backing: {backings:?}");
    assert_eq!(backings[0].key.tag, memory_ledger::Tag::kv("test/kv-arena"));
    assert_eq!(backings[0].bytes, arena.layout().bytes());
    for view in &views {
        assert_eq!(view.tag(), Some(arena.tag()));
        assert_eq!(view.tag().unwrap().kind, MemoryKind::Kv);
    }
    // A weight arena view ignores the standalone tag a caller passes.
    let weights = DeviceWeightArena::new(device, cuteafd_loader::serving_capacity::WeightArenaLayout::new(
        "test/weight-arena/0", [("w".to_string(), 256)])?, ArenaTag::weights("test/weight-arena"))?;
    let view = weight_allocation(device, Some(&weights), "w", 256, ArenaTag::drafter_weights("test/elsewhere"))?;
    assert_eq!(view.tag(), Some(memory_ledger::Tag::weights("test/weight-arena")));
    let standalone = weight_allocation(device, None, "w", 256, ArenaTag::drafter_weights("test/elsewhere"))?;
    assert_eq!(standalone.tag(), Some(memory_ledger::Tag::drafter_weights("test/elsewhere")));
    Ok(())
}
