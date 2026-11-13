// Included as a unit module so the fixture can exercise the real private
// staging owner without adding production test hooks. These directed tests
// build a tiny CPU-only DSO and require an explicitly allocated build slot:
// CUTEAFD_NATIVE_LIFETIME_FIXTURE_DIR=$TASK/fixtures cargo test -p cuteafd-ffi \
//   --lib native_library_lifetime -- --ignored --nocapture
use super::*;
use crate::native_library_lifetime_fixture::Fixture;

#[test]
#[ignore = "requires an allocated CPU build slot and explicit NVMe fixture directory"]
fn rdma_ring_ledger_uses_native_backing_and_retires_only_after_destroy() -> Result<()> {
    let _ledger = memory_ledger::test_guard();
    use memory_ledger::{AllocationAudit, AllocationEvent, Space};
    let fixture = Fixture::build()?;
    let library = fixture.load()?;
    let owner = memory_ledger::owner("fixture.rdma.rings");
    let audit = AllocationAudit::begin();
    for (flags, space) in [(CUTEAFD_HOST_BUFFER_FLAG_PINNED | CUTEAFD_HOST_BUFFER_FLAG_MAPPED,
        Space::Pinned), (CUTEAFD_HOST_BUFFER_FLAG_NONE, Space::Host)] {
        let info = library.rdma_rc_endpoint_create_on_gid_with_buffer_flags(
            "fixture", 1, 0, 17, 64, 64, 512, 768, 2, 2, 1, flags)?;
        let views = [library.rdma_rc_endpoint_buffer_view(info.handle, false)?,
            library.rdma_rc_endpoint_buffer_view(info.handle, true)?];
        let records = memory_ledger::allocations().into_iter()
            .filter(|record| record.endpoint == Some(info.handle as usize)).collect::<Vec<_>>();
        assert_eq!(records.len(), 2);
        for view in views {
            let record = records.iter().find(|r| r.address == view.host_ptr as usize).unwrap();
            assert_eq!(record.bytes, view.bytes);
            assert_eq!(record.key.space, space);
            assert_eq!(record.key.owner, Some("fixture.rdma.rings"));
            assert_ne!(record.address, info.handle as usize);
        }
        fixture.configure_endpoint_failures(&library, 0, 1)?;
        assert!(library.rdma_rc_endpoint_destroy(info.handle).is_err());
        assert!(records.iter().all(|record| memory_ledger::allocations().contains(record)));
        fixture.configure_endpoint_failures(&library, 0, 0)?;
        library.rdma_rc_endpoint_destroy(info.handle)?;
        assert!(memory_ledger::allocations().iter().all(|r| r.endpoint != Some(info.handle as usize)));
    }
    fixture.configure_endpoint_failures(&library, 1, 0)?;
    let before = fixture.events()?;
    assert!(library.rdma_rc_endpoint_create_on_gid_with_buffer_flags(
        "fixture", 1, 0, 17, 64, 64, 512, 768, 2, 2, 1, 0).is_err());
    assert_eq!(fixture.events()?, format!("{before}X"));
    drop(owner);
    let events = audit.finish().events.into_iter().filter(|event| match event {
        AllocationEvent::Allocated(r) | AllocationEvent::Freed(r) | AllocationEvent::Enriched(r) | AllocationEvent::UnobservedRelease(r) =>
            r.key.owner == Some("fixture.rdma.rings"),
    }).collect::<Vec<_>>();
    assert_eq!(events.len(), 8);
    assert_eq!(events.iter().filter(|e| matches!(e, AllocationEvent::Allocated(_))).count(), 4);
    assert_eq!(events.iter().filter(|e| matches!(e, AllocationEvent::Freed(_))).count(), 4);
    Ok(())
}

#[test]
#[ignore = "requires an allocated CPU build slot and explicit NVMe fixture directory"]
fn symbols_resolve_once_and_optional_exports_stay_lazy() -> Result<()> {
    let fixture = Fixture::build()?;
    let library = fixture.load()?;
    assert!(library.symbols.cuteafd_cuda_stream_query.get().is_none());
    assert!(library.symbols.cuteafd_xgrammar_compile.get().is_none());
    let first = library.cuteafd_cuda_stream_query()?;
    let second = library.cuteafd_cuda_stream_query()?;
    assert_eq!(first as usize, second as usize);
    std::thread::scope(|scope| {
        for _ in 0..8 {
            let library = &library;
            scope.spawn(move || assert_eq!(
                library.cuteafd_cuda_stream_query().unwrap() as usize, first as usize));
        }
    });
    // SAFETY: the CPU fixture accepts a null synthetic stream and writes ready.
    assert!(!unsafe { library.cuda_stream_query(std::ptr::null_mut())? });
    assert!(library.cuteafd_xgrammar_compile().is_err());
    let first_error = library.symbols.cuteafd_xgrammar_compile.get().unwrap().as_ref().unwrap_err();
    let error = library.cuteafd_xgrammar_compile().unwrap_err();
    let cached_error = error.downcast_ref::<Arc<libloading::Error>>().unwrap();
    assert!(Arc::ptr_eq(first_error, cached_error), "missing exports are cached too");
    assert!(error.to_string().contains("cuteafd_xgrammar_compile"));
    drop(library);
    assert!(!fixture.resident()?, "the cache must not retain a healthy module");
    Ok(())
}

#[test]
#[ignore = "requires an allocated CPU build slot and explicit NVMe fixture directory"]
fn cached_k6_wire_owners_match_diagnostic_rounding() -> Result<()> {
    let fixture = Fixture::build()?;
    let library = fixture.load()?;
    let narrow = library.f32_bf16_narrow()?;
    let add = library.bf16_f32_add()?;
    let source = [f32::from_bits(0x3f808000), f32::from_bits(0x3f818000), f32::INFINITY, f32::NEG_INFINITY];
    let mut rank0 = [0u16; 4];
    let rank1 = [0xbf80u16, 0xbf80, 0x7f80, 0xff80];
    let mut output = [0f32; 4];
    let view = |ptr, bytes| CuteafdDeviceBuffer { ptr, bytes, device_id: 0, flags: 0 };
    // SAFETY: the fixture uses live, disjoint host arrays and a synthetic stream.
    unsafe {
        narrow.launch(source.as_ptr().cast(), rank0.as_mut_ptr().cast(), 4, std::ptr::null_mut())?;
        add.launch(view(rank0.as_mut_ptr().cast(), 8), view(rank1.as_ptr().cast_mut().cast(), 8),
            view(output.as_mut_ptr().cast(), 16), 4, std::ptr::null_mut())?;
    }
    assert_eq!(rank0, [0x3f80, 0x3f82, 0x7f80, 0xff80]);
    assert_eq!(output.map(f32::to_bits), [0, 0x3c800000, 0x7f800000, 0xff800000]);
    assert!(library.symbols.cuteafd_load_narrow_f32_bf16.get().unwrap().is_ok());
    assert!(library.symbols.cuteafd_bf16_f32_add_async.get().unwrap().is_ok());
    Ok(())
}

#[test]
#[ignore = "requires an allocated CPU build slot and explicit NVMe fixture directory"]
fn staging_census_getters_preserve_absence_backing_and_native_history() -> Result<()> {
    let _ledger = memory_ledger::test_guard();
    let fixture = Fixture::build()?;
    let library = fixture.load()?;
    let mut dst = library.alloc_device_buffer(32, crate::memory_ledger::TEST)?;
    let stream = 1usize as *mut c_void;
    let audit = memory_ledger::AllocationAudit::begin();
    let before = fixture.events()?;
    assert!(library.sync_h2d_staging_buffer()?.is_none());
    assert!(library.ordered_h2d_staging_buffer(dst, stream)?.is_none());
    assert_eq!(audit.checkpoint(), 0);
    assert_eq!(fixture.events()?, before);
    fixture.stage(&library)?;
    fixture.reserve_ordered_upload(&library, dst, stream as usize)?;
    let position = audit.checkpoint();
    let before = fixture.events()?;
    let sync = library.sync_h2d_staging_buffer()?.unwrap();
    let ordered = library.ordered_h2d_staging_buffer(dst, stream)?.unwrap();
    assert_eq!(sync.bytes, 64);
    assert_eq!(ordered.bytes, dst.bytes);
    assert_ne!(sync.ptr, ordered.ptr);
    assert!(library.ordered_h2d_staging_buffer(dst, 2usize as *mut c_void)?.is_none());
    assert!(library.ordered_h2d_staging_buffer(CuteafdDeviceBuffer { device_id: 1, ..dst }, stream)?.is_none());
    assert_eq!(library.sync_h2d_staging_buffer()?.unwrap().ptr, sync.ptr);
    assert_eq!(library.ordered_h2d_staging_buffer(dst, stream)?.unwrap().ptr, ordered.ptr);
    assert_eq!(audit.checkpoint(), position);
    assert_eq!(fixture.events()?, before, "getters never allocate, copy, fence or free");
    assert_eq!(library.cuda_get_device()?, 0);
    library.release_ordered_h2d_staging()?;
    library.release_sync_h2d_staging()?;
    assert!(library.sync_h2d_staging_buffer()?.is_none());
    assert!(library.ordered_h2d_staging_buffer(dst, stream)?.is_none());
    library.free_device_buffer(&mut dst)?;
    Ok(())
}

#[test]
#[ignore = "requires an allocated CPU build slot and explicit NVMe fixture directory"]
fn ordered_upload_retains_each_source_until_its_copy_event() -> Result<()> {
    let _ledger = memory_ledger::test_guard();
    let fixture = Fixture::build()?;
    let library = fixture.load()?;
    let mut a = library.alloc_device_buffer(4, crate::memory_ledger::TEST)?;
    let mut b = library.alloc_device_buffer(4, crate::memory_ledger::TEST)?;
    let stream = 1usize as *mut c_void;
    // SAFETY: the CPU fixture owns both buffers and accepts a synthetic stream.
    unsafe {
        library.copy_h2d_ordered(a, &[1, 2, 3, 4], stream)?;
        library.copy_h2d_ordered(b, &[5, 6, 7, 8], stream)?;
        library.copy_h2d_ordered(a, &[9, 10, 11, 12], stream)?;
        assert_eq!(std::slice::from_raw_parts(a.ptr.cast::<u8>(), 4), &[1, 2, 3, 4],
            "reuse waits only through the previous upload before replacing its source");
    }
    library.release_ordered_h2d_staging()?;
    // SAFETY: the completed fixture uploads initialized these live four-byte buffers.
    unsafe {
        assert_eq!(std::slice::from_raw_parts(a.ptr.cast::<u8>(), 4), &[9, 10, 11, 12]);
        assert_eq!(std::slice::from_raw_parts(b.ptr.cast::<u8>(), 4), &[5, 6, 7, 8]);
    }
    assert!(fixture.events()?.contains("CAQR"));
    library.free_device_buffer(&mut a)?;
    library.free_device_buffer(&mut b)?;
    Ok(())
}

#[test]
#[ignore = "requires an allocated CPU build slot and explicit NVMe fixture directory"]
fn ordered_upload_sizes_staging_once_and_refuses_growth() -> Result<()> {
    let _ledger = memory_ledger::test_guard();
    // Pinned staging is created once per destination at the destination's whole
    // extent. A later upload never frees/reallocates it: cudaFreeHost would
    // synchronize the device while a peer wait may be queued (slot reuse).
    let fixture = Fixture::build()?;
    let library = fixture.load()?;
    let mut a = library.alloc_device_buffer(16, crate::memory_ledger::TEST)?;
    let mut b = library.alloc_device_buffer(16, crate::memory_ledger::TEST)?;
    let stream = 1usize as *mut c_void;
    let allocations = |events: &str| events.matches('A').count();
    // SAFETY: the CPU fixture owns both buffers and accepts a synthetic stream.
    unsafe {
        let before = allocations(&fixture.events()?);
        library.copy_h2d_ordered(a, &[1; 4], stream)?;
        let first = allocations(&fixture.events()?);
        assert_eq!(first, before + 1, "first upload creates the destination's staging");
        library.copy_h2d_ordered(a, &[2; 16], stream)?;
        assert_eq!(allocations(&fixture.events()?), first, "a longer source reuses whole-extent staging");
        // A view narrowed below the first source's extent shares the key but not the size.
        let narrow = CuteafdDeviceBuffer { bytes: 4, ..b };
        library.copy_h2d_ordered(narrow, &[3; 4], stream)?;
        let events = fixture.events()?;
        let error = library.copy_h2d_ordered(b, &[4; 16], stream).unwrap_err().to_string();
        assert!(error.contains("exceeds the 4-byte staging"), "{error}");
        assert_eq!(fixture.events()?, events, "a refused grow neither frees nor allocates host memory");
        // Reservation before readiness sizes the staging with no copy.
        let mut c = library.alloc_device_buffer(32, crate::memory_ledger::TEST)?;
        library.reserve_h2d_ordered(c, stream)?;
        let reserved = fixture.events()?;
        library.reserve_h2d_ordered(c, stream)?;
        library.copy_h2d_ordered(CuteafdDeviceBuffer { bytes: 8, ..c }, &[5; 8], stream)?;
        library.copy_h2d_ordered(c, &[6; 32], stream)?;
        assert_eq!(allocations(&fixture.events()?), allocations(&reserved), "reserved staging is never replaced");
        assert!(!fixture.events()?[reserved.len()..].contains('F'));
        library.release_ordered_h2d_staging()?;
        assert_eq!(std::slice::from_raw_parts(c.ptr.cast::<u8>(), 32), &[6; 32]);
        library.free_device_buffer(&mut c)?;
    }
    library.free_device_buffer(&mut a)?;
    library.free_device_buffer(&mut b)?;
    Ok(())
}

#[test]
#[ignore = "requires an allocated CPU build slot and explicit NVMe fixture directory"]
fn ordered_upload_failed_fence_retains_sources_and_module() -> Result<()> {
    let _ledger = memory_ledger::test_guard();
    let fixture = Fixture::build()?;
    let library = fixture.load()?;
    let a = library.alloc_device_buffer(4, crate::memory_ledger::TEST)?;
    // SAFETY: fixture-owned buffer and synthetic stream stay live for this test.
    unsafe { library.copy_h2d_ordered(a, &[1, 2, 3, 4], 1usize as *mut c_void)?; }
    fixture.configure_pack(&library, 0, 1)?;
    // SAFETY: same fixture stream and destination as above.
    assert!(unsafe { library.copy_h2d_ordered(a, &[9; 4], 1usize as *mut c_void) }.is_err());
    assert!(library.is_quarantined_after_failed_drain());
    let before = fixture.events()?;
    drop(library);
    assert_eq!(fixture.events()?, before, "failed fence retains pinned source and module");
    assert!(fixture.resident()?);
    Ok(())
}

#[test]
#[ignore = "CPU-only budget fixture; requires an explicit NVMe fixture directory"]
fn coordinator_budget_refuses_before_allocating_and_checks_untracked_graphs() -> Result<()> {
    let _ledger = memory_ledger::test_guard();
    // The process-wide ceiling is immutable; isolate this test from the other
    // unit tests rather than changing production configuration back and forth.
    if std::env::var_os("CUTEAFD_BUDGET_FIXTURE_CHILD").is_none() {
        let status = std::process::Command::new(std::env::current_exe()?)
            .args(["--exact", "native_library_lifetime::coordinator_budget_refuses_before_allocating_and_checks_untracked_graphs",
                "--ignored", "--nocapture"])
            .env("CUTEAFD_BUDGET_FIXTURE_CHILD", "1").status()?;
        anyhow::ensure!(status.success(), "CPU budget fixture child failed");
        return Ok(());
    }
    let fixture = Fixture::build()?;
    let library = fixture.load()?;
    std::fs::write(fixture.directory().join("events"), "")?;
    // SAFETY: the fixture's control export accepts only synthetic byte counts;
    // no GPU or CUDA runtime is present in this DSO.
    unsafe {
        let configure: Symbol<unsafe extern "C" fn(i32, usize, usize, usize)> =
            library.lib.get(b"fixture_configure_memory")?;
        configure(0, 4096, 100, 400);
        configure(1, 512, 100, 400);
    }
    assert_eq!(library.cuda_memory_info()?, (3996, 4096), "no budget preserves the physical sample");
    set_coordinator_gpu_budget(1024)?;
    assert!(fixture.events()?.is_empty(), "installing a ceiling allocates no physical guard");
    assert_eq!(library.cuda_physical_memory_info()?, (3996, 4096));
    assert_eq!(library.cuda_memory_info()?, (924, 1024));
    let mut weights = library.alloc_device_buffer(300, crate::memory_ledger::TEST)?;
    let mut draft = library.alloc_managed_device_buffer(200, crate::memory_ledger::TEST)?;
    assert_eq!(library.cuda_physical_memory_info()?, (3696, 4096), "managed fixture pages stay on host");
    assert_eq!(library.cuda_memory_info()?, (424, 1024));
    let events = fixture.events()?;
    let error = library.alloc_device_buffer(425, crate::memory_ledger::TEST).unwrap_err().to_string();
    assert!(error.contains("shortfall 1 bytes"), "{error}");
    assert_eq!(fixture.events()?, events, "refused before the native allocator");
    // SAFETY: the CPU graph fixture accepts a null stream and returns an
    // unlaunched executable owned by this test.
    let graph = unsafe { library.cuda_graph_end_capture(std::ptr::null_mut())? };
    assert_eq!(library.cuda_memory_info()?, (24, 1024));
    // SAFETY: graph belongs to this fixture and has never launched.
    let error = unsafe { library.cuda_graph_end_capture(std::ptr::null_mut()) }.unwrap_err();
    assert!(format!("{error:#}").contains("shortfall 376 bytes"), "{error:#}");
    assert_eq!(library.cuda_memory_info()?, (24, 1024), "over-budget graph destroyed before publishing");
    // SAFETY: the retained graph has never launched.
    unsafe { library.cuda_graph_exec_destroy(graph)? };
    library.free_device_buffer(&mut draft)?;
    library.free_device_buffer(&mut weights)?;
    library.cuda_set_device(1)?;
    assert_eq!(library.cuda_memory_info()?, (412, 512), "budget never enlarges a physical device");
    let error = library.alloc_device_buffer(413, crate::memory_ledger::TEST).unwrap_err().to_string();
    assert!(error.contains("GPU 1") && error.contains("shortfall 1 bytes"), "{error}");
    Ok(())
}

#[test]
#[ignore = "requires an allocated CPU build slot and explicit NVMe fixture directory"]
fn normal_drop_frees_pinned_staging_before_unloading_module() -> Result<()> {
    let _ledger = memory_ledger::test_guard();
    let fixture = Fixture::build()?;
    let library = fixture.load()?;
    library
        .sync_h2d_staging
        .lock()
        .unwrap()
        .ensure(&library, 64)?;
    assert!(fixture.resident()?);
    assert_eq!(fixture.events()?, "A");
    drop(library);
    assert_eq!(fixture.events()?, "AFU");
    assert!(
        !fixture.resident()?,
        "ordinary Drop must unload the actual module"
    );
    Ok(())
}

#[test]
#[ignore = "requires an allocated CPU build slot and explicit NVMe fixture directory"]
fn failed_drain_marker_retains_module_and_pinned_staging() -> Result<()> {
    let _ledger = memory_ledger::test_guard();
    let fixture = Fixture::build()?;
    let library = fixture.load()?;
    library
        .sync_h2d_staging
        .lock()
        .unwrap()
        .ensure(&library, 64)?;
    library.quarantine_module_after_failed_drain();
    library.quarantine_module_after_failed_drain(); // Repeated caller drains are harmless.
    assert!(library.is_quarantined_after_failed_drain());
    drop(library);
    assert_eq!(
        fixture.events()?,
        "A",
        "quarantine must neither free pinned staging nor unload the module"
    );
    assert!(
        fixture.resident()?,
        "the actual DSO must remain mapped after owner Drop"
    );

    // The failure is local to one NativeLibrary, not a global cleanup switch.
    let healthy = Fixture::build()?;
    let library = healthy.load()?;
    assert!(!library.is_quarantined_after_failed_drain());
    library
        .sync_h2d_staging
        .lock()
        .unwrap()
        .ensure(&library, 64)?;
    drop(library);
    assert_eq!(healthy.events()?, "AFU");
    assert!(!healthy.resident()?);
    assert!(fixture.resident()?);
    Ok(())
}

#[test]
#[ignore = "requires an allocated CPU build slot and explicit NVMe fixture directory"]
fn event_waiters_retain_module_until_workers_join() -> Result<()> {
    let fixture = Fixture::build()?;
    let library = fixture.load()?;
    let event = library.cuda_event_create()?;
    let destroy = library.cuteafd_cuda_event_destroy()?;
    // SAFETY: the fixture event is live and the fixture accepts a null stream.
    unsafe { library.cuda_event_record(event, std::ptr::null_mut())?; }
    let mut waiters = crate::event_waiters::EventWaiters::new(&library, 0, 4)?;
    drop(library);
    assert!(fixture.resident()?, "idle workers must retain their function pointers' DSO");
    // SAFETY: workers retain the DSO; the recorded event stays live until completion.
    unsafe { waiters.submit(0, event, 37)?; }
    assert_eq!(waiters.complete()?, 37);
    assert!(fixture.resident()?);
    assert!(!fixture.events()?.contains('U'));
    // SAFETY: successful completion drained this event and workers retain destroy's DSO.
    assert_eq!(unsafe { destroy(event) }, 0);
    waiters.join()?;
    assert!(!fixture.resident()?, "joined workers must release the healthy DSO");
    assert!(fixture.events()?.ends_with("EU"));
    Ok(())
}


/// A failed allocation records nothing, but native still reports the tag it
/// was asked for; an untagged request is refused before any allocation.
#[test]
fn failed_allocations_keep_the_attempted_tag_and_untagged_requests_are_refused() -> Result<()> {
    let _ledger = memory_ledger::test_guard();
    let fixture = crate::native_library_lifetime_fixture::Fixture::build_for_tests()?;
    let library = fixture.load()?;
    let before = memory_ledger::snapshot().total(memory_ledger::Space::Device, 0);
    let tag = memory_ledger::Tag::kv("test/oversized-kv");
    assert!(library.alloc_device_buffer(usize::MAX / 2, tag).is_err());
    assert_eq!(memory_ledger::snapshot().total(memory_ledger::Space::Device, 0), before);
    assert_eq!(library.last_allocation_tag()?, Some(tag));
    // SAFETY: fixture export with the native ABI; null owner is refused before allocation.
    let allocate = unsafe { *library.lib.get::<unsafe extern "C" fn(usize, *mut CuteafdDeviceBuffer, u32, *const c_char) -> i32>(
        b"cuteafd_alloc_device_buffer")? };
    let mut buffer = CuteafdDeviceBuffer::default();
    for (kind, owner) in [(0u32, c"test/x".as_ptr()), (13, c"test/x".as_ptr()), (9, c"".as_ptr()), (9, std::ptr::null())] {
        assert_ne!(unsafe { allocate(64, &mut buffer, kind, owner) }, 0);
        assert!(buffer.ptr.is_null());
    }
    // An owner of any length round-trips whole: the API sizes, then reads.
    let long = format!("test/{}", "o".repeat(400));
    let long_tag = memory_ledger::tag(memory_ledger::MemoryKind::Workspace, &long);
    let mut buffer = library.alloc_device_buffer(64, long_tag)?;
    assert_eq!(library.last_allocation_tag()?, Some(long_tag));
    library.free_device_buffer(&mut buffer)?;
    let mut host = library.alloc_host_buffer(64, memory_ledger::Tag::staging("test/host"))?;
    assert_eq!(memory_ledger::tag_of(host.ptr as usize), Some(memory_ledger::Tag::staging("test/host")));
    library.free_host_buffer(&mut host)?;
    Ok(())
}

#[test]
fn synchronous_upload_chunks_preserve_bytes_odd_tails_and_fixed_peak_staging() -> Result<()> {
    let _ledger = memory_ledger::test_guard();
    let fixture = Fixture::build_for_tests()?;
    let library = fixture.load()?;
    fixture.immediate_copies(&library)?;
    let capacity = 2 * SYNC_H2D_CHUNK_BYTES + 19;
    let mut dst = library.alloc_device_buffer(capacity, memory_ledger::TEST)?;
    let source = (0..capacity).map(|index| ((index * 17 + index / 251) % 256) as u8).collect::<Vec<_>>();
    let mut output = vec![0; capacity];
    let mut peak = 0;
    for bytes in [1, SYNC_H2D_CHUNK_BYTES - 1, SYNC_H2D_CHUNK_BYTES,
        SYNC_H2D_CHUNK_BYTES + 1, capacity] {
        library.copy_h2d(dst, &source[..bytes])?;
        library.copy_d2h(&mut output[..bytes], dst)?;
        assert_eq!(&output[..bytes], &source[..bytes], "{bytes} byte upload");
        peak = peak.max(bytes.min(SYNC_H2D_CHUNK_BYTES));
        assert_eq!(library.sync_h2d_staging_buffer()?.unwrap().bytes, peak);
    }
    let staging = library.sync_h2d_staging_buffer()?.unwrap();
    let before = fixture.events()?;
    let short = [0x93; 17];
    library.copy_h2d(dst, &short)?;
    library.copy_d2h(&mut output, dst)?;
    assert_eq!(&output[..short.len()], &short);
    assert_eq!(&output[short.len()..], &source[short.len()..]);
    assert_eq!(library.sync_h2d_staging_buffer()?.unwrap().ptr, staging.ptr);
    assert_eq!(fixture.events()?.matches('A').count(), before.matches('A').count());
    assert_eq!(library.release_sync_h2d_staging()?, SYNC_H2D_CHUNK_BYTES);
    assert!(library.sync_h2d_staging_buffer()?.is_none());
    library.copy_h2d(dst, &short)?;
    assert_eq!(library.sync_h2d_staging_buffer()?.unwrap().bytes, short.len());
    library.release_sync_h2d_staging()?;
    library.free_device_buffer(&mut dst)?;
    Ok(())
}

#[test]
fn sealed_synchronous_uploads_cannot_regrow_but_ordered_uploads_remain_valid() -> Result<()> {
    let _ledger = memory_ledger::test_guard();
    let fixture = Fixture::build_for_tests()?;
    let library = fixture.load()?;
    fixture.immediate_copies(&library)?;
    let mut dst = library.alloc_device_buffer(8, memory_ledger::TEST)?;
    assert!(library.copy_h2d(CuteafdDeviceBuffer { bytes: 1, ..dst }, &[1, 2]).is_err());
    assert!(library.sync_h2d_staging_buffer()?.is_none());
    library.copy_h2d(dst, &[1; 8])?;
    assert_eq!(library.seal_sync_h2d_staging()?, 8);
    assert!(library.sync_h2d_staging_buffer()?.is_none());
    assert!(library.copy_h2d(dst, &[2]).unwrap_err().to_string().contains("sealed after startup"));
    assert!(library.sync_h2d_staging_buffer()?.is_none());
    assert_eq!(library.release_sync_h2d_staging()?, 0);
    assert!(library.copy_h2d(dst, &[3]).is_err());
    library.copy_h2d(dst, &[])?;
    let stream = 1usize as *mut c_void;
    // SAFETY: CPU fixture owns the destination and accepts this synthetic stream.
    unsafe { library.copy_h2d_ordered(dst, &[4; 8], stream)?; }
    library.release_ordered_h2d_staging()?;
    let mut output = [0; 8];
    library.copy_d2h(&mut output, dst)?;
    assert_eq!(output, [4; 8]);
    library.free_device_buffer(&mut dst)?;
    Ok(())
}

#[test]
fn small_graph_capture_is_typed_and_keeps_native_node_count() -> Result<()> {
    let fixture = native_library_lifetime_fixture::Fixture::build_for_tests()?;
    let library = fixture.load()?;
    fixture.configure_capture(&library, CUTEAFD_STATUS_GRAPH_TOO_SMALL)?;
    // SAFETY: the CPU fake has no stream or device work.
    let error = unsafe { library.cuda_graph_end_capture_topology(std::ptr::null_mut()) }.unwrap_err();
    assert_eq!(error.downcast_ref::<SmallGraphCapture>(), Some(&SmallGraphCapture { nodes: 2 }));
    assert!(is_small_graph(&error));
    assert!(error.downcast_ref::<NativeStatusError>().is_none());
    Ok(())
}
