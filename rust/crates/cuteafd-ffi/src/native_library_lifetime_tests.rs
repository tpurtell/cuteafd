// Included as a unit module so the fixture can exercise the real private
// staging owner without adding production test hooks. These directed tests
// build a tiny CPU-only DSO and require an explicitly allocated build slot:
// CUTEAFD_NATIVE_LIFETIME_FIXTURE_DIR=$TASK/fixtures cargo test -p cuteafd-ffi \
//   --lib native_library_lifetime -- --ignored --nocapture
use super::*;
use crate::native_library_lifetime_fixture::Fixture;

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
fn ordered_upload_retains_each_source_until_its_copy_event() -> Result<()> {
    let fixture = Fixture::build()?;
    let library = fixture.load()?;
    let mut a = library.alloc_device_buffer(4)?;
    let mut b = library.alloc_device_buffer(4)?;
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
    // Pinned staging is created once per destination at the destination's whole
    // extent. A later upload never frees/reallocates it: cudaFreeHost would
    // synchronize the device while a peer wait may be queued (slot reuse).
    let fixture = Fixture::build()?;
    let library = fixture.load()?;
    let mut a = library.alloc_device_buffer(16)?;
    let mut b = library.alloc_device_buffer(16)?;
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
        let mut c = library.alloc_device_buffer(32)?;
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
    let fixture = Fixture::build()?;
    let library = fixture.load()?;
    let a = library.alloc_device_buffer(4)?;
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
    let mut weights = library.alloc_device_buffer(300)?;
    let mut draft = library.alloc_managed_device_buffer(200)?;
    assert_eq!(library.cuda_physical_memory_info()?, (3696, 4096), "managed fixture pages stay on host");
    assert_eq!(library.cuda_memory_info()?, (424, 1024));
    let events = fixture.events()?;
    let error = library.alloc_device_buffer(425).unwrap_err().to_string();
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
    let error = library.alloc_device_buffer(413).unwrap_err().to_string();
    assert!(error.contains("GPU 1") && error.contains("shortfall 1 bytes"), "{error}");
    Ok(())
}

#[test]
#[ignore = "requires an allocated CPU build slot and explicit NVMe fixture directory"]
fn normal_drop_frees_pinned_staging_before_unloading_module() -> Result<()> {
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
