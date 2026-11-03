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
fn peer_copy_trace_resolves_live_owners_and_preserves_copy_modes() -> Result<()> {
    use crate::native_library_lifetime_fixture::CopyKind;
    for mode in 0..3 {
        let fixture = Fixture::build()?;
        let library = fixture.load()?;
        for device in [0, 1] { fixture.configure_memory(&library, device, 1 << 20)?; }
        match mode {
            1 => fixture.immediate_copies(&library)?,
            2 => fixture.queued_experts(&library)?,
            _ => {},
        }
        library.cuda_set_device(1)?;
        let mut source = library.alloc_device_buffer(256)?;
        let input = (0..256).map(|i| (i % 251 + 1) as u8).collect::<Vec<_>>();
        library.copy_h2d(source, &input)?;
        library.cuda_set_device(0)?;
        let mut destination = library.alloc_device_buffer(256)?;
        let view = |owner: CuteafdDeviceBuffer, offset, bytes| {
            // SAFETY: all calls below use bounded offsets into these live owners.
            CuteafdDeviceBuffer { ptr: unsafe { owner.ptr.cast::<u8>().add(offset).cast() },
                bytes, flags: 0, ..owner }
        };
        fixture.record_peer_copies(&library)?;
        let peer = library.v41_peer_copy()?;
        let stream = 1usize as *mut c_void;
        // SAFETY: disjoint live fixture allocations and bounded views; owners and
        // source bytes are retained until explicit advance/drain below.
        unsafe {
            peer.launch(view(destination, 32, 64), view(source, 17, 64), 64, stream)?;
            peer.launch_rows(view(destination, 128, 27), view(source, 128, 23), 5, 3, 11, 9, stream)?;
            library.copy_d2d_async(view(destination, 192, 16), view(source, 192, 16), 16, stream)?;
            library.copy_d2d_2d_async(view(destination, 224, 12), 8,
                view(source, 224, 12), 8, 4, 2, stream)?;
        }
        let records = fixture.peer_copy_records(&library)?;
        assert_eq!(records.iter().map(|r| r.kind).collect::<Vec<_>>(),
            [CopyKind::Initialize, CopyKind::SmLinear, CopyKind::SmRows,
                CopyKind::DmaLinear, CopyKind::DmaRows]);
        assert_eq!((records[0].device, records[0].destination_device, records[0].source_device), (0, -1, -1));
        for (record, (dst, src, width, rows, dst_pitch, src_pitch)) in records[1..].iter().zip([
            (32, 17, 64, 1, 64, 64), (128, 128, 5, 3, 11, 9),
            (192, 192, 16, 1, 16, 16), (224, 224, 4, 2, 8, 8)]) {
            assert_eq!((record.device, record.destination_device, record.source_device), (0, 0, 1));
            assert_eq!((record.destination_backing, record.source_backing),
                (destination.ptr as u64, source.ptr as u64));
            assert_eq!((record.destination_offset, record.source_offset), (dst, src));
            assert_eq!((record.destination, record.source),
                (destination.ptr as u64 + dst, source.ptr as u64 + src));
            assert_eq!((record.bytes, record.width, record.rows, record.destination_pitch,
                record.source_pitch, record.stream), (width * rows, width, rows, dst_pitch, src_pitch, 1));
        }
        let mut landed = vec![0; 256];
        library.copy_d2h(&mut landed, destination)?;
        if mode != 1 { assert_eq!(&landed[32..96], &[0; 64]); }
        if mode == 2 {
            fixture.advance_queued_experts(&library, 1)?;
            library.copy_d2h(&mut landed, destination)?;
            assert_eq!(&landed[32..96], &input[17..81]);
            assert_eq!(&landed[128..133], &[0; 5], "one advance must not run the next copy");
        }
        fixture.drain(&library)?;
        library.copy_d2h(&mut landed, destination)?;
        assert_eq!(&landed[32..96], &input[17..81]);
        for row in 0..3 { assert_eq!(&landed[128 + row * 11..133 + row * 11], &input[128 + row * 9..133 + row * 9]); }
        assert_eq!(&landed[192..208], &input[192..208]);
        for row in 0..2 { assert_eq!(&landed[224 + row * 8..228 + row * 8], &input[224 + row * 8..228 + row * 8]); }
        assert_eq!(fixture.peer_copy_records(&library)?, records, "draining does not submit more copies");
        // Rust operation validation still rejects owning buffers, not just bad extents.
        // SAFETY: every negative uses retained allocations and cannot enqueue unsafe work.
        unsafe {
            assert!(peer.launch(destination, source, 4, stream).is_err());
            assert!(peer.launch(view(destination, 0, 4), view(source, 0, 4), 5, stream).is_err());
            assert!(peer.launch(view(destination, 0, 4), view(source, 0, 4), 4, std::ptr::null_mut()).is_err());
            library.cuda_set_device(1)?;
            assert!(peer.launch(view(destination, 0, 4), view(source, 0, 4), 4, stream).is_err());
            library.cuda_set_device(0)?;
            assert!(peer.launch_rows(view(destination, 0, 16), view(destination, 0, 16),
                4, 2, 8, 8, stream).is_err());
        }
        assert_eq!(fixture.peer_copy_records(&library)?, records);
        // The registry must reject a stale source even if the raw view metadata is unchanged.
        let stale = view(source, 0, 4);
        library.free_device_buffer(&mut source)?;
        // SAFETY: the stale pointer is submitted only to the CPU fixture, which
        // must reject it before dereferencing it. No allocation can reuse it here.
        unsafe { assert!(peer.launch(view(destination, 0, 4), stale, 4, stream).is_err()); }
        library.free_device_buffer(&mut destination)?;
    }
    Ok(())
}

#[test]
#[ignore = "requires an allocated CPU build slot and explicit NVMe fixture directory"]
fn peer_rows_keep_local_views_and_deferred_sources_until_their_fence() -> Result<()> {
    use crate::native_library_lifetime_fixture::CopyKind;
    let fixture = Fixture::build()?;
    let library = fixture.load()?;
    let mut owner = library.alloc_device_buffer(128)?;
    library.copy_h2d(owner, &[7; 128])?;
    let view = |offset, bytes| CuteafdDeviceBuffer {
        // SAFETY: these bounded interior views share the retained 128-byte owner.
        ptr: unsafe { owner.ptr.cast::<u8>().add(offset).cast() }, bytes, flags: 0, ..owner
    };
    fixture.record_peer_copies(&library)?;
    let peer = library.v41_peer_copy()?;
    let stream = 1usize as *mut c_void;
    // SAFETY: source/destination planes are disjoint and live through the event.
    unsafe { peer.launch_rows(view(64, 24), view(0, 24), 8, 2, 16, 16, stream)?; }
    let record = &fixture.peer_copy_records(&library)?[1];
    assert_eq!(record.kind, CopyKind::SmRows);
    assert_eq!((record.source_device, record.destination_device), (0, 0));
    assert_eq!((record.source_backing, record.destination_backing), (owner.ptr as u64, owner.ptr as u64));
    // Mutating the CPU stand-in source before completion proves the fixture did
    // not eagerly snapshot it; this is not permission to race a real CUDA copy.
    library.copy_h2d(view(0, 24), &[9; 24])?;
    let event = library.cuda_event_create_ordering()?;
    // SAFETY: fixture event and stream remain live, and failed drains do no work.
    unsafe { library.cuda_event_record(event, stream)?; }
    fixture.configure_pack(&library, 0, 1)?;
    // SAFETY: this event was recorded on the synthetic stream above.
    assert!(unsafe { library.cuda_event_synchronize(event) }.is_err());
    let mut landed = vec![0; 128];
    library.copy_d2h(&mut landed, owner)?;
    assert_eq!(&landed[64..72], &[7; 8]);
    fixture.configure_pack(&library, 0, 0)?;
    // SAFETY: owners survive both the failed and successful completion attempts.
    unsafe { library.cuda_event_synchronize(event)?; library.cuda_event_destroy(event)?; }
    library.copy_d2h(&mut landed, owner)?;
    assert_eq!(&landed[64..72], &[9; 8]);
    assert_eq!(&landed[80..88], &[9; 8]);
    library.free_device_buffer(&mut owner)?;
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
    let reserved = library.reserve_sync_h2d_staging(64)?;
    let retained = library.sync_h2d_staging_buffer()?;
    assert_eq!((reserved.ptr, reserved.bytes), (retained.ptr, 64));
    let smaller = library.reserve_sync_h2d_staging(32)?;
    assert_eq!((smaller.ptr, smaller.bytes), (retained.ptr, 64));
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
