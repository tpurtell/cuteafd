use super::*;
use std::future::Future;
use std::io::{Seek, SeekFrom, Write};

type TestResult = std::result::Result<(), Box<dyn std::error::Error>>;

#[test]
fn fixed_staging_charges_actual_wave_count_and_separate_decode_rows() -> TestResult {
    for k in 1..=4 {
        let staging = MappedStaging::new(2048 * 16, k, 64 * 16, 320)?;
        assert_eq!(staging.slot_bytes().collect::<Vec<_>>(),
            std::iter::repeat_n(2048 * 16 * 320, 2 * k).chain(std::iter::repeat_n(64 * 16 * 320, 2)).collect::<Vec<_>>());
        assert_eq!(staging.workspace_bytes, (2048 + 64) * 16 * 1024 + (2 << 20));
        assert_eq!(staging.total_bytes, 2 * (k * 2048 + 64) * 16 * 320 + staging.workspace_bytes);
        assert_eq!(staging.slot_bytes().sum::<usize>() + staging.workspace_bytes, staging.total_bytes);
    }
    for (rows, k, decode, bytes) in [(0, 1, 64, 1), (1, 0, 64, 1), (1, 5, 64, 1),
        (1, 1, 0, 1), (1, 1, 64, 0), (usize::MAX, 1, 64, 320), (usize::MAX / 2, 4, 64, 1)] {
        assert!(MappedStaging::new(rows, k, decode, bytes).is_err());
    }
    Ok(())
}

#[test]
fn live_reader_loads_only_atomics_without_interval_or_device_locks() {
    let stats = Arc::new(TableStats::default());
    let reader = MappedTableStatsReader { tables: vec![("lock-free".into(), Arc::downgrade(&stats))] };
    let _previous = stats.previous.lock().unwrap();
    let _devices = stats.devices.lock().unwrap();
    let _previous_devices = stats.previous_devices.lock().unwrap();
    let _registry = TABLES.lock().unwrap();
    let initial = reader.snapshot();
    assert_eq!(initial[0].cumulative, TableStatsSnapshot::default());
    stats.record_gather(2, 8, Duration::from_micros(4), [0; 3]);
    let fresh = reader.snapshot();
    assert_eq!(fresh[0].backend, stats.backend().to_string());
    assert_eq!(fresh[0].cumulative.gathers, 1);
    assert_eq!(fresh[0].cumulative.rows, 2);
    assert!(fresh[0].interval.is_none());
    assert!(fresh[0].host_wide_device_reads.is_empty());
    drop(_previous);
    drop(_devices);
    drop(_previous_devices);
    drop(stats);
    assert!(reader.snapshot().is_empty());
}

#[test]
fn uring_setup_failure_falls_back_without_changing_bytes() -> TestResult {
    let (_dir, parts) = parted([3, 3, 2])?;
    // SAFETY: the test owns immutable shard files through the table's lifetime.
    let table = unsafe { MappedTable::open(&parts, u8_rows(4))? };
    table.initialize_with_probe(|| Err(MappedTableError::Io {
        context: "injected io_uring_setup seccomp failure".into(),
        source: std::io::Error::from_raw_os_error(libc::EPERM),
    }));
    assert_eq!(table.backend() == TableBackend::MincoreRouted, reactor::ready());
    let mut output = [0; 12];
    table.gather_into(&[7, 2, 7], &mut output)?;
    assert_eq!(output, [7, 7, 7, 7, 2, 2, 2, 2, 7, 7, 7, 7]);
    assert_eq!(table.stats().snapshot().page_hits + table.stats().snapshot().page_misses, 3);
    Ok(())
}

#[test]
fn runtime_batch_failure_recovers_without_disabling_table() -> TestResult {
    let (_dir, parts) = parted([3, 3, 2])?;
    // SAFETY: this test owns immutable files through all owned reactor reads.
    let table = unsafe { MappedTable::open(&parts, u8_rows(4))? };
    table.initialize_reads();
    if table.backend() != TableBackend::MincoreRouted { return Ok(()); }
    for errno in [libc::ENOMEM, libc::EPERM] {
        let read = || uring::Read { part: &table.parts[0], row: 2, slot: 0 };
        let mut bytes = [0; 4];
        uring::fail_next_setup(errno);
        assert!(uring::gather(&[read()], &mut bytes, 4, &[Some(false)]).is_err());
        uring::gather(&[read()], &mut bytes, 4, &[Some(false)])?;
        assert_eq!(bytes, [2; 4]);
        assert_eq!(table.backend(), TableBackend::MincoreRouted);
    }
    Ok(())
}

#[test]
fn miss_histogram_tiers_and_interval_counts_are_exact() {
    for (ns, tier) in [(0, 0), (4999, 0), (5000, 1), (9999, 1), (10000, 2),
        (50000, 3), (200000, 4), (1000000, 5), (10000000, 6)] {
        assert_eq!(uring::tier(ns), tier);
    }
    let stats = TableStats::default();
    let before = stats.snapshot();
    stats.record_reads(uring::ReadStats { hits: 2, misses: 3, hit_bytes: 16,
        miss_bytes: 24, device_bytes_estimate: 12288, miss_histogram: [0, 1, 2, 0, 0, 0, 0], miss_ns: 35000,
        miss_max_ns: 15000, nowait_ns: 7000, nowait_max_ns: 7000, nowait_batches: 1, ..Default::default() }, false);
    let delta = stats.snapshot().since(&before);
    let first = stats.interval();
    assert_eq!(first.miss_max_ns, 15000);
    assert_eq!(first.nowait_max_ns, 7000);
    assert_eq!(stats.interval(), TableStatsSnapshot::default());
    stats.record_reads(uring::ReadStats { misses: 1, miss_ns: 5000, miss_max_ns: 5000, ..Default::default() }, false);
    assert_eq!(stats.interval().miss_max_ns, 5000);
    assert_eq!(stats.snapshot().miss_max_ns, 15000);
    assert_eq!((delta.page_hits, delta.page_misses, delta.page_hit_bytes, delta.miss_request_bytes), (2, 3, 16, 24));
    assert_eq!(delta.miss_histogram.iter().sum::<u64>(), delta.page_misses);
    assert_eq!((delta.miss_ns, delta.miss_max_ns, delta.nowait_ns), (35000, 15000, 7000));
    stats.record_reads(uring::ReadStats { hits: 4, misses: 5, ..Default::default() }, true);
    assert_eq!((stats.snapshot().page_hits, stats.snapshot().prefetch_hits), (2, 4));
}

#[test]
fn unclassified_batch_intervals_keep_units_and_lifetime_maxima() {
    let stats = TableStats::default();
    let reads = uring::ReadStats {
        unclassified_rows: 320, unclassified_bytes: 1280, unclassified_batches: 2,
        unclassified_batch_ns: 12000, unclassified_batch_max_ns: 7000,
        unclassified_batch_histogram: [0, 2, 0, 0, 0, 0, 0], ..Default::default()
    };
    stats.record_reads(reads, false);
    stats.record_reads(reads, true);
    let first = stats.interval();
    assert_eq!((first.unclassified_rows, first.prefetch_unclassified_rows), (320, 320));
    assert_eq!(first.unclassified_batch_histogram.iter().sum::<u64>(), first.unclassified_batches);
    assert_eq!(first.prefetch_unclassified_batch_histogram.iter().sum::<u64>(), first.prefetch_unclassified_batches);
    assert_eq!((first.unclassified_batch_ns, first.prefetch_unclassified_batch_ns), (12000, 12000));
    assert_eq!((first.unclassified_batch_max_ns, first.prefetch_unclassified_batch_max_ns), (7000, 7000));
    assert_eq!(stats.interval(), TableStatsSnapshot::default());
    stats.record_reads(uring::ReadStats { unclassified_batches: 1, unclassified_batch_max_ns: 3000, ..Default::default() }, true);
    assert_eq!(stats.interval().prefetch_unclassified_batch_max_ns, 3000);
    assert_eq!(stats.snapshot().since(&first).prefetch_unclassified_batch_max_ns, 7000);
    let mut historical = serde_json::to_value(first).unwrap();
    let fields = historical.as_object_mut().unwrap();
    fields.retain(|key, _| !key.contains("unclassified_batch"));
    fields.insert("completion_histogram".into(), serde_json::json!([0, 320, 0, 0, 0, 0, 0]));
    fields.insert("prefetch_completion_histogram".into(), serde_json::json!([0, 320, 0, 0, 0, 0, 0]));
    let old: TableStatsSnapshot = serde_json::from_value(historical).unwrap();
    assert_eq!((old.unclassified_rows, old.prefetch_unclassified_rows), (320, 320));
    assert_eq!((old.unclassified_batches, old.prefetch_unclassified_batches), (0, 0));
    assert_eq!(old.unclassified_batch_histogram, [0; 7], "legacy row tiers must not become batch tiers");
}

#[test]
fn unclassified_waves_count_batches_not_rows() -> TestResult {
    let (_dir, parts) = parted([3, 3, 2])?;
    // SAFETY: this test owns the immutable shards until all reads are drained.
    let table = unsafe { MappedTable::open(&parts, u8_rows(4))? };
    table.initialize_reads();
    if table.backend() != TableBackend::MincoreRouted {
        eprintln!("SKIP unclassified wave execution: kernel/seccomp disallows io_uring");
        return Ok(());
    }
    table.parts[0].nowait.store(false, Ordering::Relaxed);
    assert_eq!(uring::MAX_BATCH, 2048);
    let n = uring::MAX_BATCH + 1;
    let reads: Vec<_> = (0..n).map(|i| uring::Read { part: &table.parts[0], row: (i % 3) as u64, slot: n - i - 1 }).collect();
    // Exercise mixed classified hits and unclassified reads in two waves.
    let resident: Vec<_> = (0..n).map(|i| if i % 2 == 0 { None } else { Some(true) }).collect();
    let mut output = vec![0; n * 4];
    let stats = uring::gather(&reads, &mut output, 4, &resident)?;
    assert_eq!((stats.unclassified_rows, stats.hits, stats.misses), (1025, 1024, 0));
    assert_eq!(stats.unclassified_bytes, 1025 * 4);
    assert_eq!(stats.unclassified_batches, 1);
    assert_eq!(stats.unclassified_batch_histogram.iter().sum::<u64>(), 1);
    assert!(stats.unclassified_batch_ns >= stats.unclassified_batch_max_ns && stats.unclassified_batch_max_ns > 0);
    assert_eq!(stats.miss_histogram.iter().sum::<u64>(), n as u64);
    for i in 0..n {
        assert_eq!(&output[(n-i-1)*4..(n-i)*4], &[(i % 3) as u8; 4]);
    }
    let empty = uring::gather(&[], &mut [], 4, &[])?;
    assert_eq!((empty.unclassified_rows, empty.unclassified_batches, empty.unclassified_batch_ns), (0, 0, 0));
    Ok(())
}

#[test]
fn every_ring_bounds_its_io_wq_workers() -> TestResult {
    let (_dir, parts) = parted([3, 3, 2])?;
    // SAFETY: this test owns the immutable shards until the probe returns.
    let table = unsafe { MappedTable::open(&parts, u8_rows(4))? };
    table.initialize_reads();
    if table.backend() != TableBackend::MincoreRouted {
        eprintln!("SKIP io-wq bounds: kernel/seccomp disallows io_uring");
        return Ok(());
    }
    assert!(Arc::ptr_eq(&reactor::global()?, &reactor::global()?));
    assert_eq!(uring::IOWQ_MAX_WORKERS, 1024);
    Ok(())
}

#[test]
fn buffered_uring_matches_mmap_across_unaligned_parts_and_duplicates() -> TestResult {
    let (_dir, parts) = parted([3, 3, 2])?;
    // SAFETY: the test owns immutable shards until both tables are dropped.
    let mmap = unsafe { MappedTable::open(&parts, u8_rows(4))? };
    // SAFETY: same immutable test files.
    let uring = unsafe { MappedTable::open(&parts, u8_rows(4))? };
    uring.initialize_reads();
    if uring.backend() != TableBackend::MincoreRouted {
        eprintln!("SKIP io_uring execution: kernel/seccomp/filesystem disallows buffered NOWAIT");
        return Ok(());
    }
    let rows = [7, 0, 3, 3, 5, 2];
    let pool = GatherPool::new("uring-parity", 3)?;
    let mut expected = [0; 24];
    let mut actual = [0; 24];
    pool.gather(&mmap, &rows, &mut expected)?;
    pool.gather(&uring, &rows, &mut actual)?;
    assert_eq!(actual, expected);
    assert_eq!(uring.backend(), TableBackend::MincoreRouted);
    assert_eq!(uring.stats().snapshot().page_hits, 6);
    assert_eq!(uring.prefetch(&rows, 16)?, 3);
    let prefetched = uring.stats().snapshot();
    assert_eq!(prefetched.prefetch_hits, 6);
    assert_eq!(prefetched.prefetch_nowait_batches, u64::from(uring.stats.nowait.load(Ordering::Relaxed)));
    // Invalid batches must not mutate the output or queue any I/O.
    assert!(uring.gather_slots(&[0, 8], &[0, 1], &mut actual).is_err());
    assert_eq!(actual, expected);
    assert!(uring.gather_slots(&[0, 1], &[0, 0], &mut actual).is_err());
    Ok(())
}

#[test]
fn hybrid_and_fallback_gather_identical_random_cold_and_warm_rows() -> TestResult {
    let page = page_bytes()?;
    let width = page + 13;
    let counts = [67, 71];
    let total: usize = counts.iter().sum();
    let payload: Vec<_> = (0..total * width).map(|i| ((i * 17 + i / width) % 251) as u8).collect();
    let dir = tempfile::tempdir_in(std::env::var_os("TMPDIR").map(std::path::PathBuf::from)
        .unwrap_or(std::env::current_dir()?))?;
    let mut parts = Vec::new();
    let mut first = 0;
    for (index, &count) in counts.iter().enumerate() {
        let path = dir.path().join(format!("part-{index}"));
        let mut file = std::fs::File::create(&path)?;
        file.write_all(&[0; 17])?;
        file.write_all(&payload[first * width..(first + count) * width])?;
        file.sync_all()?;
        parts.push(TablePart { path, offset: 17, rows: count as u64 });
        first += count;
    }
    let mut random = 0x1234_5678_u64;
    let mut rows = vec![0, 66, 67, 137, 67, 0];
    for _ in 0..250 {
        random ^= random << 13;
        random ^= random >> 7;
        random ^= random << 17;
        rows.push(random % total as u64);
    }
    let slots: Vec<_> = (0..rows.len()).rev().collect();
    let mut expected = vec![0; rows.len() * width];
    for (&row, &slot) in rows.iter().zip(&slots) {
        expected[slot * width..(slot + 1) * width]
            .copy_from_slice(&payload[row as usize * width..(row as usize + 1) * width]);
    }
    for (backend, buffered) in [(TableBackend::Mmap, false), (TableBackend::MincoreRouted, false), (TableBackend::MincoreRouted, true)] {
        for mixed in [false, true] {
            // SAFETY: the test owns immutable shards until this table is dropped.
            let table = unsafe { MappedTable::open(&parts, u8_rows(width))? };
            table.initialize_reads();
            if table.backend() != backend {
                assert!(std::env::var("CUTEAFD_REQUIRE_URING").as_deref() != Ok("1"),
                    "required backend {backend:?} unavailable");
                eprintln!("SKIP {backend:?} parity: kernel/seccomp disallows io_uring");
                continue;
            }
            if buffered {
                for part in &table.parts { part.nowait.store(false, Ordering::Relaxed); }
                table.stats.nowait.store(false, Ordering::Relaxed);
            }
            // Evict whole files, including the partial page before the payload.
            // SAFETY: fadvise only operates on these live immutable test fds.
            for part in &table.parts {
                assert_eq!(unsafe { libc::posix_fadvise(part.file.as_raw_fd(), 0, 0,
                    libc::POSIX_FADV_DONTNEED) }, 0);
                let pages: Vec<_> = (0..part.mapped_len.div_ceil(page)).map(|p| (0, p)).collect();
                let mut resident = vec![false; pages.len()];
                uring::resident(part, &pages, &mut resident, &mut Vec::new())?;
                assert!(resident.iter().all(|&hit| !hit), "test file is not fully cold");
            }
            if mixed {
                let row = table.row(0)?;
                for &byte in row.iter().step_by(page) { std::hint::black_box(byte); }
            }
            let residency = table.residency(&rows, false)?;
            assert!(residency.iter().any(|&hit| !hit));
            assert_eq!(residency.iter().any(|&hit| hit), mixed);
            let before = table.stats().snapshot();
            let mut cold = vec![0xa5; expected.len()];
            table.gather_slots(&rows, &slots, &mut cold)?;
            assert_eq!(cold, expected, "cold/mixed {backend:?} buffered={buffered}");
            assert_eq!(table.backend(), backend, "runtime fallback is not qualification");
            let delta = table.stats().snapshot().since(&before);
            assert!(delta.page_misses > 0);
            if mixed && buffered && backend == TableBackend::MincoreRouted {
                assert!(delta.resident_copy_rows > 0, "resident memcpy path was not exercised");
            }
            assert!(table.residency(&rows, false)?.iter().all(|&resident| resident));
            let before = table.stats().snapshot();
            let mut warm = vec![0x5a; expected.len()];
            table.gather_slots(&rows, &slots, &mut warm)?;
            assert_eq!(warm, expected, "warm {backend:?} buffered={buffered}");
            assert_eq!(warm, cold);
            assert_eq!(table.backend(), backend);
            if buffered && backend == TableBackend::MincoreRouted {
                assert_eq!(table.stats().snapshot().since(&before).resident_copy_rows, rows.len() as u64);
            }
        }
    }
    Ok(())
}

#[test]
fn dontneed_produces_mincore_and_nowait_misses_then_hits() -> TestResult {
    // /tmp is often tmpfs (which cannot evict clean file pages). Cargo's
    // current working directory is the disk-backed workspace on our hosts.
    let mut file = tempfile::NamedTempFile::new_in(std::env::current_dir()?)?;
    let page = page_bytes()?;
    let payload: Vec<_> = (0..8 * page).map(|i| (i % 251) as u8).collect();
    file.write_all(&payload)?;
    file.as_file().sync_all()?;
    for backend in [TableBackend::Mmap, TableBackend::MincoreRouted] {
        // SAFETY: test file is immutable while the table lives.
        let table = unsafe { MappedTable::single(file.path(), 0, 8, u8_rows(page))? };
        table.split_only_for_test();
        table.initialize_reads();
        if backend == TableBackend::MincoreRouted && table.backend() != backend {
            eprintln!("SKIP NOWAIT miss execution: unsupported kernel/seccomp/filesystem");
            continue;
        }
        // No mapping has been touched. Drop only this clean test file's pages.
        // SAFETY: fadvise does not access userspace memory, and fd is live.
        assert_eq!(unsafe { libc::posix_fadvise(file.as_file().as_raw_fd(), 0,
            payload.len() as i64, libc::POSIX_FADV_DONTNEED) }, 0);
        let rows = [1, 3, 6];
        let mut out = vec![0; rows.len() * page];
        table.gather_into(&rows, &mut out)?;
        let cold = table.stats().snapshot();
        assert_eq!((cold.page_hits, cold.page_misses), (0, 3));
        assert_eq!(cold.miss_request_bytes, (3 * page) as u64);
        if backend == TableBackend::MincoreRouted {
            assert_eq!(cold.miss_histogram.iter().sum::<u64>(), 3);
            assert!(cold.miss_ns > 0 && cold.miss_max_ns > 0);
        }
        for (slot, &row) in rows.iter().enumerate() {
            assert_eq!(&out[slot * page..(slot + 1) * page], &payload[row as usize * page..(row as usize + 1) * page]);
        }
        table.gather_into(&rows, &mut out)?;
        let warm = table.stats().snapshot().since(&cold);
        assert_eq!((warm.page_hits, warm.page_misses), (3, 0));
        assert_eq!(warm.page_hit_bytes, (3 * page) as u64);
    }
    Ok(())
}

#[test]
fn mincore_routing_preserves_slots_duplicates_and_prefetch_accounting() -> TestResult {
    let mut file = tempfile::NamedTempFile::new_in(std::env::current_dir()?)?;
    let page = page_bytes()?;
    let payload: Vec<_> = (0..8 * page).map(|i| (i % 251) as u8).collect();
    file.write_all(&payload)?;
    file.as_file().sync_all()?;
    let open = || -> Result<MappedTable> {
        // SAFETY: this test owns the immutable file until all tables drop.
        let table = unsafe { MappedTable::single(file.path(), 0, 8, u8_rows(page))? };
        table.split_only_for_test();
        table.initialize_reads();
        table.parts[0].nowait.store(false, Ordering::Relaxed);
        table.stats.nowait.store(false, Ordering::Relaxed);
        Ok(table)
    };
    let table = open()?;
    if table.backend() != TableBackend::MincoreRouted {
        eprintln!("SKIP hybrid execution: kernel/seccomp disallows io_uring");
        return Ok(());
    }
    table.name_stats("mincore-routed-test");
    let named = mapped_table_stats().into_iter().find(|t| t.name == "mincore-routed-test").unwrap();
    assert_eq!((named.backend.as_str(), named.accounting), ("mincore-routed", "mincore-routed"));
    assert_eq!(uring::dontneed(&table.parts[0]), 0);
    let rows = [1, 3, 1, 6];
    assert_eq!(table.residency(&rows, false)?, [false; 4]);
    // Populate one resident source row; two duplicate slots route to memcpy.
    std::hint::black_box(table.row(1)?[0]);
    assert_eq!(table.residency(&rows, false)?, [true, false, true, false]);
    let slots = [3, 0, 1, 2];
    let mut output = vec![0; 4 * page];
    table.gather_slots(&rows, &slots, &mut output)?;
    let mixed = table.stats().snapshot();
    assert_eq!((mixed.page_hits, mixed.page_misses, mixed.resident_copy_rows), (2, 2, 2));
    assert_eq!((mixed.nowait_batches, mixed.unclassified_rows, mixed.late_major_faults), (0, 0, 0));
    assert_eq!(mixed.miss_histogram.iter().sum::<u64>(), 2);
    assert_eq!(mixed.miss_request_bytes, (2 * page) as u64);
    for (&row, &slot) in rows.iter().zip(&slots) {
        assert_eq!(&output[slot * page..(slot + 1) * page], &payload[row as usize * page..(row as usize + 1) * page]);
    }
    let expected = output.clone();
    assert!(table.gather_slots(&[1, 3], &[0, 0], &mut output).is_err());
    assert!(table.gather_slots(&[1, 8], &[0, 1], &mut output).is_err());
    assert_eq!(output, expected);
    let pool = GatherPool::new("hybrid-contract", 2)?;
    pool.gather(&table, &rows, &mut output)?;
    let warm = table.stats().snapshot().since(&mixed);
    assert_eq!((warm.page_hits, warm.page_misses, warm.resident_copy_rows), (4, 0, 4));
    assert_eq!((warm.miss_ns, warm.unclassified_rows), (0, 0));
    // Remove PTE references before attempting clean-file eviction for prefetch.
    drop(table);
    let table = open()?;
    assert_eq!(uring::dontneed(&table.parts[0]), 0);
    assert_eq!(table.residency(&rows, true)?, [false; 4]);
    assert_eq!(table.prefetch(&rows, 8)?, 3);
    let prefetch = table.stats().snapshot();
    assert_eq!((prefetch.page_hits, prefetch.page_misses, prefetch.prefetch_misses), (0, 0, 4));
    assert_eq!(prefetch.prefetch_miss_histogram.iter().sum::<u64>(), 4);
    table.prefetch(&rows, 8)?;
    let warm = table.stats().snapshot().since(&prefetch);
    assert_eq!((warm.prefetch_hits, warm.prefetch_misses, warm.prefetch_resident_copy_rows), (4, 0, 4));
    assert_eq!((warm.page_hits, warm.resident_copy_rows), (0, 0));
    Ok(())
}

#[test]
fn failed_misses_keep_initial_residency_and_copy_hits_only_once() -> TestResult {
    let mut file = tempfile::NamedTempFile::new_in(std::env::current_dir()?)?;
    let page = page_bytes()?;
    file.write_all(&vec![0x5a; 8 * page])?;
    file.as_file().sync_all()?;
    // SAFETY: test-owned immutable file outlives all owned reads and mappings.
    let table = unsafe { MappedTable::single(file.path(), 0, 8, u8_rows(page))? };
    table.split_only_for_test();
    assert_eq!(uring::dontneed(&table.parts[0]), 0);
    std::hint::black_box(table.row(1)?[0]);
    assert_eq!(table.residency(&[1, 3, 6], false)?, [true, false, false]);
    uring::fail_next_setup(libc::EPERM);
    let mut output = vec![0; 3 * page];
    table.gather_into(&[1, 3, 6], &mut output)?;
    assert!(output.iter().all(|&byte| byte == 0x5a));
    let stats = table.stats().snapshot();
    assert_eq!((stats.page_hits, stats.page_misses, stats.resident_copy_rows), (1, 2, 1));
    assert_eq!(stats.miss_ns, 0);
    Ok(())
}

#[test]
fn mincore_routing_keeps_hits_on_caller_and_falls_back_safely() -> TestResult {
    let (_dir, parts) = parted([3, 3, 2])?;
    // SAFETY: test-owned shard files remain immutable through the table lifetime.
    let table = unsafe { MappedTable::open(&parts, u8_rows(4))? };
    table.split_only_for_test();
    table.initialize_reads();
    if table.backend() == TableBackend::MincoreRouted {
        let mut output = [0; 12];
        table.gather_into(&[7, 2, 7], &mut output)?;
        let stats = table.stats().snapshot();
        assert_eq!(table.stats.accounting(), "mincore-routed");
        assert_eq!((stats.page_hits, stats.resident_copy_rows), (3, 3));
        assert!(stats.residency_ns > 0);
        assert_eq!(stats.nowait_batches, 0);
    }
    table.initialize_with_probe(|| Err(MappedTableError::Io {
        context: "injected unavailable io_uring".into(),
        source: std::io::Error::from_raw_os_error(libc::EPERM),
    }));
    assert_eq!(table.backend() == TableBackend::MincoreRouted, reactor::ready());
    let mut output = [0; 12];
    table.gather_into(&[7, 2, 7], &mut output)?;
    assert_eq!(output, [7, 7, 7, 7, 2, 2, 2, 2, 7, 7, 7, 7]);
    Ok(())
}

#[test]
#[ignore = "Read-only buffered file latency characterization; run release on NVMe"]
fn buffered_file_probe_measurement() -> TestResult {
    use std::os::unix::fs::FileExt;
    use std::hint::black_box;
    let mut fixture = tempfile::NamedTempFile::new_in(std::env::current_dir()?)?;
    fixture.write_all(&vec![0x5a; 1 << 20])?;
    fixture.as_file().sync_all()?;
    let path = std::env::var_os("CUTEAFD_TABLE_PROBE_FILE").map(PathBuf::from)
        .unwrap_or_else(|| fixture.path().into());
    let offset = std::env::var("CUTEAFD_TABLE_PROBE_OFFSET").unwrap_or_default().parse().unwrap_or(0);
    // SAFETY: fixture and selected sealed checkpoint remain immutable; this
    // measurement only reads and never drops the selected checkpoint's cache.
    let part = unsafe { MappedRows::open(&path, offset, 4096, 256)? };
    let nowait = uring::probe(&part)?;
    part.nowait.store(false, Ordering::Relaxed);
    eprintln!("file={} offset={offset} io_uring_nowait={nowait} fuse={} mode=plain-buffered", path.display(), part.fuse);
    let advice = uring::dontneed(&part);
    eprintln!("bounded_dontneed_result={advice} bytes=1048576 (does not prove backing eviction)");
    let mut scratch = Vec::new();
    for stage in ["after-dontneed-attempt", "warm"] {
        let mut hits = 0;
        let mut disagreement = 0;
        let mut tiers = [0u64; 7];
        for i in 0..100 {
            let row = (i * 37 % 4096) as u64;
            let pages: Vec<_> = part.pages(row)?.map(|page| (0, page)).collect();
            let mut flags = vec![false; pages.len()];
            uring::resident(&part, &pages, &mut flags, &mut scratch)?;
            let hit = flags.iter().all(|&hit| hit);
            hits += usize::from(hit);
            let read = uring::Read { part: &part, row, slot: 0 };
            let mut output = [0; 256];
            let started = Instant::now();
            uring::gather(&[read], &mut output, 256, &[Some(hit)])?;
            let elapsed = started.elapsed().as_nanos() as u64;
            tiers[uring::tier(elapsed)] += 1;
            disagreement += usize::from((!hit && elapsed < 10_000) || (hit && elapsed > 100_000));
        }
        eprintln!("stage={stage} rows=100 mincore_hits={hits} mincore_misses={} disagreement_rows={disagreement} all_read_latency_tiers={tiers:?}", 100 - hits);
    }
    for count in [1, 24, 384] {
        let reads: Vec<_> = (0..count).map(|slot| uring::Read { part: &part, row: (slot * 7919 % 4096) as u64, slot }).collect();
        let mut expected = vec![0; count * 256];
        for read in &reads {
            part.file.read_exact_at(&mut expected[read.slot * 256..(read.slot + 1) * 256], offset + read.row * 256)?;
        }
        let mut output = vec![0; expected.len()];
        let resident = vec![Some(true); count];
        for _ in 0..10 { uring::gather(&reads, &mut output, 256, &resident)?; }
        assert_eq!(output, expected);
        let repeats = 1000;
        let started = Instant::now();
        for _ in 0..repeats { black_box(uring::gather(&reads, &mut output, 256, &resident)?); }
        eprintln!("rows={count} repeats={repeats} warm_buffered_uring_batch_us={:.3}", started.elapsed().as_secs_f64() * 1e6 / repeats as f64);
        assert_eq!(output, expected);
    }
    Ok(())
}

fn u8_rows(width: usize) -> RowFormat {
    RowFormat { dtype: DType::U8, width, row_bytes: width }
}

#[test]
fn unaligned_payload_gather_preserves_order_and_rejects_bad_batches() -> TestResult {
    let mut shard = tempfile::NamedTempFile::new()?;
    shard.write_all(&[99; 13])?;
    shard.write_all(&[1, 2, 3, 4, 5, 6, 7, 8])?;
    let table = unsafe { MappedRows::open(shard.path(), 13, 4, 2)? };
    assert_eq!(table.prefetch(&[3, 1, 3], 1)?, 1);
    let mut out = [0; 6];
    table.gather_into(&[3, 1, 3], &mut out)?;
    assert_eq!(out, [7, 8, 3, 4, 7, 8]);
    assert!(table.gather_into(&[0, 4, 1], &mut out).is_err());
    assert_eq!(out, [7, 8, 3, 4, 7, 8]);
    assert!(table.prefetch(&[0], 0).is_err());
    assert_eq!(table.prefetch(&[], 0)?, 0);
    assert!(unsafe { MappedRows::open(shard.path(), 14, 4, 2) }.is_err());
    Ok(())
}

#[test]
fn sparse_checkpoint_rows_past_two_gib_use_wide_offsets() -> TestResult {
    let mut shard = tempfile::NamedTempFile::new()?;
    let row_bytes = 256;
    let high = (1_u64 << 31) / row_bytes as u64 + 7;
    shard.as_file().set_len((high + 1) * row_bytes as u64)?;
    shard.seek(SeekFrom::Start(high * row_bytes as u64))?;
    shard.write_all(&[0x5a; 256])?;
    let table = unsafe { MappedTable::single(shard.path(), 0, high + 1, u8_rows(row_bytes))? };
    assert_eq!(table.prefetch(&[high, high], 1)?, 1);
    let mut out = [0; 256];
    table.gather_into(&[high], &mut out)?;
    assert_eq!(out, [0x5a; 256]);
    assert!(table.prefetch(&[0, high], 1).is_err());
    Ok(())
}

/// Three parts (two shards, one with a header gap) of 4-byte rows; row r holds [r; 4].
fn parted(rows: [u64; 3]) -> std::result::Result<(tempfile::TempDir, Vec<TablePart>), Box<dyn std::error::Error>> {
    let dir = tempfile::tempdir()?;
    let mut parts = Vec::new();
    let mut next = 0u64;
    let mut a = std::fs::File::create(dir.path().join("a.safetensors"))?;
    let mut b = std::fs::File::create(dir.path().join("b.safetensors"))?;
    b.write_all(&[0xee; 5])?;
    let mut b_offset = 5u64;
    for (index, count) in rows.into_iter().enumerate() {
        let bytes: Vec<u8> = (next..next + count).flat_map(|r| [r as u8; 4]).collect();
        if index == 0 {
            a.write_all(&bytes)?;
            parts.push(TablePart { path: dir.path().join("a.safetensors"), offset: 0, rows: count });
        } else {
            b.write_all(&bytes)?;
            parts.push(TablePart { path: dir.path().join("b.safetensors"), offset: b_offset, rows: count });
            b_offset += bytes.len() as u64;
        }
        next += count;
    }
    Ok((dir, parts))
}

#[test]
fn parts_form_one_row_space_for_uniform_and_ragged_splits() -> TestResult {
    for split in [[3, 3, 2], [2, 5, 1]] {
        let (_dir, parts) = parted(split)?;
        let table = unsafe { MappedTable::open(&parts, u8_rows(4))? };
        assert_eq!(table.rows(), 8);
        assert_eq!(table.part_count(), 3);
        let rows = [7, 0, 3, 3, 5, 2];
        let mut out = vec![0; rows.len() * 4];
        table.gather_into(&rows, &mut out)?;
        let expected: Vec<u8> = rows.iter().flat_map(|&r| [r as u8; 4]).collect();
        assert_eq!(out, expected);
        assert!(matches!(table.gather_into(&[8], &mut [0; 4]), Err(MappedTableError::RowRange { row: 8, rows: 8 })));
        // Pages are per part: rows in three parts are at least three pages.
        assert!(table.prefetch(&[0, 3, 7], 16)? >= 2);
        assert!(table.prefetch(&[0, 7], 1).is_err());
        let pool = GatherPool::new("test-gather", 3)?;
        let mut pooled = vec![0; out.len()];
        let report = pool.gather(&table, &rows, &mut pooled)?;
        assert_eq!(pooled, expected);
        assert_eq!(report.rows, rows.len());
        assert!(pool.gather(&table, &[1, 9], &mut [0; 8]).is_err());
        let stats = table.stats().snapshot();
        assert_eq!((stats.gathers, stats.rows, stats.bytes), (1, 6, 24));
    }
    Ok(())
}

#[test]
fn checkpoint_tensors_validate_dtype_width_payload_and_paths() -> TestResult {
    let (dir, parts) = parted([3, 3, 2])?;
    let meta = |name: &str, part: &TablePart, dtype: DType, width: usize| SafetensorsTensorMetadata {
        name: name.into(),
        dtype,
        shape: vec![part.rows as usize, width],
        byte_offset: part.offset,
        byte_length: part.rows * 4,
    };
    let shard = |part: &TablePart| part.path.file_name().unwrap().to_str().unwrap().to_owned();
    let metas: Vec<_> = parts.iter().enumerate().map(|(i, p)| meta(&format!("t.shard_{i}"), p, DType::U8, 4)).collect();
    let shards: Vec<_> = parts.iter().map(shard).collect();
    let tensors: Vec<_> = shards.iter().map(String::as_str).zip(&metas).collect();
    let table = unsafe { MappedTable::from_tensors(dir.path(), &tensors)? };
    assert_eq!((table.rows(), table.row_bytes(), table.format().width), (8, 4, 4));
    let mut out = [0; 4];
    table.gather_into(&[6], &mut out)?;
    assert_eq!(out, [6; 4]);
    // BF16 rows of two elements are also 4 bytes, but a part may not change dtype.
    let mixed = meta("t.shard_1", &parts[1], DType::Bf16, 2);
    let tensors_mixed = vec![(shards[0].as_str(), &metas[0]), (shards[1].as_str(), &mixed)];
    assert!(unsafe { MappedTable::from_tensors(dir.path(), &tensors_mixed) }.is_err());
    let mut short = metas[0].clone();
    short.byte_length -= 1;
    assert!(unsafe { MappedTable::from_tensors(dir.path(), &[(shards[0].as_str(), &short)]) }.is_err());
    assert!(unsafe { MappedTable::from_tensors(dir.path(), &[("../a.safetensors", &metas[0])]) }.is_err());
    assert!(unsafe { MappedTable::from_tensors(dir.path(), &[]) }.is_err());
    Ok(())
}

#[test]
fn gather_worker_recycles_slots_and_cancels() -> TestResult {
    let (_dir, parts) = parted([3, 3, 2])?;
    let table = Arc::new(unsafe { MappedTable::open(&parts, u8_rows(4))? });
    let reader = Arc::clone(&table);
    let worker: GatherWorker<Vec<u8>, Vec<u64>, MappedTableError> =
        GatherWorker::new("test-worker", vec![vec![0; 8]], move |slot: &mut Vec<u8>, rows: &Vec<u64>| reader.gather_into(rows, slot))?;
    let wait = |ticket: &mut GatherTicket<Vec<u8>, Vec<u64>, MappedTableError>| loop {
        match ticket.poll() {
            Ok(GatherPoll::Pending) => std::thread::yield_now(),
            other => return other,
        }
    };
    let mut ticket = worker.try_submit(vec![7, 2], true)?.expect("one free slot");
    // The only slot is leased: backpressure, not blocking.
    assert!(worker.try_submit(vec![0, 0], false)?.is_none());
    let Ok(GatherPoll::Ready(lease)) = wait(&mut ticket) else { panic!("gather failed") };
    assert_eq!(lease.slot().unwrap(), &vec![7, 7, 7, 7, 2, 2, 2, 2]);
    assert_eq!(lease.job(), &vec![7, 2]);
    assert!(lease.timing().is_some());
    assert!(ticket.poll().is_err(), "a completion is consumed once");
    drop(lease);
    // The recycled slot is reused; a job error comes back as a job failure.
    let mut failing = loop {
        if let Some(ticket) = worker.try_submit(vec![9, 0], false)? {
            break ticket;
        }
    };
    assert!(matches!(wait(&mut failing), Err(GatherFailure::Job(MappedTableError::RowRange { .. }))));
    let mut cancelled = loop {
        if let Some(ticket) = worker.try_submit(vec![1, 1], false)? {
            break ticket;
        }
    };
    cancelled.cancel();
    assert!(matches!(cancelled.poll(), Ok(GatherPoll::Cancelled)));
    Ok(())
}

struct GatherTestWake(mpsc::SyncSender<()>);
impl std::task::Wake for GatherTestWake {
    fn wake(self: Arc<Self>) { let _ = self.0.try_send(()); }
    fn wake_by_ref(self: &Arc<Self>) { let _ = self.0.try_send(()); }
}

fn await_gather<F: std::future::Future>(future: F) -> F::Output {
    let (send, receive) = mpsc::sync_channel(1);
    let waker = std::task::Waker::from(Arc::new(GatherTestWake(send)));
    let mut context = std::task::Context::from_waker(&waker);
    let mut future = std::pin::pin!(future);
    loop {
        match future.as_mut().poll(&mut context) {
            std::task::Poll::Ready(result) => return result,
            std::task::Poll::Pending => receive.recv_timeout(Duration::from_secs(5)).expect("gather must wake its waiter"),
        }
    }
}

#[test]
fn gather_wait_wakes_for_success_failure_and_worker_panic() -> TestResult {
    let (started, entered) = mpsc::sync_channel(1);
    let (release, resume) = mpsc::sync_channel(1);
    let worker: GatherWorker<Vec<u8>, u8, &'static str> = GatherWorker::new(
        "test-gather-wake", vec![vec![0]], move |slot, job| {
            started.send(()).unwrap();
            resume.recv_timeout(Duration::from_secs(5)).unwrap();
            match job {
                0 => { slot[0] = 42; Ok(()) }
                1 => Err("injected gather failure"),
                _ => panic!("injected gather panic"),
            }
        })?;
    for job in 0..3 {
        let mut ticket = worker.try_submit(job, false)?.expect("free staging slot");
        entered.recv_timeout(Duration::from_secs(5))?;
        let (send, receive) = mpsc::sync_channel(1);
        let waker = std::task::Waker::from(Arc::new(GatherTestWake(send)));
        let mut context = std::task::Context::from_waker(&waker);
        let mut future = std::pin::pin!(ticket.wait());
        assert!(future.as_mut().poll(&mut context).is_pending());
        release.send(())?;
        receive.recv_timeout(Duration::from_secs(5)).expect("worker completion must signal the async waiter");
        let result = await_gather(future);
        match job {
            0 => {
                let Ok(GatherPoll::Ready(lease)) = result else { panic!("expected completed gather") };
                assert_eq!(lease.slot().unwrap(), &[42]);
            }
            1 => assert!(matches!(result, Err(GatherFailure::Job("injected gather failure")))),
            _ => assert!(matches!(result, Err(GatherFailure::Worker(MappedTableError::Stopped(_))))),
        }
    }
    Ok(())
}

#[test]
fn gather_wait_handles_early_completion_cancellation_and_dropped_future() -> TestResult {
    let (submitted, entered) = mpsc::sync_channel(1);
    let (release, resume) = mpsc::sync_channel(1);
    let worker: GatherWorker<(), (), ()> = GatherWorker::new("test-gather-cancel-wait", vec![()], move |_, _| {
        submitted.send(()).unwrap();
        resume.recv_timeout(Duration::from_secs(5)).unwrap();
        Ok(())
    })?;
    let mut ticket = worker.try_submit((), false)?.expect("free slot");
    entered.recv_timeout(Duration::from_secs(5))?;
    let (send, receive) = mpsc::sync_channel(1);
    let waker = std::task::Waker::from(Arc::new(GatherTestWake(send)));
    let mut context = std::task::Context::from_waker(&waker);
    {
        let mut future = std::pin::pin!(ticket.wait());
        assert!(future.as_mut().poll(&mut context).is_pending());
    }
    release.send(())?;
    receive.recv_timeout(Duration::from_secs(5))?;
    // Completion predates the next registration, and dropping the first future
    // must neither consume the lease nor lose the completed result.
    let Ok(GatherPoll::Ready(lease)) = await_gather(ticket.wait()) else { panic!("expected completed gather") };
    assert!(matches!(await_gather(ticket.wait()), Err(GatherFailure::Worker(_))));
    drop(lease);
    let mut cancelled = worker.try_submit((), false)?.expect("recycled slot");
    cancelled.cancel();
    assert!(matches!(await_gather(cancelled.wait()), Ok(GatherPoll::Cancelled)));
    // The worker may have passed its cancellation check before cancel was called.
    release.send(())?;
    Ok(())
}

#[test]
fn gather_admission_waits_for_release_and_preserves_unsubmitted_job() -> TestResult {
    let worker: GatherWorker<Vec<u8>, u8, ()> = GatherWorker::new(
        "test-gather-admission", vec![vec![0]], |slot, job| { slot[0] = *job; Ok(()) })?;
    let mut first = worker.try_submit(7, false)?.expect("one free slot");
    let Ok(GatherPoll::Ready(lease)) = await_gather(first.wait()) else { panic!("expected first gather") };
    let (send, receive) = mpsc::sync_channel(1);
    let waker = std::task::Waker::from(Arc::new(GatherTestWake(send)));
    let mut context = std::task::Context::from_waker(&waker);
    let mut capacity = std::pin::pin!(worker.wait_for_capacity());
    assert!(capacity.as_mut().poll(&mut context).is_pending());
    let mut admission = std::pin::pin!(worker.submit(19, false));
    assert!(admission.as_mut().poll(&mut context).is_pending());
    drop(lease);
    receive.recv_timeout(Duration::from_secs(5)).expect("lease release must wake admission");
    await_gather(capacity)?;
    let mut second = await_gather(admission)?;
    let Ok(GatherPoll::Ready(lease)) = await_gather(second.wait()) else { panic!("expected second gather") };
    assert_eq!(lease.job(), &19);
    assert_eq!(lease.slot().unwrap(), &[19]);
    drop(lease);
    await_gather(worker.wait_for_capacity())?;
    Ok(())
}

#[test]
fn gather_admission_prunes_dropped_waiters_and_replaces_repolled_wakers() -> TestResult {
    let worker: GatherWorker<(), (), ()> = GatherWorker::new(
        "test-gather-admission-drop", vec![()], |_, _| Ok(()))?;
    let mut ticket = worker.try_submit((), false)?.expect("one free slot");
    let Ok(GatherPoll::Ready(lease)) = await_gather(ticket.wait()) else { panic!("expected gather") };
    for _ in 0..1000 {
        let (send, _) = mpsc::sync_channel(1);
        let waker = std::task::Waker::from(Arc::new(GatherTestWake(send)));
        let mut context = std::task::Context::from_waker(&waker);
        let mut capacity = Box::pin(worker.wait_for_capacity());
        let mut admission = Box::pin(worker.submit((), false));
        assert!(capacity.as_mut().poll(&mut context).is_pending());
        assert!(admission.as_mut().poll(&mut context).is_pending());
        assert_eq!(worker.slots.admission.0.lock().unwrap().len(), 2);
        // A different task may poll the same future; replace its registration.
        let (send, _) = mpsc::sync_channel(1);
        let replacement = std::task::Waker::from(Arc::new(GatherTestWake(send)));
        let mut context = std::task::Context::from_waker(&replacement);
        assert!(capacity.as_mut().poll(&mut context).is_pending());
        assert!(admission.as_mut().poll(&mut context).is_pending());
        assert_eq!(worker.slots.admission.0.lock().unwrap().len(), 2);
        drop(capacity);
        assert_eq!(worker.slots.admission.0.lock().unwrap().len(), 1);
        drop(admission);
        assert!(worker.slots.admission.0.lock().unwrap().is_empty());
    }
    drop(lease);
    Ok(())
}

#[test]
fn spawned_gathers_complete_into_caller_storage() -> TestResult {
    let (_dir, parts) = parted([3, 3, 2])?;
    let table = Arc::new(unsafe { MappedTable::open(&parts, u8_rows(4))? });
    let pool = Arc::new(GatherPool::new("test-spawn", 2)?);
    let mut out = vec![0u8; 12];
    let pending = unsafe { pool.spawn_gather(table.clone(), vec![6, 0, 4], out.as_mut_ptr(), out.len()) };
    assert_eq!(pending.wait()?.rows, 3);
    assert_eq!(out, [6, 6, 6, 6, 0, 0, 0, 0, 4, 4, 4, 4]);
    // An invalid row fails without writing; dropping an unwaited gather waits for it.
    let failing = unsafe { pool.spawn_gather(table.clone(), vec![1, 8, 1], out.as_mut_ptr(), out.len()) };
    assert!(failing.wait().is_err());
    assert_eq!(out, [6, 6, 6, 6, 0, 0, 0, 0, 4, 4, 4, 4]);
    drop(unsafe { pool.spawn_gather(table.clone(), vec![2, 2, 2], out.as_mut_ptr(), out.len()) });
    assert_eq!(out, [2; 12]);
    Ok(())
}

#[test]
#[ignore = "CPU residency overhead measurement, run in release mode on NVMe"]
fn residency_overhead_measurement() -> TestResult {
    use std::hint::black_box;
    let number = |name: &str, default: usize| -> std::result::Result<usize, Box<dyn std::error::Error>> {
        Ok(std::env::var(name).ok().map(|value| value.parse()).transpose()?.unwrap_or(default))
    };
    let width = number("MAPPED_BENCH_WIDTH", 256)?;
    let count = number("MAPPED_BENCH_ROWS", 121_465)?;
    let repeats = number("MAPPED_BENCH_REPEATS", 10)?;
    let offset = number("MAPPED_BENCH_OFFSET", 0)?;
    assert!([256, 264].contains(&width) && count > 0 && repeats > 0);
    let fixture = if std::env::var_os("MAPPED_BENCH_PATH").is_none() {
        let file = tempfile::NamedTempFile::new_in(std::env::current_dir()?)?;
        file.as_file().set_len((1u64 << 26) * width as u64)?;
        file.as_file().sync_all()?;
        Some(file)
    } else { None };
    let path = std::env::var_os("MAPPED_BENCH_PATH").map(std::path::PathBuf::from)
        .unwrap_or_else(|| fixture.as_ref().unwrap().path().into());
    let length = std::fs::metadata(&path)?.len();
    let rows = number("MAPPED_BENCH_TABLE_ROWS", ((length - offset as u64) / width as u64) as usize)?;
    assert!(rows >= count && (rows as u64 * width as u64) <= length - offset as u64);
    // SAFETY: the fixture or immutable checkpoint remains live throughout every gather.
    let table = unsafe { MappedTable::open(&[TablePart { path: path.clone(), offset: offset as u64,
        rows: rows as u64 }], u8_rows(width))? };
    let ids: Vec<_> = (0..count).map(|i| (i as u64 * 7919) % rows as u64).collect();
    let slots: Vec<_> = (0..count).collect();
    let mut output = vec![0; count * width];
    // Populate only the selected pages, not a whole hundred-GB table.
    for (&row, slot) in ids.iter().zip(output.chunks_exact_mut(width)) {
        slot.copy_from_slice(table.parts[0].row(row)?);
    }
    table.gather_slots(&ids, &slots, &mut output)?;
    let before = table.stats().snapshot();
    let full = Instant::now();
    for _ in 0..repeats { table.gather_slots(black_box(&ids), &slots, black_box(&mut output))?; }
    let full = full.elapsed();
    let delta = table.stats().snapshot().since(&before);
    assert_eq!(delta.page_misses, 0, "warm benchmark must not measure table I/O");
    assert_eq!(delta.late_major_faults, 0, "resident pages faulted during copies");
    assert_eq!(delta.page_hits, (count * repeats) as u64);
    let copying = Instant::now();
    for _ in 0..repeats {
        for (&row, slot) in ids.iter().zip(output.chunks_exact_mut(width)) {
            slot.copy_from_slice(table.parts[0].row(row)?);
        }
        black_box(&output);
    }
    let copying = copying.elapsed();
    let accounting = Instant::now();
    for _ in 0..repeats { table.record_residency(black_box(&ids))?; }
    let accounting = accounting.elapsed();
    // Isolate the residency page collection/sort/dedup from mincore and lookup.
    let sorting = Instant::now();
    for _ in 0..repeats {
        let mut pages = Vec::with_capacity(count * 2);
        for &row in &ids { pages.extend(table.parts[0].pages(row)?); }
        pages.sort_unstable();
        pages.dedup();
        black_box(pages);
    }
    let sorting = sorting.elapsed();
    let validation = Instant::now();
    for _ in 0..repeats {
        let mut unique = BTreeSet::new();
        for &slot in &slots { black_box(unique.insert(slot)); }
        black_box(unique);
    }
    let validation = validation.elapsed();
    let us = |elapsed: Duration| elapsed.as_secs_f64() * 1e6 / repeats as f64;
    eprintln!("mapped_bench path={} rows={count} table_rows={rows} width={width} repeats={repeats} full_us={:.3} residency_us={:.3} accounting_us={:.3} sort_dedup_us={:.3} validation_tree_us={:.3} copy_us={:.3} hits={} misses={} late_faults={}",
        path.display(), us(full), delta.residency_ns as f64 / repeats as f64 / 1e3,
        us(accounting), us(sorting), us(validation), us(copying), delta.page_hits,
        delta.page_misses, delta.late_major_faults);
    Ok(())
}

#[test]
fn preloaded_rows_are_exact_without_random_io_or_page_accounting() -> TestResult {
    let (_dir, parts) = parted([3, 3, 2])?;
    // SAFETY: immutable test-owned files outlive both tables.
    let mut table = unsafe { MappedTable::open(&parts, u8_rows(4))? };
    let mut expected = [0; 12];
    table.gather_into(&[7, 2, 7], &mut expected)?;
    let before = table.stats.snapshot();
    MappedTable::preload_all(&mut [&mut table], 2)?;
    assert_eq!(table.backend(), TableBackend::Resident);
    let mut actual = [0; 12];
    table.gather_into(&[7, 2, 7], &mut actual)?;
    assert_eq!(actual, expected);
    table.gather_slots(&[7, 2, 7], &[2, 0, 1], &mut actual)?;
    assert_eq!(actual, [2, 2, 2, 2, 7, 7, 7, 7, 7, 7, 7, 7]);
    assert_eq!(table.row(7)?, &[7; 4]);
    assert_eq!(table.prefetch(&[7, 2, 7], 0)?, 0);
    assert!(table.prefetch(&[8], 0).is_err());
    assert!(table.gather_slots(&[0, 1], &[0, 0], &mut actual).is_err());
    assert_eq!(table.warm_bytes(), 0);
    assert_eq!(table.warm(4, &AtomicBool::new(false), |_| panic!("a resident table reads nothing"))?,
        WarmReport::default());
    assert_eq!(table.stats.snapshot(), before);
    Ok(())
}

#[test]
fn warm_reads_exactly_each_part_payload() -> TestResult {
    let (_dir, parts) = parted([3, 3, 2])?;
    // SAFETY: test-owned shards stay immutable while the table reads them;
    // the truncation below is only ever read through the descriptor.
    let table = unsafe { MappedTable::open(&parts, u8_rows(4))? };
    let chunks = Mutex::new(Vec::new());
    let before = table.stats.snapshot();
    let report = table.warm(3, &AtomicBool::new(false), |bytes| chunks.lock().unwrap().push(bytes))?;
    assert_eq!((report.bytes, table.warm_bytes()), (32, 32));
    let mut chunks = chunks.into_inner()?;
    chunks.sort_unstable();
    // One read per part payload, never the shard header ahead of the second part.
    assert_eq!(chunks, [8, 12, 12]);
    // Warm reads are not gathers: no table counters move.
    assert_eq!(table.stats.snapshot(), before);
    // The last part's read ends exactly at its payload end: one byte short fails.
    std::fs::OpenOptions::new().write(true).open(&parts[2].path)?.set_len(5 + 12 + 7)?;
    assert!(matches!(table.warm(3, &AtomicBool::new(false), |_| ()), Err(MappedTableError::Io { .. })));
    Ok(())
}

#[test]
fn warm_splits_parts_into_chunks_across_readers_and_stops_between_chunks() -> TestResult {
    let file = tempfile::NamedTempFile::new()?;
    let chunk = PAYLOAD_CHUNK_BYTES as u64;
    let row_bytes = 4096u64;
    // Sparse fixture: an unaligned 13-byte header, then two full chunks and one row.
    let rows = 2 * chunk / row_bytes + 1;
    file.as_file().set_len(13 + rows * row_bytes)?;
    // SAFETY: the sparse fixture stays immutable while the table lives.
    let table = unsafe { MappedTable::single(file.path(), 13, rows, u8_rows(row_bytes as usize))? };
    let chunks = Mutex::new(Vec::new());
    let report = table.warm(4, &AtomicBool::new(false), |bytes| chunks.lock().unwrap().push(bytes))?;
    assert_eq!(report.bytes, rows * row_bytes);
    let mut chunks = chunks.into_inner()?;
    chunks.sort_unstable();
    assert_eq!(chunks, [row_bytes, chunk, chunk]);
    // A stop set before the pass reads nothing; set during a chunk, the reader
    // finishes that chunk and takes no other.
    assert_eq!(table.warm(4, &AtomicBool::new(true), |_| panic!("read after stop"))?.bytes, 0);
    let stop = AtomicBool::new(false);
    assert_eq!(table.warm(1, &stop, |_| stop.store(true, Ordering::Release))?.bytes, chunk);
    Ok(())
}

#[test]
fn warm_populates_the_page_cache_without_the_mapping() -> TestResult {
    // Disk-backed (tmpfs cannot evict clean pages), as dontneed_produces_mincore_and_nowait_misses_then_hits.
    let mut file = tempfile::NamedTempFile::new_in(std::env::current_dir()?)?;
    let page = page_bytes()?;
    file.write_all(&vec![0x5a; 8 * page])?;
    file.as_file().sync_all()?;
    // SAFETY: the test file is immutable while the table lives.
    let table = unsafe { MappedTable::single(file.path(), 0, 8, u8_rows(page))? };
    let rows: Vec<u64> = (0..8).collect();
    assert_eq!(uring::dontneed(&table.parts[0]), 0);
    assert_eq!(table.residency(&rows, false)?, [false; 8]);
    assert_eq!(table.warm(2, &AtomicBool::new(false), |_| ())?.bytes, 8 * page as u64);
    assert_eq!(table.residency(&rows, false)?, [true; 8]);
    Ok(())
}

#[test]
fn preload_releases_the_page_cache_behind_its_reads() -> TestResult {
    // Disk-backed, as warm_populates_the_page_cache_without_the_mapping.
    let mut file = tempfile::NamedTempFile::new_in(std::env::current_dir()?)?;
    let page = page_bytes()?;
    file.write_all(&(0..8 * page).map(|i| (i / page) as u8).collect::<Vec<_>>())?;
    file.as_file().sync_all()?;
    // SAFETY: the test file is immutable while the table lives.
    let mut table = unsafe { MappedTable::single(file.path(), 0, 8, u8_rows(page))? };
    let rows: Vec<u64> = (0..8).collect();
    assert_eq!(table.warm(2, &AtomicBool::new(false), |_| ())?.bytes, 8 * page as u64);
    assert_eq!(table.residency(&rows, false)?, [true; 8]);
    // Two pages per chunk on three readers: the host keeps one copy of each row.
    MappedTable::preload_chunked(&mut [&mut table], 3, 2 * page)?;
    assert_eq!(table.residency(&rows, false)?, [false; 8]);
    for row in rows {
        assert_eq!(table.row(row)?, &vec![row as u8; page][..]);
    }
    Ok(())
}

#[test]
fn parallel_preload_matches_mapped_rows_across_parts_and_chunks() -> TestResult {
    let (_dir, parts) = parted([3, 3, 2])?;
    // SAFETY: immutable test-owned files outlive both tables.
    let mapped = unsafe { MappedTable::open(&parts, u8_rows(4))? };
    // SAFETY: as above.
    let mut table = unsafe { MappedTable::open(&parts, u8_rows(4))? };
    // 3-byte chunks of 4-byte rows on 4 readers: reads straddle rows and end at part ends.
    MappedTable::preload_chunked(&mut [&mut table], 4, 3)?;
    assert_eq!(table.backend(), TableBackend::Resident);
    let rows: Vec<u64> = (0..8).rev().collect();
    let (mut expected, mut actual) = (vec![0; 32], vec![0; 32]);
    mapped.gather_into(&rows, &mut expected)?;
    table.gather_into(&rows, &mut actual)?;
    assert_eq!(actual, expected);
    Ok(())
}

#[test]
fn preload_read_failure_keeps_every_table_mapped() -> TestResult {
    let (_dir, parts) = parted([3, 3, 2])?;
    // SAFETY: files remain immutable during all mapping access; the injected
    // read failure below never dereferences the truncated mapping.
    let mut first = unsafe { MappedTable::open(&parts[..1], u8_rows(4))? };
    let mut second = unsafe { MappedTable::open(&parts[1..], u8_rows(4))? };
    std::fs::OpenOptions::new().write(true).open(&parts[1].path)?.set_len(0)?;
    assert!(MappedTable::preload_all(&mut [&mut first, &mut second], 2).is_err());
    assert!(first.resident.is_none());
    assert!(second.resident.is_none());
    assert_ne!(first.backend(), TableBackend::Resident);
    assert_ne!(second.backend(), TableBackend::Resident);
    Ok(())
}

#[test]
fn fully_preloaded_tables_never_start_the_process_reactor() -> TestResult {
    const CHILD: &str = "CUTEAFD_TEST_PRELOADED_REACTOR_CHILD";
    if std::env::var_os(CHILD).is_none() {
        let status = std::process::Command::new(std::env::current_exe()?)
            .args(["--exact", "formats::mapped_table::tests::fully_preloaded_tables_never_start_the_process_reactor", "--nocapture"])
            .env(CHILD, "1").status()?;
        assert!(status.success());
        return Ok(());
    }
    assert!(!table_reactor_started());
    let (_dir, parts) = parted([3, 3, 2])?;
    // SAFETY: test-owned shards remain immutable for the table's lifetime.
    let mut table = unsafe { MappedTable::open(&parts, u8_rows(4))? };
    assert!(!table_reactor_started(), "opening is not final placement");
    MappedTable::preload_all(&mut [&mut table], 2)?;
    table.prepare_reads();
    let mut output = [0; 8];
    table.gather_into(&[0, 7], &mut output)?;
    assert!(!table_reactor_started(), "private gathers do not need a reactor");
    assert_eq!(table_reactor_workspace_bytes(0), 0);
    assert_eq!(table_reactor_workspace_bytes(8), 18 << 20);
    Ok(())
}

#[test]
fn split_gathers_keep_priority_selection_after_finisher_admission() -> TestResult {
    let (prepared_tx, prepared_rx) = mpsc::channel();
    let (finishing_tx, finishing_rx) = mpsc::channel();
    let gate = Arc::new((Mutex::new(false), std::sync::Condvar::new()));
    let blocked = Arc::clone(&gate);
    let worker = GatherWorker::split("split-priority", vec![0usize; 4],
        move |_: &mut usize, job: &usize| { prepared_tx.send(*job).unwrap(); Ok::<_, ()>(()) },
        move |_: &mut usize, job: &usize, ()| {
            if *job <= 2 {
                finishing_tx.send(*job).unwrap();
                let released = blocked.0.lock().unwrap();
                // Bound the gate even if an assertion unwinds before release.
                let _ = blocked.1.wait_timeout_while(released, Duration::from_secs(5), |done| !*done).unwrap();
            }
            Ok(())
        })?;
    let mut tickets = Vec::new();
    for job in 1..=2 {
        tickets.push(worker.try_submit(job, false)?.unwrap());
        assert_eq!(prepared_rx.recv_timeout(Duration::from_secs(5))?, job);
        assert_eq!(finishing_rx.recv_timeout(Duration::from_secs(5))?, job);
    }
    tickets.push(worker.try_submit(3, false)?.unwrap());
    assert!(prepared_rx.recv_timeout(Duration::from_millis(100)).is_err());
    // No finisher permit exists: normal work must remain in the shared queue,
    // rather than being hidden in the preparation owner's local variable.
    assert_eq!(worker.queue.0.lock().unwrap().normal.len(), 1);
    tickets.push(worker.try_submit_with(4, false, true)?.unwrap());
    *gate.0.lock().unwrap() = true;
    gate.1.notify_all();
    assert_eq!(prepared_rx.recv_timeout(Duration::from_secs(5))?, 4);
    assert_eq!(prepared_rx.recv_timeout(Duration::from_secs(5))?, 3);
    for ticket in &mut tickets {
        assert!(matches!(ticket.wait_timeout(Duration::from_secs(5)), Ok(GatherPoll::Ready(_))));
    }
    Ok(())
}

#[test]
fn split_gathers_prepare_second_before_first_finishes_and_bound_third() -> TestResult {
    let (prepared_tx, prepared_rx) = mpsc::channel();
    let (finish_tx, finish_rx) = mpsc::channel();
    let gate = Arc::new((Mutex::new(false), std::sync::Condvar::new()));
    let blocked = Arc::clone(&gate);
    let worker = GatherWorker::split("split-test", vec![0usize; 4],
        move |_: &mut usize, job: &usize| { prepared_tx.send(*job).unwrap(); Ok::<_, ()>(()) },
        move |slot: &mut usize, job: &usize, ()| {
            finish_tx.send(*job).unwrap();
            if *job == 1 {
                let (lock, ready) = &*blocked;
                let released = lock.lock().unwrap();
                let _ = ready.wait_timeout_while(released, Duration::from_secs(5), |done| !*done).unwrap();
            }
            *slot = *job;
            Ok(())
        })?;
    let mut first = worker.try_submit(1, false)?.unwrap();
    assert_eq!(prepared_rx.recv_timeout(Duration::from_secs(5))?, 1);
    assert_eq!(finish_rx.recv_timeout(Duration::from_secs(5))?, 1);
    let mut second = worker.try_submit(2, false)?.unwrap();
    assert_eq!(prepared_rx.recv_timeout(Duration::from_secs(5))?, 2);
    assert_eq!(finish_rx.recv_timeout(Duration::from_secs(5))?, 2);
    let GatherPoll::Ready(second) = second.wait_timeout(Duration::from_secs(5)).unwrap() else { panic!("second cancelled") };
    assert_eq!(second.slot(), Some(&2));
    assert!(matches!(first.poll(), Ok(GatherPoll::Pending)));
    let mut third = worker.try_submit(3, false)?.unwrap();
    assert_eq!(prepared_rx.recv_timeout(Duration::from_secs(5))?, 3);
    assert_eq!(finish_rx.recv_timeout(Duration::from_secs(5))?, 3);
    assert!(matches!(third.wait_timeout(Duration::from_secs(5)), Ok(GatherPoll::Ready(_))));
    *gate.0.lock().unwrap() = true;
    gate.1.notify_all();
    assert!(matches!(first.wait_timeout(Duration::from_secs(5)), Ok(GatherPoll::Ready(_))));
    Ok(())
}

#[test]
fn serial_and_split_preparer_panics_reject_queued_tickets_and_wake_waiters() -> TestResult {
    for split in [false, true] {
        let (started, entered) = mpsc::sync_channel(1);
        let (release, resume) = mpsc::sync_channel(1);
        let run = move |_: &mut (), _: &usize| -> std::result::Result<(), ()> {
            started.send(()).unwrap();
            resume.recv_timeout(Duration::from_secs(5)).unwrap();
            panic!("injected preparation panic");
        };
        let worker = if split { GatherWorker::split("split-panic", vec![(); 3], run, |_, _, ()| Ok(()))? }
            else { GatherWorker::new("serial-panic", vec![(); 3], run)? };
        let mut active = worker.try_submit(0, false)?.unwrap();
        entered.recv_timeout(Duration::from_secs(5))?;
        let mut normal = worker.try_submit(1, false)?.unwrap();
        let mut priority = worker.try_submit_with(2, false, true)?.unwrap();
        let (send, receive) = mpsc::sync_channel(1);
        let waker = std::task::Waker::from(Arc::new(GatherTestWake(send)));
        let mut context = std::task::Context::from_waker(&waker);
        let mut waiting = std::pin::pin!(priority.wait());
        assert!(waiting.as_mut().poll(&mut context).is_pending());
        release.send(())?;
        receive.recv_timeout(Duration::from_secs(5))?;
        assert!(matches!(await_gather(waiting), Err(GatherFailure::Worker(_))));
        for ticket in [&mut active, &mut normal] {
            assert!(matches!(ticket.wait_timeout(Duration::from_secs(5)), Err(GatherFailure::Worker(_))));
        }
        assert!(worker.try_submit(3, false).is_err());
        assert!(worker.queue.0.lock().unwrap().closed);
    }
    Ok(())
}

#[test]
fn split_finisher_panic_keeps_other_finisher_and_preparer_live() -> TestResult {
    let worker = GatherWorker::split("split-finish-panic", vec![0usize; 3],
        |_: &mut usize, _: &usize| Ok::<_, ()>(()),
        |slot: &mut usize, job: &usize, ()| {
            if *job == 0 { panic!("injected finisher panic"); }
            *slot = *job;
            Ok(())
        })?;
    let mut first = worker.try_submit(0, false)?.unwrap();
    assert!(matches!(first.wait_timeout(Duration::from_secs(5)), Err(GatherFailure::Worker(_))));
    for job in 1..=4 {
        let mut ticket = worker.try_submit(job, false)?.expect("recycled staging");
        let Ok(GatherPoll::Ready(lease)) = ticket.wait_timeout(Duration::from_secs(5)) else { panic!("surviving finisher stopped") };
        assert_eq!(lease.slot(), Some(&job));
    }
    Ok(())
}

#[test]
fn split_both_finisher_panics_reject_prepared_and_queued_jobs() -> TestResult {
    let (entered, finishing) = mpsc::channel();
    let gate = Arc::new((Mutex::new(false), std::sync::Condvar::new()));
    let blocked = Arc::clone(&gate);
    let (preparing, prepared) = mpsc::sync_channel(1);
    let (release, resume) = mpsc::sync_channel(1);
    let (dropped, drops) = mpsc::channel();
    struct Prepared(usize, mpsc::Sender<usize>);
    impl Drop for Prepared { fn drop(&mut self) { let _ = self.1.send(self.0); } }
    let mut worker = GatherWorker::split("split-both-panic", vec![(); 4],
        move |_: &mut (), job: &usize| {
            if *job == 2 {
                preparing.send(()).unwrap();
                resume.recv_timeout(Duration::from_secs(5)).unwrap();
            }
            Ok::<_, ()>(Prepared(*job, dropped.clone()))
        }, move |_, job: &usize, _: Prepared| {
            entered.send(*job).unwrap();
            let released = blocked.0.lock().unwrap();
            let (released, _) = blocked.1.wait_timeout_while(released,
                Duration::from_secs(5), |done| !*done).unwrap();
            drop(released);
            panic!("injected finisher panic");
        })?;
    let mut tickets = Vec::new();
    for job in 0..2 {
        tickets.push(worker.try_submit(job, false)?.unwrap());
        assert_eq!(finishing.recv_timeout(Duration::from_secs(5))?, job);
    }
    tickets.push(worker.try_submit(2, false)?.unwrap());
    tickets.push(worker.try_submit(3, false)?.unwrap());
    *gate.0.lock().unwrap() = true;
    gate.1.notify_all();
    prepared.recv_timeout(Duration::from_secs(5))?;
    // With both permits in finishers, no prepared channel job can coexist.
    // Join both dead receivers before allowing the next prepared send: it must
    // disconnect, recycle its payload and reject the still-queued fourth job.
    for finisher in worker.finishers.drain(..) { assert!(finisher.join().is_err()); }
    release.send(())?;
    for ticket in &mut tickets {
        assert!(matches!(ticket.wait_timeout(Duration::from_secs(5)), Err(GatherFailure::Worker(_))));
    }
    let slots = Arc::clone(&worker.slots);
    drop(worker);
    let mut recycled = (0..3).map(|_| drops.recv_timeout(Duration::from_secs(5)))
        .collect::<std::result::Result<Vec<_>, _>>()?;
    recycled.sort_unstable();
    assert_eq!(recycled, [0, 1, 2]);
    assert_eq!(slots.free.lock().unwrap().len(), 4);
    Ok(())
}

#[test]
fn split_cancel_and_shutdown_recycle_prepared_and_queued_jobs() -> TestResult {
    let (entered, started) = mpsc::channel();
    let (release, resume) = mpsc::sync_channel(1);
    let (dropped, drops) = mpsc::channel();
    struct Prepared(mpsc::Sender<()>);
    impl Drop for Prepared { fn drop(&mut self) { let _ = self.0.send(()); } }
    let worker = GatherWorker::split("split-shutdown", vec![(); 4],
        move |_: &mut (), job: &usize| {
            if *job == 1 { entered.send(()).unwrap(); resume.recv_timeout(Duration::from_secs(5)).unwrap(); }
            Ok::<_, ()>(Prepared(dropped.clone()))
        }, |_, _, _: Prepared| Ok(()))?;
    let mut first = worker.try_submit(1, false)?.unwrap();
    started.recv_timeout(Duration::from_secs(5))?;
    let mut queued = worker.try_submit(2, false)?.unwrap();
    first.cancel();
    assert!(matches!(first.poll(), Ok(GatherPoll::Cancelled)));
    let slots = Arc::clone(&worker.slots);
    let stopping = Arc::clone(&worker.shutdown);
    let (done, joined) = mpsc::sync_channel(1);
    let owner = std::thread::spawn(move || { drop(worker); done.send(()).unwrap(); });
    // Observe shutdown deterministically without an unbounded gate. Once set,
    // even the prepared job must drain/drop rather than reach its finisher.
    let deadline = Instant::now() + Duration::from_secs(5);
    while !stopping.load(Ordering::Acquire) && Instant::now() < deadline { std::thread::yield_now(); }
    assert!(stopping.load(Ordering::Acquire));
    release.send(())?;
    joined.recv_timeout(Duration::from_secs(5))?;
    owner.join().unwrap();
    drops.recv_timeout(Duration::from_secs(5))?;
    assert!(matches!(queued.wait_timeout(Duration::from_secs(5)), Ok(GatherPoll::Cancelled) | Err(GatherFailure::Worker(_))));
    assert_eq!(slots.free.lock().unwrap().len(), 4);
    Ok(())
}

#[test]
fn copy_first_copies_resident_batches_and_switches_cold_ones_to_the_split() -> TestResult {
    let mut file = tempfile::NamedTempFile::new_in(std::env::current_dir()?)?;
    let page = page_bytes()?;
    let rows = 256u64;
    let payload: Vec<u8> = (0..rows as usize * page).map(|i| (i % 251) as u8).collect();
    file.write_all(&payload)?;
    file.as_file().sync_all()?;
    // SAFETY: the test owns the immutable file for the table's lifetime.
    let table = unsafe { MappedTable::single(file.path(), 0, rows, u8_rows(page))? };
    table.prepare_reads();
    let ids: Vec<u64> = (0..rows).rev().collect();
    let expect = |output: &[u8]| for (index, &row) in ids.iter().enumerate() {
        assert_eq!(&output[index * page..(index + 1) * page], &payload[row as usize * page..(row as usize + 1) * page]);
    };
    // Warm: one plain copy, no residency split.
    let mut output = vec![0; ids.len() * page];
    table.gather_into(&ids, &mut output)?;
    let mut warm = vec![0; ids.len() * page];
    let before = table.stats().snapshot();
    table.gather_into(&ids, &mut warm)?;
    expect(&warm);
    let delta = table.stats().snapshot().since(&before);
    assert_eq!((delta.copy_switches, delta.page_misses), (0, 0));
    // Cold: the first fault switches the rest of the batch to the split; bytes stay exact.
    assert_eq!(uring::dontneed(&table.parts[0]), 0);
    // SAFETY: advice over exactly this live mapping.
    unsafe { libc::madvise(table.parts[0].base.as_ptr().cast(), table.parts[0].mapped_len, 21 /* MADV_PAGEOUT */) };
    let before = table.stats().snapshot();
    let mut cold = vec![0; ids.len() * page];
    table.gather_into(&ids, &mut cold)?;
    expect(&cold);
    let delta = table.stats().snapshot().since(&before);
    if delta.late_major_faults > 0 {
        assert_eq!(delta.copy_switches, 1);
        assert!(delta.page_hits <= COPY_CHECK_ROWS as u64, "at most one check interval copied before the switch");
        // Faulted copies count as device reads, not hits.
        assert!(delta.page_misses >= 1);
        // The next COLD_SKIPS - 1 gathers start with the split; then one probes the copy again.
        for _ in 1..COLD_SKIPS {
            let before = table.stats().snapshot();
            table.gather_into(&ids, &mut cold)?;
            expect(&cold);
            assert_eq!(table.stats().snapshot().since(&before).copy_switches, 0);
        }
        assert_eq!(table.cold_skips.load(Ordering::Relaxed), 0);
    }
    Ok(())
}
