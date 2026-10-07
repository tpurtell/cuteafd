use super::*;
use std::io::{Seek, SeekFrom, Write};

type TestResult = std::result::Result<(), Box<dyn std::error::Error>>;

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
    stats.uring.store(true, Ordering::Relaxed);
    let fresh = reader.snapshot();
    assert_eq!(fresh[0].backend, "uring");
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
    table.select_with_probe(TableBackend::Uring, || Err(MappedTableError::Io {
        context: "injected io_uring_setup seccomp failure".into(),
        source: std::io::Error::from_raw_os_error(libc::EPERM),
    }));
    assert_eq!(table.backend(), TableBackend::Mmap);
    let mut output = [0; 12];
    table.gather_into(&[7, 2, 7], &mut output)?;
    assert_eq!(output, [7, 7, 7, 7, 2, 2, 2, 2, 7, 7, 7, 7]);
    assert_eq!(table.stats().snapshot().page_hits + table.stats().snapshot().page_misses, 3);
    Ok(())
}

#[test]
fn runtime_ring_setup_failure_falls_back_for_demand_and_prefetch() -> TestResult {
    for errno in [libc::ENOMEM, libc::EPERM] {
        for backend in [TableBackend::Uring, TableBackend::MincoreRouted] {
            for prefetch in [false, true] {
                let (_dir, parts) = parted([3, 3, 2])?;
                // SAFETY: this test owns immutable shards through all gathers.
                let table = unsafe { MappedTable::open(&parts, u8_rows(4))? };
                for part in &table.parts { part.nowait.store(true, Ordering::Relaxed); }
                // Startup probe succeeded; the worker's first setup then fails.
                table.select_with_probe(backend, || Ok(()));
                uring::fail_next_setup(errno);
                let mut output = [0; 12];
                if prefetch { assert_eq!(table.prefetch(&[7, 2, 7], 8)?, 2); }
                else { table.gather_slots(&[7, 2, 7], &[2, 0, 1], &mut output)?; }
                assert_eq!(table.backend(), TableBackend::Mmap);
                table.gather_slots(&[7, 2, 7], &[2, 0, 1], &mut output)?;
                assert_eq!(output, [2, 2, 2, 2, 7, 7, 7, 7, 7, 7, 7, 7]);
                assert_eq!(table.stats().snapshot().unclassified_batches, 0);
                assert_eq!(table.stats().snapshot().prefetch_unclassified_batches, 0);
            }
        }
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
    table.select_backend(TableBackend::Uring);
    if table.backend() != TableBackend::Uring {
        eprintln!("SKIP unclassified wave execution: kernel/seccomp disallows io_uring");
        return Ok(());
    }
    table.parts[0].nowait.store(false, Ordering::Relaxed);
    assert_eq!(uring::MAX_BATCH, 4096);
    let n = uring::MAX_BATCH + 1;
    let reads: Vec<_> = (0..n).map(|i| uring::Read { part: &table.parts[0], row: (i % 3) as u64, slot: n - i - 1 }).collect();
    // Exercise mixed classified hits and unclassified reads in two waves.
    let resident: Vec<_> = (0..n).map(|i| if i % 2 == 0 { None } else { Some(true) }).collect();
    let mut output = vec![0; n * 4];
    let stats = uring::gather(&reads, &mut output, 4, &resident)?;
    assert_eq!((stats.unclassified_rows, stats.hits, stats.misses), (2049, 2048, 0));
    assert_eq!(stats.unclassified_bytes, 2049 * 4);
    assert_eq!(stats.unclassified_batches, 2);
    assert_eq!(stats.unclassified_batch_histogram.iter().sum::<u64>(), 2);
    assert!(stats.unclassified_batch_ns >= stats.unclassified_batch_max_ns && stats.unclassified_batch_max_ns > 0);
    assert_eq!(stats.miss_histogram, [0; 7]);
    for i in 0..n {
        assert_eq!(&output[(n-i-1)*4..(n-i)*4], &[(i % 3) as u8; 4]);
    }
    let empty = uring::gather(&[], &mut [], 4, &[])?;
    assert_eq!((empty.unclassified_rows, empty.unclassified_batches, empty.unclassified_batch_ns), (0, 0, 0));
    Ok(())
}

#[test]
fn buffered_uring_matches_mmap_across_unaligned_parts_and_duplicates() -> TestResult {
    let (_dir, parts) = parted([3, 3, 2])?;
    // SAFETY: the test owns immutable shards until both tables are dropped.
    let mmap = unsafe { MappedTable::open(&parts, u8_rows(4))? };
    mmap.select_backend(TableBackend::Mmap);
    // SAFETY: same immutable test files.
    let uring = unsafe { MappedTable::open(&parts, u8_rows(4))? };
    uring.select_backend(TableBackend::Uring);
    if uring.backend() != TableBackend::Uring {
        eprintln!("SKIP io_uring execution: kernel/seccomp/filesystem disallows buffered NOWAIT");
        return Ok(());
    }
    let rows = [7, 0, 3, 3, 5, 2];
    let pool = GatherPool::new("uring-parity", 3)?;
    let mut expected = [0; 24];
    let mut actual = [0; 24];
    pool.gather(&mmap, &rows, &mut expected, None)?;
    pool.gather(&uring, &rows, &mut actual, None)?;
    assert_eq!(actual, expected);
    assert_eq!(uring.backend(), TableBackend::Uring);
    assert_eq!(uring.stats().snapshot().page_hits, 6);
    assert_eq!(uring.prefetch(&rows, 16)?, 3);
    let prefetched = uring.stats().snapshot();
    assert_eq!(prefetched.prefetch_hits, 6);
    assert_eq!(prefetched.prefetch_nowait_batches, u64::from(uring.stats.nowait.load(Ordering::Relaxed)));
    // Invalid batches must not mutate the output or queue any I/O.
    assert!(uring.gather_slots(&[0, 8], &[0, 1], &mut actual).is_err());
    assert_eq!(actual, expected);
    assert!(uring.gather_slots(&[0, 1], &[0, 0], &mut actual).is_err());
    let cache = HotRowCache::new(4, 64).unwrap();
    pool.gather(&uring, &rows, &mut actual, Some(&cache))?;
    let before = uring.stats().snapshot();
    pool.gather(&uring, &rows, &mut actual, Some(&cache))?;
    let delta = uring.stats().snapshot().since(&before);
    assert_eq!(delta.page_hits + delta.page_misses, 0, "managed cache hits are not page-cache reads");
    assert_eq!(delta.cache_hits, 6);
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
    for backend in [TableBackend::Mmap, TableBackend::Uring] {
        // SAFETY: test file is immutable while the table lives.
        let table = unsafe { MappedTable::single(file.path(), 0, 8, u8_rows(page))? };
        table.select_backend(backend);
        if backend == TableBackend::Uring && table.backend() != backend {
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
        if backend == TableBackend::Uring {
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
fn unsupported_nowait_keeps_buffered_uring_and_mincore_accounting() -> TestResult {
    let mut file = tempfile::NamedTempFile::new_in(std::env::current_dir()?)?;
    let page = page_bytes()?;
    let payload: Vec<_> = (0..8 * page).map(|i| (i % 251) as u8).collect();
    file.write_all(&payload)?;
    file.as_file().sync_all()?;
    // SAFETY: test owns the immutable file until the table is dropped.
    let table = unsafe { MappedTable::single(file.path(), 0, 8, u8_rows(page))? };
    table.select_backend(TableBackend::Uring);
    if table.backend() != TableBackend::Uring {
        eprintln!("SKIP buffered io_uring: kernel/seccomp unavailable");
        return Ok(());
    }
    table.parts[0].nowait.store(false, Ordering::Relaxed);
    table.select_with_probe(TableBackend::Uring, || Ok(()));
    table.name_stats("uring-unsupported-nowait-test");
    let named = mapped_table_stats().into_iter().find(|t| t.name == "uring-unsupported-nowait-test").unwrap();
    assert_eq!((named.backend.as_str(), named.accounting), ("uring", "mincore"));
    // SAFETY: fadvise only evicts this clean, test-owned file's pages.
    assert_eq!(unsafe { libc::posix_fadvise(file.as_file().as_raw_fd(), 0,
        payload.len() as i64, libc::POSIX_FADV_DONTNEED) }, 0);
    let rows = [1, 3, 3, 6];
    let mut out = vec![0; rows.len() * page];
    table.gather_into(&rows, &mut out)?;
    let cold = table.stats().snapshot();
    assert_eq!(table.backend(), TableBackend::Uring);
    assert_eq!((cold.page_hits, cold.page_misses, cold.nowait_batches), (0, 4, 0));
    assert_eq!(cold.miss_histogram.iter().sum::<u64>(), 4);
    assert_eq!(cold.miss_request_bytes, (4 * page) as u64);
    assert!(cold.miss_ns > 0 && cold.residency_ns > 0);
    for (slot, &row) in rows.iter().enumerate() {
        assert_eq!(&out[slot * page..(slot + 1) * page], &payload[row as usize * page..(row as usize + 1) * page]);
    }
    table.gather_into(&rows, &mut out)?;
    let warm = table.stats().snapshot().since(&cold);
    assert_eq!((warm.page_hits, warm.page_misses, warm.miss_ns), (4, 0, 0));
    assert_eq!(warm.miss_histogram, [0; 7]);
    assert_eq!(table.prefetch(&rows, 8)?, 3);
    let prefetch = table.stats().snapshot().since(&cold);
    assert_eq!((prefetch.prefetch_hits, prefetch.prefetch_misses, prefetch.prefetch_nowait_batches), (4, 0, 0));
    assert_eq!(prefetch.page_hits, 4, "prefetch must not count demand hits");
    let before = table.stats().snapshot();
    table.stats.accounting_off.store(true, Ordering::Relaxed);
    table.gather_into(&rows, &mut out)?;
    let off = table.stats().snapshot().since(&before);
    assert_eq!((off.page_hits, off.page_misses, off.residency_ns), (0, 0, 0));
    assert_eq!((off.unclassified_rows, off.unclassified_bytes), (4, (4 * page) as u64));
    assert_eq!(off.unclassified_batch_histogram.iter().sum::<u64>(), 1);
    assert_eq!(off.unclassified_batches, 1);
    assert!(off.unclassified_batch_ns > 0);
    assert_eq!(off.unclassified_batch_ns, off.unclassified_batch_max_ns);
    assert_eq!(off.miss_histogram, [0; 7]);
    assert_eq!(table.prefetch(&rows, 8)?, 3);
    let off = table.stats().snapshot().since(&before);
    assert_eq!((off.unclassified_rows, off.prefetch_unclassified_rows), (4, 4));
    assert_eq!(off.prefetch_unclassified_batch_histogram.iter().sum::<u64>(), 1);
    assert_eq!((off.unclassified_batches, off.prefetch_unclassified_batches), (1, 1));
    assert!(off.prefetch_unclassified_batch_ns > 0);
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
        table.select_backend(TableBackend::MincoreRouted);
        table.parts[0].nowait.store(false, Ordering::Relaxed);
        table.stats.nowait.store(false, Ordering::Relaxed);
        // Routing still needs residency when ordinary uring accounting is off.
        table.stats.accounting_off.store(true, Ordering::Relaxed);
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
    pool.gather(&table, &rows, &mut output, None)?;
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
fn mincore_routing_keeps_nowait_exact_and_falls_back_safely() -> TestResult {
    let (_dir, parts) = parted([3, 3, 2])?;
    // SAFETY: test-owned shard files remain immutable through the table lifetime.
    let table = unsafe { MappedTable::open(&parts, u8_rows(4))? };
    table.select_backend(TableBackend::MincoreRouted);
    if table.backend() == TableBackend::MincoreRouted && table.stats.nowait.load(Ordering::Relaxed) {
        let mut output = [0; 12];
        table.gather_into(&[7, 2, 7], &mut output)?;
        let stats = table.stats().snapshot();
        assert_eq!(table.stats.accounting(), "nowait-exact");
        assert_eq!((stats.page_hits, stats.resident_copy_rows, stats.residency_ns), (3, 0, 0));
        assert!(stats.nowait_batches > 0);
    }
    table.select_with_probe(TableBackend::MincoreRouted, || Err(MappedTableError::Io {
        context: "injected unavailable io_uring".into(),
        source: std::io::Error::from_raw_os_error(libc::EPERM),
    }));
    assert_eq!(table.backend(), TableBackend::Mmap);
    let mut output = [0; 12];
    table.gather_into(&[7, 2, 7], &mut output)?;
    assert_eq!(output, [7, 7, 7, 7, 2, 2, 2, 2, 7, 7, 7, 7]);
    assert_eq!("mincore-routed".parse::<TableBackend>()?, TableBackend::MincoreRouted);
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
        let report = pool.gather(&table, &rows, &mut pooled, None)?;
        assert_eq!(pooled, expected);
        assert_eq!(report.rows, rows.len());
        assert!(pool.gather(&table, &[1, 9], &mut [0; 8], None).is_err());
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
fn hot_row_cache_serves_hits_and_evicts_least_recent() -> TestResult {
    let (_dir, parts) = parted([3, 3, 2])?;
    let table = unsafe { MappedTable::open(&parts, u8_rows(4))? };
    let cache = HotRowCache::new(4, 8).expect("two rows");
    assert!(HotRowCache::new(4, 3).is_none());
    let pool = GatherPool::new("test-cache", 2)?;
    let gather = |rows: &[u64]| -> std::result::Result<(Vec<u8>, GatherReport), MappedTableError> {
        let mut out = vec![0; rows.len() * 4];
        let report = pool.gather(&table, rows, &mut out, Some(&cache))?;
        Ok((out, report))
    };
    let (out, report) = gather(&[1, 2])?;
    assert_eq!((out, report.cache_hits), (vec![1, 1, 1, 1, 2, 2, 2, 2], 0));
    let (out, report) = gather(&[2, 1, 2])?;
    assert_eq!((out, report.cache_hits), (vec![2, 2, 2, 2, 1, 1, 1, 1, 2, 2, 2, 2], 3));
    // Row 1 is least recent after [2, 1, 2]; inserting 5 evicts it.
    gather(&[5])?;
    assert_eq!(cache.len(), 2);
    let (out, report) = gather(&[1, 5])?;
    assert_eq!((out, report.cache_hits), (vec![1, 1, 1, 1, 5, 5, 5, 5], 1));
    let stats = table.stats().snapshot();
    assert_eq!((stats.cache_hits, stats.cache_misses), (4, 4));
    assert!(pool.gather(&table, &[0], &mut [0; 4], HotRowCache::new(8, 64).as_ref()).is_err());
    Ok(())
}

#[test]
fn prefetcher_bounds_rows_reports_pages_and_detaches() -> TestResult {
    let (_dir, parts) = parted([3, 3, 2])?;
    let table: Arc<MappedTable> = Arc::new(unsafe { MappedTable::open(&parts, u8_rows(4))? });
    let worker = TablePrefetcher::new("test-prefetch", 1, 4, 8)?;
    assert!(worker.try_submit(table.clone(), &[8]).is_err());
    assert!(worker.try_submit(table.clone(), &[0; 5]).is_err());
    let ticket = worker.try_submit(table.clone(), &[1, 0, 1])?.expect("queue has room");
    assert_eq!(ticket.wait()?, TablePrefetchOutcome::Advised(vec![1]));
    let cancelled = worker.try_submit(table.clone(), &[7])?.expect("queue has room");
    cancelled.cancel();
    assert!(matches!(cancelled.wait()?, TablePrefetchOutcome::Cancelled | TablePrefetchOutcome::Advised(_)));
    // Detached advice truncates to the row cap instead of failing.
    while !worker.submit_detached(table.clone(), &[0, 1, 2, 3, 4, 5])? {}
    drop(worker);
    let stats = table.stats().snapshot();
    assert!(stats.prefetch_jobs >= 2 && stats.prefetch_pages >= 2);
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

#[test]
fn spawned_gathers_complete_into_caller_storage() -> TestResult {
    let (_dir, parts) = parted([3, 3, 2])?;
    let table = Arc::new(unsafe { MappedTable::open(&parts, u8_rows(4))? });
    let pool = Arc::new(GatherPool::new("test-spawn", 2)?);
    let mut out = vec![0u8; 12];
    let pending = unsafe { pool.spawn_gather(table.clone(), vec![6, 0, 4], out.as_mut_ptr(), out.len(), None) };
    assert_eq!(pending.wait()?.rows, 3);
    assert_eq!(out, [6, 6, 6, 6, 0, 0, 0, 0, 4, 4, 4, 4]);
    // An invalid row fails without writing; dropping an unwaited gather waits for it.
    let failing = unsafe { pool.spawn_gather(table.clone(), vec![1, 8, 1], out.as_mut_ptr(), out.len(), None) };
    assert!(failing.wait().is_err());
    assert_eq!(out, [6, 6, 6, 6, 0, 0, 0, 0, 4, 4, 4, 4]);
    let cache = Arc::new(HotRowCache::new(4, 64).expect("rows"));
    drop(unsafe { pool.spawn_gather(table.clone(), vec![2, 2, 2], out.as_mut_ptr(), out.len(), Some(cache.clone())) });
    assert_eq!(out, [2; 12]);
    assert_eq!(cache.len(), 1);
    Ok(())
}

#[test]
fn warm_reads_every_part_and_stops_on_request() -> TestResult {
    let (_dir, parts) = parted([3, 3, 2])?;
    let table = unsafe { MappedTable::open(&parts, u8_rows(4))? };
    // Each part's mapping starts at its page; part 2 shares b's page with part 1.
    let total = table.warm(&AtomicBool::new(false), None)?;
    assert!(total >= 8 * 4);
    assert_eq!(table.warm(&AtomicBool::new(true), None)?, 0);
    // Pacing: the whole table at twice its size per second takes about half a second.
    let started = Instant::now();
    assert_eq!(table.warm(&AtomicBool::new(false), Some(2 * total))?, total);
    assert!(started.elapsed() >= Duration::from_millis(400));
    let mut out = [0; 4];
    table.gather_into(&[7], &mut out)?;
    assert_eq!(out, [7; 4]);
    Ok(())
}

#[test]
#[ignore = "Warm/cold backend latency distributions; release-mode CPU characterization"]
fn mapped_backend_latency_measurement() -> TestResult {
    let file = tempfile::NamedTempFile::new_in(std::env::current_dir()?)?;
    file.as_file().set_len(16u64 << 30)?;
    let path = std::env::var_os("CUTEAFD_TABLE_PROBE_FILE").map(PathBuf::from)
        .unwrap_or_else(|| file.path().into());
    let offset: u64 = std::env::var("CUTEAFD_TABLE_PROBE_OFFSET").unwrap_or_default().parse().unwrap_or(0);
    let rows = ((std::fs::metadata(&path)?.len() - offset) / 256).min(1 << 26);
    let table = |backend, off| -> Result<MappedTable> {
        // SAFETY: fixture and selected sealed checkpoint stay immutable.
        let table = unsafe { MappedTable::single(&path, offset, rows, u8_rows(256))? };
        table.select_backend(backend);
        if backend != TableBackend::Mmap && table.backend() != backend { return Err(invalid("io_uring unavailable in latency measurement")); }
        // Compare the same ordinary buffered reads on every filesystem; the
        // separate capability probe reports whether native NOWAIT is supported.
        table.parts[0].nowait.store(false, Ordering::Relaxed);
        table.stats.nowait.store(false, Ordering::Relaxed);
        table.stats.accounting_off.store(off, Ordering::Relaxed);
        Ok(table)
    };
    let mmap = table(TableBackend::Mmap, false)?;
    let off = table(TableBackend::Uring, true)?;
    let full = table(TableBackend::Uring, false)?;
    let hybrid = table(TableBackend::MincoreRouted, false)?;
    let report = |phase: &str, mode: &str, count: usize, samples: &mut Vec<f64>| {
        samples.sort_by(f64::total_cmp);
        eprintln!("phase={phase} mode={mode} rows={count} samples={} median_us={:.3} p99_us={:.3}", samples.len(), samples[samples.len()/2], samples[(samples.len()*99/100).min(samples.len()-1)]);
    };
    let ids = |count| {
        let mut state = 0x1234_5678u64;
        (0..count).map(|_| { state = state.wrapping_mul(6364136223846793005).wrapping_add(1); state % rows }).collect::<Vec<_>>()
    };
    eprintln!("file={} offset={offset} rows_in_space={rows} width=256", path.display());
    for count in [1, 24, 384, 1536, 3072] {
        let ids = ids(count);
        let mut output = vec![0; count * 256];
        let mut expected = vec![0; output.len()];
        mmap.gather_into(&ids, &mut expected)?;
        for (mode, table) in [("mmap+mincore", &mmap), ("uring-off", &off), ("uring+mincore", &full), ("mincore-routed", &hybrid)] {
            for _ in 0..10 { table.gather_into(&ids, &mut output)?; }
            let mut samples = Vec::with_capacity(100);
            for _ in 0..100 {
                let started = Instant::now();
                table.gather_into(&ids, &mut output)?;
                samples.push(started.elapsed().as_secs_f64() * 1e6);
            }
            assert_eq!(output, expected);
            report("warm", mode, count, &mut samples);
        }
    }
    drop((mmap, off, full, hybrid));
    if std::env::var("CUTEAFD_TABLE_PROBE_COLD").ok().as_deref() == Some("1") {
        for count in [1, 24, 384, 1536, 3072] {
            let ids = ids(count);
            for backend in [TableBackend::Mmap, TableBackend::Uring] {
                let mut samples = Vec::with_capacity(3);
                let mut misses = 0;
                let mut tiers = [0u64; 7];
                for _ in 0..3 {
                    let table = table(backend, false)?;
                    if uring::dontneed(&table.parts[0]) != 0 { return Err("bounded DONTNEED unavailable".into()); }
                    let mut output = vec![0; count * 256];
                    let started = Instant::now();
                    table.gather_into(&ids, &mut output)?;
                    samples.push(started.elapsed().as_secs_f64() * 1e6);
                    let stats = table.stats().snapshot();
                    misses += stats.page_misses;
                    for (total, value) in tiers.iter_mut().zip(stats.miss_histogram) { *total += value; }
                }
                report("after-dontneed-attempt", &backend.to_string(), count, &mut samples);
                eprintln!("cold_observed mode={backend} rows={count} misses={misses} requested={} miss_tiers={tiers:?}", 3 * count);
            }
        }
    }
    Ok(())
}

#[test]
#[ignore = "Fresh nonresident-file witness; explicit sealed PLE file required"]
fn mapped_backend_fresh_cold_measurement() -> TestResult {
    use std::os::unix::fs::FileExt;
    let path = PathBuf::from(std::env::var_os("CUTEAFD_TABLE_PROBE_FILE").ok_or("an explicit sealed PLE file is required")?);
    let offset: u64 = std::env::var("CUTEAFD_TABLE_PROBE_OFFSET")?.parse()?;
    let rows = ((std::fs::metadata(&path)?.len() - offset) / 256).min(1 << 26);
    let open = |backend| -> Result<MappedTable> {
        // SAFETY: the selected sealed checkpoint remains immutable.
        let table = unsafe { MappedTable::single(&path, offset, rows, u8_rows(256))? };
        table.select_backend(backend);
        if table.backend() != backend { return Err(invalid("cold witness backend unavailable")); }
        // Ordinary buffered reads compare the same unsupported-NOWAIT route;
        // native NOWAIT capability is characterized by the separate probe.
        table.parts[0].nowait.store(false, Ordering::Relaxed);
        table.stats.nowait.store(false, Ordering::Relaxed);
        table.stats.accounting_off.store(false, Ordering::Relaxed);
        Ok(table)
    };
    let tables = [open(TableBackend::Mmap)?, open(TableBackend::Uring)?, open(TableBackend::MincoreRouted)?];
    let part = &tables[0].parts[0];
    let page_count = part.mapped_len / part.page_bytes;
    let mut state = 0x98ab_cdef_8765_4321u64;
    let mut seen = BTreeSet::new();
    let mut scanned = 0usize;
    eprintln!("fresh_cold file={} offset={offset} width=256 page_bytes={} cache_advice=none", path.display(), part.page_bytes);
    for count in [384, 1536, 3072] {
        let mut samples = [Vec::new(), Vec::new(), Vec::new()];
        for repetition in 0..3 {
            for step in 0..3 {
                let index = (repetition + step) % 3;
                let table = &tables[index];
                let mut selected = Vec::with_capacity(count);
                while selected.len() < count {
                    if scanned >= 262_144 {
                        eprintln!("UNAVAILABLE fresh_nonresident rows={count} mode={} scanned={scanned} found={}", table.backend(), selected.len());
                        return Err("bounded search could not find enough fresh nonresident pages".into());
                    }
                    let mut candidates = Vec::with_capacity(4096);
                    for _ in 0..4096 {
                        scanned += 1;
                        state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                        let page = (state as usize % page_count).max(1);
                        let row = (page * part.page_bytes - part.data_offset).div_ceil(256) as u64;
                        if row < rows && seen.insert(page) { candidates.push(row); }
                    }
                    let flags = table.residency(&candidates, false)?;
                    for (row, hit) in candidates.into_iter().zip(flags) {
                        if !hit && selected.len() < count { selected.push(row); }
                    }
                    // The snapshot is the cold witness; neither successful
                    // DONTNEED nor a first-time PTE fault establishes coldness.
                    let flags = table.residency(&selected, false)?;
                    selected = selected.into_iter().zip(flags).filter_map(|(row, hit)| (!hit).then_some(row)).collect();
                }
                let before = table.stats().snapshot();
                let mut output = vec![0; count * 256];
                let started = Instant::now();
                table.gather_into(&selected, &mut output)?;
                let elapsed = started.elapsed().as_secs_f64() * 1e6;
                samples[index].push(elapsed);
                let stats = table.stats().snapshot().since(&before);
                eprintln!("fresh_sample mode={} rows={count} repetition={repetition} confirmed_nonresident={count} observed_misses={} resident_copy_rows={} elapsed_us={elapsed:.3} miss_tiers={:?}", table.backend(), stats.page_misses, stats.resident_copy_rows, stats.miss_histogram);
                assert_eq!(stats.page_misses, count as u64, "cold pages changed before the backend's snapshot");
                assert_eq!(stats.resident_copy_rows, 0);
                if table.backend() != TableBackend::Mmap { assert_eq!(stats.miss_histogram.iter().sum::<u64>(), count as u64); }
                // Check bytes after timing; positioned reads cannot warm the witness.
                let mut expected = vec![0; output.len()];
                for (&row, slot) in selected.iter().zip(expected.chunks_exact_mut(256)) {
                    part.file.read_exact_at(slot, offset + row * 256)?;
                }
                assert_eq!(output, expected);
            }
        }
        for (index, sample) in samples.iter_mut().enumerate() {
            sample.sort_by(f64::total_cmp);
            eprintln!("fresh_summary mode={} rows={count} samples=3 median_us={:.3} max_us={:.3}", tables[index].backend(), sample[1], sample[2]);
        }
    }
    Ok(())
}

#[test]
#[ignore = "CPU residency overhead measurement, run in release mode on NVMe"]
fn residency_overhead_measurement() -> TestResult {
    use std::hint::black_box;
    let width = 256;
    let rows = 1usize << 26;
    let file = tempfile::NamedTempFile::new_in(std::env::current_dir()?)?;
    // Sparse 16 GiB address space: random decode rows rarely share pages.
    file.as_file().set_len((rows * width) as u64)?;
    file.as_file().sync_all()?;
    // SAFETY: immutable fixture survives every gather and mapping.
    let table = unsafe { MappedTable::open(&[TablePart { path: file.path().into(), offset: 0, rows: rows as u64 }], u8_rows(width))? };
    table.select_backend(TableBackend::Mmap);
    for count in [24, 384, 1536, 3072] {
        let ids: Vec<_> = (0..count).map(|i| (i as u64 * 7919) % rows as u64).collect();
        let mut output = vec![0; count * width];
        table.gather_into(&ids, &mut output)?;
        let repeats = 500;
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
        eprintln!("rows={count} repeats={repeats} copy_us={:.3} mincore_accounting_us={:.3}",
            copying.as_secs_f64() * 1e6 / repeats as f64,
            accounting.as_secs_f64() * 1e6 / repeats as f64);
    }
    Ok(())
}
