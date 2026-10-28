//! Opt-in background page-cache warm of mapped tables, held by whichever
//! owner maps them ([`TableWarm::start`]: V4.1 Engram; Qwen PLE tables are the
//! same [`MappedTable`]).
//!
//! Every prompt token reads random table rows; from a cold page cache each
//! costs a storage fault, which a store with slow random reads turns into the
//! prefill bottleneck. After the server is ready, one thread reads each
//! table's payload once with [`MappedTable::warm`] so later rows hit the page
//! cache. Advisory only: readiness never waits for it, the pages stay
//! reclaimable, and it is skipped unless available host memory holds the
//! tables plus a margin. Dropping the handle stops it between chunks.
use super::mapped_table::{MappedTable, MappedTableError, Result};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

/// Available host memory a warm leaves beyond the bytes it reads.
pub const WARM_MARGIN_BYTES: u64 = 16 << 30;
/// How often a warm waiting for readiness checks it.
pub const READY_POLL: Duration = Duration::from_secs(1);

/// How a warm ended; every outcome but a stop before readiness is logged.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum WarmOutcome {
    /// Every table is resident: nothing to read.
    Resident,
    /// Available host memory (None: unreadable) is below the tables plus the margin.
    Skipped { available: Option<u64>, required: u64 },
    /// Stopped before readiness or between chunks.
    Stopped { bytes: u64 },
    Warmed { bytes: u64, elapsed: Duration },
}

/// Wait until `ready` (checked every `poll`; an unpark wakes it early), admit
/// host memory (`available`), then warm `tables` in order on `readers`
/// threads, logging the start, every 10% of the bytes and the end.
pub(crate) fn warm_when_ready(name: &str, tables: &[&MappedTable], readers: usize, stop: &AtomicBool,
    ready: impl Fn() -> bool, available: impl FnOnce() -> anyhow::Result<u64>, poll: Duration)
    -> Result<WarmOutcome> {
    let total: u64 = tables.iter().map(|table| table.warm_bytes()).sum();
    if total == 0 {
        tracing::info!(warm = name, "mapped table warm skipped: every table is resident");
        return Ok(WarmOutcome::Resident);
    }
    while !ready() {
        if stop.load(Ordering::Acquire) {
            return Ok(WarmOutcome::Stopped { bytes: 0 });
        }
        thread::park_timeout(poll);
    }
    let required = total.saturating_add(WARM_MARGIN_BYTES);
    let available = match available() {
        Ok(bytes) if bytes >= required => bytes,
        Ok(bytes) => {
            tracing::info!(warm = name, table_bytes = total, available_bytes = bytes, margin_bytes = WARM_MARGIN_BYTES,
                "mapped table warm skipped: available host memory is below the tables plus the margin");
            return Ok(WarmOutcome::Skipped { available: Some(bytes), required });
        }
        Err(error) => {
            tracing::info!(warm = name, %error, "mapped table warm skipped: available host memory is unknown");
            return Ok(WarmOutcome::Skipped { available: None, required });
        }
    };
    tracing::info!(warm = name, table_bytes = total, tables = tables.len(), readers, available_bytes = available,
        "mapped table warm started");
    let started = Instant::now();
    let done = AtomicU64::new(0);
    let progress = |chunk: u64| {
        let after = done.fetch_add(chunk, Ordering::Relaxed) + chunk;
        if let Some(percent) = crossed_decile(after - chunk, after, total) {
            tracing::info!(warm = name, percent, bytes = after, elapsed_s = started.elapsed().as_secs_f64(),
                "mapped table warm progress");
        }
    };
    let mut bytes = 0;
    for table in tables {
        let report = table.warm(readers, stop, &progress)?;
        bytes += report.bytes;
        // A pass that returns early without an error was stopped.
        if report.bytes < table.warm_bytes() {
            tracing::info!(warm = name, bytes, elapsed_s = started.elapsed().as_secs_f64(), "mapped table warm stopped");
            return Ok(WarmOutcome::Stopped { bytes });
        }
    }
    let elapsed = started.elapsed();
    tracing::info!(warm = name, bytes, seconds = elapsed.as_secs_f64(),
        gb_per_s = bytes as f64 / elapsed.as_secs_f64().max(1e-9) / 1e9, "mapped table warm complete");
    Ok(WarmOutcome::Warmed { bytes, elapsed })
}

/// The 10% mark (10..=90) that reading `before..after` of `total` bytes
/// passed, if any; the end is left to the completion line.
fn crossed_decile(before: u64, after: u64, total: u64) -> Option<u64> {
    let tenths = |bytes: u64| bytes.saturating_mul(10) / total.max(1);
    let reached = tenths(after);
    (reached > tenths(before) && reached < 10).then_some(reached * 10)
}

/// A warm on its own thread. Dropping it sets the stop flag, wakes a
/// readiness wait and joins; readers finish at most their current chunk.
pub struct TableWarm {
    stop: Arc<AtomicBool>,
    thread: Option<JoinHandle<()>>,
}

impl TableWarm {
    /// Warm the tables `owner` holds, `tables` listing them in read order, on
    /// a `{name}-warm` thread: once `ready` holds (checked every
    /// [`READY_POLL`]) and available host memory admits them plus
    /// [`WARM_MARGIN_BYTES`], `readers` threads read each payload once. The
    /// thread owns `owner`, so the tables outlive every read, and the table
    /// owner keeps the handle; a failure is logged and rows keep loading on
    /// demand.
    pub fn start<T: Send + 'static>(name: &'static str, owner: T, tables: fn(&T) -> Vec<&MappedTable>,
        readers: usize, ready: impl Fn() -> bool + Send + 'static) -> Result<Self> {
        let bytes: u64 = tables(&owner).iter().map(|table| table.warm_bytes()).sum();
        tracing::info!(warm = name, readers, table_bytes = bytes, "mapped table warm armed; starts after readiness");
        Self::spawn(&format!("{name}-warm"), move |stop| {
            let available = || super::mapped_preload::host_memory().map(|(_, available)| available);
            if let Err(error) = warm_when_ready(name, &tables(&owner), readers, stop, ready, available, READY_POLL) {
                tracing::warn!(warm = name, %error, "mapped table warm failed; rows keep loading on demand");
            }
        })
    }

    fn spawn(name: &str, body: impl FnOnce(&AtomicBool) + Send + 'static) -> Result<Self> {
        let stop = Arc::new(AtomicBool::new(false));
        let flag = Arc::clone(&stop);
        let thread = thread::Builder::new().name(name.into()).spawn(move || body(&flag))
            .map_err(|source| MappedTableError::Io { context: format!("starting {name}"), source })?;
        Ok(Self { stop, thread: Some(thread) })
    }
}

impl Drop for TableWarm {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Release);
        if let Some(thread) = self.thread.take() {
            thread.thread().unpark();
            if thread.join().is_err() {
                tracing::error!("mapped table warm panicked");
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::formats::mapped_table::RowFormat;
    use std::sync::atomic::AtomicUsize;

    type TestResult = std::result::Result<(), Box<dyn std::error::Error>>;

    /// A table of `rows` 4 KiB rows over a sparse file.
    fn table(rows: u64) -> std::result::Result<(tempfile::NamedTempFile, MappedTable), Box<dyn std::error::Error>> {
        let file = tempfile::NamedTempFile::new()?;
        file.as_file().set_len(rows * 4096)?;
        let format = RowFormat { dtype: cuteafd_core::DType::U8, width: 4096, row_bytes: 4096 };
        // SAFETY: the sparse fixture stays immutable while the table lives.
        let table = unsafe { MappedTable::single(file.path(), 0, rows, format)? };
        Ok((file, table))
    }

    #[test]
    fn warm_waits_for_readiness_then_reads_every_table() -> TestResult {
        let (_a, first) = table(3)?;
        let (_b, second) = table(5)?;
        let checks = AtomicUsize::new(0);
        let outcome = warm_when_ready("test", &[&first, &second], 2, &AtomicBool::new(false),
            || checks.fetch_add(1, Ordering::Relaxed) == 2, || Ok(u64::MAX), Duration::from_millis(1))?;
        assert_eq!(checks.load(Ordering::Relaxed), 3);
        assert!(matches!(outcome, WarmOutcome::Warmed { bytes: 32768, .. }), "{outcome:?}");
        Ok(())
    }

    #[test]
    fn warm_admits_host_memory_for_the_tables_plus_the_margin() -> TestResult {
        let (_file, table) = table(4)?;
        let required = 4 * 4096 + WARM_MARGIN_BYTES;
        let warm = |available: anyhow::Result<u64>| warm_when_ready("test", &[&table], 1,
            &AtomicBool::new(false), || true, || available, READY_POLL);
        assert_eq!(warm(Ok(required - 1))?, WarmOutcome::Skipped { available: Some(required - 1), required });
        assert_eq!(warm(Err(anyhow::anyhow!("no meminfo")))?, WarmOutcome::Skipped { available: None, required });
        assert!(matches!(warm(Ok(required))?, WarmOutcome::Warmed { bytes: 16384, .. }));
        Ok(())
    }

    #[test]
    fn resident_tables_read_nothing_and_never_wait() -> TestResult {
        let (_file, mut table) = table(2)?;
        MappedTable::preload_all(&mut [&mut table])?;
        let outcome = warm_when_ready("test", &[&table], 1, &AtomicBool::new(false),
            || panic!("resident tables need no readiness"), || panic!("nor host memory"), READY_POLL)?;
        assert_eq!(outcome, WarmOutcome::Resident);
        Ok(())
    }

    #[test]
    fn a_stop_after_admission_reads_nothing() -> TestResult {
        let (_file, table) = table(2)?;
        let stop = AtomicBool::new(false);
        let outcome = warm_when_ready("test", &[&table], 1, &stop, || true,
            || { stop.store(true, Ordering::Release); Ok(u64::MAX) }, READY_POLL)?;
        assert_eq!(outcome, WarmOutcome::Stopped { bytes: 0 });
        Ok(())
    }

    #[test]
    fn dropping_the_handle_ends_a_readiness_wait_at_once() -> TestResult {
        let (_file, table) = table(2)?;
        let table = Arc::new(table);
        let (sender, outcome) = std::sync::mpsc::channel();
        let warm = TableWarm::spawn("warm-test", move |stop| {
            // A minute between checks: only the drop's unpark ends this wait quickly.
            let _ = sender.send(warm_when_ready("test", &[&table], 1, stop, || false, || Ok(u64::MAX),
                Duration::from_secs(60)));
        })?;
        let started = Instant::now();
        drop(warm);
        assert!(started.elapsed() < Duration::from_secs(30), "{:?}", started.elapsed());
        assert_eq!(outcome.recv()??, WarmOutcome::Stopped { bytes: 0 });
        Ok(())
    }

    #[test]
    fn a_started_warm_holds_its_tables_until_the_handle_joins() -> TestResult {
        let (_file, table) = table(2)?;
        let table = Arc::new(table);
        // Readiness never comes: only the drop ends the wait.
        let warm = TableWarm::start("warm-test", Arc::clone(&table), |table| vec![&**table], 1, || false)?;
        assert_eq!(Arc::strong_count(&table), 2, "the warm thread owns the tables it reads");
        let started = Instant::now();
        drop(warm);
        assert!(started.elapsed() < Duration::from_secs(30), "{:?}", started.elapsed());
        assert_eq!(Arc::strong_count(&table), 1, "a joined warm holds no table");
        Ok(())
    }

    #[test]
    fn progress_logs_each_tenth_once_and_leaves_the_end_to_completion() {
        assert_eq!(crossed_decile(0, 9, 100), None);
        assert_eq!(crossed_decile(9, 10, 100), Some(10));
        assert_eq!(crossed_decile(5, 35, 100), Some(30));
        assert_eq!(crossed_decile(90, 99, 100), None);
        assert_eq!(crossed_decile(85, 100, 100), None);
        // Chunks that tile the total mark 10% through 90% exactly once each.
        let marks: Vec<_> = (0..40).filter_map(|i| crossed_decile(i * 5, (i + 1) * 5, 200)).collect();
        assert_eq!(marks, (1..10).map(|tenth| tenth * 10).collect::<Vec<_>>());
    }
}
