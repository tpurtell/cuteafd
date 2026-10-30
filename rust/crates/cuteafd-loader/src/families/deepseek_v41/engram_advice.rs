//! Per-request advisory Engram read-ahead (#860). As soon as a request's
//! prompt tokens are known (prefill start, right after admission), both
//! layers' table rows are read into the page cache on background workers, so
//! the k-ahead gathers find them resident instead of faulting them on the
//! prefill's critical path. It matters where the tables do not stay cached:
//! a host whose memory cannot hold them, or shares it with other caches.
//!
//! Advice reads like the gathers ([`MappedTable::read_ahead`]: a mincore
//! snapshot on the worker, then the misses through the process-wide table read
//! reactor at prefill priority, behind the gathers' first reads and decode)
//! into a transient scratch of one read's rows. It holds no private copy, no
//! gather lease and no pinned staging, so it can never take a slot an
//! in-flight unit needs. One worker per Engram layer works through a bounded
//! queue of requests; a full queue drops the advice.
//!
//! Always on, with no setting; it backs off by itself. A piece (one layer's
//! rows for [`PIECE_TOKENS`] tokens) first reads every [`PROBE_STRIDE`]-th
//! row, and when none of those missed the page cache it skips the rest, so a
//! warm host pays one row in sixteen. Pieces the request has already committed
//! are skipped, and a worker never reads more than [`WINDOW_TOKENS`] past the
//! committed tokens, so the pages it fills are used before memory pressure
//! can evict them.
//!
//! [`MappedTable::read_ahead`]: crate::MappedTable::read_ahead
use crate::EngramTable;
use anyhow::{Context, Result};
use cuteafd_core::{EngramHistory, ENGRAM_LAYERS};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{mpsc, Arc};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

/// Tokens hashed and read per piece: one layer's rows for this many tokens.
pub const PIECE_TOKENS: usize = 1024;
/// Rows per read, which bounds a read's scratch (freed when the read returns)
/// and how long a stop or cancel waits for the read in flight.
const SLICE_ROWS: usize = 4096;
/// A piece first reads every `PROBE_STRIDE`-th row; when none of those missed
/// the page cache, the rest of the piece is skipped.
pub const PROBE_STRIDE: usize = 16;
/// Requests queued per worker. Prefill runs one request at a time, so this
/// holds the running request and the next few; beyond it advice is dropped.
const QUEUE: usize = 4;
/// How far past a request's committed tokens a worker reads ahead: 32768
/// tokens x 2 layers x 48 rows, at most 12 GiB of page cache when every row
/// sits on its own 4 KiB page.
pub const WINDOW_TOKENS: u64 = 32768;
/// How often a worker held at the window rechecks the committed tokens.
const WINDOW_POLL: Duration = Duration::from_millis(2);

/// One request's advice, shared by both layer workers.
struct AdviceJob {
    /// The request's Engram frontier under a fresh identity: batches hashed
    /// from it can never be accepted by the request.
    history: EngramHistory,
    /// Compressed token ids from that frontier on (None: an image row).
    tokens: Vec<Option<u32>>,
    /// The request has committed every token before this position.
    committed: AtomicU64,
    cancelled: AtomicBool,
}

impl AdviceJob {
    fn end(&self) -> u64 { self.history.position() + self.tokens.len() as u64 }
}

/// A request's queued advice. Dropping it (release) stops the workers at their
/// next read; reads already issued only fill the page cache.
pub struct EngramAdvice(Arc<AdviceJob>);

impl EngramAdvice {
    /// The request committed every token before `position`: workers skip
    /// pieces behind it and read at most [`WINDOW_TOKENS`] past it. Returns
    /// whether any advised token lies at or past `position`.
    pub fn commit_to(&self, position: u64) -> bool {
        self.0.committed.fetch_max(position, Ordering::Release);
        position < self.0.end()
    }
}

impl Drop for EngramAdvice {
    fn drop(&mut self) {
        self.0.cancelled.store(true, Ordering::Release);
    }
}

struct Worker {
    sender: Option<mpsc::SyncSender<Arc<AdviceJob>>>,
    thread: Option<JoinHandle<()>>,
}

/// Both layers' advice workers. Dropping it stops them at their next read
/// and joins them.
pub struct EngramAdvisor {
    tables: [Arc<EngramTable>; 2],
    workers: Vec<Worker>,
    stop: Arc<AtomicBool>,
}

impl EngramAdvisor {
    /// One worker per layer's table (layer 1, then layer 14).
    pub fn new(tables: [Arc<EngramTable>; 2]) -> Result<Self> {
        let mut advisor = Self { tables: tables.clone(), workers: Vec::with_capacity(2),
            stop: Arc::new(AtomicBool::new(false)) };
        for (layer, table) in tables.into_iter().enumerate() {
            let (sender, receiver) = mpsc::sync_channel::<Arc<AdviceJob>>(QUEUE);
            let stop = Arc::clone(&advisor.stop);
            // A failed spawn drops `advisor`, which joins the workers already started.
            let thread = thread::Builder::new().name(format!("engram-advice-{}", ENGRAM_LAYERS[layer]))
                .spawn(move || {
                    while let Ok(job) = receiver.recv() {
                        if let Err(error) = advise(&table, layer, &job, &stop) {
                            tracing::warn!(layer = ENGRAM_LAYERS[layer], %error, "Engram advice stopped; rows load on demand");
                        }
                    }
                }).context("starting Engram advice worker")?;
            advisor.workers.push(Worker { sender: Some(sender), thread: Some(thread) });
        }
        tracing::info!(workers = advisor.workers.len(), piece_tokens = PIECE_TOKENS, probe_stride = PROBE_STRIDE,
            window_tokens = WINDOW_TOKENS, queue = QUEUE, "Engram read-ahead armed");
        Ok(advisor)
    }

    /// Upper bound of both workers' transient memory while they advise: per
    /// worker, a piece's hashes, rows and probe split (`PIECE_TOKENS` x 1 KiB
    /// for 776 B a token), and one read's scratch rows, descriptors, page set
    /// and residency flags (`SLICE_ROWS` x 1 KiB for under 512 B a row). The
    /// reactor's buffers and ring are admitted once per process, not here.
    /// Queued requests' tokens are request batches, not workspace.
    pub fn workspace_bytes() -> usize {
        2 * (PIECE_TOKENS * 1024 + SLICE_ROWS * 1024)
    }

    /// Queue advice for compressed `tokens` from `history`'s position on, on
    /// every layer whose tables are not resident. None when nothing was
    /// queued: no text rows, every table resident, or every queue full (each
    /// full queue counts a drop on its layer's weight table).
    pub fn advise(&self, history: &EngramHistory, tokens: Vec<Option<u32>>) -> Result<Option<EngramAdvice>> {
        let layers: Vec<usize> = (0..2).filter(|&layer| !resident(&self.tables[layer])).collect();
        if layers.is_empty() || tokens.iter().all(Option::is_none) {
            return Ok(None);
        }
        let job = Arc::new(AdviceJob { history: history.fork()?, tokens,
            committed: AtomicU64::new(history.position()), cancelled: AtomicBool::new(false) });
        let mut queued = false;
        for layer in layers {
            let sender = self.workers[layer].sender.as_ref().context("Engram advisor stopped")?;
            match sender.try_send(Arc::clone(&job)) {
                Ok(()) => queued = true,
                Err(mpsc::TrySendError::Full(_)) => self.tables[layer].weights().stats().record_prefetch_dropped(),
                Err(mpsc::TrySendError::Disconnected(_)) => anyhow::bail!("Engram advice worker stopped"),
            }
        }
        Ok(queued.then(|| EngramAdvice(job)))
    }
}

impl Drop for EngramAdvisor {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Release);
        for worker in &mut self.workers {
            worker.sender.take();
        }
        for worker in &mut self.workers {
            if let Some(thread) = worker.thread.take() {
                thread.thread().unpark();
                if thread.join().is_err() {
                    tracing::error!("Engram advice worker panicked");
                }
            }
        }
    }
}

/// Preloaded tables are already in memory: there is nothing to read ahead.
fn resident(table: &EngramTable) -> bool {
    parts(table).all(|part| part.warm_bytes() == 0)
}

/// The row storage, then the separate scales when the rows do not carry them.
fn parts(table: &EngramTable) -> impl Iterator<Item = &crate::MappedTable> {
    std::iter::once(table.weights()).chain(table.scales())
}

/// Read one request's `layer` rows piece by piece, in token order.
fn advise(table: &EngramTable, layer: usize, job: &AdviceJob, stop: &AtomicBool) -> Result<()> {
    let live = || !stop.load(Ordering::Acquire) && !job.cancelled.load(Ordering::Acquire);
    let started = Instant::now();
    let (mut read, mut skipped) = (0usize, 0usize);
    let mut history = job.history.fork()?;
    for tokens in job.tokens.chunks(PIECE_TOKENS) {
        let start = history.position();
        let end = start + tokens.len() as u64;
        let batch = history.prepare(start, tokens, tokens.len())?;
        history.commit(&batch, tokens.len())?;
        // Hold at the window until the request commits far enough.
        while live() && end > job.committed.load(Ordering::Acquire)
            && start > job.committed.load(Ordering::Acquire).saturating_add(WINDOW_TOKENS) {
            thread::park_timeout(WINDOW_POLL);
        }
        if !live() {
            break;
        }
        let rows = batch.prefetch_rows(layer)?;
        if end <= job.committed.load(Ordering::Acquire) {
            // Already consumed: the gathers read these rows.
            table.weights().stats().record_prefetch_skipped(rows.len());
            skipped += 1;
        } else if read_piece(table, &rows, &live)? {
            read += 1;
        } else {
            skipped += 1;
        }
    }
    tracing::debug!(layer = ENGRAM_LAYERS[layer], tokens = job.tokens.len(), pieces_read = read, pieces_skipped = skipped,
        cancelled = !live(), elapsed_ms = started.elapsed().as_millis() as u64, "Engram advice");
    Ok(())
}

/// Read every `PROBE_STRIDE`-th row first and the rest only if one of them
/// missed the page cache. Returns whether the rest was read.
fn read_piece(table: &EngramTable, rows: &[u64], live: &dyn Fn() -> bool) -> Result<bool> {
    let probe: Vec<u64> = rows.iter().step_by(PROBE_STRIDE).copied().collect();
    if read_rows(table, &probe, live)? == Some(0) {
        table.weights().stats().record_prefetch_skipped(rows.len() - probe.len());
        return Ok(false);
    }
    let rest: Vec<u64> = rows.iter().enumerate().filter(|&(index, _)| index % PROBE_STRIDE != 0)
        .map(|(_, &row)| row).collect();
    read_rows(table, &rest, live)?;
    Ok(true)
}

/// Read `rows` of every table part, `SLICE_ROWS` at a time. Returns the misses found:
/// None when residency is unknown (mmap advice) or a stop ended the reads early.
fn read_rows(table: &EngramTable, rows: &[u64], live: &dyn Fn() -> bool) -> Result<Option<u64>> {
    let mut misses = Some(0);
    for slice in rows.chunks(SLICE_ROWS) {
        if !live() {
            return Ok(None);
        }
        for part in parts(table) {
            if part.warm_bytes() == 0 {
                continue; // resident (preloaded scales)
            }
            // Rows are at most a page wide, so each spans at most two pages.
            let read = part.read_ahead(slice, 2 * slice.len())?;
            part.stats().record_prefetch(read.pages);
            misses = misses.zip(read.misses).map(|(total, part)| total + part);
        }
    }
    Ok(misses)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{MappedTable, RowFormat};
    use std::io::Write;

    type TestResult = std::result::Result<(), Box<dyn std::error::Error>>;
    type Fixture = (tempfile::NamedTempFile, tempfile::NamedTempFile, EngramTable);

    /// An FP8 Engram table over `rows` rows of the given weight and scale
    /// bytes, preloaded into memory when `resident`.
    fn table_of(rows: u64, resident: bool, fill: impl Fn(&mut std::fs::File, &mut std::fs::File) -> std::io::Result<()>)
        -> std::result::Result<Fixture, Box<dyn std::error::Error>> {
        let weights = tempfile::NamedTempFile::new()?;
        let scales = tempfile::NamedTempFile::new()?;
        fill(&mut weights.reopen()?, &mut scales.reopen()?)?;
        // SAFETY: the fixture files stay immutable while the tables live.
        let (mut weight_table, mut scale_table) = unsafe {
            (MappedTable::single(weights.path(), 0, rows, RowFormat::of(cuteafd_core::DType::F8E4M3, 256)?)?,
             MappedTable::single(scales.path(), 0, rows, RowFormat::of(cuteafd_core::DType::F8E8M0, 8)?)?)
        };
        if resident {
            MappedTable::preload_all(&mut [&mut weight_table, &mut scale_table], 1)?;
        }
        Ok((weights, scales, EngramTable::new(weight_table, scale_table)?))
    }

    /// Small dense rows, addressed directly (no hashing).
    fn small(rows: u64, resident: bool) -> std::result::Result<Fixture, Box<dyn std::error::Error>> {
        table_of(rows, resident, |weights, scales| {
            for row in 0..rows {
                weights.write_all(&[row as u8; 256])?;
                scales.write_all(&[127; 8])?;
            }
            Ok(())
        })
    }

    /// A layer's official row count over sparse files, so its hashed rows are in range.
    fn official(layer: usize) -> std::result::Result<Fixture, Box<dyn std::error::Error>> {
        let rows = cuteafd_core::ENGRAM_ROWS[layer];
        table_of(rows, false, |weights, scales| { weights.set_len(rows * 256)?; scales.set_len(rows * 8) })
    }

    fn job(position: u64, tokens: usize, committed: u64) -> std::result::Result<AdviceJob, Box<dyn std::error::Error>> {
        let recent: Vec<_> = (0..position.min(3) as u32).map(|t| Some(t + 5)).collect();
        Ok(AdviceJob { history: EngramHistory::from_recent(2, position, &recent)?,
            tokens: (0..tokens as u32).map(|t| Some(t % 997)).collect(),
            committed: AtomicU64::new(committed), cancelled: AtomicBool::new(false) })
    }

    #[test]
    fn a_probe_with_no_misses_skips_the_rest_of_the_piece() -> TestResult {
        let (_w, _s, table) = small(64, false)?;
        let rows: Vec<u64> = (0..64).collect();
        let warmed = read_rows(&table, &rows, &|| true)?;
        let before = table.stats();
        let rest_read = read_piece(&table, &rows, &|| true)?;
        let after = table.weights().stats().snapshot().since(&before);
        let probed = 64usize.div_ceil(PROBE_STRIDE) as u64;
        match warmed {
            // The mmap fallback only advises, so residency is unknown: read it all.
            None => assert!(rest_read),
            Some(_) => {
                assert!(!rest_read, "a fully resident probe skips the rest");
                assert_eq!((after.prefetch_hits, after.prefetch_misses), (probed, 0));
                assert_eq!(after.prefetch_skipped_rows, 64 - probed);
            }
        }
        Ok(())
    }

    #[test]
    fn a_probe_miss_reads_every_row_once() -> TestResult {
        // Clean, unmapped pages on a disk file (not tmpfs) can be evicted.
        let dir = tempfile::tempdir_in(std::env::current_dir()?)?;
        let fixture = |name: &str, bytes: Vec<u8>| -> std::io::Result<std::path::PathBuf> {
            use std::os::fd::AsRawFd;
            let path = dir.path().join(name);
            let file = std::fs::File::create(&path)?;
            (&file).write_all(&bytes)?;
            file.sync_all()?;
            // SAFETY: fadvise only drops this live, synced fd's clean cached pages.
            unsafe { libc::posix_fadvise(file.as_raw_fd(), 0, 0, libc::POSIX_FADV_DONTNEED) };
            Ok(path)
        };
        let rows = 512u64;
        let weights = fixture("weights", (0..rows * 256).map(|i| (i / 256) as u8).collect())?;
        let scales = fixture("scales", vec![127; rows as usize * 8])?;
        // SAFETY: the fixture files stay immutable while the table lives.
        let table = unsafe { EngramTable::new(
            MappedTable::single(&weights, 0, rows, RowFormat::of(cuteafd_core::DType::F8E4M3, 256)?)?,
            MappedTable::single(&scales, 0, rows, RowFormat::of(cuteafd_core::DType::F8E8M0, 8)?)?)? };
        // One row per 4 KiB page, duplicates included: every row is its own read.
        let piece: Vec<u64> = (0..40).map(|i| (i * 16 + 7) % rows).collect();
        let before = table.stats();
        let rest_read = read_piece(&table, &piece, &|| true)?;
        let after = table.weights().stats().snapshot().since(&before);
        let probed = piece.len().div_ceil(PROBE_STRIDE) as u64;
        if after.prefetch_hits + after.prefetch_misses == 0 {
            eprintln!("SKIP: table reactor unavailable; the mmap fallback only advises");
            assert!(rest_read);
        } else if rest_read {
            assert!(after.prefetch_misses > 0, "only a probe miss reads the rest");
            assert_eq!(after.prefetch_hits + after.prefetch_misses, piece.len() as u64, "each row is read once");
            assert_eq!(after.prefetch_skipped_rows, 0);
        } else {
            eprintln!("SKIP cold probe: eviction left the pages cached");
            assert_eq!((after.prefetch_hits, after.prefetch_misses), (probed, 0));
            assert_eq!(after.prefetch_skipped_rows, piece.len() as u64 - probed);
        }
        Ok(())
    }

    #[test]
    fn stopped_reads_report_unknown_residency_and_read_nothing() -> TestResult {
        let (_w, _s, table) = small(8, false)?;
        let before = table.stats();
        assert_eq!(read_rows(&table, &[0, 1, 2], &|| false)?, None);
        assert_eq!(table.weights().stats().snapshot().since(&before).prefetch_jobs, 0);
        Ok(())
    }

    #[test]
    fn committed_pieces_are_skipped_without_reads() -> TestResult {
        let (_w, _s, table) = official(0)?;
        let tokens = 3 * PIECE_TOKENS;
        let before = table.stats();
        advise(&table, 0, &job(0, tokens, tokens as u64)?, &AtomicBool::new(false))?;
        let delta = table.weights().stats().snapshot().since(&before);
        assert_eq!(delta.prefetch_jobs, 0);
        assert_eq!(delta.prefetch_skipped_rows, (tokens * 24) as u64);
        Ok(())
    }

    #[test]
    fn a_piece_past_the_window_waits_for_commits_and_a_cancel_ends_it() -> TestResult {
        let (_w, _s, table) = official(0)?;
        let table = Arc::new(table);
        let far = WINDOW_TOKENS + 1;
        for cancel in [false, true] {
            let ahead = Arc::new(job(far, 1, 0)?);
            let before = table.stats();
            let worker = {
                let (ahead, table) = (Arc::clone(&ahead), Arc::clone(&table));
                thread::spawn(move || advise(&table, 0, &ahead, &AtomicBool::new(false)))
            };
            thread::sleep(Duration::from_millis(50));
            assert!(!worker.is_finished(), "a piece past the window must wait");
            assert_eq!(table.weights().stats().snapshot().since(&before).prefetch_jobs, 0);
            if cancel {
                ahead.cancelled.store(true, Ordering::Release);
            } else {
                ahead.committed.store(far, Ordering::Release);
            }
            worker.join().expect("advice worker")?;
            let delta = table.weights().stats().snapshot().since(&before);
            assert_eq!(delta.prefetch_jobs > 0, !cancel, "a commit releases the read; a cancel ends it unread");
        }
        Ok(())
    }

    #[test]
    fn advice_never_queues_resident_tables_and_drops_on_a_full_queue() -> TestResult {
        let (_a, _b, first) = small(4, true)?;
        let (_c, _d, second) = official(1)?;
        let advisor = EngramAdvisor::new([Arc::new(first), Arc::new(second)])?;
        let history = EngramHistory::new(2)?;
        assert!(advisor.advise(&history, vec![None, None])?.is_none(), "image rows read nothing");
        // Hold layer 14's worker at its window, so its queue fills.
        let held = Arc::new(job(WINDOW_TOKENS + 1, 1, 0)?);
        advisor.workers[1].sender.as_ref().unwrap().send(Arc::clone(&held))?;
        let advice: Vec<_> = (0..64).filter_map(|_| advisor.advise(&history, vec![Some(7)]).transpose())
            .collect::<Result<_>>()?;
        // Layer 1 is resident, so nothing queued there; layer 14's queue held at most QUEUE.
        let dropped = advisor.tables[1].weights().stats().snapshot().prefetch_dropped;
        assert!((QUEUE - 1..=QUEUE).contains(&advice.len()), "{} queued", advice.len());
        assert_eq!(advice.len() as u64 + dropped, 64);
        assert_eq!(advisor.tables[0].weights().stats().snapshot().prefetch_jobs, 0);
        held.cancelled.store(true, Ordering::Release);
        drop(advice);
        drop(advisor);
        Ok(())
    }

    #[test]
    fn the_workspace_bound_covers_a_piece_and_a_read() {
        let word = std::mem::size_of::<usize>();
        // Per token: the batch's tokens and both layers' hashes, one layer's rows,
        // and the probe/rest split of those rows.
        let token = 8 + 2 * 24 * 8 + 24 * 8 + 24 * 8;
        // Per row of a read: scratch (264 B strided FP8 row), read descriptor, two-page
        // set entries with tree overhead, residency pages and flags, the miss
        // list with its hit/residency flags, the reactor's read descriptor and
        // the slot hash set.
        let row = 264 + 3 * word + 2 * 3 * 2 * word + (2 * 2 * word + 2) + (3 * word + 2) + 3 * word + 4 * word;
        assert!(token <= 1024 && row <= 512, "{token} B a token, {row} B a row");
        assert_eq!(EngramAdvisor::workspace_bytes(), 2 * (PIECE_TOKENS * 1024 + SLICE_ROWS * 1024));
    }
}
