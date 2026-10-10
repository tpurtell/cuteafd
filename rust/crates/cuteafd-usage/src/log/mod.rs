//! The full-log tier: payloads in `usage-log.sqlite` and media files under
//! `media/`, never in the metadata file. Requests are stored as delta entries
//! in virtual sessions; the serving side only hands over `Bytes` refcounts.
mod fold;
mod normalize;
mod redact;
#[cfg(test)]
mod tests;

use crate::store::{vacuum, Clock, Error, Result, Settings};
use arc_swap::ArcSwap;
use cuteafd_api::{
    usage::Counters,
    usage_log::{LogRecord, LogSink},
};
use normalize::{chain_hash, content_hash, item_hash, Split};
use rusqlite::{params, params_from_iter, Connection, OptionalExtension};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{
    path::{Path, PathBuf},
    sync::{
        atomic::{AtomicBool, AtomicUsize, Ordering::Relaxed},
        mpsc::{self, SyncSender},
        Arc, Mutex,
    },
    time::{Duration, Instant},
};

const INFLIGHT_BYTES: usize = 64 << 20;
/// Longest string kept verbatim in a stored entry.
const STRING_CAP: usize = 256 << 10;
/// Stored bytes per entry (delta items plus response) before whole items are elided.
const ENTRY_CAP: usize = 8 << 20;

enum Command {
    Record(Box<LogRecord>, usize),
    Flush(mpsc::Sender<Result<()>>),
    Clear(mpsc::Sender<Result<()>>),
    Prune(mpsc::Sender<Result<()>>),
}

pub struct LogStore {
    tx: SyncSender<Command>,
    bytes: Arc<AtomicUsize>,
    pub(crate) enabled: Arc<AtomicBool>,
    settings: Arc<ArcSwap<Settings>>,
    counters: Arc<Counters>,
    reader: Mutex<Connection>,
    dir: Option<PathBuf>,
    clock: Arc<dyn Clock>,
    secrets: Arc<ArcSwap<Vec<String>>>,
}

pub(crate) fn on(s: &Settings) -> bool {
    s.log_enabled && s.log_hours > 0 && s.log_cap_mb > 0
}

const SCHEMA: &str = "PRAGMA auto_vacuum=INCREMENTAL; PRAGMA journal_mode=WAL; PRAGMA synchronous=NORMAL;
DROP TABLE IF EXISTS log;
CREATE TABLE IF NOT EXISTS entries(
  rid TEXT PRIMARY KEY, ts_ms INTEGER NOT NULL, protocol TEXT NOT NULL,
  vsid TEXT NOT NULL, kind TEXT NOT NULL, parent_rid TEXT, parent_count INTEGER,
  divergence INTEGER, diverged_from TEXT, n_items INTEGER NOT NULL, chain TEXT NOT NULL,
  hashes BLOB NOT NULL, system_h TEXT, tools_h TEXT, settings_h TEXT,
  items BLOB NOT NULL, response BLOB, response_id TEXT,
  client TEXT, model TEXT, session_id TEXT, session_source TEXT, bench INTEGER NOT NULL DEFAULT 0,
  status INTEGER, outcome TEXT, title TEXT, meta BLOB NOT NULL,
  bytes INTEGER NOT NULL, truncated INTEGER NOT NULL, scope TEXT NOT NULL DEFAULT '');
CREATE INDEX IF NOT EXISTS entries_ts ON entries(ts_ms);
CREATE INDEX IF NOT EXISTS entries_chain ON entries(chain, scope);
CREATE INDEX IF NOT EXISTS entries_parent ON entries(parent_rid) WHERE parent_rid IS NOT NULL;
CREATE INDEX IF NOT EXISTS entries_vsid ON entries(vsid, ts_ms);
CREATE INDEX IF NOT EXISTS entries_session ON entries(session_id, ts_ms) WHERE session_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS entries_response ON entries(response_id) WHERE response_id IS NOT NULL;
CREATE TABLE IF NOT EXISTS blobs(h TEXT PRIMARY KEY, value BLOB NOT NULL);
CREATE TABLE IF NOT EXISTS media(sha256 TEXT PRIMARY KEY, mime TEXT NOT NULL, ext TEXT NOT NULL,
  bytes INTEGER NOT NULL, stored INTEGER NOT NULL);
CREATE TABLE IF NOT EXISTS entry_media(rid TEXT NOT NULL, sha256 TEXT NOT NULL, PRIMARY KEY(rid, sha256));
CREATE INDEX IF NOT EXISTS entry_media_sha ON entry_media(sha256);";

impl LogStore {
    pub(crate) fn open(
        directory: Option<&Path>,
        settings: Arc<ArcSwap<Settings>>,
        counters: Arc<Counters>,
        clock: Arc<dyn Clock>,
    ) -> Result<Arc<Self>> {
        let path = directory.map(|d| d.join("usage-log.sqlite"));
        let uri = path
            .as_ref()
            .map(|p| p.to_string_lossy().into_owned())
            .unwrap_or_else(|| format!("file:usage-log-{}?mode=memory&cache=shared", uuid::Uuid::new_v4()));
        let c = Connection::open(&uri)?;
        c.busy_timeout(Duration::from_millis(2000))?;
        c.execute_batch("PRAGMA secure_delete=ON;")?;
        // Entries from an earlier layout lack the chain scope; they cannot be matched safely.
        let scoped = c.prepare("PRAGMA table_info(entries)")?.query_map([], |r| r.get::<_, String>(1))?
            .collect::<std::result::Result<Vec<_>, _>>()?;
        if !scoped.is_empty() && !scoped.iter().any(|n| n == "scope") {
            c.execute_batch("DROP TABLE entries; DROP TABLE IF EXISTS entry_media;")?;
        }
        c.execute_batch(SCHEMA)?;
        if let Some(dir) = directory {
            crate::store::private_files(dir, "usage-log.sqlite");
        }
        let reader = if path.is_some() {
            Connection::open_with_flags(&uri, rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY)?
        } else {
            Connection::open(&uri)?
        };
        reader.busy_timeout(Duration::from_millis(2000))?;
        let (tx, rx) = mpsc::sync_channel(1024);
        let bytes = Arc::new(AtomicUsize::new(0));
        let enabled = Arc::new(AtomicBool::new(on(&settings.load())));
        let dir = directory.map(Path::to_path_buf);
        let secrets = Arc::new(ArcSwap::from_pointee(Vec::new()));
        let store = Arc::new(Self {
            tx,
            bytes: bytes.clone(),
            enabled,
            settings: settings.clone(),
            counters: counters.clone(),
            reader: Mutex::new(reader),
            dir: dir.clone(),
            clock: clock.clone(),
            secrets: secrets.clone(),
        });
        let s = settings.load();
        if on(&s) {
            tracing::info!(
                hours = s.log_hours,
                cap_mb = s.log_cap_mb,
                media = s.log_media,
                "usage full log on: prompts and model outputs are stored in plain text for the retention period (turn off on /usage settings or with --usage off)"
            );
        } else {
            tracing::info!("usage full log off");
        }
        let mut writer = Writer { c, dir, settings, counters, clock, secrets };
        writer.update_bytes();
        std::thread::Builder::new()
            .name("usage-log-writer".into())
            .spawn(move || writer.run(rx, bytes))?;
        Ok(store)
    }
    fn command(&self, make: impl FnOnce(mpsc::Sender<Result<()>>) -> Command) -> Result<()> {
        let (tx, rx) = mpsc::channel();
        self.tx.send(make(tx)).map_err(|_| Error::Stopped)?;
        rx.recv().map_err(|_| Error::Stopped)?
    }
    pub fn flush(&self) -> Result<()> {
        self.command(Command::Flush)
    }
    /// Server secrets (the API key, the console secret) scrubbed from every payload.
    pub fn set_secrets(&self, secrets: Vec<String>) {
        self.secrets.store(Arc::new(secrets));
    }
    pub fn clear(&self) -> Result<()> {
        self.command(Command::Clear)
    }
    pub fn prune(&self) -> Result<()> {
        self.command(Command::Prune)
    }
    fn read<T>(&self, f: impl FnOnce(&Connection) -> Result<T>) -> Result<T> {
        f(&*self.reader.lock().map_err(|_| Error::Stopped)?)
    }
    /// One entry with its full request rebuilt from its chain.
    pub fn get(&self, rid: &str) -> Result<Option<Value>> {
        self.read(|c| view::entry_full(c, rid))
    }
    /// Virtual sessions, newest activity first.
    pub fn sessions(&self, f: &crate::query::Filter) -> Result<Value> {
        let (from, to) = f.bounds(self.clock.now_ms())?;
        self.read(|c| view::sessions(c, f, from, to))
    }
    /// One virtual session's entries in order, each with only its new items.
    pub fn session(&self, vsid: &str) -> Result<Option<Value>> {
        self.read(|c| view::session(c, vsid))
    }
    /// A stored media file: (mime, path), or None when not retained or reference-only.
    pub fn media(&self, sha256: &str) -> Result<Option<(String, PathBuf)>> {
        if sha256.len() != 64 || !sha256.bytes().all(|b| b.is_ascii_hexdigit() && !b.is_ascii_uppercase()) {
            return Ok(None);
        }
        let Some(dir) = &self.dir else { return Ok(None) };
        let row: Option<(String, String)> = self.read(|c| {
            Ok(c.query_row(
                "SELECT mime, ext FROM media WHERE sha256=?1 AND stored=1",
                [sha256],
                |r| Ok((r.get(0)?, r.get(1)?)),
            )
            .optional()?)
        })?;
        Ok(row.map(|(mime, ext)| (mime, media_path(dir, sha256, &ext))))
    }
}

fn media_path(dir: &Path, sha256: &str, ext: &str) -> PathBuf {
    dir.join("media").join(&sha256[..2]).join(format!("{sha256}.{ext}"))
}

impl LogSink for LogStore {
    fn enabled(&self) -> bool {
        self.enabled.load(Relaxed)
    }
    fn bench(&self) -> bool {
        self.settings.load().record_bench
    }
    fn record_log(&self, r: LogRecord) {
        if !self.enabled() || !cuteafd_api::usage_log::loggable(&r.meta.protocol) {
            return;
        }
        let n = r.byte_len();
        if self
            .bytes
            .fetch_update(Relaxed, Relaxed, |v| v.checked_add(n).filter(|t| *t <= INFLIGHT_BYTES))
            .is_err()
        {
            self.counters.log_dropped.fetch_add(1, Relaxed);
            return;
        }
        if self.tx.try_send(Command::Record(Box::new(r), n)).is_err() {
            self.bytes.fetch_sub(n, Relaxed);
            self.counters.log_dropped.fetch_add(1, Relaxed);
        }
    }
}

struct Writer {
    c: Connection,
    dir: Option<PathBuf>,
    settings: Arc<ArcSwap<Settings>>,
    counters: Arc<Counters>,
    clock: Arc<dyn Clock>,
    secrets: Arc<ArcSwap<Vec<String>>>,
}

/// The stored shape of one entry, before insert.
struct Entry {
    kind: &'static str,
    vsid: String,
    parent: Option<String>,
    parent_count: Option<usize>,
    divergence: Option<usize>,
    diverged_from: Option<String>,
    title: Option<String>,
    stored_items: Vec<Value>,
}

impl Writer {
    fn run(&mut self, rx: mpsc::Receiver<Command>, inflight: Arc<AtomicUsize>) {
        let mut last = Instant::now();
        loop {
            match rx.recv_timeout(Duration::from_millis(50)) {
                Ok(Command::Record(r, n)) => {
                    let result = if on(&self.settings.load()) { self.write(&r) } else { Ok(false) };
                    drop(r);
                    inflight.fetch_sub(n, Relaxed);
                    match result {
                        Ok(true) => {
                            self.counters.log_recorded.fetch_add(1, Relaxed);
                        }
                        Ok(false) => {}
                        Err(e) => {
                            self.counters.log_dropped.fetch_add(1, Relaxed);
                            tracing::error!(error = %e, "usage full log write failed");
                        }
                    }
                    self.update_bytes();
                }
                Ok(Command::Flush(reply)) => {
                    self.update_bytes();
                    let _ = reply.send(Ok(()));
                }
                Ok(Command::Clear(reply)) => {
                    let result = self.clear();
                    self.update_bytes();
                    let _ = reply.send(result);
                }
                Ok(Command::Prune(reply)) => {
                    let result = self.prune(self.clock.now_ms());
                    self.update_bytes();
                    let _ = reply.send(result);
                }
                Err(mpsc::RecvTimeoutError::Timeout) => {}
                Err(_) => break,
            }
            if last.elapsed() >= Duration::from_secs(60) {
                if let Err(e) = self.prune(self.clock.now_ms()) {
                    tracing::error!(error = %e, "usage full log prune failed");
                }
                self.update_bytes();
                last = Instant::now();
            }
        }
    }

    fn update_bytes(&mut self) {
        let used = db_used(&self.c).unwrap_or(0);
        let (media, files): (i64, i64) = self
            .c
            .query_row("SELECT coalesce(sum(bytes),0), count(*) FROM media WHERE stored=1", [], |r| Ok((r.get(0)?, r.get(1)?)))
            .unwrap_or((0, 0));
        self.counters.log_bytes.store(used, Relaxed);
        self.counters.media_bytes.store(media as u64, Relaxed);
        self.counters.media_files.store(files as u64, Relaxed);
    }

    fn blob(&self, v: &Option<Value>) -> Result<Option<String>> {
        let Some(v) = v else { return Ok(None) };
        let h = content_hash(v);
        self.c.execute("INSERT OR IGNORE INTO blobs VALUES(?1, ?2)", params![h, serde_json::to_vec(v)?])?;
        Ok(Some(h))
    }

    /// Records media rows inside the entry's transaction; returns the blobs whose
    /// files must be written once that transaction commits.
    fn record_media<'m>(&self, media: &'m [redact::Media], keep_bytes: bool) -> Result<Vec<&'m redact::Media>> {
        let keep = keep_bytes && self.dir.is_some();
        let mut pending = vec![];
        for m in media {
            let ext = redact::extension(&m.mime);
            let exists: Option<bool> = self
                .c
                .query_row("SELECT stored FROM media WHERE sha256=?1", [&m.sha256], |r| r.get(0))
                .optional()?;
            if exists == Some(true) || (exists == Some(false) && !keep) {
                continue;
            }
            self.c.execute(
                "INSERT INTO media VALUES(?1,?2,?3,?4,?5) ON CONFLICT(sha256) DO UPDATE SET stored=excluded.stored, ext=excluded.ext",
                params![m.sha256, m.mime, ext, m.bytes.len() as i64, keep],
            )?;
            if keep {
                pending.push(m);
            }
        }
        Ok(pending)
    }

    /// Writes committed media files (0600 in 0700 directories); a failed write
    /// downgrades the row to a reference.
    fn write_media(&self, pending: &[&redact::Media]) {
        let Some(dir) = &self.dir else { return };
        for m in pending {
            let path = media_path(dir, &m.sha256, redact::extension(&m.mime));
            let written = (|| -> std::io::Result<()> {
                use std::io::Write;
                use std::os::unix::fs::OpenOptionsExt;
                let parent = path.parent().expect("media file has a parent");
                crate::store::private_dir(&dir.join("media"))?;
                crate::store::private_dir(parent)?;
                if !path.exists() {
                    let tmp = parent.join(format!(".{}.tmp", m.sha256));
                    let mut f = std::fs::OpenOptions::new().write(true).create(true).truncate(true).mode(0o600).open(&tmp)?;
                    f.write_all(&m.bytes)?;
                    f.sync_data()?;
                    std::fs::rename(&tmp, &path)?;
                }
                Ok(())
            })();
            if let Err(e) = written {
                tracing::error!(error = %e, "usage media write failed; keeping a reference only");
                let _ = self.c.execute("UPDATE media SET stored=0 WHERE sha256=?1", [&m.sha256]);
            }
        }
    }

    /// Deletes files under `media/` that no stored row accounts for (a crash
    /// between commit and write, or files from a cleared store).
    fn sweep_media(&self) -> Result<()> {
        let Some(dir) = &self.dir else { return Ok(()) };
        let Ok(top) = std::fs::read_dir(dir.join("media")) else { return Ok(()) };
        for sub in top.flatten() {
            let Ok(files) = std::fs::read_dir(sub.path()) else { continue };
            for f in files.flatten() {
                let name = f.file_name().to_string_lossy().into_owned();
                let sha = name.split('.').next().unwrap_or("");
                let tracked: Option<String> = self
                    .c
                    .query_row("SELECT ext FROM media WHERE sha256=?1 AND stored=1", [sha], |r| r.get(0))
                    .optional()?;
                if tracked.is_none_or(|ext| name != format!("{sha}.{ext}")) {
                    let _ = std::fs::remove_file(f.path());
                }
            }
            let _ = std::fs::remove_dir(sub.path());
        }
        Ok(())
    }

    fn write(&mut self, r: &LogRecord) -> Result<bool> {
        let meta = &r.meta;
        let protocol = meta.protocol.as_str();
        let mut redactor = redact::Redactor::with_secrets(&self.secrets.load());
        let mut request = match &r.request_value {
            Some(v) => v.clone(),
            None => fold::parse(&r.request.iter().flat_map(|b| b.iter().copied()).collect::<Vec<u8>>()),
        };
        redactor.redact(&mut request);
        let mut response = fold::fold(protocol, &r.response);
        redactor.redact(&mut response);
        let mut truncated = r.request_truncated || response.get("log_truncated").is_some();
        let split = normalize::split(protocol, request);
        let hashes = split.items.iter().map(item_hash).collect::<Vec<_>>();
        let chain = chain_hash(&hashes);
        let settings = self.settings.load();

        // A request too large to store whole starts its own chain position.
        let oversized = split.items.iter().map(|v| v.to_string().len()).sum::<usize>() > ENTRY_CAP
            || split.items.iter().any(has_long_string);
        let entry = if oversized {
            Placement::Entry(Entry { kind: "base", vsid: meta.rid.clone(), parent: None, parent_count: None,
                divergence: None, diverged_from: None, title: None, stored_items: split.items.clone() })
        } else {
            self.place(meta, protocol, &split, &hashes)?
        };
        let (entry, full_hashes) = match entry {
            Placement::Chained { parent, parent_full, vsid, title } => {
                let all = [parent_full.as_slice(), hashes.as_slice()].concat();
                let e = Entry {
                    kind: "chained",
                    vsid,
                    parent: Some(parent),
                    parent_count: Some(parent_full.len()),
                    divergence: None,
                    diverged_from: None,
                    title,
                    stored_items: split.items.clone(),
                };
                (e, all)
            }
            Placement::Entry(e) => (e, hashes.clone()),
        };
        let full_chain = if entry.kind == "chained" { chain_hash(&full_hashes) } else { chain };
        let title = entry.title.clone().or_else(|| normalize::title(protocol, &split.items));
        let system_h = self.blob(&split.system)?;
        let tools_h = self.blob(&split.tools)?;
        let settings_h = self.blob(&Some(split.settings.clone()))?;
        let mut items = json!(entry.stored_items);
        truncated |= cap_strings(&mut items);
        truncated |= cap_strings(&mut response);
        let mut items = serde_json::to_vec(&items)?;
        let mut response_bytes = if response.is_null() { None } else { Some(serde_json::to_vec(&response)?) };
        if items.len() + response_bytes.as_ref().map_or(0, Vec::len) > ENTRY_CAP {
            truncated = true;
            items = serde_json::to_vec(&json!([{"$truncated":{"bytes":items.len()}}]))?;
            if response_bytes.as_ref().is_some_and(|b| b.len() > ENTRY_CAP / 2) {
                response_bytes = Some(serde_json::to_vec(&json!({"$truncated":{"bytes":response_bytes.as_ref().map_or(0, Vec::len)}}))?);
            }
        }
        let meta_bytes = serde_json::to_vec(meta)?;
        let bytes = items.len() + response_bytes.as_ref().map_or(0, Vec::len) + meta_bytes.len();
        let hash_blob = full_hashes.iter().flatten().copied().collect::<Vec<u8>>();
        let tx = self.c.unchecked_transaction()?;
        let pending = self.record_media(&redactor.media, settings.log_media)?;
        tx.execute(
            "INSERT OR REPLACE INTO entries VALUES(?1,?2,?3,?4,?5,?6,?7,?8,?9,?10,?11,?12,?13,?14,?15,?16,?17,?18,?19,?20,?21,?22,?23,?24,?25,?26,?27,?28,?29,?30)",
            params![
                meta.rid, meta.ts_ms, protocol, entry.vsid, entry.kind, entry.parent,
                entry.parent_count.map(|n| n as i64), entry.divergence.map(|n| n as i64), entry.diverged_from,
                full_hashes.len() as i64, full_chain, hash_blob, system_h, tools_h, settings_h,
                items, response_bytes, normalize::response_id(protocol, &response),
                meta.client_kind, meta.model_served.as_ref().or(meta.model_requested.as_ref()),
                meta.session_id, meta.session_source, meta.bench, meta.status, meta.outcome, title,
                meta_bytes, bytes as i64, truncated, scope_of(meta)
            ],
        )?;
        let mut refs = vec![];
        normalize::media_refs(&serde_json::from_slice(&items)?, &mut refs);
        normalize::media_refs(&response, &mut refs);
        for m in refs {
            if let Some(h) = m["sha256"].as_str() {
                tx.execute("INSERT OR IGNORE INTO entry_media VALUES(?1,?2)", params![meta.rid, h])?;
            }
        }
        tx.commit()?;
        self.write_media(&pending);
        Ok(true)
    }

    /// Where a request goes: the entry it extends, or a new chain position.
    fn place(
        &self,
        meta: &cuteafd_api::usage::Record,
        protocol: &str,
        split: &Split,
        hashes: &[[u8; 8]],
    ) -> Result<Placement> {
        // Chains never cross clients, API keys or sessions (no session matches only no session),
        // and never extend a truncated entry, whose elided items cannot be rebuilt.
        let scope = scope_of(meta);
        // 1. Responses `previous_response_id`: the parent's history plus its output.
        if let Some(previous) = &split.previous {
            let row = self
                .c
                .query_row(
                    "SELECT rid, vsid, title, response FROM entries WHERE response_id=?1 AND scope=?2 AND truncated=0 ORDER BY ts_ms DESC LIMIT 1",
                    [previous, &scope],
                    |r| Ok((r.get::<_, String>(0)?, r.get::<_, String>(1)?, r.get::<_, Option<String>>(2)?, r.get::<_, Option<Vec<u8>>>(3)?)),
                )
                .optional()?;
            if let Some((parent, vsid, title, response)) = row {
                let mut parent_full = view::hashes(&self.c, &parent)?;
                let response = response.map(|b| fold::parse(&b)).unwrap_or(Value::Null);
                parent_full.extend(normalize::response_items(protocol, &response).iter().map(item_hash));
                return Ok(Placement::Chained { parent, parent_full, vsid, title });
            }
        }
        // 2. The longest earlier entry whose whole history is a prefix of this one.
        let mut prefixes = Vec::with_capacity(hashes.len());
        let mut h = Sha256::new();
        for x in hashes {
            h.update(x);
            prefixes.push(normalize::hex16(&h.clone().finalize()));
        }
        let mut best: Option<(usize, String, String, Option<String>)> = None;
        for chunk in prefixes.chunks(400) {
            let marks = vec!["?"; chunk.len()].join(",");
            let mut stmt = self.c.prepare_cached(&format!(
                "SELECT rid, vsid, n_items, session_id, title FROM entries WHERE protocol=? AND scope=? AND truncated=0 AND chain IN ({marks}) ORDER BY ts_ms DESC"
            ))?;
            let args = [protocol.to_owned(), scope.clone()].into_iter().chain(chunk.iter().cloned());
            let rows = stmt.query_map(params_from_iter(args), |r| {
                Ok((r.get::<_, String>(0)?, r.get::<_, String>(1)?, r.get::<_, i64>(2)? as usize, r.get::<_, Option<String>>(3)?, r.get::<_, Option<String>>(4)?))
            })?;
            for row in rows {
                let (rid, vsid, n, _session, title) = row?;
                let better = match &best {
                    None => true,
                    Some((m, ..)) => n > *m,
                };
                if better {
                    best = Some((n, rid, vsid, title));
                }
            }
        }
        if let Some((n, rid, vsid, title)) = best {
            // A request that does not extend the session's latest entry rewound or
            // edited it: store the full history, marked where it diverged.
            let latest: Option<(String, Vec<u8>)> = self
                .c
                .query_row(
                    "SELECT rid, hashes FROM entries WHERE vsid=?1 ORDER BY ts_ms DESC, rid DESC LIMIT 1",
                    [&vsid],
                    |r| Ok((r.get(0)?, r.get(1)?)),
                )
                .optional()?;
            if let Some((latest, blob)) = latest.filter(|(l, _)| *l != rid) {
                let common = common_prefix(&blob, hashes);
                if common < blob.len() / 8 {
                    return Ok(Placement::Entry(Entry {
                        kind: "edit",
                        vsid,
                        parent: None,
                        parent_count: None,
                        divergence: Some(common),
                        diverged_from: Some(latest),
                        title,
                        stored_items: split.items.clone(),
                    }));
                }
            }
            return Ok(Placement::Entry(Entry {
                kind: "append",
                vsid,
                parent: Some(rid),
                parent_count: Some(n),
                divergence: None,
                diverged_from: None,
                title,
                stored_items: split.items[n..].to_vec(),
            }));
        }
        // 3. Edited, truncated or spliced history: the closest recent entry by common prefix.
        let (sql, arg) = ("SELECT rid, vsid, hashes, title FROM entries WHERE protocol=?1 AND scope=?2 ORDER BY ts_ms DESC LIMIT 64", scope.clone());
        let need = if meta.session_id.is_some() { 1 } else { 2 };
        let mut stmt = self.c.prepare_cached(sql)?;
        let mut close: Option<(usize, String, String, Option<String>)> = None;
        let rows = stmt.query_map(params![protocol, arg], |r| {
            Ok((r.get::<_, String>(0)?, r.get::<_, String>(1)?, r.get::<_, Vec<u8>>(2)?, r.get::<_, Option<String>>(3)?))
        })?;
        for row in rows {
            let (rid, vsid, blob, title) = row?;
            let common = common_prefix(&blob, hashes);
            if common >= need && close.as_ref().is_none_or(|(c, ..)| common > *c) {
                close = Some((common, rid, vsid, title));
            }
        }
        if let Some((common, rid, vsid, title)) = close {
            return Ok(Placement::Entry(Entry {
                kind: "edit",
                vsid,
                parent: None,
                parent_count: None,
                divergence: Some(common),
                diverged_from: Some(rid),
                title,
                stored_items: split.items.clone(),
            }));
        }
        Ok(Placement::Entry(Entry {
            kind: "base",
            vsid: meta.rid.clone(),
            parent: None,
            parent_count: None,
            divergence: None,
            diverged_from: None,
            title: None,
            stored_items: split.items.clone(),
        }))
    }

    fn clear(&mut self) -> Result<()> {
        self.c.execute_batch("BEGIN; DELETE FROM entries; DELETE FROM entry_media; DELETE FROM blobs; DELETE FROM media; COMMIT;")?;
        if let Some(dir) = &self.dir {
            let media = dir.join("media");
            if media.exists() {
                std::fs::remove_dir_all(&media)?;
            }
        }
        vacuum(&self.c)
    }

    /// Rewrites each retained child of a doomed entry with its full history,
    /// so it becomes the base of what is left of its chain.
    fn rebase_children(&self, doomed: &str) -> Result<usize> {
        let children = {
            let mut stmt = self.c.prepare(&format!(
                "SELECT rid FROM entries WHERE parent_rid IN (SELECT rid FROM entries WHERE {doomed}) AND NOT ({doomed})"
            ))?;
            let rows = stmt.query_map([], |r| r.get::<_, String>(0))?;
            rows.collect::<std::result::Result<Vec<_>, _>>()?
        };
        for rid in &children {
            let items = view::full_items(&self.c, rid)?;
            let mut refs = vec![];
            normalize::media_refs(&json!(items), &mut refs);
            let items = serde_json::to_vec(&items)?;
            self.c.execute(
                "UPDATE entries SET kind='rebased', parent_rid=NULL, items=?2, bytes=bytes-length(items)+length(?2) WHERE rid=?1",
                params![rid, items],
            )?;
            for m in refs {
                if let Some(h) = m["sha256"].as_str() {
                    self.c.execute("INSERT OR IGNORE INTO entry_media VALUES(?1,?2)", params![rid, h])?;
                }
            }
        }
        Ok(children.len())
    }

    fn delete(&self, doomed: &str) -> Result<usize> {
        let tx = self.c.unchecked_transaction()?;
        self.rebase_children(doomed)?;
        tx.execute(&format!("DELETE FROM entry_media WHERE rid IN (SELECT rid FROM entries WHERE {doomed})"), [])?;
        let n = tx.execute(&format!("DELETE FROM entries WHERE {doomed}"), [])?;
        tx.commit()?;
        Ok(n)
    }

    fn prune_media(&self) -> Result<()> {
        let orphans = {
            let mut stmt = self.c.prepare(
                "SELECT sha256, ext, stored FROM media WHERE sha256 NOT IN (SELECT sha256 FROM entry_media)",
            )?;
            let rows = stmt.query_map([], |r| Ok((r.get::<_, String>(0)?, r.get::<_, String>(1)?, r.get::<_, bool>(2)?)))?;
            rows.collect::<std::result::Result<Vec<_>, _>>()?
        };
        for (sha, ext, stored) in orphans {
            if stored {
                if let Some(dir) = &self.dir {
                    let path = media_path(dir, &sha, &ext);
                    if let Err(e) = std::fs::remove_file(&path) {
                        if e.kind() != std::io::ErrorKind::NotFound {
                            tracing::error!(error = %e, "usage media prune failed");
                            continue;
                        }
                    }
                }
            }
            self.c.execute("DELETE FROM media WHERE sha256=?1", [&sha])?;
        }
        self.sweep_media()?;
        self.c.execute(
            "DELETE FROM blobs WHERE h NOT IN (SELECT system_h FROM entries WHERE system_h IS NOT NULL
             UNION SELECT tools_h FROM entries WHERE tools_h IS NOT NULL UNION SELECT settings_h FROM entries WHERE settings_h IS NOT NULL)",
            [],
        )?;
        Ok(())
    }

    fn used(&self) -> Result<u64> {
        let media: i64 = self.c.query_row("SELECT coalesce(sum(bytes),0) FROM media WHERE stored=1", [], |r| r.get(0))?;
        Ok(db_used(&self.c)? + media as u64)
    }

    fn prune(&mut self, now: i64) -> Result<()> {
        let s = self.settings.load();
        let cutoff = if s.log_hours == 0 || s.log_cap_mb == 0 {
            i64::MAX
        } else {
            now.saturating_sub(i64::from(s.log_hours) * 3_600_000)
        };
        self.delete(&format!("ts_ms < {cutoff}"))?;
        self.prune_media()?;
        let cap = u64::from(s.log_cap_mb) * 1_048_576;
        loop {
            if self.used()? <= cap {
                break;
            }
            let n = self.delete(
                "rid IN (SELECT rid FROM entries ORDER BY ts_ms LIMIT max(1,(SELECT count(*)/10 FROM entries)))",
            )?;
            self.prune_media()?;
            if n == 0 {
                break;
            }
        }
        vacuum(&self.c)
    }
}

enum Placement {
    Chained { parent: String, parent_full: Vec<[u8; 8]>, vsid: String, title: Option<String> },
    Entry(Entry),
}

/// Client kind, API key label and session: entries chain only within one scope.
fn scope_of(meta: &cuteafd_api::usage::Record) -> String {
    normalize::content_hash(&json!([meta.client_kind, meta.key_label, meta.session_id]))
}

fn has_long_string(v: &Value) -> bool {
    match v {
        Value::String(s) => s.len() > STRING_CAP,
        Value::Array(a) => a.iter().any(has_long_string),
        Value::Object(o) => o.values().any(has_long_string),
        _ => false,
    }
}

fn common_prefix(blob: &[u8], hashes: &[[u8; 8]]) -> usize {
    blob.chunks_exact(8).zip(hashes).take_while(|(a, b)| *a == b.as_slice()).count()
}

fn db_used(c: &Connection) -> Result<u64> {
    let pages: u64 = c.query_row("PRAGMA page_count", [], |r| r.get(0))?;
    let free: u64 = c.query_row("PRAGMA freelist_count", [], |r| r.get(0))?;
    let size: u64 = c.query_row("PRAGMA page_size", [], |r| r.get(0))?;
    Ok(pages.saturating_sub(free) * size)
}

/// Replaces strings past `STRING_CAP` with a marker; true when any changed.
fn cap_strings(v: &mut Value) -> bool {
    match v {
        Value::String(s) if s.len() > STRING_CAP => {
            let mut end = 4096;
            while !s.is_char_boundary(end) {
                end -= 1;
            }
            *v = json!({"$truncated":{"bytes":s.len(),"prefix":&s[..end]}});
            true
        }
        Value::Array(a) => a.iter_mut().fold(false, |t, v| cap_strings(v) | t),
        Value::Object(o) => o.values_mut().fold(false, |t, v| cap_strings(v) | t),
        _ => false,
    }
}

mod view;
