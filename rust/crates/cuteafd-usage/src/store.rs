use arc_swap::ArcSwap;
use cuteafd_api::usage::{Counters, Record, UsageSink};
use rusqlite::{params, types::Value as SqlValue, Connection};
use serde::{Deserialize, Serialize};
use std::{
    path::{Path, PathBuf},
    sync::{
        atomic::Ordering::Relaxed,
        mpsc::{self, Receiver, SyncSender, TrySendError},
        Arc, Mutex,
    },
    time::{Duration, SystemTime, UNIX_EPOCH},
};

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("usage storage: {0}")]
    Sql(#[from] rusqlite::Error),
    #[error("usage filesystem: {0}")]
    Io(#[from] std::io::Error),
    #[error("usage JSON: {0}")]
    Json(#[from] serde_json::Error),
    #[error("usage writer stopped")]
    Stopped,
    #[error("invalid usage settings: {0}")]
    Settings(&'static str),
}
pub type Result<T> = std::result::Result<T, Error>;
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(default, deny_unknown_fields)]
pub struct Settings {
    pub metadata_days: u32,
    pub metadata_cap_mb: u32,
    pub log_enabled: bool,
    pub log_hours: u32,
    pub log_cap_mb: u32,
    pub daily_days: u32,
    pub client_ip: bool,
    pub record_bench: bool,
}
impl Default for Settings {
    fn default() -> Self {
        Self {
            metadata_days: 7,
            metadata_cap_mb: 256,
            log_enabled: true,
            log_hours: 24,
            log_cap_mb: 1024,
            daily_days: 90,
            client_ip: false,
            record_bench: true,
        }
    }
}
impl Settings {
    pub fn validate(&self) -> Result<()> {
        if self.metadata_cap_mb == 0 || self.log_cap_mb == 0 {
            return Err(Error::Settings("size caps must be positive"));
        }
        if self.metadata_days > 3650 || self.log_hours > 87600 || self.daily_days > 3650 {
            return Err(Error::Settings("retention exceeds ten years"));
        }
        Ok(())
    }
}
pub trait Clock: Send + Sync + 'static {
    fn now_ms(&self) -> i64;
}
pub struct SystemClock;
impl Clock for SystemClock {
    fn now_ms(&self) -> i64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as i64
    }
}
enum Command {
    Record(Record),
    Flush(mpsc::Sender<Result<()>>),
    Settings(Settings, mpsc::Sender<Result<()>>),
    Prune(mpsc::Sender<Result<()>>),
    Clear(mpsc::Sender<Result<()>>),
}
pub struct Store {
    pub log: Arc<crate::log::LogStore>,
    tx: SyncSender<Command>,
    pub(crate) settings: Arc<ArcSwap<Settings>>,
    pub(crate) counters: Arc<Counters>,
    pub(crate) reader: Mutex<Connection>,
    pub(crate) clock: Arc<dyn Clock>,
    pub(crate) path: Option<PathBuf>,
}
impl Store {
    pub fn open(directory: Option<&Path>) -> Result<Arc<Self>> {
        Self::open_with(directory, Arc::new(SystemClock), 4096)
    }
    pub fn open_with(
        directory: Option<&Path>,
        clock: Arc<dyn Clock>,
        capacity: usize,
    ) -> Result<Arc<Self>> {
        let path = directory.map(|p| p.join("usage.sqlite"));
        if let Some(dir) = directory {
            std::fs::create_dir_all(dir)?;
        }
        let uri = path
            .as_ref()
            .map(|p| p.to_string_lossy().into_owned())
            .unwrap_or_else(|| {
                format!(
                    "file:usage-{}?mode=memory&cache=shared",
                    uuid::Uuid::new_v4()
                )
            });
        let connection = Connection::open(&uri)?;
        schema(&connection)?;
        let saved: Option<String> = connection
            .query_row("SELECT value FROM settings WHERE key='settings'", [], |r| {
                r.get(0)
            })
            .optional()?;
        let settings = Arc::new(ArcSwap::from_pointee(
            saved
                .map(|s| serde_json::from_str::<Settings>(&s))
                .transpose()?
                .unwrap_or_default(),
        ));
        settings.load().validate()?;
        let reader = if path.is_some() {
            Connection::open_with_flags(&uri, rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY)?
        } else {
            Connection::open(&uri)?
        };
        reader.busy_timeout(Duration::from_millis(2000))?;
        let counters = Arc::new(Counters::default());
        let (tx, rx) = mpsc::sync_channel(capacity);
        let log = crate::log::LogStore::open(
            directory,
            settings.clone(),
            counters.clone(),
            clock.clone(),
        )?;
        let store = Arc::new(Self {
            log,
            tx,
            settings: settings.clone(),
            counters: counters.clone(),
            reader: Mutex::new(reader),
            clock: clock.clone(),
            path: path.clone(),
        });
        std::thread::Builder::new()
            .name("usage-writer".into())
            .spawn(move || writer(connection, rx, settings, counters, clock, path))?;
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
    pub fn prune(&self) -> Result<()> {
        self.command(Command::Prune)
    }
    pub fn clear(&self) -> Result<()> {
        self.command(Command::Clear)
    }
    pub fn settings(&self) -> Settings {
        (**self.settings.load()).clone()
    }
    pub fn update_settings(&self, settings: Settings) -> Result<()> {
        settings.validate()?;
        let enabled = settings.log_enabled && settings.log_hours > 0;
        self.command(|tx| Command::Settings(settings, tx))?;
        self.log.enabled.store(enabled, Relaxed);
        Ok(())
    }
    pub fn rows(&self) -> Result<Vec<Record>> {
        read_records(&*self.reader.lock().map_err(|_| Error::Stopped)?)
    }
}
use rusqlite::OptionalExtension;
impl UsageSink for Store {
    fn record(&self, record: Record) {
        let settings = self.settings.load();
        if settings.metadata_days == 0 || (!settings.record_bench && record.bench) {
            return;
        }
        match self.tx.try_send(Command::Record(record)) {
            Ok(()) => {}
            Err(TrySendError::Full(_)) | Err(TrySendError::Disconnected(_)) => {
                self.counters.dropped.fetch_add(1, Relaxed);
            }
        }
    }
    fn counters(&self) -> &Counters {
        &self.counters
    }
    fn client_ip(&self) -> bool {
        self.settings.load().client_ip
    }
}
fn schema(c: &Connection) -> Result<()> {
    c.busy_timeout(Duration::from_millis(2000))?;
    c.execute_batch("PRAGMA auto_vacuum=INCREMENTAL; PRAGMA journal_mode=WAL; PRAGMA synchronous=NORMAL;
CREATE TABLE IF NOT EXISTS requests (
id INTEGER PRIMARY KEY, rid TEXT NOT NULL UNIQUE, ts_ms INTEGER NOT NULL,
protocol TEXT NOT NULL, route TEXT NOT NULL, method TEXT NOT NULL, client_kind TEXT NOT NULL,
client_ua TEXT, key_label TEXT, client_ip TEXT, model_requested TEXT, model_served TEXT,
session_id TEXT, session_source TEXT, turn_index INTEGER, stream INTEGER NOT NULL,
n_items INTEGER, n_tools INTEGER, n_images INTEGER, n_audio INTEGER,
tokens_in INTEGER, tokens_cached INTEGER, tokens_out INTEGER, tokens_reasoning INTEGER,
draft_proposed INTEGER, draft_accepted INTEGER, rounds INTEGER,
t_queue_ms REAL, t_admit_ms REAL, t_ttft_ms REAL, t_total_ms REAL, prefill_tps REAL, decode_tps REAL,
concurrency_http INTEGER, concurrency_engine INTEGER, status INTEGER NOT NULL, outcome TEXT NOT NULL,
stop_reason TEXT, error_class TEXT, bytes_in INTEGER, bytes_out INTEGER, bench INTEGER NOT NULL DEFAULT 0);
CREATE INDEX IF NOT EXISTS requests_ts ON requests(ts_ms);
CREATE INDEX IF NOT EXISTS requests_session ON requests(session_id,ts_ms) WHERE session_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS requests_model ON requests(model_served,ts_ms);
CREATE INDEX IF NOT EXISTS requests_client ON requests(client_kind,ts_ms);
CREATE TABLE IF NOT EXISTS sessions (session_id TEXT PRIMARY KEY, source TEXT, client_kind TEXT, model TEXT,
first_ms INTEGER, last_ms INTEGER, turns INTEGER, tokens_in INTEGER, tokens_cached INTEGER, tokens_out INTEGER, errors INTEGER);
CREATE TABLE IF NOT EXISTS daily (day TEXT, protocol TEXT, client_kind TEXT, model TEXT, requests INTEGER, errors INTEGER,
tokens_in INTEGER, tokens_cached INTEGER, tokens_out INTEGER, draft_proposed INTEGER, draft_accepted INTEGER,
ttft_hist TEXT, decode_hist TEXT, bench INTEGER NOT NULL DEFAULT 0, PRIMARY KEY(day,protocol,client_kind,model,bench));
CREATE TABLE IF NOT EXISTS settings (key TEXT PRIMARY KEY,value TEXT NOT NULL);")?;
    let has_bench = c
        .prepare("PRAGMA table_info(daily)")?
        .query_map([], |r| r.get::<_, String>(1))?
        .collect::<std::result::Result<Vec<_>, _>>()?
        .iter()
        .any(|name| name == "bench");
    if !has_bench {
        c.execute_batch("BEGIN; ALTER TABLE daily RENAME TO daily_old;
CREATE TABLE daily(day TEXT,protocol TEXT,client_kind TEXT,model TEXT,requests INTEGER,errors INTEGER,
tokens_in INTEGER,tokens_cached INTEGER,tokens_out INTEGER,draft_proposed INTEGER,draft_accepted INTEGER,
ttft_hist TEXT,decode_hist TEXT,bench INTEGER NOT NULL DEFAULT 0,PRIMARY KEY(day,protocol,client_kind,model,bench));
INSERT INTO daily SELECT *,0 FROM daily_old; DROP TABLE daily_old; COMMIT;")?;
    }
    Ok(())
}
fn writer(
    mut c: Connection,
    rx: Receiver<Command>,
    settings: Arc<ArcSwap<Settings>>,
    counters: Arc<Counters>,
    clock: Arc<dyn Clock>,
    path: Option<PathBuf>,
) {
    let mut pending = None;
    let mut last = std::time::Instant::now();
    loop {
        let cmd = match pending.take() {
            Some(cmd) => cmd,
            None => match rx.recv_timeout(Duration::from_millis(50)) {
                Ok(cmd) => cmd,
                Err(mpsc::RecvTimeoutError::Timeout) => {
                    if last.elapsed() >= Duration::from_secs(60) {
                        if let Err(e) = prune(&c, &settings.load(), clock.now_ms()) {
                            tracing::error!(error=%e, "usage prune failed");
                        }
                        update_bytes(&c, &counters, path.as_deref());
                        last = std::time::Instant::now();
                    }
                    continue;
                }
                Err(_) => break,
            },
        };
        match cmd {
            Command::Record(first) => {
                let mut batch = vec![first];
                while batch.len() < 256 {
                    match rx.try_recv() {
                        Ok(Command::Record(r)) => batch.push(r),
                        Ok(other) => {
                            pending = Some(other);
                            break;
                        }
                        Err(_) => break,
                    }
                }
                let result = (|| -> Result<()> {
                    let tx = c.transaction()?;
                    for r in &mut batch {
                        insert(&tx, r)?;
                    }
                    tx.commit()?;
                    Ok(())
                })();
                match result {
                    Ok(()) => {
                        counters.recorded.fetch_add(batch.len() as u64, Relaxed);
                    }
                    Err(e) => {
                        counters.dropped.fetch_add(batch.len() as u64, Relaxed);
                        tracing::error!(error=%e, "usage write failed");
                    }
                }
            }
            Command::Settings(s, reply) => {
                let result = (|| -> Result<()> {
                    c.execute("INSERT INTO settings VALUES('settings',?1) ON CONFLICT(key) DO UPDATE SET value=excluded.value", [serde_json::to_string(&s)?])?;
                    settings.store(Arc::new(s));
                    Ok(())
                })();
                let _ = reply.send(result);
            }
            Command::Flush(reply) => {
                update_bytes(&c, &counters, path.as_deref());
                let _ = reply.send(Ok(()));
            }
            Command::Prune(reply) => {
                let result = prune(&c, &settings.load(), clock.now_ms());
                update_bytes(&c, &counters, path.as_deref());
                let _ = reply.send(result);
            }
            Command::Clear(reply) => {
                let result = c.execute_batch("BEGIN; DELETE FROM requests; DELETE FROM sessions; DELETE FROM daily; COMMIT; PRAGMA incremental_vacuum;").map_err(Error::from);
                let _ = reply.send(result);
            }
        }
        if last.elapsed() >= Duration::from_secs(60) {
            if let Err(e) = prune(&c, &settings.load(), clock.now_ms()) {
                tracing::error!(error=%e, "usage prune failed");
            }
            update_bytes(&c, &counters, path.as_deref());
            last = std::time::Instant::now();
        }
    }
}
fn insert(c: &Connection, r: &mut Record) -> Result<()> {
    if let Some(id) = &r.session_id {
        let turns: u64 = c.query_row(
            "SELECT count(*) FROM requests WHERE session_id=?1",
            [id],
            |row| row.get(0),
        )?;
        r.turn_index = Some(turns + 1);
    }
    let value = serde_json::to_value(&*r)?;
    let fields = value.as_object().expect("record object");
    let columns = fields
        .keys()
        .map(String::as_str)
        .collect::<Vec<_>>()
        .join(",");
    let placeholders = vec!["?"; fields.len()].join(",");
    let values = fields.values().map(sql_value).collect::<Vec<_>>();
    c.execute(
        &format!("INSERT INTO requests ({columns}) VALUES ({placeholders})"),
        rusqlite::params_from_iter(values),
    )?;
    if let Some(id) = &r.session_id {
        c.execute("INSERT INTO sessions VALUES(?1,?2,?3,?4,?5,?5,1,?6,?7,?8,?9)
ON CONFLICT(session_id) DO UPDATE SET first_ms=min(first_ms,excluded.first_ms),last_ms=max(last_ms,excluded.last_ms),
turns=turns+1,tokens_in=tokens_in+excluded.tokens_in,tokens_cached=tokens_cached+excluded.tokens_cached,
tokens_out=tokens_out+excluded.tokens_out,errors=errors+excluded.errors",
            params![id, r.session_source, r.client_kind, r.model_served, r.ts_ms, r.tokens_in.unwrap_or(0), r.tokens_cached.unwrap_or(0), r.tokens_out.unwrap_or(0), u64::from(r.outcome != "ok")])?;
    }
    Ok(())
}
fn sql_value(v: &serde_json::Value) -> SqlValue {
    match v {
        serde_json::Value::Null => SqlValue::Null,
        serde_json::Value::Bool(b) => SqlValue::Integer(i64::from(*b)),
        serde_json::Value::Number(n) => n
            .as_i64()
            .map(SqlValue::Integer)
            .unwrap_or_else(|| SqlValue::Real(n.as_f64().unwrap_or(0.))),
        serde_json::Value::String(s) => SqlValue::Text(s.clone()),
        _ => SqlValue::Null,
    }
}
pub(crate) fn read_records(c: &Connection) -> Result<Vec<Record>> {
    read_records_query(c, "SELECT * FROM requests ORDER BY ts_ms,id", vec![])
}
pub(crate) fn read_records_query(
    c: &Connection,
    sql: &str,
    args: Vec<SqlValue>,
) -> Result<Vec<Record>> {
    let mut stmt = c.prepare(sql)?;
    let columns = stmt
        .column_names()
        .iter()
        .map(|s| s.to_string())
        .collect::<Vec<_>>();
    let rows = stmt.query_map(rusqlite::params_from_iter(args), |row| {
        let mut object = serde_json::Map::new();
        for (i, name) in columns.iter().enumerate().skip(1) {
            let v: SqlValue = row.get(i)?;
            let value = match v {
                SqlValue::Null => serde_json::Value::Null,
                SqlValue::Integer(n) if name == "stream" || name == "bench" => {
                    serde_json::json!(n != 0)
                }
                SqlValue::Integer(n) => serde_json::json!(n),
                SqlValue::Real(n) => serde_json::json!(n),
                SqlValue::Text(s) => serde_json::json!(s),
                SqlValue::Blob(_) => serde_json::Value::Null,
            };
            object.insert(name.clone(), value);
        }
        Ok(serde_json::Value::Object(object))
    })?;
    rows.map(|row| Ok(serde_json::from_value(row?)?)).collect()
}
fn prune(c: &Connection, s: &Settings, now: i64) -> Result<()> {
    let cutoff = now.saturating_sub(i64::from(s.metadata_days) * 86400000);
    let tx = c.unchecked_transaction()?;
    let cutoff = if s.metadata_days == 0 {
        i64::MAX
    } else {
        cutoff
    };
    crate::daily::roll(&tx, s, "ts_ms < ?", vec![SqlValue::Integer(cutoff)])?;
    tx.execute("DELETE FROM requests WHERE ts_ms < ?1", [cutoff])?;
    tx.commit()?;
    loop {
        let pages: u64 = c.query_row("PRAGMA page_count", [], |r| r.get(0))?;
        let free: u64 = c.query_row("PRAGMA freelist_count", [], |r| r.get(0))?;
        let size: u64 = c.query_row("PRAGMA page_size", [], |r| r.get(0))?;
        if (pages - free) * size <= u64::from(s.metadata_cap_mb) * 1048576 {
            break;
        }
        let tx = c.unchecked_transaction()?;
        let selection="id IN (SELECT id FROM requests ORDER BY ts_ms,id LIMIT max(1,(SELECT count(*)/10 FROM requests)))";
        crate::daily::roll(&tx, s, selection, vec![])?;
        let n = tx.execute(&format!("DELETE FROM requests WHERE {selection}"), [])?;
        tx.commit()?;
        if n == 0 {
            break;
        }
    }
    c.execute("DELETE FROM sessions WHERE session_id NOT IN (SELECT session_id FROM requests WHERE session_id IS NOT NULL)", [])?;
    crate::daily::expire(c, s, now)?;
    vacuum(c)?;
    Ok(())
}
pub(crate) fn vacuum(c: &Connection) -> Result<()> {
    loop {
        let free: u64 = c.query_row("PRAGMA freelist_count", [], |r| r.get(0))?;
        if free == 0 {
            break;
        }
        c.execute_batch("PRAGMA incremental_vacuum(256);")?;
        let after: u64 = c.query_row("PRAGMA freelist_count", [], |r| r.get(0))?;
        if after >= free {
            break;
        }
    }
    c.execute_batch("PRAGMA wal_checkpoint(PASSIVE);")?;
    Ok(())
}
fn update_bytes(c: &Connection, counters: &Counters, path: Option<&Path>) {
    let bytes = path
        .and_then(|p| std::fs::metadata(p).ok())
        .map(|m| m.len())
        .unwrap_or_else(|| {
            let pages: u64 = c
                .query_row("PRAGMA page_count", [], |r| r.get(0))
                .unwrap_or(0);
            let size: u64 = c
                .query_row("PRAGMA page_size", [], |r| r.get(0))
                .unwrap_or(0);
            pages * size
        });
    counters.db_bytes.store(bytes, Relaxed);
}
#[cfg(test)]
mod tests {
    use super::*;
    struct Fixed(std::sync::atomic::AtomicI64);
    impl Clock for Fixed {
        fn now_ms(&self) -> i64 {
            self.0.load(Relaxed)
        }
    }
    fn record(i: usize, ts: i64) -> Record {
        Record {
            rid: format!("r{i}"),
            ts_ms: ts,
            protocol: "chat".into(),
            route: "/v1/chat/completions".into(),
            method: "POST".into(),
            client_kind: "curl".into(),
            outcome: "ok".into(),
            status: 200,
            ..Record::default()
        }
    }
    #[test]
    fn settings_roundtrip_and_age() {
        let dir = tempfile::tempdir().unwrap();
        let clock = Arc::new(Fixed(std::sync::atomic::AtomicI64::new(10 * 86400000)));
        let store = Store::open_with(Some(dir.path()), clock, 4096).unwrap();
        store.record(record(0, 0));
        store.record(record(1, 9 * 86400000));
        store.flush().unwrap();
        store.prune().unwrap();
        assert_eq!(store.rows().unwrap().len(), 1);
        let mut s = store.settings();
        s.metadata_days = 2;
        store.update_settings(s.clone()).unwrap();
        let other = Store::open(Some(dir.path())).unwrap();
        assert_eq!(other.settings(), s);
    }
    #[test]
    fn overflow_drops_without_blocking() {
        let store = Store::open_with(None, Arc::new(SystemClock), 1).unwrap();
        for i in 0..10000 {
            store.record(record(i, 0));
        }
        store.flush().unwrap();
        assert!(store.counters.dropped.load(Relaxed) > 0);
        assert_eq!(
            store.counters.recorded.load(Relaxed) + store.counters.dropped.load(Relaxed),
            10000
        );
    }
    #[test]
    fn size_cap_vacuums_with_injected_clock() {
        let clock = Arc::new(Fixed(std::sync::atomic::AtomicI64::new(0)));
        let store = Store::open_with(None, clock, 4096).unwrap();
        let mut s = store.settings();
        s.metadata_cap_mb = 1;
        store.update_settings(s).unwrap();
        for i in 0..1000 {
            let mut r = record(i, 0);
            r.client_ua = Some("x".repeat(4096));
            store.record(r);
        }
        store.flush().unwrap();
        let before: u64 = store
            .reader
            .lock()
            .unwrap()
            .query_row("PRAGMA page_count", [], |r| r.get(0))
            .unwrap();
        store.prune().unwrap();
        let c = store.reader.lock().unwrap();
        let after: u64 = c.query_row("PRAGMA page_count", [], |r| r.get(0)).unwrap();
        assert!(after < before);
        assert!(after * 4096 <= 1048576);
    }
    #[tokio::test]
    async fn metadata_privacy_sentinel_and_unlock_query() {
        use tower::ServiceExt;
        let dir = tempfile::tempdir().unwrap();
        let store = Store::open(Some(dir.path())).unwrap();
        let app = axum::Router::new()
            .route("/console/unlock", axum::routing::post(|| async { "ok" }))
            .layer(axum::middleware::from_fn_with_state(
                cuteafd_api::usage::Middleware::new(store.clone()),
                cuteafd_api::usage::track,
            ));
        let response = app
            .oneshot(
                axum::http::Request::post("/console/unlock?token=PRIVATE_SECRET_SENTINEL")
                    .header("Authorization", "Bearer PRIVATE_KEY_SENTINEL")
                    .body(axum::body::Body::from("PRIVATE_PAYLOAD_SENTINEL"))
                    .unwrap(),
            )
            .await
            .unwrap();
        axum::body::to_bytes(response.into_body(), 4096)
            .await
            .unwrap();
        store.flush().unwrap();
        store.prune().unwrap();
        for entry in std::fs::read_dir(dir.path()).unwrap() {
            let bytes = std::fs::read(entry.unwrap().path()).unwrap();
            let text = String::from_utf8_lossy(&bytes);
            for sentinel in [
                "PRIVATE_SECRET_SENTINEL",
                "PRIVATE_KEY_SENTINEL",
                "PRIVATE_PAYLOAD_SENTINEL",
            ] {
                assert!(!text.contains(sentinel));
            }
        }
    }
}
