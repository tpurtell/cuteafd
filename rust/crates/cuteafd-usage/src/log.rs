//! Payloads live in a separate database and bounded writer, never metadata.
use crate::store::{vacuum, Clock, Error, Result, Settings};
use arc_swap::ArcSwap;
use cuteafd_api::{
    usage::Counters,
    usage_log::{LogRecord, LogSink, ResponsePayload},
};
use rusqlite::{params, Connection, OptionalExtension};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{
    path::Path,
    sync::{
        atomic::{AtomicBool, AtomicUsize, Ordering::Relaxed},
        mpsc::{self, SyncSender},
        Arc, Mutex,
    },
    time::{Duration, Instant},
};
const INFLIGHT_BYTES: usize = 64 << 20;
const RECORD_CAP: usize = 1 << 20;
enum Command {
    Record(LogRecord, usize),
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
}
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
            .unwrap_or_else(|| {
                format!(
                    "file:usage-log-{}?mode=memory&cache=shared",
                    uuid::Uuid::new_v4()
                )
            });
        let c = Connection::open(&uri)?;
        c.busy_timeout(Duration::from_millis(2000))?;
        c.execute_batch("PRAGMA auto_vacuum=INCREMENTAL; PRAGMA journal_mode=WAL; PRAGMA synchronous=NORMAL;
CREATE TABLE IF NOT EXISTS log(rid TEXT PRIMARY KEY,ts_ms INTEGER,protocol TEXT,request BLOB,response BLOB,bytes INTEGER,truncated INTEGER);
CREATE INDEX IF NOT EXISTS log_ts ON log(ts_ms);")?;
        let reader = if path.is_some() {
            Connection::open_with_flags(&uri, rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY)?
        } else {
            Connection::open(&uri)?
        };
        reader.busy_timeout(Duration::from_millis(2000))?;
        let (tx, rx) = mpsc::sync_channel(1024);
        let bytes = Arc::new(AtomicUsize::new(0));
        let enabled = Arc::new(AtomicBool::new(
            settings.load().log_enabled && settings.load().log_hours > 0,
        ));
        let store = Arc::new(Self {
            tx,
            bytes: bytes.clone(),
            enabled,
            settings: settings.clone(),
            counters: counters.clone(),
            reader: Mutex::new(reader),
        });
        if store.enabled() {
            tracing::info!(hours=settings.load().log_hours, "usage full log on: prompts and model outputs stored in plain text for the retention period");
        }
        std::thread::Builder::new()
            .name("usage-log-writer".into())
            .spawn(move || {
                let mut last = Instant::now();
                loop {
                    match rx.recv_timeout(Duration::from_millis(50)) {
                        Ok(Command::Record(r, n)) => {
                            let result =
                                if settings.load().log_enabled && settings.load().log_hours > 0 {
                                    write(&c, &r)
                                } else {
                                    Ok(false)
                                };
                            bytes.fetch_sub(n, Relaxed);
                            match result {
                                Ok(true) => {
                                    counters.log_recorded.fetch_add(1, Relaxed);
                                }
                                Ok(false) => {}
                                Err(e) => {
                                    counters.log_dropped.fetch_add(1, Relaxed);
                                    tracing::error!(error=%e, "usage full log write failed");
                                }
                            }
                        }
                        Ok(Command::Flush(reply)) => {
                            let _ = reply.send(Ok(()));
                        }
                        Ok(Command::Clear(reply)) => {
                            let _ = reply.send(
                                c.execute("DELETE FROM log", [])
                                    .map_err(Error::from)
                                    .and_then(|_| vacuum(&c)),
                            );
                        }
                        Ok(Command::Prune(reply)) => {
                            let _ = reply.send(prune(&c, &settings.load(), clock.now_ms()));
                        }
                        Err(mpsc::RecvTimeoutError::Timeout) => {}
                        Err(_) => break,
                    }
                    if last.elapsed() >= Duration::from_secs(60) {
                        if let Err(e) = prune(&c, &settings.load(), clock.now_ms()) {
                            tracing::error!(error=%e, "usage full log prune failed");
                        }
                        last = Instant::now();
                    }
                    let pages: u64 = c
                        .query_row("PRAGMA page_count", [], |r| r.get(0))
                        .unwrap_or(0);
                    let size: u64 = c
                        .query_row("PRAGMA page_size", [], |r| r.get(0))
                        .unwrap_or(0);
                    counters.log_bytes.store(pages * size, Relaxed);
                }
            })?;
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
    pub fn clear(&self) -> Result<()> {
        self.command(Command::Clear)
    }
    pub fn prune(&self) -> Result<()> {
        self.command(Command::Prune)
    }
    pub fn get(&self, rid: &str) -> Result<Option<Value>> {
        self.reader.lock().map_err(|_| Error::Stopped)?.query_row("SELECT ts_ms,protocol,request,response,bytes,truncated FROM log WHERE rid=?1", [rid], |r| {
            let request: Vec<u8> = r.get(2)?; let response: Vec<u8> = r.get(3)?;
            Ok(json!({"rid":rid,"ts_ms":r.get::<_,i64>(0)?,"protocol":r.get::<_,String>(1)?,"request":serde_json::from_slice::<Value>(&request).unwrap_or(Value::Null),
                "response":serde_json::from_slice::<Value>(&response).unwrap_or(Value::Null),"bytes":r.get::<_,i64>(4)?,"truncated":r.get::<_,bool>(5)?}))
        }).optional().map_err(Error::from)
    }
}
impl LogSink for LogStore {
    fn enabled(&self) -> bool {
        self.enabled.load(Relaxed) && self.settings.load().log_hours > 0
    }
    fn record_log(&self, r: LogRecord) {
        if !self.enabled() {
            return;
        }
        // Console unlock tokens and any admin inputs never enter either database.
        if !matches!(
            r.protocol.as_str(),
            "chat" | "completions" | "messages" | "responses" | "realtime"
        ) {
            return;
        }
        let n = r.byte_len();
        if n > INFLIGHT_BYTES
            || self
                .bytes
                .fetch_update(Relaxed, Relaxed, |v| {
                    v.checked_add(n).filter(|total| *total <= INFLIGHT_BYTES)
                })
                .is_err()
        {
            self.counters.log_dropped.fetch_add(1, Relaxed);
            return;
        }
        if self.tx.try_send(Command::Record(r, n)).is_err() {
            self.bytes.fetch_sub(n, Relaxed);
            self.counters.log_dropped.fetch_add(1, Relaxed);
        }
    }
}
fn parse(bytes: &[u8]) -> Value {
    serde_json::from_slice(bytes)
        .unwrap_or_else(|_| json!({"truncated":true,"reason":"invalid JSON omitted"}))
}
fn write(c: &Connection, r: &LogRecord) -> Result<bool> {
    let mut request = parse(&r.request);
    redact(&mut request);
    let mut response = match &r.response {
        ResponsePayload::Object(b) => parse(b),
        ResponsePayload::Deltas(d) => fold(d),
    };
    redact(&mut response);
    let mut request = serde_json::to_vec(&request)?;
    let mut response = serde_json::to_vec(&response)?;
    let truncated = request.len() + response.len() > RECORD_CAP;
    if truncated {
        request = capped(&request, RECORD_CAP / 2)?;
        response = capped(&response, RECORD_CAP / 2)?;
    }
    c.execute(
        "INSERT OR REPLACE INTO log VALUES(?1,?2,?3,?4,?5,?6,?7)",
        params![
            r.rid,
            r.ts_ms,
            r.protocol,
            request,
            response,
            request.len() + response.len(),
            truncated
        ],
    )?;
    Ok(true)
}
fn capped(bytes: &[u8], cap: usize) -> Result<Vec<u8>> {
    if bytes.len() <= cap {
        return Ok(bytes.to_vec());
    }
    let prefix = String::from_utf8_lossy(&bytes[..cap / 8]);
    Ok(serde_json::to_vec(
        &json!({"truncated":true,"original_bytes":bytes.len(),"prefix":prefix}),
    )?)
}
fn reference(mime: &str, bytes: &[u8]) -> Value {
    json!({"type":"image_url","ref":{"mime":mime,"bytes":bytes.len(),"sha256":format!("{:x}",Sha256::digest(bytes))}})
}
fn credential(name: &str) -> bool {
    matches!(
        name.to_ascii_lowercase().as_str(),
        "headers"
            | "authorization"
            | "x-api-key"
            | "api_key"
            | "api-key"
            | "cookie"
            | "set-cookie"
            | "proxy-authorization"
            | "x-goog-api-key"
    )
}
fn redact(value: &mut Value) {
    match value {
        Value::String(s) if s.starts_with("data:") => {
            let (meta, body) = s[5..].split_once(',').unwrap_or(("", ""));
            let mime = meta.split(';').next().unwrap_or("application/octet-stream");
            let bytes = if meta.ends_with(";base64") {
                use base64::Engine;
                base64::engine::general_purpose::STANDARD
                    .decode(body)
                    .unwrap_or_default()
            } else {
                body.as_bytes().to_vec()
            };
            *value = reference(mime, &bytes);
        }
        Value::String(s)
            if s.len() > 65536
                && s.bytes()
                    .filter(|b| {
                        b.is_ascii_alphanumeric() || matches!(b, b'+' | b'/' | b'=' | b'-' | b'_')
                    })
                    .count()
                    * 100
                    / s.len()
                    >= 95 =>
        {
            use base64::Engine;
            let bytes = base64::engine::general_purpose::STANDARD
                .decode(&*s)
                .unwrap_or_else(|_| s.as_bytes().to_vec());
            *value = reference("application/octet-stream", &bytes);
        }
        Value::Array(a) => {
            a.retain(|v| {
                !v.as_array()
                    .and_then(|a| a.first())
                    .and_then(Value::as_str)
                    .is_some_and(credential)
            });
            for v in a {
                redact(v);
            }
        }
        Value::Object(o) => {
            // Some clients wrap a body with headers; never retain this wrapper.
            o.retain(|key, _| !credential(key));
            let event = o
                .get("type")
                .and_then(Value::as_str)
                .unwrap_or("")
                .to_owned();
            if event == "input_audio_buffer.append" || event.contains("audio.delta") {
                for field in ["audio", "delta"] {
                    if let Some(Value::String(data)) = o.remove(field) {
                        use base64::Engine;
                        let bytes = base64::engine::general_purpose::STANDARD
                            .decode(data)
                            .unwrap_or_default();
                        o.insert(field.into(), reference("audio/unknown", &bytes));
                    }
                }
            }
            if let Some(audio) = o.get("input_audio").and_then(Value::as_object) {
                use base64::Engine;
                let bytes = audio
                    .get("data")
                    .and_then(Value::as_str)
                    .and_then(|s| base64::engine::general_purpose::STANDARD.decode(s).ok())
                    .unwrap_or_default();
                let mime = format!(
                    "audio/{}",
                    audio
                        .get("format")
                        .and_then(Value::as_str)
                        .unwrap_or("unknown")
                );
                o.insert("input_audio".into(), reference(&mime, &bytes));
            }
            if o.get("type")
                .and_then(Value::as_str)
                .is_some_and(|s| matches!(s, "input_audio" | "audio"))
            {
                if let Some(Value::String(data)) = o.remove("data") {
                    use base64::Engine;
                    let bytes = base64::engine::general_purpose::STANDARD
                        .decode(data)
                        .unwrap_or_default();
                    o.insert(
                        "ref".into(),
                        reference("audio/unknown", &bytes)["ref"].clone(),
                    );
                }
            }
            if o.get("type").and_then(Value::as_str) == Some("base64") {
                if let Some(Value::String(data)) = o.remove("data") {
                    use base64::Engine;
                    let bytes = base64::engine::general_purpose::STANDARD
                        .decode(data)
                        .unwrap_or_default();
                    let mime = o
                        .get("media_type")
                        .and_then(Value::as_str)
                        .unwrap_or("application/octet-stream");
                    *value = reference(mime, &bytes);
                    return;
                }
            }
            for v in o.values_mut() {
                redact(v);
            }
        }
        _ => {}
    }
}
fn fold(deltas: &[bytes::Bytes]) -> Value {
    let mut result = json!({"object":"chat.completion","choices":[]});
    let mut choices = std::collections::BTreeMap::<u64, Value>::new();
    for bytes in deltas {
        let chunk = parse(bytes);
        if chunk["log_truncated"] == true {
            result["truncated"] = json!(true);
        }
        for field in ["id", "model", "created", "usage"] {
            if let Some(v) = chunk.get(field) {
                result[field] = v.clone();
            }
        }
        for choice in chunk["choices"].as_array().into_iter().flatten() {
            let index = choice["index"].as_u64().unwrap_or(0);
            let output = choices.entry(index).or_insert_with(|| json!({"index":index,"message":{"role":"assistant","content":""},"finish_reason":null}));
            let delta = &choice["delta"];
            for field in ["content", "reasoning_content", "reasoning"] {
                if let Some(text) = delta[field].as_str() {
                    let old = output["message"][field].as_str().unwrap_or("");
                    output["message"][field] = json!(format!("{old}{text}"));
                }
            }
            if let Some(reason) = choice.get("finish_reason").filter(|v| !v.is_null()) {
                output["finish_reason"] = reason.clone();
            }
            for tool in delta["tool_calls"].as_array().into_iter().flatten() {
                let index = tool["index"].as_u64().unwrap_or(0) as usize;
                if index > 1024 {
                    continue;
                }
                if !output["message"]["tool_calls"].is_array() {
                    output["message"]["tool_calls"] = json!([]);
                }
                let tools = output["message"]["tool_calls"].as_array_mut().unwrap();
                while tools.len() <= index {
                    tools.push(json!({"type":"function","function":{"name":"","arguments":""}}));
                }
                if let Some(id) = tool.get("id") {
                    tools[index]["id"] = id.clone();
                }
                for field in ["name", "arguments"] {
                    if let Some(text) = tool["function"][field].as_str() {
                        let old = tools[index]["function"][field].as_str().unwrap_or("");
                        tools[index]["function"][field] = json!(format!("{old}{text}"));
                    }
                }
            }
        }
    }
    result["choices"] = json!(choices.into_values().collect::<Vec<_>>());
    result
}
fn prune(c: &Connection, s: &Settings, now: i64) -> Result<()> {
    c.execute(
        "DELETE FROM log WHERE ts_ms < ?1",
        [if s.log_hours == 0 {
            i64::MAX
        } else {
            now.saturating_sub(i64::from(s.log_hours) * 3600000)
        }],
    )?;
    loop {
        let (pages, free, size): (u64, u64, u64) = (
            c.query_row("PRAGMA page_count", [], |r| r.get(0))?,
            c.query_row("PRAGMA freelist_count", [], |r| r.get(0))?,
            c.query_row("PRAGMA page_size", [], |r| r.get(0))?,
        );
        if (pages - free) * size <= u64::from(s.log_cap_mb) * 1048576 {
            break;
        }
        if c.execute("DELETE FROM log WHERE rid IN (SELECT rid FROM log ORDER BY ts_ms LIMIT max(1,(SELECT count(*)/10 FROM log)))", [])? == 0 { break; }
    }
    vacuum(c)
}
#[cfg(test)]
mod tests {
    use super::*;
    use cuteafd_api::usage::UsageSink;
    fn record(id: &str, ts: i64, request: Value) -> LogRecord {
        LogRecord {
            rid: id.into(),
            ts_ms: ts,
            protocol: "chat".into(),
            request: serde_json::to_vec(&request).unwrap().into(),
            response: ResponsePayload::Object(bytes::Bytes::from_static(b"{}")),
        }
    }
    #[test]
    fn redaction_truncation_disable_and_clear_isolated() {
        let dir = tempfile::tempdir().unwrap();
        let store = crate::Store::open(Some(dir.path())).unwrap();
        store.record(cuteafd_api::usage::Record {
            rid: "kept".into(),
            ts_ms: store.clock.now_ms(),
            ..Default::default()
        });
        store.flush().unwrap();
        store.prune().unwrap();
        let log = &store.log;
        log.record_log(record("redact",store.clock.now_ms(),json!({"headers":{"Authorization":"RAW_KEY_SENTINEL"},"api_key":"RAW_KEY_SENTINEL","image":"data:image/png;base64,aGVsbG8=","input_audio":{"data":"aGVsbG8=","format":"wav"}})));
        log.record_log(record("cases",store.clock.now_ms(),json!({"Authorization":"RAW_KEY_SENTINEL","X-Api-Key":"RAW_KEY_SENTINEL","Api-Key":"RAW_KEY_SENTINEL","COOKIE":"RAW_KEY_SENTINEL","Set-Cookie":"RAW_KEY_SENTINEL","Proxy-Authorization":"RAW_KEY_SENTINEL","x-goog-api-key":"RAW_KEY_SENTINEL","pairs":[["AUTHORIZATION","RAW_KEY_SENTINEL"]]})));
        log.record_log(record(
            "audio",
            store.clock.now_ms(),
            json!({"type":"input_audio_buffer.append","audio":"aGVsbG8="}),
        ));
        log.record_log(record(
            "unknown",
            store.clock.now_ms(),
            json!({"unknown_media":"YQ==".repeat(20000)}),
        ));
        log.record_log(record(
            "huge",
            store.clock.now_ms(),
            json!({"text":"text with spaces ".repeat(RECORD_CAP/4)}),
        ));
        log.flush().unwrap();
        assert!(log.get("audio").unwrap().unwrap()["request"]["audio"]["ref"].is_object());
        assert!(
            log.get("unknown").unwrap().unwrap()["request"]["unknown_media"]["ref"].is_object()
        );
        assert_eq!(
            log.get("cases").unwrap().unwrap()["request"]["pairs"],
            json!([])
        );
        let row = log.get("redact").unwrap().unwrap();
        assert!(row["request"]["image"]["ref"]["sha256"].is_string());
        assert_eq!(row["request"]["image"]["ref"]["bytes"], 5);
        assert!(log.get("huge").unwrap().unwrap()["truncated"]
            .as_bool()
            .unwrap());
        let mut s = store.settings();
        s.log_enabled = false;
        store.update_settings(s).unwrap();
        store.flush().unwrap();
        let before = Sha256::digest(std::fs::read(dir.path().join("usage.sqlite")).unwrap());
        let log_before =
            Sha256::digest(std::fs::read(dir.path().join("usage-log.sqlite")).unwrap());
        log.record_log(record(
            "disabled",
            store.clock.now_ms(),
            json!({"text":"not stored"}),
        ));
        log.flush().unwrap();
        assert_eq!(
            before,
            Sha256::digest(std::fs::read(dir.path().join("usage.sqlite")).unwrap())
        );
        assert_eq!(
            log_before,
            Sha256::digest(std::fs::read(dir.path().join("usage-log.sqlite")).unwrap())
        );
        assert!(log.get("disabled").unwrap().is_none());
        log.clear().unwrap();
        assert!(log.get("redact").unwrap().is_none());
        assert_eq!(store.rows().unwrap().len(), 1);
        for entry in std::fs::read_dir(dir.path()).unwrap() {
            let bytes = std::fs::read(entry.unwrap().path()).unwrap();
            let s = String::from_utf8_lossy(&bytes);
            assert!(!s.contains("RAW_KEY_SENTINEL"));
            assert!(!s.contains("data:image"));
        }
    }
    #[test]
    fn folds_text_reasoning_tools_and_usage() {
        let deltas=vec![bytes::Bytes::from_static(br#"{"choices":[{"index":0,"delta":{"content":"hi","reasoning_content":"think","tool_calls":[{"index":0,"id":"t","function":{"name":"run","arguments":"{"}}]}}]}"#),bytes::Bytes::from_static(br#"{"choices":[{"index":0,"delta":{"content":"!","tool_calls":[{"index":0,"function":{"arguments":"}"}}]},"finish_reason":"tool_calls"}],"usage":{"completion_tokens":2}}"#)];
        let v = fold(&deltas);
        assert_eq!(v["choices"][0]["message"]["content"], "hi!");
        assert_eq!(
            v["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"],
            "{}"
        );
        assert_eq!(v["usage"]["completion_tokens"], 2);
    }
    #[test]
    fn hours_and_size_cap() {
        struct Fixed;
        impl Clock for Fixed {
            fn now_ms(&self) -> i64 {
                100 * 3600000
            }
        }
        let store = crate::Store::open_with(None, Arc::new(Fixed), 4096).unwrap();
        let log = &store.log;
        log.record_log(record("old", 0, json!({"text":"old"})));
        log.record_log(record("new", 100 * 3600000, json!({"text":"new"})));
        log.flush().unwrap();
        log.prune().unwrap();
        assert!(log.get("old").unwrap().is_none());
        assert!(log.get("new").unwrap().is_some());
        let mut s = store.settings();
        s.log_cap_mb = 1;
        store.update_settings(s).unwrap();
        for i in 0..20 {
            log.record_log(record(
                &format!("big{i}"),
                100 * 3600000,
                json!({"text":"x".repeat(100000)}),
            ));
        }
        log.flush().unwrap();
        log.prune().unwrap();
        let c = log.reader.lock().unwrap();
        let pages: u64 = c.query_row("PRAGMA page_count", [], |r| r.get(0)).unwrap();
        assert!(pages * 4096 <= 1048576);
    }
}
