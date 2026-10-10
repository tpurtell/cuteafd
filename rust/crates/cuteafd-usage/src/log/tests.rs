use super::*;
use crate::Store;
use bytes::Bytes;
use cuteafd_api::usage::Record;
use cuteafd_api::usage_log::ResponsePayload;
use std::sync::atomic::AtomicI64;

struct Fixed(AtomicI64);
impl Clock for Fixed {
    fn now_ms(&self) -> i64 {
        self.0.load(Relaxed)
    }
}
const HOUR: i64 = 3_600_000;

fn store(dir: Option<&Path>) -> (Arc<Store>, Arc<Fixed>) {
    let clock = Arc::new(Fixed(AtomicI64::new(1000 * HOUR)));
    (Store::open_with(dir, clock.clone(), 4096).unwrap(), clock)
}

fn meta(rid: &str, ts: i64, protocol: &str, session: Option<&str>) -> Record {
    Record {
        rid: rid.into(),
        ts_ms: ts,
        protocol: protocol.into(),
        client_kind: "claude_code".into(),
        session_id: session.map(Into::into),
        session_source: session.map(|_| "cache_key".into()),
        status: 200,
        outcome: "ok".into(),
        ..Default::default()
    }
}

fn send(log: &LogStore, meta: Record, request: Value, response: Value) {
    log.record_log(LogRecord {
        meta,
        request: vec![Bytes::from(serde_json::to_vec(&request).unwrap())],
        request_value: None, request_truncated: false,
        response: ResponsePayload::Object(Bytes::from(serde_json::to_vec(&response).unwrap())),
    });
}

fn user(t: &str) -> Value {
    json!({"role":"user","content":t})
}
fn assistant(t: &str) -> Value {
    json!({"role":"assistant","content":[{"type":"text","text":t}]})
}
fn messages(history: &[Value]) -> Value {
    json!({"model":"m","max_tokens":10,"system":"you are helpful","tools":[{"name":"Bash","input_schema":{}}],"messages":history,"metadata":{"user_id":"x"}})
}
fn reply(t: &str) -> Value {
    json!({"type":"message","role":"assistant","content":[{"type":"text","text":t}],"stop_reason":"end_turn","usage":{"input_tokens":10,"output_tokens":2}})
}
fn entry(c: &Connection, rid: &str) -> (String, Option<String>, Option<i64>, Option<i64>, Vec<Value>) {
    c.query_row("SELECT kind, parent_rid, parent_count, divergence, items FROM entries WHERE rid=?1", [rid], |r| {
        Ok((r.get(0)?, r.get(1)?, r.get(2)?, r.get(3)?, fold::parse(&r.get::<_, Vec<u8>>(4)?).as_array().cloned().unwrap()))
    })
    .unwrap()
}

/// A Claude Code-like session: append, append, edit (a rewritten turn), truncate (rewind).
#[test]
fn delta_chaining_append_edit_truncate() {
    let (store, clock) = store(None);
    let log = &store.log;
    let t = clock.now_ms() - 100;
    let s = Some("sess");
    let h1 = vec![user("first")];
    send(log, meta("r1", t, "messages", s), messages(&h1), reply("one"));
    let h2 = [h1.clone(), vec![assistant("one"), user("second")]].concat();
    send(log, meta("r2", t + 1, "messages", s), messages(&h2), reply("two"));
    let h3 = [h2.clone(), vec![assistant("two"), user("third")]].concat();
    send(log, meta("r3", t + 2, "messages", s), messages(&h3), reply("three"));
    // Edit: the second user message is rewritten.
    let edited = vec![user("first"), assistant("one"), user("second, edited")];
    send(log, meta("r4", t + 3, "messages", s), messages(&edited), reply("two'"));
    // Truncate: rewound to after the first answer, then a new message.
    let truncated = vec![user("first"), assistant("one"), user("other")];
    send(log, meta("r5", t + 4, "messages", s), messages(&truncated), reply("x"));
    // A strict prefix (a resend of the first turn) chains as an append of zero new items.
    send(log, meta("r6", t + 5, "messages", s), messages(&h1), reply("one again"));
    log.flush().unwrap();
    let c = log.reader.lock().unwrap();
    let (kind, parent, count, _, items) = entry(&c, "r1");
    assert_eq!((kind.as_str(), parent, count, items.len()), ("base", None, None, 1));
    let (kind, parent, count, _, items) = entry(&c, "r2");
    assert_eq!((kind.as_str(), parent.as_deref(), count, items.len()), ("append", Some("r1"), Some(1), 2));
    assert_eq!(items[1]["content"], "second");
    let (kind, parent, count, _, items) = entry(&c, "r3");
    assert_eq!((kind.as_str(), parent.as_deref(), count, items.len()), ("append", Some("r2"), Some(3), 2));
    let (kind, parent, _, divergence, items) = entry(&c, "r4");
    assert_eq!((kind.as_str(), parent, divergence, items.len()), ("edit", None, Some(2), 3));
    let (kind, _, _, divergence, items) = entry(&c, "r5");
    assert_eq!((kind.as_str(), divergence, items.len()), ("edit", Some(2), 3));
    // A rewind to the first turn is a truncation of the latest entry: full history, marked.
    let (kind, parent, _, divergence, items) = entry(&c, "r6");
    assert_eq!((kind.as_str(), parent, divergence, items.len()), ("edit", None, Some(1), 1));
    // Continuing after the edit appends to it.
    drop(c);
    send(log, meta("r7", t + 6, "messages", s), messages(&[h1.clone(), vec![assistant("one again"), user("go on")]].concat()), reply("y"));
    log.flush().unwrap();
    let c = log.reader.lock().unwrap();
    let (kind, parent, count, _, _) = entry(&c, "r7");
    assert_eq!((kind.as_str(), parent.as_deref(), count), ("append", Some("r6"), Some(1)));
    // System prompt and tools were stored once.
    let blobs: i64 = c.query_row("SELECT count(*) FROM blobs", [], |r| r.get(0)).unwrap();
    assert_eq!(blobs, 3, "system, tools, settings stored once each");
    drop(c);
    // The rebuilt request equals what the client sent.
    let full = log.get("r3").unwrap().unwrap();
    assert_eq!(full["request"], messages(&h3));
    assert_eq!(log.get("r4").unwrap().unwrap()["request"], messages(&edited));
    // One virtual session; the viewer shows only the new turns per entry.
    let session = log.session("r1").unwrap().unwrap();
    let entries = session["entries"].as_array().unwrap();
    assert_eq!(entries.len(), 7);
    let r3 = &entries[2];
    assert_eq!(r3["items"].as_array().unwrap().len(), 2);
    assert_eq!(r3["items"][0]["echo"], true, "the repeated assistant turn is an echo");
    assert_eq!(r3["items"][1]["text"], "third");
    assert_eq!(r3["system"]["changed"], false);
    assert_eq!(entries[0]["system"]["text"], "you are helpful");
    assert_eq!(entries[3]["kind"], "edit");
    assert_eq!(entries[3]["divergence"], 2);
    let list = log.sessions(&crate::query::Filter::default()).unwrap();
    assert_eq!(list["sessions"].as_array().unwrap().len(), 1);
    assert_eq!(list["sessions"][0]["entries"], 7);
    assert_eq!(list["sessions"][0]["edits"], 3);
    assert_eq!(list["sessions"][0]["title"], "first");
}

#[test]
fn unrelated_sessions_and_protocols_do_not_chain() {
    let (store, clock) = store(None);
    let log = &store.log;
    let t = clock.now_ms();
    send(log, meta("a1", t, "messages", Some("a")), messages(&[user("hello")]), reply("hi"));
    send(log, meta("b1", t + 1, "messages", Some("b")), messages(&[user("hello"), assistant("hi"), user("again")]), reply("hi"));
    send(log, meta("c1", t + 2, "chat", Some("a")), json!({"model":"m","messages":[user("hello")]}), json!({"choices":[]}));
    log.flush().unwrap();
    let c = log.reader.lock().unwrap();
    assert_eq!(entry(&c, "b1").0, "base");
    assert_eq!(entry(&c, "c1").0, "base");
}

#[test]
fn responses_previous_response_id_chains() {
    let (store, clock) = store(None);
    let log = &store.log;
    let t = clock.now_ms();
    let out = |id: &str, text: &str| json!({"id":id,"object":"response","status":"completed","output":[{"type":"message","role":"assistant","content":[{"type":"output_text","text":text}]}]});
    send(log, meta("q1", t, "responses", None), json!({"model":"m","input":"hi"}), out("resp_1", "hello"));
    send(log, meta("q2", t + 2 * HOUR, "responses", None), json!({"model":"m","previous_response_id":"resp_1","input":[{"type":"message","role":"user","content":"more"}]}), out("resp_2", "sure"));
    log.flush().unwrap();
    let c = log.reader.lock().unwrap();
    let (kind, parent, count, _, items) = entry(&c, "q2");
    assert_eq!((kind.as_str(), parent.as_deref(), count, items.len()), ("chained", Some("q1"), Some(2), 1));
    drop(c);
    let full = log.get("q2").unwrap().unwrap()["request"]["input"].clone();
    assert_eq!(full.as_array().unwrap().len(), 3, "input, inherited output, new input");
    assert_eq!(log.session("q1").unwrap().unwrap()["entries"].as_array().unwrap().len(), 2);
    // Expiring the parent rebases the child with the inherited output as history.
    clock.0.store(t + 25 * HOUR, Relaxed);
    log.prune().unwrap();
    let c = log.reader.lock().unwrap();
    let (kind, parent, _, _, items) = entry(&c, "q2");
    assert_eq!((kind.as_str(), parent, items.len()), ("rebased", None, 3));
    drop(c);
    assert_eq!(log.get("q2").unwrap().unwrap()["request"]["input"], full);
}

/// When a chain's base expires, the oldest retained entry is rewritten with
/// its full history and becomes the new base.
#[test]
fn expiry_rebases_oldest_retained_entry() {
    let (store, clock) = store(None);
    let log = &store.log;
    let t = clock.now_ms();
    let s = Some("sess");
    let mut history = vec![];
    for i in 0..4 {
        history.push(user(&format!("q{i}")));
        send(log, meta(&format!("e{i}"), t + i as i64 * HOUR, "messages", s), messages(&history), reply(&format!("a{i}")));
        history.push(assistant(&format!("a{i}")));
    }
    log.flush().unwrap();
    let before = log.get("e3").unwrap().unwrap()["request"].clone();
    // e0 and e1 fall out of the 24 h window.
    clock.0.store(t + 25 * HOUR + HOUR / 2, Relaxed);
    log.prune().unwrap();
    let c = log.reader.lock().unwrap();
    let n: i64 = c.query_row("SELECT count(*) FROM entries", [], |r| r.get(0)).unwrap();
    assert_eq!(n, 2);
    let (kind, parent, _, _, items) = entry(&c, "e2");
    assert_eq!((kind.as_str(), parent, items.len()), ("rebased", None, 5));
    let (kind, parent, ..) = entry(&c, "e3");
    assert_eq!((kind.as_str(), parent.as_deref()), ("append", Some("e2")));
    drop(c);
    assert_eq!(log.get("e3").unwrap().unwrap()["request"], before, "no child is orphaned");
    assert_eq!(log.session("e0").unwrap().unwrap()["entries"].as_array().unwrap().len(), 2);
}

fn png(seed: u8) -> String {
    use base64::Engine;
    format!("data:image/png;base64,{}", base64::engine::general_purpose::STANDARD.encode(vec![seed; 3000]))
}
fn image_request(images: &[String]) -> Value {
    let content = images.iter().map(|u| json!({"type":"image_url","image_url":{"url":u}})).collect::<Vec<_>>();
    json!({"model":"m","messages":[{"role":"user","content":content}]})
}
fn media_files(dir: &Path) -> Vec<PathBuf> {
    let mut out = vec![];
    if let Ok(top) = std::fs::read_dir(dir.join("media")) {
        for d in top.flatten() {
            for f in std::fs::read_dir(d.path()).unwrap().flatten() {
                out.push(f.path());
            }
        }
    }
    out
}

#[test]
fn media_dedupe_prune_and_reference_only() {
    let dir = tempfile::tempdir().unwrap();
    let (store, clock) = store(Some(dir.path()));
    let log = &store.log;
    let t = clock.now_ms();
    send(log, meta("m1", t, "chat", None), image_request(&[png(1), png(1)]), json!({}));
    send(log, meta("m2", t + HOUR, "chat", None), image_request(&[png(1), png(2)]), json!({}));
    log.flush().unwrap();
    let files = media_files(dir.path());
    assert_eq!(files.len(), 2, "one file per distinct image");
    let sha = format!("{:x}", Sha256::digest(vec![1u8; 3000]));
    let path = dir.path().join("media").join(&sha[..2]).join(format!("{sha}.png"));
    assert_eq!(std::fs::read(&path).unwrap(), vec![1u8; 3000]);
    assert_eq!(log.media(&sha).unwrap().unwrap().0, "image/png");
    assert!(log.media("../../etc/passwd").unwrap().is_none());
    // The entry references media by hash; no base64 survives in the database.
    let db = std::fs::read(dir.path().join("usage-log.sqlite")).unwrap();
    assert!(!String::from_utf8_lossy(&db).contains("data:image"));
    // The viewer exposes stored media through the cookie-gated route.
    let session = log.session("m1").unwrap().unwrap();
    assert_eq!(session["entries"][0]["items"][0]["media"][0]["url"], format!("/console/usage/media/{sha}"));
    // m1 expires: image 1 is still referenced by m2 and stays.
    clock.0.store(t + 24 * HOUR + HOUR / 2, Relaxed);
    log.prune().unwrap();
    assert_eq!(media_files(dir.path()).len(), 2);
    // m2 expires: both files go.
    clock.0.store(t + 26 * HOUR, Relaxed);
    log.prune().unwrap();
    assert!(media_files(dir.path()).is_empty());
    assert_eq!(store.counters.media_files.load(Relaxed), 0);
    // log_media off stores references only.
    let mut s = store.settings();
    s.log_media = false;
    store.update_settings(s).unwrap();
    send(log, meta("m3", clock.now_ms(), "chat", None), image_request(&[png(3)]), json!({}));
    log.flush().unwrap();
    assert!(media_files(dir.path()).is_empty());
    let e = log.session("m3").unwrap().unwrap();
    assert_eq!(e["entries"][0]["items"][0]["media"][0]["stored"], false);
    assert_eq!(e["entries"][0]["items"][0]["media"][0]["bytes"], 3000);
}

#[test]
fn media_bytes_count_toward_cap_and_zero_cap_disables() {
    let dir = tempfile::tempdir().unwrap();
    let (store, clock) = store(Some(dir.path()));
    let log = &store.log;
    let mut s = store.settings();
    s.log_cap_mb = 1;
    store.update_settings(s).unwrap();
    let t = clock.now_ms();
    use base64::Engine;
    for i in 0..8u8 {
        let url = format!("data:image/png;base64,{}", base64::engine::general_purpose::STANDARD.encode(vec![i; 300_000]));
        send(log, meta(&format!("big{i}"), t + i64::from(i), "chat", None), image_request(&[url]), json!({}));
    }
    log.flush().unwrap();
    assert_eq!(media_files(dir.path()).len(), 8);
    log.prune().unwrap();
    let files = media_files(dir.path());
    let media: u64 = files.iter().map(|f| std::fs::metadata(f).unwrap().len()).sum();
    assert!(media <= 1 << 20, "media {media} bytes");
    assert!(!files.is_empty() && files.len() < 8);
    let c = log.reader.lock().unwrap();
    assert!(db_used(&c).unwrap() + media <= 1 << 20);
    drop(c);
    // A zero cap turns the full log off.
    let mut s = store.settings();
    s.log_cap_mb = 0;
    store.update_settings(s).unwrap();
    assert!(!LogSink::enabled(&**log));
    log.prune().unwrap();
    assert!(media_files(dir.path()).is_empty());
}

#[test]
fn privacy_no_credentials_or_data_uris_on_disk() {
    let dir = tempfile::tempdir().unwrap();
    let (store, clock) = store(Some(dir.path()));
    let log = &store.log;
    send(log, meta("p1", clock.now_ms(), "messages", None),
        json!({"headers":{"Authorization":"Bearer RAW_KEY_SENTINEL"},"api_key":"RAW_KEY_SENTINEL","x-api-key":"RAW_KEY_SENTINEL",
            "messages":[{"role":"user","content":[{"type":"image","source":{"type":"base64","media_type":"image/png","data":"aGVsbG8="}},
            {"type":"text","text":"PROMPT_TEXT"}]}]}),
        reply("ok"));
    log.flush().unwrap();
    store.flush().unwrap();
    for path in walk(dir.path()) {
        let bytes = std::fs::read(&path).unwrap();
        let s = String::from_utf8_lossy(&bytes);
        assert!(!s.contains("RAW_KEY_SENTINEL"), "{}", path.display());
        assert!(!s.contains("aGVsbG8="), "{}", path.display());
    }
    // The prompt itself is payload and is kept in the log file only.
    let meta_file = std::fs::read(dir.path().join("usage.sqlite")).unwrap();
    assert!(!String::from_utf8_lossy(&meta_file).contains("PROMPT_TEXT"));
}

fn walk(dir: &Path) -> Vec<PathBuf> {
    let mut out = vec![];
    for e in std::fs::read_dir(dir).unwrap().flatten() {
        if e.path().is_dir() {
            out.extend(walk(&e.path()));
        } else {
            out.push(e.path());
        }
    }
    out
}

#[test]
fn clear_and_disable_leave_metadata_untouched() {
    let dir = tempfile::tempdir().unwrap();
    let (store, clock) = store(Some(dir.path()));
    use cuteafd_api::usage::UsageSink;
    store.record(meta("kept", clock.now_ms(), "chat", None));
    store.flush().unwrap();
    send(&store.log, meta("x", clock.now_ms(), "chat", None), image_request(&[png(9)]), json!({}));
    store.log.flush().unwrap();
    assert_eq!(media_files(dir.path()).len(), 1);
    store.log.clear().unwrap();
    assert!(store.log.get("x").unwrap().is_none());
    assert!(media_files(dir.path()).is_empty());
    assert_eq!(store.rows().unwrap().len(), 1);
    let mut s = store.settings();
    s.log_enabled = false;
    store.update_settings(s).unwrap();
    store.flush().unwrap();
    let before = Sha256::digest(std::fs::read(dir.path().join("usage.sqlite")).unwrap());
    send(&store.log, meta("y", clock.now_ms(), "chat", None), json!({"messages":[]}), json!({}));
    store.log.flush().unwrap();
    assert!(store.log.get("y").unwrap().is_none());
    assert_eq!(before, Sha256::digest(std::fs::read(dir.path().join("usage.sqlite")).unwrap()));
}

#[test]
fn oversized_strings_are_truncated_with_marker() {
    let (store, clock) = store(None);
    send(&store.log, meta("huge", clock.now_ms(), "chat", None),
        json!({"messages":[user(&"word ".repeat(STRING_CAP))]}), json!({}));
    store.log.flush().unwrap();
    let v = store.log.get("huge").unwrap().unwrap();
    assert_eq!(v["truncated"], true);
    assert!(v["request"]["messages"][0]["content"]["$truncated"]["bytes"].as_u64().unwrap() > STRING_CAP as u64);
}

/// Claude Code's observed wire shape: the latest user turn is a block list, and
/// the next request resends it as a plain string, with a cache marker moving.
#[test]
fn claude_code_respelled_history_still_appends() {
    let (store, clock) = store(None);
    let log = &store.log;
    let t = clock.now_ms() - 100;
    let s = Some("cc");
    let first = vec![
        json!({"role":"user","content":[{"type":"text","text":"<system-reminder>ctx</system-reminder>"},{"type":"text","text":"Fix calc.py"}]}),
        json!({"role":"system","content":[{"type":"text","text":"# Environment","cache_control":{"type":"ephemeral"}}]}),
    ];
    send(log, meta("c1", t, "messages", s), messages(&first), reply("reading"));
    let second = vec![
        json!({"role":"user","content":[{"type":"text","text":"<system-reminder>ctx</system-reminder>"},{"type":"text","text":"Fix calc.py"}]}),
        json!({"role":"system","content":"# Environment"}),
        json!({"role":"assistant","content":[{"type":"tool_use","id":"t1","name":"Read","input":{"file_path":"calc.py"}}]}),
        json!({"role":"user","content":[{"type":"tool_result","tool_use_id":"t1","content":"def add(a,b): return a-b","cache_control":{"type":"ephemeral"}}]}),
    ];
    send(log, meta("c2", t + 1, "messages", s), messages(&second), reply("fixed"));
    log.flush().unwrap();
    let c = log.reader.lock().unwrap();
    let (kind, parent, count, _, items) = entry(&c, "c2");
    assert_eq!((kind.as_str(), parent.as_deref(), count, items.len()), ("append", Some("c1"), Some(2), 2));
    drop(c);
    assert_eq!(log.sessions(&crate::query::Filter::default()).unwrap()["sessions"][0]["title"], "Fix calc.py");
    // The stored delta rebuilds to what the client sent (its own spelling of new items).
    assert_eq!(log.get("c2").unwrap().unwrap()["request"]["messages"][3], second[3]);
}

/// M1: a turn with a huge tool result is stored truncated and never becomes a
/// parent, so every later turn still rebuilds exactly.
#[test]
fn truncated_entries_never_become_parents() {
    let (store, clock) = store(None);
    let log = &store.log;
    let t = clock.now_ms() - 100;
    let s = Some("big");
    let mut h = vec![user("start")];
    send(log, meta("t1", t, "messages", s), messages(&h), reply("ok"));
    h.push(assistant("ok"));
    h.push(json!({"role":"user","content":[{"type":"tool_result","tool_use_id":"x","content":"z".repeat(STRING_CAP + 10)}]}));
    send(log, meta("t2", t + 1, "messages", s), messages(&h), reply("read it"));
    h.push(assistant("read it"));
    h.push(user("next"));
    send(log, meta("t3", t + 2, "messages", s), messages(&h), reply("done"));
    log.flush().unwrap();
    // A short turn in the same session after the big one.
    send(log, meta("t4", t + 3, "messages", s), messages(&[user("start"), assistant("ok"), user("fresh")]), reply("y"));
    log.flush().unwrap();
    let c = log.reader.lock().unwrap();
    for rid in ["t2", "t3"] {
        let (kind, parent, ..) = entry(&c, rid);
        assert_eq!((kind.as_str(), parent), ("base", None), "{rid}: an oversized request is self-contained");
    }
    let parents: Vec<String> = c.prepare("SELECT parent_rid FROM entries WHERE parent_rid IS NOT NULL").unwrap()
        .query_map([], |r| r.get::<_, String>(0)).unwrap().map(|r| r.unwrap()).collect();
    assert!(!parents.iter().any(|p| p == "t2" || p == "t3"), "nothing chains onto a truncated entry");
    drop(c);
    // Every retained entry rebuilds to what the client sent, except the elided strings it marks.
    for (rid, sent) in [("t1", messages(&h[..1])), ("t4", messages(&[user("start"), assistant("ok"), user("fresh")]))] {
        assert_eq!(log.get(rid).unwrap().unwrap()["request"], sent, "{rid}");
    }
    let v = log.get("t3").unwrap().unwrap();
    assert_eq!(v["truncated"], true);
    assert_eq!(v["request"]["messages"].as_array().unwrap().len(), h.len(), "no items lost, only the long string elided");
    let session = log.session(v["vsid"].as_str().unwrap()).unwrap().unwrap();
    assert_eq!(session["entries"][0]["truncated"], true, "the viewer shows truncation");
}

/// L3: chains stay within one client, API key and session.
#[test]
fn chains_never_cross_clients_keys_or_sessions() {
    let (store, clock) = store(None);
    let log = &store.log;
    let t = clock.now_ms() - 100;
    let h1 = vec![user("same opening")];
    let h2 = [h1.clone(), vec![assistant("one"), user("more")]].concat();
    send(log, meta("s1", t, "messages", Some("a")), messages(&h1), reply("one"));
    send(log, meta("n1", t + 1, "messages", None), messages(&h2), reply("x"));
    let mut other_key = meta("k1", t + 2, "messages", Some("a"));
    other_key.key_label = Some("k:other".into());
    send(log, other_key, messages(&h2), reply("x"));
    let mut other_client = meta("c1", t + 3, "messages", Some("a"));
    other_client.client_kind = "codex".into();
    send(log, other_client, messages(&h2), reply("x"));
    send(log, meta("s2", t + 4, "messages", Some("a")), messages(&h2), reply("x"));
    log.flush().unwrap();
    let c = log.reader.lock().unwrap();
    for rid in ["n1", "k1", "c1"] {
        assert_eq!(entry(&c, rid).0, "base", "{rid}");
    }
    assert_eq!(entry(&c, "s2").1.as_deref(), Some("s1"));
}

/// M2: cleared and expired payloads leave no plaintext in the db or WAL.
#[test]
fn clear_and_expiry_leave_no_plaintext() {
    let dir = tempfile::tempdir().unwrap();
    let (store, clock) = store(Some(dir.path()));
    let log = &store.log;
    for i in 0..50 {
        send(log, meta(&format!("x{i}"), clock.now_ms(), "chat", None),
            json!({"model":"m","messages":[user(&format!("PLAINTEXT_SENTINEL_{i} {}", "pad ".repeat(200)))]}), json!({"choices":[]}));
    }
    log.flush().unwrap();
    let found = || walk(dir.path()).iter().any(|p| String::from_utf8_lossy(&std::fs::read(p).unwrap()).contains("PLAINTEXT_SENTINEL"));
    assert!(found(), "the sentinel is stored before clearing");
    log.clear().unwrap();
    assert!(!found(), "clear leaves no plaintext in any file");
    send(log, meta("old", clock.now_ms(), "chat", None), json!({"messages":[user("EXPIRY_SENTINEL")]}), json!({}));
    log.flush().unwrap();
    clock.0.store(clock.now_ms() + 25 * HOUR, Relaxed);
    log.prune().unwrap();
    for p in walk(dir.path()) {
        assert!(!String::from_utf8_lossy(&std::fs::read(&p).unwrap()).contains("EXPIRY_SENTINEL"), "{}", p.display());
    }
}

/// L2 and L10: private modes; media written only after commit, untracked files swept.
#[test]
fn private_modes_and_untracked_media_swept() {
    use std::os::unix::fs::PermissionsExt;
    let dir = tempfile::tempdir().unwrap();
    let usage = dir.path().join("usage");
    let (store, clock) = store(Some(&usage));
    send(&store.log, meta("m", clock.now_ms(), "chat", None), image_request(&[png(5)]), json!({}));
    store.log.flush().unwrap();
    let mode = |p: &Path| std::fs::metadata(p).unwrap().permissions().mode() & 0o777;
    assert_eq!(mode(&usage), 0o700);
    assert_eq!(mode(&usage.join("usage-log.sqlite")), 0o600);
    assert_eq!(mode(&usage.join("usage.sqlite")), 0o600);
    let files = media_files(&usage);
    assert_eq!(files.len(), 1);
    assert_eq!(mode(&files[0]), 0o600);
    assert_eq!(mode(files[0].parent().unwrap()), 0o700);
    // A file no committed row accounts for (e.g. a crash after write) is swept on prune.
    let stray = usage.join("media").join("ab").join(format!("{}.png", "ab".repeat(32)));
    std::fs::create_dir_all(stray.parent().unwrap()).unwrap();
    std::fs::write(&stray, b"x").unwrap();
    store.log.prune().unwrap();
    assert!(!stray.exists());
    assert_eq!(media_files(&usage).len(), 1, "tracked media stays");
}
