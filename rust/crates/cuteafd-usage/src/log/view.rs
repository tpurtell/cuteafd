//! Read side: rebuild requests by walking parents, list virtual sessions and
//! render each entry's new items for the conversation view.
use super::{fold, normalize};
use crate::{query::Filter, store::Result};
use rusqlite::{params_from_iter, types::Value as Sql, Connection, OptionalExtension};
use serde_json::{json, Value};

struct Row {
    rid: String,
    ts_ms: i64,
    protocol: String,
    vsid: String,
    kind: String,
    parent_rid: Option<String>,
    parent_count: Option<i64>,
    divergence: Option<i64>,
    diverged_from: Option<String>,
    n_items: i64,
    system_h: Option<String>,
    tools_h: Option<String>,
    settings_h: Option<String>,
    items: Vec<u8>,
    response: Option<Vec<u8>>,
    meta: Vec<u8>,
    bytes: i64,
    truncated: bool,
}

const COLUMNS: &str = "rid, ts_ms, protocol, vsid, kind, parent_rid, parent_count, divergence, diverged_from, n_items, system_h, tools_h, settings_h, items, response, meta, bytes, truncated";

fn row(r: &rusqlite::Row<'_>) -> rusqlite::Result<Row> {
    Ok(Row {
        rid: r.get(0)?,
        ts_ms: r.get(1)?,
        protocol: r.get(2)?,
        vsid: r.get(3)?,
        kind: r.get(4)?,
        parent_rid: r.get(5)?,
        parent_count: r.get(6)?,
        divergence: r.get(7)?,
        diverged_from: r.get(8)?,
        n_items: r.get(9)?,
        system_h: r.get(10)?,
        tools_h: r.get(11)?,
        settings_h: r.get(12)?,
        items: r.get(13)?,
        response: r.get(14)?,
        meta: r.get(15)?,
        bytes: r.get(16)?,
        truncated: r.get(17)?,
    })
}

fn load(c: &Connection, rid: &str) -> Result<Option<Row>> {
    Ok(c.query_row(&format!("SELECT {COLUMNS} FROM entries WHERE rid=?1"), [rid], row).optional()?)
}

pub(super) fn hashes(c: &Connection, rid: &str) -> Result<Vec<[u8; 8]>> {
    let blob: Vec<u8> = c.query_row("SELECT hashes FROM entries WHERE rid=?1", [rid], |r| r.get(0))?;
    Ok(blob.chunks_exact(8).map(|c| c.try_into().expect("8 bytes")).collect())
}

fn blob(c: &Connection, h: &Option<String>) -> Result<Option<Value>> {
    let Some(h) = h else { return Ok(None) };
    let bytes: Option<Vec<u8>> = c.query_row("SELECT value FROM blobs WHERE h=?1", [h], |r| r.get(0)).optional()?;
    Ok(bytes.map(|b| fold::parse(&b)))
}

fn items_of(row: &Row) -> Vec<Value> {
    fold::parse(&row.items).as_array().cloned().unwrap_or_default()
}

/// The full item list of an entry, walking parents (iteratively, any depth).
pub(super) fn full_items(c: &Connection, rid: &str) -> Result<Vec<Value>> {
    let mut stack = vec![];
    let mut next = Some(rid.to_owned());
    while let Some(rid) = next.take() {
        let Some(r) = load(c, &rid)? else { break };
        next = r.parent_rid.clone().filter(|_| matches!(r.kind.as_str(), "append" | "chained"));
        stack.push(r);
    }
    let mut items: Vec<Value> = vec![];
    while let Some(r) = stack.pop() {
        match r.kind.as_str() {
            "append" => {
                items.truncate(r.parent_count.unwrap_or(items.len() as i64) as usize);
                items.extend(items_of(&r));
            }
            "chained" => {
                items.extend(items_of(&r));
            }
            _ => items = items_of(&r),
        }
        // A chained child inherits this entry's output as history.
        if let Some(child) = stack.last() {
            if child.kind == "chained" {
                let response = r.response.as_deref().map(fold::parse).unwrap_or(Value::Null);
                items.extend(normalize::response_items(&r.protocol, &response));
            }
        }
    }
    Ok(items)
}

pub(super) fn entry_full(c: &Connection, rid: &str) -> Result<Option<Value>> {
    let Some(r) = load(c, rid)? else { return Ok(None) };
    let items = full_items(c, rid)?;
    let response = r.response.as_deref().map(fold::parse).unwrap_or(Value::Null);
    let request = normalize::join(
        &r.protocol,
        normalize::Split {
            system: blob(c, &r.system_h)?,
            tools: blob(c, &r.tools_h)?,
            settings: blob(c, &r.settings_h)?.unwrap_or(json!({})),
            items: items.clone(),
            previous: None,
        },
    );
    let mut v = summary(c, &r, None)?;
    v["request"] = request;
    v["display"] = json!({
        "items": normalize::display_items(&r.protocol, &items, 0),
        "response": normalize::display_response(&r.protocol, &response),
    });
    v["response"] = response;
    Ok(Some(v))
}

fn media_urls(c: &Connection, items: &mut [Value]) -> Result<()> {
    for d in items {
        if let Some(media) = d["media"].as_array_mut() {
            for m in media {
                let sha = m["sha256"].as_str().unwrap_or("").to_owned();
                let stored: bool = c
                    .query_row("SELECT stored FROM media WHERE sha256=?1", [&sha], |r| r.get(0))
                    .optional()?
                    .unwrap_or(false);
                m["stored"] = json!(stored);
                m["url"] = if stored { json!(format!("/console/usage/media/{sha}")) } else { Value::Null };
            }
        }
    }
    Ok(())
}

/// One entry for the conversation view; `previous` is the prior entry in the
/// same virtual session (for system/tools change detection and echo marking).
fn summary(c: &Connection, r: &Row, previous: Option<&Row>) -> Result<Value> {
    let meta: Value = fold::parse(&r.meta);
    let response = r.response.as_deref().map(fold::parse).unwrap_or(Value::Null);
    let start = match r.kind.as_str() {
        "append" => r.parent_count.unwrap_or(0) as usize,
        "chained" => (r.n_items as usize).saturating_sub(items_of(r).len()),
        _ => 0,
    };
    let own = items_of(r);
    let mut items = normalize::display_items(&r.protocol, &own, start);
    let mut response_display = normalize::display_response(&r.protocol, &response);
    // An appended turn begins with the assistant message the previous entry already showed.
    if let Some(p) = previous {
        let shown = p.response.as_deref().map(fold::parse).map(|v| normalize::display_response(&p.protocol, &v)).unwrap_or_default();
        for d in items.iter_mut().filter(|d| d["role"] == "assistant") {
            let same = shown.iter().any(|s| s["kind"] == d["kind"] && s["text"] == d["text"] && s["arguments"] == d["arguments"]);
            if same {
                d["echo"] = json!(true);
            }
        }
    }
    media_urls(c, &mut items)?;
    media_urls(c, &mut response_display)?;
    let system = blob(c, &r.system_h)?;
    let system_changed = previous.is_none_or(|p| p.system_h != r.system_h);
    let tools = blob(c, &r.tools_h)?;
    let tool_names = tools
        .as_ref()
        .and_then(Value::as_array)
        .map(|a| {
            a.iter()
                .filter_map(|t| t["name"].as_str().or_else(|| t["function"]["name"].as_str()).map(str::to_owned))
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    let system_text = system.as_ref().map(|s| match s {
        Value::String(s) => s.clone(),
        other => normalize::display_items("messages", &[json!({"role":"system","content":other})], 0)
            .iter()
            .filter_map(|d| d["text"].as_str().map(str::to_owned))
            .collect::<Vec<_>>()
            .join("\n"),
    });
    Ok(json!({
        "rid": r.rid, "ts_ms": r.ts_ms, "protocol": r.protocol, "vsid": r.vsid, "kind": r.kind,
        "parent_rid": r.parent_rid, "parent_count": r.parent_count, "divergence": r.divergence,
        "diverged_from": r.diverged_from, "n_items": r.n_items,
        "system": {"changed": system_changed, "text": if system_changed { system_text.map(Value::from).unwrap_or(Value::Null) } else { Value::Null }},
        "tools": {"changed": previous.is_none_or(|p| p.tools_h != r.tools_h), "count": tool_names.len(), "names": tool_names},
        "items": items, "response": response_display,
        "usage": normalize::usage(&response), "stop": normalize::stop(&r.protocol, &response),
        "truncated": r.truncated, "bytes": r.bytes,
        "status": meta["status"], "outcome": meta["outcome"],
        "model": meta["model_served"].as_str().or(meta["model_requested"].as_str()),
        "client": meta["client_kind"], "session_id": meta["session_id"], "session_source": meta["session_source"],
    }))
}

pub(super) fn session(c: &Connection, vsid: &str) -> Result<Option<Value>> {
    let rows = {
        let mut stmt = c.prepare(&format!("SELECT {COLUMNS} FROM entries WHERE vsid=?1 ORDER BY ts_ms, rid"))?;
        let rows = stmt.query_map([vsid], row)?;
        rows.collect::<std::result::Result<Vec<_>, _>>()?
    };
    if rows.is_empty() {
        return Ok(None);
    }
    let mut entries = vec![];
    for (i, r) in rows.iter().enumerate() {
        // Echo detection follows the chain: the parent when retained, else the previous entry.
        let previous = r
            .parent_rid
            .as_ref()
            .and_then(|p| rows[..i].iter().rev().find(|x| &x.rid == p))
            .or_else(|| i.checked_sub(1).map(|j| &rows[j]));
        entries.push(summary(c, r, previous)?);
    }
    Ok(Some(json!({"vsid": vsid, "entries": entries})))
}

pub(super) fn sessions(c: &Connection, f: &Filter, from: i64, to: i64) -> Result<Value> {
    let mut sql = String::from(
        "SELECT vsid, min(ts_ms), max(ts_ms), count(*), group_concat(DISTINCT protocol), max(client), max(model),
         max(session_id), max(session_source), sum(kind='edit'), sum(bytes),
         (SELECT title FROM entries t WHERE t.vsid=e.vsid AND title IS NOT NULL ORDER BY ts_ms LIMIT 1)
         FROM entries e WHERE ts_ms>=? AND ts_ms<=?",
    );
    let mut args = vec![Sql::Integer(from), Sql::Integer(to)];
    for (column, value) in [("client", &f.client), ("protocol", &f.protocol), ("model", &f.model), ("session_id", &f.session)] {
        if let Some(v) = value {
            sql.push_str(&format!(" AND {column}=?"));
            args.push(Sql::Text(v.clone()));
        }
    }
    if !f.bench.unwrap_or(false) {
        sql.push_str(" AND bench=0");
    }
    sql.push_str(" GROUP BY vsid");
    if let Some(cursor) = &f.cursor {
        let (ts, vsid) = cursor.split_once(':').ok_or(crate::store::Error::Settings("invalid cursor"))?;
        let ts: i64 = ts.parse().map_err(|_| crate::store::Error::Settings("invalid cursor"))?;
        sql.push_str(" HAVING max(ts_ms)<? OR (max(ts_ms)=? AND vsid<?)");
        args.extend([Sql::Integer(ts), Sql::Integer(ts), Sql::Text(vsid.into())]);
    }
    let limit = f.limit.unwrap_or(50).clamp(1, 500);
    sql.push_str(" ORDER BY max(ts_ms) DESC, vsid DESC LIMIT ?");
    args.push(Sql::Integer(limit as i64 + 1));
    let mut stmt = c.prepare(&sql)?;
    let rows = stmt.query_map(params_from_iter(args), |r| {
        let protocols: String = r.get::<_, Option<String>>(4)?.unwrap_or_default();
        Ok(json!({
            "vsid": r.get::<_, String>(0)?, "first_ms": r.get::<_, i64>(1)?, "last_ms": r.get::<_, i64>(2)?,
            "entries": r.get::<_, i64>(3)?, "protocols": protocols.split(',').filter(|s| !s.is_empty()).collect::<Vec<_>>(),
            "client": r.get::<_, Option<String>>(5)?, "model": r.get::<_, Option<String>>(6)?,
            "session_id": r.get::<_, Option<String>>(7)?, "session_source": r.get::<_, Option<String>>(8)?,
            "edits": r.get::<_, i64>(9)?, "bytes": r.get::<_, i64>(10)?, "title": r.get::<_, Option<String>>(11)?,
        }))
    })?;
    let mut sessions = rows.collect::<std::result::Result<Vec<_>, _>>()?;
    let more = sessions.len() > limit;
    sessions.truncate(limit);
    let next = more
        .then(|| sessions.last().map(|s| format!("{}:{}", s["last_ms"], s["vsid"].as_str().unwrap_or(""))))
        .flatten();
    Ok(json!({"sessions": sessions, "next_cursor": next}))
}
