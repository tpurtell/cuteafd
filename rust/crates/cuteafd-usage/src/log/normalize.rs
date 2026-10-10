//! Protocol-specific views of a redacted request: the head (system prompt,
//! tools, settings) stored once by content hash, and the item list that delta
//! entries extend. `join` is the inverse of `split`.
use serde_json::{json, Map, Value};
use sha2::{Digest, Sha256};

pub(crate) struct Split {
    pub system: Option<Value>,
    pub tools: Option<Value>,
    pub settings: Value,
    pub items: Vec<Value>,
    /// Responses `previous_response_id`: the request continues a stored response.
    pub previous: Option<String>,
}

/// The field holding the item list, and the one holding the system prompt.
fn fields(protocol: &str) -> (&'static str, Option<&'static str>) {
    match protocol {
        "messages" => ("messages", Some("system")),
        "responses" => ("input", Some("instructions")),
        "realtime" => ("items", Some("instructions")),
        "completions" => ("prompt", None),
        _ => ("messages", None),
    }
}

pub(crate) fn split(protocol: &str, request: Value) -> Split {
    let (items_field, system_field) = fields(protocol);
    let mut o = match request {
        Value::Object(o) => o,
        other => {
            let mut o = Map::new();
            o.insert("body".into(), other);
            o
        }
    };
    let items = match o.remove(items_field) {
        Some(Value::Array(a)) => a,
        Some(Value::Null) | None => vec![],
        // A Responses string input or a completions prompt is one user item.
        Some(Value::String(s)) if protocol == "responses" => {
            vec![json!({"type":"message","role":"user","content":s})]
        }
        Some(other) => vec![json!({"$scalar": other})],
    };
    let system = system_field.and_then(|f| o.remove(f)).filter(|v| !v.is_null());
    let tools = o.remove("tools").filter(|v| !v.is_null());
    let previous = o
        .get("previous_response_id")
        .and_then(Value::as_str)
        .map(str::to_owned);
    Split { system, tools, settings: Value::Object(o), items, previous }
}

pub(crate) fn join(protocol: &str, split: Split) -> Value {
    let (items_field, system_field) = fields(protocol);
    let mut o = match split.settings {
        Value::Object(o) => o,
        _ => Map::new(),
    };
    if let (Some(f), Some(s)) = (system_field, split.system) {
        o.insert(f.into(), s);
    }
    if let Some(t) = split.tools {
        o.insert("tools".into(), t);
    }
    let items = if let [one] = split.items.as_slice() {
        one.get("$scalar").cloned().unwrap_or_else(|| json!([one]))
    } else {
        json!(split.items)
    };
    o.insert(items_field.into(), items);
    Value::Object(o)
}

/// Prompt-caching markers move between turns; they do not change content.
fn strip_volatile(v: &mut Value) {
    match v {
        Value::Object(o) => {
            o.remove("cache_control");
            for v in o.values_mut() {
                strip_volatile(v);
            }
        }
        Value::Array(a) => a.iter_mut().for_each(strip_volatile),
        _ => {}
    }
}

/// Equivalent spellings hash alike: clients alternate between a single text
/// block and a plain string for the same message (Claude Code does, turn to turn).
fn canonical(v: &mut Value) {
    if let Some(content) = v.get_mut("content") {
        let single = content.as_array().filter(|a| a.len() == 1).and_then(|a| a[0].as_object()).filter(|b| {
            b.len() == 2 && b.get("type").and_then(Value::as_str) == Some("text") && b.get("text").is_some_and(Value::is_string)
        }).map(|b| b["text"].clone());
        if let Some(text) = single {
            *content = text;
        }
    }
}

pub(crate) fn item_hash(item: &Value) -> [u8; 8] {
    let mut v = item.clone();
    strip_volatile(&mut v);
    canonical(&mut v);
    let digest = Sha256::digest(v.to_string().as_bytes());
    digest[..8].try_into().expect("8 bytes")
}

pub(crate) fn chain_hash(hashes: &[[u8; 8]]) -> String {
    let mut h = Sha256::new();
    for x in hashes {
        h.update(x);
    }
    hex16(&h.finalize())
}

pub(crate) fn hex16(bytes: &[u8]) -> String {
    bytes[..8].iter().map(|b| format!("{b:02x}")).collect()
}

pub(crate) fn content_hash(v: &Value) -> String {
    hex16(&Sha256::digest(v.to_string().as_bytes()))
}

/// Items a continuation (previous_response_id) inherits from its parent's response.
pub(crate) fn response_items(protocol: &str, response: &Value) -> Vec<Value> {
    match protocol {
        "responses" => response["output"].as_array().cloned().unwrap_or_default(),
        _ => vec![],
    }
}

pub(crate) fn response_id(protocol: &str, response: &Value) -> Option<String> {
    (protocol == "responses").then(|| response["id"].as_str().map(str::to_owned)).flatten()
}

// ---------------------------------------------------------------- display

fn text_of(v: &Value) -> Option<String> {
    match v {
        Value::String(s) => Some(s.clone()),
        Value::Array(a) => {
            let parts = a
                .iter()
                .filter_map(|p| {
                    p.get("text")
                        .or_else(|| p.get("transcript"))
                        .or_else(|| p.get("refusal"))
                        .and_then(Value::as_str)
                        .map(str::to_owned)
                        .or_else(|| p.as_str().map(str::to_owned))
                })
                .collect::<Vec<_>>();
            (!parts.is_empty()).then(|| parts.join("\n"))
        }
        Value::Null => None,
        other => Some(other.to_string()),
    }
}

pub(crate) fn media_refs(v: &Value, out: &mut Vec<Value>) {
    match v {
        Value::Object(o) => {
            if let Some(m) = o.get("$media") {
                if !out.iter().any(|x| x["sha256"] == m["sha256"]) {
                    out.push(m.clone());
                }
                return;
            }
            o.values().for_each(|v| media_refs(v, out));
        }
        Value::Array(a) => a.iter().for_each(|v| media_refs(v, out)),
        _ => {}
    }
}

fn display(role: &str, kind: &str, source: &Value) -> Value {
    let mut media = vec![];
    media_refs(source, &mut media);
    json!({"role":role,"kind":kind,"text":null,"name":null,"arguments":null,"call_id":null,"media":media,"echo":false})
}

fn role_of(v: &Value) -> &str {
    match v["role"].as_str().unwrap_or("user") {
        "developer" | "system" => "system",
        "assistant" | "model" => "assistant",
        "tool" | "function" => "tool",
        _ => "user",
    }
}

/// One wire item to display items (Anthropic content blocks split into several).
fn item_display(protocol: &str, item: &Value) -> Vec<Value> {
    if let Some(s) = item.get("$scalar") {
        let mut d = display("user", "text", s);
        d["text"] = json!(text_of(s));
        return vec![d];
    }
    match protocol {
        "messages" => {
            let role = role_of(item);
            match &item["content"] {
                Value::Array(blocks) => blocks
                    .iter()
                    .map(|b| {
                        let t = b["type"].as_str().unwrap_or("");
                        let mut d = match t {
                            "thinking" | "redacted_thinking" => {
                                let mut d = display(role, "reasoning", b);
                                d["text"] = json!(b["thinking"].as_str());
                                d
                            }
                            "tool_use" | "server_tool_use" => {
                                let mut d = display(role, "tool_call", b);
                                d["name"] = b["name"].clone();
                                d["call_id"] = b["id"].clone();
                                d["arguments"] = json!(b["input"].to_string());
                                d
                            }
                            "tool_result" | "web_search_tool_result" => {
                                let mut d = display("tool", "tool_result", b);
                                d["call_id"] = b["tool_use_id"].clone();
                                d["text"] = json!(text_of(&b["content"]));
                                d
                            }
                            "text" => {
                                let mut d = display(role, "text", b);
                                d["text"] = b["text"].clone();
                                d
                            }
                            _ => {
                                let mut d = display(role, "other", b);
                                d["name"] = json!(t);
                                d
                            }
                        };
                        if t == "tool_result" && b["is_error"] == true {
                            d["name"] = json!("error");
                        }
                        d
                    })
                    .collect(),
                other => {
                    let mut d = display(role, "text", other);
                    d["text"] = json!(text_of(other));
                    vec![d]
                }
            }
        }
        "responses" => {
            let t = item["type"].as_str().unwrap_or("message");
            let mut d = match t {
                "function_call" | "custom_tool_call" | "local_shell_call" | "web_search_call" => {
                    let mut d = display("assistant", "tool_call", item);
                    d["name"] = item.get("name").cloned().unwrap_or(json!(t));
                    d["call_id"] = item["call_id"].clone();
                    d["arguments"] = item
                        .get("arguments")
                        .or_else(|| item.get("input"))
                        .or_else(|| item.get("action"))
                        .map(|v| v.as_str().map(|s| json!(s)).unwrap_or_else(|| json!(v.to_string())))
                        .unwrap_or(Value::Null);
                    d
                }
                "function_call_output" | "custom_tool_call_output" | "local_shell_call_output" => {
                    let mut d = display("tool", "tool_result", item);
                    d["call_id"] = item["call_id"].clone();
                    d["text"] = json!(text_of(&item["output"]));
                    d
                }
                "reasoning" => {
                    let mut d = display("assistant", "reasoning", item);
                    d["text"] = json!(text_of(&item["summary"]).or_else(|| text_of(&item["content"])));
                    d
                }
                "message" => {
                    let mut d = display(role_of(item), "text", item);
                    d["text"] = json!(text_of(&item["content"]));
                    d
                }
                other => {
                    let mut d = display(role_of(item), "other", item);
                    d["name"] = json!(other);
                    d
                }
            };
            if t == "message" && item.get("role").is_none() {
                d["role"] = json!("user");
            }
            vec![d]
        }
        "realtime" => {
            // The gateway's protocol-neutral items (`turn::Item`).
            match item["item"].as_str().unwrap_or("") {
                "message" => {
                    let mut d = display(role_of(item), "text", item);
                    d["text"] = json!(text_of(&item["content"]));
                    vec![d]
                }
                "reasoning" => {
                    let mut d = display("assistant", "reasoning", item);
                    d["text"] = item["text"].clone();
                    vec![d]
                }
                "tool_call" | "server_tool_call" => {
                    let mut d = display("assistant", "tool_call", item);
                    d["name"] = item["name"].clone();
                    d["call_id"] = item["id"].clone();
                    d["arguments"] = item
                        .get("arguments")
                        .cloned()
                        .unwrap_or_else(|| json!(item["input"].to_string()));
                    vec![d]
                }
                "tool_result" | "server_tool_result" => {
                    let mut d = display("tool", "tool_result", item);
                    d["call_id"] = item["call_id"].clone();
                    d["text"] = json!(text_of(&item["content"]).or_else(|| text_of(&item["output"])));
                    vec![d]
                }
                _ => vec![display("user", "other", item)],
            }
        }
        _ => {
            // Chat completions messages.
            let role = role_of(item);
            let mut out = vec![];
            if let Some(r) = item
                .get("reasoning_content")
                .or_else(|| item.get("reasoning"))
                .and_then(Value::as_str)
                .filter(|s| !s.is_empty())
            {
                let mut d = display(role, "reasoning", &Value::Null);
                d["text"] = json!(r);
                out.push(d);
            }
            let text = text_of(&item["content"]).filter(|s| !s.is_empty());
            if text.is_some() || item["tool_calls"].as_array().is_none_or(|a| a.is_empty()) {
                let kind = if role == "tool" { "tool_result" } else { "text" };
                let mut d = display(role, kind, &item["content"]);
                d["text"] = json!(text);
                d["call_id"] = item.get("tool_call_id").cloned().unwrap_or(Value::Null);
                out.push(d);
            }
            for call in item["tool_calls"].as_array().into_iter().flatten() {
                let mut d = display(role, "tool_call", call);
                d["name"] = call["function"]["name"].clone();
                d["arguments"] = call["function"]["arguments"].clone();
                d["call_id"] = call["id"].clone();
                out.push(d);
            }
            out
        }
    }
}

/// Display items for wire items starting at history index `start`.
pub(crate) fn display_items(protocol: &str, items: &[Value], start: usize) -> Vec<Value> {
    let mut out = vec![];
    for (i, item) in items.iter().enumerate() {
        for mut d in item_display(protocol, item) {
            d["index"] = json!(start + i);
            out.push(d);
        }
    }
    out
}

/// The assistant's output as display items.
pub(crate) fn display_response(protocol: &str, response: &Value) -> Vec<Value> {
    if response.is_null() {
        return vec![];
    }
    let mut out = match protocol {
        "messages" => item_display("messages", &json!({"role":"assistant","content":response["content"]})),
        "responses" | "realtime" => response["output"]
            .as_array()
            .into_iter()
            .flatten()
            .flat_map(|o| {
                let mut o = o.clone();
                if o["type"] == "message" && o.get("role").is_none() {
                    o["role"] = json!("assistant");
                }
                item_display("responses", &o)
            })
            .collect(),
        "completions" => {
            let mut d = display("assistant", "text", &Value::Null);
            d["text"] = response["choices"][0]["text"].clone();
            vec![d]
        }
        _ => response["choices"]
            .as_array()
            .and_then(|c| c.first())
            .map(|c| item_display("chat", &c["message"]))
            .unwrap_or_default(),
    };
    if let Some(e) = response.get("error").filter(|e| !e.is_null()) {
        let mut d = display("assistant", "other", &Value::Null);
        d["name"] = json!("error");
        d["text"] = json!(e["message"].as_str().map(str::to_owned).unwrap_or_else(|| e.to_string()));
        out.push(d);
    }
    for (i, d) in out.iter_mut().enumerate() {
        d["index"] = json!(i);
    }
    out
}

/// Token usage reported in a response, in one shape for every protocol.
pub(crate) fn usage(response: &Value) -> Value {
    let u = &response["usage"];
    if u.is_null() {
        return Value::Null;
    }
    let pick = |keys: &[&str]| keys.iter().find_map(|k| {
        k.split('.').try_fold(u, |v, p| v.get(p)).and_then(Value::as_u64)
    });
    let input = pick(&["prompt_tokens", "input_tokens"]);
    let cached = pick(&[
        "prompt_tokens_details.cached_tokens",
        "input_tokens_details.cached_tokens",
        "input_token_details.cached_tokens",
        "cache_read_input_tokens",
    ]);
    let output = pick(&["completion_tokens", "output_tokens"]);
    json!({"input_tokens":input,"cached_tokens":cached,"output_tokens":output})
}

pub(crate) fn stop(protocol: &str, response: &Value) -> Option<String> {
    let v = match protocol {
        "messages" => &response["stop_reason"],
        "responses" | "realtime" => &response["status"],
        _ => &response["choices"][0]["finish_reason"],
    };
    v.as_str().map(str::to_owned)
}

/// The first user text in the history, for session titles.
pub(crate) fn title(protocol: &str, items: &[Value]) -> Option<String> {
    display_items(protocol, items, 0)
        .into_iter()
        .filter(|d| d["role"] == "user" && d["kind"] == "text")
        .filter_map(|d| d["text"].as_str().map(str::to_owned))
        // Agent clients add tagged context blocks; prefer the user's own words.
        .map(|t| strip_tags(&t).split_whitespace().collect::<Vec<_>>().join(" "))
        .find(|t| !t.is_empty())
        .map(|t| t.chars().take(120).collect())
}

fn strip_tags(text: &str) -> String {
    let mut out = String::new();
    let mut rest = text;
    while let Some(start) = rest.find('<') {
        let tag = &rest[start + 1..];
        let Some(end) = tag.find('>') else { break };
        let name = &tag[..end];
        if name.is_empty() || !name.chars().all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_') {
            out.push_str(&rest[..start + 1]);
            rest = &rest[start + 1..];
            continue;
        }
        let close = format!("</{name}>");
        out.push_str(&rest[..start]);
        rest = match tag.find(&close) {
            Some(i) => &tag[i + close.len()..],
            None => &tag[end + 1..],
        };
    }
    out.push_str(rest);
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn split_join_roundtrip_each_protocol() {
        for (protocol, request) in [
            ("chat", json!({"model":"m","messages":[{"role":"system","content":"s"},{"role":"user","content":"hi"}],"tools":[{"type":"function"}],"stream":true})),
            ("messages", json!({"model":"m","system":[{"type":"text","text":"sys"}],"messages":[{"role":"user","content":"hi"}],"max_tokens":5})),
            ("responses", json!({"model":"m","instructions":"i","input":[{"type":"message","role":"user","content":"hi"}],"previous_response_id":"resp_1"})),
            ("completions", json!({"model":"m","prompt":"once"})),
        ] {
            let s = split(protocol, request.clone());
            assert!(!s.items.is_empty());
            if protocol == "responses" {
                assert_eq!(s.previous.as_deref(), Some("resp_1"));
            }
            let back = join(protocol, s);
            for (k, v) in request.as_object().unwrap() {
                assert_eq!(&back[k], v, "{protocol} {k}");
            }
        }
        let s = split("responses", json!({"input":"plain"}));
        assert_eq!(s.items[0]["content"], "plain");
    }
    #[test]
    fn cache_control_and_text_block_spelling_do_not_change_item_hash() {
        let a = json!({"role":"user","content":[{"type":"text","text":"x","cache_control":{"type":"ephemeral"}}]});
        let b = json!({"role":"user","content":[{"type":"text","text":"x"}]});
        let c = json!({"role":"user","content":"x"});
        assert_eq!(item_hash(&a), item_hash(&b));
        assert_eq!(item_hash(&b), item_hash(&c));
        assert_ne!(item_hash(&c), item_hash(&json!({"role":"assistant","content":"x"})));
        let two = json!({"role":"user","content":[{"type":"text","text":"x"},{"type":"text","text":"y"}]});
        assert_ne!(item_hash(&two), item_hash(&c));
    }
    #[test]
    fn title_skips_reminder_only_blocks() {
        let items = vec![json!({"role":"user","content":[{"type":"text","text":"<system-reminder>\nctx\n</system-reminder>"},
            {"type":"text","text":"<system-reminder>more</system-reminder>"},{"type":"text","text":"Fix calc.py"}]})];
        assert_eq!(title("messages", &items).as_deref(), Some("Fix calc.py"));
    }
    #[test]
    fn display_shapes_and_titles() {
        let items = vec![
            json!({"role":"user","content":[{"type":"text","text":"<system-reminder>ctx</system-reminder>Fix the bug"}]}),
            json!({"role":"assistant","content":[{"type":"thinking","thinking":"hm"},{"type":"tool_use","id":"t1","name":"Bash","input":{"cmd":"ls"}}]}),
            json!({"role":"user","content":[{"type":"tool_result","tool_use_id":"t1","content":"a.rs"}]}),
        ];
        let d = display_items("messages", &items, 3);
        assert_eq!(d.len(), 4);
        assert_eq!(d[1]["kind"], "reasoning");
        assert_eq!(d[2]["name"], "Bash");
        assert_eq!(d[3]["kind"], "tool_result");
        assert_eq!(d[3]["index"], 5);
        assert_eq!(title("messages", &items).as_deref(), Some("Fix the bug"));
        let chat = display_response("chat", &json!({"choices":[{"message":{"role":"assistant","content":"ok","tool_calls":[{"id":"c","function":{"name":"f","arguments":"{}"}}]}}]}));
        assert_eq!(chat.len(), 2);
        assert_eq!(usage(&json!({"usage":{"input_tokens":3,"cache_read_input_tokens":1,"output_tokens":2}}))["cached_tokens"], 1);
    }
}
