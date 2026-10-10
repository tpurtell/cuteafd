//! Folds a captured response into the object a non-streaming call returns.
use cuteafd_api::usage_log::ResponsePayload;
use serde_json::{json, Value};
use std::collections::BTreeMap;

pub(crate) fn parse(bytes: &[u8]) -> Value {
    serde_json::from_slice(bytes).unwrap_or_else(|_| {
        let text = String::from_utf8_lossy(&bytes[..bytes.len().min(4096)]).into_owned();
        json!({"non_json": text, "bytes": bytes.len()})
    })
}

fn concat(frames: &[bytes::Bytes]) -> Vec<u8> {
    let mut out = Vec::with_capacity(frames.iter().map(|f| f.len()).sum());
    for f in frames {
        out.extend_from_slice(f);
    }
    out
}

/// `data:` payloads of an event stream, in order (comments and `[DONE]` skipped).
fn sse_events(bytes: &[u8]) -> Vec<Value> {
    let text = String::from_utf8_lossy(bytes);
    let mut events = Vec::new();
    for block in text.split("\n\n") {
        let data = block
            .lines()
            .filter_map(|l| l.strip_prefix("data:"))
            .map(|l| l.strip_prefix(' ').unwrap_or(l))
            .collect::<Vec<_>>()
            .join("\n");
        if data.is_empty() || data == "[DONE]" {
            continue;
        }
        if let Ok(v) = serde_json::from_str::<Value>(&data) {
            events.push(v);
        }
    }
    events
}

pub(crate) fn fold(protocol: &str, payload: &ResponsePayload) -> Value {
    match payload {
        ResponsePayload::None => Value::Null,
        ResponsePayload::Object(b) => parse(b),
        ResponsePayload::Frames { frames, sse, truncated } => {
            let bytes = concat(frames);
            let mut v = if *sse {
                let events = sse_events(&bytes);
                match protocol {
                    "messages" => fold_messages(&events),
                    "responses" => fold_responses(&events),
                    _ => fold_chat(&events),
                }
            } else {
                parse(&bytes)
            };
            if *truncated {
                if let Some(o) = v.as_object_mut() {
                    o.insert("log_truncated".into(), json!(true));
                }
            }
            v
        }
    }
}

fn append(slot: &mut Value, text: &str) {
    let old = slot.as_str().unwrap_or("");
    *slot = json!(format!("{old}{text}"));
}

pub(crate) fn fold_chat(chunks: &[Value]) -> Value {
    let mut result = json!({"object":"chat.completion","choices":[]});
    let mut choices = BTreeMap::<u64, Value>::new();
    for chunk in chunks {
        if chunk.get("error").is_some() {
            result["error"] = chunk["error"].clone();
        }
        for field in ["id", "model", "created", "usage", "system_fingerprint"] {
            if let Some(v) = chunk.get(field).filter(|v| !v.is_null()) {
                result[field] = v.clone();
            }
        }
        for choice in chunk["choices"].as_array().into_iter().flatten() {
            let index = choice["index"].as_u64().unwrap_or(0);
            let output = choices.entry(index).or_insert_with(|| {
                json!({"index":index,"message":{"role":"assistant","content":""},"finish_reason":null})
            });
            let delta = &choice["delta"];
            for field in ["content", "reasoning_content", "reasoning"] {
                if let Some(text) = delta[field].as_str() {
                    append(&mut output["message"][field], text);
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
                let tools = output["message"]["tool_calls"].as_array_mut().expect("array");
                while tools.len() <= index {
                    tools.push(json!({"type":"function","function":{"name":"","arguments":""}}));
                }
                if let Some(id) = tool.get("id") {
                    tools[index]["id"] = id.clone();
                }
                for field in ["name", "arguments"] {
                    if let Some(text) = tool["function"][field].as_str() {
                        append(&mut tools[index]["function"][field], text);
                    }
                }
            }
        }
    }
    result["choices"] = json!(choices.into_values().collect::<Vec<_>>());
    result
}

pub(crate) fn fold_messages(events: &[Value]) -> Value {
    let mut message = json!({"type":"message","role":"assistant","content":[]});
    let mut blocks = BTreeMap::<u64, Value>::new();
    for e in events {
        match e["type"].as_str().unwrap_or("") {
            "message_start" => {
                if let Some(m) = e["message"].as_object() {
                    for (k, v) in m {
                        if k != "content" {
                            message[k] = v.clone();
                        }
                    }
                }
            }
            "content_block_start" => {
                let index = e["index"].as_u64().unwrap_or(0);
                let mut block = e["content_block"].clone();
                if block["type"] == "tool_use" || block["type"] == "server_tool_use" {
                    block["input_json"] = json!("");
                }
                blocks.insert(index, block);
            }
            "content_block_delta" => {
                let index = e["index"].as_u64().unwrap_or(0);
                let block = blocks.entry(index).or_insert_with(|| json!({"type":"text","text":""}));
                let d = &e["delta"];
                match d["type"].as_str().unwrap_or("") {
                    "text_delta" => append(&mut block["text"], d["text"].as_str().unwrap_or("")),
                    "thinking_delta" => append(&mut block["thinking"], d["thinking"].as_str().unwrap_or("")),
                    "signature_delta" => append(&mut block["signature"], d["signature"].as_str().unwrap_or("")),
                    "input_json_delta" => {
                        append(&mut block["input_json"], d["partial_json"].as_str().unwrap_or(""))
                    }
                    _ => {}
                }
            }
            "message_delta" => {
                for (k, v) in e["delta"].as_object().into_iter().flatten() {
                    message[k] = v.clone();
                }
                if let Some(u) = e["usage"].as_object() {
                    for (k, v) in u {
                        message["usage"][k] = v.clone();
                    }
                }
            }
            "error" => message["error"] = e["error"].clone(),
            _ => {}
        }
    }
    let content = blocks
        .into_values()
        .map(|mut b| {
            if let Some(raw) = b.as_object_mut().and_then(|o| o.remove("input_json")) {
                let raw = raw.as_str().unwrap_or("");
                b["input"] = if raw.is_empty() {
                    b.get("input").cloned().unwrap_or(json!({}))
                } else {
                    serde_json::from_str(raw).unwrap_or_else(|_| json!(raw))
                };
            }
            b
        })
        .collect::<Vec<_>>();
    message["content"] = json!(content);
    message
}

pub(crate) fn fold_responses(events: &[Value]) -> Value {
    // The terminal event carries the full response object.
    for e in events.iter().rev() {
        if matches!(
            e["type"].as_str(),
            Some("response.completed" | "response.failed" | "response.incomplete")
        ) {
            return e["response"].clone();
        }
    }
    let mut last = events
        .iter()
        .rev()
        .find_map(|e| e.get("response").cloned())
        .unwrap_or_else(|| json!({"object":"response","output":[]}));
    let output = events
        .iter()
        .filter(|e| e["type"] == "response.output_item.done")
        .map(|e| e["item"].clone())
        .collect::<Vec<_>>();
    if !output.is_empty() {
        last["output"] = json!(output);
    }
    if let Some(e) = events.iter().find(|e| e["type"] == "error") {
        last["error"] = e["error"].clone();
    }
    last
}

#[cfg(test)]
mod tests {
    use super::*;
    use bytes::Bytes;
    fn frames(text: &str) -> ResponsePayload {
        ResponsePayload::Frames {
            frames: text.split_inclusive("\n\n").map(|s| Bytes::from(s.to_owned())).collect(),
            sse: true,
            truncated: false,
        }
    }
    #[test]
    fn chat_stream_folds_text_reasoning_tools_usage() {
        let v = fold("chat", &frames(concat!(
            "data: {\"id\":\"c1\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"hi\",\"reasoning_content\":\"think\",\"tool_calls\":[{\"index\":0,\"id\":\"t\",\"function\":{\"name\":\"run\",\"arguments\":\"{\"}}]}}]}\n\n",
            ": keepalive\n\n",
            "data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"!\",\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\"}\"}}]},\"finish_reason\":\"tool_calls\"}]}\n\n",
            "data: {\"choices\":[],\"usage\":{\"completion_tokens\":2}}\n\n",
            "data: [DONE]\n\n")));
        assert_eq!(v["id"], "c1");
        assert_eq!(v["choices"][0]["message"]["content"], "hi!");
        assert_eq!(v["choices"][0]["message"]["reasoning_content"], "think");
        assert_eq!(v["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"], "{}");
        assert_eq!(v["choices"][0]["finish_reason"], "tool_calls");
        assert_eq!(v["usage"]["completion_tokens"], 2);
    }
    #[test]
    fn messages_stream_folds_blocks() {
        let v = fold("messages", &frames(concat!(
            "event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"id\":\"m1\",\"model\":\"x\",\"usage\":{\"input_tokens\":5}}}\n\n",
            "event: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"thinking\",\"thinking\":\"\"}}\n\n",
            "event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"thinking_delta\",\"thinking\":\"hm\"}}\n\n",
            "event: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":1,\"content_block\":{\"type\":\"tool_use\",\"id\":\"tu\",\"name\":\"Bash\",\"input\":{}}}\n\n",
            "event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":1,\"delta\":{\"type\":\"input_json_delta\",\"partial_json\":\"{\\\"cmd\\\":\"}}\n\n",
            "event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":1,\"delta\":{\"type\":\"input_json_delta\",\"partial_json\":\"\\\"ls\\\"}\"}}\n\n",
            "event: message_delta\ndata: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"tool_use\"},\"usage\":{\"output_tokens\":9}}\n\n",
            "event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n")));
        assert_eq!(v["id"], "m1");
        assert_eq!(v["content"][0]["thinking"], "hm");
        assert_eq!(v["content"][1]["input"]["cmd"], "ls");
        assert_eq!(v["stop_reason"], "tool_use");
        assert_eq!(v["usage"]["output_tokens"], 9);
        assert_eq!(v["usage"]["input_tokens"], 5);
    }
    #[test]
    fn responses_stream_uses_terminal_object() {
        let v = fold("responses", &frames(concat!(
            "event: response.created\ndata: {\"type\":\"response.created\",\"response\":{\"id\":\"r\",\"status\":\"in_progress\"}}\n\n",
            "event: response.completed\ndata: {\"type\":\"response.completed\",\"response\":{\"id\":\"r\",\"status\":\"completed\",\"output\":[{\"type\":\"message\"}]}}\n\n")));
        assert_eq!(v["status"], "completed");
        assert_eq!(v["output"][0]["type"], "message");
    }
}
