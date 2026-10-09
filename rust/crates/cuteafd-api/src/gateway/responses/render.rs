use std::collections::{HashMap, HashSet};

use serde_json::{json, Value};

use super::parse::{encode, Parsed, ToolKind};
use crate::gateway::{turn::*, GatewayError};

pub(super) fn id(prefix: &str) -> String {
    format!("{prefix}_{}", uuid::Uuid::new_v4().simple())
}
pub(super) fn now() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

/// One folding state drives both JSON and streaming, so item order and final
/// output cannot diverge between the two transports.
pub(super) struct Fold {
    pub response: Value,
    pub events: Vec<Value>,
    pub terminal: bool,
    sequence: u64,
    text: Option<usize>,
    reasoning: Option<usize>,
    calls: HashMap<usize, usize>,
    hosted: HashMap<String, usize>,
    hosted_results: HashMap<usize, Value>,
    arguments: HashMap<usize, String>,
    finished: HashSet<usize>,
    kinds: HashMap<String, ToolKind>,
    names: HashMap<String, (String, String)>,
    encrypted: bool,
    summary: bool,
}

impl Fold {
    pub fn new(p: &Parsed) -> Self {
        let mut response = json!({"id":id("resp"),"object":"response","created_at":now(),"completed_at":null,
            "status":"in_progress","error":null,"incomplete_details":null,"model":p.turn.requested_model,
            "output":[],"usage":null,"instructions":null,"tools":[],"tool_choice":"auto","parallel_tool_calls":true,
            "reasoning":{"effort":null,"summary":null},"text":{"format":{"type":"text"}},"store":p.store,
            "previous_response_id":null,"max_output_tokens":null,"temperature":1.0,"top_p":1.0,"truncation":"disabled",
            "metadata":{},"background":false,"service_tier":"default"});
        for key in [
            "instructions",
            "tools",
            "tool_choice",
            "parallel_tool_calls",
            "reasoning",
            "text",
            "previous_response_id",
            "max_output_tokens",
            "temperature",
            "top_p",
            "truncation",
            "metadata",
            "service_tier",
            "user",
            "safety_identifier",
            "prompt_cache_key",
            "include",
        ] {
            if let Some(v) = p.wire.get(key) {
                response[key] = v.clone();
            }
        }
        let mut fold = Self {
            response,
            events: Vec::new(),
            terminal: false,
            sequence: 0,
            text: None,
            reasoning: None,
            calls: HashMap::new(),
            hosted: HashMap::new(),
            hosted_results: HashMap::new(),
            arguments: HashMap::new(),
            finished: HashSet::new(),
            kinds: p.kinds.clone(),
            names: p.names.clone(),
            encrypted: p.encrypted,
            summary: p.summary,
        };
        fold.emit("response.created", json!({"response":fold.response}));
        fold.emit("response.in_progress", json!({"response":fold.response}));
        fold
    }

    pub fn emit(&mut self, kind: &str, mut body: Value) {
        body["type"] = json!(kind);
        body["sequence_number"] = json!(self.sequence);
        self.sequence += 1;
        self.events.push(body);
    }

    fn add(&mut self, item: Value) -> usize {
        let output = self.response["output"].as_array_mut().unwrap();
        let index = output.len();
        output.push(item.clone());
        self.emit(
            "response.output_item.added",
            json!({"output_index":index,"item":item}),
        );
        index
    }

    fn text_index(&mut self) -> usize {
        if let Some(i) = self.text {
            if !self.finished.contains(&i) {
                return i;
            }
        }
        let i = self.add(json!({"id":id("msg"),"type":"message","role":"assistant","status":"in_progress","content":[]}));
        let part = json!({"type":"output_text","text":"","annotations":[],"logprobs":[]});
        self.response["output"][i]["content"] = json!([part]);
        self.emit("response.content_part.added", json!({"output_index":i,"item_id":self.response["output"][i]["id"],"content_index":0,"part":part}));
        self.text = Some(i);
        i
    }

    fn reasoning_index(&mut self) -> usize {
        if let Some(i) = self.reasoning {
            if !self.finished.contains(&i) {
                return i;
            }
        }
        let i = self.add(json!({"id":id("rs"),"type":"reasoning","summary":[]}));
        // Raw reasoning is retained internally and only exposed as a requested
        // summary or opaque round-trip token, never as unsolicited trace text.
        self.arguments.insert(i, String::new());
        if self.summary {
            let part = json!({"type":"summary_text","text":""});
            self.response["output"][i]["summary"] = json!([part]);
            self.emit("response.reasoning_summary_part.added", json!({"output_index":i,"item_id":self.response["output"][i]["id"],"summary_index":0,"part":part}));
        }
        self.reasoning = Some(i);
        i
    }

    fn close_prose(&mut self) -> Result<(), GatewayError> {
        if let Some(i) = self.reasoning {
            self.finish_item(i)?;
        }
        if let Some(i) = self.text {
            self.finish_item(i)?;
        }
        Ok(())
    }

    pub fn accept(&mut self, event: TurnEvent) -> Result<(), GatewayError> {
        if self.terminal {
            return Err(GatewayError::upstream("backend emitted events after Done"));
        }
        match event {
            TurnEvent::TextDelta { text } => {
                if let Some(i) = self.reasoning {
                    self.finish_item(i)?;
                }
                let i = self.text_index();
                self.response["output"][i]["content"][0]["text"]
                    .as_str()
                    .map(str::to_owned)
                    .map(|mut s| {
                        s.push_str(&text);
                        self.response["output"][i]["content"][0]["text"] = json!(s);
                    });
                self.emit("response.output_text.delta", json!({"output_index":i,"item_id":self.response["output"][i]["id"],"content_index":0,"delta":text,"logprobs":[]}));
            }
            TurnEvent::ReasoningDelta { text } => {
                if let Some(i) = self.text {
                    self.finish_item(i)?;
                }
                let i = self.reasoning_index();
                self.arguments.get_mut(&i).unwrap().push_str(&text);
                if self.summary {
                    self.response["output"][i]["summary"][0]["text"] = json!(self.arguments[&i]);
                    self.emit("response.reasoning_summary_text.delta", json!({"output_index":i,"item_id":self.response["output"][i]["id"],"summary_index":0,"delta":text}));
                }
            }
            TurnEvent::ReasoningSignature { .. } => { /* Gateway tokens encode the text, not a provider's foreign token. */
            }
            TurnEvent::ToolCallStart {
                index,
                id: call_id,
                name,
            } => {
                self.close_prose()?;
                if self.calls.contains_key(&index) {
                    return Err(GatewayError::upstream("duplicate backend tool index"));
                }
                let kind = self.kinds.get(&name).cloned().unwrap_or(ToolKind::Function);
                let mut item = match kind {
                    ToolKind::Function => {
                        json!({"type":"function_call","id":id("fc"),"call_id":call_id,"name":name,"arguments":"","status":"in_progress"})
                    }
                    ToolKind::Custom => {
                        json!({"type":"custom_tool_call","id":id("ctc"),"call_id":call_id,"name":name,"input":"","status":"in_progress"})
                    }
                    ToolKind::LocalShell => {
                        json!({"type":"local_shell_call","id":id("lsc"),"call_id":call_id,"action":{"type":"exec","command":[]},"status":"in_progress"})
                    }
                };
                if let Some((namespace, local)) = self.names.get(&name) {
                    item["namespace"] = json!(namespace);
                    item["name"] = json!(local);
                }
                let i = self.add(item);
                self.calls.insert(index, i);
                self.arguments.insert(i, String::new());
            }
            TurnEvent::ToolCallDelta { index, arguments } => {
                let i = *self
                    .calls
                    .get(&index)
                    .ok_or_else(|| GatewayError::upstream("tool delta without start"))?;
                if self.finished.contains(&i) {
                    return Err(GatewayError::upstream("tool delta after end"));
                }
                self.arguments.get_mut(&i).unwrap().push_str(&arguments);
                match self.response["output"][i]["type"].as_str().unwrap() {
                    "function_call" => {
                        self.response["output"][i]["arguments"] = json!(self.arguments[&i]);
                        self.emit("response.function_call_arguments.delta", json!({"output_index":i,"item_id":self.response["output"][i]["id"],"delta":arguments}));
                    }
                    "custom_tool_call" => {
                        let value = custom_prefix(&self.arguments[&i]);
                        let old = self.response["output"][i]["input"].as_str().unwrap();
                        if let Some(delta) = value
                            .strip_prefix(old)
                            .filter(|d| !d.is_empty())
                            .map(str::to_owned)
                        {
                            self.response["output"][i]["input"] = json!(value);
                            self.emit("response.custom_tool_call_input.delta", json!({"output_index":i,"item_id":self.response["output"][i]["id"],"delta":delta}));
                        }
                    }
                    _ => {}
                }
            }
            TurnEvent::ToolCallEnd { index } => {
                let i = *self
                    .calls
                    .get(&index)
                    .ok_or_else(|| GatewayError::upstream("tool end without start"))?;
                // Chat adapters close calls before reporting a max-token stop.
                // Buffer invalid structured calls until Done distinguishes a
                // truncated response from malformed successful model output.
                let structured = matches!(
                    self.response["output"][i]["type"].as_str(),
                    Some("custom_tool_call" | "local_shell_call")
                );
                if !structured || serde_json::from_str::<Value>(&self.arguments[&i]).is_ok() {
                    self.finish_item(i)?;
                }
            }
            TurnEvent::ServerToolCall {
                id: call_id,
                name,
                input,
            } => {
                self.close_prose()?;
                if name != "web_search" {
                    return Err(GatewayError::upstream("unsupported server tool event"));
                }
                let query = input["query"].as_str().unwrap_or("");
                let i = self.add(json!({"type":"web_search_call","id":call_id,"status":"in_progress","action":{"type":"search","query":query,"sources":[]}}));
                self.hosted.insert(call_id.clone(), i);
                self.emit(
                    "response.web_search_call.in_progress",
                    json!({"output_index":i,"item_id":call_id}),
                );
                self.response["output"][i]["status"] = json!("searching");
                self.emit(
                    "response.web_search_call.searching",
                    json!({"output_index":i,"item_id":call_id}),
                );
            }
            TurnEvent::ServerToolResult { id, output, .. } => {
                let i = *self
                    .hosted
                    .get(&id)
                    .ok_or_else(|| GatewayError::upstream("server tool result without call"))?;
                let sources: Vec<Value> = output["results"]
                    .as_array()
                    .into_iter()
                    .flatten()
                    .filter_map(|r| r["url"].as_str())
                    .map(|url| json!({"type":"url","url":url}))
                    .collect();
                self.response["output"][i]["action"]["sources"] = json!(sources);
                self.hosted_results.insert(i, output);
                self.finish_item(i)?;
            }
            TurnEvent::Usage { usage } => {
                self.response["usage"] = json!({"input_tokens":usage.input_tokens,"input_tokens_details":{"cached_tokens":usage.cached_input_tokens},
                    "output_tokens":usage.output_tokens,"output_tokens_details":{"reasoning_tokens":usage.reasoning_tokens},
                    "total_tokens":u64::from(usage.input_tokens)+u64::from(usage.output_tokens)});
            }
            TurnEvent::Done { stop } => {
                for i in 0..self.response["output"].as_array().unwrap().len() {
                    let kind = self.response["output"][i]["type"].as_str().unwrap();
                    if matches!(
                        stop,
                        StopReason::MaxTokens | StopReason::Cancelled | StopReason::ContentFilter
                    ) && !self.finished.contains(&i)
                        && matches!(
                            kind,
                            "function_call" | "custom_tool_call" | "local_shell_call"
                        )
                    {
                        self.response["output"][i]["status"] = json!("incomplete");
                        self.finished.insert(i);
                        self.emit(
                            "response.output_item.done",
                            json!({"output_index":i,"item":self.response["output"][i]}),
                        );
                    } else {
                        self.finish_item(i)?;
                    }
                }
                let (status, reason) = match stop {
                    StopReason::MaxTokens => ("incomplete", Some("max_output_tokens")),
                    StopReason::ContentFilter => ("incomplete", Some("content_filter")),
                    StopReason::Cancelled => ("incomplete", Some("steered")),
                    _ => ("completed", None),
                };
                self.response["status"] = json!(status);
                self.response["completed_at"] = json!(now());
                if let Some(reason) = reason {
                    self.response["incomplete_details"] = json!({"reason":reason});
                }
                self.emit(
                    &format!("response.{status}"),
                    json!({"response":self.response}),
                );
                self.terminal = true;
            }
        }
        Ok(())
    }

    fn finish_item(&mut self, i: usize) -> Result<(), GatewayError> {
        if self.finished.contains(&i) {
            return Ok(());
        }
        let item = self.response["output"][i].clone();
        let ident = json!({"output_index":i,"item_id":item["id"]});
        match item["type"].as_str().unwrap() {
            "message" => {
                let part = item["content"][0].clone();
                let mut e = ident.clone();
                e["content_index"] = json!(0);
                e["text"] = part["text"].clone();
                e["logprobs"] = json!([]);
                self.emit("response.output_text.done", e);
                let mut e = ident.clone();
                e["content_index"] = json!(0);
                e["part"] = part;
                self.emit("response.content_part.done", e);
                self.response["output"][i]["status"] = json!("completed");
            }
            "reasoning" => {
                if self.summary {
                    let part = item["summary"][0].clone();
                    let mut e = ident.clone();
                    e["summary_index"] = json!(0);
                    e["text"] = part["text"].clone();
                    self.emit("response.reasoning_summary_text.done", e);
                    let mut e = ident.clone();
                    e["summary_index"] = json!(0);
                    e["part"] = part;
                    self.emit("response.reasoning_summary_part.done", e);
                }
                if self.encrypted {
                    self.response["output"][i]["encrypted_content"] =
                        json!(encode("reasoning", &self.arguments[&i]));
                }
            }
            "function_call" => {
                let mut e = ident.clone();
                e["arguments"] = item["arguments"].clone();
                self.emit("response.function_call_arguments.done", e);
                self.response["output"][i]["status"] = json!("completed");
            }
            "custom_tool_call" => {
                let args: Value = serde_json::from_str(&self.arguments[&i]).map_err(|_| {
                    GatewayError::upstream("custom tool arguments are not valid JSON")
                })?;
                let input = args["input"].as_str().ok_or_else(|| {
                    GatewayError::upstream("custom tool arguments must contain string input")
                })?;
                let old = item["input"].as_str().unwrap();
                if let Some(delta) = input.strip_prefix(old).filter(|s| !s.is_empty()) {
                    let mut e = ident.clone();
                    e["delta"] = json!(delta);
                    self.emit("response.custom_tool_call_input.delta", e);
                }
                self.response["output"][i]["input"] = json!(input);
                let mut e = ident.clone();
                e["input"] = json!(input);
                self.emit("response.custom_tool_call_input.done", e);
                self.response["output"][i]["status"] = json!("completed");
            }
            "local_shell_call" => {
                let action: Value = serde_json::from_str(&self.arguments[&i])
                    .map_err(|_| GatewayError::upstream("local_shell action must be valid JSON"))?;
                if action["type"] != "exec" || !action["command"].is_array() {
                    return Err(GatewayError::upstream("invalid local_shell action"));
                }
                self.response["output"][i]["action"] = action;
                self.response["output"][i]["status"] = json!("completed");
            }
            "web_search_call" => {
                self.response["output"][i]["status"] = json!("completed");
                self.emit("response.web_search_call.completed", ident);
            }
            _ => {}
        }
        self.finished.insert(i);
        self.emit(
            "response.output_item.done",
            json!({"output_index":i,"item":self.response["output"][i]}),
        );
        Ok(())
    }

    pub fn history_item(&self, i: usize) -> Result<Vec<Item>, GatewayError> {
        let v = &self.response["output"][i];
        if v["type"] == "reasoning" {
            return Ok(vec![Item::Reasoning {
                text: self.arguments.get(&i).cloned().unwrap_or_default(),
                signature: None,
            }]);
        }
        let mut items = super::parse::parse_item(v)?;
        if let Some(output) = self.hosted_results.get(&i) {
            items.push(Item::ServerToolResult {
                call_id: v["id"].as_str().unwrap().into(),
                name: "web_search".into(),
                output: output.clone(),
            });
        }
        Ok(items)
    }

    pub fn fail(&mut self, error: &GatewayError) {
        self.response["status"] = json!("failed");
        self.response["error"] = json!({"code":error.openai_body()["error"]["code"].as_str().unwrap_or("server_error"),"message":error.message});
        self.emit("error", json!({"code":self.response["error"]["code"],"message":error.message,"param":error.param}));
        self.emit("response.failed", json!({"response":self.response}));
        self.terminal = true;
    }
}

/// Decode only complete JSON string characters; split escapes (including
/// surrogate pairs) stay buffered until the next backend chunk.
fn custom_prefix(args: &str) -> String {
    let Some((_, rest)) = args.split_once("\"input\"") else {
        return String::new();
    };
    let Some(rest) = rest.trim_start().strip_prefix(':') else {
        return String::new();
    };
    let Some(rest) = rest.trim_start().strip_prefix('"') else {
        return String::new();
    };
    let mut escaped = false;
    for (i, c) in rest.char_indices() {
        if c == '"' && !escaped {
            return serde_json::from_str(&format!("\"{}\"", &rest[..i])).unwrap_or_default();
        }
        escaped = c == '\\' && !escaped;
    }
    for end in rest
        .char_indices()
        .map(|(i, _)| i)
        .chain(std::iter::once(rest.len()))
        .collect::<Vec<_>>()
        .into_iter()
        .rev()
    {
        if let Ok(s) = serde_json::from_str::<String>(&format!("\"{}\"", &rest[..end])) {
            return s;
        }
    }
    String::new()
}
