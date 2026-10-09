//! Response event lifecycle; one actor owns both socket and turn stream.
use super::{
    super::turn::{Item, Part, Role, StopReason, TurnEvent, Usage},
    protocol::{id, wire_item},
};
use serde_json::{json, Value};

pub struct Output {
    pub id: String,
    pub item: Item,
    pub finished: bool,
    pub tool_index: Option<usize>,
    pub published: bool,
    pub status: &'static str,
    pub synced_final: bool,
    tool_ended: bool,
}
pub struct Response {
    pub id: String,
    pub conversation_id: Option<String>,
    pub metadata: Value,
    pub output: Vec<Output>,
    pub usage: Usage,
    pub stream: super::super::TurnStream,
    pub beta: bool,
    pub previous_item_id: Option<String>,
    pub config: Value,
}
impl Response {
    pub fn object(&self, status: &str, details: Value) -> Value {
        let mut object = json!({"id":self.id,"object":"realtime.response","status":status,"status_details":details,
            "output":self.output.iter().map(|o|wire_item(&o.id,&o.item,o.status,self.beta)).collect::<Vec<_>>(),
            "conversation_id":self.conversation_id,"metadata":self.metadata,
            "usage":if status=="in_progress" {Value::Null} else {json!({"total_tokens":self.usage.input_tokens.saturating_add(self.usage.output_tokens),"input_tokens":self.usage.input_tokens,"output_tokens":self.usage.output_tokens,"input_token_details":{"cached_tokens":self.usage.cached_input_tokens,"text_tokens":self.usage.input_tokens,"audio_tokens":0,"cached_tokens_details":{"text_tokens":self.usage.cached_input_tokens,"audio_tokens":0}},"output_token_details":{"text_tokens":self.usage.output_tokens,"audio_tokens":0}})}});
        let modalities = if self.beta {
            "modalities"
        } else {
            "output_modalities"
        };
        let max = if self.beta {
            "max_response_output_tokens"
        } else {
            "max_output_tokens"
        };
        object[modalities] = self.config[modalities].clone();
        object[max] = self.config[max].clone();
        object
    }
    fn fields(&self, index: usize) -> Value {
        json!({"response_id":self.id,"item_id":self.output[index].id,"output_index":index})
    }
    fn event(&self, index: usize, kind: &str, extra: Value) -> Value {
        let mut event = self.fields(index);
        event["type"] = json!(kind);
        for (k, v) in extra.as_object().unwrap() {
            event[k] = v.clone();
        }
        event
    }
    fn add(&mut self, item: Item, tool_index: Option<usize>) -> Vec<Value> {
        let index = self.output.len();
        self.output.push(Output {
            id: id("item"),
            item,
            finished: false,
            tool_index,
            published: false,
            status: "in_progress",
            synced_final: false,
            tool_ended: false,
        });
        let mut events = vec![];
        if self.conversation_id.is_some() {
            let previous_item_id = self
                .output
                .get(index.wrapping_sub(1))
                .map(|o| o.id.clone())
                .or_else(|| self.previous_item_id.clone());
            events.push(json!({"type":if self.beta {"conversation.item.created"} else {"conversation.item.added"},"previous_item_id":previous_item_id,"item":wire_item(&self.output[index].id,&self.output[index].item ,"in_progress",self.beta)}));
        }
        events.push(self.event(index,"response.output_item.added",json!({"item":wire_item(&self.output[index].id,&self.output[index].item ,"in_progress",self.beta)})));
        if tool_index.is_none() {
            events.push(self.event(index,"response.content_part.added",json!({"content_index":0,"part":{"type":if self.beta {"text"} else {"output_text"},"text":""}})));
        }
        events
    }
    fn finish(&mut self, index: usize, completed: bool) -> Vec<Value> {
        if self.output[index].finished {
            return vec![];
        }
        let mut events = vec![];
        match &self.output[index].item {
            Item::Message { content, .. } => {
                let Part::Text { text } = &content[0] else {
                    unreachable!()
                };
                events.push(self.event(
                    index,
                    if self.beta {
                        "response.text.done"
                    } else {
                        "response.output_text.done"
                    },
                    json!({"content_index":0,"text":text}),
                ));
                events.push(self.event(index,"response.content_part.done",json!({"content_index":0,"part":{"type":if self.beta {"text"} else {"output_text"},"text":text}})));
            }
            Item::ToolCall {
                arguments,
                id: call_id,
                name,
            } => events.push(self.event(
                index,
                "response.function_call_arguments.done",
                json!({"call_id":call_id,"name":name,"arguments":arguments}),
            )),
            _ => {}
        }
        self.output[index].finished = true;
        self.output[index].status = if completed { "completed" } else { "incomplete" };
        events.push(self.event(index,"response.output_item.done",json!({"item":wire_item(&self.output[index].id,&self.output[index].item ,self.output[index].status,self.beta)})));
        if !self.beta && self.conversation_id.is_some() {
            events.push(json!({"type":"conversation.item.done","item":wire_item(&self.output[index].id,&self.output[index].item ,self.output[index].status,self.beta)}));
        }
        events
    }
    pub fn event_turn(&mut self, event: TurnEvent) -> Vec<Value> {
        let mut events = vec![];
        match event {
            TurnEvent::TextDelta { text } => {
                let index = match self
                    .output
                    .iter()
                    .position(|o| o.tool_index.is_none() && !o.finished)
                {
                    Some(i) => i,
                    None => {
                        events.extend(self.add(
                            Item::Message {
                                role: Role::Assistant,
                                content: vec![Part::text("")],
                            },
                            None,
                        ));
                        self.output.len() - 1
                    }
                };
                if let Item::Message { content, .. } = &mut self.output[index].item {
                    if let Part::Text { text: all } = &mut content[0] {
                        all.push_str(&text);
                    }
                }
                events.push(self.event(
                    index,
                    if self.beta {
                        "response.text.delta"
                    } else {
                        "response.output_text.delta"
                    },
                    json!({"content_index":0,"delta":text}),
                ));
            }
            TurnEvent::ToolCallStart {
                index,
                id: call_id,
                name,
            } => {
                // A later text segment belongs after this call, not in an earlier message.
                if let Some(i) = self
                    .output
                    .iter()
                    .rposition(|o| o.tool_index.is_none() && !o.finished)
                {
                    events.extend(self.finish(i, true));
                }
                events.extend(self.add(
                    Item::ToolCall {
                        id: call_id,
                        name,
                        arguments: String::new(),
                    },
                    Some(index),
                ));
            }
            TurnEvent::ToolCallDelta { index, arguments } => {
                if let Some(i) = self.output.iter().position(|o| o.tool_index == Some(index)) {
                    if let Item::ToolCall { arguments: all, .. } = &mut self.output[i].item {
                        all.push_str(&arguments);
                    }
                    let Item::ToolCall { id: call_id, .. } = &self.output[i].item else {
                        unreachable!()
                    };
                    events.push(self.event(
                        i,
                        "response.function_call_arguments.delta",
                        json!({"call_id":call_id,"delta":arguments}),
                    ));
                }
            }
            // Upstreams flush ToolCallEnd even for length-truncated calls. Only
            // the terminal stop reason establishes whether the call is executable.
            TurnEvent::ToolCallEnd { index } => {
                if let Some(output) = self.output.iter_mut().find(|o| o.tool_index == Some(index)) {
                    output.tool_ended = true;
                }
            }
            TurnEvent::Usage { usage } => self.usage = usage,
            // Realtime has no reasoning/server-tool extension events.
            _ => {}
        }
        events
    }
    pub fn finish_turn(
        &mut self,
        stop: Option<StopReason>,
        error: Option<super::super::GatewayError>,
    ) -> Vec<Value> {
        let mut events = vec![];
        let completed = error.is_none()
            && !matches!(
                stop,
                Some(
                    StopReason::Cancelled
                        | StopReason::MaxTokens
                        | StopReason::Refusal
                        | StopReason::ContentFilter
                )
            );
        for i in 0..self.output.len() {
            let item_completed =
                completed && (self.output[i].tool_index.is_none() || self.output[i].tool_ended);
            events.extend(self.finish(i, item_completed));
        }
        let (status, details) = if let Some(e) = error {
            (
                "failed",
                json!({"type":"failed","error":{"type":e.openai_type(),"code":"backend_error","message":e.message}}),
            )
        } else {
            match stop {
                Some(StopReason::Cancelled) => (
                    "cancelled",
                    json!({"type":"cancelled","reason":"client_cancelled"}),
                ),
                Some(StopReason::MaxTokens) => (
                    "incomplete",
                    json!({"type":"incomplete","reason":"max_output_tokens"}),
                ),
                Some(StopReason::Refusal | StopReason::ContentFilter) => (
                    "incomplete",
                    json!({"type":"incomplete","reason":"content_filter"}),
                ),
                _ => ("completed", Value::Null),
            }
        };
        events.push(json!({"type":"response.done","response":self.object(status,details)}));
        events
    }
}
