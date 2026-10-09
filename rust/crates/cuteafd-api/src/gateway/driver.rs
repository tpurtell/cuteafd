//! Runs one turn against the backend, executing hosted tools (web search)
//! in a loop so front ends see a single event stream.
use std::sync::Arc;

use futures::StreamExt;

use super::backend::TurnStream;
use super::error::GatewayError;
use super::search::{self, SearchQuery};
use super::turn::{Item, Part, Role, StopReason, ToolChoice, TurnEvent, TurnRequest, Usage};
use super::Gateway;

/// Hosted searches per turn when the client sets no `max_uses`.
const DEFAULT_SEARCH_ROUNDS: u32 = 5;
/// Backend rounds per turn in the hosted-tool loop, whatever the model does.
const MAX_HOSTED_ROUNDS: u32 = 16;

struct PendingCall { id: String, name: String, arguments: String, hosted: bool, out_index: Option<usize> }

pub(super) async fn run(gateway: Arc<Gateway>, mut turn: TurnRequest) -> Result<TurnStream, GatewayError> {
    let hosted = turn.hosted.web_search.take();
    let provider = match (&hosted, &gateway.search) {
        (Some(_), None) => return Err(GatewayError::unsupported(
            "web search is not configured on this server (start it with a search provider)")),
        (Some(_), Some(provider)) => Some(provider.clone()),
        (None, _) => None,
    };
    if let Some(spec) = &hosted {
        if turn.tools.iter().any(|tool| tool.name == search::TOOL_NAME) {
            return Err(GatewayError::invalid(format!(
                "a client tool named '{}' conflicts with the hosted web search tool", search::TOOL_NAME)));
        }
        turn.tools.push(search::tool_spec(spec));
    }
    let backend = gateway.backend.clone();
    let first = backend.start(turn.clone()).await?;
    let Some(provider) = provider else { return Ok(seal_tool_calls(first)) };
    let spec = hosted.expect("provider implies hosted spec");
    let max_rounds = spec.max_uses.unwrap_or(DEFAULT_SEARCH_ROUNDS);
    Ok(seal_tool_calls(Box::pin(async_stream::stream! {
        let mut stream = first;
        let mut total = Usage::default();
        let mut rounds = 0u32;
        let mut next_index = 0usize;
        loop {
            let mut calls: Vec<Option<PendingCall>> = Vec::new();
            let mut text = String::new();
            let mut reasoning = String::new();
            let mut signature = None;
            let mut round_usage = Usage::default();
            let mut stop = None;
            while let Some(event) = stream.next().await {
                let event = match event { Ok(event) => event, Err(error) => { yield Err(error); return; } };
                match event {
                    TurnEvent::TextDelta { text: delta } => { text.push_str(&delta); yield Ok(TurnEvent::TextDelta { text: delta }); }
                    TurnEvent::ReasoningDelta { text: delta } => { reasoning.push_str(&delta); yield Ok(TurnEvent::ReasoningDelta { text: delta }); }
                    TurnEvent::ReasoningSignature { signature: value } => { signature = Some(value.clone()); yield Ok(TurnEvent::ReasoningSignature { signature: value }); }
                    TurnEvent::ToolCallStart { index, id, name } => {
                        if calls.len() <= index { calls.resize_with(index + 1, || None); }
                        let hosted = name == search::TOOL_NAME;
                        let out_index = (!hosted).then(|| { next_index += 1; next_index - 1 });
                        if let Some(out) = out_index { yield Ok(TurnEvent::ToolCallStart { index: out, id: id.clone(), name: name.clone() }); }
                        calls[index] = Some(PendingCall { id, name, arguments: String::new(), hosted, out_index });
                    }
                    TurnEvent::ToolCallDelta { index, arguments } => {
                        if let Some(Some(call)) = calls.get_mut(index) {
                            call.arguments.push_str(&arguments);
                            if let Some(out) = call.out_index { yield Ok(TurnEvent::ToolCallDelta { index: out, arguments }); }
                        }
                    }
                    TurnEvent::ToolCallEnd { index } => {
                        if let Some(Some(PendingCall { out_index: Some(out), .. })) = calls.get(index) {
                            yield Ok(TurnEvent::ToolCallEnd { index: *out });
                        }
                    }
                    TurnEvent::Usage { usage } => round_usage = usage,
                    TurnEvent::Done { stop: reason } => { stop = Some(reason); break; }
                    other => yield Ok(other),
                }
            }
            total.add(round_usage);
            let Some(stop) = stop else {
                yield Err(GatewayError::upstream("backend stream ended without a stop reason"));
                return;
            };
            let calls: Vec<PendingCall> = calls.into_iter().flatten().collect();
            let searches: Vec<&PendingCall> = calls.iter().filter(|call| call.hosted).collect();
            let client_calls = calls.iter().any(|call| !call.hosted);
            if searches.is_empty() {
                yield Ok(TurnEvent::Usage { usage: total });
                yield Ok(TurnEvent::Done { stop });
                return;
            }
            // Record what the model said this round, then run its searches.
            if !reasoning.is_empty() { turn.items.push(Item::Reasoning { text: reasoning, signature }); }
            if !text.is_empty() { turn.items.push(Item::Message { role: Role::Assistant, content: vec![Part::text(text)] }); }
            for call in &searches {
                let input: serde_json::Value = serde_json::from_str(&call.arguments)
                    .unwrap_or_else(|_| serde_json::json!({"query": call.arguments}));
                let query = input.get("query").and_then(|q| q.as_str()).unwrap_or_default().to_string();
                yield Ok(TurnEvent::ServerToolCall { id: call.id.clone(), name: call.name.clone(), input: input.clone() });
                let output = if total.web_search_requests >= max_rounds {
                    serde_json::json!({"error": "max_uses_exceeded"})
                } else if query.trim().is_empty() {
                    serde_json::json!({"error": "invalid_input"})
                } else {
                    total.web_search_requests += 1;
                    match provider.search_with_tape(SearchQuery { query, allowed_domains: spec.allowed_domains.clone(),
                        blocked_domains: spec.blocked_domains.clone(), max_results: 5 }, turn.tape.clone()).await {
                        Ok(hits) => serde_json::json!({"results": hits}),
                        Err(error) => {
                            tracing::warn!(provider = provider.name(), %error, "hosted web search failed");
                            serde_json::json!({"error": "unavailable"})
                        }
                    }
                };
                yield Ok(TurnEvent::ServerToolResult { id: call.id.clone(), name: call.name.clone(), output: output.clone() });
                turn.items.push(Item::ServerToolCall { id: call.id.clone(), name: call.name.clone(), input });
                turn.items.push(Item::ServerToolResult { call_id: call.id.clone(), name: call.name.clone(), output });
            }
            rounds += 1;
            // A forced tool choice applies to the first call only (as with
            // Anthropic's server tools); continuations let the model answer.
            // Once searches are exhausted, stop offering the tool so the
            // model must answer from what it has.
            if matches!(turn.tool_choice, ToolChoice::Required | ToolChoice::Named { .. }) {
                turn.tool_choice = ToolChoice::Auto;
            }
            if total.web_search_requests >= max_rounds || rounds >= MAX_HOSTED_ROUNDS {
                turn.tools.retain(|tool| tool.name != search::TOOL_NAME);
                if turn.tools.is_empty() { turn.tool_choice = ToolChoice::Auto; }
            }
            if client_calls {
                // The client must run its own calls before the model continues
                // (checked first: runnable calls must end with tool_use).
                yield Ok(TurnEvent::Usage { usage: total });
                yield Ok(TurnEvent::Done { stop: StopReason::ToolUse });
                return;
            }
            if rounds >= MAX_HOSTED_ROUNDS {
                yield Ok(TurnEvent::Usage { usage: total });
                yield Ok(TurnEvent::Done { stop: StopReason::PauseTurn });
                return;
            }
            stream = match backend.start(turn.clone()).await {
                Ok(stream) => stream,
                Err(error) => { yield Err(error); return; }
            };
        }
    })))
}

/// Tool calls end only once the turn's stop reason is known. `ToolCallEnd`
/// events are held until `Done` and released just before it when the turn
/// ended normally (`EndTurn`, `ToolUse`, `StopSequence`, `PauseTurn`). On any
/// other end (`MaxTokens`, `ContentFilter`, `Refusal`, `Cancelled`, an error
/// event, or the stream ending without `Done`, which is what a dropped upstream
/// connection looks like) the held ends are dropped. A front end therefore
/// treats any tool call still open at the end as cut off (incomplete) and never
/// reports it as complete or runnable. This is conservative: a call that did
/// finish earlier in a `MaxTokens` turn is also left open, so the client
/// re-asks rather than running a call from a turn that was cut short.
pub(super) fn seal_tool_calls(mut inner: TurnStream) -> TurnStream {
    Box::pin(async_stream::stream! {
        let mut ends = Vec::new();
        while let Some(event) = inner.next().await {
            match event {
                Ok(TurnEvent::ToolCallEnd { index }) => ends.push(index),
                Ok(TurnEvent::Done { stop }) => {
                    let normal = matches!(stop, StopReason::EndTurn | StopReason::ToolUse
                        | StopReason::StopSequence { .. } | StopReason::PauseTurn);
                    if normal {
                        for index in ends.drain(..) { yield Ok(TurnEvent::ToolCallEnd { index }); }
                    }
                    yield Ok(TurnEvent::Done { stop });
                    return;
                }
                Err(error) => { yield Err(error); return; }
                other => yield other,
            }
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::stream;

    async fn sealed(events: Vec<TurnEvent>) -> Vec<TurnEvent> {
        seal_tool_calls(Box::pin(stream::iter(events.into_iter().map(Ok)))).map(Result::unwrap).collect().await
    }

    #[tokio::test]
    async fn tool_call_ends_wait_for_the_stop_reason() {
        let call = |stop| vec![
            TurnEvent::ToolCallStart { index: 0, id: "c".into(), name: "f".into() },
            TurnEvent::ToolCallDelta { index: 0, arguments: "{\"a\":".into() },
            TurnEvent::ToolCallEnd { index: 0 },
            TurnEvent::Usage { usage: Usage::default() },
            TurnEvent::Done { stop },
        ];
        let cut = sealed(call(StopReason::MaxTokens)).await;
        assert!(!cut.iter().any(|e| matches!(e, TurnEvent::ToolCallEnd { .. })), "truncated call stays open");
        let done = sealed(call(StopReason::ToolUse)).await;
        assert!(matches!(done[done.len() - 2], TurnEvent::ToolCallEnd { index: 0 }));
        assert!(matches!(done[done.len() - 3], TurnEvent::Usage { .. }), "ends move to just before Done");
        for stop in [StopReason::ContentFilter, StopReason::Refusal, StopReason::Cancelled] {
            assert!(!sealed(call(stop)).await.iter().any(|e| matches!(e, TurnEvent::ToolCallEnd { .. })));
        }
    }

    #[tokio::test]
    async fn eof_or_error_without_done_leaves_calls_open() {
        let mut events = vec![
            TurnEvent::ToolCallStart { index: 0, id: "c".into(), name: "f".into() },
            TurnEvent::ToolCallDelta { index: 0, arguments: "{}".into() },
            TurnEvent::ToolCallEnd { index: 0 },
        ];
        let eof = sealed(events.clone()).await;
        assert_eq!(eof.len(), 2, "dropped connection: no end, no Done");
        events.push(TurnEvent::Done { stop: StopReason::ToolUse });
        let mut with_error: Vec<Result<TurnEvent, GatewayError>> = events[..3].iter().cloned().map(Ok).collect();
        with_error.push(Err(GatewayError::upstream("connection reset")));
        let out: Vec<_> = seal_tool_calls(Box::pin(stream::iter(with_error))).collect().await;
        assert!(out.iter().all(|e| !matches!(e, Ok(TurnEvent::ToolCallEnd { .. }))));
        assert!(matches!(out.last(), Some(Err(_))));
    }
}

#[cfg(test)]
mod hosted_tests {
    use super::*;
    use crate::gateway::{models::ModelMap, search::{SearchHit, SearchProvider, SearchQuery}, testing::Scripted, turn::WebSearchSpec, Gateway};
    use futures::future::BoxFuture;

    struct Fixed;
    impl SearchProvider for Fixed {
        fn name(&self) -> &str { "fixed" }
        fn search(&self, _: SearchQuery) -> BoxFuture<'static, Result<Vec<SearchHit>, GatewayError>> {
            Box::pin(async { Ok(vec![SearchHit { url: "https://example.org".into(), title: "t".into(), content: "c".into(), published: None }]) })
        }
    }

    fn search_round(id: &str) -> Vec<TurnEvent> {
        vec![TurnEvent::ToolCallStart { index: 0, id: id.into(), name: search::TOOL_NAME.into() },
            TurnEvent::ToolCallDelta { index: 0, arguments: "{\"query\":\"q\"}".into() },
            TurnEvent::ToolCallEnd { index: 0 }, TurnEvent::Done { stop: StopReason::ToolUse }]
    }

    /// Live Claude Code run (2026-10-09): its WebSearch subagent forces the
    /// search tool; repeating the forced choice on every continuation looped
    /// for ~300 rounds. Forcing applies to the first call; `max_uses` bounds
    /// searches; the tool is withdrawn once they're spent.
    #[tokio::test]
    async fn forced_search_relaxes_and_max_uses_withdraws_the_tool() {
        let mut scripts: Vec<Vec<TurnEvent>> = (0..3).map(|i| search_round(&format!("s{i}"))).collect();
        scripts.push(vec![TurnEvent::TextDelta { text: "answer".into() }, TurnEvent::Done { stop: StopReason::EndTurn }]);
        let backend = Scripted::new(scripts);
        let gateway = Arc::new(Gateway::new(Arc::new(backend.clone()), ModelMap::single("served-model")).with_search(Arc::new(Fixed)));
        let turn = TurnRequest { requested_model: "served-model".into(), tool_choice: ToolChoice::Named { name: search::TOOL_NAME.into() },
            hosted: crate::gateway::turn::HostedTools { web_search: Some(WebSearchSpec { name: "web_search".into(), max_uses: Some(2), ..Default::default() }) },
            ..Default::default() };
        let events: Vec<TurnEvent> = gateway.run(turn).await.unwrap().map(Result::unwrap).collect().await;
        assert!(matches!(events.last(), Some(TurnEvent::Done { stop: StopReason::EndTurn })));
        let seen = backend.turns();
        assert_eq!(seen.len(), 4);
        assert!(matches!(seen[0].tool_choice, ToolChoice::Named { .. }));
        assert!(seen[1..].iter().all(|t| t.tool_choice == ToolChoice::Auto), "forcing applies to the first call only");
        assert!(seen[1].tools.iter().any(|t| t.name == search::TOOL_NAME));
        assert!(seen[3].tools.iter().all(|t| t.name != search::TOOL_NAME), "tool withdrawn after max_uses");
        let usage = events.iter().rev().find_map(|e| if let TurnEvent::Usage { usage } = e { Some(*usage) } else { None }).unwrap();
        assert_eq!(usage.web_search_requests, 2, "third search answered max_uses_exceeded, not executed");
    }
}
