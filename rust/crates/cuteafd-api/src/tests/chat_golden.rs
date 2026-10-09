//! Byte-identity gate for `/v1/chat/completions` (work/api-gateway): the
//! gateway must not change this route. For a fixed request set and a fixed
//! synthetic worker, the engine-facing request (prompt, limits, sampling, stop
//! ids) and the full response bytes are hashed per encoding, with only
//! generated ids and timestamps normalized. The golden was recorded on
//! origin/work/p0 (ccc3fc6b) before any gateway change.
//!
//! Regenerate only for an intended chat-path change:
//! `CUTEAFD_CHAT_GOLDEN_PRINT=1 cargo test -p cuteafd-api chat_golden -- --nocapture`.
use std::sync::{Arc, Mutex};

use axum::body::{to_bytes, Body};
use deepseek_recipe::stream::{InferenceChunk, InferenceFinishReason, PromptUsage};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use tower::ServiceExt;

use crate::openai::chat::{glm5, qwen4};
use crate::openai::{router_for_model, ConsoleHub, ModelEncoding, ModelProfile, NativeLimits, NativeRequest};

const GOLDEN: &[(&str, &str)] = &[
    ("deepseek-ai/DeepSeek-V4.1-Flash", "540e9529107b0028eafa944da7c4cfcc8dc8fe7e1931f39f650d9e587e0f8d39"),
    ("test-dsv4", "19aabcbc9ad50e7b3696841bcb44e05d7fea3eb7290a93813fc26e37082baa93"),
    ("test-glm", "e2b1abc92c411f08459fdda575486fafcc37cca4ea252423855d3b13887f261b"),
    ("test-qwen", "49c07547dd60c7ca8f1d27752215ba2c4253443b1183bc17dc2be34d3ba2dcd4"),
];

fn profiles() -> Vec<ModelProfile> {
    vec![
        ModelProfile::default(),
        ModelProfile::new("test-dsv4", ModelEncoding::DeepseekV4),
        ModelProfile::new("test-glm", ModelEncoding::Glm(Arc::new(glm5::fixtures::encoding()))),
        ModelProfile::new("test-qwen", ModelEncoding::Qwen(Arc::new(qwen4::fixtures::encoding()))),
    ]
}

fn requests(model: &str) -> Vec<Value> {
    let tools = json!([{"type": "function", "function": {"name": "get_weather", "description": "Weather for a city",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}]);
    vec![
        json!({"model": model, "messages": [{"role": "user", "content": "2+2?"}], "max_tokens": 16}),
        json!({"model": model, "messages": [{"role": "system", "content": "Be terse."}, {"role": "user", "content": "2+2?"}],
            "max_tokens": 16, "stream": true, "stream_options": {"include_usage": true}}),
        json!({"model": model, "messages": [{"role": "user", "content": "Weather in Paris?"}], "tools": tools,
            "max_tokens": 64, "temperature": 0.7, "top_p": 0.9, "seed": 7, "stream": true}),
        json!({"model": model, "messages": [
            {"role": "user", "content": "Weather in Paris?"},
            {"role": "assistant", "content": null, "tool_calls": [{"id": "call_1", "type": "function",
                "function": {"name": "get_weather", "arguments": "{\"city\":\"Paris\"}"}}]},
            {"role": "tool", "tool_call_id": "call_1", "content": "Sunny, 21C"}],
            "tools": tools, "max_tokens": 64}),
        json!({"model": model, "messages": [{"role": "user", "content": "hi"}], "max_tokens": 8,
            "thinking": {"type": "disabled"}, "stream": true}),
        json!({"model": "wrong-model", "messages": [{"role": "user", "content": "hi"}]}),
        json!({"model": model, "messages": "not-a-list"}),
    ]
}

fn chunks() -> Vec<InferenceChunk> {
    vec![
        InferenceChunk::Ready { system_fingerprint: Some("fp-golden".into()),
            prompt_usage: PromptUsage { prompt_tokens: 12, prompt_cache_hit_tokens: 4 } },
        InferenceChunk::Text { content: "The answer".into(), content_tokens: 2 },
        InferenceChunk::Text { content: " is 4.".into(), content_tokens: 3 },
        InferenceChunk::Finish { finish_reason: InferenceFinishReason::Stop },
    ]
}

fn describe(job: &NativeRequest) -> String {
    // Greedy requests draw an unused clock-derived seed; pin it so the digest is stable.
    let sampling = if job.sampling.is_greedy() { job.sampling.with_seed(0) } else { job.sampling };
    format!("prompt={:?}\nmax_tokens={}\nsampling={:?}\nstop={:?}\nconstraint={}\nimages={} media={} audio={}",
        job.prompt, job.max_tokens, sampling, job.stop_token_ids, job.constraint.is_some(),
        job.images.len(), job.media.len(), job.audio.len())
}

/// Replace generated ids and clock values so only behaviour is compared.
fn normalize(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut rest = text;
    // Field-level pass over every JSON document embedded in the text.
    while let Some(start) = rest.find('{') {
        out.push_str(&rest[..start]);
        let mut stream = serde_json::Deserializer::from_str(&rest[start..]).into_iter::<Value>();
        match stream.next() {
            Some(Ok(mut value)) => {
                let used = stream.byte_offset();
                scrub(&mut value);
                out.push_str(&value.to_string());
                rest = &rest[start + used..];
            }
            _ => { out.push('{'); rest = &rest[start + 1..]; }
        }
    }
    out.push_str(rest);
    out
}

fn scrub(value: &mut Value) {
    match value {
        Value::Object(map) => {
            for (key, field) in map.iter_mut() {
                match key.as_str() {
                    "id" if field.is_string() => *field = json!("<id>"),
                    "created" if field.is_number() => *field = json!(0),
                    _ => scrub(field),
                }
            }
        }
        Value::Array(items) => items.iter_mut().for_each(scrub),
        _ => {}
    }
}

async fn run(profile: &ModelProfile, body: &Value) -> String {
    let (tx, mut rx) = tokio::sync::mpsc::channel::<NativeRequest>(1);
    let seen = Arc::new(Mutex::new(String::from("<not admitted>")));
    let witness = seen.clone();
    let worker = tokio::spawn(async move {
        if let Some(job) = rx.recv().await {
            *witness.lock().unwrap() = describe(&job);
            for chunk in chunks() { let _ = job.events.send(Ok(chunk)); }
        }
    });
    let app = router_for_model(tx, NativeLimits::default(), Arc::new(Mutex::new(Value::Null)),
        std::time::Duration::from_secs(25), ConsoleHub::disabled(), profile.clone());
    let response = app.oneshot(axum::http::Request::post("/v1/chat/completions")
        .header("content-type", "application/json").body(Body::from(body.to_string())).unwrap()).await.unwrap();
    let status = response.status();
    let content_type = response.headers().get("content-type").map(|v| v.to_str().unwrap_or("").to_string());
    let bytes = to_bytes(response.into_body(), 1 << 22).await.unwrap();
    worker.abort();
    let _ = worker.await;
    let seen = seen.lock().unwrap().clone();
    format!("status={status}\ncontent-type={content_type:?}\n{seen}\nbody={}\n", normalize(&String::from_utf8_lossy(&bytes)))
}

#[tokio::test]
async fn chat_completions_bytes_are_unchanged() {
    let print = std::env::var_os("CUTEAFD_CHAT_GOLDEN_PRINT").is_some();
    let mut failures = Vec::new();
    for profile in profiles() {
        let mut transcript = String::new();
        for body in requests(&profile.id) { transcript.push_str(&run(&profile, &body).await); }
        let digest = format!("{:x}", Sha256::digest(transcript.as_bytes()));
        if print { println!("(\"{}\", \"{digest}\"),\n----\n{transcript}----", profile.id); continue; }
        match GOLDEN.iter().find(|(id, _)| *id == profile.id) {
            Some((_, expected)) if *expected == digest => {}
            Some((_, expected)) => failures.push(format!("{}: {digest} != golden {expected}\n{transcript}", profile.id)),
            None => failures.push(format!("{}: no golden (digest {digest})", profile.id)),
        }
    }
    assert!(failures.is_empty(), "chat completions behaviour changed:\n{}", failures.join("\n"));
}
