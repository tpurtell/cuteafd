//! Benchmark request policy derived from the active checkpoint renderer.
use crate::report::Configuration;
use serde_json::Value;

/// Apply at construction, before the request is recorded or sent. Both chat
/// requests and replay specifications use this policy (object/string forms).
pub fn thinking_off(configuration: &Configuration, mut body: Value) -> Value {
    if !configuration.supports_thinking_off() {
        body.as_object_mut().expect("benchmark request object").remove("thinking");
    }
    body
}

#[derive(Debug, thiserror::Error)]
#[error("checkpoint template does not support disabling thinking; benchmark request builder must use the template capability")]
pub struct UnsupportedThinkingOff;

/// The transport validates builders; it never rewrites their recorded request.
pub fn validate(configuration: &Configuration, body: &Value) -> Result<(), UnsupportedThinkingOff> {
    let is = |value: &Value, expected: &str| value.as_str().is_some_and(|s| s.eq_ignore_ascii_case(expected));
    let disabled = body.get("thinking").is_some_and(|v| is(v, "disabled") || is(&v["type"], "disabled"))
        || body.get("enable_thinking").and_then(Value::as_bool) == Some(false)
        || body.pointer("/chat_template_kwargs/enable_thinking").and_then(Value::as_bool) == Some(false)
        || body.pointer("/chat_template_kwargs/thinking").and_then(Value::as_bool) == Some(false)
        || body.get("reasoning_effort").is_some_and(|v| is(v, "none"))
        || body.pointer("/chat_template_kwargs/reasoning_effort").is_some_and(|v| is(v, "none"));
    if !configuration.supports_thinking_off() && disabled { Err(UnsupportedThinkingOff) } else { Ok(()) }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    /// Walk the constructors shared by every panel: speed/quality/sweep use
    /// common::plain; speed's multi-turn case uses context_load::body.
    fn builders(info: &crate::report::ServerInfo) -> Vec<Value> {
        use crate::{baseline, client::Client, panels::{common, context_load, lossless, reasoning}};
        use cuteafd_api::openai::probe::ProbeRecord;
        let mut client = Client::new("http://127.0.0.1:1", None, std::sync::Arc::default());
        client.configuration = info.configuration.clone();
        let record = ProbeRecord::default();
        let set = lossless::CaseSet::default_set();
        let messages = json!([{"role":"user","content":"test"}]);
        let mut bodies = vec![
            baseline::plain(&client, "test", 128),
            common::plain(&client, "test", 128),
            context_load::body(&info.configuration, messages.clone(), 128),
            context_load::request(&info.configuration, &messages, "test", "warm", 16, 0, 128, 0),
            context_load::request(&info.configuration, &messages, "test", "timed", 16, 0, 128, 512),
            baseline::replay(&info.configuration, "test", "test", 128, &record),
            lossless::evidence(&info.configuration, &set.cases[0], &record, &record),
        ];
        bodies.extend(reasoning::levels(info).into_iter().map(|(_, body)| body));
        bodies
    }

    #[test]
    fn all_panel_request_builders_honor_step5_renderer_capability() {
        let mut info = crate::report::ServerInfo { family: Some("step5".into()), ..Default::default() };
        info.configuration.supports_thinking_off = Some(false);
        for body in builders(&info) {
            validate(&info.configuration, &body).unwrap();
            assert!(!body.to_string().contains("disabled"), "{body}");
        }
        assert!(!crate::panels::reasoning::levels(&info).iter().any(|(label, _)| label == "off"));
        let absent = builders(&crate::report::ServerInfo::default());
        let mut supported = crate::report::ServerInfo::default();
        supported.configuration.supports_thinking_off = Some(true);
        let explicit = builders(&supported);
        assert_eq!(absent.len(), explicit.len());
        for (old, new) in absent.iter().zip(explicit.iter()) {
            assert_eq!(old.to_string(), new.to_string());
        }
        let expected = json!({"messages":[{"role":"user","content":"test"}], "max_tokens":128,
            "temperature":0, "thinking":{"type":"disabled"}});
        assert_eq!(explicit[0].to_string(), expected.to_string());
        assert_eq!(explicit[1].to_string(), expected.to_string());
        assert_eq!(explicit[2].to_string(), expected.to_string());
    }

    #[test]
    fn client_boundaries_reject_missed_builder_without_network_or_rewrite() {
        let mut client = crate::client::Client::new("http://127.0.0.1:1", None, std::sync::Arc::default());
        client.configuration.supports_thinking_off = Some(false);
        let body = json!({"thinking":{"type":"disabled"},"messages":[]});
        let before = body.to_string();
        let error = client.chat(body.clone(), None).unwrap_err();
        assert!(error.downcast_ref::<UnsupportedThinkingOff>().is_some());
        let error = client.media_probe(body.clone(), json!({})).unwrap_err();
        assert!(error.downcast_ref::<UnsupportedThinkingOff>().is_some());
        assert_eq!(body.to_string(), before);
    }

    #[test]
    fn boundary_rejects_each_disabled_thinking_spelling() {
        let configuration = Configuration { supports_thinking_off: Some(false), ..Default::default() };
        for body in [json!({"thinking":{"type":"DISABLED"}}), json!({"thinking":"disabled"}),
            json!({"enable_thinking":false}), json!({"chat_template_kwargs":{"enable_thinking":false}}),
            json!({"chat_template_kwargs":{"thinking":false}}), json!({"reasoning_effort":"NONE"}),
            json!({"chat_template_kwargs":{"reasoning_effort":"none"}})] {
            assert!(validate(&configuration, &body).is_err(), "{body}");
            validate(&Configuration::default(), &body).unwrap();
        }
    }

    #[test]
    fn off_capable_requests_are_byte_identical_and_always_on_is_explicit() {
        for body in [json!({"messages":[],"thinking":{"type":"disabled"},"temperature":0}),
            json!({"text":"probe","thinking":"disabled","prompt_ids":[1,2]})] {
            let original = body.to_string();
            assert_eq!(thinking_off(&Configuration::default(), body.clone()).to_string(), original);
            let configuration = Configuration { supports_thinking_off: Some(false), ..Default::default() };
            assert!(validate(&configuration, &body).is_err());
            let built = thinking_off(&configuration, body);
            assert!(built.get("thinking").is_none());
            validate(&configuration, &built).unwrap();
            assert_eq!(configuration.thinking_label(), "thinking always on (template)");
        }
    }
}
