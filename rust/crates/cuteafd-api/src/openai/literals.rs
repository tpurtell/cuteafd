//! Keep caller text separate from the control tokens a chat template emits.
use cuteafd_loader::{LiteralSpans, LiteralTokens};
use serde_json::Value;

/// Each request chooses a marker absent from its whole JSON input. ASCII keeps
/// the marker intact through JSON tool-argument and template serialization.
pub(super) struct Shield {
    prefix: String,
    values: Vec<(String, String)>,
}

impl Shield {
    pub(super) fn new(body: &Value) -> Self {
        let mut input = body.to_string();
        for message in body["messages"].as_array().into_iter().flatten() {
            for call in message["tool_calls"].as_array().into_iter().flatten() {
                if let Some(text) = call["function"]["arguments"].as_str() {
                    if let Ok(value) = serde_json::from_str::<Value>(text) {
                        input.push_str(&value.to_string());
                    }
                }
            }
        }
        let mut prefix = "CUTEAFD_LITERAL_".to_owned();
        while input.contains(&prefix) {
            prefix.push('_');
        }
        Self {
            prefix,
            values: Vec::new(),
        }
    }

    fn text(&mut self, text: &mut String, tokens: &LiteralTokens, media_only: bool) {
        let mut start = 0;
        let mut out = String::new();
        for span in tokens.spans(text, media_only) {
            out.push_str(&text[start..span.start]);
            let marker = format!("{}{index}_END", self.prefix, index = self.values.len());
            out.push_str(&marker);
            self.values.push((marker, text[span.clone()].to_owned()));
            start = span.end;
        }
        if start != 0 {
            out.push_str(&text[start..]);
            *text = out;
        }
    }

    fn value(&mut self, value: &mut Value, tokens: &LiteralTokens, media_only: bool) {
        match value {
            Value::String(text) => self.text(text, tokens, media_only),
            Value::Array(items) => {
                for item in items {
                    self.value(item, tokens, media_only);
                }
            }
            Value::Object(fields) => {
                let old = std::mem::take(fields);
                for (mut key, mut value) in old {
                    self.text(&mut key, tokens, media_only);
                    self.value(&mut value, tokens, media_only);
                    fields.insert(key, value);
                }
            }
            _ => {}
        }
    }

    /// Shield an already-rendered payload, after JSON decoding and sorting.
    pub(super) fn literal(&mut self, text: &str, tokens: &LiteralTokens) -> String {
        let mut text = text.to_owned();
        self.text(&mut text, tokens, false);
        text
    }

    pub(super) fn messages(&mut self, body: &mut Value, tokens: &LiteralTokens) {
        for message in body
            .get_mut("messages")
            .and_then(Value::as_array_mut)
            .into_iter()
            .flatten()
        {
            // Assistant text is prior generated output, including malformed tool
            // envelopes returned as content. Preserve its control-token identity.
            // Media is never generated; a literal media marker must stay text.
            let assistant = message["role"] == "assistant";
            if let Some(content) = message.get_mut("content") {
                match content {
                    Value::String(text) => self.text(text, tokens, assistant),
                    Value::Array(parts) => {
                        for part in parts {
                            if part["type"] == "text" {
                                if let Some(Value::String(text)) = part.get_mut("text") {
                                    self.text(text, tokens, assistant);
                                }
                            }
                        }
                    }
                    _ => {}
                }
            }
            if let Some(reasoning) = message.get_mut("reasoning_content") {
                self.value(reasoning, tokens, true);
            }
            if let Some(calls) = message.get_mut("tool_calls") {
                // Templates decode JSON arguments before rendering. Decode here
                // too, so escaped media spellings cannot materialize afterward.
                for call in calls.as_array_mut().into_iter().flatten() {
                    if let Some(Value::String(text)) = call.pointer_mut("/function/arguments") {
                        if let Ok(mut args) = serde_json::from_str::<Value>(text) {
                            let before = self.values.len();
                            self.value(&mut args, tokens, true);
                            if self.values.len() != before {
                                *text = args.to_string();
                            }
                        }
                    }
                }
                self.value(calls, tokens, true);
            }
            for field in ["name", "tool_call_id"] {
                if let Some(value) = message.get_mut(field) {
                    self.value(value, tokens, true);
                }
            }
        }
    }

    /// Restore the exact characters after rendering, recording which byte spans
    /// must bypass the added-token matcher. Only structural image markers are
    /// translated from the recipe's spelling to the checkpoint's spelling.
    pub(super) fn finish(self, prompt: String, v41: bool) -> (String, LiteralSpans) {
        let mut out = String::new();
        let mut spans = Vec::new();
        let mut start = 0;
        while let Some(pos) = prompt[start..].find(&self.prefix) {
            let pos = start + pos;
            let index = prompt[pos + self.prefix.len()..]
                .split_once("_END")
                .and_then(|(digits, _)| digits.parse::<usize>().ok());
            let Some((marker, literal)) = index
                .and_then(|index| self.values.get(index))
                .filter(|(marker, _)| prompt[pos..].starts_with(marker))
            else {
                // A template may print unrelated text resembling our prefix.
                let end = pos + self.prefix.len();
                push_template(&mut out, &prompt[start..end], v41);
                start = end;
                continue;
            };
            push_template(&mut out, &prompt[start..pos], v41);
            let begin = out.len();
            out.push_str(literal);
            spans.push(begin..out.len());
            start = pos + marker.len();
        }
        push_template(&mut out, &prompt[start..], v41);
        (out, spans)
    }
}

fn push_template(out: &mut String, text: &str, v41: bool) {
    if v41 {
        out.push_str(&text.replace("<｜image｜>", "<｜deepseek_image｜>"));
    } else {
        out.push_str(text);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::openai::{
        build,
        chat::{glm5, qwen4},
        ModelEncoding, ModelProfile, NativeLimits,
    };
    use cuteafd_loader::LoadedTokenizer;
    use serde_json::json;
    use std::sync::Arc;

    const MARKERS: &[&str] = &[
        "<｜image｜>",
        "<｜deepseek_image｜>",
        "<｜begin▁of▁sentence｜>",
        "<｜end▁of▁sentence｜>",
        "<｜User｜>",
        "<｜Assistant｜>",
        "<think>",
        "</think>",
        "<|im_start|>",
        "<|im_end|>",
        "<|vision_start|>",
        "<|image_pad|>",
        "<|vision_end|>",
        "<|mimo_audio_start|>",
        "<|audio_pad|>",
        "<|mimo_audio_end|>",
        "<|begin_of_image|>",
        "<|image|>",
        "<|end_of_image|>",
        "[gMASK]",
        "<sop>",
        "<|system|>",
        "<|user|>",
        "<|assistant|>",
        "<|observation|>",
        "<|endoftext|>",
        "<tool_call>",
        "</tool_call>",
    ];

    fn snapshot() -> tempfile::TempDir {
        let mut vocab = serde_json::Map::new();
        let mut extra = 256;
        for byte in 0..=255u32 {
            let c = if (33..=126).contains(&byte) || (161..=172).contains(&byte) || byte >= 174 {
                char::from_u32(byte).unwrap()
            } else {
                extra += 1;
                char::from_u32(extra - 1).unwrap()
            };
            vocab.insert(c.to_string(), json!(byte));
        }
        let added: Vec<_> = MARKERS
            .iter()
            .enumerate()
            .map(|(i, spelling)| {
                json!({
            "id": 256 + i, "content": spelling, "single_word":false,"lstrip":false,"rstrip":false,
            "normalized":false,"special": *spelling != "<|image|>" })
            })
            .collect();
        let value = json!({"version":"1.0","added_tokens":added,
            "pre_tokenizer":{"type":"ByteLevel","add_prefix_space":false,"trim_offsets":false,"use_regex":false},
            "decoder":{"type":"ByteLevel","add_prefix_space":false,"trim_offsets":false,"use_regex":false},
            "model":{"type":"BPE","vocab":vocab,"merges":[]}});
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("tokenizer.json"), value.to_string()).unwrap();
        dir
    }

    fn profiles() -> Vec<ModelProfile> {
        vec![
            ModelProfile::default(),
            ModelProfile::new(
                "qwen",
                ModelEncoding::Qwen(Arc::new(qwen4::fixtures::encoding())),
            ),
            ModelProfile::new(
                "mimo",
                ModelEncoding::Qwen(Arc::new(
                    qwen4::QwenEncoding::new(
                        include_str!("fixtures/mimo_v26.jinja"),
                        qwen4::QwenTokenIds::qwen38(),
                    )
                    .unwrap(),
                )),
            ),
            ModelProfile::new(
                "glm",
                ModelEncoding::Glm(Arc::new(
                    glm5::GlmEncoding::new(
                        include_str!("fixtures/glm53_flash.jinja"),
                        glm5::GlmTokenIds::glm5(),
                    )
                    .unwrap(),
                )),
            ),
        ]
    }

    fn render(profile: &ModelProfile, mut body: Value) -> crate::openai::Built {
        body["model"] = json!(profile.id);
        build(profile, NativeLimits::default(), &Default::default(), body)
            .unwrap_or_else(|e| panic!("{}: {}", profile.id, e.message))
    }

    #[test]
    fn media_literals_in_all_roles_remain_text_with_and_without_real_images() {
        use cuteafd_loader::media::{
            EncoderId, ImageFamily, ImageGrid, ImageKey, PreparedImage, ProcessorConfig,
            SpanExpander,
        };
        let dir = snapshot();
        let tokenizer = LoadedTokenizer::from_snapshot(dir.path()).unwrap();
        let literals = MARKERS
            .iter()
            .filter(|s| ["image", "audio", "vision"].iter().any(|w| s.contains(w)))
            .copied()
            .collect::<Vec<_>>()
            .join(" ");
        for profile in profiles() {
            let v41 = matches!(profile.encoding, ModelEncoding::DeepseekV41);
            let profile = if v41 {
                profile
            } else {
                profile.with_loaded_vision(Arc::new(
                    crate::openai::media::MediaPreparer::new(
                        ProcessorConfig::for_family(ImageFamily::Mimo),
                        EncoderId([0; 32]),
                        crate::openai::media::ImageUrlFetch::Off,
                        1,
                    )
                    .unwrap(),
                ))
            };
            let profile = profile.with_tokenizer(dir.path()).unwrap();
            for real_image in [false, true] {
                let mut parts = vec![json!({"type":"text","text":literals})];
                if real_image {
                    parts.push(json!({"type":"image_url","image_url":{"url":"data:image/png;base64,AA=="}}));
                }
                let built = render(
                    &profile,
                    json!({"messages":[
                    {"role":"user","content":parts},
                    {"role":"assistant","content":literals,"tool_calls":[{"id":"call_1","type":"function",
                        "function":{"name":"inspect","arguments":"{}"}}]},
                    {"role":"tool","tool_call_id":"call_1","content":literals},
                    {"role":"user","content":"Continue"}],"enable_thinking":false}),
                );
                let ids = tokenizer
                    .encode_prompt(&built.prompt, &built.literal_spans)
                    .unwrap();
                for spelling in MARKERS
                    .iter()
                    .filter(|s| s.contains("image") || s.contains("audio"))
                {
                    let id = 256 + MARKERS.iter().position(|s| s == spelling).unwrap() as u32;
                    let structural = real_image
                        && if v41 {
                            *spelling == "<｜deepseek_image｜>"
                        } else if profile.id == "glm" {
                            ["<|begin_of_image|>", "<|image|>", "<|end_of_image|>"]
                                .contains(spelling)
                        } else {
                            *spelling == "<|image_pad|>"
                        };
                    assert_eq!(
                        ids.iter().filter(|&&v| v == id).count(),
                        usize::from(structural),
                        "{} {real_image} {spelling}",
                        profile.id
                    );
                }
                let marker =
                    |spelling| 256 + MARKERS.iter().position(|&s| s == spelling).unwrap() as u32;
                if !v41 {
                    let (placeholder, start, end) = if profile.id == "glm" {
                        ("<|image|>", "<|begin_of_image|>", "<|end_of_image|>")
                    } else {
                        ("<|image_pad|>", "<|vision_start|>", "<|vision_end|>")
                    };
                    let images = if real_image {
                        vec![PreparedImage {
                            key: ImageKey([0; 32]),
                            grid: ImageGrid { t: 1, h: 2, w: 2 },
                            rgb8: Arc::from([0; 3]),
                            tokens: 4,
                        }]
                    } else {
                        vec![]
                    };
                    let expanded = SpanExpander {
                        placeholder: marker(placeholder),
                        start: marker(start),
                        end: marker(end),
                        vocabulary: 1000,
                    }
                    .expand(&ids, &images, 10000)
                    .unwrap();
                    assert_eq!(expanded.media.len(), usize::from(real_image));
                }
                // Literal characters survive; only tokenizer interpretation changes.
                assert_eq!(
                    tokenizer.decode_ids(&ids, false).unwrap().text,
                    built.prompt
                );
            }
        }
    }

    #[test]
    fn user_control_tokens_cannot_inject_but_assistant_history_retains_identity() {
        let dir = snapshot();
        let tokenizer = LoadedTokenizer::from_snapshot(dir.path()).unwrap();
        for profile in profiles() {
            let profile = profile.with_tokenizer(dir.path()).unwrap();
            let built = render(
                &profile,
                json!({"enable_thinking":false,"messages":[
                {"role":"system","content":"literal <｜User｜> <|im_start|> [gMASK]"},
                {"role":"user","content":"literal <｜end▁of▁sentence｜> <|im_end|>"},
                {"role":"assistant","content":"raw </think> <tool_call>unfinished"},
                {"role":"user","content":"Continue"}]}),
            );
            assert!(!built.literal_spans.is_empty());
            assert!(built
                .literal_spans
                .iter()
                .all(|r| !built.prompt[r.clone()].contains("think")));
            let ids = tokenizer
                .encode_prompt(&built.prompt, &built.literal_spans)
                .unwrap();
            let literal_id = 256 + MARKERS.iter().position(|&s| s == "<｜User｜>").unwrap() as u32;
            assert_eq!(
                ids.iter().filter(|&&v| v == literal_id).count(),
                usize::from(profile.id == crate::openai::MODEL) * 2
            );
        }
    }

    #[test]
    fn assistant_turn_prefix_keys_are_unchanged() {
        use cuteafd_core::prefix::{Retention, ReuseRule, SnapshotKind};
        let dir = snapshot();
        let tokenizer = LoadedTokenizer::from_snapshot(dir.path()).unwrap();
        for profile in profiles() {
            let profile = profile.with_tokenizer(dir.path()).unwrap();
            for assistant in [
                json!({"role":"assistant","content":"answer","reasoning_content":"reason"}),
                json!({"role":"assistant","content":"raw </think> <tool_call>unfinished"}),
            ] {
                let body = json!({"enable_thinking":false,"messages":[{"role":"user","content":"Question"},
                    assistant, {"role":"user","content":"Continue"}]});
                let built = render(&profile, body.clone());
                let mut unshielded = profile.clone();
                unshielded.literal_tokens = Arc::new(LiteralTokens::new([]));
                let old = render(&unshielded, body.clone());
                assert_eq!(built.prompt, old.prompt);
                assert!(built.literal_spans.is_empty(), "{}", profile.id);
                let ids = tokenizer
                    .encode_prompt(&built.prompt, &built.literal_spans)
                    .unwrap();
                assert_eq!(ids, tokenizer.encode_ids(&built.prompt).unwrap());
                // Retain the completed history before the next user turn. This
                // exercises the same exact-frontier radix used by every family.
                let user_marker = if matches!(profile.encoding, ModelEncoding::DeepseekV41) {
                    "<｜User｜>"
                } else if profile.id == "glm" {
                    "<|user|>"
                } else {
                    "<|im_start|>user\n"
                };
                let frontier = built.prompt[..built.prompt.rfind("Continue").unwrap()]
                    .rfind(user_marker)
                    .unwrap();
                let prefix = tokenizer.encode_ids(&built.prompt[..frontier]).unwrap();
                assert!(ids.starts_with(&prefix), "{}", profile.id);
                let mut cache = Retention::with_rule(4, ReuseRule::EXACT);
                cache.bank_mut(SnapshotKind::Turn).insert(&prefix, "turn");
                let replay = render(&profile, serde_json::from_str(&body.to_string()).unwrap());
                let replay_ids = tokenizer
                    .encode_prompt(&replay.prompt, &replay.literal_spans)
                    .unwrap();
                assert_eq!(
                    cache.lookup_reusable(&replay_ids),
                    Some((prefix.len(), prefix.len(), &"turn")),
                    "{}",
                    profile.id
                );
            }
        }
    }

    #[test]
    fn generated_assistant_answer_roundtrip_hits_turn_end_prefix() {
        use cuteafd_core::prefix::{Retention, ReuseRule, SnapshotKind};
        let dir = snapshot();
        let tokenizer = LoadedTokenizer::from_snapshot(dir.path()).unwrap();
        for profile in profiles() {
            let profile = profile.with_tokenizer(dir.path()).unwrap();
            let first = render(
                &profile,
                json!({"enable_thinking":false,"messages":[{"role":"user","content":"Question"}]}),
            );
            // The visible answer emitted by the output parser is sent back as
            // assistant content; capture at its last generated (non-stop) token.
            let turn = tokenizer
                .encode_ids(&format!("{}answer", first.prompt))
                .unwrap();
            let mut cache = Retention::with_rule(4, ReuseRule::EXACT);
            cache
                .bank_mut(SnapshotKind::Turn)
                .insert(&turn, "generated turn");
            let next = render(
                &profile,
                json!({"enable_thinking":false,"messages":[{"role":"user","content":"Question"},
                {"role":"assistant","content":"answer"},{"role":"user","content":"Continue"}]}),
            );
            let ids = tokenizer
                .encode_prompt(&next.prompt, &next.literal_spans)
                .unwrap();
            assert_eq!(
                cache.lookup_reusable(&ids),
                Some((turn.len(), turn.len(), &"generated turn")),
                "{}",
                profile.id
            );
        }
    }

    #[test]
    fn ordinary_image_request_ids_match_unshielded_checkpoint_rendering() {
        use cuteafd_loader::media::{EncoderId, ImageFamily, ProcessorConfig};
        let dir = snapshot();
        let tokenizer = LoadedTokenizer::from_snapshot(dir.path()).unwrap();
        for profile in profiles() {
            let profile = if matches!(profile.encoding, ModelEncoding::DeepseekV41) {
                profile
            } else {
                profile.with_loaded_vision(Arc::new(
                    crate::openai::media::MediaPreparer::new(
                        ProcessorConfig::for_family(ImageFamily::Mimo),
                        EncoderId([0; 32]),
                        crate::openai::media::ImageUrlFetch::Off,
                        1,
                    )
                    .unwrap(),
                ))
            }
            .with_tokenizer(dir.path())
            .unwrap();
            let body = json!({"messages":[{"role":"user","content":[{"type":"text","text":"Describe"},
                {"type":"image_url","image_url":{"url":"data:image/png;base64,AA=="}}]}],"enable_thinking":false});
            let built = render(&profile, body.clone());
            let mut unshielded = profile.clone();
            unshielded.literal_tokens = Arc::new(LiteralTokens::new([]));
            let old = render(&unshielded, body);
            assert_eq!(built.prompt, old.prompt);
            assert!(built.literal_spans.is_empty());
            assert_eq!(
                tokenizer
                    .encode_prompt(&built.prompt, &built.literal_spans)
                    .unwrap(),
                tokenizer.encode_ids(&built.prompt).unwrap(),
                "{}",
                profile.id
            );
            assert!(!built.prompt.contains("<｜image｜>"));
        }
    }

    #[test]
    fn escaped_json_argument_media_is_shielded_before_template_decoding() {
        let dir = snapshot();
        let tokenizer = LoadedTokenizer::from_snapshot(dir.path()).unwrap();
        for profile in profiles() {
            let profile = profile.with_tokenizer(dir.path()).unwrap();
            let built = render(
                &profile,
                json!({"enable_thinking":false,"messages":[
                {"role":"user","content":"inspect"},
                {"role":"assistant","content":null,"tool_calls":[{"id":"call_1","type":"function",
                    "function":{"name":"inspect","arguments":r#"{"x":"\u003c|image|>"}"#}}]},
                {"role":"tool","tool_call_id":"call_1","content":"done"},
                {"role":"user","content":"Continue"}]}),
            );
            let ids = tokenizer
                .encode_prompt(&built.prompt, &built.literal_spans)
                .unwrap();
            let id = 256 + MARKERS.iter().position(|&s| s == "<|image|>").unwrap() as u32;
            assert!(!ids.contains(&id), "{}", profile.id);
            assert!(!built.literal_spans.is_empty(), "{}", profile.id);
        }
    }

    #[test]
    fn structural_audio_survives_literal_media_spans() {
        use cuteafd_core::AudioKey;
        use cuteafd_loader::media::{
            audio::{AudioGeometry, PreparedAudio},
            SpanExpander,
        };
        let dir = snapshot();
        let tokenizer = LoadedTokenizer::from_snapshot(dir.path()).unwrap();
        let tokens = LiteralTokens::from_snapshot(dir.path()).unwrap();
        for real_audio in [false, true] {
            let mut body = json!({"messages":[{"role":"user","content":"<|audio_pad|> <|mimo_audio_start|> <|mimo_audio_end|>"}]});
            let mut shield = Shield::new(&body);
            shield.messages(&mut body, &tokens);
            let mut prompt = body["messages"][0]["content"].as_str().unwrap().to_owned();
            if real_audio {
                prompt.push_str("<|mimo_audio_start|><|audio_pad|><|mimo_audio_end|>");
            }
            let (prompt, spans) = shield.finish(prompt, false);
            let ids = tokenizer.encode_prompt(&prompt, &spans).unwrap();
            let marker =
                |spelling| 256 + MARKERS.iter().position(|&s| s == spelling).unwrap() as u32;
            let config = json!({"audio_token_id":marker("<|audio_pad|>"),
                "audio_start_token_id":marker("<|mimo_audio_start|>"),"audio_end_token_id":marker("<|mimo_audio_end|>")});
            let audio = if real_audio {
                vec![PreparedAudio {
                    key: AudioKey([0; 32]),
                    pcm: Arc::from(vec![0.0; 24000]),
                    geometry: AudioGeometry::for_samples(24000).unwrap(),
                }]
            } else {
                vec![]
            };
            assert_eq!(
                SpanExpander::expand_media(&config, 1000, &ids, &[], &audio, 10000)
                    .unwrap()
                    .media
                    .len(),
                usize::from(real_audio)
            );
        }
    }

    #[test]
    fn sentinel_collisions_and_repeated_rendering_restore_exact_literals() {
        let mut body =
            json!({"messages":[{"role":"user","content":"CUTEAFD_LITERAL_0_END <|image|>"}]});
        let mut shield = Shield::new(&body);
        shield.messages(&mut body, &LiteralTokens::new(["<|image|>".to_owned()]));
        let text = body["messages"][0]["content"].as_str().unwrap();
        let (prompt, spans) = shield.finish(format!("{text}{text}"), false);
        assert_eq!(prompt, "CUTEAFD_LITERAL_0_END <|image|>".repeat(2));
        assert_eq!(spans.len(), 2);
    }
}

#[cfg(test)]
mod official_tests {
    use crate::openai::*;
    use tower::ServiceExt;

    #[tokio::test]
    #[ignore = "requires CUTEAFD_LITERAL_SNAPSHOT and CUTEAFD_LITERAL_FAMILY"]
    async fn official_media_literals_http_admission_and_count() {
        use base64::Engine;
        use cuteafd_loader::media::{EncoderId, ImageFamily, ProcessorConfig, SpanExpander};
        let snapshot =
            std::path::PathBuf::from(std::env::var_os("CUTEAFD_LITERAL_SNAPSHOT").unwrap());
        let family = std::env::var("CUTEAFD_LITERAL_FAMILY").unwrap();
        let encoding = match family.as_str() {
            "deepseek" => ModelEncoding::DeepseekV41,
            "glm" => ModelEncoding::Glm(Arc::new(
                glm5::GlmEncoding::from_snapshot(&snapshot).unwrap(),
            )),
            _ => ModelEncoding::Qwen(Arc::new(
                qwen4::QwenEncoding::from_snapshot(&snapshot).unwrap(),
            )),
        };
        let mut profile = ModelProfile::new("literal-test", encoding)
            .with_tokenizer(&snapshot)
            .unwrap();
        if family != "deepseek" {
            let image_family = match family.as_str() {
                "glm" => ImageFamily::GlmFlash,
                "qwen" => ImageFamily::Qwen,
                _ => ImageFamily::Mimo,
            };
            profile = profile.with_loaded_vision(Arc::new(
                media::MediaPreparer::new(
                    ProcessorConfig::from_snapshot(&snapshot, image_family).unwrap(),
                    EncoderId([0; 32]),
                    media::ImageUrlFetch::Off,
                    1,
                )
                .unwrap(),
            ));
        }
        let tokenizer = cuteafd_loader::LoadedTokenizer::from_snapshot(&snapshot).unwrap();
        let config: Value =
            serde_json::from_slice(&std::fs::read(snapshot.join("config.json")).unwrap()).unwrap();
        let url = format!(
            "data:image/png;base64,{}",
            base64::engine::general_purpose::STANDARD.encode(include_bytes!("fixtures/black.png"))
        );
        for real_image in [false, true] {
            let (tx, mut rx) = mpsc::channel::<NativeRequest>(1);
            let mut parts = vec![
                json!({"type":"text","text":"<｜image｜> <｜deepseek_image｜> <|image|> <|image_pad|> <|audio_pad|>"}),
            ];
            if real_image {
                parts.push(json!({"type":"image_url","image_url":{"url":url}}));
            }
            let body = json!({"model":profile.id,"enable_thinking":false,"max_tokens":1,"messages":[
                {"role":"user","content":parts},
                {"role":"assistant","content":"<｜image｜> <｜deepseek_image｜> <|image|> <|image_pad|>"},
                {"role":"user","content":"Continue"}]});
            let state = NativeState {
                queue: tx.clone(),
                limits: NativeLimits::default(),
                images: images::ImageDecoder::new(4),
                stats: Arc::new(Mutex::new(Value::Null)),
                tables: cuteafd_loader::MappedTableStatsReader::default(),
                admission: admission::Admission::new(4, std::time::Duration::from_secs(25)),
                profile: Arc::new(profile.clone()),
            };
            let built = build(
                &profile,
                NativeLimits::default(),
                &Default::default(),
                body.clone(),
            )
            .unwrap_or_else(|e| panic!("{}", e.message));
            let counted = if family == "deepseek" {
                Some(
                    count_prompt(&state, built, snapshot.clone())
                        .await
                        .unwrap_or_else(|e| panic!("{}", e.message)),
                )
            } else {
                None
            };
            let app = router_for_model(
                tx,
                NativeLimits::default(),
                Arc::new(Mutex::new(Value::Null)),
                std::time::Duration::from_secs(25),
                ConsoleHub::disabled(),
                profile.clone(),
            );
            let response = tokio::spawn(async move {
                app.oneshot(
                    axum::http::Request::post("/v1/chat/completions")
                        .header("content-type", "application/json")
                        .body(Body::from(body.to_string()))
                        .unwrap(),
                )
                .await
                .unwrap()
            });
            let job = tokio::time::timeout(std::time::Duration::from_secs(30), rx.recv())
                .await
                .unwrap()
                .unwrap();
            let ids = job.encode_prompt(&tokenizer).unwrap();
            let expanded_len = if family == "deepseek" {
                let expanded =
                    cuteafd_loader::V41VisionPrompt::expand(&ids, job.images, 100000).unwrap();
                assert_eq!(expanded.images.len(), usize::from(real_image));
                expanded.tokens.len()
            } else {
                let images = job.media.iter().map(|v| (**v).clone()).collect::<Vec<_>>();
                let expanded = SpanExpander::from_config(
                    &config,
                    config["vocab_size"]
                        .as_u64()
                        .or_else(|| config["text_config"]["vocab_size"].as_u64())
                        .unwrap() as u32,
                )
                .unwrap()
                .expand(&ids, &images, 100000)
                .unwrap();
                assert_eq!(expanded.media.len(), usize::from(real_image));
                expanded.tokens.len()
            };
            if let Some(counted) = counted {
                assert_eq!(counted as usize, expanded_len);
            }
            job.events
                .send(Ok(InferenceChunk::Ready {
                    system_fingerprint: None,
                    prompt_usage: PromptUsage {
                        prompt_tokens: expanded_len,
                        prompt_cache_hit_tokens: 0,
                    },
                }))
                .unwrap();
            job.events
                .send(Ok(InferenceChunk::Finish {
                    finish_reason: InferenceFinishReason::Stop,
                }))
                .unwrap();
            let response = response.await.unwrap();
            let status = response.status();
            let bytes = axum::body::to_bytes(response.into_body(), 1 << 20)
                .await
                .unwrap();
            assert_eq!(
                status,
                StatusCode::OK,
                "{}",
                String::from_utf8_lossy(&bytes)
            );
        }
    }
}
