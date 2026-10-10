//! Credentials never survive; media become `{"$media":{sha256,mime,bytes}}`
//! references, with the decoded bytes handed to the media store.
use base64::Engine;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};

/// One decoded media blob found while redacting.
pub(crate) struct Media {
    pub sha256: String,
    pub mime: String,
    pub bytes: Vec<u8>,
}

fn credential(name: &str) -> bool {
    matches!(
        name.to_ascii_lowercase().as_str(),
        "headers"
            | "authorization"
            | "x-api-key"
            | "api_key"
            | "api-key"
            | "apikey"
            | "cookie"
            | "set-cookie"
            | "proxy-authorization"
            | "x-goog-api-key"
            | "client_secret"
    )
}

fn decode(data: &str) -> Vec<u8> {
    let data = data.trim();
    base64::engine::general_purpose::STANDARD
        .decode(data)
        .or_else(|_| base64::engine::general_purpose::URL_SAFE.decode(data))
        .or_else(|_| base64::engine::general_purpose::STANDARD_NO_PAD.decode(data))
        .unwrap_or_else(|_| data.as_bytes().to_vec())
}

fn base64ish(s: &str) -> bool {
    s.len() > 65536
        && s.bytes()
            .filter(|b| b.is_ascii_alphanumeric() || matches!(b, b'+' | b'/' | b'=' | b'-' | b'_'))
            .count()
            * 100
            / s.len()
            >= 95
}

pub(crate) struct Redactor {
    pub media: Vec<Media>,
}

impl Redactor {
    pub fn new() -> Self {
        Self { media: Vec::new() }
    }
    fn reference(&mut self, mime: &str, bytes: Vec<u8>) -> Value {
        let sha256 = format!("{:x}", Sha256::digest(&bytes));
        let r = json!({"$media":{"sha256":sha256,"mime":mime,"bytes":bytes.len()}});
        if !self.media.iter().any(|m| m.sha256 == sha256) {
            self.media.push(Media { sha256, mime: mime.to_owned(), bytes });
        }
        r
    }
    pub fn redact(&mut self, value: &mut Value) {
        match value {
            Value::String(s) if s.starts_with("data:") => {
                let (meta, body) = s[5..].split_once(',').unwrap_or(("", ""));
                let mime = meta.split(';').next().filter(|m| !m.is_empty()).unwrap_or("application/octet-stream").to_owned();
                let bytes = if meta.ends_with(";base64") { decode(body) } else { body.as_bytes().to_vec() };
                *value = self.reference(&mime, bytes);
            }
            Value::String(s) if base64ish(s) => {
                let bytes = decode(s);
                *value = self.reference("application/octet-stream", bytes);
            }
            Value::Array(a) => {
                a.retain(|v| {
                    !v.as_array()
                        .and_then(|a| a.first())
                        .and_then(Value::as_str)
                        .is_some_and(credential)
                });
                for v in a {
                    self.redact(v);
                }
            }
            Value::Object(o) => {
                // Some clients wrap a body with headers; never retain the wrapper.
                o.retain(|key, _| !credential(key));
                let kind = o.get("type").and_then(Value::as_str).unwrap_or("").to_owned();
                // Media payloads sit in a `data`/`audio`/`delta` string beside a type or format.
                let mime = if let Some(m) = o.get("media_type").and_then(Value::as_str) {
                    Some(m.to_owned())
                } else if let Some(f) = o.get("format").and_then(Value::as_str) {
                    Some(format!("audio/{f}"))
                } else if matches!(kind.as_str(), "base64" | "audio" | "input_audio" | "image") {
                    Some(if kind == "image" { "image/unknown" } else { "audio/unknown" }.to_owned())
                } else if kind == "input_audio_buffer.append" || kind.contains("audio.delta") {
                    Some("audio/unknown".to_owned())
                } else {
                    None
                };
                if let Some(mime) = mime {
                    for field in ["data", "audio", "delta"] {
                        let take = matches!(o.get(field), Some(Value::String(s)) if field == "data" || s.len() > 64);
                        if take {
                            if let Some(Value::String(data)) = o.remove(field) {
                                let r = self.reference(&mime, decode(&data));
                                o.insert(field.into(), r);
                            }
                        }
                    }
                }
                for (key, v) in o.iter_mut() {
                    if key != "$media" {
                        self.redact(v);
                    }
                }
            }
            _ => {}
        }
    }
}

pub(crate) fn extension(mime: &str) -> &'static str {
    match mime.to_ascii_lowercase().as_str() {
        "image/png" => "png",
        "image/jpeg" | "image/jpg" => "jpg",
        "image/gif" => "gif",
        "image/webp" => "webp",
        "image/bmp" => "bmp",
        "image/svg+xml" => "svg",
        "audio/wav" | "audio/x-wav" | "audio/wave" => "wav",
        "audio/mpeg" | "audio/mp3" => "mp3",
        "audio/ogg" | "audio/opus" => "ogg",
        "audio/flac" => "flac",
        "audio/webm" => "webm",
        "audio/pcm16" | "audio/pcm" | "audio/g711_ulaw" | "audio/g711_alaw" => "pcm",
        "application/pdf" => "pdf",
        "text/plain" => "txt",
        _ => "bin",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn credentials_media_and_audio() {
        let mut r = Redactor::new();
        let mut v = json!({
            "headers":{"Authorization":"RAW"}, "X-Api-Key":"RAW", "pairs":[["AUTHORIZATION","RAW"]],
            "messages":[{"role":"user","content":[
                {"type":"image_url","image_url":{"url":"data:image/png;base64,aGVsbG8="}},
                {"type":"input_audio","input_audio":{"data":"aGVsbG8=","format":"wav"}},
                {"type":"image","source":{"type":"base64","media_type":"image/jpeg","data":"aGVsbG8="}},
                {"type":"text","text":"keep"}]}],
            "big":"YQ==".repeat(20000)});
        r.redact(&mut v);
        let text = v.to_string();
        assert!(!text.contains("RAW") && !text.contains("aGVsbG8"));
        let parts = &v["messages"][0]["content"];
        assert_eq!(parts[0]["image_url"]["url"]["$media"]["mime"], "image/png");
        assert_eq!(parts[0]["image_url"]["url"]["$media"]["bytes"], 5);
        assert_eq!(parts[1]["input_audio"]["data"]["$media"]["mime"], "audio/wav");
        assert_eq!(parts[2]["source"]["data"]["$media"]["mime"], "image/jpeg");
        assert_eq!(parts[3]["text"], "keep");
        assert_eq!(v["pairs"], json!([]));
        // "hello" decoded three times (png, wav, jpeg share bytes) dedupes to one blob.
        assert_eq!(r.media.iter().filter(|m| m.bytes == b"hello").count(), 1);
        assert!(v["big"]["$media"].is_object());
    }
}
