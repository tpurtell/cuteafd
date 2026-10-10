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

/// Field names whose values are credentials, in any case and either separator.
pub(crate) fn credential(name: &str) -> bool {
    let n = name.to_ascii_lowercase().replace('-', "_");
    matches!(
        n.as_str(),
        "headers" | "authorization" | "authorization_token" | "x_api_key" | "api_key" | "apikey" | "cookie"
            | "set_cookie" | "proxy_authorization" | "x_goog_api_key" | "client_secret" | "access_token"
            | "refresh_token" | "id_token" | "password" | "passwd" | "secret" | "private_key"
    ) || n.ends_with("_token")
        || n.ends_with("_secret")
        || n.ends_with("_api_key")
        || n.ends_with("_password")
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

/// The media type of decoded bytes, from their magic number.
fn sniff(bytes: &[u8]) -> Option<&'static str> {
    let starts = |m: &[u8]| bytes.starts_with(m);
    Some(if starts(b"\x89PNG") {
        "image/png"
    } else if starts(b"\xff\xd8\xff") {
        "image/jpeg"
    } else if starts(b"GIF8") {
        "image/gif"
    } else if starts(b"RIFF") && bytes.get(8..12) == Some(b"WEBP") {
        "image/webp"
    } else if starts(b"RIFF") && bytes.get(8..12) == Some(b"WAVE") {
        "audio/wav"
    } else if starts(b"ID3") || starts(b"\xff\xfb") || starts(b"\xff\xf3") {
        "audio/mpeg"
    } else if starts(b"OggS") {
        "audio/ogg"
    } else if starts(b"fLaC") {
        "audio/flac"
    } else if starts(b"%PDF") {
        "application/pdf"
    } else {
        return None;
    })
}

/// Part shapes whose `data` (or `audio`/`delta`) string is media bytes.
fn media_part(o: &serde_json::Map<String, Value>) -> Option<String> {
    let kind = o.get("type").and_then(Value::as_str).unwrap_or("");
    if let Some(m) = o.get("media_type").and_then(Value::as_str).filter(|_| kind == "base64" || o.contains_key("data")) {
        return Some(m.to_owned());
    }
    if let Some(f) = o.get("format").and_then(Value::as_str).filter(|_| o.contains_key("data")) {
        return Some(format!("audio/{f}"));
    }
    match kind {
        "input_audio" | "audio" | "output_audio" => Some("audio/unknown".into()),
        "image" | "input_image" => Some("image/unknown".into()),
        "input_audio_buffer.append" => Some("audio/unknown".into()),
        k if k.ends_with("audio.delta") => Some("audio/unknown".into()),
        _ => None,
    }
}

pub(crate) struct Redactor {
    pub media: Vec<Media>,
    /// Server secrets (API key, console secret) scrubbed wherever they appear.
    secrets: Vec<String>,
}

impl Redactor {
    pub fn new() -> Self {
        Self::with_secrets(&[])
    }
    pub fn with_secrets(secrets: &[String]) -> Self {
        Self { media: Vec::new(), secrets: secrets.iter().filter(|s| s.len() >= 8).cloned().collect() }
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
                if let Some(mime) = sniff(&bytes) {
                    *value = self.reference(mime, bytes);
                }
            }
            Value::String(s) => {
                for secret in &self.secrets {
                    if s.contains(secret.as_str()) {
                        *s = s.replace(secret.as_str(), "[REDACTED]");
                    }
                }
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
                if let Some(mime) = media_part(o) {
                    for field in ["data", "audio", "delta"] {
                        if let Some(Value::String(data)) = o.get(field) {
                            if field != "data" && data.len() <= 64 {
                                continue;
                            }
                            let bytes = decode(data);
                            let mime = if mime.ends_with("/unknown") { sniff(&bytes).map(str::to_owned).unwrap_or(mime.clone()) } else { mime.clone() };
                            let r = self.reference(&mime, bytes);
                            o.insert(field.into(), r);
                        }
                    }
                }
                // OpenAI `input_audio: {data, format}` nests its payload.
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
        let png = { use base64::Engine; base64::engine::general_purpose::STANDARD.encode([b"\x89PNG\r\n".as_slice(), &[7u8; 60000]].concat()) };
        v["png_blob"] = json!(png);
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
        assert!(v["big"].is_string(), "base64-looking text without a media magic number stays text");
        assert_eq!(v["png_blob"]["$media"]["mime"], "image/png");
    }
    #[test]
    fn wider_credentials_and_server_secrets() {
        let mut r = Redactor::with_secrets(&["sk-SERVER-KEY-1234".into(), "CONSOLE_SECRET_ABCDEF".into()]);
        let mut v = json!({"Authorization_Token":"X1","access_token":"X2","refresh-token":"X3","Password":"X4","SECRET":"X5",
            "github_token":"X6","aws_secret":"X7","OPENAI_API_KEY":"X8","nested":{"session_token":"X9"},
            "text":"please use sk-SERVER-KEY-1234 and CONSOLE_SECRET_ABCDEF", "token_count": 5,
            "data":"not media", "content":[{"type":"text","text":"a data field"}]});
        r.redact(&mut v);
        let s = v.to_string();
        for x in ["X1","X2","X3","X4","X5","X6","X7","X8","X9","sk-SERVER-KEY","CONSOLE_SECRET_ABCDEF"] {
            assert!(!s.contains(x), "{x} survived: {s}");
        }
        assert_eq!(v["token_count"], 5, "only *_token names are credentials");
        assert_eq!(v["data"], "not media", "data outside media shapes is kept");
        assert!(r.media.is_empty());
    }
}
