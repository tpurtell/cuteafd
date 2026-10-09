//! Bounded audio input and 10ms energy VAD. Audio generation/transcription are
//! deliberately separate seams: accepting a buffer does not imply a codec model.
use super::{
    super::error::GatewayError,
    protocol::{id, invalid},
};
use base64::{engine::general_purpose::STANDARD, Engine};
use futures::future::BoxFuture;
use serde_json::Value;

pub trait Transcriber: Send + Sync {
    fn transcribe(
        &self,
        format: String,
        audio: Vec<u8>,
    ) -> BoxFuture<'static, Result<String, GatewayError>>;
}
pub trait Synthesizer: Send + Sync {
    fn synthesize(&self, text: String) -> BoxFuture<'static, Result<Vec<u8>, GatewayError>>;
}
const MAX_BYTES: usize = 24 * 1024 * 1024;
#[derive(Clone, Debug)]
pub struct Settings {
    pub format: String,
    pub vad: Option<Vad>,
    pub transcription: bool,
}
#[derive(Clone, Debug)]
pub struct Vad {
    pub threshold: f64,
    pub prefix_ms: u32,
    pub silence_ms: u32,
    pub create_response: bool,
    pub interrupt_response: bool,
    pub idle_timeout_ms: Option<u64>,
}
impl Settings {
    pub fn bytes_per_ms(&self) -> usize {
        if self.format == "pcm16" {
            48
        } else {
            8
        }
    }
}
pub fn settings(config: &Value, beta: bool) -> Result<Settings, GatewayError> {
    let input = if beta {
        config
    } else {
        &config["audio"]["input"]
    };
    let format = if beta {
        input["input_audio_format"].as_str().unwrap_or("pcm16")
    } else {
        match input["format"]["type"].as_str().unwrap_or("audio/pcm") {
            "audio/pcm" => {
                if input["format"]["rate"].as_u64().unwrap_or(24000) != 24000 {
                    return Err(invalid(
                        "audio.input.format.rate",
                        "PCM must be 24000 Hz mono",
                    ));
                }
                "pcm16"
            }
            "audio/pcmu" => "g711_ulaw",
            "audio/pcma" => "g711_alaw",
            _ => return Err(invalid("audio.input.format", "unsupported audio format")),
        }
    };
    if !["pcm16", "g711_ulaw", "g711_alaw"].contains(&format) {
        return Err(invalid("input_audio_format", "unsupported audio format"));
    }
    let td = &input["turn_detection"];
    let vad = if td.is_null() {
        None
    } else {
        if td["type"] == "semantic_vad" {
            return Err(GatewayError::unsupported(
                "semantic_vad requires a turn detection model; use server_vad or null",
            )
            .with_param("turn_detection.type"));
        }
        if td["type"] != "server_vad" {
            return Err(invalid(
                "turn_detection.type",
                "turn detection must be server_vad or null",
            ));
        }
        let threshold = td.get("threshold").map_or(Ok(0.5), |v| {
            v.as_f64()
                .ok_or_else(|| invalid("turn_detection.threshold", "threshold must be numeric"))
        })?;
        if !(0.0..=1.0).contains(&threshold) {
            return Err(invalid(
                "turn_detection.threshold",
                "threshold must be between 0 and 1",
            ));
        }
        let integer = |key: &str, default: u32, max: u32| -> Result<u32, GatewayError> {
            let n = td
                .get(key)
                .map_or(Some(default as u64), Value::as_u64)
                .ok_or_else(|| invalid(key, "must be a non-negative integer"))?;
            if n > max as u64 {
                return Err(invalid(key, "VAD duration exceeds supported limit"));
            }
            Ok(n as u32)
        };
        let boolean = |key: &str| -> Result<bool, GatewayError> {
            td.get(key).map_or(Ok(true), |v| {
                v.as_bool().ok_or_else(|| invalid(key, "must be a boolean"))
            })
        };
        let idle_timeout_ms =
            if let Some(timeout) = td.get("idle_timeout_ms").filter(|v| !v.is_null()) {
                let n = timeout.as_u64().ok_or_else(|| {
                    invalid("turn_detection.idle_timeout_ms", "must be an integer")
                })?;
                if !(6000..=30000).contains(&n) {
                    return Err(invalid(
                        "turn_detection.idle_timeout_ms",
                        "must be 6000..30000",
                    ));
                }
                Some(n)
            } else {
                None
            };
        Some(Vad {
            threshold,
            prefix_ms: integer("prefix_padding_ms", 300, 10000)?,
            silence_ms: integer("silence_duration_ms", 500, 10000)?,
            create_response: boolean("create_response")?,
            interrupt_response: boolean("interrupt_response")?,
            idle_timeout_ms,
        })
    };
    let trans = if beta {
        &config["input_audio_transcription"]
    } else {
        &input["transcription"]
    };
    if !trans.is_null() && (!trans.is_object() || !trans["model"].is_string()) {
        return Err(invalid(
            "input_audio_transcription",
            "transcription requires a model",
        ));
    }
    Ok(Settings {
        format: format.into(),
        vad,
        transcription: !trans.is_null(),
    })
}
pub fn decode(data: &str, format: &str) -> Result<Vec<u8>, GatewayError> {
    if data.len() > MAX_BYTES * 4 / 3 + 4 {
        return Err(invalid("audio", "audio buffer exceeds 24 MiB"));
    }
    let bytes = STANDARD
        .decode(data)
        .map_err(|_| invalid("audio", "audio must be valid base64"))?;
    if format == "pcm16" && bytes.len() % 2 != 0 {
        return Err(invalid(
            "audio",
            "PCM16 audio requires complete 16-bit samples",
        ));
    }
    Ok(bytes)
}
#[derive(Debug)]
pub enum Activity {
    Started {
        item_id: String,
        start_ms: u64,
    },
    Stopped {
        item_id: String,
        end_ms: u64,
        audio: Vec<u8>,
    },
}
#[derive(Default)]
pub struct Buffer {
    bytes: Vec<u8>,
    pending: Vec<u8>,
    timeline_ms: u64,
    speech: Option<String>,
    silence_ms: u32,
}
impl Buffer {
    pub fn is_empty(&self) -> bool {
        self.bytes.is_empty() && self.pending.is_empty()
    }
    pub fn speaking(&self) -> bool {
        self.speech.is_some()
    }
    /// Idle commits may contain less than the manual commit's 100ms minimum.
    pub fn drain_idle(&mut self, settings: &Settings) -> (String, Vec<u8>, u64, u64) {
        let end = self.timeline_ms + (self.pending.len() / settings.bytes_per_ms()) as u64;
        self.bytes.append(&mut self.pending);
        let start = end.saturating_sub((self.bytes.len() / settings.bytes_per_ms()) as u64);
        self.timeline_ms = end;
        let item_id = self.speech.take().unwrap_or_else(|| id("item"));
        self.silence_ms = 0;
        (item_id, std::mem::take(&mut self.bytes), start, end)
    }
    pub fn clear(&mut self) {
        self.bytes.clear();
        self.pending.clear();
        self.speech = None;
        self.silence_ms = 0;
    }
    pub fn commit(&mut self, settings: &Settings) -> Result<(String, Vec<u8>), GatewayError> {
        let len = self.bytes.len() + self.pending.len();
        if len < 100 * settings.bytes_per_ms() {
            return Err(invalid(
                "audio",
                "input_audio_buffer_commit_empty: at least 100ms of audio is required",
            ));
        }
        self.bytes.append(&mut self.pending);
        self.silence_ms = 0;
        Ok((
            self.speech.take().unwrap_or_else(|| id("item")),
            std::mem::take(&mut self.bytes),
        ))
    }
    pub fn append(
        &mut self,
        data: &str,
        settings: &Settings,
    ) -> Result<Vec<Activity>, GatewayError> {
        let bytes = decode(data, &settings.format)?;
        if self.bytes.len() + self.pending.len() + bytes.len() > MAX_BYTES {
            return Err(invalid(
                "audio",
                "audio buffer exceeds 24 MiB; commit or clear it",
            ));
        }
        let Some(vad) = &settings.vad else {
            self.bytes.extend(bytes);
            return Ok(vec![]);
        };
        self.pending.extend(bytes);
        let frame_bytes = settings.bytes_per_ms() * 10;
        let full = self.pending.len() / frame_bytes * frame_bytes;
        let frames: Vec<u8> = self.pending.drain(..full).collect();
        let mut events = Vec::new();
        for frame in frames.chunks_exact(frame_bytes) {
            let energy = energy(frame, &settings.format);
            if energy >= vad.threshold && energy > 0.0 {
                if self.speech.is_none() {
                    let item_id = id("item");
                    let prefix_ms = (self.bytes.len() / settings.bytes_per_ms()) as u64;
                    events.push(Activity::Started {
                        item_id: item_id.clone(),
                        start_ms: self.timeline_ms.saturating_sub(prefix_ms),
                    });
                    self.speech = Some(item_id);
                }
                self.silence_ms = 0;
            } else if self.speech.is_some() {
                self.silence_ms += 10;
            }
            self.bytes.extend_from_slice(frame);
            self.timeline_ms += 10;
            if self.speech.is_some() && self.silence_ms >= vad.silence_ms.max(10) {
                let item_id = self.speech.take().unwrap();
                events.push(Activity::Stopped {
                    item_id,
                    end_ms: self.timeline_ms,
                    audio: std::mem::take(&mut self.bytes),
                });
                self.silence_ms = 0;
            } else if self.speech.is_none() {
                let keep = vad.prefix_ms as usize * settings.bytes_per_ms();
                if self.bytes.len() > keep {
                    self.bytes.drain(..self.bytes.len() - keep);
                }
            }
        }
        Ok(events)
    }
}
fn energy(frame: &[u8], format: &str) -> f64 {
    let samples: Vec<i16> = if format == "pcm16" {
        frame
            .chunks_exact(2)
            .map(|b| i16::from_le_bytes([b[0], b[1]]))
            .collect()
    } else {
        frame
            .iter()
            .map(|b| {
                if format == "g711_ulaw" {
                    let u = !*b;
                    let magnitude = (((u & 15) as i32) << 3) + 132;
                    let magnitude = (magnitude << ((u >> 4) & 7)) - 132;
                    (if u & 128 != 0 { -magnitude } else { magnitude }) as i16
                } else {
                    let a = *b ^ 0x55;
                    let exponent = (a >> 4) & 7;
                    let mut magnitude = ((a & 15) as i32) << 4;
                    magnitude += if exponent == 0 { 8 } else { 264 };
                    if exponent > 1 {
                        magnitude <<= exponent - 1;
                    }
                    (if a & 128 == 0 { -magnitude } else { magnitude }) as i16
                }
            })
            .collect()
    };
    if samples.is_empty() {
        return 0.0;
    }
    (samples
        .iter()
        .map(|s| (*s as f64 / 32768.0).powi(2))
        .sum::<f64>()
        / samples.len() as f64)
        .sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn fragmented_vad_and_g711() {
        for (format, speech, silence) in [
            ("pcm16", vec![0xff, 0x7f], vec![0, 0]),
            ("g711_ulaw", vec![0x80], vec![0xff]),
            ("g711_alaw", vec![0xaa], vec![0xd5]),
        ] {
            let settings = Settings {
                format: format.into(),
                vad: Some(Vad {
                    threshold: 0.2,
                    prefix_ms: 20,
                    silence_ms: 30,
                    create_response: true,
                    interrupt_response: true,
                    idle_timeout_ms: None,
                }),
                transcription: false,
            };
            let mut buffer = Buffer::default();
            let mut events = vec![];
            let frames = [
                silence.repeat(settings.bytes_per_ms() * 20 / silence.len()),
                speech.repeat(settings.bytes_per_ms() * 100 / speech.len()),
                silence.repeat(settings.bytes_per_ms() * 30 / silence.len()),
            ]
            .concat();
            for chunk in frames.chunks(speech.len() * 3) {
                events.extend(buffer.append(&STANDARD.encode(chunk), &settings).unwrap());
            }
            assert!(matches!(&events[0], Activity::Started { start_ms: 0, .. }));
            assert!(matches!(&events[1], Activity::Stopped { end_ms: 150, .. }));
            assert_eq!(events.len(), 2);
        }
    }
}
