//! Full-log payload capture. The serving path only clones `Bytes` frames
//! (refcounts); parsing, folding, redaction and storage run on the log writer.
//! No headers are accepted by this interface.
use bytes::Bytes;

/// Request frames held per capture. Larger bodies are dropped and counted.
pub const REQUEST_CAP: usize = 64 << 20;
/// Response frames held per capture; streams past this are marked truncated.
pub const RESPONSE_CAP: usize = 16 << 20;

/// What the handler returned, as the client saw it.
#[derive(Clone, Debug, Default)]
pub enum ResponsePayload {
    #[default]
    None,
    /// A complete JSON object (non-streaming bodies, WebSocket turns).
    Object(Bytes),
    /// Body frames in order; `sse` when the content type was an event stream.
    Frames { frames: Vec<Bytes>, sse: bool, truncated: bool },
}

/// One request's payloads plus the metadata record it was emitted with.
#[derive(Clone, Debug)]
pub struct LogRecord {
    pub meta: crate::usage::Record,
    pub request: Vec<Bytes>,
    pub request_truncated: bool,
    pub response: ResponsePayload,
}

impl LogRecord {
    pub fn byte_len(&self) -> usize {
        self.request.iter().map(Bytes::len).sum::<usize>()
            + match &self.response {
                ResponsePayload::None => 0,
                ResponsePayload::Object(b) => b.len(),
                ResponsePayload::Frames { frames, .. } => frames.iter().map(Bytes::len).sum(),
            }
    }
}

/// The full-log store as seen by the serving side.
pub trait LogSink: Send + Sync + 'static {
    /// One relaxed load; `false` means no capture state is created at all.
    fn enabled(&self) -> bool;
    /// Whether benchmark requests are kept (the usage `record_bench` setting).
    fn bench(&self) -> bool {
        false
    }
    fn record_log(&self, record: LogRecord);
}

/// Protocols whose payloads the full log keeps (never console or admin routes).
pub fn loggable(protocol: &str) -> bool {
    matches!(protocol, "chat" | "completions" | "messages" | "responses" | "realtime")
}

/// Accumulates body frames up to a cap, cloning refcounts only.
#[derive(Default)]
pub(crate) struct FrameTee {
    pub frames: Vec<Bytes>,
    pub bytes: usize,
    pub truncated: bool,
}

impl FrameTee {
    pub fn push(&mut self, data: &Bytes, cap: usize) {
        if self.truncated {
            return;
        }
        if self.bytes.saturating_add(data.len()) > cap {
            self.truncated = true;
            return;
        }
        self.bytes += data.len();
        self.frames.push(data.clone());
    }
}
