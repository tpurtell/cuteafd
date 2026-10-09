//! Opt-in payload capture handles. No headers are accepted by this interface.
use bytes::Bytes;
use std::sync::Arc;
#[derive(Clone)]
pub enum ResponsePayload { Object(Bytes), Deltas(Vec<Bytes>) }
#[derive(Clone)]
pub struct LogRecord { pub rid: String, pub ts_ms: i64, pub protocol: String, pub request: Bytes, pub response: ResponsePayload }
impl LogRecord {
    pub fn byte_len(&self) -> usize { self.request.len() + match &self.response { ResponsePayload::Object(b) => b.len(), ResponsePayload::Deltas(d) => d.iter().map(Bytes::len).sum() } }
}
pub trait LogSink: Send + Sync + 'static {
    fn enabled(&self) -> bool;
    fn record_log(&self, record: LogRecord);
}
/// Created only when enabled. Handler-owned buffers need no synchronization.
pub struct LogCapture { sink: Arc<dyn LogSink>, record: LogRecord, buffered: usize }
impl LogCapture {
    pub fn begin(sink: Arc<dyn LogSink>, rid: &str, ts_ms: i64, protocol: &str, body: &Bytes) -> Option<Self> {
        sink.enabled().then(|| Self { sink, buffered: 0, record: LogRecord { rid: rid.into(), ts_ms, protocol: protocol.into(), request: body.clone(), response: ResponsePayload::Deltas(Vec::with_capacity(32)) } })
    }
    pub fn delta(&mut self, delta: &Bytes) {
        if self.sink.enabled() && self.buffered < 1 << 20 {
            if let ResponsePayload::Deltas(d) = &mut self.record.response {
                self.buffered = self.buffered.saturating_add(delta.len());
                d.push(delta.clone());
                if self.buffered >= 1 << 20 { d.push(Bytes::from_static(br#"{"log_truncated":true}"#)); }
            }
        }
    }
    pub fn response(&mut self, response: &Bytes) { if self.sink.enabled() { self.record.response = ResponsePayload::Object(response.clone()); } }
}
impl Drop for LogCapture { fn drop(&mut self) { if self.sink.enabled() { self.sink.record_log(self.record.clone()); } } }
