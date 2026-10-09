//! The backend seam: one trait, implemented by the upstream HTTP backend now
//! and by the in-process engine (phase B).
use futures::future::BoxFuture;
use futures::stream::BoxStream;
use serde::Serialize;

use super::error::GatewayError;
use super::turn::{TurnEvent, TurnRequest};

/// A running turn. Dropping the stream cancels the turn: backends must stop
/// generating (close the upstream connection, release the engine slot).
pub type TurnStream = BoxStream<'static, Result<TurnEvent, GatewayError>>;

/// What a backend can do, so front ends can refuse clearly instead of failing
/// deep inside a turn.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize)]
pub struct BackendCapabilities {
    pub vision: bool,
    pub audio_in: bool,
    /// Spoken output. No backend has it yet (the TTS seam, phase C).
    pub audio_out: bool,
    pub reasoning: bool,
    /// Exact prompt token counts (`count_tokens`); otherwise an estimate.
    pub exact_token_count: bool,
    /// Engine KV operations (pin/evict/mark, prefix forks). Phase B.
    pub kv_hooks: bool,
}

/// A model the backend serves.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ModelInfo {
    pub id: String,
    pub context_tokens: Option<u32>,
    pub max_output_tokens: Option<u32>,
    pub owned_by: String,
}

pub trait Backend: Send + Sync + 'static {
    /// Short name for logs and `/v1/models` metadata (`upstream`, `engine`).
    fn name(&self) -> &str;
    fn capabilities(&self) -> BackendCapabilities;
    /// Models this backend serves; the first is the default alias target.
    fn models(&self) -> Vec<ModelInfo>;
    /// Start one turn. Errors before the first event (bad request, upstream
    /// refused) come back here so front ends can answer with an HTTP error
    /// instead of a broken stream.
    fn start(&self, turn: TurnRequest) -> BoxFuture<'static, Result<TurnStream, GatewayError>>;
    /// Prompt tokens `turn` would consume, without generating.
    fn count_tokens(&self, turn: TurnRequest) -> BoxFuture<'static, Result<u32, GatewayError>>;
}

/// Rough token estimate (bytes / 4 over the serialized turn) for backends
/// without a tokenizer endpoint. Never used for limits, only for
/// `count_tokens` answers that clients use to decide when to compact.
pub fn estimate_tokens(turn: &TurnRequest) -> u32 {
    let mut bytes = turn.system.as_ref().map_or(0, String::len);
    bytes += serde_json::to_string(&turn.items).map_or(0, |s| s.len());
    bytes += serde_json::to_string(&turn.tools).map_or(0, |s| s.len());
    u32::try_from(bytes.div_ceil(4)).unwrap_or(u32::MAX)
}
