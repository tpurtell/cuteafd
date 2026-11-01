//! Incremental tool-call argument streaming, shared by every family's parser.
//!
//! A parser that streams emits `ToolCallBegin`, `ToolCall { name, "" }` and each
//! `ToolArgumentsDelta` as it arrives, while still assembling the whole call so
//! `close()` validates it exactly as the hold path does. The assembled call is
//! byte-identical to the non-streamed one.
//!
//! The knob is `CUTEAFD_STREAM_TOOLCALL_ARGS=1`, read only for streamed
//! requests: a non-streamed response always buffers, so a call cut at
//! `max_tokens` never returns partial JSON in `arguments`.
//!
//! A call whose name is unreadable is never streamed as a tool call (the name
//! check runs before `ToolCall`); it is returned as content, as in hold mode.
//! A call that streams and then turns out unreadable cannot be retracted: it is
//! counted so the finish is `tool_calls`, and the validator rejects its
//! assembled arguments with a clean 400/SSE error rather than a 500.
use deepseek_recipe::stream::OutputChunk;
use deepseek_recipe_core::tools::ToolDefinition;

/// Whether to stream tool-call argument deltas for this request.
///
/// `streaming` is the request's own `stream` flag: a non-streamed request
/// always buffers, whatever the environment says.
pub fn stream_toolcall_args_enabled(streaming: bool) -> bool {
    streaming && matches!(std::env::var("CUTEAFD_STREAM_TOOLCALL_ARGS").as_deref(), Ok("1"))
}

/// Whether a call's name may be emitted as a `ToolCall`.
///
/// A name is readable when it is non-empty, carries no markup, and — when the
/// request declared tools — is one of them. Checking before the `ToolCall`
/// chunk means a call with an unreadable name is never streamed as a tool call;
/// it is returned as content instead, exactly as in hold mode.
pub fn name_is_readable(name: &str, tools: Option<&[ToolDefinition]>) -> bool {
    !name.is_empty()
        && !name.contains(['<', '>', '\n'])
        && tools.map_or(true, |tools| tools.iter().any(|tool| tool.name == name))
}

/// Emits a call's chunks as they arrive when streaming is on.
///
/// Every method is a no-op when streaming is off, so a parser can call it
/// unconditionally and keep the hold path unchanged.
#[derive(Debug, Clone, Copy, Default)]
pub struct ArgStream {
    enabled: bool,
}

impl ArgStream {
    pub fn new(enabled: bool) -> Self {
        Self { enabled }
    }

    pub fn enabled(&self) -> bool {
        self.enabled
    }

    /// The call opened: emit `ToolCallBegin` now when streaming.
    pub fn begin(&self, out: &mut Vec<OutputChunk>) {
        if self.enabled {
            out.push(OutputChunk::ToolCallBegin);
        }
    }

    /// The call's name is known and readable: emit `ToolCall` now when
    /// streaming.
    pub fn call(&self, name: &str, out: &mut Vec<OutputChunk>) {
        if self.enabled {
            out.push(OutputChunk::ToolCall { tool_name: name.to_owned(), arguments: String::new() });
        }
    }

    /// An argument fragment arrived: emit it now when streaming.
    pub fn delta(&self, content: &str, out: &mut Vec<OutputChunk>) {
        if self.enabled && !content.is_empty() {
            out.push(OutputChunk::ToolArgumentsDelta { content: content.to_owned() });
        }
    }
}
