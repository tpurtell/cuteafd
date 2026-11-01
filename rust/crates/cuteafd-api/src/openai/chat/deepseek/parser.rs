//! DeepSeek V4/V4.1 output parser: the DSML tool-call grammar, emitted as
//! protocol-neutral [`OutputChunk`]s through the same [`TextParser`] contract
//! GLM and Qwen use ([`GlmStreamProcessor`]).
//!
//! ```text
//! <｜DSML｜ calls>
//! <｜DSML｜ invoke name="search">
//! <｜DSML｜ parameter name="query" string="true">cats</｜DSML｜ parameter>
//! </｜DSML｜ invoke>
//! </｜DSML｜ calls>
//! ```
//!
//! The grammar itself is `deepseek_recipe`'s own incremental state machine
//! (`state_machine::{StateMachine, OutputAction}`), reused here rather than
//! re-implemented: [`DeepseekOutputParser`] drives it and translates its
//! actions to [`OutputChunk`]s exactly as `deepseek_recipe::stream::processor`
//! does, so a well-formed response parses identically.
//!
//! FR-G.14 (hold a call until it closes; return an unreadable one as content;
//! `finish_reason: tool_calls` only for a call that parsed; log each lost
//! call) is the one intended difference — the crate's processor drops a
//! malformed call and claims a tool call as soon as the name completes:
//!
//! - a call is **held** from `<｜DSML｜ invoke name="` until its closing
//!   `</｜DSML｜ invoke>`; a client never sees part of a call that turns out
//!   malformed (the SSE keepalive covers the wait);
//! - a call that never closes (truncated by `max_tokens`, or its name is empty
//!   or carries markup) is returned as content from its `<｜DSML｜` marker and
//!   logged, and does not count as a tool call;
//! - a call that *does* close is still lost when the machine skipped material
//!   inside it that is not part of the DSML skeleton (a malformed parameter
//!   envelope, an empty or unquoted parameter name), or when the arguments it
//!   assembled are not valid JSON. The first of those is invisible to a JSON
//!   check alone: the machine skips the unreadable bytes and goes on to emit a
//!   perfectly valid `{}`, so the call would otherwise be accepted with the
//!   argument silently missing. The skipped run is validated against the
//!   grammar's own delimiters ([`skip_is_structural`]), never per segment,
//!   because the machine splits a run wherever a partial match fails.
//!
//! The hold-and-return-as-content contract is adopted from glm53f-api's
//! `dialect/glm.rs` (Hugh Madden, MIT), which FR-G.14 applies to every
//! dialect. No donor source is reproduced: the code here is written against
//! `deepseek_recipe`'s own state machine, and the crate's processor is the
//! differential reference in the tests below.
use std::collections::VecDeque;

use deepseek_recipe::stream::OutputChunk;
use deepseek_recipe::stream::state_machine::{
    OutputAction, OutputActionSegment, ParsingOptions, StateMachine,
};

use crate::openai::chat::glm5::TextParser;
use crate::openai::chat::glm5::parser::GlmStop;
use crate::openai::chat::stream_args::{ArgStream, name_is_readable};

/// The DSML tool-call envelope opener; a call's content is returned from here.
const DSML_BEGIN: &str = "<｜DSML｜";

/// The DSML closing tag, shared by an envelope's own close and the block's.
const DSML_END: &str = "</｜DSML｜";

/// One DSML tool-call envelope: `<｜DSML｜ invoke name="…"> … </｜DSML｜ invoke>`,
/// the unit returned as content when a call cannot be read. The leading
/// ` calls>` block wrapper and the trailing ` invoke>` tail belong to the
/// machine, so the raw text is cut back to the last `<｜DSML｜` (opener) and up
/// to the matching close — never into a neighbouring call.
fn back_over_newlines(text: &str, mut index: usize) -> usize {
    while index > 0 && text.as_bytes()[index - 1] == b'\n' {
        index -= 1;
    }
    index
}

/// Where a call's envelope starts in `text`: its last `<｜DSML｜` marker, or the
/// end of the text when there is none (an empty envelope).
fn opener_index(text: &str) -> usize {
    match text.rfind(DSML_BEGIN) {
        Some(index) => back_over_newlines(text, index),
        None => text.len(),
    }
}

/// The only bytes a `Skip` run may carry inside a *well-formed* open call.
///
/// The DSML grammar (`deepseek_recipe`'s `state_machine`) reaches its
/// structural bytes through unmatched branches, which are its `Skip` action:
/// the `>` closing `name="…"`, the `<｜DSML｜`/`</｜DSML｜` tag bodies (the
/// latter also ends a non-string parameter's value), the ` parameter>` tail,
/// and a parameter's `string="` / `false">` type. Anything else skipped while a
/// call is open is junk the machine discarded — the shape that silently turns a
/// malformed parameter envelope into a valid `{}`.
const STRUCTURAL_SKIPS: [&str; 6] =
    ["<｜DSML｜", "</｜DSML｜", "parameter>", "string=\"", "false\">", ">"];

/// Whether a `Skip` run consists only of the grammar's own delimiters.
///
/// Whitespace carries no meaning inside a run (the delimiters' own spacing is
/// fixed by the template), so it is removed before matching: a run is compared
/// as a concatenation of the tokens above. The comparison is per *run* — all
/// the skipped bytes between two structural actions — never per grammar
/// segment, because the machine splits a run wherever a partial match fails, so
/// any single segment may be a fragment like `string` or `=` on its own.
fn skip_is_structural(bytes: &str) -> bool {
    let mut rest: String = bytes.chars().filter(|character| !character.is_whitespace()).collect();
    while !rest.is_empty() {
        match STRUCTURAL_SKIPS.iter().find_map(|token| rest.strip_prefix(*token)) {
            Some(tail) => rest = tail.to_owned(),
            None => return false,
        }
    }
    true
}

/// One request's DSML parser, driven by [`GlmStreamProcessor`].
pub struct DeepseekOutputParser {
    /// Taken by [`TextParser::finish`], which consumes the machine.
    machine: Option<StateMachine>,
    /// Fed text not yet consumed by an action, one entry per pushed chunk.
    chunks: VecDeque<Option<String>>,
    /// A segment whose bytes span several pushed chunks, held between pushes.
    last_stashed_action: Option<OutputActionSegment>,
    /// Tool-name bytes gathered between `ToolName` segments.
    stashed_tool_name: String,
    /// The previous popped action (dedup guard, as in the crate's processor).
    last_pop_action: OutputAction,
    /// Set by a matched client stop sequence.
    stop: Option<GlmStop>,
    /// Calls that closed: the count behind `finish_reason: tool_calls`.
    calls: usize,
    /// Chunks of the open call, released only when the call closes.
    held: Vec<OutputChunk>,
    /// The open call's raw bytes, from its `<｜DSML｜` marker (its content on loss).
    retained: String,
    /// The open call's name, for the readability check and the log line.
    open_name: String,
    /// True from `ToolCallBegin` until the call's `ToolCallArgumentsEnd`.
    open: bool,
    /// Set when the machine skipped material inside the open call that is not
    /// part of the DSML skeleton (see [`skip_is_structural`]): the call is
    /// malformed even if the arguments it did assemble happen to be valid JSON.
    malformed: bool,
    /// Set when a closed call turned out unreadable: its raw bytes are still
    /// being collected (the ` invoke>` tail is consumed after the close) and
    /// are emitted whole, as content, when the envelope's end is known.
    closing: bool,
    /// `retained`'s length once the lost call's own close is consumed: only
    /// after it can a block close end the envelope (the call's own
    /// `</｜DSML｜` would otherwise cut it short). `None` until the first byte
    /// after that close arrives — the close itself is consumed in fragments,
    /// so its end is only known later.
    lost_len: Option<usize>,
    /// Skipped bytes since the last structural action inside the open call,
    /// validated as one run when that action arrives (segments are fragments).
    skip_run: String,
    /// Parameter-name bytes gathered between `{`/`, ` and `: ` while open:
    /// `Some` exactly while the machine is inside a parameter name.
    pending_name: Option<String>,
    /// Skipped bytes since the last emitted chunk: the opener a call begins with.
    pending: String,
    /// Emit tool-call argument deltas as they arrive instead of holding the
    /// whole call until it closes (`CUTEAFD_STREAM_TOOLCALL_ARGS=1`).
    stream_args: ArgStream,
    /// The open call's `ToolCall` was emitted to the client: a call that turns
    /// out unreadable has already been streamed and cannot be retracted.
    streamed: bool,
}

impl DeepseekOutputParser {
    pub fn new(options: ParsingOptions) -> Self {
        Self::with_stream_args(options, false)
    }

    /// [`Self::new`] with the streaming knob set explicitly, so tests and the
    /// router can enable it without touching process env.
    pub fn with_stream_args(options: ParsingOptions, stream_args: bool) -> Self {
        Self {
            machine: Some(StateMachine::new(options)),
            chunks: VecDeque::new(),
            last_stashed_action: None,
            stashed_tool_name: String::new(),
            last_pop_action: OutputAction::Skip,
            stop: None,
            calls: 0,
            held: Vec::new(),
            retained: String::new(),
            open_name: String::new(),
            open: false,
            malformed: false,
            closing: false,
            lost_len: None,
            skip_run: String::new(),
            pending_name: None,
            pending: String::new(),
            stream_args: ArgStream::new(stream_args),
            streamed: false,
        }
    }

    fn push_text(&mut self, text: &str) -> Vec<OutputChunk> {
        let actions = self.machine.as_mut().expect("a machine until finish").feed(text);
        self.chunks.push_back(Some(text.to_owned()));
        self.apply_actions(actions)
    }

    /// The call is malformed and its envelope is complete: return its bytes as
    /// content and log it. Its held chunks are discarded so no part of it is
    /// ever seen as a tool call.
    fn lose(&mut self, out: &mut Vec<OutputChunk>) {
        let text = std::mem::take(&mut self.retained);
        let streamed = std::mem::take(&mut self.streamed);
        self.held.clear();
        self.open = false;
        self.closing = false;
        self.malformed = false;
        self.skip_run.clear();
        self.pending_name = None;
        self.pending.clear();
        if streamed {
            // The call's bytes already reached the client as streamed tool
            // chunks; count it so the finish is `tool_calls` and the validator
            // rejects the assembled arguments cleanly instead of a 500.
            self.calls += 1;
            tracing::warn!(tool = %self.open_name, text = %excerpt(&text),
                "DeepSeek tool call streamed, then found unreadable");
        } else if !text.is_empty() {
            tracing::warn!(tool = %self.open_name, text = %excerpt(&text),
                "DeepSeek tool call returned as content");
            out.push(OutputChunk::Raw { content: text });
        }
        self.open_name.clear();
    }

    /// Emit a lost call's envelope, cut at `index`: `retained[..index]` is the
    /// content and the remainder (a following call's opener) is kept for that
    /// call. Called once the envelope's end is known — the next call's opener,
    /// the block's `</｜DSML｜ calls>` close, or the end of the stream — so the
    /// model's whole ` invoke>` tail is preserved rather than truncated at the
    /// `</｜DSML｜` the machine reads as the argument close.
    fn flush_lost(&mut self, index: usize, out: &mut Vec<OutputChunk>) {
        self.closing = false;
        self.lost_len = None;
        let text = self.retained[..index].to_owned();
        self.retained = self.retained[index..].to_owned();
        let streamed = std::mem::take(&mut self.streamed);
        if streamed {
            self.calls += 1;
            tracing::warn!(tool = %self.open_name, text = %excerpt(&text),
                "DeepSeek tool call streamed, then found unreadable");
        } else if !text.is_empty() {
            tracing::warn!(tool = %self.open_name, text = %excerpt(&text),
                "DeepSeek tool call returned as content");
            out.push(OutputChunk::Raw { content: text });
        }
        self.open_name.clear();
    }

    /// Validate the skip run accumulated since the last structural action: in a
    /// well-formed call it is only the grammar's own delimiters, and anything
    /// else means the machine read past bytes it could not place — the case a
    /// JSON check alone cannot see, because the arguments it goes on to emit
    /// are still well-formed (often just `{}`).
    fn flush_skip_run(&mut self) {
        if self.open && !self.skip_run.is_empty() && !skip_is_structural(&self.skip_run) {
            self.malformed = true;
        }
        self.skip_run.clear();
    }

    /// Validate the parameter name the machine assembled between `parameter
    /// name=` and the type attribute: `"key"`, quoted and never empty. An empty
    /// name yields `{"": …}`, which is valid JSON but not the call the model
    /// meant.
    fn flush_param_name(&mut self) {
        if let Some(name) = self.pending_name.take() {
            let quoted = name.len() > 2 && name.starts_with('"') && name.ends_with('"');
            if !quoted {
                self.malformed = true;
            }
        }
    }

    /// Close the open call: release its held chunks and count it, or lose it.
    ///
    /// Run on *every* `ToolCallArgumentsEnd`, including the repeated segments
    /// the dedup guards suppress from the output — those guards decide what is
    /// emitted, not whether a call closed. A call with no open segment (a
    /// repeated close after the first already closed it) is a no-op.
    ///
    /// A call is unreadable when its name is empty or carries markup, when the
    /// machine skipped non-structural bytes inside it ([`Self::malformed`], the
    /// case JSON validity alone cannot see: a malformed parameter envelope is
    /// skipped into a perfectly valid `{}`), or when the arguments assembled
    /// are not valid JSON.
    fn close(&mut self, out: &mut Vec<OutputChunk>) {
        if !self.open {
            return;
        }
        self.flush_skip_run();
        self.flush_param_name();
        let arguments: String = self.held.iter().filter_map(|chunk| match chunk {
            OutputChunk::ToolArgumentsDelta { content } => Some(content.as_str()),
            _ => None,
        }).collect();
        let unreadable = self.open_name.is_empty() || self.open_name.contains(['<', '>', '\n'])
            || self.malformed
            || serde_json::from_str::<serde_json::Value>(&arguments).is_err();
        if unreadable {
            if self.streamed {
                // The call's bytes already reached the client as streamed tool
                // chunks and cannot be retracted. Count it so the finish is
                // `tool_calls`; the validator then rejects the assembled
                // arguments with a clean 400/SSE error, never a 500.
                self.calls += 1;
                tracing::warn!(tool = %self.open_name, text = %excerpt(&self.retained),
                    "DeepSeek tool call streamed, then found unreadable");
                self.held.clear();
                self.open = false;
                self.closing = false;
                self.lost_len = None;
                self.malformed = false;
                self.skip_run.clear();
                self.pending_name = None;
                self.retained.clear();
                self.pending.clear();
                self.open_name.clear();
                self.streamed = false;
                return;
            }
            // The envelope is not complete yet: the machine still consumes the
            // ` invoke>` tail as skipped bytes, and they belong to the returned
            // text. Keep collecting and emit at `flush_lost`.
            self.held.clear();
            self.open = false;
            self.closing = true;
            self.lost_len = None;
            self.malformed = false;
            self.skip_run.clear();
            self.pending_name = None;
            return;
        }
        if self.streamed {
            // The deltas were emitted as they arrived; only the count is owed.
            self.held.clear();
        } else {
            out.extend(self.held.drain(..));
        }
        self.calls += 1;
        self.open = false;
        self.malformed = false;
        self.skip_run.clear();
        self.pending_name = None;
        self.retained.clear();
        self.pending.clear();
        self.open_name.clear();
        self.streamed = false;
    }

    /// The opened call's bytes, back to the last `<｜DSML｜` in `pending` (which
    /// also drops a preceding call's `</｜DSML｜ invoke>` tail). The newlines the
    /// machine's opener branch consumed are kept.
    fn retained_prefix(&self) -> &str {
        &self.pending[opener_index(&self.pending)..]
    }

    /// Bytes for this action, and the chunks to emit, mirroring the crate's
    /// `StashedChunks::pop` with the FR-G.14 call hold.
    fn pop(&mut self, action: OutputActionSegment) -> Vec<OutputChunk> {
        let Some(mut front_chunk) = self.chunks.pop_front() else { return Vec::new() };
        let front_len = front_chunk.as_ref().map(String::len).unwrap_or(0);
        if action.len < front_len {
            let remaining = front_chunk.as_mut().map(|content| content.split_off(action.len));
            self.chunks.push_front(remaining);
        }
        let last_pop_action = self.last_pop_action;
        self.last_pop_action = action.action;
        let bytes = front_chunk.clone();
        let mut out: Vec<OutputChunk> = Vec::new();

        // Every byte the machine consumes inside an open call — and inside a
        // lost one still being collected — is retained, so a lost call can be
        // returned whole; outside a call, only skipped bytes are remembered
        // (they carry the `<｜DSML｜` opener).
        //
        // A matched segment is popped once per buffered chunk (the crate's
        // `apply_whole_chunks`), so the same action arrives many times for one
        // match, the last fragment included. State that a *match* sets — never a
        // byte's worth — is therefore taken on the action's first pop only,
        // under the same `last_pop_action` guard the crate uses for its own
        // dedup; per-byte state (arguments, retained bytes) is taken every time.
        match action.action {
            OutputAction::ToolCallBegin => {
                if last_pop_action != OutputAction::ToolCallBegin {
                    if self.closing {
                        // The opener now seen belongs to the *next* call: cut
                        // the lost envelope just before it (never into it).
                        let index = opener_index(&self.retained);
                        self.flush_lost(index, &mut out);
                    } else {
                        let prefix = self.retained_prefix().to_owned();
                        self.retained = prefix;
                    }
                    self.pending.clear();
                    self.open = true;
                }
            }
            OutputAction::Skip | OutputAction::SkipInvalid { .. } => {
                // Junk skipped inside an open call never reaches the arguments
                // the machine assembles, so it has to mark the call itself. The
                // run is validated whole, when the next structural action ends
                // it (see `flush_skip_run`), because a run is split into
                // fragments wherever a partial match fails.
                if self.open {
                    if let Some(text) = bytes.as_deref() {
                        self.skip_run.push_str(text);
                    }
                }
            }
            _ => {
                if self.open {
                    self.flush_skip_run();
                } else {
                    // A new action while a lost call is still being collected:
                    // its envelope ended without a following opener or block
                    // close (both of those are handled where they arrive).
                    // Fragments of the close itself are the same action, so
                    // only a genuinely new one ends the envelope here.
                    if self.closing && last_pop_action != action.action {
                        let index = self.retained.len();
                        self.flush_lost(index, &mut out);
                    }
                    self.pending.clear();
                }
            }
        }
        if let Some(text) = bytes.as_deref() {
            if self.open || self.closing {
                if self.closing && self.lost_len.is_none() && matches!(action.action,
                    OutputAction::Skip | OutputAction::SkipInvalid { .. }) {
                    // The lost call's own `</｜DSML｜` close is consumed: what
                    // follows is the envelope's tail, and only there can the
                    // block's `</｜DSML｜ calls>` end the envelope.
                    self.lost_len = Some(self.retained.len());
                }
                self.retained.push_str(text);
                if self.closing {
                    if let Some(tail) = self.lost_len {
                        if let Some(index) = self.retained[tail..].rfind(DSML_END) {
                            let index = back_over_newlines(&self.retained, tail + index);
                            self.flush_lost(index, &mut out);
                            self.retained.clear();
                        }
                    }
                }
            } else if matches!(action.action, OutputAction::Skip | OutputAction::SkipInvalid { .. }) {
                self.pending.push_str(text);
            }
        }

        out.extend(match action.action {
            OutputAction::Raw => non_empty(front_chunk).map(|content| OutputChunk::Raw { content }).into_iter().collect(),
            OutputAction::Reasoning => {
                non_empty(front_chunk).map(|content| OutputChunk::Reasoning { content }).into_iter().collect()
            }
            OutputAction::ToSpace => vec![OutputChunk::Raw { content: " ".to_string() }],
            // Structural skips (`</｜DSML｜ calls>`, separators, extra `</think>`):
            // the crate drops them and so do we, so a well-formed multi-call
            // response is byte-identical.
            OutputAction::Skip | OutputAction::SkipInvalid { .. } => Vec::new(),
            OutputAction::StopSequence => {
                self.stop = Some(GlmStop::Sequence(front_chunk.unwrap_or_default()));
                Vec::new()
            }
            OutputAction::ToolCallBegin => {
                let mut out = Vec::new();
                if last_pop_action != OutputAction::ToolCallBegin {
                    self.held.push(OutputChunk::ToolCallBegin);
                    self.stream_args.begin(&mut out);
                }
                out
            }
            OutputAction::ToolName => {
                if let Some(content) = front_chunk {
                    self.stashed_tool_name.push_str(&content);
                }
                Vec::new()
            }
            OutputAction::ToolNameEnd => {
                let mut out = Vec::new();
                if last_pop_action != OutputAction::ToolNameEnd {
                    let tool_name = std::mem::take(&mut self.stashed_tool_name);
                    self.open_name = tool_name.clone();
                    self.held.push(OutputChunk::ToolCall { tool_name: tool_name.clone(), arguments: String::new() });
                    // Name check before emitting: a call with an unreadable name
                    // is never streamed as a tool call (it is returned as
                    // content on loss, as in hold mode).
                    if self.stream_args.enabled() && name_is_readable(&tool_name, None) {
                        self.stream_args.call(&tool_name, &mut out);
                        self.streamed = true;
                    }
                }
                out
            }
            OutputAction::LabelToolCallArguments { label, id } => {
                // A new parameter's name starts, or the one just read ends: a
                // match, so taken once, not once per fragment (an empty
                // fragment would otherwise close a name that never started).
                let mut out = Vec::new();
                if last_pop_action != action.action {
                    match id {
                        // `{` / `, `: the machine is now inside a parameter name.
                        0 | 1 => {
                            self.flush_param_name();
                            self.pending_name = Some(String::new());
                        }
                        // `: `: the name just ended; it must be a quoted key.
                        2 => self.flush_param_name(),
                        _ => {}
                    }
                    self.held.push(OutputChunk::ToolArgumentsDelta { content: label.to_string() });
                    if self.streamed {
                        self.stream_args.delta(label, &mut out);
                    }
                }
                out
            }
            OutputAction::RawToolCallArguments { string } => {
                let content = if string { escape_json_string(front_chunk.unwrap_or_default()) }
                    else { front_chunk.unwrap_or_default() };
                // Bytes between `{`/`, ` and `: ` are the parameter name.
                if let Some(name) = self.pending_name.as_mut() {
                    name.push_str(&content);
                }
                self.held.push(OutputChunk::ToolArgumentsDelta { content: content.clone() });
                let mut out = Vec::new();
                if self.streamed {
                    self.stream_args.delta(&content, &mut out);
                }
                out
            }
            OutputAction::ToolCallArgumentsEnd { output } => {
                let mut closed = Vec::new();
                if last_pop_action != action.action {
                    if let Some(output) = output {
                        self.held.push(OutputChunk::ToolArgumentsDelta { content: output.to_string() });
                        if self.streamed {
                            self.stream_args.delta(output, &mut closed);
                        }
                    }
                }
                // Every `ToolCallArgumentsEnd` closes the call, including the
                // repeated segments the guard above suppresses from the output.
                self.close(&mut closed);
                closed
            }
            OutputAction::Label(label) => {
                if last_pop_action != action.action {
                    vec![OutputChunk::Raw { content: label.to_string() }]
                } else {
                    Vec::new()
                }
            }
        });
        out
    }

    /// Emit each whole buffered chunk an action covers, exactly as the crate does.
    fn apply_whole_chunks(&mut self, action: &mut OutputActionSegment, outputs: &mut Vec<OutputChunk>) {
        loop {
            let front_len = self.chunks.front()
                .map(|content| content.as_ref().map(String::len).unwrap_or(0));
            if front_len.is_some_and(|front_len| front_len <= action.len) {
                let front_len = front_len.unwrap();
                outputs.extend(self.pop(OutputActionSegment::new(action.action, front_len)));
                action.len -= front_len;
            } else {
                return;
            }
        }
    }

    fn apply_actions(&mut self, actions: Vec<OutputActionSegment>) -> Vec<OutputChunk> {
        let mut outputs = Vec::new();
        let mut last_stashed_action = self.last_stashed_action.take();
        for action in actions {
            if let Some(last_stashed_action) = &mut last_stashed_action {
                if action.action == last_stashed_action.action {
                    last_stashed_action.len += action.len;
                } else {
                    self.apply_whole_chunks(last_stashed_action, &mut outputs);
                    if last_stashed_action.len > 0 {
                        outputs.extend(self.pop(last_stashed_action.clone()));
                    }
                    *last_stashed_action = action;
                }
            } else {
                last_stashed_action = Some(action);
            }
        }
        if let Some(last_stashed_action) = &mut last_stashed_action {
            self.apply_whole_chunks(last_stashed_action, &mut outputs);
        }
        self.last_stashed_action = last_stashed_action;
        outputs
    }
}

impl TextParser for DeepseekOutputParser {
    fn push(&mut self, text: &str) -> Vec<OutputChunk> {
        self.push_text(text)
    }

    fn finish(&mut self) -> Vec<OutputChunk> {
        let mut outputs = match self.machine.take() {
            Some(machine) => self.apply_actions(machine.finish()),
            None => Vec::new(),
        };
        // A call still open at end of stream never closed: it is content.
        if self.open {
            self.lose(&mut outputs);
        } else if self.closing {
            // A lost call whose envelope end never arrived: return what there is.
            let index = self.retained.len();
            self.flush_lost(index, &mut outputs);
        }
        outputs
    }

    fn stop(&self) -> Option<&GlmStop> {
        self.stop.as_ref()
    }

    fn tool_calls(&self) -> usize {
        self.calls
    }
}

fn non_empty(content: Option<String>) -> Option<String> {
    content.filter(|content| !content.is_empty())
}

/// `content` as a JSON string body, for a schema-typed (string) DSML parameter.
fn escape_json_string(content: String) -> String {
    match serde_json::to_string(&content) {
        Ok(escaped) if escaped.len() > 2 => escaped[1..escaped.len() - 1].to_string(),
        _ => String::new(),
    }
}

/// A bounded excerpt of a lost call for the log line.
fn excerpt(text: &str) -> String {
    const LIMIT: usize = 256;
    if text.len() <= LIMIT { return text.to_owned(); }
    let mut end = LIMIT;
    while !text.is_char_boundary(end) { end -= 1; }
    format!("{}…", &text[..end])
}

#[cfg(test)]
mod tests {
    use super::*;
    use deepseek_recipe::stream::{
        ChunkGenerator, InferenceChunk, InferenceFinishReason, PromptUsage, StreamProcessor,
    };
    use futures::StreamExt;

    const CALL: &str = "Sure.\n<｜DSML｜ calls>\n<｜DSML｜ invoke name=\"search\">\n\
<｜DSML｜ parameter name=\"query\" string=\"true\">cats</｜DSML｜ parameter>\n\
</｜DSML｜ invoke>\n</｜DSML｜ calls>";

    /// Two calls in one `calls` block, the second with a schema-typed
    /// (non-string) parameter: a `false">` type and its `</｜DSML｜` value end are
    /// skips inside an open call too.
    const TWO_CALLS: &str = "Sure.\n<｜DSML｜ calls>\n\
<｜DSML｜ invoke name=\"search\">\n<｜DSML｜ parameter name=\"query\" string=\"true\">cats</｜DSML｜ parameter>\n</｜DSML｜ invoke>\n\
<｜DSML｜ invoke name=\"lookup\">\n<｜DSML｜ parameter name=\"id\" string=\"false\">42</｜DSML｜ parameter>\n</｜DSML｜ invoke>\n\
</｜DSML｜ calls>";

    /// A malformed call in the middle of two valid ones: the valid calls parse,
    /// the malformed call's envelope comes back whole, and no valid call's
    /// bytes leak into it.
    const MIXED_CALLS: &str = "Sure.\n<｜DSML｜ calls>\n\
<｜DSML｜ invoke name=\"search\">\n<｜DSML｜ parameter name=\"query\" string=\"true\">cats</｜DSML｜ parameter>\n</｜DSML｜ invoke>\n\
<｜DSML｜ invoke name=\"lookup\">\n<｜DSML｜ parameter name=\"id\" string\"false\">42</｜DSML｜ parameter>\n</｜DSML｜ invoke>\n\
<｜DSML｜ invoke name=\"search\">\n<｜DSML｜ parameter name=\"query\" string=\"true\">dogs</｜DSML｜ parameter>\n</｜DSML｜ invoke>\n\
</｜DSML｜ calls>";

    /// The middle call of [`MIXED_CALLS`] as it must be returned: whole, from
    /// its own opener through its own ` invoke>`, and nothing else.
    const MIXED_LOST: &str = "Sure.\n<｜DSML｜ invoke name=\"lookup\">\n\
<｜DSML｜ parameter name=\"id\" string\"false\">42</｜DSML｜ parameter>\n</｜DSML｜ invoke>";

    /// A closed invoke whose parameter envelope the machine cannot read: the
    /// `string` type attribute lost its `=`. Those bytes are skipped, the call
    /// still reaches its close, and the arguments assembled are the valid
    /// `{"query": "cats"}` — so JSON validity alone accepts the call with the
    /// parameter silently dropped.
    const MALFORMED_ENVELOPE: &str = "Sure.\n<｜DSML｜ invoke name=\"search\">\n\
<｜DSML｜ parameter name=\"query\" string\"true\">cats</｜DSML｜ parameter>\n</｜DSML｜ invoke>";

    /// A closed invoke whose parameter name lost its opening quote.
    const MALFORMED_NAME: &str = "Sure.\n<｜DSML｜ invoke name=\"search\">\n\
<｜DSML｜ parameter name=query\" string=\"true\">cats</｜DSML｜ parameter>\n</｜DSML｜ invoke>";

    /// A closed invoke with an empty parameter name: at the call level the name
    /// is fine and the arguments assembled (`{"": "cats"}`) are valid JSON, so
    /// only the per-parameter name check catches it.
    const EMPTY_PARAM_NAME: &str = "Sure.\n<｜DSML｜ invoke name=\"search\">\n\
<｜DSML｜ parameter name=\"\" string=\"true\">cats</｜DSML｜ parameter>\n</｜DSML｜ invoke>";

    /// One well-formed call, for the differential check against the crate.
    fn options() -> ParsingOptions {
        ParsingOptions { parse_tool_calls: true, ..ParsingOptions::default() }
    }

    /// Feed `text` one character at a time (chunking must not matter) and
    /// return every chunk, the parsed-call count and the matched stop.
    fn parse(options: ParsingOptions, text: &str) -> (Vec<OutputChunk>, usize, Option<GlmStop>) {
        let mut parser = DeepseekOutputParser::new(options);
        let mut chunks = Vec::new();
        for character in text.chars() {
            chunks.extend(parser.push(&character.to_string()));
        }
        chunks.extend(parser.finish());
        (chunks, parser.tool_calls(), parser.stop().cloned())
    }

    fn content(chunks: &[OutputChunk]) -> String {
        chunks.iter().filter_map(|chunk| match chunk {
            OutputChunk::Raw { content } => Some(content.as_str()),
            _ => None,
        }).collect()
    }

    /// Feed `text` in one piece, for the chunking-invariance check.
    fn parse_whole(options: ParsingOptions, text: &str) -> (Vec<OutputChunk>, usize) {
        let mut parser = DeepseekOutputParser::new(options);
        let mut chunks = parser.push(text);
        chunks.extend(parser.finish());
        (chunks, parser.tool_calls())
    }

    /// A chunking-invariant view of a parse: the answer text, the assembled
    /// tool arguments and the call names, none of which may depend on how the
    /// input was split (only the number of `OutputChunk`s may).
    fn signature(chunks: &[OutputChunk]) -> (String, String, Vec<String>) {
        let mut raw = String::new();
        let mut arguments = String::new();
        let mut names = Vec::new();
        for chunk in chunks {
            match chunk {
                OutputChunk::Raw { content } => raw.push_str(content),
                OutputChunk::ToolArgumentsDelta { content } => arguments.push_str(content),
                OutputChunk::ToolCall { tool_name, .. } => names.push(tool_name.clone()),
                _ => {}
            }
        }
        (raw, arguments, names)
    }

    fn is_tool_chunk(chunk: &OutputChunk) -> bool {
        matches!(chunk, OutputChunk::ToolCallBegin | OutputChunk::ToolCall { .. }
            | OutputChunk::ToolArgumentsDelta { .. })
    }

    /// A mock protocol generator that echoes each parsed chunk.
    struct Echo;

    impl ChunkGenerator for Echo {
        type Chunk = OutputChunk;
        async fn generate(&mut self, chunk: OutputChunk) -> Vec<OutputChunk> {
            vec![chunk]
        }
    }

    fn inference(text: &str) -> impl futures::Stream<Item = InferenceChunk> + Send {
        futures::stream::iter(vec![
            InferenceChunk::Ready { system_fingerprint: None, prompt_usage: PromptUsage::default() },
            InferenceChunk::Text { content: text.to_owned(), content_tokens: 1 },
            InferenceChunk::Finish { finish_reason: InferenceFinishReason::Stop },
        ])
    }

    /// The crate's own processor, for parity. Feed it once through a mock
    /// generator and collect the protocol chunks.
    async fn crate_chunks(options: ParsingOptions, text: &str) -> Vec<OutputChunk> {
        StreamProcessor::new(Echo, options).process(inference(text))
            .map(|chunk| chunk.expect("no stream error")).collect().await
    }

    /// My parser, through the same `GlmStreamProcessor` the server uses.
    async fn ours_chunks(options: ParsingOptions, text: &str) -> Vec<OutputChunk> {
        crate::openai::chat::glm5::GlmStreamProcessor::new(Echo, DeepseekOutputParser::new(options))
            .process(inference(text)).map(|chunk| chunk.expect("no stream error")).collect().await
    }

    /// A well-formed answer and call parse byte-for-byte like the crate does,
    /// holding only changes when the chunks are emitted, never their content.
    #[tokio::test]
    async fn well_formed_output_matches_the_crate_processor() {
        for text in ["Just an answer.", CALL, &format!("{CALL}\nTrailing text is dropped.")] {
            assert_eq!(ours_chunks(options(), text).await, crate_chunks(options(), text).await,
                "text {text:?}");
        }
    }

    /// A call is held from its opener until it closes: no part of it is emitted
    /// before the call's own `</｜DSML｜` — the grammar's argument close, which
    /// is the last `</｜DSML｜` of its `</｜DSML｜ invoke>` tail.
    #[test]
    fn a_complete_call_is_held_until_it_closes() {
        let is_call_chunk = |chunk: &OutputChunk| matches!(chunk,
            OutputChunk::ToolCallBegin | OutputChunk::ToolCall { .. } | OutputChunk::ToolArgumentsDelta { .. });
        let close = CALL.find("</｜DSML｜ invoke>").expect("the call closes") + DSML_END.len();
        let mut parser = DeepseekOutputParser::new(options());
        let mut seen = Vec::new();
        for (index, character) in CALL.char_indices() {
            seen.extend(parser.push(&character.to_string()));
            if index + character.len_utf8() < close {
                assert!(!seen.iter().any(is_call_chunk), "call leaked before its close at {index}");
            }
        }
        seen.extend(parser.finish());
        assert_eq!(parser.tool_calls(), 1);
        assert!(seen.iter().any(is_call_chunk), "the closed call is released");
        let arguments: String = seen.iter().filter_map(|chunk| match chunk {
            OutputChunk::ToolArgumentsDelta { content } => Some(content.as_str()), _ => None }).collect();
        assert_eq!(serde_json::from_str::<serde_json::Value>(&arguments).unwrap(), serde_json::json!({"query": "cats"}));
    }

    /// One fixture per malformed shape: the call comes back as content from its
    /// `<｜DSML｜` marker, no part of it is a tool call, and it is not counted.
    ///
    /// Each fixture's text *is* the envelope the model emitted, and that whole
    /// text is what a client sees — the model's ` invoke>` tail included, never
    /// a call decoded from bytes the machine could not read.
    #[test]
    fn unreadable_calls_return_as_content() {
        for (shape, text) in [
            // Truncated inside the argument value (max_tokens).
            ("truncated value",
                "Sure.\n<｜DSML｜ invoke name=\"search\">\n<｜DSML｜ parameter name=\"query\" string=\"true\">ca"),
            // Truncated before the name closes.
            ("truncated name", "Sure.\n<｜DSML｜ invoke name=\"sea"),
            // Truncated after the arguments, before the call's close.
            ("missing invoke close",
                "Sure.\n<｜DSML｜ invoke name=\"search\">\n<｜DSML｜ parameter name=\"query\" string=\"true\">cats</｜DSML｜ parameter>"),
            // An empty tool name.
            ("empty name", "Sure.\n<｜DSML｜ invoke name=\"\">\n</｜DSML｜ invoke>"),
            // Markup in the name.
            ("markup in the name", "Sure.\n<｜DSML｜ invoke name=\"se<arch\">\n</｜DSML｜ invoke>"),
            // A closed call whose parameter bytes do not assemble valid JSON: a
            // schema-typed (non-string) value that is not JSON. A *string*
            // value cannot do this — it is escaped into a JSON string body — so
            // the fixture has to be an untyped one.
            ("unparseable arguments",
                "Sure.\n<｜DSML｜ invoke name=\"search\">\n<｜DSML｜ parameter name=\"n\" string=\"false\">not json</｜DSML｜ parameter>\n</｜DSML｜ invoke>"),
            // Closed calls the arguments alone cannot condemn: the machine
            // skips the malformed bytes and still emits valid JSON.
            ("malformed parameter envelope", MALFORMED_ENVELOPE),
            ("missing quote in the parameter name", MALFORMED_NAME),
            ("empty parameter name", EMPTY_PARAM_NAME),
        ] {
            let (chunks, calls, _) = parse(options(), text);
            assert_eq!(calls, 0, "{shape}");
            assert!(!chunks.iter().any(is_tool_chunk), "{shape}: no part of the call may be a tool call");
            assert_eq!(content(&chunks), text, "{shape}");
        }
    }

    /// A malformed call between two valid ones: both valid calls parse, the
    /// malformed envelope is returned whole (its ` invoke>` included), and it
    /// takes neither valid call's bytes with it.
    #[test]
    fn a_lost_call_between_valid_ones_takes_only_its_own_bytes() {
        let (chunks, calls, _) = parse(options(), MIXED_CALLS);
        assert_eq!(calls, 2, "the two readable calls parse; the malformed one does not");
        let (raw, arguments, names) = signature(&chunks);
        assert_eq!(names, ["search", "search"]);
        assert_eq!(arguments, "{\"query\": \"cats\"}{\"query\": \"dogs\"}");
        assert_eq!(raw, MIXED_LOST, "the lost envelope, whole, and only its own bytes");
    }

    /// A malformed call never claims `tool_calls`, and the answer around it is
    /// still delivered.
    #[test]
    fn a_lost_call_leaves_the_surrounding_answer_intact() {
        let (chunks, calls, _) = parse(options(), "Before.\n<｜DSML｜ invoke name=\"search\">\n<｜DSML｜ parameter name=\"query\" string=\"true\">ca");
        assert_eq!(calls, 0);
        assert_eq!(content(&chunks),
            "Before.\n<｜DSML｜ invoke name=\"search\">\n<｜DSML｜ parameter name=\"query\" string=\"true\">ca");
    }

    /// A malformed closed call must not be accepted *with the parameter that
    /// the skip check exists to catch*: the arguments of the malformed-envelope
    /// fixture are otherwise the valid `{"query": "cats"}`.
    #[test]
    fn a_skipped_parameter_envelope_does_not_parse_as_a_call() {
        let mut parser = DeepseekOutputParser::new(options());
        let mut chunks = parser.push(MALFORMED_ENVELOPE);
        chunks.extend(parser.finish());
        assert_eq!(parser.tool_calls(), 0, "the call closed, but must not count");
        assert!(!chunks.iter().any(is_tool_chunk));
        assert_eq!(content(&chunks), MALFORMED_ENVELOPE, "the envelope is returned whole");
    }

    /// Two calls in one block parse as two, in order, each with its own
    /// arguments — including an untyped-by-schema (non-string) parameter, whose
    /// `false">` type and `</｜DSML｜` value end are skips inside an open call.
    #[test]
    fn two_calls_in_one_block_both_parse() {
        let (chunks, calls, _) = parse(options(), TWO_CALLS);
        assert_eq!(calls, 2);
        let (_, arguments, names) = signature(&chunks);
        assert_eq!(names, ["search", "lookup"]);
        assert_eq!(arguments, "{\"query\": \"cats\"}{\"id\": 42}");
    }

    /// Chunking must not matter: the answer, the assembled arguments and the
    /// call count are identical however the text is split — for a well-formed
    /// call, two calls, and both malformed shapes. Splitting also produces the
    /// repeated `ToolCallArgumentsEnd` segments the output dedup guards
    /// suppress, so a call must still close exactly once per `ToolCallArgumentsEnd`.
    #[test]
    fn any_chunk_boundary_parses_the_same() {
        for (shape, text) in [
            ("valid call", CALL.to_string()),
            ("two calls", TWO_CALLS.to_string()),
            ("mixed calls", MIXED_CALLS.to_string()),
            ("malformed envelope", MALFORMED_ENVELOPE.to_string()),
            ("malformed name", MALFORMED_NAME.to_string()),
        ] {
            let (whole, whole_calls) = parse_whole(options(), &text);
            let expected = signature(&whole);

            let (charwise, charwise_calls, _) = parse(options(), &text);
            assert_eq!((signature(&charwise), charwise_calls), (expected.clone(), whole_calls),
                "{shape}: one character at a time");

            for split in text.char_indices().map(|(index, character)| index + character.len_utf8()) {
                let mut parser = DeepseekOutputParser::new(options());
                let mut chunks = parser.push(&text[..split]);
                chunks.extend(parser.push(&text[split..]));
                chunks.extend(parser.finish());
                assert_eq!((signature(&chunks), parser.tool_calls()), (expected.clone(), whole_calls),
                    "{shape}: split at byte {split}");
            }
        }
    }

    // ---------------------------------------------------------------- router

    /// The same contract through the real OpenAI route: an unreadable call is
    /// ordinary content with `finish_reason: stop`, a parsed call is
    /// `tool_calls`, and neither is a failed request.
    mod router {
        use super::*;
        use crate::openai::{router_for_model, ConsoleHub, ModelEncoding, ModelProfile, NativeLimits,
            NativeRequest};
        use axum::body::{to_bytes, Body};
        use axum::http::{Request, StatusCode};
        use serde_json::json;
        use std::sync::{Arc, Mutex};
        use tokio::sync::mpsc;
        use tower::ServiceExt;

        const MODEL: &str = "deepseek-ai/DeepSeek-V4.1-Flash";

        /// Serve one request whose worker streams `text` one character at a time
        /// (the streaming parser is the one under test).
        async fn serve(body: serde_json::Value, text: &str) -> (StatusCode, Vec<u8>) {
            let (queue, mut receive) = mpsc::channel::<NativeRequest>(1);
            let text = text.to_owned();
            let worker = tokio::spawn(async move {
                let job = receive.recv().await.unwrap();
                let _ = job.events.send(Ok(InferenceChunk::Ready { system_fingerprint: None,
                    prompt_usage: PromptUsage { prompt_tokens: 9, prompt_cache_hit_tokens: 0 } }));
                for character in text.chars() {
                    if job.events.send(Ok(InferenceChunk::Text { content: character.to_string(),
                        content_tokens: 1 })).is_err() {
                        return;
                    }
                }
                let _ = job.events.send(Ok(InferenceChunk::Finish { finish_reason: InferenceFinishReason::Stop }));
            });
            let app = router_for_model(queue, NativeLimits::default(), Arc::new(Mutex::new(serde_json::Value::Null)),
                std::time::Duration::from_secs(5), ConsoleHub::disabled(),
                ModelProfile::new(MODEL, ModelEncoding::DeepseekV41));
            let request = Request::post("/v1/chat/completions").header("content-type", "application/json")
                .body(Body::from(body.to_string())).unwrap();
            let response = app.oneshot(request).await.unwrap();
            let status = response.status();
            let bytes = to_bytes(response.into_body(), 1 << 20).await.unwrap().to_vec();
            worker.await.unwrap();
            (status, bytes)
        }

        fn sse_events(bytes: &[u8]) -> Vec<serde_json::Value> {
            let text = std::str::from_utf8(bytes).unwrap();
            assert!(text.ends_with("data: [DONE]\n\n"), "{text}");
            text.split("\n\n").filter_map(|event| event.strip_prefix("data: "))
                .filter(|data| *data != "[DONE]").map(|data| serde_json::from_str(data).unwrap()).collect()
        }

        #[tokio::test]
        async fn unreadable_calls_come_back_as_content_in_both_response_modes() {
            let tools = json!([{"type": "function", "function": {"name": "search", "parameters": {"type": "object",
                "properties": {"query": {"type": "string"}}}}}]);
            let valid = "Plan.\n<｜DSML｜ calls>\n<｜DSML｜ invoke name=\"search\">\n\
<｜DSML｜ parameter name=\"query\" string=\"true\">cats</｜DSML｜ parameter>\n</｜DSML｜ invoke>\n</｜DSML｜ calls>";
            for (shape, text, content, calls, finish) in [
                // An unreadable call: content, stop, never a failed request.
                ("unreadable", "Plan.\n<｜DSML｜ invoke name=\"search\">\n\
<｜DSML｜ parameter name=\"query\" string\"true\">cats</｜DSML｜ parameter>\n</｜DSML｜ invoke>".to_owned(),
                    "Plan.\n<｜DSML｜ invoke name=\"search\">\n\
<｜DSML｜ parameter name=\"query\" string\"true\">cats</｜DSML｜ parameter>\n</｜DSML｜ invoke>".to_owned(),
                    0, "stop"),
                // A call that parsed: tool_calls.
                ("readable", valid.to_owned(), "Plan.".to_owned(), 1, "tool_calls"),
            ] {
                for streaming in [false, true] {
                    let body = json!({"model": MODEL, "stream": streaming, "enable_thinking": false,
                        "tools": tools, "messages": [{"role": "user", "content": "Search for cats."}]});
                    let (status, bytes) = serve(body, &text).await;
                    assert_eq!(status, StatusCode::OK, "{shape} streaming={streaming}");
                    let (reply, names, reason) = if streaming {
                        let events = sse_events(&bytes);
                        let reply: String = events.iter()
                            .filter_map(|event| event["choices"][0]["delta"]["content"].as_str()).collect();
                        let names: Vec<String> = events.iter()
                            .flat_map(|event| event["choices"][0]["delta"]["tool_calls"].as_array().cloned()
                                .unwrap_or_default())
                            .filter_map(|delta| delta["function"]["name"].as_str().map(str::to_owned)).collect();
                        (reply, names, events.last().unwrap()["choices"][0]["finish_reason"].clone())
                    } else {
                        let value: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
                        let message = &value["choices"][0]["message"];
                        let names: Vec<String> = message["tool_calls"].as_array().into_iter().flatten()
                            .map(|call| call["function"]["name"].as_str().unwrap().to_owned()).collect();
                        (message["content"].as_str().unwrap_or_default().to_owned(), names,
                            value["choices"][0]["finish_reason"].clone())
                    };
                    assert_eq!((reply.as_str(), names.len(), reason.as_str()),
                        (content.as_str(), calls, Some(finish)), "{shape} streaming={streaming}");
                }
            }
        }
    }

    // ------------------------------------------------------- streamed args

    /// The concatenated `ToolArgumentsDelta` content of a parse.
    fn deltas(chunks: &[OutputChunk]) -> String {
        chunks.iter().filter_map(|chunk| match chunk {
            OutputChunk::ToolArgumentsDelta { content } => Some(content.as_str()),
            _ => None,
        }).collect()
    }

    fn parse_streaming(text: &str, streaming: bool) -> (Vec<OutputChunk>, usize) {
        let mut parser = DeepseekOutputParser::with_stream_args(options(), streaming);
        let mut chunks = Vec::new();
        for character in text.chars() {
            chunks.extend(parser.push(&character.to_string()));
        }
        chunks.extend(parser.finish());
        (chunks, parser.tool_calls())
    }

    /// With streaming on, argument deltas arrive before the call's close and
    /// concatenate to exactly the held arguments.
    #[test]
    fn streamed_deltas_arrive_before_close_and_match_hold() {
        let (hold, hold_calls) = parse_streaming(CALL, false);
        let (streamed, streamed_calls) = parse_streaming(CALL, true);
        assert_eq!((hold_calls, streamed_calls), (1, 1));
        assert_eq!(deltas(&streamed), deltas(&hold), "streamed == held, byte for byte");
        assert_eq!(serde_json::from_str::<serde_json::Value>(&deltas(&streamed)).unwrap(),
            serde_json::json!({"query": "cats"}));

        let close = CALL.find("</｜DSML｜ invoke>").expect("the call closes") + DSML_END.len();
        let mut parser = DeepseekOutputParser::with_stream_args(options(), true);
        let mut saw_delta_before_close = false;
        for (index, character) in CALL.char_indices() {
            let chunks = parser.push(&character.to_string());
            if index < close
                && chunks.iter().any(|chunk| matches!(chunk, OutputChunk::ToolArgumentsDelta { .. }))
            {
                saw_delta_before_close = true;
            }
        }
        parser.finish();
        assert!(saw_delta_before_close, "a delta must arrive before the call closes");
    }

    /// A non-streamed request always buffers: no delta is emitted before the
    /// close, whatever the knob says.
    #[test]
    fn non_streamed_request_always_buffers() {
        let (chunks, calls) = parse_streaming(CALL, false);
        assert_eq!(calls, 1);
        let close = CALL.find("</｜DSML｜ invoke>").expect("the call closes") + DSML_END.len();
        let mut parser = DeepseekOutputParser::with_stream_args(options(), false);
        let mut saw_delta_before_close = false;
        for (index, character) in CALL.char_indices() {
            let chunks = parser.push(&character.to_string());
            if index + character.len_utf8() < close
                && chunks.iter().any(|chunk| matches!(chunk, OutputChunk::ToolArgumentsDelta { .. }))
            {
                saw_delta_before_close = true;
            }
        }
        parser.finish();
        assert!(!saw_delta_before_close, "a non-streamed request must buffer");
        assert_eq!(deltas(&chunks), "{\"query\": \"cats\"}");
    }

    /// A call whose name is unreadable is never streamed as a tool call: only
    /// the opener `ToolCallBegin` is emitted, no `ToolCall`/delta chunk, and it
    /// comes back as content.
    #[test]
    fn unreadable_name_is_never_streamed_as_a_tool_call() {
        let text = "Sure.\n<｜DSML｜ invoke name=\"se<arch\">\n</｜DSML｜ invoke>";
        let (chunks, calls) = parse_streaming(text, true);
        assert_eq!(calls, 0);
        assert!(!chunks.iter().any(|chunk| matches!(chunk,
            OutputChunk::ToolCall { .. } | OutputChunk::ToolArgumentsDelta { .. })));
        assert_eq!(content(&chunks), text);
    }

    /// Two calls in one block each stream their own deltas, in order.
    #[test]
    fn two_calls_stream_independently() {
        let (chunks, calls) = parse_streaming(TWO_CALLS, true);
        assert_eq!(calls, 2);
        let names: Vec<String> = chunks.iter().filter_map(|chunk| match chunk {
            OutputChunk::ToolCall { tool_name, .. } => Some(tool_name.clone()),
            _ => None,
        }).collect();
        assert_eq!(names, ["search", "lookup"]);
        assert_eq!(deltas(&chunks), "{\"query\": \"cats\"}{\"id\": 42}");
    }

    /// A call truncated mid-argument has already streamed its deltas and cannot
    /// be retracted: it is counted so the finish is `tool_calls`, and the
    /// validator rejects the assembled arguments. The call's bytes are not
    /// returned as content.
    #[test]
    fn truncated_call_is_counted_not_returned_as_content() {
        let text = "Sure.\n<｜DSML｜ invoke name=\"search\">\n<｜DSML｜ parameter name=\"query\" string=\"true\">ca";
        let (chunks, calls) = parse_streaming(text, true);
        assert_eq!(calls, 1);
        assert!(!content(&chunks).contains("search"), "the call was streamed, not returned as content");
    }
}
