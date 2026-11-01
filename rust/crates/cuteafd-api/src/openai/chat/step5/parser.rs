//! Step-5 turn semantics over the shared, byte-identical function/parameter XML grammar.
use deepseek_recipe::stream::OutputChunk;
use deepseek_recipe_core::tools::ToolDefinition;
use crate::openai::chat::{glm5::{parser::GlmStop, processor::TextParser},
    qwen4::{QwenOutputParser, QwenParserOptions}};

#[derive(Debug, Clone, Default)]
pub struct Step5ParserOptions {
    pub tools: Option<Vec<ToolDefinition>>,
    pub stop_sequences: Vec<String>,
}

#[derive(Debug)]
pub struct Step5OutputParser(QwenOutputParser);

impl Step5OutputParser {
    pub fn new(options: Step5ParserOptions) -> Self {
        Self::with_stream_args(options, false)
    }

    /// [`Self::new`] with the streaming knob set explicitly. Step-5 delegates
    /// to Qwen's XML grammar, so it inherits Qwen's incremental deltas.
    pub fn with_stream_args(options: Step5ParserOptions, stream_args: bool) -> Self {
        Self(QwenOutputParser::with_turn_markers_and_stream_args(QwenParserOptions { thinking: true,
            tools: options.tools, stop_sequences: options.stop_sequences }, &super::STOP_MARKERS, stream_args))
    }
}

impl TextParser for Step5OutputParser {
    fn push(&mut self, text: &str) -> Vec<OutputChunk> { self.0.push(text) }
    fn finish(&mut self) -> Vec<OutputChunk> { self.0.finish() }
    fn stop(&self) -> Option<&GlmStop> { self.0.stop() }
    fn tool_calls(&self) -> usize { self.0.tool_calls() }
}
