//! Hy4-flash (`hy_v4_flash`) chat template rendering and incremental XML tool-call parsing.
pub mod parser;
pub mod prompt;

use anyhow::{Context, Result};
use serde_json::Value;
use std::path::Path;

use super::glm5::template::ChatTemplate;
pub use parser::{Hy4OutputParser, Hy4ParserOptions};
pub use prompt::Hy4PromptOptions;

pub const HY_START: &str = "<｜hy_start:opensource｜>";
pub const HY_MIDDLE: &str = "<｜hy_middle:opensource｜>";
pub const HY_END: &str = "<｜hy_end:opensource｜>";
pub const THINK_OPEN: &str = "<think:opensource>";
pub const THINK_CLOSE: &str = "</think:opensource>";
pub const TOOL_CALLS_OPEN: &str = "<tool_calls:opensource>";
pub const TOOL_CALLS_CLOSE: &str = "</tool_calls:opensource>";
pub const TOOL_CALL_OPEN: &str = "<tool_call:opensource>";
pub const TOOL_CALL_CLOSE: &str = "</tool_call:opensource>";
pub const ARG_KEY_OPEN: &str = "<arg_key:opensource>";
pub const ARG_KEY_CLOSE: &str = "</arg_key:opensource>";
pub const ARG_VALUE_OPEN: &str = "<arg_value:opensource>";
pub const ARG_VALUE_CLOSE: &str = "</arg_value:opensource>";
pub const TOOL_RESPONSE_OPEN: &str = "<tool_response:opensource>";
pub const TOOL_RESPONSE_CLOSE: &str = "</tool_response:opensource>";

/// Special token IDs from the Hy4-flash checkpoint tokenizer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Hy4TokenIds {
    pub eos: Vec<u32>,
    pub think_open: u32,
    pub think_close: u32,
    pub tool_calls: [u32; 2],
    pub tool_call: [u32; 2],
    pub turn_markers: Vec<u32>,
}

impl Hy4TokenIds {
    /// Default Hy4-flash (`tencent/Hy4-flash`, 120,832 vocab) special token IDs.
    pub fn hy4_flash() -> Self {
        Self {
            eos: vec![120_025],
            think_open: 120_029,
            think_close: 120_030,
            tool_calls: [120_035, 120_036],
            tool_call: [120_033, 120_034],
            turn_markers: vec![120_000, 120_001, 120_037, 120_038],
        }
    }
}

#[derive(Debug)]
pub struct Hy4Encoding {
    template: ChatTemplate,
    tokens: Hy4TokenIds,
    allow_max_effort: bool,
}

impl Hy4Encoding {
    pub fn new(source: impl Into<String>, tokens: Hy4TokenIds) -> Result<Self> {
        anyhow::ensure!(!tokens.eos.is_empty(), "Hy4 EOS token list must not be empty");
        let source = source.into();
        let allow_max_effort = source.contains("'max'") || source.contains("\"max\"");
        Ok(Self {
            template: ChatTemplate::new(source).context("compile Hy4 chat template")?,
            tokens,
            allow_max_effort,
        })
    }

    pub fn from_snapshot(snapshot: &Path) -> Result<Self> {
        let template = std::fs::read_to_string(snapshot.join("chat_template.jinja"))
            .with_context(|| format!("reading {}", snapshot.join("chat_template.jinja").display()))?;
        let tokenizer: Value = serde_json::from_str(
            &std::fs::read_to_string(snapshot.join("tokenizer.json"))
                .with_context(|| format!("reading {}", snapshot.join("tokenizer.json").display()))?,
        )?;
        let added = tokenizer["added_tokens"]
            .as_array()
            .context("Hy4-flash tokenizer.json missing added_tokens")?;
        let id = |content: &str| -> Result<u32> {
            added
                .iter()
                .find(|token| token["content"] == content)
                .and_then(|token| token["id"].as_u64())
                .map(|id| id as u32)
                .with_context(|| format!("Hy4-flash tokenizer.json has no {content} token"))
        };
        let tokens = Hy4TokenIds {
            eos: vec![id(HY_END)?],
            think_open: id(THINK_OPEN)?,
            think_close: id(THINK_CLOSE)?,
            tool_calls: [id(TOOL_CALLS_OPEN)?, id(TOOL_CALLS_CLOSE)?],
            tool_call: [id(TOOL_CALL_OPEN)?, id(TOOL_CALL_CLOSE)?],
            turn_markers: [HY_START, HY_MIDDLE, TOOL_RESPONSE_OPEN, TOOL_RESPONSE_CLOSE]
                .into_iter()
                .filter_map(|m| id(m).ok())
                .collect(),
        };
        Self::new(template, tokens)
    }

    pub fn tokens(&self) -> &Hy4TokenIds {
        &self.tokens
    }

    pub fn stop_token_ids(&self, tools_declared: bool) -> Vec<u32> {
        let mut ids = self.tokens.eos.clone();
        ids.extend(&self.tokens.turn_markers);
        if !tools_declared {
            ids.extend(self.tokens.tool_calls);
            ids.extend(self.tokens.tool_call);
        }
        ids.sort_unstable();
        ids.dedup();
        ids
    }

    pub fn render(&self, body: &Value, options: &Hy4PromptOptions) -> Result<String, String> {
        self.template
            .render(&prompt::template_context_with_max(body, options, self.allow_max_effort)?)
            .map_err(|error| format!("Hy4 chat template: {error:#}"))
    }
}

#[cfg(test)]
pub(crate) mod fixtures {
    pub const TEMPLATE: &str = include_str!("hy4_flash/fixtures/hy4_flash.jinja");
    pub fn encoding() -> super::Hy4Encoding {
        super::Hy4Encoding::new(TEMPLATE, super::Hy4TokenIds::hy4_flash()).unwrap()
    }
}

#[cfg(test)]
mod tests;
