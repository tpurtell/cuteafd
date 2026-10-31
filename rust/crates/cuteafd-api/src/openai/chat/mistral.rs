//! Mistral instruct chat (tekken control tokens): the checkpoint's own
//! `chat_template.jinja`, which renders exactly as `mistral_common`'s v15
//! instruct tokenizer does, and the `[THINK]`/`[TOOL_CALLS]name[ARGS]{json}`
//! output grammar Mistral's vLLM parsers read.
//!
//! The encoding is selected from the checkpoint itself ([`MistralEncoding::recognizes`]):
//! its tokenizer carries the Mistral control tokens and its template renders them.
pub mod parser;
pub mod prompt;

use anyhow::{bail, Context, Result};
use serde_json::Value;
use std::path::Path;

use super::glm5::template::ChatTemplate;
pub use parser::{MistralOutputParser, MistralParserOptions};
pub use prompt::MistralPromptOptions;

pub const INST_CLOSE: &str = "[/INST]";
pub const THINK_OPEN: &str = "[THINK]";
pub const THINK_CLOSE: &str = "[/THINK]";
pub const TOOL_CALLS: &str = "[TOOL_CALLS]";
pub const ARGS: &str = "[ARGS]";
/// Prompt-side control tokens: none may appear in a generated turn, and each
/// ends one if it does. `[TOOL_CALLS]`/`[ARGS]` are output grammar, not stops.
pub const TURN_MARKERS: [&str; 10] = ["[INST]", INST_CLOSE, "[AVAILABLE_TOOLS]", "[/AVAILABLE_TOOLS]",
    "[TOOL_RESULTS]", "[/TOOL_RESULTS]", "[SYSTEM_PROMPT]", "[/SYSTEM_PROMPT]", "[MODEL_SETTINGS]", "[/MODEL_SETTINGS]"];

/// Special-token ids the Mistral API contract depends on.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MistralTokenIds {
    pub eos: Vec<u32>,
    pub think: [u32; 2],
    pub tool_calls: u32,
    pub args: u32,
    pub turn_markers: Vec<u32>,
}

#[derive(Debug)]
pub struct MistralEncoding {
    template: ChatTemplate,
    tokens: MistralTokenIds,
    bos_token: String,
    eos_token: String,
}

impl MistralEncoding {
    /// The template always renders a reasoning effort; it has no thinking-off form.
    pub fn supports_thinking_off(&self) -> bool { false }

    pub fn new(source: impl Into<String>, tokens: MistralTokenIds, bos_token: String, eos_token: String) -> Result<Self> {
        anyhow::ensure!(!tokens.eos.is_empty(), "Mistral EOS token list must not be empty");
        Ok(Self { template: ChatTemplate::new(source).context("compile Mistral chat template")?, tokens, bos_token, eos_token })
    }

    /// Whether `snapshot` speaks this chat contract: its tokenizer has the
    /// Mistral instruct and tool-call control tokens and its template emits them.
    pub fn recognizes(snapshot: &Path) -> Result<bool> {
        let Ok(source) = read_template(snapshot) else { return Ok(false) };
        if ![THINK_OPEN, TOOL_CALLS, ARGS, INST_CLOSE].iter().all(|marker| source.contains(marker)) {
            return Ok(false);
        }
        let tokenizer = read_json(&snapshot.join("tokenizer.json"))?;
        let added = tokenizer["added_tokens"].as_array().context("tokenizer.json added_tokens")?;
        Ok([THINK_OPEN, THINK_CLOSE, TOOL_CALLS, ARGS, INST_CLOSE].iter()
            .all(|marker| added.iter().any(|token| token["content"] == *marker && token["special"] == true)))
    }

    /// Read the checkpoint's own template and tokenizer headers, without side files.
    pub fn from_snapshot(snapshot: &Path) -> Result<Self> {
        let source = read_template(snapshot)?;
        let config = read_json(&snapshot.join("tokenizer_config.json"))?;
        let tokenizer = read_json(&snapshot.join("tokenizer.json"))?;
        let added = tokenizer["added_tokens"].as_array().context("tokenizer.json added_tokens")?;
        let id = |content: &str| -> Result<u32> {
            let raw = added.iter().find(|token| token["content"] == content)
                .and_then(|token| token["id"].as_u64())
                .with_context(|| format!("Mistral tokenizer has no {content} token"))?;
            u32::try_from(raw).context("Mistral token id exceeds u32")
        };
        let bos_token = token_content(&config["bos_token"]).context("Mistral bos_token")?.to_owned();
        let eos_token = token_content(&config["eos_token"]).context("Mistral eos_token")?.to_owned();
        let generation = match std::fs::read_to_string(snapshot.join("generation_config.json")) {
            Ok(text) => Some(serde_json::from_str::<Value>(&text).context("parse generation_config.json")?),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => None,
            Err(error) => return Err(error).context("read generation_config.json"),
        };
        let eos = match generation.as_ref().and_then(|value| value.get("eos_token_id")) {
            None | Some(Value::Null) => vec![id(&eos_token)?],
            Some(Value::Number(number)) => vec![u32::try_from(number.as_u64().context("invalid eos_token_id")?)?],
            Some(Value::Array(ids)) => ids.iter().map(|value| {
                Ok(u32::try_from(value.as_u64().context("invalid eos_token_id")?)?)
            }).collect::<Result<Vec<_>>>()?,
            Some(_) => bail!("Mistral eos_token_id must be an integer or integer array"),
        };
        let tokens = MistralTokenIds { eos, think: [id(THINK_OPEN)?, id(THINK_CLOSE)?],
            tool_calls: id(TOOL_CALLS)?, args: id(ARGS)?,
            turn_markers: TURN_MARKERS.iter().map(|marker| id(marker)).collect::<Result<_>>()? };
        Self::new(source, tokens, bos_token, eos_token)
    }

    pub fn tokens(&self) -> &MistralTokenIds { &self.tokens }

    pub fn render(&self, body: &Value, options: &MistralPromptOptions) -> Result<String, String> {
        let context = prompt::template_context(body, options, &self.bos_token, &self.eos_token)?;
        self.template.render(&context).map_err(|error| format!("Mistral chat template: {error:#}"))
    }

    pub fn stop_token_ids(&self, tools_declared: bool) -> Vec<u32> {
        let mut ids = self.tokens.eos.clone();
        ids.extend(&self.tokens.turn_markers);
        if !tools_declared { ids.extend([self.tokens.tool_calls, self.tokens.args]); }
        ids.sort_unstable();
        ids.dedup();
        ids
    }
}

fn read_template(snapshot: &Path) -> Result<String> {
    match std::fs::read_to_string(snapshot.join("chat_template.jinja")) {
        Ok(source) => Ok(source),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            let config = read_json(&snapshot.join("tokenizer_config.json"))?;
            Ok(config["chat_template"].as_str().context("checkpoint has no chat template")?.to_owned())
        }
        Err(error) => Err(error).context("read chat template"),
    }
}

fn token_content(value: &Value) -> Option<&str> {
    value.as_str().or_else(|| value.get("content").and_then(Value::as_str))
}

fn read_json(path: &Path) -> Result<Value> {
    let text = std::fs::read_to_string(path).with_context(|| format!("read {}", path.display()))?;
    serde_json::from_str(&text).with_context(|| format!("parse {}", path.display()))
}

/// The checkpoint template, for native constraint tests in other crates.
#[doc(hidden)]
pub const TEMPLATE_FOR_TESTS: &str = include_str!("mistral/fixtures/mistral_large4.jinja");

#[cfg(test)]
pub(crate) mod fixtures {
    /// `mistralai/Mistral-Large-4-1T-A52B@f12dabdc` chat_template.jinja, unmodified.
    pub const TEMPLATE: &str = include_str!("mistral/fixtures/mistral_large4.jinja");
    /// Rendered by the checkpoint template under Transformers and by mistral_common's
    /// instruct tokenizer (see scripts/qualify/mistral_large4/mistral-template-goldens.py).
    pub const GOLDENS: &str = include_str!("mistral/fixtures/mistral_template_goldens.json");

    pub fn encoding() -> super::MistralEncoding {
        super::MistralEncoding::new(TEMPLATE, super::MistralTokenIds { eos: vec![2], think: [34, 35],
            tool_calls: 9, args: 32, turn_markers: vec![3, 4, 5, 6, 7, 8, 17, 18, 36, 37] },
            "<s>".into(), "</s>".into()).unwrap()
    }
}

#[cfg(test)]
mod tests;
