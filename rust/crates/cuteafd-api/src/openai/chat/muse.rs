//! Muse Spark chat: the checkpoint's own template and its recipient-addressed
//! message stream (`<|start|>assistant to=RECIPIENT<|message|>…<|eom|>`).
//!
//! The prompt ends at `<|start|>assistant`, so generation begins with the
//! first message's ` to=RECIPIENT` header. Each message ends with `<|eom|>`
//! (the turn continues: another header follows) or `<|eot|>` (the turn ends,
//! a stop token). Recipients: `self` is private reasoning, `commentary` and
//! `user` are visible content, anything else names a tool and its body is an
//! `<atem:function_calls>` block.
pub mod parser;
pub mod prompt;

use anyhow::{bail, Context, Result};
use serde_json::Value;
use std::path::Path;

use super::glm5::template::ChatTemplate;
pub use parser::{MuseOutputParser, MuseParserOptions};
pub use prompt::{MusePromptOptions, MuseToolChoice};

pub const START: &str = "<|start|>";
pub const MESSAGE: &str = "<|message|>";
pub const END_OF_MESSAGE: &str = "<|eom|>";
pub const END_OF_TURN: &str = "<|eot|>";
pub const END_OF_TEXT: &str = "<|end_of_text|>";

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MuseTokenIds {
    pub bos: u32,
    /// `generation_config.eos_token_id`: `<|end_of_text|>` and `<|eot|>`.
    pub eos: Vec<u32>,
    pub start: u32,
    pub message: u32,
    pub end_of_message: u32,
}

#[derive(Debug)]
pub struct MuseEncoding {
    template: ChatTemplate,
    tokens: MuseTokenIds,
    bos_token: String,
}

impl MuseEncoding {
    pub fn new(source: impl Into<String>, tokens: MuseTokenIds, bos_token: String) -> Result<Self> {
        anyhow::ensure!(!tokens.eos.is_empty(), "Muse EOS token list must not be empty");
        Ok(Self { template: ChatTemplate::new(source).context("compile Muse chat template")?, tokens, bos_token })
    }

    /// The checkpoint's own template and tokenizer headers, without side files.
    pub fn from_snapshot(snapshot: &Path) -> Result<Self> {
        let config = read_json(&snapshot.join("tokenizer_config.json"))?;
        let source = match std::fs::read_to_string(snapshot.join("chat_template.jinja")) {
            Ok(source) => source,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => config["chat_template"]
                .as_str().context("Muse checkpoint has no chat template")?.to_owned(),
            Err(error) => return Err(error).context("read Muse chat template"),
        };
        let tokenizer = read_json(&snapshot.join("tokenizer.json"))?;
        let added = tokenizer["added_tokens"].as_array().context("tokenizer.json added_tokens")?;
        let id = |content: &str| -> Result<u32> {
            let raw = added.iter().find(|token| token["content"] == content)
                .and_then(|token| token["id"].as_u64())
                .with_context(|| format!("Muse tokenizer has no {content} token"))?;
            u32::try_from(raw).context("Muse token id exceeds u32")
        };
        let bos_token = token_content(&config["bos_token"]).context("Muse bos_token")?.to_owned();
        let generation = read_json(&snapshot.join("generation_config.json"))?;
        let eos = match generation.get("eos_token_id") {
            None | Some(Value::Null) => vec![id(token_content(&config["eos_token"]).context("Muse eos_token")?)?],
            Some(Value::Number(number)) => vec![u32::try_from(number.as_u64().context("invalid eos_token_id")?)?],
            Some(Value::Array(ids)) => ids.iter().map(|value| {
                Ok(u32::try_from(value.as_u64().context("invalid eos_token_id")?)?)
            }).collect::<Result<Vec<_>>>()?,
            Some(_) => bail!("Muse eos_token_id must be an integer or integer array"),
        };
        let tokens = MuseTokenIds { bos: id(&bos_token)?, eos, start: id(START)?, message: id(MESSAGE)?,
            end_of_message: id(END_OF_MESSAGE)? };
        anyhow::ensure!(tokens.eos.contains(&id(END_OF_TURN)?), "Muse generation_config must stop at {END_OF_TURN}");
        Self::new(source, tokens, bos_token)
    }

    pub fn tokens(&self) -> &MuseTokenIds { &self.tokens }

    pub fn render(&self, body: &Value, options: &MusePromptOptions) -> Result<String, String> {
        let context = prompt::template_context(body, options, &self.bos_token)?;
        self.template.render(&context).map_err(|error| format!("Muse chat template: {error:#}"))
    }

    /// The checkpoint's EOS ids (`<|eot|>` ends the assistant turn).
    pub fn stop_token_ids(&self) -> Vec<u32> {
        let mut ids = self.tokens.eos.clone();
        ids.sort_unstable();
        ids.dedup();
        ids
    }
}

fn token_content(value: &Value) -> Option<&str> {
    value.as_str().or_else(|| value.get("content").and_then(Value::as_str))
}

fn read_json(path: &Path) -> Result<Value> {
    let text = std::fs::read_to_string(path).with_context(|| format!("read {}", path.display()))?;
    serde_json::from_str(&text).with_context(|| format!("parse {}", path.display()))
}

#[cfg(test)]
pub(crate) mod fixtures {
    pub const TEMPLATE: &str = include_str!("muse/fixtures/muse_spark.jinja");
    pub const GOLDENS: &str = include_str!("muse/fixtures/muse_template_goldens.json");

    /// The checkpoint's ids (tokenizer.json at rev 0fb02478).
    pub fn encoding() -> super::MuseEncoding {
        super::MuseEncoding::new(TEMPLATE, super::MuseTokenIds { bos: 200_000, eos: vec![200_001, 200_008],
            start: 200_022, message: 200_023, end_of_message: 200_007 }, "<|begin_of_text|>".into()).unwrap()
    }
}

#[cfg(test)]
mod tests;
