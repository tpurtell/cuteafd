//! MiniMax M3's checkpoint template, adaptive reasoning and namespaced XML calls.
pub mod parser;
pub mod prompt;

use anyhow::{Context, Result};
use serde_json::Value;
use std::path::Path;
use super::glm5::template::ChatTemplate;
pub use parser::{MiniMaxOutputParser, MiniMaxParserOptions};
pub use prompt::MiniMaxPromptOptions;

pub const NAMESPACE: &str = "]<]minimax[>[";
pub const THINK_OPEN: &str = "<mm:think>";
pub const THINK_CLOSE: &str = "</mm:think>";
pub const CALL_OPEN: &str = "]<]minimax[>[<tool_call>";
pub const CALL_CLOSE: &str = "]<]minimax[>[</tool_call>";
pub const STOP_MARKERS: [&str; 5] = ["[e~[", "]~b]", "]~!b[", "]!d~[", "]!p~["];

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MiniMaxTokenIds {
    pub eos: Vec<u32>,
    pub think: [u32; 2],
    pub namespace: u32,
    pub tool_call: [u32; 2],
    pub turn_markers: Vec<u32>,
}

#[derive(Debug)]
pub struct MiniMaxEncoding {
    template: ChatTemplate,
    tokens: MiniMaxTokenIds,
}

impl MiniMaxEncoding {
    /// The shipped template is adaptive and reads no thinking-off switch.
    pub fn supports_thinking_off(&self) -> bool { false }

    pub fn new(source: impl Into<String>, tokens: MiniMaxTokenIds) -> Result<Self> {
        anyhow::ensure!(!tokens.eos.is_empty(), "MiniMax EOS token list must not be empty");
        Ok(Self { template: ChatTemplate::new(source).context("compile MiniMax chat template")?, tokens })
    }

    /// Read the checkpoint's native headers, not an engine-generated side file.
    pub fn from_snapshot(snapshot: &Path) -> Result<Self> {
        let config = read_json(&snapshot.join("tokenizer_config.json"))?;
        let special = read_json(&snapshot.join("special_tokens_map.json"))?;
        let added = read_json(&snapshot.join("added_tokens.json"))?;
        let generation = read_json(&snapshot.join("generation_config.json"))?;
        let source = match std::fs::read_to_string(snapshot.join("chat_template.jinja")) {
            Ok(source) => source,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => config["chat_template"]
                .as_str().context("MiniMax checkpoint has no chat template")?.to_owned(),
            Err(error) => return Err(error).context("read MiniMax chat template"),
        };
        let id = |marker: &str| -> Result<u32> {
            let raw = added[marker].as_u64().with_context(|| format!("MiniMax added_tokens.json has no {marker} token"))?;
            let id = u32::try_from(raw).context("MiniMax token id exceeds u32")?;
            if let Some(entry) = config["added_tokens_decoder"].get(raw.to_string()) {
                anyhow::ensure!(entry["content"] == marker, "MiniMax tokenizer headers disagree for {marker}");
            }
            Ok(id)
        };
        for key in ["bos_token", "eos_token"] {
            let text = token_content(&config[key]).with_context(|| format!("MiniMax {key}"))?;
            anyhow::ensure!(token_content(&special[key]) == Some(text), "MiniMax special token headers disagree for {key}");
            let token_id = id(text)?;
            if let Some(raw) = generation[format!("{key}_id")].as_u64() {
                anyhow::ensure!(raw == u64::from(token_id), "MiniMax generation and tokenizer headers disagree for {key}");
            }
        }
        let eos = match &generation["eos_token_id"] {
            Value::Null => vec![id(token_content(&config["eos_token"]).context("MiniMax eos_token")?)?],
            Value::Array(ids) => ids.iter().map(|value| {
                Ok(u32::try_from(value.as_u64().context("invalid MiniMax eos_token_id")?)?)
            }).collect::<Result<Vec<_>>>()?,
            value => vec![u32::try_from(value.as_u64().context("invalid MiniMax eos_token_id")?)?],
        };
        Self::new(source, MiniMaxTokenIds { eos, think: [id(THINK_OPEN)?, id(THINK_CLOSE)?],
            namespace: id(NAMESPACE)?, tool_call: [id("<tool_call>")?, id("</tool_call>")?],
            turn_markers: STOP_MARKERS[1..].iter().map(|marker| id(marker)).collect::<Result<_>>()? })
    }

    pub fn tokens(&self) -> &MiniMaxTokenIds { &self.tokens }

    pub fn render(&self, body: &Value, options: &MiniMaxPromptOptions) -> Result<String, String> {
        self.template.render(&prompt::template_context(body, options)?)
            .map_err(|error| format!("MiniMax chat template: {error:#}"))
    }

    pub fn stop_token_ids(&self, tools_declared: bool) -> Vec<u32> {
        let mut ids = self.tokens.eos.clone();
        ids.extend(&self.tokens.turn_markers);
        // The namespace is a separate special token preceding every XML tag.
        if !tools_declared { ids.push(self.tokens.namespace); ids.extend(self.tokens.tool_call); }
        ids.sort_unstable();
        ids.dedup();
        ids
    }
}

fn token_content(value: &Value) -> Option<&str> {
    value.as_str().or_else(|| value.get("content").and_then(Value::as_str))
}

fn read_json(path: &Path) -> Result<Value> {
    serde_json::from_str(&std::fs::read_to_string(path).with_context(|| format!("read {}", path.display()))?)
        .with_context(|| format!("parse {}", path.display()))
}

#[cfg(test)]
pub(crate) mod fixtures {
    // MiniMaxAI/MiniMax-M3.1-Flash@54e4e69dfc452b1e94a426c24a4fdeeaa8e3ad5e.
    // Unmodified template and headers; goldens use Transformers 4.57.1/Jinja2 3.1.6.
    pub const TEMPLATE: &str = include_str!("minimax/fixtures/minimax_m3.jinja");
    pub const GOLDENS: &str = include_str!("minimax/fixtures/minimax_template_goldens.json");
    pub fn encoding() -> super::MiniMaxEncoding {
        super::MiniMaxEncoding::new(TEMPLATE, super::MiniMaxTokenIds { eos: vec![200020],
            think: [200059, 200060], namespace: 200058, tool_call: [200052, 200053],
            turn_markers: vec![200019, 200034, 200021, 200000] }).unwrap()
    }
}

#[cfg(test)]
mod tests;
