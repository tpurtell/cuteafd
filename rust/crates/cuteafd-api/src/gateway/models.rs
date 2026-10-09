//! Model aliasing: which requested model ids the gateway accepts and what it
//! runs for them.
//!
//! Claude Code and Codex CLI ask for their own model ids (main model, small
//! or fast model, subagent model) and some call a listing endpoint. With
//! `accept_any`, every id resolves to the served model and is echoed back to
//! the client unchanged, so the CLI keeps working without per-client config.
use super::backend::ModelInfo;
use super::error::GatewayError;

#[derive(Debug, Clone, Default)]
pub struct ModelMap {
    /// The model the backend runs when nothing more specific matches.
    pub served: String,
    /// Accept any requested id and run `served` for it.
    pub accept_any: bool,
    /// Explicit `(requested, served)` aliases, checked first. A requested
    /// pattern ending in `*` matches by prefix (`claude-*`, `gpt-*`).
    pub aliases: Vec<(String, String)>,
    /// Extra ids to advertise in listings (e.g. the ids a CLI's picker
    /// expects), each resolving through the rules above.
    pub listed: Vec<String>,
}

/// Model ids Claude Code shows and requests (its `/model` picker, default
/// main/small/subagent slots). Claude Code's `GET /v1/models?limit=1000`
/// discovery ignores ids without `claude`/`anthropic` in them, so alias mode
/// advertises these. Taken from Claude Code 2.1.289 (strings in its binary);
/// refresh with `scripts/gateway/official-model-names.py`, or override at run
/// time with `--official-model-names-file`.
pub const CLAUDE_CODE_MODELS: &[&str] = &[
    "claude-opus-5-5", "claude-opus-5", "claude-sonnet-5-5", "claude-sonnet-5", "claude-fable-5-1",
    "claude-fable-5", "claude-opus-4-8", "claude-opus-4-7", "claude-opus-4-6", "claude-sonnet-4-6",
    "claude-opus-4-5", "claude-sonnet-4-5", "claude-haiku-5-5", "claude-haiku-4-5", "claude-haiku-4-5-20251001",
];

/// Model slugs Codex CLI (0.161) lists in its picker (`~/.codex/models_cache.json`,
/// `visibility` list/hide); refreshed the same way. Codex picks tool
/// shapes per slug from its bundled model metadata, so serving under a known
/// slug gets the full tool set (unified exec, freeform apply_patch).
pub const CODEX_MODELS: &[&str] = &[
    "gpt-6.1-sol", "gpt-6-sol", "gpt-6-astra", "gpt-6-luna", "gpt-5.6-sol", "gpt-5.6-terra", "gpt-5.6-luna", "gpt-5.5",
];

impl ModelMap {
    pub fn single(served: impl Into<String>) -> Self {
        Self { served: served.into(), ..Self::default() }
    }

    /// Accept any requested id and advertise the ids Claude Code and Codex
    /// expect, all running `served`.
    pub fn official_names(served: impl Into<String>) -> Self {
        let listed = CLAUDE_CODE_MODELS.iter().chain(CODEX_MODELS).map(|id| id.to_string()).collect();
        Self { served: served.into(), accept_any: true, aliases: Vec::new(), listed }
    }

    /// Apply an official-names file so ids can be refreshed without a
    /// rebuild (`scripts/gateway/official-model-names.py` writes one from the
    /// installed CLIs). Accepted forms: one id per line (`#` comments, blank
    /// lines ignored; a first line `!replace` drops the built-in list), or
    /// JSON `["id", ...]` / `{"replace": bool, "models": ["id", ...]}`.
    /// Ids extend the built-in list unless the file asks to replace it.
    /// Requests are unaffected either way: alias mode accepts any id.
    pub fn apply_names_file(&mut self, text: &str) -> Result<(), String> {
        let (replace, ids): (bool, Vec<String>) = match serde_json::from_str::<serde_json::Value>(text) {
            Ok(serde_json::Value::Array(items)) => (false, json_ids(&items)?),
            Ok(serde_json::Value::Object(map)) => {
                let items = map.get("models").and_then(|v| v.as_array()).ok_or("JSON names file needs a \"models\" array")?;
                (map.get("replace").and_then(|v| v.as_bool()).unwrap_or(false), json_ids(items)?)
            }
            _ => {
                let mut lines = text.lines().map(str::trim).filter(|l| !l.is_empty() && !l.starts_with('#')).peekable();
                let replace = lines.peek() == Some(&"!replace");
                if replace { lines.next(); }
                (replace, lines.map(str::to_string).collect())
            }
        };
        if let Some(bad) = ids.iter().find(|id| id.chars().any(char::is_whitespace)) {
            return Err(format!("model id {bad:?} contains whitespace"));
        }
        if replace { self.listed.clear(); }
        for id in ids { if !self.listed.contains(&id) { self.listed.push(id); } }
        self.accept_any = true;
        Ok(())
    }

    /// Resolve a requested id to the served id.
    pub fn resolve(&self, requested: &str) -> Result<String, GatewayError> {
        if requested.is_empty() || requested == self.served { return Ok(self.served.clone()); }
        for (pattern, target) in &self.aliases {
            let hit = match pattern.strip_suffix('*') {
                Some(prefix) => requested.starts_with(prefix),
                None => pattern == requested,
            };
            if hit { return Ok(target.clone()); }
        }
        if self.accept_any { return Ok(self.served.clone()); }
        Err(GatewayError::not_found(format!("model '{requested}' is not served here (serving '{}')", self.served))
            .with_param("model"))
    }

    /// Ids to list: the served model first, then advertised extras.
    pub fn listing(&self, backend: &[ModelInfo]) -> Vec<ModelInfo> {
        let base = backend.iter().find(|m| m.id == self.served).cloned().unwrap_or_else(|| ModelInfo {
            id: self.served.clone(), context_tokens: None, max_output_tokens: None, owned_by: "cuteafd".into(),
        });
        let mut out = vec![base.clone()];
        for id in &self.listed {
            if out.iter().any(|m| &m.id == id) { continue; }
            out.push(ModelInfo { id: id.clone(), ..base.clone() });
        }
        out
    }
}

fn json_ids(items: &[serde_json::Value]) -> Result<Vec<String>, String> {
    items.iter().map(|v| v.as_str().map(str::to_string).ok_or_else(|| "model ids must be strings".to_string())).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn exact_prefix_and_any() {
        let mut map = ModelMap::single("deepseek-v4-flash");
        assert!(map.resolve("claude-sonnet-4-5").is_err());
        map.aliases.push(("claude-*".into(), "deepseek-v4-flash".into()));
        assert_eq!(map.resolve("claude-sonnet-4-5").unwrap(), "deepseek-v4-flash");
        assert!(map.resolve("gpt-5.1-codex").is_err());
        map.accept_any = true;
        assert_eq!(map.resolve("gpt-5.1-codex").unwrap(), "deepseek-v4-flash");
        assert_eq!(map.resolve("").unwrap(), "deepseek-v4-flash");
    }
    #[test]
    fn official_names_list_claude_ids_for_discovery() {
        let map = ModelMap::official_names("served");
        let ids: Vec<String> = map.listing(&[]).into_iter().map(|m| m.id).collect();
        assert_eq!(ids[0], "served");
        assert!(ids.iter().filter(|id| id.contains("claude")).count() >= 10);
        assert!(ids.iter().any(|id| id == "gpt-6.1-sol"));
        assert_eq!(map.resolve("claude-opus-5-5").unwrap(), "served");
    }
    #[test]
    fn names_file_extends_or_replaces() {
        let mut map = ModelMap::official_names("served");
        let builtin = map.listed.len();
        map.apply_names_file("# refreshed\nclaude-opus-6\n\nclaude-opus-5-5\n").unwrap();
        assert_eq!(map.listed.len(), builtin + 1);
        map.apply_names_file("!replace\nclaude-x").unwrap();
        assert_eq!(map.listed, vec!["claude-x"]);
        map.apply_names_file(r#"{"replace": true, "models": ["a", "b"]}"#).unwrap();
        assert_eq!(map.listed, vec!["a", "b"]);
        map.apply_names_file(r#"["c"]"#).unwrap();
        assert_eq!(map.listed, vec!["a", "b", "c"]);
        assert!(map.apply_names_file(r#"[1]"#).is_err());
        assert!(map.apply_names_file("bad id").is_err());
        assert_eq!(map.resolve("anything-stale").unwrap(), "served", "stale lists never affect requests");
    }
}
