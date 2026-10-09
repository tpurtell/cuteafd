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

impl ModelMap {
    pub fn single(served: impl Into<String>) -> Self {
        Self { served: served.into(), ..Self::default() }
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
}
