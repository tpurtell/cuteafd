//! The agentic reasoning-on coding session (scripts/bench/bench-agentic-session.py's
//! record loop, in process): a coding agent with six tools works a seeded task
//! in the fixture repository (scripts/fixtures/agentic-repo, in memory, four
//! planted bugs), thinking on, reasoning echoed back, until it answers without
//! tool calls or 12 turns pass. Per turn: TTFT, emitted tok/s, cache hits;
//! the session's success is the task's tests passing.
use super::common::table;
use super::{Ctx, Panel, Rates};
use crate::report::ServerInfo;
use anyhow::Result;
use serde_json::{json, Value};
use std::collections::BTreeMap;

include!(concat!(env!("OUT_DIR"), "/agentic_repo.rs"));
const RUNNER: &str = include_str!("../../assets/agentic-runner.py");
const SYSTEM: &str = include_str!("../../assets/agentic-system.txt");
const MAX_TURNS: usize = 12;
const MAX_READ_LINES: usize = 400;

/// A fixture repository file's text (for workloads that quote it).
pub(crate) fn fixture(path: &str) -> Option<&'static str> {
    AGENTIC_REPO.iter().find(|(name, _)| *name == path).map(|(_, text)| *text)
}

pub struct Agentic;
pub static AGENTIC: Agentic = Agentic;

/// (name, tests, prompt)
const TASKS: [(&str, &[&str], &str); 4] = [
    ("money", &["tests/test_money.py"], "Our auditors report that splitting a negative expense in half is off by a \
        cent: `Money(-1005).scale(1, 2)` gives -503 cents where -502 is expected, and \
        `tests/test_money.py::test_negative_rounding` fails. Find the root cause, fix it without breaking anything \
        else, and run the tests to confirm."),
    ("parser", &["tests/test_parser.py"], "Importing `data/export-2024.csv` fails with `ParseError: line 2: \
        expected 4 fields, found 5`. The bank quotes descriptions that contain commas (and doubles quotes inside \
        them), and `tests/test_parser.py` has failing tests for this. Fix the CSV handling so the whole export \
        imports, keep the other importers working, and run the tests."),
    ("rates", &["tests/test_rates.py"], "Currency conversions keep using a stale exchange rate for days, while the \
        rate provider is called on almost every lookup. `tests/test_rates.py::test_expired_rate_refetched` fails. \
        Fix the rate cache and run the tests."),
    ("dates", &["tests/test_dates.py", "tests/test_report.py"], "The December monthly report comes out empty even \
        though there were rent and dinner expenses on 2024-12-20 and 2024-12-31. \
        `tests/test_report.py::test_december_report` and `tests/test_dates.py::test_december_month_end` fail. Find \
        and fix the cause, then run the whole suite."),
];

fn tools() -> Value {
    let f = |name: &str, description: &str, parameters: Value| json!({"type": "function", "function": {
        "name": name, "description": description, "parameters": parameters}});
    json!([
        f("read_file", "Read a text file of the repository. Lines are numbered from 1. Reads at most 400 lines; use \
            start_line/end_line for longer files.", json!({"type": "object", "properties": {"path": {"type": "string"},
            "start_line": {"type": "integer", "minimum": 1}, "end_line": {"type": "integer", "minimum": 1}},
            "required": ["path"], "additionalProperties": false})),
        f("list_dir", "List a directory of the repository (directories end with '/').", json!({"type": "object",
            "properties": {"path": {"type": "string"}}, "required": ["path"], "additionalProperties": false})),
        f("grep", "Search file contents with a regular expression. Returns 'path:line: text' for at most 100 matches.",
            json!({"type": "object", "properties": {"pattern": {"type": "string"}, "path": {"type": "string"}},
            "required": ["pattern"], "additionalProperties": false})),
        f("edit", "Replace exactly one occurrence of old_string with new_string in a file. Fails if old_string is \
            missing or occurs more than once; include enough context to be unique.", json!({"type": "object",
            "properties": {"path": {"type": "string"}, "old_string": {"type": "string"}, "new_string": {"type": "string"}},
            "required": ["path", "old_string", "new_string"], "additionalProperties": false})),
        f("write_file", "Create or overwrite a file with the given content.", json!({"type": "object", "properties": {
            "path": {"type": "string"}, "content": {"type": "string"}}, "required": ["path", "content"],
            "additionalProperties": false})),
        f("run_tests", "Run the test suite (plain test functions under tests/). Optionally a file ('tests/test_x.py') \
            or one test ('tests/test_x.py::test_name').", json!({"type": "object", "properties": {"path": {"type": "string"}},
            "required": [], "additionalProperties": false})),
    ])
}

/// The fixture repository in memory; every tool is deterministic.
pub struct Workspace {
    files: BTreeMap<String, String>,
}

impl Workspace {
    pub fn new() -> Self {
        Self { files: AGENTIC_REPO.iter().map(|(p, c)| (p.to_string(), c.to_string())).collect() }
    }

    fn path(&self, path: &str) -> Result<String, String> {
        let mut parts: Vec<&str> = Vec::new();
        for part in path.trim().split('/') {
            match part {
                "" | "." => {}
                ".." => return Err(format!("path {path:?} is outside the repository")),
                p => parts.push(p),
            }
        }
        if path.trim().starts_with('/') {
            return Err(format!("path {path:?} is outside the repository"));
        }
        Ok(parts.join("/"))
    }

    fn read_file(&self, a: &Value) -> Result<String, String> {
        let name = self.path(a["path"].as_str().unwrap_or(""))?;
        let text = self.files.get(&name).ok_or(format!("no such file: {name}"))?;
        let lines: Vec<&str> = text.lines().collect();
        let start = a["start_line"].as_u64().unwrap_or(1).max(1) as usize;
        let end = (a["end_line"].as_u64().map_or(lines.len(), |e| e as usize)).min(lines.len()).min(start + MAX_READ_LINES - 1);
        let body: Vec<String> = (start..=end).filter(|n| *n >= 1 && *n <= lines.len())
            .map(|n| format!("{n:>5}\t{}", lines[n - 1])).collect();
        let more = if end < lines.len() { format!("\n[lines {}-{} not shown]", end + 1, lines.len()) } else { String::new() };
        Ok(format!("{name} ({} lines)\n{}{more}", lines.len(), body.join("\n")))
    }

    fn list_dir(&self, a: &Value) -> Result<String, String> {
        let prefix = self.path(a["path"].as_str().unwrap_or("."))?;
        let prefix = if prefix.is_empty() { prefix } else { format!("{prefix}/") };
        let mut entries: Vec<String> = self.files.keys().filter_map(|n| n.strip_prefix(&prefix)).map(|rest| match rest.split_once('/') {
            Some((dir, _)) => format!("{dir}/"),
            None => rest.to_string(),
        }).collect();
        entries.dedup();
        if entries.is_empty() { Err(format!("no such directory: {}", a["path"])) } else { Ok(entries.join("\n")) }
    }

    fn grep(&self, a: &Value) -> Result<String, String> {
        let pattern = a["pattern"].as_str().unwrap_or("");
        let scope = self.path(a["path"].as_str().unwrap_or("."))?;
        // A literal search (the Python harness uses regular expressions; plain substrings cover agent use).
        let needle = pattern.trim_matches('^').trim_matches('$').replace("\\.", ".").replace("\\(", "(").replace("\\)", ")");
        let mut matches = Vec::new();
        for (name, text) in &self.files {
            if !scope.is_empty() && *name != scope && !name.starts_with(&format!("{scope}/")) {
                continue;
            }
            for (n, line) in text.lines().enumerate() {
                if line.contains(&needle) {
                    matches.push(format!("{name}:{}: {line}", n + 1));
                }
            }
        }
        let total = matches.len();
        matches.truncate(100);
        Ok(if total == 0 { "no matches".into() } else if total > 100 {
            format!("{}\n[{} more matches]", matches.join("\n"), total - 100) } else { matches.join("\n") })
    }

    fn edit(&mut self, a: &Value) -> Result<String, String> {
        let name = self.path(a["path"].as_str().unwrap_or(""))?;
        let old = a["old_string"].as_str().unwrap_or("");
        let new = a["new_string"].as_str().unwrap_or("");
        let text = self.files.get_mut(&name).ok_or(format!("no such file: {name}"))?;
        let count = if old.is_empty() { 0 } else { text.matches(old).count() };
        if count != 1 {
            return Err(if count == 0 { "old_string not found".into() } else { format!("old_string is not unique ({count} matches)") });
        }
        *text = text.replacen(old, new, 1);
        Ok(format!("edited {name}: -{} +{} lines", old.matches('\n').count() + 1, new.matches('\n').count() + 1))
    }

    fn write_file(&mut self, a: &Value) -> Result<String, String> {
        let name = self.path(a["path"].as_str().unwrap_or(""))?;
        if name.is_empty() {
            return Err("a file path is required".into());
        }
        let content = a["content"].as_str().unwrap_or("").to_string();
        let bytes = content.len();
        self.files.insert(name.clone(), content);
        Ok(format!("wrote {bytes} bytes to {name}"))
    }

    pub fn run_tests(&self, selection: &str) -> String {
        let dir = std::env::temp_dir().join(format!("cuteafd-agentic-{}", uuid::Uuid::new_v4().simple()));
        for (name, text) in &self.files {
            let target = dir.join(name);
            if let Some(parent) = target.parent() {
                let _ = std::fs::create_dir_all(parent);
            }
            let _ = std::fs::write(target, text);
        }
        let output = std::process::Command::new("timeout").args(["30", "python3", "-I", "-B", "-c", RUNNER])
            .arg(&dir).arg(selection).env_clear().env("PYTHONHASHSEED", "0").env("PATH", "/usr/bin:/bin").output();
        let _ = std::fs::remove_dir_all(&dir);
        match output {
            Ok(o) => {
                let text = format!("{}{}", String::from_utf8_lossy(&o.stdout), String::from_utf8_lossy(&o.stderr))
                    .replace(&dir.display().to_string(), "<repo>");
                if text.trim().is_empty() { format!("test runner exited with {}", o.status) } else { text.trim().to_string() }
            }
            Err(e) => format!("could not run the tests: {e}"),
        }
    }

    pub fn passing(&self, tests: &[&str]) -> bool {
        tests.iter().all(|t| self.run_tests(t).lines().last().is_some_and(|l| l.ends_with(" 0 failed")))
    }

    /// Runs one tool call: (tool message, whether the call was valid).
    pub fn execute(&mut self, name: &str, arguments: &str) -> (String, bool) {
        let Ok(args) = serde_json::from_str::<Value>(if arguments.trim().is_empty() { "{}" } else { arguments }) else {
            return ("error: arguments are not JSON".into(), false);
        };
        let result = match name {
            "read_file" => self.read_file(&args),
            "list_dir" => self.list_dir(&args),
            "grep" => self.grep(&args),
            "edit" => self.edit(&args),
            "write_file" => self.write_file(&args),
            "run_tests" => Ok(self.run_tests(args["path"].as_str().unwrap_or(""))),
            other => return (format!("error: unknown tool {other:?}"), false),
        };
        match result {
            Ok(text) => (text, true),
            Err(error) => (format!("error: {error}"), true),
        }
    }

    pub fn system_prompt(&self, nonce: &str) -> String {
        let tree: Vec<String> = self.files.keys().map(|k| format!("  {k}")).collect();
        let file = |n: &str| self.files.get(n).map(|s| s.trim().to_string()).unwrap_or_default();
        SYSTEM.replace("{nonce}", nonce).replace("{tree}", &tree.join("\n")).replace("{readme}", &file("README.md"))
            .replace("{contributing}", &file("CONTRIBUTING.md")).replace("{architecture}", &file("docs/architecture.md"))
    }
}

impl Default for Workspace {
    fn default() -> Self {
        Self::new()
    }
}

impl Panel for Agentic {
    fn id(&self) -> &'static str { "agentic" }
    fn title(&self) -> &'static str { "Agentic session" }
    fn description(&self) -> &'static str {
        "A reasoning-on coding agent with six tools fixes a planted bug in a fixture repository (up to 12 turns): \
         per-turn TTFT, emitted tok/s and prefix-cache hits; success when the task's tests pass."
    }
    fn unavailable(&self, _info: &ServerInfo) -> Option<String> {
        let ok = std::process::Command::new("python3").arg("--version").output().is_ok_and(|o| o.status.success());
        (!ok).then(|| "python3 is not available to run the fixture's tests".to_string())
    }
    fn estimate_s(&self, rates: &Rates, _info: &ServerInfo) -> f64 {
        8.0 * rates.seconds(4000.0, 900.0)
    }
    fn run(&self, ctx: &Ctx<'_>) -> Result<Value> {
        let (task, tests, prompt) = TASKS[(ctx.pass as usize - 1 + ctx.history.len()) % TASKS.len()];
        let mut workspace = Workspace::new();
        let nonce = crate::text::nonce();
        let mut messages = vec![json!({"role": "system", "content": workspace.system_prompt(&nonce)}),
            json!({"role": "user", "content": prompt})];
        let mut turns: Vec<Value> = Vec::new();
        let (mut previous_prompt, mut previous_completion) = (0u64, 0u64);
        let started = std::time::Instant::now();
        for turn in 0..MAX_TURNS {
            ctx.progress.step(turn as f64 / MAX_TURNS as f64, format!("{task} · turn {}", turn + 1));
            let body = json!({"messages": messages, "tools": tools(), "tool_choice": "auto", "temperature": 0,
                "max_tokens": 8192u64.min(ctx.max_output), "reasoning_effort": "high"});
            let chat = ctx.client.chat(body, None)?;
            let t = &chat.timing;
            let valid = chat.tool_calls.iter().filter(|c| serde_json::from_str::<Value>(c["function"]["arguments"]
                .as_str().unwrap_or("")).is_ok()).count();
            turns.push(json!({"turn": turn + 1, "prompt_tokens": t.prompt_tokens, "cached_tokens": t.cached_tokens,
                "ttft_s": t.ttft_s, "decode_tok_s": t.decode_tok_s(), "completion_tokens": t.completion_tokens,
                "tool_calls": chat.tool_calls.len(), "invalid_tool_calls": chat.tool_calls.len() - valid,
                "full_turn_reused": turn > 0 && t.cached_tokens + 1 >= previous_prompt + previous_completion,
                "finish": t.finish_reason}));
            (previous_prompt, previous_completion) = (t.prompt_tokens, t.completion_tokens);
            let mut assistant = json!({"role": "assistant", "content": chat.content});
            if !chat.reasoning.is_empty() {
                assistant["reasoning_content"] = json!(chat.reasoning);
            }
            let calls: Vec<Value> = chat.tool_calls.iter().enumerate().map(|(i, c)| {
                let id = c["id"].as_str().filter(|s| !s.is_empty()).map(str::to_string).unwrap_or(format!("call_{turn}_{i}"));
                json!({"id": id, "type": "function", "function": c["function"]})
            }).collect();
            if !calls.is_empty() {
                assistant["tool_calls"] = json!(calls);
            }
            messages.push(assistant);
            ctx.progress.partial(json!({"task": task, "turns": turns}));
            if calls.is_empty() {
                break;
            }
            for call in &calls {
                let (output, _) = workspace.execute(call["function"]["name"].as_str().unwrap_or(""),
                    call["function"]["arguments"].as_str().unwrap_or(""));
                messages.push(json!({"role": "tool", "tool_call_id": call["id"], "content": output}));
            }
        }
        let success = workspace.passing(tests);
        let rows = turns.iter().map(|t| vec![t["turn"].clone(), t["prompt_tokens"].clone(), t["cached_tokens"].clone(),
            json!(t["ttft_s"].as_f64().unwrap_or(0.0) * 1e3), t["decode_tok_s"].clone(), t["tool_calls"].clone(),
            t["invalid_tool_calls"].clone()]).collect();
        Ok(json!({"task": task, "success": success, "turns": turns, "seconds": started.elapsed().as_secs_f64(),
            "table": table(&["turn", "prompt", "cached", "TTFT ms", "tok/s", "tool calls", "invalid"], rows)}))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_fixture_is_compiled_in_and_the_tools_work() {
        let mut w = Workspace::new();
        assert!(w.files.contains_key("ledger/money.py"), "{:?}", w.files.keys().take(5).collect::<Vec<_>>());
        assert!(w.execute("list_dir", r#"{"path": "."}"#).0.contains("ledger/"));
        assert!(w.execute("read_file", r#"{"path": "ledger/money.py"}"#).0.starts_with("ledger/money.py ("));
        assert!(w.execute("read_file", r#"{"path": "../etc/passwd"}"#).0.starts_with("error"));
        assert!(!w.execute("grep", r#"{"pattern": "class Money"}"#).0.contains("no matches"));
        assert!(!w.execute("nope", "{}").1);
        assert!(w.system_prompt("abc").starts_with("session abc"));
        if std::process::Command::new("python3").arg("--version").output().is_ok() {
            // The planted bug fails its test.
            assert!(!w.passing(&["tests/test_money.py"]));
        }
    }
}
