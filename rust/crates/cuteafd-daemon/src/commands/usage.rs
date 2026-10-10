//! `cuteafd usage`: manage the usage history without a browser, and a
//! CPU-only demo server over synthetic history for page development.
use anyhow::{Context, Result};
use clap::{Args, Subcommand};
use cuteafd_api::usage::{Record, UsageSink};
use cuteafd_api::usage_log::{LogRecord, LogSink, ResponsePayload};
use serde_json::json;
use std::{io::Write, net::SocketAddr, path::PathBuf, sync::Arc};

#[derive(Debug, Args)]
pub(crate) struct UsageArgs {
    #[command(subcommand)]
    command: UsageCommand,
}

#[derive(Debug, Subcommand)]
enum UsageCommand {
    /// Delete both tiers: request metadata, daily rollups and the full log with its media.
    Clear(Dir),
    /// Delete only the full log (payloads and media files); metadata stays.
    ClearLog(Dir),
    /// Write retained request metadata as CSV (never payloads).
    ExportCsv(ExportArgs),
    /// Serve /usage over a synthetic history (no GPUs, no model).
    Demo(DemoArgs),
}

#[derive(Debug, Args)]
struct Dir {
    /// The usage directory (`~/.cache/cuteafd/<instance>/usage`; run.sh uses `default`).
    #[arg(long, default_value_os_t = default_dir())]
    dir: PathBuf,
}

#[derive(Debug, Args)]
struct ExportArgs {
    #[command(flatten)]
    dir: Dir,
    /// Range ending now: 1h, 6h, 24h or 7d.
    #[arg(long, default_value = "7d")]
    range: String,
    /// Include benchmark requests.
    #[arg(long)]
    bench: bool,
    /// Output file (stdout when absent).
    #[arg(long)]
    out: Option<PathBuf>,
}

#[derive(Debug, Args)]
struct DemoArgs {
    #[arg(long, default_value = "127.0.0.1:8090")]
    listen: SocketAddr,
    /// Directory for the synthetic history (created; reused when it has one).
    #[arg(long)]
    dir: PathBuf,
    /// Console secret file; without one every data route stays locked.
    #[arg(long)]
    console_secret_file: Option<PathBuf>,
}

fn default_dir() -> PathBuf {
    std::env::var_os("HOME").map(PathBuf::from).unwrap_or_default().join(".cache/cuteafd/default/usage")
}

fn open(dir: &std::path::Path) -> Result<Arc<cuteafd_usage::Store>> {
    anyhow::ensure!(dir.join("usage.sqlite").exists(), "no usage history at {}", dir.display());
    cuteafd_usage::Store::open(Some(dir)).context("open usage history")
}

pub(crate) async fn run(args: UsageArgs) -> Result<()> {
    match args.command {
        UsageCommand::Clear(d) => tokio::task::spawn_blocking(move || {
            let store = open(&d.dir)?;
            store.log.clear()?;
            store.clear()?;
            eprintln!("cleared usage metadata and the full log in {}", d.dir.display());
            Ok(())
        })
        .await?,
        UsageCommand::ClearLog(d) => tokio::task::spawn_blocking(move || {
            let store = open(&d.dir)?;
            store.log.clear()?;
            eprintln!("cleared the full log in {}", d.dir.display());
            Ok(())
        })
        .await?,
        UsageCommand::ExportCsv(a) => tokio::task::spawn_blocking(move || export(a)).await?,
        UsageCommand::Demo(a) => demo(a).await,
    }
}

fn csv_field(v: &serde_json::Value) -> String {
    let s = match v {
        serde_json::Value::Null => return String::new(),
        serde_json::Value::String(s) => s.clone(),
        other => other.to_string(),
    };
    if s.contains([',', '"', '\n', '\r']) {
        format!("\"{}\"", s.replace('"', "\"\""))
    } else {
        s
    }
}

fn export(a: ExportArgs) -> Result<()> {
    let store = open(&a.dir.dir)?;
    let filter = cuteafd_usage::query::Filter { range: Some(a.range), bench: Some(a.bench), ..Default::default() };
    let rows = store.query_rows(&filter)?;
    let mut out: Box<dyn Write> = match &a.out {
        Some(path) => Box::new(std::io::BufWriter::new(std::fs::File::create(path)?)),
        None => Box::new(std::io::BufWriter::new(std::io::stdout().lock())),
    };
    let columns = serde_json::to_value(Record::default())?
        .as_object()
        .expect("record object")
        .keys()
        .cloned()
        .collect::<Vec<_>>();
    writeln!(out, "{}", columns.join(","))?;
    for r in &rows {
        let v = serde_json::to_value(r)?;
        writeln!(out, "{}", columns.iter().map(|c| csv_field(&v[c])).collect::<Vec<_>>().join(","))?;
    }
    out.flush()?;
    eprintln!("exported {} requests", rows.len());
    Ok(())
}

/// Seven days of plausible traffic plus chained full-log sessions.
fn synthesize(store: &cuteafd_usage::Store) {
    let now = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap_or_default().as_millis() as i64;
    let mut seed = 0x9e37_79b9_7f4a_7c15u64;
    let mut rand = move || { seed ^= seed << 13; seed ^= seed >> 7; seed ^= seed << 17; (seed % 10_000) as f64 / 10_000. };
    let clients = [("claude_code", "messages", "/v1/messages"), ("codex", "responses", "/v1/responses"),
        ("openai_sdk", "chat", "/v1/chat/completions"), ("curl", "chat", "/v1/chat/completions"), ("bench", "chat", "/v1/chat/completions")];
    for i in 0..3000i64 {
        let (client, protocol, route) = clients[(rand() * clients.len() as f64) as usize % clients.len()];
        let ts = now - (rand() * 7. * 86_400_000.) as i64;
        let input = (200. + rand() * rand() * 60_000.) as u64;
        let cached = if rand() < 0.7 { (input as f64 * rand()) as u64 } else { 0 };
        let output = (20. + rand() * 1500.) as u64;
        let queue = rand() * rand() * 400.;
        let admit = queue + 2.;
        let prefill_rate = 3000. + rand() * 5000.;
        let ttft = admit + (input - cached) as f64 * 1000. / prefill_rate + 5.;
        let decode_rate = 60. + rand() * 70.;
        let retire = ttft + output.saturating_sub(1) as f64 * 1000. / decode_rate;
        let outcome = if rand() < 0.03 { "engine_error" } else if rand() < 0.04 { "cancelled" } else { "ok" };
        store.record(Record {
            rid: format!("demo-{i:05}"), ts_ms: ts, protocol: protocol.into(), route: route.into(), method: "POST".into(),
            client_kind: client.into(), model_requested: Some("GLM-5.3-Flash".into()), model_served: Some(if rand() < 0.7 { "GLM-5.3-Flash" } else { "DeepSeek-V4.1-Flash" }.into()),
            session_id: Some(format!("s{}", (rand() * 40.) as u32)), session_source: Some(if client == "claude_code" { "cache_key" } else { "prefix" }.into()),
            stream: true, tokens_in: Some(input), tokens_cached: Some(cached), tokens_out: Some(output), tokens_reasoning: Some(output / 3),
            draft_proposed: Some(output * 3), draft_accepted: Some((output as f64 * 3. * (0.4 + rand() * 0.4)) as u64), rounds: Some(output / 3 + 1),
            t_queue_ms: Some(queue), t_admit_ms: Some(admit), t_ttft_ms: Some(ttft), t_retire_ms: Some(retire), t_total_ms: Some(retire + 3.),
            decode_tps: Some(output.saturating_sub(1) as f64 * 1000. / (retire - ttft).max(1.)),
            prefill_tps: Some((input - cached) as f64 * 1000. / (ttft - admit)),
            concurrency_engine: Some((1. + rand() * 8.) as u64), status: if outcome == "engine_error" { 500 } else { 200 }, outcome: outcome.into(),
            stop_reason: Some(if outcome == "cancelled" { "cancelled" } else if rand() < 0.5 { "tool_use" } else { "end_turn" }.into()),
            error_class: (outcome == "engine_error").then(|| "worker".into()), bench: client == "bench",
            ..Default::default()
        });
    }
    // Two chained conversations (one with an edit) and a Responses chain, with an image.
    // A 48x48 gradient PNG, so the conversation view shows a stored image.
    const IMAGE: &str = "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";
    let send = |rid: &str, ts: i64, protocol: &str, session: &str, request: serde_json::Value, response: serde_json::Value| {
        LogSink::record_log(&*store.log, LogRecord {
            meta: Record { rid: rid.into(), ts_ms: ts, protocol: protocol.into(), client_kind: if protocol == "messages" { "claude_code" } else { "codex" }.into(),
                model_served: Some("GLM-5.3-Flash".into()), session_id: Some(session.into()), session_source: Some("cache_key".into()),
                status: 200, outcome: "ok".into(), ..Default::default() },
            request: vec![serde_json::to_vec(&request).expect("json").into()], request_value: None, request_truncated: false,
            response: ResponsePayload::Object(serde_json::to_vec(&response).expect("json").into()),
        });
    };
    let tools = json!([{"name":"Bash","description":"Run a shell command","input_schema":{"type":"object"}},{"name":"Read","input_schema":{"type":"object"}}]);
    let mut history = vec![json!({"role":"user","content":[{"type":"text","text":"Find why the build fails and fix it."},{"type":"image","source":{"type":"base64","media_type":"image/png","data":IMAGE}}]})];
    let turns = [
        (json!([{"type":"thinking","thinking":"Run the build first."},{"type":"tool_use","id":"t1","name":"Bash","input":{"command":"cargo build"}}]), "tool_use"),
        (json!([{"type":"tool_use","id":"t2","name":"Read","input":{"file_path":"src/lib.rs"}}]), "tool_use"),
        (json!([{"type":"text","text":"The missing import is fixed; the build passes."}]), "end_turn"),
    ];
    let results = ["error[E0432]: unresolved import `crate::usage`", "pub mod usage_log;\npub mod usage;"];
    for (i, (content, stop)) in turns.iter().enumerate() {
        let ts = now - 3_600_000 + i as i64 * 60_000;
        send(&format!("demo-cc-{i}"), ts, "messages", "cc-1",
            json!({"model":"claude-opus","max_tokens":4096,"system":"You are Claude Code.","tools":tools,"messages":history,"stream":true}),
            json!({"type":"message","role":"assistant","content":content,"stop_reason":stop,"usage":{"input_tokens":12000 + i * 900,"cache_read_input_tokens":11000 + i * 800,"output_tokens":120}}));
        history.push(json!({"role":"assistant","content":content}));
        if let Some(result) = results.get(i) {
            history.push(json!({"role":"user","content":[{"type":"tool_result","tool_use_id":format!("t{}", i + 1),"content":result}]}));
        }
    }
    // The user rewinds and edits the first message.
    send("demo-cc-edit", now - 3_000_000, "messages", "cc-1",
        json!({"model":"claude-opus","max_tokens":4096,"system":"You are Claude Code.","tools":tools,"messages":[history[0].clone(), history[1].clone(), json!({"role":"user","content":"Stop: just explain the error."})],"stream":true}),
        json!({"type":"message","role":"assistant","content":[{"type":"text","text":"The crate root never declares `mod usage`."}],"stop_reason":"end_turn","usage":{"input_tokens":12400,"output_tokens":40}}));
    let out = |id: &str, text: &str| json!({"id":id,"object":"response","status":"completed","output":[{"type":"message","role":"assistant","content":[{"type":"output_text","text":text}]}],"usage":{"input_tokens":800,"output_tokens":30}});
    send("demo-cx-0", now - 1_800_000, "responses", "cx-1", json!({"model":"gpt-6.1-sol","instructions":"You are Codex.","input":"List the crates."}), out("resp_demo_0", "cuteafd-api, cuteafd-usage, cuteafd-daemon, ..."));
    send("demo-cx-1", now - 1_700_000, "responses", "cx-1", json!({"model":"gpt-6.1-sol","previous_response_id":"resp_demo_0","input":[{"type":"message","role":"user","content":"Which one holds the log writer?"}]}), out("resp_demo_1", "cuteafd-usage (src/log)."));
    let _ = store.flush();
    let _ = store.log.flush();
}

async fn demo(a: DemoArgs) -> Result<()> {
    std::fs::create_dir_all(&a.dir)?;
    let fresh = !a.dir.join("usage.sqlite").exists();
    let store = cuteafd_usage::Store::open(Some(&a.dir))?;
    if fresh {
        let mut settings = store.settings();
        settings.record_bench = true;
        store.update_settings(settings)?;
        let s = store.clone();
        tokio::task::spawn_blocking(move || synthesize(&s)).await?;
    }
    let gate = a.console_secret_file.as_deref()
        .map(|p| cuteafd_api::console_gate::ConsoleGate::from_file(p, false)).transpose()?
        .unwrap_or_else(cuteafd_api::console_gate::ConsoleGate::locked);
    let app = axum::Router::new()
        .route("/", axum::routing::get(|| async { axum::response::Redirect::temporary("/usage") }))
        .merge(cuteafd_api::openai::console::asset_routes());
    let app = gate.mount(cuteafd_usage::http::mount(app, store.clone(), gate.clone()));
    let app = app.layer(axum::middleware::from_fn_with_state(cuteafd_api::usage::Middleware::new(store), cuteafd_api::usage::track));
    let listener = tokio::net::TcpListener::bind(a.listen).await.context("bind demo listener")?;
    tracing::info!(listen = %a.listen, "usage demo ready at /usage");
    axum::serve(listener, app.into_make_service_with_connect_info::<SocketAddr>())
        .with_graceful_shutdown(async { let _ = tokio::signal::ctrl_c().await; }).await?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn csv_quotes_and_cli_parses() {
        assert_eq!(csv_field(&json!("a,\"b\"")), "\"a,\"\"b\"\"\"");
        assert_eq!(csv_field(&json!(3)), "3");
        assert_eq!(csv_field(&serde_json::Value::Null), "");
        use clap::Parser;
        for argv in [vec!["cuteafd", "usage", "clear-log", "--dir", "/tmp/x"], vec!["cuteafd", "usage", "export-csv", "--range", "24h"],
            vec!["cuteafd", "usage", "clear"], vec!["cuteafd", "usage", "demo", "--dir", "/tmp/y"]] {
            assert!(crate::cli::Cli::try_parse_from(argv).is_ok());
        }
    }
    #[test]
    fn export_and_clear_on_disk() {
        let dir = tempfile::tempdir().unwrap();
        let store = cuteafd_usage::Store::open(Some(dir.path())).unwrap();
        synthesize(&store);
        drop(store);
        let out = dir.path().join("x.csv");
        export(ExportArgs { dir: Dir { dir: dir.path().into() }, range: "7d".into(), bench: false, out: Some(out.clone()) }).unwrap();
        let text = std::fs::read_to_string(&out).unwrap();
        assert!(text.starts_with("rid,ts_ms,protocol"));
        assert!(text.lines().count() > 1000);
        assert!(!text.contains("Find why the build fails"), "metadata export never holds payloads");
        let store = open(dir.path()).unwrap();
        assert!(store.log.get("demo-cc-2").unwrap().is_some());
        store.log.clear().unwrap();
        assert!(store.log.get("demo-cc-2").unwrap().is_none());
        assert!(!store.rows().unwrap().is_empty());
        store.clear().unwrap();
        assert!(store.rows().unwrap().is_empty());
    }
}
