//! `cuteafd bench`: start a run on a server, follow its progress, write the
//! exports. The measurement happens inside the server (the same runner as the
//! dashboard); this side only asks and downloads.
use crate::report::{Report, RunStatus};
use anyhow::{bail, Context, Result};
use serde_json::{json, Value};
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::time::Duration;

#[derive(Debug, Clone, Default)]
pub struct RunOptions {
    pub url: String,
    pub profile: Option<String>,
    pub panels: Option<Vec<String>>,
    pub passes: Vec<(String, u32)>,
    /// `Some(false)` asks the server to run explicit panels without its
    /// baseline (servers that predate the field run it anyway).
    pub baseline: Option<bool>,
    /// `svg` (report, card, panels), `png` (report and card), `card` (card.png only), `json`.
    pub export: Vec<String>,
    /// Output directory; `None`: `benchmarks/<family>/<date>-<profile>-<hardware>/` under `root`.
    pub out: Option<PathBuf>,
    /// Appended to the default directory name (a smoke card's name), so runs of
    /// one checkpoint on the same hardware and day do not overwrite each other.
    pub label: Option<String>,
    pub root: PathBuf,
    pub api_key: Option<String>,
    pub quiet: bool,
    /// Cancel the run and fail when it takes longer (a server too slow to finish).
    pub deadline: Option<Duration>,
    /// Cases for the `spec_lossless_set` panel (default: the server's shipped set).
    pub lossless_set: Option<crate::panels::lossless::CaseSet>,
}

fn agent() -> ureq::Agent {
    ureq::AgentBuilder::new().timeout_connect(Duration::from_secs(10)).timeout_read(Duration::from_secs(120)).build()
}

fn authorize(request: ureq::Request, key: &Option<String>) -> ureq::Request {
    match key {
        Some(key) => request.set("authorization", &format!("Bearer {key}")),
        None => request,
    }
}

/// `benchmarks/<family>/<date>-<profile>-<hardware>`.
pub fn default_dir(root: &Path, report: &Report) -> PathBuf {
    let family = report.server.family.clone().unwrap_or_else(|| "unknown".into());
    root.join("benchmarks").join(family).join(format!("{}-{}-{}-{}", crate::render::date(&report.created),
        report.profile, model_slug(&report.server.checkpoint()), report.server.hardware.slug()))
}

/// [`default_dir`] with `-<label>` appended (slugged) when a label is given.
pub fn labeled_dir(root: &Path, report: &Report, label: Option<&str>) -> PathBuf {
    let dir = default_dir(root, report);
    match label.map(model_slug).filter(|l| !l.is_empty()) {
        Some(label) => {
            let name = format!("{}-{label}", dir.file_name().map(|n| n.to_string_lossy()).unwrap_or_default());
            dir.with_file_name(name)
        }
        None => dir,
    }
}

/// The checkpoint's last path segment, lowercase, runs of other characters as one `-`
/// (two quants of one family on the same hardware and day get separate directories).
fn model_slug(model: &str) -> String {
    let name = model.trim_end_matches('/').rsplit('/').next().unwrap_or(model).to_lowercase();
    let mut slug = String::new();
    for c in name.chars() {
        if c.is_ascii_alphanumeric() {
            slug.push(c);
        } else if !slug.is_empty() && !slug.ends_with('-') {
            slug.push('-');
        }
    }
    slug.trim_end_matches('-').to_string()
}

/// Writes the exports of `report` into `dir`.
pub fn write_exports(report: &Report, dir: &Path, kinds: &[String]) -> Result<Vec<PathBuf>> {
    std::fs::create_dir_all(dir).with_context(|| format!("creating {}", dir.display()))?;
    let mut written = Vec::new();
    let mut files: Vec<String> = Vec::new();
    let want = |kind: &str| kinds.iter().any(|k| k == kind);
    if want("json") {
        files.push("report.json".into());
    }
    if want("svg") {
        files.extend(["report.svg".to_string(), "card.svg".to_string()]);
        files.extend(crate::render::report::shown(report).iter().map(|id| format!("panel-{id}.svg")));
    }
    if want("png") {
        files.extend(["report.png".to_string(), "card.png".to_string()]);
    } else if want("card") {
        files.push("card.png".to_string());
    }
    for file in files {
        let bytes = if file.ends_with(".png") {
            let svg = crate::http::export(report, &file.replace(".png", ".svg")).context("export")?.1;
            crate::render::png::png(&String::from_utf8_lossy(&svg), 1.0)?
        } else {
            crate::http::export(report, &file).with_context(|| format!("no export {file}"))?.1
        };
        let path = dir.join(&file);
        std::fs::write(&path, bytes).with_context(|| format!("writing {}", path.display()))?;
        written.push(path);
    }
    Ok(written)
}

/// Starts a run on `options.url`, follows it to the end and writes the exports.
pub fn run(options: &RunOptions) -> Result<(Report, PathBuf)> {
    let base = options.url.trim_end_matches('/').to_string();
    let agent = agent();
    let mut body = json!({});
    if let Some(profile) = &options.profile {
        body["profile"] = json!(profile);
    }
    if let Some(panels) = &options.panels {
        body["panels"] = json!(panels);
    }
    body["passes"] = Value::Object(options.passes.iter().map(|(k, v)| (k.clone(), json!(v))).collect());
    if let Some(baseline) = options.baseline {
        body["baseline"] = json!(baseline);
    }
    if let Some(set) = &options.lossless_set {
        body["lossless_set"] = json!(crate::panels::lossless::SetRequest::of(set));
    }
    let response = authorize(agent.post(&format!("{base}/v1/bench/runs")), &options.api_key)
        .send_json(body);
    let id = match response {
        Ok(response) => response.into_json::<Value>()?["id"].as_str().context("run id")?.to_string(),
        Err(ureq::Error::Status(code, response)) => {
            bail!("starting the run: HTTP {code}: {}", response.into_string().unwrap_or_default())
        }
        Err(error) => return Err(error).context("starting the run"),
    };
    if !options.quiet {
        eprintln!("run {id} on {base}");
    }
    let followed = follow(&base, &id, options.quiet, options.deadline, &options.api_key);
    if let Err(error) = &followed {
        if error.downcast_ref::<Overdue>().is_some() {
            let _ = authorize(agent.post(&format!("{base}/v1/bench/runs/{id}/cancel")), &options.api_key).call();
            return Err(followed.unwrap_err()).with_context(|| format!("run {id} on {base}"));
        }
    }
    let served = followed.as_ref().ok().and_then(|_| authorize(agent.get(&format!("{base}/v1/bench/runs/{id}")), &options.api_key).call().ok())
        .map(|response| response.into_json::<Report>().context("the finished report"));
    let report = match served {
        Some(report) => report?,
        // The server went away (it exited after or during the run): its history is
        // on this host when the server runs here (run.sh mounts the store).
        None => match crate::store::Store::open_read(&crate::store::default_dir()).ok().and_then(|s| s.load(&id).ok().flatten())
            .filter(|r| !matches!(r.status, RunStatus::Running | RunStatus::Queued)) {
            Some(report) => {
                eprintln!("warning: the server at {base} stopped answering; run {id} read from the local store");
                report
            }
            None => return Err(followed.err().unwrap_or_else(|| anyhow::anyhow!("the server stopped answering")))
                .with_context(|| format!("run {id} on {base}")),
        },
    };
    let dir = options.out.clone().unwrap_or_else(|| labeled_dir(&options.root, &report, options.label.as_deref()));
    let written = write_exports(&report, &dir, &options.export)?;
    if !options.quiet {
        for path in &written {
            eprintln!("wrote {}", path.display());
        }
    }
    match report.status {
        RunStatus::Done => {}
        RunStatus::Cancelled => bail!("the run was cancelled"),
        _ => bail!("the run ended {:?}: {}", report.status, report.error.clone().unwrap_or_default()),
    }
    Ok((report, dir))
}

/// A followed run passed its deadline.
#[derive(Debug, thiserror::Error)]
#[error("the run did not finish within {0} s")]
struct Overdue(u64);

/// How long a server may refuse connections before a followed run counts as lost.
const UNREACHABLE: Duration = Duration::from_secs(60);

/// Prints progress lines from the event stream until run `id` finishes.
fn follow(base: &str, id: &str, quiet: bool, deadline: Option<Duration>, api_key: &Option<String>) -> Result<()> {
    let started = std::time::Instant::now();
    let overdue = || deadline.filter(|d| started.elapsed() > *d).map(|d| Overdue(d.as_secs()));
    let events = ureq::AgentBuilder::new().timeout_connect(Duration::from_secs(10))
        .timeout_read(Duration::from_secs(60)).build();
    let mut last_line = String::new();
    let mut unreachable: Option<std::time::Instant> = None;
    loop {
        let response = match authorize(events.get(&format!("{base}/v1/bench/events")), api_key).call() {
            Ok(response) => {
                unreachable = None;
                response
            }
            Err(error) => {
                // A dropped stream: check whether the run ended meanwhile, else reconnect,
                // giving up once the server has refused connections for a while.
                if finished(base, id, api_key)? {
                    return Ok(());
                }
                if let Some(overdue) = overdue() {
                    return Err(overdue.into());
                }
                let since = *unreachable.get_or_insert_with(std::time::Instant::now);
                if since.elapsed() > UNREACHABLE {
                    bail!("the server has not answered for {} s: {error}", UNREACHABLE.as_secs());
                }
                if !quiet {
                    eprintln!("event stream: {error}; reconnecting");
                }
                std::thread::sleep(Duration::from_secs(2));
                continue;
            }
        };
        for line in BufReader::new(response.into_reader()).lines() {
            if let Some(overdue) = overdue() {
                return Err(overdue.into());
            }
            let Ok(line) = line else { break };
            let Some(data) = line.strip_prefix("data: ") else { continue };
            let Ok(event) = serde_json::from_str::<Value>(data) else { continue };
            match event["type"].as_str() {
                Some("progress") if event["run"] == id && !quiet => {
                    let live = &event["live"];
                    let mut text = format!("[{:>3.0}%] {} · {} · ETA {}", 100.0 * event["fraction"].as_f64().unwrap_or(0.0),
                        event["title"].as_str().unwrap_or(""), event["label"].as_str().unwrap_or(""),
                        crate::render::seconds(event["eta_s"].as_f64().unwrap_or(0.0)));
                    if let Some(tok_s) = live["tok_s"].as_f64().filter(|v| *v > 0.0) {
                        text.push_str(&format!(" · {} tok/s", crate::render::rate(tok_s)));
                    }
                    if text != last_line {
                        eprintln!("{text}");
                        last_line = text;
                    }
                }
                Some("report") if event["run"] == id => {
                    let status = event["report"]["status"].as_str().unwrap_or("");
                    if !matches!(status, "running" | "queued") {
                        return Ok(());
                    }
                }
                _ => {}
            }
        }
        if finished(base, id, api_key)? {
            return Ok(());
        }
    }
}

fn finished(base: &str, id: &str, api_key: &Option<String>) -> Result<bool> {
    let report: Value = match authorize(agent().get(&format!("{base}/v1/bench/runs/{id}")), api_key).call() {
        Ok(response) => response.into_json()?,
        Err(_) => return Ok(false),
    };
    Ok(!matches!(report["status"].as_str(), Some("running" | "queued")))
}

/// Cancels the server's active run.
pub fn cancel(url: &str, api_key: &Option<String>) -> Result<()> {
    let base = url.trim_end_matches('/');
    let status: Value = authorize(agent().get(&format!("{base}/v1/bench/status")), api_key).call()?.into_json()?;
    let Some(id) = status["active"]["id"].as_str() else { bail!("no run is active") };
    authorize(agent().post(&format!("{base}/v1/bench/runs/{id}/cancel")), api_key).call()?;
    eprintln!("cancelled {id}");
    Ok(())
}

/// Writes `text` to `path` only when it changed.
pub fn write_if_changed(path: &Path, text: &str) -> Result<bool> {
    if std::fs::read_to_string(path).ok().as_deref() == Some(text) {
        return Ok(false);
    }
    let mut file = std::fs::File::create(path).with_context(|| format!("writing {}", path.display()))?;
    file.write_all(text.as_bytes())?;
    Ok(true)
}

#[cfg(test)]
mod tests {
    #[test]
    fn a_label_keeps_two_cards_of_one_checkpoint_apart() {
        let report = crate::sample::report(false);
        let root = std::path::Path::new("/r");
        let base = super::default_dir(root, &report);
        let a = super::labeled_dir(root, &report, Some("glm53f-exl3-min"));
        let b = super::labeled_dir(root, &report, Some("glm53f-bf16kda-c8"));
        assert_ne!(a, b);
        assert_eq!(a.parent(), base.parent());
        assert!(a.file_name().unwrap().to_string_lossy().ends_with("-glm53f-exl3-min"));
        assert_eq!(super::labeled_dir(root, &report, None), base);
    }
}
