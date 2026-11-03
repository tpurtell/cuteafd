//! `/bench` and `/v1/bench/*`, the lockout of other inference while a run is
//! active. Controls require the server's API key, including on private networks.
use crate::client::BENCH_HEADER;
use crate::profiles::Profile;
use crate::render;
use crate::report::Report;
use crate::runner::{Bench, RunRequest, StartError};
use axum::body::Body;
use axum::extract::{ConnectInfo, Path, Request, State};
use axum::http::{header, HeaderMap, HeaderValue, Method, StatusCode};
use axum::middleware::Next;
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post, put};
use axum::{Json, Router};
use anyhow::{Context, Result};
use serde_json::json;
use std::net::{IpAddr, SocketAddr};
use std::sync::Arc;

/// The benchmark page, compiled in.
pub const PAGE: &str = include_str!("../assets/bench.html");
/// The BENCHMARKING banner any page can include (`/bench/banner.js`).
pub const BANNER: &str = include_str!("../assets/banner.js");

/// Paths that run inference (and so are refused while a benchmark runs):
/// inference POSTs, and new Responses or Realtime sockets. Turns on sockets
/// opened before the run are refused by the gateway's turn gate (`gate`).
fn inference(method: &Method, headers: &HeaderMap, path: &str) -> bool {
    (method == Method::POST && matches!(path, "/v1/chat/completions" | "/v1/completions" | "/v1/responses"
        | "/v1/messages" | "/v1/embeddings"))
        || (headers.contains_key(header::UPGRADE) && matches!(path, "/v1/responses" | "/v1/realtime"))
}

/// Refuses other clients' inference with 503 + Retry-After while a run is active.
pub async fn lockout(State(bench): State<Arc<Bench>>, request: Request, next: Next) -> Response {
    if inference(request.method(), request.headers(), request.uri().path()) {
        // The run's own requests carry its token; tools it runs as subprocesses
        // (tool-eval-bench) pass it as their API key.
        let token = request.headers().get(BENCH_HEADER).and_then(|v| v.to_str().ok()).or_else(|| {
            request.headers().get(header::AUTHORIZATION).and_then(|v| v.to_str().ok())
                .and_then(|v| v.strip_prefix("Bearer ")).map(str::trim)
        });
        if let Some(retry) = bench.locked(token) {
            let mut response = (StatusCode::SERVICE_UNAVAILABLE, Json(json!({"error": {
                "message": format!("a benchmark is running on this server; retry in about {retry} s"),
                "type": "benchmark_running"}}))).into_response();
            response.headers_mut().insert(header::RETRY_AFTER, HeaderValue::from(retry));
            return response;
        }
    }
    next.run(request).await
}

/// The gateway turn gate for this benchmark: the run drives the engine through
/// chat completions with its own token, so every gateway turn (open sockets
/// included) is another client's and waits for the run to end.
pub fn gate(bench: Arc<Bench>) -> cuteafd_api::gateway::TurnGate {
    Arc::new(move || bench.locked(None))
}

/// Loopback, RFC 1918, link-local, CGNAT and IPv6 unique-local addresses.
pub fn local_network(ip: IpAddr) -> bool {
    match ip {
        IpAddr::V4(v4) => v4.is_loopback() || v4.is_private() || v4.is_link_local()
            || (v4.octets()[0] == 100 && (64..128).contains(&v4.octets()[1])),
        IpAddr::V6(v6) => v6.is_loopback() || (v6.segments()[0] & 0xfe00) == 0xfc00
            || (v6.segments()[0] & 0xffc0) == 0xfe80
            || v6.to_ipv4_mapped().is_some_and(|v4| local_network(IpAddr::V4(v4))),
    }
}

/// Benchmark controls change server state: the console unlock cookie, or the
/// API key when the operator configured one. Never open on a keyless server.
fn authorized(bench: &Bench, peer: Option<SocketAddr>, headers: &HeaderMap) -> bool {
    let _ = peer;
    bench.api_key.get().is_some_and(|key| key.accepts(headers))
        || bench.console_gate.get().is_some_and(|gate| gate.unlocked(headers))
}

fn forbidden() -> Response {
    (StatusCode::FORBIDDEN, Json(json!({"error": {"message":
        "benchmark controls need the admin console link (or the server's API key, when one is configured)",
        "type": "console_locked"}}))).into_response()
}

fn not_found(what: &str) -> Response {
    (StatusCode::NOT_FOUND, Json(json!({"error": {"message": format!("{what} not found"), "type": "not_found"}})))
        .into_response()
}

fn peer(connect: Option<ConnectInfo<SocketAddr>>) -> Option<SocketAddr> {
    connect.map(|ConnectInfo(addr)| addr)
}

async fn page() -> Response {
    let page = match std::env::var_os("CUTEAFD_BENCH_PAGE") {
        Some(path) => tokio::fs::read_to_string(&path).await.unwrap_or_else(|_| PAGE.to_string()),
        None => PAGE.to_string(),
    };
    ([(header::CACHE_CONTROL, "no-cache")], axum::response::Html(page)).into_response()
}

/// A page script, or for page development (`CUTEAFD_BENCH_PAGE` set) the
/// same-named file beside that page, re-read on every load.
async fn script(name: &str, builtin: &'static str) -> Response {
    let body = match std::env::var_os("CUTEAFD_BENCH_PAGE").map(std::path::PathBuf::from) {
        Some(page) => tokio::fs::read_to_string(page.with_file_name(name)).await.unwrap_or_else(|_| builtin.to_string()),
        None => builtin.to_string(),
    };
    ([(header::CONTENT_TYPE, "text/javascript; charset=utf-8"), (header::CACHE_CONTROL, "no-cache")], body).into_response()
}

async fn fabric_script() -> Response {
    script("fabric.js", include_str!("../assets/fabric.js")).await
}

async fn banner() -> Response {
    script("banner.js", BANNER).await
}

async fn status(State(bench): State<Arc<Bench>>) -> Json<serde_json::Value> {
    Json(bench.status())
}

async fn panels(State(bench): State<Arc<Bench>>) -> Json<serde_json::Value> {
    Json(bench.catalog())
}

async fn profiles(State(bench): State<Arc<Bench>>) -> Json<serde_json::Value> {
    Json(json!({"profiles": bench.profiles()}))
}

async fn save_profile(State(bench): State<Arc<Bench>>, connect: Option<ConnectInfo<SocketAddr>>, headers: HeaderMap,
    Path(name): Path<String>, Json(mut profile): Json<Profile>) -> Response {
    if !authorized(&bench, peer(connect), &headers) {
        return forbidden();
    }
    if crate::profiles::builtin().iter().any(|p| p.name == name) {
        return (StatusCode::CONFLICT, Json(json!({"error": {"message": "built-in profiles cannot be replaced"}})))
            .into_response();
    }
    profile.name = name;
    profile.builtin = false;
    match bench.store(|s| s.save_profile(&profile)) {
        Ok(()) => Json(json!({"saved": profile.name})).into_response(),
        Err(error) => (StatusCode::INTERNAL_SERVER_ERROR, format!("{error:#}")).into_response(),
    }
}

async fn delete_profile(State(bench): State<Arc<Bench>>, connect: Option<ConnectInfo<SocketAddr>>, headers: HeaderMap,
    Path(name): Path<String>) -> Response {
    if !authorized(&bench, peer(connect), &headers) {
        return forbidden();
    }
    match bench.store(|s| s.delete_profile(&name)) {
        Ok(true) => Json(json!({"deleted": name})).into_response(),
        Ok(false) => not_found("profile"),
        Err(error) => (StatusCode::INTERNAL_SERVER_ERROR, format!("{error:#}")).into_response(),
    }
}

async fn runs(State(bench): State<Arc<Bench>>) -> Response {
    match bench.store(|s| s.list(200)) {
        Ok(rows) => Json(json!({"runs": rows})).into_response(),
        Err(error) => (StatusCode::INTERNAL_SERVER_ERROR, format!("{error:#}")).into_response(),
    }
}

async fn start(State(bench): State<Arc<Bench>>, connect: Option<ConnectInfo<SocketAddr>>, headers: HeaderMap,
    Json(request): Json<RunRequest>) -> Response {
    if !authorized(&bench, peer(connect), &headers) {
        return forbidden();
    }
    match bench.start(request) {
        Ok(id) => (StatusCode::ACCEPTED, Json(json!({"id": id}))).into_response(),
        Err(error @ StartError::Busy(_)) => (StatusCode::CONFLICT, Json(json!({"error": {"message": error.to_string(),
            "type": "busy"}}))).into_response(),
        Err(error) => (StatusCode::BAD_REQUEST, Json(json!({"error": {"message": error.to_string()}}))).into_response(),
    }
}

#[derive(serde::Deserialize)]
struct ProbeRequest {
    body: serde_json::Value,
    #[serde(default)]
    spec: cuteafd_api::openai::probe::ProbeSpec,
    /// Baseline prompt/turn restore checks with one ordinary media fixture.
    #[serde(default)]
    cache_exact: bool,
}

fn probe_error(error: anyhow::Error) -> Response {
    let status = if error.downcast_ref::<StartError>().is_some_and(|e| matches!(e, StartError::Busy(_))) {
        StatusCode::CONFLICT
    } else if let Some(upstream) = error.downcast_ref::<crate::client::UpstreamHttpError>() {
        let status = StatusCode::from_u16(upstream.code).unwrap_or(StatusCode::BAD_GATEWAY);
        if serde_json::from_str::<serde_json::Value>(&upstream.body).is_ok() {
            return (status, [(header::CONTENT_TYPE, "application/json")], upstream.body.clone()).into_response();
        }
        status
    } else {
        StatusCode::BAD_REQUEST
    };
    (status, Json(json!({"error": {"message": format!("{error:#}")}}))).into_response()
}

async fn probe_request(State(bench): State<Arc<Bench>>, connect: Option<ConnectInfo<SocketAddr>>, headers: HeaderMap,
    Json(request): Json<ProbeRequest>) -> Response {
    // A running panel's own media probes carry its run token.
    let own = headers.get(BENCH_HEADER).and_then(|v| v.to_str().ok()).is_some_and(|t| bench.accepts_internal(t));
    if !own && !authorized(&bench, peer(connect), &headers) { return forbidden(); }
    match tokio::task::spawn_blocking(move || bench.probe_fixture(request.body, request.spec, request.cache_exact)).await {
        Ok(Ok(chat)) => Json(chat).into_response(),
        Ok(Err(error)) => probe_error(error),
        Err(error) => (StatusCode::INTERNAL_SERVER_ERROR, error.to_string()).into_response(),
    }
}

async fn cancel(State(bench): State<Arc<Bench>>, connect: Option<ConnectInfo<SocketAddr>>, headers: HeaderMap,
    Path(id): Path<String>) -> Response {
    if !authorized(&bench, peer(connect), &headers) {
        return forbidden();
    }
    if bench.cancel(&id) { Json(json!({"cancelled": id})).into_response() } else { not_found("active run") }
}

async fn import(State(bench): State<Arc<Bench>>, connect: Option<ConnectInfo<SocketAddr>>, headers: HeaderMap,
    Json(report): Json<Report>) -> Response {
    if !authorized(&bench, peer(connect), &headers) {
        return forbidden();
    }
    match bench.import(report) {
        Ok(id) => Json(json!({"id": id})).into_response(),
        Err(error) => (StatusCode::BAD_REQUEST, Json(json!({"error": {"message": format!("{error:#}")}}))).into_response(),
    }
}

fn svg(body: String) -> Response {
    ([(header::CONTENT_TYPE, "image/svg+xml"), (header::CACHE_CONTROL, "no-cache")], body).into_response()
}

fn png(body: Vec<u8>, file: &str) -> Response {
    ([(header::CONTENT_TYPE, "image/png".to_string()),
        (header::CONTENT_DISPOSITION, format!("inline; filename=\"{file}\""))], body).into_response()
}

/// The SVG as PNG bytes, or why it could not be rasterized.
///
/// A failure here is an error naming the file, never a missing export: a caller
/// that writes a run's images can tell "this run places no such export" from
/// "the redraw failed", and must not keep the stale image a failed redraw was
/// replacing.
fn rasterize(svg: &str, file: &str) -> Result<Vec<u8>> {
    render::png::png(svg, 1.0).with_context(|| format!("rasterizing {file}"))
}

/// Every export of one run, by file name.
///
/// The single boundary between a report's data and a rendered file: nothing
/// here trusts the status stored in the report, so an export written under an
/// earlier fidelity rule (or served from a dashboard that stored one) shows the
/// verdict the rule in force now reaches. Everything that writes or serves an
/// image or a report goes through this, which is what keeps the surfaces from
/// disagreeing.
///
/// `Ok(None)` is a name this run has no export for. A renderer that fails is
/// the `Err` arm instead, so no caller can mistake it for an absent file.
pub fn export(report: &Report, file: &str) -> Result<Option<(&'static str, Vec<u8>)>> {
    let report = &crate::quality::under_current_rule(report);
    let Some((stem, extension)) = file.rsplit_once('.') else { return Ok(None) };
    let svg = match stem {
        "report" => render::report::report_svg(report),
        "card" => render::card::card_svg(report),
        _ if stem.starts_with("panel-") => render::report::panel_svg(report, &stem["panel-".len()..]),
        _ => return Ok(None),
    };
    Ok(match extension {
        "svg" => Some(("image/svg+xml", svg.into_bytes())),
        "png" => Some(("image/png", rasterize(&svg, file)?)),
        "json" if stem == "report" => Some(("application/json", serde_json::to_vec_pretty(report)?)),
        _ => None,
    })
}

#[derive(serde::Deserialize, Default)]
struct FileQuery {
    /// Panel exports: the body alone at this width (the dashboard's chart view).
    #[serde(default)]
    bare: Option<f64>,
    /// Optional saved run to compare fidelity against; never starts inference.
    compare: Option<String>,
}

/// `earlier(panel id)`: that panel of the saved run, read on its own (never the whole report).
fn attach_fidelity_comparison(report: &mut Report, earlier: impl Fn(&str) -> Option<crate::report::PanelResult>, earlier_id: &str) {
    for panel in report.panels.iter_mut().filter(|p| matches!(p.id.as_str(), "fidelity" | "fidelity_full")) {
        let earlier_panel = earlier(&panel.id);
        let previous = earlier_panel.as_ref().and_then(|p| p.latest())
            .and_then(|v| serde_json::from_value::<crate::fidelity::Run>(v["decode"].clone()).ok());
        let latest = if let Some(partial) = &mut panel.partial { Some(partial) } else { panel.passes.last_mut() };
        let Some(latest) = latest else { continue };
        let paired = match (&previous, serde_json::from_value::<crate::fidelity::Run>(latest["decode"].clone())) {
            (Some(previous), Ok(current)) => crate::panels::fidelity::record(&current, Some(previous))["paired"].clone(),
            (None, _) => json!({"unavailable": "selected saved run has no matching fidelity decode result"}),
            (_, Err(_)) => json!({"unavailable": "current decode result is not complete"}),
        };
        latest["paired"] = paired;
        latest["paired"]["earlier_run"] = json!(earlier_id);
        if latest["tier"] == crate::fidelity_dataset::STANDARD_TIER && !latest["prefill"].is_null() {
            let previous = earlier_panel.as_ref().and_then(|p| p.latest())
                .and_then(|v| serde_json::from_value::<crate::fidelity::Run>(v["prefill"].clone()).ok());
            latest["paired_prefill"] = match (&previous, serde_json::from_value::<crate::fidelity::Run>(latest["prefill"].clone())) {
                (Some(previous), Ok(current)) => crate::panels::fidelity::record(&current, Some(previous))["paired"].clone(),
                (None, _) => json!({"unavailable": "selected saved run has no matching fidelity prefill result"}),
                (_, Err(_)) => json!({"unavailable": "current prefill result is not complete"}),
            };
            latest["paired_prefill"]["earlier_run"] = json!(earlier_id);
        }
    }
}

async fn run_file(State(bench): State<Arc<Bench>>, Path((id, file)): Path<(String, String)>,
    axum::extract::Query(query): axum::extract::Query<FileQuery>) -> Response {
    let report = if id == "latest" { bench.latest() } else { bench.report(&id) };
    let Some(mut report) = report else { return not_found("run") };
    if let Some(earlier_id) = &query.compare {
        attach_fidelity_comparison(&mut report, |panel| bench.panel(earlier_id, panel), earlier_id);
    }
    // The two surfaces below render the report itself rather than one of its
    // exports; `export` applies the same rule to everything else.
    report = crate::quality::under_current_rule(&report);
    if let (Some(width), Some(panel)) = (query.bare, file.strip_prefix("panel-").and_then(|f| f.strip_suffix(".svg"))) {
        return svg(render::report::panel_body_svg(&report, panel, width.clamp(320.0, 2400.0)));
    }
    if file == "report.json" {
        return ([(header::CONTENT_TYPE, "application/json")], serde_json::to_string_pretty(&report)
            .unwrap_or_default()).into_response();
    }
    match export(&report, &file) {
        Ok(Some((_, bytes))) if file.ends_with(".svg") => svg(String::from_utf8_lossy(&bytes).into_owned()),
        Ok(Some((_, bytes))) if file.ends_with(".png") => png(bytes, &file),
        Ok(Some((kind, bytes))) => ([(header::CONTENT_TYPE, kind)], bytes).into_response(),
        Ok(None) => not_found("export"),
        Err(error) => (StatusCode::INTERNAL_SERVER_ERROR, format!("{error:#}")).into_response(),
    }
}

/// One run's measured data, read the way every surface reads it.
///
/// The dashboard renders its verdict (scary mode, the failing checks, the
/// quality line) from this response, so it goes through the same rule as the
/// exports: a run stored before a checkpoint was reclassified does not keep
/// presenting the verdict the data no longer earns.
async fn run_report(State(bench): State<Arc<Bench>>, Path(id): Path<String>) -> Response {
    let report = if id == "latest" { bench.latest() } else { bench.report(&id) };
    match report {
        Some(report) => Json(crate::quality::under_current_rule(&report)).into_response(),
        None => not_found("run"),
    }
}

/// Every panel's newest result on this server fingerprint (see `Bench::merged`).
async fn merged(State(bench): State<Arc<Bench>>, Path(fingerprint): Path<String>) -> Response {
    match tokio::task::spawn_blocking(move || bench.merged(&fingerprint)).await.ok().flatten() {
        Some((report, baseline)) => {
            let mut value = serde_json::to_value(crate::quality::under_current_rule(&report)).unwrap_or_default();
            value["merged"] = json!(true);
            if let Some((run, created)) = baseline { value["baseline_run"] = json!(run); value["baseline_created"] = json!(created); }
            Json(value).into_response()
        }
        None => not_found("runs for this fingerprint"),
    }
}

async fn events(State(bench): State<Arc<Bench>>) -> Response {
    let mut receiver = bench.subscribe();
    let first = json!({"type": "status", "status": bench.status()}).to_string();
    let stream = async_stream::stream! {
        yield Ok::<_, std::convert::Infallible>(format!("data: {first}\n\n"));
        loop {
            match tokio::time::timeout(std::time::Duration::from_secs(15), receiver.recv()).await {
                Ok(Ok(event)) => yield Ok(format!("data: {event}\n\n")),
                Ok(Err(tokio::sync::broadcast::error::RecvError::Lagged(_))) => continue,
                Ok(Err(_)) => break,
                Err(_) => yield Ok(": keepalive\n\n".to_string()),
            }
        }
    };
    ([(header::CONTENT_TYPE, "text/event-stream"), (header::CACHE_CONTROL, "no-cache")], Body::from_stream(stream))
        .into_response()
}

/// Whether the usage history keeps benchmark requests (default: skip them).
async fn usage(State(bench): State<Arc<Bench>>) -> Response {
    match bench.usage.get() {
        Some(toggle) => Json(json!({"record_bench": (toggle.get)(), "available": true})).into_response(),
        None => Json(json!({"record_bench": false, "available": false})).into_response(),
    }
}

#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct UsageSwitch {
    record_bench: bool,
}

async fn set_usage(State(bench): State<Arc<Bench>>, body: Result<Json<UsageSwitch>, axum::extract::rejection::JsonRejection>) -> Response {
    let Ok(Json(switch)) = body else {
        return (StatusCode::BAD_REQUEST, Json(json!({"error": {"message": "expected {\"record_bench\": bool}", "type": "invalid_request_error"}}))).into_response();
    };
    let Some(toggle) = bench.usage.get().cloned() else { return not_found("usage history") };
    match tokio::task::spawn_blocking(move || (toggle.set)(switch.record_bench).map(|()| (toggle.get)())).await {
        Ok(Ok(record)) => Json(json!({"record_bench": record, "available": true})).into_response(),
        Ok(Err(message)) => (StatusCode::BAD_REQUEST, Json(json!({"error": {"message": message, "type": "invalid_request_error"}}))).into_response(),
        Err(_) => (StatusCode::INTERNAL_SERVER_ERROR, Json(json!({"error": {"message": "usage settings unavailable", "type": "server_error"}}))).into_response(),
    }
}

async fn require_control_key(State(bench): State<Arc<Bench>>, request: Request, next: Next) -> Response {
    // Status stays public: the BENCHMARKING banner on every page polls it.
    let path = request.uri().path();
    let public = request.method() == Method::GET && path == "/v1/bench/status";
    let own_probe = path == "/v1/bench/probe" && request.headers().get(BENCH_HEADER)
        .and_then(|v| v.to_str().ok()).is_some_and(|t| bench.accepts_internal(t));
    if path.starts_with("/v1/bench/") && !public && !own_probe && !authorized(&bench, None, request.headers()) {
        return forbidden();
    }
    next.run(request).await
}
/// The benchmark routes.
pub fn routes(bench: Arc<Bench>) -> Router {
    Router::new()
        .route("/bench", get(page))
        .route("/bench/banner.js", get(banner))
        .route("/bench/fabric.js", get(fabric_script))
        .route("/v1/bench/status", get(status))
        .route("/v1/bench/probe", post(probe_request))
        .route("/v1/bench/panels", get(panels))
        .route("/v1/bench/profiles", get(profiles))
        .route("/v1/bench/profiles/:name", put(save_profile).delete(delete_profile))
        .route("/v1/bench/runs", get(runs).post(start))
        .route("/v1/bench/runs/:id", get(run_report))
        .route("/v1/bench/merged/:fingerprint", get(merged))
        .route("/v1/bench/runs/:id/cancel", post(cancel))
        .route("/v1/bench/runs/:id/:file", get(run_file))
        .route("/v1/bench/import", post(import))
        .route("/v1/bench/events", get(events))
        .route("/v1/bench/usage", get(usage).put(set_usage))
        .layer(axum::extract::DefaultBodyLimit::max(64 << 20))
        .with_state(bench.clone())
        .layer(axum::middleware::from_fn_with_state(bench, require_control_key))
}

/// `router` with the benchmark mounted and the lockout in front of it.
pub fn mount(router: Router, bench: Arc<Bench>) -> Router {
    router.merge(routes(bench.clone())).layer(axum::middleware::from_fn_with_state(bench, lockout))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Keyless bench server: inference open, controls need the console cookie.
    /// With a key configured, the controls take either credential; status stays public.
    #[tokio::test]
    async fn bench_controls_take_the_console_cookie_or_a_configured_key() {
        use tower::ServiceExt;
        let secret = tempfile::NamedTempFile::new().unwrap();
        std::fs::write(secret.path(), "c".repeat(64)).unwrap();
        let gate = cuteafd_api::console_gate::ConsoleGate::from_file(secret.path(), false).unwrap();
        for key in [None, Some("op-key")] {
            let bench = Bench::new(crate::store::Store::memory().unwrap());
            bench.set_console_gate(gate.clone());
            let key = key.map(|k| cuteafd_api::openai::auth::ApiKey::new(k).unwrap());
            if let Some(key) = &key { bench.set_api_key(key.clone()); }
            let inner = Router::new().route("/v1/chat/completions", post(|| async { "chat" }));
            let app = gate.mount(mount(inner, bench)).layer(axum::middleware::from_fn_with_state(
                cuteafd_api::openai::auth::Auth { key: key.clone(), internal: None }, cuteafd_api::openai::auth::require_key));
            let unlock = app.clone().oneshot(Request::get(format!("/console/unlock?token={}", "c".repeat(64))).body(Body::empty()).unwrap()).await.unwrap();
            let cookie = unlock.headers()[header::SET_COOKIE].to_str().unwrap().split(';').next().unwrap().to_string();
            let call = |method: Method, path: &str, credential: Option<(&'static str, String)>| {
                let mut request = Request::builder().method(method).uri(path).header("content-type", "application/json");
                if let Some((name, value)) = credential { request = request.header(name, value); }
                let request = request.body(Body::from("{}")).unwrap();
                let app = app.clone();
                async move { app.oneshot(request).await.unwrap().status() }
            };
            let chat = call(Method::POST, "/v1/chat/completions", None).await;
            assert_eq!(chat == StatusCode::OK, key.is_none(), "keyless inference is open; a configured key guards it");
            assert_eq!(call(Method::GET, "/v1/bench/status", None).await, StatusCode::OK);
            for (method, path) in [(Method::GET, "/v1/bench/runs"), (Method::POST, "/v1/bench/probe"), (Method::PUT, "/v1/bench/usage")] {
                assert_eq!(call(method.clone(), path, None).await, StatusCode::FORBIDDEN, "{path} without credentials");
                assert_ne!(call(method.clone(), path, Some(("cookie", cookie.clone()))).await, StatusCode::FORBIDDEN, "{path} with the cookie");
                let bearer = call(method.clone(), path, Some(("authorization", "Bearer op-key".into()))).await;
                assert_eq!(bearer != StatusCode::FORBIDDEN, key.is_some(), "{path} with a bearer");
            }
        }
    }

    /// A stored Qwen NVFP4 max run whose own `config.json` declares `algo` for
    /// its routed experts, with `failing` (a check id, or `""`) left failing
    /// beside the fidelity check: the failure the export was stored with.
    fn stored_run(algo: &str, failing: &str) -> (tempfile::TempDir, Arc<Bench>, Report) {
        let dir = tempfile::tempdir().unwrap();
        let report = crate::sample::qwen_nvfp4_max(dir.path(), algo, failing);
        (dir, Bench::new(crate::store::Store::memory().unwrap()), report)
    }

    /// `/v1/bench/runs/<id>`, the response the dashboard renders a run's verdict
    /// from, for a run imported exactly as it was stored.
    async fn served_run(algo: &str, failing: &str) -> Report {
        use tower::ServiceExt;
        let (_hub, bench, report) = stored_run(algo, failing);
        bench.set_api_key(cuteafd_api::openai::auth::ApiKey::new("k").unwrap());
        let id = bench.import(report).unwrap();
        let request = axum::http::Request::get(format!("/v1/bench/runs/{id}"))
            .header("Authorization", "bearer k").body(Body::empty()).unwrap();
        let response = routes(bench).oneshot(request).await.unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let bytes = axum::body::to_bytes(response.into_body(), 1 << 20).await.unwrap();
        serde_json::from_slice(&bytes).unwrap()
    }

    /// The dashboard renders a run's verdict (scary mode, the failing checks,
    /// the quality line) from `/v1/bench/runs/<id>`, so a run measured before
    /// its checkpoint was reclassified must read informational there too — while
    /// the same numbers from a checkpoint the bar still gates keep failing.
    #[tokio::test]
    async fn the_dashboard_reads_a_stored_run_under_the_rule_in_force_now() {
        use crate::report::CheckStatus;
        for (algo, expected) in [("NVFP4", CheckStatus::Info), ("W4A16_NVFP4", CheckStatus::Fail)] {
            let served = served_run(algo, "").await;
            let check = served.baseline.as_ref().unwrap().quality.checks.iter()
                .find(|c| c.id == "fidelity").unwrap();
            assert_eq!(check.status, expected, "{algo}");
            assert_eq!(check.note.as_deref(), (expected == CheckStatus::Info)
                .then_some(crate::quality::CALIBRATED_W4A4), "{algo}");
        }
    }

    /// A gate still failing beside the reclassified fidelity check reaches the
    /// run view as a failure: the informational check is information, not a
    /// verdict, and the failing one keeps its name.
    #[tokio::test]
    async fn the_run_view_keeps_a_failing_gate_beside_a_reclassified_fidelity_check() {
        use crate::report::CheckStatus;
        for (failing, title) in [("cache_exact", "Prefix-cache restore"), ("spec_lossless", "Speculation lossless")] {
            let served = served_run("NVFP4", failing).await;
            let baseline = served.baseline.as_ref().unwrap();
            assert_eq!(baseline.quality.status, CheckStatus::Fail, "{failing}");
            assert!(served.quality_failed(), "{failing}");
            let check = |id: &str| baseline.quality.checks.iter().find(|check| check.id == id).unwrap();
            assert_eq!(check("fidelity").status, CheckStatus::Info, "{failing}");
            assert_eq!(check(failing).status, CheckStatus::Fail, "{failing}");
            let line = crate::render::bodies::quality_line(baseline);
            assert_eq!(line, format!("failed: {title}"), "{failing}");
            assert!(baseline.quality.badge().contains("top-1 93.7% (info: calibrated W4A4)"), "{failing}");
        }
    }

    /// A rasterization failure is an error naming the file, never an absent
    /// export: a publisher re-rendering a run's images must be able to tell a
    /// failure from a file the run does not have, or it keeps a stale image.
    #[test]
    fn media_cache_probe_accepts_fixture_without_manual_probe_switches() {
        let request: ProbeRequest = serde_json::from_value(json!({"cache_exact": true,
            "body": {"messages": [{"role": "user", "content": [{"type": "input_audio",
                "input_audio": {"data": "AA==", "format": "wav"}}]}]}})).unwrap();
        assert!(request.cache_exact);
        assert!(request.spec.prompt_ids.is_none());
        assert!(!request.spec.cold);
        assert_eq!(crate::baseline::cache_fixture(&request.body).unwrap()["type"], "input_audio");
        let legacy: ProbeRequest = serde_json::from_value(json!({"body": {}, "spec": {"record_rows": 2}})).unwrap();
        assert!(!legacy.cache_exact);
        assert_eq!(legacy.spec.record_rows, 2);
    }

    #[test]
    fn a_failed_rasterization_is_an_error_not_a_missing_export() {
        let error = rasterize("<svg", "card.png").unwrap_err();
        assert!(format!("{error:#}").contains("rasterizing card.png"), "{error:#}");
        let report = crate::sample::report(false);
        // The same call for an export the run has: the arm a stale PNG hid in.
        let (kind, bytes) = export(&report, "card.png").unwrap().expect("card.png renders");
        assert_eq!(kind, "image/png");
        assert_eq!(&bytes[..4], b"\x89PNG", "the rasterized card is a PNG");
        // A name this run has no export for is the absent arm, not an error.
        assert!(export(&report, "notes.txt").unwrap().is_none());
        assert!(export(&report, "thumbnail.svg").unwrap().is_none());
    }

    #[tokio::test]
    async fn usage_switch_requires_the_key_and_round_trips() {
        use tower::ServiceExt;
        let bench = Bench::new(crate::store::Store::memory().unwrap());
        bench.set_api_key(cuteafd_api::openai::auth::ApiKey::new("k").unwrap());
        let value = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let (get_v, set_v) = (value.clone(), value.clone());
        bench.set_usage(crate::UsageToggle {
            get: Arc::new(move || get_v.load(std::sync::atomic::Ordering::Relaxed)),
            set: Arc::new(move |v| { set_v.store(v, std::sync::atomic::Ordering::Relaxed); Ok(()) }),
        });
        let app = routes(bench);
        let call = |method: &str, key: Option<&str>, body: &str| {
            let mut r = axum::http::Request::builder().method(method).uri("/v1/bench/usage").header("content-type", "application/json");
            if let Some(k) = key { r = r.header("Authorization", format!("Bearer {k}")); }
            app.clone().oneshot(r.body(Body::from(body.to_owned())).unwrap())
        };
        assert_eq!(call("PUT", None, r#"{"record_bench":true}"#).await.unwrap().status(), StatusCode::FORBIDDEN);
        let r = call("GET", Some("k"), "").await.unwrap();
        let v: serde_json::Value = serde_json::from_slice(&axum::body::to_bytes(r.into_body(), 1024).await.unwrap()).unwrap();
        assert_eq!(v, json!({"record_bench": false, "available": true}));
        let r = call("PUT", Some("k"), r#"{"record_bench":true}"#).await.unwrap();
        assert_eq!(r.status(), StatusCode::OK);
        assert!(value.load(std::sync::atomic::Ordering::Relaxed));
        assert_eq!(call("PUT", Some("k"), r#"{"record_bench":1}"#).await.unwrap().status(), StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn probe_preserves_typed_upstream_status_and_message() {
        for code in [400, 429, 503] {
            let error = anyhow::Error::new(crate::client::UpstreamHttpError {
                code, body: "vision encoder unavailable".into(),
            }).context("probe chat");
            let response = probe_error(error);
            assert_eq!(response.status().as_u16(), code);
            let bytes = axum::body::to_bytes(response.into_body(), 4096).await.unwrap();
            let body: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
            assert!(body["error"]["message"].as_str().unwrap().contains("vision encoder unavailable"));
        }
        assert_eq!(probe_error(StartError::Busy("active".into()).into()).status(), StatusCode::CONFLICT);
        assert_eq!(probe_error(anyhow::anyhow!("invalid probe")).status(), StatusCode::BAD_REQUEST);
        // A string resembling an HTTP error is not an upstream status contract.
        assert_eq!(probe_error(anyhow::anyhow!("HTTP 503: invalid input")).status(), StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn probe_passes_upstream_json_body_through_verbatim() {
        let body = format!("{{ \"error\": {{\"message\":\"vision encoder unavailable\",\"type\":\"native_v41_error\"}}, \"detail\":\"{}\" }}", "x".repeat(512));
        let error = anyhow::Error::new(crate::client::UpstreamHttpError { code: 503, body: body.clone() })
            .context("probe chat");
        let response = probe_error(error);
        assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
        assert_eq!(response.headers()[header::CONTENT_TYPE], "application/json");
        let bytes = axum::body::to_bytes(response.into_body(), 4096).await.unwrap();
        assert_eq!(bytes.as_ref(), body.as_bytes());
    }

    #[test]
    fn lockout_covers_inference_posts_and_new_gateway_sockets() {
        let mut upgrade = HeaderMap::new();
        upgrade.insert(header::UPGRADE, HeaderValue::from_static("websocket"));
        let none = HeaderMap::new();
        for path in ["/v1/chat/completions", "/v1/messages", "/v1/responses"] {
            assert!(inference(&Method::POST, &none, path), "{path}");
        }
        for path in ["/v1/responses", "/v1/realtime"] {
            assert!(inference(&Method::GET, &upgrade, path), "new {path} sockets wait for the run");
            assert!(!inference(&Method::GET, &none, path), "plain GET {path} is not inference");
        }
        assert!(!inference(&Method::GET, &upgrade, "/v1/console"), "the console socket is not inference");
    }

    #[tokio::test]
    async fn probe_control_requires_authorization_before_execution() {
        use tower::ServiceExt;
        let bench = Bench::new(crate::store::Store::memory().unwrap());
        let request = axum::http::Request::builder().method("POST").uri("/v1/bench/probe")
            .header("content-type", "application/json")
            .body(Body::from(r#"{"body":{"messages":[]},"spec":{}}"#)).unwrap();
        let response = routes(bench.clone()).oneshot(request).await.unwrap();
        assert_eq!(response.status(), StatusCode::FORBIDDEN);
        assert!(bench.active().is_none());
    }

    #[tokio::test]
    async fn private_network_does_not_bypass_key_and_auth_precedes_json() {
        use tower::ServiceExt;
        let bench = Bench::new(crate::store::Store::memory().unwrap());
        bench.set_api_key(cuteafd_api::openai::auth::ApiKey::new("bench-secret").unwrap());
        for key in [None, Some("Bearer wrong")] {
            let mut request = axum::http::Request::builder().method("POST").uri("/v1/bench/probe")
                .header("content-type", "application/json")
                .extension(ConnectInfo("10.55.0.1:8000".parse::<SocketAddr>().unwrap()));
            if let Some(key) = key { request = request.header("Authorization", key); }
            assert_eq!(routes(bench.clone()).oneshot(request.body(Body::from("malformed")).unwrap())
                .await.unwrap().status(), StatusCode::FORBIDDEN);
        }
        let response = routes(bench).oneshot(axum::http::Request::get("/v1/bench/status")
            .header("Authorization", "bearer bench-secret").body(Body::empty()).unwrap()).await.unwrap();
        assert_eq!(response.status(), StatusCode::OK);
    }

    #[test]
    fn missing_saved_fidelity_pair_is_explicit_and_updates_partial_only() {
        let mut report = Report { schema: crate::report::SCHEMA.into(), prompt_content: Some(crate::text::PROMPT_CONTENT.into()), id: "current".into(),
            created: String::new(), finished: None, status: crate::report::RunStatus::Running,
            profile: "fidelity".into(), plan: vec![], server: Default::default(), fingerprint: String::new(),
            baseline: None, panels: vec![crate::report::PanelResult {
                id: "fidelity".into(), passes: vec![json!({"saved":true})], partial: Some(json!({"decode":null})),
                ..Default::default() }], fabric: vec![], error: None };
        attach_fidelity_comparison(&mut report, |_| None, "missing");
        let panel = report.panel("fidelity").unwrap();
        assert_eq!(panel.latest().unwrap()["paired"]["earlier_run"], "missing");
        assert!(panel.latest().unwrap()["paired"]["unavailable"].as_str().unwrap().contains("no matching"));
        assert_eq!(panel.passes, vec![json!({"saved":true})]);
    }

    #[test]
    fn standard_history_checks_both_paths_and_rejects_legacy_version() {
        let mut current = crate::sample::full_report();
        current.panels.retain(|p| p.id == "fidelity");
        let mut earlier = current.clone();
        attach_fidelity_comparison(&mut current,|id| earlier.panel(id).cloned(),"earlier");
        let latest = current.panel("fidelity").unwrap().latest().unwrap();
        assert_eq!(latest["paired"]["comparison"]["pass"],true);
        assert_eq!(latest["paired_prefill"]["comparison"]["pass"],true);
        let old = earlier.panels[0].passes.last_mut().unwrap();
        old["decode"]["tier"] = json!("standard"); old["prefill"]["tier"] = json!("standard");
        attach_fidelity_comparison(&mut current,|id| earlier.panel(id).cloned(),"legacy");
        let latest = current.panel("fidelity").unwrap().latest().unwrap();
        assert!(latest["paired"]["unavailable"].as_str().unwrap().contains("different tiers"));
        assert!(latest["paired_prefill"]["unavailable"].as_str().unwrap().contains("different tiers"));
    }

    #[test]
    fn local_networks() {
        for ip in ["127.0.0.1", "10.55.0.3", "192.168.1.9", "172.20.0.1", "::1", "fd00::1", "fe80::1",
            "::ffff:10.0.0.1", "100.100.1.1"] {
            assert!(local_network(ip.parse().unwrap()), "{ip}");
        }
        for ip in ["8.8.8.8", "2001:4860::8888", "172.32.0.1"] {
            assert!(!local_network(ip.parse().unwrap()), "{ip}");
        }
    }
}
