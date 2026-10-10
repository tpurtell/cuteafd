//! All usage data and mutations require the console cookie, never an API key.
use crate::{
    query::Filter,
    store::{Error, Settings},
    Store,
};
use axum::{
    extract::{Path, Query, State},
    http::StatusCode,
    response::{IntoResponse, Response},
    routing::{get, post},
    Json, Router,
};
use cuteafd_api::console_gate::{require_console, ConsoleGate};
use serde_json::{json, Value};
use std::{
    collections::HashMap,
    sync::{Arc, Mutex},
    time::{Duration, Instant},
};
#[derive(Clone)]
struct Http {
    store: Arc<Store>,
    cache: Arc<Mutex<HashMap<String, (Instant, Value)>>>,
}
/// The usage dashboard, compiled in (`/usage`; the page itself is public, its data needs the cookie).
pub const PAGE: &str = include_str!("../assets/usage.html");

/// Serve the compiled-in page, or for page development the file named by `CUTEAFD_USAGE_PAGE`.
async fn page() -> Response {
    let page = match std::env::var_os("CUTEAFD_USAGE_PAGE") {
        Some(path) => tokio::fs::read_to_string(&path).await.unwrap_or_else(|_| PAGE.to_string()),
        None => PAGE.to_string(),
    };
    ([(axum::http::header::CACHE_CONTROL, "no-cache")], axum::response::Html(page)).into_response()
}

pub fn mount(router: Router, store: Arc<Store>, gate: ConsoleGate) -> Router {
    let state = Http {
        store,
        cache: Arc::new(Mutex::new(HashMap::new())),
    };
    let routes = Router::new()
        .route("/console/usage/summary", get(summary))
        .route("/console/usage/series", get(series))
        .route("/console/usage/latency", get(latency))
        .route("/console/usage/flow", get(flow))
        .route("/console/usage/sessions", get(sessions))
        .route("/console/usage/sessions/:id", get(session))
        .route("/console/usage/cache", get(cache))
        .route("/console/usage/speculation", get(speculation))
        .route("/console/usage/errors", get(errors))
        .route("/console/usage/requests", get(requests))
        .route("/console/usage/requests/:rid", get(request))
        .route("/console/usage/log", get(log_sessions))
        .route("/console/usage/log/sessions/:vsid", get(log_session))
        .route("/console/usage/log/:rid", get(log))
        .route("/console/usage/media/:sha256", get(media))
        .route(
            "/console/usage/settings",
            get(settings).put(update_settings),
        )
        .route("/console/usage/log/clear", post(clear_log))
        .route("/console/usage/clear", post(clear))
        .with_state(state)
        .layer(axum::middleware::from_fn_with_state(gate, require_console));
    router.merge(routes).route("/usage", get(page))
}
fn error(e: Error) -> Response {
    let status = if matches!(e, Error::Settings(_)) {
        StatusCode::BAD_REQUEST
    } else {
        tracing::error!(error=%e,"usage query failed");
        StatusCode::INTERNAL_SERVER_ERROR
    };
    (status,Json(json!({"error":{"type":"usage_error","message":if status==StatusCode::BAD_REQUEST{e.to_string()}else{"usage storage unavailable".into()}}}))).into_response()
}
async fn query(state: Http, f: Filter, kind: &'static str, id: Option<String>) -> Response {
    let key = format!(
        "{kind}|{}|{}",
        id.as_deref().unwrap_or(""),
        serde_json::to_string(&f).expect("filter JSON")
    );
    if let Some((_, value)) = state
        .cache
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .get(&key)
        .filter(|(at, _)| at.elapsed() < Duration::from_secs(5))
        .cloned()
    {
        return Json(value).into_response();
    }
    let store = state.store.clone();
    match tokio::task::spawn_blocking(move || store.query(kind, &f, id.as_deref())).await {
        Ok(Ok(v)) => {
            if v.is_null() {
                return (
                    StatusCode::NOT_FOUND,
                    Json(json!({"error":{"type":"not_retained"}})),
                )
                    .into_response();
            }
            let mut cache = state.cache.lock().unwrap_or_else(|e| e.into_inner());
            cache.retain(|_, (at, _)| at.elapsed() < Duration::from_secs(5));
            if cache.len() >= 128 {
                cache.clear();
            }
            cache.insert(key, (Instant::now(), v.clone()));
            Json(v).into_response()
        }
        Ok(Err(e)) => error(e),
        Err(_) => error(Error::Stopped),
    }
}
macro_rules! list_route {
    ($name:ident) => {
        async fn $name(State(s): State<Http>, Query(f): Query<Filter>) -> Response {
            query(s, f, stringify!($name), None).await
        }
    };
}
list_route!(summary);
list_route!(series);
list_route!(latency);
list_route!(flow);
list_route!(sessions);
list_route!(cache);
list_route!(speculation);
list_route!(errors);
list_route!(requests);
async fn session(
    State(s): State<Http>,
    Path(id): Path<String>,
    Query(f): Query<Filter>,
) -> Response {
    query(s, f, "sessions", Some(id)).await
}
async fn request(
    State(s): State<Http>,
    Path(id): Path<String>,
    Query(f): Query<Filter>,
) -> Response {
    query(s, f, "requests", Some(id)).await
}
async fn log(State(s): State<Http>, Path(id): Path<String>) -> Response {
    query(s, Filter::default(), "log", Some(id)).await
}
async fn log_sessions(State(s): State<Http>, Query(f): Query<Filter>) -> Response {
    let store = s.store.clone();
    blocking(move || store.log.sessions(&f).map(Some)).await
}
async fn log_session(State(s): State<Http>, Path(vsid): Path<String>) -> Response {
    let store = s.store.clone();
    blocking(move || {
        let Some(mut session) = store.log.session(&vsid)? else { return Ok(None) };
        let rids = session["entries"].as_array().into_iter().flatten()
            .filter_map(|e| e["rid"].as_str().map(str::to_owned)).collect::<Vec<_>>();
        session["perf"] = store.perf_for(&rids)?;
        Ok(Some(session))
    })
    .await
}
/// Runs a full-log read off the async workers; `None` is 404 not_retained.
async fn blocking(f: impl FnOnce() -> Result<Option<Value>, Error> + Send + 'static) -> Response {
    match tokio::task::spawn_blocking(f).await {
        Ok(Ok(Some(v))) => Json(v).into_response(),
        Ok(Ok(None)) => (StatusCode::NOT_FOUND, Json(json!({"error":{"type":"not_retained"}}))).into_response(),
        Ok(Err(e)) => error(e),
        Err(_) => error(Error::Stopped),
    }
}
async fn media(State(s): State<Http>, Path(sha256): Path<String>) -> Response {
    let store = s.store.clone();
    let found = tokio::task::spawn_blocking(move || store.log.media(&sha256)).await;
    let (mime, path) = match found {
        Ok(Ok(Some(found))) => found,
        Ok(Ok(None)) => return (StatusCode::NOT_FOUND, Json(json!({"error":{"type":"not_retained"}}))).into_response(),
        Ok(Err(e)) => return error(e),
        Err(_) => return error(Error::Stopped),
    };
    match tokio::fs::read(&path).await {
        Ok(bytes) => {
            // Stored media render inline only as images or audio; anything else downloads.
            let inline = mime.starts_with("image/") && mime != "image/svg+xml" || mime.starts_with("audio/");
            let content_type = if inline { mime.as_str() } else { "application/octet-stream" };
            let mut response = (
                [
                    (axum::http::header::CONTENT_TYPE, content_type.to_owned()),
                    (axum::http::header::HeaderName::from_static("x-content-type-options"), "nosniff".to_owned()),
                    (axum::http::header::CONTENT_SECURITY_POLICY, "default-src 'none'; sandbox".to_owned()),
                ],
                bytes,
            )
                .into_response();
            if !inline {
                response.headers_mut().insert(axum::http::header::CONTENT_DISPOSITION, axum::http::HeaderValue::from_static("attachment"));
            }
            response
        }
        Err(_) => (StatusCode::NOT_FOUND, Json(json!({"error":{"type":"not_retained"}}))).into_response(),
    }
}
fn settings_body(s: &Http) -> Value {
    json!({"settings":s.store.settings(),"usage":s.store.counters.snapshot(),"warning":"With the full log on, user prompts and model outputs (and, with media on, images and audio) are stored in plain text on this host for the retention period."})
}
async fn settings(State(s): State<Http>) -> Json<Value> {
    Json(settings_body(&s))
}
/// PUT takes the same object GET returns under `settings` (a `{"settings":…}` wrapper is accepted).
async fn update_settings(State(s): State<Http>, Json(body): Json<Value>) -> Response {
    let body = match body.get("settings") {
        Some(inner) if body.as_object().is_some_and(|o| o.len() <= 3) && inner.is_object() => inner.clone(),
        _ => body,
    };
    let settings: Settings = match serde_json::from_value(body) {
        Ok(settings) => settings,
        Err(e) => {
            return (StatusCode::BAD_REQUEST, Json(json!({"error":{"type":"usage_error","message":e.to_string()}}))).into_response()
        }
    };
    let store = s.store.clone();
    match tokio::task::spawn_blocking(move || store.update_settings(settings)).await {
        Ok(Ok(())) => {
            s.cache.lock().unwrap_or_else(|e| e.into_inner()).clear();
            Json(settings_body(&s)).into_response()
        }
        Ok(Err(e)) => error(e),
        Err(_) => error(Error::Stopped),
    }
}
async fn clear_log(State(s): State<Http>) -> Response {
    clear_impl(s, false).await
}
async fn clear(State(s): State<Http>) -> Response {
    clear_impl(s, true).await
}
async fn clear_impl(s: Http, all: bool) -> Response {
    let store = s.store.clone();
    match tokio::task::spawn_blocking(move || {
        store.log.clear()?;
        if all {
            store.clear()?;
        }
        Ok::<_, Error>(())
    })
    .await
    {
        Ok(Ok(())) => {
            s.cache.lock().unwrap_or_else(|e| e.into_inner()).clear();
            Json(json!({"cleared":true})).into_response()
        }
        Ok(Err(e)) => error(e),
        Err(_) => error(Error::Stopped),
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn page_is_self_contained() {
        assert!(PAGE.contains("/assets/cuteafd-ui.css") && PAGE.contains("/assets/cuteafd-ui.js"));
        assert!(PAGE.contains("/console/usage/"));
        for external in ["http://", "https://"] {
            assert!(!PAGE.contains(&format!("src=\"{external}")), "page loads an external script");
            assert!(!PAGE.contains(&format!("href=\"{external}")), "page loads an external stylesheet");
            assert!(!PAGE.contains(&format!("url({external}")), "page loads an external resource");
            assert!(!PAGE.contains(&format!("@import \"{external}")) && !PAGE.contains(&format!("fetch('{external}")));
        }
    }
    #[tokio::test]
    async fn usage_page_is_public_and_settings_round_trip() {
        let f = tempfile::NamedTempFile::new().unwrap();
        std::fs::write(f.path(), "c".repeat(64)).unwrap();
        let gate = ConsoleGate::from_file(f.path(), false).unwrap();
        let store = Store::open(None).unwrap();
        let app = gate.mount(mount(Router::new(), store, gate.clone()));
        let page = app.clone().oneshot(Request::get("/usage").body(Body::empty()).unwrap()).await.unwrap();
        assert_eq!(page.status(), StatusCode::OK);
        let r = app.clone().oneshot(Request::get(format!("/console/unlock?token={}", "c".repeat(64))).body(Body::empty()).unwrap()).await.unwrap();
        let cookie = r.headers()["set-cookie"].to_str().unwrap().split(';').next().unwrap().to_owned();
        let call = |method: &str, body: String| {
            app.clone().oneshot(Request::builder().method(method).uri("/console/usage/settings").header("cookie", &cookie)
                .header("content-type", "application/json").body(Body::from(body)).unwrap())
        };
        let read = |r: Response| async { serde_json::from_slice::<Value>(&axum::body::to_bytes(r.into_body(), 65536).await.unwrap()).unwrap() };
        let got = read(call("GET", String::new()).await.unwrap()).await;
        assert_eq!(got["settings"]["log_enabled"], true);
        assert_eq!(got["settings"]["log_media"], true);
        assert_eq!(got["settings"]["record_bench"], false);
        let mut settings = got["settings"].clone();
        settings["log_hours"] = json!(6);
        let put = read(call("PUT", settings.to_string()).await.unwrap()).await;
        assert_eq!(put["settings"], settings, "PUT returns what GET returns");
        assert_eq!(read(call("GET", String::new()).await.unwrap()).await["settings"], settings);
        // The GET envelope is accepted back as-is.
        let mut envelope = put.clone();
        envelope["settings"]["log_hours"] = json!(12);
        assert_eq!(read(call("PUT", envelope.to_string()).await.unwrap()).await["settings"]["log_hours"], 12);
        let bad = call("PUT", r#"{"log_hours":"x"}"#.into()).await.unwrap();
        assert_eq!(bad.status(), StatusCode::BAD_REQUEST);
    }
    use axum::{body::Body, http::Request};
    use tower::ServiceExt;
    #[tokio::test]
    async fn every_route_requires_cookie_before_extraction() {
        let app = mount(
            Router::new(),
            Store::open(None).unwrap(),
            ConsoleGate::locked(),
        );
        let media = format!("media/{}", "a".repeat(64));
        for (method, path) in [
            ("GET", "summary"),
            ("GET", "series"),
            ("GET", "latency"),
            ("GET", "flow"),
            ("GET", "sessions"),
            ("GET", "sessions/id"),
            ("GET", "cache"),
            ("GET", "speculation"),
            ("GET", "errors"),
            ("GET", "requests"),
            ("GET", "requests/rid"),
            ("GET", "log/rid"),
            ("GET", "log"),
            ("GET", "log/sessions/vsid"),
            ("GET", media.as_str()),
            ("GET", "settings"),
            ("PUT", "settings"),
            ("POST", "log/clear"),
            ("POST", "clear"),
        ] {
            let r = app
                .clone()
                .oneshot(
                    Request::builder()
                        .method(method)
                        .uri(format!("/console/usage/{path}"))
                        .header("Authorization", "Bearer API_KEY")
                        .body(Body::from("malformed"))
                        .unwrap(),
                )
                .await
                .unwrap();
            assert_eq!(r.status(), StatusCode::UNAUTHORIZED, "{method} {path}");
            assert_eq!(
                serde_json::from_slice::<Value>(
                    &axum::body::to_bytes(r.into_body(), 4096).await.unwrap()
                )
                .unwrap()["error"]["type"],
                "console_locked"
            );
        }
    }
    #[tokio::test]
    async fn unlocked_settings_and_clear_invalidate_cache() {
        let f = tempfile::NamedTempFile::new().unwrap();
        std::fs::write(f.path(), "a".repeat(64)).unwrap();
        let gate = ConsoleGate::from_file(f.path(), false).unwrap();
        let store = Store::open(None).unwrap();
        let app = gate.mount(mount(Router::new(), store, gate.clone()));
        let r = app
            .clone()
            .oneshot(
                Request::get(format!("/console/unlock?token={}", "a".repeat(64)))
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        let cookie = r.headers()["set-cookie"]
            .to_str()
            .unwrap()
            .split(';')
            .next()
            .unwrap();
        for path in ["settings", "summary"] {
            assert_eq!(
                app.clone()
                    .oneshot(
                        Request::get(format!("/console/usage/{path}"))
                            .header("cookie", cookie)
                            .body(Body::empty())
                            .unwrap()
                    )
                    .await
                    .unwrap()
                    .status(),
                StatusCode::OK
            );
        }
        assert_eq!(
            app.oneshot(
                Request::post("/console/usage/clear")
                    .header("cookie", cookie)
                    .body(Body::empty())
                    .unwrap()
            )
            .await
            .unwrap()
            .status(),
            StatusCode::OK
        );
    }
}
