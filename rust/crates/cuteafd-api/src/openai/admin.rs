//! Runtime concurrency limits: `GET/POST /v1/admin/limits`.
//!
//! The allocated ceiling is fixed at start-up; the live limit can be changed
//! without a restart. GET sits under the ordinary bearer-key middleware (open
//! when the server has no API key, like `/v1/chat/completions`). POST is
//! authorized by the existing console secret (`--console-secret-file`,
//! `~/.cache/cuteafd/console/secret`), sent either as the console cookie or as
//! `Authorization: Bearer <secret>`: a locked gate refuses every POST, so a
//! LAN-open serving port never lets an anonymous caller change the limit. It is
//! therefore on `auth::key_exempt` for the API key, which one bearer header
//! cannot carry beside the console secret.
//!
//! ```text
//! GET /v1/admin/limits
//!   -> {"allocated":128,"live":8,"active":3,"slots_max":128}
//!
//! POST /v1/admin/limits   Authorization: Bearer <console secret>   {"concurrency": 12}
//!   -> {"requested":{"concurrency":12},"effective":{"concurrency":12},"allocated":128}
//! ```
//!
//! A request above the allocated ceiling is clamped to it; zero concurrency and
//! a non-integer body are rejected with 400, and a body with no known field is
//! 400 ("nothing to set").

use axum::{
    extract::{rejection::JsonRejection, ConnectInfo, State},
    http::{HeaderMap, StatusCode},
    response::{IntoResponse, Response},
    Json,
};
use serde::Deserialize;
use serde_json::{json, Value};
use std::{
    net::SocketAddr,
    sync::{Arc, RwLock},
};

use super::NativeState;
use crate::console_gate::ConsoleGate;

/// A refused runtime-limit change. Typed so the handler can map each case to
/// its own status and message instead of parsing a string.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum LimitError {
    #[error("concurrency must be at least 1")]
    Zero,
    #[error("concurrency {requested} exceeds the allocated ceiling {allocated}")]
    AboveCeiling { requested: u32, allocated: u32 },
    #[error("runtime limits are already installed (allocated {allocated}, live {live})")]
    AlreadyInstalled { allocated: u32, live: u32 },
}

/// The daemon-installed bridge to the live limit. The API crate cannot depend
/// on the daemon, so the daemon installs these closures once before serving.
pub struct AdminLimits {
    /// The start-time ceiling the live limit is clamped to.
    pub allocated: u32,
    /// The console gate POST requires; a locked gate refuses every POST.
    pub gate: ConsoleGate,
    /// The live limit the scheduler admits against (read at request time).
    pub live: Arc<dyn Fn() -> u32 + Send + Sync>,
    /// Set the live limit; returns the effective value or a typed refusal.
    pub set_concurrency: Arc<dyn Fn(u32) -> Result<u32, LimitError> + Send + Sync>,
}

static ADMIN: RwLock<Option<Arc<AdminLimits>>> = RwLock::new(None);

/// Install (or replace) the process-wide admin bridge.
pub fn install(limits: AdminLimits) {
    *ADMIN.write().unwrap_or_else(|e| e.into_inner()) = Some(Arc::new(limits));
}

fn current() -> Option<Arc<AdminLimits>> {
    ADMIN.read().unwrap_or_else(|e| e.into_inner()).clone()
}

#[cfg(test)]
pub(crate) fn clear() {
    *ADMIN.write().unwrap_or_else(|e| e.into_inner()) = None;
}

/// `GET /v1/admin/limits`: the allocated ceiling, the live limit, the current
/// occupancy and the admission bound. The ceiling and the live limit come from
/// the installed bridge (authoritative from the first request); `active`
/// comes from the last `/v1/stats` publication, so the handler never touches
/// the scheduler. Without a bridge the stats snapshot is the fallback.
pub(crate) async fn get_limits(State(state): State<NativeState>) -> Response {
    let stats = state.stats.lock().map(|stats| stats.clone()).unwrap_or(Value::Null);
    let limits = stats.get("limits").cloned().unwrap_or_else(|| json!({}));
    let bridge = current();
    let value = json!({
        "allocated": bridge.as_ref().map_or_else(|| limits.get("allocated").cloned().unwrap_or(json!(0)),
            |bridge| json!(bridge.allocated)),
        "live": bridge.as_ref().map_or_else(|| limits.get("live").cloned().unwrap_or(json!(0)),
            |bridge| json!((bridge.live)())),
        "active": stats.get("active").cloned().unwrap_or(json!(0)),
        "slots_max": limits.get("slots_max").cloned().unwrap_or(json!(0)),
    });
    Json(value).into_response()
}

#[derive(Deserialize)]
pub struct SetLimits {
    pub concurrency: Option<u32>,
}

/// `POST /v1/admin/limits`: set the live limit. Values above the allocated
/// ceiling are clamped; zero concurrency and a malformed body are 400, and a
/// body with no known field is 400 ("nothing to set").
pub(crate) async fn post_limits(
    remote: Option<ConnectInfo<SocketAddr>>,
    headers: HeaderMap,
    body: Result<Json<SetLimits>, JsonRejection>,
) -> Response {
    let Some(limits) = current() else {
        return (StatusCode::SERVICE_UNAVAILABLE,
            Json(json!({"error": {"message": "runtime limits are not installed"}}))).into_response();
    };
    // The console secret, checked before the body is read.
    if !limits.gate.authorized(&headers) {
        return (StatusCode::UNAUTHORIZED, Json(json!({"error": {"message":
            "send the console secret as Authorization: Bearer <secret> or unlock the console",
            "type": "invalid_request_error", "code": "invalid_admin_token"}}))).into_response();
    }
    let Json(body) = match body {
        Ok(body) => body,
        Err(rejection) => return bad_request(rejection.body_text()),
    };
    let Some(n) = body.concurrency else {
        return bad_request("nothing to set".into());
    };
    // Zero is refused here as well as in the bridge, so a bridge that forgets
    // to validate cannot disable admission.
    if n == 0 {
        return limit_error(LimitError::Zero);
    }
    let allocated = limits.allocated;
    let clamped = n.min(allocated.max(1));
    let effective = match (limits.set_concurrency)(clamped) {
        Ok(value) => value,
        Err(error) => return limit_error(error),
    };
    let remote = remote.map_or_else(|| "-".into(), |ConnectInfo(addr)| addr.ip().to_string());
    tracing::info!(target: "cuteafd::limits", requested = n, effective, allocated,
        remote = %remote, "runtime concurrency limit changed");
    Json(json!({"requested": {"concurrency": n}, "effective": {"concurrency": effective},
        "allocated": allocated})).into_response()
}

fn bad_request(message: String) -> Response {
    (StatusCode::BAD_REQUEST, Json(json!({"error": {"message": message}}))).into_response()
}

fn limit_error(error: LimitError) -> Response {
    let status = match error {
        LimitError::Zero | LimitError::AboveCeiling { .. } => StatusCode::BAD_REQUEST,
        LimitError::AlreadyInstalled { .. } => StatusCode::CONFLICT,
    };
    (status, Json(json!({"error": {"message": error.to_string()}}))).into_response()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::openai::{
        admission, auth::{ApiKey, Auth, require_key}, images, NativeLimits, ModelProfile,
    };
    use axum::{
        body::Body,
        http::Request,
        routing::get,
        Router,
    };
    use std::sync::Mutex;
    use tower::ServiceExt;

    static TEST_LOCK: Mutex<()> = Mutex::new(());
    const SECRET: &str = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";

    fn post(body: Value, token: Option<&str>) -> Request<Body> {
        let mut request = Request::post("/v1/admin/limits").header("content-type", "application/json");
        if let Some(token) = token { request = request.header("authorization", format!("Bearer {token}")); }
        request.body(Body::from(body.to_string())).unwrap()
    }

    fn state(stats: Value) -> NativeState {
        let (queue, _receive) = tokio::sync::mpsc::channel(4);
        NativeState {
            queue,
            limits: NativeLimits::default(),
            images: images::ImageDecoder::new(4),
            stats: Arc::new(Mutex::new(stats)),
            tables: cuteafd_loader::MappedTableStatsReader::default(),
            admission: admission::Admission::new(4, std::time::Duration::from_secs(25)),
            profile: Arc::new(ModelProfile::default()),
        }
    }

    fn app(stats: Value) -> Router {
        Router::new()
            .route("/v1/admin/limits", get(get_limits).post(post_limits))
            .with_state(state(stats))
    }

    fn keyed(stats: Value) -> Router {
        app(stats).layer(axum::middleware::from_fn_with_state(
            Auth { key: Some(ApiKey::new("secret").unwrap()), internal: None },
            require_key,
        ))
    }

    fn gate() -> ConsoleGate {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("secret");
        std::fs::write(&path, SECRET).unwrap();
        ConsoleGate::from_file(&path, false).unwrap()
    }

    fn install_test_limits(allocated: u32, live: u32) -> Arc<Mutex<Vec<u32>>> {
        let calls = Arc::new(Mutex::new(Vec::new()));
        let recorded = calls.clone();
        let current = Arc::new(std::sync::atomic::AtomicU32::new(live));
        let setter = current.clone();
        install(AdminLimits {
            allocated,
            gate: gate(),
            live: Arc::new(move || current.load(std::sync::atomic::Ordering::Relaxed)),
            set_concurrency: Arc::new(move |n| {
                recorded.lock().unwrap().push(n);
                let effective = n.min(allocated).max(1);
                setter.store(effective, std::sync::atomic::Ordering::Relaxed);
                Ok(effective)
            }),
        });
        calls
    }

    async fn body(response: Response) -> Value {
        let bytes = axum::body::to_bytes(response.into_body(), usize::MAX).await.unwrap();
        serde_json::from_slice(&bytes).unwrap()
    }

    #[tokio::test]
    async fn get_reports_allocated_live_active() {
        let _guard = TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        clear();
        install_test_limits(16, 8);
        let stats = json!({"limits": {"allocated": 16, "live": 8, "slots_max": 16}, "active": 3});
        let response = app(stats)
            .oneshot(Request::get("/v1/admin/limits").body(Body::empty()).unwrap())
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let value = body(response).await;
        assert_eq!(value["allocated"], 16);
        assert_eq!(value["live"], 8);
        assert_eq!(value["active"], 3);
        assert_eq!(value["slots_max"], 16);
        clear();
    }

    /// GET is authoritative before the first `/v1/stats` publish: the ceiling
    /// and the live limit come from the bridge, not the stats snapshot.
    #[tokio::test]
    async fn get_reads_the_bridge_before_the_first_stats_publish() {
        let _guard = TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        clear();
        install_test_limits(16, 8);
        let response = app(Value::Null)
            .oneshot(Request::get("/v1/admin/limits").body(Body::empty()).unwrap())
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let value = body(response).await;
        assert_eq!(value["allocated"], 16);
        assert_eq!(value["live"], 8);
        assert_eq!(value["active"], 0);
        clear();
    }

    #[tokio::test]
    async fn post_clamps_and_echoes_effective() {
        let _guard = TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        clear();
        let calls = install_test_limits(16, 8);
        let response = app(json!({}))
            .oneshot(post(json!({"concurrency": 99}), Some(SECRET)))
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let value = body(response).await;
        assert_eq!(value["requested"]["concurrency"], 99);
        assert_eq!(value["effective"]["concurrency"], 16);
        assert_eq!(value["allocated"], 16);
        assert_eq!(*calls.lock().unwrap(), vec![16]);
        clear();
    }

    #[tokio::test]
    async fn post_rejects_zero_and_non_integer_with_400() {
        let _guard = TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        clear();
        install_test_limits(16, 8);
        for body in [json!({"concurrency": 0}), json!({"concurrency": "many"})] {
            let response = app(json!({}))
                .oneshot(post(body.clone(), Some(SECRET)))
                .await
                .unwrap();
            assert_eq!(response.status(), StatusCode::BAD_REQUEST, "{body}");
        }
        clear();
    }

    /// POST needs the console secret in `Authorization: Bearer`; a wrong or
    /// missing secret is 401 and nothing changes.
    #[tokio::test]
    async fn post_requires_the_console_secret() {
        let _guard = TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        clear();
        let calls = install_test_limits(16, 8);
        for token in [None, Some("wrong"), Some(&SECRET[..63])] {
            let response = app(json!({})).oneshot(post(json!({"concurrency": 8}), token)).await.unwrap();
            assert_eq!(response.status(), StatusCode::UNAUTHORIZED, "{token:?}");
        }
        // Auth runs before the body is parsed.
        let response = app(json!({})).oneshot(post(json!({"concurrency": "many"}), None)).await.unwrap();
        assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
        assert!(calls.lock().unwrap().is_empty());
        let response = app(json!({})).oneshot(post(json!({"concurrency": 8}), Some(SECRET))).await.unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        assert_eq!(*calls.lock().unwrap(), vec![8]);
        clear();
    }

    /// A locked gate (no console secret configured) refuses every POST; GET
    /// stays open.
    #[tokio::test]
    async fn post_with_a_locked_gate_is_401_and_get_stays_open() {
        let _guard = TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        clear();
        let calls = Arc::new(Mutex::new(Vec::<u32>::new()));
        let recorded = calls.clone();
        install(AdminLimits { allocated: 16, gate: ConsoleGate::locked(), live: Arc::new(|| 8),
            set_concurrency: Arc::new(move |n| { recorded.lock().unwrap().push(n); Ok(n) }) });
        for token in [None, Some(SECRET), Some("")] {
            let response = app(json!({})).oneshot(post(json!({"concurrency": 8}), token)).await.unwrap();
            assert_eq!(response.status(), StatusCode::UNAUTHORIZED, "{token:?}");
        }
        assert!(calls.lock().unwrap().is_empty());
        let response = app(json!({}))
            .oneshot(Request::get("/v1/admin/limits").body(Body::empty()).unwrap()).await.unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        clear();
    }

    /// Under an API key the POST is authorized by the console secret alone (one
    /// bearer header carries one secret); the API key does not unlock it. GET
    /// follows the API key like every other `/v1` route.
    #[tokio::test]
    async fn keyed_server_post_takes_the_console_secret_not_the_api_key() {
        let _guard = TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        clear();
        install_test_limits(16, 8);
        let response = keyed(json!({})).oneshot(post(json!({"concurrency": 8}), Some("secret"))).await.unwrap();
        assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
        let response = keyed(json!({})).oneshot(post(json!({"concurrency": 8}), Some(SECRET))).await.unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let response = keyed(json!({}))
            .oneshot(Request::get("/v1/admin/limits").body(Body::empty()).unwrap()).await.unwrap();
        assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
        let response = keyed(json!({}))
            .oneshot(Request::get("/v1/admin/limits").header("authorization", "Bearer secret")
                .body(Body::empty()).unwrap()).await.unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        clear();
    }

    /// A body with no known field is 400 ("nothing to set").
    #[tokio::test]
    async fn post_rejects_empty_body() {
        let _guard = TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        clear();
        install_test_limits(16, 8);
        for payload in [json!({}), json!({"unknown": 1})] {
            let response = app(json!({}))
                .oneshot(post(payload.clone(), Some(SECRET)))
                .await
                .unwrap();
            assert_eq!(response.status(), StatusCode::BAD_REQUEST, "{payload}");
            assert!(body(response).await["error"]["message"].as_str().unwrap().contains("nothing to set"));
        }
        clear();
    }
}
