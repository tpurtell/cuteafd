//! Console-only cookies grant no API authority. Secrets never implement Debug.
use axum::{extract::{Query, Request, State}, http::{header, HeaderMap, HeaderValue, StatusCode}, middleware::Next, response::{IntoResponse, Response}, routing::get, Json, Router};
use hmac::{Hmac, Mac};
use sha2::Sha256;
use std::{path::{Path, PathBuf}, sync::{Arc, Mutex}, time::{Duration, Instant, SystemTime, UNIX_EPOCH}};
use crate::openai::auth::constant_time_eq;
const MAX_AGE: u64 = 31536000;
const RENEW: u64 = 30 * 86400;
struct Secret { bytes: Vec<u8>, mtime: Option<SystemTime>, checked: Instant, valid: bool }
#[derive(Clone)]
pub struct ConsoleGate { secret: Arc<Mutex<Secret>>, path: Option<PathBuf>, secure: bool, clock: Arc<dyn Fn() -> u64 + Send + Sync> }
impl ConsoleGate {
    pub fn from_file(path: &Path, secure: bool) -> std::io::Result<Self> {
        let bytes = read_secret(path)?;
        Ok(Self { secret: Arc::new(Mutex::new(Secret { bytes, mtime: std::fs::metadata(path)?.modified().ok(), checked: Instant::now(), valid: true })), path: Some(path.into()), secure,
            clock: Arc::new(|| SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_secs()) })
    }
    /// Without a configured secret protected routes remain locked.
    pub fn locked() -> Self {
        Self { secret: Arc::new(Mutex::new(Secret { bytes: vec![], mtime: None, checked: Instant::now(), valid: false })), path: None, secure: false,
            clock: Arc::new(|| SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_secs()) }
    }
    fn with_secret<T>(&self, f: impl FnOnce(&[u8], bool) -> T) -> T {
        let mut s = self.secret.lock().unwrap_or_else(|e| e.into_inner());
        if s.checked.elapsed() >= Duration::from_secs(10) {
            s.checked = Instant::now();
            if let Some(path) = &self.path {
                match std::fs::metadata(path).and_then(|m| m.modified()) {
                    Ok(mtime) if s.mtime != Some(mtime) => match read_secret(path) { Ok(bytes) => { s.bytes = bytes; s.mtime = Some(mtime); s.valid = true; }, Err(_) => {} },
                    Err(_) => s.valid = false, _ => {},
                }
            }
        }
        f(&s.bytes, s.valid)
    }
    fn signature(secret: &[u8], issued: u64) -> String {
        let mut mac = Hmac::<Sha256>::new_from_slice(secret).expect("HMAC permits any key length");
        mac.update(format!("console|{issued}").as_bytes());
        mac.finalize().into_bytes().iter().map(|b| format!("{b:02x}")).collect()
    }
    fn cookie_value(&self, issued: u64) -> String { self.with_secret(|s, _| format!("v1.{issued}.{}", Self::signature(s, issued))) }
    fn issued(&self, headers: &HeaderMap) -> Option<u64> {
        let value = headers.get_all(header::COOKIE).iter().filter_map(|v| v.to_str().ok()).flat_map(|v| v.split(';'))
            .filter_map(|pair| pair.trim().split_once('=')) .find(|(key, _)| *key == "cuteafd_console")?.1;
        let mut parts = value.split('.'); if parts.next()? != "v1" { return None; }
        let raw = parts.next()?; let issued: u64 = raw.parse().ok()?; if raw != issued.to_string() { return None; }
        let sig = parts.next()?; if parts.next().is_some() { return None; }
        let now = (self.clock)(); if issued > now || now - issued >= MAX_AGE { return None; }
        self.with_secret(|s, valid| (valid && constant_time_eq(Self::signature(s, issued).as_bytes(), sig.as_bytes())).then_some(issued))
    }
    pub fn unlocked(&self, headers: &HeaderMap) -> bool { self.issued(headers).is_some() }
    fn set_cookie(&self, headers: &HeaderMap) -> HeaderValue {
        let secure = self.secure || headers.get("x-forwarded-proto").is_some_and(|v| v == "https");
        HeaderValue::from_str(&format!("cuteafd_console={}; HttpOnly; SameSite=Lax; Path=/; Max-Age={MAX_AGE}{}", self.cookie_value((self.clock)()), if secure { "; Secure" } else { "" })).expect("cookie is ASCII")
    }
    pub fn mount(&self, router: Router) -> Router {
        router.merge(Router::new().route("/console/unlock", get(unlock)).with_state(self.clone()))
    }
}
fn read_secret(path: &Path) -> std::io::Result<Vec<u8>> {
    let text = std::fs::read_to_string(path)?; let secret = text.trim_end_matches(['\r', '\n']);
    if secret.len() != 64 || !secret.bytes().all(|b| b.is_ascii_hexdigit()) {
        return Err(std::io::Error::new(std::io::ErrorKind::InvalidData, "console secret must be 32 bytes encoded as hex"));
    }
    Ok(secret.as_bytes().to_vec())
}
fn locked() -> Response { (StatusCode::UNAUTHORIZED, Json(serde_json::json!({"error":{"type":"console_locked"}}))).into_response() }
#[derive(serde::Deserialize)]
struct Unlock { token: String }
async fn unlock(State(gate): State<ConsoleGate>, headers: HeaderMap, query: std::result::Result<Query<Unlock>, axum::extract::rejection::QueryRejection>) -> Response {
    let mut response = match query {
        Ok(Query(query)) if gate.with_secret(|s, valid| valid && constant_time_eq(s, query.token.as_bytes())) => {
            let mut response = StatusCode::SEE_OTHER.into_response(); response.headers_mut().insert(header::LOCATION, HeaderValue::from_static("/"));
            response.headers_mut().insert(header::SET_COOKIE, gate.set_cookie(&headers)); response
        }, _ => locked(),
    };
    response.headers_mut().insert(header::REFERRER_POLICY, HeaderValue::from_static("no-referrer"));
    response.headers_mut().insert(header::CACHE_CONTROL, HeaderValue::from_static("no-store")); response
}
pub async fn require_console(State(gate): State<ConsoleGate>, request: Request, next: Next) -> Response {
    let Some(issued) = gate.issued(request.headers()) else { return locked(); };
    let renewal = ((gate.clock)() - issued >= RENEW).then(|| gate.set_cookie(request.headers()));
    let mut response = next.run(request).await;
    if let Some(cookie) = renewal { response.headers_mut().append(header::SET_COOKIE, cookie); }
    response.headers_mut().insert(header::CACHE_CONTROL, HeaderValue::from_static("no-store")); response
}
#[cfg(test)]
mod tests {
    use super::*;
    use axum::{body::Body, http::Request}; use tower::ServiceExt;
    fn fixture() -> (tempfile::NamedTempFile, ConsoleGate) { let f = tempfile::NamedTempFile::new().unwrap(); std::fs::write(f.path(), "a".repeat(64)).unwrap(); let mut g = ConsoleGate::from_file(f.path(), false).unwrap(); g.clock = Arc::new(|| 4000000); (f, g) }
    fn app(g: &ConsoleGate) -> Router { g.mount(Router::new().route("/console/usage/settings", get(|| async { "ok" })).layer(axum::middleware::from_fn_with_state(g.clone(), require_console))) }
    #[tokio::test]
    async fn unlock_attributes_wrong_stale_api_and_renewal() {
        let (_file, gate) = fixture(); let app = app(&gate);
        for token in ["wrong".into(), "a".repeat(63), "a".repeat(65)] { assert_eq!(app.clone().oneshot(Request::get(format!("/console/unlock?token={token}")).body(Body::empty()).unwrap()).await.unwrap().status(), StatusCode::UNAUTHORIZED); }
        let response = app.clone().oneshot(Request::get(format!("/console/unlock?token={}", "a".repeat(64))).header("x-forwarded-proto", "https").body(Body::empty()).unwrap()).await.unwrap();
        assert_eq!(response.status(), StatusCode::SEE_OTHER); assert_eq!(response.headers()[header::LOCATION], "/"); assert_eq!(response.headers()[header::REFERRER_POLICY], "no-referrer"); assert_eq!(response.headers()[header::CACHE_CONTROL], "no-store");
        let cookie = response.headers()[header::SET_COOKIE].to_str().unwrap(); for value in ["HttpOnly", "SameSite=Lax", "Path=/", "Max-Age=31536000", "Secure"] { assert!(cookie.contains(value)); } assert!(!cookie.contains(&"a".repeat(64)));
        for issued in [4000001, 0] { let bad = if issued == 0 { "v1.1.wrong".into() } else { gate.cookie_value(issued) }; assert_eq!(app.clone().oneshot(Request::get("/console/usage/settings").header("cookie", format!("cuteafd_console={bad}")).body(Body::empty()).unwrap()).await.unwrap().status(), StatusCode::UNAUTHORIZED); }
        assert_eq!(app.clone().oneshot(Request::get("/console/usage/settings").header("Authorization", "Bearer API_KEY").body(Body::empty()).unwrap()).await.unwrap().status(), StatusCode::UNAUTHORIZED);
        let old = gate.cookie_value(4000000 - RENEW - 1); let response = app.oneshot(Request::get("/console/usage/settings").header("cookie", format!("cuteafd_console={old}")).body(Body::empty()).unwrap()).await.unwrap(); assert_eq!(response.status(), StatusCode::OK); assert!(response.headers().contains_key(header::SET_COOKIE));
        let mut stale = gate.clone(); stale.clock = Arc::new(|| 4000000 + MAX_AGE); let mut headers = HeaderMap::new(); headers.insert("cookie", format!("cuteafd_console={}", gate.cookie_value(4000000)).parse().unwrap()); assert!(!stale.unlocked(&headers));
    }
    #[tokio::test]
    async fn rotation_and_api_isolation() {
        let (file, gate) = fixture(); let mut headers = HeaderMap::new(); headers.insert("cookie", format!("cuteafd_console={}", gate.cookie_value(4000000)).parse().unwrap()); assert!(gate.unlocked(&headers));
        std::fs::write(file.path(), "").unwrap(); gate.secret.lock().unwrap().checked = Instant::now() - Duration::from_secs(11); assert!(gate.unlocked(&headers));
        std::fs::write(file.path(), "b".repeat(64)).unwrap(); gate.secret.lock().unwrap().checked = Instant::now() - Duration::from_secs(11); assert!(!gate.unlocked(&headers));
        let app = Router::new().route("/v1/models", get(|| async { "ok" })).layer(axum::middleware::from_fn_with_state(crate::openai::auth::Auth { key: Some(crate::openai::auth::ApiKey::new("api-key").unwrap()), internal: None }, crate::openai::auth::require_key));
        assert_eq!(app.oneshot(Request::get("/v1/models").header("cookie", format!("cuteafd_console={}", gate.cookie_value(4000000))).body(Body::empty()).unwrap()).await.unwrap().status(), StatusCode::UNAUTHORIZED);
    }
    #[tokio::test]
    async fn unlock_query_never_traced() {
        #[derive(Clone)] struct Buffer(Arc<Mutex<Vec<u8>>>);
        impl std::io::Write for Buffer { fn write(&mut self,b:&[u8])->std::io::Result<usize> { self.0.lock().unwrap().extend_from_slice(b); Ok(b.len()) } fn flush(&mut self)->std::io::Result<()> { Ok(()) } }
        let bytes = Arc::new(Mutex::new(vec![])); let writer = bytes.clone();
        let subscriber = tracing_subscriber::fmt().with_writer(move || Buffer(writer.clone())).with_max_level(tracing::Level::TRACE).finish();
        let _guard = tracing::subscriber::set_default(subscriber); let (_f,g) = fixture();
        let response = app(&g).oneshot(Request::get("/console/unlock?token=QUERY_SECRET_SENTINEL").body(Body::empty()).unwrap()).await.unwrap(); assert_eq!(response.status(), StatusCode::UNAUTHORIZED); assert!(!String::from_utf8_lossy(&bytes.lock().unwrap()).contains("QUERY_SECRET_SENTINEL"));
    }
}
