//! Console-gated, same-origin facet for the loopback-only DSH sidecar.
use axum::{body::Body, extract::{Request, State, ws::{WebSocketUpgrade, Message}},
    http::{header, HeaderMap, HeaderValue, StatusCode}, response::{Html, IntoResponse, Response}, routing::{any, get}, Router};
use base64::{engine::general_purpose::URL_SAFE_NO_PAD, Engine};
use futures::{SinkExt, StreamExt};
use sha2::{Digest, Sha256};
use std::{path::PathBuf, time::Duration};
use tokio_tungstenite::tungstenite::{client::IntoClientRequest, Message as UpMessage};

#[derive(Clone)]
pub struct AgentProxy {
    upstream: String,
    token_file: PathBuf,
    client: reqwest::Client,
}
impl AgentProxy {
    pub fn new(port: u16, token_file: PathBuf) -> Self {
        Self { upstream: format!("http://127.0.0.1:{port}"), token_file,
            client: reqwest::Client::builder().no_proxy().redirect(reqwest::redirect::Policy::none())
                .connect_timeout(Duration::from_secs(2)).build().expect("HTTP client") }
    }
    pub fn from_env() -> Self {
        let port = std::env::var("CUTEAFD_AGENT_PORT").ok().and_then(|s| s.parse().ok()).unwrap_or(3010);
        let home = std::env::var_os("HOME").map(PathBuf::from).unwrap_or_default();
        let path = std::env::var_os("CUTEAFD_AGENT_TOKEN_FILE").map(PathBuf::from)
            .unwrap_or_else(|| home.join(".local/share/cuteafd/agent/dsh/launch-token"));
        Self::new(port, path)
    }
    pub fn mount(self, router: Router, gate: crate::console_gate::ConsoleGate) -> Router {
        router.merge(Router::new().route("/agent", get(page))
            .route("/agent/theme.css", get(theme))
            .route("/agent/logo-light.svg", get(|| async { ([(header::CONTENT_TYPE, "image/svg+xml")], include_str!("../../../../assets/brand/cuteafd-logo-color.svg")) }))
            .route("/agent/app", any(proxy)).route("/agent/app/", any(proxy))
            .route("/agent/app/*path", any(proxy)).with_state(self)
            .layer(axum::middleware::from_fn_with_state(gate, crate::console_gate::require_console)))
    }
    async fn bootstrap(&self, authority: &str) -> Result<HeaderValue, ()> {
        use std::os::unix::fs::PermissionsExt;
        let metadata = tokio::fs::metadata(&self.token_file).await.map_err(|_| ())?;
        if metadata.permissions().mode() & 0o777 != 0o600 || metadata.len() > 4096 { return Err(()); }
        let raw = tokio::fs::read_to_string(&self.token_file).await.map_err(|_| ())?;
        let token = raw.trim_end_matches(['\r', '\n']);
        if token.is_empty() || !token.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'-' || b == b'_') { return Err(()); }
        // No redirects, body forwarding, tracing or error formatting on the secret-bearing request.
        let response = tokio::time::timeout(Duration::from_secs(5), self.client.get(format!("{}/", self.upstream))
            .query(&[("token", token)]).header(header::HOST, authority).send()).await.map_err(|_| ())?.map_err(|_| ())?;
        response.headers().get_all(header::SET_COOKIE).iter().find_map(|cookie| scoped_cookie(cookie, authority)).ok_or(())
    }
}
pub fn mount(router: Router, gate: crate::console_gate::ConsoleGate) -> Router {
    AgentProxy::from_env().mount(router, gate)
}
fn authority(headers: &HeaderMap) -> Option<String> {
    let raw = headers.get(header::HOST)?.to_str().ok()?;
    let parsed: axum::http::uri::Authority = raw.parse().ok()?;
    if parsed.as_str() != raw || raw.contains('@') { return None; }
    let url = url::Url::parse(&format!("http://{raw}/")).ok()?;
    Some(url[url::Position::BeforeHost..url::Position::AfterPort].to_owned())
}
fn cookie_name(authority: &str) -> String {
    format!("dsh-auth-{}", URL_SAFE_NO_PAD.encode(Sha256::digest(authority.as_bytes())))
}
fn dsh_cookie(headers: &HeaderMap, authority: &str) -> Option<String> {
    let name = cookie_name(authority);
    headers.get_all(header::COOKIE).iter().filter_map(|v| v.to_str().ok())
        .flat_map(|v| v.split(';')).find_map(|pair| {
            let (key, value) = pair.trim().split_once('=')?;
            (key == name && !value.is_empty()).then(|| format!("{key}={value}"))
        })
}
fn scoped_cookie(value: &HeaderValue, authority: &str) -> Option<HeaderValue> {
    let text = value.to_str().ok()?;
    let mut pieces = text.split(';');
    let first = pieces.next()?;
    if first.split_once('=')?.0 != cookie_name(authority) { return None; }
    let mut result = first.to_owned();
    for piece in pieces {
        let attr = piece.trim();
        let name = attr.split('=').next()?.to_ascii_lowercase();
        if name != "path" && name != "domain" { result.push_str("; "); result.push_str(attr); }
    }
    result.push_str("; Path=/agent/app/; HttpOnly; SameSite=Strict");
    result.parse().ok()
}
async fn theme() -> impl IntoResponse {
    ([(header::CONTENT_TYPE, "text/css; charset=utf-8")], include_str!("../assets/agent-theme.css"))
}
fn unavailable() -> Response {
    (StatusCode::SERVICE_UNAVAILABLE, "Agent sidecar is not running or ready. Start it with scripts/launch/agent.sh start.").into_response()
}
async fn page(State(state): State<AgentProxy>, headers: HeaderMap) -> Response {
    let Some(host) = authority(&headers) else { return StatusCode::BAD_REQUEST.into_response(); };
    let cookie = match state.bootstrap(&host).await { Ok(cookie) => cookie, Err(_) => {
        return Html(include_str!("../assets/agent.html").replace("<!--WORKSPACE-->",
            "<section class=\"agent-offline\"><h1>Agent workspace is offline</h1><p>Start the experimental sidecar, then reload this page.</p><code>scripts/launch/agent.sh start</code></section>")).into_response();
    }};
    let mut response = Html(include_str!("../assets/agent.html").replace("<!--WORKSPACE-->",
        "<iframe title=\"Agent workspace (experimental)\" src=\"/agent/app/\" allow=\"clipboard-read; clipboard-write\"></iframe>")).into_response();
    response.headers_mut().append(header::SET_COOKIE, cookie);
    response.headers_mut().insert(header::REFERRER_POLICY, HeaderValue::from_static("no-referrer"));
    response
}
fn hop(name: &header::HeaderName, headers: &HeaderMap) -> bool {
    matches!(name.as_str(), "connection" | "keep-alive" | "proxy-authenticate" | "proxy-authorization" | "te" | "trailer" | "transfer-encoding" | "upgrade")
        || headers.get(header::CONNECTION).and_then(|v| v.to_str().ok()).is_some_and(|s| s.split(',').any(|v| v.trim().eq_ignore_ascii_case(name.as_str())))
}
async fn proxy(State(state): State<AgentProxy>, ws: Option<WebSocketUpgrade>, request: Request) -> Response {
    let Some(host) = authority(request.headers()) else { return StatusCode::BAD_REQUEST.into_response(); };
    // The launch exchange is exclusively server-side; a browser token is never forwarded.
    if request.uri().query().is_some_and(|q| url::form_urlencoded::parse(q.as_bytes()).any(|(k, _)| k.eq_ignore_ascii_case("token"))) {
        return StatusCode::BAD_REQUEST.into_response();
    }
    if let Some(origin) = request.headers().get(header::ORIGIN) {
        let valid = origin.to_str().ok().and_then(|s| url::Url::parse(s).ok()).is_some_and(|u|
            matches!(u.scheme(), "http" | "https") && u[url::Position::BeforeHost..url::Position::AfterPort] == host);
        if !valid { return StatusCode::FORBIDDEN.into_response(); }
    }
    let bootstrap = if dsh_cookie(request.headers(), &host).is_none() {
        match state.bootstrap(&host).await { Ok(cookie) => Some(cookie), Err(_) => return unavailable() }
    } else { None };
    let cookie = dsh_cookie(request.headers(), &host).or_else(|| bootstrap.as_ref().and_then(|v| v.to_str().ok()).and_then(|v| v.split(';').next().map(str::to_owned))).unwrap_or_default();
    let suffix = request.uri().path_and_query().map(|v| v.as_str()).unwrap_or("/agent/app/")
        .strip_prefix("/agent/app").unwrap_or("/");
    let suffix = if suffix.is_empty() { "/" } else { suffix }.to_owned();
    if let Some(ws) = ws {
        let url = format!("ws://127.0.0.1:{}{suffix}", url::Url::parse(&state.upstream).expect("fixed URL").port().expect("explicit port"));
        let Ok(mut upstream) = url.into_client_request() else { return StatusCode::BAD_REQUEST.into_response(); };
        upstream.headers_mut().insert(header::HOST, host.parse().expect("validated host"));
        upstream.headers_mut().insert(header::COOKIE, cookie.parse().expect("validated cookie"));
        if let Some(origin) = request.headers().get(header::ORIGIN) { upstream.headers_mut().insert(header::ORIGIN, origin.clone()); }
        if let Some(protocols) = request.headers().get(header::SEC_WEBSOCKET_PROTOCOL) { upstream.headers_mut().insert(header::SEC_WEBSOCKET_PROTOCOL, protocols.clone()); }
        let Ok(Ok((remote, reply))) = tokio::time::timeout(Duration::from_secs(5), tokio_tungstenite::connect_async(upstream)).await else { return unavailable(); };
        let protocol = reply.headers().get(header::SEC_WEBSOCKET_PROTOCOL).and_then(|v| v.to_str().ok()).map(str::to_owned);
        let ws = if let Some(p) = protocol { ws.protocols([p]) } else { ws };
        let mut response = ws.on_upgrade(move |client| async move {
            let (mut out, mut incoming) = client.split();
            let (mut remote_out, mut remote_in) = remote.split();
            let send = async {
                while let Some(Ok(message)) = incoming.next().await {
                    let converted = match message {
                        Message::Text(v) => UpMessage::Text(v), Message::Binary(v) => UpMessage::Binary(v),
                        Message::Ping(v) => UpMessage::Ping(v), Message::Pong(v) => UpMessage::Pong(v),
                        Message::Close(frame) => UpMessage::Close(frame.map(|f| tokio_tungstenite::tungstenite::protocol::CloseFrame { code: f.code.into(), reason: f.reason })),
                    };
                    if remote_out.send(converted).await.is_err() { break; }
                }
            };
            let receive = async {
                while let Some(Ok(message)) = remote_in.next().await {
                    let converted = match message {
                        UpMessage::Text(v) => Message::Text(v), UpMessage::Binary(v) => Message::Binary(v),
                        UpMessage::Ping(v) => Message::Ping(v), UpMessage::Pong(v) => Message::Pong(v),
                        UpMessage::Close(frame) => Message::Close(frame.map(|f| axum::extract::ws::CloseFrame { code: f.code.into(), reason: f.reason })), UpMessage::Frame(_) => continue,
                    };
                    if out.send(converted).await.is_err() { break; }
                }
            };
            tokio::select! { _ = send => {}, _ = receive => {} }
        }).into_response();
        if let Some(cookie) = bootstrap { response.headers_mut().append(header::SET_COOKIE, cookie); }
        return response;
    }
    let (parts, body) = request.into_parts();
    let mut headers = HeaderMap::new();
    for (name, value) in &parts.headers {
        if !hop(name, &parts.headers) && !matches!(name.as_str(), "host" | "cookie" | "authorization" | "x-api-key" | "x-forwarded-host" | "x-forwarded-proto" | "forwarded") {
            headers.append(name, value.clone());
        }
    }
    headers.insert(header::HOST, host.parse().expect("validated host"));
    headers.insert(header::COOKIE, cookie.parse().expect("validated cookie"));
    let remote = state.client.request(parts.method, format!("{}{suffix}", state.upstream)).headers(headers)
        .body(reqwest::Body::wrap_stream(body.into_data_stream())).send().await;
    let Ok(remote) = remote else { return unavailable(); };
    let mut response = Response::builder().status(remote.status());
    for (name, value) in remote.headers() {
        if hop(name, remote.headers()) { continue; }
        if name == header::SET_COOKIE {
            if let Some(cookie) = scoped_cookie(value, &host) { response = response.header(name, cookie); }
        } else if name == header::LOCATION {
            if let Ok(path) = value.to_str() {
                if path.starts_with('/') && !path.starts_with("//") && !path.contains("token=") { response = response.header(name, format!("/agent/app{path}")); }
                else { return StatusCode::BAD_GATEWAY.into_response(); }
            }
        } else { response = response.header(name, value); }
    }
    if let Some(cookie) = bootstrap { response = response.header(header::SET_COOKIE, cookie); }
    response.body(Body::from_stream(remote.bytes_stream())).expect("upstream response")
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::{http::Request as HttpRequest, extract::ws::WebSocket};
    use std::{sync::{Arc, atomic::{AtomicUsize, Ordering}}, os::unix::fs::PermissionsExt};
    use tower::ServiceExt;
    async fn fixture() -> (tempfile::TempDir, crate::console_gate::ConsoleGate, AgentProxy, Arc<AtomicUsize>, tokio::task::JoinHandle<()>) {
        let dir = tempfile::tempdir().unwrap();
        let secret = dir.path().join("console");
        std::fs::write(&secret, "a".repeat(64)).unwrap();
        let token = dir.path().join("launch-token");
        std::fs::write(&token, "TOKEN_SENTINEL\n").unwrap();
        std::fs::set_permissions(&token, std::fs::Permissions::from_mode(0o600)).unwrap();
        let calls = Arc::new(AtomicUsize::new(0));
        let count = calls.clone();
        let upstream = Router::new().route("/", get(move |headers: HeaderMap, uri: axum::http::Uri| {
            let count = count.clone();
            async move {
                count.fetch_add(1, Ordering::SeqCst);
                assert_eq!(headers[header::HOST], "console.test:8000");
                if uri.query() == Some("token=TOKEN_SENTINEL") {
                    ([(header::SET_COOKIE, format!("{}=signed; HttpOnly; SameSite=Strict; Path=/", cookie_name("console.test:8000")))], "never-forward-token-body").into_response()
                } else { "stub workspace".into_response() }
            }
        })).route("/echo", any(|headers: HeaderMap, uri: axum::http::Uri| async move {
            assert!(!headers.contains_key(header::AUTHORIZATION));
            assert!(!headers[header::COOKIE].to_str().unwrap().contains("cuteafd_console"));
            uri.to_string()
        })).route("/api/remote.mux", get(|ws: WebSocketUpgrade| async move {
            ws.on_upgrade(|mut socket: WebSocket| async move {
                while let Some(Ok(message)) = socket.recv().await { if socket.send(message).await.is_err() { break; } }
            })
        }));
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = listener.local_addr().unwrap().port();
        let task = tokio::spawn(async move { axum::serve(listener, upstream).await.unwrap(); });
        let gate = crate::console_gate::ConsoleGate::from_file(&secret, false).unwrap();
        let proxy = AgentProxy::new(port, token);
        (dir, gate, proxy, calls, task)
    }
    async fn console_cookie(gate: &crate::console_gate::ConsoleGate) -> String {
        let response = gate.mount(Router::new()).oneshot(HttpRequest::get(format!("/console/unlock?token={}", "a".repeat(64))).body(Body::empty()).unwrap()).await.unwrap();
        response.headers()[header::SET_COOKIE].to_str().unwrap().split(';').next().unwrap().to_owned()
    }
    #[tokio::test]
    async fn locked_http_cookie_bootstrap_and_token_secrecy() {
        let (_dir, gate, proxy, calls, task) = fixture().await;
        let cookie = console_cookie(&gate).await;
        let app = proxy.mount(Router::new(), gate);
        for route in ["/agent", "/agent/app/", "/agent/app/api/remote.mux", "/agent/theme.css"] {
            let response = app.clone().oneshot(HttpRequest::get(route).header("host", "console.test:8000").body(Body::empty()).unwrap()).await.unwrap();
            assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
        }
        assert_eq!(calls.load(Ordering::SeqCst), 0);
        let response = app.clone().oneshot(HttpRequest::get("/agent").header("host", "console.test:8000").header("cookie", &cookie).body(Body::empty()).unwrap()).await.unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let set = response.headers()[header::SET_COOKIE].to_str().unwrap();
        assert!(set.contains("Path=/agent/app/"));
        assert!(!set.contains("TOKEN_SENTINEL"));
        let body = axum::body::to_bytes(response.into_body(), 65536).await.unwrap();
        let body = String::from_utf8_lossy(&body);
        assert!(body.contains("src=\"/agent/app/\""));
        assert!(!body.contains("TOKEN_SENTINEL") && !body.contains("never-forward-token-body"));
        let response = app.clone().oneshot(HttpRequest::get("/agent/app/echo?a=b").header("host", "console.test:8000").header("cookie", &cookie).header("authorization", "Bearer no-leak").body(Body::empty()).unwrap()).await.unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        assert_eq!(axum::body::to_bytes(response.into_body(), 65536).await.unwrap(), "/echo?a=b");
        let response = app.oneshot(HttpRequest::get("/agent/app/?token=browser-secret").header("host", "console.test:8000").header("cookie", cookie).body(Body::empty()).unwrap()).await.unwrap();
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        task.abort();
    }
    #[tokio::test]
    async fn websocket_remote_mux_passes_text_and_binary() {
        let (_dir, gate, proxy, _, task) = fixture().await;
        let cookie = console_cookie(&gate).await;
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = listener.local_addr().unwrap().port();
        let app = proxy.mount(Router::new(), gate);
        let server = tokio::spawn(async move { axum::serve(listener, app).await.unwrap(); });
        let mut request = format!("ws://127.0.0.1:{port}/agent/app/api/remote.mux").into_client_request().unwrap();
        request.headers_mut().insert(header::HOST, "console.test:8000".parse().unwrap());
        request.headers_mut().insert(header::COOKIE, cookie.parse().unwrap());
        let (mut socket, response) = tokio_tungstenite::connect_async(request).await.unwrap();
        assert_eq!(response.status(), StatusCode::SWITCHING_PROTOCOLS);
        for message in [UpMessage::Text("remote hello".into()), UpMessage::Binary(vec![0, 1, 255])] {
            socket.send(message.clone()).await.unwrap();
            assert_eq!(socket.next().await.unwrap().unwrap(), message);
        }
        socket.close(None).await.unwrap();
        server.abort(); task.abort();
    }
}
