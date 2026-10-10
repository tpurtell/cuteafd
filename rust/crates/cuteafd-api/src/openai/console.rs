//! Live engine console: the `/` page, its WebSocket feed and a JSON snapshot.
//!
//! The CUDA worker never touches this module's sockets. A producer on the worker
//! side checks [`ConsoleHub::viewers`] (one relaxed atomic load) before building
//! any per-round telemetry, hands owned events to a separate console thread, and
//! that thread publishes serialized frames here. Frames fan out to every viewer
//! through a bounded broadcast channel; a viewer that falls behind receives a
//! fresh snapshot instead of the frames it missed.
use axum::{
    extract::{
        ws::{Message, WebSocket, WebSocketUpgrade},
        State,
    },
    http::{header, HeaderMap},
    response::{Html, IntoResponse, Response},
};
use std::sync::{
    atomic::{AtomicBool, AtomicUsize, Ordering},
    Arc, Mutex, OnceLock,
};
use tokio::sync::broadcast;

/// The console page, compiled into the binary so it needs no asset path or CDN.
pub const PAGE: &str = include_str!("../../assets/console.html");
/// Shared page shell, palette and chart styles for every built-in page
/// (`/assets/cuteafd-ui.css`).
pub const UI_CSS: &str = include_str!("../../assets/cuteafd-ui.css");
/// Shared page shell, formatting and SVG chart primitives (`/assets/cuteafd-ui.js`, `window.CuteUI`).
pub const UI_JS: &str = include_str!("../../assets/cuteafd-ui.js");
/// The header logo for the dark pages (`/assets/cuteafd-logo.svg`).
pub const LOGO: &str = include_str!("../../../../../assets/brand/cuteafd-logo-color-dark.svg");
/// The square swift mark, the pages' favicon (`/assets/cuteafd-mark.svg`).
pub const MARK: &str = include_str!("../../../../../assets/brand/cuteafd-mark-color-dark.svg");

const DISABLED: &str = r#"{"type":"snapshot","disabled":true}"#;
const STARTING: &str = r#"{"type":"snapshot","starting":true}"#;

/// One published frame: the full variant, and the variant without token text
/// for locked viewers when the two differ.
#[derive(Clone)]
struct Frame {
    full: Arc<str>,
    plain: Option<Arc<str>>,
}
impl Frame {
    fn new(full: String, plain: Option<String>) -> Self {
        Self { full: Arc::from(full), plain: plain.map(Arc::from) }
    }
    fn for_viewer(&self, unlocked: bool) -> Arc<str> {
        match (&self.plain, unlocked) {
            (Some(plain), false) => plain.clone(),
            _ => self.full.clone(),
        }
    }
}

pub struct ConsoleHub {
    viewers: AtomicUsize,
    /// Viewers whose connection may receive token text (unlocked at connect).
    text_viewers: AtomicUsize,
    text: bool,
    /// Token text is allowed while a benchmark run holds the server: its own
    /// synthetic prompts are the only requests the bench lockout admits.
    bench_text: AtomicBool,
    enabled: bool,
    frames: broadcast::Sender<Frame>,
    snapshot: Mutex<Frame>,
    /// Token text reaches only viewers this gate unlocks (the console cookie).
    gate: OnceLock<crate::console_gate::ConsoleGate>,
}

/// What a viewer may see of token text.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TextAccess {
    /// Text streams to this viewer.
    On,
    /// The server streams text, but only to unlocked viewers.
    Locked,
    Off,
}
impl TextAccess {
    pub fn name(self) -> &'static str {
        match self {
            Self::On => "on",
            Self::Locked => "locked",
            Self::Off => "off",
        }
    }
}

impl ConsoleHub {
    /// A hub the serving worker publishes to. `text` allows token text on the wire.
    pub fn new(text: bool) -> Arc<Self> {
        Self::build(true, text, STARTING)
    }
    /// A hub with no producer: the page loads and reports that the feed is off.
    pub fn disabled() -> Arc<Self> {
        Self::build(false, false, DISABLED)
    }
    fn build(enabled: bool, text: bool, snapshot: &str) -> Arc<Self> {
        let (frames, _) = broadcast::channel(256);
        Arc::new(Self {
            viewers: AtomicUsize::new(0),
            text_viewers: AtomicUsize::new(0),
            text,
            bench_text: AtomicBool::new(false),
            enabled,
            frames,
            snapshot: Mutex::new(Frame::new(snapshot.into(), None)),
            gate: OnceLock::new(),
        })
    }
    /// Token text then reaches only viewers holding the console cookie.
    pub fn set_gate(&self, gate: crate::console_gate::ConsoleGate) {
        let _ = self.gate.set(gate);
    }
    /// The bench's synthetic prompts are public; otherwise text needs the cookie.
    pub fn bench_text_active(&self) -> bool {
        self.bench_text.load(Ordering::Relaxed)
    }
    /// What a connection with these headers may see, decided once at connect.
    pub fn text_access(&self, headers: &HeaderMap) -> TextAccess {
        if !self.text_enabled() {
            TextAccess::Off
        } else if self.bench_text_active() || self.gate.get().is_some_and(|g| g.unlocked(headers)) {
            TextAccess::On
        } else {
            TextAccess::Locked
        }
    }
    /// The producer builds token text only while some viewer can receive it.
    #[inline]
    pub fn text_wanted(&self) -> bool {
        self.text_enabled() && self.viewers() > 0
            && (self.bench_text_active() || self.text_viewers.load(Ordering::Relaxed) > 0)
    }
    /// Connected console sockets. Producers skip all per-round work at zero.
    #[inline]
    pub fn viewers(&self) -> usize {
        self.viewers.load(Ordering::Relaxed)
    }
    /// Token text may go on the wire: the server's own switch, or a benchmark
    /// run holding the server. A hub with no producer is always off.
    pub fn text_enabled(&self) -> bool {
        self.enabled && (self.text || self.bench_text.load(Ordering::Relaxed))
    }
    /// The bench runner sets this on while a run holds the server and clears it
    /// on every exit path.
    pub fn set_bench_active(&self, active: bool) {
        self.bench_text.store(active, Ordering::Relaxed);
    }
    pub fn publish(&self, frame: String) {
        self.publish_pair(frame, None);
    }
    /// `plain` is the frame without token text, for viewers without the cookie.
    pub fn publish_pair(&self, full: String, plain: Option<String>) {
        // No receivers is the normal idle case, not an error.
        let _ = self.frames.send(Frame::new(full, plain));
    }
    pub fn set_snapshot(&self, snapshot: String) {
        self.set_snapshot_pair(snapshot, None);
    }
    pub fn set_snapshot_pair(&self, full: String, plain: Option<String>) {
        if let Ok(mut slot) = self.snapshot.lock() {
            *slot = Frame::new(full, plain);
        }
    }
    /// The snapshot a locked viewer sees (the public one).
    pub fn snapshot(&self) -> Arc<str> {
        self.snapshot_for(false)
    }
    pub fn snapshot_for(&self, unlocked: bool) -> Arc<str> {
        self.snapshot
            .lock()
            .map(|slot| slot.for_viewer(unlocked))
            .unwrap_or_else(|_| Arc::from("{}"))
    }
    fn viewer(self: &Arc<Self>, headers: &HeaderMap) -> (Option<Viewer>, bool) {
        let text = self.text_access(headers) == TextAccess::On;
        let unlocked = self.gate.get().is_some_and(|g| g.unlocked(headers));
        let viewer = self.enabled.then(|| {
            self.viewers.fetch_add(1, Ordering::Relaxed);
            if text { self.text_viewers.fetch_add(1, Ordering::Relaxed); }
            Viewer(self.clone(), text)
        });
        (viewer, unlocked)
    }
}

struct Viewer(Arc<ConsoleHub>, bool);
impl Drop for Viewer {
    fn drop(&mut self) {
        self.0.viewers.fetch_sub(1, Ordering::Relaxed);
        if self.1 { self.0.text_viewers.fetch_sub(1, Ordering::Relaxed); }
    }
}

/// Serve the compiled-in page, or, for page development, the file named by
/// `CUTEAFD_CONSOLE_PAGE`, re-read on every load.
pub(super) async fn page() -> Response {
    let page = match std::env::var_os("CUTEAFD_CONSOLE_PAGE") {
        Some(path) => match tokio::fs::read_to_string(&path).await {
            Ok(page) => page,
            Err(error) => {
                tracing::warn!(%error, path = %path.to_string_lossy(), "console page override unreadable");
                PAGE.to_string()
            }
        },
        None => PAGE.to_string(),
    };
    ([(header::CACHE_CONTROL, "no-cache")], Html(page)).into_response()
}

/// A shared UI asset, or for page development the same-named file in the
/// directory `CUTEAFD_CONSOLE_ASSETS`, re-read on every load.
async fn asset(name: &str, builtin: &'static str, content_type: &'static str) -> Response {
    let body = match std::env::var_os("CUTEAFD_CONSOLE_ASSETS") {
        Some(dir) => tokio::fs::read_to_string(std::path::Path::new(&dir).join(name)).await
            .unwrap_or_else(|_| builtin.to_string()),
        None => builtin.to_string(),
    };
    ([(header::CONTENT_TYPE, content_type), (header::CACHE_CONTROL, "no-cache")], body).into_response()
}

pub(super) async fn ui_css() -> Response {
    asset("cuteafd-ui.css", UI_CSS, "text/css; charset=utf-8").await
}

pub(super) async fn ui_js() -> Response {
    asset("cuteafd-ui.js", UI_JS, "text/javascript; charset=utf-8").await
}

/// The shared UI assets alone, for servers without the live console (the CPU gateway).
pub fn asset_routes() -> axum::Router {
    use axum::routing::get;
    axum::Router::new()
        .route("/assets/cuteafd-ui.css", get(ui_css))
        .route("/assets/cuteafd-ui.js", get(ui_js))
        .route("/assets/cuteafd-logo.svg", get(logo))
        .route("/assets/cuteafd-mark.svg", get(mark))
}

pub(super) async fn logo() -> Response {
    asset("cuteafd-logo.svg", LOGO, "image/svg+xml").await
}

pub(super) async fn mark() -> Response {
    asset("cuteafd-mark.svg", MARK, "image/svg+xml").await
}

pub(super) async fn snapshot(State(hub): State<Arc<ConsoleHub>>, headers: HeaderMap) -> Response {
    let unlocked = hub.gate.get().is_some_and(|g| g.unlocked(&headers));
    (
        [
            (header::CONTENT_TYPE, "application/json"),
            (header::CACHE_CONTROL, "no-cache"),
        ],
        hub.snapshot_for(unlocked).to_string(),
    )
        .into_response()
}

/// Browsers cannot send bearer headers on WebSocket handshakes; offer the same
/// authenticated feed over fetch/SSE without putting credentials in a URL.
pub(super) async fn events(State(hub): State<Arc<ConsoleHub>>, headers: HeaderMap) -> Response {
    let mut frames = hub.frames.subscribe();
    let (viewer, unlocked) = hub.viewer(&headers);
    let stream = async_stream::stream! {
        let _viewer = viewer;
        yield Ok::<_, std::convert::Infallible>(format!("data: {}\n\n", hub.snapshot_for(unlocked)));
        loop {
            match tokio::time::timeout(std::time::Duration::from_secs(15), frames.recv()).await {
                Ok(Ok(frame)) => yield Ok(format!("data: {}\n\n", frame.for_viewer(unlocked))),
                Ok(Err(broadcast::error::RecvError::Lagged(_))) => yield Ok(format!("data: {}\n\n", hub.snapshot_for(unlocked))),
                Ok(Err(broadcast::error::RecvError::Closed)) => break,
                Err(_) => yield Ok(": keepalive\n\n".to_string()),
            }
        }
    };
    ([(header::CONTENT_TYPE, "text/event-stream"), (header::CACHE_CONTROL, "no-cache")],
        axum::body::Body::from_stream(stream)).into_response()
}
pub(super) async fn socket(State(hub): State<Arc<ConsoleHub>>, headers: HeaderMap, upgrade: WebSocketUpgrade) -> Response {
    upgrade.on_upgrade(move |socket| serve(hub, headers, socket))
}

async fn serve(hub: Arc<ConsoleHub>, headers: HeaderMap, mut socket: WebSocket) {
    // Subscribe before counting the viewer so no frame published after the
    // producer sees this viewer can be missed.
    let mut frames = hub.frames.subscribe();
    let (_viewer, unlocked) = hub.viewer(&headers);
    if socket.send(Message::Text(hub.snapshot_for(unlocked).to_string())).await.is_err() {
        return;
    }
    loop {
        tokio::select! {
            frame = frames.recv() => {
                let text = match frame {
                    Ok(frame) => frame.for_viewer(unlocked).to_string(),
                    Err(broadcast::error::RecvError::Lagged(_)) => hub.snapshot_for(unlocked).to_string(),
                    Err(broadcast::error::RecvError::Closed) => break,
                };
                if socket.send(Message::Text(text)).await.is_err() { break; }
            }
            incoming = socket.recv() => match incoming {
                Some(Ok(Message::Close(_))) | Some(Err(_)) | None => break,
                Some(Ok(_)) => {}
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn page_is_self_contained() {
        assert!(PAGE.contains("/v1/console"));
        assert!(PAGE.contains("/assets/cuteafd-ui.css") && PAGE.contains("/assets/cuteafd-ui.js"));
        assert!(UI_JS.contains("window.CuteUI") && UI_JS.contains("/bench"));
        // The console must work on hosts without internet access.
        assert!(LOGO.starts_with("<svg") && MARK.starts_with("<svg") && !LOGO.contains("href="));
        for text in [PAGE, UI_CSS, UI_JS] {
            for external in ["http://", "https://"] {
                assert!(!text.contains(&format!("src=\"{external}")), "page loads an external script");
                assert!(!text.contains(&format!("href=\"{external}")), "page loads an external stylesheet");
                assert!(!text.contains(&format!("url({external}")), "page loads an external resource");
            }
        }
    }

    #[test]
    fn bench_text_toggles_text_and_a_disabled_hub_stays_off() {
        let hub = ConsoleHub::new(false);
        assert!(!hub.text_enabled());
        hub.set_bench_active(true);
        assert!(hub.text_enabled());
        hub.set_bench_active(false);
        assert!(!hub.text_enabled());
        // `--console-text` keeps text on whatever the bench does.
        let hub = ConsoleHub::new(true);
        hub.set_bench_active(true);
        hub.set_bench_active(false);
        assert!(hub.text_enabled());
        // A hub with no producer never streams text.
        let disabled = ConsoleHub::disabled();
        disabled.set_bench_active(true);
        assert!(!disabled.text_enabled());
    }

    #[tokio::test]
    async fn viewers_are_counted_only_while_connected() {
        let hub = ConsoleHub::new(false);
        assert_eq!(hub.viewers(), 0);
        let viewer = {
            hub.viewers.fetch_add(1, Ordering::Relaxed);
            Viewer(hub.clone(), false)
        };
        assert_eq!(hub.viewers(), 1);
        drop(viewer);
        assert_eq!(hub.viewers(), 0);
        let mut frames = hub.frames.subscribe();
        hub.publish("{\"type\":\"frame\"}".into());
        assert_eq!(&*frames.recv().await.unwrap().full, "{\"type\":\"frame\"}");
    }

    /// A locked SSE client never receives a text piece while an unlocked one does.
    #[tokio::test]
    async fn token_text_reaches_only_unlocked_viewers() {
        use futures::StreamExt;
        let f = tempfile::NamedTempFile::new().unwrap();
        std::fs::write(f.path(), "b".repeat(64)).unwrap();
        let gate = crate::console_gate::ConsoleGate::from_file(f.path(), false).unwrap();
        let hub = ConsoleHub::new(true);
        hub.set_gate(gate.clone());
        let app = gate.mount(axum::Router::new().route("/v1/console/events", axum::routing::get(events)).with_state(hub.clone()));
        use tower::ServiceExt;
        let unlock = app.clone().oneshot(axum::http::Request::get(format!("/console/unlock?token={}", "b".repeat(64)))
            .body(axum::body::Body::empty()).unwrap()).await.unwrap();
        let cookie = unlock.headers()["set-cookie"].to_str().unwrap().split(';').next().unwrap().to_owned();
        assert!(!hub.text_wanted(), "no viewer, no text work");
        let open = |cookie: Option<String>| {
            let app = app.clone();
            async move {
                let mut request = axum::http::Request::get("/v1/console/events");
                if let Some(c) = cookie { request = request.header("cookie", c); }
                app.oneshot(request.body(axum::body::Body::empty()).unwrap()).await.unwrap().into_body().into_data_stream()
            }
        };
        let mut locked = open(None).await;
        locked.next().await.unwrap().unwrap();
        assert_eq!(hub.text_access(&HeaderMap::new()), TextAccess::Locked);
        assert!(!hub.text_wanted(), "only locked viewers: the producer skips text");
        let mut unlocked = open(Some(cookie)).await;
        unlocked.next().await.unwrap().unwrap();
        assert!(hub.text_wanted());
        hub.publish_pair(r#"{"text":"SECRET"}"#.into(), Some(r#"{"text":null}"#.into()));
        let a = locked.next().await.unwrap().unwrap();
        let b = unlocked.next().await.unwrap().unwrap();
        assert!(!String::from_utf8_lossy(&a).contains("SECRET"));
        assert!(String::from_utf8_lossy(&b).contains("SECRET"));
        // A bench run makes its synthetic prompts public.
        hub.set_bench_active(true);
        assert_eq!(hub.text_access(&HeaderMap::new()), TextAccess::On);
    }
}
