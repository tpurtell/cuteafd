use super::*;
use crate::{
    gateway::{
        auth::{require_key, GatewayAuth},
        testing::Scripted,
        ModelMap, SearchProvider,
    },
    openai::auth::ApiKey,
};
use axum::{
    body::{to_bytes, Body},
    http::{Request as HttpRequest, StatusCode},
    middleware, Router,
};
use futures::future::BoxFuture;
use tower::ServiceExt;

#[derive(Clone, Default)]
struct Fake {
    queries: Arc<Mutex<Vec<(SearchQuery, Option<u64>)>>>,
    fetches: Arc<AtomicU64>,
}
impl SearchProvider for Fake {
    fn name(&self) -> &str {
        "fake"
    }
    fn search(
        &self,
        query: SearchQuery,
    ) -> BoxFuture<'static, Result<Vec<SearchHit>, GatewayError>> {
        self.search_recent_with_tape(query, None, Tape::default())
    }
    fn search_recent_with_tape(
        &self,
        query: SearchQuery,
        recency: Option<u64>,
        _: Tape,
    ) -> BoxFuture<'static, Result<Vec<SearchHit>, GatewayError>> {
        self.queries.lock().unwrap().push((query, recency));
        Box::pin(async {
            Ok(vec![
                SearchHit {
                    url: "https://openai.com/result".into(),
                    title: "Public test page".into(),
                    content: "Small snippet".into(),
                    published: None,
                },
                SearchHit {
                    url: "https://blocked.openai.com/result".into(),
                    title: "Blocked".into(),
                    content: "Never included".into(),
                    published: None,
                },
                SearchHit {
                    url: "https://example.com/result".into(),
                    title: "Outside allowed list".into(),
                    content: "Never included".into(),
                    published: None,
                },
            ])
        })
    }
    fn fetch_with_tape(
        &self,
        url: String,
        _: Tape,
    ) -> BoxFuture<'static, Result<SearchHit, GatewayError>> {
        self.fetches.fetch_add(1, Ordering::Relaxed);
        Box::pin(async move {
            Ok(SearchHit {
                url,
                title: "Fetched page".into(),
                content: "zero\none\nfind needle here\nthree\nfour".into(),
                published: None,
            })
        })
    }
}
fn gateway(provider: Option<Arc<dyn SearchProvider>>) -> Arc<Gateway> {
    let mut gateway = Gateway::new(
        Arc::new(Scripted::default()),
        ModelMap::single("served-model"),
    );
    gateway.search = provider;
    gateway
        .standalone_search
        .next_turn
        .store(0, Ordering::Relaxed);
    Arc::new(gateway)
}
fn app(gateway: Arc<Gateway>) -> Router {
    crate::gateway::router(gateway).layer(middleware::from_fn_with_state(
        GatewayAuth {
            key: Some(ApiKey::new("local-test-key").unwrap()),
        },
        require_key,
    ))
}
async fn post(app: &Router, body: Value, key: bool) -> (StatusCode, Value) {
    let mut request = HttpRequest::builder()
        .method("POST")
        .uri("/v1/alpha/search")
        .header("content-type", "application/json");
    if key {
        request = request.header("authorization", "Bearer local-test-key");
    }
    let response = app
        .clone()
        .oneshot(request.body(Body::from(body.to_string())).unwrap())
        .await
        .unwrap();
    (
        response.status(),
        serde_json::from_slice(&to_bytes(response.into_body(), 1024 * 1024).await.unwrap())
            .unwrap(),
    )
}
fn request(commands: Value) -> Value {
    json!({"id":"search-session","model":"gpt-test","commands":commands})
}

#[tokio::test]
async fn search_posts_typed_request_and_parses_output() {
    let provider = Arc::new(Fake::default());
    let gateway = gateway(Some(provider.clone()));
    let app = app(gateway.clone());
    // Exact request JSON from Codex codex-api endpoint/search.rs rust-v0.161.0.
    let wire = json!({
        "id":"search-session","model":"gpt-test",
        "input":[{"type":"message","id":"msg_search","role":"user","content":[{"type":"input_text","text":"find this"},{"type":"input_image","image_url":"https://example.com/image.png"}]}],
        "commands":{"search_query":[{"q":"OpenAI news","recency":7,"domains":["openai.com"]}],"open":[{"ref_id":"https://openai.com","lineno":12}]},
        "settings":{"user_location":{"type":"approximate","country":"US","city":"San Francisco"},"search_context_size":"low","filters":{"allowed_domains":["openai.com"],"blocked_domains":["example.com"]},"image_settings":{"max_results":4,"caption":true},"allowed_callers":["direct"],"external_web_access":true},
        "max_output_tokens":2500
    });
    let (status, value) = post(&app, wire, true).await;
    assert_eq!(status, StatusCode::OK);
    assert!(value["output"].as_str().unwrap().contains("turn0search0"));
    assert_eq!(value["results"][0]["type"], "text_result");
    assert_eq!(value["results"][0]["url"], "https://openai.com/result");
    assert!(value.get("encrypted_output").is_none());
    let queries = provider.queries.lock().unwrap();
    assert_eq!(queries[0].0.query, "OpenAI news");
    assert_eq!(queries[0].0.max_results, 3);
    assert_eq!(queries[0].1, Some(7));
    assert_eq!(gateway.standalone_search.search_requests(), 1);
}

#[tokio::test]
async fn references_open_find_and_session_isolation() {
    let provider = Arc::new(Fake::default());
    let app = app(gateway(Some(provider.clone())));
    let (_, first) = post(
        &app,
        request(json!({"search_query":[{"q":"public fixture"}]})),
        true,
    )
    .await;
    let reference = first["results"][0]["ref_id"].as_str().unwrap();
    let (_,opened) = post(&app,request(json!({"open":[{"ref_id":reference,"lineno":1}],"find":[{"ref_id":reference,"pattern":"needle"}]})),true).await;
    assert!(opened["output"].as_str().unwrap().contains("L1: one"));
    assert!(opened["output"]
        .as_str()
        .unwrap()
        .contains("L2: find needle here"));
    post(&app, request(json!({"open":[{"ref_id":reference}]})), true).await;
    assert_eq!(
        provider.fetches.load(Ordering::Relaxed),
        1,
        "opened page reused"
    );
    let (_, next) = post(&app, request(json!({"search_query":[{"q":"next"}]})), true).await;
    assert_eq!(next["results"][0]["ref_id"], "turn3search0");
    let (_,unknown) = post(&app,json!({"id":"different","model":"gpt-test","commands":{"open":[{"ref_id":reference}],"find":[{"ref_id":reference,"pattern":"needle"}]}}),true).await;
    assert!(unknown["output"]
        .as_str()
        .unwrap()
        .contains("unknown reference"));
    assert!(unknown["output"]
        .as_str()
        .unwrap()
        .contains("open it first"));
}

#[tokio::test]
async fn filters_intersect_and_block_open_and_find() {
    let provider = Arc::new(Fake::default());
    let app = app(gateway(Some(provider.clone())));
    let mut wire = request(json!({"search_query":[{"q":"test","domains":["openai.com"]}]}));
    wire["settings"] = json!({"filters":{"allowed_domains":["openai.com"],"blocked_domains":["blocked.openai.com"]}});
    let (_, response) = post(&app, wire, true).await;
    assert_eq!(response["results"].as_array().unwrap().len(), 1);
    let mut wire = request(json!({"open":[{"ref_id":"turn0search0"}]}));
    wire["settings"] = json!({"filters":{"allowed_domains":["example.org"]}});
    let (_, response) = post(&app, wire, true).await;
    assert!(response["output"]
        .as_str()
        .unwrap()
        .contains("excluded by filters"));
    assert_eq!(provider.fetches.load(Ordering::Relaxed), 0);
    let mut wire = request(json!({"search_query":[{"q":"disjoint","domains":["openai.com"]}]}));
    wire["settings"] = json!({"filters":{"allowed_domains":["example.org"]}});
    let (_, response) = post(&app, wire, true).await;
    assert!(response["results"].as_array().unwrap().is_empty());
    assert_eq!(
        provider.queries.lock().unwrap().len(),
        1,
        "disjoint filters must not cause a provider request"
    );
}

#[tokio::test]
async fn unsupported_commands_and_time_are_tool_output() {
    let app = app(gateway(None));
    let (status,response) = post(&app,request(json!({"image_query":[{"q":"bird"}],"click":[{"ref_id":"x","id":1}],"screenshot":[{"ref_id":"x","pageno":0}],"finance":[{"ticker":"ABC","type":"equity"}],"weather":[{"location":"Example"}],"sports":[{"fn":"standings","league":"nba"}],"time":[{"utc_offset":"+05:30"}],"open":[{"ref_id":"https://example.org"}]})),true).await;
    assert_eq!(status, StatusCode::OK);
    let output = response["output"].as_str().unwrap();
    for name in [
        "image_query",
        "click",
        "screenshot",
        "finance",
        "weather",
        "sports",
    ] {
        assert!(output.contains(&format!("{name} not supported")));
    }
    assert!(output.contains("Time +05:30:"));
    assert!(output.contains("Open not supported"));
    assert_eq!(
        time_at("-03:30", time::OffsetDateTime::UNIX_EPOCH).unwrap(),
        "1969-12-31T20:30:00-03:30"
    );
}

#[tokio::test]
async fn auth_no_provider_and_invalid_requests_use_openai_errors() {
    let app = app(gateway(None));
    let search = request(json!({"search_query":[{"q":"test"}]}));
    let (status, error) = post(&app, search.clone(), false).await;
    assert_eq!(status, StatusCode::UNAUTHORIZED);
    assert_eq!(error["error"]["type"], "authentication_error");
    let (status, error) = post(&app, search, true).await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert!(error["error"]["message"]
        .as_str()
        .unwrap()
        .contains("not configured"));
    for wire in [
        json!({"commands":{}}),
        request(json!({})),
        request(json!({"time":[{"utc_offset":"+99:00"}]})),
        request(json!({"find":[{"ref_id":"x","pattern":""}]})),
        request(json!({"search_query":[{"q":""}]})),
    ] {
        let (status, error) = post(&app, wire, true).await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert!(error.get("error").is_some());
    }
}

#[tokio::test]
async fn output_budget_cached_note_and_cache_bounds() {
    let gateway = gateway(Some(Arc::new(Fake::default())));
    let app = app(gateway.clone());
    let mut wire = request(json!({"search_query":[{"q":"public"}]}));
    wire["settings"] = json!({"external_web_access":false});
    let (_, response) = post(&app, wire.clone(), true).await;
    assert!(response["output"]
        .as_str()
        .unwrap()
        .contains("cache-only access is not guaranteed"));
    wire["max_output_tokens"] = json!(10);
    let (_, response) = post(&app, wire, true).await;
    assert_eq!(response["output"].as_str().unwrap().chars().count(), 10);
    let cache = SearchCache::default();
    for i in 0..MAX_SESSIONS + 5 {
        cache.session(&i.to_string()).unwrap();
    }
    assert_eq!(cache.sessions.lock().unwrap().len(), MAX_SESSIONS);
    let busy = SearchCache::default();
    let held: Vec<_> = (0..MAX_SESSIONS)
        .map(|i| busy.session(&i.to_string()).unwrap())
        .collect();
    assert!(
        busy.session("overflow").is_err(),
        "active sessions must not be evicted or duplicated"
    );
    drop(held);
    assert!(busy.session("overflow").is_ok());
    let mut session = Session::default();
    for i in 0..MAX_REFS + 5 {
        session.insert(
            i.to_string(),
            SearchHit {
                url: "https://example.org".into(),
                title: String::new(),
                content: String::new(),
                published: None,
            },
            false,
        );
    }
    assert_eq!(session.refs.len(), MAX_REFS);
    assert!(session.page("0").is_none());
    let mut session = Session::default();
    for i in 0..100 {
        session.insert(
            i.to_string(),
            SearchHit {
                url: format!("https://example.org/{i}"),
                title: String::new(),
                content: "x".repeat(MAX_PAGE_CHARS),
                published: None,
            },
            true,
        );
    }
    assert!(
        session
            .refs
            .values()
            .map(|p| p.hit.url.len() + p.hit.content.len())
            .sum::<usize>()
            <= MAX_SESSION_BYTES
    );
}

#[tokio::test]
async fn exa_contents_and_recency_wire_are_bounded() {
    let seen = Arc::new(Mutex::new(Vec::<Value>::new()));
    let search_seen = seen.clone();
    let contents_seen = seen.clone();
    let server = Router::new()
        .route("/search",axum::routing::post(move |Json(value): Json<Value>| { let seen = search_seen.clone(); async move { seen.lock().unwrap().push(value); Json(json!({"results":[{"url":"https://example.org","title":"Fixture","text":"short"}]})) } }))
        .route("/contents",axum::routing::post(move |Json(value): Json<Value>| { let seen = contents_seen.clone(); async move { seen.lock().unwrap().push(value); Json(json!({"results":[{"url":"https://example.org","title":"Fixture","text":"a".repeat(40000)}]})) } }));
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!("http://{}", listener.local_addr().unwrap());
    let task = tokio::spawn(async move { axum::serve(listener, server).await.unwrap() });
    let provider = crate::gateway::search::Exa::with_url("fake-test-key".into(), &url).unwrap();
    let hits = provider
        .search_recent_with_tape(
            SearchQuery {
                query: "public".into(),
                allowed_domains: vec![],
                blocked_domains: vec![],
                max_results: 3,
            },
            Some(7),
            Tape::default(),
        )
        .await
        .unwrap();
    assert_eq!(hits.len(), 1);
    let page = provider
        .fetch_with_tape("https://example.org".into(), Tape::default())
        .await
        .unwrap();
    assert_eq!(page.content.len(), 32768);
    let requests = seen.lock().unwrap();
    assert!(requests[0]["startPublishedDate"]
        .as_str()
        .unwrap()
        .ends_with('Z'));
    assert_eq!(requests[1]["ids"], json!(["https://example.org"]));
    assert_eq!(requests[1]["text"]["maxCharacters"], 32768);
    task.abort();
}

#[tokio::test]
async fn stale_references_do_not_alias_after_expiry_or_eviction() {
    let gateway = gateway(Some(Arc::new(Fake::default())));
    let app = app(gateway.clone());
    let (_, first) = post(&app, request(json!({"search_query":[{"q":"first"}]})), true).await;
    let old = first["results"][0]["ref_id"].as_str().unwrap();
    gateway
        .standalone_search
        .sessions
        .lock()
        .unwrap()
        .get_mut("search-session")
        .unwrap()
        .0 = Instant::now() - SESSION_TTL - Duration::from_secs(1);
    let (_, second) = post(
        &app,
        request(json!({"search_query":[{"q":"second"}]})),
        true,
    )
    .await;
    assert_ne!(second["results"][0]["ref_id"], old);
    let (_, response) = post(&app, request(json!({"open":[{"ref_id":old}]})), true).await;
    assert!(response["output"]
        .as_str()
        .unwrap()
        .contains("unknown reference"));
    let stale = second["results"][0]["ref_id"].as_str().unwrap();
    for i in 0..MAX_SESSIONS {
        gateway
            .standalone_search
            .session(&format!("other-{i}"))
            .unwrap();
    }
    let (_, third) = post(&app, request(json!({"search_query":[{"q":"third"}]})), true).await;
    assert_ne!(third["results"][0]["ref_id"], stale);
    let (_, response) = post(&app, request(json!({"open":[{"ref_id":stale}]})), true).await;
    assert!(response["output"]
        .as_str()
        .unwrap()
        .contains("unknown reference"));
    assert_ne!(
        SearchCache::default().next_turn.load(Ordering::Relaxed),
        SearchCache::default().next_turn.load(Ordering::Relaxed),
        "gateway generations should not reuse refs"
    );
}

#[tokio::test]
async fn cached_mode_never_fetches_unopened_pages() {
    let provider = Arc::new(Fake::default());
    let app = app(gateway(Some(provider.clone())));
    for mode in [json!(false), json!("cached")] {
        let mut wire = request(json!({"open":[{"ref_id":"https://example.org"}]}));
        wire["settings"] = json!({"external_web_access":mode});
        let (status, response) = post(&app, wire, true).await;
        assert_eq!(status, StatusCode::OK);
        assert!(response["output"]
            .as_str()
            .unwrap()
            .contains("external fetch disabled"));
    }
    assert_eq!(provider.fetches.load(Ordering::Relaxed), 0);
    post(
        &app,
        request(json!({"open":[{"ref_id":"https://example.org"}]})),
        true,
    )
    .await;
    let mut wire = request(
        json!({"open":[{"ref_id":"https://example.org"}],"find":[{"ref_id":"https://example.org","pattern":"needle"}]}),
    );
    wire["settings"] = json!({"external_web_access":false});
    let (_, response) = post(&app, wire, true).await;
    assert!(response["output"]
        .as_str()
        .unwrap()
        .contains("find needle here"));
    assert_eq!(
        provider.fetches.load(Ordering::Relaxed),
        1,
        "cached open reuses the already-opened page"
    );
}
