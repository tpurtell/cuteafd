//! Codex rust-v0.161.0's provider-relative `alpha/search` protocol.
use std::{
    collections::{HashMap, VecDeque},
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc, Mutex,
    },
    time::{Duration, Instant},
};

use super::super::{
    record::Tape,
    search::{SearchHit, SearchQuery},
    Gateway, GatewayError,
};
use axum::{
    extract::{rejection::JsonRejection, State},
    response::{IntoResponse, Response},
    Json,
};
use serde::Deserialize;
use serde_json::{json, Value};

const MAX_SESSIONS: usize = 64;
const MAX_REFS: usize = 128;
const MAX_PAGE_CHARS: usize = 32768;
const MAX_SESSION_BYTES: usize = 1024 * 1024;
const SESSION_TTL: Duration = Duration::from_secs(3600);

#[derive(Default)]
pub struct SearchCache {
    sessions: Mutex<HashMap<String, (Instant, Arc<tokio::sync::Mutex<Session>>)>>,
    searches: AtomicU64,
}
impl SearchCache {
    pub fn search_requests(&self) -> u64 {
        self.searches.load(Ordering::Relaxed)
    }
    fn session(&self, id: &str) -> Result<Arc<tokio::sync::Mutex<Session>>, GatewayError> {
        let mut sessions = self.sessions.lock().unwrap_or_else(|e| e.into_inner());
        sessions.retain(|_, (last, session)| {
            last.elapsed() < SESSION_TTL || Arc::strong_count(session) > 1
        });
        if !sessions.contains_key(id) && sessions.len() >= MAX_SESSIONS {
            if let Some(oldest) = sessions
                .iter()
                .filter(|(_, (_, session))| Arc::strong_count(session) == 1)
                .min_by_key(|(_, (last, _))| *last)
                .map(|(id, _)| id.clone())
            {
                sessions.remove(&oldest);
            } else {
                return Err(GatewayError::new(
                    super::super::ErrorKind::Overloaded,
                    "standalone search session cache is busy",
                ));
            }
        }
        let (last, session) = sessions.entry(id.into()).or_insert_with(|| {
            (
                Instant::now(),
                Arc::new(tokio::sync::Mutex::new(Session::default())),
            )
        });
        *last = Instant::now();
        Ok(session.clone())
    }
}
#[derive(Default)]
struct Session {
    turn: u64,
    refs: HashMap<String, Page>,
    order: VecDeque<String>,
}
#[derive(Clone)]
struct Page {
    hit: SearchHit,
    opened: bool,
}
impl Session {
    fn insert(&mut self, reference: String, hit: SearchHit, opened: bool) {
        if !self.refs.contains_key(&reference) {
            if self.order.len() >= MAX_REFS {
                if let Some(old) = self.order.pop_front() {
                    self.refs.remove(&old);
                }
            }
            self.order.push_back(reference.clone());
        }
        self.refs.insert(reference, Page { hit, opened });
        while self
            .refs
            .values()
            .map(|p| p.hit.url.len() + p.hit.title.len() + p.hit.content.len())
            .sum::<usize>()
            > MAX_SESSION_BYTES
        {
            if let Some(old) = self.order.pop_front() {
                self.refs.remove(&old);
            } else {
                break;
            }
        }
    }
    fn page(&self, reference: &str) -> Option<Page> {
        let url = self
            .refs
            .get(reference)
            .map(|page| page.hit.url.as_str())
            .unwrap_or(reference);
        self.refs
            .values()
            .find(|page| page.opened && page.hit.url == url)
            .or_else(|| self.refs.get(reference))
            .or_else(|| self.refs.values().find(|page| page.hit.url == url))
            .cloned()
    }
}

#[derive(Deserialize)]
pub(super) struct Request {
    id: String,
    model: String,
    #[serde(default)]
    commands: Commands,
    #[serde(default)]
    settings: Settings,
    max_output_tokens: Option<u64>,
    // Input and reasoning are context for model-based search planners, not sent to our provider.
    input: Option<Value>,
    reasoning: Option<Value>,
}
#[derive(Default, Deserialize)]
struct Commands {
    #[serde(default)]
    search_query: Vec<Query>,
    #[serde(default)]
    open: Vec<Open>,
    #[serde(default)]
    find: Vec<Find>,
    #[serde(default)]
    time: Vec<Time>,
    #[serde(default)]
    image_query: Vec<Value>,
    #[serde(default)]
    click: Vec<Value>,
    #[serde(default)]
    screenshot: Vec<Value>,
    #[serde(default)]
    finance: Vec<Value>,
    #[serde(default)]
    weather: Vec<Value>,
    #[serde(default)]
    sports: Vec<Value>,
    response_length: Option<String>,
}
#[derive(Deserialize)]
struct Query {
    q: String,
    recency: Option<u64>,
    domains: Option<Vec<String>>,
}
#[derive(Deserialize)]
struct Open {
    ref_id: String,
    lineno: Option<u64>,
}
#[derive(Deserialize)]
struct Find {
    ref_id: String,
    pattern: String,
}
#[derive(Deserialize)]
struct Time {
    utc_offset: String,
}
#[derive(Default, Deserialize)]
struct Settings {
    #[serde(default)]
    filters: Filters,
    external_web_access: Option<Value>,
    search_context_size: Option<String>,
}
#[derive(Default, Deserialize)]
struct Filters {
    #[serde(default)]
    allowed_domains: Vec<String>,
    #[serde(default)]
    blocked_domains: Vec<String>,
}

pub(super) async fn create(
    State(gateway): State<Arc<Gateway>>,
    tape: Tape,
    request: Result<Json<Request>, JsonRejection>,
) -> Response {
    let request = match request {
        Ok(Json(request)) => request,
        Err(_) => {
            return GatewayError::invalid("invalid standalone search request").openai_response()
        }
    };
    match execute(&gateway, request, tape).await {
        Ok(value) => Json(value).into_response(),
        Err(error) => error.openai_response(),
    }
}
fn domain_match(host: &str, domain: &str) -> bool {
    let domain = domain.trim().trim_start_matches("*.").to_ascii_lowercase();
    host == domain || host.ends_with(&format!(".{domain}"))
}
fn allowed(url: &str, domains: &[String], filters: &Filters) -> bool {
    let Ok(url) = reqwest::Url::parse(url) else {
        return false;
    };
    let Some(host) = url.host_str() else {
        return false;
    };
    matches!(url.scheme(), "https" | "http")
        && url.username().is_empty()
        && url.password().is_none()
        && (domains.is_empty() || domains.iter().any(|d| domain_match(host, d)))
        && (filters.allowed_domains.is_empty()
            || filters
                .allowed_domains
                .iter()
                .any(|d| domain_match(host, d)))
        && !filters
            .blocked_domains
            .iter()
            .any(|d| domain_match(host, d))
}
fn limit(s: &str, max: usize) -> String {
    s.chars().take(max).collect()
}
fn lines(page: &Page, start: usize, count: usize) -> String {
    page.hit
        .content
        .lines()
        .enumerate()
        .skip(start)
        .take(count)
        .map(|(n, text)| format!("L{n}: {text}\n"))
        .collect()
}
fn time_at(offset: &str, now: time::OffsetDateTime) -> Result<String, GatewayError> {
    let b = offset.as_bytes();
    if b.len() != 6
        || !matches!(b[0], b'+' | b'-')
        || b[3] != b':'
        || ![b[1], b[2], b[4], b[5]].iter().all(u8::is_ascii_digit)
    {
        return Err(GatewayError::invalid("utc_offset must be +/-HH:MM"));
    }
    let hours = ((b[1] - b'0') * 10 + b[2] - b'0') as i32;
    let minutes = ((b[4] - b'0') * 10 + b[5] - b'0') as i32;
    if hours > 14 || minutes > 59 || (hours == 14 && minutes != 0) {
        return Err(GatewayError::invalid("utc_offset outside +/-14:00"));
    }
    let seconds = (hours * 3600 + minutes * 60) * if b[0] == b'-' { -1 } else { 1 };
    let zone = time::UtcOffset::from_whole_seconds(seconds)
        .map_err(|_| GatewayError::invalid("invalid utc_offset"))?;
    now.to_offset(zone)
        .format(&time::format_description::well_known::Rfc3339)
        .map_err(|_| GatewayError::internal("time formatting failed"))
}
fn validate(r: &Request) -> Result<(), GatewayError> {
    if r.id.trim().is_empty()
        || r.id.len() > 256
        || r.model.trim().is_empty()
        || r.model.len() > 256
    {
        return Err(GatewayError::invalid(
            "id and model must be nonempty and at most 256 bytes",
        ));
    }
    let c = &r.commands;
    let count = c.search_query.len()
        + c.open.len()
        + c.find.len()
        + c.time.len()
        + c.image_query.len()
        + c.click.len()
        + c.screenshot.len()
        + c.finance.len()
        + c.weather.len()
        + c.sports.len();
    if count == 0 || count > 16 {
        return Err(GatewayError::invalid(
            "commands must contain 1..16 operations",
        ));
    }
    if r.max_output_tokens == Some(0) {
        return Err(GatewayError::invalid("max_output_tokens must be positive"));
    }
    if c.response_length
        .as_deref()
        .is_some_and(|s| !matches!(s, "short" | "medium" | "long"))
        || r.settings
            .search_context_size
            .as_deref()
            .is_some_and(|s| !matches!(s, "low" | "medium" | "high"))
    {
        return Err(GatewayError::invalid(
            "invalid search response/context size",
        ));
    }
    for q in &c.search_query {
        if q.q.trim().is_empty() || q.q.len() > 8192 || q.recency.is_some_and(|days| days > 36500) {
            return Err(GatewayError::invalid(
                "query empty/too long or recency exceeds 36500 days",
            ));
        }
        validate_domains(q.domains.as_deref().unwrap_or_default())?;
    }
    validate_domains(&r.settings.filters.allowed_domains)?;
    validate_domains(&r.settings.filters.blocked_domains)?;
    for reference in c
        .open
        .iter()
        .map(|o| &o.ref_id)
        .chain(c.find.iter().map(|f| &f.ref_id))
    {
        if reference.is_empty() || reference.len() > 8192 {
            return Err(GatewayError::invalid("invalid ref_id"));
        }
    }
    for find in &c.find {
        if find.pattern.is_empty() || find.pattern.len() > 4096 {
            return Err(GatewayError::invalid(
                "find pattern must be nonempty and at most 4096 bytes",
            ));
        }
    }
    for t in &c.time {
        time_at(&t.utc_offset, time::OffsetDateTime::UNIX_EPOCH)?;
    }
    Ok(())
}
fn validate_domains(domains: &[String]) -> Result<(), GatewayError> {
    if domains.len() > 64
        || domains
            .iter()
            .any(|d| d.is_empty() || d.len() > 253 || d.contains(['/', ':', ' ', '\n']))
    {
        return Err(GatewayError::invalid("invalid domain filters"));
    }
    Ok(())
}
async fn execute(gateway: &Gateway, request: Request, tape: Tape) -> Result<Value, GatewayError> {
    validate(&request)?;
    let provider = gateway.search.as_ref();
    if !request.commands.search_query.is_empty() && provider.is_none() {
        return Err(GatewayError::unsupported(
            "web search is not configured on this server",
        ));
    }
    let session = gateway.standalone_search.session(&request.id)?;
    // Serialize a session's operations so concurrent batches cannot reuse reference IDs.
    let mut session = session.lock().await;
    let turn = session.turn;
    session.turn = session.turn.saturating_add(1);
    let mut output = String::new();
    let mut results = Vec::new();
    let filters = &request.settings.filters;
    if request
        .settings
        .external_web_access
        .as_ref()
        .is_some_and(|v| v == &json!(false) || v == "cached")
    {
        output.push_str("Cached mode requested: searches use the configured provider's index; cache-only access is not guaranteed.\n");
    }
    let max_results = match request.settings.search_context_size.as_deref() {
        Some("low") => 3,
        Some("high") => 10,
        _ => 5,
    };
    let line_count = match request.commands.response_length.as_deref() {
        Some("short") => 20,
        Some("long") => 120,
        _ => 60,
    };
    let mut index = 0;
    for query in request.commands.search_query {
        let domains = query.domains.unwrap_or_default();
        // Intersect both allowlists before dispatch; retain the narrower subdomain.
        let allowed_domains = if domains.is_empty() {
            filters.allowed_domains.clone()
        } else if filters.allowed_domains.is_empty() {
            domains.clone()
        } else {
            domains
                .iter()
                .flat_map(|query_domain| {
                    filters
                        .allowed_domains
                        .iter()
                        .filter_map(move |setting_domain| {
                            if domain_match(query_domain, setting_domain) {
                                Some(query_domain.clone())
                            } else if domain_match(setting_domain, query_domain) {
                                Some(setting_domain.clone())
                            } else {
                                None
                            }
                        })
                })
                .collect::<Vec<_>>()
        };
        if !domains.is_empty() && !filters.allowed_domains.is_empty() && allowed_domains.is_empty()
        {
            output.push_str(&format!(
                "Search: {}\nNo results: domain allowlists do not overlap.\n",
                query.q
            ));
            continue;
        }
        gateway
            .standalone_search
            .searches
            .fetch_add(1, Ordering::Relaxed);
        let hits = provider
            .unwrap()
            .search_recent_with_tape(
                SearchQuery {
                    query: query.q.clone(),
                    allowed_domains,
                    blocked_domains: filters.blocked_domains.clone(),
                    max_results,
                },
                query.recency,
                tape.clone(),
            )
            .await?;
        output.push_str(&format!("Search: {}\n", query.q));
        let mut found = false;
        for mut hit in hits
            .into_iter()
            .filter(|hit| allowed(&hit.url, &domains, filters))
            .take(max_results)
        {
            found = true;
            hit.url = limit(&hit.url, 8192);
            hit.title = limit(&hit.title, 512);
            hit.content = limit(&hit.content, 4096);
            let reference = format!("turn{turn}search{index}");
            index += 1;
            output.push_str(&format!(
                "[{reference}] {}\n{}\n{}\n\n",
                hit.title, hit.url, hit.content
            ));
            results.push(json!({"type":"text_result","ref_id":reference,"url":hit.url,"title":hit.title,"snippet":hit.content}));
            session.insert(reference, hit, false);
        }
        if !found {
            output.push_str("No results.\n");
        }
    }
    for open in request.commands.open {
        let prior = session.page(&open.ref_id);
        let url = prior
            .as_ref()
            .map(|p| p.hit.url.clone())
            .unwrap_or_else(|| open.ref_id.clone());
        if !allowed(&url, &[], filters) {
            output.push_str(&format!(
                "Open {}: unknown reference or URL excluded by filters.\n",
                open.ref_id
            ));
            continue;
        }
        let page = if let Some(page) = prior.filter(|p| p.opened) {
            page
        } else {
            let Some(provider) = provider else {
                output.push_str("Open not supported: no search provider configured.\n");
                continue;
            };
            match provider.fetch_with_tape(url.clone(), tape.clone()).await {
                Ok(mut hit) => {
                    hit.url = url.clone();
                    hit.title = limit(&hit.title, 512);
                    hit.content = limit(&hit.content, MAX_PAGE_CHARS);
                    Page { hit, opened: true }
                }
                Err(_) => {
                    output.push_str(&format!(
                        "Open {}: open not supported or page contents unavailable.\n",
                        open.ref_id
                    ));
                    continue;
                }
            }
        };
        let reference = if session.refs.contains_key(&open.ref_id) {
            open.ref_id.clone()
        } else {
            let reference = format!("turn{turn}open{index}");
            index += 1;
            reference
        };
        let start = usize::try_from(open.lineno.unwrap_or(0)).unwrap_or(usize::MAX);
        output.push_str(&format!(
            "[{reference}] {}\n{}\n{}",
            page.hit.title,
            url,
            lines(&page, start, line_count)
        ));
        results.push(
            json!({"type":"text_result","ref_id":reference,"url":url,"title":page.hit.title}),
        );
        session.insert(reference, page.hit, true);
    }
    for find in request.commands.find {
        output.push_str(&format!("Find {}: {:?}\n", find.ref_id, find.pattern));
        let Some(page) = session.page(&find.ref_id).filter(|p| p.opened) else {
            output.push_str("Page not opened or reference expired; open it first.\n");
            continue;
        };
        if !allowed(&page.hit.url, &[], filters) {
            output.push_str("URL excluded by filters.\n");
            continue;
        }
        let positions: Vec<usize> = page
            .hit
            .content
            .lines()
            .enumerate()
            .filter_map(|(n, line)| line.contains(&find.pattern).then_some(n))
            .take(10)
            .collect();
        if positions.is_empty() {
            output.push_str("No matches.\n");
        }
        for position in positions {
            output.push_str(&lines(&page, position.saturating_sub(2), 5));
        }
    }
    for command in request.commands.time {
        output.push_str(&format!(
            "Time {}: {}\n",
            command.utc_offset,
            time_at(&command.utc_offset, time::OffsetDateTime::now_utc())?
        ));
    }
    for (name, count) in [
        ("image_query", request.commands.image_query.len()),
        ("click", request.commands.click.len()),
        ("screenshot", request.commands.screenshot.len()),
        ("finance", request.commands.finance.len()),
        ("weather", request.commands.weather.len()),
        ("sports", request.commands.sports.len()),
    ] {
        if count > 0 {
            output.push_str(&format!("{name} not supported by this gateway.\n"));
        }
    }
    let budget = request.max_output_tokens.unwrap_or(16000).min(16000) as usize;
    // UTF-8 byte bound is conservative for byte-based tokenizers, including non-ASCII text.
    let mut end = output.len().min(budget);
    while !output.is_char_boundary(end) {
        end -= 1;
    }
    output.truncate(end);
    tape.record("standalone_search", || json!({"id":request.id,"model":request.model,"search_requests":gateway.standalone_search.search_requests(),"results":results.len()}));
    let _ = (request.input, request.reasoning);
    Ok(json!({"output":output,"results":results}))
}

#[cfg(test)]
mod tests;
