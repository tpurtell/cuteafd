use std::time::Duration;
use futures::future::BoxFuture;
use serde_json::{json, Value};
use super::{SearchHit, SearchProvider, SearchQuery};
use crate::gateway::{record::Tape, upstream::{bounded_body, http_error}, GatewayError};

fn client() -> Result<reqwest::Client, GatewayError> {
    reqwest::Client::builder().redirect(reqwest::redirect::Policy::none()).timeout(Duration::from_secs(30))
        .build().map_err(|_| GatewayError::internal("could not create search client"))
}
fn endpoint(url: &str, suffix: &str) -> Result<String, GatewayError> {
    let parsed = reqwest::Url::parse(url).map_err(|_| GatewayError::invalid("invalid search URL"))?;
    if !matches!(parsed.scheme(), "https" | "http") || !parsed.username().is_empty() || parsed.password().is_some() || parsed.query().is_some() || parsed.fragment().is_some() {
        return Err(GatewayError::invalid("search URL must be HTTP(S) without credentials, query or fragment"));
    }
    Ok(format!("{}/{suffix}",url.trim_end_matches('/')))
}
#[derive(Clone)]
pub struct Exa { client: reqwest::Client, key: String, endpoint: String }
impl Exa {
    pub fn new(key: String) -> Result<Self, GatewayError> { Self::with_url(key, "https://api.exa.ai") }
    pub fn with_url(key: String, url: &str) -> Result<Self, GatewayError> { Ok(Self { client:client()?,key,endpoint:endpoint(url,"search")? }) }
}
#[derive(Clone)]
pub struct Searxng { client: reqwest::Client, endpoint: String }
impl Searxng {
    pub fn new(url: &str) -> Result<Self, GatewayError> { Ok(Self { client:client()?,endpoint:endpoint(url,"search")? }) }
}
fn exa_request(query: &SearchQuery) -> Value {
    let mut request = json!({"query":query.query,"numResults":query.max_results.clamp(1,10),"contents":{"text":{"maxCharacters":4096}}});
    if !query.allowed_domains.is_empty() { request["includeDomains"] = json!(query.allowed_domains); }
    if !query.blocked_domains.is_empty() { request["excludeDomains"] = json!(query.blocked_domains); }
    request
}
fn searx_query(query: &SearchQuery) -> String {
    let mut text = query.query.clone();
    if !query.allowed_domains.is_empty() {
        text.push_str(" (");
        text.push_str(&query.allowed_domains.iter().map(|d| format!("site:{d}")).collect::<Vec<_>>().join(" OR "));
        text.push(')');
    }
    for domain in &query.blocked_domains { text.push_str(&format!(" -site:{domain}")); }
    text
}
fn domain_match(host: &str, domain: &str) -> bool {
    let domain = domain.trim().trim_start_matches("*.").to_ascii_lowercase();
    host == domain || host.ends_with(&format!(".{domain}"))
}
fn hits(value: &Value, query: &SearchQuery, exa: bool) -> Vec<SearchHit> {
    value["results"].as_array().into_iter().flatten().filter_map(|hit| {
        let url = hit["url"].as_str()?;
        let parsed = reqwest::Url::parse(url).ok()?;
        if !matches!(parsed.scheme(),"http" | "https") { return None; }
        let host = parsed.host_str()?.to_ascii_lowercase();
        if (!query.allowed_domains.is_empty() && !query.allowed_domains.iter().any(|d| domain_match(&host,d)))
            || query.blocked_domains.iter().any(|d| domain_match(&host,d)) { return None; }
        Some(SearchHit { url:url.chars().take(8192).collect(),title:hit["title"].as_str().unwrap_or_default().chars().take(512).collect(),
            content:hit[if exa { "text" } else { "content" }].as_str().unwrap_or_default().chars().take(4096).collect(),
            published:hit["publishedDate"].as_str().map(|s| s.chars().take(128).collect()) })
    }).take(query.max_results.clamp(1,10)).collect()
}
async fn response(response: reqwest::Response, query: &SearchQuery, tape: &Tape, provider: &str, request: Value, exa: bool) -> Result<Vec<SearchHit>,GatewayError> {
    let status = response.status().as_u16();
    let body = bounded_body(response, 2 * 1024 * 1024).await?;
    let parsed: Result<Value,_> = serde_json::from_slice(&body);
    let output = if (200..300).contains(&status) {
        parsed.as_ref().map(|v| hits(v,query,exa)).map_err(|_| GatewayError::upstream("invalid search response JSON"))
    } else { Err(http_error(status)) };
    tape.record("search", || json!({"provider":provider,"query":query,"request":request,"status":status,
        "body":String::from_utf8_lossy(&body),"hits":output.as_ref().ok()}));
    output
}
impl SearchProvider for Exa {
    fn name(&self) -> &str { "exa" }
    fn search(&self, query: SearchQuery) -> BoxFuture<'static, Result<Vec<SearchHit>,GatewayError>> { self.search_with_tape(query,Tape::default()) }
    fn search_with_tape(&self, query: SearchQuery, tape: Tape) -> BoxFuture<'static, Result<Vec<SearchHit>,GatewayError>> {
        let this = self.clone();
        Box::pin(async move {
            let request = exa_request(&query);
            let result = this.client.post(&this.endpoint).header("x-api-key",&this.key).json(&request).send().await
                .map_err(|_| GatewayError::upstream("search request transport failure"))?;
            response(result,&query,&tape,"exa",request,true).await
        })
    }
}
impl SearchProvider for Searxng {
    fn name(&self) -> &str { "searxng" }
    fn search(&self, query: SearchQuery) -> BoxFuture<'static, Result<Vec<SearchHit>,GatewayError>> { self.search_with_tape(query,Tape::default()) }
    fn search_with_tape(&self, query: SearchQuery, tape: Tape) -> BoxFuture<'static, Result<Vec<SearchHit>,GatewayError>> {
        let this = self.clone();
        Box::pin(async move {
            let text = searx_query(&query);
            let result = this.client.get(&this.endpoint).query(&[("q",&text),("format",&"json".to_string())]).send().await
                .map_err(|_| GatewayError::upstream("search request transport failure"))?;
            response(result,&query,&tape,"searxng",json!({"q":text,"format":"json"}),false).await
        })
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[tokio::test]
    #[ignore = "paid Exa verification; invented public query only"]
    async fn verify_exa_public_search() {
        let provider = Exa::new(std::env::var("EXA_API_KEY").expect("Exa key required")).unwrap();
        let hits = provider.search(SearchQuery { query:"volcano geology educational overview".into(),allowed_domains:vec![],blocked_domains:vec![],max_results:3 }).await.unwrap();
        assert!(!hits.is_empty());
        assert!(hits.iter().all(|hit| hit.content.chars().count() <= 4096));
        println!("Exa returned {} bounded hits",hits.len());
    }
    #[test]
    fn search_mapping_bounds_and_enforces_domain_filters() {
        let query = SearchQuery { query:"made up".into(),allowed_domains:vec!["example.org".into()],blocked_domains:vec!["bad.example.org".into()],max_results:100 };
        let request = exa_request(&query);
        assert_eq!(request["numResults"],10);
        assert_eq!(request["contents"]["text"]["maxCharacters"],4096);
        let results = hits(&json!({"results":[{"url":"https://example.org/a","title":"A","text":"x".repeat(8000)},
            {"url":"https://bad.example.org/a","text":"bad"},{"url":"https://notexample.org/a"},{"url":"file:///etc/passwd"}]}),&query,true);
        assert_eq!(results.len(),1);
        assert_eq!(results[0].content.len(),4096);
        assert!(searx_query(&query).contains("-site:bad.example.org"));
    }
}
