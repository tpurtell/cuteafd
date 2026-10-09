//! All usage data and mutations require the console cookie, never an API key.
use crate::{query::Filter, store::{Error, Settings}, Store};
use axum::{extract::{Path, Query, State}, http::StatusCode, response::{IntoResponse, Response}, routing::{get, post}, Json, Router};
use cuteafd_api::console_gate::{require_console, ConsoleGate};
use serde_json::{json, Value};
use std::{collections::HashMap, sync::{Arc, Mutex}, time::{Duration, Instant}};
#[derive(Clone)]
struct Http { store: Arc<Store>, cache: Arc<Mutex<HashMap<String,(Instant,Value)>>> }
pub fn mount(router:Router, store:Arc<Store>, gate:ConsoleGate)->Router {
    let state=Http{store,cache:Arc::new(Mutex::new(HashMap::new()))};
    let routes=Router::new()
        .route("/console/usage/summary",get(summary)).route("/console/usage/series",get(series))
        .route("/console/usage/latency",get(latency)).route("/console/usage/flow",get(flow))
        .route("/console/usage/sessions",get(sessions)).route("/console/usage/sessions/:id",get(session))
        .route("/console/usage/cache",get(cache)).route("/console/usage/speculation",get(speculation))
        .route("/console/usage/errors",get(errors)).route("/console/usage/requests",get(requests))
        .route("/console/usage/requests/:rid",get(request)).route("/console/usage/log/:rid",get(log))
        .route("/console/usage/settings",get(settings).put(update_settings))
        .route("/console/usage/log/clear",post(clear_log)).route("/console/usage/clear",post(clear))
        .with_state(state).layer(axum::middleware::from_fn_with_state(gate,require_console));
    router.merge(routes)
}
fn error(e:Error)->Response {
    let status=if matches!(e,Error::Settings(_)){StatusCode::BAD_REQUEST}else{tracing::error!(error=%e,"usage query failed");StatusCode::INTERNAL_SERVER_ERROR};
    (status,Json(json!({"error":{"type":"usage_error","message":if status==StatusCode::BAD_REQUEST{e.to_string()}else{"usage storage unavailable".into()}}}))).into_response()
}
async fn query(state:Http,f:Filter,kind:&'static str,id:Option<String>)->Response {
    let key=format!("{kind}|{}|{}",id.as_deref().unwrap_or(""),serde_json::to_string(&f).expect("filter JSON"));
    if let Some((_,value))=state.cache.lock().unwrap_or_else(|e|e.into_inner()).get(&key).filter(|(at,_)|at.elapsed()<Duration::from_secs(5)).cloned(){return Json(value).into_response();}
    let store=state.store.clone();
    match tokio::task::spawn_blocking(move||store.query(kind,&f,id.as_deref())).await {
        Ok(Ok(v))=>{if v.is_null(){return(StatusCode::NOT_FOUND,Json(json!({"error":{"type":"not_retained"}}))).into_response();}
            let mut cache=state.cache.lock().unwrap_or_else(|e|e.into_inner());
            cache.retain(|_,(at,_)|at.elapsed()<Duration::from_secs(5));if cache.len()>=128{cache.clear();}cache.insert(key,(Instant::now(),v.clone()));Json(v).into_response()},
        Ok(Err(e))=>error(e),Err(_)=>error(Error::Stopped),
    }
}
macro_rules! list_route {($name:ident)=>{async fn $name(State(s):State<Http>,Query(f):Query<Filter>)->Response{query(s,f,stringify!($name),None).await}};}
list_route!(summary);list_route!(series);list_route!(latency);list_route!(flow);list_route!(sessions);list_route!(cache);list_route!(speculation);list_route!(errors);list_route!(requests);
async fn session(State(s):State<Http>,Path(id):Path<String>,Query(f):Query<Filter>)->Response{query(s,f,"sessions",Some(id)).await}
async fn request(State(s):State<Http>,Path(id):Path<String>,Query(f):Query<Filter>)->Response{query(s,f,"requests",Some(id)).await}
async fn log(State(s):State<Http>,Path(id):Path<String>)->Response{query(s,Filter::default(),"log",Some(id)).await}
async fn settings(State(s):State<Http>)->Json<Value>{Json(json!({"settings":s.store.settings(),"usage":s.store.counters.snapshot(),"warning":"With the full log on, user prompts and model outputs are stored in plain text for the retention period."}))}
async fn update_settings(State(s):State<Http>,Json(settings):Json<Settings>)->Response {
    let store=s.store.clone();match tokio::task::spawn_blocking(move||store.update_settings(settings)).await{Ok(Ok(()))=>{s.cache.lock().unwrap_or_else(|e|e.into_inner()).clear();Json(json!({"settings":s.store.settings()})).into_response()},Ok(Err(e))=>error(e),Err(_)=>error(Error::Stopped)}
}
async fn clear_log(State(s):State<Http>)->Response{clear_impl(s,false).await}
async fn clear(State(s):State<Http>)->Response{clear_impl(s,true).await}
async fn clear_impl(s:Http,all:bool)->Response {
    let store=s.store.clone();match tokio::task::spawn_blocking(move||{store.log.clear()?;if all{store.clear()?;}Ok::<_,Error>(())}).await{Ok(Ok(()))=>{s.cache.lock().unwrap_or_else(|e|e.into_inner()).clear();Json(json!({"cleared":true})).into_response()},Ok(Err(e))=>error(e),Err(_)=>error(Error::Stopped)}
}
#[cfg(test)]
mod tests {
    use super::*;use axum::{body::Body,http::Request};use tower::ServiceExt;
    #[tokio::test]
    async fn every_route_requires_cookie_before_extraction() {
        let app=mount(Router::new(),Store::open(None).unwrap(),ConsoleGate::locked());
        for (method,path) in [("GET","summary"),("GET","series"),("GET","latency"),("GET","flow"),("GET","sessions"),("GET","sessions/id"),("GET","cache"),("GET","speculation"),("GET","errors"),("GET","requests"),("GET","requests/rid"),("GET","log/rid"),("GET","settings"),("PUT","settings"),("POST","log/clear"),("POST","clear")] {
            let r=app.clone().oneshot(Request::builder().method(method).uri(format!("/console/usage/{path}")).header("Authorization","Bearer API_KEY").body(Body::from("malformed")).unwrap()).await.unwrap();assert_eq!(r.status(),StatusCode::UNAUTHORIZED,"{method} {path}");
            assert_eq!(serde_json::from_slice::<Value>(&axum::body::to_bytes(r.into_body(),4096).await.unwrap()).unwrap()["error"]["type"],"console_locked");
        }
    }
    #[tokio::test]
    async fn unlocked_settings_and_clear_invalidate_cache() {
        let f=tempfile::NamedTempFile::new().unwrap();std::fs::write(f.path(),"a".repeat(64)).unwrap();let gate=ConsoleGate::from_file(f.path(),false).unwrap();let store=Store::open(None).unwrap();
        let app=gate.mount(mount(Router::new(),store,gate.clone()));
        let r=app.clone().oneshot(Request::get(format!("/console/unlock?token={}","a".repeat(64))).body(Body::empty()).unwrap()).await.unwrap();let cookie=r.headers()["set-cookie"].to_str().unwrap().split(';').next().unwrap();
        for path in ["settings","summary"]{assert_eq!(app.clone().oneshot(Request::get(format!("/console/usage/{path}")).header("cookie",cookie).body(Body::empty()).unwrap()).await.unwrap().status(),StatusCode::OK);}
        assert_eq!(app.oneshot(Request::post("/console/usage/clear").header("cookie",cookie).body(Body::empty()).unwrap()).await.unwrap().status(),StatusCode::OK);
    }
}
