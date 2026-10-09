//! Retained-row analytics. Percentiles are computed in Rust after indexed SQL filtering.
use crate::{store::{Error, Result}, Store};
use cuteafd_api::usage::Record;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::BTreeMap;
#[derive(Clone, Debug, Default, Deserialize, Serialize)]
pub struct Filter {
    pub range: Option<String>, pub from: Option<i64>, pub to: Option<i64>, pub client: Option<String>,
    pub protocol: Option<String>, pub model: Option<String>, pub key: Option<String>, pub session: Option<String>,
    pub bench: Option<bool>, pub bucket: Option<String>, pub split: Option<String>, pub cursor: Option<String>, pub limit: Option<usize>,
}
impl Filter {
    pub fn bounds(&self, now: i64) -> Result<(i64,i64)> {
        let duration = match self.range.as_deref().unwrap_or("24h") { "1h" => 3600000, "6h" => 21600000, "24h" => 86400000, "7d" => 604800000, _ => return Err(Error::Settings("invalid range")) };
        let to = self.to.unwrap_or(now); let from = self.from.unwrap_or(to.saturating_sub(duration));
        if from > to { return Err(Error::Settings("from must not exceed to")); } Ok((from,to))
    }
    fn bucket_ms(&self) -> Result<i64> { match self.bucket.as_deref().unwrap_or("1h") { "1m" => Ok(60000), "5m" => Ok(300000), "1h" => Ok(3600000), "1d" => Ok(86400000), _ => Err(Error::Settings("invalid bucket")) } }
}
pub fn percentile(values: impl Iterator<Item=f64>, p: f64) -> Option<f64> {
    let mut v = values.filter(|n| n.is_finite()).collect::<Vec<_>>(); if v.is_empty() { return None; }
    v.sort_by(f64::total_cmp); Some(v[((v.len()-1) as f64*p).round() as usize])
}
pub fn histogram(values: impl Iterator<Item=f64>) -> [u64;24] {
    let mut bins=[0;24]; for v in values.filter(|v| v.is_finite() && *v>=0.) { let i=if v<=1. {0} else {v.log2().floor() as usize}; bins[i.min(23)]+=1; } bins
}
fn latency(rows: &[Record]) -> Value {
    let metric=|get:fn(&Record)->Option<f64>| json!({"p50":percentile(rows.iter().filter_map(get),0.5),"p95":percentile(rows.iter().filter_map(get),0.95),"p99":percentile(rows.iter().filter_map(get),0.99),"histogram":histogram(rows.iter().filter_map(get))});
    json!({"ttft":metric(|r|r.t_ttft_ms),"decode":metric(|r|r.decode_tps),"prefill":metric(|r|r.prefill_tps)})
}
fn sum(rows:&[Record],get:fn(&Record)->Option<u64>)->u64 { rows.iter().filter_map(get).sum() }
fn ratio(n:u64,d:u64)->Option<f64> { (d>0).then(||n as f64/d as f64) }
fn summary(rows:&[Record])->Value {
    let errors=rows.iter().filter(|r|r.outcome!="ok").count(); let input=sum(rows,|r|r.tokens_in); let cached=sum(rows,|r|r.tokens_cached);
    json!({"requests":rows.len(),"errors":errors,"error_rate":ratio(errors as u64,rows.len() as u64),"tokens_in":input,"tokens_cached":cached,"tokens_out":sum(rows,|r|r.tokens_out),
        "cache_hit":ratio(cached,input),"acceptance":ratio(sum(rows,|r|r.draft_accepted),sum(rows,|r|r.draft_proposed)),"latency":latency(rows),
        "concurrency_max":rows.iter().filter_map(|r|r.concurrency_engine).max(),"concurrency_mean":ratio(sum(rows,|r|r.concurrency_engine),rows.iter().filter(|r|r.concurrency_engine.is_some()).count() as u64)})
}
fn group(rows:&[Record],key:impl Fn(&Record)->String)->BTreeMap<String,Vec<Record>> { let mut groups=BTreeMap::new(); for r in rows { groups.entry(key(r)).or_insert_with(Vec::new).push(r.clone()); } groups }
fn series(rows:&[Record],f:&Filter)->Result<Value> {
    let bucket=f.bucket_ms()?; let split=f.split.as_deref().unwrap_or("protocol");
    if !matches!(split,"protocol"|"client"|"model") { return Err(Error::Settings("invalid split")); }
    let groups=group(rows,|r|format!("{}|{}",r.ts_ms.div_euclid(bucket)*bucket,match split {"client"=>r.client_kind.as_str(),"model"=>r.model_served.as_deref().unwrap_or(""),_=>r.protocol.as_str()}));
    Ok(json!(groups.into_iter().map(|(key,rows)|{let (ts,split)=key.split_once('|').unwrap();let mut v=summary(&rows);v["ts_ms"]=json!(ts.parse::<i64>().unwrap());v["split"]=json!(split);v}).collect::<Vec<_>>()))
}
impl Store {
    pub fn query_rows(&self,f:&Filter)->Result<Vec<Record>> {
        use rusqlite::types::Value as S;
        let (from,to)=f.bounds(self.clock.now_ms())?;
        let mut sql="SELECT * FROM requests WHERE ts_ms>=? AND ts_ms<=?".to_string(); let mut args=vec![S::Integer(from),S::Integer(to)];
        for (column,value) in [("client_kind",&f.client),("protocol",&f.protocol),("model_served",&f.model),("key_label",&f.key),("session_id",&f.session)] {
            if let Some(value)=value {sql.push_str(&format!(" AND {column}=?"));args.push(S::Text(value.clone()));}
        }
        if !f.bench.unwrap_or(false) {sql.push_str(" AND bench=0");}
        if let Some(cursor)=&f.cursor {
            let (ts,rid)=cursor.split_once(':').ok_or(Error::Settings("invalid cursor"))?;
            let ts=ts.parse::<i64>().map_err(|_|Error::Settings("invalid cursor"))?;
            sql.push_str(" AND (ts_ms<? OR (ts_ms=? AND rid<?))");args.extend([S::Integer(ts),S::Integer(ts),S::Text(rid.into())]);
        }
        if let Some(limit)=f.limit {sql.push_str(" ORDER BY ts_ms DESC,rid DESC LIMIT ?");args.push(S::Integer(limit.clamp(1,500) as i64+1));}
        else {sql.push_str(" ORDER BY ts_ms,rid");}
        crate::store::read_records_query(&*self.reader.lock().map_err(|_|Error::Stopped)?,&sql,args)
    }
    pub fn query(&self,kind:&str,f:&Filter,id:Option<&str>)->Result<Value> {
        if kind=="log" {return Ok(self.log.get(id.unwrap_or(""))?.unwrap_or(Value::Null));}
        let mut f=f.clone();
        if kind=="sessions" {if let Some(id)=id {f.session=Some(id.into());}}
        if kind=="requests" && id.is_none(){f.limit=Some(f.limit.unwrap_or(100).clamp(1,500));}
        if id.is_some() && f.from.is_none() && f.range.is_none(){f.range=Some("7d".into());}
        let rows=self.query_rows(&f)?;
        match kind {
            "summary"=>{let mut v=summary(&rows);v["series"]=series(&rows,&f)?;v["usage"]=self.counters.snapshot();Ok(v)},
            "series"=>series(&rows,&f),
            "latency"=>{
                let mut v=latency(&rows);
                v["prompt_facets"]=json!(group(&rows,|r|match r.tokens_in.unwrap_or(0) {0..=1023=>"<1K",1024..=8191=>"1-8K",8192..=32767=>"8-32K",_=>"32K+"}.into()).into_iter().map(|(key,rows)|json!({"prompt":key,"latency":latency(&rows)})).collect::<Vec<_>>());
                v["cache_facets"]=json!(cache_facets(&rows)); Ok(v)
            },
            "flow"=>Ok(json!(group(&rows,|r|format!("{}|{}|{}",r.client_kind,r.protocol,r.model_served.as_deref().unwrap_or(""))).into_values().map(|rows|json!({"client":rows[0].client_kind,"protocol":rows[0].protocol,"model":rows[0].model_served,"requests":rows.len(),"tokens":sum(&rows,|r|r.tokens_in)+sum(&rows,|r|r.tokens_out)})).collect::<Vec<_>>())),
            "sessions" if id.is_some()=>Ok(json!({"session_id":id,"turns":rows})),
            "sessions"=>{
                let mut sessions=group(&rows.iter().filter(|r|r.session_id.is_some()).cloned().collect::<Vec<_>>(),|r|r.session_id.clone().unwrap()).into_iter().map(|(id,rows)|json!({"session_id":id,"source":rows[0].session_source,"client":rows[0].client_kind,"model":rows[0].model_served,"first_ms":rows.first().map(|r|r.ts_ms),"last_ms":rows.last().map(|r|r.ts_ms),"summary":summary(&rows),"turns":rows.iter().map(|r|json!({"rid":r.rid,"ts_ms":r.ts_ms,"tokens_in":r.tokens_in,"tokens_cached":r.tokens_cached,"tokens_out":r.tokens_out})).collect::<Vec<_>>()})).collect::<Vec<_>>();
                sessions.sort_by_key(|v|std::cmp::Reverse(v["last_ms"].as_i64())); Ok(json!(sessions))
            },
            "cache"=>Ok(json!({"series":series(&rows,&f)?,"facets":cache_facets(&rows),"tokens_saved":sum(&rows,|r|r.tokens_cached),"hit_rate":ratio(sum(&rows,|r|r.tokens_cached),sum(&rows,|r|r.tokens_in))})),
            "speculation"=>Ok(json!({"series":series(&rows,&f)?,"by_model":group(&rows,|r|r.model_served.clone().unwrap_or_default()).into_iter().map(|(name,r)|json!({"model":name,"summary":summary(&r),"tokens_per_round":ratio(sum(&r,|r|r.tokens_out),sum(&r,|r|r.rounds)),"histogram":histogram(r.iter().filter_map(|r|ratio(r.tokens_out?,r.rounds?)))})).collect::<Vec<_>>(),"by_client":group(&rows,|r|r.client_kind.clone()).into_iter().map(|(name,r)|json!({"client":name,"summary":summary(&r)})).collect::<Vec<_>>()})),
            "errors"=>Ok(json!({"series":series(&rows.iter().filter(|r|r.outcome!="ok").cloned().collect::<Vec<_>>(),&f)?,"classes":group(&rows.iter().filter(|r|r.outcome!="ok").cloned().collect::<Vec<_>>(),|r|format!("{}|{}|{}",r.error_class.as_deref().unwrap_or(&r.outcome),r.route,r.client_kind)).into_iter().map(|(class,r)|json!({"class_route_client":class,"count":r.len(),"last_ms":r.last().map(|r|r.ts_ms)})).collect::<Vec<_>>(),"stops":group(&rows,|r|r.stop_reason.clone().unwrap_or_default()).into_iter().map(|(reason,r)|json!({"reason":reason,"count":r.len()})).collect::<Vec<_>>()})),
            "requests" if id.is_some()=>Ok(rows.iter().find(|r|Some(r.rid.as_str())==id).map(serde_json::to_value).transpose()?.unwrap_or(Value::Null)),
            "requests"=>{
                let cursor=f.cursor.as_ref().map(|s|{let (ts,rid)=s.split_once(':').ok_or(Error::Settings("invalid cursor"))?;Ok::<_,Error>((ts.parse::<i64>().map_err(|_|Error::Settings("invalid cursor"))?,rid.to_string()))}).transpose()?;
                let limit=f.limit.unwrap_or(100).clamp(1,500);
                let mut page=rows.into_iter().filter(|r|cursor.as_ref().is_none_or(|(ts,id)|(r.ts_ms,r.rid.as_str())<(*ts,id.as_str()))).take(limit+1).collect::<Vec<_>>();
                let more=page.len()>limit;page.truncate(limit);let next=more.then(||page.last().map(|r|format!("{}:{}",r.ts_ms,r.rid))).flatten();Ok(json!({"requests":page,"next_cursor":next}))
            },
            _=>Err(Error::Settings("unknown usage query")),
        }
    }
}
fn cache_facets(rows:&[Record])->Vec<Value> {
    group(rows,|r|{let fraction=ratio(r.tokens_cached.unwrap_or(0),r.tokens_in.unwrap_or(0)).unwrap_or(0.);if fraction==0. {"0"} else if fraction<0.5 {"<50%"} else {">=50%"}.into()}).into_iter().map(|(key,rows)|json!({"fraction":key,"requests":rows.len(),"latency":latency(&rows)})).collect()
}
#[cfg(test)]
mod tests {
    use super::*;use cuteafd_api::usage::UsageSink;use std::sync::Arc;
    struct Fixed;impl crate::Clock for Fixed {fn now_ms(&self)->i64{7*86400000}}
    #[test]
    fn synthetic_seven_days_percentiles_buckets_sessions_and_cursor() {
        let store=Store::open_with(None,Arc::new(Fixed),4096).unwrap();
        for i in 0..700 {store.record(Record{rid:format!("r{i:04}"),ts_ms:i*864000,protocol:"chat".into(),client_kind:"codex".into(),session_id:Some(format!("s{}",i/100)),tokens_in:Some(100),tokens_cached:Some(50),tokens_out:Some(10),t_ttft_ms:Some(i as f64),decode_tps:Some(100.),outcome:"ok".into(),..Default::default()});}store.flush().unwrap();
        let f=Filter{range:Some("7d".into()),bucket:Some("1d".into()),..Default::default()};let summary=store.query("summary",&f,None).unwrap();assert_eq!(summary["requests"],700);assert_eq!(summary["tokens_in"],70000);assert_eq!(summary["latency"]["ttft"]["p50"],350.);assert_eq!(summary["latency"]["ttft"]["p95"],664.);
        assert_eq!(store.query("series",&f,None).unwrap().as_array().unwrap().len(),7);assert_eq!(store.query("sessions",&f,None).unwrap().as_array().unwrap().len(),7);
        let mut page=f.clone();page.limit=Some(30);let first=store.query("requests",&page,None).unwrap();page.cursor=first["next_cursor"].as_str().map(str::to_owned);let second=store.query("requests",&page,None).unwrap();assert_ne!(first["requests"][0]["rid"],second["requests"][0]["rid"]);
        for kind in ["flow","latency","cache","speculation","errors"] {assert!(!store.query(kind,&f,None).unwrap().is_null());}
    }
}
