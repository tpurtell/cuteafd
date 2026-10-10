//! Retained-row analytics. Percentiles are computed in Rust after indexed SQL filtering.
use crate::{
    store::{Error, Result},
    Store,
};
use cuteafd_api::usage::Record;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::BTreeMap;
#[derive(Clone, Debug, Default, Deserialize, Serialize)]
pub struct Filter {
    pub range: Option<String>,
    pub from: Option<i64>,
    pub to: Option<i64>,
    pub client: Option<String>,
    pub protocol: Option<String>,
    pub model: Option<String>,
    pub key: Option<String>,
    pub session: Option<String>,
    pub bench: Option<bool>,
    pub bucket: Option<String>,
    pub split: Option<String>,
    pub cursor: Option<String>,
    pub limit: Option<usize>,
    /// Row predicates the charts set by clicking: exact matches...
    pub route: Option<String>,
    pub outcome: Option<String>,
    pub stop: Option<String>,
    /// ...the error class, or the outcome for rows without one (the errors table key)...
    pub error: Option<String>,
    /// ...prompt-length bucket (`<1K`, `1-8K`, `8-32K`, `32K+`) and cached fraction (`0`, `<50%`, `>=50%`)...
    pub prompt: Option<String>,
    pub cached: Option<String>,
    /// ...and a half-open `[min,max)` range over one metric
    /// (`t_ttft_ms`, `decode_tps`, `prefill_tps`, `t_total_ms`, `tokens_per_round`).
    pub metric: Option<String>,
    pub min: Option<f64>,
    pub max: Option<f64>,
    /// Order requests or sessions by `decode_tps`, `prefill_tps` (sessions: their medians) or `time` (default).
    pub sort: Option<String>,
    /// `desc` (default) or `asc`.
    pub order: Option<String>,
}
impl Filter {
    pub fn bounds(&self, now: i64) -> Result<(i64, i64)> {
        let duration = match self.range.as_deref().unwrap_or("24h") {
            "1h" => 3600000,
            "6h" => 21600000,
            "24h" => 86400000,
            "7d" => 604800000,
            _ => return Err(Error::Settings("invalid range")),
        };
        let to = self.to.unwrap_or(now);
        let from = self.from.unwrap_or(to.saturating_sub(duration));
        if from > to {
            return Err(Error::Settings("from must not exceed to"));
        }
        Ok((from, to))
    }
    /// The offset a rate-sorted request page starts at (`o:<n>` cursors).
    fn offset(&self) -> Result<i64> {
        match &self.cursor {
            None => Ok(0),
            Some(c) => c.strip_prefix("o:").and_then(|n| n.parse().ok()).ok_or(Error::Settings("invalid cursor")),
        }
    }
    fn bucket_ms(&self) -> Result<i64> {
        match self.bucket.as_deref().unwrap_or("1h") {
            "1m" => Ok(60000),
            "5m" => Ok(300000),
            "1h" => Ok(3600000),
            "1d" => Ok(86400000),
            _ => Err(Error::Settings("invalid bucket")),
        }
    }
}
pub fn percentile(values: impl Iterator<Item = f64>, p: f64) -> Option<f64> {
    let mut v = values.filter(|n| n.is_finite()).collect::<Vec<_>>();
    if v.is_empty() {
        return None;
    }
    v.sort_by(f64::total_cmp);
    Some(v[((v.len() - 1) as f64 * p).round() as usize])
}
pub fn histogram(values: impl Iterator<Item = f64>) -> [u64; 24] {
    let mut bins = [0; 24];
    for v in values.filter(|v| v.is_finite() && *v >= 0.) {
        let i = if v <= 1. {
            0
        } else {
            v.log2().floor() as usize
        };
        bins[i.min(23)] += 1;
    }
    bins
}
fn latency(rows: &[Record]) -> Value {
    let metric = |get: fn(&Record) -> Option<f64>| json!({"p50":percentile(rows.iter().filter_map(get),0.5),"p95":percentile(rows.iter().filter_map(get),0.95),"p99":percentile(rows.iter().filter_map(get),0.99),"histogram":histogram(rows.iter().filter_map(get))});
    json!({"ttft":metric(|r|r.t_ttft_ms),"decode":metric(|r|r.decode_tps),"prefill":metric(|r|r.prefill_tps)})
}
/// Prefill and decode performance over a set of requests. Prefill counts only
/// computed prompt tokens (input minus prefix-cache hits) over admit → first
/// token; decode counts output after the first token over first token → retire.
/// Totals are token sums over time sums; medians are over per-request rates.
pub fn perf(rows: &[Record]) -> Value {
    let (mut computed, mut cached, mut prefill_ms, mut decoded, mut decode_ms) = (0u64, 0u64, 0f64, 0u64, 0f64);
    let (mut queue_total, mut prefill_total, mut decode_total) = (0f64, 0f64, 0f64);
    let (mut prefill_rates, mut decode_rates) = (vec![], vec![]);
    for r in rows {
        let input = r.tokens_in.unwrap_or(0);
        let hit = r.tokens_cached.unwrap_or(0).min(input);
        cached += hit;
        if let Some(admit) = r.t_admit_ms {
            queue_total += admit;
        }
        if let (Some(admit), Some(first)) = (r.t_admit_ms, r.t_ttft_ms) {
            let ms = (first - admit).max(0.);
            prefill_total += ms;
            if ms > 0. && r.tokens_in.is_some() {
                computed += input - hit;
                prefill_ms += ms;
                prefill_rates.push((input - hit) as f64 * 1000. / ms);
            }
        }
        if let (Some(first), Some(output)) = (r.t_ttft_ms, r.tokens_out) {
            let end = r.t_retire_ms.or(r.t_total_ms).unwrap_or(first);
            let ms = (end - first).max(0.);
            decode_total += ms;
            if ms > 0. && output > 1 {
                decoded += output - 1;
                decode_ms += ms;
                decode_rates.push((output - 1) as f64 * 1000. / ms);
            }
        }
    }
    let input = computed + cached;
    json!({
        "prefill": {"computed_tokens": computed, "cached_tokens": cached, "ms": prefill_ms,
            "total_tps": (prefill_ms > 0.).then(|| computed as f64 * 1000. / prefill_ms),
            "median_tps": percentile(prefill_rates.into_iter(), 0.5)},
        "decode": {"tokens": decoded, "ms": decode_ms,
            "total_tps": (decode_ms > 0.).then(|| decoded as f64 * 1000. / decode_ms),
            "median_tps": percentile(decode_rates.into_iter(), 0.5)},
        "cache_hit": ratio(cached, sum(rows, |r| r.tokens_in).max(input)),
        "time_ms": {"queue": queue_total, "prefill": prefill_total, "decode": decode_total},
    })
}

/// The column an offset-paged request list orders by; None is the default
/// newest-first keyset cursor.
fn sort_key(f: &Filter) -> Result<Option<&'static str>> {
    match f.sort.as_deref() {
        None | Some("time") if !descending(f)? => Ok(Some("ts_ms")),
        None | Some("time") => Ok(None),
        Some("decode_tps") => Ok(Some("decode_tps")),
        Some("prefill_tps") => Ok(Some("prefill_tps")),
        _ => Err(Error::Settings("sort must be time, decode_tps or prefill_tps")),
    }
}
fn descending(f: &Filter) -> Result<bool> {
    match f.order.as_deref() {
        None | Some("desc") => Ok(true),
        Some("asc") => Ok(false),
        _ => Err(Error::Settings("order must be asc or desc")),
    }
}

fn sum(rows: &[Record], get: fn(&Record) -> Option<u64>) -> u64 {
    rows.iter().filter_map(get).sum()
}
fn ratio(n: u64, d: u64) -> Option<f64> {
    (d > 0).then(|| n as f64 / d as f64)
}
fn summary(rows: &[Record]) -> Value {
    let errors = rows.iter().filter(|r| r.outcome != "ok").count();
    let input = sum(rows, |r| r.tokens_in);
    let cached = sum(rows, |r| r.tokens_cached);
    json!({"requests":rows.len(),"errors":errors,"error_rate":ratio(errors as u64,rows.len() as u64),"tokens_in":input,"tokens_cached":cached,"tokens_out":sum(rows,|r|r.tokens_out),
        "cache_hit":ratio(cached,input),"acceptance":ratio(sum(rows,|r|r.draft_accepted),sum(rows,|r|r.draft_proposed)),"latency":latency(rows),
        "concurrency_max":rows.iter().filter_map(|r|r.concurrency_engine).max(),"concurrency_mean":ratio(sum(rows,|r|r.concurrency_engine),rows.iter().filter(|r|r.concurrency_engine.is_some()).count() as u64)})
}
fn group(rows: &[Record], key: impl Fn(&Record) -> String) -> BTreeMap<String, Vec<Record>> {
    let mut groups = BTreeMap::new();
    for r in rows {
        groups
            .entry(key(r))
            .or_insert_with(Vec::new)
            .push(r.clone());
    }
    groups
}
fn series(rows: &[Record], f: &Filter) -> Result<Value> {
    let bucket = f.bucket_ms()?;
    let split = f.split.as_deref().unwrap_or("protocol");
    if !matches!(split, "protocol" | "client" | "model") {
        return Err(Error::Settings("invalid split"));
    }
    let groups = group(rows, |r| {
        format!(
            "{}|{}",
            r.ts_ms.div_euclid(bucket) * bucket,
            match split {
                "client" => r.client_kind.as_str(),
                "model" => r.model_served.as_deref().unwrap_or(""),
                _ => r.protocol.as_str(),
            }
        )
    });
    Ok(json!(groups
        .into_iter()
        .map(|(key, rows)| {
            let (ts, split) = key.split_once('|').unwrap();
            let mut v = summary(&rows);
            v["ts_ms"] = json!(ts.parse::<i64>().unwrap());
            v["split"] = json!(split);
            v
        })
        .collect::<Vec<_>>()))
}
impl Store {
    /// Prefill/decode performance over the metadata rows of these requests.
    pub fn perf_for(&self, rids: &[String]) -> Result<Value> {
        let mut rows = vec![];
        for chunk in rids.chunks(400) {
            let marks = vec!["?"; chunk.len()].join(",");
            rows.extend(crate::store::read_records_query(
                &*self.reader.lock().map_err(|_| Error::Stopped)?,
                &format!("SELECT * FROM requests WHERE rid IN ({marks})"),
                chunk.iter().map(|r| rusqlite::types::Value::Text(r.clone())).collect(),
            )?);
        }
        Ok(perf(&rows))
    }
    pub fn query_rows(&self, f: &Filter) -> Result<Vec<Record>> {
        use rusqlite::types::Value as S;
        let (from, to) = f.bounds(self.clock.now_ms())?;
        let mut sql = "SELECT * FROM requests WHERE ts_ms>=? AND ts_ms<=?".to_string();
        let mut args = vec![S::Integer(from), S::Integer(to)];
        for (column, value) in [
            ("client_kind", &f.client),
            ("protocol", &f.protocol),
            ("model_served", &f.model),
            ("key_label", &f.key),
            ("session_id", &f.session),
        ] {
            if let Some(value) = value {
                sql.push_str(&format!(" AND {column}=?"));
                args.push(S::Text(value.clone()));
            }
        }
        if !f.bench.unwrap_or(false) {
            sql.push_str(" AND bench=0");
        }
        for (column, value) in [("route", &f.route), ("outcome", &f.outcome), ("stop_reason", &f.stop)] {
            if let Some(value) = value {
                sql.push_str(&format!(" AND {column}=?"));
                args.push(S::Text(value.clone()));
            }
        }
        if let Some(error) = &f.error {
            sql.push_str(" AND outcome<>'ok' AND coalesce(error_class,outcome)=?");
            args.push(S::Text(error.clone()));
        }
        if let Some(prompt) = &f.prompt {
            let (lo, hi) = match prompt.as_str() {
                "<1K" => (0, 1024),
                "1-8K" => (1024, 8192),
                "8-32K" => (8192, 32768),
                "32K+" => (32768, i64::MAX),
                _ => return Err(Error::Settings("invalid prompt bucket")),
            };
            sql.push_str(" AND coalesce(tokens_in,0)>=? AND coalesce(tokens_in,0)<?");
            args.extend([S::Integer(lo), S::Integer(hi)]);
        }
        if let Some(cached) = &f.cached {
            let fraction = "(CAST(coalesce(tokens_cached,0) AS REAL)/max(1,coalesce(tokens_in,0)))";
            sql.push_str(&match cached.as_str() {
                "0" => format!(" AND {fraction}=0"),
                "<50%" => format!(" AND {fraction}>0 AND {fraction}<0.5"),
                ">=50%" => format!(" AND {fraction}>=0.5"),
                _ => return Err(Error::Settings("invalid cached bucket")),
            });
        }
        if let Some(metric) = &f.metric {
            let expr = match metric.as_str() {
                "t_ttft_ms" | "decode_tps" | "prefill_tps" | "t_total_ms" => metric.clone(),
                "tokens_per_round" => "(CAST(tokens_out AS REAL)/nullif(rounds,0))".into(),
                _ => return Err(Error::Settings("invalid metric")),
            };
            if f.min.is_none() && f.max.is_none() {
                return Err(Error::Settings("a metric filter needs min or max"));
            }
            if let Some(min) = f.min {
                sql.push_str(&format!(" AND {expr}>=?"));
                args.push(S::Real(min));
            }
            if let Some(max) = f.max {
                sql.push_str(&format!(" AND {expr}<?"));
                args.push(S::Real(max));
            }
        }
        if let Some(cursor) = f.cursor.as_ref().filter(|_| sort_key(f).ok().flatten().is_none()) {
            let (ts, rid) = cursor
                .split_once(':')
                .ok_or(Error::Settings("invalid cursor"))?;
            let ts = ts
                .parse::<i64>()
                .map_err(|_| Error::Settings("invalid cursor"))?;
            sql.push_str(" AND (ts_ms<? OR (ts_ms=? AND rid<?))");
            args.extend([S::Integer(ts), S::Integer(ts), S::Text(rid.into())]);
        }
        if let (Some(limit), Some(key)) = (f.limit, sort_key(f)?) {
            // Rate-sorted pages use an offset cursor; rows without the rate sort last.
            let direction = if descending(f)? { "DESC" } else { "ASC" };
            sql.push_str(&format!(" ORDER BY {key} IS NULL, {key} {direction}, ts_ms DESC, rid DESC LIMIT ? OFFSET ?"));
            args.push(S::Integer(limit.clamp(1, 500) as i64 + 1));
            args.push(S::Integer(f.offset()?));
        } else if let Some(limit) = f.limit {
            sql.push_str(" ORDER BY ts_ms DESC,rid DESC LIMIT ?");
            args.push(S::Integer(limit.clamp(1, 500) as i64 + 1));
        } else {
            sql.push_str(" ORDER BY ts_ms,rid");
        }
        crate::store::read_records_query(
            &*self.reader.lock().map_err(|_| Error::Stopped)?,
            &sql,
            args,
        )
    }
    pub fn query(&self, kind: &str, f: &Filter, id: Option<&str>) -> Result<Value> {
        if kind == "log" {
            return Ok(self.log.get(id.unwrap_or(""))?.unwrap_or(Value::Null));
        }
        let mut f = f.clone();
        if kind == "sessions" {
            if let Some(id) = id {
                f.session = Some(id.into());
            }
        }
        if kind == "requests" && id.is_none() {
            f.limit = Some(f.limit.unwrap_or(100).clamp(1, 500));
        } else {
            // Paging applies to the request list only; aggregates cover the whole window.
            f.limit = None;
            f.cursor = None;
        }
        if id.is_some() && f.from.is_none() && f.range.is_none() {
            f.range = Some("7d".into());
        }
        let rows = self.query_rows(&f)?;
        let (from, _) = f.bounds(self.clock.now_ms())?;
        if matches!(kind, "summary" | "series")
            && from < self.clock.now_ms() - i64::from(self.settings().metadata_days) * 86400000
        {
            if f.key.is_some() || f.session.is_some() {
                return Err(Error::Settings(
                    "daily aggregates have no key or session IDs",
                ));
            }
            let daily = self.daily(&f)?;
            if kind == "series" {
                return Ok(
                    json!({"resolution":"daily","daily":daily,"retained":series(&rows,&f)?}),
                );
            }
            let mut v = summary(&rows);
            for field in [
                "requests",
                "errors",
                "tokens_in",
                "tokens_cached",
                "tokens_out",
            ] {
                let old = v[field].as_u64().unwrap_or(0);
                v[field] = json!(old + daily.iter().filter_map(|d| d[field].as_u64()).sum::<u64>());
            }
            v["error_rate"] = json!(ratio(
                v["errors"].as_u64().unwrap_or(0),
                v["requests"].as_u64().unwrap_or(0)
            ));
            v["cache_hit"] = json!(ratio(
                v["tokens_cached"].as_u64().unwrap_or(0),
                v["tokens_in"].as_u64().unwrap_or(0)
            ));
            let accepted = sum(&rows, |r| r.draft_accepted)
                + daily
                    .iter()
                    .filter_map(|d| d["draft_accepted"].as_u64())
                    .sum::<u64>();
            let proposed = sum(&rows, |r| r.draft_proposed)
                + daily
                    .iter()
                    .filter_map(|d| d["draft_proposed"].as_u64())
                    .sum::<u64>();
            v["acceptance"] = json!(ratio(accepted, proposed));
            v["resolution"] = json!("daily");
            v["daily"] = json!(daily);
            v["latency_resolution"] = json!("retained_rows_only");
            v["usage"] = self.counters.snapshot();
            return Ok(v);
        }

        match kind {
            "summary"=>{let mut v=summary(&rows);v["series"]=series(&rows,&f)?;v["usage"]=self.counters.snapshot();Ok(v)},
            "series"=>series(&rows,&f),
            "latency"=>{
                let mut v=latency(&rows);
                v["prompt_facets"]=json!(group(&rows,|r|match r.tokens_in.unwrap_or(0) {0..=1023=>"<1K",1024..=8191=>"1-8K",8192..=32767=>"8-32K",_=>"32K+"}.into()).into_iter().map(|(key,rows)|json!({"prompt":key,"latency":latency(&rows)})).collect::<Vec<_>>());
                v["cache_facets"]=json!(cache_facets(&rows)); Ok(v)
            },
            "flow"=>Ok(json!(group(&rows,|r|format!("{}|{}|{}",r.client_kind,r.protocol,r.model_served.as_deref().unwrap_or(""))).into_values().map(|rows|json!({"client":rows[0].client_kind,"protocol":rows[0].protocol,"model":rows[0].model_served,"requests":rows.len(),"tokens":sum(&rows,|r|r.tokens_in)+sum(&rows,|r|r.tokens_out)})).collect::<Vec<_>>())),
            "sessions" if id.is_some()=>Ok(json!({"session_id":id,"perf":perf(&rows),"turns":rows})),
            "sessions"=>{
                let mut sessions=group(&rows.iter().filter(|r|r.session_id.is_some()).cloned().collect::<Vec<_>>(),|r|r.session_id.clone().unwrap()).into_iter().map(|(id,rows)|json!({"session_id":id,"source":rows[0].session_source,"client":rows[0].client_kind,"model":rows[0].model_served,"first_ms":rows.first().map(|r|r.ts_ms),"last_ms":rows.last().map(|r|r.ts_ms),"summary":summary(&rows),"perf":perf(&rows),"turns":rows.iter().map(|r|json!({"rid":r.rid,"ts_ms":r.ts_ms,"tokens_in":r.tokens_in,"tokens_cached":r.tokens_cached,"tokens_out":r.tokens_out})).collect::<Vec<_>>()})).collect::<Vec<_>>();
                sessions.sort_by_key(|v|std::cmp::Reverse(v["last_ms"].as_i64()));
                if sort_key(&f)? == Some("ts_ms") {
                    sessions.reverse();
                } else if let Some(key) = sort_key(&f)? {
                    let field = if key == "decode_tps" { "decode" } else { "prefill" };
                    let rate = |v: &Value| v["perf"][field]["median_tps"].as_f64();
                    let desc = descending(&f)?;
                    sessions.sort_by(|a, b| match (rate(a), rate(b)) {
                        (Some(x), Some(y)) => if desc { y.total_cmp(&x) } else { x.total_cmp(&y) },
                        (Some(_), None) => std::cmp::Ordering::Less,
                        (None, Some(_)) => std::cmp::Ordering::Greater,
                        (None, None) => std::cmp::Ordering::Equal,
                    });
                }
                Ok(json!(sessions))
            },
            "cache"=>Ok(json!({"series":series(&rows,&f)?,"facets":cache_facets(&rows),"tokens_saved":sum(&rows,|r|r.tokens_cached),"hit_rate":ratio(sum(&rows,|r|r.tokens_cached),sum(&rows,|r|r.tokens_in))})),
            "speculation"=>Ok(json!({"series":series(&rows,&f)?,"by_model":group(&rows,|r|r.model_served.clone().unwrap_or_default()).into_iter().map(|(name,r)|json!({"model":name,"summary":summary(&r),"tokens_per_round":ratio(sum(&r,|r|r.tokens_out),sum(&r,|r|r.rounds)),"histogram":histogram(r.iter().filter_map(|r|ratio(r.tokens_out?,r.rounds?)))})).collect::<Vec<_>>(),"by_client":group(&rows,|r|r.client_kind.clone()).into_iter().map(|(name,r)|json!({"client":name,"summary":summary(&r)})).collect::<Vec<_>>()})),
            "errors"=>Ok(json!({"series":series(&rows.iter().filter(|r|r.outcome!="ok").cloned().collect::<Vec<_>>(),&f)?,"outcomes":outcomes(&rows,&f)?,"classes":group(&rows.iter().filter(|r|r.outcome!="ok").cloned().collect::<Vec<_>>(),|r|format!("{}|{}|{}",r.error_class.as_deref().unwrap_or(&r.outcome),r.route,r.client_kind)).into_iter().map(|(class,r)|json!({"class_route_client":class,"count":r.len(),"last_ms":r.last().map(|r|r.ts_ms)})).collect::<Vec<_>>(),"stops":group(&rows,|r|r.stop_reason.clone().unwrap_or_default()).into_iter().map(|(reason,r)|json!({"reason":reason,"count":r.len()})).collect::<Vec<_>>()})),
            "requests" if id.is_some()=>Ok(rows.iter().find(|r|Some(r.rid.as_str())==id).map(serde_json::to_value).transpose()?.unwrap_or(Value::Null)),
            "requests"=>{
                let limit=f.limit.unwrap_or(100).clamp(1,500);
                if sort_key(&f)?.is_some() {
                    let mut page = rows; let more = page.len() > limit; page.truncate(limit);
                    let next = more.then(|| format!("o:{}", f.offset().unwrap_or(0) + limit as i64));
                    return Ok(json!({"requests":page,"next_cursor":next}));
                }
                let cursor=f.cursor.as_ref().map(|s|{let (ts,rid)=s.split_once(':').ok_or(Error::Settings("invalid cursor"))?;Ok::<_,Error>((ts.parse::<i64>().map_err(|_|Error::Settings("invalid cursor"))?,rid.to_string()))}).transpose()?;
                let mut page=rows.into_iter().filter(|r|cursor.as_ref().is_none_or(|(ts,id)|(r.ts_ms,r.rid.as_str())<(*ts,id.as_str()))).take(limit+1).collect::<Vec<_>>();
                let more=page.len()>limit;page.truncate(limit);let next=more.then(||page.last().map(|r|format!("{}:{}",r.ts_ms,r.rid))).flatten();Ok(json!({"requests":page,"next_cursor":next}))
            },
            _=>Err(Error::Settings("unknown usage query")),
        }
    }
}
/// Requests per bucket and outcome class (`ok` included), for stacked columns.
fn outcomes(rows: &[Record], f: &Filter) -> Result<Value> {
    let bucket = f.bucket_ms()?;
    let mut buckets = BTreeMap::<i64, BTreeMap<String, u64>>::new();
    for r in rows {
        *buckets.entry(r.ts_ms.div_euclid(bucket) * bucket).or_default().entry(r.outcome.clone()).or_default() += 1;
    }
    Ok(json!(buckets.into_iter().map(|(ts, counts)| json!({"ts_ms": ts, "counts": counts})).collect::<Vec<_>>()))
}

fn cache_facets(rows: &[Record]) -> Vec<Value> {
    group(rows, |r| {
        let fraction = ratio(r.tokens_cached.unwrap_or(0), r.tokens_in.unwrap_or(0)).unwrap_or(0.);
        if fraction == 0. {
            "0"
        } else if fraction < 0.5 {
            "<50%"
        } else {
            ">=50%"
        }
        .into()
    })
    .into_iter()
    .map(|(key, rows)| json!({"fraction":key,"requests":rows.len(),"latency":latency(&rows)}))
    .collect()
}
#[cfg(test)]
mod tests {
    use super::*;
    use cuteafd_api::usage::UsageSink;
    use std::sync::Arc;
    struct Fixed;
    impl crate::Clock for Fixed {
        fn now_ms(&self) -> i64 {
            7 * 86400000
        }
    }
    #[test]
    fn synthetic_seven_days_percentiles_buckets_sessions_and_cursor() {
        let store = Store::open_with(None, Arc::new(Fixed), 4096).unwrap();
        for i in 0..700 {
            store.record(Record {
                rid: format!("r{i:04}"),
                ts_ms: i * 864000,
                protocol: "chat".into(),
                client_kind: "codex".into(),
                session_id: Some(format!("s{}", i / 100)),
                tokens_in: Some(100),
                tokens_cached: Some(50),
                tokens_out: Some(10),
                t_ttft_ms: Some(i as f64),
                decode_tps: Some(100.),
                outcome: "ok".into(),
                ..Default::default()
            });
        }
        store.flush().unwrap();
        let f = Filter {
            range: Some("7d".into()),
            bucket: Some("1d".into()),
            ..Default::default()
        };
        let summary = store.query("summary", &f, None).unwrap();
        assert_eq!(summary["requests"], 700);
        assert_eq!(summary["tokens_in"], 70000);
        assert_eq!(summary["latency"]["ttft"]["p50"], 350.);
        assert_eq!(summary["latency"]["ttft"]["p95"], 664.);
        assert_eq!(
            store
                .query("series", &f, None)
                .unwrap()
                .as_array()
                .unwrap()
                .len(),
            7
        );
        assert_eq!(
            store
                .query("sessions", &f, None)
                .unwrap()
                .as_array()
                .unwrap()
                .len(),
            7
        );
        let mut page = f.clone();
        page.limit = Some(30);
        let first = store.query("requests", &page, None).unwrap();
        page.cursor = first["next_cursor"].as_str().map(str::to_owned);
        let second = store.query("requests", &page, None).unwrap();
        assert_ne!(first["requests"][0]["rid"], second["requests"][0]["rid"]);
        for kind in ["flow", "latency", "cache", "speculation", "errors"] {
            assert!(!store.query(kind, &f, None).unwrap().is_null());
        }
    }

    /// Chart clicks filter server-side over the whole window, not one page.
    #[test]
    fn row_predicates_and_outcome_breakdown() {
        let store = Store::open_with(None, Arc::new(Fixed), 4096).unwrap();
        for i in 0..200i64 {
            store.record(Record {
                rid: format!("p{i:04}"),
                ts_ms: 6 * 86400000 + i * 1000,
                protocol: "messages".into(),
                route: if i % 2 == 0 { "/v1/messages" } else { "/v1/chat/completions" }.into(),
                client_kind: "claude_code".into(),
                tokens_in: Some(if i < 50 { 500 } else { 20000 }),
                tokens_cached: Some(if i < 100 { 0 } else { 15000 }),
                tokens_out: Some(40),
                rounds: Some(10),
                t_ttft_ms: Some(i as f64),
                stop_reason: Some(if i % 4 == 0 { "max_tokens" } else { "end_turn" }.into()),
                outcome: if i % 10 == 0 { "engine_error" } else { "ok" }.into(),
                error_class: (i % 20 == 0).then(|| "worker".into()),
                ..Default::default()
            });
        }
        store.flush().unwrap();
        let base = Filter { range: Some("7d".into()), bucket: Some("1d".into()), limit: Some(5), ..Default::default() };
        let count = |f: Filter| {
            let mut all = f;
            all.limit = None;
            store.query("summary", &all, None).unwrap()["requests"].as_u64().unwrap()
        };
        assert_eq!(count(Filter { metric: Some("t_ttft_ms".into()), min: Some(10.), max: Some(20.), ..base.clone() }), 10);
        assert_eq!(count(Filter { metric: Some("tokens_per_round".into()), min: Some(4.), ..base.clone() }), 200);
        assert_eq!(count(Filter { stop: Some("max_tokens".into()), ..base.clone() }), 50);
        assert_eq!(count(Filter { route: Some("/v1/messages".into()), ..base.clone() }), 100);
        assert_eq!(count(Filter { error: Some("worker".into()), ..base.clone() }), 10);
        assert_eq!(count(Filter { error: Some("engine_error".into()), ..base.clone() }), 10);
        assert_eq!(count(Filter { prompt: Some("<1K".into()), ..base.clone() }), 50);
        assert_eq!(count(Filter { prompt: Some("8-32K".into()), ..base.clone() }), 150);
        assert_eq!(count(Filter { cached: Some("0".into()), ..base.clone() }), 100);
        assert_eq!(count(Filter { cached: Some(">=50%".into()), ..base.clone() }), 100);
        // The cursor-paged table applies the same predicate on every page.
        let page = store.query("requests", &Filter { stop: Some("max_tokens".into()), ..base.clone() }, None).unwrap();
        assert!(page["requests"].as_array().unwrap().iter().all(|r| r["stop_reason"] == "max_tokens"));
        assert!(page["next_cursor"].is_string());
        assert!(store.query("summary", &Filter { metric: Some("rowid".into()), min: Some(1.), ..base.clone() }, None).is_err());
        let errors = store.query("errors", &base, None).unwrap();
        let counts = &errors["outcomes"][0]["counts"];
        assert_eq!((counts["ok"].as_u64(), counts["engine_error"].as_u64()), (Some(180), Some(20)));
    }

    /// Known prefill/decode numbers: prefill counts computed tokens only over
    /// admit → first token; decode counts output after the first token.
    #[test]
    fn prefill_and_decode_rates_per_session_and_sorting() {
        let store = Store::open_with(None, Arc::new(Fixed), 4096).unwrap();
        // Session "a": two requests. r1: 1000 in, 0 cached, admit 10 ms, first 110 ms
        // (10000 tok/s), 101 out, retire 1110 ms (100 tok/s). r2: 1000 in, 900 cached,
        // admit 20, first 70 (2000 tok/s computed), 51 out, retire 270 (200 tok/s).
        // Session "b": one request at 50000 prefill tok/s and 50 decode tok/s.
        let row = |rid: &str, session: &str, input, cached, admit, first, out, retire, total| Record {
            rid: rid.into(), ts_ms: 6 * 86400000 + rid.len() as i64, protocol: "chat".into(), client_kind: "codex".into(),
            session_id: Some(session.into()), tokens_in: Some(input), tokens_cached: Some(cached), tokens_out: Some(out),
            t_admit_ms: Some(admit), t_ttft_ms: Some(first), t_retire_ms: Some(retire), t_total_ms: Some(total),
            prefill_tps: Some((input - cached) as f64 * 1000. / (first - admit)),
            decode_tps: Some((out - 1) as f64 * 1000. / (retire - first)), outcome: "ok".into(), ..Default::default()
        };
        store.record(row("r1", "a", 1000, 0, 10., 110., 101, 1110., 1200.));
        store.record(row("r22", "a", 1000, 900, 20., 70., 51, 270., 300.));
        store.record(row("r333", "b", 5000, 0, 0., 100., 11, 300., 300.));
        store.flush().unwrap();
        let f = Filter { range: Some("7d".into()), ..Default::default() };
        let a = store.query("sessions", &f, Some("a")).unwrap();
        let p = &a["perf"];
        assert_eq!(p["prefill"]["computed_tokens"], 1100);
        assert_eq!(p["prefill"]["cached_tokens"], 900);
        assert_eq!(p["prefill"]["total_tps"], 1100. * 1000. / 150.);
        assert_eq!(p["prefill"]["median_tps"], 10000., "median of 10000 and 2000 rounds up");
        assert_eq!(p["decode"]["tokens"], 150);
        assert_eq!(p["decode"]["total_tps"], 150. * 1000. / 1200.);
        assert_eq!(p["cache_hit"], 0.45);
        assert_eq!(p["time_ms"], json!({"queue": 30., "prefill": 150., "decode": 1200.}));
        let list = store.query("sessions", &Filter { sort: Some("prefill_tps".into()), ..f.clone() }, None).unwrap();
        assert_eq!(list[0]["session_id"], "b");
        assert!(list[0]["perf"]["decode"]["median_tps"].is_number());
        let list = store.query("sessions", &Filter { sort: Some("decode_tps".into()), ..f.clone() }, None).unwrap();
        assert_eq!(list[0]["session_id"], "a");
        let page = store.query("requests", &Filter { sort: Some("decode_tps".into()), order: Some("asc".into()), limit: Some(2), ..f.clone() }, None).unwrap();
        let rids = page["requests"].as_array().unwrap().iter().map(|r| r["rid"].as_str().unwrap().to_owned()).collect::<Vec<_>>();
        assert_eq!(rids, ["r333", "r1"]);
        assert_eq!(page["next_cursor"], "o:2");
        let next = store.query("requests", &Filter { sort: Some("decode_tps".into()), order: Some("asc".into()), limit: Some(2), cursor: Some("o:2".into()), ..f.clone() }, None).unwrap();
        assert_eq!(next["requests"][0]["rid"], "r22");
        assert_eq!(next["requests"][0]["t_retire_ms"], 270.);
        let oldest = store.query("requests", &Filter { order: Some("asc".into()), limit: Some(1), ..f.clone() }, None).unwrap();
        assert_eq!((oldest["requests"][0]["rid"].as_str(), oldest["next_cursor"].as_str()), (Some("r1"), Some("o:1")));
        let second = store.query("requests", &Filter { sort: Some("time".into()), order: Some("asc".into()), limit: Some(1), cursor: Some("o:1".into()), ..f.clone() }, None).unwrap();
        assert_eq!(second["requests"][0]["rid"], "r22");
        let sessions = store.query("sessions", &Filter { order: Some("asc".into()), ..f.clone() }, None).unwrap();
        assert_eq!(sessions[0]["session_id"], "a");
        assert!(store.query("requests", &Filter { sort: Some("rowid".into()), ..f }, None).is_err());
    }
}
