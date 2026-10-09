//! Roll only deleted rows, in the same transaction, so repeated pruning is idempotent.
use crate::{
    query::{histogram, Filter},
    store::{read_records_query, Error, Result, Settings},
    Store,
};
use rusqlite::{params, types::Value as SqlValue, Connection};
use serde_json::{json, Value};
use std::collections::BTreeMap;
type GroupKey = (String, String, String, String);
pub(crate) fn roll(
    c: &Connection,
    settings: &Settings,
    where_sql: &str,
    args: Vec<SqlValue>,
) -> Result<()> {
    if settings.daily_days == 0 {
        return Ok(());
    }
    let rows = read_records_query(
        c,
        &format!("SELECT * FROM requests WHERE {where_sql} ORDER BY ts_ms,rid"),
        args,
    )?;
    let mut groups = BTreeMap::<GroupKey, Vec<cuteafd_api::usage::Record>>::new();
    for r in rows {
        let day: String = c.query_row(
            "SELECT strftime('%Y-%m-%d',?1/1000,'unixepoch')",
            [r.ts_ms],
            |row| row.get(0),
        )?;
        groups
            .entry((
                day,
                r.protocol.clone(),
                r.client_kind.clone(),
                r.model_served.clone().unwrap_or_default(),
            ))
            .or_default()
            .push(r);
    }
    for ((day, protocol, client, model), rows) in groups {
        let mut ttft = histogram(rows.iter().filter_map(|r| r.t_ttft_ms));
        let mut decode = histogram(rows.iter().filter_map(|r| r.decode_tps));
        use rusqlite::OptionalExtension;
        let old:Option<(String,String)>=c.query_row("SELECT ttft_hist,decode_hist FROM daily WHERE day=?1 AND protocol=?2 AND client_kind=?3 AND model=?4",params![day,protocol,client,model],|r|Ok((r.get(0)?,r.get(1)?))).optional()?;
        if let Some((a, b)) = old {
            let a: [u64; 24] = serde_json::from_str(&a)?;
            let b: [u64; 24] = serde_json::from_str(&b)?;
            for i in 0..24 {
                ttft[i] += a[i];
                decode[i] += b[i];
            }
        }
        let sum = |get: fn(&cuteafd_api::usage::Record) -> Option<u64>| {
            rows.iter().filter_map(get).sum::<u64>()
        };
        c.execute("INSERT INTO daily VALUES(?1,?2,?3,?4,?5,?6,?7,?8,?9,?10,?11,?12,?13)
ON CONFLICT(day,protocol,client_kind,model) DO UPDATE SET requests=requests+excluded.requests,errors=errors+excluded.errors,
tokens_in=tokens_in+excluded.tokens_in,tokens_cached=tokens_cached+excluded.tokens_cached,tokens_out=tokens_out+excluded.tokens_out,
draft_proposed=draft_proposed+excluded.draft_proposed,draft_accepted=draft_accepted+excluded.draft_accepted,ttft_hist=excluded.ttft_hist,decode_hist=excluded.decode_hist",
            params![day,protocol,client,model,rows.len() as u64,rows.iter().filter(|r|r.outcome!="ok").count() as u64,sum(|r|r.tokens_in),sum(|r|r.tokens_cached),sum(|r|r.tokens_out),sum(|r|r.draft_proposed),sum(|r|r.draft_accepted),serde_json::to_string(&ttft)?,serde_json::to_string(&decode)?])?;
    }
    Ok(())
}
pub(crate) fn expire(c: &Connection, s: &Settings, now: i64) -> Result<()> {
    if s.daily_days == 0 {
        c.execute("DELETE FROM daily", [])?;
    } else {
        c.execute(
            "DELETE FROM daily WHERE day < date(?1/1000,'unixepoch')",
            [now.saturating_sub(i64::from(s.daily_days) * 86400000)],
        )?;
    }
    Ok(())
}
impl Store {
    pub fn daily(&self, f: &Filter) -> Result<Vec<Value>> {
        let (from, to) = f.bounds(self.clock.now_ms())?;
        let mut sql="SELECT day,protocol,client_kind,model,requests,errors,tokens_in,tokens_cached,tokens_out,draft_proposed,draft_accepted,ttft_hist,decode_hist FROM daily WHERE day>=date(?/1000,'unixepoch') AND day<=date(?/1000,'unixepoch')".to_string();
        let mut args = vec![SqlValue::Integer(from), SqlValue::Integer(to)];
        for (column, value) in [
            ("protocol", &f.protocol),
            ("client_kind", &f.client),
            ("model", &f.model),
        ] {
            if let Some(v) = value {
                sql.push_str(&format!(" AND {column}=?"));
                args.push(SqlValue::Text(v.clone()));
            }
        }
        sql.push_str(" ORDER BY day,protocol,client_kind,model");
        let c = self.reader.lock().map_err(|_| Error::Stopped)?;
        let mut stmt = c.prepare(&sql)?;
        let rows=stmt.query_map(rusqlite::params_from_iter(args),|r|Ok(json!({"day":r.get::<_,String>(0)?,"protocol":r.get::<_,String>(1)?,"client":r.get::<_,String>(2)?,"model":r.get::<_,String>(3)?,"requests":r.get::<_,u64>(4)?,"errors":r.get::<_,u64>(5)?,"tokens_in":r.get::<_,u64>(6)?,"tokens_cached":r.get::<_,u64>(7)?,"tokens_out":r.get::<_,u64>(8)?,"draft_proposed":r.get::<_,u64>(9)?,"draft_accepted":r.get::<_,u64>(10)?,"ttft_hist":serde_json::from_str::<Value>(&r.get::<_,String>(11)?).unwrap_or(Value::Null),"decode_hist":serde_json::from_str::<Value>(&r.get::<_,String>(12)?).unwrap_or(Value::Null)})))?;
        rows.map(|r| r.map_err(Error::from)).collect()
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use cuteafd_api::usage::{Record, UsageSink};
    use std::sync::{
        atomic::{AtomicI64, Ordering::Relaxed},
        Arc,
    };
    struct Clock(AtomicI64);
    impl crate::Clock for Clock {
        fn now_ms(&self) -> i64 {
            self.0.load(Relaxed)
        }
    }
    #[test]
    fn roll_sums_histograms_idempotence_and_ninety_day_expiry() {
        let clock = Arc::new(Clock(AtomicI64::new(10 * 86400000)));
        let store = Store::open_with(None, clock.clone(), 4096).unwrap();
        for i in 0..100 {
            store.record(Record {
                rid: format!("r{i}"),
                ts_ms: 86400000,
                protocol: "chat".into(),
                client_kind: "codex".into(),
                tokens_in: Some(10),
                tokens_cached: Some(5),
                tokens_out: Some(2),
                t_ttft_ms: Some(8.),
                decode_tps: Some(32.),
                outcome: "ok".into(),
                ..Default::default()
            });
        }
        store.flush().unwrap();
        store.prune().unwrap();
        assert!(store.rows().unwrap().is_empty());
        let f = Filter {
            from: Some(0),
            to: Some(10 * 86400000),
            ..Default::default()
        };
        let rows = store.daily(&f).unwrap();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0]["requests"], 100);
        assert_eq!(rows[0]["tokens_in"], 1000);
        assert_eq!(rows[0]["tokens_cached"], 500);
        assert_eq!(rows[0]["tokens_out"], 200);
        assert_eq!(rows[0]["ttft_hist"][3], 100);
        store.prune().unwrap();
        assert_eq!(store.daily(&f).unwrap(), rows);
        let summary = store.query("summary", &f, None).unwrap();
        assert_eq!(summary["requests"], 100);
        assert_eq!(summary["resolution"], "daily");
        clock.0.store(101 * 86400000, Relaxed);
        store.prune().unwrap();
        assert!(store.daily(&f).unwrap().is_empty());
    }
}
