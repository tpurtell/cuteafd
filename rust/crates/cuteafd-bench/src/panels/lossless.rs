//! Lossless speculation over a fixed prompt set: every case sends the baseline
//! check's cold probes (drafts on, then drafts off, 128 greedy tokens, rows
//! recorded with their top two) and passes only when the generated ids and
//! every recorded row hash are identical. No tie tolerance: a near-tie flip or
//! a row that differs without a flip fails, so a lucky prompt cannot pass the
//! gate. Fixed nonces make every failure replayable, and each case keeps its
//! probe pair in the pass's `evidence` (report.json only, never rendered).
use super::common::{plain, table};
use super::{Ctx, Panel, Rates};
use crate::baseline::{honoured, probe_of, top_two_margin, RowNoise};
use crate::report::ServerInfo;
use anyhow::{Context, Result};
use cuteafd_api::openai::probe::{ProbeRecord, ProbeSpec};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

/// Greedy tokens per probe, as the baseline's single check.
pub const TOKENS: u64 = 128;
/// The versioned default set shipped with this build.
pub const DEFAULT_NAME: &str = "spec-lossless-set-v1";
const DEFAULT: &str = include_str!("../../../../../scripts/bench/fixtures/spec-lossless-set-v1.json");
const MAX_CASES: usize = 64;
const MAX_PROMPT_BYTES: usize = 32 * 1024;
const MAX_NONCE_CHARS: usize = 128;

/// One case: the request text is `[{nonce}] {prompt}`, the baseline's form.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Case {
    pub id: String,
    pub prompt: String,
    /// The full nonce string, e.g. `7ec2b2071ff95934-0000000026` for a seeded run's 27th request.
    pub nonce: String,
    /// Where the case came from (a run that failed on it).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source: Option<String>,
    /// Rebuilt from a seed rather than a stored prompt, until a run reproduces it.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub reconstructed: bool,
}

impl Case {
    pub fn text(&self) -> String {
        format!("[{}] {}", self.nonce, self.prompt)
    }
}

/// A validated set and where it came from.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CaseSet {
    /// `spec-lossless-set-v1`, or the file name a run was given.
    pub name: String,
    /// SHA-256 of the canonical case list: which exact set ran.
    pub sha256: String,
    pub cases: Vec<Case>,
}

impl CaseSet {
    pub fn new(name: &str, cases: Vec<Case>) -> Result<Self, String> {
        validate(&cases)?;
        use sha2::{Digest, Sha256};
        let canonical = serde_json::to_vec(&cases).map_err(|e| e.to_string())?;
        let sha256 = Sha256::digest(canonical).iter().map(|b| format!("{b:02x}")).collect();
        Ok(Self { name: name.into(), sha256, cases })
    }

    /// Parses and validates a set file's JSON (`[{id, prompt, nonce}]`).
    pub fn parse(name: &str, text: &str) -> Result<Self, String> {
        let cases: Vec<Case> = serde_json::from_str(text).map_err(|e| format!("{name}: {e}"))?;
        Self::new(name, cases).map_err(|e| format!("{name}: {e}"))
    }

    pub fn load(path: &std::path::Path) -> anyhow::Result<Self> {
        let text = std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
        let name = path.file_stem().map(|s| s.to_string_lossy().into_owned()).unwrap_or_else(|| "custom".into());
        Self::parse(&name, &text).map_err(anyhow::Error::msg)
    }

    pub fn default_set() -> Self {
        Self::parse(DEFAULT_NAME, DEFAULT).expect("the shipped lossless set is valid")
    }
}

/// A set as a client sends it with a run (`cuteafd bench --lossless-set FILE`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SetRequest {
    pub name: String,
    pub cases: Vec<Case>,
}

impl SetRequest {
    pub fn of(set: &CaseSet) -> Self {
        Self { name: set.name.clone(), cases: set.cases.clone() }
    }

    pub fn validated(self) -> Result<CaseSet, String> {
        if self.name.is_empty() || self.name.len() > 128 || self.name.chars().any(char::is_control) {
            return Err("set name must be 1-128 printable characters".into());
        }
        CaseSet::new(&self.name, self.cases)
    }
}

fn validate(cases: &[Case]) -> Result<(), String> {
    if cases.is_empty() || cases.len() > MAX_CASES {
        return Err(format!("a set holds 1 to {MAX_CASES} cases, not {}", cases.len()));
    }
    let mut ids = std::collections::HashSet::new();
    for case in cases {
        if case.id.is_empty() || case.id.len() > 64
            || !case.id.chars().all(|c| c.is_ascii_alphanumeric() || "-_.".contains(c)) {
            return Err(format!("case id {:?} must be 1-64 of [A-Za-z0-9._-]", case.id));
        }
        if !ids.insert(case.id.as_str()) {
            return Err(format!("duplicate case id {}", case.id));
        }
        if case.prompt.trim().is_empty() || case.prompt.len() > MAX_PROMPT_BYTES {
            return Err(format!("{}: prompt must be non-empty and at most {MAX_PROMPT_BYTES} bytes", case.id));
        }
        if case.nonce.is_empty() || case.nonce.chars().count() > MAX_NONCE_CHARS
            || case.nonce.chars().any(|c| c.is_control() || c == '[' || c == ']') {
            return Err(format!("{}: nonce must be 1-{MAX_NONCE_CHARS} characters without brackets or control \
                characters", case.id));
        }
    }
    Ok(())
}

/// One case's verdict from its drafts-on and drafts-off records.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Verdict {
    pub id: String,
    pub pass: bool,
    pub tokens: usize,
    /// Generated tokens identical from the start.
    pub identical_prefix: usize,
    /// Generated index of the first differing token, when the ids differ.
    pub first_token_divergence: Option<usize>,
    /// `[drafts on, drafts off]` token ids there.
    pub divergent_tokens: Option<[Option<u32>; 2]>,
    pub rows_compared: usize,
    pub rows_differ: usize,
    /// Generated index of the first row whose hash differs.
    pub first_row_divergence: Option<usize>,
    /// Top-two margins (nats) of the row at the first divergence (token, else row), drafts off and on.
    pub margin_off: Option<f64>,
    pub margin_on: Option<f64>,
    pub prefix_rows_compared: usize,
    pub prefix_rows_identical: usize,
    pub prefix_noise_max: f64,
    pub prefix_noise_median: f64,
    pub reason: String,
}

/// Strict lossless judgment: equal ids, every off row present in the drafted
/// record, and every hash equal. Nothing is excused as a tie.
pub fn judge(id: &str, on: &ProbeRecord, off: &ProbeRecord) -> Verdict {
    let start = off.prompt_ids.len();
    let (a, b) = (&on.generated, &off.generated);
    let same = a.iter().zip(b).take_while(|(x, y)| x == y).count();
    let ids_equal = a == b;
    let mut compared = 0;
    let mut differ = Vec::new();
    let mut missing = 0;
    for row in &off.rows {
        match on.rows.iter().find(|r| r.position == row.position) {
            Some(other) => {
                compared += 1;
                if other.hash != row.hash {
                    differ.push(row.position);
                }
            }
            None => missing += 1,
        }
    }
    let first_row = differ.iter().min().map(|p| p.saturating_sub(start));
    let first_token = (!ids_equal).then_some(same);
    // Margins and noise where the runs first part: the flip, else the first differing row.
    let at = first_token.or(first_row);
    let noise = RowNoise::between(on, off, start + at.unwrap_or(b.len()));
    let (margin_off, margin_on) = match at {
        Some(at) => (top_two_margin(off, start + at), top_two_margin(on, start + at)),
        None => (None, None),
    };
    let complete = !off.rows.is_empty() && missing == 0 && on.rows.len() == off.rows.len();
    let pass = ids_equal && differ.is_empty() && complete && !b.is_empty();
    let margin = margin_off.or(margin_on).map(|m| format!(", top-two margin {m:.4} nats")).unwrap_or_default();
    let reason = if pass {
        format!("{} tokens and {compared} rows identical", b.len())
    } else if b.is_empty() {
        "drafts off generated nothing".into()
    } else if let Some(at) = first_token {
        format!("token {at} differs ({:?} on vs {:?} off){margin}{}", a.get(at), b.get(at), noise.describe())
    } else if let Some(at) = first_row {
        format!("ids identical, but {} of {compared} rows differ from token {at}{margin}{}", differ.len(),
            noise.describe())
    } else {
        format!("ids identical, but drafts on recorded {} of {} rows ({missing} missing)", on.rows.len(), off.rows.len())
    };
    Verdict { id: id.into(), pass, tokens: b.len(), identical_prefix: same, first_token_divergence: first_token,
        divergent_tokens: first_token.map(|at| [a.get(at).copied(), b.get(at).copied()]), rows_compared: compared,
        rows_differ: differ.len(), first_row_divergence: first_row, margin_off, margin_on,
        prefix_rows_compared: noise.compared, prefix_rows_identical: noise.identical, prefix_noise_max: noise.max,
        prefix_noise_median: noise.median, reason }
}

/// The probe pair as kept evidence: ids, row hashes and top two per row.
fn evidence(case: &Case, on: &ProbeRecord, off: &ProbeRecord) -> Value {
    let record = |r: &ProbeRecord| json!({"engine": r.engine, "cold": r.cold, "no_speculation": r.no_speculation,
        "generated": r.generated, "rows": r.rows.iter().map(|row| json!({"position": row.position, "hash": row.hash,
            "top": row.top})).collect::<Vec<_>>()});
    json!({"nonce": case.nonce, "text": case.text(), "max_tokens": TOKENS, "thinking": "disabled", "temperature": 0,
        "prompt_ids": off.prompt_ids, "on": record(on), "off": record(off)})
}

/// The pass record from the verdicts (`k/N`, a gate status and a table).
pub fn aggregate(set: &CaseSet, speculator: &str, verdicts: &[Verdict], evidence: serde_json::Map<String, Value>) -> Value {
    let passed = verdicts.iter().filter(|v| v.pass).count();
    let total = set.cases.len();
    let complete = verdicts.len() == total;
    let gate = if complete && passed == total { "pass" } else { "fail" };
    let failing: Vec<String> = verdicts.iter().filter(|v| !v.pass).map(|v| format!("{}: {}", v.id, v.reason)).collect();
    let summary = if gate == "pass" {
        format!("{speculator}: {passed}/{total} cases lossless (ids and row hashes identical)")
    } else if failing.is_empty() {
        format!("{speculator}: {passed}/{total} cases ran")
    } else {
        format!("{speculator}: {passed}/{total} lossless; {}", failing.join("; "))
    };
    let cases: std::collections::HashMap<&str, &Case> = set.cases.iter().map(|c| (c.id.as_str(), c)).collect();
    let rows: Vec<Value> = verdicts.iter().map(|v| {
        let mut row = serde_json::to_value(v).unwrap_or_default();
        row["name"] = json!(v.id);
        if let Some(case) = cases.get(v.id.as_str()) {
            row["nonce"] = json!(case.nonce);
            if case.reconstructed {
                row["reconstructed"] = json!(true);
            }
            if let Some(source) = &case.source {
                row["source"] = json!(source);
            }
        }
        row
    }).collect();
    let table_rows = verdicts.iter().map(|v| vec![json!(v.id), json!(if v.pass { "lossless" } else { "FAIL" }),
        json!(v.first_token_divergence), json!(v.first_row_divergence), json!(v.margin_off.or(v.margin_on)),
        json!(format!("{}/{}", v.prefix_rows_identical, v.prefix_rows_compared)), json!(v.prefix_noise_max),
        json!(v.reason)]).collect();
    json!({"gate": gate, "summary": summary, "passed": passed, "total": total, "speculator": speculator,
        "set": {"name": set.name, "sha256": set.sha256, "cases": total}, "rows": rows,
        "table": table(&["case", "result", "first token", "first row", "margin", "prefix rows identical",
            "prefix noise max", "detail"], table_rows),
        "evidence": evidence})
}

pub struct LosslessSet;
pub static LOSSLESS_SET: LosslessSet = LosslessSet;

impl Panel for LosslessSet {
    fn id(&self) -> &'static str { "spec_lossless_set" }
    fn title(&self) -> &'static str { "Lossless speculation set" }
    fn description(&self) -> &'static str {
        "Speculation lossless over a fixed prompt set (default spec-lossless-set-v1): cold drafts-on vs drafts-off, \
         128 greedy tokens per case; passes only with identical token ids and row hashes in every case."
    }
    fn estimate_s(&self, rates: &Rates, _info: &ServerInfo) -> f64 {
        let cases = CaseSet::default_set().cases.len() as f64;
        cases * 2.0 * rates.seconds(80.0, TOKENS as f64)
    }
    fn unavailable(&self, info: &ServerInfo) -> Option<String> {
        info.configuration.speculator.is_none().then(|| "no speculator: nothing to check".into())
    }
    fn run(&self, ctx: &Ctx<'_>) -> Result<Value> {
        let default;
        let set = match ctx.lossless_set {
            Some(set) => set,
            None => { default = CaseSet::default_set(); &default }
        };
        let speculator = ctx.info.configuration.speculator.clone().unwrap_or_default();
        let spec = |off: bool| ProbeSpec { cold: true, no_speculation: off, record_rows: TOKENS as usize, top_k: 2,
            ..ProbeSpec::default() };
        let mut verdicts = Vec::new();
        let mut kept = serde_json::Map::new();
        for (i, case) in set.cases.iter().enumerate() {
            ctx.progress.step(i as f64 / set.cases.len() as f64, format!("lossless {} ({}/{})", case.id, i + 1,
                set.cases.len()));
            let text = case.text();
            let on = ctx.client.chat(plain(&text, TOKENS), Some(spec(false)))
                .with_context(|| format!("{} drafts on", case.id))?;
            let on = probe_of(&on)?.clone();
            anyhow::ensure!(honoured(&on), "this engine does not run benchmark probes");
            let off = ctx.client.chat(plain(&text, TOKENS), Some(spec(true)))
                .with_context(|| format!("{} drafts off", case.id))?;
            let off = probe_of(&off)?.clone();
            let mut verdict = judge(&case.id, &on, &off);
            if verdict.pass && !(on.cold && off.cold && off.no_speculation && !on.no_speculation) {
                verdict.pass = false;
                verdict.reason = "the engine did not honour cold / drafts-off as asked".into();
            }
            kept.insert(case.id.clone(), evidence(case, &on, &off));
            verdicts.push(verdict);
            ctx.progress.partial(aggregate(set, &speculator, &verdicts, serde_json::Map::new()));
        }
        Ok(aggregate(set, &speculator, &verdicts, kept))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cuteafd_api::openai::probe::ProbeRow;

    fn record(generated: &[u32], hashes: &[&str], no_speculation: bool) -> ProbeRecord {
        ProbeRecord { engine: Some("test".into()), prompt_ids: vec![9; 10], cold: true, no_speculation,
            generated: generated.to_vec(),
            rows: hashes.iter().enumerate().map(|(i, hash)| ProbeRow { position: 10 + i, hash: (*hash).into(),
                top: vec![(generated.get(i).copied().unwrap_or(0), -0.1), (99, -0.4 - i as f32 * 0.1)],
                ..ProbeRow::default() }).collect(),
            ..ProbeRecord::default() }
    }

    fn set(n: usize) -> CaseSet {
        CaseSet::new("t", (0..n).map(|i| Case { id: format!("c{i}"), prompt: "p".into(), nonce: format!("n{i}"),
            source: None, reconstructed: false }).collect()).unwrap()
    }

    #[test]
    fn identical_ids_and_hashes_pass() {
        let v = judge("a", &record(&[1, 2, 3], &["x", "y", "z"], false), &record(&[1, 2, 3], &["x", "y", "z"], true));
        assert!(v.pass, "{v:?}");
        assert_eq!((v.rows_compared, v.rows_differ, v.first_token_divergence), (3, 0, None));
    }

    #[test]
    fn a_hash_mismatch_with_equal_ids_fails() {
        let v = judge("a", &record(&[1, 2, 3], &["x", "q", "z"], false), &record(&[1, 2, 3], &["x", "y", "z"], true));
        assert!(!v.pass);
        assert_eq!((v.first_token_divergence, v.first_row_divergence, v.rows_differ), (None, Some(1), 1));
        assert!(v.reason.contains("ids identical, but 1 of 3 rows differ from token 1"), "{}", v.reason);
        // The margin is the differing row's: -0.1 - (-0.5).
        assert!((v.margin_off.unwrap() - 0.4).abs() < 1e-6, "{v:?}");
        // Row noise over the rows before it.
        assert_eq!((v.prefix_rows_compared, v.prefix_rows_identical), (1, 1));
    }

    #[test]
    fn a_token_flip_fails_with_no_tie_tolerance() {
        let mut on = record(&[1, 2, 7], &["x", "y", "w"], false);
        let off = record(&[1, 2, 3], &["x", "y", "z"], true);
        // A near tie (margin 1e-5) is still a failure here.
        on.rows[2].top = vec![(7, -0.30000), (3, -0.30001)];
        let v = judge("a", &on, &off);
        assert!(!v.pass);
        assert_eq!((v.first_token_divergence, v.divergent_tokens, v.identical_prefix), (Some(2), Some([Some(7), Some(3)]), 2));
        assert!(v.margin_on.unwrap() < 1e-4);
        assert!(v.reason.starts_with("token 2 differs (Some(7) on vs Some(3) off)"), "{}", v.reason);
    }

    #[test]
    fn missing_drafted_rows_fail() {
        let v = judge("a", &record(&[1, 2, 3], &["x", "y"], false), &record(&[1, 2, 3], &["x", "y", "z"], true));
        assert!(!v.pass);
        assert!(v.reason.contains("recorded 2 of 3 rows (1 missing)"), "{}", v.reason);
        assert!(!judge("a", &record(&[], &[], false), &record(&[], &[], true)).pass);
    }

    #[test]
    fn aggregation_reports_k_of_n_and_every_failing_case() {
        let s = set(3);
        let good = judge("c0", &record(&[1], &["x"], false), &record(&[1], &["x"], true));
        let flip = judge("c1", &record(&[2], &["y"], false), &record(&[1], &["x"], true));
        let rows = judge("c2", &record(&[1], &["q"], false), &record(&[1], &["x"], true));
        let all = aggregate(&s, "dflash", &[good.clone(), Verdict { id: "c1".into(), ..good.clone() },
            Verdict { id: "c2".into(), ..good.clone() }], serde_json::Map::new());
        assert_eq!((all["gate"].as_str(), all["passed"].as_u64(), all["total"].as_u64()), (Some("pass"), Some(3), Some(3)));
        assert_eq!(all["summary"], "dflash: 3/3 cases lossless (ids and row hashes identical)");
        let mixed = aggregate(&s, "dflash", &[good.clone(), flip, rows], serde_json::Map::new());
        assert_eq!((mixed["gate"].as_str(), mixed["passed"].as_u64()), (Some("fail"), Some(1)));
        let summary = mixed["summary"].as_str().unwrap();
        assert!(summary.starts_with("dflash: 1/3 lossless; c1: token 0 differs"), "{summary}");
        assert!(summary.contains("c2: ids identical, but 1 of 1 rows differ"), "{summary}");
        // A run cut short never passes, even when every case that ran did.
        let short = aggregate(&s, "dflash", &[good], serde_json::Map::new());
        assert_eq!(short["gate"], "fail");
    }

    #[test]
    fn report_record_schema() {
        let s = CaseSet::new("t", vec![Case { id: "glmf-v19-base".into(), prompt: "p".into(),
            nonce: "7ec2b2071ff95934-0000000026".into(), source: Some("run v19".into()), reconstructed: true }]).unwrap();
        let on = record(&[1, 2], &["x", "q"], false);
        let off = record(&[1, 2], &["x", "y"], true);
        let mut kept = serde_json::Map::new();
        kept.insert("glmf-v19-base".into(), evidence(&s.cases[0], &on, &off));
        let value = aggregate(&s, "dflash", &[judge("glmf-v19-base", &on, &off)], kept);
        let keys = |v: &Value| { let mut k: Vec<String> = v.as_object().unwrap().keys().cloned().collect(); k.sort(); k };
        assert_eq!(keys(&value), ["evidence", "gate", "passed", "rows", "set", "speculator", "summary", "table", "total"]);
        assert_eq!(keys(&value["set"]), ["cases", "name", "sha256"]);
        assert_eq!(value["set"]["sha256"].as_str().unwrap().len(), 64);
        let row = &value["rows"][0];
        for key in ["id", "name", "pass", "first_token_divergence", "first_row_divergence", "margin_off", "margin_on",
            "prefix_rows_compared", "prefix_rows_identical", "prefix_noise_max", "reason", "nonce", "source"] {
            assert!(row.get(key).is_some(), "row lacks {key}: {row}");
        }
        assert_eq!(row["reconstructed"], true);
        let e = &value["evidence"]["glmf-v19-base"];
        assert_eq!(keys(e), ["max_tokens", "nonce", "off", "on", "prompt_ids", "temperature", "text", "thinking"]);
        assert_eq!(e["text"], "[7ec2b2071ff95934-0000000026] p");
        assert_eq!(e["prompt_ids"].as_array().unwrap().len(), 10);
        assert_eq!(keys(&e["on"]), ["cold", "engine", "generated", "no_speculation", "rows"]);
        assert_eq!(e["off"]["rows"][1], json!({"position": 11, "hash": "y", "top": [[2, -0.1f32], [99, -0.5]]}));
        assert_eq!(value["table"]["rows"][0][1], "FAIL");
        // The exports draw one mark per case and never the evidence.
        let mut report = crate::sample::report(false);
        report.panels.push(crate::report::PanelResult { id: "spec_lossless_set".into(), title: "t".into(),
            passes: vec![value], ..Default::default() });
        let svg = crate::render::report::panel_svg(&report, "spec_lossless_set");
        assert!(svg.contains("0/1") && svg.contains("glmf-v19-base") && !svg.contains("evidence"), "{svg}");
    }

    #[test]
    fn the_default_set_is_valid_versioned_and_starts_from_the_code_prompt() {
        let set = CaseSet::default_set();
        assert_eq!(set.name, DEFAULT_NAME);
        let code: Vec<&Case> = set.cases.iter().filter(|c| c.prompt == crate::baseline::CONTENT[0].1).collect();
        assert_eq!(code.len(), 8);
        let mut nonces: Vec<&str> = code.iter().map(|c| c.nonce.as_str()).collect();
        nonces.sort();
        nonces.dedup();
        assert_eq!(nonces.len(), 8);
        for (_, prompt) in &crate::baseline::CONTENT[1..] {
            assert!(set.cases.iter().any(|c| c.prompt == *prompt), "{prompt}");
        }
        let v19 = set.cases.iter().find(|c| c.id == "glmf-v19-base").unwrap();
        assert!(v19.reconstructed && v19.source.as_deref().is_some_and(|s| s.contains("drafter-efficiency-serving-v19")));
        assert_eq!(v19.text(), format!("[7ec2b2071ff95934-0000000026] {}", crate::baseline::CONTENT[0].1));
    }

    #[test]
    fn set_files_parse_and_malformed_sets_are_refused() {
        let ok = r#"[{"id": "a", "prompt": "Write code.", "nonce": "seed-0000000003"}]"#;
        let set = CaseSet::parse("mine", ok).unwrap();
        assert_eq!((set.name.as_str(), set.cases[0].text().as_str()), ("mine", "[seed-0000000003] Write code."));
        assert_ne!(set.sha256, CaseSet::parse("mine", &ok.replace("0003", "0004")).unwrap().sha256);
        for (bad, why) in [
            ("[]", "1 to 64 cases"),
            ("{}", "invalid type"),
            (r#"[{"id": "a", "prompt": "p"}]"#, "missing field `nonce`"),
            (r#"[{"id": "a", "prompt": "p", "nonce": "n", "extra": 1}]"#, "unknown field `extra`"),
            (r#"[{"id": "a b", "prompt": "p", "nonce": "n"}]"#, "case id"),
            (r#"[{"id": "a", "prompt": "p", "nonce": "n"}, {"id": "a", "prompt": "q", "nonce": "m"}]"#, "duplicate case id a"),
            (r#"[{"id": "a", "prompt": " ", "nonce": "n"}]"#, "prompt must be non-empty"),
            (r#"[{"id": "a", "prompt": "p", "nonce": ""}]"#, "nonce must be"),
            (r#"[{"id": "a", "prompt": "p", "nonce": "x]y"}]"#, "nonce must be"),
            (r#"[{"id": "a", "prompt": "p", "nonce": "x\ny"}]"#, "nonce must be"),
        ] {
            let error = CaseSet::parse("bad", bad).unwrap_err();
            assert!(error.contains(why), "{bad}: {error}");
        }
    }
}
