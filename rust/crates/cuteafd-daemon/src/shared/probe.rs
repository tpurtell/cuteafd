//! Benchmark probes in the serving loops (`cuteafd_api::openai::probe`): a
//! probed request may skip the prefix cache, decode without drafts, run given
//! token ids, record logits rows, or be a teacher-forced scoring pass that
//! ends without generating. Ordinary requests carry no probe and none of this
//! runs for them.
use super::token_io::DeviceLogits;
use anyhow::Result;
use cuteafd_api::openai::probe::Probe;
use cuteafd_ffi::NativeLibrary;
use std::sync::Arc;

pub(crate) type ProbeRef = Option<Arc<Probe>>;

/// Adds cumulative capture counts and site deltas since the last stats publication.
/// Gate clients subtract the cumulative fields across their own warmed interval.
pub(crate) fn graph_capture_stats(stats: &mut serde_json::Value) {
    static PREVIOUS: std::sync::Mutex<Vec<(String, u64)>> = std::sync::Mutex::new(Vec::new());
    let mut previous = PREVIOUS.lock().unwrap_or_else(|p| p.into_inner());
    let tables = cuteafd_loader::mapped_table_stats_with_intervals();
    if !tables.is_empty() { stats["mapped_tables"] = serde_json::json!(tables); }
    let sites = cuteafd_ffi::graph_capture_sites();
    record_graph_captures(stats, cuteafd_ffi::graph_captures(), sites, &mut previous);
}

fn record_graph_captures(stats: &mut serde_json::Value, total: u64, sites: Vec<(String, u64)>,
    previous: &mut Vec<(String, u64)>) {
    let deltas: Vec<_> = sites.iter().filter_map(|(site, count)| {
        let before = previous.iter().find(|(name, _)| name == site).map_or(0, |(_, n)| *n);
        let delta = count.saturating_sub(before);
        (delta > 0).then(|| (site.clone(), delta))
    }).collect();
    stats["graph_captures"] = total.into();
    stats["graph_capture_sites"] = serde_json::json!(sites);
    stats["graph_capture_site_deltas"] = serde_json::json!(deltas);
    *previous = sites;
}

#[cfg(test)]
mod graph_capture_tests {
    use super::*;

    #[test]
    fn stats_include_capture_total_and_per_site_deltas() {
        let mut stats = serde_json::json!({"active": 0});
        let mut previous = vec![("engine.rs:1937".into(), 2)];
        record_graph_captures(&mut stats, 5,
            vec![("engine.rs:1937".into(), 4), ("draft.rs:20".into(), 1)], &mut previous);
        assert_eq!(stats["graph_captures"], 5);
        assert_eq!(stats["graph_capture_site_deltas"],
            serde_json::json!([["engine.rs:1937", 2], ["draft.rs:20", 1]]));
        record_graph_captures(&mut stats, 5, previous.clone(), &mut previous);
        assert_eq!(stats["graph_capture_site_deltas"], serde_json::json!([]));
        assert_eq!(stats["graph_capture_sites"],
            serde_json::json!([["engine.rs:1937", 4], ["draft.rs:20", 1]]));
    }

    #[test]
    fn live_counter_fields_are_present_without_cuda() {
        let mut stats = serde_json::json!({});
        graph_capture_stats(&mut stats);
        assert!(stats["graph_captures"].is_u64());
        assert!(stats["graph_capture_sites"].is_array());
        assert!(stats["graph_capture_site_deltas"].is_array());
    }
}

/// The prompt ids a request runs: the probe's own, else `tokenize()`.
pub(crate) fn prompt_ids(probe: &ProbeRef, tokenize: impl FnOnce() -> Result<Vec<u32>>) -> Result<Vec<u32>> {
    match probe.as_ref().and_then(|p| p.spec.prompt_ids.clone()) {
        Some(ids) => Ok(ids),
        None => tokenize(),
    }
}

/// No prefix-cache lookup and nothing retained for this request.
pub(crate) fn cold(probe: &ProbeRef) -> bool {
    probe.as_ref().is_some_and(|p| p.spec.cold || p.spec.score_from.is_some())
}

/// Decode one token per step for this request.
pub(crate) fn no_speculation(probe: &ProbeRef) -> bool {
    probe.as_ref().is_some_and(|p| p.spec.no_speculation || p.scoring().is_some())
}

/// The scoring start, for a teacher-forced scoring request.
pub(crate) fn scoring(probe: &ProbeRef) -> Option<usize> {
    probe.as_ref().and_then(|p| p.scoring())
}

/// Whether the first generated token's row is wanted.
pub(crate) fn wants_first(probe: &ProbeRef) -> bool {
    probe.as_ref().is_some_and(|p| p.spec.record_first || p.spec.record_rows > 0)
}

/// Records decode row `row` (selecting generated token `generated`, at
/// `position`) when the probe wants that many rows.
pub(crate) fn decode_row(library: &NativeLibrary, probe: &ProbeRef, logits: &DeviceLogits, row: usize,
    generated: usize, position: usize) {
    let Some(p) = probe else { return };
    if generated >= p.spec.record_rows {
        return;
    }
    match logits.row_host(library, row) {
        Ok(host) => p.row(position, &host),
        Err(error) => p.fail(format!("decode row: {error:#}")),
    }
}

fn prompt_token_hash(ids: &[u32]) -> u64 {
    ids.iter().flat_map(|id| id.to_le_bytes()).fold(0xcbf2_9ce4_8422_2325, |hash, byte| {
        (hash ^ u64::from(byte)).wrapping_mul(0x0000_0100_0000_01b3)
    })
}

#[cfg(test)]
mod matched_prompt_tests {
    #[test]
    fn hashes_token_ids_in_order() {
        assert_eq!(super::prompt_token_hash(&[]), 0xcbf2_9ce4_8422_2325);
        assert_eq!(super::prompt_token_hash(&[1, 2]), 0xc9c2_8939_c996_68c6);
        assert_ne!(super::prompt_token_hash(&[1, 2]), super::prompt_token_hash(&[2, 1]));
    }
}

pub(crate) fn admitted(probe: &ProbeRef, engine: &str, ids: &[u32], cached: usize) {
    // Matched-card evidence is opt-in; ordinary serving does no hashing/logging.
    if std::env::var("CUTEAFD_BENCH_NONCE_SEED").is_ok_and(|seed| !seed.is_empty()) {
        tracing::info!(engine, prompt_tokens = ids.len(), prompt_token_hash = format_args!("{:016x}", prompt_token_hash(ids)), "matched benchmark prompt");
    }
    if let Some(probe) = probe {
        probe.admitted(engine, ids, cached);
    }
}

pub(crate) fn token(probe: &ProbeRef, token: u32) {
    if let Some(probe) = probe {
        probe.token(token);
    }
}

/// Records a host row predicting token `position`.
pub(crate) fn host_row(probe: &ProbeRef, position: usize, logits: &[f32]) {
    if let Some(probe) = probe {
        probe.row(position, logits);
    }
}

/// Records device rows `first..first + n` as predicting tokens `position..`.
pub(crate) fn device_rows(library: &NativeLibrary, probe: &ProbeRef, logits: &DeviceLogits, first: usize, n: usize,
    position: usize) -> Result<()> {
    let Some(probe) = probe else { return Ok(()) };
    anyhow::ensure!(logits.vocab > 0 && logits.stride >= logits.vocab, "invalid scoring vocabulary/stride");
    anyhow::ensure!(first.checked_add(n).is_some_and(|end| end <= logits.rows),
        "scoring needs {n} logits rows at {first}, got {}", logits.rows);
    let host = logits.to_host(library)?;
    for j in 0..n {
        let row = &host[(first + j) * logits.vocab..][..logits.vocab];
        probe.row(position + j, row);
    }
    Ok(())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ScorePath {
    Decode,
    Prefill,
}

impl ScorePath {
    pub(crate) fn parse(requested: Option<&str>, default: Self) -> Result<Self> {
        match requested {
            None => Ok(default),
            Some("decode") => Ok(Self::Decode),
            Some("prefill") => Ok(Self::Prefill),
            Some(other) => anyhow::bail!("unsupported probe score_path={other:?}; expected decode or prefill"),
        }
    }

    pub(crate) fn name(self) -> &'static str {
        match self { Self::Decode => "decode", Self::Prefill => "prefill" }
    }
}

pub(crate) fn verify_rows(probe: &ProbeRef) -> Option<usize> {
    probe.as_ref().and_then(|p| p.spec.verify_rows)
}

/// Diagnostic tails may be below a family's minimum pipelined chunk size.
/// Use a serial lane while preserving existing non-diagnostic chunking.
pub(crate) fn scoring_prefill_capacity(admitted: bool, lane_rows: usize, normal_rows: usize) -> usize {
    if admitted { lane_rows } else { normal_rows }
}

/// Keep the family's existing scoring width unless explicitly overridden.
/// Reject unsupported widths before prefill or any device work is queued.
fn scoring_width(capacity: usize, requested: Option<usize>) -> Result<usize> {
    anyhow::ensure!(capacity > 0, "scoring verify capacity must be positive");
    let rows = requested.unwrap_or(capacity);
    anyhow::ensure!((1..=capacity).contains(&rows),
        "unsupported probe verify_rows={rows}; family supports 1..={capacity}");
    Ok(rows)
}

/// Validate before grammar setup, cache admission, or any request device work.
pub(crate) fn validate_scoring(probe: &ProbeRef, full_prefill_logits: bool) -> Result<()> {
    let Some(probe) = probe.as_ref().filter(|p| p.scoring().is_some()) else { return Ok(()) };
    let path = ScorePath::parse(probe.spec.score_path.as_deref(), ScorePath::Decode)?;
    score_plan(2, 1, 1, usize::MAX, probe.spec.verify_rows, path, full_prefill_logits)?;
    Ok(())
}

/// Pipelined prefill may return rows spanning several lane workspaces.
/// Keep those host rows ordered rather than requiring another device buffer.
pub(crate) enum ScoreLogits {
    Device(DeviceLogits),
    Host { values: Vec<f32>, vocab: usize },
}

impl ScoreLogits {
    fn record(&self, library: &NativeLibrary, probe: &ProbeRef, n: usize, position: usize) -> Result<()> {
        match self {
            Self::Device(logits) => {
                anyhow::ensure!(logits.rows >= n, "scoring needs {n} rows, got {}", logits.rows);
                device_rows(library, probe, logits, logits.rows - n, n, position)
            }
            Self::Host { values, vocab } => {
                anyhow::ensure!(*vocab > 0 && values.len() % vocab == 0, "invalid scoring host logits shape");
                let rows = values.len() / vocab;
                anyhow::ensure!(rows >= n, "scoring needs {n} rows, got {rows}");
                for (j, row) in values[(rows - n) * vocab..].chunks_exact(*vocab).enumerate() {
                    host_row(probe, position + j, row);
                }
                Ok(())
            }
        }
    }
}

#[derive(Debug, PartialEq, Eq)]
struct ScoreStep {
    tokens: std::ops::Range<usize>,
    path: ScorePath,
    logit_rows: usize,
    position: usize,
}

fn score_plan(len: usize, from: usize, prefill_rows: usize, verify_capacity: usize,
    requested_verify_rows: Option<usize>, path: ScorePath, full_prefill_logits: bool) -> Result<Vec<ScoreStep>> {
    anyhow::ensure!(len >= 2, "scoring needs at least two tokens");
    let rows = match path {
        ScorePath::Decode => scoring_width(verify_capacity, requested_verify_rows)?,
        ScorePath::Prefill => {
            anyhow::ensure!(requested_verify_rows.is_none(), "probe verify_rows requires score_path=decode");
            anyhow::ensure!(full_prefill_logits,
                "prefill-shaped probe scoring requires --full-prefill-logits at server launch (FULL_PREFILL_LOGITS=on)");
            prefill_rows.max(1)
        }
    };
    let from = from.clamp(1, len - 1);
    let mut steps = Vec::new();
    let mut done = 0;
    while done < from {
        let end = done.saturating_add(prefill_rows.max(1)).min(from);
        steps.push(ScoreStep { tokens: done..end, path: ScorePath::Prefill,
            logit_rows: usize::from(end == from), position: from });
        done = end;
    }
    // Row j over tokens[p..end] predicts token p + j + 1. The final input
    // token has no successor to score, so it is never run.
    while done < len - 1 {
        let end = done.saturating_add(rows).min(len - 1);
        steps.push(ScoreStep { tokens: done..end, path, logit_rows: end - done, position: done + 1 });
        done = end;
    }
    Ok(steps)
}

/// Teacher-forced scoring; `prefill` requests 0, 1, or every chunk row's
/// logits. Its cache commit is identical to ordinary prefill. Decode remains
/// the default, with no drafts, retained prefix, or diagnostic workspace growth.
#[allow(clippy::too_many_arguments)]
pub(crate) fn score<S>(library: &NativeLibrary, probe: &ProbeRef, tokens: &[u32], from: usize, prefill_rows: usize,
    verify_capacity: usize, requested_verify_rows: Option<usize>, full_prefill_logits: bool, state: &mut S,
    mut prefill: impl FnMut(&mut S, &[u32], usize) -> Result<Option<ScoreLogits>>,
    mut verify: impl FnMut(&mut S, &[u32]) -> Result<DeviceLogits>) -> Result<usize> {
    let path = ScorePath::parse(probe.as_ref().and_then(|p| p.spec.score_path.as_deref()), ScorePath::Decode)?;
    let steps = score_plan(tokens.len(), from, prefill_rows, verify_capacity, requested_verify_rows, path,
        full_prefill_logits)?;
    if let Some(probe) = probe { probe.selected_score_path(path.name()); }
    let mut scored = 0;
    for step in steps {
        let chunk = &tokens[step.tokens];
        let logits = match step.path {
            ScorePath::Prefill => prefill(state, chunk, step.logit_rows)?,
            ScorePath::Decode => Some(ScoreLogits::Device(verify(state, chunk)?)),
        };
        if step.logit_rows > 0 {
            logits.ok_or_else(|| anyhow::anyhow!("scoring produced no logits"))?
                .record(library, probe, step.logit_rows, step.position)?;
            scored += step.logit_rows;
        }
    }
    Ok(scored)
}

#[cfg(test)]
mod tests {
    use super::*;
    use cuteafd_api::openai::probe::ProbeSpec;

    #[test]
    fn scoring_path_parser_keeps_family_defaults_and_rejects_unknown_paths() {
        for default in [ScorePath::Decode, ScorePath::Prefill] {
            assert_eq!(ScorePath::parse(None, default).unwrap(), default);
            assert_eq!(ScorePath::parse(Some("decode"), default).unwrap(), ScorePath::Decode);
            assert_eq!(ScorePath::parse(Some("prefill"), default).unwrap(), ScorePath::Prefill);
            assert!(ScorePath::parse(Some("other"), default).is_err());
        }
    }

    #[test]
    fn both_shapes_map_every_scored_row_to_its_next_token() {
        for len in [2, 3, 9, 65, 130, 513] {
            for requested_from in [0, 1, 8, 64, len - 1, len, usize::MAX] {
                for width in [0, 1, 3, 8, 64, 256, usize::MAX] {
                    for path in [ScorePath::Decode, ScorePath::Prefill] {
                        let from = requested_from.clamp(1, len - 1);
                        let plan = score_plan(len, requested_from, width, 8, None, path, true).unwrap();
                        let mut next_input = 0;
                        let mut positions = Vec::new();
                        for step in plan {
                            assert_eq!(step.tokens.start, next_input);
                            next_input = step.tokens.end;
                            let rows = step.tokens.len();
                            assert!(rows > 0 && step.tokens.end < len);
                            if step.logit_rows > 0 {
                                assert_eq!(step.position, step.tokens.end - step.logit_rows + 1);
                                positions.extend(step.position..step.position + step.logit_rows);
                            }
                            if step.tokens.start >= from {
                                assert_eq!(step.path, path);
                                assert_eq!(step.logit_rows, rows);
                                assert!(rows <= if path == ScorePath::Decode { 8 } else { width.max(1) });
                            }
                        }
                        assert_eq!(next_input, len - 1);
                        assert_eq!(positions, (from..len).collect::<Vec<_>>());
                    }
                }
            }
        }
    }

    #[test]
    fn admitted_scoring_splits_short_pipelined_tails_into_serial_lanes() {
        assert_eq!(scoring_prefill_capacity(false, 256, 1024), 1024);
        let rows = scoring_prefill_capacity(true, 256, 1024);
        let plan = score_plan(513, 1, rows, 8, None, ScorePath::Prefill, true).unwrap();
        let continuation: Vec<_> = plan.iter().filter(|s| s.tokens.start >= 1).collect();
        assert_eq!(continuation.iter().map(|s| s.tokens.len()).collect::<Vec<_>>(), [256, 255]);
        assert_eq!(continuation.iter().map(|s| s.logit_rows).sum::<usize>(), 511);
    }

    #[test]
    fn prefill_requires_launch_admission_and_rejects_verify_override() {
        let error = score_plan(20, 4, 8, 8, None, ScorePath::Prefill, false).unwrap_err();
        assert!(error.to_string().contains("--full-prefill-logits"));
        assert!(score_plan(20, 4, 8, 8, Some(1), ScorePath::Prefill, true).unwrap_err()
            .to_string().contains("verify_rows requires score_path=decode"));
        assert!(score_plan(1, 0, 8, 8, None, ScorePath::Decode, false).is_err());
        assert!(score_plan(20, 4, 8, 8, None, ScorePath::Decode, false).is_ok());
    }

    #[test]
    fn scoring_probes_are_cold_draft_free_and_validated_before_execution() {
        assert!(!cold(&None));
        assert!(!no_speculation(&None));
        assert!(validate_scoring(&None, false).is_ok());
        let probe = Some(Probe::new(ProbeSpec { score_from: Some(4), score_path: Some("prefill".into()),
            ..ProbeSpec::default() }));
        assert!(cold(&probe));
        assert!(no_speculation(&probe));
        assert!(validate_scoring(&probe, false).unwrap_err().to_string().contains("--full-prefill-logits"));
        assert!(validate_scoring(&probe, true).is_ok());
    }

    #[test]
    fn scoring_width_defaults_and_override_bounds() {
        for capacity in [1, 8, 48] {
            assert_eq!(scoring_width(capacity, None).unwrap(), capacity);
            assert_eq!(scoring_width(capacity, Some(1)).unwrap(), 1);
            assert_eq!(scoring_width(capacity, Some(capacity)).unwrap(), capacity);
            assert!(scoring_width(capacity, Some(0)).is_err());
            assert!(scoring_width(capacity, Some(capacity + 1)).is_err());
            assert!(scoring_width(capacity, Some(usize::MAX)).is_err());
        }
        assert!(scoring_width(0, None).is_err());
        assert_eq!(verify_rows(&None), None);
        let probe = Some(Probe::new(ProbeSpec { verify_rows: Some(5), ..ProbeSpec::default() }));
        assert_eq!(verify_rows(&probe), Some(5));
    }
}
