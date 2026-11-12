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

pub(crate) mod features;

pub(crate) type ProbeRef = Option<Arc<Probe>>;

pub(crate) fn image_echo(job: &cuteafd_api::openai::NativeRequest, spans: &[cuteafd_core::MediaSpan]) -> Result<()> {
    if let Some(probe) = &job.probe {
        anyhow::ensure!(spans.len() == job.media.len(), "probe image count differs");
        probe.media(spans.iter().zip(&job.media).enumerate().map(|(i, (span, image))| {
            cuteafd_api::openai::probe::ProbeMedia { start: span.start, len: span.len, kind: "image".into(),
                key: image.key.0.iter().map(|b| format!("{b:02x}")).collect(),
                grid: [image.grid.t, image.grid.h, image.grid.w],
                identity: Some(cuteafd_api::openai::probe::ProbeImageIdentity::prepared(image)),
                input_key: probe.spec.media.get(i).filter(|s| s.identity.is_some()).map(|s| s.key.clone()),
                fixture: probe.spec.media.get(i).and_then(|s| s.fixture.clone()), image_url: None }
        }).collect());
    }
    Ok(())
}

#[derive(Clone, Copy)]
pub(crate) struct RowMarkers {
    pub image: u32,
    pub row_break: u32,
    pub end: u32,
    pub merge: u32,
    pub vocabulary: u32,
}

#[derive(Clone, Copy)]
pub(crate) enum ImageMarkers {
    Bounded(cuteafd_loader::media::SpanExpander),
    Rows(RowMarkers),
}

/// Bind explicit probe ids to prepared image identities and checkpoint marker positions.
pub(crate) fn image_spans(spec: &cuteafd_api::openai::probe::ProbeSpec, tokens: &[u32],
    images: &[cuteafd_loader::media::PreparedImage], markers: ImageMarkers, max_context: usize)
    -> Result<Vec<cuteafd_core::MediaSpan>> {
    use cuteafd_core::media::ImageTokenGeometry;
    let (vocabulary, placeholder, reserved, geometry) = match markers {
        ImageMarkers::Bounded(m) => (m.vocabulary, m.placeholder, [m.placeholder, m.start, m.end], ImageTokenGeometry::MERGED),
        ImageMarkers::Rows(m) => (m.vocabulary, m.image, [m.image, m.row_break, m.end],
            ImageTokenGeometry { merge: m.merge, row_markers: 1 }),
    };
    spec.validate_media_geometry(geometry)?;
    anyhow::ensure!(spec.prompt_ids.as_deref() == Some(tokens) && spec.media.len() == images.len()
        && tokens.len() <= max_context, "expanded probe media count/context differs");
    anyhow::ensure!(vocabulary > 0 && vocabulary < 1 << 31 && tokens.iter().all(|&id| id < vocabulary)
        && reserved.iter().all(|&id| id < vocabulary)
        && reserved[0] != reserved[1] && reserved[0] != reserved[2] && reserved[1] != reserved[2],
        "probe image markers/token outside vocabulary");
    let mut spans = Vec::with_capacity(images.len());
    let mut covered = std::collections::BTreeSet::new();
    for (span, image) in spec.media.iter().zip(images) {
        let end = span.start.checked_add(span.len).ok_or_else(|| anyhow::anyhow!("probe media extent"))?;
        span.verify_image(image)?;
        match markers {
            ImageMarkers::Bounded(m) => {
                anyhow::ensure!(span.start > 0 && end < tokens.len() && tokens[span.start - 1] == m.start
                    && tokens[end] == m.end && tokens[span.start..end].iter().all(|&id| id == placeholder),
                    "probe image rows/marker boundaries differ");
                covered.extend(span.start - 1..=end);
            }
            ImageMarkers::Rows(m) => {
                let (h, w, _) = geometry.extent(span.grid).ok_or_else(|| anyhow::anyhow!("probe image grid"))?;
                for row in 0..h {
                    let start = span.start + row * (w + 1);
                    anyhow::ensure!(tokens[start..start + w].iter().all(|&id| id == placeholder)
                        && tokens[start + w] == if row + 1 == h { m.end } else { m.row_break },
                        "probe image rows/marker boundaries differ");
                }
                covered.extend(span.start..end);
            }
        }
        spans.push(cuteafd_core::MediaSpan { start: span.start, len: span.len, key: image.key.into() });
    }
    anyhow::ensure!(tokens.iter().enumerate().all(|(i, id)| !reserved.contains(id) || covered.contains(&i)),
        "unbound probe image placeholders/markers");
    Ok(spans)
}

/// Preserve the joined scheduler boundary independently of HTTP stats sampling.
pub(crate) fn first_request(once: &std::sync::Once, completed: u64, capture: impl FnOnce()) {
    if completed > 0 { once.call_once(capture); }
}

#[cfg(test)]
mod first_request_tests {
    #[test]
    fn reconciliation_fires_once_at_the_completed_request_boundary_not_on_fetch() {
        let once = std::sync::Once::new();
        let calls = std::cell::Cell::new(0);
        for completed in [0, 0, 1, 1, 2, 5] {
            super::first_request(&once, completed, || calls.set(calls.get() + 1));
            assert_eq!(calls.get(), usize::from(completed > 0));
        }
        for source in [include_str!("../families/glm5/serve.rs"),
            include_str!("../families/glm5_flash/serve.rs"), include_str!("qwen/scheduler.rs")] {
            let publish = source.split_once("\nfn publish").unwrap().1.split_once("\n}\n").unwrap().0;
            assert!(publish.contains("first_request(&FIRST, requests"));
            assert!(!publish.contains("graph_capture_stats"));
            assert!(source.contains("cache.tick();\n        publish(stats, requests"));
        }
    }
}

/// Request-time diagnostics, called by the HTTP blocking worker, never a scheduler.
/// Adds cumulative capture counts and site deltas since the last stats fetch.
/// Gate clients subtract the cumulative fields across their own warmed interval.
/// Startup-miss variants are cumulative too: each distinct shape a sealed graph
/// set lacked with its eager run count, so a client sees which one to add.
pub(crate) fn graph_capture_stats(stats: &mut serde_json::Value) {
    static PREVIOUS: std::sync::Mutex<Vec<(String, u64)>> = std::sync::Mutex::new(Vec::new());
    let mut previous = PREVIOUS.lock().unwrap_or_else(|p| p.into_inner());
    let tables = cuteafd_loader::mapped_table_stats_with_intervals();
    if !tables.is_empty() { stats["mapped_tables"] = serde_json::json!(tables); }
    stats["graph_cache"] = serde_json::json!(crate::shared::decode_graph::device::snapshots());
    stats["memory_reconciliation"] = crate::shared::memory_report::reconcile::snapshot();
    let sites = cuteafd_ffi::graph_capture_sites();
    record_graph_captures(stats, cuteafd_ffi::graph_captures(), sites, &mut previous);
    stats["graph_missing_variants"] = serde_json::json!(crate::shared::decode_graph::missing_variants()
        .into_iter().map(|miss| serde_json::json!({
            "family": miss.family, "shape": miss.shape, "runs": miss.runs })).collect::<Vec<_>>());
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
mod graph_stats_tests {
    use super::*;

    #[test]
    fn stats_publish_each_startup_miss_variant_with_its_runs() {
        use crate::shared::decode_graph::{GraphBank, GraphPolicy};
        let mut bank = GraphBank::<u32, ()>::with_family(GraphPolicy::FixedStartup, "probe-test-miss");
        bank.warm(&[1], |_| Ok(((), None))).unwrap();
        bank.enqueue(&9);
        bank.enqueue(&9);
        let mut stats = serde_json::json!({});
        graph_capture_stats(&mut stats);
        let variants = stats["graph_missing_variants"].as_array().expect("variant array");
        let ours: Vec<_> = variants.iter().filter(|variant| variant["family"] == "probe-test-miss").collect();
        assert_eq!(ours.len(), 1);
        assert_eq!((ours[0]["shape"].as_str(), ours[0]["runs"].as_u64()), (Some("9"), Some(2)));
        assert!(stats["graph_captures"].is_u64());
    }

    /// A bank miss and an owner that holds its own executable (Qwen4) report
    /// through the same logger, so both land in one /v1/stats inventory and an
    /// eager variant is described only on its first miss.
    #[test]
    fn stats_publish_qwen4_and_bank_misses_together() {
        use crate::shared::decode_graph::{GraphBank, GraphPolicy};
        let mut bank = GraphBank::<u32, ()>::with_family(GraphPolicy::FixedStartup, "probe-test-bank");
        bank.warm(&[1], |_| Ok(((), None))).unwrap();
        bank.enqueue(&9);
        let shape = "rows=6 spec=false long=false pool_width=64 page_stride=8192 pool_stride=524288";
        crate::shared::decode_graph::note_eager("qwen4", shape, || "segment=11 max_context=4096 slots=8 pages=4".into());
        crate::shared::decode_graph::note_eager("qwen4", shape,
            || panic!("only a variant's first miss describes its request"));
        let mut stats = serde_json::json!({});
        graph_capture_stats(&mut stats);
        let variants = stats["graph_missing_variants"].as_array().expect("variant array");
        let bank: Vec<_> = variants.iter().filter(|variant| variant["family"] == "probe-test-bank").collect();
        assert_eq!(bank.len(), 1);
        assert_eq!((bank[0]["shape"].as_str(), bank[0]["runs"].as_u64()), (Some("9"), Some(1)));
        let qwen4: Vec<_> = variants.iter().filter(|variant| variant["family"] == "qwen4").collect();
        assert_eq!(qwen4.len(), 1);
        assert_eq!((qwen4[0]["shape"].as_str(), qwen4[0]["runs"].as_u64()), (Some(shape), Some(2)));
    }
}

#[cfg(test)]
mod image_span_tests {
    use super::*;
    use cuteafd_api::openai::probe::{ProbeImageUrl, ProbeMedia, ProbeSpec};
    use cuteafd_loader::media::{ImageGrid, ImageKey, PreparedImage, SpanExpander};

    #[test]
    fn row_marker_fixture_rebinds_encoder_but_not_prepared_bytes() {
        use cuteafd_loader::media::{EncoderId, PreprocessId};
        use cuteafd_api::openai::probe::ProbeImageIdentity;
        let grid = ImageGrid { t: 1, h: 4, w: 4 };
        let processor = PreprocessId([2; 32]);
        let rgb: Arc<[u8]> = Arc::from([37; 48]);
        let key = PreparedImage::key_for(EncoderId([120; 32]), processor, grid, &rgb);
        let mut remote = PreparedImage { key: PreparedImage::key_for(EncoderId([121; 32]), processor, grid, &rgb),
            encoder: EncoderId([121; 32]), processor, grid, rgb8: rgb, tokens: 6 };
        let tokens = vec![1, 5, 5, 4, 5, 5, 6, 2];
        let spec = ProbeSpec { prompt_ids: Some(tokens.clone()), media: vec![ProbeMedia {
            start: 1, len: 6, kind: "image".into(), key: key.0.iter().map(|b| format!("{b:02x}")).collect(),
            grid: [1,4,4], identity: Some(ProbeImageIdentity { encoder: "78".repeat(32), processor: "02".repeat(32) }),
            input_key: None, fixture: None, image_url: Some(ProbeImageUrl { url: "fixture".into(), detail: None }) }],
            ..Default::default() };
        let markers = ImageMarkers::Rows(RowMarkers { image: 5, row_break: 4, end: 6, merge: 2, vocabulary: 32 });
        let spans = image_spans(&spec, &tokens, &[remote.clone()], markers, 16).unwrap();
        assert_eq!(spans[0].key, remote.key.into());
        remote.rgb8 = Arc::from([38; 48]);
        assert!(image_spans(&spec, &tokens, &[remote], markers, 16).is_err());
    }

    #[test]
    fn bounded_and_row_marker_probes_bind_exactly_once() {
        let rows = ImageMarkers::Rows(RowMarkers { image: 5, row_break: 4, end: 6, merge: 2, vocabulary: 32 });
        let bounded = ImageMarkers::Bounded(SpanExpander { placeholder: 5, start: 4, end: 6, vocabulary: 32 });
        for (markers, tokens, start, len) in [(bounded, vec![1, 4, 5, 5, 5, 5, 6, 2], 2, 4),
            (rows, vec![1, 5, 5, 4, 5, 5, 6, 2], 1, 6)] {
            let image = PreparedImage { key: ImageKey([7; 32]), encoder: cuteafd_loader::media::EncoderId([0; 32]), processor: cuteafd_loader::media::PreprocessId([0; 32]), grid: ImageGrid { t: 1, h: 4, w: 4 },
                rgb8: Arc::from([]), tokens: len };
            let mut spec = ProbeSpec { prompt_ids: Some(tokens.clone()), media: vec![ProbeMedia { start, len,
                kind: "image".into(), key: "07".repeat(32), grid: [1, 4, 4], identity: None, input_key: None, fixture: None,
                image_url: Some(ProbeImageUrl { url: "fixture".into(), detail: None }) }], ..Default::default() };
            let images = [image];
            let spans = image_spans(&spec, &tokens, &images, markers, 16).unwrap();
            assert_eq!((spans[0].start, spans[0].len), (start, len));
            for i in start..start + len {
                let mut wrong = tokens.clone(); wrong[i] = 3; spec.prompt_ids = Some(wrong.clone());
                assert!(image_spans(&spec, &wrong, &images, markers, 16).is_err());
            }
            let mut wrong = tokens.clone(); wrong[0] = 5; spec.prompt_ids = Some(wrong.clone());
            assert!(image_spans(&spec, &wrong, &images, markers, 16).is_err());
            spec.prompt_ids = Some(tokens.clone()); spec.media[0].key = "08".repeat(32);
            assert!(image_spans(&spec, &tokens, &images, markers, 16).is_err());
        }
    }
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

/// The evidence line for one admission. A prompt that re-renders model output
/// differs between engines by design, so it is reported without a hash: a
/// matched pair compares the hashes of shared, input-determined prompts only.
/// Panel-local diagnostics may run on just one arm and also carry no hash.
fn matched_prompt_line(probe: &ProbeRef, engine: &str, ids: &[u32]) {
    if probe.as_ref().is_some_and(|probe| probe.spec.pair_exempt) {
        tracing::info!(engine, prompt_tokens = ids.len(), "benchmark arm-local prompt");
    } else if probe.as_ref().is_none_or(|probe| probe.input_determined()) {
        tracing::info!(engine, prompt_tokens = ids.len(), prompt_token_hash = format_args!("{:016x}", prompt_token_hash(ids)), "matched benchmark prompt");
    } else {
        tracing::info!(engine, prompt_tokens = ids.len(), "benchmark output-dependent prompt");
    }
}

#[cfg(test)]
mod matched_prompt_tests {
    use super::*;
    use cuteafd_api::openai::probe::ProbeSpec;
    use std::sync::{Arc, Mutex};

    #[test]
    fn hashes_token_ids_in_order() {
        assert_eq!(super::prompt_token_hash(&[]), 0xcbf2_9ce4_8422_2325);
        assert_eq!(super::prompt_token_hash(&[1, 2]), 0xc9c2_8939_c996_68c6);
        assert_ne!(super::prompt_token_hash(&[1, 2]), super::prompt_token_hash(&[2, 1]));
    }

    /// The evidence line one admission writes.
    fn evidence_line(probe: ProbeRef) -> String {
        struct Buffer(Arc<Mutex<Vec<u8>>>);
        impl std::io::Write for Buffer {
            fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
                self.0.lock().unwrap().extend_from_slice(bytes);
                Ok(bytes.len())
            }
            fn flush(&mut self) -> std::io::Result<()> {
                Ok(())
            }
        }
        let bytes = Arc::new(Mutex::new(Vec::new()));
        let writer = bytes.clone();
        let subscriber = tracing_subscriber::fmt().with_writer(move || Buffer(writer.clone()))
            .without_time().with_ansi(false).finish();
        {
            let _guard = tracing::subscriber::set_default(subscriber);
            matched_prompt_line(&probe, "test", &[1, 2, 3]);
        }
        let line = String::from_utf8(bytes.lock().unwrap().clone()).unwrap();
        line
    }

    #[test]
    fn an_input_determined_prompt_logs_its_token_hash() {
        let line = evidence_line(None);
        assert!(line.contains("matched benchmark prompt"), "{line}");
        assert!(line.contains(&format!("prompt_token_hash={:016x}", prompt_token_hash(&[1, 2, 3]))), "{line}");
    }

    #[test]
    fn an_output_dependent_prompt_logs_no_token_hash() {
        let probe = Some(Probe::new(ProbeSpec { output_dependent: true, ..ProbeSpec::default() }));
        let line = evidence_line(probe);
        assert!(line.contains("benchmark output-dependent prompt"), "{line}");
        assert!(!line.contains("prompt_token_hash"), "{line}");
    }

    #[test]
    fn an_arm_local_prompt_logs_no_token_hash() {
        for output_dependent in [false, true] {
            let probe = Some(Probe::new(ProbeSpec { pair_exempt: true, output_dependent, ..ProbeSpec::default() }));
            let line = evidence_line(probe);
            assert!(line.contains("benchmark arm-local prompt"), "{line}");
            assert!(!line.contains("prompt_token_hash"), "{line}");
        }
    }

    #[test]
    fn an_ordinary_probe_still_logs_its_token_hash() {
        let line = evidence_line(Some(Probe::new(ProbeSpec { cold: true, ..ProbeSpec::default() })));
        assert!(line.contains("matched benchmark prompt") && line.contains("prompt_token_hash"), "{line}");
    }
}

pub(crate) fn admitted(probe: &ProbeRef, engine: &str, ids: &[u32], cached: usize) {
    // Matched-card evidence is opt-in; ordinary serving does no hashing/logging.
    if std::env::var("CUTEAFD_BENCH_NONCE_SEED").is_ok_and(|seed| !seed.is_empty()) {
        matched_prompt_line(probe, engine, ids);
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
    let selected = (0..n).filter(|&j| probe.wants(position + j)).collect::<Vec<_>>();
    let spans = selected.iter().map(|&j| (first + j, 1)).collect::<Vec<_>>();
    if spans.is_empty() { return Ok(()); }
    let host = logits.download(library, &spans)?;
    for (j, row) in selected.into_iter().zip(host.chunks_exact(logits.vocab)) {
        probe.row(position + j, row);
    }
    Ok(())
}

/// Head tiles only: transformer calls and their causal/pipeline geometry stay unchanged.
pub(crate) fn head_spans(rows: usize, capacity: usize) -> Result<impl Iterator<Item = std::ops::Range<usize>>> {
    anyhow::ensure!(rows > 0 && capacity > 0, "scoring head rows/capacity must be positive");
    Ok((0..rows).step_by(capacity).map(move |first| first..first.saturating_add(capacity).min(rows)))
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
    /// A streaming head consumed its bounded spans before returning.
    Recorded,
}

impl ScoreLogits {
    fn record(&self, library: &NativeLibrary, probe: &ProbeRef, n: usize, position: usize) -> Result<()> {
        match self {
            Self::Recorded => Ok(()),
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
    mut prefill: impl FnMut(&mut S, &[u32], usize, usize) -> Result<Option<ScoreLogits>>,
    mut verify: impl FnMut(&mut S, &[u32]) -> Result<DeviceLogits>) -> Result<usize> {
    let path = ScorePath::parse(probe.as_ref().and_then(|p| p.spec.score_path.as_deref()), ScorePath::Decode)?;
    let steps = score_plan(tokens.len(), from, prefill_rows, verify_capacity, requested_verify_rows, path,
        full_prefill_logits)?;
    if let Some(probe) = probe { probe.selected_score_path(path.name()); }
    let mut scored = 0;
    for step in steps {
        let chunk = &tokens[step.tokens];
        let logits = match step.path {
            ScorePath::Prefill => prefill(state, chunk, step.logit_rows, step.position)?,
            ScorePath::Decode => Some(ScoreLogits::Device(verify(state, chunk)?)),
        };
        if step.logit_rows > 0 {
            logits.ok_or_else(|| anyhow::anyhow!("scoring produced no logits"))?
                .record(library, probe, step.logit_rows, step.position)?;
            if step.path == ScorePath::Prefill && step.logit_rows == chunk.len() {
                if let Some(probe) = probe {
                    probe.scored_prefill_chunk(step.position, chunk.len());
                }
            }
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
    fn head_tiles_cover_the_call_without_changing_its_prefill_steps() {
        for capacity in [1, 48, 64] {
            for rows in [1, 47, 48, 49, 64, 65, 512, 8256] {
                let spans = head_spans(rows, capacity).unwrap().collect::<Vec<_>>();
                assert!(spans.iter().all(|span| !span.is_empty() && span.len() <= capacity));
                assert_eq!(spans.into_iter().flatten().collect::<Vec<_>>(), (0..rows).collect::<Vec<_>>());
            }
        }
        assert!(head_spans(0, 64).is_err());
        assert!(head_spans(64, 0).is_err());
        let plan = score_plan(9000, 1, 8256, 64, None, ScorePath::Prefill, true).unwrap();
        assert_eq!(plan.iter().map(|step| step.tokens.len()).collect::<Vec<_>>(), [1, 8256, 742]);
    }

    #[test]
    fn kimi_streams_the_same_head_tiles_without_growing_output_owners() {
        let source = include_str!("../families/kimi_k3/engine.rs");
        assert!(source.contains("head_spans(logit_rows, 64)"));
        assert!(source.contains("consume(first, describe(logits, rows))?"));
        let backend = include_str!("../families/kimi_k3/backend.rs");
        assert!(backend.contains("fn streamed_prefill_scoring(&self) -> bool { true }"));
        assert!(backend.contains("self.geometry.full_prefill_logits || logit_rows <= 1"));
    }

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
    fn kimi_context_replays_mandatory_graphs_after_prefill_and_ungraphed_scoring() -> Result<()> {
        use crate::families::kimi_k3::{context::split_bucket, graphs::{self, Key, Op}};
        use cuteafd_loader::families::kimi_k3::{config::{KimiAttention, KimiK3Config}, context::DecodeBuckets};
        let mut cfg = KimiK3Config::from_hf(&cuteafd_loader::plan::testing::kimi_k3_config())?;
        cfg.layers = 1;
        cfg.attention = vec![KimiAttention::Mla];
        let decode = DecodeBuckets::from_manifest(&serde_json::json!({"programs":[{
            "name":"dense_trellis_k6_trellis_mla_q_a_mcg",
            "geometry":{"bits":6,"decode_route_thresholds":[8,16,32,48]}}]}))?;
        let set = graphs::inventory_with_context(&cfg, 1, 2752, 16, 8, 2,
            |_, _| true, |_| true, Some(&decode))?;
        let keys = set.keys.iter().copied().collect::<std::collections::HashSet<_>>();
        let replay = |rows, speculative, optional, history: usize| -> Result<()> {
            let rows = decode.dispatch_rows(rows, speculative)?;
            let op = Op::Context { layer: 0, speculative };
            assert!(graphs::replay_allowed(op, optional, false)?);
            assert!(graphs::replay_allowed(op, optional, true).is_err());
            assert_eq!(graphs::replay_allowed(Op::Ffn(0), optional, false)?, optional);
            for rank in 0..2 {
                assert!(keys.contains(&Key { lane: 0, rank, rows, op }),
                    "missing scoring/decode Context graph at history={history}, rows={rows}");
                for sms in [170, 188] {
                    let logical = (0..history.div_ceil(64) as u32).collect::<Vec<_>>();
                    let pages = crate::shared::peer_split::context::local_pages(&logical, rank)?.len();
                    assert!(crate::families::kimi_k3::context::SPLIT_BUCKETS
                        .contains(&split_bucket(pages, rows, sms)?));
                }
            }
            Ok(())
        };
        // Walk the actual card sequence: a wide prefill, then the 513-token
        // teacher-forced probe, then ordinary and mixed speculative C1..C16.
        for prompt in [8027, 8192, 65536] {
            let units = crate::shared::prefill_pipeline::prefill_units(prompt, 2752, 3, 256)?;
            assert_eq!(units.iter().map(|unit| unit.len()).sum::<usize>(), prompt);
            for unit in units { if unit.len() <= 64 { replay(unit.len(), false, true, unit.end)?; } }
            for requested in [None, Some(1), Some(8), Some(13), Some(64)] {
                let steps = score_plan(513, 1, 2752 * 3, 64, requested, ScorePath::Decode, false)?;
                for step in steps {
                    if step.tokens.len() <= 64 {
                        replay(step.tokens.len(), false, step.path == ScorePath::Prefill, step.tokens.end)?;
                    }
                }
            }
            for active in 1..=16 {
                for optional in [false, true] {
                    replay(active, false, optional, prompt + 1)?;
                    for rows in active.max(2)..=(active * 8).min(64) {
                        replay(rows, true, optional, prompt + 8)?;
                    }
                }
            }
        }
        Ok(())
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
