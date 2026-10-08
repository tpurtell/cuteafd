//! Benchmark probes: per-request diagnostics the in-server benchmark attaches
//! to its own loopback requests.
//!
//! The benchmark registers a [`ProbeSpec`] in the process-wide [`registry`] and
//! sends the request with the `x-cuteafd-probe: <id>` header. The chat handler
//! claims the probe (an id is good for one request) and hands it to the engine
//! in [`super::NativeRequest::probe`]; the engine honours the switches and
//! records what was asked for into the shared [`ProbeRecord`], which the
//! benchmark reads once the response is complete. Unknown ids are ignored, so
//! the header does nothing for an ordinary client.
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, Mutex, OnceLock};

mod rows;

/// The request header naming a registered probe.
pub const HEADER: &str = "x-cuteafd-probe";

/// What the engine should do differently for one request.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ProbeSpec {
    /// No prefix-cache lookup and nothing retained: a cold prefill.
    #[serde(default)]
    pub cold: bool,
    /// Decode one token per step (no drafts of any kind).
    #[serde(default)]
    pub no_speculation: bool,
    /// Token ids to run instead of tokenizing the rendered prompt.
    #[serde(default)]
    pub prompt_ids: Option<Vec<u32>>,
    /// Already-expanded native image spans, verified against prepared sources by the engine.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub media: Vec<ProbeMedia>,
    /// Already-expanded audio spans bound to ordinary input_audio sources.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub audio: Vec<ProbeAudio>,
    /// Teacher-forced scoring: run the prompt and record the logits row
    /// predicting every prompt token from this index on; the request then
    /// ends without generating.
    #[serde(default)]
    pub score_from: Option<usize>,
    /// Decode-shaped scoring width, bounded by the family's verify capacity.
    #[serde(default)]
    pub verify_rows: Option<usize>,
    /// MiMo/GLM Flash/Qwen cold replay, reproducing source prefill/decode geometry.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub cold_steps: Vec<ProbeColdStep>,
    /// Scoring kernel shape: `decode` or `prefill`. When absent, retain the
    /// family's legacy path (V4.1 prefill-shaped, other families decode-shaped).
    #[serde(default)]
    pub score_path: Option<String>,
    /// New server-local directory for streamed full-vocabulary F32 log-probs.
    /// Each row is a safetensors `log_probs` tensor; `manifest.jsonl` identifies
    /// its predicted-token position, vocabulary size and file. Existing paths
    /// are rejected rather than overwritten. No dump is written when absent.
    #[serde(default)]
    pub dump_rows: Option<PathBuf>,
    /// Record (and dump) only the rows predicting these positions: a scorer whose reference
    /// holds a subsample of a window's rows. Absent, every row is recorded.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub score_positions: Option<Vec<usize>>,
    /// Record the row the first generated token is selected from.
    #[serde(default)]
    pub record_first: bool,
    /// Record the rows the first N generated tokens are selected from (the
    /// first from the prefill or a retained row, the rest from decode steps).
    #[serde(default)]
    pub record_rows: usize,
    /// Top entries kept per recorded row.
    #[serde(default)]
    pub top_k: usize,
    /// Per predicted position: token ids whose log-probability to report
    /// (a reference's top-k, so KL can be estimated against it).
    #[serde(default)]
    pub want: HashMap<usize, Vec<u32>>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct ProbeColdStep {
    pub end: usize,
    #[serde(default)]
    pub decode: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct ProbeFixture {
    pub path: String,
    pub sha256: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct ProbeImageUrl {
    pub url: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub detail: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct ProbeMedia {
    pub start: usize,
    pub len: usize,
    pub kind: String,
    pub key: String,
    pub grid: [u32; 3],
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fixture: Option<ProbeFixture>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub image_url: Option<ProbeImageUrl>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct ProbeAudio {
    pub start: usize,
    pub len: usize,
    pub key: String,
    pub samples: usize,
    /// SHA256 of canonical finite mono F32 LE PCM, independent of encoder identity.
    pub pcm_sha256: String,
}

pub fn sha256_hex(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|c| c.is_ascii_digit() || (b'a'..=b'f').contains(&c))
}

impl ProbeSpec {
    pub fn validate_cold_steps(&self, len: usize, prefill_rows: usize, decode_rows: usize) -> anyhow::Result<()> {
        if self.cold_steps.is_empty() { return Ok(()); }
        anyhow::ensure!(self.cold && self.no_speculation && self.score_from.is_none(),
            "cold_steps requires cold non-speculative generation");
        anyhow::ensure!(self.cold_steps.len() <= len, "too many cold_steps");
        let mut previous = 0;
        for step in &self.cold_steps {
            let capacity = if step.decode { decode_rows } else { prefill_rows };
            anyhow::ensure!(step.end > previous && step.end <= len && step.end - previous <= capacity,
                "cold_steps must cover positive bounded chunks within the prompt");
            previous = step.end;
        }
        anyhow::ensure!(previous == len && !self.cold_steps.last().unwrap().decode,
            "cold_steps must end with a prefill at the prompt length");
        Ok(())
    }

    /// Remote callers may choose only a new leaf under an explicitly enabled root.
    pub fn constrain_dump_root(&mut self, root: &std::path::Path) -> anyhow::Result<()> {
        let Some(path) = self.dump_rows.as_ref() else { return Ok(()); };
        let root = root.canonicalize()?;
        anyhow::ensure!(root.is_dir(), "probe dump root must be an existing directory");
        let leaf = path.file_name().filter(|name| !name.is_empty()).ok_or_else(|| anyhow::anyhow!("dump needs a new leaf"))?;
        anyhow::ensure!(!path.components().any(|c| matches!(c, std::path::Component::ParentDir)), "dump traversal forbidden");
        let path = if path.is_absolute() { path.clone() } else { root.join(path) };
        let parent = path.parent().ok_or_else(|| anyhow::anyhow!("dump parent required"))?.canonicalize()?;
        anyhow::ensure!(parent.starts_with(&root), "dump parent escapes configured root");
        let destination = parent.join(leaf);
        anyhow::ensure!(destination != root && std::fs::symlink_metadata(&destination).is_err_and(|e| e.kind() == std::io::ErrorKind::NotFound),
            "dump leaf must not exist");
        self.dump_rows = Some(destination);
        Ok(())
    }

    pub fn validate_audio(&self) -> anyhow::Result<()> {
        if self.audio.is_empty() { return Ok(()); }
        anyhow::ensure!(self.media.is_empty(), "mixed expanded image/audio probes are not supported");
        let tokens = self.prompt_ids.as_ref().ok_or_else(|| anyhow::anyhow!("probe audio requires prompt_ids"))?;
        anyhow::ensure!(self.audio.len() <= cuteafd_loader::media::audio::MAX_CLIPS, "too many probe audio clips");
        let mut previous = 0;
        let mut samples = 0usize;
        for span in &self.audio {
            let geometry = cuteafd_loader::media::audio::AudioGeometry::for_samples(span.samples)?;
            let end = span.start.checked_add(span.len).ok_or_else(|| anyhow::anyhow!("probe audio extent overflow"))?;
            anyhow::ensure!(span.len == geometry.tokens && span.start >= previous && end <= tokens.len()
                && self.score_from.is_none_or(|from| end <= from), "probe audio spans must match geometry and precede scoring");
            anyhow::ensure!(sha256_hex(&span.key) && sha256_hex(&span.pcm_sha256), "invalid probe audio identity");
            samples = samples.checked_add(span.samples).ok_or_else(|| anyhow::anyhow!("probe audio sample overflow"))?;
            anyhow::ensure!(samples <= cuteafd_loader::media::audio::MAX_REQUEST_SAMPLES, "probe audio history exceeds sample bound");
            previous = end;
        }
        Ok(())
    }

    pub fn validate_media(&self) -> anyhow::Result<()> {
        self.validate_audio()?;
        if self.media.is_empty() { return Ok(()); }
        let tokens = self.prompt_ids.as_ref().ok_or_else(|| anyhow::anyhow!("probe media requires prompt_ids"))?;
        anyhow::ensure!(self.media.len() <= 128, "probe media exceeds history limit");
        let mut previous = 0;
        for span in &self.media {
            let end = span.start.checked_add(span.len).ok_or_else(|| anyhow::anyhow!("probe media extent overflow"))?;
            let [t, h, w] = span.grid;
            anyhow::ensure!(span.kind == "image" && span.len > 0 && span.start >= previous && end <= tokens.len(),
                "probe media spans must be sorted, disjoint and inside prompt_ids");
            anyhow::ensure!(sha256_hex(&span.key) && t == 1 && h > 0 && w > 0 && h % 2 == 0 && w % 2 == 0
                && u64::from(h) * u64::from(w) / 4 == span.len as u64, "invalid probe media identity/grid");
            let source = span.image_url.as_ref().ok_or_else(|| anyhow::anyhow!("probe media image_url required"))?;
            anyhow::ensure!(!source.url.is_empty() && source.detail.as_deref().is_none_or(|v| matches!(v, "auto" | "high" | "low")),
                "invalid probe image source/detail");
            if let Some(fixture) = &span.fixture {
                anyhow::ensure!(sha256_hex(&fixture.sha256) && !fixture.path.is_empty() && !fixture.path.contains('\\')
                    && !std::path::Path::new(&fixture.path).is_absolute()
                    && fixture.path.split('/').all(|v| !v.is_empty() && v != "." && v != ".."), "invalid probe fixture identity");
            }
            previous = end;
        }
        Ok(())
    }
}

/// One recorded logits row (log-softmax over the full vocabulary).
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct ProbeRow {
    /// Index of the token this row predicts (the prompt length for the first
    /// generated token).
    pub position: usize,
    /// FNV-1a 64 of the raw f32 row bytes: equal rows are byte-identical.
    pub hash: String,
    pub argmax: u32,
    /// The row's `top_k` entries, most likely first.
    pub top: Vec<(u32, f32)>,
    /// Log-probabilities of the requested ids at this position.
    pub wanted: Vec<(u32, f32)>,
    /// Whether every logit was finite.
    pub finite: bool,
}

/// Server-observed V4.1 prefill scheduling, not inferred from SSE readiness.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ProbePrefillShare {
    pub decode_share: f64,
    pub parked_waves: u64,
    pub resumed_waves: u64,
    /// Successful decode/verify lane commits by other requests between this
    /// request's completed prefill waves (not emitted token count).
    pub interleaved_decode_steps: u64,
}

impl ProbePrefillShare {
    pub fn exercised(&self) -> bool {
        self.decode_share > 0.0 && self.decode_share < 1.0
            && self.resumed_waves >= 1 && self.interleaved_decode_steps >= 1
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ProbeRecord {
    /// Whether an engine honoured the probe at all.
    pub engine: Option<String>,
    pub prompt_ids: Vec<u32>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub media: Vec<ProbeMedia>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub audio: Vec<ProbeAudio>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub provenance: Option<serde_json::Value>,
    pub cached_tokens: usize,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub cold_steps: Vec<ProbeColdStep>,
    pub rows: Vec<ProbeRow>,
    pub generated: Vec<u32>,
    /// Speculation actually skipped / cache actually bypassed, as the engine saw it.
    pub cold: bool,
    pub no_speculation: bool,
    pub scored: usize,
    /// The scoring path actually selected by the engine, not just requested.
    #[serde(default)]
    pub score_path: Option<String>,
    pub error: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prefill_share: Option<ProbePrefillShare>,
}

/// A probe shared by the benchmark and the engine serving its request.
#[derive(Debug)]
pub struct Probe {
    pub spec: ProbeSpec,
    record: Mutex<ProbeRecord>,
    dump: Mutex<Option<rows::RowDump>>,
    /// `spec.score_positions` as a set.
    positions: Option<std::collections::HashSet<usize>>,
}

impl Probe {
    pub fn new(spec: ProbeSpec) -> Arc<Self> {
        let dump = spec.dump_rows.as_ref().map(|path| rows::RowDump::new(path.clone()));
        let positions = spec.score_positions.as_ref().map(|positions| positions.iter().copied().collect());
        Arc::new(Self { spec, record: Mutex::new(ProbeRecord::default()), dump: Mutex::new(dump), positions })
    }

    /// Whether the row predicting `position` is recorded (`score_positions`); an engine may skip
    /// computing the others.
    pub fn wants(&self, position: usize) -> bool {
        self.positions.as_ref().is_none_or(|set| set.contains(&position))
    }

    fn with(&self, f: impl FnOnce(&mut ProbeRecord)) {
        if let Ok(mut record) = self.record.lock() {
            f(&mut record);
        }
    }

    /// The engine took the request: its name, the prompt ids it runs and the
    /// rows restored from the prefix cache.
    pub fn admitted(&self, engine: &str, prompt_ids: &[u32], cached_tokens: usize) {
        self.with(|r| {
            r.engine = Some(engine.to_owned());
            r.prompt_ids = prompt_ids.to_vec();
            r.cached_tokens = cached_tokens;
            if matches!(engine, "mimo_v2" | "glm5_flash" | "qwen4") { r.cold_steps = self.spec.cold_steps.clone(); }
            r.cold = self.spec.cold;
            r.no_speculation = self.spec.no_speculation;
        });
    }

    /// Echo verified descriptors only, never the image's data URL.
    pub fn media(&self, mut media: Vec<ProbeMedia>) {
        for span in &mut media { span.image_url = None; }
        self.with(|r| r.media = media);
    }

    pub fn audio(&self, audio: Vec<ProbeAudio>) {
        self.with(|r| r.audio = audio);
    }

    pub fn provenance(&self, value: serde_json::Value) {
        self.with(|r| r.provenance = Some(value));
    }

    pub fn prefill_policy(&self, decode_share: f64) {
        self.with(|r| r.prefill_share = Some(ProbePrefillShare { decode_share, ..Default::default() }));
    }

    pub fn prefill_parked(&self) {
        self.with(|r| { if let Some(p) = &mut r.prefill_share { p.parked_waves += 1; } });
    }

    pub fn prefill_resumed(&self, decode_steps: u64) {
        self.with(|r| {
            if let Some(p) = &mut r.prefill_share {
                p.resumed_waves += 1;
                p.interleaved_decode_steps += decode_steps;
            }
        });
    }

    /// Records one host logits row predicting token `position` (unless `score_positions` leaves it out).
    pub fn row(&self, position: usize, logits: &[f32]) {
        if !self.wants(position) {
            return;
        }
        let want = self.spec.want.get(&position).map(Vec::as_slice).unwrap_or(&[]);
        let softmax = LogSoftmax::new(logits);
        let row = summarize_with(position, logits, self.spec.top_k.max(1), want, &softmax);
        if self.spec.dump_rows.is_some() {
            let result = self.dump.lock().map_err(|_| "row dump lock poisoned".to_owned()).and_then(|mut dump| {
                dump.as_mut().expect("dump configured").write(position, logits, &softmax).map_err(|e| e.to_string())
            });
            if let Err(error) = result {
                self.fail(format!("dump rows: {error}"));
            }
        }
        self.with(|r| {
            r.rows.push(row);
            if self.spec.score_from.is_some_and(|from| position >= from) {
                r.scored += 1;
            }
        });
    }

    /// Records a generated token.
    pub fn token(&self, token: u32) {
        self.with(|r| r.generated.push(token));
    }

    pub fn fail(&self, message: impl Into<String>) {
        let message = message.into();
        self.with(|r| { r.error.get_or_insert(message); });
    }

    pub fn record(&self) -> ProbeRecord {
        self.record.lock().map(|r| r.clone()).unwrap_or_default()
    }

    pub fn selected_score_path(&self, path: &str) {
        self.with(|r| r.score_path = Some(path.to_owned()));
    }

    /// The scoring request's rows: positions `from..len` of the prompt.
    pub fn scoring(&self) -> Option<usize> {
        self.spec.score_from
    }
}

/// FNV-1a 64 over bytes.
pub fn fnv1a(bytes: &[u8]) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    for &byte in bytes {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    hash
}

/// Log-softmax summary of one logits row.
pub fn summarize(position: usize, logits: &[f32], top_k: usize, want: &[u32]) -> ProbeRow {
    summarize_with(position, logits, top_k, want, &LogSoftmax::new(logits))
}

struct LogSoftmax(f64);

impl LogSoftmax {
    fn new(logits: &[f32]) -> Self {
        let max = logits.iter().copied().filter(|v| v.is_finite()).fold(f32::NEG_INFINITY, f32::max);
        let sum: f64 = logits.iter().filter(|v| v.is_finite()).map(|&v| f64::from(v - max).exp()).sum();
        Self(f64::from(max) + sum.ln())
    }

    fn at(&self, value: f32) -> f32 {
        if value.is_finite() { (f64::from(value) - self.0) as f32 } else { f32::NEG_INFINITY }
    }
}

fn summarize_with(position: usize, logits: &[f32], top_k: usize, want: &[u32], softmax: &LogSoftmax) -> ProbeRow {
    let mut bytes = Vec::with_capacity(logits.len() * 4);
    for value in logits {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    let hash = format!("{:016x}", fnv1a(&bytes));
    let finite = logits.iter().all(|v| v.is_finite());
    let lp = |v| softmax.at(v);
    // Partial selection of the top entries (ties to the lower id).
    let mut order: Vec<u32> = (0..logits.len() as u32).collect();
    let k = top_k.min(order.len());
    let key = |&i: &u32| (std::cmp::Reverse(OrderedF32(logits[i as usize])), i);
    if k > 0 && k < order.len() {
        order.select_nth_unstable_by_key(k - 1, key);
        order.truncate(k);
    }
    order.sort_by_key(key);
    let top: Vec<(u32, f32)> = order.iter().map(|&i| (i, lp(logits[i as usize]))).collect();
    let argmax = top.first().map_or(0, |&(i, _)| i);
    let wanted = want.iter().filter(|&&i| (i as usize) < logits.len()).map(|&i| (i, lp(logits[i as usize]))).collect();
    ProbeRow { position, hash, argmax, top, wanted, finite }
}

/// Total order on f32 for selection (NaN lowest).
#[derive(Debug, Clone, Copy, PartialEq)]
struct OrderedF32(f32);
impl Eq for OrderedF32 {}
impl PartialOrd for OrderedF32 {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for OrderedF32 {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        let key = |v: f32| if v.is_nan() { f32::NEG_INFINITY } else { v };
        key(self.0).total_cmp(&key(other.0))
    }
}

/// Probes registered by the benchmark and not yet claimed by a request.
#[derive(Default)]
pub struct ProbeRegistry {
    pending: Mutex<HashMap<String, Arc<Probe>>>,
}

impl ProbeRegistry {
    /// Registers `spec`; the returned id goes in the [`HEADER`] of one request.
    pub fn register(&self, spec: ProbeSpec) -> (String, Arc<Probe>) {
        let id = uuid::Uuid::new_v4().simple().to_string();
        let probe = Probe::new(spec);
        if let Ok(mut pending) = self.pending.lock() {
            // A probe whose request never arrived is dropped with its owner.
            pending.retain(|_, p| Arc::strong_count(p) > 1);
            pending.insert(id.clone(), probe.clone());
        }
        (id, probe)
    }

    /// Takes the probe registered under `id`, if any.
    pub fn claim(&self, id: &str) -> Option<Arc<Probe>> {
        self.pending.lock().ok()?.remove(id)
    }
}

pub fn registry() -> &'static ProbeRegistry {
    static REGISTRY: OnceLock<ProbeRegistry> = OnceLock::new();
    REGISTRY.get_or_init(ProbeRegistry::default)
}

#[cfg(test)]
mod tests {
    #[test]
    fn prefill_share_requires_successful_resume_and_other_decode_commits() {
        use super::*;
        let probe = Probe::new(ProbeSpec::default());
        probe.prefill_policy(0.2);
        probe.prefill_parked();
        probe.prefill_resumed(0);
        assert!(!probe.record().prefill_share.unwrap().exercised());
        probe.prefill_parked();
        probe.prefill_resumed(3);
        let record = probe.record();
        let evidence = record.prefill_share.as_ref().unwrap();
        assert_eq!(evidence.parked_waves, 2);
        assert_eq!(evidence.resumed_waves, 2);
        assert_eq!(evidence.interleaved_decode_steps, 3);
        assert!(evidence.exercised());
        let copy: ProbeRecord = serde_json::from_value(serde_json::to_value(record).unwrap()).unwrap();
        assert!(copy.prefill_share.unwrap().exercised());
        probe.prefill_policy(0.0);
        probe.prefill_resumed(3);
        assert!(!probe.record().prefill_share.unwrap().exercised());
    }

    #[test]
    fn image_probe_serialization_is_unchanged_when_audio_is_empty() {
        use super::*;
        let media = ProbeMedia { start: 2, len: 4, kind: "image".into(), key: "ab".repeat(32), grid: [1,4,4],
            fixture: None, image_url: None };
        assert_eq!(serde_json::to_string(&media).unwrap(), format!(
            "{{\"start\":2,\"len\":4,\"kind\":\"image\",\"key\":\"{}\",\"grid\":[1,4,4]}}", "ab".repeat(32)));
        let spec = ProbeSpec { media: vec![media], ..Default::default() };
        assert_eq!(serde_json::to_string(&spec).unwrap(), format!(concat!(
            "{{\"cold\":false,\"no_speculation\":false,\"prompt_ids\":null,\"media\":[",
            "{{\"start\":2,\"len\":4,\"kind\":\"image\",\"key\":\"{}\",\"grid\":[1,4,4]}}],",
            "\"score_from\":null,\"verify_rows\":null,\"score_path\":null,\"dump_rows\":null,",
            "\"record_first\":false,\"record_rows\":0,\"top_k\":0,\"want\":{{}}}}"), "ab".repeat(32)));
        assert!(serde_json::to_value(ProbeRecord::default()).unwrap().get("audio").is_none());
    }
    #[test]
    fn audio_probe_geometry_identity_and_history_fail_closed() {
        use super::*;
        let span = ProbeAudio { start: 2, len: 7, key: "ab".repeat(32), samples: 24000, pcm_sha256: "cd".repeat(32) };
        let spec = ProbeSpec { prompt_ids: Some(vec![0;12]), score_from: Some(10), audio: vec![span], ..Default::default() };
        spec.validate_audio().unwrap();
        for mutation in ["ids","len","short","huge","key","pcm","overlap","score","overflow","clips"] {
            let mut bad = spec.clone();
            match mutation {
                "ids" => bad.prompt_ids = None, "len" => bad.audio[0].len = 6,
                "short" => bad.audio[0].samples = 480, "huge" => bad.audio[0].samples = 7200001,
                "key" => bad.audio[0].key = "AB".repeat(32), "pcm" => bad.audio[0].pcm_sha256 = "bad".into(),
                "overlap" => bad.audio.push(bad.audio[0].clone()), "score" => bad.score_from = Some(8),
                "overflow" => bad.audio[0].start = usize::MAX, _ => bad.audio = vec![bad.audio[0].clone();5],
            }
            assert!(bad.validate_audio().is_err(), "{mutation}");
        }
    }
    use super::*;

    fn dumped_row(path: &std::path::Path) -> Vec<f32> {
        let bytes = std::fs::read(path).unwrap();
        let header_len = u64::from_le_bytes(bytes[..8].try_into().unwrap()) as usize;
        assert_eq!(header_len % 8, 0);
        let header: serde_json::Value = serde_json::from_slice(&bytes[8..8 + header_len]).unwrap();
        let tensor = &header["log_probs"];
        assert_eq!(tensor["dtype"], "F32");
        let vocab = tensor["shape"][0].as_u64().unwrap() as usize;
        assert_eq!(tensor["data_offsets"], serde_json::json!([0, vocab * 4]));
        let metadata = cuteafd_loader::read_safetensors_metadata(path).unwrap();
        assert_eq!(metadata.len(), 1, "the engine loader accepts the dump");
        assert_eq!(metadata[0].name, "log_probs");
        assert_eq!(metadata[0].shape, vec![vocab]);
        assert_eq!(metadata[0].byte_length, (vocab * 4) as u64);
        let data = &bytes[8 + header_len..];
        assert_eq!(data.len(), vocab * 4);
        data.chunks_exact(4).map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap())).collect()
    }

    #[test]
    fn registered_probe_streams_full_rows_matching_in_band_top_k() {
        let temporary = tempfile::tempdir().unwrap();
        let path = temporary.path().join("arm");
        let (id, probe) = registry().register(ProbeSpec {
            dump_rows: Some(path.clone()), score_from: Some(3), top_k: 3,
            want: HashMap::from([(3, vec![0, 4])]), ..ProbeSpec::default()
        });
        assert!(!path.exists(), "no IO until the first recorded row");
        let engine = registry().claim(&id).unwrap();
        engine.admitted("host-loopback", &[1, 2, 3, 4, 5], 0);
        for (position, logits) in [(3, [1.0, 3.0, 2.0, 3.0, -1.0]), (4, [-9.0, 0.0, 5.0, 2.0, 1.0])] {
            engine.row(position, &logits);
            assert_eq!(std::fs::read_to_string(path.join("manifest.jsonl")).unwrap().lines().count(), position - 2,
                "each complete row is published before the next step");
        }
        let record = probe.record();
        assert!(record.error.is_none(), "{:?}", record.error);
        assert_eq!(record.scored, 2);
        let manifest = std::fs::read_to_string(path.join("manifest.jsonl")).unwrap();
        for (line, row) in manifest.lines().zip(&record.rows) {
            let entry: serde_json::Value = serde_json::from_str(line).unwrap();
            assert_eq!(entry["position"], row.position);
            assert_eq!(entry["vocab_size"], 5);
            assert_eq!(entry["tensor"], "log_probs");
            assert_eq!(entry["byte_order"], "little");
            let values = dumped_row(&path.join(entry["file"].as_str().unwrap()));
            let mut ids: Vec<usize> = (0..values.len()).collect();
            ids.sort_by(|&a, &b| values[b].total_cmp(&values[a]).then(a.cmp(&b)));
            assert_eq!(row.top.iter().map(|&(id, _)| id as usize).collect::<Vec<_>>(), ids[..3]);
            for &(id, lp) in row.top.iter().chain(&row.wanted) {
                assert_eq!(values[id as usize].to_bits(), lp.to_bits());
            }
            assert!((values.iter().map(|&v| f64::from(v).exp()).sum::<f64>() - 1.0).abs() < 1e-6);
        }
        assert_eq!(std::fs::read_dir(&path).unwrap().count(), 3);
    }

    #[test]
    fn score_positions_select_the_recorded_and_dumped_rows() {
        let temporary = tempfile::tempdir().unwrap();
        let path = temporary.path().join("subset");
        let probe = Probe::new(ProbeSpec { dump_rows: Some(path.clone()), score_from: Some(1),
            score_positions: Some(vec![2, 4]), ..ProbeSpec::default() });
        assert!(!probe.wants(1) && probe.wants(2) && !probe.wants(3) && probe.wants(4));
        for position in 1..=4 {
            probe.row(position, &[position as f32, 0.5, -1.0]);
        }
        let record = probe.record();
        assert!(record.error.is_none(), "{:?}", record.error);
        assert_eq!(record.rows.iter().map(|r| r.position).collect::<Vec<_>>(), [2, 4]);
        assert_eq!(record.scored, 2);
        let manifest = std::fs::read_to_string(path.join("manifest.jsonl")).unwrap();
        let positions: Vec<u64> = manifest.lines()
            .map(|line| serde_json::from_str::<serde_json::Value>(line).unwrap()["position"].as_u64().unwrap()).collect();
        assert_eq!(positions, [2, 4]);
        assert_eq!(dumped_row(&path.join("row-00000001.safetensors"))[0].to_bits(),
            record.rows[1].top[0].1.to_bits(), "position 4's row, its log-probabilities as recorded");
        // Absent, every row is wanted.
        assert!(Probe::new(ProbeSpec::default()).wants(7));
    }

    #[test]
    fn dump_rejects_existing_paths_and_vocabulary_changes_without_orphans() {
        let temporary = tempfile::tempdir().unwrap();
        let probe = Probe::new(ProbeSpec { dump_rows: Some(temporary.path().to_owned()), ..ProbeSpec::default() });
        probe.row(1, &[1.0, 2.0]);
        assert!(probe.record().error.unwrap().contains("dump rows"));
        assert_eq!(std::fs::read_dir(temporary.path()).unwrap().count(), 0);
        let path = temporary.path().join("new");
        let probe = Probe::new(ProbeSpec { dump_rows: Some(path.clone()), ..ProbeSpec::default() });
        probe.row(1, &[1.0, 2.0]);
        probe.row(2, &[1.0, 2.0, 3.0]);
        probe.row(3, &[1.0, 2.0]);
        assert!(probe.record().error.unwrap().contains("vocabulary changed"));
        assert_eq!(std::fs::read_to_string(path.join("manifest.jsonl")).unwrap().lines().count(), 1);
        assert_eq!(std::fs::read_dir(&path).unwrap().count(), 2);
    }

    #[test]
    fn cold_replay_requires_bounded_complete_non_speculative_geometry() {
        let mut spec = ProbeSpec { cold: true, no_speculation: true,
            cold_steps: vec![ProbeColdStep { end: 8, decode: false },
                ProbeColdStep { end: 9, decode: true }, ProbeColdStep { end: 12, decode: false }],
            ..Default::default() };
        assert!(spec.validate_cold_steps(12, 8, 2).is_ok());
        for engine in ["mimo_v2", "glm5_flash", "qwen4"] {
            let probe = Probe::new(spec.clone());
            probe.admitted(engine, &[0; 12], 0);
            assert_eq!(probe.record().cold_steps, spec.cold_steps);
        }
        let unsupported = Probe::new(spec.clone());
        unsupported.admitted("host-loopback", &[0; 12], 0);
        assert!(unsupported.record().cold_steps.is_empty());
        assert!(spec.validate_cold_steps(12, 7, 2).is_err());
        assert!(spec.validate_cold_steps(13, 8, 2).is_err());
        spec.cold_steps[1].end = 8;
        assert!(spec.validate_cold_steps(12, 8, 2).is_err());
        spec.cold_steps[1].end = 11;
        assert!(spec.validate_cold_steps(12, 8, 2).is_err());
        spec.cold_steps[1].end = 9;
        spec.cold_steps[2].decode = true;
        assert!(spec.validate_cold_steps(12, 8, 4).is_err());
        spec.cold_steps[2].decode = false;
        spec.cold = false;
        assert!(spec.validate_cold_steps(12, 8, 2).is_err());
        spec.cold = true; spec.no_speculation = false;
        assert!(spec.validate_cold_steps(12, 8, 2).is_err());
        spec.no_speculation = true; spec.score_from = Some(9);
        assert!(spec.validate_cold_steps(12, 8, 2).is_err());
        assert!(ProbeSpec::default().validate_cold_steps(0, 0, 0).is_ok());
    }

    #[test]
    fn probe_options_are_backward_compatible_and_round_trip() {
        let legacy: ProbeSpec = serde_json::from_str("{}").unwrap();
        assert!(legacy.dump_rows.is_none() && legacy.verify_rows.is_none());
        assert!(legacy.score_path.is_none());
        assert!(legacy.score_positions.is_none() && serde_json::to_value(&legacy).unwrap().get("score_positions").is_none());
        let configured: ProbeSpec = serde_json::from_value(serde_json::json!({ "dump_rows": "arm", "verify_rows": 5,
            "score_path": "decode", "score_positions": [3, 9] })).unwrap();
        assert_eq!(configured.score_positions.as_deref(), Some(&[3, 9][..]));
        assert_eq!(configured.dump_rows.as_deref(), Some(std::path::Path::new("arm")));
        assert_eq!(configured.verify_rows, Some(5));
        assert_eq!(configured.score_path.as_deref(), Some("decode"));
        let serialized = serde_json::to_value(configured).unwrap();
        assert_eq!(serialized["verify_rows"], 5);
        assert_eq!(serialized["score_path"], "decode");
        let probe = Probe::new(ProbeSpec::default());
        probe.selected_score_path("prefill");
        assert_eq!(probe.record().score_path.as_deref(), Some("prefill"));
    }

    #[test]
    fn remote_dump_destination_stays_under_an_existing_root_with_a_new_leaf() {
        let root = tempfile::tempdir().unwrap();
        let outside = tempfile::tempdir().unwrap();
        let check = |path: PathBuf| {
            let mut spec = ProbeSpec { dump_rows: Some(path), ..Default::default() };
            spec.constrain_dump_root(root.path()).map(|()| spec.dump_rows.unwrap())
        };
        assert_eq!(check("new".into()).unwrap(), root.path().canonicalize().unwrap().join("new"));
        for path in [root.path().to_owned(), outside.path().join("new"), "../escape".into(), "missing/new".into()] {
            assert!(check(path).is_err());
        }
        std::fs::write(root.path().join("exists"), "untouched").unwrap();
        assert!(check("exists".into()).is_err());
        #[cfg(unix)] {
            std::os::unix::fs::symlink(outside.path(), root.path().join("escape")).unwrap();
            std::os::unix::fs::symlink(outside.path().join("missing"), root.path().join("dangling")).unwrap();
            assert!(check("escape/new".into()).is_err());
            assert!(check("dangling".into()).is_err());
        }
        assert!(!root.path().join("new").exists());
    }

    #[test]
    fn summary_is_log_softmax_with_ordered_top() {
        let logits = [1.0f32, 3.0, 2.0, 3.0, -1.0];
        let row = summarize(7, &logits, 3, &[4, 9]);
        assert_eq!(row.position, 7);
        assert_eq!(row.argmax, 1, "ties go to the lower id");
        assert_eq!(row.top.iter().map(|t| t.0).collect::<Vec<_>>(), vec![1, 3, 2]);
        let total: f64 = logits.iter().map(|&v| f64::from(v).exp()).sum();
        let expect = (3.0 - total.ln()) as f32;
        assert!((row.top[0].1 - expect).abs() < 1e-6);
        assert_eq!(row.wanted.len(), 1, "out-of-vocabulary ids are skipped");
        assert!(row.finite);
        assert_eq!(row.hash, summarize(0, &logits, 1, &[]).hash);
        assert_ne!(row.hash, summarize(0, &[1.0, 3.0, 2.0, 3.0, -1.5], 1, &[]).hash);
    }

    #[test]
    fn registry_hands_a_probe_out_once() {
        let (id, probe) = registry().register(ProbeSpec { cold: true, ..ProbeSpec::default() });
        let claimed = registry().claim(&id).expect("registered");
        assert!(Arc::ptr_eq(&probe, &claimed));
        assert!(registry().claim(&id).is_none());
        claimed.admitted("test", &[1, 2, 3], 2);
        claimed.token(5);
        let record = probe.record();
        assert_eq!((record.prompt_ids.len(), record.cached_tokens, record.generated.as_slice(), record.cold),
            (3, 2, &[5u32][..], true));
    }
}
