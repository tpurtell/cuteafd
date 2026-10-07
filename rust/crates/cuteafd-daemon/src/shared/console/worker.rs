//! The console thread: turns worker events into page frames and snapshots.
//!
//! Wire format (JSON text frames on `/v1/console`):
//! - `{"type":"snapshot", now, text, config, requests, recent, g}` on connect
//!   and every 250 ms in `/v1/console/snapshot`; `config` is the family
//!   [`Layout`](super::Layout).
//! - `{"type":"frame", now, ev, g}` every 50 ms while a viewer is connected.
//!   Events `ev`: `admit`, `first`, `retire`, `prefill`, `round`, `layers`.
//!   A round is `{lane, shared, t0, t1, s: {stage: µs}, layers?: [µs],
//!   req: [[id, drafted, verified, accepted, emitted, masked, finished, segments?]]}`.
//! - Gauges `g`: totals, active, lanes, queued, prefilling, pending, kv, host,
//!   prefix, facts, layer_class.
use super::{totals, Event, Gauges, Layout, PrefillKind, Round, RoundRequest};
use cuteafd_api::openai::ConsoleHub;
use serde_json::{json, Map, Value};
use std::collections::{BTreeMap, HashMap, VecDeque};
use std::sync::mpsc::{Receiver, RecvTimeoutError};
use std::sync::Arc;
use std::time::{Duration, Instant};

const FRAME: Duration = Duration::from_millis(50);
const RECENT: usize = 16;

struct Request {
    lane: u8,
    prompt: u32,
    cached: u32,
    max: u32,
    grammar: bool,
    images: u16,
    admitted: f64,
    first: Option<f64>,
    generated: u32,
    /// Prompt rows prefilled or restored so far (while prefilling).
    done: u64,
    decoder: Option<cuteafd_loader::StreamingTokenDecoder>,
}

pub(super) struct Worker {
    hub: Arc<ConsoleHub>,
    layout: Layout,
    config: Value,
    base: Instant,
    requests: BTreeMap<u64, Request>,
    /// The most recent admission (prefill steps without a request id belong to it).
    latest: Option<u64>,
    recent: VecDeque<Value>,
    gauges: Map<String, Value>,
    dynamic: Value,
    dynamic_at: Instant,
    events: Vec<Value>,
    pieces: HashMap<u32, String>,
    frame_at: Instant,
    snapshot_at: Instant,
}

impl Worker {
    pub(super) fn new(hub: Arc<ConsoleHub>, layout: Layout, base: Instant) -> Self {
        let long_ago = base.checked_sub(Duration::from_secs(10)).unwrap_or(base);
        let config = layout.json(hub.text_enabled());
        Self { hub, layout, config, base, requests: BTreeMap::new(), latest: None, recent: VecDeque::new(),
            gauges: Map::new(), dynamic: Value::Null, dynamic_at: long_ago, events: Vec::new(), pieces: HashMap::new(),
            frame_at: base, snapshot_at: long_ago }
    }

    fn ms(&self, at: Instant) -> f64 {
        (at.saturating_duration_since(self.base).as_micros() as f64) / 1000.0
    }

    pub(super) fn run(mut self, receive: Receiver<Event>) {
        loop {
            match receive.recv_timeout(FRAME) {
                Ok(event) => {
                    self.handle(event);
                    while let Ok(event) = receive.try_recv() { self.handle(event); }
                }
                Err(RecvTimeoutError::Timeout) => {}
                Err(RecvTimeoutError::Disconnected) => return,
            }
            let viewers = self.hub.viewers() > 0;
            if !viewers {
                // Keep request text state bounded while nobody is watching.
                self.events.clear();
                for request in self.requests.values_mut() { request.decoder = None; }
            }
            if self.dynamic_at.elapsed() >= Duration::from_secs(1) {
                self.dynamic_at = Instant::now();
                if let Some(dynamic) = &self.layout.dynamic { self.dynamic = dynamic(); }
                self.update_layer_classes();
            }
            if viewers && self.frame_at.elapsed() >= FRAME {
                self.frame_at = Instant::now();
                let frame = json!({
                    "type": "frame", "now": self.ms(Instant::now()),
                    "ev": std::mem::take(&mut self.events),
                    "g": self.gauges(),
                });
                self.hub.publish(frame.to_string());
            }
            let period = if viewers { Duration::from_millis(250) } else { Duration::from_secs(2) };
            if self.snapshot_at.elapsed() >= period {
                self.snapshot_at = Instant::now();
                let snapshot = self.snapshot();
                self.hub.set_snapshot(snapshot.to_string());
            }
        }
    }

    /// The dynamic hook may name the layer classes (V4.1: the RTX/Spark split).
    fn update_layer_classes(&mut self) {
        if let Some(class) = self.dynamic.get("layer_class").cloned() {
            if let Some(layers) = self.config.get_mut("layers").filter(|layers| layers.is_object()) {
                layers["class"] = class;
            }
        }
    }

    fn gauges(&self) -> Value {
        let mut object = self.gauges.clone();
        object.insert("totals".into(), totals::snapshot());
        let tables = cuteafd_loader::mapped_table_stats();
        if !tables.is_empty() { object.insert("mapped_tables".into(), json!(tables)); }
        object.insert("active".into(), json!(self.requests.len()));
        object.insert("ids".into(), json!(self.requests.keys().collect::<Vec<_>>()));
        // The oldest admitted request still prefilling, and how many are.
        let prefilling: Vec<(&u64, &Request)> = self.requests.iter().filter(|(_, r)| r.first.is_none()).collect();
        object.insert("prefilling".into(), json!(prefilling.first().map(|(&id, r)| json!({"id": id,
            "done": r.done, "prompt": r.prompt, "cached": r.cached}))));
        if !object.contains_key("prefilling_n") {
            object.insert("prefilling_n".into(), json!(prefilling.len()));
        }
        if let Some(facts) = self.dynamic.get("facts") { object.insert("facts".into(), facts.clone()); }
        Value::Object(object)
    }

    fn snapshot(&self) -> Value {
        let requests: Vec<_> = self.requests.iter().map(|(&id, r)| json!({
            "id": id, "lane": r.lane, "prompt": r.prompt, "cached": r.cached, "max": r.max,
            "gen": r.generated, "grammar": r.grammar, "images": r.images,
            "admitted": r.admitted, "first": r.first,
        })).collect();
        // The config is built once at startup, so its `text` fact carries the
        // live value here: the bench override toggles without a page reload.
        let mut config = self.config.clone();
        if let Some(config) = config.as_object_mut() {
            config.insert("text".into(), json!(self.hub.text_enabled()));
        }
        json!({
            "type": "snapshot", "now": self.ms(Instant::now()), "text": self.hub.text_enabled(),
            "config": config, "requests": requests, "recent": self.recent,
            "g": self.gauges(),
        })
    }

    fn handle(&mut self, event: Event) {
        match event {
            Event::Admit { id, at, prompt, cached, max, lane, grammar, images } => {
                let admitted = self.ms(at);
                self.requests.insert(id, Request { lane, prompt, cached, max, grammar, images, admitted,
                    first: None, generated: 0, done: u64::from(cached), decoder: None });
                self.latest = Some(id);
                self.events.push(json!({"e": "admit", "id": id, "t": admitted, "prompt": prompt,
                    "cached": cached, "max": max, "lane": lane, "grammar": grammar, "images": images}));
            }
            Event::First { id, at, token } => {
                let t = self.ms(at);
                let text = self.hub.text_enabled() && self.hub.viewers() > 0;
                let mut piece = None;
                if let Some(request) = self.requests.get_mut(&id) {
                    request.first = Some(t);
                    request.generated = 1;
                    if text { piece = Some(Self::step(&self.layout, request, token, false)); }
                }
                let ttft = self.requests.get(&id).map(|r| t - r.admitted);
                self.events.push(json!({"e": "first", "id": id, "t": t, "ttft": ttft,
                    "text": piece.map(|p| json!([["t", p]]))}));
            }
            Event::Retire { id, at, reason, generated } => {
                let t = self.ms(at);
                if let Some(request) = self.requests.remove(&id) {
                    let generated = generated.max(request.generated);
                    let record = json!({"id": id, "t": t, "reason": reason, "prompt": request.prompt,
                        "cached": request.cached, "gen": generated, "admitted": request.admitted,
                        "first": request.first, "grammar": request.grammar});
                    self.recent.push_front(record);
                    self.recent.truncate(RECENT);
                }
                self.events.push(json!({"e": "retire", "id": id, "t": t, "reason": reason, "gen": generated}));
            }
            Event::Prefill(step) => {
                let id = step.id.or(self.latest);
                if matches!(step.kind, PrefillKind::Chunk | PrefillKind::Single | PrefillKind::Continuation) {
                    if let Some(request) = id.and_then(|id| self.requests.get_mut(&id)) {
                        request.done += u64::from(step.rows);
                    }
                }
                self.events.push(json!({"e": "prefill", "id": id, "k": step.kind.name(),
                    "lane": step.lane, "i": step.index, "n": step.of, "rows": step.rows,
                    "t0": self.ms(step.started), "t1": self.ms(step.finished)}));
            }
            Event::Gauges(gauges) => self.set_gauges(gauges),
            Event::LayerClasses(class) => {
                if let Some(layers) = self.config.get_mut("layers").filter(|layers| layers.is_object()) {
                    layers["class"] = json!(class);
                }
                self.events.push(json!({"e": "layers", "class": class}));
            }
            Event::Round(round) => self.round(round),
        }
    }

    fn set_gauges(&mut self, gauges: Gauges) {
        let pending = gauges.pending.map(Value::from).or_else(|| self.gauges.get("pending").cloned());
        let mut object = Map::new();
        object.insert("lanes".into(), json!(gauges.lanes));
        object.insert("queued".into(), json!(gauges.queued));
        if let Some(pending) = pending { object.insert("pending".into(), pending); }
        if let Some(prefilling) = gauges.prefilling { object.insert("prefilling_n".into(), json!(prefilling)); }
        if let Some(kv) = gauges.kv {
            object.insert("kv".into(), json!({"pages": kv.pages, "free": kv.free, "active": kv.active,
                "tokens_per_page": kv.tokens_per_page}));
        }
        if let Some(host) = gauges.host { object.insert("host".into(), host); }
        if let Some(prefix) = gauges.prefix { object.insert("prefix".into(), prefix); }
        self.gauges = object;
    }

    fn round(&mut self, round: Round) {
        let text = self.hub.text_enabled() && self.hub.viewers() > 0;
        let mut rows = Vec::with_capacity(round.requests.len());
        for member in &round.requests {
            let mut row = json!([member.id, member.drafted, member.verified, member.accepted,
                member.emitted, member.masked as u8, member.finished as u8]);
            if let Some(request) = self.requests.get_mut(&member.id) {
                request.lane = round.lane;
                request.generated += u32::from(member.emitted);
                if text && !member.emissions.is_empty() {
                    let segments = Self::segments(&self.layout, &mut self.pieces, request, member);
                    row.as_array_mut().unwrap().push(segments);
                }
            }
            rows.push(row);
        }
        let round_us = |v: f32| if v.is_finite() { json!((f64::from(v) * 10.0).round() / 10.0) } else { Value::Null };
        let stages: Map<String, Value> = round.stages.iter().filter(|(_, v)| v.is_finite())
            .map(|&(key, v)| (key.to_string(), round_us(v))).collect();
        let mut event = json!({
            "e": "round", "lane": round.lane, "shared": round.shared,
            "t0": self.ms(round.started), "t1": self.ms(round.finished),
            "s": stages, "req": rows,
        });
        if !round.layer_us.is_empty() {
            event["layers"] = Value::Array(round.layer_us.iter().map(|&v| round_us(v)).collect());
        }
        self.events.push(event);
    }

    /// Stream-decode one emitted token for its request.
    fn step(layout: &Layout, request: &mut Request, token: u32, finish: bool) -> String {
        let mut out = String::new();
        if request.decoder.is_none() {
            request.decoder = cuteafd_loader::streaming_token_decoder(&layout.snapshot, false).ok();
        }
        let Some(decoder) = request.decoder.as_mut() else { return out };
        let eos = layout.eos.contains(&token);
        if eos {
            out.push('␄');
        } else if let Ok(Some(text)) = decoder.step(token) {
            out.push_str(&text);
        }
        if finish || eos {
            if let Ok(Some(text)) = decoder.finish() { out.push_str(&text); }
            request.decoder = None;
        }
        out
    }

    /// One draft token decoded on its own; partial UTF-8 shows as U+FFFD.
    fn piece(layout: &Layout, pieces: &mut HashMap<u32, String>, token: u32) -> String {
        if let Some(piece) = pieces.get(&token) { return piece.clone(); }
        let piece = cuteafd_loader::streaming_token_decoder(&layout.snapshot, false).ok()
            .map(|mut decoder| {
                let first = decoder.step(token).ok().flatten().unwrap_or_default();
                let rest = decoder.finish().ok().flatten().unwrap_or_default();
                first + &rest
            }).unwrap_or_default();
        if pieces.len() < 65536 { pieces.insert(token, piece.clone()); }
        piece
    }

    /// Text segments of one request's round: `a` accepted drafts, `t` the
    /// target's token, `r` verified drafts that were rejected or discarded,
    /// `s` drafts that were never verified.
    fn segments(layout: &Layout, pieces: &mut HashMap<u32, String>, request: &mut Request,
        member: &RoundRequest) -> Value {
        let accepted = usize::from(member.accepted).min(member.emissions.len());
        let mut segments = Vec::new();
        let mut accepted_text = String::new();
        for (index, &token) in member.emissions.iter().enumerate() {
            let last = index + 1 == member.emissions.len();
            let piece = Self::step(layout, request, token, last && member.finished);
            if index < accepted { accepted_text.push_str(&piece); }
            else {
                if !accepted_text.is_empty() { segments.push(json!(["a", std::mem::take(&mut accepted_text)])); }
                segments.push(json!(["t", piece]));
            }
        }
        if !accepted_text.is_empty() { segments.push(json!(["a", accepted_text])); }
        let verified = usize::from(member.verified).min(member.proposal.len());
        let rejected: String = member.proposal.get(accepted..verified).unwrap_or_default().iter()
            .map(|&token| Self::piece(layout, pieces, token)).collect();
        if !rejected.is_empty() { segments.push(json!(["r", rejected])); }
        let skipped: String = member.proposal.get(verified..).unwrap_or_default().iter()
            .map(|&token| Self::piece(layout, pieces, token)).collect();
        if !skipped.is_empty() { segments.push(json!(["s", skipped])); }
        Value::Array(segments)
    }
}

#[cfg(test)]
mod tests {
    use super::super::{Kv, Prefill, Round, RoundRequest};
    use super::*;

    fn worker() -> Worker {
        let mut layout = Layout::new("test", "org/model".into(), "/nonexistent".into());
        layout.dynamic = Some(Box::new(|| json!({"facts": [["Policy", "fixed", "-"]], "layer_class": [1, 0]})));
        layout.layers = Some(super::super::Layers { title: "l".into(), first: 0, classes: Vec::new(), class: vec![0, 0] });
        Worker::new(ConsoleHub::new(false), layout, Instant::now())
    }

    #[test]
    fn rounds_carry_stages_and_count_generated_tokens() {
        let mut worker = worker();
        let now = Instant::now();
        worker.handle(Event::Admit { id: 3, at: now, prompt: 100, cached: 40, max: 50, lane: 0, grammar: true,
            images: 0 });
        worker.handle(Event::Prefill(Prefill { id: Some(3), kind: PrefillKind::Chunk, lane: 0, index: 0, of: 1,
            rows: 60, started: now, finished: now }));
        assert_eq!(worker.gauges()["prefilling"]["done"], 100);
        worker.handle(Event::First { id: 3, at: now, token: 5 });
        assert!(worker.gauges()["prefilling"].is_null());
        worker.handle(Event::Round(Round { lane: 0, shared: false, started: now, finished: now,
            stages: vec![("verify", 12.34), ("missing", f32::NAN)], layer_us: Vec::new(),
            requests: vec![RoundRequest { id: 3, drafted: 4, verified: 2, accepted: 1, emitted: 2, masked: true,
                finished: false, proposal: Vec::new(), emissions: Vec::new() }] }));
        let round = worker.events.last().unwrap();
        assert_eq!(round["s"], json!({"verify": 12.3}));
        assert!(round.get("layers").is_none());
        assert_eq!(round["req"][0], json!([3, 4, 2, 1, 2, 1, 0]));
        worker.handle(Event::Retire { id: 3, at: now, reason: "finished", generated: 0 });
        assert_eq!(worker.recent[0]["gen"], 3);
    }

    #[test]
    fn snapshots_carry_the_live_bench_text_override() {
        let worker = worker();
        assert_eq!(worker.snapshot()["text"], false);
        assert_eq!(worker.snapshot()["config"]["text"], false);
        worker.hub.set_bench_active(true);
        assert_eq!(worker.snapshot()["text"], true);
        assert_eq!(worker.snapshot()["config"]["text"], true);
        worker.hub.set_bench_active(false);
        assert_eq!(worker.snapshot()["text"], false);
        assert_eq!(worker.snapshot()["config"]["text"], false);
    }

    #[test]
    fn gauges_and_dynamic_facts_reach_the_page() {
        let mut worker = worker();
        worker.handle(Event::Gauges(Gauges { lanes: vec![2], queued: 1, prefilling: Some(1), pending: None,
            kv: Some(Kv { pages: 10, free: 4, active: 3, tokens_per_page: 256 }), host: None,
            prefix: Some(json!({"hits": 1})) }));
        worker.dynamic = (worker.layout.dynamic.as_ref().unwrap())();
        worker.update_layer_classes();
        let gauges = worker.gauges();
        assert_eq!(gauges["kv"]["tokens_per_page"], 256);
        assert_eq!(gauges["prefilling_n"], 1);
        assert_eq!(gauges["facts"][0][1], "fixed");
        assert_eq!(worker.snapshot()["config"]["layers"]["class"], json!([1, 0]));
    }
}
