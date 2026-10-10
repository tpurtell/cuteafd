//! Producer side of the live engine console (`GET /` on the API port), shared
//! by every family.
//!
//! Cost contract for the CUDA worker:
//! - No viewer connected: every per-round hook is one relaxed load of the
//!   hub's viewer count (through [`live`]) plus the always-on lifetime totals
//!   in [`totals`], a handful of relaxed atomic adds per round. Request
//!   lifecycle events (admit, first token, retire) are rare and always sent
//!   while a console is installed, so a page that connects mid-generation
//!   starts from a correct request list.
//! - Viewers connected: a hook builds one small owned event and `try_send`s it
//!   into a bounded channel. It never blocks, never serializes and never
//!   decodes text; a full channel drops the event.
//!
//! The console thread ([`worker`]) owns everything else: per-request state,
//! token text decoding, JSON encoding, the 50 ms frame cadence and the
//! snapshot a newly connected page starts from.
//!
//! What a family shows is declared once in its [`Layout`]: header facts, the
//! speculator, the lane count, the named "pipeline micro-step" stages its
//! rounds report and, optionally, a per-layer profile. The page renders
//! whatever the layout declares. Generic families feed it through
//! [`Ticket`] (one per request) and [`Step`] (one per decode round).
use cuteafd_api::openai::ConsoleHub;
use serde_json::Value;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc::{sync_channel, SyncSender};
use std::sync::{Arc, OnceLock};
use std::time::Instant;

mod layout;
mod worker;
pub(crate) use layout::{layer_class, Color, Layers, Layout, Speculator, StepGroup};

const CHANNEL: usize = 8192;
const GAUGES_MS: u64 = 250;

struct Producer {
    hub: Arc<ConsoleHub>,
    events: SyncSender<Event>,
    base: Instant,
    /// Milliseconds since `base` of the last gauge event.
    gauges_at: AtomicU64,
}

static PRODUCER: OnceLock<Producer> = OnceLock::new();

/// A handle that exists only while at least one console viewer is connected.
#[derive(Clone, Copy)]
pub(crate) struct Live(&'static Producer);

/// The console handle if a viewer is connected; `None` costs one atomic load.
#[inline]
pub(crate) fn live() -> Option<Live> {
    let producer = PRODUCER.get()?;
    (producer.hub.viewers() > 0).then_some(Live(producer))
}

impl Live {
    /// Whether token ids should be captured for the text view: only while a
    /// viewer that may receive text (cookie-unlocked, or a bench run) is connected.
    #[inline]
    pub fn text(self) -> bool { self.0.hub.text_wanted() }
    #[inline]
    pub fn push(self, event: Event) { let _ = self.0.events.try_send(event); }
    /// True at most every 250 ms; the caller then pushes a [`Gauges`] event.
    pub fn gauges_due(self) -> bool {
        let now = self.0.base.elapsed().as_millis() as u64;
        let last = self.0.gauges_at.load(Ordering::Relaxed);
        now.saturating_sub(last) >= GAUGES_MS
            && self.0.gauges_at.compare_exchange(last, now, Ordering::Relaxed, Ordering::Relaxed).is_ok()
    }
}

/// Per-request lifecycle events are rare, so they are sent whenever the console
/// is installed. That keeps the snapshot's request list right for a page that
/// connects in the middle of a long generation.
pub(crate) fn lifecycle(event: Event) {
    if let Some(producer) = PRODUCER.get() { let _ = producer.events.try_send(event); }
}

pub(crate) enum Event {
    Round(Round),
    Prefill(Prefill),
    Admit { id: u64, at: Instant, prompt: u32, cached: u32, max: u32, lane: u8, grammar: bool, images: u16 },
    First { id: u64, at: Instant, token: u32 },
    Retire { id: u64, at: Instant, reason: &'static str, generated: u32 },
    Gauges(Gauges),
    /// The layer profile's class of every layer, once placement knows it.
    LayerClasses(Vec<u8>),
}

/// One completed decode round of a lane: its wall time, the family's named
/// stage times and every member request's outcome.
pub(crate) struct Round {
    pub lane: u8,
    /// Another lane ran concurrently (V4.1's two decode lanes).
    pub shared: bool,
    pub started: Instant,
    pub finished: Instant,
    /// Microseconds per stage key the layout declares (non-finite: not measured).
    pub stages: Vec<(&'static str, f32)>,
    /// Per-layer microseconds for the layer profile (NaN: not measured).
    pub layer_us: Vec<f32>,
    pub requests: Vec<RoundRequest>,
}

pub(crate) struct RoundRequest {
    pub id: u64,
    /// Draft tokens proposed before grammar and length-policy truncation.
    pub drafted: u8,
    /// Draft rows the target verified.
    pub verified: u8,
    /// Draft rows accepted; the next emitted token is the target's own.
    pub accepted: u8,
    pub emitted: u8,
    pub masked: bool,
    pub finished: bool,
    /// Text view only: the proposal after the anchor and the emitted ids.
    pub proposal: Vec<u32>,
    pub emissions: Vec<u32>,
}

#[derive(Clone, Copy)]
pub(crate) enum PrefillKind { Chunk, Single, Continuation, Replay, Restore }
impl PrefillKind {
    fn name(self) -> &'static str {
        match self {
            Self::Chunk => "chunk", Self::Single => "single", Self::Continuation => "continuation",
            Self::Replay => "replay", Self::Restore => "restore",
        }
    }
}

pub(crate) struct Prefill {
    /// The request; `None` attributes the step to the most recent admission.
    pub id: Option<u64>,
    pub kind: PrefillKind,
    pub lane: u8,
    pub index: u32,
    pub of: u32,
    pub rows: u32,
    pub started: Instant,
    pub finished: Instant,
}

impl Prefill {
    /// Record a prefill step that has just completed.
    pub fn done(kind: PrefillKind, lane: usize, index: usize, of: usize, rows: usize, started: Instant) {
        if let Some(live) = live() {
            live.push(Event::Prefill(Prefill { id: None, kind, lane: lane as u8, index: index as u32,
                of: of as u32, rows: rows as u32, started, finished: Instant::now() }));
        }
    }
}

/// Device KV pool occupancy in pages.
#[derive(Clone, Copy, Default)]
pub(crate) struct Kv {
    pub pages: u64,
    pub free: u64,
    /// Pages held by running requests (the rest of the used pages are retained snapshots).
    pub active: u64,
    pub tokens_per_page: u64,
}

#[derive(Default)]
pub(crate) struct Gauges {
    /// Decoding requests per lane.
    pub lanes: Vec<u8>,
    pub queued: u32,
    /// Admitted requests still prefilling (None: the console tracks them).
    pub prefilling: Option<u32>,
    pub pending: Option<bool>,
    pub kv: Option<Kv>,
    /// `cuteafd_hostcache::metrics::Snapshot` of the pinned host tier.
    pub host: Option<Value>,
    /// Device prefix cache counters (entries, hits, lookups, hit tokens) for
    /// families whose snapshots stay on the device.
    pub prefix: Option<Value>,
}

impl Gauges {
    /// Gauges of a generic family over the engine prefix cache, which also
    /// owns its device page pool.
    pub fn prefix_cache<E: cuteafd_hostcache::copy::CopyEngine>(cache: &cuteafd_engine::prefix::PrefixCache<E>,
        decoding: usize, prefilling: usize, queued: usize) -> Self {
        let stats = cache.stats();
        let used = stats.pages.saturating_sub(stats.pages_free);
        let kv = Kv { pages: stats.pages as u64, free: stats.pages_free as u64,
            active: used.saturating_sub(stats.pages_retained) as u64,
            tokens_per_page: cache.layout().page_rows as u64 };
        let prefix = serde_json::json!({"entries": stats.entries_prompt + stats.entries_turn,
            "hits": stats.hits, "lookups": stats.lookups, "hit_tokens": stats.hit_tokens,
            "enabled": cache.enabled()});
        Self { lanes: vec![decoding.min(255) as u8], queued: queued as u32, prefilling: Some(prefilling as u32),
            pending: None, kv: Some(kv), host: stats.host.and_then(|host| serde_json::to_value(host).ok()),
            prefix: Some(prefix) }
    }
}

/// Name the layer profile's class of every layer (index into the layout's
/// classes) once the engine knows its placement.
pub(crate) fn layer_classes(classes: Vec<u8>) {
    lifecycle(Event::LayerClasses(classes));
}

/// Push gauges built by `build` when a viewer is connected and 250 ms passed.
#[inline]
pub(crate) fn gauges(build: impl FnOnce() -> Gauges) {
    if let Some(live) = live().filter(|live| live.gauges_due()) { live.push(Event::Gauges(build())); }
}

/// Push gauges now (going idle: the page must not keep showing busy lanes).
#[inline]
pub(crate) fn gauges_now(build: impl FnOnce() -> Gauges) {
    if let Some(live) = live() { live.push(Event::Gauges(build())); }
}

/// Lifetime serving counters. Always on; exported in `/v1/stats` and the console.
pub(crate) mod totals {
    use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
    static ADMITTED: AtomicU64 = AtomicU64::new(0);
    static RETIRED: AtomicU64 = AtomicU64::new(0);
    static INPUT: AtomicU64 = AtomicU64::new(0);
    static CACHED: AtomicU64 = AtomicU64::new(0);
    static PREFILL: AtomicU64 = AtomicU64::new(0);
    static OUTPUT: AtomicU64 = AtomicU64::new(0);
    static ROUNDS: AtomicU64 = AtomicU64::new(0);
    static DRAFTED: AtomicU64 = AtomicU64::new(0);
    static VERIFIED: AtomicU64 = AtomicU64::new(0);
    static ACCEPTED: AtomicU64 = AtomicU64::new(0);

    pub fn admitted(prompt: usize, cached: usize) {
        ADMITTED.fetch_add(1, Relaxed);
        INPUT.fetch_add(prompt as u64, Relaxed);
        CACHED.fetch_add(cached as u64, Relaxed);
    }
    pub fn active() -> u64 { ADMITTED.load(Relaxed).saturating_sub(RETIRED.load(Relaxed)) }
    pub fn retired() { RETIRED.fetch_add(1, Relaxed); }
    pub fn prefill(rows: usize) { PREFILL.fetch_add(rows as u64, Relaxed); }
    pub fn output(tokens: usize) { OUTPUT.fetch_add(tokens as u64, Relaxed); }
    pub fn round(drafted: u64, verified: u64, accepted: u64, emitted: u64) {
        ROUNDS.fetch_add(1, Relaxed);
        DRAFTED.fetch_add(drafted, Relaxed);
        VERIFIED.fetch_add(verified, Relaxed);
        ACCEPTED.fetch_add(accepted, Relaxed);
        OUTPUT.fetch_add(emitted, Relaxed);
    }
    pub fn snapshot() -> serde_json::Value {
        serde_json::json!({
            "requests_admitted": ADMITTED.load(Relaxed),
            "requests_retired": RETIRED.load(Relaxed),
            "input_tokens": INPUT.load(Relaxed),
            "cached_input_tokens": CACHED.load(Relaxed),
            "prefill_tokens": PREFILL.load(Relaxed),
            "output_tokens": OUTPUT.load(Relaxed),
            "verification_rounds": ROUNDS.load(Relaxed),
            "drafted_tokens": DRAFTED.load(Relaxed),
            "verified_drafts": VERIFIED.load(Relaxed),
            "accepted_drafts": ACCEPTED.load(Relaxed),
        })
    }
}

/// Bind the console to its hub and start the console thread. A second install
/// in one process (tests) is refused.
pub(crate) fn install(hub: Arc<ConsoleHub>, layout: Layout) -> anyhow::Result<()> {
    let (events, receive) = sync_channel(CHANNEL);
    let base = Instant::now();
    let producer = Producer { hub: hub.clone(), events, base, gauges_at: AtomicU64::new(0) };
    if PRODUCER.set(producer).is_err() {
        anyhow::bail!("console producer is already installed");
    }
    std::thread::Builder::new().name("cuteafd-console".into())
        .spawn(move || worker::Worker::new(hub, layout, base).run(receive))?;
    Ok(())
}

/// The hub a generic family serves the console from: installed with `layout`
/// when it can be, else a page that reports the feed is off.
pub(crate) fn hub(text: bool, layout: impl FnOnce() -> anyhow::Result<Layout>) -> Arc<ConsoleHub> {
    let hub = ConsoleHub::new(text);
    match layout().and_then(|layout| install(hub.clone(), layout)) {
        Ok(()) => hub,
        Err(error) => {
            tracing::warn!("live console off: {error:#}");
            ConsoleHub::disabled()
        }
    }
}

/// The console switch every serve command takes.
#[derive(Debug, Clone, Copy, clap::Args)]
pub(crate) struct ConsoleArgs {
    /// Let the live console at `/` show generated token text. Only viewers
    /// holding the console unlock cookie receive it (bench runs excepted).
    #[arg(long, env = "CUTEAFD_CONSOLE_TEXT", num_args = 0..=1, default_value = "true",
        default_missing_value = "true", value_parser = clap::builder::BoolishValueParser::new())]
    pub console_text: bool,
}

static NEXT_ID: AtomicU64 = AtomicU64::new(1);

/// One admitted request of a generic family. Its id names the request on the
/// page; dropping it without [`Ticket::done`] retires the request as failed.
pub(crate) struct Ticket {
    id: u64,
    done: bool,
    /// The request sent its finish (EOS, stop or length) to the client.
    finished: bool,
    usage: Option<cuteafd_api::usage::UsageHandle>,
}

/// Admit a request: always counted, announced while a console is installed.
/// `cached` prompt rows came from the prefix cache in the admission that began
/// at `admit_started` (shown as a restore step when nonzero).
pub(crate) fn admit(prompt: usize, cached: usize, max: usize, grammar: bool, images: usize,
    admit_started: Instant, usage: Option<cuteafd_api::usage::UsageHandle>) -> Ticket {
    if let Some(usage) = &usage {
        usage.admitted(totals::active() as u64 + 1);
        usage.prompt_tokens(prompt as u64, cached as u64);
    }
    let id = NEXT_ID.fetch_add(1, Ordering::Relaxed);
    totals::admitted(prompt, cached);
    let at = Instant::now();
    lifecycle(Event::Admit { id, at: admit_started, prompt: prompt as u32, cached: cached as u32,
        max: max.min(u32::MAX as usize) as u32, lane: 0, grammar, images: images.min(u16::MAX as usize) as u16 });
    if cached > 0 {
        if let Some(live) = live() {
            live.push(Event::Prefill(Prefill { id: Some(id), kind: PrefillKind::Restore, lane: 0, index: 0, of: 1,
                rows: cached as u32, started: admit_started, finished: at }));
        }
    }
    Ticket { id, done: false, finished: false, usage }
}

impl Ticket {
    pub fn id(&self) -> u64 { self.id }
    pub fn session(&self) -> Option<String> { self.usage.as_ref().map(|u| u.session_id().to_owned()) }

    /// A prefill chunk of `rows` prompt rows that began at `started` just finished.
    #[inline]
    pub fn prefill(&self, rows: usize, index: usize, of: usize, started: Instant) {
        totals::prefill(rows);
        if let Some(live) = live() {
            live.push(Event::Prefill(Prefill { id: Some(self.id), kind: PrefillKind::Chunk, lane: 0,
                index: index as u32, of: of.max(index + 1) as u32, rows: rows as u32, started,
                finished: Instant::now() }));
        }
    }

    /// The first generated token was emitted.
    pub fn first(&self, token: u32) {
        if let Some(usage) = &self.usage { usage.first_token(); }
        totals::output(1);
        lifecycle(Event::First { id: self.id, at: Instant::now(), token });
    }

    /// The request is sending its finish to the client (call before the send).
    pub fn finishing(&mut self) { self.finished = true; }

    /// The request left the scheduler after `generated` tokens: finished if it
    /// sent its finish ([`Self::finishing`]), else its client left.
    pub fn done(&mut self, generated: usize) {
        self.retire(if self.finished { "finished" } else { "cancelled" }, generated);
    }

    /// The client left before the request produced anything.
    pub fn cancel(&mut self) { self.retire("cancelled", 0); }

    fn retire(&mut self, reason: &'static str, generated: usize) {
        if std::mem::replace(&mut self.done, true) { return; }
        if let Some(usage) = &self.usage { usage.retired(generated as u64, reason); }
        totals::retired();
        lifecycle(Event::Retire { id: self.id, at: Instant::now(), reason, generated: generated as u32 });
    }
}

impl Drop for Ticket {
    fn drop(&mut self) { self.retire("failed", 0); }
}

thread_local! {
    /// Layer-end marks of the decode step running on this thread (armed only
    /// while a viewer is connected; see [`layer_mark`]).
    static LAYER_MARKS: std::cell::RefCell<Option<Vec<(u16, Instant)>>> = const { std::cell::RefCell::new(None) };
}

/// A decode step finished layer `index` on the host clock. Engines call this
/// once per layer of their decode loop; it records only while a [`Step`] with
/// a viewer runs on this thread (otherwise one thread-local check).
#[inline]
pub(crate) fn layer_mark(index: usize) {
    LAYER_MARKS.with(|marks| {
        if let Some(marks) = marks.borrow_mut().as_mut() { marks.push((index.min(u16::MAX as usize) as u16, Instant::now())); }
    });
}

/// Per-layer microseconds from layer-end marks: each layer's time is from the
/// previous mark (repeats of a layer add up); the first mark has no start.
fn layer_times(marks: &[(u16, Instant)]) -> Vec<f32> {
    let Some(layers) = marks.iter().map(|&(index, _)| usize::from(index) + 1).max() else { return Vec::new() };
    let mut times = vec![f32::NAN; layers];
    for pair in marks.windows(2) {
        let (index, us) = (usize::from(pair[1].0), pair[1].1.saturating_duration_since(pair[0].1).as_secs_f32() * 1e6);
        times[index] = if times[index].is_nan() { us } else { times[index] + us };
    }
    times
}

/// One decode round of a generic family. Always counts the lifetime totals;
/// builds the round event only while a viewer is connected.
pub(crate) struct Step {
    started: Instant,
    lane: u8,
    live: Option<Live>,
    text: bool,
    requests: Vec<RoundRequest>,
    tally: [u64; 4],
}

impl Step {
    #[inline]
    pub fn begin(lane: usize) -> Self {
        let live = live();
        if live.is_some() { LAYER_MARKS.with(|marks| *marks.borrow_mut() = Some(Vec::with_capacity(128))); }
        Self { started: Instant::now(), lane: lane as u8, live, text: live.is_some_and(Live::text),
            requests: Vec::new(), tally: [0; 4] }
    }

    /// Whether the round event will be built (callers skip extra timing otherwise).
    #[inline]
    pub fn live(&self) -> bool { self.live.is_some() }

    /// One member: its draft `proposal` (after the anchor, before truncation),
    /// the `verified` draft rows, the tokens it `emitted` this round, whether a
    /// grammar masked it and whether it finished.
    #[inline]
    pub fn member(&mut self, ticket: &Ticket, proposal: &[u32], verified: usize, emitted: &[u32], masked: bool,
        finished: bool) {
        let accepted = emitted.len().saturating_sub(1).min(verified);
        if let Some(usage) = &ticket.usage { usage.round(proposal.len().max(verified) as u64, accepted as u64); }
        self.tally[0] += proposal.len().max(verified) as u64;
        self.tally[1] += verified as u64;
        self.tally[2] += accepted as u64;
        self.tally[3] += emitted.len() as u64;
        if self.live.is_none() { return; }
        let byte = |n: usize| n.min(255) as u8;
        self.requests.push(RoundRequest { id: ticket.id, drafted: byte(proposal.len().max(verified)),
            verified: byte(verified), accepted: byte(accepted), emitted: byte(emitted.len()), masked, finished,
            proposal: if self.text { proposal.to_vec() } else { Vec::new() },
            emissions: if self.text { emitted.to_vec() } else { Vec::new() } });
    }

    /// Count the round; with a viewer, send it with the stage times `stages`
    /// returns (microseconds by the layout's stage keys).
    pub fn end(mut self, stages: impl FnOnce() -> Vec<(&'static str, f64)>) {
        let [drafted, verified, accepted, emitted] = self.tally;
        totals::round(drafted, verified, accepted, emitted);
        let Some(live) = self.live else { return };
        let marks = LAYER_MARKS.with(|marks| marks.borrow_mut().take()).unwrap_or_default();
        let stages = stages().into_iter().map(|(key, us)| (key, us as f32)).collect();
        live.push(Event::Round(Round { lane: self.lane, shared: false, started: self.started,
            finished: Instant::now(), stages, layer_us: layer_times(&marks),
            requests: std::mem::take(&mut self.requests) }));
    }
}

impl Drop for Step {
    fn drop(&mut self) {
        if self.live.is_some() { LAYER_MARKS.with(|marks| *marks.borrow_mut() = None); }
    }
}

/// Microseconds since `at`.
#[inline]
pub(crate) fn us(at: Instant) -> f64 { at.elapsed().as_secs_f64() * 1e6 }

/// The header's model id of a snapshot directory (`models--ORG--NAME/...`).
pub(crate) fn checkpoint(snapshot: &std::path::Path) -> String {
    let id = snapshot.ancestors().find_map(|dir| {
        let name = dir.file_name()?.to_str()?.strip_prefix("models--")?;
        let (org, model) = name.split_once("--")?;
        Some(format!("{org}/{model}"))
    });
    let revision = snapshot.parent().and_then(|parent| (parent.file_name()? == "snapshots").then_some(()))
        .and_then(|()| snapshot.file_name()?.to_str()).map(|rev| rev.chars().take(8).collect::<String>());
    match (id, revision) {
        (Some(id), Some(revision)) => format!("{id}@{revision}"),
        (Some(id), None) => id,
        (None, _) => snapshot.file_name().map_or_else(|| snapshot.display().to_string(),
            |name| name.to_string_lossy().into_owned()),
    }
}

/// The layout description of a coordinator with `gpus` RTX GPUs and `sparks`
/// Spark ranks (`local`: experts on the RTX GPUs).
pub(crate) fn hardware(gpus: usize, sparks: usize, local: bool) -> String {
    match (sparks, local) {
        (0, true) => format!("{gpus}×RTX, local experts"),
        (0, false) => format!("{gpus}×RTX"),
        (n, _) => format!("{gpus}×RTX + {n} Spark"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn step_counts_members_without_a_viewer() {
        let ticket = Ticket { id: 7, done: true, finished: false, usage: None };
        let mut step = Step { started: Instant::now(), lane: 0, live: None, text: false, requests: Vec::new(),
            tally: [0; 4] };
        // 5 drafted, 3 verified, 2 accepted (3 emitted); then a plain decode row.
        step.member(&ticket, &[1, 2, 3, 4, 5], 3, &[1, 2, 9], false, false);
        step.member(&ticket, &[], 0, &[4], true, true);
        assert_eq!(step.tally, [5, 3, 2, 4]);
        assert!(step.requests.is_empty());
    }

    /// A console fed by synthetic rounds for page development:
    /// `CUTEAFD_CONSOLE_DEMO=127.0.0.1:8123 cargo test -p cuteafd-daemon --bin cuteafd console_demo -- --ignored`
    /// serves `/` until killed (`CUTEAFD_CONSOLE_DEMO_SPEC=0`: no speculator).
    #[test]
    #[ignore]
    fn console_demo() {
        let listen = std::env::var("CUTEAFD_CONSOLE_DEMO").unwrap_or_else(|_| "127.0.0.1:8123".into());
        let speculate = std::env::var("CUTEAFD_CONSOLE_DEMO_SPEC").map_or(true, |v| v != "0");
        let mut layout = Layout::new("glm5", "zai-org/GLM-5.3".into(),
            "/hf/models--zai-org--GLM-5.3/snapshots/0123456789".into());
        layout.hardware = hardware(2, 6, false);
        layout.split = Some("head split".into());
        layout.concurrency = 16;
        layout.speculator = speculate.then(|| Speculator { name: "DFlash2".into(), positions: 7, policy: "adaptive".into() });
        layout.steps = vec![StepGroup::new("Decode step", "host clock", &[("round.cycle", "step", Color::Target),
            ("draft", "DFlash2 draft", Color::Accepted), ("verify", "verify pass + head", Color::Target),
            ("emit", "accept + stream", Color::Ink)]),
            StepGroup::new("Verify pass", "host clock", &[("experts", "Spark experts wait", Color::Spark)]),
            StepGroup::layers(), StepGroup::admission(false)];
        layout.layers = Some(Layers::host_clock());
        let hub = ConsoleHub::new(false);
        install(hub.clone(), layout).unwrap();
        layer_classes((0..78).map(|l| layer_class(l < 3, true)).collect());
        let runtime = tokio::runtime::Runtime::new().unwrap();
        let (queue, _receive) = tokio::sync::mpsc::channel(4);
        let router = cuteafd_api::openai::router_with_console(queue, cuteafd_api::openai::NativeLimits::default(),
            Arc::new(std::sync::Mutex::new(Value::Null)), std::time::Duration::from_secs(1), hub);
        runtime.spawn(async move {
            let listener = tokio::net::TcpListener::bind(&listen).await.unwrap();
            axum::serve(listener, router).await.unwrap();
        });
        let mut requests: Vec<(Ticket, usize, usize)> = Vec::new();
        let mut seed = 0x2545_f491_4f6c_dd1du64;
        let mut random = move |n: usize| { seed ^= seed << 13; seed ^= seed >> 7; seed ^= seed << 17; (seed % n as u64) as usize };
        loop {
            while requests.len() < 6 {
                let started = Instant::now();
                std::thread::sleep(std::time::Duration::from_millis(3));
                let ticket = admit(1500 + random(500), 1300, 300, random(3) == 0, 0, started, None);
                let chunk = Instant::now();
                std::thread::sleep(std::time::Duration::from_millis(20));
                ticket.prefill(200 + random(100), 0, 1, chunk);
                ticket.first(42);
                requests.push((ticket, 1, 100 + random(200)));
            }
            let mut step = Step::begin(0);
            let draft = Instant::now();
            std::thread::sleep(std::time::Duration::from_millis(4));
            let draft_us = us(draft);
            for layer in 0..78 {
                std::thread::sleep(std::time::Duration::from_micros(if layer < 3 { 80 } else { 250 + 3 * layer as u64 }));
                layer_mark(layer);
            }
            for (ticket, generated, _) in &mut requests {
                let width = if speculate { 7 } else { 0 };
                let verified = width.min(random(8));
                let emitted = 1 + random(verified + 1);
                *generated += emitted;
                step.member(ticket, &vec![1; width], verified, &vec![2; emitted], false, false);
            }
            step.end(|| vec![("draft", draft_us), ("verify", 25_000.0), ("emit", 300.0), ("experts", 9_000.0)]);
            gauges(|| Gauges { lanes: vec![requests.len() as u8], queued: 0, prefilling: Some(0), pending: None,
                kv: Some(Kv { pages: 4096, free: 3000, active: 600, tokens_per_page: 256 }), host: None,
                prefix: Some(serde_json::json!({"entries": 12, "hits": 30, "lookups": 40, "hit_tokens": 52000,
                    "enabled": true})) });
            requests.retain_mut(|(ticket, generated, max)| {
                let done = *generated >= *max;
                if done { ticket.finishing(); ticket.done(*generated); }
                !done
            });
        }
    }

    #[test]
    fn layer_marks_become_per_layer_times() {
        let t = Instant::now();
        let at = |us: u64| t + std::time::Duration::from_micros(us);
        // Layers 0..3 end at 10, 30, 60 µs; an MTP pass re-runs layer 2 (ends at 70).
        let times = layer_times(&[(0, at(10)), (1, at(30)), (2, at(60)), (2, at(70))]);
        assert!(times[0].is_nan());
        assert_eq!((times[1].round(), times[2].round()), (20.0, 40.0));
        assert!(layer_times(&[]).is_empty());
        // Unarmed marks are dropped.
        layer_mark(3);
        assert!(LAYER_MARKS.with(|marks| marks.borrow().is_none()));
    }

    #[test]
    fn checkpoint_names_the_hub_id_and_revision() {
        let path = std::path::Path::new("/hf/hub/models--zai-org--GLM-5.3/snapshots/0123456789abcdef");
        assert_eq!(checkpoint(path), "zai-org/GLM-5.3@01234567");
        assert_eq!(checkpoint(std::path::Path::new("/models/glm")), "glm");
    }
}
