//! Process-wide ledger of every device, pinned-host and managed allocation made
//! through [`crate::NativeLibrary`].
//!
//! Every allocation constructor states a [`Tag`]: its [`MemoryKind`] and owner.
//! A thread-local scope stack adds context only (a sub-owner label, the
//! checkpoint tensor and resident format a weight reader is uploading); it
//! never decides the kind. Frees look the pointer up, so the ledger always
//! holds the live bytes. Memory the runtime takes without this library
//! (context, modules, cuBLAS, graph executables) is a measured residual: the
//! gap between the ledger and `cudaMemGetInfo`.

pub mod fault;

use std::cell::RefCell;
use std::collections::{BTreeMap, HashMap, HashSet};
use std::sync::{Arc, Mutex, OnceLock, Weak};

pub use cuteafd_core::memory_layout::{MemoryKind, Tag};

/// Which pool an allocation came from.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Space {
    Device,
    Pinned,
    Managed,
    /// Native malloc backing registered for RDMA, not CUDA-pinned storage.
    Host,
}

impl Space {
    pub fn label(self) -> &'static str {
        match self {
            Self::Device => "device",
            Self::Pinned => "pinned",
            Self::Managed => "managed",
            Self::Host => "host",
        }
    }
}

#[derive(Default)]
struct Context {
    scopes: Vec<&'static str>,
    site: Option<&'static std::panic::Location<'static>>,
    /// The owner tag ([`owner`]): survives nested category scopes.
    owner: Option<&'static str>,
    tensor: Option<&'static str>,
    /// The full checkpoint name behind `tensor` (dual-format detection per tensor).
    name: Option<&'static str>,
    format: Option<&'static str>,
}

thread_local! {
    static CONTEXT: RefCell<Context> = RefCell::new(Context::default());
}

#[derive(Clone, Copy, Debug)]
struct Live {
    key: Key,
    bytes: usize,
    endpoint: Option<usize>,
    name: Option<&'static str>,
    site: &'static std::panic::Location<'static>,
    native_site: Option<&'static str>,
}

/// Aggregation key: pool, device (-1 for host memory), category, tensor stem
/// (layer indices replaced by `*`) and resident format.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Key {
    pub space: Space,
    pub device: i32,
    /// What the allocation is, required at every constructor.
    pub tag: Tag,
    /// Innermost context scope; the tag's owner when no scope is open.
    pub scope: &'static str,
    /// The enclosing accounting [`owner`] guard, if any: what plan == runtime
    /// checks sum with [`Snapshot::by_owner`]. Distinct from `tag.owner` (the
    /// component that made the allocation): a tag never sets or replaces it, so
    /// removing a guard drops its bytes from those checks. Category scopes nested
    /// inside an owner (`sampler`, `workspace/fp8-linear`) relabel `scope` but keep it.
    pub owner: Option<&'static str>,
    pub tensor: Option<&'static str>,
    pub format: Option<&'static str>,
}

#[derive(Default)]
struct State {
    live: HashMap<usize, Live>,
    /// Non-scale formats each full tensor name was uploaded in, per device.
    formats: BTreeMap<(i32, &'static str), Vec<&'static str>>,
    totals: BTreeMap<Key, (usize, usize)>,
    peak: BTreeMap<(Space, i32), usize>,
    current: BTreeMap<(Space, i32), usize>,
    audits: Vec<Weak<Mutex<AuditEvents>>>,
    ready: bool,
    post_ready_bytes: usize,
    post_ready_sites: BTreeMap<&'static std::panic::Location<'static>, usize>,
    post_ready_native_sites: BTreeMap<&'static str, usize>,
}

/// An actual native allocation, before aggregation removes pointer identity.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AllocationRecord {
    pub address: usize,
    pub key: Key,
    pub bytes: usize,
    /// Native endpoint owning an RDMA ring; never the backing address.
    pub endpoint: Option<usize>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AllocationEvent {
    Allocated(AllocationRecord),
    Freed(AllocationRecord),
    /// Ownership metadata changed without a new physical allocation.
    Enriched(AllocationRecord),
    /// The allocator reused an address whose release bypassed this ledger.
    /// This is not evidence of a free at the time the reuse was observed.
    UnobservedRelease(AllocationRecord),
}

#[derive(Default)]
struct AuditEvents {
    events: Vec<AllocationEvent>,
}

/// Caller-owned, startup-only history. Dropping this handle stops recording;
/// the process ledger retains only a weak subscription, never the history.
pub struct AllocationAudit {
    initial: Vec<AllocationRecord>,
    events: Arc<Mutex<AuditEvents>>,
}

#[derive(Debug)]
pub struct AllocationHistory {
    pub initial: Vec<AllocationRecord>,
    pub events: Vec<AllocationEvent>,
    pub final_allocations: Vec<AllocationRecord>,
}

impl AllocationAudit {
    pub fn begin() -> Self {
        let events = Arc::new(Mutex::new(AuditEvents::default()));
        let mut state = state().lock().unwrap_or_else(|e| e.into_inner());
        let initial = live_allocations(&state);
        state.audits.retain(|audit| audit.strong_count() != 0);
        state.audits.push(Arc::downgrade(&events));
        Self { initial, events }
    }

    /// Position in the ordered stream at an existing startup boundary. This
    /// neither synchronizes CUDA nor changes the lifetime of any allocation.
    pub fn checkpoint(&self) -> usize {
        self.events.lock().unwrap_or_else(|e| e.into_inner()).events.len()
    }

    /// Sample exact live backing and its event position under the same ledger
    /// lock, so a worker allocation cannot fall between those observations.
    pub fn checkpoint_allocations(&self) -> (usize, Vec<AllocationRecord>) {
        let state = state().lock().unwrap_or_else(|e| e.into_inner());
        let position = self.events.lock().unwrap_or_else(|e| e.into_inner()).events.len();
        (position, live_allocations(&state))
    }

    /// Close the subscription under the allocator's lock before taking history.
    pub fn finish(self) -> AllocationHistory {
        let mut state = state().lock().unwrap_or_else(|e| e.into_inner());
        state.audits.retain(|audit| !audit.ptr_eq(&Arc::downgrade(&self.events))
            && audit.strong_count() != 0);
        let events = std::mem::take(&mut self.events.lock().unwrap_or_else(|e| e.into_inner()).events);
        AllocationHistory { initial: self.initial, events, final_allocations: live_allocations(&state) }
    }
}

fn live_allocations(state: &State) -> Vec<AllocationRecord> {
    let mut allocations = state.live.iter().map(|(&address, live)| AllocationRecord {
        address, key: live.key, bytes: live.bytes, endpoint: live.endpoint,
    }).collect::<Vec<_>>();
    allocations.sort_unstable_by_key(|record| (record.key.space, record.key.device, record.address));
    allocations
}

/// Pointer-level snapshot of all currently tracked native owners.
pub fn allocations() -> Vec<AllocationRecord> {
    live_allocations(&state().lock().unwrap_or_else(|e| e.into_inner()))
}

fn audit_event(state: &mut State, event: AllocationEvent) {
    state.audits.retain(|audit| {
        let Some(audit) = audit.upgrade() else { return false };
        audit.lock().unwrap_or_else(|e| e.into_inner()).events.push(event);
        true
    });
}

fn state() -> &'static Mutex<State> {
    static STATE: OnceLock<Mutex<State>> = OnceLock::new();
    STATE.get_or_init(Default::default)
}

/// Interns a label so it can live in the ledger for the life of the process.
/// The label set is small (categories, tensor stems, formats).
pub fn intern(label: &str) -> &'static str {
    static LABELS: OnceLock<Mutex<HashSet<&'static str>>> = OnceLock::new();
    let mut labels = LABELS.get_or_init(Default::default).lock().unwrap_or_else(|e| e.into_inner());
    if let Some(&existing) = labels.get(label) {
        return existing;
    }
    let leaked: &'static str = Box::leak(label.to_owned().into_boxed_str());
    labels.insert(leaked);
    leaked
}

/// `model.layers.12.mlp.experts.7.w1` -> `model.layers.*.mlp.experts.*.w1`.
pub fn tensor_stem(name: &str) -> String {
    name.split('.')
        .map(|part| if !part.is_empty() && part.bytes().all(|b| b.is_ascii_digit()) { "*" } else { part })
        .collect::<Vec<_>>()
        .join(".")
}

/// Scope guard: allocations on this thread are labelled `label` until it drops.
/// Nested scopes replace the label (the innermost wins) and restore it after.
#[must_use = "the scope ends when the guard drops"]
pub struct Scope {
    depth: usize,
    tensor: Option<&'static str>,
    format: Option<&'static str>,
}

pub fn scope(label: &'static str) -> Scope {
    CONTEXT.with(|context| {
        let mut context = context.borrow_mut();
        context.scopes.push(label);
        context.name = None;
        Scope { depth: context.scopes.len(), tensor: context.tensor.take(), format: context.format.take() }
    })
}

/// [`scope`] with a runtime label (interned).
pub fn scope_owned(label: &str) -> Scope {
    scope(intern(label))
}

impl Drop for Scope {
    fn drop(&mut self) {
        CONTEXT.with(|context| {
            let mut context = context.borrow_mut();
            context.scopes.truncate(self.depth.saturating_sub(1));
            context.tensor = self.tensor;
            context.name = None;
            context.format = self.format;
        });
    }
}

/// Owner guard: allocations on this thread carry `label` as their owner until
/// it drops, whatever category scopes nest inside it (the innermost owner wins
/// and the previous one is restored). Plan == runtime checks sum by owner.
#[must_use = "the owner tag ends when the guard drops"]
pub struct Owner(Option<&'static str>);

pub fn owner(label: &str) -> Owner {
    let label = intern(label);
    CONTEXT.with(|context| Owner(context.borrow_mut().owner.replace(label)))
}

impl Drop for Owner {
    fn drop(&mut self) {
        CONTEXT.with(|context| context.borrow_mut().owner = self.0);
    }
}

/// Names the checkpoint tensor the next allocations on this thread hold.
/// Cleared by the enclosing [`scope`] when it ends.
pub fn tensor(name: &str) {
    let stem = intern(&tensor_stem(name));
    let full = intern(name);
    CONTEXT.with(|context| {
        let mut context = context.borrow_mut();
        context.tensor = Some(stem);
        context.name = Some(full);
    });
}

/// Names the work itself when it is not a checkpoint tensor (a decode graph's
/// shape key, a scratch region): a failure or live row on this thread reports
/// it as `tensor` without a `tensor_stem`, so it never joins the weights table.
#[must_use = "the subject ends when the guard drops"]
pub struct Subject(Option<&'static str>);

pub fn subject(label: &str) -> Subject {
    let label = intern(label);
    CONTEXT.with(|context| Subject(context.borrow_mut().name.replace(label)))
}

impl Drop for Subject {
    fn drop(&mut self) {
        CONTEXT.with(|context| context.borrow_mut().name = self.0);
    }
}

/// Format guard: allocations on this thread are tagged with this resident
/// format (`bf16`, `fp8`, `fp8-scale`, ...) until it drops.
#[must_use = "the format tag ends when the guard drops"]
pub struct Format(Option<&'static str>);

pub fn format(label: &'static str) -> Format {
    CONTEXT.with(|context| Format(context.borrow_mut().format.replace(label)))
}

impl Drop for Format {
    fn drop(&mut self) {
        CONTEXT.with(|context| context.borrow_mut().format = self.0);
    }
}

/// Preserve an owning wrapper's caller through device-selection closures.
#[must_use]
pub struct AllocationSite(Option<&'static std::panic::Location<'static>>);

pub fn allocation_site(site: &'static std::panic::Location<'static>) -> AllocationSite {
    AllocationSite(CONTEXT.with(|context| context.borrow_mut().site.replace(site)))
}

impl Drop for AllocationSite {
    fn drop(&mut self) {
        CONTEXT.with(|context| context.borrow_mut().site = self.0);
    }
}

fn current_key(space: Space, device: i32, tag: Tag) -> (Key, Option<&'static str>) {
    CONTEXT.with(|context| {
        let context = context.borrow();
        (Key {
            space,
            device,
            tag,
            scope: context.scopes.last().copied().unwrap_or(tag.owner),
            owner: context.owner,
            tensor: context.tensor,
            format: context.format,
        }, context.name)
    })
}

/// A tag whose owner is built at runtime (`kv/step5-pages`, a per-rank name):
/// the owner is interned for the life of the process.
pub fn tag(kind: MemoryKind, owner: &str) -> Tag {
    Tag::new(kind, intern(owner))
}

/// The two tags a native media runtime (vision, audio, PCM tower) receives at
/// create: its immutable weight arena and its scratch arena. After create,
/// Rust reads the runtime's backings and records each with [`record_tagged`]
/// under these tags, then retires each with [`record_released`] once native
/// reports that backing's free succeeded.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MediaTags {
    pub weights: Tag,
    pub workspace: Tag,
}

impl MediaTags {
    /// `tower` names the encoder (`muse/vision`, `mimo/audio`, ...).
    pub fn new(tower: &str) -> Self {
        Self { weights: tag(MemoryKind::Weights, tower), workspace: tag(MemoryKind::Workspace, tower) }
    }
}

/// Records native backing this library did not allocate itself (a media
/// runtime's weights and scratch, read back after create) under its tag.
/// Retire it with [`record_released`] only after the native free succeeded.
#[track_caller]
pub fn record_tagged(space: Space, device: i32, ptr: usize, bytes: usize, tag: Tag) {
    record_backing(space, device, ptr, bytes, None, None, tag);
}

/// Retires backing recorded by [`record_tagged`] after its native free succeeded.
pub fn record_released(ptr: usize) {
    record_free(ptr);
}

#[track_caller]
pub(crate) fn record_alloc(space: Space, device: i32, ptr: usize, bytes: usize, tag: Tag) {
    record_backing(space, device, ptr, bytes, None, None, tag);
}

#[track_caller]
pub(crate) fn record_endpoint_alloc(space: Space, ptr: usize, bytes: usize, endpoint: usize, tag: Tag) {
    record_backing(space, -1, ptr, bytes, Some(endpoint), None, tag);
}

/// A backing native allocated internally and reported through its event
/// callback, with the tag native allocated it under.
#[track_caller]
pub(crate) fn record_native_alloc(space: Space, device: i32, ptr: usize, bytes: usize, site: &str, tag: Tag) {
    record_backing(space, device, ptr, bytes, None, Some(intern(site)), tag);
}

#[track_caller]
fn record_backing(space: Space, device: i32, ptr: usize, bytes: usize, endpoint: Option<usize>,
    native_site: Option<&'static str>, tag: Tag) {
    if ptr == 0 || bytes == 0 {
        return;
    }
    let (key, name) = current_key(space, device, tag);
    let caller = std::panic::Location::caller();
    let site = CONTEXT.with(|context| context.borrow().site).unwrap_or(caller);
    let mut state = state().lock().unwrap_or_else(|e| e.into_inner());
    if endpoint.is_some() { if let Some(existing) = state.live.get(&ptr).copied() {
        if existing.endpoint.is_none() && existing.bytes == bytes && existing.key.space == space {
            // Adopt the authoritative transport label, retaining physical lifetime,
            // native site and cumulative growth rather than inventing a reuse.
            release(&mut state, existing);
            let total = state.totals.entry(key).or_default();
            total.0 += bytes; total.1 += 1;
            *state.current.entry((space, device)).or_default() += bytes;
            state.live.insert(ptr, Live { key, endpoint, ..existing });
            audit_event(&mut state, AllocationEvent::Enriched(AllocationRecord { address: ptr, key, bytes, endpoint }));
            return;
        }
    } }
    if let (Some(name), Some(format)) = (name, key.format) {
        if !format.ends_with("scale") {
            let formats = state.formats.entry((device, name)).or_default();
            if !formats.contains(&format) {
                formats.push(format);
            }
        }
    }
    let total = state.totals.entry(key).or_default();
    total.0 += bytes;
    total.1 += 1;
    let current = state.current.entry((space, device)).or_default();
    *current += bytes;
    let now = *current;
    let peak = state.peak.entry((space, device)).or_default();
    *peak = (*peak).max(now);
    if let Some(previous) = state.live.insert(ptr, Live { key, bytes, endpoint, name, site, native_site }) {
        // A pointer reused without a recorded free (freed outside the library).
        release(&mut state, previous);
        audit_event(&mut state, AllocationEvent::UnobservedRelease(AllocationRecord {
            address: ptr, key: previous.key, bytes: previous.bytes, endpoint: previous.endpoint,
        }));
    }
    audit_event(&mut state, AllocationEvent::Allocated(AllocationRecord { address: ptr, key, bytes, endpoint }));
    let growth = state.ready && matches!(space, Space::Device | Space::Pinned | Space::Managed);
    let new_site = growth && match native_site {
        Some(site) => !state.post_ready_native_sites.contains_key(site),
        None => !state.post_ready_sites.contains_key(site),
    };
    if growth {
        state.post_ready_bytes = state.post_ready_bytes.saturating_add(bytes);
        match native_site {
            Some(site) => *state.post_ready_native_sites.entry(site).or_default() += bytes,
            None => *state.post_ready_sites.entry(site).or_default() += bytes,
        }
    }
    let post_ready_bytes = state.post_ready_bytes;
    drop(state);
    if new_site {
        tracing::info!(target: "cuteafd::memory", %site, bytes, post_ready_bytes,
            native_site, scope = key.scope, "post-ready growth");
    }
}

fn release(state: &mut State, live: Live) {
    if let Some(total) = state.totals.get_mut(&live.key) {
        total.0 = total.0.saturating_sub(live.bytes);
        total.1 = total.1.saturating_sub(1);
        if total.1 == 0 {
            state.totals.remove(&live.key);
        }
    }
    if let Some(current) = state.current.get_mut(&(live.key.space, live.key.device)) {
        *current = current.saturating_sub(live.bytes);
    }
}

/// Read-only source locations keyed by live physical backing address.
pub fn allocation_sites() -> BTreeMap<usize, &'static std::panic::Location<'static>> {
    state().lock().unwrap_or_else(|e| e.into_inner()).live.iter()
        .map(|(&address, live)| (address, live.site)).collect()
}

/// The tag a live backing was recorded with (`None`: not live).
pub fn tag_of(ptr: usize) -> Option<Tag> {
    state().lock().unwrap_or_else(|e| e.into_inner()).live.get(&ptr).map(|live| live.key.tag)
}

/// Read one live-byte counter without cloning the allocation ledger.
pub(crate) fn current_bytes(space: Space, device: i32) -> usize {
    state().lock().unwrap_or_else(|e| e.into_inner()).current.get(&(space, device)).copied().unwrap_or(0)
}

pub(crate) fn record_free(ptr: usize) {
    if ptr == 0 {
        return;
    }
    let mut state = state().lock().unwrap_or_else(|e| e.into_inner());
    if let Some(live) = state.live.remove(&ptr) {
        release(&mut state, live);
        audit_event(&mut state, AllocationEvent::Freed(AllocationRecord {
            address: ptr, key: live.key, bytes: live.bytes, endpoint: live.endpoint,
        }));
    }
}

/// Retire all backing of an endpoint only after its native destroy succeeds.
pub(crate) fn record_endpoint_free(endpoint: usize) {
    let mut state = state().lock().unwrap_or_else(|e| e.into_inner());
    let addresses = state.live.iter().filter(|(_, live)| live.endpoint == Some(endpoint))
        .map(|(&address, _)| address).collect::<Vec<_>>();
    for address in addresses {
        let live = state.live.remove(&address).expect("endpoint backing retained under lock");
        release(&mut state, live);
        audit_event(&mut state, AllocationEvent::Freed(AllocationRecord {
            address, key: live.key, bytes: live.bytes, endpoint: live.endpoint,
        }));
    }
}

/// One aggregated ledger row.
#[derive(Clone, Debug)]
pub struct Row {
    pub key: Key,
    pub bytes: usize,
    pub allocations: usize,
}

/// Backing a direct device allocation reserves beyond its requested bytes.
///
/// SM120, driver/runtime 13020, LAZY: requests above one MiB round
/// independently to the 2 MiB granule; smaller requests share driver blocks,
/// so the group carries one rounding (never one per request). Qualified on
/// raptor's SM120 cards (#1511); other pools report zero.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Granularity {
    /// Rounding beyond the requests, already included in measured bytes used.
    pub tail: usize,
    /// Live requests above one MiB, each with its own backing.
    pub large_requests: usize,
    /// Live request bytes at or below one MiB, sharing driver blocks.
    pub small_bytes: usize,
}

const GRANULARITY_PAGE: usize = 2 << 20;
const GRANULARITY_SMALL: usize = 1 << 20;

fn round_to_granule(bytes: usize) -> usize {
    bytes.div_ceil(GRANULARITY_PAGE) * GRANULARITY_PAGE
}

/// Live bytes by key, plus the peak per pool and device since start.
#[derive(Clone, Debug, Default)]
pub struct Snapshot {
    pub rows: Vec<Row>,
    /// Full tensor names uploaded in more than one non-scale format, per device.
    pub duals: Vec<(i32, &'static str, Vec<&'static str>)>,
    pub peak: BTreeMap<(Space, i32), usize>,
    /// Per device pool: the live allocations' own rounding (zero off SM120).
    pub granularity: BTreeMap<(Space, i32), Granularity>,
}

impl Snapshot {
    /// Devices holding any tracked device or managed allocation.
    pub fn devices(&self) -> Vec<i32> {
        let mut devices: Vec<i32> = self.rows.iter().filter(|r| r.key.space != Space::Pinned && r.key.device >= 0)
            .map(|r| r.key.device).collect();
        devices.sort_unstable();
        devices.dedup();
        devices
    }

    pub fn total(&self, space: Space, device: i32) -> usize {
        self.rows.iter().filter(|r| r.key.space == space && r.key.device == device).map(|r| r.bytes).sum()
    }

    /// Bytes by category for one pool and device.
    pub fn by_scope(&self, space: Space, device: i32) -> BTreeMap<&'static str, usize> {
        let mut out = BTreeMap::new();
        for row in self.rows.iter().filter(|r| r.key.space == space && r.key.device == device) {
            *out.entry(row.key.scope).or_default() += row.bytes;
        }
        out
    }

    /// Bytes by kind. The census's untagged counter is always zero: a live
    /// backing cannot enter the ledger without a tag.
    pub fn by_kind(&self, space: Space, device: i32) -> (BTreeMap<MemoryKind, usize>, usize) {
        let mut kinds = BTreeMap::new();
        for row in self.rows.iter().filter(|r| r.key.space == space && r.key.device == device) {
            *kinds.entry(row.key.tag.kind).or_default() += row.bytes;
        }
        (kinds, 0)
    }

    /// Bytes by accounting [`owner`] guard for one pool and device.
    pub fn by_owner(&self, space: Space, device: i32) -> BTreeMap<&'static str, usize> {
        let mut out = BTreeMap::new();
        for row in self.rows.iter().filter(|r| r.key.space == space && r.key.device == device) {
            if let Some(owner) = row.key.owner {
                *out.entry(owner).or_default() += row.bytes;
            }
        }
        out
    }

    /// Tensors uploaded to one device in more than one format (scale
    /// companions excluded): the dual-residency check, by full tensor name.
    pub fn dual_formats(&self) -> Vec<(i32, &'static str, Vec<&'static str>)> {
        self.duals.clone()
    }
}

pub fn snapshot() -> Snapshot {
    let state = state().lock().unwrap_or_else(|e| e.into_inner());
    let mut granularity: BTreeMap<(Space, i32), Granularity> = BTreeMap::new();
    for live in state.live.values() {
        // Only discrete device memory carries the SM120 2 MiB granule; managed
        // pools and pinned/host backings are owned by their own rules.
        if live.key.space != Space::Device { continue; }
        let entry = granularity.entry((live.key.space, live.key.device)).or_default();
        if live.bytes > GRANULARITY_SMALL {
            entry.large_requests += 1;
            entry.tail += round_to_granule(live.bytes) - live.bytes;
        } else {
            entry.small_bytes += live.bytes;
        }
    }
    for entry in granularity.values_mut() {
        entry.tail += round_to_granule(entry.small_bytes) - entry.small_bytes;
    }
    Snapshot {
        rows: state.totals.iter().map(|(&key, &(bytes, allocations))| Row { key, bytes, allocations }).collect(),
        duals: state.formats.iter().filter(|(_, f)| f.len() > 1)
            .map(|(&(device, name), formats)| (device, name, formats.clone())).collect(),
        peak: state.peak.clone(),
        granularity,
    }
}

/// Allocation state captured before a native error unwinds owning buffers.
#[derive(Debug)]
pub struct FailureSnapshot {
    pub ledger: Snapshot,
    pub device_memory: BTreeMap<i32, (usize, usize)>,
    pub report: Option<Arc<fault::FailureReport>>,
    source: anyhow::Error,
}

impl std::fmt::Display for FailureSnapshot {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Display::fmt(&self.source, formatter)
    }
}

impl std::error::Error for FailureSnapshot {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> { Some(self.source.as_ref()) }
}

impl FailureSnapshot {
    pub fn new(source: anyhow::Error, ledger: Snapshot,
        device_memory: BTreeMap<i32, (usize, usize)>) -> Self {
        Self { ledger, device_memory, source, report: None }
    }

    fn into_error(self) -> anyhow::Error {
        // Keep evidence in the source chain for typed startup wrappers, and
        // expose the native tag through anyhow's direct context downcasts too.
        let native = self.source.downcast_ref::<crate::NativeStatusError>().cloned();
        let error = anyhow::Error::new(self);
        match native {
            Some(native) => error.context(native),
            None => error,
        }
    }
}

pub(crate) fn capture_failure(error: anyhow::Error) -> anyhow::Error {
    if error.chain().any(|cause| cause.is::<FailureSnapshot>()) {
        fault::clear_failure();
        return error;
    }
    let report = fault::take_failure().or_else(|| {
        let memory = error.downcast_ref::<crate::NativeStatusError>()
            .is_some_and(|status| status.status == crate::CUTEAFD_STATUS_ALLOCATION_FAILED)
            || error.downcast_ref::<cuteafd_core::serving_capacity::CapacityError>()
                .is_some_and(|error| matches!(error, cuteafd_core::serving_capacity::CapacityError::GpuBudgetExceeded { .. }));
        memory.then(fault::fallback).flatten()
    });
    let ledger = snapshot();
    // A native callback has already sampled before cleanup. Do not replace it
    // with post-cleanup telemetry or change the failing thread's device.
    let device_memory = if report.is_none() {
        ledger.devices().into_iter().filter_map(|device| fault::device_memory(device).map(|value| (device, value))).collect()
    } else { BTreeMap::new() };
    let mut failure = FailureSnapshot::new(error, ledger, device_memory);
    failure.report = report;
    failure.into_error()
}

fn loaded_cuda_runtime() -> Option<&'static libloading::os::unix::Library> {
    use libloading::os::unix::Library;
    static RUNTIME: OnceLock<Library> = OnceLock::new();
    match RUNTIME.get() {
        Some(runtime) => Some(runtime),
        None => {
            let found = ["libcudart.so.13", "libcudart.so.12"].iter().find_map(|soname| {
                // SAFETY: RTLD_NOLOAD only binds a runtime that is already loaded.
                unsafe { Library::open(Some(*soname), 2 | 0x4).ok() }
            })?;
            Some(RUNTIME.get_or_init(|| found))
        }
    }
}

fn cuda_pool_snapshot(mut query: impl FnMut(i32) -> Option<u64>) -> Option<serde_json::Value> {
    // CUDA 12/13 cudaMemPoolAttr IDs; report a failed query as unknown, not zero.
    Some(serde_json::json!({
        "reserved_current_bytes": query(5)?,
        "reserved_high_bytes": query(6)?,
        "used_current_bytes": query(7)?,
        "used_high_bytes": query(8)?,
    }))
}

/// Current-device physical memory and CUDA pool telemetry. No device allocation,
/// synchronization, device switch or runtime loading; unsupported pool queries
/// remain null, never zero. Call only at idle or a bounded diagnostic boundary.
pub fn current_cuda_memory_snapshot() -> Option<serde_json::Value> {
    use libloading::os::unix::Symbol;
    type GetDevice = unsafe extern "C" fn(*mut i32) -> i32;
    type MemGetInfo = unsafe extern "C" fn(*mut usize, *mut usize) -> i32;
    type GetPool = unsafe extern "C" fn(*mut *mut std::ffi::c_void, i32) -> i32;
    type PoolAttribute = unsafe extern "C" fn(*mut std::ffi::c_void, i32, *mut std::ffi::c_void) -> i32;
    let runtime = loaded_cuda_runtime()?;
    // SAFETY: signatures and attribute IDs match CUDA 12/13's runtime API.
    // Pointers name local outputs; queries do not mutate stream or pool state.
    unsafe {
        let device_fn: Symbol<GetDevice> = runtime.get(b"cudaGetDevice").ok()?;
        let info: Symbol<MemGetInfo> = runtime.get(b"cudaMemGetInfo").ok()?;
        let (mut device, mut free, mut total) = (0i32, 0usize, 0usize);
        if device_fn(&mut device) != 0 || info(&mut free, &mut total) != 0 { return None; }
        let mut pool_data = serde_json::Value::Null;
        if let (Ok(get_pool), Ok(attribute)) = (runtime.get::<GetPool>(b"cudaDeviceGetMemPool"),
            runtime.get::<PoolAttribute>(b"cudaMemPoolGetAttribute")) {
            let mut pool = std::ptr::null_mut();
            if get_pool(&mut pool, device) == 0 && !pool.is_null() {
                pool_data = cuda_pool_snapshot(|id| {
                    let mut value = 0u64;
                    (attribute(pool, id, (&mut value as *mut u64).cast()) == 0).then_some(value)
                }).unwrap_or(serde_json::Value::Null);
            }
        }
        Some(serde_json::json!({"device": device, "physical_free_bytes": free,
            "physical_total_bytes": total, "cuda_current_pool": pool_data,
            "scope": "raw cudaMemGetInfo under-reports reclaimable page cache on GB10; use host MemAvailable for allocatable headroom; current process/device CUDA runtime queries; no synchronization or pool trimming; CUDA pool bytes are physical telemetry, not additional tracked allocations"}))
    }
}

/// Scratch allocated by this crate's tests. Strictly `cfg(test)`: no feature
/// exposes it, so no serving build can borrow it instead of stating a kind.
/// Each consuming crate declares its own test-only constant.
#[cfg(test)]
pub(crate) const TEST: Tag = Tag::new(MemoryKind::Workspace, "test/scratch");

/// Hold for the whole test, including allocation cleanup: audits observe all
/// threads' events, not just the allocations owned by the current test.
#[cfg(test)]
pub(crate) fn test_guard() -> std::sync::MutexGuard<'static, ()> {
    static TEST_LOCK: Mutex<()> = Mutex::new(());
    TEST_LOCK.lock().unwrap_or_else(|error| error.into_inner())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn native_failure_retains_owners_before_unwind() {
        let _ledger = test_guard();
        let address = 0x7600_0000_0000usize;
        let _scope = scope("workspace/failure-fixture");
        record_alloc(Space::Device, 31, address, 512, TEST);
        let error = capture_failure(anyhow::anyhow!("native fixture error"));
        record_free(address);
        let saved = error.downcast_ref::<FailureSnapshot>().unwrap();
        assert_eq!(saved.ledger.by_scope(Space::Device, 31)["workspace/failure-fixture"], 512);
        assert_eq!(snapshot().total(Space::Device, 31), 0);
        assert!(format!("{error:#}").contains("native fixture error"));
    }

    #[test]
    fn failure_evidence_preserves_direct_typed_downcasts() {
        let _ledger = test_guard();
        let source = anyhow::Error::new(crate::NativeStatusError {
            context: "snapshot fixture".into(), status: 98, detail: "native detail".into(),
        }).context("startup context");
        let error = capture_failure(source).context("family initialization");
        let native = error.downcast_ref::<crate::NativeStatusError>().unwrap();
        assert_eq!(native.status, 98);
        assert_eq!(native.detail, "native detail");
        assert!(error.downcast_ref::<FailureSnapshot>().is_some());
        assert!(error.chain().any(|cause| cause.is::<FailureSnapshot>()));
        let original = error.chain().find_map(|cause| cause.downcast_ref::<crate::NativeStatusError>()).unwrap();
        assert_eq!(native, original);
        assert!(!std::ptr::eq(native, original));
        assert_eq!(native.to_string(), original.to_string());
        assert!(format!("{error:#}").contains("startup context"));
    }

    #[test]
    fn endpoint_enrichment_preserves_native_site_growth_and_allocation_history() {
        let _guard = test_guard();
        let address = 0x7990_0000_0000;
        let audit = AllocationAudit::begin();
        record_native_alloc(Space::Host, -1, address, 64, "native-ring.cc:99", TEST);
        let growth = state().lock().unwrap().post_ready_bytes;
        {
            let _scope = scope("transport/rdma-rings");
            record_endpoint_alloc(Space::Host, address, 64, 123, Tag::transport("transport/rdma-rings"));
        }
        {
            let state = state().lock().unwrap();
            assert_eq!(state.post_ready_bytes, growth);
            let live = state.live.get(&address).unwrap();
            assert_eq!(live.native_site, Some("native-ring.cc:99"));
            assert_eq!(live.key.scope, "transport/rdma-rings");
        }
        record_endpoint_free(123);
        let history = audit.finish();
        assert_eq!(history.events.iter().filter(|e| matches!(e, AllocationEvent::Allocated(r) if r.address == address)).count(), 1);
        assert!(history.events.iter().any(|e| matches!(e, AllocationEvent::Enriched(r) if r.address == address)));
        assert!(!history.events.iter().any(|e| matches!(e, AllocationEvent::UnobservedRelease(r) if r.address == address)));
    }

    #[test]
    fn tags_name_the_kind_independently_of_scope() {
        let _ledger = test_guard();
        let base = 0x7800_0000_0000usize;
        let kv = tag(MemoryKind::Kv, &format!("test/kv-{}", 3));
        {
            // Context scopes refine the label but never the kind.
            let _scope = scope("weights/misleading-context");
            record_alloc(Space::Device, 41, base, 4096, kv);
        }
        record_alloc(Space::Device, 41, base + 1, 512, Tag::new(MemoryKind::Workspace, "test/lane"));
        record_tagged(Space::Device, 41, base + 3, 256, Tag::new(MemoryKind::Weights, "test/media"));
        let snap = snapshot();
        let (kinds, untagged) = snap.by_kind(Space::Device, 41);
        assert_eq!(kinds.into_iter().collect::<Vec<_>>(),
            [(MemoryKind::Weights, 256), (MemoryKind::Kv, 4096), (MemoryKind::Workspace, 512)]);
        assert_eq!(untagged, 0);
        let row = snap.rows.iter().find(|r| r.key.device == 41 && r.key.tag == kv).unwrap();
        assert_eq!((row.key.scope, row.key.tag.owner), ("weights/misleading-context", "test/kv-3"));
        let lane = snap.rows.iter().find(|r| r.key.device == 41 && r.bytes == 512).unwrap();
        assert_eq!(lane.key.scope, "test/lane", "an unscoped tagged allocation is labelled by its owner");
        for ptr in base..base + 3 { record_free(ptr); }
        record_released(base + 3);
        assert_eq!(snapshot().total(Space::Device, 41), 0);
    }

    /// Accounting owners come only from `owner()` guards; a tag's component
    /// owner never fills or overrides them, so by_owner sums stay exactly the
    /// guarded allocations.
    #[test]
    fn tag_owner_is_not_an_accounting_owner() {
        let _ledger = test_guard();
        let base = 0x7900_0000_0000usize;
        record_alloc(Space::Device, 42, base, 64, Tag::workspace("test/component"));
        {
            let _owner = owner("test/accounting");
            record_alloc(Space::Device, 42, base + 1, 128, Tag::workspace("test/component"));
        }
        let snap = snapshot();
        assert_eq!(snap.by_owner(Space::Device, 42).into_iter().collect::<Vec<_>>(), [("test/accounting", 128)]);
        let rows = snap.rows.iter().filter(|r| r.key.device == 42).collect::<Vec<_>>();
        assert!(rows.iter().all(|r| r.key.tag.owner == "test/component"));
        assert_eq!(rows.iter().filter(|r| r.key.owner.is_none()).map(|r| r.bytes).sum::<usize>(), 64);
        record_free(base);
        record_free(base + 1);
    }

    #[test]
    fn startup_history_preserves_transients_reuse_and_initial_owners() {
        let _ledger = test_guard();
        let base = 0x7300_0000_0000usize;
        record_alloc(Space::Pinned, -1, base, 64, TEST);
        let audit = AllocationAudit::begin();
        let position = audit.checkpoint();
        let (sample_position, live) = audit.checkpoint_allocations();
        assert_eq!(sample_position, position);
        assert!(live.iter().any(|r| r.address == base && r.bytes == 64));
        record_alloc(Space::Device, 17, base + 1, 128, TEST);
        record_free(base + 1);
        record_alloc(Space::Device, 17, base + 1, 256, TEST);
        record_alloc(Space::Device, 17, base + 1, 512, TEST);
        record_free(base + 1);
        let history = audit.finish();
        assert!(history.initial.iter().any(|r| r.address == base && r.bytes == 64));
        assert!(history.final_allocations.iter().any(|r| r.address == base && r.bytes == 64));
        assert!(!history.final_allocations.iter().any(|r| r.address == base + 1));
        let events = history.events[position..].iter().filter(|event| match event {
            AllocationEvent::Allocated(r) | AllocationEvent::Freed(r)
                | AllocationEvent::Enriched(r) | AllocationEvent::UnobservedRelease(r) => r.address == base + 1,
        }).copied().collect::<Vec<_>>();
        let record = |bytes| AllocationRecord { address: base + 1, bytes, endpoint: None,
            key: Key { space: Space::Device, device: 17, tag: TEST, scope: TEST.owner,
                owner: None, tensor: None, format: None } };
        assert_eq!(events, [AllocationEvent::Allocated(record(128)),
            AllocationEvent::Freed(record(128)), AllocationEvent::Allocated(record(256)),
            AllocationEvent::UnobservedRelease(record(256)),
            AllocationEvent::Allocated(record(512)), AllocationEvent::Freed(record(512))]);
        record_free(base);
    }

    #[test]
    fn dropped_audit_does_not_retain_steady_state_history() {
        let _ledger = test_guard();
        let audit = AllocationAudit::begin();
        let events = Arc::downgrade(&audit.events);
        drop(audit);
        assert!(events.upgrade().is_none());
        let base = 0x7400_0000_0000usize;
        record_alloc(Space::Managed, 18, base, 32, TEST);
        let live = allocations();
        assert!(live.iter().any(|r| r.address == base && r.key.space == Space::Managed
            && r.key.device == 18 && r.bytes == 32));
        record_free(base);
        assert!(!allocations().iter().any(|r| r.address == base));
    }

    #[test]
    fn scopes_tensors_and_frees_balance() {
        let _ledger = test_guard();
        let base = 0x7000_0000_0000usize;
        {
            let _weights = scope("test/weights");
            tensor("model.layers.3.self_attn.o_proj.weight");
            {
                let _fp8 = format("fp8");
                record_alloc(Space::Device, 9, base, 1000, TEST);
                let _scale = format("fp8-scale");
                record_alloc(Space::Device, 9, base + 1, 10, TEST);
            }
            let _bf16 = format("bf16");
            record_alloc(Space::Device, 9, base + 2, 2000, TEST);
        }
        record_alloc(Space::Device, 9, base + 3, 5, TEST);
        let snap = snapshot();
        let scopes = snap.by_scope(Space::Device, 9);
        assert_eq!(scopes["test/weights"], 3010);
        assert_eq!(scopes[TEST.owner], 5);
        let dual = snap.dual_formats();
        assert_eq!(dual, vec![(9, "model.layers.3.self_attn.o_proj.weight", vec!["fp8", "bf16"])]);
        for ptr in base..base + 4 {
            record_free(ptr);
        }
        assert_eq!(snapshot().total(Space::Device, 9), 0);
        assert_eq!(snapshot().peak[&(Space::Device, 9)], 3015);
    }

    /// Graph capture and instantiate allocate driver storage this ledger never
    /// sees: a subject names that work in a failure dump without pretending it
    /// is a checkpoint tensor (no stem, so it never joins the weights table).
    #[test]
    fn subject_names_non_tensor_work_without_a_weights_stem() {
        let _ledger = test_guard();
        {
            let _scope = scope("test/graph");
            let _subject = subject("decode graph rank=0 segment=1 rows=64");
            let (key, name) = current_key(Space::Device, 12, TEST);
            assert_eq!(name, Some("decode graph rank=0 segment=1 rows=64"));
            assert_eq!(key.tensor, None);
            assert_eq!(key.scope, "test/graph");
        }
        assert_eq!(CONTEXT.with(|context| context.borrow().name), None);
    }

    /// A category scope nested inside an owner relabels the category but keeps
    /// the owner, so plan == runtime sums every allocation a component makes.
    #[test]
    fn owners_survive_nested_category_scopes() {
        let _ledger = test_guard();
        let base = 0x7100_0000_0000usize;
        {
            let _owner = owner("test/lanes/head");
            let _outer = scope("test/lanes/head");
            record_alloc(Space::Device, 11, base, 100, TEST);
            {
                let _inner = scope("test/sampler");
                record_alloc(Space::Device, 11, base + 1, 20, TEST);
            }
        }
        record_alloc(Space::Device, 11, base + 2, 3, TEST);
        let snap = snapshot();
        assert_eq!(snap.by_owner(Space::Device, 11).into_iter().collect::<Vec<_>>(), vec![("test/lanes/head", 120)]);
        assert_eq!(snap.by_scope(Space::Device, 11)["test/sampler"], 20);
        for ptr in base..base + 3 {
            record_free(ptr);
        }
    }

    #[test]
    fn cuda_pool_queries_preserve_current_high_and_real_zero() {
        let mut ids = Vec::new();
        let pool = cuda_pool_snapshot(|id| {
            ids.push(id);
            Some([4096, 8192, 0, 2048][(id - 5) as usize])
        }).unwrap();
        assert_eq!(ids, [5, 6, 7, 8]);
        assert_eq!(pool["reserved_current_bytes"], 4096);
        assert_eq!(pool["reserved_high_bytes"], 8192);
        assert_eq!(pool["used_current_bytes"], 0);
        assert_eq!(pool["used_high_bytes"], 2048);
    }

    #[test]
    fn cuda_pool_failed_attribute_is_unknown_not_zero() {
        for failed in 5..=8 {
            assert!(cuda_pool_snapshot(|id| (id != failed).then_some(0)).is_none());
        }
        assert!(cuda_pool_snapshot(|_| None).is_none());
    }

    #[test]
    fn stems_replace_indices() {
        assert_eq!(tensor_stem("model.layers.12.mlp.experts.7.w1"), "model.layers.*.mlp.experts.*.w1");
    }
}
