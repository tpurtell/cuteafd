//! Process-wide ledger of every device, host, pinned-host and managed allocation made
//! through [`crate::NativeLibrary`].
//!
//! Callers label allocations without touching each call site: a thread-local
//! scope stack names the category (`weights`, `kv`, `workspace/prefill`, ...),
//! and weight readers name the checkpoint tensor and resident format they are
//! about to upload. Frees look the pointer up, so the ledger always holds the
//! live bytes. Allocations nobody labelled land in `other`, which shows where a
//! scope is still missing; memory the runtime takes without this library
//! (context, modules, cuBLAS, graph executables) is the gap between the ledger
//! and `cudaMemGetInfo`.

use std::cell::RefCell;
use std::collections::{BTreeMap, HashMap, HashSet};
use std::sync::{Arc, Mutex, OnceLock, Weak};

/// Which pool an allocation came from.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Space {
    Device,
    Host,
    Pinned,
    Managed,
}

impl Space {
    pub fn label(self) -> &'static str {
        match self {
            Self::Device => "device",
            Self::Host => "host",
            Self::Pinned => "pinned",
            Self::Managed => "managed",
        }
    }
}

#[derive(Default)]
struct Context {
    scopes: Vec<&'static str>,
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
}

/// Aggregation key: pool, device (-1 for host memory), category, tensor stem
/// (layer indices replaced by `*`) and resident format.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Key {
    pub space: Space,
    pub device: i32,
    pub scope: &'static str,
    /// The enclosing [`owner`] tag, if any. Category scopes nested inside an
    /// owner (`sampler`, `workspace/fp8-linear`) relabel `scope` but keep it.
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
    audits: Vec<Weak<Mutex<Vec<AllocationEvent>>>>,
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

fn current_key(space: Space, device: i32) -> (Key, Option<&'static str>) {
    CONTEXT.with(|context| {
        let context = context.borrow();
        (Key {
            space,
            device,
            scope: context.scopes.last().copied().unwrap_or("other"),
            owner: context.owner,
            tensor: context.tensor,
            format: context.format,
        }, context.name)
    })
}

pub(crate) fn record_alloc(space: Space, device: i32, ptr: usize, bytes: usize) {
    if ptr == 0 || bytes == 0 {
        return;
    }
    let (key, name) = current_key(space, device);
    let mut state = state().lock().unwrap_or_else(|e| e.into_inner());
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
    if let Some(previous) = state.live.insert(ptr, Live { key, bytes }) {
        // A pointer reused without a recorded free (freed outside the library).
        release(&mut state, previous);
    }
    audit_event(&mut state, AllocationEvent::Allocated(AllocationRecord { address: ptr, key, bytes }));
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
        audit_event(&mut state, AllocationEvent::Freed(AllocationRecord { address: ptr, key: live.key, bytes: live.bytes }));
        release(&mut state, live);
    }
}

/// Moves every live allocation still labelled `other` to `label`: stage
/// checkpoints use it to name everything allocated since the previous one
/// without scoping each allocation site.
pub fn relabel_other(label: &'static str) {
    let mut state = state().lock().unwrap_or_else(|e| e.into_inner());
    let moved: Vec<(usize, Live)> = state.live.iter().filter(|(_, live)| live.key.scope == "other")
        .map(|(&ptr, &live)| (ptr, live)).collect();
    for (ptr, live) in moved {
        release(&mut state, live);
        let key = Key { scope: label, ..live.key };
        let total = state.totals.entry(key).or_default();
        total.0 += live.bytes;
        total.1 += 1;
        if let Some(current) = state.current.get_mut(&(key.space, key.device)) {
            *current += live.bytes;
        }
        state.live.insert(ptr, Live { key, bytes: live.bytes });
    }
}

/// One physical owner, without tensor-stem aggregation. Startup reconciliation
/// uses addresses to join actual Rust owners to the complete allocation census.
#[derive(Clone, Copy, Debug)]
pub struct AllocationRecord { pub address: usize, pub key: Key, pub bytes: usize }

#[derive(Clone, Copy, Debug)]
pub enum AllocationEvent { Allocated(AllocationRecord), Freed(AllocationRecord) }

/// Startup-only event history. Dropping the audit releases its history; normal
/// serving keeps no unbounded allocation event log.
pub struct AllocationAudit { events: Arc<Mutex<Vec<AllocationEvent>>> }
impl AllocationAudit {
    pub fn start() -> Self {
        let events = Arc::new(Mutex::new(Vec::new()));
        state().lock().unwrap_or_else(|e| e.into_inner()).audits.push(Arc::downgrade(&events));
        Self { events }
    }
    pub fn events(&self) -> Vec<AllocationEvent> {
        self.events.lock().unwrap_or_else(|e| e.into_inner()).clone()
    }
}
fn audit_event(state: &mut State, event: AllocationEvent) {
    state.audits.retain(|audit| {
        if let Some(events) = audit.upgrade() {
            events.lock().unwrap_or_else(|e| e.into_inner()).push(event);
            true
        } else { false }
    });
}

pub fn allocations() -> Vec<AllocationRecord> {
    let state = state().lock().unwrap_or_else(|e| e.into_inner());
    state.live.iter().map(|(&address, live)| AllocationRecord { address, key: live.key, bytes: live.bytes }).collect()
}

/// One aggregated ledger row.
#[derive(Clone, Debug)]
pub struct Row {
    pub key: Key,
    pub bytes: usize,
    pub allocations: usize,
}

/// Live bytes by key, plus the peak per pool and device since start.
#[derive(Clone, Debug, Default)]
pub struct Snapshot {
    pub rows: Vec<Row>,
    /// Full tensor names uploaded in more than one non-scale format, per device.
    pub duals: Vec<(i32, &'static str, Vec<&'static str>)>,
    pub peak: BTreeMap<(Space, i32), usize>,
}

impl Snapshot {
    /// Devices holding any tracked device or managed allocation.
    pub fn devices(&self) -> Vec<i32> {
        let mut devices: Vec<i32> = self.rows.iter()
            .filter(|r| matches!(r.key.space, Space::Device | Space::Managed) && r.key.device >= 0)
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

    /// Bytes by [`owner`] tag for one pool and device (untagged rows omitted).
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
    Snapshot {
        rows: state.totals.iter().map(|(&key, &(bytes, allocations))| Row { key, bytes, allocations }).collect(),
        duals: state.formats.iter().filter(|(_, f)| f.len() > 1)
            .map(|(&(device, name), formats)| (device, name, formats.clone())).collect(),
        peak: state.peak.clone(),
    }
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

/// Free and total bytes of `device` from the CUDA runtime already in the
/// process image (never loads a second runtime), queried on the calling
/// thread: callers that use the device themselves must restore their device.
/// `None` before the native library has loaded the runtime.
pub fn device_memory(device: i32) -> Option<(usize, usize)> {
    use libloading::os::unix::Symbol;
    type SetDevice = unsafe extern "C" fn(i32) -> i32;
    type MemGetInfo = unsafe extern "C" fn(*mut usize, *mut usize) -> i32;
    let runtime = loaded_cuda_runtime()?;
    // SAFETY: symbol types match the CUDA runtime API; both calls only touch
    // this thread's device selection and write two integers.
    unsafe {
        let set: Symbol<SetDevice> = runtime.get(b"cudaSetDevice").ok()?;
        let info: Symbol<MemGetInfo> = runtime.get(b"cudaMemGetInfo").ok()?;
        if set(device) != 0 {
            return None;
        }
        let (mut free, mut total) = (0usize, 0usize);
        (info(&mut free, &mut total) == 0).then_some((free, total))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scopes_tensors_and_frees_balance() {
        let base = 0x7000_0000_0000usize;
        {
            let _weights = scope("test/weights");
            tensor("model.layers.3.self_attn.o_proj.weight");
            {
                let _fp8 = format("fp8");
                record_alloc(Space::Device, 9, base, 1000);
                let _scale = format("fp8-scale");
                record_alloc(Space::Device, 9, base + 1, 10);
            }
            let _bf16 = format("bf16");
            record_alloc(Space::Device, 9, base + 2, 2000);
        }
        record_alloc(Space::Device, 9, base + 3, 5);
        let snap = snapshot();
        let scopes = snap.by_scope(Space::Device, 9);
        assert_eq!(scopes["test/weights"], 3010);
        assert_eq!(scopes["other"], 5);
        let dual = snap.dual_formats();
        assert_eq!(dual, vec![(9, "model.layers.3.self_attn.o_proj.weight", vec!["fp8", "bf16"])]);
        for ptr in base..base + 4 {
            record_free(ptr);
        }
        assert_eq!(snapshot().total(Space::Device, 9), 0);
        assert_eq!(snapshot().peak[&(Space::Device, 9)], 3015);
    }

    /// A category scope nested inside an owner relabels the category but keeps
    /// the owner, so plan == runtime sums every allocation a component makes.
    #[test]
    fn owners_survive_nested_category_scopes() {
        let base = 0x7100_0000_0000usize;
        {
            let _owner = owner("test/lanes/head");
            let _outer = scope("test/lanes/head");
            record_alloc(Space::Device, 11, base, 100);
            {
                let _inner = scope("test/sampler");
                record_alloc(Space::Device, 11, base + 1, 20);
            }
        }
        record_alloc(Space::Device, 11, base + 2, 3);
        let snap = snapshot();
        assert_eq!(snap.by_owner(Space::Device, 11).into_iter().collect::<Vec<_>>(), vec![("test/lanes/head", 120)]);
        assert_eq!(snap.by_scope(Space::Device, 11)["test/sampler"], 20);
        for ptr in base..base + 3 {
            record_free(ptr);
        }
    }

    #[test]
    fn allocation_audit_tracks_unaggregated_owners_frees_and_pointer_reuse() {
        let base = 0x7200_0000_0000usize;
        let audit = AllocationAudit::start();
        let observer = AllocationAudit::start();
        {
            let _scope = scope("test/startup-audit");
            let _owner = owner("test/actual-owner");
            record_alloc(Space::Device, 12, base, 1024);
            record_alloc(Space::Pinned, -1, base + 4096, 64);
        }
        let live = allocations();
        let device = live.iter().find(|a| a.address == base).unwrap();
        assert_eq!((device.key.space, device.key.device, device.key.owner, device.bytes),
            (Space::Device, 12, Some("test/actual-owner"), 1024));
        record_free(base);
        record_alloc(Space::Managed, 12, base, 2048);
        record_free(base);
        record_free(base + 4096);
        let own_events = |audit: &AllocationAudit| audit.events().into_iter().filter(|event| {
            let a = match event { AllocationEvent::Allocated(a) | AllocationEvent::Freed(a) => a };
            a.address == base || a.address == base + 4096
        }).collect::<Vec<_>>();
        let events = own_events(&audit);
        assert_eq!(events.len(), 6);
        assert!(matches!(events[0], AllocationEvent::Allocated(a) if a.bytes == 1024));
        assert!(matches!(events[2], AllocationEvent::Freed(a) if a.bytes == 1024));
        assert!(matches!(events[3], AllocationEvent::Allocated(a) if a.key.space == Space::Managed && a.bytes == 2048));
        assert_eq!(own_events(&observer).len(), events.len());
        assert!(!allocations().iter().any(|a| a.address == base || a.address == base + 4096));
        let history = Arc::downgrade(&audit.events);
        drop(audit);
        assert!(history.upgrade().is_none(), "the ledger must not retain dropped startup history");
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
