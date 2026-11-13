//! Failure-only streaming evidence, captured before native constructors clean up.
use super::{state, Context, Space, CONTEXT};
use serde::{ser::{SerializeMap, SerializeSeq}, Serialize, Serializer};
use std::{cell::{Cell, RefCell}, ffi::{c_char, CStr}, fs::{File, OpenOptions}, io::{self, Write}, path::{Path, PathBuf}, sync::{Arc, Mutex, OnceLock, RwLock, Weak}};
use cuteafd_core::memory_layout::{Category, MemoryKind};

const LIMITS: &[&str] = &[
    "plan: unavailable: accepted-plan registration not captured in this version",
    "arenas: partial: shared coordinator weight arenas only; native/private arenas not registered",
    "inputs: partial: daemon values; launcher provenance and unresolved family defaults not captured in this version",
    "kv: partial: resolved planner capacity/page geometry in inputs where available; live pool occupancy not captured in this version",
    "memory: partial: CUDA physical free/total and current-pool telemetry; opaque library allocations cannot be enumerated",
    "capture: partial: eight preopened reports per process; busy/exhausted captures retain an append-only fixed-size trigger record; interrupted/disk-full files may be incomplete",
    "identity: partial: host/role/thread/PID recorded; build/checkpoint/UUID provenance not captured in this version",
    "trigger: partial: raw native API code recorded; translated ABI status remains in typed error chain, not the pre-cleanup JSON",
    "memory: unavailable: NVML/graph/host extensions not captured in this version",
    "workers: unavailable: Spark worker delivery not captured in this version",
    "ledger: partial: instrumented Rust/native backings only; third-party private allocations are opaque",
];

#[repr(C)]
pub(crate) struct NativeFailure {
    pub api: *const c_char, pub domain: *const c_char, pub file: *const c_char,
    pub line: i32, pub code: i32, pub bytes: usize, pub bytes_known: i32, pub handle: usize,
}

#[derive(Debug)]
pub struct FailureReport { pub path: PathBuf, pub summary: String, pub complete: bool, pub sequence: u64, pub raw_code: i32, pub domain: String, pub api: String }

#[derive(Clone, Debug, Serialize)]
pub struct ArenaView { pub name: String, pub offset: usize, pub bytes: usize }
#[derive(Debug, Serialize)]
pub struct Arena {
    pub id: String, pub kind: MemoryKind, pub device: i32, pub bytes: usize,
    pub padding_bytes: usize, pub tail_bytes: Option<usize>, pub views: Vec<ArenaView>,
}
#[derive(Default)]
struct Attachments { arenas: std::collections::BTreeMap<u64, Arc<Arena>>, inputs: serde_json::Value, backings: std::collections::BTreeMap<u64, usize> }
fn attachments() -> &'static Mutex<Attachments> {
    static DATA: OnceLock<Mutex<Attachments>> = OnceLock::new(); DATA.get_or_init(Default::default)
}
/// Keep on the physical backing, not its short-lived loader: tensor views may
/// outlive DeviceWeightArena and must keep its diagnostic registration alive.
#[derive(Debug)]
pub struct ArenaRegistration(u64);
impl ArenaRegistration {
    pub fn bind(&self, address: usize) {
        attachments().lock().unwrap_or_else(|e| e.into_inner()).backings.insert(self.0, address);
    }
}
pub fn register_arena(arena: Arena) -> ArenaRegistration {
    static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);
    let id = NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    attachments().lock().unwrap_or_else(|e| e.into_inner()).arenas.insert(id, Arc::new(arena));
    ArenaRegistration(id)
}
impl Drop for ArenaRegistration {
    fn drop(&mut self) { let mut data = attachments().lock().unwrap_or_else(|e| e.into_inner()); data.arenas.remove(&self.0); data.backings.remove(&self.0); }
}
pub fn set_inputs(inputs: serde_json::Value) {
    let mut data = attachments().lock().unwrap_or_else(|e| e.into_inner());
    if !data.inputs.is_object() { data.inputs = serde_json::json!({}); }
    if let Some(inputs) = inputs.as_object() {
        for (name, value) in inputs {
            data.inputs.as_object_mut().unwrap().insert(name.clone(),
                serde_json::json!({"value": value, "source": "daemon"}));
        }
    }
}

/// A copy of the daemon's allowlisted, resolved admission inputs.
/// This performs no CUDA calls and never exposes arbitrary environment values.
pub fn inputs() -> serde_json::Value {
    let data = attachments().lock().unwrap_or_else(|e| e.into_inner());
    serde_json::Value::Object(data.inputs.as_object().into_iter().flatten()
        .map(|(key, value)| (key.clone(), value["value"].clone())).collect())
}

struct Service { spools: Vec<(File, PathBuf)>, directory: PathBuf, host: String, role: String, scratch: Box<[u8; 65536]> }
static SERVICE: OnceLock<Mutex<Service>> = OnceLock::new();
static TRIGGERS: OnceLock<(File, PathBuf)> = OnceLock::new();
/// Remove only this process's unused private spools on ordinary shutdown.
pub fn shutdown() {
    if let Some(service) = SERVICE.get() {
        let mut service = service.lock().unwrap_or_else(|e| e.into_inner());
        for (_, path) in service.spools.drain(..) { let _ = std::fs::remove_file(path); }
        if let Some((file, path)) = TRIGGERS.get() {
            if file.metadata().is_ok_and(|metadata| metadata.len() == 0) { let _ = std::fs::remove_file(path); }
        }
    }
}

thread_local! {
    static ACTIVE: Cell<bool> = const { Cell::new(false) };
    static PENDING: RefCell<Option<Arc<FailureReport>>> = const { RefCell::new(None) };
}

fn lock_spool(file: &File) -> io::Result<()> {
    use std::os::fd::AsRawFd;
    // SAFETY: flock only locks the live file descriptor; no pointer arguments.
    if unsafe { libc::flock(file.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) } == 0 { Ok(()) }
    else { Err(io::Error::last_os_error()) }
}

fn sweep_unused_spools(directory: &Path) -> io::Result<()> {
    use std::os::unix::fs::MetadataExt;
    for entry in std::fs::read_dir(directory)? {
        let entry = entry?;
        let name = entry.file_name();
        let Some(name) = name.to_str() else { continue };
        let (pid, trigger) = if let Some(rest) = name.strip_prefix(".memory-spool-") {
            let Some((pid, index)) = rest.split_once('-') else { continue };
            if !index.parse::<u32>().is_ok_and(|index| index < 8) { continue; }
            (pid, false)
        } else if let Some(rest) = name.strip_prefix("memory-triggers-").and_then(|name| name.strip_suffix(".jsonl")) {
            let Some((_, pid)) = rest.rsplit_once('-') else { continue };
            (pid, true)
        } else { continue };
        let Ok(pid) = pid.parse::<i32>() else { continue };
        if pid <= 0 { continue; }
        // SAFETY: signal zero probes process existence without sending a signal.
        let dead = unsafe { libc::kill(pid, 0) } != 0 && io::Error::last_os_error().raw_os_error() == Some(libc::ESRCH);
        if !dead && pid as u32 != std::process::id() { continue; }
        let Ok(file) = OpenOptions::new().read(true).custom_flags(libc::O_NOFOLLOW).open(entry.path()) else { continue };
        let metadata = file.metadata()?;
        // A restarted private PID namespace may reuse the same PID. The lock
        // distinguishes its stale files from a live process with that PID.
        // SAFETY: geteuid reads the process's effective UID.
        if !metadata.is_file() || metadata.uid() != unsafe { libc::geteuid() }
            || (trigger && metadata.len() != 0) || lock_spool(&file).is_err() { continue; }
        std::fs::remove_file(entry.path())?;
    }
    Ok(())
}

/// Preopen private storage and fixed host scratch before any native allocation.
/// No report file is published on successful startup unless census was asked for.
pub fn initialize(directory: &Path, host: &str, role: &str) -> io::Result<()> {
    if SERVICE.get().is_some() { return Ok(()); }
    std::fs::create_dir_all(directory)?;
    sweep_unused_spools(directory)?;
    let startup_ns = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap_or_default().as_nanos();
    let trigger_path = directory.join(format!("memory-triggers-{}-{startup_ns}-{}-{}-{}.jsonl", utc_timestamp(), sanitize(host), sanitize(role), std::process::id()));
    let trigger_file = OpenOptions::new().append(true).create_new(true).mode(0o600).open(&trigger_path)?;
    lock_spool(&trigger_file)?;
    let mut spools = Vec::new();
    for index in 0..8 {
        let path = directory.join(format!(".memory-spool-{}-{index}", std::process::id()));
        let file = match OpenOptions::new().write(true).create_new(true).mode(0o600).open(&path) {
            Ok(file) => file,
            Err(error) => {
                for (_, path) in spools { let _ = std::fs::remove_file(path); }
                let _ = std::fs::remove_file(trigger_path);
                return Err(error);
            }
        };
        lock_spool(&file)?;
        spools.push((file, path));
    }
    let service = Service { spools, directory: directory.into(),
        host: host.into(), role: sanitize(role), scratch: Box::new([0; 65536]) };
    let _ = TRIGGERS.set((trigger_file, trigger_path));
    let _ = SERVICE.set(Mutex::new(service));
    Ok(())
}
use std::os::unix::fs::OpenOptionsExt;
fn sanitize(text: &str) -> String { text.chars().map(|c| if c.is_ascii_alphanumeric() || matches!(c, '-' | '_') { c } else { '_' }).collect() }

/// Install after loading each native DSO; old native binaries remain usable but
/// explicitly lack the pre-cleanup hook.
type LastTag = unsafe extern "C" fn(*mut u32, *mut c_char, usize, *mut usize) -> i32;
struct TagModule {
    library: Weak<libloading::Library>,
    last: LastTag,
    ranges: Vec<std::ops::Range<usize>>,
}
static TAG_MODULES: RwLock<Vec<TagModule>> = RwLock::new(Vec::new());

/// Discover only at install; the failure path never calls the loader or allocates
/// registry storage. A native __FILE__ literal identifies its owning DSO.
fn module_ranges(last: LastTag) -> Vec<std::ops::Range<usize>> {
    struct Search { address: usize, ranges: Vec<std::ops::Range<usize>> }
    unsafe extern "C" fn visit(info: *mut libc::dl_phdr_info, _: usize, data: *mut std::ffi::c_void) -> i32 {
        // SAFETY: dl_iterate_phdr supplies a live header array and our stack Search.
        let (info, search) = unsafe { (&*info, &mut *data.cast::<Search>()) };
        // SAFETY: the loader retains all dlpi_phnum headers during this callback.
        let headers = unsafe { std::slice::from_raw_parts(info.dlpi_phdr, info.dlpi_phnum as usize) };
        let range = |header: &libc::Elf64_Phdr| {
            let start = (info.dlpi_addr as usize).checked_add(header.p_vaddr as usize)?;
            Some(start..start.checked_add(header.p_memsz as usize)?)
        };
        if !headers.iter().filter(|h| h.p_type == libc::PT_LOAD)
            .filter_map(range).any(|r| r.contains(&search.address)) { return 0; }
        search.ranges.extend(headers.iter().filter(|h| h.p_type == libc::PT_LOAD).filter_map(range));
        1
    }
    let mut search = Search { address: last as usize, ranges: Vec::new() };
    // SAFETY: visit uses only loader-owned headers and this live, exclusive Search.
    unsafe { libc::dl_iterate_phdr(Some(visit), (&mut search as *mut Search).cast()); }
    search.ranges
}

/// The tag the failing thread last attempted: kind label, owner (in `buffer`),
/// and whether that owner exceeded the fixed failure-path buffer.
fn attempted_tag(file: *const c_char, buffer: &mut [u8; 512]) -> (Option<&'static str>, Option<&str>, bool) {
    // Never wait behind install-time allocation (or a reentrant allocator failure).
    // Unknown/busy source is null, not another module's most recent owner.
    let source = TAG_MODULES.try_read().ok().and_then(|modules| modules.iter().rev()
        .filter(|module| module.ranges.iter().any(|range| range.contains(&(file as usize))))
        .find_map(|module| module.library.upgrade().map(|library| (library, module.last))));
    let Some((library, last)) = source else { return (None, None, false) };
    let (mut kind, mut required) = (0u32, 0usize);
    // SAFETY: library keeps this function mapped for its entire call. Native
    // writes the kind, required size and at most buffer.len() bytes.
    let status = unsafe { last(&mut kind, buffer.as_mut_ptr().cast(), buffer.len(), &mut required) };
    drop(library);
    let kind = MemoryKind::from_raw(kind).map(MemoryKind::label);
    if status != 0 { return (kind, None, required > buffer.len()); }
    let end = buffer.iter().position(|&b| b == 0).unwrap_or(0);
    (kind, std::str::from_utf8(&buffer[..end]).ok().filter(|o| !o.is_empty()), false)
}

pub(crate) fn install(library: &Arc<libloading::Library>) {
    // SAFETY: optional versioned export; a library without it reports no attempted tag.
    if let Ok(last) = unsafe { library.get::<LastTag>(b"cuteafd_last_allocation_tag") } {
        let module = TagModule { library: Arc::downgrade(library), last: *last, ranges: module_ranges(*last) };
        let mut modules = TAG_MODULES.write().unwrap_or_else(|e| e.into_inner());
        modules.retain(|entry| entry.library.strong_count() != 0 && !Weak::ptr_eq(&entry.library, &module.library));
        modules.push(module);
    }
    type Install = unsafe extern "C" fn(extern "C" fn(*const NativeFailure));
    // SAFETY: optional symbol implements the bundled POD callback ABI. The
    // callback is process-static and never unwinds into the native caller.
    let _ = cuda_queries();
    type Events = unsafe extern "C" fn(extern "C" fn(i32, usize, usize, i32, *const c_char, i32, u32, *const c_char));
    // SAFETY: bundled callback takes scalars and synchronously-copied strings, and stays process-live.
    if let Ok(events) = unsafe { library.get::<Events>(b"cuteafd_set_memory_event_callback_v3") } {
        unsafe { events(event_callback); }
    } else {
        tracing::warn!(target: "cuteafd::memory", "native private-backing events unavailable: tagged event ABI unsupported");
    }
    match unsafe { library.get::<Install>(b"cuteafd_set_memory_failure_callback") } {
        Ok(install) => unsafe { install(callback) },
        Err(_) => tracing::warn!(target: "cuteafd::memory", "failure capture unavailable: native pre-cleanup callback unsupported"),
    }
}
/// A native event names the tag native allocated under; the owner is copied
/// (interned) here, never retained. Invalid tags violate the bundled native ABI.
extern "C" fn event_callback(space: i32, address: usize, bytes: usize, released: i32, file: *const c_char, line: i32,
    kind: u32, owner: *const c_char) {
    let _ = std::panic::catch_unwind(|| {
        if released != 0 { super::record_free(address); return; }
        let space = match space { 1 => Space::Pinned, 2 => Space::Managed, 3 => Space::Host, _ => Space::Device };
        let mut device = -1;
        if matches!(space, Space::Device | Space::Managed) { if let Some(q) = cuda_queries() {
            // SAFETY: valid stack output, query does not select a device.
            unsafe { (q.get)(&mut device); }
        } }
        let tag = MemoryKind::from_raw(kind).zip((!owner.is_null()).then(|| native_text(owner)))
            .filter(|(_, owner)| !owner.is_empty()).map(|(kind, owner)| super::tag(kind, &owner));
        let Some(tag) = tag else {
            // A callback cannot return an error. Never hide owned backing by
            // silently discarding a malformed native allocation event.
            eprintln!("invalid native allocation tag at {}:{line}: kind={kind}", native_text(file));
            std::process::abort();
        };
        super::record_native_alloc(space, device, address, bytes, &format!("{}:{line}", native_text(file)), tag);
    });
}
extern "C" fn callback(trigger: *const NativeFailure) {
    if trigger.is_null() || ACTIVE.with(|active| active.replace(true)) { return; }
    let result = std::panic::catch_unwind(|| {
        // SAFETY: native wrapper retains the POD and its literal strings for
        // this synchronous callback; no pointer escapes the callback.
        let trigger = unsafe { &*trigger };
        let first = PENDING.with(|pending| pending.borrow().as_ref().map(|report| report.sequence));
        if let Some(first) = first {
            trigger_record(trigger, sequence(), Some(first), "linked: later failure before ABI return");
        } else if let Some(report) = capture(trigger, true) {
            PENDING.with(|pending| *pending.borrow_mut() = Some(report));
        }
    });
    ACTIVE.with(|active| active.set(false));
    if result.is_err() { eprintln!("memory failure capture unavailable: callback panicked; original native error retained"); }
}
pub(crate) fn fallback() -> Option<Arc<FailureReport>> {
    capture(&NativeFailure { api: c"Rust error boundary".as_ptr(), domain: c"unavailable: no native callback".as_ptr(),
        file: c"unavailable".as_ptr(), line: 0, code: 0, bytes: 0, bytes_known: 0, handle: 0 }, false)
}
/// Scope the few native constructors which may tolerate an allocation failure
/// and return success (for example a pageable fallback). Cleanup stays inside
/// this boundary, so it cannot overwrite or prematurely clear the first token.
#[must_use]
pub(crate) struct Boundary;
pub(crate) fn boundary() -> Boundary { clear_failure(); Boundary }
impl Drop for Boundary { fn drop(&mut self) { clear_failure(); } }
pub(crate) fn clear_failure() { PENDING.with(|pending| { pending.borrow_mut().take(); }); }
pub(crate) fn take_failure() -> Option<Arc<FailureReport>> { PENDING.with(|pending| pending.borrow_mut().take()) }

/// Reserved append-only evidence remains writable after detailed spools run out.
/// A bounded record is emitted in one write, without formatting heap buffers.
fn trigger_record(raw: &NativeFailure, id: u64, linked: Option<u64>, status: &str) {
    use std::os::fd::AsRawFd;
    let Some((file, _)) = TRIGGERS.get() else { return };
    let mut bytes = [0u8; 2048];
    let mut out = io::Cursor::new(bytes.as_mut_slice());
    #[derive(Serialize)]
    struct Record<'a> { sequence: u64, linked_to: Option<u64>, api: &'a str, domain: &'a str,
        code: i32, requested_bytes: Option<usize>, status: &'a str,
        attempted_kind: Option<&'static str>, attempted_owner: Option<&'a str>, attempted_owner_truncated: bool }
    let mut owner = [0u8; 512];
    let (attempted_kind, attempted_owner, attempted_owner_truncated) = attempted_tag(raw.file, &mut owner);
    let record = Record { sequence: id, linked_to: linked, api: native_text(raw.api),
        domain: native_text(raw.domain), code: raw.code,
        requested_bytes: (raw.bytes_known != 0).then_some(raw.bytes), status,
        attempted_kind, attempted_owner, attempted_owner_truncated };
    if serde_json::to_writer(&mut out, &record).is_ok() && out.write_all(b"\n").is_ok() {
        let length = out.position() as usize;
        // SAFETY: file is preopened append-only and bytes is live stack storage.
        let written = unsafe { libc::write(file.as_raw_fd(), bytes.as_ptr().cast(), length) };
        if written != length as isize { eprintln!("memory trigger record partial: write failed"); }
    } else { eprintln!("memory trigger record partial: trigger exceeds reserved buffer"); }
}

struct Buffered<'a> { file: &'a mut File, buffer: &'a mut [u8], used: usize }
impl Write for Buffered<'_> {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        if self.used == self.buffer.len() { self.flush()?; }
        let n = bytes.len().min(self.buffer.len() - self.used);
        self.buffer[self.used..self.used+n].copy_from_slice(&bytes[..n]); self.used += n; Ok(n)
    }
    fn flush(&mut self) -> io::Result<()> { self.file.write_all(&self.buffer[..self.used])?; self.used = 0; Ok(()) }
}

struct DisplayValue<T>(T);
impl<T: std::fmt::Display> Serialize for DisplayValue<T> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> { serializer.collect_str(&self.0) }
}
struct Hex(usize);
impl std::fmt::Display for Hex {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result { write!(f, "0x{:x}", self.0) }
}
impl Serialize for Hex {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> { serializer.collect_str(self) }
}
#[derive(Serialize)]
struct Trigger<'a> { api: &'a str, domain: &'a str, code: i32, file: &'a str, line: i32,
    requested_bytes: Option<usize>, handle: Hex, scope: &'a str, owner: Option<&'a str>,
    tensor: Option<&'a str>, tensor_stem: Option<&'a str>, format: Option<&'a str> }
fn native_text<'a>(ptr: *const c_char) -> &'a str {
    if ptr.is_null() { return "unavailable"; }
    // SAFETY: used only within the synchronous callback for literal native
    // strings supplied by the instrumented wrappers.
    unsafe { CStr::from_ptr(ptr) }.to_str().unwrap_or("unavailable: non-UTF8 native label")
}
struct LiveRows<'a>(&'a super::State);
impl Serialize for LiveRows<'_> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let mut seq = serializer.serialize_seq(Some(self.0.live.len()))?;
        for (address, live) in &self.0.live {
            #[derive(Serialize)]
            struct Row<'a> { address: Hex, bytes: usize, device: i32, space: &'a str, scope: &'a str,
                kind: MemoryKind, component: &'a str,
                category: Category, owner: Option<&'a str>, tensor: Option<&'a str>, tensor_stem: Option<&'a str>,
                format: Option<&'a str>, site: DisplayValue<&'a std::panic::Location<'static>>, native_site: Option<&'a str> }
            seq.serialize_element(&Row { address: Hex(*address), bytes: live.bytes,
                device: live.key.device, space: live.key.space.label(), scope: live.key.scope,
                kind: live.key.tag.kind, component: live.key.tag.owner,
                category: live.key.tag.kind.category(),
                owner: live.key.owner, tensor: live.name,
                tensor_stem: live.key.tensor, format: live.key.format, site: DisplayValue(live.site), native_site: live.native_site })?;
        }
        seq.end()
    }
}
struct Growth<'a>(&'a super::State);
impl Serialize for Growth<'_> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let mut map = serializer.serialize_map(None)?;
        map.serialize_entry("ready", &self.0.ready)?;
        map.serialize_entry("allocated_bytes_since_ready", &self.0.post_ready_bytes)?;
        struct Sites<'a>(&'a super::State);
        impl Serialize for Sites<'_> {
            fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
                let mut map = serializer.serialize_map(None)?;
                for (site, bytes) in &self.0.post_ready_sites { map.serialize_entry(&DisplayValue(*site), bytes)?; }
                for (site, bytes) in &self.0.post_ready_native_sites { map.serialize_entry(site, bytes)?; }
                map.end()
            }
        }
        map.serialize_entry("sites", &Sites(self.0))?; map.end()
    }
}

type GetDevice = unsafe extern "C" fn(*mut i32) -> i32;
type GetPool = unsafe extern "C" fn(*mut *mut std::ffi::c_void, i32) -> i32;
type PoolAttribute = unsafe extern "C" fn(*mut std::ffi::c_void, i32, *mut std::ffi::c_void) -> i32;
type GetContext = unsafe extern "C" fn(*mut *mut std::ffi::c_void) -> i32;
type SetContext = unsafe extern "C" fn(*mut std::ffi::c_void) -> i32;
type PrimaryState = unsafe extern "C" fn(i32, *mut u32, *mut i32) -> i32;
type Retain = unsafe extern "C" fn(*mut *mut std::ffi::c_void, i32) -> i32;
type Release = unsafe extern "C" fn(i32) -> i32;
type MemInfo = unsafe extern "C" fn(*mut usize, *mut usize) -> i32;
struct CudaQueries { get: GetDevice, current: GetContext, select: SetContext,
    state: PrimaryState, retain: Retain, release: Release, info: MemInfo,
    pool: Option<GetPool>, attribute: Option<PoolAttribute> }
fn cuda_queries() -> Option<&'static CudaQueries> {
    static QUERIES: OnceLock<CudaQueries> = OnceLock::new();
    static DRIVER: OnceLock<libloading::os::unix::Library> = OnceLock::new();
    if let Some(queries) = QUERIES.get() { return Some(queries); }
    let runtime = super::loaded_cuda_runtime()?;
    // SAFETY: NOLOAD binds the already-loaded driver; never initializes CUDA.
    let driver = match DRIVER.get() {
        Some(driver) => driver,
        None => {
            // SAFETY: NOLOAD refuses to initialize a driver not already loaded.
            let found = unsafe { libloading::os::unix::Library::open(Some("libcuda.so.1"), 2 | 0x4).ok()? };
            DRIVER.get_or_init(|| found)
        }
    };
    // SAFETY: documented driver/runtime query ABIs; both libraries stay live.
    let queries = unsafe { CudaQueries {
        get: *driver.get::<GetDevice>(b"cuCtxGetDevice").ok()?,
        current: *driver.get::<GetContext>(b"cuCtxGetCurrent").ok()?,
        select: *driver.get::<SetContext>(b"cuCtxSetCurrent").ok()?,
        state: *driver.get::<PrimaryState>(b"cuDevicePrimaryCtxGetState").ok()?,
        retain: *driver.get::<Retain>(b"cuDevicePrimaryCtxRetain").ok()?,
        release: *driver.get::<Release>(b"cuDevicePrimaryCtxRelease_v2").ok()?,
        info: *driver.get::<MemInfo>(b"cuMemGetInfo_v2").ok()?,
        pool: runtime.get::<GetPool>(b"cudaDeviceGetMemPool").ok().map(|symbol| *symbol),
        attribute: runtime.get::<PoolAttribute>(b"cudaMemPoolGetAttribute").ok().map(|symbol| *symbol),
    } };
    Some(QUERIES.get_or_init(|| queries))
}
#[derive(Serialize)]
struct PoolObservation { bytes: [Option<u64>; 4], raw_status: [Option<i32>; 4], status: &'static str }
impl PoolObservation {
    fn unavailable() -> Self { Self { bytes: [None; 4], raw_status: [None; 4], status: "unavailable: CUDA current-pool API unsupported, skipped, or failed" } }
}
#[derive(Serialize)]
struct Observation { device: i32, free_bytes: Option<usize>, total_bytes: Option<usize>, status: &'static str, raw_status: Option<i32>,
    cuda_current_pool: PoolObservation }
fn observe(device: i32, sticky: bool) -> Observation {
    let unavailable = |reason| Observation { device, free_bytes: None, total_bytes: None, status: reason, raw_status: None, cuda_current_pool: PoolObservation::unavailable() };
    if sticky { return unavailable("unavailable: context lost or sticky CUDA error; follow-up skipped"); }
    let Some(q) = cuda_queries() else { return unavailable("unavailable: CUDA runtime query API unsupported or not loaded"); };
    observe_with(q, device)
}
fn observe_with(q: &CudaQueries, device: i32) -> Observation {
    let unavailable = |reason, raw_status| Observation { device, free_bytes: None, total_bytes: None,
        status: reason, raw_status, cuda_current_pool: PoolObservation::unavailable() };
    let (mut previous, mut primary) = (std::ptr::null_mut(), std::ptr::null_mut());
    let (mut flags, mut active, mut free, mut total) = (0, 0, 0, 0);
    let mut pool = PoolObservation::unavailable();
    // SAFETY: stack outputs match the documented driver ABI. Inactive primary
    // contexts are never retained. Restore the exact saved context, including NULL.
    unsafe {
        let status = (q.current)(&mut previous);
        if status != 0 { return unavailable("unavailable: cuCtxGetCurrent failed", Some(status)); }
        let status = (q.state)(device, &mut flags, &mut active);
        if status != 0 { return unavailable("unavailable: primary-context state query failed", Some(status)); }
        if active == 0 { return unavailable("unavailable: primary context inactive; query skipped", None); }
        let status = (q.retain)(&mut primary, device);
        if status != 0 { return unavailable("unavailable: active primary-context retain failed", Some(status)); }
        let selected = (q.select)(primary);
        let status = if selected == 0 { (q.info)(&mut free, &mut total) } else { selected };
        if status == 0 { if let (Some(get_pool), Some(attribute)) = (q.pool, q.attribute) {
            let mut handle = std::ptr::null_mut();
            let pool_status = get_pool(&mut handle, device);
            if pool_status == 0 && !handle.is_null() {
                for (index, id) in [5, 6, 7, 8].into_iter().enumerate() {
                    let mut bytes = 0u64;
                    let result = attribute(handle, id, (&mut bytes as *mut u64).cast());
                    pool.raw_status[index] = Some(result);
                    pool.bytes[index] = (result == 0).then_some(bytes);
                }
                pool.status = "available/partial: reserved_current,reserved_high,used_current,used_high bytes; null means unknown; physical telemetry, not ownership";
            } else { pool.raw_status[0] = Some(pool_status); }
        } }
        let restored = (q.select)(previous);
        let released = (q.release)(device);
        if status != 0 || restored != 0 || released != 0 {
            return unavailable("unavailable: CUDA memory query/context restore/release failed",
                Some(if status != 0 { status } else if restored != 0 { restored } else { released }));
        }
    }
    Observation { device, free_bytes: Some(free), total_bytes: Some(total), status: "available: raw cudaMemGetInfo; device-wide, includes other processes; GB10 reclaimable page cache is not allocatable headroom here", raw_status: Some(0), cuda_current_pool: pool }
}

pub fn device_memory(device: i32) -> Option<(usize, usize)> {
    let value = observe(device, false);
    value.free_bytes.zip(value.total_bytes)
}

/// Active-context diagnostic sample with the same observer exclusion as admission.
/// Returns raw free/total and the independently measured external observer bytes.
pub fn device_memory_observer(device: i32) -> anyhow::Result<Option<(usize, usize, usize)>> {
    let observer = crate::observer_context::bytes(device)?;
    let Some((free, total)) = device_memory(device) else { return Ok(None) };
    crate::observer_context::exclude(free, total, observer)?;
    Ok(Some((free, total, observer)))
}

fn capture(raw: &NativeFailure, before_cleanup: bool) -> Option<Arc<FailureReport>> {
    let id = sequence();
    trigger_record(raw, id, None, "capture attempted");
    let Some(service) = SERVICE.get() else { eprintln!("memory failure dump unavailable: spool initialization not completed"); return None; };
    let Ok(mut service) = service.try_lock() else { trigger_record(raw, id, None, "detailed capture busy"); eprintln!("memory failure dump partial: concurrent capture busy; trigger domain={} code={} api={}", native_text(raw.domain), raw.code, native_text(raw.api)); return None; };
    let timestamp = utc_timestamp();
    let census = native_text(raw.domain) == "requested census";
    let prefix = if census { "memory-census" } else { "memory-failure" };
    let path = service.directory.join(format!("{prefix}-{timestamp}-{}-{}-{}-{}.json", sanitize(&service.host), service.role, std::process::id(), id));
    let Some((mut file, spool_path)) = service.spools.pop() else { trigger_record(raw, id, None, "detailed spools exhausted"); eprintln!("memory failure dump unavailable: preopened spool exhausted; original error retained"); return None; };
    let sticky = native_text(raw.domain) == "cuda" && matches!(raw.code, 700 | 709 | 710 | 714 | 715 | 716 | 717 | 718 | 719);
    let mut devices = [-1i32; 32];
    let mut device_count = 0;
    let result = (|| -> Result<(), Box<dyn std::error::Error>> {
        let Service { scratch, host, role, .. } = &mut *service;
        let mut out = Buffered { file: &mut file, buffer: scratch.as_mut_slice(), used: 0 };
        // Core records go first. No CUDA query runs while the ledger is locked.
        {
            let ledger = state().lock().map_err(|_| io::Error::other("ledger lock poisoned"))?;
            let data = attachments().lock().map_err(|_| io::Error::other("attachments lock poisoned"))?;
            out.write_all(b"{\"schema_version\":1,\"identity\":")?;
            #[derive(Serialize)] struct ThreadIdentity<'a> { sequence: u64, utc: &'a str, pid: u32, host: &'a str, role: &'a str, thread: u64, pre_native_cleanup: bool }
            // SAFETY: gettid reads the calling Linux thread's identity without allocation.
            let thread = unsafe { libc::syscall(libc::SYS_gettid) } as u64;
            serde_json::to_writer(&mut out, &ThreadIdentity { sequence: id, utc: &timestamp, pid: std::process::id(), host, role, thread, pre_native_cleanup: before_cleanup })?;
            out.write_all(b",\"trigger\":")?;
            CONTEXT.with(|context| -> Result<(), serde_json::Error> {
                let context = context.borrow();
                serde_json::to_writer(&mut out, &trigger(raw, &context))
            })?;
            out.write_all(b",\"live_allocations\":")?; serde_json::to_writer(&mut out, &LiveRows(&ledger))?;
            out.write_all(b",\"post_ready\":")?; serde_json::to_writer(&mut out, &Growth(&ledger))?;
            struct Arenas<'a>(&'a Attachments);
            impl Serialize for Arenas<'_> {
                fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
                    let mut seq = serializer.serialize_seq(Some(self.0.arenas.len()))?;
                    for (id, arena) in &self.0.arenas {
                        #[derive(Serialize)] struct Registered<'a> {
                            #[serde(flatten)] arena: &'a Arena, state: &'static str, backing_address: Option<Hex>,
                        }
                        let address = self.0.backings.get(id).copied();
                        seq.serialize_element(&Registered { arena, state: if address.is_some() { "live" } else { "pending" },
                            backing_address: address.map(Hex) })?;
                    }
                    seq.end()
                }
            }
            out.write_all(b",\"arenas\":")?; serde_json::to_writer(&mut out, &Arenas(&data))?;
            for (_, device) in ledger.current.keys() {
                if *device >= 0 && !devices[..device_count].contains(device) && device_count < devices.len() {
                    devices[device_count] = *device; device_count += 1;
                }
            }
            out.write_all(b",\"inputs\":")?; serde_json::to_writer(&mut out, &data.inputs)?;
            out.write_all(b",\"limits\":")?; serde_json::to_writer(&mut out, LIMITS)?;
        }
        out.write_all(b",\"memory\":")?;
        // Iterate only the tiny device-counter table without holding the
        // allocation lock across runtime APIs.
        let mut serializer = serde_json::Serializer::new(&mut out);
        let mut seq = serializer.serialize_seq(None)?;
        let mut current = -1;
        if !sticky { if let Some(q) = cuda_queries() {
            // SAFETY: runtime writes one valid stack output.
            unsafe { (q.get)(&mut current); }
        } }
        if current >= 0 && !devices[..device_count].contains(&current) && device_count < devices.len() {
            devices[device_count] = current; device_count += 1;
        }
        for &device in &devices[..device_count] { seq.serialize_element(&observe(device, sticky))?; }
        if device_count == 0 { seq.serialize_element(&observe(-1, sticky))?; }
        SerializeSeq::end(seq)?;
        out.write_all(b",\"capture\":{\"complete\":true,\"pre_rust_unwind\":true}}\n")?; out.flush()?;
        Ok(())
    })();
    let complete = result.is_ok();
    if let Err(error) = result { eprintln!("memory failure dump partial: serialization/I/O failure: {error}"); }
    if let Err(error) = std::fs::rename(&spool_path, &path) {
        eprintln!("memory failure dump unavailable: cannot publish spool {}: {error}", spool_path.display()); return None;
    }
    drop(service);
    let summary = summary(if census { "census" } else { "failure" }, sticky);
    if census { tracing::info!(target: "cuteafd::memory", path = %path.display(), "requested memory census\n{summary}"); }
    else { tracing::error!(target: "cuteafd::memory", path = %path.display(), api = native_text(raw.api), domain = native_text(raw.domain), code = raw.code, complete, "memory failure dump\n{summary}"); }
    Some(Arc::new(FailureReport { path, summary, complete, sequence: id, raw_code: raw.code, domain: native_text(raw.domain).into(), api: native_text(raw.api).into() }))
}
fn trigger<'a>(raw: &'a NativeFailure, context: &'a Context) -> Trigger<'a> {
    Trigger { api: native_text(raw.api), domain: native_text(raw.domain), code: raw.code,
        file: native_text(raw.file), line: raw.line, requested_bytes: (raw.bytes_known != 0).then_some(raw.bytes),
        handle: Hex(raw.handle), scope: context.scopes.last().copied().unwrap_or("other"),
        owner: context.owner, tensor: context.name, tensor_stem: context.tensor, format: context.format }
}
fn sequence() -> u64 { static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0); NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed) }
fn utc_timestamp() -> String {
    let mut time = 0;
    let mut calendar = std::mem::MaybeUninit::<libc::tm>::uninit();
    let mut text = [0u8; 32];
    // SAFETY: all outputs are valid stack storage; UTC conversion is reentrant.
    unsafe {
        libc::time(&mut time); libc::gmtime_r(&time, calendar.as_mut_ptr());
        libc::strftime(text.as_mut_ptr().cast(), text.len(), c"%Y%m%dT%H%M%SZ".as_ptr(), calendar.as_ptr());
    }
    String::from_utf8_lossy(&text[..text.iter().position(|&b| b == 0).unwrap_or(text.len())]).into_owned()
}

pub fn ready() {
    state().lock().unwrap_or_else(|e| e.into_inner()).ready = true;
    if std::env::var_os("CUTEAFD_STARTUP_MEMORY_PROBE").is_some_and(|value| value == "1") {
        let _ = capture(&NativeFailure { api: c"Ready census".as_ptr(), domain: c"requested census".as_ptr(),
            file: c"Rust Ready boundary".as_ptr(), line: 0, code: 0, bytes: 0, bytes_known: 0, handle: 0 }, false);
    }
}
/// Shared, bounded human-readable view for Ready and failure. Raw planned and
/// observed values remain separate; the signed residual is not a diagnosis.
pub fn compact_summary(stage: &str) -> String { summary(stage, false) }
/// Available byte inputs are numbers; absent optional inputs stay explicitly
/// unresolved rather than being silently represented as zero.

fn optional(value: Option<impl std::fmt::Display>) -> String { value.map(|value| value.to_string()).unwrap_or_else(|| "unknown".into()) }

fn summary(stage: &str, sticky: bool) -> String {
    use std::fmt::Write as _;
    let snapshot = super::snapshot();
    let mut out = format!("memory summary stage={stage}\n");
    for device in snapshot.devices() {
        let tracked = snapshot.total(Space::Device, device) + snapshot.total(Space::Managed, device);
        let observed = observe(device, sticky);
        let _ = writeln!(out, "device={device} free={} total={} tracked={tracked} residual={} status={}",
            optional(observed.free_bytes), optional(observed.total_bytes), optional(observed.total_bytes.zip(observed.free_bytes).map(|(t,f)| t as i128-f as i128-tracked as i128)), observed.status);
        // Every owned backing has a kind; the census counter stays zero by construction.
        let (mut kinds, mut untagged) = snapshot.by_kind(Space::Device, device);
        let (managed, managed_untagged) = snapshot.by_kind(Space::Managed, device);
        for (kind, bytes) in managed { *kinds.entry(kind).or_default() += bytes; }
        untagged += managed_untagged;
        let kinds = kinds.into_iter().map(|(kind, bytes)| (kind.label(), bytes)).collect::<std::collections::BTreeMap<_, _>>();
        let _ = writeln!(out, "device={device} kinds={kinds:?} untagged={untagged}");
    }
    let data = attachments().lock().unwrap_or_else(|e| e.into_inner());
    for arena in data.arenas.values().take(20) { let _ = writeln!(out, "arena={} kind={} device={} bytes={} padding={} tail={} views={}",
        arena.id, arena.kind.label(), arena.device, arena.bytes, arena.padding_bytes, optional(arena.tail_bytes), arena.views.len()); }
    if data.arenas.len() > 20 { let _ = writeln!(out, "+{} more arenas ({} bytes); full views in failure dump",
        data.arenas.len()-20, data.arenas.values().skip(20).map(|arena| arena.bytes).sum::<usize>()); }
    if let Some(inputs) = data.inputs.as_object() {
        for (index, (name, value)) in inputs.iter().enumerate() {
            let _ = write!(out, "{name} = {} ({}){}", value["value"], value["source"], if index % 2 == 1 { "\n" } else { "; " });
        }
        if inputs.len() % 2 != 0 { out.push('\n'); }
    }
    drop(data);
    let ledger = state().lock().unwrap_or_else(|e| e.into_inner());
    let _ = writeln!(out, "post_ready ready={} allocated_bytes={}", ledger.ready, ledger.post_ready_bytes);
    for reason in LIMITS { let _ = writeln!(out, "{reason}"); }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    fn fixture(api: &'static CStr, domain: &'static CStr, code: i32, bytes: Option<usize>, handle: usize) -> NativeFailure {
        NativeFailure { api: api.as_ptr(), domain: domain.as_ptr(), file: c"fake-native.cc".as_ptr(),
            line: 42, code, bytes: bytes.unwrap_or(0), bytes_known: i32::from(bytes.is_some()), handle }
    }
    fn setup() {
        let root = std::env::var_os("CARGO_TARGET_DIR").map(PathBuf::from).unwrap_or_else(std::env::temp_dir)
            .join(format!("memory-failure-tests-{}", std::process::id()));
        initialize(&root, "fixture-host", "rank3").unwrap();
    }
    #[test]
    fn tag_callback_survives_module_retirement_and_attributes_only_its_source() -> anyhow::Result<()> {
        const CHILD: &str = "CUTEAFD_TEST_TAG_MODULE_CHILD";
        if std::env::var_os(CHILD).is_none() {
            let output = std::process::Command::new(std::env::current_exe()?)
                .args(["--exact", "memory_ledger::fault::tests::tag_callback_survives_module_retirement_and_attributes_only_its_source", "--nocapture"])
                .env(CHILD, "1").output()?;
            assert!(output.status.success(), "{:?}\n{}\n{}", output.status,
                String::from_utf8_lossy(&output.stdout), String::from_utf8_lossy(&output.stderr));
            assert!(String::from_utf8_lossy(&output.stdout).contains("1 passed; 0 failed"));
            return Ok(());
        }
        let _guard = super::super::test_guard();
        setup(); clear_failure();
        let first = crate::native_library_lifetime_fixture::Fixture::build_for_tests()?;
        let library = first.load()?;
        drop(library);
        assert!(!first.resident()?, "diagnostics must not permanently pin the first DSO");
        trigger_record(&fixture(c"retired-module", c"cuda", 2, Some(1), 0), sequence(), None, "fixture");
        let second = crate::native_library_lifetime_fixture::Fixture::build_for_tests()?;
        let library = second.load()?;
        // SAFETY: private fixture exports synchronously copy the tag and emit a
        // POD failure whose __FILE__ literal lives in this module.
        unsafe {
            let tag = library.lib.get::<unsafe extern "C" fn(u32, *const c_char) -> i32>(b"fixture_attempt_tag")?;
            assert_eq!(tag(MemoryKind::Kv.as_raw(), c"test/second-library".as_ptr()), 1);
            library.lib.get::<unsafe extern "C" fn()>(b"fixture_emit_memory_failure")?();
        }
        let report = take_failure().expect("second DSO failure is captured");
        // A Rust-origin failure cannot inherit even the most recent live DSO's tag.
        trigger_record(&fixture(c"unknown-module", c"cuda", 2, Some(1), 0), sequence(), None, "fixture");
        let text = std::fs::read_to_string(&TRIGGERS.get().unwrap().1)?;
        let rows: Vec<serde_json::Value> = text.lines().map(serde_json::from_str).collect::<Result<_, _>>()?;
        let native = rows.iter().find(|r| r["api"] == "fixture-allocator").unwrap();
        assert_eq!(native["attempted_owner"], "test/second-library");
        assert_eq!(native["attempted_kind"], "kv");
        let unknown = rows.iter().find(|r| r["api"] == "unknown-module").unwrap();
        assert!(unknown["attempted_owner"].is_null());
        assert!(unknown["attempted_kind"].is_null());
        drop(library);
        assert!(!second.resident()?, "diagnostics must not permanently pin the second DSO");
        let _ = std::fs::remove_file(&report.path);
        Ok(())
    }

    #[test]
    fn tag_callback_holds_module_across_concurrent_last_owner_drop() -> anyhow::Result<()> {
        use std::sync::atomic::{AtomicI32, Ordering};
        let _guard = super::super::test_guard();
        let fixture = crate::native_library_lifetime_fixture::Fixture::build_for_tests()?;
        for _ in 0..16 {
            // SAFETY: this test just compiled the trusted CPU-only ABI fixture.
            let library = Arc::new(unsafe { libloading::Library::new(fixture.module_path())? });
            install(&library);
            let entered = AtomicI32::new(0);
            let release = AtomicI32::new(0);
            // SAFETY: gate storage outlives the joined thread and the entire DSO call.
            let file = unsafe {
                library.get::<unsafe extern "C" fn(*const AtomicI32, *const AtomicI32)>(b"fixture_tag_call_gate")?(&entered, &release);
                library.get::<unsafe extern "C" fn() -> *const c_char>(b"fixture_failure_file")?() as usize
            };
            let weak = Arc::downgrade(&library);
            std::thread::scope(|scope| {
                let worker = scope.spawn(|| {
                    let mut bytes = [0u8; 512];
                    assert_eq!(attempted_tag(file as *const c_char, &mut bytes), (None, None, false));
                });
                let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
                while entered.load(Ordering::Acquire) == 0 && std::time::Instant::now() < deadline {
                    std::thread::sleep(std::time::Duration::from_millis(1));
                }
                let called = entered.load(Ordering::Acquire) != 0;
                drop(library);
                let retained = weak.strong_count();
                release.store(1, Ordering::Release);
                worker.join().unwrap();
                assert!(called, "last-tag call did not enter before deadline");
                assert_eq!(retained, 1, "the in-flight call alone retains the module");
            });
            assert!(weak.upgrade().is_none());
            assert!(!fixture.resident()?, "module must unload after last-tag returns");
        }
        Ok(())
    }

    #[test]
    fn tag_callback_returns_unknown_without_waiting_for_registry_writer() {
        let _guard = super::super::test_guard();
        let _writer = TAG_MODULES.write().unwrap();
        let mut bytes = [0u8; 512];
        assert_eq!(attempted_tag(c"unknown".as_ptr(), &mut bytes), (None, None, false));
    }

    #[test]
    fn native_failures_preserve_live_backings_raw_status_and_timestamp() {
        let _guard = super::super::test_guard();
        setup();
        let _scope = super::super::scope("workspace/fault-fixture");
        let address = 0x7800_0000_0000;
        event_callback(0, address, 4096, 0, c"fake-native.cc".as_ptr(), 1,
            MemoryKind::Transport.as_raw(), c"test/native-event".as_ptr());
        assert_eq!(super::super::tag_of(address), Some(super::super::Tag::transport("test/native-event")));
        callback(&fixture(c"cudaMalloc", c"cuda", 2, Some(8192), 0));
        event_callback(0, address, 0, 1, std::ptr::null(), 0, 0, std::ptr::null());
        let report = take_failure().unwrap();
        let data: serde_json::Value = serde_json::from_slice(&std::fs::read(&report.path).unwrap()).unwrap();
        assert_eq!(data["trigger"]["code"], 2);
        assert_eq!(data["trigger"]["requested_bytes"], 8192);
        assert!(data["live_allocations"].as_array().unwrap().iter().any(|row| row["address"] == "0x780000000000" && row["bytes"] == 4096));
        assert!(data["capture"]["complete"].as_bool().unwrap());
        let name = report.path.file_name().unwrap().to_str().unwrap();
        let timestamp = name.strip_prefix("memory-failure-").unwrap();
        assert_eq!(&timestamp[8..9], "T"); assert_eq!(&timestamp[15..16], "Z");
        assert!(name.contains("fixture-host-rank3"));
        assert!(data["limits"].as_array().unwrap().iter().any(|v| v.as_str().unwrap().contains("unavailable:")));
        callback(&fixture(c"cublasCreate", c"cublas", 3, None, 0));
        let error = super::super::capture_failure(anyhow::anyhow!("ABI status -3")).context("outer wrapper");
        let saved = error.chain().find_map(|cause| cause.downcast_ref::<super::super::FailureSnapshot>()).unwrap();
        let report = saved.report.as_ref().unwrap();
        assert_eq!(report.raw_code, 3); assert_eq!(report.domain, "cublas");
        let data: serde_json::Value = serde_json::from_slice(&std::fs::read(&report.path).unwrap()).unwrap();
        assert!(data["trigger"]["requested_bytes"].is_null());
        assert!(format!("{error:#}").contains("ABI status -3"));
        std::fs::remove_file(&report.path).unwrap();
    }
    #[test]
    fn ready_summary_kinds_include_managed_and_device_backings_once() {
        let _guard = super::super::test_guard();
        let address = 0x7900_0000_0000;
        super::super::record_tagged(Space::Device, 91, address, 1024, super::super::Tag::weights("test/summary"));
        super::super::record_tagged(Space::Managed, 91, address + 0x1000, 2048, super::super::Tag::weights("test/summary"));
        super::super::record_tagged(Space::Managed, 91, address + 0x2000, 512, super::super::Tag::kv("test/summary"));
        let text = compact_summary("ready");
        super::super::record_released(address);
        super::super::record_released(address + 0x1000);
        super::super::record_released(address + 0x2000);
        assert!(text.lines().any(|line| line.starts_with("device=91 ") && line.contains("tracked=3584")), "{text}");
        assert!(text.lines().any(|line| line == "device=91 kinds={\"kv\": 512, \"weights\": 3072} untagged=0"), "{text}");
    }

    #[test]
    fn ready_summary_with_ten_arenas_is_compact_and_registrations_retire() {
        let _guard = super::super::test_guard();
        let arenas: Vec<_> = (0..10).map(|i| register_arena(Arena { id: format!("arena-{i}"),
            kind: MemoryKind::Weights, device: 0, bytes: 512, padding_bytes: 0, tail_bytes: None,
            views: vec![ArenaView { name: "weight".into(), offset: 0, bytes: 512 }] })).collect();
        let text = compact_summary("ready");
        assert!(text.lines().count() < 60, "{text}");
        assert_eq!(text.lines().filter(|line| line.starts_with("arena=")).count(), 10);
        drop(arenas);
        assert!(!compact_summary("failure").contains("arena=arena-"));
    }
    #[test]
    fn borrowed_trigger_strings_first_token_and_pending_arena_are_retained() {
        let _guard = super::super::test_guard(); setup(); clear_failure();
        let pending = register_arena(Arena { id: "pending-fixture".into(), kind: MemoryKind::Weights,
            device: 1, bytes: 64, padding_bytes: 0, tail_bytes: None, views: vec![] });
        let live = register_arena(Arena { id: "live-fixture".into(), kind: MemoryKind::Weights,
            device: 1, bytes: 64, padding_bytes: 0, tail_bytes: Some(0), views: vec![] });
        live.bind(0x1234);
        {
            let owner = String::from("temporary-owner");
            let _owner = super::super::owner(&owner);
            let api = std::ffi::CString::new("temporary-api").unwrap();
            let domain = std::ffi::CString::new("cublas").unwrap();
            let file = std::ffi::CString::new("temporary-file").unwrap();
            callback(&NativeFailure { api: api.as_ptr(), domain: domain.as_ptr(), file: file.as_ptr(),
                code: 3, line: 71, bytes: 42, bytes_known: 1, handle: 0 });
        }
        callback(&fixture(c"later", c"cuda", 2, Some(99), 0));
        let first = take_failure().unwrap();
        assert_eq!(first.api, "temporary-api");
        let value: serde_json::Value = serde_json::from_slice(&std::fs::read(&first.path).unwrap()).unwrap();
        assert_eq!(value["trigger"]["file"], "temporary-file");
        assert_eq!(value["trigger"]["owner"], "temporary-owner");
        assert!(value["arenas"].as_array().unwrap().iter().any(|a| a["id"] == "pending-fixture" && a["state"] == "pending"));
        assert!(value["arenas"].as_array().unwrap().iter().any(|a| a["id"] == "live-fixture" && a["backing_address"] == "0x1234"));
        let (_, path) = TRIGGERS.get().unwrap();
        let triggers = std::fs::read_to_string(path).unwrap();
        assert!(triggers.lines().filter_map(|line| serde_json::from_str::<serde_json::Value>(line).ok())
            .any(|v| v["linked_to"] == first.sequence && v["api"] == "later"));
        let _ = std::fs::remove_file(first.path.clone());
        drop((pending, live));
        assert!(take_failure().is_none());
    }

    #[test]
    fn nonmemory_status_does_not_spend_spools_and_boundary_retires_tolerated_failure() {
        let _guard = super::super::test_guard(); setup(); clear_failure();
        let count = SERVICE.get().unwrap().lock().unwrap().spools.len();
        let error = super::super::capture_failure(anyhow::Error::new(crate::NativeStatusError {
            context: "probe".into(), status: crate::CUTEAFD_STATUS_RDMA_UNAVAILABLE, detail: "unsupported".into() }));
        let snapshot = error.chain().find_map(|e| e.downcast_ref::<super::super::FailureSnapshot>()).unwrap();
        assert!(snapshot.report.is_none());
        assert_eq!(SERVICE.get().unwrap().lock().unwrap().spools.len(), count);
        {
            let _boundary = boundary();
            callback(&fixture(c"fallback", c"cuda", 2, Some(1), 0));
        }
        assert!(take_failure().is_none());
    }

    #[test]
    fn logical_budget_refusal_retains_a_detailed_report() {
        let _guard = super::super::test_guard(); setup(); clear_failure();
        let error = super::super::capture_failure(anyhow::Error::new(
            cuteafd_core::serving_capacity::CapacityError::GpuBudgetExceeded {
                device: 1, required: 2, budget: 1, shortfall: 1 }));
        let snapshot = error.chain().find_map(|e| e.downcast_ref::<super::super::FailureSnapshot>()).unwrap();
        let report = snapshot.report.as_ref().unwrap();
        assert_eq!(report.api, "Rust error boundary");
        assert!(error.chain().any(|e| e.is::<cuteafd_core::serving_capacity::CapacityError>()));
        let _ = std::fs::remove_file(&report.path);
    }

    #[test]
    fn arena_summary_is_bounded_with_realistic_inputs() {
        let _guard = super::super::test_guard();
        let old = { let mut data = attachments().lock().unwrap(); std::mem::take(&mut data.inputs) };
        set_inputs(serde_json::Value::Object((0..35).map(|i| (format!("input-{i}"), serde_json::json!(i))).collect()));
        let mut arenas: Vec<_> = (0..25).map(|i| register_arena(Arena { id: format!("bounded-{i}"),
            kind: MemoryKind::Weights, device: 1, bytes: 64, padding_bytes: 0, tail_bytes: None, views: vec![] })).collect();
        let text = compact_summary("ready");
        assert_eq!(text.lines().filter(|line| line.starts_with("arena=")).count(), 20);
        assert!(text.contains("+5 more arenas (320 bytes)"));
        arenas.truncate(10);
        let text = compact_summary("ready");
        assert!(text.lines().count() < 60, "{text}");
        drop(arenas);
        attachments().lock().unwrap().inputs = old;
    }

    #[test]
    fn guarded_driver_queries_restore_null_and_do_not_touch_inactive_devices() {
        thread_local! { static CALLS: RefCell<Vec<(char, usize)>> = const { RefCell::new(Vec::new()) }; }
        unsafe extern "C" fn get(_: *mut i32) -> i32 { 0 }
        unsafe extern "C" fn current(out: *mut *mut std::ffi::c_void) -> i32 {
            // SAFETY: fixture receives a valid stack output.
            unsafe { *out = std::ptr::null_mut(); } 0
        }
        unsafe extern "C" fn state(device: i32, _: *mut u32, active: *mut i32) -> i32 {
            CALLS.with(|c| c.borrow_mut().push(('s', device as usize)));
            // SAFETY: fixture receives a valid stack output.
            unsafe { *active = i32::from(device == 1); } 0
        }
        unsafe extern "C" fn retain(out: *mut *mut std::ffi::c_void, device: i32) -> i32 {
            CALLS.with(|c| c.borrow_mut().push(('r', device as usize)));
            // SAFETY: fixture receives a valid stack output, pointer is never dereferenced.
            unsafe { *out = 17usize as *mut _; } 0
        }
        unsafe extern "C" fn select(ptr: *mut std::ffi::c_void) -> i32 {
            CALLS.with(|c| c.borrow_mut().push(('c', ptr as usize))); 0
        }
        unsafe extern "C" fn release(device: i32) -> i32 {
            CALLS.with(|c| c.borrow_mut().push(('d', device as usize))); 0
        }
        unsafe extern "C" fn failed(_: *mut usize, _: *mut usize) -> i32 { 719 }
        let q = CudaQueries { get, current, select, state, retain, release, info: failed, pool: None, attribute: None };
        let observation = observe_with(&q, 0);
        assert!(observation.free_bytes.is_none());
        assert_eq!(CALLS.with(|c| std::mem::take(&mut *c.borrow_mut())), vec![('s', 0)]);
        let observation = observe_with(&q, 1);
        assert_eq!(observation.raw_status, Some(719));
        assert_eq!(CALLS.with(|c| std::mem::take(&mut *c.borrow_mut())), vec![('s', 1), ('r', 1), ('c', 17), ('c', 0), ('d', 1)]);
        unsafe extern "C" fn info(free: *mut usize, total: *mut usize) -> i32 {
            // SAFETY: fixture outputs are valid stack addresses.
            unsafe { *free = 2048; *total = 4096; } 0
        }
        unsafe extern "C" fn nonnull(out: *mut *mut std::ffi::c_void) -> i32 {
            // SAFETY: fixture pointer is an opaque context handle, never dereferenced.
            unsafe { *out = 33usize as *mut _; } 0
        }
        let q = CudaQueries { current: nonnull, info, ..q };
        let observation = observe_with(&q, 1);
        assert_eq!(observation.free_bytes, Some(2048));
        assert_eq!(observation.total_bytes, Some(4096));
        assert_eq!(CALLS.with(|c| std::mem::take(&mut *c.borrow_mut())), vec![('s', 1), ('r', 1), ('c', 17), ('c', 33), ('d', 1)]);
    }

    #[test]
    fn busy_and_exhausted_dumps_keep_reserved_triggers() {
        let _guard = super::super::test_guard(); setup();
        let service = SERVICE.get().unwrap();
        let mut locked = service.lock().unwrap();
        let raw = fixture(c"busy-fixture", c"cuda", 2, Some(123), 0);
        assert!(capture(&raw, true).is_none());
        let spools = std::mem::take(&mut locked.spools);
        drop(locked);
        let raw = fixture(c"exhausted-fixture", c"cuda", 2, Some(456), 0);
        assert!(capture(&raw, true).is_none());
        service.lock().unwrap().spools = spools;
        let (_, path) = TRIGGERS.get().unwrap();
        let text = std::fs::read_to_string(path).unwrap();
        assert!(text.contains("detailed capture busy"));
        assert!(text.contains("detailed spools exhausted"));
    }

    #[test]
    fn stale_spools_are_swept_but_live_locks_and_trigger_evidence_survive() {
        let _guard = super::super::test_guard();
        let root = std::env::var_os("CARGO_TARGET_DIR").map(PathBuf::from).unwrap_or_else(std::env::temp_dir)
            .join(format!("memory-spool-sweep-tests-{}", std::process::id()));
        std::fs::create_dir_all(&root).unwrap();
        let mut child = std::process::Command::new("true").spawn().unwrap();
        let dead = child.id();
        child.wait().unwrap();
        let stale = root.join(format!(".memory-spool-{dead}-0"));
        let empty = root.join(format!("memory-triggers-fixture-rank3-{dead}.jsonl"));
        let evidence = root.join(format!("memory-triggers-fixture-rank4-{dead}.jsonl"));
        let reused = root.join(format!(".memory-spool-{}-1", std::process::id()));
        let live = root.join(format!(".memory-spool-{}-2", std::process::id()));
        for path in [&stale, &empty, &reused, &live] { std::fs::write(path, b"").unwrap(); }
        std::fs::write(&evidence, b"{\"sequence\":1}\n").unwrap();
        let locked = File::open(&live).unwrap(); lock_spool(&locked).unwrap();
        sweep_unused_spools(&root).unwrap();
        for path in [&stale, &empty, &reused] { assert!(!path.exists(), "{}", path.display()); }
        assert!(live.exists()); assert!(evidence.exists());
        drop(locked);
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn already_retained_error_clears_an_untaken_native_token() {
        let _guard = super::super::test_guard(); setup(); clear_failure();
        let error = super::super::capture_failure(anyhow::anyhow!("retained fixture"));
        callback(&fixture(c"untaken-fixture", c"cuda", 2, Some(123), 0));
        let path = PENDING.with(|pending| pending.borrow().as_ref().unwrap().path.clone());
        let error = super::super::capture_failure(error);
        assert!(error.chain().any(|cause| cause.is::<super::super::FailureSnapshot>()));
        assert!(take_failure().is_none());
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn unused_spools_and_empty_trigger_file_are_removed_at_shutdown() {
        const CHILD: &str = "CUTEAFD_TEST_SPOOL_CHILD";
        if std::env::var_os(CHILD).is_some() {
            setup();
            let directory = SERVICE.get().unwrap().lock().unwrap().directory.clone();
            assert_eq!(std::fs::read_dir(&directory).unwrap().count(), 9);
            shutdown();
            assert_eq!(std::fs::read_dir(&directory).unwrap().count(), 0);
            return;
        }
        let result = std::process::Command::new(std::env::current_exe().unwrap())
            .args(["--exact", "memory_ledger::fault::tests::unused_spools_and_empty_trigger_file_are_removed_at_shutdown"])
            .env(CHILD, "1").output().unwrap();
        assert!(result.status.success(), "{}", String::from_utf8_lossy(&result.stdout));
    }

    #[test]
    fn unsupported_api_is_unknown_not_zero() {
        let value = observe(0, true);
        assert!(value.free_bytes.is_none() && value.total_bytes.is_none());
        assert!(value.status.starts_with("unavailable:"));
    }
}
