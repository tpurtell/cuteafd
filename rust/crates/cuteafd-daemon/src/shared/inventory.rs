//! Phase-labelled runtime samples for coordinator admission and loaded workers.
//! The runtime side of `cuteafd_loader::placement::inventory`: coordinator
//! admission uses pre-weight samples ([`RuntimeInventory::baselines`]); loaded
//! workers report their own architecture and retained memory after loading.
use anyhow::{Context, Result};
use cuteafd_ffi::NativeLibrary;
use cuteafd_loader::placement::{Baseline, RuntimeSample};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum InventoryPhase {
    PreWeight,
    PostLoad,
}

impl InventoryPhase {
    fn description(self) -> &'static str {
        match self {
            Self::PreWeight => "after context and program modules, before weights",
            Self::PostLoad => "after retained weights and execution workspace",
        }
    }

    fn module_bytes(self, before: usize, after: usize) -> u64 {
        match self {
            Self::PreWeight => before.saturating_sub(after) as u64,
            Self::PostLoad => 0,
        }
    }
}

#[derive(Debug, Clone)]
pub(crate) struct RuntimeInventory {
    pub gpus: Vec<RuntimeSample>,
}

impl RuntimeInventory {
    /// Samples `devices` (lead first), restoring the calling thread's device.
    /// Pre-weight callbacks load program modules between samples; post-load
    /// callbacks only synchronize/sample retained owners and do not attribute
    /// free-memory changes to module loading. `context_bytes` is CUDA's used
    /// bytes minus this process's ledger-tracked device allocations.
    pub fn measure(library: &NativeLibrary, devices: &[i32], phase: InventoryPhase,
        mut load_modules: impl FnMut(i32) -> Result<()>) -> Result<Self> {
        let current = library.cuda_get_device()?;
        let mut gpus = Vec::with_capacity(devices.len());
        for &device in devices {
            library.cuda_set_device(device)?;
            let sample = (|| -> Result<RuntimeSample> {
                let (before, _) = library.cuda_memory_info()?;
                let (free, total) = after_modules(device, &mut |id| {
                    load_modules(id)?;
                    if phase == InventoryPhase::PreWeight { preload_native_modules(library)?; }
                    Ok(())
                }, || library.cuda_memory_info())?;
                let tracked = cuteafd_ffi::memory_ledger::snapshot().total(cuteafd_ffi::memory_ledger::Space::Device, device);
                let info = library.cuda_device_info(device)?;
                let text = |chars: &[std::ffi::c_char]| {
                    // SAFETY: the native info call NUL-terminates every fixed-width string field.
                    unsafe { std::ffi::CStr::from_ptr(chars.as_ptr()) }.to_string_lossy().into_owned()
                };
                Ok(RuntimeSample {
                    device,
                    total_bytes: total as u64,
                    free_bytes: free as u64,
                    context_bytes: (total.saturating_sub(free) as u64).saturating_sub(tracked as u64),
                    module_bytes: phase.module_bytes(before, free),
                    sms: library.sm_count().unwrap_or(0) as u32,
                    arch: format!("sm_{}{}", info.compute_capability_major, info.compute_capability_minor),
                    driver: text(&info.driver_version),
                })
            })();
            library.cuda_set_device(current)?;
            gpus.push(sample.with_context(|| format!("sampling GPU {device} for placement"))?);
        }
        for gpu in &gpus {
            tracing::info!(device = gpu.device, total_bytes = gpu.total_bytes, free_bytes = gpu.free_bytes,
                context_bytes = gpu.context_bytes, module_bytes = gpu.module_bytes, sms = gpu.sms, arch = %gpu.arch,
                driver = %gpu.driver, phase = ?phase, stage = phase.description(), "runtime inventory");
        }
        Ok(Self { gpus })
    }

    /// Logs one recordable JSON line (`placement inventory`) with each GPU's
    /// inventory sample and the admission's measured free bytes, as the
    /// equality tests' hardware fixtures are recorded
    /// (`shared/placement_fixtures/*.json`).
    pub fn record(&self, family: &str, admission_free: &[u64]) {
        let gpus: Vec<_> = self.gpus.iter().zip(admission_free.iter().copied().chain(std::iter::repeat(0)))
            .map(|(g, free)| serde_json::json!({"sample": g, "admission_free_bytes": free})).collect();
        tracing::info!(target: "cuteafd::placement", record = %serde_json::json!({"family": family, "gpus": gpus}),
            "placement inventory");
    }

    /// The solver's per-GPU `(total, Measured)` pairs, less `loaded` bytes the
    /// family allocated between this sample and admission.
    pub fn baselines(&self, loaded: &[u64]) -> Vec<(u64, Baseline)> {
        self.gpus.iter().enumerate().map(|(i, g)| (g.total_bytes,
            Baseline::Measured { free_bytes: g.free_bytes.saturating_sub(loaded.get(i).copied().unwrap_or(0)) }))
            .collect()
    }
}

/// Full runtime-registered inventory, not only selected AOT programs. Called
/// inside each family's measured module window, before admission samples.
pub(crate) fn preload_native_modules(library: &NativeLibrary) -> Result<()> {
    library.tp2_native_initialize()
}

fn after_modules<T>(device: i32, load: &mut impl FnMut(i32) -> Result<()>,
    measure: impl FnOnce() -> Result<T>) -> Result<T> {
    load(device)?;
    measure()
}

#[cfg(test)]
mod phase_tests {
    use super::InventoryPhase;

    #[test]
    fn selected_modules_load_before_admission_sample_and_failure_stops_sampling() {
        use std::cell::RefCell;
        let events = RefCell::new(Vec::new());
        let selected = ["exlr8_dense_native", "glmf_stage_kda_projected_m64"];
        let value = super::after_modules(7, &mut |device| {
            assert_eq!(device, 7);
            events.borrow_mut().extend(selected);
            Ok(())
        }, || {
            assert_eq!(*events.borrow(), selected);
            events.borrow_mut().push("sample");
            Ok(42)
        }).unwrap();
        assert_eq!(value, 42);
        assert_eq!(*events.borrow(), [selected[0], selected[1], "sample"]);
        let failed: anyhow::Result<()> = super::after_modules(7, &mut |_| anyhow::bail!("module rejected"), || {
            panic!("admission must not sample after a failed module load")
        });
        assert!(failed.is_err());
    }

    #[test]
    fn unknown_package_live_reservation_uses_same_program_estimate_as_plan() {
        use cuteafd_loader::placement::{inventory::loaded_code_allowance, ArchContext, ProgramSet};
        let manifest = serde_json::json!({"programs":[{"name":"qwen4_decode"}]});
        let arch = ArchContext::for_device("sm_120", 96 << 30);
        let programs = ProgramSet::from_manifest(&manifest, &["qwen4"]);
        let allowance = loaded_code_allowance("qwen4", "unmeasured", false, 0, arch, Some(&programs));
        let planned = cuteafd_loader::plan::layout::runtime_item(Some(&manifest), &["qwen4".into()],
            "unmeasured", false, 0, 96 << 30);
        let pending = allowance.pending(arch.context_bytes, arch.context_bytes);
        assert!(pending > 0);
        assert_eq!(pending + arch.context_bytes, planned.bytes);
        assert_eq!(allowance.basis, cuteafd_core::memory_layout::Basis::Estimated);
    }

    #[test]
    fn only_pre_weight_samples_attribute_module_growth() {
        assert_eq!(InventoryPhase::PreWeight.module_bytes(100, 60), 40);
        assert_eq!(InventoryPhase::PreWeight.module_bytes(60, 100), 0);
        assert_eq!(InventoryPhase::PostLoad.module_bytes(100, 60), 0);
        assert!(InventoryPhase::PreWeight.description().contains("before weights"));
        assert!(!InventoryPhase::PostLoad.description().contains("before weights"));
    }
}

fn host_high_water_bytes(status: &str) -> Result<u64> {
    let mut fields = status.lines().find_map(|line| line.strip_prefix("VmHWM:"))
        .context("process status has no VmHWM")?.split_whitespace();
    let kib = fields.next().context("VmHWM has no value")?.parse::<u64>()?;
    anyhow::ensure!(fields.next() == Some("kB") && fields.next().is_none(), "invalid VmHWM units");
    kib.checked_mul(1024).context("VmHWM bytes overflow")
}

#[cfg(test)]
mod startup_probe_tests {
    #[test]
    fn layer_phase_sampling_is_bounded_to_first_three_and_last() {
        for layers in [0, 1, 2, 3, 4, 61, usize::MAX] {
            let candidates = [0, 1, 2, 3, layers.saturating_sub(1), layers];
            for layer in candidates {
                assert_eq!(super::sampled_startup_layer(layer, layers),
                    layer < layers && (layer < 3 || layer + 1 == layers));
            }
        }
        assert_eq!((0..61).filter(|&layer| super::sampled_startup_layer(layer, 61)).collect::<Vec<_>>(),
            [0, 1, 2, 60]);
    }

    #[test]
    fn phase_deltas_preserve_signed_unclassified_growth() -> anyhow::Result<()> {
        use super::StartupPhaseSample;
        let before = StartupPhaseSample { total: 1000, free: 900, tracked: 50 };
        let allocated = StartupPhaseSample { total: 1000, free: 500, tracked: 350 };
        assert_eq!(allocated.delta(before)?, (400, 300, 100));
        assert_eq!(before.delta(allocated)?, (-400, -300, -100));
        let freed = StartupPhaseSample { total: 1000, free: 950, tracked: 0 };
        assert_eq!(freed.delta(before)?, (-50, -50, 0));
        assert!(StartupPhaseSample { total: 999, ..before }.delta(before).is_err());
        assert!(StartupPhaseSample { free: 1001, ..before }.delta(before).is_err());
        Ok(())
    }

    #[test]
    fn host_high_water_uses_kernel_kib_units_and_rejects_bad_status() {
        assert_eq!(super::host_high_water_bytes("Name:\tcuteafd\nVmHWM:\t1234 kB\nVmRSS:\t1 kB\n").unwrap(), 1_263_616);
        for status in ["", "VmHWM: 1 B", "VmHWM: -1 kB", "VmHWM: 18446744073709551615 kB"] {
            assert!(super::host_high_water_bytes(status).is_err());
        }
    }
}

mod startup_memory;

pub(crate) fn startup_probe_enabled() -> bool {
    std::env::var("CUTEAFD_STARTUP_MEMORY_PROBE").as_deref() == Ok("1")
}

/// Evidence owned by the family, not a second observer or inferred lifetime.
#[derive(Default)]
pub(crate) struct StartupEvidence<'a> {
    pub audit: Option<&'a cuteafd_ffi::memory_ledger::AllocationAudit>,
    pub history: Option<&'a cuteafd_ffi::memory_ledger::AllocationHistory>,
    pub census: Option<&'a crate::shared::memory::receipts::Census>,
}

/// Opt-in startup attribution only; never sample on the request hot path.
pub(crate) fn startup_probe(library: &NativeLibrary, devices: &[i32], stage: &str,
    evidence: StartupEvidence<'_>) -> Result<()> {
    use cuteafd_ffi::memory_ledger::{self, Space};
    use serde_json::json;
    if !startup_probe_enabled() { return Ok(()); }
    anyhow::ensure!(evidence.audit.is_none() || evidence.history.is_none(), "two native histories in startup probe");
    let (position, native) = if let Some(audit) = evidence.audit {
        let (position, records) = audit.checkpoint_allocations();
        (Some(position), records)
    } else if let Some(history) = evidence.history {
        (Some(history.events.len()), history.final_allocations.clone())
    } else { (None, memory_ledger::allocations()) };
    let host = startup_memory::host(&native, stage).unwrap_or_else(|error|
        json!({"complete": false, "findings": [format!("host sampling unavailable: {error:#}")]}));
    let census = evidence.census.map(|census| startup_memory::census(census, &native, devices)).transpose()?;
    let current = library.cuda_get_device()?;
    let result = (|| -> Result<()> {
        for (rank, &device) in devices.iter().enumerate() {
            library.cuda_set_device(device)?;
            let (free, total) = library.cuda_physical_memory_info()?;
            let tracked = native.iter().filter(|r| r.key.device == device
                && matches!(r.key.space, Space::Device | Space::Managed)).map(|r| r.bytes as u64).sum::<u64>();
            let used = total.checked_sub(free).context("CUDA free exceeds total")? as u64;
            tracing::info!(target: "cuteafd::placement", probe = %json!({
                "schema": 2, "stage": stage, "device": device, "logical_rank": rank,
                "free": free, "total": total,
                "cuda_memory_scope": "raw cudaMemGetInfo; GB10 under-reports reclaimable page cache",
                "allocatable_free_bytes": library.cuda_allocatable_memory_info()?.0,
                "tracked": tracked, "untracked": used as i64 - tracked as i64,
                "cuda_tolerance_bytes": 16 << 20,
                "runtime_delta_scope": "signed unclassified observation; not a receipt or automatic allowance",
                "host": host, "native_event_position": position,
                "native_record_count": native.len(), "census": census,
                "history_closed": evidence.history.is_some(),
                "ready_authorized_by_probe": false
            }), "startup memory probe");
        }
        for (index, records) in native.chunks(64).enumerate() {
            tracing::info!(target: "cuteafd::placement", probe = %json!({"schema": 2,
                "stage": stage, "kind": "native-live", "event_position": position, "offset": index * 64,
                "records": records.iter().map(startup_memory::record).collect::<Vec<_>>() }), "startup memory probe");
        }
        if let Some(census) = evidence.census {
            for (index, records) in census.receipts.chunks(64).enumerate() {
                tracing::info!(target: "cuteafd::placement", probe = %json!({"schema": 2,
                    "stage": stage, "kind": "retained-owners", "offset": index * 64,
                    "records": records.iter().map(|r| json!({"address": r.address, "allocation": r.allocation})).collect::<Vec<_>>() }), "startup memory probe");
            }
            for (index, aliases) in census.aliases.chunks(64).enumerate() {
                tracing::info!(target: "cuteafd::placement", probe = %json!({"schema": 2,
                    "stage": stage, "kind": "mapped-aliases", "offset": index * 64,
                    "records": aliases.iter().map(|a| json!({"owner": a.owner, "backing": a.backing,
                        "location": a.location, "address": a.address, "bytes": a.bytes})).collect::<Vec<_>>() }), "startup memory probe");
            }
        }
        if let Some(history) = evidence.history {
            // One bounded startup history, emitted in chunks rather than one
            // multi-megabyte tracing line. The subscription is already closed.
            for (index, records) in history.initial.chunks(64).enumerate() {
                tracing::info!(target: "cuteafd::placement", probe = %json!({"schema": 2,
                    "stage": stage, "kind": "native-initial", "offset": index * 64,
                    "records": records.iter().map(startup_memory::record).collect::<Vec<_>>() }), "startup memory probe");
            }
            for (index, events) in history.events.chunks(64).enumerate() {
                tracing::info!(target: "cuteafd::placement", probe = %json!({"schema": 2,
                    "stage": stage, "kind": "native-events", "offset": index * 64,
                    "events": events.iter().map(startup_memory::event).collect::<Vec<_>>() }), "startup memory probe");
            }
        }
        Ok(())
    })();
    library.cuda_set_device(current)?;
    result
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize)]
struct StartupPhaseSample {
    total: u64,
    free: u64,
    tracked: u64,
}

impl StartupPhaseSample {
    fn used(self) -> Result<i64> {
        i64::try_from(self.total.checked_sub(self.free).context("CUDA free exceeds total")?)
            .context("CUDA used bytes exceed signed diagnostic range")
    }

    fn delta(self, before: Self) -> Result<(i64, i64, i64)> {
        anyhow::ensure!(self.total == before.total, "startup CUDA capacity changed");
        let used = self.used()? - before.used()?;
        let tracked = i64::try_from(self.tracked)? - i64::try_from(before.tracked)?;
        Ok((used, tracked, used - tracked))
    }
}

fn sampled_startup_layer(layer: usize, layers: usize) -> bool {
    layer < layers && (layer < 3 || layer == layers - 1)
}

/// Bounded in-process startup observations, never admission owners or a hot-path probe.
/// Disabled callers do not synchronize, sample CUDA, or snapshot the native ledger.
pub(crate) struct StartupMemoryPhases<'a> {
    library: &'a NativeLibrary,
    devices: Vec<i32>,
    enabled: bool,
    previous: Option<(String, Vec<StartupPhaseSample>)>,
    baseline: Vec<StartupPhaseSample>,
    native: std::collections::BTreeMap<usize, cuteafd_ffi::memory_ledger::AllocationRecord>,
    stages: usize,
}

impl<'a> StartupMemoryPhases<'a> {
    pub(crate) fn new(library: &'a NativeLibrary, devices: &[i32], enabled: bool) -> Self {
        Self { library, devices: if enabled { devices.to_vec() } else { Vec::new() }, enabled,
            previous: None, baseline: Vec::new(), native: Default::default(), stages: 0 }
    }

    pub(crate) fn enabled(&self) -> bool { self.enabled }

    pub(crate) fn sample_layer(&mut self, layer: usize, layers: usize) -> Result<()> {
        if self.enabled && sampled_startup_layer(layer, layers) {
            self.sample(&format!("weights-core-layer-{layer}-upload-staging-recorded"))?;
        }
        Ok(())
    }

    pub(crate) fn sample(&mut self, stage: &str) -> Result<()> {
        use cuteafd_ffi::memory_ledger::{self, Space};
        use serde_json::json;
        if !self.enabled { return Ok(()); }
        anyhow::ensure!(self.stages < 64, "startup phase diagnostic exceeded 64 boundaries");
        let current = self.library.cuda_get_device()?;
        let result = (|| -> Result<_> {
            for &device in &self.devices {
                self.library.cuda_set_device(device)?;
                self.library.cuda_device_synchronize()?;
            }
            let native = memory_ledger::allocations().into_iter()
                .map(|record| (record.address, record)).collect::<std::collections::BTreeMap<_, _>>();
            let mut samples = Vec::with_capacity(self.devices.len());
            for (rank, &device) in self.devices.iter().enumerate() {
                self.library.cuda_set_device(device)?;
                let (free, total) = self.library.cuda_physical_memory_info()?;
                let tracked = native.values().filter(|r| r.key.device == device
                    && matches!(r.key.space, Space::Device | Space::Managed))
                    .try_fold(0u64, |sum, r| sum.checked_add(r.bytes as u64).context("startup tracked bytes overflow"))?;
                let sample = StartupPhaseSample { total: total as u64, free: free as u64, tracked };
                let (used_delta, tracked_delta, untracked_delta) = self.previous.as_ref()
                    .map(|(_, samples)| sample.delta(samples[rank])).transpose()?.unwrap_or_default();
                let baseline = self.baseline.get(rank).copied().unwrap_or(sample);
                let (_, _, untracked_since_context) = sample.delta(baseline)?;
                tracing::info!(target: "cuteafd::placement", probe = %json!({
                    "schema": 3, "kind": "startup-phase", "family": "kimi_k3", "stage": stage,
                    "index": self.stages, "device": device, "logical_rank": rank,
                    "previous_stage": self.previous.as_ref().map(|(stage, _)| stage),
                    "free": sample.free, "total": sample.total, "tracked": tracked,
                    "used": sample.used()?, "untracked": sample.used()? - i64::try_from(tracked)?,
                    "used_delta": used_delta, "tracked_delta": tracked_delta,
                    "untracked_delta": untracked_delta, "untracked_since_context": untracked_since_context,
                    "allocator": memory_ledger::current_cuda_memory_snapshot(),
                    "allocator_scope": "CUDA current-pool reserved/used telemetry, not PyTorch caching-allocator metrics; core records use direct cudaMalloc; no trim",
                    "cuda_memory_scope": "in-process raw cudaMemGetInfo; no external observer",
                    "runtime_delta_scope": "signed unclassified observation; not a receipt or automatic allowance",
                    "ready_authorized_by_probe": false
                }), "startup memory phase");
                samples.push(sample);
            }
            for (kind, records) in [
                ("phase-native-removed", self.native.iter().filter(|(address, record)| native.get(*address) != Some(*record))
                    .map(|(_, record)| record).collect::<Vec<_>>()),
                ("phase-native-added", native.iter().filter(|(address, record)| self.native.get(*address) != Some(*record))
                    .map(|(_, record)| record).collect::<Vec<_>>()),
            ] {
                for (chunk, records) in records.chunks(64).enumerate() {
                    tracing::info!(target: "cuteafd::placement", probe = %json!({"schema": 3,
                        "kind": kind, "family": "kimi_k3", "stage": stage, "index": self.stages,
                        "offset": chunk * 64, "records": records.iter().map(|r| startup_memory::record(r)).collect::<Vec<_>>()
                    }), "startup memory phase");
                }
            }
            Ok((samples, native))
        })();
        self.library.cuda_set_device(current)?;
        let (samples, native) = result?;
        if self.previous.is_none() { self.baseline = samples.clone(); }
        self.previous = Some((stage.to_owned(), samples));
        self.native = native;
        self.stages += 1;
        Ok(())
    }
}

/// Ready code still missing from the current sample, using the same calibrated
/// or manifest-derived allowance as offline planning.
pub(crate) fn pending_code(library: &NativeLibrary, device: i32, rank: usize, split: bool, program_family: &str,
    experts: &str, programs: &cuteafd_loader::placement::ProgramSet) -> Result<u64> {
    let current = library.cuda_get_device()?;
    library.cuda_set_device(device)?;
    let sample = (|| -> Result<_> { Ok((library.cuda_memory_info()?, library.cuda_device_info(device)?)) })();
    library.cuda_set_device(current)?;
    let ((free, total), info) = sample?;
    let tracked = cuteafd_ffi::memory_ledger::snapshot().total(cuteafd_ffi::memory_ledger::Space::Device, device);
    let untracked = (total.saturating_sub(free) as u64).saturating_sub(tracked as u64);
    let arch = format!("sm_{}{}", info.compute_capability_major, info.compute_capability_minor);
    let context = cuteafd_loader::placement::ArchContext::for_device(&arch, total as u64);
    let code = cuteafd_loader::placement::inventory::loaded_code_allowance(program_family, experts,
        split, rank as u8, context, Some(programs));
    let pending = code.pending(untracked, context.context_bytes);
    tracing::info!(device, family = program_family, experts, loaded_code_bytes = code.bytes, untracked_bytes = untracked,
        basis = ?code.basis, pending_bytes = pending, "loaded code reserved beside admission");
    Ok(pending)
}

#[test]
fn observer_capacity_reaches_glm_forecasts_and_v4_physical_inputs() -> Result<()> {
    // A child isolates the process-wide observer declaration from parallel tests.
    if std::env::var_os("CUTEAFD_TEST_OBSERVER_CHILD").is_none() {
        let root = tempfile::tempdir()?;
        let file = root.path().join("observer.json");
        std::fs::write(&file, r#"{"0":550}"#)?;
        let output = std::process::Command::new(std::env::current_exe()?)
            .args(["--exact", "shared::inventory::observer_capacity_reaches_glm_forecasts_and_v4_physical_inputs", "--nocapture"])
            .env("CUTEAFD_TEST_OBSERVER_CHILD", "1")
            .env("CUTEAFD_OBSERVER_CONTEXT_FILE", file).output()?;
        anyhow::ensure!(output.status.success(), "observer child failed: {} {}",
            String::from_utf8_lossy(&output.stdout), String::from_utf8_lossy(&output.stderr));
        return Ok(());
    }
    let fixture = cuteafd_ffi::native_library_lifetime_fixture::Fixture::build_for_tests()?;
    let library = fixture.load()?;
    fixture.configure_memory(&library, 0, 10_000)?;
    // Stand in for the external observer's resident context in raw CUDA counters.
    let mut observer = library.alloc_device_buffer(550, crate::shared::memory::TEST)?;
    assert_eq!(library.cuda_physical_memory_info()?, (9_450, 10_000));
    let (free, physical) = library.cuda_allocatable_memory_info()?;
    assert_eq!((free, physical), (9_450, 9_450));
    assert_eq!(library.cuda_memory_info()?, (9_450, 9_450));
    let budgets = cuteafd_loader::placement::families::glm5::admission::production_budgets(
        &[(physical as u64, physical as u64)], 0, (10_000, 10_000), 0, &[])?;
    assert_eq!(budgets[0].available()?, 9_450);
    let capped = cuteafd_loader::placement::families::glm5::admission::production_budgets(
        &[(physical as u64, 8_000)], 0, (10_000, 10_000), 0, &[])?;
    assert_eq!(capped[0].available()?, 8_000);
    // V4 reads this same pre-budget total separately from the logical sample.
    let v4_physical = library.cuda_allocatable_memory_info()?.1;
    assert_eq!(v4_physical, physical);
    library.free_device_buffer(&mut observer)?;
    Ok(())
}
