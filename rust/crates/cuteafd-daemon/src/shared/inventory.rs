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
                let (free, total) = after_modules(device, &mut load_modules, || library.cuda_memory_info())?;
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
    fn unknown_package_live_reservation_uses_same_positive_formula_as_plan() {
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
        assert_eq!(allowance.basis, cuteafd_core::memory_layout::Basis::Formula);
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

/// Opt-in startup attribution only; never sample on the request hot path.
pub(crate) fn startup_probe(library: &NativeLibrary, devices: &[i32], stage: &str) -> Result<()> {
    if std::env::var("CUTEAFD_STARTUP_MEMORY_PROBE").as_deref() != Ok("1") { return Ok(()); }
    let current = library.cuda_get_device()?;
    let result = (|| -> Result<()> {
        for &device in devices {
            library.cuda_set_device(device)?;
            let (free, total) = library.cuda_physical_memory_info()?;
            let tracked = cuteafd_ffi::memory_ledger::snapshot()
                .total(cuteafd_ffi::memory_ledger::Space::Device, device);
            tracing::info!(target: "cuteafd::placement", probe = %serde_json::json!({
                "stage": stage, "device": device, "free": free, "total": total,
                "cuda_memory_scope": "raw cudaMemGetInfo; GB10 under-reports reclaimable page cache",
                "allocatable_free_bytes": library.cuda_allocatable_memory_info()?.0,
                "tracked": tracked, "untracked": total.saturating_sub(free).saturating_sub(tracked)
            }), "startup memory probe");
        }
        Ok(())
    })();
    library.cuda_set_device(current)?;
    result
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
