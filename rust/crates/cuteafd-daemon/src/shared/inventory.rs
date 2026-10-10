//! `RuntimeInventory::measure`: one sample per coordinator GPU after the CUDA
//! context, cuBLAS and the family's program modules, before any weight. The
//! runtime side of `cuteafd_loader::placement::inventory`: admission sizes
//! from these samples ([`RuntimeInventory::baselines`]); the planner charges
//! `ArchContext` for the same bytes.
use anyhow::{Context, Result};
use cuteafd_ffi::NativeLibrary;
use cuteafd_loader::placement::{Baseline, RuntimeSample};

#[derive(Debug, Clone)]
pub(crate) struct RuntimeInventory {
    pub gpus: Vec<RuntimeSample>,
}

impl RuntimeInventory {
    /// Samples `devices` (lead first) with `load_modules` run on each device
    /// between a context sample and the admission sample, so `module_bytes`
    /// is the family's program modules alone. Restores the calling thread's
    /// device. `context_bytes` is CUDA's used bytes minus this process's
    /// ledger-tracked allocations on the device.
    pub fn measure(library: &NativeLibrary, devices: &[i32],
        mut load_modules: impl FnMut(i32) -> Result<()>) -> Result<Self> {
        let current = library.cuda_get_device()?;
        let mut gpus = Vec::with_capacity(devices.len());
        for &device in devices {
            library.cuda_set_device(device)?;
            let sample = (|| -> Result<RuntimeSample> {
                let (before, _) = library.cuda_memory_info()?;
                load_modules(device)?;
                let (free, total) = library.cuda_memory_info()?;
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
                    module_bytes: before.saturating_sub(free) as u64,
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
                driver = %gpu.driver, "runtime inventory (after context and program modules, before weights)");
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

/// Native code still to arrive on `device` (coordinator `rank` of a `split`
/// layout) before ready ([`cuteafd_loader::placement::LoadedCode::pending`]):
/// the measured loaded code of `program_family` with `experts` less what the
/// device already holds beyond the arch context and this process's tracked
/// allocations. Samples the device now: call it at the admission sample. 0 for
/// a family the table does not measure (it keeps its module formula).
pub(crate) fn pending_code(library: &NativeLibrary, device: i32, rank: usize, split: bool, program_family: &str,
    experts: &str) -> Result<u64> {
    let Some(code) = cuteafd_loader::placement::loaded_code(program_family, experts, split, rank as u8)
        else { return Ok(0) };
    let current = library.cuda_get_device()?;
    library.cuda_set_device(device)?;
    let sample = (|| -> Result<_> { Ok((library.cuda_memory_info()?, library.cuda_device_info(device)?)) })();
    library.cuda_set_device(current)?;
    let ((free, total), info) = sample?;
    let tracked = cuteafd_ffi::memory_ledger::snapshot().total(cuteafd_ffi::memory_ledger::Space::Device, device);
    let untracked = (total.saturating_sub(free) as u64).saturating_sub(tracked as u64);
    let arch = format!("sm_{}{}", info.compute_capability_major, info.compute_capability_minor);
    let context = cuteafd_loader::placement::ArchContext::for_device(&arch, total as u64).context_bytes;
    let pending = code.pending(untracked, context);
    tracing::info!(device, family = program_family, experts, loaded_code_bytes = code.bytes, untracked_bytes = untracked,
        pending_bytes = pending, "loaded code reserved beside admission (placement::inventory::LOADED_CODE)");
    Ok(pending)
}
