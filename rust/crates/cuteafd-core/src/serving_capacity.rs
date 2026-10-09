//! Pure startup capacity policy. Families describe physical allocation costs;
//! the daemon supplies memory measured before its allocations. Every device
//! constrains the same logical pool, whether KV is replicated or partitioned.
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use thiserror::Error;

/// Common logical GPU pool across model families; larger pools are explicit.
pub const DEFAULT_GPU_KV_TOKENS: u64 = 2 << 20;

/// Absolute free-memory floor for the logical 32 GiB candidate profiles.
/// Round upward so integer-byte admission never leaves less than 2.9 GiB.
pub const SMALL_CARD_HEADROOM_BYTES: u64 = (29 * (1u64 << 30)).div_ceil(10);

pub fn small_card_headroom_bytes(total_bytes: u64) -> u64 {
    if total_bytes <= 32u64 << 30 { SMALL_CARD_HEADROOM_BYTES } else { 0 }
}

pub fn admission_ceiling(total_bytes: u64, occupancy_percent: u32, minimum_free_bytes: u64)
    -> Result<u64, CapacityError> {
    if !(1..=100).contains(&occupancy_percent) {
        return Err(CapacityError::Invalid("GPU occupancy percent must be in 1..=100"));
    }
    let percentage = (u128::from(total_bytes) * u128::from(occupancy_percent) / 100) as u64;
    Ok(percentage.min(total_bytes.saturating_sub(minimum_free_bytes)))
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct CapacityPolicy {
    pub concurrency: u32,
    /// Logical tokens shared by active requests, independent of model context.
    /// Remaining GPU budget is available for expert onboarding.
    pub target_pool_tokens: u64,
    pub gpu_occupancy_percent: u32,
    /// Apply the absolute 2.9 GiB floor on each logical <=32 GiB device only.
    #[serde(default)]
    pub small_card_headroom: bool,
    /// None chooses the checkpoint length bounded by verified kernel support.
    /// Explicit requests beyond either capability are rejected, never clipped.
    pub max_context_tokens: Option<u64>,
    /// Explicit benchmark overrides retain their requested pool size, rounded
    /// up to the family's allocation quantum and checked before allocation.
    pub pool_tokens: Option<u64>,
}

impl Default for CapacityPolicy {
    fn default() -> Self {
        Self {
            concurrency: 16,
            target_pool_tokens: DEFAULT_GPU_KV_TOKENS,
            gpu_occupancy_percent: 97,
            small_card_headroom: false,
            max_context_tokens: None,
            pool_tokens: None,
        }
    }
}

impl CapacityPolicy {
    pub fn state_slots(self) -> Result<u32, CapacityError> {
        let slots = u64::from(self.concurrency)
            .checked_mul(5)
            .ok_or(CapacityError::Overflow("state slots"))?
            .div_ceil(4);
        u32::try_from(slots).map_err(|_| CapacityError::Overflow("state slots"))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct ContextLimits {
    pub checkpoint_max_tokens: u64,
    /// None only for a family whose context extent is dynamic. Indexed
    /// families must supply the validated compiled manifest bound.
    pub compiled_index_max_tokens: Option<u64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct DeviceMemory {
    pub device: u32,
    pub total_bytes: u64,
    /// Read before loading this engine's weights, modules or workspaces.
    pub baseline_free_bytes: u64,
}

/// A logical per-device ceiling, not an allocation occupying unused VRAM.
/// Existing usage (including other processes and CUDA runtime/graphs) is
/// charged against the smaller of the physical device and this ceiling.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GpuMemoryBudget(pub u64);

impl GpuMemoryBudget {
    pub fn from_gib(gib: f64) -> Result<Self, CapacityError> {
        let bytes = gib * (1u64 << 30) as f64;
        if !gib.is_finite() || bytes < 1.0 || bytes >= u64::MAX as f64 {
            return Err(CapacityError::Invalid("GPU budget GiB must be finite, positive and representable"));
        }
        Ok(Self(bytes as u64))
    }

    pub fn apply(self, memory: DeviceMemory) -> Result<DeviceMemory, CapacityError> {
        if self.0 == 0 || memory.total_bytes == 0 || memory.baseline_free_bytes > memory.total_bytes {
            return Err(CapacityError::Invalid("invalid GPU budget or physical memory sample"));
        }
        let used = memory.total_bytes - memory.baseline_free_bytes;
        let budget = self.0.min(memory.total_bytes);
        if used > budget {
            return Err(CapacityError::GpuBudgetExceeded { device: memory.device, required: used,
                budget, shortfall: used - budget });
        }
        Ok(DeviceMemory { total_bytes: budget, baseline_free_bytes: budget - used, ..memory })
    }

    pub fn admit(self, memory: DeviceMemory, additional: u64) -> Result<(), CapacityError> {
        if self.0 == 0 || memory.total_bytes == 0 || memory.baseline_free_bytes > memory.total_bytes {
            return Err(CapacityError::Invalid("invalid GPU budget or physical memory sample"));
        }
        let used = memory.total_bytes - memory.baseline_free_bytes;
        let required = used.checked_add(additional).ok_or(CapacityError::Overflow("GPU allocation budget"))?;
        let budget = self.0.min(memory.total_bytes);
        if required > budget {
            return Err(CapacityError::GpuBudgetExceeded { device: memory.device, required,
                budget, shortfall: required - budget });
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MemoryReservation {
    pub name: String,
    pub bytes: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DeviceCosts {
    pub device: u32,
    /// Include actual loaded representations (such as BF16 + FP8 copies),
    /// all workspace shapes/lanes, capture storage, transport, drafts, active
    /// state and retained marks. Header weight bytes alone are insufficient.
    pub reservations: Vec<MemoryReservation>,
    /// This device's bytes per logical pool allocation unit, including all
    /// persistent KV/index records and pool-sized tables/workspace metadata.
    /// A replica has the full cost on each device; partitioned heads use
    /// each device's actual share. Zero means this device does not own KV.
    pub pool_unit_bytes: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CapacityProfile {
    pub context: ContextLimits,
    pub pool_unit_rows: u64,
    pub devices: Vec<DeviceCosts>,
    /// A resolved inactive host snapshot quota; reported separately and
    /// never included in the active GPU token capacity calculation.
    pub host_prefix_bytes: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ResolvedDeviceCapacity {
    pub device: u32,
    pub total_bytes: u64,
    pub non_engine_bytes: u64,
    pub engine_budget_bytes: u64,
    pub reservations: Vec<MemoryReservation>,
    pub reserved_bytes: u64,
    pub pool_bytes: u64,
    pub unused_budget_bytes: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ResolvedCapacity {
    pub concurrency: u32,
    pub state_slots: u32,
    pub checkpoint_max_context_tokens: u64,
    pub compiled_index_max_context_tokens: Option<u64>,
    pub effective_max_context_tokens: u64,
    pub requested_kv_floor_tokens: u64,
    pub feasible_gpu_kv_tokens: u64,
    pub allocated_gpu_kv_tokens: u64,
    pub requested_floor_fits_hardware: bool,
    pub requested_floor_allocated: bool,
    pub requested_floor_shortfall_tokens: u64,
    /// Full effective-context sequences the pool can hold, capped at Cmax.
    /// Short requests can still reach Cmax when this count is lower.
    pub active_max_context_sequences: u32,
    pub host_prefix_bytes: u64,
    pub explicit_pool_override: bool,
    pub devices: Vec<ResolvedDeviceCapacity>,
}

#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum CapacityError {
    #[error("GPU {device} coordinator memory budget shortfall: need {required} bytes, budget {budget} bytes, shortfall {shortfall} bytes; weights, KV, workspaces, graphs and drafters must fit (reduce the pool/placement or raise the budget)")]
    GpuBudgetExceeded { device: u32, required: u64, budget: u64, shortfall: u64 },
    #[error("invalid capacity input: {0}")]
    Invalid(&'static str),
    #[error("capacity arithmetic overflows: {0}")]
    Overflow(&'static str),
    #[error("requested context {requested} exceeds checkpoint maximum {checkpoint}")]
    CheckpointContextExceeded { requested: u64, checkpoint: u64 },
    #[error("requested context {requested} exceeds compiled index extent {compiled}; export matching wider programs")]
    CompiledContextExceeded { requested: u64, compiled: u64 },
    #[error("GPU {device} fixed reservations need {reserved} bytes but its engine budget is {budget}; reduce placement/workspaces or choose a smaller compatible checkpoint")]
    ReservationsExceeded {
        device: u32,
        reserved: u64,
        budget: u64,
    },
    #[error(
        "explicit aligned pool {requested} tokens exceeds feasible GPU pool {feasible} tokens"
    )]
    PoolExceeded { requested: u64, feasible: u64 },
}

/// Admit a startup phase's fixed storage against the same physical GPU budget
/// used for serving. Loading temporaries need not consume the later KV pool.
/// Exact-boundary admission is allowed; this phase needs no KV allocation unit.
pub fn admit_device_reservations(
    gpu_occupancy_percent: u32,
    memory: DeviceMemory,
    reservations: &[MemoryReservation],
) -> Result<ResolvedDeviceCapacity, CapacityError> {
    admit_device_reservations_with_headroom(gpu_occupancy_percent, memory, reservations, 0)
}

pub fn admit_device_reservations_with_headroom(
    gpu_occupancy_percent: u32,
    memory: DeviceMemory,
    reservations: &[MemoryReservation],
    minimum_free_bytes: u64,
) -> Result<ResolvedDeviceCapacity, CapacityError> {
    if memory.total_bytes == 0 || memory.baseline_free_bytes > memory.total_bytes {
        return Err(CapacityError::Invalid("invalid physical GPU memory sample"));
    }
    let non_engine = memory.total_bytes - memory.baseline_free_bytes;
    let ceiling = admission_ceiling(memory.total_bytes, gpu_occupancy_percent, minimum_free_bytes)?;
    let budget = ceiling.saturating_sub(non_engine);
    let reserved = reservations
        .iter()
        .try_fold(0u64, |sum, item| sum.checked_add(item.bytes))
        .ok_or(CapacityError::Overflow("fixed device reservations"))?;
    if reserved > budget {
        return Err(CapacityError::ReservationsExceeded {
            device: memory.device,
            reserved,
            budget,
        });
    }
    Ok(ResolvedDeviceCapacity {
        device: memory.device,
        total_bytes: memory.total_bytes,
        non_engine_bytes: non_engine,
        engine_budget_bytes: budget,
        reservations: reservations.to_vec(),
        reserved_bytes: reserved,
        pool_bytes: 0,
        unused_budget_bytes: budget - reserved,
    })
}

/// Resolve once before allocation, then hand this exact result to startup.
/// An automatic target that cannot fit reports the physical shortfall; an
/// explicit oversized pool is rejected. Host copies do not add GPU capacity.
pub fn resolve_capacity(
    policy: CapacityPolicy,
    profile: &CapacityProfile,
    hardware: &[DeviceMemory],
) -> Result<ResolvedCapacity, CapacityError> {
    resolve_capacity_with_startup_peaks(policy, profile, hardware, &[])
}

/// Reserve each device's maximum temporary startup allocation before sizing KV.
/// Temporaries reuse the small-card floor, but must also fit the occupancy ceiling.
pub fn resolve_capacity_with_startup_peaks(
    policy: CapacityPolicy,
    profile: &CapacityProfile,
    hardware: &[DeviceMemory],
    startup_peaks: &[(u32, u64)],
) -> Result<ResolvedCapacity, CapacityError> {
    if policy.concurrency == 0 || policy.target_pool_tokens == 0 {
        return Err(CapacityError::Invalid(
            "concurrency and pool target must be positive",
        ));
    }
    if !(1..=100).contains(&policy.gpu_occupancy_percent) {
        return Err(CapacityError::Invalid(
            "GPU occupancy percent must be in 1..=100",
        ));
    }
    if profile.context.checkpoint_max_tokens == 0
        || profile.context.compiled_index_max_tokens == Some(0)
        || profile.pool_unit_rows == 0
        || profile.devices.is_empty()
    {
        return Err(CapacityError::Invalid(
            "positive checkpoint/kernel context and pool quantum required",
        ));
    }
    let context = policy.max_context_tokens.unwrap_or_else(|| {
        profile
            .context
            .compiled_index_max_tokens
            .map_or(profile.context.checkpoint_max_tokens, |limit| {
                limit.min(profile.context.checkpoint_max_tokens)
            })
    });
    if context == 0 {
        return Err(CapacityError::Invalid("requested context must be positive"));
    }
    if context > profile.context.checkpoint_max_tokens {
        return Err(CapacityError::CheckpointContextExceeded {
            requested: context,
            checkpoint: profile.context.checkpoint_max_tokens,
        });
    }
    if let Some(compiled) = profile.context.compiled_index_max_tokens {
        if context > compiled {
            return Err(CapacityError::CompiledContextExceeded {
                requested: context,
                compiled,
            });
        }
    }
    let floor = policy.pool_tokens.unwrap_or(policy.target_pool_tokens);
    let mut memory_by_device = BTreeMap::new();
    for &memory in hardware {
        if memory.total_bytes == 0
            || memory.baseline_free_bytes > memory.total_bytes
            || memory_by_device.insert(memory.device, memory).is_some()
        {
            return Err(CapacityError::Invalid(
                "invalid or duplicate physical GPU memory sample",
            ));
        }
    }
    if memory_by_device.len() != profile.devices.len() {
        return Err(CapacityError::Invalid(
            "one cost profile per selected physical GPU required",
        ));
    }
    let mut feasible_units = u64::MAX;
    let mut has_kv = false;
    let mut devices = Vec::with_capacity(profile.devices.len());
    for costs in &profile.devices {
        let memory = memory_by_device
            .remove(&costs.device)
            .ok_or(CapacityError::Invalid(
                "missing or duplicate physical GPU cost profile",
            ))?;
        let floor = if policy.small_card_headroom { small_card_headroom_bytes(memory.total_bytes) } else { 0 };
        let peak = startup_peaks.iter().filter(|(device, _)| *device == costs.device)
            .map(|(_, bytes)| *bytes).max().unwrap_or(0);
        let mut peak_costs = costs.reservations.clone();
        if peak > 0 {
            peak_costs.push(MemoryReservation { name: "startup.temporary_peak".into(), bytes: peak });
        }
        let mut admitted = admit_device_reservations_with_headroom(
            policy.gpu_occupancy_percent, memory, &peak_costs, floor.saturating_sub(peak))?;
        // Keep the temporary out of the permanent contract, while preserving
        // its unavailable pool budget. It is released before serving.
        admitted.reservations = costs.reservations.clone();
        admitted.reserved_bytes -= peak;
        admitted.engine_budget_bytes -= peak;
        if costs.pool_unit_bytes > 0 {
            has_kv = true;
            feasible_units =
                feasible_units.min(admitted.unused_budget_bytes / costs.pool_unit_bytes);
        }
        devices.push(admitted);
    }
    if !has_kv {
        return Err(CapacityError::Invalid(
            "at least one physical GPU must own the logical KV pool",
        ));
    }
    let feasible = feasible_units
        .checked_mul(profile.pool_unit_rows)
        .ok_or(CapacityError::Overflow("feasible KV tokens"))?;
    let units = match policy.pool_tokens {
        Some(0) => {
            return Err(CapacityError::Invalid(
                "explicit pool tokens must be positive",
            ))
        }
        Some(tokens) => tokens.div_ceil(profile.pool_unit_rows),
        None => floor.div_ceil(profile.pool_unit_rows).min(feasible_units),
    };
    let allocated = units
        .checked_mul(profile.pool_unit_rows)
        .ok_or(CapacityError::Overflow("aligned pool tokens"))?;
    if allocated > feasible {
        return Err(CapacityError::PoolExceeded {
            requested: allocated,
            feasible,
        });
    }
    for (device, costs) in devices.iter_mut().zip(&profile.devices) {
        device.pool_bytes = units
            .checked_mul(costs.pool_unit_bytes)
            .ok_or(CapacityError::Overflow("device pool bytes"))?;
        device.unused_budget_bytes -= device.pool_bytes;
    }
    let full_sequence_units = context.div_ceil(profile.pool_unit_rows);
    let active = (units / full_sequence_units).min(u64::from(policy.concurrency)) as u32;
    Ok(ResolvedCapacity {
        concurrency: policy.concurrency,
        state_slots: policy.state_slots()?,
        checkpoint_max_context_tokens: profile.context.checkpoint_max_tokens,
        compiled_index_max_context_tokens: profile.context.compiled_index_max_tokens,
        effective_max_context_tokens: context,
        requested_kv_floor_tokens: floor,
        feasible_gpu_kv_tokens: feasible,
        allocated_gpu_kv_tokens: allocated,
        requested_floor_fits_hardware: feasible >= floor,
        requested_floor_allocated: allocated >= floor,
        requested_floor_shortfall_tokens: floor.saturating_sub(allocated),
        active_max_context_sequences: active,
        host_prefix_bytes: profile.host_prefix_bytes,
        explicit_pool_override: policy.pool_tokens.is_some(),
        devices,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    const GIB: u64 = 1 << 30;

    fn device(id: u32, kv_unit_bytes: u64, reserved: u64) -> DeviceCosts {
        DeviceCosts {
            device: id,
            pool_unit_bytes: kv_unit_bytes,
            reservations: vec![MemoryReservation {
                name: "weights + all workspace shapes + marks".into(),
                bytes: reserved,
            }],
        }
    }
    fn hardware(id: u32, occupied: u64) -> DeviceMemory {
        DeviceMemory {
            device: id,
            total_bytes: 96 * GIB,
            baseline_free_bytes: 96 * GIB - occupied,
        }
    }
    fn profile(devices: Vec<DeviceCosts>) -> CapacityProfile {
        CapacityProfile {
            context: ContextLimits {
                checkpoint_max_tokens: 1024,
                compiled_index_max_tokens: None,
            },
            pool_unit_rows: 64,
            devices,
            host_prefix_bytes: 0,
        }
    }

    #[test]
    fn small_card_absolute_floor_charges_existing_usage_and_preserves_pro() {
        let total = GpuMemoryBudget::from_gib(31.8).unwrap().0;
        let floor = small_card_headroom_bytes(total);
        assert!(floor as f64 / GIB as f64 >= 2.9);
        assert_eq!(admission_ceiling(total, 97, floor).unwrap(), total - floor);
        assert_eq!(admission_ceiling(total, 80, floor).unwrap(), total * 80 / 100);
        let memory = DeviceMemory { device: 0, total_bytes: total, baseline_free_bytes: total - GIB };
        let budget = total - floor - GIB;
        let costs = vec![MemoryReservation { name: "fixed".into(), bytes: budget }];
        let exact = admit_device_reservations_with_headroom(97, memory, &costs, floor).unwrap();
        assert_eq!(exact.engine_budget_bytes, budget);
        let too_large = vec![MemoryReservation { name: "fixed".into(), bytes: budget + 1 }];
        assert!(matches!(admit_device_reservations_with_headroom(97, memory, &too_large, floor),
            Err(CapacityError::ReservationsExceeded { .. })));
        assert_eq!(small_card_headroom_bytes(32 * GIB), SMALL_CARD_HEADROOM_BYTES);
        assert_eq!(small_card_headroom_bytes(32 * GIB + 1), 0);
        let p = profile(vec![device(0, GIB, GIB)]);
        let small = resolve_capacity(CapacityPolicy { small_card_headroom: true, ..Default::default() }, &p, &[memory]).unwrap();
        assert_eq!(small.devices[0].engine_budget_bytes, budget);
        assert_eq!(small.allocated_gpu_kv_tokens, 26 * 64);
        let pro = hardware(0, GIB);
        let old = resolve_capacity(CapacityPolicy::default(), &p, &[pro]).unwrap();
        let new = resolve_capacity(CapacityPolicy { small_card_headroom: true, ..Default::default() }, &p, &[pro]).unwrap();
        assert_eq!(old, new);
    }

    #[test]
    fn measured_mimo_small_card_contract_reports_shortfall_after_absolute_floor() {
        let memory = DeviceMemory { device: 0, total_bytes: 34_144_990_003,
            baseline_free_bytes: 34_144_990_003 - 586_416_128 };
        let mut profile = profile(vec![device(0, 829_704, 17_965_280_768)]);
        profile.context.checkpoint_max_tokens = 1_048_576;
        let policy = CapacityPolicy { target_pool_tokens: 1_048_576,
            small_card_headroom: true, ..Default::default() };
        let new = resolve_capacity(policy, &profile, &[memory]).unwrap();
        assert_eq!(new.allocated_gpu_kv_tokens, 962_560);
        assert_eq!(new.requested_floor_shortfall_tokens, 86_016);
        assert_eq!(new.active_max_context_sequences, 0);
        assert!(memory.total_bytes - new.devices[0].non_engine_bytes
            - new.devices[0].reserved_bytes - new.devices[0].pool_bytes >= SMALL_CARD_HEADROOM_BYTES);
        assert!(matches!(resolve_capacity(CapacityPolicy { pool_tokens: Some(1_048_576), ..policy },
            &profile, &[memory]), Err(CapacityError::PoolExceeded { .. })));
    }

    #[test]
    fn mimo_pro_auto_pool_reserves_intake_peak_and_explicit_overask_stays_strict() {
        let memory = DeviceMemory { device: 0, total_bytes: 101_973_491_712,
            baseline_free_bytes: 101_973_491_712 - 586_416_128 };
        let budget = 98_327_870_832;
        let probe = 64 << 20;
        // Freeze rc1's combined fixed reservation and pool-metadata overhead.
        let fixed = 98_393_355_060 - 1_981_376 * 28_804 - probe;
        let profile = CapacityProfile { context: ContextLimits { checkpoint_max_tokens: 1 << 20,
            compiled_index_max_tokens: None }, pool_unit_rows: 64,
            devices: vec![DeviceCosts { device: 0, reservations: vec![MemoryReservation {
                name: "steady.fixed".into(), bytes: fixed }], pool_unit_bytes: 64 * 28_804 }],
            host_prefix_bytes: 0 };
        let policy = CapacityPolicy { small_card_headroom: true, ..Default::default() };
        let old = resolve_capacity(policy, &profile, &[memory]).unwrap();
        assert_eq!(old.devices[0].engine_budget_bytes, budget);
        assert_eq!(old.allocated_gpu_kv_tokens, 1_981_376);
        assert_eq!(fixed + old.devices[0].pool_bytes + probe, 98_393_355_060);
        assert_eq!(fixed + old.devices[0].pool_bytes + probe - budget, 65_484_228);
        let new = resolve_capacity_with_startup_peaks(policy, &profile, &[memory], &[(0, probe)]).unwrap();
        assert_eq!(new.allocated_gpu_kv_tokens, 1_979_072);
        assert!(fixed + new.devices[0].pool_bytes + probe <= budget);
        assert!(new.devices[0].reservations.iter().all(|r| !r.name.starts_with("startup.")));
        assert!(matches!(resolve_capacity_with_startup_peaks(CapacityPolicy {
            pool_tokens: Some(1_981_376), ..policy }, &profile, &[memory], &[(0, probe)]),
            Err(CapacityError::PoolExceeded { .. })));
        assert!(resolve_capacity_with_startup_peaks(CapacityPolicy { pool_tokens: Some(1_979_072),
            ..policy }, &profile, &[memory], &[(0, probe)]).is_ok());
    }

    #[test]
    fn startup_intake_probe_uses_floor_slack_without_shrinking_the_pool() {
        let memory = DeviceMemory { device: 0, total_bytes: 34_144_990_003,
            baseline_free_bytes: 34_144_990_003 - 586_416_128 };
        let floor = small_card_headroom_bytes(memory.total_bytes);
        let probe = 64 << 20;
        let costs = vec![
            MemoryReservation { name: "steady.fixed".into(), bytes: 17_965_280_768 },
            MemoryReservation { name: "kv.logical_pool".into(), bytes: 962_560 / 64 * 829_704 },
            MemoryReservation { name: "startup.spark_intake_probe_temporary".into(), bytes: probe },
        ];
        let stacked = admit_device_reservations_with_headroom(97, memory, &costs, floor);
        assert!(matches!(stacked, Err(CapacityError::ReservationsExceeded { .. })));
        let admitted = admit_device_reservations_with_headroom(97, memory, &costs,
            floor.saturating_sub(probe)).unwrap();
        assert_eq!(admitted.reserved_bytes, 30_511_137_792);
        assert_eq!(admitted.engine_budget_bytes - admitted.reserved_bytes, 693_657);
        let steady = admit_device_reservations_with_headroom(97, memory, &costs[..2], floor).unwrap();
        assert_eq!(steady.engine_budget_bytes - steady.reserved_bytes, 693_657);
        // A temporary larger than the floor still has to fit the physical ceiling.
        let oversized = vec![MemoryReservation { name: "startup".into(), bytes: memory.total_bytes }];
        assert!(admit_device_reservations_with_headroom(97, memory, &oversized, 0).is_err());
    }

    #[test]
    fn logical_budget_charges_existing_usage_and_never_enlarges_a_gpu() {
        let cap = GpuMemoryBudget::from_gib(32.0).unwrap();
        let sample = cap.apply(hardware(7, 5 * GIB)).unwrap();
        assert_eq!((sample.device, sample.total_bytes, sample.baseline_free_bytes), (7, 32 * GIB, 27 * GIB));
        cap.admit(hardware(7, 5 * GIB), 27 * GIB).unwrap();
        assert!(matches!(cap.admit(hardware(7, 5 * GIB), 28 * GIB),
            Err(CapacityError::GpuBudgetExceeded { device: 7, shortfall: GIB, .. })));
        assert_eq!(GpuMemoryBudget(128 * GIB).apply(hardware(0, 5 * GIB)).unwrap(), hardware(0, 5 * GIB));
        assert_eq!(cap.apply(cap.apply(hardware(0, 5 * GIB)).unwrap()).unwrap().baseline_free_bytes, 27 * GIB);
        let error = cap.apply(hardware(0, 33 * GIB)).unwrap_err().to_string();
        assert!(error.contains("shortfall 1073741824 bytes"), "{error}");
    }

    #[test]
    fn budget_validation_and_joint_pool_reservations_are_cpu_only() {
        for gib in [0.0, -1.0, f64::NAN, f64::INFINITY, 1e-12, 1e30] {
            assert!(GpuMemoryBudget::from_gib(gib).is_err(), "{gib}");
        }
        assert_eq!(GpuMemoryBudget::from_gib(0.5).unwrap().0, GIB / 2);
        let cap = GpuMemoryBudget(32 * GIB);
        let p = profile(vec![DeviceCosts { device: 3, pool_unit_bytes: GIB,
            reservations: [("weights", 8), ("workspace", 3), ("graphs", 2), ("draft", 4)]
                .into_iter().map(|(name, gib)| MemoryReservation { name: name.into(), bytes: gib * GIB }).collect() }]);
        let sample = cap.apply(hardware(3, 2 * GIB)).unwrap();
        let policy = CapacityPolicy { gpu_occupancy_percent: 100, ..Default::default() };
        let result = resolve_capacity(policy, &p, &[sample]).unwrap();
        assert_eq!(result.allocated_gpu_kv_tokens, 13 * 64);
        assert_eq!(result.devices[0].reserved_bytes, 17 * GIB);
        assert!(matches!(resolve_capacity(CapacityPolicy { pool_tokens: Some(14 * 64), ..policy }, &p, &[sample]),
            Err(CapacityError::PoolExceeded { .. })));
        let peer = cap.apply(hardware(9, 12 * GIB)).unwrap();
        let replicated = profile(vec![p.devices[0].clone(), DeviceCosts { device: 9, ..p.devices[0].clone() }]);
        let result = resolve_capacity(policy, &replicated, &[sample, peer]).unwrap();
        assert_eq!(result.allocated_gpu_kv_tokens, 3 * 64);
    }

    #[test]
    fn total_gpu_ceiling_subtracts_existing_usage_before_joint_reservations() {
        let p = profile(vec![device(0, GIB, 20 * GIB)]);
        let result =
            resolve_capacity(CapacityPolicy::default(), &p, &[hardware(0, 10 * GIB)]).unwrap();
        let gpu = &result.devices[0];
        assert_eq!(gpu.engine_budget_bytes, 96 * GIB * 97 / 100 - 10 * GIB);
        assert_eq!(result.allocated_gpu_kv_tokens, 63 * 64);
        assert_eq!((result.concurrency, result.state_slots), (16, 20));
        assert!(gpu.non_engine_bytes + gpu.reserved_bytes + gpu.pool_bytes <= 96 * GIB * 97 / 100);
        assert!(gpu.unused_budget_bytes < GIB);
    }

    #[test]
    fn replicated_kv_never_adds_gpu_capacities_and_partitioned_heads_use_actual_shares() {
        let one = resolve_capacity(
            CapacityPolicy::default(),
            &profile(vec![device(0, GIB, 20 * GIB)]),
            &[hardware(0, 0)],
        )
        .unwrap();
        let copies = resolve_capacity(
            CapacityPolicy::default(),
            &profile(vec![device(0, GIB, 20 * GIB), device(1, GIB, 20 * GIB)]),
            &[hardware(0, 0), hardware(1, 0)],
        )
        .unwrap();
        assert_eq!(copies.allocated_gpu_kv_tokens, one.allocated_gpu_kv_tokens);
        let split = resolve_capacity(
            CapacityPolicy::default(),
            &profile(vec![
                device(0, GIB / 2, 20 * GIB),
                device(1, GIB / 2, 20 * GIB),
            ]),
            &[hardware(0, 0), hardware(1, 0)],
        )
        .unwrap();
        assert!(split.allocated_gpu_kv_tokens >= 2 * one.allocated_gpu_kv_tokens);
        let bottleneck = resolve_capacity(
            CapacityPolicy::default(),
            &profile(vec![device(0, GIB, 20 * GIB), device(1, GIB, 40 * GIB)]),
            &[hardware(0, 0), hardware(1, 0)],
        )
        .unwrap();
        assert!(bottleneck.allocated_gpu_kv_tokens < copies.allocated_gpu_kv_tokens);
    }

    #[test]
    fn larger_automatic_target_reports_shortfall_without_counting_host_as_active_capacity() {
        let mut p = profile(vec![
            device(0, 64 * 14400, 10 * GIB),
            device(1, 64 * 14400, 10 * GIB),
        ]);
        p.context.checkpoint_max_tokens = 1 << 20;
        let policy = CapacityPolicy {
            target_pool_tokens: 8 << 20,
            ..Default::default()
        };
        let result = resolve_capacity(policy, &p, &[hardware(0, 0), hardware(1, 0)]).unwrap();
        assert_eq!(result.requested_kv_floor_tokens, 8 << 20);
        assert!(!result.requested_floor_fits_hardware && !result.requested_floor_allocated);
        assert_eq!(
            result.requested_floor_shortfall_tokens,
            (8 << 20) - result.allocated_gpu_kv_tokens
        );
        assert!(result.active_max_context_sequences < 8);
        assert_eq!(result.effective_max_context_tokens, 1 << 20);
        p.host_prefix_bytes = 500 * GIB;
        let host = resolve_capacity(policy, &p, &[hardware(0, 0), hardware(1, 0)]).unwrap();
        assert_eq!(host.allocated_gpu_kv_tokens, result.allocated_gpu_kv_tokens);
        assert_eq!(
            host.active_max_context_sequences,
            result.active_max_context_sequences
        );
    }

    #[test]
    fn checkpoint_and_compiled_context_are_separate_and_explicit_limits_never_clip() {
        let mut p = profile(vec![device(0, 64 * 11803, 10 * GIB)]);
        p.context = ContextLimits {
            checkpoint_max_tokens: 1 << 20,
            compiled_index_max_tokens: Some(131072),
        };
        let result = resolve_capacity(CapacityPolicy::default(), &p, &[hardware(0, 0)]).unwrap();
        assert_eq!(result.effective_max_context_tokens, 131072);
        assert_eq!(result.requested_kv_floor_tokens, DEFAULT_GPU_KV_TOKENS);
        let policy = CapacityPolicy {
            max_context_tokens: Some(131073),
            ..Default::default()
        };
        assert_eq!(
            resolve_capacity(policy, &p, &[hardware(0, 0)]).unwrap_err(),
            CapacityError::CompiledContextExceeded {
                requested: 131073,
                compiled: 131072
            }
        );
    }

    #[test]
    fn explicit_pool_override_rounds_to_real_units_and_is_admitted_before_allocation() {
        let p = profile(vec![device(0, GIB, 20 * GIB)]);
        let result = resolve_capacity(
            CapacityPolicy {
                pool_tokens: Some(65),
                ..Default::default()
            },
            &p,
            &[hardware(0, 0)],
        )
        .unwrap();
        assert_eq!(result.allocated_gpu_kv_tokens, 128);
        assert!(result.explicit_pool_override);
        assert!(result.requested_floor_allocated);
        assert_eq!(result.requested_kv_floor_tokens, 65);
        assert!(matches!(
            resolve_capacity(
                CapacityPolicy {
                    pool_tokens: Some(10000),
                    ..Default::default()
                },
                &p,
                &[hardware(0, 0)]
            ),
            Err(CapacityError::PoolExceeded { .. })
        ));
        assert!(matches!(
            resolve_capacity(
                CapacityPolicy::default(),
                &profile(vec![device(0, GIB, 95 * GIB)]),
                &[hardware(0, 0)]
            ),
            Err(CapacityError::ReservationsExceeded { device: 0, .. })
        ));
    }

    #[test]
    fn invalid_inputs_and_capacity_overflow_fail_typed() {
        let p = profile(vec![device(0, GIB, 0)]);
        assert!(resolve_capacity(
            CapacityPolicy::default(),
            &p,
            &[hardware(0, 0), hardware(0, 0)]
        )
        .is_err());
        assert!(resolve_capacity(
            CapacityPolicy {
                concurrency: 0,
                ..Default::default()
            },
            &p,
            &[hardware(0, 0)]
        )
        .is_err());
        let mut huge = p.clone();
        huge.pool_unit_rows = 64;
        assert_eq!(
            resolve_capacity(
                CapacityPolicy {
                    pool_tokens: Some(u64::MAX),
                    ..Default::default()
                },
                &huge,
                &[hardware(0, 0)]
            ),
            Err(CapacityError::Overflow("aligned pool tokens"))
        );
    }

    #[test]
    fn common_pool_target_is_independent_of_context_and_leaves_expert_budget() {
        let mut p = profile(vec![device(0, 64 * 14400, 20 * GIB)]);
        for context in [131072, 262144, 1 << 20] {
            p.context.checkpoint_max_tokens = context;
            let result =
                resolve_capacity(CapacityPolicy::default(), &p, &[hardware(0, 0)]).unwrap();
            assert_eq!(result.allocated_gpu_kv_tokens, DEFAULT_GPU_KV_TOKENS);
            assert_eq!(result.requested_kv_floor_tokens, DEFAULT_GPU_KV_TOKENS);
            assert!(result.requested_floor_allocated && result.requested_floor_fits_hardware);
            assert!(result.devices[0].unused_budget_bytes > 40 * GIB);
        }
    }

    #[test]
    fn loading_phase_admits_exact_boundary_without_reserving_its_temporary_forever() {
        let memory = DeviceMemory {
            device: 7,
            total_bytes: 1000,
            baseline_free_bytes: 900,
        };
        let costs = [MemoryReservation {
            name: "weights plus loading temporary".into(),
            bytes: 870,
        }];
        let admitted = admit_device_reservations(97, memory, &costs).unwrap();
        assert_eq!(
            (
                admitted.engine_budget_bytes,
                admitted.reserved_bytes,
                admitted.unused_budget_bytes
            ),
            (870, 870, 0)
        );
        assert_eq!(admitted.pool_bytes, 0);
        let steady = CapacityProfile {
            context: ContextLimits {
                checkpoint_max_tokens: 100,
                compiled_index_max_tokens: None,
            },
            pool_unit_rows: 1,
            devices: vec![device(7, 1, 500)],
            host_prefix_bytes: 0,
        };
        let result = resolve_capacity(
            CapacityPolicy {
                target_pool_tokens: 300,
                ..Default::default()
            },
            &steady,
            &[memory],
        )
        .unwrap();
        assert_eq!(result.allocated_gpu_kv_tokens, 300);
        assert_eq!(result.devices[0].unused_budget_bytes, 70);
        let above = [MemoryReservation {
            name: "load".into(),
            bytes: 871,
        }];
        assert_eq!(
            admit_device_reservations(97, memory, &above),
            Err(CapacityError::ReservationsExceeded {
                device: 7,
                reserved: 871,
                budget: 870
            })
        );
        let overflow = [
            MemoryReservation {
                name: "a".into(),
                bytes: u64::MAX,
            },
            MemoryReservation {
                name: "b".into(),
                bytes: 1,
            },
        ];
        assert_eq!(
            admit_device_reservations(97, memory, &overflow),
            Err(CapacityError::Overflow("fixed device reservations"))
        );
    }
}
