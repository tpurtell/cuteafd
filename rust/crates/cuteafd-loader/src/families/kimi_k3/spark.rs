//! Offline calibration for #1178's TP6 allocation budget, not a live inventory.
//! Other worlds and wider exports retain the conservative package bound.
use anyhow::{Context, Result};
use cuteafd_core::{OwnerBytes, OwnerExtent, SparkExecutions};

fn calibrated(world: usize, executions: &SparkExecutions) -> bool {
    cuteafd_core::memory_layout::calibrated_kimi_spark(Some((cuteafd_core::ExpertGeometry::KIMI_K3,
        world, executions.compiled_capacity())))
}

/// The 2026-11-04 TP6 census retained m1/m80 decode and m1/m80/m256/m1024
/// prefill owners. Use those maxima for smaller admitted exports too; never
/// extrapolate this rank-width measurement to other Spark worlds.
pub fn owner_bytes(world: usize, extent: OwnerExtent) -> Result<OwnerBytes> {
    if world != 6 || extent.compiled > 1024 { return super::spark_owner_bytes(extent); }
    Ok(OwnerBytes {
        scratch: if extent.compiled <= 80 { 29_782_116 } else { 219_531_260 },
        io: u64::from(extent.capacity) * (3584 + 3584 / 32 + 16 * 8),
        pinned_staging: u64::from(extent.staging_rows) * 16 * 8,
    })
}

/// Exact owners plus fixed startup reserve; the inventory adds arena-size growth.
pub fn reserve_bytes(world: usize, waves: u64, unit_rows: u64) -> Result<u64> {
    let executions = super::spark_executions(waves, unit_rows)?;
    let runtime = cuteafd_core::memory_layout::spark_runtime_bytes(Some((
        cuteafd_core::ExpertGeometry::KIMI_K3, world, executions.compiled_capacity())), 0)
        .context("Kimi Spark runtime reserve overflow")?;
    executions.workspace_bytes(|extent| {
        if calibrated(world, &executions) { owner_bytes(world, extent) } else { super::spark_owner_bytes(extent) }
    })?
        .and_then(|bytes| bytes.total()).and_then(|bytes| bytes.checked_add(runtime))
        .context("Kimi calibrated Spark reserve overflow")
}

/// Use the same calibration in pre-placement admission and the published ledger.
pub fn charge(device: &mut cuteafd_core::memory_layout::DeviceLayout, world: usize,
    executions: &SparkExecutions) -> Result<()> {
    use cuteafd_core::memory_layout::{self, Basis, Category, Item};
    device.items.retain(|item| item.group != "Kimi synchronous upload staging");
    memory_layout::charge_spark_runtime(device).context("Spark arena charge overflow")?;
    if calibrated(world, executions) {
        device.items.push(Item::new(Category::Staging, "Kimi synchronous upload staging", "pinned",
            4096, Basis::Calibrated));
    }
    crate::plan::layout::spark_slots::charge_owners(device, executions, |extent| {
        if calibrated(world, executions) { owner_bytes(world, extent) } else { super::spark_owner_bytes(extent) }
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use cuteafd_core::memory_layout::{DeviceKind, DeviceLayout, Item, Basis, Category};

    #[test]
    fn measured_tp6_execution_and_startup_owners_preserve_other_worlds() -> Result<()> {
        let executions = super::super::spark_executions(3, 1024)?;
        let retained = executions.workspace_bytes(|extent| owner_bytes(6, extent))?.unwrap();
        assert_eq!(retained.device_bytes, 253_473_888);
        assert_eq!(retained.pinned_bytes, 401_408);
        assert_eq!(reserve_bytes(6, 3, 1024)?, retained.total().unwrap()
            + cuteafd_core::memory_layout::SPARK_WORKER_PROCESS_RESIDENT_BYTES + 4096);
        for world in [1, 2, 3, 4, 5, 7, 8] {
            assert_eq!(reserve_bytes(world, 3, 1024)?, super::super::spark_reserve_bytes(3, 1024)?);
        }
        assert_eq!(reserve_bytes(6, 3, 2752)?, super::super::spark_reserve_bytes(3, 2752)?);
        Ok(())
    }

    #[test]
    fn calibration_recharges_the_ledger_without_duplicate_upload_or_runtime_reserve() -> Result<()> {
        let executions = super::super::spark_executions(3, 1024)?;
        let mut device = DeviceLayout { kind: DeviceKind::Spark, index: 0,
            capacity_bytes: 121 << 30, worker_budget_bytes: 110 << 30,
            kv_tokens: 0, items: vec![Item::new(Category::Experts, "arena", "", 64 << 30, Basis::Exact)] };
        for _ in 0..2 {
            device.items.push(cuteafd_core::memory_layout::spark_worker_process_resident_item());
            charge(&mut device, 6, &executions)?;
            assert_eq!(device.items.iter().map(|item| item.bytes).sum::<u64>(), (64 << 30)
                + cuteafd_core::memory_layout::spark_arena_driver_overhead_bytes(64 << 30)
                + reserve_bytes(6, 3, 1024)?);
        }
        // Other worlds keep their original residual and owner accounting.
        let mut device = DeviceLayout { items: vec![Item::new(Category::Workspace,
            cuteafd_core::memory_layout::SPARK_WORKER_PROCESS_RESIDENT_GROUP, "", 5 << 29, Basis::Calibrated),
            Item::new(Category::Experts, "arena", "", 64 << 30, Basis::Exact)], ..device };
        charge(&mut device, 5, &executions)?;
        assert_eq!(device.items.iter().map(|item| item.bytes).sum::<u64>(), (64 << 30)
            + cuteafd_core::memory_layout::spark_arena_driver_overhead_bytes(64 << 30)
            + reserve_bytes(5, 3, 1024)?);
        Ok(())
    }
}
