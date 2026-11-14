//! Cold partial-golden contract uses shared phase/arena/tail admission.
use super::{experts::Mxfp8Layout, head::Head, weights::execution_arena};
use anyhow::{Context as _, Result};
use cuteafd_core::memory_layout::{Category, MemoryKind};
use cuteafd_loader::{families::inkling2::{execution::UnitLayout, weight_layout::CoordinatorWeights, Inkling2Config},
    placement::planner::*, plan::Checkpoint};

pub(crate) fn admit(checkpoint: &Checkpoint, cfg: &Inkling2Config, target: &CoordinatorWeights,
    expert: &Mxfp8Layout, scratch: usize, unit: &UnitLayout, head_rows: usize,
    free: u64, total: u64, host: (u64, u64)) -> Result<ResolvedPlan> {
    let empty = Facts { phase_order: PhaseOrder::TARGET_THEN_DRAFT, resident_weights: vec![], allocations: vec![],
        deferred: vec![], sequential: vec![], reservations: vec![], allowances: vec![],
        driver_page_tables: vec![], allocation_granularity: vec![] };
    let mut contract = Contract::fixed(64, 1, 1, empty);
    contract.components.extend(weight_arena_components(&execution_arena(target)?, Location::coordinator(0),
        Lifetime::RESIDENT, 1, MemoryKind::Weights)?);
    contract.components.extend(weight_arena_components(&expert.arena, Location::coordinator(0),
        Lifetime::RESIDENT, 1, MemoryKind::Experts)?);
    let mut buffers = unit.buffers.iter().map(|(name, bytes)| (format!("unit.{name}"), Location::coordinator(0),
        *bytes as u64, Lifetime::TARGET, Category::Workspace)).collect::<Vec<_>>();
    for (name, bytes) in [("head.tokens", head_rows*4), ("head.normalized", head_rows*cfg.hidden*2),
        ("head.logits", head_rows*cfg.unpadded_vocab_size*4), ("head.workspace", Head::workspace_bytes(cfg, head_rows)?),
        ("expert.scratch", scratch)] {
        buffers.push((name.into(), Location::coordinator(0), bytes as u64, Lifetime::TARGET, Category::Workspace));
    }
    let mut mount = target.mount_load(checkpoint)?;
    mount.merge(expert.mount);
    for (name, bytes, location) in [("source", mount.pinned_source, Location::PINNED),
        ("rows", mount.full_row_cache, Location::HOST), ("tables", mount.pinned_tables, Location::PINNED)] {
        if bytes>0 { buffers.push((format!("mount.{name}"), location, bytes, Lifetime::RESIDENT, Category::Staging)); }
    }
    let sync = mount.pinned_source.max(1024*64*2);
    buffers.push(("sync-h2d".into(), Location::PINNED, sync, Lifetime::RESIDENT, Category::Staging));
    buffers.push(("relative.transpose.source".into(), Location::HOST, 16*1024*2,
        Lifetime { first: Phase::Load, last: Phase::Load }, Category::Staging));
    buffers.push(("relative.transpose.padded".into(), Location::HOST, 1024*64*2,
        Lifetime { first: Phase::Load, last: Phase::Load }, Category::Staging));
    for name in ["head.download", "head.host-logits"] {
        buffers.push((name.into(), Location::HOST, (head_rows*cfg.unpadded_vocab_size*4) as u64,
            Lifetime::TARGET, Category::Workspace));
    }
    for (name, location, bytes, lifetime, category) in buffers {
        contract.components.push(Component::allocation(Allocation::buffer(name, location, bytes, lifetime,
            category, "Inkling cold native forward owner"), ByteCurve::fixed(bytes, 1), false, 1));
    }
    contract.components.push(Component { activation: Activation::always(),
        bytes: ByteCurve::fixed(cuda_stream_slab_bytes(1)?, 1),
        kind: ComponentKind::Allowance(Allowance { location: Location::coordinator(0),
            bytes: cuda_stream_slab_bytes(1)?, lifetime: Lifetime::RESIDENT,
            source: "driver.stream-slabs: one ordered cold golden stream".into(), assumed: false }) });
    let head = crate::shared::target_head::TargetHead::configured().blas_head(true);
    contract.components.extend(serving::blas_components("sm_120", &[serving::BlasOwners {
        name: "inkling2/head".into(), process: "coordinator".into(), thread: "golden".into(),
        location: Location::coordinator(0), gemm: head.gemm(), lt: head.lt(), lifetime: Lifetime::TARGET,
        activation: Activation::always() }], 1, PhaseOrder::TARGET_THEN_DRAFT)?);
    fixed_request_tails(&mut contract.components, 1, &[])?;
    contract.add_driver_page_tables();
    let facts = contract.facts(&[], 1)?;
    let host = cuteafd_loader::placement::families::qwen4::host::HostMemory { total: host.0, available: host.1 };
    let policy = cuteafd_core::memory_policy::MemoryPolicy::current();
    let budgets = vec![Budget { id: "rtx0".into(), locations: vec![Location::coordinator(0)], physical_bytes: total,
        logical_bytes: policy.gpu_cap_bytes(total), baseline: Baseline::Sampled { free, phase: Phase::Load,
            source: "Inkling pre-weight sample after native module load".into() }, headroom: 0 }, host.budget("Inkling golden")?];
    let phases = evaluate(&facts, &budgets)?;
    tracing::info!(record = %serde_json::json!({"facts":facts,"budgets":budgets,"phases":phases}),
        "Inkling cold native forward admission");
    let shape = PrefillShape { rows: unit.rows.next_multiple_of(64) as u64, waves: 1 };
    Ok(ResolvedPlan { encoder_diagnostics: vec![],
        request: Request { context: Context::pinned(64), pool: Pin::Exact(64), onboard: Pin::Exact(0),
            prefill: Pin::Exact(shape), operation: Operation::Scoring, resident_weight_limit: None,
            spark_reply_formats: Default::default() },
        decisions: Decisions { attention: vec![], onboard_layers: vec![], expert_homes: Default::default(),
            embedding: None, draft_home: None, head_home: Some(0), intake_home: None, encoder: None,
            prefill: shape, lanes: 1, spark_reply_formats: None },
        pool_tokens: 64, pool_page_rows: 64, budgets, facts, phases })
}

/// Project only independently retained native device backings; loading pools
/// and host scoring temporaries keep their own phase accounting. Tensor aliases
/// remain Inventory views, and never become a second residency charge.
pub(crate) fn reconcile_native(plan: &ResolvedPlan, devices: CoordinatorDevices<'_>,
    receipts: &[Receipt]) -> Result<()> {
    let mut projected = devices.remap_plan_weight_arenas(plan)?;
    projected.facts.allocations.retain(|a| a.view.is_none() && a.location == Location::coordinator(0));
    projected.facts.driver_page_tables.retain(|table| table.location == Location::coordinator(0));
    projected.facts.allowances.retain(|owner| owner.location == Location::coordinator(0));
    projected.facts.reservations.clear();
    projected.facts.sequential.clear();
    projected.facts.deferred.clear();
    let expected = projected.facts.allocations.iter().map(|a| (a.id.clone(), a.clone())).collect::<std::collections::BTreeMap<_, _>>();
    let mut actual = Vec::new();
    for receipt in receipts.iter().filter(|r| r.allocation.view.is_none()) {
        let id = if receipt.allocation.group == "weight arena" { receipt.allocation.id.as_str() }
            else { receipt.allocation.id.strip_prefix("inkling2.").unwrap_or(&receipt.allocation.id) };
        let id = if id == "experts.scratch" { "expert.scratch" } else { id };
        let spec = expected.get(id).with_context(|| format!("Inkling unadmitted retained owner {id}"))?;
        anyhow::ensure!(receipt.allocation.bytes == spec.bytes && receipt.allocation.location == spec.location
            && receipt.allocation.lifetime == spec.lifetime && receipt.allocation.category == spec.category,
            "Inkling retained owner {id} bytes/location differ: actual{}/admitted{}", receipt.allocation.bytes, spec.bytes);
        // Metadata comes from the canonical pre-allocation contract after the
        // measured payload/location check, never from observed bytes.
        actual.push(Receipt { allocation: spec.clone(), address: receipt.address });
    }
    // Alias records are independently checked against admitted arena geometry
    // by Inventory; preserve them for shared address/backing validation without
    // inventing a second physical residency charge.
    for receipt in receipts.iter().filter(|r| r.allocation.view.is_some()) {
        let view = receipt.allocation.view.as_ref().unwrap();
        let backing = expected.get(view.backing.as_str()).context("Inkling alias backing not admitted")?;
        anyhow::ensure!(receipt.allocation.location == backing.location
            && view.offset.checked_add(receipt.allocation.bytes).is_some_and(|end| end <= backing.bytes),
            "Inkling alias outside admitted backing");
        projected.facts.allocations.push(receipt.allocation.clone());
        actual.push(receipt.clone());
    }
    projected.budgets.retain(|budget| budget.locations.contains(&Location::coordinator(0)));
    projected.phases = evaluate(&projected.facts, &projected.budgets)?;
    cuteafd_loader::placement::planner::reconcile(&projected, Phase::PostTargetKv, &actual)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    fn fixture() -> ResolvedPlan {
        let owner = Allocation::buffer("unit.partial", Location::coordinator(0), 4096,
            Lifetime::TARGET, Category::Workspace, "Inkling cold native forward owner");
        let facts = Facts { phase_order: PhaseOrder::TARGET_THEN_DRAFT, resident_weights: vec![], allocations: vec![owner],
            deferred: vec![], sequential: vec![], reservations: vec![], allowances: vec![Allowance {
                location: Location::coordinator(0), bytes: 1024, lifetime: Lifetime::RESIDENT,
                source: "driver.stream-slabs".into(), assumed: false }], driver_page_tables: vec![], allocation_granularity: vec![] };
        let budgets = vec![Budget { id: "rtx0".into(), locations: vec![Location::coordinator(0)],
            physical_bytes: 1<<30, logical_bytes: 1<<30, headroom: 0,
            baseline: Baseline::Sampled { free: 1<<30, phase: Phase::Load, source: "CPU fixture".into() } }];
        let shape = PrefillShape { rows: 64, waves: 1 };
        ResolvedPlan { encoder_diagnostics: vec![], request: Request { context: Context::pinned(64), pool: Pin::Exact(64),
            onboard: Pin::Exact(0), prefill: Pin::Exact(shape), operation: Operation::Scoring, resident_weight_limit: None,
            spark_reply_formats: Default::default() }, decisions: Decisions { attention: vec![], onboard_layers: vec![],
            expert_homes: Default::default(), embedding: None, draft_home: None, head_home: Some(0), intake_home: None,
            encoder: None, prefill: shape, lanes: 1, spark_reply_formats: None }, pool_tokens: 64, pool_page_rows: 64,
            phases: evaluate(&facts, &budgets).unwrap(), facts, budgets }
    }
    #[test]
    fn shared_reconcile_rejects_missing_extra_duplicate_and_wrong_extent() {
        let plan = fixture();
        let physical = [1];
        let devices = CoordinatorDevices::new(&physical).unwrap();
        let receipt = Receipt { allocation: Allocation::buffer("inkling2.unit.partial", Location::coordinator(0),
            4096, Lifetime::TARGET, Category::Workspace, "retained"), address: 0x1000 };
        reconcile_native(&plan, devices, &[receipt.clone()]).unwrap();
        assert!(reconcile_native(&plan, devices, &[]).is_err());
        assert!(reconcile_native(&plan, devices, &[receipt.clone(), receipt.clone()]).is_err());
        let mut wrong = receipt.clone(); wrong.allocation.bytes = 4095;
        assert!(reconcile_native(&plan, devices, &[wrong]).is_err());
        let mut wrong = receipt.clone(); wrong.allocation.id = "inkling2.unit.unadmitted".into();
        assert!(reconcile_native(&plan, devices, &[wrong]).is_err());
        let mut wrong = receipt.clone(); wrong.allocation.location = Location::coordinator(1);
        assert!(reconcile_native(&plan, devices, &[wrong]).is_err());
        let mut wrong = receipt.clone(); wrong.allocation.category = Category::Weights;
        assert!(reconcile_native(&plan, devices, &[wrong]).is_err());
        let mut wrong = receipt.clone(); wrong.allocation.lifetime = Lifetime::RESIDENT;
        assert!(reconcile_native(&plan, devices, &[wrong]).is_err());
        let mut alias = receipt.clone();
        alias.allocation.id = "inkling2.unit.partial.alias".into();
        alias.allocation.bytes = 64; alias.allocation.shape = vec![64];
        alias.allocation.view = Some(View { backing: "unit.partial".into(), offset: 128 });
        alias.address = 0x1080;
        reconcile_native(&plan, devices, &[receipt.clone(), alias.clone()]).unwrap();
        alias.address = 0x1090;
        assert!(reconcile_native(&plan, devices, &[receipt.clone(), alias.clone()]).is_err());
        alias.address = 0x1080;
        alias.allocation.view.as_mut().unwrap().offset = 4090;
        assert!(reconcile_native(&plan, devices, &[receipt, alias]).is_err());
    }
}
