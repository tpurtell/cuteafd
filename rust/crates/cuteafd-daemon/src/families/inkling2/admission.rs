//! Cold partial-golden contract uses shared phase/arena/tail admission.
use super::{experts::Mxfp8Layout, head::Head, weights::execution_arena};
use anyhow::Result;
use cuteafd_core::memory_layout::{Category, MemoryKind};
use cuteafd_loader::{families::inkling2::{execution::UnitLayout, weight_layout::CoordinatorWeights, Inkling2Config},
    placement::planner::*, plan::Checkpoint};

pub(crate) fn admit(checkpoint: &Checkpoint, cfg: &Inkling2Config, target: &CoordinatorWeights,
    expert: &Mxfp8Layout, scratch: usize, unit: &UnitLayout, head_rows: usize,
    free: u64, total: u64, host: (u64, u64)) -> Result<Facts> {
    let empty = Facts { phase_order: PhaseOrder::TARGET_THEN_DRAFT, resident_weights: vec![], allocations: vec![],
        deferred: vec![], sequential: vec![], reservations: vec![], allowances: vec![],
        driver_page_tables: vec![], allocation_granularity: vec![] };
    let mut contract = Contract::fixed(64, 1, 1, empty);
    contract.components.extend(weight_arena_components(&execution_arena(target)?, Location::coordinator(0),
        Lifetime::RESIDENT, 1, MemoryKind::Weights)?);
    contract.components.extend(weight_arena_components(&expert.arena, Location::coordinator(0),
        Lifetime::RESIDENT, 1, MemoryKind::Weights)?);
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
    contract.components.push(Component::allocation(Allocation::buffer("driver.stream-slabs", Location::coordinator(0),
        cuda_stream_slab_bytes(1)?, Lifetime::RESIDENT, Category::Runtime, "one ordered cold golden stream"),
        ByteCurve::fixed(cuda_stream_slab_bytes(1)?, 1), false, 1));
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
    Ok(facts)
}
