//! Per-family `FAMILY_RESERVE` constants and the shared reserve helpers
//! (`cuteafd_core::memory_policy`). Every constant defaults to 0: the named,
//! measured components (CUDA context, loaded code, graphs, runtime growth)
//! carry each family's reserve. Each comment records the headroom the
//! constant replaced and what it probably covered; the values are
//! placeholders until the family's memory budget is measured.
use cuteafd_core::memory_policy::{DeviceKind, MemoryPolicy};

/// GLM 5.3 and Mistral Large 4 (GLM DSA executor). Inherited: 2 GiB hardcoded
/// headroom (`families/glm5/mod.rs:1274` serve, `LayoutOptions::headroom_bytes`
/// plan, dev `3ce3fbd19`). Probably covered allocator rounding and post-ready
/// growth beyond the #1242 census figures. Placeholder for measurement.
pub const GLM5: u64 = 0;
/// GLM 5.3 Flash. Inherited: 2 GiB `--headroom-gib` / `GLM5_FLASH_HEADROOM_GIB`
/// (`families/glm5_flash/mod.rs:155-159`, dev `3ce3fbd19`). Covered lazily
/// loaded kernels, cuBLAS and allocator rounding the startup cannot measure.
/// RESTORED to 2 GiB: the K4 1-Spark memory probe (2026-11-07) peaked above
/// its own admitted Growth/Ready bound on GPU0 (+0.28-0.73 GB) and GPU1 in
/// both drafter arms, fitting only inside this margin. Back to 0 once the
/// missing owner is charged: #1442.
pub const GLM5_FLASH: u64 = 2 << 30;
/// Kimi K3. Inherited: 2 GiB hardcoded headroom (`families/kimi_k3/startup.rs:194`,
/// dev `3ce3fbd19`). Probably covered allocator slack and unmeasured workspaces
/// (no warmed census yet). Placeholder for measurement.
pub const KIMI_K3: u64 = 0;
/// Muse Spark. Inherited: 2 GiB hardcoded headroom (`families/muse_spark/serve.rs:211`,
/// dev `3ce3fbd19`). Covered first-use CUDA/module/allocator growth beyond a
/// 160 MiB opaque per-GPU estimate. Deleted in the 2026-11-08 memory campaign:
/// module loads now happen before the measured pre-weight sample (#1554), the
/// driver stream slabs are charged per GPU (`muse/driver-stream-slabs/gpuN`),
/// and the measured unexplained residual at Ready is inside `MEMORY_PADDING`.
pub const MUSE_SPARK: u64 = 0;
/// Step-5. Inherited: 2 GiB headroom (`families/step5/serve.rs:317`, dev
/// `3ce3fbd19`), covering allocator slack and runtime growth; Step-5 captures
/// no graphs. Measured 0 on 2026-11-09; HQA EXL3 K4/K6 core, MTP3, ostrich,
/// dodo and emu, full-device cap, 50 MiB padding, `FAMILY_RESERVE=0`, KV
/// target = max = 100M. One RTX PRO 6000 charged named owners 101,905,094,192 B
/// against 101,907,562,496 B CUDA used (+2,468,304 B, 2.35 MiB); two RTX PRO
/// 6000 charged 101,836,169,716 B against 101,840,388,096 B on the first card
/// and 91,740,479,988 B against 91,744,698,368 B on the second (+4,218,380 B,
/// 4.02 MiB each). Samples through prefill, decode and the post-sweep window
/// stay at the Ready value and the daemon reports 0 B allocated after Ready, so
/// the residual is the same after a served request. The 1x measurement reads
/// 2,402,768 B at the shipping-default pool too, so it is independent of pool
/// size. Every owner is charged from `placement/families/step5.rs`: weight
/// arenas, SM120 allocation tails, BLAS handles per device and thread, measured
/// peak-live-stream slabs, and driver page tables for pinned host.
pub const STEP5: u64 = 0;
/// Qwen MoE (qwen4). Inherited: max(2 GiB, 3 GiB) = 3 GiB headroom
/// (`families/qwen4/admission.rs:80`, `dual.rs:93`, dev `3ce3fbd19`). Probably
/// covered lazy cuBLAS/graph growth beyond the startup-graph margin.
/// Placeholder for measurement.
pub const QWEN4: u64 = 0;
/// Dense Qwen (qwen3_5). Inherited: 1 GiB headroom
/// (`placement/families/qwen3_5/admission.rs:63`, dev `3ce3fbd19`). Probably
/// covered allocator slack beside the 512 MiB head/runtime growth estimate.
/// Placeholder for measurement.
pub const QWEN3_5: u64 = 0;
/// DeepSeek V4. Inherited: 3 GiB headroom floor (2.9 GiB on cards <=32 GiB)
/// from `deepseek_v4_headroom_bytes` and `--reserve-gib` 10 (dev `3ce3fbd19`).
/// Probably covered allocator slack and unmeasured workspace beside the lazy
/// graph budget. Placeholder for measurement.
pub const DEEPSEEK_V4: u64 = 0;
/// DeepSeek V4.1. Inherited: none (its measured graph and runtime-growth
/// reserves stay as named components; `--memory-reservation` was an optional
/// ceiling, now `GPU_MEMORY_CAP`). Placeholder for measurement.
pub const DEEPSEEK_V41: u64 = 0;
/// MiMo V2.6. Inherited: max(3% of the card + 64 MiB Spark intake probe,
/// 2.9 GiB small-card floor) (`families/mimo_v2/admission.rs:354`, dev
/// `3ce3fbd19`), about 2.9 GiB on a PRO. Probably covered allocator slack and
/// post-ready growth beyond the 570 MiB measured runtime figure; the 64 MiB
/// probe stays a named component. Placeholder for measurement.
pub const MIMO_V2: u64 = 0;

/// The family's `FAMILY_RESERVE` constant by family id.
pub fn family_constant(family: &str) -> u64 {
    match family {
        "glm5" | "mistral_large4" => GLM5,
        "glm5_flash" => GLM5_FLASH,
        "kimi_k3" => KIMI_K3,
        "muse_spark" => MUSE_SPARK,
        "step5" => STEP5,
        "qwen4" => QWEN4,
        "qwen3_5" => QWEN3_5,
        "deepseek_v4" => DEEPSEEK_V4,
        "deepseek_v41" => DEEPSEEK_V41,
        "mimo_v2" => MIMO_V2,
        _ => 0,
    }
}

/// The per-GPU named reserve a family's budgets keep free: `GPU_RESERVE`
/// when set, else `FAMILY_RESERVE` (the key or the family constant) plus release padding.
pub fn gpu(family: &str) -> u64 {
    MemoryPolicy::current().named_reserve(DeviceKind::Gpu, family_constant(family))
}

/// The per-Spark named reserve (`SPARK_RESERVE`, else 0).
pub fn spark() -> u64 {
    MemoryPolicy::current().named_reserve(DeviceKind::Spark, 0)
}

/// Nonzero layout items for the named reserve: `GPU_RESERVE` when overridden,
/// else separate `family reserve` and `release padding` components.
pub fn gpu_item(family: &str) -> Vec<cuteafd_core::memory_layout::Item> {
    use cuteafd_core::memory_layout::{Basis, Category, Item, ItemLifetime};
    let bytes = gpu(family);
    let name = if MemoryPolicy::current().gpu_reserve.is_some() { "GPU_RESERVE" } else { "family reserve" };
    let padding = MemoryPolicy::current().release_padding();
    [(name, bytes.saturating_sub(padding)), ("release padding", padding)].into_iter()
        .filter(|(_, bytes)| *bytes > 0)
        .map(|(name, bytes)| Item::new(Category::Reserved, name, "", bytes, Basis::Formula)
            .with_lifetime(ItemLifetime::PostReadyReserve)).collect()
}

/// A built-in GPU reserve component's charged bytes (0 under `GPU_RESERVE`).
pub fn gpu_component(bytes: u64) -> u64 {
    MemoryPolicy::current().component(DeviceKind::Gpu, bytes)
}

/// The per-Spark cap for runtime admission: `SPARK_MEMORY_CAP`, else the
/// shared Spark device default.
pub fn spark_cap() -> u64 {
    MemoryPolicy::current().spark_cap_bytes(crate::plan::layout::DEFAULT_SPARK_DEVICE_BYTES)
}

/// A built-in Spark reserve component's charged bytes (0 under `SPARK_RESERVE`).
pub fn spark_component(bytes: u64) -> u64 {
    MemoryPolicy::current().component(DeviceKind::Spark, bytes)
}

/// The startup breakdown of one device from its layout items: cap, weights,
/// workspaces, the named reserve components (runtime, graph and reserved
/// items, plus the policy's named reserve) and KV.
pub fn breakdown(device: impl Into<String>, kind: DeviceKind, cap_bytes: u64, named_reserve: u64,
    items: &[cuteafd_core::memory_layout::Item]) -> cuteafd_core::memory_policy::DeviceBreakdown {
    use cuteafd_core::memory_layout::Category;
    use cuteafd_core::memory_policy::{DeviceBreakdown, ReserveComponent};
    let policy = MemoryPolicy::current();
    let overridden = policy.reserve_override(kind).is_some();
    let mut breakdown = DeviceBreakdown { device: device.into(), cap_bytes, weights_bytes: 0, workspace_bytes: 0,
        reserve: Vec::new(), reserve_overridden: overridden, kv_bytes: 0, other_bytes: 0 };
    let mut components = std::collections::BTreeMap::<String, u64>::new();
    for item in items {
        let sum = match item.category {
            Category::Weights | Category::Embedding | Category::Experts | Category::Drafter => &mut breakdown.weights_bytes,
            Category::Workspace | Category::Transport | Category::Staging => &mut breakdown.workspace_bytes,
            Category::Kv | Category::Prefix => &mut breakdown.kv_bytes,
            Category::Tables => &mut breakdown.other_bytes,
            Category::Runtime | Category::Reserved => {
                // Named by the item group's leading label (sources carry long evidence).
                let name = item.group.split([':', ';', '(']).next().unwrap_or(&item.group).trim();
                if !matches!(name, "family reserve" | "release padding" | "GPU_RESERVE" | "SPARK_RESERVE") {
                    let entry = components.entry(name.to_owned()).or_default();
                    *entry = entry.saturating_add(item.bytes);
                }
                continue;
            }
        };
        *sum = sum.saturating_add(item.bytes);
    }
    let named = match (overridden, kind) {
        (true, DeviceKind::Gpu) => "GPU_RESERVE",
        (true, DeviceKind::Spark) => "SPARK_RESERVE",
        (false, _) => "family reserve",
    };
    breakdown.reserve = components.into_iter().map(|(name, bytes)| ReserveComponent { name, bytes }).collect();
    // Sparks carry no family reserve; only an explicit SPARK_RESERVE is named there.
    if kind == DeviceKind::Gpu || overridden {
        let padding = if kind == DeviceKind::Gpu { policy.release_padding() } else { 0 };
        breakdown.reserve.push(ReserveComponent { name: named.into(), bytes: named_reserve.saturating_sub(padding) });
        if kind == DeviceKind::Gpu && !overridden {
            breakdown.reserve.push(ReserveComponent { name: "release padding".into(), bytes: padding });
        }
    }
    breakdown
}

/// Log one line per device (each coordinator GPU and each Spark): the cap,
/// weights, workspaces, the named reserve components, KV target/min/max and
/// the final pool. `cuteafd plan` prints the same lines.
pub fn log_layout(layout: &cuteafd_core::memory_layout::MemoryLayout, family: &str, context: u64, gpu_caps: &[u64],
    physical_card: u64) {
    for line in layout_lines(layout, family, context, gpu_caps, physical_card) {
        tracing::info!(target: "cuteafd::memory_policy", "{line}");
    }
}

/// `gpu_caps` are the coordinator GPUs' caps by index (a layout's own RTX
/// capacity may already exclude the named reserve); `physical_card` the
/// smallest physical card, which picks the default KV target.
pub fn layout_lines(layout: &cuteafd_core::memory_layout::MemoryLayout, family: &str, context: u64,
    gpu_caps: &[u64], physical_card: u64) -> Vec<String> {
    use cuteafd_core::memory_layout::DeviceKind as Layout;
    let policy = MemoryPolicy::current();
    let smallest = physical_card;
    let Ok(kv) = policy.kv(smallest, context) else { return Vec::new() };
    let pool = cuteafd_core::memory_policy::PoolBreakdown { kv, pool_tokens: layout.pool_tokens };
    layout.devices.iter().map(|device| {
        let (kind, named) = match device.kind {
            Layout::Rtx => (DeviceKind::Gpu, gpu(family)),
            Layout::Spark => (DeviceKind::Spark, spark()),
        };
        let cap = match device.kind {
            Layout::Rtx => gpu_caps.get(device.index as usize).copied().unwrap_or(device.capacity_bytes),
            // A Spark's cap is its worker ceiling.
            Layout::Spark if device.worker_budget_bytes > 0 => device.worker_budget_bytes,
            Layout::Spark => device.capacity_bytes,
        };
        cuteafd_core::memory_policy::render_line(&breakdown(device.name(), kind, cap, named, &device.items), &pool)
    }).collect()
}

/// Log the admitted per-device breakdown of a shared-planner plan (each
/// coordinator GPU and each Spark budget), at runtime admission.
pub fn log_plan(plan: &super::planner::ResolvedPlan, family: &str) {
    use super::planner::{Device, Domain, Location};
    let policy = MemoryPolicy::current();
    let smallest = plan.budgets.iter().filter(|budget| budget.locations.iter()
        .any(|location| matches!(location.device, Device::Coordinator(_))))
        .map(|budget| budget.physical_bytes).min().unwrap_or(u64::MAX);
    let Ok(kv) = policy.kv(smallest, plan.request.context.maximum) else { return };
    let pool = cuteafd_core::memory_policy::PoolBreakdown { kv, pool_tokens: plan.pool_tokens };
    cuteafd_ffi::memory_ledger::fault::set_inputs(serde_json::json!({
        "kv_geometry": {"pages": plan.pool_tokens.checked_div(plan.pool_page_rows),
            "tokens_per_page": plan.pool_page_rows, "capacity_tokens": plan.pool_tokens,
            "status": "admitted planner geometry; live pool occupancy not captured"},
        "memory_padding": {"bytes_per_gpu": policy.release_padding(),
            "source": if policy.gpu_reserve.is_some() { "GPU_RESERVE override" }
                else if policy.memory_padding.is_some() { "MEMORY_PADDING / --memory-padding" }
                else { "cuteafd-core::memory_policy::DEFAULT_MEMORY_PADDING" }},
        "family": family, "context_tokens": plan.served_context(), "pool_tokens": plan.pool_tokens,
        "kv_target_tokens": kv.target, "kv_min_tokens": kv.min, "kv_max_tokens": kv.max,
        "prefill_rows": plan.decisions.prefill.rows, "prefill_waves": plan.decisions.prefill.waves,
        "prefill_span_rows": plan.decisions.prefill.rows.saturating_mul(plan.decisions.prefill.waves),
        "lanes": plan.decisions.lanes, "onboard_layers": plan.decisions.onboard_layers,
        "spark_reply_formats": plan.decisions.spark_reply_formats,
    }));
    for budget in &plan.budgets {
        let (kind, location) = match budget.locations.first().map(|location| location.device) {
            Some(Device::Coordinator(gpu)) => (DeviceKind::Gpu, Location::coordinator(gpu)),
            Some(Device::Spark(rank)) => (DeviceKind::Spark, Location { device: Device::Spark(rank), domain: Domain::Managed }),
            _ => continue,
        };
        let Ok(mut items) = plan.items(location) else { continue };
        // Growth-only reserve allowances (lazy graphs, runtime growth) are reserve too.
        items.extend(plan.facts.allocations.iter().filter(|allocation| allocation.location == location
            && allocation.view.is_none() && matches!(allocation.category,
                cuteafd_core::memory_layout::Category::Runtime | cuteafd_core::memory_layout::Category::Reserved)
            && !allocation.lifetime.contains(super::planner::Phase::Ready, plan.facts.phase_order))
            .map(|allocation| cuteafd_core::memory_layout::Item::new(allocation.category, &allocation.group, "",
                allocation.bytes, cuteafd_core::memory_layout::Basis::Exact)));
        let cap = budget.physical_bytes.min(budget.logical_bytes);
        let breakdown = breakdown(budget.id.clone(), kind, cap, budget.headroom, &items);
        cuteafd_ffi::memory_ledger::fault::set_inputs(serde_json::json!({format!("device.{}", budget.id): breakdown}));
        let line = cuteafd_core::memory_policy::render_line(&breakdown, &pool);
        tracing::info!(target: "cuteafd::memory_policy", "{line}");
    }
}

/// Log the admitted per-GPU breakdown of a legacy-solver placement.
pub fn log_placement(placement: &super::Placement, request: &super::PlacementRequest) {
    let policy = MemoryPolicy::current();
    let smallest = request.inventory.gpus.iter().map(|gpu| gpu.capacity_bytes).min().unwrap_or(u64::MAX);
    let Ok(kv) = policy.kv(smallest, request.pool.compiled_context) else { return };
    let pool = cuteafd_core::memory_policy::PoolBreakdown { kv, pool_tokens: placement.pool_tokens };
    for (gpu, (budget, items)) in request.inventory.gpus.iter().zip(&placement.items).enumerate() {
        let line = cuteafd_core::memory_policy::render_line(&breakdown(format!("rtx{gpu}"), DeviceKind::Gpu,
            budget.capacity_bytes, budget.reserve_bytes, items), &pool);
        tracing::info!(target: "cuteafd::memory_policy", "{line}");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn family_constants_default_to_zero_and_policy_overrides() {
        for family in ["glm5", "mistral_large4", "glm5_flash", "kimi_k3", "muse_spark", "step5", "qwen4",
            "qwen3_5", "deepseek_v4", "deepseek_v41", "mimo_v2"] {
            assert_eq!(gpu(family), family_constant(family) + cuteafd_core::memory_policy::DEFAULT_MEMORY_PADDING, "{family}");
        }
        MemoryPolicy { family_reserve: Some(5), ..Default::default() }.scoped(|| {
            assert_eq!(gpu("glm5"), 5 + cuteafd_core::memory_policy::DEFAULT_MEMORY_PADDING);
            assert_eq!(gpu_component(7), 7);
        });
        MemoryPolicy { gpu_reserve: Some(9), family_reserve: Some(5), ..Default::default() }.scoped(|| {
            assert_eq!((gpu("glm5"), gpu_component(7), spark_component(7), spark()), (9, 0, 7, 0));
        });
        MemoryPolicy { spark_reserve: Some(3), ..Default::default() }.scoped(|| {
            assert_eq!((spark(), spark_component(7), gpu_component(7)), (3, 0, 7));
        });
    }

    #[test]
    fn breakdown_names_every_reserve_component_and_the_family_reserve() {
        use cuteafd_core::memory_layout::{Basis, Category, Item};
        let items = [Item::new(Category::Weights, "attention", "bf16", 40 << 30, Basis::Exact),
            Item::new(Category::Workspace, "decode", "", 2 << 30, Basis::Formula),
            Item::new(Category::Runtime, "CUDA context and cuBLAS", "", 627 << 20, Basis::Calibrated),
            Item::new(Category::Runtime, "GLM startup graph instantiate: Measured; #1242", "", 2 << 30, Basis::Calibrated),
            Item::new(Category::Kv, "records", "", 30 << 30, Basis::Formula)];
        let device = MemoryPolicy { family_reserve: Some(1 << 30), ..Default::default() }
            .scoped(|| breakdown("rtx0", DeviceKind::Gpu, 96 << 30, gpu("glm5"), &items));
        assert_eq!((device.weights_bytes, device.workspace_bytes, device.kv_bytes), (40 << 30, 2 << 30, 30 << 30));
        let names = device.reserve.iter().map(|c| (c.name.as_str(), c.bytes)).collect::<Vec<_>>();
        assert_eq!(names, [("CUDA context and cuBLAS", 627 << 20), ("GLM startup graph instantiate", 2 << 30),
            ("family reserve", 1 << 30), ("release padding", cuteafd_core::memory_policy::DEFAULT_MEMORY_PADDING)]);
        assert!(!device.reserve_overridden);
        let overridden = MemoryPolicy { gpu_reserve: Some(5 << 30), ..Default::default() }
            .scoped(|| breakdown("rtx0", DeviceKind::Gpu, 96 << 30, gpu("glm5"), &items[..2]));
        assert!(overridden.reserve_overridden);
        assert_eq!(overridden.reserve_bytes(), 5 << 30);
        assert_eq!(overridden.reserve.last().unwrap().name, "GPU_RESERVE");
    }

    #[test]
    fn layout_lines_render_one_line_per_device_with_cap_and_components() {
        use cuteafd_core::memory_layout::{Basis, Category, DeviceKind as Kind, DeviceLayout, Item, MemoryLayout};
        let gpu = DeviceLayout { kind: Kind::Rtx, index: 0, capacity_bytes: 90 << 30, worker_budget_bytes: 0, kv_tokens: 1 << 20,
            items: vec![Item::new(Category::Weights, "attention", "", 40 << 30, Basis::Exact),
                Item::new(Category::Runtime, "context+modules", "", 600 << 20, Basis::Calibrated),
                Item::new(Category::Kv, "records", "", 20 << 30, Basis::Formula)] };
        let spark = DeviceLayout { kind: Kind::Spark, index: 0, capacity_bytes: 121 << 30, worker_budget_bytes: 108 << 30,
            kv_tokens: 0, items: vec![Item::new(Category::Experts, "routed", "", 90 << 30, Basis::Exact),
                cuteafd_core::memory_layout::spark_worker_process_resident_item()] };
        let layout = MemoryLayout { devices: vec![gpu, spark], pool_tokens: 1 << 20, waste: vec![], notes: vec![] };
        let lines = MemoryPolicy { family_reserve: Some(1 << 30), ..Default::default() }
            .scoped(|| layout_lines(&layout, "glm5", 1 << 20, &[96 << 30], 96 << 30));
        assert_eq!(lines.len(), 2, "{lines:?}");
        // The GPU cap is the plan's cap, not the layout capacity already less the reserve.
        assert!(lines[0].starts_with("rtx0: cap 98304MiB; weights 40960MiB"), "{}", lines[0]);
        assert!(lines[0].contains("reserve 1724MiB [context+modules 600MiB, family reserve 1024MiB, release padding 100MiB]"), "{}", lines[0]);
        // The same shared default minimum applies with Sparks.
        assert!(lines[0].contains("KV target 2097152 min 262144 max 2097152, pool 1048576 tokens / 20480MiB"), "{}", lines[0]);
        // A Spark's cap is its worker ceiling; its residual is a named component.
        assert!(lines[1].starts_with("spark0: cap 110592MiB; weights 92160MiB"), "{}", lines[1]);
        // Sparks carry no family reserve; the residual is their named component.
        assert!(lines[1].contains("reserve 1523MiB [Spark worker process resident 1523MiB]"), "{}", lines[1]);
        let overridden = MemoryPolicy { spark_reserve: Some(3 << 30), ..Default::default() }
            .scoped(|| layout_lines(&layout, "glm5", 1 << 20, &[96 << 30], 96 << 30));
        assert!(overridden[1].contains("(overridden)") && overridden[1].contains("SPARK_RESERVE 3072MiB"), "{}", overridden[1]);
    }
}
