//! Dense single-device allocation facts, independent of planner policy.
//!
//! The executor allocates exactly these owners: checkpoint weights, the native
//! step planes and their pinned ordered-upload staging, the GDN bank (conv,
//! state, replay and commit tables), the full-GQA KV arena, the exact retained
//! prefix marks, block-verify marks and the optional drafter. Values measured
//! nowhere are explicit estimates and never receipts.
use super::{DrafterBytes, NativeBuffers, NativeWeights, Qwen35Config, PAGE_ROWS, PREFILL_ROWS, WHOLE};
use crate::families::qwen::HybridAttention;
use crate::placement::{planner::*, ArchContext};
use cuteafd_core::memory_layout::Category;

/// The shared qwen scheduler's token selector and its device sampler tables.
#[derive(Debug, Clone, Copy)]
pub struct ServingFacts {
    pub capacity: u32,
    pub device_sampling: bool,
}

impl ServingFacts {
    fn allocations(self, vocab: usize) -> Result<Vec<Allocation>, Refusal> {
        serving::SelectorOwners {
            location: Location::coordinator(0), capacity: self.capacity as usize,
            vocab, device_sampling: self.device_sampling,
        }.allocations("selector", serving::ServingOwners::LIFETIME)
    }
}

#[derive(Clone)]
pub struct DenseFacts<'a> {
    pub cfg: &'a Qwen35Config,
    pub weights: &'a NativeWeights,
    pub target_arena: &'a crate::serving_capacity::WeightArenaLayout,
    pub buffers: &'a NativeBuffers,
    pub slots: usize,
    pub record_bytes: u64,
    pub drafter: Option<DrafterBytes>,
    pub blas_head: serving::BlasHead,
    pub serving: Option<ServingFacts>,
    pub host_prefix: serving::HostPrefixOwners,
    /// Diagnostic resets retain one ordered-upload buffer per GDN state region.
    pub verify_check: bool,
}

/// Offline forecast: the physical card class supplies the CUDA context and
/// module image; the logical cap only lowers the ceiling.
pub fn forecast_budgets(physical: &ArchContext, capacity: u64, module_bytes: u64, host: (u64, u64))
    -> Result<Vec<Budget>, Refusal> {
    let context = physical.context_bytes.checked_add(module_bytes)
        .ok_or(Refusal::Overflow("context/modules"))?;
    if host.0 == 0 {
        return Err(Refusal::InvalidFact("dense Qwen plan requires a host memory sample".into()));
    }
    Ok(vec![
        Budget {
            id: "coordinator.0".into(),
            locations: vec![Location::coordinator(0)],
            physical_bytes: if physical.total_bytes == 0 { 32 << 30 } else { physical.total_bytes },
            logical_bytes: capacity,
            baseline: Baseline::Forecast { occupied: context,
                source: if physical.total_bytes == 0 {
                    format!("{}; physical capacity assumes nominal 32 GiB (CUDA total unmeasured)", physical.source)
                } else { physical.source.into() },
                assumed: physical.source.contains("5090") },
            headroom: crate::placement::reserve::gpu("qwen3_5"),
        },
        Budget {
            id: "host".into(),
            locations: vec![Location::HOST, Location::PINNED],
            physical_bytes: host.0,
            logical_bytes: host.0,
            baseline: Baseline::Sampled { free: host.1, phase: Phase::Load,
                source: "host/cgroup available".into() },
            headroom: 8 << 30,
        },
    ])
}

/// A live sample fixes the physical witness; the logical cap keeps its ceiling.
pub fn sampled_budgets(free: u64, total: u64, logical: Option<u64>, host: (u64, u64))
    -> Result<Vec<Budget>, Refusal> {
    let physical = ArchContext::for_device("sm_120", total);
    let mut budgets = forecast_budgets(physical, logical.unwrap_or(total), 0, host)?;
    budgets[0].physical_bytes = total;
    budgets[0].baseline = Baseline::Sampled { free, phase: Phase::Load,
        source: format!("CUDA physical sample after modules, before target weights; {}", physical.source) };
    budgets[0].available()?;
    Ok(budgets)
}

pub fn admit(request: Request, budgets: Vec<Budget>, facts: DenseFacts<'_>)
    -> Result<ResolvedPlan, Refusal> {
    if matches!(request.onboard, Pin::Exact(layers) if layers > 0) {
        return Err(Refusal::InvalidPin("model has 0 routed-expert layers"));
    }
    if facts.weights.layers.len() != facts.cfg.attention.len()
        || facts.record_bytes == 0
        || facts.slots == 0
        || facts.target_arena.id() != "qwen3_5/target/gpu0"
        || facts.target_arena.payload_bytes() as u64 != facts.weights.layers.iter().sum::<u64>()
            + facts.weights.norm + facts.weights.head + facts.weights.embedding
        || facts.drafter.as_ref().is_some_and(|draft| draft.weights.0 > 0 && draft.weight_arena.is_none())
        || facts.buffers.head_rows != match request.operation {
            Operation::Serving => super::VERIFY_ROWS,
            Operation::Scoring => PREFILL_ROWS,
        }
    {
        return Err(Refusal::InvalidFact(
            "dense geometry or operation differs from allocation recipe".into()));
    }
    solve(request, budgets, &facts)
}

impl Recipe for DenseFacts<'_> {
    fn prepare(&self, _request: &Request) -> Result<Contract, Refusal> {
        // Full-context draft KV is one owner per ring; dense MTP and block
        // dSpark both index a 64-row page table.
        let token_capacity = if self.drafter.is_some() { i32::MAX as u64 } else { u32::MAX as u64 };
        let maximum = token_capacity / PAGE_ROWS;
        let mut contract = Contract::fixed(PAGE_ROWS, maximum, 1, self.owner_templates(1)?);
        let gpu = Location::coordinator(0);
        contract.components.extend(weight_arena_components(self.target_arena, gpu, Lifetime::RESIDENT, maximum, cuteafd_core::memory_layout::MemoryKind::Weights)?);
        let draft_start = contract.components.len();
        if let Some((id, bytes)) = self.drafter.as_ref().and_then(|draft| draft.weight_arena) {
            contract.components.extend(weight_arena_extent_components(id, bytes, gpu,
                Lifetime { first: Phase::DraftLoad, last: Phase::Growth }, maximum, cuteafd_core::memory_layout::MemoryKind::DrafterWeights)?);
        }
        contract.variables = self.cfg.attention.iter().enumerate()
            .map(|(layer, _)| Variable::Attention { layer, values: vec![WHOLE] })
            .collect();
        contract.variables.push(Variable::Prefill(vec![PrefillShape { rows: PREFILL_ROWS as u64, waves: 1 }]));
        contract.variables.push(Variable::Lanes(vec![1]));
        contract.variables.push(Variable::Embedding(vec![EmbeddingHome::Device(0)]));
        // Every full-attention layer and every draft ring is one cudaMalloc
        // sized to the whole admitted pool, so the page count stays unaligned
        // and each allocation prices its own SM120 backing round-up.
        let kv_page = PAGE_ROWS.checked_mul(self.record_bytes)
            .ok_or(Refusal::Overflow("dense KV page"))?;
        for (layer, kind) in self.cfg.attention.iter().enumerate() {
            if *kind == HybridAttention::Full {
                contract.set_byte_extent(&format!("kv.{layer}"), pool_curve(kv_page, maximum)?)?;
            }
        }
        if let Some(draft) = self.drafter.as_ref().filter(|draft| !draft.pools.is_empty()) {
            for (index, unit) in draft.pools.iter().enumerate() {
                contract.set_byte_extent(&format!("draft.kv.{index}"), pool_curve(*unit, maximum)?)?;
            }
        }
        if self.drafter.is_some() {
            let home = contract.variables.len();
            contract.variables.push(Variable::DraftHome(vec![0]));
            for (index, component) in contract.components.iter_mut().enumerate() {
                let draft = index >= draft_start || match &component.kind {
                    ComponentKind::Allocation { template, .. } => template.id.starts_with("draft."),
                    ComponentKind::ResidentWeight(owner) => owner.id.starts_with("draft."),
                    _ => false,
                };
                if draft { component.activation = Activation::choice(home, 0); }
            }
        }
        let target = serving::BlasOwners { name: "dense Qwen target vocabulary".into(),
            process: "coordinator".into(), thread: "qwen3_5/blas/serve".into(), location: gpu,
            gemm: self.blas_head.gemm(), lt: self.blas_head.lt(),
            lifetime: Lifetime::TARGET, activation: Activation::always() };
        let mut blas = vec![target.clone()];
        if self.drafter.as_ref().is_some_and(|draft| draft.gemm_head) {
            blas.push(serving::BlasOwners { name: "dense Qwen block dSpark vocabulary".into(),
                gemm: true, lt: false, lifetime: Lifetime { first: Phase::DraftLoad, last: Phase::Growth },
                ..target });
        }
        contract.components.extend(serving::blas_components("sm_120", &blas, maximum, contract.phase_order)?);
        // Every view-less device owner the executor allocates as its own
        // cudaMalloc prices its SM120 backing round-up here, once per lifetime
        // group. Pool-shaped KV owners price their own round-up in `pool_curve`.
        // Weight arenas carry their own single `.tail` granulator already.
        let fixed = contract.components.iter().filter_map(|component| match &component.kind {
            ComponentKind::Allocation { template, deferred: false, .. }
                if template.location == gpu && template.view.is_none()
                    && template.group != "weight arena" =>
                Some((template.id.clone(), template.lifetime, template.bytes)),
            _ => None,
        }).collect::<Vec<_>>();
        for (id, lifetime) in [("dense.runtime.tail", Lifetime::TARGET),
            ("draft.runtime.tail", Lifetime { first: Phase::DraftLoad, last: Phase::Growth }),
            ("serving.runtime.tail", serving::ServingOwners::LIFETIME)] {
            let backings = fixed.iter().filter(|(backing, owner, _)| *owner == lifetime
                    && !backing.starts_with("kv.") && !backing.starts_with("draft.kv"))
                .map(|(backing, _, bytes)| GranularityBacking { backing: backing.clone(),
                    // The draft workspace is one logical owner over the exact
                    // cudaMalloc list; every other owner is one request.
                    requests: match (backing.as_str(), self.drafter.as_ref()) {
                        ("draft.runtime", Some(draft)) if !draft.workspace_requests.is_empty() =>
                            draft.workspace_requests.clone(),
                        _ => vec![*bytes],
                    } })
                .collect::<Vec<_>>();
            if !backings.is_empty() {
                contract.components.push(allocation_granularity_component(AllocationGranularity {
                    id: id.into(), location: gpu, lifetime, backings,
                    evidence: GranularityEvidence::Forecast }, maximum)?);
            }
        }
        // The retained driver stream slab exists once a second live stream
        // does: the engine stream plus the two prefix host-cache copy streams.
        // It is not a cudaMalloc request, so its own tail is not charged.
        let peak = 1 + if self.host_prefix.quota_bytes > 0 { 2 } else { 0 };
        let slabs = cuda_stream_slab_bytes(peak)?;
        if slabs > 0 {
            contract.components.push(Component { activation: Activation::always(),
                bytes: ByteCurve::fixed(slabs, maximum),
                kind: ComponentKind::Allowance(Allowance { location: gpu, bytes: slabs,
                    lifetime: Lifetime::RESIDENT, assumed: false,
                    source: format!("dense Qwen driver stream slabs ({peak} peak live streams, \
                        SM120 driver/runtime13020 LAZY, #1511 2026-11-08)") }) });
        }
        // Last: every pinned backing (module staging, GDN staging, drafter
        // landing and ordered uploads, host prefix chunks) maps into the
        // coordinator context at one page table entry per 512 mapped bytes.
        contract.add_driver_page_tables();
        Ok(contract)
    }
}

/// One cudaMalloc-sized pool owner: pages stay unaligned and the backing's
/// physical round-up is the owner's own tail, never a pool-wide envelope.
fn pool_curve(unit: u64, maximum: u64) -> Result<ByteCurve, Refusal> {
    unit.checked_mul(maximum)
        .and_then(|bytes| bytes.div_ceil(SM120_ALLOCATION_GRANULARITY_BYTES)
            .checked_mul(SM120_ALLOCATION_GRANULARITY_BYTES))
        .ok_or(Refusal::Overflow("dense pool backing"))?;
    Ok(ByteCurve(vec![BytePiece { first: 1, last: maximum, bytes_at_first: unit,
        bytes_per_page: unit, granularity: SM120_ALLOCATION_GRANULARITY_BYTES, backings: 1 }]))
}

impl DenseFacts<'_> {
    /// Construct physical owners and typed views once; only their declared
    /// page-dependent extents change when the winning plan is materialized.
    fn owner_templates(&self, pages: u64) -> Result<Facts, Refusal> {
        let mut allocations = Vec::new();
        let resident_weights = Vec::new();
        let mut add = |id: String, location, bytes: u64, lifetime, category, group: &str| {
            if bytes > 0 {
                allocations.push(Allocation::buffer(id, location, bytes, lifetime, category, group));
            }
        };
        let gpu = Location::coordinator(0);
        // The dense loader keeps one reusable pinned staging buffer for the
        // mount copies and releases it when loading ends (STEP_BYTES bound).
        let step = crate::shared::dense_read::STEP_BYTES as u64;
        add("weights.upload".into(), Location::PINNED, step,
            Lifetime { first: Phase::Load, last: Phase::Load }, Category::Staging,
            "native operand mount staging");
        for (index, bytes) in self.buffers.planes.into_iter().enumerate() {
            add(format!("planes.{index}"), gpu, bytes as u64, Lifetime::TARGET,
                Category::Workspace, "native step planes and scratch");
            // copy_h2d_ordered retains one pinned buffer per destination, sized
            // to the destination, from its first ordered table upload onward.
            if index >= 7 {
                add(format!("planes.{index}.upload"), Location::PINNED, bytes as u64,
                    Lifetime::TARGET, Category::Staging, "ordered execution table staging");
            }
        }
        for (id, group, bytes, category) in [
            ("ids", "token ids", self.buffers.ids, Category::Workspace),
            ("logits", "logits", self.buffers.logits, Category::Workspace),
            ("head", "vocabulary head workspace", self.buffers.head, Category::Workspace),
            ("gdn.commit", "GDN commit table", self.buffers.commit, Category::Workspace),
            ("prefix", "exact retained prefix marks", self.buffers.prefix, Category::Prefix),
        ] {
            add(id.into(), gpu, bytes as u64, Lifetime::TARGET, category, group);
        }
        // GdnBank reserves its commit table's ordered staging before readiness.
        add("gdn.commit.upload".into(), Location::PINNED, self.buffers.commit as u64,
            Lifetime::TARGET, Category::Staging, "ordered GDN commit staging");
        for (index, bytes) in self.buffers.verify.into_iter().enumerate() {
            add(format!("verify.{index}"), gpu, bytes as u64, Lifetime::TARGET,
                if index == 0 { Category::Kv } else { Category::Workspace }, "block verify marks, logits and taps");
        }
        let layers = self.cfg.attention.iter().filter(|&&kind| kind == HybridAttention::Gdn).count() as u64;
        let conv_slot = ((self.cfg.gdn.conv_kernel - 1) * self.cfg.gdn.conv_width() * 2) as u64;
        let state_slot = self.buffers.state_per_layer as u64 - conv_slot;
        for (id, per_layer, group, category) in [
            ("gdn.conv", conv_slot.checked_mul(self.slots as u64), "GDN state", Category::Kv),
            ("gdn.state", state_slot.checked_mul(self.slots as u64), "GDN state", Category::Kv),
            ("gdn.replay", Some(self.buffers.replay_per_layer as u64), "GDN replay", Category::Workspace),
        ] {
            let bytes = per_layer.and_then(|n| n.checked_mul(layers))
                .ok_or(Refusal::Overflow("GDN backing"))?;
            if layers > 0 {
                add(id.into(), gpu, bytes.max(256), Lifetime::TARGET, category, group);
            }
        }
        if self.verify_check {
            for ordinal in 0..layers {
                for slot in 0..self.slots {
                    for (region, bytes) in [("conv", conv_slot), ("state", state_slot)] {
                        add(format!("gdn.{region}.{ordinal}.slot.{slot}.upload"), Location::PINNED,
                            bytes, Lifetime::TARGET, Category::Staging, "diagnostic GDN reset staging");
                    }
                }
            }
        }
        for (layer, kind) in self.cfg.attention.iter().enumerate() {
            if *kind == HybridAttention::Full {
                add(format!("kv.{layer}"), gpu, pages.checked_mul(PAGE_ROWS)
                    .and_then(|n| n.checked_mul(self.record_bytes))
                    .ok_or(Refusal::Overflow("dense KV"))?,
                    Lifetime::TARGET, Category::Kv, "full GQA KV");
            }
        }
        if let Some(draft) = self.drafter.as_ref() {
            let life = Lifetime { first: Phase::DraftLoad, last: Phase::Growth };
            let runtime = draft.workspace_requests.iter().try_fold(0u64, |sum, &bytes| sum.checked_add(bytes))
                .ok_or(Refusal::Overflow("draft workspace requests"))?;
            add("draft.prefetch".into(), Location::HOST, draft.host_prefetch,
                Lifetime { first: Phase::DraftLoad, last: Phase::DraftLoad }, Category::Staging,
                "drafter prefetched checkpoint tensors");
            add("draft.runtime".into(), gpu, runtime, life, Category::Drafter, "native drafter workspaces");
            add("draft.page-tables".into(), gpu, draft.page_table_bytes, life, Category::Drafter,
                "paged drafter page table");
            add("draft.scratch".into(), gpu, draft.scratch, life, Category::Drafter, "native drafter scratch");
            add("draft.landing".into(), Location::PINNED, draft.landing, life, Category::Staging,
                "drafter D2H landing row");
            for (id, bytes) in &draft.staging {
                add((*id).into(), Location::PINNED, *bytes, life, Category::Staging,
                    "ordered drafter table staging");
            }
            for (index, unit) in draft.pools.iter().enumerate() {
                add(format!("draft.kv.{index}"), gpu, pages.checked_mul(*unit)
                    .ok_or(Refusal::Overflow("draft KV ring"))?, life, Category::Drafter,
                    "drafter full-context KV ring");
            }
        }
        drop(add);
        let mut view = |id: String, backing: &str, offset, bytes, slot, dtype: &str, width| {
            let mut allocation = Allocation::buffer(id, gpu, bytes, Lifetime::TARGET, Category::Kv,
                "exact state view");
            allocation.view = Some(View { backing: backing.into(), offset });
            allocation.slot = Some(slot);
            allocation.dtype = dtype.into();
            allocation.shape = vec![bytes / width];
            allocations.push(allocation);
        };
        for ordinal in 0..layers {
            for slot in 0..self.slots as u64 {
                let record = ordinal * self.slots as u64 + slot;
                view(format!("gdn.conv.{ordinal}.slot.{slot}"), "gdn.conv", record * conv_slot,
                    conv_slot, slot as u32, "bf16", 2);
                view(format!("gdn.state.{ordinal}.slot.{slot}"), "gdn.state", record * state_slot,
                    state_slot, slot as u32, "f32", 4);
                let mark_offset = record * self.buffers.state_per_layer as u64;
                view(format!("verify.state.{ordinal}.slot.{slot}.conv"), "verify.0", mark_offset,
                    conv_slot, slot as u32, "bf16", 2);
                view(format!("verify.state.{ordinal}.slot.{slot}.recurrent"), "verify.0",
                    mark_offset + conv_slot, state_slot, slot as u32, "f32", 4);
            }
        }
        for mark in 0..self.buffers.marks as u64 {
            view(format!("prefix.mark.{mark}"), "prefix",
                mark * self.buffers.mark_bytes as u64, self.buffers.mark_bytes as u64,
                mark as u32, "bytes", 1);
        }
        if let Some(serving) = self.serving {
            allocations.extend(serving.allocations(self.cfg.vocab)?);
        }
        allocations.extend(self.host_prefix.allocations("host.prefix", serving::ServingOwners::LIFETIME)?);
        // The dense loader keeps one packed operand read in host memory at a
        // time; every device byte is a named owner above.
        let mut allowances = vec![
            Allowance { location: Location::HOST, bytes: step,
                lifetime: Lifetime { first: Phase::Load, last: Phase::Load },
                source: "estimated: one packed operand read and Vec capacity; read_exact_at, no shard buffer".into(),
                assumed: true },
        ];
        allowances.retain(|allowance| allowance.bytes > 0);
        Ok(Facts { allocation_granularity: vec![], driver_page_tables: vec![], resident_weights, phase_order: PhaseOrder::TARGET_THEN_DRAFT, allocations,
            deferred: vec![], sequential: vec![], reservations: vec![], allowances })
    }
}

#[cfg(test)]
mod tests;
