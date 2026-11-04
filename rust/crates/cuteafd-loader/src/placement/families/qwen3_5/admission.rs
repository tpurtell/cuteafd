//! Dense single-device allocation facts, independent of planner policy.
use super::{
    DenseDraft, NativeBuffers, NativeWeights, Qwen35Config, PAGE_ROWS, PREFILL_ROWS, WHOLE,
};
use crate::families::qwen::HybridAttention;
use crate::placement::{planner::*, ArchContext};
use cuteafd_core::memory_layout::Category;

#[derive(Debug, Clone, Copy)]
pub struct ServingRequest {
    pub concurrency: u64,
    pub prefix_entries: u64,
    /// None is automatic; Some(0) disables the host tier.
    pub host_cache_bytes: Option<u64>,
    pub host_headroom: u64,
    pub device_sampling: bool,
}

#[derive(Debug, Clone, Copy)]
pub struct ServingFacts {
    pub device_sampling: bool,
    pub host_quota: u64,
    pub host_chunk: u64,
    pub host_runtime: u64,
    pub host_headroom: u64,
}

impl ServingFacts {
    pub fn new(
        cfg: &Qwen35Config,
        buffers: &NativeBuffers,
        record_bytes: u64,
        draft_page_bytes: u64,
        request: ServingRequest,
        host_total: u64,
    ) -> Result<Self, Refusal> {
        if request.concurrency == 0 || request.concurrency > super::VERIFY_ROWS as u64 {
            return Err(Refusal::InvalidFact(
                "dense scheduler concurrency exceeds decode rows".into(),
            ));
        }
        let geometry = cuteafd_core::prefix::HostCacheGeometry {
            unit_rows: PAGE_ROWS,
            page_bytes: record_bytes
                .checked_mul(PAGE_ROWS)
                .and_then(|n| {
                    n.checked_mul(
                        cfg.attention
                            .iter()
                            .filter(|&&a| a == HybridAttention::Full)
                            .count() as u64,
                    )
                })
                .ok_or(Refusal::Overflow("host prefix page"))?,
            mark_bytes: buffers.mark_bytes as u64,
            draft_bytes: 0,
            draft_page_bytes,
        };
        let quota = if request.prefix_entries == 0 {
            0
        } else {
            geometry
                .resolve(
                    request.host_cache_bytes,
                    cfg.max_context as u64,
                    request.concurrency,
                    host_total,
                    geometry.chunk_bytes(),
                )
                .ok_or(Refusal::Overflow("host prefix quota"))?
        };
        if quota == 0 && request.prefix_entries > 0 && request.host_cache_bytes != Some(0) {
            return Err(Refusal::InvalidFact(
                "host prefix quota resolves to zero".into(),
            ));
        }
        Ok(Self {
            device_sampling: request.device_sampling,
            host_quota: quota,
            host_headroom: request.host_headroom,
            host_chunk: geometry.chunk_bytes().min(quota),
            host_runtime: cuteafd_core::memory_layout::host_logits_bytes(
                buffers.head_rows as u64,
                PREFILL_ROWS as u64,
                cfg.vocab as u64,
            )
            .ok_or(Refusal::Overflow("host logits"))?,
        })
    }

    fn apply_budgets(self, budgets: &mut [Budget]) -> Result<(), Refusal> {
        let host = budgets
            .iter_mut()
            .find(|budget| budget.id == "host")
            .ok_or_else(|| Refusal::InvalidFact("dense serving has no host budget".into()))?;
        host.headroom = host.headroom.max(self.host_headroom);
        Ok(())
    }

    fn allocations(self, vocab: usize) -> Result<Vec<Allocation>, Refusal> {
        serving::ServingOwners {
            selector: serving::SelectorOwners {
                location: Location::coordinator(0),
                capacity: super::VERIFY_ROWS,
                vocab,
                device_sampling: self.device_sampling,
            },
            prefix: serving::HostPrefixOwners {
                quota_bytes: self.host_quota,
                chunk_bytes: self.host_chunk,
            },
        }.allocations()
    }
}

#[derive(Clone, Copy)]
pub struct DenseFacts<'a> {
    pub cfg: &'a Qwen35Config,
    pub weights: &'a NativeWeights,
    pub buffers: &'a NativeBuffers,
    pub slots: usize,
    pub record_bytes: u64,
    pub drafter: Option<DenseDraft<'a>>,
    pub serving: Option<ServingFacts>,
}

pub fn forecast_budgets(
    physical: &ArchContext,
    capacity: u64,
    module_bytes: u64,
    host: (u64, u64),
) -> Result<Vec<Budget>, Refusal> {
    let context = physical
        .context_bytes
        .checked_add(module_bytes)
        .ok_or(Refusal::Overflow("context/modules"))?;
    Ok(vec![
        Budget {
            id: "coordinator.0".into(),
            locations: vec![Location::coordinator(0)],
            physical_bytes: if physical.total_bytes == 0 {
                32 << 30
            } else {
                physical.total_bytes
            },
            logical_bytes: capacity,
            baseline: Baseline::Forecast {
                occupied: context,
                source: if physical.total_bytes == 0 {
                    format!(
                        "{}; physical capacity assumes nominal 32 GiB (CUDA total unmeasured)",
                        physical.source
                    )
                } else {
                    physical.source.into()
                },
                assumed: physical.source.contains("5090"),
            },
            headroom: 1 << 30,
        },
        Budget {
            id: "host".into(),
            locations: vec![Location::HOST, Location::PINNED],
            physical_bytes: host.0,
            logical_bytes: host.0,
            baseline: Baseline::Sampled {
                free: host.1,
                phase: Phase::Load,
                source: "host/cgroup available".into(),
            },
            headroom: host.0 / 10,
        },
    ])
}

/// A live sample is physical; a logical cap changes only the ceiling.
pub fn sampled_budgets(
    free: u64,
    total: u64,
    logical: Option<u64>,
    host: (u64, u64),
) -> Result<Vec<Budget>, Refusal> {
    let physical = ArchContext::for_device("sm_120", total);
    let mut budgets = forecast_budgets(physical, logical.unwrap_or(total), 0, host)?;
    budgets[0].physical_bytes = total;
    budgets[0].baseline = Baseline::Sampled {
        free,
        phase: Phase::Load,
        source: format!(
            "CUDA physical sample after modules, before target weights; {}",
            physical.source
        ),
    };
    budgets[0].available()?;
    Ok(budgets)
}

pub fn admit(
    request: Request,
    mut budgets: Vec<Budget>,
    facts: DenseFacts<'_>,
) -> Result<ResolvedPlan, Refusal> {
    if matches!(request.onboard, Pin::Exact(layers) if layers > 0) {
        return Err(Refusal::InvalidPin("model has 0 routed-expert layers"));
    }
    if facts.weights.layers.len() != facts.cfg.attention.len()
        || facts.record_bytes == 0
        || facts.slots == 0
        || facts.buffers.head_rows
            != match request.operation {
                Operation::Serving => super::VERIFY_ROWS,
                Operation::Scoring => PREFILL_ROWS,
            }
    {
        return Err(Refusal::InvalidFact(
            "dense geometry or operation differs from allocation recipe".into(),
        ));
    }
    for (layer, &expected) in facts.weights.layers.iter().enumerate() {
        let prefix = format!("layer.{layer}.");
        let actual = facts
            .weights
            .allocations
            .iter()
            .filter(|a| a.id.starts_with(&prefix))
            .try_fold(0u64, |total, a| {
                total
                    .checked_add(a.bytes)
                    .ok_or(Refusal::Overflow("dense layer weights"))
            })?;
        if actual == 0 || actual != expected {
            return Err(Refusal::InvalidFact(format!(
                "dense layer {layer} inventory differs from physical operands"
            )));
        }
    }
    if let Some(serving) = facts.serving {
        serving.apply_budgets(&mut budgets)?;
    }
    solve(request, budgets, &facts)
}

impl Recipe for DenseFacts<'_> {
    fn prepare(&self, _request: &Request) -> Result<Contract, Refusal> {
        let token_capacity = match self.drafter {
            Some(DenseDraft::Block(_)) => crate::formats::block_dspark::BlockDraftLayout::MAX_POOL_TOKENS as u64,
            _ => u32::MAX as u64,
        };
        let maximum = token_capacity / PAGE_ROWS;
        let mut contract = Contract::fixed(PAGE_ROWS, maximum, 1, self.owner_templates(PAGE_ROWS)?);
        contract.variables = self
            .cfg
            .attention
            .iter()
            .enumerate()
            .map(|(layer, _)| Variable::Attention {
                layer,
                values: vec![WHOLE],
            })
            .collect();
        contract
            .variables
            .push(Variable::Prefill(vec![PrefillShape {
                rows: PREFILL_ROWS as u64,
                waves: 1,
            }]));
        contract.variables.push(Variable::Lanes(vec![1]));
        contract.variables.push(Variable::Embedding(vec![EmbeddingHome::Device(0)]));
        let kv_page = PAGE_ROWS
            .checked_mul(self.record_bytes)
            .ok_or(Refusal::Overflow("dense KV page"))?;
        for (layer, kind) in self.cfg.attention.iter().enumerate() {
            if *kind == HybridAttention::Full {
                contract.set_byte_extent(
                    &format!("kv.{layer}"),
                    ByteCurve::linear(kv_page, kv_page, maximum),
                )?;
            }
        }
        match self.drafter {
            Some(DenseDraft::NativeMtp(layout)) => {
                let page = layout
                    .pool_unit_bytes()
                    .map_err(|e| Refusal::InvalidFact(e.to_string()))?;
                let one_page = layout
                    .with_pool_tokens(PAGE_ROWS as usize)
                    .map_err(|e| Refusal::InvalidFact(e.to_string()))?;
                let sidecar = one_page
                    .owners
                    .iter()
                    .find(|o| o.id == "draft.mtp.sidecars")
                    .ok_or_else(|| Refusal::InvalidFact("MTP sidecar owner missing".into()))?;
                let kv = page
                    .checked_sub(sidecar.bytes as u64)
                    .ok_or(Refusal::Overflow("MTP KV page"))?;
                contract.set_byte_extent("draft.mtp.kv", ByteCurve::linear(kv, kv, maximum))?;
                let component = contract.components.iter_mut().find(|c| matches!(&c.kind,
                    ComponentKind::Allocation { template, .. } if template.id == "draft.mtp.sidecars"))
                    .ok_or_else(|| Refusal::InvalidFact("MTP sidecar component missing".into()))?;
                component.bytes =
                    ByteCurve::linear(sidecar.bytes as u64, sidecar.bytes as u64, maximum);
                if let ComponentKind::Allocation { shape, .. } = &mut component.kind {
                    shape[0] = ByteCurve::linear(1, 1, maximum);
                }
            }
            Some(DenseDraft::Block(layout)) => {
                let one_page = layout.with_pool_tokens(PAGE_ROWS as usize)
                    .map_err(|e| Refusal::InvalidFact(e.to_string()))?;
                if layout.cache.full_layers > 0 {
                    for owner in one_page.owners.iter().filter(|o| o.id.ends_with(".k_ring") || o.id.ends_with(".v_ring")) {
                        contract.set_byte_extent(&format!("draft.block.{}", owner.id),
                            ByteCurve::linear(owner.bytes as u64, owner.bytes as u64, maximum))?;
                    }
                }
            }
            None => {}
        }
        Ok(contract)
    }
}

impl DenseFacts<'_> {
    /// Construct physical owners and typed views once; only their declared
    /// page-dependent extents are changed when the winning plan is materialized.
    fn owner_templates(&self, pool: u64) -> Result<Facts, Refusal> {
        let mut allocations = Vec::new();
        let mut resident_weights = Vec::new();
        let mut add = |id: String, location, bytes, lifetime, category, group: &str| {
            if bytes > 0 {
                allocations.push(Allocation::buffer(
                    id, location, bytes, lifetime, category, group,
                ));
            }
        };
        let gpu = Location::coordinator(0);
        for operand in &self.weights.allocations {
            resident_weights.push(ResidentWeight { id: format!("weights.{}", operand.id), location: gpu,
                weights: crate::placement::ResidentWeights(operand.bytes) });
            add(
                format!("weights.{}", operand.id),
                gpu,
                operand.bytes,
                Lifetime::RESIDENT,
                Category::Weights,
                "native dense weights",
            );
        }
        resident_weights.push(ResidentWeight { id: "weights.embedding".into(), location: gpu,
            weights: crate::placement::ResidentWeights(self.weights.embedding) });
        add(
            "weights.embedding".into(),
            gpu,
            self.weights.embedding,
            Lifetime::RESIDENT,
            Category::Embedding,
            "token embedding",
        );
        // Embedding loading overlaps native weight loading, but ends before KV.
        add(
            "embedding.upload".into(),
            Location::PINNED,
            64 << 20,
            Lifetime {
                first: Phase::Load,
                last: Phase::Load,
            },
            Category::Staging,
            "embedding upload staging",
        );
        let staging = self
            .weights
            .allocations
            .iter()
            .map(|a| a.upload_bytes)
            .max()
            .unwrap_or(0)
            .max(self.drafter.map_or(0, DenseDraft::sync_upload_bytes));
        add(
            "upload.sync".into(),
            Location::PINNED,
            staging,
            Lifetime::RESIDENT,
            Category::Staging,
            "retained sync-H2D staging",
        );
        for (index, bytes) in self.buffers.planes.into_iter().enumerate() {
            add(
                format!("planes.{index}"),
                gpu,
                bytes as u64,
                Lifetime::TARGET,
                Category::Workspace,
                "native step planes and scratch",
            );
            if index >= 7 {
                add(
                    format!("planes.{index}.upload"),
                    Location::PINNED,
                    bytes as u64,
                    Lifetime::TARGET,
                    Category::Staging,
                    "ordered execution table staging",
                );
            }
        }
        add(
            "ids.upload".into(),
            Location::PINNED,
            self.buffers.ids as u64,
            Lifetime::TARGET,
            Category::Staging,
            "ordered token id staging",
        );
        for (id, group, bytes, category) in [
            ("ids", "token ids", self.buffers.ids, Category::Workspace),
            ("logits", "logits", self.buffers.logits, Category::Workspace),
            (
                "head",
                "vocabulary head workspace",
                self.buffers.head,
                Category::Workspace,
            ),
            (
                "gdn.commit",
                "GDN commit table",
                self.buffers.commit,
                Category::Workspace,
            ),
            (
                "prefix",
                "exact retained prefix marks",
                self.buffers.prefix,
                Category::Prefix,
            ),
        ] {
            add(
                id.into(),
                gpu,
                bytes as u64,
                Lifetime::TARGET,
                category,
                group,
            );
        }
        add(
            "gdn.commit.upload".into(),
            Location::PINNED,
            self.buffers.commit as u64,
            Lifetime::TARGET,
            Category::Staging,
            "ordered GDN commit staging",
        );
        for (index, bytes) in self.buffers.verify.into_iter().enumerate() {
            add(
                format!("verify.{index}"),
                gpu,
                bytes as u64,
                Lifetime::TARGET,
                Category::Workspace,
                "block verify marks, logits and taps",
            );
        }
        let layers = self
            .cfg
            .attention
            .iter()
            .filter(|&&kind| kind == HybridAttention::Gdn)
            .count() as u64;
        let conv_slot = ((self.cfg.gdn.conv_kernel - 1) * self.cfg.gdn.conv_width() * 2) as u64;
        let state_slot = self.buffers.state_per_layer as u64 - conv_slot;
        for (id, per_layer, group) in [
            (
                "gdn.conv",
                conv_slot.checked_mul(self.slots as u64),
                "GDN state",
            ),
            (
                "gdn.state",
                state_slot.checked_mul(self.slots as u64),
                "GDN state",
            ),
            (
                "gdn.replay",
                Some(self.buffers.replay_per_layer as u64),
                "GDN replay",
            ),
        ] {
            let bytes = per_layer
                .and_then(|n| n.checked_mul(layers))
                .ok_or(Refusal::Overflow("GDN backing"))?;
            if layers > 0 {
                add(
                    id.into(),
                    gpu,
                    bytes.max(256),
                    Lifetime::TARGET,
                    if id == "gdn.replay" {
                        Category::Workspace
                    } else {
                        Category::Kv
                    },
                    group,
                );
            }
        }
        for (layer, kind) in self.cfg.attention.iter().enumerate() {
            if *kind == HybridAttention::Full {
                add(
                    format!("kv.{layer}"),
                    gpu,
                    pool.checked_mul(self.record_bytes)
                        .ok_or(Refusal::Overflow("dense KV"))?,
                    Lifetime::TARGET,
                    Category::Kv,
                    "full GQA KV",
                );
            }
        }
        drop(add);
        let mut view = |id: String, backing: &str, offset, bytes, slot, dtype: &str, width| {
            let mut a = Allocation::buffer(
                id,
                gpu,
                bytes,
                Lifetime::TARGET,
                Category::Kv,
                "exact state view",
            );
            a.view = Some(View {
                backing: backing.into(),
                offset,
            });
            a.slot = Some(slot);
            a.dtype = dtype.into();
            a.shape = vec![bytes / width];
            allocations.push(a);
        };
        for ordinal in 0..layers {
            for slot in 0..self.slots as u64 {
                let record = ordinal * self.slots as u64 + slot;
                view(
                    format!("gdn.conv.{ordinal}.slot.{slot}"),
                    "gdn.conv",
                    record * conv_slot,
                    conv_slot,
                    slot as u32,
                    "bf16",
                    2,
                );
                view(
                    format!("gdn.state.{ordinal}.slot.{slot}"),
                    "gdn.state",
                    record * state_slot,
                    state_slot,
                    slot as u32,
                    "f32",
                    4,
                );
                let mark_offset = record * self.buffers.state_per_layer as u64;
                view(
                    format!("verify.state.{ordinal}.slot.{slot}.conv"),
                    "verify.0",
                    mark_offset,
                    conv_slot,
                    slot as u32,
                    "bf16",
                    2,
                );
                view(
                    format!("verify.state.{ordinal}.slot.{slot}.recurrent"),
                    "verify.0",
                    mark_offset + conv_slot,
                    state_slot,
                    slot as u32,
                    "f32",
                    4,
                );
            }
        }
        for mark in 0..self.buffers.marks as u64 {
            view(
                format!("prefix.mark.{mark}"),
                "prefix",
                mark * self.buffers.mark_bytes as u64,
                self.buffers.mark_bytes as u64,
                mark as u32,
                "bytes",
                1,
            );
        }
        if let Some(serving) = self.serving {
            allocations.extend(serving.allocations(self.cfg.vocab)?);
        }
        let mut sequential = Vec::new();
        if let Some(DenseDraft::NativeMtp(layout)) = self.drafter {
            use crate::formats::dflash::{MlaDraftDomain, MlaDraftDtype};
            let layout = layout
                .with_pool_tokens(usize::try_from(pool).map_err(|_| Refusal::Overflow("MTP pool"))?)
                .map_err(|e| Refusal::InvalidFact(e.to_string()))?;
            for owner in layout.owners {
                let mut allocation = Allocation::buffer(
                    owner.id,
                    match owner.domain {
                        MlaDraftDomain::Device => gpu,
                        MlaDraftDomain::HostPinned => Location::PINNED,
                    },
                    owner.bytes as u64,
                    Lifetime {
                        first: Phase::DraftLoad,
                        last: Phase::Growth,
                    },
                    Category::Drafter,
                    "native MTP owner",
                );
                allocation.alignment = owner.alignment as u64;
                allocation.dtype = match owner.dtype {
                    MlaDraftDtype::Bytes => "bytes",
                    MlaDraftDtype::Bf16 => "bf16",
                    MlaDraftDtype::F32 => "f32",
                    MlaDraftDtype::I32 => "i32",
                    MlaDraftDtype::U32 => "u32",
                    MlaDraftDtype::I64 => "i64",
                }
                .into();
                allocation.shape = owner.shape.into_iter().map(|n| n as u64).collect();
                if owner.resident_weights.0 > 0 {
                    resident_weights.push(ResidentWeight::from_allocation(&allocation, owner.resident_weights));
                }
                allocations.push(allocation);
            }
        }
        if let Some(DenseDraft::Block(layout)) = self.drafter {
            let layout = layout.with_pool_tokens(usize::try_from(pool).map_err(|_| Refusal::Overflow("block pool"))?)
                .map_err(|e| Refusal::InvalidFact(e.to_string()))?;
            let life = Lifetime { first: Phase::DraftLoad, last: Phase::Growth };
            allocations.extend(layout.owners.iter().filter(|o| !o.on_use).map(|owner| {
                let allocation = owner.allocation(format!("draft.block.{}", owner.id), 0, life, "block dSpark owner");
                if owner.resident_weights.0 > 0 {
                    resident_weights.push(ResidentWeight::from_allocation(&allocation, owner.resident_weights));
                }
                allocation
            }));
            if !layout.packing_sources.is_empty() {
                sequential.push(Sequential { id: "draft.block.packing".into(),
                    members: layout.packing_sources.iter().map(|owner| owner.allocation(
                        format!("draft.block.{}", owner.id), 0,
                        Lifetime { first: Phase::DraftLoad, last: Phase::DraftLoad }, "BF16 packing source")).collect() });
            }
        }
        // This was an ownerless Exact demand. It remains an explicit unmeasured
        // growth allowance, never a receipt or a physical-card measurement.
        let mut allowances = vec![Allowance { location: gpu, bytes: 512 << 20, lifetime: Lifetime::TARGET,
            source: "dense head/runtime growth: unqualified pre-existing allowance".into(), assumed: true },
            Allowance { location: Location::HOST, bytes: self.weights.loading_host_bytes,
                lifetime: Lifetime { first: Phase::Load, last: Phase::Load },
                source: "checkpoint headers: one tensor/operand read plus packed Vec capacity; read_exact_at, no shard buffer".into(),
                assumed: true }];
        if let Some(DenseDraft::NativeMtp(layout)) = self.drafter {
            allowances.push(Allowance { location: Location::HOST, bytes: layout.loading_host_bytes,
                lifetime: Lifetime { first: Phase::DraftLoad, last: Phase::DraftLoad },
                source: "native MTP headers: serialized read plus packed Vec capacity; ordinary Rust heap".into(), assumed: true });
        }
        if let Some(DenseDraft::Block(layout)) = self.drafter {
            allowances.push(Allowance { location: Location::HOST, bytes: layout.loading_host_bytes,
                lifetime: Lifetime { first: Phase::DraftLoad, last: Phase::DraftLoad },
                source: "block dSpark headers: all prefetched tensor payloads plus retained mask; ordinary Rust heap".into(),
                assumed: true });
        }
        if let Some(serving) = self.serving {
            allowances.push(Allowance { location: Location::HOST, bytes: serving.host_runtime,
                lifetime: Lifetime { first: Phase::Warmup, last: Phase::Growth },
                source: "bounded host logit download/gather allowance; ordinary Rust heap, not exact receipts".into(),
                assumed: true });
        }
        allowances.retain(|allowance| allowance.bytes > 0);
        Ok(Facts { resident_weights,
            phase_order: crate::placement::planner::PhaseOrder::TARGET_THEN_DRAFT,
            allocations,
            deferred: vec![],
            sequential,
            reservations: vec![],
            allowances,
        })
    }
}

#[cfg(test)]
mod tests;
