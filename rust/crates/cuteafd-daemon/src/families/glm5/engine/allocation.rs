//! Independent retained-field census for target, hook and exchange subrecipes.
//! This is not a complete family census and cannot by itself authorize Ready.
use super::*;
use cuteafd_core::memory_layout::Category;
use cuteafd_loader::placement::planner::{Allocation, CoordinatorDevices, Lifetime, Location, Phase, Receipt, View};
use std::collections::BTreeMap;

pub(super) struct Census<'a> {
    pub(super) prefix: String,
    pub(super) receipts: Vec<Receipt>,
    pub(super) devices: CoordinatorDevices<'a>,
    pub(super) rank: u8,
}
impl Census<'_> {
    fn buffer(
        &mut self,
        id: &str,
        buffer: cuteafd_ffi::CuteafdDeviceBuffer,
        lifetime: Lifetime,
        category: Category,
    ) -> Result<()> {
        self.record(id, buffer, lifetime, category, "GLM target buffers")
    }
    pub(super) fn record(
        &mut self,
        id: &str,
        buffer: cuteafd_ffi::CuteafdDeviceBuffer,
        lifetime: Lifetime,
        category: Category,
        source: &str,
    ) -> Result<()> {
        ensure!(!buffer.ptr.is_null() && buffer.bytes > 0, "GLM owner {id} is empty or null");
        let rank = self.devices.logical_rank(buffer.device_id)?;
        ensure!(rank == self.rank, "GLM buffer belongs to the wrong logical rank");
        self.receipts.push(Receipt {
            allocation: Allocation::buffer(
                format!("{}.{id}", self.prefix),
                self.devices.location(buffer.device_id)?,
                buffer.bytes as u64,
                lifetime,
                category,
                source,
            ),
            address: buffer.ptr as u64,
        });
        Ok(())
    }
    fn projection(
        &mut self,
        kind: &str,
        scratch: &crate::shared::dense_trellis::ProjectionScratch<'_>,
        lifetime: Lifetime,
        category: Category,
    ) -> Result<()> {
        let mut inventory = crate::shared::memory::receipts::Inventory::new(
            self.devices, &self.prefix, lifetime, category, "target hooks");
        scratch.allocation_receipts(&mut inventory, kind)?;
        ensure!(inventory.receipts.iter().all(|receipt|
            receipt.allocation.location == Location::coordinator(self.rank)),
            "GLM projection belongs to the wrong logical rank");
        self.receipts.extend(inventory.receipts);
        Ok(())
    }
    /// Decode owners exist from the startup warm pass; prefill lane owners
    /// exist from the first prefill, which only scoring runs before Ready.
    fn workspace_lifetime(kind: &str, scoring: bool) -> Lifetime {
        Lifetime {
            first: if kind == "prefill" && !scoring {
                Phase::Growth
            } else {
                Phase::Warmup
            },
            last: Phase::Growth,
        }
    }
    fn hooks(&mut self, kind: &str, slots: &[Workspace<'_>], scoring: bool) -> Result<()> {
        let lifetime = Self::workspace_lifetime(kind, scoring);
        let mut shared: Option<Vec<Receipt>> = None;
        for w in slots {
            let mut actual = Census {
                prefix: self.prefix.clone(),
                receipts: Vec::new(),
                devices: self.devices, rank: self.rank,
            };
            if let Some(native) = &w.native_dense {
                for (name, buffer) in native.allocation_buffers() {
                    actual.record(
                        &format!("{kind}.native.{name}"),
                        buffer,
                        lifetime,
                        Category::Workspace,
                        "GLM target hooks",
                    )?;
                }
            }
            if let Some(core) = &w.core {
                actual.projection(
                    &format!("{kind}.core.projection"),
                    &core.projection,
                    lifetime,
                    Category::Workspace,
                )?;
                if let Some(native) = &core.native {
                    actual.record(&format!("{kind}.core.native.scratch"), native.buffer, lifetime,
                        Category::Workspace, "GLM target hooks")?;
                }
                for (name, buffer) in core.allocation_stages() {
                    actual.record(
                        &format!("{kind}.core.stage.{name}"),
                        buffer,
                        lifetime,
                        Category::Workspace,
                        "GLM target hooks",
                    )?;
                }
            }
            if let Some(native) = &w.native_shared {
                let physical = self.devices.physical_device(self.rank)?;
                actual.receipts.extend(native.workspace_receipts(
                    &format!("{}.{kind}.native.shared", self.prefix),
                    self.devices.location(physical)?, physical, lifetime)?);
            }
            if let Some(previous) = &shared {
                ensure!(
                    previous.len() == actual.receipts.len()
                        && previous
                            .iter()
                            .zip(&actual.receipts)
                            .all(|(a, b)| a.allocation == b.allocation && a.address == b.address),
                    "GLM hook owners differ between retained slots"
                );
            } else {
                shared = Some(actual.receipts);
            }
        }
        self.receipts.extend(shared.into_iter().flatten());
        Ok(())
    }
    fn rows(
        &mut self,
        kind: &str,
        slots: &[Workspace<'_>],
        decode: bool,
        library: &NativeLibrary,
        stream: *mut c_void,
        scoring: bool,
    ) -> Result<()> {
        let lifetime = Self::workspace_lifetime(kind, scoring);
        let mut owners = BTreeMap::new();
        let mut router = None;
        let mut uploads = BTreeMap::new();
        for (slot, w) in slots.iter().enumerate() {
            ensure!(w.lane == slot, "GLM retained slot ordinal changed");
            if let Some(host) = w.router_host.try_borrow()?.as_ref() {
                let actual = (host.buffer.ptr as usize, host.buffer.bytes);
                ensure!(router.is_none_or(|previous| previous == actual),
                    "GLM router staging changed backing between slots");
                if router.replace(actual).is_none() {
                    self.receipts.push(Receipt {
                        allocation: Allocation::buffer(format!("{}.{kind}.router-host", self.prefix),
                            Location::PINNED, host.buffer.bytes as u64, lifetime,
                            Category::Staging, "GLM target buffers"),
                        address: host.buffer.ptr as u64,
                    });
                }
            }
            let mut rows = vec![
                ("h", &w.h),
                ("x", &w.x),
                ("query", &w.query),
                ("q_resid", &w.q_resid),
                ("q_fp8", &w.q_fp8),
                ("head_weights", &w.head_weights),
                ("indices", &w.indices),
                ("lengths", &w.lengths),
                ("attn", &w.attn),
                ("delta", &w.delta),
                ("positions", &w.positions),
                ("slots", &w.slots),
                ("cache_lengths", &w.cache_lengths),
                ("router_logits", &w.router_logits),
                ("route_ids", &w.route_ids),
                ("route_weights", &w.route_weights),
                ("wire", &w.wire),
                ("shared", &w.shared),
                ("ids", &w.ids),
                ("select", &w.select),
            ];
            if let Some(payload) = &w.payload {
                rows.push(("payload", payload));
            }
            if let Some(core) = &w.core_rows {
                rows.extend([
                    ("core.attention", &core[0]),
                    ("core.down", &core[1]),
                    ("core.reduction", &core[2]),
                ]);
            }
            for (field, buf) in rows {
                let id = format!("{kind}.rows.{field}");
                let owner = buf._owner.buffer;
                if let Some(previous) = owners.insert(id.clone(), owner) {
                    ensure!(
                        previous.ptr == owner.ptr
                            && previous.bytes == owner.bytes
                            && previous.device_id == owner.device_id,
                        "GLM row field {id} changed backing between slots"
                    );
                } else {
                    self.buffer(&id, owner, lifetime, Category::Workspace)?;
                }
                if !decode {
                    let offset = (buf.buffer.ptr as u64)
                        .checked_sub(owner.ptr as u64)
                        .context("GLM slot before its retained backing")?;
                    ensure!(
                        offset
                            .checked_add(buf.buffer.bytes as u64)
                            .is_some_and(|end| end <= owner.bytes as u64),
                        "GLM slot past its retained backing"
                    );
                    self.buffer(
                        &format!("{kind}.slot.{slot}.{field}"),
                        buf.buffer,
                        lifetime,
                        Category::Workspace,
                    )?;
                    let spec = &mut self
                        .receipts
                        .last_mut()
                        .context("GLM slot receipt")?
                        .allocation;
                    spec.view = Some(View {
                        backing: format!("{}.{id}", self.prefix),
                        offset,
                    });
                    // A cloned zero-width placeholder is shared, not independent slot storage.
                    spec.slot = if slots.len() == 1 || buf.buffer.bytes == owner.bytes {
                        None
                    } else {
                        Some(u32::try_from(slot).context("GLM slot ordinal")?)
                    };
                } else {
                    ensure!(
                        buf.buffer.ptr == owner.ptr && buf.buffer.bytes == owner.bytes,
                        "GLM decode row is not its whole retained owner"
                    );
                }
            }
            for (field, buf) in [
                ("logits", &w.logits),
                ("scratch", &w.scratch),
                ("topk", &w.topk_scratch),
            ]
            .into_iter()
            .chain(w._head_workspace.as_ref().map(|b| ("head", b)))
            {
                let id = format!("{kind}.{field}");
                let buffer = buf._owner.buffer;
                ensure!(
                    field == "logits"
                        || (buffer.ptr == buf.buffer.ptr
                            && buffer.bytes == buf.buffer.bytes
                            && buffer.device_id == buf.buffer.device_id),
                    "GLM hook field {id} is not its whole owner"
                );
                if let Some(previous) = owners.insert(id.clone(), buffer) {
                    ensure!(
                        previous.ptr == buffer.ptr
                            && previous.bytes == buffer.bytes
                            && previous.device_id == buffer.device_id,
                        "GLM hook field {id} changed owner"
                    );
                } else {
                    self.buffer(&id, buffer, lifetime, Category::Workspace)?;
                }
                if !decode && field == "logits" {
                    self.buffer(
                        &format!("{kind}.slot.{slot}.logits"),
                        buf.buffer,
                        lifetime,
                        Category::Workspace,
                    )?;
                    self.receipts
                        .last_mut()
                        .context("GLM logits receipt")?
                        .allocation
                        .view = Some(View {
                        backing: format!("{}.{id}", self.prefix),
                        offset: (buf.buffer.ptr as u64)
                            .checked_sub(buffer.ptr as u64)
                            .context("GLM logits view before backing")?,
                    });
                }
            }
            ensure!(
                w.page_table.buffer.ptr == w.page_table._owner.buffer.ptr
                    && w.page_table.buffer.bytes == w.page_table._owner.buffer.bytes
                    && w.page_table.buffer.device_id == w.page_table._owner.buffer.device_id,
                "GLM page table is not its whole retained owner"
            );
            self.buffer(
                &format!("{kind}.page_table.{slot}"),
                w.page_table._owner.buffer,
                lifetime,
                Category::Workspace,
            )?;
            for (field, buf) in [
                ("positions", &w.positions),
                ("slots", &w.slots),
                ("page_table", &w.page_table),
                ("cache_lengths", &w.cache_lengths),
                ("lengths", &w.lengths),
                ("ids", &w.ids),
            ] {
                uploads
                    .entry((buf.buffer.device_id, buf.buffer.ptr as usize))
                    .or_insert((format!("{kind}.upload.{slot}.{field}"), buf.buffer));
            }
        }
        // Query the stores that actually exist; an unprimed destination is absent,
        // not a receipt synthesized from the forecast's expected staging size.
        for (_, (id, dst)) in uploads {
            let Some(actual) = library.ordered_h2d_staging_buffer(dst, stream)? else { continue };
            self.receipts.push(Receipt {
                allocation: Allocation::buffer(
                    format!("{}.{id}", self.prefix),
                    Location::PINNED,
                    actual.bytes as u64,
                    lifetime,
                    Category::Staging,
                    "GLM target buffers",
                ),
                address: actual.ptr as u64,
            });
        }
        Ok(())
    }
}

impl GlmEngine<'_> {
    /// Passive startup snapshot of the retained serving roots. This does not
    /// establish observed lifetimes or replace measured runtime/RSS deltas.
    pub(crate) fn allocation_census(&self,
        prefix: &super::super::prefix::GlmPrefix<'_, '_>,
        cache: &cuteafd_engine::prefix::PrefixCache<crate::shared::prefix::CudaCopyEngine<'_>>,
        selector: &crate::shared::token_io::TokenSelector<'_>,
        decode_link: Option<&crate::shared::spark_intake::SparkLink<'_>>)
        -> Result<crate::shared::memory::receipts::Census> {
        let physical = self.census_physical_devices()?;
        let devices = CoordinatorDevices::new(&physical)?;
        let mut census = crate::shared::memory::receipts::Census::default();
        for receipts in [self.target_buffer_receipts()?, self.target_hook_receipts()?,
            self.weight_receipts()?, self.expert_receipts()?, self.exchange_receipts()?,
            self.serving_probe_receipts()?, prefix.allocation_receipts(devices)?] {
            census.owners(receipts);
        }
        self.embedding.allocation_aliases(&mut census, devices, "glm.embedding")?;
        if let Some(drafter) = &self.drafter {
            census.owners(drafter.allocation_receipts(devices, "glm.draft")?);
        }
        let mut inventory = crate::shared::memory::receipts::Inventory::new(devices, "glm.serving",
            Lifetime { first: Phase::Warmup, last: Phase::Growth }, Category::Workspace, "serving owners");
        selector.allocation_receipts(&mut inventory, "selector")?;
        if let Some(copy) = cache.host_engine() {
            copy.allocation_receipts(&mut inventory, "host_cache")?;
        }
        if let Some(skip) = &self.skip { skip.allocation_receipts(&mut inventory, "skip_intake")?; }
        census.owners(inventory.receipts);
        if let Some(upload) = self.warmup_upload_receipt()? { census.owners(vec![upload]); }
        if let Some(link) = decode_link { census.extend(link.allocation_census(devices, "glm.spark.decode")?); }
        if let Some(link) = self.prefill_link.try_borrow()?.as_ref() {
            census.extend(link.allocation_census(devices, "glm.spark.prefill")?);
        }
        for rank in 0..self.ranks() {
            census.unclassified.push(cuteafd_transport::UnclassifiedTransportOwner {
                owner: format!("glm.runtime.rank.{rank}"),
                reason: "CUDA context/module/graph/stream/event storage requires owner-labelled startup deltas".into(),
            });
        }
        census.unclassified.push(cuteafd_transport::UnclassifiedTransportOwner {
            owner: "host-heap (unattributed)".into(),
            reason: "process RSS phase observations, excluding named pinned and mapped-file residency, are missing".into(),
        });
        census.normalize(cuteafd_loader::placement::planner::PhaseOrder::DRAFT_THEN_TARGET)?;
        Ok(census)
    }

    pub(crate) fn expert_receipts(&self) -> Result<Vec<Receipt>> {
        let physical = self.census_physical_devices()?;
        let devices = CoordinatorDevices::new(&physical)?;
        let mut inventory = crate::shared::memory::receipts::Inventory::new(devices, "glm.expert",
            Lifetime::TARGET, Category::Experts, "resident expert owners");
        for (rank, shared) in self.native_experts.iter().enumerate() {
            shared.allocation_receipts(&mut inventory, &format!("shared.rank.{rank}"))?;
        }
        if let Some(pair) = &self.tp2 { pair.allocation_receipts(&mut inventory, "routed")?; }
        if let Some(whole) = &self.whole { whole.allocation_receipts(&mut inventory, "routed.rank.0")?; }
        super::resident_expert_weights(&mut inventory.receipts)?;
        Ok(inventory.receipts)
    }

    pub(crate) fn weight_receipts(&self) -> Result<Vec<Receipt>> {
        let physical = self.census_physical_devices()?;
        let devices = CoordinatorDevices::new(&physical)?;
        let mut inventory = crate::shared::memory::receipts::Inventory::new(devices, "glm.weight",
            Lifetime::RESIDENT, Category::Weights, "target checkpoint owners");
        for rank in 0..self.ranks() {
            let layers = if rank == 0 { &self.weights.layers } else { &self.peer()?.layers };
            for (layer, weights) in layers.iter().enumerate() {
                for (_, buffer) in weights.allocation_buffers() {
                    ensure!(devices.logical_rank(buffer.device_id)? as usize == rank, "weight on wrong logical rank");
                }
                weights.allocation_receipts(&mut inventory, &format!("rank.{rank}.layer.{layer}"))?;
            }
        }
        self.weights.norm.inventory(&mut inventory, "norm")?;
        match &self.weights.head {
            super::super::weights::GlmHead::Bf16(owner) => owner.inventory(&mut inventory, "head")?,
            super::super::weights::GlmHead::RowFp8(owner) => {
                ensure!(self.embedding.shares_vocabulary(owner), "head and embedding do not share vocabulary owner");
                owner.head.inventory(&mut inventory, "head")?;
            }
            super::super::weights::GlmHead::K6(owner) => {
                for (name, allocation) in [("payload", &owner.payload), ("suh", &owner.suh), ("svh", &owner.svh)] {
                    allocation.inventory(&mut inventory, &format!("head.{name}"))?;
                }
            }
        }
        if let Some(alpha) = &self.weights.rope_alpha { inventory.host("rope_alpha", alpha)?; }
        inventory.receipts.extend(self.embedding.allocation_receipts(devices, "glm.embedding")?);
        Ok(inventory.receipts)
    }

    pub(crate) fn serving_probe_receipts(&self) -> Result<Vec<Receipt>> {
        let physical = self.census_physical_devices()?;
        let mut inventory = crate::shared::memory::receipts::Inventory::new(
            CoordinatorDevices::new(&physical)?, "glm.serving",
            Lifetime { first: Phase::Warmup, last: Phase::Growth }, Category::Workspace, "serving probe");
        if let Some(probe) = self.draft_probe.try_borrow()?.as_ref() {
            probe.allocation_receipts(&mut inventory, "draft_probe")?;
        }
        Ok(inventory.receipts)
    }

    pub(crate) fn warmup_upload_receipt(&self) -> Result<Option<Receipt>> {
        let Some(actual) = self.library.sync_h2d_staging_buffer()? else { return Ok(None) };
        Ok(Some(Receipt {
            allocation: Allocation::buffer(
                "glm.upload.sync-successor", Location::PINNED, actual.bytes as u64,
                Lifetime { first: Phase::Warmup, last: Phase::Growth },
                Category::Staging, "GLM warmup sync-H2D",
            ),
            address: actual.ptr as u64,
        }))
    }

    pub(crate) fn census_physical_devices(&self) -> Result<Vec<i32>> {
        (0..self.ranks()).map(|rank| self.exchange_device(rank)).collect()
    }

    pub(crate) fn exchange_receipts(&self) -> Result<Vec<Receipt>> {
        let mut out = Vec::new();
        let physical_devices = self.census_physical_devices()?;
        let devices = CoordinatorDevices::new(&physical_devices)?;
        if let Some(exchange) = &self.exchange {
            for (rank, name, buffer) in exchange.allocation_buffers() {
                let mut census = Census {
                    prefix: format!("glm.exchange.rank.{rank}"), receipts: Vec::new(),
                    devices, rank: u8::try_from(rank)?,
                };
                census.record(&name, buffer, Lifetime::TARGET, Category::Transport, "GLM exchanges")?;
                out.extend(census.receipts);
            }
            if let Some((address, bytes)) = exchange.diagnostic_host_backing()? {
                out.push(Receipt {
                    allocation: Allocation::buffer("glm.exchange.diagnostic-ordinals", Location::HOST,
                        bytes, Lifetime::TARGET, Category::Transport, "GLM exchanges"),
                    address,
                });
            }
        }
        if let Some(context) = &self.context {
            out.extend(context.allocation_receipts(self)?);
        }
        Ok(out)
    }

    pub(crate) fn target_hook_receipts(&self) -> Result<Vec<Receipt>> {
        let mut out = Vec::new();
        let physical_devices = self.census_physical_devices()?;
        let devices = CoordinatorDevices::new(&physical_devices)?;
        for rank in 0..self.ranks() {
            let (decode, prefill) = if rank == 0 {
                (&self.decode_workspace, &self.prefill_slots)
            } else {
                let peer = self.peer()?;
                (&peer.decode_workspace, &peer.prefill_slots)
            };
            let logical_rank = devices.logical_rank(self.exchange_device(rank)?)?;
            let mut census = Census {
                prefix: format!("glm.target.gpu.{logical_rank}"),
                receipts: Vec::new(),
                devices, rank: logical_rank,
            };
            if let Some(w) = decode.borrow().as_ref() {
                census.hooks("decode", std::slice::from_ref(w), self.full_prefill_logits)?;
            }
            census.hooks("prefill", &prefill.borrow(), self.full_prefill_logits)?;
            if rank == 0 {
                for (name, head, category) in [
                    (
                        "target_head",
                        self.target_core_head.as_deref(),
                        Category::Workspace,
                    ),
                    (
                        "draft_head",
                        self.draft_core_head.as_ref(),
                        Category::Drafter,
                    ),
                ] {
                    if let Some(head) = head {
                        census.projection(
                            &format!("core.{name}"),
                            head.allocation_scratch(),
                            Lifetime::TARGET,
                            category,
                        )?;
                    }
                }
                for w in decode.borrow().iter().chain(prefill.borrow().iter()) {
                    ensure!(
                        match (&w.core_head, &self.target_core_head) {
                            (Some(actual), Some(owner)) => Rc::ptr_eq(actual, owner),
                            (None, None) => true,
                            _ => false,
                        },
                        "GLM retained target head does not share its engine owner"
                    );
                }
            } else {
                ensure!(
                    decode
                        .borrow()
                        .iter()
                        .chain(prefill.borrow().iter())
                        .all(|w| w.core_head.is_none()),
                    "GLM peer retains an unexpected target head"
                );
            }
            out.extend(census.receipts);
        }
        Ok(out)
    }

    pub(crate) fn target_buffer_receipts(&self) -> Result<Vec<Receipt>> {
        let mut out = Vec::new();
        let physical_devices = self.census_physical_devices()?;
        let devices = CoordinatorDevices::new(&physical_devices)?;
        for rank in 0..self.ranks() {
            let (kv, index, rope, decode, prefill) = if rank == 0 {
                (
                    &self.kv,
                    &self.index,
                    &self.cos_sin,
                    &self.decode_workspace,
                    &self.prefill_slots,
                )
            } else {
                let peer = self.peer()?;
                (
                    &peer.kv,
                    &peer.index,
                    &peer.cos_sin,
                    &peer.decode_workspace,
                    &peer.prefill_slots,
                )
            };
            let logical_rank = devices.logical_rank(self.exchange_device(rank)?)?;
            let mut census = Census {
                prefix: format!("glm.target.gpu.{logical_rank}"),
                receipts: Vec::new(),
                devices, rank: logical_rank,
            };
            for (layer, buffer) in kv.iter().enumerate() {
                census.buffer(
                    &format!("kv.{layer}"),
                    buffer.buffer,
                    Lifetime::TARGET,
                    Category::Kv,
                )?;
            }
            for (layer, buffer) in index.iter().enumerate() {
                if let Some(buffer) = buffer {
                    census.buffer(
                        &format!("index.{layer}"),
                        buffer.buffer,
                        Lifetime::TARGET,
                        Category::Kv,
                    )?;
                }
            }
            census.buffer("rope", rope.buffer, Lifetime::TARGET, Category::Kv)?;
            if let Some(workspace) = decode.borrow().as_ref() {
                census.rows(
                    "decode",
                    std::slice::from_ref(workspace),
                    true,
                    self.library,
                    self.stream_of(rank),
                    self.full_prefill_logits,
                )?;
            }
            census.rows(
                "prefill",
                &prefill.borrow(),
                false,
                self.library,
                self.stream_of(rank),
                self.full_prefill_logits,
            )?;
            out.extend(census.receipts);
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests;
