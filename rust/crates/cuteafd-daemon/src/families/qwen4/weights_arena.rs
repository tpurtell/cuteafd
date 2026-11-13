//! Physical coordinator scopes; every final view retains its backing.
use super::*;
use crate::shared::memory::weight_arena::DeviceWeightArena;
use std::{collections::BTreeMap, rc::Rc};
use cuteafd_core::memory_layout::ArenaTag;

pub(crate) struct Arenas<'a> {
    target: BTreeMap<i32, DeviceWeightArena<'a>>,
    draft: Option<DeviceWeightArena<'a>>,
}

impl<'a> Qwen4Loader<'a> {
    pub(super) fn allocate(&self, name: &str, bytes: usize) -> Result<Allocation<'a>> {
        let device = self.device()?;
        let arenas = self.arenas.as_ref().context("Qwen weights require planned arenas")?;
        let arena = if name.starts_with("mtp.") { arenas.draft.as_ref() }
            else { arenas.target.get(&device.id) }.context("Qwen weight arena device/scope")?;
        crate::shared::memory::weight_arena::weight_allocation(device, Some(arena), name, bytes, ArenaTag::weights("qwen4/target"))
    }

    pub(crate) fn embedding_allocation(&self, name: &str, bytes: usize) -> Result<Allocation<'a>> {
        self.allocate(name, bytes)
    }

    pub(crate) fn with_arenas(&self, cfg: &Qwen4Config, owners: &[usize],
        ranks: &[(Device<'a>, *mut c_void)], mtp: bool, fp8_head: bool,
        embedding: bool) -> Result<Self> {
        ensure!(self.arenas.is_none(), "Qwen arenas already planned");
        ensure!(!ranks.is_empty() && ranks.iter().all(|(device, _)| std::ptr::eq(device.library, self.library))
            && ranks.iter().map(|(d, _)| d.id).collect::<std::collections::BTreeSet<_>>().len() == ranks.len(),
            "Qwen arena needs distinct devices in one native library");
        let embedding_bytes = if embedding {
            cuteafd_loader::families::qwen4::weight_layout::embedding_table_bytes(self.checkpoint)?
        } else { 0 };
        let layouts = cuteafd_loader::families::qwen4::weight_layout::coordinator_arenas(
            self.checkpoint, cfg, owners, ranks.len(), mtp,
            resident::Qwen4Representation { fp8_projections: self.fp8_decode, fp8_head }, embedding_bytes)?;
        // Sample every rank and admit all simultaneously live owners before the
        // first backing allocates. Widening/head jobs use mapped host staging;
        // projection RowQuant alone retains a STEP_BYTES device scratch.
        let mut available = Vec::with_capacity(ranks.len());
        let mut sm120 = Vec::with_capacity(ranks.len());
        let mut staging = vec![0; ranks.len()];
        for (rank, (device, _)) in ranks.iter().enumerate() {
            let (free, info) = device.run(|| Ok((self.library.cuda_memory_info()?.0,
                self.library.cuda_device_info(device.id)?)))?;
            available.push(free);
            sm120.push(info.compute_capability_major == 12 && info.compute_capability_minor == 0
                && info.integrated == 0);
            if self.fp8_decode && (owners.contains(&rank) || rank == 0 && mtp) {
                staging[rank] = cuteafd_loader::shared::dense_read::STEP_BYTES;
            }
        }
        admit_startup(&layouts, &staging, &sm120, &available)?;
        let mut target = BTreeMap::new();
        let mut draft = None;
        for (rank, layout) in layouts {
            let is_draft = layout.id().contains("/mtp/");
            let arena = DeviceWeightArena::new(ranks[rank].0, layout, if is_draft { ArenaTag::drafter_weights("qwen4/mtp") } else { ArenaTag::weights("qwen4/target") })?;
            if is_draft { draft = Some(arena); }
            else { target.insert(ranks[rank].0.id, arena); }
        }
        Ok(Self { arenas: Some(Rc::new(Arenas { target, draft })), ..self.clone() })
    }
}

fn admit_startup(layouts: &[(usize, cuteafd_loader::serving_capacity::WeightArenaLayout)],
    staging: &[usize], sm120: &[bool], available: &[usize]) -> Result<()> {
    ensure!(staging.len() == available.len() && sm120.len() == available.len(),
        "Qwen startup admission rank mismatch");
    let mut required = vec![0usize; available.len()];
    for (rank, bytes) in layouts.iter().map(|(rank, layout)| (*rank, layout.bytes()))
        .chain(staging.iter().copied().enumerate().filter(|(_, bytes)| *bytes > 0)) {
        ensure!(rank < required.len(), "Qwen startup admission owner outside ranks");
        let physical = if sm120[rank] && bytes > 1 << 20 {
            bytes.div_ceil(2 << 20).checked_mul(2 << 20).context("Qwen backing extent overflow")?
        } else { bytes };
        required[rank] = required[rank].checked_add(physical).context("Qwen startup bytes overflow")?;
    }
    for (rank, (&required, &free)) in required.iter().zip(available).enumerate() {
        ensure!(required <= free,
            "Qwen rank {rank} startup arenas and dense workspace need {required} bytes, only {free} available");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use cuteafd_loader::serving_capacity::WeightArenaLayout;

    #[test]
    fn startup_refuses_when_payload_fits_but_backings_and_staging_do_not() -> Result<()> {
        let target = WeightArenaLayout::new("qwen4/target/rank0", [("target".into(), (3 << 20) + 1)])?;
        let mtp = WeightArenaLayout::new("qwen4/mtp/rank0", [("mtp".into(), (1 << 20) + 1)])?;
        let payload = target.payload_bytes() + mtp.payload_bytes();
        let layouts = vec![(0, target), (0, mtp)];
        let staging = cuteafd_loader::shared::dense_read::STEP_BYTES;
        assert!(payload < 6 << 20);
        assert!(admit_startup(&layouts, &[staging], &[true], &[6 << 20]).is_err());
        assert!(admit_startup(&layouts, &[staging], &[true], &[staging + (6 << 20) - 1]).is_err());
        admit_startup(&layouts, &[staging], &[true], &[staging + (6 << 20)])?;
        let logical = layouts.iter().map(|(_, layout)| layout.bytes()).sum::<usize>();
        admit_startup(&layouts, &[0], &[false], &[logical])?;
        assert!(admit_startup(&layouts, &[0], &[false], &[logical - 1]).is_err());
        Ok(())
    }
}

/// Physical bytes of the distinct device weight backings actually retained
/// (arena roots, plus the device embedding when it is one), read from the
/// same receipts the census inventories: the runtime side of TP1's declared
/// `single_owner_weights`, compared exactly.
pub(crate) fn retained_backing_bytes(physical: &[i32], model: &Qwen4Weights<'_>,
    embedding: &crate::shared::token_io::TokenEmbedding<'_>) -> Result<u64> {
    use cuteafd_loader::placement::planner::{CoordinatorDevices, Domain};
    let devices = CoordinatorDevices::new(physical)?;
    let mut receipts = model.allocation_receipts(devices)?;
    receipts.extend(embedding.allocation_receipts(devices, "qwen4.embedding")?);
    let mut roots = std::collections::BTreeMap::new();
    for receipt in receipts.iter().filter(|r| r.allocation.view.is_none()
        && r.allocation.location.domain == Domain::Device) {
        let previous = roots.insert(receipt.allocation.id.clone(), (receipt.address, receipt.allocation.bytes));
        ensure!(previous.is_none_or(|p| p == (receipt.address, receipt.allocation.bytes)),
            "Qwen weight backing {} reported with two extents", receipt.allocation.id);
    }
    roots.values().try_fold(0u64, |sum, (_, bytes)| sum.checked_add(*bytes).context("Qwen backing bytes overflow"))
}

pub(crate) fn probe(library: &NativeLibrary, physical: &[i32], model: &Qwen4Weights<'_>,
    embedding: &crate::shared::token_io::TokenEmbedding<'_>) -> Result<()> {
    use cuteafd_loader::placement::planner::{CoordinatorDevices, PhaseOrder};
    let devices = CoordinatorDevices::new(physical)?;
    let mut census = crate::shared::memory::receipts::Census::default();
    census.owners(model.allocation_receipts(devices)?);
    census.owners(embedding.allocation_receipts(devices, "qwen4.embedding")?);
    embedding.allocation_aliases(&mut census, devices, "qwen4.embedding")?;
    census.normalize(PhaseOrder::TARGET_THEN_DRAFT)?;
    crate::shared::inventory::startup_probe(library, physical, "qwen4/weights-loaded",
        crate::shared::inventory::StartupEvidence { census: Some(&census), ..Default::default() })
}

impl Qwen4Weights<'_> {
    pub(crate) fn allocation_receipts(&self,
        devices: cuteafd_loader::placement::planner::CoordinatorDevices<'_>)
        -> Result<Vec<cuteafd_loader::placement::planner::Receipt>> {
        use cuteafd_core::memory_layout::Category;
        use cuteafd_loader::placement::planner::Lifetime;
        let mut inventory = crate::shared::memory::receipts::Inventory::new(devices, "qwen4.weights",
            Lifetime::RESIDENT, Category::Weights, "coordinator weights");
        for (layer, part) in self.layers.iter().enumerate() {
            for (name, owner) in &part.operands {
                owner.inventory(&mut inventory, &format!("layers.{layer}.{name}"))?;
            }
        }
        for (index, owner) in self.mixer.iter().enumerate() { owner.inventory(&mut inventory, &format!("mixer.{index}"))?; }
        for (index, owner) in self.head.allocations().iter().enumerate() {
            owner.inventory(&mut inventory, &format!("head.{index}"))?;
        }
        if let Some(mtp) = &self.mtp {
            for (name, owner) in &mtp.layer.operands { owner.inventory(&mut inventory, &format!("mtp.layer.{name}"))?; }
            for (name, owner) in [("norm_hidden", &mtp.norm_hidden), ("norm_embed", &mtp.norm_embed),
                ("fc_hidden", &mtp.fc_hidden), ("fc_embed", &mtp.fc_embed)] {
                owner.inventory(&mut inventory, &format!("mtp.{name}"))?;
            }
            for (index, owner) in mtp.mixer.iter().enumerate() { owner.inventory(&mut inventory, &format!("mtp.mixer.{index}"))?; }
        }
        Ok(inventory.receipts)
    }
}
