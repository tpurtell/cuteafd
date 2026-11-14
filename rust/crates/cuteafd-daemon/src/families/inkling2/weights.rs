//! Unchanged core tensor bytes in one caller-admitted arena per weight set.
use crate::shared::{experts::rtx::dense::{DenseLoad, Operand, Rect},
    memory::{device::{Allocation, Device}, receipts::Inventory, weight_arena::DeviceWeightArena}};
use anyhow::{ensure, Context, Result};
use cuteafd_core::memory_layout::ArenaTag;
use cuteafd_loader::{families::inkling2::weight_layout::CoordinatorWeights, plan::Checkpoint};
use std::collections::BTreeMap;

/// Views retain their physical backing independently of the loading object.
/// DenseLoad supplies the shared mount/recorded-push path and staging owner.
pub(crate) struct TargetWeights<'a> {
    pub rank: usize,
    pub layers: Vec<usize>,
    operands: BTreeMap<String, Allocation<'a>>,
}

impl<'a> TargetWeights<'a> {
    pub fn load(device: Device<'a>, checkpoint: &Checkpoint, layout: &CoordinatorWeights) -> Result<Self> {
        ensure!(layout.rank == 0 && layout.arena.views().keys().all(|name| !name.contains('#')),
            "Inkling stage2 core mount requires whole coordinator operands");
        let mut rects = BTreeMap::new();
        for (name, view) in layout.arena.views() {
            let rect = Rect::tensor(checkpoint, name, None, None)?;
            ensure!(rect.bytes() == view.bytes(), "Inkling core {name}: checkpoint differs from admitted view");
            rects.insert(name.clone(), rect);
        }
        let dense = DenseLoad::mount(device.library);
        dense.admit_mount(layout.mount_load(checkpoint)?)?;
        let arena = DeviceWeightArena::new(device, layout.arena.clone(), ArenaTag::weights("inkling2/target"))?;
        let mut operands = BTreeMap::new();
        for (name, rect) in rects {
            let bytes = rect.bytes();
            let allocation = arena.view(&name, bytes)?;
            dense.operand(Operand { device: device.id, buffer: allocation.buffer, lands: vec![(rect, 0)], bytes })?;
            operands.insert(name, allocation);
        }
        dense.release();
        Ok(Self { rank: layout.rank, layers: layout.layers.clone(), operands })
    }
    pub fn operand(&self, name: &str) -> Result<&Allocation<'a>> {
        self.operands.get(name).with_context(|| format!("Inkling target lacks operand {name}"))
    }
    pub fn inventory(&self, inventory: &mut Inventory<'_>, prefix: &str) -> Result<()> {
        for (name, owner) in &self.operands { owner.inventory(inventory, &format!("{prefix}.{name}"))?; }
        Ok(())
    }
}
