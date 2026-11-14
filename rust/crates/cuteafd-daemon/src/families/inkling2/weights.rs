//! Unchanged core tensor bytes in one caller-admitted arena per weight set.
use crate::shared::{experts::rtx::dense::{DenseLoad, Operand, Rect},
    memory::{device::{Allocation, Device}, receipts::Inventory, weight_arena::DeviceWeightArena}};
use anyhow::{ensure, Context, Result};
use cuteafd_core::memory_layout::ArenaTag;
use cuteafd_loader::{families::inkling2::weight_layout::CoordinatorWeights, plan::Checkpoint,
    serving_capacity::WeightArenaLayout};
use std::collections::BTreeMap;

/// Views retain their physical backing independently of the loading object.
/// DenseLoad supplies the shared mount/recorded-push path and staging owner.
pub(crate) struct TargetWeights<'a> {
    pub rank: usize,
    pub layers: Vec<usize>,
    operands: BTreeMap<String, Allocation<'a>>,
}

pub(crate) fn execution_arena(layout: &CoordinatorWeights) -> Result<WeightArenaLayout> {
    let mut tensors = layout.arena.views().iter().map(|(name, view)| (name.clone(), view.bytes())).collect::<Vec<_>>();
    for &layer in &layout.layers {
        tensors.push((format!("model.llm.layers.{layer}.attn.rel_logits_proj.padded"), 1024*64*2));
    }
    WeightArenaLayout::new(layout.arena.id(), tensors)
}

fn transpose_relative(source: &[u8]) -> Result<Vec<u8>> {
    ensure!(source.len() == 16*1024*2, "Inkling relative projection must be logical [16,1024] BF16");
    let mut output = vec![0u8; 1024*64*2];
    for row in 0..1024 {
        for column in 0..16 {
            output[(row*64+column)*2..(row*64+column+1)*2]
                .copy_from_slice(&source[(column*1024+row)*2..(column*1024+row+1)*2]);
        }
    }
    Ok(output)
}

impl<'a> TargetWeights<'a> {
    pub fn load(device: Device<'a>, checkpoint: &Checkpoint, layout: &CoordinatorWeights) -> Result<Self> {
        ensure!(layout.rank == 0 && layout.arena.views().keys().all(|name| !name.contains('#')),
            "Inkling stage2 core mount requires whole coordinator operands");
        ensure!(!crate::shared::experts::rtx::dense::is_recording(),
            "Inkling relative transpose is a cold mount transform, not yet a push finish");
        let mut rects = BTreeMap::new();
        for (name, view) in layout.arena.views() {
            let rect = Rect::tensor(checkpoint, name, None, None)?;
            ensure!(rect.bytes() == view.bytes(), "Inkling core {name}: checkpoint differs from admitted view");
            rects.insert(name.clone(), rect);
        }
        let dense = DenseLoad::mount(device.library);
        dense.admit_mount(layout.mount_load(checkpoint)?)?;
        let arena = DeviceWeightArena::new(device, execution_arena(layout)?, ArenaTag::weights("inkling2/target"))?;
        let mut operands = BTreeMap::new();
        for (name, rect) in rects {
            let bytes = rect.bytes();
            let allocation = arena.view(&name, bytes)?;
            dense.operand(Operand { device: device.id, buffer: allocation.buffer, lands: vec![(rect, 0)], bytes })?;
            operands.insert(name, allocation);
        }
        dense.release();
        for &layer in &layout.layers {
            let name = format!("model.llm.layers.{layer}.attn.rel_logits_proj");
            let mut source = vec![0u8; 16*1024*2];
            Rect::tensor(checkpoint, &format!("{name}.proj"), None, None)?.read(&mut source)?;
            let padded = transpose_relative(&source)?;
            let allocation = arena.view(&format!("{name}.padded"), padded.len())?;
            device.run(|| device.library.copy_h2d(allocation.buffer, &padded))?;
            tracing::info!(layer, logical = "[16,1024]", physical = "[1024,64]", logical_k = 16,
                physical_k = 64, source_staging_bytes = source.len(), output_staging_bytes = padded.len(),
                padding_bytes = 1024*48*2, padding_zero_exact = true, "Inkling relative raw-BF16 transpose/pad load");
            operands.insert(format!("{name}.padded"), allocation);
        }
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

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn relative_transpose_preserves_bf16_bits_and_exact_zero_padding() {
        let source = (0..16*1024).flat_map(|n| ((n*29%65536) as u16).to_le_bytes()).collect::<Vec<_>>();
        let padded = transpose_relative(&source).unwrap();
        for row in 0..1024 {
            for col in 0..16 {
                assert_eq!(&padded[(row*64+col)*2..(row*64+col+1)*2],
                    &source[(col*1024+row)*2..(col*1024+row+1)*2]);
            }
            assert!(padded[(row*64+16)*2..(row+1)*64*2].iter().all(|&b| b == 0));
        }
        assert!(transpose_relative(&source[..100]).is_err());
    }
}
