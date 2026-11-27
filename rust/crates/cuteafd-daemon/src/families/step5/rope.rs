//! Step-5 selects checkpoint recipes; shared code owns and initializes storage.
use crate::shared::memory::{device::Device, receipts::Inventory,
    weight_arena::{DeviceWeightArena, weight_allocation}};
use crate::shared::rope::{RotaryTables, Table};
use anyhow::Result;
use cuteafd_ffi::programs::Programs;
use cuteafd_loader::plan::families::step5::coordinator::CoordinatorContract;
use std::rc::Rc;
use cuteafd_core::memory_layout::ArenaTag;

pub(super) struct LayerTables<'a> {
    pub attention: Rc<Table<'a>>,
    pub indexer: Option<Rc<Table<'a>>>,
}

pub(super) struct TargetTables<'a> {
    pub tables: RotaryTables<'a>,
    pub layers: Vec<LayerTables<'a>>,
}

impl<'a> TargetTables<'a> {
    pub fn load(
        device: Device<'a>,
        programs: &Programs<'a>,
        contract: &CoordinatorContract,
        prefix: &str,
        available: u64,
        reservations: &[u64],
    ) -> Result<Self> {
        Self::load_in(device, programs, contract, prefix, available, reservations, None)
    }

    /// Rotary planes are immutable target weights; on a coordinator rank they
    /// are views of that rank's arena.
    pub fn load_in(
        device: Device<'a>,
        programs: &Programs<'a>,
        contract: &CoordinatorContract,
        prefix: &str,
        available: u64,
        reservations: &[u64],
        arena: Option<&DeviceWeightArena<'a>>,
    ) -> Result<Self> {
        let tables = RotaryTables::load_with(
            device,
            programs,
            contract.rope_tables.clone(),
            prefix,
            available,
            reservations,
            |name, bytes| weight_allocation(device, arena, name, bytes, ArenaTag::weights("step5/target")),
        )?;
        let layers = contract
            .layers
            .iter()
            .map(|layer| {
                Ok(LayerTables {
                    attention: tables.get(layer.rope)?,
                    indexer: layer
                        .indexer_rope
                        .map(|recipe| tables.get(recipe))
                        .transpose()?,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self { tables, layers })
    }

    /// Each distinct recipe's cos/sin planes once, however many layers share it.
    pub(super) fn inventory(&self, inventory: &mut Inventory<'_>, prefix: &str) -> Result<()> {
        for (index, recipe) in self.tables.plan.recipes.iter().enumerate() {
            let table = self.tables.get(*recipe)?;
            table.cos.inventory(inventory, &format!("{prefix}.rotary.{index}.cos"))?;
            table.sin.inventory(inventory, &format!("{prefix}.rotary.{index}.sin"))?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use anyhow::{ensure, Context};
    use cuteafd_loader::plan::checkpoint::Checkpoint;

    #[test]
    #[ignore = "requires Step-5 snapshot, native library, Python RoPE fixtures and CUDA"]
    fn native_rope_owner_is_admitted_shared_drained_and_byte_exact() -> Result<()> {
        // SAFETY: the operator supplies a sealed native library with matching exports.
        let library =
            unsafe { cuteafd_ffi::NativeLibrary::load(std::env::var("CUTEAFD_NATIVE_LIB")?)? };
        let device = crate::shared::memory::device::cache_test_device(&library)?;
        let manifest = std::path::PathBuf::from(std::env::var("CUTEAFD_PROGRAM_MANIFEST")?);
        let programs = library.programs()?.with_manifest(&manifest)?;
        let checkpoint =
            Checkpoint::open(std::path::Path::new(&std::env::var("CUTEAFD_SNAPSHOT")?))?;
        let contract = CoordinatorContract::from_checkpoint(&checkpoint)?;
        let bytes = contract.rope_tables.allocation_bytes;
        assert_eq!(bytes, 576 << 20);
        assert!(
            TargetTables::load(device, &programs, &contract, "step5_", bytes - 1, &[]).is_err()
        );
        assert!(TargetTables::load(device, &programs, &contract, "step5_", bytes, &[1]).is_err());
        let started = std::time::Instant::now();
        let owner = TargetTables::load(device, &programs, &contract, "step5_", bytes, &[])?;
        let elapsed = started.elapsed();
        tracing::info!(
            ?elapsed,
            allocation_bytes = bytes,
            recipes = owner.tables.plan.recipes.len(),
            "native rotary owner readiness"
        );
        eprintln!(
            "native rotary owner readiness: {:.3} ms, {bytes} bytes",
            elapsed.as_secs_f64() * 1000.0
        );
        assert_eq!(owner.layers.len(), 92);
        assert!(Rc::ptr_eq(
            &owner.layers[0].attention,
            &owner.layers[1].attention
        ));
        assert!(Rc::ptr_eq(
            &owner.layers[3].attention,
            &owner.layers[7].attention
        ));
        assert!(Rc::ptr_eq(
            owner.layers[3].indexer.as_ref().unwrap(),
            owner.layers[7].indexer.as_ref().unwrap()
        ));
        assert!(!Rc::ptr_eq(
            &owner.layers[0].attention,
            &owner.layers[3].attention
        ));
        let retained = owner
            .tables
            .plan
            .recipes
            .iter()
            .map(|&recipe| Ok((recipe, owner.tables.get(recipe)?)))
            .collect::<Result<Vec<_>>>()?;
        drop(owner);
        // Layer/model ownership may end before the last consumer; its Rc keeps
        // storage live. Load returned only after the builder stream was drained.
        let root = std::path::PathBuf::from(std::env::var("CUTEAFD_ROPE_FIXTURE_DIR")?);
        for (recipe, table) in retained {
            assert_eq!(Rc::strong_count(&table), 1);
            for (name, buffer) in ["cos", "sin"]
                .into_iter()
                .zip([table.cos.buffer, table.sin.buffer])
            {
                let expected = std::fs::read(root.join(format!("{}_{name}.bin", recipe.program())))
                    .context("Python native/official byte-equal rotary fixture")?;
                ensure!(
                    expected.len() == buffer.bytes,
                    "rotary fixture length mismatch"
                );
                let mut actual = vec![0; buffer.bytes];
                device.run(|| library.copy_d2h_quiescent(&mut actual, buffer))?;
                assert_eq!(
                    actual, expected,
                    "native owner table differs from Python AOT fixture"
                );
            }
        }
        Ok(())
    }
}
