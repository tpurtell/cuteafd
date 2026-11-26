//! Immutable, deduplicated native rotary tables, built once before requests.
use crate::shared::memory::device::{Allocation, Device, Stream};
use anyhow::{ensure, Context, Result};
use cuteafd_ffi::{
    programs::{Programs, Scalar},
    CuteafdDeviceBuffer,
};
use cuteafd_loader::plan::rope::{Recipe, Tables};
use std::rc::Rc;
use crate::shared::memory::Tag;

pub(crate) struct Table<'a> {
    pub cos: Allocation<'a>,
    pub sin: Allocation<'a>,
}

pub(crate) struct RotaryTables<'a> {
    pub plan: Tables,
    tables: Vec<Rc<Table<'a>>>,
}

impl<'a> RotaryTables<'a> {
    pub fn load(
        device: Device<'a>,
        programs: &Programs<'a>,
        plan: Tables,
        prefix: &str,
        available: u64,
        reservations: &[u64],
    ) -> Result<Self> {
        Self::load_with(device, programs, plan, prefix, available, reservations,
            |_, bytes| Allocation::new(device, bytes, Tag::weights("shared/rotary")))
    }

    pub fn arena_requests(plan: &Tables, prefix: &str) -> Result<Vec<(String, usize)>> {
        let mut requests = Vec::new();
        for (index, recipe) in plan.recipes.iter().enumerate() {
            let plane = usize::try_from(recipe.allocation_bytes(plan.extent)? / 2)?;
            for name in ["cos", "sin"] { requests.push((format!("{prefix}rotary.{index}.{name}"), plane)); }
        }
        Ok(requests)
    }

    pub fn load_with(device: Device<'a>, programs: &Programs<'a>, plan: Tables, prefix: &str,
        available: u64, reservations: &[u64],
        mut allocate: impl FnMut(&str, usize) -> Result<Allocation<'a>>) -> Result<Self> {
        // Budget and every ABI are checked before the first device allocation.
        plan.check_budget(available, reservations)?;
        for recipe in &plan.recipes {
            let name = format!("{prefix}{}", recipe.program());
            let spec = programs.spec(&name)?;
            ensure!(
                spec.scalar_kinds == b"ifffff",
                "rotary table scalar ABI mismatch"
            );
            ensure!(
                spec.geometry["rotary_span"].as_u64() == Some(recipe.span as u64)
                    && spec.geometry["llama3"].as_bool() == Some(recipe.scaling.is_some()),
                "rotary table geometry mismatch"
            );
            ensure!(
                spec.pointers == ["cos", "sin"],
                "rotary table pointer ABI mismatch"
            );
        }
        let mut tables = Vec::new();
        for (index, recipe) in plan.recipes.iter().enumerate() {
            let program = device.run(|| {
                programs.program(&format!("{prefix}{}", recipe.program()), &["cos", "sin"])
            })?;
            let plane = usize::try_from(recipe.allocation_bytes(plan.extent)? / 2)?;
            let _scope = cuteafd_ffi::memory_ledger::scope("rotary_tables");
            let table = Table {
                cos: crate::shared::memory::weight_arena::checked_weight_allocation(
                    device, &format!("{prefix}rotary.{index}.cos"), plane, &mut allocate)?,
                sin: crate::shared::memory::weight_arena::checked_weight_allocation(
                    device, &format!("{prefix}rotary.{index}.sin"), plane, &mut allocate)?,
            };
            // The stream is created after storage, so error unwinding drains before freeing.
            let stream = Stream::new(device)?;
            let scaling = recipe.scaling;
            let scalars = [
                Scalar::I32(plan.extent as i32),
                Scalar::F32(recipe.theta),
                Scalar::F32(scaling.map_or(1.0, |s| s.factor)),
                Scalar::F32(scaling.map_or(plan.extent as f32, |s| s.original_max)),
                Scalar::F32(scaling.map_or(1.0, |s| s.low_factor)),
                Scalar::F32(scaling.map_or(32.0, |s| s.high_factor)),
            ];
            device.run(|| {
                // SAFETY: both admitted BF16 planes belong to this device, span the
                // complete extent, and outlive the drained load-only stream.
                unsafe {
                    program.launch(
                        &[table.cos.buffer.ptr, table.sin.buffer.ptr],
                        &scalars,
                        stream.raw,
                    )
                }
            })?;
            stream.drain()?;
            tables.push(Rc::new(table));
        }
        Ok(Self { plan, tables })
    }

    pub(crate) fn inventory(&self, inventory: &mut crate::shared::memory::receipts::Inventory<'_>,
        prefix: &str) -> Result<()> {
        for (index, table) in self.tables.iter().enumerate() {
            table.inventory(inventory, &format!("{prefix}rotary.{index}"))?;
        }
        Ok(())
    }

    pub fn get(&self, recipe: Recipe) -> Result<Rc<Table<'a>>> {
        let index = self
            .plan
            .recipes
            .iter()
            .position(|r| *r == recipe)
            .context("rotary recipe was not admitted")?;
        Ok(Rc::clone(&self.tables[index]))
    }
}

impl Table<'_> {
    pub(crate) fn inventory(&self, inventory: &mut crate::shared::memory::receipts::Inventory<'_>,
        name: &str) -> Result<()> {
        self.cos.inventory(inventory, &format!("{name}.cos"))?;
        self.sin.inventory(inventory, &format!("{name}.sin"))
    }

    pub fn buffers(&self) -> (CuteafdDeviceBuffer, CuteafdDeviceBuffer) {
        (self.cos.buffer, self.sin.buffer)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn rotary_requests_and_retained_table_inventory_use_the_same_planes() -> Result<()> {
        use crate::shared::memory::{receipts::Inventory, weight_arena::DeviceWeightArena};
        use cuteafd_core::memory_layout::Category;
        use cuteafd_loader::{serving_capacity::WeightArenaLayout, placement::planner::{CoordinatorDevices,Lifetime,Phase}};
        let fixture=cuteafd_ffi::native_library_lifetime_fixture::Fixture::build_for_tests()?;
        let library=fixture.load()?;
        fixture.configure_memory(&library,1,1<<30)?;
        fixture.immediate_copies(&library)?;
        let device=Device { library:&library,id:1 };
        let recipe=Recipe::new(32,10000.0,None)?;
        let plan=Tables::new(3,[recipe,recipe,Recipe::new(64,10000.0,None)?])?;
        let requests=RotaryTables::arena_requests(&plan,"draft.")?;
        assert_eq!(requests.len(),4);
        assert_eq!(requests.iter().map(|(_,bytes)|bytes).sum::<usize>() as u64,plan.allocation_bytes);
        let arena=DeviceWeightArena::new(device,WeightArenaLayout::new("test/rotary",requests.clone())?, crate::shared::memory::TEST_ARENA)?;
        let tables=requests.chunks_exact(2).map(|pair| -> Result<_> {
            let cos=arena.view(&pair[0].0,pair[0].1)?;
            let sin=arena.view(&pair[1].0,pair[1].1)?;
            device.run(|| { library.copy_h2d(cos.buffer,&vec![37;cos.buffer.bytes])?;
                library.copy_h2d(sin.buffer,&vec![91;sin.buffer.bytes]) })?;
            Ok(Rc::new(Table { cos,sin }))
        }).collect::<Result<Vec<_>>>()?;
        let rotary=RotaryTables { plan:plan.clone(),tables };
        drop(arena);
        let mut inventory=Inventory::new(CoordinatorDevices::new(&[1])?,"rotary",
            Lifetime { first:Phase::Load,last:Phase::Growth },Category::Weights,"rotary constants");
        rotary.inventory(&mut inventory,"draft.")?;
        assert_eq!(inventory.receipts.iter().filter(|r|r.allocation.view.is_none()).count(),1);
        assert_eq!(inventory.receipts.iter().filter(|r|r.allocation.view.is_some()).count(),4);
        let table=rotary.get(recipe)?;
        let mut bytes=vec![0;table.cos.buffer.bytes];
        device.run(|| library.copy_d2h_quiescent(&mut bytes,table.cos.buffer))?;
        assert!(bytes.iter().all(|&b|b==37));
        // An absent ABI must refuse before invoking a storage callback.
        let programs=library.programs()?;
        let mut called=false;
        assert!(RotaryTables::load_with(device,&programs,plan,"missing.",1<<30,&[],|_,bytes| {
            called=true; Allocation::new(device,bytes, crate::shared::memory::TEST)
        }).is_err());
        assert!(!called);
        Ok(())
    }
}
