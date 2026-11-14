//! Named cold-forward working owners; admission provides the exact layout.
//! This is not a serving cache: each layer's histories are independently
//! retained by the serving cache when that stage lands.
use crate::shared::memory::{device::{Allocation, Stream}, receipts::Inventory};
use anyhow::{Context, Result};
use cuteafd_core::memory_layout::Tag;
use cuteafd_loader::families::inkling2::execution::UnitLayout;
use std::{collections::BTreeMap, rc::Rc};

pub(crate) struct UnitOwner<'a> {
    pub stream: Rc<Stream<'a>>,
    pub layout: UnitLayout,
    buffers: BTreeMap<String, Allocation<'a>>,
}
impl<'a> UnitOwner<'a> {
    /// Caller admits each named physical request and its allocation tail first.
    pub fn new(stream: Rc<Stream<'a>>, layout: UnitLayout) -> Result<Self> {
        let buffers = layout.buffers.iter().map(|(name, bytes)| Ok((name.clone(),
            Allocation::new(stream.device, *bytes, Tag::workspace("inkling2/unit"))?)))
            .collect::<Result<BTreeMap<_, _>>>()?;
        let owner = Self { stream, layout, buffers };
        owner.zero()?;
        Ok(owner)
    }
    pub fn buffer(&self, name: &str) -> Result<cuteafd_ffi::CuteafdDeviceBuffer> {
        Ok(self.buffers.get(name).with_context(|| format!("Inkling unit lacks owner {name}"))?.buffer)
    }
    pub fn zero(&self) -> Result<()> {
        self.stream.device.run(|| {
            for buffer in self.buffers.values() {
                // SAFETY: these independently retained fresh owners are not yet
                // published to readers; all writes use the retained unit stream.
                unsafe { self.stream.device.library.cuda_zero_bytes_async(buffer.buffer, buffer.buffer.bytes,
                    self.stream.raw)?; }
            }
            Ok(())
        })
    }
    pub fn inventory(&self, inventory: &mut Inventory<'_>, prefix: &str) -> Result<()> {
        for (name, owner) in &self.buffers { owner.inventory(inventory, &format!("{prefix}.{name}"))?; }
        Ok(())
    }
}
impl Drop for UnitOwner<'_> {
    fn drop(&mut self) { self.stream.drain_or_abort("Inkling unit owners"); }
}
