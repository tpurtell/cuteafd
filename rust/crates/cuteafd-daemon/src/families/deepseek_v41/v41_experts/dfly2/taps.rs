//! Per-request DFly2 target tap store: last accepted normalized hidden state
//! `[max_requests, 5120]` BF16 and layers 28/32/36 indexer recall
//! `[max_requests, 3, 512]` I32 in one packed `DeviceArena`.

use crate::shared::draft::dfly2::Dfly2Frontiers;
use crate::shared::memory::device::{Allocation, Device};
use crate::shared::memory::weight_arena::DeviceArena;
use anyhow::{ensure, Context, Result};
use cuteafd_core::memory_layout::{arena_class, ArenaTag};
use cuteafd_ffi::{CuteafdDeviceBuffer, NativeLibrary};
use cuteafd_loader::serving_capacity::WeightArenaLayout;
use std::collections::BTreeMap;
use std::ffi::c_void;

pub(crate) const HIDDEN_ROW_BYTES: usize = 5120 * 2;
pub(crate) const RECALL_ROW_BYTES: usize = 512 * 4;

fn slice_buf(buf: CuteafdDeviceBuffer, offset: usize, bytes: usize) -> Result<CuteafdDeviceBuffer> {
    ensure!(offset <= buf.bytes && bytes <= buf.bytes - offset, "DFly2 tap slice out of range");
    Ok(CuteafdDeviceBuffer {
        ptr: unsafe { buf.ptr.cast::<u8>().add(offset).cast() },
        bytes,
        ..buf
    })
}

pub(crate) struct Dfly2TapStore<'a> {
    device: Device<'a>,
    prev_hidden: Allocation<'a>,
    recall: [Allocation<'a>; 3],
    _arena: DeviceArena<'a, arena_class::ReadWrite>,
    frontiers: Dfly2Frontiers,
    max_requests: usize,
}

impl<'a> Dfly2TapStore<'a> {
    fn layout(max_requests: usize) -> Result<WeightArenaLayout> {
        ensure!(
            (1..=cuteafd_core::V41_MAX_REQUESTS).contains(&max_requests),
            "invalid DFly2 tap store request limit {max_requests}"
        );
        let mut entries = BTreeMap::new();
        entries.insert("prev_hidden".to_string(), max_requests * HIDDEN_ROW_BYTES);
        for stage in 0..3 {
            entries.insert(format!("recall_{stage}"), max_requests * RECALL_ROW_BYTES);
        }
        WeightArenaLayout::new("v41/dfly2/taps", entries)
    }

    pub fn device_bytes(max_requests: usize) -> Result<usize> {
        Ok(Self::layout(max_requests)?.bytes())
    }

    pub fn new(device: Device<'a>, max_requests: usize) -> Result<Self> {
        let layout = Self::layout(max_requests)?;
        let arena = device.run(|| {
            DeviceArena::new(device, layout.clone(), ArenaTag::drafter_workspace("v41/dfly2/taps"))
        })?;
        let view = |name: &str| -> Result<Allocation<'a>> {
            let bytes = layout.view(name).context("missing DFly2 tap view")?.bytes();
            arena.view(name, bytes)
        };
        Ok(Self {
            device,
            prev_hidden: view("prev_hidden")?,
            recall: [view("recall_0")?, view("recall_1")?, view("recall_2")?],
            _arena: arena,
            frontiers: Dfly2Frontiers::new(max_requests),
            max_requests,
        })
    }

    pub fn device(&self) -> Device<'a> { self.device }
    pub fn admit(&mut self, id: u64) -> Result<usize> { self.frontiers.admit(id) }
    pub fn release(&mut self, id: u64) { self.frontiers.release(id); }
    pub fn invalidate(&mut self, id: u64) { self.frontiers.invalidate(id); }
    pub fn contains(&self, id: u64) -> bool { self.frontiers.contains(id) }
    pub fn ready_at(&self, id: u64, end: u64) -> bool { self.frontiers.ready_at(id, end) }
    pub fn committed_end(&self, id: u64) -> Option<u64> { self.frontiers.committed_end(id) }

    pub fn prev_hidden_row(&self, id: u64) -> Result<CuteafdDeviceBuffer> {
        let slot = self.frontiers.slot(id)?;
        slice_buf(self.prev_hidden.buffer, slot * HIDDEN_ROW_BYTES, HIDDEN_ROW_BYTES)
    }

    pub fn recall_row(&self, id: u64, stage: usize) -> Result<CuteafdDeviceBuffer> {
        ensure!(stage < 3, "invalid DFly2 recall stage {stage}");
        let slot = self.frontiers.slot(id)?;
        slice_buf(self.recall[stage].buffer, slot * RECALL_ROW_BYTES, RECALL_ROW_BYTES)
    }

    /// Enqueue per-request `prev_hidden` (10 KiB) and 3-layer `recall` (3 × 2 KiB)
    /// device-to-device copies on `stream` without yet advancing frontiers.
    /// Call [`Self::publish_frontiers`] once `stream` completes.
    ///
    /// # Safety
    /// Source buffers reside on `self.device` and stay live through `stream` completion.
    pub unsafe fn enqueue_commit_rows(
        &self,
        library: &NativeLibrary,
        updates: &[(u64, u64, CuteafdDeviceBuffer, [CuteafdDeviceBuffer; 3])],
        stream: *mut c_void,
    ) -> Result<()> {
        for &(id, _, src_hidden, src_recall) in updates {
            let slot = self.frontiers.slot(id)?;
            ensure!(slot < self.max_requests, "DFly2 tap slot out of bounds");
            ensure!(
                src_hidden.device_id == self.device.id && src_hidden.bytes >= HIDDEN_ROW_BYTES,
                "DFly2 commit prev_hidden device or size mismatch"
            );
            let dst_hidden = slice_buf(self.prev_hidden.buffer, slot * HIDDEN_ROW_BYTES, HIDDEN_ROW_BYTES)?;
            unsafe {
                library.copy_d2d_async(dst_hidden, src_hidden, HIDDEN_ROW_BYTES, stream)?;
            }
            for stage in 0..3 {
                ensure!(
                    src_recall[stage].device_id == self.device.id
                        && src_recall[stage].bytes >= RECALL_ROW_BYTES,
                    "DFly2 commit recall[{stage}] device or size mismatch"
                );
                let dst_recall = slice_buf(self.recall[stage].buffer, slot * RECALL_ROW_BYTES, RECALL_ROW_BYTES)?;
                unsafe {
                    library.copy_d2d_async(dst_recall, src_recall[stage], RECALL_ROW_BYTES, stream)?;
                }
            }
        }
        Ok(())
    }

    /// Publish committed frontiers (or invalidate requests whose last accepted row
    /// was absent from the target head selection) after the copy stream completes.
    pub fn publish_frontiers(&mut self, updates: &[(u64, Option<u64>)]) -> Result<()> {
        for &(id, new_end) in updates {
            match new_end {
                Some(end) => { self.frontiers.set_frontier(id, end)?; }
                None => self.frontiers.invalidate(id),
            }
        }
        Ok(())
    }

    /// Copy per-request committed `prev_hidden` (10 KiB) and 3-layer `recall`
    /// (3 × 2 KiB) into their persistent slot buffers on `stream` and publish
    /// the updated committed frontiers.
    ///
    /// # Safety
    /// Source buffers reside on `self.device` and stay live through `stream` completion.
    pub unsafe fn commit_rows(
        &mut self,
        library: &NativeLibrary,
        updates: &[(u64, u64, CuteafdDeviceBuffer, [CuteafdDeviceBuffer; 3])],
        stream: *mut c_void,
    ) -> Result<()> {
        unsafe { self.enqueue_commit_rows(library, updates, stream)?; }
        for &(id, new_end, _, _) in updates {
            self.frontiers.set_frontier(id, new_end)?;
        }
        Ok(())
    }
}
