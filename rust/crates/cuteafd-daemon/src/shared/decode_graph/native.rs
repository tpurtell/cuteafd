//! Native graph adapters. The containing wave owns stable workspace/buffers and
//! drains them before insert/clear; GraphOwner retains weight references.
use super::{GraphBank, GraphOwner};
use anyhow::{ensure, Result};
use cuteafd_ffi::NativeLibrary;
use std::ffi::c_void;

/// Large (above-decode) shapes a bank retains per layer: the full prefill chunk
/// and the decoder replay. Other large shapes are executed eagerly upstream.
pub(crate) const LARGE_SHAPES: usize = 2;

pub(crate) struct LayerGraphs<'w, 'a, W> {
    library: &'a NativeLibrary,
    bank: GraphBank<(usize, u32), GraphOwner<'a, &'w W>>,
    current: [Option<u32>; 40],
    retain_small: bool,
    decode_rows: u32,
}

impl<'w, 'a, W> LayerGraphs<'w, 'a, W> {
    pub fn new(library: &'a NativeLibrary, decode_rows: u32) -> Self {
        Self { library, bank: GraphBank::new(None), current: [None; 40], retain_small: false, decode_rows }
    }
    pub fn get(&self, layer: usize, weights: &W) -> Option<(*mut c_void, u32)> {
        self.get_shape(layer, weights, self.current.get(layer).copied().flatten()?)
    }
    pub fn get_shape(&self, layer: usize, weights: &W, rows: u32) -> Option<(*mut c_void, u32)> {
        let owner = self.bank.get(&(layer, rows))?;
        std::ptr::eq(owner.pins, weights).then_some((owner.raw, rows))
    }
    pub fn enable_small_shapes(&mut self) { self.retain_small = true; }
    pub fn declare_startup(&mut self, keys: &[(usize, u32)]) -> Result<()> {
        ensure!(keys.iter().all(|&(layer, rows)| layer < 40 && (1..=self.decode_rows).contains(&rows)),
            "startup layer graph key exceeds decode geometry");
        self.bank.declare_startup(keys)?;
        self.retain_small = true;
        Ok(())
    }
    pub fn finish_startup(&mut self) -> Result<()> { self.bank.finish_startup() }
    pub fn capture_allowed(&self, layer: usize, rows: u32) -> bool {
        self.bank.capture_allowed(&(layer, rows))
    }
    /// # Safety
    /// All launches using the containing wave's storage have drained. The caller
    /// retains that storage until clear. On failure caller still owns graph.
    pub unsafe fn insert(&mut self, layer: usize, weights: &'w W, rows: u32, graph: *mut c_void) -> Result<()> {
        ensure!(layer < 40 && (1..=4096).contains(&rows) && !graph.is_null(), "invalid layer graph binding");
        ensure!(self.capture_allowed(layer, rows), "layer graph capture outside startup inventory or after seal");
        let different_owner = self.bank.count(|(l, _)| *l == layer) > 0
            && self.current[layer].and_then(|r| self.bank.get(&(layer, r)))
                .is_some_and(|owner| !std::ptr::eq(owner.pins, weights));
        if !self.retain_small || different_owner {
            // SAFETY: caller has drained all graph launches before replacing an owner.
            unsafe { self.remove(layer)?; }
        } else if rows > self.decode_rows {
            // Large (prefill) graphs per layer: keep the most recent LARGE_SHAPES
            // (a full chunk and the decoder replay), so neither decode rounds nor
            // alternating prefill units recapture them on every request.
            let decode_rows = self.decode_rows;
            let large = self.bank.count(|&(l, r)| l == layer && r > decode_rows && r != rows);
            if large >= LARGE_SHAPES {
                self.bank.retire_oldest(|&(l, r)| l == layer && r > decode_rows && r != rows);
            }
        }
        let device = self.library.cuda_get_device()?;
        // SAFETY: the weight borrow pins its storage and the containing wave pins workspace.
        let owner = unsafe { GraphOwner::new(self.library, device, graph, weights)? };
        self.bank.insert((layer, rows), owner, None);
        self.current[layer] = Some(rows);
        drop(self.bank.drain_retired(|| Ok(()))?);
        tracing::debug!(target: "cuteafd::graph_capture", layer, rows, device,
            bank = self as *const Self as usize, owner = std::any::type_name::<W>(),
            binding = weights as *const W as usize,
            retained = self.bank.count(|(l, _)| *l == layer), "native layer graph captured");
        Ok(())
    }
    /// # Safety
    /// Every launch of this layer's graph completed on every consumer stream.
    pub unsafe fn remove(&mut self, layer: usize) -> Result<()> {
        ensure!(layer < 40, "invalid layer graph index");
        self.bank.retire_matching(|(l, _)| *l == layer);
        self.current[layer] = None;
        drop(self.bank.drain_retired(|| Ok(()))?);
        Ok(())
    }
    /// # Safety
    /// Every queued use of this wave's captured pointers completed.
    pub unsafe fn clear(&mut self) -> Result<()> {
        self.bank.retire_all();
        self.current = [None; 40];
        drop(self.bank.drain_retired(|| Ok(()))?);
        Ok(())
    }
    #[cfg(test)]
    pub fn len(&self) -> usize { self.bank.len() }
}

pub(crate) struct RowGraphs<'a> {
    library: &'a NativeLibrary,
    bank: GraphBank<usize, GraphOwner<'a, ()>>,
    decode_rows: usize,
    site: &'static str,
}
impl<'a> RowGraphs<'a> {
    pub fn new(library: &'a NativeLibrary, site: &'static str, decode_rows: usize) -> Self {
        Self { library, bank: GraphBank::new(None), decode_rows, site }
    }
    pub fn get(&self, rows: usize) -> Option<*mut c_void> { self.bank.get(&rows).map(|owner| owner.raw) }
    pub fn declare_startup(&mut self, rows: &[usize]) -> Result<()> {
        ensure!(rows.iter().all(|r| (1..=self.decode_rows).contains(r)),
            "startup row graph key exceeds decode geometry");
        self.bank.declare_startup(rows)
    }
    pub fn finish_startup(&mut self) -> Result<()> { self.bank.finish_startup() }
    pub fn capture_allowed(&self, rows: usize) -> bool { self.bank.capture_allowed(&rows) }
    /// # Safety
    /// All launches on this wave's storage drained. The wave retains weights,
    /// workspace, geometry and buffers until clear, then drops them afterward.
    /// Whether inserting `rows` retires (destroys) another graph, whose
    /// queued launches must have drained first.
    pub fn would_retire(&self, rows: usize) -> bool {
        let decode_rows = self.decode_rows;
        rows > decode_rows && self.bank.count(|&r| r > decode_rows && r != rows) >= LARGE_SHAPES
    }
    pub unsafe fn insert(&mut self, rows: usize, graph: *mut c_void) -> Result<()> {
        ensure!((1..=4096).contains(&rows) && !graph.is_null(), "invalid row graph binding");
        ensure!(self.capture_allowed(rows), "row graph capture outside startup inventory or after seal");
        let device = self.library.cuda_get_device()?;
        // SAFETY: the containing wave owns and pins every captured allocation.
        let owner = unsafe { GraphOwner::new(self.library, device, graph, ())? };
        if rows > self.decode_rows {
            let decode_rows = self.decode_rows;
            if self.bank.count(|&r| r > decode_rows && r != rows) >= LARGE_SHAPES {
                self.bank.retire_oldest(|&r| r > decode_rows && r != rows);
            }
        }
        self.bank.insert(rows, owner, None);
        drop(self.bank.drain_retired(|| Ok(()))?);
        tracing::debug!(target: "cuteafd::graph_capture", site = self.site, rows, device,
            bank = self as *const Self as usize, retained = self.bank.len(), "native row graph captured");
        Ok(())
    }
    /// # Safety
    /// Every queued use completed before releasing graph executables or storage.
    pub unsafe fn clear(&mut self) -> Result<()> {
        self.bank.retire_all();
        drop(self.bank.drain_retired(|| Ok(()))?);
        Ok(())
    }
    #[cfg(test)]
    pub fn len(&self) -> usize { self.bank.len() }
}
