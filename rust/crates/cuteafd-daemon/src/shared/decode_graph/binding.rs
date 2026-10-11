//! Exact binding adapter for waves that select one executable at a time.
use super::{GraphBank, GraphOwner};
use anyhow::Result;
use cuteafd_ffi::NativeLibrary;
use std::{ffi::c_void, hash::Hash};

pub(crate) struct BindingGraphs<'a, K> {
    library: &'a NativeLibrary,
    device: i32,
    bank: GraphBank<K, GraphOwner<'a, ()>>,
    current: Option<K>,
}

impl<'a, K: Hash + Eq + Clone + std::fmt::Debug> BindingGraphs<'a, K> {
    pub fn new(library: &'a NativeLibrary) -> Result<Self> {
        Ok(Self { library, device: library.cuda_get_device()?, bank: GraphBank::new(None), current: None })
    }
    pub fn is_none(&self) -> bool { self.current.is_none() }
    pub fn get(&self) -> Option<(*mut c_void, K)> {
        let key = self.current.as_ref()?;
        Some((self.bank.get(key)?.raw, key.clone()))
    }
    /// # Safety
    /// The containing wave pins every captured allocation and drains before
    /// clearing or replacing this binding. Capture itself enqueues no execution.
    pub unsafe fn insert(&mut self, key: K, raw: *mut c_void) -> Result<()> {
        // SAFETY: the containing wave retains weights, buffers and workspace.
        let owner = unsafe { GraphOwner::new(self.library, self.device, raw, ())? };
        self.bank.insert(key.clone(), owner, None);
        self.current = Some(key.clone());
        tracing::debug!(target: "cuteafd::graph_capture", site="dspark_binding", ?key,
            device=self.device, bank=self as *const Self as usize, "draft graph captured");
        Ok(())
    }
    /// # Safety
    /// Every stream using these captured pointers has successfully drained.
    pub unsafe fn clear(&mut self) -> Result<()> {
        self.bank.retire_all();
        self.current = None;
        drop(self.bank.drain_retired(|| Ok(()))?);
        Ok(())
    }
}
