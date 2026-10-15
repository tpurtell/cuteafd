//! Family provenance adapter for the shared coordinator-local expert executor.
use super::{exl3::Exl3Weights, ExpertWeights};
use crate::families::deepseek_v41::{
    v41_backbone_router::RouterOutput, v41_backbone_shared::SharedOutput,
};
use crate::shared::experts::local::LocalExpertWave as Execution;
use anyhow::{ensure, Result};
use cuteafd_ffi::{CuteafdDeviceBuffer, NativeLibrary};
use std::{path::Path, rc::Rc};

pub(crate) struct LocalExpertWave<'a>(Execution<'a>);
impl<'a> LocalExpertWave<'a> {
    pub fn device_bytes(library: &NativeLibrary, capacity: u32) -> Result<usize> {
        Execution::device_bytes(
            library,
            capacity,
            cuteafd_core::ExpertGeometry::DEEPSEEK_V41,
        )
    }
    pub fn device_bytes_for(library: &NativeLibrary, capacity: u32, nvfp4: bool) -> Result<usize> {
        Execution::device_bytes_for(
            library,
            capacity,
            nvfp4,
            cuteafd_core::ExpertGeometry::DEEPSEEK_V41,
        )
    }
    pub fn exl3_device_bytes(directory: &Path, capacity: u32) -> Result<usize> {
        Execution::exl3_device_bytes(
            directory,
            capacity,
            cuteafd_core::ExpertGeometry::DEEPSEEK_V41,
        )
    }
    pub fn new(
        library: &'a NativeLibrary,
        weights: Rc<Vec<ExpertWeights<'a>>>,
        capacity: u32,
        budget: usize,
    ) -> Result<Self> {
        Ok(Self(Execution::new(
            library,
            weights,
            capacity,
            budget,
            cuteafd_core::ExpertGeometry::DEEPSEEK_V41,
        )?))
    }
    /// # Safety
    /// The operator-selected package is trusted native code.
    pub unsafe fn new_exl3(
        library: &'a NativeLibrary,
        weights: Rc<Vec<Exl3Weights<'a>>>,
        directory: &Path,
        capacity: u32,
        budget: usize,
        modules: Vec<crate::shared::experts::exl3::execution::Exl3Modules<'a>>,
    ) -> Result<Self> {
        // SAFETY: the adapter forwards the trusted-package contract unchanged.
        Ok(Self(unsafe {
            Execution::new_exl3_with_modules(
                library,
                weights,
                directory,
                capacity,
                budget,
                cuteafd_core::ExpertGeometry::DEEPSEEK_V41,
                Some(modules),
            )?
        }))
    }
    pub fn contains(&self, layer: usize) -> bool {
        self.0.contains(layer)
    }
    fn inputs(
        &self,
        routed: &RouterOutput<'_>,
        shared: &SharedOutput<'_>,
    ) -> Result<[CuteafdDeviceBuffer; 3]> {
        ensure!(
            routed.binding()? == shared.binding()?
                && routed.layer == shared.layer
                && routed.rows == shared.rows,
            "local router and shared expert binding differ"
        );
        Ok([
            self.0
                .select_input(routed.rows, routed.input, routed.expert_input)?,
            routed.ids,
            routed.routing,
        ])
    }
    /// # Safety
    /// Router/shared outputs remain immutable until completion or failure drainage.
    pub unsafe fn execute(
        &mut self,
        routed: &RouterOutput<'_>,
        shared: &SharedOutput<'_>,
    ) -> Result<CuteafdDeviceBuffer> {
        let inputs = self.inputs(routed, shared)?;
        // SAFETY: provenance was checked; the caller retains both family outputs.
        unsafe {
            self.0
                .execute_buffers(routed.layer, routed.rows, inputs, shared.values)
        }
    }
    /// # Safety
    /// Same retained input contract as execute, including cancellation drainage.
    pub async unsafe fn execute_cooperative(
        &mut self,
        routed: &RouterOutput<'_>,
        shared: &SharedOutput<'_>,
    ) -> Result<CuteafdDeviceBuffer> {
        let inputs = self.inputs(routed, shared)?;
        // SAFETY: provenance was checked; the caller retains both family outputs.
        unsafe {
            self.0
                .execute_buffers_cooperative(routed.layer, routed.rows, inputs, shared.values)
                .await
        }
    }
}
