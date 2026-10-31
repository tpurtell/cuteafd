//! The per-expert partial-reply epilogue on the shared EXL3 worker: for a
//! request flagged `EXPERT_PROTOCOL_V2_FLAG_PARTIAL_REPLIES`, the core runs as
//! always and, instead of the weighted top-k sum, `expert_partial_{bf16,nvfp4,
//! fp8}` writes each routed (row, slot)'s unweighted partial (inverse H128
//! rotation and svh, as the sum's own helper) into the region its row
//! descriptor names. The coordinator computed the regions; the worker only
//! checks them against the reply slot and writes. Zero-row regions are
//! skipped. Models without a per-expert output norm never send the flag, so
//! their weighted path is untouched.
use anyhow::{ensure, Context, Result};
use cuteafd_ffi::{programs::{Program, Programs, Scalar}, CuteafdDeviceBuffer, NativeLibrary};
use cuteafd_transport::{PartialReplyPlan, PartialWire};
use std::ffi::c_void;

const POINTERS: [&str; 6] = ["fc2", "svh", "ids", "expert_map", "row_map", "out"];

/// The bound epilogue programs and the row-map upload buffer.
pub(crate) struct PartialEpilogue<'a> {
    bf16: Program<'a>,
    nvfp4: Program<'a>,
    fp8: Program<'a>,
    /// Device row maps (BF16 region, then the wide region), `capacity` u32 each,
    /// uploaded from pinned staging (reused only after the worker's stream drains,
    /// as its route staging is).
    row_maps: crate::shared::memory::DeviceAllocation<'a>,
    staging: crate::shared::memory::HostAllocation<'a>,
    capacity: usize,
    library: &'a NativeLibrary,
}

/// The two region row maps of a plan, in ascending wave-row order.
pub(crate) fn row_maps(plan: &PartialReplyPlan) -> (Vec<u32>, Vec<u32>) {
    let (mut bf16, mut wide) = (Vec::new(), Vec::new());
    for (row, &region) in plan.regions.iter().enumerate() {
        if region == 0 { bf16.push(row as u32) } else { wide.push(row as u32) }
    }
    (bf16, wide)
}

impl<'a> PartialEpilogue<'a> {
    /// Resolve the three programs before accepting requests (startup refuses a
    /// missing or drifted export before any allocation of the worker).
    pub(crate) fn load(library: &'a NativeLibrary, programs: &'a Programs<'a>, capacity: usize) -> Result<Self> {
        let program = |name: &str| programs.program(name, &POINTERS)
            .with_context(|| format!("partial-reply epilogue {name}"));
        let bf16 = program("expert_partial_bf16")?;
        let nvfp4 = program("expert_partial_nvfp4")?;
        let fp8 = program("expert_partial_fp8")?;
        let row_maps = crate::shared::memory::DeviceAllocation::named(library, 2 * capacity * 4,
            "experts/workspace/partial_row_maps")?;
        let staging = crate::shared::memory::HostAllocation::named(library, 2 * capacity * 4,
            "experts/workspace/partial_row_map_staging")?;
        Ok(Self { bf16, nvfp4, fp8, row_maps, staging, capacity, library })
    }

    /// Write `plan`'s regions of `rows` rows into `out` (the reply payload).
    ///
    /// # Safety
    /// `sources` are the core's live FC2/svh/map of this layer on the current
    /// device; `ids` the request's staged route ids; `out` an exclusive device
    /// or mapped-host slot of at least the plan's payload, live through `stream`.
    pub(crate) unsafe fn launch(&mut self, plan: &PartialReplyPlan, rows: usize, topk: usize, width: usize,
        sources: super::execution::PartialSources, ids: CuteafdDeviceBuffer, out: CuteafdDeviceBuffer,
        stream: *mut c_void) -> Result<()> {
        ensure!(rows > 0 && rows <= self.capacity && plan.regions.len() == rows,
            "partial-reply plan of {} regions for {rows} rows (capacity {})", plan.regions.len(), self.capacity);
        let (bf16_at, wide_at, wide_bytes, total) = plan.payload(topk, width)
            .map(|(bf16, at, wide, total)| (bf16, at, wide, total))?;
        ensure!(out.bytes >= total, "partial-reply slot of {} bytes under the {total}-byte payload", out.bytes);
        let (bf16_rows, wide_rows) = row_maps(plan);
        let maps = self.row_maps.buffer;
        let capacity = self.capacity;
        {
            let staged = self.staging.bytes_mut();
            for (index, map) in [&bf16_rows, &wide_rows].into_iter().enumerate() {
                for (i, row) in map.iter().enumerate() {
                    staged[(index * capacity + i) * 4..][..4].copy_from_slice(&row.to_le_bytes());
                }
            }
        }
        let wide_program = match plan.wide { PartialWire::Nvfp4 => &self.nvfp4, PartialWire::Fp8 => &self.fp8 };
        let regions = [(&self.bf16, bf16_rows, 0usize, bf16_at), (wide_program, wide_rows, wide_at, wide_bytes)];
        for (index, (program, map, offset, bytes)) in regions.into_iter().enumerate() {
            if map.is_empty() { continue; }
            let map_buffer = CuteafdDeviceBuffer { ptr: maps.ptr.cast::<u8>().wrapping_add(index * capacity * 4).cast(),
                bytes: map.len() * 4, ..maps };
            let source = cuteafd_ffi::CuteafdHostBuffer { ptr: self.staging.buffer.ptr.cast::<u8>()
                .wrapping_add(index * capacity * 4).cast(), bytes: map.len() * 4, ..self.staging.buffer };
            // SAFETY: pinned staging and the map buffer are this epilogue's own allocations; the
            // worker drains `stream` before the next request rewrites the staging.
            unsafe { self.library.copy_host_buffer_h2d_async(map_buffer, source, map.len() * 4, stream)?; }
            let region = CuteafdDeviceBuffer { ptr: out.ptr.cast::<u8>().wrapping_add(offset).cast(), bytes, ..out };
            let pointers = [sources.fc2.ptr, sources.svh.ptr, ids.ptr, sources.expert_map.ptr, map_buffer.ptr, region.ptr];
            let scalars = [Scalar::I32(i32::try_from(map.len())?), Scalar::I32(i32::try_from(rows)?),
                Scalar::I32(i32::try_from(sources.local_experts)?), Scalar::I32(i32::try_from(sources.experts)?)];
            // SAFETY: forwarded from the caller; counts and scalar kinds are checked by `launch`.
            unsafe { program.launch(&pointers, &scalars, stream)?; }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn row_maps_partition_the_wave_by_region() {
        let plan = PartialReplyPlan { wide: PartialWire::Nvfp4, regions: vec![0, 1, 1, 0, 1] };
        assert_eq!(row_maps(&plan), (vec![0, 3], vec![1, 2, 4]));
        let empty = PartialReplyPlan { wide: PartialWire::Fp8, regions: vec![0; 3] };
        assert_eq!(row_maps(&empty), (vec![0, 1, 2], vec![]));
    }
}
