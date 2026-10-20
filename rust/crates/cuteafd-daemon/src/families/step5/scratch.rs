//! Stable per-rank scratch views; each layer retires before the next reuses them.
use crate::shared::memory::device::{Allocation, Device};
use anyhow::{ensure, Context, Result};
use cuteafd_ffi::CuteafdDeviceBuffer;
use std::rc::Rc;

pub(super) trait BufferOwner {
    fn buffer(&self) -> CuteafdDeviceBuffer;
}
impl BufferOwner for Allocation<'_> {
    fn buffer(&self) -> CuteafdDeviceBuffer { self.buffer }
}

pub(super) struct Plane<'a> {
    pub buffer: CuteafdDeviceBuffer,
    _owner: Rc<Allocation<'a>>,
}
impl BufferOwner for Plane<'_> {
    fn buffer(&self) -> CuteafdDeviceBuffer { self.buffer }
}

#[derive(Clone)]
pub(super) struct Arena<'a>(Rc<Allocation<'a>>);
impl<'a> Arena<'a> {
    pub fn new(device: Device<'a>, bytes: usize) -> Result<Self> {
        let _scope = cuteafd_ffi::memory_ledger::scope("step5_scratch_workspace");
        Ok(Self(Rc::new(Allocation::new(device, bytes)?)))
    }

    pub fn poison_after(&self, keep: usize) -> Result<()> {
        ensure!(keep <= self.0.buffer.bytes, "Step-5 poison retained input extent");
        let poison = vec![0xff; (1 << 20).min(self.0.buffer.bytes - keep)];
        self.0.device.run(|| {
            for offset in (keep..self.0.buffer.bytes).step_by(1 << 20) {
                let bytes = poison.len().min(self.0.buffer.bytes - offset);
                let view = CuteafdDeviceBuffer { ptr: self.0.buffer.ptr.wrapping_byte_add(offset),
                    bytes, ..self.0.buffer };
                self.0.device.library.copy_h2d(view, &poison[..bytes])?;
            }
            Ok(())
        })
    }

    pub fn planes(&self, device: Device<'a>, sizes: &[usize], offset: usize) -> Result<Vec<Plane<'a>>> {
        ensure!(device.id == self.0.device.id
            && std::ptr::eq(device.library, self.0.device.library), "Step-5 scratch device mismatch");
        view_layout(self.0.buffer.bytes, sizes, offset)?.into_iter().map(|(offset, bytes)| {
            let buffer = CuteafdDeviceBuffer { ptr: self.0.buffer.ptr.wrapping_byte_add(offset),
                bytes, ..self.0.buffer };
            Ok(Plane { buffer, _owner: self.0.clone() })
        }).collect()
    }
}

fn view_layout(capacity: usize, sizes: &[usize], offset: usize) -> Result<Vec<(usize, usize)>> {
    ensure!(offset % 16 == 0, "Step-5 scratch view alignment");
    let mut next = offset;
    sizes.iter().map(|&bytes| {
        let start = next;
        let end = start.checked_add(bytes).context("Step-5 scratch view overflow")?;
        ensure!(end <= capacity, "Step-5 scratch view extent");
        next = end.checked_add(15).context("Step-5 scratch alignment overflow")? / 16 * 16;
        Ok((start, bytes))
    }).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use cuteafd_loader::plan::families::step5::native;

    #[test]
    fn arena_views_match_admission_and_preserve_residual_until_consumed() -> Result<()> {
        for rows in [1, 3, 16, 63, 64, 65, 256, 640, 1025, 4095, 4096] {
            let residual = rows * 4096 * 4;
            for attention in [native::swa_planes(rows, 32)?.to_vec(),
                native::swa_planes(rows, 64)?.to_vec(), native::csa_planes(rows)?.to_vec()] {
                let attention_bytes = usize::try_from(native::plane_bytes(&attention)?)?;
                for ffn in [native::dense_planes(rows)?.to_vec(), native::routed_planes(rows)?.to_vec()] {
                    let capacity = attention_bytes + usize::try_from(native::plane_bytes(&ffn)?)?;
                    let attention_views = view_layout(capacity, &attention, 0)?;
                    let ffn_views = view_layout(capacity, &ffn, attention_bytes)?;
                    assert_eq!(attention_views[0], (0, residual));
                    assert!(attention_views.iter().all(|&(offset, bytes)| offset + bytes <= attention_bytes));
                    assert!(ffn_views.iter().all(|&(offset, _)| offset >= attention_bytes));
                    let output = ffn_views.last().unwrap();
                    // The next layer copies this residual to offset zero before any reuse.
                    assert!(output.0 >= residual);
                    assert_eq!(output.1, residual);
                    // The head narrows into planes 0/1 before logits may overwrite the residual.
                    let head = native::head_planes(rows)?;
                    assert!(head[0] + head[1] <= output.0);
                }
            }
        }
        assert!(view_layout(1024, &[1025], 0).is_err());
        assert!(view_layout(1024, &[16], 1).is_err());
        assert!(view_layout(usize::MAX, &[usize::MAX], 0).is_err());
        Ok(())
    }
}

pub(super) fn planes<'a>(device: Device<'a>, sizes: &[usize],
    arena: Option<(&Arena<'a>, usize)>) -> Result<Vec<Plane<'a>>> {
    if let Some((arena, offset)) = arena { return arena.planes(device, sizes, offset); }
    sizes.iter().map(|&bytes| {
        let owner = Rc::new(Allocation::new(device, bytes)?);
        Ok(Plane { buffer: owner.buffer, _owner: owner })
    }).collect()
}
