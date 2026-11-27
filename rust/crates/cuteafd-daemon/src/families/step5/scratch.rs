//! Stable per-rank scratch views; each layer retires before the next reuses them.
use crate::shared::memory::{device::{Allocation, Device, Stream}, Tag};
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
impl Plane<'_> {
    pub(super) fn inventory(&self, inventory: &mut crate::shared::memory::receipts::Inventory<'_>, name: &str) -> Result<()> {
        let first = inventory.receipts.len();
        inventory.backed_device(name, self.buffer, self._owner.buffer)?;
        for receipt in &mut inventory.receipts[first..] {
            receipt.allocation.category = cuteafd_core::memory_layout::Category::Workspace;
            receipt.allocation.group = "step5 attention workspace".into();
        }
        Ok(())
    }
}
impl BufferOwner for Plane<'_> {
    fn buffer(&self) -> CuteafdDeviceBuffer { self.buffer }
}

/// One Workspace backing: target and MTP per-step scratch reuse it in stream
/// order, after live outputs have been copied to retained cache/history storage.
#[derive(Clone)]
pub(super) struct Arena<'a>(Rc<Allocation<'a>>, Rc<crate::shared::memory::StandaloneStream<'a>>);
impl<'a> Arena<'a> {
    pub fn new(device: Device<'a>, bytes: usize, tag: Tag) -> Result<Self> {
        Ok(Self(Rc::new(Allocation::new(device, bytes, tag)?), Rc::new(crate::shared::memory::StandaloneStream::new(device))))
    }

    pub(super) fn standalone(&self) -> Rc<crate::shared::memory::StandaloneStream<'a>> {
        self.1.clone()
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

/// Shared writer, with private target-lane or arena-backed predictor planes.
/// Reuse is safe only in stream order, after copies into retained cache storage.
pub(super) struct CompactKv<'a> {
    planes: Vec<Plane<'a>>,
    writers: Vec<(usize, cuteafd_ffi::programs::Program<'a>)>,
    device: Device<'a>,
    rows: usize,
}
impl<'a> CompactKv<'a> {
    pub(super) fn new(device: Device<'a>, programs: &'a cuteafd_ffi::programs::Programs<'a>,
        rows: usize, heads: &[usize], sizes: [usize; 2], arena: Option<(&Arena<'a>, usize)>,
        tag: Tag) -> Result<Self> {
        Self::with_planes(device, programs, rows, heads, planes(device, &sizes, arena, tag)?)
    }

    pub(super) fn with_planes(device: Device<'a>, programs: &'a cuteafd_ffi::programs::Programs<'a>,
        rows: usize, heads: &[usize], planes: Vec<Plane<'a>>) -> Result<Self> {
        use cuteafd_loader::families::qwen4::Qwen4KvCache;
        ensure!(rows > 0 && !heads.is_empty() && planes.len() == 2, "Step-5 compact producer capacity");
        let sizes = planes.iter().map(|plane| plane.buffer.bytes).collect::<Vec<_>>();
        let writers = heads.iter().copied().map(|heads| {
            ensure!(matches!(heads, 32 | 64), "Step-5 compact producer heads");
            let groups = heads / 16;
            let name = if heads == 32 { "step52_pack_fp8_kv_plane" } else { "step5_pack_fp8_kv_plane" };
            let spec = programs.spec(name)?;
            let record = Qwen4KvCache::Fp8.plane_record_bytes(groups, 192);
            ensure!(sizes.iter().all(|&n| rows.checked_mul(record).is_some_and(|bytes| bytes <= n))
                && spec.pointers == ["x", "out"] && spec.scalar_kinds == b"i"
                && spec.capacity_rows as usize >= rows && spec.scratch.is_empty()
                && spec.geometry["kv_heads"] == groups && spec.geometry["head_dim"] == 192
                && spec.geometry["kv_format"] == "fp8" && spec.geometry["record_bytes"] == record
                && spec.geometry["scale_offset"] == groups * 192,
                "{name}: compact producer ABI/geometry/capacity");
            Ok((heads, device.run(|| programs.program(name, &["x", "out"]))?))
        }).collect::<Result<Vec<_>>>()?;
        Ok(Self { planes, writers, device, rows })
    }

    pub(super) fn enqueue(&self, stream: &Stream<'a>, sources: [CuteafdDeviceBuffer; 2],
        rows: usize, heads: usize) -> Result<[CuteafdDeviceBuffer; 2]> {
        use cuteafd_ffi::programs::Scalar;
        let (_, writer) = self.writers.iter().find(|(count, _)| *count == heads)
            .context("Step-5 compact producer head geometry")?;
        ensure!((1..=self.rows).contains(&rows) && stream.device.id == self.device.id
            && std::ptr::eq(stream.device.library, self.device.library),
            "Step-5 compact producer stream/capacity");
        let bytes = rows * (heads / 16) * 192 * 2;
        ensure!(sources.iter().all(|source| !source.ptr.is_null()
            && source.device_id == self.device.id && source.bytes >= bytes),
            "Step-5 compact producer BF16 source extent/device");
        self.device.run(|| {
            for (source, target) in sources.iter().zip(&self.planes) {
                // SAFETY: validated BF16 projections and compact planes are borrowed
                // through the caller's cancellation guard and stream drain.
                unsafe { writer.launch(&[source.ptr, target.buffer.ptr],
                    &[Scalar::I32(rows as i32)], stream.raw)?; }
            }
            Ok(())
        })?;
        Ok([self.planes[0].buffer, self.planes[1].buffer])
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
    fn arena_components_share_one_lazy_standalone_stream() -> Result<()> {
        let (_directory, fixture) = crate::shared::memory::cpu_stream_fixture()?;
        let library = fixture.load()?;
        let device = Device { library: &library, id: 0 };
        let arena = Arena::new(device, 256, crate::shared::memory::TEST)?;
        let first = standalone(device, Some((&arena, 0)));
        let second = standalone(device, Some((&arena, 128)));
        assert!(Rc::ptr_eq(&first, &second));
        first.drain()?;
        assert!(!fixture.events()?.contains('V'));
        assert_eq!(first.get()?.raw, second.get()?.raw);
        assert_eq!(fixture.events()?.matches('V').count(), 1);
        let lane = standalone(device, None);
        assert_ne!(lane.get()?.raw, first.get()?.raw);
        assert_eq!(fixture.events()?.matches('V').count(), 2);
        Ok(())
    }

    #[test]
    fn bounded_head_arena_keeps_projection_backing_outside_phase_maximum() -> Result<()> {
        use cuteafd_loader::formats::trellis_core::ProjectionScratchLayout;
        let (unit, span) = (2752, 3 * 2752);
        let mut projection = ProjectionScratchLayout::new(13824, 4096, unit)?;
        for (k, n) in [(4096, 13824), (4096, 768), (12288, 4096), (4096, 12288)] {
            projection.merge(&ProjectionScratchLayout::new(k, n, unit)?)?;
        }
        let projection = projection.packed_bytes()? as u64;
        assert_eq!(projection, 257_851_656);
        let layer = native::layer_workspace(unit, span, true, 64)?;
        let head_planes = native::head_planes(span)?;
        let head = native::plane_bytes(&head_planes)?;
        let total = native::with_arena_phase(layer + projection, projection, head)?;
        let arena = usize::try_from(total - projection)?;
        assert_eq!(total, layer.max(head) + projection);
        view_layout(arena, &head_planes, 0)?;
        for depth in [0, 1, 3] {
            let predictor = native::mtp_forecast_workspace(unit, depth,
                cuteafd_loader::families::qwen4::Qwen4KvCache::Fp8)?;
            let admitted = native::with_arena_phase(total, projection, predictor)?;
            assert_eq!(admitted - projection, (arena as u64).max(predictor));
        }
        Ok(())
    }

    #[test]
    fn output_scratch_is_disjoint_and_inventoried_once_per_backing() -> Result<()> {
        for rows in [1, 64, 65, 4096] {
            for sizes in [native::swa_planes(rows, 32)?.to_vec(),
                native::swa_planes(rows, 64)?.to_vec(), native::csa_planes(rows)?.to_vec()] {
                let views = view_layout(native::plane_bytes(&sizes)? as usize, &sizes, 0)?;
                let &(offset, bytes) = views.last().unwrap();
                assert_eq!(bytes, native::OUTPUT_SCRATCH_BYTES);
                assert!(views[..views.len()-1].iter().all(|&(start, size)| start + size <= offset));
            }
        }
        let (_directory, fixture) = crate::shared::memory::cpu_stream_fixture()?;
        let library = fixture.load()?;
        let device = Device { library: &library, id: 0 };
        let arena = Arena::new(device, 512, crate::shared::memory::Tag::workspace("step5/test-output-scratch"))?;
        let plane = arena.planes(device, &[256], 256)?.remove(0);
        use cuteafd_loader::placement::planner::{CoordinatorDevices, Lifetime};
        use cuteafd_core::memory_layout::Category;
        let ids = [0];
        let mut inventory = crate::shared::memory::receipts::Inventory::new(
            CoordinatorDevices::new(&ids)?, "step5", Lifetime::RESIDENT, Category::Weights, "test");
        plane.inventory(&mut inventory, "lane0.output-scratch")?;
        plane.inventory(&mut inventory, "lane1.output-scratch")?;
        assert_eq!(inventory.receipts.iter().filter(|r| r.allocation.view.is_none()).count(), 1);
        assert_eq!(inventory.receipts.iter().filter(|r| r.allocation.view.is_some()).count(), 2);
        assert!(inventory.receipts.iter().all(|r| r.allocation.category == Category::Workspace));
        assert!(inventory.receipts.iter().filter_map(|r| r.allocation.view.as_ref()).all(|v| v.offset == 256));
        Ok(())
    }

    #[test]
    fn predictor_arena_spans_preserve_live_inputs_output_and_selected_tokens() -> Result<()> {
        let disjoint = |a: (usize, usize), b: (usize, usize)|
            a.0 + a.1 <= b.0 || b.0 + b.1 <= a.0;
        use cuteafd_loader::families::qwen4::Qwen4KvCache;
        for (rows, kv_format) in [1, 3, 16, 63, 64, 65, 256, 640, 1025, 4095, 4096]
            .into_iter().flat_map(|rows| [Qwen4KvCache::Bf16, Qwen4KvCache::Fp8]
                .into_iter().map(move |format| (rows, format))) {
            let predictor = usize::try_from(native::mtp_forecast_workspace(rows, 3, kv_format)?)?;
            let attention_sizes = native::swa_planes(rows, 64)?;
            let attention_bytes = usize::try_from(native::plane_bytes(&attention_sizes)?)?;
            let input_offset = usize::try_from(native::mtp_input_offset(rows)?)?;
            let metadata_offset = usize::try_from(native::mtp_metadata_offset(rows)?)?;
            let core = usize::try_from(native::mtp_scratch_core(rows, kv_format)?)?;
            let attention = view_layout(predictor, &attention_sizes, 0)?;
            let dense = native::dense_layout(rows, rows)?.map(|(offset, bytes)| (attention_bytes + offset, bytes)).to_vec();
            let inputs = view_layout(predictor, &native::mtp_input_planes(rows)?, input_offset)?;
            let metadata_sizes = native::mtp_metadata_planes(rows)?;
            let metadata = view_layout(predictor, &metadata_sizes[..4], metadata_offset)?;
            let head = view_layout(predictor, &native::head_planes(native::head_rows(rows))?, 0)?;
            let selected = (0..3).map(|stage| view_layout(predictor, &metadata_sizes[4..],
                core + stage * 256).map(|views| views[0])).collect::<Result<Vec<_>>>()?;
            assert_eq!(input_offset, attention_bytes + usize::try_from(native::ffn_region_bytes(rows, rows)?)?);
            let compact = if kv_format == Qwen4KvCache::Fp8 {
                view_layout(predictor, &native::mtp_compact_planes(rows, kv_format)?,
                    usize::try_from(native::mtp_compact_offset(rows)?)?)?
            } else { Vec::new() };
            let live_core = attention.iter().chain(&dense).chain(&inputs).chain(&metadata)
                .chain(&compact).copied().collect::<Vec<_>>();
            for (i, &a) in live_core.iter().enumerate() {
                assert!(a.0 + a.1 <= core);
                assert!(live_core[i + 1..].iter().all(|&b| disjoint(a, b)));
            }
            // FFN consumes attention POST before writing its own disjoint output.
            let output = dense[5];
            assert!(attention.iter().all(|&span| disjoint(output, span)));
            // Head reads the F32 output before its later logits overwrite scratch.
            assert!(head[..2].iter().all(|&span| disjoint(output, span)));
            for (i, &token) in selected.iter().enumerate() {
                assert!(live_core.iter().chain(&head).all(|&span| disjoint(token, span)));
                assert!(selected[i + 1..].iter().all(|&other| disjoint(token, other)));
            }
            // Target outputs can alias predictor scratch only after the backend's
            // history copy and the scheduler's logit selection have retired them.
            for (full, heads) in [(false, 32), (false, 64), (true, 64)] {
                let target = usize::try_from(native::layer_workspace(rows, rows, full, heads)?)?;
                let capacity = target.max(predictor);
                for &(offset, bytes) in live_core.iter().chain(&head).chain(&selected) {
                    assert!(offset + bytes <= capacity);
                }
            }
            if rows == 3 {
                let mut scratch = vec![0xff; predictor];
                let mut history = Vec::new();
                for (stage, &(token, _)) in selected.iter().enumerate() {
                    // Poison every retired plane, preserving only earlier selected ids.
                    scratch[..core].fill(0xff);
                    let residual = inputs[5];
                    scratch[residual.0..residual.0 + residual.1].fill(0x20 + stage as u8);
                    let input = attention[0];
                    scratch.copy_within(residual.0..residual.0 + input.1, input.0);
                    scratch[input.0 + input.1..core].fill(0xff);
                    assert!(scratch[input.0..input.0 + input.1].iter()
                        .all(|&byte| byte == 0x20 + stage as u8));
                    scratch[output.0..output.0 + output.1].fill(0x40 + stage as u8);
                    history.push(scratch[output.0..output.0 + output.1].to_vec());
                    // A private history copy, not target scratch, feeds the next stage.
                    scratch[..core].fill(0xff);
                    scratch[token..token + 4].copy_from_slice(&(stage as u32 + 17).to_le_bytes());
                    for (prior, &(offset, _)) in selected[..=stage].iter().enumerate() {
                        assert_eq!(&scratch[offset..offset + 4], &(prior as u32 + 17).to_le_bytes());
                        assert!(history[prior].iter().all(|&byte| byte == 0x40 + prior as u8));
                    }
                }
            }
        }
        Ok(())
    }

    #[test]
    fn arena_views_match_admission_and_preserve_residual_until_consumed() -> Result<()> {
        for rows in [1, 3, 16, 63, 64, 65, 256, 640, 1025, 2048, 4095, 4096] {
            let residual = rows * 4096 * 4;
            let largest = usize::try_from(native::attention_bytes(rows, true, 64)?)?;
            let region = usize::try_from(native::ffn_region_bytes(rows, rows)?)?;
            for attention in [native::swa_planes(rows, 32)?.to_vec(),
                native::swa_planes(rows, 64)?.to_vec(), native::csa_planes(rows)?.to_vec()] {
                let attention_bytes = usize::try_from(native::plane_bytes(&attention)?)?;
                assert!(attention_bytes <= largest);
                let attention_views = view_layout(largest + region, &attention, 0)?;
                assert_eq!(attention_views[0], (0, residual));
                // Attention never reaches the shared FFN region at the rank's offset.
                assert!(attention_views.iter().all(|&(offset, bytes)| offset + bytes <= largest));
            }
            for ffn in [native::dense_layout(rows, rows)?.to_vec(), native::routed_layout(rows, rows, false)?.to_vec(),
                native::routed_layout(rows, rows, true)?.to_vec()] {
                let views = ffn.iter().map(|&(offset, bytes)| (largest + offset, bytes)).collect::<Vec<_>>();
                assert!(views.iter().all(|&(offset, bytes)| offset >= largest && offset + bytes <= largest + region));
                // Every kind's OUTPUT is the region's first plane.
                let output = *views.iter().min().unwrap();
                assert_eq!(output, (largest, residual));
                // The next layer copies this residual to offset zero before any reuse.
                assert!(output.0 >= residual);
                // The head narrows into planes 0/1 before logits may overwrite the residual.
                let head = native::head_planes(native::head_rows(rows))?;
                assert!(head[0] + head[1] <= output.0);
            }
        }
        assert!(view_layout(1024, &[1025], 0).is_err());
        assert!(view_layout(1024, &[16], 1).is_err());
        assert!(view_layout(usize::MAX, &[usize::MAX], 0).is_err());
        Ok(())
    }
}

/// FFN planes at the shared region offsets (`native::dense_layout`/`routed_layout`)
/// when arena-backed, so every FFN kind keeps its live planes at common offsets.
pub(super) fn region_planes<'a>(device: Device<'a>, spans: &[(usize, usize)],
    arena: Option<(&Arena<'a>, usize)>, tag: Tag) -> Result<Vec<Plane<'a>>> {
    let Some((arena, base)) = arena else {
        return planes(device, &spans.iter().map(|&(_, bytes)| bytes).collect::<Vec<_>>(), None, tag);
    };
    ensure!(device.id == arena.0.device.id && std::ptr::eq(device.library, arena.0.device.library),
        "Step-5 scratch device mismatch");
    spans.iter().map(|&(offset, bytes)| {
        let start = base.checked_add(offset).context("Step-5 region view overflow")?;
        ensure!(start % 16 == 0 && start.checked_add(bytes).is_some_and(|end| end <= arena.0.buffer.bytes),
            "Step-5 region view extent");
        Ok(Plane { buffer: CuteafdDeviceBuffer { ptr: arena.0.buffer.ptr.wrapping_byte_add(start), bytes,
            ..arena.0.buffer }, _owner: arena.0.clone() })
    }).collect()
}

pub(super) fn planes<'a>(device: Device<'a>, sizes: &[usize],
    arena: Option<(&Arena<'a>, usize)>, tag: Tag) -> Result<Vec<Plane<'a>>> {
    if let Some((arena, offset)) = arena { return arena.planes(device, sizes, offset); }
    sizes.iter().map(|&bytes| {
        let owner = Rc::new(Allocation::new(device, bytes, tag)?);
        Ok(Plane { buffer: owner.buffer, _owner: owner })
    }).collect()
}

/// Arena users share one standalone stream; a separately loaded owner/lane has
/// its own lazy context. No CUDA stream is created while serving owners load.
pub(super) fn standalone<'a>(device: Device<'a>, arena: Option<(&Arena<'a>, usize)>)
    -> Rc<crate::shared::memory::StandaloneStream<'a>> {
    arena.map_or_else(|| Rc::new(crate::shared::memory::StandaloneStream::new(device)),
        |(arena, _)| arena.standalone())
}
