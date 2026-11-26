//! Native TP1 routed experts for Inkling 2 Mini: NVFP4 W4A4 (layers 0..20) + MXFP8 W8A8 (layers 20..24).
use crate::shared::memory::{device::Stream, receipts::Inventory};
use anyhow::{ensure, Context, Result};
use cuteafd_ffi::CuteafdDeviceBuffer as Buffer;
use cuteafd_loader::placement::families::inkling2_mini::LocalBacking;
use std::{path::Path, rc::Rc};

const HIDDEN: usize = 2304;
const WIRE_ROW_BYTES: usize = 2376; // 2304 + 2304/32

pub(crate) trait Dispatcher {
    fn wire_input(&self) -> bool;
    /// Executes routed experts and returns the BF16 device buffer holding `[rows, 2304]` output.
    /// When the backend already produces BF16 directly in its persistent workspace, it returns
    /// that view without an intermediate `copy_rows` launch.
    fn enqueue(
        &self,
        input: Buffer,
        ids: Buffer,
        weights: Buffer,
        output: Buffer,
        rows: usize,
        stream: &Stream<'_>,
    ) -> Result<Buffer>;
    fn inventory(&self, inventory: &mut Inventory<'_>, prefix: &str) -> Result<()>;
}

pub(crate) struct Resident<'a> {
    layer: usize,
    backend: Rc<std::cell::RefCell<Box<dyn crate::shared::experts::rtx::RtxExpertLayer + 'a>>>,
    stream: Rc<Stream<'a>>,
    copy: Rc<cuteafd_ffi::DeviceRowCopy<'a>>,
}

impl<'a> Resident<'a> {
    pub fn new(
        layer: usize,
        backend: Box<dyn crate::shared::experts::rtx::RtxExpertLayer + 'a>,
        stream: Rc<Stream<'a>>,
    ) -> Result<Self> {
        ensure!(
            backend.shard() == crate::shared::experts::rtx::RtxShard::Whole
                && backend.layers().contains(&layer)
                && backend.device() == stream.device.id,
            "Inkling 2 Mini resident adapter requires a whole TP1 native backend"
        );
        let element = if backend.partial() == crate::shared::experts::rtx::PartialDtype::F32 { 4 } else { 2 };
        ensure!(
            !backend.output().is_null() && backend.workspace_bytes() >= HIDDEN * element,
            "Inkling 2 Mini resident workspace cannot hold one output row"
        );
        let copy = Rc::new(stream.device.run(|| stream.device.library.device_row_copy())?);
        Ok(Self {
            layer,
            backend: Rc::new(std::cell::RefCell::new(backend)),
            stream,
            copy,
        })
    }

    pub fn for_layers(
        backend: Box<dyn crate::shared::experts::rtx::RtxExpertLayer + 'a>,
        stream: Rc<Stream<'a>>,
    ) -> Result<Vec<Self>> {
        let layers = backend.layers().to_vec();
        ensure!(
            layers.windows(2).all(|pair| pair[0] < pair[1]),
            "Inkling 2 Mini resident layer inventory is not ordered"
        );
        let first = *layers.first().context("Inkling 2 Mini resident backend has no layers")?;
        let owner = Self::new(first, backend, stream)?;
        Ok(layers
            .into_iter()
            .map(|layer| Self {
                layer,
                backend: owner.backend.clone(),
                stream: owner.stream.clone(),
                copy: owner.copy.clone(),
            })
            .collect())
    }
}

impl Dispatcher for Resident<'_> {
    fn wire_input(&self) -> bool {
        self.backend.borrow().uses_fp8_input()
    }

    fn enqueue(
        &self,
        input: Buffer,
        ids: Buffer,
        weights: Buffer,
        output: Buffer,
        rows: usize,
        stream: &Stream<'_>,
    ) -> Result<Buffer> {
        use crate::shared::experts::rtx::{ExpertInput, PartialDtype, PartialLayout, Routes};
        ensure!(
            self.stream.raw == stream.raw && std::ptr::eq(self.stream.device.library, stream.device.library),
            "Inkling 2 Mini resident workspace changed issuing stream"
        );
        let mut backend = self.backend.borrow_mut();
        ensure!(
            rows > 0
                && backend.device() == stream.device.id
                && backend.partial_layout(rows)? == PartialLayout::Tokens
                && output.bytes >= rows * HIDDEN * 2,
            "Inkling 2 Mini resident owner/output layout differs"
        );
        let element = if backend.partial() == PartialDtype::F32 { 4 } else { 2 };
        let result_bytes = rows
            .checked_mul(HIDDEN * element)
            .context("Inkling 2 Mini resident output extent overflow")?;
        ensure!(
            !backend.output().is_null() && backend.workspace_bytes() >= result_bytes,
            "Inkling 2 Mini resident workspace cannot hold submitted output rows"
        );
        let wire = backend.uses_fp8_input();
        let input_bytes = rows * if wire { WIRE_ROW_BYTES } else { HIDDEN * 2 };
        for (buffer, bytes) in [
            (input, input_bytes),
            (ids, rows * 8 * 4),
            (weights, rows * 8 * 4),
            (output, rows * HIDDEN * 2),
        ] {
            ensure!(
                buffer.device_id == stream.device.id
                    && buffer.bytes >= bytes
                    && !buffer.ptr.is_null()
                    && buffer.flags == 0
                    && buffer.ptr as usize % 16 == 0,
                "Inkling 2 Mini resident native operand differs"
            );
        }
        stream.device.run(|| {
            let input = if wire {
                ExpertInput::Fp8K32(input.ptr)
            } else {
                ExpertInput::Bf16(input.ptr)
            };
            // SAFETY: unit retains typed inputs/routes/output; Whole retains arena + scratch.
            unsafe {
                backend.enqueue(
                    self.layer,
                    rows,
                    input,
                    Routes {
                        ids: ids.ptr,
                        weights: weights.ptr,
                    },
                    stream.raw,
                    &mut || {},
                )?;
            }
            let source = Buffer {
                ptr: backend.output(),
                bytes: rows * HIDDEN * element,
                ..output
            };
            ensure!(source.ptr != output.ptr, "Inkling 2 Mini resident output aliases backend storage");
            // SAFETY: distinct buffers on the same ordered stream. When backend already outputs
            // BF16 (`element == 2`), return its persistent workspace view directly to avoid a
            // redundant `copy_rows` kernel before `join_visit` consumes it immediately.
            unsafe {
                if element == 4 {
                    stream
                        .device
                        .library
                        .cuda_f32_to_bf16_async(source, output, rows * HIDDEN, stream.raw)?;
                    Ok(output)
                } else {
                    ensure!(source.ptr as usize % 16 == 0, "Inkling 2 Mini resident output unaligned");
                    let _ = &self.copy;
                    Ok(source)
                }
            }
        })
    }

    fn inventory(&self, inventory: &mut Inventory<'_>, prefix: &str) -> Result<()> {
        self.backend.borrow().allocation_receipts(inventory, prefix)
    }
}

impl Drop for Resident<'_> {
    fn drop(&mut self) {
        self.stream.drain_or_abort("Inkling 2 Mini resident backend retirement");
    }
}

/// Builds all 24 TP1 layer dispatchers from the two homogeneous native backends
/// (NVFP4 layers 0..20 and MXFP8 layers 20..24), sharing one arena and scratch per group.
pub(crate) fn serving_dispatchers<'a>(
    stream: Rc<Stream<'a>>,
    catalog: &cuteafd_loader::OfficialV41Catalog,
    native_lib: &Path,
    backings: &[LocalBacking],
    rows: usize,
    pushed: Option<Vec<crate::shared::experts::rtx::push::PushedExperts<'a>>>,
) -> Result<Vec<Box<dyn Dispatcher + 'a>>> {
    use crate::shared::experts::rtx::{push::PushedExperts, whole::Whole, RtxExpertLayer};
    let mut pushed_iter = pushed.map(Vec::into_iter);
    if let Some(iter) = &pushed_iter {
        ensure!(
            iter.len() == backings.len(),
            "Inkling 2 Mini pushed expert set count differs from admission"
        );
    }
    let mut dispatchers = Vec::<(usize, Box<dyn Dispatcher + 'a>)>::new();
    for backing in backings {
        ensure!(
            backing.gpu == 0 && !backing.layers.is_empty(),
            "Inkling 2 Mini expert backing differs from TP1 GPU0"
        );
        let workspace = backing.workspace.iter().try_fold(0usize, |bytes, (_, size)| {
            bytes
                .checked_add(usize::try_from(*size)?)
                .context("Inkling 2 Mini expert workspace overflow")
        })?;
        let budget = backing
            .arena
            .bytes()
            .checked_add(workspace)
            .context("Inkling 2 Mini expert owner budget overflow")?;
        let tensors = catalog
            .fp8()
            .context("Inkling 2 Mini native expert catalog absent")?
            .for_layer_metadata(backing.layers[0])?;
        let package = crate::shared::experts::fp8::package_directory(native_lib, 1, tensors.format());
        let device = stream.device;
        let backend: Box<dyn RtxExpertLayer + 'a> = match pushed_iter.as_mut().and_then(Iterator::next) {
            Some(PushedExperts::Fp8(layers)) => {
                Box::new(Whole::from_pushed(device, &tensors, &package, backing.layers.clone(), rows, budget, layers)?)
            }
            None => Box::new(Whole::load(device, &tensors, &package, backing.layers.clone(), rows, budget)?),
            Some(_) => anyhow::bail!("Inkling 2 Mini pushed expert storage differs from native NVFP4/MXFP8"),
        };
        ensure!(
            backend.workspace_bytes() == workspace,
            "Inkling 2 Mini resident workspace differs from admitted physical requests"
        );
        let layers = backing.layers.clone();
        let handles = Resident::for_layers(backend, stream.clone())?;
        for (layer, handle) in layers.into_iter().zip(handles) {
            dispatchers.push((layer, Box::new(handle)));
        }
    }
    dispatchers.sort_by_key(|(layer, _)| *layer);
    ensure!(
        dispatchers.iter().enumerate().all(|(idx, (layer, _))| idx == *layer),
        "Inkling 2 Mini resident layers do not cover 0..24 contiguously"
    );
    Ok(dispatchers.into_iter().map(|(_, d)| d).collect())
}
