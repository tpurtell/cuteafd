//! GLM 5.3 coordinator operands that are checkpoint bytes landing unchanged:
//! each operand is a list of rectangles (a row range of one tensor, or the
//! same byte range of each of its rows) at fixed offsets. The mount fills
//! them by reading; a push plan targets the same offsets. Combination and
//! transform steps stay in the loader.
use anyhow::{ensure, Result};
use cuteafd_core::DType;
use cuteafd_loader::OfficialV41Catalog;
use std::ops::Range;

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Land {
    pub tensor: String,
    /// Source rows of the tensor.
    pub rows: Range<usize>,
    /// Byte range within each source row (packed at the destination).
    pub cols: Range<usize>,
    /// Destination byte offset within the operand.
    pub offset: usize,
}

impl Land {
    pub fn width(&self) -> usize { self.cols.len() }
    pub fn bytes(&self) -> usize { self.rows.len() * self.cols.len() }
    /// The whole tensor at `offset`.
    pub fn whole(catalog: &OfficialV41Catalog, tensor: &str, offset: usize) -> Result<Self> {
        let (rows, row_bytes) = geometry(catalog, tensor)?;
        Ok(Self { tensor: tensor.into(), rows: 0..rows, cols: 0..row_bytes, offset })
    }
}

/// Rows and bytes per row of a checkpoint tensor (1-D tensors are one row).
pub(crate) fn geometry(catalog: &OfficialV41Catalog, tensor: &str) -> Result<(usize, usize)> {
    let meta = &catalog.tensor(tensor)?.metadata;
    let rows = if meta.shape.len() >= 2 { meta.shape[0] } else { 1 };
    let bytes = usize::try_from(meta.byte_length)?;
    ensure!(rows > 0 && bytes % rows == 0, "{tensor}: rows do not divide its bytes");
    Ok((rows, bytes / rows))
}

/// Whether `name` is an E4M3 matrix with its own FP32 128x128 block grid
/// (`{name}_scale_inv`), the form whose bytes land unchanged.
/// The FP32 128x128 block-scale grid of FP8 weight `name` (`*.weight`): the
/// Hugging Face `weight_scale_inv`, or Mistral's consolidated `qscale_weight`
/// (the same multiplied grid). The HF name when neither is present.
pub(crate) fn block_grid(catalog: &OfficialV41Catalog, name: &str) -> String {
    let hf = format!("{name}_scale_inv");
    if catalog.tensor(&hf).is_ok() { return hf; }
    match name.strip_suffix(".weight") {
        Some(stem) if catalog.tensor(&format!("{stem}.qscale_weight")).is_ok() => format!("{stem}.qscale_weight"),
        _ => hf,
    }
}

pub(crate) fn native_fp8_block(catalog: &OfficialV41Catalog, name: &str) -> Result<bool> {
    let meta = &catalog.tensor(name)?.metadata;
    if meta.dtype != DType::F8E4M3 || meta.shape.len() != 2 { return Ok(false); }
    let Ok(scale) = catalog.tensor(&block_grid(catalog, name)) else { return Ok(false) };
    Ok(scale.metadata.dtype == DType::F32
        && scale.metadata.shape == [meta.shape[0].div_ceil(128), meta.shape[1].div_ceil(128)])
}

impl Land {
    /// The shared coordinator rectangle and its operand offset.
    pub fn rect(&self, catalog: &OfficialV41Catalog) -> Result<(crate::shared::experts::rtx::dense::Rect, usize)> {
        let tensor = catalog.tensor(&self.tensor)?;
        let (rows, row_bytes) = geometry(catalog, &self.tensor)?;
        ensure!(self.rows.end <= rows && self.cols.end <= row_bytes && !self.rows.is_empty() && !self.cols.is_empty(),
            "{}: landing outside the tensor", self.tensor);
        Ok((crate::shared::experts::rtx::dense::Rect { name: self.tensor.clone(),
            path: catalog.snapshot().join(&tensor.shard), base: tensor.metadata.byte_offset, row_bytes,
            rows: self.rows.clone(), cols: self.cols.clone() }, self.offset))
    }
}

/// The shared class-2 operand of `lands` in `buffer` on `device`.
pub(crate) fn operand(catalog: &OfficialV41Catalog, device: i32, buffer: cuteafd_ffi::CuteafdDeviceBuffer,
    lands: &[Land], bytes: usize) -> Result<crate::shared::experts::rtx::dense::Operand> {
    ensure!(lands.iter().all(|l| l.offset.checked_add(l.bytes()).is_some_and(|end| end <= bytes)) && bytes <= buffer.bytes,
        "landing outside its operand");
    Ok(crate::shared::experts::rtx::dense::Operand { device, buffer, bytes,
        lands: lands.iter().map(|land| land.rect(catalog)).collect::<Result<_>>()? })
}

/// The operand's host image of `lands` read from the checkpoint (the mount
/// path); unspecified gaps stay zero.
pub(crate) fn read(catalog: &OfficialV41Catalog, lands: &[Land], bytes: usize) -> Result<Vec<u8>> {
    let mut out = vec![0u8; bytes];
    let mut rows = None;
    for land in lands {
        let (rect, at) = land.rect(catalog)?;
        ensure!(at.checked_add(rect.bytes()).is_some_and(|end| end <= bytes), "{}: landing outside the operand", land.tensor);
        rect.read_cached(&mut out[at..at + rect.bytes()], &mut rows)?;
    }
    Ok(out)
}

/// `names` row-concatenated (each whole) from `offset`, and where each lands.
pub(crate) fn concat(catalog: &OfficialV41Catalog, names: &[String]) -> Result<(Vec<Land>, usize)> {
    let mut at = 0;
    let mut lands = Vec::with_capacity(names.len());
    for name in names {
        let land = Land::whole(catalog, name, at)?;
        at += land.bytes();
        lands.push(land);
    }
    Ok((lands, at))
}

/// Rank `rank`'s share of `names` row-concatenated and split over `ranks`
/// by rows in whole `block`-row groups: each part's rows `rank` of `ranks`.
pub(crate) fn row_shares(catalog: &OfficialV41Catalog, names: &[String], ranks: usize, rank: usize, block: usize)
    -> Result<(Vec<Land>, usize)> {
    let mut at = 0;
    let mut lands = Vec::with_capacity(names.len());
    for name in names {
        let (rows, row_bytes) = geometry(catalog, name)?;
        ensure!(rows % (block * ranks) == 0, "{name}: {rows} rows do not split into whole {block}-row blocks over {ranks}");
        let share = rows / ranks;
        lands.push(Land { tensor: name.clone(), rows: rank * share..(rank + 1) * share, cols: 0..row_bytes, offset: at });
        at += share * row_bytes;
    }
    Ok((lands, at))
}

/// Rank `rank`'s column share (bytes `rank` of `ranks` of each row) of one tensor.
pub(crate) fn col_share(catalog: &OfficialV41Catalog, name: &str, ranks: usize, rank: usize) -> Result<(Land, usize)> {
    let (rows, row_bytes) = geometry(catalog, name)?;
    ensure!(row_bytes % ranks == 0, "{name}: rows of {row_bytes} bytes do not split over {ranks}");
    let width = row_bytes / ranks;
    let land = Land { tensor: name.into(), rows: 0..rows, cols: rank * width..(rank + 1) * width, offset: 0 };
    let bytes = land.bytes();
    Ok((land, bytes))
}

/// A K-major widening of a landed grid, run once its grid is delivered.
#[derive(Clone, Copy)]
pub(crate) struct Kmajor {
    pub device: i32,
    pub grid: cuteafd_ffi::CuteafdDeviceBuffer,
    pub kmajor: cuteafd_ffi::CuteafdDeviceBuffer,
    pub rows: usize,
    pub kb: usize,
}

impl Kmajor {
    /// On the current device (`device`), drained before return.
    pub fn run(&self, library: &cuteafd_ffi::NativeLibrary, stream: *mut std::ffi::c_void) -> Result<()> {
        ensure!(self.grid.bytes >= self.rows.div_ceil(128) * self.kb * 4 && self.kmajor.bytes >= self.rows * self.kb * 4,
            "K-major widening exceeds its operands");
        // SAFETY: both buffers are live operand allocations on the current device
        // sized as checked above; the stream drains before return.
        unsafe {
            library.fp8_grid_kmajor(self.grid.ptr, self.kmajor.ptr, self.rows, self.kb, stream)?;
            library.cuda_stream_synchronize(stream)
        }
    }
}

/// Push-mode loader output: the class-2 operands to fill (shared records)
/// and the transforms that read them after commit, in loader order.
#[derive(Default)]
pub(crate) struct PushSink {
    pub dense: crate::shared::experts::rtx::dense::DenseSink,
    deferred: std::cell::RefCell<Vec<Kmajor>>,
}

impl PushSink {
    pub fn record(&self, catalog: &OfficialV41Catalog, device: i32, buffer: cuteafd_ffi::CuteafdDeviceBuffer,
        lands: &[Land], bytes: usize) -> Result<()> {
        self.dense.operand(operand(catalog, device, buffer, lands, bytes)?)
    }
    pub fn defer(&self, job: Kmajor) -> Result<()> {
        self.deferred.borrow_mut().push(job);
        Ok(())
    }
    /// The recorded operands and deferred transforms (leaves the sink empty).
    pub fn take(&self) -> (Vec<crate::shared::experts::rtx::dense::Operand>, Vec<Kmajor>) {
        (self.dense.take().0, self.deferred.take())
    }
}
