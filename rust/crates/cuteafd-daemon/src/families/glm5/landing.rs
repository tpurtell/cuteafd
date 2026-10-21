//! GLM 5.3 coordinator operands that are checkpoint bytes landing unchanged:
//! each operand is a list of rectangles (a row range of one tensor, or the
//! same byte range of each of its rows) at fixed offsets. The mount fills
//! them by reading; a push plan targets the same offsets. Combination and
//! transform steps stay in the loader.
use anyhow::{ensure, Context, Result};
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
pub(crate) fn native_fp8_block(catalog: &OfficialV41Catalog, name: &str) -> Result<bool> {
    let meta = &catalog.tensor(name)?.metadata;
    if meta.dtype != DType::F8E4M3 || meta.shape.len() != 2 { return Ok(false); }
    let Ok(scale) = catalog.tensor(&format!("{name}_scale_inv")) else { return Ok(false) };
    Ok(scale.metadata.dtype == DType::F32
        && scale.metadata.shape == [meta.shape[0].div_ceil(128), meta.shape[1].div_ceil(128)])
}

/// The operand's host image of `lands` read from the checkpoint (the mount
/// path); unspecified gaps stay zero. One positioned read per row range.
pub(crate) fn read(catalog: &OfficialV41Catalog, lands: &[Land], bytes: usize) -> Result<Vec<u8>> {
    use std::os::unix::fs::FileExt;
    let mut out = vec![0u8; bytes];
    for land in lands {
        let tensor = catalog.tensor(&land.tensor)?;
        let (rows, row_bytes) = geometry(catalog, &land.tensor)?;
        ensure!(land.rows.end <= rows && land.cols.end <= row_bytes && !land.rows.is_empty() && !land.cols.is_empty(),
            "{}: landing outside the tensor", land.tensor);
        ensure!(land.offset.checked_add(land.bytes()).is_some_and(|end| end <= bytes),
            "{}: landing outside the operand", land.tensor);
        let file = std::fs::File::open(catalog.snapshot().join(&tensor.shard))?;
        let start = tensor.metadata.byte_offset + (land.rows.start * row_bytes) as u64;
        if land.cols.len() == row_bytes {
            crate::shared::memory::load_timing::read(land.bytes(), ||
                file.read_exact_at(&mut out[land.offset..land.offset + land.bytes()], start))
                .with_context(|| format!("reading {}", land.tensor))?;
        } else {
            let mut source = vec![0u8; land.rows.len() * row_bytes];
            crate::shared::memory::load_timing::read(source.len(), || file.read_exact_at(&mut source, start))
                .with_context(|| format!("reading {}", land.tensor))?;
            for (row, chunk) in source.chunks_exact(row_bytes).enumerate() {
                let at = land.offset + row * land.width();
                out[at..at + land.width()].copy_from_slice(&chunk[land.cols.clone()]);
            }
        }
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

/// One class-2 operand allocation awaiting its pushed bytes.
pub(crate) struct Record {
    pub device: i32,
    pub buffer: cuteafd_ffi::CuteafdDeviceBuffer,
    pub lands: Vec<Land>,
    pub bytes: usize,
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

/// Push-mode loader output: the class-2 allocations to fill and the
/// transforms that read them, in loader order.
#[derive(Default)]
pub(crate) struct PushSink {
    records: std::cell::RefCell<Vec<Record>>,
    deferred: std::cell::RefCell<Vec<Kmajor>>,
}

impl PushSink {
    pub fn record(&self, device: i32, buffer: cuteafd_ffi::CuteafdDeviceBuffer, lands: &[Land], bytes: usize) -> Result<()> {
        ensure!(bytes <= buffer.bytes && lands.iter().all(|l| l.offset + l.bytes() <= bytes),
            "pushed landing outside its operand");
        self.records.borrow_mut().push(Record { device, buffer, lands: lands.to_vec(), bytes });
        Ok(())
    }
    pub fn defer(&self, job: Kmajor) -> Result<()> {
        self.deferred.borrow_mut().push(job);
        Ok(())
    }
    /// The recorded operands and deferred transforms (leaves the sink empty).
    pub fn take(&self) -> (Vec<Record>, Vec<Kmajor>) {
        (self.records.take(), self.deferred.take())
    }
}
