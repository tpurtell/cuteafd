//! Opt-in activation diagnostics; never read a device plane before its producer drains.
use anyhow::{ensure, Context, Result};
use cuteafd_ffi::{CuteafdDeviceBuffer as Buffer, NativeLibrary};
use std::{ffi::OsStr, io::Write, path::{Path, PathBuf}};

pub(crate) fn write_device(library: &NativeLibrary, directory: &Path, layer: usize,
    component: &str, buffer: Buffer, stream: *mut std::ffi::c_void) -> Result<()> {
    let mut bytes = vec![0; buffer.bytes];
    // SAFETY: caller supplies the live producing stream on `buffer`'s current device.
    unsafe { library.copy_d2h(&mut bytes, buffer, stream)?; }
    write_bytes(directory, layer, component, &bytes)
}

pub(crate) fn write_bytes(directory: &Path, layer: usize, component: &str, bytes: &[u8]) -> Result<()> {
    std::fs::create_dir_all(directory)?;
    let path = directory.join(format!("layer{layer}-{component}.bin"));
    let mut file = std::fs::OpenOptions::new().write(true).create_new(true).open(path)?;
    file.write_all(bytes)?;
    Ok(())
}

/// Preserve native F32 diagnostics too, with a BF16 view matching reference trace files.
pub(crate) fn write_f32(library: &NativeLibrary, directory: &Path, layer: usize,
    component: &str, buffer: Buffer, stream: *mut std::ffi::c_void) -> Result<()> {
    ensure!(buffer.bytes % 4 == 0, "activation trace F32 plane is not element aligned");
    let mut bytes = vec![0; buffer.bytes];
    // SAFETY: caller supplies the live producing stream on `buffer`'s current device.
    unsafe { library.copy_d2h(&mut bytes, buffer, stream)?; }
    let bf16: Vec<u8> = bytes.chunks_exact(4).flat_map(|word|
        bf16_rne(u32::from_le_bytes(word.try_into().unwrap())).to_le_bytes()).collect();
    write_bytes(directory, layer, &format!("{component}.f32"), &bytes)?;
    write_bytes(directory, layer, component, &bf16)
}

fn bf16_rne(bits: u32) -> u16 {
    if bits & 0x7fff_ffff > 0x7f80_0000 { 0x7fc0 }
    else { (bits.wrapping_add(0x7fff + ((bits >> 16) & 1)) >> 16) as u16 }
}

const POSITION: &str = "CUTEAFD_ACTIVATION_TRACE_POSITION";
const DIRECTORY: &str = "CUTEAFD_ACTIVATION_TRACE_DIR";

/// Whole-batch trace: the batch whose first row sits at
/// `CUTEAFD_ACTIVATION_TRACE_POSITION` traces into
/// `CUTEAFD_ACTIVATION_TRACE_DIR/<name>`. An unset position traces nothing.
pub(crate) fn batch_directory(first: Option<u64>, name: impl FnOnce() -> String) -> Result<Option<PathBuf>> {
    batch_directory_from(std::env::var_os(POSITION).as_deref(), std::env::var_os(DIRECTORY).as_deref(), first, name)
}

fn batch_directory_from(position: Option<&OsStr>, directory: Option<&OsStr>, first: Option<u64>,
    name: impl FnOnce() -> String) -> Result<Option<PathBuf>> {
    let Some(position) = position else { return Ok(None) };
    let position: u64 = position.to_str().context("activation trace position is not UTF-8")?.parse()?;
    if first != Some(position) { return Ok(None) }
    let directory = directory.context("activation trace requires CUTEAFD_ACTIVATION_TRACE_DIR")?;
    Ok(Some(PathBuf::from(directory).join(name())))
}

#[derive(Debug, Clone)]
pub(crate) struct RowTrace {
    pub directory: PathBuf,
    pub row: usize,
    layer: Option<usize>,
}

impl RowTrace {
    pub fn select(positions: &[i64], prefill: bool) -> Result<Option<Self>> {
        let Some(position) = std::env::var_os(POSITION) else { return Ok(None) };
        let position: i64 = position.to_str().context("activation trace position is not UTF-8")?.parse()?;
        ensure!(position >= 0, "activation trace position must be nonnegative");
        let rows: Vec<_> = positions.iter().enumerate().filter(|(_, p)| **p == position).map(|(i, _)| i).collect();
        if rows.is_empty() { return Ok(None) }
        ensure!(rows.len() == 1, "activation trace position {position} belongs to multiple sequences");
        let directory = PathBuf::from(std::env::var_os(DIRECTORY)
            .context("activation trace requires CUTEAFD_ACTIVATION_TRACE_DIR")?)
            .join(format!("position{position}-{}", if prefill { "prefill" } else { "decode" }));
        let layer = std::env::var("CUTEAFD_TRACE_LAYER").ok().map(|s| s.parse()).transpose()?;
        Ok(Some(Self { directory, row: rows[0], layer }))
    }

    pub fn includes(&self, layer: usize) -> bool { self.layer.is_none_or(|selected| selected == layer) }

    pub fn row(&self, buffer: Buffer, stride: usize) -> Result<Buffer> {
        let offset = self.row.checked_mul(stride).context("activation trace row offset overflow")?;
        ensure!(offset.checked_add(stride).is_some_and(|end| end <= buffer.bytes),
            "activation trace selected row exceeds its plane");
        Ok(Buffer { ptr: buffer.ptr.cast::<u8>().wrapping_add(offset).cast(), bytes: stride, ..buffer })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn f32_trace_rounds_bf16_ties_and_special_values() {
        for (input, expected) in [(0, 0), (0x80000000, 0x8000),
            (0x3f808000, 0x3f80), (0x3f818000, 0x3f82),
            (0x7f800000, 0x7f80), (0xff800000, 0xff80), (0x7f800001, 0x7fc0)] {
            assert_eq!(bf16_rne(input), expected);
        }
    }

    #[test]
    fn batch_trace_is_opt_in_and_needs_a_directory_only_when_selected() {
        let name = || "batch".to_string();
        for first in [None, Some(0), Some(7)] {
            assert!(batch_directory_from(None, None, first, name).unwrap().is_none());
        }
        let position = Some(OsStr::new("7"));
        assert!(batch_directory_from(position, None, Some(3), name).unwrap().is_none());
        assert!(batch_directory_from(position, None, Some(7), name).is_err());
        assert!(batch_directory_from(Some(OsStr::new("x")), None, Some(7), name).is_err());
        assert_eq!(batch_directory_from(position, Some(OsStr::new("/trace")), Some(7), name).unwrap(),
            Some(PathBuf::from("/trace/batch")));
    }

    #[test]
    fn selected_row_is_bounded_and_layer_filtered() {
        let trace = RowTrace { directory: PathBuf::new(), row: 2, layer: Some(3) };
        assert!(trace.includes(3) && !trace.includes(2));
        let buffer = Buffer { bytes: 48, ..Buffer::default() };
        assert_eq!(trace.row(buffer, 16).unwrap().bytes, 16);
        assert!(trace.row(buffer, 17).is_err());
    }
}
