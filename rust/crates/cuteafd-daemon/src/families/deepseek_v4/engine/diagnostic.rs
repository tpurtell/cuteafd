//! Private, never-merged verify/serial localization. Copies are outside graph capture.
use super::*;
use std::io::Write;
use std::os::unix::fs::{DirBuilderExt, OpenOptionsExt, PermissionsExt};
use std::path::{Path, PathBuf};
use crate::shared::prefix::view;

const CAPACITY: usize = 8;
const FIELDS: [&str; 8] = ["stream_a", "attention_sum", "stream_b", "y", "wire", "route_ids", "route_weights", "shared"];

#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct DiagnosticConfig {
    pub prompt: Vec<u32>,
    pub generated: usize,
    pub output: PathBuf,
}

impl DiagnosticConfig {
    pub fn from_env() -> Result<Option<Self>> {
        let Some(path) = std::env::var_os("CUTEAFD_DIAG_V4_VERIFY") else { return Ok(None) };
        let meta = std::fs::symlink_metadata(&path)?;
        ensure!(meta.is_file() && meta.permissions().mode() & 0o077 == 0, "private diagnostic config required");
        let config: Self = serde_json::from_slice(&std::fs::read(path)?)?;
        ensure!(!config.prompt.is_empty() && config.generated > 0, "invalid diagnostic request selector");
        std::fs::DirBuilder::new().mode(0o700).create(&config.output)
            .context("diagnostic output must be a fresh private directory")?;
        Ok(Some(config))
    }
}

pub(crate) fn private_write(path: &Path, bytes: &[u8]) -> Result<()> {
    let mut file = std::fs::OpenOptions::new().write(true).create_new(true).mode(0o600).open(path)?;
    file.write_all(bytes)?;
    Ok(())
}

struct Tensor<'a> {
    name: String,
    stride: usize,
    data: Dev<'a>,
}

struct ReplayPage<'a> {
    rank: usize,
    source: cuteafd_ffi::CuteafdDeviceBuffer,
    stride: usize,
    saved: Dev<'a>,
}

pub(super) struct Boundaries<'a> {
    tensors: Vec<Tensor<'a>>,
    pages: Vec<ReplayPage<'a>>,
    row: Option<usize>,
}

fn destination_row(offset: usize, rows: usize) -> Result<std::ops::Range<usize>> {
    ensure!(rows > 0 && offset <= CAPACITY && rows <= CAPACITY - offset, "diagnostic exceeds eight-row capacity");
    Ok(offset..offset + rows)
}

impl<'a> Engine<'a> {
    pub(crate) fn enable_boundaries(&self) -> Result<()> {
        ensure!(self.ranks() == 2 && self.boundaries.borrow().is_none(), "diagnostic requires heads and one initialization");
        let _scope = cuteafd_ffi::memory_ledger::scope("private-diagnostic");
        let (h, k) = (self.cfg.dim, self.cfg.n_activated_experts);
        let strides = [4*h*2, h*2, 4*h*2, h*2, h+h/32, k*4, k*4, h*2];
        let mut tensors = Vec::new();
        for layer in 0..self.cfg.n_layers {
            for (name, stride) in FIELDS.iter().zip(strides) {
                tensors.push(Tensor { name: format!("layer{layer:02}-{name}"), stride,
                    data: self.alloc(CAPACITY * stride)? });
            }
        }
        for (name, stride) in [("final_stream", 4*h*2), ("logits", self.cfg.vocab_size*4)] {
            tensors.push(Tensor { name: name.into(), stride, data: self.alloc(CAPACITY * stride)? });
        }
        tracing::info!(bytes = tensors.iter().map(|t| t.data.buffer.bytes).sum::<usize>(), "private V4 boundaries allocated");
        let mut pages = Vec::new();
        for rank in 0..self.ranks() {
            for layer in self.caches_on(rank) {
                for cache in layer.compressed.iter().chain(layer.index.iter()) {
                    ensure!(cache.buffer.bytes % self.shape.units == 0, "partial diagnostic cache page");
                    let stride = cache.buffer.bytes / self.shape.units;
                    pages.push(ReplayPage { rank, source: cache.buffer, stride,
                        saved: self.on(rank, || self.alloc(2 * stride))? });
                }
            }
        }
        *self.boundaries.borrow_mut() = Some(Boundaries { tensors, pages, row: None });
        Ok(())
    }

    pub(crate) fn replay_pages(&self, placement: &Placement, start: usize, rows: usize, capture: bool) -> Result<()> {
        destination_row(0, rows)?;
        let page_rows = super::super::metadata::SOURCE_PAGE_TOKENS;
        let end = start.checked_add(rows).context("diagnostic position overflow")?;
        let units = placement.units.get(start / page_rows..(end - 1) / page_rows + 1)
            .context("diagnostic rows outside allocated units")?;
        ensure!(units.len() <= 2, "diagnostic touches more than two units");
        let boundaries = self.boundaries.borrow();
        let boundaries = boundaries.as_ref().context("diagnostic not enabled")?;
        for page in &boundaries.pages {
            for (index, &unit) in units.iter().enumerate() {
                let live = view(page.source, unit as usize * page.stride, page.stride)?;
                let saved = view(page.saved.buffer, index * page.stride, page.stride)?;
                let (dst, src) = if capture { (saved, live) } else { (live, saved) };
                // SAFETY: both checked page ranges stay owned until the round drains;
                // each copy is ordered with the cache writers on its owning rank stream.
                self.on(page.rank, || unsafe {
                    self.library.copy_d2d_async(dst, src, page.stride, self.stream_of(page.rank))
                })?;
            }
        }
        Ok(())
    }

    pub(crate) fn boundary_row(&self, row: Option<usize>) -> Result<()> {
        if let Some(row) = row { destination_row(row, 1)?; }
        self.boundaries.borrow_mut().as_mut().context("diagnostic not enabled")?.row = row;
        Ok(())
    }

    fn snapshot(&self, index: usize, rows: usize, source: &Dev<'_>) -> Result<()> {
        let snapshots = self.boundaries.borrow();
        let Some(snapshots) = snapshots.as_ref() else { return Ok(()) };
        let Some(offset) = snapshots.row else { return Ok(()) };
        ensure!(!self.capture_only.get(), "diagnostic copy attempted during capture");
        destination_row(offset, rows)?;
        let tensor = &snapshots.tensors[index];
        let bytes = rows * tensor.stride;
        // SAFETY: checked disjoint slices are owned by the engine until round-end drainage;
        // the copy follows the producer and precedes every source overwrite on this stream.
        unsafe { self.library.copy_d2d_async(view(tensor.data.buffer, offset * tensor.stride, bytes)?,
            view(source.buffer, 0, bytes)?, bytes, self.stream) }
    }

    pub(super) fn snapshot_front(&self, layer: usize, rows: usize, w: &Workspace<'_>, lane: &Lane<'_>) -> Result<()> {
        if self.boundaries.borrow().as_ref().is_none_or(|s| s.row.is_none()) { return Ok(()) }
        for (field, source) in [&lane.stream_a, &w.sum, &lane.stream_b, &w.y, &w.wire, &w.route_ids, &w.route_weights].into_iter().enumerate() {
            self.snapshot(layer * FIELDS.len() + field, rows, source)?;
        }
        Ok(())
    }

    pub(super) fn snapshot_shared(&self, layer: usize, rows: usize, lane: &Lane<'_>) -> Result<()> {
        self.snapshot(layer * FIELDS.len() + 7, rows, &lane.shared)
    }

    pub(super) fn snapshot_tail(&self, rows: usize, w: &Workspace<'_>, lane: &Lane<'_>) -> Result<()> {
        self.snapshot(self.cfg.n_layers * FIELDS.len(), rows, &lane.stream_a)?;
        self.snapshot(self.cfg.n_layers * FIELDS.len() + 1, rows, &w.vocab_logits)
    }

    pub(crate) fn dump_boundaries(&self, path: &Path, rows: usize) -> Result<()> {
        destination_row(0, rows)?;
        self.check_device()?;
        // SAFETY: all boundary destinations remain owned until both rank streams drain.
        for rank in 0..self.ranks() { unsafe { self.library.cuda_stream_synchronize(self.stream_of(rank))?; } }
        std::fs::DirBuilder::new().mode(0o700).create(path)?;
        let snapshots = self.boundaries.borrow();
        let snapshots = snapshots.as_ref().context("diagnostic not enabled")?;
        let mut manifest = Vec::new();
        for tensor in &snapshots.tensors {
            let bytes = self.download(&tensor.data, rows * tensor.stride)?;
            private_write(&path.join(format!("{}.bin", tensor.name)), &bytes)?;
            manifest.push(serde_json::json!({"name": tensor.name, "rows": rows, "row_bytes": tensor.stride}));
        }
        private_write(&path.join("manifest.json"), &serde_json::to_vec_pretty(&manifest)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn snapshot_capacity_is_checked_without_overflow() {
        assert_eq!(destination_row(0, 8).unwrap(), 0..8);
        assert_eq!(destination_row(7, 1).unwrap(), 7..8);
        for (offset, rows) in [(0, 0), (0, 9), (7, 2), (8, 1), (usize::MAX, 1), (0, usize::MAX)] {
            assert!(destination_row(offset, rows).is_err());
        }
    }
    #[test]
    fn private_trace_rejects_unknown_fields() {
        assert!(serde_json::from_str::<DiagnosticConfig>(r#"{"prompt":[1],"generated":1,"output":"/private","typo":1}"#).is_err());
    }
}
