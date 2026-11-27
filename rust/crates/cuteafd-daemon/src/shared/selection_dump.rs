//! Opt-in diagnostic readback of sparse-attention selections
//! (`CUTEAFD_DIAG_SELECTION_DUMP`), shared by the selecting families.
//!
//! Unset `CUTEAFD_DIAG_SELECTION_DUMP` resolves to `None` at startup: no
//! stream drain, no host copy, no behavior change. When set, each
//! coordinator decode step drains the producing stream after the family's
//! selector and records, per batch row and selecting layer, the selected
//! **logical** units at the family's native selection granularity (tokens,
//! pooled key groups, compressed rows or 128-token blocks), ascending and
//! `-1` padded. Rows whose engine position is below
//! `CUTEAFD_DIAG_SELECTION_DUMP_FROM` (default `0`) are dropped before the
//! readback decision, so a run can skip a warm prefix.
//!
//! Unlike `CUTEAFD_KIMI_DSA_DUMP` (golden windows, one record per position),
//! this writer is append-only: agentic sessions re-run positions after a
//! prefix restore, and every occurrence is kept with its per-layer step
//! index. Format (little endian), one file per layer `layer-NNN.bin`:
//! header `SSEL0001`, layer u32, width u32, unit_tokens u32, source layer u32,
//! heads u32, reserved u32, rows u64 (40 bytes); then rows of `step u64, position u64, frontier i32, count i32,
//! ids [width] i32`. `frontier` is the selector's causal candidate count in
//! units (`-1` when the family does not expose it). `manifest.jsonl` holds one
//! line per recorded batch.

use anyhow::{ensure, Context, Result};
use std::{
    collections::BTreeMap,
    ffi::OsStr,
    fs::{File, OpenOptions},
    io::{Seek, SeekFrom, Write},
    path::PathBuf,
    sync::{Arc, Mutex, OnceLock},
};

pub(crate) const ENV_DIR: &str = "CUTEAFD_DIAG_SELECTION_DUMP";
pub(crate) const ENV_FROM: &str = "CUTEAFD_DIAG_SELECTION_DUMP_FROM";
pub(crate) const MAGIC: &[u8; 8] = b"SSEL0001";
pub(crate) const HEADER_BYTES: usize = 40;

/// How a layer's selection is expressed: `width` slots of logical units of
/// `unit_tokens` tokens each, chosen by layer `source` (itself, or the layer
/// whose index it shares).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Layout {
    pub width: usize,
    pub unit_tokens: u32,
    pub source: usize,
    /// Independently selecting KV heads folded into each id as
    /// `unit * heads + head` (1: one selection shared by every head).
    pub heads: u32,
}

/// One batch row's selection as the engine read it: logical unit ids in any
/// order, negatives as padding.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Row {
    pub position: u64,
    pub frontier: i32,
    pub ids: Vec<i32>,
}

pub(crate) fn record_bytes(width: usize) -> usize { 8 + 8 + 4 + 4 + width * 4 }

struct LayerWriter {
    file: File,
    layout: Layout,
    rows: u64,
    steps: u64,
}

#[derive(Default)]
struct State {
    layers: BTreeMap<usize, LayerWriter>,
    manifest: Option<File>,
    entries: u64,
}

pub(crate) struct SelectionDump {
    family: &'static str,
    dir: PathBuf,
    from: u64,
    state: Mutex<State>,
}

static SHARED: OnceLock<Option<Arc<SelectionDump>>> = OnceLock::new();

/// The process's dump (resolved from the environment once; every lane and
/// wave of a family shares the one writer). A second family in one process
/// is refused.
pub(crate) fn shared(family: &'static str) -> Result<Option<Arc<SelectionDump>>> {
    if let Some(resolved) = SHARED.get() {
        if let Some(dump) = resolved { ensure!(dump.family == family, "selection dump already bound to {}", dump.family); }
        return Ok(resolved.clone());
    }
    let resolved = SelectionDump::from_env(family)?;
    Ok(SHARED.get_or_init(|| resolved).clone())
}

impl SelectionDump {
    fn from_env(family: &'static str) -> Result<Option<Arc<Self>>> {
        Self::from_env_vars(family, std::env::var_os(ENV_DIR).as_deref(), std::env::var_os(ENV_FROM).as_deref())
    }

    pub(crate) fn from_env_vars(family: &'static str, dir: Option<&OsStr>, from: Option<&OsStr>)
        -> Result<Option<Arc<Self>>> {
        let Some(raw) = dir else { return Ok(None) };
        ensure!(!raw.is_empty(), "{ENV_DIR} must name a non-empty directory path");
        let from = match from {
            None => 0,
            Some(value) => value.to_str().with_context(|| format!("{ENV_FROM} is not valid UTF-8"))?
                .parse::<u64>().with_context(|| format!("invalid {ENV_FROM}"))?,
        };
        let dir = PathBuf::from(raw);
        std::fs::create_dir_all(&dir).with_context(|| format!("create selection dump directory {}", dir.display()))?;
        tracing::warn!(family, dir = %dir.display(), from, "{ENV_DIR} set: decode drains the stream at every selecting layer (diagnostic only)");
        Ok(Some(Arc::new(Self { family, dir, from, state: Mutex::new(State::default()) })))
    }

    /// Whether any of these engine positions is recorded (the readback gate).
    pub(crate) fn wants(&self, positions: impl IntoIterator<Item = u64>) -> bool {
        positions.into_iter().any(|position| position >= self.from)
    }

    /// Appends `rows` (those at or after FROM) to layer `layer`'s file.
    pub(crate) fn write(&self, layer: usize, layout: Layout, rows: Vec<Row>) -> Result<()> {
        ensure!(layout.width > 0 && layout.unit_tokens > 0 && layout.heads > 0, "selection dump layout must be positive");
        let kept: Vec<Row> = rows.into_iter().filter(|row| row.position >= self.from).collect();
        if kept.is_empty() { return Ok(()); }
        let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());
        if !state.layers.contains_key(&layer) {
            let path = self.dir.join(format!("layer-{layer:03}.bin"));
            let mut file = OpenOptions::new().read(true).write(true).create_new(true).open(&path)
                .with_context(|| format!("create selection dump {}", path.display()))?;
            let mut header = [0u8; HEADER_BYTES];
            header[0..8].copy_from_slice(MAGIC);
            header[8..12].copy_from_slice(&u32::try_from(layer)?.to_le_bytes());
            header[12..16].copy_from_slice(&u32::try_from(layout.width)?.to_le_bytes());
            header[16..20].copy_from_slice(&layout.unit_tokens.to_le_bytes());
            header[20..24].copy_from_slice(&u32::try_from(layout.source)?.to_le_bytes());
            header[24..28].copy_from_slice(&layout.heads.to_le_bytes());
            file.write_all(&header)?;
            state.layers.insert(layer, LayerWriter { file, layout, rows: 0, steps: 0 });
        }
        let writer = state.layers.get_mut(&layer).unwrap();
        ensure!(writer.layout == layout, "selection dump layout of layer {layer} changed");
        let step = writer.steps;
        writer.steps += 1;
        let mut encoded = Vec::with_capacity(kept.len() * record_bytes(layout.width));
        let (first, last) = (kept.first().unwrap().position, kept.last().unwrap().position);
        for row in &kept {
            let mut ids: Vec<i32> = row.ids.iter().copied().filter(|&id| id >= 0).collect();
            ids.sort_unstable();
            ensure!(ids.windows(2).all(|pair| pair[0] < pair[1]),
                "{} selection repeats a unit at layer {layer} position {}", self.family, row.position);
            ensure!(ids.len() <= layout.width, "{} selection wider than {}", self.family, layout.width);
            encoded.extend_from_slice(&step.to_le_bytes());
            encoded.extend_from_slice(&row.position.to_le_bytes());
            encoded.extend_from_slice(&row.frontier.to_le_bytes());
            encoded.extend_from_slice(&(ids.len() as i32).to_le_bytes());
            for slot in 0..layout.width {
                encoded.extend_from_slice(&ids.get(slot).copied().unwrap_or(-1).to_le_bytes());
            }
        }
        writer.file.seek(SeekFrom::End(0))?;
        writer.file.write_all(&encoded)?;
        writer.rows += kept.len() as u64;
        writer.file.seek(SeekFrom::Start(32))?;
        writer.file.write_all(&writer.rows.to_le_bytes())?;
        writer.file.flush()?;
        let total = writer.rows;
        if state.manifest.is_none() {
            let path = self.dir.join("manifest.jsonl");
            state.manifest = Some(OpenOptions::new().append(true).create(true).open(&path)
                .with_context(|| format!("open selection dump manifest {}", path.display()))?);
        }
        let seq = state.entries;
        state.entries += 1;
        let line = serde_json::json!({
            "schema": "cuteafd.selection-dump/1", "family": self.family, "seq": seq, "layer": layer,
            "source_layer": layout.source, "layer_step": step, "kept_rows": kept.len(), "total_rows": total,
            "first_position": first, "last_position": last, "from": self.from, "width": layout.width,
            "unit_tokens": layout.unit_tokens, "heads": layout.heads, "header_bytes": HEADER_BYTES,
            "record_bytes": record_bytes(layout.width), "order": "ascending logical unit ids, -1 padded",
        });
        let manifest = state.manifest.as_mut().unwrap();
        serde_json::to_writer(&mut *manifest, &line)?;
        manifest.write_all(b"\n")?;
        manifest.flush()?;
        Ok(())
    }
}

/// Records one layer's selection when a dump is attached and any position is
/// wanted. `read` (the drain plus host copies) runs only then; unset means no
/// call at all.
pub(crate) fn record(dump: Option<&SelectionDump>, layer: usize, layout: Layout, positions: &[u64],
    read: impl FnOnce() -> Result<Vec<Row>>) -> Result<()> {
    let Some(dump) = dump else { return Ok(()) };
    if !dump.wants(positions.iter().copied()) { return Ok(()) }
    dump.write(layer, layout, read()?)
}

/// Inverts a row's physical slot ids (`page * page_rows + offset`) through its
/// page-table row (logical page -> physical page) into logical token ids.
pub(crate) fn logical_tokens(slots: &[i32], table: &[i32], page_rows: usize) -> Result<Vec<i32>> {
    let mut logical = BTreeMap::new();
    for (index, &page) in table.iter().enumerate() {
        if page >= 0 { logical.entry(page).or_insert(index); }
    }
    slots.iter().copied().filter(|&slot| slot >= 0).map(|slot| {
        let page = slot / page_rows as i32;
        let index = *logical.get(&page).with_context(|| format!("selected slot {slot} outside its page table"))?;
        Ok((index * page_rows + (slot as usize % page_rows)) as i32)
    }).collect()
}

/// Device buffers of a paged selector (Kimi): i64 positions, i32 retained
/// lengths and causal frontiers per row, i32 `[rows, table_width]` page table,
/// i32 `[rows, width]` physical slots.
pub(crate) struct PagedBuffers {
    pub positions: cuteafd_ffi::CuteafdDeviceBuffer,
    pub lengths: cuteafd_ffi::CuteafdDeviceBuffer,
    pub frontiers: cuteafd_ffi::CuteafdDeviceBuffer,
    pub table: cuteafd_ffi::CuteafdDeviceBuffer,
    pub slots: cuteafd_ffi::CuteafdDeviceBuffer,
}

/// Drains `stream`, reads a paged selector's rows back and records them as
/// logical tokens. Called only with a dump attached.
#[allow(clippy::too_many_arguments)]
pub(crate) fn read_paged(dump: &SelectionDump, stream: &crate::shared::memory::device::Stream<'_>, layer: usize,
    rows: usize, width: usize, table_width: usize, page_rows: usize, buffers: PagedBuffers) -> Result<()> {
    use crate::shared::prefix::view;
    stream.drain()?;
    let read = |buffer: cuteafd_ffi::CuteafdDeviceBuffer, bytes: usize| -> Result<Vec<u8>> {
        let mut out = vec![0u8; bytes];
        stream.copy_d2h(&mut out, view(buffer, 0, bytes)?)?;
        Ok(out)
    };
    let positions: Vec<u64> = read(buffers.positions, rows * 8)?.chunks_exact(8)
        .map(|c| i64::from_le_bytes(c.try_into().unwrap()).max(0) as u64).collect();
    if !dump.wants(positions.iter().copied()) { return Ok(()) }
    let lengths = i32s(&read(buffers.lengths, rows * 4)?);
    let frontiers = i32s(&read(buffers.frontiers, rows * 4)?);
    let table = i32s(&read(buffers.table, rows * table_width * 4)?);
    let slots = i32s(&read(buffers.slots, rows * width * 4)?);
    let rows = (0..rows).map(|row| {
        let pages = (frontiers[row].max(0) as usize).div_ceil(page_rows).min(table_width);
        let length = (lengths[row].max(0) as usize).min(width);
        Ok(Row { position: positions[row], frontier: frontiers[row],
            ids: logical_tokens(&slots[row * width..row * width + length],
                &table[row * table_width..row * table_width + pages], page_rows)? })
    }).collect::<Result<Vec<_>>>()?;
    dump.write(layer, Layout { width, unit_tokens: 1, source: layer, heads: 1 }, rows)
}

/// Host copy helpers for the families' raw-stream engines.
pub(crate) fn i32s(bytes: &[u8]) -> Vec<i32> {
    bytes.chunks_exact(4).map(|chunk| i32::from_le_bytes(chunk.try_into().unwrap())).collect()
}

#[cfg(test)]
pub(crate) fn read_layer(path: &std::path::Path) -> Result<(u32, Layout, Vec<(u64, Row)>)> {
    let bytes = std::fs::read(path)?;
    ensure!(bytes.len() >= HEADER_BYTES && &bytes[0..8] == MAGIC, "selection dump header");
    let word = |at: usize| u32::from_le_bytes(bytes[at..at + 4].try_into().unwrap());
    let layout = Layout { width: word(12) as usize, unit_tokens: word(16), source: word(20) as usize, heads: word(24) };
    let rows = u64::from_le_bytes(bytes[32..40].try_into().unwrap()) as usize;
    let size = record_bytes(layout.width);
    ensure!(bytes.len() == HEADER_BYTES + rows * size, "selection dump payload length");
    let records = bytes[HEADER_BYTES..].chunks_exact(size).map(|r| {
        let step = u64::from_le_bytes(r[0..8].try_into().unwrap());
        let position = u64::from_le_bytes(r[8..16].try_into().unwrap());
        let frontier = i32::from_le_bytes(r[16..20].try_into().unwrap());
        let count = i32::from_le_bytes(r[20..24].try_into().unwrap()) as usize;
        let ids = i32s(&r[24..]);
        assert!(ids[count..].iter().all(|&id| id == -1));
        (step, Row { position, frontier, ids: ids[..count].to_vec() })
    }).collect();
    Ok((word(8), layout, records))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::Cell;

    const LAYOUT: Layout = Layout { width: 4, unit_tokens: 128, source: 3, heads: 1 };

    #[test]
    fn unset_resolves_none_and_never_reads_back() -> Result<()> {
        assert!(SelectionDump::from_env_vars("test", None, None)?.is_none());
        assert!(SelectionDump::from_env_vars("test", None, Some(OsStr::new("9")))?.is_none());
        if std::env::var_os(ENV_DIR).is_none() {
            assert!(SelectionDump::from_env("test")?.is_none());
            assert!(shared("test")?.is_none());
        }
        let reads = Cell::new(0);
        let read = || { reads.set(reads.get() + 1); Ok(vec![]) };
        record(None, 3, LAYOUT, &[10, 11], read)?;
        assert_eq!(reads.get(), 0, "unset must not drain or copy");
        // Positive control: a set dump reads back, and FROM gates the readback itself.
        let temp = tempfile::tempdir()?;
        let dump = SelectionDump::from_env_vars("test", Some(temp.path().as_os_str()), Some(OsStr::new("100")))?.unwrap();
        record(Some(&dump), 3, LAYOUT, &[10, 11], || { reads.set(reads.get() + 1); Ok(vec![]) })?;
        assert_eq!(reads.get(), 0, "rows below FROM must not read back");
        record(Some(&dump), 3, LAYOUT, &[99, 100], || { reads.set(reads.get() + 1); Ok(vec![]) })?;
        assert_eq!(reads.get(), 1);
        assert!(SelectionDump::from_env_vars("test", Some(OsStr::new("")), None).is_err());
        assert!(SelectionDump::from_env_vars("test", Some(temp.path().as_os_str()), Some(OsStr::new("x"))).is_err());
        Ok(())
    }

    #[test]
    fn writer_appends_repeated_positions_sorted_and_padded() -> Result<()> {
        let temp = tempfile::tempdir()?;
        let dump = SelectionDump::from_env_vars("test", Some(temp.path().join("d").as_os_str()), Some(OsStr::new("5")))?.unwrap();
        let row = |position, ids: &[i32]| Row { position, frontier: 7, ids: ids.to_vec() };
        dump.write(3, LAYOUT, vec![row(4, &[1]), row(5, &[9, -1, 2])])?;
        // A prefix restore re-runs position 5: both occurrences are kept, in order.
        dump.write(3, LAYOUT, vec![row(5, &[3, 0, 1, 2])])?;
        dump.write(7, Layout { source: 7, ..LAYOUT }, vec![row(6, &[])])?;
        let (layer, layout, records) = read_layer(&temp.path().join("d/layer-003.bin"))?;
        assert_eq!((layer, layout), (3, LAYOUT));
        assert_eq!(records, vec![(0, row(5, &[2, 9])), (1, row(5, &[0, 1, 2, 3]))]);
        assert_eq!(read_layer(&temp.path().join("d/layer-007.bin"))?.2, vec![(0, row(6, &[]))]);
        let lines: Vec<serde_json::Value> = std::fs::read_to_string(temp.path().join("d/manifest.jsonl"))?
            .lines().map(serde_json::from_str).collect::<std::result::Result<_, _>>()?;
        assert_eq!(lines.len(), 3);
        assert_eq!((lines[1]["layer_step"].as_u64(), lines[1]["total_rows"].as_u64()), (Some(1), Some(2)));
        assert_eq!(lines[2]["source_layer"], 7);
        assert!(dump.write(3, LAYOUT, vec![row(8, &[1, 1])]).is_err(), "a repeated unit is a selector defect");
        assert!(dump.write(3, Layout { width: 8, ..LAYOUT }, vec![row(8, &[1])]).is_err());
        assert!(dump.write(3, LAYOUT, vec![row(8, &[1, 2, 3, 4, 5])]).is_err());
        Ok(())
    }

    #[test]
    fn physical_slots_invert_through_the_page_table() -> Result<()> {
        // Logical pages 0,1,2 live in physical pages 7,3,9 (64 rows each).
        let table = [7, 3, 9, -1];
        let slots = [7 * 64 + 5, 3 * 64, 9 * 64 + 63, -1];
        assert_eq!(logical_tokens(&slots, &table, 64)?, vec![5, 64, 191]);
        assert!(logical_tokens(&[4 * 64], &table, 64).is_err());
        Ok(())
    }
}
