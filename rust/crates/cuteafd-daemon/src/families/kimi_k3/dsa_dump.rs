//! Opt-in diagnostic readback of post-selector DSA token indices (`CUTEAFD_KIMI_DSA_DUMP`).
//!
//! Unset `CUTEAFD_KIMI_DSA_DUMP` resolves to `None` at startup with zero stream drains,
//! zero host copies, and zero behavior change. When set, non-graph runs drain the exact
//! producing lane stream after `KimiIndexTopK` and read back `DSA positions`,
//! `DSA selected lengths`, and `DSA indices` (`rows * 2048 * 4`), writing rows whose
//! engine position is `>= CUTEAFD_KIMI_DSA_DUMP_FROM` (default `0`).

use crate::shared::{memory::device::Stream, prefix::view};
use anyhow::{ensure, Context, Result};
use cuteafd_ffi::CuteafdDeviceBuffer;
use std::{
    cell::RefCell,
    collections::BTreeMap,
    ffi::OsStr,
    fs::{File, OpenOptions},
    io::{Read, Seek, SeekFrom, Write},
    path::{Path, PathBuf},
    rc::Rc,
};

pub(crate) const ENV_DIR: &str = "CUTEAFD_KIMI_DSA_DUMP";
pub(crate) const ENV_FROM: &str = "CUTEAFD_KIMI_DSA_DUMP_FROM";
pub(crate) const MAGIC: &[u8; 8] = b"KDSA0001";
pub(crate) const WIDTH: usize = 2048;
pub(crate) const HEADER_BYTES: usize = 24;
pub(crate) const RECORD_BYTES: usize = 8 + 4 + WIDTH * 4;

pub(crate) struct RawBatch {
    pub rows: usize,
    pub positions: Vec<u8>,
    pub lengths: Vec<u8>,
    pub indices: Vec<u8>,
}

pub(crate) trait Readback {
    fn drain_and_read(&mut self, rows: usize) -> Result<RawBatch>;
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct RowRecord {
    pub position: u64,
    pub length: i32,
    pub indices: Vec<i32>,
}

impl RowRecord {
    fn encode_into(&self, out: &mut Vec<u8>) {
        out.extend_from_slice(&self.position.to_le_bytes());
        out.extend_from_slice(&self.length.to_le_bytes());
        for &idx in &self.indices {
            out.extend_from_slice(&idx.to_le_bytes());
        }
    }

    fn decode(bytes: &[u8]) -> Result<Self> {
        ensure!(bytes.len() == RECORD_BYTES, "DSA dump record byte length mismatch");
        let position = u64::from_le_bytes(bytes[0..8].try_into().unwrap());
        let length = i32::from_le_bytes(bytes[8..12].try_into().unwrap());
        let indices = bytes[12..]
            .chunks_exact(4)
            .map(|chunk| i32::from_le_bytes(chunk.try_into().unwrap()))
            .collect::<Vec<_>>();
        Ok(Self { position, length, indices })
    }
}

struct LayerWriter {
    file: File,
    rows: u64,
    steps: u64,
    last_position: Option<u64>,
}

#[derive(Default)]
struct State {
    layers: BTreeMap<usize, LayerWriter>,
    manifest: Option<File>,
    manifest_entries: u64,
}

pub(crate) struct DsaDump {
    dir: PathBuf,
    from: u64,
    state: RefCell<State>,
}

impl DsaDump {
    pub(crate) fn from_env() -> Result<Option<Rc<Self>>> {
        Self::from_env_vars(
            std::env::var_os(ENV_DIR).as_deref(),
            std::env::var_os(ENV_FROM).as_deref(),
        )
    }

    pub(crate) fn from_env_vars(dir: Option<&OsStr>, from: Option<&OsStr>) -> Result<Option<Rc<Self>>> {
        let Some(raw) = dir else {
            return Ok(None);
        };
        ensure!(!raw.is_empty(), "{ENV_DIR} must name a non-empty directory path");
        let from = match from {
            None => 0,
            Some(value) => value
                .to_str()
                .with_context(|| format!("{ENV_FROM} is not valid UTF-8"))?
                .parse::<u64>()
                .with_context(|| format!("invalid {ENV_FROM}"))?,
        };
        Self::new(PathBuf::from(raw), from).map(Some)
    }

    pub(crate) fn new(dir: PathBuf, from: u64) -> Result<Rc<Self>> {
        std::fs::create_dir_all(&dir)
            .with_context(|| format!("create Kimi DSA dump directory {}", dir.display()))?;
        Ok(Rc::new(Self { dir, from, state: RefCell::new(State::default()) }))
    }

    pub(crate) fn maybe_record(
        dump: Option<&Self>,
        rank: usize,
        layer: usize,
        rows: usize,
        readback: &mut impl Readback,
    ) -> Result<()> {
        let Some(dump) = dump.filter(|_| rank == 0) else {
            return Ok(());
        };
        let batch = readback.drain_and_read(rows)?;
        dump.write_batch(layer, batch)
    }

    fn write_batch(&self, layer: usize, batch: RawBatch) -> Result<()> {
        ensure!(batch.rows > 0, "Kimi DSA dump requires positive batch rows");
        ensure!(
            batch.positions.len() == batch.rows * 8
                && batch.lengths.len() == batch.rows * 4
                && batch.indices.len() == batch.rows * WIDTH * 4,
            "Kimi DSA dump buffer extents do not match batch rows {}",
            batch.rows
        );
        let mut kept = Vec::new();
        for row in 0..batch.rows {
            let pos = i64::from_le_bytes(batch.positions[row * 8..(row + 1) * 8].try_into().unwrap());
            ensure!(pos >= 0, "Kimi DSA dump saw negative engine position {pos} at layer {layer}");
            let position = pos as u64;
            if position < self.from {
                continue;
            }
            let length = i32::from_le_bytes(batch.lengths[row * 4..(row + 1) * 4].try_into().unwrap());
            let idx_slice = &batch.indices[row * WIDTH * 4..(row + 1) * WIDTH * 4];
            let indices = idx_slice
                .chunks_exact(4)
                .map(|chunk| i32::from_le_bytes(chunk.try_into().unwrap()))
                .collect::<Vec<_>>();
            kept.push(RowRecord { position, length, indices });
        }
        if kept.is_empty() {
            return Ok(());
        }
        kept.sort_by_key(|record| record.position);
        for pair in kept.windows(2) {
            ensure!(
                pair[0].position < pair[1].position,
                "Kimi DSA dump duplicate position {} in layer {layer}",
                pair[0].position
            );
        }

        let first_position = kept.first().unwrap().position;
        let last_position = kept.last().unwrap().position;
        let kept_rows = kept.len();
        let file_name = format!("layer-{layer:02}.bin");
        let layer_u32 = u32::try_from(layer).context("Kimi DSA layer index overflow")?;

        let mut state = self.state.borrow_mut();
        let (layer_step, total_rows) = {
            if !state.layers.contains_key(&layer) {
                let path = self.dir.join(&file_name);
                let mut file = OpenOptions::new()
                    .read(true)
                    .write(true)
                    .create_new(true)
                    .open(&path)
                    .with_context(|| format!("create DSA layer dump {}", path.display()))?;
                write_header(&mut file, layer_u32, 0)?;
                state.layers.insert(layer, LayerWriter { file, rows: 0, steps: 0, last_position: None });
            }
            let writer = state.layers.get_mut(&layer).unwrap();
            if writer.last_position.is_none_or(|prev| first_position > prev) {
                let mut encoded = Vec::with_capacity(kept_rows * RECORD_BYTES);
                for record in &kept {
                    record.encode_into(&mut encoded);
                }
                writer.file.seek(SeekFrom::End(0))?;
                writer.file.write_all(&encoded)?;
                writer.rows = writer
                    .rows
                    .checked_add(u64::try_from(kept_rows)?)
                    .context("Kimi DSA dump row count overflow")?;
                writer.file.seek(SeekFrom::Start(16))?;
                writer.file.write_all(&writer.rows.to_le_bytes())?;
                writer.file.flush()?;
                writer.last_position = Some(last_position);
            } else {
                let mut existing = read_all_records(&mut writer.file, writer.rows)?;
                existing.extend(kept);
                existing.sort_by_key(|record| record.position);
                for pair in existing.windows(2) {
                    ensure!(
                        pair[0].position < pair[1].position,
                        "Kimi DSA dump duplicate position {} in layer {layer}",
                        pair[0].position
                    );
                }
                let total = u64::try_from(existing.len())?;
                let mut image = Vec::with_capacity(HEADER_BYTES + existing.len() * RECORD_BYTES);
                image.extend_from_slice(MAGIC);
                image.extend_from_slice(&layer_u32.to_le_bytes());
                image.extend_from_slice(&(WIDTH as u32).to_le_bytes());
                image.extend_from_slice(&total.to_le_bytes());
                for record in &existing {
                    record.encode_into(&mut image);
                }
                writer.file.set_len(0)?;
                writer.file.seek(SeekFrom::Start(0))?;
                writer.file.write_all(&image)?;
                writer.file.flush()?;
                writer.rows = total;
                writer.last_position = existing.last().map(|record| record.position);
            }
            let step = writer.steps;
            writer.steps += 1;
            (step, writer.rows)
        };

        if state.manifest.is_none() {
            let manifest_path = self.dir.join("manifest.jsonl");
            let file = OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(&manifest_path)
                .with_context(|| format!("create DSA dump manifest {}", manifest_path.display()))?;
            state.manifest = Some(file);
        }
        let seq = state.manifest_entries;
        state.manifest_entries += 1;
        let manifest = state.manifest.as_mut().unwrap();
        let entry = serde_json::json!({
            "schema": "cuteafd.kimi.dsa-dump/1",
            "seq": seq,
            "layer_step": layer_step,
            "layer": layer,
            "file": file_name,
            "batch_rows": batch.rows,
            "kept_rows": kept_rows,
            "total_rows": total_rows,
            "first_position": first_position,
            "last_position": last_position,
            "from": self.from,
            "width": WIDTH,
            "header_bytes": HEADER_BYTES,
            "record_bytes": RECORD_BYTES,
        });
        serde_json::to_writer(&mut *manifest, &entry)?;
        manifest.write_all(b"\n")?;
        manifest.flush()?;
        Ok(())
    }
}

fn write_header(file: &mut File, layer: u32, rows: u64) -> Result<()> {
    let mut header = [0u8; HEADER_BYTES];
    header[0..8].copy_from_slice(MAGIC);
    header[8..12].copy_from_slice(&layer.to_le_bytes());
    header[12..16].copy_from_slice(&(WIDTH as u32).to_le_bytes());
    header[16..24].copy_from_slice(&rows.to_le_bytes());
    file.seek(SeekFrom::Start(0))?;
    file.write_all(&header)?;
    file.flush()?;
    Ok(())
}

fn read_all_records(file: &mut File, count: u64) -> Result<Vec<RowRecord>> {
    let count = usize::try_from(count)?;
    file.seek(SeekFrom::Start(HEADER_BYTES as u64))?;
    let mut raw = vec![0u8; count * RECORD_BYTES];
    file.read_exact(&mut raw)?;
    raw.chunks_exact(RECORD_BYTES).map(RowRecord::decode).collect()
}

pub(crate) fn read_layer_file(path: &Path) -> Result<(u32, u32, Vec<RowRecord>)> {
    let bytes = std::fs::read(path)?;
    ensure!(bytes.len() >= HEADER_BYTES, "DSA layer dump shorter than header");
    ensure!(&bytes[0..8] == MAGIC, "DSA layer dump magic mismatch");
    let layer = u32::from_le_bytes(bytes[8..12].try_into().unwrap());
    let width = u32::from_le_bytes(bytes[12..16].try_into().unwrap());
    let rows = u64::from_le_bytes(bytes[16..24].try_into().unwrap());
    ensure!(width as usize == WIDTH, "DSA layer dump width {width} != {WIDTH}");
    let body = &bytes[HEADER_BYTES..];
    ensure!(
        body.len() == usize::try_from(rows)? * RECORD_BYTES,
        "DSA layer dump payload length {} != rows {rows} * {RECORD_BYTES}",
        body.len()
    );
    let records = body.chunks_exact(RECORD_BYTES).map(RowRecord::decode).collect::<Result<Vec<_>>>()?;
    Ok((layer, width, records))
}

struct DeviceReadback<'s, 'a, F> {
    stream: &'s Stream<'a>,
    buffers: Option<F>,
}

impl<F> Readback for DeviceReadback<'_, '_, F>
where
    F: FnOnce() -> Result<(
        CuteafdDeviceBuffer,
        CuteafdDeviceBuffer,
        CuteafdDeviceBuffer,
    )>,
{
    fn drain_and_read(&mut self, rows: usize) -> Result<RawBatch> {
        ensure!(rows > 0, "Kimi DSA dump requires positive rows");
        let buffers = self.buffers.take().context("Kimi DSA readback buffers already consumed")?;
        let (positions, selected_lengths, indices) = buffers()?;
        let pos_view = view(positions, 0, rows.checked_mul(8).context("DSA positions byte overflow")?)?;
        let len_view = view(selected_lengths, 0, rows.checked_mul(4).context("DSA lengths byte overflow")?)?;
        let idx_view = view(indices, 0, rows.checked_mul(WIDTH * 4).context("DSA indices byte overflow")?)?;
        // Drain the exact producing lane stream before host readback, matching Capture::record.
        self.stream.drain()?;
        let stream = self.stream;
        stream.device.run(|| {
            let mut pos_bytes = vec![0u8; pos_view.bytes];
            let mut len_bytes = vec![0u8; len_view.bytes];
            let mut idx_bytes = vec![0u8; idx_view.bytes];
            stream.device.library.copy_d2h(&mut pos_bytes, pos_view)?;
            stream.device.library.copy_d2h(&mut len_bytes, len_view)?;
            stream.device.library.copy_d2h(&mut idx_bytes, idx_view)?;
            Ok(RawBatch { rows, positions: pos_bytes, lengths: len_bytes, indices: idx_bytes })
        })
    }
}

pub(crate) fn record_selected(
    dump: Option<&DsaDump>,
    rank: usize,
    stream: &Stream<'_>,
    layer: usize,
    rows: usize,
    buffers: impl FnOnce() -> Result<(
        CuteafdDeviceBuffer,
        CuteafdDeviceBuffer,
        CuteafdDeviceBuffer,
    )>,
) -> Result<()> {
    let mut readback = DeviceReadback { stream, buffers: Some(buffers) };
    DsaDump::maybe_record(dump, rank, layer, rows, &mut readback)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::Cell;

    fn synthetic_indices(seed: i32) -> Vec<i32> {
        (0..WIDTH as i32)
            .map(|col| if col + 4 >= WIDTH as i32 { -1 } else { seed * 10_000 + col })
            .collect()
    }

    fn make_raw_batch(rows: &[(i64, i32, i32)]) -> RawBatch {
        let mut positions = Vec::with_capacity(rows.len() * 8);
        let mut lengths = Vec::with_capacity(rows.len() * 4);
        let mut indices = Vec::with_capacity(rows.len() * WIDTH * 4);
        for &(pos, len, seed) in rows {
            positions.extend_from_slice(&pos.to_le_bytes());
            lengths.extend_from_slice(&len.to_le_bytes());
            for value in synthetic_indices(seed) {
                indices.extend_from_slice(&value.to_le_bytes());
            }
        }
        RawBatch { rows: rows.len(), positions, lengths, indices }
    }

    struct FixedReadback(Option<RawBatch>);

    impl Readback for FixedReadback {
        fn drain_and_read(&mut self, rows: usize) -> Result<RawBatch> {
            let batch = self.0.take().context("readback already consumed")?;
            ensure!(batch.rows == rows, "synthetic row count mismatch");
            Ok(batch)
        }
    }

    struct CountingFixture<'a> {
        readbacks: &'a Cell<usize>,
        batch: RawBatch,
    }

    impl Readback for CountingFixture<'_> {
        fn drain_and_read(&mut self, rows: usize) -> Result<RawBatch> {
            self.readbacks.set(self.readbacks.get() + 1);
            ensure!(self.batch.rows == rows, "counting fixture row mismatch");
            Ok(RawBatch {
                rows: self.batch.rows,
                positions: self.batch.positions.clone(),
                lengths: self.batch.lengths.clone(),
                indices: self.batch.indices.clone(),
            })
        }
    }

    #[test]
    fn writer_filters_below_from_sorts_positions_and_records_each_window_layer_once() -> Result<()> {
        let temp = tempfile::tempdir()?;
        let dump_dir = temp.path().join("nested/dsa");
        let dump = DsaDump::from_env_vars(Some(dump_dir.as_os_str()), Some(OsStr::new("4279")))?
            .expect("set env creates dump");
        assert!(dump_dir.is_dir());

        // Step 0: non-window prefix chunk (all positions < 4279) writes no files or manifest lines.
        for layer in [3usize, 7usize] {
            let mut step0 = FixedReadback(Some(make_raw_batch(&[
                (4276, 2048, 1),
                (4277, 2048, 2),
                (4278, 2048, 3),
            ])));
            DsaDump::maybe_record(Some(&dump), 0, layer, 3, &mut step0)?;
        }
        assert!(!dump_dir.join("manifest.jsonl").exists());
        assert!(!dump_dir.join("layer-03.bin").exists());

        // Step 1: mixed < FROM and >= FROM unsorted positions on layers 3 and 7.
        for layer in [3usize, 7usize] {
            let mut step1 = FixedReadback(Some(make_raw_batch(&[
                (4281, 2048, 11),
                (4278, 2048, 99),
                (4279, 2044, 10),
            ])));
            DsaDump::maybe_record(Some(&dump), 0, layer, 3, &mut step1)?;
        }

        // Rank 1 is ignored so TP2 never duplicates layer entries.
        let mut ignored_rank = FixedReadback(Some(make_raw_batch(&[(4290, 2048, 50)])));
        DsaDump::maybe_record(Some(&dump), 1, 3, 1, &mut ignored_rank)?;
        assert!(ignored_rank.0.is_some(), "non-zero rank must not read back");

        // Step 2: out-of-order position 4280 merges into sorted order; Step 3: monotonic 4282 appends.
        let mut step2 = FixedReadback(Some(make_raw_batch(&[(4280, 2045, 12)])));
        DsaDump::maybe_record(Some(&dump), 0, 3, 1, &mut step2)?;
        let mut step3 = FixedReadback(Some(make_raw_batch(&[(4282, 2048, 13)])));
        DsaDump::maybe_record(Some(&dump), 0, 3, 1, &mut step3)?;

        let (layer_id, width, records_l3) = read_layer_file(&dump_dir.join("layer-03.bin"))?;
        assert_eq!((layer_id, width), (3, WIDTH as u32));
        assert_eq!(records_l3.len(), 4);
        assert_eq!(
            records_l3.iter().map(|r| (r.position, r.length)).collect::<Vec<_>>(),
            vec![(4279, 2044), (4280, 2045), (4281, 2048), (4282, 2048)]
        );
        assert_eq!(records_l3[0].indices, synthetic_indices(10));
        assert_eq!(records_l3[1].indices, synthetic_indices(12));
        assert_eq!(records_l3[2].indices, synthetic_indices(11));
        assert_eq!(records_l3[3].indices, synthetic_indices(13));

        let (layer7_id, width7, records_l7) = read_layer_file(&dump_dir.join("layer-07.bin"))?;
        assert_eq!((layer7_id, width7), (7, WIDTH as u32));
        assert_eq!(
            records_l7.iter().map(|r| (r.position, r.length)).collect::<Vec<_>>(),
            vec![(4279, 2044), (4281, 2048)]
        );

        let manifest_lines = std::fs::read_to_string(dump_dir.join("manifest.jsonl"))?
            .lines()
            .map(serde_json::from_str::<serde_json::Value>)
            .collect::<std::result::Result<Vec<_>, _>>()?;
        assert_eq!(manifest_lines.len(), 4, "step1 (layers 3,7) + step2 (layer 3) + step3 (layer 3)");
        assert_eq!(manifest_lines[0]["layer"], 3);
        assert_eq!(manifest_lines[0]["layer_step"], 0);
        assert_eq!(manifest_lines[0]["kept_rows"], 2);
        assert_eq!(manifest_lines[0]["first_position"], 4279);
        assert_eq!(manifest_lines[0]["last_position"], 4281);
        assert_eq!(manifest_lines[1]["layer"], 7);
        assert_eq!(manifest_lines[1]["layer_step"], 0);
        assert_eq!(manifest_lines[2]["layer"], 3);
        assert_eq!(manifest_lines[2]["layer_step"], 1);
        assert_eq!(manifest_lines[3]["layer"], 3);
        assert_eq!(manifest_lines[3]["layer_step"], 2);
        assert_eq!(manifest_lines[3]["total_rows"], 4);
        Ok(())
    }

    #[test]
    fn unset_env_resolves_none_and_attempts_no_sync_or_readback() -> Result<()> {
        assert!(DsaDump::from_env_vars(None, None)?.is_none());
        assert!(DsaDump::from_env_vars(None, Some(OsStr::new("4279")))?.is_none());
        if std::env::var_os(ENV_DIR).is_none() {
            assert!(DsaDump::from_env()?.is_none());
        }

        let readbacks = Cell::new(0usize);
        let mut counting = CountingFixture {
            readbacks: &readbacks,
            batch: make_raw_batch(&[(4279, 2048, 1)]),
        };
        let unset = DsaDump::from_env_vars(None, Some(OsStr::new("4279")))?;
        DsaDump::maybe_record(unset.as_deref(), 0, 3, 1, &mut counting)?;
        assert_eq!(readbacks.get(), 0, "unset env must attempt zero readbacks");

        // Positive control on the same counting fixture confirms it increments when active.
        let temp = tempfile::tempdir()?;
        let enabled = DsaDump::from_env_vars(
            Some(temp.path().join("enabled").as_os_str()),
            Some(OsStr::new("4279")),
        )?;
        DsaDump::maybe_record(enabled.as_deref(), 0, 3, 1, &mut counting)?;
        assert_eq!(readbacks.get(), 1);
        Ok(())
    }
}
