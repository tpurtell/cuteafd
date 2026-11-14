//! Opt-in diagnostic readback of the DSA scored selector's per-slot scores
//! (`CUTEAFD_KIMI_DSA_SCORES`).
//!
//! Unset `CUTEAFD_KIMI_DSA_SCORES` resolves to `None` at startup with zero
//! stream drains, zero host copies, and zero behavior change. When set, each
//! non-graph, non-staging DSA step launches the checkpoint's scored selector
//! (`programs.scored`) beside the live plain selector, writing its retained
//! sort values into `DSA scores` and its logical ids into a `rows * 2048 * 4`
//! view of `DSA producer scratch` (already dead for this step). The twin's
//! logical selection is mapped through the same page table the plain route
//! read and must equal the plain route's physical selection exactly; the row
//! is written only after that check, because the scores are only meaningful
//! for a faithful stand-in. `CUTEAFD_KIMI_DSA_SCORE_LAYERS` (default: every DSA
//! layer) restricts the launch and `CUTEAFD_KIMI_DSA_SCORE_FROM` (default `0`)
//! restricts written engine positions.
//!
//! The scored selector returns its 2048 retained slots in descending score
//! order, `scores[i]` aligned with `indices[i]` (see the fork's `_ScoreOrder`).
//! Indices are cache-logical token positions over the whole causal frontier,
//! not physical slots; each recorded row keeps the frontier and the writer
//! stores the indices as returned, marking `logical_indices` in its manifest.

use crate::shared::{memory::device::Stream, prefix::view};
use anyhow::{ensure, Context, Result};
use cuteafd_ffi::CuteafdDeviceBuffer;
use std::{
    cell::RefCell,
    collections::{BTreeMap, BTreeSet},
    ffi::OsStr,
    fs::{File, OpenOptions},
    io::{Seek, SeekFrom, Write},
    path::PathBuf,
    rc::Rc,
};

pub(crate) const ENV_DIR: &str = "CUTEAFD_KIMI_DSA_SCORES";
pub(crate) const ENV_LAYERS: &str = "CUTEAFD_KIMI_DSA_SCORE_LAYERS";
pub(crate) const ENV_FROM: &str = "CUTEAFD_KIMI_DSA_SCORE_FROM";
pub(crate) const MAGIC: &[u8; 8] = b"KDSAS001";
pub(crate) const WIDTH: usize = 2048;
pub(crate) const PAGE_ROWS: usize = 64;
pub(crate) const HEADER_BYTES: usize = 24;
pub(crate) const RECORD_BYTES: usize = 8 + 4 + WIDTH * 4 * 2;

/// One dumped position: retained scores and their aligned logical token ids.
/// `length` is the causal frontier (the selector's candidate count), not the
/// retained count; the retained count is `min(length, 2048)`.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct ScoredRow {
    pub position: u64,
    pub length: i32,
    pub scores: Vec<f32>,
    pub indices: Vec<i32>,
}

impl ScoredRow {
    fn encode_into(&self, out: &mut Vec<u8>) {
        out.extend_from_slice(&self.position.to_le_bytes());
        out.extend_from_slice(&self.length.to_le_bytes());
        for &score in &self.scores {
            out.extend_from_slice(&score.to_le_bytes());
        }
        for &index in &self.indices {
            out.extend_from_slice(&index.to_le_bytes());
        }
    }

    fn decode(bytes: &[u8]) -> Result<Self> {
        ensure!(bytes.len() == RECORD_BYTES, "Kimi DSA score record byte length mismatch");
        let position = u64::from_le_bytes(bytes[0..8].try_into().unwrap());
        let length = i32::from_le_bytes(bytes[8..12].try_into().unwrap());
        let scores = bytes[12..12 + WIDTH * 4]
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
            .collect::<Vec<_>>();
        let indices = bytes[12 + WIDTH * 4..]
            .chunks_exact(4)
            .map(|chunk| i32::from_le_bytes(chunk.try_into().unwrap()))
            .collect::<Vec<_>>();
        Ok(Self { position, length, scores, indices })
    }
}

/// Host-side images of one producing batch. `plain`, `twin`, and `scores` are
/// `rows * 2048 * 4` bytes; `lengths` holds the retained counts and `frontiers`
/// the causal candidate counts the selector read; `table` holds this route's
/// page table rows.
pub(crate) struct ScoreBatch {
    pub rows: usize,
    pub per_row_table: bool,
    pub table_width: usize,
    pub positions: Vec<u8>,
    pub lengths: Vec<u8>,
    pub frontiers: Vec<u8>,
    pub plain: Vec<u8>,
    pub twin: Vec<u8>,
    pub scores: Vec<u8>,
    pub table: Vec<u8>,
}

fn ints(bytes: &[u8]) -> impl Iterator<Item = i32> + '_ {
    bytes.chunks_exact(4).map(|chunk| i32::from_le_bytes(chunk.try_into().unwrap()))
}

fn floats(bytes: &[u8]) -> impl Iterator<Item = f32> + '_ {
    bytes.chunks_exact(4).map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
}

impl ScoreBatch {
    fn table_row(&self, row: usize) -> Result<Vec<i32>> {
        let first = if self.per_row_table { row * self.table_width } else { 0 };
        let bytes = first * 4;
        let end = bytes + self.table_width * 4;
        ensure!(end <= self.table.len(), "Kimi DSA score table extent past host image");
        Ok(ints(&self.table[bytes..end]).collect())
    }
}

/// The twin writes logical token positions over its whole causal frontier; the
/// live plain route writes physical slots. Map the twin through the same page
/// table and require exact set equality. Both routes retain the 2048
/// highest-scoring causal tokens, so a mismatch means the scored route is not
/// a faithful stand-in.
pub(crate) fn verify_selection(
    layer: usize,
    position: u64,
    frontier: i32,
    selected: i32,
    plain: &[i32],
    indices: &[i32],
    scores: &[f32],
    table_row: &[i32],
) -> Result<()> {
    ensure!(frontier >= 0, "Kimi DSA score row frontier {frontier} is negative");
    let frontier = frontier as usize;
    let retained = frontier.min(WIDTH);
    ensure!(selected as usize == retained,
        "Kimi DSA score row retains {selected} of frontier {frontier}, expected {retained}");
    let length = retained;
    ensure!(plain.len() == WIDTH && indices.len() == WIDTH && scores.len() == WIDTH,
        "Kimi DSA score row width differs from 2048");
    let mut expected = BTreeSet::new();
    let mut live = BTreeSet::new();
    for &physical in &plain[..length] {
        ensure!(physical >= 0, "Kimi DSA plain selection holds padding inside position {position}");
        ensure!(live.insert(physical), "Kimi DSA plain selection repeats physical slot {physical} at position {position}");
    }
    let mut seen = BTreeSet::new();
    for i in 0..length {
        let logical = indices[i];
        ensure!(logical >= 0, "Kimi DSA scored twin holds padding at ranked slot {i} of position {position}");
        let logical = logical as usize;
        ensure!(logical < frontier,
            "Kimi DSA scored twin index {logical} outside its causal frontier {frontier} at position {position}");
        ensure!(seen.insert(logical),
            "Kimi DSA scored twin repeats logical token {logical} at position {position}");
        let page = *table_row.get(logical / PAGE_ROWS)
            .with_context(|| format!("Kimi DSA scored twin page {} outside the page table at position {position}",
                logical / PAGE_ROWS))?;
        ensure!(page >= 0, "Kimi DSA scored twin maps logical {logical} to padding page {page} at position {position}");
        expected.insert(page * PAGE_ROWS as i32 + (logical % PAGE_ROWS) as i32);
    }
    let symdiff = expected.symmetric_difference(&live).count();
    ensure!(symdiff == 0,
        "Kimi DSA scored twin selection differs from the plain route: layer {layer} position {position} symdiff {symdiff}");
    for pair in scores[..length].windows(2) {
        ensure!(pair[0].is_finite() && pair[1].is_finite(),
            "Kimi DSA scored twin holds a non-finite score at position {position}");
        ensure!(pair[0] >= pair[1],
            "Kimi DSA scored twin scores are not descending at position {position}");
    }
    Ok(())
}

#[derive(Default)]
struct LayerWriter {
    records: BTreeMap<u64, ScoredRow>,
    file: Option<File>,
}

#[derive(Default)]
struct State {
    layers: BTreeMap<usize, LayerWriter>,
    manifest: Option<File>,
    manifest_entries: u64,
}

pub(crate) struct ScoreDump {
    dir: PathBuf,
    from: u64,
    layers: BTreeSet<usize>,
    state: RefCell<State>,
}

impl ScoreDump {
    pub(crate) fn from_env() -> Result<Option<Rc<Self>>> {
        Self::from_env_vars(
            std::env::var_os(ENV_DIR).as_deref(),
            std::env::var_os(ENV_LAYERS).as_deref(),
            std::env::var_os(ENV_FROM).as_deref(),
        )
    }

    pub(crate) fn from_env_vars(dir: Option<&OsStr>, layers: Option<&OsStr>, from: Option<&OsStr>)
        -> Result<Option<Rc<Self>>> {
        let Some(raw) = dir else { return Ok(None) };
        ensure!(!raw.is_empty(), "{ENV_DIR} must name a non-empty directory path");
        let mut selected = BTreeSet::new();
        if let Some(value) = layers.filter(|value| !value.is_empty()) {
            for part in value.to_str().with_context(|| format!("{ENV_LAYERS} is not valid UTF-8"))?
                .split(',').filter(|part| !part.is_empty()) {
                let layer = part.trim().parse::<usize>()
                    .with_context(|| format!("invalid {ENV_LAYERS} entry {part:?}"))?;
                ensure!(layer < 128, "{ENV_LAYERS} layer {layer} outside 0..128");
                selected.insert(layer);
            }
        }
        let from = match from {
            None => 0,
            Some(value) => value.to_str().with_context(|| format!("{ENV_FROM} is not valid UTF-8"))?
                .parse::<u64>().with_context(|| format!("invalid {ENV_FROM}"))?,
        };
        Self::new(PathBuf::from(raw), selected, from).map(Some)
    }

    pub(crate) fn new(dir: PathBuf, layers: BTreeSet<usize>, from: u64) -> Result<Rc<Self>> {
        std::fs::create_dir_all(&dir)
            .with_context(|| format!("create Kimi DSA score directory {}", dir.display()))?;
        Ok(Rc::new(Self { dir, from, layers, state: RefCell::new(State::default()) }))
    }

    /// An empty filter selects every DSA layer the family instantiates.
    pub(crate) fn enabled(&self, layer: usize) -> bool {
        self.layers.is_empty() || self.layers.contains(&layer)
    }

    fn write_batch(&self, layer: usize, batch: ScoreBatch) -> Result<()> {
        ensure!(batch.rows > 0, "Kimi DSA score dump requires positive batch rows");
        ensure!(batch.table_width > 0, "Kimi DSA score dump requires a page table width");
        ensure!(
            batch.positions.len() == batch.rows * 8
                && batch.lengths.len() == batch.rows * 4
                && batch.frontiers.len() == batch.rows * 4
                && batch.plain.len() == batch.rows * WIDTH * 4
                && batch.twin.len() == batch.rows * WIDTH * 4
                && batch.scores.len() == batch.rows * WIDTH * 4,
            "Kimi DSA score dump buffer extents do not match batch rows {}",
            batch.rows
        );
        let mut kept = Vec::new();
        for row in 0..batch.rows {
            let pos = i64::from_le_bytes(batch.positions[row * 8..(row + 1) * 8].try_into().unwrap());
            ensure!(pos >= 0, "Kimi DSA score dump saw negative engine position {pos} at layer {layer}");
            if (pos as u64) < self.from {
                continue;
            }
            let length = i32::from_le_bytes(batch.lengths[row * 4..(row + 1) * 4].try_into().unwrap());
            let frontier = i32::from_le_bytes(batch.frontiers[row * 4..(row + 1) * 4].try_into().unwrap());
            let span = row * WIDTH * 4..(row + 1) * WIDTH * 4;
            let plain = ints(&batch.plain[span.clone()]).collect::<Vec<_>>();
            let indices = ints(&batch.twin[span.clone()]).collect::<Vec<_>>();
            let scores = floats(&batch.scores[span]).collect::<Vec<_>>();
            let table_row = batch.table_row(row)?;
            verify_selection(layer, pos as u64, frontier, length, &plain, &indices, &scores, &table_row)?;
            kept.push(ScoredRow { position: pos as u64, length: frontier, scores, indices });
        }
        if kept.is_empty() {
            return Ok(());
        }
        kept.sort_by_key(|record| record.position);
        for pair in kept.windows(2) {
            ensure!(pair[0].position < pair[1].position,
                "Kimi DSA score dump duplicate position {} in layer {layer}", pair[0].position);
        }
        let first_position = kept.first().unwrap().position;
        let last_position = kept.last().unwrap().position;
        let kept_rows = kept.len();
        let file_name = format!("layer-{layer:02}-scores.bin");
        let layer_u32 = u32::try_from(layer).context("Kimi DSA score layer index overflow")?;

        let mut state = self.state.borrow_mut();
        let (layer_step, total_rows) = {
            let writer = state.layers.entry(layer).or_default();
            if writer.file.is_none() {
                let path = self.dir.join(&file_name);
                let file = OpenOptions::new().read(true).write(true).create_new(true)
                    .open(&path).with_context(|| format!("create DSA score layer dump {}", path.display()))?;
                writer.file = Some(file);
            }
            for record in kept {
                let position = record.position;
                ensure!(writer.records.insert(position, record).is_none(),
                    "Kimi DSA score dump duplicate position {position} in layer {layer}");
            }
            let total = u64::try_from(writer.records.len())?;
            let mut image = Vec::with_capacity(HEADER_BYTES + writer.records.len() * RECORD_BYTES);
            image.extend_from_slice(MAGIC);
            image.extend_from_slice(&layer_u32.to_le_bytes());
            image.extend_from_slice(&(WIDTH as u32).to_le_bytes());
            image.extend_from_slice(&total.to_le_bytes());
            for record in writer.records.values() {
                record.encode_into(&mut image);
            }
            let file = writer.file.as_mut().unwrap();
            file.set_len(0)?;
            file.seek(SeekFrom::Start(0))?;
            file.write_all(&image)?;
            file.flush()?;
            let step = state.manifest_entries;
            state.manifest_entries += 1;
            (step, total)
        };

        if state.manifest.is_none() {
            let manifest_path = self.dir.join("manifest.jsonl");
            let file = OpenOptions::new().write(true).create_new(true).open(&manifest_path)
                .with_context(|| format!("create DSA score dump manifest {}", manifest_path.display()))?;
            state.manifest = Some(file);
        }
        let manifest = state.manifest.as_mut().unwrap();
        let entry = serde_json::json!({
            "schema": "cuteafd.kimi.dsa-scores/1",
            "seq": layer_step,
            "layer": layer,
            "file": file_name,
            "batch_rows": batch.rows,
            "kept_rows": kept_rows,
            "total_rows": total_rows,
            "first_position": first_position,
            "last_position": last_position,
            "from": self.from,
            "layers": self.layers.iter().copied().collect::<Vec<_>>(),
            "width": WIDTH,
            "page_rows": PAGE_ROWS,
            "per_row_table": batch.per_row_table,
            "table_width": batch.table_width,
            "header_bytes": HEADER_BYTES,
            "record_bytes": RECORD_BYTES,
            "logical_indices": true,
            "order": "score-descending",
            "length": "causal frontier (candidate count) read by the selector; retained slots are min(length, width)",
        });
        serde_json::to_writer(&mut *manifest, &entry)?;
        manifest.write_all(b"\n")?;
        manifest.flush()?;
        Ok(())
    }
}

pub(crate) fn read_layer_file(path: &std::path::Path) -> Result<(u32, u32, Vec<ScoredRow>)> {
    let bytes = std::fs::read(path)?;
    ensure!(bytes.len() >= HEADER_BYTES, "DSA score layer dump shorter than header");
    ensure!(&bytes[0..8] == MAGIC, "DSA score layer dump magic mismatch");
    let layer = u32::from_le_bytes(bytes[8..12].try_into().unwrap());
    let width = u32::from_le_bytes(bytes[12..16].try_into().unwrap());
    let rows = u64::from_le_bytes(bytes[16..24].try_into().unwrap());
    ensure!(width as usize == WIDTH, "DSA score layer dump width {width} != {WIDTH}");
    let body = &bytes[HEADER_BYTES..];
    ensure!(body.len() == usize::try_from(rows)? * RECORD_BYTES,
        "DSA score layer dump payload length {} != rows {rows} * {RECORD_BYTES}", body.len());
    let records = body.chunks_exact(RECORD_BYTES).map(ScoredRow::decode).collect::<Result<Vec<_>>>()?;
    Ok((layer, width, records))
}

/// Buffers a scored-twin readback needs. `twin` is a caller-owned view of
/// `DSA producer scratch`; `scores` is the `DSA scores` owner; `frontiers` is
/// the `DSA lengths` operand the selector read.
pub(crate) struct ScoreBuffers {
    pub positions: CuteafdDeviceBuffer,
    pub lengths: CuteafdDeviceBuffer,
    pub frontiers: CuteafdDeviceBuffer,
    pub plain: CuteafdDeviceBuffer,
    pub twin: CuteafdDeviceBuffer,
    pub scores: CuteafdDeviceBuffer,
    pub table: CuteafdDeviceBuffer,
}

/// Drain the producing lane stream, read the twin and the plain selection
/// back, and verify/write every position at or above the window floor.
#[allow(clippy::too_many_arguments)]
pub(crate) fn record_scores(
    dump: &ScoreDump,
    rank: usize,
    stream: &Stream<'_>,
    layer: usize,
    rows: usize,
    table_width: usize,
    per_row_table: bool,
    buffers: impl FnOnce() -> Result<ScoreBuffers>,
) -> Result<()> {
    if rank != 0 {
        return Ok(());
    }
    ensure!(rows > 0, "Kimi DSA score dump requires positive rows");
    let buffers = buffers()?;
    let pos_view = view(buffers.positions, 0, rows.checked_mul(8).context("DSA positions byte overflow")?)?;
    let len_view = view(buffers.lengths, 0, rows.checked_mul(4).context("DSA lengths byte overflow")?)?;
    let frontier_view = view(buffers.frontiers, 0, rows.checked_mul(4).context("DSA frontier byte overflow")?)?;
    let row_bytes = rows.checked_mul(WIDTH * 4).context("DSA score row byte overflow")?;
    let plain_view = view(buffers.plain, 0, row_bytes)?;
    let twin_view = view(buffers.twin, 0, row_bytes)?;
    let score_view = view(buffers.scores, 0, row_bytes)?;
    let table_rows = if per_row_table { rows } else { 1 };
    let table_view = view(buffers.table, 0,
        table_rows.checked_mul(table_width * 4).context("DSA page table byte overflow")?)?;
    // Drain the exact producing lane stream before host readback, matching Capture::record.
    stream.drain()?;
    let batch = stream.device.run(|| {
        let mut positions = vec![0u8; pos_view.bytes];
        let mut lengths = vec![0u8; len_view.bytes];
        let mut frontiers = vec![0u8; frontier_view.bytes];
        let mut plain = vec![0u8; plain_view.bytes];
        let mut twin = vec![0u8; twin_view.bytes];
        let mut scores = vec![0u8; score_view.bytes];
        let mut table = vec![0u8; table_view.bytes];
        for (host, device) in [(&mut positions, pos_view), (&mut lengths, len_view),
            (&mut frontiers, frontier_view), (&mut plain, plain_view),
            (&mut twin, twin_view), (&mut scores, score_view), (&mut table, table_view)] {
            stream.device.library.copy_d2h(host, device)?;
        }
        Ok(ScoreBatch { rows, per_row_table, table_width, positions, lengths, frontiers, plain, twin, scores, table })
    })?;
    dump.write_batch(layer, batch)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn identity_table(width: usize) -> Vec<i32> {
        (0..width as i32).collect()
    }

    /// Logical ids descending by score: the retained highest ids of a causal
    /// frontier, 2048 at most.
    fn twin_row(frontier: usize) -> (Vec<f32>, Vec<i32>) {
        let retained = frontier.min(WIDTH);
        let scores = (0..WIDTH).map(|i| if i < retained { 1.0 / (i as f32 + 1.0) } else { f32::NEG_INFINITY })
            .collect::<Vec<_>>();
        let mut indices = ((frontier - retained)..frontier).rev().map(|i| i as i32).collect::<Vec<_>>();
        indices.resize(WIDTH, -1);
        (scores, indices)
    }

    /// Physical slots for a route whose output equals the twin's logical ids
    /// under an identity page table.
    fn plain_from(indices: &[i32]) -> Vec<i32> {
        let mut plain = indices.iter().copied().filter(|&value| value >= 0).collect::<Vec<_>>();
        plain.resize(WIDTH, -1);
        plain
    }

    fn batch_from(entries: &[(i64, usize)]) -> ScoreBatch {
        let mut positions = Vec::with_capacity(entries.len() * 8);
        let mut lengths = Vec::with_capacity(entries.len() * 4);
        let mut frontiers = Vec::with_capacity(entries.len() * 4);
        let mut plain = Vec::with_capacity(entries.len() * WIDTH * 4);
        let mut twin = Vec::with_capacity(entries.len() * WIDTH * 4);
        let mut scores = Vec::with_capacity(entries.len() * WIDTH * 4);
        for &(position, frontier) in entries {
            let retained = frontier.min(WIDTH);
            positions.extend_from_slice(&position.to_le_bytes());
            lengths.extend_from_slice(&(retained as i32).to_le_bytes());
            frontiers.extend_from_slice(&(frontier as i32).to_le_bytes());
            let (row_scores, row_indices) = twin_row(frontier);
            for value in plain_from(&row_indices) { plain.extend_from_slice(&value.to_le_bytes()); }
            for value in row_scores { scores.extend_from_slice(&value.to_le_bytes()); }
            for value in row_indices { twin.extend_from_slice(&value.to_le_bytes()); }
        }
        let width = 256;
        ScoreBatch { rows: entries.len(), per_row_table: false, table_width: width,
            positions, lengths, frontiers, plain, twin, scores,
            table: identity_table(width).iter().flat_map(|value| value.to_le_bytes()).collect() }
    }

    #[test]
    fn unset_env_resolves_none_and_layer_filter_defaults_to_every_layer() -> Result<()> {
        assert!(ScoreDump::from_env_vars(None, None, None)?.is_none());
        assert!(ScoreDump::from_env_vars(None, Some(OsStr::new("3")), Some(OsStr::new("4279")))?.is_none());

        let temp = tempfile::tempdir()?;
        let dir = temp.path().join("scores");
        let all = ScoreDump::from_env_vars(Some(dir.as_os_str()), None, Some(OsStr::new("4279")))?
            .expect("set env creates dump");
        assert!(all.enabled(3) && all.enabled(59));
        let only = ScoreDump::from_env_vars(Some(dir.join("only").as_os_str()),
            Some(OsStr::new("3,59")), None)?.expect("layers filter creates dump");
        assert!(only.enabled(3) && only.enabled(59) && !only.enabled(7));
        assert!(ScoreDump::from_env_vars(Some(dir.join("bad").as_os_str()),
            Some(OsStr::new("three")), None).is_err());
        Ok(())
    }

    #[test]
    fn twin_selection_maps_through_the_page_table_and_mismatch_names_the_row() -> Result<()> {
        // Deep frontier: logical ids exceed 2048, and only the leading 2048 of
        // the full frontier are retained.
        let frontier = 3000usize;
        let retained = WIDTH;
        let (scores, indices) = twin_row(frontier);
        let plain = plain_from(&indices);
        assert_eq!(indices[0], (frontier - 1) as i32);
        verify_selection(3, 4279, frontier as i32, retained as i32, &plain, &indices, &scores,
            &identity_table(256))?;

        // A permuted page table maps the same logical ids to different slots,
        // so the live plain route must present the mapped slots.
        let mut table = identity_table(256);
        table.swap(0, 5);
        let mapped = |logical: usize| table[logical / PAGE_ROWS] * PAGE_ROWS as i32
            + (logical % PAGE_ROWS) as i32;
        let mut permuted = plain.clone();
        for slot in permuted.iter_mut().take(retained) { *slot = mapped(*slot as usize); }
        verify_selection(3, 4279, frontier as i32, retained as i32, &permuted, &indices, &scores, &table)?;

        let mut off = indices.clone();
        off[0] = 5;
        let error = verify_selection(3, 4279, frontier as i32, retained as i32, &plain, &off, &scores,
            &identity_table(256)).unwrap_err().to_string();
        assert!(error.contains("layer 3 position 4279 symdiff"), "{error}");

        let mut padded = indices.clone();
        padded[0] = -1;
        assert!(verify_selection(3, 4279, frontier as i32, retained as i32, &plain, &padded, &scores,
            &identity_table(256)).is_err());

        let mut far = indices.clone();
        far[0] = 5000;
        assert!(verify_selection(3, 4279, frontier as i32, retained as i32, &plain, &far, &scores,
            &identity_table(256)).is_err());

        let mut repeated = indices.clone();
        repeated[0] = repeated[1];
        let error = verify_selection(3, 4279, frontier as i32, retained as i32, &plain, &repeated, &scores,
            &identity_table(256)).unwrap_err().to_string();
        assert!(error.contains("repeats logical token"), "{error}");

        // The retained count must equal min(frontier, 2048).
        let error = verify_selection(3, 4279, frontier as i32, 2047, &plain, &indices, &scores,
            &identity_table(256)).unwrap_err().to_string();
        assert!(error.contains("retains 2047"), "{error}");
        Ok(())
    }

    #[test]
    fn writer_requires_descending_scores() -> Result<()> {
        let frontier = 8usize;
        let (mut scores, indices) = twin_row(frontier);
        scores.swap(0, 1);
        let plain = plain_from(&indices);
        let error = verify_selection(3, 4279, frontier as i32, frontier as i32, &plain, &indices, &scores,
            &identity_table(256)).unwrap_err().to_string();
        assert!(error.contains("not descending"), "{error}");
        Ok(())
    }

    #[test]
    fn writer_filters_below_the_window_floor_sorts_and_round_trips() -> Result<()> {
        let temp = tempfile::tempdir()?;
        let dir = temp.path().join("nested/scores");
        let dump = ScoreDump::from_env_vars(Some(dir.as_os_str()), Some(OsStr::new("3,7")),
            Some(OsStr::new("4279")))?.expect("set env creates dump");
        assert!(dir.is_dir());

        // Below the floor: verified, but no file or manifest entry appears.
        dump.write_batch(3, batch_from(&[(4276, 2048), (4278, 2048)]))?;
        assert!(!dir.join("manifest.jsonl").exists());
        assert!(!dir.join("layer-03-scores.bin").exists());

        // Mixed and unsorted: positions below the floor are dropped, the rest
        // land in ascending position order; a deep frontier keeps ids past 2048.
        dump.write_batch(3, batch_from(&[(4283, 3000), (4281, 2048), (4278, 2048), (4279, 2044)]))?;
        dump.write_batch(7, batch_from(&[(4280, 2048)]))?;
        let (layer, width, records) = read_layer_file(&dir.join("layer-03-scores.bin"))?;
        assert_eq!((layer, width), (3, WIDTH as u32));
        assert_eq!(records.iter().map(|r| (r.position, r.length)).collect::<Vec<_>>(),
            vec![(4279, 2044), (4281, 2048), (4283, 3000)]);
        let (expected_scores, expected_indices) = twin_row(2044);
        assert_eq!(records[0].scores, expected_scores);
        assert_eq!(records[0].indices, expected_indices);
        assert_eq!(records[1].indices, twin_row(2048).1);
        assert_eq!(records[2].indices, twin_row(3000).1);
        assert_eq!(records[2].indices[0], 2999);

        let manifest = std::fs::read_to_string(dir.join("manifest.jsonl"))?
            .lines().map(serde_json::from_str::<serde_json::Value>)
            .collect::<std::result::Result<Vec<_>, _>>()?;
        assert_eq!(manifest.len(), 2);
        assert_eq!(manifest[0]["schema"], "cuteafd.kimi.dsa-scores/1");
        assert_eq!(manifest[0]["kept_rows"], 3);
        assert_eq!(manifest[0]["logical_indices"], true);
        assert_eq!(manifest[0]["first_position"], 4279);
        assert_eq!(manifest[0]["last_position"], 4283);
        assert_eq!(manifest[1]["layer"], 7);
        Ok(())
    }

    #[test]
    fn writer_rejects_a_mismatch_before_writing_the_layer() -> Result<()> {
        let temp = tempfile::tempdir()?;
        let dump = ScoreDump::from_env_vars(Some(temp.path().join("scores").as_os_str()), None, None)?
            .expect("set env creates dump");
        let mut batch = batch_from(&[(4279, 3000)]);
        for value in batch.twin.chunks_exact_mut(4).take(1) { value.copy_from_slice(&(5i32).to_le_bytes()); }
        let error = dump.write_batch(3, batch).unwrap_err().to_string();
        assert!(error.contains("scored twin selection differs"), "{error}");
        assert!(!temp.path().join("scores/layer-03-scores.bin").exists());
        assert!(!temp.path().join("scores/manifest.jsonl").exists());
        Ok(())
    }
}
