//! Opt-in diagnostic readback of the DSA index cache's stored key pages
//! (`CUTEAFD_KIMI_DSA_KEYS`).
//!
//! The producer writes each token's index key into a page of the layer's index
//! cache: `PAGE_ROWS` (64) rows of `DIM` (128) E4M3 bytes, followed by one FP32
//! UE8M0 scale per row, `PAGE_BYTES` (8448) per page. The selector dequantizes
//! `key = code * scale` for every candidate slot while scoring, so the served
//! key contract is exactly the stored pair. No engine owner retains the
//! pre-quantization BF16 key, so this dump reads the stored page the way the
//! selector reads it — for the physical page the live page table maps the
//! requested logical token to — and leaves the arithmetic to the analyzer.
//!
//! Unset `CUTEAFD_KIMI_DSA_KEYS` resolves to `None` at startup with zero stream
//! drains, zero host copies and zero behavior change. When set,
//! `CUTEAFD_KIMI_DSA_KEY_TOKENS` (comma-separated logical token positions) is
//! required, `CUTEAFD_KIMI_DSA_KEY_LAYERS` (default: every DSA layer) restricts
//! the readback, and `CUTEAFD_KIMI_DSA_KEY_FROM` (default `0`) restricts the
//! engine positions whose rows are read. Only non-graph, non-staging arms can
//! dump: the readback drains the producing lane stream.

use crate::shared::{memory::device::Stream, prefix::view};
use anyhow::{ensure, Context, Result};
use cuteafd_ffi::CuteafdDeviceBuffer;
use std::{
    cell::RefCell,
    collections::{BTreeMap, BTreeSet, HashMap},
    ffi::OsStr,
    fs::{File, OpenOptions},
    io::{Seek, SeekFrom, Write},
    path::PathBuf,
    rc::Rc,
};

pub(crate) const ENV_DIR: &str = "CUTEAFD_KIMI_DSA_KEYS";
pub(crate) const ENV_LAYERS: &str = "CUTEAFD_KIMI_DSA_KEY_LAYERS";
pub(crate) const ENV_TOKENS: &str = "CUTEAFD_KIMI_DSA_KEY_TOKENS";
pub(crate) const ENV_FROM: &str = "CUTEAFD_KIMI_DSA_KEY_FROM";
pub(crate) const MAGIC: &[u8; 8] = b"KDSK0001";
pub(crate) const PAGE_MAGIC: &[u8; 8] = b"KDSKP001";
pub(crate) const PAGE_ROWS: usize = 64;
pub(crate) const DIM: usize = 128;
pub(crate) const KEY_BYTES: usize = PAGE_ROWS * DIM;
pub(crate) const SCALE_BYTES: usize = PAGE_ROWS * 4;
pub(crate) const PAGE_BYTES: usize = KEY_BYTES + SCALE_BYTES;
pub(crate) const HEADER_BYTES: usize = 24;
/// `position u64 | token i32 | row i32 | page i32 | page id u32`.
pub(crate) const RECORD_BYTES: usize = 8 + 4 + 4 + 4 + 4;
/// `page i32 | PAGE_BYTES` of stored page contents.
pub(crate) const PAGE_RECORD_BYTES: usize = 4 + PAGE_BYTES;

/// One requested (position, token) pair with the physical page the live page
/// table maps it to, and the batch row whose step wrote that mapping.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct RequestedKey {
    pub position: u64,
    pub token: i32,
    pub row: i32,
    pub page: i32,
}

/// The rows and token set whose stored key pages one DSA step must read.
/// Pure host arithmetic: `positions` is the step's `rows * 8` byte image and
/// `table` the page table the selector read (`rows * table_width` when the
/// step builds a table per row, one row when it shares a table).
#[allow(clippy::too_many_arguments)]
pub(crate) fn plan_readback(
    from: u64,
    tokens: &[i32],
    rows: usize,
    per_row_table: bool,
    table_width: usize,
    cache_pages: usize,
    positions: &[u8],
    table: &[u8],
) -> Result<Vec<RequestedKey>> {
    ensure!(rows > 0, "Kimi DSA key dump requires positive batch rows");
    ensure!(table_width > 0, "Kimi DSA key dump requires a page table width");
    ensure!(positions.len() == rows * 8, "Kimi DSA key positions extent differs from batch rows {rows}");
    let table_rows = if per_row_table { rows } else { 1 };
    ensure!(table.len() == table_rows * table_width * 4,
        "Kimi DSA key page table extent differs from {table_rows} rows of width {table_width}");
    let mut planned = Vec::new();
    for row in 0..rows {
        let position = i64::from_le_bytes(positions[row * 8..(row + 1) * 8].try_into().unwrap());
        ensure!(position >= 0, "Kimi DSA key dump saw negative engine position {position}");
        let position = position as u64;
        if position < from {
            continue;
        }
        let first = if per_row_table { row * table_width * 4 } else { 0 };
        let row_table = &table[first..first + table_width * 4];
        for &token in tokens {
            ensure!(token >= 0, "Kimi DSA key dump requires nonnegative token positions");
            let token = token as u64;
            // A token is a candidate of this row only once it is at or below
            // the row's own position; earlier steps in a decode arm simply do
            // not reach later requested tokens.
            if token > position {
                continue;
            }
            let page_index = usize::try_from(token / PAGE_ROWS as u64)?;
            ensure!(page_index < table_width,
                "Kimi DSA key token {token} is outside the page table width {table_width}");
            let page = i32::from_le_bytes(row_table[page_index * 4..(page_index + 1) * 4].try_into().unwrap());
            ensure!(page >= 0,
                "Kimi DSA key token {token} maps to padding page {page} at position {position}");
            ensure!((page as usize) < cache_pages,
                "Kimi DSA key token {token} maps to page {page} past the {cache_pages}-page index cache");
            planned.push(RequestedKey { position, token: token as i32, row: row as i32, page });
        }
    }
    Ok(planned)
}

fn encode_record(record: &RequestedKey, page_id: u32, out: &mut Vec<u8>) {
    out.extend_from_slice(&record.position.to_le_bytes());
    out.extend_from_slice(&record.token.to_le_bytes());
    out.extend_from_slice(&record.row.to_le_bytes());
    out.extend_from_slice(&record.page.to_le_bytes());
    out.extend_from_slice(&page_id.to_le_bytes());
}

#[derive(Default)]
struct LayerWriter {
    /// One stored pair per (engine position, logical token).
    records: BTreeMap<(u64, i32), (RequestedKey, u32)>,
    pages: Vec<(i32, Vec<u8>)>,
    by_content: HashMap<Vec<u8>, u32>,
    records_file: Option<File>,
    pages_file: Option<File>,
}

#[derive(Default)]
struct State {
    layers: BTreeMap<usize, LayerWriter>,
    manifest: Option<File>,
    manifest_entries: u64,
}

pub(crate) struct KeyDump {
    dir: PathBuf,
    from: u64,
    layers: BTreeSet<usize>,
    tokens: Vec<i32>,
    state: RefCell<State>,
}

impl KeyDump {
    pub(crate) fn from_env() -> Result<Option<Rc<Self>>> {
        Self::from_env_vars(
            std::env::var_os(ENV_DIR).as_deref(),
            std::env::var_os(ENV_LAYERS).as_deref(),
            std::env::var_os(ENV_TOKENS).as_deref(),
            std::env::var_os(ENV_FROM).as_deref(),
        )
    }

    pub(crate) fn from_env_vars(
        dir: Option<&OsStr>,
        layers: Option<&OsStr>,
        tokens: Option<&OsStr>,
        from: Option<&OsStr>,
    ) -> Result<Option<Rc<Self>>> {
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
        let raw_tokens = tokens.with_context(|| format!("{ENV_DIR} requires {ENV_TOKENS}"))?;
        let mut selected_tokens = Vec::new();
        for part in raw_tokens.to_str()
            .with_context(|| format!("{ENV_TOKENS} is not valid UTF-8"))?
            .split(',').filter(|part| !part.is_empty()) {
            let token = part.trim().parse::<i32>()
                .with_context(|| format!("invalid {ENV_TOKENS} entry {part:?}"))?;
            ensure!(token >= 0, "{ENV_TOKENS} token {token} is negative");
            if !selected_tokens.contains(&token) {
                selected_tokens.push(token);
            }
        }
        ensure!(!selected_tokens.is_empty(), "{ENV_DIR} requires at least one {ENV_TOKENS} entry");
        let from = match from {
            None => 0,
            Some(value) => value.to_str().with_context(|| format!("{ENV_FROM} is not valid UTF-8"))?
                .parse::<u64>().with_context(|| format!("invalid {ENV_FROM}"))?,
        };
        Self::new(PathBuf::from(raw), selected, selected_tokens, from).map(Some)
    }

    pub(crate) fn new(dir: PathBuf, layers: BTreeSet<usize>, tokens: Vec<i32>, from: u64) -> Result<Rc<Self>> {
        ensure!(!tokens.is_empty(), "Kimi DSA key dump requires at least one token position");
        std::fs::create_dir_all(&dir)
            .with_context(|| format!("create Kimi DSA key directory {}", dir.display()))?;
        Ok(Rc::new(Self { dir, from, layers, tokens, state: RefCell::new(State::default()) }))
    }

    /// An empty filter selects every DSA layer the family instantiates.
    pub(crate) fn enabled(&self, layer: usize) -> bool { self.layers.is_empty() || self.layers.contains(&layer) }

    pub(crate) fn from(&self) -> u64 { self.from }

    pub(crate) fn tokens(&self) -> &[i32] { &self.tokens }

    fn write_batch(&self, layer: usize, batch: KeyBatch) -> Result<()> {
        ensure!(batch.rows > 0, "Kimi DSA key dump requires positive batch rows");
        let mut kept = Vec::new();
        for record in &batch.records {
            if record.position < self.from {
                continue;
            }
            ensure!(batch.pages.iter().any(|(page, _)| *page == record.page),
                "Kimi DSA key page {} was planned but never read", record.page);
            kept.push(record.clone());
        }
        if kept.is_empty() {
            return Ok(());
        }
        kept.sort_by_key(|record| (record.position, record.token));
        for pair in kept.windows(2) {
            ensure!(pair[0].position != pair[1].position || pair[0].token != pair[1].token,
                "Kimi DSA key dump duplicate position {} token {} in layer {layer}",
                pair[0].position, pair[0].token);
        }
        let kept_rows = kept.len();
        let file_name = format!("layer-{layer:02}-keys.bin");
        let pages_name = format!("layer-{layer:02}-pages.bin");
        let layer_u32 = u32::try_from(layer).context("Kimi DSA key layer index overflow")?;
        let first_position = kept.first().map(|record| record.position);
        let last_position = kept.last().map(|record| record.position);

        let mut state = self.state.borrow_mut();
        let (layer_step, total_rows, total_pages) = {
            let writer = state.layers.entry(layer).or_default();
            if writer.records_file.is_none() {
                let path = self.dir.join(&file_name);
                let file = OpenOptions::new().read(true).write(true).create_new(true)
                    .open(&path).with_context(|| format!("create DSA key layer dump {}", path.display()))?;
                writer.records_file = Some(file);
            }
            if writer.pages_file.is_none() {
                let path = self.dir.join(&pages_name);
                let file = OpenOptions::new().read(true).write(true).create_new(true)
                    .open(&path).with_context(|| format!("create DSA key page dump {}", path.display()))?;
                writer.pages_file = Some(file);
            }
            // Pages are interned by content across the whole arm: a page the
            // producer rewrote is stored again, an unchanged one is shared by
            // every step that reads it.
            let mut final_ids = HashMap::new();
            for (page, image) in batch.pages {
                ensure!(image.len() == PAGE_BYTES,
                    "Kimi DSA key page {page} image is {} bytes, expected {PAGE_BYTES}", image.len());
                let id = match writer.by_content.get(&image) {
                    Some(&id) => id,
                    None => {
                        let id = u32::try_from(writer.pages.len())?;
                        writer.by_content.insert(image.clone(), id);
                        writer.pages.push((page, image));
                        id
                    }
                };
                final_ids.insert(page, id);
            }
            for record in kept {
                let id = *final_ids.get(&record.page).with_context(||
                    format!("Kimi DSA key page {} was planned but never read", record.page))?;
                ensure!(writer.records.insert((record.position, record.token), (record, id)).is_none(),
                    "Kimi DSA key dump duplicate position {} token {} in layer {layer}",
                    record.position, record.token);
            }
            let total = u64::try_from(writer.records.len())?;
            let pages = u64::try_from(writer.pages.len())?;
            let mut image = Vec::with_capacity(HEADER_BYTES + writer.records.len() * RECORD_BYTES);
            image.extend_from_slice(MAGIC);
            image.extend_from_slice(&layer_u32.to_le_bytes());
            image.extend_from_slice(&(DIM as u32).to_le_bytes());
            image.extend_from_slice(&total.to_le_bytes());
            for ((_, _), (record, id)) in &writer.records {
                encode_record(record, *id, &mut image);
            }
            let file = writer.records_file.as_mut().unwrap();
            file.set_len(0)?;
            file.seek(SeekFrom::Start(0))?;
            file.write_all(&image)?;
            file.flush()?;

            let mut pages_image = Vec::with_capacity(HEADER_BYTES + writer.pages.len() * PAGE_RECORD_BYTES);
            pages_image.extend_from_slice(PAGE_MAGIC);
            pages_image.extend_from_slice(&(PAGE_ROWS as u32).to_le_bytes());
            // Reserved slot: the layer, for a reader eyeballing the file. The
            // record file is what identifies the layer.
            pages_image.extend_from_slice(&layer_u32.to_le_bytes());
            pages_image.extend_from_slice(&pages.to_le_bytes());
            for (page, bytes) in &writer.pages {
                pages_image.extend_from_slice(&page.to_le_bytes());
                pages_image.extend_from_slice(bytes);
            }
            let file = writer.pages_file.as_mut().unwrap();
            file.set_len(0)?;
            file.seek(SeekFrom::Start(0))?;
            file.write_all(&pages_image)?;
            file.flush()?;
            let step = state.manifest_entries;
            state.manifest_entries += 1;
            (step, total, pages)
        };

        if state.manifest.is_none() {
            let manifest_path = self.dir.join("manifest.jsonl");
            let file = OpenOptions::new().write(true).create_new(true).open(&manifest_path)
                .with_context(|| format!("create DSA key dump manifest {}", manifest_path.display()))?;
            state.manifest = Some(file);
        }
        let manifest = state.manifest.as_mut().unwrap();
        let entry = serde_json::json!({
            "schema": "cuteafd.kimi.dsa-keys/1",
            "seq": layer_step,
            "layer": layer,
            "file": file_name,
            "pages_file": pages_name,
            "batch_rows": batch.rows,
            "kept_rows": kept_rows,
            "total_rows": total_rows,
            "total_pages": total_pages,
            "first_position": first_position,
            "last_position": last_position,
            "from": self.from,
            "layers": self.layers.iter().copied().collect::<Vec<_>>(),
            "tokens": self.tokens,
            "dim": DIM,
            "page_rows": PAGE_ROWS,
            "page_bytes": PAGE_BYTES,
            "key_bytes": KEY_BYTES,
            "scale_bytes": SCALE_BYTES,
            "per_row_table": batch.per_row_table,
            "table_width": batch.table_width,
            "header_bytes": HEADER_BYTES,
            "record_bytes": RECORD_BYTES,
            "page_record_bytes": PAGE_RECORD_BYTES,
            "logical_tokens": true,
            "key_contract": "E4M3 code * stored UE8M0 FP32 scale, exactly as the selector dequantizes",
        });
        serde_json::to_writer(&mut *manifest, &entry)?;
        manifest.write_all(b"\n")?;
        manifest.flush()?;
        Ok(())
    }
}

/// One producing step's host images: the planned (position, token) records and
/// the unique physical pages they read.
pub(crate) struct KeyBatch {
    pub rows: usize,
    pub per_row_table: bool,
    pub table_width: usize,
    pub records: Vec<RequestedKey>,
    pub pages: Vec<(i32, Vec<u8>)>,
}

/// Buffers one key readback needs: the step's engine positions, the page table
/// the selector read, and the layer's index cache.
pub(crate) struct KeyBuffers {
    pub positions: CuteafdDeviceBuffer,
    pub table: CuteafdDeviceBuffer,
    pub cache: CuteafdDeviceBuffer,
}

/// Drain the producing lane stream, read the stored index pages the live page
/// table maps the requested tokens to, and record them.
#[allow(clippy::too_many_arguments)]
pub(crate) fn record_keys(
    dump: &KeyDump,
    rank: usize,
    stream: &Stream<'_>,
    layer: usize,
    rows: usize,
    table_width: usize,
    per_row_table: bool,
    cache_pages: usize,
    buffers: impl FnOnce() -> Result<KeyBuffers>,
) -> Result<()> {
    if rank != 0 {
        return Ok(());
    }
    ensure!(rows > 0, "Kimi DSA key dump requires positive rows");
    let buffers = buffers()?;
    let pos_view = view(buffers.positions, 0, rows.checked_mul(8).context("DSA key positions byte overflow")?)?;
    let table_rows = if per_row_table { rows } else { 1 };
    let table_view = view(buffers.table, 0,
        table_rows.checked_mul(table_width * 4).context("DSA key page table byte overflow")?)?;
    // Drain the exact producing lane stream before host readback, matching Capture::record.
    stream.drain()?;
    let (positions, table) = stream.device.run(|| {
        let mut positions = vec![0u8; pos_view.bytes];
        let mut table = vec![0u8; table_view.bytes];
        stream.device.library.copy_d2h(&mut positions, pos_view)?;
        stream.device.library.copy_d2h(&mut table, table_view)?;
        Ok((positions, table))
    })?;
    let planned = plan_readback(dump.from(), dump.tokens(), rows, per_row_table, table_width,
        cache_pages, &positions, &table)?;
    if planned.is_empty() {
        return Ok(());
    }
    let mut pages = BTreeMap::new();
    for record in &planned {
        pages.entry(record.page).or_insert(());
    }
    let reads = stream.device.run(|| {
        let mut paged = Vec::with_capacity(pages.len());
        for &page in pages.keys() {
            let bytes = (page as usize).checked_mul(PAGE_BYTES).context("DSA key page offset overflow")?;
            let page_view = view(buffers.cache, bytes, PAGE_BYTES)
                .with_context(|| format!("DSA key page {page} outside the index cache"))?;
            let mut image = vec![0u8; PAGE_BYTES];
            stream.device.library.copy_d2h(&mut image, page_view)?;
            paged.push((page, image));
        }
        Ok(paged)
    })?;
    dump.write_batch(layer, KeyBatch { rows, per_row_table, table_width, records: planned, pages: reads })
}

pub(crate) fn read_layer_file(path: &std::path::Path) -> Result<(u32, u32, Vec<(RequestedKey, u32)>)> {
    let bytes = std::fs::read(path)?;
    ensure!(bytes.len() >= HEADER_BYTES, "DSA key layer dump shorter than header");
    ensure!(&bytes[0..8] == MAGIC, "DSA key layer dump magic mismatch");
    let layer = u32::from_le_bytes(bytes[8..12].try_into().unwrap());
    let dim = u32::from_le_bytes(bytes[12..16].try_into().unwrap());
    let rows = u64::from_le_bytes(bytes[16..24].try_into().unwrap());
    ensure!(dim as usize == DIM, "DSA key layer dump dim {dim} != {DIM}");
    let body = &bytes[HEADER_BYTES..];
    ensure!(body.len() == usize::try_from(rows)? * RECORD_BYTES,
        "DSA key layer dump payload length {} != rows {rows} * {RECORD_BYTES}", body.len());
    let records = body.chunks_exact(RECORD_BYTES).map(|chunk| {
        Ok((
            RequestedKey {
                position: u64::from_le_bytes(chunk[0..8].try_into().unwrap()),
                token: i32::from_le_bytes(chunk[8..12].try_into().unwrap()),
                row: i32::from_le_bytes(chunk[12..16].try_into().unwrap()),
                page: i32::from_le_bytes(chunk[16..20].try_into().unwrap()),
            },
            u32::from_le_bytes(chunk[20..24].try_into().unwrap()),
        ))
    }).collect::<Result<Vec<_>>>()?;
    Ok((layer, dim, records))
}

pub(crate) fn read_pages_file(path: &std::path::Path) -> Result<Vec<(i32, Vec<u8>)>> {
    let bytes = std::fs::read(path)?;
    ensure!(bytes.len() >= HEADER_BYTES, "DSA key page dump shorter than header");
    ensure!(&bytes[0..8] == PAGE_MAGIC, "DSA key page dump magic mismatch");
    let page_rows = u32::from_le_bytes(bytes[8..12].try_into().unwrap());
    let count = u64::from_le_bytes(bytes[16..24].try_into().unwrap());
    ensure!(page_rows as usize == PAGE_ROWS, "DSA key page dump page rows {page_rows} != {PAGE_ROWS}");
    let body = &bytes[HEADER_BYTES..];
    ensure!(body.len() == usize::try_from(count)? * PAGE_RECORD_BYTES,
        "DSA key page dump payload length {} != pages {count} * {PAGE_RECORD_BYTES}", body.len());
    Ok(body.chunks_exact(PAGE_RECORD_BYTES).map(|chunk| {
        (i32::from_le_bytes(chunk[0..4].try_into().unwrap()), chunk[4..].to_vec())
    }).collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn positions_image(positions: &[i64]) -> Vec<u8> {
        positions.iter().flat_map(|value| value.to_le_bytes()).collect()
    }

    fn identity_table(width: usize) -> Vec<u8> {
        (0..width as i32).flat_map(|value| value.to_le_bytes()).collect()
    }

    fn page_image(seed: u8) -> Vec<u8> {
        let mut image = vec![seed; PAGE_BYTES];
        for (row, scale) in image[KEY_BYTES..].chunks_exact_mut(4).enumerate() {
            scale.copy_from_slice(&(0.5f32 * (row as f32 + 1.0)).to_le_bytes());
        }
        image
    }

    #[test]
    fn unset_env_resolves_none_and_requires_a_token_list() -> Result<()> {
        assert!(KeyDump::from_env_vars(None, None, None, None)?.is_none());
        let temp = tempfile::tempdir()?;
        let dir = temp.path().join("keys");
        assert!(KeyDump::from_env_vars(Some(dir.as_os_str()), None, None, None).is_err());
        assert!(KeyDump::from_env_vars(Some(dir.as_os_str()), None, Some(OsStr::new("")), None).is_err());
        assert!(KeyDump::from_env_vars(Some(dir.as_os_str()), None, Some(OsStr::new("-3")), None).is_err());
        let dump = KeyDump::from_env_vars(Some(dir.as_os_str()), Some(OsStr::new("3,59")),
            Some(OsStr::new("2323,4721,2323")), Some(OsStr::new("4279")))?
            .expect("set env creates dump");
        assert_eq!(dump.tokens(), &[2323, 4721]);
        assert_eq!(dump.from(), 4279);
        assert!(dump.enabled(3) && dump.enabled(59) && !dump.enabled(7));
        Ok(())
    }

    #[test]
    fn readback_maps_tokens_through_the_live_page_table_and_skips_future_tokens() -> Result<()> {
        let width = 64;
        let mut table = identity_table(width);
        // Logical page 36 (token 2323 lives there) maps to physical page 1.
        table[36 * 4..37 * 4].copy_from_slice(&1i32.to_le_bytes());
        let planned = plan_readback(4279, &[2323, 300], 2, false, width, 64,
            &positions_image(&[4279, 4300]), &table)?;
        assert_eq!(planned, vec![
            RequestedKey { position: 4279, token: 2323, row: 0, page: 1 },
            RequestedKey { position: 4279, token: 300, row: 0, page: 4 },
            RequestedKey { position: 4300, token: 2323, row: 1, page: 1 },
            RequestedKey { position: 4300, token: 300, row: 1, page: 4 },
        ]);

        // Below the floor is dropped, not an error.
        let planned = plan_readback(4280, &[2050], 1, false, width, 64,
            &positions_image(&[4279]), &identity_table(width))?;
        assert!(planned.is_empty());
        // A token past the row's own position is not yet a candidate.
        let planned = plan_readback(0, &[300], 1, false, width, 64,
            &positions_image(&[199]), &identity_table(width))?;
        assert!(planned.is_empty());

        // Padding pages are hard errors: a live candidate always maps to a
        // real page, and a page past the cache is a table defect.
        let mut padding = identity_table(width);
        padding[36 * 4..37 * 4].copy_from_slice(&(-1i32).to_le_bytes());
        assert!(plan_readback(0, &[2323], 1, false, width, 64, &positions_image(&[4279]), &padding).is_err());
        assert!(plan_readback(0, &[2323], 1, false, width, 32, &positions_image(&[4279]),
            &identity_table(width)).is_err());
        // A token outside the table width errors rather than reading garbage.
        assert!(plan_readback(0, &[300], 1, false, 4, 64, &positions_image(&[4279]),
            &identity_table(4)).is_err());
        Ok(())
    }

    #[test]
    fn writer_interns_pages_filters_and_round_trips() -> Result<()> {
        let temp = tempfile::tempdir()?;
        let dir = temp.path().join("keys");
        let dump = KeyDump::from_env_vars(Some(dir.as_os_str()), Some(OsStr::new("3")),
            Some(OsStr::new("2323,4721")), Some(OsStr::new("4279")))?.expect("set env creates dump");
        let page_a = page_image(7);
        let page_b = page_image(9);
        let records = vec![
            RequestedKey { position: 4281, token: 4721, row: 2, page: 2 },
            RequestedKey { position: 4279, token: 2323, row: 0, page: 1 },
            RequestedKey { position: 4276, token: 2323, row: 0, page: 1 },
            RequestedKey { position: 4281, token: 2323, row: 2, page: 1 },
        ];
        dump.write_batch(3, KeyBatch { rows: 3, per_row_table: false, table_width: 64,
            records, pages: vec![(1, page_a.clone()), (2, page_b.clone())] })?;
        // A second step re-reads page 1 unchanged: it is not stored twice.
        dump.write_batch(3, KeyBatch { rows: 1, per_row_table: false, table_width: 64,
            records: vec![RequestedKey { position: 4282, token: 2323, row: 0, page: 1 }],
            pages: vec![(1, page_a.clone())] })?;

        let (layer, dim, records) = read_layer_file(&dir.join("layer-03-keys.bin"))?;
        assert_eq!((layer, dim), (3, DIM as u32));
        assert_eq!(records, vec![
            (RequestedKey { position: 4279, token: 2323, row: 0, page: 1 }, 0),
            (RequestedKey { position: 4281, token: 2323, row: 2, page: 1 }, 0),
            (RequestedKey { position: 4281, token: 4721, row: 2, page: 2 }, 1),
            (RequestedKey { position: 4282, token: 2323, row: 0, page: 1 }, 0),
        ]);
        let pages = read_pages_file(&dir.join("layer-03-pages.bin"))?;
        assert_eq!(pages, vec![(1, page_a), (2, page_b)]);

        let manifest = std::fs::read_to_string(dir.join("manifest.jsonl"))?
            .lines().map(serde_json::from_str::<serde_json::Value>)
            .collect::<std::result::Result<Vec<_>, _>>()?;
        assert_eq!(manifest.len(), 2);
        assert_eq!(manifest[0]["schema"], "cuteafd.kimi.dsa-keys/1");
        assert_eq!(manifest[0]["kept_rows"], 3);
        assert_eq!(manifest[1]["kept_rows"], 1);
        assert_eq!(manifest[1]["total_rows"], 4);
        assert_eq!(manifest[1]["total_pages"], 2);
        assert_eq!(manifest[1]["tokens"], serde_json::json!([2323, 4721]));
        Ok(())
    }

    #[test]
    fn writer_rejects_a_duplicate_position_and_token() -> Result<()> {
        let temp = tempfile::tempdir()?;
        let dump = KeyDump::from_env_vars(Some(temp.path().join("keys").as_os_str()), None,
            Some(OsStr::new("2323")), None)?.expect("set env creates dump");
        let record = RequestedKey { position: 4279, token: 2323, row: 0, page: 1 };
        let batch = |records| KeyBatch { rows: 1, per_row_table: true, table_width: 64,
            records, pages: vec![(1, page_image(1))] };
        dump.write_batch(3, batch(vec![record]))?;
        let error = dump.write_batch(3, batch(vec![record])).unwrap_err().to_string();
        assert!(error.contains("duplicate position 4279 token 2323"), "{error}");
        Ok(())
    }

    #[test]
    fn readback_plan_rejects_a_short_position_image() {
        let error = plan_readback(0, &[1], 2, false, 4, 8, &positions_image(&[0]), &identity_table(4))
            .unwrap_err().to_string();
        assert!(error.contains("positions extent differs"), "{error}");
    }
}
