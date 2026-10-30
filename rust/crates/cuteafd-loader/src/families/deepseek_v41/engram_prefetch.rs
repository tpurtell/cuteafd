//! Native paired or row-interleaved engram tables on the shared mapped-table
//! infrastructure (`formats::mapped_table`): mapping, the layer's bounded row
//! cache, row reads through the shared read reactor, and per-table stats.
use crate::formats::mapped_table::row_cache::RowCache;
use crate::formats::mapped_table::{GatherMode, PreparedTableGather, ReadPriority};
use crate::{MappedTable, RowFormat};
use anyhow::{ensure, Context, Result};
use std::sync::Arc;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EngramEncoding { Fp8, Nvfp4 }
impl EngramEncoding {
    pub fn weight_bytes(self) -> usize { match self { Self::Fp8 => 256, Self::Nvfp4 => 128 } }
    pub fn scale_bytes(self) -> usize { match self { Self::Fp8 => 8, Self::Nvfp4 => 16 } }
}

pub struct EngramTable {
    weights: MappedTable,
    scales: Option<MappedTable>,
    encoding: EngramEncoding,
    global_scale: f32,
    /// The layer's bounded row cache: every gather and look-ahead read lands here.
    cache: Arc<RowCache>,
}

/// A small cache for tables built outside the serving pipeline (tests, tools);
/// [`EngramTable::set_cache_shares`] sizes the serving one.
const DEFAULT_SHARES: [usize; 2] = [4096, 1024];

/// Rows of one table read into private buffers through the mapped-table hybrid
/// (resident rows copied now, misses on the shared read reactor).
pub(crate) struct RowRead {
    weight: PreparedTableGather,
    scale: Option<PreparedTableGather>,
    weights: Vec<u8>,
    scales: Vec<u8>,
    widths: [usize; 2],
    /// Rows not resident when the read started (a device read for each).
    misses: usize,
}

impl RowRead {
    /// Block until every row has landed.
    pub(crate) fn finish(&mut self, table: &EngramTable) -> Result<()> {
        let (mut weight_done, mut scale_done) = (false, self.scale.is_none());
        while !weight_done || !scale_done {
            if !weight_done { weight_done = self.weight.step(&table.weights, &mut self.weights)?; }
            if !scale_done {
                let scales = table.scales.as_ref().context("paired engram has no scale table")?;
                scale_done = self.scale.as_mut().expect("paired read").step(scales, &mut self.scales)?;
            }
        }
        Ok(())
    }
    /// Row `index` as the cache stores it: weight bytes, then scale bytes.
    pub(crate) fn row(&self, index: usize) -> [&[u8]; 2] {
        let [w, s] = self.widths;
        [&self.weights[index * w..(index + 1) * w], &self.scales[index * s..(index + 1) * s]]
    }
    pub(crate) fn misses(&self) -> usize { self.misses }
    /// Land whatever has completed without waiting; true once every row is here.
    pub(crate) fn poll(&mut self, table: &EngramTable) -> Result<bool> {
        let weight_done = self.weight.poll(&table.weights, &mut self.weights)?;
        let scale_done = match self.scale.as_mut() {
            None => true,
            Some(scale) => scale.poll(table.scales.as_ref().context("paired engram has no scale table")?, &mut self.scales)?,
        };
        Ok(weight_done && scale_done)
    }
}

impl EngramTable {
    fn assemble(weights: MappedTable, scales: Option<MappedTable>, encoding: EngramEncoding, global_scale: f32)
        -> Result<Self> {
        let width = weights.row_bytes() + scales.as_ref().map_or(0, MappedTable::row_bytes);
        let cache = RowCache::new(width, DEFAULT_SHARES, weights.shared_stats())?;
        Ok(Self { weights, scales, encoding, global_scale, cache })
    }
    /// Replace the row cache with one of `shares` (prefill, decode) rows, before
    /// the table is shared.
    pub fn set_cache_shares(&mut self, shares: [usize; 2]) -> Result<()> {
        self.cache = RowCache::new(self.cache.width(), shares, self.weights.shared_stats())?;
        Ok(())
    }
    pub(crate) fn cache(&self) -> &Arc<RowCache> { &self.cache }
    /// Bytes of one cached row: weight then scale.
    pub fn row_width(&self) -> usize { self.cache.width() }

    /// Start reading `rows` (distinct) into private buffers; resident rows are
    /// copied on this thread, misses go to the shared read reactor.
    pub(crate) fn start_read(&self, rows: &[u64], priority: ReadPriority, mode: GatherMode) -> Result<RowRead> {
        self.start_read_window(rows, priority, 2, mode)
    }
    /// [`Self::start_read`] holding at most `window` reactor buffers per plane
    /// (a paired table reads two planes, each with its own buffer).
    pub(crate) fn start_read_window(&self, rows: &[u64], priority: ReadPriority, window: usize, mode: GatherMode)
        -> Result<RowRead> {
        let slots: Vec<usize> = (0..rows.len()).collect();
        let weight_width = self.weights.row_bytes();
        let mut weights = vec![0; rows.len() * weight_width];
        match &self.scales {
            None => {
                let weight = self.weights.prepare_gather(rows, &slots, &mut weights, priority, window.clamp(1, 2), mode)?;
                let misses = weight.misses();
                Ok(RowRead { weight, scale: None, weights, scales: Vec::new(), widths: [weight_width, 0], misses })
            }
            Some(table) => {
                let mut scales = vec![0; rows.len() * table.row_bytes()];
                // Both tables start before either is waited for; one buffer window each.
                let weight = self.weights.prepare_gather(rows, &slots, &mut weights, priority, 1, mode)?;
                let scale = table.prepare_gather(rows, &slots, &mut scales, priority, 1, mode)?;
                let misses = weight.misses().max(scale.misses());
                Ok(RowRead { weight, scale: Some(scale), weights, scales, widths: [weight_width, table.row_bytes()],
                    misses })
            }
        }
    }
    pub fn encoding(&self) -> EngramEncoding { self.encoding }
    pub fn global_scale(&self) -> f32 { self.global_scale }
    pub fn new_nvfp4(weights: MappedTable, scales: MappedTable, global_scale: f32) -> Result<Self> {
        ensure!(weights.rows() == scales.rows() && weights.row_bytes() == 128 && scales.row_bytes() == 16,
            "NVFP4 engram requires paired 128-byte weight and 16-byte scale rows");
        ensure!(global_scale.is_finite() && global_scale > 0.0, "invalid NVFP4 engram global scale");
        Self::assemble(weights, Some(scales), EngramEncoding::Nvfp4, global_scale)
    }
    pub fn weights(&self) -> &MappedTable {
        &self.weights
    }
    /// Row storage and optional separate scales, before the tables are shared.
    pub(crate) fn tables_mut(&mut self) -> (&mut MappedTable, Option<&mut MappedTable>) {
        (&mut self.weights, self.scales.as_mut())
    }
    pub fn scales(&self) -> Option<&MappedTable> { self.scales.as_ref() }
    pub fn is_strided(&self) -> bool { self.scales.is_none() }
    pub fn new_strided(rows: MappedTable) -> Result<Self> {
        ensure!(rows.row_bytes() == 264 && rows.format().dtype == cuteafd_core::DType::U8,
            "strided engram requires opaque U8 rows of 264 bytes");
        Self::assemble(rows, None, EngramEncoding::Fp8, 1.0)
    }
    /// Gather stats are kept on the weight table (scales share its rows).
    pub fn stats(&self) -> crate::TableStatsSnapshot {
        self.weights.stats().snapshot()
    }

    /// Map the official embedding and per-row scale tensors for one engram layer.
    ///
    /// # Safety
    /// All referenced checkpoint files must remain immutable until this table is dropped.
    pub unsafe fn from_catalog(catalog: &cuteafd_core::TensorCatalog, layer: u32) -> Result<Self> {
        use cuteafd_core::DType;
        use std::path::{Component, Path};
        let rows = match layer {
            1 => 384_006_168_usize,
            14 => 384_016_682_usize,
            _ => anyhow::bail!("V4.1 Flash has no engram table at layer {layer}"),
        };
        let map = |suffix: &str, dtype: DType, width: usize| -> Result<MappedTable> {
            let name = format!("layers.{layer}.engram.embed.{suffix}");
            let mut matching = catalog.tensors.iter().filter(|tensor| tensor.name == name);
            let tensor = matching
                .next()
                .with_context(|| format!("missing engram tensor {name}"))?;
            ensure!(matching.next().is_none(), "duplicate engram tensor {name}");
            ensure!(
                tensor.dtype == dtype && tensor.shape == [rows, width],
                "invalid native engram representation for {name}"
            );
            ensure!(
                tensor.byte_length == rows as u64 * width as u64,
                "invalid engram payload length for {name}"
            );
            ensure!(
                !tensor.file.is_empty()
                    && Path::new(&tensor.file)
                        .components()
                        .all(|part| matches!(part, Component::Normal(_))),
                "invalid engram checkpoint shard path"
            );
            // SAFETY: forwarded from this function's contract.
            let table = unsafe {
                MappedTable::single(
                    &Path::new(&catalog.snapshot_path).join(&tensor.file),
                    tensor.byte_offset,
                    rows as u64,
                    RowFormat::of(dtype, width)?,
                )?
            };
            table.name_stats(format!("engram.{layer}.{suffix}"));
            Ok(table)
        };
        let config_path = std::path::Path::new(&catalog.snapshot_path).join("config.json");
        if config_path.exists() {
            let config = crate::read_official_v41_config(Path::new(&catalog.snapshot_path))?;
            if config.strided_engram() {
                let prefix = format!("layers.{layer}.engram.embed.");
                let declared = config.engram_parts(layer as usize);
                let matching: Vec<_> = catalog.tensors.iter().filter(|t| t.name.starts_with(&prefix)).collect();
                ensure!(matching.len() == declared.len(), "mixed or missing strided engram parts");
                let parts = declared.iter().map(|part| {
                    let tensor = matching.iter().find(|t| t.name == part.tensor)
                        .with_context(|| format!("missing strided part {}", part.tensor))?;
                    ensure!(tensor.dtype == DType::U8 && tensor.shape == [part.rows, 264]
                        && tensor.byte_length == part.rows as u64 * 264, "invalid strided part {}", part.tensor);
                    ensure!(!tensor.file.is_empty() && Path::new(&tensor.file).components()
                        .all(|part| matches!(part, Component::Normal(_))), "invalid strided shard path");
                    Ok(crate::TablePart { path: Path::new(&catalog.snapshot_path).join(&tensor.file),
                        offset: tensor.byte_offset, rows: part.rows as u64 })
                }).collect::<Result<Vec<_>>>()?;
                // SAFETY: forwarded from this function's immutable checkpoint contract.
                let table = unsafe { MappedTable::open(&parts, RowFormat::of(DType::U8, 264)?)? };
                table.name_stats(format!("engram.{layer}.strided"));
                return Self::new_strided(table);
            }
        }
        Self::new(
            map("weight", DType::F8E4M3, 256)?,
            map("scale", DType::F8E8M0, 8)?,
        )
    }
    pub fn new(weights: MappedTable, scales: MappedTable) -> Result<Self> {
        ensure!(
            weights.rows() == scales.rows(),
            "engram weight/scale row counts differ"
        );
        ensure!(
            weights.row_bytes() == 256 && scales.row_bytes() == 8,
            "native engram rows require 256 FP8 bytes and eight UE8M0 scale bytes"
        );
        Self::assemble(weights, Some(scales), EngramEncoding::Fp8, 1.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn official_size_catalog_maps_sparse_shard_and_rejects_eager_loading() -> Result<()> {
        use cuteafd_core::{DType, ModelFacts, TensorCatalog, TensorInfo, TensorRole};
        use std::io::{Seek, SeekFrom};
        let directory = tempfile::tempdir()?;
        let path = directory.path().join("engram.safetensors");
        let mut shard = std::fs::File::create(&path)?;
        let rows = 384_006_168_usize;
        let weight_offset = 13_u64;
        let scale_offset = weight_offset + rows as u64 * 256;
        shard.set_len(scale_offset + rows as u64 * 8)?;
        shard.seek(SeekFrom::Start(scale_offset - 256))?;
        shard.write_all(&[0x38; 256])?;
        shard.seek(SeekFrom::Start(scale_offset + (rows as u64 - 1) * 8))?;
        shard.write_all(&[127; 8])?;
        let mut catalog = TensorCatalog {
            model_id: "deepseek-ai/DeepSeek-V4.1-Flash".into(),
            snapshot_path: directory.path().to_str().unwrap().into(),
            facts: ModelFacts::default(),
            tensors: [
                ("weight", DType::F8E4M3, 256, weight_offset),
                ("scale", DType::F8E8M0, 8, scale_offset),
            ]
            .into_iter()
            .map(|(suffix, dtype, width, offset)| TensorInfo {
                name: format!("layers.1.engram.embed.{suffix}"),
                file: "engram.safetensors".into(),
                dtype,
                shape: vec![rows, width],
                byte_offset: offset,
                byte_length: rows as u64 * width as u64,
                role: TensorRole::Other,
                layer_id: Some(1),
                expert_id: None,
                is_quantization_metadata: suffix == "scale",
            })
            .collect(),
        };
        let table = unsafe { EngramTable::from_catalog(&catalog, 1)? };
        let mut weights = [0; 256];
        let mut scales = [0; 8];
        table
            .weights()
            .gather_into(&[rows as u64 - 1], &mut weights)?;
        table
            .scales().unwrap()
            .gather_into(&[rows as u64 - 1], &mut scales)?;
        assert_eq!(weights, [0x38; 256]);
        assert_eq!(scales, [127; 8]);
        assert!(crate::load_tensor_bytes(&catalog, "layers.1.engram.embed.weight").is_err());
        assert!(
            crate::read_tensor_bytes_into(&catalog, "layers.1.engram.embed.scale", &mut [])
                .is_err()
        );
        assert!(unsafe { EngramTable::from_catalog(&catalog, 2) }.is_err());
        catalog.tensors[1].dtype = DType::U8;
        assert!(unsafe { EngramTable::from_catalog(&catalog, 1) }.is_err());
        Ok(())
    }
}
