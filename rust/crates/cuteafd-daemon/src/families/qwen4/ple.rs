//! The PLE n-gram table (`ple.ple_embedding.ngram_embedding.shard_*`): 128
//! shards of 2,500,012 rows x 160, BF16 (~95 GiB) or E4M3 with one BF16
//! scale (~48 GiB). Token ids hash to rows on the host (`NgramHasher`), 16
//! per token; the `qwen4_ple_*` program gathers `table[ids]`.
//!
//! Shards stay mapped; each step gathers rows into a shared pinned token-table pipeline.
use crate::shared::mapped_table::MappedTableDevice;
use crate::shared::memory::DeviceAllocation;
use anyhow::{ensure, Context, Result};
use cuteafd_core::DType;
use cuteafd_ffi::NativeLibrary;
use cuteafd_loader::plan::checkpoint::Checkpoint;
use cuteafd_loader::families::qwen4::{NgramHasher, Qwen4Config};
use std::time::Instant;

pub(crate) struct PleTable<'a> {
    mapped: MappedTableDevice<'a>,
    /// FP32 [1] table scale on the device (1.0 for BF16 tables).
    pub scale: DeviceAllocation<'a>,
    pub fp8: bool,
    pub rows: usize,
    pub row_bytes: usize,
    pub hasher: NgramHasher,
}

fn i64s(bytes: &[u8]) -> Vec<i64> {
    bytes.chunks_exact(8).map(|b| i64::from_le_bytes(b.try_into().unwrap())).collect()
}

impl<'a> PleTable<'a> {
    pub fn mapped(&self) -> &MappedTableDevice<'a> {
        &self.mapped
    }

    /// Token rows per admitted prefill wave and per decode/verify step.
    pub fn load(library: &'a NativeLibrary, checkpoint: &Checkpoint, cfg: &Qwen4Config, layer: usize,
        prefill_rows: usize, prefill_slots: usize, decode_rows: usize) -> Result<Self> {
        let _memory_scope = cuteafd_ffi::memory_ledger::scope("ple");
        let loader = super::weights::Qwen4Loader { library, checkpoint, fp8_decode: false,
            fp8_scales: crate::shared::fp8_linear::Fp8Scales::Amax, stream: std::ptr::null_mut(),
            dense: std::rc::Rc::new(crate::shared::experts::rtx::dense::DenseLoad::mount(library)) };
        let prefix = format!("{}layers.{layer}.ple.ple_embedding.", super::weights::PREFIX);
        let multipliers = i64s(&loader.raw(&format!("{prefix}layer_multipliers"))?.0);
        let sizes = i64s(&loader.raw(&format!("{prefix}ngram_heads_vocab_sizes"))?.0);
        let offsets = i64s(&loader.raw(&format!("{prefix}ngram_heads_offsets"))?.0);
        let hasher = NgramHasher { multipliers, sizes, offsets, heads_per_ngram: cfg.heads_per_ngram, eos: cfg.eos };
        ensure!(hasher.heads() == cfg.ple_rows() && hasher.ngram_size() == cfg.ngram_size,
            "PLE hash tables do not match the config");
        let ple_index = cfg.ple_layers.iter().position(|&l| l == layer).context("not a PLE layer")?;
        let derived = NgramHasher::from_config(cfg.vocab_size, 20_000_000, cfg.ngram_size, cfg.heads_per_ngram,
            ple_index, 1234, cfg.eos);
        if derived != hasher {
            tracing::warn!("PLE hash tables differ from the config-derived ones; using the checkpoint's");
        }
        // The shards, in row order.
        let mut shards = Vec::new();
        for shard in 0.. {
            let name = format!("{prefix}ngram_embedding.shard_{shard}.weight");
            let Ok(tensor) = loader.tensor(&name) else { break };
            shards.push(tensor);
        }
        ensure!(!shards.is_empty(), "no PLE table shards under {prefix}");
        let dtype = shards[0].meta.dtype.clone();
        let fp8 = match dtype {
            DType::Bf16 => false,
            DType::F8E4M3 => true,
            other => anyhow::bail!("PLE table dtype {other:?} is not BF16 or E4M3"),
        };
        let width = shards[0].meta.shape[1];
        ensure!(width * cfg.ple_rows() == cfg.ple_dim, "PLE rows of {width} do not tile {}", cfg.ple_dim);
        ensure!(shards.iter().all(|s| s.meta.dtype == dtype && s.meta.shape.len() == 2 && s.meta.shape[1] == width),
            "PLE shards disagree on dtype or width");
        let rows: usize = shards.iter().map(|s| s.meta.shape[0]).sum();
        ensure!(rows as i64 >= hasher.rows(), "PLE table has {rows} rows, the hash addresses {}", hasher.rows());
        let row_bytes = width * if fp8 { 1 } else { 2 };
        let total = rows * row_bytes;
        let scale = if fp8 {
            let (bytes, dtype, _) = loader.raw(&format!("{prefix}ngram_embedding.weight_scale"))?;
            match dtype {
                DType::Bf16 => f32::from_bits(u32::from(u16::from_le_bytes([bytes[0], bytes[1]])) << 16),
                DType::F32 => f32::from_le_bytes(bytes[..4].try_into().unwrap()),
                other => anyhow::bail!("PLE table scale dtype {other:?}"),
            }
        } else {
            1.0
        };
        let scale_dev = DeviceAllocation::new(library, 256)?;
        library.copy_h2d(cuteafd_ffi::CuteafdDeviceBuffer { bytes: 4, ..scale_dev.buffer }, &scale.to_le_bytes())?;
        let started = Instant::now();
        let tensors: Vec<_> = shards.iter().map(|s| (s.shard.as_str(), &s.meta)).collect();
        // SAFETY: checkpoint snapshots are immutable while the server runs.
        let mapped = unsafe { cuteafd_loader::MappedTable::from_tensors(&checkpoint.snapshot, &tensors)? };
        ensure!(mapped.row_bytes() == row_bytes && mapped.rows() as usize == rows, "mapped PLE table geometry");
        let mapped = MappedTableDevice::new(library, "ple", mapped, prefill_rows * cfg.ple_rows(),
            prefill_slots, decode_rows * cfg.ple_rows(), cfg.ple_rows())?;
        tracing::info!(gib = total as f64 / (1u64 << 30) as f64, fp8, parts = shards.len(),
            elapsed_ms = started.elapsed().as_millis() as u64, "PLE n-gram table ready");
        cuteafd_bench::context::set_mapped_placement(cuteafd_loader::formats::mapped_preload::MappedPlacement::resolve(
            cfg.ple_layers.len(), 0, 0, cuteafd_loader::formats::mapped_preload::DEFAULT_CAP_BYTES, 0));
        Ok(Self { mapped, scale: scale_dev, fp8, rows, row_bytes, hasher })
    }
}
