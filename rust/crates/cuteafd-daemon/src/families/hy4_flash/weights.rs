//! Header-admitted packed target weight arena and host-resident FP8 Engram tables
//! with earliest-possible look-ahead and background pinned staging for Hy4-flash.
use super::native::{dequant_kv_b_trellis_absorbed, Hy4NativeCore};
use crate::shared::mapped_table::wait_rows;
use crate::shared::memory::{
    budget::admit_host_memory,
    device::{Allocation, Device},
    weight_arena::DeviceWeightArena,
    Tag,
};
use crate::shared::peer_split::{slice_2d, Axis};
use crate::shared::target_head::TargetHead;
use crate::shared::token_io::{EmbedPlacement, EmbedSource, TokenEmbedding};
use anyhow::{bail, ensure, Context, Result};
use cuteafd_core::memory_layout::ArenaTag;
use cuteafd_core::DType;
use cuteafd_ffi::programs::{Programs, VocabularyHead};
use cuteafd_ffi::{CuteafdDeviceBuffer, NativeLibrary, PinnedBytes};
use cuteafd_loader::families::hy4_flash::engram::{
    read_lookup_table, Hy4EngramHasher, Hy4EngramHistory, Hy4EngramLookahead, Hy4EngramShape,
    Hy4EngramTable, ENGRAM_DECODED_TOKEN_BYTES, ENGRAM_HEADS,
};
use cuteafd_loader::families::hy4_flash::weight_layout::{core_split, target_arenas, CoordinatorWeights};
use cuteafd_loader::families::hy4_flash::Hy4FlashConfig;
use cuteafd_loader::formats::mapped_preload::{MappedPlacement, DEFAULT_CAP_BYTES};
use cuteafd_loader::formats::mapped_table::lookahead::{LookaheadHandle, UnitPlan};
use cuteafd_loader::formats::mapped_table::ReadPriority;
use cuteafd_loader::formats::trellis_core::CoreSplit;
use cuteafd_loader::placement::families::hy4_flash as hy4_placement;
use cuteafd_loader::placement::AttentionPlacement;
use cuteafd_loader::plan::checkpoint::Checkpoint;
use cuteafd_loader::plan::families::hy4_flash as hy4_plan;
use cuteafd_loader::shared::token_table::{AdmissionTimedOut, TablePipeline};
use cuteafd_loader::{GatherLease, GatherTicket, TableStatsSnapshot};
use std::cell::Cell;
use std::collections::BTreeMap;
use std::fs::File;
use std::ops::Range;
use std::os::unix::fs::FileExt;
use std::sync::Arc;
use std::time::{Duration, Instant};

const HEAD_WS_BYTES: usize = 32 << 20;
const ENGRAM_GATHER_WAIT: Duration = Duration::from_secs(120);

pub(crate) struct Hy4EngramSlot {
    pub(crate) bytes: PinnedBytes,
}

#[derive(Clone)]
pub(crate) struct Hy4EngramJob {
    layer_idx: usize,
    row_ids: Vec<u64>,
    masked: Vec<u8>,
    priority: ReadPriority,
}

type EngramPipeline = TablePipeline<Hy4EngramSlot, Hy4EngramJob>;
type EngramTicket = GatherTicket<Hy4EngramSlot, Hy4EngramJob, anyhow::Error>;
pub(crate) type EngramLease = GatherLease<Hy4EngramSlot, Hy4EngramJob>;

enum LayerGather {
    Pending(Hy4EngramJob),
    Queued(EngramTicket),
    Consumed,
}

fn engram_gather_timeout(cause: anyhow::Error) -> anyhow::Error {
    anyhow::Error::new(crate::shared::experts::rtx::pair::RecoverableStepError::new(
        crate::shared::experts::rtx::pair::Recovery::HostGather,
        format!("hy4_engram token-table: {cause:#}"),
    ))
}

/// Request-owned early Engram I/O wave spanning all Engram layers in one unit.
/// Background gather + FP8->BF16 decode runs overlapped with GPU layers; dropping
/// the wave cancels any unconsumed tickets and returns their pinned staging slots.
pub(crate) struct Hy4EngramWave {
    tables: Vec<Arc<Hy4EngramTable>>,
    pipeline: Arc<EngramPipeline>,
    layers: BTreeMap<usize, (usize, LayerGather)>,
    rows: usize,
    decode: bool,
}

impl Hy4EngramWave {
    fn admit_pending(&mut self) -> Result<()> {
        self.pipeline.reclaim_copies()?;
        for (layer_idx, state) in self.layers.values_mut() {
            if let LayerGather::Pending(job) = state {
                if let Some(ticket) = self.pipeline.try_submit(job.clone(), *layer_idx == 0)? {
                    *state = LayerGather::Queued(ticket);
                }
            }
        }
        Ok(())
    }

    /// Wait for `layer`'s decoded BF16 + mask pinned staging buffer right before launching
    /// that layer's GPU Engram kernels. Returns `(lease, raw_bytes, masked_bytes)`.
    pub(crate) fn take_layer(&mut self, layer: usize) -> Result<(EngramLease, usize, usize)> {
        let (layer_idx, state) = self
            .layers
            .get_mut(&layer)
            .with_context(|| format!("Hy4 Engram wave has no entry for layer {layer}"))?;
        let layer_idx = *layer_idx;
        let current = std::mem::replace(state, LayerGather::Consumed);
        let started = Instant::now();
        let mut ticket = match current {
            LayerGather::Queued(ticket) => ticket,
            LayerGather::Pending(job) => {
                self.pipeline.reclaim_copies()?;
                match self.pipeline.submit_timeout(job, layer_idx == 0, ENGRAM_GATHER_WAIT) {
                    Ok(ticket) => ticket,
                    Err(error) if error.is::<AdmissionTimedOut>() => return Err(engram_gather_timeout(error)),
                    Err(error) => return Err(error),
                }
            }
            LayerGather::Consumed => bail!("Hy4 Engram layer {layer} already consumed"),
        };
        let lease = wait_rows("hy4_engram", &mut ticket, ENGRAM_GATHER_WAIT)?;
        let elapsed = started.elapsed();
        let stats = self.tables[layer_idx].mapped().stats();
        stats.record_stall(elapsed);
        if self.decode {
            stats.record_decode_stall(elapsed);
        }
        if elapsed >= Duration::from_micros(100) {
            stats.record_consumer_stall(self.decode, elapsed);
        }
        let _ = self.admit_pending();
        Ok((lease, self.rows * ENGRAM_DECODED_TOKEN_BYTES, self.rows))
    }

    pub(crate) fn retain_copy(&self, lease: EngramLease) {
        self.pipeline.retain_copy(lease);
    }
}

/// Host-resident FP8 Engram state across all 4 Hy4-flash Engram layers (`[1, 11, 21, 31]`):
/// shared `TokenTable` row caches, multi-layer `Hy4EngramLookahead`, and pinned prefill/decode
/// background gather pipelines.
pub(crate) struct Hy4EngramHost {
    shape: Hy4EngramShape,
    layer_ids: Vec<usize>,
    hashers: Vec<Arc<Hy4EngramHasher>>,
    tables: Vec<Arc<Hy4EngramTable>>,
    lookahead: Hy4EngramLookahead,
    prefill: Arc<EngramPipeline>,
    decode: Arc<EngramPipeline>,
    logged: Vec<Cell<TableStatsSnapshot>>,
}

impl Hy4EngramHost {
    /// Open all 4 host-mapped FP8 Engram tables (`engram-fp8-layer-*-part-*.safetensors`),
    /// admit host row-cache and pinned staging memory, and start the shared look-ahead
    /// and background gather pipelines.
    ///
    /// # Safety
    /// Checkpoint shard files must remain immutable for the lifetime of this struct.
    pub(crate) unsafe fn open(
        library: &NativeLibrary,
        checkpoint: &Checkpoint,
        cfg: &Hy4FlashConfig,
        max_rows: usize,
        prefill_waves: usize,
        decode_rows: usize,
    ) -> Result<Self> {
        let shape = Hy4EngramShape::new(
            max_rows,
            prefill_waves.max(1),
            decode_rows,
            cfg.engram_layers.len(),
        )?;
        let _scope = cuteafd_ffi::memory_ledger::scope("hy4-engram");
        admit_host_memory(
            0,
            &[
                ("hy4_flash/engram-staging", shape.staging_bytes() as u64),
                ("hy4_flash/engram-row-cache", shape.cache_bytes() as u64),
            ],
            0,
            0,
        )?;

        let shares = shape.cache_shares();
        let mut layer_ids = Vec::with_capacity(cfg.engram_layers.len());
        let mut hashers = Vec::with_capacity(cfg.engram_layers.len());
        let mut tables = Vec::with_capacity(cfg.engram_layers.len());
        let mut token_tables = Vec::with_capacity(cfg.engram_layers.len());
        let mut logged = Vec::with_capacity(cfg.engram_layers.len());

        for &layer in &cfg.engram_layers {
            let lookup = read_lookup_table(checkpoint, layer, cfg.engram_lookup_table_size)?;
            let hasher = Arc::new(Hy4EngramHasher::new(cfg, layer, lookup)?);
            // SAFETY: forwarded from caller.
            let table = Arc::new(unsafe { Hy4EngramTable::open(checkpoint, cfg, layer, shares)? });
            tracing::info!(
                layer,
                rows = table.rows(),
                row_bytes = table.row_bytes(),
                prefill_cache_rows = shares[0],
                decode_cache_rows = shares[1],
                "Hy4-flash FP8 Engram host table ready"
            );
            logged.push(Cell::new(table.mapped().stats().snapshot()));
            token_tables.push(Arc::clone(table.token_table()));
            layer_ids.push(layer);
            hashers.push(hasher);
            tables.push(table);
        }

        let lookahead = Hy4EngramLookahead::new(token_tables, hashers.clone())?;

        let make_pipeline = |name: &str, slots: usize, slot_bytes: usize| -> Result<Arc<EngramPipeline>> {
            let worker_tables = tables.clone();
            let storage = (0..slots)
                .map(|_| {
                    Ok(Hy4EngramSlot {
                        bytes: library.pinned_bytes(slot_bytes, Tag::staging("hy4_flash/engram-staging"))?,
                    })
                })
                .collect::<Result<Vec<_>>>()?;
            Ok(Arc::new(TablePipeline::new(
                name,
                storage,
                move |slot: &mut Hy4EngramSlot, job: &Hy4EngramJob| {
                    let rows = job.masked.len();
                    let raw_bytes = rows * ENGRAM_DECODED_TOKEN_BYTES;
                    let total = raw_bytes + rows;
                    let dst = slot.bytes.bytes_mut();
                    ensure!(total <= dst.len(), "Hy4 Engram staging slot overflow ({total} > {})", dst.len());
                    worker_tables[job.layer_idx].gather_decoded_bf16_into(
                        &job.row_ids,
                        job.priority,
                        job.layer_idx == 0,
                        &mut dst[..raw_bytes],
                    )?;
                    dst[raw_bytes..total].copy_from_slice(&job.masked);
                    Ok(())
                },
                |slot| Ok(slot.bytes.copy_ready()?),
            )?))
        };

        let prefill = make_pipeline("hy4-engram-prefill", shape.prefill_slots(), shape.prefill_slot_bytes())?;
        let decode = make_pipeline("hy4-engram-decode", shape.decode_slots(), shape.decode_slot_bytes())?;

        cuteafd_bench::context::set_mapped_placement(MappedPlacement::resolve(
            cfg.engram_layers.len(),
            0,
            0,
            DEFAULT_CAP_BYTES,
            0,
        ));

        Ok(Self {
            shape,
            layer_ids,
            hashers,
            tables,
            lookahead,
            prefill,
            decode,
            logged,
        })
    }

    /// Start earliest-possible prompt look-ahead across all 4 Engram layers from admission.
    pub(crate) fn start_lookahead(
        &self,
        tag: u64,
        start: u64,
        initial_tail: [u32; 2],
        tokens: Vec<u32>,
        units: UnitPlan,
    ) -> LookaheadHandle {
        self.lookahead.start(tag, start, initial_tail, tokens, units)
    }

    /// Non-blocking decode/verify prefetch across all 4 Engram layers as soon as candidate tokens exist.
    pub(crate) fn prefetch_decode(&self, members: &[([u32; 2], &[u32])]) -> Result<()> {
        self.lookahead.decode(members)
    }

    /// Hash all Engram layers in `layer_span` from forked rolling histories (never mutating
    /// committed tails) and submit background gather + FP8->BF16 decode into pinned staging
    /// immediately so execution overlaps with GPU layers.
    pub(crate) fn begin_wave(
        &self,
        layer_span: &Range<usize>,
        members: &[([u32; 2], &[u32])],
        priority: ReadPriority,
    ) -> Result<Hy4EngramWave> {
        let total_rows: usize = members.iter().map(|(_, t)| t.len()).sum();
        let decode = priority == ReadPriority::Decode && total_rows <= self.shape.decode_rows;
        let pipeline = Arc::clone(if decode { &self.decode } else { &self.prefill });
        pipeline.reclaim_copies()?;

        let mut masked = Vec::with_capacity(total_rows);
        if let Some(first_hasher) = self.hashers.first() {
            for &(_, tokens) in members {
                masked.extend(tokens.iter().map(|&t| u8::from(first_hasher.is_masked(t))));
            }
        }

        let mut layers = BTreeMap::new();
        for (layer_idx, &layer) in self.layer_ids.iter().enumerate() {
            if !layer_span.contains(&layer) {
                continue;
            }
            let hasher = &self.hashers[layer_idx];
            let mut row_ids = Vec::with_capacity(total_rows * ENGRAM_HEADS);
            for &(tail, tokens) in members {
                let mut hist = Hy4EngramHistory(tail.to_vec());
                hasher.hash(&mut hist, tokens, &mut row_ids)?;
            }
            let job = Hy4EngramJob {
                layer_idx,
                row_ids,
                masked: masked.clone(),
                priority,
            };
            let state = match pipeline.try_submit(job.clone(), layer_idx == 0)? {
                Some(ticket) => LayerGather::Queued(ticket),
                None => LayerGather::Pending(job),
            };
            layers.insert(layer, (layer_idx, state));
        }

        Ok(Hy4EngramWave {
            tables: self.tables.clone(),
            pipeline,
            layers,
            rows: total_rows,
            decode,
        })
    }

    /// Reclaim completed H2D copies across both staging pipelines after stream drain.
    pub(crate) fn drain(&self) -> Result<()> {
        self.prefill.reclaim_copies()?;
        self.decode.reclaim_copies()
    }

    /// Log interval Engram row-cache, look-ahead, and blocking-wait telemetry per layer.
    pub(crate) fn log_stats(&self, event: &str) {
        for (idx, table) in self.tables.iter().enumerate() {
            let now = table.mapped().stats().snapshot();
            let d = now.since(&self.logged[idx].replace(now));
            if d.cache_hit_rows
                + d.distinct_rows
                + d.cache_wait_rows
                + d.lookahead_rows
                + d.stall_prefill_fallback_rows
                + d.stall_decode_fallback_rows
                == 0
            {
                continue;
            }
            tracing::info!(
                table = %format!("hy4_engram.{}.fp8_strided", self.layer_ids[idx]),
                layer = self.layer_ids[idx],
                event,
                cache_hit_rows = d.cache_hit_rows,
                cache_wait_rows = d.cache_wait_rows,
                distinct_rows = d.distinct_rows,
                demand_rows = d.demand_rows,
                demand_hit_rows = d.demand_hit_rows,
                lookahead_rows = d.lookahead_rows,
                prefill_fallback_rows = d.stall_prefill_fallback_rows,
                prefill_pending_rows = d.stall_prefill_pending_rows,
                decode_fallback_rows = d.stall_decode_fallback_rows,
                decode_pending_rows = d.stall_decode_pending_rows,
                device_rows = d.page_misses,
                device_ms = d.miss_ns as f64 / 1e6,
                consumer_steps = d.consumer_steps,
                decode_steps = d.decode_steps,
                stall_ms = d.stall_ns as f64 / 1e6,
                decode_stall_ms = d.decode_stall_ns as f64 / 1e6,
                stall_prefill_events = d.stall_prefill_events,
                stall_prefill_ms = d.stall_prefill_ns as f64 / 1e6,
                stall_decode_events = d.stall_decode_events,
                stall_decode_ms = d.stall_decode_ns as f64 / 1e6,
                "mapped table"
            );
        }
    }
}

/// One coordinator rank's packed target weights, RoPE tables, embedding and vocabulary head.
pub(crate) struct Hy4RankWeights<'a> {
    pub(crate) device: Device<'a>,
    pub(crate) rank: usize,
    pub(crate) gpus: usize,
    pub(crate) layout: CoordinatorWeights,
    /// Single packed backing for all target weights & globals on this GPU.
    pub(crate) arena: DeviceWeightArena<'a>,
    /// Named BF16/F32 views inside `arena` (zero separate per-tensor `cudaMalloc`).
    pub(crate) views: BTreeMap<String, Allocation<'a>>,
    /// Compact-core native linears (`k8-compact`, `k6-compact`, `fp8-compact`) backed by `arena`.
    pub(crate) native: Hy4NativeCore<'a>,
    pub(crate) embedding: Option<TokenEmbedding<'a>>,
    pub(crate) inv_freq_dsa: Allocation<'a>,
    pub(crate) inv_freq_swa: Allocation<'a>,
    pub(crate) head_ws: Option<Allocation<'a>>,
    pub(crate) head: Option<VocabularyHead<'a>>,
    pub(crate) local_vocab: usize,
}

fn read_tensor_bytes(checkpoint: &Checkpoint, name: &str) -> Result<(Vec<u8>, DType, Vec<usize>)> {
    let entry = checkpoint
        .tensors
        .iter()
        .find(|t| t.meta.name == name)
        .with_context(|| format!("Hy4-flash checkpoint missing tensor {name}"))?;
    let file = File::open(checkpoint.snapshot.join(&entry.shard))
        .with_context(|| format!("opening shard {} for {name}", entry.shard))?;
    let mut bytes = vec![0u8; usize::try_from(entry.meta.byte_length)?];
    file.read_exact_at(&mut bytes, entry.meta.byte_offset)?;
    Ok((bytes, entry.meta.dtype.clone(), entry.meta.shape.clone()))
}

fn read_packed_engram(checkpoint: &Checkpoint, cfg: &Hy4FlashConfig, name: &str) -> Result<Option<Vec<u8>>> {
    let pack = |prefix: &str, count: usize, suffix: &str| -> Result<Vec<u8>> {
        let mut out = Vec::new();
        for j in 0..count {
            let tname = format!("{prefix}.{j}{suffix}");
            let (raw, dtype, _) = read_tensor_bytes(checkpoint, &tname)?;
            ensure!(dtype == DType::Bf16, "Hy4-flash Engram tensor {tname} expected BF16, got {dtype:?}");
            out.extend_from_slice(&raw);
        }
        Ok(out)
    };
    if let Some(p) = name.strip_suffix(".lookup_per_head_norms.packed") {
        return Ok(Some(pack(&format!("{p}.lookup_per_head_norms"), cfg.engram_total_heads(), ".weight")?));
    }
    if let Some(p) = name.strip_suffix(".key_projs.packed.weight") {
        return Ok(Some(pack(&format!("{p}.key_projs"), cfg.hc_mult, ".weight")?));
    }
    if let Some(p) = name.strip_suffix(".key_projs.packed.bias") {
        return Ok(Some(pack(&format!("{p}.key_projs"), cfg.hc_mult, ".bias")?));
    }
    if let Some(p) = name.strip_suffix(".norm_key.packed") {
        return Ok(Some(pack(&format!("{p}.norm_key"), cfg.hc_mult, ".weight")?));
    }
    if let Some(p) = name.strip_suffix(".norm_query.packed") {
        return Ok(Some(pack(&format!("{p}.norm_query"), cfg.hc_mult, ".weight")?));
    }
    if let Some(p) = name.strip_suffix(".unified_per_hc_scales.packed") {
        return Ok(Some(pack(&format!("{p}.unified_per_hc_scales"), cfg.hc_mult, "")?));
    }
    Ok(None)
}

fn pack_kv_b_absorbed(
    raw: &[u8],
    local_heads: usize,
    nope: usize,
    v_dim: usize,
    kv_lora: usize,
) -> Result<Vec<u8>> {
    let head_rows = nope + v_dim;
    ensure!(
        raw.len() == local_heads * head_rows * kv_lora * 2,
        "kv_b_proj raw length {} != expected {}",
        raw.len(),
        local_heads * head_rows * kv_lora * 2
    );
    let uk_bytes = local_heads * kv_lora * nope * 2;
    let uv_bytes = local_heads * v_dim * kv_lora * 2;
    let mut out = vec![0u8; uk_bytes + uv_bytes];
    // Transform [local_heads, nope + v_dim, kv_lora] ->
    //   w_uk [local_heads, kv_lora, nope] followed by w_uv [local_heads, v_dim, kv_lora].
    for h in 0..local_heads {
        let src_h = h * head_rows * kv_lora * 2;
        // w_uk: transpose [nope, kv_lora] -> [kv_lora, nope]
        let dst_uk = h * kv_lora * nope * 2;
        for d in 0..nope {
            for c in 0..kv_lora {
                let s = src_h + (d * kv_lora + c) * 2;
                let t = dst_uk + (c * nope + d) * 2;
                out[t] = raw[s];
                out[t + 1] = raw[s + 1];
            }
        }
        // w_uv: copy [v_dim, kv_lora] verbatim
        let src_v = src_h + nope * kv_lora * 2;
        let dst_uv = uk_bytes + h * v_dim * kv_lora * 2;
        let v_span = v_dim * kv_lora * 2;
        out[dst_uv..dst_uv + v_span].copy_from_slice(&raw[src_v..src_v + v_span]);
    }
    Ok(out)
}

/// Pack column-major `W^T [kv_lora, local_heads * (nope + v_dim)]` BF16 (as returned by
/// `Projection::local(I_kv_lora)` on EXL3 `kv_b_proj`) into `[w_uk, w_uv]` identical to `pack_kv_b_absorbed`.
pub(crate) fn pack_kv_b_absorbed_from_transposed(
    transposed: &[u8],
    local_heads: usize,
    nope: usize,
    v_dim: usize,
    kv_lora: usize,
) -> Result<Vec<u8>> {
    let head_rows = nope + v_dim;
    let total_cols = local_heads * head_rows;
    ensure!(
        transposed.len() == kv_lora * total_cols * 2,
        "kv_b_proj transposed length {} != expected {}",
        transposed.len(),
        kv_lora * total_cols * 2
    );
    let uk_bytes = local_heads * kv_lora * nope * 2;
    let uv_bytes = local_heads * v_dim * kv_lora * 2;
    let mut out = vec![0u8; uk_bytes + uv_bytes];
    for h in 0..local_heads {
        let h_col0 = h * head_rows;
        // w_uk [h, c, d] = transposed[c, h_col0 + d] (contiguous in d).
        for c in 0..kv_lora {
            let src = (c * total_cols + h_col0) * 2;
            let dst = (h * kv_lora + c) * nope * 2;
            out[dst..dst + nope * 2].copy_from_slice(&transposed[src..src + nope * 2]);
        }
        // w_uv [h, v, c] = transposed[c, h_col0 + nope + v].
        let dst_uv_h = uk_bytes + h * v_dim * kv_lora * 2;
        for c in 0..kv_lora {
            let src_base = (c * total_cols + h_col0 + nope) * 2;
            for v in 0..v_dim {
                let s = src_base + v * 2;
                let t = dst_uv_h + (v * kv_lora + c) * 2;
                out[t] = transposed[s];
                out[t + 1] = transposed[s + 1];
            }
        }
    }
    Ok(out)
}

fn make_inv_freq(device: Device<'_>, rope_dim: usize, theta: f64) -> Result<Allocation<'_>> {
    let half = rope_dim / 2;
    let mut bytes = Vec::with_capacity(half * 4);
    for i in 0..half {
        let freq = (1.0f64 / theta.powf((2 * i) as f64 / rope_dim as f64)) as f32;
        bytes.extend_from_slice(&freq.to_le_bytes());
    }
    let alloc = Allocation::new(device, bytes.len(), Tag::weights("hy4_flash/rope-inv-freq"))?;
    device.run(|| device.library.copy_h2d(alloc.buffer, &bytes))?;
    Ok(alloc)
}

impl<'a> Hy4RankWeights<'a> {
    pub(crate) fn load_all(
        devices: &[Device<'a>],
        programs: &'a Programs<'a>,
        checkpoint: &Checkpoint,
        cfg: &Hy4FlashConfig,
        max_rows: usize,
        max_head_rows: usize,
    ) -> Result<Vec<Self>> {
        let gpus = devices.len();
        ensure!(matches!(gpus, 1 | 2), "Hy4-flash supports 1 or 2 coordinator GPUs, not {gpus}");
        let model = hy4_plan::Model::open(checkpoint)?;
        let modes = hy4_placement::layer_modes(cfg, gpus, AttentionPlacement::Heads);
        let layouts = target_arenas(checkpoint, cfg, &model.core, &modes, gpus)?;
        ensure!(layouts.len() == gpus, "target_arenas rank count mismatch");

        let mut ranks = Vec::with_capacity(gpus);
        for (rank, layout) in layouts.into_iter().enumerate() {
            let device = devices[rank];
            let arena = DeviceWeightArena::new(
                device,
                layout.arena.clone(),
                ArenaTag::weights("hy4_flash/target"),
            )?;

            // Load compact-core EXL3/FP8 projections directly into their preplanned arena views.
            let native = Hy4NativeCore::load(device, programs, checkpoint, &layout, max_rows, &arena)?;
            let claimed_by_native = |name: &str| -> bool {
                for suffix in [".trellis", ".suh", ".svh"] {
                    if let Some(stem) = name.strip_suffix(suffix) {
                        if layout.core.contains_key(stem) {
                            return true;
                        }
                    }
                }
                for suffix in [".payload", ".scales"] {
                    if let Some(stem) = name.strip_suffix(suffix) {
                        if layout.fp8.contains_key(stem) {
                            return true;
                        }
                    }
                }
                false
            };

            let mut views = BTreeMap::new();
            let mut embedding = None;

            for (name, view_meta) in layout.arena.views() {
                if claimed_by_native(name) {
                    continue;
                }
                if name == "model.embed_tokens.weight" {
                    let entry = checkpoint
                        .tensors
                        .iter()
                        .find(|t| t.meta.name == *name)
                        .context("Hy4-flash missing model.embed_tokens.weight")?;
                    let source = EmbedSource::new(
                        &checkpoint.snapshot,
                        &entry.shard,
                        &entry.meta,
                        cfg.hidden,
                    )?;
                    let (emb, ()) = device.run(|| {
                        TokenEmbedding::load_with(
                            device.library,
                            source,
                            EmbedPlacement::Gpu,
                            || Ok(()),
                            |_, bytes| arena.view("model.embed_tokens.weight", bytes),
                        )
                    })?;
                    embedding = Some(emb);
                    continue;
                }

                let alloc = arena.view(name, view_meta.bytes())?;
                let upload_bytes: Vec<u8> = if let Some(packed) = read_packed_engram(checkpoint, cfg, name)? {
                    packed
                } else {
                    let stem = name.strip_suffix(".weight").unwrap_or(name);
                    if stem.ends_with(".self_attn.kv_b_proj") {
                        if let Some(proj) = model.core.get(stem) {
                            let local_heads = (proj.shape[0] / gpus) / (cfg.qk_nope_dim + cfg.v_head_dim);
                            dequant_kv_b_trellis_absorbed(
                                device,
                                programs,
                                checkpoint,
                                proj,
                                rank,
                                gpus,
                                local_heads,
                                cfg.qk_nope_dim,
                                cfg.v_head_dim,
                                cfg.kv_lora_rank,
                            )?
                        } else {
                            let (raw, dtype, shape) = read_tensor_bytes(checkpoint, name)?;
                            ensure!(dtype == DType::Bf16 && shape.len() == 2, "bad kv_b_proj shape for {name}");
                            let sliced = if gpus == 2 {
                                slice_2d(&raw, shape[0], shape[1], 2, Axis::Rows, rank, gpus)
                            } else {
                                raw
                            };
                            let local_heads = (shape[0] / gpus) / (cfg.qk_nope_dim + cfg.v_head_dim);
                            pack_kv_b_absorbed(&sliced, local_heads, cfg.qk_nope_dim, cfg.v_head_dim, cfg.kv_lora_rank)?
                        }
                    } else {
                        let ckpt_name = if checkpoint.contains_tensor(name) {
                            name.clone()
                        } else {
                            format!("{name}.weight")
                        };
                        let (raw, dtype, shape) = read_tensor_bytes(checkpoint, &ckpt_name)?;
                        let elem = match dtype {
                            DType::Bf16 | DType::F16 => 2usize,
                            DType::F32 => 4usize,
                            other => bail!("Hy4-flash unsupported target tensor dtype {other:?} for {ckpt_name}"),
                        };

                        if stem.ends_with(".self_attn.learnable_sink_param") {
                            ensure!(dtype == DType::F32 && shape.len() == 1, "bad sink shape for {name}");
                            let local_heads = shape[0] / gpus;
                            let start = rank * local_heads * 4;
                            raw[start..start + local_heads * 4].to_vec()
                        } else if stem == "lm_head" || (gpus == 2 && shape.len() == 2) {
                            let split = if stem == "lm_head" {
                                CoreSplit::Output
                            } else {
                                core_split(stem)
                            };
                            match split {
                                CoreSplit::Output if gpus == 2 => {
                                    slice_2d(&raw, shape[0], shape[1], elem, Axis::Rows, rank, gpus)
                                }
                                CoreSplit::Input if gpus == 2 => {
                                    slice_2d(&raw, shape[0], shape[1], elem, Axis::Cols, rank, gpus)
                                }
                                _ => raw,
                            }
                        } else {
                            raw
                        }
                    }
                };

                ensure!(
                    upload_bytes.len() == view_meta.bytes(),
                    "Hy4-flash {name}: upload bytes {} != arena view {}",
                    upload_bytes.len(),
                    view_meta.bytes()
                );
                device.run(|| device.library.copy_h2d(alloc.buffer, &upload_bytes))?;
                views.insert(name.clone(), alloc);
            }

            let inv_freq_dsa = make_inv_freq(device, cfg.qk_rope_dim, cfg.sparse_rope_theta)?;
            let inv_freq_swa = make_inv_freq(device, cfg.swa_qk_rope_dim, cfg.swa_rope_theta)?;

            let local_vocab = cfg.vocab_size / gpus;
            let (head_ws, head) = if views.contains_key("lm_head.weight") {
                let ws = Allocation::new(device, HEAD_WS_BYTES, Tag::workspace("hy4_flash/head-cublas-ws"))?;
                let h = device.run(|| {
                    // SAFETY: `ws` stays alive in `Hy4RankWeights` for the full lifetime of `head`.
                    unsafe {
                        device.library.vocabulary_head_rows(
                            ws.buffer.ptr,
                            cfg.hidden as u32,
                            max_head_rows.min(64) as u32,
                            local_vocab as u32,
                            TargetHead::configured().native_mode(),
                        )
                    }
                })?;
                (Some(ws), Some(h))
            } else {
                (None, None)
            };

            ranks.push(Self {
                device,
                rank,
                gpus,
                layout,
                arena,
                views,
                native,
                embedding,
                inv_freq_dsa,
                inv_freq_swa,
                head_ws,
                head,
                local_vocab,
            });
        }
        Ok(ranks)
    }

    pub(crate) fn buf(&self, name: &str) -> Result<CuteafdDeviceBuffer> {
        self.views
            .get(name)
            .map(|a| a.buffer)
            .with_context(|| format!("Hy4-flash rank {} missing weight view {name}", self.rank))
    }

    pub(crate) fn weight(&self, stem: &str) -> Result<CuteafdDeviceBuffer> {
        self.buf(&format!("{stem}.weight"))
    }

    /// Return `(w_uk [local_heads, 512, 192], w_uv [local_heads, 256, 512])` pointers inside `kv_b_proj`.
    pub(crate) fn kv_b_absorbed(&self, layer: usize, local_heads: usize) -> Result<(*mut std::ffi::c_void, *mut std::ffi::c_void)> {
        let buf = self.weight(&format!("model.layers.{layer}.self_attn.kv_b_proj"))?;
        let uk_bytes = local_heads * 512 * 192 * 2;
        let uv_bytes = local_heads * 256 * 512 * 2;
        ensure!(buf.bytes == uk_bytes + uv_bytes, "kv_b_proj absorbed view size mismatch");
        Ok((buf.ptr, buf.ptr.cast::<u8>().wrapping_add(uk_bytes).cast()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cuteafd_loader::plan::checkpoint::{CheckpointView, CoreView};

    #[test]
    fn pack_kv_b_absorbed_from_transposed_matches_row_major_pack() -> Result<()> {
        let (heads, nope, v_dim, kv_lora) = (2usize, 4usize, 6usize, 8usize);
        let head_rows = nope + v_dim;
        let total_rows = heads * head_rows;
        let mut row_major = vec![0u8; total_rows * kv_lora * 2];
        let mut col_major = vec![0u8; kv_lora * total_rows * 2];
        for r in 0..total_rows {
            for c in 0..kv_lora {
                let val = ((r * 31 + c * 7 + 1) as u16).to_le_bytes();
                let rm = (r * kv_lora + c) * 2;
                let cm = (c * total_rows + r) * 2;
                row_major[rm..rm + 2].copy_from_slice(&val);
                col_major[cm..cm + 2].copy_from_slice(&val);
            }
        }
        let a = pack_kv_b_absorbed(&row_major, heads, nope, v_dim, kv_lora)?;
        let b = pack_kv_b_absorbed_from_transposed(&col_major, heads, nope, v_dim, kv_lora)?;
        assert_eq!(a, b);
        Ok(())
    }

    #[test]
    #[ignore = "requires installed Hy4-flash-EXL3-K4-MC checkpoint in HF_HOME"]
    fn installed_hy4_target_arena_views_match_checkpoint_headers() -> Result<()> {
        let snap = std::path::PathBuf::from(std::env::var("HOME")?).join(
            ".cache/huggingface/hub/models--wrldsuksgo2mars--Hy4-flash-EXL3-K4-MC/snapshots/cc72179113573d751ed3245e3e58d3640694378d",
        );
        let cfg = Hy4FlashConfig::read(&snap)?;
        for view in [CoreView::Original, CoreView::K8Compact, CoreView::K6Compact, CoreView::Fp8Compact] {
            let selected = CheckpointView::select(&snap, view)?;
            let checkpoint = Checkpoint::open_view(&selected)?;
            let model = hy4_plan::Model::open(&checkpoint)?;
            for gpus in [1usize, 2usize] {
                let modes = hy4_placement::layer_modes(&cfg, gpus, AttentionPlacement::Heads);
                let layouts = target_arenas(&checkpoint, &cfg, &model.core, &modes, gpus)?;
                assert_eq!(layouts.len(), gpus);
                for (rank, layout) in layouts.iter().enumerate() {
                    if matches!(view, CoreView::K8Compact | CoreView::K6Compact) {
                        assert!(!layout.core.is_empty(), "view={view} must populate native EXL3 core");
                        assert!(layout.fp8.is_empty());
                    } else if view == CoreView::Fp8Compact {
                        assert!(layout.core.is_empty());
                        assert!(!layout.fp8.is_empty(), "view={view} must populate native FP8 core");
                    } else {
                        assert!(layout.core.is_empty() && layout.fp8.is_empty());
                    }
                    for (name, view_meta) in layout.arena.views() {
                        if name == "model.embed_tokens.weight" {
                            assert!(checkpoint.contains_tensor(name));
                            continue;
                        }
                        if name.ends_with(".trellis") || name.ends_with(".suh") || name.ends_with(".svh") {
                            let stem = name.rsplit_once('.').unwrap().0;
                            assert!(layout.core.contains_key(stem), "view={view} unexpected trellis view {name}");
                            continue;
                        }
                        if name.ends_with(".payload") || name.ends_with(".scales") {
                            let stem = name.rsplit_once('.').unwrap().0;
                            assert!(layout.fp8.contains_key(stem), "view={view} unexpected fp8 view {name}");
                            continue;
                        }
                        if name.ends_with(".packed") || name.ends_with(".packed.weight") || name.ends_with(".packed.bias") {
                            let packed = read_packed_engram(&checkpoint, &cfg, name)?
                                .with_context(|| format!("missing packed handler for {name}"))?;
                            assert_eq!(packed.len(), view_meta.bytes(), "view={view} rank {rank}/{gpus} {name}");
                            continue;
                        }
                        let stem = name.strip_suffix(".weight").unwrap_or(name);
                        if stem.ends_with(".self_attn.kv_b_proj") && model.core.contains_key(stem) {
                            let proj = &model.core[stem];
                            let expected = (proj.shape[0] / gpus) * proj.shape[1] * 2;
                            assert_eq!(expected, view_meta.bytes(), "view={view} rank {rank}/{gpus} {name}");
                            continue;
                        }
                        let entry = checkpoint
                            .tensors
                            .iter()
                            .find(|t| t.meta.name == *name)
                            .with_context(|| format!("view={view} rank {rank}/{gpus} missing checkpoint tensor {name}"))?;
                        let raw_bytes = usize::try_from(entry.meta.byte_length)?;
                        let expected = if stem.ends_with(".self_attn.learnable_sink_param") {
                            raw_bytes / gpus
                        } else if gpus == 2 && entry.meta.shape.len() == 2 {
                            let split = if stem == "lm_head" { CoreSplit::Output } else { core_split(stem) };
                            match split {
                                CoreSplit::Output | CoreSplit::Input => raw_bytes / 2,
                                CoreSplit::Whole => raw_bytes,
                            }
                        } else {
                            raw_bytes
                        };
                        assert_eq!(expected, view_meta.bytes(), "view={view} rank {rank}/{gpus} {name}");
                    }
                }
            }
        }
        Ok(())
    }
}
