//! Header-admitted packed target weight arena and host-resident FP8 Engram tables
//! with earliest-possible look-ahead and background pinned staging for Hy4-flash.
use super::native::{dequant_kv_b_trellis_absorbed, load_native_core, Hy4NativeCore};
pub(super) use crate::families::hy4_common::weights::{
    inv_freq_bytes, kv_b_absorbed_ptrs, pack_kv_b_absorbed, read_tensor_bytes,
};
use crate::shared::mapped_table::wait_rows;
use crate::shared::memory::{
    budget::admit_host_memory,
    device::{Allocation, Device},
    receipts::Inventory,
    weight_arena::DeviceWeightArena,
    Tag,
};
use crate::shared::experts::rtx::dense::{DenseLoad, Job, Operand, Rect, Step};
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
use cuteafd_loader::families::hy4_flash::weight_layout::{
    core_split, target_arenas, CoordinatorWeights, ROPE_INV_FREQ_DSA, ROPE_INV_FREQ_SWA,
};
use cuteafd_loader::families::hy4_flash::Hy4FlashConfig;
use cuteafd_loader::formats::mapped_preload::{MappedPlacement, DEFAULT_CAP_BYTES};
use cuteafd_loader::formats::mapped_table::lookahead::{LookaheadHandle, UnitPlan};
use cuteafd_loader::formats::mapped_table::ReadPriority;
use cuteafd_loader::placement::families::hy4_flash as hy4_placement;
use cuteafd_loader::placement::AttentionPlacement;
use cuteafd_loader::plan::checkpoint::Checkpoint;
use cuteafd_loader::plan::families::hy4_flash as hy4_plan;
use cuteafd_loader::shared::token_table::{AdmissionTimedOut, TablePipeline};
use cuteafd_loader::{GatherLease, GatherTicket, TableStatsSnapshot};
use std::cell::Cell;
use std::collections::BTreeMap;
use std::ops::Range;
use std::sync::Arc;
use std::time::{Duration, Instant};

const HEAD_WS_BYTES: usize = cuteafd_ffi::programs::VOCABULARY_HEAD_WORKSPACE;
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

fn packed_engram_rects(checkpoint: &Checkpoint, cfg: &Hy4FlashConfig, name: &str) -> Result<Option<Vec<Rect>>> {
    for (packed, source, count, suffix) in [
        (".lookup_per_head_norms.packed", ".lookup_per_head_norms", cfg.engram_total_heads(), ".weight"),
        (".key_projs.packed.weight", ".key_projs", cfg.hc_mult, ".weight"),
        (".key_projs.packed.bias", ".key_projs", cfg.hc_mult, ".bias"),
        (".norm_key.packed", ".norm_key", cfg.hc_mult, ".weight"),
        (".norm_query.packed", ".norm_query", cfg.hc_mult, ".weight"),
        (".unified_per_hc_scales.packed", ".unified_per_hc_scales", cfg.hc_mult, ""),
    ] {
        if let Some(prefix) = name.strip_suffix(packed) {
            return (0..count).map(|i| {
                let tensor = format!("{prefix}{source}.{i}{suffix}");
                ensure!(checkpoint.header(&tensor).is_some_and(|h| h.dtype == DType::Bf16),
                    "Hy4-flash Engram tensor {tensor} must be BF16");
                Rect::tensor(checkpoint, &tensor, None, None)
            }).collect::<Result<Vec<_>>>().map(Some);
        }
    }
    Ok(None)
}

fn dense_rect(checkpoint: &Checkpoint, name: &str, rank: usize, gpus: usize) -> Result<Rect> {
    crate::families::hy4_common::weights::dense_rect(checkpoint, name, rank, gpus, core_split)
}

/// Pack column-major `W^T [kv_lora, local_heads * (nope + v_dim)]` BF16 (as returned by
/// `Projection::local(I_kv_lora)` on EXL3 `kv_b_proj`) into `[w_uk, w_uv]` identical to `pack_kv_b_absorbed`.
#[cfg(test)]
fn pack_kv_b_absorbed_from_transposed(
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

impl<'a> Hy4RankWeights<'a> {
    pub(crate) fn load_all(
        devices: &[Device<'a>],
        programs: &'a Programs<'a>,
        checkpoint: &Checkpoint,
        cfg: &Hy4FlashConfig,
        max_rows: usize,
        max_head_rows: usize,
        media: crate::shared::vision::push::Contribution,
    ) -> Result<Vec<Self>> {
        let gpus = devices.len();
        ensure!(matches!(gpus, 1 | 2), "Hy4-flash supports 1 or 2 coordinator GPUs, not {gpus}");
        let model = hy4_plan::Model::open(checkpoint)?;
        let modes = hy4_placement::layer_modes(cfg, gpus, AttentionPlacement::Heads);
        let layouts = target_arenas(checkpoint, cfg, &model.core, &modes, gpus)?;
        ensure!(layouts.len() == gpus, "target_arenas rank count mismatch");

        let library = devices[0].library;
        let dense = DenseLoad::mount(library);
        let mount = cuteafd_loader::families::hy4_flash::weight_layout::target_mount_load(checkpoint, gpus)?;
        // At most a serial native conversion reader and the shared dense
        // reader can coexist on a mounted snapshot.
        let mount_owners = [("hy4_flash/dense-mount-source", 2 * mount.pinned_source),
            ("hy4_flash/dense-mount-rows", 2 * mount.full_row_cache),
            ("hy4_flash/dense-mount-tables", 2 * mount.pinned_tables)];
        admit_host_memory(0, &mount_owners, 0, 0)?;
        dense.admit_mount(mount)?;
        let ranks = crate::shared::experts::rtx::push::load_alone_with_media(&checkpoint.snapshot, devices, media, || {
            let mut ranks = Vec::with_capacity(gpus);
            for (rank, layout) in layouts.into_iter().enumerate() {
                let device = devices[rank];
                let arena = DeviceWeightArena::new(
                    device,
                    layout.arena.clone(),
                    ArenaTag::weights("hy4_flash/target"),
                )?;

                // Load compact-core EXL3/FP8 projections directly into their preplanned arena views.
                let native = load_native_core(device, programs, checkpoint, &layout, max_rows, &arena)?;
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
                let mut inv_freq_dsa = None;
                let mut inv_freq_swa = None;

                for (name, view_meta) in layout.arena.views() {
                    if claimed_by_native(name) {
                        continue;
                    }
                    if name == ROPE_INV_FREQ_DSA || name == ROPE_INV_FREQ_SWA {
                        let (dim, theta, view) = if name == ROPE_INV_FREQ_DSA {
                            (cfg.qk_rope_dim, cfg.sparse_rope_theta, &mut inv_freq_dsa)
                        } else {
                            (cfg.swa_qk_rope_dim, cfg.swa_rope_theta, &mut inv_freq_swa)
                        };
                        let bytes = inv_freq_bytes(dim, theta);
                        let alloc = arena.view(name, bytes.len())?;
                        device.run(|| device.library.copy_h2d(alloc.buffer, &bytes))?;
                        *view = Some(alloc);
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
                    let stem = name.strip_suffix(".weight").unwrap_or(name);
                    if stem.ends_with(".self_attn.kv_b_proj") {
                        if !model.core.contains_key(stem) {
                            let header = checkpoint.header(name).context("Hy4 BF16 KV-B header missing")?;
                            ensure!(header.dtype == DType::Bf16 && header.shape.len() == 2,
                                "bad BF16 KV-B shape for {name}");
                            let nope = cfg.qk_nope_dim;
                            let v = cfg.v_head_dim;
                            let lat = cfg.kv_lora_rank;
                            let rect = dense_rect(checkpoint, name, rank, gpus)?;
                            let heads = rect.rows.len() / (nope + v);
                            let uk_bytes = heads * lat * nope * 2;
                            let uv_bytes = heads * v * lat * 2;
                            ensure!(rect.rows.len() == heads * (nope + v) && rect.width() == lat * 2
                                && rect.bytes() == uk_bytes + uv_bytes && alloc.buffer.bytes == uk_bytes + uv_bytes,
                                "Hy4 KV-B source or destination extent differs for {name}");
                            dense.step(Step { device: device.id, rects: vec![rect], jobs: vec![Job::Absorb {
                                heads, nope, v, lat,
                                uk: CuteafdDeviceBuffer { bytes: uk_bytes, ..alloc.buffer },
                                uv: CuteafdDeviceBuffer { ptr: alloc.buffer.ptr.wrapping_byte_add(uk_bytes), bytes: uv_bytes, ..alloc.buffer },
                            }] }, std::ptr::null_mut())?;
                        }
                    } else {
                        let rects = match packed_engram_rects(checkpoint, cfg, name)? {
                            Some(rects) => rects,
                            None => vec![dense_rect(checkpoint, name, rank, gpus)?],
                        };
                        let mut bytes = 0;
                        let lands = rects.into_iter().map(|rect| {
                            let start = bytes;
                            bytes += rect.bytes();
                            (rect, start)
                        }).collect();
                        ensure!(bytes == alloc.buffer.bytes, "Hy4-flash {name}: source and arena sizes differ");
                        dense.operand(Operand { device: device.id, buffer: alloc.buffer, lands, bytes })?;
                    }
                    views.insert(name.clone(), alloc);
                }

                let inv_freq_dsa = inv_freq_dsa.context("Hy4-flash target arena missing inv_freq_dsa view")?;
                let inv_freq_swa = inv_freq_swa.context("Hy4-flash target arena missing inv_freq_swa view")?;

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
        })?;
        dense.release();
        // Keep at most one quantized KV-B conversion live. The push group
        // publishes regular weights first; a failed conversion never exposes
        // the uninitialized absorbed view to execution. Only target arena
        // views are eligible: the compact inventory includes the drafter.
        for r in &ranks {
            for (name, dst) in &r.views {
                let Some(stem) = name.strip_suffix(".weight").filter(|stem| stem.ends_with(".self_attn.kv_b_proj")) else {
                    continue;
                };
                let Some(proj) = model.core.get(stem) else { continue };
                let local_heads = proj.shape[0] / gpus / (cfg.qk_nope_dim + cfg.v_head_dim);
                dequant_kv_b_trellis_absorbed(r.device, programs, checkpoint, proj, r.rank, gpus, local_heads,
                    cfg.qk_nope_dim, cfg.v_head_dim, cfg.kv_lora_rank, dst.buffer)?;
            }
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
        kv_b_absorbed_ptrs(buf, local_heads, 192, 256, 512)
    }

    /// Record the packed target weight arena (one physical backing + all named views)
    /// and optional BF16 head workspace into `inventory`.
    pub(crate) fn inventory(&self, weights_inv: &mut Inventory<'_>, retained_inv: &mut Inventory<'_>) -> Result<()> {
        let rank = self.rank;
        for (name, alloc) in &self.views {
            alloc.inventory(weights_inv, &format!("rank{rank}.{name}"))?;
        }
        self.inv_freq_dsa
            .inventory(weights_inv, &format!("rank{rank}.{ROPE_INV_FREQ_DSA}"))?;
        self.inv_freq_swa
            .inventory(weights_inv, &format!("rank{rank}.{ROPE_INV_FREQ_SWA}"))?;
        if let Some(emb) = &self.embedding {
            emb.inventory(weights_inv, &format!("rank{rank}.embed_tokens"), weights_inv.lifetime())?;
        }
        self.native.inventory(weights_inv, rank)?;
        if let Some(ws) = &self.head_ws {
            ws.inventory(retained_inv, &format!("hy4_flash.head.{rank}.workspace"))?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cuteafd_loader::formats::trellis_core::CoreSplit;
    use cuteafd_loader::plan::checkpoint::{CheckpointView, CoreView};

    #[test]
    fn head_workspace_matches_shared_vocabulary_head_contract() {
        assert_eq!(HEAD_WS_BYTES, cuteafd_ffi::programs::VOCABULARY_HEAD_WORKSPACE);
    }

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
                        if name == ROPE_INV_FREQ_DSA {
                            assert_eq!(view_meta.bytes(), (cfg.qk_rope_dim / 2) * 4);
                            continue;
                        }
                        if name == ROPE_INV_FREQ_SWA {
                            assert_eq!(view_meta.bytes(), (cfg.swa_qk_rope_dim / 2) * 4);
                            continue;
                        }
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
                            let packed = packed_engram_rects(&checkpoint, &cfg, name)?
                                .with_context(|| format!("missing packed handler for {name}"))?;
                            assert_eq!(packed.iter().map(Rect::bytes).sum::<usize>(), view_meta.bytes(), "view={view} rank {rank}/{gpus} {name}");
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
