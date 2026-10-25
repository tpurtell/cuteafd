//! DeepSeek V4 token-split (`context`) attention on the C4 layers (#58; PLAN.md "v3 attention
//! placement without latent replication", V4 Flash/Pro). Projections stay head-split; each
//! 256-token unit's C4 compressed page and index page live once: logical unit `j` on GPU
//! `j mod 2`, at local page `id >> 1` of its parity pool id. C128 pages, window rings and
//! compressor carry stay replicated (the compressor runs on both GPUs; only the unit's owner
//! keeps its C4 record and index keys, the other writes a per-CTA scratch row past its local
//! pages). The residual is replicated after every layer exactly as under `heads`.
//!
//! **Decode/verify**, per C4 layer on both GPUs, through the shared [`context_decode`]
//! ([`ContextKernels`]): mHC pre, the producer (this GPU's heads' query straight into its half
//! of the assembled exchange query), the replicated compressor; push q. The index producer and
//! the scored top-k over this GPU's units (compact half table); push candidates, wait the
//! peer's, merge into the global top-k by (score desc, logical asc), keeping this GPU's picks as
//! local C4 slots. Wait q; the peer's heads' partial over the local selection (and the window on
//! the GPU whose turn it is by C4 ordinal parity), push it; this GPU's heads; wait the peer's
//! partial of them; `lse_combine2` in GPU0, GPU1 order with this GPU's heads' sink, once, into
//! the head-split attention output; `wo` and the attention all-reduce as under `heads`.
//!
//! **Prefill** gathers: per C4 layer each GPU copies every history unit's C4 page and index page
//! into a contiguous staging view at its logical index (its own units locally, the peer's through
//! peer reads), the compressor writes the chunk's completed groups at their logical rows there,
//! and the unchanged head-split programs read the staging through an identity page table, so the
//! top-k's selection and every record equal `heads`'; then the groups this GPU owns are copied
//! into its pool. Goldens and exact prefix restores therefore do not depend on ownership.
//!
//! **Tables.** A step's per-rank C4 tables (slots, compact half tables, shard lengths, gather
//! lists, commit blocks) are uploaded at the step's start after both streams drained, as the
//! head split's tables are, so no queued kernel reads a table the host rewrites.
//!
//! **Lifetime.** [`V4ContextSplit`] is created after the peer and its exchange, never replaced
//! while the engine lives, and dropped before them (the exchange borrows the head split's abort
//! words); graphs capture only its persistent buffers.
use super::*;
use crate::shared::peer_split::attention::{context_decode, Candidates, ContextKernels, Partial, Query};
use crate::shared::peer_split::context::{local_pages, ContextExchange};
use cuteafd_core::coordinator_programs::AttentionReductionPlan;
use cuteafd_ffi::CuteafdDeviceBuffer;
use cuteafd_loader::placement::families::deepseek_v4::{context_buffers, context_commit_units, context_workspace,
    V4Context};
use super::metadata::{c4_context_commits, c4_context_prefill_tables, c4_context_tables, C4Shard};

/// C4 compressed rows per unit page (64 groups of four tokens).
const C4_ROWS: usize = 64;

/// One GPU's context buffers, in the order and sizes `deepseek_v4::context_workspace` gives the
/// solver (each with the engine's 256-byte floor), then staging.
pub(super) struct ContextRank<'a> {
    /// Scored top-k output: FP32 scores, then i32 logical indices, `[decode rows, K]` each.
    candidates: Dev<'a>,
    /// This GPU's heads' and the peer's heads' partials: BF16 `[rows, H/2, 512]`, then FP32 LSE.
    own: Dev<'a>,
    peer: Dev<'a>,
    topk_scratch: Dev<'a>,
    partial_scratch: Dev<'a>,
    /// Compact local-unit lists of a prefill's history: GPU0's units, then GPU1's.
    gather: Dev<'a>,
    /// The merge's local selection length per row (its slots go to the workspace's `selected`).
    local_lengths: Dev<'a>,
    /// Zero window lengths for the GPU that does not attend the window this layer.
    empty_window: Dev<'a>,
    /// Per prefill lane: C4 payload, footer, index keys and index scales, each `[src | dst | bytes]` u64.
    commit: Vec<Dev<'a>>,
    /// Two layer-parity views: `[units, 37440]` C4 pages, then `[units, 8448]` index pages.
    staging: [Dev<'a>; 2],
}

/// Both GPUs' context buffers, the exchange and the per-GPU partial programs.
pub(super) struct V4ContextSplit<'a> {
    ranks: [ContextRank<'a>; 2],
    /// Dropped before the head split's exchange (it borrows its abort words).
    pub exchange: ContextExchange<'a>,
    /// Logical units one staging view holds: the compiled extent's.
    staging_units: usize,
    /// Per GPU, the partial program of heads `[0, H/2)` and `[H/2, H)`: one split plan chosen
    /// at load from the attention reduction plan, the same for every step's row count, so k
    /// one-row steps and one k-row verify reduce identically (lossless speculation).
    partial: [[String; 2]; 2],
    /// Physical layer -> C4 ordinal (`None`: not a C4 layer).
    ordinal: Vec<Option<usize>>,
    commit_limit: usize,
}

/// A step's per-rank C4 context tables: the shard (decode) or the gathered prefill tables, with
/// each prefill lane's gather and commit counts.
#[derive(Debug, Clone, Default)]
pub(super) struct ContextUnit {
    /// Full history units per owner; attention skips gathering only after its exact predecessor.
    pub shards: [usize; 2],
    /// Commit blocks per plane (payload, footer, keys, scales).
    pub commits: [usize; 4],
}

impl<'a> V4ContextSplit<'a> {
    /// Allocates both GPUs' context buffers and the exchange for `engine` (whose peer is attached)
    /// over `extent` tokens; `abort`: the head split's terminal abort words.
    pub fn new(engine: &Engine<'a>, context: V4Context, extent: usize, manifest: &serde_json::Value,
        abort: Option<[*const u32; 2]>, plan: &AttentionReductionPlan) -> Result<Self> {
        let cfg = &engine.cfg;
        let (prefill, decode, lanes) = (engine.prefill_rows as u64, engine.decode_rows as u64, engine.prefill_waves as u64);
        let buffers = context_buffers(cfg, extent as u64, prefill, decode, lanes, context);
        let ranks = [RankDevice { device: engine.device, stream: engine.stream_of(0) },
            RankDevice { device: engine.peer()?.device, stream: engine.stream_of(1) }];
        let exchange = ContextExchange::new(engine.library, ranks, &buffers, abort)?;
        ensure!(exchange.heads() == cfg.n_heads / 2 && exchange.topk() == cfg.index_topk, "V4 context exchange geometry");
        let staging_units = extent.div_ceil(metadata::SOURCE_PAGE_TOKENS);
        // The per-layer scratch pages live inside the C4 pools (`engine::c4_units`), past the
        // local units, so compressor slots address them; the rest is allocated here as listed.
        let list: Vec<_> = context_workspace(cfg, extent as u64, prefill, decode, lanes, context).into_iter()
            .filter(|(name, _)| !name.starts_with("C4 scratch")).collect();
        ensure!(metadata::C4_SCRATCH_UNITS as u64 >= cuteafd_loader::placement::families::deepseek_v4::context_scratch_units(decode),
            "{decode} decode rows need more than {} C4 scratch pages", metadata::C4_SCRATCH_UNITS);
        let mut partial: [[String; 2]; 2] = Default::default();
        let mut build = |rank: usize| -> Result<ContextRank<'a>> {
            engine.on(rank, || {
                let sms = engine.library.sm_count()?;
                for (slot, begin) in partial[rank].iter_mut().zip([0, cfg.n_heads / 2]) {
                    let program = cuteafd_core::coordinator_programs::attention_partial_program(manifest, engine.family, begin, engine.decode_rows, sms, plan)?;
                    *slot = program.strip_prefix(engine.family).and_then(|p| p.strip_prefix('_'))
                        .context("partial program of another family")?.to_string();
                }
                let _scope = cuteafd_ffi::memory_ledger::scope("context-workspace");
                let mut allocations = list.iter().map(|&(_, bytes)| engine.alloc(bytes as usize))
                    .collect::<Result<Vec<_>>>()?.into_iter();
                let mut next = || allocations.next().context("V4 context workspace list");
                let (candidates, own, peer, topk_scratch, partial_scratch, gather, local_lengths, empty_window) =
                    (next()?, next()?, next()?, next()?, next()?, next()?, next()?, next()?);
                let commit = (0..lanes).map(|_| next()).collect::<Result<Vec<_>>>()?;
                ensure!(allocations.next().is_none(), "V4 context workspace list has unbound entries");
                for zeroed in [&topk_scratch, &empty_window] {
                    engine.library.cuda_zero_bytes(zeroed.buffer, zeroed.buffer.bytes)?;
                }
                let _scope = cuteafd_ffi::memory_ledger::scope("context-staging");
                let view = staging_units * (metadata::compressed_page_bytes(4) + INDEX_PAGE_BYTES);
                let staging = [engine.alloc(view)?, engine.alloc(view)?];
                let half = staging_units.div_ceil(2);
                for slot in 0..lanes as usize {
                    for owner in 0..2 {
                        engine.reserve_upload(super::context::view(&gather, (slot * 2 + owner) * half * 4, half * 4)?)?;
                    }
                    engine.reserve_upload(commit[slot].buffer)?;
                }
                Ok(ContextRank { candidates, own, peer, topk_scratch, partial_scratch, gather, local_lengths,
                    empty_window, commit, staging })
            })
        };
        let ranks = [build(0)?, build(1)?];
        let mut c4 = 0;
        let ordinal = cfg.compress_ratios.iter().map(|&ratio| (ratio == 4).then(|| { c4 += 1; c4 - 1 })).collect();
        Ok(Self { ranks, exchange, staging_units, partial, ordinal,
            commit_limit: context_commit_units(prefill) as usize })
    }

    /// Bytes this split allocated on each GPU beside the exchange (the solver's "context workspace"
    /// with its floors + "context staging").
    pub fn bytes(&self) -> [usize; 2] {
        self.ranks.each_ref().map(|r| [&r.candidates, &r.own, &r.peer, &r.topk_scratch, &r.partial_scratch,
            &r.gather, &r.local_lengths, &r.empty_window].into_iter().chain(&r.commit)
            .chain(&r.staging).map(|d| d.buffer.bytes).sum())
    }

    /// The C4 ordinal of physical layer `layer` (`None`: head-split C128/window layer).
    pub fn ordinal(&self, layer: usize) -> Option<usize> {
        self.ordinal.get(layer).copied().flatten()
    }
}

/// A view of `bytes` at `offset` inside `dev`.
fn view(dev: &Dev<'_>, offset: usize, bytes: usize) -> Result<CuteafdDeviceBuffer> {
    ensure!(offset + bytes <= dev.buffer.bytes, "view of {bytes} B at {offset} past a {} B buffer", dev.buffer.bytes);
    Ok(CuteafdDeviceBuffer {
        // SAFETY: inside the allocation (checked above).
        ptr: unsafe { dev.buffer.ptr.cast::<u8>().add(offset) }.cast(),
        bytes,
        ..dev.buffer
    })
}

impl<'a> Engine<'a> {
    /// Whether the C4 pools are partitioned by unit parity (token-split attention).
    pub(crate) fn context_kv(&self) -> bool {
        self.context_mode.is_some()
    }

    /// C4 pool pages on each GPU: half the logical units under context (scratch pages are separate).
    pub(crate) fn c4_local_units(&self) -> usize {
        if self.context_kv() { self.shape.units / 2 } else { self.shape.units }
    }

    /// Allocates the token-split buffers and exchange once the peer is attached; `extent`: the
    /// admitted compiled context (the staging the solver charged).
    pub fn attach_context(&mut self, extent: usize, manifest: &serde_json::Value, plan_rows: Option<usize>)
        -> Result<()> {
        let context = self.context_mode.context("this engine was built without context attention")?;
        let exchange = self.exchange()?;
        let abort = match (exchange.abort_word(0)?, exchange.abort_word(1)?) {
            (Some(a), Some(b)) => Some([a, b]),
            _ => None,
        };
        // One reduction plan for every step, after the drafter loaded: each sequence verifies
        // draft_block() + 1 rows (C1), plain decode one.
        let block = self.draft_block();
        let plan = AttentionReductionPlan::new(self.decode_rows, (block > 0).then_some(block), plan_rows)
            .map_err(|error| anyhow::anyhow!("V4 attention reduction plan: {error:?}"))?;
        let split = V4ContextSplit::new(self, context, extent, manifest, abort, &plan)?;
        tracing::info!(bytes = ?split.bytes(), exchange = split.exchange.bytes(), plan_rows = plan.rows(),
            partial = ?split.partial, "V4 C4 context attention buffers");
        self.context = Some(split);
        Ok(())
    }

    pub(super) fn context(&self) -> Result<&V4ContextSplit<'a>> {
        self.context.as_ref().context("V4 context split")
    }

    /// Rank `rank`'s C4 tables for a step under context, uploaded into `lane`'s buffers (both
    /// streams ordered by the caller): decode/verify: owned or scratch output slots, compact half
    /// tables, shard lengths; prefill: the gathered view's logical slots and identity table, the
    /// history units to gather and the owned groups to commit (lane `index`; `gather` 0 skips the
    /// gather: an earlier lane of the same step gathered every unit this lane reads).
    pub(super) fn put_context_tables(&self, rank: usize, m: &StepBuffers<'_>, tables: &StepTables, index: usize,
        gather: bool, units: &[u32]) -> Result<ContextUnit> {
        let split = self.context()?;
        let own = &split.ranks[rank];
        let mut unit = ContextUnit::default();
        let shard = if tables.decode {
            c4_context_tables(tables, rank, self.shape.units)?
        } else {
            let shard = c4_context_prefill_tables(tables)?;
            // History: every unit holding groups below this chunk (a partial tail is history too).
            let history = if gather { tables.start.div_ceil(metadata::SOURCE_PAGE_TOKENS) } else { 0 };
            ensure!(history <= units.len() && history <= split.staging_units, "C4 history past the sequence's units");
            let half = split.staging_units.div_ceil(2);
            for (owner, count) in unit.shards.iter_mut().enumerate() {
                let local = local_pages(&units[..history], owner)?;
                *count = local.len();
                if !local.is_empty() {
                    self.on(rank, || self.upload(view(&own.gather, (index * 2 + owner) * half * 4, local.len() * 4)?,
                        bytes_of(&local)))?;
                }
            }
            let first = tables.start / 4;
            let groups = (tables.start + tables.rows) / 4 - first;
            let commits = c4_context_commits(rank, first, groups, units, split.commit_limit)?;
            unit.commits.copy_from_slice(&commits.counts);
            let commit = own.commit.get(index).context("commit lane")?;
            self.on(rank, || self.upload(commit.buffer, bytes_of(&commits.table)))?;
            shard
        };
        self.on(rank, || self.fill_c4(m, &shard))?;
        Ok(unit)
    }

    /// Uploads a rank's C4 shard over the lane's replicated C4 tables (after [`Self::fill`]).
    fn fill_c4(&self, m: &StepBuffers<'_>, shard: &C4Shard) -> Result<()> {
        let put = |buffer: &Dev<'_>, bytes: &[u8]| -> Result<()> {
            ensure!(bytes.len() <= buffer.buffer.bytes, "C4 shard table exceeds its buffer");
            self.upload(CuteafdDeviceBuffer { bytes: bytes.len(), ..buffer.buffer }, bytes)
        };
        for (buffer, (_, values)) in m.c4.iter().zip(&shard.c4_tables) {
            put(buffer, bytes_of(values))?;
        }
        put(&m.c4_page_table, bytes_of(&shard.c4_page_table))?;
        put(&m.c4_visible, bytes_of(&shard.c4_visible))
    }

    /// Rank `rank`'s C4 record and index pools of layer `layer` (its local units, then the scratch
    /// pages its non-owner compressor rows write).
    fn c4_pools(&self, rank: usize, layer: usize) -> Result<(*mut c_void, *mut c_void)> {
        let cache = &self.caches_on(rank)[layer];
        Ok((cache.compressed.as_ref().context("C4 pool")?.buffer.ptr, cache.index.as_ref().context("C4 index")?.buffer.ptr))
    }

    /// Decode/verify attention of C4 layer `layer` on `rank` under context (module docs): leaves
    /// this rank's head-split `wo` partial in `delta`, as [`Self::attention_front`] does.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn context_decode_front(&self, rank: usize, layer: usize, weights: &LayerWeights<'_>,
        tables: &StepTables, lane: &UnitSlot<'_>, w: &Workspace<'_>, rows: Scalar, cap: usize) -> Result<()> {
        struct Bindings<'e, 'a, 'w, 'l> {
            engine: &'e Engine<'a>, weights: &'e LayerWeights<'l>, tables: &'e StepTables, lane: &'e UnitSlot<'w>,
            w: &'e Workspace<'w>, rank: usize, layer: usize, ordinal: usize, rows: Scalar, cap: usize, t: usize,
        }
        impl ContextKernels for Bindings<'_, '_, '_, '_> {
            fn produce(&mut self) -> Result<Query> {
                let (e, w, rank) = (self.engine, self.w, self.rank);
                // mHC pre and this GPU's heads' query `[t, H/2, 512]` (the exchange places it in
                // both GPUs' assembled query), then the replicated compressor: owned groups into
                // this GPU's pool, the peer's into its scratch rows.
                e.attention_produce(rank, self.layer, self.weights, self.lane, w, self.rows, self.cap)?;
                let cache = &e.caches_on(rank)[self.layer];
                e.compress_records(rank, self.layer, self.weights, cache, None, self.tables, &self.lane.tables, w,
                    e.attention_rope(rank, 4), self.rows, self.cap)?;
                Ok(Query { values: w.query.buffer.ptr, row_pitch: e.cfg.n_heads / 2 * 512 * 2 })
            }
            fn score(&mut self) -> Result<Option<Candidates>> {
                let (e, w, rank) = (self.engine, self.w, self.rank);
                if self.tables.c4_groups == 0 { return Ok(None); }
                let m = &self.lane.tables;
                e.index_produce(rank, self.weights, m, w, e.attention_rope(rank, 4), self.rows, self.cap)?;
                let own = &e.context()?.ranks[rank];
                let k = e.cfg.index_topk;
                let indices = view(&own.candidates, e.decode_rows * k * 4, self.t * k * 4)?.ptr;
                let (_, index) = e.c4_pools(rank, self.layer)?;
                e.run_on(rank, false, &format!("scored_index_topk_decode_m{}", e.decode_rows), &[
                    ("q", w.index_query.buffer.ptr), ("weights", w.index_weights.buffer.ptr), ("cache", index),
                    ("table", m.c4_page_table.buffer.ptr), ("lengths", m.c4_visible.buffer.ptr), ("indices", indices),
                    ("scores", own.candidates.buffer.ptr), ("scratch", own.topk_scratch.buffer.ptr)],
                    &[self.rows, Scalar::I32(self.width()), Scalar::I32(self.stride()), Scalar::I32(2),
                        Scalar::I32(rank as i32)])?;
                Ok(Some(Candidates { scores: own.candidates.buffer.ptr, indices }))
            }
            fn merge(&mut self, scores: *mut c_void, indices: *mut c_void) -> Result<()> {
                let (e, rank) = (self.engine, self.rank);
                let own = &e.context()?.ranks[rank];
                let k = e.cfg.index_topk;
                e.run_on(rank, false, "dsa_candidate_merge", &[("scores", scores), ("indices", indices),
                    ("out_scores", own.candidates.buffer.ptr),
                    ("out_indices", view(&own.candidates, e.decode_rows * k * 4, self.t * k * 4)?.ptr),
                    ("table", self.lane.tables.c4_page_table.buffer.ptr), ("local_slots", self.w.selected.buffer.ptr),
                    ("local_lengths", own.local_lengths.buffer.ptr)],
                    &[self.rows, Scalar::I32(self.stride()), Scalar::I32(rank as i32)])
            }
            fn partial(&mut self, query: *mut c_void, heads_rank: usize) -> Result<Partial> {
                let (e, w, rank, m) = (self.engine, self.w, self.rank, &self.lane.tables);
                let split = e.context()?;
                let own = &split.ranks[rank];
                let out = if heads_rank == rank { &own.own } else { &own.peer };
                let heads = e.cfg.n_heads / 2;
                let lse = view(out, e.decode_rows * heads * 512 * 2, self.t * heads * 4)?.ptr;
                let (records, _) = e.c4_pools(rank, self.layer)?;
                // The window is attended once, by the GPU whose turn it is by C4 ordinal parity.
                let window = if self.ordinal % 2 == rank { m.swa_lengths.buffer.ptr } else { own.empty_window.buffer.ptr };
                // Before the first completed group nothing is selected: an empty C4 list.
                let selected = if self.tables.c4_groups == 0 { own.empty_window.buffer.ptr } else { own.local_lengths.buffer.ptr };
                e.run_on(rank, false, &split.partial[rank][heads_rank], &[
                    ("q", query), ("cache", e.caches_on(rank)[self.layer].main.buffer.ptr),
                    ("indices", m.swa_indices.buffer.ptr), ("lengths", window), ("extra_cache", records),
                    ("extra_indices", w.selected.buffer.ptr), ("extra_lengths", selected), ("out", out.buffer.ptr),
                    ("lse", lse), ("scratch", own.partial_scratch.buffer.ptr)], &[self.rows])?;
                Ok(Partial { values: out.buffer.ptr, lse })
            }
            fn combine(&mut self, [first, second]: [Partial; 2]) -> Result<()> {
                // This GPU's heads' sink (its head-split share), counted once here.
                self.engine.run_on(self.rank, false, "lse_combine2", &[("o0", first.values), ("l0", first.lse),
                    ("o1", second.values), ("l1", second.lse), ("sink", self.weights.ptr("attn_sink")?),
                    ("out", self.w.attn_out.buffer.ptr)], &[self.rows])
            }
        }
        impl Bindings<'_, '_, '_, '_> {
            fn width(&self) -> i32 { self.tables.c4_table_width.div_ceil(2).max(1) as i32 }
            fn stride(&self) -> i32 { self.tables.c4_table_stride.div_ceil(2).max(1) as i32 }
        }
        let split = self.context()?;
        let ordinal = split.ordinal(layer).context("context attention on a non-C4 layer")?;
        let t = tables.rows;
        context_decode(&split.exchange, rank, ordinal, 0, t, &mut Bindings { engine: self, weights, tables, lane,
            w, rank, layer, ordinal, rows, cap, t })?;
        self.attention_project(rank, weights, lane, w, rows, cap)
    }

    /// Prefill attention of C4 layer `layer` on `rank` under context (module docs): gather the
    /// history into the layer parity's staging view, the head-split programs over it, then this
    /// GPU's completed groups into its pools. Leaves the `wo` partial in `delta`.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn context_prefill_front(&self, rank: usize, layer: usize, weights: &LayerWeights<'_>,
        tables: &StepTables, lane: &UnitSlot<'_>, index: usize, w: &Workspace<'_>, rows: Scalar, cap: usize,
        unit: &ContextUnit) -> Result<()> {
        let split = self.context()?;
        let ordinal = split.ordinal(layer).context("context attention on a non-C4 layer")?;
        let own = &split.ranks[rank];
        let staging = &own.staging[ordinal % 2];
        let page = metadata::compressed_page_bytes(4);
        let records = view(staging, 0, split.staging_units * page)?;
        let keys = view(staging, records.bytes, split.staging_units * INDEX_PAGE_BYTES)?;
        let half = split.staging_units.div_ceil(2);
        for (owner, &count) in unit.shards.iter().enumerate().filter(|(_, &n)| n > 0) {
            let (pool, pool_keys) = self.c4_pools(owner, layer)?;
            let table = view(&own.gather, (index * 2 + owner) * half * 4, count * 4)?.ptr;
            // Whole pages (payload, footer, padding), staged at logical unit 2i + owner.
            let gather = |name: &str, pool: *mut c_void, staging: *mut c_void| self.run_on(rank, false, name,
                &[("pool", pool), ("table", table), ("staging", staging)],
                &[Scalar::I32((count * C4_ROWS) as i32), Scalar::I32(2), Scalar::I32(owner as i32)]);
            gather("paged_staging_gather_c4", pool, records.ptr)?;
            gather("paged_staging_gather_index", pool_keys, keys.ptr)?;
        }
        // The unchanged head-split layer over the staging view: the compressor writes the chunk's
        // groups at their logical rows, the top-k and sparse MLA read it through an identity table.
        self.attention_produce(rank, layer, weights, lane, w, rows, cap)?;
        let cache = &self.caches_on(rank)[layer];
        let rope = self.attention_rope(rank, 4);
        let (_, indexed_cache, indexed_indices, indexed_lengths) = self.compress(rank, layer, weights, cache,
            Some((records.ptr, keys.ptr)), tables, &lane.tables, w, rope, rows, cap)?;
        self.attention_sparse(rank, weights, tables, lane, w, rows, cap, cache.main.buffer.ptr, indexed_cache,
            indexed_indices, indexed_lengths)?;
        self.attention_project(rank, weights, lane, w, rows, cap)?;
        // This GPU's completed groups, staging -> pool, per plane.
        let commit = own.commit.get(index).context("commit lane")?;
        let limit = split.commit_limit;
        let (pool, pool_keys) = self.c4_pools(rank, layer)?;
        for (plane, (&count, (source, destination))) in unit.commits.iter()
            .zip([(records, pool), (records, pool), (keys, pool_keys), (keys, pool_keys)]).enumerate() {
            if count == 0 { continue; }
            let list = |part: usize| -> Result<*const c_void> {
                Ok(view(commit, ((plane * 3 + part) * limit) * 8, count * 8)?.ptr.cast_const())
            };
            // SAFETY: staging and pool are this GPU's live buffers; each listed block lies inside
            // both (built from the sequence's units in `put_context_tables`), and the lists stay
            // unchanged until the next step, which follows a drain of this stream.
            self.on(rank, || unsafe { self.library.cuda_kv_cache_write_blocks_async(source.ptr, destination,
                list(0)?, list(1)?, list(2)?, count, self.stream_of(rank)) })?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg(heads: usize, topk: usize, ratios: Vec<usize>) -> cuteafd_loader::families::deepseek_v4::DeepseekV4Config {
        let mut cfg = cuteafd_loader::families::deepseek_v4::DeepseekV4Config::from_model_args(&serde_json::json!({
            "vocab_size":129280,"dim":4096,"moe_inter_dim":2048,"n_layers":ratios.len(),"n_hash_layers":3,
            "n_heads":heads,"n_routed_experts":256,"n_shared_experts":1,"n_activated_experts":6,
            "score_func":"sqrtsoftplus","route_scale":1.5,"swiglu_limit":10.0,"q_lora_rank":1024,
            "head_dim":512,"rope_head_dim":64,"o_groups":8,"o_lora_rank":1024,"window_size":128,
            "original_seq_len":65536,"rope_theta":10000,"rope_factor":16,"beta_fast":32,"beta_slow":1,
            "index_n_heads":64,"index_head_dim":128,"index_topk":topk,"hc_mult":4,"hc_sinkhorn_iters":20,
            "compress_rope_theta":160000,"compress_ratios":ratios}), 0).unwrap();
        cfg.n_heads = heads;
        cfg
    }

    /// The engine allocates the solver's context workspace exactly: every listed buffer except
    /// the per-layer scratch pages, which live inside each C4 pool (`engine::c4_units`), at the
    /// same bytes the solver charged for them.
    #[test]
    fn engine_allocations_are_the_solver_charge() {
        for (heads, topk, ratios) in [(64, 512, (0..43).map(|i| if i < 2 { 0 } else if i % 2 == 0 { 4 } else { 128 }).collect()),
            (128, 1024, (0..61).map(|i| if i < 2 { 0 } else if i % 2 == 0 { 4 } else { 128 }).collect::<Vec<_>>())] {
            let cfg = cfg(heads, topk, ratios);
            let c4 = cfg.compress_ratios.iter().filter(|&&r| r == 4).count();
            let context = V4Context { topk_scratch: 9_178_112, partial_scratch: 4_000_000 };
            let list = context_workspace(&cfg, 1 << 20, 4096, 64, PREFILL_WAVES as u64, context);
            let charged: u64 = list.iter().map(|&(_, b)| b.max(256)).sum();
            let allocated: u64 = list.iter().filter(|(n, _)| !n.starts_with("C4 scratch")).map(|&(_, b)| b.max(256)).sum();
            let in_pools = (c4 * metadata::C4_SCRATCH_UNITS * (metadata::compressed_page_bytes(4) + INDEX_PAGE_BYTES)) as u64;
            assert_eq!(allocated + in_pools, charged, "{heads} heads");
            assert_eq!(context_buffers(&cfg, 1 << 20, 4096, 64, PREFILL_WAVES as u64, context).workspace_bytes, charged);
            // The listed order the engine binds: 8 shared buffers, then one commit list per lane.
            let names: Vec<_> = list.iter().map(|(n, _)| *n).filter(|n| !n.starts_with("C4 scratch")).collect();
            assert_eq!(names, ["candidates", "own-head partial", "peer-head partial", "scored top-k scratch",
                "partial scratch", "gather tables", "local selected lengths", "empty window lengths",
                "commit blocks", "commit blocks"]);
        }
        // One scratch page covers every decode row (a step's discarded groups write distinct rows).
        assert!(metadata::C4_SCRATCH_UNITS as u64
            >= cuteafd_loader::placement::families::deepseek_v4::context_scratch_units(64));
    }

    /// V4's reduction plan is chosen once from dSpark's verify width (block 5 + 1 = 6 rows) and
    /// selects the same partial for every step row count; without a drafter, the one-row plan.
    #[test]
    fn one_partial_plan_per_card_from_the_draft_width() {
        // As the exporter records them: each variant lists only the SM counts it serves.
        let caps = |s: &str, c: serde_json::Value| {
            let sms: Vec<u64> = c.as_object().unwrap().keys().map(|k| k.parse().unwrap()).collect();
            serde_json::json!({"name": s, "params": {"row_caps": c, "device_sm_counts": sms}})
        };
        let mut programs = Vec::new();
        for begin in [0, 32] {
            programs.push(caps(&format!("dsv4f_sparse_mla_partial_h{begin}_s10_m64"), serde_json::json!({"170": [1, 8], "188": [1, 8]})));
            programs.push(caps(&format!("dsv4f_sparse_mla_partial_h{begin}_s4_m64"), serde_json::json!({"188": [64]})));
            programs.push(caps(&format!("dsv4f_sparse_mla_partial_h{begin}_s1_m64"), serde_json::json!({"170": [64]})));
        }
        let manifest = serde_json::json!({"programs": programs});
        let dspark = AttentionReductionPlan::new(64, Some(5), None).unwrap();
        assert_eq!(dspark.rows(), 6);
        for sms in [170, 188] {
            for begin in [0, 32] {
                assert_eq!(cuteafd_core::coordinator_programs::attention_partial_program(&manifest, "dsv4f", begin, 64, sms, &dspark).unwrap(),
                    format!("dsv4f_sparse_mla_partial_h{begin}_s10_m64"));
            }
        }
        let plain = AttentionReductionPlan::new(64, None, None).unwrap();
        assert_eq!(cuteafd_core::coordinator_programs::attention_partial_program(&manifest, "dsv4f", 0, 64, 188, &plain).unwrap(), "dsv4f_sparse_mla_partial_h0_s10_m64");
        let wide = AttentionReductionPlan::new(64, Some(5), Some(48)).unwrap();
        assert_eq!(cuteafd_core::coordinator_programs::attention_partial_program(&manifest, "dsv4f", 0, 64, 188, &wide).unwrap(), "dsv4f_sparse_mla_partial_h0_s4_m64");
    }

    /// Every decode/verify C4 write lands inside a GPU's local pool or its scratch page: the
    /// compressor's slots address `[0, local units + scratch) * 64` rows of `engine::c4_units`.
    #[test]
    fn shard_slots_stay_inside_the_local_pool_and_its_scratch() -> Result<()> {
        use crate::families::deepseek_v4::pool::{Placement, PoolShape};
        let shape = PoolShape::new(4, 4096, 32);
        let a = Placement::new(0, vec![4, 9, 0, 3, 6, 11]);
        let b = Placement::new(1, vec![8, 13, 2, 5]);
        let rows = 64usize;
        for tables in [metadata::decode_step(&[(&a, 255), (&b, 767), (&a, 1023)], &shape, 512, 8)?,
            metadata::verify_step(&[(&a, 1000, 40), (&b, 700, 24)], &shape, 512, 8)?] {
            for rank in 0..2 {
                let shard = c4_context_tables(&tables, rank, shape.units)?;
                let limit = (shape.units / 2 + metadata::C4_SCRATCH_UNITS) * C4_ROWS;
                let slots = &shard.c4_tables.iter().find(|(n, _)| *n == "compressed_slots").unwrap().1;
                assert!(slots.iter().all(|&s| s >= 0 && (s as usize) < limit), "rank {rank}");
                assert!(tables.rows <= rows);
                // Compact half tables index only local pages (or the padding page past them).
                assert!(shard.c4_page_table.iter().all(|&p| p >= 0 && (p as usize) <= shape.units / 2));
            }
        }
        Ok(())
    }
}
