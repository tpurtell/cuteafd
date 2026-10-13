//! GLM 5.3 token-split (`context`) attention (PLAN.md "v3 attention placement
//! without latent replication", sections 1-3; K3). Projections stay
//! head-split; each 64-row page of the latent and index caches lives once:
//! logical page `j` on GPU `j mod 2`, at local page `id >> 1` of its parity
//! pool id (`RefPagePool::parity`). The residual is replicated after every
//! layer exactly as under `heads`.
//!
//! **Decode/verify**, per layer on both GPUs in queue order (the schedule
//! `order::fixtures::context_decode` checks): the producer writes each row's
//! record only on its page's owner (elsewhere the slot is this GPU's scratch
//! row) and this GPU's heads' absorbed query straight into its half of the
//! assembled exchange query; push q. Full-indexer layers score this GPU's
//! pages through each row's compact half table; push candidates, wait the
//! peer's, merge into the global top-k by (score desc, logical asc) keeping
//! this GPU's picks as physical local slots (the selection the layer's
//! shared-indexer layers reuse). Wait q; the sparse MLA partial of the peer's
//! heads over the local selection, push it; this GPU's heads; wait the peer's
//! partial of them; `lse_combine2` in GPU0, GPU1 order into the head-split
//! attention output; `glm2_o` and the attention all-reduce as under `heads`.
//!
//! **Prefill** gathers: per layer each GPU copies every history page's records
//! (and index pages on full-indexer layers) into a contiguous staging view at
//! its logical index, its own pages locally, the peer's through peer reads,
//! whole pages; the producers write the chunk's rows at their logical slots
//! there; the unchanged head-split programs read the staging through an
//! identity page table, so the top-k's (score, logical) selection, its slot
//! order and every record are those of `heads`; then the chunk's rows on this
//! GPU's pages are copied into its pool. Goldens and exact prefix restores
//! therefore do not depend on chunking or ownership.
//!
//! **Tables.** A step's per-rank tables (slots, compact half tables, shard
//! lengths, gather lists, commit blocks) are uploaded once at the step's start,
//! after both streams drained (as the head split's `peer_tables` does), so no
//! queued kernel ever reads a table the host rewrites.
use super::*;
use crate::shared::peer_split::context::{local_pages, page_owner, shard_rows, ContextExchange};
use cuteafd_ffi::CuteafdDeviceBuffer;
use cuteafd_loader::placement::families::glm5::{context_buffers, context_commit_pages, context_workspace, GlmContext};

/// Bytes of an index page (64 x 128 E4M3 keys, then 64 FP32 scales).
const INDEX_KEYS: usize = PAGE_ROWS * 128;

/// One GPU's context buffers, in the order and sizes `glm5::context_workspace`
/// gives the solver (each with the engine's 256-byte floor), then staging.
pub(super) struct ContextRank<'a> {
    /// Scored top-k output: FP32 scores, then i32 logical indices, `[64, K]` each.
    candidates: Dev<'a>,
    /// This GPU's heads' and the peer's heads' partials: BF16 `[64, H/2, 512]`, then FP32 LSE.
    own: Dev<'a>,
    peer: Dev<'a>,
    topk_scratch: Dev<'a>,
    partial_scratch: Dev<'a>,
    /// Compact local-page lists of a prefill's history: GPU0's pages, then GPU1's.
    gather: Dev<'a>,
    /// Per prefill lane: records, index keys and index scales, each as `[src | dst | bytes]` u64.
    commit: Vec<Dev<'a>>,
    /// Two layer-parity views: `[pages, 64 x 656]` records, then `[pages, 8448]` index pages.
    staging: [Dev<'a>; 2],
}

/// Both GPUs' context buffers, the exchange and the per-GPU programs.
pub(super) struct GlmContextSplit<'a> {
    ranks: [ContextRank<'a>; 2],
    /// Dropped before the head split's exchange (it borrows its abort words).
    pub exchange: ContextExchange<'a>,
    /// Logical pages one staging view holds: the compiled extent's.
    staging_pages: usize,
    /// Per GPU, one immutable partial recipe for each head range.
    partial: [[String; 2]; 2],
    /// `lse_combine2`'s sink operand (unused without a sink; GLM 5.3 has none).
    sink: [Dev<'a>; 2],
}

/// A view of `bytes` at `offset` inside `dev`.
fn view(dev: &Dev<'_>, offset: usize, bytes: usize) -> Result<CuteafdDeviceBuffer> {
    ensure!(offset + bytes <= dev.buffer.bytes, "view of {bytes} B at {offset} past a {} B buffer", dev.buffer.bytes);
    Ok(GlmEngine::region(dev, offset, bytes))
}

impl<'a> GlmContextSplit<'a> {
    /// Allocates both GPUs' context buffers and the exchange for `engine` (whose peer is
    /// attached) over `extent` tokens; `abort`: the head split's terminal abort words.
    pub fn new(engine: &GlmEngine<'a>, context: GlmContext, extent: usize, manifest: &serde_json::Value,
        abort: Option<[*const u32; 2]>) -> Result<Self> {
        let cfg = &engine.cfg;
        let (rows, lanes) = (engine.prefill_rows as u64, engine.prefill_lanes as u64);
        let buffers = context_buffers(cfg, extent as u64, rows, lanes, context);
        let ranks = [RankDevice { device: engine.device, stream: engine.stream_of(0) },
            RankDevice { device: engine.peer()?.device, stream: engine.stream_of(1) }];
        let exchange = ContextExchange::new(engine.library, ranks, &buffers, abort)?;
        ensure!(exchange.heads() == cfg.heads / 2 && exchange.topk() == cfg.index_topk, "context exchange geometry");
        let staging_pages = extent.div_ceil(PAGE_ROWS);
        let list = context_workspace(cfg, extent as u64, rows, lanes, context);
        let mut partial: [[String; 2]; 2] = Default::default();
        let rank = |rank: usize, partial: &mut [String; 2]| -> Result<(ContextRank<'a>, Dev<'a>)> {
            engine.on(rank, || {
                let sms = engine.library.sm_count()?;
                for (slot, begin) in partial.iter_mut().zip([0, cfg.heads / 2]) {
                    *slot = crate::shared::peer_split::context::partial_program(
                        manifest, "glm", begin, DECODE_ROWS, sms, &engine.attention_plan)?;
                }
                let _scope = cuteafd_ffi::memory_ledger::scope("context-workspace");
                let mut allocations = list.iter().map(|&(_, bytes)| engine.alloc(bytes as usize))
                    .collect::<Result<Vec<_>>>()?.into_iter();
                let mut next = || allocations.next().context("context workspace list");
                let (candidates, own, peer, topk_scratch, partial_scratch, gather) =
                    (next()?, next()?, next()?, next()?, next()?, next()?);
                let commit = (0..lanes).map(|_| next()).collect::<Result<Vec<_>>>()?;
                engine.library.cuda_zero_bytes(topk_scratch.buffer, topk_scratch.buffer.bytes)?;
                let _scope = cuteafd_ffi::memory_ledger::scope("context-staging");
                let view = staging_pages * (RECORD_PAGE_BYTES + INDEX_PAGE_BYTES);
                let staging = [engine.alloc(view)?, engine.alloc(view)?];
                // The sink operand is never read (no sink): a 256-byte floor allocation, uncharged.
                let sink = engine.alloc(0)?;
                Ok((ContextRank { candidates, own, peer, topk_scratch, partial_scratch, gather, commit, staging }, sink))
            })
        };
        let [mut first, mut second] = partial.clone();
        let (r0, s0) = rank(0, &mut first)?;
        let (r1, s1) = rank(1, &mut second)?;
        partial = [first, second];
        Ok(Self { ranks: [r0, r1], exchange, staging_pages, partial, sink: [s0, s1] })
    }

    /// Bytes this split allocated on each GPU beside the exchange (the solver's
    /// "context workspace" + "context staging").
    pub fn bytes(&self) -> [usize; 2] {
        self.ranks.each_ref().map(|r| [&r.candidates, &r.own, &r.peer, &r.topk_scratch, &r.partial_scratch, &r.gather]
            .into_iter().chain(&r.commit).chain(&r.staging).map(|d| d.buffer.bytes).sum())
    }
}

/// What a context prefill unit does per GPU at each layer: the history pages it gathers
/// (`[own, peer]` lengths, uploaded in `gather`), and the commit block count.
#[derive(Debug, Clone, Copy, Default)]
pub(super) struct ContextUnit {
    pub history: usize,
    pub shards: [usize; 2],
    pub commits: usize,
}

impl<'a> GlmEngine<'a> {
    /// Local pool pages per GPU (the scratch page follows them): half the logical ids under
    /// context, all of them otherwise.
    pub(super) fn local_pages(&self) -> usize {
        if self.context_mode.is_some() { self.pages / 2 } else { self.pages }
    }

    /// `rank`'s copy of a step's tables under context ([`rank_tables`]).
    pub(super) fn rank_tables(&self, tables: &StepTables, rank: usize) -> Result<StepTables> {
        rank_tables(tables, rank, self.pages)
    }

    /// Uploads a prefill unit's gather lists and commit blocks for `rank` (its stream is
    /// idle: called with the step's other tables). The unit gathers the pages holding rows
    /// `[0, gather)` (0: the step's first lane gathers for every lane) and commits rows
    /// `[start, start + rows)` of `pages` (the sequence's page ids).
    pub(super) fn context_prefill_tables(&self, rank: usize, lane: usize, gather: usize, start: usize, rows: usize,
        pages: &[u32]) -> Result<ContextUnit> {
        let split = self.context.as_ref().context("context split")?;
        let own = &split.ranks[rank];
        ensure!((start + rows).div_ceil(PAGE_ROWS) <= pages.len(), "prefill past the sequence's pages");
        // Gathered pages include a partial tail (a restore's copied tail page continues
        // inside the chunk's first page: its earlier rows are history).
        let history = gather.div_ceil(PAGE_ROWS);
        let half = split.staging_pages.div_ceil(2);
        let mut shards = [0; 2];
        for (owner, shard) in shards.iter_mut().enumerate() {
            let local = local_pages(&pages[..history], owner)?;
            *shard = local.len();
            if !local.is_empty() {
                self.on(rank, || self.library.copy_h2d(view(&own.gather, owner * half * 4, local.len() * 4)?,
                    bytes_of(&local)))?;
            }
        }
        let limit = context_commit_pages(self.prefill_rows as u64) as usize;
        let (table, commits) = commit_blocks(rank, start, rows, pages, limit)?;
        let commit = own.commit.get(lane).context("commit lane")?;
        self.on(rank, || self.library.copy_h2d(commit.buffer, bytes_of(&table)))?;
        Ok(ContextUnit { history, shards, commits })
    }

    /// Decode attention of layer `index` on `rank` under context (module docs); leaves the
    /// rank's head-split o_proj partial in `delta`, as [`Self::attention_on`] does.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn context_decode_attention(&self, rank: usize, w: &Workspace<'_>, index: usize,
        layer: &GlmLayer<'_>, rows: Scalar, t: usize, tables: &StepTables) -> Result<()> {
        use crate::shared::peer_split::attention::{context_decode, Candidates, ContextKernels, Partial, Query};
        struct Bindings<'e, 'a, 'w, 'l> {
            engine: &'e GlmEngine<'a>, w: &'e Workspace<'w>, layer: &'e GlmLayer<'l>,
            tables: &'e StepTables, rank: usize, index: usize, rows: Scalar, t: usize,
        }
        impl ContextKernels for Bindings<'_, '_, '_, '_> {
            fn produce(&mut self) -> Result<Query> {
                let e = self.engine;
                let (kv, _, _) = e.rank_caches(self.rank, self.index);
                e.producer(self.rank, self.w, self.layer, self.rows, "m64", kv.buffer, self.w.query.buffer.ptr)?;
                Ok(Query { values: self.w.query.buffer.ptr, row_pitch: e.cfg.heads / 2 * 576 * 2 })
            }
            fn score(&mut self) -> Result<Option<Candidates>> {
                let (e, w, rank) = (self.engine, self.w, self.rank);
                let (_, cache, _) = e.rank_caches(rank, self.index);
                let Some(cache) = cache else { return Ok(None) };
                let own = &e.context.as_ref().context("context split")?.ranks[rank];
                e.index_producer(rank, w, self.layer, self.rows, "m64", cache.buffer)?;
                let k = e.cfg.index_topk;
                let candidates = Candidates { scores: own.candidates.buffer.ptr,
                    indices: view(&own.candidates, DECODE_ROWS * k * 4, self.t * k * 4)?.ptr };
                let width = Scalar::I32(self.tables.table_stride as i32);
                e.run_on(rank, "glm_scored_index_topk_decode_m64", &[("q", w.q_fp8.buffer.ptr),
                    ("weights", w.head_weights.buffer.ptr), ("cache", cache.buffer.ptr),
                    ("table", w.page_table.buffer.ptr), ("lengths", w.cache_lengths.buffer.ptr),
                    ("indices", candidates.indices.cast_mut()), ("scores", candidates.scores.cast_mut()),
                    ("scratch", own.topk_scratch.buffer.ptr)],
                    &[self.rows, width, width, Scalar::I32(2), Scalar::I32(rank as i32)])?;
                Ok(Some(candidates))
            }
            fn merge(&mut self, scores: *mut c_void, indices: *mut c_void) -> Result<()> {
                let (e, w, rank) = (self.engine, self.w, self.rank);
                let own = &e.context.as_ref().context("context split")?.ranks[rank];
                let k = e.cfg.index_topk;
                e.run_on(rank, "glm_dsa_candidate_merge", &[("scores", scores), ("indices", indices),
                    ("out_scores", own.candidates.buffer.ptr),
                    ("out_indices", view(&own.candidates, DECODE_ROWS * k * 4, self.t * k * 4)?.ptr),
                    ("table", w.page_table.buffer.ptr), ("local_slots", w.indices.buffer.ptr),
                    ("local_lengths", w.lengths.buffer.ptr)],
                    &[self.rows, Scalar::I32(self.tables.table_stride as i32), Scalar::I32(rank as i32)])
            }
            fn partial(&mut self, query: *mut c_void, heads_rank: usize) -> Result<Partial> {
                let (e, w, rank) = (self.engine, self.w, self.rank);
                let split = e.context.as_ref().context("context split")?;
                let own = &split.ranks[rank];
                let out = if heads_rank == rank { &own.own } else { &own.peer };
                let heads = e.cfg.heads / 2;
                let lse = view(out, DECODE_ROWS * heads * 512 * 2, self.t * heads * 4)?.ptr;
                let (kv, _, _) = e.rank_caches(rank, self.index);
                e.run_on(rank, &split.partial[rank][heads_rank], &[("q", query), ("cache", kv.buffer.ptr),
                    ("indices", w.indices.buffer.ptr), ("lengths", w.lengths.buffer.ptr),
                    ("extra_cache", kv.buffer.ptr), ("extra_indices", w.indices.buffer.ptr),
                    ("extra_lengths", w.lengths.buffer.ptr), ("out", out.buffer.ptr), ("lse", lse),
                    ("scratch", own.partial_scratch.buffer.ptr)], &[self.rows])?;
                Ok(Partial { values: out.buffer.ptr, lse })
            }
            fn combine(&mut self, [first, second]: [Partial; 2]) -> Result<()> {
                let e = self.engine;
                let split = e.context.as_ref().context("context split")?;
                e.run_on(self.rank, "glm_lse_combine2", &[("o0", first.values), ("l0", first.lse),
                    ("o1", second.values), ("l1", second.lse), ("sink", split.sink[self.rank].buffer.ptr),
                    ("out", self.w.attn.buffer.ptr)], &[self.rows])
            }
        }
        let split = self.context.as_ref().context("context split")?;
        context_decode(&split.exchange, rank, index, 0, t,
            &mut Bindings { engine: self, w, layer, tables, rank, index, rows, t })?;
        self.o_proj(rank, w, layer, "m64", rows)
    }

    /// Prefill attention of layer `index` on `rank` under context (module docs): gather the
    /// history into the layer parity's staging view, the head-split programs over it, then
    /// this GPU's chunk rows into its pools. Leaves the o_proj partial in `delta`.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn context_prefill_attention(&self, rank: usize, w: &Workspace<'_>, index: usize,
        layer: &GlmLayer<'_>, rows: Scalar, cap: &str, tables: &StepTables, lane: usize, unit: ContextUnit)
        -> Result<()> {
        let split = self.context.as_ref().context("context split")?;
        let own = &split.ranks[rank];
        let staging = &own.staging[index % 2];
        let records = view(staging, 0, split.staging_pages * RECORD_PAGE_BYTES)?;
        let keys = view(staging, records.bytes, split.staging_pages * INDEX_PAGE_BYTES)?;
        let (kv, index_cache, _) = self.rank_caches(rank, index);
        let half = split.staging_pages.div_ceil(2);
        for (owner, &shard) in unit.shards.iter().enumerate().filter(|(_, &n)| n > 0) {
            let (pool, pool_keys, _) = self.rank_caches(owner, index);
            let table = view(&own.gather, owner * half * 4, shard * 4)?.ptr;
            let gather = |name: &str, pool: *mut c_void, staging: *mut c_void| self.run_on(rank, name,
                &[("pool", pool), ("table", table), ("staging", staging)],
                &[Scalar::I32((shard * PAGE_ROWS) as i32), Scalar::I32(2), Scalar::I32(owner as i32)]);
            gather("glm_paged_staging_gather_kv", pool.buffer.ptr, records.ptr)?;
            if let (Some(pool_keys), true) = (pool_keys, index_cache.is_some()) {
                gather("glm_paged_staging_gather_index", pool_keys.buffer.ptr, keys.ptr)?;
            }
        }
        self.attention_over(rank, w, index, layer, rows, cap, tables, records,
            index_cache.map(|_| keys))?;
        if unit.commits == 0 {
            return Ok(());
        }
        let commit = own.commit.get(lane).context("commit lane")?;
        let limit = context_commit_pages(self.prefill_rows as u64) as usize;
        let list = |kind: usize, part: usize| -> Result<*const c_void> {
            Ok(view(commit, ((kind * 3 + part) * limit) * 8, unit.commits * 8)?.ptr.cast_const())
        };
        let copy = |kind: usize, source: CuteafdDeviceBuffer, pool: &Dev<'_>| -> Result<()> {
            // SAFETY: staging and pool are this GPU's live buffers; each listed block lies inside
            // both (built from the pages in `context_prefill_tables`), and the lists stay unchanged
            // until the next step, which follows a drain of this stream.
            self.on(rank, || unsafe { self.library.cuda_kv_cache_write_blocks_async(source.ptr, pool.buffer.ptr,
                list(kind, 0)?, list(kind, 1)?, list(kind, 2)?, unit.commits, self.stream_of(rank)) })
        };
        copy(0, records, kv)?;
        if let Some(index_cache) = index_cache {
            copy(1, keys, index_cache)?;
            copy(2, keys, index_cache)?;
        }
        Ok(())
    }
}

/// `rank`'s copy of a step's tables under context, over `pages` logical page ids (an even
/// pool: GPU `rank` holds `pages / 2` local pages, then its scratch page). Rows on the
/// peer's pages, and padded rows (slots past the logical pool, `pad_rows`), write this
/// GPU's scratch row. Decode: each row's compact half table (its own pages' local indices,
/// padded with the scratch page) and its shard's row count (0 for padded rows). Prefill:
/// the staging view's identity table, logical slots.
fn rank_tables(tables: &StepTables, rank: usize, pages: usize) -> Result<StepTables> {
    let local_count = pages / 2;
    let scratch = (local_count * PAGE_ROWS) as i64;
    let mut out = StepTables { decode: tables.decode, positions: tables.positions.clone(), slots: Vec::new(),
        page_table: Vec::new(), table_width: tables.table_width, table_stride: tables.table_stride,
        cache_lengths: tables.cache_lengths.clone(), lengths: tables.lengths.clone(),
        exchange_rows: tables.exchange_rows };
    if !tables.decode {
        out.slots = tables.positions.clone();
        out.page_table = (0..tables.table_width as i32).collect();
        return Ok(out);
    }
    let half = tables.table_stride.div_ceil(2);
    let pool_slots = (pages * PAGE_ROWS) as i64;
    for row in 0..tables.positions.len() {
        let slot = tables.slots[row];
        let padded = slot >= pool_slots;
        let position = tables.positions[row] as usize;
        out.slots.push(if !padded && page_owner(position / PAGE_ROWS) == rank {
            ensure!((slot / PAGE_ROWS as i64) as usize % 2 == rank, "row {row}: slot {slot} on the wrong GPU");
            (slot / PAGE_ROWS as i64 >> 1) * PAGE_ROWS as i64 + slot % PAGE_ROWS as i64
        } else { scratch });
        let cached = tables.cache_lengths[row] as usize;
        let local = if padded { Vec::new() } else {
            let used = cached.div_ceil(PAGE_ROWS).min(tables.table_stride);
            let ids: Vec<u32> = tables.page_table[row * tables.table_stride..][..used].iter().map(|&p| p as u32).collect();
            local_pages(&ids, rank)?
        };
        out.page_table.extend(local.iter().map(|&p| p as i32));
        out.page_table.extend(std::iter::repeat_n(local_count as i32, half - local.len()));
        out.cache_lengths[row] = if padded { 0 } else { shard_rows(cached, PAGE_ROWS, rank) as i32 };
    }
    out.table_width = tables.table_width.div_ceil(2);
    out.table_stride = half;
    Ok(out)
}

/// The commit blocks of rows `[start, start + rows)` on `rank`'s pages, staging -> pool:
/// records, index keys, index scales, each as `[src | dst | bytes]` lists of `limit` u64
/// (zero padded); and the block count per kind.
fn commit_blocks(rank: usize, start: usize, rows: usize, pages: &[u32], limit: usize) -> Result<(Vec<u64>, usize)> {
    use crate::shared::peer_split::context::{commit_blocks, CommitPlane};
    let planes = [(RECORD_BYTES, 0, RECORD_PAGE_BYTES), (128, 0, INDEX_PAGE_BYTES),
        (4, INDEX_KEYS, INDEX_PAGE_BYTES)].map(|(row_bytes, base, unit_bytes)| CommitPlane {
            first: start, rows, rows_per_unit: PAGE_ROWS, row_bytes, base, unit_bytes,
        });
    let blocks = commit_blocks(rank, pages, &planes, limit)?;
    Ok((blocks.table, blocks.counts[0]))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A parity pool's ids for logical pages `0..n` (page `j` of parity `j % 2`).
    fn ids(n: usize, base: u32) -> Vec<u32> {
        (0..n as u32).map(|j| 2 * (base + j) + (j & 1)).collect()
    }

    #[test]
    fn partial_variants_follow_the_split_plan_per_sm_count_and_row_cap() {
        let manifest = serde_json::json!({"programs": [
            {"name": "glm_sparse_mla_partial_h0_s1_m64", "params": {"device_sm_counts": [170], "row_caps": {"170": [64]}}},
            {"name": "glm_sparse_mla_partial_h0_s4_m64", "params": {"device_sm_counts": [188], "row_caps": {"188": [64]}}},
            {"name": "glm_sparse_mla_partial_h0_s8_m64", "params": {"device_sm_counts": [170], "row_caps": {"170": [8]}}},
            {"name": "glm_sparse_mla_partial_h0_s11_m64", "params": {"device_sm_counts": [188], "row_caps": {"188": [8]}}},
            {"name": "glm_sparse_mla_partial_h0_s32_m64", "params": {"device_sm_counts": [170, 188],
                "row_caps": {"170": [1], "188": [1]}}},
            {"name": "glm_sparse_mla_partial_h32_s4_m64", "params": {"device_sm_counts": [188]}}]});
        use crate::shared::peer_split::context::partial_program;
        let plan = cuteafd_core::coordinator_programs::AttentionReductionPlan::new(DECODE_ROWS, Some(7), None).unwrap();
        assert_eq!(partial_program(&manifest, "glm", 0, DECODE_ROWS, 188, &plan).unwrap(),
            "glm_sparse_mla_partial_h0_s11_m64");
        assert_eq!(partial_program(&manifest, "glm", 0, DECODE_ROWS, 170, &plan).unwrap(),
            "glm_sparse_mla_partial_h0_s8_m64");
        // A legacy single variant serves the plan on either card.
        for sms in [170, 188] {
            assert_eq!(partial_program(&manifest, "glm", 32, DECODE_ROWS, sms, &plan).unwrap(),
                "glm_sparse_mla_partial_h32_s4_m64");
        }
        assert!(partial_program(&serde_json::json!({"programs": []}), "glm", 0, DECODE_ROWS, 188, &plan).is_err());
    }

    #[test]
    fn decode_tables_split_by_page_parity_and_pad_to_scratch() {
        let pages = 64; // 32 local pages per GPU, scratch at local 32
        let seq = ids(5, 3); // 300 rows: pages 0..4, tail of 44 rows on page 4 (GPU0)
        let width = 16;
        let mut tables = StepTables { decode: true, positions: vec![299, 300], slots: Vec::new(),
            page_table: Vec::new(), table_width: width, table_stride: width, cache_lengths: vec![300, 301],
            lengths: vec![300, 301], exchange_rows: 2 };
        for &p in &tables.positions.clone() {
            let page = seq[p as usize / PAGE_ROWS] as i64;
            tables.slots.push(page * PAGE_ROWS as i64 + p % PAGE_ROWS as i64);
            tables.page_table.extend(seq.iter().map(|&x| x as i32));
            tables.page_table.extend(std::iter::repeat_n(0, width - seq.len()));
        }
        tables.pad_rows(pages, 4);
        for rank in 0..2 {
            let r = rank_tables(&tables, rank, pages).unwrap();
            assert_eq!((r.table_width, r.table_stride), (8, 8));
            let scratch = 32 * PAGE_ROWS as i64;
            // Both rows sit on logical page 4 (GPU0).
            for row in 0..2 {
                let expected = if rank == 0 { (seq[4] as i64 >> 1) * 64 + tables.positions[row] % 64 } else { scratch };
                assert_eq!(r.slots[row], expected, "rank {rank} row {row}");
                let local: Vec<i32> = seq.iter().skip(rank).step_by(2).map(|&p| (p >> 1) as i32).collect();
                assert_eq!(&r.page_table[row * 8..][..local.len()], &local[..]);
                assert!(r.page_table[row * 8 + local.len()..(row + 1) * 8].iter().all(|&p| p == 32));
            }
            assert_eq!(r.cache_lengths[..2], [shard_rows(300, 64, rank) as i32, shard_rows(301, 64, rank) as i32]);
            assert_eq!(r.cache_lengths[0] + rank_tables(&tables, 1 - rank, pages).unwrap().cache_lengths[0], 300);
            // Padded rows: scratch slot, empty shard, all-scratch table.
            for row in 2..4 {
                assert_eq!((r.slots[row], r.cache_lengths[row]), (scratch, 0));
                assert!(r.page_table[row * 8..(row + 1) * 8].iter().all(|&p| p == 32));
            }
        }
        // A slot whose page id has the wrong parity for its logical page is refused.
        let mut wrong = tables.clone();
        wrong.slots[0] += PAGE_ROWS as i64;
        assert!(rank_tables(&wrong, 0, pages).is_err());
    }

    #[test]
    fn prefill_tables_are_the_staging_identity_and_commits_cover_owned_rows_once() {
        let seq = ids(9, 1);
        let tables = StepTables { decode: false, positions: (100..550).collect(), slots: vec![0; 450],
            page_table: seq.iter().map(|&p| p as i32).collect(), table_width: 9, table_stride: 0,
            cache_lengths: vec![0; 450], lengths: vec![0; 450], exchange_rows: 450 };
        let r = rank_tables(&tables, 1, 64).unwrap();
        assert_eq!(r.slots, tables.positions);
        assert_eq!(r.page_table, (0..9).collect::<Vec<_>>());
        let limit = context_commit_pages(512) as usize;
        let mut covered = vec![0u8; 550];
        for rank in 0..2 {
            let (table, commits) = commit_blocks(rank, 100, 450, &seq, limit).unwrap();
            assert_eq!(table.len(), 9 * limit);
            for i in 0..commits {
                let (src, dst, bytes) = (table[i] as usize, table[limit + i] as usize, table[2 * limit + i] as usize);
                assert_eq!(bytes % RECORD_BYTES, 0);
                let (page, first) = (src / RECORD_PAGE_BYTES, src % RECORD_PAGE_BYTES / RECORD_BYTES);
                assert_eq!(page_owner(page), rank);
                assert_eq!(dst, (seq[page] >> 1) as usize * RECORD_PAGE_BYTES + first * RECORD_BYTES);
                for row in page * 64 + first..page * 64 + first + bytes / RECORD_BYTES { covered[row] += 1; }
                // Keys and scales of the same rows.
                let (keys, scales) = (3 * limit, 6 * limit);
                assert_eq!(table[keys + 2 * limit + i] as usize, bytes / RECORD_BYTES * 128);
                assert_eq!(table[scales + i] as usize, page * INDEX_PAGE_BYTES + INDEX_KEYS + first * 4);
                assert_eq!(table[scales + limit + i] as usize, (seq[page] >> 1) as usize * INDEX_PAGE_BYTES
                    + INDEX_KEYS + first * 4);
            }
        }
        assert!(covered[..100].iter().all(|&c| c == 0) && covered[100..].iter().all(|&c| c == 1));
        // A 4,096-row lane at an odd page offset needs the full limit on one GPU.
        assert_eq!(commit_blocks(0, 32, 4096, &ids(66, 0), context_commit_pages(4096) as usize).unwrap().1, 33);
    }
}
