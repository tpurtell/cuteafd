//! GLM 5.3 Flash token-split (`context`) MLA attention (#57; PLAN.md "v3 attention placement
//! without latent replication", GLM Flash DSA). The 11 MLA layers' projections stay head-split
//! (each GPU its 32 heads' absorbed query, its share of W_UV + o_proj); every 256-token
//! allocation unit (four 64-row record pages and the pool-key page of the same index) lives once:
//! logical unit `u` of every sequence on GPU `u % 2`, at local unit `id >> 1` of its parity pool
//! id. The 34 KDA layers keep their head split and its partitioned state; the residual is
//! replicated after every layer as under `heads`. The DSA index cache is compact: each sequence's
//! open-pool tail (at most three key | gate rows) is per-sequence state, replicated on both GPUs,
//! so both build the same pooled key of any pool its owner then stores.
//!
//! **Decode/verify**, per MLA layer on both GPUs, in the order of the shared
//! [`crate::shared::peer_split::attention::context_decode`]: the producer writes each row's latent
//! record only on its unit's owner (elsewhere the row's slot is -1: skipped) and this GPU's heads'
//! absorbed query; push q. The compact index producer writes each completed pool's key on its
//! owner (else -1) and advances the replicated tails; the scored pool top-k over this GPU's units
//! (compact pool table, shard pool counts); push candidates, wait the peer's, merge into the
//! global top-512 by (score desc, logical asc), keeping this GPU's picks as local pool slots;
//! `glmf_index_expand_owned` turns them (or, below the dense limit, this GPU's own tokens) and the
//! open tail pool on its owner into local record slots. Wait q; the sparse MLA partial of the
//! peer's heads over the local selection, push it; this GPU's heads; wait the peer's partial of
//! them; `lse_combine2` in GPU0, GPU1 order into the head-split latent; `o` and the attention
//! all-reduce as under `heads`.
//!
//! **Prefill** gathers: per MLA layer each GPU copies the history's units (records and pool
//! keys, whole pages) into a contiguous staging view at their logical positions, its own locally
//! and the peer's through peer reads; the producers write the chunk's rows at their logical slots
//! there; the unchanged head-split programs read it through identity tables, so the selection,
//! its slot order and every record are those of `heads` (one GPU's, keys or compact: the same
//! pooled keys bit for bit); then the chunk's rows on this GPU's units go into its pools. Goldens
//! and exact prefix restores therefore do not depend on chunking or ownership.
//!
//! **Tables.** A step's per-rank tables are built on the host and uploaded once at the step's
//! start, after both streams drained, so no queued kernel reads a table the host rewrites.
use super::*;
use crate::shared::peer_split::context::{commit_blocks, local_pages, page_owner, partial_program,
    CommitPlane, ContextExchange};
use cuteafd_ffi::CuteafdDeviceBuffer;
use cuteafd_loader::placement::families::glm5_flash::{context_buffers, context_commit_units, context_workspace,
    GlmfAttention};

/// Pool-key page bytes: 64 x 128 E4M3 keys, then 64 FP32 scales.
pub(super) const POOL_PAGE_BYTES: usize = PAGE_ROWS * 132;
/// Record bytes of one unit (four 64-row pages).
pub(super) const UNIT_RECORD_BYTES: usize = UNIT_ROWS * RECORD_BYTES;

/// One GPU's context buffers, in the order and sizes `glm5_flash::context_workspace` gives the
/// solver (each with the engine's 256-byte floor), then staging.
pub(super) struct ContextRank<'a> {
    /// Scored top-k output: FP32 scores, then i32 logical pool indices, `[64, K]` each.
    candidates: Dev<'a>,
    /// The candidate merge's local pool slots `[64, K]`, then their counts `[64]`.
    local: Dev<'a>,
    /// This GPU's heads' and the peer's heads' partials: BF16 `[64, H/2, 512]`, then FP32 LSE.
    own: Dev<'a>,
    peer: Dev<'a>,
    topk_scratch: Dev<'a>,
    partial_scratch: Dev<'a>,
    /// A prefill's history as gather tables: per owner a record-page table over every logical
    /// page (its local page, -1 where the other GPU owns the unit), then per owner its local
    /// units in logical order (the pool-key pages, one per unit).
    gather: Dev<'a>,
    /// Per prefill lane: records, pool keys and pool scales, each as `[src | dst | bytes]` u64.
    commit: Vec<Dev<'a>>,
    /// The staging view's pool-page map: page `j` is logical page `j` (`glmf_index_expand`).
    identity: Dev<'a>,
    /// Two layer-parity views: `[units, 256 x 528]` records, then `[units, 64 x 132]` pool keys.
    staging: [Dev<'a>; 2],
}

/// Both GPUs' context buffers, the exchange and the per-GPU partial programs.
pub(super) struct GlmfContextSplit<'a> {
    ranks: [ContextRank<'a>; 2],
    /// Dropped before the head split's exchange (it borrows its abort words).
    pub exchange: ContextExchange<'a>,
    /// Logical units one staging view holds: the compiled extent's.
    staging_units: usize,
    /// Per GPU, one immutable startup recipe for heads `[0, H/2)` and `[H/2, H)`.
    partial: [[String; 2]; 2],
    /// `lse_combine2`'s sink operand (never read: GLM Flash has no sink).
    sink: [Dev<'a>; 2],
}

/// A view of `bytes` at `offset` inside `dev`.
fn view(dev: &Dev<'_>, offset: usize, bytes: usize) -> Result<CuteafdDeviceBuffer> {
    ensure!(offset + bytes <= dev.buffer.bytes, "view of {bytes} B at {offset} past a {} B buffer", dev.buffer.bytes);
    // SAFETY (pointer arithmetic only): the range lies inside the allocation (checked above).
    Ok(CuteafdDeviceBuffer { ptr: unsafe { dev.buffer.ptr.cast::<u8>().add(offset) }.cast(), bytes, ..dev.buffer })
}

impl<'a> GlmfContextSplit<'a> {
    /// Allocates both GPUs' context buffers and the exchange for `engine` (whose peer is attached)
    /// over `extent` tokens; `abort`: the head split's terminal abort words.
    pub fn new(engine: &GlmfEngine<'a>, context: GlmfAttention, extent: usize, manifest: &serde_json::Value,
        abort: Option<[*const u32; 2]>) -> Result<Self> {
        let cfg = &engine.cfg;
        let (rows, lanes) = (engine.prefill_rows as u64, engine.prefill_lane_count as u64);
        let buffers = context_buffers(cfg, extent as u64, rows, lanes, context);
        let ranks = [RankDevice { device: engine.device, stream: engine.stream_of(0) },
            RankDevice { device: engine.peer()?.device, stream: engine.stream_of(1) }];
        let exchange = ContextExchange::new(engine.library, ranks, &buffers, abort)?;
        ensure!(exchange.heads() == cfg.heads / 2 && exchange.topk() == cfg.index_topk / KPOOL, "context exchange geometry");
        let staging_units = extent.div_ceil(UNIT_ROWS);
        let list = context_workspace(cfg, extent as u64, rows, lanes, context);
        let rank = |rank: usize| -> Result<(ContextRank<'a>, Dev<'a>, [String; 2])> {
            engine.on(rank, || {
                let sms = engine.library.sm_count()?;
                let partial = [0, cfg.heads / 2].map(|begin| partial_program(
                    manifest, "glmf", begin, DECODE_ROWS, sms, &engine.attention_plan));
                let [a, b] = partial;
                let partial = [a?, b?];
                let _scope = cuteafd_ffi::memory_ledger::scope("context-workspace");
                let mut allocations = list.iter().map(|&(_, bytes)| engine.alloc(bytes as usize))
                    .collect::<Result<Vec<_>>>()?.into_iter();
                let mut next = || allocations.next().context("context workspace list");
                let (candidates, local, own, peer, topk_scratch, partial_scratch, gather) =
                    (next()?, next()?, next()?, next()?, next()?, next()?, next()?);
                let commit = (0..lanes).map(|_| next()).collect::<Result<Vec<_>>>()?;
                let identity = next()?;
                engine.library.cuda_zero_bytes(topk_scratch.buffer, topk_scratch.buffer.bytes)?;
                let map: Vec<i32> = (0..staging_units as i32).collect();
                engine.library.copy_h2d(view(&identity, 0, map.len() * 4)?, bytes_of(&map))?;
                let _scope = cuteafd_ffi::memory_ledger::scope("context-staging");
                let unit = UNIT_RECORD_BYTES + POOL_PAGE_BYTES;
                let staging = [engine.alloc(staging_units * unit)?, engine.alloc(staging_units * unit)?];
                // The sink operand is never read (no sink): a 256-byte floor allocation, uncharged.
                let sink = engine.alloc(0)?;
                Ok((ContextRank { candidates, local, own, peer, topk_scratch, partial_scratch, gather, commit, identity,
                    staging },
                    sink, partial))
            })
        };
        let (r0, s0, p0) = rank(0)?;
        let (r1, s1, p1) = rank(1)?;
        Ok(Self { ranks: [r0, r1], exchange, staging_units, partial: [p0, p1], sink: [s0, s1] })
    }

    /// Bytes this split allocated on each GPU beside the exchange (the solver's "context
    /// workspace" + "context staging").
    pub fn bytes(&self) -> [usize; 2] {
        self.ranks.each_ref().map(|r| [&r.candidates, &r.local, &r.own, &r.peer, &r.topk_scratch, &r.partial_scratch,
            &r.gather].into_iter().chain(&r.commit).chain([&r.identity]).chain(&r.staging).map(|d| d.buffer.bytes).sum())
    }
}

/// What a context prefill unit does per GPU at each MLA layer: the history units it gathers
/// (`history`, and `shards`: each owner's of them), and the commit block counts (records, pool
/// keys, pool scales).
#[derive(Debug, Clone, Copy, Default)]
pub(super) struct ContextUnit {
    pub history: usize,
    pub shards: [usize; 2],
    pub commits: [usize; 3],
}

/// Byte offsets in a rank's gather tables: `owner`'s record-page table, and its local-unit list.
fn gather_offsets(staging_units: usize, owner: usize) -> (usize, usize) {
    let pages = staging_units * UNIT_PAGES;
    (owner * pages * 4, (2 * pages + owner * staging_units.div_ceil(2)) * 4)
}

/// Per prefill lane (decode: lane 0) and rank: this step's context units.
pub(super) type ContextUnits = Vec<[ContextUnit; 2]>;

impl<'a> GlmfEngine<'a> {
    /// Local units per GPU: half the pool's under context, all of them otherwise.
    pub(crate) fn local_units(&self) -> usize {
        if self.context_mode.is_some() { self.pool_pages / 2 } else { self.pool_pages }
    }

    /// Whether the MLA caches are partitioned by unit (token-split attention).
    pub fn context_kv(&self) -> bool {
        self.context_mode.is_some()
    }

    /// Allocates the token-split buffers and exchange once the peer is attached; `extent`: the
    /// admitted compiled context (the staging the solver charged).
    pub fn attach_context(&mut self, extent: usize, manifest: &serde_json::Value) -> Result<()> {
        let context = self.context_mode.context("this engine was built without context attention")?;
        let exchange = self.exchange()?;
        let abort = match (exchange.abort_word(0)?, exchange.abort_word(1)?) {
            (Some(a), Some(b)) => Some([a, b]),
            _ => None,
        };
        let split = GlmfContextSplit::new(self, context, extent, manifest, abort)?;
        self.context = Some(split);
        Ok(())
    }

    /// `bytes` into `dst` on `rank`'s stream, after that stream's earlier consumers.
    fn put_at(&self, rank: usize, dst: CuteafdDeviceBuffer, bytes: &[u8]) -> Result<()> {
        // SAFETY: `dst` is a live allocation of `rank`'s GPU at least `bytes` long; the ordered copy
        // follows every queued reader on that rank's stream and precedes later ones.
        self.on(rank, || unsafe { self.library.copy_h2d_ordered(dst, bytes, self.stream_of(rank)) })
    }

    /// `rank`'s copy of a step's tables under context ([`rank_tables`]).
    pub(super) fn rank_tables(&self, tables: &StepTables, rank: usize) -> Result<StepTables> {
        rank_tables(tables, rank, self.pool_pages)
    }

    /// Uploads a prefill unit's gather tables and commit blocks for `rank` (its stream is idle:
    /// called with the step's other tables). The unit gathers the units holding rows `[0, gather)`
    /// (0: the step's first lane gathers for every lane) and commits rows `[start, start + rows)`
    /// of `units` (the sequence's unit ids).
    pub(super) fn context_prefill_tables(&self, rank: usize, lane: usize, gather: usize, start: usize, rows: usize,
        units: &[u32]) -> Result<ContextUnit> {
        let split = self.context.as_ref().context("context split")?;
        let own = &split.ranks[rank];
        ensure!((start + rows).div_ceil(UNIT_ROWS) <= units.len(), "prefill past the sequence's units");
        // Gathered units include a partial tail (a restore's copied tail unit continues inside the
        // chunk's first unit: its earlier rows are history).
        let history = gather.div_ceil(UNIT_ROWS);
        ensure!(history <= split.staging_units, "a history of {history} units past the {}-unit staging", split.staging_units);
        let mut shards = [0; 2];
        for (owner, shard) in shards.iter_mut().enumerate() {
            let local = local_pages(&units[..history], owner)?;
            *shard = local.len();
            if history == 0 { continue; }
            let (records, pools) = gather_offsets(split.staging_units, owner);
            let pages: Vec<i32> = (0..history * UNIT_PAGES).map(|page| {
                let unit = page / UNIT_PAGES;
                if page_owner(unit) == owner { ((units[unit] >> 1) as usize * UNIT_PAGES + page % UNIT_PAGES) as i32 }
                else { -1 }
            }).collect();
            self.put_at(rank, view(&own.gather, records, pages.len() * 4)?, bytes_of(&pages))?;
            if !local.is_empty() {
                let list: Vec<i32> = local.iter().map(|&l| l as i32).collect();
                self.put_at(rank, view(&own.gather, pools, list.len() * 4)?, bytes_of(&list))?;
            }
        }
        let limit = context_commit_units(self.prefill_rows as u64) as usize;
        let (table, commits) = commit_tables(rank, start, rows, units, limit)?;
        let commit = own.commit.get(lane).context("commit lane")?;
        self.put_at(rank, commit.buffer, bytes_of(&table))?;
        Ok(ContextUnit { history, shards, commits })
    }

    /// Decode attention of MLA layer `index` on `rank` under context (module docs); leaves the
    /// rank's head-split o_proj partial in `delta`, as [`Self::mla_on`] does.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn context_decode_attention(&self, rank: usize, w: &Workspace<'_>, index: usize, layer: &GlmfLayer<'_>,
        rows: Scalar, cap: &str, tables: &StepTables) -> Result<()> {
        use crate::shared::peer_split::attention::{context_decode, Candidates, ContextKernels, Partial, Query};
        struct Bindings<'e, 'a, 'w, 'l> {
            engine: &'e GlmfEngine<'a>, w: &'e Workspace<'w>, layer: &'e GlmfLayer<'l>,
            tables: &'e StepTables, rank: usize, index: usize, rows: Scalar, cap: &'e str, t: usize,
        }
        impl Bindings<'_, '_, '_, '_> {
            /// This GPU's selection (`glmf_index_expand_owned`): its own earlier tokens below the
            /// dense limit, else its merged pool picks (`local_lengths`: their counts) and the open
            /// tail pool on its owner. A short step passes no picks (no row reads them).
            fn expand(&self, local_lengths: *mut c_void) -> Result<()> {
                let (e, w, rank) = (self.engine, self.w, self.rank);
                let own = &e.context.as_ref().context("context split")?.ranks[rank];
                let k = e.cfg.index_topk / KPOOL;
                let local_lengths = if local_lengths.is_null() { view(&own.local, DECODE_ROWS * k * 4, self.t * 4)?.ptr }
                    else { local_lengths };
                e.run_on(rank, false, "index_expand_owned", &[("positions", w.positions.buffer.ptr),
                    ("local_pools", own.local.buffer.ptr), ("local_lengths", local_lengths),
                    ("units", w.pool_table.buffer.ptr), ("indices", w.indices.buffer.ptr),
                    ("lengths", w.lengths.buffer.ptr)],
                    &[self.rows, Scalar::I32(self.tables.context_stride as i32), Scalar::I32(rank as i32)])
            }
        }
        impl ContextKernels for Bindings<'_, '_, '_, '_> {
            fn produce(&mut self) -> Result<Query> {
                let (e, w) = (self.engine, self.w);
                e.mla_producer(self.rank, w, self.index, self.layer, self.rows, self.cap, true)?;
                Ok(Query { values: w.query.buffer.ptr, row_pitch: e.cfg.heads / 2 * e.cfg.kv_lora_rank * 2 })
            }
            fn score(&mut self) -> Result<Option<Candidates>> {
                let (e, w, rank) = (self.engine, self.w, self.rank);
                e.index_producer(rank, w, self.index, self.layer, self.rows, self.cap, self.tables)?;
                if !self.tables.long {
                    // Every row within the dense context (both GPUs see the same step): no top-k, no
                    // candidate exchange, as the head split skips its pool top-k; each GPU selects
                    // its own earlier tokens. Graphs of short steps key no pool geometry.
                    self.expand(std::ptr::null_mut())?;
                    return Ok(None);
                }
                let own = &e.context.as_ref().context("context split")?.ranks[rank];
                let k = e.cfg.index_topk / KPOOL;
                let candidates = Candidates { scores: own.candidates.buffer.ptr,
                    indices: view(&own.candidates, DECODE_ROWS * k * 4, self.t * k * 4)?.ptr };
                let caches = e.caches_of(rank);
                let pools = &caches.index[self.index].as_ref().context("MLA layer without an index cache")?.pools;
                let width = Scalar::I32(self.tables.context_stride as i32);
                e.run_on(rank, false, "scored_index_topk_decode_m64", &[("q", w.q_fp8.buffer.ptr),
                    ("weights", w.head_weights.buffer.ptr), ("cache", pools.buffer.ptr),
                    ("table", w.pool_table.buffer.ptr), ("lengths", w.cache_lengths.buffer.ptr),
                    ("indices", candidates.indices.cast_mut()), ("scores", candidates.scores.cast_mut()),
                    ("scratch", own.topk_scratch.buffer.ptr)],
                    &[self.rows, Scalar::I32(self.tables.context_width as i32), width, Scalar::I32(2),
                        Scalar::I32(rank as i32)])?;
                Ok(Some(candidates))
            }
            fn merge(&mut self, scores: *mut c_void, indices: *mut c_void) -> Result<()> {
                let (e, w, rank) = (self.engine, self.w, self.rank);
                let own = &e.context.as_ref().context("context split")?.ranks[rank];
                let k = e.cfg.index_topk / KPOOL;
                let local_lengths = view(&own.local, DECODE_ROWS * k * 4, self.t * 4)?.ptr;
                e.run_on(rank, false, "dsa_candidate_merge", &[("scores", scores), ("indices", indices),
                    ("out_scores", own.candidates.buffer.ptr),
                    ("out_indices", view(&own.candidates, DECODE_ROWS * k * 4, self.t * k * 4)?.ptr),
                    ("table", w.pool_table.buffer.ptr), ("local_slots", own.local.buffer.ptr),
                    ("local_lengths", local_lengths)],
                    &[self.rows, Scalar::I32(self.tables.context_stride as i32), Scalar::I32(rank as i32)])?;
                self.expand(local_lengths)
            }
            fn partial(&mut self, query: *mut c_void, heads_rank: usize) -> Result<Partial> {
                let (e, w, rank) = (self.engine, self.w, self.rank);
                let split = e.context.as_ref().context("context split")?;
                let own = &split.ranks[rank];
                let out = if heads_rank == rank { &own.own } else { &own.peer };
                let heads = e.cfg.heads / 2;
                let lse = view(out, DECODE_ROWS * heads * 512 * 2, self.t * heads * 4)?.ptr;
                let kv = e.caches_of(rank).kv[self.index].as_ref().context("MLA layer without a record pool")?.buffer.ptr;
                let name = split.partial[rank][heads_rank].strip_prefix("glmf_").context("partial program name")?;
                e.run_on(rank, false, name, &[("q", query), ("cache", kv),
                    ("indices", w.indices.buffer.ptr), ("lengths", w.lengths.buffer.ptr),
                    ("extra_cache", kv), ("extra_indices", w.indices.buffer.ptr),
                    ("extra_lengths", w.lengths.buffer.ptr), ("out", out.buffer.ptr), ("lse", lse),
                    ("scratch", own.partial_scratch.buffer.ptr)], &[self.rows])?;
                Ok(Partial { values: out.buffer.ptr, lse })
            }
            fn combine(&mut self, [first, second]: [Partial; 2]) -> Result<()> {
                let e = self.engine;
                let split = e.context.as_ref().context("context split")?;
                e.run_on(self.rank, false, "lse_combine2", &[("o0", first.values), ("l0", first.lse),
                    ("o1", second.values), ("l1", second.lse), ("sink", split.sink[self.rank].buffer.ptr),
                    ("out", self.w.latent.buffer.ptr)], &[self.rows])
            }
        }
        let t = tables.positions.len();
        ensure!(t <= DECODE_ROWS, "a context decode step of {t} rows (the exchange holds {DECODE_ROWS})");
        let ordinal = self.caches_of(rank).mla_ordinal[index].context("MLA layer without an ordinal")?;
        let split = self.context.as_ref().context("context split")?;
        context_decode(&split.exchange, rank, ordinal, 0, t,
            &mut Bindings { engine: self, w, layer, tables, rank, index, rows, cap, t })?;
        self.mla_output(rank, w, layer, cap, rows, true)
    }

    /// Prefill attention of MLA layer `index` on `rank` under context (module docs): gather the
    /// history into the layer parity's staging view, the head-split programs over it through
    /// identity tables, then this GPU's chunk rows into its pools. Leaves the o_proj partial in
    /// `delta`.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn context_prefill_attention(&self, rank: usize, w: &Workspace<'_>, index: usize,
        layer: &GlmfLayer<'_>, rows: Scalar, cap: &str, tables: &StepTables, lane: usize, unit: ContextUnit)
        -> Result<()> {
        let split = self.context.as_ref().context("context split")?;
        let own = &split.ranks[rank];
        let ordinal = self.caches_of(rank).mla_ordinal[index].context("MLA layer without an ordinal")?;
        let staging = &own.staging[ordinal % 2];
        let records = view(staging, 0, split.staging_units * UNIT_RECORD_BYTES)?;
        let pools = view(staging, records.bytes, split.staging_units * POOL_PAGE_BYTES)?;
        for (owner, &shard) in unit.shards.iter().enumerate().filter(|(_, &n)| n > 0) {
            let caches = self.caches_of(owner);
            let kv = caches.kv[index].as_ref().context("MLA layer without a record pool")?;
            let pool_keys = &caches.index[index].as_ref().context("MLA layer without an index cache")?.pools;
            let (record_table, pool_table) = gather_offsets(split.staging_units, owner);
            // Records: every history page, the peer's masked (-1); pool keys: this owner's units, unit
            // column j to logical unit 2j + owner. Whole pages (a partial tail unit's later rows are
            // overwritten by the chunk's producer or never read).
            self.run_on(rank, false, "paged_staging_gather_kv", &[("pool", kv.buffer.ptr),
                ("table", view(&own.gather, record_table, unit.history * UNIT_PAGES * 4)?.ptr), ("staging", records.ptr)],
                &[Scalar::I32((unit.history * UNIT_ROWS) as i32), Scalar::I32(1), Scalar::I32(0)])?;
            self.run_on(rank, false, "paged_staging_gather_index", &[("pool", pool_keys.buffer.ptr),
                ("table", view(&own.gather, pool_table, shard * 4)?.ptr), ("staging", pools.ptr)],
                &[Scalar::I32((shard * PAGE_ROWS) as i32), Scalar::I32(2), Scalar::I32(owner as i32)])?;
        }
        self.mla_over(rank, w, index, layer, rows, cap, tables, records, pools, own.identity.buffer.ptr, None)?;
        if unit.commits.iter().all(|&n| n == 0) {
            return Ok(());
        }
        let commit = own.commit.get(lane).context("commit lane")?;
        let limit = context_commit_units(self.prefill_rows as u64) as usize;
        let list = |kind: usize, part: usize| -> Result<*const c_void> {
            Ok(view(commit, (kind * 3 + part) * limit * 8, unit.commits[kind].max(1) * 8)?.ptr.cast_const())
        };
        let caches = self.caches_of(rank);
        let kv = caches.kv[index].as_ref().context("MLA layer without a record pool")?;
        let pool_keys = &caches.index[index].as_ref().context("MLA layer without an index cache")?.pools;
        let copy = |kind: usize, source: CuteafdDeviceBuffer, pool: &Dev<'_>| -> Result<()> {
            if unit.commits[kind] == 0 { return Ok(()); }
            // SAFETY: staging and pool are this GPU's live buffers; each listed block lies inside
            // both (built from the units in `context_prefill_tables`), and the lists stay unchanged
            // until the next step, which follows a drain of this stream.
            self.on(rank, || unsafe { self.library.cuda_kv_cache_write_blocks_async(source.ptr, pool.buffer.ptr,
                list(kind, 0)?, list(kind, 1)?, list(kind, 2)?, unit.commits[kind], self.stream_of(rank)) })
        };
        copy(0, records, kv)?;
        copy(1, pools, pool_keys)?;
        copy(2, pools, pool_keys)
    }
}

/// `rank`'s copy of a step's tables under context, over `pool_pages` logical unit ids (an even
/// pool: GPU `rank` holds `pool_pages / 2` local units). Rows whose unit is the peer's, and
/// padded rows, get -1 record and pool slots (the producers skip them). Decode: each row's compact
/// unit table (its own units' local indices, padded with 0) in the pool table, and its shard's
/// complete pools in the cache lengths. Prefill: the staging view's identity tables and logical
/// slots, as one GPU over the gathered history.
pub(super) fn rank_tables(tables: &StepTables, rank: usize, pool_pages: usize) -> Result<StepTables> {
    ensure!(tables.segments.is_empty(), "a packed prefill runs on one GPU");
    let mut out = StepTables { decode: tables.decode, eager: tables.eager, real_rows: tables.real_rows,
        positions: tables.positions.clone(), kda_slots: tables.kda_slots.clone(), seq_first: tables.seq_first.clone(),
        long: tables.long, spec: tables.spec, pool_width: tables.pool_width, ..Default::default() };
    let unit_of_slot = |slot: i64, per_unit: i64| (slot / per_unit) as usize;
    if !tables.decode {
        // Identity: logical record slot = position, logical pool slot = position / 4.
        out.kv_slots = tables.positions.clone();
        out.pool_slots = tables.positions.iter().zip(&tables.pool_slots)
            .map(|(&p, &s)| if s < 0 { -1 } else { p / KPOOL as i64 }).collect();
        out.cache_lengths = tables.cache_lengths.clone();
        let units = tables.pool_table.len();
        out.page_table = (0..(units * UNIT_PAGES) as i32).collect();
        out.pool_table = (0..units as i32).collect();
        out.page_stride = tables.page_stride;
        out.pool_stride = tables.pool_stride;
        return Ok(out);
    }
    // A short step's startup graph geometry keys no pool stride (0: masked warm-up rows only).
    let stride = tables.pool_stride;
    ensure!(tables.pool_table.len() == stride * tables.positions.len(), "decode pool table geometry");
    // The compact unit table's stride follows the record-page stride, which every decode graph
    // keys (the pool stride only long steps key): units a row's pages span, this GPU's half.
    let half = (tables.page_stride / UNIT_PAGES).div_ceil(2);
    for row in 0..tables.positions.len() {
        let position = tables.positions[row];
        let masked = position < 0;
        let mine = !masked && page_owner(position as usize / UNIT_ROWS) == rank;
        let kv = tables.kv_slots[row];
        out.kv_slots.push(if mine {
            let id = unit_of_slot(kv, UNIT_ROWS as i64);
            ensure!(id < pool_pages && id % 2 == rank, "row {row}: record slot {kv} on the wrong GPU");
            ((id >> 1) * UNIT_ROWS) as i64 + kv % UNIT_ROWS as i64
        } else { -1 });
        let pool = tables.pool_slots[row];
        out.pool_slots.push(if mine && pool >= 0 {
            let id = unit_of_slot(pool, PAGE_ROWS as i64);
            ((id >> 1) * PAGE_ROWS) as i64 + pool % PAGE_ROWS as i64
        } else { -1 });
        let pools = if masked { 0 } else { tables.cache_lengths[row].max(0) as usize };
        let local = if masked { Vec::new() } else {
            let used = (position as usize + 1).div_ceil(UNIT_ROWS);
            ensure!(stride > 0, "row {row}: a real decode row without a pool table");
            ensure!(used <= stride && used <= 2 * half, "row {row}: {used} units past the decode table strides");
            let ids: Vec<u32> = tables.pool_table[row * stride..][..used].iter().map(|&p| p as u32).collect();
            local_pages(&ids, rank)?
        };
        ensure!(local.len() <= half, "row {row}: {} local units past a half table of {half}", local.len());
        out.pool_table.extend(local.iter().map(|&p| p as i32));
        out.pool_table.extend(std::iter::repeat_n(0, half - local.len()));
        out.cache_lengths.push(shard_pools(pools, rank) as i32);
    }
    // Graph keys read the logical geometry; the context kernels the compact one.
    out.pool_stride = tables.pool_stride;
    out.pool_width = tables.pool_width;
    out.context_stride = half;
    out.context_width = tables.pool_width.div_ceil(2).max(1);
    // The record pages are never read under context decode (the owned expansion reads units).
    out.page_stride = tables.page_stride;
    out.page_table = tables.page_table.clone();
    Ok(out)
}

/// Complete pools of a row's first `pools` pools that live on `rank`'s units (64 pools a unit).
fn shard_pools(pools: usize, rank: usize) -> usize {
    crate::shared::peer_split::context::shard_rows(pools, PAGE_ROWS, rank)
}

/// The commit blocks of rows `[start, start + rows)` on `rank`'s units, staging -> pool: records,
/// pool keys, pool scales, each as `[src | dst | bytes]` lists of `limit` u64 (zero padded); and
/// the block count per kind. Pool keys commit the pools the rows complete.
fn commit_tables(rank: usize, start: usize, rows: usize, units: &[u32], limit: usize) -> Result<(Vec<u64>, [usize; 3])> {
    let (first_pool, end_pool) = (start / KPOOL, (start + rows) / KPOOL);
    let planes = [CommitPlane { first: start, rows, rows_per_unit: UNIT_ROWS, row_bytes: RECORD_BYTES, base: 0,
            unit_bytes: UNIT_RECORD_BYTES },
        CommitPlane { first: first_pool, rows: end_pool - first_pool, rows_per_unit: PAGE_ROWS, row_bytes: 128, base: 0,
            unit_bytes: POOL_PAGE_BYTES },
        CommitPlane { first: first_pool, rows: end_pool - first_pool, rows_per_unit: PAGE_ROWS, row_bytes: 4,
            base: PAGE_ROWS * 128, unit_bytes: POOL_PAGE_BYTES }];
    let blocks = commit_blocks(rank, units, &planes, limit)?;
    Ok((blocks.table, [blocks.counts[0], blocks.counts[1], blocks.counts[2]]))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A parity pool's ids for logical units `0..n` (unit `j` of parity `j % 2`).
    fn ids(n: usize, base: u32) -> Vec<u32> {
        (0..n as u32).map(|j| 2 * (base + j) + (j & 1)).collect()
    }

    fn decode_tables(seq: &[u32], positions: &[i64], stride: usize) -> StepTables {
        let mut tables = StepTables { decode: true, pool_stride: stride, page_stride: stride * UNIT_PAGES,
            pool_width: stride, ..Default::default() };
        for &p in positions {
            let u = p as usize / UNIT_ROWS;
            tables.positions.push(p);
            tables.kv_slots.push(seq[u] as i64 * UNIT_ROWS as i64 + p % UNIT_ROWS as i64);
            tables.pool_slots.push(if p as usize % KPOOL == KPOOL - 1 {
                seq[u] as i64 * PAGE_ROWS as i64 + (p as usize / KPOOL % PAGE_ROWS) as i64 } else { -1 });
            tables.cache_lengths.push(((p + 1) / KPOOL as i64) as i32);
            tables.kda_slots.push(0);
            tables.seq_first.push(0);
            let mut row: Vec<i32> = seq.iter().map(|&x| x as i32).collect();
            row.resize(stride, 0);
            tables.pool_table.extend(&row);
            tables.page_table.extend(row.iter().flat_map(|&u| (0..UNIT_PAGES as i32).map(move |i| u * 4 + i)));
        }
        tables
    }

    #[test]
    fn decode_tables_split_by_unit_parity_and_mask_the_peers_rows() {
        let seq = ids(5, 3); // 1,200 rows: units 0..4, the open tail on unit 4 (GPU0)
        let tables = decode_tables(&seq, &[1199, 1100, 900], 8);
        let mut padded = tables.clone();
        for _ in 0..2 {
            let row = crate::shared::decode_graph::masked_row(padded.positions.len());
            padded.positions.push(row.position);
            padded.kv_slots.push(row.kv_slot);
            padded.pool_slots.push(row.pool_slot);
            padded.cache_lengths.push(row.cache_length);
            padded.kda_slots.push(row.state_slot);
            padded.seq_first.push(row.seq_first);
            padded.pool_table.extend([0; 8]);
            padded.page_table.extend([0; 32]);
        }
        for rank in 0..2 {
            let r = rank_tables(&padded, rank, 64).unwrap();
            assert_eq!((r.pool_stride, r.context_stride), (8, 4));
            // Rows 0, 1 on unit 4 (GPU0), row 2 on unit 3 (GPU1).
            for (row, owner) in [(0, 0), (1, 0), (2, 1)] {
                let p = padded.positions[row];
                let want = if owner == rank { (seq[p as usize / UNIT_ROWS] as i64 >> 1) * UNIT_ROWS as i64 + p % 256 }
                    else { -1 };
                assert_eq!(r.kv_slots[row], want, "rank {rank} row {row}");
                // The row's own units (up to its position's), this GPU's, then zero padding.
                let used = p as usize / UNIT_ROWS + 1;
                let local: Vec<i32> = seq[..used].iter().skip(rank).step_by(2).map(|&u| (u >> 1) as i32).collect();
                assert_eq!(&r.pool_table[row * 4..][..local.len()], &local[..]);
                assert!(r.pool_table[row * 4 + local.len()..(row + 1) * 4].iter().all(|&u| u == 0));
            }
            // Row 0 (position 1199) completes pool 299 on unit 4 (GPU0).
            assert_eq!(r.pool_slots[0], if rank == 0 { (seq[4] as i64 >> 1) * 64 + 299 % 64 } else { -1 });
            assert_eq!(r.cache_lengths[0], shard_pools(300, rank) as i32);
            for row in 3..5 {
                assert_eq!((r.kv_slots[row], r.pool_slots[row], r.cache_lengths[row]), (-1, -1, 0));
            }
        }
        assert_eq!(rank_tables(&padded, 0, 64).unwrap().cache_lengths[0]
            + rank_tables(&padded, 1, 64).unwrap().cache_lengths[0], 300);
        let mut wrong = tables.clone();
        wrong.kv_slots[0] += UNIT_ROWS as i64;
        assert!(rank_tables(&wrong, 0, 64).is_err(), "a slot of the wrong parity is refused");
    }

    #[test]
    fn masked_startup_rows_need_no_pool_table() {
        // A short startup geometry keys no pool stride: every row masked, no pool columns.
        let mut tables = StepTables { decode: true, page_stride: 64, ..Default::default() };
        for row in 0..4 {
            let r = crate::shared::decode_graph::masked_row(row);
            tables.positions.push(r.position);
            tables.kv_slots.push(r.kv_slot);
            tables.pool_slots.push(r.pool_slot);
            tables.cache_lengths.push(r.cache_length);
            tables.kda_slots.push(r.state_slot);
            tables.seq_first.push(r.seq_first);
            tables.page_table.extend([0; 64]);
        }
        for rank in 0..2 {
            let r = rank_tables(&tables, rank, 64).unwrap();
            assert_eq!(r.context_stride, 8);
            assert_eq!(r.pool_table, vec![0; 4 * 8]);
            assert!(r.kv_slots.iter().chain(&r.pool_slots).all(|&s| s == -1));
            assert!(r.cache_lengths.iter().all(|&n| n == 0));
        }
    }

    #[test]
    fn prefill_tables_are_the_staging_identity_and_commits_cover_owned_rows_once() {
        let seq = ids(9, 1);
        let tables = StepTables { positions: (300..1500).collect(), kv_slots: vec![0; 1200],
            pool_slots: (300..1500).map(|p: i64| if p % 4 == 3 { 0 } else { -1 }).collect(),
            pool_table: seq.iter().map(|&u| u as i32).collect(), cache_lengths: vec![0; 1200],
            kda_slots: vec![0; 1200], seq_first: vec![0; 1200], ..Default::default() };
        let r = rank_tables(&tables, 1, 64).unwrap();
        assert_eq!(r.kv_slots, tables.positions);
        assert_eq!(r.pool_slots.iter().filter(|&&s| s >= 0).count(), 300);
        assert_eq!(r.pool_table, (0..9).collect::<Vec<_>>());
        assert_eq!(r.page_table, (0..36).collect::<Vec<_>>());
        let limit = context_commit_units(1200) as usize;
        let mut covered = vec![0u8; 1500];
        let mut pools = vec![0u8; 375];
        for rank in 0..2 {
            let (table, commits) = commit_tables(rank, 300, 1200, &seq, limit).unwrap();
            assert_eq!(table.len(), 9 * limit);
            for i in 0..commits[0] {
                let (src, dst, bytes) = (table[i] as usize, table[limit + i] as usize, table[2 * limit + i] as usize);
                let (unit, first) = (src / UNIT_RECORD_BYTES, src % UNIT_RECORD_BYTES / RECORD_BYTES);
                assert_eq!(page_owner(unit), rank);
                assert_eq!(dst, (seq[unit] >> 1) as usize * UNIT_RECORD_BYTES + first * RECORD_BYTES);
                for row in unit * UNIT_ROWS + first..unit * UNIT_ROWS + first + bytes / RECORD_BYTES { covered[row] += 1; }
            }
            for i in 0..commits[1] {
                let (src, bytes) = (table[3 * limit + i] as usize, table[5 * limit + i] as usize);
                let (unit, first) = (src / POOL_PAGE_BYTES, src % POOL_PAGE_BYTES / 128);
                assert_eq!(page_owner(unit), rank);
                for pool in unit * 64 + first..unit * 64 + first + bytes / 128 { pools[pool] += 1; }
                assert_eq!(table[6 * limit + i] as usize, unit * POOL_PAGE_BYTES + 64 * 128 + first * 4);
            }
        }
        assert!(covered[..300].iter().all(|&c| c == 0) && covered[300..].iter().all(|&c| c == 1));
        assert!(pools[..75].iter().all(|&c| c == 0) && pools[75..].iter().all(|&c| c == 1));
    }
}
