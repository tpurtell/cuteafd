//! KV reservation by decode chunks for every serve loop (issue #951): the launch options and
//! the per-round glue over [`cuteafd_engine::reserve`].
//!
//! A scheduler keeps its decoding requests in two lists: `active` (in the next round) and
//! `stalled` (waiting for pages; they keep their slot and pages). Before each round it calls
//! [`Reserve::round`], which plans top-ups for both lists together, grows what was granted
//! through the family's [`PrefixCache::extend`], moves requests between the lists, and returns
//! those that end early with `length` for the scheduler to finish and release. A stalled request
//! never enters a round, so graph shapes are those of a smaller round: no capture.
use anyhow::Result;
use cuteafd_engine::prefix::{PrefixCache, PrefixFamily};
use cuteafd_engine::reserve::{Demand, KvReserve, Ledger, Lease, ReserveConfig, ReserveStats, Room};
use cuteafd_hostcache::copy::CopyEngine;

/// The reservation options, the same in every family.
#[derive(Debug, Clone, Copy, clap::Args)]
pub(crate) struct ReserveArgs {
    /// Most output tokens a request may ask for (and the default when it omits max_tokens),
    /// further bounded by the context left after its prompt; 0 or less: no cap but the context.
    #[arg(long, env = "CUTEAFD_MAX_OUTPUT_TOKENS",
        default_value_t = cuteafd_api::openai::DEFAULT_MAX_OUTPUT_TOKENS, allow_negative_numbers = true)]
    pub max_output_tokens: i64,
    /// KV tokens a request reserves past its prompt at admission and adds each time decode
    /// reaches its reservation (rounded down to a page; a remaining tail under half a chunk
    /// is reserved whole).
    #[arg(long, env = "CUTEAFD_KV_DECODE_RESERVE_CHUNK", default_value_t = cuteafd_engine::reserve::DEFAULT_CHUNK)]
    pub kv_decode_reserve_chunk: usize,
    /// Output tokens the drain rule guarantees each request under KV pressure. 0: its whole
    /// remaining maximum output (requests stall, never end early). Nonzero (below the maximum
    /// output): the request kept running may end with `length` at this many tokens.
    #[arg(long, env = "CUTEAFD_KV_RESERVE_DRAIN_BUFFER", default_value_t = 0)]
    pub kv_reserve_drain_buffer: usize,
    /// A request that would stall after producing at least this many tokens ends with
    /// `length` instead (-1: off; at most the drain buffer; a nonzero drain buffer applies
    /// itself when this is off).
    #[arg(long, env = "CUTEAFD_KV_EARLY_TERMINATION_MIN_OUTPUT_TOKENS", default_value_t = -1,
        allow_negative_numbers = true)]
    pub kv_early_termination_min_output_tokens: i64,
}

impl Default for ReserveArgs {
    fn default() -> Self {
        Self { max_output_tokens: cuteafd_api::openai::DEFAULT_MAX_OUTPUT_TOKENS,
            kv_decode_reserve_chunk: cuteafd_engine::reserve::DEFAULT_CHUNK, kv_reserve_drain_buffer: 0,
            kv_early_termination_min_output_tokens: -1 }
    }
}

impl ReserveArgs {
    /// The served limits for a context extent.
    pub fn limits(&self, context: usize) -> Result<cuteafd_api::openai::NativeLimits> {
        cuteafd_api::openai::NativeLimits::new(u32::try_from(context)?, self.max_output_tokens)
    }

    /// The policy options against the effective maximum output for `context`; refuses (at
    /// startup) a drain buffer at or above it or below an explicit early-termination value.
    pub fn config(&self, context: usize) -> Result<ReserveConfig> {
        let max_output = self.limits(context)?.output() as usize;
        Ok(ReserveConfig::new(self.kv_decode_reserve_chunk, self.kv_reserve_drain_buffer,
            self.kv_early_termination_min_output_tokens, max_output)?)
    }
}

/// Draft rows a request may verify past its next token within its reserved rows (`len` rows
/// committed, `pages` pages of `page_rows` reserved): the next token's row and its drafts.
pub(crate) fn reserved_drafts(len: usize, pages: usize, page_rows: usize) -> usize {
    (pages * page_rows).saturating_sub(len + 1)
}

/// What a decoding request exposes to the reservation.
pub(crate) trait Reserving {
    type Placement;
    fn lease(&self) -> &Lease;
    /// Rows already in its KV.
    fn committed(&self) -> usize;
    /// Output tokens produced.
    fn produced(&self) -> usize;
    fn placement(&self) -> &Self::Placement;
    fn placement_mut(&mut self) -> &mut Self::Placement;
    /// The client went away (a stalled request never reaches the round that would notice).
    fn closed(&self) -> bool;
}

/// A prefilling request: it holds its admission reservation and is not a top-up candidate.
pub(crate) struct Holding<'p> {
    pub lease: Lease,
    pub pages: &'p [u32],
}

pub(crate) struct Reserve {
    policy: KvReserve,
}

/// One serve loop's round over the shared policy ([`KvReserve::round`]): decoding requests are
/// `active` (running) then `stalled`, holders (prefilling) follow; an ended request is handed to
/// `end`, which finishes it with `length` and releases its pages before anything is granted.
struct Glue<'c, 'f, 'h, 'p, E: CopyEngine, M: Clone + Default, F: PrefixFamily<M>, A: Reserving<Placement = F::Placement>,
    End: FnMut(&mut PrefixCache<E, M>, A)> {
    cache: &'c mut PrefixCache<E, M>,
    family: &'f F,
    decoding: Vec<(A, bool)>,
    holding: &'h [Holding<'p>],
    end: End,
    error: Option<anyhow::Error>,
}

impl<E: CopyEngine, M: Clone + Default, F: PrefixFamily<M>, A: Reserving<Placement = F::Placement>,
    End: FnMut(&mut PrefixCache<E, M>, A)> Ledger for Glue<'_, '_, '_, '_, E, M, F, A, End> {
    fn room(&self) -> Room { self.cache.room(self.family) }
    fn settled(&self) -> bool { self.cache.settled() }
    fn demands(&self) -> (Vec<Demand<'_>>, Vec<bool>) {
        let decoding = self.decoding.iter().map(|(a, _)| {
            let pages = self.family.pages(a.placement());
            Demand { lease: *a.lease(), held: pages.len(), pages, committed: a.committed(), produced: a.produced(),
                decoding: true, draft: self.family.draft_demand(a.placement()) }
        });
        let holding = self.holding.iter().map(|h| Demand { lease: h.lease, held: h.pages.len(), pages: h.pages,
            committed: h.lease.prompt(), produced: 0, decoding: false, draft: None });
        (decoding.chain(holding).collect(), self.decoding.iter().map(|(_, stalled)| *stalled).collect())
    }
    fn end(&mut self, index: usize) {
        let (request, _) = self.decoding.remove(index);
        (self.end)(self.cache, request);
    }
    fn grow(&mut self, index: usize, pages: usize) -> bool {
        let rows = pages * self.cache.pool().page_rows();
        match self.cache.extend(self.family, self.decoding[index].0.placement_mut(), rows) {
            Ok(grown) => grown,
            Err(error) => { self.error.get_or_insert(error.into()); false }
        }
    }
    fn set_stalled(&mut self, index: usize, stalled: bool) { self.decoding[index].1 = stalled; }
}

impl Reserve {
    pub fn new(config: ReserveConfig, page_rows: usize) -> Self {
        tracing::info!(chunk = config.chunk(), drain_buffer = ?config.drain_buffer(),
            early_termination = ?config.termination_threshold(), page_rows, "KV decode reservation");
        Self { policy: KvReserve::new(config, page_rows) }
    }

    pub fn policy(&self) -> &KvReserve { &self.policy }
    pub fn stats(&self) -> ReserveStats { self.policy.stats() }

    /// The `/v1/stats` entry: lifetime counters and the stalled gauge.
    pub fn snapshot(&self, stalled: usize) -> serde_json::Value {
        let mut value = serde_json::to_value(self.stats()).unwrap_or_default();
        value["stalled"] = stalled.into();
        value["config"] = serde_json::to_value(self.policy.config()).unwrap_or_default();
        value
    }

    /// Rows past which a request's KV may not grow: the context, what the pool can hold at all,
    /// and what the drafter's page pool can hold with its headroom (a lifetime the pools cannot
    /// hold ends with `length` there).
    pub fn limit<E: CopyEngine, M: Clone + Default, F: PrefixFamily<M>>(cache: &PrefixCache<E, M>, family: &F,
        context: usize) -> usize {
        let target = context.min(cache.pool().capacity() * cache.pool().page_rows());
        family.draft_lifetime_rows().map_or(target, |rows| target.min(rows))
    }

    /// The lease of a request: `output` tokens at most, rounds of `step` rows, within `limit`
    /// rows (the context extent).
    pub fn lease(&self, prompt: usize, output: usize, step: usize, limit: usize)
        -> Result<Lease, cuteafd_engine::reserve::NoRoom> {
        self.policy.lease(prompt, output, step, limit)
    }

    /// Rows to admit a request with, or `None` while it waits behind the drain rule. `origin`
    /// is where its drafter's live context starts (its expected restore point, else 0).
    #[allow(clippy::too_many_arguments)]
    pub fn admit<E: CopyEngine, M: Clone + Default, F: PrefixFamily<M>, A: Reserving<Placement = F::Placement>>(
        &mut self, cache: &PrefixCache<E, M>, family: &F, active: &[A], stalled: &[A], holding: &[Holding<'_>],
        lease: &Lease) -> Option<usize> {
        let decoding = active.iter().chain(stalled).map(|a| {
            let pages = family.pages(a.placement());
            Demand { lease: *a.lease(), held: pages.len(), pages, committed: a.committed(), produced: a.produced(),
                decoding: true, draft: family.draft_demand(a.placement()) }
        });
        let holders = holding.iter().map(|h| Demand { lease: h.lease, held: h.pages.len(), pages: h.pages,
            committed: h.lease.prompt(), produced: 0, decoding: false, draft: None });
        let demands: Vec<Demand<'_>> = decoding.chain(holders).collect();
        let pages = self.policy.admit(cache.room(family), &demands, lease, family.draft_admission(0),
            !stalled.is_empty())?;
        Some(self.policy.admission_rows(lease).min(pages * self.policy.page_rows()))
    }

    /// Plan and apply this round's reservations. On return `active` holds the requests that
    /// run (grown as needed) and `stalled` those that wait. A request that ends now (early
    /// termination, the deadlock break) or a stalled one whose client left is handed to `end`,
    /// which must finish it (`length`, unless closed) and release its pages at once: the round
    /// grants from the pool as it stands after those releases.
    pub fn round<E: CopyEngine, M: Clone + Default, F: PrefixFamily<M>, A: Reserving<Placement = F::Placement>>(
        &mut self, cache: &mut PrefixCache<E, M>, family: &F, active: &mut Vec<A>, stalled: &mut Vec<A>,
        holding: &[Holding<'_>], mut end: impl FnMut(&mut PrefixCache<E, M>, A)) -> Result<()> {
        let mut index = 0;
        while index < stalled.len() {
            if stalled[index].closed() { end(cache, stalled.remove(index)); } else { index += 1; }
        }
        if active.is_empty() && stalled.is_empty() { return Ok(()); }
        let decoding = active.drain(..).map(|a| (a, false)).chain(stalled.drain(..).map(|a| (a, true))).collect();
        let mut glue = Glue { cache, family, decoding, holding, end: &mut end, error: None };
        self.policy.round(&mut glue);
        let Glue { decoding, error, .. } = glue;
        for (request, is_stalled) in decoding {
            if is_stalled { stalled.push(request) } else { active.push(request) }
        }
        error.map_or(Ok(()), Err)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;
    use cuteafd_engine::prefix::{BoxError, FamilyLayout, MarkSlot, MarkStore, PageOwners, PrefixConfig, ReuseRule,
        TailCopy};
    use cuteafd_hostcache::copy::{DeviceRange, StubCopyEngine};

    const ROWS: usize = 64;

    /// A paged family whose kernels read the placement's page list each round (GLM, MiMo).
    struct Paged { pages: usize }
    struct Placed { pages: Vec<u32> }
    impl PrefixFamily for Paged {
        type Placement = Placed;
        fn layout(&self) -> FamilyLayout {
            FamilyLayout { page_rows: ROWS, pages: self.pages, page_bytes: 64, mark_bytes: 0, draft_bytes: 0,
                rule: ReuseRule::EXACT, mark_store: MarkStore::Arena, page_owners: PageOwners::Uniform }
        }
        fn pages<'p>(&self, placement: &'p Placed) -> &'p [u32] { &placement.pages }
        fn extend_pages(&self, placement: &mut Placed, pages: &[u32]) -> Result<(), BoxError> {
            placement.pages.extend_from_slice(pages);
            Ok(())
        }
        fn commit_point(&self, _: &Placed) -> usize { 0 }
        fn capture(&self, _: MarkSlot, _: &Placed, _: usize) -> Result<(), BoxError> { Ok(()) }
        fn restore(&self, _: Option<MarkSlot>, _: &mut Placed, _: usize) -> Result<(), BoxError> { Ok(()) }
        fn copy_rows(&self, _: TailCopy) -> Result<(), BoxError> { Ok(()) }
        fn drain(&self) -> Result<(), BoxError> { Ok(()) }
        fn page_segments(&self, _: u32) -> Vec<DeviceRange> { Vec::new() }
        fn mark_segments(&self, _: MarkSlot) -> Vec<DeviceRange> { Vec::new() }
    }

    struct Request { lease: Lease, placement: Placed, len: usize, produced: usize, closed: bool }
    impl Reserving for Request {
        type Placement = Placed;
        fn lease(&self) -> &Lease { &self.lease }
        fn committed(&self) -> usize { self.len }
        fn produced(&self) -> usize { self.produced }
        fn placement(&self) -> &Placed { &self.placement }
        fn placement_mut(&mut self) -> &mut Placed { &mut self.placement }
        fn closed(&self) -> bool { self.closed }
    }

    fn setup(pages: usize, chunk: usize, buffer: usize, early: i64) -> (Paged, PrefixCache<StubCopyEngine>, Reserve) {
        let family = Paged { pages };
        let cache = PrefixCache::new(family.layout(), PrefixConfig { entries: 0, mark_slots: 0, keep_logits: false,
            min_tokens: 64 }, None).unwrap();
        let config = ReserveConfig::new(chunk, buffer, early, 1 << 20).unwrap();
        (family, cache, Reserve::new(config, ROWS))
    }

    fn admit(reserve: &mut Reserve, cache: &mut PrefixCache<StubCopyEngine>, family: &Paged, active: &[Request],
        stalled: &[Request], prompt: usize, output: usize) -> Option<Request> {
        let lease = reserve.lease(prompt, output, 4, Reserve::limit(cache, family, 1 << 20)).unwrap();
        let rows = reserve.admit(cache, family, active, stalled, &[], &lease)?;
        let placement = cache.admit_cold(family, prompt, rows, |pages| Placed { pages }).unwrap().placement;
        Some(Request { lease, placement, len: prompt, produced: 1, closed: false })
    }

    /// Decode `tokens` rows for every running request (one row per round, as without drafts).
    fn decode(active: &mut [Request], rows: usize) {
        for request in active { request.len += rows; request.produced += rows; }
    }

    #[test]
    fn admission_reserves_prompt_plus_a_chunk_and_the_tail_rule_takes_the_whole_lifetime() {
        let (family, mut cache, mut reserve) = setup(1024, 1024, 0, -1);
        // 100 + 1024 rows = 1124, rounded down to 1088: 17 pages.
        let long = admit(&mut reserve, &mut cache, &family, &[], &[], 100, 100_000).unwrap();
        assert_eq!(long.placement.pages.len(), 17);
        // 100 + 1200 + 4 = 1304 rows: 1088 would leave 216 < 512, so the whole lifetime (21 pages).
        let tail = admit(&mut reserve, &mut cache, &family, &[long], &[], 100, 1200).unwrap();
        assert_eq!(tail.placement.pages.len(), 21);
    }

    #[test]
    fn decode_tops_up_by_a_chunk_then_stalls_and_resumes_when_pages_return() -> Result<()> {
        // 48 pages: three requests of 23-page lifetimes cannot all run to the end at once.
        let (family, mut cache, mut reserve) = setup(48, 640, 0, -1);
        let mut active = Vec::new();
        let mut stalled: Vec<Request> = Vec::new();
        for _ in 0..3 {
            let request = admit(&mut reserve, &mut cache, &family, &active, &stalled, 64, 1400).unwrap();
            active.push(request);
        }
        assert_eq!(active.iter().map(|r| r.placement.pages.len()).collect::<Vec<_>>(), [11, 11, 11]);
        let mut rounds = 0;
        let mut saw_stall = false;
        while !(active.is_empty() && stalled.is_empty()) {
            rounds += 1;
            assert!(rounds < 10_000);
            let mut ended = 0;
            reserve.round(&mut cache, &family, &mut active, &mut stalled, &[], |cache, request| {
                cache.release(&family, &request.placement.pages).unwrap();
                ended += 1;
            })?;
            assert_eq!(ended, 0, "nothing ends early by default");
            saw_stall |= !stalled.is_empty();
            for request in &active {
                assert!(request.len + 4 <= request.placement.pages.len() * ROWS || request.len + 4 > request.lease.lifetime());
            }
            decode(&mut active, 4);
            let (done, running): (Vec<_>, Vec<_>) = active.drain(..).partition(|r| r.len >= r.lease.end());
            active = running;
            for request in done { cache.release(&family, &request.placement.pages)?; }
            let held: usize = active.iter().chain(&stalled).map(|r| r.placement.pages.len()).sum();
            assert!(held <= 48);
        }
        assert!(saw_stall, "a request waits for another to finish");
        assert!(reserve.stats().top_ups > 0 && reserve.stats().stalls > 0);
        assert_eq!(reserve.stats().deadlock_breaks, 0);
        assert_eq!(cache.pool().free(), 48);
        Ok(())
    }

    #[test]
    fn early_termination_ends_a_request_that_would_stall_and_cancelled_stalls_are_released() -> Result<()> {
        let (family, mut cache, mut reserve) = setup(48, 256, 0, 200);
        let mut active = Vec::new();
        let mut stalled: Vec<Request> = Vec::new();
        for _ in 0..3 {
            let request = admit(&mut reserve, &mut cache, &family, &active, &stalled, 64, 1400).unwrap();
            active.push(request);
        }
        let mut ended = 0;
        for _ in 0..2000 {
            reserve.round(&mut cache, &family, &mut active, &mut stalled, &[], |cache, request| {
                assert!(request.produced >= 200, "only past the threshold");
                cache.release(&family, &request.placement.pages).unwrap();
                ended += 1;
            })?;
            decode(&mut active, 4);
            let (done, running): (Vec<_>, Vec<_>) = active.drain(..).partition(|r| r.len >= r.lease.end());
            active = running;
            for request in done { cache.release(&family, &request.placement.pages)?; }
            if active.is_empty() && stalled.is_empty() { break; }
        }
        assert!(ended > 0 && reserve.stats().early_terminations as usize == ended);
        // A stalled request whose client left is released, not resumed.
        let (family, mut cache, mut reserve) = setup(24, 256, 0, -1);
        let mut stalled = vec![admit(&mut reserve, &mut cache, &family, &[], &[], 64, 100).unwrap()];
        stalled[0].closed = true;
        let mut cancelled = 0;
        reserve.round(&mut cache, &family, &mut Vec::new(), &mut stalled, &[], |cache, request| {
            cache.release(&family, &request.placement.pages).unwrap();
            cancelled += 1;
        })?;
        assert_eq!((cancelled, stalled.len()), (1, 0));
        Ok(())
    }

    #[test]
    fn every_serve_loop_reserves_by_chunks_and_stalls_instead_of_truncating() {
        for (family, source) in [
            ("Qwen scheduler (Qwen, Kimi, Step-5)", include_str!("qwen/scheduler.rs")),
            ("V4", include_str!("../families/deepseek_v4/serve.rs")),
            ("GLM", include_str!("../families/glm5/serve.rs")),
            ("GLM Flash", include_str!("../families/glm5_flash/serve.rs")),
            ("MiMo", include_str!("../families/mimo_v2/serve.rs")),
        ] {
            assert!(source.contains("reserve.admit(") && source.contains("reserve.round("), "{family}");
            assert!(source.contains("let mut stalled"), "{family} keeps stalled requests apart");
            assert!(!source.contains("cache.grow("), "{family} grows only through the shared reservation");
            assert!(!source.contains("KV growth exhausted"), "{family} never truncates a reservation");
        }
        let v41 = include_str!("../families/deepseek_v41/v41_native_serve/scheduler.rs");
        assert!(v41.contains("reserve.admit(") && v41.contains("reserve_round(") && !v41.contains("shrink_output"));
        let lanes = include_str!("../families/deepseek_v41/v41_native_serve/scheduler/independent.rs");
        assert!(lanes.contains("!r.stalled") && lanes.contains("if short { lease.stop(); return Ok(()); }"));
    }

    #[test]
    fn admission_waits_behind_a_stalled_request() {
        let (family, mut cache, mut reserve) = setup(1024, 1024, 0, -1);
        let stalled = vec![admit(&mut reserve, &mut cache, &family, &[], &[], 64, 100).unwrap()];
        assert!(admit(&mut reserve, &mut cache, &family, &[], &stalled, 64, 100).is_none());
        assert!(admit(&mut reserve, &mut cache, &family, &stalled, &[], 64, 100).is_some());
    }

    #[derive(Parser)]
    struct Cli {
        #[command(flatten)]
        reserve: ReserveArgs,
    }

    #[test]
    fn options_default_to_128k_output_16k_chunks_and_no_early_end() -> Result<()> {
        let cli = Cli::try_parse_from(["serve"])?;
        assert_eq!(cli.reserve.max_output_tokens, 131_072);
        assert_eq!(cli.reserve.kv_decode_reserve_chunk, 16_384);
        assert_eq!(cli.reserve.kv_reserve_drain_buffer, 0);
        assert_eq!(cli.reserve.kv_early_termination_min_output_tokens, -1);
        let config = cli.reserve.config(1 << 20)?;
        assert_eq!((config.drain_buffer(), config.termination_threshold()), (None, None));
        assert_eq!(cli.reserve.limits(1 << 20)?.output(), 131_072);
        // ≤ 0: the context is the only cap.
        let cli = Cli::try_parse_from(["serve", "--max-output-tokens", "-1"])?;
        assert_eq!(cli.reserve.limits(1 << 20)?.output(), 1 << 20);
        let cli = Cli::try_parse_from(["serve", "--max-output-tokens", "0"])?;
        assert_eq!(cli.reserve.limits(4096)?.output(), 4096);
        Ok(())
    }

    #[test]
    fn startup_refuses_a_drain_buffer_below_an_explicit_early_termination_value() -> Result<()> {
        let cli = Cli::try_parse_from(["serve", "--kv-reserve-drain-buffer", "4096",
            "--kv-early-termination-min-output-tokens", "8192"])?;
        let error = cli.reserve.config(1 << 20).unwrap_err().to_string();
        assert!(error.contains("--kv-reserve-drain-buffer 4096 is below --kv-early-termination-min-output-tokens 8192"), "{error}");
        let cli = Cli::try_parse_from(["serve", "--kv-reserve-drain-buffer", "131072"])?;
        assert!(cli.reserve.config(1 << 20).unwrap_err().to_string().contains("below the effective maximum output"));
        // With max output uncapped the full context bounds the buffer.
        let cli = Cli::try_parse_from(["serve", "--max-output-tokens", "0", "--kv-reserve-drain-buffer", "131072"])?;
        assert_eq!(cli.reserve.config(1 << 20)?.drain_buffer(), Some(131_072));
        let cli = Cli::try_parse_from(["serve", "--kv-reserve-drain-buffer", "8192"])?;
        assert_eq!(cli.reserve.config(1 << 20)?.termination_threshold(), Some(8192));
        Ok(())
    }
}
