//! Text-only MiniMax adapter; admission, lanes and selection stay in the shared scheduler.
use super::{cache::{Placement, PAGE_ROWS}, engine::{Requests, ServingPrefill, ServingRequest},
    prefix::MiniMaxPrefix, owners::Owners};
use crate::shared::{prefix::{CudaCopyEngine, PrefixArgs, Toggle},
    prefill_packing::{Limits, MemberOutcome}, probe,
    qwen::scheduler::{self, Backend}, token_io::{DeviceLogits, TokenSelector}};
use anyhow::{ensure, Context, Result};
use cuteafd_api::openai::NativeRequest;
use cuteafd_engine::{media::{EncodeJob, EmbeddingCache, RequestMedia},
    prefix::{PrefixCache, PrefixConfig, PrefixFamily, RefPagePool}};
use cuteafd_loader::families::minimax_m3::MiniMaxM3Config;
use cuteafd_transport::ExpertV2SourceKind;
use std::{cell::{Cell, RefCell}, collections::{BTreeMap, BTreeSet}, ops::Range, path::Path};

const MESSAGE_STARTS: &[&str] = &["]~b]"];

fn retain_logit_tile(library:&cuteafd_ffi::NativeLibrary, retained:cuteafd_ffi::CuteafdDeviceBuffer,
    first:usize, vocab:usize, logits:DeviceLogits) -> Result<()> {
    ensure!(logits.vocab == vocab && logits.stride == vocab && logits.rows > 0,
        "MiniMax retained logit tile geometry differs");
    let row_bytes=vocab.checked_mul(4).context("MiniMax logit row extent overflow")?;
    let bytes=logits.rows.checked_mul(row_bytes).context("MiniMax logit tile extent overflow")?;
    let offset=first.checked_mul(row_bytes).context("MiniMax logit tile offset overflow")?;
    let target=super::cache::view(retained,offset,bytes)?;
    let source=cuteafd_ffi::CuteafdDeviceBuffer { ptr:logits.ptr.cast_mut(),bytes,
        device_id:retained.device_id,..Default::default() };
    // SAFETY: the callback owns this live head tile until its next projection;
    // the retained plane survives selection and this copy uses its producer stream.
    unsafe { library.copy_d2d_async(target,source,bytes,logits.stream) }
}

fn retain_decode_taps(library: &cuteafd_ffi::NativeLibrary, source: cuteafd_ffi::CuteafdDeviceBuffer,
    target: cuteafd_ffi::CuteafdDeviceBuffer, rows: usize, pitch: usize,
    stream: *mut std::ffi::c_void) -> Result<()> {
    ensure!(rows > 0 && pitch > 0 && source.device_id == target.device_id,
        "MiniMax decode tap geometry/device differs");
    let bytes = rows.checked_mul(pitch).context("MiniMax decode tap extent overflow")?;
    let source = super::cache::view(source, 0, bytes)?;
    let target = super::cache::view(target, 0, bytes)?;
    // SAFETY: native decode initialized these live rows; disjoint retained
    // verify storage and the producer stream survive the round's accept.
    unsafe { library.copy_d2d_async(target, source, bytes, stream) }
}

fn packing_rows(mode: crate::shared::prefill_packing::PackingMode, admitted: usize) -> usize {
    match mode {
        crate::shared::prefill_packing::PackingMode::Unrestricted => admitted,
        // Native NVFP4 route occupancy has no exact cap attestation yet;
        // the shared grouping contract runs unknown routes as singletons.
        crate::shared::prefill_packing::PackingMode::Exact => 0,
    }
}

fn validate_prefix_concurrency(requested: usize, admitted: usize) -> Result<()> {
    ensure!(requested > 0 && requested == admitted,
        "MiniMax prefix concurrency differs from admitted request slots");
    Ok(())
}

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub(crate) enum BackendError {
    #[error("MiniMax neural/speculative drafts are not qualified")]
    DraftsNotQualified,
    #[error("MiniMax backend is terminal after a failed native submission")]
    Terminal,
}

#[derive(Default)]
pub(crate) struct DraftState { anchor:Option<u32>, confidence:Vec<f32> }
impl scheduler::DraftState for DraftState {
    fn pending_rows(&self) -> usize { usize::from(self.anchor.is_some()) }
    fn close(&mut self, token: u32) { self.anchor=Some(token); }
}

#[derive(Clone, Debug)]
struct Submitted {
    identity: u64,
    placement: Placement,
    tokens: Vec<u32>,
}
impl Submitted {
    fn validate(&self, identity: u64, placement: &Placement, start: usize, tokens: &[u32]) -> Result<()> {
        ensure!(identity == self.identity && placement.slot == self.placement.slot
            && placement.units == self.placement.units && start == self.placement.len && tokens == self.tokens,
            "MiniMax accept identity/pages/start/tokens differ from submitted work");
        Ok(())
    }
    fn validate_published(&self, logical: usize, kept: Option<(usize, u32)>, finished: bool) -> Result<()> {
        let end = self.placement.len.checked_add(self.tokens.len()).context("MiniMax submitted extent overflow")?;
        ensure!(self.tokens.len() == 1 && logical == end
            && kept.is_none_or(|(rows, _)| rows == 1) && (kept.is_some() || finished),
            "MiniMax non-speculative accept differs from already-published input row");
        Ok(())
    }
}

fn publish<F: PrefixFamily<Placement = Placement>>(family: &F,
    sequences: &mut [(&mut Placement, &[u32])], submitted: &[Submitted]) -> Result<()> {
    ensure!(sequences.len() == submitted.len(), "MiniMax publication batch differs");
    for ((placement, tokens), request) in sequences.iter_mut().zip(submitted) {
        request.validate(request.identity, placement, placement.len, tokens)?;
        let mut next = Placement { len: request.placement.len.checked_add(request.tokens.len())
            .context("MiniMax publication extent overflow")?, ..request.placement.clone() };
        family.bind(&mut next).map_err(|error| anyhow::anyhow!(error.to_string()))?;
        **placement = next;
    }
    Ok(())
}

/// No allocations or placement choices here: startup passes its admitted context
/// and host-prefix config, and the Owners borrow pins every native backing.
pub(crate) struct MiniMaxBackend<'e, 'a> {
    media_enabled: bool,
    snapshot: std::path::PathBuf,
    owners: &'e Owners<'a>,
    cfg: &'e MiniMaxM3Config,
    execution: RefCell<ServingPrefill<'a>>,
    schedule: Requests,
    context: usize,
    host: Option<cuteafd_hostcache::config::Config>,
    profile: RefCell<[f64; 2]>,
    identities: RefCell<BTreeMap<usize, u64>>,
    next_identity: Cell<u64>,
    pending: RefCell<Option<Vec<Submitted>>>,
    terminal: Cell<bool>,
    selector_receipts: RefCell<Option<Vec<cuteafd_loader::placement::planner::Receipt>>>,
    packing: crate::shared::prefill_packing::PackingMode,
}

impl<'e, 'a> MiniMaxBackend<'e, 'a> {
    pub(crate) fn with_media_enabled(mut self, enabled: bool) -> Self { self.media_enabled=enabled; self }
    pub(crate) fn with_probe_snapshot(mut self, snapshot: &Path) -> Self {
        self.snapshot = snapshot.to_owned(); self
    }
    pub fn new(owners: &'e Owners<'a>, cfg: &'e MiniMaxM3Config, execution: ServingPrefill<'a>,
        served_context: usize, host: Option<cuteafd_hostcache::config::Config>,
        packing: crate::shared::prefill_packing::PackingMode) -> Result<Self> {
        let storage = &owners.storage;
        ensure!(matches!(storage.ranks.len(), 1 | 2)
            && (storage.ranks.len() == 2) == (storage.split.is_some() || storage.heads.is_some())
            && !(storage.split.is_some() && storage.heads.is_some()),
            "MiniMax backend rank count differs from admitted split execution");
        let state = storage.requests.borrow();
        let bank = &state.ranks[0].bank;
        ensure!(served_context > 0 && served_context <= bank.rope.positions as usize
            && served_context <= bank.layout.pool_pages as usize * PAGE_ROWS,
            "MiniMax served context exceeds admitted cache");
        let rank = &storage.ranks[0];
        let layers = storage.ranks.iter().flat_map(|rank| rank.layout.layers.iter().copied()).collect::<BTreeSet<_>>();
        ensure!(layers.iter().copied().eq(0..cfg.layers), "MiniMax backend attention owners do not cover every layer");
        for (gpu, rank) in storage.ranks.iter().enumerate() {
            let stream = rank.decode.borrow().stream.raw;
            ensure!(state.ranks[gpu].prefill.len() == rank.prefill.len()
                && rank.prefill.iter().all(|owner| owner.borrow().stream.raw == stream)
                && state.ranks[gpu].bank.readers().iter().all(|reader| reader.raw == stream),
                super::engine::ServingError::DifferentStreams);
        }
        let schedule = Requests::new(state.capacity, served_context, cfg.layers, rank.prefill.len())?;
        drop(state);
        Ok(Self { media_enabled: false, snapshot: Default::default(), owners, cfg, execution: RefCell::new(execution), schedule, context: served_context,
            host, profile: RefCell::new([0.0; 2]), identities: RefCell::new(BTreeMap::new()),
            next_identity: Cell::new(1), pending: RefCell::new(None), terminal: Cell::new(false),
            selector_receipts: RefCell::new(None), packing })
    }

    fn live(&self) -> Result<()> {
        ensure!(!self.terminal.get(), BackendError::Terminal);
        // Split waits are direct abortable waits: a timed-out one skipped itself
        // and desynchronized every later receive, so the backend is terminal.
        if self.owners.storage.ranks.len() == 2 {
            if let Err(error) = self.library().check_peer_wait_timeout() {
                self.terminal.set(true);
                return Err(error);
            }
        }
        Ok(())
    }
    fn idle(&self) -> Result<()> { self.live() }
    fn identity(&self, placement: &Placement) -> Result<u64> {
        self.identities.borrow().get(&placement.slot).copied().context("MiniMax request is not bound")
    }
    fn drain(&self) -> Result<()> {
        let result = self.execution.borrow().drain(&self.owners.storage);
        if result.is_err() { self.terminal.set(true); }
        result
    }
    fn completed<T>(&self, result: Result<T>) -> Result<T> {
        if result.is_err() { self.terminal.set(true); }
        let drain = self.drain();
        if drain.is_err() { self.terminal.set(true); }
        // A drained step whose peer wait timed out skipped that wait: its results are void.
        let drain = drain.and_then(|()| self.completed_health());
        match result { Ok(value) => { drain?; Ok(value) }, Err(error) => Err(error) }
    }
    fn completed_health(&self) -> Result<()> {
        self.live()?;
        crate::shared::spark_waves::healthy().inspect_err(|_| self.terminal.set(true))
    }
    fn queued<T>(&self, result: Result<T>) -> Result<T> {
        match result {
            Ok(value) => Ok(value),
            Err(error) => self.completed(Err(error)),
        }
    }
    fn validate(&self, sequences: &[(&mut Placement, &[u32])], pool: &RefPagePool) -> Result<Vec<Submitted>> {
        self.idle()?;
        let state = self.owners.storage.requests.borrow();
        let bank = &state.ranks[0].bank;
        ensure!(!pool.is_parity() && pool.reserved() == 0 && pool.page_rows() == PAGE_ROWS
            && pool.capacity() == bank.layout.pool_pages as usize,
            "MiniMax request pool differs from admitted uniform cache");
        let mut slots = BTreeSet::new();
        let mut writable = BTreeSet::new();
        let mut submitted = Vec::with_capacity(sequences.len());
        for (placement, tokens) in sequences {
            let end = placement.len.checked_add(tokens.len()).context("MiniMax append overflow")?;
            ensure!(!tokens.is_empty() && end <= self.context && slots.insert(placement.slot),
                "MiniMax invalid or duplicate request append");
            self.owners.storage.embedding.check(tokens)?;
            bank.validate(placement, end)?;
            for &page in &placement.units {
                ensure!(pool.refs(page) > 0, "MiniMax request refers to an unowned shared-pool page");
            }
            for position in placement.len..end {
                let page = placement.units[position / PAGE_ROWS];
                ensure!(pool.refs(page) == 1 && writable.insert((page, position % PAGE_ROWS)),
                    "MiniMax write requires private nonaliased COW pages");
            }
            submitted.push(Submitted { identity: self.identity(placement)?, placement: (**placement).clone(),
                tokens: tokens.to_vec() });
        }
        ensure!(!submitted.is_empty() && submitted.len() <= state.capacity.requests as usize,
            "MiniMax request count exceeds admitted concurrency");
        Ok(submitted)
    }

    fn prefill(&self, sequences: &mut [(&mut Placement, &[u32])], heads: &[Range<usize>],
        pool: &RefPagePool, consume: impl FnMut(usize, Range<usize>, DeviceLogits) -> Result<()>) -> Result<()> {
        let submitted = self.validate(sequences, pool)?;
        ensure!(heads.len() == submitted.len(), "MiniMax head inventory differs from append batch");
        let mut placements = submitted.iter().map(|request| request.placement.clone()).collect::<Vec<_>>();
        let mut requests = placements.iter_mut().zip(&submitted).zip(heads).map(|((placement, request), heads)|
            ServingRequest { identity: request.identity, kind: ExpertV2SourceKind::Prefill,
                start: request.placement.len, tokens: &request.tokens, heads: heads.clone(), placement }).collect::<Vec<_>>();
        let result = self.execution.borrow_mut().execute(&self.owners.storage, self.cfg, &self.schedule,
            &mut requests, false, consume);
        self.completed(result)?;
        drop(requests);
        let published = (|| {
            for ((placement, _), next) in sequences.iter_mut().zip(placements) {
                let mut next = next;
                self.owners.storage.prefix.bind(&mut next).map_err(|error| anyhow::anyhow!(error.to_string()))?;
                **placement = next;
            }
            Ok(())
        })();
        self.completed(published)
    }

    fn decode(&self, sequences: &mut [(&mut Placement, &[u32])], pool: &RefPagePool,
        retain_receipt: bool, captured: bool) -> Result<Option<DeviceLogits>> {
        ensure!(self.pending.borrow().is_none(), "MiniMax decode overlaps an unaccepted round");
        let submitted = self.validate(sequences, pool)?;
        let queries = submitted.iter().enumerate().flat_map(|(sequence, request)|
            (0..request.tokens.len()).map(move |row| (sequence, row))).collect::<Vec<_>>();
        ensure!(queries.len() <= self.geometry().decode_rows, "MiniMax verify queries exceed admission");
        let mut placements = queries.iter().map(|&(sequence, row)| Placement {
            len: submitted[sequence].placement.len + row, ..submitted[sequence].placement.clone()
        }).collect::<Vec<_>>();
        let requests = placements.iter_mut().zip(&queries).map(|(placement, &(sequence, row))| {
            let request = &submitted[sequence];
            ServingRequest { identity: request.identity, kind: ExpertV2SourceKind::Decode,
                start: placement.len, heads: placement.len..placement.len + 1,
                tokens: &request.tokens[row..row + 1], placement }
        }).collect::<Vec<_>>();
        let mut graphs = self.owners.graphs.borrow_mut();
        let result = self.execution.borrow_mut().decode(&self.owners.storage, self.cfg, &self.schedule,
            &requests, &mut graphs, captured);
        let logits = self.queued(result)?;
        let paired = self.owners.storage.ranks.len() > 1;
        if paired { self.completed_health()?; } // The paired executor drained its rank tails.
        for graph in graphs.iter_mut() { graph.tick(); }
        let observe = graphs.iter().any(|graph| graph.needs_boundary());
        let draft = self.owners.storage.draft_prefix.as_ref();
        if !paired && (observe || draft.is_none()) {
            // No drafter: canonical state publishes before selection. Graph
            // maintenance likewise needs a completed sample.
            let wait = std::time::Instant::now();
            self.completed(Ok(()))?;
            if observe && draft.is_some() { crate::shared::decode_graph::device::record_boundary_drain(self.owners.storage.ranks.iter().map(|r| r.decode.borrow().stream.device.id), wait.elapsed()); }
        }
        if observe {
            // SAFETY: paired execution or the publication/maintenance fence
            // just drained the target queues. Pins borrow persistent storage.
            unsafe { crate::shared::decode_graph::device::observe_completed(self.library(), |visit| {
                for graph in graphs.iter_mut() { graph.visit(visit)?; }
                Ok(())
            })?; }
            for graph in graphs.iter_mut() { graph.observed(); }
        }
        drop(graphs);
        if let Some(draft) = draft {
            let staged = (|| {
                let owner = self.owners.storage.ranks.last().context("MiniMax tap rank missing")?.decode.borrow();
                let source = owner.target_taps().context("MiniMax admitted decode taps missing")?.output();
                let target = self.owners.storage.verify_taps.as_ref().context("MiniMax retained verify taps missing")?;
                retain_decode_taps(self.library(), source, target.buffer, queries.len(),
                    draft.drafter.cfg.taps.len() * self.cfg.hidden * 2, owner.stream.raw)
            })();
            self.queued(staged)?;
            // Keep native target writes and taps staged just like eager verify:
            // accept advances target and draft frontiers together after selection.
            *self.pending.borrow_mut() = Some(submitted);
            return Ok(Some(logits));
        }
        // Non-speculative input is committed even if selection later cancels.
        // A retained receipt validates the scheduler's optional accept callback;
        // publication itself never depends on that callback or UngraphedKind.
        let published = publish(&self.owners.storage.prefix, sequences, &submitted);
        self.completed(published)?;
        *self.pending.borrow_mut() = retain_receipt.then_some(submitted);
        Ok(Some(logits))
    }
}
impl Drop for MiniMaxBackend<'_, '_> {
    fn drop(&mut self) {
        let _ = self.drain();
        self.pending.get_mut().take();
    }
}

impl<'e, 'a> Backend<'a> for MiniMaxBackend<'e, 'a> {
    type Placement = Placement;
    type Draft = DraftState;
    type History = u64;
    type Coordinates = ();
    type Encoder = crate::shared::qwen::Encoder;
    type Meta = ();
    type Prefix = MiniMaxPrefix<'a>;

    fn geometry(&self) -> scheduler::Geometry {
        let state = self.owners.storage.requests.borrow();
        scheduler::Geometry { family: "minimax_m3", vocab_size: self.cfg.vocab_size, max_context: self.context,
            slots: state.capacity.requests as usize, prefill_rows: state.capacity.prefill.rows as usize,
            decode_rows: self.owners.storage.verify_logits.as_ref().map_or(state.capacity.decode.rows as usize,
                |owner| owner.buffer.bytes / (self.cfg.vocab_size * 4)), }
    }
    fn message_starts(&self) -> &[&str] { MESSAGE_STARTS }
    fn library(&self) -> &'a cuteafd_ffi::NativeLibrary { self.owners.storage.ranks[0].decode.borrow().stream.device.library }
    fn profile(&self) -> &RefCell<[f64; 2]> { &self.profile }
    fn selector_ready(&self, selector: &mut TokenSelector<'a>) -> Result<()> {
        ensure!(self.selector_receipts.borrow().is_none(), "MiniMax selector initialized more than once");
        selector.reserve_sampler()?;
        let receipts = self.owners.reconcile_selector(selector)?;
        tracing::info!(owners = receipts.len(), "MiniMax scheduler selector reconciled before readiness");
        *self.selector_receipts.borrow_mut() = Some(receipts);
        Ok(())
    }
    fn prefix_cache(&self, args: &PrefixArgs, concurrency: usize)
        -> Result<(Self::Prefix, PrefixCache<CudaCopyEngine<'a>>)> {
        self.idle()?;
        validate_prefix_concurrency(concurrency, self.geometry().slots)?;
        ensure!(args.prefix_partial == Toggle::Off, "MiniMax requires exact prefix restores");
        let family = self.owners.storage.prefix.clone();
        let host = self.host.clone().map(|config| CudaCopyEngine::registered_owned(self.library(),
            family.snapshot_owners()).map(|copy| (config, copy))).transpose()?;
        let host_bytes = host.as_ref().map_or(0, |(config, _)| config.bytes);
        let layout = family.layout();
        let cache = PrefixCache::new(layout, PrefixConfig { entries: args.prefix_cache_entries,
            mark_slots: family.mark_slots(), keep_logits: true, min_tokens: args.prefix_cache_min_tokens }, host)?;
        cuteafd_bench::context::set_kv((layout.pages * layout.page_rows) as u64,
            Some((layout.pages * layout.page_rows) as u64), layout.pages as u64, family.record_format(), host_bytes);
        Ok((family, cache))
    }
    fn prepare(&self, job: NativeRequest, tokens: Vec<u32>) -> Result<(scheduler::Prompt, RequestMedia, Vec<EncodeJob>)> {
        self.owners.storage.embedding.check(&tokens)?;
        if job.probe.as_ref().is_some_and(|p|!p.spec.media.is_empty()) {
            return crate::shared::probe::still::prepare(job,tokens,self.cfg.vocab_size as u32,self.cfg.hidden,
                self.context,&self.snapshot,cuteafd_loader::media::ImageFamily::MiniMax);
        }
        ensure!(self.media_enabled || job.media.is_empty(),"vision is disabled");
        if job.media.is_empty() {
            return crate::shared::probe::still::prepare(job,tokens,self.cfg.vocab_size as u32,self.cfg.hidden,
                self.context,&self.snapshot,cuteafd_loader::media::ImageFamily::MiniMax);
        }
        let tokenizer=cuteafd_loader::cached_tokenizer(&self.snapshot)?;
        let ids=tokenizer.added_token_ids(&["]<]image[>[","]<]start of image[>[","]<]end of image[>["]);
        ensure!(ids.len()==3,"MiniMax image markers absent");
        let mut job=job;
        let mut groups=std::mem::take(&mut job.image_groups);
        ensure!(groups.len()==job.media.len(),"MiniMax source groups absent");
        for group in &mut groups {
            group.fragments=vec![cuteafd_loader::media::ImageFragment::Tokens(vec![ids[1]]),
                cuteafd_loader::media::ImageFragment::Features {image:0,placeholder:ids[0]},
                cuteafd_loader::media::ImageFragment::Tokens(vec![ids[2]])];
        }
        crate::shared::vision::still::Binding::prepare(job,tokens,self.cfg.vocab_size as u32,self.cfg.hidden,
            self.context,ids[0],groups)
    }
    fn probe_features(&self, prompt: &scheduler::Prompt, media: &mut RequestMedia,
        cache: &mut EmbeddingCache, snapshot: &Path) -> Result<()> {
        if prompt.job.probe.as_ref().is_some_and(|p|p.spec.feature_scoring()) {
            crate::shared::probe::still::features(prompt, media, cache, snapshot, cuteafd_loader::media::ImageFamily::MiniMax)?;
        }
        Ok(())
    }
    fn coordinates(&self, tokens: &[u32], job: &NativeRequest, media: &RequestMedia) -> Result<()> {
        if self.media_enabled && job.probe.as_ref().is_none_or(|p|p.spec.media.is_empty()) {
            ensure!(media.prompt_len()==tokens.len(),"MiniMax media prompt extent");return Ok(());
        }
        let spans = crate::shared::probe::still::spans(job, tokens, self.cfg.vocab_size as u32,
            self.context, &self.snapshot, cuteafd_loader::media::ImageFamily::MiniMax)?;
        ensure!(spans == media.spans(), "MiniMax probe media coordinates differ");
        Ok(())
    }
    fn placement(&self, units: Vec<u32>, slot: i32) -> Placement { Placement { media_spans: Vec::new(), units, slot: slot as usize, len: 0 } }
    fn bind_request(&self, placement: &mut Placement, _: (), media: RequestMedia,
        tokens: &[u32], resume: usize) -> Result<()> {
        self.idle()?;
        self.owners.storage.probe_media.borrow_mut().remove(&placement.slot);
        placement.media_spans=media.spans().to_vec();
        if !media.spans().is_empty() { self.owners.storage.probe_media.borrow_mut().insert(placement.slot, media); }
        ensure!(placement.len == resume && resume <= tokens.len(), "MiniMax restored request frontier differs");
        self.owners.storage.requests.borrow().ranks[0].bank.validate(placement, resume)?;
        let identity = self.next_identity.get();
        self.next_identity.set(identity.checked_add(1).context("MiniMax request identity exhausted")?);
        self.identities.borrow_mut().insert(placement.slot, identity);
        if let Some(draft) = &self.owners.storage.draft_prefix {
            draft.bind_prompt(placement.slot, tokens.len())?;
        }
        Ok(())
    }
    fn plan_draft_context(&self, placement: &Placement, end: usize, points: &[usize]) -> Result<()> {
        if let Some(draft) = &self.owners.storage.draft_prefix {
            draft.prompt_ends.borrow_mut()[placement.slot] = end;
            draft.points.borrow_mut()[placement.slot] = points.to_vec();
        }
        Ok(())
    }
    fn history(&self, placement: &Placement) -> u64 { self.identities.borrow().get(&placement.slot).copied().unwrap_or(0) }
    fn restore_draft(&self, placement:&Placement, frontier:usize, tokens:&[u32], state:&mut DraftState) -> Result<()> {
        let draft=self.owners.storage.draft_prefix.as_ref().context("MiniMax drafter was not admitted")?;
        ensure!(draft.frontier(placement.slot)? == frontier,"MiniMax restored draft frontier differs");
        state.anchor=tokens.get(frontier).copied();
        Ok(())
    }
    fn require_live(&self) -> Result<()> { self.live() }
    fn log_table_stats(&self, _: &str) {}
    fn prefill_device(&self, placement: &mut Placement, tokens: &[u32], logit_rows: usize,
        pool: &RefPagePool) -> Result<Option<DeviceLogits>> {
        ensure!(logit_rows <= 1 && logit_rows <= tokens.len(), "MiniMax serving returns at most one prefill row; scoring streams tiles");
        let end = placement.len.checked_add(tokens.len()).context("MiniMax prefill overflow")?;
        let mut logits = None;
        self.prefill(&mut [(placement, tokens)], &[end - logit_rows..end], pool, |_, _, row| {
            ensure!(logits.is_none(), "MiniMax final prefill row was projected more than once");
            logits = Some(row); Ok(())
        })?;
        Ok(logits)
    }
    fn streamed_prefill_scoring(&self) -> bool { true }
    fn prefill_scoring(&self, placement: &mut Placement, tokens: &[u32], logit_rows: usize,
        position: usize, probe: &probe::ProbeRef, pool: &RefPagePool) -> Result<Option<probe::ScoreLogits>> {
        ensure!(logit_rows <= tokens.len(), "MiniMax scored suffix exceeds append");
        let end = placement.len.checked_add(tokens.len()).context("MiniMax scoring overflow")?;
        let first = end - logit_rows;
        let scoring = probe::ScoringRows { consumer: probe, position };
        self.prefill(&mut [(placement, tokens)], &[first..end], pool, |_, span, logits|
            scoring.record(self.library(), span.start - first, &logits))?;
        Ok((logit_rows > 0).then_some(probe::ScoreLogits::Recorded))
    }
    fn packed_limits(&self) -> Option<Limits> {
        Some(Limits { rows: packing_rows(self.packing, self.geometry().prefill_rows),
            chunk_rows: self.geometry().prefill_rows, max_context: self.context, taps: None })
    }
    fn prefill_packed(&self, sequences: &mut [scheduler::PrefillSeq<'_, Placement, DraftState>], pool: &RefPagePool,
        on_logits: &mut dyn FnMut(usize, DeviceLogits) -> MemberOutcome) -> Result<Vec<Result<()>>> {
        ensure!(self.owners.storage.draft_prefix.is_some() || sequences.iter().all(|sequence| !sequence.neural),
            BackendError::DraftsNotQualified);
        let mut heads = Vec::with_capacity(sequences.len());
        for sequence in sequences.iter() {
            ensure!(sequence.start == sequence.placement.len && !sequence.tokens.is_empty(), "MiniMax packed start differs");
            let end = sequence.start.checked_add(sequence.tokens.len()).context("MiniMax packed extent overflow")?;
            heads.push(end - 1..end);
        }
        let mut target = sequences.iter_mut().map(|sequence| (&mut *sequence.placement, sequence.tokens)).collect::<Vec<_>>();
        let mut outcomes = (0..target.len()).map(|_| None).collect::<Vec<_>>();
        self.prefill(&mut target, &heads, pool, |index, _, logits| {
            ensure!(outcomes[index].is_none(), "MiniMax packed member head consumed twice");
            outcomes[index] = Some(on_logits(index, logits)?); Ok(())
        })?;
        drop(target);
        for sequence in sequences.iter_mut().filter(|sequence| sequence.neural) {
            sequence.draft.anchor=sequence.next;
        }
        outcomes.into_iter().map(|outcome| outcome.context("MiniMax packed member head missing")).collect()
    }
    fn verify_device(&self, sequences: &mut [(&mut Placement, &[u32])], speculative: bool,
        pool: &RefPagePool) -> Result<Option<DeviceLogits>> {
        let _ = speculative;
        self.decode(sequences, pool, true, true)
    }
    fn verify_device_ungraphed(&self, sequences: &mut [(&mut Placement, &[u32])], speculative: bool,
        kind: scheduler::UngraphedKind, pool: &RefPagePool) -> Result<Option<DeviceLogits>> {
        let _ = speculative;
        if kind == scheduler::UngraphedKind::Decode { return self.decode(sequences, pool, true, false); }
        let total: usize = sequences.iter().map(|(_, tokens)| tokens.len()).sum();
        ensure!(sequences.len() == 1 && total <= self.geometry().decode_rows,
            "MiniMax diagnostic scoring requires one bounded sequence");
        let start = sequences[0].0.len;
        if let Some(retained)=&self.owners.storage.verify_logits {
            let mut stream=None;
            self.prefill(sequences,&[start..start + total],pool,|_,span,logits| {
                retain_logit_tile(self.library(),retained.buffer,span.start - start,self.cfg.vocab_size,logits)?;
                stream=Some(logits.stream);
                Ok(())
            })?;
            return Ok(Some(DeviceLogits { ptr:retained.buffer.ptr,rows:total,vocab:self.cfg.vocab_size,
                stride:self.cfg.vocab_size,stream:stream.context("MiniMax diagnostic scoring produced no rows")?,greedy:None }));
        }
        if total == 1 { return self.decode(sequences, pool, false, false); }
        let mut result = None;
        self.prefill(sequences, &[start..start + total], pool, |_, _, logits| {
            ensure!(result.is_none(), "MiniMax bounded diagnostic head split unexpectedly");
            result = Some(logits); Ok(())
        })?;
        Ok(result)
    }
    fn verify_bucket_rows(&self, rows: usize, _: bool, _: bool) -> usize { rows }
    fn verify_attention_trace(&self, contexts: &[Vec<usize>], rows: usize) -> crate::shared::speculation_trace::Deferred {
        let contexts = contexts.to_vec();
        crate::shared::speculation_trace::Deferred::new(move || serde_json::json!({
            "kind":"minimax_msa", "rows":rows, "contexts":contexts, "attention":"eager", "arithmetic":"eager" }))
    }
    fn prefill_draft(&self, placement:&Placement, _:usize, _: &[u32], next:Option<u32>, state:&mut DraftState) -> Result<()> {
        let draft=self.owners.storage.draft_prefix.as_ref().context("MiniMax drafter not admitted")?;
        ensure!(draft.frontier(placement.slot)? == placement.len,"MiniMax draft prefill frontier differs");
        state.anchor=next;
        Ok(())
    }
    fn draft(&self,sequences:&mut [scheduler::DraftSeq<'_,Placement,DraftState>],timing:&mut scheduler::DraftTiming,
        _: &mut TokenSelector<'_>) -> Result<Vec<Vec<u32>>> {
        use crate::shared::speculation::dflash::{DraftSeq,TargetHead};
        let draft=self.owners.storage.draft_prefix.as_ref().context("MiniMax drafter not admitted")?;
        let mut output=vec![Vec::new();sequences.len()];
        for sequence in sequences.iter_mut() { sequence.seq.confidence.clear(); }
        let selected=sequences.iter().enumerate().filter(|(_,sequence)| sequence.depth > 0)
            .map(|(index,sequence)| (index,sequence.depth)).collect::<Vec<_>>();
        if selected.is_empty() { return Ok(output); }
        let rows=selected.iter().map(|&(index,_)| {
            let sequence=&sequences[index];
            ensure!(draft.frontier(sequence.placement.slot)? == sequence.placement.len,"MiniMax draft target frontier differs");
            Ok(DraftSeq { slot:sequence.placement.slot,anchor:sequence.seq.anchor.context("MiniMax draft anchor missing")?,
                position:sequence.placement.len,valid_from:0 })
        }).collect::<Result<Vec<_>>>()?;
        let head=self.owners.storage.ranks.last().context("MiniMax head rank")?.weights
            .operand("language_model.lm_head.weight")?.buffer;
        let started=std::time::Instant::now();
        let proposed=draft.drafter.draft_device(&rows,&self.owners.storage.embedding,TargetHead::Bf16Buffer(&head))?;
        timing.steps+=1;timing.seconds+=started.elapsed().as_secs_f64();
        for ((index,depth),draft) in selected.into_iter().zip(proposed) {
            output[index]=draft.tokens.into_iter().take(depth).collect();
            sequences[index].seq.confidence=draft.confidence;
        }
        Ok(output)
    }
    // The bundled dSpark's trained confidence head, one sigmoid per drafted position.
    fn draft_head_kind(&self) -> cuteafd_core::HeadScore { cuteafd_core::HeadScore::Probability }
    fn draft_head<'d>(&self,draft:&'d DraftState) -> Option<&'d [f32]> { Some(&draft.confidence) }
    fn accept(&self, verified: &mut [scheduler::Verified<'_, Placement, DraftState, u64>], speculative: bool) -> Result<()> {
        self.live()?;
        if let Some(draft)=&self.owners.storage.draft_prefix {
            let submitted=self.pending.borrow_mut().take().context("MiniMax staged accept has no receipt")?;
            let checked=(|| {
                ensure!(submitted.len() == verified.len(),"MiniMax staged accept request count differs");
                let mut first=0;
                for (sent,sequence) in submitted.iter().zip(verified.iter()) {
                    sent.validate(sequence.history,sequence.placement,sequence.start,sequence.rows)?;
                    let keep=if speculative { sequence.kept.map_or(0,|(rows,_)| rows) } else { sequence.rows.len() };
                    ensure!(keep <= sequence.rows.len() && sequence.first_row == first
                        && self.identity(sequence.placement)? == sent.identity
                        && draft.frontier(sequence.placement.slot)? == sequence.start,
                        "MiniMax staged accept identity/frontier/rows differ");
                    first+=sequence.rows.len();
                }
                let taps=self.owners.storage.verify_taps.as_ref().context("MiniMax retained verify taps missing")?;
                let stream=self.owners.storage.ranks.last().context("MiniMax head rank")?.decode.borrow().stream.raw;
                for sequence in verified.iter_mut() {
                    let keep=if speculative { sequence.kept.map_or(0,|(rows,_)| rows) } else { sequence.rows.len() };
                    if keep > 0 {
                        // SAFETY: retained verify taps are initialized and owned until this accept drains.
                        unsafe { draft.drafter.capture_concatenated(taps.buffer,sequence.first_row,keep,0,stream)?; }
                        let rows=(0..keep).map(|tap_row| crate::shared::speculation::dflash::ContextRow {
                            tap_row,slot:sequence.placement.slot,position:sequence.start + tap_row }).collect::<Vec<_>>();
                        draft.drafter.update(&rows)?;
                        draft.advance(sequence.placement.slot,sequence.start,keep)?;
                    }
                    let mut next=Placement { len:sequence.start + keep,..sequence.placement.clone() };
                    self.owners.storage.prefix.bind(&mut next).map_err(|error| anyhow::anyhow!(error.to_string()))?;
                    *sequence.placement=next;
                    if let Some((_,anchor))=sequence.kept { sequence.seq.anchor=Some(anchor); }
                }
                Ok(())
            })();
            return self.completed(checked);
        }
        ensure!(!speculative && verified.iter().all(|sequence| !sequence.neural), BackendError::DraftsNotQualified);
        let submitted = self.pending.borrow_mut().take().context("MiniMax accept has no native receipt")?;
        let checked = (|| {
            ensure!(submitted.len() == verified.len(), "MiniMax accepted batch count differs");
            let mut first_row = 0;
            for (sent, actual) in submitted.iter().zip(verified.iter()) {
                sent.validate(actual.history, actual.placement, actual.start, actual.rows)?;
                sent.validate_published(actual.placement.len, actual.kept, actual.finished)?;
                ensure!(self.identity(actual.placement)? == sent.identity && actual.first_row == first_row,
                    "MiniMax accepted request generation or packed offset differs");
                first_row += actual.rows.len();
            }
            Ok(())
        })();
        // The plain receipt only validates already-published host metadata.
        self.queued(checked)
    }
    fn cancel(&self, placement: &Placement) -> Result<()> {
        // Drain before removing evidence or returning pages/slots to the scheduler.
        self.drain()?;
        if self.pending.borrow().as_ref().is_some_and(|wave| wave.iter().any(|sent| sent.placement.slot == placement.slot)) {
            self.pending.borrow_mut().take();
        }
        self.identities.borrow_mut().remove(&placement.slot);
        self.owners.storage.probe_media.borrow_mut().remove(&placement.slot);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::shared::memory::device::{Allocation, Device, Stream};
    use cuteafd_core::memory_layout::Tag;
    use cuteafd_engine::prefix::{After, SnapshotKind};
    use cuteafd_ffi::{native_library_lifetime_fixture::Fixture, NativeLibrary, CuteafdDeviceBuffer as Buffer};
    use cuteafd_loader::{families::minimax_m3::execution::{MsaCache, MsaRope},
        placement::families::minimax_m3::RequestCapacity};
    use std::rc::Rc;

    #[test]
    #[ignore = "requires sealed MiniMax model, native library and three/four Spark workers"]
    fn native_zero_proposal_matches_plain_decode_state_and_continuation() -> Result<()> {
        use clap::Parser;
        use sha2::{Digest, Sha256};
        use std::sync::Arc;
        use cuteafd_loader::{plan::Checkpoint, placement::planner::PrefillShape};
        cuteafd_core::set_expert_geometry(cuteafd_core::ExpertGeometry::MINIMAX_M3)
            .map_err(|previous| anyhow::anyhow!("MiniMax native proof needs an isolated process, found {previous:?}"))?;
        let snapshot = std::env::var("CUTEAFD_GOLDEN_SNAPSHOT")?;
        let native = std::env::var("CUTEAFD_NATIVE_LIB")?;
        let manifest = std::env::var("CUTEAFD_PROGRAM_MANIFEST")?;
        let peers = std::env::var("CUTEAFD_GOLDEN_PEERS")?;
        let mut argv = vec!["cuteafd", "serve-minimax", "--snapshot", &snapshot,
            "--native-lib", &native, "--manifest", &manifest, "--peers", &peers,
            "--max-sequences", "4", "--dspark"];
        let split = std::env::var("CUTEAFD_MINIMAX_PROOF_SPLIT").unwrap_or_else(|_| "off".into());
        if split == "on" { argv.extend(["--split-device", "1"]); }
        let crate::cli::Commands::ServeMinimax(mut args) = crate::cli::Cli::try_parse_from(argv)?.command
            else { unreachable!() };
        args.prefix.resolve_entries(4);
        let view = args.attention.select(&args.snapshot)?;
        args.attention.validate_view(&view)?;
        let checkpoint = Checkpoint::coordinator_view(&view, false, false)?;
        let cfg = MiniMaxM3Config::from_hf(&checkpoint.config)?;
        let prefill = cuteafd_loader::serving_capacity::prefill_waves::prefill_shape(args.peers.len(),
            cuteafd_loader::plan::Family::prefill_limits(&cuteafd_loader::plan::families::minimax::MINIMAX_M3),
            Default::default())?;
        // SAFETY: the sealed native library outlives all admitted owners/streams.
        let library = Arc::new(unsafe { NativeLibrary::load(&args.native_lib)? });
        let programs = library.programs()?.with_manifest(&args.manifest)?;
        let loaded = super::super::startup::load(&library, &programs, &view, &checkpoint, &cfg,
            super::super::startup::Options { media: &crate::shared::vision::still::Options {vision:cuteafd_loader::plan::MediaMode::Off,..Default::default()},
                device: args.device, split_device: args.split_device, peers: &args.peers,
                native_lib: &args.native_lib, manifest_path: &args.manifest,
                split_wire: super::super::dense::PartialPrecision::DEFAULT,
                attention_placement: cuteafd_loader::placement::attention::parse(&args.attention_placement).map_err(anyhow::Error::msg)?
                    .unwrap_or(cuteafd_loader::placement::AttentionPlacement::Heads),
                prefill: PrefillShape { rows: prefill.unit_rows, waves: prefill.waves },
                decode_rows: args.decode_rows.as_option().unwrap_or(64),
                sequences: 4, context_tokens: 0, pool_tokens: None,
                onboard: args.rtx_expert_layers.resolve(cfg.moe.iter().filter(|&&moe| moe).count()),
                prefix: &args.prefix, expert_usage: args.expert_usage,
                usage_reader: args.expert_usage_reader.clone(),
                resident_weights: crate::shared::memory::budget::resident_weight_limit(), draft: true,
            })?;
        let context = usize::try_from(loaded.admitted.solved.served_context())?;
        let backend = MiniMaxBackend::new(&loaded.owners, &cfg, loaded.execution, context,
            loaded.host_prefix, args.prefill_packing)?;
        let family = &backend.owners.storage.prefix;
        let layout = family.layout();
        let mut pool = RefPagePool::new(layout.pages, layout.page_rows);
        let tokenizer = cuteafd_loader::LoadedTokenizer::from_snapshot(&args.snapshot)?;
        let text = "Write a Rust function that merges two sorted slices. Explain bounds, empty inputs, duplicate elements, and why the result stays sorted.\n";
        let tokens = tokenizer.encode_ids(&text.repeat(32))?;
        let prompt: usize = 64;
        let steps: usize = 256;
        ensure!(tokens.len() >= prompt + steps + 4 && context >= prompt + steps + 2,
            "MiniMax width-exact prompt exceeds admitted context");
        let draft = backend.owners.storage.draft_prefix.as_ref().context("missing proof draft")?;
        let tap_pitch = draft.drafter.cfg.taps.len() * cfg.hidden * 2;
        let banks = backend.owners.storage.requests.borrow().banks();
        let read = |buffer: Buffer, bytes: usize| -> Result<Vec<u8>> {
            backend.drain()?;
            let buffer = super::super::cache::view(buffer, 0, bytes)?;
            let mut output = vec![0; bytes];
            library.cuda_set_device(buffer.device_id)?;
            library.copy_d2h(&mut output, buffer)?;
            Ok(output)
        };
        let hash = |bytes: &[u8]| format!("{:x}", Sha256::digest(bytes));
        let target_state = |placements: &[Placement], position: Option<usize>| -> Result<String> {
            let mut digest = Sha256::new();
            for bank in &banks {
                for layer in bank.layers.clone() {
                    for placement in placements {
                        let pages = position.map_or_else(|| placement.units.clone(), |position|
                            vec![placement.units[position / PAGE_ROWS]]);
                        for page in pages {
                            for (plane, _) in super::super::cache::PAGE_PLANES {
                                let buffer = bank.page(layer, plane, page)?;
                                digest.update(read(buffer, buffer.bytes)?);
                            }
                        }
                    }
                }
            }
            Ok(format!("{:x}", digest.finalize()))
        };
        let draft_state = |width: usize| -> Result<String> {
            let mut digest = Sha256::new();
            for slot in 0..width {
                digest.update((draft.frontier(slot)? as u64).to_le_bytes());
                for buffer in draft.drafter.window_planes(slot)? { digest.update(read(buffer, buffer.bytes)?); }
            }
            Ok(format!("{:x}", digest.finalize()))
        };
        let values = |bytes: &[u8]| bytes.chunks_exact(4)
            .map(|word| f32::from_le_bytes(word.try_into().unwrap())).collect::<Vec<_>>();
        let gap = |a: &[u8], b: &[u8]| -> Result<serde_json::Value> {
            let (a, b) = (values(a), values(b));
            ensure!(a.len() == b.len() && a.len() % cfg.vocab_size == 0
                && a.iter().chain(&b).all(|v| v.is_finite()), "invalid diagnostic logits");
            let (mut maximum, mut kl, mut agrees, mut count) = (0f64, 0f64, 0usize, 0usize);
            for (a, b) in a.chunks_exact(cfg.vocab_size).zip(b.chunks_exact(cfg.vocab_size)) {
                let top = |row: &[f32]| row.iter().enumerate().max_by(|x,y| x.1.total_cmp(y.1)).unwrap().0;
                agrees += usize::from(top(a) == top(b)); count += 1;
                let log_probs = |row: &[f32]| {
                    let peak = row.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
                    let norm = row.iter().map(|v| (f64::from(*v)-peak).exp()).sum::<f64>().ln() + peak;
                    row.iter().map(|v| f64::from(*v)-norm).collect::<Vec<_>>()
                };
                let (p,q) = (log_probs(a),log_probs(b));
                kl += p.iter().zip(q).map(|(p,q)| p.exp() * (p-q)).sum::<f64>();
                for (&x,&y) in a.iter().zip(b) { maximum = maximum.max((f64::from(x)-f64::from(y)).abs()); }
            }
            Ok(serde_json::json!({"position":64,"max_abs":maximum,"argmax_agreement":agrees as f64/count as f64,
                "kl":kl/count as f64}))
        };
        if std::env::var("CUTEAFD_MINIMAX_PROOF_BATCH_ONLY").as_deref() == Ok("on") {
            let pages = (0..4).map(|_| pool.alloc((prompt+8).div_ceil(PAGE_ROWS)))
                .collect::<std::result::Result<Vec<_>,_>>()?;
            let mut results = Vec::new();
            let mut outputs = Vec::new();
            for route in ["native", "eager"] {
                let mut batch_rows = Vec::new();
                let mut solo_rows = Vec::new();
                for members in [vec![0usize,1,2,3],vec![0],vec![1],vec![2],vec![3]] {
                    backend.drain()?;
                    for bank in &banks {
                        for layer in bank.layers.clone() {
                            for &slot in &members {
                                for &page in &pages[slot] {
                                    for (plane,_) in super::super::cache::PAGE_PLANES {
                                        let buffer=bank.page(layer,plane,page)?;
                                        bank.stream.device.run(|| {
                                            // SAFETY: exclusively owned proof pages and drained consumers.
                                            unsafe { library.cuda_zero_bytes_async(buffer,buffer.bytes,bank.stream.raw) }
                                        })?;
                                    }
                                }
                            }
                        }
                    }
                    let mut placements=members.iter().map(|&slot| -> Result<_> {
                        let mut placement=Placement { media_spans: Vec::new(), units:pages[slot].clone(),slot,len:0 };
                        family.bind(&mut placement).map_err(|e| anyhow::anyhow!(e.to_string()))?;
                        backend.identities.borrow_mut().insert(slot,(slot+1) as u64);
                        backend.prefill_device(&mut placement,&tokens[slot..slot+prompt],0,&pool)?;
                        Ok(placement)
                    }).collect::<Result<Vec<_>>>()?;
                    let initial=placements.iter().map(|p| target_state(std::slice::from_ref(p),None))
                        .collect::<Result<Vec<_>>>()?;
                    let input=members.iter().map(|&slot| tokens[slot+prompt]).collect::<Vec<_>>();
                    let mut rows=placements.iter_mut().zip(&input).map(|(p,t)| (p,std::slice::from_ref(t))).collect::<Vec<_>>();
                    let logits=if route == "native" { backend.verify_device(&mut rows,false,&pool)? }
                        else { backend.decode(&mut rows,&pool,true,false)? }.context("batch discriminator logits")?;
                    let bytes=logits.to_host(&library)?.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>();
                    drop(rows);
                    let taps=read(backend.owners.storage.verify_taps.as_ref().unwrap().buffer,members.len()*tap_pitch)?;
                    for (row,&slot) in members.iter().enumerate() {
                        let logits=bytes[row*cfg.vocab_size*4..(row+1)*cfg.vocab_size*4].to_vec();
                        let tap=taps[row*tap_pitch..(row+1)*tap_pitch].to_vec();
                        let state=target_state(std::slice::from_ref(&placements[row]),None)?;
                        let record=(logits,tap,state,initial[row].clone());
                        if members.len()==4 { batch_rows.push(record); }
                        else { ensure!(slot==solo_rows.len(),"singleton row order");solo_rows.push(record); }
                    }
                    for placement in &mut placements {
                        backend.cancel(placement)?;
                        family.discard(placement).map_err(|e| anyhow::anyhow!(e.to_string()))?;
                    }
                }
                let mut rows=Vec::new();
                for (row,(batch,solo)) in batch_rows.iter().zip(&solo_rows).enumerate() {
                    ensure!(batch.3==solo.3,"batch/singleton initial prefill bytes differ row {row}");
                    let diagnostic=gap(&solo.0,&batch.0)?;
                    rows.push(serde_json::json!({"row":row,"logits_identical":batch.0==solo.0,
                        "taps_identical":batch.1==solo.1,"kv_index_identical":batch.2==solo.2,
                        "gap":diagnostic,"batch_logits_sha256":hash(&batch.0),"solo_logits_sha256":hash(&solo.0)}));
                }
                results.push(serde_json::json!({"route":route,"rows":rows}));
                outputs.push((batch_rows,solo_rows));
            }
            let cross=(0..4).map(|row| -> Result<_> { Ok(serde_json::json!({"row":row,
                "singleton_native_vs_eager":gap(&outputs[0].1[row].0,&outputs[1].1[row].0)?,
                "batch_native_vs_eager":gap(&outputs[0].0[row].0,&outputs[1].0[row].0)?})) })
                .collect::<Result<Vec<_>>>()?;
            let report=serde_json::json!({"position":64,"reference":"same exact prefix alone at C1",
                "hf_reference":"unavailable: synthetic proof prompts have no published fidelity rows",
                "routes":results,"cross_route":cross});
            println!("MiniMax batch route discriminator: {report}");
            std::fs::write(Path::new(&std::env::var("CUTEAFD_GOLDEN_OUT")?).join("batch-route-discriminator.json"),
                serde_json::to_vec_pretty(&report)?)?;
            for pages in &pages { pool.release(pages); }
            return Ok(());
        }
        let mut reports = Vec::new();
        for width in [1usize, 4] {
            let pages = (0..width).map(|_| pool.alloc((prompt+steps+8).div_ceil(PAGE_ROWS)))
                .collect::<std::result::Result<Vec<_>,_>>()?;
            let mut reference = Vec::new();
            let mut first_plain = Vec::new();
            let mut expected_proposals: Option<Vec<Vec<u32>>> = None;
            let mut starting = None;
            let mut diagnostic = serde_json::Value::Null;
            for arm in ["plain", "staged", "eager-diagnostic"] {
                // Reset both inactive physical bytes and the draft window. No
                // arm can consume cache entries written by its predecessor.
                backend.drain()?;
                for bank in &banks {
                    for layer in bank.layers.clone() {
                        for page in pages.iter().flatten() {
                            for (plane, _) in super::super::cache::PAGE_PLANES {
                                let buffer = bank.page(layer, plane, *page)?;
                                bank.stream.device.run(|| {
                                    // SAFETY: proof owns every page; all readers drained above.
                                    unsafe { library.cuda_zero_bytes_async(buffer, buffer.bytes, bank.stream.raw) }
                                })?;
                            }
                        }
                    }
                }
                let mut placements = pages.iter().enumerate().map(|(slot,pages)| -> Result<_> {
                    let mut placement = Placement { media_spans: Vec::new(), units:pages.clone(),slot,len:0 };
                    family.bind(&mut placement).map_err(|e| anyhow::anyhow!(e.to_string()))?;
                    backend.identities.borrow_mut().insert(slot,(slot+1) as u64);
                    Ok(placement)
                }).collect::<Result<Vec<_>>>()?;
                let mut states = (0..width).map(|_| DraftState::default()).collect::<Vec<_>>();
                for (slot, placement) in placements.iter_mut().enumerate() {
                    backend.prefill_device(placement,&tokens[slot..slot+prompt],0,&pool)?;
                }
                let start = (target_state(&placements,None)?,draft_state(width)?);
                if let Some(expected) = &starting { ensure!(&start == expected,"independent arm prefill state differs"); }
                else { starting = Some(start); }
                let zero_steps = if arm == "eager-diagnostic" { 1 } else { steps };
                for step in 0..zero_steps {
                    let position = prompt+step;
                    let input = (0..width).map(|slot| tokens[slot+position]).collect::<Vec<_>>();
                    let logits = if arm == "plain" {
                        let submitted = placements.iter().zip(&input).enumerate().map(|(slot,(placement,token))|
                            Submitted { identity:(slot+1) as u64,placement:placement.clone(),tokens:vec![*token] }).collect::<Vec<_>>();
                        let mut copies = placements.clone();
                        let requests = copies.iter_mut().zip(&submitted).map(|(placement,sent)| ServingRequest {
                            identity:sent.identity,kind:ExpertV2SourceKind::Decode,start:position,
                            heads:position..position+1,tokens:&sent.tokens,placement }).collect::<Vec<_>>();
                        let logits = backend.execution.borrow_mut().decode(&backend.owners.storage,&cfg,&backend.schedule,
                            &requests,&mut backend.owners.graphs.borrow_mut(),true)?;
                        backend.drain()?;
                        // The plain route publishes its input unconditionally;
                        // emulate the normal committed tap consumer independently
                        // of the new staged-copy/accept path.
                        let owner = backend.owners.storage.ranks.last().unwrap().decode.borrow();
                        let taps = owner.target_taps().context("plain target taps")?.output();
                        for slot in 0..width {
                            // SAFETY: completed plain decode initialized live retained taps.
                            unsafe { draft.drafter.capture_concatenated(taps,slot,1,0,owner.stream.raw)?; }
                            draft.drafter.update(&[crate::shared::speculation::dflash::ContextRow {
                                tap_row:0,slot,position }])?;
                            draft.advance(slot,position,1)?;
                        }
                        drop(owner);
                        let mut rows = placements.iter_mut().zip(&input).map(|(p,t)| (p,std::slice::from_ref(t))).collect::<Vec<_>>();
                        publish(family,&mut rows,&submitted)?;
                        logits
                    } else {
                        let mut rows = placements.iter_mut().zip(&input).map(|(p,t)| (p,std::slice::from_ref(t))).collect::<Vec<_>>();
                        if arm == "staged" { backend.verify_device(&mut rows,false,&pool)? }
                        else { backend.decode(&mut rows,&pool,true,false)? }.context("proof logits missing")?
                    };
                    let bytes = logits.to_host(&library)?.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>();
                    ensure!(values(&bytes).iter().all(|v| v.is_finite()),"nonfinite proof logits");
                    let tap_buffer = if arm == "plain" {
                        backend.owners.storage.ranks.last().unwrap().decode.borrow().target_taps().unwrap().output()
                    } else { backend.owners.storage.verify_taps.as_ref().unwrap().buffer };
                    let taps = read(tap_buffer,width*tap_pitch)?;
                    if arm != "plain" {
                        let mut verified = placements.iter_mut().zip(&mut states).zip(&input).enumerate()
                            .map(|(slot,((placement,state),token))| scheduler::Verified {
                                placement,seq:state,start:position,history:(slot+1) as u64,
                                rows:std::slice::from_ref(token),first_row:slot,kept:Some((1,*token)),
                                finished:false,neural:true }).collect::<Vec<_>>();
                        backend.accept(&mut verified,false)?;
                    }
                    ensure!(placements.iter().all(|p| p.len == position+1)
                        && (0..width).all(|slot| draft.frontier(slot).ok() == Some(position+1)),"proof commit frontier differs");
                    let record = (hash(&bytes),hash(&taps),target_state(&placements,Some(position))?,draft_state(width)?);
                    if arm == "plain" {
                        if step == 0 { first_plain = bytes; }
                        reference.push(record);
                    } else if arm == "staged" {
                        ensure!(record == reference[step],"native staged/plain logits/taps/KV/index/draft differ C{width} step {step}");
                    } else {
                        diagnostic = gap(&first_plain,&bytes)?;
                        diagnostic["target_state_identical"] = serde_json::json!(record.2 == reference[0].2);
                        diagnostic["draft_state_identical"] = serde_json::json!(record.3 == reference[0].3);
                    }
                }
                if arm != "eager-diagnostic" {
                    // Verify one real proposal immediately after the final
                    // zero-proposal step, from each independently built state.
                    let position = prompt+steps;
                    let seqs = (0..width).map(|slot| crate::shared::speculation::dflash::DraftSeq {
                        slot,anchor:tokens[slot+position],position,valid_from:0 }).collect::<Vec<_>>();
                    let head = backend.owners.storage.ranks.last().unwrap().weights.operand("language_model.lm_head.weight")?.buffer;
                    let proposals = draft.drafter.draft_device(&seqs,&backend.owners.storage.embedding,
                        crate::shared::speculation::dflash::TargetHead::Bf16Buffer(&head))?
                        .into_iter().map(|p| p.tokens.into_iter().take(1).collect::<Vec<_>>()).collect::<Vec<_>>();
                    ensure!(proposals.iter().all(|p| p.len() == 1),"missing continuation proposal");
                    if let Some(expected) = &expected_proposals { ensure!(&proposals == expected,"following proposal ids differ"); }
                    else { expected_proposals = Some(proposals.clone()); }
                    let input = proposals.iter().enumerate().map(|(slot,p)| vec![tokens[slot+position],p[0]]).collect::<Vec<_>>();
                    let mut rows = placements.iter_mut().zip(&input).map(|(p,t)| (p,t.as_slice())).collect::<Vec<_>>();
                    let logits = backend.verify_device(&mut rows,true,&pool)?.context("continuation logits")?;
                    let bytes = logits.to_host(&library)?.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>();
                    drop(rows);
                    let taps = read(backend.owners.storage.verify_taps.as_ref().unwrap().buffer,width*2*tap_pitch)?;
                    let mut verified = placements.iter_mut().zip(&mut states).zip(&input).enumerate()
                        .map(|(slot,((placement,state),tokens))| scheduler::Verified {
                            placement,seq:state,start:position,history:(slot+1) as u64,rows:tokens,
                            first_row:slot*2,kept:Some((2,tokens[1])),finished:false,neural:true }).collect::<Vec<_>>();
                    backend.accept(&mut verified,true)?;
                    drop(verified);
                    let record = (hash(&bytes),hash(&taps),target_state(&placements,None)?,draft_state(width)?);
                    if arm == "plain" { reference.push(record); }
                    else { ensure!(record == reference[steps],"following proposal verify logits/taps/target/draft state differ C{width}"); }
                }
                for placement in &mut placements {
                    backend.cancel(placement)?;
                    family.discard(placement).map_err(|e| anyhow::anyhow!(e.to_string()))?;
                }
            }
            reports.push(serde_json::json!({"width":width,"zero_proposal_steps":steps,
                "plain_decode_logits_taps_and_state_byte_identical":true,"following_proposal_verify_byte_identical":true,
                "eager_verify_position64":diagnostic,"records":reference}));
            for pages in &pages { pool.release(pages); }
        }
        let report = serde_json::json!({"reference":"plain native decode, identical independent prefill",
            "target_state":"physical KV/index page bytes","draft_state":"full physical window plus frontier",
            "widths":reports});
        println!("MiniMax native zero-proposal exactness: {report}");
        if let Ok(output) = std::env::var("CUTEAFD_GOLDEN_OUT") {
            std::fs::write(Path::new(&output).join("native-zero-proposal.json"),serde_json::to_vec_pretty(&report)?)?;
        }
        Ok(())
    }

    #[test]
    fn native_decode_taps_retain_only_live_rows_across_producer_reuse() -> Result<()> {
        let fixture = Fixture::build_for_tests()?;
        let library = fixture.load()?;
        fixture.immediate_copies(&library)?;
        let device = Device { library: &library, id: 0 };
        let pitch = 5 * 6144 * 2;
        let source = Allocation::new(device, 16 * pitch, Tag::workspace("test/decode-taps"))?;
        let target = Allocation::new(device, 32 * pitch, Tag::workspace("test/verify-taps"))?;
        for rows in [1, 4, 16] {
            let bytes = (0..source.buffer.bytes).map(|i| (i / pitch * 37 + i % pitch) as u8).collect::<Vec<_>>();
            library.copy_h2d(source.buffer, &bytes)?;
            library.copy_h2d(target.buffer, &vec![197; target.buffer.bytes])?;
            retain_decode_taps(&library, source.buffer, target.buffer, rows, pitch, std::ptr::null_mut())?;
            library.copy_h2d(source.buffer, &vec![83; source.buffer.bytes])?;
            let mut actual = vec![0; target.buffer.bytes];
            library.copy_d2h(&mut actual, target.buffer)?;
            assert_eq!(&actual[..rows * pitch], &bytes[..rows * pitch]);
            assert!(actual[rows * pitch..].iter().all(|&byte| byte == 197));
        }
        for (rows, stride) in [(0, pitch), (17, pitch), (1, 0), (usize::MAX, 2)] {
            assert!(retain_decode_taps(&library, source.buffer, target.buffer, rows, stride, std::ptr::null_mut()).is_err());
        }
        let wrong_device = Buffer { device_id: 1, ..target.buffer };
        assert!(retain_decode_taps(&library, source.buffer, wrong_device, 1, pitch, std::ptr::null_mut()).is_err());
        Ok(())
    }

    #[test]
    fn zero_proposal_rounds_use_native_decode_and_stage_before_publication() {
        let source = include_str!("backend.rs");
        let verify = source.split_once("    fn verify_device(").unwrap().1
            .split_once("    fn verify_device_ungraphed(").unwrap().0;
        assert!(!verify.contains("verify_staged"));
        assert!(verify.contains("self.decode(sequences, pool, true, true)"));
        assert!(!verify.contains("draft_prefix.is_some()"));
        let decode = source.split_once("    fn decode(").unwrap().1
            .split_once("impl Drop").unwrap().0;
        let staged = decode.split_once("if let Some(draft) = draft").unwrap().1
            .split_once("// Non-speculative input").unwrap().0;
        assert!(staged.contains("retain_decode_taps("));
        assert!(staged.contains("Some(submitted)"));
        assert!(staged.contains("return Ok(Some(logits))"));
        assert!(!staged.contains("publish("));
    }

    #[test]
    fn diagnostic_logit_tiles_survive_bounded_head_reuse() -> Result<()> {
        let fixture=Fixture::build_for_tests()?;
        let library=fixture.load()?;
        fixture.immediate_copies(&library)?;
        let device=Device { library:&library,id:0 };
        let head=Allocation::new(device,4 * 8 * 4,Tag::workspace("test/head"))?;
        let retained=Allocation::new(device,32 * 8 * 4,Tag::workspace("test/retained-logits"))?;
        let mut expected=Vec::new();
        for first in (0..32).step_by(4) {
            let bytes=(first * 8..(first + 4) * 8).flat_map(|n| (n as f32).to_le_bytes()).collect::<Vec<_>>();
            library.copy_h2d(head.buffer,&bytes)?;
            retain_logit_tile(&library,retained.buffer,first,8,DeviceLogits { ptr:head.buffer.ptr,
                rows:4,vocab:8,stride:8,stream:std::ptr::null_mut(),greedy:None })?;
            expected.extend(bytes);
        }
        library.copy_h2d(head.buffer,&vec![197;head.buffer.bytes])?;
        let mut actual=vec![0;retained.buffer.bytes];
        library.copy_d2h(&mut actual,retained.buffer)?;
        assert_eq!(actual,expected);
        assert!(retain_logit_tile(&library,retained.buffer,31,8,DeviceLogits { ptr:head.buffer.ptr,
            rows:4,vocab:8,stride:8,stream:std::ptr::null_mut(),greedy:None }).is_err());
        Ok(())
    }

    #[test]
    fn prefix_scheduler_argument_is_boot_concurrency_not_decode_lanes() -> Result<()> {
        for concurrency in [1, 16, 64] {
            validate_prefix_concurrency(concurrency, concurrency)?;
            assert!(validate_prefix_concurrency(concurrency + 1, concurrency).is_err());
        }
        assert!(validate_prefix_concurrency(0, 0).is_err());
        assert!(validate_prefix_concurrency(1, 16).is_err());
        Ok(())
    }

    #[test]
    fn packing_modes_use_shared_grouping_without_silently_changing_mode() {
        use crate::shared::prefill_packing::{self, PackingMode};
        let chunks = [(0, 128), (256, 128)];
        for (mode, expected) in [(PackingMode::Exact, vec![0..1, 1..2]),
            (PackingMode::Unrestricted, vec![0..2])] {
            let limits = Limits { rows: packing_rows(mode, 2752), chunk_rows: 2752,
                max_context: 4096, taps: None };
            let groups = prefill_packing::groups(chunks.len(), |range|
                prefill_packing::plan(&chunks[range], &limits).is_ok());
            assert_eq!(groups, expected);
        }
    }

    #[test]
    fn message_start_is_the_served_checkpoints_single_added_token() -> Result<()> {
        assert_eq!(MESSAGE_STARTS, &["]~b]"]);
        let snapshot = std::path::PathBuf::from(std::env::var_os("CUTEAFD_DEPLOYMENT_TEST_HF_HOME")
            .expect("CUTEAFD_DEPLOYMENT_TEST_HF_HOME is required"))
            .join("hub/models--MiniMaxAI--MiniMax-M3.1-Flash/snapshots/54e4e69dfc452b1e94a426c24a4fdeeaa8e3ad5e");
        ensure!(snapshot.join("tokenizer.json").is_file(), "required MiniMax tokenizer missing");
        assert_eq!(crate::shared::prefix::marker_ids(&snapshot, MESSAGE_STARTS)?, vec![200_019]);
        Ok(())
    }

    fn family(library: &NativeLibrary) -> Result<(Rc<super::super::cache::PageBank<'_>>, MiniMaxPrefix<'_>)> {
        let mut cfg = MiniMaxM3Config::from_hf(&serde_json::from_str(include_str!(
            "../../../../cuteafd-loader/src/families/minimax_m3/fixtures/config.json"))?)?;
        cfg.layers = 2;
        let stream = Rc::new(Stream::new(Device { library, id: 1 })?);
        let scratch = Rc::new(Allocation::new(stream.device, 256, Tag::workspace("minimax/test/prefix"))?);
        let bank = Rc::new(super::super::cache::PageBank::new(vec![stream], &cfg,
            RequestCapacity::new(4, 64)?, MsaCache::new(2, 24, 4, 257)?, MsaRope::new(&cfg, 257)?, 0..2)?);
        let family = MiniMaxPrefix::new(vec![bank.clone()], scratch)?;
        Ok((bank, family))
    }
    fn read(bank: &super::super::cache::PageBank<'_>, buffer: Buffer) -> Result<Vec<u8>> {
        bank.stream.drain()?;
        let mut bytes = vec![0; buffer.bytes];
        bank.stream.device.run(|| bank.stream.device.library.copy_d2h(&mut bytes, buffer))?;
        Ok(bytes)
    }
    fn row(bank: &super::super::cache::PageBank<'_>, placement: &Placement, layer: usize,
        plane: &str, position: usize, head: usize) -> Result<Buffer> {
        let page = bank.page(layer, plane, placement.units[position / PAGE_ROWS])?;
        let local = position % PAGE_ROWS;
        let (offset, bytes) = if plane == "index" {
            (if head == 0 { local * 128 } else { 8192 + local * 4 }, if head == 0 { 128 } else { 4 })
        } else { ((local * 4 + head) * 128, 128) };
        super::super::cache::view(page, offset, bytes)
    }
    // Explicit CPU arithmetic model, not a successful CUDA-kernel stub. Each
    // new row reads prior native cache bytes, so a bad restore changes its suffix.
    fn model_append(bank: &super::super::cache::PageBank<'_>, placement: &Placement, tokens: &[u32]) -> Result<()> {
        bank.validate(placement, placement.len + tokens.len())?;
        for layer in bank.layers.clone() {
            for (offset, &token) in tokens.iter().enumerate() {
                let position = placement.len + offset;
                let previous = if position == 0 { 0 } else {
                    read(bank, row(bank, placement, layer, "keys", position - 1, 0)?)?[0]
                };
                let value = previous.wrapping_mul(31).wrapping_add(token as u8).wrapping_add(layer as u8 + 1);
                for plane in ["keys", "values", "index"] {
                    for head in 0..if plane == "index" { 2 } else { 4 } {
                        let buffer = row(bank, placement, layer, plane, position, head)?;
                        let bytes = (0..buffer.bytes).map(|column| value.wrapping_add((head * 17 + column) as u8))
                            .collect::<Vec<_>>();
                        bank.stream.device.run(|| bank.stream.device.library.copy_h2d(buffer, &bytes))?;
                    }
                }
            }
        }
        Ok(())
    }
    fn binding(bank: &super::super::cache::PageBank<'_>, placement: &Placement) -> Result<()> {
        let length = super::super::cache::view(bank.plane("lengths")?, placement.slot * 4, 4)?;
        assert_eq!(read(bank, length)?, (placement.len as i32).to_le_bytes());
        let width = bank.layout.table_width as usize;
        let table = super::super::cache::view(bank.plane("page_table")?, placement.slot * width * 4, width * 4)?;
        let bytes = read(bank, table)?;
        for (index, entry) in bytes.chunks_exact(4).enumerate() {
            assert_eq!(i32::from_le_bytes(entry.try_into()?), placement.units.get(index).copied().unwrap_or(0) as i32);
        }
        Ok(())
    }
    fn append(family: &MiniMaxPrefix<'_>, bank: &super::super::cache::PageBank<'_>,
        placement: &mut Placement, tokens: &[u32]) -> Result<()> {
        model_append(bank, placement, tokens)?;
        let receipt = Submitted { identity: 17, placement: placement.clone(), tokens: tokens.to_vec() };
        publish(family, &mut [(placement, tokens)], &[receipt])?;
        family.drain().map_err(|error| anyhow::anyhow!(error.to_string()))?;
        binding(bank, placement)
    }

    #[test]
    fn plain_decode_publishes_exact_prefix_state_without_accept() -> Result<()> {
        let fixture = Fixture::build_for_tests()?;
        let library = fixture.load()?;
        fixture.immediate_copies(&library)?;
        let (bank, family) = family(&library)?;
        let mut placement = Placement { media_spans: Vec::new(), units: vec![7, 3, 11], slot: 2, len: 0 };
        let tokens = (0..65).map(|i| (i * 37 % 251) as u32).collect::<Vec<_>>();
        append(&family, &bank, &mut placement, &tokens[..64])?;
        append(&family, &bank, &mut placement, &tokens[64..])?;
        assert_eq!(placement.len, 65);
        let receipt = Submitted { identity: 17, placement: Placement { len: 64, ..placement.clone() }, tokens: vec![tokens[64]] };
        receipt.validate_published(65, None, true)?;
        binding(&bank, &placement)?;
        Ok(())
    }

    fn assert_live_rows_equal(bank: &super::super::cache::PageBank<'_>, a: &Placement,
        b: &Placement, positions: Range<usize>) -> Result<()> {
        assert_eq!(a.len, b.len);
        for layer in bank.layers.clone() {
            for position in positions.clone() {
                for plane in ["keys", "values", "index"] {
                    for head in 0..if plane == "index" { 2 } else { 4 } {
                        assert_eq!(read(bank, row(bank, a, layer, plane, position, head)?)?,
                            read(bank, row(bank, b, layer, plane, position, head)?)?,
                            "layer {layer} plane {plane} row {position} head {head}");
                    }
                }
            }
        }
        Ok(())
    }
    // CPU-model logits deliberately depend on every plane of the last causal
    // row. This exercises snapshot continuation, not native CUDA numerics.
    fn model_logits(bank: &super::super::cache::PageBank<'_>, placement: &Placement) -> Result<Vec<u8>> {
        let mut output = vec![0u8; 32];
        for layer in bank.layers.clone() {
            for plane in ["keys", "values", "index"] {
                for head in 0..if plane == "index" { 2 } else { 4 } {
                    for (column, byte) in read(bank, row(bank, placement, layer, plane, placement.len - 1, head)?)?
                        .into_iter().enumerate() {
                        let index = (column + layer + head) % output.len();
                        output[index] = output[index].wrapping_mul(17).wrapping_add(byte);
                    }
                }
            }
        }
        Ok(output)
    }
    fn continuation(resume: usize, end: usize) -> Result<()> {
        let fixture = Fixture::build_for_tests()?;
        let library = fixture.load()?;
        fixture.immediate_copies(&library)?;
        let (bank, family) = family(&library)?;
        let mut cache: PrefixCache<CudaCopyEngine<'_>> = PrefixCache::new(family.layout(), PrefixConfig {
            entries: 8, mark_slots: 0, keep_logits: true, min_tokens: 1 }, None)?;
        let tokens = (0..end).map(|i| (i * 73 + i / 7) as u32 % 251).collect::<Vec<_>>();
        let mut original = cache.admit(&family, &tokens[..65], 256, false,
            |units| Placement { media_spans: Vec::new(), units, slot: 0, len: 0 })?.placement;
        append(&family, &bank, &mut original, &tokens[..65])?;
        assert!(cache.capture(&family, SnapshotKind::Prompt, &tokens[..65], &original,
            After::from_logits(&[0.0, 1.0], true))?);
        let kind = if resume == 129 {
            append(&family, &bank, &mut original, &tokens[65..129])?;
            assert!(cache.capture(&family, SnapshotKind::Turn, &tokens[..129], &original,
                After::from_logits(&[1.0, 0.0], true))?);
            SnapshotKind::Turn
        } else { assert_eq!(resume, 65); SnapshotKind::Prompt };
        // The old live request dirties more than forty rows after its snapshot.
        append(&family, &bank, &mut original, &[239; 41])?;
        cache.release(&family, &original.units)?;
        let hit = cache.admit(&family, &tokens, 256, false,
            |units| Placement { media_spans: Vec::new(), units, slot: 1, len: 0 })?;
        assert_eq!(hit.resume, resume);
        assert_eq!(hit.source.as_ref().unwrap().kind, kind);
        assert!(hit.source.as_ref().unwrap().plan.exact());
        let mut restored = hit.placement;
        binding(&bank, &restored)?;
        let mut cold = cache.admit_cold(&family, tokens.len(), 256,
            |units| Placement { media_spans: Vec::new(), units, slot: 2, len: 0 })?.placement;
        append(&family, &bank, &mut cold, &tokens[..resume])?;
        assert_live_rows_equal(&bank, &restored, &cold, 0..resume)?;
        assert_eq!(model_logits(&bank, &restored)?, model_logits(&bank, &cold)?);
        assert!(end - resume > 40 && end / PAGE_ROWS > resume / PAGE_ROWS);
        for position in resume..end {
            append(&family, &bank, &mut restored, &tokens[position..position + 1])?;
            append(&family, &bank, &mut cold, &tokens[position..position + 1])?;
            assert_eq!(model_logits(&bank, &restored)?, model_logits(&bank, &cold)?, "position {position}");
            assert_live_rows_equal(&bank, &restored, &cold, position..position + 1)?;
        }
        assert_live_rows_equal(&bank, &restored, &cold, 0..end)?;
        // Noncausal recycled tails are not logical state. Native scalar-K tail
        // masking passed fork PR #297 (722d4963); never zero pages on restore.
        // At an aligned end, every byte is live and the complete pages must match.
        if end % PAGE_ROWS == 0 {
            for (&restored, &cold) in restored.units.iter().zip(&cold.units) {
                for layer in bank.layers.clone() {
                    for (plane, _) in super::super::cache::PAGE_PLANES {
                        assert_eq!(read(&bank, bank.page(layer, plane, restored)?)?,
                            read(&bank, bank.page(layer, plane, cold)?)?, "layer {layer} plane {plane}");
                    }
                }
            }
        }
        assert_eq!((restored.len, cold.len), (end, end));
        cache.release(&family, &restored.units)?;
        cache.release(&family, &cold.units)?;
        cache.clear(&family)?;
        assert_eq!(cache.pool().used(), 0);
        Ok(())
    }

    #[test]
    fn cpu_model_prompt65_continuation_matches_cold_per_step_and_aligned_pages() -> Result<()> {
        continuation(65, 201)?;
        continuation(65, 256)
    }
    #[test]
    fn cpu_model_deepest_turn129_continuation_matches_cold_per_step_and_aligned_pages() -> Result<()> {
        continuation(129, 201)?;
        continuation(129, 256)
    }

    #[test]
    fn accept_checks_original_identity_pages_start_and_tokens_not_mutated_length() -> Result<()> {
        let sent = Submitted { identity: 91, placement: Placement { media_spans: Vec::new(), units: vec![3, 7], slot: 2, len: 63 }, tokens: vec![101] };
        let mut actual = sent.placement.clone();
        actual.len = 64;
        sent.validate(91, &actual, 63, &[101])?;
        sent.validate_published(actual.len, Some((1, 102)), false)?;
        sent.validate_published(actual.len, None, true)?;
        assert!(sent.validate_published(actual.len, None, false).is_err());
        assert!(sent.validate_published(65, Some((1, 102)), false).is_err());
        assert!(sent.validate(92, &actual, 63, &[101]).is_err());
        assert!(sent.validate(91, &actual, 64, &[101]).is_err());
        assert!(sent.validate(91, &actual, 63, &[103]).is_err());
        actual.units.reverse();
        assert!(sent.validate(91, &actual, 63, &[101]).is_err());
        Ok(())
    }
}
