//! Text-only MiniMax adapter; admission, lanes and selection stay in the shared scheduler.
use super::{cache::{Placement, PAGE_ROWS}, engine::{Requests, ServingPrefill, ServingRequest},
    prefix::MiniMaxPrefix, owners::Owners};
use crate::shared::{draft::routes::RoundRoutes, prefix::{CudaCopyEngine, PrefixArgs, Toggle},
    prefill_packing::{Limits, MemberOutcome}, probe,
    qwen::scheduler::{self, Backend}, token_io::{DeviceLogits, TokenSelector}};
use anyhow::{ensure, Context, Result};
use cuteafd_api::openai::NativeRequest;
use cuteafd_engine::{media::{EncodeJob, EmbeddingCache, MediaKeys, RequestMedia},
    prefix::{PrefixCache, PrefixConfig, PrefixFamily, RefPagePool}};
use cuteafd_loader::families::minimax_m3::MiniMaxM3Config;
use cuteafd_transport::ExpertV2SourceKind;
use std::{cell::{Cell, RefCell}, collections::{BTreeMap, BTreeSet}, ops::Range, path::Path};

const MESSAGE_STARTS: &[&str] = &["]~b]"];

fn validate_prefix_concurrency(requested: usize, admitted: usize) -> Result<()> {
    ensure!(requested > 0 && requested == admitted,
        "MiniMax prefix concurrency differs from admitted request slots");
    Ok(())
}

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub(crate) enum BackendError {
    #[error("MiniMax neural/speculative drafts are not qualified")]
    DraftsNotQualified,
    #[error("MiniMax bundled media serving is not qualified")]
    MediaNotQualified,
    #[error("MiniMax shared-scheduler backend is qualified for one RTX only")]
    TwoRtxNotQualified,
    #[error("MiniMax backend is terminal after a failed native submission")]
    Terminal,
}

#[derive(Default)]
pub(crate) struct NoDraft;
impl scheduler::DraftState for NoDraft {
    fn pending_rows(&self) -> usize { 0 }
    fn close(&mut self, _: u32) {}
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
}

impl<'e, 'a> MiniMaxBackend<'e, 'a> {
    pub fn new(owners: &'e Owners<'a>, cfg: &'e MiniMaxM3Config, execution: ServingPrefill<'a>,
        served_context: usize, host: Option<cuteafd_hostcache::config::Config>) -> Result<Self> {
        let storage = &owners.storage;
        ensure!(storage.ranks.len() == 1, BackendError::TwoRtxNotQualified);
        let state = storage.requests.borrow();
        let bank = &state.ranks[0].bank;
        ensure!(served_context > 0 && served_context <= bank.rope.positions as usize
            && served_context <= bank.layout.pool_pages as usize * PAGE_ROWS,
            "MiniMax served context exceeds admitted cache");
        let rank = &storage.ranks[0];
        let stream = rank.decode.borrow().stream.raw;
        ensure!(rank.layout.layers.iter().copied().eq(0..cfg.layers)
            && state.ranks[0].prefill.len() == rank.prefill.len()
            && rank.prefill.iter().all(|owner| owner.borrow().stream.raw == stream)
            && bank.readers().iter().all(|reader| reader.raw == stream),
            super::engine::ServingError::DifferentStreams);
        let schedule = Requests::new(state.capacity, served_context, cfg.layers, rank.prefill.len())?;
        drop(state);
        Ok(Self { owners, cfg, execution: RefCell::new(execution), schedule, context: served_context,
            host, profile: RefCell::new([0.0; 2]), identities: RefCell::new(BTreeMap::new()),
            next_identity: Cell::new(1), pending: RefCell::new(None), terminal: Cell::new(false),
            selector_receipts: RefCell::new(None) })
    }

    fn live(&self) -> Result<()> { ensure!(!self.terminal.get(), BackendError::Terminal); Ok(()) }
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
        match result { Ok(value) => { drain?; Ok(value) }, Err(error) => Err(error) }
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
        ensure!(!submitted.is_empty() && submitted.len() <= state.capacity.decode.requests as usize,
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
        retain_receipt: bool) -> Result<Option<DeviceLogits>> {
        let submitted = self.validate(sequences, pool)?;
        let mut placements = submitted.iter().map(|request| request.placement.clone()).collect::<Vec<_>>();
        let requests = placements.iter_mut().zip(&submitted).map(|(placement, request)| ServingRequest {
            identity: request.identity, kind: ExpertV2SourceKind::Decode, start: request.placement.len,
            heads: request.placement.len..request.placement.len + request.tokens.len(),
            tokens: &request.tokens, placement }).collect::<Vec<_>>();
        let result = self.execution.borrow_mut().decode(&self.owners.storage, self.cfg, &self.schedule, &requests);
        let logits = self.completed(result)?;
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
    type Draft = NoDraft;
    type History = u64;
    type Coordinates = ();
    type Encoder = crate::shared::qwen::Encoder;
    type Meta = ();
    type Prefix = MiniMaxPrefix<'a>;

    fn geometry(&self) -> scheduler::Geometry {
        let state = self.owners.storage.requests.borrow();
        scheduler::Geometry { family: "minimax_m3", vocab_size: self.cfg.vocab_size, max_context: self.context,
            slots: state.capacity.decode.requests as usize, prefill_rows: state.capacity.prefill.rows as usize,
            decode_rows: state.capacity.decode.rows as usize, }
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
            mark_slots: 0, keep_logits: true, min_tokens: args.prefix_cache_min_tokens }, host)?;
        cuteafd_bench::context::set_kv((layout.pages * layout.page_rows) as u64,
            Some((layout.pages * layout.page_rows) as u64), layout.pages as u64, family.record_format(), host_bytes);
        Ok((family, cache))
    }
    fn prepare(&self, job: NativeRequest, tokens: Vec<u32>) -> Result<(scheduler::Prompt, RequestMedia, Vec<EncodeJob>)> {
        ensure!(job.media.is_empty(), BackendError::MediaNotQualified);
        self.owners.storage.embedding.check(&tokens)?;
        let keys = MediaKeys::new(&tokens, u32::try_from(self.cfg.vocab_size)?, &[])?;
        let media = RequestMedia::new(Vec::new(), self.cfg.hidden, tokens.len())?;
        Ok((scheduler::Prompt { job, tokens, keys }, media, Vec::new()))
    }
    fn probe_features(&self, prompt: &scheduler::Prompt, media: &mut RequestMedia,
        _: &mut EmbeddingCache, _: &Path) -> Result<()> {
        ensure!(prompt.job.media.is_empty() && media.spans().is_empty(), BackendError::MediaNotQualified);
        Ok(())
    }
    fn coordinates(&self, _: &[u32], job: &NativeRequest, media: &RequestMedia) -> Result<()> {
        ensure!(job.media.is_empty() && media.spans().is_empty(), BackendError::MediaNotQualified);
        Ok(())
    }
    fn placement(&self, units: Vec<u32>, slot: i32) -> Placement { Placement { units, slot: slot as usize, len: 0 } }
    fn bind_request(&self, placement: &mut Placement, _: (), media: RequestMedia,
        tokens: &[u32], resume: usize) -> Result<()> {
        self.idle()?;
        ensure!(media.spans().is_empty(), BackendError::MediaNotQualified);
        ensure!(placement.len == resume && resume <= tokens.len(), "MiniMax restored request frontier differs");
        self.owners.storage.requests.borrow().ranks[0].bank.validate(placement, resume)?;
        let identity = self.next_identity.get();
        self.next_identity.set(identity.checked_add(1).context("MiniMax request identity exhausted")?);
        self.identities.borrow_mut().insert(placement.slot, identity);
        Ok(())
    }
    fn history(&self, placement: &Placement) -> u64 { self.identities.borrow().get(&placement.slot).copied().unwrap_or(0) }
    fn restore_draft(&self, _: &Placement, _: usize, _: &[u32], _: &mut NoDraft) -> Result<()> {
        Err(BackendError::DraftsNotQualified.into())
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
        Some(Limits { rows: self.geometry().prefill_rows, chunk_rows: self.geometry().prefill_rows,
            max_context: self.context, taps: None })
    }
    fn prefill_packed(&self, sequences: &mut [scheduler::PrefillSeq<'_, Placement, NoDraft>], pool: &RefPagePool,
        on_logits: &mut dyn FnMut(usize, DeviceLogits) -> MemberOutcome) -> Result<Vec<Result<()>>> {
        ensure!(sequences.iter().all(|sequence| !sequence.neural), BackendError::DraftsNotQualified);
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
        outcomes.into_iter().map(|outcome| outcome.context("MiniMax packed member head missing")).collect()
    }
    fn verify_device(&self, sequences: &mut [(&mut Placement, &[u32])], speculative: bool,
        pool: &RefPagePool) -> Result<Option<DeviceLogits>> {
        ensure!(!speculative, BackendError::DraftsNotQualified);
        self.decode(sequences, pool, true)
    }
    fn verify_device_ungraphed(&self, sequences: &mut [(&mut Placement, &[u32])], speculative: bool,
        kind: scheduler::UngraphedKind, pool: &RefPagePool) -> Result<Option<DeviceLogits>> {
        ensure!(!speculative, BackendError::DraftsNotQualified);
        if kind == scheduler::UngraphedKind::Decode { return self.decode(sequences, pool, true); }
        let total: usize = sequences.iter().map(|(_, tokens)| tokens.len()).sum();
        ensure!(sequences.len() == 1 && total <= self.geometry().decode_rows,
            "MiniMax diagnostic scoring requires one bounded sequence");
        if total == 1 { return self.decode(sequences, pool, false); }
        let start = sequences[0].0.len;
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
    fn draft_geometry(&self, _: &Path, _: usize) -> Result<cuteafd_core::PolicyGeometry> { Err(BackendError::DraftsNotQualified.into()) }
    fn arm_draft_probe(&self) -> Result<()> { Err(BackendError::DraftsNotQualified.into()) }
    fn prefill_draft(&self, _: &Placement, _: usize, _: &[u32], _: Option<u32>, _: &mut NoDraft) -> Result<()> {
        Err(BackendError::DraftsNotQualified.into())
    }
    fn draft(&self, _: &mut [scheduler::DraftSeq<'_, Placement, NoDraft>], _: &mut scheduler::DraftTiming,
        _: &mut TokenSelector<'_>) -> Result<Vec<Vec<u32>>> { Err(BackendError::DraftsNotQualified.into()) }
    fn accept(&self, verified: &mut [scheduler::Verified<'_, Placement, NoDraft, u64>], speculative: bool) -> Result<()> {
        self.live()?;
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
        self.completed(checked)
    }
    fn cancel(&self, placement: &Placement) -> Result<()> {
        // Drain before removing evidence or returning pages/slots to the scheduler.
        self.drain()?;
        if self.pending.borrow().as_ref().is_some_and(|wave| wave.iter().any(|sent| sent.placement.slot == placement.slot)) {
            self.pending.borrow_mut().take();
        }
        self.identities.borrow_mut().remove(&placement.slot);
        Ok(())
    }
    fn probe_begin(&self, _: usize) {}
    fn probe_end(&self) {}
    fn probe_finish<T>(&self, _: impl FnOnce(&RoundRoutes, &[Option<f64>]) -> T) -> Option<T> { None }
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
    fn message_start_is_the_served_checkpoints_single_added_token() -> Result<()> {
        assert_eq!(MESSAGE_STARTS, &["]~b]"]);
        let snapshot = Path::new("/mnt/sparknest/hf-home/hub/models--MiniMaxAI--MiniMax-M3.1-Flash/snapshots/54e4e69dfc452b1e94a426c24a4fdeeaa8e3ad5e");
        if !snapshot.join("tokenizer.json").is_file() { return Ok(()); }
        assert_eq!(crate::shared::prefix::marker_ids(snapshot, MESSAGE_STARTS)?, vec![200_019]);
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
        let mut placement = Placement { units: vec![7, 3, 11], slot: 2, len: 0 };
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
            |units| Placement { units, slot: 0, len: 0 })?.placement;
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
            |units| Placement { units, slot: 1, len: 0 })?;
        assert_eq!(hit.resume, resume);
        assert_eq!(hit.source.as_ref().unwrap().kind, kind);
        assert!(hit.source.as_ref().unwrap().plan.exact());
        let mut restored = hit.placement;
        binding(&bank, &restored)?;
        let mut cold = cache.admit_cold(&family, tokens.len(), 256,
            |units| Placement { units, slot: 2, len: 0 })?.placement;
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
        let sent = Submitted { identity: 91, placement: Placement { units: vec![3, 7], slot: 2, len: 63 }, tokens: vec![101] };
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
