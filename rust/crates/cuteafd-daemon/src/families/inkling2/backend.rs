//! Inkling adapter for the shared request scheduler; no family scheduling loop.
use super::{cache::{InklingPrefix, Placement}, engine::Execution, prefill::HeadRows};
use crate::shared::{prefix::{CudaCopyEngine, PrefixArgs}, probe,
    prefill_packing::{Limits, MemberOutcome}, prefill_pipeline::{sequence_units, Unit, UnitSegment},
    qwen::scheduler::{self, Backend}, token_io::{DeviceLogits, TokenSelector}};
use anyhow::{ensure, Context, Result};
use cuteafd_api::openai::NativeRequest;
use cuteafd_engine::{media::{EncodeJob, EmbeddingCache, RequestMedia},
    prefix::{PrefixCache, PrefixConfig, PrefixFamily, RefPagePool}};
use std::{cell::RefCell, collections::BTreeSet, path::Path};

fn check_writable(placement: &Placement, rows: usize, context: usize, pool: &RefPagePool) -> Result<()> {
    let end = placement.len.checked_add(rows).context("Inkling append extent overflow")?;
    ensure!(rows > 0 && end <= context && end.div_ceil(64) <= placement.units.len(), "Inkling append exceeds reserved pages");
    let mut writable = BTreeSet::new();
    for position in placement.len..end {
        let page = placement.units[position/64];
        ensure!((page as usize) < pool.capacity() && pool.refs(page) == 1 && writable.insert((page, position%64)),
            "Inkling append requires private nonaliased COW pages");
    }
    Ok(())
}

/// One request's append units. A unit gets a head segment only when it holds
/// a row `heads` asks for: the last unit for serving, and those reaching the
/// scored suffix (1-based `first`) for streamed scoring.
fn append_units(start: usize, rows: usize, unit_rows: usize, waves: usize, layers: usize,
    heads: Option<HeadRows>) -> Result<Vec<Unit>> {
    let end = start+rows;
    let mut units = sequence_units(rows, unit_rows, waves, layers)?;
    for unit in &mut units {
        unit.rows = unit.rows.start+start..unit.rows.end+start;
        let head = match heads {
            None => false,
            Some(HeadRows::All) => true,
            Some(HeadRows::MemberLast) => unit.rows.end == end,
            Some(HeadRows::FromPosition(first)) => unit.rows.end >= first,
        };
        unit.segments = vec![UnitSegment { sequence: 0, rows: 0..unit.rows.len(), tokens: unit.rows.clone(),
            prompt_end: unit.rows.end == end, snapshot: false, head }];
    }
    Ok(units)
}

fn decode_spans(widths: impl IntoIterator<Item = usize>, capacity: usize) -> Result<Vec<std::ops::Range<usize>>> {
    let mut rows = 0usize;
    widths.into_iter().map(|width| {
        ensure!(width > 0, "Inkling decode member has no rows");
        let end = rows.checked_add(width).context("Inkling decode row extent overflow")?;
        ensure!(end <= capacity, "Inkling decode exceeds admitted {capacity}-row owner");
        let span = rows..end;
        rows = end;
        Ok(span)
    }).collect()
}

#[derive(Default)]
pub(crate) struct Draft {anchor:Option<u32>,
    /// The current proposal's per-token candidate probabilities (empty unless read back).
    confidence:Vec<f32>}
impl scheduler::DraftState for Draft {
    fn pending_rows(&self) -> usize { usize::from(self.anchor.is_some()) }
    fn close(&mut self, token: u32) {self.anchor=Some(token);}
}

fn scoring_replay(
    sequences: &mut [(&mut Placement, &[u32])], speculative: bool,
    decode: impl FnOnce(&mut [(&mut Placement, &[u32])]) -> Result<Option<DeviceLogits>>,
    accept: impl FnOnce(&mut [scheduler::Verified<'_, Placement, Draft, usize>]) -> Result<()>,
) -> Result<Option<DeviceLogits>> {
    ensure!(!speculative, "Inkling scoring replay cannot speculate");
    let logits = decode(sequences)?;
    // Teacher-forced rows (including cold prompt replay) have no scheduler
    // accept round. Commit their receipt and draft taps before the next tile.
    let mut drafts = (0..sequences.len()).map(|_| Draft::default()).collect::<Vec<_>>();
    let mut verified = sequences.iter_mut().zip(&mut drafts).scan(0, |first_row, ((placement, rows), seq)| {
        let verified = scheduler::Verified { start: placement.len-rows.len(), history: placement.slot,
            placement: &mut **placement, seq, rows: *rows, first_row: *first_row,
            kept: None, finished: false, neural: false };
        *first_row += rows.len();
        Some(verified)
    }).collect::<Vec<_>>();
    accept(&mut verified)?;
    Ok(logits)
}

enum Pending<'a>{
    Plain(Vec<(usize,usize,Vec<u32>)>),
    Spec{mark:super::verify::TargetMark<'a>,tokens:Vec<Vec<u32>>},
}

pub(crate) struct InklingBackend<'a> {
    execution: RefCell<Execution<'a>>,
    family: InklingPrefix<'a>,
    library: &'a cuteafd_ffi::NativeLibrary,
    context: usize,
    concurrency: usize,
    unit_rows: usize,
    waves: usize,
    vocab: usize,
    host: Option<cuteafd_hostcache::config::Config>,
    profile: RefCell<[f64; 2]>,
    pending: RefCell<Option<Pending<'a>>>,
    selector: Vec<cuteafd_loader::placement::planner::Allocation>,
    selector_ready: std::cell::Cell<bool>,
    vision: Option<crate::shared::vision::perception::Binding>,
}
impl<'a> InklingBackend<'a> {
    pub fn new(execution: Execution<'a>, context: usize,
        host: Option<cuteafd_hostcache::config::Config>, plan: &cuteafd_loader::placement::planner::ResolvedPlan) -> Result<Self> {
        let library = execution.requests.decode.stream.device.library;
        let cache = execution.requests.cache.clone();
        ensure!(context > 0 && context <= execution.cfg.max_position && context <= cache.layout.pool_pages*64,
            "Inkling served context outside admitted pool/checkpoint");
        let concurrency = cache.layout.sequences;
        let unit_rows = execution.requests.prefill[0].layout.rows;
        let waves = execution.requests.prefill.len();
        let vocab = execution.cfg.unpadded_vocab_size;
        let selector = plan.facts.allocations.iter().filter(|owner| owner.id.starts_with("selector.")).cloned().collect::<Vec<_>>();
        ensure!(!selector.is_empty(), "Inkling selector was not admitted before startup");
        let family=InklingPrefix::for_execution(&execution);
        Ok(Self { execution: RefCell::new(execution), family, library,
            context, concurrency, unit_rows, waves, vocab, host, profile: RefCell::new([0.; 2]),
            pending: RefCell::new(None), selector, vision: None, selector_ready: std::cell::Cell::new(false) })
    }
    pub fn with_vision(mut self, binding: crate::shared::vision::perception::Binding) -> Self {
        self.vision = Some(binding); self
    }
    fn check(&self, placement: &Placement, tokens: &[u32], pool: &RefPagePool) -> Result<()> {
        ensure!(!tokens.is_empty() && tokens.iter().all(|&token| (token as usize) < self.vocab)
            && placement.len == placement.state_len && placement.slot < self.concurrency,
            "Inkling append token/frontier/slot differs");
        check_writable(placement, tokens.len(), self.context, pool)

    }
    /// `heads` names the rows whose logits the callback reads; units holding
    /// none of them only normalize (and tap, with a drafter).
    fn append(&self, placement: &mut Placement, tokens: &[u32], heads: Option<HeadRows>, pool: &RefPagePool,
        callback: &mut dyn FnMut(usize, usize, DeviceLogits) -> Result<()>) -> Result<()> {
        self.check(placement, tokens, pool)?;
        let start = placement.len;
        let units = append_units(start, tokens.len(), self.unit_rows, self.waves,
            self.execution.borrow().cfg.layers, heads)?;
        let mut next = [placement.clone()];
        self.execution.borrow_mut().prefill(&mut next, &[tokens], &units, 1, heads.unwrap_or(HeadRows::All), callback)?;
        *placement = next.into_iter().next().unwrap();
        Ok(())
    }
    fn decode(&self, sequences: &mut [(&mut Placement, &[u32])], speculative: bool,
        pool: &RefPagePool) -> Result<Option<DeviceLogits>> {
        ensure!(!sequences.is_empty() && sequences.len() <= self.concurrency && self.pending.borrow().is_none(),
            "Inkling target decode requires admitted requests and prior commit");
        let spans = decode_spans(sequences.iter().map(|(_, tokens)| tokens.len()),
            self.execution.borrow().requests.decode.layout.rows)?;
        let rows = spans.last().context("Inkling decode has no rows")?.end;
        let mut slots = BTreeSet::new();
        let mut members = Vec::new();
        let mut placements = Vec::new();
        let mut sources = Vec::new();
        let mut receipt = Vec::new();
        for (index, (placement, tokens)) in sequences.iter().enumerate() {
            self.check(placement, tokens, pool)?;
            let cfg = &self.execution.borrow().cfg;
            let ring = cuteafd_loader::families::inkling2::execution::SequenceLayout::new(cfg, true, 1)?.ring_rows;
            debug_assert!(tokens.len() <= ring-cfg.window, "verify append exceeds ring dead horizon");
            cuteafd_loader::families::inkling2::verify::verify_ring(cfg.window, ring, tokens.len())?;
            ensure!(slots.insert(placement.slot), "Inkling decode aliases a request state slot");
            members.push(UnitSegment { sequence: index, rows: spans[index].clone(),
                tokens: placement.len..placement.len+tokens.len(),
                prompt_end: false, snapshot: false, head: true });
            receipt.push((placement.slot, placement.len, tokens.to_vec()));
            placements.push((**placement).clone()); sources.push(*tokens);
        }
        let layers = self.execution.borrow().cfg.layers;
        let unit = Unit { index: 0, sequence: 0, segments: members, rows: 0..rows,
            layers: 0..layers, slot: 0, barrier: false };
        let (logits,pending)=if speculative{
            use crate::shared::speculation::target::{BlockVerifyTarget,VerifyRequest,VerifyRow};
            let mut execution=self.execution.borrow_mut();
            let taps=execution.draft.as_ref().context("speculative verify without admitted drafter")?.layout.config.taps.clone();
            let ids=placements.iter().map(|p|p.slot).collect::<Vec<_>>();
            let rows=receipt.iter().flat_map(|(slot,start,tokens)|tokens.iter().enumerate().map(move |(offset,&token)|
                VerifyRow{slot:*slot,position:start+offset,token:Some(token)})).collect::<Vec<_>>();
            let mut target=super::verify_target::Target::new(&mut execution,&mut placements);
            let mark=target.checkpoint(&ids)?;
            let logits=target.verify(&mark,VerifyRequest{rows:&rows,tap_sites:&taps})?.logits;
            (logits,Pending::Spec{mark,tokens:receipt.iter().map(|(_,_,tokens)|tokens.clone()).collect()})
        }else{
            (self.execution.borrow_mut().decode(&mut placements,&sources,&unit,1)?,Pending::Plain(receipt))
        };
        // A speculative proposal's placements are disposable. Accept carries
        // the saved FIR mark and publishes only its selected prefix.
        if !speculative {
            for ((placement, _), next) in sequences.iter_mut().zip(placements) { **placement = next; }
        }
        *self.pending.borrow_mut() = Some(pending);
        Ok(Some(logits))
    }
}

impl<'a> Backend<'a> for InklingBackend<'a> {
    type Placement = Placement;
    type Draft = Draft;
    type History = usize;
    type Coordinates = ();
    type Encoder = crate::shared::qwen::Encoder;
    type Meta = ();
    type Prefix = InklingPrefix<'a>;
    fn geometry(&self) -> scheduler::Geometry {
        scheduler::Geometry { family: "inkling2", vocab_size: self.vocab, max_context: self.context,
            slots: self.concurrency, prefill_rows: self.unit_rows*self.waves,
            decode_rows: self.execution.borrow().requests.decode.layout.rows }
    }
    fn scoring_verify_capacity(&self) -> usize {
        let execution = self.execution.borrow();
        scoring_verify_capacity(execution.requests.decode.layout.rows, &execution.cfg)
    }
    fn message_starts(&self) -> &[&str] { &["<|message_system|>", "<|message_user|>", "<|message_model|>", "<|message_tool|>"] }
    fn library(&self) -> &'a cuteafd_ffi::NativeLibrary { self.library }
    fn profile(&self) -> &RefCell<[f64; 2]> { &self.profile }
    fn selector_ready(&self, selector: &mut TokenSelector<'a>) -> Result<()> {
        use cuteafd_loader::placement::planner::{CoordinatorDevices, Location};
        ensure!(!self.selector_ready.get(), "Inkling selector initialized twice");
        selector.reserve_sampler()?;
        let physical = [self.execution.borrow().requests.decode.stream.device.id];
        let mut inventory = crate::shared::memory::receipts::Inventory::new(CoordinatorDevices::new(&physical)?, "selector",
            cuteafd_loader::placement::planner::serving::ServingOwners::LIFETIME,
            cuteafd_core::memory_layout::Category::Workspace, "Inkling selector");
        selector.allocation_receipts(&mut inventory, "next")?;
        for receipt in &mut inventory.receipts {
            receipt.allocation.id = receipt.allocation.id.replacen("selector.next.", "selector.", 1);
        }
        let actual = inventory.receipts.iter().filter(|receipt| receipt.allocation.location != Location::HOST)
            .map(|receipt| (receipt.allocation.id.as_str(), receipt)).collect::<std::collections::BTreeMap<_, _>>();
        ensure!(actual.len() == self.selector.len(), "Inkling selector owner count differs from admission");
        for planned in &self.selector {
            let id = match planned.id.strip_prefix("selector.sampler.") {
                Some(id) => format!("selector.sampler.{}", cuteafd_core::target_sampler_receipt_key(id)
                    .with_context(|| format!("Inkling sampler owner has no shared receipt key {id}"))?),
                None => planned.id.clone(),
            };
            let receipt = actual.get(id.as_str()).with_context(|| format!("Inkling missing selector owner {id}"))?;
            ensure!(receipt.allocation.bytes == planned.bytes && receipt.allocation.location == planned.location
                && receipt.allocation.lifetime == planned.lifetime && receipt.address != 0,
                "Inkling selector physical request differs from admission {id}");
        }
        self.selector_ready.set(true);
        Ok(())
    }
    fn prefix_cache(&self, args: &PrefixArgs, concurrency: usize)
        -> Result<(Self::Prefix, PrefixCache<CudaCopyEngine<'a>>)> {
        ensure!(concurrency == self.concurrency && args.prefix_partial == crate::shared::prefix::Toggle::Off,
            "Inkling prefix geometry requires exact admitted frontiers");
        let bank = self.execution.borrow().requests.cache.clone();
        let host = self.host.clone().map(|config| {
            CudaCopyEngine::registered_retained(self.library, vec![], self.family.snapshot_owners()).map(|engine| (config, engine))
        }).transpose()?;
        let host_bytes = host.as_ref().map_or(0, |(config, _)| config.bytes);
        let layout = self.family.layout();
        let cache = PrefixCache::new_with_draft_pages(layout, PrefixConfig { entries: args.prefix_cache_entries,
            mark_slots: bank.mark_slots(), keep_logits: true, min_tokens: args.prefix_cache_min_tokens }, host,
            self.family.draft_page_bytes())?;
        cuteafd_bench::context::set_kv((layout.pages * layout.page_rows) as u64,
            Some((layout.pages * layout.page_rows) as u64), layout.pages as u64,
            self.family.record_format(), host_bytes);
        Ok((self.family.clone(), cache))
    }
    fn prepare(&self, job: NativeRequest, tokens: Vec<u32>) -> Result<(scheduler::Prompt, RequestMedia, Vec<EncodeJob>)> {
        if self.vision.is_none() {
            ensure!(job.media.is_empty() && job.image_groups.is_empty(), "vision is disabled");
            ensure!(job.audio.is_empty(), "no bundled audio execution owner");
            let keys = cuteafd_engine::media::MediaKeys::new(&tokens, u32::try_from(self.vocab)?, &[])?;
            let media = RequestMedia::new(vec![], 4608, tokens.len())?;
            return Ok((scheduler::Prompt { job, tokens, keys }, media, vec![]));
        }
        self.vision.as_ref().context("Inkling media binding absent")?.prepare(job, tokens,
            u32::try_from(self.vocab)?, 4608, self.context)
    }
    fn probe_features(&self, _: &scheduler::Prompt, _: &mut RequestMedia, _: &mut EmbeddingCache, _: &Path) -> Result<()> { Ok(()) }
    fn coordinates(&self, _: &[u32], _: &NativeRequest, _: &RequestMedia) -> Result<()> { Ok(()) }
    fn placement(&self, units: Vec<u32>, slot: i32) -> Placement { Placement { media: None, units, slot: slot as usize, len: 0, state_len: 0 } }
    fn bind_request(&self, placement: &mut Placement, _: (), media: RequestMedia, tokens: &[u32], resume: usize) -> Result<()> {
        ensure!(placement.len == resume && placement.state_len == resume && resume <= tokens.len(),
            "Inkling restored request frontier differs");
        placement.media = (!media.spans().is_empty()).then_some(media);
        Ok(())
    }
    fn history(&self, placement: &Placement) -> usize { placement.slot }
    fn require_live(&self) -> Result<()> { self.execution.borrow().require_live() }
    fn log_table_stats(&self, _: &str) {}
    fn prefill_device(&self, placement: &mut Placement, tokens: &[u32], rows: usize, pool: &RefPagePool) -> Result<Option<DeviceLogits>> {
        ensure!(rows <= 1, "Inkling prefill head retains one row; scoring streams tiles");
        let mut last = None;
        self.append(placement, tokens, (rows > 0).then_some(HeadRows::MemberLast), pool, &mut |_, _, logits| {
            let offset = (logits.rows-1)*logits.stride*4;
            last = Some(DeviceLogits { ptr: logits.ptr.wrapping_byte_add(offset), rows: 1, ..logits }); Ok(())
        })?;
        Ok(last)
    }
    fn streamed_prefill_scoring(&self) -> bool { true }
    fn prefill_scoring(&self, placement: &mut Placement, tokens: &[u32], rows: usize, position: usize,
        probe: &probe::ProbeRef, pool: &RefPagePool) -> Result<Option<probe::ScoreLogits>> {
        ensure!(rows <= tokens.len(), "Inkling score suffix exceeds submitted rows");
        let first = placement.len+tokens.len()-rows+1;
        let scoring = probe::ScoringRows { consumer: probe, position };
        self.append(placement, tokens, (rows > 0).then_some(HeadRows::FromPosition(first)), pool, &mut |_, at, logits| {
            let end = at.checked_add(logits.rows).context("Inkling score tile overflow")?;
            if end > first {
                let skip = first.saturating_sub(at);
                let selected = DeviceLogits { ptr: logits.ptr.wrapping_byte_add(skip*logits.stride*4),
                    rows: logits.rows-skip, ..logits };
                scoring.record(self.library, at+skip-first, &selected)?;
            }
            Ok(())
        })?;
        Ok((rows > 0).then_some(probe::ScoreLogits::Recorded))
    }
    fn packed_limits(&self) -> Option<Limits> { Some(Limits { rows: self.unit_rows, chunk_rows: self.unit_rows, max_context: self.context, taps: None }) }
    fn prefill_packed(&self, sequences: &mut [scheduler::PrefillSeq<'_, Placement, Draft>], pool: &RefPagePool,
        on_logits: &mut dyn FnMut(usize, DeviceLogits) -> MemberOutcome) -> Result<Vec<Result<()>>> {
        ensure!(!sequences.is_empty(), "Inkling packed submission is empty");
        let limits = self.packed_limits().unwrap();
        ensure!(sequences.iter().map(|sequence| sequence.placement.slot).collect::<BTreeSet<_>>().len() == sequences.len(),
            "Inkling packed members alias a request state slot");
        let chunks = sequences.iter().map(|sequence| {
            ensure!(sequence.start == sequence.placement.len, "Inkling packed start differs");
            ensure!(!sequence.neural || self.execution.borrow().draft.is_some(), "Inkling packed neural prefill needs admitted drafter");
            self.check(sequence.placement, sequence.tokens, pool)?;
            Ok((sequence.start, sequence.tokens.len()))
        }).collect::<Result<Vec<_>>>()?;
        let segments = crate::shared::prefill_packing::plan(&chunks, &limits)?;
        let rows = segments.last().context("Inkling pack has no members")?.end_row();
        let mut placements = sequences.iter().map(|sequence| sequence.placement.clone()).collect::<Vec<_>>();
        let sources = sequences.iter().map(|sequence| sequence.tokens).collect::<Vec<_>>();
        let layers = self.execution.borrow().cfg.layers;
        let unit = Unit { index: 0, sequence: 0, rows: 0..rows, layers: 0..layers, slot: 0, barrier: false,
            segments: segments.iter().enumerate().map(|(sequence, segment)| UnitSegment { sequence,
                rows: segment.first..segment.end_row(), tokens: segment.start..segment.start+segment.rows,
                prompt_end: true, snapshot: false, head: true }).collect() };
        let mut last = (0..sequences.len()).map(|_| None).collect::<Vec<_>>();
        self.execution.borrow_mut().prefill(&mut placements, &sources, &[unit], 1, HeadRows::MemberLast, &mut |member, position, logits| {
            if position+logits.rows == chunks[member].0+chunks[member].1+1 {
                let offset = (logits.rows-1)*logits.stride*4;
                let logits = DeviceLogits { ptr: logits.ptr.wrapping_byte_add(offset), rows: 1, ..logits };
                last[member] = Some(on_logits(member, logits)?);
            }
            Ok(())
        })?;
        for (sequence, placement) in sequences.iter_mut().zip(placements) {
            *sequence.placement = placement;
            if sequence.neural {
                self.prefill_draft(sequence.placement, sequence.start, sequence.tokens, sequence.next, sequence.draft)?;
            }
        }
        last.into_iter().map(|outcome| outcome.context("Inkling packed member last-row callback missing")).collect()
    }
    fn verify_device(&self, sequences: &mut [(&mut Placement, &[u32])], speculative: bool, pool: &RefPagePool) -> Result<Option<DeviceLogits>> {
        self.decode(sequences, speculative, pool)
    }
    fn score_device(&self, sequences: &mut [(&mut Placement, &[u32])], pool: &RefPagePool)
        -> Result<Option<DeviceLogits>> {
        scoring_replay(sequences, false, |sequences| self.decode(sequences, false, pool),
            |verified| self.accept(verified, false))
    }
    fn verify_bucket_rows(&self, rows: usize, _: bool) -> usize { rows }
    fn verify_attention_trace(&self, _: &[Vec<usize>], rows: usize) -> crate::shared::speculation_trace::Deferred {
        crate::shared::speculation_trace::Deferred::new(move || serde_json::json!({"kind":"inkling2", "rows":rows, "attention":"eager"}))
    }
    // The recurrent block dSpark has no trained confidence head (the checkpoint
    // ships none); its per-token score is the target-head argmax probability,
    // calibrated online like every other score.
    fn draft_head_kind(&self) -> cuteafd_core::HeadScore {
        if self.execution.borrow().draft.is_some() { cuteafd_core::HeadScore::CandidateProbability }
        else { cuteafd_core::HeadScore::None }
    }
    fn configure_draft_scores(&self, _traced: bool, scored: bool) {
        if let Some(draft) = &self.execution.borrow().draft { draft.drafter.set_score_readback(scored); }
    }
    fn draft_head<'d>(&self, draft: &'d Draft) -> Option<&'d [f32]> {
        (!draft.confidence.is_empty()).then_some(draft.confidence.as_slice())
    }
    fn prefill_draft(&self,p:&Placement,_:usize,_:&[u32],next:Option<u32>,state:&mut Draft)->Result<()>{
        if let Some(draft)=&self.execution.borrow().draft{ensure!(draft.drafter.frontier(p.slot)?.1==p.len,"Inkling draft prefill frontier differs");}
        state.anchor=next;Ok(())
    }
    fn draft(&self,sequences:&mut [scheduler::DraftSeq<'_,Placement,Draft>],timing:&mut scheduler::DraftTiming,
        _:&mut TokenSelector<'_>)->Result<Vec<Vec<u32>>>{
        let execution=self.execution.borrow();let Some(draft)=&execution.draft else{return Ok(vec![vec![];sequences.len()]);};
        let start=std::time::Instant::now();let mut output=vec![vec![];sequences.len()];
        for first in (0..sequences.len()).step_by(draft.layout.runtime[0].sequences){
            let end=(first+draft.layout.runtime[0].sequences).min(sequences.len());
            let selected=sequences[first..end].iter().enumerate().filter(|(_,s)|s.depth>0).filter_map(|(index,s)|s.seq.anchor.map(|anchor|(index,anchor,s))).collect::<Vec<_>>();
            if selected.is_empty(){continue;}
            let rows=selected.iter().map(|(_,anchor,s)|crate::shared::speculation::dflash::DraftSeq{
                slot:s.placement.slot,anchor:*anchor,position:s.placement.len,valid_from:0}).collect::<Vec<_>>();
            let proposed=draft.drafter.draft(&rows,execution.target.operand("model.llm.embed.weight")?.buffer,
                execution.target.operand("model.llm.embed_norm.weight")?.buffer,|sequences, anchors| {
                    draft.select(&execution.head,&execution.kernels,execution.target.projection("model.llm.unembed.weight")?,
                        execution.peer.as_ref().map(|peer| super::head::PeerHead { kernels: &peer.kernels, target: &peer.target }),
                        sequences,anchors)
                })?;
            let placed=selected.into_iter().map(|(index,_,s)|(index,s.depth)).collect::<Vec<_>>();
            for ((index,depth),tokens) in placed.into_iter().zip(proposed){
                sequences[first+index].seq.confidence=tokens.confidence.into_iter().take(depth).collect();
                output[first+index]=tokens.tokens.into_iter().take(depth).collect();
            }
        }
        timing.steps+=1;timing.seconds+=start.elapsed().as_secs_f64();Ok(output)
    }
    fn accept(&self, verified: &mut [scheduler::Verified<'_, Placement, Draft, usize>], speculative: bool) -> Result<()> {
        let pending=self.pending.borrow_mut().take().context("Inkling accept lacks submitted receipt")?;
        let mut execution=self.execution.borrow_mut();let draft=execution.draft.clone();
        match pending{
            Pending::Plain(sent)=>{
                ensure!(!speculative && sent.len()==verified.len(),"Inkling plain accept inventory differs");
                for ((slot,start,tokens),request) in sent.iter().zip(verified.iter()){
                    ensure!(*slot==request.history && *slot==request.placement.slot && *start==request.start
                        && tokens==request.rows,"Inkling plain accept receipt differs");
                }
            }
            Pending::Spec{mark,tokens}=>{
                use crate::shared::speculation::target::{AcceptedPrefix,BlockVerifyTarget};
                ensure!(speculative && tokens.len()==verified.len() && tokens.iter().zip(verified.iter()).all(|(tokens,s)|tokens==s.rows),
                    "Inkling speculative accept receipt differs");
                let accepted=verified.iter().map(|s|AcceptedPrefix{slot:s.placement.slot,rows:s.kept.map_or(0,|(rows,_)|rows)}).collect::<Vec<_>>();
                let mut placements=verified.iter().map(|s|s.placement.clone()).collect::<Vec<_>>();
                super::verify_target::Target::new(&mut execution,&mut placements).commit(&mark,&accepted)?;
                for (s,p) in verified.iter_mut().zip(placements){*s.placement=p;}
            }
        }
        if let Some(draft)=draft{
            for s in verified.iter_mut(){
                let keep=if speculative{s.kept.map_or(0,|(rows,_)|rows)}else{s.rows.len()};
                if keep>0{draft.update(&draft.decode,s.placement.slot,s.start,s.first_row,keep)?;}
                if let Some((_,anchor))=s.kept{s.seq.anchor=Some(anchor);}
            }
        }
        Ok(())
    }
    fn cancel(&self,_:&Placement)->Result<()>{
        let execution=self.execution.borrow();
        if let Some(Pending::Spec{mark,..})=self.pending.borrow_mut().take(){
            execution.restore_verify(&mark)?;
        }
        execution.requests.cache.drain()?;Ok(())
    }

}

fn scoring_verify_capacity(
    decode_rows: usize,
    cfg: &cuteafd_loader::families::inkling2::Inkling2Config,
) -> usize {
    let ring = cuteafd_loader::families::inkling2::execution::SequenceLayout::new(cfg, true, 1)
        .map(|layout| layout.ring_rows.saturating_sub(cfg.window))
        .unwrap_or(cuteafd_loader::placement::families::inkling2::DECODE_ROWS as usize);
    decode_rows.min(ring)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wide_decode_rows_chunks_inkling_scoring_within_swa_ring_allowance() -> Result<()> {
        use crate::shared::probe::{score_plan, ScorePath};
        let cfg = cuteafd_loader::families::inkling2::Inkling2Config {
            vocab_size: 201024, unpadded_vocab_size: 200058, hidden: 5120, layers: 64,
            heads: 40, kv_heads: 8, head_dim: 128, d_rel: 16, rel_extent: 1024,
            local: (0..64).map(|l| (l + 1) % 4 != 0).collect(), window: 1024, sconv_kernel: 4,
            rms_norm_eps: 1e-6, qk_l2_clip: 12.0, rel_l2_clip: 12.0, log_scaling: Some((1024, 0.2)),
            res_block: 8, experts: 128, topk: 8, intermediate: 1536,
            gate_softcap: 15.0, up_softcap: 15.0, route_scale: 1.0, final_softcap: 30.0,
            max_position: 131_072, eos: vec![200_006],
        };
        let ring = cuteafd_loader::families::inkling2::execution::SequenceLayout::new(&cfg, true, 1)?.ring_rows;
        assert_eq!(ring - cfg.window, 64);

        let capacity = scoring_verify_capacity(128, &cfg);
        assert_eq!(capacity, 64);
        assert_eq!(scoring_verify_capacity(32, &cfg), 32);

        let steps = score_plan(583, 64, 4096, capacity, None, ScorePath::Decode, false)?;
        let decode_steps: Vec<_> = steps.iter().filter(|step| step.path == ScorePath::Decode).collect();
        assert!(!decode_steps.is_empty());
        for step in decode_steps {
            assert!(step.tokens.len() <= 64, "scoring chunk {} exceeds 64-row ring step", step.tokens.len());
            cuteafd_loader::families::inkling2::verify::verify_ring(cfg.window, ring, step.tokens.len())?;
        }

        let err = score_plan(583, 64, 4096, capacity, Some(65), ScorePath::Decode, false).unwrap_err();
        assert!(err.to_string().contains("1..=64"), "{err:#}");
        let err = score_plan(583, 64, 4096, capacity, Some(128), ScorePath::Decode, false).unwrap_err();
        assert!(err.to_string().contains("1..=64"), "{err:#}");
        Ok(())
    }

    /// Vocabulary projections a call issues: one per projected 64-row span.
    fn projections(units: &[Unit], heads: HeadRows) -> Result<usize> {
        units.iter().map(|unit| Ok(super::super::prefill::projected_spans(unit.rows.len(), &unit.segments, heads)?
            .into_iter().filter(|(_, project)| *project).count())).sum()
    }

    #[test]
    fn serving_prefill_projects_one_head_span_per_member_not_every_64_rows() -> Result<()> {
        for (start, rows, unit_rows, waves) in [(0, 8192, 4096, 2), (0, 8192, 2048, 4), (37, 8191, 4096, 2),
            (0, 100, 4096, 2), (5000, 3000, 4096, 3), (0, 65, 256, 1)] {
            let serving = append_units(start, rows, unit_rows, waves, 1, Some(HeadRows::MemberLast))?;
            assert_eq!(projections(&serving, HeadRows::MemberLast)?, 1, "{rows} rows from {start}");
            assert_eq!(serving.iter().filter(|unit| unit.segments[0].head).count(), 1);
            let last = serving.last().unwrap();
            assert!(last.segments[0].head && last.rows.end == start+rows);
            // The old shape: every unit a head segment, every row projected.
            let every = append_units(start, rows, unit_rows, waves, 1, Some(HeadRows::All))?;
            assert_eq!(projections(&every, HeadRows::All)?,
                every.iter().map(|unit| unit.rows.len().div_ceil(64)).sum::<usize>());
            let none = append_units(start, rows, unit_rows, waves, 1, None)?;
            assert_eq!(projections(&none, HeadRows::MemberLast)?, 0);
        }
        // Packed prefill: three members in one unit, one projection each, even
        // when two members' last rows share a span.
        let segment = |sequence, rows: std::ops::Range<usize>, start| UnitSegment { sequence,
            tokens: start..start+rows.len(), rows, prompt_end: true, snapshot: false, head: true };
        let unit = Unit { index: 0, sequence: 0, rows: 0..3000, layers: 0..1, slot: 0, barrier: false,
            segments: vec![segment(0, 0..1000, 0), segment(1, 1000..1010, 64), segment(2, 1010..3000, 7)] };
        let spans = super::super::prefill::projected_spans(3000, &unit.segments, HeadRows::MemberLast)?;
        let projected = spans.iter().filter(|(_, project)| *project).map(|(span, _)| span.clone()).collect::<Vec<_>>();
        assert_eq!(projected, [960..1024, 2944..3000]);
        assert_eq!(projections(std::slice::from_ref(&unit), HeadRows::MemberLast)?, 2);
        assert_eq!(spans.len(), 3000usize.div_ceil(64));
        // Streamed scoring projects the spans reaching the scored suffix only.
        let scoring = append_units(0, 8192, 4096, 2, 1, Some(HeadRows::FromPosition(8192-99)))?;
        assert_eq!(scoring.iter().filter(|unit| unit.segments[0].head).count(), 1);
        assert_eq!(projections(&scoring, HeadRows::FromPosition(8192-99))?, 2);
        let scoring = append_units(0, 8192, 4096, 2, 1, Some(HeadRows::FromPosition(4000)))?;
        assert_eq!(projections(&scoring, HeadRows::FromPosition(4000))?, 8192/64-3999/64);
        Ok(())
    }

    #[test]
    fn teacher_forced_windows_commit_receipts_and_draft_taps_before_next_decode() -> Result<()> {
        use std::cell::Cell;
        let pending = RefCell::new(None);
        let frontiers = RefCell::new([11, 29]);
        let commits = Cell::new(0);
        let mut first = Placement { media: None, units: vec![], slot: 0, len: 11, state_len: 11 };
        let mut second = Placement { slot: 1, len: 29, state_len: 29, ..first.clone() };
        let mut decode = |sequences: &mut [(&mut Placement, &[u32])]| -> Result<Option<DeviceLogits>> {
            ensure!(pending.borrow().is_none(), "Inkling target decode requires admitted requests and prior commit");
            let receipt = sequences.iter().map(|(p, rows)| (p.slot, p.len, rows.to_vec())).collect::<Vec<_>>();
            for (p, rows) in sequences {
                p.len += rows.len(); p.state_len = p.len;
            }
            *pending.borrow_mut() = Some(receipt);
            Ok(None)
        };
        let mut accept = |verified: &mut [scheduler::Verified<'_, Placement, Draft, usize>]| -> Result<()> {
            let sent = pending.borrow_mut().take().context("missing receipt")?;
            ensure!(sent.len() == verified.len(), "member count differs");
            let mut first_row = 0;
            for ((slot, start, rows), v) in sent.iter().zip(verified) {
                assert_eq!((*slot, *start, rows.as_slice()), (v.history, v.start, v.rows));
                assert_eq!(v.first_row, first_row);
                assert_eq!(v.placement.state_len, v.start+v.rows.len());
                assert!(!v.neural && v.kept.is_none());
                let mut frontiers = frontiers.borrow_mut();
                assert_eq!(frontiers[*slot], *start, "drafter taps must remain contiguous");
                frontiers[*slot] += rows.len();
                first_row += rows.len();
            }
            commits.set(commits.get()+1);
            Ok(())
        };
        // Reproduce the old scoring path's refusal on its second window.
        decode(&mut [(&mut first, &[1, 2])])?;
        assert!(decode(&mut [(&mut first, &[3])]).err().unwrap().to_string().contains("prior commit"));
        *pending.borrow_mut() = None;
        first.len = 11; first.state_len = 11;
        for _ in 0..3 {
            scoring_replay(&mut [(&mut first, &[1, 2]), (&mut second, &[3, 4, 5])], false,
                &mut decode, &mut accept)?;
            assert!(pending.borrow().is_none());
        }
        assert_eq!(commits.get(), 3);
        assert_eq!(*frontiers.borrow(), [17, 38]);
        assert!(scoring_replay(&mut [(&mut first, &[1])], true, &mut decode, &mut accept).is_err());
        assert_eq!(commits.get(), 3);
        let failure = scoring_replay(&mut [(&mut first, &[1])], false,
            |_| anyhow::bail!("decode failed"), &mut accept).err().unwrap();
        assert_eq!(failure.to_string(), "decode failed");
        assert_eq!(commits.get(), 3);
        Ok(())
    }

    #[test]
    fn cold_decode_probe_commits_once_before_prefill_draft_binding() -> Result<()> {
        let scheduler = include_str!("../../shared/qwen/scheduler.rs");
        let cold = scheduler.split("let decode = p.job.probe.as_ref()").nth(1).unwrap()
            .split("p.done += chunk.len();").next().unwrap();
        assert!(cold.contains("step.decode"));
        assert!(cold.contains("engine.score_device("));
        assert!(cold.contains("engine.prefill_draft(&p.placement, start, chunk"));
        assert!(!cold.contains("engine.accept("), "cold replay must not accept twice");
        assert!(!cold.contains("lanes::commit("));
        let backend = include_str!("backend.rs").split("#[cfg(test)]").next().unwrap();
        let replay = backend.split("fn score_device(").nth(1).unwrap()
            .split("fn verify_bucket_rows(").next().unwrap();
        assert!(replay.contains("scoring_replay(sequences, false"));
        assert!(replay.contains("self.accept(verified, false)"));

        let mut placement = Placement { media: None, units: vec![], slot: 0, len: 8, state_len: 8 };
        let frontier = std::cell::Cell::new(8);
        let commits = std::cell::Cell::new(0);
        for rows in [&[10, 11][..], &[12][..]] {
            scoring_replay(&mut [(&mut placement, rows)], false, |sequences| {
                for (p, rows) in sequences { p.len += rows.len(); p.state_len = p.len; }
                Ok(None)
            }, |verified| {
                assert_eq!(verified.len(), 1);
                assert_eq!(verified[0].start, frontier.get());
                frontier.set(verified[0].placement.len);
                commits.set(commits.get()+1);
                Ok(())
            })?;
            // Inkling prefill_draft binds the existing, already published taps.
            assert_eq!(frontier.get(), placement.len);
        }
        assert_eq!(commits.get(), 2);
        assert_eq!(frontier.get(), 11);
        Ok(())
    }

    #[test]
    fn recurrent_native_seven_token_only_history_always_explores() -> Result<()> {
        use cuteafd_core::{DraftControl, DraftHistory, DraftOutcome, DrafterSpec, HeadScore};
        let config = cuteafd_loader::formats::block_dspark::recurrent::RecurrentBlockConfig::from_hf(
            &serde_json::from_str(include_str!("../../../../cuteafd-loader/src/families/inkling2/fixtures/config.json"))?)?;
        let width = config.predictions();
        assert_eq!(width, 7);
        let mut policy = DraftHistory::new(DrafterSpec { block: width, head: HeadScore::None, copy: 0 }, DraftControl::History);
        assert_eq!(policy.plan(&[(1, width)], 64), [7]);
        for _ in 0..256 {
            let depth = policy.plan(&[(1, width)], 64)[0];
            let kept = policy.refine(1, depth, None);
            assert!(kept >= 1, "headless drafter must explore suppressed first position");
            policy.observe(&[DraftOutcome { id: 1, proposed: kept, accepted: 0, censored: false }], false);
        }
        assert_eq!(policy.plan(&[(1, width)], 64), [1]);
        assert!(policy.stats().explored > 0);
        for _ in 0..32 {
            let depth = policy.plan(&[(1, width)], 64)[0];
            policy.observe(&[DraftOutcome { id: 1, proposed: depth, accepted: depth, censored: false }], false);
        }
        assert_eq!(policy.plan(&[(1, width)], 64), [7]);
        Ok(())
    }

    #[test]
    fn agreeing_history_copies_preserve_exact_fir_prefixes() -> Result<()> {
        use cuteafd_loader::families::inkling2::verify::{accepted_history, Source};
        let mut history: Vec<u32> = (1..=8).collect();
        history.extend(20..60);
        history.extend(900..910);
        history.extend(1..=8);
        let draft = [20,21,22,23,24,25,26];
        let tail = crate::shared::draft::copy::continuation(&history, &draft, true).unwrap();
        assert_eq!(tail.len(), crate::shared::draft::copy::COPY_EXTENSION);
        let rows = std::iter::once(9).chain(draft).chain(tail.iter().copied()).collect::<Vec<_>>();
        assert_eq!(rows.len(), 1 + draft.len() + crate::shared::draft::copy::COPY_EXTENSION);
        let prior = [100,99,98];
        for keep in 0..=rows.len() {
            let committed = accepted_history(3, rows.len(), keep)?.into_iter().map(|source| match source {
                Source::Input(i) => rows[i], Source::Prior(i) => prior[i] }).collect::<Vec<_>>();
            let mut serial = prior.to_vec();
            for &token in &rows[..keep] { serial.insert(0, token); serial.truncate(3); }
            assert_eq!(committed, serial);
        }
        assert_eq!(decode_spans([40, 24], 64)?.last().unwrap().end, 64);
        Ok(())
    }

    #[test]
    fn packed_neural_prefill_updates_context_and_anchor_before_returning() {
        let source = include_str!("backend.rs").split("#[cfg(test)]").next().unwrap();
        let packed = source.split("fn prefill_packed(").nth(1).unwrap().split("fn verify_device(").next().unwrap();
        assert!(!packed.contains("!sequence.neural &&"));
        assert!(packed.contains("self.prefill_draft(sequence.placement, sequence.start, sequence.tokens, sequence.next, sequence.draft)?"));
        assert!(packed.find("*sequence.placement = placement;").unwrap() < packed.find("self.prefill_draft(").unwrap());
    }

    #[test]
    fn decode_members_share_admitted_rows_without_overlapping_logits() -> Result<()> {
        assert_eq!(decode_spans([1, 7, 16, 40], 64)?, vec![0..1, 1..8, 8..24, 24..64]);
        assert_eq!(decode_spans([64], 64)?, vec![0..64]);
        assert!(decode_spans([1, 64], 64).is_err());
        assert!(decode_spans([0], 64).is_err());
        assert!(decode_spans([usize::MAX, 1], usize::MAX).is_err());
        Ok(())
    }
    #[test]
    fn append_writes_only_reserved_private_nonaliased_pages() -> Result<()> {
        let mut pool = RefPagePool::new(4, 64);
        let units = pool.alloc_for(0..2)?;
        let placement = Placement { media: None, units: units.clone(), slot: 0, len: 63, state_len: 63 };
        check_writable(&placement, 3, 128, &pool)?;
        pool.share(&units[..1]);
        assert!(check_writable(&placement, 3, 128, &pool).is_err());
        pool.release(&units[..1]);
        let alias = Placement { units: vec![units[0], units[0]], len: 0, state_len: 0, ..placement.clone() };
        assert!(check_writable(&alias, 128, 128, &pool).is_err());
        assert!(check_writable(&placement, 66, 128, &pool).is_err());
        assert!(check_writable(&placement, 0, 128, &pool).is_err());
        let invalid = Placement { units: vec![99], len: 0, state_len: 0, ..placement };
        assert!(check_writable(&invalid, 1, 128, &pool).is_err());
        Ok(())
    }
    #[test]
    fn adapter_uses_shared_scheduler_units_prefix_and_streamed_scoring() {
        let source = include_str!("backend.rs").split("#[cfg(test)]").next().unwrap();
        assert!(source.contains("Backend<'a> for InklingBackend"));
        assert!(source.contains("registered_retained("));
        assert!(source.contains("prefill_packing::plan("));
        assert!(source.contains("streamed_prefill_scoring(&self) -> bool { true }"));
        assert!(!source.contains("tokio::select!"));
        assert!(!source.contains("loop {"));
        assert!(!source.contains("for layer in"));
    }
}
