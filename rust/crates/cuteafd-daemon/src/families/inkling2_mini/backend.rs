//! Inkling 2 Mini adapter for the shared request scheduler (`Backend`).
use super::{
    cache::{MiniPrefix, Placement},
    engine::Execution,
};
use crate::shared::{
    prefix::{CudaCopyEngine, PrefixArgs},
    prefill_packing::{Limits, MemberOutcome},
    prefill_pipeline::{sequence_units, Unit, UnitSegment},
    probe,
    qwen::scheduler::{self, Backend},
    token_io::{DeviceLogits, TokenSelector},
};
use anyhow::{ensure, Context, Result};
use cuteafd_api::openai::NativeRequest;
use cuteafd_engine::{
    media::{EmbeddingCache, EncodeJob, RequestMedia},
    prefix::{PrefixCache, PrefixConfig, PrefixFamily, RefPagePool},
};
use std::{cell::RefCell, collections::BTreeSet, path::Path};

const HIDDEN: usize = 2304;

fn check_writable(placement: &Placement, rows: usize, context: usize, pool: &RefPagePool) -> Result<()> {
    let end = placement
        .len
        .checked_add(rows)
        .context("Inkling 2 Mini append extent overflow")?;
    ensure!(
        rows > 0 && end <= context && end.div_ceil(64) <= placement.units.len(),
        "Inkling 2 Mini append exceeds reserved pages"
    );
    let mut writable = BTreeSet::new();
    for position in placement.len..end {
        let page = placement.units[position / 64];
        ensure!(
            (page as usize) < pool.capacity() && pool.refs(page) == 1 && writable.insert((page, position % 64)),
            "Inkling 2 Mini append requires private nonaliased COW pages"
        );
    }
    Ok(())
}

fn decode_spans(widths: impl IntoIterator<Item = usize>, capacity: usize) -> Result<Vec<std::ops::Range<usize>>> {
    let mut rows = 0usize;
    widths
        .into_iter()
        .map(|width| {
            ensure!(width > 0, "Mini decode member has no rows");
            let end = rows.checked_add(width).context("Mini decode row extent overflow")?;
            ensure!(end <= capacity, "Mini decode exceeds admitted {capacity}-row owner");
            let span = rows..end;
            rows = end;
            Ok(span)
        })
        .collect()
}

#[derive(Default)]
pub(crate) struct Draft {
    anchor: Option<u32>,
    confidence: Vec<f32>,
}

impl scheduler::DraftState for Draft {
    fn pending_rows(&self) -> usize {
        usize::from(self.anchor.is_some())
    }
    fn close(&mut self, token: u32) {
        self.anchor = Some(token);
    }
}

enum Pending<'a> {
    Plain(Vec<(usize, usize, Vec<u32>)>),
    Spec {
        mark: super::verify::TargetMark<'a>,
        tokens: Vec<Vec<u32>>,
    },
}

pub(crate) struct MiniBackend<'a> {
    execution: RefCell<Execution<'a>>,
    family: MiniPrefix<'a>,
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

impl<'a> MiniBackend<'a> {
    pub fn new(
        execution: Execution<'a>,
        context: usize,
        host: Option<cuteafd_hostcache::config::Config>,
        plan: &cuteafd_loader::placement::planner::ResolvedPlan,
    ) -> Result<Self> {
        let library = execution.requests.decode.stream.device.library;
        let cache = execution.requests.cache.clone();
        ensure!(
            context > 0 && context <= execution.cfg.text.max_position && context <= cache.layout.pool_pages * 64,
            "Inkling 2 Mini served context outside admitted pool/checkpoint"
        );
        let concurrency = cache.layout.sequences;
        let unit_rows = execution.requests.prefill[0].layout.rows;
        let waves = execution.requests.prefill.len();
        let vocab = execution.cfg.text.unpadded_vocab_size;
        let selector: Vec<_> = plan
            .facts
            .allocations
            .iter()
            .filter(|owner| owner.id.starts_with("selector."))
            .cloned()
            .collect();
        ensure!(!selector.is_empty(), "Inkling 2 Mini selector was not admitted before startup");
        let family = MiniPrefix::for_execution(&execution);
        Ok(Self {
            execution: RefCell::new(execution),
            family,
            library,
            context,
            concurrency,
            unit_rows,
            waves,
            vocab,
            host,
            profile: RefCell::new([0.; 2]),
            pending: RefCell::new(None),
            selector,
            selector_ready: std::cell::Cell::new(false),
            vision: None,
        })
    }

    pub fn with_vision(mut self, binding: crate::shared::vision::perception::Binding) -> Self {
        self.vision = Some(binding);
        self
    }

    fn check(&self, placement: &Placement, tokens: &[u32], pool: &RefPagePool) -> Result<()> {
        ensure!(
            !tokens.is_empty()
                && tokens.iter().all(|&tok| (tok as usize) < self.vocab)
                && placement.len == placement.state_len
                && placement.slot < self.concurrency,
            "Inkling 2 Mini append token/frontier/slot differs"
        );
        check_writable(placement, tokens.len(), self.context, pool)
    }

    fn append(
        &self,
        placement: &mut Placement,
        tokens: &[u32],
        head_projection: Option<super::prefill::HeadProjection>,
        pool: &RefPagePool,
        callback: &mut dyn FnMut(usize, usize, DeviceLogits) -> Result<()>,
    ) -> Result<()> {
        self.check(placement, tokens, pool)?;
        let start = placement.len;
        let end = start + tokens.len();
        let visits = self.execution.borrow().cfg.total_visits();
        let mut units = sequence_units(tokens.len(), self.unit_rows, self.waves, visits)?;
        for unit in &mut units {
            unit.rows = unit.rows.start + start..unit.rows.end + start;
            let unit_head = match head_projection {
                None => false,
                Some(super::prefill::HeadProjection::AllHeadRows) => true,
                Some(super::prefill::HeadProjection::MemberLastRow) => unit.rows.end == end,
                Some(super::prefill::HeadProjection::FromPosition(first_pos)) => unit.rows.end >= first_pos,
            };
            unit.segments = vec![UnitSegment {
                sequence: 0,
                rows: 0..unit.rows.len(),
                tokens: unit.rows.clone(),
                prompt_end: unit.rows.end == end,
                snapshot: false,
                head: unit_head,
            }];
        }
        let mut next = [placement.clone()];
        self.execution.borrow_mut().prefill(
            &mut next,
            &[tokens],
            &units,
            head_projection.unwrap_or(super::prefill::HeadProjection::AllHeadRows),
            callback,
        )?;
        *placement = next.into_iter().next().unwrap();
        Ok(())
    }

    fn decode(
        &self,
        sequences: &mut [(&mut Placement, &[u32])],
        speculative: bool,
        pool: &RefPagePool,
    ) -> Result<Option<DeviceLogits>> {
        ensure!(
            !sequences.is_empty() && sequences.len() <= self.concurrency && self.pending.borrow().is_none(),
            "Inkling 2 Mini decode requires admitted requests and prior commit"
        );
        let spans = decode_spans(
            sequences.iter().map(|(_, tokens)| tokens.len()),
            self.execution.borrow().requests.decode.layout.rows,
        )?;
        let rows = spans.last().context("Mini decode has no rows")?.end;
        let mut slots = BTreeSet::new();
        let mut members = Vec::new();
        let mut placements = Vec::new();
        let mut sources = Vec::new();
        let mut receipt = Vec::new();
        for (index, (placement, tokens)) in sequences.iter().enumerate() {
            self.check(placement, tokens, pool)?;
            let t = &self.execution.borrow().cfg.text;
            let ring = cuteafd_loader::families::inkling2::execution::SequenceLayout::new(t, true, 1)?.ring_rows;
            cuteafd_loader::families::inkling2::verify::verify_ring(t.window, ring, tokens.len())?;
            ensure!(slots.insert(placement.slot), "Mini decode aliases a request state slot");
            members.push(UnitSegment {
                sequence: index,
                rows: spans[index].clone(),
                tokens: placement.len..placement.len + tokens.len(),
                prompt_end: false,
                snapshot: false,
                head: true,
            });
            receipt.push((placement.slot, placement.len, tokens.to_vec()));
            placements.push((**placement).clone());
            sources.push(*tokens);
        }
        let visits = self.execution.borrow().cfg.total_visits();
        let unit = Unit {
            index: 0,
            sequence: 0,
            segments: members,
            rows: 0..rows,
            layers: 0..visits,
            slot: 0,
            barrier: false,
        };
        let (logits, pending) = if speculative {
            use crate::shared::speculation::target::{BlockVerifyTarget, VerifyRequest, VerifyRow};
            let mut execution = self.execution.borrow_mut();
            let taps = execution
                .draft
                .as_ref()
                .context("Mini speculative verify without admitted drafter")?
                .sites
                .clone();
            let ids: Vec<usize> = placements.iter().map(|p| p.slot).collect();
            let rows: Vec<VerifyRow> = receipt
                .iter()
                .flat_map(|(slot, start, tokens)| {
                    tokens.iter().enumerate().map(move |(offset, &token)| VerifyRow {
                        slot: *slot,
                        position: start + offset,
                        token: Some(token),
                    })
                })
                .collect();
            let mut target = super::verify_target::Target::new(&mut execution, &mut placements);
            let mark = target.checkpoint(&ids)?;
            let logits = target.verify(&mark, VerifyRequest { rows: &rows, tap_sites: &taps })?.logits;
            (
                logits,
                Pending::Spec {
                    mark,
                    tokens: receipt.iter().map(|(_, _, tokens)| tokens.clone()).collect(),
                },
            )
        } else {
            (
                self.execution.borrow_mut().decode(&mut placements, &sources, &unit)?,
                Pending::Plain(receipt),
            )
        };
        if !speculative {
            for ((placement, _), next) in sequences.iter_mut().zip(placements) {
                **placement = next;
            }
        }
        *self.pending.borrow_mut() = Some(pending);
        Ok(Some(logits))
    }
}

impl<'a> Backend<'a> for MiniBackend<'a> {
    type Placement = Placement;
    type Draft = Draft;
    type History = usize;
    type Coordinates = ();
    type Encoder = crate::shared::qwen::Encoder;
    type Meta = ();
    type Prefix = MiniPrefix<'a>;

    fn geometry(&self) -> scheduler::Geometry {
        scheduler::Geometry {
            family: "inkling2_mini",
            vocab_size: self.vocab,
            max_context: self.context,
            slots: self.concurrency,
            prefill_rows: self.unit_rows * self.waves,
            decode_rows: 64,
        }
    }

    fn message_starts(&self) -> &[&str] {
        &[
            "<|message_system|>",
            "<|message_user|>",
            "<|message_model|>",
            "<|message_tool|>",
        ]
    }
    fn library(&self) -> &'a cuteafd_ffi::NativeLibrary {
        self.library
    }
    fn profile(&self) -> &RefCell<[f64; 2]> {
        &self.profile
    }

    fn selector_ready(&self, selector: &mut TokenSelector<'a>) -> Result<()> {
        use cuteafd_loader::placement::planner::{CoordinatorDevices, Location};
        ensure!(!self.selector_ready.get(), "Mini selector initialized twice");
        selector.reserve_sampler()?;
        let physical = [self.execution.borrow().requests.decode.stream.device.id];
        let mut inventory = crate::shared::memory::receipts::Inventory::new(
            CoordinatorDevices::new(&physical)?,
            "selector",
            cuteafd_loader::placement::planner::serving::ServingOwners::LIFETIME,
            cuteafd_core::memory_layout::Category::Workspace,
            "Inkling 2 Mini selector",
        );
        selector.allocation_receipts(&mut inventory, "next")?;
        for receipt in &mut inventory.receipts {
            receipt.allocation.id = receipt.allocation.id.replacen("selector.next.", "selector.", 1);
        }
        let actual: std::collections::BTreeMap<_, _> = inventory
            .receipts
            .iter()
            .filter(|r| r.allocation.location != Location::HOST)
            .map(|r| (r.allocation.id.as_str(), r))
            .collect();
        ensure!(actual.len() == self.selector.len(), "Mini selector owner count differs from admission");
        for planned in &self.selector {
            let id = match planned.id.strip_prefix("selector.sampler.") {
                Some(id) => format!(
                    "selector.sampler.{}",
                    cuteafd_core::target_sampler_receipt_key(id)
                        .with_context(|| format!("Mini sampler owner has no shared receipt key {id}"))?
                ),
                None => planned.id.clone(),
            };
            let receipt = actual
                .get(id.as_str())
                .with_context(|| format!("Mini missing selector owner {id}"))?;
            ensure!(
                receipt.allocation.bytes == planned.bytes
                    && receipt.allocation.location == planned.location
                    && receipt.allocation.lifetime == planned.lifetime
                    && receipt.address != 0,
                "Mini selector physical request differs from admission {id}"
            );
        }
        self.selector_ready.set(true);
        Ok(())
    }

    fn prefix_cache(
        &self,
        args: &PrefixArgs,
        concurrency: usize,
    ) -> Result<(Self::Prefix, PrefixCache<CudaCopyEngine<'a>>)> {
        ensure!(
            concurrency == self.concurrency && args.prefix_partial == crate::shared::prefix::Toggle::Off,
            "Inkling 2 Mini prefix geometry requires exact admitted frontiers"
        );
        let bank = self.execution.borrow().requests.cache.clone();
        let host = self
            .host
            .clone()
            .map(|config| {
                CudaCopyEngine::registered_retained(self.library, vec![], self.family.snapshot_owners())
                    .map(|engine| (config, engine))
            })
            .transpose()?;
        let host_bytes = host.as_ref().map_or(0, |(config, _)| config.bytes);
        let layout = self.family.layout();
        let cache = PrefixCache::new_with_draft_pages(
            layout,
            PrefixConfig {
                entries: args.prefix_cache_entries,
                mark_slots: bank.mark_slots(),
                keep_logits: true,
                min_tokens: args.prefix_cache_min_tokens,
            },
            host,
            self.family.draft_page_bytes(),
        )?;
        cuteafd_bench::context::set_kv(
            (layout.pages * layout.page_rows) as u64,
            Some((layout.pages * layout.page_rows) as u64),
            layout.pages as u64,
            self.family.record_format(),
            host_bytes,
        );
        Ok((self.family.clone(), cache))
    }

    fn prepare(&self, job: NativeRequest, tokens: Vec<u32>) -> Result<(scheduler::Prompt, RequestMedia, Vec<EncodeJob>)> {
        if self.vision.is_none() {
            ensure!(job.media.is_empty() && job.image_groups.is_empty(), "vision is disabled");
            ensure!(job.audio.is_empty(), "no bundled audio execution owner");
            let keys = cuteafd_engine::media::MediaKeys::new(&tokens, u32::try_from(self.vocab)?, &[])?;
            let media = RequestMedia::new(vec![], HIDDEN, tokens.len())?;
            return Ok((scheduler::Prompt { job, tokens, keys }, media, vec![]));
        }
        self.vision
            .as_ref()
            .context("Mini media binding absent")?
            .prepare(job, tokens, u32::try_from(self.vocab)?, HIDDEN, self.context)
    }

    fn probe_features(&self, _: &scheduler::Prompt, _: &mut RequestMedia, _: &mut EmbeddingCache, _: &Path) -> Result<()> {
        Ok(())
    }
    fn coordinates(&self, _: &[u32], _: &NativeRequest, _: &RequestMedia) -> Result<()> {
        Ok(())
    }
    fn placement(&self, units: Vec<u32>, slot: i32) -> Placement {
        Placement {
            media: None,
            units,
            slot: slot as usize,
            len: 0,
            state_len: 0,
        }
    }
    fn bind_request(&self, placement: &mut Placement, _: (), media: RequestMedia, tokens: &[u32], resume: usize) -> Result<()> {
        ensure!(
            placement.len == resume && placement.state_len == resume && resume <= tokens.len(),
            "Mini restored request frontier differs"
        );
        placement.media = (!media.spans().is_empty()).then_some(media);
        Ok(())
    }
    fn history(&self, placement: &Placement) -> usize {
        placement.slot
    }
    fn require_live(&self) -> Result<()> {
        self.execution.borrow().require_live()
    }
    fn log_table_stats(&self, _: &str) {}

    fn prefill_device(
        &self,
        placement: &mut Placement,
        tokens: &[u32],
        rows: usize,
        pool: &RefPagePool,
    ) -> Result<Option<DeviceLogits>> {
        ensure!(rows <= 1, "Mini prefill head retains one row; scoring streams tiles");
        let mut last = None;
        let projection = (rows > 0).then_some(super::prefill::HeadProjection::MemberLastRow);
        self.append(placement, tokens, projection, pool, &mut |_, _, logits| {
            let offset = (logits.rows - 1) * logits.stride * 4;
            last = Some(DeviceLogits {
                ptr: logits.ptr.wrapping_byte_add(offset),
                rows: 1,
                ..logits
            });
            Ok(())
        })?;
        Ok(last)
    }

    fn streamed_prefill_scoring(&self) -> bool {
        true
    }

    fn prefill_scoring(
        &self,
        placement: &mut Placement,
        tokens: &[u32],
        rows: usize,
        position: usize,
        probe: &probe::ProbeRef,
        pool: &RefPagePool,
    ) -> Result<Option<probe::ScoreLogits>> {
        ensure!(rows <= tokens.len(), "Mini score suffix exceeds submitted rows");
        let first = placement.len + tokens.len() - rows + 1;
        let scoring = probe::ScoringRows {
            consumer: probe,
            position,
        };
        let projection = (rows > 0).then_some(super::prefill::HeadProjection::FromPosition(first));
        self.append(placement, tokens, projection, pool, &mut |_, at, logits| {
            let end = at.checked_add(logits.rows).context("Mini score tile overflow")?;
            if end > first {
                let skip = first.saturating_sub(at);
                let selected = DeviceLogits {
                    ptr: logits.ptr.wrapping_byte_add(skip * logits.stride * 4),
                    rows: logits.rows - skip,
                    ..logits
                };
                scoring.record(self.library, at + skip - first, &selected)?;
            }
            Ok(())
        })?;
        Ok((rows > 0).then_some(probe::ScoreLogits::Recorded))
    }

    fn packed_limits(&self) -> Option<Limits> {
        Some(Limits {
            rows: self.unit_rows,
            chunk_rows: self.unit_rows,
            max_context: self.context,
            taps: None,
        })
    }

    fn prefill_packed(
        &self,
        sequences: &mut [scheduler::PrefillSeq<'_, Placement, Draft>],
        pool: &RefPagePool,
        on_logits: &mut dyn FnMut(usize, DeviceLogits) -> MemberOutcome,
    ) -> Result<Vec<Result<()>>> {
        ensure!(!sequences.is_empty(), "Mini packed submission is empty");
        let limits = self.packed_limits().unwrap();
        ensure!(
            sequences.iter().map(|s| s.placement.slot).collect::<BTreeSet<_>>().len() == sequences.len(),
            "Mini packed members alias a request state slot"
        );
        let chunks = sequences
            .iter()
            .map(|s| {
                ensure!(s.start == s.placement.len, "Mini packed start differs");
                ensure!(
                    !s.neural || self.execution.borrow().draft.is_some(),
                    "Mini packed neural prefill needs admitted drafter"
                );
                self.check(s.placement, s.tokens, pool)?;
                Ok((s.start, s.tokens.len()))
            })
            .collect::<Result<Vec<_>>>()?;
        let segments = crate::shared::prefill_packing::plan(&chunks, &limits)?;
        let rows = segments.last().context("Mini pack has no members")?.end_row();
        let mut placements: Vec<Placement> = sequences.iter().map(|s| s.placement.clone()).collect();
        let sources: Vec<&[u32]> = sequences.iter().map(|s| s.tokens).collect();
        let visits = self.execution.borrow().cfg.total_visits();
        let unit = Unit {
            index: 0,
            sequence: 0,
            rows: 0..rows,
            layers: 0..visits,
            slot: 0,
            barrier: false,
            segments: segments
                .iter()
                .enumerate()
                .map(|(sequence, seg)| UnitSegment {
                    sequence,
                    rows: seg.first..seg.end_row(),
                    tokens: seg.start..seg.start + seg.rows,
                    prompt_end: true,
                    snapshot: false,
                    head: true,
                })
                .collect(),
        };
        let mut last: Vec<Option<Result<()>>> = (0..sequences.len()).map(|_| None).collect();
        self.execution.borrow_mut().prefill(
            &mut placements,
            &sources,
            &[unit],
            super::prefill::HeadProjection::MemberLastRow,
            &mut |member, position, logits| {
                if position + logits.rows == chunks[member].0 + chunks[member].1 + 1 {
                    let offset = (logits.rows - 1) * logits.stride * 4;
                    let logits = DeviceLogits {
                        ptr: logits.ptr.wrapping_byte_add(offset),
                        rows: 1,
                        ..logits
                    };
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
        last.into_iter()
            .map(|o| o.context("Mini packed member last-row callback missing"))
            .collect()
    }

    fn verify_device(
        &self,
        sequences: &mut [(&mut Placement, &[u32])],
        speculative: bool,
        pool: &RefPagePool,
    ) -> Result<Option<DeviceLogits>> {
        self.decode(sequences, speculative, pool)
    }

    fn verify_device_ungraphed(
        &self,
        sequences: &mut [(&mut Placement, &[u32])],
        speculative: bool,
        kind: scheduler::UngraphedKind,
        pool: &RefPagePool,
    ) -> Result<Option<DeviceLogits>> {
        if kind == scheduler::UngraphedKind::Decode {
            return self.decode(sequences, speculative, pool);
        }
        ensure!(!speculative, "Mini scoring replay cannot speculate");
        let logits = self.decode(sequences, false, pool)?;
        let mut drafts = (0..sequences.len()).map(|_| Draft::default()).collect::<Vec<_>>();
        let mut verified = sequences
            .iter_mut()
            .zip(&mut drafts)
            .scan(0, |first_row, ((placement, rows), seq)| {
                let v = scheduler::Verified {
                    start: placement.len - rows.len(),
                    history: placement.slot,
                    placement: &mut **placement,
                    seq,
                    rows: *rows,
                    first_row: *first_row,
                    kept: None,
                    finished: false,
                    neural: false,
                };
                *first_row += rows.len();
                Some(v)
            })
            .collect::<Vec<_>>();
        self.accept(&mut verified, false)?;
        Ok(logits)
    }

    fn verify_bucket_rows(&self, rows: usize, _: bool, _: bool) -> usize {
        rows
    }
    fn verify_attention_trace(&self, _: &[Vec<usize>], rows: usize) -> crate::shared::speculation_trace::Deferred {
        crate::shared::speculation_trace::Deferred::new(move || {
            serde_json::json!({"kind": "inkling2_mini", "rows": rows, "attention": "eager"})
        })
    }

    fn draft_head_kind(&self) -> cuteafd_core::HeadScore {
        if self.execution.borrow().draft.is_some() {
            cuteafd_core::HeadScore::CandidateProbability
        } else {
            cuteafd_core::HeadScore::None
        }
    }
    fn configure_draft_scores(&self, traced: bool, scored: bool) {
        if let Some(draft) = &self.execution.borrow().draft {
            draft.set_scores(traced, scored);
        }
    }
    fn draft_head<'d>(&self, draft: &'d Draft) -> Option<&'d [f32]> {
        (!draft.confidence.is_empty()).then_some(draft.confidence.as_slice())
    }
    fn draft_copies(&self) -> bool {
        false
    }

    fn prefill_draft(&self, p: &Placement, _: usize, _: &[u32], next: Option<u32>, state: &mut Draft) -> Result<()> {
        if let Some(draft) = &self.execution.borrow().draft {
            ensure!(
                draft.frontier(p.slot)?.1 == p.len,
                "Mini draft prefill frontier differs"
            );
        }
        state.anchor = next;
        Ok(())
    }

    fn draft(
        &self,
        sequences: &mut [scheduler::DraftSeq<'_, Placement, Draft>],
        timing: &mut scheduler::DraftTiming,
        _: &mut TokenSelector<'_>,
    ) -> Result<Vec<Vec<u32>>> {
        let execution = self.execution.borrow();
        let Some(draft) = &execution.draft else {
            return Ok(vec![vec![]; sequences.len()]);
        };
        let start = std::time::Instant::now();
        let mut output = vec![vec![]; sequences.len()];
        let unembed = execution.target.operand("model.llm.unembed.weight")?.buffer;
        let embed = execution.target.operand("model.llm.embed.weight")?.buffer;
        let embed_norm = execution.target.operand("model.llm.embed_norm.weight")?.buffer;
        for first in (0..sequences.len()).step_by(draft.sequences) {
            let end = (first + draft.sequences).min(sequences.len());
            let selected: Vec<_> = sequences[first..end]
                .iter()
                .enumerate()
                .filter(|(_, s)| s.depth > 0)
                .filter_map(|(index, s)| s.seq.anchor.map(|anchor| (index, anchor, s)))
                .collect();
            if selected.is_empty() {
                continue;
            }
            let rows: Vec<_> = selected
                .iter()
                .map(|(_, anchor, s)| crate::shared::speculation::dflash::DraftSeq {
                    slot: s.placement.slot,
                    anchor: *anchor,
                    position: s.placement.len,
                    valid_from: 0,
                })
                .collect();
            let proposed = draft.propose(&rows, embed, embed_norm, unembed)?;
            let placed: Vec<_> = selected.into_iter().map(|(index, _, s)| (index, s.depth)).collect();
            for ((index, depth), tokens) in placed.into_iter().zip(proposed) {
                sequences[first + index].seq.confidence = tokens.confidence.into_iter().take(depth).collect();
                output[first + index] = tokens.tokens.into_iter().take(depth).collect();
            }
        }
        timing.steps += 1;
        timing.seconds += start.elapsed().as_secs_f64();
        Ok(output)
    }

    fn accept(&self, verified: &mut [scheduler::Verified<'_, Placement, Draft, usize>], speculative: bool) -> Result<()> {
        let pending = self
            .pending
            .borrow_mut()
            .take()
            .context("Mini accept lacks submitted receipt")?;
        let mut execution = self.execution.borrow_mut();
        let draft = execution.draft.clone();
        match pending {
            Pending::Plain(sent) => {
                ensure!(
                    !speculative && sent.len() == verified.len(),
                    "Mini plain accept inventory differs"
                );
                for ((slot, start, tokens), request) in sent.iter().zip(verified.iter()) {
                    ensure!(
                        *slot == request.history
                            && *slot == request.placement.slot
                            && *start == request.start
                            && tokens == request.rows,
                        "Mini plain accept receipt differs"
                    );
                }
            }
            Pending::Spec { mark, tokens } => {
                use crate::shared::speculation::target::{AcceptedPrefix, BlockVerifyTarget};
                ensure!(
                    speculative
                        && tokens.len() == verified.len()
                        && tokens.iter().zip(verified.iter()).all(|(t, s)| t == s.rows),
                    "Mini speculative accept receipt differs"
                );
                let accepted: Vec<_> = verified
                    .iter()
                    .map(|s| AcceptedPrefix {
                        slot: s.placement.slot,
                        rows: s.kept.map_or(0, |(rows, _)| rows),
                    })
                    .collect();
                let mut placements: Vec<_> = verified.iter().map(|s| s.placement.clone()).collect();
                super::verify_target::Target::new(&mut execution, &mut placements).commit(&mark, &accepted)?;
                for (s, p) in verified.iter_mut().zip(placements) {
                    *s.placement = p;
                }
            }
        }
        if let Some(draft) = draft {
            for s in verified.iter_mut() {
                let keep = if speculative {
                    s.kept.map_or(0, |(rows, _)| rows)
                } else {
                    s.rows.len()
                };
                if keep > 0 {
                    draft.update(&draft.decode, s.placement.slot, s.start, s.first_row, keep)?;
                }
                if let Some((_, anchor)) = s.kept {
                    s.seq.anchor = Some(anchor);
                }
            }
        }
        Ok(())
    }

    fn cancel(&self, _: &Placement) -> Result<()> {
        let execution = self.execution.borrow();
        if let Some(Pending::Spec { mark, .. }) = self.pending.borrow_mut().take() {
            execution.restore_verify(&mark)?;
        }
        execution.requests.cache.drain()?;
        Ok(())
    }
}
