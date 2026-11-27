//! Hy4 (`hy_v4_vl`) adapter for the shared request scheduler (`shared::qwen::scheduler::Backend`).
use super::cache::{Hy4Placement, Hy4Prefix};
use super::execution::{HeadMode, Hy4Engine};
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
    media::{EncodeJob, EmbeddingCache, RequestMedia},
    prefix::{PrefixCache, PrefixConfig, PrefixFamily, RefPagePool},
};
use cuteafd_loader::families::hy4_common::PAGE_ROWS;
use cuteafd_loader::formats::mapped_table::ReadPriority;
use std::cell::RefCell;
use std::collections::BTreeSet;
use std::path::Path;
use std::sync::Arc;

fn check_writable(
    placement: &Hy4Placement,
    rows: usize,
    context: usize,
    pool: &RefPagePool,
) -> Result<()> {
    let end = placement
        .len
        .checked_add(rows)
        .context("Hy4 append extent overflow")?;
    ensure!(
        rows > 0 && end <= context && end.div_ceil(PAGE_ROWS) <= placement.units.len(),
        "Hy4 append exceeds reserved pages"
    );
    let mut writable = BTreeSet::new();
    for position in placement.len..end {
        let page = placement.units[position / PAGE_ROWS];
        ensure!(
            (page as usize) < pool.capacity()
                && pool.refs(page) == 1
                && writable.insert((page, position % PAGE_ROWS)),
            "Hy4 append requires private nonaliased COW pages"
        );
    }
    Ok(())
}

fn decode_spans(
    widths: impl IntoIterator<Item = usize>,
    capacity: usize,
) -> Result<Vec<std::ops::Range<usize>>> {
    let mut rows = 0usize;
    widths
        .into_iter()
        .map(|width| {
            ensure!(width > 0, "Hy4 decode member has no rows");
            let end = rows
                .checked_add(width)
                .context("Hy4 decode row extent overflow")?;
            ensure!(end <= capacity, "Hy4 decode exceeds admitted {capacity}-row owner");
            let span = rows..end;
            rows = end;
            Ok(span)
        })
        .collect()
}

#[derive(Default)]
pub(crate) struct Draft {
    anchor: Option<u32>,
}

impl scheduler::DraftState for Draft {
    fn pending_rows(&self) -> usize {
        usize::from(self.anchor.is_some())
    }
    fn close(&mut self, token: u32) {
        self.anchor = Some(token);
    }
}

fn scoring_replay(
    sequences: &mut [(&mut Hy4Placement, &[u32])],
    speculative: bool,
    decode: impl FnOnce(&mut [(&mut Hy4Placement, &[u32])]) -> Result<Option<DeviceLogits>>,
    accept: impl FnOnce(&mut [scheduler::Verified<'_, Hy4Placement, Draft, usize>]) -> Result<()>,
) -> Result<Option<DeviceLogits>> {
    ensure!(!speculative, "Hy4 scoring replay cannot speculate");
    let logits = decode(sequences)?;
    let mut drafts = (0..sequences.len()).map(|_| Draft::default()).collect::<Vec<_>>();
    let mut verified = sequences
        .iter_mut()
        .zip(&mut drafts)
        .scan(0, |first_row, ((placement, rows), seq)| {
            let verified = scheduler::Verified {
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
            Some(verified)
        })
        .collect::<Vec<_>>();
    accept(&mut verified)?;
    Ok(logits)
}

/// Rows the scheduler hands one prefill call: the full multi-wave span, so
/// `sequence_units` cuts it into `waves` units that overlap on the Spark
/// exchange. One unit per call serialized the waves (#2550).
fn prefill_span(unit_rows: usize, waves: usize) -> usize {
    unit_rows * waves.max(1)
}

pub(crate) struct Hy4Backend<'a> {
    engine: RefCell<Hy4Engine<'a>>,
    family: Hy4Prefix<'a>,
    library: &'a cuteafd_ffi::NativeLibrary,
    context: usize,
    concurrency: usize,
    decode_rows: usize,
    unit_rows: usize,
    prefill_waves: usize,
    vocab: usize,
    host: Option<cuteafd_hostcache::config::Config>,
    profile: RefCell<[f64; 2]>,
    pending: RefCell<Option<Vec<(usize, usize, Vec<u32>)>>>,
    /// Bundled 40-layer ViT contract when vision is enabled (None: VISION=off).
    vision: Option<cuteafd_loader::families::hy4_flash::vision::VisionContract>,
    config: serde_json::Value,
}

impl<'a> Hy4Backend<'a> {
    pub(crate) fn new(
        engine: Hy4Engine<'a>,
        context: usize,
        unit_rows: usize,
        host: Option<cuteafd_hostcache::config::Config>,
        snapshot: &Path,
        vision: bool,
    ) -> Result<Self> {
        let config: serde_json::Value = cuteafd_loader::read_json_file(&snapshot.join("config.json"))?;
        let vision = vision
            .then(|| cuteafd_loader::families::hy4::vision::vision_contract(&config))
            .transpose()?;
        let library = engine.library();
        let family = engine.prefix()?;
        let concurrency = engine.max_sequences;
        let decode_rows = engine.decode_rows;
        let prefill_waves = engine.prefill_waves();
        let vocab = engine.cfg.vocab_size;
        let layout = family.layout();
        ensure!(
            context > 0 && context <= engine.cfg.max_position && context <= layout.pages * PAGE_ROWS,
            "Hy4 served context {context} outside admitted pool/checkpoint"
        );
        ensure!(
            unit_rows > 0 && unit_rows <= engine.max_rows,
            "Hy4 unit_rows {unit_rows} outside 1..={}",
            engine.max_rows
        );
        Ok(Self {
            engine: RefCell::new(engine),
            family,
            library,
            context,
            concurrency,
            decode_rows,
            unit_rows,
            prefill_waves,
            vocab,
            host,
            profile: RefCell::new([0.0; 2]),
            pending: RefCell::new(None),
            vision,
            config,
        })
    }

    fn check(&self, placement: &Hy4Placement, tokens: &[u32], pool: &RefPagePool) -> Result<()> {
        ensure!(
            !tokens.is_empty()
                && tokens.iter().all(|&token| (token as usize) < self.vocab)
                && placement.slot < self.concurrency,
            "Hy4 append token/slot differs"
        );
        check_writable(placement, tokens.len(), self.context, pool)
    }

    pub(crate) fn enable_expert_usage(&self) {
        self.engine.borrow_mut().enable_expert_usage();
    }

    fn decode(
        &self,
        sequences: &mut [(&mut Hy4Placement, &[u32])],
        speculative: bool,
        pool: &RefPagePool,
        allow_graphs: bool,
    ) -> Result<Option<DeviceLogits>> {
        ensure!(
            !sequences.is_empty()
                && sequences.len() <= self.concurrency
                && self.pending.borrow().is_none(),
            "Hy4 decode requires admitted requests and prior commit"
        );
        let spans = decode_spans(sequences.iter().map(|(_, tokens)| tokens.len()), self.decode_rows)?;
        let total_rows = spans.last().context("Hy4 decode has no rows")?.end;
        let mut slots = BTreeSet::new();
        let mut segments = Vec::with_capacity(sequences.len());
        let mut placements = Vec::with_capacity(sequences.len());
        let mut sources = Vec::with_capacity(sequences.len());
        let mut receipt = Vec::with_capacity(sequences.len());
        for (index, (placement, tokens)) in sequences.iter().enumerate() {
            self.check(placement, tokens, pool)?;
            ensure!(slots.insert(placement.slot), "Hy4 decode aliases a request state slot");
            segments.push(UnitSegment {
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
        let layers = self.engine.borrow().cfg.layers;
        let unit = Unit {
            index: 0,
            sequence: 0,
            segments,
            rows: 0..total_rows,
            layers: 0..layers,
            slot: 0,
            barrier: false,
        };
        let mut logits = None;
        let mut engine = self.engine.borrow_mut();
        engine.set_provisional(speculative)?;
        let mut run = |engine: &mut Hy4Engine<'a>| {
            engine.run_units(
                &mut placements,
                &sources,
                &[unit],
                ReadPriority::Decode,
                |_| HeadMode::Suffix(0),
                &mut |_, _, tile| {
                    logits = Some(tile);
                    Ok(())
                },
            )
        };
        let result = if allow_graphs {
            run(&mut engine)
        } else {
            engine.without_graphs_mut(run)
        };
        engine.set_provisional(false)?;
        result?;
        for ((placement, _), next) in sequences.iter_mut().zip(placements) {
            **placement = next;
        }
        *self.pending.borrow_mut() = Some(receipt);
        Ok(logits)
    }
}

impl<'a> Backend<'a> for Hy4Backend<'a> {
    type Placement = Hy4Placement;
    type Draft = Draft;
    type History = usize;
    type Coordinates = Arc<cuteafd_loader::families::hy4::vision::RopePositions>;
    type Encoder = crate::shared::qwen::Encoder;
    type Meta = ();
    type Prefix = Hy4Prefix<'a>;

    fn geometry(&self) -> scheduler::Geometry {
        scheduler::Geometry {
            family: "hy4",
            vocab_size: self.vocab,
            max_context: self.context,
            slots: self.concurrency,
            prefill_rows: prefill_span(self.unit_rows, self.prefill_waves),
            decode_rows: self.decode_rows,
        }
    }

    fn message_starts(&self) -> &[&str] {
        &["<｜hy_start:opensource｜>"]
    }

    fn library(&self) -> &'a cuteafd_ffi::NativeLibrary {
        self.library
    }

    fn profile(&self) -> &RefCell<[f64; 2]> {
        &self.profile
    }

    fn selector_ready(&self, selector: &mut TokenSelector<'a>) -> Result<()> {
        selector.reserve_sampler()
    }

    fn prefix_cache(
        &self,
        args: &PrefixArgs,
        concurrency: usize,
    ) -> Result<(Self::Prefix, PrefixCache<CudaCopyEngine<'a>>)> {
        ensure!(
            concurrency == self.concurrency && args.prefix_partial == crate::shared::prefix::Toggle::Off,
            "Hy4 prefix geometry requires exact admitted frontiers"
        );
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
        let cache = PrefixCache::new(
            layout,
            PrefixConfig {
                entries: args.prefix_cache_entries,
                mark_slots: 0,
                keep_logits: true,
                min_tokens: args.prefix_cache_min_tokens,
            },
            host,
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

    fn prepare(
        &self,
        job: NativeRequest,
        tokens: Vec<u32>,
    ) -> Result<(scheduler::Prompt, RequestMedia, Vec<EncodeJob>)> {
        let hidden = self.engine.borrow().cfg.hidden;
        super::media::prepare(
            job,
            tokens,
            &self.config,
            self.vision.as_ref(),
            self.vocab,
            hidden,
            self.context,
        )
    }

    fn probe_features(
        &self,
        prompt: &scheduler::Prompt,
        media: &mut RequestMedia,
        cache: &mut EmbeddingCache,
        snapshot: &Path,
    ) -> Result<()> {
        if prompt
            .job
            .probe
            .as_ref()
            .is_some_and(|p| p.spec.feature_scoring())
            && !media.spans().is_empty()
            && std::env::var_os("CUTEAFD_MEDIA_FEATURES_DIR").is_some()
        {
            crate::shared::probe::still::features(
                prompt,
                media,
                cache,
                snapshot,
                cuteafd_loader::media::ImageFamily::Hy4,
            )?;
        }
        Ok(())
    }

    fn coordinates(
        &self,
        tokens: &[u32],
        job: &NativeRequest,
        media: &RequestMedia,
    ) -> Result<Self::Coordinates> {
        super::media::rope_positions(tokens, job, media, self.vision.as_ref())
    }

    fn placement(&self, units: Vec<u32>, slot: i32) -> Hy4Placement {
        Hy4Placement::new(units, slot as usize)
    }

    fn bind_request(
        &self,
        placement: &mut Hy4Placement,
        rope: Self::Coordinates,
        media: RequestMedia,
        tokens: &[u32],
        resume: usize,
    ) -> Result<()> {
        ensure!(
            placement.len == resume && resume <= tokens.len(),
            "Hy4 restored request frontier differs"
        );
        placement.rope = rope;
        placement.media = (!media.spans().is_empty()).then(|| Arc::new(media));
        Ok(())
    }

    fn restore_draft(&self, placement: &Hy4Placement, _: usize, _: &[u32], _: &mut Draft) -> Result<()> {
        self.engine.borrow().invalidate_draft(placement.slot);
        Ok(())
    }

    fn history(&self, placement: &Hy4Placement) -> usize {
        placement.slot
    }

    fn require_live(&self) -> Result<()> {
        self.engine.borrow().require_live()
    }

    fn log_table_stats(&self, _phase: &str) {}

    fn prefill_device(
        &self,
        placement: &mut Hy4Placement,
        tokens: &[u32],
        rows: usize,
        pool: &RefPagePool,
    ) -> Result<Option<DeviceLogits>> {
        ensure!(rows <= 1, "Hy4 prefill head retains one row; scoring streams tiles");
        self.check(placement, tokens, pool)?;
        let start = placement.len;
        let layers = self.engine.borrow().cfg.layers;
        let mut units = sequence_units(tokens.len(), self.unit_rows, self.prefill_waves, layers)?;
        let last_idx = units.len() - 1;
        for (i, unit) in units.iter_mut().enumerate() {
            let u_start = start + unit.rows.start;
            let u_end = start + unit.rows.end;
            unit.rows = u_start..u_end;
            unit.segments = vec![UnitSegment {
                sequence: 0,
                rows: 0..unit.rows.len(),
                tokens: u_start..u_end,
                prompt_end: i == last_idx,
                snapshot: false,
                head: rows > 0 && i == last_idx,
            }];
        }
        let mut next = [placement.clone()];
        let mut last = None;
        self.engine.borrow_mut().run_units(
            &mut next,
            &[tokens],
            &units,
            ReadPriority::Prefill,
            |unit| {
                if rows > 0 && unit.index == last_idx {
                    HeadMode::LastPerSeq
                } else {
                    HeadMode::None
                }
            },
            &mut |_, _, logits| {
                last = Some(logits);
                Ok(())
            },
        )?;
        *placement = next.into_iter().next().unwrap();
        Ok(last)
    }

    fn streamed_prefill_scoring(&self) -> bool {
        true
    }

    fn prefill_scoring(
        &self,
        placement: &mut Hy4Placement,
        tokens: &[u32],
        rows: usize,
        position: usize,
        probe: &probe::ProbeRef,
        pool: &RefPagePool,
    ) -> Result<Option<probe::ScoreLogits>> {
        ensure!(rows <= tokens.len(), "Hy4 score suffix exceeds submitted rows");
        self.check(placement, tokens, pool)?;
        let start = placement.len;
        let first = start + tokens.len() - rows + 1;
        let layers = self.engine.borrow().cfg.layers;
        let mut units = sequence_units(tokens.len(), self.unit_rows, self.prefill_waves, layers)?;
        let last_idx = units.len() - 1;
        for (i, unit) in units.iter_mut().enumerate() {
            let u_start = start + unit.rows.start;
            let u_end = start + unit.rows.end;
            unit.rows = u_start..u_end;
            unit.segments = vec![UnitSegment {
                sequence: 0,
                rows: 0..unit.rows.len(),
                tokens: u_start..u_end,
                prompt_end: i == last_idx,
                snapshot: false,
                head: rows > 0 && u_end + 1 > first,
            }];
        }
        let scoring = probe::ScoringRows {
            consumer: probe,
            position,
        };
        let mut next = [placement.clone()];
        self.engine.borrow_mut().run_units(
            &mut next,
            &[tokens],
            &units,
            ReadPriority::Prefill,
            |unit| {
                if rows == 0 || unit.rows.end + 1 <= first {
                    HeadMode::None
                } else {
                    HeadMode::Suffix(first.saturating_sub(unit.rows.start + 1))
                }
            },
            &mut |unit, row_off, logits| {
                let at = unit.rows.start + row_off + 1;
                ensure!(at >= first, "Hy4 scoring tile started before first scored position");
                scoring.record(self.library, at - first, &logits)
            },
        )?;
        *placement = next.into_iter().next().unwrap();
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
        sequences: &mut [scheduler::PrefillSeq<'_, Hy4Placement, Draft>],
        pool: &RefPagePool,
        on_logits: &mut dyn FnMut(usize, DeviceLogits) -> MemberOutcome,
    ) -> Result<Vec<Result<()>>> {
        ensure!(!sequences.is_empty(), "Hy4 packed submission is empty");
        ensure!(sequences.len() <= 64, "Hy4 packed member count exceeds 64-row head");
        let limits = self.packed_limits().unwrap();
        ensure!(
            sequences.iter().map(|s| s.placement.slot).collect::<BTreeSet<_>>().len() == sequences.len(),
            "Hy4 packed members alias a request state slot"
        );
        let chunks = sequences
            .iter()
            .map(|sequence| {
                ensure!(sequence.start == sequence.placement.len, "Hy4 packed start differs");
                self.check(sequence.placement, sequence.tokens, pool)?;
                Ok((sequence.start, sequence.tokens.len()))
            })
            .collect::<Result<Vec<_>>>()?;
        let segments = crate::shared::prefill_packing::plan(&chunks, &limits)?;
        let rows = segments.last().context("Hy4 pack has no members")?.end_row();
        let mut placements: Vec<Hy4Placement> = sequences.iter().map(|s| s.placement.clone()).collect();
        let sources: Vec<&[u32]> = sequences.iter().map(|s| s.tokens).collect();
        let layers = self.engine.borrow().cfg.layers;
        let unit = Unit {
            index: 0,
            sequence: 0,
            rows: 0..rows,
            layers: 0..layers,
            slot: 0,
            barrier: false,
            segments: segments
                .iter()
                .enumerate()
                .map(|(sequence, segment)| UnitSegment {
                    sequence,
                    rows: segment.first..segment.end_row(),
                    tokens: segment.start..segment.start + segment.rows,
                    prompt_end: true,
                    snapshot: false,
                    head: true,
                })
                .collect(),
        };
        let mut outcomes: Vec<Option<MemberOutcome>> = (0..sequences.len()).map(|_| None).collect();
        self.engine.borrow_mut().run_units(
            &mut placements,
            &sources,
            &[unit],
            ReadPriority::Prefill,
            |_| HeadMode::LastPerSeq,
            &mut |_, _, logits| {
                ensure!(logits.rows == sequences.len(), "Hy4 packed head rows mismatch");
                for member in 0..sequences.len() {
                    let row = DeviceLogits {
                        ptr: logits.ptr.wrapping_byte_add(member * logits.stride * 4),
                        rows: 1,
                        ..logits
                    };
                    outcomes[member] = Some(on_logits(member, row));
                }
                Ok(())
            },
        )?;
        for (sequence, placement) in sequences.iter_mut().zip(placements) {
            *sequence.placement = placement;
        }
        outcomes
            .into_iter()
            .map(|o| o.context("Hy4 packed member last-row callback missing")?)
            .collect()
    }

    fn verify_device(
        &self,
        sequences: &mut [(&mut Hy4Placement, &[u32])],
        speculative: bool,
        pool: &RefPagePool,
    ) -> Result<Option<DeviceLogits>> {
        self.decode(sequences, speculative, pool, true)
    }

    fn verify_device_ungraphed(
        &self,
        sequences: &mut [(&mut Hy4Placement, &[u32])],
        speculative: bool,
        kind: scheduler::UngraphedKind,
        pool: &RefPagePool,
    ) -> Result<Option<DeviceLogits>> {
        if kind == scheduler::UngraphedKind::Decode {
            return self.decode(sequences, speculative, pool, false);
        }
        scoring_replay(
            sequences,
            speculative,
            |sequences| self.decode(sequences, false, pool, false),
            |verified| self.accept(verified, false),
        )
    }

    fn verify_bucket_rows(&self, rows: usize, _: bool, _: bool) -> usize {
        rows
    }

    fn verify_attention_trace(
        &self,
        _: &[Vec<usize>],
        rows: usize,
    ) -> crate::shared::speculation_trace::Deferred {
        let mode = if self.engine.borrow().graphs_active() {
            "graphed"
        } else {
            "eager"
        };
        crate::shared::speculation_trace::Deferred::new(move || {
            serde_json::json!({"kind": "hy4", "rows": rows, "attention": mode})
        })
    }

    fn prefill_draft(
        &self,
        _: &Hy4Placement,
        _: usize,
        _: &[u32],
        next: Option<u32>,
        state: &mut Draft,
    ) -> Result<()> {
        state.anchor = next;
        Ok(())
    }

    fn draft(
        &self,
        sequences: &mut [scheduler::DraftSeq<'_, Hy4Placement, Draft>],
        timing: &mut scheduler::DraftTiming,
        _: &mut TokenSelector<'_>,
    ) -> Result<Vec<Vec<u32>>> {
        let start = std::time::Instant::now();
        let engine = self.engine.borrow();
        let selected = sequences
            .iter()
            .enumerate()
            .filter_map(|(i, request)| {
                let bonus = request.seq.anchor?;
                let p = request.placement;
                (request.depth > 0 && engine.draft_valid(p)).then_some((
                    i,
                    super::draft::Member {
                        slot: p.slot,
                        pos: p.len,
                        bonus,
                        depth: request.depth,
                        rope_delta: 0,
                    },
                ))
            })
            .collect::<Vec<_>>();
        let mut output = vec![Vec::new(); sequences.len()];
        if !selected.is_empty() {
            let members: Vec<_> = selected.iter().map(|(_, m)| *m).collect();
            let proposals = engine.draft_batch(&members)?;
            ensure!(
                proposals.len() == selected.len(),
                "Hy4 DFly2 batch returned the wrong member count"
            );
            for ((index, _), tokens) in selected.into_iter().zip(proposals) {
                output[index] = tokens;
            }
            timing.steps += 1;
        }
        timing.seconds += start.elapsed().as_secs_f64();
        Ok(output)
    }

    fn accept(
        &self,
        verified: &mut [scheduler::Verified<'_, Hy4Placement, Draft, usize>],
        speculative: bool,
    ) -> Result<()> {
        let sent = self
            .pending
            .borrow_mut()
            .take()
            .context("Hy4 accept lacks submitted receipt")?;
        ensure!(sent.len() == verified.len(), "Hy4 accept inventory differs");
        for ((slot, start, tokens), request) in sent.iter().zip(verified.iter()) {
            ensure!(
                *slot == request.history
                    && *slot == request.placement.slot
                    && *start == request.start
                    && tokens == request.rows,
                "Hy4 accept receipt differs"
            );
        }
        let engine = self.engine.borrow();
        for request in verified {
            let Some((kept, after)) = request.kept else {
                continue;
            };
            if speculative {
                ensure!(
                    kept > 0
                        && kept <= request.rows.len()
                        && request.placement.len == request.start + kept,
                    "Hy4 provisional acceptance moved outside the committed prefix"
                );
            }
            if request.neural {
                if speculative {
                    engine.accept_draft(
                        request.history,
                        request.first_row + kept - 1,
                        request.placement.len,
                    )?;
                }
                request.seq.anchor = Some(after);
            }
        }
        Ok(())
    }

    fn cancel(&self, placement: &Hy4Placement) -> Result<()> {
        let _ = self.pending.borrow_mut().take();
        let engine = self.engine.borrow();
        engine.invalidate_draft(placement.slot);
        engine.drain()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn probe_records_hy4_prefill_units_and_avoids_duplicate_first_request_reconciliation() -> Result<()> {
        use crate::shared::prefill_pipeline::{FfnRoute, PrefillHooks, run_prefill_units};
        use crate::shared::qwen::scheduler::observe_prefill;
        use cuteafd_api::openai::probe::{Probe, ProbeSpec};

        struct NoopHooks;
        impl PrefillHooks for NoopHooks {
            type Ticket = ();
            fn route(&self, _: usize) -> FfnRoute { FfnRoute::Spark }
            fn begin_unit(&mut self, _: &Unit) -> Result<()> { Ok(()) }
            fn attention(&mut self, _: usize, _: &Unit) -> Result<()> { Ok(()) }
            fn local_ffn(&mut self, _: usize, _: &Unit) -> Result<()> { Ok(()) }
            fn send_experts(&mut self, _: usize, _: &Unit) -> Result<()> { Ok(()) }
            fn land_experts(&mut self, _: usize, _: &Unit, _: ()) -> Result<()> { Ok(()) }
            fn end_unit(&mut self, _: &Unit) -> Result<()> { Ok(()) }
            fn commit_unit(&mut self, _: &Unit) -> Result<()> { Ok(()) }
        }

        let seed = Some(Probe::new(ProbeSpec { record_rows: 1, ..ProbeSpec::default() }));
        let ((), seed_units) = observe_prefill(|| {
            let units = sequence_units(2752, 2752, 3, 2)?;
            run_prefill_units(&mut NoopHooks, &units, 3)
        })?;
        probe::prefill_units(&seed, seed_units);
        assert_eq!(seed.as_ref().unwrap().record().last_prefill_units, Some(2));

        let warm = Some(Probe::new(ProbeSpec { record_rows: 1, ..ProbeSpec::default() }));
        let ((), warm_units) = observe_prefill(|| {
            let units = sequence_units(64, 2752, 3, 2)?;
            run_prefill_units(&mut NoopHooks, &units, 3)
        })?;
        probe::prefill_units(&warm, warm_units);
        assert_eq!(warm.as_ref().unwrap().record().last_prefill_units, Some(1));

        let backend = include_str!("backend.rs").split("#[cfg(test)]").next().unwrap();
        assert!(!backend.contains("reconcile::capture(\"first_request\")"),
            "Hy4 must rely on the shared scheduler's post-retirement first_request reconciliation");
        Ok(())
    }

    /// The scheduler chunks a prompt at `geometry().prefill_rows`; `prefill_device`
    /// then cuts each chunk into wave units. At the shipping 1x RTX + 4 Spark shape
    /// (3 x 2752) an 8K prompt must run as one call of three overlapped units, not
    /// three serial calls of [1408, 1344], [1408, 1344], [2688] (#2550).
    #[test]
    fn scheduler_geometry_hands_prefill_the_full_wave_span() -> Result<()> {
        use cuteafd_loader::serving_capacity::prefill_waves::{prefill_shape, PrefillOverrides, PrefillPlan};
        let shape = prefill_shape(4, cuteafd_loader::plan::Family::prefill_limits(&cuteafd_loader::plan::families::hy4::HY4),
            PrefillOverrides::default())?;
        assert_eq!(shape, PrefillPlan { unit_rows: 2752, waves: 3, span_rows: 8256 });
        let (unit_rows, waves) = (shape.unit_rows as usize, shape.waves as usize);
        let rows = prefill_span(unit_rows, waves);
        assert_eq!(rows as u64, shape.span_rows);
        let policy = cuteafd_engine::prefix::PointPolicy { gap: 0, boundaries: 0, per_request: 0 };
        let calls = cuteafd_engine::prefix::plan_points(0, 8192, rows, &[], 0, 0, policy).chunks;
        assert_eq!(calls, vec![8192]);
        let units = sequence_units(8192, unit_rows, waves, 1)?;
        assert_eq!(units.iter().map(|u| (u.rows.clone(), u.slot)).collect::<Vec<_>>(),
            vec![(0..2752, 0), (2752..5504, 1), (5504..8192, 2)]);
        // The old one-unit geometry: three serial calls, five units, two in flight at most.
        let old = cuteafd_engine::prefix::plan_points(0, 8192, unit_rows, &[], 0, 0, policy).chunks;
        assert_eq!(old, vec![2752, 5504, 8192]);
        assert_eq!(sequence_units(2752, unit_rows, waves, 1)?.iter().map(|u| u.rows.len()).collect::<Vec<_>>(),
            vec![1408, 1344]);
        // `geometry()` composes exactly this span.
        let backend = include_str!("backend.rs").split("#[cfg(test)]").next().unwrap();
        assert!(backend.contains("prefill_rows: prefill_span(self.unit_rows, self.prefill_waves),"));
        assert_eq!(prefill_span(4096, 1), 4096);
        Ok(())
    }
}
