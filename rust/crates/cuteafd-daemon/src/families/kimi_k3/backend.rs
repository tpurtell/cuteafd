//! Text-only target and opt-in native MLA adapter to the shared scheduler.
use super::{engine::Engine, prefix::{KimiPrefix, Placement}};
use crate::shared::{draft::routes::RoundRoutes, prefix::{CudaCopyEngine, PrefixArgs, Toggle},
    qwen::scheduler::{self, Backend}, token_io::DeviceLogits};
use anyhow::{ensure, Result};
use cuteafd_api::openai::NativeRequest;
use cuteafd_engine::{media::{EncodeJob, EmbeddingCache, MediaKeys, RequestMedia},
    prefix::{PrefixCache, PrefixConfig, PrefixFamily, RefPagePool}};
use std::{cell::{Cell, RefCell}, path::Path};

#[derive(Debug, Default)]
pub(crate) struct DraftState {
    anchor: Option<u32>,
    ready: bool,
    valid_from: usize,
}
impl scheduler::DraftState for DraftState {
    fn pending_rows(&self) -> usize { usize::from(self.ready && self.anchor.is_some()) }
    fn close(&mut self, token: u32) { self.anchor = Some(token); }
}

pub(crate) struct KimiBackend<'e, 'a> {
    engine: &'e Engine<'a>,
    profile: RefCell<[f64; 2]>,
    geometry: scheduler::Geometry,
    steady_census: Cell<bool>,
    origins: RefCell<Vec<usize>>,
}

impl<'e, 'a> KimiBackend<'e, 'a> {
    pub(crate) fn new(engine: &'e Engine<'a>) -> Result<Self> {
        Ok(Self { engine, profile: RefCell::new([0.0; 2]), geometry: engine.geometry()?,
            steady_census: Cell::new(false), origins: RefCell::new(vec![0; engine.geometry()?.slots]) })
    }
}

impl<'e, 'a> Backend<'a> for KimiBackend<'e, 'a> {
    type Placement = Placement;
    type Draft = DraftState;
    type History = ();
    type Coordinates = ();
    type Encoder = crate::shared::qwen::Encoder;
    type Meta = super::prefix::SnapshotMeta;
    type Prefix = KimiPrefix<'a>;

    fn geometry(&self) -> scheduler::Geometry {
        scheduler::Geometry { family: "kimi_k3", vocab_size: self.geometry.vocab_size,
            max_context: self.geometry.max_context, slots: self.geometry.slots,
            prefill_rows: self.geometry.prefill_rows, decode_rows: self.geometry.decode_rows,
            full_prefill_logits: false }
    }
    fn message_starts(&self) -> &[&str] { &["<|open|>"] }
    fn library(&self) -> &'a cuteafd_ffi::NativeLibrary { self.engine.library() }
    fn profile(&self) -> &RefCell<[f64; 2]> { &self.profile }
    fn prefix_cache(&self, args: &PrefixArgs, lanes: usize) -> Result<(Self::Prefix, PrefixCache<CudaCopyEngine<'a>, Self::Meta>)> {
        ensure!(args.prefix_partial == Toggle::Off, "Kimi restores exact KDA/MLA snapshots only");
        let args = args.with_scoring(self.geometry.decode_rows, self.geometry.prefill_rows, self.geometry.vocab_size)?;
        let family = self.engine.prefix_family()?;
        let layout = family.layout();
        let host = args.host_config(layout, self.geometry.max_context, lanes)?.map(|config|
            CudaCopyEngine::registered_retained(self.library(), family.snapshot_owners(), family.snapshot_retained()).map(|copy| (config, copy)))
            .transpose()?;
        let host_bytes = host.as_ref().map_or(0, |(config, _)| config.bytes);
        let config = PrefixConfig { entries: args.prefix_cache_entries, mark_slots: family.slots(),
            keep_logits: true, min_tokens: args.prefix_cache_min_tokens };
        let cache = PrefixCache::new_with_draft_pages(layout, config, host, family.draft_page_bytes())?;
        cuteafd_bench::context::set_kv((layout.pages * layout.page_rows) as u64, self.engine.max_pool_tokens_no_onboard, layout.pages as u64,
            family.record_format(), host_bytes);
        Ok((family, cache))
    }
    fn prepare(&self, job: NativeRequest, tokens: Vec<u32>) -> Result<(scheduler::Prompt, RequestMedia, Vec<EncodeJob>)> {
        ensure!(job.media.is_empty(), "Kimi serving currently supports text only");
        let keys = MediaKeys::new(&tokens, u32::try_from(self.geometry.vocab_size)?, &[])?;
        let media = RequestMedia::new(Vec::new(), 7168, tokens.len())?;
        Ok((scheduler::Prompt { job, tokens, keys }, media, Vec::new()))
    }
    fn probe_features(&self, prompt: &scheduler::Prompt, media: &mut RequestMedia,
        _cache: &mut EmbeddingCache, _snapshot: &Path) -> Result<()> {
        ensure!(prompt.job.media.is_empty() && media.spans().is_empty(), "Kimi media is not admitted");
        Ok(())
    }
    fn coordinates(&self, _tokens: &[u32], job: &NativeRequest, media: &RequestMedia) -> Result<()> {
        ensure!(job.media.is_empty() && media.spans().is_empty(), "Kimi text coordinates require no media");
        Ok(())
    }
    fn placement(&self, units: Vec<u32>, slot: i32) -> Placement {
        Placement { units, slot: slot as usize, len: 0, state_len: 0 }
    }
    fn bind_request(&self, placement: &mut Placement, _coordinates: (), _media: RequestMedia,
        _tokens: &[u32], resume: usize) -> Result<()> {
        if let Some(draft) = self.engine.native_draft() {
            let (origin, end) = draft.frontier(placement.slot)?;
            ensure!(end == resume, "Kimi draft binding differs from prefix admission");
            self.origins.borrow_mut()[placement.slot] = origin;
        }
        Ok(())
    }
    fn history(&self, _placement: &Placement) {}
    fn restore_draft(&self, placement: &Placement, frontier: usize, _tokens: &[u32], seq: &mut DraftState) -> Result<()> {
        let draft = self.engine.native_draft().ok_or_else(|| anyhow::anyhow!("Kimi drafter was not admitted"))?;
        let (origin, end) = draft.frontier(placement.slot)?;
        ensure!(end == frontier, "Kimi draft restored frontier differs from target");
        seq.ready = end > origin;
        seq.valid_from = origin;
        Ok(())
    }
    fn needs_snapshot_filter(&self, neural: bool) -> bool { neural }
    fn snapshot_compatible(&self, saved: &Self::Meta, frontier: usize, neural: bool) -> bool {
        !neural || saved.draft == Some((0, frontier))
    }

    fn require_live(&self) -> Result<()> { self.engine.require_live() }
    fn log_table_stats(&self, _phase: &str) {}
    fn prefill_device(&self, placement: &mut Placement, tokens: &[u32], logit_rows: usize,
        pool: &RefPagePool) -> Result<Option<DeviceLogits>> {
        self.engine.step_device(&mut [(placement, tokens)], pool, logit_rows.max(1), true)
            .map(|logits| (logit_rows > 0).then_some(logits))
    }
    fn packed_limits(&self) -> Option<crate::shared::prefill_packing::Limits> {
        self.engine.packed_limits().ok()
    }
    fn prefill_packed(&self, sequences: &mut [scheduler::PrefillSeq<'_, Placement, DraftState>],
        pool: &RefPagePool, on_logits: &mut dyn FnMut(usize, DeviceLogits)
            -> crate::shared::prefill_packing::MemberOutcome) -> Result<Vec<Result<()>>> {
        let segments = crate::shared::prefill_packing::plan(&sequences.iter()
            .map(|s| (s.start, s.tokens.len())).collect::<Vec<_>>(), &self.engine.packed_limits()?)?;
        let mut target = Vec::with_capacity(sequences.len());
        let mut drafts = Vec::with_capacity(sequences.len());
        for sequence in sequences {
            ensure!(sequence.placement.len == sequence.start, "Kimi packed member frontier differs");
            let slot = sequence.placement.slot;
            drafts.push((slot, sequence.next, sequence.neural, &mut *sequence.draft));
            target.push((&mut *sequence.placement, sequence.tokens));
        }
        self.engine.prefill_packed(&mut target, pool, &mut |i, logits| {
            let (slot, next, neural, state) = &mut drafts[i];
            if *neural {
                let draft = self.engine.native_draft().ok_or_else(|| anyhow::anyhow!("Kimi drafter not admitted"))?;
                let segment = segments[i];
                let rows = (0..segment.tap_rows).map(|offset| crate::shared::speculation::dflash::ContextRow {
                    tap_row: segment.tap_offset + offset, slot: *slot, position: segment.start + offset,
                }).collect::<Vec<_>>();
                draft.update(&rows)?;
                state.ready = true;
                state.valid_from = self.origins.borrow()[*slot];
                state.anchor = *next;
            }
            on_logits(i, logits)
        })
    }
    fn verify_device(&self, sequences: &mut [(&mut Placement, &[u32])], speculative: bool,
        pool: &RefPagePool) -> Result<Option<DeviceLogits>> {
        if speculative { return self.engine.verify_device(sequences, pool, true).map(Some); }
        let rows = sequences.iter().map(|(_, tokens)| tokens.len()).sum();
        self.engine.step_device(sequences, pool, rows, true).map(Some)
    }
    fn verify_device_ungraphed(&self, sequences: &mut [(&mut Placement, &[u32])], speculative: bool,
        _kind: scheduler::UngraphedKind, pool: &RefPagePool) -> Result<Option<DeviceLogits>> {
        if speculative { return self.engine.verify_device(sequences, pool, false).map(Some); }
        let rows = sequences.iter().map(|(_, tokens)| tokens.len()).sum();
        self.engine.step_device(sequences, pool, rows, false).map(Some)
    }
    fn verify_bucket_rows(&self, rows: usize, _speculative: bool, _diagnostic: bool) -> usize { rows }
    fn verify_attention_trace(&self, contexts: &[Vec<usize>], rows: usize) -> serde_json::Value {
        serde_json::json!({"kind":"kimi_owned_mla_tp2_kda", "rows":rows, "contexts":contexts})
    }
    fn draft_geometry(&self, snapshot: &Path, width: usize) -> Result<cuteafd_core::PolicyGeometry> {
        ensure!(self.engine.native_draft().is_some(), "Kimi drafter was not admitted");
        self.engine.draft_geometry(snapshot, width)
    }
    fn arm_draft_probe(&self) -> Result<()> { self.engine.arm_draft_probe() }
    fn prefill_draft(&self, placement: &Placement, start: usize, chunk: &[u32], next: Option<u32>,
        seq: &mut DraftState) -> Result<()> {
        let draft = self.engine.native_draft().ok_or_else(|| anyhow::anyhow!("Kimi drafter was not admitted"))?;
        ensure!(!chunk.is_empty() && placement.state_len == start + chunk.len(),
            "Kimi draft prefill differs from committed target frontier");
        let rows = (0..chunk.len()).map(|tap_row| crate::shared::speculation::dflash::ContextRow {
            tap_row, slot: placement.slot, position: start + tap_row }).collect::<Vec<_>>();
        draft.update(&rows)?;
        seq.ready = true;
        seq.valid_from = self.origins.borrow()[placement.slot];
        seq.anchor = next;
        Ok(())
    }
    fn draft(&self, sequences: &mut [scheduler::DraftSeq<'_, Placement, DraftState>], timing: &mut scheduler::DraftTiming)
        -> Result<Vec<Vec<u32>>> {
        let draft = self.engine.native_draft().ok_or_else(|| anyhow::anyhow!("Kimi drafter was not admitted"))?;
        let started = std::time::Instant::now();
        let mut output = vec![Vec::new(); sequences.len()];
        let selected = sequences.iter().enumerate().filter(|(_, seq)| seq.depth > 0)
            .map(|(index, seq)| {
                ensure!(seq.seq.ready && seq.depth <= draft.drafts(), "Kimi draft continuation is cold or too wide");
                Ok((index, crate::shared::speculation::dflash::DraftSeq { slot: seq.placement.slot,
                    anchor: seq.seq.anchor.ok_or_else(|| anyhow::anyhow!("Kimi draft anchor absent"))?,
                    position: seq.placement.len, valid_from: seq.seq.valid_from }))
            }).collect::<Result<Vec<_>>>()?;
        for batch in selected.chunks(draft.max_batch_sequences()) {
            let rows = batch.iter().map(|(_, seq)| *seq).collect::<Vec<_>>();
            let proposals = draft.propose(&rows)?;
            ensure!(proposals.len() == batch.len(), "Kimi draft output count differs from batch");
            for ((index, _), proposal) in batch.iter().zip(proposals) {
                output[*index] = proposal.tokens.into_iter().take(sequences[*index].depth).collect();
            }
            timing.steps += 1;
        }
        timing.seconds += started.elapsed().as_secs_f64();
        Ok(output)
    }
    fn accept(&self, verified: &mut [scheduler::Verified<'_, Placement, DraftState, ()>], speculative: bool)
        -> Result<()> {
        if speculative {
            self.engine.accept_provisional(verified)?;
        } else {
            for sequence in verified.iter() {
                ensure!(sequence.placement.state_len == sequence.start + sequence.rows.len(),
                    "Kimi non-speculative state did not commit all input rows");
            }
        }
        if verified.iter().any(|sequence| sequence.neural) {
            let drafter = self.engine.native_draft().ok_or_else(|| anyhow::anyhow!("Kimi drafter was not admitted"))?;
            let mut context = Vec::new();
            for sequence in verified.iter_mut() {
                let Some((keep, after)) = sequence.neural_kept() else { continue; };
                ensure!(keep > 0 && keep <= sequence.rows.len()
                    && sequence.placement.state_len == sequence.start + keep,
                    "Kimi draft accept differs from exact committed target state");
                if !sequence.seq.ready {
                    // Admission already bound cold(P) and reserved this request's growth.
                    // Do not reset again: that would release its upfront live credits.
                    let (origin, end) = drafter.frontier(sequence.placement.slot)?;
                    ensure!(origin == sequence.start && end == sequence.start,
                        "Kimi cold draft continuation differs from admitted frontier");
                    sequence.seq.valid_from = origin;
                }
                context.extend((0..keep).map(|offset| crate::shared::speculation::dflash::ContextRow {
                    tap_row: sequence.first_row + offset, slot: sequence.placement.slot,
                    position: sequence.start + offset }));
                sequence.seq.ready = true;
                sequence.seq.anchor = Some(after);
            }
            drafter.update(&context)?;
        }
        // Shared token selection completed the producing stream before accept.
        // Record one steady census, never a per-step sampler or a compute drain.
        if !self.steady_census.replace(true) {
            let devices = self.engine.devices();
            crate::shared::inventory::RuntimeInventory::measure(self.library(), &devices,
                crate::shared::inventory::InventoryPhase::PostLoad, |_| Ok(()))?
                .record("kimi_k3-serving-steady", &[]);
        }
        Ok(())
    }
    fn cancel(&self, placement: &Placement) -> Result<()> {
        self.engine.cancel_provisional(placement)?;
        if let Some(draft) = self.engine.native_draft() { draft.discard(placement.slot)?; }
        Ok(())
    }
    fn probe_begin(&self, rows: usize) { self.engine.probe_begin(rows); }
    fn probe_end(&self) { self.engine.probe_end(); }
    fn probe_finish<T>(&self, observe: impl FnOnce(&RoundRoutes, &[Option<f64>]) -> T) -> Option<T> {
        self.engine.probe_finish(observe)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::shared::qwen::scheduler::DraftState as _;
    use anyhow::Context;

    struct LiveMember {
        logits: Vec<Vec<f32>>,
        state: String,
        draft: crate::shared::speculation::dflash::MlaReadback,
    }

    fn live_arm(backend: &KimiBackend<'_, '_>, pool: &mut RefPagePool, prompts: &[Vec<u32>],
        packed: bool, cancelled: bool) -> Result<Vec<LiveMember>> {
        use cuteafd_engine::prefix::PrefixFamily;
        let family = backend.engine.prefix_family()?;
        let starts = [0, 3, 7];
        let mut placements = prompts.iter().enumerate().map(|(slot, prompt)| {
            let mut placement = Placement { units: pool.alloc((prompt.len() + 1).div_ceil(64))?,
                slot, len: 0, state_len: 0 };
            family.bind(&mut placement).map_err(|error| anyhow::anyhow!(error.to_string()))?;
            family.admit_draft_pages(&mut placement, prompt.len() + 1)
                .map_err(|error| anyhow::anyhow!(error.to_string()))?;
            Ok(placement)
        }).collect::<Result<Vec<_>>>()?;
        let mut states = (0..prompts.len()).map(|_| DraftState::default()).collect::<Vec<_>>();
        for i in 0..prompts.len() {
            if starts[i] > 0 {
                backend.prefill_device(&mut placements[i], &prompts[i][..starts[i]], 1, pool)?;
                backend.prefill_draft(&placements[i], 0, &prompts[i][..starts[i]], Some(prompts[i][starts[i]]), &mut states[i])?;
            }
        }
        let mut logits = vec![Vec::new(); prompts.len()];
        if packed {
            let mut members = placements.iter_mut().zip(&mut states).enumerate().map(|(i, (placement, draft))|
                scheduler::PrefillSeq { placement, tokens: &prompts[i][starts[i]..], start: starts[i],
                    next: Some(42), draft, neural: true }).collect::<Vec<_>>();
            let outcomes = backend.prefill_packed(&mut members, pool, &mut |i, row| {
                logits[i].push(row.row_host(backend.library(), 0)?);
                Ok(if cancelled && i == 1 { Err(anyhow::anyhow!("test client cancelled")) } else { Ok(()) })
            })?;
            for (i, result) in outcomes.into_iter().enumerate() {
                ensure!(result.is_err() == (cancelled && i == 1), "Kimi packed cancellation escaped member {i}");
            }
        } else {
            for i in 0..prompts.len() {
                let chunk = &prompts[i][starts[i]..];
                let row = backend.prefill_device(&mut placements[i], chunk, 1, pool)?.context("Kimi native head")?;
                logits[i].push(row.row_host(backend.library(), 0)?);
                backend.prefill_draft(&placements[i], starts[i], chunk, Some(42), &mut states[i])?;
            }
        }
        // Identical teacher-forced continuation exposes corrupted recurrent/KV history.
        for i in 0..prompts.len() {
            let start = placements[i].len;
            let row = backend.prefill_device(&mut placements[i], &[42], 1, pool)?.context("Kimi continuation head")?;
            logits[i].push(row.row_host(backend.library(), 0)?);
            backend.prefill_draft(&placements[i], start, &[42], Some(43), &mut states[i])?;
        }
        let draft = backend.engine.native_draft().context("native packing gate requires DFlash2")?;
        draft.drain()?;
        let mut output = Vec::new();
        for (i, placement) in placements.iter_mut().enumerate() {
            ensure!(states[i].ready && states[i].anchor == Some(43)
                && draft.frontier(i)? == (0, placement.len), "Kimi packed draft member frontier {i}");
            output.push(LiveMember { logits: std::mem::take(&mut logits[i]),
                state: family.live_digest(placement)?, draft: draft.readback(i, 0)? });
            family.discard(placement).map_err(|error| anyhow::anyhow!(error.to_string()))?;
            pool.release(&placement.units);
        }
        Ok(output)
    }

    fn live_logits_score(reference: &[Vec<f32>], actual: &[Vec<f32>]) -> Result<(f64, f64)> {
        ensure!(!reference.is_empty() && reference.len() == actual.len(), "Kimi packed score rows");
        let (mut kl, mut matches) = (0.0, 0usize);
        for (reference, actual) in reference.iter().zip(actual) {
            ensure!(!reference.is_empty() && reference.len() == actual.len()
                && reference.iter().chain(actual).all(|v| v.is_finite()), "Kimi packed logits invalid");
            let log_probs = |row: &[f32]| {
                let peak = row.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
                let norm = row.iter().map(|v| (f64::from(*v) - peak).exp()).sum::<f64>().ln() + peak;
                row.iter().map(|v| f64::from(*v) - norm).collect::<Vec<_>>()
            };
            let p = log_probs(reference);
            let q = log_probs(actual);
            kl += p.iter().zip(q).map(|(p, q)| p.exp() * (p - q)).sum::<f64>();
            let top = |row: &[f32]| row.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).unwrap().0;
            matches += usize::from(top(reference) == top(actual));
        }
        Ok((kl / reference.len() as f64, matches as f64 / reference.len() as f64))
    }

    fn live_golden_packed(backend: &KimiBackend<'_, '_>, pool: &mut RefPagePool) -> Result<()> {
        use cuteafd_api::openai::probe::{Probe, ProbeSpec};
        use cuteafd_bench::fidelity_dataset;
        use cuteafd_engine::prefix::PrefixFamily;
        use sha2::{Digest, Sha256};
        let root = std::path::PathBuf::from(std::env::var("CUTEAFD_KIMI_PACK_GATE_GOLDEN")?);
        let bytes = std::fs::read(root.join("manifest.json"))?;
        ensure!(format!("{:x}", Sha256::digest(&bytes)) == fidelity_dataset::KIMI_K3_MANIFEST_SHA256,
            "Kimi packing golden manifest differs from immutable publication");
        let manifest: serde_json::Value = serde_json::from_slice(&bytes)?;
        ensure!(manifest["config"] == fidelity_dataset::KIMI_K3_CONFIG && manifest["vocab"] == 163840,
            "Kimi packing golden identity differs");
        let reference = fidelity_dataset::load(&root, &manifest)?;
        let windows = ["d00", "d01", "e00"].into_iter().map(|id| {
            let mut window = reference.windows.iter().find(|w| w.id == id)
                .with_context(|| format!("Kimi packing golden lacks {id}"))?.clone();
            window.positions.retain(|p| window.roles.get(p.pos).is_some_and(|role| role == "gen"));
            ensure!(window.positions.len() == 512 && window.media.is_empty()
                && window.positions.windows(2).all(|pair| pair[1].pos == pair[0].pos + 1),
                "Kimi packing golden needs 512 contiguous genuine generated rows per member");
            Ok(window)
        }).collect::<Result<Vec<_>>>()?;
        let family = backend.engine.prefix_family()?;
        let mut arms = Vec::new();
        for packed in [false, true] {
            let mut placements = windows.iter().enumerate().map(|(slot, window)| {
                let end = window.positions.last().context("Kimi packing golden empty")?.pos;
                let mut placement = Placement { units: pool.alloc(end.div_ceil(64))?, slot, len: 0, state_len: 0 };
                family.bind(&mut placement).map_err(|e| anyhow::anyhow!(e.to_string()))?;
                family.admit_draft_pages(&mut placement, end).map_err(|e| anyhow::anyhow!(e.to_string()))?;
                Ok(placement)
            }).collect::<Result<Vec<_>>>()?;
            let mut states = (0..windows.len()).map(|_| DraftState::default()).collect::<Vec<_>>();
            let probes = windows.iter().map(|window| {
                let probe = Probe::new(ProbeSpec { score_positions: Some(window.positions.iter().map(|p| p.pos).collect()),
                    want: window.want(), ..Default::default() });
                probe.admitted("kimi_k3", &window.tokens, 0);
                probe
            }).collect::<Vec<_>>();
            for (i, first_rows) in [29, 33, 35].into_iter().enumerate() {
                let end = windows[i].positions[0].pos.checked_sub(first_rows).context("Kimi golden prefix too short")?;
                while placements[i].len < end {
                    let start = placements[i].len;
                    let next = (start + backend.engine.packed_limits()?.chunk_rows).min(end);
                    let chunk = &windows[i].tokens[start..next];
                    backend.prefill_device(&mut placements[i], chunk, 1, pool)?;
                    backend.prefill_draft(&placements[i], start, chunk, Some(windows[i].tokens[next]), &mut states[i])?;
                }
            }
            // Score all 512 genuine rows per member. Identical 32-row chunks
            // retain short-member routing while giving the pooled bar resolution.
            for scored_end in std::iter::once(1).chain((33..512).step_by(32)).chain(std::iter::once(512)) {
                let mut sequences = placements.iter_mut().zip(&mut states).enumerate().map(|(i, (placement, draft))| {
                    let end = windows[i].positions[scored_end - 1].pos;
                    let start = placement.len;
                    scheduler::PrefillSeq { placement, tokens: &windows[i].tokens[start..end], start,
                        next: Some(windows[i].tokens[end]), draft, neural: true }
                }).collect::<Vec<_>>();
                ensure!(sequences.iter().map(|s| s.tokens.len()).sum::<usize>() > 64,
                    "Kimi golden pack must cross the total-row arithmetic boundary");
                if packed {
                    let segments = crate::shared::prefill_packing::plan(&sequences.iter()
                        .map(|s| (s.start, s.tokens.len())).collect::<Vec<_>>(), &backend.engine.packed_limits()?)?;
                    let starts = sequences.iter().map(|s| s.start).collect::<Vec<_>>();
                    let mut target = sequences.iter_mut().map(|s| (&mut *s.placement, s.tokens)).collect::<Vec<_>>();
                    let outcomes = backend.engine.prefill_packed_all_rows(&mut target, pool, &mut |i, offset, rows| {
                        for row in 0..rows.rows {
                            let position = starts[i] + offset + row + 1;
                            if windows[i].positions.binary_search_by_key(&position, |p| p.pos).is_ok() {
                                probes[i].row(position, &rows.row_host(backend.library(), row)?);
                            }
                        }
                        Ok(Ok(()))
                    })?;
                    for outcome in outcomes { outcome?; }
                    for (i, sequence) in sequences.iter_mut().enumerate() {
                        let segment = segments[i];
                        let rows = (0..segment.tap_rows).map(|offset| crate::shared::speculation::dflash::ContextRow {
                            tap_row: segment.tap_offset + offset, slot: sequence.placement.slot,
                            position: segment.start + offset }).collect::<Vec<_>>();
                        backend.engine.native_draft().context("Kimi golden drafter")?.update(&rows)?;
                        sequence.draft.ready = true;
                        sequence.draft.valid_from = backend.origins.borrow()[sequence.placement.slot];
                        sequence.draft.anchor = sequence.next;
                    }
                } else {
                    for (i, sequence) in sequences.iter_mut().enumerate() {
                        let count = sequence.tokens.len();
                        let rows = backend.prefill_device(sequence.placement, sequence.tokens, count, pool)?
                            .context("Kimi original-golden singleton head")?;
                        let first = sequence.start + sequence.tokens.len() - count + 1;
                        for row in 0..count {
                            let position = first + row;
                            if windows[i].positions.binary_search_by_key(&position, |p| p.pos).is_ok() {
                                probes[i].row(position, &rows.row_host(backend.library(), row)?);
                            }
                        }
                        backend.prefill_draft(sequence.placement, sequence.start, sequence.tokens,
                            sequence.next, sequence.draft)?;
                    }
                }
            }
            let mut scores = Vec::new();
            for (i, window) in windows.iter().enumerate() {
                let score = window.score(&probes[i].record().rows);
                println!("Kimi original golden packed={packed} member {i} {}: rows={} missing={} nonfinite={} KL={:.8} top1={:.4}",
                    window.id, score.positions, score.missing, score.non_finite, score.kl, score.top1);
                ensure!(score.positions == window.positions.len() && score.missing == 0 && score.non_finite == 0,
                    "Kimi original-golden member {i} coverage differs");
                scores.push(score);
                family.drain().map_err(|e| anyhow::anyhow!(e.to_string()))?;
                family.discard(&mut placements[i]).map_err(|e| anyhow::anyhow!(e.to_string()))?;
                pool.release(&placements[i].units);
            }
            arms.push(scores);
        }
        for (i, (alone, packed)) in arms[0].iter().zip(&arms[1]).enumerate() {
            let delta_kl = packed.kl - alone.kl;
            let delta_top1 = packed.top1 - alone.top1;
            println!("Kimi member {i} packing drift (informational): delta_KL={delta_kl:.8} delta_top1={delta_top1:.4}");
        }
        let pooled = arms.iter().map(|scores| cuteafd_bench::reference::Fidelity::from_records(scores.iter()
            .flat_map(|score| score.records.iter().cloned()).collect())).collect::<Vec<_>>();
        let delta_kl = pooled[1].kl - pooled[0].kl;
        let delta_top1 = pooled[1].top1 - pooled[0].top1;
        println!("Kimi packed pooled original golden: rows={} KL={:.8} top1={:.4} delta_KL={delta_kl:.8} delta_top1={delta_top1:.4}",
            pooled[1].positions, pooled[1].kl, pooled[1].top1);
        ensure!(delta_kl <= 0.005 && delta_top1 >= -0.01,
            "Kimi packed pooled fidelity regresses against matched singleton controls");
        Ok(())
    }

    #[test]
    #[ignore = "requires one RTX GPU and sealed CUTEAFD_NATIVE_LIB/CUTEAFD_PROGRAM_MANIFEST"]
    fn live_router_row_total_discriminator() -> Result<()> {
        use crate::shared::memory::device::{Allocation, Device, Stream};
        // SAFETY: explicit sealed library; allocations and stream drop before it.
        let library = unsafe { cuteafd_ffi::NativeLibrary::load(std::env::var("CUTEAFD_NATIVE_LIB")?)? };
        let device = Device { library: &library, id: 0 };
        library.cuda_set_device(0)?;
        let manifest = std::path::PathBuf::from(std::env::var("CUTEAFD_PROGRAM_MANIFEST")?);
        let programs = library.programs()?.with_manifest(&manifest)?;
        let program = programs.program("kimik3_router_scores", &["x", "w", "logits"])?;
        let input = Allocation::new(device, 96 * 7168 * 2)?;
        let weights = Allocation::new(device, 896 * 7168 * 2)?;
        let output = Allocation::new(device, 96 * 896 * 4)?;
        let stream = Stream::new(device)?;
        let mut seed = 0x517cc1b727220a95u64;
        let mut bf16 = |elements: usize| -> Vec<u8> {
            (0..elements).flat_map(|_| {
                seed ^= seed << 13; seed ^= seed >> 7; seed ^= seed << 17;
                let value = ((seed >> 40) as f32 / (1u32 << 24) as f32 - 0.5) * 0.125;
                ((value.to_bits() >> 16) as u16).to_ne_bytes()
            }).collect()
        };
        library.copy_h2d(input.buffer, &bf16(96 * 7168))?;
        library.copy_h2d(weights.buffer, &bf16(896 * 7168))?;
        let mut arms = Vec::new();
        for rows in [19, 64, 96] {
            // SAFETY: sealed program pointer/scalar contract, retained disjoint
            // dense planes sized for 96 rows; drain precedes readback and reuse.
            unsafe { program.launch(&[input.buffer.ptr, weights.buffer.ptr, output.buffer.ptr],
                &[cuteafd_ffi::programs::Scalar::I32(rows)], stream.raw)?; }
            stream.drain()?;
            let mut bytes = vec![0; 96 * 896 * 4];
            library.copy_d2h(&mut bytes, output.buffer)?;
            arms.push(bytes[..19 * 896 * 4].to_vec());
        }
        for (i, rows) in [64, 96].into_iter().enumerate() {
            let mut changed = 0;
            let mut max_diff = 0.0f32;
            for (a, b) in arms[0].chunks_exact(4).zip(arms[i + 1].chunks_exact(4)) {
                let a = f32::from_ne_bytes(a.try_into()?);
                let b = f32::from_ne_bytes(b.try_into()?);
                ensure!(a.is_finite() && b.is_finite(), "Kimi router discriminator nonfinite logits");
                changed += usize::from(a.to_bits() != b.to_bits());
                max_diff = max_diff.max((a - b).abs());
            }
            println!("Kimi router-only same first19 rows: total19 vs total{rows} changed_logits={changed}/{} max_abs_diff={max_diff:.10} byte_identical={}",
                19 * 896, arms[0] == arms[i + 1]);
        }
        Ok(())
    }

    #[test]
    #[ignore = "requires two RTX GPUs, six live Spark workers, DFlash2, CUTEAFD_KIMI_PACK_GATE_ARGS JSON serve-kimi argv, CUTEAFD_KIMI_PACK_GATE_TOKENS JSON token IDs and CUTEAFD_KIMI_PACK_GATE_GOLDEN immutable reference directory"]
    fn live_packed_prefill_matches_singletons_with_private_dflash() -> Result<()> {
        use anyhow::Context;
        use clap::Parser;
        use cuteafd_engine::prefix::PrefixFamily;
        use std::sync::Arc;
        let argv: Vec<String> = serde_json::from_str(&std::env::var("CUTEAFD_KIMI_PACK_GATE_ARGS")?)?;
        let crate::cli::Commands::ServeKimi(args) = crate::cli::Cli::try_parse_from(argv)?.command else {
            anyhow::bail!("packing gate needs serve-kimi arguments");
        };
        ensure!(args.peers.len() == 6 && args.max_sequences >= 3, "Kimi native packing gate needs six peers and three slots");
        let source: Vec<u32> = serde_json::from_str(&std::env::var("CUTEAFD_KIMI_PACK_GATE_TOKENS")?)?;
        ensure!(!source.is_empty(), "Kimi packing gate token panel empty");
        cuteafd_core::set_expert_geometry(cuteafd_core::ExpertGeometry::KIMI_K3)
            .map_err(|_| anyhow::anyhow!("native packing gate requires isolated process"))?;
        super::super::startup::preflight(&args.manifest, &args.native_lib)?;
        // SAFETY: explicit installed native library; every borrower drops before it.
        let library = Arc::new(unsafe { cuteafd_ffi::NativeLibrary::load(&args.native_lib)? });
        let programs = library.programs()?.with_manifest(&args.manifest)?;
        let checkpoint = cuteafd_loader::plan::Checkpoint::coordinator(&args.snapshot, false, false)?;
        let cfg = cuteafd_loader::families::kimi_k3::config::KimiK3Config::from_hf(&checkpoint.config)?;
        let prefill = args.prefill()?;
        let loaded = super::super::startup::load(&library, &programs, &checkpoint, &cfg, &cfg,
            super::super::startup::Options { cache_format: args.kv_cache, devices: [args.device, args.split_device],
                native_lib: &args.native_lib, peers: args.peers.clone(), rtx_expert_layers: args.rtx_expert_layers,
                spark_budget_gib: args.spark_budget_gib, rows: usize::try_from(prefill.span_rows)?,
                waves: usize::try_from(prefill.waves)?, prefill_rows: Some(usize::try_from(prefill.unit_rows)?),
                slots: usize::from(args.max_sequences), marks: 1,
                max_context: args.max_context.unwrap_or(cuteafd_loader::families::kimi_k3::COMPILED_CONTEXT),
                pool_tokens: args.pool_tokens, graphs: true, ffn_mode: cuteafd_ffi::KimiFfnMode::default(),
                hosted_layers: None, draft: Some(super::super::draft::Options {
                    snapshot: args.draft.clone().context("native packing gate requires --draft")?,
                    fp8: args.draft_fp8, cache_format: args.kv_cache }), verify_rows: 0 })?;
        let engine = &loaded.engine;
        let backend = KimiBackend::new(engine)?;
        library.cuda_set_device(args.split_device)?;
        let mut pool = RefPagePool::new(engine.prefix_family()?.layout().pages, 64);
        if std::env::var("CUTEAFD_KIMI_PACK_GATE_ROUTE64").as_deref() == Ok("1") {
            // live_arm prefills private prefixes of 0/3/7 rows, leaving 64
            // total packed rows; both arms therefore stay on the skinny route.
            let prompts = [19, 24, 31].iter().enumerate().map(|(i, &len)|
                (0..len).map(|j| source[(i * 97 + j) % source.len()]).collect::<Vec<_>>()).collect::<Vec<_>>();
            engine.configure_packing(crate::shared::prefill_packing::PackingMode::Unrestricted)?;
            let alone = live_arm(&backend, &mut pool, &prompts, false, false)?;
            let packed = live_arm(&backend, &mut pool, &prompts, true, false)?;
            for i in 0..prompts.len() {
                println!("Kimi route64 discriminator member {i}: packed_rows=64 logits_bytes_equal={} target_bytes_equal={} draft_bytes_equal={}",
                    alone[i].logits == packed[i].logits, alone[i].state == packed[i].state,
                    alone[i].draft == packed[i].draft);
            }
            for i in 0..prompts.len() {
                ensure!(alone[i].logits == packed[i].logits && alone[i].state == packed[i].state
                    && alone[i].draft == packed[i].draft,
                    "Kimi route64 singleton/packed bytes differ for member {i}");
            }
            return Ok(());
        }
        let prompts = [29, 68, 134].iter().enumerate().map(|(i, &len)|
            (0..len).map(|j| source[(i * 97 + j) % source.len()]).collect::<Vec<_>>()).collect::<Vec<_>>();
        engine.configure_packing(crate::shared::prefill_packing::PackingMode::Exact)?;
        let alone = live_arm(&backend, &mut pool, &prompts, false, false)?;
        let exact = live_arm(&backend, &mut pool, &prompts, false, false)?;
        for i in 0..prompts.len() {
            ensure!(alone[i].logits == exact[i].logits && alone[i].state == exact[i].state
                && alone[i].draft == exact[i].draft, "Kimi exact singleton golden differs for member {i}");
        }
        engine.configure_packing(crate::shared::prefill_packing::PackingMode::Unrestricted)?;
        let packed = live_arm(&backend, &mut pool, &prompts, true, false)?;
        let cancelled = live_arm(&backend, &mut pool, &prompts, true, true)?;
        let mut changed = prompts.clone();
        for (i, token) in changed[1].iter_mut().enumerate() { *token = source[(i + 11) % source.len()]; }
        let partner = live_arm(&backend, &mut pool, &changed, true, false)?;
        for i in 0..prompts.len() {
            let (kl, top1) = live_logits_score(&alone[i].logits, &packed[i].logits)?;
            println!("Kimi packed-vs-singleton member {i} (informational): KL={kl:.8} top1={top1:.4} target_bytes_equal={} draft_bytes_equal={}",
                alone[i].state == packed[i].state, alone[i].draft == packed[i].draft);
            ensure!(packed[i].logits == cancelled[i].logits && packed[i].state == cancelled[i].state
                && packed[i].draft == cancelled[i].draft, "Kimi cancellation changed member {i}");
            if i != 1 {
                let (kl, top1) = live_logits_score(&packed[i].logits, &partner[i].logits)?;
                println!("Kimi changed partner member {i} (informational): KL={kl:.8} top1={top1:.4} target_bytes_equal={} draft_bytes_equal={}",
                    packed[i].state == partner[i].state, packed[i].draft == partner[i].draft);
            }
        }
        live_golden_packed(&backend, &mut pool)
    }

    #[test]
    fn kimi_verify_cycle_observes_simple_wall_time_and_shared_routes() -> Result<()> {
        use crate::shared::draft::{resource::{self, LayerHome}, clock::RoundTimes, evidence::DraftSource};
        use cuteafd_core::{DraftMode, DraftPolicy, PolicySeed};
        // Same scheduler resource handoff as production; simple deliberately has no routes.
        let homes = [LayerHome::Dense, LayerHome::Local { slice_bytes: 1024.0 },
            LayerHome::Remote { slice_bytes: 1024.0 }];
        let geometry = resource::geometry(&homes, 896, 16, 64, 7, 0)?;
        for mode in [DraftMode::Simple, DraftMode::Shared] {
            let mut policy = DraftPolicy::new(geometry.clone(), PolicySeed::weak(7), false)?;
            policy.set_draft_mode(mode);
            let mut routes = RoundRoutes::new(3, 16);
            let layer_us = if policy.needs_routes() {
                routes.begin(4);
                routes.dense(0)?;
                let ids = (0..64).map(|id| id as u32).collect::<Vec<_>>();
                routes.push(1, &ids, u32::MAX)?;
                let staged = ids.iter().flat_map(|id| id.to_le_bytes()).collect::<Vec<_>>();
                routes.push_bytes(2, &staged, u32::MAX)?;
                routes.complete()?;
                assert_eq!(routes.flat().len(), 3 * 4 * 16);
                vec![None, Some(20.0), Some(80.0)]
            } else {
                assert!(routes.flat().is_empty());
                Vec::new()
            };
            let prior = [0.9; 3];
            let requests = [resource::Verified { id: 1, rows: 4, accepted: 3,
                source: DraftSource::Neural, prior: Some(&prior), features: None, censor: None }];
            resource::observe_drafted(&mut policy, &requests, &routes, &layer_us,
                RoundTimes { total_us: 200, draft_us: Some(40) }, 7, None, Some(1))?;
            assert_eq!(policy.stats().rounds, 1);
            if mode == DraftMode::Simple { assert!(policy.simple_round().is_some()); }
        }
        Ok(())
    }

    #[test]
    fn whole_prompt_hit_waits_for_first_target_row_before_drafting() {
        let mut seq = DraftState::default();
        assert_eq!(seq.pending_rows(), 0);
        seq.close(42);
        assert_eq!(seq.pending_rows(), 0, "a cached anchor cannot expose stale drafter context");
        seq.ready = true;
        seq.valid_from = 8192;
        assert_eq!(seq.pending_rows(), 1);
        assert_eq!(seq.anchor, Some(42));
        assert_eq!(seq.valid_from, 8192);
        seq.close(43);
        assert_eq!(seq.anchor, Some(43));
        assert_eq!(seq.valid_from, 8192, "closing a round cannot move the cold storage origin");
    }
}
