//! A Muse prefill call on the shared prefill pipeline (`run_prefill_units`):
//! one sequence's rows cut into units that run interleaved per layer, each
//! with its own in-flight slot. A unit carries its state across hooks only in
//! its slot: the lane's span planes (`LaneViews::slot`: residual stream,
//! routes, the expert wire) and its step-metadata bank. Every other plane is
//! written and read within one hook.
//!
//! Hooks: `begin_unit` uploads the unit's metadata into its slot bank and
//! embeds its tokens; `attention` + `send_experts` are the layer's front half
//! (`RankDevice::layer_front`, the Spark send on rank 0); `land_experts` is the
//! back half; `end_unit` scores the requested suffix on the head rank before
//! its unit slot is reused; `commit_unit` has nothing to publish (K|V records and ring
//! rows were written by the layers; the caller commits the frontier).
//! Onboarded layers (RTX experts, `FfnRoute::Local`) run whole in `local_ffn`.
use super::{device::{PeerExchange, StepPlanes}, engine::RankResources, experts::{SparkExperts, Wave, WaveSegment, WaveStep},
    lane::LaneViews, step::{StepMetadata, StepSequence}};
use crate::shared::prefill_pipeline::{FfnRoute, PrefillHooks, Unit};
use anyhow::{ensure, Context, Result};
use cuteafd_ffi::CuteafdDeviceBuffer as Buffer;
use cuteafd_loader::families::muse_spark::MuseSparkConfig;
use std::ffi::c_void;

/// One sequence's prefill call: its tokens from `start`, ring slot and pages.
pub(crate) struct PrefillCall<'c> {
    pub request: u64,
    pub slot: usize,
    pub start: usize,
    pub tokens: &'c [u32],
    pub pages: &'c [u32],
    pub ring_rows: usize,
    pub slots: usize,
    pub logit_rows: usize,
}

/// Head microbatches within a unit, paired with their output-suffix row offset.
fn score_chunks(total: usize, requested: usize, unit: std::ops::Range<usize>)
    -> Result<Vec<(std::ops::Range<usize>, usize)>> {
    ensure!(requested <= total && unit.start <= unit.end && unit.end <= total, "Muse prefill score range");
    let first = total - requested;
    let from = unit.start.max(first);
    Ok((from..unit.end).step_by(cuteafd_loader::families::muse_spark::lane::HEAD_ROWS).map(|start| {
        let end = (start + cuteafd_loader::families::muse_spark::lane::HEAD_ROWS).min(unit.end);
        (start - unit.start..end - unit.start, start - first)
    }).collect())
}

/// A unit's per-rank state between hooks.
struct UnitState { lanes: Vec<LaneViews>, planes: Vec<StepPlanes>, wave: Option<Wave> }

pub(crate) struct MusePrefill<'h, 'r, 'a> {
    pub cfg: &'h MuseSparkConfig,
    pub ranks: &'r [RankResources<'a>],
    pub streams: Vec<*mut c_void>,
    pub table: &'h crate::shared::token_io::TokenEmbedding<'a>,
    pub experts: &'h mut dyn SparkExperts,
    pub peers: Vec<&'h mut dyn PeerExchange>,
    pub call: PrefillCall<'h>,
    /// Each rank's onboarded experts (empty: every layer on the Sparks).
    pub local: &'h mut [super::onboard::OnboardedExperts<'a>],
    /// The drafter: units in the call's final `TAP_ROWS` rows tap their rows there.
    pub draft: Option<&'h super::draft::MuseDraft<'a>>,
    units: Vec<Option<UnitState>>,
    /// Contiguous requested logits, retained by the head rank (not a unit slot).
    pub logits: Option<Buffer>,
}

impl<'h, 'r, 'a> MusePrefill<'h, 'r, 'a> {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(cfg: &'h MuseSparkConfig, ranks: &'r [RankResources<'a>], streams: Vec<*mut c_void>,
        table: &'h crate::shared::token_io::TokenEmbedding<'a>, experts: &'h mut dyn SparkExperts,
        peers: Vec<&'h mut dyn PeerExchange>, call: PrefillCall<'h>, local: &'h mut [super::onboard::OnboardedExperts<'a>],
        draft: Option<&'h super::draft::MuseDraft<'a>>, units: usize) -> Result<Self> {
        ensure!(ranks.len() == streams.len() && (ranks.len() == 1 || peers.len() == 2)
            && (local.is_empty() || local.len() == ranks.len()), "Muse prefill geometry");
        Ok(Self { cfg, ranks, streams, table, experts, peers, call, local, draft, units: (0..units).map(|_| None).collect(),
            logits: None })
    }

    fn state(&mut self, unit: &Unit) -> Result<&mut UnitState> {
        self.units.get_mut(unit.index).and_then(Option::as_mut).context("Muse prefill unit not begun")
    }

    /// Runs `body` on every rank in rank order with the unit's slot views, its
    /// metadata planes, the rank's peer end and (rank 0) the Spark path.
    fn each_rank(&mut self, unit: &Unit, mut body: impl FnMut(usize, &super::device::RankDevice<'a, '_>,
        &StepPlanes, Option<&mut (dyn PeerExchange + '_)>, Option<&mut (dyn SparkExperts + '_)>, &mut Option<Wave>)
        -> Result<()>) -> Result<()> {
        let (cfg, ranks) = (self.cfg, self.ranks);
        let state = self.units.get_mut(unit.index).and_then(Option::as_mut).context("Muse prefill unit not begun")?;
        let experts: &mut (dyn SparkExperts + '_) = &mut *self.experts;
        let mut experts = Some(experts);
        for (index, rank) in ranks.iter().enumerate() {
            let device = rank.device_with(cfg, self.streams[index], &state.lanes[index]);
            let peer: Option<&mut (dyn PeerExchange + '_)> = match self.peers.get_mut(index) {
                Some(end) => Some(&mut **end),
                None => None,
            };
            let path = if index == 0 { experts.take() } else { None };
            rank.device.run(|| body(index, &device, &state.planes[index], peer, path, &mut state.wave))?;
        }
        Ok(())
    }
}

impl MusePrefill<'_, '_, '_> {
    /// The call's tapped window: its last `TAP_ROWS` rows.
    pub(crate) fn tap_window(&self) -> std::ops::Range<usize> {
        let rows = self.call.tokens.len();
        rows.saturating_sub(crate::shared::speculation::dflash::TAP_ROWS)..rows
    }

    /// After `layer` of `unit` on every rank: its rows inside the tap window,
    /// from the head rank's `hidden` (the unit's slot view), at their window offset.
    fn tap(&self, layer: usize, unit: &Unit) -> Result<()> {
        let Some(draft) = self.draft else { return Ok(()) };
        let window = self.tap_window();
        let (from, to) = (unit.rows.start.max(window.start), unit.rows.end.min(window.end));
        if from >= to { return Ok(()); }
        let state = self.units[unit.index].as_ref().context("Muse prefill unit not begun")?;
        let hidden = state.lanes[self.ranks.len() - 1].plane("hidden")?;
        draft.tap(layer, hidden, from - unit.rows.start, to - from, from - window.start)
    }
}

impl PrefillHooks for MusePrefill<'_, '_, '_> {
    type Ticket = ();

    fn route(&self, layer: usize) -> FfnRoute {
        if self.local.first().is_some_and(|local| local.holds(layer)) { FfnRoute::Local } else { FfnRoute::Spark }
    }

    fn begin_unit(&mut self, unit: &Unit) -> Result<()> {
        ensure!(unit.sequence == 0 && unit.segments.is_empty() && unit.layers.start == 0,
            "Muse prefill units run every layer of one sequence");
        let rows = unit.rows.len();
        let tokens = &self.call.tokens[unit.rows.clone()];
        let sequence = StepSequence { slot: self.call.slot, start: self.call.start + unit.rows.start, rows,
            pages: self.call.pages };
        let meta = StepMetadata::build(&[sequence], self.call.ring_rows, self.call.slots, self.ranks[0].lane.views.rows, true)?;
        let wave = WaveStep { segments: vec![WaveSegment { request: self.call.request, rows }],
            kind: cuteafd_transport::ExpertV2SourceKind::Prefill };
        let mut lanes = Vec::new();
        let mut planes = Vec::new();
        for (index, rank) in self.ranks.iter().enumerate() {
            let lane = rank.lane.views.slot(unit.slot)?;
            let stream = self.streams[index];
            planes.push(rank.device.run(|| rank.step.upload(rank.device.library, unit.slot, &meta, true, wave.clone(), stream))?);
            super::forward::embed(self.cfg, rank, &lane, self.table, tokens, stream)?;
            lanes.push(lane);
        }
        self.units[unit.index] = Some(UnitState { lanes, planes, wave: None });
        Ok(())
    }

    fn attention(&mut self, layer: usize, unit: &Unit) -> Result<()> {
        // The front half runs whole in `send_experts` (attention and the frontend share
        // the hook's planes); a layer's attention hook only checks the unit's state.
        let _ = (layer, self.state(unit)?);
        Ok(())
    }

    /// An onboarded layer whole (`RankDevice::layer_local`) on every rank in
    /// rank order: its attention hook was a no-op like a Spark layer's.
    fn local_ffn(&mut self, layer: usize, unit: &Unit) -> Result<()> {
        let (cfg, ranks) = (self.cfg, self.ranks);
        let state = self.units.get_mut(unit.index).and_then(Option::as_mut).context("Muse prefill unit not begun")?;
        for (index, rank) in ranks.iter().enumerate() {
            let device = rank.device_with(cfg, self.streams[index], &state.lanes[index]);
            let peer: Option<&mut (dyn PeerExchange + '_)> = match self.peers.get_mut(index) {
                Some(end) => Some(&mut **end),
                None => None,
            };
            let local = self.local.get_mut(index).context("Muse onboarded layer without the rank's experts")?;
            rank.device.run(|| device.layer_local(layer, &state.planes[index], local, peer))?;
        }
        self.tap(layer, unit)
    }

    fn send_experts(&mut self, layer: usize, unit: &Unit) -> Result<()> {
        self.each_rank(unit, |_, device, planes, peer, experts, wave| {
            let sent = device.layer_front(layer, planes, experts, peer)?;
            if sent.is_some() { *wave = sent; }
            Ok(())
        })
    }

    fn land_experts(&mut self, layer: usize, unit: &Unit, _: ()) -> Result<()> {
        let rows = unit.rows.len();
        self.each_rank(unit, |index, device, _, peer, experts, wave| {
            let landed = if index == 0 { wave.take() } else { None };
            device.layer_back(layer, rows, landed, experts, peer)
        })?;
        self.tap(layer, unit)
    }

    fn end_unit(&mut self, unit: &Unit) -> Result<()> {
        let chunks = score_chunks(self.call.tokens.len(), self.call.logit_rows, unit.rows.clone())?;
        if chunks.is_empty() { return Ok(()); }
        let last = self.ranks.len() - 1;
        let state = self.units[unit.index].as_ref().context("Muse prefill unit not begun")?;
        let (rank, lane, stream) = (&self.ranks[last], &state.lanes[last], self.streams[last]);
        for (rows, offset) in chunks {
            let logits = rank.device.run(|| super::forward::head(self.cfg, rank, lane, rows, stream))?;
            if self.call.logit_rows == 1 {
                self.logits = Some(logits);
            } else {
                let output = rank.prefill_logits.as_ref().context("Muse prefill logits not admitted")?.buffer;
                let at = offset * self.cfg.vocab_size * 4;
                ensure!(at <= output.bytes && logits.bytes <= output.bytes - at, "Muse prefill logits output range");
                let target = Buffer { ptr: output.ptr.cast::<u8>().wrapping_add(at).cast(), bytes: logits.bytes, ..output };
                rank.device.run(|| {
                    // SAFETY: both allocations outlive the call; the head, copy and next scratch reuse
                    // are ordered on the head stream, before this unit's lane slot can be reused.
                    unsafe { rank.device.library.copy_d2d_async(target, logits, logits.bytes, stream) }
                })?;
                self.logits = Some(Buffer { bytes: self.call.logit_rows * self.cfg.vocab_size * 4, ..output });
            }
        }
        Ok(())
    }

    fn commit_unit(&mut self, unit: &Unit) -> Result<()> {
        self.units[unit.index] = None;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use crate::shared::prefill_pipeline::{prefill_schedule, sequence_units, FfnRoute, Step};

    #[test]
    fn scored_suffix_tiles_units_and_head_microbatches() -> anyhow::Result<()> {
        use cuteafd_loader::families::muse_spark::lane::HEAD_ROWS;
        for total in [1usize, 63, 64, 65, 700, 2752, 8256, 8257] {
            for unit in [64usize, 1024, 2752, 4096] {
                for requested in [0, 1, total / 2, total] {
                    let mut covered = vec![0u8; requested];
                    let mut starts: Vec<_> = (0..total).step_by(unit).collect();
                    starts.reverse(); // Completion order cannot affect the output row offsets.
                    for start in starts {
                        let rows = start..(start + unit).min(total);
                        for (chunk, offset) in super::score_chunks(total, requested, rows.clone())? {
                            assert!(!chunk.is_empty() && chunk.len() <= HEAD_ROWS && chunk.end <= rows.len());
                            assert_eq!(start + chunk.start, total - requested + offset);
                            for row in &mut covered[offset..offset + chunk.len()] { *row += 1; }
                        }
                    }
                    assert!(covered.iter().all(|&n| n == 1), "total {total}, unit {unit}, requested {requested}");
                }
            }
        }
        assert!(super::score_chunks(64, 65, 0..64).is_err());
        assert!(super::score_chunks(64, 64, 0..65).is_err());
        let source = include_str!("prefill.rs").split("#[cfg(test)]").next().unwrap();
        let end = source.split("fn end_unit(").nth(1).unwrap().split("fn commit_unit(").next().unwrap();
        assert!(end.find("super::forward::head(").unwrap() < end.find("copy_d2d_async(").unwrap());
        assert!(end.contains("rank.prefill_logits.as_ref()"));
        Ok(())
    }

    /// The hooks run a layer's whole front half in `send_experts`: correct only
    /// if the shared schedule always follows a unit's attention step with that
    /// unit's send (or local) step of the same layer, no other unit between
    /// (the exchange model's per-hook order assumes it too).
    #[test]
    fn attention_is_always_followed_by_its_own_ffn_step() -> anyhow::Result<()> {
        for waves in 1..=3 {
            for units in 1..=6usize {
                for onboarded in [vec![], vec![0], vec![1, 2]] {
                    let routes: Vec<FfnRoute> = (0..4).map(|l| if onboarded.contains(&l) { FfnRoute::Local }
                        else { FfnRoute::Spark }).collect();
                    let call = sequence_units(units * 1024, 1024, waves, 4)?;
                    let steps = prefill_schedule(&routes, &call, waves)?;
                    for (i, step) in steps.iter().enumerate() {
                        if let Step::Attention { layer, unit } = *step {
                            assert!(matches!(steps[i + 1], Step::Send { layer: l, unit: u } | Step::Local { layer: l, unit: u }
                                if l == layer && u == unit), "waves {waves} units {units}: {:?} then {:?}", step, steps[i + 1]);
                        }
                    }
                }
            }
        }
        Ok(())
    }

    /// Units carry state only in their slot: views from `LaneViews::slot`, the
    /// metadata bank of the unit's slot.
    #[test]
    fn units_address_their_slot() {
        let source = include_str!("prefill.rs").split("#[cfg(test)]").next().unwrap();
        let begin = source.split("fn begin_unit(").nth(1).unwrap().split("fn attention(").next().unwrap();
        assert!(begin.contains("rank.lane.views.slot(unit.slot)"));
        assert!(begin.contains("rank.step.upload(rank.device.library, unit.slot, &meta, true"));
        assert!(begin.contains("super::forward::embed(self.cfg, rank, &lane"));
        assert!(source.contains("rank.device_with(cfg, self.streams[index], &state.lanes[index])"));
    }

    /// The drafter's tap rows are the call's last `TAP_ROWS` rows: the units
    /// overlapping that window tap their overlap at its window offset, so the
    /// window is tiled exactly once whatever the unit cut; every layer's back
    /// half (Spark land or onboarded) taps after it runs.
    #[test]
    fn units_tile_the_draft_tap_window_once() -> anyhow::Result<()> {
        use crate::shared::speculation::dflash::TAP_ROWS;
        for rows in [1usize, 700, 2048, 2049, 5000, 8256] {
            for unit in [64usize, 1024, 2752, 4096] {
                let window = rows.saturating_sub(TAP_ROWS)..rows;
                let mut covered = vec![0u8; window.len()];
                for start in (0..rows).step_by(unit) {
                    let unit_rows = start..(start + unit).min(rows);
                    let (from, to) = (unit_rows.start.max(window.start), unit_rows.end.min(window.end));
                    if from >= to { continue; }
                    // (first row within the unit, rows, offset within the window), as `tap` computes it.
                    let (first, n, at) = (from - unit_rows.start, to - from, from - window.start);
                    assert!(first + n <= unit_rows.len() && at + n <= TAP_ROWS);
                    for row in &mut covered[at..at + n] { *row += 1; }
                }
                assert!(covered.iter().all(|&c| c == 1), "rows {rows} unit {unit}");
            }
        }
        let source = include_str!("prefill.rs").split("#[cfg(test)]").next().unwrap();
        for hook in ["fn local_ffn(", "fn land_experts("] {
            let body = source.split(hook).nth(1).unwrap().split("\n    fn ").next().unwrap();
            assert!(body.trim_end().ends_with("self.tap(layer, unit)\n    }") || body.contains("self.tap(layer, unit)"), "{hook}");
        }
        Ok(())
    }
}
