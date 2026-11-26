//! Inkling 2 Mini 48-visit TP1 unit executor on the shared `PrefillHooks` pipeline.
use super::{
    experts::Dispatcher,
    graphs::{self, DecodeGraphs, Mode},
    layers::{self, MiniKernels},
    owners::{PackedMember, PackedVisitUnit, UnitBuffers, UnitOwner},
    weights::MiniTargetWeights,
};
use crate::shared::prefill_pipeline::{run_prefill_units, FfnRoute, PrefillHooks, Unit, UnitSegment};
use anyhow::{bail, ensure, Context, Result};
use cuteafd_loader::families::inkling2::execution::UnitMetadata;
use cuteafd_loader::families::inkling2_mini::{Inkling2MiniConfig, LoopVisit};

pub(crate) trait UnitIo<'a> {
    fn begin(
        &mut self,
        unit: &Unit,
        owner: &UnitOwner<'a>,
        uploads: &mut crate::shared::token_table::RetainedUploads<'a>,
    ) -> Result<()>;
    fn end(&mut self, unit: &Unit, owner: &UnitOwner<'a>) -> Result<()>;
    fn tap(&mut self, _visit_idx: usize, _unit: &Unit, _owner: &UnitOwner<'a>) -> Result<()> {
        Ok(())
    }
    fn has_taps(&self) -> bool {
        false
    }
}

/// Selects which rows inside `member.head` segments are projected through the vocabulary head.
/// Drafter final-norm taps (when `draft.is_some()`) always normalize all rows of the unit.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum HeadProjection {
    /// Project every row of every `member.head` segment (decode, verify, golden).
    AllHeadRows,
    /// Project only the final 1 row of each `member.head` segment (serving prefill & packed prefill).
    MemberLastRow,
    /// Project only rows whose 1-based token position is `>= first_position` (serving prefill scoring).
    FromPosition(usize),
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct HeadEmit {
    member_index: usize,
    rows: std::ops::Range<usize>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct EndSpanAction {
    /// Rows to run through `MiniHead::normalize` (full 64-row span when drafter is active or
    /// all rows are projected; narrowed to the exact projected slice otherwise).
    normalize: std::ops::Range<usize>,
    /// Whether to write `normalize` output into `draft.taps().final_norm`.
    tap_final_norm: bool,
    /// Exact row slices inside `normalize` to project through the vocabulary unembed head.
    projections: Vec<HeadEmit>,
}

fn plan_end_actions(
    unit_rows: usize,
    head_rows: usize,
    members: &[UnitSegment],
    projection: HeadProjection,
    has_draft: bool,
) -> Result<Vec<EndSpanAction>> {
    let any_head = members.iter().any(|m| m.head);
    if !any_head && !has_draft {
        return Ok(Vec::new());
    }
    let mut actions = Vec::new();
    for span in crate::shared::probe::head_spans(unit_rows, head_rows)? {
        let mut projections = Vec::new();
        for (member_index, member) in members.iter().enumerate() {
            if !member.head || member.rows.is_empty() {
                continue;
            }
            let target_rows = match projection {
                HeadProjection::AllHeadRows => member.rows.clone(),
                HeadProjection::MemberLastRow => (member.rows.end - 1)..member.rows.end,
                HeadProjection::FromPosition(first_pos) => {
                    let member_first_pos = member.tokens.start + 1;
                    let skip = first_pos.saturating_sub(member_first_pos);
                    let start = member.rows.start.saturating_add(skip);
                    if start >= member.rows.end {
                        continue;
                    }
                    start..member.rows.end
                }
            };
            let first = span.start.max(target_rows.start);
            let end = span.end.min(target_rows.end);
            if first < end {
                projections.push(HeadEmit {
                    member_index,
                    rows: first..end,
                });
            }
        }
        if has_draft {
            actions.push(EndSpanAction {
                normalize: span,
                tap_final_norm: true,
                projections,
            });
        } else if !projections.is_empty() {
            // Without a drafter, normalize only the minimal bounding slice of rows that will be projected.
            let min_start = projections.iter().map(|p| p.rows.start).min().unwrap();
            let max_end = projections.iter().map(|p| p.rows.end).max().unwrap();
            actions.push(EndSpanAction {
                normalize: min_start..max_end,
                tap_final_norm: false,
                projections,
            });
        }
    }
    Ok(actions)
}

pub(crate) struct TokenIo<'s, 'a> {
    pub kernels: &'s MiniKernels<'a>,
    pub head: &'s super::head::MiniHead<'a>,
    pub cache: &'s super::cache::CacheBank<'a>,
    pub target: &'s MiniTargetWeights<'a>,
    pub cfg: &'s Inkling2MiniConfig,
    pub tokens: &'s [&'s [u32]],
    pub starts: &'s [usize],
    pub slots: &'s [usize],
    pub retained_rows: bool,
    pub head_projection: HeadProjection,
    pub draft: Option<&'s super::draft::Binding<'a>>,
    pub publish_draft: bool,
    pub media: &'s [Option<cuteafd_engine::media::RequestMedia>],
    pub on_logits: &'s mut dyn FnMut(usize, usize, crate::shared::token_io::DeviceLogits) -> Result<()>,
}

impl<'a> UnitIo<'a> for TokenIo<'_, 'a> {
    fn tap(&mut self, visit_idx: usize, unit: &Unit, owner: &UnitOwner<'a>) -> Result<()> {
        if let Some(draft) = self.draft {
            draft
                .taps(self.retained_rows, unit.slot)?
                .entry(visit_idx, owner.buffer("partial")?, unit.rows.len())?;
        }
        Ok(())
    }

    fn has_taps(&self) -> bool {
        self.draft.is_some()
    }

    fn begin(
        &mut self,
        unit: &Unit,
        owner: &UnitOwner<'a>,
        uploads: &mut crate::shared::token_table::RetainedUploads<'a>,
    ) -> Result<()> {
        let tokens = unit_tokens(unit, self.tokens, self.starts)?;
        let mut chunk = cuteafd_engine::media::MediaChunk::default();
        for member in members(unit) {
            if let Some(media) = self.media.get(member.sequence).and_then(Option::as_ref) {
                let mut part = cuteafd_engine::media::MediaChunk::default();
                media.write_chunk(member.tokens.start, member.tokens.end, &mut part)?;
                chunk
                    .indices
                    .extend(part.indices.into_iter().map(|row| row + member.rows.start as u32));
                chunk.features.extend(part.features);
            }
        }
        self.head
            .embed_unit(self.kernels, owner, self.target, self.cfg, &tokens, &chunk, uploads)
    }

    fn end(&mut self, unit: &Unit, owner: &UnitOwner<'a>) -> Result<()> {
        let members = members(unit);
        let actions = plan_end_actions(
            unit.rows.len(),
            self.head.rows(),
            &members,
            self.head_projection,
            self.draft.is_some(),
        )?;
        if actions.is_empty() && !(self.publish_draft && self.draft.is_some()) {
            return Ok(());
        }
        let sequence = *self
            .slots
            .get(members[0].sequence)
            .context("Inkling 2 Mini head sequence slot absent")?;
        let final_candidates = self.cfg.text.residual_blocks() + 1; // 5
        let final_slot = self.cfg.cache_slot(self.cfg.text.layers - 1, 0);
        for action in actions {
            let state = owner.segment(self.cache, sequence, final_slot, action.normalize.clone())?;
            self.head
                .normalize(self.kernels, &state, self.target, action.normalize.len(), final_candidates)?;
            if action.tap_final_norm {
                if let Some(draft) = self.draft {
                    draft.taps(self.retained_rows, unit.slot)?.final_norm(
                        self.head.normalized(action.normalize.len())?,
                        action.normalize.start,
                        action.normalize.len(),
                    )?;
                }
            }
            if action.projections.is_empty() {
                continue;
            }
            // Fast path: when projections contiguously cover the entire normalized slice
            // (decode, verify, golden, or single-row last-token prefill), project in one GEMM.
            let full_contiguous = action.projections.first().unwrap().rows.start == action.normalize.start
                && action.projections.last().unwrap().rows.end == action.normalize.end
                && action
                    .projections
                    .windows(2)
                    .all(|pair| pair[0].rows.end == pair[1].rows.start);
            if full_contiguous {
                let output_row = if self.retained_rows {
                    action.normalize.start
                } else {
                    0
                };
                let logits = self.head.project_normalized(
                    self.target,
                    self.cfg,
                    0,
                    action.normalize.len(),
                    output_row,
                )?;
                for emit in &action.projections {
                    let member = &members[emit.member_index];
                    let offset = (emit.rows.start - action.normalize.start) * self.cfg.text.unpadded_vocab_size * 4;
                    (self.on_logits)(
                        member.sequence,
                        member.tokens.start + emit.rows.start - member.rows.start + 1,
                        crate::shared::token_io::DeviceLogits {
                            ptr: logits.ptr.wrapping_byte_add(offset),
                            rows: emit.rows.len(),
                            vocab: self.cfg.text.unpadded_vocab_size,
                            stride: self.cfg.text.unpadded_vocab_size,
                            stream: owner.stream.raw,
                            greedy: None,
                        },
                    )?;
                }
            } else {
                for emit in &action.projections {
                    let member = &members[emit.member_index];
                    let norm_offset = emit.rows.start - action.normalize.start;
                    let output_row = if self.retained_rows { emit.rows.start } else { 0 };
                    let logits = self.head.project_normalized(
                        self.target,
                        self.cfg,
                        norm_offset,
                        emit.rows.len(),
                        output_row,
                    )?;
                    (self.on_logits)(
                        member.sequence,
                        member.tokens.start + emit.rows.start - member.rows.start + 1,
                        crate::shared::token_io::DeviceLogits {
                            ptr: logits.ptr,
                            rows: emit.rows.len(),
                            vocab: self.cfg.text.unpadded_vocab_size,
                            stride: self.cfg.text.unpadded_vocab_size,
                            stream: owner.stream.raw,
                            greedy: None,
                        },
                    )?;
                }
            }
        }
        if self.publish_draft {
            if let Some(draft) = self.draft {
                let taps = draft.taps(self.retained_rows, unit.slot)?;
                for member in &members {
                    let slot = *self
                        .slots
                        .get(member.sequence)
                        .context("Mini draft member slot missing")?;
                    draft.update(taps, slot, member.tokens.start, member.rows.start, member.rows.len())?;
                }
            }
        }
        Ok(())
    }
}

pub(crate) struct Stateful<'s, 'a> {
    pub(super) graphs: Option<&'s mut DecodeGraphs<'a>>,
    pub kernels: &'s MiniKernels<'a>,
    pub owners: &'s [UnitOwner<'a>],
    pub cache: &'s super::cache::CacheBank<'a>,
    pub target: &'s MiniTargetWeights<'a>,
    pub cfg: &'s Inkling2MiniConfig,
    pub visits: &'s [LoopVisit],
    pub placements: &'s mut [super::cache::Placement],
    pub uploads: &'s mut [crate::shared::token_table::RetainedUploads<'a>],
    pub local: &'s [Box<dyn Dispatcher + 'a>],
    pub io: &'s mut dyn UnitIo<'a>,
}

impl Stateful<'_, '_> {
    pub fn run(&mut self, units: &[Unit]) -> Result<()> {
        let total_visits = self.visits.len();
        ensure!(
            !units.is_empty()
                && self.owners.len() == self.uploads.len()
                && self.local.len() == self.cfg.text.layers
                && total_visits == self.cfg.total_visits(),
            "Inkling 2 Mini stateful prefill owners/visits differ"
        );
        let stream = &self
            .owners
            .first()
            .context("Inkling 2 Mini prefill has no unit owners")?
            .stream;
        ensure!(
            self.owners.iter().all(|o| std::rc::Rc::ptr_eq(&o.stream, stream)),
            "Inkling 2 Mini causal state requires one ordered stream"
        );
        ensure!(
            units.iter().all(|u| {
                u.slot < self.owners.len()
                    && u.layers == (0..total_visits)
                    && members(u).iter().all(|m| m.sequence < self.placements.len())
            }),
            "Inkling 2 Mini prefill unit differs from admitted requests"
        );
        // Monolithic 48-visit TP1 decode/verify graph path: upload metadata once outside
        // capture, run all 48 visits (taps + attention + routes + local experts + join)
        // inside a single kernel-only CUDA graph, then run head/commit outside.
        if self.graphs.is_some() {
            ensure!(
                units.len() == 1 && self.owners.len() == 1 && units[0].slot == 0,
                "Inkling 2 Mini graphed step requires one single-slot decode/verify unit"
            );
            let mut ops = StatefulOps {
                kernels: self.kernels,
                target: self.target,
                local: self.local,
                io: self.io,
            };
            return run_monolithic_step(
                self.graphs.as_deref_mut(),
                self.cfg,
                self.visits,
                &self.owners[0],
                self.cache,
                self.placements,
                &mut self.uploads[0],
                &units[0],
                ops.io.has_taps(),
                &mut ops,
            );
        }

        let waves = self.owners.len();
        let mut prepared = PreparedUnitRunner {
            cfg: self.cfg,
            visits: self.visits,
            owners: self.owners,
            cache: self.cache,
            placements: self.placements,
            uploads: self.uploads,
            members: vec![Vec::new(); waves],
            ops: StatefulOps {
                kernels: self.kernels,
                target: self.target,
                local: self.local,
                io: self.io,
            },
        };
        run_prefill_units(&mut prepared, units, waves)
    }

    /// Runs one eager warm pass (`capture = false`) or one capture-only monolithic
    /// prewarm (`capture = true`, no head projection, no replay launch).
    pub(super) fn prewarm_unit(&mut self, unit: &Unit, capture: bool) -> Result<bool> {
        ensure!(
            self.owners.len() == 1 && self.uploads.len() == 1 && unit.slot == 0,
            "Inkling 2 Mini prewarm requires single decode unit owner"
        );
        let owner = &self.owners[0];
        let uploads = &mut self.uploads[0];
        let mut ops = StatefulOps {
            kernels: self.kernels,
            target: self.target,
            local: self.local,
            io: self.io,
        };
        if !capture {
            ops.begin_io(unit, owner, uploads)?;
        }
        let packed = build_packed_members(self.cfg, self.cache, self.placements, unit)?;
        owner.packed_metadata(self.cache, &packed, uploads)?;
        let rows = unit.rows.len();
        let mode = if owner.replaying() {
            Mode::Verify
        } else {
            Mode::Decode
        };
        let has_taps = ops.io.has_taps();
        let mut body = || {
            for (visit_idx, &visit) in self.visits.iter().enumerate() {
                let state = PackedVisitUnit {
                    unit: owner,
                    cache: self.cache,
                    cache_slot: visit.cache_slot,
                    members: &packed,
                };
                ops.attention(visit_idx, visit, unit, self.cfg, &state)?;
                ops.local_ffn(visit_idx, visit, unit, self.cfg, &state)?;
            }
            Ok(())
        };
        if capture {
            let Some(graphs) = self.graphs.as_deref_mut() else {
                return Ok(false);
            };
            graphs::prewarm_step(graphs, &owner.stream, rows, mode, has_taps, body)
        } else {
            body()?;
            Ok(false)
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn run_monolithic_step<'a, O: VisitOps<'a>>(
    graphs: Option<&mut DecodeGraphs<'a>>,
    cfg: &Inkling2MiniConfig,
    visits: &[LoopVisit],
    owner: &UnitOwner<'a>,
    cache: &super::cache::CacheBank<'a>,
    placements: &mut [super::cache::Placement],
    uploads: &mut crate::shared::token_table::RetainedUploads<'a>,
    unit: &Unit,
    has_taps: bool,
    ops: &mut O,
) -> Result<()> {
    ops.begin_io(unit, owner, uploads)?;
    let packed = build_packed_members(cfg, cache, placements, unit)?;
    owner.packed_metadata(cache, &packed, uploads)?;
    let rows = unit.rows.len();
    let slots = packed.len();
    let mode = if owner.replaying() {
        Mode::Verify
    } else {
        Mode::Decode
    };
    graphs::step(graphs, &owner.stream, rows, slots, mode, has_taps, || {
        for (visit_idx, &visit) in visits.iter().enumerate() {
            let state = PackedVisitUnit {
                unit: owner,
                cache,
                cache_slot: visit.cache_slot,
                members: &packed,
            };
            ops.attention(visit_idx, visit, unit, cfg, &state)?;
            ops.local_ffn(visit_idx, visit, unit, cfg, &state)?;
        }
        Ok(())
    })?;
    ops.end_io(unit, owner)?;
    for member in members(unit) {
        let placement = &mut placements[member.sequence];
        ensure!(
            placement.state_len == member.tokens.start,
            "Inkling 2 Mini causal commit is out of order"
        );
        placement.state_len = member.tokens.end;
        placement.len = member.tokens.end;
    }
    Ok(())
}

trait VisitOps<'a> {
    fn begin_io(
        &mut self,
        unit: &Unit,
        owner: &UnitOwner<'a>,
        uploads: &mut crate::shared::token_table::RetainedUploads<'a>,
    ) -> Result<()>;
    fn attention(
        &mut self,
        visit_idx: usize,
        visit: LoopVisit,
        unit: &Unit,
        cfg: &Inkling2MiniConfig,
        state: &PackedVisitUnit<'_, 'a>,
    ) -> Result<()>;
    fn local_ffn(
        &mut self,
        visit_idx: usize,
        visit: LoopVisit,
        unit: &Unit,
        cfg: &Inkling2MiniConfig,
        state: &PackedVisitUnit<'_, 'a>,
    ) -> Result<()>;
    fn end_io(&mut self, unit: &Unit, owner: &UnitOwner<'a>) -> Result<()>;
}

struct StatefulOps<'p, 's, 'a> {
    kernels: &'s MiniKernels<'a>,
    target: &'s MiniTargetWeights<'a>,
    local: &'s [Box<dyn Dispatcher + 'a>],
    io: &'p mut dyn UnitIo<'a>,
}

impl<'a> VisitOps<'a> for StatefulOps<'_, '_, 'a> {
    fn begin_io(
        &mut self,
        unit: &Unit,
        owner: &UnitOwner<'a>,
        uploads: &mut crate::shared::token_table::RetainedUploads<'a>,
    ) -> Result<()> {
        self.io.begin(unit, owner, uploads)
    }

    fn attention(
        &mut self,
        visit_idx: usize,
        visit: LoopVisit,
        unit: &Unit,
        cfg: &Inkling2MiniConfig,
        state: &PackedVisitUnit<'_, 'a>,
    ) -> Result<()> {
        self.io.tap(visit_idx, unit, state.unit)?;
        let rows = unit.rows.len();
        layers::begin_visit(self.kernels, state, self.target, cfg, visit, rows)
    }

    fn local_ffn(
        &mut self,
        _visit_idx: usize,
        visit: LoopVisit,
        unit: &Unit,
        cfg: &Inkling2MiniConfig,
        state: &PackedVisitUnit<'_, 'a>,
    ) -> Result<()> {
        let rows = unit.rows.len();
        let experts = self.local[visit.layer].as_ref();
        layers::prepare_routes(self.kernels, state.unit, self.target, experts, visit.layer, rows)?;
        layers::dispatch_experts(state.unit, experts, rows)?;
        layers::join_visit(self.kernels, state, self.target, cfg, visit, rows)
    }

    fn end_io(&mut self, unit: &Unit, owner: &UnitOwner<'a>) -> Result<()> {
        self.io.end(unit, owner)
    }
}

fn build_packed_members(
    cfg: &Inkling2MiniConfig,
    cache: &super::cache::CacheBank<'_>,
    placements: &[super::cache::Placement],
    unit: &Unit,
) -> Result<Vec<PackedMember>> {
    members(unit)
        .into_iter()
        .map(|member| {
            let placement = &placements[member.sequence];
            Ok(PackedMember {
                sequence: placement.slot,
                rows: member.rows,
                metadata: UnitMetadata::new(
                    &cfg.text,
                    member.tokens,
                    &placement.units,
                    cache.layout.pool_pages,
                )?,
            })
        })
        .collect()
}

struct PreparedUnitRunner<'s, 'a, O> {
    cfg: &'s Inkling2MiniConfig,
    visits: &'s [LoopVisit],
    owners: &'s [UnitOwner<'a>],
    cache: &'s super::cache::CacheBank<'a>,
    placements: &'s mut [super::cache::Placement],
    uploads: &'s mut [crate::shared::token_table::RetainedUploads<'a>],
    members: Vec<Vec<PackedMember>>,
    ops: O,
}

fn unit_tokens(unit: &Unit, sources: &[&[u32]], starts: &[usize]) -> Result<Vec<u32>> {
    let mut tokens = Vec::with_capacity(unit.rows.len());
    for member in members(unit) {
        let source = sources
            .get(member.sequence)
            .context("Mini unit token source absent")?;
        let origin = *starts
            .get(member.sequence)
            .context("Mini token source origin absent")?;
        let first = member
            .tokens
            .start
            .checked_sub(origin)
            .context("Mini token span precedes submitted suffix")?;
        let end = member
            .tokens
            .end
            .checked_sub(origin)
            .context("Mini token end precedes submitted suffix")?;
        tokens.extend_from_slice(source.get(first..end).context("Mini unit token span outside request")?);
    }
    ensure!(tokens.len() == unit.rows.len(), "Mini token members do not cover unit");
    Ok(tokens)
}

fn members(unit: &Unit) -> Vec<UnitSegment> {
    if !unit.segments.is_empty() {
        return unit.segments.clone();
    }
    vec![UnitSegment {
        sequence: unit.sequence,
        rows: 0..unit.rows.len(),
        tokens: unit.rows.clone(),
        prompt_end: false,
        snapshot: false,
        head: false,
    }]
}

impl<'a, O: VisitOps<'a>> PrefillHooks for PreparedUnitRunner<'_, 'a, O> {
    type Ticket = ();

    fn route(&self, _visit_idx: usize) -> FfnRoute {
        FfnRoute::Local
    }

    fn begin_unit(&mut self, unit: &Unit) -> Result<()> {
        let owner = &self.owners[unit.slot];
        let uploads = &mut self.uploads[unit.slot];
        // Embed runs first (it writes token ids into `first` before reading them);
        // then upload all 9 read-only metadata planes once for all visits.
        self.ops.begin_io(unit, owner, uploads)?;
        let packed = build_packed_members(self.cfg, self.cache, self.placements, unit)?;
        owner.packed_metadata(self.cache, &packed, uploads)?;
        self.members[unit.slot] = packed;
        Ok(())
    }

    fn attention(&mut self, visit_idx: usize, unit: &Unit) -> Result<()> {
        let visit = self.visits[visit_idx];
        let owner = &self.owners[unit.slot];
        let members = &self.members[unit.slot];
        ensure!(
            !members.is_empty(),
            "Inkling 2 Mini attention ran before begin_unit prepared metadata"
        );
        let state = PackedVisitUnit {
            unit: owner,
            cache: self.cache,
            cache_slot: visit.cache_slot,
            members,
        };
        self.ops.attention(visit_idx, visit, unit, self.cfg, &state)
    }

    fn local_ffn(&mut self, visit_idx: usize, unit: &Unit) -> Result<()> {
        let visit = self.visits[visit_idx];
        let owner = &self.owners[unit.slot];
        let members = &self.members[unit.slot];
        ensure!(
            !members.is_empty(),
            "Inkling 2 Mini local_ffn ran before begin_unit prepared metadata"
        );
        let state = PackedVisitUnit {
            unit: owner,
            cache: self.cache,
            cache_slot: visit.cache_slot,
            members,
        };
        self.ops.local_ffn(visit_idx, visit, unit, self.cfg, &state)
    }

    fn send_experts(&mut self, _: usize, _: &Unit) -> Result<()> {
        bail!("Inkling 2 Mini is TP1 local-only; Spark expert offload is unused")
    }

    fn land_experts(&mut self, _: usize, _: &Unit, _: ()) -> Result<()> {
        bail!("Inkling 2 Mini is TP1 local-only; Spark expert offload is unused")
    }

    fn end_unit(&mut self, unit: &Unit) -> Result<()> {
        self.ops.end_io(unit, &self.owners[unit.slot])
    }

    fn commit_unit(&mut self, unit: &Unit) -> Result<()> {
        for member in members(unit) {
            let placement = &mut self.placements[member.sequence];
            ensure!(
                placement.state_len == member.tokens.start,
                "Inkling 2 Mini causal commit is out of order"
            );
            placement.state_len = member.tokens.end;
            placement.len = member.tokens.end;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::shared::memory::device::{Device, Stream};
    use cuteafd_ffi::native_library_lifetime_fixture::Fixture;
    use cuteafd_loader::families::inkling2_mini::{
        execution::{MiniCacheLayout, MiniUnitLayout},
        InklingKvCache,
    };
    use std::rc::Rc;

    fn test_config() -> Result<Inkling2MiniConfig> {
        Inkling2MiniConfig::from_hf(&serde_json::from_str(include_str!(
            "../../../../cuteafd-loader/src/families/inkling2_mini/fixtures/config.json"
        ))?)
    }

    struct StubVisitOps {
        upload_tokens: bool,
        attention_visits: usize,
        ffn_visits: usize,
    }

    impl<'a> VisitOps<'a> for StubVisitOps {
        fn begin_io(
            &mut self,
            unit: &Unit,
            owner: &UnitOwner<'a>,
            uploads: &mut crate::shared::token_table::RetainedUploads<'a>,
        ) -> Result<()> {
            if self.upload_tokens {
                let bytes = vec![0u8; unit.rows.len() * 4];
                // SAFETY: matches embed_unit uploading token ids into `first`.
                unsafe {
                    uploads.upload(owner.buffer("first")?, &bytes, owner.stream.raw)?;
                }
            }
            Ok(())
        }

        fn attention(
            &mut self,
            _visit_idx: usize,
            _visit: LoopVisit,
            _unit: &Unit,
            _cfg: &Inkling2MiniConfig,
            state: &PackedVisitUnit<'_, 'a>,
        ) -> Result<()> {
            assert!(!state.members.is_empty());
            // Verify every visit resolves the shared device metadata views.
            let _ = state.buffer("positions")?;
            let _ = state.buffer("cache_slots")?;
            let _ = state.buffer("ring_slots")?;
            let _ = state.buffer("page_table")?;
            self.attention_visits += 1;
            Ok(())
        }

        fn local_ffn(
            &mut self,
            _visit_idx: usize,
            _visit: LoopVisit,
            _unit: &Unit,
            _cfg: &Inkling2MiniConfig,
            state: &PackedVisitUnit<'_, 'a>,
        ) -> Result<()> {
            assert!(!state.members.is_empty());
            self.ffn_visits += 1;
            Ok(())
        }

        fn end_io(&mut self, _unit: &Unit, _owner: &UnitOwner<'a>) -> Result<()> {
            Ok(())
        }
    }

    #[test]
    fn prefill_metadata_h2d_copies_are_bounded_and_independent_of_visit_count() -> Result<()> {
        let fixture = Fixture::build_for_tests()?;
        let library = fixture.load()?;
        fixture.immediate_copies(&library)?;
        fixture.configure_device_flags(&library, 0)?;
        let device = Device {
            library: &library,
            id: 0,
        };
        let stream = Rc::new(Stream::new(device)?);
        let cfg = test_config()?;
        let all_visits = cfg.visits();
        assert_eq!(all_visits.len(), 48);
        let cache = super::super::cache::CacheBank::new(
            vec![stream.clone()],
            MiniCacheLayout::with_format(&cfg, 2, 4, InklingKvCache::Fp8)?,
            0,
        )?;
        let owners = vec![UnitOwner::new(
            stream.clone(),
            MiniUnitLayout::serving(&cfg, 4, 128, InklingKvCache::Fp8, 2)?,
        )?];
        let mut uploads = vec![owners[0].uploads(&cfg, 2)?];

        fn run_for_visits<'a>(
            fixture: &Fixture,
            cfg: &Inkling2MiniConfig,
            all_visits: &[LoopVisit],
            stream: &Stream<'a>,
            cache: &super::super::cache::CacheBank<'a>,
            owners: &[UnitOwner<'a>],
            uploads: &mut [crate::shared::token_table::RetainedUploads<'a>],
            visit_count: usize,
            upload_tokens: bool,
        ) -> Result<usize> {
            let mut placements = vec![
                super::super::cache::Placement {
                    units: vec![2],
                    slot: 1,
                    len: 63,
                    state_len: 63,
                    media: None,
                },
                super::super::cache::Placement {
                    units: vec![0, 3],
                    slot: 0,
                    len: 65,
                    state_len: 65,
                    media: None,
                },
            ];
            let unit = Unit {
                index: 0,
                sequence: 0,
                segments: vec![
                    UnitSegment {
                        sequence: 0,
                        rows: 0..1,
                        tokens: 63..64,
                        prompt_end: false,
                        snapshot: false,
                        head: true,
                    },
                    UnitSegment {
                        sequence: 1,
                        rows: 1..4,
                        tokens: 65..68,
                        prompt_end: false,
                        snapshot: false,
                        head: true,
                    },
                ],
                rows: 0..4,
                layers: 0..visit_count,
                slot: 0,
                barrier: false,
            };
            let before = fixture.events()?.chars().filter(|&c| c == 'C').count();
            let mut runner = PreparedUnitRunner {
                cfg,
                visits: &all_visits[..visit_count],
                owners,
                cache,
                placements: &mut placements,
                uploads,
                members: vec![Vec::new(); 1],
                ops: StubVisitOps {
                    upload_tokens,
                    attention_visits: 0,
                    ffn_visits: 0,
                },
            };
            run_prefill_units(&mut runner, &[unit], 1)?;
            stream.drain()?;
            assert_eq!(runner.ops.attention_visits, visit_count);
            assert_eq!(runner.ops.ffn_visits, visit_count);
            assert_eq!(placements[0].len, 64);
            assert_eq!(placements[1].len, 68);
            let after = fixture.events()?.chars().filter(|&c| c == 'C').count();
            Ok(after - before)
        }

        // Metadata-only path: 1 batched H2D copy for 1 visit AND 1 batched H2D copy for 48 visits.
        let copies_1 = run_for_visits(&fixture, &cfg, &all_visits, &stream, &cache, &owners, &mut uploads, 1, false)?;
        let copies_48 =
            run_for_visits(&fixture, &cfg, &all_visits, &stream, &cache, &owners, &mut uploads, 48, false)?;
        assert_eq!(copies_1, 1);
        assert_eq!(copies_48, 1);
        assert_eq!(copies_48, copies_1, "metadata H2D copies must not scale with visit count");

        // Full begin_unit path (token ids + batched metadata): 2 copies for 1 visit AND 2 for 48 visits.
        let full_1 = run_for_visits(&fixture, &cfg, &all_visits, &stream, &cache, &owners, &mut uploads, 1, true)?;
        let full_48 = run_for_visits(&fixture, &cfg, &all_visits, &stream, &cache, &owners, &mut uploads, 48, true)?;
        assert_eq!(full_1, 2);
        assert_eq!(full_48, 2);
        assert_eq!(full_48, full_1, "full unit H2D copies must stay bounded and independent of 48 visits");
        Ok(())
    }

    #[test]
    fn prefill_head_projects_only_last_row_while_normalizing_all_rows_for_drafter() -> Result<()> {
        // Two 4096-row units covering an 8192-token prompt: unit 0 has head=false, unit 1 has head=true.
        let unit0 = [UnitSegment {
            sequence: 0,
            rows: 0..4096,
            tokens: 0..4096,
            prompt_end: false,
            snapshot: false,
            head: false,
        }];
        let unit1 = [UnitSegment {
            sequence: 0,
            rows: 0..4096,
            tokens: 4096..8192,
            prompt_end: true,
            snapshot: false,
            head: true,
        }];

        // Without drafter: unit 0 does zero work; unit 1 normalizes & projects only the final row (4095..4096).
        let u0_off = plan_end_actions(4096, 64, &unit0, HeadProjection::MemberLastRow, false)?;
        let u1_off = plan_end_actions(4096, 64, &unit1, HeadProjection::MemberLastRow, false)?;
        assert!(u0_off.is_empty());
        assert_eq!(u1_off.len(), 1);
        assert_eq!(u1_off[0].normalize, 4095..4096);
        assert!(!u1_off[0].tap_final_norm);
        assert_eq!(u1_off[0].projections, vec![HeadEmit { member_index: 0, rows: 4095..4096 }]);

        // With drafter (`dspark`): EVERY row of BOTH units (8192 total) is normalized & tapped for
        // `draft.taps().final_norm`, while ONLY the single final row (4095..4096 of unit 1) is projected!
        let u0_draft = plan_end_actions(4096, 64, &unit0, HeadProjection::MemberLastRow, true)?;
        let u1_draft = plan_end_actions(4096, 64, &unit1, HeadProjection::MemberLastRow, true)?;
        let norm_rows = |actions: &[EndSpanAction]| -> usize { actions.iter().map(|a| a.normalize.len()).sum() };
        let tap_rows = |actions: &[EndSpanAction]| -> usize {
            actions.iter().filter(|a| a.tap_final_norm).map(|a| a.normalize.len()).sum()
        };
        let proj_rows = |actions: &[EndSpanAction]| -> usize {
            actions
                .iter()
                .flat_map(|a| a.projections.iter())
                .map(|p| p.rows.len())
                .sum()
        };
        assert_eq!(u0_draft.len(), 64);
        assert_eq!(u1_draft.len(), 64);
        assert_eq!(norm_rows(&u0_draft) + norm_rows(&u1_draft), 8192);
        assert_eq!(tap_rows(&u0_draft) + tap_rows(&u1_draft), 8192);
        assert_eq!(proj_rows(&u0_draft), 0);
        assert_eq!(proj_rows(&u1_draft), 1);
        assert_eq!(
            u1_draft.last().unwrap().projections,
            vec![HeadEmit { member_index: 0, rows: 4095..4096 }]
        );

        // Packed multi-member prefill with drafter: all 100 rows normalized & tapped, only the 2 member last rows projected.
        let packed = [
            UnitSegment {
                sequence: 0,
                rows: 0..20,
                tokens: 0..20,
                prompt_end: true,
                snapshot: false,
                head: true,
            },
            UnitSegment {
                sequence: 1,
                rows: 20..100,
                tokens: 0..80,
                prompt_end: true,
                snapshot: false,
                head: true,
            },
        ];
        let packed_actions = plan_end_actions(100, 64, &packed, HeadProjection::MemberLastRow, true)?;
        assert_eq!(norm_rows(&packed_actions), 100);
        assert_eq!(tap_rows(&packed_actions), 100);
        assert_eq!(proj_rows(&packed_actions), 2);
        assert_eq!(packed_actions[0].projections, vec![HeadEmit { member_index: 0, rows: 19..20 }]);
        assert_eq!(packed_actions[1].projections, vec![HeadEmit { member_index: 1, rows: 99..100 }]);

        // Scoring suffix (`FromPosition`) projects only positions >= first_pos while normalizing all rows for drafter.
        let score_actions = plan_end_actions(4096, 64, &unit1, HeadProjection::FromPosition(8180), true)?;
        assert_eq!(norm_rows(&score_actions), 4096);
        assert_eq!(tap_rows(&score_actions), 4096);
        assert_eq!(proj_rows(&score_actions), 13); // 1-based positions 8180..=8192 -> rows 4083..4096

        // Decode / golden (`AllHeadRows`) still projects all rows, including 128-row decode heads in one span.
        let all_actions = plan_end_actions(8, 64, &[UnitSegment {
            sequence: 0,
            rows: 0..8,
            tokens: 100..108,
            prompt_end: false,
            snapshot: false,
            head: true,
        }], HeadProjection::AllHeadRows, true)?;
        assert_eq!(norm_rows(&all_actions), 8);
        assert_eq!(proj_rows(&all_actions), 8);
        let wide_actions = plan_end_actions(128, 128, &[UnitSegment {
            sequence: 0,
            rows: 0..128,
            tokens: 100..228,
            prompt_end: false,
            snapshot: false,
            head: true,
        }], HeadProjection::AllHeadRows, true)?;
        assert_eq!(wide_actions.len(), 1);
        assert_eq!(norm_rows(&wide_actions), 128);
        assert_eq!(proj_rows(&wide_actions), 128);
        Ok(())
    }

    #[test]
    fn monolithic_decode_graph_records_one_kernel_only_graph_invariant_to_prompt_length() -> Result<()> {
        let fixture = Fixture::build_for_tests()?;
        let library = fixture.load()?;
        fixture.immediate_copies(&library)?;
        fixture.configure_device_flags(&library, 0)?;
        fixture.configure_memory_usage(&library, 0, 1024 << 20, 0)?;
        let device = Device {
            library: &library,
            id: 0,
        };
        let stream = Rc::new(Stream::new(device)?);
        let cfg = test_config()?;
        let all_visits = cfg.visits();
        assert_eq!(all_visits.len(), 48);
        let cache = super::super::cache::CacheBank::new(
            vec![stream.clone()],
            MiniCacheLayout::with_format(&cfg, 2, 512, InklingKvCache::Fp8)?,
            0,
        )?;
        let mut owner = UnitOwner::new(
            stream.clone(),
            MiniUnitLayout::serving_decode(&cfg, 32_768, InklingKvCache::Fp8, 2)?,
        )?;
        owner.decode_attention = true;
        let mut uploads = owner.uploads(&cfg, 2)?;
        let policy = cuteafd_loader::serving_capacity::inkling2_mini_graphs::policy(
            0,
            cuteafd_loader::serving_capacity::inkling2_mini_graphs::KeySpace::for_config(
                2,
                cuteafd_loader::families::inkling2_mini::execution::DECODE_ROWS,
                true,
            ),
        )?;
        let mut graphs = DecodeGraphs::new_for_test(&stream, -270_101, policy)?;

        // Structural guard: decode/verify captured paths must stay kernel-only (no captured
        // `cudaMemcpyAsync` / `cudaMemsetAsync` nodes; experts and draft taps use `DeviceRowCopy`).
        let code_only = |src: &str| -> String {
            src.lines()
                .filter(|line| !line.trim_start().starts_with("//"))
                .collect::<Vec<_>>()
                .join("\n")
        };
        let experts_code = code_only(include_str!("experts.rs"));
        assert!(
            !experts_code.contains("copy_d2d_async")
                && !experts_code.contains("cuda_zero_bytes_async")
                && experts_code.contains("launch_rows"),
            "experts.rs must use shared DeviceRowCopy::launch_rows instead of memcpy/memset inside captured graphs"
        );
        let taps_entry_code = code_only(
            include_str!("draft_taps.rs")
                .split("pub fn final_norm")
                .next()
                .expect("draft_taps entry section"),
        );
        assert!(
            !taps_entry_code.contains("copy_d2d_2d_async")
                && !taps_entry_code.contains("cuda_zero_bytes_async")
                && taps_entry_code.contains("launch_rows"),
            "draft_taps::entry must use shared DeviceRowCopy::launch_rows inside captured graphs"
        );

        struct CaptureAuditOps<'f> {
            fixture: &'f Fixture,
            attention_visits: usize,
            ffn_visits: usize,
            captured_events: String,
            first_visit_event_len: Option<usize>,
        }
        impl<'a> VisitOps<'a> for CaptureAuditOps<'_> {
            fn begin_io(
                &mut self,
                unit: &Unit,
                owner: &UnitOwner<'a>,
                uploads: &mut crate::shared::token_table::RetainedUploads<'a>,
            ) -> Result<()> {
                let bytes = vec![0u8; unit.rows.len() * 4];
                // SAFETY: matches embed_unit uploading token ids before graph capture.
                unsafe {
                    uploads.upload(owner.buffer("first")?, &bytes, owner.stream.raw)?;
                }
                Ok(())
            }
            fn attention(
                &mut self,
                visit_idx: usize,
                _visit: LoopVisit,
                _unit: &Unit,
                _cfg: &Inkling2MiniConfig,
                state: &PackedVisitUnit<'_, 'a>,
            ) -> Result<()> {
                if visit_idx == 0 {
                    self.first_visit_event_len = Some(self.fixture.events()?.len());
                }
                assert!(state.decode_attention());
                self.attention_visits += 1;
                Ok(())
            }
            fn local_ffn(
                &mut self,
                visit_idx: usize,
                _visit: LoopVisit,
                _unit: &Unit,
                _cfg: &Inkling2MiniConfig,
                _state: &PackedVisitUnit<'_, 'a>,
            ) -> Result<()> {
                self.ffn_visits += 1;
                if visit_idx == 47 {
                    let all = self.fixture.events()?;
                    let start = self.first_visit_event_len.unwrap_or(all.len());
                    self.captured_events = all[start..].to_string();
                }
                Ok(())
            }
            fn end_io(&mut self, _unit: &Unit, _owner: &UnitOwner<'a>) -> Result<()> {
                Ok(())
            }
        }

        #[allow(clippy::too_many_arguments)]
        fn run_step<'a>(
            fixture: &Fixture,
            library: &'a cuteafd_ffi::NativeLibrary,
            cfg: &Inkling2MiniConfig,
            all_visits: &[LoopVisit],
            stream: &Stream<'a>,
            cache: &super::super::cache::CacheBank<'a>,
            owner: &UnitOwner<'a>,
            graphs: &mut DecodeGraphs<'a>,
            uploads: &mut crate::shared::token_table::RetainedUploads<'a>,
            seq_specs: &[(usize, usize, usize)], // (slot, prompt_len, rows_in_step)
            taps: bool,
        ) -> Result<(usize, usize, usize, usize, usize, String)> {
            let mut placements = Vec::new();
            let mut segments = Vec::new();
            let mut row_cursor = 0usize;
            for (idx, &(slot, prompt_len, step_rows)) in seq_specs.iter().enumerate() {
                let pages = (prompt_len + step_rows).div_ceil(64);
                let page_base = (idx * 256) as u32;
                placements.push(super::super::cache::Placement {
                    units: (page_base..page_base + pages as u32).collect(),
                    slot,
                    len: prompt_len,
                    state_len: prompt_len,
                    media: None,
                });
                segments.push(UnitSegment {
                    sequence: idx,
                    rows: row_cursor..row_cursor + step_rows,
                    tokens: prompt_len..prompt_len + step_rows,
                    prompt_end: false,
                    snapshot: false,
                    head: true,
                });
                row_cursor += step_rows;
            }
            let unit = Unit {
                index: 0,
                sequence: 0,
                segments,
                rows: 0..row_cursor,
                layers: 0..48,
                slot: 0,
                barrier: false,
            };
            let before = fixture.events()?;
            let mut ops = CaptureAuditOps {
                fixture,
                attention_visits: 0,
                ffn_visits: 0,
                captured_events: String::new(),
                first_visit_event_len: None,
            };
            run_monolithic_step(
                Some(graphs),
                cfg,
                all_visits,
                owner,
                cache,
                &mut placements,
                uploads,
                &unit,
                taps,
                &mut ops,
            )?;
            stream.drain()?;
            if graphs.needs_boundary() {
                // SAFETY: stream drained above.
                unsafe {
                    crate::shared::decode_graph::device::observe_completed(library, |visit| graphs.visit(visit))?;
                }
                graphs.observed();
            } else {
                graphs.tick();
            }
            let after = fixture.events()?;
            let delta = &after[before.len()..];
            let captures = delta.chars().filter(|&c| c == 'G').count();
            let launches = delta.chars().filter(|&c| c == 'L').count();
            let updates = delta.chars().filter(|&c| c == 'U').count();
            Ok((captures, launches, updates, ops.attention_visits, ops.ffn_visits, ops.captured_events))
        }

        // 1. First touch at prompt_len=64, rows=4, taps=true: runs eagerly once to warm.
        let (c1, l1, u1, av1, fv1, _) = run_step(
            &fixture, &library, &cfg, &all_visits, &stream, &cache, &owner, &mut graphs, &mut uploads,
            &[(0, 64, 4)], true,
        )?;
        assert_eq!((c1, l1, u1, av1, fv1), (0, 0, 0, 48, 48));

        // 2. Capture pass at prompt_len=64, rows=4, taps=true:
        //    EXACTLY ONE graph recorded for all 48 visits, launched once, ZERO memcpy/memset inside capture.
        let (c2, l2, u2, av2, fv2, captured_body) = run_step(
            &fixture, &library, &cfg, &all_visits, &stream, &cache, &owner, &mut graphs, &mut uploads,
            &[(0, 64, 4)], true,
        )?;
        assert_eq!(c2, 1, "48-visit decode step must record exactly ONE monolithic graph");
        assert_eq!(l2, 1, "captured monolithic graph must launch once per step");
        assert_eq!(u2, 0);
        assert_eq!((av2, fv2), (48, 48));
        assert!(
            !captured_body.chars().any(|c| matches!(c, 'C' | 'H' | 'D' | 'Z')),
            "monolithic graph capture body must be kernel-only (no memcpy/memset); saw events: {captured_body:?}"
        );

        // 3. Replay at DIFFERENT prompt length (prompt_len=257, same rows=4, taps=true):
        //    cache HIT (0 captures, 0 updates, 1 launch, 48-visit closure not re-run).
        let (c3, l3, u3, av3, fv3, _) = run_step(
            &fixture, &library, &cfg, &all_visits, &stream, &cache, &owner, &mut graphs, &mut uploads,
            &[(0, 257, 4)], true,
        )?;
        assert_eq!((c3, l3, u3, av3, fv3), (0, 1, 0, 0, 0), "graph key must be invariant to prompt length");

        // 4. Replay across two sequences with very different prompt lengths (16_384 and 512, total rows=4):
        //    still a cache HIT (no per-request lengths in cache key).
        let (c4, l4, u4, av4, fv4, _) = run_step(
            &fixture, &library, &cfg, &all_visits, &stream, &cache, &owner, &mut graphs, &mut uploads,
            &[(0, 16_384, 3), (1, 512, 1)], true,
        )?;
        assert_eq!((c4, l4, u4, av4, fv4), (0, 1, 0, 0, 0), "graph key must carry no per-request prompt lengths");

        // 5. Different batch row count (rows=8) or different tap flag (taps=false) does NOT alias.
        let (c5, l5, _, av5, _, _) = run_step(
            &fixture, &library, &cfg, &all_visits, &stream, &cache, &owner, &mut graphs, &mut uploads,
            &[(0, 257, 8)], true,
        )?;
        assert_eq!((c5, l5, av5), (0, 0, 48), "distinct row count must not alias cached rows=4 graph");
        let (c6, l6, _, av6, _, _) = run_step(
            &fixture, &library, &cfg, &all_visits, &stream, &cache, &owner, &mut graphs, &mut uploads,
            &[(0, 257, 4)], false,
        )?;
        assert_eq!((c6, l6, av6), (0, 0, 48), "distinct taps flag must not alias cached taps=true graph");

        graphs.clear_drained();
        Ok(())
    }
}
