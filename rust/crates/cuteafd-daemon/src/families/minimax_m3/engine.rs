//! Request shapes on shared packing and causal unit execution; no serve loop.
use crate::shared::{prefill_packing::{self, Limits},
    prefill_pipeline::{prefill_units, run_prefill_units, PrefillHooks, Unit, UnitSegment}};
use anyhow::{ensure, Context, Result};
use cuteafd_loader::placement::families::minimax_m3::RequestCapacity;
use std::{collections::BTreeSet, ops::Range};

#[derive(Clone, Copy, Debug)]
pub(crate) struct RequestRows {
    pub slot: usize,
    pub start: usize,
    pub rows: usize,
}

/// Only geometry from admission, not a family-local packing or wave policy.
pub(crate) struct Requests {
    capacity: RequestCapacity,
    context: usize,
    layers: usize,
    waves: usize,
}

impl Requests {
    pub fn new(capacity: RequestCapacity, context: usize, layers: usize, waves: usize) -> Result<Self> {
        ensure!(RequestCapacity::new(u64::from(capacity.decode.rows), u64::from(capacity.prefill.rows))? == capacity,
            "MiniMax request capacity differs from admission");
        ensure!(context > 0 && layers > 0, "MiniMax request geometry must be positive");
        super::admission::serve_admission::validate_prefill(cuteafd_loader::placement::planner::PrefillShape {
            rows: u64::from(capacity.prefill.rows), waves: u64::try_from(waves)?,
        })?;
        Ok(Self { capacity, context, layers, waves })
    }

    fn validate(&self, requests: &[RequestRows]) -> Result<()> {
        ensure!(!requests.is_empty() && requests.len() <= self.capacity.decode.requests as usize,
            "MiniMax request batch exceeds admitted concurrency");
        let mut slots = BTreeSet::new();
        for request in requests {
            ensure!(request.slot < self.capacity.decode.requests as usize && slots.insert(request.slot),
                "MiniMax duplicate or unadmitted request slot");
            ensure!(request.rows > 0 && request.start.checked_add(request.rows)
                .is_some_and(|end| end <= self.context), "MiniMax request exceeds admitted context");
        }
        Ok(())
    }

    pub fn decode_rows(&self, requests: &[RequestRows]) -> Result<usize> {
        self.validate(requests)?;
        ensure!(requests.iter().all(|request| request.rows == 1),
            "MiniMax native decode requires one row per request");
        ensure!(requests.len() <= self.capacity.execution_rows() as usize,
            "MiniMax native decode exceeds admitted execution rows");
        Ok(requests.len())
    }

    /// Multi-token verify is ordinary packed prefill. Both cutting and grouping
    /// use the shared planners; repeated pieces of a request cannot share a unit.
    pub fn verify_units(&self, requests: &[RequestRows]) -> Result<Vec<Unit>> {
        self.validate(requests)?;
        let limits = Limits { rows: self.capacity.prefill.rows as usize,
            chunk_rows: self.capacity.prefill.rows as usize, max_context: self.context, taps: None };
        let mut chunks = Vec::new();
        for (sequence, request) in requests.iter().enumerate() {
            for rows in prefill_units(request.rows, limits.rows, 1, 1)? {
                let start = request.start.checked_add(rows.start).context("MiniMax verify position overflow")?;
                chunks.push((sequence, (start, rows.len())));
            }
        }
        let fits = |range: Range<usize>| {
            let members = &chunks[range];
            members.len() <= self.capacity.prefill.requests as usize
                && members.iter().map(|(sequence, _)| sequence).collect::<BTreeSet<_>>().len() == members.len()
                && prefill_packing::plan(&members.iter().map(|(_, chunk)| *chunk).collect::<Vec<_>>(), &limits).is_ok()
        };
        let mut offset = 0usize;
        prefill_packing::groups(chunks.len(), fits).into_iter().enumerate().map(|(index, range)| {
            let members = &chunks[range];
            let packed = prefill_packing::plan(&members.iter().map(|(_, chunk)| *chunk).collect::<Vec<_>>(), &limits)?;
            let rows = packed.last().context("MiniMax verify unit is empty")?.end_row();
            ensure!(members.len() <= self.capacity.prefill.requests as usize
                && rows <= self.capacity.execution_rows() as usize,
                "MiniMax verify unit exceeds admitted execution capacity");
            let first = offset;
            offset = offset.checked_add(rows).context("MiniMax verify row overflow")?;
            let segments = members.iter().zip(packed).map(|((sequence, _), segment)| UnitSegment {
                sequence: *sequence, rows: segment.first..segment.end_row(),
                tokens: segment.start..segment.start + segment.rows,
                prompt_end: false, snapshot: false, head: true,
            }).collect();
            Ok(Unit { index, sequence: members[0].0, segments, rows: first..offset,
                layers: 0..self.layers, slot: index % self.waves, barrier: false })
        }).collect()
    }

    /// end_unit consumes bounded head tiles before commit frees a slot. The
    /// arithmetic hooks retain all storage and own abort/drain on submission error.
    pub fn verify<H: PrefillHooks>(&self, requests: &[RequestRows], hooks: &mut H) -> Result<()> {
        run_prefill_units(hooks, &self.verify_units(requests)?, self.waves)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::shared::prefill_pipeline::FfnRoute;
    use std::collections::BTreeMap;

    fn requests(concurrency: usize, rows: usize) -> Vec<RequestRows> {
        (0..concurrency).map(|slot| RequestRows { slot, start: 63 + slot, rows }).collect()
    }

    #[test]
    fn native_decode_is_one_token_per_unique_admitted_request() -> Result<()> {
        for concurrency in [1, 16, 64] {
            let schedule = Requests::new(RequestCapacity::new(concurrency, 64)?, 4096, 60, 3)?;
            let mut batch = requests(concurrency as usize, 1);
            assert_eq!(schedule.decode_rows(&batch)?, concurrency as usize);
            batch.push(RequestRows { slot: concurrency as usize, start: 0, rows: 1 });
            assert!(schedule.decode_rows(&batch).is_err());
            batch.pop();
            batch[0].rows = 2;
            assert!(schedule.decode_rows(&batch).is_err());
        }
        Ok(())
    }

    #[test]
    fn verify_uses_shared_units_with_bounded_rows_members_and_causal_positions() -> Result<()> {
        for concurrency in [1, 16, 64] {
            for unit_rows in [64, 128] {
                let capacity = RequestCapacity::new(concurrency, unit_rows)?;
                let schedule = Requests::new(capacity, 4096, 60, 3)?;
                for rows in [1, 17, 65, 257] {
                    let batch = requests(concurrency as usize, rows);
                    let units = schedule.verify_units(&batch)?;
                    let mut next = batch.iter().map(|request| request.start).collect::<Vec<_>>();
                    let mut flattened = 0;
                    for (index, unit) in units.iter().enumerate() {
                        assert_eq!((unit.index, unit.slot, unit.layers.clone()), (index, index % 3, 0..60));
                        assert_eq!(unit.rows.start, flattened);
                        assert!(unit.rows.len() <= unit_rows as usize);
                        assert!(unit.rows.len() <= capacity.execution_rows() as usize);
                        assert!(unit.segments.len() <= capacity.prefill.requests as usize);
                        let mut local = 0;
                        let mut seen = BTreeSet::new();
                        for segment in &unit.segments {
                            assert!(seen.insert(segment.sequence));
                            assert_eq!(segment.rows.start, local);
                            assert_eq!(segment.rows.len(), segment.tokens.len());
                            assert_eq!(segment.tokens.start, next[segment.sequence]);
                            assert!(segment.head && !segment.snapshot && !segment.prompt_end);
                            next[segment.sequence] = segment.tokens.end;
                            local = segment.rows.end;
                        }
                        assert_eq!(local, unit.rows.len());
                        flattened = unit.rows.end;
                    }
                    assert_eq!(flattened, concurrency as usize * rows);
                    assert_eq!(next, batch.iter().map(|request| request.start + rows).collect::<Vec<_>>());
                    if flattened > unit_rows as usize { assert!(units.len() > 1); }
                }
            }
        }
        Ok(())
    }

    #[test]
    fn draft_depth_three_keeps_twenty_and_sixty_four_live_rows_unpadded() -> Result<()> {
        for concurrency in [5, 16] {
            let schedule = Requests::new(RequestCapacity::new(concurrency, 64)?, 4096, 60, 3)?;
            let units = schedule.verify_units(&requests(concurrency as usize, 4))?;
            assert_eq!(units.len(), 1);
            assert_eq!(units[0].rows.len(), concurrency as usize * 4);
            assert_eq!(units[0].segments.len(), concurrency as usize);
            assert!(units[0].segments.iter().all(|segment| segment.rows.len() == 4));
        }
        Ok(())
    }

    #[test]
    fn duplicate_slots_and_invalid_geometry_fail_before_submission() -> Result<()> {
        let capacity = RequestCapacity::new(64, 64)?;
        let schedule = Requests::new(capacity, 4096, 60, 3)?;
        let mut batch = requests(2, 1);
        batch[1].slot = batch[0].slot;
        assert!(schedule.decode_rows(&batch).is_err());
        assert!(schedule.verify_units(&batch).is_err());
        batch[1].slot = 64;
        assert!(schedule.verify_units(&batch).is_err());
        batch[1].slot = 1;
        batch[1].start = usize::MAX;
        assert!(schedule.verify_units(&batch).is_err());
        assert!(schedule.verify_units(&[]).is_err());
        assert!(Requests::new(capacity, 0, 60, 3).is_err());
        assert!(Requests::new(capacity, 4096, 60, 0).is_err());
        assert!(Requests::new(RequestCapacity::new(16, 32)?, 4096, 60, 3).is_err());
        let mut invalid = capacity;
        invalid.prefill.requests = 1;
        assert!(Requests::new(invalid, 4096, 60, 3).is_err());
        Ok(())
    }

    #[derive(Default)]
    struct Hooks {
        live: BTreeMap<usize, usize>,
        heads: BTreeSet<usize>,
        committed: Vec<usize>,
        attended: BTreeMap<(usize, usize), usize>,
    }
    impl PrefillHooks for Hooks {
        type Ticket = usize;
        fn route(&self, layer: usize) -> FfnRoute {
            if layer == 1 { FfnRoute::Spark } else { FfnRoute::Local }
        }
        fn begin_unit(&mut self, unit: &Unit) -> Result<()> {
            assert!(self.live.insert(unit.slot, unit.index).is_none(), "slot reused before commit");
            Ok(())
        }
        fn attention(&mut self, layer: usize, unit: &Unit) -> Result<()> {
            for segment in &unit.segments {
                let next = self.attended.entry((segment.sequence, layer)).or_insert(segment.tokens.start);
                assert_eq!(*next, segment.tokens.start, "later causal unit attended first");
                *next = segment.tokens.end;
            }
            Ok(())
        }
        fn local_ffn(&mut self, _: usize, _: &Unit) -> Result<()> { Ok(()) }
        fn send_experts(&mut self, _: usize, unit: &Unit) -> Result<usize> { Ok(unit.index) }
        fn land_experts(&mut self, _: usize, unit: &Unit, ticket: usize) -> Result<()> {
            assert_eq!(unit.index, ticket);
            Ok(())
        }
        fn end_unit(&mut self, unit: &Unit) -> Result<()> {
            assert!(self.heads.insert(unit.index), "head consumed twice");
            Ok(())
        }
        fn commit_unit(&mut self, unit: &Unit) -> Result<()> {
            assert!(self.heads.contains(&unit.index), "slot freed before head consumption");
            assert_eq!(self.live.remove(&unit.slot), Some(unit.index));
            self.committed.push(unit.index);
            Ok(())
        }
    }

    #[test]
    fn shared_executor_preserves_causality_and_consumes_heads_before_slot_reuse() -> Result<()> {
        let schedule = Requests::new(RequestCapacity::new(16, 64)?, 4096, 4, 3)?;
        let batch = requests(16, 257);
        let units = schedule.verify_units(&batch)?;
        let mut hooks = Hooks::default();
        schedule.verify(&batch, &mut hooks)?;
        assert!(hooks.live.is_empty());
        assert_eq!(hooks.committed, (0..units.len()).collect::<Vec<_>>());
        assert_eq!(hooks.heads.len(), units.len());
        for (sequence, request) in batch.iter().enumerate() {
            for layer in 0..4 { assert_eq!(hooks.attended[&(sequence, layer)], request.start + request.rows); }
        }
        Ok(())
    }
}
