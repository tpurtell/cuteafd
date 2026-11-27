//! Shared BlockVerifyTarget adapter for Inkling 2 Mini (48-visit TP1 target + 48-slot FIR commit).
use super::{
    cache::Placement,
    engine::Execution,
    verify::{Submitted, TargetMark},
};
use crate::shared::{
    prefill_pipeline::{Unit, UnitSegment},
    speculation::target::{AcceptedPrefix, BlockVerifyTarget, HiddenTap, VerifyOutput, VerifyRequest},
};
use anyhow::{ensure, Context, Result};

pub(crate) struct Target<'e, 'a> {
    pub execution: &'e mut Execution<'a>,
    pub placements: &'e mut [Placement],
    taps: Vec<HiddenTap>,
}

impl<'e, 'a> Target<'e, 'a> {
    pub fn new(execution: &'e mut Execution<'a>, placements: &'e mut [Placement]) -> Self {
        Self {
            execution,
            placements,
            taps: Vec::new(),
        }
    }
}

impl<'a> BlockVerifyTarget for Target<'_, 'a> {
    type Mark = TargetMark<'a>;

    fn checkpoint(&mut self, slots: &[usize]) -> Result<Self::Mark> {
        self.execution.require_live()?;
        ensure!(
            slots.len() == self.placements.len()
                && slots
                    .iter()
                    .zip(self.placements.iter())
                    .all(|(&slot, p)| slot == p.slot && p.len == p.state_len),
            "Inkling 2 Mini checkpoint differs from committed placements"
        );
        let draft = self.execution.draft.as_ref().context("Mini speculation not admitted")?;
        let starts: Vec<(usize, usize)> = self.placements.iter().map(|p| (p.slot, p.len)).collect();
        let conv = self
            .execution
            .requests
            .decode
            .stream
            .device
            .run(|| draft.verify.checkpoint(self.execution.requests.cache.conv_backing(), &starts))?;
        Ok(TargetMark {
            conv,
            placements: self.placements.to_vec(),
        })
    }

    fn verify<'e>(&'e mut self, mark: &Self::Mark, request: VerifyRequest<'_>) -> Result<VerifyOutput<'e>> {
        request.validate(
            self.execution.cfg.total_visits(),
            self.execution.cfg.text.unpadded_vocab_size,
            &mark.conv.slots,
        )?;
        let draft = self.execution.draft.as_ref().context("Mini drafter missing")?.clone();
        ensure!(
            request.tap_sites == draft.sites
                && request.rows.len() <= self.execution.requests.decode.layout.rows
                && request.rows.iter().all(|row| row.token.is_some()),
            "Inkling 2 Mini verify requires admitted native sites and active packed rows"
        );
        let mut tokens = Vec::<Vec<u32>>::new();
        let mut submitted = Vec::new();
        let mut segments = Vec::new();
        let mut first = 0;
        for (index, p) in mark.placements.iter().enumerate() {
            let rows: Vec<_> = request.rows.iter().filter(|row| row.slot == p.slot).collect();
            ensure!((1..=8).contains(&rows.len()), "Mini verify member exceeds bundled block");
            let source: Vec<u32> = rows.iter().map(|row| row.token.unwrap()).collect();
            ensure!(
                request.rows[first..first + rows.len()].iter().all(|row| row.slot == p.slot),
                "Mini verify members must be contiguous"
            );
            segments.push(UnitSegment {
                sequence: index,
                rows: first..first + rows.len(),
                tokens: p.len..p.len + rows.len(),
                prompt_end: false,
                snapshot: false,
                head: true,
            });
            submitted.push(Submitted {
                slot: p.slot,
                start: p.len,
                first,
                rows: rows.len(),
            });
            first += rows.len();
            tokens.push(source);
        }
        ensure!(first == request.rows.len(), "Mini verify rows include unmarked slots");
        let unit = Unit {
            index: 0,
            sequence: 0,
            rows: 0..first,
            layers: 0..self.execution.cfg.total_visits(),
            slot: 0,
            barrier: false,
            segments,
        };
        self.execution
            .requests
            .decode
            .set_replay(Some((draft.verify.replay.buffer, self.execution.requests.cache.layout.conv.width)))?;
        let sources: Vec<&[u32]> = tokens.iter().map(Vec::as_slice).collect();
        let result = self.execution.verify(self.placements, &sources, &unit);
        self.execution.requests.decode.set_replay(None)?;
        let logits = match result {
            Ok(logits) => logits,
            Err(error) => {
                self.restore(mark)?;
                return Err(error);
            }
        };
        draft.verify.submitted(&mark.conv, submitted)?;
        let output = draft.decode.output.buffer;
        let h = self.execution.cfg.text.hidden;
        self.taps = request
            .tap_sites
            .iter()
            .enumerate()
            .map(|(index, &site)| HiddenTap {
                site,
                ptr: output.ptr.wrapping_byte_add(index * h * 2),
                rows: first,
                hidden: h,
                stride: request.tap_sites.len() * h,
                stream: self.execution.requests.decode.stream.raw,
            })
            .collect();
        Ok(VerifyOutput { logits, taps: &self.taps })
    }

    fn restore(&mut self, mark: &Self::Mark) -> Result<()> {
        self.execution.restore_verify(mark)?;
        for (p, saved) in self.placements.iter_mut().zip(&mark.placements) {
            *p = saved.clone();
        }
        Ok(())
    }

    fn commit(&mut self, mark: &Self::Mark, accepted: &[AcceptedPrefix]) -> Result<()> {
        let draft = self.execution.draft.as_ref().context("Mini drafter missing")?.clone();
        let frontiers = self.execution.requests.decode.stream.device.run(|| {
            draft
                .verify
                .commit(&mark.conv, self.execution.requests.cache.conv_backing(), accepted)
        })?;
        self.execution.requests.decode.stream.drain()?;
        self.execution.require_live()?;
        for (p, saved) in self.placements.iter_mut().zip(&mark.placements) {
            let frontier = frontiers
                .iter()
                .find(|(slot, _)| *slot == p.slot)
                .context("Mini accepted frontier omitted")?
                .1;
            p.len = frontier;
            p.state_len = frontier;
            ensure!(frontier >= saved.len, "Mini accepted prefix precedes mark");
        }
        Ok(())
    }
}
