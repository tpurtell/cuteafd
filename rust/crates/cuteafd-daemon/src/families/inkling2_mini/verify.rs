//! Accepted-prefix 48-slot FIR verification owners for Inkling 2 Mini (width 6656).
use crate::shared::{
    memory::{
        device::{Allocation, Stream},
        Tag,
    },
    qwen::MarkLayout,
};
use anyhow::{ensure, Context, Result};
use cuteafd_ffi::{
    fir_history::{Commit, Layout},
    programs::Programs,
    CuteafdDeviceBuffer as Buffer,
};
use std::{
    cell::{Cell, RefCell},
    rc::Rc,
    sync::atomic::{AtomicU64, Ordering},
};

static NEXT_OWNER: AtomicU64 = AtomicU64::new(1);

pub(crate) struct Mark<'a> {
    owner: u64,
    generation: u64,
    checkpoint: Rc<Allocation<'a>>,
    pub slots: Vec<(usize, usize)>,
}

pub(crate) struct TargetMark<'a> {
    pub conv: Mark<'a>,
    pub placements: Vec<super::cache::Placement>,
}

#[derive(Clone)]
pub(crate) struct Submitted {
    pub slot: usize,
    pub start: usize,
    pub first: usize,
    pub rows: usize,
}

pub(crate) struct Verify<'a> {
    pub checkpoint: Rc<Allocation<'a>>,
    pub replay: Allocation<'a>,
    table: Allocation<'a>,
    commit: Commit<'a>,
    commit_128: Option<Commit<'a>>,
    layout: Layout,
    mark_layout: MarkLayout,
    stream: Rc<Stream<'a>>,
    decode_rows: usize,
    owner: u64,
    generation: Cell<u64>,
    pending: RefCell<Option<Vec<Submitted>>>,
}

pub(crate) fn select_commit_capacity(total_rows: usize) -> usize {
    cuteafd_loader::families::inkling2_mini::execution::replay_capacity(total_rows)
}

impl<'a> Verify<'a> {
    pub fn new(
        programs: &'a Programs<'a>,
        stream: Rc<Stream<'a>>,
        slots: usize,
        width: usize,
        decode_rows: usize,
    ) -> Result<Self> {
        let replay_rows = cuteafd_loader::families::inkling2_mini::execution::replay_capacity(decode_rows);
        let layout = Layout::contiguous_rows(slots, width, replay_rows)?;
        let commit = Commit::load_capacity(programs, layout, 64)?;
        let commit_128 = if replay_rows == 128 {
            Some(Commit::load_capacity(programs, layout, 128)?)
        } else {
            None
        };
        let mark_layout = MarkLayout::new(&[(stream.device.id, layout.history_bytes()?)])?;
        let checkpoint = Rc::new(Allocation::new(
            stream.device,
            layout.history_bytes()?,
            Tag::kv("inkling2_mini/verify-conv-checkpoint"),
        )?);
        let replay = Allocation::new(
            stream.device,
            layout.replay_bytes()?,
            Tag::workspace("inkling2_mini/verify-conv-replay"),
        )?;
        let table = Allocation::new(
            stream.device,
            replay_rows * 12,
            Tag::workspace("inkling2_mini/verify-conv-table"),
        )?;
        let owner = NEXT_OWNER
            .try_update(Ordering::Relaxed, Ordering::Relaxed, |id| id.checked_add(1))
            .map_err(|_| anyhow::anyhow!("Inkling 2 Mini verify owner identity exhausted"))?;
        Ok(Self {
            checkpoint,
            replay,
            table,
            commit,
            commit_128,
            layout,
            mark_layout,
            stream,
            decode_rows,
            owner,
            generation: Cell::new(0),
            pending: Default::default(),
        })
    }

    pub fn checkpoint(&self, live: Buffer, slots: &[(usize, usize)]) -> Result<Mark<'a>> {
        ensure!(
            self.pending.borrow().is_none() && !slots.is_empty(),
            "Inkling 2 Mini verify requires committed state"
        );
        let mut unique = std::collections::BTreeSet::new();
        ensure!(
            slots.iter().all(|&(slot, _)| slot < self.layout.slots && unique.insert(slot)),
            "Inkling 2 Mini verify slot aliases or exceeds admission"
        );
        let generation = self.generation.get().checked_add(1).context("Mini verify generation exhausted")?;
        self.generation.set(generation);
        ensure!(
            live.bytes == self.layout.history_bytes()? && self.mark_layout.bytes == live.bytes,
            "Inkling 2 Mini verify live FIR backing differs from admitted mark"
        );
        // SAFETY: disjoint 48-slot live FIR backing and checkpoint on same stream.
        unsafe {
            crate::shared::memory::device::copy_device_ordered(
                self.stream.device.library,
                self.checkpoint.buffer,
                live,
                live.bytes,
                self.stream.raw,
            )?;
        }
        Ok(Mark {
            owner: self.owner,
            generation,
            checkpoint: Rc::clone(&self.checkpoint),
            slots: slots.to_vec(),
        })
    }

    fn validate(&self, mark: &Mark<'_>) -> Result<()> {
        ensure!(
            mark.owner == self.owner
                && mark.generation == self.generation.get()
                && Rc::ptr_eq(&mark.checkpoint, &self.checkpoint),
            "stale or foreign Inkling 2 Mini verify mark"
        );
        Ok(())
    }

    pub fn submitted(&self, mark: &Mark<'_>, rows: Vec<Submitted>) -> Result<()> {
        self.validate(mark)?;
        ensure!(
            self.pending.borrow().is_none() && rows.len() == mark.slots.len(),
            "Mini pending verify inventory differs"
        );
        let mut total = 0;
        for (row, &(slot, start)) in rows.iter().zip(&mark.slots) {
            ensure!(
                row.slot == slot && row.start == start && row.first == total && (1..=self.decode_rows).contains(&row.rows),
                "Mini verified receipt differs from marked slots"
            );
            total += row.rows;
        }
        ensure!(total <= self.decode_rows, "Mini verified rows exceed bounded replay");
        *self.pending.borrow_mut() = Some(rows);
        Ok(())
    }

    pub fn commit(
        &self,
        mark: &Mark<'_>,
        live: Buffer,
        accepted: &[crate::shared::speculation::target::AcceptedPrefix],
    ) -> Result<Vec<(usize, usize)>> {
        self.validate(mark)?;
        let pending = self.pending.borrow();
        let submitted = pending.as_ref().context("Mini accepted commit lacks successful verify")?;
        ensure!(accepted.len() == submitted.len(), "Mini accepted slot count differs");
        let mut entries = Vec::with_capacity(accepted.len());
        let mut available = Vec::with_capacity(accepted.len());
        let mut frontiers = Vec::new();
        for row in submitted {
            let keep = accepted
                .iter()
                .find(|e| e.slot == row.slot)
                .context("Mini accepted slot omitted")?
                .rows;
            ensure!(
                accepted.iter().filter(|e| e.slot == row.slot).count() == 1,
                "Mini accepted slot repeated"
            );
            entries.push((row.slot, row.first, keep));
            available.push((row.first, row.rows));
            frontiers.push((row.slot, row.start + keep));
        }
        let total_rows = submitted.iter().map(|r| r.first + r.rows).max().unwrap_or(0);
        let commit = if select_commit_capacity(total_rows) == 128 {
            self.commit_128.as_ref().context("Mini 128-row FIR commit not bound")?
        } else {
            &self.commit
        };
        let table = commit.table(&entries, &available)?;
        let bytes = table.iter().flat_map(|word| word.to_le_bytes()).collect::<Vec<_>>();
        // SAFETY: one table upload + single FIR gather across all 48 slots.
        unsafe {
            self.stream
                .device
                .library
                .copy_h2d_ordered(self.table.buffer, &bytes, self.stream.raw)?;
            commit.launch(
                self.checkpoint.buffer,
                self.replay.buffer,
                self.table.buffer,
                live,
                accepted.len(),
                self.stream.raw,
            )?;
        }
        drop(pending);
        self.pending.borrow_mut().take();
        self.generation
            .set(self.generation.get().checked_add(1).context("verify generation overflow")?);
        Ok(frontiers)
    }

    pub fn restore(&self, mark: &Mark<'_>, live: Buffer) -> Result<Vec<(usize, usize)>> {
        self.validate(mark)?;
        // SAFETY: restores checkpointed 48-slot FIR backing on retained stream.
        unsafe {
            crate::shared::memory::device::copy_device_ordered(
                self.stream.device.library,
                live,
                self.checkpoint.buffer,
                live.bytes,
                self.stream.raw,
            )?;
        }
        self.pending.borrow_mut().take();
        Ok(mark.slots.clone())
    }
}

impl Drop for Verify<'_> {
    fn drop(&mut self) {
        self.stream.drain_or_abort("Inkling 2 Mini FIR verify retirement");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn runtime_commit_capacity_selects_64_at_8_rows_and_128_at_80_rows() {
        assert_eq!(select_commit_capacity(1), 64);
        assert_eq!(select_commit_capacity(8), 64);
        assert_eq!(select_commit_capacity(64), 64);
        assert_eq!(select_commit_capacity(65), 128);
        assert_eq!(select_commit_capacity(80), 128);
        assert_eq!(select_commit_capacity(128), 128);
    }
}
