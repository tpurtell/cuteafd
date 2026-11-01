//! Exact TP2 KDA state owners. Replay is private to each independently queued
//! lane; active recurrent slots and immutable prefix marks are shared by lanes.
use crate::shared::{memory::device::{Allocation, Device, Stream}, prefix::view};
use anyhow::{ensure, Context, Result};

use cuteafd_ffi::{CuteafdDeviceBuffer, KimiKda};
use cuteafd_loader::{families::kimi_k3::config::{KimiAttention, KimiK3Config}, placement::Demand};
use std::rc::Rc;

/// A replay belongs to exactly one verified batch until accept or discard.
/// Placement.len is scheduler-owned and may already have advanced at accept.
pub(crate) struct Provisional {
    pub sequences: Vec<(super::prefix::Placement, Vec<u32>, usize)>,
}

pub(crate) struct Accepted<'p> {
    pub placement: &'p super::prefix::Placement,
    pub start: usize,
    pub first_row: usize,
    pub rows: &'p [u32],
    pub kept: Option<usize>,
}

impl Provisional {
    pub(crate) fn new(sequences: &[(&mut super::prefix::Placement, &[u32])], slots: usize) -> Result<Self> {
        ensure!(!sequences.is_empty(), "Kimi provisional batch is empty");
        let mut first = 0usize;
        let mut seen = std::collections::BTreeSet::new();
        let sequences = sequences.iter().map(|(placement, tokens)| {
            ensure!(placement.slot < slots && seen.insert(placement.slot)
                && placement.len == placement.state_len && !tokens.is_empty(),
                "Kimi provisional slot/frontier is invalid or repeated");
            let start = first;
            first = first.checked_add(tokens.len()).context("Kimi provisional rows overflow")?;
            ensure!(first <= 64 && placement.len.checked_add(tokens.len()).is_some(),
                "Kimi provisional batch exceeds native replay rows");
            Ok(((**placement).clone(), tokens.to_vec(), start))
        }).collect::<Result<Vec<_>>>()?;
        Ok(Self { sequences })
    }

    /// Cancellation is request-owned: never clear another slot's replay.
    pub(crate) fn cancel(&mut self, placement: &super::prefix::Placement) -> Result<()> {
        if let Some(index) = self.sequences.iter().position(|(original, _, _)| original.slot == placement.slot) {
            let (original, rows, _) = &self.sequences[index];
            ensure!(placement.units == original.units && placement.state_len == original.state_len
                && (original.len..=original.len + rows.len()).contains(&placement.len),
                "Kimi cancel identity/frontier differs from provisional replay");
            self.sequences.remove(index);
        }
        Ok(())
    }

    /// Validate every selection before returning the native SoA commit table.
    /// None means explicit discard, not a zero-row replay or an emitted bonus.
    pub(crate) fn tables(&self, accepted: &[Accepted<'_>]) -> Result<Vec<i32>> {
        ensure!(accepted.len() == self.sequences.len(), "Kimi accept batch differs from replay");
        let mut selected = Vec::new();
        for ((original, rows, first), accepted) in self.sequences.iter().zip(accepted) {
            let placement = accepted.placement;
            ensure!(placement.slot == original.slot && placement.units == original.units
                && placement.state_len == original.state_len && accepted.start == original.len
                && accepted.first_row == *first && accepted.rows == rows,
                "Kimi accept identity/frontier differs from provisional replay");
            if let Some(keep) = accepted.kept {
                ensure!((1..=rows.len()).contains(&keep) && placement.len == original.len + keep,
                    "Kimi accept must commit exactly the kept input rows");
                selected.push((placement.slot as i32, *first as i32, keep as i32));
            } else {
                ensure!((original.len..=original.len + rows.len()).contains(&placement.len),
                    "Kimi discarded placement is outside verified input rows");
            }
        }
        Ok((0..3).flat_map(|column| selected.iter().map(move |&(slot, first, keep)|
            [slot, first, keep][column])).collect())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct KdaLayout {
    ordinals: Vec<Option<usize>>,
    layers: usize,
    slots: usize,
    marks: usize,
    lanes: usize,
    state: usize,
    conv: usize,
    replay: usize,
    mark: usize,
}

impl KdaLayout {
    pub(crate) fn new(cfg: &KimiK3Config, slots: usize, marks: usize, lanes: usize) -> Result<Self> {
        ensure!(cfg.layers == cfg.attention.len() && cfg.kda_heads == 96
            && cfg.kda_head_dim == 128 && cfg.kda_conv == 4,
            "Kimi KDA state layout differs from native geometry");
        ensure!((1..=cuteafd_core::concurrency::state_slots(cuteafd_core::concurrency::MAX_CONCURRENCY) as usize).contains(&slots) && (1..=2).contains(&lanes),
            "Kimi KDA state slot/mark/lane geometry");
        let mut layers = 0;
        let ordinals = cfg.attention.iter().map(|kind| if *kind == KimiAttention::Kda {
            let ordinal = layers;
            layers += 1;
            Some(ordinal)
        } else { None }).collect();
        ensure!((1..=69).contains(&layers), "Kimi KDA state layer count");
        let state = KimiKda::state_bytes(slots)?;
        let conv = KimiKda::conv_state_bytes(slots)?;
        let replay = KimiKda::replay_bytes();
        let mark = KimiKda::state_bytes(1)?.checked_add(KimiKda::conv_state_bytes(1)?)
            .and_then(|bytes| bytes.checked_mul(layers)).context("Kimi KDA mark extent overflow")?;
        let layout = Self { ordinals, layers, slots, marks, lanes, state, conv, replay, mark };
        layout.extents()?;
        Ok(layout)
    }

    pub(crate) fn mark_bytes(&self) -> usize { self.mark }

    /// One contiguous pool for each native commit operand; only replay/tables
    /// are lane-local. The commit ABI strides state by slots and replay by 64 rows.
    fn extents(&self) -> Result<[usize; 4]> {
        Ok([self.state.checked_mul(self.layers).context("Kimi KDA state pool overflow")?,
            self.conv.checked_mul(self.layers).context("Kimi KDA convolution pool overflow")?,
            self.replay.checked_mul(self.layers).context("Kimi KDA replay pool overflow")?,
            self.mark.checked_mul(self.marks).context("Kimi KDA mark arena overflow")?])
    }

    pub(crate) fn demands(&self, rank: u8) -> Result<[Demand; 4]> {
        let [state, conv, replay, marks] = self.extents()?;
        let demands = cuteafd_loader::families::kimi_k3::allocation::kda_demands_for_layers(
            self.layers, self.slots, self.marks, self.lanes, rank)?;
        ensure!(demands.iter().map(|d| d.bytes).collect::<Vec<_>>() ==
            vec![(state + conv) as u64, (replay * self.lanes) as u64, marks as u64,
                (self.lanes * self.slots * 3 * 4) as u64], "Kimi state owner differs from shared admission");
        Ok(demands)
    }

    fn lane_inputs(&self, lane: usize, replay: &[CuteafdDeviceBuffer], tables: &[CuteafdDeviceBuffer])
        -> Result<[CuteafdDeviceBuffer; 2]> {
        ensure!(lane < self.lanes && replay.len() == self.lanes && tables.len() == self.lanes,
            "Kimi KDA lane inputs differ from admission");
        let required = self.extents()?[2];
        let buffers = replay.iter().chain(tables).copied().collect::<Vec<_>>();
        for (index, buffer) in buffers.iter().enumerate() {
            let bytes = if index < self.lanes { required } else { self.slots * 3 * 4 };
            ensure!(!buffer.ptr.is_null() && buffer.bytes == bytes && buffer.device_id == buffers[0].device_id,
                "Kimi KDA lane input extent/device");
            let start = buffer.ptr as usize;
            let end = start.checked_add(bytes).context("Kimi KDA lane input address overflow")?;
            for previous in &buffers[..index] {
                let previous_start = previous.ptr as usize;
                let previous_end = previous_start.checked_add(previous.bytes).context("Kimi KDA lane input address overflow")?;
                ensure!(start >= previous_end || previous_start >= end, "Kimi KDA replay/commit inputs alias");
            }
        }
        Ok([replay[lane], tables[lane]])
    }

    fn ordinal(&self, layer: usize) -> Result<usize> {
        self.ordinals.get(layer).copied().flatten().context("Kimi layer has no KDA state")
    }

    fn slot_ranges(&self, layer: usize, slot: usize) -> Result<[(usize, usize); 2]> {
        ensure!(slot < self.slots, "Kimi KDA state slot outside admission");
        let record = self.ordinal(layer)? * self.slots + slot;
        let state = KimiKda::state_bytes(1)?;
        let conv = KimiKda::conv_state_bytes(1)?;
        Ok([(record * state, state), (record * conv, conv)])
    }

    fn mark_ranges(&self, layer: usize, mark: usize) -> Result<[(usize, usize); 2]> {
        ensure!(mark < self.marks, "Kimi KDA prefix mark outside admission");
        let state = KimiKda::state_bytes(1)?;
        let conv = KimiKda::conv_state_bytes(1)?;
        let offset = mark * self.mark + self.ordinal(layer)? * (state + conv);
        Ok([(offset, state), (offset + state, conv)])
    }

    pub(super) fn allocate<'a>(&self, device: Device<'a>, readers: Vec<Rc<Stream<'a>>>) -> Result<KdaBank<'a>> {
        ensure!(readers.len() == self.lanes && readers.iter().all(|stream|
            stream.device.id == device.id && std::ptr::eq(stream.device.library, device.library)),
            "Kimi KDA readers differ from admitted lanes/device");
        ensure!(readers.iter().enumerate().all(|(i, stream)| readers[..i].iter()
            .all(|previous| !Rc::ptr_eq(previous, stream))), "Kimi KDA lanes share a stream owner");
        let stream = Stream::new(device)?;
        let [state, conv, replay, marks] = self.extents()?;
        let allocate = |bytes, category| -> Result<Rc<Allocation<'a>>> {
            let _scope = cuteafd_ffi::memory_ledger::scope(category);
            let allocation = Rc::new(Allocation::new(device, bytes)?);
            ensure!(allocation.buffer.device_id == device.id && allocation.buffer.bytes == bytes,
                "Kimi KDA allocation differs from admitted device/extent");
            Ok(allocation)
        };
        let state = allocate(state, "kv/kimi KDA state")?;
        let conv = allocate(conv, "kv/kimi KDA convolution")?;
        let marks = (marks > 0).then(|| allocate(marks, "prefix/kimi KDA marks")).transpose()?;
        let replay = (0..self.lanes).map(|_| allocate(replay, "workspace/kimi KDA replay"))
            .collect::<Result<Vec<_>>>()?;
        let tables = (0..self.lanes).map(|_| allocate(self.slots * 3 * 4, "workspace/kimi KDA commit tables"))
            .collect::<Result<Vec<_>>>()?;
        // Keep the initialized bank intact on error: its stream drains first.
        let bank = KdaBank { stream, readers, state, conv, marks, replay, tables, layout: self.clone() };
        let replays = bank.replay.iter().map(|allocation| allocation.buffer).collect::<Vec<_>>();
        let tables = bank.tables.iter().map(|allocation| allocation.buffer).collect::<Vec<_>>();
        self.lane_inputs(0, &replays, &tables)?;
        device.run(|| {
            for allocation in [&bank.state, &bank.conv] {
                // SAFETY: this bank retains every destination until its stream drains.
                unsafe { device.library.cuda_zero_bytes_async(allocation.buffer, allocation.buffer.bytes, bank.stream.raw)?; }
            }
            Ok(())
        })?;
        bank.stream.drain()?;
        Ok(bank)
    }
}

/// Retain every compute stream and drain its readers before releasing any bank
/// storage, regardless of whether the enclosing engine drops bank or lanes first.
pub(crate) struct KdaBank<'a> {
    pub stream: Stream<'a>,
    readers: Vec<Rc<Stream<'a>>>,
    state: Rc<Allocation<'a>>,
    conv: Rc<Allocation<'a>>,
    marks: Option<Rc<Allocation<'a>>>,
    replay: Vec<Rc<Allocation<'a>>>,
    tables: Vec<Rc<Allocation<'a>>>,
    layout: KdaLayout,
}

impl Drop for KdaBank<'_> {
    fn drop(&mut self) {
        for stream in &self.readers { stream.drain_or_abort("Kimi KDA lane reader"); }
        self.stream.drain_or_abort("Kimi KDA prefix copies");
    }
}

impl<'a> KdaBank<'a> {
    pub(super) fn slots(&self) -> usize { self.layout.slots }
    pub(super) fn marks(&self) -> usize { self.layout.marks }
    pub(super) fn mark_bytes(&self) -> usize { self.layout.mark }
    pub(super) fn readers(&self) -> &[Rc<Stream<'a>>] { &self.readers }

    pub(super) fn check_reader(&self, lane: usize, stream: &Rc<Stream<'a>>) -> Result<()> {
        ensure!(self.readers.get(lane).is_some_and(|reader| Rc::ptr_eq(reader, stream))
            && self.stream.device.id == stream.device.id
            && std::ptr::eq(self.stream.device.library, stream.device.library),
            "Kimi KDA bank differs from admitted lane stream");
        Ok(())
    }

    pub(crate) fn layer(&self, layer: usize, lane: usize) -> Result<[CuteafdDeviceBuffer; 3]> {
        ensure!(lane < self.layout.lanes, "Kimi KDA replay lane outside admission");
        let ordinal = self.layout.ordinal(layer)?;
        Ok([view(self.state.buffer, ordinal * self.layout.state, self.layout.state)?,
            view(self.conv.buffer, ordinal * self.layout.conv, self.layout.conv)?,
            view(self.replay[lane].buffer, ordinal * self.layout.replay, self.layout.replay)?])
    }

    pub(crate) fn slot(&self, layer: usize, slot: usize) -> Result<[CuteafdDeviceBuffer; 2]> {
        let [(state, state_bytes), (conv, conv_bytes)] = self.layout.slot_ranges(layer, slot)?;
        Ok([view(self.state.buffer, state, state_bytes)?, view(self.conv.buffer, conv, conv_bytes)?])
    }

    pub(crate) fn mark(&self, layer: usize, mark: usize) -> Result<[CuteafdDeviceBuffer; 2]> {
        let [(state, state_bytes), (conv, conv_bytes)] = self.layout.mark_ranges(layer, mark)?;
        let marks = self.marks.as_ref().context("Kimi prefix mark storage is disabled")?;
        Ok([view(marks.buffer, state, state_bytes)?, view(marks.buffer, conv, conv_bytes)?])
    }

    pub(crate) fn commit_operands(&self, lane: usize) -> Result<[CuteafdDeviceBuffer; 4]> {
        ensure!(lane < self.layout.lanes, "Kimi KDA commit lane outside admission");
        Ok([self.state.buffer, self.conv.buffer, self.replay[lane].buffer, self.tables[lane].buffer])
    }

    pub(crate) fn mark_owner(&self) -> Option<Rc<Allocation<'a>>> { self.marks.clone() }

    pub(crate) fn owners(&self) -> Vec<Rc<Allocation<'a>>> {
        [&self.state, &self.conv].into_iter().chain(self.marks.iter()).chain(self.replay.iter()).chain(self.tables.iter()).cloned().collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn provisional_fixture() -> Result<(Provisional, Vec<super::super::prefix::Placement>, Vec<Vec<u32>>)> {
        let mut placements = vec![
            super::super::prefix::Placement { units: vec![1, 2], slot: 1, len: 65, state_len: 65 },
            super::super::prefix::Placement { units: vec![3], slot: 0, len: 9, state_len: 9 },
        ];
        let rows = vec![vec![10, 11, 12, 13], vec![20, 21, 22]];
        let mut sequences = placements.iter_mut().zip(&rows).map(|(p, t)| (p, t.as_slice())).collect::<Vec<_>>();
        let pending = Provisional::new(&mut sequences, 2)?;
        drop(sequences);
        Ok((pending, placements, rows))
    }

    #[test]
    fn provisional_cancellation_checks_identity_and_retains_other_requests() -> Result<()> {
        let (mut pending, placements, _) = provisional_fixture()?;
        let mut stale = placements[1].clone();
        stale.units = vec![99];
        assert!(pending.cancel(&stale).is_err());
        assert_eq!(pending.sequences.len(), 2);
        pending.cancel(&placements[0])?;
        assert_eq!(pending.sequences.len(), 1);
        assert_eq!(pending.sequences[0].0.slot, placements[1].slot);
        assert_eq!(pending.sequences[0].2, 4);
        pending.cancel(&placements[0])?;
        assert_eq!(pending.sequences.len(), 1);
        pending.cancel(&placements[1])?;
        assert!(pending.sequences.is_empty());
        Ok(())
    }

    #[test]
    fn provisional_accept_uses_verified_start_and_exact_input_soa_not_bonus() -> Result<()> {
        let (pending, mut placements, rows) = provisional_fixture()?;
        for first in 1..=4 {
            for second in 1..=3 {
                placements[0].len = 65 + first;
                placements[1].len = 9 + second;
                let accepted = [Accepted { placement: &placements[0], start: 65, first_row: 0,
                    rows: &rows[0], kept: Some(first) }, Accepted { placement: &placements[1], start: 9,
                    first_row: 4, rows: &rows[1], kept: Some(second) }];
                assert_eq!(pending.tables(&accepted)?, vec![1, 0, 0, 4, first as i32, second as i32]);
                // Provisional verification never publishes either recurrent frontier.
                assert_eq!(placements[0].state_len, 65);
                assert_eq!(placements[1].state_len, 9);
            }
        }
        placements[0].len = 69;
        placements[1].len = 12;
        for keep in [None, Some(2)] {
            placements[1].len = if keep.is_some() { 11 } else { 12 };
            let accepted = [Accepted { placement: &placements[0], start: 65, first_row: 0,
                rows: &rows[0], kept: None }, Accepted { placement: &placements[1], start: 9,
                first_row: 4, rows: &rows[1], kept: keep }];
            assert_eq!(pending.tables(&accepted)?, if keep.is_some() { vec![0, 4, 2] } else { vec![] });
        }
        Ok(())
    }

    #[test]
    fn provisional_accept_rejects_bad_later_member_before_any_commit_table() -> Result<()> {
        let (pending, placements, rows) = provisional_fixture()?;
        for bad in 0..8 {
            let mut placements = placements.clone();
            placements[0].len = 67;
            placements[1].len = 11;
            let mut start = 9;
            let mut first = 4;
            let mut tokens = rows[1].clone();
            let mut kept = Some(2);
            match bad {
                0 => placements[1].slot = 1,
                1 => placements[1].units = vec![8],
                2 => placements[1].state_len = 11,
                3 => start = 11,
                4 => first = 0,
                5 => tokens[1] = 99,
                6 => kept = Some(0),
                7 => kept = Some(4),
                _ => unreachable!(),
            }
            let accepted = [Accepted { placement: &placements[0], start: 65, first_row: 0,
                rows: &rows[0], kept: Some(2) }, Accepted { placement: &placements[1], start,
                first_row: first, rows: &tokens, kept }];
            assert!(pending.tables(&accepted).is_err(), "accepted bad member {bad}");
        }
        assert!(pending.tables(&[]).is_err());
        let mut placements = placements.clone();
        placements[1].slot = placements[0].slot;
        let sequences = placements.iter_mut().zip(&rows).map(|(p, t)| (p, t.as_slice())).collect::<Vec<_>>();
        assert!(Provisional::new(&sequences, 2).is_err());
        let mut placement = placements[0].clone();
        assert!(Provisional::new(&[(&mut placement, &[1; 65])], 2).is_err());
        assert!(Provisional::new(&[], 2).is_err());
        Ok(())
    }

    #[test]
    #[ignore = "requires CUTEAFD_NATIVE_LIFETIME_FIXTURE_DIR; CPU-only fake DSO"]
    fn bank_destruction_drains_every_retained_lane_before_releasing_storage() -> Result<()> {
        use cuteafd_ffi::native_library_lifetime_fixture::Fixture;
        let fixture = Fixture::build()?;
        let library = fixture.load()?;
        let device = Device { library: &library, id: 1 };
        let reader = || Rc::new(Stream { device, raw: std::ptr::null_mut() });
        let readers = vec![reader(), reader()];
        let cfg = KimiK3Config::from_hf(&cuteafd_loader::plan::testing::kimi_k3_config())?;
        let allocation = || Allocation::new(device, 16).map(Rc::new);
        let bank = KdaBank { stream: Stream { device, raw: std::ptr::null_mut() }, readers: readers.clone(),
            state: allocation()?, conv: allocation()?, marks: Some(allocation()?),
            replay: vec![allocation()?, allocation()?], tables: vec![allocation()?, allocation()?],
            layout: KdaLayout::new(&cfg, 1, 1, 2)? };
        for (lane, stream) in readers.iter().enumerate() {
            bank.check_reader(lane, stream)?;
            assert!(bank.check_reader(1 - lane, stream).is_err());
        }
        assert!(bank.check_reader(2, &readers[0]).is_err());
        let impostor = reader();
        assert!(bank.check_reader(0, &impostor).is_err());
        drop(impostor);
        let bank = Rc::new(bank);
        let registered = Rc::downgrade(&bank);
        let retained = bank.clone();
        drop(bank);
        assert!(registered.upgrade().is_some());
        let before = fixture.events()?;
        assert!(!before.contains('d'));
        drop(readers);
        drop(retained);
        assert!(registered.upgrade().is_none());
        let events = fixture.events()?;
        let first_free = events.find('d').context("Kimi KDA fixture released no storage")?;
        assert_eq!(events[before.len()..first_free].matches('S').count(), 6,
            "bank drains both compute readers and copy stream, then stream drops drain again before first free: {events}");
        assert_eq!(events.matches('d').count(), 7);
        Ok(())
    }

    #[test]
    fn independent_lanes_never_alias_same_layer_replay_or_commit_inputs() -> Result<()> {
        let cfg = KimiK3Config::from_hf(&cuteafd_loader::plan::testing::kimi_k3_config())?;
        let layout = KdaLayout::new(&cfg, 2, 2, 2)?;
        let replay_bytes = layout.extents()?[2];
        let buffer = |address, bytes| CuteafdDeviceBuffer { ptr: address as *mut std::ffi::c_void,
            bytes, device_id: 7, flags: 0 };
        let replays = [buffer(0x1000usize, replay_bytes), buffer(0x1000 + replay_bytes, replay_bytes)];
        let base = 0x1000 + 2 * replay_bytes;
        let tables = [buffer(base, 24), buffer(base + 24, 24)];
        for layer in 0..cfg.layers {
            if cfg.attention[layer] != KimiAttention::Kda { continue; }
            let [a, ta] = layout.lane_inputs(0, &replays, &tables)?;
            let [b, tb] = layout.lane_inputs(1, &replays, &tables)?;
            let offset = layout.ordinal(layer)? * layout.replay;
            let a = view(a, offset, layout.replay)?;
            let b = view(b, offset, layout.replay)?;
            assert!((a.ptr as usize + a.bytes) <= b.ptr as usize);
            assert_ne!(ta.ptr, tb.ptr);
        }
        assert!(layout.lane_inputs(0, &[replays[0]; 2], &tables).is_err());
        assert!(layout.lane_inputs(0, &replays, &[tables[0]; 2]).is_err());
        assert!(layout.lane_inputs(2, &replays, &tables).is_err());
        let overlap = [tables[0], buffer(base + 16, 24)];
        assert!(layout.lane_inputs(0, &replays, &overlap).is_err());
        Ok(())
    }

    #[test]
    fn native_commit_pools_and_marks_cover_every_kda_byte_without_overlap() -> Result<()> {
        let cfg = KimiK3Config::from_hf(&cuteafd_loader::plan::testing::kimi_k3_config())?;
        let layout = KdaLayout::new(&cfg, 64, 2, 2)?;
        assert_eq!(layout.layers, 69);
        let [state, conv, replay, marks] = layout.extents()?;
        assert_eq!(state, 69 * KimiKda::state_bytes(64)?);
        assert_eq!(conv, 69 * KimiKda::conv_state_bytes(64)?);
        assert_eq!(replay, 69 * KimiKda::replay_bytes());
        assert_eq!(marks, 2 * 69 * (KimiKda::state_bytes(1)? + KimiKda::conv_state_bytes(1)?));
        let demands = layout.demands(1)?;
        assert_eq!(demands[0].bytes as usize, state + conv);
        assert_eq!(demands[1].bytes as usize, 2 * replay);
        assert_eq!(demands[2].bytes as usize, marks);
        assert_eq!(demands[3].bytes, 2 * 64 * 3 * 4);
        let mut live_ends = [0; 2];
        let mut mark_end = 0;
        for mark in 0..2 {
            for layer in 0..cfg.layers {
                if cfg.attention[layer] == KimiAttention::Mla {
                    assert!(layout.slot_ranges(layer, 0).is_err());
                    assert!(layout.mark_ranges(layer, mark).is_err());
                    continue;
                }
                for (offset, bytes) in layout.mark_ranges(layer, mark)? {
                    assert_eq!(offset, mark_end);
                    mark_end += bytes;
                }
                if mark == 0 {
                    for slot in 0..64 {
                        for (end, (offset, bytes)) in live_ends.iter_mut().zip(layout.slot_ranges(layer, slot)?) {
                            assert_eq!(offset, *end);
                            *end += bytes;
                        }
                    }
                }
            }
        }
        assert_eq!(live_ends, [state, conv]);
        assert_eq!(mark_end, marks);
        assert!(layout.slot_ranges(cfg.layers, 0).is_err());
        assert!(layout.slot_ranges(0, 64).is_err());
        assert!(layout.mark_ranges(0, 2).is_err());
        assert!(layout.demands(2).is_err());
        let no_marks = KdaLayout::new(&cfg, 16, 0, 1)?;
        assert_eq!(no_marks.extents()?[3], 0);
        assert_eq!(no_marks.demands(1)?[2].bytes, 0);
        assert!(no_marks.mark_ranges(0, 0).is_err());
        for (slots, marks, lanes) in [(0, 1, 1), (129, 1, 1), (1, 1, 0), (1, 1, 3), (1, usize::MAX, 1)] {
            assert!(KdaLayout::new(&cfg, slots, marks, lanes).is_err());
        }
        Ok(())
    }
}
