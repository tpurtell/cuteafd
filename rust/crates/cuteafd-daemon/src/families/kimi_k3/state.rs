//! Exact TP2 KDA state owners. Replay is private to each independently queued
//! lane; active recurrent slots and immutable prefix marks are shared by lanes.
use crate::shared::{memory::device::{Allocation, Device, Stream}, prefix::view};
use anyhow::{ensure, Context, Result};
use cuteafd_core::memory_layout::{Basis, Category};
use cuteafd_ffi::{CuteafdDeviceBuffer, KimiKda};
use cuteafd_loader::{families::kimi_k3::config::{KimiAttention, KimiK3Config}, placement::Demand};
use std::rc::Rc;

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
        ensure!((1..=64).contains(&slots) && marks > 0 && (1..=2).contains(&lanes),
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

    /// One contiguous pool for each native commit operand; only replay/tables
    /// are lane-local. The commit ABI strides state by slots and replay by 64 rows.
    fn extents(&self) -> Result<[usize; 4]> {
        Ok([self.state.checked_mul(self.layers).context("Kimi KDA state pool overflow")?,
            self.conv.checked_mul(self.layers).context("Kimi KDA convolution pool overflow")?,
            self.replay.checked_mul(self.layers).context("Kimi KDA replay pool overflow")?,
            self.mark.checked_mul(self.marks).context("Kimi KDA mark arena overflow")?])
    }

    pub(crate) fn demands(&self, rank: u8) -> Result<[Demand; 4]> {
        ensure!(rank < 2, "Kimi KDA rank outside TP2");
        let [state, conv, replay, marks] = self.extents()?;
        let state = state.checked_add(conv).context("Kimi KDA active state overflow")?;
        let replay = replay.checked_mul(self.lanes).context("Kimi KDA lane replay overflow")?;
        Ok([(Category::Kv, "Kimi KDA active state", state),
            (Category::Workspace, "Kimi KDA replay", replay),
            (Category::Prefix, "Kimi exact KDA marks", marks),
            (Category::Workspace, "Kimi KDA commit tables", self.lanes * self.slots * 3 * 4)]
            .map(|(category, group, bytes)| Demand::new(rank, category, group, bytes as u64, Basis::Exact)))
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
        let marks = allocate(marks, "prefix/kimi KDA marks")?;
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
    marks: Rc<Allocation<'a>>,
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
        Ok([view(self.marks.buffer, state, state_bytes)?, view(self.marks.buffer, conv, conv_bytes)?])
    }

    pub(crate) fn commit_operands(&self, lane: usize) -> Result<[CuteafdDeviceBuffer; 4]> {
        ensure!(lane < self.layout.lanes, "Kimi KDA commit lane outside admission");
        Ok([self.state.buffer, self.conv.buffer, self.replay[lane].buffer, self.tables[lane].buffer])
    }

    pub(crate) fn mark_owner(&self) -> Rc<Allocation<'a>> { self.marks.clone() }

    pub(crate) fn owners(&self) -> Vec<Rc<Allocation<'a>>> {
        [&self.state, &self.conv, &self.marks].into_iter().chain(self.replay.iter()).chain(self.tables.iter()).cloned().collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
            state: allocation()?, conv: allocation()?, marks: allocation()?,
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
        for (slots, marks, lanes) in [(0, 1, 1), (65, 1, 1), (1, 0, 1), (1, 1, 0), (1, 1, 3), (1, usize::MAX, 1)] {
            assert!(KdaLayout::new(&cfg, slots, marks, lanes).is_err());
        }
        Ok(())
    }
}
