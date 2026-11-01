//! Muse TP2's exchange over the shared two-GPU `PeerExchange`: per
//! [`Channel`], `receive_slots(channel)` round-robin receive slots on each GPU,
//! sized by that channel's widest payload at the admitted rows
//! (`tp2::slot_bytes`), as the schedule-driven interleaving model in this
//! module's tests proves sufficient. Every push on a channel goes to the
//! channel's next slot in order; both GPUs issue the same exchange sequence, so
//! the sender's push ordinal is the receiver's receive ordinal.
use super::device::PeerExchange as RankExchange;
use crate::shared::peer_split::{PeerExchange, RankDevice};
use anyhow::{ensure, Context, Result};
use cuteafd_ffi::{peer_exchange::PeerPlane, CuteafdDeviceBuffer as Buffer, NativeLibrary};
use cuteafd_loader::families::muse_spark::tp2::{receive_slot, receive_slots, slot_bytes, Channel};
use std::cell::Cell;

/// First exchange slot of `channel`: channels in [`Channel::ALL`] order, each
/// taking `receive_slots(channel)` consecutive slots.
pub(crate) fn base(channel: Channel) -> usize {
    Channel::ALL.iter().take_while(|c| **c != channel).map(|&c| receive_slots(c)).sum()
}

/// Per-slot receive bytes of the pair, in slot order. `Partials` is sized only
/// when a layer onboards; otherwise its slots are 16 bytes, keeping every
/// channel's indices (and flags) fixed.
pub(crate) fn slot_sizes(hidden: usize, expert: usize, topk: usize, rows: usize, onboarded: bool) -> Result<Vec<usize>> {
    let mut sizes = Vec::new();
    for &channel in &Channel::ALL {
        let bytes = if channel == Channel::Partials && !onboarded { 16 }
            else { slot_bytes(channel, hidden, expert, topk, rows)? };
        sizes.extend(std::iter::repeat(bytes).take(receive_slots(channel)));
    }
    Ok(sizes)
}

/// The pair's exchange (both ends); each rank uses it through [`RankEnd`].
pub(crate) struct MusePeer<'a> {
    exchange: PeerExchange<'a>,
    sizes: Vec<usize>,
    /// Per rank and channel: pushes sent and events received (round-robin ordinals).
    sent: [[Cell<u64>; 5]; 2],
    received: [[Cell<u64>; 5]; 2],
}

impl<'a> MusePeer<'a> {
    pub(crate) fn new(library: &'a NativeLibrary, ranks: [RankDevice; 2], sizes: Vec<usize>) -> Result<Self> {
        ensure!(sizes.len() == Channel::ALL.iter().map(|&c| receive_slots(c)).sum::<usize>(),
            "Muse TP2 needs every channel's slots");
        let exchange = PeerExchange::new_abortable_sized(library, ranks, &sizes)?;
        Ok(Self { exchange, sizes, sent: Default::default(), received: Default::default() })
    }

    pub(crate) fn rank(&self, rank: usize) -> RankEnd<'_, 'a> { RankEnd { peer: self, rank } }

    pub(crate) fn exchange(&self) -> &PeerExchange<'a> { &self.exchange }

    /// The next slot of `channel` for `rank`'s `counter`, advancing it.
    fn next(counter: &[[Cell<u64>; 5]; 2], rank: usize, channel: Channel) -> usize {
        let index = Channel::ALL.iter().position(|c| *c == channel).expect("listed channel");
        let ordinal = counter[rank][index].get();
        counter[rank][index].set(ordinal + 1);
        base(channel) + receive_slot(channel, ordinal)
    }
}

/// One rank's end of the pair's exchange.
pub(crate) struct RankEnd<'p, 'a> { peer: &'p MusePeer<'a>, rank: usize }

impl RankEnd<'_, '_> {
    fn view(&self, at: usize, bytes: usize) -> Result<Buffer> {
        ensure!(bytes <= self.peer.sizes[at], "Muse TP2 payload of {bytes} bytes past slot {at}");
        Ok(Buffer { ptr: self.peer.exchange.recv(self.rank, at)?, bytes,
            device_id: self.peer.exchange.device(self.rank), flags: 0 })
    }
}

impl RankExchange for RankEnd<'_, '_> {
    fn swap(&mut self, channel: Channel, local: Buffer) -> Result<Buffer> {
        let to = MusePeer::next(&self.peer.sent, self.rank, channel);
        ensure!(local.bytes <= self.peer.sizes[to], "Muse TP2 {channel:?} swap past its slot");
        self.peer.exchange.push(self.rank, to, local.ptr.cast_const(), local.bytes)?;
        let from = MusePeer::next(&self.peer.received, self.rank, channel);
        self.peer.exchange.wait(self.rank, from)?;
        self.view(from, local.bytes)
    }

    fn send(&mut self, channel: Channel, source: Buffer, rows: usize, row_bytes: usize, pitch: usize) -> Result<()> {
        let to = MusePeer::next(&self.peer.sent, self.rank, channel);
        let bytes = rows.checked_mul(row_bytes).context("Muse TP2 send extent")?;
        ensure!(bytes <= self.peer.sizes[to] && row_bytes <= pitch && row_bytes % 16 == 0 && pitch % 16 == 0,
            "Muse TP2 {channel:?} send geometry");
        if row_bytes == pitch {
            return self.peer.exchange.push(self.rank, to, source.ptr.cast_const(), bytes);
        }
        let destination = self.peer.exchange.recv(1 - self.rank, to)?;
        self.peer.exchange.push_planes(self.rank, to, &[PeerPlane { destination, source: source.ptr.cast_const(),
            rows: rows as u64, row_bytes: row_bytes as u64, destination_pitch: row_bytes as u64,
            source_pitch: pitch as u64 }])
    }

    fn receive(&mut self, channel: Channel, bytes: usize) -> Result<Buffer> {
        let from = MusePeer::next(&self.peer.received, self.rank, channel);
        self.peer.exchange.wait(self.rank, from)?;
        self.view(from, bytes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn channel_slots_sized_by_their_payload() -> Result<()> {
        let (core, half) = (2752 * 8192 * 2, 2752 * 4096);
        let sizes = slot_sizes(8192, 4096, 8, 2752, false)?;
        assert_eq!(sizes, [core, core, half, 16, half, core, core]);
        crate::shared::peer_split::validate_slot_sizes([0, 1], &sizes)?;
        let onboarded = slot_sizes(8192, 4096, 8, 2752, true)?;
        assert_eq!(onboarded[base(Channel::Partials)], 2752 * 8 * 4096 * 2);
        assert_eq!(Channel::ALL.map(base), [0, 2, 3, 4, 5]);
        // Round-robin ordinals within each channel's slots.
        assert_eq!((0..4).map(|n| base(Channel::Moe) + receive_slot(Channel::Moe, n)).collect::<Vec<_>>(), [5, 6, 5, 6]);
        Ok(())
    }

    /// The exchanges each pipeline hook issues on both GPUs, in stream order
    /// (`device.rs`): attention swaps its o_proj core; a Spark layer's send
    /// gathers the expert-input half to GPU 0 and its land pushes the combined
    /// half to GPU 1 then swaps the post-expert core; an onboarded (local)
    /// layer swaps the expert input, the partials and the post-expert core.
    fn hook_exchanges(step: crate::shared::prefill_pipeline::Step) -> Vec<(Channel, [bool; 2], [bool; 2])> {
        use crate::shared::prefill_pipeline::Step;
        // (channel, senders, receivers) by GPU.
        const SWAP: ([bool; 2], [bool; 2]) = ([true, true], [true, true]);
        match step {
            Step::Attention { .. } => vec![(Channel::Attention, SWAP.0, SWAP.1)],
            Step::Send { .. } => vec![(Channel::ExpertInput, [false, true], [true, false])],
            Step::Land { .. } => vec![(Channel::Combined, [true, false], [false, true]), (Channel::Moe, SWAP.0, SWAP.1)],
            Step::Local { .. } => vec![(Channel::ExpertInput, SWAP.0, SWAP.1), (Channel::Partials, SWAP.0, SWAP.1),
                (Channel::Moe, SWAP.0, SWAP.1)],
            _ => vec![],
        }
    }

    /// The shared prefill scheduler's real step order (`prefill_schedule`: units
    /// interleave per layer, waves land later) drives the exchange model: every
    /// interleaving of the two GPU streams, per channel the most received but
    /// unread events. `receive_slots` must equal it. Prefill calls of 1..6 units
    /// at 1..3 waves in flight, every onboarding pattern of 3 layers, and a
    /// decode pass inserted at every step boundary (decode lanes inside a call).
    #[test]
    fn receive_slots_cover_the_shared_prefill_schedule() -> Result<()> {
        use crate::shared::prefill_pipeline::{prefill_schedule, sequence_units, FfnRoute, Step};
        use std::collections::{BTreeMap, HashSet, VecDeque};
        #[derive(Clone, Copy, PartialEq, Eq, Hash)]
        enum Op { Push(u32), Wait(u32), Read(u32) }
        let mut worst = BTreeMap::<Channel, usize>::new();
        for waves in 1..=3 {
            for units in 1..=6usize {
                for onboarded in [vec![], vec![0], vec![1], vec![2], vec![0, 2], vec![0, 1, 2]] {
                    let routes: Vec<FfnRoute> = (0..3).map(|l| if onboarded.contains(&l) { FfnRoute::Local }
                        else { FfnRoute::Spark }).collect();
                    let call = sequence_units(units * 1024, 1024, waves, 3)?;
                    let steps = prefill_schedule(&routes, &call, waves)?;
                    // A decode step (one layer-major pass of every layer, on the same
                    // streams) at each step boundary in turn, and none: covers decode
                    // lanes that may run inside a prefill call (#717 direction).
                    let decode: Vec<Step> = (0..3).flat_map(|layer| [Step::Attention { layer, unit: usize::MAX },
                        if routes[layer] == FfnRoute::Spark { Step::Send { layer, unit: usize::MAX } }
                        else { Step::Local { layer, unit: usize::MAX } }]
                        .into_iter().chain((routes[layer] == FfnRoute::Spark)
                            .then_some(Step::Land { layer, unit: usize::MAX }))).collect();
                    for insert in std::iter::once(None).chain((0..=steps.len()).map(Some)) {
                    let mut walk = steps.clone();
                    if let Some(at) = insert { walk.splice(at..at, decode.iter().copied()); }
                    let mut channel_of = Vec::new();
                    let mut programs = [Vec::new(), Vec::new()];
                    for step in walk {
                        for (channel, send, receive) in hook_exchanges(step) {
                            let e = channel_of.len() as u32;
                            channel_of.push(channel);
                            for g in 0..2 {
                                if send[g] { programs[g].push(Op::Push(e)); }
                                if receive[g] { programs[g].extend([Op::Wait(e), Op::Read(e)]); }
                            }
                        }
                    }
                    // Positions of every push and read per GPU (u32::MAX: absent), precomputed.
                    let events = channel_of.len();
                    let mut push = [vec![usize::MAX; events], vec![usize::MAX; events]];
                    let mut read = [vec![usize::MAX; events], vec![usize::MAX; events]];
                    for g in 0..2 {
                        for (pc, op) in programs[g].iter().enumerate() {
                            match op { Op::Push(e) => push[g][*e as usize] = pc, Op::Read(e) => read[g][*e as usize] = pc,
                                Op::Wait(_) => {} }
                        }
                    }
                    let mut seen = HashSet::new();
                    let mut queue = VecDeque::from([[0usize, 0usize]]);
                    while let Some(pcs) = queue.pop_front() {
                        if !seen.insert(pcs) { continue; }
                        for g in 0..2 {
                            let mut unread = [0usize; 5];
                            for e in 0..events {
                                if push[1 - g][e] != usize::MAX && pcs[1 - g] > push[1 - g][e]
                                    && read[g][e] != usize::MAX && pcs[g] <= read[g][e] {
                                    unread[Channel::ALL.iter().position(|c| *c == channel_of[e]).unwrap()] += 1;
                                }
                            }
                            for (i, channel) in Channel::ALL.iter().enumerate() {
                                let entry = worst.entry(*channel).or_insert(0);
                                *entry = (*entry).max(unread[i]);
                            }
                        }
                        for g in 0..2 {
                            let Some(&op) = programs[g].get(pcs[g]) else { continue };
                            let runnable = match op {
                                Op::Wait(e) => push[1 - g][e as usize] != usize::MAX && pcs[1 - g] > push[1 - g][e as usize],
                                _ => true,
                            };
                            if runnable { let mut next = pcs; next[g] += 1; queue.push_back(next); }
                        }
                    }
                    }
                }
            }
        }
        eprintln!("muse TP2 schedule bounds: {worst:?}");
        for channel in Channel::ALL {
            assert_eq!(worst[&channel], receive_slots(channel), "{channel:?} slots differ from the real schedule's bound");
        }
        Ok(())
    }


    /// The runtime's slots are what admission charged: the fixed channels as
    /// "Muse TP2 exchange slots", `Partials` as the peer half of `tp2_workspace`.
    #[test]
    fn runtime_slots_equal_the_admitted_exchange() -> Result<()> {
        use cuteafd_loader::placement::families::muse_spark::{tp2_exchange_bytes, tp2_partial_bytes};
        let cfg = cuteafd_loader::families::muse_spark::MuseSparkConfig::from_hf(
            &cuteafd_loader::plan::testing::muse_spark_config())?;
        for unit in [64, 2752, 4096] {
            let fixed = slot_sizes(cfg.hidden, cfg.expert_hidden, cfg.topk, unit, false)?;
            assert_eq!(fixed.iter().sum::<usize>() - 16, tp2_exchange_bytes(&cfg, unit)?);
            let onboarded = slot_sizes(cfg.hidden, cfg.expert_hidden, cfg.topk, unit, true)?;
            assert_eq!(2 * onboarded[base(Channel::Partials)], tp2_partial_bytes(&cfg, unit));
        }
        Ok(())
    }
}
