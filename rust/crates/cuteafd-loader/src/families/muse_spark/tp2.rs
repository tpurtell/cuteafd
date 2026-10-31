//! Muse TP2 (a head-split RTX pair): the exchanges between the two GPUs and
//! their receive slots.
//!
//! Single residency: attention is head-split; the MoE frontend and backend are
//! K6 splits too (pre_expert_proj by output columns, post_expert_proj by input
//! columns over the expert width), so no projection is held twice. Router,
//! norms and residual gates are tiny and whole on both GPUs.
//!
//! Each GPU runs one stream; one host loop enqueues both GPUs in the same order.
//! Per layer:
//!
//! | channel | when | direction | payload per row |
//! |---|---|---|---|
//! | `Attention` | every layer | swap | o_proj BF16 core, H |
//! | `ExpertInput` | Spark layer | GPU 1 to GPU 0 | pre_expert half, E/2 |
//! | `ExpertInput` | onboarded layer | swap | pre_expert half, E/2 (both need all of E) |
//! | `Partials` | onboarded layer | swap | expert partials, top-k x E |
//! | `Combined` | Spark layer | GPU 0 to GPU 1 | combined half, E/2 (GPU 0 holds the link and combines) |
//! | `Moe` | every layer | swap | post_expert_proj BF16 core, H |
//!
//! The centered pre-expert norm and the input-scaled post-expert norm need
//! full rows over E, so the expert input is assembled before the norm and an
//! onboarded layer's combine runs on full partials.
//!
//! Each channel has its own round-robin receive slots on each receiving GPU,
//! sized by its own widest payload; a push may overwrite a slot only once the
//! receiver has read the event that slot held last. [`receive_slots`] is what the
//! exhaustive interleaving model in this module's tests proves sufficient for
//! every schedule the shared prefill scheduler issues (1..3 units in flight,
//! decode steps anywhere, any onboarding).
use anyhow::{ensure, Result};

/// The TP2 exchanges, each with its own receive slots.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Channel { Attention, ExpertInput, Partials, Combined, Moe }

impl Channel {
    pub const ALL: [Channel; 5] = [Channel::Attention, Channel::ExpertInput, Channel::Partials, Channel::Combined,
        Channel::Moe];

    /// The widest payload of one row on this channel.
    pub fn row_bytes(self, hidden: usize, expert: usize, topk: usize) -> usize {
        match self {
            Channel::Attention | Channel::Moe => hidden * 2,
            Channel::ExpertInput | Channel::Combined => expert,
            Channel::Partials => topk * expert * 2,
        }
    }
}

/// Receive slots per GPU of `channel`, round-robin in receive order: the bound
/// the exhaustive interleaving model reaches under the shared prefill
/// scheduler's real step order (the daemon's
/// `peer::tests::receive_slots_cover_the_shared_prefill_schedule`, which
/// asserts these are exact). Units interleave per layer there: after one
/// unit's attention swap a GPU sends its expert-input half one-way and runs on
/// to the next unit's attention push before the peer read the first, so the
/// two swapped cores need two slots; the one-way halves and the partials need one.
/// The model also inserts a decode pass at every step boundary of the call
/// (decode lanes that may run inside a prefill call, #717); the bounds hold.
pub fn receive_slots(channel: Channel) -> usize {
    match channel {
        Channel::Attention | Channel::Moe => 2,
        Channel::ExpertInput | Channel::Partials | Channel::Combined => 1,
    }
}

/// Receive slot of a channel's `ordinal`-th received event.
pub fn receive_slot(channel: Channel, ordinal: u64) -> usize { (ordinal % receive_slots(channel) as u64) as usize }

/// Bytes of each receive slot of `channel` at `rows` rows; `Partials` exists
/// only when a layer onboards.
pub fn slot_bytes(channel: Channel, hidden: usize, expert: usize, topk: usize, rows: usize) -> Result<usize> {
    ensure!(rows > 0, "Muse TP2 slot rows");
    rows.checked_mul(channel.row_bytes(hidden, expert, topk)).ok_or_else(|| anyhow::anyhow!("Muse TP2 slot overflow"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::{HashSet, VecDeque};

    /// One op on a GPU stream; `e` is the exchange ordinal (common to both GPUs).
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    enum Op { Push(u32), Wait(u32), Read(u32) }

    /// Who sends on an exchange.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    enum Way { Swap, OneToZero, ZeroToOne }

    /// One layer pass of one unit, in stream order (the module table).
    fn layer(onboarded: bool) -> Vec<(Channel, Way)> {
        let mut out = vec![(Channel::Attention, Way::Swap)];
        if onboarded {
            out.extend([(Channel::ExpertInput, Way::Swap), (Channel::Partials, Way::Swap)]);
        } else {
            out.extend([(Channel::ExpertInput, Way::OneToZero), (Channel::Combined, Way::ZeroToOne)]);
        }
        out.push((Channel::Moe, Way::Swap));
        out
    }

    /// The host's enqueue order: `units` prefill units in flight, layer-major
    /// over the units, and decode steps (a full pass each) before listed passes.
    fn exchanges(units: usize, layers: usize, onboarded: &[usize], decode_before: &[usize]) -> Vec<(Channel, Way)> {
        let mut out = Vec::new();
        let mut pass = 0;
        for l in 0..layers {
            for _unit in 0..units {
                if decode_before.contains(&pass) {
                    for d in 0..layers { out.extend(layer(onboarded.contains(&d))); }
                }
                out.extend(layer(onboarded.contains(&l)));
                pass += 1;
            }
        }
        out
    }

    /// Each GPU's stream and the channel of every exchange ordinal.
    fn programs(exchanges: &[(Channel, Way)]) -> [Vec<Op>; 2] {
        let mut gpus = [Vec::new(), Vec::new()];
        for (e, (_, way)) in exchanges.iter().enumerate() {
            let e = e as u32;
            let (send, receive): (&[usize], &[usize]) = match way {
                Way::Swap => (&[0, 1], &[0, 1]),
                Way::OneToZero => (&[1], &[0]),
                Way::ZeroToOne => (&[0], &[1]),
            };
            for g in 0..2 {
                if send.contains(&g) { gpus[g].push(Op::Push(e)); }
                if receive.contains(&g) { gpus[g].extend([Op::Wait(e), Op::Read(e)]); }
            }
        }
        gpus
    }

    /// Every interleaving of the two streams (a wait blocks until the peer
    /// pushed that exchange): per channel and GPU, the most events received but
    /// not yet read at any reachable state.
    fn max_unread(exchanges: &[(Channel, Way)]) -> std::collections::BTreeMap<(Channel, usize), usize> {
        let programs = programs(exchanges);
        let at = |g: usize, op: Op| programs[g].iter().position(|o| *o == op);
        let mut seen = HashSet::new();
        let mut queue = VecDeque::from([[0usize, 0usize]]);
        let mut worst = std::collections::BTreeMap::new();
        while let Some(pcs) = queue.pop_front() {
            if !seen.insert(pcs) { continue; }
            for g in 0..2 {
                for channel in Channel::ALL {
                    let unread = programs[1 - g].iter().filter_map(|op| if let Op::Push(e) = op { Some(*e) } else { None })
                        .filter(|&e| exchanges[e as usize].0 == channel)
                        .filter(|&e| pcs[1 - g] > at(1 - g, Op::Push(e)).unwrap()
                            && at(g, Op::Read(e)).is_some_and(|read| pcs[g] <= read)).count();
                    let entry = worst.entry((channel, g)).or_insert(0);
                    *entry = (*entry).max(unread);
                }
            }
            for g in 0..2 {
                let Some(&op) = programs[g].get(pcs[g]) else { continue };
                let runnable = match op {
                    Op::Wait(e) => at(1 - g, Op::Push(e)).is_some_and(|push| pcs[1 - g] > push),
                    _ => true,
                };
                if runnable { let mut next = pcs; next[g] += 1; queue.push_back(next); }
            }
        }
        worst
    }

    /// Every channel's slot count covers the most received-but-unread events of
    /// that channel in every interleaving of layer-major passes (1..3 units,
    /// 3 layers, every onboarding pattern, decode steps before any pass). The
    /// real scheduler interleaves units within a layer and needs more; the
    /// daemon's schedule-driven model sets the counts exactly.
    #[test]
    fn receive_slots_cover_every_interleaving() {
        let mut worst = std::collections::BTreeMap::<Channel, usize>::new();
        for units in 1..=3 {
            for onboarded in [vec![], vec![0], vec![1], vec![2], vec![0, 2], vec![0, 1, 2]] {
                for decode in [vec![], vec![0], vec![1], vec![units], vec![3 * units - 1], vec![0, units, 2 * units]] {
                    for ((channel, gpu), unread) in max_unread(&exchanges(units, 3, &onboarded, &decode)) {
                        assert!(unread <= receive_slots(channel), "units {units} onboarded {onboarded:?} \
                            decode {decode:?}: {channel:?} on GPU {gpu} holds {unread} > {}", receive_slots(channel));
                        let entry = worst.entry(channel).or_insert(0);
                        *entry = (*entry).max(unread);
                    }
                }
            }
        }
        assert!(worst.values().all(|&n| n == 1), "layer-major passes need one slot per channel: {worst:?}");
    }

    /// The model is not vacuous: one-way exchanges alone let the receiver's
    /// unread events grow with the program.
    #[test]
    fn one_way_only_skew_is_unbounded() {
        let unread = max_unread(&[(Channel::Combined, Way::ZeroToOne); 6]);
        assert_eq!(unread[&(Channel::Combined, 1)], 6);
    }

    #[test]
    fn slots_hold_each_channels_widest_payload() -> Result<()> {
        assert_eq!(slot_bytes(Channel::Attention, 8192, 4096, 8, 2752)?, 2752 * 8192 * 2);
        assert_eq!(slot_bytes(Channel::Combined, 8192, 4096, 8, 2752)?, 2752 * 4096);
        assert_eq!(slot_bytes(Channel::Partials, 8192, 4096, 8, 2752)?, 2752 * 8 * 4096 * 2);
        assert!(slot_bytes(Channel::Moe, 8192, 4096, 8, 0).is_err());
        Ok(())
    }
}
