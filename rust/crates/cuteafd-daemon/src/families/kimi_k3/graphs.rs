//! Exact startup-only rank-local arithmetic graphs. Transport and event joins
//! remain in the shared layer executor, outside these captured segments.
use super::block::MixSite;
use crate::shared::{decode_graph::{GraphBank, GraphDecision, GraphOwner, GraphPolicy},
    memory::device::Stream};
use anyhow::{ensure, Result};
use cuteafd_loader::{families::kimi_k3::config::{KimiAttention, KimiK3Config},
    placement::{GraphSet, Lifetime, inventory::StartupGraphs}};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) enum Op { Residual(MixSite), Attention { layer: usize, sequences: usize }, Ffn(usize) }

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) struct Key { pub lane: usize, pub rank: usize, pub rows: usize, pub op: Op }

pub(crate) const RESERVE: &str = "Kimi startup graph reserve";

pub(crate) fn demand(set: &GraphSet, rank: usize) -> cuteafd_loader::placement::Demand {
    let mut demand = set.demand(rank);
    demand.group = RESERVE.into();
    demand.basis = cuteafd_core::memory_layout::Basis::Estimated;
    demand
}

pub(crate) const ROWS: [usize; 5] = [1, 4, 16, 64, 4096];

/// An estimated reserve until the first real serving census qualifies its bytes.
/// The exact same key set is charged by admission and sealed by each binding.
pub(crate) fn inventory(cfg: &KimiK3Config, lanes: usize, capacity: usize, slots: usize,
    resident: impl Fn(usize, usize) -> bool) -> Result<StartupGraphs<Key>> {
    ensure!((1..=2).contains(&lanes) && (1..=4096).contains(&capacity) && (1..=64).contains(&slots), "Kimi graph lane/row capacity");
    let mut keys = Vec::new();
    let mut counts = [0u64; 2];
    for lane in 0..lanes {
        for rank in 0..2 {
            for rows in ROWS.into_iter().filter(|&rows| rows <= capacity) {
                for layer in 0..cfg.layers {
                    keys.push(Key { lane, rank, rows, op: Op::Residual(MixSite::Attention(layer)) });
                    keys.push(Key { lane, rank, rows, op: Op::Residual(MixSite::Ffn(layer)) });
                    keys.push(Key { lane, rank, rows, op: Op::Ffn(layer) });
                    if resident(rank, layer) {
                        keys.push(Key { lane, rank, rows, op: Op::Attention { layer, sequences: 1 } });
                        if cfg.attention[layer] == KimiAttention::Kda && (2..=slots).contains(&rows) {
                            keys.push(Key { lane, rank, rows, op: Op::Attention { layer, sequences: rows } });
                        }
                    }
                }
                if rank == 1 { keys.push(Key { lane, rank, rows, op: Op::Residual(MixSite::Output) }); }
            }
        }
    }
    for key in &keys { counts[key.rank] += 1; }
    GraphSet::new(&counts, 256 << 10, 10, 64 << 20, keys.len() as u64, Lifetime::Startup)
        .startup(keys).map_err(anyhow::Error::msg)
}

/// Use the solver's alternating selected-layer helper, not layer parity: KDA
/// gaps must not alter the MLA owner's ordinal.
pub(crate) fn owned_inventory(cfg: &KimiK3Config, lanes: usize, rows: usize, slots: usize)
    -> Result<StartupGraphs<Key>> {
    let owners = cuteafd_loader::placement::attention::alternating_owners(
        cfg.attention.iter().map(|kind| (*kind == KimiAttention::Mla, None)), 0)?;
    inventory(cfg, lanes, rows, slots, |rank, layer|
        cfg.attention[layer] == KimiAttention::Kda || rank == usize::from(owners[layer]))
}

pub(crate) fn check_placement(cfg: &KimiK3Config, placement: &cuteafd_loader::placement::Placement,
    keys: &[Key]) -> Result<()> {
    use cuteafd_loader::placement::LayerMode;
    ensure!(placement.layers.len() == cfg.layers, "Kimi graph placement layer count");
    for key in keys {
        if let Op::Attention { layer, .. } = key.op {
            let mode = placement.layers.get(layer).ok_or_else(|| anyhow::anyhow!("Kimi graph layer"))?.mode;
            ensure!(matches!(mode, LayerMode::HeadSplit)
                || matches!(mode, LayerMode::Whole { gpu, .. } if usize::from(gpu) == key.rank),
                "Kimi attention graph is absent from accepted placement");
        }
    }
    for (layer, assignment) in placement.layers.iter().enumerate() {
        for rank in 0..2 {
            let resident = matches!(assignment.mode, LayerMode::HeadSplit)
                || matches!(assignment.mode, LayerMode::Whole { gpu, .. } if usize::from(gpu) == rank);
            ensure!(keys.iter().any(|key| key.rank == rank && matches!(key.op,
                Op::Attention { layer: captured, .. } if captured == layer)) == resident,
                "Kimi graph inventory differs from accepted attention residency");
        }
    }
    Ok(())
}

pub(crate) struct Graphs<'a> { bank: GraphBank<Key, GraphOwner<'a, ()>> }

impl<'a> Graphs<'a> {
    pub(crate) fn capture(stream: &Stream<'a>, keys: &[Key],
        mut arithmetic: impl FnMut(Key) -> Result<()>) -> Result<Self> {
        let mut bank = GraphBank::with_policy(GraphPolicy::FixedStartup);
        bank.warm(keys, |&key| stream.device.run(|| {
            // SAFETY: startup is exclusive; only rank-local arithmetic enters
            // capture. The containing LocalLane pins every operand and drains
            // its exact stream before releasing this bank or its allocations.
            unsafe { stream.device.library.cuda_graph_begin_capture(stream.raw)?; }
            let submitted = arithmetic(key);
            // End even a failed submission, so startup cannot leave capture active.
            // SAFETY: this ends the matching capture on the same retained stream.
            let ended = unsafe { stream.device.library.cuda_graph_end_capture(stream.raw) };
            match (submitted, ended) {
                (Ok(()), Ok(raw)) => {
                    // SAFETY: LocalLane retains all captured pointers and drains before drop.
                    let owner = unsafe { GraphOwner::new(stream.device.library, stream.device.id, raw, ())? };
                    Ok((owner, None))
                }
                (Err(error), Ok(raw)) => {
                    // SAFETY: the failed startup executable was never launched.
                    unsafe { stream.device.library.cuda_graph_exec_destroy(raw)?; }
                    Err(error)
                }
                (Err(error), Err(ended)) => Err(error.context(format!("ending failed capture: {ended:#}"))),
                (Ok(()), Err(error)) => Err(error),
            }
        }))?;
        Ok(Self { bank })
    }

    pub(crate) fn launch(&mut self, key: Key, stream: &Stream<'a>) -> Result<bool> {
        let (decision, owner) = self.bank.enqueue(&key);
        match decision {
            GraphDecision::Replay => {
                let owner = owner.expect("replay has executable");
                ensure!(owner.device() == stream.device.id, "Kimi graph replay device changed");
                stream.device.run(|| {
                    // SAFETY: binding owns immutable capture pointers and this exact stream.
                    unsafe { stream.device.library.cuda_graph_launch(owner.raw, stream.raw) }
                })?;
                Ok(true)
            }
            GraphDecision::Eager => Ok(false),
            GraphDecision::Capture => anyhow::bail!("Kimi live submission attempted graph capture"),
        }
    }
}

pub(crate) fn check_captures(startup: u64, current: u64) -> Result<()> {
    ensure!(current == startup, "Kimi zero-steady-capture contract: startup={startup}, current={current}");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn owned_keys_match_shared_admission_and_reject_wrong_residency() -> Result<()> {
        let cfg = KimiK3Config::from_hf(&cuteafd_loader::plan::testing::kimi_k3_config())?;
        let admitted = super::super::admission::admit(super::super::admission::tests::inputs(&cfg))?;
        let inventory = owned_inventory(&cfg, 1, 4096, 4)?;
        check_placement(&cfg, &admitted.placement, &inventory.keys)?;
        let mut wrong = inventory.keys.clone();
        let layer = cfg.attention.iter().position(|kind| *kind == KimiAttention::Mla).unwrap();
        let key = wrong.iter_mut().find(|key| matches!(key.op,
            Op::Attention { layer: captured, .. } if captured == layer)).unwrap();
        key.rank = 1 - key.rank;
        assert!(check_placement(&cfg, &admitted.placement, &wrong).is_err());
        let missing = inventory.keys.into_iter().filter(|key| !matches!(key.op,
            Op::Attention { layer: captured, .. } if captured == layer)).collect::<Vec<_>>();
        assert!(check_placement(&cfg, &admitted.placement, &missing).is_err());
        Ok(())
    }

    #[test]
    fn startup_inventory_is_exact_and_partition_sensitive() -> Result<()> {
        let cfg = KimiK3Config::from_hf(&cuteafd_loader::plan::testing::kimi_k3_config())?;
        let admitted = inventory(&cfg, 2, 4096, 16, |rank, layer|
            cfg.attention[layer] == KimiAttention::Kda || rank == (layer % 2))?;
        let unique = admitted.keys.iter().collect::<std::collections::HashSet<_>>();
        assert_eq!(unique.len(), admitted.keys.len());
        for rank in 0..2 {
            assert_eq!(admitted.inventory.ranks[rank].executables,
                admitted.keys.iter().filter(|key| key.rank == rank).count() as u64);
            assert!(admitted.inventory.bytes(rank) > 0);
            let reserve = demand(&admitted.inventory, rank);
            assert_eq!(reserve.group, RESERVE);
            assert_eq!(reserve.basis, cuteafd_core::memory_layout::Basis::Estimated);
            assert_eq!(reserve.bytes, admitted.inventory.bytes(rank));
        }
        let layer = cfg.attention.iter().position(|kind| *kind == KimiAttention::Kda).unwrap();
        for sequences in [1, 16] {
            assert!(unique.contains(&Key { lane: 1, rank: 1, rows: 16, op: Op::Attention { layer, sequences } }));
        }
        let small = inventory(&cfg, 1, 4096, 4, |_, _| true)?;
        assert!(!small.keys.iter().any(|key| matches!(key.op,
            Op::Attention { sequences, .. } if sequences > 4)));
        for (lanes, rows, slots) in [(0, 1, 1), (1, 0, 1), (1, 4097, 1), (1, 1, 0), (1, 1, 65)] {
            assert!(inventory(&cfg, lanes, rows, slots, |_, _| true).is_err());
        }
        check_captures(19, 19)?;
        assert!(check_captures(19, 20).is_err());
        Ok(())
    }
}
