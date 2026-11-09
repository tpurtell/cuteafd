//! Rank-local step graphs. Context steps include same-stream device exchange;
//! ordinary hidden/event joins remain in the shared layer executor.
//!
//! Executables are owned by the shared bounded device cache: pinned geometry is
//! prewarmed once at startup and everything else is captured lazily on the miss
//! that first needs it, or run eagerly when the controller declines. A step
//! never refuses for a missing executable.
use super::block::MixSite;
use crate::shared::{decode_graph::{device::{Basis, Cycle, Preference, Site}, CacheAttached,
        CaptureTicket, GraphBank, GraphOwner, GraphPolicy},
    memory::device::Stream};
use anyhow::{ensure, Context, Result};
use cuteafd_loader::{families::kimi_k3::config::{KimiAttention, KimiK3Config},
    placement::inventory::StartupGraphs,
    serving_capacity::kimi_prewarm::{kimi_bootstrap, kimi_layout, kimi_selected, KimiPrewarmLayout, KimiPrewarmOp}};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) enum Op {
    Residual(MixSite),
    Attention { layer: usize, sequences: usize, speculative: bool },
    Context { layer: usize, speculative: bool },
    Ffn(usize),
}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct Key { pub lane: usize, pub rank: usize, pub rows: usize, pub op: Op }

/// One line per captured key. The shared cache stores this as the trace shape,
/// so `scripts/dev/graph-prewarm-from-trace.py` parses these fields.
impl std::fmt::Debug for Key {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "lane={} rank={} rows={} ", self.lane, self.rank, self.rows)?;
        match self.op {
            Op::Residual(site) => write!(f, "op=residual site={site:?}"),
            Op::Attention { layer, sequences, speculative } => write!(
                f, "op=attention layer={layer} seq={sequences} spec={}", u8::from(speculative)),
            Op::Context { layer, speculative } => write!(
                f, "op=context layer={layer} spec={}", u8::from(speculative)),
            Op::Ffn(layer) => write!(f, "op=ffn layer={layer}"),
        }
    }
}

impl Op {
    /// The speculative (verify) variant. Residual and FFN keys carry no split.
    pub(crate) fn speculative(self) -> bool {
        match self { Self::Attention { speculative, .. } | Self::Context { speculative, .. } => speculative,
            Self::Residual(_) | Self::Ffn(_) => false }
    }
}

/// Scoring and diagnostic decode disable optional arithmetic graphs, not the
/// mandatory Context exchange executable. Context never runs in a row window.
pub(crate) fn replay_allowed(op: Op, optional: bool, window: bool) -> Result<bool> {
    if matches!(op, Op::Context { .. }) {
        ensure!(!window, "Kimi context graph needs its fixed row storage");
        Ok(true)
    } else {
        Ok(optional && !window)
    }
}

pub(crate) use cuteafd_loader::families::kimi_k3::allocation::{GRAPH_RESERVE as RESERVE,
    graph_demand as demand};
use cuteafd_loader::families::kimi_k3::allocation::{graph_rows, verify_graph_rows};

fn geometry(op: Op) -> (KimiPrewarmOp, Option<usize>) {
    match op {
        Op::Residual(_) => (KimiPrewarmOp::Residual, None),
        Op::Ffn(_) => (KimiPrewarmOp::Ffn, None),
        Op::Attention { sequences, .. } => (KimiPrewarmOp::Attention, Some(sequences)),
        Op::Context { .. } => (KimiPrewarmOp::Context, None),
    }
}

pub(crate) fn prewarm_selected(layout: &KimiPrewarmLayout, key: Key, pinned: bool) -> bool {
    let (op, sequences) = geometry(key.op);
    kimi_selected(layout, key.rows, key.op.speculative(), op, sequences, pinned)
}

pub(crate) fn pinned(ranks: usize, key: Key) -> bool {
    prewarm_selected(kimi_layout(ranks), key, true)
}

/// Project observed trace geometries onto the placement's actual sites. No new
/// prefill/window keys or unobserved attention row/sequence pairs are invented.
pub(crate) fn prewarm_keys(cfg: &KimiK3Config, inventory: &[Key], layout: &KimiPrewarmLayout,
    slots: usize, capacity: usize) -> Vec<Key> {
    let mut keys = std::collections::HashSet::new();
    for entry in &layout.entries {
        if !(1..=64.min(capacity)).contains(&entry.rows) { continue }
        for &template in inventory {
            if entry.op.is_none() {
                if prewarm_selected(layout, template, false) { keys.insert(template); }
                continue;
            }
            let op = match (entry.op.unwrap(), template.op) {
                (KimiPrewarmOp::Residual, Op::Residual(site)) => Op::Residual(site),
                (KimiPrewarmOp::Ffn, Op::Ffn(layer)) => Op::Ffn(layer),
                (KimiPrewarmOp::Context, Op::Context { layer, .. }) =>
                    Op::Context { layer, speculative: entry.speculative },
                (KimiPrewarmOp::Attention, Op::Attention { layer, .. }) => {
                    let Some(sequences) = entry.sequences else { continue };
                    if sequences == 0 || sequences > slots || sequences > entry.rows
                        || (cfg.attention[layer] != KimiAttention::Kda && sequences != 1) { continue }
                    Op::Attention { layer, sequences, speculative: entry.speculative }
                }
                _ => continue,
            };
            let key = Key { rows: entry.rows, op, ..template };
            if prewarm_selected(layout, key, false) { keys.insert(key); }
        }
    }
    let mut keys = keys.into_iter().collect::<Vec<_>>();
    keys.sort_by_cached_key(|key| format!("{key:?}"));
    keys
}

/// Every executable the model can need, enumerated from the accepted placement.
/// Admission charges the graph reserve separately; this set enumerates keys and
/// validates placement. `verify` (the drafter's block, 0 without one) and `slots` determine every
/// flattened speculative batch width, including mixed-width multi-request steps.
pub(crate) fn inventory(cfg: &KimiK3Config, lanes: usize, capacity: usize, slots: usize, verify: usize,
    resident: impl Fn(usize, usize) -> bool) -> Result<StartupGraphs<Key>> {
    inventory_for_ranks(cfg, lanes, capacity, slots, verify, 2, resident)
}

pub(crate) fn inventory_for_ranks(cfg: &KimiK3Config, lanes: usize, capacity: usize, slots: usize, verify: usize,
    ranks: usize, resident: impl Fn(usize, usize) -> bool) -> Result<StartupGraphs<Key>> {
    inventory_with_context(cfg, lanes, capacity, slots, verify, ranks, resident, |_| false, None)
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn inventory_with_context(cfg: &KimiK3Config, lanes: usize, capacity: usize, slots: usize,
    verify: usize, ranks: usize, resident: impl Fn(usize, usize) -> bool, context: impl Fn(usize) -> bool,
    decode: Option<&cuteafd_loader::families::kimi_k3::context::DecodeBuckets>)
    -> Result<StartupGraphs<Key>> {
    ensure!(matches!(ranks, 1 | 2) && (1..=2).contains(&lanes) && (1..=4096).contains(&capacity) && (1..=cuteafd_core::concurrency::state_slots(cuteafd_core::concurrency::MAX_CONCURRENCY) as usize).contains(&slots), "Kimi graph lane/row capacity");
    let mut keys = Vec::new();
    for lane in 0..lanes {
        for rank in 0..ranks {
            for rows in graph_rows(verify, capacity) {
                for layer in 0..cfg.layers {
                    keys.push(Key { lane, rank, rows, op: Op::Residual(MixSite::Attention(layer)) });
                    keys.push(Key { lane, rank, rows, op: Op::Residual(MixSite::Ffn(layer)) });
                    keys.push(Key { lane, rank, rows, op: Op::Ffn(layer) });
                    if resident(rank, layer) && !context(layer) {
                        let mut attention = |speculative| {
                            keys.push(Key { lane, rank, rows, op: Op::Attention { layer, sequences: 1, speculative } });
                        };
                        if cuteafd_loader::families::kimi_k3::allocation::GRAPH_ROWS.contains(&rows) {
                            attention(false);
                            if cfg.attention[layer] == KimiAttention::Kda && (2..=slots).contains(&rows) {
                                keys.push(Key { lane, rank, rows, op: Op::Attention { layer, sequences: rows, speculative: false } });
                            }
                        }
                        if verify_graph_rows(verify, 1, capacity).contains(&rows) {
                            // Keep ordinary and speculative keys distinct even
                            // when they share a plain row bucket.
                            keys.push(Key { lane, rank, rows, op: Op::Attention { layer, sequences: 1, speculative: true } });
                        }
                    }
                }
                if rank == ranks - 1 { keys.push(Key { lane, rank, rows, op: Op::Residual(MixSite::Output) }); }
            }
            if let Some(decode) = decode {
                for rows in decode.graph_rows(verify, slots, capacity) {
                    for layer in (0..cfg.layers).filter(|&layer| context(layer) && resident(rank, layer)) {
                        for speculative in [false, true] {
                            if if speculative { verify_graph_rows(verify, slots, capacity).contains(&rows) }
                                else { decode.plain().contains(&rows) } {
                                keys.push(Key { lane, rank, rows, op: Op::Context { layer, speculative } });
                            }
                        }
                    }
                }
            }
        }
    }
    cuteafd_loader::families::kimi_k3::allocation::graphs_with_context(cfg, lanes, capacity, slots, verify, ranks, resident, context, decode)?
        .startup(keys).map_err(anyhow::Error::msg)
}

/// Use the solver's alternating selected-layer helper, not layer parity: KDA
/// gaps must not alter the MLA owner's ordinal.
pub(crate) fn owned_inventory(cfg: &KimiK3Config, lanes: usize, rows: usize, slots: usize, verify: usize)
    -> Result<StartupGraphs<Key>> {
    let owners = cuteafd_loader::placement::attention::alternating_owners(
        cfg.attention.iter().map(|kind| (*kind == KimiAttention::Mla, None)), 0)?;
    inventory(cfg, lanes, rows, slots, verify, |rank, layer|
        cfg.attention[layer] == KimiAttention::Kda || rank == usize::from(owners[layer]))
}

pub(crate) fn check_placement(cfg: &KimiK3Config, placement: &cuteafd_loader::placement::Placement,
    keys: &[Key]) -> Result<()> {
    check_placement_for_ranks(cfg, placement, keys, 2)
}

pub(crate) fn check_placement_for_ranks(cfg: &KimiK3Config, placement: &cuteafd_loader::placement::Placement,
    keys: &[Key], ranks: usize) -> Result<()> {
    use cuteafd_loader::placement::LayerMode;
    ensure!(matches!(ranks, 1 | 2) && keys.iter().all(|key| key.rank < ranks),
        "Kimi graph physical rank count");
    ensure!(placement.layers.len() == cfg.layers, "Kimi graph placement layer count");
    let inventory: std::collections::HashSet<_> = keys.iter().copied().collect();
    for key in keys {
        if let Op::Context { layer, speculative } = key.op {
            ensure!(ranks == 2 && cfg.attention.get(layer) == Some(&KimiAttention::Mla)
                && placement.layers.get(layer).is_some_and(|a| a.mode == LayerMode::ContextSplit)
                && (1..=64).contains(&key.rows) && (!speculative || key.rows >= 2),
                "Kimi context graph differs from accepted placement");
        }
        if let Op::Attention { layer, .. } = key.op {
            let mode = placement.layers.get(layer).ok_or_else(|| anyhow::anyhow!("Kimi graph layer"))?.mode;
            ensure!(matches!(mode, LayerMode::HeadSplit)
                || matches!(mode, LayerMode::Whole { gpu, .. } if usize::from(gpu) == key.rank),
                "Kimi attention graph is absent from accepted placement");
        }
    }
    for (layer, assignment) in placement.layers.iter().enumerate() {
        ensure!(ranks == 2 || matches!(assignment.mode, LayerMode::Whole { gpu: 0, .. }),
            "Kimi single-owner graphs require whole attention residency");
        for rank in 0..ranks {
            let resident = matches!(assignment.mode, LayerMode::HeadSplit | LayerMode::ContextSplit)
                || matches!(assignment.mode, LayerMode::Whole { gpu, .. } if usize::from(gpu) == rank);
            ensure!(keys.iter().any(|key| key.rank == rank && matches!(key.op,
                Op::Attention { layer: captured, .. } | Op::Context { layer: captured, .. } if captured == layer)) == resident,
                "Kimi graph inventory differs from accepted attention residency");
            if assignment.mode == LayerMode::ContextSplit {
                for key in keys.iter().filter(|key| key.rank == rank && matches!(key.op,
                    Op::Context { layer: captured, .. } if captured == layer)) {
                    let Op::Context { speculative, .. } = key.op else { unreachable!() };
                    ensure!(inventory.contains(&Key { rank: 1 - rank,
                        op: Op::Context { layer, speculative }, ..*key }),
                        "Kimi context peer graph inventory is incomplete");
                }
            }
        }
    }
    Ok(())
}

/// One lane's graph bank. Every lane on a physical device attaches to that
/// device's single bounded cache with the same reserve and basis.
pub(crate) struct Graphs<'a> {
    bank: GraphBank<Key, GraphOwner<'a, ()>>,
    replays: u64,
    captured: usize,
    device: i32,
    spans_devices: bool,
}

impl<'a> Graphs<'a> {
    pub(crate) fn new() -> Self {
        Self { bank: GraphBank::with_family(GraphPolicy::Budgeted { bytes: None }, "kimi_k3"),
            replays: 0, captured: 0, device: -1, spans_devices: false }
    }

    /// Attach to `device`'s bounded cache. `bootstrap` is one more capture at the
    /// family's measured per-executable rate, used until the device is measured.
    pub(crate) fn attach(&mut self, device: i32, ranks: usize, rank: usize, reserve: u64) -> Result<()> {
        self.device = device;
        self.spans_devices = ranks > 1;
        self.bank.attach_device(device, reserve, Site { name: "kimi_k3.step",
            cycle: Cycle::Other, preference: Preference::Adaptive,
            bootstrap: kimi_bootstrap(ranks, rank), basis: Basis::Bytes,
            spans_devices: self.spans_devices })
    }

    pub(crate) fn attached(&self) -> bool { self.bank.cache_attached() }

    /// The physical device this lane's executables live on.
    pub(crate) fn device(&self) -> i32 { self.device }
    pub(crate) fn spans_devices(&self) -> bool { self.spans_devices }
    /// Whether any executable is resident; startup-only reconfiguration and
    /// state restore run before the first one.
    pub(crate) fn is_empty(&self) -> bool { self.captured == 0 }

    pub(crate) fn cached(&self, key: &Key) -> bool { self.bank.get(key).is_some() }

    pub(crate) fn cycle(&mut self, cycle: Cycle) { self.bank.cache_cycle(cycle); }
    pub(crate) fn needs_boundary(&self) -> bool { self.bank.cache_needs_boundary() }
    pub(crate) fn tick(&mut self) { self.bank.cache_tick(); }

    /// Control surface for the engine's completed-step measurement and eviction.
    pub(crate) fn owner(&mut self) -> &mut dyn CacheAttached { &mut self.bank }

    /// Drop every executable and this bank's cache records. The caller has
    /// already drained the lane's stream.
    pub(crate) fn clear(&mut self) {
        self.bank.cache_clear_drained();
        self.captured = 0;
    }

    pub(crate) fn replays(&self) -> u64 { self.replays }

    /// Best-effort startup prewarm of the checked-in pinned geometry. A geometry
    /// that can never replay inside a lane window is skipped, and a controller
    /// refusal only leaves that geometry lazy.
    pub(crate) fn prewarm(&mut self, stream: &Stream<'a>, ranks: usize, keys: &[Key],
        mut body: impl FnMut(Key) -> Result<()>) -> Result<usize> {
        let mut captured = 0;
        for &key in keys {
            if !prewarm_selected(kimi_layout(ranks), key, false) || self.cached(&key) { continue }
            let pinned = pinned(ranks, key);
            let Some(ticket) = self.bank.cache_ticket(&key, pinned, None) else { continue };
            if self.capture(stream, key, ticket, &mut || body(key))? { captured += 1; }
        }
        Ok(captured)
    }

    /// Run one step body through the cache: replay a resident executable, else
    /// capture it, else run the body eagerly.
    pub(crate) fn graph(&mut self, stream: &Stream<'a>, key: Key, allow_capture: bool,
        pinned: bool, mut body: impl FnMut() -> Result<()>) -> Result<()> {
        if let Some(owner) = self.bank.cached_launch(&key) {
            let (device, raw) = (owner.device(), owner.raw);
            ensure!(device == stream.device.id, "Kimi graph replay device changed");
            // SAFETY: the bank owns immutable capture pointers and LocalLane
            // retains this exact stream until every launch has drained.
            stream.device.run(|| unsafe { stream.device.library.cuda_graph_launch(raw, stream.raw) })?;
            self.replays += 1;
            return Ok(());
        }
        if !allow_capture { return body() }
        let Some(ticket) = self.bank.cache_ticket(&key, pinned, None) else { return body() };
        // The body is recorded on this lane's stream; on refusal it runs eagerly
        // instead. Its own nested cache lookups are disabled by the caller, so
        // the recorded segment is the eager kernel sequence.
        if !self.capture(stream, key, ticket, &mut body)? { return body() }
        let owner = self.bank.get(&key).context("inserted Kimi graph")?;
        let (device, raw) = (owner.device(), owner.raw);
        ensure!(device == stream.device.id, "Kimi graph captured a foreign device");
        // SAFETY: just instantiated on this device and not yet launched.
        stream.device.run(|| unsafe { stream.device.library.cuda_graph_launch(raw, stream.raw) })?;
        self.replays += 1;
        Ok(())
    }

    /// Capture `body` and own the executable. `false` means the capture was
    /// declined or failed to instantiate: the caller must run the body eagerly.
    /// The ticket cancels every failed path, including by drop.
    fn capture(&mut self, stream: &Stream<'a>, key: Key, ticket: CaptureTicket,
        body: &mut impl FnMut() -> Result<()>) -> Result<bool> {
        // SAFETY: the lane is the only writer on this persistent stream for the
        // duration of the step body; every captured pointer is pinned by LocalLane.
        if let Err(error) = stream.device.run(|| unsafe {
            stream.device.library.cuda_graph_begin_capture(stream.raw) }) {
            tracing::warn!(%error, ?key, "Kimi graph capture unavailable; running eagerly");
            return Ok(false);
        }
        let submitted = body();
        // SAFETY: ends the matching capture on this same retained stream.
        let ended = stream.device.run(|| unsafe {
            stream.device.library.cuda_graph_end_capture_topology(stream.raw) });
        match (submitted, ended) {
            (Ok(()), Ok((raw, nodes, width, spans, inspection_error))) => {
                if let Some(error) = inspection_error {
                    tracing::warn!(%error, ?key, "Kimi graph topology unknown; capture is not chain reusable");
                }
                // SAFETY: LocalLane retains every captured pointer and drains this
                // stream before releasing the owner or its allocations.
                let owner = match unsafe {
                    GraphOwner::new(stream.device.library, stream.device.id, raw, ()) } {
                    Ok(owner) => owner,
                    Err(error) => {
                        // SAFETY: the executable was never launched.
                        unsafe { stream.device.library.cuda_graph_exec_destroy(raw)?; }
                        return Err(error);
                    }
                };
                crate::shared::decode_graph::census::capture_key("kimi_k3", "decode",
                    stream.raw as usize, key.rank as i32, Some(key.lane), &key, owner.raw);
                self.bank.cache_commit(key, owner, ticket, nodes as u64, width,
                    spans || self.spans_devices);
                self.captured += 1;
                Ok(true)
            }
            (Err(error), Ok((raw, ..))) => {
                // SAFETY: the failed capture's executable has never launched.
                unsafe { stream.device.library.cuda_graph_exec_destroy(raw)?; }
                Err(error)
            }
            (Err(error), Err(ended)) => Err(error.context(format!("ending failed Kimi capture: {ended:#}"))),
            (Ok(()), Err(error)) => {
                tracing::warn!(%error, ?key, "Kimi graph instantiation unavailable; running eagerly");
                Ok(false)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_observed_geometry_expands_to_reachable_keys_and_nothing_else() -> Result<()> {
        use cuteafd_loader::serving_capacity::kimi_prewarm::KimiPrewarmEntry;
        let cfg = KimiK3Config::from_hf(&cuteafd_loader::plan::testing::kimi_k3_config())?;
        let layout = KimiPrewarmLayout { ranks: 1, layout: "trace".into(),
            reserve_base_bytes: 1, reserve_extra_bytes: 2 << 20,
            entries: vec![
                KimiPrewarmEntry { rows: 20, speculative: true, pin: true,
                    op: Some(KimiPrewarmOp::Attention), sequences: Some(4) },
                KimiPrewarmEntry { rows: 20, speculative: false, pin: false,
                    op: Some(KimiPrewarmOp::Ffn), sequences: None },
            ] };
        let inventory = inventory_for_ranks(&cfg, 1, 4096, 16, 8, 1, |_, _| true)?;
        let keys = prewarm_keys(&cfg, &inventory.keys, &layout, 16, 4096);
        assert!(!keys.is_empty());
        for entry in &layout.entries {
            assert!(keys.iter().any(|key| key.rows == entry.rows
                && key.op.speculative() == entry.speculative
                && geometry(key.op) == (entry.op.unwrap(), entry.sequences)));
        }
        for key in keys {
            assert!(prewarm_selected(&layout, key, false));
            assert!(replay_allowed(key.op, true, false)?);
            assert!(!replay_allowed(key.op, true, true)?);
            if let Op::Attention { sequences, .. } = key.op { assert_eq!(sequences, 4); }
        }
        assert!(prewarm_keys(&cfg, &inventory.keys, &layout, 2, 4096).iter()
            .all(|key| !matches!(key.op, Op::Attention { .. })));
        Ok(())
    }

    #[test]
    fn the_controller_selects_capture_from_the_reserve_and_runs_eagerly_without_one() {
        use crate::shared::decode_graph::device::Miss;
        let hot = Key { lane: 0, rank: 0, rows: 1, op: Op::Ffn(0) };
        // Distinct synthetic devices: the controller is per physical device.
        let mut graphs = Graphs::new();
        graphs.attach(-901, 1, 0, 8 << 20).unwrap();
        assert!(matches!(graphs.bank.cache_miss(&hot, true, None), Miss::Capture(_)));
        // A ticket is a reservation, not residency.
        assert!(graphs.is_empty() && !graphs.cached(&hot));
        // Zero reserve means every miss runs eagerly.
        let mut eager = Graphs::new();
        eager.attach(-902, 1, 0, 0).unwrap();
        assert!(matches!(eager.bank.cache_miss(&hot, true, None), Miss::Eager));
        // No reserve at all also runs eagerly.
        let mut detached = Graphs::new();
        assert!(matches!(detached.bank.cache_miss(&hot, true, None), Miss::Eager));
        // One physical device has one reserve and basis.
        let mut same = Graphs::new();
        assert!(same.attach(-901, 1, 0, 8 << 20).is_ok());
        let mut other = Graphs::new();
        assert!(other.attach(-901, 1, 0, 16 << 20).is_err());
    }

    #[test]
    fn context_decode_miss_enqueues_the_same_protocol_as_startup() {
        let source = include_str!("device.rs");
        let step = source.split("    fn context_step(").nth(1).unwrap()
            .split("    pub(crate) fn context_decode(").next().unwrap();
        assert!(!step.contains("self.graphs"));
        assert!(step.contains("crate::shared::peer_split::attention::context_decode("));
        let decode = source.split("    pub(crate) fn context_decode(").nth(1).unwrap()
            .split("    pub(crate) fn context_attention(").next().unwrap();
        assert!(decode.contains("self.graph_step(bucket, super::graphs::Op::Context { layer, speculative },\n            |lane| lane.context_step(layer, bucket, speculative))?;"));
        assert!(!decode.contains("cuda_graph_begin_capture"));
    }

    #[test]
    fn context_whole_steps_match_admission_and_capture_each_native_row_bucket() -> Result<()> {
        use cuteafd_loader::placement::LayerMode;
        let cfg = KimiK3Config::from_hf(&cuteafd_loader::plan::testing::kimi_k3_config())?;
        let inputs = super::super::admission::tests::inputs(&cfg);
        let mut placement = super::super::admission::admit(inputs)?.placement;
        for (kind, assignment) in cfg.attention.iter().zip(&mut placement.layers) {
            if *kind == KimiAttention::Mla { assignment.mode = LayerMode::ContextSplit; }
        }
        let decode = cuteafd_loader::families::kimi_k3::context::DecodeBuckets::from_manifest(
            &serde_json::json!({"programs":[{"name":"dense_trellis_k6_trellis_mla_q_a_mcg",
                "geometry":{"bits":6,"decode_route_thresholds":[8,16,32,48]}}]}))?;
        for slots in [1, 4, 16] { for capacity in [64, 2752, 4096] { for verify in [0, 8, 64] {
            let set = inventory_with_context(&cfg, 1, capacity, slots, verify, 2, |_, _| true,
                |layer| cfg.attention[layer] == KimiAttention::Mla, Some(&decode))?;
            let unique = set.keys.iter().collect::<std::collections::HashSet<_>>();
            assert_eq!(unique.len(), set.keys.len());
            check_placement(&cfg, &placement, &set.keys)?;
            for rank in 0..2 {
                assert_eq!(set.inventory.ranks[rank].executables,
                    set.keys.iter().filter(|key| key.rank == rank).count() as u64);
            }
            let layer = cfg.attention.iter().position(|kind| *kind == KimiAttention::Mla).unwrap();
            assert!(unique.contains(&Key { lane: 0, rank: 0, rows: 16,
                op: Op::Context { layer, speculative: false } }));
            assert!(!set.keys.iter().any(|key| matches!(key.op,
                Op::Attention { layer, .. } if cfg.attention[layer] == KimiAttention::Mla)));
            for rank in 0..2 {
                let plain = set.keys.iter().filter(|key| key.rank == rank && matches!(key.op,
                    Op::Context { layer: captured, speculative: false, .. } if captured == layer)).count();
                assert_eq!(plain, 7);
                for rows in [8, 32, 48] {
                    assert!(!set.keys.iter().any(|key| key.rank == rank && key.rows == rows
                        && matches!(key.op, Op::Ffn(_) | Op::Residual(_)))
                        || verify_graph_rows(verify, 1, capacity).contains(&rows));
                }
            }
            assert!(!set.keys.iter().any(|key| key.rows > 64 && matches!(key.op, Op::Context { .. })));
            let missing = set.keys.iter().copied().filter(|key| !(key.rank == 1 && key.rows == 16
                && matches!(key.op, Op::Context { .. }))).collect::<Vec<_>>();
            assert!(check_placement(&cfg, &placement, &missing).is_err());
        }}}
        assert!(inventory_with_context(&cfg, 1, 64, 16, 0, 1, |_, _| true,
            |layer| cfg.attention[layer] == KimiAttention::Mla, Some(&decode)).is_err());
        Ok(())
    }

    #[test]
    fn context_startup_covers_every_c1_to_c16_mixed_verify_batch() -> Result<()> {
        let mut cfg = KimiK3Config::from_hf(&cuteafd_loader::plan::testing::kimi_k3_config())?;
        // One MLA layer is sufficient for the exhaustive row/slot geometry;
        // the full-family test checks every real layer and graph accounting.
        cfg.layers = 1;
        cfg.attention = vec![KimiAttention::Mla];
        let decode = cuteafd_loader::families::kimi_k3::context::DecodeBuckets::from_manifest(
            &serde_json::json!({"programs":[{"name":"dense_trellis_k6_trellis_mla_q_a_mcg",
                "geometry":{"bits":6,"decode_route_thresholds":[8,16,32,48]}}]}))?;
        for slots in 1..=16 { for verify in 2..=64 {
            let set = inventory_with_context(&cfg, 1, 2752, slots, verify, 2, |_, _| true,
                |layer| cfg.attention[layer] == KimiAttention::Mla, Some(&decode))?;
            let declared = set.keys.iter().copied().collect::<std::collections::HashSet<_>>();
            for active in 1..=slots {
                for speculative in [false, true] {
                    // Independently enumerate the scheduler's flattened batch:
                    // each active request supplies an anchor and 0..verify-1 drafts.
                    let reachable = if speculative { active.max(2)..=(active * verify).min(64) }
                        else { active..=active };
                    for rows in reachable {
                        let bucket = decode.dispatch_rows(rows, speculative)?;
                        for rank in 0..2 {
                            for layer in (0..cfg.layers).filter(|&layer| cfg.attention[layer] == KimiAttention::Mla) {
                                assert!(declared.contains(&Key { lane: 0, rank, rows: bucket,
                                    op: Op::Context { layer, speculative } }),
                                    "missing Context graph: slots={slots} verify={verify} active={active} rows={rows}");
                            }
                        }
                    }
                }
            }
            for rank in 0..2 {
                assert_eq!(set.inventory.ranks[rank].executables,
                    set.keys.iter().filter(|key| key.rank == rank).count() as u64);
            }
        }}
        Ok(())
    }

    #[test]
    fn single_owner_keys_capture_output_without_a_shadow_rank() -> Result<()> {
        let cfg = KimiK3Config::from_hf(&cuteafd_loader::plan::testing::kimi_k3_config())?;
        for capacity in [64, 2752, 4096] {
            let set = inventory_for_ranks(&cfg, 2, capacity, 16, 8, 1, |rank, _| rank == 0)?;
            assert_eq!(set.inventory.ranks.len(), 1);
            assert!(set.keys.iter().all(|key| key.rank == 0));
            assert_eq!(set.inventory.ranks[0].executables, set.keys.len() as u64);
            let unique = set.keys.iter().collect::<std::collections::HashSet<_>>();
            assert_eq!(unique.len(), set.keys.len());
            for lane in 0..2 { for rows in graph_rows(8, capacity) {
                assert!(unique.contains(&Key { lane, rank: 0, rows, op: Op::Residual(MixSite::Output) }));
            }}
            let pair = inventory(&cfg, 2, capacity, 16, 8, |_, _| true)?;
            let output = set.keys.iter().filter(|key| key.op == Op::Residual(MixSite::Output)).count();
            assert_eq!(pair.keys.len(), 2 * set.keys.len() - output);
        }
        assert!(inventory_for_ranks(&cfg, 1, 64, 16, 8, 0, |_, _| true).is_err());
        Ok(())
    }

    #[test]
    fn owned_keys_match_shared_admission_and_reject_wrong_residency() -> Result<()> {
        let cfg = KimiK3Config::from_hf(&cuteafd_loader::plan::testing::kimi_k3_config())?;
        let mut inputs = super::super::admission::tests::inputs(&cfg);
        let provisional = owned_inventory(&cfg, inputs.lanes, inputs.rows, inputs.state_slots, inputs.verify_rows)?;
        inputs.startup.extend((0..2).map(|rank| demand(&provisional.inventory, rank)));
        let admitted = super::super::admission::admit(inputs)?;
        let inventory = admitted.graph_set.as_ref().expect("winning owner graph inventory");
        check_placement(&cfg, &admitted.placement, &inventory.keys)?;
        let mut wrong = inventory.keys.clone();
        let layer = cfg.attention.iter().position(|kind| *kind == KimiAttention::Mla).unwrap();
        let key = wrong.iter_mut().find(|key| matches!(key.op,
            Op::Attention { layer: captured, .. } if captured == layer)).unwrap();
        key.rank = 1 - key.rank;
        assert!(check_placement(&cfg, &admitted.placement, &wrong).is_err());
        let missing = inventory.keys.iter().copied().filter(|key| !matches!(key.op,
            Op::Attention { layer: captured, .. } if captured == layer)).collect::<Vec<_>>();
        assert!(check_placement(&cfg, &admitted.placement, &missing).is_err());
        Ok(())
    }

    #[test]
    fn startup_inventory_is_exact_and_partition_sensitive() -> Result<()> {
        let cfg = KimiK3Config::from_hf(&cuteafd_loader::plan::testing::kimi_k3_config())?;
        let admitted = inventory(&cfg, 2, 4096, 16, 0, |rank, layer|
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
            assert!(unique.contains(&Key { lane: 1, rank: 1, rows: 16, op: Op::Attention { layer, sequences, speculative: false } }));
        }
        let small = inventory(&cfg, 1, 4096, 4, 0, |_, _| true)?;
        assert!(!small.keys.iter().any(|key| matches!(key.op,
            Op::Attention { sequences, .. } if sequences > 4)));
        for (lanes, rows, slots) in [(0, 1, 1), (1, 0, 1), (1, 4097, 1), (1, 1, 0), (1, 1, 161)] {
            assert!(inventory(&cfg, lanes, rows, slots, 0, |_, _| true).is_err());
        }
        // Verify widths add exact rows and speculative single-sequence attention only.
        let verify = inventory(&cfg, 1, 4096, 4, 8, |_, _| true)?;
        let unique = verify.keys.iter().collect::<std::collections::HashSet<_>>();
        assert_eq!(unique.len(), verify.keys.len());
        for rows in 2..=8 {
            assert!(unique.contains(&Key { lane: 0, rank: 0, rows, op: Op::Ffn(0) }));
            assert!(unique.contains(&Key { lane: 0, rank: 1, rows, op: Op::Attention { layer, sequences: 1, speculative: true } }));
        }
        assert!(!verify.keys.iter().any(|key| matches!(key.op, Op::Attention { speculative: true, sequences, .. } if sequences > 1)));
        assert!(!verify.keys.iter().any(|key| key.rows > 8 && matches!(key.op, Op::Attention { speculative: true, .. })));
        for rank in 0..2 {
            assert_eq!(verify.inventory.ranks[rank].executables,
                verify.keys.iter().filter(|key| key.rank == rank).count() as u64);
        }
        assert!(verify.keys.len() > small.keys.len());
        Ok(())
    }
}
