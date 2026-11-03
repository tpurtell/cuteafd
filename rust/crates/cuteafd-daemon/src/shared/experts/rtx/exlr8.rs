//! Native EXLR8 routed experts on one RTX or the shared paired-expert executor.
use super::native::RankCompletion;
use super::{ExpertInput, PartialDtype, Routes, RtxExpertLayer, RtxShard};
use crate::shared::experts::exlr8::{grouped_decode_from_row, view, Exlr8Execution, Exlr8PrefillActivations, Exlr8Weights};
use crate::shared::memory::device::{Allocation, Device, DeviceOwner, Event};
use anyhow::{bail, ensure, Context, Result};
use cuteafd_ffi::{exlr8::KernelFamily, programs::Programs, CuteafdDeviceBuffer};
use cuteafd_loader::plan::exlr8::residency::ExpertSet;
use std::{ffi::c_void, rc::Rc};

const DECODE_ROWS: usize = 64;
const TP2_UNITS: usize = 8;

fn native_shard(world: usize, rank: usize) -> Result<RtxShard> {
    ensure!((1..=2).contains(&world) && rank < world, "EXLR8 RTX requires world 1 or 2 and a valid rank");
    Ok(if world == 1 { RtxShard::Whole } else { RtxShard::Tp2 { rank: rank as u8 } })
}

#[cfg(test)]
mod copy_tests {
    #[test]
    fn native_rtx_shards_follow_actual_world_and_rank() {
        use super::{native_shard, RtxShard};
        assert_eq!(native_shard(1, 0).unwrap(), RtxShard::Whole);
        for rank in 0..2 { assert_eq!(native_shard(2, rank).unwrap(), RtxShard::Tp2 { rank: rank as u8 }); }
        for (world, rank) in [(0, 0), (1, 1), (2, 2), (3, 0), (usize::MAX, 0)] {
            assert!(native_shard(world, rank).is_err());
        }
    }

    #[test]
    fn all_decode_operands_use_the_shared_sm_copy_route() {
        use crate::shared::memory::device::sm_peer_copy_for;
        for rows in [1, 16, DECODE_ROWS] {
            for bytes in [rows * 6144 * 33 / 32, rows * 8 * 4, rows * 6144 * 2] {
                assert!(sm_peer_copy_for(true, bytes, DECODE_ROWS * 6144 * 2));
            }
        }
        assert!(!sm_peer_copy_for(true, DECODE_ROWS * 6144 * 2 + 1, DECODE_ROWS * 6144 * 2));
        assert!(!sm_peer_copy_for(true, 4096 * 6144 * 2, DECODE_ROWS * 6144 * 2));
        let source = include_str!("exlr8.rs");
        let enqueue = source.split("unsafe fn enqueue(").nth(1).unwrap();
        assert_eq!(enqueue.matches("copy_d2d_async(").count(), 1);
        assert!(enqueue.contains("sm_peer_copy_for(true, input.bytes, DECODE_ROWS * self.owner.hidden * 2)"));
        assert!(enqueue.contains("self.sm_copy.launch_rows("), "same-device SM copies require the local-capable entry point");
        assert!(enqueue.contains("copy(view(self.output.buffer, 0, bytes)?,"));
    }
    use super::DECODE_ROWS;

    #[test]
    #[ignore = "requires an allocated CPU build slot and explicit NVMe fixture directory"]
    fn retained_glm_rtx_constructors_reconcile_mount_and_push_owners() -> anyhow::Result<()> {
        constructor_owners(super::KernelFamily::Glm)
    }

    #[test]
    #[ignore = "requires an allocated CPU build slot and explicit NVMe fixture directory"]
    fn retained_flash_rtx_constructors_reconcile_mount_and_push_owners() -> anyhow::Result<()> {
        constructor_owners(super::KernelFamily::GlmFlash)
    }

    /// Full family geometry and descriptor extents, deliberately no expert payload.
    /// This can test allocation/ownership construction, never checkpoint or math fidelity.
    fn allocation_only_residency(family: super::KernelFamily, layer: usize, world: usize, rank: usize)
        -> cuteafd_loader::plan::exlr8::residency::Residency {
        use cuteafd_loader::plan::exlr8::residency::{ExpertSet, Residency};
        let geometry = family.geometry();
        let (hidden, experts, units) = (geometry.hidden, geometry.experts, geometry.units);
        let mut bytes = 0usize;
        let mut reserve = |extent: usize| {
            let start = bytes;
            bytes = (bytes + extent).next_multiple_of(256);
            start
        };
        let signs_offsets = [reserve(hidden), reserve(hidden)];
        let pointers_offset = reserve(experts * units * 3 * 3 * 8);
        let bits_offset = reserve(experts * units * 3 * 4);
        let compact_offsets = [reserve(experts * 4), reserve(experts * units * 4)];
        Residency { layer, world, rank, hidden, experts, units, set: ExpertSet::Routed,
            set_id: "allocation-only".into(), manifest: "allocation-only".into(), loads: vec![],
            signs: [vec![1; hidden], vec![255; hidden]], signs_offsets, pointers_offset,
            bits_offset, compact_offsets, resident_bytes: bytes, payload_bytes: 0, staging_bytes: 0 }
    }

    fn constructor_weights<'a>(device: super::Device<'a>, layouts: &[cuteafd_loader::plan::exlr8::residency::Residency],
        pushed: bool) -> anyhow::Result<Vec<super::Exlr8Weights<'a>>> {
        use super::Exlr8Weights;
        use crate::shared::memory::{push_initialize, weight_arena::*};
        use std::rc::Rc;
        device.run(|| {
            let mounted = layouts.iter().cloned().map(|layout|
                Exlr8Weights::load(device.library, layout, 128 << 20)).collect::<anyhow::Result<Vec<_>>>()?;
            if !pushed { return Ok(mounted); }
            let bytes = layouts.iter().map(|layout| layout.resident_bytes).sum();
            let arena = ArenaLayout::new(layouts.iter().map(|layout| layout.resident_bytes), bytes)?;
            let mut loading = LoadingArena::new(device, arena.clone(), ExpectedCoverage::spans(&[(0, bytes as u64)]))?;
            for (index, state) in mounted.iter().enumerate() {
                let mut image = vec![0; state.image().bytes];
                device.library.copy_d2h(&mut image, state.image())?;
                device.library.copy_h2d(loading.destination(arena.region(index)?.start, image.len())?, &image)?;
            }
            // Images above prove only this constructor's synthetic delivery. Run the
            // real post-Done sign validation and descriptor relocation before Ready.
            loading.copied(0, bytes as u64)?;
            let layers = layouts.iter().enumerate().map(|(region, residency)|
                push_initialize::Layer::Exlr8 { region, residency }).collect::<Vec<_>>();
            let ready = Rc::new(push_initialize::initialize(loading.done()?, &layers, None, 0)?);
            layouts.iter().cloned().enumerate().map(|(region, layout)| {
                let budget = Exlr8Weights::plan(&layout)?;
                Exlr8Weights::from_push(ready.clone(), region, layout, budget)
            }).collect()
        })
    }

    fn constructor_owners(family: super::KernelFamily) -> anyhow::Result<()> {
        use super::*;
        use cuteafd_core::memory_layout::Category;
        use cuteafd_ffi::{memory_ledger::{self, AllocationAudit, AllocationEvent, Space},
            native_library_lifetime_fixture::{CopyKind, Fixture}};
        use cuteafd_loader::{placement::planner::{self, *},
            plan::exlr8::{execution_workspace, residency::WeightLayout, ExecutionVariant}};
        use std::collections::BTreeMap;
        let fixture = Fixture::build()?;
        let library = fixture.load()?;
        fixture.quiet_events(&library, true)?;
        fixture.immediate_copies(&library)?;
        fixture.exlr8_programs(&library, family == KernelFamily::GlmFlash)?;
        let manifest = fixture.directory().join("PROGRAMS.json");
        std::fs::write(&manifest, serde_json::to_vec(&crate::shared::experts::exlr8::owner_manifest(family))?)?;
        let programs = library.programs()?.with_manifest(&manifest)?;
        let lifetime = Lifetime::RESIDENT;
        let layers = vec![3, 7];
        let capacity = 128usize << 20;
        for (world, rank) in [(1, 0), (2, 0), (2, 1)] {
            // Logical rank 0 deliberately lives on physical device 1.
            let physical = 1 - rank as i32;
            let device = crate::shared::memory::device::Device { library: &library, id: physical };
            let location = Location::coordinator(rank as u8);
            fixture.configure_memory(&library, physical, capacity)?;
            let layouts = layers.iter().map(|&layer| allocation_only_residency(family, layer, world, rank)).collect::<Vec<_>>();
            for activations in [Exlr8PrefillActivations::A16, Exlr8PrefillActivations::A8] {
                for rows in [DECODE_ROWS, 128] {
                    let unit_limit = layouts[0].compact_unit_limit();
                    let workspace = Exlr8RtxWeights::workspace_bytes_with_limit(&programs, family, rows, unit_limit, activations)?;
                    let resident = layouts.iter().map(|layout| layout.resident_bytes).sum::<usize>();
                    for pushed in [false, true] {
                        for short in [true, false] {
                            let tag = format!("{}/world{world}/rank{rank}/{activations:?}/{rows}/{pushed}/{short}", fixture.directory().display());
                            let _owner = memory_ledger::owner(&tag);
                            let audit = AllocationAudit::start();
                            let weights = constructor_weights(device, &layouts, pushed)?;
                            fixture.finish_ordered_uploads(&library)?;
                            fixture.record_peer_copies(&library)?;
                            library.cuda_set_device(1 - physical)?;
                            let owner = Exlr8RtxWeights::pushed_rank(device, &programs, layers.clone(), rows,
                                resident + workspace - usize::from(short), activations, world, rank, weights);
                            assert_eq!(library.cuda_get_device()?, 1 - physical);
                            if short {
                                assert!(owner.is_err(), "one-byte-short peak must refuse before execution allocation");
                                assert!(fixture.peer_copy_records(&library)?.is_empty());
                            } else {
                                let expert = owner?.execution()?;
                                assert_eq!(expert.shard(), native_shard(world, rank)?);
                                assert_eq!(expert.layers(), &layers);
                                assert_eq!(expert.device(), physical);
                                assert_eq!(expert.workspace_bytes(), workspace);
                                assert_eq!(expert.executions.len(), if rows == DECODE_ROWS { 2 } else { 3 });
                                let topology = if pushed { WeightArenaTopology::SharedRank } else { WeightArenaTopology::PerLayer };
                                let mut allocations = WeightLayout::from_residencies("test.weights", &layouts.iter().collect::<Vec<_>>(),
                                    topology, location, lifetime)?.allocations;
                                let mut receipts = Exlr8Weights::weight_receipts(&expert.owner.weights, "test.weights", location, physical, lifetime)?;
                                for (index, execution) in expert.executions.iter().enumerate() {
                                    let variant = [ExecutionVariant::DecodeCompact, ExecutionVariant::DecodeGrouped, ExecutionVariant::Prefill][index];
                                    let shape = if index < 2 { DECODE_ROWS } else { rows };
                                    let format = if index < 2 { Exlr8PrefillActivations::A16 } else { activations };
                                    let prefix = format!("test.execution.{index}");
                                    let layout = execution_workspace(programs.spec(variant.program(family, format))?, shape,
                                        ExpertSet::Routed, variant, unit_limit, format)?;
                                    allocations.extend(layout.allocations(&prefix, location, lifetime)?);
                                    receipts.extend(execution.workspace_receipts(&prefix, location, physical, lifetime)?);
                                }
                                let mut output = planner::Allocation::buffer("test.output", location,
                                    (rows * family.geometry().hidden * 2) as u64, lifetime, Category::Workspace, "EXLR8 stable output");
                                output.alignment = 256;
                                allocations.push(output);
                                let buffer = expert.output.buffer;
                                let mut actual = planner::Allocation::buffer("test.output", location, buffer.bytes as u64,
                                    lifetime, Category::Workspace, "EXLR8 stable output");
                                actual.alignment = 256;
                                receipts.push(Receipt { allocation: actual, address: buffer.ptr as u64 });
                                let facts = Facts { allocations, deferred: vec![], sequential: vec![], allowances: vec![] };
                                let budgets = vec![Budget { id: "CPU constructor".into(), locations: vec![location],
                                    physical_bytes: capacity as u64, logical_bytes: capacity as u64, headroom: 0,
                                    baseline: Baseline::Forecast { occupied: 0, source: "CPU fixture".into(), assumed: false } }];
                                let phases = planner::evaluate(&facts, &budgets)?;
                                let plan = ResolvedPlan { request: Request { context: 64, pool: Pin::Exact(64), onboard: Pin::Auto,
                                        prefill: Pin::Auto, operation: Operation::Serving },
                                    decisions: Decisions { attention: vec![], onboard_layers: vec![], expert_homes: Default::default(),
                                        embedding: None, prefill: PrefillShape { rows: 64, waves: 1 }, lanes: 1 },
                                    pool_tokens: 64, facts, budgets, phases };
                                planner::reconcile(&plan, Phase::Load, &receipts)?;
                                let owners = receipts.iter().filter(|receipt| receipt.allocation.view.is_none())
                                    .map(|receipt| (receipt.address as usize, receipt.allocation.bytes as usize)).collect::<BTreeMap<_, _>>();
                                let live = memory_ledger::allocations().into_iter().filter(|record| record.key.owner == Some(tag.as_str()))
                                    .map(|record| {
                                        assert_eq!((record.key.space, record.key.device), (Space::Device, physical));
                                        (record.address, record.bytes)
                                    }).collect::<BTreeMap<_, _>>();
                                assert_eq!(live, owners, "every actual retained owner must have exactly one receipt");
                                assert_eq!(owners.len(), usize::from(!pushed) + 1 + expert.executions.len() + 1);
                                for execution in expert.executions.iter() {
                                    assert_ne!(execution.allocation_buffer().ptr, buffer.ptr);
                                    assert_ne!(execution.output()?.ptr, buffer.ptr);
                                }
                                assert_eq!(expert.output(), buffer.ptr);
                                assert_eq!(buffer.device_id, physical);
                                assert_eq!(expert.partial(), PartialDtype::Bf16);
                                assert!(expert.uses_fp8_input());
                                assert_eq!(expert.output(), buffer.ptr);
                                let trace = fixture.peer_copy_records(&library)?;
                                assert_eq!(trace.len(), 1);
                                assert_eq!(trace[0].kind, CopyKind::Initialize);
                                assert_eq!(trace[0].device, physical);
                                drop(expert);
                                assert_eq!(fixture.peer_copy_records(&library)?, trace);
                            }
                            assert_eq!(library.cuda_get_device()?, 1 - physical);
                            assert!(memory_ledger::allocations().iter().all(|record| record.key.owner != Some(tag.as_str())));
                            assert_eq!(device.run(|| library.cuda_memory_info())?, (capacity, capacity));
                            let mut live = BTreeMap::new();
                            let mut allocated = 0;
                            for event in audit.events() {
                                let (add, record) = match event {
                                    AllocationEvent::Allocated(record) => (true, record),
                                    AllocationEvent::Freed(record) => (false, record),
                                };
                                if record.key.owner != Some(tag.as_str()) { continue; }
                                if add {
                                    allocated += 1;
                                    assert!(live.insert(record.address, (record.key, record.bytes)).is_none());
                                } else {
                                    assert_eq!(live.remove(&record.address), Some((record.key, record.bytes)));
                                }
                            }
                            assert!(allocated > 0 && live.is_empty(), "constructor histories must pair every allocation/free");
                        }
                    }
                }
            }
        }
        Ok(())
    }
}

pub(crate) struct Exlr8RtxWeights<'a> {
    device: Device<'a>,
    shard: RtxShard,
    unit_limit: usize,
    layers: Vec<usize>,
    programs: &'a Programs<'a>,
    max_rows: usize,
    family: KernelFamily,
    hidden: usize,
    activations: Exlr8PrefillActivations,
    workspace_bytes: usize,
    grouped_from_row: usize,
    weights: Rc<DeviceOwner<'a, Vec<Exlr8Weights<'a>>>>,
}

pub(crate) struct Exlr8Rtx<'a> {
    // Drain every consumer before releasing graph-visible scratch or weights.
    completion: RankCompletion<Event<'a>>,
    owner: Exlr8RtxWeights<'a>,
    executions: DeviceOwner<'a, Vec<Exlr8Execution<'a>>>,
    output: Allocation<'a>,
    sm_copy: cuteafd_ffi::V41PeerCopy<'a>,
}

impl<'a> Exlr8RtxWeights<'a> {
    pub(crate) fn workspace_bytes(programs: &'a Programs<'a>, family: KernelFamily, max_rows: usize,
        activations: Exlr8PrefillActivations) -> Result<usize> {
        Self::workspace_bytes_with_limit(programs, family, max_rows, TP2_UNITS, activations)
    }

    pub(crate) fn workspace_bytes_with_limit(programs: &'a Programs<'a>, family: KernelFamily,
        max_rows: usize, unit_limit: usize, activations: Exlr8PrefillActivations) -> Result<usize> {
        ensure!((DECODE_ROWS..=4096).contains(&max_rows), "EXLR8 RTX row capacity");
        ensure!((1..=family.geometry().units).contains(&unit_limit), "EXLR8 RTX unit capacity");
        let hidden = Exlr8Execution::routed_hidden(programs, family)?;
        let compact = Exlr8Execution::decode_compact_workspace_bytes(programs, family, DECODE_ROWS, unit_limit)?;
        let grouped = Exlr8Execution::decode_grouped_workspace_bytes(programs, family, DECODE_ROWS, unit_limit)?;
        let prefill = if max_rows > DECODE_ROWS {
            Exlr8Execution::workspace_bytes_with_limit(programs, family, max_rows, ExpertSet::Routed, unit_limit, activations)?
        } else { 0 };
        compact.checked_add(grouped).and_then(|bytes| bytes.checked_add(prefill))
            .and_then(|bytes| max_rows.checked_mul(hidden)?.checked_mul(2)?.checked_add(bytes))
            .context("EXLR8 RTX workspace overflow")
    }

    pub(crate) fn pushed_pair(devices: [Device<'a>; 2], programs: &'a Programs<'a>,
        layers: Vec<usize>, max_rows: usize, budgets: [usize; 2],
        activations: Exlr8PrefillActivations, pushed: [Vec<Exlr8Weights<'a>>; 2]) -> Result<[Self; 2]> {
        let [left, right] = pushed;
        Ok([Self::pushed_rank(devices[0], programs, layers.clone(), max_rows, budgets[0], activations, 2, 0, left)?,
            Self::pushed_rank(devices[1], programs, layers, max_rows, budgets[1], activations, 2, 1, right)?])
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn pushed_rank(device: Device<'a>, programs: &'a Programs<'a>, layers: Vec<usize>,
        max_rows: usize, budget: usize, activations: Exlr8PrefillActivations,
        world: usize, rank: usize, weights: Vec<Exlr8Weights<'a>>) -> Result<Self> {
        let shard = native_shard(world, rank)?;
        ensure!(!layers.is_empty() && layers.windows(2).all(|ids| ids[0] < ids[1]),
            "EXLR8 RTX needs ordered resident layer IDs");
        let first = &weights.first().context("EXLR8 RTX has no pushed weights")?.layout;
        let expected = (first.hidden, first.units, first.experts);
        let family = KernelFamily::from_geometry(expected.0, expected.1, 8, expected.2)?;
        let hidden = Exlr8Execution::routed_hidden(programs, family)?;
        let unit_limit = first.compact_unit_limit();
        let workspace_bytes = Self::workspace_bytes_with_limit(programs, family, max_rows, unit_limit, activations)?;
        let info = device.library.cuda_device_info(device.id)?;
        let grouped_from_row = grouped_decode_from_row(unit_limit,
            (info.compute_capability_major, info.compute_capability_minor));
        ensure!(weights.len() == layers.len(), "EXLR8 pushed layer count differs from admission");
        let resident = weights.iter().zip(&layers).try_fold(0usize, |sum, (weight, &layer)| {
            ensure!(weight.layout.layer == layer && weight.layout.world == world && weight.layout.rank == rank
                && weight.layout.hidden == hidden && weight.layout.experts == expected.2
                && weight.layout.units == expected.1 && weight.layout.set == ExpertSet::Routed
                && weight.layout.compact_unit_limit() == unit_limit,
                "EXLR8 pushed RTX selection differs from admission");
            sum.checked_add(weight.budget.resident_bytes).context("EXLR8 RTX residency overflow")
        })?;
        ensure!(resident.checked_add(workspace_bytes).is_some_and(|bytes| bytes <= budget),
            "EXLR8 RTX weights and execution exceed admitted peak");
        let weights = device.own(|| Ok(weights))?;
        Ok(Self { device, shard, unit_limit, layers, programs, max_rows, family, hidden,
            activations, workspace_bytes, grouped_from_row, weights: Rc::new(weights) })
    }

    pub(crate) fn execution(self) -> Result<Exlr8Rtx<'a>> {
        let device = self.device;
        let executions = device.own(|| {
            let mut executions = vec![
                Exlr8Execution::new_decode_compact(device.library, self.programs, self.family, DECODE_ROWS,
                    self.workspace_bytes, self.unit_limit)?,
                Exlr8Execution::new_decode_grouped(device.library, self.programs, self.family, DECODE_ROWS,
                    self.workspace_bytes, self.unit_limit)?,
            ];
            if self.max_rows > DECODE_ROWS {
                executions.push(Exlr8Execution::new_with_limit(device.library, self.programs, self.family,
                    self.max_rows, self.workspace_bytes, ExpertSet::Routed, self.unit_limit, self.activations)?);
            }
            for execution in &mut executions { execution.seal_graph_storage()?; }
            Ok(executions)
        })?;
        let completion = RankCompletion::new(Event::new(device)?);
        let output = Allocation::new(device, self.max_rows * self.hidden * 2)?;
        let sm_copy = device.run(|| device.library.v41_peer_copy())?;
        Ok(Exlr8Rtx { completion, owner: self, executions, output, sm_copy })
    }
}

impl RtxExpertLayer for Exlr8Rtx<'_> {
    fn shard(&self) -> RtxShard { self.owner.shard }
    fn partial(&self) -> PartialDtype { PartialDtype::Bf16 }
    fn uses_fp8_input(&self) -> bool { true }
    fn layers(&self) -> &[usize] { &self.owner.layers }
    fn device(&self) -> i32 { self.owner.device.id }
    fn workspace_bytes(&self) -> usize { self.owner.workspace_bytes }
    fn output(&self) -> *mut c_void { self.output.buffer.ptr }
    fn group_rows(&self, rows: usize) -> Result<usize> {
        ensure!(rows <= self.owner.max_rows, "EXLR8 grouping rows exceed admission");
        let kind = if rows == 0 || rows > DECODE_ROWS { cuteafd_ffi::exlr8::KernelKind::GroupedA16 }
            else { cuteafd_ffi::exlr8::KernelKind::Routed };
        let name = self.owner.family.program(kind);
        Ok(cuteafd_ffi::exlr8::group_rows(self.owner.programs.spec(name)?)?)
    }
    fn weight_read_bytes(&self) -> Result<usize> {
        self.owner.weights.iter().map(|weights| weights.budget.resident_bytes / weights.layout.experts)
            .max().context("EXLR8 local weights missing")
    }
    unsafe fn enqueue(&mut self, layer: usize, rows: usize, input: ExpertInput, routes: Routes,
        stream: *mut c_void, abort: &mut dyn FnMut()) -> Result<()> {
        let result = (|| {
            ensure!((1..=self.owner.max_rows).contains(&rows), "EXLR8 local rows exceed admission");
            let index = super::resident_index(&self.owner.layers, layer)?;
            let ExpertInput::Fp8K32(wire) = input else { bail!("EXLR8 RTX requires FP8 K32 wire input") };
            ensure!(!wire.is_null() && !routes.ids.is_null() && !routes.weights.is_null(), "null EXLR8 RTX input");
            let selected = if rows > DECODE_ROWS { 2 }
                else { usize::from(rows >= self.owner.grouped_from_row) };
            let execution = &self.executions[selected];
            let device = self.owner.device;
            let source = |ptr, bytes| CuteafdDeviceBuffer { ptr, bytes, device_id: device.id, flags: 0 };
            device.run(|| {
                let copy = |target, input: CuteafdDeviceBuffer| -> Result<()> {
                    // SAFETY: rank-local operands outlive this stream's completion;
                    // decode copies execute on SMs, never behind a parked copy engine.
                    unsafe {
                        if crate::shared::memory::device::sm_peer_copy_for(true, input.bytes, DECODE_ROWS * self.owner.hidden * 2) {
                            self.sm_copy.launch_rows(target, input, input.bytes, 1, input.bytes, input.bytes, stream)
                        } else {
                            device.library.copy_d2d_async(target, input, input.bytes, stream)
                        }
                    }
                };
                for (target, input) in [
                    (execution.input()?, source(wire, rows * self.owner.hidden * 33 / 32)),
                    (execution.ids()?, source(routes.ids, rows * 8 * 4)),
                    (execution.route_weights()?, source(routes.weights, rows * 8 * 4)),
                ] {
                    copy(view(target, 0, input.bytes)?, input)?;
                }
                // SAFETY: immutable weights and sealed arena remain owned until the completion event drains.
                unsafe {
                    execution.run(&self.owner.weights[index], rows, stream)?;
                    let bytes = rows * self.owner.hidden * 2;
                    copy(view(self.output.buffer, 0, bytes)?, view(execution.output()?, 0, bytes)?)?;
                }
                Ok(())
            })
        })();
        self.completion.finish(result, stream, abort)
    }
}
