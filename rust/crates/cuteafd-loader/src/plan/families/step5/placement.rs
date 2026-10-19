//! Header-only lower bounds, using the shared pool solver; not runtime admission.
use super::Model;
use crate::{placement::*, plan::{Component, ExpertPlacement, Hint, PlanReport}};
use cuteafd_core::memory_layout::{Basis, Category, DeviceKind, DeviceLayout, Item, MemoryLayout};

const GIB: u64 = 1 << 30;
const FLOOR: u64 = 262_144;
const PAGE: u64 = 64;
const SINGLE: LayerMode = LayerMode::Whole { gpu: 0, ffn: FfnMode::Owner };
const MODES: &[LayerMode] = &[SINGLE, LayerMode::HeadSplit];

pub(super) fn layout(model: &Model, report: &mut PlanReport, opts: &crate::plan::layout::LayoutOptions,
    requested_placement: ExpertPlacement) -> MemoryLayout {
    let cfg = &model.cfg;
    let ranks = requested_placement.spark_ranks();
    let gpus = opts.rtx_bytes.len();
    let mut memory = MemoryLayout { devices: Vec::new(), pool_tokens: 0, waste: Vec::new(), notes: vec![
        "Step-5 header-only lower bound: native BF16 K/V; no workspace/graphs/prefix/transport/runtime allowance inferred; NOT serving admission".into(),
        format!("Maximum request {} tokens; aggregate pool is distinct and may reach 2M", cfg.max_context),
    ] };
    let refusal = |report: &mut PlanReport, what: String| {
        report.fits = false;
        report.placement_supported = false;
        report.hints.push(Hint { what, how: "Use the fitting official FP8 TP6 or NVFP4 TP4/TP6 recipe on 1/2 RTX PRO 6000; implement native kernels and exact state/workspace admission before serving.".into() });
    };
    report.bytes_by_owner.clear();
    if !model.complete {
        refusal(report, "Step-5 placement cannot infer fit from missing safetensors headers".into());
        return memory;
    }
    if !(1..=2).contains(&gpus) || (gpus == 2 && !opts.head_split)
        || report.requested_attention.is_some_and(|mode| mode != AttentionPlacement::Heads) {
        refusal(report, "Step-5 only head/provider split has a metadata placement: context/layer ownership requires executor and exact state-transfer gates".into());
        return memory;
    }
    let experts = *model.bytes.get(&Component::RoutedExpert).unwrap_or(&0);
    let vision = *model.bytes.get(&Component::Vision).unwrap_or(&0);
    let coordinator: u64 = model.bytes.iter().filter(|(c,_)| **c != Component::RoutedExpert && **c != Component::Vision)
        .map(|(_,bytes)| *bytes).sum();
    // Every source tensor, including all three independent MTP heads, remains
    // represented. Conservatively replicate the bundled vision tower on TP2.
    let per_gpu = coordinator.div_ceil(gpus as u64) + vision;
    report.bytes_by_owner.insert("coordinator source weights (all MTP + vision)".into(), coordinator+vision);
    report.bytes_by_owner.insert("routed expert source weights".into(), experts);
    let fp8 = model.formats.iter().any(|f| f == "fp8-block128");
    let nvfp4 = model.formats.iter().any(|f| f == "nvfp4-g16-w4a4");
    let worlds: &[usize] = if fp8 { &[6] } else if nvfp4 { &[4,6] } else { &[] };
    report.min_spark_ranks = worlds.iter().copied().find(|r| experts.div_ceil(*r as u64) <= opts.spark_bytes);
    report.spark_rank_share = if ranks > 0 { 1.0/ranks as f64 } else { 0.0 };
    if !worlds.contains(&ranks) {
        refusal(report, if requested_placement == ExpertPlacement::Local {
            "Step-5 no 0-Spark recipe fits official FP8/NVFP4 routed weights".into()
        } else { format!("Step-5 {} TP{ranks} refused: routed weights {:.3} GiB/rank; required card recipe {:?}",
            model.formats.join("+"), experts as f64 / ranks.max(1) as f64 / GIB as f64, worlds) });
    }
    if opts.rtx_bytes.iter().any(|bytes| *bytes <= 32*GIB) {
        refusal(report, "Step-5 5090 refused: coordinator weights plus >=256K agentic KV exceed 32 GiB; more expert Sparks do not fix coordinator fit; requires shared coordinator offload or calibrated smaller quant".into());
    }
    if ranks > 0 {
        for rank in 0..ranks {
            let resident = experts.div_ceil(ranks as u64);
            memory.devices.push(DeviceLayout { kind: DeviceKind::Spark, index: rank as u32,
                capacity_bytes: opts.spark_bytes, kv_tokens: 0,
                items: vec![Item::new(Category::Experts, "stacked experts including native BF16 tail", model.formats.join("+"), resident, Basis::Formula)] });
            if resident > opts.spark_bytes {
                refusal(report, format!("Step-5 TP{ranks} experts need {:.3} GiB/rank before runtime, over {:.3} GiB budget",
                    resident as f64/GIB as f64, opts.spark_bytes as f64/GIB as f64));
            }
        }
    }
    let requested = opts.pool_tokens.filter(|n| *n > 0);
    let tokens_per_layer = (2*cfg.groups*cfg.dim*2) as u64;
    let full = cfg.full[..cfg.layers].iter().filter(|full| **full).count() as u64;
    let per_token = full*(tokens_per_layer+256/8);
    memory.notes.push(format!("BF16 target GQA K/V + shared CSA index summaries: {per_token} bytes/token (not attention KV / 8); {:.3} GiB at 256K, {:.3} GiB at 1M",
        per_token as f64*FLOOR as f64/GIB as f64, per_token as f64*(1<<20) as f64/GIB as f64));
    memory.notes.push(format!("Proposed FP8 K/V ideal lower bound only (NOT selected or quality-qualified): {} bytes/token plus scales/padding", full*(tokens_per_layer/2+256/8)));
    let slots = opts.concurrency.max(1);
    // Header-only plan uses the minimum 64-row step. Runtime charges its actual
    // prefill/verify extent before allocating, never treating slack as pool-fixed.
    let ring_rows = 576u64;
    let slack_per_sequence = ((cfg.layers as u64-full)+cfg.mtp as u64)
        * (ring_rows-cfg.window as u64)*tokens_per_layer;
    memory.notes.push(format!("Append-only SWA/MTP slack lower bound: {slack_per_sequence} bytes per admitted sequence across coordinator ranks at step64; {slots} sequences charge {} bytes. Runtime charges larger prefill steps separately.", slack_per_sequence*slots));
    let mut fixed = Vec::new();
    for gpu in 0..gpus {
        fixed.push(Demand::new(gpu as u8, Category::Weights, "target + 3 MTP stages, vision conservatively replicated", per_gpu, Basis::Formula));
        // SWA target and MTP rings retain original BF16 rows. CSA pending K/Z
        // is shared across provider groups, so replicated on both head owners.
        let target_rings = (cfg.layers as u64-full)*ring_rows*tokens_per_layer*slots/gpus as u64;
        let mtp_rings = if gpu == 0 { cfg.mtp as u64*ring_rows*tokens_per_layer*slots } else { 0 };
        let rings = target_rings + mtp_rings;
        fixed.push(Demand::new(gpu as u8, Category::Kv, "SWA512 target + MTP rings with per-sequence step64 slack", rings, Basis::Formula));
        fixed.push(Demand::new(gpu as u8, Category::Kv, "CSA-8 pending BF16 K/Z, two banks per sequence", full*8*256*2*2*2*slots, Basis::Formula));
    }
    let mode = if gpus == 1 { SINGLE } else { LayerMode::HeadSplit };
    let layers = (0..cfg.layers).map(|layer| {
        let record = if cfg.full[layer] { tokens_per_layer*PAGE } else { 0 };
        LayerDemand { kind: if cfg.full[layer] { AttentionClass::Gqa } else { AttentionClass::Swa },
            weights: ModeBytes::default(), kv_unit: KvDemand { unit_bytes_whole: record,
                unit_bytes_split: [record/2,record/2], unit_bytes_context: None },
            fixed_bytes: ModeBytes::default(), extra: Vec::new(), context_indexer: false, colocate: None,
            experts: None, modes: vec![mode] }
    }).collect();
    // Exactly one 256-byte summary per region/layer. Under TP2 it is
    // single-owned (gpu0) with peer readers, never four provider copies.
    let summaries = BoundedDemand { gpu: 0, category: Category::Kv, group: "single-owned CSA index summaries".into(),
        bytes_per_unit: full*(PAGE/8)*256, offset_units: 0, cap_units: u64::MAX, basis: Basis::Formula };
    let request = PlacementRequest {
        inventory: Inventory { gpus: opts.rtx_bytes.iter().map(|bytes| GpuBudget {
            capacity_bytes: *bytes, headroom_bytes: opts.headroom_bytes,
            baseline: Baseline::Planned { context_bytes: 0, loaded_bytes: 0 } }).collect(), spark_ranks: ranks, peer_access: gpus == 2 },
        attention_placement: Some(AttentionPlacement::Heads), context_buffers: ContextBuffers::default(), layers_first_gpu: 0,
        pool: PoolPolicy { compiled_context: opts.context_tokens,
            requested, target: opts.target_pool_tokens.max(FLOOR), floor: FLOOR,
            ceiling: opts.target_pool_tokens.max(requested.unwrap_or(0)).max(FLOOR), unit_rows: PAGE },
        layers, pool_overhead: vec![0;gpus], pool_bounded: vec![summaries], fixed, startup: Vec::new(), movables: vec![],
        expert_workspace: 0, tp2_workspace: [0;2], onboard: Onboard::Auto, expert_gpus: 0, minimum_expert_layers: 0,
        policy: LayerPolicy { default: vec![mode], ..Default::default() },
        hops: HopSpec { storage: HopStorage::ReceiveSlots, row_bytes: cfg.hidden as u64*4, rows: 0, lanes: 0,
            entry_gpu: 0, head_gpu: 0, terminal: if gpus == 1 { TerminalHome::AttentionOwner } else { TerminalHome::Replicated } },
        // This is a hypothetical metadata layout, not an executor capability.
        executor: ExecutorModes { family: "step5 metadata only", modes: MODES, hops: false, attention: AttentionPlacement::Heads },
    };
    match solve(&request) {
        Ok(placed) => {
            memory.pool_tokens = placed.pool_tokens;
            for gpu in 0..gpus {
                memory.devices.push(DeviceLayout { kind: DeviceKind::Rtx, index: gpu as u32,
                    capacity_bytes: opts.rtx_bytes[gpu], items: placed.items[gpu].clone(), kv_tokens: placed.pool_tokens });
            }
            memory.notes.push(format!("Header lower bound fits {} RTX + {ranks} Sparks with BF16 pool {} tokens; actual capacity will be lower after native runtime admission", gpus, placed.pool_tokens));
        }
        Err(error) => {
            refusal(report, format!("Step-5 coordinator BF16 pool lower bound refused: {error}"));
            for gpu in 0..gpus {
                memory.devices.push(DeviceLayout { kind: DeviceKind::Rtx, index: gpu as u32,
                    capacity_bytes: opts.rtx_bytes[gpu], kv_tokens: 0,
                    items: vec![Item::new(Category::Weights, "coordinator source weights", "bf16+fp32", per_gpu, Basis::Formula)] });
            }
        }
    }
    memory
}
