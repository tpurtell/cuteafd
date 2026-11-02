//! Device-bound native programs and residual scheduling for the coordinator.
//! The unbound operation backend still refuses with its owning work item
//! (#273), never substitute arithmetic.
use super::block::{MixSite, Ops, Routes};
use anyhow::{bail, ensure, Context, Result};
use cuteafd_ffi::{programs::Programs, Bf16Add, KimiExpertInput, KimiFfn, KimiFfnBatch,
    KimiKda, KimiKdaCommit, KimiLatent, KimiMlaMode, KimiPrefixMode, KimiResidual,
    KimiResidualBatch, KimiResidualSite, KimiRmsNorm, KimiRouter};
use crate::shared::{memory::device::{Allocation, Device, DeviceOwner, Stream}, prefix::view};
use cuteafd_ffi::{CuteafdDeviceBuffer, programs::VocabularyHead};
use std::rc::Rc;
use super::weights::GlobalWeights;
use cuteafd_core::memory_layout::{Basis, Category};
use cuteafd_loader::{families::kimi_k3::config::KimiK3Config, placement::Demand};
use std::collections::BTreeMap;

/// A device-resident batch: `rows` BF16 rows of `width` in one plane.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Plane {
    pub(crate) ptr: *mut std::ffi::c_void,
    pub(crate) rows: usize,
    pub(crate) width: usize,
}

/// The exported KDA programs (fork #29, exporter geometry `kimik3`, 48 heads
/// per TP2 rank): `kda_m{rows}` writes the head-local `o_norm(o, g)` output;
/// `o_proj` is not part of the program (`w_o` stays in the packed operand set
/// for the binding).
pub(crate) const KDA_PROGRAMS: [&str; 3] = ["kimik3_kda_m64", "kimik3_kda_m4096", "kimik3_kda_commit"];

/// Fixed-arena schedule for the fused residual update/mix/norm programs.
/// FFN output is folded into the next attention site (or the output site),
/// while a block-boundary FFN replaces the prefix with attention output.
pub(crate) fn residual_site(cfg: &KimiK3Config, site: MixSite) -> Result<KimiResidualSite> {
    ensure!(cfg.residual_block > 0 && cfg.max_residual_blocks() <= 8,
        "Kimi residual arena requires at most eight stored blocks");
    let (candidates, prefix, append) = match site {
        MixSite::Attention(layer) => {
            ensure!(layer < cfg.layers, "Kimi residual attention layer {layer} out of range");
            if layer == 0 { (1, KimiPrefixMode::Keep, true) }
            else { (layer.div_ceil(cfg.residual_block) + 1, KimiPrefixMode::Add,
                layer % cfg.residual_block == 0) }
        }
        MixSite::Ffn(layer) => {
            ensure!(layer < cfg.layers, "Kimi residual FFN layer {layer} out of range");
            (layer / cfg.residual_block + 2,
                if layer % cfg.residual_block == 0 { KimiPrefixMode::Replace } else { KimiPrefixMode::Add }, false)
        }
        MixSite::Output => (cfg.max_residual_blocks() + 1, KimiPrefixMode::Add, false),
    };
    Ok(KimiResidualSite { candidates, prefix, append })
}

/// Resolve each used residual specialization and both dense/shared FFNs once.
/// W9 supplies admitted allocations and TP2 projection slices to these calls.
pub(crate) struct W8<'a> {
    residuals: Vec<(KimiResidualSite, KimiResidual<'a>)>,
    dense: KimiFfn<'a>,
    shared: KimiFfn<'a>,
    cfg: KimiK3Config,
}
impl<'a> W8<'a> {
    pub(crate) fn load(cfg: &KimiK3Config, programs: &'a Programs<'a>, mode: cuteafd_ffi::KimiFfnMode) -> Result<Self> {
        ensure!(cfg.hidden == 7168 && cfg.dense_intermediate == 33792
            && cfg.moe_intermediate * cfg.shared_experts == 6144,
            "unsupported Kimi W8 hidden/dense/shared geometry");
        let mut residuals = Vec::new();
        for site in (0..cfg.layers).flat_map(|l| [MixSite::Attention(l), MixSite::Ffn(l)])
            .chain(std::iter::once(MixSite::Output)) {
            let key = residual_site(cfg, site)?;
            if !residuals.iter().any(|(existing, _)| *existing == key) {
                residuals.push((key, KimiResidual::load(programs, key)?));
            }
        }
        Ok(Self { residuals, dense: KimiFfn::load(programs, cfg.dense_intermediate / 2, mode)?,
            shared: KimiFfn::load(programs, cfg.moe_intermediate * cfg.shared_experts / 2, mode)?, cfg: cfg.clone() })
    }

    pub(crate) fn residual(&self, site: MixSite, batch: KimiResidualBatch,
        stream: *mut std::ffi::c_void) -> Result<()> {
        let key = residual_site(&self.cfg, site)?;
        let (_, program) = self.residuals.iter().find(|(existing, _)| *existing == key)
            .ok_or_else(|| anyhow::anyhow!("unresolved Kimi residual site"))?;
        program.launch(batch, stream)?;
        Ok(())
    }

    pub(crate) fn ffn(&self, layer: usize, batch: KimiFfnBatch,
        stream: *mut std::ffi::c_void) -> Result<()> {
        let dense = self.cfg.dense.get(layer).ok_or_else(|| anyhow::anyhow!("Kimi FFN layer out of range"))?;
        if *dense { self.dense.launch(batch, stream)?; } else { self.shared.launch(batch, stream)?; }
        Ok(())
    }
}

/// Explicit packed-path catalogue. Attention keeps its singleton row geometry;
/// dense K6 stages remain conservatively bounded even inside attention producers.
fn packing_inventory(core: Option<&serde_json::Value>) -> Result<Vec<(String, crate::shared::prefill_packing::ProgramRows)>> {
    use crate::shared::prefill_packing::ProgramRows::{PackedRows, SequenceLocal};
    let mut inventory = BTreeMap::new();
    let mut add = |name: String, role| -> Result<()> {
        if let Some(previous) = inventory.insert(name, role) {
            ensure!(previous == role, "Kimi packed program has conflicting row roles");
        }
        Ok(())
    };
    for name in ["kimik3_kda_commit", "kimik3_kda_m64", "kimik3_kda_m4096"] {
        add(name.into(), SequenceLocal)?;
    }
    for heads in [48, 96] {
        let suffix = if heads == 48 { "" } else { "_h96" };
        for capacity in [64, 4096] {
            add(format!("kimik3_mla_output{suffix}_m{capacity}"), SequenceLocal)?;
            add(format!("kda_core_h{heads}_m{capacity}"), SequenceLocal)?;
            add(format!("mla_query_h{heads}_m{capacity}"), SequenceLocal)?;
            add(format!("mla_gate_output_h{heads}_m{capacity}"), SequenceLocal)?;
            for cache in ["", "_fp8"] {
                add(format!("kimik3_mla_producer{suffix}{cache}_m{capacity}"), SequenceLocal)?;
                for stride in [576, 640] {
                    add(format!("mla_rank_norm_h{heads}_kv{stride}{cache}_m{capacity}"), SequenceLocal)?;
                }
                for mode in if capacity == 64 { &["decode", "verify"][..] } else { &["extend"][..] } {
                    add(format!("kimik3_dense_mla_{mode}{suffix}{cache}_m{capacity}"), SequenceLocal)?;
                }
            }
        }
    }
    // The drafter consumes each member's private context after packed target work.
    add("kimik3_mla_draft_output_m64".into(), SequenceLocal)?;
    for cache in ["", "_fp8"] {
        add(format!("kimik3_mla_draft_producer{cache}_m64"), SequenceLocal)?;
        for mode in ["full", "w4096"] {
            add(format!("kimik3_mla_draft_attention_{mode}{cache}_m64"), SequenceLocal)?;
        }
    }
    add("kimik3_attn_res_c1_keep_append".into(), PackedRows)?;
    for candidates in 2..=9 {
        for prefix in ["add", "replace", "keep"] {
            add(format!("kimik3_attn_res_c{candidates}_{prefix}"), PackedRows)?;
        }
        if candidates <= 8 { add(format!("kimik3_attn_res_c{candidates}_add_append"), PackedRows)?; }
    }
    for intermediate in [3072, 6144, 16896, 33792] {
        for stem in ["situ", "ffn_gate_up", "ffn_down"] {
            add(format!("kimik3_{stem}_i{intermediate}"), PackedRows)?;
        }
        add(format!("kimik3_ffn_gate_up_i{intermediate}_comp"), PackedRows)?;
    }
    for name in ["kimik3_rmsnorm_h3584", "kimik3_rmsnorm_h7168", "kimik3_latent_down",
        "kimik3_latent_up", "kimik3_router_scores", "kimik3_expert_input_quant"] {
        add(name.into(), PackedRows)?;
    }
    if let Some(core) = core {
        let policies = core["row_policies"].as_array().context("Kimi K6 row policies missing")?;
        for value in policies {
            let policy = cuteafd_ffi::dense_trellis::DenseTrellisPolicy::parse(value)?;
            for name in [policy.native_input, policy.native, policy.output_program] {
                add(format!("dense_trellis_k6_{name}"), PackedRows)?;
            }
            if let Some(grouped) = cuteafd_ffi::dense_trellis::GroupedPolicy::parse(value)? {
                for name in [grouped.native_input, grouped.native, grouped.output_program] {
                    add(name, PackedRows)?;
                }
            }
        }
    }
    Ok(inventory.into_iter().collect())
}

fn packing_program(name: &str, core: bool) -> bool {
    name.starts_with("kimik3_") || core && (name.starts_with("kda_core_")
        || name.starts_with("mla_rank_norm_") || name.starts_with("mla_query_")
        || name.starts_with("mla_gate_output_") || name.starts_with("dense_trellis_k6_"))
}

fn classify_packing_inventory<'a>(names: impl IntoIterator<Item = &'a str>,
    catalogue: &'a [(String, crate::shared::prefill_packing::ProgramRows)])
    -> Result<Vec<(&'a str, crate::shared::prefill_packing::ProgramRows)>> {
    let names = names.into_iter().collect::<std::collections::BTreeSet<_>>();
    let inventory = catalogue.iter().filter(|(name, _)| names.contains(name.as_str()))
        .map(|(name, role)| (name.as_str(), *role)).collect::<Vec<_>>();
    crate::shared::prefill_packing::validate_program_inventory(names.iter().copied(), &inventory)?;
    Ok(inventory)
}

fn coordinator_packing_cap(programs: &Programs<'_>, core: bool, rows: usize) -> Result<usize> {
    let catalogue = packing_inventory(core.then(|| programs.family_metadata("dense_trellis_k6")).transpose()?)?;
    let inventory = classify_packing_inventory(programs.names().filter(|name| packing_program(name, core)), &catalogue)?;
    crate::shared::prefill_packing::coordinator_row_cap(programs, inventory, rows)
}

/// Native module handles for one coordinator device. Load on each actual rank
/// before measuring its CUDA baseline; this allocates no family tensor storage.
/// The shared layer executor, not this table, owns cross-device ordering.
pub(crate) struct NativePrograms<'a> {
    pub device: Device<'a>,
    pub w8: W8<'a>,
    pub norm: KimiRmsNorm<'a>,
    pub kda: [KimiKda<'a>; 2],
    pub commit: KimiKdaCommit<'a>,
    pub mla: [MlaPrograms<'a>; 2],
    pub latent: KimiLatent<'a>,
    pub router: KimiRouter<'a>,
    pub wire: KimiExpertInput<'a>,
    pub add: Bf16Add<'a>,
    pub core:Option<super::core_runtime::CorePrograms<'a>>,
    pub(crate) packing_cap: usize,
    scratch: NativeScratch,
    wire_grid: crate::shared::launch_grid::Fp8QuantizeGrid,
}

pub(crate) struct MlaPrograms<'a> {
    pub decode: super::mla::Mla<'a>,
    pub verify: super::mla::Mla<'a>,
    pub extend: super::mla::Mla<'a>,
}

/// Reusable native temporary storage for one serialized layer lane. These are
/// validated manifest requirements, not estimated total-engine allowances.
pub(crate) use cuteafd_loader::families::kimi_k3::allocation::NativeScratch;

impl<'a> NativePrograms<'a> {
    pub(crate) fn load(device: Device<'a>, cfg: &KimiK3Config,
        programs: &'a Programs<'a>, ffn_mode: cuteafd_ffi::KimiFfnMode, cache_format: cuteafd_core::MlaCacheFormat) -> Result<Self> {
        Self::load_with_core(device, cfg, programs, ffn_mode, cache_format, &Default::default(), 4096)
    }
    pub(crate) fn load_with_core(device: Device<'a>, cfg: &KimiK3Config, programs: &'a Programs<'a>,
        ffn_mode: cuteafd_ffi::KimiFfnMode, cache_format: cuteafd_core::MlaCacheFormat,
        inventory: &cuteafd_loader::families::kimi_k3::core::CoreInventory, rows: usize) -> Result<Self> {
        ensure!(device.id >= 0 && std::ptr::eq(device.library, programs.library()),
            "Kimi programs and device must share their native library");
        let packing_cap = coordinator_packing_cap(programs, !inventory.projections.is_empty(), rows)?;
        device.run(|| {
            let kda = [KimiKda::load(programs, 64, true)?, KimiKda::load(programs, 4096, false)?];
            // Admission can select owned MLA or explicit head split. Resolve
            // both before its measured baseline, never lazily after admission.
            let load_mla = |heads| -> Result<MlaPrograms<'a>> {
                Ok(MlaPrograms {
                    decode: super::mla::Mla::load(cfg, programs, heads, 64, KimiMlaMode::Decode, cache_format)?,
                    verify: super::mla::Mla::load(cfg, programs, heads, 64, KimiMlaMode::Verify, cache_format)?,
                    extend: super::mla::Mla::load(cfg, programs, heads, 4096, KimiMlaMode::Extend, cache_format)?,
                })
            };
            let mla = [load_mla(48)?, load_mla(96)?];
            let scratch = NativeScratch::combine(kda.each_ref().map(|p| p.scratch_bytes()),
                mla.iter().flat_map(|p| [p.decode.scratch_bytes(), p.verify.scratch_bytes(), p.extend.scratch_bytes()]))?;
            Ok(Self { device, w8: W8::load(cfg, programs, ffn_mode)?, norm: KimiRmsNorm::load(programs, cfg.hidden)?,
                kda, commit: KimiKdaCommit::load(programs)?, mla, latent: KimiLatent::load(programs)?,
                router: KimiRouter::load(programs)?, wire: KimiExpertInput::load(programs)?,
                add: device.library.bf16_add()?, core:super::core_runtime::CorePrograms::load(programs,cfg,inventory,rows,cache_format)?, packing_cap, scratch,
                wire_grid: crate::shared::launch_grid::Fp8QuantizeGrid::new(device.library.sm_count()?, None)? })
        })
    }

    pub(crate) fn scratch(&self) -> NativeScratch { self.scratch }

    pub(super) fn check_config(&self, cfg: &KimiK3Config) -> Result<()> {
        ensure!(*cfg == self.w8.cfg, "Kimi native programs differ from admitted checkpoint config");
        Ok(())
    }

    pub(super) fn check_inventory(&self, cfg: &KimiK3Config) -> Result<()> {
        super::admission::check_prefix(&self.w8.cfg, cfg)
    }

    /// `rows` lane rows of units of at most `unit` rows (the programs' launch capacity).
    pub(crate) fn lane_layout(&self, cfg: &KimiK3Config, rank: usize, rows: usize, unit: usize, max_context: usize) -> Result<LaneLayout> {
        self.check_config(cfg)?;
        ensure!(rank < 2, "Kimi lane logical rank outside TP2");
        ensure!(self.mla.iter().all(|p| [p.decode.max_context(), p.verify.max_context(), p.extend.max_context()]
            .into_iter().all(|capacity| max_context <= capacity)), "Kimi lane context exceeds native programs");
        let mut layout=LaneLayout::new(cfg, rows, unit, max_context, self.scratch)?;
        if let Some(core)=&self.core {
            layout.core=core.layouts[rank].clone();
            // Split producers consume the same named scratch planes as fused
            // producers; admission retains the larger exact requirement.
            let kda=core.kda.iter().map(|p|p.scratch_bytes()).max().unwrap_or(0);
            let mla=core.mla.values().map(|p|p.scratch_bytes()).max().unwrap_or(0);
            let stages = cuteafd_loader::families::kimi_k3::core::lane_extents(rows, unit,
                layout.operands["KDA scratch"],layout.operands["MLA producer scratch"],kda,mla)?;
            layout.base = layout.base.with_core_stages(stages)?;
            layout.bytes()?;
        }
        Ok(layout)
    }
}

/// Named storage for one lane, shared by its serialized layer operations.
/// Cache/state pools, prefix marks, head I/O and peer/expert exchange are separate
/// owners; these bytes must not be used as a total-engine allowance.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct LaneLayout {
    base: cuteafd_loader::families::kimi_k3::allocation::NativeLaneLayout,
    core: Option<crate::shared::dense_trellis::ProjectionLayout>,
}

impl std::ops::Deref for LaneLayout {
    type Target = cuteafd_loader::families::kimi_k3::allocation::NativeLaneLayout;
    fn deref(&self) -> &Self::Target { &self.base }
}
impl std::ops::DerefMut for LaneLayout {
    fn deref_mut(&mut self) -> &mut Self::Target { &mut self.base }
}

impl LaneLayout {
    pub(crate) fn new(cfg: &KimiK3Config, rows: usize, unit: usize, max_context: usize,
        scratch: NativeScratch) -> Result<Self> {
        Ok(Self { base: cuteafd_loader::families::kimi_k3::allocation::NativeLaneLayout::new(
            cfg, rows, unit, max_context, scratch)?, core: None })
    }

    pub(crate) fn bytes(&self) -> Result<usize> {
        let base=self.arena_bytes.checked_add(self.core.as_ref().map_or(Ok(0),|core|core.bytes())?)
            .context("Kimi K6 lane scratch total overflow")?;
        self.operands.iter().filter(|(name, _)| !self.arena_offsets.contains_key(*name))
            .try_fold(base, |sum, (_, bytes)| sum.checked_add(*bytes)
                .ok_or_else(|| anyhow::anyhow!("Kimi lane total extent overflow")))
    }

    pub(crate) fn rows(&self) -> usize { self.rows }

    pub(crate) fn max_context(&self) -> usize { self.max_context }

    pub(crate) fn demand(&self, rank: u8, lane: usize) -> Result<Demand> {
        ensure!(rank < 2 && lane < 2, "Kimi lane rank/index outside TP2 two-lane executor");
        Ok(Demand::new(rank, Category::Workspace, format!("Kimi lane {lane} native operands"),
            u64::try_from(self.bytes()?)?, Basis::Exact))
    }

    pub(super) fn allocate<'a>(&self, device: Device<'a>) -> Result<Lane<'a>> {
        self.allocate_on(device, Rc::new(Stream::new(device)?))
    }

    pub(super) fn allocate_on<'a>(&self, device: Device<'a>, stream: Rc<Stream<'a>>) -> Result<Lane<'a>> {
        ensure!(stream.device.id == device.id && std::ptr::eq(stream.device.library, device.library),
            "Kimi admitted lane stream differs from device");
        let _scope = cuteafd_ffi::memory_ledger::scope("workspace/kimi lane");
        let arena = Allocation::new(device, self.arena_bytes)?;
        let operands = self.operands.iter().filter(|(name, _)| !self.arena_offsets.contains_key(*name))
            .map(|(name, bytes)| {
                let allocation = Allocation::new(device, *bytes)?;
                ensure!(allocation.buffer.bytes == *bytes && allocation.buffer.device_id == device.id,
                    "Kimi allocated lane operand {name} differs from admitted extent/device");
                Ok((*name, allocation))
            }).collect::<Result<BTreeMap<_, _>>>()?;
        let core = self.core.as_ref().map(|layout| layout.allocate(device)).transpose()?;
        let lane = Lane { stream, operands, arena: Some(arena), layout: self.clone(), window: None, metadata_bank: 0, core };
        lane.prime_attention()?;
        Ok(lane)
    }
}

/// Stream drops first and drains queued work before any operand is released,
/// including cancellation/unwind. No allocation occurs during a lane launch.
pub(crate) struct Lane<'a> {
    pub stream: std::rc::Rc<Stream<'a>>,
    operands: BTreeMap<&'static str, Allocation<'a>>,
    arena: Option<Allocation<'a>>,
    layout: LaneLayout,
    window: Option<std::ops::Range<usize>>,
    metadata_bank: usize,
    core: Option<crate::shared::dense_trellis::ProjectionScratch<'a>>,
}

impl Drop for Lane<'_> {
    fn drop(&mut self) {
        // A shared state owner also retains the stream; do not defer the drain
        // until its last reference drops while these operands are freed now.
        self.stream.drain_or_abort("Kimi lane operands");
    }
}

fn subwave_extent(name: &str, bytes: usize, capacity: usize, window: &std::ops::Range<usize>)
    -> Result<(usize, usize)> {
    ensure!(capacity > 0 && window.start < window.end && window.end <= capacity,
        "Kimi sub-wave operand {name} differs from admitted row layout");
    if cuteafd_loader::families::kimi_k3::allocation::HOOK_SCRATCH.contains(&name) {
        return Ok((0, bytes));
    }
    ensure!(bytes % capacity == 0, "Kimi row plane {name} stride differs from admission");
    let stride = bytes / capacity;
    Ok((window.start * stride, window.len() * stride))
}

impl Lane<'_> {
    pub(crate) fn buffer(&self, name: &str) -> Result<cuteafd_ffi::CuteafdDeviceBuffer> {
        if let Some(&offset) = self.layout.arena_offsets.get(name) {
            let arena = self.arena.as_ref().ok_or_else(|| anyhow::anyhow!("Kimi lane arena absent"))?;
            return view(arena.buffer, offset, self.layout.operands[name]);
        }
        let buffer = self.operands.get(name).map(|allocation| allocation.buffer)
            .ok_or_else(|| anyhow::anyhow!("unknown Kimi lane operand {name}"))?;
        if ATTENTION_METADATA.contains(&name) {
            let (offset, bytes) = self.layout.metadata_extent(name, self.metadata_bank)?;
            return view(buffer, offset, bytes);
        }
        match &self.window {
            Some(window) => {
                let (offset, bytes) = subwave_extent(name, self.layout.operands[name], self.layout.rows, window)?;
                view(buffer, offset, bytes)
            }
            _ => Ok(buffer),
        }
    }

    pub(crate) fn rows(&self) -> usize {
        self.window.as_ref().map_or(self.layout.rows, |window| window.len())
    }

    fn prime_attention(&self) -> Result<()> {
        for bank in 0..self.layout.metadata_banks() {
            for name in ATTENTION_METADATA {
                let (offset, bytes) = self.layout.metadata_extent(name, bank)?;
                let buffer = view(self.operands[name].buffer, offset, bytes)?;
                self.stream.device.run(|| {
                    // SAFETY: fresh admitted destinations and pinned sources stay
                    // live through the copy; all banks are primed before capture.
                    unsafe { self.stream.device.library.copy_h2d_ordered(buffer, &vec![0; buffer.bytes], self.stream.raw) }
                })?;
            }
        }
        self.stream.drain()
    }

    fn upload_attention(&self, metadata: &AttentionRows) -> Result<()> {
        metadata.check_layout(&self.layout)?;
        self.stream.device.run(|| {
            for (name, bytes) in metadata.uploads() {
                // SAFETY: this lane retains the fixed destination and stream;
                // shared staging retains the pinned source outside capture.
                // This is an ordered copy fence, not a compute drain.
                unsafe { self.stream.device.library.copy_h2d_ordered(self.buffer(name)?, bytes, self.stream.raw)?; }
            }
            Ok(())
        })
    }
}

/// Persistent engine owner, not a per-layer guard. Retain the exact admitted
/// weights/programs until teardown drains the lane; native enqueue never waits.
/// Metadata staging outside capture uses the shared ordered-copy source fence.
/// Peer ordering and complete Spark combination belong to the shared executor.
pub(crate) struct LocalLane<'a> {
    lane: Lane<'a>,
    programs: Rc<NativePrograms<'a>>,
    cfg: KimiK3Config,
    layers: Vec<Rc<super::weights::LayerWeights<'a>>>,
    global: Rc<GlobalWeights<'a>>,
    lane_index: usize,
    rank: usize,
    kda: Rc<super::state::KdaBank<'a>>,
    mla: Rc<super::mla::CacheBank<'a>>,
    attention: Option<AttentionShape>,
    graphs: Option<super::graphs::Graphs<'a>>,
    replay_enabled: bool,
}

impl Drop for LocalLane<'_> {
    fn drop(&mut self) {
        self.lane.stream.drain_or_abort("Kimi persistent local lane");
        drop(self.graphs.take());
    }
}

impl<'a> LocalLane<'a> {
    pub(super) fn new(lane: Lane<'a>, programs: Rc<NativePrograms<'a>>, cfg: KimiK3Config,
        layers: Vec<Rc<super::weights::LayerWeights<'a>>>, global: Rc<GlobalWeights<'a>>,
        lane_index: usize, rank: usize, kda: Rc<super::state::KdaBank<'a>>, mla: Rc<super::mla::CacheBank<'a>>) -> Result<Self> {
        ensure!(rank < 2, "Kimi local rank outside TP2 admission");
        let device = lane.stream.device;
        ensure!(programs.device.id == device.id && std::ptr::eq(programs.device.library, device.library),
            "Kimi local programs differ from admitted lane device");
        programs.check_inventory(&cfg)?;
        ensure!(layers.len() == cfg.layers, "Kimi local layer inventory differs from admitted prefix");
        for weights in &layers { weights.check_device(device)?; }
        global.check_device(device)?;
        kda.check_reader(lane_index, &lane.stream)?;
        mla.check_reader(lane_index, &lane.stream)?;
        Ok(Self { lane, programs, cfg, layers, global, lane_index, rank, kda, mla, attention: None, graphs: None, replay_enabled: true })
    }

    fn layer(&self, layer: usize) -> Result<&super::weights::LayerWeights<'a>> {
        self.layers.get(layer).map(Rc::as_ref).ok_or_else(|| anyhow::anyhow!("Kimi local layer outside inventory"))
    }

    fn plane(&self, name: &str, rows: usize, width: usize) -> Result<CuteafdDeviceBuffer> {
        ensure!((1..=self.lane.rows()).contains(&rows), "Kimi local rows exceed admitted lane");
        view(self.lane.buffer(name)?, 0, rows * width * 2)
    }

    pub(crate) fn stream(&self) -> &Rc<Stream<'a>> { &self.lane.stream }

    /// External scheduling/staging uses only the admitted lane's actual owners.
    pub(crate) fn buffer(&self, name: &str) -> Result<CuteafdDeviceBuffer> { self.lane.buffer(name) }

    pub(super) fn rows(&self) -> usize { self.lane.rows() }
    pub(super) fn lane_index(&self) -> usize { self.lane_index }
    pub(super) fn config(&self) -> &KimiK3Config { &self.cfg }
    pub(super) fn max_context(&self) -> usize { self.lane.layout.max_context() }
    pub(super) fn prefix_banks(&self) -> (Rc<super::state::KdaBank<'a>>, Rc<super::mla::CacheBank<'a>>) {
        (self.kda.clone(), self.mla.clone())
    }

    pub(super) fn attention_share(&self, layer: usize) -> Result<crate::shared::placement::layers::AttentionShare> {
        Ok(self.layer(layer)?.attention)
    }

    /// Report resident operands and banks, not just the intended placement.
    pub(super) fn attention_resident(&self, layer: usize) -> Result<(bool, bool)> {
        use cuteafd_loader::families::kimi_k3::config::KimiAttention;
        let weights = self.layer(layer)?;
        Ok(match self.config().attention.get(layer) {
            Some(KimiAttention::Kda) => (weights.kda().is_ok() || weights.is_core("kda.output"), self.kda.layer(layer, self.lane_index).is_ok()),
            Some(KimiAttention::Mla) => (weights.mla().is_ok() || weights.is_core("mla.output"), self.mla.layer(layer).is_ok()),
            None => bail!("Kimi attention census layer outside inventory"),
        })
    }

    /// Copies a completed peer/local result into a fixed admitted operand.
    /// Producers must already be ordered on this exact lane stream.
    pub(super) fn adopt(&mut self, name: &str, source: CuteafdDeviceBuffer, rows: usize, width: usize) -> Result<()> {
        let destination = self.plane(name, rows, width)?;
        ensure!(source.device_id == destination.device_id && source.bytes >= destination.bytes,
            "Kimi adopted result device or extent differs from lane");
        io_disjoint(source, destination)?;
        self.stream().device.run(|| {
            // SAFETY: the shared executor retains the source and orders its
            // producer before this lane; destination is a retained lane plane.
            unsafe { self.stream().device.library.copy_d2d_async(
                destination, source, destination.bytes, self.stream().raw) }
        })
    }

    /// Cold diagnostic window reuse on the exact owner stream. The caller must
    /// order prior consumers before resetting; this never releases state storage.
    pub(crate) fn reset_cold_state(&self) -> Result<()> {
        self.stream().device.run(|| {
            for (layer, attention) in self.cfg.attention.iter().enumerate() {
                if *attention != cuteafd_loader::families::kimi_k3::config::KimiAttention::Kda { continue; }
                for buffer in self.kda.slot(layer, 0)? {
                    // SAFETY: retained slot zero belongs to this lane and all
                    // previous work is ordered before this exact stream.
                    unsafe { self.stream().device.library.cuda_zero_bytes_async(buffer, buffer.bytes, self.stream().raw)?; }
                }
            }
            Ok(())
        })
    }

    /// Outside capture: replace metadata contents, never the admitted addresses.
    /// The scheduler retains pages/slots; graph keys include the KDA sequence
    /// count because it is a launch scalar, unlike the mutable table contents.
    pub(crate) fn stage_attention(&mut self, sequences: &[SequenceRows<'_>],
        pool: &cuteafd_engine::prefix::RefPagePool) -> Result<()> {
        self.attention = None;
        self.lane.metadata_bank = 0;
        ensure!(!pool.is_parity() && pool.capacity().checked_add(pool.reserved()) == Some(self.mla.pages()),
            "Kimi metadata page pool differs from admitted MLA bank");
        let metadata = AttentionRows::new(sequences, self.lane.layout.unit, self.kda.slots(),
            self.lane.layout.max_context, pool)?;
        self.lane.upload_attention(&metadata)?;
        self.attention = Some(AttentionShape { rows: metadata.rows, sequences: metadata.sequences });
        Ok(())
    }

    /// Attention alone selects member row views. Residual mixing and FFNs retain
    /// the packed shape; recurrent slots and MLA page tables never cross members.
    pub(crate) fn packed_attention(&self, sequences: &[SequenceRows<'_>],
        pool: &cuteafd_engine::prefix::RefPagePool) -> Result<Vec<AttentionRows>> {
        ensure!(!pool.is_parity() && pool.capacity().checked_add(pool.reserved()) == Some(self.mla.pages()),
            "Kimi packed page pool differs from admitted MLA bank");
        AttentionRows::packed(sequences, self.lane.layout.unit, self.kda.slots(), self.max_context(), pool)
    }

    pub(crate) fn attention_packed(&mut self, layer: usize, members: &[AttentionRows]) -> Result<()> {
        let window = self.lane.window.clone();
        let shape = self.attention;
        let bank = self.lane.metadata_bank;
        let result = (|| {
            let mut first = 0;
            for metadata in members {
                self.select_subwave(Some(first..first + metadata.rows))?;
                self.lane.metadata_bank = 0;
                self.lane.upload_attention(metadata)?;
                self.attention = Some(AttentionShape { rows: metadata.rows, sequences: 1 });
                self.attention(layer, metadata.rows, false)?;
                first += metadata.rows;
            }
            Ok(())
        })();
        self.lane.window = window;
        self.lane.metadata_bank = bank;
        self.attention = shape;
        result
    }

    /// Stage immutable metadata in a disjoint admitted bank for every unit.
    /// Repeated layers only select views; they never upload or fence the host.
    pub(crate) fn stage_attention_units(&mut self, units: &[crate::shared::prefill_pipeline::Unit],
        sequence: &SequenceRows<'_>, pool: &cuteafd_engine::prefix::RefPagePool) -> Result<()> {
        ensure!(!units.is_empty() && units.len() <= self.lane.layout.metadata_banks(),
            "Kimi attention units exceed admitted metadata banks");
        ensure!(!pool.is_parity() && pool.capacity().checked_add(pool.reserved()) == Some(self.mla.pages()),
            "Kimi metadata page pool differs from admitted MLA bank");
        crate::shared::prefill_pipeline::with_cleanup(self, |local| {
            for (bank, unit) in units.iter().enumerate() {
                ensure!(unit.index == bank && unit.sequence == 0 && unit.rows.start < unit.rows.end
                    && unit.rows.end <= sequence.rows, "Kimi attention unit metadata extent");
                let metadata = AttentionRows::new(&[SequenceRows { slot: sequence.slot,
                    start: sequence.start + unit.rows.start, rows: unit.rows.len(), pages: sequence.pages }],
                    local.lane.layout.unit, local.kda.slots(), local.lane.layout.max_context, pool)?;
                local.lane.metadata_bank = bank;
                local.lane.upload_attention(&metadata)?;
            }
            Ok(())
        }, |local| { local.lane.metadata_bank = 0; local.attention = None; Ok(()) })
    }

    pub(crate) fn warm_graphs(&mut self, keys: &[super::graphs::Key], rank: usize) -> Result<()> {
        use super::graphs::Op;
        ensure!(self.graphs.is_none() && keys.iter().all(|key|
            rank == self.rank && key.lane == self.lane_index && key.rank == rank && key.rows <= self.rows()),
            "Kimi startup graph keys differ from local binding");
        let stream = self.stream().clone();
        let staged = self.attention.take();
        let graphs = super::graphs::Graphs::capture(&stream, keys, |key| {
            match key.op {
                Op::Residual(site) => { self.residual(site, key.rows)?; }
                Op::Ffn(layer) => { self.ffn(layer, key.rows)?; }
                Op::Attention { layer, sequences, speculative } => {
                    self.attention = Some(AttentionShape { rows: key.rows, sequences });
                    self.attention(layer, key.rows, speculative)?;
                }
            }
            Ok(())
        });
        self.attention = staged;
        self.graphs = Some(graphs?);
        Ok(())
    }

    pub(crate) fn commit_kda(&self, tables: &[i32]) -> Result<()> {
        ensure!(!tables.is_empty() && tables.len() % 3 == 0 && tables.len() / 3 <= self.kda.slots(),
            "Kimi accepted replay table outside admission");
        let [state, conv, replay, table] = self.kda.commit_operands(self.lane_index)?;
        let bytes = tables.iter().flat_map(|value| value.to_le_bytes()).collect::<Vec<_>>();
        let layers = self.cfg.attention.iter().filter(|kind|
            **kind == cuteafd_loader::families::kimi_k3::config::KimiAttention::Kda).count();
        let stream = self.stream();
        stream.device.run(|| {
            // SAFETY: ordered copy fence, not a compute drain. The admitted table
            // and replay remain live on the exact producer/commit stream.
            unsafe { stream.device.library.copy_h2d_ordered(table, &bytes, stream.raw)?; }
            self.programs.commit.launch(tables.len() / 3, layers, self.kda.slots(),
                state, conv, replay, table, stream.raw)?;
            Ok(())
        })
    }

    /// Views partition existing persistent row operands. Phase scratch and metadata
    /// are reused only on this exact serialized stream; no allocation is added.
    pub(crate) fn select_subwave(&mut self, window: Option<std::ops::Range<usize>>) -> Result<()> {
        ensure!(window.as_ref().is_none_or(|w| w.start < w.end && w.end <= self.lane.layout.rows),
            "Kimi sub-wave outside admitted storage");
        self.lane.window = window;
        Ok(())
    }

    pub(crate) fn clear_subwave(&mut self) {
        self.lane.window = None;
        self.lane.metadata_bank = 0;
    }

    pub(crate) fn select_attention_unit(&mut self, bank: usize, rows: usize) -> Result<()> {
        self.lane.layout.metadata_extent("KDA slots", bank)?;
        ensure!(rows > 0 && rows <= self.lane.layout.unit, "Kimi attention unit rows");
        self.lane.metadata_bank = bank;
        self.attention = Some(AttentionShape { rows, sequences: 1 });
        Ok(())
    }

    pub(crate) fn enable_replay(&mut self, enabled: bool) { self.replay_enabled = enabled; }
    pub(crate) fn graph_replays(&self) -> u64 { self.graphs.as_ref().map_or(0, |graphs| graphs.replays()) }
    /// Diagnostic A/B only, before startup graphs are captured.
    pub(crate) fn set_grouped_kda(&self, enabled: bool) -> Result<()> {
        ensure!(self.graphs.is_none(), "Kimi grouped KDA selection after graph capture");
        if let Some(core) = &self.programs.core { core.set_grouped(enabled); }
        Ok(())
    }

    fn replay(&mut self, rows: usize, op: super::graphs::Op) -> Result<bool> {
        if !self.replay_enabled || self.lane.window.is_some() { return Ok(false); }
        let rank = self.rank;
        let key = super::graphs::Key { lane: self.lane_index, rank, rows, op };
        let stream = self.stream().clone();
        self.graphs.as_mut().map_or(Ok(false), |graphs| graphs.launch(key, &stream))
    }

    /// Enqueue only rank-local attention. KDA and head-split MLA return a TP2
    /// partial; whole MLA returns the owner's complete delta. The shared layer
    /// executor owns the meet/publication and recurrent-slot ordering.
    pub(crate) fn attention(&mut self, layer: usize, rows: usize, speculative: bool) -> Result<CuteafdDeviceBuffer> {
        let shape = self.attention.ok_or_else(|| anyhow::anyhow!("Kimi attention metadata not staged"))?;
        ensure!(shape.rows == rows && (!speculative || rows <= 64), "Kimi attention staged row shape/speculation");
        let sequences = if self.cfg.attention[layer] == cuteafd_loader::families::kimi_k3::config::KimiAttention::Kda {
            shape.sequences
        } else { 1 };
        // Speculative keys replay the verify programs (replay records, verify MLA);
        // slots, pages and frontiers are device metadata staged outside capture.
        if self.replay(rows, super::graphs::Op::Attention { layer, sequences, speculative })? {
            return self.attention_result(layer,rows);
        }
        let weights = self.layer(layer)?;
        if self.native_attention(layer)? {return self.attention_core(layer,rows,speculative);}
        let output = self.plane("attention output", rows, 7168)?;
        let device = self.lane.stream.device;
        match self.programs.w8.cfg.attention.get(layer) {
            Some(cuteafd_loader::families::kimi_k3::config::KimiAttention::Kda) => {
                let [state, conv_state, replay] = self.kda.layer(layer, self.lane_index)?;
                let weights = weights.kda()?;
                let batch = cuteafd_ffi::KimiKdaBatch { rows, sequences: shape.sequences, state_slots: self.kda.slots(),
                    speculative, input: self.plane("normalized", rows, 7168)?, state, conv_state,
                    slots: self.lane.buffer("KDA slots")?, first: self.lane.buffer("KDA first")?,
                    normalized: self.lane.buffer("KDA normalized")?, output,
                    replay: (rows <= 64).then_some(replay), scratch: self.lane.buffer("KDA scratch")? };
                device.run(|| { self.programs.kda[usize::from(rows > 64)].launch(weights, batch, self.lane.stream.raw)?; Ok(()) })?;
            }
            Some(cuteafd_loader::families::kimi_k3::config::KimiAttention::Mla) => {
                let programs = &self.programs.mla[usize::from(weights.attention == crate::shared::placement::layers::AttentionShare::Whole)];
                let program = if rows > 64 { &programs.extend } else if speculative { &programs.verify } else { &programs.decode };
                let weights = weights.mla()?;
                let batch = cuteafd_ffi::KimiMlaBatch { rows, requests: if rows <= 64 { rows } else { 1 },
                    page_table_width: self.lane.layout.max_context.div_ceil(64), cache_pages: self.mla.pages(),
                    active_splits: program.splits(), x: self.plane("normalized", rows, 7168)?,
                    slots: self.lane.buffer("MLA slots")?, cache: self.mla.layer(layer)?,
                    page_table: self.lane.buffer("MLA page table")?, lengths: self.lane.buffer("MLA lengths")?,
                    query_offsets: self.lane.buffer("MLA query offsets")?, query: self.lane.buffer("MLA query")?,
                    q_resid: self.lane.buffer("MLA q residual")?, attention: self.lane.buffer("MLA attention")?,
                    lse: self.lane.buffer("MLA LSE")?, output,
                    producer_scratch: self.lane.buffer("MLA producer scratch")?,
                    attention_scratch: self.lane.buffer("MLA attention scratch")?,
                    output_scratch: self.lane.buffer("MLA output scratch")? };
                device.run(|| program.forward(layer, weights, batch, self.lane.stream.raw))?;
            }
            None => bail!("Kimi attention layer outside inventory"),
        }
        Ok(output)
    }

    pub(crate) fn native_attention(&self,layer:usize)->Result<bool> {
        let weights=self.layer(layer)?;
        Ok(weights.is_core("kda.output") || weights.is_core("mla.output"))
    }
    pub(crate) fn native_ffn(&self,layer:usize)->Result<bool> {Ok(self.layer(layer)?.is_core("ffn.down"))}
    fn attention_result(&self,layer:usize,rows:usize)->Result<CuteafdDeviceBuffer> {
        if self.native_attention(layer)? && self.layer(layer)?.attention==crate::shared::placement::layers::AttentionShare::Split {
            view(self.lane.buffer("K6 attention partial")?,0,rows*7168*2)
        } else {self.plane("attention output",rows,7168)}
    }
    fn ffn_result(&self,layer:usize,rows:usize)->Result<CuteafdDeviceBuffer> {
        if self.native_ffn(layer)? {view(self.lane.buffer("K6 FFN partial")?,0,rows*7168*2)}
        else {self.plane("ffn output",rows,7168)}
    }
    fn attention_core(&self,layer:usize,rows:usize,speculative:bool)->Result<CuteafdDeviceBuffer> {
        use cuteafd_ffi::{KimiKdaCoreWeights,KimiKdaCoreBatch,KimiMlaRankBatch};
        use cuteafd_loader::families::kimi_k3::config::KimiAttention;
        let weights=self.layer(layer)?;
        let programs=self.programs.core.as_ref().context("K6 programs not resolved before admission")?;
        let scratch=self.lane.core.as_ref().context("K6 scratch not admitted")?;
        let projected=|name|programs.projected(weights.core(name)?,scratch);
        let input=self.plane("normalized",rows,7168)?;
        let normalized=self.lane.buffer("K6 normalized")?;
        let shape=self.attention.context("Kimi attention metadata not staged")?;
        let stream=self.stream();
        stream.device.run(|| {
            let output_name=match self.cfg.attention[layer] {
                KimiAttention::Kda=> {
                    let [state,conv_state,replay]=self.kda.layer(layer,self.lane_index)?;
                    let batch=KimiKdaCoreBatch{rows,sequences:shape.sequences,state_slots:self.kda.slots(),speculative,
                        input,q:self.lane.buffer("K6 q")?,k:self.lane.buffer("K6 k")?,v:self.lane.buffer("K6 v")?,
                        gate:self.lane.buffer("K6 gate")?,state,conv_state,slots:self.lane.buffer("KDA slots")?,
                        first:self.lane.buffer("KDA first")?,output:normalized,replay,scratch:self.lane.buffer(
                            if self.lane.layout.operands.contains_key("K6 KDA scratch") {"K6 KDA scratch"}else{"KDA scratch"})?};
                    let core_weights=KimiKdaCoreWeights{fab:weights.buffer("kda.fab")?,forget:weights.buffer("kda.forget")?,
                        convolution:weights.buffer("kda.conv")?,a_log:weights.buffer("kda.a_log")?,
                        dt_bias:weights.buffer("kda.dt_bias")?,norm:weights.buffer("kda.norm")?};
                    super::core_runtime::kda_projected(&programs.kda[usize::from(rows>64)],
                        [projected("kda.q")?,projected("kda.k")?,projected("kda.v")?,projected("kda.gate")?],
                        programs.grouped(weights.core("kda.q")?),core_weights,batch,stream.raw)?;
                    "kda.output"
                }
                KimiAttention::Mla=> {
                    let whole=weights.attention==crate::shared::placement::layers::AttentionShare::Whole;
                    let heads=if whole {96}else{48};
                    let capacity=if rows>64 {4096}else{64};
                    let stride=weights.core("mla.kv_a")?.shape[0];
                    let split=programs.mla.get(&(heads,stride,capacity)).context("K6 MLA split shape not resolved")?;
                    let legacy=&self.programs.mla[usize::from(whole)];
                    let attention=if rows>64 {&legacy.extend}else if speculative {&legacy.verify}else{&legacy.decode};
                    let batch=cuteafd_ffi::KimiMlaBatch{rows,requests:if rows<=64 {rows}else{1},
                        page_table_width:self.lane.layout.max_context.div_ceil(64),cache_pages:self.mla.pages(),
                        active_splits:attention.splits(),x:input,slots:self.lane.buffer("MLA slots")?,cache:self.mla.layer(layer)?,
                        page_table:self.lane.buffer("MLA page table")?,lengths:self.lane.buffer("MLA lengths")?,
                        query_offsets:self.lane.buffer("MLA query offsets")?,query:self.lane.buffer("MLA query")?,
                        q_resid:self.lane.buffer("MLA q residual")?,attention:self.lane.buffer("MLA attention")?,
                        lse:self.lane.buffer("MLA LSE")?,output:self.plane("attention output",rows,7168)?,
                        producer_scratch:self.lane.buffer("MLA producer scratch")?,
                        attention_scratch:self.lane.buffer("MLA attention scratch")?,output_scratch:self.lane.buffer("MLA output scratch")?};
                    super::core_runtime::mla_rank_projected(split,projected("mla.q_a")?,projected("mla.kv_a")?,input,
                        KimiMlaRankBatch{rows,cache_pages:batch.cache_pages,q_a:self.lane.buffer("K6 q_a")?,
                            kv_a:self.lane.buffer("K6 kv_a")?,q_norm:weights.buffer("mla.q_norm")?,
                            kv_norm:weights.buffer("mla.kv_norm")?,slots:batch.slots,q_resid:batch.q_resid,cache:batch.cache,
                            scratch:self.lane.buffer(if self.lane.layout.operands.contains_key("K6 MLA scratch") {
                                "K6 MLA scratch"}else{"MLA producer scratch"})?},stream.raw)?;
                    super::core_runtime::mla_query_projected(split,projected("mla.q_b")?,rows,batch.q_resid,
                        self.lane.buffer("K6 q_b")?,weights.buffer("mla.uk")?,batch.query,stream.raw)?;
                    attention.attention(layer,batch,stream.raw)?;
                    super::core_runtime::mla_gate_projected(split,projected("mla.gate")?,rows,input,batch.attention,
                        weights.buffer("mla.uv")?,self.lane.buffer("K6 gate")?,normalized,stream.raw)?;
                    "mla.output"
                }
            };
            let weight=weights.core(output_name)?;
            let projection=programs.projection(weight)?;
            let output=self.attention_result(layer,rows)?;
            if weights.attention==crate::shared::placement::layers::AttentionShare::Whole {
                projection.local(rows,normalized,weight,scratch,output,stream.raw)?;
            } else {
                let partial=projection.core(rows,normalized,weight,scratch,stream.raw)?;
                crate::shared::dense_trellis::retain_partial(&programs.narrow,partial,output,rows*7168,stream.raw)?;
            }
            Ok(output)
        })
    }
    fn ffn_core(&self,layer:usize,rows:usize)->Result<CuteafdDeviceBuffer> {
        let weights=self.layer(layer)?;
        let programs=self.programs.core.as_ref().context("K6 programs not resolved")?;
        let scratch=self.lane.core.as_ref().context("K6 scratch not admitted")?;
        let projected=|name|programs.projected(weights.core(name)?,scratch);
        let output=self.ffn_result(layer,rows)?;
        self.stream().device.run(|| {
            let partial=super::core_runtime::ffn_projected(projected("ffn.gate")?,projected("ffn.up")?,projected("ffn.down")?,
                &programs.situ,&programs.copy,rows,self.plane("normalized",rows,7168)?,self.lane.buffer("K6 FFN gate")?,
                self.lane.buffer("K6 FFN up")?,self.lane.buffer("gate-up")?,self.lane.buffer("activated")?,self.stream().raw)?;
            crate::shared::dense_trellis::retain_partial(&programs.narrow,partial,output,rows*7168,self.stream().raw)?;
            Ok(output)
        })
    }
    /// Both ranks narrow independently before publication. Sum their BF16
    /// partials in rank order into F32, then run exactly one H128 egress.
    pub(crate) fn reduce_core(&self,layer:usize,ffn:bool,rows:usize,peer:CuteafdDeviceBuffer,
        destination:&str,stream:*mut std::ffi::c_void)->Result<()> {
        let weights=self.layer(layer)?;
        let name=if ffn {"ffn.down"}else if self.cfg.attention[layer]==cuteafd_loader::families::kimi_k3::config::KimiAttention::Kda {
            "kda.output"
        }else{"mla.output"};
        let weight=weights.core(name)?;
        let programs=self.programs.core.as_ref().context("K6 programs not resolved")?;
        let local=if ffn {self.ffn_result(layer,rows)?}else{self.attention_result(layer,rows)?};
        let (rank0,rank1)=if self.rank==0 {(local,peer)}else{(peer,local)};
        programs.projection(weight)?.reduced(&programs.core_add,rows,rank0,rank1,self.lane.buffer("K6 reduced")?,
            weight,self.plane(destination,rows,7168)?,stream)
    }

    /// Seed only the mutable prefix from admitted embedded/delta rows. Slot 0
    /// is written by the first fused residual specialization, not a host copy.
    pub(crate) fn seed(&mut self, rows: usize) -> Result<()> {
        let source = self.plane("delta", rows, 7168)?;
        let prefix = view(self.lane.buffer("residual arena")?, 8 * self.lane.rows() * 7168 * 2, source.bytes)?;
        self.lane.stream.device.run(|| {
            // SAFETY: disjoint admitted allocations retained by this persistent owner.
            unsafe { self.lane.stream.device.library.copy_d2d_async(prefix, source, source.bytes, self.lane.stream.raw) }
        })
    }

    /// A decoder layer returns prefix plus its final FFN delta, before the
    /// model's output attention-residual mix or final normalization.
    pub(crate) fn terminal_residual(&mut self, rows: usize) -> Result<CuteafdDeviceBuffer> {
        let delta = self.plane("delta", rows, 7168)?;
        let prefix = view(self.lane.buffer("residual arena")?,
            8 * self.lane.rows() * 7168 * 2, delta.bytes)?;
        let output = self.plane("normalized", rows, 7168)?;
        self.lane.stream.device.run(|| {
            // SAFETY: the exact persistent lane owns all three disjoint planes.
            unsafe { self.programs.add.launch(prefix, delta, output, rows * 7168, self.lane.stream.raw) }
        })?;
        Ok(output)
    }

    pub(crate) fn residual(&mut self, site: MixSite, rows: usize) -> Result<CuteafdDeviceBuffer> {
        if self.replay(rows, super::graphs::Op::Residual(site))? {
            return self.plane("normalized", rows, 7168);
        }
        let (score_weight, norm_weight) = match site {
            MixSite::Attention(layer) => (self.layer(layer)?.buffer("attention.score")?, self.layer(layer)?.buffer("attention.norm")?),
            MixSite::Ffn(layer) => (self.layer(layer)?.buffer("ffn.score")?, self.layer(layer)?.buffer("ffn.norm")?),
            MixSite::Output => (self.global.buffer("output.score")?, self.global.buffer("output.norm")?),
        };
        let output = self.plane("normalized", rows, 7168)?;
        let batch = KimiResidualBatch { rows, arena_rows: self.lane.rows(), arena: self.lane.buffer("residual arena")?,
            delta: self.plane("delta", rows, 7168)?, score_weight, norm_weight, output };
        self.lane.stream.device.run(|| self.programs.w8.residual(site, batch, self.lane.stream.raw))?;
        Ok(output)
    }

    /// Returns a TP2 partial. The shared executor meets both ranks before this
    /// partial can become the next fused residual's delta.
    pub(crate) fn ffn(&mut self, layer: usize, rows: usize) -> Result<CuteafdDeviceBuffer> {
        if self.replay(rows, super::graphs::Op::Ffn(layer))? {
            return self.ffn_result(layer,rows);
        }
        let weights = self.layer(layer)?;
        if weights.is_core("ffn.down") {return self.ffn_core(layer,rows);}
        let output = self.plane("ffn output", rows, 7168)?;
        let batch = KimiFfnBatch { rows, input: self.plane("normalized", rows, 7168)?,
            gate_up_weight: weights.buffer("ffn.gate_up")?, down_weight: weights.buffer("ffn.down")?,
            gate_up: self.lane.buffer("gate-up")?, activated: self.lane.buffer("activated")?, output };
        self.lane.stream.device.run(|| self.programs.w8.ffn(layer, batch, self.lane.stream.raw))?;
        Ok(output)
    }

    /// Route and quantize on this device; normal RouteStaging/SparkLane reads
    /// these admitted outputs. No CPU Vec<Routes> enters the operator hot path.
    pub(crate) fn route(&mut self, layer: usize, rows: usize) -> Result<(cuteafd_ffi::KimiRoutes, CuteafdDeviceBuffer)> {
        let weights = self.layer(layer)?;
        ensure!(self.programs.w8.cfg.dense.get(layer) == Some(&false), "Kimi dense layer has no routes");
        let input = self.plane("normalized", rows, 7168)?;
        let latent = self.plane("latent input", rows, 3584)?;
        let wire = view(self.lane.buffer("expert wire")?, 0, rows * KimiExpertInput::ROW_BYTES)?;
        let routes = cuteafd_ffi::KimiRoutes { ids: view(self.lane.buffer("route ids")?, 0, rows * 16 * 4)?,
            weights: view(self.lane.buffer("route weights")?, 0, rows * 16 * 4)? };
        let logits = self.lane.buffer("router logits")?;
        let router = weights.buffer("router.weight")?;
        let bias = weights.buffer("router.bias")?;
        self.lane.stream.device.run(|| {
            self.programs.router.launch(rows, input, router, bias, logits, routes, self.lane.stream.raw)?;
            if weights.is_core("latent.down") {
                let programs=self.programs.core.as_ref().context("K6 latent programs not resolved")?;
                let scratch=self.lane.core.as_ref().context("K6 latent scratch not admitted")?;
                let down=weights.core("latent.down")?;
                programs.projection(down)?.local(rows,input,down,scratch,latent,self.lane.stream.raw)?;
            } else {
                self.programs.latent.down(rows,input,weights.buffer("latent.down")?,latent,self.lane.stream.raw)?;
            }
            self.programs.wire.launch(rows, self.programs.wire_grid.blocks(rows, 3584), latent, wire, self.lane.stream.raw)?;
            Ok(())
        })?;
        Ok((routes, wire))
    }

    /// Invoke only after all six Spark partials have been reduced into the
    /// admitted expert-combined plane and ordered before this lane's stream.
    pub(crate) fn latent_up(&mut self, layer: usize, rows: usize) -> Result<CuteafdDeviceBuffer> {
        let weights = self.layer(layer)?;
        let combined = self.plane("expert combined", rows, 3584)?;
        let normalized = self.plane("latent normalized", rows, 3584)?;
        let output = self.plane("routed output", rows, 7168)?;
        let norm_weight = weights.buffer("latent.norm")?;
        self.lane.stream.device.run(|| {
            if weights.is_core("latent.up") {
                let programs=self.programs.core.as_ref().context("K6 latent programs not resolved")?;
                let scratch=self.lane.core.as_ref().context("K6 latent scratch not admitted")?;
                programs.norm.launch(rows,combined,norm_weight,normalized,self.lane.stream.raw)?;
                let up=weights.core("latent.up")?;
                programs.projection(up)?.local(rows,normalized,up,scratch,output,self.lane.stream.raw)?;
            } else {
                self.programs.latent.up(rows,combined,norm_weight,normalized,weights.buffer("latent.up")?,output,self.lane.stream.raw)?;
            }
            Ok(())
        })?;
        Ok(output)
    }

    /// Combine routed output with the executor's completed shared-FFN meet
    /// in `shared output`, rounding once before this can become residual delta.
    pub(crate) fn combine_moe(&mut self, rows: usize) -> Result<CuteafdDeviceBuffer> {
        let a = self.plane("routed output", rows, 7168)?;
        let b = self.plane("shared output", rows, 7168)?;
        let output = self.plane("combined output", rows, 7168)?;
        self.lane.stream.device.run(|| {
            // SAFETY: the persistent owner retains operands and stream; shared add rejects aliases.
            unsafe { self.programs.add.launch(a, b, output, rows * 7168, self.lane.stream.raw) }
        })?;
        Ok(output)
    }
}

/// A scheduler-owned sequence stretch. Pages come from the shared RefPagePool;
/// a copied partial tail must be private before any append targets it.
pub(crate) struct SequenceRows<'p> {
    pub slot: usize,
    pub start: usize,
    pub rows: usize,
    pub pages: &'p [u32],
}

use cuteafd_loader::families::kimi_k3::allocation::ATTENTION_METADATA;

#[derive(Clone, Copy)]
struct AttentionShape {
    rows: usize,
    sequences: usize,
}

/// Host metadata is rebuilt outside capture, then copied into fixed lane buffers.
/// For <=64 rows, MLA treats each query as one request with its own visible
/// length, so the native request scalar depends only on rows, not partitions.
pub(crate) struct AttentionRows {
    rows: usize,
    sequences: usize,
    width: usize,
    kda_slots: Vec<u8>,
    kda_first: Vec<u8>,
    mla_slots: Vec<u8>,
    page_table: Vec<u8>,
    lengths: Vec<u8>,
    offsets: Vec<u8>,
}

impl AttentionRows {
    fn packed(sequences: &[SequenceRows<'_>], capacity: usize, slots: usize, max_context: usize,
        pool: &cuteafd_engine::prefix::RefPagePool) -> Result<Vec<Self>> {
        ensure!(!sequences.is_empty() && sequences.len() <= slots, "Kimi packed member count");
        let mut used = std::collections::BTreeSet::new();
        let mut pages = BTreeMap::<u32, bool>::new();
        let mut total = 0usize;
        let mut members = Vec::with_capacity(sequences.len());
        for sequence in sequences {
            ensure!(used.insert(sequence.slot), "Kimi packed recurrent slot repeats");
            total = total.checked_add(sequence.rows).context("Kimi packed rows overflow")?;
            ensure!(total <= capacity, "Kimi packed rows exceed admitted unit");
            let metadata = Self::new(std::slice::from_ref(sequence), capacity, slots, max_context, pool)?;
            for (index, &page) in sequence.pages[..(sequence.start + sequence.rows).div_ceil(64)].iter().enumerate() {
                let writable = index >= sequence.start / 64;
                if let Some(previous) = pages.insert(page, writable) {
                    ensure!(!previous && !writable, "Kimi packed members alias a writable page");
                }
            }
            members.push(metadata);
        }
        Ok(members)
    }

    fn new(sequences: &[SequenceRows<'_>], capacity: usize, slots: usize, max_context: usize,
        pool: &cuteafd_engine::prefix::RefPagePool) -> Result<Self> {
        ensure!((1..=4096).contains(&capacity) && (1..=cuteafd_core::concurrency::state_slots(cuteafd_core::concurrency::MAX_CONCURRENCY) as usize).contains(&slots)
            && (1..=1_048_576).contains(&max_context), "Kimi row descriptor capacity");
        ensure!(!pool.is_parity() && pool.page_rows() == 64, "Kimi rows require shared uniform 64-row pages");
        ensure!(!sequences.is_empty() && sequences.len() <= slots, "Kimi row descriptor sequence count");
        let rows = sequences.iter().try_fold(0usize, |n, sequence| n.checked_add(sequence.rows)
            .ok_or_else(|| anyhow::anyhow!("Kimi row descriptor overflow")))?;
        ensure!(rows > 0 && rows <= capacity && (rows <= 64 || sequences.len() == 1),
            "Kimi wide prefill requires one sequence and admitted rows");
        let width = max_context.div_ceil(64);
        let requests = if rows <= 64 { rows } else { 1 };
        let mut metadata = Self { rows, sequences: sequences.len(), width,
            kda_slots: Vec::with_capacity(rows * 4), kda_first: Vec::with_capacity(rows * 4),
            mla_slots: Vec::with_capacity(rows * 8), page_table: vec![0; requests * width * 4],
            lengths: Vec::with_capacity(requests * 4), offsets: Vec::with_capacity((requests + 1) * 4) };
        let mut used_slots = std::collections::BTreeSet::new();
        let mut page_users = BTreeMap::<u32, (usize, bool)>::new();
        let mut row = 0;
        metadata.offsets.extend_from_slice(&0i32.to_le_bytes());
        for (index, sequence) in sequences.iter().enumerate() {
            ensure!(sequence.rows > 0 && sequence.slot < slots && used_slots.insert(sequence.slot),
                "Kimi sequence slots repeat or exceed admission");
            let end = sequence.start.checked_add(sequence.rows).ok_or_else(|| anyhow::anyhow!("Kimi row position overflow"))?;
            ensure!(end <= max_context && sequence.pages.len() >= end.div_ceil(64),
                "Kimi sequence context/pages too short");
            let pages = &sequence.pages[..end.div_ceil(64)];
            let mut unique = std::collections::BTreeSet::new();
            for (position, &page) in pages.iter().enumerate() {
                ensure!((page as usize) < pool.capacity() + pool.reserved() && page as usize >= pool.reserved()
                    && unique.insert(page), "Kimi sequence page outside pool or repeated");
                let writable = position >= sequence.start / 64;
                ensure!(pool.refs(page) > 0 && (!writable || pool.refs(page) == 1),
                    "Kimi write requires an allocated private page (copy shared tails first)");
                if let Some(&(previous, previous_writable)) = page_users.get(&page) {
                    ensure!(previous == index || (!writable && !previous_writable),
                        "Kimi requests alias a writable page");
                }
                page_users.insert(page, (index, writable));
            }
            let first = row;
            for offset in 0..sequence.rows {
                metadata.kda_slots.extend_from_slice(&(sequence.slot as i32).to_le_bytes());
                metadata.kda_first.extend_from_slice(&(first as i32).to_le_bytes());
                let position = sequence.start + offset;
                let physical = i64::from(pages[position / 64]) * 64 + (position % 64) as i64;
                metadata.mla_slots.extend_from_slice(&physical.to_le_bytes());
                if rows <= 64 {
                    metadata.lengths.extend_from_slice(&((position + 1) as i32).to_le_bytes());
                    metadata.offsets.extend_from_slice(&((row + 1) as i32).to_le_bytes());
                    for (j, page) in pages.iter().enumerate() {
                        let at = (row * width + j) * 4;
                        metadata.page_table[at..at + 4].copy_from_slice(&page.to_le_bytes());
                    }
                }
                row += 1;
            }
            if rows > 64 {
                metadata.lengths.extend_from_slice(&(end as i32).to_le_bytes());
                metadata.offsets.extend_from_slice(&(rows as i32).to_le_bytes());
                for (j, page) in pages.iter().enumerate() {
                    metadata.page_table[j * 4..j * 4 + 4].copy_from_slice(&page.to_le_bytes());
                }
            }
        }
        Ok(metadata)
    }

    fn check_layout(&self, layout: &LaneLayout) -> Result<()> {
        ensure!(self.rows <= layout.unit && self.width == layout.max_context.div_ceil(64),
            "Kimi metadata differs from admitted row capacity/page stride");
        for (name, bytes) in self.uploads() {
            let (_, capacity) = layout.metadata_extent(name, 0)?;
            ensure!(bytes.len() <= capacity, "Kimi metadata exceeds admitted operand {name}");
        }
        Ok(())
    }

    fn uploads(&self) -> [(&'static str, &[u8]); 6] {
        [("KDA slots", &self.kda_slots), ("KDA first", &self.kda_first), ("MLA slots", &self.mla_slots),
            ("MLA page table", &self.page_table), ("MLA lengths", &self.lengths), ("MLA query offsets", &self.offsets)]
    }
}

/// Head handles/workspaces are lane-local: independent compute streams must
/// never enqueue concurrent work through the same cuBLAS handle or workspace.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct IoLayout {
    rank: usize,
    rows: usize,
    hidden: usize,
    vocab: usize,
    core_head: bool,
}

impl IoLayout {
    pub(crate) fn new(cfg: &KimiK3Config, rank: usize, rows: usize) -> Result<Self> {
        ensure!(cfg.hidden == 7168 && cfg.vocab_size == 163840, "Kimi native IO hidden/vocabulary geometry");
        ensure!(rank < 2 && (1..=cuteafd_loader::families::kimi_k3::allocation::LANE_ROWS_MAX).contains(&rows),
            "Kimi IO rank/row geometry");
        Ok(Self { rank, rows, hidden: cfg.hidden, vocab: cfg.vocab_size, core_head:false })
    }

    pub(crate) fn with_core(cfg:&KimiK3Config,rank:usize,rows:usize,
        core:&cuteafd_loader::families::kimi_k3::core::CoreInventory)->Result<Self> {
        let mut layout=Self::new(cfg,rank,rows)?;
        layout.core_head=rank==1 && core.get_weight("language_model.lm_head").is_some();
        Ok(layout)
    }

    fn extents(&self) -> [usize; 3] {
        cuteafd_loader::families::kimi_k3::allocation::io_extents(
            self.vocab, self.rank, self.rows, self.core_head)
    }

    pub(crate) fn demand(&self, lane: usize) -> Result<Demand> {
        ensure!(lane < 2, "Kimi IO lane outside admission");
        Ok(Demand::new(self.rank as u8, Category::Workspace, format!("Kimi lane {lane} token IO"),
            self.extents().into_iter().sum::<usize>() as u64, Basis::Exact))
    }

    pub(super) fn allocate<'a>(&self, device: Device<'a>, stream: Rc<Stream<'a>>,
        weights: Rc<GlobalWeights<'a>>) -> Result<TokenIo<'a>> {
        ensure!(stream.device.id == device.id && std::ptr::eq(stream.device.library, device.library),
            "Kimi IO stream differs from admitted lane device");
        weights.check_device(device)?;
        ensure!(weights.core_head.is_some()==self.core_head,"Kimi head storage differs from admitted IO format");
        if !self.core_head {weights.buffer(if self.rank == 0 { "embedding" } else { "head" })?;}
        let _scope = cuteafd_ffi::memory_ledger::scope("workspace/kimi token IO");
        let [ids, logits, workspace] = self.extents();
        let allocate = |bytes| -> Result<Option<Allocation<'a>>> {
            if bytes == 0 { return Ok(None); }
            let allocation = Allocation::new(device, bytes)?;
            ensure!(allocation.buffer.bytes == bytes && allocation.buffer.device_id == device.id,
                "Kimi IO allocation differs from admitted extent/device");
            Ok(Some(allocation))
        };
        let ids = allocate(ids)?;
        let logits = allocate(logits)?;
        let workspace = allocate(workspace)?;
        let head = workspace.as_ref().map(|workspace| device.own(|| {
            // SAFETY: lane-local workspace remains after the head and all stream readers drain.
            unsafe { device.library.vocabulary_head_rows(workspace.buffer.ptr, self.hidden as u32,
                self.rows.min(64) as u32, self.vocab as u32) }
        })).transpose()?;
        Ok(TokenIo { stream, head, workspace, ids, logits, weights, layout: self.clone() })
    }
}

pub(crate) struct TokenIo<'a> {
    stream: Rc<Stream<'a>>,
    head: Option<DeviceOwner<'a, VocabularyHead<'a>>>,
    workspace: Option<Allocation<'a>>,
    ids: Option<Allocation<'a>>,
    logits: Option<Allocation<'a>>,
    weights: Rc<GlobalWeights<'a>>,
    layout: IoLayout,
}

impl Drop for TokenIo<'_> {
    fn drop(&mut self) { self.stream.drain_or_abort("Kimi token IO readers"); }
}

impl TokenIo<'_> {
    pub(super) fn check_lane(&self, local: &LocalLane<'_>, rank: usize) -> Result<()> {
        ensure!(self.layout.rank == rank && self.layout.rows == local.rows()
            && Rc::ptr_eq(&self.stream, local.stream()) && Rc::ptr_eq(&self.weights, &local.global),
            "Kimi token IO differs from exact admitted local lane/global owner");
        Ok(())
    }

    pub(crate) fn embed(&self, tokens: &[u32], output: CuteafdDeviceBuffer) -> Result<()> {
        ensure!(self.layout.rank == 0 && !tokens.is_empty() && tokens.len() <= self.layout.rows
            && tokens.iter().all(|&id| (id as usize) < self.layout.vocab), "Kimi embedding row/token outside admission");
        let bytes = tokens.len() * self.layout.hidden * 2;
        ensure!(output.device_id == self.stream.device.id && !output.ptr.is_null() && output.bytes >= bytes,
            "Kimi embedding output extent/device");
        let ids = self.ids.as_ref().ok_or_else(|| anyhow::anyhow!("Kimi entry token IDs absent"))?.buffer;
        let table = self.weights.buffer("embedding")?;
        let output = view(output, 0, bytes)?;
        io_disjoint(output, ids)?;
        io_disjoint(output, table)?;
        let data = tokens.iter().flat_map(|id| id.to_le_bytes()).collect::<Vec<_>>();
        self.stream.device.run(|| {
            // SAFETY: ordered staging retains host data; this owner retains IDs, table and stream.
            unsafe { self.stream.device.library.copy_h2d_ordered(ids, &data, self.stream.raw)?; }
            // SAFETY: checked IDs, table and output extents; caller retains output through the lane drain.
            unsafe { self.stream.device.library.cuda_embedding_lookup_bf16_async(table, ids, output,
                tokens.len(), self.layout.vocab, self.layout.hidden, self.stream.raw) }
        })
    }

    /// Reuses at most 64 logits rows. Callers consume/download each span before
    /// the next projection; wide prefill never allocates a full-context logit tensor.
    pub(crate) fn project(&self, input: CuteafdDeviceBuffer, rows: usize, local:&LocalLane<'_>) -> Result<CuteafdDeviceBuffer> {
        ensure!(self.layout.rank == 1 && (1..=self.layout.rows.min(64)).contains(&rows),
            "Kimi head rows exceed admitted IO capacity");
        self.check_lane(local,1)?;
        if let Some(weight)=&self.weights.core_head {
            let programs=local.programs.core.as_ref().context("K6 head programs not resolved")?;
            let scratch=local.lane.core.as_ref().context("K6 head scratch not admitted")?;
            let logits=self.logits.as_ref().context("K6 logits not admitted")?.buffer;
            self.stream.device.run(||programs.projection(weight)?.local(rows,input,weight,scratch,logits,self.stream.raw))?;
            return view(logits,0,rows*self.layout.vocab*4);
        }
        let head = self.head.as_ref().ok_or_else(|| anyhow::anyhow!("Kimi terminal head absent"))?;
        let logits = self.logits.as_ref().ok_or_else(|| anyhow::anyhow!("Kimi terminal logits absent"))?.buffer;
        ensure!(input.device_id == self.stream.device.id && !input.ptr.is_null()
            && input.bytes >= rows * self.layout.hidden * 2, "Kimi head input extent/device");
        let weight = self.weights.buffer("head")?;
        let input = view(input, 0, rows * self.layout.hidden * 2)?;
        io_disjoint(input, logits)?;
        io_disjoint(input, weight)?;
        let workspace = self.workspace.as_ref().ok_or_else(|| anyhow::anyhow!("Kimi terminal head workspace absent"))?;
        io_disjoint(input, workspace.buffer)?;
        self.stream.device.run(|| {
            // SAFETY: checked live input and rank-local weight/logits; the lane-local handle
            // and workspace remain owned until the stream drains. Caller retains input.
            unsafe { crate::shared::target_head::launch(crate::shared::target_head::TargetHead::configured(),
                head.get(), input.ptr.cast_const(), weight.ptr.cast_const(), logits.ptr.cast(), rows, self.stream.raw) }
        })?;
        view(logits, 0, rows * self.layout.vocab * 4)
    }
}

fn io_disjoint(a: CuteafdDeviceBuffer, b: CuteafdDeviceBuffer) -> Result<()> {
    let start_a = a.ptr as usize;
    let start_b = b.ptr as usize;
    let end_a = start_a.checked_add(a.bytes).ok_or_else(|| anyhow::anyhow!("Kimi IO address overflow"))?;
    let end_b = start_b.checked_add(b.bytes).ok_or_else(|| anyhow::anyhow!("Kimi IO address overflow"))?;
    ensure!(a.device_id == b.device_id && start_a != 0 && start_b != 0
        && (end_a <= start_b || end_b <= start_a), "Kimi IO operands overlap or differ in device");
    Ok(())
}

/// Which export each op waits on.
pub(crate) const PENDING: &[(&str, &str)] = &[
    ("residual_mix", "W8 shared residual mixer"),
    ("norm", "W8/W9 RMSNorm at hidden 7168 (fused with the residual mix)"),
    ("kda", "W6 exports kimik3_kda_m64 (decode/verify, replay), kimik3_kda_m4096 (prefill) and kimik3_kda_commit; \
        their output is the head-local gated norm, then o_proj by columns and the TP2 partial-sum meet (W9 binding)"),
    ("mla", "W7 NoPE MLA producer/dense attention/gated output exports and mla::Mla caller; W9 binds owned cache/weights and TP2 meet"),
    ("dense_ffn", "W8 Situ dense FFN"),
    ("route", "W5 sigmoid router E896/K16, scale 1.0"),
    ("latent_down", "W9 BF16 GEMM 7168->3584"),
    ("experts", "W4 EXL3 kimik3:exl3-k23 packages + W5 reducer width 3584"),
    ("latent_up", "W9 RMSNorm(3584) + BF16 GEMM 3584->7168"),
    ("shared_ffn", "W8 Situ shared FFN (TP2 over intermediate 6144)"),
    ("add", "W9 BF16 add (fused into the next mix)"),
];

/// The not-yet-bound coordinator backend.
pub(crate) struct Pending;

fn pending<T>(op: &str) -> Result<T> {
    let owner = PENDING.iter().find(|(name, _)| *name == op).map_or("unassigned", |(_, owner)| owner);
    bail!("Kimi K3 {op}: no coordinator export bound yet ({owner}; #273)")
}

impl Ops for Pending {
    type Rows = Plane;
    fn residual_mix(&mut self, _: MixSite, _: &[&Plane], _: &Plane) -> Result<Plane> { pending("residual_mix") }
    fn norm(&mut self, _: Option<usize>, _: bool, _: &Plane) -> Result<Plane> { pending("norm") }
    fn kda(&mut self, _: usize, _: &Plane) -> Result<Plane> { pending("kda") }
    fn mla(&mut self, _: usize, _: &Plane) -> Result<Plane> { pending("mla") }
    fn dense_ffn(&mut self, _: usize, _: &Plane) -> Result<Plane> { pending("dense_ffn") }
    fn route(&mut self, _: usize, _: &Plane) -> Result<Vec<Routes>> { pending("route") }
    fn latent_down(&mut self, _: usize, _: &Plane) -> Result<Plane> { pending("latent_down") }
    fn experts(&mut self, _: usize, _: &Plane, _: &[Routes]) -> Result<Plane> { pending("experts") }
    fn latent_up(&mut self, _: usize, _: &Plane) -> Result<Plane> { pending("latent_up") }
    fn shared_ffn(&mut self, _: usize, _: &Plane) -> Result<Plane> { pending("shared_ffn") }
    fn add(&mut self, _: &Plane, _: &Plane) -> Result<Plane> { pending("add") }
}

#[cfg(test)]
mod packing_inventory_tests {
    use super::*;
    use crate::shared::prefill_packing::ProgramRows::{PackedRows, SequenceLocal};

    #[test]
    fn packed_coordinator_catalogue_classifies_k6_and_rejects_new_programs() -> Result<()> {
        let core = serde_json::json!({"row_policies": [{"shape":"shared", "input_features":7168,
            "output_features":3072, "sharded_side":1, "codebook":"mcg",
            "native_input":"input", "native_program":"core", "output_program":"output",
            "grouped":{"groups":4, "max_rows":4096, "native_input":"input_g4",
                "native_program":"core_g4", "output_program":"output_g4"}}]});
        let catalogue = packing_inventory(Some(&core))?;
        let inventory = classify_packing_inventory(catalogue.iter().map(|(name, _)| name.as_str()), &catalogue)?;
        let roles = inventory.iter().copied().collect::<BTreeMap<_, _>>();
        for name in ["kimik3_router_scores", "kimik3_ffn_gate_up_i16896", "kimik3_rmsnorm_h7168",
            "dense_trellis_k6_input", "dense_trellis_k6_core", "dense_trellis_k6_output",
            "dense_trellis_k6_input_g4", "dense_trellis_k6_core_g4", "dense_trellis_k6_output_g4"] {
            assert_eq!(roles[name], PackedRows, "{name}");
        }
        for name in ["kimik3_kda_m64", "kimik3_kda_m4096", "kda_core_h48_m4096",
            "kimik3_dense_mla_extend_h96_fp8_m4096", "mla_rank_norm_h48_kv640_fp8_m64",
            "mla_query_h96_m4096", "kimik3_mla_draft_attention_full_m64"] {
            assert_eq!(roles[name], SequenceLocal, "{name}");
        }
        for name in ["kimik3_new_projection", "dense_trellis_k6_new_projection", "mla_query_new"] {
            assert!(packing_program(name, true));
            assert!(classify_packing_inventory(inventory.iter().map(|(name, _)| *name)
                .chain(std::iter::once(name)), &catalogue).is_err());
        }
        let missing = catalogue.iter().filter(|(name, _)| name != "kimik3_router_scores")
            .cloned().collect::<Vec<_>>();
        assert!(classify_packing_inventory(inventory.iter().map(|(name, _)| *name), &missing).is_err());
        assert!(!packing_program("glm_stage_router", true));
        assert!(!packing_program("dense_trellis_k6_core", false));
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cuteafd_ffi::programs::VOCABULARY_HEAD_WORKSPACE;
    use cuteafd_loader::families::kimi_k3::allocation::{SCRATCH_PHASES, ARENA_ALIGNMENT};

    fn i32_metadata(bytes: &[u8]) -> Vec<i32> {
        bytes.chunks_exact(4).map(|value| i32::from_le_bytes(value.try_into().unwrap())).collect()
    }

    #[test]
    fn subwave_views_partition_persistent_rows_and_preserve_residual_planes() -> Result<()> {
        let cfg = KimiK3Config::from_hf(&cuteafd_loader::plan::testing::kimi_k3_config())?;
        for (rows, unit, waves) in [(4096, 1408, 1), (8192, 4096, 2), (8256, 2752, 3), (8192, 2048, 4)] {
            let mut layout = LaneLayout::new(&cfg, rows, unit, 1_048_576, NativeScratch { kda: 16, mla: [16; 3] })?;
            layout.base=layout.base.with_core_stages(cuteafd_loader::families::kimi_k3::core::
                lane_extents(rows,unit,16,16,16,16)?)?;
            let windows = crate::shared::prefill_pipeline::prefill_units(rows, unit, waves, 256)?;
            for name in cuteafd_loader::families::kimi_k3::allocation::CORE_MEET_PLANES {
                let element=if name=="K6 reduced" {4}else{2};
                assert_eq!(layout.operands[name],rows*7168*element);
                for window in &windows {
                    assert_eq!(subwave_extent(name,layout.operands[name],rows,window)?,
                        (window.start*7168*element,window.len()*7168*element));
                }
            }
            assert_eq!(windows.len(), if waves == 1 { 3 } else { waves });
            for (&name, &bytes) in &layout.operands {
                if layout.arena_offsets.contains_key(name) || ATTENTION_METADATA.contains(&name) { continue; }
                if cuteafd_loader::families::kimi_k3::allocation::HOOK_SCRATCH.contains(&name) {
                    for window in &windows {
                        assert_eq!(subwave_extent(name, bytes, layout.rows, window)?, (0, bytes));
                    }
                    // Negative control: row-plane addressing must not be used for scratch.
                    assert_ne!(subwave_extent(name, bytes, layout.rows, &windows[1])?,
                        (windows[1].start * bytes / rows, windows[1].len() * bytes / rows));
                    continue;
                }
                let mut end = 0;
                for window in &windows {
                    let (offset, len) = subwave_extent(name, bytes, layout.rows, window)?;
                    assert_eq!(offset, end, "{name} views overlap or leave a gap");
                    assert_eq!(offset % 16, 0, "{name} alignment");
                    if name == "residual arena" {
                        let plane_bytes = window.len() * 7168 * 2;
                        assert_eq!(len, 9 * plane_bytes);
                        assert_eq!(offset + 8 * plane_bytes + plane_bytes, offset + len);
                    }
                    end = offset + len;
                }
                assert_eq!(end, bytes, "{name} admission extent changed");
            }
        }
        assert!(subwave_extent("invalid", 4095, 4096, &(0..256)).is_err());
        Ok(())
    }

    #[test]
    fn three_wave_metadata_preserves_exact_unpadded_tail_rows() -> Result<()> {
        let cfg = KimiK3Config::from_hf(&cuteafd_loader::plan::testing::kimi_k3_config())?;
        let layout = LaneLayout::new(&cfg, 8256, 2752, 16384, NativeScratch { kda: 16, mla: [16; 3] })?;
        for rows in [65usize, 1375, 1376, 2751, 2752] {
            let start = 63;
            let mut pool = cuteafd_engine::prefix::RefPagePool::with_reserved(256, 64, 1);
            let pages = pool.alloc((start + rows).div_ceil(64))?;
            let metadata = AttentionRows::new(&[SequenceRows { slot: 0, start, rows, pages: &pages }],
                2752, 1, 16384, &pool)?;
            metadata.check_layout(&layout)?;
            assert_eq!(metadata.rows, rows);
            assert_eq!(metadata.kda_slots.len(), rows * 4);
            assert_eq!(metadata.kda_first.len(), rows * 4);
            assert_eq!(metadata.mla_slots.len(), rows * 8);
            assert_eq!(i32_metadata(&metadata.offsets), [0, rows as i32]);
            assert_eq!(i32_metadata(&metadata.lengths), [(start + rows) as i32]);
            let physical = metadata.mla_slots.chunks_exact(8).map(|value|
                i64::from_le_bytes(value.try_into().unwrap())).collect::<Vec<_>>();
            for (offset, slot) in physical.into_iter().enumerate() {
                let position = start + offset;
                assert_eq!(slot, i64::from(pages[position / 64]) * 64 + (position % 64) as i64);
            }
        }
        Ok(())
    }

    #[test]
    fn unit_metadata_banks_match_individual_staging_including_decode_sized_tail() -> Result<()> {
        let cfg = KimiK3Config::from_hf(&cuteafd_loader::plan::testing::kimi_k3_config())?;
        let layout = LaneLayout::new(&cfg, 8256, 2752, 16384, NativeScratch { kda: 16, mla: [16; 3] })?;
        for span in [65usize, 2048, 2752, 5505, 8256] {
            let mut pool = cuteafd_engine::prefix::RefPagePool::new(256, 64);
            let pages = pool.alloc((span + 63).div_ceil(64))?;
            let windows = crate::shared::prefill_pipeline::prefill_units(span, 2752, 3, 256)?;
            let mut banks = ATTENTION_METADATA.into_iter().map(|name|
                (name, vec![0xa5; layout.operands[name]])).collect::<BTreeMap<_, _>>();
            let mut expected = Vec::new();
            for (bank, window) in windows.iter().enumerate() {
                let metadata = AttentionRows::new(&[SequenceRows { slot: 0, start: 63 + window.start,
                    rows: window.len(), pages: &pages }], 2752, 1, 16384, &pool)?;
                metadata.check_layout(&layout)?;
                for (name, bytes) in metadata.uploads() {
                    let (offset, capacity) = layout.metadata_extent(name, bank)?;
                    assert!(bytes.len() <= capacity);
                    banks.get_mut(name).unwrap()[offset..offset + bytes.len()].copy_from_slice(bytes);
                }
                expected.push(metadata);
            }
            for (bank, metadata) in expected.iter().enumerate() {
                for (name, bytes) in metadata.uploads() {
                    let (offset, _) = layout.metadata_extent(name, bank)?;
                    assert_eq!(&banks[name][offset..offset + bytes.len()], bytes);
                }
            }
            if span == 5505 {
                assert_eq!(expected.last().unwrap().rows, 1);
                assert_eq!(i32_metadata(&expected.last().unwrap().offsets), [0, 1]);
                // Negative control: a shared bank overwrites the first unit's lengths.
                let tail = &expected.last().unwrap().lengths;
                let mut shared = expected[0].lengths.clone();
                shared[..tail.len()].copy_from_slice(tail);
                assert_ne!(shared, expected[0].lengths);
            }
        }
        Ok(())
    }

    #[test]
    fn packed_attention_metadata_matches_singletons_and_rejects_member_aliases() -> Result<()> {
        let mut pool = cuteafd_engine::prefix::RefPagePool::with_reserved(32, 64, 1);
        let pages = [pool.alloc(3)?, pool.alloc(3)?, pool.alloc(1)?];
        let mut sequences = [SequenceRows { slot: 2, start: 63, rows: 65, pages: &pages[0] },
            SequenceRows { slot: 0, start: 3, rows: 127, pages: &pages[1] },
            SequenceRows { slot: 1, start: 0, rows: 3, pages: &pages[2] }];
        let members = AttentionRows::packed(&sequences, 256, 4, 256, &pool)?;
        for (member, sequence) in members.iter().zip(&sequences) {
            let alone = AttentionRows::new(std::slice::from_ref(sequence), 256, 4, 256, &pool)?;
            assert_eq!(member.uploads(), alone.uploads());
            assert!(i32_metadata(&member.kda_first).iter().all(|&first| first == 0));
            assert!(i32_metadata(&member.kda_slots).iter().all(|&slot| slot == sequence.slot as i32));
        }
        sequences[2].slot = 2;
        assert!(AttentionRows::packed(&sequences, 256, 4, 256, &pool).is_err());
        sequences[2].slot = 1;
        sequences[2].pages = &pages[0];
        assert!(AttentionRows::packed(&sequences, 256, 4, 256, &pool).is_err());
        sequences[2].pages = &pages[2];
        assert!(AttentionRows::packed(&sequences, 194, 4, 256, &pool).is_err());
        Ok(())
    }

    #[test]
    fn attention_rows_keep_mla_shape_independent_of_sequence_partition() -> Result<()> {
        let mut pool = cuteafd_engine::prefix::RefPagePool::with_reserved(8, 64, 1);
        let a = pool.alloc(2)?;
        let b = pool.alloc(1)?;
        let metadata = AttentionRows::new(&[
            SequenceRows { slot: 3, start: 63, rows: 3, pages: &a },
            SequenceRows { slot: 1, start: 7, rows: 2, pages: &b },
        ], 64, 4, 256, &pool)?;
        assert_eq!((metadata.rows, metadata.sequences, metadata.width), (5, 2, 4));
        assert_eq!(i32_metadata(&metadata.kda_slots), [3, 3, 3, 1, 1]);
        assert_eq!(i32_metadata(&metadata.kda_first), [0, 0, 0, 3, 3]);
        assert_eq!(i32_metadata(&metadata.lengths), [64, 65, 66, 8, 9]);
        assert_eq!(i32_metadata(&metadata.offsets), [0, 1, 2, 3, 4, 5]);
        let physical = metadata.mla_slots.chunks_exact(8).map(|value|
            i64::from_le_bytes(value.try_into().unwrap())).collect::<Vec<_>>();
        assert_eq!(physical, [i64::from(a[0]) * 64 + 63, i64::from(a[1]) * 64,
            i64::from(a[1]) * 64 + 1, i64::from(b[0]) * 64 + 7, i64::from(b[0]) * 64 + 8]);
        let table = i32_metadata(&metadata.page_table);
        for row in 0..3 { assert_eq!(&table[row * 4..row * 4 + 4], &[a[0] as i32, a[1] as i32, 0, 0]); }
        for row in 3..5 { assert_eq!(&table[row * 4..row * 4 + 4], &[b[0] as i32, 0, 0, 0]); }
        let cfg = cuteafd_loader::families::kimi_k3::config::KimiK3Config::from_hf(
            &cuteafd_loader::plan::testing::kimi_k3_config())?;
        let layout = LaneLayout::new(&cfg, 64, 64, 256, NativeScratch { kda: 16, mla: [16; 3] })?;
        metadata.check_layout(&layout)?;
        for (name, bytes) in metadata.uploads() { assert!(bytes.len() <= layout.operands[name]); }
        let narrow = LaneLayout::new(&cfg, 64, 64, 128, NativeScratch { kda: 16, mla: [16; 3] })?;
        assert!(metadata.check_layout(&narrow).is_err());
        let short = LaneLayout::new(&cfg, 4, 4, 256, NativeScratch { kda: 16, mla: [16; 3] })?;
        assert!(metadata.check_layout(&short).is_err());
        let single = AttentionRows::new(&[SequenceRows { slot: 0, start: 60, rows: 5, pages: &a }],
            64, 4, 256, &pool)?;
        assert_eq!(single.page_table.len(), metadata.page_table.len());
        assert_eq!(single.offsets, metadata.offsets);
        assert_eq!(single.lengths.len(), metadata.lengths.len());
        Ok(())
    }

    #[test]
    fn attention_rows_allow_shared_full_prefix_but_reject_shared_writes() -> Result<()> {
        let mut pool = cuteafd_engine::prefix::RefPagePool::new(8, 64);
        let prefix = pool.alloc(1)?;
        pool.share(&prefix);
        let a = [prefix[0], pool.alloc(1)?[0]];
        let b = [prefix[0], pool.alloc(1)?[0]];
        let rows = [SequenceRows { slot: 0, start: 64, rows: 1, pages: &a },
            SequenceRows { slot: 1, start: 64, rows: 1, pages: &b }];
        assert!(AttentionRows::new(&rows, 64, 2, 128, &pool).is_ok());
        assert!(AttentionRows::new(&[SequenceRows { slot: 0, start: 63, rows: 1, pages: &a }],
            64, 2, 128, &pool).is_err());
        let aliased = [SequenceRows { slot: 0, start: 64, rows: 1, pages: &a },
            SequenceRows { slot: 1, start: 64, rows: 1, pages: &a }];
        assert!(AttentionRows::new(&aliased, 64, 2, 128, &pool).is_err());
        Ok(())
    }

    #[test]
    fn attention_rows_validate_slots_pool_pages_and_wide_prefill() -> Result<()> {
        let mut pool = cuteafd_engine::prefix::RefPagePool::with_reserved(8, 64, 1);
        let pages = pool.alloc(3)?;
        for (slot, start, rows, ids) in [(0, 0, 0, pages.as_slice()), (2, 0, 1, &pages[..1]),
            (0, 127, 2, pages.as_slice()), (0, 64, 1, &pages[..1]), (0, usize::MAX, 1, pages.as_slice()),
            (0, 0, 1, &[0][..]), (0, 0, 1, &[7][..]), (0, 0, 1, &[8][..]),
            (0, 63, 2, &[pages[0], pages[0]][..])] {
            assert!(AttentionRows::new(&[SequenceRows { slot, start, rows, pages: ids }],
                128, 2, 128, &pool).is_err());
        }
        let repeated = [SequenceRows { slot: 0, start: 0, rows: 1, pages: &pages[..1] },
            SequenceRows { slot: 0, start: 0, rows: 1, pages: &pages[1..2] }];
        assert!(AttentionRows::new(&repeated, 64, 2, 128, &pool).is_err());
        let wide = AttentionRows::new(&[SequenceRows { slot: 0, start: 1, rows: 65, pages: &pages }],
            128, 2, 128, &pool)?;
        assert_eq!((wide.rows, wide.sequences, wide.width), (65, 1, 2));
        assert_eq!(i32_metadata(&wide.lengths), [66]);
        assert_eq!(i32_metadata(&wide.offsets), [0, 65]);
        let multi_wide = [SequenceRows { slot: 0, start: 0, rows: 64, pages: &pages[..1] },
            SequenceRows { slot: 1, start: 0, rows: 1, pages: &pages[1..2] }];
        assert!(AttentionRows::new(&multi_wide, 128, 2, 128, &pool).is_err());
        let parity = cuteafd_engine::prefix::RefPagePool::parity(8, 64, 0);
        let sequence = [SequenceRows { slot: 0, start: 0, rows: 1, pages: &pages }];
        assert!(AttentionRows::new(&sequence, 64, 2, 128, &parity).is_err());
        let wrong_rows = cuteafd_engine::prefix::RefPagePool::new(8, 32);
        assert!(AttentionRows::new(&sequence, 64, 2, 128, &wrong_rows).is_err());
        assert!(AttentionRows::new(&[], 64, 2, 128, &pool).is_err());
        Ok(())
    }

    #[test]
    #[ignore = "requires CUTEAFD_NATIVE_LIFETIME_FIXTURE_DIR; CPU-only fake DSO"]
    fn attention_uploads_reuse_primed_storage_and_retain_sources() -> Result<()> {
        use cuteafd_ffi::native_library_lifetime_fixture::Fixture;
        let fixture = Fixture::build()?;
        let library = fixture.load()?;
        let device = Device { library: &library, id: 1 };
        let cfg = KimiK3Config::from_hf(&cuteafd_loader::plan::testing::kimi_k3_config())?;
        let mut layout = LaneLayout::new(&cfg, 64, 64, 128, NativeScratch { kda: 16, mla: [16; 3] })?;
        layout.operands.retain(|name, _| ATTENTION_METADATA.contains(name));
        layout.arena_offsets.clear();
        layout.arena_bytes = 0;
        let operands = device.run(|| layout.operands.iter().map(|(&name, &bytes)|
            Ok((name, Allocation::new(device, bytes)?))).collect::<Result<BTreeMap<_, _>>>())?;
        let lane = Lane { stream: Rc::new(Stream { device, raw: std::ptr::null_mut() }), operands, arena: None, layout,
            window: None, metadata_bank: 0, core: None };
        lane.prime_attention()?;
        let primed = fixture.events()?;
        assert_eq!(primed.matches('C').count(), 6);
        assert_eq!(primed.matches('A').count(), 6);
        let mut pool = cuteafd_engine::prefix::RefPagePool::new(4, 64);
        let pages = pool.alloc(2)?;
        let a = AttentionRows::new(&[SequenceRows { slot: 0, start: 63, rows: 2, pages: &pages }],
            64, 2, 128, &pool)?;
        let b = AttentionRows::new(&[SequenceRows { slot: 1, start: 62, rows: 2, pages: &pages }],
            64, 2, 128, &pool)?;
        let pointers = ATTENTION_METADATA.map(|name| lane.buffer(name).unwrap().ptr);
        lane.upload_attention(&a)?;
        lane.upload_attention(&b)?;
        // The fixture defers DMA until a fence: staging B completes A's copy,
        // but must not overwrite A's pinned source before that completion.
        for (name, bytes) in a.uploads() {
            let buffer = lane.buffer(name)?;
            // SAFETY: CPU fake DSO allocated ordinary host memory and the
            // previous ordered-copy fence initialized exactly these bytes.
            let actual = unsafe { std::slice::from_raw_parts(buffer.ptr.cast::<u8>(), bytes.len()) };
            assert_eq!(actual, bytes);
        }
        let reused = fixture.events()?;
        assert_eq!(reused.matches('C').count(), 6);
        assert_eq!(reused.matches('A').count(), 6);
        assert_eq!(reused.matches('D').count(), 6);
        assert_eq!(ATTENTION_METADATA.map(|name| lane.buffer(name).unwrap().ptr), pointers);
        let mut wrong = b;
        wrong.width += 1;
        assert!(lane.upload_attention(&wrong).is_err());
        assert_eq!(fixture.events()?, reused, "reject bad stride before any upload");
        wrong.width -= 1;
        fixture.finish_ordered_uploads(&library)?;
        for (name, bytes) in wrong.uploads() {
            let buffer = lane.buffer(name)?;
            // SAFETY: the fixture's explicit completion initialized these live buffers.
            let actual = unsafe { std::slice::from_raw_parts(buffer.ptr.cast::<u8>(), bytes.len()) };
            assert_eq!(actual, bytes);
        }
        Ok(())
    }

    #[test]
    fn persistent_local_lane_never_drains_per_operator_and_keeps_device_routes() {
        let source = include_str!("device.rs").split("#[cfg(test)]").next().unwrap();
        let owner = source.split("pub(crate) struct LocalLane").nth(1).unwrap().split("/// Head handles").next().unwrap();
        assert!(owner.contains("programs: Rc<NativePrograms"));
        assert!(owner.contains("layers: Vec<Rc<super::weights::LayerWeights"));
        assert!(owner.contains("kda: Rc<super::state::KdaBank"));
        assert!(owner.contains("mla: Rc<super::mla::CacheBank"));
        assert!(owner.contains("kda.check_reader(lane_index, &lane.stream)?"));
        assert!(owner.contains("mla.check_reader(lane_index, &lane.stream)?"));
        assert!(owner.contains("drain_or_abort(\"Kimi persistent local lane\")"));
        let methods = owner.split("impl<'a> LocalLane").nth(1).unwrap();
        for forbidden in [".drain(", "drain_or_abort", ".wait(", "synchronize", "Allocation::new", "sm_count()"] {
            assert!(!methods.contains(forbidden), "operator path contains {forbidden}");
        }
        assert!(methods.contains("Result<(cuteafd_ffi::KimiRoutes, CuteafdDeviceBuffer)>"));
        assert!(methods.contains("wire_grid.blocks(rows, 3584)"));
        assert!(!methods.contains("Result<Vec<Routes>>"));
    }

    #[test]
    #[ignore = "requires CUTEAFD_NATIVE_LIFETIME_FIXTURE_DIR; CPU-only fake DSO"]
    fn lane_destruction_drains_before_operands_with_retained_stream() -> Result<()> {
        use cuteafd_ffi::native_library_lifetime_fixture::Fixture;
        let fixture = Fixture::build()?;
        let library = fixture.load()?;
        let device = Device { library: &library, id: 1 };
        let stream = Rc::new(Stream { device, raw: std::ptr::null_mut() });
        let lane = Lane { stream: stream.clone(),
            operands: BTreeMap::from([("delta", Allocation::new(device, 16)?)]),
            arena: Some(Allocation::new(device, 32)?),
            layout: LaneLayout { base: cuteafd_loader::families::kimi_k3::allocation::NativeLaneLayout {
                rows: 1, unit: 1, max_context: 64, operands: BTreeMap::from([("delta", 16)]),
                arena_offsets: BTreeMap::new(), arena_bytes: 32 }, core: None }, window: None, metadata_bank: 0, core: None };
        assert!(!fixture.events()?.contains('d'));
        drop(lane);
        let events = fixture.events()?;
        let first_free = events.find('d').ok_or_else(|| anyhow::anyhow!("lane released no storage"))?;
        assert_eq!(events[..first_free].matches('S').count(), 1,
            "lane teardown must drain even when another owner retains its stream: {events}");
        assert_eq!(events.matches('d').count(), 2);
        drop(stream);
        Ok(())
    }

    #[test]
    #[ignore = "requires CUTEAFD_NATIVE_LIFETIME_FIXTURE_DIR; CPU-only fake DSO"]
    fn entry_token_io_retains_weights_and_drains_before_storage() -> Result<()> {
        use cuteafd_ffi::native_library_lifetime_fixture::Fixture;
        use cuteafd_core::{DType, TensorInfo, TensorRole};
        use cuteafd_loader::LoadedTensor;
        use super::super::weights::GlobalLayout;
        let fixture = Fixture::build()?;
        let library = fixture.load()?;
        let device = Device { library: &library, id: 1 };
        let mut cfg = KimiK3Config::from_hf(&cuteafd_loader::plan::testing::kimi_k3_config())?;
        cfg.hidden = 4;
        cfg.vocab_size = 8;
        let layout = GlobalLayout::pair(&cfg)?;
        let packed = layout[0].pack(&cfg, |name| Ok(LoadedTensor {
            info: TensorInfo { name: name.into(), file: "test".into(), dtype: DType::Bf16,
                shape: vec![8, 4], byte_offset: 0, byte_length: 64, role: TensorRole::Other,
                layer_id: None, expert_id: None, is_quantization_metadata: false },
            source_path: "test".into(), bytes: vec![0; 64], elapsed_micros: 0, sha256: String::new(),
        }))?;
        let weights = Rc::new(layout[0].upload(device, packed)?);
        let weak = Rc::downgrade(&weights);
        let stream = Rc::new(Stream { device, raw: std::ptr::null_mut() });
        let io = IoLayout { rank: 0, rows: 16, hidden: 4, vocab: 8, core_head:false }
            .allocate(device, stream.clone(), weights.clone())?;
        drop(weights);
        drop(stream);
        assert!(weak.upgrade().is_some());
        assert!(!fixture.events()?.contains('d'));
        drop(io);
        assert!(weak.upgrade().is_none());
        let events = fixture.events()?;
        let first_free = events.find('d').ok_or_else(|| anyhow::anyhow!("token IO released no storage"))?;
        assert_eq!(events[..first_free].matches('S').count(), 2, "lane drains before IDs/weights free: {events}");
        assert_eq!(events.matches('d').count(), 2, "one IDs and one embedding allocation: {events}");
        Ok(())
    }

    #[test]
    fn token_io_rejects_aliases_other_devices_and_wrapping_addresses() -> Result<()> {
        let buffer = |ptr, bytes, device_id| CuteafdDeviceBuffer {
            ptr: ptr as *mut std::ffi::c_void, bytes, device_id, flags: 0,
        };
        let a = buffer(4096usize, 128, 0);
        io_disjoint(a, buffer(4224, 64, 0))?;
        io_disjoint(buffer(4000, 96, 0), a)?;
        for other in [a, buffer(4200, 64, 0), buffer(4095, 2, 0), buffer(8192, 64, 1),
            buffer(0, 64, 0), buffer(usize::MAX - 16, 32, 0)] {
            assert!(io_disjoint(a, other).is_err());
            assert!(io_disjoint(other, a).is_err());
        }
        Ok(())
    }

    #[test]
    fn token_io_extents_are_native_bounded_and_lane_private() -> Result<()> {
        let cfg = KimiK3Config::from_hf(&cuteafd_loader::plan::testing::kimi_k3_config())?;
        for rows in [1, 16, 64, 4096, 8192, 8256] {
            let entry = IoLayout::new(&cfg, 0, rows)?;
            let terminal = IoLayout::new(&cfg, 1, rows)?;
            assert_eq!(entry.extents(), [rows * 4, 0, 0]);
            assert_eq!(terminal.extents(), [0, rows.min(64) * 163840 * 4, VOCABULARY_HEAD_WORKSPACE]);
            for rank in [&entry, &terminal] {
                assert_ne!(rank.demand(0)?.group, rank.demand(1)?.group);
                assert_eq!(rank.demand(0)?.bytes, rank.extents().into_iter().sum::<usize>() as u64);
                assert!(rank.demand(2).is_err());
            }
        }
        assert!(IoLayout::new(&cfg, 0, 8192).is_ok(), "token ids span a two-unit lane");
        for (rank, rows) in [(2, 1), (0, 0), (1, 8257)] {
            assert!(IoLayout::new(&cfg, rank, rows).is_err());
        }
        let source = include_str!("device.rs").split("#[cfg(test)]").next().unwrap();
        assert!(source.contains(".vocabulary_head_rows("));
        assert!(source.contains("crate::shared::target_head::launch("));
        assert!(!source.contains(".vocab_head_rows("));
        Ok(())
    }


    #[test]
    fn native_head_reuses_layer_scratch_and_drops_only_legacy_head_workspace() -> Result<()> {
        use cuteafd_loader::families::kimi_k3::core::{CoreInventory,CoreProjection,CoreSplit};
        use cuteafd_loader::formats::exl3_storage::Exl3Codebook;
        let cfg=KimiK3Config::from_hf(&cuteafd_loader::plan::testing::kimi_k3_config())?;
        let stem="language_model.lm_head".to_owned();
        let core=CoreInventory{projections:BTreeMap::from([(stem.clone(),CoreProjection{stem,bits:6,
            codebook:Exl3Codebook::Mcg,shape:[cfg.vocab_size,cfg.hidden],stored_outputs:cfg.vocab_size,
            source_bytes:0,split:CoreSplit::Whole})])};
        for rows in [1,8,64,4096] {
            assert_eq!(IoLayout::with_core(&cfg,0,rows,&core)?,IoLayout::new(&cfg,0,rows)?);
            let native=IoLayout::with_core(&cfg,1,rows,&core)?;
            assert_eq!(native.extents(),[0,rows.min(64)*cfg.vocab_size*4,0]);
            assert_eq!(IoLayout::new(&cfg,1,rows)?.demand(0)?.bytes-native.demand(0)?.bytes,
                VOCABULARY_HEAD_WORKSPACE as u64);
        }
        Ok(())
    }

    #[test]
    fn fused_residual_schedule_matches_every_layer_and_output_frontier() {
        let cfg = KimiK3Config::from_hf(&cuteafd_loader::plan::testing::kimi_k3_config()).unwrap();
        let mut stored = 0;
        for layer in 0..cfg.layers {
            let attention = residual_site(&cfg, MixSite::Attention(layer)).unwrap();
            assert_eq!(attention.candidates, stored + 1);
            assert_eq!(attention.prefix, if layer == 0 { KimiPrefixMode::Keep } else { KimiPrefixMode::Add });
            assert_eq!(attention.append, layer % cfg.residual_block == 0);
            if attention.append { stored += 1; }
            let ffn = residual_site(&cfg, MixSite::Ffn(layer)).unwrap();
            assert_eq!(ffn.candidates, stored + 1);
            assert_eq!(ffn.prefix, if attention.append { KimiPrefixMode::Replace } else { KimiPrefixMode::Add });
            assert!(!ffn.append);
        }
        let output = residual_site(&cfg, MixSite::Output).unwrap();
        assert_eq!(output.candidates, 9);
        assert_eq!(output.prefix, KimiPrefixMode::Add);
        assert!(!output.append);
        assert!(residual_site(&cfg, MixSite::Attention(cfg.layers)).is_err());
        assert!(residual_site(&cfg, MixSite::Ffn(cfg.layers)).is_err());
    }

    #[test]
    fn owned_mla_and_kda_keep_replicated_stored_residuals_bit_equal() -> Result<()> {
        use cuteafd_loader::families::kimi_k3::{config::KimiAttention, reference::bf16};
        let cfg = KimiK3Config::from_hf(&cuteafd_loader::plan::testing::kimi_k3_config())?;
        let embedding = (0..32).map(|i| bf16((i as f32 - 15.0) / 13.0)).collect::<Vec<_>>();
        let mut prefixes = [embedding.clone(), embedding.clone()];
        let mut stored: [Vec<Vec<f32>>; 2] = [vec![], vec![]];
        let mut hidden = embedding;
        let mut expected_stored = vec![];
        let mut previous_ffn = vec![0.0; hidden.len()];
        let bits = |row: &[f32]| row.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
        let add = |a: &[f32], b: &[f32]| a.iter().zip(b).map(|(a, b)| bf16(a + b)).collect::<Vec<_>>();
        let apply = |prefix: &mut Vec<f32>, blocks: &mut Vec<Vec<f32>>, site: KimiResidualSite, delta: &[f32]| {
            *prefix = match site.prefix {
                KimiPrefixMode::Keep => prefix.clone(),
                KimiPrefixMode::Add => add(prefix, delta),
                KimiPrefixMode::Replace => delta.to_vec(),
            };
            if site.append { blocks.push(prefix.clone()); }
            assert_eq!(site.candidates, blocks.len() + usize::from(!site.append));
        };
        let mut mla_ordinal = 0;
        for layer in 0..cfg.layers {
            let attention = residual_site(&cfg, MixSite::Attention(layer))?;
            for rank in 0..2 { apply(&mut prefixes[rank], &mut stored[rank], attention, &previous_ffn); }
            if layer % cfg.residual_block == 0 { expected_stored.push(hidden.clone()); }
            for rank in 0..2 {
                assert_eq!(bits(&prefixes[rank]), bits(&hidden), "attention prefix layer {layer} rank {rank}");
                assert_eq!(stored[rank].iter().map(|row| bits(row)).collect::<Vec<_>>(),
                    expected_stored.iter().map(|row| bits(row)).collect::<Vec<_>>());
            }
            let partials: [Vec<f32>; 2] = std::array::from_fn(|rank| (0..hidden.len())
                .map(|i| bf16(((layer * 17 + i * 3 + rank * 5) % 61) as f32 / 29.0 - 1.0)).collect());
            let attended = match cfg.attention[layer] {
                KimiAttention::Kda => {
                    // Both ranks use the same operand order at the TP2 meet.
                    let result = add(&partials[0], &partials[1]);
                    [result.clone(), result]
                }
                KimiAttention::Mla => {
                    let owner = mla_ordinal % 2;
                    mla_ordinal += 1;
                    let mut local = [vec![f32::NAN; hidden.len()], vec![f32::NAN; hidden.len()]];
                    local[owner] = partials[owner].clone();
                    // Model the shared after-attention hop: only the attended
                    // BF16 delta moves, never the prefix or stored arena slots.
                    local[1 - owner] = local[owner].clone();
                    local
                }
            };
            let ffn_site = residual_site(&cfg, MixSite::Ffn(layer))?;
            for rank in 0..2 { apply(&mut prefixes[rank], &mut stored[rank], ffn_site, &attended[rank]); }
            let expected_prefix = if layer % cfg.residual_block == 0 { attended[0].clone() }
                else { add(&hidden, &attended[0]) };
            for rank in 0..2 {
                assert_eq!(bits(&prefixes[rank]), bits(&expected_prefix), "FFN prefix layer {layer} rank {rank}");
                assert_eq!(stored[rank].iter().map(|row| bits(row)).collect::<Vec<_>>(),
                    expected_stored.iter().map(|row| bits(row)).collect::<Vec<_>>());
            }
            previous_ffn = (0..hidden.len()).map(|i| bf16(((layer + i * 7) % 37) as f32 / 31.0 - 0.5)).collect();
            hidden = add(&expected_prefix, &previous_ffn);
        }
        let output = residual_site(&cfg, MixSite::Output)?;
        for rank in 0..2 {
            apply(&mut prefixes[rank], &mut stored[rank], output, &previous_ffn);
            assert_eq!(bits(&prefixes[rank]), bits(&hidden));
        }
        assert_eq!(mla_ordinal, 24);
        assert_eq!(stored[0].iter().map(|row| bits(row)).collect::<Vec<_>>(),
            stored[1].iter().map(|row| bits(row)).collect::<Vec<_>>());
        Ok(())
    }

    #[test]
    fn native_scratch_uses_each_manifest_peak_and_checks_total_overflow() -> Result<()> {
        let scratch = NativeScratch::combine([1024, 4096],
            [[256, 1024, 4096], [512, 2048, 2048], [128, 8192, 1024]])?;
        assert_eq!(scratch, NativeScratch { kda: 4096, mla: [512, 8192, 4096] });
        assert_eq!(scratch.bytes()?, 16_896);
        assert!(NativeScratch::combine([usize::MAX, 1], [[1, 0, 0]]).is_err());
        assert!(NativeScratch::combine([0, 0], [[0, usize::MAX, 1]]).is_err());
        Ok(())
    }

    #[test]
    fn lane_layout_uses_native_extents_without_hidden_engine_allowances() -> Result<()> {
        let cfg = KimiK3Config::from_hf(&cuteafd_loader::plan::testing::kimi_k3_config())?;
        let scratch = NativeScratch { kda: 4096, mla: [512, 0, 1024] };
        for rows in [1, 16, 4096] {
            let layout = LaneLayout::new(&cfg, rows, (rows).min(4096), 1_048_576, scratch)?;
            assert_eq!(layout.operands["residual arena"], KimiResidual::arena_bytes(rows)?);
            assert_eq!(layout.operands["gate-up"], rows * 2 * 16896 * 2);
            assert_eq!(layout.metadata_extent("MLA page table", 0)?.1, rows.min(64) * 16384 * 4);
            assert_eq!(layout.operands["KDA normalized"], rows * 6144 * 2);
            assert_eq!(layout.operands["MLA query"], rows * 96 * 576 * 2);
            assert_eq!(layout.operands["expert wire"], rows * KimiExpertInput::ROW_BYTES);
            assert_eq!(layout.operands["MLA attention scratch"], 16);
            let separate = layout.operands.iter().filter(|(name, _)| !layout.arena_offsets.contains_key(*name))
                .map(|(_, bytes)| bytes).sum::<usize>();
            assert_eq!(layout.bytes()?, separate + layout.arena_bytes);
            assert!(layout.bytes()? < layout.operands.values().sum::<usize>());
            for phase in SCRATCH_PHASES {
                for (i, &a) in phase.iter().enumerate() {
                    let start = layout.arena_offsets[a];
                    assert_eq!(start % ARENA_ALIGNMENT, 0);
                    assert!(start + layout.operands[a] <= layout.arena_bytes);
                    for &b in &phase[..i] {
                        assert!(start >= layout.arena_offsets[b] + layout.operands[b]);
                    }
                }
            }
            for name in ATTENTION_METADATA.into_iter().chain(["residual arena", "delta", "normalized",
                "attention output", "ffn output", "shared output", "routed output", "combined output",
                "route ids", "route weights", "expert wire", "expert combined", "latent normalized"]) {
                assert!(!layout.arena_offsets.contains_key(name), "persistent/cross-stream operand {name}");
            }
            assert_eq!(layout.demand(1, 0)?.bytes, layout.bytes()? as u64);
            assert!(layout.demand(2, 0).is_err());
            assert!(layout.demand(0, 2).is_err());
        }
        for (rows, context) in [(0, 1), (8257, 1), (1, 0), (1, 1_048_577)] {
            assert!(LaneLayout::new(&cfg, rows, (rows).min(4096), context, scratch).is_err());
        }
        // Shared lanes include the rounded three-wave span; units stay at most 4096 rows.
        assert!(LaneLayout::new(&cfg, 8192, 4096, 1_048_576, scratch).is_ok());
        assert!(LaneLayout::new(&cfg, 8192, 4097, 1_048_576, scratch).is_err());
        assert!(LaneLayout::new(&cfg, 1, 1, 1, NativeScratch { kda: usize::MAX, mla: [0; 3] }).is_err());
        Ok(())
    }

    /// Vector-clock model: host enqueue order is not a completion dependency.
    /// A phase's consumers retain all of its views conservatively until the last
    /// native stage, even when an individual stage only touches a subset.
    #[test]
    fn phase_arena_layer_sequences_require_stream_order_or_event_join() -> Result<()> {
        struct Use { phase: usize, clock: [usize; 2] }
        struct Schedule<'l> {
            layout: &'l LaneLayout,
            clocks: [[usize; 2]; 2],
            uses: Vec<Use>,
        }
        impl Schedule<'_> {
            fn kernel(&mut self, stream: usize, phase: Option<usize>) -> Result<()> {
                self.clocks[stream][stream] += 1;
                let clock = self.clocks[stream];
                if let Some(phase) = phase {
                    for prior in &self.uses {
                        let ordered = prior.clock.iter().zip(clock).all(|(&a, b)| a <= b);
                        for &a in SCRATCH_PHASES[prior.phase] {
                            for &b in SCRATCH_PHASES[phase] {
                                let start_a = self.layout.arena_offsets[a];
                                let start_b = self.layout.arena_offsets[b];
                                let overlap = start_a < start_b + self.layout.operands[b]
                                    && start_b < start_a + self.layout.operands[a];
                                ensure!(!overlap || ordered, "unordered arena reuse: {a} -> {b}");
                            }
                        }
                    }
                    self.uses.push(Use { phase, clock });
                }
                Ok(())
            }
            fn record(&self, stream: usize) -> [usize; 2] { self.clocks[stream] }
            fn wait(&mut self, stream: usize, event: [usize; 2]) {
                for (clock, completed) in self.clocks[stream].iter_mut().zip(event) {
                    *clock = (*clock).max(completed);
                }
            }
        }
        let cfg = KimiK3Config::from_hf(&cuteafd_loader::plan::testing::kimi_k3_config())?;
        for rows in [1, 4, 16, 64, 4096] {
            let layout = LaneLayout::new(&cfg, rows, (rows).min(4096), 1_048_576,
                NativeScratch { kda: 4096, mla: [1024, 8192, 4096] })?;
            let make = || Schedule { layout: &layout, clocks: [[0; 2]; 2], uses: Vec::new() };
            // Walk every layer, including both attention kinds, dense and routed
            // FFNs. All actual arena-consuming launches use the retained lane stream.
            let mut serial = make();
            for layer in 0..cfg.layers {
                serial.kernel(0, None)?; // attention residual mix
                match cfg.attention[layer] {
                    cuteafd_loader::families::kimi_k3::config::KimiAttention::Kda => {
                        serial.kernel(0, Some(0))?; // recurrent producer
                        serial.kernel(0, Some(0))?; // normalized output projection consumer
                    }
                    cuteafd_loader::families::kimi_k3::config::KimiAttention::Mla => {
                        for _ in 0..3 { serial.kernel(0, Some(1))?; } // producer, attention, output
                    }
                }
                serial.kernel(0, None)?; // attention publication + FFN residual mix
                if !cfg.dense[layer] { serial.kernel(0, None)?; } // route/down/wire + expert enqueue/reduce
                for _ in 0..3 { serial.kernel(0, Some(2))?; } // gate-up, activation, down
                if !cfg.dense[layer] { serial.kernel(0, None)?; } // intake/reduce/latent-up/combine
            }
            // A consumer on a second stream extends the old phase lifetime.
            // Waiting only for its producer does not permit arena reuse.
            for attention in [0, 1] {
                let mut asynchronous = make();
                asynchronous.kernel(0, Some(attention))?;
                let produced = asynchronous.record(0);
                asynchronous.wait(1, produced);
                asynchronous.kernel(1, Some(attention))?;
                assert!(asynchronous.kernel(0, Some(2)).is_err());
                let consumed = asynchronous.record(1);
                asynchronous.wait(0, consumed);
                asynchronous.kernel(0, Some(2))?;

                let mut early_reuse = make();
                early_reuse.kernel(0, Some(attention))?;
                assert!(early_reuse.kernel(1, Some(2)).is_err());
                early_reuse.wait(1, early_reuse.record(0));
                early_reuse.kernel(1, Some(2))?;
            }
        }
        // Production has no second-stream arena user; peer/intake operands are
        // separate. Keep the native launch stream contract visible to this guard.
        let source = include_str!("device.rs").split("#[cfg(test)]").next().unwrap();
        let operators = source.split("pub(crate) fn attention(&mut self").nth(1).unwrap()
            .split("/// A scheduler-owned sequence stretch").next().unwrap();
        assert!(!operators.contains("Stream::new"));
        assert!(!operators.contains("peer.stream"));
        assert!(operators.contains("self.lane.stream.raw"));
        Ok(())
    }

    #[test]
    fn core_stage_arena_preserves_bf16_offsets_and_full_prefill_capacity() -> Result<()> {
        use cuteafd_loader::families::kimi_k3::{allocation::CORE_MEET_PLANES, core::lane_extents};
        let cfg = KimiK3Config::from_hf(&cuteafd_loader::plan::testing::kimi_k3_config())?;
        // Installed-package peaks; the split KDA recurrence is larger than fused.
        let scratch = NativeScratch { kda: 629_464_064,
            mla: [168_296_448, 101_056_512, 201_326_592] };
        for (rows, unit) in [(1, 1), (4, 4), (16, 16), (63, 63), (64, 64), (4096, 4096), (8192, 4096), (8256, 2752), (8192, 2048)] {
            for larger in [false, true] {
                let split = if larger { 630_905_856 } else { scratch.kda };
                let stages = lane_extents(rows, unit, scratch.kda, scratch.mla[0], split, 0)?;
                let base = LaneLayout::new(&cfg, rows, unit, 1_048_576, scratch)?;
                let shared = base.base.clone().with_core_stages(stages.clone())?;
                for (&name, &offset) in &base.arena_offsets {
                    assert_eq!(shared.arena_offsets[name], offset, "BF16 offset {name}");
                }
                assert_eq!(shared.rows, rows);
                for name in CORE_MEET_PLANES {
                    assert!(!shared.arena_offsets.contains_key(name));
                    let element_bytes = if name == "K6 reduced" { 4 } else { 2 };
                    assert_eq!(shared.operands[name], rows * 7168 * element_bytes);
                }
                for phase in shared.core_phases() {
                    for (i, &a) in phase.iter().enumerate() {
                        let start = shared.arena_offsets[a];
                        assert_eq!(start % ARENA_ALIGNMENT, 0);
                        assert!(start + shared.operands[a] <= shared.arena_bytes);
                        for &b in &phase[..i] {
                            assert!(start + shared.operands[a] <= shared.arena_offsets[b]
                                || shared.arena_offsets[b] + shared.operands[b] <= start,
                                "live overlap {a} / {b}");
                        }
                    }
                }
                if unit == 4096 && larger {
                    // Hook-local scratch serves one unit: the 8192-row lane
                    // keeps the 4096-row arena; only row planes double.
                    assert_eq!(base.arena_bytes, 1_340_473_344);
                    assert_eq!(shared.arena_bytes, base.arena_bytes);
                    // Two retained BF16 planes and the ordered F32 sum over the lane.
                    let meets = rows * 7168 * (2 + 2 + 4);
                    let old = stages.values().sum::<usize>();
                    let added = shared.bytes()? - base.bytes()?;
                    assert_eq!(old - meets, 1_554_701_312 - 352_321_536);
                    assert_eq!(added, meets);
                }
            }
        }
        Ok(())
    }

    #[test]
    fn captured_core_phases_require_lane_order_and_fenced_peer_meets() -> Result<()> {
        use cuteafd_loader::families::kimi_k3::{allocation::CORE_MEET_PLANES, core::lane_extents};
        struct Schedule<'a> {
            layout: &'a cuteafd_loader::families::kimi_k3::allocation::NativeLaneLayout,
            clocks: [[usize; 2]; 2],
            uses: Vec<(&'static str, [usize; 2])>,
        }
        impl Schedule<'_> {
            fn overlap(&self, a: &str, b: &str) -> bool {
                match (self.layout.arena_offsets.get(a), self.layout.arena_offsets.get(b)) {
                    (Some(&a0), Some(&b0)) => a0 < b0 + self.layout.operands[b]
                        && b0 < a0 + self.layout.operands[a],
                    (None, None) => a == b,
                    _ => false,
                }
            }
            fn kernel(&mut self, stream: usize, operands: &[&'static str]) -> Result<()> {
                self.clocks[stream][stream] += 1;
                let clock = self.clocks[stream];
                for &(prior, finished) in &self.uses {
                    let ordered = finished.iter().zip(clock).all(|(&a, b)| a <= b);
                    for &next in operands {
                        ensure!(ordered || !self.overlap(prior, next),
                            "unfenced captured reuse {prior} -> {next}");
                    }
                }
                self.uses.extend(operands.iter().map(|&name| (name, clock)));
                Ok(())
            }
            fn wait(&mut self, stream: usize, event: [usize; 2]) {
                for (clock, finished) in self.clocks[stream].iter_mut().zip(event) {
                    *clock = (*clock).max(finished);
                }
            }
            fn meet(&mut self, partial: &'static str) -> Result<()> {
                // Protocol::producers joins the exact lane before peer.push;
                // Protocol::consumer joins reduction before the next lane node.
                self.kernel(0, &[partial])?;
                self.wait(1, self.clocks[0]);
                self.kernel(1, &[partial, "K6 reduced"])?;
                self.wait(0, self.clocks[1]);
                Ok(())
            }
        }
        let cfg = KimiK3Config::from_hf(&cuteafd_loader::plan::testing::kimi_k3_config())?;
        for (rows, unit) in [(1, 1), (4, 4), (16, 16), (64, 64), (4096, 4096), (8192, 4096), (8256, 2752), (8192, 2048)] {
            let scratch = NativeScratch { kda: 629_464_064,
                mla: [168_296_448, 101_056_512, 201_326_592] };
            let layout = LaneLayout::new(&cfg, rows, unit, 1_048_576, scratch)?.base.with_core_stages(
                lane_extents(rows, unit, scratch.kda, scratch.mla[0], 630_905_856, 0)?)?;
            let phases = layout.core_phases();
            let make = || Schedule { layout: &layout, clocks: [[0; 2]; 2], uses: Vec::new() };
            let mut captured = make();
            for layer in 0..cfg.layers {
                // Mixed checkpoints retain fused BF16 captures between K6 ones.
                if layer % 2 == 0 {
                    let phase = usize::from(cfg.attention[layer]
                        == cuteafd_loader::families::kimi_k3::config::KimiAttention::Mla);
                    captured.kernel(0, SCRATCH_PHASES[phase])?;
                } else {
                    let range = if cfg.attention[layer]
                        == cuteafd_loader::families::kimi_k3::config::KimiAttention::Kda { 0..3 } else { 3..7 };
                    for phase in &phases[range] { captured.kernel(0, phase)?; }
                    captured.meet("K6 attention partial")?;
                }
                for phase in &phases[7..] { captured.kernel(0, phase)?; }
                captured.meet("K6 FFN partial")?;
            }
            for phase in &phases {
                let mut missing_producer = make();
                missing_producer.kernel(0, phase)?;
                assert!(missing_producer.kernel(1, phase).is_err());
                missing_producer.wait(1, missing_producer.clocks[0]);
                missing_producer.kernel(1, phase)?;
                // An exchange-stream reader extends the phase lifetime until
                // its consumer event is joined, not merely host-enqueued.
                assert!(missing_producer.kernel(0, phase).is_err());
                missing_producer.wait(0, missing_producer.clocks[1]);
                missing_producer.kernel(0, phase)?;
            }
            let mut parallel = make();
            parallel.kernel(1, &CORE_MEET_PLANES)?;
            for phase in &phases { parallel.kernel(0, phase)?; }
        }
        // The captured helper functions propagate the caller's lane stream;
        // exchange reduction touches only the retained, non-arena meet planes.
        let source = include_str!("core_runtime.rs").split("#[cfg(test)]").next().unwrap();
        let projected = source.split("pub(crate) fn kda_projected").nth(1).unwrap()
            .split("pub(crate) struct CorePrograms").next().unwrap();
        assert!(!projected.contains("Stream::new") && !projected.contains("peer.stream"));
        assert!(projected.contains("core.launch(weights,batch,stream)"));
        assert!(projected.contains("situ.launch(rows,gate_up,activated,stream)"));
        let layers = include_str!("layers.rs").split("#[cfg(test)]").next().unwrap();
        assert!(layers.contains("cuda_stream_wait_event"));
        let bindings = include_str!("layers.rs").split("fn meet_core").nth(1).unwrap()
            .split("fn transfer").next().unwrap();
        assert!(bindings.contains("self.protocol.consumer(self.lane,rank)?"));
        Ok(())
    }

    #[test]
    fn unbound_ops_name_their_work_item() {
        let cfg = cuteafd_loader::families::kimi_k3::config::KimiK3Config::from_hf(
            &cuteafd_loader::plan::testing::kimi_k3_config()).unwrap();
        let plane = Plane { ptr: std::ptr::null_mut(), rows: 1, width: cfg.hidden };
        let error = super::super::block::forward(&cfg, &mut Pending, plane, cfg.layers).unwrap_err().to_string();
        // Layer 0 runs no attention mix: the first op is its input norm.
        assert!(error.contains("Kimi K3 norm") && error.contains("#273"), "{error}");
        let error = Pending.kda(0, &plane).unwrap_err().to_string();
        assert!(error.contains("W6"), "{error}");
    }
}
