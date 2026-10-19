//! Device-bound native programs and residual scheduling for the coordinator.
//! The unbound operation backend still refuses with its owning work item
//! (#273), never substitute arithmetic.
use super::block::{MixSite, Ops, Routes};
use anyhow::{bail, ensure, Result};
use cuteafd_ffi::{programs::Programs, Bf16Add, KimiExpertInput, KimiFfn, KimiFfnBatch,
    KimiKda, KimiKdaCommit, KimiLatent, KimiMlaMode, KimiPrefixMode, KimiResidual,
    KimiResidualBatch, KimiResidualSite, KimiRmsNorm, KimiRouter};
use crate::shared::{memory::device::{Allocation, Device, DeviceOwner, Stream}, prefix::view};
use cuteafd_ffi::{CuteafdDeviceBuffer, programs::{VocabularyHead, VOCABULARY_HEAD_WORKSPACE}};
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
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct NativeScratch {
    pub kda: usize,
    pub mla: [usize; 3],
}

impl NativeScratch {
    fn combine(kda: [usize; 2], mla: impl IntoIterator<Item = [usize; 3]>) -> Result<Self> {
        let mut scratch = Self { kda: *kda.iter().max().unwrap(), mla: [0; 3] };
        for requirements in mla {
            for (current, required) in scratch.mla.iter_mut().zip(requirements) {
                *current = (*current).max(required);
            }
        }
        scratch.bytes()?;
        Ok(scratch)
    }

    pub(crate) fn bytes(&self) -> Result<usize> {
        self.mla.iter().try_fold(self.kda, |sum, bytes| sum.checked_add(*bytes)
            .ok_or_else(|| anyhow::anyhow!("Kimi native scratch extent overflow")))
    }
}

impl<'a> NativePrograms<'a> {
    pub(crate) fn load(device: Device<'a>, cfg: &KimiK3Config,
        programs: &'a Programs<'a>, ffn_mode: cuteafd_ffi::KimiFfnMode) -> Result<Self> {
        ensure!(device.id >= 0 && std::ptr::eq(device.library, programs.library()),
            "Kimi programs and device must share their native library");
        device.run(|| {
            let kda = [KimiKda::load(programs, 64, true)?, KimiKda::load(programs, 4096, false)?];
            // Admission can select owned MLA or explicit head split. Resolve
            // both before its measured baseline, never lazily after admission.
            let load_mla = |heads| -> Result<MlaPrograms<'a>> {
                Ok(MlaPrograms {
                    decode: super::mla::Mla::load(cfg, programs, heads, 64, KimiMlaMode::Decode)?,
                    verify: super::mla::Mla::load(cfg, programs, heads, 64, KimiMlaMode::Verify)?,
                    extend: super::mla::Mla::load(cfg, programs, heads, 4096, KimiMlaMode::Extend)?,
                })
            };
            let mla = [load_mla(48)?, load_mla(96)?];
            let scratch = NativeScratch::combine(kda.each_ref().map(|p| p.scratch_bytes()),
                mla.iter().flat_map(|p| [p.decode.scratch_bytes(), p.verify.scratch_bytes(), p.extend.scratch_bytes()]))?;
            Ok(Self { device, w8: W8::load(cfg, programs, ffn_mode)?, norm: KimiRmsNorm::load(programs, cfg.hidden)?,
                kda, commit: KimiKdaCommit::load(programs)?, mla, latent: KimiLatent::load(programs)?,
                router: KimiRouter::load(programs)?, wire: KimiExpertInput::load(programs)?,
                add: device.library.bf16_add()?, scratch,
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

    pub(crate) fn lane_layout(&self, cfg: &KimiK3Config, rows: usize, max_context: usize) -> Result<LaneLayout> {
        self.check_config(cfg)?;
        ensure!(self.mla.iter().all(|p| [p.decode.max_context(), p.verify.max_context(), p.extend.max_context()]
            .into_iter().all(|capacity| max_context <= capacity)), "Kimi lane context exceeds native programs");
        LaneLayout::new(cfg, rows, max_context, self.scratch)
    }
}

/// Named storage for one lane, shared by its serialized layer operations.
/// Cache/state pools, prefix marks, head I/O and peer/expert exchange are separate
/// owners; these bytes must not be used as a total-engine allowance.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct LaneLayout {
    rows: usize,
    max_context: usize,
    operands: BTreeMap<&'static str, usize>,
    arena_offsets: BTreeMap<&'static str, usize>,
    arena_bytes: usize,
}

// Each native call keeps its internal operands disjoint. Only different phases
// alias, exclusively on the lane's exact compute stream, including graph replay.
const SCRATCH_PHASES: [&[&str]; 3] = [
    &["KDA normalized", "KDA scratch"],
    &["MLA query", "MLA q residual", "MLA attention", "MLA LSE",
        "MLA producer scratch", "MLA attention scratch", "MLA output scratch"],
    &["gate-up", "activated"],
];
const ARENA_ALIGNMENT: usize = 1024;

impl LaneLayout {
    pub(crate) fn new(cfg: &KimiK3Config, rows: usize, max_context: usize,
        scratch: NativeScratch) -> Result<Self> {
        cfg.release_geometry()?;
        ensure!((1..=4096).contains(&rows) && (1..=1_048_576).contains(&max_context),
            "Kimi lane row/context capacity");
        let mut operands = BTreeMap::new();
        let mut plane = |name, shape: &[usize]| -> Result<()> {
            let bytes = shape.iter().try_fold(1usize, |n, size| n.checked_mul(*size)
                .ok_or_else(|| anyhow::anyhow!("Kimi lane {name} extent overflow")))?;
            ensure!(operands.insert(name, bytes.max(16)).is_none(), "duplicate Kimi lane operand");
            Ok(())
        };
        let hidden = [rows, cfg.hidden, 2];
        plane("residual arena", &[9, rows, cfg.hidden, 2])?;
        for name in ["delta", "normalized", "attention output", "ffn output", "shared output", "routed output", "combined output"] {
            plane(name, &hidden)?;
        }
        let intermediate = (cfg.dense_intermediate / 2).max(cfg.moe_intermediate * cfg.shared_experts / 2);
        plane("gate-up", &[rows, 2, intermediate, 2])?;
        plane("activated", &[rows, intermediate, 2])?;
        for name in ["latent input", "expert combined", "latent normalized"] {
            plane(name, &[rows, cfg.latent, 2])?;
        }
        plane("router logits", &[rows, cfg.experts, 4])?;
        for name in ["route ids", "route weights"] { plane(name, &[rows, cfg.topk, 4])?; }
        plane("expert wire", &[rows, KimiExpertInput::ROW_BYTES])?;
        plane("KDA normalized", &[rows, 48, 128, 2])?;
        for name in ["KDA slots", "KDA first"] { plane(name, &[rows, 4])?; }
        plane("MLA slots", &[rows, 8])?;
        plane("MLA query", &[rows, cfg.heads, 576, 2])?;
        plane("MLA q residual", &[rows, cfg.q_lora_rank, 2])?;
        plane("MLA attention", &[rows, cfg.heads, 512, 2])?;
        plane("MLA LSE", &[rows, cfg.heads, 4])?;
        // Decode/verify permits one request per row, at most the 64-row
        // specialization. Wide prefill is one sequence. Cache pages live outside.
        let requests = rows.min(64);
        plane("MLA page table", &[requests, max_context.div_ceil(64), 4])?;
        plane("MLA lengths", &[requests, 4])?;
        plane("MLA query offsets", &[requests + 1, 4])?;
        plane("KDA scratch", &[scratch.kda])?;
        for (name, bytes) in ["MLA producer scratch", "MLA attention scratch", "MLA output scratch"]
            .into_iter().zip(scratch.mla) { plane(name, &[bytes])?; }
        let mut arena_offsets = BTreeMap::new();
        let mut arena_bytes = 0;
        for phase in SCRATCH_PHASES {
            let mut end = 0usize;
            for &name in phase {
                let offset = end.checked_add(ARENA_ALIGNMENT - 1)
                    .map(|n| n / ARENA_ALIGNMENT * ARENA_ALIGNMENT)
                    .ok_or_else(|| anyhow::anyhow!("Kimi phase alignment overflow"))?;
                ensure!(arena_offsets.insert(name, offset).is_none(), "duplicate Kimi arena operand");
                end = offset.checked_add(operands[name])
                    .ok_or_else(|| anyhow::anyhow!("Kimi phase extent overflow"))?;
            }
            arena_bytes = arena_bytes.max(end);
        }
        let layout = Self { rows, max_context, operands, arena_offsets, arena_bytes };
        layout.bytes()?;
        Ok(layout)
    }

    pub(crate) fn bytes(&self) -> Result<usize> {
        self.operands.iter().filter(|(name, _)| !self.arena_offsets.contains_key(*name))
            .try_fold(self.arena_bytes, |sum, (_, bytes)| sum.checked_add(*bytes)
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
        let stream = Stream::new(device)?;
        let _scope = cuteafd_ffi::memory_ledger::scope("workspace/kimi lane");
        let arena = Allocation::new(device, self.arena_bytes)?;
        let operands = self.operands.iter().filter(|(name, _)| !self.arena_offsets.contains_key(*name))
            .map(|(name, bytes)| {
                let allocation = Allocation::new(device, *bytes)?;
                ensure!(allocation.buffer.bytes == *bytes && allocation.buffer.device_id == device.id,
                    "Kimi allocated lane operand {name} differs from admitted extent/device");
                Ok((*name, allocation))
            }).collect::<Result<BTreeMap<_, _>>>()?;
        let lane = Lane { stream: std::rc::Rc::new(stream), operands, arena: Some(arena), layout: self.clone() };
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
}

impl Drop for Lane<'_> {
    fn drop(&mut self) {
        // A shared state owner also retains the stream; do not defer the drain
        // until its last reference drops while these operands are freed now.
        self.stream.drain_or_abort("Kimi lane operands");
    }
}

impl Lane<'_> {
    pub(crate) fn buffer(&self, name: &str) -> Result<cuteafd_ffi::CuteafdDeviceBuffer> {
        if let Some(&offset) = self.layout.arena_offsets.get(name) {
            let arena = self.arena.as_ref().ok_or_else(|| anyhow::anyhow!("Kimi lane arena absent"))?;
            return view(arena.buffer, offset, self.layout.operands[name]);
        }
        self.operands.get(name).map(|allocation| allocation.buffer)
            .ok_or_else(|| anyhow::anyhow!("unknown Kimi lane operand {name}"))
    }

    pub(crate) fn rows(&self) -> usize { self.layout.rows }

    fn prime_attention(&self) -> Result<()> {
        for name in ATTENTION_METADATA {
            let buffer = self.buffer(name)?;
            self.stream.device.run(|| {
                // SAFETY: fresh admitted destinations and pinned sources stay live
                // through the ordered copy; this runs before graph capture.
                unsafe { self.stream.device.library.copy_h2d_ordered(buffer, &vec![0; buffer.bytes], self.stream.raw) }
            })?;
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
            Some(KimiAttention::Kda) => (weights.kda().is_ok(), self.kda.layer(layer, self.lane_index).is_ok()),
            Some(KimiAttention::Mla) => (weights.mla().is_ok(), self.mla.layer(layer).is_ok()),
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
        ensure!(!pool.is_parity() && pool.capacity().checked_add(pool.reserved()) == Some(self.mla.pages()),
            "Kimi metadata page pool differs from admitted MLA bank");
        let metadata = AttentionRows::new(sequences, self.lane.rows(), self.kda.slots(),
            self.lane.layout.max_context, pool)?;
        self.lane.upload_attention(&metadata)?;
        self.attention = Some(AttentionShape { rows: metadata.rows, sequences: metadata.sequences });
        Ok(())
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
                Op::Attention { layer, sequences } => {
                    self.attention = Some(AttentionShape { rows: key.rows, sequences });
                    self.attention(layer, key.rows, false)?;
                }
            }
            Ok(())
        });
        self.attention = staged;
        self.graphs = Some(graphs?);
        Ok(())
    }

    pub(crate) fn enable_replay(&mut self, enabled: bool) { self.replay_enabled = enabled; }

    fn replay(&mut self, rows: usize, op: super::graphs::Op) -> Result<bool> {
        if !self.replay_enabled { return Ok(false); }
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
        if !speculative && self.replay(rows, super::graphs::Op::Attention { layer, sequences })? {
            return self.plane("attention output", rows, 7168);
        }
        let weights = self.layer(layer)?;
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
            return self.plane("ffn output", rows, 7168);
        }
        let weights = self.layer(layer)?;
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
        let down = weights.buffer("latent.down")?;
        self.lane.stream.device.run(|| {
            self.programs.router.launch(rows, input, router, bias, logits, routes, self.lane.stream.raw)?;
            self.programs.latent.down(rows, input, down, latent, self.lane.stream.raw)?;
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
        let up_weight = weights.buffer("latent.up")?;
        self.lane.stream.device.run(|| {
            self.programs.latent.up(rows, combined, norm_weight, normalized, up_weight, output, self.lane.stream.raw)?;
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

const ATTENTION_METADATA: [&str; 6] = ["KDA slots", "KDA first", "MLA slots",
    "MLA page table", "MLA lengths", "MLA query offsets"];

#[derive(Clone, Copy)]
struct AttentionShape {
    rows: usize,
    sequences: usize,
}

/// Host metadata is rebuilt outside capture, then copied into fixed lane buffers.
/// For <=64 rows, MLA treats each query as one request with its own visible
/// length, so the native request scalar depends only on rows, not partitions.
struct AttentionRows {
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
    fn new(sequences: &[SequenceRows<'_>], capacity: usize, slots: usize, max_context: usize,
        pool: &cuteafd_engine::prefix::RefPagePool) -> Result<Self> {
        ensure!((1..=4096).contains(&capacity) && (1..=64).contains(&slots)
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
        ensure!(self.rows <= layout.rows && self.width == layout.max_context.div_ceil(64),
            "Kimi metadata differs from admitted row capacity/page stride");
        for (name, bytes) in self.uploads() {
            ensure!(layout.operands.get(name).is_some_and(|capacity| bytes.len() <= *capacity),
                "Kimi metadata exceeds admitted operand {name}");
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
}

impl IoLayout {
    pub(crate) fn new(cfg: &KimiK3Config, rank: usize, rows: usize) -> Result<Self> {
        ensure!(cfg.hidden == 7168 && cfg.vocab_size == 163840, "Kimi native IO hidden/vocabulary geometry");
        ensure!(rank < 2 && (1..=4096).contains(&rows), "Kimi IO rank/row geometry");
        Ok(Self { rank, rows, hidden: cfg.hidden, vocab: cfg.vocab_size })
    }

    fn extents(&self) -> [usize; 3] {
        if self.rank == 0 { [self.rows * 4, 0, 0] }
        else { [0, self.rows.min(64) * self.vocab * 4, VOCABULARY_HEAD_WORKSPACE] }
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
        weights.buffer(if self.rank == 0 { "embedding" } else { "head" })?;
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
            unsafe { self.stream.device.library.copy_h2d_ordered(view(ids, 0, data.len())?, &data, self.stream.raw)?; }
            // SAFETY: checked IDs, table and output extents; caller retains output through the lane drain.
            unsafe { self.stream.device.library.cuda_embedding_lookup_bf16_async(table, ids, output,
                tokens.len(), self.layout.vocab, self.layout.hidden, self.stream.raw) }
        })
    }

    /// Reuses at most 64 logits rows. Callers consume/download each span before
    /// the next projection; wide prefill never allocates a full-context logit tensor.
    pub(crate) fn project(&self, input: CuteafdDeviceBuffer, rows: usize) -> Result<CuteafdDeviceBuffer> {
        ensure!(self.layout.rank == 1 && (1..=self.layout.rows.min(64)).contains(&rows),
            "Kimi head rows exceed admitted IO capacity");
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
mod tests {
    use super::*;

    fn i32_metadata(bytes: &[u8]) -> Vec<i32> {
        bytes.chunks_exact(4).map(|value| i32::from_le_bytes(value.try_into().unwrap())).collect()
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
        let layout = LaneLayout::new(&cfg, 64, 256, NativeScratch { kda: 16, mla: [16; 3] })?;
        metadata.check_layout(&layout)?;
        for (name, bytes) in metadata.uploads() { assert!(bytes.len() <= layout.operands[name]); }
        let narrow = LaneLayout::new(&cfg, 64, 128, NativeScratch { kda: 16, mla: [16; 3] })?;
        assert!(metadata.check_layout(&narrow).is_err());
        let short = LaneLayout::new(&cfg, 4, 256, NativeScratch { kda: 16, mla: [16; 3] })?;
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
        let mut layout = LaneLayout::new(&cfg, 64, 128, NativeScratch { kda: 16, mla: [16; 3] })?;
        layout.operands.retain(|name, _| ATTENTION_METADATA.contains(name));
        layout.arena_offsets.clear();
        layout.arena_bytes = 0;
        let operands = device.run(|| layout.operands.iter().map(|(&name, &bytes)|
            Ok((name, Allocation::new(device, bytes)?))).collect::<Result<BTreeMap<_, _>>>())?;
        let lane = Lane { stream: Rc::new(Stream { device, raw: std::ptr::null_mut() }), operands, arena: None, layout };
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
            layout: LaneLayout { rows: 1, max_context: 64, operands: BTreeMap::from([("delta", 16)]),
                arena_offsets: BTreeMap::new(), arena_bytes: 32 } };
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
        let io = IoLayout { rank: 0, rows: 16, hidden: 4, vocab: 8 }
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
        for rows in [1, 16, 64, 4096] {
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
        for (rank, rows) in [(2, 1), (0, 0), (1, 4097)] {
            assert!(IoLayout::new(&cfg, rank, rows).is_err());
        }
        let source = include_str!("device.rs").split("#[cfg(test)]").next().unwrap();
        assert!(source.contains(".vocabulary_head_rows("));
        assert!(source.contains("crate::shared::target_head::launch("));
        assert!(!source.contains(".vocab_head_rows("));
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
            let layout = LaneLayout::new(&cfg, rows, 1_048_576, scratch)?;
            assert_eq!(layout.operands["residual arena"], KimiResidual::arena_bytes(rows)?);
            assert_eq!(layout.operands["gate-up"], rows * 2 * 16896 * 2);
            assert_eq!(layout.operands["MLA page table"], rows.min(64) * 16384 * 4);
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
        for (rows, context) in [(0, 1), (4097, 1), (1, 0), (1, 1_048_577)] {
            assert!(LaneLayout::new(&cfg, rows, context, scratch).is_err());
        }
        assert!(LaneLayout::new(&cfg, 1, 1, NativeScratch { kda: usize::MAX, mla: [0; 3] }).is_err());
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
            let layout = LaneLayout::new(&cfg, rows, 1_048_576,
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
