//! Device-bound native programs and residual scheduling for the coordinator.
//! The unbound operation backend still refuses with its owning work item
//! (#273), never substitute arithmetic.
use super::block::{MixSite, Ops, Routes};
use anyhow::{bail, ensure, Result};
use cuteafd_ffi::{programs::Programs, Bf16Add, KimiExpertInput, KimiFfn, KimiFfnBatch,
    KimiKda, KimiKdaCommit, KimiLatent, KimiMlaMode, KimiPrefixMode, KimiResidual,
    KimiResidualBatch, KimiResidualSite, KimiRmsNorm, KimiRouter};
use crate::shared::memory::device::{Allocation, Device, Stream};
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
    pub(crate) fn load(cfg: &KimiK3Config, programs: &'a Programs<'a>) -> Result<Self> {
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
        Ok(Self { residuals, dense: KimiFfn::load(programs, cfg.dense_intermediate / 2)?,
            shared: KimiFfn::load(programs, cfg.moe_intermediate * cfg.shared_experts / 2)?, cfg: cfg.clone() })
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
        programs: &'a Programs<'a>) -> Result<Self> {
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
            Ok(Self { device, w8: W8::load(cfg, programs)?, norm: KimiRmsNorm::load(programs, cfg.hidden)?,
                kda, commit: KimiKdaCommit::load(programs)?, mla, latent: KimiLatent::load(programs)?,
                router: KimiRouter::load(programs)?, wire: KimiExpertInput::load(programs)?,
                add: device.library.bf16_add()?, scratch })
        })
    }

    pub(crate) fn scratch(&self) -> NativeScratch { self.scratch }

    pub(crate) fn lane_layout(&self, cfg: &KimiK3Config, rows: usize, max_context: usize) -> Result<LaneLayout> {
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
    operands: BTreeMap<&'static str, usize>,
}

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
        let layout = Self { rows, operands };
        layout.bytes()?;
        Ok(layout)
    }

    pub(crate) fn bytes(&self) -> Result<usize> {
        self.operands.values().try_fold(0usize, |sum, bytes| sum.checked_add(*bytes)
            .ok_or_else(|| anyhow::anyhow!("Kimi lane total extent overflow")))
    }

    pub(crate) fn rows(&self) -> usize { self.rows }

    pub(crate) fn demand(&self, rank: u8, lane: usize) -> Result<Demand> {
        ensure!(rank < 2 && lane < 2, "Kimi lane rank/index outside TP2 two-lane executor");
        Ok(Demand::new(rank, Category::Workspace, format!("Kimi lane {lane} native operands"),
            u64::try_from(self.bytes()?)?, Basis::Exact))
    }

    pub(super) fn allocate<'a>(&self, device: Device<'a>) -> Result<Lane<'a>> {
        let stream = Stream::new(device)?;
        let _scope = cuteafd_ffi::memory_ledger::scope("workspace/kimi lane");
        let operands = self.operands.iter().map(|(name, bytes)| {
            let allocation = Allocation::new(device, *bytes)?;
            ensure!(allocation.buffer.bytes == *bytes && allocation.buffer.device_id == device.id,
                "Kimi allocated lane operand {name} differs from admitted extent/device");
            Ok((*name, allocation))
        }).collect::<Result<BTreeMap<_, _>>>()?;
        Ok(Lane { stream: std::rc::Rc::new(stream), operands, layout: self.clone() })
    }
}

/// Stream drops first and drains queued work before any operand is released,
/// including cancellation/unwind. No allocation occurs during a lane launch.
pub(crate) struct Lane<'a> {
    pub stream: std::rc::Rc<Stream<'a>>,
    operands: BTreeMap<&'static str, Allocation<'a>>,
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
        self.operands.get(name).map(|allocation| allocation.buffer)
            .ok_or_else(|| anyhow::anyhow!("unknown Kimi lane operand {name}"))
    }

    pub(crate) fn rows(&self) -> usize { self.layout.rows }
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
            assert_eq!(layout.bytes()?, layout.operands.values().sum::<usize>());
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
