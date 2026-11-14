//! GLM 5.x coordinator weights, packed for the exported glm_* programs.
//!
//! The checkpoint's FP8 projections stay FP8 (ModelOpt per-tensor FP8 as the
//! same bytes under a uniform block grid; a ModelOpt release's BF16 projections
//! are quantized to 128x128 blocks at load): E4M3 bytes (`{w}_fp8`) with
//! their FP32 128x128 scale grids (`{w}_scale`, `[ceil(N/128), K/128]`) are
//! the only copies of `w_qkv_a`, `w_q_b`, `w_iq`, `w_o`, `w_gate_up` and
//! `w_down` (decode programs: FP8 GEMV up to 16 rows, W8A16 GEMMs above, bitwise
//! the BF16 programs over the dequantized weights; prefill programs: W8A8 or
//! W8A16). Prefill reads `q_a|kv_a` (2624 rows, the last block partial) with
//! per-row scales K-block major (`w_qkv_a_kscale`, `[K/128, N]`). `kv_b` is
//! split per head on the host into E4M3 `w_uk_fp8 [N, 512, 192]` (key rows
//! transposed) and `w_uv_fp8 [N, 256, 512]` with one FP32 scale per weight
//! row and 64-wide K tile (`w_uk_scale [N, 512, 3]`, `w_uv_scale [N, 256, 8]`:
//! heads span 3.5 blocks of the grid). Row-concatenated operands (`w_qkv_a`,
//! `w_gate_up`) concatenate their grids (every part but the last is a whole
//! number of 128-row blocks). BF16 operands: norms, the router, the LM head and
//! `w_ik` (`wk` FP8 dequantized on the GPU with its scales, `weights_proj` BF16).
use crate::shared::memory::{DeviceAllocation, Tag, device::Device};
use crate::shared::dense_trellis::{CoreWeightLayout, CoreWeights};
use cuteafd_loader::families::glm5::core::CoreInventory;
use cuteafd_loader::families::glm5::names::Operand as Op;
use anyhow::{ensure, Context, Result};
use cuteafd_core::DType;
use cuteafd_ffi::{CuteafdDeviceBuffer, NativeLibrary};
use cuteafd_loader::families::glm5::{GlmDsaConfig, GlmIndexer};
use cuteafd_loader::OfficialV41Catalog;
use std::collections::HashMap;
use std::ffi::c_void;
use std::os::unix::fs::FileExt;
use crate::shared::peer_split::{slice_2d, Axis};
use super::landing::{self, Land};
use cuteafd_core::memory_layout::ArenaTag;

pub(crate) struct GlmLayer<'a> {
    pub dense: bool,
    pub full_indexer: bool,
    pub mlp_numerics: cuteafd_loader::plan::checkpoint::ModuleNumerics,
    /// Dense/shared FFNs cover this GPU's intermediate slice (`glm2` programs).
    pub split: bool,
    /// q_b, kv_b and o_proj cover this GPU's heads, producing a partial sum.
    pub attention_split: bool,
    /// Whole attention and its cache exist only on this rank when present.
    pub attention_owner: Option<u8>,
    /// Set only after this rank's attention operands have loaded successfully.
    attention_resident: bool,
    operands: HashMap<&'static str, DeviceAllocation<'a>>,
    /// The EXLR8 checkpoint's own dense operands (`requests::key`), when
    /// the layer runs natively; empty on the official path.
    pub(crate) native: HashMap<String, DeviceAllocation<'a>>,
    pub(crate) native_projections: HashMap<&'static str, cuteafd_loader::plan::exlr8::dense::requests::NativeProjection>,
    /// Standard HF K6 payload/vector owners, separate from EXLR8 encodings.
    pub(crate) core: std::collections::BTreeMap<&'static str, CoreWeights<'a>>,
    pub(crate) attention_native: std::collections::BTreeMap<&'static str, crate::shared::memory::device::DeviceOwner<'a, crate::shared::native_attention::NativeAttentionWeights<'a>>>,
}

/// The `{w}_fp8` / `{w}_scale` operand names of an FP8 weight, and its
/// per-row K-block-major scales `{w}_kscale` (prefill `w_qkv_a` only).
pub(crate) fn fp8_operand_names(name: &str) -> (&'static str, &'static str, &'static str) {
    match name {
        "w_qkv_a" => ("w_qkv_a_fp8", "w_qkv_a_scale", "w_qkv_a_kscale"),
        "w_q_b" => ("w_q_b_fp8", "w_q_b_scale", ""),
        "w_iq" => ("w_iq_fp8", "w_iq_scale", ""),
        "w_o" => ("w_o_fp8", "w_o_scale", ""),
        "w_gate_up" => ("w_gate_up_fp8", "w_gate_up_scale", ""),
        "w_down" => ("w_down_fp8", "w_down_scale", ""),
        "w_uk" => ("w_uk_fp8", "w_uk_scale", ""),
        "w_uv" => ("w_uv_fp8", "w_uv_scale", ""),
        other => unreachable!("{other} has no FP8 operand"),
    }
}

/// Native replacements occupy their original projection's place in the read
/// order, before the prefetch budget is applied (never an unordered tail).
fn decode_ranges(range: impl Fn(&str) -> Option<crate::shared::l2_prefetch::Range>,
    projection: impl Fn(&str) -> Vec<crate::shared::l2_prefetch::Range>)
    -> Vec<crate::shared::l2_prefetch::Range> {
    let mut out = Vec::new();
    let order: &[(&str, &[&str])] = &[
        ("input_norm", &[]), ("w_qkv_a", &["q_a", "kv_a"]), ("q_a_norm", &[]), ("kv_a_norm", &[]),
        ("w_q_b", &["q_b"]), ("w_iq", &["index_q"]), ("w_ik", &["index_k", "index_w"]),
        ("k_norm_w", &[]), ("k_norm_b", &[]), ("w_uk", &[]), ("w_uv", &[]), ("w_o", &["o"]),
        ("post_norm", &[]), ("gate", &[]), ("gate.bias", &[]), ("w_gate_up", &["gate", "up"]),
        ("w_down", &["down"]),
    ];
    for &(name, projections) in order {
        let native = projections.iter().flat_map(|name| projection(name)).collect::<Vec<_>>();
        if native.is_empty() {
            out.extend(crate::shared::l2_prefetch::operands(&[name], &range));
        } else { out.extend(native); }
    }
    out
}

pub(super) fn head_read_order(rank: usize, norm: crate::shared::l2_prefetch::Range,
    head: impl FnOnce() -> Vec<crate::shared::l2_prefetch::Range>) -> Vec<crate::shared::l2_prefetch::Range> {
    if rank == 0 { std::iter::once(norm).chain(head()).collect() } else { Vec::new() }
}

fn trellis_ranges(payload: CuteafdDeviceBuffer, suh: CuteafdDeviceBuffer, svh: CuteafdDeviceBuffer)
    -> Vec<crate::shared::l2_prefetch::Range> {
    [suh, payload, svh].map(|b| (b.ptr.cast_const(), b.bytes)).to_vec()
}

fn native_projection_ranges(name: &str, range: impl Fn(&str) -> Option<crate::shared::l2_prefetch::Range>)
    -> Vec<crate::shared::l2_prefetch::Range> {
    // Input transform, payload, output transform; scaled native formats keep
    // their payload/scale order. Only this rank's resident components enter.
    ["suh", "trellis", "svh", "q4", "s4", "fp8", "e8m0", "scale", "bf16", "fp32"]
        .into_iter().filter_map(|component| range(&cuteafd_loader::plan::exlr8::dense::requests::key(name, component))).collect()
}

impl GlmLayer<'_> {
    pub(crate) fn allocation_receipts(&self, inventory: &mut crate::shared::memory::receipts::Inventory<'_>, prefix: &str) -> Result<()> {
        for (name, owner) in &self.operands { owner.inventory(inventory, &format!("{prefix}.operand.{name}"))?; }
        for (name, owner) in &self.native { owner.inventory(inventory, &format!("{prefix}.native.{name}"))?; }
        for (name, weights) in &self.core {
            for (component, owner) in [("payload", &weights.payload), ("suh", &weights.suh), ("svh", &weights.svh)] {
                owner.inventory(inventory, &format!("{prefix}.core.{name}.{component}"))?;
            }
        }
        for (name, weights) in &self.attention_native {
            weights.values.inventory(inventory, &format!("{prefix}.native.{name}.values"))?;
            if let Some(scales) = &weights.scales { scales.inventory(inventory, &format!("{prefix}.native.{name}.scales"))?; }
        }
        Ok(())
    }

    pub(crate) fn allocation_buffers(&self) -> Vec<(String, CuteafdDeviceBuffer)> {
        let mut out = self.operands.iter().map(|(name, owner)| (format!("operand.{name}"), owner.buffer))
            .chain(self.native.iter().map(|(name, owner)| (format!("native.{name}"), owner.buffer)))
            .collect::<Vec<_>>();
        for (name, weights) in &self.core {
            for (component, owner) in [("payload", &weights.payload), ("suh", &weights.suh), ("svh", &weights.svh)] {
                out.push((format!("core.{name}.{component}"), owner.buffer));
            }
        }
        for (name, weights) in &self.attention_native {
            out.push((format!("native.{name}.values"), weights.values.buffer));
            if let Some(scales) = &weights.scales {
                out.push((format!("native.{name}.scales"), scales.buffer));
            }
        }
        out.sort_by(|a, b| a.0.cmp(&b.0));
        out
    }

    /// The loader's attention residency controls cache and prefix-snapshot ownership.
    pub fn holds_attention(&self) -> bool {
        self.attention_resident
    }

    /// The device range of `operand`, when the layer has it.
    pub fn range(&self, operand: &str) -> Option<crate::shared::l2_prefetch::Range> {
        self.operands.get(operand).map(|a| (a.buffer.ptr.cast_const(), a.buffer.bytes))
    }

    pub fn decode_read_order(&self) -> Vec<crate::shared::l2_prefetch::Range> {
        decode_ranges(|name| self.range(name).or_else(|| self.native.get(name)
            .map(|a| (a.buffer.ptr.cast_const(), a.buffer.bytes))), |name| {
            if let Some(w) = self.core.get(name) {
                return trellis_ranges(w.payload.buffer, w.suh.buffer, w.svh.buffer);
            }
            if let Some(weight) = self.attention_native.get(name) {
                return std::iter::once(&weight.values).chain(weight.scales.as_ref())
                    .map(|a| (a.buffer.ptr.cast_const(), a.buffer.bytes)).collect();
            }
            native_projection_ranges(name, |key| self.native.get(key)
                .map(|a| (a.buffer.ptr.cast_const(), a.buffer.bytes)))
        })
    }

    pub fn ptr(&self, operand: &str) -> Result<*mut c_void> {
        Ok(self.operands.get(operand).with_context(|| format!("layer has no weight {operand}"))?.buffer.ptr)
    }

    /// Device bytes of the layer's operands.
    pub fn bytes(&self) -> usize {
        self.operands.values().chain(self.native.values()).map(|a| a.buffer.bytes).sum::<usize>()
            + self.core.values().map(|w| w.payload.buffer.bytes + w.suh.buffer.bytes + w.svh.buffer.bytes).sum::<usize>()
            + self.attention_native.values().map(|w| w.values.buffer.bytes + w.scales.as_ref().map_or(0, |s| s.buffer.bytes)).sum::<usize>()
    }

    /// A native operand's device buffer.
    pub fn native(&self, key: &str) -> Result<CuteafdDeviceBuffer> {
        Ok(self.native.get(key).with_context(|| format!("layer has no native operand {key}"))?.buffer)
    }

    /// Whether the layer runs the checkpoint's native dense operands.
    pub fn is_native(&self) -> bool { !self.native.is_empty() }
}

pub(crate) struct GlmWeights<'a> {
    pub layers: Vec<GlmLayer<'a>>,
    pub norm: DeviceAllocation<'a>,
    pub head: GlmHead<'a>,
    /// Retained through group commit and both ranks' rotary-table construction.
    pub rope_alpha: Option<RopeAlpha<'a>>,
}

pub(crate) struct RopeAlpha<'a> {
    storage: crate::shared::memory::HostAllocation<'a>,
    landing: crate::shared::memory::HostLanding,
}
impl RopeAlpha<'_> {
    pub(crate) fn values(&self) -> Result<Vec<f32>> {
        ensure!(self.landing.ready.load(std::sync::atomic::Ordering::Acquire),
            "Mistral Large 4 lerope.alpha read before load-group commit");
        Ok(self.storage.bytes().chunks_exact(2).map(|bytes|
            f32::from_bits(u32::from(u16::from_le_bytes([bytes[0], bytes[1]])) << 16)).collect())
    }
}

/// The vocabulary head rank 0 owns: the checkpoint's BF16 head (the shared
/// tensor-core route), or an EXLR8 checkpoint's row-FP8 head (its own program).
pub(crate) enum GlmHead<'a> {
    Bf16(DeviceAllocation<'a>),
    /// Shared with the token embedding (one rank-0 owner of both tables).
    RowFp8(std::rc::Rc<crate::shared::token_io::native::NativeVocabulary<'a>>),
    K6(CoreWeights<'a>),
}

impl<'a> GlmHead<'a> {
    /// Device bytes of the head only; TokenEmbedding charges the embedding table.
    pub fn bytes(&self) -> usize {
        match self { Self::Bf16(head) => head.buffer.bytes, Self::RowFp8(owner) => owner.head.residency().bytes,
            Self::K6(head) => head.payload.buffer.bytes + head.suh.buffer.bytes + head.svh.buffer.bytes }
    }

    /// All resident head components, in read order; rank 1 never calls this.
    pub fn ranges(&self) -> Vec<crate::shared::l2_prefetch::Range> {
        match self {
            Self::Bf16(head) => vec![(head.buffer.ptr.cast_const(), head.buffer.bytes)],
            Self::RowFp8(owner) => [owner.head.payload(), owner.head.scales()]
                .map(|b| (b.ptr.cast_const(), b.bytes)).to_vec(),
            Self::K6(head) => trellis_ranges(head.payload.buffer, head.suh.buffer, head.svh.buffer),
        }
    }

    /// The BF16 head, for diagnostics that read it directly.
    pub fn bf16(&self) -> Result<&DeviceAllocation<'a>> {
        match self { Self::Bf16(head) => Ok(head),
            Self::RowFp8(_) => anyhow::bail!("this diagnostic reads a BF16 head; the EXLR8 checkpoint's head is row-FP8"),
            Self::K6(_) => anyhow::bail!("this diagnostic reads a BF16 head; the checkpoint's head is native K6") }
    }

    /// The head the DFlash2 drafter borrows (its only resident copy).
    pub fn target(&self) -> Result<crate::shared::speculation::dflash::TargetHead<'_>> {
        use crate::shared::speculation::dflash::TargetHead;
        Ok(match self {
            Self::Bf16(head) => TargetHead::Bf16(head),
            // SAFETY (inside): the drafter passes its live normed rows and logits on its stream.
            Self::RowFp8(owner) => TargetHead::Launch(Box::new(move |x, logits, rows, stream| unsafe {
                owner.head_logits(x, logits, rows, stream)
            })),
            Self::K6(_) => anyhow::bail!("native K6 target head needs the drafter's admitted projection scratch"),
        })
    }
}

#[derive(Clone)]
pub(crate) struct GlmLoader<'a> {
    pub library: &'a NativeLibrary,
    pub catalog: &'a OfficialV41Catalog,
    pub stream: *mut c_void,
    /// This loader's device (rank 0 of a head split).
    pub device: i32,
    /// The other GPU of a head split (rank 1), if any.
    pub peers: Vec<crate::shared::peer_split::RankDevice>,
    /// Push mode: class-2 operands are allocated and recorded, filled later by
    /// the push group; transforms of their bytes wait for the commit.
    pub push: Option<std::rc::Rc<super::landing::PushSink>>,
    pub arenas: Option<std::rc::Rc<std::collections::BTreeMap<i32, crate::shared::memory::weight_arena::DeviceWeightArena<'a>>>>,
    pub allocation_prefix: String,
}

#[path = "weight_routes.rs"]
mod routes;

fn f32_bytes(values: &[f32]) -> Vec<u8> {
    let mut out = vec![0u8; values.len() * 4];
    for (dst, v) in out.chunks_exact_mut(4).zip(values) {
        dst.copy_from_slice(&v.to_le_bytes());
    }
    out
}

pub(super) fn peer_router_names(cfg: &GlmDsaConfig, layer: usize) -> impl Iterator<Item = (&'static str, String)> + '_ {
    [("gate", Op::Router), ("gate.bias", Op::RouterBias)].into_iter()
        .filter_map(move |(operand, role)| cfg.weight(layer, role).map(|name| (operand, name)))
}

impl<'a> GlmLoader<'a> {
    pub(crate) fn allocate(&self, name: &str, bytes: usize) -> Result<DeviceAllocation<'a>> {
        Ok(DeviceAllocation::from_allocation(self.allocate_owner(name, bytes)?))
    }

    pub(crate) fn allocate_owner(&self, name: &str, bytes: usize) -> Result<crate::shared::memory::device::Allocation<'a>> {
        let device = Device { library: self.library, id: self.library.cuda_get_device()? };
        let arena = self.arenas.as_ref().map(|arenas| arenas.get(&device.id)
            .context("GLM current device has no admitted target arena")).transpose()?;
        let rank = arena.map(|arena| arena.layout().id().rsplit('/').next().unwrap()).unwrap_or("0");
        let name = if name == "embedding" { "glm.weights.embedding".into() }
            else { format!("glm.weights.gpu.{}.{}{name}", rank, self.allocation_prefix) };
        crate::shared::memory::weight_arena::weight_allocation(device, arena, &name, bytes, ArenaTag::weights("glm/target"))
    }

    fn scoped(&self, prefix: String) -> Self { Self { allocation_prefix: prefix, ..self.clone() } }

    pub(crate) fn rope_alpha(&self, cfg: &GlmDsaConfig) -> Result<RopeAlpha<'a>> {
        let tensor = self.catalog.tensor("lerope.alpha")?;
        ensure!(tensor.metadata.dtype == DType::Bf16
            && tensor.metadata.shape == [cfg.qk_rope_head_dim / 2]
            && tensor.metadata.byte_length == cfg.qk_rope_head_dim as u64,
            "lerope.alpha must be BF16 [rope/2]");
        let _owner = cuteafd_ffi::memory_ledger::owner("glm.load.rope_alpha");
        let mut storage = crate::shared::memory::HostAllocation::new(self.library,
            cfg.qk_rope_head_dim, Tag::staging("glm/load/rope-alpha"))?;
        let landing = crate::shared::memory::HostLanding::new(&mut storage);
        crate::shared::experts::rtx::dense::host_operand(crate::shared::experts::rtx::dense::HostOperand {
            device: self.device, landing: landing.clone(),
            rect: Land::whole(self.catalog, "lerope.alpha", 0)?.rect(self.catalog)?.0,
        }, crate::shared::experts::rtx::push::PushPolicy::for_snapshot(self.catalog.snapshot())?)?;
        Ok(RopeAlpha { storage, landing })
    }

    /// Replicate only admitted local routers into the peer's target arena.
    pub(crate) fn peer_routers(&self, cfg: &GlmDsaConfig, layers: &mut [GlmLayer<'a>], range: std::ops::Range<usize>) -> Result<()> {
        let peer = *self.peers.first().context("TP2 router replica needs peer")?;
        let native = (self.catalog.format() == cuteafd_loader::plan::checkpoint::CheckpointFormat::Exlr8)
            .then(|| cuteafd_loader::plan::exlr8::dense::DenseRecord::inventory(self.catalog.snapshot())).transpose()?;
        for index in range {
            for (operand, name) in peer_router_names(cfg, index) {
                let allocation = crate::shared::peer_split::on_device(self.library, peer.device, || {
                    let _scope = cuteafd_ffi::memory_ledger::scope("experts/TP2 router replica");
                    let scoped = self.scoped(format!("layer.{index}.operand."));
                    if let Some(records) = &native {
                        let record = records.iter().find(|record| record.tensor == name)
                            .with_context(|| format!("EXLR8 local router missing {name}"))?;
                        ensure!(record.components.len() == 1, "EXLR8 router must be one native tensor");
                        let load = crate::shared::experts::rtx::dense::DenseLoad::mount(self.library);
                        crate::shared::experts::exlr8::dense_weights::landed_with(self.library, &load,
                            &name, &record.components[0], cuteafd_loader::plan::exlr8::dense::DenseTransform::Identity,
                            peer.stream, |_, bytes| scoped.allocate_owner(operand, bytes))
                    } else { scoped.one(operand, &name) }
                })?;
                layers[index].operands.insert(operand, allocation);
            }
        }
        Ok(())
    }

    fn raw(&self, name: &str) -> Result<(Vec<u8>, DType, Vec<usize>)> {
        cuteafd_ffi::memory_ledger::tensor(name);
        let tensor = self.catalog.tensor(name)?;
        let mut bytes = vec![0u8; tensor.metadata.byte_length as usize];
        crate::shared::memory::load_timing::read(bytes.len(), || std::fs::File::open(self.catalog.snapshot().join(&tensor.shard))?
            .read_exact_at(&mut bytes, tensor.metadata.byte_offset))
            .with_context(|| format!("reading {name}"))?;
        Ok((bytes, tensor.metadata.dtype.clone(), tensor.metadata.shape.clone()))
    }

    /// One operand on this loader's current device filled from checkpoint
    /// rectangles that land unchanged (class 2).
    fn landed(&self, key: &str, lands: &[Land], bytes: usize) -> Result<DeviceAllocation<'a>> {
        if let Some(push) = &self.push {
            let allocation = self.allocate(key, bytes.max(256))?;
            push.record(self.catalog, self.library.cuda_get_device()?, allocation.buffer, lands, bytes)?;
            return Ok(allocation);
        }
        self.upload(key, &landing::read(self.catalog, lands, bytes)?)
    }

    /// `lands` replicated on every rank (one read, rank 0 upload, peer copies;
    /// pushed, each rank receives its own copy).
    fn landed_all(&self, key: &str, ranks: usize, lands: &[Land], bytes: usize) -> Result<Vec<DeviceAllocation<'a>>> {
        if self.push.is_some() {
            return (0..ranks).map(|rank| self.on_rank(rank, |_| self.landed(key, lands, bytes))).collect();
        }
        self.upload_all(key, ranks, &landing::read(self.catalog, lands, bytes)?)
    }

    /// Whether every part is E4M3 with its own FP32 block grid.
    fn native_fp8(&self, names: &[String]) -> Result<bool> {
        names.iter().try_fold(true, |all, name| Ok(all && landing::native_fp8_block(self.catalog, name)?))
    }

    fn upload(&self, key: &str, bytes: &[u8]) -> Result<DeviceAllocation<'a>> {
        let allocation = self.allocate(key, bytes.len().max(256))?;
        crate::shared::memory::load_timing::upload(bytes.len(), || self.library.copy_h2d(allocation.buffer, bytes))?;
        Ok(allocation)
    }

    /// The row-concatenation of 2-D `names` as one BF16 operand (FP8 parts
    /// dequantized on the GPU with their FP32 block scales).
    fn rows(&self, key: &str, names: &[String]) -> Result<DeviceAllocation<'a>> {
        let _memory_format = cuteafd_ffi::memory_ledger::format("bf16");
        let tensors = names.iter().map(|n| self.raw(n).map(|t| (n, t))).collect::<Result<Vec<_>>>()?;
        let cols = tensors[0].1 .2[1];
        let rows: usize = tensors.iter().map(|(_, (_, _, shape))| shape[0]).sum();
        ensure!(tensors.iter().all(|(_, (_, _, s))| s.len() == 2 && s[1] == cols), "{names:?} do not share columns");
        let out = self.allocate(key, rows * cols * 2)?;
        let k_blocks = cols.div_ceil(128);
        let mut row = 0;
        for (name, (bytes, dtype, shape)) in &tensors {
            let dest = CuteafdDeviceBuffer {
                // SAFETY: rows row..row+shape[0] lie inside `out`.
                ptr: unsafe { out.buffer.ptr.cast::<u8>().add(row * cols * 2) }.cast(),
                bytes: shape[0] * cols * 2,
                ..out.buffer
            };
            match dtype {
                DType::Bf16 => crate::shared::memory::load_timing::upload(bytes.len(), || self.library.copy_h2d(dest, bytes))?,
                DType::F8E4M3 => {
                    let (scale, scale_dtype, _) = self.raw(&landing::block_grid(self.catalog, name))?;
                    ensure!(scale_dtype == DType::F32, "{name}: block scales must be FP32");
                    ensure!(scale.len() == shape[0].div_ceil(128) * k_blocks * 4, "{name}: unexpected scale grid");
                    let temporary = GlmLoader { arenas: None, ..self.clone() };
                    let (w, s) = (temporary.upload("transient.values", bytes)?, temporary.upload("transient.scales", &scale)?);
                    // SAFETY: fp8 [shape], its scale grid and the destination rows
                    // are live; the stream is drained before the temporaries drop.
                    unsafe {
                        self.library.fp8_block_dequant(w.buffer.ptr, s.buffer.ptr, dest.ptr, shape[0], cols, self.stream)?;
                        self.library.cuda_stream_synchronize(self.stream)?;
                    }
                }
                other => anyhow::bail!("{name}: unsupported coordinator dtype {other:?}"),
            }
            row += shape[0];
        }
        Ok(out)
    }

    /// Reads tensor `name`'s bytes into `out[at..]` (one positioned read into the
    /// operand's buffer: no second copy) and returns its byte length, dtype and shape.
    fn read_into(&self, name: &str, out: &mut [u8], at: usize) -> Result<(usize, DType, Vec<usize>)> {
        cuteafd_ffi::memory_ledger::tensor(name);
        let tensor = self.catalog.tensor(name)?;
        let length = tensor.metadata.byte_length as usize;
        ensure!(at + length <= out.len(), "{name}: {length} bytes past the operand buffer");
        crate::shared::memory::load_timing::read(length, || std::fs::File::open(self.catalog.snapshot().join(&tensor.shard))?
            .read_exact_at(&mut out[at..at + length], tensor.metadata.byte_offset))
            .with_context(|| format!("reading {name}"))?;
        Ok((length, tensor.metadata.dtype.clone(), tensor.metadata.shape.clone()))
    }

    /// Reads the row-concatenation of the FP8 checkpoint weights `names` into
    /// this thread's staging buffer and hands `body` the E4M3 bytes `[N, K]`,
    /// the FP32 block grids `[ceil(N/128), K/128]` (every part but the last a
    /// whole number of 128-row blocks), N and K.
    fn with_fp8<T>(&self, names: &[String], body: impl FnOnce(&[u8], Vec<f32>, usize, usize) -> Result<T>)
        -> Result<T> {
        let _memory_format = cuteafd_ffi::memory_ledger::format("fp8");
        // BF16 parts need room for their padded rows while they quantize in place.
        let total: u64 = names.iter().map(|n| self.catalog.tensor(n).map(|t| t.metadata.byte_length))
            .sum::<Result<u64>>()? + 128 * 8192 * 2;
        crate::shared::memory::staging::with_staging(total as usize, |values| {
            let mut grid = Vec::new();
            let (mut rows, mut cols, mut at) = (0usize, None, 0usize);
            for (i, name) in names.iter().enumerate() {
                let (length, dtype, shape) = self.read_into(name, values, at)?;
                ensure!(shape.len() == 2 && matches!(dtype, DType::F8E4M3 | DType::Bf16),
                    "{name}: the glm programs take this weight as FP8 (or BF16 quantized at load), found {dtype:?} {shape:?}");
                ensure!(cols.is_none_or(|c| c == shape[1]), "{names:?} do not share columns");
                ensure!(i + 1 == names.len() || shape[0] % 128 == 0,
                    "{names:?}: only the last concatenated weight may end inside a 128-row block");
                cols = Some(shape[1]);
                let (n, k) = (shape[0], shape[1]);
                let block_rows = n.div_ceil(128);
                if dtype == DType::Bf16 {
                    // An unquantized weight of a ModelOpt release (nvidia/GLM-5.3-NVFP4 keeps
                    // attention and the shared experts BF16): 128x128 E4M3 blocks at load,
                    // power-of-two scales (the glmf rule), rows padded to whole blocks.
                    let mut padded = values[at..at + length].to_vec();
                    padded.resize(block_rows * 128 * k * 2, 0);
                    let (q, scales) = crate::families::glm5_flash::fp8::quantize(&padded, block_rows * 128, k,
                        crate::families::glm5_flash::fp8::Layout::Block, crate::shared::fp8_linear::Fp8Scales::Pow2);
                    values[at..at + n * k].copy_from_slice(&q[..n * k]);
                    grid.extend(scales);
                } else if self.catalog.tensor(&landing::block_grid(self.catalog, name)).is_ok() {
                    let (scale, scale_dtype, scale_shape) = self.raw(&landing::block_grid(self.catalog, name))?;
                    ensure!(scale_dtype == DType::F32 && scale_shape == [block_rows, k.div_ceil(128)],
                        "{name}: expected FP32 128x128 block scales, found {scale_dtype:?} {scale_shape:?}");
                    grid.extend(scale.chunks_exact(4).map(|b| f32::from_le_bytes(b.try_into().unwrap())));
                } else {
                    // ModelOpt per-tensor FP8 (one FP32 `weight_scale`): the same E4M3 bytes
                    // with every block scale equal to it, exactly (decode GEMVs and W8A16).
                    // Prefill W8A8 takes its static input_scale (`tensor_scales`).
                    let (scale, scale_dtype, _) = self.raw(&format!("{name}_scale"))
                        .with_context(|| format!("{name}: an FP8 weight needs weight_scale_inv blocks or a weight_scale"))?;
                    ensure!(scale_dtype == DType::F32 && scale.len() == 4, "{name}_scale: expected one FP32 value");
                    let value = f32::from_le_bytes(scale[..4].try_into().unwrap());
                    grid.extend(std::iter::repeat_n(value, block_rows * k.div_ceil(128)));
                }
                at += n * k;
                rows += n;
            }
            // The E4M3 bytes only (BF16 parts left their staging tail unused).
            body(&values[..at], grid, rows, cols.context("no FP8 rows")?)
        })
    }

    /// The prefill programs' FP32 `[12]` tensor-scale operand of row-concatenated ModelOpt
    /// per-tensor FP8 weights `parts` (E4M3, one FP32 `weight_scale`, one calibrated
    /// `input_scale` shared by the parts): `input_scale` at 0 and `input_scale * weight_scale`
    /// of part `i` at `4 * (i + 1)`. None unless every part is stored that way.
    fn tensor_scales(&self, parts: &[String]) -> Result<Option<Vec<u8>>> {
        ensure!(parts.len() <= 2, "{parts:?}: the tensor-scale operand holds two parts");
        let scalar = |name: &str| -> Result<Option<f32>> {
            if self.catalog.tensor(name).is_err() {
                return Ok(None);
            }
            let (bytes, dtype, _) = self.raw(name)?;
            ensure!(dtype == DType::F32 && bytes.len() == 4, "{name}: expected one FP32 value");
            Ok(Some(f32::from_le_bytes(bytes[..4].try_into().unwrap())))
        };
        let mut out = [0f32; 12];
        let mut input = None;
        for (i, name) in parts.iter().enumerate() {
            let Some(stem) = name.strip_suffix("weight") else { return Ok(None) };
            if self.catalog.tensor(name)?.metadata.dtype != DType::F8E4M3 {
                return Ok(None);
            }
            let (Some(weight_scale), Some(input_scale)) =
                (scalar(&format!("{name}_scale"))?, scalar(&format!("{stem}input_scale"))?) else { return Ok(None) };
            ensure!(input.is_none_or(|s| s == input_scale),
                "{parts:?}: concatenated per-tensor FP8 parts need one input_scale");
            input = Some(input_scale);
            out[4 * (i + 1)] = input_scale * weight_scale;
        }
        out[0] = input.context("no parts")?;
        Ok(Some(f32_bytes(&out)))
    }

    /// E4M3 `[N, K]` and the FP32 block grid of `names` on the device.
    fn fp8(&self, key: &str, names: &[String]) -> Result<(DeviceAllocation<'a>, DeviceAllocation<'a>)> {
        let _memory_format = cuteafd_ffi::memory_ledger::format("fp8");
        if let Some(((values, values_bytes), (grid, grid_bytes))) = self.fp8_lands(names)? {
            return Ok((self.landed(fp8_operand_names(key).0, &values, values_bytes)?, self.landed(fp8_operand_names(key).1, &grid, grid_bytes)?));
        }
        self.with_fp8(names, |values, grid, _, _| Ok((self.upload(fp8_operand_names(key).0, values)?, self.upload(fp8_operand_names(key).1, &f32_bytes(&grid))?)))
    }

    /// Native block FP8 `names` row-concatenated: the E4M3 values and the
    /// FP32 grids land unchanged. None for any other storage (quantized at load).
    #[allow(clippy::type_complexity)]
    fn fp8_lands(&self, names: &[String]) -> Result<Option<((Vec<Land>, usize), (Vec<Land>, usize))>> {
        if !self.native_fp8(names)? { return Ok(None); }
        let cols = names.iter().map(|n| Ok(self.catalog.tensor(n)?.metadata.shape[1])).collect::<Result<Vec<_>>>()?;
        ensure!(cols.windows(2).all(|w| w[0] == w[1]), "{names:?} do not share columns");
        for (i, name) in names.iter().enumerate() {
            ensure!(i + 1 == names.len() || self.catalog.tensor(name)?.metadata.shape[0] % 128 == 0,
                "{names:?}: only the last concatenated weight may end inside a 128-row block");
        }
        let grids = names.iter().map(|n| landing::block_grid(self.catalog, n)).collect::<Vec<_>>();
        Ok(Some((landing::concat(self.catalog, names)?, landing::concat(self.catalog, &grids)?)))
    }

    /// `w_uk [N, 512, D]` / `w_uv [N, V, 512]` E4M3 slices of the FP8 `kv_b`
    /// with one scale per weight row and 64-wide K tile (`[N, 512, D/64]`,
    /// `[N, V, 512/64]`): `w_uk[h,c,d] = kv_b[h*(D+V)+d, c]`,
    /// `w_uv[h,v,c] = kv_b[h*(D+V)+D+v, c]`.
    /// Over `ranks` GPUs each takes its heads' slices (head-major, so contiguous).
    #[allow(clippy::type_complexity)]
    fn kv_b(&self, cfg: &GlmDsaConfig, prefix: &str, ranks: usize)
        -> Result<Vec<((DeviceAllocation<'a>, DeviceAllocation<'a>), (DeviceAllocation<'a>, DeviceAllocation<'a>))>> {
        let _memory_format = cuteafd_ffi::memory_ledger::format("fp8");
        let (heads, d, v, c) = (cfg.heads, cfg.qk_nope_head_dim, cfg.v_head_dim, cfg.kv_lora_rank);
        self.with_fp8(&[format!("{prefix}.self_attn.kv_b_proj.weight")], |kv_b, grid, rows, cols| {
        ensure!(rows == heads * (d + v) && cols == c && d % 64 == 0 && c % 128 == 0,
            "{prefix}: kv_b_proj [{rows}, {cols}] does not match {heads} heads x ({d} + {v}) x {c}");
        let kb = c / 128;
        let scale = |row: usize, col: usize| grid[(row / 128) * kb + col / 128];
        // Each 64-row K tile of a key slice lies in one 128-row block (448 = 3.5 x 128).
        ensure!((d + v) % 64 == 0, "{prefix}: head rows must be whole 64-row tiles");
        let (mut uk, mut uk_s) = (vec![0u8; heads * c * d], vec![0f32; heads * c * d / 64]);
        let (mut uv, mut uv_s) = (vec![0u8; heads * v * c], vec![0f32; heads * v * c / 64]);
        // One head per task: key rows transposed in 64 x 64 tiles, value rows copied.
        std::thread::scope(|scope| {
            let parts = uk.chunks_mut(c * d).zip(uk_s.chunks_mut(c * d / 64)).zip(uv.chunks_mut(v * c))
                .zip(uv_s.chunks_mut(v * c / 64)).enumerate();
            let threads = std::thread::available_parallelism().map_or(8, |p| p.get()).clamp(1, 16);
            let mut groups: Vec<Vec<_>> = (0..threads).map(|_| Vec::new()).collect();
            for (h, part) in parts {
                groups[h % threads].push((h, part));
            }
            for group in groups {
                let (kv_b, scale) = (&kv_b, &scale);
                scope.spawn(move || {
                    for (h, (((uk, uk_s), uv), uv_s)) in group {
                        let base = h * (d + v);
                        for c0 in (0..c).step_by(64) {
                            for d0 in (0..d).step_by(64) {
                                for dd in d0..d0 + 64 {
                                    let src = &kv_b[(base + dd) * c..][..c];
                                    for col in c0..c0 + 64 {
                                        uk[col * d + dd] = src[col];
                                    }
                                }
                            }
                        }
                        for col in 0..c {
                            for t in 0..d / 64 {
                                uk_s[col * (d / 64) + t] = scale(base + t * 64, col);
                            }
                        }
                        uv.copy_from_slice(&kv_b[(base + d) * c..(base + d + v) * c]);
                        for r in 0..v {
                            for t in 0..c / 64 {
                                uv_s[r * (c / 64) + t] = scale(base + d + r, t * 64);
                            }
                        }
                    }
                });
            }
        });
        ensure!(heads % ranks == 0, "{prefix}: {heads} heads do not split over {ranks} GPUs");
        let part = |bytes: &[u8], rank: usize| -> Vec<u8> {
            bytes[rank * bytes.len() / ranks..(rank + 1) * bytes.len() / ranks].to_vec()
        };
        let (uk_s, uv_s) = (f32_bytes(&uk_s), f32_bytes(&uv_s));
        (0..ranks).map(|rank| self.on_rank(rank, |_| Ok((
            (self.upload("w_uk_fp8", &part(&uk, rank))?, self.upload("w_uk_scale", &part(&uk_s, rank))?),
            (self.upload("w_uv_fp8", &part(&uv, rank))?, self.upload("w_uv_scale", &part(&uv_s, rank))?))))).collect()
        })
    }

    /// Whether `names` run as the checkpoint's BF16 tensors (the BF16 programs): every one is
    /// BF16 and CUTEAFD_GLM_BF16=native asks for them ([`bf16_programs`]).
    fn bf16_native(&self, names: &[String]) -> Result<bool> {
        if !bf16_programs() {
            return Ok(false);
        }
        for name in names {
            if self.catalog.tensor(name)?.metadata.dtype != DType::Bf16 {
                return Ok(false);
            }
        }
        Ok(true)
    }

    /// The row-concatenation of BF16 `names`: its bytes, rows and columns.
    fn bf16_rows(&self, names: &[String]) -> Result<(Vec<u8>, usize, usize)> {
        let mut out = Vec::new();
        let (mut rows, mut cols) = (0usize, None);
        for name in names {
            let (bytes, dtype, shape) = self.raw(name)?;
            ensure!(dtype == DType::Bf16 && shape.len() == 2, "{name}: expected a BF16 matrix, found {dtype:?} {shape:?}");
            ensure!(cols.is_none_or(|c| c == shape[1]), "{names:?} do not share columns");
            cols = Some(shape[1]);
            rows += shape[0];
            out.extend_from_slice(&bytes);
        }
        Ok((out, rows, cols.context("no BF16 parts")?))
    }

    /// The BF16 weights `names` concatenated by rows, sliced over `ranks` along `axis`:
    /// rank `r` takes part `r` of every concatenated weight's rows (so a head- or
    /// intermediate-split `gate | up` stays gate rows then up rows), or of the columns.
    fn bf16_split(&self, key: &str, names: &[String], axis: Axis, ranks: usize) -> Result<Vec<DeviceAllocation<'a>>> {
        let _memory_format = cuteafd_ffi::memory_ledger::format("bf16");
        if ranks == 1 {
            return Ok(vec![self.upload(key, &self.bf16_rows(names)?.0)?]);
        }
        let parts = names.iter().map(|n| self.bf16_rows(std::slice::from_ref(n))).collect::<Result<Vec<_>>>()?;
        ensure!(axis == Axis::Rows || parts.len() == 1, "{names:?}: concatenated weights split by rows");
        (0..ranks).map(|rank| {
            let mut bytes = Vec::new();
            for (values, rows, cols) in &parts {
                ensure!(if axis == Axis::Rows { rows % ranks == 0 } else { cols % ranks == 0 },
                    "{names:?}: [{rows}, {cols}] does not split over {ranks} GPUs");
                bytes.extend(slice_2d(values, *rows, *cols, 2, axis, rank, ranks));
            }
            self.on_rank(rank, |_| self.upload(key, &bytes))
        }).collect()
    }

    /// `w_uk [N, 512, D]` / `w_uv [N, V, 512]` BF16 per-head slices of the BF16 `kv_b`
    /// (`w_uk[h,c,d] = kv_b[h*(D+V)+d, c]`, `w_uv[h,v,c] = kv_b[h*(D+V)+D+v, c]`), each rank
    /// its heads.
    /// `kv_b` split into per-head `w_uk [H, latent, nope]` (transposed) and
    /// `w_uv [H, v, latent]`; `nope` is the layer's (NoPE layers widen it).
    fn kv_b_bf16(&self, cfg: &GlmDsaConfig, layer: usize, ranks: usize)
        -> Result<Vec<(DeviceAllocation<'a>, DeviceAllocation<'a>)>> {
        let _memory_format = cuteafd_ffi::memory_ledger::format("bf16");
        let (heads, d, v, c) = (cfg.heads, cfg.nope_dim(layer), cfg.v_head_dim, cfg.kv_lora_rank);
        let prefix = cfg.stem(layer, Op::KvB).context("kv_b")?;
        let (kv_b, rows, cols) = self.bf16_rows(&[format!("{prefix}.weight")])?;
        ensure!(rows == heads * (d + v) && cols == c && heads % ranks == 0,
            "{prefix}: kv_b_proj [{rows}, {cols}] does not match {heads} heads x ({d} + {v}) x {c} over {ranks} GPUs");
        let (mut uk, mut uv) = (vec![0u8; heads * c * d * 2], vec![0u8; heads * v * c * 2]);
        std::thread::scope(|scope| {
            for (h, (uk, uv)) in uk.chunks_mut(c * d * 2).zip(uv.chunks_mut(v * c * 2)).enumerate() {
                let kv_b = &kv_b;
                scope.spawn(move || {
                    let base = h * (d + v);
                    for dd in 0..d {
                        let src = &kv_b[(base + dd) * c * 2..][..c * 2];
                        for col in 0..c {
                            uk[(col * d + dd) * 2..][..2].copy_from_slice(&src[col * 2..col * 2 + 2]);
                        }
                    }
                    uv.copy_from_slice(&kv_b[(base + d) * c * 2..(base + d + v) * c * 2]);
                });
            }
        });
        let part = |bytes: &[u8], rank: usize| bytes[rank * bytes.len() / ranks..(rank + 1) * bytes.len() / ranks].to_vec();
        (0..ranks).map(|rank| self.on_rank(rank, |_| Ok((self.upload("w_uk", &part(&uk, rank))?, self.upload("w_uv", &part(&uv, rank))?))))
            .collect()
    }

    fn one(&self, key: &str, name: &str) -> Result<DeviceAllocation<'a>> {
        let _memory_format = cuteafd_ffi::memory_ledger::format("bf16");
        let meta = &self.catalog.tensor(name)?.metadata;
        if meta.shape.len() == 2 && meta.dtype == DType::F8E4M3 {
            return self.rows(key, &[name.to_string()]);
        }
        let land = Land::whole(self.catalog, name, 0)?;
        let bytes = land.bytes();
        self.landed(key, &[land], bytes)
    }

    /// GPUs of the head split this loader fills (1: no split).
    pub fn ranks(&self) -> usize {
        1 + self.peers.len()
    }

    /// Runs `body` with rank `rank`'s device current and its load stream.
    fn on_rank<T>(&self, rank: usize, body: impl FnOnce(*mut c_void) -> Result<T>) -> Result<T> {
        if rank == 0 {
            return body(self.stream);
        }
        let peer = self.peers.get(rank - 1).with_context(|| format!("no rank {rank}"))?;
        crate::shared::peer_split::on_device(self.library, peer.device, || body(peer.stream))
    }

    /// The FP8 weights `names` concatenated by rows, each sliced over `ranks`
    /// along `axis` in whole 128-row / 128-K blocks: per rank its E4M3 bytes and
    /// the matching part of the FP32 128x128 grid. Rows-sliced weights read once.
    fn fp8_split(&self, key: &str, names: &[String], axis: Axis, ranks: usize)
        -> Result<Vec<(DeviceAllocation<'a>, DeviceAllocation<'a>)>> {
        let _memory_format = cuteafd_ffi::memory_ledger::format("fp8");
        if ranks == 1 {
            return Ok(vec![self.fp8(key, names)?]);
        }
        if let ([name], Axis::Cols) = (names, axis) {
            return self.fp8_cols(key, name, ranks);
        }
        ensure!(axis == Axis::Rows, "{names:?}: concatenated weights split by rows");
        if self.native_fp8(names)? {
            let grids = names.iter().map(|n| landing::block_grid(self.catalog, n)).collect::<Vec<_>>();
            return (0..ranks).map(|rank| {
                let (values, values_bytes) = landing::row_shares(self.catalog, names, ranks, rank, 128)?;
                let (grid, grid_bytes) = landing::row_shares(self.catalog, &grids, ranks, rank, 1)?;
                self.on_rank(rank, |_| Ok((self.landed(fp8_operand_names(key).0, &values, values_bytes)?, self.landed(fp8_operand_names(key).1, &grid, grid_bytes)?)))
            }).collect();
        }
        // Per rank: the E4M3 rows and grid rows of every part, uploaded straight from the
        // staging buffer into the part's offset (whole 128-row blocks, so grids slice too).
        let mut out: Vec<Option<(DeviceAllocation<'a>, DeviceAllocation<'a>)>> = (0..ranks).map(|_| None).collect();
        let (mut row_at, mut grid_at) = (0usize, 0usize);
        let (total_rows, cols) = names.iter().try_fold((0usize, 0usize), |(rows, _), name| -> Result<_> {
            let shape = &self.catalog.tensor(name)?.metadata.shape;
            Ok((rows + shape[0], shape[1]))
        })?;
        for name in names {
            self.with_fp8(std::slice::from_ref(name), |values, grid, rows, part_cols| {
                ensure!(part_cols == cols && rows % (128 * ranks) == 0 && cols % 128 == 0,
                    "{name}: [{rows}, {cols}] does not split into whole 128-row blocks over {ranks} GPUs");
                let (share, kb) = (rows / ranks, cols / 128);
                for (rank, slot) in out.iter_mut().enumerate() {
                    self.on_rank(rank, |_| {
                        if slot.is_none() {
                            *slot = Some((self.allocate(fp8_operand_names(key).0, total_rows / ranks * cols)?,
                                self.allocate(fp8_operand_names(key).1, total_rows / ranks / 128 * kb * 4)?));
                        }
                        let (dv, dg) = slot.as_ref().context("rank buffers")?;
                        let at = |buffer: CuteafdDeviceBuffer, offset: usize, bytes: usize| CuteafdDeviceBuffer {
                            // SAFETY: offset + bytes lie inside the rank's buffer (sized above).
                            ptr: unsafe { buffer.ptr.cast::<u8>().add(offset) }.cast(), bytes, ..buffer };
                        crate::shared::memory::load_timing::upload(share * cols, || self.library.copy_h2d(at(dv.buffer, row_at / ranks * cols, share * cols),
                            &values[rank * share * cols..(rank + 1) * share * cols]))?;
                        let grid_part = f32_bytes(&grid[rank * share / 128 * kb..(rank + 1) * share / 128 * kb]);
                        crate::shared::memory::load_timing::upload(grid_part.len(), || self.library.copy_h2d(at(dg.buffer, grid_at / ranks * 4, grid_part.len()), &grid_part))
                    })?;
                }
                row_at += rows;
                grid_at += rows / 128 * kb;
                Ok(())
            })?;
        }
        out.into_iter().map(|slot| slot.context("no FP8 parts")).collect()
    }

    /// One FP8 weight sliced by columns over `ranks`: the whole E4M3 weight goes up
    /// once to rank 0 and each rank's columns are a pitched device copy from it
    /// (over peer memory for rank 1); the grid's column blocks are sliced on the host.
    fn fp8_cols(&self, key: &str, name: &str, ranks: usize) -> Result<Vec<(DeviceAllocation<'a>, DeviceAllocation<'a>)>> {
        let _memory_format = cuteafd_ffi::memory_ledger::format("fp8");
        if self.native_fp8(&[name.to_string()])? {
            let shape = &self.catalog.tensor(name)?.metadata.shape;
            ensure!(shape[0] % 128 == 0 && shape[1] % (128 * ranks) == 0,
                "{name}: {shape:?} does not split into whole 128-K blocks over {ranks} GPUs");
            return (0..ranks).map(|rank| {
                let (values, values_bytes) = landing::col_share(self.catalog, name, ranks, rank)?;
                let (grid, grid_bytes) = landing::col_share(self.catalog, &landing::block_grid(self.catalog, name), ranks, rank)?;
                self.on_rank(rank, |_| Ok((self.landed(fp8_operand_names(key).0, &[values], values_bytes)?, self.landed(fp8_operand_names(key).1, &[grid], grid_bytes)?)))
            }).collect();
        }
        let (whole, grid, rows, cols) = self.with_fp8(&[name.to_string()], |values, grid, rows, cols| {
            ensure!(rows % 128 == 0 && cols % (128 * ranks) == 0,
                "{name}: [{rows}, {cols}] does not split into whole 128-K blocks over {ranks} GPUs");
            Ok((GlmLoader { arenas: None, ..self.clone() }.upload("transient.values", values)?, grid, rows, cols))
        })?;
        let width = cols / ranks;
        let grid_bytes = f32_bytes(&grid);
        let out = (0..ranks).map(|rank| {
            let part = slice_2d(&grid_bytes, rows / 128, cols / 128, 4, Axis::Cols, rank, ranks);
            self.on_rank(rank, |stream| {
                let values = self.allocate(fp8_operand_names(key).0, rows * width)?;
                let source = CuteafdDeviceBuffer {
                    // SAFETY: column block `rank` of every row lies inside the whole weight.
                    ptr: unsafe { whole.buffer.ptr.cast::<u8>().add(rank * width) }.cast(),
                    bytes: whole.buffer.bytes - rank * width,
                    ..whole.buffer
                };
                // SAFETY: both buffers are live and sized for these pitched spans; peer access
                // to rank 0 is enabled on every rank (the split setup); drained below, before
                // `whole` drops.
                unsafe {
                    self.library.copy_d2d_2d_async(values.buffer, width, source, cols, width, rows, stream)?;
                    self.library.cuda_stream_synchronize(stream)?;
                }
                Ok((values, self.upload(fp8_operand_names(key).1, &part)?))
            })
        }).collect();
        drop(whole);
        out
    }

    /// Every rank's copy of a small operand read once.
    /// A whole non-FP8 tensor (a norm) on every rank, landing unchanged.
    fn replicated_tensor(&self, key: &str, ranks: usize, name: &str) -> Result<Vec<DeviceAllocation<'a>>> {
        let meta = &self.catalog.tensor(name)?.metadata;
        ensure!(!(meta.shape.len() == 2 && meta.dtype == DType::F8E4M3), "{name}: replicated operands are not FP8");
        let land = Land::whole(self.catalog, name, 0)?;
        let bytes = land.bytes();
        self.landed_all(key, ranks, &[land], bytes)
    }

    /// A per-head BF16 vector `[heads]`, each rank its contiguous `heads / ranks` (the query
    /// heads a head split gives it); one rank keeps the whole vector.
    fn head_vector(&self, key: &str, ranks: usize, name: &str) -> Result<Vec<DeviceAllocation<'a>>> {
        let _memory_format = cuteafd_ffi::memory_ledger::format("bf16");
        let meta = &self.catalog.tensor(name)?.metadata;
        ensure!(ranks > 0 && meta.dtype == DType::Bf16 && meta.shape.len() == 1 && meta.shape[0] % ranks == 0,
            "{name}: expected a BF16 [heads] vector over {ranks} GPUs, found {:?} {:?}", meta.dtype, meta.shape);
        (0..ranks).map(|rank| {
            let (land, bytes) = landing::col_share(self.catalog, name, ranks, rank)?;
            self.on_rank(rank, |_| self.landed(key, &[land], bytes))
        }).collect()
    }

    /// `bytes` on every rank: one upload to rank 0, peer copies to the others (faster than
    /// a second pageable host upload).
    fn upload_all(&self, key: &str, ranks: usize, bytes: &[u8]) -> Result<Vec<DeviceAllocation<'a>>> {
        let first = self.upload(key, bytes)?;
        let mut out = Vec::with_capacity(ranks);
        for rank in 1..ranks {
            out.push(self.on_rank(rank, |stream| {
                let copy = self.allocate(key, bytes.len().max(256))?;
                // SAFETY: both allocations hold `bytes.len()` bytes; peer access to rank 0 is
                // enabled (the split setup); the stream drains before return.
                unsafe {
                    self.library.copy_d2d_async(copy.buffer, first.buffer, bytes.len(), stream)?;
                    self.library.cuda_stream_synchronize(stream)?;
                }
                Ok(copy)
            })?);
        }
        out.insert(0, first);
        Ok(out)
    }

    fn core_weights(&self, layout: cuteafd_loader::formats::trellis_core::CoreWeightLayout,
        checkpoint: &cuteafd_loader::plan::Checkpoint, rank: usize, prefix: &str) -> Result<CoreWeights<'a>> {
        use crate::shared::experts::rtx::dense::DenseLoad;
        let layout = CoreWeightLayout::from_shared(layout);
        let dense = if self.push.is_some() { DenseLoad::push(self.library) }
            else { DenseLoad::mount(self.library) };
        let weights = self.on_rank(rank, |stream| {
            let device = Device { library: self.library, id: self.library.cuda_get_device()? };
            let arena = self.arenas.as_ref().map(|arenas| arenas.get(&device.id)
                .context("K6 current device has no admitted GLM target arena")).transpose()?;
            layout.load_recorded_with(device, checkpoint, &dense, stream, |name, bytes| {
                let suffix = name.rsplit('.').next().context("K6 member suffix")?;
                let suffix = if suffix == "trellis" { "payload" } else { suffix };
                crate::shared::memory::weight_arena::weight_allocation(device, arena,
                    &format!("glm.weights.gpu.{rank}.{prefix}.{suffix}"), bytes, ArenaTag::weights("glm/target"))
            })
        })?;
        if let Some(push) = &self.push {
            let (operands, steps) = dense.sink.as_ref().context("K6 push sink")?.take();
            for operand in operands { push.dense.operand(operand)?; }
            for step in steps { push.dense.step(step)?; }
        }
        Ok(weights)
    }

    fn packed_fp8(&self, qkv_a: &[String]) -> Result<Vec<(DeviceAllocation<'a>, DeviceAllocation<'a>, DeviceAllocation<'a>)>> {
        let ranks = self.ranks();
        let (values, scale, kscale) = if let Some(((values, values_bytes), (grid, grid_bytes))) = self.fp8_lands(qkv_a)? {
            // Prefill: per-row scales K-block major, widened on each GPU from its landed grid.
            let rows: usize = values.iter().map(|land| land.rows.len()).sum();
            let kb = values[0].width().div_ceil(128);
            let (values, grid) = (self.landed_all("w_qkv_a_fp8", ranks, &values, values_bytes)?, self.landed_all("w_qkv_a_scale", ranks, &grid, grid_bytes)?);
            let kscale = grid.iter().enumerate().map(|(rank, grid)| self.on_rank(rank, |stream| {
                let kscale = self.allocate("w_qkv_a_kscale", (rows * kb * 4).max(256))?;
                let job = landing::Kmajor { device: self.library.cuda_get_device()?, grid: grid.buffer,
                    kmajor: kscale.buffer, rows, kb };
                match &self.push {
                    // The grid is not delivered yet: widen after the group commits.
                    Some(push) => push.defer(job)?,
                    None => job.run(self.library, stream)?,
                }
                Ok(kscale)
            })).collect::<Result<Vec<_>>>()?;
            (values, grid, kscale)
        } else { self.with_fp8(qkv_a, |values, grid, rows, cols| {
            // Prefill: per-row scales K-block major (the block-FP8 GEMM wants whole 128-row blocks).
            let kb = cols.div_ceil(128);
            let grid_ref = &grid;
            let kscale: Vec<f32> = (0..kb).flat_map(|b| (0..rows).map(move |r| grid_ref[(r / 128) * kb + b])).collect();
            Ok((self.upload_all("w_qkv_a_fp8", ranks, values)?, self.upload_all("w_qkv_a_scale", ranks, &f32_bytes(&grid))?, self.upload_all("w_qkv_a_kscale", ranks, &f32_bytes(&kscale))?))
        })? };
        
        
        


        Ok(values.into_iter().zip(scale).zip(kscale).map(|((v,s),k)| (v,s,k)).collect())
    }

    fn layer(&self, cfg: &GlmDsaConfig, layer: usize,
        checkpoint: &cuteafd_loader::plan::Checkpoint, core: &CoreInventory, owner: Option<u8>,
        native_shared: bool) -> Result<Vec<GlmLayer<'a>>> {
        use routes::{Operation, layer_routes};
        let scoped = self.scoped(format!("layer.{layer}.operand."));
        let ranks = self.ranks();
        let dense = layer < cfg.first_moe_layer;
        let gate = cfg.stem(layer, Op::Gate).context("FFN gate")?;
        let mlp_numerics = if native_shared && !dense {
            cuteafd_loader::plan::checkpoint::ModuleNumerics::CheckpointPrecision
        } else { checkpoint.module_numerics(&gate)? };
        let mut layers = (0..ranks).map(|rank| GlmLayer {
            dense, full_indexer: cfg.indexers[layer] == GlmIndexer::Full, mlp_numerics,
            split: ranks > 1, attention_split: ranks > 1 && owner.is_none(), attention_owner: owner,
            attention_resident: owner.is_none_or(|owner| usize::from(owner) == rank),
            operands: HashMap::new(), native: HashMap::new(), native_projections: HashMap::new(), core: Default::default(),
            attention_native: Default::default(),
        }).collect::<Vec<_>>();
        layer_routes(self.catalog, cfg, layer, checkpoint, core, ranks, owner, native_shared, bf16_programs(), |route| {
            let mut loaded = Vec::<(usize, &'static str, DeviceAllocation<'a>)>::new();
            if let Operation::Core(layout) = route.operation {
                let rank = route.ranks[0];
                let weights = self.core_weights(layout, checkpoint, rank, &format!("layer.{layer}.core.{}", route.key))?;
                let actual = [("payload", weights.payload.buffer), ("suh", weights.suh.buffer), ("svh", weights.svh.buffer)]
                    .into_iter().map(|(suffix, buffer)| (format!("glm.weights.gpu.{rank}.layer.{layer}.core.{}.{suffix}", route.key), buffer)).collect::<Vec<_>>();
                self.check_weight_requests(&route.requests, &actual)?;
                layers[rank].core.insert(route.key, weights);
                return Ok(());
            }
            if let Operation::NativeAttention(layout) = route.operation {
                use crate::shared::native_attention::{attention_records, AttentionRecord, NativeAttentionWeights};
                let rank = route.ranks[0];
                let device = crate::shared::memory::device::Device { library: self.library,
                    id: if rank == 0 { self.library.cuda_get_device()? } else { self.peers[rank - 1].device } };
                let weights = device.own(|| self.on_rank(rank, |stream| {
                    let [n, k] = layout.shape;
                    let native = self.scoped(format!("layer.{layer}.native.{}.", route.key));
                    let values = native.allocate("values", (n * k * if layout.fp8 { 1 } else { 2 }).max(256))?;
                    let scales = layout.fp8.then(|| native.allocate("scales",
                        (n.div_ceil(128) * k.div_ceil(128) * 4).max(256))).transpose()?;
                    let selected = checkpoint.projection(&layout.stem)?;
                    let cuteafd_loader::plan::checkpoint::ProjectionStorage::Native { shape, .. } = &selected.storage
                        else { anyhow::bail!("native attention load selected compressed source"); };
                    let receipt = cuteafd_loader::placement::families::glm5::weights::native_attention_mount(
                        &layout, [shape[0], shape[1]],
                        checkpoint.converts_to_fp8(&format!("{}.weight", layout.stem)))?;
                    let records = attention_records(self.catalog.snapshot(), &selected,
                        checkpoint.projection_view(&layout.stem), &layout, cuteafd_loader::formats::native_attention::Fp8Role::exact("projection", true), crate::shared::fp8_linear::Fp8Scales::Amax,
                        values.buffer, scales.as_ref().map(|s| s.buffer))?;
                    let mount = crate::shared::experts::rtx::dense::DenseLoad::mount(self.library);
                    mount.admit_mount(receipt)?;
                    for record in records {
                        match (&self.push, record) {
                            (Some(push), AttentionRecord::Operand(record)) => push.dense.operand(record)?,
                            (Some(push), AttentionRecord::Step(record)) => push.dense.step(record)?,
                            (None, AttentionRecord::Operand(record)) => mount.operand(record)?,
                            (None, AttentionRecord::Step(record)) => mount.step(record, stream)?,
                        }
                    }
                    // SAFETY: this rank owns the load stream and every destination
                    // remains live until its queued load/transform work completes.
                    if let Err(drain) = unsafe { self.library.cuda_stream_synchronize(stream) } {
                        self.library.quarantine_module_after_failed_drain();
                        return Err(drain.context("GLM native attention owners quarantined after failed load drain"));
                    }
                    Ok(NativeAttentionWeights { layout, values, scales })
                }))?;
                let mut actual = vec![(format!("glm.weights.gpu.{rank}.layer.{layer}.native.{}.values", route.key), weights.values.buffer)];
                if let Some(scales) = &weights.scales {
                    actual.push((format!("glm.weights.gpu.{rank}.layer.{layer}.native.{}.scales", route.key), scales.buffer));
                }
                self.check_weight_requests(&route.requests, &actual)?;
                layers[rank].attention_native.insert(route.key, weights);
                return Ok(());
            }
            let first = route.ranks[0];
            let device = if first == 0 { self.device } else { self.peers[first-1].device };
            let stream = if first == 0 { self.stream } else { self.peers[first-1].stream };
            let local = GlmLoader { device, stream, peers: Vec::new(), ..scoped.clone() };
            let loader = if route.ranks.len() == ranks { &scoped } else { &local };
            let parts = route.ranks.len();
            self.on_rank(first, |_| {
                let mut put = |key, buffers: Vec<DeviceAllocation<'a>>| -> Result<()> {
                    ensure!(buffers.len() == parts, "weight operation returned wrong rank count");
                    loaded.extend(route.ranks.iter().copied().zip(buffers).map(|(rank, buffer)| (rank, key, buffer)));
                    Ok(())
                };
                match route.operation {
                    Operation::Tensor(name) => put(route.key, if parts == 1 { vec![loader.one(route.key, &name)?] }
                        else { loader.replicated_tensor(route.key, parts, &name)? })?,
                    Operation::Rows(names) => {
                        let first = loader.rows(route.key, &names)?;
                        let mut buffers = Vec::with_capacity(parts);
                        for rank in 1..parts {
                            buffers.push(loader.on_rank(rank, |stream| {
                                let buffer = loader.allocate(route.key, first.buffer.bytes)?;
                                // SAFETY: both complete operands are live; the copy drains before publication.
                                unsafe {
                                    self.library.copy_d2d_async(buffer.buffer, first.buffer, first.buffer.bytes, stream)?;
                                    self.library.cuda_stream_synchronize(stream)?;
                                }
                                Ok(buffer)
                            })?);
                        }
                        put(route.key, std::iter::once(first).chain(buffers).collect())?;
                    }
                    Operation::Projection { names, axis, bf16, kmajor, replicated } => {
                        if kmajor {
                            let (mut values, mut grids, mut ks) = (Vec::new(), Vec::new(), Vec::new());
                            for (v, s, k) in loader.packed_fp8(&names)? { values.push(v); grids.push(s); ks.push(k); }
                            put("w_qkv_a_fp8", values)?; put("w_qkv_a_scale", grids)?; put("w_qkv_a_kscale", ks)?;
                        } else if bf16 {
                            let buffers = if replicated { loader.upload_all(route.key, parts, &loader.bf16_rows(&names)?.0)? }
                                else { loader.bf16_split(route.key, &names, axis, parts)? };
                            put(route.key, buffers)?;
                        } else {
                            let buffers = if replicated && parts > 1 {
                                if let Some(((values, values_bytes), (grid, grid_bytes))) = loader.fp8_lands(&names)? {
                                    loader.landed_all(fp8_operand_names(route.key).0, parts, &values, values_bytes)?.into_iter()
                                        .zip(loader.landed_all(fp8_operand_names(route.key).1, parts, &grid, grid_bytes)?).collect()
                                } else { loader.with_fp8(&names, |values, grid, _, _| {
                                    Ok(loader.upload_all(fp8_operand_names(route.key).0, parts, values)?.into_iter().zip(loader.upload_all(fp8_operand_names(route.key).1, parts, &f32_bytes(&grid))?).collect())
                                })? }
                            } else { loader.fp8_split(route.key, &names, axis, parts)? };
                            let (values, grids): (Vec<_>, Vec<_>) = buffers.into_iter().unzip();
                            let (value_key, grid_key, _) = fp8_operand_names(route.key);
                            put(value_key, values)?; put(grid_key, grids)?;
                        }
                    }
                    Operation::KvB { layer, bf16 } => {
                        if bf16 {
                            let (uk, uv): (Vec<_>, Vec<_>) = loader.kv_b_bf16(cfg, layer, parts)?.into_iter().unzip();
                            put("w_uk", uk)?; put("w_uv", uv)?;
                        } else {
                            let (mut uk, mut uks, mut uv, mut uvs) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
                            for ((k, ks), (v, vs)) in loader.kv_b(cfg, &format!("model.layers.{layer}"), parts)? {
                                uk.push(k); uks.push(ks); uv.push(v); uvs.push(vs);
                            }
                            put("w_uk_fp8", uk)?; put("w_uk_scale", uks)?; put("w_uv_fp8", uv)?; put("w_uv_scale", uvs)?;
                        }
                    }
                    Operation::HeadVector(name) => put(route.key, loader.head_vector(route.key, parts, &name)?)?,
                    Operation::TensorScales(names) => put(route.key, loader.upload_all(route.key, parts,
                        &loader.tensor_scales(&names)?.context("declared tensor scales missing")?)?)?,
                    Operation::Core(_) | Operation::NativeAttention(_) => unreachable!(),
                }
                Ok(())
            })?;
            let actual = loaded.iter().map(|(rank, key, buffer)|
                (format!("glm.weights.gpu.{rank}.layer.{layer}.operand.{key}"), buffer.buffer)).collect::<Vec<_>>();
            self.check_weight_requests(&route.requests, &actual)?;
            for (rank, key, buffer) in loaded {
                ensure!(layers[rank].operands.insert(key, buffer).is_none(), "duplicate weight operand {key}");
            }
            Ok(())
        })?;
        if mlp_numerics == cuteafd_loader::plan::checkpoint::ModuleNumerics::W8A8 {
            ensure!(layers.iter().all(|layer| layer.operands.contains_key("w_gate_up_tscale")
                && layer.operands.contains_key("w_down_tscale")), "declared W8A8 needs native activation scales");
        }
        Ok(layers)
    }

    fn check_weight_requests(&self, requests: &[cuteafd_loader::placement::planner::Allocation],
        actual: &[(String, CuteafdDeviceBuffer)]) -> Result<()> {
        ensure!(requests.len() == actual.len(), "GLM weight request/owner count differs");
        for request in requests {
            let buffer = actual.iter().find(|(id, _)| id == &request.id)
                .with_context(|| format!("missing loaded owner {}", request.id))?.1;
            let cuteafd_loader::placement::planner::Device::Coordinator(rank) = request.location.device else {
                anyhow::bail!("weight request is not coordinator-owned");
            };
            let device = if rank == 0 { self.device } else { self.peers[usize::from(rank)-1].device };
            ensure!(!buffer.ptr.is_null() && buffer.bytes as u64 == request.bytes && buffer.device_id == device,
                "{}: actual weight extent/device {}/{} differs from request {}/{}", request.id,
                buffer.bytes, buffer.device_id, request.bytes, device);
        }
        Ok(())
    }

    /// Layers `0..layers` (all of them unless the caller stops early): rank 0's
    /// weights (its layer shares, the norm and head) and with a head split the
    /// other rank's layer shares.
    #[allow(clippy::type_complexity)]
    pub fn model(&self, cfg: &GlmDsaConfig, layers: usize, owners: Option<&[u8]>, native_shared: bool)
        -> Result<(GlmWeights<'a>, Vec<Vec<GlmLayer<'a>>>)> {
        ensure!(self.arenas.is_some(), "GLM target model requires admitted coordinator arenas");
        ensure!(owners.is_none_or(|o| o.len() == layers && self.ranks() == 2 && o.iter().all(|&r| r < 2)),
            "whole attention needs one owner per layer and two ranks");
        let split = crate::shared::memory::load_timing::Split::start();
        let mut checkpoint = if let Some(view) = self.catalog.selected_view() {
            cuteafd_loader::plan::Checkpoint::coordinator_view(view, false, false)?
        } else { cuteafd_loader::plan::checkpoint::Checkpoint::inventory(self.catalog.snapshot())? };
        let core = CoreInventory::from_checkpoint(cfg, &checkpoint)?;
        // Inventory is header-light. Reuse only the native core headers already
        // opened by the catalog; recorded recipes need their exact rectangles.
        for projection in core.projections.values() {
            for suffix in ["trellis", "suh", "svh", projection.codebook.marker()] {
                let name = format!("{}.{suffix}", projection.stem);
                if !checkpoint.tensors.iter().any(|tensor| tensor.meta.name == name) {
                    let tensor = self.catalog.tensor(&name)?;
                    checkpoint.tensors.push(cuteafd_loader::plan::checkpoint::CheckpointTensor {
                        shard: tensor.shard.clone(), meta: tensor.metadata.clone() });
                }
            }
        }
        if !core.native.is_empty() {
            let view = self.catalog.selected_view().context("GLM native attention selected view")?;
            let native = cuteafd_loader::plan::Checkpoint::open_view_for_role(view,
                "GLM native attention load records", |name| name.rsplit_once('.')
                    .is_some_and(|(stem, _)| core.native.contains_key(stem)))?;
            for tensor in native.tensors {
                if !checkpoint.tensors.iter().any(|current| current.meta.name == tensor.meta.name) {
                    checkpoint.tensors.push(tensor);
                }
            }
        }
        checkpoint.tensors.sort_by(|a, b| a.meta.name.cmp(&b.meta.name));
        let mut shares: Vec<Vec<GlmLayer<'a>>> = (0..self.ranks()).map(|_| Vec::new()).collect();
        for layer in 0..layers.min(cfg.layers) {
            for (share, part) in shares.iter_mut().zip(self.layer(cfg, layer, &checkpoint, &core, owners.map(|o| o[layer]), native_shared)?) {
                share.push(part);
            }
        }
        let (mut norm, mut head) = (None, None);
        routes::global_routes(self.catalog, cfg, &core, |route| {
            let actual = match route.operation {
                routes::Operation::Tensor(name) => {
                    let allocation = self.one(route.key, &name)?;
                    let actual = vec![(format!("glm.weights.gpu.0.{}", route.key), allocation.buffer)];
                    if route.key == "norm" { norm = Some(allocation); }
                    else { head = Some(GlmHead::Bf16(allocation)); }
                    actual
                }
                routes::Operation::Core(layout) => {
                    let weights = self.core_weights(layout, &checkpoint, 0, "core.head")?;
                    let actual = [("payload", weights.payload.buffer), ("suh", weights.suh.buffer), ("svh", weights.svh.buffer)]
                        .into_iter().map(|(suffix, buffer)| (format!("glm.weights.gpu.0.core.head.{suffix}"), buffer)).collect();
                    head = Some(GlmHead::K6(weights));
                    actual
                }
                _ => anyhow::bail!("invalid global weight operation"),
            };
            self.check_weight_requests(&route.requests, &actual)
        })?;
        let mut shares = shares.into_iter();
        let weights = GlmWeights {
            layers: shares.next().context("rank 0")?,
            norm: norm.context("global norm route")?,
            head: head.context("global head route")?,
            rope_alpha: cfg.variant.ml4().map(|_| self.rope_alpha(cfg)).transpose()?,
        };
        crate::shared::memory::staging::release_staging();
        split.log("glm5 dense");
        Ok((weights, shares.collect()))
    }

    fn native_layer_owner(cfg: &GlmDsaConfig, layer: usize, name: &str) -> Result<String> {
        let prefix = format!("layers.{layer}.");
        let name = name.strip_prefix(&prefix).context("native GLM allocation layer differs")?;
        let key = if name.starts_with("w_") {
            format!("operand.{name}")
        } else if name == cuteafd_loader::plan::exlr8::dense::requests::key("router", "bf16") {
            "operand.gate".into()
        } else if let Some((key, _)) = cfg.plain_owners(layer).into_iter().find(|(_, tensor)| tensor == name) {
            format!("operand.{key}")
        } else {
            ensure!(name.starts_with("n."), "unknown native GLM allocation {name}");
            format!("native.{name}")
        };
        Ok(format!("layer.{layer}.{key}"))
    }

    /// An EXLR8 checkpoint's own dense operands (no official snapshot):
    /// `records[rank]` for each head-split rank, filled through `load` (mount
    /// now, or recorded for the push group), the final norm on rank 0 and the
    /// row-FP8 `vocabulary` owner (shared with the token embedding) as the head.
    /// Routed experts and the shared expert stay on their native owners.
    #[allow(clippy::type_complexity)]
    pub fn native_model(&self, cfg: &GlmDsaConfig, layers: usize,
        records: &[cuteafd_loader::plan::exlr8::dense::requests::NativeDenseRecords], owners: Option<&[u8]>, load: &crate::shared::experts::rtx::dense::DenseLoad<'a>,
        vocabulary: std::rc::Rc<crate::shared::token_io::native::NativeVocabulary<'a>>)
        -> Result<(GlmWeights<'a>, Vec<Vec<GlmLayer<'a>>>)> {
        ensure!(self.arenas.is_some(), "native GLM target model requires admitted coordinator arenas");
        let ranks = self.ranks();
        ensure!(records.len() == ranks && cfg.heads % ranks == 0, "native GLM records per coordinator rank");
        ensure!(owners.is_none_or(|owners| ranks == 2 && owners.len() == layers
            && owners.iter().all(|&owner| usize::from(owner) < ranks)), "native GLM attention owners");
        let split = crate::shared::memory::load_timing::Split::start();
        let mut shares: Vec<Vec<GlmLayer<'a>>> = (0..ranks).map(|_| Vec::new()).collect();
        for layer in 0..layers.min(cfg.layers) {
            for (rank, share) in shares.iter_mut().enumerate() {
                let layer_records = records[rank].layers.get(layer).context("native layer records")?;
                let owner = owners.map(|owners| owners[layer]);
                let attention_resident = owner.is_none_or(|owner| usize::from(owner) == rank);
                ensure!(layer_records.kv_b.is_some() == attention_resident, "native GLM attention record ownership");
                let part = self.on_rank(rank, |stream| -> Result<GlmLayer<'a>> {
                    let _memory_format = cuteafd_ffi::memory_ledger::format("native");
                    let mut native = crate::shared::experts::exlr8::dense_weights::load_layer_packed_with(
                        self.library, load, layer_records, &[], stream, |name, bytes| {
                            self.allocate_owner(&Self::native_layer_owner(cfg, layer, name)?, bytes)
                        })?;
                    let p = format!("model.layers.{layer}");
                    let mut operands: HashMap<&'static str, DeviceAllocation<'a>> = HashMap::new();
                    for (key, name) in [("w_uk_fp8", "w_uk_fp8"), ("w_uk_scale", "w_uk_scale"), ("w_uv_fp8", "w_uv_fp8"),
                        ("w_uv_scale", "w_uv_scale")] {
                        if attention_resident {
                            operands.insert(key, native.remove(name).context("native kv_b owner")?);
                        }
                    }
                    for (key, tensor) in cfg.plain_owners(layer) {
                        if let Some(allocation) = native.remove(&tensor) { operands.insert(key, allocation); }
                    }
                    // The router (BF16 record) on rank 0, as on the official path.
                    if let Some(gate) = native.remove(&cuteafd_loader::plan::exlr8::dense::requests::key("router", "bf16")) {
                        operands.insert("gate", gate);
                    }
                    let dense = layer < cfg.first_moe_layer;
                    ensure!(operands.contains_key("input_norm") && operands.contains_key("post_norm")
                        && (dense || rank != 0 || (operands.contains_key("gate") && operands.contains_key("gate.bias"))),
                        "{p}: native layer operands incomplete on rank {rank}");
                    Ok(GlmLayer { dense, full_indexer: cfg.indexers[layer] == GlmIndexer::Full,
                        mlp_numerics: cuteafd_loader::plan::checkpoint::ModuleNumerics::CheckpointPrecision,
                        split: ranks > 1, attention_split: ranks > 1 && owner.is_none(), attention_owner: owner,
                        attention_resident, operands, native,
                        native_projections: layer_records.projections.iter().map(|(&stem, (p, _))| (stem, p.clone())).collect(),
                        core: Default::default(), attention_native: Default::default() })
                })?;
                share.push(part);
            }
        }
        let norm = self.on_rank(0, |stream| {
            crate::shared::experts::exlr8::dense_weights::landed_with(self.library, load,
                cfg.final_norm(), &records[0].norm,
                cuteafd_loader::plan::exlr8::dense::DenseTransform::Identity, stream,
                |_, bytes| self.allocate_owner("norm", bytes))
        })?;
        let mut shares = shares.into_iter();
        let weights = GlmWeights { layers: shares.next().context("rank 0")?, norm, head: GlmHead::RowFp8(vocabulary),
            rope_alpha: None };
        split.log("glm5 native dense");
        Ok((weights, shares.collect()))
    }
}

/// BF16 checkpoint weights (nvidia/GLM-5.3-NVFP4's attention, indexer and shared experts) run
/// as-is on the BF16 programs with CUTEAFD_GLM_BF16=native; by default they are quantized to
/// 128x128 FP8 blocks at load (power-of-two scales) for the FP8 programs: BF16 reads twice
/// the bytes (one RTX PRO 6000, coordinator alone: C1 step 28.6 vs 21.0 ms, 8K prefill 3.35
/// vs 2.74 s), a decision pending in PLAN.md Phase 5.
pub(crate) fn bf16_programs() -> bool {
    cuteafd_loader::families::glm5::native_bf16()
}

#[cfg(test)]
#[path = "weights_tests.rs"]
mod operand_tests;

#[cfg(test)]
#[path = "weight_inventory_tests.rs"]
mod inventory_tests;
