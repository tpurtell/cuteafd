//! DeepSeek V4 coordinator weights, built from raw checkpoint bytes exactly as
//! b12x.integration.cuteafd.weights.WEIGHT_SOURCES prescribes: every operand is
//! checkpoint bytes (row-concatenated where listed) except block-FP8 scales,
//! which the scale-prep program re-lays into MMA tile order.
use crate::shared::memory::DeviceAllocation;
use anyhow::{ensure, Context, Result};
use cuteafd_ffi::programs::{Programs, Scalar};
use cuteafd_ffi::NativeLibrary;
use cuteafd_loader::families::deepseek_v4::DeepseekV4Config;
use cuteafd_loader::OfficialV41Catalog;
use std::collections::HashMap;
use std::ffi::c_void;
use std::os::unix::fs::FileExt;
use crate::shared::peer_split::{slice_2d, Axis};

enum Prep {
    Raw,
    /// Block-FP8 scales for a weight [groups*n, k].
    Scale { n: usize, k: usize, groups: usize },
}

struct Source {
    operand: &'static str,
    tensors: &'static [&'static str],
    prep: Prep,
}

pub(crate) struct LayerWeights<'a> {
    pub ratio: usize,
    pub hash: bool,
    /// One GPU's share of a head split: its heads' `w_q` rows and sinks, its
    /// output groups' `wo_a` / `wo_b` columns (a partial sum) and its slice of
    /// the shared expert (a partial sum); it runs the split programs.
    pub split: bool,
    /// Device operands; routing adds `gate.bias` (FP32, score layers) or
    /// `gate.tid2eid` (I32 [vocab, topk], hash layers).
    operands: HashMap<&'static str, DeviceAllocation<'a>>,
}

impl LayerWeights<'_> {
    pub fn ptr(&self, operand: &str) -> Result<*mut c_void> {
        Ok(self.operands.get(operand).with_context(|| format!("layer has no weight {operand}"))?.buffer.ptr)
    }
}

/// The dSpark drafter (`mtp.*`): three window-only blocks, the target-tap
/// projection of stage 0 and the head extras of the last stage.
pub(crate) struct DsparkWeights<'a> {
    pub stages: Vec<LayerWeights<'a>>,
    /// Block-FP8 [dim, taps * dim] and its raw UE8M0 128x128 block scales.
    pub main_proj: DeviceAllocation<'a>,
    pub main_proj_scale: DeviceAllocation<'a>,
    pub main_norm: DeviceAllocation<'a>,
    pub head_fn: DeviceAllocation<'a>,
    pub head_scale: DeviceAllocation<'a>,
    pub head_base: DeviceAllocation<'a>,
    pub norm: DeviceAllocation<'a>,
    /// Markov head, BF16 [vocab, rank] each.
    pub markov_w1: DeviceAllocation<'a>,
    pub markov_w2: DeviceAllocation<'a>,
}

pub(crate) struct ModelWeights<'a> {
    pub layers: Vec<LayerWeights<'a>>,
    pub dspark: Option<DsparkWeights<'a>>,
    pub head: DeviceAllocation<'a>,
    pub head_fn: DeviceAllocation<'a>,
    pub head_scale: DeviceAllocation<'a>,
    pub head_base: DeviceAllocation<'a>,
    pub norm: DeviceAllocation<'a>,
}

fn layer_sources(cfg: &DeepseekV4Config, ratio: usize) -> Vec<Source> {
    let (h, q, heads, g, r) = (cfg.dim, cfg.q_lora_rank, cfg.n_heads, cfg.o_groups, cfg.o_lora_rank);
    let w = heads * 512 / g;
    let inter = cfg.moe_inter_dim;
    let mut sources = vec![
        Source { operand: "attn.fn", tensors: &["hc_attn_fn"], prep: Prep::Raw },
        Source { operand: "attn.scale", tensors: &["hc_attn_scale"], prep: Prep::Raw },
        Source { operand: "attn.base", tensors: &["hc_attn_base"], prep: Prep::Raw },
        Source { operand: "attn.norm", tensors: &["attn_norm.weight"], prep: Prep::Raw },
        Source { operand: "ffn.fn", tensors: &["hc_ffn_fn"], prep: Prep::Raw },
        Source { operand: "ffn.scale", tensors: &["hc_ffn_scale"], prep: Prep::Raw },
        Source { operand: "ffn.base", tensors: &["hc_ffn_base"], prep: Prep::Raw },
        Source { operand: "ffn.norm", tensors: &["ffn_norm.weight"], prep: Prep::Raw },
        Source { operand: "w_qkv", tensors: &["attn.wq_a.weight", "attn.wkv.weight"], prep: Prep::Raw },
        Source { operand: "w_qkv_scale", tensors: &["attn.wq_a.scale", "attn.wkv.scale"],
            prep: Prep::Scale { n: q + 512, k: h, groups: 1 } },
        Source { operand: "w_q", tensors: &["attn.wq_b.weight"], prep: Prep::Raw },
        Source { operand: "w_q_scale", tensors: &["attn.wq_b.scale"], prep: Prep::Scale { n: heads * 512, k: q, groups: 1 } },
        Source { operand: "q_norm", tensors: &["attn.q_norm.weight"], prep: Prep::Raw },
        Source { operand: "kv_norm", tensors: &["attn.kv_norm.weight"], prep: Prep::Raw },
        Source { operand: "attn_sink", tensors: &["attn.attn_sink"], prep: Prep::Raw },
        Source { operand: "wo_a", tensors: &["attn.wo_a.weight"], prep: Prep::Raw },
        Source { operand: "wo_a_scale", tensors: &["attn.wo_a.scale"], prep: Prep::Scale { n: r, k: w, groups: g } },
        Source { operand: "wo_b", tensors: &["attn.wo_b.weight"], prep: Prep::Raw },
        Source { operand: "wo_b_scale", tensors: &["attn.wo_b.scale"], prep: Prep::Scale { n: h, k: g * r, groups: 1 } },
        Source { operand: "w13", tensors: &["ffn.shared_experts.w1.weight", "ffn.shared_experts.w3.weight"], prep: Prep::Raw },
        Source { operand: "w13_scale", tensors: &["ffn.shared_experts.w1.scale", "ffn.shared_experts.w3.scale"],
            prep: Prep::Scale { n: 2 * inter, k: h, groups: 1 } },
        Source { operand: "w2", tensors: &["ffn.shared_experts.w2.weight"], prep: Prep::Raw },
        Source { operand: "w2_scale", tensors: &["ffn.shared_experts.w2.scale"], prep: Prep::Scale { n: h, k: inter, groups: 1 } },
        Source { operand: "gate", tensors: &["ffn.gate.weight"], prep: Prep::Raw },
    ];
    if ratio == 4 {
        sources.extend([
            Source { operand: "joint_projection", tensors: &["attn.compressor.wkv.weight", "attn.compressor.wgate.weight",
                "attn.indexer.compressor.wkv.weight", "attn.indexer.compressor.wgate.weight"], prep: Prep::Raw },
            Source { operand: "index_ape", tensors: &["attn.indexer.compressor.ape"], prep: Prep::Raw },
            Source { operand: "index_norm", tensors: &["attn.indexer.compressor.norm.weight"], prep: Prep::Raw },
            Source { operand: "index_w_q", tensors: &["attn.indexer.wq_b.weight"], prep: Prep::Raw },
            Source { operand: "index_w_q_scale", tensors: &["attn.indexer.wq_b.scale"],
                prep: Prep::Scale { n: cfg.index_n_heads * cfg.index_head_dim, k: q, groups: 1 } },
            Source { operand: "index_w_proj", tensors: &["attn.indexer.weights_proj.weight"], prep: Prep::Raw },
        ]);
    } else if ratio == 128 {
        sources.push(Source { operand: "joint_projection",
            tensors: &["attn.compressor.wkv.weight", "attn.compressor.wgate.weight"], prep: Prep::Raw });
    }
    if ratio != 0 {
        sources.extend([
            Source { operand: "main_ape", tensors: &["attn.compressor.ape"], prep: Prep::Raw },
            Source { operand: "main_norm", tensors: &["attn.compressor.norm.weight"], prep: Prep::Raw },
        ]);
    }
    sources
}

pub(crate) struct WeightLoader<'a, 'p> {
    pub library: &'a NativeLibrary,
    pub catalog: &'a OfficialV41Catalog,
    pub programs: &'p Programs<'a>,
    pub family: &'static str,
    pub stream: *mut c_void,
    /// This loader's device (rank 0 of a head split).
    pub device: i32,
    /// The other GPU of a head split (rank 1), if any.
    pub peers: Vec<crate::shared::peer_split::RankDevice>,
}

/// How a head split shares an operand's checkpoint tensors.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Share {
    /// Every rank holds the whole operand.
    All,
    /// Rows split evenly (per tensor, concatenated): heads, wo groups, shared-expert intermediate.
    Rows,
    /// Columns split evenly: wo_b over the groups, the shared expert's w2 over the intermediate.
    Cols,
}

/// The head-split share of operand `operand`.
fn share(operand: &str) -> Share {
    match operand {
        "w_q" | "w_q_scale" | "attn_sink" | "wo_a" | "wo_a_scale" | "w13" | "w13_scale" => Share::Rows,
        "wo_b" | "wo_b_scale" | "w2" | "w2_scale" => Share::Cols,
        _ => Share::All,
    }
}

impl<'a> WeightLoader<'a, '_> {
    fn read(&self, names: &[String]) -> Result<Vec<u8>> {
        cuteafd_ffi::memory_ledger::tensor(names.first().map_or("", String::as_str));
        let mut bytes = Vec::new();
        for name in names {
            let tensor = self.catalog.tensor(name)?;
            let start = bytes.len();
            bytes.resize(start + tensor.metadata.byte_length as usize, 0);
            std::fs::File::open(self.catalog.snapshot().join(&tensor.shard))?
                .read_exact_at(&mut bytes[start..], tensor.metadata.byte_offset)
                .with_context(|| format!("reading {name}"))?;
        }
        Ok(bytes)
    }

    pub fn upload(&self, bytes: &[u8]) -> Result<DeviceAllocation<'a>> {
        let allocation = DeviceAllocation::new(self.library, bytes.len().max(16))?;
        self.library.copy_h2d(allocation.buffer, bytes)?;
        Ok(allocation)
    }

    pub fn tensor(&self, name: &str) -> Result<DeviceAllocation<'a>> {
        cuteafd_ffi::memory_ledger::tensor(name);
        self.upload(&self.read(&[name.to_string()])?)
    }

    fn scale(&self, raw: &[u8], n: usize, k: usize, groups: usize) -> Result<DeviceAllocation<'a>> {
        ensure!(n % 128 == 0 && k % 128 == 0, "block-FP8 weights need 128-multiple extents");
        let (n_blocks, k_blocks) = (groups * n / 128, k / 128);
        ensure!(raw.len() == n_blocks * k_blocks, "scale bytes {} do not cover {n_blocks}x{k_blocks} blocks", raw.len());
        let source = self.upload(raw)?;
        let output = DeviceAllocation::new(self.library, n_blocks * k_blocks * 512)?;
        let program = self.programs.program(&format!("{}_block_fp8_scale_prep", self.family), &["scale", "scale_mma"])?;
        // SAFETY: both buffers are live device allocations sized for the grid;
        // the stream is synchronized before `source` drops.
        unsafe {
            program.launch(&[source.buffer.ptr, output.buffer.ptr],
                &[Scalar::I32(n_blocks as i32), Scalar::I32(k_blocks as i32)], self.stream)?;
            self.library.cuda_stream_synchronize(self.stream)?;
        }
        Ok(output)
    }

    pub fn layer(&self, cfg: &DeepseekV4Config, layer: usize) -> Result<Vec<LayerWeights<'a>>> {
        if self.peers.is_empty() {
            return Ok(vec![self.block(cfg, &format!("layers.{layer}"), cfg.compress_ratios[layer],
                cfg.is_hash_layer(layer))?]);
        }
        self.block_split(cfg, &format!("layers.{layer}"), cfg.compress_ratios[layer], cfg.is_hash_layer(layer))
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
        crate::shared::peer_split::on_device(self.library, peer.device, self.device, || body(peer.stream))
    }

    /// [`Self::scale`] on `stream` (the current device's).
    fn scale_on(&self, raw: &[u8], n: usize, k: usize, groups: usize, stream: *mut c_void) -> Result<DeviceAllocation<'a>> {
        ensure!(n % 128 == 0 && k % 128 == 0, "block-FP8 weights need 128-multiple extents");
        let (n_blocks, k_blocks) = (groups * n / 128, k / 128);
        ensure!(raw.len() == n_blocks * k_blocks, "scale bytes {} do not cover {n_blocks}x{k_blocks} blocks", raw.len());
        let source = self.upload(raw)?;
        let output = DeviceAllocation::new(self.library, n_blocks * k_blocks * 512)?;
        let program = self.programs.program(&format!("{}_block_fp8_scale_prep", self.family), &["scale", "scale_mma"])?;
        // SAFETY: both buffers are live device allocations sized for the grid;
        // the stream is synchronized before `source` drops.
        unsafe {
            program.launch(&[source.buffer.ptr, output.buffer.ptr],
                &[Scalar::I32(n_blocks as i32), Scalar::I32(k_blocks as i32)], stream)?;
            self.library.cuda_stream_synchronize(stream)?;
        }
        Ok(output)
    }

    /// One block's operands over the head split's ranks (see [`share`]): each
    /// tensor read once, every rank's slice uploaded on its GPU (scales re-laid
    /// by that GPU's scale-prep program at the slice's extents).
    fn block_split(&self, cfg: &DeepseekV4Config, prefix: &str, ratio: usize, hash: bool) -> Result<Vec<LayerWeights<'a>>> {
        let ranks = self.ranks();
        let mut operands: Vec<HashMap<&'static str, DeviceAllocation<'a>>> = (0..ranks).map(|_| HashMap::new()).collect();
        for source in layer_sources(cfg, ratio) {
            let kind = share(source.operand);
            let names: Vec<String> = source.tensors.iter().map(|t| format!("{prefix}.{t}")).collect();
            // Per rank, the concatenation of each tensor's slice.
            let mut parts: Vec<Vec<u8>> = vec![Vec::new(); ranks];
            for name in &names {
                let raw = self.read(std::slice::from_ref(name))?;
                let shape = self.catalog.tensor(name)?.metadata.shape.clone();
                for (rank, part) in parts.iter_mut().enumerate() {
                    match kind {
                        Share::All => part.extend_from_slice(&raw),
                        Share::Rows => {
                            ensure!(raw.len() % ranks == 0 && shape.first().is_some_and(|r| r % ranks == 0),
                                "{name}: {shape:?} does not split by rows over {ranks} GPUs");
                            part.extend_from_slice(&raw[rank * raw.len() / ranks..(rank + 1) * raw.len() / ranks]);
                        }
                        Share::Cols => {
                            ensure!(shape.len() == 2 && shape[1] % ranks == 0 && raw.len() == shape[0] * shape[1],
                                "{name}: {shape:?} does not split by byte columns over {ranks} GPUs");
                            part.extend(slice_2d(&raw, shape[0], shape[1], 1, Axis::Cols, rank, ranks));
                        }
                    }
                }
            }
            for (rank, part) in parts.into_iter().enumerate() {
                let allocation = self.on_rank(rank, |stream| match source.prep {
                    Prep::Raw => self.upload(&part),
                    Prep::Scale { n, k, groups } => {
                        let (n, k, groups) = match (source.operand, kind) {
                            ("wo_a_scale", _) => (n, k, groups / ranks),
                            (_, Share::Rows) => (n / ranks, k, groups),
                            (_, Share::Cols) => (n, k / ranks, groups),
                            _ => (n, k, groups),
                        };
                        self.scale_on(&part, n, k, groups, stream)
                    }
                })?;
                operands[rank].insert(source.operand, allocation);
            }
        }
        if hash {
            // The checkpoint stores I64 expert ids; the router reads I32.
            let raw = self.read(&[format!("{prefix}.ffn.gate.tid2eid")])?;
            let ids = raw.chunks_exact(8).map(|b| i64::from_le_bytes(b.try_into().unwrap()))
                .map(|id| i32::try_from(id).map(i32::to_le_bytes)).collect::<Result<Vec<_>, _>>()?;
            for (rank, operands) in operands.iter_mut().enumerate() {
                operands.insert("gate.tid2eid", self.on_rank(rank, |_| self.upload(ids.as_flattened()))?);
            }
        } else {
            let raw = self.read(&[format!("{prefix}.ffn.gate.bias")])?;
            for (rank, operands) in operands.iter_mut().enumerate() {
                operands.insert("gate.bias", self.on_rank(rank, |_| self.upload(&raw))?);
            }
        }
        Ok(operands.into_iter().map(|operands| LayerWeights { ratio, hash, split: true, operands }).collect())
    }

    /// One block's operands from the checkpoint names under `prefix`.
    fn block(&self, cfg: &DeepseekV4Config, prefix: &str, ratio: usize, hash: bool) -> Result<LayerWeights<'a>> {
        let mut operands = HashMap::new();
        for source in layer_sources(cfg, ratio) {
            let names: Vec<String> = source.tensors.iter().map(|t| format!("{prefix}.{t}")).collect();
            let raw = self.read(&names)?;
            let allocation = match source.prep {
                Prep::Raw => self.upload(&raw)?,
                Prep::Scale { n, k, groups } => self.scale(&raw, n, k, groups)?,
            };
            operands.insert(source.operand, allocation);
        }
        if hash {
            // The checkpoint stores I64 expert ids; the router reads I32.
            let raw = self.read(&[format!("{prefix}.ffn.gate.tid2eid")])?;
            let ids = raw.chunks_exact(8).map(|b| i64::from_le_bytes(b.try_into().unwrap()))
                .map(|id| i32::try_from(id).map(i32::to_le_bytes)).collect::<Result<Vec<_>, _>>()?;
            operands.insert("gate.tid2eid", self.upload(ids.as_flattened())?);
        } else {
            operands.insert("gate.bias", self.tensor(&format!("{prefix}.ffn.gate.bias"))?);
        }
        Ok(LayerWeights { ratio, hash, split: false, operands })
    }

    /// The drafter, when the checkpoint carries one (`mtp.0.main_proj`).
    fn dspark(&self, cfg: &DeepseekV4Config) -> Result<Option<DsparkWeights<'a>>> {
        if cfg.dspark_block_size == 0 || self.catalog.tensor("mtp.0.main_proj.weight").is_err() {
            return Ok(None);
        }
        let stages = (0..3).map(|stage| self.block(cfg, &format!("mtp.{stage}"), 0, false))
            .collect::<Result<Vec<_>>>()?;
        let last = stages.len() - 1;
        Ok(Some(DsparkWeights {
            stages,
            main_proj: self.tensor("mtp.0.main_proj.weight")?,
            main_proj_scale: self.tensor("mtp.0.main_proj.scale")?,
            main_norm: self.tensor("mtp.0.main_norm.weight")?,
            head_fn: self.tensor(&format!("mtp.{last}.hc_head_fn"))?,
            head_scale: self.tensor(&format!("mtp.{last}.hc_head_scale"))?,
            head_base: self.tensor(&format!("mtp.{last}.hc_head_base"))?,
            norm: self.tensor(&format!("mtp.{last}.norm.weight"))?,
            markov_w1: self.tensor(&format!("mtp.{last}.markov_head.markov_w1.weight"))?,
            markov_w2: self.tensor(&format!("mtp.{last}.markov_head.markov_w2.weight"))?,
        }))
    }

    /// Every layer (rank 0's shares under a head split, the other rank's in the
    /// second value), the drafter and the head.
    #[allow(clippy::type_complexity)]
    pub fn model(&self, cfg: &DeepseekV4Config) -> Result<(ModelWeights<'a>, Vec<Vec<LayerWeights<'a>>>)> {
        let mut shares: Vec<Vec<LayerWeights<'a>>> = (0..self.ranks()).map(|_| Vec::new()).collect();
        for layer in 0..cfg.n_layers {
            for (share, part) in shares.iter_mut().zip(self.layer(cfg, layer)?) {
                share.push(part);
            }
        }
        let mut shares = shares.into_iter();
        let layers = shares.next().context("rank 0")?;
        Ok((ModelWeights {
            layers,
            dspark: self.dspark(cfg)?,
            head: self.tensor("head.weight")?,
            head_fn: self.tensor("hc_head_fn")?,
            head_scale: self.tensor("hc_head_scale")?,
            head_base: self.tensor("hc_head_base")?,
            norm: self.tensor("norm.weight")?,
        }, shares.collect()))
    }

}
