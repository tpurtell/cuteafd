//! A CPU `Ops` over dense BF16 weights with the official module's rounding
//! points: the chain's oracle for small synthetic geometries, checked against
//! the shared Python reference (`python/reference/families/muse_spark/model.py`,
//! the official transformers classes). Routed experts take dequantized BF16
//! weights: the contract an EXL3 kernel must meet after decode.
use super::block::{Gate, Norm, Ops, Routes};
use anyhow::{ensure, Context, Result};
use cuteafd_loader::families::muse_spark::{reference as contract, MuseAttention, MuseSparkConfig};

pub(crate) type Rows = Vec<Vec<f32>>;

/// A row-major `[n, k]` BF16 matrix (`nn.Linear.weight`).
#[derive(Clone)]
pub(crate) struct Matrix { pub(crate) n: usize, pub(crate) k: usize, pub(crate) w: Vec<f32> }

impl Matrix {
    /// `F.linear` in BF16: FP32 accumulation, one rounding of each output.
    pub(crate) fn apply(&self, x: &[f32]) -> Vec<f32> {
        debug_assert_eq!(x.len(), self.k);
        (0..self.n).map(|i| contract::bf16(self.w[i * self.k..(i + 1) * self.k].iter().zip(x)
            .map(|(w, v)| *w as f64 * *v as f64).sum::<f64>() as f32)).collect()
    }
}

#[derive(Clone)]
pub(crate) struct Layer {
    pub(crate) input_norm: Vec<f32>, pub(crate) pre_ffn_norm: Vec<f32>, pub(crate) post_ffn_norm: Vec<f32>,
    pub(crate) attention_gate: Vec<f32>, pub(crate) ffn_gate: Vec<f32>,
    pub(crate) q: Matrix, pub(crate) k: Matrix, pub(crate) v: Matrix, pub(crate) o: Matrix, pub(crate) g: Matrix,
    /// FP32 router `[experts, hidden]` and selection bias.
    pub(crate) router: Matrix, pub(crate) bias: Vec<f32>,
    pub(crate) pre_expert: Matrix, pub(crate) pre_expert_norm: Vec<f32>,
    pub(crate) post_expert: Matrix, pub(crate) post_expert_norm: Vec<f32>,
    /// Per expert: fused `gate_up` `[2I, E]` and `down` `[E, I]`.
    pub(crate) gate_up: Vec<Vec<f32>>, pub(crate) down: Vec<Vec<f32>>,
}

pub(crate) struct Reference {
    pub(crate) cfg: MuseSparkConfig,
    pub(crate) embedding: Matrix,
    pub(crate) layers: Vec<Layer>,
    pub(crate) norm: Vec<f32>,
    pub(crate) head: Matrix,
    /// One sequence: per layer, every earlier position's (k, v) after norm/RoPE.
    cache: Vec<Vec<(Vec<f32>, Vec<f32>)>>,
    position: usize,
    /// Online (flash-style) softmax: FP32 scores and PV with running max and
    /// sum over key tiles of `tile` rows, one BF16 rounding of the output;
    /// `None` is the eager module's normalized-BF16-probability path.
    pub(crate) online_tile: Option<usize>,
    /// Exchange with a native attention kernel on the same inputs: `dump`
    /// writes each layer's normed/rotated q, k, v and this oracle's attention
    /// output (`layer{L}.{q,k,v,out}.bin`, BF16 bits, [rows, heads, 64]);
    /// `inject` reads `layer{L}.out.bin` (the kernel's output) in place of
    /// computing attention. One prefill call over the whole prompt.
    pub(crate) attention_dump: Option<std::path::PathBuf>,
    pub(crate) attention_inject: Option<std::path::PathBuf>,
}

fn write_bf16(path: &std::path::Path, rows: &[Vec<f32>]) -> Result<()> {
    let bytes: Vec<u8> = rows.iter().flatten().flat_map(|v| ((v.to_bits() >> 16) as u16).to_le_bytes()).collect();
    Ok(std::fs::write(path, bytes)?)
}

fn read_bf16(path: &std::path::Path, rows: usize, width: usize) -> Result<Vec<Vec<f32>>> {
    let bytes = std::fs::read(path).with_context(|| format!("injected attention {}", path.display()))?;
    ensure!(bytes.len() == rows * width * 2, "{}: {} bytes, expected [{rows}, {width}] BF16", path.display(), bytes.len());
    Ok(bytes.chunks_exact(width * 2).map(|row| row.chunks_exact(2)
        .map(|b| f32::from_bits((u16::from_le_bytes([b[0], b[1]]) as u32) << 16)).collect()).collect())
}

impl Reference {
    pub(crate) fn new(cfg: MuseSparkConfig, embedding: Matrix, layers: Vec<Layer>, norm: Vec<f32>, head: Matrix) -> Self {
        let cache = vec![Vec::new(); cfg.layers];
        Self { cfg, embedding, layers, norm, head, cache, position: 0, online_tile: None,
            attention_dump: None, attention_inject: None }
    }

    fn layer(&self, layer: usize) -> Result<&Layer> { self.layers.get(layer).context("layer outside the reference weights") }

    /// `MuseSparkTextRotaryEmbedding` (default rope) for one position, as BF16 cos/sin.
    fn rope(&self, position: usize) -> (Vec<f32>, Vec<f32>) {
        let dim = self.cfg.head_dim;
        let half: Vec<f32> = (0..dim / 2).map(|i| {
            let inv = 1.0f32 / (self.cfg.rope_theta as f32).powf((2 * i) as f32 / dim as f32);
            position as f32 * inv
        }).collect();
        let freqs: Vec<f32> = half.iter().chain(&half).copied().collect();
        (freqs.iter().map(|f| contract::bf16(f.cos())).collect(), freqs.iter().map(|f| contract::bf16(f.sin())).collect())
    }
}

impl Ops for Reference {
    type Rows = Rows;

    fn embed(&mut self, tokens: &[u32]) -> Result<Rows> {
        let h = self.cfg.hidden;
        tokens.iter().map(|&t| {
            let t = t as usize;
            ensure!(t < self.embedding.n, "token {t} outside the vocabulary");
            Ok(contract::rms(&self.embedding.w[t * h..(t + 1) * h], self.cfg.rms_norm_eps))
        }).collect()
    }

    fn norm(&mut self, layer: usize, norm: Norm, x: &Rows) -> Result<Rows> {
        let l = self.layer(layer)?;
        let (eps, post) = (self.cfg.rms_norm_eps, self.cfg.post_norm_eps);
        Ok(x.iter().map(|row| match norm {
            Norm::Input => contract::centered(row, &l.input_norm, eps),
            Norm::PreFeedforward => contract::centered(row, &l.pre_ffn_norm, eps),
            Norm::PostAttention => contract::rms(row, post),
            Norm::PostFeedforward => contract::input_scaled(row, &l.post_ffn_norm, post),
        }).collect())
    }

    /// One sequence, rows at consecutive positions from the cache's length.
    fn attention(&mut self, layer: usize, x: &Rows) -> Result<Rows> {
        let cfg = self.cfg.clone();
        let l = self.layer(layer)?.clone();
        let (heads, kv_heads, dim) = (cfg.heads, cfg.kv_heads, cfg.head_dim);
        let group = heads / kv_heads;
        let scale = (dim as f64).powf(-0.5);
        let sliding = cfg.attention[layer] == MuseAttention::Sliding;
        let eps = cfg.rms_norm_eps;
        let start = self.cache[layer].len();
        let mut out = Vec::with_capacity(x.len());
        let (mut dump_q, mut dump_k, mut dump_v, mut dump_out) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
        let injected = self.attention_inject.as_ref().map(|dir|
            read_bf16(&dir.join(format!("layer{layer}.out.bin")), x.len(), heads * dim)).transpose()?;
        for (offset, row) in x.iter().enumerate() {
            let position = start + offset;
            let (q, k, v) = (l.q.apply(row), l.k.apply(row), l.v.apply(row));
            let rotate = |head: &[f32], (cos, sin): &(Vec<f32>, Vec<f32>)| -> Vec<f32> {
                let half = dim / 2;
                (0..dim).map(|d| {
                    let rotated = if d < half { -head[d + half] } else { head[d - half] };
                    contract::bf16(contract::bf16(head[d] * cos[d]) + contract::bf16(rotated * sin[d]))
                }).collect()
            };
            let table = self.rope(position);
            let mut qh: Vec<Vec<f32>> = (0..heads).map(|h| {
                let normed = contract::rms(&q[h * dim..(h + 1) * dim], eps);
                normed.iter().map(|v| contract::bf16(v * cfg.qk_scale as f32)).collect()
            }).collect();
            let mut kh: Vec<Vec<f32>> = (0..kv_heads).map(|h| contract::rms(&k[h * dim..(h + 1) * dim], eps)).collect();
            if sliding {
                qh = qh.iter().map(|h| rotate(h, &table)).collect();
                kh = kh.iter().map(|h| rotate(h, &table)).collect();
            }
            let vh: Vec<Vec<f32>> = (0..kv_heads).map(|h| v[h * dim..(h + 1) * dim].to_vec()).collect();
            dump_q.push(qh.concat());
            dump_k.push(kh.concat());
            dump_v.push(vh.concat());
            self.cache[layer].push((kh.concat(), vh.concat()));
            let history = &self.cache[layer];
            let first = if sliding { (position + 1).saturating_sub(cfg.window) } else { 0 };
            let mut attended = Vec::with_capacity(heads * dim);
            for (h, qrow) in qh.iter().enumerate() {
                let kv = h / group;
                let keys = &history[first..=position];
                let head_out: Vec<f32> = if let Some(tile) = self.online_tile {
                    // FP32 scores q.k * scale (no BF16 score rounding), running max/sum per tile.
                    let (mut max, mut sum, mut acc) = (f32::NEG_INFINITY, 0f32, vec![0f32; dim]);
                    for block in keys.chunks(tile) {
                        let scores: Vec<f32> = block.iter().map(|(k, _)| qrow.iter().zip(&k[kv * dim..(kv + 1) * dim])
                            .map(|(a, b)| a * b).sum::<f32>() * scale as f32).collect();
                        let block_max = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                        let next = max.max(block_max);
                        let correction = (max - next).exp();
                        sum *= correction;
                        for value in acc.iter_mut() { *value *= correction; }
                        for (score, (_, v)) in scores.iter().zip(block) {
                            let p = (score - next).exp();
                            sum += p;
                            for d in 0..dim { acc[d] += p * v[kv * dim + d]; }
                        }
                        max = next;
                    }
                    acc.iter().map(|v| contract::bf16(v / sum)).collect()
                } else {
                    let scores: Vec<f64> = keys.iter().map(|(k, _)| {
                        // matmul in BF16 then * scaling (rounded), as eager attention.
                        let dot = contract::bf16(qrow.iter().zip(&k[kv * dim..(kv + 1) * dim])
                            .map(|(a, b)| *a as f64 * *b as f64).sum::<f64>() as f32);
                        contract::bf16((dot as f64 * scale) as f32) as f64
                    }).collect();
                    let max = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                    let exp: Vec<f64> = scores.iter().map(|s| (s - max).exp()).collect();
                    let total: f64 = exp.iter().sum();
                    let mut head_out = vec![0f64; dim];
                    for (p, (_, v)) in exp.iter().zip(keys) {
                        let p = contract::bf16((p / total) as f32) as f64;
                        for d in 0..dim { head_out[d] += p * v[kv * dim + d] as f64; }
                    }
                    head_out.iter().map(|v| contract::bf16(*v as f32)).collect()
                };
                attended.extend(head_out);
            }
            if let Some(injected) = &injected { attended = injected[offset].clone(); }
            dump_out.push(attended.clone());
            let attended: Vec<f32> = attended.chunks_exact(dim).flat_map(|head| contract::rms(head, eps)).collect();
            let gate = l.g.apply(row);
            let gated: Vec<f32> = attended.iter().zip(&gate)
                .map(|(a, g)| contract::bf16(a * contract::bf16(1.0 / (1.0 + (-g).exp())))).collect();
            out.push(l.o.apply(&gated));
        }
        if let Some(dir) = &self.attention_dump {
            std::fs::create_dir_all(dir)?;
            for (name, rows) in [("q", &dump_q), ("k", &dump_k), ("v", &dump_v), ("out", &dump_out)] {
                write_bf16(&dir.join(format!("layer{layer}.{name}.bin")), rows)?;
            }
        }
        self.position = start + x.len();
        Ok(out)
    }

    fn residual_gate(&mut self, layer: usize, gate: Gate, residual: &Rows, x: &Rows) -> Result<Rows> {
        let l = self.layer(layer)?;
        let g = match gate { Gate::Attention => &l.attention_gate, Gate::Feedforward => &l.ffn_gate };
        let t = self.cfg.residual_gate_temperature as f32;
        Ok(residual.iter().zip(x).map(|(r, h)| contract::residual_gate(r, h, g, t)).collect())
    }

    fn route(&mut self, layer: usize, x: &Rows) -> Result<Vec<Routes>> {
        let l = self.layer(layer)?;
        let cfg = &self.cfg;
        Ok(x.iter().map(|row| {
            // FP32 linear: no BF16 rounding of the logits.
            let logits: Vec<f32> = (0..l.router.n).map(|e| l.router.w[e * l.router.k..(e + 1) * l.router.k]
                .iter().zip(row).map(|(w, v)| *w as f64 * *v as f64).sum::<f64>() as f32).collect();
            let routes = contract::route(&logits, &l.bias, cfg.topk, cfg.output_multiplier as f32, cfg.routed_scale as f32);
            Routes { experts: routes.iter().map(|r| r.0 as u32).collect(), weights: routes.iter().map(|r| r.1).collect() }
        }).collect())
    }

    fn pre_expert(&mut self, layer: usize, x: &Rows) -> Result<Rows> {
        let l = self.layer(layer)?;
        Ok(x.iter().map(|row| contract::centered(&l.pre_expert.apply(row), &l.pre_expert_norm, self.cfg.rms_norm_eps)).collect())
    }

    /// Per route: expert, post norm, BF16 weighted, summed in F32 in router order.
    fn experts(&mut self, layer: usize, x: &Rows, routes: &[Routes]) -> Result<Rows> {
        let l = self.layer(layer)?;
        let cfg = &self.cfg;
        Ok(x.iter().zip(routes).map(|(row, route)| {
            let mut sum = vec![0f32; row.len()];
            for (&e, &w) in route.experts.iter().zip(&route.weights) {
                let out = contract::expert(row, &l.gate_up[e as usize], &l.down[e as usize], cfg.moe_intermediate,
                    &l.post_expert_norm, cfg.post_norm_eps);
                for (s, o) in sum.iter_mut().zip(&out) { *s += contract::bf16(o * w); }
            }
            sum.iter().map(|v| contract::bf16(*v)).collect()
        }).collect())
    }

    fn post_expert(&mut self, layer: usize, x: &Rows) -> Result<Rows> {
        let l = self.layer(layer)?;
        Ok(x.iter().map(|row| l.post_expert.apply(row)).collect())
    }

    fn head(&mut self, x: &Rows) -> Result<Vec<Vec<f32>>> {
        let cfg = &self.cfg;
        Ok(x.iter().map(|row| {
            let normed = contract::rms_scaled(row, &self.norm, cfg.rms_norm_eps);
            // The head linear is BF16; its BF16 logits then multiply/divide/tanh in BF16 steps.
            self.head.apply(&normed).iter().map(|l| {
                let scaled = contract::bf16(l * cfg.output_multiplier as f32);
                let divided = contract::bf16(scaled / cfg.softcap as f32);
                contract::bf16(contract::bf16(divided.tanh()) * cfg.softcap as f32)
            }).collect()
        }).collect())
    }
}
