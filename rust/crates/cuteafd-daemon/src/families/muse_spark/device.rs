//! One coordinator rank's Muse Spark layer step on the device: the bound
//! programs, this rank's packed layer weights and K6 projections, the lane's
//! planes, the rotary table and the KV storage, launched in the order of
//! `block::forward_layer` on the rank's lane stream.
//!
//! Head-split TP2: attention runs on this rank's heads; o_proj returns an F32
//! partial that [`PeerReduce`] sums with the peer's (BF16 rounding once, in
//! the shared K-split reduction); everything else is whole on each rank. The
//! routed experts go through [`SparkExperts`] (the Spark partial path), which
//! leaves the combined expert rows in the lane's `combined` plane.
use super::{core::CorePrograms, experts::SparkExperts, lane::LaneViews, programs, step::StepMetadata,
    weights::LayerWeights};
use anyhow::{ensure, Context, Result};
use cuteafd_ffi::{programs::{Program, Programs, Scalar}, CuteafdDeviceBuffer as Buffer};
use cuteafd_loader::families::muse_spark::{MuseAttention, MuseSparkConfig};
use std::{collections::BTreeMap, ffi::c_void};

/// The TP2 o_proj reduction: this rank's F32 partial with its peer's, into
/// the BF16 `attended` rows on both ranks (the shared K-split reduction).
pub(crate) trait PeerReduce {
    fn reduce(&mut self, layer: usize, rows: usize, partial: Buffer, output: Buffer) -> Result<()>;
}

/// A rank's view of one layer's KV storage.
#[derive(Debug, Clone, Copy)]
pub(crate) enum LayerKv {
    /// Full (NoPE): the layer's paged K|V records.
    Paged { cache: Buffer },
    /// Sliding: every sequence's ring of this layer.
    Ring { ring: Buffer },
}

/// Bound programs of one rank, by stem (`muse_` or `muse2_` resolved).
pub(crate) struct RankPrograms<'a> {
    bound: BTreeMap<String, Program<'a>>,
    prefix: &'static str,
}

impl<'a> RankPrograms<'a> {
    pub(crate) fn bind(programs: &'a Programs<'a>, ranks: usize) -> Result<Self> {
        Ok(Self { bound: programs::bind(programs, ranks)?.into_iter().collect(), prefix: programs::prefix(ranks)? })
    }

    /// A geometry-free (`muse_`) or per-heads (`muse_`/`muse2_`) program.
    fn get(&self, stem: &str) -> Result<&Program<'a>> {
        let heads = format!("{}_{stem}", self.prefix);
        self.bound.get(&heads).or_else(|| self.bound.get(&format!("muse_{stem}")))
            .with_context(|| format!("Muse program {stem} not bound"))
    }

    /// # Safety
    /// Every buffer is a live device allocation of the documented extent on the
    /// current device, retained until `stream` reaches the launch.
    unsafe fn launch(&self, stem: &str, buffers: &[Buffer], scalars: &[Scalar], stream: *mut c_void) -> Result<()> {
        let program = self.get(stem)?;
        let pointers: Vec<*mut c_void> = buffers.iter().map(|b| b.ptr).collect();
        // SAFETY: forwarded from the caller; counts and scalar kinds are checked by `launch`.
        unsafe { program.launch(&pointers, scalars, stream) }
    }
}

/// One rank's device state for the layer chain.
pub(crate) struct RankDevice<'a, 'w> {
    pub cfg: &'w MuseSparkConfig,
    pub programs: &'w RankPrograms<'a>,
    pub core: &'w CorePrograms<'a>,
    pub layers: &'w [LayerWeights<'a>],
    pub lane: &'w LaneViews,
    /// The shared half-width rotary table (cos, sin) [positions, 32].
    pub rope: (Buffer, Buffer),
    pub kv: &'w [LayerKv],
    pub ranks: usize,
    pub stream: *mut c_void,
}

/// One step's device metadata planes (copied from [`StepMetadata`] before the layers).
#[derive(Debug, Clone, Copy)]
pub(crate) struct StepPlanes {
    pub rows: usize,
    pub positions: Buffer,
    pub ring_slots: Buffer,
    pub seq_first: Buffer,
    /// Full layers' record slots (`page_slots`) and SWA records' (`step_slots`).
    pub page_slots: Buffer,
    pub step_slots: Buffer,
    pub page_table: Buffer,
    pub table_stride: usize,
    /// Prefill (one sequence, shared table row) or decode/verify.
    pub prefill: bool,
}

impl StepPlanes {
    pub(crate) fn check(&self, meta: &StepMetadata) -> Result<()> {
        ensure!(self.rows == meta.positions.len() && self.table_stride == meta.table_stride
            && self.positions.bytes >= self.rows * 8 && self.ring_slots.bytes >= self.rows * 8
            && self.seq_first.bytes >= self.rows * 4 && self.page_slots.bytes >= self.rows * 8
            && self.step_slots.bytes >= self.rows * 8 && self.page_table.bytes >= meta.page_table.len() * 4,
            "Muse step planes differ from the step metadata");
        Ok(())
    }
}

fn rows_i32(rows: usize) -> Result<Scalar> { Ok(Scalar::I32(i32::try_from(rows)?)) }

impl RankDevice<'_, '_> {
    fn weights(&self, layer: usize) -> Result<&LayerWeights<'_>> {
        self.layers.get(layer).with_context(|| format!("Muse rank holds no layer {layer}"))
    }

    /// A projection that is K6 core or BF16, output BF16.
    fn project(&self, layer: usize, name: &str, rows: usize, input: Buffer, output: Buffer) -> Result<()> {
        let weights = self.weights(layer)?;
        if let Some(core) = weights.core(name) {
            return self.core.local(rows, input, core, output, self.stream);
        }
        // BF16 checkpoint projections (gate_proj): the exported BF16 projection.
        ensure!(name == "attn.gate", "Muse BF16 projection {name} has no program");
        // SAFETY: lane planes and resident weights are live and sized by admission.
        unsafe { self.programs.launch("gate_proj", &[input, weights.buffer(name)?, output],
            &[rows_i32(rows)?], self.stream) }
    }

    /// `norm` over `x` into `out` (weightless norms take no weight).
    pub(crate) fn norm(&self, layer: usize, stem: &str, weight: Option<&str>, rows: usize, x: Buffer, out: Buffer)
        -> Result<()> {
        let mut buffers = vec![x];
        if let Some(name) = weight { buffers.push(self.weights(layer)?.buffer(name)?); }
        buffers.push(out);
        // SAFETY: as `project`.
        unsafe { self.programs.launch(stem, &buffers, &[rows_i32(rows)?], self.stream) }
    }

    /// Attention of one layer for the step's rows: q/k/v/gate projections,
    /// q/k norm (+RoPE on sliding) writing q and each row's K|V record, the
    /// attention route, the output gate and o_proj into `attended`.
    pub(crate) fn attention(&self, layer: usize, step: &StepPlanes, reduce: Option<&mut dyn PeerReduce>) -> Result<()> {
        let rows = step.rows;
        let lane = self.lane;
        let normed = lane.rows("normed", rows)?;
        // v lands in the `attention` plane ([rows, H_q, 64] >= [rows, H_kv, 64]): qk_norm copies it
        // into the record before the attention overwrites the plane with its output (same stream).
        for (name, plane) in [("attn.q", "q_raw"), ("attn.k", "k_raw"), ("attn.v", "attention"), ("attn.gate", "gate")] {
            self.project(layer, name, rows, normed, lane.rows(plane, rows)?)?;
        }
        let v = lane.rows("attention", rows)?;
        let full = self.cfg.attention[layer] == MuseAttention::Full;
        let (kv_dst, slots) = match (self.kv.get(layer).context("Muse layer KV absent")?, full) {
            (LayerKv::Paged { cache }, true) => (*cache, step.page_slots),
            (LayerKv::Ring { .. }, false) => (lane.rows("kv_step", rows)?, step.step_slots),
            _ => anyhow::bail!("Muse layer {layer} KV kind differs from its attention"),
        };
        let q = lane.rows("q", rows)?;
        let qk = if full { "qk_norm_nope_full" } else { "qk_norm_rope_swa" };
        // SAFETY: as `project`; record writes precede the attention on this stream.
        unsafe {
            self.programs.launch(qk, &[lane.rows("q_raw", rows)?, lane.rows("k_raw", rows)?, v, step.positions,
                self.rope.0, self.rope.1, slots, q, kv_dst], &[rows_i32(rows)?], self.stream)?;
        }
        let out = lane.rows("attention", rows)?;
        let scratch = lane.plane("router_logits")?;
        let route = if step.prefill { "prefill" } else { "decode" };
        // SAFETY: as `project`; `scratch` (1 KiB used) is the idle router plane, rewritten later this layer.
        unsafe {
            match self.kv[layer] {
                LayerKv::Paged { cache } => {
                    let mut scalars = vec![rows_i32(rows)?, Scalar::I32(i32::try_from(step.table_stride)?)];
                    if !step.prefill { scalars.push(Scalar::I32(1)); }
                    self.programs.launch(&format!("attention_full_{route}"), &[q, cache, step.positions,
                        step.page_table, out, scratch], &scalars, self.stream)?;
                }
                LayerKv::Ring { ring } => {
                    let kv_step = lane.rows("kv_step", rows)?;
                    self.programs.launch(&format!("attention_swa_{route}"), &[q, kv_step, ring, step.positions,
                        step.ring_slots, step.seq_first, out, scratch], &[rows_i32(rows)?], self.stream)?;
                    self.programs.launch("swa_commit", &[kv_step, ring, step.ring_slots, step.seq_first],
                        &[rows_i32(rows)?], self.stream)?;
                }
            }
            self.programs.launch("attention_output", &[out, lane.rows("gate", rows)?, lane.rows("gated", rows)?],
                &[rows_i32(rows)?], self.stream)?;
        }
        let gated = lane.rows("gated", rows)?;
        let attended = lane.rows("attended", rows)?;
        let weights = self.weights(layer)?;
        match (self.ranks, weights.core("attn.o")) {
            (1, Some(core)) => self.core.local(rows, gated, core, attended, self.stream),
            (2, Some(core)) => {
                let partial = self.core.core(rows, gated, core, self.stream)?;
                reduce.context("Muse TP2 o_proj needs the peer reduction")?.reduce(layer, rows, partial, attended)
            }
            _ => anyhow::bail!("Muse o_proj must be a K6 core projection"),
        }
    }

    /// `out = bf16(alpha * residual + beta * x)` with the layer's site gate.
    pub(crate) fn residual_gate(&self, layer: usize, site: &str, rows: usize, residual: Buffer, x: Buffer, out: Buffer)
        -> Result<()> {
        let weights = self.weights(layer)?;
        let (alpha, beta) = (weights.buffer(&format!("gate.{site}.alpha"))?, weights.buffer(&format!("gate.{site}.beta"))?);
        // SAFETY: as `project`.
        unsafe { self.programs.launch("residual_gate", &[residual, x, alpha, beta, out], &[rows_i32(rows)?], self.stream) }
    }

    /// The MoE frontend: FP32 router + selection on the pre-feedforward-normed
    /// rows, and pre_expert_proj + its centered norm into `expert_input`.
    pub(crate) fn moe_frontend(&self, layer: usize, rows: usize) -> Result<()> {
        let lane = self.lane;
        let weights = self.weights(layer)?;
        let normed = lane.rows("normed", rows)?;
        let logits = lane.rows("router_logits", rows)?;
        // SAFETY: as `project`.
        unsafe {
            self.programs.launch("router", &[normed, weights.buffer("moe.router")?, logits], &[rows_i32(rows)?], self.stream)?;
            self.programs.launch("route_select", &[logits, weights.buffer("moe.bias")?, lane.rows("route_ids", rows)?,
                lane.rows("route_weights", rows)?], &[rows_i32(rows)?], self.stream)?;
        }
        // pre_expert_proj into `combined` (E wide, free until this layer's experts land), then its
        // centered norm into `expert_input`, the plane the Spark path quantizes for the wire.
        let staged = lane.rows("combined", rows)?;
        self.project(layer, "moe.pre_expert", rows, normed, staged)?;
        self.norm(layer, "pre_expert_norm", Some("moe.pre_expert_norm"), rows, staged, lane.rows("expert_input", rows)?)
    }

    /// The MoE backend after the experts landed: post_expert_proj into `moe`.
    pub(crate) fn moe_backend(&self, layer: usize, rows: usize) -> Result<()> {
        self.project(layer, "moe.post_expert", rows, self.lane.rows("combined", rows)?, self.lane.rows("moe", rows)?)
    }

    /// One decoder layer, in `block::forward_layer` order, on `hidden` (in place).
    pub(crate) fn layer(&self, layer: usize, step: &StepPlanes, experts: &mut dyn SparkExperts,
        reduce: Option<&mut dyn PeerReduce>) -> Result<()> {
        let (rows, lane) = (step.rows, self.lane);
        let (hidden, normed, residual) = (lane.rows("hidden", rows)?, lane.rows("normed", rows)?, lane.rows("residual", rows)?);
        self.norm(layer, "input_norm", Some("norm.input"), rows, hidden, normed)?;
        self.attention(layer, step, reduce)?;
        let attended = lane.rows("attended", rows)?;
        self.norm(layer, "post_attn_norm", None, rows, attended, normed)?;
        self.residual_gate(layer, "attention", rows, hidden, normed, residual)?;
        self.norm(layer, "pre_ffn_norm", Some("norm.pre_ffn"), rows, residual, normed)?;
        self.moe_frontend(layer, rows)?;
        let wave = experts.send(layer, rows)?;
        experts.land(wave)?;
        self.moe_backend(layer, rows)?;
        let moe = lane.rows("moe", rows)?;
        self.norm(layer, "post_ffn_norm", Some("norm.post_ffn"), rows, moe, normed)?;
        self.residual_gate(layer, "ffn", rows, residual, normed, hidden)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cuteafd_loader::families::muse_spark::lane::LanePlanes;

    /// Every plane a launch passes is at least its program operand's extent at
    /// the step's rows (rank heads under TP2): `[rows, ...]` operands by row bytes.
    #[test]
    fn launch_planes_cover_each_operand_extent() -> Result<()> {
        let cfg = MuseSparkConfig::from_hf(&cuteafd_loader::plan::testing::muse_spark_config())?;
        for ranks in [1usize, 2] {
            let planes = LanePlanes::new(&cfg, ranks, 2752)?;
            let row = |name: &str| planes.planes[name] / planes.rows;
            let (h, e, d) = (cfg.hidden, cfg.expert_hidden, cfg.head_dim);
            let (hq, hkv) = (cfg.heads / ranks, cfg.kv_heads / ranks);
            // (plane, the widest operand row bytes it is passed as)
            for (plane, needed) in [
                ("q_raw", hq * d * 2), ("k_raw", hkv * d * 2), ("q", hq * d * 2),
                // v (H_kv) then the attention output (H_q) share the attention plane.
                ("attention", (hq * d * 2).max(hkv * d * 2)),
                ("gate", hq * d * 2), ("gated", hq * d * 2), ("kv_step", hkv * 128 * 2),
                ("attended", h * 2), ("hidden", h * 2), ("normed", h * 2), ("residual", h * 2),
                ("router_logits", cfg.experts * 4), ("route_ids", cfg.topk * 4), ("route_weights", cfg.topk * 4),
                // pre_expert_proj stages in `combined` (E wide) before the experts land there.
                ("combined", e * 2), ("expert_input", e * 2), ("moe", h * 2),
                ("expert_wire", e + e / 32), ("positions", 8), ("ring_slots", 8), ("seq_first", 4),
            ] {
                assert!(row(plane) >= needed, "rank layout {ranks}: plane {plane} {} < {needed}", row(plane));
            }
            // The 1 KiB attention scratch borrows the router plane at a single row.
            assert!(planes.planes["router_logits"] >= 1024 || planes.rows == 1);
        }
        Ok(())
    }
}
