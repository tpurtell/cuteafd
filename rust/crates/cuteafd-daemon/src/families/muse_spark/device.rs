//! One coordinator rank's Muse Spark layer step on the device: the bound
//! programs, this rank's packed layer weights and K6 projections, the lane's
//! planes, the rotary table and the KV storage, launched in the order of
//! `block::forward_layer` on the rank's lane stream.
//!
//! Head-split TP2: attention runs on this rank's heads; o_proj's F32 core is
//! narrowed to BF16 (`o_core`), exchanged with the peer ([`PeerExchange`]),
//! and the two cores are added in rank order in F32 (`o_reduced`) before the
//! one egress (the shared K-split rule); everything else is whole on each
//! rank. The routed experts go through [`SparkExperts`] on rank 0 (the Spark
//! link's device), which leaves the combined expert rows in `combined`; rank
//! 0 pushes them to rank 1, which projects straight from its receive slot.
//! Every exchange lands in the peer's round-robin receive slots
//! (`cuteafd_loader::families::muse_spark::tp2::RECEIVE_SLOTS`, proven by its
//! interleaving model).
use super::{core::CorePrograms, experts::SparkExperts, lane::LaneViews, programs, step::StepMetadata,
    weights::LayerWeights};
use cuteafd_loader::families::muse_spark::tp2::Channel;
use anyhow::{ensure, Context, Result};
use cuteafd_ffi::{programs::{Program, Programs, Scalar}, CuteafdDeviceBuffer as Buffer};
use cuteafd_loader::families::muse_spark::{MuseAttention, MuseSparkConfig};
use std::{collections::BTreeMap, ffi::c_void};

/// One rank's end of the TP2 exchange (implemented over the shared
/// `PeerExchange`, one receive slot per [`Channel`] as the interleaving model in
/// `cuteafd_loader::families::muse_spark::tp2` proves): every call queues on
/// this rank's stream only, and the host enqueues both ranks in the same order.
pub(crate) trait PeerExchange {
    /// This rank's `local` to the peer's `channel` slot, then a wait for the
    /// peer's part of the same exchange; returns the received part's view.
    fn swap(&mut self, channel: Channel, local: Buffer) -> Result<Buffer>;
    /// One-way push of `rows` rows of `row_bytes` (source pitch `pitch`) from
    /// `source` into the peer's `channel` slot, packed.
    fn send(&mut self, channel: Channel, source: Buffer, rows: usize, row_bytes: usize, pitch: usize) -> Result<()>;
    /// Wait for the peer's one-way push on `channel`; returns its packed view.
    fn receive(&mut self, channel: Channel, bytes: usize) -> Result<Buffer>;
}

/// The shared TP2 helpers a rank's o_proj reduction launches.
pub(crate) struct Tp2Ops<'a> {
    pub add: cuteafd_ffi::Bf16F32Add<'a>,
    pub narrow: cuteafd_ffi::F32Bf16Narrow<'a>,
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
    pub(crate) unsafe fn launch(&self, stem: &str, buffers: &[Buffer], scalars: &[Scalar], stream: *mut c_void) -> Result<()> {
        let program = self.get(stem)?;
        let pointers: Vec<*mut c_void> = buffers.iter().map(|b| b.ptr).collect();
        // SAFETY: forwarded from the caller; counts and scalar kinds are checked by `launch`.
        unsafe { program.launch(&pointers, scalars, stream) }
    }
}

/// One rank's device state for the layer chain.
pub(crate) struct RankDevice<'a, 'w> {
    pub library: &'a cuteafd_ffi::NativeLibrary,
    pub cfg: &'w MuseSparkConfig,
    pub programs: &'w RankPrograms<'a>,
    pub core: &'w CorePrograms<'a>,
    pub layers: &'w [LayerWeights<'a>],
    pub lane: &'w LaneViews,
    /// The shared half-width rotary table (cos, sin) [positions, 32].
    pub rope: (Buffer, Buffer),
    pub kv: &'w [LayerKv],
    pub ranks: usize,
    /// This GPU's rank (0 holds the Spark link under TP2) and the TP2 helpers.
    pub rank: usize,
    pub tp2: Option<&'w Tp2Ops<'a>>,
    /// The wire quantizer's device-sized grid (cached at startup).
    pub wire_grid: crate::shared::launch_grid::Fp8QuantizeGrid,
    pub stream: *mut c_void,
}

/// One step's device metadata planes (copied from [`StepMetadata`] before the layers).
#[derive(Debug, Clone)]
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
    /// The step's requests and kind (each Spark wave's reply regions).
    pub wave: super::experts::WaveStep,
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
    pub(crate) fn attention(&self, layer: usize, step: &StepPlanes, peer: Option<&mut (dyn PeerExchange + '_)>)
        -> Result<()> {
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
            (2, Some(core)) => self.k_split(Channel::Attention, rows, gated, core, attended,
                peer.context("Muse TP2 o_proj needs the peer exchange")?),
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

    /// The shared K-split rule on a TP2 pair: this rank's F32 core narrowed to
    /// BF16 (`o_core`), swapped with the peer's on `channel`, both widened and
    /// added in rank order in F32 (`o_reduced`), then the one egress into
    /// `output`. Identical bits on both ranks.
    fn k_split(&self, channel: Channel, rows: usize, input: Buffer, core: &crate::shared::dense_trellis::CoreWeights<'_>,
        output: Buffer, peer: &mut dyn PeerExchange) -> Result<()> {
        let ops = self.tp2.context("Muse TP2 needs the shared add/narrow helpers")?;
        let accumulator = self.core.core(rows, input, core, self.stream)?;
        let local = self.lane.rows("o_core", rows)?;
        crate::shared::dense_trellis::retain_partial(&ops.narrow, accumulator, local, rows * self.cfg.hidden,
            self.stream)?;
        let received = peer.swap(channel, local)?;
        let (rank0, rank1) = if self.rank == 0 { (local, received) } else { (received, local) };
        self.core.reduced(&ops.add, rows, rank0, rank1, self.lane.rows("o_reduced", rows)?, core, output, self.stream)
    }

    /// `rows` rows of `width` bytes from `source` (pitch `source_pitch`) into
    /// `destination` (pitch `destination_pitch`), on this stream.
    fn copy_rows(&self, destination: Buffer, destination_pitch: usize, source: Buffer, source_pitch: usize,
        width: usize, rows: usize) -> Result<()> {
        // SAFETY: lane planes and receive slots are live allocations of this device,
        // ordered on this stream (a received slot behind its wait).
        unsafe { self.library.copy_d2d_2d_async(destination, destination_pitch, source, source_pitch, width, rows,
            self.stream) }
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

    /// TP2 Spark layer frontend: the router on both ranks (whole, tiny); this
    /// rank's half of pre_expert_proj (output split over E) into `expert_half`;
    /// rank 1 pushes it to rank 0, which assembles the full row in `combined`
    /// (its half left, rank 1's right) and runs the centered norm (full rows) into
    /// `expert_input`. Only rank 0 holds the Spark link.
    fn moe_frontend_split(&self, layer: usize, rows: usize, peer: &mut dyn PeerExchange) -> Result<()> {
        self.moe_frontend_halves(layer, rows, peer, false)
    }

    /// The split frontend; `both`: an onboarded layer, whose experts run on both
    /// GPUs, so the halves are swapped and both assemble the full expert input.
    fn moe_frontend_halves(&self, layer: usize, rows: usize, peer: &mut dyn PeerExchange, both: bool) -> Result<()> {
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
        let half_bytes = self.cfg.expert_hidden; // E/2 BF16 values
        let half = lane.rows("expert_half", rows)?;
        self.project(layer, "moe.pre_expert", rows, normed, half)?;
        let received = if both { peer.swap(Channel::ExpertInput, half)? } else {
            if self.rank == 1 { return peer.send(Channel::ExpertInput, half, rows, half_bytes, half_bytes); }
            peer.receive(Channel::ExpertInput, rows * half_bytes)?
        };
        // Rank order: rank 0's half is the left columns on both GPUs.
        let (left, right_half) = if self.rank == 0 { (half, received) } else { (received, half) };
        let staged = lane.rows("combined", rows)?;
        let full = 2 * half_bytes;
        self.copy_rows(staged, full, left, half_bytes, half_bytes, rows)?;
        let right = Buffer { ptr: staged.ptr.cast::<u8>().wrapping_add(half_bytes).cast(), ..staged };
        self.copy_rows(right, full, right_half, half_bytes, half_bytes, rows)?;
        self.norm(layer, "pre_expert_norm", Some("moe.pre_expert_norm"), rows, staged, lane.rows("expert_input", rows)?)
    }

    /// The expert-width rows onto the shared FP8 K32 Spark wire (`expert_wire`,
    /// 4224-byte rows: values then the UE8M0 scales), read by `SparkExperts::send`.
    pub(crate) fn expert_wire(&self, rows: usize) -> Result<()> {
        let wire = self.lane.rows("expert_wire", rows)?;
        let scales = Buffer { ptr: wire.ptr.cast::<u8>().wrapping_add(self.cfg.expert_hidden).cast(), ..wire };
        // SAFETY: as `project`; scale_mma is unused in wire mode (the exporter's contract).
        unsafe {
            self.programs.launch("expert_input_quant", &[self.lane.rows("expert_input", rows)?, wire, scales, wire],
                &[rows_i32(rows)?, Scalar::I32(i32::try_from(self.wire_grid.blocks(rows, self.cfg.expert_hidden))?)],
                self.stream)
        }
    }

    /// The MoE backend after the experts landed: post_expert_proj of `combined` into `moe`.
    pub(crate) fn moe_backend(&self, layer: usize, rows: usize) -> Result<()> {
        self.project(layer, "moe.post_expert", rows, self.lane.rows("combined", rows)?, self.lane.rows("moe", rows)?)
    }

    /// TP2 backend: post_expert_proj K-split over E. Rank 0 (which combined)
    /// pushes the right half of `combined` to rank 1 and packs its own left
    /// half into `expert_half`; each rank's half core is swapped and reduced in
    /// rank order into `moe` on both.
    fn moe_backend_split(&self, layer: usize, rows: usize, peer: &mut dyn PeerExchange) -> Result<()> {
        self.moe_backend_halves(layer, rows, peer, false)
    }

    /// The K-split backend; `both`: an onboarded layer, where both ranks
    /// combined the full rows, so each packs its own half locally (no
    /// `Combined` exchange).
    fn moe_backend_halves(&self, layer: usize, rows: usize, peer: &mut dyn PeerExchange, both: bool) -> Result<()> {
        let lane = self.lane;
        let half_bytes = self.cfg.expert_hidden;
        let full = 2 * half_bytes;
        let input = if both {
            let combined = lane.rows("combined", rows)?;
            let own = Buffer { ptr: combined.ptr.cast::<u8>().wrapping_add(self.rank * half_bytes).cast(), ..combined };
            let half = lane.rows("expert_half", rows)?;
            self.copy_rows(half, half_bytes, own, full, half_bytes, rows)?;
            half
        } else if self.rank == 0 {
            let combined = lane.rows("combined", rows)?;
            let right = Buffer { ptr: combined.ptr.cast::<u8>().wrapping_add(half_bytes).cast(), ..combined };
            peer.send(Channel::Combined, right, rows, half_bytes, full)?;
            let half = lane.rows("expert_half", rows)?;
            self.copy_rows(half, half_bytes, combined, full, half_bytes, rows)?;
            half
        } else {
            peer.receive(Channel::Combined, rows * half_bytes)?
        };
        let core = self.weights(layer)?.core("moe.post_expert").context("Muse TP2 post_expert must be K6 core")?;
        self.k_split(Channel::Moe, rows, input, core, lane.rows("moe", rows)?, peer)
    }

    /// `muse_tp_combine_*` for one landed wave into `combined`: the layer's
    /// post_expert_norm and the step's routing weights over every holder's
    /// partials, read in place through the wave's `rank_regions` tables.
    /// `tables` is the fixed device copy of [`CombineTables::image`], written
    /// on this stream before the launch (graph-capturable: contents change,
    /// addresses do not). Empty regions are skipped (`combine_plan`).
    ///
    /// [`CombineTables::image`]: super::experts::CombineTables::image
    pub(crate) fn combine(&self, layer: usize, rows: usize, ranks: usize, tables: Buffer,
        layout: super::experts::CombineTables,
        plan: &cuteafd_loader::families::muse_spark::replies::ReplyPlan) -> Result<()> {
        ensure!((1..=super::experts::MAX_RANKS).contains(&ranks) && tables.bytes >= layout.bytes()
            && plan.bf16_rows.len() + plan.wide_rows.len() == rows, "Muse combine geometry");
        let norm = self.weights(layer)?.buffer("moe.post_expert_norm")?;
        let (weights, out) = (self.lane.rows("route_weights", rows)?, self.lane.rows("combined", rows)?);
        for (stem, table, map, region) in super::experts::combine_plan(layout, plan) {
            let at = |offset: usize, bytes: usize| Buffer { ptr: tables.ptr.cast::<u8>().wrapping_add(offset).cast(),
                bytes, ..tables };
            // SAFETY: as `project`; the tables were uploaded on this stream before
            // this launch and every rank region they name is a landed, unconsumed plane.
            unsafe {
                self.programs.launch(stem, &[at(table, ranks * 8), at(map, region * 4), weights, norm, out],
                    &[Scalar::I32(i32::try_from(ranks)?), rows_i32(region)?, rows_i32(rows)?], self.stream)?;
            }
        }
        Ok(())
    }

    /// One decoder layer, in `block::forward_layer` order, on `hidden` (in place):
    /// [`Self::layer_front`] then [`Self::layer_back`] around the Spark wave.
    /// `experts`: the Spark path on rank 0 (absent on rank 1 of a TP2 pair).
    pub(crate) fn layer(&self, layer: usize, step: &StepPlanes, mut experts: Option<&mut (dyn SparkExperts + '_)>,
        mut peer: Option<&mut (dyn PeerExchange + '_)>) -> Result<()> {
        let wave = self.layer_front(layer, step, experts.as_mut().map(|e| &mut **e as &mut dyn SparkExperts),
            peer.as_mut().map(|p| &mut **p as &mut dyn PeerExchange))?;
        self.layer_back(layer, step.rows, wave, experts, peer)
    }

    /// The layer up to the Spark wave: input norm, attention, the gated
    /// attention residual (into `residual`), the pre-feedforward norm, the MoE
    /// frontend (split on a pair), the wire and the send (rank 0). Everything
    /// the back half reads is a span plane (`residual`, `hidden`, the routes,
    /// the wire) or written by the land: other units' hooks may run between.
    pub(crate) fn layer_front(&self, layer: usize, step: &StepPlanes, experts: Option<&mut (dyn SparkExperts + '_)>,
        mut peer: Option<&mut (dyn PeerExchange + '_)>) -> Result<Option<super::experts::Wave>> {
        let (rows, lane) = (step.rows, self.lane);
        let (hidden, normed, residual) = (lane.rows("hidden", rows)?, lane.rows("normed", rows)?, lane.rows("residual", rows)?);
        self.norm(layer, "input_norm", Some("norm.input"), rows, hidden, normed)?;
        self.attention(layer, step, peer.as_mut().map(|p| &mut **p as &mut dyn PeerExchange))?;
        let attended = lane.rows("attended", rows)?;
        self.norm(layer, "post_attn_norm", None, rows, attended, normed)?;
        self.residual_gate(layer, "attention", rows, hidden, normed, residual)?;
        self.norm(layer, "pre_ffn_norm", Some("norm.pre_ffn"), rows, residual, normed)?;
        match (self.ranks, self.rank, experts) {
            (1, 0, Some(experts)) => {
                self.moe_frontend(layer, rows)?;
                self.expert_wire(rows)?;
                Ok(Some(experts.send(layer, &step.wave, self.wave_sources(rows)?)?))
            }
            (2, rank, experts) if (rank == 0) == experts.is_some() => {
                self.moe_frontend_split(layer, rows, peer.context("Muse TP2 needs the peer exchange")?)?;
                match experts {
                    Some(experts) => {
                        self.expert_wire(rows)?;
                        Ok(Some(experts.send(layer, &step.wave, self.wave_sources(rows)?)?))
                    }
                    None => Ok(None),
                }
            }
            _ => anyhow::bail!("Muse rank {} of {} with the wrong expert path", self.rank, self.ranks),
        }
    }

    /// An onboarded layer (its experts on the RTXs), whole within one hook: the
    /// front half up to the frontend (halves swapped on a pair), the wire, this
    /// GPU's partials (`Exl3Tp2::enqueue_partials`, all rows BF16), the
    /// partials swap on a pair, the combine with ranks 2 (own, peer in rank
    /// order) or 1, then the back half.
    pub(crate) fn layer_local(&self, layer: usize, step: &StepPlanes, onboarded: &mut super::onboard::OnboardedExperts<'_>,
        mut peer: Option<&mut (dyn PeerExchange + '_)>) -> Result<()> {
        let (rows, lane) = (step.rows, self.lane);
        ensure!(onboarded.holds(layer), "Muse layer {layer} is not onboarded on rank {}", self.rank);
        let (hidden, normed, residual) = (lane.rows("hidden", rows)?, lane.rows("normed", rows)?, lane.rows("residual", rows)?);
        self.norm(layer, "input_norm", Some("norm.input"), rows, hidden, normed)?;
        self.attention(layer, step, peer.as_mut().map(|p| &mut **p as &mut dyn PeerExchange))?;
        let attended = lane.rows("attended", rows)?;
        self.norm(layer, "post_attn_norm", None, rows, attended, normed)?;
        self.residual_gate(layer, "attention", rows, hidden, normed, residual)?;
        self.norm(layer, "pre_ffn_norm", Some("norm.pre_ffn"), rows, residual, normed)?;
        match peer.as_mut() {
            Some(peer) => self.moe_frontend_halves(layer, rows, &mut **peer, true)?,
            None => self.moe_frontend(layer, rows)?,
        }
        self.expert_wire(rows)?;
        let [ids, weights, wire] = self.wave_sources(rows)?;
        // All rows BF16: the epilogue's BF16 row map is the identity prefix 0..rows, so its
        // per-launch rewrite of the pinned map staging writes the bytes an in-flight copy
        // already reads (no drain between layers is needed), and the combine's cached
        // full-capacity tables (`tables_for`) serve every wave.
        let plan = cuteafd_transport::PartialReplyPlan { wide: cuteafd_transport::PartialWire::Nvfp4, regions: vec![0; rows] };
        let own = onboarded.own_rows(self.cfg, rows);
        // SAFETY: the wire, routes and own plane are this GPU's, written on this stream before;
        // the executor's sources are read by its epilogue on the same stream.
        unsafe {
            onboarded.executor.enqueue_partials(layer, rows, wire.ptr,
                crate::shared::experts::rtx::Routes { ids: ids.ptr, weights: weights.ptr },
                &mut onboarded.epilogue, &plan, own, self.stream)?;
        }
        let planes: Vec<u64> = match peer.as_mut() {
            Some(peer) => {
                let received = peer.swap(Channel::Partials, own)?;
                let (rank0, rank1) = if self.rank == 0 { (own, received) } else { (received, own) };
                vec![rank0.ptr as u64, rank1.ptr as u64]
            }
            None => vec![own.ptr as u64],
        };
        onboarded.tables_for(self.library, &planes, self.stream)?;
        let reply = cuteafd_loader::families::muse_spark::replies::ReplyPlan { bf16_rows: (0..rows as u32).collect(),
            wide_rows: Vec::new(), wide: cuteafd_loader::families::muse_spark::replies::PartialFormat::Nvfp4 };
        self.combine(layer, rows, planes.len(), onboarded.tables.buffer, onboarded.layout, &reply)?;
        match peer {
            Some(peer) => self.moe_backend_halves(layer, rows, peer, true)?,
            None => self.moe_backend(layer, rows)?,
        }
        let moe = lane.rows("moe", rows)?;
        self.norm(layer, "post_ffn_norm", Some("norm.post_ffn"), rows, moe, normed)?;
        self.residual_gate(layer, "ffn", rows, residual, normed, hidden)
    }

    /// The wave's staging sources from this lane (the unit's slot views): route
    /// ids, route weights and the FP8 K32 wire.
    fn wave_sources(&self, rows: usize) -> Result<[Buffer; 3]> {
        Ok([self.lane.rows("route_ids", rows)?, self.lane.rows("route_weights", rows)?, self.lane.rows("expert_wire", rows)?])
    }

    /// The layer after the Spark wave: land and combine (rank 0), the MoE
    /// backend (K-split on a pair), the post-feedforward norm and the gated
    /// residual back into `hidden`.
    pub(crate) fn layer_back(&self, layer: usize, rows: usize, wave: Option<super::experts::Wave>,
        experts: Option<&mut (dyn SparkExperts + '_)>, peer: Option<&mut (dyn PeerExchange + '_)>) -> Result<()> {
        let lane = self.lane;
        if let (Some(wave), Some(experts)) = (wave, experts) {
            let landed = experts.land(wave)?;
            self.combine(layer, rows, landed.ranks, landed.tables, landed.layout, &landed.plan)?;
            experts.consumed(wave)?;
        } else {
            ensure!(self.rank == 1 && wave.is_none(), "Muse rank {} lands without its wave", self.rank);
        }
        if self.ranks == 1 { self.moe_backend(layer, rows)?; }
        else { self.moe_backend_split(layer, rows, peer.context("Muse TP2 needs the peer exchange")?)?; }
        let (moe, normed) = (lane.rows("moe", rows)?, lane.rows("normed", rows)?);
        let (residual, hidden) = (lane.rows("residual", rows)?, lane.rows("hidden", rows)?);
        self.norm(layer, "post_ffn_norm", Some("norm.post_ffn"), rows, moe, normed)?;
        self.residual_gate(layer, "ffn", rows, residual, normed, hidden)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cuteafd_loader::families::muse_spark::lane::LanePlanes;

    /// An onboarded layer runs the Spark layer's front half, then on the RTXs
    /// the wire, this GPU's partials, the partials swap, the combine and the
    /// backend with exactly the exchanges the TP2 slot model counts for a
    /// `Local` step (`peer::hook_exchanges`): ExpertInput, Partials, Moe swaps,
    /// never the one-way Spark-layer channels.
    #[test]
    fn onboarded_layer_order_and_exchanges() {
        let source = include_str!("device.rs").split("#[cfg(test)]").next().unwrap();
        let body = |name: &str| source.split(&format!("fn {name}(")).nth(1).unwrap()
            .split("\n    pub(crate) fn ").next().unwrap().split("\n    fn ").next().unwrap().to_string();
        let local = body("layer_local");
        let order = ["\"input_norm\"", "self.attention(", "\"pre_ffn_norm\"", "moe_frontend_halves(layer, rows, &mut **peer, true)",
            "self.expert_wire(rows)", ".enqueue_partials(", "swap(Channel::Partials, own)", "tables_for(", "self.combine(",
            "moe_backend_halves(layer, rows, peer, true)", "\"post_ffn_norm\"", "residual_gate(layer, \"ffn\""];
        let at: Vec<usize> = order.iter().map(|needle| local.find(needle).unwrap_or_else(|| panic!("{needle}"))).collect();
        assert!(at.windows(2).all(|pair| pair[0] < pair[1]), "{order:?}");
        assert!(local.contains("vec![rank0.ptr as u64, rank1.ptr as u64]") && local.contains("regions: vec![0; rows]"));
        let front = body("moe_frontend_halves");
        let swapped = front.split("if both {").nth(1).unwrap().split("} else {").next().unwrap();
        assert!(swapped.contains("peer.swap(Channel::ExpertInput, half)"));
        let back = body("moe_backend_halves");
        let local_back = back.split("if both {").nth(1).unwrap().split("} else if self.rank == 0 {").next().unwrap();
        assert!(!local_back.contains("Channel::Combined") && local_back.contains("self.rank * half_bytes"));
        assert!(back.contains("self.k_split(Channel::Moe,"));
    }

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
                ("expert_wire", e + e / 32), ("token_ids", 4),
            ] {
                assert!(row(plane) >= needed, "rank layout {ranks}: plane {plane} {} < {needed}", row(plane));
            }
            if ranks == 2 {
                for (plane, needed) in [("o_core", h * 2), ("o_reduced", h * 4), ("expert_half", e)] {
                    assert!(row(plane) >= needed, "TP2 plane {plane} {} < {needed}", row(plane));
                }
            }
            for (plane, needed) in [("hidden", h * 2)] {
                assert!(row(plane) >= needed, "rank layout {ranks}: plane {plane} {} < {needed}", row(plane));
            }
            // The 1 KiB attention scratch borrows the router plane at a single row.
            assert!(planes.planes["router_logits"] >= 1024 || planes.rows == 1);
        }
        Ok(())
    }
}

#[cfg(test)]
mod weight_tests {
    /// Every layer operand the device layer reads by name is one the admitted
    /// layout packs (plain operand or K6 core), for one GPU and a pair; the
    /// release publication's gate_proj is BF16, everything projected else K6.
    #[test]
    fn device_reads_only_packed_layer_operands() -> anyhow::Result<()> {
        let dir = tempfile::tempdir()?;
        cuteafd_loader::plan::testing::write_muse_spark_snapshot(dir.path(), &[]);
        let checkpoint = cuteafd_loader::plan::Checkpoint::inventory(dir.path())?;
        let cfg = cuteafd_loader::families::muse_spark::MuseSparkConfig::from_hf(&checkpoint.config)?;
        let core = cuteafd_loader::families::muse_spark::core::CoreInventory::from_checkpoint(&cfg, &checkpoint)?;
        let source = include_str!("device.rs").split("#[cfg(test)]").next().unwrap();
        let read = ["moe.router", "moe.bias", "moe.pre_expert_norm", "moe.post_expert_norm", "norm.input", "norm.pre_ffn",
            "norm.post_ffn", "gate.attention.alpha", "gate.attention.beta", "gate.ffn.alpha", "gate.ffn.beta"];
        let projected = ["attn.q", "attn.k", "attn.v", "attn.gate", "attn.o", "moe.pre_expert", "moe.post_expert"];
        for name in read.iter().chain(&projected) {
            let stem = name.split('.').last().unwrap();
            assert!(source.contains(&format!("\"{name}\"")) || source.contains(stem), "{name} unused by the device");
        }
        for ranks in [1, 2] {
            for layer in [0, 1, 61] {
                for layout in cuteafd_loader::families::muse_spark::weight_layout::LayerLayout::ranks(&cfg, layer, ranks, &core)? {
                    for name in read { assert!(layout.operands.contains_key(name), "{name} not packed (ranks {ranks})"); }
                    for name in projected {
                        assert!(layout.core.contains_key(name) || (name == "attn.gate" && layout.operands.contains_key(name)),
                            "{name} not packed (ranks {ranks})");
                    }
                }
            }
        }
        Ok(())
    }
}
