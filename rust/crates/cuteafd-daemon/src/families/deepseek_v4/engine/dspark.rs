//! dSpark drafter (`mtp.*`): three window-only blocks that propose
//! `dspark_block_size` tokens after a sequence's next token.
//!
//! Every main step taps the target layers' outputs (mean over the mHC
//! copies), projects them (`main_norm(main_proj(taps))`) and writes each
//! stage's main KV at those rows' positions; rows a verify later rejects are
//! overwritten when their positions are processed again. A draft step embeds
//! `[token, noise...]` at the next positions and runs the stages as decode
//! rows whose window holds the latest main positions and the whole draft block
//! (the reference attends to 128 main entries plus the block; here the block
//! displaces the oldest main entries so the window stays 128 wide, which only
//! affects draft quality, never the verified output). The last stage's
//! mHC head, the shared vocabulary head and the Markov head then pick the
//! drafts greedily on the device.
use super::{Dev, Engine, Lane, Workspace, LOCAL_EXPERTS, TAP_ROWS};
use crate::families::deepseek_v4::local::LocalLayer;
use crate::families::deepseek_v4::metadata::WINDOW;
use crate::families::deepseek_v4::pool::Placement;
use anyhow::{ensure, Context, Result};
use cuteafd_ffi::programs::Scalar;
use std::ffi::c_void;

/// One sequence to draft for: `token` is its next token, at `placement.len`.
pub(crate) struct DraftRequest<'p> {
    pub placement: &'p Placement,
    pub token: u32,
}

fn offset(dev: &Dev<'_>, bytes: usize) -> *mut c_void {
    // SAFETY: callers stay inside the allocation (row offsets below its rows).
    unsafe { dev.buffer.ptr.cast::<u8>().add(bytes) }.cast()
}

impl<'a> Engine<'a> {
    /// Tokens a draft step proposes per sequence (0 unless the drafter and
    /// its stage experts are resident).
    pub fn draft_block(&self) -> usize {
        if self.draft_ready() { self.cfg.dspark_block_size } else { 0 }
    }

    /// Stages whose experts are resident here (drafting needs all of them).
    fn draft_ready(&self) -> bool {
        let stages = self.weights.dspark.as_ref().map_or(0, |d| d.stages.len());
        stages > 0 && self.local.borrow().as_ref().is_some_and(|l| l.stages() == stages)
    }

    /// Records `layer`'s output for the lane's last rows when it is a target.
    pub(super) fn tap(&self, _w: &Workspace<'_>, lane: &Lane<'_>, layer: usize, t: usize) -> Result<()> {
        if !self.draft_ready() {
            return Ok(());
        }
        let Some(index) = self.cfg.dspark_target_layer_ids.iter().position(|&l| l == layer) else {
            return Ok(());
        };
        let h = self.cfg.dim;
        let n = t.min(TAP_ROWS);
        let taps = self.cfg.dspark_target_layer_ids.len();
        // SAFETY: the lane stream holds `t` rows [t, 4, h] and the tap buffer
        // TAP_ROWS rows of `taps * h`; the stream orders the launch.
        unsafe {
            self.library.dsv4_hc_mean(offset(&lane.stream_a, (t - n) * 4 * h * 2), lane.taps.buffer.ptr, n, h,
                taps * h, index * h, self.stream)
        }
    }

    /// Writes every stage's main KV for the lane's tapped rows.
    pub(super) fn write_draft_kv(&self, w: &Workspace<'_>, lane: &Lane<'_>, t: usize, cap: usize) -> Result<()> {
        let Some(dspark) = self.weights.dspark.as_ref().filter(|_| self.draft_ready()) else { return Ok(()) };
        let (h, n) = (self.cfg.dim, t.min(TAP_ROWS));
        let k = self.cfg.dspark_target_layer_ids.len() * h;
        // SAFETY: taps hold `n` rows of `k`, main_x/main_work `n` rows of `h`.
        unsafe {
            self.library.dsv4_fp8_linear_rmsnorm(lane.taps.buffer.ptr, dspark.main_proj.buffer.ptr,
                dspark.main_proj_scale.buffer.ptr, dspark.main_norm.buffer.ptr, w.main_x.buffer.ptr,
                w.main_work.buffer.ptr, n, h, k, self.cfg.norm_eps as f32, self.stream)?;
        }
        let first = t - n;
        for (stage, weights) in dspark.stages.iter().enumerate() {
            let cache = &self.pools[self.cfg.n_layers + stage];
            self.run(&format!("producer_m{cap}"), &[
                ("hidden", w.main_x.buffer.ptr), ("positions", offset(&lane.tables.positions, first * 8)),
                ("main_slots", offset(&lane.tables.main_slots, first * 8)), ("cos_sin", self.rope_window.buffer.ptr),
                ("w_qkv", weights.ptr("w_qkv")?), ("w_qkv_scale", weights.ptr("w_qkv_scale")?),
                ("w_q", weights.ptr("w_q")?), ("w_q_scale", weights.ptr("w_q_scale")?), ("q_norm", weights.ptr("q_norm")?),
                ("kv_norm", weights.ptr("kv_norm")?), ("main_kv_cache", cache.main.buffer.ptr),
                ("query", w.query.buffer.ptr), ("q_rank", w.q_rank.buffer.ptr), ("scratch", w.scratch.buffer.ptr),
            ], &[Scalar::I32(n as i32)])?;
        }
        Ok(())
    }

    /// Proposes `draft_block()` tokens after each request's next token.
    /// `inputs` holds the block's input tokens per request: the token, then
    /// the noise token (embedded on the device).
    pub fn draft(&self, requests: &[DraftRequest<'_>], inputs: &[u32]) -> Result<Vec<Vec<u32>>> {
        let dspark = self.weights.dspark.as_ref().context("the checkpoint has no dSpark drafter")?;
        ensure!(self.draft_ready(), "dSpark stage experts are not resident on the coordinator");
        let (h, block) = (self.cfg.dim, self.cfg.dspark_block_size);
        let rows = requests.len() * block;
        ensure!(!requests.is_empty() && rows <= self.decode_rows && inputs.len() == rows,
            "draft step of {} requests", requests.len());
        let slot = &self.decode_workspace;
        if slot.borrow().is_none() {
            *slot.borrow_mut() = Some(self.workspace(self.decode_rows, 1)?);
        }
        let workspace = slot.borrow();
        let w = workspace.as_ref().context("workspace")?;
        // Target and draft steps share one stream and the workspace's query,
        // scratch and logits buffers. Reuse the first lane too: target-only
        // decoding may have initialized this workspace with just one lane.
        let lane = w.lanes.first().context("dSpark decode workspace has no lane")?;
        let (mut positions, mut slots) = (Vec::with_capacity(rows), Vec::with_capacity(rows));
        let (mut indices, mut lengths) = (vec![-1i32; rows * WINDOW], Vec::with_capacity(rows));
        for (index, request) in requests.iter().enumerate() {
            let (placement, next) = (request.placement, request.placement.len);
            // Latest main positions, then the whole block (non-causal).
            let main = next.min(WINDOW - block);
            let visible: Vec<i32> = (next - main..next + block)
                .map(|p| placement.window_slot(&self.shape, p) as i32).collect();
            for j in 0..block {
                let row = index * block + j;
                positions.push((next + j) as i64);
                slots.push(placement.window_slot(&self.shape, next + j));
                indices[row * WINDOW..][..visible.len()].copy_from_slice(&visible);
                lengths.push(visible.len() as i32);
            }
        }
        let bytes = |values: &[u8], dev: &Dev<'_>| -> Result<()> {
            self.library.copy_h2d(cuteafd_ffi::CuteafdDeviceBuffer { bytes: values.len(), ..dev.buffer }, values)
        };
        let m = &lane.tables;
        bytes(super::bytes_of(&positions), &m.positions)?;
        bytes(super::bytes_of(&slots), &m.main_slots)?;
        bytes(super::bytes_of(&indices), &m.swa_indices)?;
        bytes(super::bytes_of(&lengths), &m.swa_lengths)?;
        let first: Vec<u32> = requests.iter().map(|r| r.token).collect();
        bytes(super::bytes_of(&first), &w.first_tokens)?;
        // The mHC streams start as four copies of each input's embedding.
        self.embedding.embed(inputs, w.draft_ids.buffer, 4, lane.stream_a.buffer, self.stream)?;
        let cap = self.decode_rows;
        let scalar = Scalar::I32(rows as i32);
        for (stage, weights) in dspark.stages.iter().enumerate() {
            let layer = self.cfg.n_layers + stage;
            let cache = &self.pools[layer];
            let (a, b) = (&lane.stream_a, &lane.stream_b);
            self.run("mhc_pre", &[
                ("residual", a.buffer.ptr), ("fn", weights.ptr("attn.fn")?), ("scale", weights.ptr("attn.scale")?),
                ("base", weights.ptr("attn.base")?), ("norm", weights.ptr("attn.norm")?), ("post", lane.post.buffer.ptr),
                ("comb", lane.comb.buffer.ptr), ("y", w.y.buffer.ptr), ("scratch", w.scratch.buffer.ptr),
            ], &[scalar])?;
            self.run(&format!("producer_m{cap}"), &[
                ("hidden", w.y.buffer.ptr), ("positions", m.positions.buffer.ptr), ("main_slots", m.main_slots.buffer.ptr),
                ("cos_sin", self.rope_window.buffer.ptr), ("w_qkv", weights.ptr("w_qkv")?),
                ("w_qkv_scale", weights.ptr("w_qkv_scale")?), ("w_q", weights.ptr("w_q")?),
                ("w_q_scale", weights.ptr("w_q_scale")?), ("q_norm", weights.ptr("q_norm")?),
                ("kv_norm", weights.ptr("kv_norm")?), ("main_kv_cache", cache.main.buffer.ptr),
                ("query", w.query.buffer.ptr), ("q_rank", w.q_rank.buffer.ptr), ("scratch", w.scratch.buffer.ptr),
            ], &[scalar])?;
            let dummy = w.dummy.buffer.ptr;
            self.run(&format!("sparse_mla_decode_win_m{cap}"), &[
                ("q", w.query.buffer.ptr), ("swa_cache", cache.main.buffer.ptr), ("swa_indices", m.swa_indices.buffer.ptr),
                ("swa_lengths", m.swa_lengths.buffer.ptr), ("indexed_cache", dummy), ("indexed_indices", dummy),
                ("indexed_lengths", dummy), ("attn_sink", weights.ptr("attn_sink")?), ("out", w.attn_out.buffer.ptr),
                ("scratch", w.scratch.buffer.ptr),
            ], &[scalar])?;
            self.run(&format!("wo_m{cap}"), &[
                ("o", w.attn_out.buffer.ptr), ("positions", m.positions.buffer.ptr),
                ("cos_sin", self.rope_window.buffer.ptr), ("wo_a", weights.ptr("wo_a")?),
                ("wo_a_scale", weights.ptr("wo_a_scale")?), ("wo_b", weights.ptr("wo_b")?),
                ("wo_b_scale", weights.ptr("wo_b_scale")?), ("out", w.delta.buffer.ptr), ("scratch", w.scratch.buffer.ptr),
            ], &[scalar])?;
            self.run(&format!("mhc_post_pre_m{cap}"), &[
                ("x", w.delta.buffer.ptr), ("residual", a.buffer.ptr), ("prev_post", lane.post.buffer.ptr),
                ("prev_comb", lane.comb.buffer.ptr), ("fn", weights.ptr("ffn.fn")?), ("scale", weights.ptr("ffn.scale")?),
                ("base", weights.ptr("ffn.base")?), ("norm", weights.ptr("ffn.norm")?), ("residual_out", b.buffer.ptr),
                ("post", lane.post.buffer.ptr), ("comb", lane.comb.buffer.ptr), ("y", w.y.buffer.ptr),
                ("scratch", w.scratch.buffer.ptr),
            ], &[scalar])?;
            self.run("router_scores", &[
                ("x", w.y.buffer.ptr), ("w", weights.ptr("gate")?), ("logits", w.logits.buffer.ptr),
            ], &[scalar])?;
            // SAFETY: logits, bias and route outputs are live buffers of `rows` rows.
            unsafe {
                self.library.dsv4_router_select(w.logits.buffer.ptr, weights.ptr("gate.bias")?, std::ptr::null_mut(),
                    std::ptr::null_mut(), w.route_ids.buffer.ptr, w.route_weights.buffer.ptr, rows,
                    self.cfg.n_routed_experts, self.cfg.n_activated_experts, self.cfg.route_scale as f32, self.stream)?;
            }
            let grid = self.quantize_grid.blocks(rows, h);
            self.run("expert_input_quant", &[
                ("source_ptr", w.y.buffer.ptr), ("values_ptr", w.wire.buffer.ptr),
                ("scale_rows_ptr", offset(&w.wire, h)), ("scale_mma_ptr", w.dummy.buffer.ptr),
            ], &[scalar, Scalar::I32(grid as i32)])?;
            self.shared_ffn(layer, w, lane, scalar, cap, weights)?;
            {
                let mut local = self.local.borrow_mut();
                let local = local.as_mut().context("local experts")?;
                // SAFETY: wire, routes and shared rows are complete in stream order.
                unsafe {
                    local.run(LocalLayer::Stage(stage), rows, w.wire.buffer.ptr, w.route_ids.buffer.ptr,
                        w.route_weights.buffer.ptr, lane.shared.buffer.ptr, self.stream)?;
                }
            }
            self.post(w, lane, LOCAL_EXPERTS, [std::ptr::null(); crate::shared::spark_intake::MAX_INTAKE_RANKS], scalar, layer)?;
        }
        self.run("mhc_head", &[
            ("residual", lane.stream_a.buffer.ptr), ("fn", dspark.head_fn.buffer.ptr),
            ("scale", dspark.head_scale.buffer.ptr), ("base", dspark.head_base.buffer.ptr),
            ("norm", dspark.norm.buffer.ptr), ("collapsed", w.delta.buffer.ptr), ("out", w.y.buffer.ptr),
        ], &[scalar])?;
        let rank = self.cfg.dspark_markov_rank;
        // SAFETY: y, head weights, logits, Markov weights, tokens and the
        // argmax workspace are live buffers of these shapes.
        unsafe {
            w.head.as_ref().context("LM head")?.launch(w.y.buffer.ptr.cast(), self.weights.head.buffer.ptr.cast(),
                w.vocab_logits.buffer.ptr.cast(), rows as u32, self.stream)?;
            self.library.dsv4_markov_drafts(w.vocab_logits.buffer.ptr, dspark.markov_w1.buffer.ptr,
                dspark.markov_w2.buffer.ptr, w.first_tokens.buffer.ptr, w.drafts.buffer.ptr, w.markov.buffer.ptr,
                requests.len(), block, self.cfg.vocab_size, rank, self.stream)?;
        }
        let drafts = self.download(&w.drafts, rows * 4)?;
        Ok(drafts.chunks_exact(block * 4)
            .map(|d| d.chunks_exact(4).map(|b| u32::from_le_bytes(b.try_into().unwrap())).collect())
            .collect())
    }
}
