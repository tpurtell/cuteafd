//! Dense block math over admitted buffers and the shared Qwen recurrent bank.
//! No allocation, host copies or private cache lifecycle occurs in a block.
use super::{programs::DensePrograms, weights::DenseLayer};
use crate::shared::qwen::{GdnBank, LayerStateMap, REPLAY_ROWS};
use anyhow::{ensure, Result};
use cuteafd_ffi::{programs::{Programs, Scalar}, CuteafdDeviceBuffer};
use cuteafd_loader::families::{qwen::HybridAttention, qwen3_5::Qwen35Config};
use std::ffi::c_void;

#[derive(Clone, Copy)]
pub(crate) struct Planes {
    pub(crate) residual: CuteafdDeviceBuffer,
    pub(crate) norm: CuteafdDeviceBuffer,
    pub(crate) delta: CuteafdDeviceBuffer,
    pub(crate) query: CuteafdDeviceBuffer,
    pub(crate) gate: CuteafdDeviceBuffer,
    pub(crate) attention: CuteafdDeviceBuffer,
    pub(crate) scratch: CuteafdDeviceBuffer,
    pub(crate) slots: CuteafdDeviceBuffer,
    pub(crate) seq_first: CuteafdDeviceBuffer,
    pub(crate) rope_positions: CuteafdDeviceBuffer,
    pub(crate) kv_slots: CuteafdDeviceBuffer,
    pub(crate) positions: CuteafdDeviceBuffer,
    pub(crate) page_table: CuteafdDeviceBuffer,
}

pub(crate) struct Step {
    pub(crate) rows: usize,
    pub(crate) capacity: usize,
    pub(crate) table_stride: usize,
    pub(crate) splits: usize,
    pub(crate) speculative: bool,
}

impl Step {
    pub(crate) fn validate(&self) -> Result<()> {
        ensure!(matches!(self.capacity, 64 | 4096) && self.rows > 0 && self.rows <= self.capacity,
            "dense step rows exceed native export capacity");
        ensure!(!self.speculative || self.capacity == REPLAY_ROWS, "dense verify requires replay export");
        ensure!((1..=32).contains(&self.splits), "dense attention split count outside export");
        i32::try_from(self.rows)?;
        i32::try_from(self.table_stride)?;
        Ok(())
    }
}

pub(crate) struct Blocks<'p, 'a> {
    pub(crate) programs: &'p Programs<'a>,
    pub(crate) admitted: &'p DensePrograms,
    pub(crate) cfg: &'p Qwen35Config,
    pub(crate) homes: &'p LayerStateMap,
    pub(crate) bank: &'p GdnBank<'a>,
    pub(crate) stream: *mut c_void,
}

impl Blocks<'_, '_> {
    fn launch(&self, op: &str, capacity: Option<usize>, pointers: &[(&str, *mut c_void)], scalars: &[Scalar]) -> Result<()> {
        let prefix = self.admitted.prefix;
        let quantized = self.admitted.format.fp8() && matches!(op, "gdn" | "attn_producer" | "attn_o" | "ffn");
        let suffix = if quantized { "_fp8" } else { "" };
        let name = match capacity { Some(rows) => format!("{prefix}_{op}{suffix}_m{rows}"), None => format!("{prefix}_{op}") };
        let names: Vec<_> = pointers.iter().map(|&(name, _)| name).collect();
        let values: Vec<_> = pointers.iter().map(|&(_, ptr)| ptr).collect();
        let program = self.programs.program(&name, &names)?;
        // SAFETY: run validates plane extents; operands and bank remain owned by the enclosing engine.
        unsafe { program.launch(&values, scalars, self.stream) }
    }

    /// Enqueue one post-norm residual block. The caller retains all supplied
    /// planes, weights, page records and tables until the stream drains.
    pub(crate) fn run(&self, index: usize, layer: &DenseLayer<'_>, p: Planes,
        kv: Option<CuteafdDeviceBuffer>, step: &Step) -> Result<()> {
        step.validate()?;
        ensure!(self.cfg.attention.get(index) == Some(&layer.attention), "dense layer/checkpoint type mismatch");
        let home = self.homes.layers.get(index).ok_or_else(|| anyhow::anyhow!("dense layer has no state owner"))?;
        ensure!(home.owner == 0 && self.homes.gdn_layers.len() == 1, "single-device dense block cannot use peer state");
        let bytes = step.rows * self.cfg.hidden * 2;
        let query = step.rows * self.cfg.heads * self.cfg.head_dim * 2;
        for buffer in [p.residual, p.norm, p.delta] {
            ensure!(!buffer.ptr.is_null() && buffer.bytes >= bytes, "dense hidden plane too small");
        }
        for buffer in [p.query, p.gate, p.attention] {
            ensure!(!buffer.ptr.is_null() && buffer.bytes >= query, "dense attention plane too small");
        }
        ensure!(p.scratch.bytes >= self.admitted.scratch_bytes, "dense scratch exceeds admission");
        for (buffer, width) in [(p.slots, 4), (p.seq_first, 4), (p.rope_positions, 12),
            (p.kv_slots, 8), (p.positions, 8)] {
            ensure!(!buffer.ptr.is_null() && buffer.bytes >= step.rows * width, "dense row table too small");
        }
        let kv = match layer.attention {
            HybridAttention::Gdn => {
                ensure!(home.gdn_ordinal.is_some(), "GDN layer has no owner-local ordinal");
                None
            }
            HybridAttention::Full => {
                let kv = kv.ok_or_else(|| anyhow::anyhow!("full GQA lacks admitted records"))?;
                ensure!(!kv.ptr.is_null() && kv.bytes >= 64 * 2 * self.cfg.kv_heads * self.cfg.head_dim * 2,
                    "full GQA record pool too small");
                let tables = if step.table_stride == 0 { 1 } else { step.rows };
                ensure!(!p.page_table.ptr.is_null() && p.page_table.bytes >= tables * step.table_stride.max(1) * 4,
                    "dense attention page table too small");
                Some(kv)
            }
        };
        let rows = [Scalar::I32(i32::try_from(step.rows)?)];
        self.launch("norm", None, &[("x", p.residual.ptr), ("weight", layer.ptr("attn.norm")?),
            ("out", p.norm.ptr)], &rows)?;
        match layer.attention {
            HybridAttention::Gdn => {
                let [conv, state, replay] = self.bank.pools(home.gdn_ordinal.unwrap())?;
                let mut pointers = vec![("x", p.norm.ptr)];
                if self.admitted.format.fp8() {
                    for name in ["w_qkvz_fp8", "w_qkvz_scale", "w_ba"] { pointers.push((name, layer.ptr(name)?)); }
                } else { pointers.push(("w_in", layer.ptr("w_in")?)); }
                for name in ["conv_w", "a_log", "dt_bias", "norm_w"] { pointers.push((name, layer.ptr(name)?)); }
                if self.admitted.format.fp8() {
                    for name in ["w_out_fp8", "w_out_scale"] { pointers.push((name, layer.ptr(name)?)); }
                } else { pointers.push(("w_out", layer.ptr("w_out")?)); }
                pointers.extend([("conv_state", conv), ("state", state), ("slots", p.slots.ptr),
                    ("seq_first", p.seq_first.ptr), ("out", p.delta.ptr)]);
                let mut scalars = rows.to_vec();
                if step.capacity == REPLAY_ROWS {
                    pointers.push(("replay", replay));
                    scalars.push(Scalar::I32(i32::from(step.speculative)));
                }
                pointers.push(("scratch", p.scratch.ptr));
                self.launch("gdn", Some(step.capacity), &pointers, &scalars)?;
            }
            HybridAttention::Full => {
                let kv = kv.unwrap();
                let mut pointers = vec![("x", p.norm.ptr)];
                if self.admitted.format.fp8() {
                    for name in ["w_in_fp8", "w_in_scale"] { pointers.push((name, layer.ptr(name)?)); }
                } else { pointers.push(("w_in", layer.ptr("w_in")?)); }
                pointers.extend([("q_norm", layer.ptr("q_norm")?), ("k_norm", layer.ptr("k_norm")?),
                    ("rope_positions", p.rope_positions.ptr), ("kv_slots", p.kv_slots.ptr), ("kv_cache", kv.ptr),
                    ("query", p.query.ptr), ("gate", p.gate.ptr), ("scratch", p.scratch.ptr)]);
                self.launch("attn_producer", Some(step.capacity), &pointers, &rows)?;
                self.launch("full_gqa", Some(step.capacity), &[("query", p.query.ptr), ("kv_cache", kv.ptr),
                    ("positions", p.positions.ptr), ("page_table", p.page_table.ptr),
                    ("out", p.attention.ptr), ("scratch", p.scratch.ptr)],
                    &[rows[0], Scalar::I32(i32::try_from(step.table_stride)?), Scalar::I32(i32::try_from(step.splits)?)])?;
                let mut pointers = vec![("attn", p.attention.ptr), ("gate", p.gate.ptr)];
                if self.admitted.format.fp8() {
                    for name in ["w_o_fp8", "w_o_scale"] { pointers.push((name, layer.ptr(name)?)); }
                } else { pointers.push(("w_o", layer.ptr("w_o")?)); }
                pointers.extend([("out", p.delta.ptr), ("scratch", p.scratch.ptr)]);
                let mut scalars = rows.to_vec();
                if self.admitted.format.fp8() && step.capacity > REPLAY_ROWS { scalars.push(Scalar::I32(255)); }
                self.launch("attn_o", Some(step.capacity), &pointers, &scalars)?;
            }
        }
        self.launch("add", None, &[("a", p.residual.ptr), ("b", p.delta.ptr), ("out", p.residual.ptr)], &rows)?;
        self.launch("norm", None, &[("x", p.residual.ptr), ("weight", layer.ptr("ffn.norm")?), ("out", p.norm.ptr)], &rows)?;
        let mut pointers = vec![("x", p.norm.ptr)];
        if self.admitted.format.fp8() {
            for (name, operand) in [("w_gate_up_fp8", "ffn.w_gate_up_fp8"), ("w_gate_up_scale", "ffn.w_gate_up_scale"),
                ("w_down_fp8", "ffn.w_down_fp8"), ("w_down_scale", "ffn.w_down_scale")] {
                pointers.push((name, layer.ptr(operand)?));
            }
        } else {
            pointers.extend([("w_gate_up", layer.ptr("ffn.w_gate_up")?), ("w_down", layer.ptr("ffn.w_down")?)]);
        }
        pointers.extend([("out", p.delta.ptr), ("scratch", p.scratch.ptr)]);
        self.launch("ffn", Some(step.capacity), &pointers, &rows)?;
        self.launch("add", None, &[("a", p.residual.ptr), ("b", p.delta.ptr), ("out", p.residual.ptr)], &rows)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn replay_and_causal_attention_capacities_fail_before_launch() {
        for (rows, capacity, speculative, splits, valid) in [(64,64,true,1,true), (65,64,false,1,false),
            (4096,4096,false,32,true), (1,4096,true,1,false), (1,64,false,33,false), (0,64,false,1,false)] {
            assert_eq!(Step { rows, capacity, speculative, splits, table_stride: 4096 }.validate().is_ok(), valid);
        }
    }
}
