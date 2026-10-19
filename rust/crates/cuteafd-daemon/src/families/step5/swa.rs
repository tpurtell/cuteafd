//! Layer-selected SWA projections and transaction-owned cache reads.
use super::{
    ffn::{DenseFfn, DensePlan},
    rope::TargetTables,
    transaction::{AttentionRead, Step, StepInputs},
    weights::load_operand,
};
use crate::shared::memory::device::{Allocation, Device, Stream};
use anyhow::{ensure, Context, Result};
use cuteafd_core::DType;
use cuteafd_ffi::{
    programs::{Program, Programs, Scalar},
    CuteafdDeviceBuffer,
};
use cuteafd_loader::plan::{
    checkpoint::Checkpoint,
    families::step5::coordinator::{CoordinatorContract, RankWeightLayout},
};

const H: usize = 4096;
const Q: usize = 64 * 192;
const KV: usize = 4 * 192;
const X: usize = 0;
const NORM: usize = 1;
const QRAW: usize = 2;
const KRAW: usize = 3;
const VPAGES: usize = 4;
const QUERY: usize = 5;
const KPAGES: usize = 6;
const LOGITS: usize = 7;
const ATTN: usize = 8;
const GATED: usize = 9;
const DELTA: usize = 10;
const POSITIONS: usize = 11;
const REQUESTS: usize = 12;
const TABLE: usize = 13;
const SELECTED: usize = 14;
const POST: usize = 15;

struct Admission {
    rows: usize,
    pages: usize,
    planes: [usize; 16],
    bytes: u64,
}
impl Admission {
    fn new(
        rows: usize,
        weights: &[u64],
        dense: u64,
        rope: u64,
        available: u64,
        reservations: &[u64],
    ) -> Result<Self> {
        Self::for_heads(rows, 64, weights, dense, rope, available, reservations)
    }

    fn for_heads(
        rows: usize,
        heads: usize,
        weights: &[u64],
        dense: u64,
        rope: u64,
        available: u64,
        reservations: &[u64],
    ) -> Result<Self> {
        ensure!(matches!(heads, 32 | 64), "SWA native query head count");
        let q = heads * 192;
        let kv = heads / 16 * 192;
        ensure!(
            (1..=4096).contains(&rows),
            "SWA row capacity must be 1..=4096"
        );
        let pages = rows.div_ceil(64);
        let planes = [
            rows * H * 4,
            rows * H * 2,
            rows * q * 2,
            rows * kv * 2,
            pages * 64 * kv * 2,
            rows * q * 2,
            pages * 64 * kv * 2,
            rows * heads * 2,
            rows * q * 2,
            rows * q * 2,
            rows * H * if heads == 32 { 4 } else { 2 },
            rows * 8,
            rows * 4,
            pages * 4,
            rows * 512 * 4,
            rows * H * 4,
        ]
        .map(|n| n.max(256));
        let bytes = planes
            .iter()
            .map(|&n| n as u64)
            .chain(weights.iter().copied())
            .chain([dense, rope])
            .try_fold(0u64, |a, b| a.checked_add(b))
            .context("SWA admission overflow")?;
        let total = reservations
            .iter()
            .try_fold(bytes, |a, b| a.checked_add(*b))
            .context("SWA reservation overflow")?;
        ensure!(
            total <= available,
            "SWA layer needs {total} bytes, available {available}"
        );
        Ok(Self {
            rows,
            pages,
            planes,
            bytes,
        })
    }
}

const DEFS: [(&str, &[&str], &[u8]); 12] = [
    (
        "step5_hidden_norm",
        &["x", "weight", "bias", "cos", "sin", "positions", "out"],
        b"if",
    ),
    ("step5_attn_q", &["x", "w", "out"], b"i"),
    ("step5_attn_kv", &["x", "w", "out"], b"i"),
    ("step5_attn_gate", &["x", "w", "out"], b"i"),
    (
        "step5_swa_q_norm_rope",
        &["x", "weight", "bias", "cos", "sin", "positions", "out"],
        b"if",
    ),
    (
        "step5_swa_k_norm_rope",
        &["x", "weight", "bias", "cos", "sin", "positions", "out"],
        b"if",
    ),
    (
        "step5_sliding_positions",
        &["positions", "requests", "selected"],
        b"i",
    ),
    (
        "step5_swa_attention",
        &[
            "query",
            "key_cache",
            "value_cache",
            "k_descale",
            "v_descale",
            "block_table",
            "request_ids",
            "selected_positions",
            "query_positions",
            "partial_output",
            "partial_lse",
            "output",
        ],
        b"lll fii",
    ),
    ("step5_head_gate", &["x", "logits", "out"], b"i"),
    ("step5_attn_out", &["x", "w", "out"], b"i"),
    ("step5_residual_bf16", &["residual", "delta", "out"], b"i"),
    ("step5_residual_f32", &["residual", "delta", "out"], b"i"),
];

pub(super) struct SwaLayer<'a> {
    // This stream drains before every composed owner, including the FFN.
    stream: Stream<'a>,
    ffn: Option<DenseFfn<'a>>,
    rope: std::rc::Rc<TargetTables<'a>>,
    layer: usize,
    weights: Vec<Allocation<'a>>,
    planes: Vec<Allocation<'a>>,
    programs: Vec<Program<'a>>,
    admission: Admission,
    epsilon: f32,
    heads: usize,
}

impl<'a> SwaLayer<'a> {
    pub fn workspace_bytes(rows: usize) -> Result<u64> {
        Self::workspace_for_heads(rows, 64)
    }

    pub fn workspace_for_heads(rows: usize, heads: usize) -> Result<u64> {
        Ok(Admission::for_heads(rows, heads, &[], 0, 0, u64::MAX, &[])?.bytes)
    }

    pub fn load(
        device: Device<'a>,
        programs: &'a Programs<'a>,
        checkpoint: &'a Checkpoint,
        layer_id: usize,
        rows: usize,
        rope: Option<std::rc::Rc<TargetTables<'a>>>,
        available: u64,
        reservations: &[u64],
    ) -> Result<Self> {
        Self::load_layout(device, programs, checkpoint, layer_id, rows, rope, available,
            reservations, RankWeightLayout::from_checkpoint(checkpoint, 0, 1)?, 64, true)
    }

    pub fn for_execution(
        device: Device<'a>,
        programs: &'a Programs<'a>,
        checkpoint: &'a Checkpoint,
        layer: usize,
        rows: usize,
        rope: Option<std::rc::Rc<TargetTables<'a>>>,
        available: u64,
        reservations: &[u64],
        rank: usize,
        ownership: &cuteafd_loader::plan::families::step5::ownership::ExecutionOwnership,
    ) -> Result<Self> {
        let heads = ownership.layer(layer, rank)?.query_heads;
        ensure!(heads > 0, "Step-5 SWA is not owned by rank {rank}");
        Self::load_layout(device, programs, checkpoint, layer, rows, rope, available,
            reservations, RankWeightLayout::for_execution(checkpoint, rank, ownership)?, heads, false)
    }

    fn load_layout(
        device: Device<'a>,
        programs: &'a Programs<'a>,
        checkpoint: &'a Checkpoint,
        layer_id: usize,
        rows: usize,
        rope: Option<std::rc::Rc<TargetTables<'a>>>,
        available: u64,
        reservations: &[u64],
        layout: RankWeightLayout,
        heads: usize,
        include_dense: bool,
    ) -> Result<Self> {
        let q = heads * 192;
        let kv = heads / 16 * 192;
        let name = |full: &str| if heads == 32 { full.replacen("step5_", "step52_", 1) }
            else { full.to_owned() };
        let output = if heads == 32 { "step52_attn_out_partial" } else { "step5_attn_out" };
        let contract = CoordinatorContract::from_checkpoint(checkpoint)?;
        let layer = contract
            .layers
            .get(layer_id)
            .context("SWA layer outside target")?;
        if let Some(rope) = &rope {
            ensure!(
                rope.tables.plan == contract.rope_tables,
                "SWA rotary plan mismatch"
            );
            ensure!(
                rope.layers[layer_id].attention.cos.device.id == device.id
                    && std::ptr::eq(
                        rope.layers[layer_id].attention.cos.device.library,
                        device.library
                    ),
                "SWA rotary device mismatch"
            );
        }
        ensure!(
            layer.sliding.is_some_and(|w| w.window == 512)
                && layer.geometry.query_heads == 64
                && layer.geometry.kv_heads == 4
                && layer.geometry.qk_dim == 192
                && layer.rope.span == 192,
            "SWA geometry mismatch"
        );
        let suffixes = [
            "input_layernorm.weight",
            "self_attn.q_proj.weight",
            "self_attn.k_proj.weight",
            "self_attn.v_proj.weight",
            "self_attn.q_norm.weight",
            "self_attn.k_norm.weight",
            "self_attn.g_proj.weight",
            "self_attn.o_proj.weight",
        ];
        let weights = suffixes
            .iter()
            .map(|suffix| {
                layout
                    .operands
                    .iter()
                    .find(|w| w.name == format!("model.layers.{layer_id}.{suffix}"))
                    .context("SWA operand missing")
            })
            .collect::<Result<Vec<_>>>()?;
        for (weight, (dtype, shape)) in weights.iter().zip([
            (DType::F32, vec![H]),
            (DType::Bf16, vec![q, H]),
            (DType::Bf16, vec![kv, H]),
            (DType::Bf16, vec![kv, H]),
            (DType::F32, vec![192]),
            (DType::F32, vec![192]),
            (DType::Bf16, vec![heads, H]),
            (DType::Bf16, vec![H, q]),
        ]) {
            ensure!(
                weight.dtype == dtype && weight.shape == shape,
                "SWA weight shape/dtype"
            );
        }
        let dense = layer.dense_activation.filter(|_| include_dense)
            .map(|_| DensePlan::prepare(programs, checkpoint, layer_id, rows, u64::MAX, &[]))
            .transpose()?;
        let admission = Admission::for_heads(
            rows,
            heads,
            &weights
                .iter()
                .map(|w| w.allocation_bytes)
                .collect::<Vec<_>>(),
            dense.as_ref().map_or(0, DensePlan::bytes),
            if rope.is_none() {
                contract.rope_tables.allocation_bytes
            } else {
                0
            },
            available,
            reservations,
        )?;
        for (full, pointers, scalars) in DEFS {
            let name = if full == "step5_attn_out" { output.to_owned() } else { name(full) };
            let spec = programs.spec(&name)?;
            let scalars = scalars
                .iter()
                .copied()
                .filter(|c| *c != b' ')
                .collect::<Vec<_>>();
            ensure!(
                spec.pointers == pointers
                    && spec.scalar_kinds == scalars
                    && spec.capacity_rows as usize >= rows
                    && spec.scratch.is_empty(),
                "{name}: SWA ABI/capacity mismatch"
            );
        }
        for (full, width, outputs) in [
            ("step5_attn_q", H, q),
            ("step5_attn_kv", H, kv),
            ("step5_attn_gate", H, heads),
            ("step5_attn_out", q, H),
        ] {
            let is_partial = heads == 32 && full == "step5_attn_out";
            let name = if is_partial { output.to_owned() } else { name(full) };
            let g = &programs.spec(&name)?.geometry;
            ensure!(
                g["width"] == width
                    && g["outputs"] == outputs
                    && g["output_dtype"] == if is_partial { "torch.float32" } else { "torch.bfloat16" }
                    && g["skinny_rows"] == 0,
                "{name}: SWA projection geometry mismatch"
            );
        }
        for (full, heads) in [("step5_swa_q_norm_rope", heads), ("step5_swa_k_norm_rope", heads / 16)] {
            let g = &programs.spec(&name(full))?.geometry;
            ensure!(
                g["width"] == 192
                    && g["heads"] == heads
                    && g["rotary"] == 192
                    && g["zero_centered"] == true
                    && g["bias"] == false
                    && g["subtract_mean"] == false
                    && g["epsilon"] == "runtime_float32",
                "SWA Q/K norm geometry mismatch"
            );
        }
        let g = &programs.spec(&name("step5_swa_attention"))?.geometry;
        ensure!(
            g["query_heads"] == heads
                && g["kv_heads"] == heads / 16
                && g["page_rows"] == 64
                && g["selection_width"] == 512
                && g["selection_groups"] == 1
                && g["head_dim"] == 192
                && g["key_strides"] == serde_json::json!([64 * kv, kv, 192])
                && g["value_strides"] == serde_json::json!([64 * kv, kv, 192]),
            "SWA attention geometry"
        );
        ensure!(
            programs.spec(&name("step5_sliding_positions"))?.geometry["window"] == 512
                && programs.spec(&name("step5_head_gate"))?.geometry["heads"] == heads
                && programs.spec(&name("step5_head_gate"))?.geometry["dim"] == 192,
            "SWA pointwise geometry"
        );
        // All composed contracts and aggregate bytes are checked before the first allocation.
        let resolved = DEFS
            .iter()
            .map(|(full, pointers, _)| {
                let name = if *full == "step5_attn_out" { output.to_owned() } else { name(full) };
                device.run(|| programs.program(&name, pointers))
            })
            .collect::<Result<Vec<_>>>()?;
        let rope = match rope {
            Some(rope) => rope,
            None => std::rc::Rc::new(TargetTables::load(
                device,
                programs,
                &contract,
                "step5_",
                u64::MAX,
                &[],
            )?),
        };
        let ffn = dense
            .map(|plan| DenseFfn::load_plan(device, programs, checkpoint, plan))
            .transpose()?;
        let weights = weights
            .into_iter()
            .map(|w| load_operand(device, checkpoint, &layout, w))
            .collect::<Result<Vec<_>>>()?;
        let planes = admission
            .planes
            .iter()
            .map(|&bytes| Allocation::new(device, bytes))
            .collect::<Result<Vec<_>>>()?;
        let stream = Stream::new(device)?;
        let owner = Self {
            stream,
            ffn,
            rope,
            layer: layer_id,
            weights,
            planes,
            programs: resolved,
            admission,
            epsilon: contract.norms.input,
            heads,
        };
        // Immutable single-request metadata is initialized once, never on replay.
        for (plane, bytes) in [
            (
                POSITIONS,
                (0..rows as i64)
                    .flat_map(i64::to_le_bytes)
                    .collect::<Vec<_>>(),
            ),
            (REQUESTS, vec![0; rows * 4]),
            (
                TABLE,
                (0..owner.admission.pages as i32)
                    .flat_map(i32::to_le_bytes)
                    .collect(),
            ),
        ] {
            device.run(|| device.library.copy_h2d(owner.planes[plane].buffer, &bytes))?;
        }
        Ok(owner)
    }

    fn launch(
        &self,
        stage: usize,
        pointers: &[*mut std::ffi::c_void],
        scalars: &[Scalar],
    ) -> Result<()> {
        self.launch_on(&self.stream, stage, pointers, scalars)
    }

    fn launch_on(
        &self,
        stream: &Stream<'a>,
        stage: usize,
        pointers: &[*mut std::ffi::c_void],
        scalars: &[Scalar],
    ) -> Result<()> {
        stream.device.run(|| {
            // SAFETY: the retained layer supplies admitted same-device planes;
            // its stream is the first field and drains before all referenced owners.
            unsafe { self.programs[stage].launch(pointers, scalars, stream.raw) }
        })
    }

    pub fn enqueue(&self, rows: usize) -> Result<()> {
        self.project_on(rows, self.planes[POSITIONS].buffer, &self.stream)?;
        self.launch(
            6,
            &[
                self.planes[POSITIONS].buffer.ptr,
                self.planes[REQUESTS].buffer.ptr,
                self.planes[SELECTED].buffer.ptr,
            ],
            &[Scalar::I32(rows as i32)],
        )?;
        self.attend(rows)?;
        self.finish_on(rows, &self.stream)?;
        self.dense_on(rows, &self.stream)
    }

    fn project_on(
        &self,
        rows: usize,
        positions: CuteafdDeviceBuffer,
        stream: &Stream<'a>,
    ) -> Result<()> {
        ensure!(
            rows > 0 && rows <= self.admission.rows,
            "SWA rows exceed capacity"
        );
        let p = |i: usize| self.planes[i].buffer.ptr;
        let w = |i: usize| self.weights[i].buffer.ptr;
        let scalars = [Scalar::I32(rows as i32)];
        let norm = [Scalar::I32(rows as i32), Scalar::F32(self.epsilon)];
        let (cos, sin) = self.rope.layers[self.layer].attention.buffers();
        self.launch_on(
            stream,
            0,
            &[p(X), w(0), w(0), w(0), w(0), w(0), p(NORM)],
            &norm,
        )?;
        self.launch_on(stream, 1, &[p(NORM), w(1), p(QRAW)], &scalars)?;
        self.launch_on(stream, 2, &[p(NORM), w(2), p(KRAW)], &scalars)?;
        self.launch_on(stream, 2, &[p(NORM), w(3), p(VPAGES)], &scalars)?;
        self.launch_on(stream, 3, &[p(NORM), w(6), p(LOGITS)], &scalars)?;
        self.launch_on(
            stream,
            4,
            &[
                p(QRAW),
                w(4),
                w(4),
                cos.ptr,
                sin.ptr,
                positions.ptr,
                p(QUERY),
            ],
            &norm,
        )?;
        self.launch_on(
            stream,
            5,
            &[
                p(KRAW),
                w(5),
                w(5),
                cos.ptr,
                sin.ptr,
                positions.ptr,
                p(KPAGES),
            ],
            &norm,
        )?;
        Ok(())
    }

    fn finish_on(&self, rows: usize, stream: &Stream<'a>) -> Result<()> {
        let p = |i: usize| self.planes[i].buffer.ptr;
        let w = |i: usize| self.weights[i].buffer.ptr;
        let scalars = [Scalar::I32(rows as i32)];
        self.launch_on(stream, 8, &[p(ATTN), p(LOGITS), p(GATED)], &scalars)?;
        self.launch_on(stream, 9, &[p(GATED), w(7), p(DELTA)], &scalars)?;
        if self.heads == 64 {
            self.launch_on(stream, 10, &[p(X), p(DELTA), p(POST)], &scalars)?;
        }
        Ok(())
    }

    fn dense_on(&self, rows: usize, stream: &Stream<'a>) -> Result<()> {
        if let Some(ffn) = &self.ffn {
            stream.device.run(|| {
                // SAFETY: admitted disjoint planes and the composed FFN survive stream drainage.
                unsafe { stream.device.library.copy_d2d_async(ffn.input(), self.planes[POST].buffer,
                    rows * H * 4, stream.raw) }
            })?;
            ffn.enqueue_on(rows, stream)?;
        }
        Ok(())
    }

    pub fn attention_output(&self) -> CuteafdDeviceBuffer {
        self.planes[POST].buffer
    }

    pub fn projection_partial(&self) -> Result<CuteafdDeviceBuffer> {
        ensure!(self.heads == 32, "SWA F32 partial requires head split");
        Ok(self.planes[DELTA].buffer)
    }

    pub(super) fn diagnostic_norm(&self) -> CuteafdDeviceBuffer {
        self.planes[NORM].buffer
    }

    pub(super) fn diagnostic_attention(&self) -> [(CuteafdDeviceBuffer, bool); 2] {
        // In head-split mode NORM holds the once-rounded merged O delta.
        [(self.planes[if self.heads == 32 { NORM } else { DELTA }].buffer, true),
            (self.planes[POST].buffer, false)]
    }

    pub(super) fn diagnostic_dense(&self) -> Result<&DenseFfn<'a>> {
        self.ffn.as_ref().context("SWA diagnostic requires embedded dense FFN")
    }

    /// Ordered rank0 + rank1 projection, one BF16 narrowing, then F32 residual.
    /// The caller retains the exchange and invokes this on its cache stream.
    pub fn merge_partial_on(&self, rows: usize, values: [*const std::ffi::c_void; 2],
        stream: &Stream<'a>) -> Result<()> {
        self.merge_partial_capture_on(rows, values, stream, None)
    }

    pub(super) fn merge_partial_capture_on(&self, rows: usize, values: [*const std::ffi::c_void; 2],
        stream: &Stream<'a>, capture: Option<(&std::path::Path, usize)>) -> Result<()> {
        ensure!(self.heads == 32 && (1..=self.admission.rows).contains(&rows)
            && values.iter().all(|value| !value.is_null())
            && stream.device.id == self.stream.device.id
            && std::ptr::eq(stream.device.library, self.stream.device.library),
            "SWA partial merge geometry/device mismatch");
        let p = |i: usize| self.planes[i].buffer.ptr;
        // POST is scratch until the residual join; NORM is no longer read by
        // attention and is large enough for the rounded hidden-width delta.
        self.launch_on(stream, 11, &[values[0].cast_mut(), values[1].cast_mut(), p(POST)],
            &[Scalar::I32(rows as i32)])?;
        if let Some((directory, rank)) = capture {
            stream.drain()?;
            super::golden::dump_plane(directory, &format!("rank{rank}-merged_o_f32"),
                stream.device, self.planes[POST].buffer, rows, false)?;
        }
        stream.device.run(|| {
            // SAFETY: disjoint admitted same-device planes survive cache-stream drainage.
            unsafe { stream.device.library.cuda_f32_to_bf16_async(self.planes[POST].buffer,
                self.planes[NORM].buffer, rows * H, stream.raw) }
        })?;
        self.launch_on(stream, 10, &[p(X), p(NORM), p(POST)], &[Scalar::I32(rows as i32)])
    }

    pub fn enqueue_step<'e, E: cuteafd_hostcache::copy::CopyEngine>(
        &'e mut self,
        step: &mut Step<'e, 'a, E>,
        rank: usize,
    ) -> Result<()> {
        step.project(rank, self.layer, &*self, |layer, input| {
            layer.check_input(&input)?;
            layer.project_on(input.rows, input.positions, input.stream)
        })?;
        step.stage(
            rank,
            self.layer,
            &self.planes[KPAGES],
            &self.planes[VPAGES],
            None,
        )?;
        step.attend(rank, self.layer, &*self, |layer, read| {
            let rows = read.inputs.rows;
            let stream = read.inputs.stream;
            layer.read_on(read)?;
            layer.dense_on(rows, stream)
        })
    }

    pub fn enqueue_attention_step<'e, E: cuteafd_hostcache::copy::CopyEngine>(
        &'e self,
        step: &mut Step<'e, 'a, E>,
        rank: usize,
    ) -> Result<()> {
        ensure!(self.ffn.is_none(), "shared SWA pipeline must own its FFN separately");
        step.project(rank, self.layer, self, |layer, input| {
            layer.check_input(&input)?;
            layer.project_on(input.rows, input.positions, input.stream)
        })?;
        step.stage(rank, self.layer, &self.planes[KPAGES], &self.planes[VPAGES], None)?;
        step.attend(rank, self.layer, self, |layer, read| layer.read_on(read))
    }

    fn check_input(&self, input: &StepInputs<'_, 'a>) -> Result<()> {
        ensure!(
            input.heads == self.heads
                && input.stream.device.id == self.stream.device.id
                && std::ptr::eq(input.stream.device.library, self.stream.device.library)
                && input
                    .base
                    .checked_add(input.rows)
                    .is_some_and(|end| end <= self.rope.tables.plan.extent as usize),
            "SWA owner device/head/rotary extent mismatch"
        );
        Ok(())
    }

    fn read_on(&self, read: AttentionRead<'_, 'a>) -> Result<()> {
        self.check_input(&read.inputs)?;
        let input = &read.inputs;
        self.launch_on(
            input.stream,
            6,
            &[
                input.positions.ptr,
                input.requests.ptr,
                self.planes[SELECTED].buffer.ptr,
            ],
            &[Scalar::I32(input.rows as i32)],
        )?;
        self.attend_on(
            input.rows,
            input.positions,
            input.requests,
            read.key,
            read.value,
            read.table,
            read.table_width,
            read.pages,
            input.stream,
        )?;
        self.finish_on(input.rows, input.stream)
    }

    fn attend(&self, rows: usize) -> Result<()> {
        self.attend_on(
            rows,
            self.planes[POSITIONS].buffer,
            self.planes[REQUESTS].buffer,
            self.planes[KPAGES].buffer,
            self.planes[VPAGES].buffer,
            self.planes[TABLE].buffer,
            self.admission.pages,
            self.admission.pages,
            &self.stream,
        )
    }

    fn attend_on(
        &self,
        rows: usize,
        positions: CuteafdDeviceBuffer,
        requests: CuteafdDeviceBuffer,
        key: CuteafdDeviceBuffer,
        value: CuteafdDeviceBuffer,
        table: CuteafdDeviceBuffer,
        table_width: usize,
        pages: usize,
        stream: &Stream<'a>,
    ) -> Result<()> {
        let p = |i: usize| self.planes[i].buffer.ptr;
        self.launch_on(
            stream,
            7,
            &[
                p(QUERY),
                key.ptr,
                value.ptr,
                p(QUERY),
                p(QUERY),
                table.ptr,
                requests.ptr,
                p(SELECTED),
                positions.ptr,
                p(QUERY),
                p(QUERY),
                p(ATTN),
            ],
            &[
                Scalar::I64(pages as i64),
                Scalar::I64(1),
                Scalar::I64(table_width as i64),
                Scalar::F32(192f32.sqrt().recip()),
                Scalar::I32(rows as i32),
                Scalar::I32(1),
            ],
        )
    }

    pub fn validate_live_weights(&self, device: Device<'_>) -> Result<()> {
        super::weights::validate_live_weights(device.id, &self.weights)
    }

    pub fn input(&self) -> CuteafdDeviceBuffer {
        self.planes[X].buffer
    }
    pub fn has_dense_ffn(&self) -> bool {
        self.ffn.is_some()
    }
    pub fn output(&self) -> CuteafdDeviceBuffer {
        self.ffn
            .as_ref()
            .map_or(self.planes[POST].buffer, DenseFfn::output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn exact_head_workspace_charges_the_f32_partial_at_every_row_capacity() -> Result<()> {
        for rows in 1..=4096 {
            let split = Admission::for_heads(rows, 32, &[], 0, 0, u64::MAX, &[])?;
            let whole = Admission::for_heads(rows, 64, &[], 0, 0, u64::MAX, &[])?;
            assert_eq!(split.planes[DELTA], (rows * H * 4).max(256));
            assert_eq!(SwaLayer::workspace_for_heads(rows, 32)?, split.bytes);
            assert!(split.bytes <= whole.bytes);
        }
        assert!(SwaLayer::workspace_for_heads(0, 32).is_err());
        assert!(SwaLayer::workspace_for_heads(4097, 32).is_err());
        assert!(SwaLayer::workspace_for_heads(64, 16).is_err());
        Ok(())
    }

    fn fixture(root: &std::path::Path, manifest: &serde_json::Value, key: &str) -> Result<Vec<u8>> {
        use sha2::{Digest, Sha256};
        let bytes = std::fs::read(root.join(format!("{key}.bin")))?;
        ensure!(
            manifest["files"][key]["bytes"] == bytes.len()
                && manifest["files"][key]["sha256"].as_str()
                    == Some(format!("{:x}", Sha256::digest(&bytes)).as_str()),
            "SWA fixture hash/length mismatch"
        );
        Ok(bytes)
    }

    fn values(bytes: &[u8], size: usize) -> Result<Vec<f64>> {
        ensure!(bytes.len() % size == 0, "SWA value alignment");
        let result = bytes
            .chunks_exact(size)
            .map(|v| match size {
                2 => f32::from_bits(u32::from(u16::from_le_bytes(v.try_into().unwrap())) << 16)
                    as f64,
                4 => f32::from_le_bytes(v.try_into().unwrap()) as f64,
                8 => f64::from_le_bytes(v.try_into().unwrap()),
                _ => unreachable!(),
            })
            .collect::<Vec<_>>();
        ensure!(result.iter().all(|v| v.is_finite()), "nonfinite SWA output");
        Ok(result)
    }

    fn metrics(
        actual: &[u8],
        expected: &[u8],
        size: usize,
        reference_size: usize,
        width: usize,
    ) -> Result<f64> {
        let a = values(actual, size)?;
        let b = values(expected, reference_size)?;
        ensure!(a.len() == b.len(), "SWA metric length mismatch");
        let error = a.iter().zip(&b).map(|(a, b)| (a - b).powi(2)).sum::<f64>();
        let norm = b.iter().map(|v| v * v).sum::<f64>().max(f64::MIN_POSITIVE);
        let maximum = a
            .iter()
            .zip(&b)
            .map(|(a, b)| (a - b).abs())
            .fold(0f64, f64::max);
        let worst = a
            .chunks(width)
            .zip(b.chunks(width))
            .enumerate()
            .map(|(row, (a, b))| {
                let n = b.iter().map(|v| v * v).sum::<f64>().max(f64::MIN_POSITIVE);
                let e = a.iter().zip(b).map(|(a, b)| (a - b).powi(2)).sum::<f64>();
                (row, (e / n).sqrt())
            })
            .max_by(|a, b| a.1.total_cmp(&b.1))
            .unwrap();
        eprintln!(
            "rel_l2={:.12e} max_abs={maximum:.12e} worst_position={} max_position_rel_l2={:.12e}",
            (error / norm).sqrt(),
            worst.0,
            worst.1
        );
        Ok((error / norm).sqrt())
    }

    fn read(chain: &SwaLayer<'_>, buffer: CuteafdDeviceBuffer, bytes: usize) -> Result<Vec<u8>> {
        let mut output = vec![0; bytes];
        chain
            .stream
            .device
            .run(|| chain.stream.device.library.copy_d2h(&mut output, buffer))?;
        Ok(output)
    }

    #[test]
    #[ignore = "requires pinned layer-0 fixtures, sealed native library and CUDA"]
    fn native_swa_layer0_boundaries_and_graph_replay() -> Result<()> {
        native_swa_component(0)
    }

    #[test]
    #[ignore = "requires pinned SWA layer-1/2 fixtures, sealed native library and CUDA"]
    fn native_swa_layers1_2_boundaries_and_graph_replay() -> Result<()> {
        for layer in 1..3 {
            native_swa_component(layer)?;
        }
        Ok(())
    }

    fn native_swa_component(layer: usize) -> Result<()> {
        let mut root = std::path::PathBuf::from(std::env::var("CUTEAFD_STEP5_SWA_FIXTURE_DIR")?);
        if layer > 0 {
            root.push(format!("layer{layer}"));
        }
        let manifest: serde_json::Value =
            serde_json::from_slice(&std::fs::read(root.join("manifest.json"))?)?;
        ensure!(
            manifest["schema"] == 1
                && manifest["seed"] == 274
                && manifest["layer"] == layer
                && manifest["checkpoint"] == "stepfun-ai/Step-5-Preview-BF16"
                && manifest["revision"] == "7734b7ffac118d70d8398bfa3eb0fab007eb45b1"
                && manifest["rows"] == serde_json::json!([1, 7, 128, 513]),
            "SWA fixture identity"
        );
        // SAFETY: the explicit GPU test supplies a sealed native library.
        let library =
            unsafe { cuteafd_ffi::NativeLibrary::load(std::env::var("CUTEAFD_NATIVE_LIB")?)? };
        let device = crate::shared::memory::device::cache_test_device(&library)?;
        let programs = library
            .programs()?
            .with_manifest(std::path::Path::new(&std::env::var(
                "CUTEAFD_PROGRAM_MANIFEST",
            )?))?;
        let checkpoint =
            Checkpoint::open(std::path::Path::new(&std::env::var("CUTEAFD_SNAPSHOT")?))?;
        ensure!(
            checkpoint.snapshot.file_name().and_then(|s| s.to_str())
                == manifest["revision"].as_str(),
            "SWA snapshot revision mismatch"
        );
        let mut chain = SwaLayer::load(
            device,
            &programs,
            &checkpoint,
            layer,
            513,
            None,
            u64::MAX,
            &[],
        )?;
        ensure!(
            manifest["epsilon"].as_f64().map(|v| v as f32) == Some(chain.epsilon),
            "SWA epsilon mismatch"
        );
        assert!(SwaLayer::load(
            device,
            &programs,
            &checkpoint,
            layer,
            513,
            None,
            chain.admission.bytes - 1,
            &[]
        )
        .is_err());
        assert!(chain.enqueue(514).is_err());
        let boundaries = [
            ("norm", NORM, H, 2),
            ("qraw", QRAW, Q, 2),
            ("kraw", KRAW, KV, 2),
            ("value", VPAGES, KV, 2),
            ("query", QUERY, Q, 2),
            ("key", KPAGES, KV, 2),
            ("logits", LOGITS, 64, 2),
            ("attention", ATTN, Q, 2),
            ("gated", GATED, Q, 2),
            ("delta", DELTA, H, 2),
        ];
        for rows in [1, 7, 128, 513] {
            let load = |name: &str| fixture(&root, &manifest, &format!("r{rows}_{name}"));
            device.run(|| library.copy_h2d(chain.input(), &load("residual")?))?;
            chain.enqueue(rows)?;
            chain.stream.drain()?;
            let mut eager = Vec::new();
            for (name, plane, width, size) in boundaries {
                let actual = read(&chain, chain.planes[plane].buffer, rows * width * size)?;
                eprintln!("SWA chained rows={rows} boundary={name}");
                metrics(&actual, &load(name)?, size, size, width)?;
                if name == "norm" {
                    super::super::norm_accuracy::check(
                        &actual,
                        &load(name)?,
                        &load("norm_fp64")?,
                        width,
                    )?;
                }
                eager.push((chain.planes[plane].buffer, actual));
            }
            for (name, buffer) in [
                ("post", chain.ffn.as_ref().unwrap().input()),
                ("output", chain.output()),
            ] {
                let actual = read(&chain, buffer, rows * H * 4)?;
                eprintln!("SWA chained rows={rows} boundary={name}");
                metrics(&actual, &load(name)?, 4, 4, H)?;
                eager.push((buffer, actual));
            }
            device.run(|| {
                // SAFETY: stream is idle; all graph operands are pinned below.
                unsafe { library.cuda_graph_begin_capture(chain.stream.raw) }
            })?;
            let enqueue = chain.enqueue(rows);
            let raw = device.run(|| {
                // SAFETY: end capture even if enqueue failed.
                unsafe { library.cuda_graph_end_capture(chain.stream.raw) }
            })?;
            // SAFETY: mutable chain borrow retains every stream, allocation and program.
            let graph = unsafe {
                crate::shared::decode_graph::GraphOwner::new(&library, device.id, raw, &mut chain)?
            };
            enqueue?;
            let mut guard = graph.pins.stream.cancellation_guard("SWA graph replay");
            device.run(|| {
                // SAFETY: retained graph operands remain live until the drain below.
                unsafe { library.cuda_graph_launch(graph.raw, graph.pins.stream.raw) }
            })?;
            graph.pins.stream.drain()?;
            guard.complete = true;
            drop(guard);
            for (buffer, expected) in eager {
                ensure!(
                    read(graph.pins, buffer, expected.len())? == expected,
                    "SWA eager/replay mismatch"
                );
            }
            drop(graph);
            // Isolate GEMMs with official producers and the accepted FP64 error bar.
            for (stage, input, weight, plane, name) in [
                (1, "norm", 1, QRAW, "qraw"),
                (2, "norm", 2, KRAW, "kraw"),
                (2, "norm", 3, VPAGES, "value"),
                (3, "norm", 6, LOGITS, "logits"),
                (9, "gated", 7, DELTA, "delta"),
            ] {
                let input_plane = if stage == 9 { GATED } else { NORM };
                device.run(|| library.copy_h2d(chain.planes[input_plane].buffer, &load(input)?))?;
                chain.launch(
                    stage,
                    &[
                        chain.planes[input_plane].buffer.ptr,
                        chain.weights[weight].buffer.ptr,
                        chain.planes[plane].buffer.ptr,
                    ],
                    &[Scalar::I32(rows as i32)],
                )?;
                chain.stream.drain()?;
                let expected = load(name)?;
                let actual = read(&chain, chain.planes[plane].buffer, expected.len())?;
                let fp64 = load(&format!("{name}_fp64"))?;
                eprintln!("SWA isolated rows={rows} op={name} native_vs_fp64");
                let native = metrics(&actual, &fp64, 2, 8, expected.len() / rows / 2)?;
                let official = metrics(&expected, &fp64, 2, 8, expected.len() / rows / 2)?;
                ensure!(
                    native <= official * 1.01,
                    "SWA GEMM exceeds 1.01 official FP64 error"
                );
            }
            // Q/K norm and RoPE consume the official projection outputs, not
            // the native GEMM's accumulation-order differences.
            for (stage, input, weight, output, name) in
                [(4, QRAW, 4, QUERY, "query"), (5, KRAW, 5, KPAGES, "key")]
            {
                let source = if stage == 4 { "qraw" } else { "kraw" };
                device.run(|| library.copy_h2d(chain.planes[input].buffer, &load(source)?))?;
                let (cos, sin) = chain.rope.layers[layer].attention.buffers();
                chain.launch(
                    stage,
                    &[
                        chain.planes[input].buffer.ptr,
                        chain.weights[weight].buffer.ptr,
                        chain.weights[weight].buffer.ptr,
                        cos.ptr,
                        sin.ptr,
                        chain.planes[POSITIONS].buffer.ptr,
                        chain.planes[output].buffer.ptr,
                    ],
                    &[Scalar::I32(rows as i32), Scalar::F32(chain.epsilon)],
                )?;
                chain.stream.drain()?;
                let expected = load(name)?;
                let actual = read(&chain, chain.planes[output].buffer, expected.len())?;
                super::super::norm_accuracy::check(
                    &actual,
                    &expected,
                    &load(&format!("{name}_fp64"))?,
                    192,
                )?;
            }
            for (plane, name) in [(QUERY, "query"), (KPAGES, "key"), (VPAGES, "value")] {
                device.run(|| library.copy_h2d(chain.planes[plane].buffer, &load(name)?))?;
            }
            chain.attend(rows)?;
            chain.stream.drain()?;
            let expected = load("attention")?;
            let actual = read(&chain, chain.planes[ATTN].buffer, expected.len())?;
            let fp64 = load("attention_fp64")?;
            eprintln!("SWA isolated rows={rows} attention native_vs_fp64");
            let native = metrics(&actual, &fp64, 2, 8, Q)?;
            eprintln!("SWA isolated rows={rows} attention official_vs_fp64");
            let official = metrics(&expected, &fp64, 2, 8, Q)?;
            // Shared flash attention rounds unnormalized online P to BF16
            // before PV, unlike this official SDPA backend. Qualify that choice
            // against an FP64 online BF16-P oracle, not a Step-5-only kernel fork.
            // Measured native/oracle FP64-error ratios at M7/128/513:
            // 1.000000809 / 1.000010453 / 0.999995771; the explicit bar is 1.01.
            let bf16_p = load("attention_bf16_p")?;
            eprintln!("SWA isolated rows={rows} attention bf16_p_vs_fp64");
            let rounded = metrics(&bf16_p, &fp64, 2, 8, Q)?;
            eprintln!("SWA isolated rows={rows} attention native_vs_bf16_p");
            metrics(&actual, &bf16_p, 2, 2, Q)?;
            eprintln!(
                "SWA attention rows={rows} native_official_ratio={} native_bf16_p_ratio={}",
                native / official,
                native / rounded
            );
            ensure!(
                native <= rounded * 1.01,
                "SWA attention exceeds 1.01 BF16-P FP64 error"
            );
            // The SDPA ratio is diagnostic only: rounding P is a shared flash
            // precision choice, with context-dependent error (about 1.09x at M7).
            // Also gate each position, so a localized mask/window error cannot
            // hide behind the whole-batch norm.
            let actual_values = values(&actual, 2)?;
            let rounded_values = values(&bf16_p, 2)?;
            let reference_values = values(&fp64, 8)?;
            for (position, ((actual, rounded), reference)) in actual_values
                .chunks(Q)
                .zip(rounded_values.chunks(Q))
                .zip(reference_values.chunks(Q))
                .enumerate()
            {
                let squared_error = |a: &[f64]| {
                    a.iter()
                        .zip(reference)
                        .map(|(a, b)| (a - b).powi(2))
                        .sum::<f64>()
                };
                ensure!(
                    squared_error(actual) <= squared_error(rounded) * 1.01f64.powi(2),
                    "SWA attention position {position} exceeds 1.01 BF16-P FP64 error"
                );
            }
            for (plane, name) in [(ATTN, "attention"), (LOGITS, "logits")] {
                device.run(|| library.copy_h2d(chain.planes[plane].buffer, &load(name)?))?;
            }
            chain.launch(
                8,
                &[
                    chain.planes[ATTN].buffer.ptr,
                    chain.planes[LOGITS].buffer.ptr,
                    chain.planes[GATED].buffer.ptr,
                ],
                &[Scalar::I32(rows as i32)],
            )?;
            chain.stream.drain()?;
            ensure!(
                read(&chain, chain.planes[GATED].buffer, load("gated")?.len())? == load("gated")?,
                "SWA isolated head gate is not byte-exact"
            );
            device.run(|| library.copy_h2d(chain.planes[DELTA].buffer, &load("delta")?))?;
            chain.launch(
                10,
                &[
                    chain.input().ptr,
                    chain.planes[DELTA].buffer.ptr,
                    chain.ffn.as_ref().unwrap().input().ptr,
                ],
                &[Scalar::I32(rows as i32)],
            )?;
            chain.stream.drain()?;
            ensure!(
                read(
                    &chain,
                    chain.ffn.as_ref().unwrap().input(),
                    load("post")?.len()
                )? == load("post")?,
                "SWA isolated residual is not byte-exact"
            );
        }
        Ok(())
    }

    #[test]
    #[ignore = "requires pinned SWA fixtures, sealed native library and CUDA"]
    fn native_swa_transaction_reads_match_contiguous_producers() -> Result<()> {
        use super::super::{
            cache::CacheRank,
            prefix::{Frontier, Placement, Step5Prefix},
            transaction::{tests::stage_other_layers, Staging},
        };
        use cuteafd_engine::prefix::{PrefixCache, PrefixConfig, PrefixFamily};
        use cuteafd_hostcache::copy::StubCopyEngine;
        use std::cell::RefCell;
        let root = std::path::PathBuf::from(std::env::var("CUTEAFD_STEP5_SWA_FIXTURE_DIR")?);
        let manifest: serde_json::Value =
            serde_json::from_slice(&std::fs::read(root.join("manifest.json"))?)?;
        ensure!(
            manifest["seed"] == 274
                && manifest["layer"] == 0
                && manifest["revision"] == "7734b7ffac118d70d8398bfa3eb0fab007eb45b1",
            "SWA fixture identity"
        );
        // SAFETY: the explicit GPU gate supplies a sealed native library.
        let library =
            unsafe { cuteafd_ffi::NativeLibrary::load(std::env::var("CUTEAFD_NATIVE_LIB")?)? };
        let device = crate::shared::memory::device::cache_test_device(&library)?;
        let programs = library
            .programs()?
            .with_manifest(std::path::Path::new(&std::env::var(
                "CUTEAFD_PROGRAM_MANIFEST",
            )?))?;
        let checkpoint =
            Checkpoint::open(std::path::Path::new(&std::env::var("CUTEAFD_SNAPSHOT")?))?;
        ensure!(
            checkpoint.snapshot.file_name().and_then(|s| s.to_str())
                == manifest["revision"].as_str(),
            "SWA snapshot"
        );
        let total = 777;
        let mut chain = SwaLayer::load(
            device,
            &programs,
            &checkpoint,
            0,
            total,
            None,
            u64::MAX,
            &[],
        )?;
        let source = fixture(&root, &manifest, "r513_residual")?;
        let residual: Vec<u8> = source.iter().copied().cycle().take(total * H * 4).collect();
        device.run(|| library.copy_h2d(chain.input(), &residual))?;
        chain.enqueue(total)?;
        chain.stream.drain()?;
        let expected = read(&chain, chain.planes[ATTN].buffer, total * Q * 2)?;
        let producers = [
            (X, H * 4),
            (QUERY, Q * 2),
            (KPAGES, KV * 2),
            (VPAGES, KV * 2),
            (LOGITS, 64 * 2),
        ]
        .into_iter()
        .map(|(plane, row)| {
            Ok((
                plane,
                row,
                read(&chain, chain.planes[plane].buffer, total * row)?,
            ))
        })
        .collect::<Result<Vec<_>>>()?;
        let layout = super::super::cache::tests::layout_for(1024, 1, 0, 64);
        let owner = RefCell::new(CacheRank::new(device, layout, 0, 1, 2, 40, usize::MAX)?);
        let staging = Staging::new(device, &programs, &owner.borrow(), usize::MAX)?;
        let family = Step5Prefix::new(vec![&owner])?;
        let mut pool = PrefixCache::<StubCopyEngine, Frontier>::new(
            family.layout(),
            PrefixConfig {
                entries: 0,
                mark_slots: 2,
                keep_logits: false,
                min_tokens: 1,
            },
            None,
        )?;
        let mut placement = pool
            .admit_cold(&family, 1, 1, |pages| Placement {
                pages,
                sequence: 0,
                committed: 0,
            })?
            .placement;
        let key = Allocation::new(device, 64 * KV * 2)?;
        let value = Allocation::new(device, 64 * KV * 2)?;
        let ik = Allocation::new(device, 64 * 512)?;
        let iz = Allocation::new(device, 64 * 512)?;
        // Deliberately cross partial pages, the 512-token window, and the 576-row ring.
        for end in [
            7, 8, 63, 64, 128, 192, 256, 320, 384, 448, 511, 512, 575, 576, 640, 704, 768, total,
        ] {
            let base = placement.committed;
            let rows = end - base;
            let before = placement.pages.clone();
            for abort in [true, false] {
                let mut step = Step::begin(
                    vec![&owner],
                    vec![&staging],
                    &mut pool,
                    &mut placement,
                    rows,
                )?;
                assert!(step.attend(0, 0, &chain, |_, _| Ok(())).is_err());
                step.project(0, 0, &chain, |layer, input| {
                    layer.check_input(&input)?;
                    for (plane, row, data) in &producers {
                        device.run(|| {
                            // SAFETY: ordered copies retain source slices; transaction retains the owner.
                            unsafe {
                                library.copy_h2d_ordered(
                                    layer.planes[*plane].buffer,
                                    &data[base * row..end * row],
                                    input.stream.raw,
                                )
                            }
                        })?;
                    }
                    Ok(())
                })?;
                step.stage(0, 0, &chain.planes[KPAGES], &chain.planes[VPAGES], None)?;
                step.attend(0, 0, &chain, |layer, read| layer.read_on(read))?;
                owner.borrow().stream.drain()?;
                ensure!(
                    read(&chain, chain.planes[ATTN].buffer, rows * Q * 2)?
                        == expected[base * Q * 2..end * Q * 2],
                    "SWA transaction attention differs at {base}..{end}"
                );
                if abort {
                    drop(step);
                    ensure!(
                        placement.committed == base && placement.pages == before,
                        "SWA abort visibility"
                    );
                } else {
                    stage_other_layers(&mut step, 0, &key, &value, &ik, &iz)?;
                    step.publish(rows)?;
                }
            }
        }
        pool.release(&family, &placement.pages)?;
        // Exercise the production projection/staging/read entry, not just borrowed producers.
        device.run(|| library.copy_h2d(chain.input(), &residual[..64 * H * 4]))?;
        chain.enqueue(64)?;
        chain.stream.drain()?;
        let eager = read(&chain, chain.output(), 64 * H * 4)?;
        let mut placement = pool
            .admit_cold(&family, 1, 1, |pages| Placement {
                pages,
                sequence: 0,
                committed: 0,
            })?
            .placement;
        let mut step = Step::begin(vec![&owner], vec![&staging], &mut pool, &mut placement, 64)?;
        chain.enqueue_step(&mut step, 0)?;
        stage_other_layers(&mut step, 0, &key, &value, &ik, &iz)?;
        step.publish(64)?;
        ensure!(
            read(&chain, chain.output(), eager.len())? == eager,
            "SWA production enqueue differs"
        );
        pool.release(&family, &placement.pages)?;
        // A routed SWA layer has no dense FFN and shares the already admitted rotary owner.
        let routed = SwaLayer::load(
            device,
            &programs,
            &checkpoint,
            4,
            1,
            Some(std::rc::Rc::clone(&chain.rope)),
            u64::MAX,
            &[],
        )?;
        ensure!(
            std::rc::Rc::ptr_eq(&routed.rope, &chain.rope) && routed.ffn.is_none(),
            "SWA shared layer binding"
        );
        Ok(())
    }

    #[test]
    fn swa_head_split_admits_native_half_heads_and_f32_projection() -> Result<()> {
        for rows in [1, 7, 64, 65, 4096] {
            let a = Admission::for_heads(rows, 32, &[], 0, 0, u64::MAX, &[])?;
            assert_eq!(a.planes[QRAW], rows * 32 * 192 * 2);
            assert_eq!(a.planes[KPAGES], rows.div_ceil(64) * 64 * 2 * 192 * 2);
            assert_eq!(a.planes[DELTA], rows * H * 4);
            assert_eq!(a.planes[NORM], rows * H * 2);
            assert_eq!(a.planes[POST], rows * H * 4);
            assert_eq!(a.bytes, a.planes.iter().sum::<usize>() as u64);
            assert!(Admission::for_heads(rows, 32, &[], 0, 0, a.bytes - 1, &[]).is_err());
        }
        for heads in [0, 16, 48, 128] {
            assert!(Admission::for_heads(1, heads, &[], 0, 0, u64::MAX, &[]).is_err());
        }
        Ok(())
    }

    #[test]
    fn swa_admission_charges_page_padding_and_composed_owners() -> Result<()> {
        let a = Admission::new(1, &[1000], 2000, 3000, u64::MAX, &[])?;
        assert_eq!(a.pages, 1);
        assert_eq!(a.planes[KPAGES], 64 * KV * 2);
        assert_eq!(a.bytes, a.planes.iter().sum::<usize>() as u64 + 6000);
        assert!(Admission::new(0, &[], 0, 0, u64::MAX, &[]).is_err());
        assert!(Admission::new(4097, &[], 0, 0, u64::MAX, &[]).is_err());
        assert!(Admission::new(1, &[1000], 2000, 3000, a.bytes - 1, &[]).is_err());
        assert!(Admission::new(1, &[], 0, 0, u64::MAX, &[u64::MAX]).is_err());
        Ok(())
    }
}
