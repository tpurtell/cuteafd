//! Layer-selected CSA branch over transaction-owned pages and summaries.
use super::scratch::{self, Arena, Plane};
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
use std::ffi::c_void;

const H: usize = 4096;
const Q: usize = 64 * 192;
const KV: usize = 4 * 192;
const TILE: usize = 64;
const CAPACITY: usize = 131072;
const SELECT: usize = 4 * 4104;
const X: usize = 0;
const NORM: usize = 1;
const QRAW: usize = 2;
const KRAW: usize = 3;
const VALUE: usize = 4;
const QUERY: usize = 5;
const KEY: usize = 6;
const LOGITS: usize = 7;
const ATTN: usize = 8;
const GATED: usize = 9;
const DELTA: usize = 10;
const POST: usize = 11;
const POSITIONS: usize = 12;
const REQUESTS: usize = 13;
const TABLE: usize = 14;
const IQRAW: usize = 15;
const IKRAW: usize = 16;
const IZ: usize = 17;
const IQ: usize = 18;
const IK: usize = 19;
const IW: usize = 20;
const IWSCR: usize = 21;
const STARTS: usize = 22;
const COUNTS: usize = 23;
const SUMMARY: usize = 24;
const UNROUNDED: usize = 25;
const SCORES: usize = 26;
const SELECTED: usize = 27;
const NORMALIZED: usize = 28;
const LENGTHS: usize = 29;
const VALUES: usize = 30;
const IDS: usize = 31;

#[derive(Clone)]
struct Admission {
    rows: usize,
    pages: usize,
    planes: [usize; 32],
    bytes: u64,
}
impl Admission {
    fn new(
        rows: usize,
        weights: &[u64],
        rope: u64,
        available: u64,
        reservations: &[u64],
    ) -> Result<Self> {
        ensure!(
            (1..=8192).contains(&rows),
            "CSA staged row capacity must be 1..=8192"
        );
        let pages = rows.div_ceil(64);
        // Scores and top-k scratch are bounded by query tiles, never prefill length.
        let planes = cuteafd_loader::plan::families::step5::native::csa_planes(rows)?;
        let bytes = planes
            .iter()
            .map(|&n| n as u64)
            .chain(weights.iter().copied())
            .chain([rope])
            .try_fold(0u64, |a, b| a.checked_add(b))
            .context("CSA admission overflow")?;
        let total = reservations
            .iter()
            .try_fold(bytes, |a, b| a.checked_add(*b))
            .context("CSA reservation overflow")?;
        ensure!(
            total <= available,
            "CSA needs {total} bytes, available {available}"
        );
        Ok(Self {
            rows,
            pages,
            planes,
            bytes,
        })
    }
}

const NORM_POINTERS: &[&str] = &["x", "weight", "bias", "cos", "sin", "positions", "out"];
const DEFS: [(&str, &[&str], &[u8]); 17] = [
    ("step5_hidden_norm", NORM_POINTERS, b"if"),
    ("step5_attn_q", &["x", "w", "out"], b"i"),
    ("step5_attn_kv", &["x", "w", "out"], b"i"),
    ("step5_attn_gate", &["x", "w", "out"], b"i"),
    ("step5_csa_q_norm_rope", NORM_POINTERS, b"if"),
    ("step5_csa_k_norm_rope", NORM_POINTERS, b"if"),
    ("step5_index_q", &["x", "w", "out"], b"i"),
    ("step5_index_kz", &["x", "w", "out"], b"i"),
    ("step5_index_q_norm", NORM_POINTERS, b"if"),
    ("step5_index_k_norm", NORM_POINTERS, b"if"),
    (
        "step5_index_weight",
        &["x", "weight", "out", "scratch"],
        b"i",
    ),
    (
        "step5_region_compress",
        &["keys", "logits", "starts", "counts", "output", "unrounded"],
        b"il",
    ),
    (
        "step5_region_scores",
        &[
            "query",
            "weights",
            "summaries",
            "requests",
            "positions",
            "scores",
        ],
        b"il",
    ),
    (
        "step5_region_selection",
        &[
            "scores",
            "positions",
            "requests",
            "selected",
            "normalized",
            "lengths",
            "values",
            "ids",
        ],
        b"i",
    ),
    (
        "step5_csa_attention",
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
        b"lllfii",
    ),
    ("step5_head_gate", &["x", "logits", "out"], b"i"),
    ("step5_attn_out", &["x", "w", "out"], b"i"),
];

struct CsaWeights<'a> {
    rope: std::rc::Rc<TargetTables<'a>>,
    layer: usize,
    weights: Vec<Allocation<'a>>,
    programs: Vec<Program<'a>>,
    residual: Program<'a>,
    epsilon: f32,
}

pub(super) struct CsaLayer<'a> {
    // Drain the complete staged branch before any pointer owner retires.
    stream: Stream<'a>,
    ffn: Option<DenseFfn<'a>>,
    planes: Vec<Plane<'a>>,
    admission: Admission,
    immutable: std::rc::Rc<CsaWeights<'a>>,
}
impl<'a> CsaLayer<'a> {
    pub(super) fn device(&self) -> Device<'a> { self.stream.device }
    pub(super) fn rows(&self) -> usize { self.admission.rows }
    pub(super) fn new_lane_in(&self, arena: &Arena<'a>) -> Result<Self> {
        Self::workspace(self.stream.device, self.admission.clone(), self.immutable.clone(), None, Some((arena, 0)))
    }

    pub fn workspace_bytes(rows: usize) -> Result<u64> {
        Ok(Admission::new(rows, &[], 0, u64::MAX, &[])?.bytes)
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
            reservations, RankWeightLayout::from_checkpoint(checkpoint, 0, 1)?, true, None)
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
        arena: &Arena<'a>,
        ownership: &cuteafd_loader::plan::families::step5::ownership::ExecutionOwnership,
    ) -> Result<Self> {
        let local = ownership.layer(layer, rank)?;
        ensure!(local.query_heads == 64 && local.kv_heads == 4,
            "Step-5 CSA needs a whole native attention owner");
        Self::load_layout(device, programs, checkpoint, layer, rows, rope, available,
            reservations, RankWeightLayout::for_execution(checkpoint, rank, ownership)?, false, Some((arena, 0)))
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
        include_dense: bool,
        arena: Option<(&Arena<'a>, usize)>,
    ) -> Result<Self> {
        let contract = CoordinatorContract::from_checkpoint(checkpoint)?;
        let layer = contract
            .layers
            .get(layer_id)
            .context("CSA layer outside target")?;
        if let Some(rope) = &rope {
            ensure!(
                rope.tables.plan == contract.rope_tables,
                "CSA rotary plan mismatch"
            );
            ensure!(
                rope.layers[layer_id].attention.cos.device.id == device.id
                    && std::ptr::eq(
                        rope.layers[layer_id].attention.cos.device.library,
                        device.library
                    ),
                "CSA rotary device mismatch"
            );
        }
        ensure!(
            layer.sliding.is_none()
                && layer.geometry.query_heads == 64
                && layer.geometry.kv_heads == 4
                && layer.geometry.qk_dim == 192
                && layer.rope.span == 64
                && layer.indexer_rope.is_some_and(|r| r.span == 32),
            "CSA layer geometry"
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
            "self_attn.sparse_indexer_q.weight",
            "self_attn.sparse_indexer_k.weight",
            "self_attn.sparse_indexer_z.weight",
            "self_attn.sparse_indexer_q_norm.weight",
            "self_attn.sparse_indexer_k_norm.weight",
            "self_attn.sparse_indexer_k_norm.bias",
            "self_attn.sparse_indexer_w.weight",
        ];
        let weights = suffixes
            .iter()
            .map(|s| {
                layout
                    .operands
                    .iter()
                    .find(|w| w.name == format!("model.layers.{layer_id}.{s}"))
                    .with_context(|| format!("CSA operand {s}"))
            })
            .collect::<Result<Vec<_>>>()?;
        for (w, (dtype, shape)) in weights.iter().zip([
            (DType::F32, vec![H]),
            (DType::Bf16, vec![Q, H]),
            (DType::Bf16, vec![KV, H]),
            (DType::Bf16, vec![KV, H]),
            (DType::F32, vec![192]),
            (DType::F32, vec![192]),
            (DType::Bf16, vec![64, H]),
            (DType::Bf16, vec![H, Q]),
            (DType::Bf16, vec![H, H]),
            (DType::Bf16, vec![256, H]),
            (DType::Bf16, vec![256, H]),
            (DType::Bf16, vec![256]),
            (DType::Bf16, vec![256]),
            (DType::Bf16, vec![256]),
            (DType::Bf16, vec![16, H]),
        ]) {
            ensure!(
                w.dtype == dtype && w.shape == shape,
                "CSA weight shape/dtype: {}",
                w.name
            );
        }
        let dense = layer.dense_activation.filter(|_| include_dense)
            .map(|_| DensePlan::prepare(programs, checkpoint, layer_id, rows, u64::MAX, &[]))
            .transpose()?;
        let admission = Admission::new(
            rows,
            &weights
                .iter()
                .map(|w| w.allocation_bytes)
                .chain(dense.as_ref().map(DensePlan::bytes))
                .collect::<Vec<_>>(),
            if rope.is_none() {
                contract.rope_tables.allocation_bytes
            } else {
                0
            },
            available,
            reservations,
        )?;
        for (name, pointers, scalars) in DEFS {
            let s = programs.spec(name)?;
            ensure!(
                s.pointers == pointers
                    && s.scalar_kinds == scalars
                    && s.capacity_rows as usize >= rows.min(4096),
                "CSA ABI/capacity {name}"
            );
            let expected: Vec<(&str, usize)> = match name {
                "step5_index_weight" => vec![("scratch", 16 * 2)],
                "step5_region_selection" => vec![
                    ("normalized", 4 * CAPACITY * 4),
                    ("lengths", 4 * 4),
                    ("values", 4 * 512 * 4),
                    ("ids", 4 * 512 * 4),
                ],
                _ => vec![],
            };
            ensure!(
                s.scratch.len() == expected.len(),
                "CSA scratch names {name}"
            );
            for (key, bytes) in expected {
                ensure!(
                    s.scratch.get(key) == Some(&(bytes as u64 * s.capacity_rows as u64)),
                    "CSA scratch extent {name}/{key}"
                );
            }
        }
        let residual_spec = programs.spec("step5_residual_bf16")?;
        ensure!(
            residual_spec.pointers == ["residual", "delta", "out"]
                && residual_spec.scalar_kinds == b"i"
                && residual_spec.capacity_rows as usize >= TILE
                && residual_spec.scratch.is_empty(),
            "CSA residual ABI"
        );
        for (name, width, outputs) in [
            ("step5_attn_q", H, Q),
            ("step5_attn_kv", H, KV),
            ("step5_attn_gate", H, 64),
            ("step5_attn_out", Q, H),
            ("step5_index_q", H, H),
            ("step5_index_kz", H, 256),
        ] {
            let g = &programs.spec(name)?.geometry;
            ensure!(
                g["width"] == width
                    && g["outputs"] == outputs
                    && g["output_dtype"] == "torch.bfloat16"
                    && g["skinny_rows"] == 0,
                "CSA projection geometry {name}"
            );
        }
        for (name, width, heads, rotary, centered, mean, bias) in [
            ("step5_hidden_norm", H, 1, 0, true, false, false),
            ("step5_csa_q_norm_rope", 192, 64, 64, true, false, false),
            ("step5_csa_k_norm_rope", 192, 4, 64, true, false, false),
            ("step5_index_q_norm", 256, 16, 32, true, false, false),
            ("step5_index_k_norm", 256, 1, 32, false, true, true),
        ] {
            let g = &programs.spec(name)?.geometry;
            ensure!(
                g["width"] == width
                    && g["heads"] == heads
                    && g["rotary"] == rotary
                    && g["zero_centered"] == centered
                    && g["subtract_mean"] == mean
                    && g["bias"] == bias
                    && g["epsilon"] == "runtime_float32",
                "CSA norm geometry {name}"
            );
        }
        let g = &programs.spec("step5_index_weight")?.geometry;
        ensure!(
            g["width"] == H && g["outputs"] == 16 && g["prescale"] == 0.5,
            "CSA index weight geometry"
        );
        let g = &programs.spec("step5_region_compress")?.geometry;
        ensure!(
            g["region_tokens"] == 8 && g["proxy_dim"] == 256,
            "CSA compression geometry"
        );
        for name in ["step5_region_scores", "step5_region_selection"] {
            let g = &programs.spec(name)?.geometry;
            ensure!(
                g["groups"] == 4 && g["capacity"] == CAPACITY,
                "CSA region geometry {name}"
            );
        }
        let g = &programs.spec("step5_region_scores")?.geometry;
        ensure!(
            g["proxy_dim"] == 256 && g["heads_per_group"] == 4,
            "CSA score geometry"
        );
        let g = &programs.spec("step5_region_selection")?.geometry;
        ensure!(
            g["topk"] == 512 && g["region_tokens"] == 8,
            "CSA selection geometry"
        );
        let g = &programs.spec("step5_csa_attention")?.geometry;
        ensure!(
            g["query_heads"] == 64
                && g["kv_heads"] == 4
                && g["page_rows"] == 64
                && g["selection_width"] == 4104
                && g["selection_groups"] == 4
                && g["head_dim"] == 192
                && g["key_strides"] == serde_json::json!([49152, 768, 192])
                && g["value_strides"] == serde_json::json!([49152, 768, 192]),
            "CSA selected attention geometry"
        );
        let g = &programs.spec("step5_head_gate")?.geometry;
        ensure!(
            g["heads"] == 64 && g["dim"] == 192,
            "CSA head gate geometry"
        );
        ensure!(
            residual_spec.geometry["width"] == H,
            "CSA residual geometry"
        );
        let resolved = DEFS
            .iter()
            .map(|(name, pointers, _)| device.run(|| programs.program(name, pointers)))
            .collect::<Result<Vec<_>>>()?;
        let residual = device
            .run(|| programs.program("step5_residual_bf16", &["residual", "delta", "out"]))?;
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
        Self::workspace(device, admission, std::rc::Rc::new(CsaWeights { rope, layer: layer_id, weights,
            programs: resolved, residual, epsilon: contract.norms.input }), ffn, arena)
    }

    /// Fresh metadata and attention/FFN planes, with no checkpoint or program reload.
    pub(super) fn new_lane(&self) -> Result<Self> {
        let ffn = self.ffn.as_ref().map(DenseFfn::new_lane).transpose()?;
        Self::workspace(self.stream.device, self.admission.clone(), self.immutable.clone(), ffn, None)
    }

    fn workspace(device: Device<'a>, admission: Admission,
        immutable: std::rc::Rc<CsaWeights<'a>>, ffn: Option<DenseFfn<'a>>, arena: Option<(&Arena<'a>, usize)>) -> Result<Self> {
        let planes = scratch::planes(device, &admission.planes, arena)?;
        let stream = Stream::new(device)?;
        let rows = admission.rows;
        let owner = Self { stream, ffn, planes, admission, immutable };
        if arena.is_none() {
            for (plane, data) in [
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
                (
                    STARTS,
                    (0..rows.div_ceil(8) as i32)
                        .flat_map(|r| (r * 8).to_le_bytes())
                        .collect(),
                ),
                (
                    COUNTS,
                    (0..rows.div_ceil(8))
                        .flat_map(|_| 8i32.to_le_bytes())
                        .collect(),
                ),
            ] {
                device.run(|| device.library.copy_h2d(owner.planes[plane].buffer, &data))?;
            }
        }
        Ok(owner)
    }
    fn ptr(&self, plane: usize, offset: usize) -> *mut c_void {
        self.planes[plane].buffer.ptr.wrapping_byte_add(offset)
    }
    fn launch(&self, stage: usize, pointers: &[*mut c_void], scalars: &[Scalar]) -> Result<()> {
        self.launch_on(&self.stream, stage, pointers, scalars)
    }
    fn launch_on(
        &self,
        stream: &Stream<'a>,
        stage: usize,
        pointers: &[*mut c_void],
        scalars: &[Scalar],
    ) -> Result<()> {
        stream.device.run(|| {
            // SAFETY: admitted row views and retained layer owners outlive the stream drain.
            unsafe { self.immutable.programs[stage].launch(pointers, scalars, stream.raw) }
        })
    }
    fn norm(
        &self,
        stage: usize,
        input: usize,
        output: usize,
        weight: usize,
        bias: usize,
        start: usize,
        rows: usize,
        width: usize,
        indexer: bool,
    ) -> Result<()> {
        self.norm_on(
            stage,
            input,
            output,
            weight,
            bias,
            start,
            rows,
            width,
            indexer,
            self.planes[POSITIONS].buffer,
            &self.stream,
        )
    }
    fn norm_on(
        &self,
        stage: usize,
        input: usize,
        output: usize,
        weight: usize,
        bias: usize,
        start: usize,
        rows: usize,
        width: usize,
        indexer: bool,
        positions: CuteafdDeviceBuffer,
        stream: &Stream<'a>,
    ) -> Result<()> {
        let table = if indexer {
            self.immutable.rope.layers[self.immutable.layer]
                .indexer
                .as_ref()
                .context("CSA indexer tables")?
        } else {
            &self.immutable.rope.layers[self.immutable.layer].attention
        };
        let (cos, sin) = table.buffers();
        self.launch_on(
            stream,
            stage,
            &[
                self.ptr(input, start * width * 2),
                self.immutable.weights[weight].buffer.ptr,
                self.immutable.weights[bias].buffer.ptr,
                cos.ptr,
                sin.ptr,
                positions.ptr.wrapping_byte_add(start * 8),
                self.ptr(output, start * width * 2),
            ],
            &[Scalar::I32(rows as i32), Scalar::F32(self.immutable.epsilon)],
        )
    }
    pub fn enqueue(&self, rows: usize, query_start: usize) -> Result<()> {
        ensure!(
            rows > 0 && rows <= self.admission.rows && query_start < rows,
            "CSA live row range"
        );
        self.project_on(rows, self.planes[POSITIONS].buffer, &self.stream)?;
        self.compress(rows)?;
        for start in (query_start..rows).step_by(TILE) {
            let count = (rows - start).min(TILE);
            self.score_select(start, count)?;
            self.attend(start, count)?;
            self.finish(start, count)?;
        }
        self.finish_dense_on(query_start, rows - query_start, &self.stream)
    }
    pub(super) fn project_on(
        &self,
        rows: usize,
        positions: CuteafdDeviceBuffer,
        stream: &Stream<'a>,
    ) -> Result<()> {
        ensure!(
            rows > 0 && rows <= self.admission.rows,
            "CSA projection rows"
        );
        for start in (0..rows).step_by(4096) {
            let count = (rows - start).min(4096);
            let scalar = [Scalar::I32(count as i32)];
            let p = |i| self.ptr(i, start * H * 2);
            let w = |i: usize| self.immutable.weights[i].buffer.ptr;
            self.launch_on(
                stream,
                0,
                &[
                    self.ptr(X, start * H * 4),
                    w(0),
                    w(0),
                    w(0),
                    w(0),
                    w(0),
                    p(NORM),
                ],
                &[scalar[0], Scalar::F32(self.immutable.epsilon)],
            )?;
            for (stage, weight, out, width) in [
                (1, 1, QRAW, Q),
                (2, 2, KRAW, KV),
                (2, 3, VALUE, KV),
                (3, 6, LOGITS, 64),
                (6, 8, IQRAW, H),
                (7, 9, IKRAW, 256),
                (7, 10, IZ, 256),
            ] {
                self.launch_on(
                    stream,
                    stage,
                    &[p(NORM), w(weight), self.ptr(out, start * width * 2)],
                    &scalar,
                )?;
            }
            self.norm_on(
                4, QRAW, QUERY, 4, 4, start, count, Q, false, positions, stream,
            )?;
            self.norm_on(
                5, KRAW, KEY, 5, 5, start, count, KV, false, positions, stream,
            )?;
            self.norm_on(
                8, IQRAW, IQ, 11, 11, start, count, H, true, positions, stream,
            )?;
            self.norm_on(
                9, IKRAW, IK, 12, 13, start, count, 256, true, positions, stream,
            )?;
            self.launch_on(
                stream,
                10,
                &[
                    p(NORM),
                    w(14),
                    self.ptr(IW, start * 16 * 4),
                    self.ptr(IWSCR, start * 16 * 2),
                ],
                &scalar,
            )?;
        }
        Ok(())
    }

    pub fn enqueue_step<'e, E: cuteafd_hostcache::copy::CopyEngine>(
        &'e mut self,
        step: &mut Step<'e, 'a, E>,
        rank: usize,
    ) -> Result<()> {
        step.project(rank, self.immutable.layer, &*self, |layer, input| {
            layer.check_input(&input)?;
            layer.project_on(input.rows, input.positions, input.stream)
        })?;
        step.stage(
            rank,
            self.immutable.layer,
            &self.planes[KEY],
            &self.planes[VALUE],
            Some((&self.planes[IK], &self.planes[IZ])),
        )?;
        step.attend(rank, self.immutable.layer, &*self, |layer, read| layer.read_attention_on(read))?;
        step.compose(rank, &*self, |layer, input| layer.finish_dense_on(0, input.rows, input.stream))
    }

    pub fn enqueue_attention_step<'e, E: cuteafd_hostcache::copy::CopyEngine>(
        &'e self, step: &mut Step<'e, 'a, E>, rank: usize,
    ) -> Result<()> {
        ensure!(self.ffn.is_none(), "shared CSA pipeline must own its FFN separately");
        step.project(rank, self.immutable.layer, self, |layer, input| {
            layer.check_input(&input)?;
            layer.project_on(input.rows, input.positions, input.stream)
        })?;
        step.stage(rank, self.immutable.layer, &self.planes[KEY], &self.planes[VALUE],
            Some((&self.planes[IK], &self.planes[IZ])))?;
        step.attend(rank, self.immutable.layer, self, |layer, read| layer.read_attention_on(read))
    }

    fn check_input(&self, input: &StepInputs<'_, 'a>) -> Result<()> {
        ensure!(
            input.heads == 64
                && input.stream.device.id == self.stream.device.id
                && std::ptr::eq(input.stream.device.library, self.stream.device.library)
                && input.extent <= self.immutable.rope.tables.plan.extent as usize,
            "CSA owner device/head/rotary extent mismatch"
        );
        Ok(())
    }

    fn read_on(&self, read: AttentionRead<'_, 'a>) -> Result<()> {
        let rows = read.inputs.rows;
        let stream = read.inputs.stream;
        self.read_attention_on(read)?;
        self.finish_dense_on(0, rows, stream)
    }

    fn read_attention_on(&self, read: AttentionRead<'_, 'a>) -> Result<()> {
        self.check_input(&read.inputs)?;
        let summary = read
            .summaries
            .context("CSA full-head read requires owned summaries")?;
        ensure!(read.start.checked_add(read.inputs.rows).is_some_and(|end| end <= self.admission.rows),
            "CSA sequence row extent");
        for local in (0..read.inputs.rows).step_by(TILE) {
            let start = read.start + local;
            let count = (read.inputs.rows - local).min(TILE);
            self.score_select_on(
                start,
                count,
                read.inputs.positions,
                read.inputs.requests,
                summary,
                read.inputs.stream,
            )?;
            self.attend_on(
                start,
                count,
                read.inputs.positions,
                read.inputs.requests,
                read.key,
                read.value,
                read.table,
                read.table_width,
                read.pages,
                read.inputs.stream,
            )?;
            self.finish_on(start, count, read.inputs.stream)?;
        }
        Ok(())
    }

    pub fn attention_output(&self) -> CuteafdDeviceBuffer {
        self.planes[POST].buffer
    }

    fn finish_dense_on(&self, start: usize, rows: usize, stream: &Stream<'a>) -> Result<()> {
        if let Some(ffn) = &self.ffn {
            let view = |mut buffer: CuteafdDeviceBuffer| {
                buffer.ptr = buffer.ptr.wrapping_byte_add(start * H * 4);
                buffer.bytes -= start * H * 4;
                buffer
            };
            stream.device.run(|| {
                // SAFETY: disjoint admitted owners remain alive through this stream's drain.
                unsafe {
                    stream.device.library.copy_d2d_async(
                        view(ffn.input()),
                        view(self.planes[POST].buffer),
                        rows * H * 4,
                        stream.raw,
                    )
                }
            })?;
            ffn.enqueue_range_on(start, rows, stream)?;
        }
        Ok(())
    }

    pub fn validate_live_weights(&self, device: Device<'_>) -> Result<()> {
        super::weights::validate_live_weights(device.id, &self.immutable.weights)
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

    fn compress(&self, rows: usize) -> Result<()> {
        let regions = rows / 8;
        if regions == 0 {
            return Ok(());
        }
        self.launch(
            11,
            &[
                self.ptr(IK, 0),
                self.ptr(IZ, 0),
                self.ptr(STARTS, 0),
                self.ptr(COUNTS, 0),
                self.ptr(SUMMARY, 0),
                self.ptr(UNROUNDED, 0),
            ],
            &[Scalar::I32(regions as i32), Scalar::I64(rows as i64)],
        )
    }
    fn score_select(&self, start: usize, rows: usize) -> Result<()> {
        ensure!(
            rows > 0 && rows <= TILE && start + rows <= self.admission.rows,
            "CSA query tile"
        );
        self.score_select_on(
            start,
            rows,
            self.planes[POSITIONS].buffer,
            self.planes[REQUESTS].buffer,
            self.planes[SUMMARY].buffer,
            &self.stream,
        )
    }
    fn score_select_on(
        &self,
        start: usize,
        rows: usize,
        positions: CuteafdDeviceBuffer,
        requests: CuteafdDeviceBuffer,
        summary: CuteafdDeviceBuffer,
        stream: &Stream<'a>,
    ) -> Result<()> {
        ensure!(
            rows > 0
                && rows <= TILE
                && start
                    .checked_add(rows)
                    .is_some_and(|end| end <= self.admission.rows),
            "CSA query tile"
        );
        self.launch_on(
            stream,
            12,
            &[
                self.ptr(IQ, start * H * 2),
                self.ptr(IW, start * 16 * 4),
                summary.ptr,
                requests.ptr.wrapping_byte_add(start * 4),
                positions.ptr.wrapping_byte_add(start * 8),
                self.ptr(SCORES, 0),
            ],
            &[Scalar::I32(rows as i32), Scalar::I64(1)],
        )?;
        self.select_on(start, rows, positions, requests, stream)
    }
    fn select(&self, start: usize, rows: usize) -> Result<()> {
        self.select_on(
            start,
            rows,
            self.planes[POSITIONS].buffer,
            self.planes[REQUESTS].buffer,
            &self.stream,
        )
    }
    fn select_on(
        &self,
        start: usize,
        rows: usize,
        positions: CuteafdDeviceBuffer,
        requests: CuteafdDeviceBuffer,
        stream: &Stream<'a>,
    ) -> Result<()> {
        self.launch_on(
            stream,
            13,
            &[
                self.ptr(SCORES, 0),
                positions.ptr.wrapping_byte_add(start * 8),
                requests.ptr.wrapping_byte_add(start * 4),
                self.ptr(SELECTED, 0),
                self.ptr(NORMALIZED, 0),
                self.ptr(LENGTHS, 0),
                self.ptr(VALUES, 0),
                self.ptr(IDS, 0),
            ],
            &[Scalar::I32(rows as i32)],
        )
    }
    fn attend(&self, start: usize, rows: usize) -> Result<()> {
        self.attend_on(
            start,
            rows,
            self.planes[POSITIONS].buffer,
            self.planes[REQUESTS].buffer,
            self.planes[KEY].buffer,
            self.planes[VALUE].buffer,
            self.planes[TABLE].buffer,
            self.admission.pages,
            self.admission.pages,
            &self.stream,
        )
    }
    fn attend_on(
        &self,
        start: usize,
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
        let q = self.ptr(QUERY, start * Q * 2);
        self.launch_on(
            stream,
            14,
            &[
                q,
                key.ptr,
                value.ptr,
                q,
                q,
                table.ptr,
                requests.ptr.wrapping_byte_add(start * 4),
                self.ptr(SELECTED, 0),
                positions.ptr.wrapping_byte_add(start * 8),
                q,
                q,
                self.ptr(ATTN, start * Q * 2),
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
    fn finish(&self, start: usize, rows: usize) -> Result<()> {
        self.finish_on(start, rows, &self.stream)
    }
    fn finish_on(&self, start: usize, rows: usize, stream: &Stream<'a>) -> Result<()> {
        let scalar = [Scalar::I32(rows as i32)];
        self.launch_on(
            stream,
            15,
            &[
                self.ptr(ATTN, start * Q * 2),
                self.ptr(LOGITS, start * 64 * 2),
                self.ptr(GATED, start * Q * 2),
            ],
            &scalar,
        )?;
        self.launch_on(
            stream,
            16,
            &[
                self.ptr(GATED, start * Q * 2),
                self.immutable.weights[7].buffer.ptr,
                self.ptr(DELTA, start * H * 2),
            ],
            &scalar,
        )?;
        stream.device.run(|| {
            // SAFETY: this tile's residual and delta are retained by the stream-first owner.
            unsafe {
                self.immutable.residual.launch(
                    &[
                        self.ptr(X, start * H * 4),
                        self.ptr(DELTA, start * H * 2),
                        self.ptr(POST, start * H * 4),
                    ],
                    &scalar,
                    stream.raw,
                )
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn fixture(root: &std::path::Path, manifest: &serde_json::Value, key: &str) -> Result<Vec<u8>> {
        use sha2::{Digest, Sha256};
        let bytes = std::fs::read(root.join(format!("{key}.bin")))?;
        ensure!(
            manifest["files"][key]["bytes"] == bytes.len()
                && manifest["files"][key]["sha256"].as_str()
                    == Some(format!("{:x}", Sha256::digest(&bytes)).as_str()),
            "CSA fixture hash/length mismatch"
        );
        Ok(bytes)
    }

    fn values(bytes: &[u8], size: usize) -> Result<Vec<f64>> {
        ensure!(bytes.len() % size == 0, "CSA value alignment");
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
        ensure!(result.iter().all(|v| v.is_finite()), "nonfinite CSA output");
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
        ensure!(a.len() == b.len(), "CSA metric length mismatch");
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

    fn read(chain: &CsaLayer<'_>, plane: usize, offset: usize, bytes: usize) -> Result<Vec<u8>> {
        let source = chain.planes[plane].buffer;
        ensure!(offset + bytes <= source.bytes, "CSA read extent");
        let mut output = vec![0; bytes];
        let buffer = CuteafdDeviceBuffer {
            ptr: source.ptr.wrapping_byte_add(offset),
            bytes: source.bytes - offset,
            ..source
        };
        chain
            .stream
            .device
            .run(|| chain.stream.device.library.copy_d2h(&mut output, buffer))?;
        Ok(output)
    }
    fn upload(chain: &CsaLayer<'_>, plane: usize, offset: usize, bytes: &[u8]) -> Result<()> {
        let source = chain.planes[plane].buffer;
        ensure!(offset + bytes.len() <= source.bytes, "CSA upload extent");
        let buffer = CuteafdDeviceBuffer {
            ptr: source.ptr.wrapping_byte_add(offset),
            bytes: source.bytes - offset,
            ..source
        };
        chain
            .stream
            .device
            .run(|| chain.stream.device.library.copy_h2d(buffer, bytes))
    }
    fn score_bytes(chain: &CsaLayer<'_>, queries: usize, columns: usize) -> Result<Vec<u8>> {
        let mut out = Vec::new();
        for row in 0..queries * 4 {
            out.extend(read(chain, SCORES, row * CAPACITY * 4, columns * 4)?);
        }
        Ok(out)
    }
    fn check_scores(actual: &[u8], expected: &[u8]) -> Result<()> {
        ensure!(actual.len() == expected.len(), "CSA score length");
        let mut a = Vec::new();
        let mut b = Vec::new();
        let mut maximum = 0f64;
        for (actual, expected) in actual.chunks_exact(4).zip(expected.chunks_exact(4)) {
            let x = f32::from_le_bytes(actual.try_into().unwrap());
            let y = f32::from_le_bytes(expected.try_into().unwrap());
            if y.is_infinite() {
                ensure!(x == y, "CSA score visibility mismatch");
            } else {
                ensure!(
                    x.is_finite() && y.is_finite(),
                    "CSA nonfinite visible score"
                );
                maximum = maximum.max((x - y).abs() as f64);
                a.extend(actual);
                b.extend(expected);
            }
        }
        if !a.is_empty() {
            let relative = metrics(&a, &b, 4, 4, a.len() / 4)?;
            ensure!(
                relative <= 1e-5 && maximum <= 1e-4,
                "CSA score error: relative={relative}, max={maximum}"
            );
        }
        Ok(())
    }
    #[test]
    fn csa_admission_bounds_query_workspace_independently_of_prefill() -> Result<()> {
        let short = Admission::new(513, &[1000], 3000, u64::MAX, &[])?;
        let long = Admission::new(4105, &[1000], 3000, u64::MAX, &[])?;
        for i in [SCORES, NORMALIZED, SELECTED, LENGTHS, VALUES, IDS] {
            assert_eq!(short.planes[i], long.planes[i]);
        }
        assert_eq!(short.planes[SCORES], 128 << 20);
        assert_eq!(short.planes[NORMALIZED], 128 << 20);
        assert_eq!(long.pages, 65);
        assert_eq!(long.planes[KEY], 65 * 64 * KV * 2);
        assert_eq!(
            short.bytes,
            short.planes.iter().sum::<usize>() as u64 + 4000
        );
        assert!(Admission::new(0, &[], 0, u64::MAX, &[]).is_err());
        assert!(Admission::new(8193, &[], 0, u64::MAX, &[]).is_err());
        assert!(Admission::new(513, &[1000], 3000, short.bytes - 1, &[]).is_err());
        assert!(Admission::new(1, &[], 0, u64::MAX, &[u64::MAX]).is_err());
        Ok(())
    }
    #[test]
    #[ignore = "requires pinned CSA fixtures, sealed native library and CUDA"]
    fn native_csa_transaction_reads_match_contiguous_producers() -> Result<()> {
        use super::super::{
            cache::CacheRank,
            prefix::{Frontier, Placement, Step5Prefix},
            transaction::{tests::stage_other_layers, Staging},
        };
        use cuteafd_engine::prefix::{PrefixCache, PrefixConfig, PrefixFamily};
        use cuteafd_hostcache::copy::StubCopyEngine;
        use std::cell::RefCell;
        let root = std::path::PathBuf::from(std::env::var("CUTEAFD_STEP5_CSA_FIXTURE_DIR")?);
        let manifest: serde_json::Value =
            serde_json::from_slice(&std::fs::read(root.join("manifest.json"))?)?;
        ensure!(
            manifest["seed"] == 274
                && manifest["layer"] == 3
                && manifest["revision"] == "7734b7ffac118d70d8398bfa3eb0fab007eb45b1",
            "CSA fixture identity"
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
            "CSA snapshot"
        );
        let total = 513;
        let mut chain = CsaLayer::load(
            device,
            &programs,
            &checkpoint,
            3,
            total,
            None,
            u64::MAX,
            &[],
        )?;
        let residual = fixture(&root, &manifest, "r513_residual")?;
        device.run(|| library.copy_h2d(chain.input(), &residual))?;
        chain.enqueue(total, 0)?;
        chain.stream.drain()?;
        let expected = read(&chain, ATTN, 0, total * Q * 2)?;
        let producers = [
            (X, H * 4),
            (QUERY, Q * 2),
            (KEY, KV * 2),
            (VALUE, KV * 2),
            (LOGITS, 64 * 2),
            (IQ, H * 2),
            (IW, 16 * 4),
            (IK, 512),
            (IZ, 512),
        ]
        .into_iter()
        .map(|(plane, row)| Ok((plane, row, read(&chain, plane, 0, total * row)?)))
        .collect::<Result<Vec<_>>>()?;
        let layout = super::super::cache::tests::layout_for(1024, 1, 0, 64);
        let owner = RefCell::new(CacheRank::new(device, layout, 0, 1, 2, 40, usize::MAX)?);
        let staging = Staging::new(device, &programs, &owner.borrow(), usize::MAX)?;
        let family = Step5Prefix::new(vec![&owner])?;
        let pool = RefCell::new(PrefixCache::<StubCopyEngine, Frontier>::new(
            family.layout(),
            PrefixConfig {
                entries: 0,
                mark_slots: 2,
                keep_logits: false,
                min_tokens: 1,
            },
            None,
        )?);
        let mut placement = pool.borrow_mut()
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
        for end in [
            7, 8, 9, 63, 64, 128, 192, 256, 320, 384, 448, 511, 512, total,
        ] {
            let base = placement.committed;
            let proposed = (base + 64).min(total);
            let rows = proposed - base;
            let before = placement.pages.clone();
            for abort in [true, false] {
                let mut step = Step::begin(
                    vec![&owner],
                    vec![&staging],
                    &pool,
                    &mut placement,
                    rows,
                )?;
                step.project(0, 3, &chain, |layer, input| {
                    layer.check_input(&input)?;
                    for (plane, row, data) in &producers {
                        device.run(|| {
                            // SAFETY: ordered copies retain source slices; transaction retains the owner.
                            unsafe {
                                library.copy_h2d_ordered(
                                    layer.planes[*plane].buffer,
                                    &data[base * row..proposed * row],
                                    input.stream.raw,
                                )
                            }
                        })?;
                    }
                    Ok(())
                })?;
                step.stage(
                    0,
                    3,
                    &chain.planes[KEY],
                    &chain.planes[VALUE],
                    Some((&chain.planes[IK], &chain.planes[IZ])),
                )?;
                step.attend(0, 3, &chain, |layer, read| layer.read_on(read))?;
                owner.borrow().drain()?;
                ensure!(
                    read(&chain, ATTN, 0, rows * Q * 2)?
                        == expected[base * Q * 2..proposed * Q * 2],
                    "CSA transaction attention differs at {base}..{proposed}"
                );
                if abort {
                    drop(step);
                    ensure!(
                        placement.committed == base && placement.pages == before,
                        "CSA abort visibility"
                    );
                } else {
                    stage_other_layers(&mut step, 3, &key, &value, &ik, &iz)?;
                    step.publish(end - base)?;
                }
            }
        }
        pool.borrow_mut().release(&family, &placement.pages)?;
        device.run(|| library.copy_h2d(chain.input(), &residual[..64 * H * 4]))?;
        chain.enqueue(64, 0)?;
        chain.stream.drain()?;
        let eager = read(&chain, POST, 0, 64 * H * 4)?;
        let mut placement = pool.borrow_mut()
            .admit_cold(&family, 1, 1, |pages| Placement {
                pages,
                sequence: 0,
                committed: 0,
            })?
            .placement;
        let mut step = Step::begin(vec![&owner], vec![&staging], &pool, &mut placement, 64)?;
        chain.enqueue_step(&mut step, 0)?;
        stage_other_layers(&mut step, 3, &key, &value, &ik, &iz)?;
        step.publish(64)?;
        ensure!(
            read(&chain, POST, 0, eager.len())? == eager,
            "CSA production enqueue differs"
        );
        pool.borrow_mut().release(&family, &placement.pages)?;
        let lane = chain.new_lane()?;
        ensure!(std::rc::Rc::ptr_eq(&chain.immutable, &lane.immutable)
            && chain.stream.raw != lane.stream.raw
            && chain.planes.iter().zip(&lane.planes).all(|(a, b)|
                a.buffer.ptr != b.buffer.ptr && a.buffer.bytes == b.buffer.bytes),
            "Step-5 lane must share immutable storage and isolate every stream/plane");
        drop(lane);
        let other = CsaLayer::load(
            device,
            &programs,
            &checkpoint,
            7,
            1,
            Some(std::rc::Rc::clone(&chain.immutable.rope)),
            u64::MAX,
            &[],
        )?;
        ensure!(
            std::rc::Rc::ptr_eq(&other.immutable.rope, &chain.immutable.rope) && other.immutable.layer == 7,
            "CSA shared layer binding"
        );
        Ok(())
    }

    #[test]
    #[ignore = "requires saved layer90/layer91 fixtures, sealed native library and CUDA"]
    fn native_sparse_dense_saved_layer90_component() -> Result<()> {
        let root = std::path::PathBuf::from(std::env::var("CUTEAFD_STEP5_CSA_FIXTURE_DIR")?);
        let manifest: serde_json::Value =
            serde_json::from_slice(&std::fs::read(root.join("manifest.json"))?)?;
        ensure!(
            manifest["layer"] == 91
                && manifest["rows"] == serde_json::json!([640])
                && manifest["revision"] == "364b3b03fed2477aa0953fe855a2c4c8c48ee3cc"
                && manifest["saved_input"]["layer90_sha256"]
                    == "7b5651a0e86ade90a263ace415d9dc8e170ac99a0d026d7a91ed365a06cd56a6",
            "saved-input component identity"
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
            "component snapshot"
        );
        let rows = 640;
        let chain = CsaLayer::load(
            device,
            &programs,
            &checkpoint,
            91,
            rows,
            None,
            u64::MAX,
            &[],
        )?;
        ensure!(chain.has_dense_ffn(), "planned sparse+dense owner");
        assert!(CsaLayer::load(
            device,
            &programs,
            &checkpoint,
            91,
            rows,
            None,
            chain.admission.bytes - 1,
            &[]
        )
        .is_err());
        let load = |name: &str| fixture(&root, &manifest, &format!("r{rows}_{name}"));
        upload(&chain, X, 0, &load("residual")?)?;
        chain.enqueue(rows, 0)?;
        chain.stream.drain()?;
        for (name, plane, width, size) in [
            ("norm", NORM, H, 2),
            ("query", QUERY, Q, 2),
            ("key", KEY, KV, 2),
            ("attention", ATTN, Q, 2),
            ("delta", DELTA, H, 2),
            ("post", POST, H, 4),
        ] {
            eprintln!("layer91 attention boundary={name}");
            ensure!(
                metrics(
                    &read(&chain, plane, 0, rows * width * size)?,
                    &load(name)?,
                    size,
                    size,
                    width
                )? < 0.03,
                "layer91 attention boundary {name}"
            );
        }
        let mut output = vec![0; rows * H * 4];
        device.run(|| library.copy_d2h(&mut output, chain.output()))?;
        ensure!(
            metrics(&output, &load("output")?, 4, 4, H)? < 0.03,
            "layer91 full output"
        );
        chain
            .ffn
            .as_ref()
            .context("dense owner")?
            .qualify_saved_boundaries(&root, &manifest, rows)?;
        // A suffix-only contiguous query must neither read nor overwrite prefix FFN rows.
        upload(&chain, X, 0, &load("residual")?)?;
        chain.enqueue(rows, 0)?;
        chain.stream.drain()?;
        device.run(|| library.copy_d2h(&mut output, chain.output()))?;
        chain.enqueue(rows, rows - 7)?;
        chain.stream.drain()?;
        let mut suffix = vec![0; output.len()];
        device.run(|| library.copy_d2h(&mut suffix, chain.output()))?;
        ensure!(suffix == output, "dense suffix view changed output");
        Ok(())
    }

    #[test]
    #[ignore = "requires pinned layer-3 CSA fixtures, sealed native library and CUDA"]
    fn native_csa_layer3_boundaries_and_graph_replay() -> Result<()> {
        let root = std::path::PathBuf::from(std::env::var("CUTEAFD_STEP5_CSA_FIXTURE_DIR")?);
        let manifest: serde_json::Value =
            serde_json::from_slice(&std::fs::read(root.join("manifest.json"))?)?;
        ensure!(
            manifest["schema"] == 1
                && manifest["seed"] == 274
                && manifest["layer"] == 3
                && manifest["checkpoint"] == "stepfun-ai/Step-5-Preview-BF16"
                && manifest["revision"] == "7734b7ffac118d70d8398bfa3eb0fab007eb45b1"
                && manifest["rows"] == serde_json::json!([1, 7, 8, 9, 128, 513, 4105])
                && manifest["query_tile"] == TILE,
            "CSA fixture identity"
        );
        // SAFETY: this explicit qualification supplies a sealed matching native library.
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
            "CSA snapshot revision"
        );
        let mut chain =
            CsaLayer::load(device, &programs, &checkpoint, 3, 4105, None, u64::MAX, &[])?;
        ensure!(
            manifest["epsilon"].as_f64().map(|v| v as f32) == Some(chain.immutable.epsilon),
            "CSA epsilon"
        );
        assert!(CsaLayer::load(
            device,
            &programs,
            &checkpoint,
            3,
            4105,
            None,
            chain.admission.bytes - 1,
            &[]
        )
        .is_err());
        assert!(chain.enqueue(4106, 0).is_err());
        assert!(chain.score_select(0, TILE + 1).is_err());
        for rows in [1usize, 7, 8, 9, 128, 513, 4105] {
            let start = if rows == 4105 { 4104 } else { 0 };
            let count = rows - start;
            let load = |name: &str| fixture(&root, &manifest, &format!("r{rows}_{name}"));
            upload(&chain, X, 0, &load("residual")?)?;
            chain.enqueue(rows, start)?;
            chain.stream.drain()?;
            let mut eager = Vec::new();
            for (name, plane, width, size, offset, length) in [
                ("norm", NORM, H, 2, 0, rows),
                ("qraw", QRAW, Q, 2, 0, rows),
                ("kraw", KRAW, KV, 2, 0, rows),
                ("value", VALUE, KV, 2, 0, rows),
                ("query", QUERY, Q, 2, 0, rows),
                ("key", KEY, KV, 2, 0, rows),
                ("iqraw", IQRAW, H, 2, 0, rows),
                ("ikraw", IKRAW, 256, 2, 0, rows),
                ("iz", IZ, 256, 2, 0, rows),
                ("iq", IQ, H, 2, 0, rows),
                ("ik", IK, 256, 2, 0, rows),
                ("iw", IW, 16, 4, 0, rows),
                ("logits", LOGITS, 64, 2, start, count),
                ("attention", ATTN, Q, 2, start, count),
                ("gated", GATED, Q, 2, start, count),
                ("delta", DELTA, H, 2, start, count),
                ("post", POST, H, 4, start, count),
            ] {
                let actual = read(&chain, plane, offset * width * size, length * width * size)?;
                eprintln!("CSA chained rows={rows} boundary={name}");
                metrics(&actual, &load(name)?, size, size, width)?;
                if name == "norm" {
                    super::super::norm_accuracy::check(
                        &actual,
                        &load(name)?,
                        &load("norm_fp64")?,
                        width,
                    )?;
                }
                eager.push((plane, offset * width * size, actual));
            }
            eager.push((SUMMARY, 0, read(&chain, SUMMARY, 0, rows / 8 * 256)?));
            let tile_count = count.min(TILE);
            let last_start = start + (count - 1) / TILE * TILE;
            let last_count = rows - last_start;
            eager.push((
                SELECTED,
                0,
                read(&chain, SELECTED, 0, last_count * SELECT * 4)?,
            ));
            let eager_scores = score_bytes(&chain, last_count, (rows / 8).max(1))?;
            device.run(|| {
                // SAFETY: the stream is idle and the complete owner is pinned below.
                unsafe { library.cuda_graph_begin_capture(chain.stream.raw) }
            })?;
            let enqueue = chain.enqueue(rows, start);
            let raw = device.run(|| {
                // SAFETY: end capture even when the enqueue returns an error.
                unsafe { library.cuda_graph_end_capture(chain.stream.raw) }
            })?;
            // SAFETY: the mutable borrow pins every stream, weight, table and workspace.
            let graph = unsafe {
                crate::shared::decode_graph::GraphOwner::new(&library, device.id, raw, &mut chain)?
            };
            enqueue?;
            let mut guard = graph.pins.stream.cancellation_guard("CSA graph replay");
            device.run(|| {
                // SAFETY: the graph's borrowed owner remains live through the stream drain.
                unsafe { library.cuda_graph_launch(graph.raw, graph.pins.stream.raw) }
            })?;
            graph.pins.stream.drain()?;
            guard.complete = true;
            drop(guard);
            for (plane, offset, expected) in eager {
                ensure!(
                    read(graph.pins, plane, offset, expected.len())? == expected,
                    "CSA eager/replay plane {plane}"
                );
            }
            ensure!(
                score_bytes(graph.pins, last_count, (rows / 8).max(1))? == eager_scores,
                "CSA score eager/replay"
            );
            drop(graph);
            // Every projection consumes the official same-BF16 producer for its FP64 gate.
            for (stage, weight, plane, name, width) in [
                (1, 1, QRAW, "qraw", Q),
                (2, 2, KRAW, "kraw", KV),
                (2, 3, VALUE, "value", KV),
                (3, 6, LOGITS, "logits", 64),
                (6, 8, IQRAW, "iqraw", H),
                (7, 9, IKRAW, "ikraw", 256),
                (7, 10, IZ, "iz", 256),
                (16, 7, DELTA, "delta", H),
            ] {
                let input = if stage == 16 { GATED } else { NORM };
                upload(
                    &chain,
                    input,
                    if stage == 16 { start * Q * 2 } else { 0 },
                    &load(if stage == 16 { "gated" } else { "norm" })?,
                )?;
                chain.launch(
                    stage,
                    &[
                        chain.ptr(
                            input,
                            if stage == 16 {
                                start * Q * 2
                            } else {
                                start * H * 2
                            },
                        ),
                        chain.immutable.weights[weight].buffer.ptr,
                        chain.ptr(plane, start * width * 2),
                    ],
                    &[Scalar::I32(count as i32)],
                )?;
                chain.stream.drain()?;
                let actual = read(&chain, plane, start * width * 2, count * width * 2)?;
                let expected = load(name)?;
                let expected = if expected.len() == rows * width * 2 {
                    &expected[start * width * 2..]
                } else {
                    &expected
                };
                let fp64 = load(&format!("{name}_fp64"))?;
                eprintln!("CSA isolated rows={rows} projection={name}");
                let native = metrics(&actual, &fp64, 2, 8, width)?;
                let official = metrics(expected, &fp64, 2, 8, width)?;
                ensure!(native <= official * 1.01, "CSA projection FP64 bar {name}");
            }
            for (stage, input, out, weight, bias, width, indexer, name, source) in [
                (4, QRAW, QUERY, 4, 4, Q, false, "query", "qraw"),
                (5, KRAW, KEY, 5, 5, KV, false, "key", "kraw"),
                (8, IQRAW, IQ, 11, 11, H, true, "iq", "iqraw"),
                (9, IKRAW, IK, 12, 13, 256, true, "ik", "ikraw"),
            ] {
                upload(&chain, input, 0, &load(source)?)?;
                for first in (0..rows).step_by(4096) {
                    chain.norm(
                        stage,
                        input,
                        out,
                        weight,
                        bias,
                        first,
                        (rows - first).min(4096),
                        width,
                        indexer,
                    )?;
                }
                chain.stream.drain()?;
                let actual = read(&chain, out, 0, rows * width * 2)?;
                let expected = load(name)?;
                metrics(&actual, &expected, 2, 2, width)?;
                super::super::norm_accuracy::check(
                    &actual,
                    &expected,
                    &load(&format!("{name}_fp64"))?,
                    if indexer { 256 } else { 192 },
                )?;
            }
            upload(&chain, NORM, 0, &load("norm")?)?;
            chain.launch(
                10,
                &[
                    chain.ptr(NORM, start * H * 2),
                    chain.immutable.weights[14].buffer.ptr,
                    chain.ptr(IW, start * 16 * 4),
                    chain.ptr(IWSCR, start * 16 * 2),
                ],
                &[Scalar::I32(count as i32)],
            )?;
            chain.stream.drain()?;
            let expected = load("iw")?;
            let fp64 = load("iw_fp64")?;
            let native = metrics(
                &read(&chain, IW, start * 16 * 4, count * 16 * 4)?,
                &fp64,
                4,
                8,
                16,
            )?;
            let official = metrics(&expected[start * 16 * 4..], &fp64, 4, 8, 16)?;
            ensure!(
                native <= official * 1.01,
                "CSA index weight projection FP64 bar"
            );
            upload(&chain, IK, 0, &load("ik")?)?;
            upload(&chain, IZ, 0, &load("iz")?)?;
            // Sentinels prove the final incomplete region is not summarized.
            upload(&chain, SUMMARY, 0, &vec![0x38; (rows / 8 + 1) * 256])?;
            chain.compress(rows)?;
            chain.stream.drain()?;
            ensure!(
                read(&chain, SUMMARY, 0, rows / 8 * 256)? == load("summary")?,
                "CSA isolated E4M3 summary"
            );
            ensure!(
                read(&chain, SUMMARY, rows / 8 * 256, 256)? == vec![0x38; 256],
                "CSA partial region changed"
            );
            upload(&chain, IQ, 0, &load("iq")?)?;
            upload(&chain, IW, 0, &load("iw")?)?;
            let expected_scores = load("scores")?;
            let columns = (rows / 8).max(1);
            let expected_selected = load("selected")?;
            upload(&chain, QUERY, 0, &load("query")?)?;
            upload(&chain, KEY, 0, &load("key")?)?;
            upload(&chain, VALUE, 0, &load("value")?)?;
            for offset in (0..count).step_by(TILE) {
                let live = (count - offset).min(TILE);
                let position = start + offset;
                chain.score_select(position, live)?;
                chain.stream.drain()?;
                let scores = score_bytes(&chain, live, columns)?;
                check_scores(
                    &scores,
                    &expected_scores[offset * 4 * columns * 4..(offset + live) * 4 * columns * 4],
                )?;
                ensure!(
                    read(&chain, SELECTED, 0, live * SELECT * 4)?
                        == expected_selected[offset * SELECT * 4..(offset + live) * SELECT * 4],
                    "CSA selected token identity"
                );
                chain.attend(position, live)?;
            }
            chain.stream.drain()?;
            let actual = read(&chain, ATTN, start * Q * 2, count * Q * 2)?;
            let fp64 = load("attention_fp64")?;
            let rounded = load("attention_bf16_p")?;
            eprintln!("CSA isolated rows={rows} attention native/oracle/official");
            let native = metrics(&actual, &fp64, 2, 8, Q)?;
            let oracle = metrics(&rounded, &fp64, 2, 8, Q)?;
            let official = metrics(&load("attention")?, &fp64, 2, 8, Q)?;
            eprintln!(
                "CSA rows={rows} native_oracle_ratio={} native_official_ratio={}",
                native / oracle,
                native / official
            );
            ensure!(native <= oracle * 1.01, "CSA attention BF16-P FP64 bar");
            let a = values(&actual, 2)?;
            let b = values(&rounded, 2)?;
            let reference = values(&fp64, 8)?;
            for (position, ((a, b), r)) in a
                .chunks(Q)
                .zip(b.chunks(Q))
                .zip(reference.chunks(Q))
                .enumerate()
            {
                let error = |v: &[f64]| v.iter().zip(r).map(|(a, b)| (a - b).powi(2)).sum::<f64>();
                ensure!(
                    error(a) <= error(b) * 1.01f64.powi(2),
                    "CSA attention worst-position gate {position}"
                );
            }
            upload(&chain, ATTN, start * Q * 2, &load("attention")?)?;
            upload(&chain, LOGITS, start * 64 * 2, &load("logits")?)?;
            for offset in (0..count).step_by(TILE) {
                chain.finish(start + offset, (count - offset).min(TILE))?;
            }
            chain.stream.drain()?;
            ensure!(
                read(&chain, GATED, start * Q * 2, count * Q * 2)? == load("gated")?,
                "CSA isolated head gate"
            );
            upload(&chain, DELTA, start * H * 2, &load("delta")?)?;
            device.run(|| {
                // SAFETY: all residual operands belong to this admitted same-device owner.
                unsafe {
                    chain.immutable.residual.launch(
                        &[
                            chain.ptr(X, start * H * 4),
                            chain.ptr(DELTA, start * H * 2),
                            chain.ptr(POST, start * H * 4),
                        ],
                        &[Scalar::I32(count as i32)],
                        chain.stream.raw,
                    )
                }
            })?;
            chain.stream.drain()?;
            ensure!(
                read(&chain, POST, start * H * 4, count * H * 4)? == load("post")?,
                "CSA isolated residual"
            );
            eprintln!("CSA rows={rows} queries={count} complete_regions={} passed (query tile {tile_count})",rows/8);
        }
        Ok(())
    }
}
