//! Family arithmetic over caller-owned native producers and expert dispatch.
use super::{experts::Dispatcher, owners::UnitOwner, weights::TargetWeights};
use anyhow::{ensure, Context, Result};
use cuteafd_ffi::{draft_projection::{DraftProjections, ProjectionOutput}, inkling2::{Attention, Kind, Route},
    inkling2::producers::{Op, Precision, Producer}, programs::Programs, CuteafdDeviceBuffer as Buffer};
use cuteafd_loader::families::inkling2::{execution::ResidualSite, Inkling2Config};
use std::collections::BTreeMap;

pub(crate) struct Kernels<'a> {
    pub projections: DraftProjections<'a>,
    pub local: Attention<'a>,
    pub global: Attention<'a>,
    producers: BTreeMap<String, Producer<'a>>,
    pub add: cuteafd_ffi::F32Add<'a>,
}
impl<'a> Kernels<'a> {
    pub fn bind(programs: &'a Programs<'a>) -> Result<Self> {
        let mut ops = Vec::new();
        for precision in [Precision::Bf16, Precision::F32] {
            for (width, dynamic) in [(4608, false), (1024, false), (1024, true)] {
                ops.push(Op::Convolution { width, dynamic, precision });
            }
            for (relative, key, global) in [(false, false, false), (false, true, false),
                (true, false, false), (false, false, true), (true, false, true)] {
                ops.push(Op::Clip { relative, key, global, precision });
            }
            for situ in [false, true] { ops.push(Op::Gate { situ, precision }); }
            for value in [false, true] { ops.push(Op::Norm { value, precision }); }
            ops.push(Op::KvPack { precision });
        }
        for queries in [1, 3] {
            for addend in [false, true] { ops.push(Op::Residual { queries, addend }); }
        }
        ops.push(Op::Router);
        let mut producers = BTreeMap::new();
        for op in ops { producers.insert(op.name()?, Producer::load(programs, op)?); }
        Ok(Self { projections: DraftProjections::load_with_output(programs, &[
            (4608, 6144), (4608, 768), (4608, 1024), (4608, 48), (4608, 32),
            (4608, 384), (6144, 4608), (64, 1024)], 4096, ProjectionOutput::F32)?,
            local: Attention::load(programs, Kind::Local, Route::Prefill)?,
            global: Attention::load(programs, Kind::Global, Route::Prefill)?,
            add: programs.library().f32_add()?, producers })
    }
    fn producer(&self, op: Op) -> Result<&Producer<'a>> {
        self.producers.get(&op.name()?).context("Inkling producer not bound at startup")
    }
    pub fn norm(&self, owner: &UnitOwner<'_>, input: Buffer, weight: Buffer, output: Buffer,
        value: bool, precision: Precision, rows: usize) -> Result<()> {
        // SAFETY: the family supplies disjoint admitted typed input/output views;
        // retained unit stream orders the producer and all consuming projections.
        unsafe { self.producer(Op::Norm { value, precision })?.launch(&[input, weight, output],
            rows, None, owner.stream.raw) }
    }
    pub fn convolution(&self, owner: &UnitOwner<'_>, input: Buffer, weight: Buffer, dynamic: Buffer,
        output: Buffer, history: &str, width: usize, global: bool, rows: usize) -> Result<()> {
        // SAFETY: native F32 planes and BF16 taps are retained by unit/weights;
        // metadata addresses the unit's exclusive history and valid packed rows.
        unsafe { self.producer(Op::Convolution { width, dynamic: global, precision: Precision::F32 })?
            .launch(&[input, weight, dynamic, owner.buffer(history)?, owner.buffer("first")?,
                owner.buffer("ring_slots")?, output], rows, None, owner.stream.raw) }
    }
    pub fn residual(&self, owner: &UnitOwner<'_>, query: Buffer, addend: Buffer, output: Buffer,
        rows: usize, queries: usize, candidates: usize, add: bool) -> Result<()> {
        // SAFETY: caller initialized each candidate in the F32 arena; the query
        // and optional addend are retained through the mix on the unit stream.
        unsafe { self.producer(Op::Residual { queries, addend: add })?.launch(&[
            owner.buffer("residual_arena")?, addend, query, output], rows,
            Some((owner.layout.rows, candidates)), owner.stream.raw) }
    }
    pub fn route(&self, owner: &UnitOwner<'_>, logits: Buffer, bias: Buffer, rows: usize) -> Result<()> {
        // SAFETY: native F32 logits/bias and output routes have admitted extents.
        unsafe { self.producer(Op::Router)?.launch(&[logits, bias, owner.buffer("route_ids")?,
            owner.buffer("route_weights")?], rows, None, owner.stream.raw) }
    }
    pub fn project(&self, owner: &UnitOwner<'_>, input: Buffer, weight: Buffer, output: Buffer,
        rows: usize, k: usize, n: usize) -> Result<()> {
        ensure!(input.bytes >= rows*k*2 && weight.bytes >= n*k*2 && output.bytes >= rows*n*4,
            "Inkling projection exceeds admitted views");
        // SAFETY: checked BF16 input/weight extents and F32 destination; arena
        // and unit retain every pointer on the preloaded shared projection stream.
        unsafe { self.projections.launch_checked(input.ptr, weight.ptr, output, rows, k, n, owner.stream.raw)?; }
        Ok(())
    }
}

/// Route/MLP stage has no knowledge of checkpoint expert format or transport.
pub(crate) fn routed_ffn(kernels: &Kernels<'_>, owner: &UnitOwner<'_>, target: &TargetWeights<'_>,
    experts: &dyn Dispatcher, cfg: &Inkling2Config, layer: usize, rows: usize) -> Result<()> {
    ensure!(layer < cfg.layers && rows > 0 && rows <= owner.layout.rows, "Inkling FFN layer/rows differ");
    let prefix = format!("model.llm.layers.{layer}");
    kernels.project(owner, owner.buffer("projection_input")?, target.operand(&format!("{prefix}.mlp.gate.weight"))?.buffer,
        owner.buffer("route_logits")?, rows, cfg.hidden, cfg.experts)?;
    kernels.route(owner, owner.buffer("route_logits")?, target.operand(&format!("{prefix}.mlp.gate.bias"))?.buffer, rows)?;
    experts.enqueue(owner.buffer("expert_wire")?, owner.buffer("route_ids")?, owner.buffer("route_weights")?,
        owner.buffer("expert_output")?, rows, &owner.stream)
}

pub(crate) fn residual_site(cfg: &Inkling2Config, layer: usize) -> Result<ResidualSite> {
    ResidualSite::layer(cfg, layer)
}
