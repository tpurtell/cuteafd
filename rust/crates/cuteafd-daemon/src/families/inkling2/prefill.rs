//! Shared executor adapter for one cold native golden layer; no private loop.
use super::{experts::Dispatcher, layers::{self, Kernels}, owners::UnitOwner, weights::TargetWeights};
use crate::shared::prefill_pipeline::{run_prefill_units, sequence_units, FfnRoute, PrefillHooks, Unit};
use anyhow::{bail, Result};
use cuteafd_loader::families::inkling2::Inkling2Config;

pub(crate) fn layer(kernels: &Kernels<'_>, owner: &UnitOwner<'_>, target: &TargetWeights<'_>,
    experts: &dyn Dispatcher, cfg: &Inkling2Config, index: usize, rows: usize) -> Result<()> {
    struct Hooks<'b, 'a> {
        kernels: &'b Kernels<'a>, owner: &'b UnitOwner<'a>, target: &'b TargetWeights<'a>,
        experts: &'b dyn Dispatcher, cfg: &'b Inkling2Config,
    }
    impl PrefillHooks for Hooks<'_, '_> {
        type Ticket = ();
        fn route(&self, _: usize) -> FfnRoute { FfnRoute::Local }
        fn begin_unit(&mut self, _: &Unit) -> Result<()> { Ok(()) }
        fn attention(&mut self, layer: usize, unit: &Unit) -> Result<()> {
            layers::begin_layer(self.kernels, self.owner, self.target, self.cfg, layer, unit.rows.len())
        }
        fn local_ffn(&mut self, layer: usize, unit: &Unit) -> Result<()> {
            layers::finish_layer(self.kernels, self.owner, self.target, self.experts, self.cfg, layer, unit.rows.len())
        }
        fn send_experts(&mut self, _: usize, _: &Unit) -> Result<()> { bail!("cold Inkling golden has no Spark route") }
        fn land_experts(&mut self, _: usize, _: &Unit, _: ()) -> Result<()> { bail!("cold Inkling golden has no Spark ticket") }
        fn end_unit(&mut self, _: &Unit) -> Result<()> { Ok(()) }
        fn commit_unit(&mut self, _: &Unit) -> Result<()> { Ok(()) }
    }
    let mut units = sequence_units(rows, owner.layout.rows, 1, cfg.layers)?;
    for unit in &mut units { unit.layers = index..index+1; }
    run_prefill_units(&mut Hooks { kernels, owner, target, experts, cfg }, &units, 1)
}
