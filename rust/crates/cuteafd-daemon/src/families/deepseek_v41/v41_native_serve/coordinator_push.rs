//! V4.1 coordinator ranks receive device-resident dense operands and onboarded
//! experts in one committed group. Namespace launches require push; planning or
//! delivery failure never permits a mounted reread. Intentional host-mapped
//! embedding and Engram tables are not coordinator device operands.
use crate::shared::experts::rtx::{dense::{self, Records}, push::{CoordinatorPush, PushedExperts, PushPolicy}};
use crate::shared::memory::device::Device;
use anyhow::{Context, Result};
use cuteafd_loader::OfficialV41Catalog;

/// Whether this launch may push its coordinator ranks (records are taken
/// while loading; unused records then fill through the mount).
pub(super) fn wanted(args: &crate::cli::NativeServeArgs, catalog: &OfficialV41Catalog) -> Result<bool> {
    let policy = PushPolicy::for_snapshot(&args.snapshot)?;
    policy.plan(|| {
        anyhow::ensure!(!args.peers.is_empty() && catalog.exl3().is_none() && catalog.nvfp4().is_none()
            && catalog.fp8().is_none() && !args.tp2_query_projection && !args.tp2_output_projection
            && !args.tp2_dspark_experts, "V4.1 coordinator push does not support this weight placement yet");
        Ok(())
    }).map(|planned| planned.is_some()).map_err(Into::into)
}

/// The coordinator ranks the Spark admission starts with, or none (the
/// coordinator then fills its records through the mount): planned from the
/// recorded dense operands and the admitted TP2 halves, before membership.
pub(super) struct Planned<'a> {
    push: CoordinatorPush,
    dense: Option<(Vec<crate::shared::experts::rtx::push::DensePlan>, Vec<dense::DenseArena<'a>>)>,
    sparks: usize,
}

#[allow(clippy::type_complexity)]
pub(super) fn plan<'a>(snapshot: &std::path::Path, catalog: &OfficialV41Catalog, devices: [Device<'a>; 2],
    records: &Records, tp2_layers: usize, tp2_budgets: [usize; 2], sparks: usize)
    -> Result<(Planned<'a>, Vec<std::net::TcpStream>, crate::shared::memory::push_exl3::CoordinatorRanks)> {
    let dense = dense::plan_records(snapshot, records.clone(), &devices, sparks)?;
    let layers = (0..tp2_layers).collect::<Vec<_>>();
    let (push, streams, ranks) = CoordinatorPush::prepare(snapshot, Some(catalog), 2,
        (tp2_layers > 0).then(|| crate::shared::experts::rtx::push::CoordinatorExperts::Tp2 { layers: layers.to_vec(), budgets: tp2_budgets.to_vec() }))?;
    Ok((Planned { push, dense: Some(dense), sparks }, streams, ranks))
}

impl<'a> Planned<'a> {
    /// Receives both GPUs' ranks (dense and TP2 halves) while the Sparks load.
    /// `Ok(None)`: the whole group fell back; the caller fills `records` and
    /// loads its TP2 halves through the mount.
    pub(super) fn receive(self, devices: [Device<'a>; 2], catalog: &OfficialV41Catalog)
        -> Result<Option<[PushedExperts<'a>; 2]>> {
        let Self { push, dense, sparks } = self;
        let started = std::time::Instant::now();
        let (pushed, ()) = push.receive_with(devices.to_vec(), Some(catalog), sparks, dense.map(Ok), || ());
        let pushed = PushPolicy::Required.committed(pushed)?;
        tracing::info!(pushed = pushed.is_some(), elapsed_ms = started.elapsed().as_millis(),
            "V4.1 coordinator push ranks received");
        pushed.map(|ranks| <[_; 2]>::try_from(ranks).map_err(|_| anyhow::anyhow!("two coordinator GPUs"))).transpose()
    }
}

/// One GPU's dense and whole-layer payload; with no Sparks this owns the driver.
pub(super) struct Whole<'a> {
    push: CoordinatorPush,
    dense: (Vec<crate::shared::experts::rtx::push::DensePlan>, Vec<dense::DenseArena<'a>>),
    sparks: usize,
}

#[allow(clippy::type_complexity)]
pub(super) fn plan_whole<'a>(snapshot: &std::path::Path, catalog: &OfficialV41Catalog,
    device: Device<'a>, records: &Records,
    selected: &[(crate::shared::experts::layer::ExpertLayer, usize)], sparks: usize)
    -> Result<(Whole<'a>, Option<(Vec<std::net::TcpStream>, crate::shared::memory::push_exl3::CoordinatorRanks)>)> {
    let dense = dense::plan_records(snapshot, records.clone(), &[device], sparks)?;
    let budget = selected.iter().try_fold(0usize, |sum, (_, bytes)| sum.checked_add(*bytes))
        .context("V4.1 whole expert resident bytes overflow")?;
    let experts = (!selected.is_empty()).then(|| crate::shared::experts::rtx::push::CoordinatorExperts::Whole {
        selections: selected.iter().map(|(selection, _)| *selection).collect(), budget,
    });
    let (push, ranks) = if sparks == 0 {
        (CoordinatorPush::prepare_alone(snapshot, Some(catalog), 1, experts)?, None)
    } else {
        let (push, streams, ranks) = CoordinatorPush::prepare(snapshot, Some(catalog), 1, experts)?;
        (push, Some((streams, ranks)))
    };
    Ok((Whole { push, dense, sparks }, ranks))
}

impl<'a> Whole<'a> {
    pub(super) fn receive(self, device: Device<'a>, catalog: &OfficialV41Catalog,
        selected: &[(crate::shared::experts::layer::ExpertLayer, usize)], admitted: usize)
        -> Result<Vec<crate::shared::experts::layer::ExpertWeights<'a>>> {
        let Self { push, dense, sparks } = self;
        let (received, ()) = push.receive_with(vec![device], Some(catalog), sparks, Some(Ok(dense)), || ());
        let mut ranks = PushPolicy::Required.committed(received)?
            .context("V4.1 required whole payload")?;
        anyhow::ensure!(ranks.len() == 1, "V4.1 whole payload requires one coordinator rank");
        ranks.remove(0).whole_native(selected, admitted)
    }
}

/// Fills `records` through the mount on their GPUs (default streams).
pub(super) fn fill(library: &cuteafd_ffi::NativeLibrary, records: &Records) -> Result<()> {
    let started = std::time::Instant::now();
    dense::fill_records_by_mount(library, records, |_| std::ptr::null_mut())
        .context("V4.1 coordinator dense mount fill")?;
    let (bytes, operands, staged, steps, largest, finishes) = records.summary();
    tracing::info!(bytes, operands, staged, steps, largest, finishes, elapsed_ms = started.elapsed().as_millis(),
        "V4.1 coordinator dense records filled through the mount");
    Ok(())
}
