//! One admitted SparkNest group for native dense weights and all expert sets.
//! Loaded targets remain private until every coordinator and Spark rank commits.
use crate::shared::{
    experts::rtx::{dense, push::{CoordinatorExperts, CoordinatorPush, DensePlan, PushedExperts, PushPolicy}},
    memory::{device::Device, push_exl3::CoordinatorRanks},
};
use anyhow::{ensure, Context, Result};
use cuteafd_loader::{plan::checkpoint::CheckpointView, push_load::PushHostEnvelope, OfficialV41Catalog};

pub(crate) type Membership = (Vec<std::net::TcpStream>, CoordinatorRanks);

pub(crate) struct Prepared<'a, T> {
    push: CoordinatorPush,
    owners: T,
    devices: Vec<Device<'a>>,
    sparks: usize,
    dense: (Vec<DensePlan>, Vec<dense::DenseArena<'a>>),
}

/// The caller has solved the complete host/device recipe before invoking this.
/// Each expert set is an independently retired onboarding owner. Spark streams
/// join this same group through `Admission::start_with`, never a second load.
/// Mount snapshots use the ordinary family loader instead of this entry point.
#[allow(clippy::too_many_arguments)]
pub(crate) fn prepare<'a, T>(view: &CheckpointView, catalog: &OfficialV41Catalog,
    devices: &[Device<'a>], sparks: usize, expert_sets: Vec<CoordinatorExperts>,
    envelope: PushHostEnvelope, load: impl FnOnce() -> Result<T>)
    -> Result<(Prepared<'a, T>, Option<Membership>)> {
    ensure!(PushPolicy::for_snapshot(view.snapshot())? == PushPolicy::Required,
        "MiniMax grouped push needs a SparkNest namespace");
    ensure!(matches!(devices.len(), 1 | 2) && devices.iter().enumerate().all(|(rank, device)|
        std::ptr::eq(device.library, devices[0].library)
            && devices[..rank].iter().all(|other| other.id != device.id)),
        "MiniMax push needs distinct coordinator devices in one library");
    let (owners, records) = dense::recording(load)?;
    let dense = dense::plan_records(view.snapshot(), records, devices, sparks)?;
    let experts = (!expert_sets.is_empty()).then_some(CoordinatorExperts::Sets(expert_sets));
    let (push, membership) = if sparks == 0 {
        (CoordinatorPush::prepare_alone_view_admitted(view, Some(catalog), devices.len(), experts, envelope)?, None)
    } else {
        let (push, streams, ranks) = CoordinatorPush::prepare_view_admitted(view, Some(catalog),
            devices.len(), experts, envelope)?;
        (push, Some((streams, ranks)))
    };
    Ok((Prepared { push, owners, devices: devices.to_vec(), sparks, dense }, membership))
}

impl<'a, T> Prepared<'a, T> {
    /// Call while awaiting the Spark admission transaction. No initialized owner
    /// escapes a declined, failed, or partially committed group.
    pub(crate) fn receive(self, catalog: &OfficialV41Catalog) -> Result<(T, Vec<Vec<PushedExperts<'a>>>)> {
        let Self { push, owners, devices, sparks, dense } = self;
        let (received, ()) = push.receive_sets_with(devices, Some(catalog), sparks, Some(Ok(dense)), || ());
        let experts = PushPolicy::Required.committed(received)?
            .context("MiniMax group did not publish coordinator ranks")?;
        Ok((owners, experts))
    }
}
