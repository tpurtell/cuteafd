//! What the Muse coordinator tells its Spark workers, derived from the solved
//! placement and the shared endpoint rule (plan == runtime): the Spark working
//! set (layers whose experts live on the Sparks), the sealed `ModelAdmission`
//! (one endpoint shared by decode, verify and prefill, its partial-reply frame
//! and depth, the typed owners), and the coordinator transport's frame budget.
use anyhow::{ensure, Context, Result};
use cuteafd_core::ExpertGeometry;
use cuteafd_loader::families::muse_spark::replies::Cutoff;
use cuteafd_loader::placement::{families::muse_spark as shared, ExpertHome, Placement, SparkWorkingSet};
use cuteafd_loader::serving_capacity::prefill_waves::PrefillPlan;
use cuteafd_transport::worker_selection::ModelAdmission;

/// The Spark side of one admitted placement.
#[derive(Debug, Clone)]
pub(crate) struct SparkPlan {
    pub working: SparkWorkingSet,
    pub model: ModelAdmission,
    /// The coordinator transport's frame budget: the endpoint's own frame.
    pub max_frame_bytes: usize,
    /// First layer whose experts live on the Sparks (the bootstrap layer).
    pub first_spark_layer: usize,
}

pub(crate) fn spark_plan(placement: &Placement, prefill: PrefillPlan, cutoff: Cutoff, sequences: usize,
    spark_ranks: usize) -> Result<SparkPlan> {
    let geometry = ExpertGeometry::MUSE_SPARK;
    let layers_of = |home: fn(&ExpertHome) -> bool| placement.layers.iter().enumerate()
        .filter_map(|(layer, assignment)| home(&assignment.experts).then_some(layer)).collect::<Vec<_>>();
    let spark_layers = layers_of(|home| *home == ExpertHome::Spark);
    let rtx_layers = layers_of(|home| matches!(home, ExpertHome::RtxTp2 | ExpertHome::RtxWhole { .. }));
    ensure!(spark_layers.len() + rtx_layers.len() == placement.layers.len(),
        "Muse places every layer's experts on the Sparks or an RTX (all layers are MoE)");
    ensure!(rtx_layers.len() == placement.onboard_layers, "Muse onboarded layers differ from the placement count");
    let first_spark_layer = *spark_layers.first().context("Muse serves with at least one Spark layer")?;
    let (endpoint, rings) = shared::spark_endpoint(prefill, cutoff, sequences)?;
    let max_frame_bytes = endpoint.frame_bytes(geometry, rings).context("Muse Spark endpoint frame")?;
    let rings = cuteafd_core::SparkRings { max_frame_bytes, ..rings };
    let executions = shared::spark_executions(prefill.waves, prefill.unit_rows)?;
    let model = ModelAdmission::new(vec![endpoint], rings, executions)?;
    model.validate(geometry)?;
    let working = SparkWorkingSet { rtx_layers, spark_layers, spark_ranks, prefill_lanes: shared::LANES };
    Ok(SparkPlan { working, model, max_frame_bytes, first_spark_layer })
}

#[cfg(test)]
mod tests {
    use super::*;
    use cuteafd_loader::plan::{layout::LayoutOptions, testing, ExpertPlacement, PlanOptions};

    /// The sealed endpoint's rings are what the planner charged each Spark, the
    /// frame budget covers the worst partial reply, and onboarded layers leave
    /// the Spark working set.
    #[test]
    fn sealed_admission_matches_the_planned_spark_io() -> Result<()> {
        let dir = tempfile::tempdir()?;
        testing::write_muse_spark_snapshot(dir.path(), &(0..62).collect::<Vec<_>>());
        let input = testing::BudgetedPlanInput::open(dir.path())?;
        let total = cuteafd_loader::placement::inventory::PRO_TOTAL_BYTES;
        let report = input.plan(&PlanOptions { placement: ExpertPlacement::from_spark_ranks(6),
            layout: Some(LayoutOptions { host_memory: Some((256 << 30, 256 << 30)), rtx_bytes: vec![total; 2],
                head_split: true, ..Default::default() }), ..Default::default() })?;
        let placement = report.solver_placement.clone().context("placed")?;
        let prefill = PrefillPlan { unit_rows: 2752, waves: 3, span_rows: 8256 };
        let plan = spark_plan(&placement, prefill, Cutoff::new(64, 1)?, 16, 6)?;
        assert_eq!(plan.working.spark_layers.len() + placement.onboard_layers, 62);
        assert!(plan.working.spark_layers.iter().all(|layer| !plan.working.rtx_layers.contains(layer)));
        let endpoint = plan.model.endpoints[0];
        let reply = endpoint.partial_reply_bytes.context("partial endpoint")? as usize;
        assert!(plan.max_frame_bytes >= reply && plan.model.rings.max_frame_bytes == plan.max_frame_bytes);
        let spark = report.memory_layout.as_ref().context("layout")?.devices.iter()
            .find(|d| d.kind == cuteafd_core::memory_layout::DeviceKind::Spark).context("spark")?;
        let rings = spark.items.iter().find(|item| item.group == "rdma rings").context("rings")?.bytes;
        assert_eq!(rings, plan.model.ring_bytes(ExpertGeometry::MUSE_SPARK)? as u64);
        // Step metadata: the runtime's StepStorage extent is the charged item.
        let charged = placement.items[0].iter().find(|item| item.group == "Muse step metadata").context("charge")?.bytes;
        let tables = shared::step_tables(2752, 16, cuteafd_loader::families::muse_spark::MuseSparkConfig::from_hf(
            &testing::muse_spark_config())?.max_position as u64)?;
        assert_eq!(charged, tables.bytes() as u64);
        Ok(())
    }
}
