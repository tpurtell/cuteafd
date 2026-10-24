//! Component selection and per-selected-rank expert transport accounting.
use super::{Demand, PlacementError};
use cuteafd_core::ExpertGeometry;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WorkingSetComponent { Always, RtxExperts, SparkExperts }

#[derive(Debug, Clone)]
pub struct ComponentDemand {
    pub component: WorkingSetComponent,
    pub demand: Demand,
}

impl ComponentDemand {
    pub fn new(component: WorkingSetComponent, gpu: u8,
        category: cuteafd_core::memory_layout::Category, group: &str, bytes: u64,
        basis: cuteafd_core::memory_layout::Basis) -> Self {
        Self { component, demand: Demand::new(gpu, category, group, bytes, basis) }
    }
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct SparkWorkingSet {
    pub rtx_layers: Vec<usize>,
    pub spark_layers: Vec<usize>,
    pub spark_ranks: usize,
    pub prefill_lanes: u64,
}

impl SparkWorkingSet {
    /// Endpoint rings are summed; one GPU owner shares intake and exchange
    /// across all its QPs, so those allocations are maxima, not lane sums.
    pub fn rank_endpoints_io(&self, rank: usize, geometry: ExpertGeometry,
        endpoints: &[cuteafd_core::SparkEndpoint], rings: cuteafd_core::SparkRings)
        -> Result<SparkIoBytes, PlacementError> {
        if !self.selected_ranks().contains(&rank) { return Ok(SparkIoBytes::default()); }
        if endpoints.is_empty() || endpoints.len() > 8 {
            return Err(PlacementError::Inventory("selected Spark needs 1..=8 endpoints"));
        }
        let mut bytes = SparkIoBytes::default();
        for endpoint in endpoints {
            let rings = endpoint.resolved_rings(rings)
                .ok_or(PlacementError::Inventory("invalid Spark endpoint ring depth"))?;
            let ring = SparkRingGeometry { max_frame_bytes: rings.max_frame_bytes, depth: rings.depth,
                slot_bytes: rings.slot_bytes, alignment: rings.alignment, endpoints: 1 };
            let bf16 = endpoint.ingress == cuteafd_core::SparkIngress::Bf16;
            let io = self.rank_io(rank, geometry, endpoint.capacity, bf16, ring)?;
            bytes.intake = bytes.intake.max(io.intake);
            if let cuteafd_core::SparkIngress::Bf16Decode { rows } = endpoint.ingress {
                let decode = self.rank_io(rank, geometry, rows, true, ring)?;
                bytes.intake = bytes.intake.max(decode.intake);
            }
            bytes.host_exchange = bytes.host_exchange.max(io.host_exchange);
            bytes.row_indices = bytes.row_indices.max(io.row_indices);
            bytes.endpoint_staging = bytes.endpoint_staging.checked_add(
                endpoint.response_prefix_bytes()
                    .ok_or(PlacementError::Inventory("invalid Spark endpoint prefix capacity"))? as u64)
                .ok_or(PlacementError::Overflow("Spark endpoint staging"))?;
            let mapped = endpoint.ring_bytes(geometry, rings)
                .ok_or(PlacementError::Inventory("invalid Spark endpoint ring geometry"))? as u64;
            bytes.mapped_rings = bytes.mapped_rings.checked_add(mapped)
                .ok_or(PlacementError::Overflow("Spark endpoint rings"))?;
        }
        Ok(bytes)
    }

    pub fn includes(&self, component: WorkingSetComponent) -> bool {
        match component {
            WorkingSetComponent::Always => true,
            WorkingSetComponent::RtxExperts => !self.rtx_layers.is_empty(),
            WorkingSetComponent::SparkExperts => !self.spark_layers.is_empty() && self.spark_ranks > 0,
        }
    }

    pub fn selected_ranks(&self) -> std::ops::Range<usize> {
        0..if self.includes(WorkingSetComponent::SparkExperts) { self.spark_ranks } else { 0 }
    }

    pub fn demands<'a>(&'a self, rows: &'a [ComponentDemand]) -> impl Iterator<Item = &'a Demand> + 'a {
        rows.iter().filter(|row| self.includes(row.component)).map(|row| &row.demand)
    }

    /// Coordinator staging is shared across ranks, but not across prefill lanes.
    pub fn coordinator_staging(&self, geometry: ExpertGeometry, capacity: u32, bf16: bool)
        -> Result<u64, PlacementError> {
        if self.selected_ranks().is_empty() { return Ok(0); }
        if self.prefill_lanes == 0 {
            return Err(PlacementError::Inventory("selected Spark staging needs nonzero lanes"));
        }
        let hidden = geometry.hidden as u64;
        let input = if bf16 { hidden * 2 } else { hidden + hidden / 32 };
        (input + geometry.topk as u64 * 8).checked_mul(capacity as u64)
            .and_then(|bytes| bytes.checked_mul(self.prefill_lanes))
            .ok_or(PlacementError::Overflow("Spark coordinator staging"))
    }

    /// Each selected rank owns its intake/exchange and the stated endpoint rings.
    /// Unselected ranks (including every rank of an empty set) charge nothing.
    pub fn rank_io(&self, rank: usize, geometry: ExpertGeometry, capacity: u32,
        bf16: bool, rings: SparkRingGeometry) -> Result<SparkIoBytes, PlacementError> {
        if !self.selected_ranks().contains(&rank) { return Ok(SparkIoBytes::default()); }
        let rows = capacity as u64;
        let hidden = geometry.hidden as u64;
        let routes = geometry.topk as u64 * 8;
        let input = if bf16 { hidden * 2 } else { hidden + hidden / 32 };
        let multiply = |stride: u64| rows.checked_mul(stride)
            .ok_or(PlacementError::Overflow("Spark intake/exchange"));
        let mapped = geometry.compact_ring_bytes(capacity, rings.max_frame_bytes,
            rings.depth, rings.slot_bytes, rings.alignment, rings.endpoints, bf16)
            .ok_or(PlacementError::Inventory("Spark ring geometry exceeds frame budget"))?;
        Ok(SparkIoBytes {
            intake: multiply(input + routes + hidden * 2)?,
            host_exchange: multiply(routes + hidden * 2)?,
            row_indices: multiply(4)?,
            mapped_rings: mapped as u64,
            endpoint_staging: cuteafd_core::SparkEndpoint { ring_depth: None, capacity,
                ingress: cuteafd_core::SparkIngress::Bf16, execution_lane: 0 }.response_prefix_bytes()
                .and_then(|bytes| bytes.checked_mul(rings.endpoints))
                .ok_or(PlacementError::Overflow("Spark endpoint staging"))? as u64,
        })
    }
}

/// One rank's header-derived contract. Workspace is deliberately absent:
/// the worker's architecture owns that measurement, not the coordinator build.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SparkAssignment {
    pub rank: u32,
    pub resident_bytes: u64,
    pub slices: Vec<cuteafd_core::SparkSlice>,
    pub io: SparkIoBytes,
}

/// Selected storage policy; runtime validates its installed package against
/// this same layout before allocating. Native shards may be replicated groups.
#[derive(Debug, Clone, Copy)]
pub enum SparkResidentLayout {
    Exl3 { world: usize, rank: usize },
    Fp8 { world: usize, rank: usize, slicing: crate::formats::fp8_experts::Slicing },
    Native { world: usize, rank: usize, stored_width: usize },
}

impl SparkAssignment {
    pub fn from_catalog(catalog: &crate::OfficialV41Catalog, working: &SparkWorkingSet,
        rank: usize, layout: SparkResidentLayout, endpoints: &[cuteafd_core::SparkEndpoint],
        rings: cuteafd_core::SparkRings) -> anyhow::Result<Self> {
        use anyhow::{ensure, Context};
        let geometry = catalog.routed_experts().geometry()?;
        let io = working.rank_endpoints_io(rank, geometry, endpoints, rings)?;
        let mut assignment = Self { rank: u32::try_from(rank)?, resident_bytes: 0, slices: vec![], io };
        if !working.selected_ranks().contains(&rank) { return Ok(assignment); }
        ensure!(working.spark_layers.windows(2).all(|pair| pair[0] < pair[1]),
            "Spark layers must be sorted and unique");
        for &layer in &working.spark_layers {
            ensure!(layer >= catalog.routed_experts().first_layer && layer < geometry.layers as usize,
                "Spark layer {layer} is outside routed checkpoint headers");
            let mut slice = cuteafd_core::SparkSlice { unit_map: None, layer: u32::try_from(layer)?, first: 0,
                width: 0, stored_width: 0, tiers: vec![], projection_counts: vec![] };
            let resident = match layout {
                SparkResidentLayout::Exl3 { world, rank } => {
                    let manifest = catalog.exl3().context("Spark EXL3 layout needs checkpoint trellis headers")?;
                    let plan = manifest.balanced_residency(crate::V41Exl3Layer::Backbone(layer), world, rank)?;
                    slice.unit_map = plan.unit_map.as_ref().map(|map| map.fingerprint());
                    slice.first = u32::try_from(plan.intermediate_start)?;
                    slice.width = u32::try_from(plan.intermediate)?;
                    slice.stored_width = slice.width;
                    slice.tiers = plan.tiers.iter().map(|&tier| u32::try_from(tier)).collect::<Result<_, _>>()?;
                    slice.projection_counts = plan.projection_counts.iter().map(|counts| {
                        Ok([u32::try_from(counts[0])?, u32::try_from(counts[1])?, u32::try_from(counts[2])?])
                    }).collect::<Result<_, std::num::TryFromIntError>>()?;
                    plan.device_arena_layout()?.1 as u64
                }
                SparkResidentLayout::Fp8 { world, rank, slicing } => {
                    let tensors = catalog.fp8().context("Spark FP8-family layout needs checkpoint weight/scale headers")?
                        .for_layer(layer)?;
                    // Exact-even mixed layers share source ranges, but seal each
                    // format's native units just as its independent worker group does.
                    let slicing = match slicing {
                        crate::formats::fp8_experts::Slicing::Blocks(block)
                            if world > 0 && geometry.intermediate as usize % (block * world) == 0
                                && geometry.intermediate as usize % (tensors.block() * world) == 0 =>
                            tensors.exact_slicing(world),
                        other => other,
                    };
                    let (first, width) = tensors.rank_range_with(world, rank, slicing)?;
                    slice.unit_map = Some(tensors.unit_map(world, rank, slicing)?.fingerprint());
                    slice.first = u32::try_from(first)?;
                    slice.width = u32::try_from(width)?;
                    slice.stored_width = u32::try_from(tensors.rank_width(world, rank, slicing)?)?;
                    (tensors.weight_read_bytes_with(world, rank, slicing)? as u64)
                        .checked_mul(tensors.shape().experts as u64).context("Spark FP8 resident overflow")?
                }
                SparkResidentLayout::Native { world, rank, stored_width } => {
                    use crate::V41ExpertSelection;
                    ensure!(catalog.exl3().is_none() && catalog.fp8().is_none(),
                        "Spark native layout cannot convert compressed checkpoint headers");
                    let width = if world == 1 {
                        ensure!(rank == 0, "single-rank native shard must have rank 0");
                        geometry.intermediate as usize
                    } else {
                        crate::families::deepseek_v41::v41_expert_staging::V41BackboneTpGeometry::new(
                            geometry.intermediate as usize, geometry.hidden as usize, world, rank)?.intermediate_size()
                    };
                    ensure!(stored_width >= width && stored_width % 32 == 0,
                        "Spark native installed width {stored_width} cannot store logical width {width}");
                    // Check every retained expert's tensor headers, not just config geometry.
                    for expert in 0..geometry.experts as usize {
                        let selected = if world == 1 { V41ExpertSelection::BackboneFull { layer, expert } }
                            else { V41ExpertSelection::BackboneTp { layer, expert, world, rank } };
                        let header_width = if catalog.nvfp4().is_some() {
                            catalog.nvfp4_expert_staging(selected)?.intermediate_size()
                        } else {
                            catalog.expert_staging(selected)?.intermediate_size()
                        };
                        ensure!(header_width == width, "Spark native header width {header_width}, expected {width}");
                    }
                    slice.first = u32::try_from(rank * width)?;
                    slice.width = u32::try_from(width)?;
                    slice.stored_width = u32::try_from(stored_width)?;
                    if catalog.nvfp4().is_none() && catalog.mxfp4_scale_codes() {
                        crate::formats::scale_codes::expert_bytes(geometry.experts as u64,
                            geometry.hidden as u64, width as u64).context("coded Spark residency overflow")?
                    } else {
                        super::families::deepseek_v41::resident_bytes(geometry.hidden as u64,
                            slice.stored_width as u64, geometry.experts as u64, catalog.nvfp4().is_some())?
                    }
                }
            };
            assignment.resident_bytes = assignment.resident_bytes.checked_add(resident)
                .context("Spark resident sum overflow")?;
            assignment.slices.push(slice);
        }
        Ok(assignment)
    }

    pub fn validate(&self, report: &cuteafd_core::ModelAdmissionResponse) -> anyhow::Result<()> {
        use anyhow::ensure;
        let rank = self.rank;
        ensure!(report.rank == rank, "Spark rank {rank}: response rank {} differs", report.rank);
        ensure!(report.resident_bytes == self.resident_bytes,
            "Spark rank {rank}: resident bytes {}, expected {}", report.resident_bytes, self.resident_bytes);
        ensure!(report.slices == self.slices, "Spark rank {rank}: header resident/slice geometry differs");
        for (field, actual, expected) in [
            ("mapped rings", report.mapped_ring_bytes, self.io.mapped_rings),
            ("host exchange", report.host_exchange_bytes, self.io.host_exchange),
            ("row indices", report.row_indices_bytes, self.io.row_indices),
        ] {
            ensure!(actual == expected, "Spark rank {rank}: {field} bytes {actual}, expected {expected}");
        }
        let runtime = &report.runtime;
        let sample = &runtime.sample;
        ensure!(matches!(sample.arch.as_str(), "sm_120" | "sm_121") && sample.sms > 0
            && !sample.driver.is_empty(), "Spark rank {rank}: missing measured runtime architecture");
        ensure!(sample.total_bytes > 0 && sample.free_bytes <= sample.total_bytes
            && sample.context_bytes <= sample.total_bytes,
            "Spark rank {rank}: invalid measured runtime inventory");
        ensure!(runtime.workspace_bytes >= self.io.intake,
            "Spark rank {rank}: measured workspace {} omits intake {}",
            runtime.workspace_bytes, self.io.intake);
        [report.resident_bytes, runtime.workspace_bytes, runtime.pinned_workspace_bytes,
            report.mapped_ring_bytes, report.host_exchange_bytes, report.row_indices_bytes]
            .into_iter().try_fold(0u64, |sum, bytes| sum.checked_add(bytes))
            .ok_or_else(|| anyhow::anyhow!("Spark rank {rank}: retained byte total overflow"))?;
        Ok(())
    }
}

/// Offline Spark ceiling: exact header arenas and endpoint allocations plus an
/// explicitly estimated worker reserve. Feeds the existing pool-first solver's
/// mandatory prefix; it does not choose RTX owners or allocate worker storage.
#[derive(Debug, Clone)]
pub struct SparkBudgetInventory {
    pub layers: Vec<usize>,
    pub resident: Vec<Vec<u64>>,
    pub overhead: Vec<u64>,
    pub budget: u64,
}

impl SparkBudgetInventory {
    pub fn exl3(catalog: &crate::OfficialV41Catalog, working: &SparkWorkingSet,
        endpoints: &[cuteafd_core::SparkEndpoint], rings: cuteafd_core::SparkRings,
        budget: u64, estimated_reserve: u64) -> anyhow::Result<Self> {
        use anyhow::Context;
        let mut resident = Vec::new();
        let mut overhead = Vec::new();
        for rank in 0..working.spark_ranks {
            let mut costs = Vec::new();
            for &layer in &working.spark_layers {
                let one = SparkWorkingSet { spark_layers: vec![layer], ..working.clone() };
                costs.push(SparkAssignment::from_catalog(catalog, &one, rank,
                    SparkResidentLayout::Exl3 { world: working.spark_ranks, rank }, endpoints, rings)?.resident_bytes);
            }
            let io = working.rank_endpoints_io(rank, catalog.routed_experts().geometry()?, endpoints, rings)?;
            let bytes = [io.intake, io.host_exchange, io.row_indices, io.mapped_rings,
                io.endpoint_staging, estimated_reserve].into_iter().try_fold(0u64, |sum, bytes|
                    sum.checked_add(bytes).context("Spark estimated overhead overflow"))?;
            resident.push(costs);
            overhead.push(bytes);
        }
        Ok(Self { layers: working.spark_layers.clone(), resident, overhead, budget })
    }

    pub fn rank_bytes(&self, rank: usize, onboard: usize) -> anyhow::Result<u64> {
        use anyhow::{ensure, Context};
        ensure!(onboard <= self.layers.len() && self.resident.len() == self.overhead.len()
            && self.resident.iter().all(|costs| costs.len() == self.layers.len()), "Spark budget inventory extent");
        let costs = self.resident.get(rank).context("Spark budget rank outside inventory")?;
        // An empty selected set allocates no worker storage or endpoint rings.
        if onboard == self.layers.len() { return Ok(0); }
        costs[onboard..].iter().try_fold(self.overhead[rank], |sum, bytes|
            sum.checked_add(*bytes).context("Spark estimated retained bytes overflow"))
    }

    pub fn minimum_rtx_layers(&self) -> anyhow::Result<usize> {
        anyhow::ensure!(self.budget > 0 && !self.resident.is_empty(), "Spark budget must be positive with selected ranks");
        for count in 0..=self.layers.len() {
            let fits = (0..self.resident.len()).map(|rank| self.rank_bytes(rank, count))
                .collect::<anyhow::Result<Vec<_>>>()?.into_iter().all(|bytes| bytes <= self.budget);
            if fits { return Ok(count); }
        }
        unreachable!("empty Spark working set has zero bytes")
    }

    pub fn constrain(&self, request: &mut super::PlacementRequest) -> anyhow::Result<()> {
        let layers = request.layers.iter().enumerate().filter_map(|(layer, demand)|
            demand.experts.is_some().then_some(layer)).collect::<Vec<_>>();
        anyhow::ensure!(layers == self.layers, "Spark budget routed inventory differs from coordinator");
        let minimum = self.minimum_rtx_layers()?;
        if let super::Onboard::Layers(count) = request.onboard {
            anyhow::ensure!(count <= layers.len(), "explicit --rtx-expert-layers {count} exceeds {} routed layers", layers.len());
        }
        if let Some(count) = request.onboard.layers(layers.len()) {
            anyhow::ensure!(count >= minimum,
                "explicit --rtx-expert-layers {count} is below {minimum} required by estimated --spark-budget-gib ({budget} bytes including reserve and I/O)",
                budget = self.budget);
        }
        request.minimum_expert_layers = request.minimum_expert_layers.max(minimum);
        Ok(())
    }
}

#[derive(Debug, Clone, Copy)]
pub struct SparkRingGeometry {
    pub max_frame_bytes: usize,
    pub depth: usize,
    pub slot_bytes: usize,
    pub alignment: usize,
    pub endpoints: usize,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct SparkIoBytes {
    /// Input rows, expert IDs, gate weights and compact BF16 output rows.
    pub intake: u64,
    pub host_exchange: u64,
    pub row_indices: u64,
    pub mapped_rings: u64,
    /// Retained response prefixes are private to each persistent connection.
    pub endpoint_staging: u64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use cuteafd_core::memory_layout::{Basis, Category};

    #[test]
    fn header_assignment_uses_installed_native_width_and_rejects_missing_headers() {
        use crate::plan::testing::{write_v41_snapshot, write_v41_nvfp4_snapshot};
        use cuteafd_core::{SparkEndpoint, SparkIngress, SparkRings};
        for nvfp4 in [false, true] {
            let dir = tempfile::tempdir().unwrap();
            if nvfp4 { write_v41_nvfp4_snapshot(dir.path()); } else { write_v41_snapshot(dir.path()); }
            let catalog = crate::read_expert_catalog(dir.path()).unwrap();
            let working = SparkWorkingSet { spark_layers: vec![0], spark_ranks: 4,
                prefill_lanes: 1, ..Default::default() };
            let endpoints = [SparkEndpoint { ring_depth: None, capacity: 80, ingress: SparkIngress::Fp8K32, execution_lane: 0 }];
            let rings = SparkRings { max_frame_bytes: 64 << 20, depth: 8, slot_bytes: 8 << 20, alignment: 4096 };
            for stored_width in [576, 640] {
                let assignment = SparkAssignment::from_catalog(&catalog, &working, 3,
                    SparkResidentLayout::Native { world: 4, rank: 3, stored_width }, &endpoints, rings).unwrap();
                assert_eq!((assignment.slices[0].first, assignment.slices[0].width, assignment.slices[0].stored_width),
                    (1728, 576, stored_width as u32));
                let expected = if nvfp4 {
                    super::super::families::deepseek_v41::resident_bytes(
                        5120, stored_width as u64, 384, true).unwrap()
                } else {
                    // Coded packages pad each FC1 half to N128; FC2 keeps logical K576.
                    let (hidden, logical, stored, experts) = (5120u64, 576u64, 640u64, 384u64);
                    let payloads = hidden * stored + hidden * stored / 2;
                    let gate_up = 2 * stored * (1 + 2 * (hidden / 128));
                    let down = hidden * (1 + 2 * logical.div_ceil(128));
                    experts * (payloads + gate_up + down)
                };
                assert_eq!(assignment.resident_bytes, expected);
            }
            for stored_width in [544, 577] {
                assert!(SparkAssignment::from_catalog(&catalog, &working, 3,
                    SparkResidentLayout::Native { world: 4, rank: 3, stored_width }, &endpoints, rings).is_err());
            }
            // Config geometry alone must not admit an expert with a missing projection.
            let index_path = dir.path().join("model.safetensors.index.json");
            let mut index: serde_json::Value = serde_json::from_slice(&std::fs::read(&index_path).unwrap()).unwrap();
            let weights = index["weight_map"].as_object_mut().unwrap();
            let name = weights.keys().find(|key| key.starts_with("layers.0.ffn.experts.383.w2.")).unwrap().clone();
            weights.remove(&name);
            std::fs::write(&index_path, serde_json::to_vec(&index).unwrap()).unwrap();
            let result = crate::read_expert_catalog(dir.path()).and_then(|catalog| SparkAssignment::from_catalog(
                &catalog, &working, 3, SparkResidentLayout::Native { world: 4, rank: 3, stored_width: 640 }, &endpoints, rings));
            assert!(result.is_err());
        }
    }

    #[test]
    fn header_assignment_uses_exl3_projection_counts_and_padded_arena() {
        use crate::plan::testing::{qwen4_config, qwen4_exl3, write_snapshot};
        use cuteafd_core::{SparkEndpoint, SparkIngress, SparkRings};
        let dir = tempfile::tempdir().unwrap();
        let (tensors, _) = qwen4_exl3(1, 4);
        write_snapshot(dir.path(), &qwen4_config(1), &tensors, None);
        let catalog = crate::read_expert_catalog(dir.path()).unwrap();
        let working = SparkWorkingSet { spark_layers: vec![0], spark_ranks: 4,
            prefill_lanes: 1, ..Default::default() };
        let endpoints = [SparkEndpoint { ring_depth: None, capacity: 80, ingress: SparkIngress::Fp8K32, execution_lane: 0 }];
        let rings = SparkRings { max_frame_bytes: 64 << 20, depth: 8, slot_bytes: 8 << 20, alignment: 4096 };
        let assignment = SparkAssignment::from_catalog(&catalog, &working, 3,
            SparkResidentLayout::Exl3 { world: 4, rank: 3 },
            &endpoints, rings).unwrap();
        let plan = catalog.exl3().unwrap().balanced_residency(crate::V41Exl3Layer::Backbone(0), 4, 3).unwrap();
        assert_eq!(assignment.resident_bytes, plan.device_arena_layout().unwrap().1 as u64);
        assert!(assignment.resident_bytes > plan.bytes() as u64);
        assert_eq!(assignment.slices[0].unit_map, plan.unit_map.as_ref().map(|map| map.fingerprint()));
        assert_eq!(assignment.slices[0].width, plan.intermediate as u32);
        assert_eq!(assignment.slices[0].tiers, plan.tiers.iter().map(|&tier| tier as u32).collect::<Vec<_>>());
        assert_eq!(assignment.slices[0].projection_counts,
            plan.projection_counts.iter().map(|row| row.map(|count| count as u32)).collect::<Vec<_>>());
    }

    #[test]
    fn header_assignment_preserves_exact_and_padded_fp8_family_slices() {
        use crate::formats::fp8_experts::Slicing;
        use crate::plan::testing::{fp8, mxfp4, mimo_flash_config, write_snapshot};
        use cuteafd_core::{SparkEndpoint, SparkIngress, SparkRings};
        for packed in [false, true] {
            let dir = tempfile::tempdir().unwrap();
            let mut config = mimo_flash_config();
            config["n_routed_experts"] = 2.into();
            config["num_experts_per_tok"] = 1.into();
            let mut tensors = Vec::new();
            for expert in 0..2 {
                for (name, rows, cols) in [("gate", 2048, 4096), ("up", 2048, 4096), ("down", 4096, 2048)] {
                    let prefix = format!("model.layers.1.mlp.experts.{expert}.{name}_proj");
                    tensors.extend(if packed { mxfp4(&prefix, rows, cols) } else { fp8(&prefix, rows, cols, None) });
                }
            }
            write_snapshot(dir.path(), &config, &tensors, None);
            let catalog = crate::read_expert_catalog(dir.path()).unwrap();
            let working = SparkWorkingSet { spark_layers: vec![1], spark_ranks: 6,
                prefill_lanes: 1, ..Default::default() };
            let endpoints = [SparkEndpoint { ring_depth: None, capacity: 4096, ingress: SparkIngress::Fp8K32, execution_lane: 0 }];
            let rings = SparkRings { max_frame_bytes: 64 << 20, depth: 8, slot_bytes: 8 << 20, alignment: 4096 };
            let exact = SparkAssignment::from_catalog(&catalog, &working, 5,
                SparkResidentLayout::Fp8 { world: 6, rank: 5, slicing: Slicing::Blocks(128) }, &endpoints, rings).unwrap();
            let padded = SparkAssignment::from_catalog(&catalog, &working, 5,
                SparkResidentLayout::Fp8 { world: 6, rank: 5, slicing: Slicing::Padded }, &endpoints, rings).unwrap();
            assert_eq!((exact.slices[0].first, exact.slices[0].width, exact.slices[0].stored_width), (1536, 256, 256));
            assert_eq!(padded.slices[0].stored_width, 384);
            assert!(exact.resident_bytes < padded.resident_bytes);
            assert_eq!(exact.io, padded.io);
            let empty = SparkAssignment::from_catalog(&catalog, &working, 6,
                SparkResidentLayout::Fp8 { world: 6, rank: 6, slicing: Slicing::Padded }, &[], rings).unwrap();
            assert_eq!(empty.resident_bytes, 0);
            assert!(empty.slices.is_empty());
            assert_eq!(empty.io, SparkIoBytes::default());
        }
    }

    #[test]
    fn header_assignment_uses_each_layers_native_storage_format() {
        use crate::formats::fp8_experts::Slicing;
        use crate::plan::testing::{fp8, t, mimo_flash_config, write_snapshot};
        use cuteafd_core::{SparkEndpoint, SparkIngress, SparkRings};
        let dir = tempfile::tempdir().unwrap();
        let mut config = mimo_flash_config();
        config["num_hidden_layers"] = 3.into();
        config["hybrid_layer_pattern"] = serde_json::json!([0, 1, 1]);
        config["moe_layer_freq"] = serde_json::json!([0, 1, 1]);
        config["n_routed_experts"] = 2.into();
        config["num_experts_per_tok"] = 1.into();
        let mut tensors = Vec::new();
        for layer in [1, 2] {
            for expert in 0..2 {
                for (name, rows, cols) in [("gate", 2048, 4096), ("up", 2048, 4096), ("down", 4096, 2048)] {
                    let prefix = format!("model.layers.{layer}.mlp.experts.{expert}.{name}_proj");
                    if layer == 1 { tensors.extend(fp8(&prefix, rows, cols, None)); }
                    else { tensors.push(t(format!("{prefix}.weight"), "BF16", &[rows, cols])); }
                }
            }
        }
        write_snapshot(dir.path(), &config, &tensors, None);
        let catalog = crate::read_expert_catalog(dir.path()).unwrap();
        let endpoints = [SparkEndpoint { ring_depth: None, capacity: 64, ingress: SparkIngress::Bf16, execution_lane: 0 }];
        let rings = SparkRings { max_frame_bytes: 64 << 20, depth: 8, slot_bytes: 8 << 20, alignment: 4096 };
        for world in [4, 6] {
            for rank in 0..world {
                let layout = SparkResidentLayout::Fp8 { world, rank, slicing: Slicing::Blocks(128) };
                let assign = |layers| SparkAssignment::from_catalog(&catalog,
                    &SparkWorkingSet { spark_layers: layers, spark_ranks: world,
                        prefill_lanes: 1, ..Default::default() }, rank, layout, &endpoints, rings).unwrap();
                let fp8 = assign(vec![1]);
                let original = catalog.fp8().unwrap();
                original.validate_layer(1).unwrap();
                assert_eq!(fp8.resident_bytes,
                    (original.weight_read_bytes_with(world, rank, Slicing::Blocks(128)).unwrap()
                        * original.shape().experts) as u64);
                let bf16 = assign(vec![2]);
                let mixed = assign(vec![1, 2]);
                let slice = &bf16.slices[0];
                assert_eq!(bf16.resident_bytes, 3 * 4096 * u64::from(slice.stored_width) * 2 * 2);
                assert_eq!(mixed.resident_bytes, fp8.resident_bytes + bf16.resident_bytes);
                assert_eq!(mixed.slices, [fp8.slices[0].clone(), slice.clone()]);
                let view = original.for_layer(2).unwrap();
                let (first, width) = view.rank_range_with(world, rank, Slicing::Blocks(128)).unwrap();
                assert_eq!((slice.first, slice.width, slice.stored_width),
                    (first as u32, width as u32, width as u32));
                assert_eq!(mixed.io, bf16.io);
            }
        }
    }

    #[test]
    fn assignment_reconciles_exact_bytes_and_worker_architecture() {
        use cuteafd_core::{ModelAdmissionResponse, RuntimeSample, SparkRuntimeInventory, SparkSlice};
        let assignment = SparkAssignment { rank: 2, resident_bytes: 1 << 30,
            slices: vec![SparkSlice { unit_map: None, layer: 3, first: 768, width: 384, stored_width: 384,
                tiers: vec![2, 3], projection_counts: vec![[10, 11, 12], [12, 11, 10]] }],
            io: SparkIoBytes { intake: 100, host_exchange: 200, row_indices: 400, mapped_rings: 800, endpoint_staging: 160 } };
        let report = ModelAdmissionResponse { rank: 2, resident_bytes: assignment.resident_bytes,
            slices: assignment.slices.clone(), mapped_ring_bytes: 800, host_exchange_bytes: 200,
            row_indices_bytes: 400, runtime: SparkRuntimeInventory { detail: Default::default(),
                sample: RuntimeSample { device: 0, total_bytes: 121 << 30, free_bytes: 100 << 30,
                    context_bytes: 1 << 30, module_bytes: 1 << 20, sms: 48,
                    arch: "sm_121".into(), driver: "test".into() },
                workspace_bytes: 100, pinned_workspace_bytes: 0 } };
        assignment.validate(&report).unwrap();
        for field in 0..11 {
            let mut invalid = report.clone();
            match field {
                0 => invalid.rank = 3,
                1 => invalid.resident_bytes += 1,
                2 => invalid.slices[0].projection_counts[0][1] += 1,
                3 => invalid.slices[0].stored_width += 128,
                4 => invalid.mapped_ring_bytes += 1,
                5 => invalid.host_exchange_bytes += 1,
                6 => invalid.row_indices_bytes += 1,
                7 => invalid.runtime.sample.arch = "sm_90".into(),
                8 => invalid.runtime.sample.free_bytes = invalid.runtime.sample.total_bytes + 1,
                9 => invalid.runtime.workspace_bytes = 99,
                _ => invalid.runtime.pinned_workspace_bytes = u64::MAX,
            }
            assert!(assignment.validate(&invalid).is_err(), "field {field}");
        }
        // Workspace belongs to the worker: no SM120 estimate is substituted.
        let mut larger = report;
        larger.runtime.workspace_bytes = 1 << 30;
        assignment.validate(&larger).unwrap();
    }

    #[test]
    fn fp8_endpoint_io_charges_bf16_partials() {
        use cuteafd_core::{SparkEndpoint, SparkIngress, SparkRings};
        let set = SparkWorkingSet { spark_layers: vec![3], spark_ranks: 4,
            ..Default::default() };
        let geometry = ExpertGeometry::GLM5;
        let endpoint = SparkEndpoint { ring_depth: None, capacity: 64, ingress: SparkIngress::Fp8K32, execution_lane: 0 };
        let rings = SparkRings { max_frame_bytes: 4 << 20, depth: 2,
            slot_bytes: 1 << 20, alignment: 4096 };
        let io = set.rank_endpoints_io(0, geometry, &[endpoint], rings).unwrap();
        assert_eq!(io.intake, 64 * (6144 + 6144 / 32 + 6144 * 2 + 8 * 8));
        for capacity in [1, 16, 64, 65, 4096] {
            let endpoint = SparkEndpoint { capacity, ..endpoint };
            let rings = SparkRings { max_frame_bytes: 128 << 20, ..rings };
            let bounded = set.rank_endpoints_io(0, geometry, &[endpoint], rings).unwrap();
            assert_eq!(bounded.intake, u64::from(capacity) * (6144 + 6144 / 32 + 6144 * 2 + 8 * 8));
            assert_eq!(bounded.host_exchange, u64::from(capacity) * (6144 * 2 + 8 * 8));
            assert_eq!(bounded.row_indices, u64::from(capacity) * 4);
            assert_eq!(bounded.mapped_rings, endpoint.ring_bytes(geometry, rings).unwrap() as u64);
        }
        assert_eq!(io.host_exchange, 64 * (6144 * 2 + 8 * 8));
        assert_eq!(io.row_indices, 64 * 4);
        assert_eq!(io.mapped_rings, endpoint.ring_bytes(geometry, rings).unwrap() as u64);
        assert_eq!(set.rank_endpoints_io(4, geometry, &[endpoint], rings).unwrap(), SparkIoBytes::default());
        assert!(set.rank_endpoints_io(0, geometry, &[endpoint], SparkRings {
            max_frame_bytes: 64 * (6144 * 2 + 4) + 95, ..rings }).is_err());
    }

    #[test]
    fn endpoint_io_sums_rings_but_shares_owner_buffers() {
        use cuteafd_core::{SparkEndpoint, SparkIngress, SparkRings};
        let set = SparkWorkingSet { spark_layers: vec![3], spark_ranks: 6,
            prefill_lanes: 3, ..Default::default() };
        let geometry = ExpertGeometry::MIMO_V26_PRO;
        let rings = SparkRings { max_frame_bytes: 64 << 20, depth: 8,
            slot_bytes: 8 << 20, alignment: 4096 };
        let main = SparkEndpoint { ring_depth: None, capacity: 4096, ingress: SparkIngress::Bf16Decode { rows: 80 }, execution_lane: 0 };
        let lane = SparkEndpoint { ring_depth: None, capacity: 1024, ingress: SparkIngress::Fp8K32, execution_lane: 0 };
        let io = set.rank_endpoints_io(0, geometry, &[main, lane], rings).unwrap();
        assert_eq!(io.mapped_rings, (main.ring_bytes(geometry, rings).unwrap()
            + lane.ring_bytes(geometry, rings).unwrap()) as u64);
        let full_fp8 = set.rank_io(0, geometry, 4096, false, two_endpoints()).unwrap();
        assert_eq!(io.intake, full_fp8.intake);
        assert_eq!(io.host_exchange, full_fp8.host_exchange);
        assert_eq!(io.row_indices, full_fp8.row_indices);
        assert_eq!(io.endpoint_staging, (main.response_prefix_bytes().unwrap()
            + lane.response_prefix_bytes().unwrap()) as u64);
        assert_ne!(io.endpoint_staging, full_fp8.endpoint_staging);
        assert!(main.ring_bytes(geometry, rings).unwrap()
            < SparkEndpoint { ingress: SparkIngress::Bf16, ..main }.ring_bytes(geometry, rings).unwrap());
        assert!(set.rank_endpoints_io(0, geometry, &[], rings).is_err());
        assert!(set.rank_endpoints_io(0, geometry, &[SparkEndpoint {
            ingress: SparkIngress::Bf16Decode { rows: 4097 }, ..main }], rings).is_err());
        assert_eq!(set.rank_endpoints_io(6, geometry, &[], rings).unwrap(), SparkIoBytes::default());
    }

    fn two_endpoints() -> SparkRingGeometry {
        SparkRingGeometry { max_frame_bytes: 64 << 20, depth: 8, slot_bytes: 8 << 20,
            alignment: 4096, endpoints: 2 }
    }

    #[test]
    fn empty_and_unselected_components_charge_zero() {
        let empty = SparkWorkingSet::default();
        assert!(empty.selected_ranks().is_empty());
        assert_eq!(empty.rank_io(0, ExpertGeometry::GLM5, 4096, true,
            two_endpoints()).unwrap(), SparkIoBytes::default());
        assert_eq!(empty.coordinator_staging(ExpertGeometry::GLM5, 4096, true).unwrap(), 0);
        let rows: Vec<_> = [WorkingSetComponent::Always, WorkingSetComponent::RtxExperts,
            WorkingSetComponent::SparkExperts].into_iter().map(|component|
            ComponentDemand { component, demand: Demand::new(0, Category::Workspace,
                "selected rows", 100, Basis::Formula) }).collect();
        assert_eq!(empty.demands(&rows).map(|d| d.bytes).sum::<u64>(), 100);
        let local = SparkWorkingSet { rtx_layers: vec![3], ..empty };
        assert_eq!(local.demands(&rows).map(|d| d.bytes).sum::<u64>(), 200);
    }

    #[test]
    fn bf16_io_is_per_selected_rank_not_world_scaled_or_lane_replicated() {
        for geometry in [ExpertGeometry::DEEPSEEK_V41, ExpertGeometry::GLM5,
            ExpertGeometry::GLM5_FLASH, ExpertGeometry::QWEN4] {
            for ranks in [2, 3, 4, 6] {
                let set = SparkWorkingSet { spark_layers: vec![3], spark_ranks: ranks,
                    prefill_lanes: 2, ..SparkWorkingSet::default() };
                let io = set.rank_io(0, geometry, 4096, true, two_endpoints()).unwrap();
                assert_eq!(io.intake, 4096 * (geometry.hidden as u64 * 4 + geometry.topk as u64 * 8));
                assert_eq!(io.host_exchange, 4096 * (geometry.hidden as u64 * 2 + geometry.topk as u64 * 8));
                assert_eq!(io.mapped_rings, geometry.compact_ring_bytes(4096, 64 << 20,
                    8, 8 << 20, 4096, 2, true).unwrap() as u64);
                assert_eq!(set.rank_io(ranks - 1, geometry, 4096, true,
                    two_endpoints()).unwrap(), io);
                assert_eq!(set.rank_io(ranks, geometry, 4096, true,
                    two_endpoints()).unwrap(), SparkIoBytes::default());
                let mut serial = set.clone(); serial.prefill_lanes = 1;
                assert_eq!(set.coordinator_staging(geometry, 4096, true).unwrap(),
                    serial.coordinator_staging(geometry, 4096, true).unwrap() * 2);
            }
        }
    }

    #[test]
    fn selected_coordinator_staging_requires_nonzero_lanes() {
        let set = SparkWorkingSet { spark_layers: vec![3], spark_ranks: 4,
            ..SparkWorkingSet::default() };
        assert!(set.coordinator_staging(ExpertGeometry::GLM5, 4096, true).is_err());
        let unselected = SparkWorkingSet { spark_layers: vec![], ..set };
        assert_eq!(unselected.coordinator_staging(ExpertGeometry::GLM5, 4096, true).unwrap(), 0);
    }

    #[test]
    fn legacy_wire_and_bad_geometry_do_not_silently_become_bf16() {
        let set = SparkWorkingSet { spark_layers: vec![3], spark_ranks: 4,
            prefill_lanes: 1, ..SparkWorkingSet::default() };
        let g = ExpertGeometry::GLM5;
        let bf16 = set.rank_io(0, g, 4096, true, two_endpoints()).unwrap();
        let wire = set.rank_io(0, g, 4096, false, two_endpoints()).unwrap();
        assert_eq!(bf16.intake - wire.intake, 4096 * (6144 - 6144 / 32));
        assert!(bf16.mapped_rings > wire.mapped_rings);
        assert!(set.rank_io(0, g, 4096, true, SparkRingGeometry {
            max_frame_bytes: 1024, ..two_endpoints() }).is_err());
        assert!(set.rank_io(0, g, 4096, true, SparkRingGeometry {
            alignment: 0, ..two_endpoints() }).is_err());
    }
}
