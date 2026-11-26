//! DeepSeek V4.1 DFly2 speculative drafter (`AngelSlim/DS-V4.1-Flash-DFly2-B6`).
//!
//! Static weights live in packed arenas charged once by admission:
//! - Dense BF16/F32 tensors (`1.52 GiB`) in one `DeviceWeightArena` (`v41/dfly2/dense`).
//! - Routed MXFP4 experts (`6.72 GiB`, 3 × 128 top-3) in `ExpertWeights` (whole
//!   on the decoder GPU, or TP2 split across GPU0/GPU1 reusing `dspark::tp2::Weights`).
//! - Zero private draft KV cache and zero prefix draft marks.

mod chain;
mod taps;

pub(crate) use chain::{Dfly2Chain, Dfly2Vocab};
pub(crate) use taps::Dfly2TapStore;

use super::dspark::{draft_lanes, tp2};
use super::{ExpertLayer, ExpertWeights};
use crate::shared::draft::dfly2::Dfly2BlockShape;
use crate::shared::memory::device::{Allocation, Device};
use crate::shared::memory::weight_arena::DeviceWeightArena;
use anyhow::{ensure, Context, Result};
use cuteafd_core::memory_layout::{ArenaTag, Basis, Category};
use cuteafd_ffi::{CuteafdDeviceBuffer, NativeLibrary};
use cuteafd_loader::{OfficialV41Catalog, V41Dfly2Catalog, V41Dfly2Config};
use std::collections::BTreeMap;
use std::rc::Rc;

pub(crate) enum Dfly2ExpertStages<'a> {
    Full([ExpertWeights<'a>; 3]),
    Tp2([Rc<tp2::Weights<'a>>; 2]),
}

pub(crate) struct Dfly2Weights<'library> {
    library: &'library NativeLibrary,
    device: Device<'library>,
    config: V41Dfly2Config,
    block: Dfly2BlockShape,
    dense_arena: DeviceWeightArena<'library>,
    dense: BTreeMap<String, Allocation<'library>>,
    experts: Dfly2ExpertStages<'library>,
    dense_bytes: usize,
    expert_resident_bytes: usize,
}

impl<'library> Dfly2Weights<'library> {
    pub fn device(&self) -> Device<'library> { self.device }
    pub fn library(&self) -> &'library NativeLibrary { self.library }
    pub fn config(&self) -> &V41Dfly2Config { &self.config }
    pub fn block(&self) -> Dfly2BlockShape { self.block }
    pub fn block_size(&self) -> usize { self.block.block_size }
    pub fn draft_width(&self) -> usize { self.block.drafts() }
    pub fn is_tp2(&self) -> bool { matches!(self.experts, Dfly2ExpertStages::Tp2(_)) }
    pub(crate) fn experts(&self) -> &Dfly2ExpertStages<'library> { &self.experts }

    pub fn tensor(&self, name: &str) -> Result<CuteafdDeviceBuffer> {
        self.dense
            .get(name)
            .map(|a| a.buffer)
            .with_context(|| format!("DFly2 dense tensor missing: {name}"))
    }

    pub fn release_raw_scales(&self) {
        if let Dfly2ExpertStages::Full(stages) = &self.experts {
            for stage in stages {
                stage.release_raw_scales();
            }
        }
    }

    pub(crate) fn inventory(
        &self,
        inventory: &mut crate::shared::memory::receipts::Inventory<'_>,
    ) -> Result<()> {
        for (name, alloc) in &self.dense {
            alloc.inventory(inventory, name)?;
        }
        Ok(())
    }

    /// Expert compiled capacity covering `members * block_size` rows.
    pub fn expert_capacity(members: u32, block_size: usize, tp2: bool) -> Result<u32> {
        let live = (members as usize)
            .checked_mul(block_size)
            .context("DFly2 live rows overflow")?;
        ensure!(
            live >= 1 && live <= cuteafd_core::V41_DRAFT_WIDE_ROWS,
            "DFly2 live rows {live} exceed V41_DRAFT_WIDE_ROWS ({})",
            cuteafd_core::V41_DRAFT_WIDE_ROWS
        );
        let cap = if tp2 {
            if live <= 16 { 16 } else if live <= 80 { 80 } else { 256 }
        } else if live <= 16 {
            16
        } else if live <= 40 {
            40
        } else if live <= 80 {
            80
        } else {
            256
        };
        Ok(cap)
    }

    /// Pre-load admission inventory for 1× RTX (all weights & workspaces on GPU 0).
    #[allow(clippy::too_many_arguments)]
    pub fn single_inventory(
        library: &NativeLibrary,
        target: &OfficialV41Catalog,
        dfly2: &V41Dfly2Catalog,
        block_size: usize,
        requests: u32,
        lanes: u8,
    ) -> Result<cuteafd_loader::placement::families::deepseek_v41::DraftInventory> {
        let (lane_count, lane_requests) = draft_lanes(requests, lanes)?;
        let expert_catalog = dfly2.expert_catalog(target.config());
        let mut expert_resident = 0usize;
        let mut expert_staging = 0usize;
        for stage in 0..3 {
            let plan = ExpertWeights::plan(library, &expert_catalog, ExpertLayer::Dspark { stage })?;
            expert_resident = expert_resident
                .checked_add(plan.resident_bytes)
                .context("DFly2 expert residency overflow")?;
            expert_staging = expert_staging.max(plan.device_staging_bytes);
        }
        let dense_layout = dfly2.dense_layout()?;
        let dense_bytes = dense_layout.bytes();
        let resident = expert_resident
            .checked_add(dense_bytes)
            .context("DFly2 resident bytes overflow")?;
        let loader_peak = expert_resident
            .checked_add(expert_staging)
            .context("DFly2 expert load peak overflow")?
            .max(resident);
        let chain_bytes = Dfly2Chain::single_device_bytes(
            library,
            lane_requests as usize,
            block_size,
        )?;
        let taps_bytes = Dfly2TapStore::device_bytes(requests as usize)?;
        let runtime_bytes = chain_bytes
            .checked_mul(lane_count as usize)
            .and_then(|n| n.checked_add(taps_bytes))
            .context("DFly2 single runtime overflow")?;
        Ok(cuteafd_loader::placement::families::deepseek_v41::DraftInventory {
            resident: resident as u64,
            runtime: vec![cuteafd_loader::placement::Demand::new(
                0,
                Category::Drafter,
                "DFly2 lanes/taps",
                runtime_bytes as u64,
                Basis::Exact,
            )],
            marks: 0,
            peer_half: 0,
            staging: [expert_staging as u64, 0],
            loading_experts: [expert_resident as u64, 0],
            loader_peak: [loader_peak as u64, 0],
            dense: Some(dense_layout),
        })
    }

    /// Pre-load admission inventory for 2× RTX (dense weights & decoder workspaces
    /// on GPU 1; optional TP2 routed expert half and split vocab shard on GPU 0).
    #[allow(clippy::too_many_arguments)]
    pub fn distributed_inventory(
        library: &NativeLibrary,
        target: &OfficialV41Catalog,
        dfly2: &V41Dfly2Catalog,
        block_size: usize,
        requests: u32,
        lanes: u8,
        tp2_experts: bool,
        split: usize,
    ) -> Result<cuteafd_loader::placement::families::deepseek_v41::DraftInventory> {
        let (lane_count, lane_requests) = draft_lanes(requests, lanes)?;
        let expert_catalog = dfly2.expert_catalog(target.config());
        let dense_layout = dfly2.dense_layout()?;
        let dense_bytes = dense_layout.bytes();

        let (resident, peer_half, staging, loading_experts, loader_peak) = if tp2_experts {
            let (mut half, mut stage_bytes) = ([0usize; 2], [0usize; 2]);
            for rank in 0..2 {
                for stage in 0..3 {
                    let plan = Device { library, id: rank as i32 }.run(|| {
                        ExpertWeights::plan(
                            library,
                            &expert_catalog,
                            ExpertLayer::DsparkTp2 { stage, rank },
                        )
                    })?;
                    half[rank] = half[rank]
                        .checked_add(plan.resident_bytes)
                        .context("DFly2 TP2 residency overflow")?;
                    stage_bytes[rank] = stage_bytes[rank].max(plan.device_staging_bytes);
                }
            }
            let home_resident = half[1]
                .checked_add(dense_bytes)
                .context("DFly2 TP2 home residency overflow")?;
            let home_peak = half[1]
                .checked_add(stage_bytes[1])
                .context("DFly2 TP2 home load peak overflow")?
                .max(home_resident);
            let peer_peak = half[0]
                .checked_add(stage_bytes[0])
                .context("DFly2 TP2 peer load peak overflow")?;
            (
                home_resident,
                half[0],
                [stage_bytes[1], stage_bytes[0]],
                [half[1], half[0]],
                [home_peak, peer_peak],
            )
        } else {
            let mut expert_resident = 0usize;
            let mut expert_staging = 0usize;
            for stage in 0..3 {
                let plan = Device { library, id: 1 }.run(|| {
                    ExpertWeights::plan(library, &expert_catalog, ExpertLayer::Dspark { stage })
                })?;
                expert_resident = expert_resident
                    .checked_add(plan.resident_bytes)
                    .context("DFly2 expert residency overflow")?;
                expert_staging = expert_staging.max(plan.device_staging_bytes);
            }
            let home_resident = expert_resident
                .checked_add(dense_bytes)
                .context("DFly2 home residency overflow")?;
            let home_peak = expert_resident
                .checked_add(expert_staging)
                .context("DFly2 home load peak overflow")?
                .max(home_resident);
            (
                home_resident,
                0,
                [expert_staging, 0],
                [expert_resident, 0],
                [home_peak, 0],
            )
        };

        let lane_bytes = Device { library, id: 1 }.run(|| {
            Dfly2Chain::distributed_device_bytes(
                library,
                lane_requests as usize,
                block_size,
                tp2_experts,
                split,
            )
        })?;
        let taps_bytes = Dfly2TapStore::device_bytes(requests as usize)?;
        let demand = |gpu, group, bytes: usize| {
            cuteafd_loader::placement::Demand::new(
                gpu,
                Category::Drafter,
                group,
                bytes as u64,
                Basis::Exact,
            )
        };
        let runtime: Vec<_> = [
            demand(0, "DFly2 request lanes", lane_bytes[0] * lane_count as usize),
            demand(1, "DFly2 request lanes", lane_bytes[1] * lane_count as usize),
            demand(1, "DFly2 taps", taps_bytes),
        ]
        .into_iter()
        .filter(|d| d.bytes > 0)
        .collect();
        Ok(cuteafd_loader::placement::families::deepseek_v41::DraftInventory {
            resident: resident as u64,
            runtime,
            marks: 0,
            peer_half: peer_half as u64,
            staging: staging.map(|b| b as u64),
            loading_experts: loading_experts.map(|b| b as u64),
            loader_peak: loader_peak.map(|b| b as u64),
            dense: Some(dense_layout),
        })
    }

    /// Load DFly2 weights onto the current device (full routed experts + packed dense arena).
    pub fn load_full(
        library: &'library NativeLibrary,
        target: &OfficialV41Catalog,
        dfly2: &V41Dfly2Catalog,
        block_size: usize,
        device_budget: usize,
    ) -> Result<Self> {
        let device = Device { library, id: library.cuda_get_device()? };
        let block = Dfly2BlockShape::new(
            block_size,
            dfly2.config().max_block_slots,
            dfly2.config().mask_token_id,
        )?;
        let expert_catalog = dfly2.expert_catalog(target.config());
        let mut expert_resident = 0usize;
        let mut stages = Vec::with_capacity(3);
        for stage in 0..3 {
            let remaining = device_budget
                .checked_sub(expert_resident)
                .context("DFly2 remaining expert budget underflow")?;
            let w = ExpertWeights::load(library, &expert_catalog, ExpertLayer::Dspark { stage }, remaining)?;
            expert_resident = expert_resident
                .checked_add(w.budget().resident_bytes)
                .context("DFly2 expert residency overflow")?;
            stages.push(w);
        }
        let experts = Dfly2ExpertStages::Full(
            stages.try_into().ok().context("DFly2 requires 3 expert stages")?,
        );
        Self::finish_dense(library, device, dfly2, block, experts, expert_resident, device_budget)
    }

    /// Load DFly2 weights on 2× RTX with TP2 routed experts across GPU0/GPU1 and
    /// dense weights on GPU1. Caller must run on GPU1.
    pub fn load_tp2(
        library: &'library NativeLibrary,
        target: &OfficialV41Catalog,
        dfly2: &V41Dfly2Catalog,
        block_size: usize,
        budgets: [usize; 2],
    ) -> Result<Self> {
        ensure!(library.cuda_get_device()? == 1, "DFly2 TP2 home weights belong on RTX1");
        let block = Dfly2BlockShape::new(
            block_size,
            dfly2.config().max_block_slots,
            dfly2.config().mask_token_id,
        )?;
        let expert_catalog = dfly2.expert_catalog(target.config());
        let devices = [Device { library, id: 0 }, Device { library, id: 1 }];
        let pair = tp2::Weights::load_pair(devices, &expert_catalog, budgets, tp2::LAYOUT)?;
        let home_experts = pair[1].resident_bytes();
        let experts = Dfly2ExpertStages::Tp2(pair);
        Self::finish_dense(library, devices[1], dfly2, block, experts, home_experts, budgets[1])
    }

    fn finish_dense(
        library: &'library NativeLibrary,
        device: Device<'library>,
        dfly2: &V41Dfly2Catalog,
        block: Dfly2BlockShape,
        experts: Dfly2ExpertStages<'library>,
        expert_resident_bytes: usize,
        home_budget: usize,
    ) -> Result<Self> {
        let layout = dfly2.dense_layout()?;
        let dense_bytes = layout.bytes();
        ensure!(
            expert_resident_bytes
                .checked_add(dense_bytes)
                .is_some_and(|total| total <= home_budget),
            "DFly2 weights ({} + {} B) exceed home budget ({home_budget} B)",
            expert_resident_bytes,
            dense_bytes
        );
        let dense_arena = DeviceWeightArena::new(
            device,
            layout.clone(),
            ArenaTag::drafter_weights("v41/dfly2/dense"),
        )?;
        let mut dense = BTreeMap::new();
        for (name, view) in layout.views() {
            let dst = dense_arena.view(name, view.bytes())?;
            let raw = dfly2.read_dense_bytes(name)?;
            ensure!(
                raw.len() == view.bytes(),
                "DFly2 tensor {name} length {} != arena view {}",
                raw.len(),
                view.bytes()
            );
            device.run(|| library.copy_h2d(dst.buffer, &raw))?;
            dense.insert(name.clone(), dst);
        }
        crate::families::deepseek_v41::weight_layout::probe_weights(
            library,
            &[device.id],
            "v41/dfly2-dense-loaded",
            |inv| {
                for (name, alloc) in &dense {
                    alloc.inventory(inv, name)?;
                }
                Ok(())
            },
        )?;
        tracing::info!(
            device = device.id,
            block_size = block.block_size,
            dense_bytes,
            expert_resident_bytes,
            tp2 = matches!(experts, Dfly2ExpertStages::Tp2(_)),
            "V4.1 DFly2 weights loaded into packed arenas"
        );
        Ok(Self {
            library,
            device,
            config: dfly2.config().clone(),
            block,
            dense_arena,
            dense,
            experts,
            dense_bytes,
            expert_resident_bytes,
        })
    }

    pub fn resident_bytes(&self) -> usize {
        self.dense_bytes + self.expert_resident_bytes
    }

    pub fn dense_arena(&self) -> &DeviceWeightArena<'library> {
        &self.dense_arena
    }
}
