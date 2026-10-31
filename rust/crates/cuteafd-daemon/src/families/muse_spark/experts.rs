//! The Spark expert path behind one engine contract. The engine hands a
//! layer's expert-width rows and their routes to [`SparkExperts::send`] and,
//! after [`SparkExperts::land`], reads the combined expert output: one E-wide
//! BF16 row per row in the lane's `combined` plane (each routed expert's
//! output post_expert_normed, weighted and summed over the top-k). How the
//! Sparks hold experts, what they reply and how the coordinator combines it
//! are the implementation's; the engine sees none of it.
//!
//! The Muse residency is TP (the EP study, `ep_layouts.py`, approved
//! 2026-10-31): every Spark holds an intermediate slice of every expert and
//! replies with its partial down-proj output per routed (row, expert) as
//! NVFP4 (E2M1 with an E4M3 scale per 16, the hardware cvt); the coordinator
//! program `muse_tp_combine` dequantizes, sums the ranks per expert, applies
//! post_expert_norm and the routing weight, and sums the top-k into `combined`.
//! FP8 (E4M3, UE8M0 per 32) is the documented fallback wire.
use anyhow::{ensure, Context, Result};

/// The wire of one rank's per-(row, expert) partial.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum PartialWire {
    /// E2M1 values with an E4M3 scale per 16 elements (4.5 bits per element).
    Nvfp4,
    /// E4M3 values with a UE8M0 scale per 32 elements (8.25 bits per element).
    Fp8,
}

impl PartialWire {
    /// Bytes of one E-wide partial row on the wire (values, then scales).
    pub(crate) fn row_bytes(self, width: usize) -> Result<usize> {
        Ok(match self {
            Self::Nvfp4 => { ensure!(width % 32 == 0, "NVFP4 partial width"); width / 2 + width / 16 }
            Self::Fp8 => { ensure!(width % 32 == 0, "FP8 partial width"); width + width / 32 }
        })
    }
}

/// The partial replies of one wave: `ranks` planes of `rows * topk` partial
/// rows, landed on the coordinator for `muse_tp_combine`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct PartialReplies {
    pub wire: PartialWire,
    pub ranks: usize,
    pub topk: usize,
    pub width: usize,
}

impl PartialReplies {
    /// Reply payload bytes one rank sends for `rows` rows.
    pub(crate) fn rank_bytes(&self, rows: usize) -> Result<usize> {
        ensure!(rows > 0 && self.ranks > 0 && self.topk > 0, "Muse partial reply geometry");
        Ok(rows * self.topk * self.wire.row_bytes(self.width)?)
    }

    /// Coordinator landing bytes for `rows` rows (every rank's partials).
    pub(crate) fn landing_bytes(&self, rows: usize) -> Result<usize> {
        Ok(self.rank_bytes(rows)? * self.ranks)
    }
}

/// The combine launches one wave needs: one per non-empty region, by format
/// (`muse_tp_combine_{bf16,nvfp4,fp8}`), each with its region's row count. A
/// region with no rows is skipped here, never launched with zero rows.
pub(crate) fn combine_launches(plan: &cuteafd_loader::families::muse_spark::replies::ReplyPlan)
    -> Vec<(&'static str, usize)> {
    use cuteafd_loader::families::muse_spark::replies::PartialFormat;
    let wide = match plan.wide { PartialFormat::Bf16 => "tp_combine_bf16", PartialFormat::Nvfp4 => "tp_combine_nvfp4",
        PartialFormat::Fp8 => "tp_combine_fp8" };
    [("tp_combine_bf16", plan.bf16_rows.len()), (wide, plan.wide_rows.len())]
        .into_iter().filter(|(_, rows)| *rows > 0).collect()
}

/// Most expert holders one combine reads (Spark ranks 2..8, an RTX pair, one RTX).
pub(crate) const MAX_RANKS: usize = 8;

/// Device layout of one wave's combine operands, written in one upload: the
/// BF16 and wide `rank_regions` tables (i64 device addresses, `MAX_RANKS`
/// each), then the BF16 and wide row maps (`capacity` u32 each).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct CombineTables { pub capacity: usize }

impl CombineTables {
    pub(crate) const TABLE_BYTES: usize = MAX_RANKS * 8;
    pub(crate) fn bytes(self) -> usize { 2 * Self::TABLE_BYTES + 2 * self.capacity * 4 }
    /// Byte offsets of (bf16 table, wide table, bf16 map, wide map).
    pub(crate) fn offsets(self) -> [usize; 4] {
        let maps = 2 * Self::TABLE_BYTES;
        [0, Self::TABLE_BYTES, maps, maps + self.capacity * 4]
    }

    /// The upload image of one wave: each rank's BF16 region base (its plane)
    /// and wide region base (plane + `wide_at`), then the region row maps in
    /// ascending wave-row order. Every base is 16-byte aligned (the kernel's
    /// vector loads rely on it): planes are allocation-aligned and `wide_at`
    /// a multiple of the 256-byte region alignment.
    pub(crate) fn image(self, planes: &[u64], plan: &ReplyPlanView<'_>) -> Result<Vec<u8>> {
        ensure!((1..=MAX_RANKS).contains(&planes.len()), "Muse combine reads 1..={MAX_RANKS} rank planes");
        let (bf16, wide) = (plan.bf16_rows, plan.wide_rows);
        ensure!(bf16.len() + wide.len() <= self.capacity, "Muse combine wave exceeds its {}-row maps", self.capacity);
        ensure!(plan.wide_at % 256 == 0, "Muse wide region offset {} is not region aligned", plan.wide_at);
        let mut out = vec![0u8; self.bytes()];
        let [bf16_table, wide_table, bf16_map, wide_map] = self.offsets();
        for (rank, &base) in planes.iter().enumerate() {
            ensure!(base % 16 == 0 && base != 0, "Muse rank {rank} plane {base:#x} is not 16-byte aligned");
            out[bf16_table + rank * 8..][..8].copy_from_slice(&(base as i64).to_le_bytes());
            let wide_base = base.checked_add(plan.wide_at as u64).context("Muse wide region address")?;
            out[wide_table + rank * 8..][..8].copy_from_slice(&(wide_base as i64).to_le_bytes());
        }
        for (at, rows) in [(bf16_map, bf16), (wide_map, wide)] {
            for (i, &row) in rows.iter().enumerate() { out[at + i * 4..][..4].copy_from_slice(&row.to_le_bytes()); }
        }
        Ok(out)
    }
}

/// One wave's regions as the combine reads them.
#[derive(Debug, Clone, Copy)]
pub(crate) struct ReplyPlanView<'p> {
    pub bf16_rows: &'p [u32],
    pub wide_rows: &'p [u32],
    /// Byte offset of the wide region in every rank's payload.
    pub wide_at: usize,
}

/// The combine launches of one wave over the uploaded tables: per non-empty
/// region, (program stem, table offset, map offset, region rows).
pub(crate) fn combine_plan(tables: CombineTables, plan: &cuteafd_loader::families::muse_spark::replies::ReplyPlan)
    -> Vec<(&'static str, usize, usize, usize)> {
    let [bf16_table, wide_table, bf16_map, wide_map] = tables.offsets();
    combine_launches(plan).into_iter().map(|(stem, rows)| if stem == "tp_combine_bf16" {
        (stem, bf16_table, bf16_map, rows)
    } else {
        (stem, wide_table, wide_map, rows)
    }).collect()
}

/// Expert width, top-k and the FP8 K32 wire row of a Muse expert request.
pub(crate) const WIDTH: usize = 4096;
pub(crate) const TOPK: usize = 8;
pub(crate) const EXPERTS: u32 = 256;
pub(crate) const WIRE_ROW_BYTES: usize = WIDTH + WIDTH / 32;

/// One request's rows in a wave: its identity (`source_request_id`) and row count.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct WaveSegment { pub request: u64, pub rows: usize }

/// The protocol request of one wave: FP8 K32 expert-width rows, top-8 routes,
/// and each row's partial-reply region from the shared per-request rule
/// (`ReplyPlan`), carried explicitly so the worker never derives it.
pub(crate) fn wave_request(layer: usize, sequence: u64, kind: cuteafd_transport::ExpertV2SourceKind,
    segments: &[WaveSegment], cutoff: cuteafd_loader::families::muse_spark::replies::Cutoff,
    routes: Vec<cuteafd_transport::ExpertProtocolV2RouteEntry>, wire: bytes::Bytes)
    -> Result<cuteafd_transport::ExpertProtocolV2Request> {
    use cuteafd_loader::families::muse_spark::replies::{PartialFormat, ReplyPlan};
    use cuteafd_transport::{ExpertProtocolV2Request, ExpertProtocolV2RowDescriptor, ExpertV2Dtype, PartialReplyPlan, PartialWire};
    let rows = segments.iter().map(|s| s.rows).sum::<usize>();
    ensure!(layer < 62 && (1..=4096).contains(&rows) && sequence > 0, "Muse Spark wave geometry or sequence");
    ensure!(routes.len() == rows * TOPK && routes.iter().all(|r| r.expert_id < EXPERTS
        && r.gate_weight.is_finite() && r.gate_weight >= 0.0), "Muse Spark route outside E256/K8 or invalid weight");
    ensure!(wire.len() == rows * WIRE_ROW_BYTES, "Muse Spark wire rows differ from the wave");
    let plan = ReplyPlan::new(cutoff, &segments.iter().map(|s| s.rows).collect::<Vec<_>>());
    let mut regions = vec![0u8; rows];
    for &row in &plan.wide_rows { regions[row as usize] = 1; }
    let wide = match plan.wide { PartialFormat::Nvfp4 => PartialWire::Nvfp4, PartialFormat::Fp8 => PartialWire::Fp8,
        PartialFormat::Bf16 => anyhow::bail!("Muse wide partial format cannot be BF16") };
    let mut descriptors = Vec::with_capacity(rows);
    for segment in segments {
        for _ in 0..segment.rows {
            let row = descriptors.len();
            descriptors.push(ExpertProtocolV2RowDescriptor { row_id: row as u64, source_kind: kind,
                source_request_id: segment.request, token_position: row as u64,
                route_offset: (row * TOPK) as u32, route_count: TOPK as u32 });
        }
    }
    ExpertProtocolV2Request::new_bytes(sequence, 17, layer as u32, WIDTH as u32, ExpertV2Dtype::Fp8E4m3Ue8m0K32,
        descriptors, routes, wire)?.with_partial_replies(PartialReplyPlan { wide, regions })
}

/// One wave in flight: the layer and rows it carries, in send order.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Wave { pub layer: usize, pub rows: usize, pub sequence: u64 }

/// What one wave carries beyond its rows: its requests (identity and rows, in
/// row order: each request's reply format follows from its own rows, never the
/// wave, `ReplyPlan`) and the source kind.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct WaveStep { pub segments: Vec<WaveSegment>, pub kind: cuteafd_transport::ExpertV2SourceKind }

impl WaveStep {
    pub(crate) fn rows(&self) -> usize { self.segments.iter().map(|s| s.rows).sum() }
}

/// The engine's view of the Spark experts. `send` copies the wave's routes and
/// FP8 K32 wire rows (`sources`: route ids, route weights, wire, the unit's slot
/// planes, written on the lane stream before the call) into the transport
/// staging on the lane stream and posts the request once they land (the host
/// never waits on the GPU); `land` makes the lane stream wait for the wave's
/// replies and uploads its `rank_regions` tables and row maps; the engine
/// combines (`RankDevice::combine`, which owns the programs, the layer's norm
/// and the lane), then calls `consumed` so the slot's planes are released to
/// the next wave behind the combine. Waves land in send order.
pub(crate) trait SparkExperts {
    fn send(&mut self, layer: usize, step: &WaveStep, sources: [cuteafd_ffi::CuteafdDeviceBuffer; 3]) -> Result<Wave>;
    fn land(&mut self, wave: Wave) -> Result<LandedWave>;
    fn consumed(&mut self, wave: Wave) -> Result<()>;
}

/// A landed wave, ready for `RankDevice::combine`.
#[derive(Debug, Clone)]
pub(crate) struct LandedWave {
    /// The fixed device tables ([`CombineTables`]) holding this wave's image.
    pub tables: cuteafd_ffi::CuteafdDeviceBuffer,
    pub layout: CombineTables,
    pub ranks: usize,
    pub plan: cuteafd_loader::families::muse_spark::replies::ReplyPlan,
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The study's reply sizes: NVFP4 partials are top-k x 4.5-bit rows per
    /// rank (2.25x one E-wide BF16 row at k = 8); FP8 is the 8.25-bit fallback.
    #[test]
    fn partial_reply_extents_match_the_layout_study() -> Result<()> {
        let nvfp4 = PartialReplies { wire: PartialWire::Nvfp4, ranks: 6, topk: 8, width: 4096 };
        assert_eq!(nvfp4.rank_bytes(1)?, 8 * (2048 + 256));
        assert_eq!(nvfp4.rank_bytes(1)? as f64 / (4096.0 * 2.0), 2.25);
        assert_eq!(nvfp4.landing_bytes(2752)?, 6 * 2752 * 8 * 2304);
        let fp8 = PartialReplies { wire: PartialWire::Fp8, ..nvfp4 };
        assert_eq!(fp8.rank_bytes(2752)?, 2752 * 8 * 4224);
        assert!(PartialWire::Nvfp4.row_bytes(4100).is_err());
        assert!(nvfp4.rank_bytes(0).is_err());
        Ok(())
    }

    #[test]
    fn empty_regions_are_never_launched() -> Result<()> {
        use cuteafd_loader::families::muse_spark::replies::{Cutoff, ReplyPlan, WideFormat};
        let cutoff = Cutoff::with_wide(64, 8, WideFormat::Nvfp4)?;
        assert_eq!(combine_launches(&ReplyPlan::new(cutoff, &[1, 8])), [("tp_combine_bf16", 9)]);
        assert_eq!(combine_launches(&ReplyPlan::new(cutoff, &[2752])), [("tp_combine_nvfp4", 2752)]);
        assert_eq!(combine_launches(&ReplyPlan::new(cutoff, &[1, 2000])),
            [("tp_combine_bf16", 1), ("tp_combine_nvfp4", 2000)]);
        let fp8 = Cutoff::with_wide(64, 8, WideFormat::Fp8)?;
        assert_eq!(combine_launches(&ReplyPlan::new(fp8, &[100])), [("tp_combine_fp8", 100)]);
        assert!(combine_launches(&ReplyPlan::new(cutoff, &[])).is_empty());
        Ok(())
    }

    /// The upload image: rank bases at any (non-contiguous, non-monotonic)
    /// addresses, wide bases at plane + wide_at, row maps in wave order; the
    /// launch plan points each region at its own table and map.
    #[test]
    fn combine_tables_point_every_rank_at_its_regions_in_place() -> Result<()> {
        use cuteafd_loader::families::muse_spark::replies::{Cutoff, ReplyPlan};
        let tables = CombineTables { capacity: 2752 };
        let plan = ReplyPlan::new(Cutoff::new(64, 8)?, &[1, 8, 70]);
        let bf16_bytes = plan.bf16_rows.len() * TOPK * WIDTH * 2;
        let wide_at = bf16_bytes.next_multiple_of(256);
        // The engine's offset is the wire's (`PartialReplyPlan::payload`).
        let mut regions = vec![0u8; 79];
        for &row in &plan.wide_rows { regions[row as usize] = 1; }
        assert_eq!(cuteafd_transport::PartialReplyPlan { wide: cuteafd_transport::PartialWire::Nvfp4, regions }
            .payload(TOPK, WIDTH)?.1, wide_at);
        let view = ReplyPlanView { bf16_rows: &plan.bf16_rows, wide_rows: &plan.wide_rows, wide_at };
        // Rank 2 below rank 0, gaps between (separate intake allocations).
        let planes = [0x7f00_0000_0000u64, 0x7f10_0000_0000, 0x7e00_0000_1000];
        let image = tables.image(&planes, &view)?;
        assert_eq!(image.len(), tables.bytes());
        let [bf16_table, wide_table, bf16_map, wide_map] = tables.offsets();
        let i64_at = |at: usize| i64::from_le_bytes(image[at..at + 8].try_into().unwrap());
        let u32_at = |at: usize| u32::from_le_bytes(image[at..at + 4].try_into().unwrap());
        for (rank, &plane) in planes.iter().enumerate() {
            assert_eq!(i64_at(bf16_table + rank * 8), plane as i64);
            assert_eq!(i64_at(wide_table + rank * 8), (plane + wide_at as u64) as i64);
        }
        assert_eq!(i64_at(bf16_table + 3 * 8), 0, "unused ranks stay null");
        assert_eq!((0..9).map(|i| u32_at(bf16_map + i * 4)).collect::<Vec<_>>(), (0..9).collect::<Vec<_>>());
        assert_eq!((0..70).map(|i| u32_at(wide_map + i * 4)).collect::<Vec<_>>(), (9..79).collect::<Vec<_>>());
        assert_eq!(combine_plan(tables, &plan), [("tp_combine_bf16", bf16_table, bf16_map, 9),
            ("tp_combine_nvfp4", wide_table, wide_map, 70)]);
        // Misaligned planes, a misaligned wide region and an overfull wave are refused.
        assert!(tables.image(&[0x7f00_0000_0008], &view).is_err());
        assert!(tables.image(&planes, &ReplyPlanView { wide_at: wide_at + 16, ..view }).is_err());
        assert!(CombineTables { capacity: 64 }.image(&planes, &view).is_err());
        assert!(tables.image(&[0x1000; 9], &view).is_err());
        Ok(())
    }

    #[test]
    fn wave_requests_carry_each_rows_region_and_round_trip() -> Result<()> {
        use cuteafd_loader::families::muse_spark::replies::{Cutoff, WideFormat};
        use cuteafd_transport::{ExpertProtocolV2Request, ExpertProtocolV2RouteEntry, ExpertV2SourceKind, PartialWire};
        let segments = [WaveSegment { request: 11, rows: 1 }, WaveSegment { request: 12, rows: 70 },
            WaveSegment { request: 13, rows: 8 }];
        let rows = 79;
        let routes = (0..rows * TOPK).map(|i| ExpertProtocolV2RouteEntry { row_index: (i / TOPK) as u32,
            expert_id: (i % 256) as u32, gate_weight: 0.125 }).collect::<Vec<_>>();
        let wire = bytes::Bytes::from(vec![0x5a; rows * WIRE_ROW_BYTES]);
        for wide in [WideFormat::Nvfp4, WideFormat::Fp8] {
            let request = wave_request(5, 3, ExpertV2SourceKind::Prefill, &segments, Cutoff::with_wide(64, 8, wide)?,
                routes.clone(), wire.clone())?;
            let plan = request.partial.as_ref().unwrap();
            // The 70-row segment is wide; the 1- and 8-row segments (decode, verify) are BF16.
            assert_eq!(plan.regions.iter().filter(|&&r| r == 1).count(), 70);
            assert!(plan.regions[..1].iter().all(|&r| r == 0) && plan.regions[71..].iter().all(|&r| r == 0));
            assert_eq!(plan.wide, if wide == WideFormat::Fp8 { PartialWire::Fp8 } else { PartialWire::Nvfp4 });
            assert_eq!(request.rows[1].source_request_id, 12);
            let mut bytes = Vec::new();
            request.encode_into(&mut bytes)?;
            assert_eq!(ExpertProtocolV2Request::decode(&bytes)?, request);
        }
        assert!(wave_request(5, 3, ExpertV2SourceKind::Decode, &segments, Cutoff::new(64, 8)?, routes.clone(),
            bytes::Bytes::from(vec![0; 10])).is_err(), "wire size");
        assert!(wave_request(62, 3, ExpertV2SourceKind::Decode, &segments, Cutoff::new(64, 8)?, routes,
            wire).is_err(), "layer");
        Ok(())
    }
}
