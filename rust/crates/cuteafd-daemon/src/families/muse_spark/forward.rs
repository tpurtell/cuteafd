//! One Muse step over the coordinator ranks (one RTX, or a head-split pair):
//! the step metadata upload, the embedding, every layer in model order
//! (`RankDevice::layer`; the host enqueues each layer on rank 0 then rank 1,
//! and every cross-GPU wait is device-side), and the head.
//!
//! - Embedding: the BF16 table lives on rank 0 (single residency); each rank
//!   gathers its step's rows into `residual` (rank 1 over peer access) and
//!   runs the weightless `embed_norm` into `hidden`.
//! - Head (the last rank): `final_norm` of a contiguous range of at most
//!   [`HEAD_ROWS`] rows, the K6 head into F32 rows at its stored (N128-padded)
//!   width, trimmed to the vocabulary and softcapped. The logits are
//!   `[rows, vocab]` F32, contiguous, in the head rank's stored-width plane.
use super::{device::{PeerExchange, StepPlanes}, engine::RankResources, experts::SparkExperts, step::StepMetadata};
use anyhow::{ensure, Context, Result};
use cuteafd_ffi::{programs::Scalar, CuteafdDeviceBuffer as Buffer};
use cuteafd_loader::families::muse_spark::{lane::HEAD_ROWS, MuseSparkConfig};
use std::ffi::c_void;

fn view(base: Buffer, offset: usize, bytes: usize) -> Buffer {
    Buffer { ptr: base.ptr.cast::<u8>().wrapping_add(offset).cast(), bytes, ..base }
}

/// The step's token ids gathered from the entry rank's table into `residual`,
/// then `embed_norm` into `hidden`, on `rank`'s stream.
pub(crate) fn embed(cfg: &MuseSparkConfig, rank: &RankResources<'_>, lane: &super::lane::LaneViews,
    table: &crate::shared::token_io::TokenEmbedding<'_>, tokens: &[u32], stream: *mut c_void) -> Result<()> {
    let rows = tokens.len();
    let (ids, gathered) = (lane.plane("token_ids")?, lane.rows("residual", rows)?);
    rank.device.run(|| table.embed(tokens, ids, 1, gathered, stream))?;
    rank.device_with(cfg, stream, lane).norm(0, "embed_norm", None, rows, gathered, lane.rows("hidden", rows)?)
}

/// The head over `rows` (a contiguous range of the step's rows) on the head rank.
pub(crate) fn head(cfg: &MuseSparkConfig, rank: &RankResources<'_>, lane: &super::lane::LaneViews,
    rows: std::ops::Range<usize>, stream: *mut c_void) -> Result<Buffer> {
    let weights = rank.globals.head.as_ref().context("Muse head on a rank without the head")?;
    let (stored_plane, trimmed) = rank.head.as_ref().context("Muse head rank without logits planes")?;
    let (h, vocab, stored) = (cfg.hidden, cfg.vocab_size, weights.shape[0]);
    let device = rank.device_with(cfg, stream, lane);
    let count = rows.len();
    ensure!(count > 0 && rows.end <= lane.rows, "Muse head rows {rows:?} outside the step");
    let hidden = view(lane.plane("hidden")?, rows.start * h * 2, count * h * 2);
    let normed = view(lane.plane("normed")?, rows.start * h * 2, count * h * 2);
    // SAFETY: lane planes and the final norm weight are live on this device, ordered on `stream`.
    unsafe {
        device.programs.launch("final_norm", &[hidden, rank.globals.final_norm.buffer, normed],
            &[Scalar::I32(i32::try_from(count)?)], stream)?;
    }
    // One launch scores at most HEAD_ROWS rows (the head program's capacity and its
    // planes); the backend scores wider steps in calls of at most that many.
    ensure!(count <= HEAD_ROWS, "Muse head scores at most {HEAD_ROWS} rows per call, not {count}");
    let staged = Buffer { bytes: count * stored * 4, ..stored_plane.buffer };
    let trimmed = Buffer { bytes: count * vocab * 4, ..trimmed.buffer };
    rank.core.local(count, normed, weights, staged, stream)?;
    // SAFETY: both planes are this device's, distinct, and ordered on `stream`.
    unsafe { rank.device.library.copy_d2d_2d_async(trimmed, vocab * 4, staged, stored * 4, vocab * 4, count, stream)?; }
    // The softcap reads the trimmed rows and writes the logits contiguously into the stored plane.
    let logits = Buffer { bytes: count * vocab * 4, ..stored_plane.buffer };
    // SAFETY: as above.
    unsafe { device.programs.launch("softcap", &[trimmed, logits], &[Scalar::I32(i32::try_from(count)?)], stream)?; }
    Ok(logits)
}

/// One step: metadata, embedding, every layer, then the head over `head_rows`.
/// `experts` is rank 0's Spark path; `peers[r]` rank r's exchange end on a pair.
#[allow(clippy::too_many_arguments)]
pub(crate) fn step(cfg: &MuseSparkConfig, ranks: &[RankResources<'_>], streams: &[*mut c_void],
    table: &crate::shared::token_io::TokenEmbedding<'_>, tokens: &[u32], meta: &StepMetadata, prefill: bool,
    wave: super::experts::WaveStep,
    experts: &mut dyn SparkExperts, mut peers: Vec<&mut dyn PeerExchange>,
    local: &mut [super::onboard::OnboardedExperts<'_>], head_rows: std::ops::Range<usize>) -> Result<Buffer> {
    ensure!(ranks.len() == streams.len() && (ranks.len() == 1 || peers.len() == 2) && tokens.len() == meta.positions.len()
        && (local.is_empty() || local.len() == ranks.len()), "Muse step geometry");
    let planes: Vec<StepPlanes> = ranks.iter().zip(streams).map(|(rank, &stream)| {
        let bank = if prefill { 0 } else { rank.step.tables.decode_bank() };
        rank.device.run(|| rank.step.upload(rank.device.library, bank, meta, prefill, wave.clone(), stream))
    }).collect::<Result<_>>()?;
    for (rank, &stream) in ranks.iter().zip(streams) { embed(cfg, rank, &rank.lane.views, table, tokens, stream)?; }
    let mut experts = Some(experts);
    for layer in 0..cfg.layers {
        for (index, (rank, &stream)) in ranks.iter().zip(streams).enumerate() {
            let peer = peers.get_mut(index).map(|end| &mut **end as &mut dyn PeerExchange);
            if let Some(onboarded) = local.get_mut(index).filter(|onboarded| onboarded.holds(layer)) {
                rank.device.run(|| rank.device(cfg, stream).layer_local(layer, &planes[index], onboarded, peer))?;
                continue;
            }
            let path = if index == 0 { experts.as_deref_mut() } else { None };
            rank.device.run(|| rank.device(cfg, stream).layer(layer, &planes[index], path, peer))?;
        }
    }
    let last = ranks.len() - 1;
    ranks[last].device.run(|| head(cfg, &ranks[last], &ranks[last].lane.views, head_rows, streams[last]))
}

#[cfg(test)]
mod tests {
    /// A step uploads its metadata, embeds, runs every layer on each rank in
    /// rank order (rank 0 alone holds the Spark path), then scores on the last rank;
    /// the head trims the padded rows before the softcap.
    #[test]
    fn step_order_and_head_trim() {
        let source = include_str!("forward.rs").split("#[cfg(test)]").next().unwrap();
        let step = source.split("pub(crate) fn step(").nth(1).unwrap();
        let order = ["rank.step.upload(", "embed(cfg, rank, &rank.lane.views, table, tokens, stream)", "for layer in 0..cfg.layers",
            ".layer(layer, &planes[index], path, peer)", "head(cfg, &ranks[last], &ranks[last].lane.views, head_rows"];
        let at: Vec<usize> = order.iter().map(|needle| step.find(needle).unwrap()).collect();
        assert!(at.windows(2).all(|pair| pair[0] < pair[1]), "{order:?}");
        assert!(step.contains("if index == 0 { experts.as_deref_mut() } else { None }"));
        let head = source.split("pub(crate) fn head(").nth(1).unwrap().split("pub(crate) fn step(").next().unwrap();
        let order = ["\"final_norm\"", "rank.core.local(", "copy_d2d_2d_async(trimmed, vocab * 4, staged, stored * 4",
            "\"softcap\""];
        let at: Vec<usize> = order.iter().map(|needle| head.find(needle).unwrap()).collect();
        assert!(at.windows(2).all(|pair| pair[0] < pair[1]), "{order:?}");
    }

    /// The head rank's logits planes are what admission charged.
    #[test]
    fn head_planes_equal_the_charge() -> anyhow::Result<()> {
        let dir = tempfile::tempdir()?;
        cuteafd_loader::plan::testing::write_muse_spark_snapshot(dir.path(), &(0..62).collect::<Vec<_>>());
        let input = cuteafd_loader::plan::testing::BudgetedPlanInput::open(dir.path())?;
        let total = cuteafd_loader::placement::inventory::PRO_TOTAL_BYTES;
        let report = input.plan(&cuteafd_loader::plan::PlanOptions {
            placement: cuteafd_loader::plan::ExpertPlacement::from_spark_ranks(6),
            layout: Some(cuteafd_loader::plan::layout::LayoutOptions { host_memory: Some((256 << 30, 256 << 30)),
                rtx_bytes: vec![total; 2], head_split: true, ..Default::default() }), ..Default::default() })?;
        let placement = report.solver_placement.as_ref().unwrap();
        let charged = placement.items[1].iter().find(|item| item.group == "Muse head logits").unwrap().bytes;
        let (stored, trimmed) = cuteafd_loader::families::muse_spark::lane::head_planes(202112, 202048)?;
        assert_eq!(charged, (stored + trimmed) as u64);
        assert!(placement.items[0].iter().all(|item| item.group != "Muse head logits"), "head rank only");
        Ok(())
    }
}
