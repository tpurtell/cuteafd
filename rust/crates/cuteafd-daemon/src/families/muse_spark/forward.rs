//! One Muse step over the coordinator ranks (one RTX, or a head-split pair):
//! the step metadata upload, the embedding, every layer in model order
//! (`RankDevice::layer` for eager execution; graph-enabled pairs submit both
//! ranks phase by phase before the next capture), and the head. Cross-GPU
//! waits remain device-side.
//!
//! - Embedding: the BF16 table lives on rank 0 (single residency); each rank
//!   gathers its step's rows into `residual` (rank 1 over peer access) and
//!   runs the weightless `embed_norm` into `hidden`.
//! - Head (the last rank): `final_norm` of a contiguous range of at most
//!   [`HEAD_ROWS`] rows, the selected head into F32 rows at its stored (optionally N128-padded)
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
    rank.device.run(|| {
        table.embed(tokens, ids, 1, gathered, stream)?;
        rank.device_with(cfg, stream, lane).norm(0, "embed_norm", None, rows, gathered, lane.rows("hidden", rows)?)
    })
}

/// Replace media rows after the weightless text norm on every participating rank.
pub(crate) fn inject_media(cfg: &MuseSparkConfig, rank: &RankResources<'_>, lane: &super::lane::LaneViews,
    media: &cuteafd_engine::media::MediaChunk, rows: usize, stream: *mut c_void) -> Result<()> {
    if media.indices.is_empty() { return Ok(()); }
    // inject_host drains these admitted scratch planes before later layer work reuses them.
    rank.device.run(|| Ok(rank.device.library.embedding_injection()?.inject_host(
        &media.features, &media.indices, lane.plane("residual")?, lane.plane("token_ids")?,
        lane.rows("hidden", rows)?, rows, cfg.hidden, 1, stream)?))
}

/// The head over `rows` (a contiguous range of the step's rows) on the head rank.
pub(crate) fn head(cfg: &MuseSparkConfig, rank: &RankResources<'_>, lane: &super::lane::LaneViews,
    rows: std::ops::Range<usize>, stream: *mut c_void) -> Result<Buffer> {
    rank.check_head_stream(stream)?;
    let (stored_plane, trimmed) = rank.head.as_ref().context("Muse head rank without logits planes")?;
    let (h, vocab, stored) = (cfg.hidden, cfg.vocab_size, rank.globals.head.as_ref().map_or(cfg.vocab_size, |weights| weights.shape[0]));
    let device = rank.device_with(cfg, stream, lane);
    let count = rows.len();
    ensure!(count > 0 && count <= HEAD_ROWS && rows.end <= lane.rows,
        "Muse head rows {rows:?} outside the step or its {HEAD_ROWS}-row capacity");
    let hidden = view(lane.plane("hidden")?, rows.start * h * 2, count * h * 2);
    let normed = view(lane.plane("normed")?, rows.start * h * 2, count * h * 2);
    // SAFETY: lane planes and the final norm weight are live on this device, ordered on `stream`.
    unsafe {
        device.programs.launch("final_norm", &[hidden, rank.globals.final_norm.buffer, normed],
            &[Scalar::I32(i32::try_from(count)?)], stream)?;
    }
    // One launch scores at most HEAD_ROWS rows (the head program's capacity and its
    // planes); the backend scores wider steps in calls of at most that many.
    let staged = Buffer { bytes: count * stored * 4, ..stored_plane.buffer };
    let trimmed = Buffer { bytes: count * vocab * 4, ..trimmed.buffer };
    let logits = Buffer { bytes: count * vocab * 4, ..stored_plane.buffer };
    project_head(cfg, rank, count, normed, staged, trimmed, logits, stream)?;
    Ok(logits)
}

/// Already-normalized target or draft rows through the same selected head, crop and softcap.
/// Logit planes are caller-owned; only the serialized native handle/workspace is borrowed.
#[allow(clippy::too_many_arguments)]
pub(crate) fn project_head(cfg: &MuseSparkConfig, rank: &RankResources<'_>, rows: usize,
    input: Buffer, staged: Buffer, trimmed: Buffer, logits: Buffer, stream: *mut c_void) -> Result<()> {
    rank.check_head_stream(stream)?;
    let (vocab, stored) = (cfg.vocab_size,
        rank.globals.head.as_ref().map(|head| head.shape[0])
            .or_else(|| rank.globals.head_native.as_ref().map(|head| head.layout.shape[0])).unwrap_or(cfg.vocab_size));
    if let Some(weights) = &rank.globals.head {
        rank.core.head_f32(rows, input, weights, staged, stream)?;
    } else if let Some(weights) = &rank.globals.head_native {
        rank.core.native_head_f32(rows, input, weights, staged, stream)?;
    } else {
        let weights = rank.globals.head_bf16.as_ref().context("Muse BF16 head weights absent")?;
        let head = rank.native_head.as_ref().context("Muse BF16 head handle absent")?;
        super::core::head_chunks(rows, input, staged, cfg.hidden, vocab, |count, x, y| {
            // SAFETY: validated disjoint device row views and rank-owned weights/handle,
            // serialized target and drafter use on the owning stream.
            unsafe { crate::shared::target_head::launch(crate::shared::target_head::TargetHead::configured(),
                head, x.ptr, weights.buffer.ptr, y.ptr.cast(), count, stream) }
        })?;
    }
    // SAFETY: caller retains distinct staged/trimmed planes; logits may reuse staged
    // only after the ordered copy, exactly as in the trellis target path.
    unsafe {
        rank.device.library.copy_d2d_2d_async(trimmed, vocab * 4, staged, stored * 4, vocab * 4, rows, stream)?;
        rank.programs.launch("softcap", &[trimmed, logits], &[Scalar::I32(i32::try_from(rows)?)], stream)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum PairPhase { Attention, AttentionExchange, Frontend, FrontendExchange, Experts, Partials, Combine, Backend, Residual }

fn pair_layer(onboarded: bool, mut submit: impl FnMut(PairPhase, usize) -> Result<()>) -> Result<()> {
    use PairPhase::*;
    // Instantiation may wait for earlier work on the device. Submit both sides
    // of each eager peer exchange before either rank enters its next capture.
    for phase in [Attention, AttentionExchange, Frontend, FrontendExchange, Experts, Partials, Combine, Backend, Residual] {
        if matches!(phase, Partials | Combine) && !onboarded { continue; }
        for rank in 0..2 { submit(phase, rank)?; }
    }
    Ok(())
}

/// One step: metadata, embedding, every layer, then the head over `head_rows`.
/// `experts` is rank 0's Spark path; `peers[r]` rank r's exchange end on a pair.
#[allow(clippy::too_many_arguments)]
pub(crate) fn step<'a>(cfg: &MuseSparkConfig, ranks: &[RankResources<'a>], streams: &[*mut c_void],
    table: &crate::shared::token_io::TokenEmbedding<'_>, tokens: &[u32], media: &cuteafd_engine::media::MediaChunk,
    meta: &StepMetadata, prefill: bool,
    wave: super::experts::WaveStep,
    experts: &mut dyn SparkExperts, mut peers: Vec<&mut dyn PeerExchange>,
    local: &mut [super::onboard::OnboardedExperts<'_>], draft: Option<&super::draft::MuseDraft<'_>>,
    mut probe: Option<&mut crate::shared::draft::probe::RoundProbe<'_>>, head_rows: std::ops::Range<usize>,
    graphs: &[super::graphs::Graphs<'a>])
    -> Result<Buffer> {
    ensure!(ranks.len() == streams.len() && (ranks.len() == 1 || peers.len() == 2) && tokens.len() == meta.positions.len()
        && (local.is_empty() || local.len() == ranks.len())
        && (graphs.is_empty() || graphs.len() == ranks.len()), "Muse step geometry");
    let planes: Vec<StepPlanes> = ranks.iter().zip(streams).map(|(rank, &stream)| {
        let bank = if prefill { 0 } else { rank.step.tables.decode_bank() };
        rank.device.run(|| rank.step.upload(rank.device.library, bank, meta, prefill, wave.clone(), stream))
    }).collect::<Result<_>>()?;
    for (rank, &stream) in ranks.iter().zip(streams) {
        embed(cfg, rank, &rank.lane.views, table, tokens, stream)?;
        inject_media(cfg, rank, &rank.lane.views, media, tokens.len(), stream)?;
    }
    let mut experts = Some(experts);
    for layer in 0..cfg.layers {
        if ranks.len() == 2 && !graphs.is_empty() && !prefill && planes.iter().all(|plane| plane.trace.is_none()) {
            let onboarded = local.first().is_some_and(|local| local.holds(layer));
            ensure!(local.iter().all(|local| local.holds(layer) == onboarded), "Muse TP2 local layer differs between ranks");
            let mut partials = [Vec::new(), Vec::new()];
            pair_layer(onboarded, |phase, index| {
                let rank = &ranks[index];
                let stream = streams[index];
                let mut device = rank.device(cfg, stream);
                device.graphs = graphs.get(index);
                let step = &planes[index];
                let rows = step.rows;
                rank.device.run(|| match phase {
                    PairPhase::Attention => device.attention_prepare(layer, step),
                    PairPhase::AttentionExchange => device.attention_exchange(layer, rows, &mut *peers[index]),
                    PairPhase::Frontend => device.frontend_prepare(layer, rows),
                    PairPhase::FrontendExchange => device.frontend_exchange(layer, rows, &mut *peers[index], onboarded),
                    PairPhase::Experts if onboarded => {
                        device.expert_wire(rows)?;
                        device.local_partials(layer, rows, &mut local[index])
                    }
                    PairPhase::Experts if index == 0 => {
                        device.expert_wire(rows)?;
                        let experts = experts.as_deref_mut().context("Muse TP2 missing Spark path")?;
                        let wave = experts.send(layer, &step.wave, device.wave_sources(rows)?)?;
                        let landed = experts.land(wave)?;
                        device.combine(layer, rows, landed.ranks, landed.tables, landed.layout, &landed.plan)?;
                        experts.consumed(wave)
                    }
                    PairPhase::Experts => Ok(()),
                    PairPhase::Partials => {
                        partials[index] = device.local_exchange(rows, &local[index], Some(&mut *peers[index]))?;
                        Ok(())
                    }
                    // tables_for may wait for ordered host staging. Do both
                    // ranks before the backend queues another peer wait.
                    PairPhase::Combine => device.local_combine(layer, rows, &mut local[index], &partials[index]),
                    PairPhase::Backend => device.moe_backend_halves(layer, rows, &mut *peers[index], onboarded),
                    PairPhase::Residual => device.finish_residual(layer, step),
                })
            })?;
        } else {
            for (index, (rank, &stream)) in ranks.iter().zip(streams).enumerate() {
                let mut device = rank.device(cfg, stream);
                device.graphs = graphs.get(index).filter(|_| !prefill && planes[index].trace.is_none());
                let peer = peers.get_mut(index).map(|end| &mut **end as &mut dyn PeerExchange);
                if let Some(onboarded) = local.get_mut(index).filter(|onboarded| onboarded.holds(layer)) {
                    rank.device.run(|| device.layer_local(layer, &planes[index], onboarded, peer))?;
                    continue;
                }
                let path = if index == 0 { experts.as_deref_mut() } else { None };
                rank.device.run(|| device.layer(layer, &planes[index], path, peer))?;
            }
        }
        // The drafter taps the head rank's hidden rows after the layer, on that rank's stream.
        let last = ranks.len() - 1;
        if let Some(draft) = draft {
            draft.tap(layer, ranks[last].lane.views.plane("hidden")?, 0, tokens.len(), 0)?;
        }
        // The draft policy's round: this layer's route ids and its end mark, on the head rank's stream.
        if let Some(probe) = probe.as_deref_mut() {
            let ids = ranks[last].lane.views.rows("route_ids", tokens.len())?;
            ranks[last].device.run(|| {
                // SAFETY: the router wrote these ids on this stream; the ring is read after the step's sync.
                if probe.wants_ring(false) { unsafe { probe.ring(ranks[last].device.library, layer, ids, tokens.len(), streams[last])?; } }
                // SAFETY: the probe's events belong to the head rank's device and stream.
                unsafe { probe.mark(layer, streams[last]) }
            })?;
        }
    }
    let last = ranks.len() - 1;
    let mut logits = None;
    ranks[last].device.run(|| {
        let launch = || {
            logits = Some(head(cfg, &ranks[last], &ranks[last].lane.views, head_rows.clone(), streams[last])?);
            Ok(())
        };
        if let Some(graphs) = graphs.get(last).filter(|_| !prefill && planes[last].trace.is_none()) {
            graphs.run(super::graphs::Key { layer: cfg.layers, rows: head_rows.len(), stride: head_rows.start,
                section: super::graphs::Section::Head }, streams[last], launch)?;
        } else { let mut launch = launch; launch()?; }
        // Replay does not execute the closure: head's persistent stored plane is the output.
        let (stored, _) = ranks[last].head.as_ref().context("Muse head planes absent")?;
        Ok(logits.unwrap_or(Buffer { bytes: head_rows.len() * cfg.vocab_size * 4, ..stored.buffer }))
    })
}

#[cfg(test)]
mod tests {
    use super::{pair_layer, PairPhase};

    #[test]
    fn tp2_peer_publications_precede_each_capture_for_remote_and_local_layers() -> anyhow::Result<()> {
        use PairPhase::*;
        for onboarded in [false, true] {
            let mut pending = [false; 2];
            let mut calls = Vec::new();
            let mut submit = |phase, rank| -> anyhow::Result<()> {
                if matches!(phase, Attention | Frontend | Experts | Combine | Residual) {
                    anyhow::ensure!(!pending.iter().any(|&waiting| waiting), "capture before peer submission: {phase:?} rank{rank}");
                } else {
                    pending[rank] = true;
                    if pending.iter().all(|&submitted| submitted) { pending = [false; 2]; }
                }
                calls.push((phase, rank));
                Ok(())
            };
            // Include the next layer/step: a last-rank join must not leak into
            // the next attention capture. The same walker drives real launches.
            for _ in 0..3 { pair_layer(onboarded, &mut submit)?; }
            assert!(submit(AttentionExchange, 0).is_ok());
            assert!(submit(Frontend, 0).is_err(), "old layer-major rank order must fail the model");
            drop(submit);
            assert_eq!(calls.iter().filter(|(phase, _)| *phase == Partials).count(), if onboarded { 6 } else { 0 });
            for pair in calls[..calls.len() - 1].chunks_exact(2) {
                assert_eq!(pair[0].0, pair[1].0);
                assert_eq!((pair[0].1, pair[1].1), (0, 1));
            }
        }
        Ok(())
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
        for rank in 0..2 {
            let items = &report.solver_placement.as_ref().unwrap().items[rank];
            let owners = cuteafd_loader::placement::reserve::breakdown(format!("gpu{rank}"),
                cuteafd_core::memory_policy::DeviceKind::Gpu, total, 0, items);
            let graph = owners.reserve.iter().filter(|row| row.name == "graph reserve").collect::<Vec<_>>();
            // The untraced bootstrap is empty; an explicit policy may charge a
            // nonzero reserve, but never a second family-named owner or tail.
            let policy = cuteafd_loader::serving_capacity::muse_graphs::policy(2, report.onboard_layers.unwrap_or(0), rank)?;
            assert_eq!(graph.iter().map(|row| row.bytes).sum::<u64>(), policy.bytes);
            assert!(graph.len() <= 1);
            assert!(!owners.reserve.iter().any(|row| row.name != "graph reserve" && row.name.contains("graph")));
        }
        let placement = report.solver_placement.as_ref().unwrap();
        let charged = placement.items[1].iter().find(|item| item.group == "Muse head logits").unwrap().bytes;
        let (stored, trimmed) = cuteafd_loader::families::muse_spark::lane::head_planes(202112, 202048)?;
        assert_eq!(charged, (stored + trimmed) as u64);
        assert!(placement.items[0].iter().all(|item| item.group != "Muse head logits"), "head rank only");
        Ok(())
    }
}
