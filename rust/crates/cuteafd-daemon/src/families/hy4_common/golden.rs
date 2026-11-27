//! Shared golden scoring and binary-reading helpers for `hy4-golden` and `hy4f-golden`.
use anyhow::{ensure, Context, Result};
use sha2::{Digest, Sha256};
use std::path::Path;

pub(crate) fn read_u32_bin(path: &Path) -> Result<Vec<u32>> {
    let bytes = std::fs::read(path).with_context(|| format!("reading {}", path.display()))?;
    ensure!(bytes.len() % 4 == 0 && !bytes.is_empty(), "invalid u32 binary {}", path.display());
    Ok(bytes
        .chunks_exact(4)
        .map(|b| u32::from_le_bytes(b.try_into().unwrap()))
        .collect())
}

pub(crate) fn digest_f32(values: &[f32]) -> String {
    let mut h = Sha256::new();
    for v in values {
        h.update(v.to_le_bytes());
    }
    format!("{:x}", h.finalize())
}

pub(crate) fn build_probe_row(
    position: usize,
    logits: &[f32],
    wanted_ids: &[u32],
) -> cuteafd_api::openai::probe::ProbeRow {
    let finite = logits.iter().all(|v| v.is_finite());
    let mut max_v = f32::NEG_INFINITY;
    let mut argmax = 0u32;
    for (i, &v) in logits.iter().enumerate() {
        if v > max_v {
            max_v = v;
            argmax = i as u32;
        }
    }
    let max_f64 = f64::from(max_v);
    let sum_exp: f64 = logits.iter().map(|&v| (f64::from(v) - max_f64).exp()).sum();
    let lse = max_f64 + sum_exp.ln();
    let wanted: Vec<(u32, f32)> = wanted_ids
        .iter()
        .map(|&id| (id, (f64::from(logits[id as usize]) - lse) as f32))
        .collect();
    let mut fnv = 0xcbf2_9ce4_8422_2325u64;
    for &v in logits {
        for b in v.to_le_bytes() {
            fnv ^= u64::from(b);
            fnv = fnv.wrapping_mul(0x0100_0000_01b3);
        }
    }
    cuteafd_api::openai::probe::ProbeRow {
        position,
        hash: format!("{fnv:016x}"),
        argmax,
        top: Vec::new(),
        wanted,
        finite,
    }
}

/// Recompute each row's KL and NLL using direct engine tail in f64 (matching
/// `cuteafd_bench::fidelity_rows::score_compact`, avoiding `1 - q_mass` cancellation).
pub(crate) fn score_window_direct_tail(
    win: &cuteafd_bench::reference::Window,
    scored_slices: &[(usize, Vec<f32>)],
    vocab: usize,
) -> cuteafd_bench::reference::Fidelity {
    let want = win.want();
    let mut rows = Vec::with_capacity(win.positions.len());
    for (pos_id, slice) in scored_slices {
        rows.push(build_probe_row(*pos_id, slice, &want[pos_id]));
    }
    let mut score = win.score(&rows);
    for ((rec, pos), (_, slice)) in score.records.iter_mut().zip(&win.positions).zip(scored_slices) {
        let max_v = slice.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let max_f64 = f64::from(max_v);
        let lse = max_f64 + slice.iter().map(|&v| (f64::from(v) - max_f64).exp()).sum::<f64>().ln();
        let mut chosen = vec![false; vocab];
        let mut kl = 0.0f64;
        for t in &pos.top {
            let id = t.id as usize;
            chosen[id] = true;
            let q_lp = f64::from(slice[id]) - lse;
            kl += t.lp.exp() * (t.lp - q_lp);
        }
        if pos.tail_lp.is_finite() {
            let mut peak = f64::NEG_INFINITY;
            for (id, &v) in slice.iter().enumerate() {
                if !chosen[id] {
                    let lp = f64::from(v) - lse;
                    if lp > peak {
                        peak = lp;
                    }
                }
            }
            let tail_sum: f64 = slice
                .iter()
                .enumerate()
                .filter(|&(id, _)| !chosen[id])
                .map(|(_, &v)| ((f64::from(v) - lse) - peak).exp())
                .sum();
            let tail = peak + tail_sum.ln();
            kl += pos.tail_lp.exp() * (pos.tail_lp - tail);
        }
        rec.kl = kl.max(0.0);
        rec.nll = -(f64::from(slice[pos.next as usize]) - lse);
    }
    cuteafd_bench::reference::Fidelity::from_records(score.records)
}

pub(crate) fn load_published_reference(
    root: &Path,
    vocab: usize,
    expected_config: &str,
    expected_sha256: &str,
    label: &str,
) -> Result<cuteafd_bench::reference::Reference> {
    use cuteafd_bench::fidelity_dataset;
    let bytes = std::fs::read(root.join("manifest.json"))
        .with_context(|| format!("reading {}", root.join("manifest.json").display()))?;
    let manifest: serde_json::Value = serde_json::from_slice(&bytes)?;
    let digest = format!("{:x}", Sha256::digest(&bytes));
    ensure!(
        manifest["config"].as_str() == Some(expected_config) && digest == expected_sha256,
        "{label} fidelity manifest mismatch: config={:?} sha256={digest}",
        manifest["config"]
    );
    let reference = fidelity_dataset::load(root, &manifest)?;
    ensure!(reference.vocab == vocab, "{label} fidelity vocab {} != {vocab}", reference.vocab);
    Ok(reference)
}


/// Admit and bootstrap real Spark workers for a Hy4-family golden. The
/// caller owns the link and drains it as part of engine termination.
pub(crate) fn spark_link<'a>(
    library: &'a std::sync::Arc<cuteafd_ffi::NativeLibrary>,
    device: crate::shared::memory::device::Device<'a>,
    view: &cuteafd_loader::plan::checkpoint::CheckpointView,
    geometry: cuteafd_core::ExpertGeometry,
    rtx_layers: Vec<usize>,
    spark_layers: Vec<usize>,
    peers: &[std::net::SocketAddr],
    unit: usize,
    decode_rows: usize,
    prefill_waves: usize,
) -> Result<crate::shared::spark_intake::SparkLink<'a>> {
    use crate::shared::{placement::{spark_rings, SparkAdmission}, spark_intake::SparkLink};
    use cuteafd_core::{ExpertGeometry, SparkEndpoint, SparkIngress};
    use cuteafd_loader::placement::{families::{hy4, hy4_flash}, planner::PrefillShape, SparkWorkingSet};
    use cuteafd_transport::{
        expert::{SparkWorld, EXPERT_PROTOCOL_V2_FLAG_V41_COMPACT_BF16},
        worker_selection::ModelAdmission,
        ExpertV2Dtype, SparkSessionShape, TcpTransportConfig,
    };
    let (family, lanes) = match geometry {
        ExpertGeometry::HY4 => ("hy4", hy4::LANES),
        ExpertGeometry::HY4_FLASH => ("hy4_flash", hy4_flash::LANES),
        _ => anyhow::bail!("unsupported Hy4 golden Spark geometry {geometry:?}"),
    };
    let catalog = cuteafd_loader::read_expert_catalog_view(view)?;
    let capacity_rows = unit.max(decode_rows);
    let working = SparkWorkingSet {
        rtx_layers,
        spark_layers,
        spark_ranks: peers.len(),
        prefill_lanes: lanes,
    };
    let endpoint = SparkEndpoint {
        ring_depth: None,
        capacity: u32::try_from(capacity_rows)?,
        ingress: SparkIngress::Fp8K32,
        partial_reply_bytes: None,
    };
    let tcp = TcpTransportConfig::default();
    let backend = cuteafd_loader::plan::spark_executions::SparkBackend::for_catalog(&catalog)?;
    let executions = cuteafd_loader::plan::spark_executions::admit_family(
        family, &backend.exports(), Some(decode_rows as u32), unit as u64, prefill_waves as u64,
    )?;
    let model = ModelAdmission::for_clients(geometry, endpoint, spark_rings(tcp.max_frame_bytes)?, 1, executions)?;
    SparkAdmission::plan_view(view, geometry, Some(&catalog), &working, peers.len(), model, None)?
        .select(peers)?;
    let prefill = PrefillShape { rows: capacity_rows as u64, waves: prefill_waves as u64 };
    let replies = cuteafd_core::spark_reply::requested();
    let links = match geometry {
        ExpertGeometry::HY4 => hy4::spark_link(0, peers.len(), prefill, replies),
        ExpertGeometry::HY4_FLASH => hy4_flash::spark_link(0, peers.len(), prefill, replies),
        _ => unreachable!("geometry checked above"),
    }.map_err(|e| anyhow::anyhow!("{family} golden Spark link refused: {e}"))?;
    let mut link = device.run(|| SparkLink::prepared(
        library, SparkWorld::canonical(peers.len())?, peers, tcp,
        links.first().context("golden Spark geometry absent")?.clone(),
    ))?;
    link.enable_terminal_ownership(std::sync::Arc::clone(library))?;
    let mut session = SparkSessionShape::new(
        u32::try_from(capacity_rows)?, geometry.hidden, geometry.topk, ExpertV2Dtype::Fp8E4m3Ue8m0K32,
    )?;
    session.flags |= EXPERT_PROTOCOL_V2_FLAG_V41_COMPACT_BF16;
    device.run(|| link.bootstrap(&session))?;
    Ok(link)
}

/// Shared multi-wave and concurrent two-sequence byte-identity proof returned by
/// `hy4-golden --verify-waves` and `hy4f-golden --verify-waves`.
pub(crate) struct TwoSeqWaveProof {
    pub(crate) gpus: usize,
    pub(crate) waves: usize,
    pub(crate) unit_rows: usize,
    pub(crate) prompt_a_len: usize,
    pub(crate) prompt_b_len: usize,
    pub(crate) units_a: usize,
    pub(crate) units_b: usize,
    pub(crate) logits_a_solo: Vec<f32>,
    pub(crate) logits_b_solo: Vec<f32>,
    pub(crate) tail_a: [u32; 2],
    pub(crate) tail_b: [u32; 2],
    pub(crate) buffer_summary: Vec<String>,
}

/// Family adapter for the shared two-sequence multi-wave byte-identity proof.
pub(crate) trait TwoSeqWaveEngine {
    type Placement;

    fn gpus(&self) -> usize;
    fn prefill_waves(&self) -> usize;
    fn max_sequences(&self) -> usize;
    fn max_rows(&self) -> usize;
    fn max_pages(&self) -> usize;
    fn context_split(&self) -> bool;
    fn layers(&self) -> usize;
    fn vocab_size(&self) -> usize;
    fn page_rows(&self) -> usize {
        64
    }
    fn check_wave_buffers(&self) -> Result<Vec<String>> {
        Ok(Vec::new())
    }
    fn reset_for_wave_proof(&mut self) -> Result<()>;
    fn make_wave_placement(&self, pages: Vec<u32>, slot: usize) -> Self::Placement;
    fn placement_tail(&self, _placement: &Self::Placement) -> [u32; 2] {
        [0, 0]
    }
    fn run_wave_units(
        &mut self,
        placements: &mut [Self::Placement],
        sources: &[&[u32]],
        units: &[crate::shared::prefill_pipeline::Unit],
        on_tile: &mut dyn FnMut(&crate::shared::prefill_pipeline::Unit, usize, &[f32]) -> Result<()>,
    ) -> Result<()>;
}

pub(crate) fn count_f32_bit_diffs(a: &[f32], b: &[f32]) -> usize {
    a.iter().zip(b.iter()).filter(|(x, y)| x.to_bits() != y.to_bits()).count() + a.len().abs_diff(b.len())
}

/// Shared 4-phase proof of single-sequence multi-wave and concurrent two-sequence
/// multi-wave byte-identity (`single_mw_diffs=0`, `conc_a_diffs=0`, `conc_b_diffs=0`).
pub(crate) fn verify_concurrent_two_seq_waves<E: TwoSeqWaveEngine>(
    engine: &mut E,
    prompt_a: &[u32],
    prompt_b: &[u32],
    unit_rows: usize,
    last_layer: usize,
) -> Result<TwoSeqWaveProof> {
    use crate::shared::prefill_pipeline::{prefill_units, sequence_units, Unit, UnitSegment};

    let page_rows = engine.page_rows();
    let layers = engine.layers();
    let vocab = engine.vocab_size();
    let waves = engine.prefill_waves();
    let gpus = engine.gpus();
    let context_split = engine.context_split();

    ensure!(
        waves >= 2 && engine.max_sequences() >= 2,
        "verify_concurrent_two_seq_waves requires prefill_waves >= 2 and max_sequences >= 2"
    );
    ensure!(
        !prompt_a.is_empty() && !prompt_b.is_empty() && unit_rows > 0 && unit_rows <= engine.max_rows(),
        "invalid prompt or unit_rows for wave verification"
    );
    ensure!(
        last_layer + 1 == layers,
        "verify_concurrent_two_seq_waves requires full {layers} layers"
    );
    let buffer_summary = engine.check_wave_buffers()?;

    let pages_a = prompt_a.len().div_ceil(page_rows);
    let pages_b = prompt_b.len().div_ceil(page_rows);
    let base_b = if context_split { (pages_a + 1) & !1 } else { pages_a };
    let pool_pages = if context_split { engine.max_pages() * 2 } else { engine.max_pages() };
    ensure!(
        base_b + pages_b <= pool_pages,
        "two-sequence wave proof needs {} pages, engine has {}",
        base_b + pages_b,
        pool_pages
    );

    let make_solo_units = |len: usize, w: usize| -> Result<Vec<Unit>> {
        let mut units = sequence_units(len, unit_rows, w, layers)?;
        let last_idx = units.len() - 1;
        for (i, u) in units.iter_mut().enumerate() {
            let span = u.rows.clone();
            u.segments = vec![UnitSegment {
                sequence: 0,
                rows: 0..span.len(),
                tokens: span,
                prompt_end: i == last_idx,
                snapshot: false,
                head: true,
            }];
        }
        Ok(units)
    };

    // 1. Prompt A solo at waves=1.
    engine.reset_for_wave_proof()?;
    let units_a1 = make_solo_units(prompt_a.len(), 1)?;
    let mut p_a1 = [engine.make_wave_placement((0..pages_a as u32).collect(), 0)];
    let mut logits_a_solo = vec![0f32; prompt_a.len() * vocab];
    for mut unit in units_a1 {
        unit.index = 0;
        unit.slot = 0;
        let src = &prompt_a[unit.rows.clone()];
        engine.run_wave_units(&mut p_a1, &[src], &[unit], &mut |u, row_off, tile| {
            let dst = (u.rows.start + row_off) * vocab;
            logits_a_solo[dst..dst + tile.len()].copy_from_slice(tile);
            Ok(())
        })?;
    }
    let tail_a_solo = engine.placement_tail(&p_a1[0]);

    // 2. Prompt B solo at waves=1.
    engine.reset_for_wave_proof()?;
    let units_b1 = make_solo_units(prompt_b.len(), 1)?;
    let mut p_b1 = [engine.make_wave_placement((base_b as u32..(base_b + pages_b) as u32).collect(), 1)];
    let mut logits_b_solo = vec![0f32; prompt_b.len() * vocab];
    for mut unit in units_b1 {
        unit.index = 0;
        unit.slot = 0;
        let src = &prompt_b[unit.rows.clone()];
        engine.run_wave_units(&mut p_b1, &[src], &[unit], &mut |u, row_off, tile| {
            let dst = (u.rows.start + row_off) * vocab;
            logits_b_solo[dst..dst + tile.len()].copy_from_slice(tile);
            Ok(())
        })?;
    }
    let tail_b_solo = engine.placement_tail(&p_b1[0]);

    // 3. Prompt A solo at waves=W (cross-wave boundary check).
    engine.reset_for_wave_proof()?;
    let units_aw = make_solo_units(prompt_a.len(), waves)?;
    ensure!(
        units_aw.len() >= 2,
        "prompt A must span >= 2 units at unit_rows={unit_rows} to cross wave boundary"
    );
    let mut p_aw = [engine.make_wave_placement((0..pages_a as u32).collect(), 0)];
    let mut logits_a_mw = vec![0f32; prompt_a.len() * vocab];
    engine.run_wave_units(&mut p_aw, &[prompt_a], &units_aw, &mut |u, row_off, tile| {
        let dst = (u.rows.start + row_off) * vocab;
        logits_a_mw[dst..dst + tile.len()].copy_from_slice(tile);
        Ok(())
    })?;
    let tail_a_mw = engine.placement_tail(&p_aw[0]);
    let diff_a_mw = count_f32_bit_diffs(&logits_a_solo, &logits_a_mw);
    ensure!(
        diff_a_mw == 0 && tail_a_mw == tail_a_solo,
        "single-sequence multi-wave (waves={waves}) diverged from waves=1: diff_elems={diff_a_mw} tail={tail_a_mw:?} != {tail_a_solo:?}"
    );

    // 4. Prompt A and Prompt B concurrently in interleaved waves.
    engine.reset_for_wave_proof()?;
    let a_ranges = prefill_units(prompt_a.len(), unit_rows, waves, 256)?;
    let b_ranges = prefill_units(prompt_b.len(), unit_rows, waves, 256)?;
    let mut interleaved = Vec::with_capacity(a_ranges.len() + b_ranges.len());
    let max_len = a_ranges.len().max(b_ranges.len());
    let mut idx = 0usize;
    for step in 0..max_len {
        if let Some(span) = a_ranges.get(step).cloned() {
            interleaved.push(Unit {
                index: idx,
                sequence: 0,
                segments: vec![UnitSegment {
                    sequence: 0,
                    rows: 0..span.len(),
                    tokens: span.clone(),
                    prompt_end: step + 1 == a_ranges.len(),
                    snapshot: false,
                    head: true,
                }],
                rows: span,
                layers: 0..layers,
                slot: idx % waves,
                barrier: false,
            });
            idx += 1;
        }
        if let Some(span) = b_ranges.get(step).cloned() {
            interleaved.push(Unit {
                index: idx,
                sequence: 1,
                segments: vec![UnitSegment {
                    sequence: 1,
                    rows: 0..span.len(),
                    tokens: span.clone(),
                    prompt_end: step + 1 == b_ranges.len(),
                    snapshot: false,
                    head: true,
                }],
                rows: span,
                layers: 0..layers,
                slot: idx % waves,
                barrier: false,
            });
            idx += 1;
        }
    }
    let mut placements = [
        engine.make_wave_placement((0..pages_a as u32).collect(), 0),
        engine.make_wave_placement((base_b as u32..(base_b + pages_b) as u32).collect(), 1),
    ];
    let mut logits_a_conc = vec![0f32; prompt_a.len() * vocab];
    let mut logits_b_conc = vec![0f32; prompt_b.len() * vocab];
    engine.run_wave_units(&mut placements, &[prompt_a, prompt_b], &interleaved, &mut |u, row_off, tile| {
        let dst = (u.rows.start + row_off) * vocab;
        if u.sequence == 0 {
            logits_a_conc[dst..dst + tile.len()].copy_from_slice(tile);
        } else {
            logits_b_conc[dst..dst + tile.len()].copy_from_slice(tile);
        }
        Ok(())
    })?;
    let tail_a_conc = engine.placement_tail(&placements[0]);
    let tail_b_conc = engine.placement_tail(&placements[1]);
    let diff_a_conc = count_f32_bit_diffs(&logits_a_solo, &logits_a_conc);
    let diff_b_conc = count_f32_bit_diffs(&logits_b_solo, &logits_b_conc);
    ensure!(
        diff_a_conc == 0 && tail_a_conc == tail_a_solo,
        "concurrent 2-seq multi-wave prompt A diverged: diff_elems={diff_a_conc} tail={tail_a_conc:?} != {tail_a_solo:?}"
    );
    ensure!(
        diff_b_conc == 0 && tail_b_conc == tail_b_solo,
        "concurrent 2-seq multi-wave prompt B diverged: diff_elems={diff_b_conc} tail={tail_b_conc:?} != {tail_b_solo:?}"
    );

    Ok(TwoSeqWaveProof {
        gpus,
        waves,
        unit_rows,
        prompt_a_len: prompt_a.len(),
        prompt_b_len: prompt_b.len(),
        units_a: a_ranges.len(),
        units_b: b_ranges.len(),
        logits_a_solo,
        logits_b_solo,
        tail_a: tail_a_solo,
        tail_b: tail_b_solo,
        buffer_summary,
    })
}
