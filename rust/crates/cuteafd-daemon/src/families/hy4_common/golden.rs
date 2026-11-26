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
        rtx_layers: Vec::new(),
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
