//! Exact native allocation shapes shared by offline admission and device owners.
use super::{
    coordinator::{CoordinatorContract, RankWeightLayout},
    ownership::ExecutionOwnership,
    Config,
};
use crate::plan::checkpoint::Checkpoint;
use anyhow::{ensure, Context, Result};
const H: usize = 4096;
const Q: usize = 64 * 192;
const KV: usize = 4 * 192;
const TILE: usize = 64;
const CAPACITY: usize = 131072;
const SELECT: usize = 4 * 4104;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RankAdmission {
    pub immutable: u64,
    pub workspace: u64,
}

pub fn swa_planes(rows: usize, heads: usize) -> Result<[usize; 16]> {
    ensure!(
        (1..=4096).contains(&rows) && matches!(heads, 32 | 64),
        "Step-5 SWA allocation geometry"
    );
    let q = heads * 192;
    let kv = heads / 16 * 192;
    let pages = rows.div_ceil(64);
    Ok([
        rows * H * 4,
        rows * H * 2,
        rows * q * 2,
        rows * kv * 2,
        pages * 64 * kv * 2,
        rows * q * 2,
        pages * 64 * kv * 2,
        rows * heads * 2,
        rows * q * 2,
        rows * q * 2,
        rows * H * if heads == 32 { 4 } else { 2 },
        rows * 8,
        rows * 4,
        pages * 4,
        rows * 512 * 4,
        rows * H * 4,
    ]
    .map(|n| n.max(256)))
}

pub fn csa_planes(rows: usize) -> Result<[usize; 32]> {
    ensure!((1..=8192).contains(&rows), "Step-5 CSA allocation rows");
    let pages = rows.div_ceil(64);
    let regions = rows.div_ceil(8);
    Ok([
        rows * H * 4,
        rows * H * 2,
        rows * Q * 2,
        rows * KV * 2,
        pages * 64 * KV * 2,
        rows * Q * 2,
        pages * 64 * KV * 2,
        rows * 64 * 2,
        rows * Q * 2,
        rows * Q * 2,
        rows * H * 2,
        rows * H * 4,
        rows * 8,
        rows * 4,
        pages * 4,
        rows * H * 2,
        rows * 256 * 2,
        rows * 256 * 2,
        rows * H * 2,
        rows * 256 * 2,
        rows * 16 * 4,
        rows * 16 * 2,
        regions * 4,
        regions * 4,
        CAPACITY * 256,
        regions * 256 * 4,
        TILE * 4 * CAPACITY * 4,
        TILE * SELECT * 4,
        TILE * 4 * CAPACITY * 4,
        TILE * 4 * 4,
        TILE * 4 * 512 * 4,
        TILE * 4 * 512 * 4,
    ]
    .map(|n| n.max(256)))
}

pub fn routed_planes(rows: usize) -> Result<[usize; 14]> {
    ensure!((1..=4096).contains(&rows), "Step-5 routed allocation rows");
    const E: usize = 352;
    const K: usize = 8;
    const I: usize = 1536;
    Ok([
        (H, 4),
        (H, 2),
        (E, 4),
        (E, 4),
        (K, 4),
        (K, 4),
        (2 * I, 2),
        (I, 2),
        (H, 2),
        (H, 2),
        (H, 4),
        (H, 4),
        (H, 4),
        (H, 4),
    ]
    .map(|(width, element)| (rows * width * element).max(256)))
}

pub fn dense_planes(rows: usize) -> Result<[usize; 6]> {
    ensure!((1..=4096).contains(&rows), "Step-5 dense allocation rows");
    Ok([(H, 4), (H, 2), (2 * 13824, 2), (13824, 2), (H, 2), (H, 4)]
        .map(|(width, elem)| (rows * width * elem).max(256)))
}

pub fn head_planes(rows: usize) -> Result<[usize; 4]> {
    ensure!((1..=4096).contains(&rows), "Step-5 head allocation rows");
    Ok([rows * H * 2, rows * H * 2, rows * 128896 * 4, 4 << 20])
}

/// Global RoPE positions, rebased KV positions, request ids, a bounded page
/// table and publication scratch. All planes have the native 256-byte floor.
pub fn mtp_metadata_planes(rows: usize) -> Result<[usize; 5]> {
    ensure!((1..=4096).contains(&rows), "Step-5 predictor metadata rows");
    let pages = (512 + 63 + rows).div_ceil(64);
    Ok([rows * 8, rows * 8, rows * 4, pages * 4, 8].map(|n| n.max(256)))
}

/// Conservative private predictor lane recipe for offline fit forecasts.
/// Whole SWA/dense/head workspaces are retained per stage; no aliasing or
/// recomputation saving is credited before a native owner proves it.
pub fn mtp_forecast_workspace(rows: usize, depth: usize) -> Result<u64> {
    ensure!(depth <= 3, "Step-5 predictor depth must be 0..=3");
    let mut stage = 0u64;
    for bytes in swa_planes(rows, 64)?.into_iter().chain(dense_planes(rows)?)
        .chain(head_planes(rows)?) {
        stage = stage.checked_add(bytes as u64).context("Step-5 MTP workspace overflow")?;
    }
    // Independent normalized inputs, concat, EH output and F32 residual;
    // include source/position/length scratch rather than assuming reuse.
    let metadata = mtp_metadata_planes(rows)?.into_iter().sum::<usize>() as u64;
    let input = (rows as u64).checked_mul((H * 16) as u64)
        .and_then(|n| n.checked_add(metadata.max(rows as u64 * 64)))
        .context("Step-5 MTP input scratch overflow")?;
    stage.checked_add(input).and_then(|n| n.checked_mul(depth as u64))
        .context("Step-5 MTP workspace overflow")
}

pub fn rank_admission(
    checkpoint: &Checkpoint,
    ownership: &ExecutionOwnership,
    rank: usize,
    rows: usize,
) -> Result<RankAdmission> {
    ensure!(
        rows > 0 && rows as u64 <= ownership.execution().spec().rows,
        "Step-5 native row admission"
    );
    let cfg = Config::parse(checkpoint)?;
    let contract = CoordinatorContract::from_checkpoint(checkpoint)?;
    let layout = RankWeightLayout::for_execution(checkpoint, rank, ownership)?;
    let immutable = layout
        .operands
        .iter()
        .filter(|w| w.name != "model.embed_tokens.weight")
        .try_fold(contract.rope_tables.allocation_bytes, |n, w| {
            n.checked_add(w.allocation_bytes)
        })
        .context("Step-5 native rank immutable overflow")?;
    let mut workspace = 0u64;
    let mut add = |planes: &[usize]| -> Result<()> {
        for &bytes in planes {
            workspace = workspace
                .checked_add(bytes as u64)
                .context("Step-5 native rank workspace overflow")?;
        }
        Ok(())
    };
    for layer in 0..cfg.layers {
        let local = ownership.layer(layer, rank)?;
        if local.query_heads == 0 {
            continue;
        }
        if cfg.full[layer] {
            add(&csa_planes(rows)?)?;
        } else {
            add(&swa_planes(rows, local.query_heads)?)?;
        }
        if cfg.moe[layer] {
            add(&routed_planes(rows)?)?;
        } else {
            add(&dense_planes(rows)?)?;
        }
    }
    if rank == ownership.execution().spec().head_gpu as usize {
        add(&head_planes(rows)?)?;
    }
    Ok(RankAdmission {
        immutable,
        workspace,
    })
}
