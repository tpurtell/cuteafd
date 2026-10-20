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
    for layer in 0..cfg.layers {
        let local = ownership.layer(layer, rank)?;
        if local.query_heads == 0 { continue; }
        workspace = workspace.max(layer_workspace(rows, cfg.full[layer], local.query_heads, cfg.moe[layer])?);
    }
    if rank == ownership.execution().spec().head_gpu as usize {
        workspace = workspace.max(plane_bytes(&head_planes(rows)?)?);
    }
    Ok(RankAdmission {
        immutable,
        workspace,
    })
}

pub fn plane_bytes(planes: &[usize]) -> Result<u64> {
    planes.iter().try_fold(0u64, |sum, &bytes| {
        let aligned = bytes.checked_add(15).context("Step-5 scratch alignment overflow")? / 16 * 16;
        sum.checked_add(aligned as u64).context("Step-5 scratch size overflow")
    })
}

pub fn layer_workspace(rows: usize, full: bool, heads: usize, routed: bool) -> Result<u64> {
    let attention = if full { plane_bytes(&csa_planes(rows)?)? }
        else { plane_bytes(&swa_planes(rows, heads)?)? };
    let ffn = if routed { plane_bytes(&routed_planes(rows)?)? }
        else { plane_bytes(&dense_planes(rows)?)? };
    attention.checked_add(ffn).context("Step-5 layer scratch overflow")
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn target_scratch_is_max_layer_not_layer_sum() -> Result<()> {
        let rows = 256;
        assert_eq!(layer_workspace(rows, false, 32, false)?, 62_213_376);
        assert_eq!(layer_workspace(rows, false, 64, false)?, 73_305_344);
        assert_eq!(layer_workspace(rows, false, 64, true)?, 69_848_320);
        let largest = layer_workspace(rows, true, 64, true)?;
        assert_eq!(largest, 381_211_392);
        assert!(largest > plane_bytes(&head_planes(rows)?)?);
        for rows in [1, 16, 64, 256, 4096] {
            for heads in [32, 64] {
                assert!(layer_workspace(rows, true, 64, true)? >= layer_workspace(rows, false, heads, true)?);
            }
        }
        Ok(())
    }
}
