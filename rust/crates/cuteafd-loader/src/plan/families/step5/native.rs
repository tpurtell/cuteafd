//! Exact native allocation shapes shared by offline admission and device owners.
use super::{
    coordinator::{CoordinatorContract, RankWeightLayout},
    ownership::ExecutionOwnership,
    Config,
};
use crate::{families::qwen4::Qwen4KvCache, plan::checkpoint::Checkpoint};
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

pub fn validate_span_rows(rows: usize) -> Result<()> {
    let limit = crate::serving_capacity::prefill_waves::PrefillLimits::default().max_span_rows;
    ensure!(rows > 0 && rows as u64 <= limit, "Step-5 allocation span must be 1..={limit} rows");
    Ok(())
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

pub fn routed_planes(rows: usize) -> Result<[usize; 15]> {
    validate_span_rows(rows)?;
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
        (H + H / 32, 1),
    ]
    .map(|(width, element)| (rows * width * element).max(256)))
}

pub fn dense_planes(rows: usize) -> Result<[usize; 6]> {
    validate_span_rows(rows)?;
    Ok([(H, 4), (H, 2), (2 * 13824, 2), (13824, 2), (H, 2), (H, 4)]
        .map(|(width, elem)| (rows * width * elem).max(256)))
}

/// Planes live from an FFN's begin to its finish (or to the next attention),
/// in FFN-region order: OUTPUT, INPUT, SHARED, then two ROUTED reply planes.
/// Every FFN kind keeps them at these offsets, so one kind's scratch never
/// overwrites another kind's pending rows, and attention never touches the
/// FFN region. Head-split (SWA) and whole-owned (CSA) layers receive the
/// peer's replicated reply in different planes: two replications with no head
/// exchange between them never share storage (`validate_routed_replications`).
const ROUTED_LIVE: [usize; 4] = [13, 0, 8, 9];
const DENSE_LIVE: [usize; 2] = [5, 0];

fn region(sizes: &[usize], order: &[usize], start: usize) -> Result<(Vec<(usize, usize)>, usize)> {
    let mut layout = vec![(0, 0); sizes.len()];
    let mut next = start;
    for &plane in order {
        layout[plane] = (next, sizes[plane]);
        next = next.checked_add(sizes[plane]).and_then(|n| n.checked_add(15))
            .context("Step-5 FFN region overflow")? / 16 * 16;
    }
    Ok((layout, next))
}

/// Live planes hold every in-flight unit's rows (`span` rows, each unit at its
/// own row offset); FFN scratch is hook-local and holds one unit.
fn routed_sizes(unit: usize, span: usize) -> Result<[usize; 15]> {
    ensure!((1..=4096).contains(&unit) && unit <= span, "Step-5 FFN unit rows exceed kernel capacity or span");
    let (local, live) = (routed_planes(unit)?, routed_planes(span)?);
    Ok(std::array::from_fn(|plane| if ROUTED_LIVE.contains(&plane) { live[plane] } else { local[plane] }))
}

fn dense_sizes(unit: usize, span: usize) -> Result<[usize; 6]> {
    ensure!((1..=4096).contains(&unit) && unit <= span, "Step-5 FFN unit rows exceed kernel capacity or span");
    let (local, live) = (dense_planes(unit)?, dense_planes(span)?);
    Ok(std::array::from_fn(|plane| if DENSE_LIVE.contains(&plane) { live[plane] } else { local[plane] }))
}

/// End of the live prefix: OUTPUT, INPUT, SHARED and both reply planes.
fn live_prefix(unit: usize, span: usize) -> Result<usize> {
    let routed = routed_sizes(unit, span)?;
    let (_, end) = region(&routed, &ROUTED_LIVE, 0)?;
    end.checked_add(routed[9]).and_then(|n| n.checked_add(15)).map(|n| n / 16 * 16)
        .context("Step-5 FFN region overflow")
}

/// (offset, bytes) per routed plane, in the routed owner's plane-index order,
/// relative to the rank's FFN region. `full` (CSA) layers use the second reply plane.
pub fn routed_layout(unit: usize, span: usize, full: bool) -> Result<[(usize, usize); 15]> {
    let sizes = routed_sizes(unit, span)?;
    let (mut layout, swa_end) = region(&sizes, &ROUTED_LIVE, 0)?;
    if full { layout[9].0 = swa_end; }
    let local = (0..15).filter(|plane| !ROUTED_LIVE.contains(plane)).collect::<Vec<_>>();
    let (tail, _) = region(&sizes, &local, live_prefix(unit, span)?)?;
    for &plane in &local { layout[plane] = tail[plane]; }
    Ok(layout.try_into().expect("routed plane count"))
}

/// Dense OUTPUT and INPUT share the routed offsets; its scratch follows the
/// whole live prefix, including the routed-only SHARED and reply planes.
pub fn dense_layout(unit: usize, span: usize) -> Result<[(usize, usize); 6]> {
    let sizes = dense_sizes(unit, span)?;
    let (mut layout, _) = region(&sizes, &DENSE_LIVE, 0)?;
    let local = [1, 2, 3, 4];
    let (tail, _) = region(&sizes, &local, live_prefix(unit, span)?)?;
    for plane in local { layout[plane] = tail[plane]; }
    Ok(layout.try_into().expect("dense plane count"))
}

/// Whether a routed (or dense) plane holds rows of every in-flight unit.
pub fn routed_live(plane: usize) -> bool { ROUTED_LIVE.contains(&plane) }
pub fn dense_live(plane: usize) -> bool { DENSE_LIVE.contains(&plane) }

/// One FFN region per rank, after that rank's largest attention region.
pub fn ffn_region_bytes(unit: usize, span: usize) -> Result<u64> {
    let end = |layout: &[(usize, usize)]| layout.iter().map(|&(offset, bytes)| offset + bytes)
        .max().unwrap_or(0);
    let bytes = end(&routed_layout(unit, span, false)?).max(end(&routed_layout(unit, span, true)?))
        .max(end(&dense_layout(unit, span)?));
    Ok(u64::try_from(bytes.div_ceil(16) * 16)?)
}

pub fn attention_bytes(rows: usize, full: bool, heads: usize) -> Result<u64> {
    if full { plane_bytes(&csa_planes(rows)?) } else { plane_bytes(&swa_planes(rows, heads)?) }
}

/// The FFN region's offset in this rank's arena: its largest attention region.
pub fn rank_attention_bytes(cfg: &Config, ownership: &ExecutionOwnership, rank: usize, rows: usize)
    -> Result<u64> {
    let mut largest = 0;
    for layer in 0..cfg.layers {
        let local = ownership.layer(layer, rank)?;
        if local.query_heads == 0 { continue; }
        largest = largest.max(attention_bytes(rows, cfg.full[layer], local.query_heads)?);
    }
    Ok(largest)
}

/// Head logit rows: the decode/verify width, or every chunk row only when
/// full-prefill scoring is admitted. Prefill never needs more than its last row.
pub fn head_rows(rows: usize, full_prefill_logits: bool) -> usize {
    if full_prefill_logits { rows } else { rows.min(64) }
}

pub fn head_planes(rows: usize) -> Result<[usize; 4]> {
    validate_span_rows(rows)?;
    Ok([rows * H * 2, rows * H * 2, rows * 128896 * 4, 4 << 20])
}

/// Global RoPE positions, rebased KV positions, request ids, a bounded page
/// table and publication scratch. All planes have the native 256-byte floor.
pub fn mtp_metadata_planes(rows: usize) -> Result<[usize; 5]> {
    ensure!((1..=4096).contains(&rows), "Step-5 predictor metadata rows");
    let pages = (512 + 63 + rows).div_ceil(64);
    Ok([rows * 8, rows * 8, rows * 4, pages * 4, 8].map(|n| n.max(256)))
}

/// One sequential predictor scratch recipe, shared with the target arena.
/// Only greedy token/status words survive across stages within a draft chain.
pub fn mtp_forecast_workspace(rows: usize, depth: usize, kv_format: Qwen4KvCache) -> Result<u64> {
    ensure!(depth <= 3, "Step-5 predictor depth must be 0..=3");
    if depth == 0 { return Ok(0); }
    mtp_scratch_core(rows, kv_format)?.checked_add((depth * 256) as u64)
        .context("Step-5 MTP workspace overflow")
}

pub fn mtp_input_planes(rows: usize) -> Result<[usize; 6]> {
    ensure!((1..=4096).contains(&rows), "Step-5 predictor input rows");
    Ok([2, 2, 2, 4, 2, 4].map(|element| rows * H * element))
}

pub fn mtp_input_offset(rows: usize) -> Result<u64> {
    layer_workspace(rows, rows, false, 64)
}

pub fn mtp_metadata_offset(rows: usize) -> Result<u64> {
    mtp_input_offset(rows)?.checked_add(plane_bytes(&mtp_input_planes(rows)?)?)
        .context("Step-5 predictor input offset overflow")
}

pub fn mtp_compact_offset(rows: usize) -> Result<u64> {
    mtp_metadata_offset(rows)?.checked_add(plane_bytes(&mtp_metadata_planes(rows)?[..4])?)
        .context("Step-5 predictor metadata offset overflow")
}

pub fn mtp_compact_planes(rows: usize, kv_format: Qwen4KvCache) -> Result<[usize; 2]> {
    ensure!((1..=4096).contains(&rows), "Step-5 predictor compact rows");
    Ok([if kv_format == Qwen4KvCache::Fp8 {
        rows * kv_format.plane_record_bytes(4, 192)
    } else { 0 }; 2])
}

pub fn mtp_scratch_core(rows: usize, kv_format: Qwen4KvCache) -> Result<u64> {
    Ok(mtp_compact_offset(rows)?.checked_add(plane_bytes(&mtp_compact_planes(rows, kv_format)?)?)
        .context("Step-5 predictor compact staging overflow")?
        .max(plane_bytes(&head_planes(head_rows(rows, false))?)?))
}

/// `unit` rows run through attention and FFN scratch per hook; `span` rows
/// (unit x waves in flight) are held by the live FFN planes.
pub fn rank_admission(
    checkpoint: &Checkpoint,
    ownership: &ExecutionOwnership,
    rank: usize,
    unit: usize,
    span: usize,
    logits: usize,
) -> Result<RankAdmission> {
    ensure!(
        unit > 0 && unit <= span && unit as u64 <= ownership.execution().spec().rows
            && (1..=span).contains(&logits),
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
    let immutable = layout.core.values().try_fold(immutable, |n, core| n.checked_add(core.bytes as u64))
        .context("Step-5 native core immutable overflow")?;
    let owns = (0..cfg.layers).any(|layer| ownership.layer(layer, rank).is_ok_and(|l| l.query_heads > 0));
    let mut workspace = if owns {
        rank_attention_bytes(&cfg, ownership, rank, unit)?.checked_add(ffn_region_bytes(unit, span)?)
            .and_then(|n| n.checked_add(native_scratch_bytes(&layout, unit).ok()?))
            .context("Step-5 rank workspace overflow")?
    } else { 0 };
    if rank == ownership.execution().spec().head_gpu as usize {
        // The head reuses the arena after every layer drained; a native head adds its own scratch.
        let head_scratch = layout.core.get("lm_head.weight").map(|core| {
            let [n, k] = core.shape;
            crate::formats::trellis_core::ProjectionScratchLayout::new(k, n, logits)?.bytes()
        }).transpose()?.unwrap_or(0) as u64;
        workspace = workspace.max(plane_bytes(&head_planes(logits)?)?.checked_add(head_scratch)
            .context("Step-5 head workspace overflow")?);
    }
    Ok(RankAdmission {
        immutable,
        workspace,
    })
}

/// The one serialized native projection scratch a rank's layers share, placed
/// after its FFN region; zero when the rank owns no native operand.
pub fn native_scratch_bytes(layout: &RankWeightLayout, unit: usize) -> Result<u64> {
    let mut merged: Option<crate::formats::trellis_core::ProjectionScratchLayout> = None;
    for core in layout.core.iter().filter(|(name, _)| name.as_str() != "lm_head.weight").map(|(_, core)| core) {
        let [n, k] = core.shape;
        let next = crate::formats::trellis_core::ProjectionScratchLayout::new(k, n, unit)?;
        match &mut merged { Some(layout) => layout.merge(&next)?, None => merged = Some(next) }
    }
    Ok(merged.map(|layout| layout.bytes()).transpose()?.map_or(0, |bytes| bytes as u64))
}

/// Each predictor stage's native owners (attention, dense FFN, eh_proj and its
/// vocabulary head) hold private projection scratch; bounded at `rows`.
pub fn mtp_native_scratch_bytes(checkpoint: &Checkpoint, depth: usize, rows: usize) -> Result<u64> {
    use crate::formats::trellis_core::ProjectionScratchLayout;
    let mut total = 0u64;
    for stage in 0..depth {
        let layout = RankWeightLayout::for_mtp_stage(checkpoint, stage)?;
        let mut owners = std::collections::BTreeMap::<&str, ProjectionScratchLayout>::new();
        for (name, core) in &layout.core {
            let owner = if name.contains(".self_attn.") { "attention" } else if name.contains(".mlp.") { "dense" }
                else if name.ends_with(".eh_proj.weight") { "eh" } else { "head" };
            let [n, k] = core.shape;
            let next = ProjectionScratchLayout::new(k, n, rows.min(4096))?;
            match owners.get_mut(owner) { Some(layout) => layout.merge(&next)?, None => { owners.insert(owner, next); } }
        }
        for layout in owners.values() {
            total = total.checked_add(layout.bytes()? as u64).context("Step-5 MTP native scratch overflow")?;
        }
    }
    Ok(total)
}

pub fn plane_bytes(planes: &[usize]) -> Result<u64> {
    planes.iter().try_fold(0u64, |sum, &bytes| {
        let aligned = bytes.checked_add(15).context("Step-5 scratch alignment overflow")? / 16 * 16;
        sum.checked_add(aligned as u64).context("Step-5 scratch size overflow")
    })
}

/// One layer kind's attention region plus the shared FFN region.
pub fn layer_workspace(unit: usize, span: usize, full: bool, heads: usize) -> Result<u64> {
    attention_bytes(unit, full, heads)?.checked_add(ffn_region_bytes(unit, span)?)
        .context("Step-5 layer scratch overflow")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn end(layout: &[(usize, usize)]) -> usize {
        layout.iter().map(|&(offset, bytes)| offset + bytes).max().unwrap()
    }

    #[test]
    fn ffn_region_keeps_live_planes_at_common_offsets_for_every_kind() -> Result<()> {
        for rows in [1, 16, 64, 256, 512, 1024, 2048, 4096] {
            let routed = routed_layout(rows, rows, false)?;
            let csa = routed_layout(rows, rows, true)?;
            let dense = dense_layout(rows, rows)?;
            // Head-split and whole-owned replies never share storage; every
            // other plane is identical between the two routed kinds.
            assert!(routed[9].0 + routed[9].1 <= csa[9].0);
            assert!((0..15).filter(|&p| p != 9).all(|p| routed[p] == csa[p]));
            let region = usize::try_from(ffn_region_bytes(rows, rows)?)?;
            // OUTPUT first, then INPUT, in both kinds.
            assert_eq!((routed[13].0, dense[5].0), (0, 0));
            assert_eq!(routed[0].0, dense[0].0);
            assert_eq!((routed[13].1, routed[0].1), (dense[5].1, dense[0].1));
            // Every plane of every kind is 16-byte aligned, disjoint inside
            // its kind, and inside the one region.
            for layout in [&routed[..], &csa[..], &dense[..]] {
                assert!(end(layout) <= region);
                for (i, a) in layout.iter().enumerate() {
                    assert_eq!(a.0 % 16, 0);
                    assert!(layout[i + 1..].iter().all(|b| a.0 + a.1 <= b.0 || b.0 + b.1 <= a.0));
                }
            }
            // Dense scratch never touches a pending routed wave's live planes
            // (OUTPUT, INPUT, SHARED, ROUTED), nor routed scratch dense's.
            let live = ROUTED_LIVE.map(|plane| routed[plane]).into_iter().chain([csa[9]]).collect::<Vec<_>>();
            for plane in [1, 2, 3, 4] {
                assert!(live.iter().all(|b| dense[plane].0 >= b.0 + b.1));
            }
            for kind in [&routed, &csa] {
                for plane in (0..15).filter(|plane| !ROUTED_LIVE.contains(plane)) {
                    assert!(live.iter().all(|b| kind[plane].0 >= b.0 + b.1));
                }
            }
            assert_eq!(region % 16, 0);
            assert!(region < usize::try_from(plane_bytes(&routed_planes(rows)?)? + plane_bytes(&dense_planes(rows)?)?)?);
        }
        // Units: live planes span every in-flight unit's rows, scratch holds one unit.
        for (unit, span) in [(256, 512), (512, 1024), (2048, 4096), (4096, 8192), (2752, 8256), (2048, 8192)] {
            let routed = routed_layout(unit, span, true)?;
            let dense = dense_layout(unit, span)?;
            for plane in 0..15 {
                let expected = if routed_live(plane) { routed_planes(span)?[plane] } else { routed_planes(unit)?[plane] };
                assert_eq!(routed[plane].1, expected);
            }
            for plane in 0..6 {
                let expected = if dense_live(plane) { dense_planes(span)?[plane] } else { dense_planes(unit)?[plane] };
                assert_eq!(dense[plane].1, expected);
            }
            if span <= 4096 { assert!(ffn_region_bytes(unit, span)? < ffn_region_bytes(span, span)?); }
        }
        let limit = crate::serving_capacity::prefill_waves::PrefillLimits::default().max_span_rows as usize;
        for span in [0, limit + 1, usize::MAX] {
            assert!(routed_planes(span).is_err());
            assert!(dense_planes(span).is_err());
            assert!(head_planes(span).is_err());
            assert!(routed_layout(512, span, false).is_err());
            assert!(dense_layout(512, span).is_err());
        }
        assert!(routed_planes(limit).is_ok());
        assert!(dense_planes(limit).is_ok());
        assert!(head_planes(limit).is_ok());
        assert!(routed_layout(512, 256, false).is_err());
        Ok(())
    }

    #[test]
    fn layer_scratch_is_attention_plus_one_shared_ffn_region() -> Result<()> {
        for rows in [1, 256, 2048, 4096] {
            let region = ffn_region_bytes(rows, rows)?;
            for (full, heads) in [(false, 32), (false, 64), (true, 64)] {
                assert_eq!(layer_workspace(rows, rows, full, heads)?, attention_bytes(rows, full, heads)? + region);
            }
            assert!(layer_workspace(rows, rows, true, 64)? >= layer_workspace(rows, rows, false, 64)?);
            assert!(layer_workspace(rows, rows, true, 64)? > plane_bytes(&head_planes(head_rows(rows, false))?)?);
        }
        assert_eq!(head_rows(4096, false), 64);
        assert_eq!(head_rows(16, false), 16);
        assert_eq!(head_rows(4096, true), 4096);
        Ok(())
    }
}
