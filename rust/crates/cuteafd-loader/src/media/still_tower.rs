//! Exact header-native arenas and phase scratch for independent still towers.
//! This inventory does not qualify native execution or choose a placement.
use crate::{plan::Checkpoint, serving_capacity::WeightArenaLayout};
use anyhow::{ensure, Context, Result};

/// Ledger identity for a packed tower backing, independent of its family or owner.
pub const fn vision_weight_arena_label() -> &'static str { "vision packed weight arena" }

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StillTowerKind { Kimi, MiniMax, Inkling2, Step5, Hy4Flash, Hy4 }

#[derive(Clone, Debug)]
pub struct StillTowerInventory {
    pub kind: StillTowerKind,
    pub arena: WeightArenaLayout,
    pub scratch: u64,
    pub max_output_rows: usize,
}

impl StillTowerInventory {
    pub fn from_checkpoint(checkpoint: &Checkpoint, max_output_rows: usize) -> Result<Self> {
        if matches!(checkpoint.model_type(), Some("inkling2" | "inkling2_loop" | "step5_vl")) {
            use super::perception_vision::{PerceptionGeometry, PerceptionKind};
            let geometry = PerceptionGeometry::from_config(&checkpoint.config).map_err(anyhow::Error::msg)?;
            let physical = geometry.resident_weights(checkpoint.tensors.iter().map(|t| &t.meta))
                .map_err(anyhow::Error::msg)?;
            let kind = match geometry.kind { PerceptionKind::Inkling2 => StillTowerKind::Inkling2,
                PerceptionKind::Step5 => StillTowerKind::Step5 };
            let regions = geometry.tensor_shapes().into_iter().map(|(name, shape)|
                (name, shape.into_iter().product::<usize>() * 2));
            let arena = WeightArenaLayout::new(if kind == StillTowerKind::Inkling2 { "inkling2/vision" }
                else { "step5/vision" }, regions.chain([("rotary.inv_freq".into(), geometry.inv_freq().len()*4)]))?;
            ensure!(arena.bytes() as u64 == physical, "PE packed arena differs from native inventory");
            let scratch = geometry.scratch_bytes(u64::try_from(max_output_rows)?).map_err(anyhow::Error::msg)?;
            return Ok(Self { kind, arena, scratch, max_output_rows });
        }
        let tensors=checkpoint.tensors.iter().map(|tensor| &tensor.meta);
        let (kind,id,regions,constant_bytes,limit)=match checkpoint.model_type() {
            Some("kimi_k3"|"kimi_k31")=>{
                let contract=crate::families::kimi_k3::vision::VisionContract::from_config(&checkpoint.config)?;
                let packed=contract.pack(tensors)?;
                let regions=packed.tensors.into_iter().map(|tensor| {
                    Ok((tensor.metadata.name,usize::try_from(tensor.metadata.byte_length)?))
                }).collect::<Result<Vec<_>>>()?;
                (StillTowerKind::Kimi,"kimi/vision",regions,128,
                    crate::families::kimi_k3::vision::MAX_IMAGE_TOKENS as usize)
            },
            Some("minimax_m3_vl")=>{
                let contract=crate::families::minimax_m3::vision::VisionContract::from_config(&checkpoint.config)?;
                let packed=contract.pack(tensors)?;
                let regions=packed.tensors.into_iter().map(|tensor| {
                    Ok((tensor.metadata.name,usize::try_from(tensor.resident_bytes)?))
                }).collect::<Result<Vec<_>>>()?;
                (StillTowerKind::MiniMax,"minimax/vision",regions,52,
                    crate::families::minimax_m3::vision::MAX_IMAGE_TOKENS)
            },
            Some("hy_v4_flash")=>{
                let contract=crate::families::hy4_flash::vision::VisionContract::from_config(&checkpoint.config)?;
                let packed=contract.pack(tensors)?;
                let regions=packed.tensors.into_iter().map(|tensor| {
                    Ok((tensor.metadata.name,usize::try_from(tensor.metadata.byte_length)?))
                }).collect::<Result<Vec<_>>>()?;
                let limit=crate::families::hy4_flash::vision::MAX_MERGED_ROWS;
                ensure!((1..=limit).contains(&max_output_rows),"still tower output rows outside 1..={limit}");
                let arena=WeightArenaLayout::new("hy4_flash/vision",regions)?;
                ensure!(arena.bytes() as u64==packed.arena_bytes,"Hy4-flash packed vision arena differs from header pack");
                let scratch=scratch_bytes(StillTowerKind::Hy4Flash,max_output_rows)?;
                return Ok(Self {kind:StillTowerKind::Hy4Flash,arena,scratch,max_output_rows});
            },
            Some("hy_v4_vl")=>{
                let contract=crate::families::hy4::vision::vision_contract(&checkpoint.config)?;
                let packed=crate::families::hy4::vision::pack(&contract,tensors)?;
                let regions=packed.tensors.into_iter().map(|tensor| {
                    Ok((tensor.metadata.name,usize::try_from(tensor.metadata.byte_length)?))
                }).collect::<Result<Vec<_>>>()?;
                let limit=crate::families::hy4::vision::MAX_MERGED_ROWS;
                ensure!((1..=limit).contains(&max_output_rows),"still tower output rows outside 1..={limit}");
                let arena=WeightArenaLayout::new("hy4/vision",regions)?;
                ensure!(arena.bytes() as u64==packed.arena_bytes,"Hy4 packed vision arena differs from header pack");
                let scratch=scratch_bytes(StillTowerKind::Hy4,max_output_rows)?;
                return Ok(Self {kind:StillTowerKind::Hy4,arena,scratch,max_output_rows});
            },
            other=>anyhow::bail!("no independent still tower for {other:?}"),
        };
        ensure!((1..=limit).contains(&max_output_rows),"still tower output rows outside 1..={limit}");
        let arena=WeightArenaLayout::new(id,regions.into_iter().chain([
            ("rotary.inv_freq".into(),constant_bytes)]))?;
        let scratch=scratch_bytes(kind,max_output_rows)?;
        Ok(Self {kind,arena,scratch,max_output_rows})
    }
}

/// Mirror the production phase liveness in scratch_generic_tower, with every
/// individually aligned request and the same peak union (not a payload sum).
pub fn scratch_bytes(kind: StillTowerKind, rows: usize) -> Result<u64> {
    ensure!(!matches!(kind, StillTowerKind::Inkling2 | StillTowerKind::Step5),
        "PE scratch requires checkpoint-native PerceptionGeometry");
    if matches!(kind, StillTowerKind::Hy4Flash | StillTowerKind::Hy4) {
        return hy4_scratch_bytes(kind, rows);
    }
    let limit=match kind {StillTowerKind::Kimi=>16698,StillTowerKind::MiniMax=>576,_=>unreachable!()};
    ensure!((1..=limit).contains(&rows),"still tower scratch capacity outside native range");
    let (hidden,attention,heads,intermediate,rotated,temporal,output,projector_hidden)=match kind {
        StillTowerKind::Kimi=>(1024u64,1536u64,12u64,4096u64,128u64,1u64,7168u64,4096u64),
        StillTowerKind::MiniMax=>(1280,1280,16,5120,78,2,6144,6144),
        _=>unreachable!(),
    };
    let units=u64::try_from(rows)?;
    let n=units.checked_mul(4).context("tower patch extent overflow")?;
    let align=|bytes:u64|->Result<u64>{Ok(bytes.checked_add(255).context("tower scratch alignment overflow")?/256*256)};
    let phase=|requests:&[u64]|->Result<u64>{requests.iter().try_fold(0u64,|sum,&bytes|
        sum.checked_add(align(bytes)?).context("tower scratch phase overflow"))};
    let persistent=phase(&[n*hidden*2,n*hidden*2,n*rotated/2*4,n*rotated/2*4,8])?;
    let mut phases=vec![
        phase(&[n*14*14*3,768*4,n*3*temporal*14*14*2])?,
        phase(&[n*hidden*2])?,
        phase(&[n*3*attention*2,n*hidden*2])?,
        phase(&[n*3*attention*2,n*attention*2,n*attention*2,n*attention*2])?,
        phase(&[n*attention*2,n*heads*4])?,
        phase(&[n*hidden*2,n*intermediate*2])?,
    ];
    if kind==StillTowerKind::MiniMax {
        phases.push(phase(&[n*6144*2,n*6144*2])?);
        // Grouping is a view of the first projector result: it cannot be
        // charged a second time, nor overlap the final projector activation.
        phases.push(phase(&[units*output*2,units*projector_hidden*2])?);
    } else {
        phases.push(phase(&[units*output*2,units*4*hidden*2,units*projector_hidden*2])?);
    }
    phases.push(phase(&[units*output*2,units*output*2])?);
    persistent.checked_add(*phases.iter().max().unwrap()).context("tower scratch total overflow")
}

/// Mirror of native `scratch_hy4_tower`: persistent residual/next/segment table,
/// then the largest phase (patch input, positions, attention, MLP, merger).
fn hy4_scratch_bytes(kind: StillTowerKind, rows: usize) -> Result<u64> {
    ensure!((1..=crate::families::hy4_flash::vision::MAX_MERGED_ROWS).contains(&rows),
        "Hy4 tower scratch capacity outside native range");
    let (h,i,o,m,heads)=match kind {
        StillTowerKind::Hy4Flash=>(1152u64,4304u64,3072u64,4608u64,16u64),
        StillTowerKind::Hy4=>(1536u64,6144u64,6144u64,6144u64,16u64),
        _=>unreachable!(),
    };
    let units=u64::try_from(rows)?;
    let n=units*4;
    let align=|bytes:u64|->Result<u64>{Ok(bytes.checked_add(255).context("tower scratch alignment overflow")?/256*256)};
    let phase=|requests:&[u64]|->Result<u64>{requests.iter().try_fold(0u64,|sum,&bytes|
        sum.checked_add(align(bytes)?).context("tower scratch phase overflow"))};
    let persistent=phase(&[n*h*2,n*h*2,9*4])?;
    let output_rows=units+units.min(8*64);
    let phases=[
        phase(&[n*16*16*3,768*4,n*768*2])?,
        phase(&[n*h*2])?,
        phase(&[n*h*2,n*h*2,n*h*2,n*h*2,n*h*2,n*heads*4])?,
        phase(&[n*h*2,n*i*2])?,
        phase(&[units*4*h*2,units*m*2,units*o*2,output_rows*o*2])?,
    ];
    persistent.checked_add(*phases.iter().max().unwrap()).context("tower scratch total overflow")
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn pe_checkpoint_inventory_matches_native_arena_and_scratch() -> Result<()> {
        use crate::media::perception_vision::PerceptionGeometry;
        use crate::plan::testing::{t, write_snapshot};
        let inkling: serde_json::Value = serde_json::from_str(include_str!("../families/inkling2/fixtures/config.json"))?;
        let step: serde_json::Value = serde_json::from_str(include_str!("../plan/families/step5/fixtures/official-nvfp4-config.json"))?;
        for (config, kind, expected) in [(inkling, StillTowerKind::Inkling2, 655_386_688),
            (step, StillTowerKind::Step5, 3_962_029_664)] {
            let geometry = PerceptionGeometry::from_config(&config).map_err(anyhow::Error::msg)?;
            let tensors = geometry.tensor_shapes().into_iter().map(|(name, shape)| t(name, "BF16", &shape)).collect::<Vec<_>>();
            let root = tempfile::tempdir()?;
            write_snapshot(root.path(), &config, &tensors, None);
            let checkpoint = Checkpoint::open(root.path())?;
            let inventory = StillTowerInventory::from_checkpoint(&checkpoint, 4096)?;
            assert_eq!(inventory.kind, kind);
            assert_eq!(inventory.arena.bytes() as u64, expected);
            assert_eq!(inventory.scratch, geometry.scratch_bytes(4096).map_err(anyhow::Error::msg)?);
            assert_eq!(inventory.arena.views().len(), tensors.len()+1);
            for (name, offset) in inventory.arena.views() {
                assert_eq!(offset.offset() % 256, 0, "{name}");
            }
            assert!(StillTowerInventory::from_checkpoint(&checkpoint, 4097).is_err());
        }
        Ok(())
    }

    #[test]
    fn hy4_scratch_is_bounded_and_monotone() {
        let full=scratch_bytes(StillTowerKind::Hy4Flash,4096).unwrap();
        assert!(full>scratch_bytes(StillTowerKind::Hy4Flash,1024).unwrap());
        // Attention phase (5 x n x 1152 BF16 + LSE) dominates at full capacity.
        assert_eq!(full,(2*16384*1152*2+256)+(5*16384*1152*2+16384*16*4));
        assert!(scratch_bytes(StillTowerKind::Hy4Flash,4097).is_err());
        assert!(scratch_bytes(StillTowerKind::Hy4Flash,0).is_err());

        let hy4_full=scratch_bytes(StillTowerKind::Hy4,4096).unwrap();
        assert!(hy4_full>full);
        assert!(hy4_full>scratch_bytes(StillTowerKind::Hy4,1024).unwrap());
        // Attention phase (5 x n x 1536 BF16 + LSE) dominates at full capacity.
        assert_eq!(hy4_full,(2*16384*1536*2+256)+(5*16384*1536*2+16384*16*4));
        assert!(scratch_bytes(StillTowerKind::Hy4,4097).is_err());
        assert!(scratch_bytes(StillTowerKind::Hy4,0).is_err());
    }

    #[test]
    fn phase_scratch_matches_native_capacity_receipts() {
        assert_eq!(scratch_bytes(StillTowerKind::Kimi,16698).unwrap(),1_538_887_936);
        assert_eq!(scratch_bytes(StillTowerKind::Kimi,4096).unwrap(),377_487_616);
        assert_eq!(scratch_bytes(StillTowerKind::MiniMax,4).unwrap(),480_512);
        assert_eq!(scratch_bytes(StillTowerKind::MiniMax,144).unwrap(),17_284_864);
        for kind in [StillTowerKind::Kimi,StillTowerKind::MiniMax] {
            assert!(scratch_bytes(kind,0).is_err());
            assert!(scratch_bytes(kind,16699).is_err());
        }
    }
}
