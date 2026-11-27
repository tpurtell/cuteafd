//! Bundled recurrent drafter facts; the shared solver decides pool and placement.
use crate::{formats::block_dspark::recurrent::{loading::{RecurrentSource,RecurrentWeightLayout},
    runtime::RecurrentRuntimeLayout,RecurrentBlockConfig},families::qwen4::Qwen4KvCache,
    plan::checkpoint::CheckpointView};
use anyhow::{ensure,Context,Result};

/// Target and drafter banks share this single combined-budget slot count.
pub fn prefix_mark_slots(sequences:u64,entries:u64,target_bytes:u64,draft_bytes:u64,
    budget:u64,host:bool)->Result<usize>{
    let bytes=target_bytes.checked_add(draft_bytes).context("Inkling combined mark bytes overflow")?;
    Ok(usize::try_from(cuteafd_core::prefix::mark_slots_with_host(sequences,entries,bytes,budget,host))?)
}

pub const CONTEXT_ROWS: usize = 2048;

pub const FOLD_TILE: usize = 32;

pub fn fold_programs(hidden: usize, rank: usize, stored: usize) -> [String; 2] {
    [format!("inkling2_padded_transpose_bf16_r{hidden}_c{rank}_p{hidden}"),
        format!("inkling2_padded_transpose_f32_r{FOLD_TILE}_c{stored}_p{FOLD_TILE}")]
}

#[derive(Debug,Clone,Copy)]
pub struct Selection {pub kv_cache:Qwen4KvCache,pub sms:usize}

#[derive(Debug,Clone)]
pub struct DraftLayout {
    pub config:RecurrentBlockConfig,
    pub weights:Vec<RecurrentWeightLayout>,
    pub runtime:Vec<RecurrentRuntimeLayout>,
    /// Target-side tap and exact FIR verification storage, independent of pool.
    pub target:Vec<(String,usize)>,
    pub peer: Vec<(String,usize)>,
    pub exchange: Vec<(String,usize)>,
}
impl DraftLayout {
    pub fn new(view:&CheckpointView,ranks:usize,slots:usize,sequences:usize,context:usize,pool:usize,
        prefill:usize,waves:usize,sms:usize,format:Qwen4KvCache,manifest:Option<&serde_json::Value>)->Result<Self>{
        let (config, weights, _) = Self::select_weights(view, ranks, 0)?;
        Self::from_weights(config, weights, slots, sequences, context, pool, prefill, waves, sms, format, manifest)
    }

    pub fn select_weights(view: &CheckpointView, ranks: usize, lead: usize) -> Result<(RecurrentBlockConfig, Vec<RecurrentWeightLayout>, crate::shared::dense_read::MountLoad)> {
        ensure!(matches!(ranks, 1 | 2), "recurrent draft rank geometry");
        let source = RecurrentSource::open(view)?;
        let weights = (0..ranks).map(|rank| source.rank(rank, ranks, lead)).collect::<Result<Vec<_>>>()?;
        let mount = source.mount_load(ranks)?;
        Ok((source.config, weights, mount))
    }

    pub fn from_weights(config: RecurrentBlockConfig, weights: Vec<RecurrentWeightLayout>, slots: usize,
        sequences: usize, context: usize, pool: usize, prefill: usize, waves: usize, sms: usize,
        format: Qwen4KvCache, manifest: Option<&serde_json::Value>) -> Result<Self> {
        Self::from_weights_for_decode_rows(config, weights, slots, sequences, context, pool, prefill, waves, 64, sms, format, manifest)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn from_weights_for_decode_rows(config: RecurrentBlockConfig, weights: Vec<RecurrentWeightLayout>, slots: usize,
        sequences: usize, context: usize, pool: usize, prefill: usize, waves: usize, decode_rows: usize, sms: usize,
        format: Qwen4KvCache, manifest: Option<&serde_json::Value>) -> Result<Self> {
        let ranks = weights.len();
        ensure!(matches!(ranks, 1 | 2) && slots > 0 && sequences > 0 && prefill > 0 && waves > 0
            && (1..=128).contains(&decode_rows),
            "recurrent draft fact geometry");
        let lead = weights[0].lead;
        ensure!(lead < ranks && weights.iter().enumerate().all(|(rank, layout)|
            layout.rank == rank && layout.ranks == ranks && layout.lead == lead), "Inkling draft weight rank ownership differs");
        let mut runtime=(0..ranks).map(|rank|RecurrentRuntimeLayout::new(&config,slots,sequences,CONTEXT_ROWS,context,pool,
            sms,format,rank,ranks,lead)).collect::<Result<Vec<_>>>()?;
        for (weight,work) in weights.iter().zip(&mut runtime){
            let rows=work.context_rows.max(work.sequences*config.transformer.block);
            let mut projection:Option<crate::formats::trellis_core::ProjectionScratchLayout>=None;
            for core in weight.core.values(){
                let need=crate::formats::trellis_core::ProjectionScratchLayout::for_weight(core,rows)?;
                if let Some(projection)=&mut projection{projection.merge(&need)?;}else{projection=Some(need);}
            }
            if let Some(projection)=projection{
                weight.trellis_programs(manifest.context("Inkling trellis drafter requires PROGRAMS.json")?, rows)?;
                work.buffers.push(("projection.trellis".into(),projection.packed_bytes()?));
            }
            if weight.core.values().any(|core| core.side == crate::formats::dense_trellis::TrellisSide::Input) {
                let bytes = work.sequences.checked_mul(config.transformer.block)
                    .and_then(|rows| rows.checked_mul(config.transformer.hidden))
                    .and_then(|elements| elements.checked_mul(4)).context("draft F32 partial extent overflow")?;
                work.buffers.push(("step.partial_f32".into(), crate::serving_capacity::WeightArenaLayout::allocation_bytes(bytes)));
            }
            if !weight.fp8.is_empty() {
                let manifest = manifest.context("Inkling FP8 drafter requires PROGRAMS.json")?;
                work.buffers.push(("projection.fp8".into(), weight.fp8_scratch(manifest, rows)?));
            }
        }
        let (target, verify) = target_owners(config.transformer.hidden, config.taps.len(), slots,
            prefill, waves, decode_rows, ranks)?;
        let (peer,exchange)=if ranks==2 {
            let bytes=runtime[0].buffers.iter().find(|(name,_)|name=="step.partial").context("draft partial owner missing")?.1;
            let slot = runtime[0].buffers.iter().find(|(name,_)|name=="step.partial_f32").map_or(bytes, |(_, bytes)| *bytes);
            let storage=crate::serving_capacity::PeerStorageLayout::new(1,slot,crate::serving_capacity::peer_wait_diagnostics())?;
            (verify,[("reduced",bytes),("peer.recv.0",storage.receive_bytes),("peer.control",storage.control_bytes),
                ("peer.progress",storage.progress_bytes),("peer.abort",256)].into_iter()
                .filter(|(_,bytes)|*bytes>0).map(|(name,bytes)|(name.into(),bytes)).collect())
        } else {(Vec::new(),Vec::new())};
        Ok(Self{config,weights,runtime,target,peer,exchange})
    }
}

/// Shared target tap staging and FIR verification owners, independent of the drafter body.
pub fn target_owners(h: usize, taps: usize, slots: usize, prefill: usize, waves: usize,
    decode_rows: usize, ranks: usize) -> Result<(Vec<(String, usize)>, Vec<(String, usize)>)> {
    let extent=|values:&[usize]|values.iter().try_fold(1usize,|n,value|n.checked_mul(*value)).context("draft target owner overflow");
    let mut target=Vec::new();
    for wave in 0..waves{
        target.push((format!("unit.{wave}.taps"),extent(&[prefill,taps,h,2])?));
        target.push((format!("unit.{wave}.narrow"),extent(&[prefill,h,2])?));
    }
    let replay_rows = super::execution::replay_capacity(decode_rows);
    target.push(("decode.taps".into(),extent(&[decode_rows,taps,h,2])?));
    target.push(("decode.narrow".into(),extent(&[decode_rows,h,2])?));
    let conv=super::conv::ConvLayout::new(slots,48,h,1024/ranks)?;
    let verify = vec![("verify.checkpoint".into(),conv.bytes()?),
        ("verify.replay".into(),extent(&[48,replay_rows,conv.width,4])?), ("verify.table".into(),replay_rows*12)];
    target.extend(verify.iter().cloned());
    Ok((target, verify))
}
